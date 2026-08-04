//! The Python-facing factor object.
//!
//! `CholFactor` is `cholmod_factor` plus the two workspaces CHOLMOD keeps
//! beside it — `Common`'s `Iwork`/`Flag`/`Xwork` and `cholmod_solve2`'s `Y`.
//! Holding them on the object is not a convenience: `gmm` calls `factorize`
//! 742 times and `solve` 1486 times per GLMM fit against one symbolic
//! analysis, and rebuilding either workspace per call costs 1.06-1.18x every
//! time (see [`super::ws::Work`]). CHOLMOD keeps one `Common` across the whole
//! sequence for the same reason.
//!
//! `refactorize` takes only the values array, so the caller never rebuilds the
//! CSC pattern; `merPredD.update_xwts_and_decomp` rebuilds `M` on every
//! deviance evaluation with the same pattern and new numbers.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

use super::amd::IntWidth;
use super::numeric::{self, Factor, Params};
use super::solve::{self, SolveWork, Sys};
use super::symbolic::{self, Ordering, Sparse};
use super::ws::{self, Work};

/// `sys` as `cholmod_solve` names it. The two permutation-only systems have no
/// `scikit-sparse` equivalent — 0.5.0's `solve()` rejects `system="P"` — and
/// they are exactly what a caller doing its own triangular solve against `L`
/// needs, since `L L' = P A P'`.
fn parse_sys(s: &str) -> PyResult<Sys> {
    Ok(match s {
        "A" => Sys::A,
        "LDLt" => Sys::LDLt,
        "LD" => Sys::LD,
        "DLt" => Sys::DLt,
        "L" => Sys::L,
        "Lt" => Sys::Lt,
        "D" => Sys::D,
        "P" => Sys::P,
        "Pt" => Sys::Pt,
        other => {
            return Err(PyValueError::new_err(format!(
                "system must be one of 'A', 'LDLt', 'LD', 'DLt', 'L', 'Lt', \
                 'D', 'P', 'Pt', not {other:?}"
            )));
        }
    })
}

fn parse_ordering(s: &str) -> PyResult<Ordering> {
    Ok(match s {
        "amd" => Ordering::Amd,
        "natural" => Ordering::Natural,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'amd' or 'natural', not {other:?}"
            )));
        }
    })
}

/// A numeric Cholesky factorization, reusable for both new values and repeated
/// solves.
#[pyclass(module = "hea._rs")]
pub struct CholFactor {
    /// The input triangle. Kept so `refactorize` can replace `x` alone, which
    /// is what `cholmod_factorize (A, L, Common)` is handed each time.
    a: Sparse,
    l: Factor,
    params: Params,
    work: Work,
    ywork: SolveWork,
    /// `Common->rowfacfl` from the last factorization.
    fl: f64,
}

#[pymethods]
impl CholFactor {
    /// `cholmod_analyze` followed by `cholmod_factorize_p`.
    ///
    /// `stype` selects the stored half the way `A->stype` does. `use_ll` picks
    /// `LL'` over CHOLMOD's default `LDL'`; the two differ in the last bits of
    /// `L`, so it is a factorization choice rather than a presentation one.
    #[new]
    #[pyo3(signature = (n, indptr, indices, data, stype, beta=0.0, ordering="amd",
                        use_ll=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        n: usize,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f64>,
        stype: i32,
        beta: f64,
        ordering: &str,
        use_ll: bool,
    ) -> PyResult<CholFactor> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let data = data.as_slice()?;
        let order = parse_ordering(ordering)?;
        let params = Params {
            final_ll: use_ll,
            ..Params::default()
        };

        py.allow_threads(|| -> Result<CholFactor, String> {
            let nz = ws::validate_csc(n, indptr, indices).map_err(|e| e.to_string())?;
            if nz > data.len() {
                return Err(format!(
                    "data has length {}, expected at least indptr[n] = {nz}",
                    data.len()
                ));
            }
            let a = Sparse {
                n,
                p: indptr[..n + 1].to_vec(),
                i: indices[..nz].to_vec(),
                x: data[..nz].to_vec(),
                numeric: true,
                stype,
                sorted: ws::columns_are_sorted(n, indptr, indices),
            };
            let s = symbolic::analyze_sparse(&a, order, true, IntWidth::I64)
                .map_err(|e| e.to_string())?;
            let mut l = Factor::from_symbolic(&s);
            let mut work = Work::new(n);
            let fl = numeric::factorize(&a, beta, &mut l, &params, &mut work)
                .map_err(|e| e.to_string())?;
            Ok(CholFactor {
                a,
                l,
                params,
                work,
                ywork: SolveWork::new(),
                fl,
            })
        })
        .map_err(PyValueError::new_err)
    }

    /// Refactorize with new values on the same pattern, reusing the symbolic
    /// analysis — `cholmod_factorize (A, L, Common)` on a numeric `L`.
    #[pyo3(signature = (data, beta=0.0))]
    fn refactorize(
        &mut self,
        py: Python<'_>,
        data: PyReadonlyArray1<'_, f64>,
        beta: f64,
    ) -> PyResult<()> {
        let data = data.as_slice()?;
        if data.len() < self.a.i.len() {
            return Err(PyValueError::new_err(format!(
                "data has length {}, expected at least nnz = {}",
                data.len(),
                self.a.i.len()
            )));
        }
        py.allow_threads(|| -> Result<(), String> {
            self.a.x.copy_from_slice(&data[..self.a.i.len()]);
            self.fl = numeric::factorize(&self.a, beta, &mut self.l, &self.params, &mut self.work)
                .map_err(|e| e.to_string())?;
            Ok(())
        })
        .map_err(PyValueError::new_err)
    }

    /// `cholmod_solve`. `b` is `n`-by-`nrhs` flattened in **column-major**
    /// order, which is what `cholmod_dense` is; the returned array has the same
    /// layout.
    #[pyo3(signature = (b, nrhs=1, system="A"))]
    fn solve(
        &mut self,
        py: Python<'_>,
        b: PyReadonlyArray1<'_, f64>,
        nrhs: usize,
        system: &str,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let b = b.as_slice()?;
        let sys = parse_sys(system)?;
        let n = self.l.n;
        if b.len() != n * nrhs {
            return Err(PyValueError::new_err(format!(
                "b has length {}, expected n*nrhs = {}",
                b.len(),
                n * nrhs
            )));
        }
        let mut x = vec![0.0f64; n * nrhs];
        py.allow_threads(|| solve::solve(sys, &self.l, b, nrhs, &mut x, &mut self.ywork))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(x.into_pyarray(py).unbind())
    }

    /// `L` as a packed CSC lower-triangular matrix: `(indptr, indices, data)`.
    ///
    /// `rowfac` leaves the columns unpacked — column `j` is
    /// `Li [Lp[j] .. Lp[j]+Lnz[j])`, not `Lp[j+1]` — so this compacts them,
    /// which is what `cholmod_pack_factor` does and what every consumer of a
    /// `scipy.sparse` matrix expects.
    fn factor_csc(
        &self,
        py: Python<'_>,
    ) -> (Py<PyArray1<i64>>, Py<PyArray1<i64>>, Py<PyArray1<f64>>) {
        let n = self.l.n;
        let nz: i64 = self.l.nz.iter().sum();
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(nz as usize);
        let mut data = Vec::with_capacity(nz as usize);
        indptr.push(0i64);
        for j in 0..n {
            let p = self.l.p[j] as usize;
            let len = self.l.nz[j] as usize;
            indices.extend_from_slice(&self.l.i[p..p + len]);
            data.extend_from_slice(&self.l.x[p..p + len]);
            indptr.push(indices.len() as i64);
        }
        (
            indptr.into_pyarray(py).unbind(),
            indices.into_pyarray(py).unbind(),
            data.into_pyarray(py).unbind(),
        )
    }

    /// `½ log|det A|`.
    ///
    /// The diagonal of an `LDL'` factor holds `D`, not `L`, so this is
    /// `½ Σ log D_kk` there and `Σ log L_kk` for `LL'` — the same number either
    /// way. Raises rather than returning `-inf`/`nan` if the factorization
    /// stopped early, because a caller reading a log-determinant off a factor
    /// that does not exist is a bug, not a limit case.
    fn half_log_det(&self) -> PyResult<f64> {
        if self.l.minor < self.l.n {
            return Err(PyZeroDivisionError::new_err(format!(
                "matrix is not positive definite (leading minor {} is not)",
                self.l.minor + 1
            )));
        }
        let mut s = 0.0f64;
        for j in 0..self.l.n {
            s += self.l.x[self.l.p[j] as usize].ln();
        }
        Ok(if self.l.is_ll { s } else { 0.5 * s })
    }

    /// `L->Perm`: `L L' = P A P'`, i.e. `A[p][:, p]` is what was factorized.
    #[getter]
    fn perm(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        self.l.perm.clone().into_pyarray(py).unbind()
    }

    #[getter]
    fn n(&self) -> usize {
        self.l.n
    }

    /// `L->minor`: `n` if `A` is positive definite, else the first column at
    /// which it was found not to be.
    #[getter]
    fn minor(&self) -> usize {
        self.l.minor
    }

    #[getter]
    fn is_ll(&self) -> bool {
        self.l.is_ll
    }

    /// `Common->rowfacfl` from the last factorization.
    #[getter]
    fn rowfacfl(&self) -> f64 {
        self.fl
    }

    #[getter]
    fn nnz(&self) -> i64 {
        self.l.nz.iter().sum()
    }
}
