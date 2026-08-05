//! Sparse Cholesky — a self-contained replacement for the `scikit-sparse`
//! (CHOLMOD) dependency, so `hea` installs from a wheel on every platform with
//! no system SuiteSparse.
//!
//! Every routine here is a mechanical port of SuiteSparse **7.6.0** — the
//! version R's `Matrix` ships, hence the one `lme4` factorizes with, which is
//! what makes "hea == lme4" a checkable claim.
//!
//! The oracle for these routines is upstream's own C at that tag, compiled and
//! driven directly — *not* `scikit-sparse`. `sksparse`'s `F.perm` is the ordering
//! `cholmod_analyze` stores, which is the fill-reducing permutation already
//! composed with a weighted etree postorder (`cholmod_analyze.c:832-845`), so
//! it is a different quantity from what `cholmod_amd` returns and only
//! coincides when that postorder is the identity.

pub mod amd;
pub mod dense;
pub mod numeric;
pub mod py;
pub mod solve;
pub mod super_numeric;
pub mod super_solve;
pub mod super_symbolic;
pub mod symbolic;
#[cfg(test)]
pub mod testcorpus;
pub mod ws;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use amd::IntWidth;

/// `cholmod_amd` — the fill-reducing permutation CHOLMOD computes for a
/// symmetric matrix, before any symbolic analysis.
///
/// `indptr`/`indices` are a CSC pattern. `stype` selects the stored half the
/// way CHOLMOD's `A->stype` does: `> 0` upper, `< 0` lower; entries in the
/// other half are ignored rather than folded in. `use_long` picks which C build
/// to reproduce — `hash` is `uint32_t` in the `int32_t` build and `uint64_t` in
/// the `int64_t` one, which is observable only when `hash` overflows.
///
/// Returns `(Perm, info)`, with `Perm[k] = i` if row/column `i` of `A` is the
/// `k`th row/column of `P A P'`. `info` carries AMD's `Info` array under its
/// upstream names, plus the two derived quantities CHOLMOD reads back —
/// `lnz = n + Info[AMD_LNZ]` and `fl = Info[AMD_NDIV] + 2*Info[AMD_NMULTSUBS_LDL]
/// + n` (`cholmod_amd.c:177-180`). Both are slight upper bounds, and both are
/// what `cholmod_analyze`'s ordering trial loop ranks candidate orderings by.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, stype, dense=amd::DEFAULT_DENSE,
                    aggressive=amd::DEFAULT_AGGRESSIVE, use_long=false))]
#[allow(clippy::too_many_arguments)]
fn amd_order(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    stype: i32,
    dense: f64,
    aggressive: bool,
    use_long: bool,
) -> PyResult<(Py<PyArray1<i64>>, Py<PyDict>)> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    if stype == 0 {
        return Err(PyValueError::new_err(
            "stype must be nonzero: cholmod_amd orders a symmetric matrix, and \
             stype selects which triangle is the stored half",
        ));
    }
    let width = if use_long {
        IntWidth::I64
    } else {
        IntWidth::I32
    };
    // `cholmod_amd` validates the pattern itself — that check is what licenses
    // its kernels to index without re-checking, so it cannot be hoisted here.
    let (perm, info) = py
        .allow_threads(|| {
            let mut work = amd::Work::new(n);
            amd::cholmod_amd(
                n,
                indptr,
                indices,
                stype,
                dense,
                aggressive,
                width,
                &mut work.all(),
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("AMD_LNZ", info.lnz)?;
    d.set_item("AMD_NDIV", info.ndiv)?;
    d.set_item("AMD_NMULTSUBS_LDL", info.nms_ldl)?;
    d.set_item("AMD_NMULTSUBS_LU", info.nms_lu)?;
    d.set_item("AMD_NDENSE", info.ndense)?;
    d.set_item("AMD_DMAX", info.dmax)?;
    d.set_item("AMD_NCMPA", info.ncmpa)?;
    d.set_item("anz", info.anz)?;
    d.set_item("lnz", info.lnz(n))?;
    d.set_item("fl", info.fl(n))?;
    Ok((perm.into_pyarray(py).unbind(), d.unbind()))
}

/// `cholmod_analyze` for a symmetric matrix — the fill-reducing ordering, the
/// elimination tree, its weighted postordering, and the column counts of `L`.
///
/// `indptr`/`indices` are a CSC pattern and `stype` selects the stored half the
/// way CHOLMOD's `A->stype` does. `ordering` is `"amd"` or `"natural"`;
/// `default_strategy` says whether `Common->nmethods` was left at its default
/// 0, which is the only setting under which upstream would consider METIS.
///
/// Returns `perm`, `colcount`, `parent` and `post`, all in the final ordering —
/// i.e. with the weighted postorder already composed in, which is what makes
/// `perm` the same quantity as `scikit-sparse`'s `F.perm` (and different from
/// `amd_order`'s output). `fl`/`lnz` are the exact counts from
/// `cholmod_rowcolcounts`, not AMD's estimates; `metis_would_be_tried` reports
/// whether upstream's default strategy would have gone on to try METIS, which
/// is the one place this port can diverge from a CHOLMOD built with the
/// Partition module.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, stype, ordering="amd", default_strategy=true,
                    use_long=false))]
fn analyze(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    stype: i32,
    ordering: &str,
    default_strategy: bool,
    use_long: bool,
) -> PyResult<Py<PyDict>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let order = match ordering {
        "amd" => symbolic::Ordering::Amd,
        "natural" => symbolic::Ordering::Natural,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'amd' or 'natural', not {other:?}"
            )));
        }
    };
    let width = if use_long {
        IntWidth::I64
    } else {
        IntWidth::I32
    };
    let s = py
        .allow_threads(|| {
            symbolic::analyze(n, indptr, indices, stype, order, default_strategy, width)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("perm", s.perm.into_pyarray(py))?;
    d.set_item("colcount", s.colcount.into_pyarray(py))?;
    d.set_item("parent", s.parent.into_pyarray(py))?;
    d.set_item("post", s.post.into_pyarray(py))?;
    d.set_item("fl", s.fl)?;
    d.set_item("lnz", s.lnz)?;
    d.set_item("anz", s.anz)?;
    d.set_item(
        "ordering",
        match s.ordering {
            symbolic::Ordering::Amd => "amd",
            symbolic::Ordering::Natural => "natural",
            symbolic::Ordering::Postordered => "postordered",
        },
    )?;
    d.set_item("metis_would_be_tried", s.metis_would_be_tried)?;
    if let Some(info) = s.amd {
        /* AMD's own estimates, which are what the METIS decision is taken on
         * — deliberately not the exact counts above */
        d.set_item("amd_fl", info.fl(n))?;
        d.set_item("amd_lnz", info.lnz(n))?;
        d.set_item("amd_anz", info.anz)?;
    }
    Ok(d.unbind())
}

/// `cholmod_analyze` followed by `cholmod_factorize` — the simplicial numeric
/// `LDL'` (or `LL'`) of `beta*I + P A P'`.
///
/// `indptr`/`indices`/`data` are a CSC matrix and `stype` selects the stored
/// half. The remaining arguments are the `cholmod_common` fields
/// `cholmod_factorize_p` reads, at their `cholmod_defaults` values.
///
/// Returns `L` in the internal unpacked form `cholmod_rowfac` leaves it in —
/// column `j` occupies `Li [Lp[j] .. Lp[j]+Lnz[j])`, which is not `Lp[j+1]`
/// unless `L` happens to be packed — plus `minor`, which is `n` when `A` was
/// positive definite and otherwise the column where it stopped being so.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, stype, beta=0.0, ordering="amd",
                    final_ll=false, final_asis=true, final_pack=true,
                    final_monotonic=true))]
#[allow(clippy::too_many_arguments)]
fn factorize(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    data: PyReadonlyArray1<'_, f64>,
    stype: i32,
    beta: f64,
    ordering: &str,
    final_ll: bool,
    final_asis: bool,
    final_pack: bool,
    final_monotonic: bool,
) -> PyResult<Py<PyDict>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    let order = match ordering {
        "amd" => symbolic::Ordering::Amd,
        "natural" => symbolic::Ordering::Natural,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'amd' or 'natural', not {other:?}"
            )));
        }
    };
    let params = numeric::Params {
        final_ll,
        final_asis,
        final_pack,
        final_monotonic,
        ..numeric::Params::default()
    };

    let (l, fl) = py
        .allow_threads(|| -> Result<_, String> {
            let nz = ws::validate_csc(n, indptr, indices).map_err(|e| e.to_string())?;
            if nz > data.len() {
                return Err(format!(
                    "data has length {}, expected at least indptr[n] = {nz}",
                    data.len()
                ));
            }
            /* one copy, analyzed and factorized from — `cholmod_analyze` and
             * `cholmod_factorize` take the same `cholmod_sparse *A` */
            let a = symbolic::Sparse {
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
            let mut l = numeric::Factor::from_symbolic(&s);
            let mut work = ws::Work::new(n);
            let fl = numeric::factorize(&a, beta, &mut l, &params, &mut work)
                .map_err(|e| e.to_string())?;
            Ok((l, fl))
        })
        .map_err(PyValueError::new_err)?;

    let d = PyDict::new(py);
    d.set_item("perm", l.perm.into_pyarray(py))?;
    d.set_item("colcount", l.colcount.into_pyarray(py))?;
    d.set_item("Lp", l.p.into_pyarray(py))?;
    d.set_item("Lnz", l.nz.into_pyarray(py))?;
    d.set_item("nzmax", l.i.len())?;
    d.set_item("Li", l.i.into_pyarray(py))?;
    d.set_item("Lx", l.x.into_pyarray(py))?;
    d.set_item("minor", l.minor)?;
    d.set_item("is_ll", l.is_ll)?;
    d.set_item("is_monotonic", l.is_monotonic)?;
    d.set_item("rowfacfl", fl)?;
    Ok(d.unbind())
}

/// `cholmod_analyze` followed by its supernodal branch
/// (`cholmod_analyze.c:886-901`) — the supernodal *symbolic* factor.
///
/// `indptr`/`indices` are a CSC pattern and `stype` selects the stored half.
/// Returns `L`'s supernodal fields (`nsuper`, `super`, `pi`, `px`, `s`,
/// `ssize`, `xsize`, `maxcsize`, `maxesize`) alongside `Perm`, plus
/// `auto_supernodal`: whether upstream's default `CHOLMOD_AUTO` would have
/// taken this branch at all, which the caller forces here either way.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, stype, ordering="amd"))]
fn super_analyze(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    stype: i32,
    ordering: &str,
) -> PyResult<Py<PyDict>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let order = match ordering {
        "amd" => symbolic::Ordering::Amd,
        "natural" => symbolic::Ordering::Natural,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'amd' or 'natural', not {other:?}"
            )));
        }
    };
    let (s, ss) = py
        .allow_threads(|| -> Result<_, String> {
            let nz = ws::validate_csc(n, indptr, indices).map_err(|e| e.to_string())?;
            let a = symbolic::Sparse {
                n,
                p: indptr[..n + 1].to_vec(),
                i: indices[..nz].to_vec(),
                x: Vec::new(),
                numeric: false,
                stype,
                sorted: ws::columns_are_sorted(n, indptr, indices),
            };
            let s = symbolic::analyze_sparse(&a, order, true, IntWidth::I64)
                .map_err(|e| e.to_string())?;
            let mut work = ws::Work::new(n);
            /* the supernodal branch permutes A a second time, for pattern
             * only: `permute_matrices (..., FALSE, ...)` */
            let a2 = symbolic::permute_sym(&a, s.ordering, &s.perm, false, false, &mut work.all());
            let sup = a2.as_ref().unwrap_or(&a);
            let ss = super_symbolic::super_symbolic(
                sup,
                &s.parent,
                &s.colcount,
                &super_symbolic::Relax::default(),
                &mut work,
            )
            .map_err(|e| e.to_string())?;
            Ok((s, ss))
        })
        .map_err(PyValueError::new_err)?;

    let d = PyDict::new(py);
    d.set_item("perm", s.perm.into_pyarray(py))?;
    d.set_item("parent", s.parent.into_pyarray(py))?;
    d.set_item("colcount", s.colcount.into_pyarray(py))?;
    d.set_item(
        "auto_supernodal",
        super_symbolic::auto_supernodal(s.fl, s.lnz, super_symbolic::DEFAULT_SUPERNODAL_SWITCH),
    )?;
    d.set_item("n", ss.n)?;
    d.set_item("nsuper", ss.nsuper)?;
    d.set_item("super", ss.sup.into_pyarray(py))?;
    d.set_item("pi", ss.pi.into_pyarray(py))?;
    d.set_item("px", ss.px.into_pyarray(py))?;
    d.set_item("s", ss.s.into_pyarray(py))?;
    d.set_item("ssize", ss.ssize)?;
    d.set_item("xsize", ss.xsize)?;
    d.set_item("maxcsize", ss.maxcsize)?;
    d.set_item("maxesize", ss.maxesize)?;
    Ok(d.unbind())
}

/// `cholmod_analyze` + `cholmod_factorize` with the supernodal path forced —
/// the supernodal `LL'` of `beta*I + P A P'`.
///
/// Returns the supernodal factor whole: the pattern (as `super_analyze` does)
/// plus `L->x`, the concatenation of every supernode's dense
/// `nsrow`-by-`nscol` column-major block, and `minor`.
/// `numeric_reps > 0` also returns `numeric_ms`, the best of that many
/// factorizations against one symbolic analysis — the *re*factorization cost,
/// which is what a caller holding a factor pays and what the plan's acceptance
/// bar is stated in. It cannot be had by differencing two whole-pipeline
/// timings: the analysis is a third of them and the noise swamps the rest.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, stype, beta=0.0, ordering="amd",
                    numeric_reps=0))]
#[allow(clippy::too_many_arguments)]
fn super_factorize(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    data: PyReadonlyArray1<'_, f64>,
    stype: i32,
    beta: f64,
    ordering: &str,
    numeric_reps: usize,
) -> PyResult<Py<PyDict>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    let order = match ordering {
        "amd" => symbolic::Ordering::Amd,
        "natural" => symbolic::Ordering::Natural,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'amd' or 'natural', not {other:?}"
            )));
        }
    };
    let l = py
        .allow_threads(|| -> Result<_, String> {
            let nz = ws::validate_csc(n, indptr, indices).map_err(|e| e.to_string())?;
            if data.len() < nz {
                return Err(format!(
                    "data has length {}, expected at least indptr[n] = {nz}",
                    data.len()
                ));
            }
            let a = symbolic::Sparse {
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
            let mut work = ws::Work::new(n);
            let a2 = symbolic::permute_sym(&a, s.ordering, &s.perm, false, false, &mut work.all());
            let sym = super_symbolic::super_symbolic(
                a2.as_ref().unwrap_or(&a),
                &s.parent,
                &s.colcount,
                &super_symbolic::Relax::default(),
                &mut work,
            )
            .map_err(|e| e.to_string())?;
            let mut l = super_numeric::SuperFactor::new(&s, sym);
            let mut cwork = super_numeric::SuperWork::new();
            super_numeric::super_factorize(&a, beta, &mut l, &mut work, &mut cwork)
                .map_err(|e| e.to_string())?;

            let mut best = f64::INFINITY;
            for _ in 0..numeric_reps {
                let t0 = std::time::Instant::now();
                super_numeric::super_factorize(&a, beta, &mut l, &mut work, &mut cwork)
                    .map_err(|e| e.to_string())?;
                best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            Ok((l, if numeric_reps == 0 { 0.0 } else { best }))
        })
        .map_err(PyValueError::new_err)?;
    let (l, numeric_ms) = l;

    let d = PyDict::new(py);
    d.set_item("numeric_ms", numeric_ms)?;
    d.set_item("perm", l.perm.into_pyarray(py))?;
    d.set_item("colcount", l.colcount.into_pyarray(py))?;
    d.set_item("nsuper", l.sym.nsuper)?;
    d.set_item("super", l.sym.sup.into_pyarray(py))?;
    d.set_item("pi", l.sym.pi.into_pyarray(py))?;
    d.set_item("px", l.sym.px.into_pyarray(py))?;
    d.set_item("s", l.sym.s.into_pyarray(py))?;
    d.set_item("xsize", l.sym.xsize)?;
    d.set_item("Lx", l.x.into_pyarray(py))?;
    d.set_item("minor", l.minor)?;
    Ok(d.unbind())
}

/// `cholmod_analyze` + `cholmod_factorize` + `cholmod_solve`, supernodal.
///
/// `b` is `n`-by-`nrhs` column-major; the solution comes back in the same
/// layout. `sys` names the system in the spelling `cholmod.h` uses — `"A"`,
/// `"LDLt"`, `"L"`, `"LD"`, `"Lt"`, `"DLt"`, `"D"`, `"P"`, `"Pt"` — so each
/// half of the solve can be exercised on its own.
///
/// `solve_reps > 0` also returns `solve_ms`, the best of that many solves
/// against the one factor, reusing the workspace as a caller holding a factor
/// would. The solve is a few percent of analyze+factorize, so it cannot be
/// measured by differencing two whole-pipeline timings.
#[pyfunction]
#[pyo3(name = "super_solve")]
#[pyo3(signature = (n, indptr, indices, data, b, nrhs, stype, sys="A", beta=0.0,
                    ordering="amd", solve_reps=0))]
#[allow(clippy::too_many_arguments)]
fn supernodal_solve(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    data: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    nrhs: usize,
    stype: i32,
    sys: &str,
    beta: f64,
    ordering: &str,
    solve_reps: usize,
) -> PyResult<Py<PyDict>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    let b = b.as_slice()?;
    let order = match ordering {
        "amd" => symbolic::Ordering::Amd,
        "natural" => symbolic::Ordering::Natural,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'amd' or 'natural', not {other:?}"
            )));
        }
    };
    let sys = match sys {
        "A" => solve::Sys::A,
        "LDLt" => solve::Sys::LDLt,
        "LD" => solve::Sys::LD,
        "DLt" => solve::Sys::DLt,
        "L" => solve::Sys::L,
        "Lt" => solve::Sys::Lt,
        "D" => solve::Sys::D,
        "P" => solve::Sys::P,
        "Pt" => solve::Sys::Pt,
        other => return Err(PyValueError::new_err(format!("unknown system {other:?}"))),
    };
    let (x, minor, solve_ms) = py
        .allow_threads(|| -> Result<_, String> {
            let nz = ws::validate_csc(n, indptr, indices).map_err(|e| e.to_string())?;
            if data.len() < nz {
                return Err(format!("data has {} entries, need {nz}", data.len()));
            }
            if b.len() < n * nrhs {
                return Err(format!("b has {} entries, need {}", b.len(), n * nrhs));
            }
            let a = symbolic::Sparse {
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
            let mut work = ws::Work::new(n);
            let a2 = symbolic::permute_sym(&a, s.ordering, &s.perm, false, false, &mut work.all());
            let sym = super_symbolic::super_symbolic(
                a2.as_ref().unwrap_or(&a),
                &s.parent,
                &s.colcount,
                &super_symbolic::Relax::default(),
                &mut work,
            )
            .map_err(|e| e.to_string())?;
            let mut l = super_numeric::SuperFactor::new(&s, sym);
            let mut cwork = super_numeric::SuperWork::new();
            super_numeric::super_factorize(&a, beta, &mut l, &mut work, &mut cwork)
                .map_err(|e| e.to_string())?;
            let mut x = vec![0.0f64; n * nrhs];
            let mut swork = super_solve::SuperSolveWork::new();
            super_solve::super_solve(sys, &l, &b[..n * nrhs], nrhs, &mut x, &mut swork)
                .map_err(|e| e.to_string())?;
            let mut best = f64::INFINITY;
            for _ in 0..solve_reps {
                let t0 = std::time::Instant::now();
                super_solve::super_solve(sys, &l, &b[..n * nrhs], nrhs, &mut x, &mut swork)
                    .map_err(|e| e.to_string())?;
                best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            Ok((x, l.minor, if solve_reps == 0 { 0.0 } else { best }))
        })
        .map_err(PyValueError::new_err)?;

    let d = PyDict::new(py);
    d.set_item("X", x.into_pyarray(py))?;
    d.set_item("minor", minor)?;
    d.set_item("solve_ms", solve_ms)?;
    Ok(d.unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(amd_order, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(super_analyze, m)?)?;
    m.add_function(wrap_pyfunction!(super_factorize, m)?)?;
    m.add_function(wrap_pyfunction!(supernodal_solve, m)?)?;
    m.add_class::<py::CholFactor>()?;
    Ok(())
}
