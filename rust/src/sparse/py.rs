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

use std::borrow::Cow;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

use super::amd::IntWidth;
use super::numeric::{self, Factor, Params};
use super::solve::{self, SolveWork, Sys};
use super::spinv;
use super::super_numeric::{self, SuperFactor, SuperWork};
use super::super_solve::{self, SuperSolveWork};
use super::super_symbolic;
use super::symbolic::{self, Method, Ordering, Sparse};
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

/// The `ordering=` argument every entry point in this crate takes.
///
/// `"best"` is `Common->nmethods == 0`, CHOLMOD's own default: try each of
/// [`symbolic::DEFAULT_METHODS`] and keep the one with the smallest nnz(L).
/// The other two pin a single method, which is what `sksparse`'s
/// `ordering_method=` does and what a parity test against a specific ordering
/// needs.
pub(super) fn parse_method(s: &str) -> PyResult<Method> {
    Ok(match s {
        "best" => Method::Default,
        "amd" => Method::Pinned(Ordering::Amd),
        "natural" => Method::Pinned(Ordering::Natural),
        "metis" => Method::Pinned(Ordering::Metis),
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'best', 'amd', 'metis' or 'natural', not {other:?}"
            )));
        }
    })
}

/// `L->ordering` on the way out, as [`parse_method`] would spell it. Never
/// `"best"`: the trial loop reports what it *selected*.
pub(super) fn ordering_name(o: Ordering) -> &'static str {
    match o {
        Ordering::Amd => "amd",
        Ordering::Metis => "metis",
        Ordering::Natural => "natural",
        Ordering::Postordered => "postordered",
    }
}

/// `Common->supernodal` (`t_cholmod_defaults.c:42`).
pub(super) fn parse_supernodal(s: &str) -> PyResult<SuperMode> {
    Ok(match s {
        "auto" => SuperMode::Auto,
        "simplicial" => SuperMode::Simplicial,
        "supernodal" => SuperMode::Super,
        other => {
            return Err(PyValueError::new_err(format!(
                "supernodal must be 'auto', 'simplicial' or 'supernodal', not {other:?}"
            )));
        }
    })
}

/// `CHOLMOD_SIMPLICIAL` / `CHOLMOD_AUTO` / `CHOLMOD_SUPERNODAL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SuperMode {
    Simplicial,
    Auto,
    Super,
}

/// `Common->supernodal_switch` (`t_cholmod_defaults.c:43`) — `CHOLMOD_AUTO`
/// takes the supernodal branch when `fl / lnz` reaches this
/// (`cholmod_analyze.c:887-891`).
const SUPERNODAL_SWITCH: f64 = 40.0;

/// Which factorization `L` holds, with the workspace that path needs.
///
/// Upstream carries both in one `cholmod_factor` behind `L->is_super` and
/// switches on it in `cholmod_factorize_p` (`:172`). The two have disjoint
/// fields and disjoint workspaces, so they are an enum here rather than a
/// struct with half its members unused.
enum Kind {
    Simplicial(Box<Simp>),
    Supernodal(Box<Super>),
}

/// The simplicial factor and the two things only it needs.
struct Simp {
    l: Factor,
    params: Params,
    ywork: SolveWork,
}

/// The supernodal factor and the two things only it needs.
struct Super {
    l: SuperFactor,
    cwork: SuperWork,
    ywork: SuperSolveWork,
}

/// Whether the pattern `(bp, bi)` is inside `(ap, ai)` — the new `A` inside the
/// one `L`'s supernodes were built on.
///
/// Both patterns have their row indices ascending, so this is a merge, not a
/// search.
///
/// Only the **supernodal** path asks. There the symbolic analysis fixes where
/// every entry of `L` lives, so an entry of `A` outside that pattern has nowhere
/// to be assembled; the answer decides whether the analysis can be reused or has
/// to be redone. The simplicial path never asks: `rowfac` derives each row's
/// pattern from `A` and the etree as it goes and grows `L` when it has to, so a
/// wider `A` is simply factorized.
fn pattern_is_contained(n: usize, ap: &[i64], ai: &[i64], bp: &[i64], bi: &[i64]) -> bool {
    for j in 0..n {
        let (mut q, qend) = (ap[j] as usize, ap[j + 1] as usize);
        for &row in &bi[bp[j] as usize..bp[j + 1] as usize] {
            while q < qend && ai[q] < row {
                q += 1;
            }
            if q == qend || ai[q] != row {
                return false;
            }
            q += 1;
        }
    }
    true
}

/// `cholmod_analyze` followed by `cholmod_factorize_p` — the body both the
/// constructor and a re-analyzing [`CholFactor::refactorize`] run.
fn analyze_and_factorize(
    a: &Sparse,
    beta: f64,
    method: Method,
    mode: SuperMode,
    params: Params,
    work: &mut Work,
) -> Result<(Kind, f64), String> {
    let s = symbolic::analyze_sparse(a, method, IntWidth::I64, work).map_err(|e| e.to_string())?;

    /* supernodal analysis, if requested or if selected automatically
     * (`cholmod_analyze.c:882-902`) */
    let want_super = match mode {
        SuperMode::Simplicial => false,
        SuperMode::Super => true,
        SuperMode::Auto => s.lnz > 0.0 && (s.fl / s.lnz) >= SUPERNODAL_SWITCH,
    };

    if want_super {
        let a2 = symbolic::permute_sym(a, s.ordering, &s.perm, false, false, &mut work.all());
        let sym = super_symbolic::super_symbolic(
            a2.as_ref().unwrap_or(a),
            &s.parent,
            &s.colcount,
            &super_symbolic::Relax::default(),
            work,
        )
        .map_err(|e| e.to_string())?;
        /* `a2` is the pattern-only `P A P'` the supernodal *symbolic* needs and
         * nothing after it does: `super_factorize` builds its own numeric
         * `tril (P A P')`. Rust would keep it to the end of the block, so it
         * would be live across the allocation of `L->x` — 0.4 GB against 4.7 on
         * a 3.4M-row system. */
        drop(a2);
        /* `SuperFactor::new` consumes the analysis, so `Perm` and `ColCount`
         * move into `L` the way `Lperm = L->Perm` puts them there upstream, and
         * `parent`/`post` — the analysis's own scratch, which nothing numeric
         * reads — are freed here rather than at the end of the block. */
        let mut l = SuperFactor::new(s, sym);
        let mut cwork = SuperWork::new();
        super_numeric::super_factorize(a, beta, &mut l, work, &mut cwork)
            .map_err(|e| e.to_string())?;
        return Ok((
            Kind::Supernodal(Box::new(Super {
                l,
                cwork,
                ywork: SuperSolveWork::new(),
            })),
            0.0,
        ));
    }

    let mut l = Factor::from_symbolic(s);
    let fl = numeric::factorize(a, beta, &mut l, &params, work).map_err(|e| e.to_string())?;
    Ok((
        Kind::Simplicial(Box::new(Simp {
            l,
            params,
            ywork: SolveWork::new(),
        })),
        fl,
    ))
}

/// A numeric Cholesky factorization, reusable for both new values and repeated
/// solves.
#[pyclass(module = "hea._rs")]
pub struct CholFactor {
    /// `A->p` and `A->i` of the last matrix factorized — the **pattern**, and
    /// not the values.
    ///
    /// `cholmod_factorize (A, L, Common)` is handed the whole matrix on every
    /// call and keeps none of it, so [`CholFactor::refactorize`] takes the
    /// caller's arrays as they arrive and builds a borrowing [`Sparse`] over
    /// them. Two things still need the pattern after the call returns, and
    /// nothing needs the values:
    ///
    /// * [`pattern_is_contained`], to decide whether the analysis still holds;
    /// * [`CholFactor::factor_csc`], whose `resymbol_noperm` re-derives `L`'s
    ///   simplicial pattern from `A`'s and reads no `A->x` at all.
    ///
    /// So `A->x` is not stored, which is 367 MiB of the 760 that the whole
    /// matrix was on a 3.4M-row system. The remaining 393 is the price of
    /// pruning `factor_csc`'s output, and it is a price rather than an
    /// oversight: `scikit-sparse` skips `resymbol` — `L()` is
    /// `cholmod_factor_to_sparse` alone — and so returns the entries relaxed
    /// amalgamation added.
    ap: Vec<i64>,
    ai: Vec<i64>,
    /// `A->stype`, kept with the pattern for the same reason: it is what says
    /// which triangle those indices are.
    stype: i32,
    /// `A->sorted` for that pattern, so a `refactorize` whose pattern is
    /// unchanged does not sweep `A->i` again to rediscover it.
    sorted: bool,
    kind: Kind,
    work: Work,
    /// `Common->rowfacfl` from the last factorization. Simplicial only —
    /// `cholmod_super_numeric` does not keep a flop count.
    fl: f64,
    /// What `cholmod_analyze` was told, kept so the analysis can be *redone*
    /// when a later `A` outgrows it — see [`CholFactor::refactorize`].
    method: Method,
    mode: SuperMode,
    params: Params,
}

impl CholFactor {
    /// The supernodal factor converted to a simplicial one, or `None` when it
    /// is simplicial already and the caller should read `Kind::Simplicial`.
    ///
    /// `from_supernodal` followed by `resymbol_noperm`, which is the one
    /// conversion in this file: `.L` and the selected inverse both come through
    /// here so they cannot drift apart.
    fn build_simplicial(&mut self) -> PyResult<Option<Factor>> {
        let CholFactor {
            ap,
            ai,
            stype,
            sorted,
            kind,
            work,
            ..
        } = self;
        let Kind::Supernodal(k) = kind else {
            return Ok(None);
        };
        let mut f = Factor::from_supernodal(&k.l);
        /* resymbol wants tril (P A P'), which is the S the supernodal
         * factorization was handed (`cholmod_factorize.c:275`) — but only its
         * pattern: `cholmod_resymbol_noperm` reads `A->p` and `A->i` and never
         * `A->x`, which is why the values are not kept and this builds a
         * `CHOLMOD_PATTERN` matrix. */
        let a = Sparse {
            n: k.l.n,
            p: Cow::Borrowed(ap),
            i: Cow::Borrowed(ai),
            x: Cow::Borrowed(&[]),
            numeric: false,
            stype: *stype,
            sorted: *sorted,
        };
        const PATTERN: bool = false;
        let s = symbolic::permute_sym(&a, f.ordering, &f.perm, PATTERN, true, &mut work.all());
        numeric::resymbol_noperm(s.as_ref().unwrap_or(&a), true, &mut f, &mut work.all())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Some(f))
    }

    /// The selected inverse of `P A P'`, with the permutation to undo it.
    ///
    /// Upstream's recursion is stated for `LDL'`, so an `LL'` factor is
    /// converted through [`Factor::change_factor`] first — on a copy, since a
    /// caller's factor must still be the one they factorized.
    fn selected(&mut self, _py: Python<'_>) -> PyResult<(spinv::Selected, Vec<i64>)> {
        let converted = self.build_simplicial()?;
        let params = match &self.kind {
            Kind::Simplicial(k) => k.params,
            /* `from_supernodal` produces a packed monotonic `LL'`, and the
             * defaults are what `change_factor` needs to take it to `LDL'`. */
            Kind::Supernodal(_) => Params::default(),
        };

        let mut owned;
        let l: &Factor = match (&converted, &self.kind) {
            (Some(f), _) => f,
            (None, Kind::Simplicial(k)) => &k.l,
            (None, Kind::Supernodal(_)) => unreachable!("build_simplicial converts the supernodal"),
        };
        let l = if l.is_ll {
            owned = l.clone();
            owned
                .change_factor(false, true, true, &params)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            &owned
        } else {
            l
        };

        let (z, _flops) =
            spinv::selected_inverse(l).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((z, l.perm.clone()))
    }
}

#[pymethods]
impl CholFactor {
    /// `cholmod_analyze` followed by `cholmod_factorize_p`.
    ///
    /// `stype` selects the stored half the way `A->stype` does.
    ///
    /// `supernodal` is `Common->supernodal`: `"auto"` takes the supernodal
    /// branch when `fl / lnz >= 40` (`cholmod_analyze.c:887-891`), which is
    /// CHOLMOD's own default and 5-20x faster on the matrices that trip it.
    /// The supernodal factorization is `LL'` and only `LL'`, so `use_ll` is
    /// read by the simplicial path alone — pass `supernodal="simplicial"` to
    /// insist on an `LDL'`. The two forms disagree about which matrices are
    /// factorizable at all (`rowfac` fails an `LL'` on any non-positive pivot
    /// and an `LDL'` only on a zero one), so that is a real choice and not a
    /// presentational one.
    #[new]
    #[pyo3(signature = (n, indptr, indices, data, stype, beta=0.0, ordering="best",
                        use_ll=false, supernodal="auto"))]
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
        supernodal: &str,
    ) -> PyResult<CholFactor> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let data = data.as_slice()?;
        let method = parse_method(ordering)?;
        let mode = parse_supernodal(supernodal)?;
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
                p: Cow::Borrowed(&indptr[..n + 1]),
                i: Cow::Borrowed(&indices[..nz]),
                x: Cow::Borrowed(&data[..nz]),
                numeric: true,
                stype,
                sorted: ws::columns_are_sorted(n, indptr, indices),
            };
            let mut work = Work::new(n);
            let sorted = a.sorted;
            let (kind, fl) = analyze_and_factorize(&a, beta, method, mode, params, &mut work)?;
            Ok(CholFactor {
                ap: indptr[..n + 1].to_vec(),
                ai: indices[..nz].to_vec(),
                stype,
                sorted,
                kind,
                work,
                fl,
                method,
                mode,
                params,
            })
        })
        .map_err(PyValueError::new_err)
    }

    /// Refactorize new values against the same symbolic analysis —
    /// `cholmod_factorize (A, L, Common)` on a numeric `L`.
    ///
    /// **The pattern comes from `A`, and is not assumed to match.** It usually
    /// does, and then this is the value copy it looks like. But a caller that
    /// builds `A` as a product recomputes its pattern every time, and an entry
    /// that comes out numerically zero is one the product simply does not
    /// emit — so the pattern moves with the values. `gmm`'s
    /// `M = Λ Zᵀ Z Λᵀ + I` does it in both directions: it drops to
    /// block-diagonal when the optimizer tries a zero variance component, and
    /// on `nlme::Machines` with `(Machine|Worker)` it *gains* the `(2,1)` entry
    /// of each block the moment a correlation goes nonzero, because no
    /// observation is on two machines at once and `Zᵀ Z` has a structural zero
    /// there. `cholmod_factorize` is handed the whole `A` for exactly this
    /// reason, so this is too.
    ///
    /// Whether a *wider* `A` is allowed is a property of the path, not a policy
    /// choice: [`pattern_is_contained`] says which and why.
    #[pyo3(signature = (indptr, indices, data, beta=0.0))]
    fn refactorize(
        &mut self,
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f64>,
        beta: f64,
    ) -> PyResult<()> {
        let (indptr, indices, data) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
        let (method, mode, params, stype) = (self.method, self.mode, self.params, self.stype);
        let n = self.n();
        if indptr.len() != self.ap.len() {
            return Err(PyValueError::new_err(format!(
                "indptr has length {}, expected n + 1 = {}",
                indptr.len(),
                self.ap.len()
            )));
        }
        let nz = indptr[n] as usize;
        if nz > indices.len() || nz > data.len() {
            return Err(PyValueError::new_err(format!(
                "indptr[n] = {nz} is out of range for {} row indices and {} values",
                indices.len(),
                data.len()
            )));
        }
        let CholFactor {
            ap,
            ai,
            sorted,
            kind,
            work,
            fl,
            ..
        } = self;
        py.allow_threads(|| -> Result<(), String> {
            /* The pattern is usually the one already analyzed, and then this is
             * the value copy it looks like: the comparison is what lets both
             * `validate_csc` and `columns_are_sorted` — two O(nnz) sweeps — be
             * skipped, which is the whole point of holding the pattern. */
            let same = indptr == &ap[..] && indices[..nz] == ai[..];
            if !same {
                ws::validate_csc(n, indptr, &indices[..nz]).map_err(|e| e.to_string())?;
                *sorted = ws::columns_are_sorted(n, indptr, indices);
            }
            /* the matrix `cholmod_factorize (A, L, Common)` is handed: a view
             * onto the caller's three arrays, built fresh on every call and
             * kept by nobody */
            let a = Sparse {
                n,
                p: Cow::Borrowed(&indptr[..n + 1]),
                i: Cow::Borrowed(&indices[..nz]),
                x: Cow::Borrowed(&data[..nz]),
                numeric: true,
                stype,
                sorted: *sorted,
            };
            /* Only the supernodal analysis can fail to hold a wider `A`;
             * see `pattern_is_contained`. When it cannot, redo it — that is
             * `cholmod_analyze` + `cholmod_factorize` together, which is
             * what a caller would otherwise have to do by hand, and it is
             * bounded: a pattern can only grow up to the structural product
             * that produced it, so this fires a handful of times per fit at
             * worst and never in the steady state. */
            let reanalyze = !same
                && matches!(kind, Kind::Supernodal(_))
                && !pattern_is_contained(n, ap, ai, indptr, &indices[..nz]);
            if !same {
                ap.copy_from_slice(indptr);
                ai.clear();
                ai.extend_from_slice(&indices[..nz]);
            }
            if reanalyze {
                let (k, f) = analyze_and_factorize(&a, beta, method, mode, params, work)?;
                *kind = k;
                *fl = f;
                return Ok(());
            }
            match kind {
                Kind::Simplicial(k) => {
                    *fl = numeric::factorize(&a, beta, &mut k.l, &k.params, work)
                        .map_err(|e| e.to_string())?;
                }
                Kind::Supernodal(k) => {
                    super_numeric::super_factorize(&a, beta, &mut k.l, work, &mut k.cwork)
                        .map_err(|e| e.to_string())?;
                }
            }
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
        let n = self.n();
        if b.len() != n * nrhs {
            return Err(PyValueError::new_err(format!(
                "b has length {}, expected n*nrhs = {}",
                b.len(),
                n * nrhs
            )));
        }
        let mut x = vec![0.0f64; n * nrhs];
        let kind = &mut self.kind;
        py.allow_threads(|| match kind {
            Kind::Simplicial(k) => solve::solve(sys, &k.l, b, nrhs, &mut x, &mut k.ywork),
            Kind::Supernodal(k) => {
                super_solve::super_solve(sys, &k.l, b, nrhs, &mut x, &mut k.ywork)
            }
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(x.into_pyarray(py).unbind())
    }

    /// `L` as a packed CSC lower-triangular matrix: `(indptr, indices, data)`.
    ///
    /// `rowfac` leaves the columns unpacked — column `j` is
    /// `Li [Lp[j] .. Lp[j]+Lnz[j])`, not `Lp[j+1]` — so this compacts them,
    /// which is what `cholmod_pack_factor` does and what every consumer of a
    /// `scipy.sparse` matrix expects.
    ///
    /// A supernodal factor is converted first, and **pruned**: relaxed
    /// supernode amalgamation leaves entries that are not in `L`, and dropping
    /// them is what makes this return the same matrix whichever path ran.
    ///
    /// That is upstream's `Common->final_resymbol`, and it is a **deviation**,
    /// deliberately: CHOLMOD leaves the flag off by default, and applies it
    /// only when `cholmod_factorize_p` has just converted `L` to simplicial
    /// (`:275`). `scikit-sparse` never sets it and never resymbols — its `L()`
    /// is `change_factor` plus `cholmod_factor_to_sparse` — so it hands back the
    /// amalgamation's structural zeros and its `L` differs between the two
    /// paths. This is what that choice costs: [`CholFactor::ap`]/`ai`, because
    /// `resymbol_noperm` re-derives the pattern from `A`'s.
    fn factor_csc(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<i64>>, Py<PyArray1<f64>>)> {
        let simplicial = self.build_simplicial()?;
        let l: &Factor = match (&simplicial, &self.kind) {
            (Some(f), _) => f,
            (None, Kind::Simplicial(k)) => &k.l,
            (None, Kind::Supernodal(_)) => unreachable!("build_simplicial converts the supernodal"),
        };

        let n = l.n;
        let nz: i64 = l.nz.iter().sum();
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(nz as usize);
        let mut data = Vec::with_capacity(nz as usize);
        indptr.push(0i64);
        for j in 0..n {
            let p = l.p[j] as usize;
            let len = l.nz[j] as usize;
            indices.extend_from_slice(&l.i[p..p + len]);
            data.extend_from_slice(&l.x[p..p + len]);
            indptr.push(indices.len() as i64);
        }
        Ok((
            indptr.into_pyarray(py).unbind(),
            indices.into_pyarray(py).unbind(),
            data.into_pyarray(py).unbind(),
        ))
    }

    /// `½ log|det A|`.
    ///
    /// The diagonal of an `LDL'` factor holds `D`, not `L`, so this is
    /// `½ Σ log D_kk` there and `Σ log L_kk` for `LL'` — the same number either
    /// way. Raises rather than returning `-inf`/`nan` if the factorization
    /// stopped early, because a caller reading a log-determinant off a factor
    /// that does not exist is a bug, not a limit case.
    /// `diag(A⁻¹)`, in `A`'s own ordering.
    ///
    /// Takahashi's recursion over the factor, so `A⁻¹` is never formed. Costs
    /// about one numeric factorization — `Σ_j |L_j|²`, not `nnz(L)` — and holds
    /// roughly two `L`s while it runs.
    fn inv_diagonal(&mut self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let (z, perm) = self.selected(py)?;
        let d = z.diagonal();
        /* L factors P A P', so entry i of the sweep is A's row perm[i]. */
        let mut out = vec![0.0f64; z.n];
        for i in 0..z.n {
            out[perm[i] as usize] = d[i];
        }
        Ok(out.into_pyarray(py).unbind())
    }

    /// The selected inverse as CSC `(indptr, indices, data)`, in `A`'s ordering.
    ///
    /// The pattern is `L + L'` pulled back through the permutation, so the
    /// columns are re-sorted on the way out.
    fn selected_inverse(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<i64>>, Py<PyArray1<f64>>)> {
        let (z, perm) = self.selected(py)?;
        let n = z.n;
        let nnz = z.i.len();

        /* Count each unpermuted column, then fill and sort. `perm[i]` is the
         * row of A that the factor's row i came from, so an entry at (i, j)
         * lands at (perm[i], perm[j]). */
        let mut count = vec![0i64; n + 1];
        for j in 0..n {
            let pj = perm[j] as usize;
            count[pj + 1] += z.p[j + 1] - z.p[j];
        }
        for j in 0..n {
            count[j + 1] += count[j];
        }
        let mut next: Vec<i64> = count[..n].to_vec();
        let mut oi = vec![0i64; nnz];
        let mut ox = vec![0.0f64; nnz];
        for j in 0..n {
            let pj = perm[j] as usize;
            for q in z.p[j] as usize..z.p[j + 1] as usize {
                let slot = next[pj] as usize;
                oi[slot] = perm[z.i[q] as usize];
                ox[slot] = z.x[q];
                next[pj] += 1;
            }
        }
        /* Each column is a permuted set of rows and so is unsorted; scipy's
         * containers and every downstream `searchsorted` want it ascending. */
        let mut order: Vec<usize> = Vec::new();
        for j in 0..n {
            let (b, e) = (count[j] as usize, count[j + 1] as usize);
            order.clear();
            order.extend(b..e);
            order.sort_unstable_by_key(|&q| oi[q]);
            let rows: Vec<i64> = order.iter().map(|&q| oi[q]).collect();
            let vals: Vec<f64> = order.iter().map(|&q| ox[q]).collect();
            oi[b..e].copy_from_slice(&rows);
            ox[b..e].copy_from_slice(&vals);
        }

        Ok((
            count.into_pyarray(py).unbind(),
            oi.into_pyarray(py).unbind(),
            ox.into_pyarray(py).unbind(),
        ))
    }

    fn half_log_det(&self) -> PyResult<f64> {
        let n = self.n();
        if self.minor() < n {
            return Err(PyZeroDivisionError::new_err(format!(
                "matrix is not positive definite (leading minor {} is not)",
                self.minor() + 1
            )));
        }
        let mut s = 0.0f64;
        match &self.kind {
            Kind::Simplicial(k) => {
                let l = &k.l;
                for j in 0..n {
                    s += l.x[l.p[j] as usize].ln();
                }
            }
            Kind::Supernodal(k) => {
                let l = &k.l;
                /* the diagonal of supernode s is x [px[s] + jj*(nsrow+1)]:
                 * column jj of an nsrow-by-nscol column-major block */
                for t in 0..l.sym.nsuper {
                    let psx = l.sym.px[t] as usize;
                    let nsrow = (l.sym.pi[t + 1] - l.sym.pi[t]) as usize;
                    let nscol = (l.sym.sup[t + 1] - l.sym.sup[t]) as usize;
                    for jj in 0..nscol {
                        s += l.x[psx + jj * (nsrow + 1)].ln();
                    }
                }
            }
        }
        Ok(if self.is_ll() { s } else { 0.5 * s })
    }

    /// `L->Perm`: `L L' = P A P'`, i.e. `A[p][:, p]` is what was factorized.
    #[getter]
    fn perm(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        match &self.kind {
            Kind::Simplicial(k) => k.l.perm.clone(),
            Kind::Supernodal(k) => k.l.perm.clone(),
        }
        .into_pyarray(py)
        .unbind()
    }

    #[getter]
    fn n(&self) -> usize {
        match &self.kind {
            Kind::Simplicial(k) => k.l.n,
            Kind::Supernodal(k) => k.l.n,
        }
    }

    /// `L->ordering` — which method the trial loop actually selected, never
    /// `"best"`. Worth reading: it is the difference between a factor that
    /// fills in and one that does not.
    #[getter]
    fn ordering(&self) -> &'static str {
        ordering_name(match &self.kind {
            Kind::Simplicial(k) => k.l.ordering,
            Kind::Supernodal(k) => k.l.ordering,
        })
    }

    /// `L->minor`: `n` if `A` is positive definite, else the first column at
    /// which it was found not to be.
    #[getter]
    fn minor(&self) -> usize {
        match &self.kind {
            Kind::Simplicial(k) => k.l.minor,
            Kind::Supernodal(k) => k.l.minor,
        }
    }

    #[getter]
    fn is_ll(&self) -> bool {
        match &self.kind {
            Kind::Simplicial(k) => k.l.is_ll,
            /* the supernodal factorization is LL' and nothing else */
            Kind::Supernodal(_) => true,
        }
    }

    /// `L->is_super`.
    #[getter]
    fn is_super(&self) -> bool {
        matches!(self.kind, Kind::Supernodal(_))
    }

    /// `Common->rowfacfl` from the last factorization. Zero on the supernodal
    /// path, which does not keep a flop count.
    #[getter]
    fn rowfacfl(&self) -> f64 {
        self.fl
    }

    /// Entries in `L` — the same number either path took, and the same as
    /// `len(factor_csc()[2])`.
    ///
    /// For the supernodal factor that is `Σ ColCount [j]` from the analysis,
    /// **not** `L->xsize`: the dense blocks also hold the entries relaxed
    /// amalgamation added, which are not in `L` and which `factor_csc` prunes.
    /// See `xsize` for what is actually allocated.
    #[getter]
    fn nnz(&self) -> i64 {
        match &self.kind {
            Kind::Simplicial(k) => k.l.nz.iter().sum(),
            Kind::Supernodal(k) => k.l.colcount.iter().sum(),
        }
    }

    /// Doubles held by `L`: `L->xsize` for a supernodal factor, `L->nzmax` for
    /// a simplicial one. Larger than [`Self::nnz`] on the supernodal path —
    /// that gap is the price of the dense blocks, and it is reported rather
    /// than smoothed over.
    #[getter]
    fn xsize(&self) -> i64 {
        match &self.kind {
            Kind::Simplicial(k) => k.l.nzmax() as i64,
            Kind::Supernodal(k) => k.l.sym.xsize as i64,
        }
    }
}
