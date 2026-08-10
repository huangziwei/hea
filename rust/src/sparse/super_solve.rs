//! Solving with a supernodal `LL'` factor — the dense-block counterpart of
//! [`super::solve`].
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `CHOLMOD/Supernodal/t_cholmod_super_solve_worker.c` → [`lsolve`],
//!     [`ltsolve`]
//!   * `CHOLMOD/Supernodal/cholmod_super_solve.c`          → their argument
//!     checks
//!   * `CHOLMOD/Cholesky/cholmod_solve.c:656-724`          → [`super_solve`]
//!
//! **Scope.** `CHOLMOD_REAL` + `CHOLMOD_DOUBLE`, i.e. upstream's
//! `rd_cholmod_super_{l,lt}solve_worker`; the other three instantiations are
//! not built, as elsewhere in this module. A sparse right-hand side (`Bset`)
//! is out of scope here for the same reason it is in [`super::solve`] — and
//! upstream would not use these workers for it anyway, since it converts a
//! supernodal `L` to simplicial first (`cholmod_solve.c:316-330`).
//!
//! **The layout is not the simplicial one.** [`super::solve`] hands its kernels
//! a *transposed* block, `nrhs`-by-`n`, because that is what
//! `t_cholmod_lsolve_template.c` wants. These workers take `X` as it comes:
//! `n`-by-`nrhs`, column-major, leading dimension `d`. So the permutation
//! applies through `perm`/`iperm` rather than `ptrans`/`iptrans`, and there is
//! no blocking over columns — the whole right-hand side goes through at once.
//!
//! **Memory contract.** `Y` and `E` are [`SuperSolveWork`], owned by the
//! caller and grown never shrunk, for the reason [`super::solve`] gives:
//! `cholmod_solve2` takes them as handles (`cholmod_solve.c:669-679`) so that
//! a caller issuing many solves allocates once.
//!
//! **`E` is the gather buffer.** A supernode's rows below its diagonal block
//! are scattered through `X`, so the off-diagonal update cannot be a strided
//! block operation. Upstream gathers those `nsrow2` rows into a dense
//! `nsrow2`-by-`nrhs` workspace `E`, updates it with one `gemv`/`gemm`, and —
//! in the forward solve — scatters it back. `ltsolve` only gathers: it reads
//! rows that are already solved and writes only its own columns.

use super::dense::{gemm_nn, gemm_tn, gemv_n, gemv_t, trsm_lln, trsm_llt, trsv_ln, trsv_lt};
use super::solve::{iperm, perm, SolveError, Sys};
use super::super_numeric::SuperFactor;
use super::symbolic::Ordering;
use super::ws::Ws;

/// The two workspaces `cholmod_solve2` keeps across calls for a supernodal
/// factor: `Y`, which is `n`-by-`nrhs`, and `E`, which is
/// `nrhs`-by-`L->maxesize` (`cholmod_solve.c:669-679`).
///
/// Both are the caller's for the reason [`super::solve::SolveWork`] gives:
/// rebuilding them per call gives the cost back on every one of `gmm`'s 1486
/// solves per fit.
#[derive(Debug, Default)]
pub struct SuperSolveWork {
    y: Vec<f64>,
    e: Vec<f64>,
}

impl SuperSolveWork {
    pub fn new() -> SuperSolveWork {
        SuperSolveWork::default()
    }

    /// `cholmod_ensure_dense` — grow to hold both, never shrink.
    fn ensure(&mut self, ylen: usize, elen: usize) -> (&mut [f64], &mut [f64]) {
        if self.y.len() < ylen {
            self.y.resize(ylen, 0.0);
        }
        if self.e.len() < elen {
            self.e.resize(elen, 0.0);
        }
        (&mut self.y[..ylen], &mut self.e[..elen])
    }
}

/// `t_cholmod_super_solve_worker.c:20-303` (`cholmod_super_lsolve_worker`) — `X := L \ X`.
///
/// `x` is `n`-by-`nrhs` column-major with leading dimension `d`, overwritten in
/// place. `e` is scratch of at least `nrhs * maxesize` doubles; its contents
/// are undefined on entry and on exit.
///
/// The two `nrhs` branches are upstream's and are kept: they are not one
/// routine and its unrolling, they are `trsv`/`gemv` against `trsm`/`gemm`,
/// which sum the same solve in different orders. See [`super::dense`].
pub fn lsolve(l: &SuperFactor, x: &mut [f64], nrhs: usize, d: usize, e: &mut [f64]) {
    let sym = &l.sym;
    let (sup, pi, px, ls) = (&sym.sup, &sym.pi, &sym.px, &sym.s);
    let lx = &l.x;

    for s in 0..sym.nsuper {
        let k1 = sup[s] as usize;
        let psi = pi[s] as usize;
        let psx = px[s] as usize;
        let nsrow = pi[s + 1] as usize - psi;
        let nscol = sup[s + 1] as usize - k1;
        let nsrow2 = nsrow - nscol;
        let ps2 = psi + nscol;

        /* L1 is nscol-by-nscol lower triangular with a non-unit diagonal and
         * L2 is nsrow2-by-nscol, both with leading dimension nsrow; x1 is the
         * nscol rows of X this supernode owns; E is nsrow2-by-nrhs. */
        if nrhs == 1 {
            gather1(ls, ps2, nsrow2, x, e);
            trsv_ln(nscol, &lx[psx..], nsrow, &mut x[k1..]);
            if nsrow2 > 0 {
                /* E = E - L2*x1 */
                gemv_n(
                    nsrow2,
                    nscol,
                    &lx[psx + nscol..],
                    nsrow,
                    &x[k1..],
                    &mut e[..nsrow2],
                );
            }
            scatter1(ls, ps2, nsrow2, e, x);
        } else {
            gather(ls, ps2, nsrow2, nrhs, d, x, e);
            trsm_lln(nscol, nrhs, &lx[psx..], nsrow, &mut x[k1..], d);
            if nsrow2 > 0 {
                /* E = E - L2*x1 */
                gemm_nn(
                    nsrow2,
                    nrhs,
                    nscol,
                    &lx[psx + nscol..],
                    nsrow,
                    &x[k1..],
                    d,
                    &mut e[..nsrow2 * nrhs],
                    nsrow2,
                );
            }
            scatter(ls, ps2, nsrow2, nrhs, d, e, x);
        }
    }
}

/// `t_cholmod_super_solve_worker.c:309-580` (`cholmod_super_ltsolve_worker`) — `X := L' \ X`.
///
/// The back substitution: supernodes in reverse, and each one's off-diagonal
/// rows are *read* from `X` rather than written to it, so there is no scatter
/// step.
pub fn ltsolve(l: &SuperFactor, x: &mut [f64], nrhs: usize, d: usize, e: &mut [f64]) {
    let sym = &l.sym;
    let (sup, pi, px, ls) = (&sym.sup, &sym.pi, &sym.px, &sym.s);
    let lx = &l.x;

    for s in (0..sym.nsuper).rev() {
        let k1 = sup[s] as usize;
        let psi = pi[s] as usize;
        let psx = px[s] as usize;
        let nsrow = pi[s + 1] as usize - psi;
        let nscol = sup[s + 1] as usize - k1;
        let nsrow2 = nsrow - nscol;
        let ps2 = psi + nscol;

        if nrhs == 1 {
            gather1(ls, ps2, nsrow2, x, e);
            if nsrow2 > 0 {
                /* x1 = x1 - L2'*E */
                gemv_t(
                    nsrow2,
                    nscol,
                    &lx[psx + nscol..],
                    nsrow,
                    &e[..nsrow2],
                    &mut x[k1..],
                );
            }
            trsv_lt(nscol, &lx[psx..], nsrow, &mut x[k1..]);
        } else {
            gather(ls, ps2, nsrow2, nrhs, d, x, e);
            if nsrow2 > 0 {
                /* x1 = x1 - L2'*E */
                gemm_tn(
                    nscol,
                    nrhs,
                    nsrow2,
                    &lx[psx + nscol..],
                    nsrow,
                    &e[..nsrow2 * nrhs],
                    nsrow2,
                    &mut x[k1..],
                    d,
                );
            }
            trsm_llt(nscol, nrhs, &lx[psx..], nsrow, &mut x[k1..], d);
        }
    }
}

/// `Ex [ii] = Xx [Ls [ps2 + ii]]`, one right-hand side.
#[inline]
fn gather1(ls: &[i64], ps2: usize, nsrow2: usize, x: &[f64], e: &mut [f64]) {
    let (ls, x) = (Ws::new_ref(ls), Ws::new_ref(x));
    let e = Ws::new(e);
    for ii in 0..nsrow2 {
        e[ii] = x[ls[ps2 + ii]];
    }
}

/// `Xx [Ls [ps2 + ii]] = Ex [ii]`, one right-hand side.
#[inline]
fn scatter1(ls: &[i64], ps2: usize, nsrow2: usize, e: &[f64], x: &mut [f64]) {
    let (ls, e) = (Ws::new_ref(ls), Ws::new_ref(e));
    let x = Ws::new(x);
    for ii in 0..nsrow2 {
        x[ls[ps2 + ii]] = e[ii];
    }
}

/// `Ex [ii + j*nsrow2] = Xx [i + j*d]`.
#[inline]
fn gather(ls: &[i64], ps2: usize, nsrow2: usize, nrhs: usize, d: usize, x: &[f64], e: &mut [f64]) {
    let (ls, x) = (Ws::new_ref(ls), Ws::new_ref(x));
    let e = Ws::new(e);
    for ii in 0..nsrow2 {
        let i = ls[ps2 + ii] as usize;
        for j in 0..nrhs {
            e[ii + j * nsrow2] = x[i + j * d];
        }
    }
}

/// `Xx [i + j*d] = Ex [ii + j*nsrow2]`.
#[inline]
fn scatter(ls: &[i64], ps2: usize, nsrow2: usize, nrhs: usize, d: usize, e: &[f64], x: &mut [f64]) {
    let (ls, e) = (Ws::new_ref(ls), Ws::new_ref(e));
    let x = Ws::new(x);
    for ii in 0..nsrow2 {
        let i = ls[ps2 + ii] as usize;
        for j in 0..nrhs {
            x[i + j * d] = e[ii + j * nsrow2];
        }
    }
}

/// `cholmod_solve.c:656-724` — the supernodal arm of `cholmod_solve2`.
///
/// `b` is `n`-by-`nrhs` column-major with leading dimension `n`, and `x`
/// receives the solution in the same layout; they may not alias.
///
/// `sys` values that name `D` are the identity here, because a supernodal
/// factor is always `LL'`: upstream reaches them through the same branch and
/// none of its three `if`s fire, leaving `X = P' P B`.
pub fn super_solve(
    sys: Sys,
    l: &SuperFactor,
    b: &[f64],
    nrhs: usize,
    x: &mut [f64],
    work: &mut SuperSolveWork,
) -> Result<(), SolveError> {
    let n = l.n;
    if !l.numeric {
        return Err(SolveError::Invalid("L is symbolic: factorize it first"));
    }
    if b.len() < n * nrhs || x.len() < n * nrhs {
        return Err(SolveError::Invalid("dimensions of L and B do not match"));
    }
    if nrhs == 0 || n == 0 {
        return Ok(());
    }

    /* cholmod_solve.c:262-267 — Perm is the identity for every system that does
     * not apply it, and for a factor that was not permuted */
    let perm_ = match sys {
        Sys::P | Sys::Pt | Sys::A if l.ordering != Ordering::Natural => Some(&l.perm[..]),
        _ => None,
    };

    match sys {
        Sys::P => perm(b, perm_, 0, nrhs, n, nrhs, x),
        Sys::Pt => iperm(b, perm_, 0, nrhs, n, nrhs, x),
        _ => {
            let (y, e) = work.ensure(n * nrhs, nrhs * l.sym.maxesize);
            perm(b, perm_, 0, nrhs, n, nrhs, y); /* Y = P*B */
            match sys {
                Sys::A | Sys::LDLt => {
                    lsolve(l, y, nrhs, n, e);
                    ltsolve(l, y, nrhs, n, e);
                }
                Sys::L | Sys::LD => lsolve(l, y, nrhs, n, e),
                Sys::Lt | Sys::DLt => ltsolve(l, y, nrhs, n, e),
                /* D on an LL' factor is the identity */
                _ => {}
            }
            iperm(y, perm_, 0, nrhs, n, nrhs, x); /* X = P'*Y */
        }
    }
    Ok(())
}

/* ========================================================================= */
/* === tests =============================================================== */
/* ========================================================================= */

#[cfg(test)]
mod tests {
    use super::super::amd::IntWidth;
    use super::super::super_numeric::{super_factorize, SuperWork};
    use super::super::super_symbolic::{super_symbolic, Relax};
    use super::super::symbolic::{analyze_sparse, permute_sym, Method, Sparse};
    use super::super::testcorpus::{corpus, spd_triangle};
    use super::super::ws::{columns_are_sorted, Work};
    use super::*;

    /// A corpus matrix, factorized supernodally the way `mod.rs` does it, with
    /// the triangle it was built from kept for the residual check.
    fn factor(
        n: usize,
        edges: &[(usize, usize)],
        ordering: Ordering,
    ) -> (SuperFactor, Sparse<'static>) {
        let (p, i, v) = spd_triangle(n, edges, false);
        let a = Sparse {
            n,
            p: p.clone().into(),
            i: i.clone().into(),
            x: v.clone().into(),
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(n, &p, &i),
        };
        let mut w = Work::new(n);
        let s = analyze_sparse(&a, Method::Pinned(ordering), IntWidth::I64, &mut w).unwrap();
        let a2 = permute_sym(&a, s.ordering, &s.perm, false, false, &mut w.all());
        let sym = super_symbolic(
            a2.as_ref().unwrap_or(&a),
            &s.parent,
            &s.colcount,
            &Relax::default(),
            &mut w,
        )
        .unwrap();
        let mut l = SuperFactor::new(s, sym);
        let mut cw = SuperWork::new();
        super_factorize(&a, 0.0, &mut l, &mut w, &mut cw).unwrap();
        (l, a)
    }

    /// `A * x` from the stored upper triangle, mirrored across the diagonal.
    fn matvec(a: &Sparse, x: &[f64], nrhs: usize) -> Vec<f64> {
        let n = a.n;
        let mut out = vec![0.0; n * nrhs];
        for j in 0..n {
            for q in a.p[j] as usize..a.p[j + 1] as usize {
                let r = a.i[q] as usize;
                for c in 0..nrhs {
                    out[j + c * n] += a.x[q] * x[r + c * n];
                    if r != j {
                        out[r + c * n] += a.x[q] * x[j + c * n];
                    }
                }
            }
        }
        out
    }

    /// The whole corpus, both orderings, `nrhs = 1..=5` — which is what puts
    /// both the vector kernels and the matrix kernels under the live `Ws` bound
    /// check, since upstream splits on `nrhs == 1`.
    #[test]
    fn the_solve_reproduces_the_right_hand_side() {
        for (name, n, edges) in corpus() {
            for ordering in [Ordering::Amd, Ordering::Natural] {
                let (l, a) = factor(n, &edges, ordering);
                if l.sym.nsuper == 0 {
                    continue;
                }
                let mut work = SuperSolveWork::new();
                for nrhs in 1..=5usize {
                    let b: Vec<f64> = (0..n * nrhs)
                        .map(|k| ((k * 37 % 19) as f64 - 9.0) / 4.0)
                        .collect();
                    let mut x = vec![0.0; n * nrhs];
                    super_solve(Sys::A, &l, &b, nrhs, &mut x, &mut work).unwrap();
                    let r = matvec(&a, &x, nrhs);
                    for k in 0..n * nrhs {
                        assert!(
                            (r[k] - b[k]).abs() < 1e-8,
                            "{name} {ordering:?} nrhs={nrhs} entry {k}: {} vs {}",
                            r[k],
                            b[k]
                        );
                    }
                }
            }
        }
    }

    /// `L` then `L'` is the same as `A` in one call, entry for entry — the two
    /// halves compose exactly, since `A` is those two calls and nothing else.
    #[test]
    fn the_two_halves_compose_into_the_whole_solve() {
        for (name, n, edges) in corpus() {
            let (l, _) = factor(n, &edges, Ordering::Amd);
            if l.sym.nsuper == 0 {
                continue;
            }
            let mut work = SuperSolveWork::new();
            for nrhs in [1usize, 3] {
                let b: Vec<f64> = (0..n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
                let mut whole = vec![0.0; n * nrhs];
                super_solve(Sys::A, &l, &b, nrhs, &mut whole, &mut work).unwrap();

                /* P, then L, then L', then P' — the pieces cholmod_solve's
                 * supernodal arm is made of */
                let mut t = vec![0.0; n * nrhs];
                super_solve(Sys::P, &l, &b, nrhs, &mut t, &mut work).unwrap();
                let mut u = vec![0.0; n * nrhs];
                super_solve(Sys::L, &l, &t, nrhs, &mut u, &mut work).unwrap();
                super_solve(Sys::Lt, &l, &u, nrhs, &mut t, &mut work).unwrap();
                super_solve(Sys::Pt, &l, &t, nrhs, &mut u, &mut work).unwrap();

                assert_eq!(whole, u, "{name} nrhs={nrhs}");
            }
        }
    }

    /// `D` is the identity on a supernodal factor, which is `LL'` by
    /// construction — the permutation still applies and cancels.
    #[test]
    fn d_is_the_identity_for_a_supernodal_factor() {
        let c = corpus();
        let (_, n, edges) = &c[3];
        let (l, _) = factor(*n, edges, Ordering::Amd);
        let n = *n;
        let b: Vec<f64> = (0..n).map(|k| (k % 5) as f64).collect();
        let mut x = vec![0.0; n];
        let mut work = SuperSolveWork::new();
        super_solve(Sys::D, &l, &b, 1, &mut x, &mut work).unwrap();
        assert_eq!(x, b);
    }

    #[test]
    fn a_symbolic_factor_cannot_be_solved_with() {
        let c = corpus();
        let (_, n, edges) = &c[3];
        let (mut l, _) = factor(*n, edges, Ordering::Amd);
        l.numeric = false;
        let mut work = SuperSolveWork::new();
        let e = super_solve(Sys::A, &l, &[0.0; 1], 1, &mut [0.0; 1], &mut work);
        assert!(matches!(e, Err(SolveError::Invalid(_))));
    }
}
