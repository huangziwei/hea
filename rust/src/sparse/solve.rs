//! Solving with the factor — `cholmod_solve`'s dispatch and the simplicial
//! triangular kernels.
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `CHOLMOD/Cholesky/cholmod_solve.c`               → [`solve`] (its `solve2` body)
//!   * `CHOLMOD/Cholesky/t_cholmod_solve_worker.c`      → [`simplicial_solver`], [`dsolve`]
//!   * `CHOLMOD/Cholesky/t_cholmod_lsolve_template.c`   → [`lsolve`]
//!   * `CHOLMOD/Cholesky/t_cholmod_ltsolve_template.c`  → [`ltsolve`]
//!   * `CHOLMOD/Cholesky/t_cholmod_psolve_worker.c`     → [`perm`], [`iperm`], [`ptrans`], [`iptrans`]
//!
//! **Scope.** `CHOLMOD_REAL` + `CHOLMOD_DOUBLE`, simplicial `LL'` or `LDL'`,
//! and `Bset == NULL`. Upstream compiles the two solve templates six times
//! (`{real, complex, zomplex} x {double, single}`, `cholmod_solve.c:70-87`),
//! each of those three times over (`LL` / `LD` / unit diagonal,
//! `t_cholmod_solve_worker.c:20-40`); this builds the three real-double forms,
//! the same way [`super::numeric`] builds one of the twelve `rowfac`
//! instantiations. The supernodal branch (`cholmod_solve.c:656-724`) belongs
//! with the supernodal factorization and is not here. `Bset` reaches only
//! `cholmod_solve2`, which no consumer calls, and it is the only thing that
//! makes `Yset` non-`NULL` for a real factor — so the `Yset` path in both
//! templates (`t_cholmod_lsolve_template.c:800-841`) is unreachable here and is
//! not ported.
//!
//! **`Y` is the transpose of the right-hand side.** `cholmod_solve` blocks the
//! columns of `B` four at a time and hands the kernels `Y`, an `nk`-by-`n`
//! array with `nk` in `1..=4` — so consecutive right-hand sides are *adjacent*
//! in memory and the scatter `X [Li [p]] -= ...` touches one cache line for all
//! of them. `nk` is a compile-time constant in the C, which instantiates each
//! kernel four times; here it is a `const` generic, for the reason the port's
//! pre-flight gives: an instantiation structure is part of the source, not an
//! implementation detail, and the four bodies also differ in *summation order*
//! (`t_cholmod_lsolve_template.c:126,164` accumulate 2 and 3 products before
//! subtracting), so collapsing them would move results.

use super::numeric::{mulsub, Factor};
use super::ws::Ws;
use crate::nmath::util::rfma;

/// The system to solve — `cholmod.h`'s `CHOLMOD_A` … `CHOLMOD_Pt`
/// (`cholmod_solve.c:14-22`). `D` is the identity for an `LL'` factor, so
/// `LD`/`L` and `DLt`/`Lt` name the same solve there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sys {
    /// `x = P' * (L' \ (D \ (L \ (P * b))))`
    A,
    /// `x = L' \ (D \ (L \ b))`
    LDLt,
    /// `x = D \ (L \ b)`
    LD,
    /// `x = L' \ (D \ b)`
    DLt,
    /// `x = L \ b`
    L,
    /// `x = L' \ b`
    Lt,
    /// `x = D \ b`
    D,
    /// `x = P * b`
    P,
    /// `x = P' * b`
    Pt,
}

/// Why a solve could not be performed.
#[derive(Debug)]
pub enum SolveError {
    /// `B` and `L` disagree on `n`, or `L` carries no numeric values.
    Invalid(&'static str),
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SolveError::Invalid(m) => write!(f, "{m}"),
        }
    }
}

/* ========================================================================= */
/* === the three forms each template is compiled in ======================== */
/* ========================================================================= */

/// `LL'`: non-unit diagonal (`#define LL`).
const LL: u8 = 0;
/// `LDL'`: fold `D` into this half of the solve (`#define LD`).
const LD: u8 = 1;
/// `LDL'`: unit diagonal, `D` handled elsewhere (neither macro defined).
const UNIT: u8 = 2;

/// `switch (Y->nrow) { case 1: ... case 4: }`
/// (`t_cholmod_lsolve_template.c:789-795`), with the rank a `const` generic.
macro_rules! by_rank {
    ($f:ident, $form:expr, $l:expr, $x:expr, $nk:expr) => {
        match $nk {
            1 => $f::<1, { $form }>($l, $x),
            2 => $f::<2, { $form }>($l, $x),
            3 => $f::<3, { $form }>($l, $x),
            _ => $f::<4, { $form }>($l, $x),
        }
    };
}

/* ========================================================================= */
/* === Lx=b, LDx=b ========================================================= */
/* ========================================================================= */

/// `t_cholmod_lsolve_template.c` — solve `Lx=b` with unit or non-unit
/// diagonal, or `LDx=b`.
///
/// `x` holds `b` on input and the solution on output, `NRHS`-by-`n` in row
/// form: entry `i` of right-hand side `c` is `x [i*NRHS + c]`.
///
/// The loop advances one, two or three columns at a time. The two- and
/// three-column branches are not an optimization of the one-column branch:
/// they fire when consecutive columns of `L` form a dense chain
/// (`lnz == Lnz[j+1] + 1 && Li[p+1] == j+1`), and they accumulate two or three
/// products before subtracting, which is a different rounding from doing them
/// one at a time.
fn lsolve<const NRHS: usize, const FORM: u8>(l: &Factor, x: &mut [f64]) {
    let (lx, li) = (Ws::new_ref(&l.x), Ws::new_ref(&l.i));
    let (lp, lnz) = (Ws::new_ref(&l.p), Ws::new_ref(&l.nz));
    let n = l.n;
    let x = Ws::new(x);

    let mut j = 0usize;
    while j < n {
        /* get the start, end, and length of column j */
        let p = lp[j];
        let nzj = lnz[j];
        let pend = p + nzj;

        /* find a chain of supernodes (up to j, j+1, and j+2) */
        if nzj < 4 || nzj != lnz[j + 1] + 1 || li[p + 1] != (j + 1) as i64 {
            /* ---- solve with a single column of L ---- */
            let mut y = [0.0f64; NRHS];
            for c in 0..NRHS {
                y[c] = x[j * NRHS + c];
                match FORM {
                    LL => {
                        y[c] /= lx[p];
                        x[j * NRHS + c] = y[c];
                    }
                    LD => x[j * NRHS + c] = y[c] / lx[p],
                    _ => {}
                }
            }
            for q in p + 1..pend {
                let i = li[q] as usize;
                for c in 0..NRHS {
                    x[i * NRHS + c] = mulsub(x[i * NRHS + c], lx[q], y[c]);
                }
            }
            j += 1;
        } else if nzj != lnz[j + 2] + 2 || li[p + 2] != (j + 2) as i64 {
            /* ---- solve with a supernode of two columns of L ---- */
            let mut y = [[0.0f64; NRHS]; 2];
            let q0 = lp[j + 1];
            for c in 0..NRHS {
                y[0][c] = x[j * NRHS + c];
                match FORM {
                    LL => {
                        y[0][c] /= lx[p];
                        y[1][c] = mulsub(x[(j + 1) * NRHS + c], lx[p + 1], y[0][c]) / lx[q0];
                        x[j * NRHS + c] = y[0][c];
                        x[(j + 1) * NRHS + c] = y[1][c];
                    }
                    LD => {
                        y[1][c] = mulsub(x[(j + 1) * NRHS + c], lx[p + 1], y[0][c]);
                        x[j * NRHS + c] = y[0][c] / lx[p];
                        x[(j + 1) * NRHS + c] = y[1][c] / lx[q0];
                    }
                    _ => {
                        y[1][c] = mulsub(x[(j + 1) * NRHS + c], lx[p + 1], y[0][c]);
                        x[(j + 1) * NRHS + c] = y[1][c];
                    }
                }
            }
            let (mut p, mut q) = (p + 2, q0 + 1);
            while p < pend {
                let i = li[p] as usize;
                let (l0, l1) = (lx[p], lx[q]);
                for c in 0..NRHS {
                    x[i * NRHS + c] -= rfma(l0, y[0][c], l1 * y[1][c]);
                }
                p += 1;
                q += 1;
            }
            j += 2;
        } else {
            /* ---- solve with a supernode of three columns of L ---- */
            let mut y = [[0.0f64; NRHS]; 3];
            let q0 = lp[j + 1];
            let r0 = lp[j + 2];
            for c in 0..NRHS {
                y[0][c] = x[j * NRHS + c];
                let (x1, x2) = (x[(j + 1) * NRHS + c], x[(j + 2) * NRHS + c]);
                match FORM {
                    LL => {
                        y[0][c] /= lx[p];
                        y[1][c] = mulsub(x1, lx[p + 1], y[0][c]) / lx[q0];
                        y[2][c] =
                            mulsub(mulsub(x2, lx[p + 2], y[0][c]), lx[q0 + 1], y[1][c]) / lx[r0];
                        x[j * NRHS + c] = y[0][c];
                        x[(j + 1) * NRHS + c] = y[1][c];
                        x[(j + 2) * NRHS + c] = y[2][c];
                    }
                    LD => {
                        y[1][c] = mulsub(x1, lx[p + 1], y[0][c]);
                        y[2][c] = mulsub(mulsub(x2, lx[p + 2], y[0][c]), lx[q0 + 1], y[1][c]);
                        x[j * NRHS + c] = y[0][c] / lx[p];
                        x[(j + 1) * NRHS + c] = y[1][c] / lx[q0];
                        x[(j + 2) * NRHS + c] = y[2][c] / lx[r0];
                    }
                    _ => {
                        y[1][c] = mulsub(x1, lx[p + 1], y[0][c]);
                        y[2][c] = mulsub(mulsub(x2, lx[p + 2], y[0][c]), lx[q0 + 1], y[1][c]);
                        x[(j + 1) * NRHS + c] = y[1][c];
                        x[(j + 2) * NRHS + c] = y[2][c];
                    }
                }
            }
            let (mut p, mut q, mut r) = (p + 3, q0 + 2, r0 + 1);
            while p < pend {
                let i = li[p] as usize;
                let (l0, l1, l2) = (lx[p], lx[q], lx[r]);
                for c in 0..NRHS {
                    x[i * NRHS + c] -= rfma(l2, y[2][c], rfma(l0, y[0][c], l1 * y[1][c]));
                }
                p += 1;
                q += 1;
                r += 1;
            }
            j += 3;
        }
    }
}

/* ========================================================================= */
/* === L'x=b, DL'x=b ======================================================= */
/* ========================================================================= */

/// `t_cholmod_ltsolve_template.c` — solve `L'x=b` with unit or non-unit
/// diagonal, or `DL'x=b`. The back-substitution counterpart of [`lsolve`],
/// walking the columns of `L` in reverse and gathering rather than scattering.
///
/// **The four ranks are not the same routine four times.** `LSOLVE(4)` has no
/// three-column branch: its third condition is commented out (`:656`) and the
/// `else` takes everything, so a four-right-hand-side back-solve walks a
/// three-column chain as 2+1 where the other three ranks walk it as 3. That
/// changes which products are summed together, so it changes the answer in the
/// last bit — this port gives all four kernels one body, and `NRHS == 4` is
/// what keeps that body from being a *different* one. `lsolve` has no such
/// asymmetry; all four of its ranks branch three ways.
fn ltsolve<const NRHS: usize, const FORM: u8>(l: &Factor, x: &mut [f64]) {
    let (lx, li) = (Ws::new_ref(&l.x), Ws::new_ref(&l.i));
    let (lp, lnz) = (Ws::new_ref(&l.p), Ws::new_ref(&l.nz));
    let n = l.n as i64;
    let x = Ws::new(x);

    let mut j = n - 1;
    while j >= 0 {
        /* get the start, end, and length of column j */
        let p = lp[j];
        let nzj = lnz[j];
        let pend = p + nzj;

        /* find a chain of supernodes (up to j, j-1, and j-2) */
        if j < 4 || nzj != lnz[j - 1] - 1 || li[lp[j - 1] + 1] != j {
            /* ---- solve with a single column of L ---- */
            let mut y = [0.0f64; NRHS];
            let d = lx[p];
            for c in 0..NRHS {
                y[c] = x[j * NRHS as i64 + c as i64];
                if FORM == LD {
                    y[c] /= d;
                }
            }
            for q in p + 1..pend {
                let i = li[q];
                for c in 0..NRHS {
                    y[c] = mulsub(y[c], lx[q], x[i * NRHS as i64 + c as i64]);
                }
            }
            for c in 0..NRHS {
                x[j * NRHS as i64 + c as i64] = if FORM == LL { y[c] / d } else { y[c] };
            }
            j -= 1;
        } else if NRHS == 4 || nzj != lnz[j - 2] - 2 || li[lp[j - 2] + 2] != j {
            /* ---- solve with a supernode of two columns of L ----
             * `NRHS == 4` is `LSOLVE(4)`'s missing third branch (`:656`), not
             * a shortcut: rank 4 takes this arm unconditionally upstream. */
            let mut y = [[0.0f64; NRHS]; 2];
            let q0 = lp[j - 1];
            let d = [lx[p], lx[q0]];
            let t = lx[q0 + 1];
            for c in 0..NRHS {
                let (x0, x1) = (
                    x[j * NRHS as i64 + c as i64],
                    x[(j - 1) * NRHS as i64 + c as i64],
                );
                if FORM == LD {
                    y[0][c] = x0 / d[0];
                    y[1][c] = x1 / d[1];
                } else {
                    y[0][c] = x0;
                    y[1][c] = x1;
                }
            }
            let (mut p, mut q) = (p + 1, q0 + 2);
            while p < pend {
                let i = li[p];
                let (l0, l1) = (lx[p], lx[q]);
                for c in 0..NRHS {
                    let xi = x[i * NRHS as i64 + c as i64];
                    y[0][c] = mulsub(y[0][c], l0, xi);
                    y[1][c] = mulsub(y[1][c], l1, xi);
                }
                p += 1;
                q += 1;
            }
            for c in 0..NRHS {
                if FORM == LL {
                    y[0][c] /= d[0];
                    y[1][c] = mulsub(y[1][c], t, y[0][c]) / d[1];
                } else {
                    y[1][c] = mulsub(y[1][c], t, y[0][c]);
                }
                x[j * NRHS as i64 + c as i64] = y[0][c];
                x[(j - 1) * NRHS as i64 + c as i64] = y[1][c];
            }
            j -= 2;
        } else {
            /* ---- solve with a supernode of three columns of L ---- */
            let mut y = [[0.0f64; NRHS]; 3];
            let q0 = lp[j - 1];
            let r0 = lp[j - 2];
            let d = [lx[p], lx[q0], lx[r0]];
            let t = [lx[q0 + 1], lx[r0 + 1], lx[r0 + 2]];
            for c in 0..NRHS {
                let (x0, x1, x2) = (
                    x[j * NRHS as i64 + c as i64],
                    x[(j - 1) * NRHS as i64 + c as i64],
                    x[(j - 2) * NRHS as i64 + c as i64],
                );
                if FORM == LD {
                    y[0][c] = x0 / d[0];
                    y[1][c] = x1 / d[1];
                    y[2][c] = x2 / d[2];
                } else {
                    y[0][c] = x0;
                    y[1][c] = x1;
                    y[2][c] = x2;
                }
            }
            let (mut p, mut q, mut r) = (p + 1, q0 + 2, r0 + 3);
            while p < pend {
                let i = li[p];
                let (l0, l1, l2) = (lx[p], lx[q], lx[r]);
                for c in 0..NRHS {
                    let xi = x[i * NRHS as i64 + c as i64];
                    y[0][c] = mulsub(y[0][c], l0, xi);
                    y[1][c] = mulsub(y[1][c], l1, xi);
                    y[2][c] = mulsub(y[2][c], l2, xi);
                }
                p += 1;
                q += 1;
                r += 1;
            }
            for c in 0..NRHS {
                if FORM == LL {
                    y[0][c] /= d[0];
                    y[1][c] = mulsub(y[1][c], t[0], y[0][c]) / d[1];
                    y[2][c] = mulsub(mulsub(y[2][c], t[2], y[0][c]), t[1], y[1][c]) / d[2];
                } else {
                    y[1][c] = mulsub(y[1][c], t[0], y[0][c]);
                    y[2][c] -= rfma(t[2], y[0][c], t[1] * y[1][c]);
                }
                x[(j - 2) * NRHS as i64 + c as i64] = y[2][c];
                x[(j - 1) * NRHS as i64 + c as i64] = y[1][c];
                x[j * NRHS as i64 + c as i64] = y[0][c];
            }
            j -= 3;
        }
    }
}

/* ========================================================================= */
/* === Dx=b ================================================================ */
/* ========================================================================= */

/// `t_cholmod_solve_worker.c:52-109` — solve `Dx=b` for an `LDL'` factor,
/// where `D(k,k)` is `L(k,k)` and the unit diagonal of `L` is implicit.
fn dsolve(l: &Factor, x: &mut [f64], nrhs: usize) {
    let lx = Ws::new_ref(&l.x);
    let lp = Ws::new_ref(&l.p);
    let x = Ws::new(x);
    for k in 0..l.n {
        let d = lx[lp[k]];
        for p in k * nrhs..(k + 1) * nrhs {
            x[p] /= d;
        }
    }
}

/* ========================================================================= */
/* === the simplicial dispatch ============================================= */
/* ========================================================================= */

/// `t_cholmod_solve_worker.c:124-187` — pick the kernels for `sys` and the
/// factor's form. `x` holds the (already permuted, already transposed)
/// right-hand side on input and the solution on output.
fn simplicial_solver(sys: Sys, l: &Factor, x: &mut [f64], nk: usize) {
    if l.is_ll {
        /* The factorization is LL' */
        match sys {
            Sys::A | Sys::LDLt => {
                by_rank!(lsolve, LL, l, x, nk);
                by_rank!(ltsolve, LL, l, x, nk);
            }
            Sys::L | Sys::LD => by_rank!(lsolve, LL, l, x, nk),
            Sys::Lt | Sys::DLt => by_rank!(ltsolve, LL, l, x, nk),
            /* Dx=b returns silently for an LL' factor: D is the identity */
            _ => {}
        }
    } else {
        /* The factorization is LDL' */
        match sys {
            Sys::A | Sys::LDLt => {
                by_rank!(lsolve, UNIT, l, x, nk);
                by_rank!(ltsolve, LD, l, x, nk);
            }
            Sys::LD => by_rank!(lsolve, LD, l, x, nk),
            Sys::L => by_rank!(lsolve, UNIT, l, x, nk),
            Sys::Lt => by_rank!(ltsolve, UNIT, l, x, nk),
            Sys::DLt => by_rank!(ltsolve, LD, l, x, nk),
            Sys::D => dsolve(l, x, nk),
            _ => {}
        }
    }
}

/* ========================================================================= */
/* === the permutation apply =============================================== */
/* ========================================================================= */

/// `t_cholmod_psolve_worker.c:40-99` — `Y = B (P (1:n), k1:k2-1)`.
///
/// `perm` is `None` for the identity, which is what `cholmod_solve` passes
/// whenever `L->ordering == CHOLMOD_NATURAL` (`cholmod_solve.c:262-267`).
fn perm(
    b: &[f64],
    p: Option<&[i64]>,
    k1: usize,
    ncols: usize,
    n: usize,
    nrhs: usize,
    y: &mut [f64],
) {
    let k2 = (k1 + ncols).min(nrhs);
    let (b, y) = (Ws::new_ref(b), Ws::new(y));
    for j in k1..k2 {
        let (dj, j2) = (n * j, n * (j - k1));
        for k in 0..n {
            y[k + j2] = b[pk(p, k) + dj];
        }
    }
}

/// `t_cholmod_psolve_worker.c:251-305` — `X (P (1:n), k1:k2-1) = Y`.
fn iperm(
    y: &[f64],
    p: Option<&[i64]>,
    k1: usize,
    ncols: usize,
    n: usize,
    nrhs: usize,
    x: &mut [f64],
) {
    let k2 = (k1 + ncols).min(nrhs);
    let (y, x) = (Ws::new_ref(y), Ws::new(x));
    for j in k1..k2 {
        let (dj, j2) = (n * j, n * (j - k1));
        for k in 0..n {
            x[pk(p, k) + dj] = y[k + j2];
        }
    }
}

/// `t_cholmod_psolve_worker.c:443-503` — `Y = B (P (1:n), k1:k2-1)'`, the
/// array transpose that puts the block's right-hand sides next to each other.
/// Returns `nk`, the number of columns actually taken, which is what the
/// kernels see as `Y->nrow`.
fn ptrans(
    b: &[f64],
    p: Option<&[i64]>,
    k1: usize,
    ncols: usize,
    n: usize,
    nrhs: usize,
    y: &mut [f64],
) -> usize {
    let k2 = (k1 + ncols).min(nrhs);
    let nk = k2 - k1;
    let (b, y) = (Ws::new_ref(b), Ws::new(y));
    for j in k1..k2 {
        let (dj, j2) = (n * j, j - k1);
        for k in 0..n {
            y[j2 + k * nk] = b[pk(p, k) + dj];
        }
    }
    nk
}

/// `t_cholmod_psolve_worker.c:670-726` — `X (P (1:n), k1:k2-1) = Y'`.
fn iptrans(
    y: &[f64],
    p: Option<&[i64]>,
    k1: usize,
    ncols: usize,
    n: usize,
    nrhs: usize,
    x: &mut [f64],
) {
    let k2 = (k1 + ncols).min(nrhs);
    let nk = k2 - k1;
    let (y, x) = (Ws::new_ref(y), Ws::new(x));
    for j in k1..k2 {
        let (dj, j2) = (n * j, j - k1);
        for k in 0..n {
            x[pk(p, k) + dj] = y[j2 + k * nk];
        }
    }
}

/// `cholmod_solve.c:64` — `#define P(k) ((Perm == NULL) ? (k) : Perm [k])`.
#[inline(always)]
fn pk(p: Option<&[i64]>, k: usize) -> usize {
    match p {
        None => k,
        Some(p) => Ws::new_ref(p)[k] as usize,
    }
}

/* ========================================================================= */
/* === cholmod_solve ======================================================= */
/* ========================================================================= */

/// The workspace `cholmod_solve2` keeps across calls in `*Y_Handle`
/// (`cholmod_solve.c:757`), grown by `cholmod_ensure_dense`.
///
/// It belongs to the caller rather than to [`solve`] for the reason the port's
/// pre-flight gives about `Common->Iwork`: `gmm` issues 1486 solves per GLMM
/// fit, and a workspace rebuilt per call gives that cost back on every one.
#[derive(Default)]
pub struct SolveWork {
    y: Vec<f64>,
}

impl SolveWork {
    pub fn new() -> SolveWork {
        SolveWork::default()
    }

    /// `cholmod_ensure_dense` — grow to hold `nr`-by-`n`, never shrink.
    fn ensure(&mut self, len: usize) -> &mut [f64] {
        if self.y.len() < len {
            self.y.resize(len, 0.0);
        }
        &mut self.y[..len]
    }
}

/// `cholmod_solve2`'s body for a dense `B` and a simplicial `L`
/// (`cholmod_solve.c:182-830`).
///
/// `b` is `n`-by-`nrhs` in column-major order with leading dimension `n`, and
/// `x` receives the solution in the same layout. They may not alias.
pub fn solve(
    sys: Sys,
    l: &Factor,
    b: &[f64],
    nrhs: usize,
    x: &mut [f64],
    work: &mut SolveWork,
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

    /* Perm is NULL — the identity — for every system that does not apply it,
     * and for a factor that was not permuted (cholmod_solve.c:262-267) */
    let perm_ = match sys {
        Sys::P | Sys::Pt | Sys::A if l.ordering != super::symbolic::Ordering::Natural => {
            Some(&l.perm[..])
        }
        _ => None,
    };

    match sys {
        Sys::P => {
            /* x = P*b */
            perm(b, perm_, 0, nrhs, n, nrhs, x);
        }
        Sys::Pt => {
            /* x = P'*b */
            iperm(b, perm_, 0, nrhs, n, nrhs, x);
        }
        _ => {
            /* solve using a simplicial LL' or LDL' factorization, up to four
             * columns of B at a time (cholmod_solve.c:732-825) */
            const NCOLS: usize = 4;
            let nr = NCOLS.max(nrhs);
            let y = work.ensure(nr * n);
            for k1 in (0..nrhs).step_by(NCOLS) {
                let nk = ptrans(b, perm_, k1, NCOLS, n, nrhs, y);
                simplicial_solver(sys, l, &mut y[..nk * n], nk);
                iptrans(y, perm_, k1, NCOLS, n, nrhs, x);
            }
        }
    }
    Ok(())
}

/* ========================================================================= */
/* === tests =============================================================== */
/* ========================================================================= */

#[cfg(test)]
mod tests {
    use super::super::symbolic::{analyze_sparse, Ordering, Sparse};
    use super::super::testcorpus::spd_triangle;
    use super::super::ws::{columns_are_sorted, Work};
    use super::super::{numeric, testcorpus};
    use super::*;

    /// The factor of a corpus matrix, as `factorize` leaves it.
    fn factor(n: usize, edges: &[(usize, usize)], ll: bool) -> Factor {
        let (p, i, v) = spd_triangle(n, edges, false);
        let a = Sparse {
            n,
            p: p.clone(),
            i: i.clone(),
            x: v.clone(),
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(n, &p, &i),
        };
        let s = analyze_sparse(&a, Ordering::Amd, true, super::super::amd::IntWidth::I64).unwrap();
        let mut l = Factor::from_symbolic(&s);
        let params = numeric::Params {
            final_ll: ll,
            ..numeric::Params::default()
        };
        let mut w = Work::new(n);
        numeric::factorize(&a, 0.0, &mut l, &params, &mut w).unwrap();
        l
    }

    /// `A * x`, for the residual check, from the same triangle the factor saw.
    fn matvec(n: usize, p: &[i64], i: &[i64], v: &[f64], x: &[f64], nrhs: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * nrhs];
        for j in 0..n {
            for q in p[j] as usize..p[j + 1] as usize {
                let r = i[q] as usize;
                for c in 0..nrhs {
                    out[j + c * n] += v[q] * x[r + c * n];
                    if r != j {
                        out[r + c * n] += v[q] * x[j + c * n];
                    }
                }
            }
        }
        out
    }

    fn rhs(n: usize, nrhs: usize) -> Vec<f64> {
        let mut b = vec![0.0; n * nrhs];
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for v in b.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        }
        b
    }

    /// Every `nrhs` in 1..=9 exercises all four unrolled kernels and the
    /// blocking loop that feeds them: 4+4+1 for nrhs = 9.
    #[test]
    fn solving_reproduces_the_right_hand_side() {
        for (name, n, edges) in testcorpus::corpus() {
            for ll in [false, true] {
                let (p, i, v) = spd_triangle(n, &edges, false);
                let l = factor(n, &edges, ll);
                for nrhs in 1..=9 {
                    let b = rhs(n, nrhs);
                    let mut x = vec![0.0; n * nrhs];
                    let mut w = SolveWork::new();
                    solve(Sys::A, &l, &b, nrhs, &mut x, &mut w).unwrap();
                    let ax = matvec(n, &p, &i, &v, &x, nrhs);
                    let err = ax
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(err < 1e-9, "{name} ll={ll} nrhs={nrhs}: residual {err:e}");
                }
            }
        }
    }

    /// `A \ b` composed out of its pieces must agree with the one-shot solve,
    /// which is what makes the `sys` dispatch checkable without a second oracle.
    #[test]
    fn the_pieces_compose_into_the_whole_solve() {
        for (name, n, edges) in testcorpus::corpus() {
            for ll in [false, true] {
                let l = factor(n, &edges, ll);
                for nrhs in [1usize, 3, 5] {
                    let b = rhs(n, nrhs);
                    let mut want = vec![0.0; n * nrhs];
                    let mut w = SolveWork::new();
                    solve(Sys::A, &l, &b, nrhs, &mut want, &mut w).unwrap();

                    /* x = P' (L' \ (D \ (L \ (P b)))) */
                    let mut t = vec![0.0; n * nrhs];
                    let mut u = vec![0.0; n * nrhs];
                    solve(Sys::P, &l, &b, nrhs, &mut t, &mut w).unwrap();
                    solve(Sys::L, &l, &t, nrhs, &mut u, &mut w).unwrap();
                    solve(Sys::D, &l, &u, nrhs, &mut t, &mut w).unwrap();
                    if l.is_ll {
                        t.copy_from_slice(&u);
                    }
                    solve(Sys::Lt, &l, &t, nrhs, &mut u, &mut w).unwrap();
                    solve(Sys::Pt, &l, &u, nrhs, &mut t, &mut w).unwrap();
                    assert_eq!(t, want, "{name} ll={ll} nrhs={nrhs}");
                }
            }
        }
    }

    /// `P` and `Pt` invert each other, which is the property `gmm` needs and
    /// scikit-sparse 0.5.0 does not expose.
    #[test]
    fn the_permutation_solves_are_inverses() {
        for (name, n, edges) in testcorpus::corpus() {
            let l = factor(n, &edges, false);
            let b = rhs(n, 3);
            let mut t = vec![0.0; n * 3];
            let mut u = vec![0.0; n * 3];
            let mut w = SolveWork::new();
            solve(Sys::P, &l, &b, 3, &mut t, &mut w).unwrap();
            solve(Sys::Pt, &l, &t, 3, &mut u, &mut w).unwrap();
            assert_eq!(u, b, "{name}");
        }
    }

    #[test]
    fn a_symbolic_factor_is_rejected() {
        let (p, i, x) = spd_triangle(4, &[(0, 1), (1, 2), (2, 3)], false);
        let a = Sparse {
            n: 4,
            p: p.clone(),
            i: i.clone(),
            x,
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(4, &p, &i),
        };
        let s = analyze_sparse(&a, Ordering::Amd, true, super::super::amd::IntWidth::I64).unwrap();
        let l = Factor::from_symbolic(&s);
        let mut w = SolveWork::new();
        assert!(solve(Sys::A, &l, &[0.0; 4], 1, &mut [0.0; 4], &mut w).is_err());
    }
}
