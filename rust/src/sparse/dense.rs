//! The dense kernels the supernodal factorization and solve call, in the exact
//! configurations they call them in.
//!
//! `t_cholmod_super_numeric_worker.c` makes exactly four BLAS/LAPACK calls, and
//! every one of them has its `uplo`/`trans`/`side`/`diag` and its `alpha`/`beta`
//! fixed at the call site:
//!
//!   * `dsyrk ("L","N", …, 1.0, …, 0.0, …)` → [`syrk_ln`]  (`:1002`)
//!   * `dgemm ("N","C", …, 1.0, …, 0.0, …)` → [`gemm_nt`]  (`:1024`)
//!   * `dpotrf ("L", …)`                    → [`potrf_l`]  (`:1069`)
//!   * `dtrsm ("R","L","C","N", …, 1.0, …)` → [`trsm_rlt`] (`:1210`)
//!
//! `t_cholmod_super_solve_worker.c` adds eight more — a vector form and a
//! matrix form of each half of the solve, since it branches on `nrhs == 1`:
//!
//!   * `dtrsv ("L","N","N", …)`                → [`trsv_ln`]  (`:93`)
//!   * `dgemv ("N", …, -1, …, 1, …)`           → [`gemv_n`]   (`:99`)
//!   * `dtrsm ("L","L","N","N", …, 1.0, …)`    → [`trsm_lln`] (`:204`)
//!   * `dgemm ("N","N", …, -1, …, 1, …)`       → [`gemm_nn`]  (`:213`)
//!   * `dgemv ("C", …, -1, …, 1, …)`           → [`gemv_t`]   (`:388`)
//!   * `dtrsv ("L","C","N", …)`                → [`trsv_lt`]  (`:398`)
//!   * `dgemm ("C","N", …, -1, …, 1, …)`       → [`gemm_tn`]  (`:494`)
//!   * `dtrsm ("L","L","C","N", …, 1.0, …)`    → [`trsm_llt`] (`:505`)
//!
//! So these are not a BLAS. They are those twelve call sites, with the branches
//! a general BLAS would take on `uplo` and friends resolved the way CHOLMOD
//! resolves them — the same discipline the rest of this module applies to
//! upstream's `#ifdef` instantiations. `"C"` is a conjugate transpose, which
//! for `CHOLMOD_REAL` is a plain transpose.
//!
//! **Everything is column-major with an explicit leading dimension**, because
//! that is how a supernode is stored: supernode `s` is the `nsrow`-by-`nscol`
//! block at `L->x [L->px[s] ..]`, and the blocks handed to these kernels are
//! sub-blocks of it sharing its `nsrow` stride.
//!
//! **Summation order follows the netlib reference**, loop for loop. That is a
//! stricter contract than "computes the same thing", and it has to be: netlib
//! is not even self-consistent across these twelve, and the differences are
//! observable. `dtrsv ("L","C")` accumulates its dot product with `i`
//! *descending* from `n-1`, while `dtrsm ("L","L","C","N")` — the same solve,
//! one right-hand side at a time — accumulates with `k` *ascending*. So a
//! supernodal solve of one right-hand side does not agree in the last bit with
//! column 0 of a solve of two, and that is upstream's behaviour, not a defect
//! to smooth over. Each kernel below names the loop nest it mirrors.
//!
//! Nothing here can be bit-compared against a tuned BLAS — Accelerate's
//! blocking is not knowable — so the port is gated against upstream's C
//! *linked to these same kernels*, which is what isolates a structural defect
//! from a summation-order difference.
//!
//! **Under `--features blas` each of the twelve hands off to the vendor**
//! above [`super::blas::MIN_FLOPS`], and then the comparison *is* available:
//! hea and upstream's C both linked to Accelerate are issuing the same twelve
//! calls with the same arguments, so `L` and `X` must agree to the bit. They
//! do, over the whole corpus and with no tolerance. That is a check on this
//! mapping specifically — a kernel whose flags or extents were transcribed
//! wrong here would still pass every residual test and would fail that one.
//!
//! **Every multiply-accumulate goes through [`rfma`], the crate's one
//! FP-contraction policy** — fuse on `aarch64`, two roundings everywhere else.
//! That is not a speed decision dressed up as a correctness one; it is what the
//! reference does. The reference writes these loops as a single expression
//! (`C(I,J) = C(I,J) + TEMP*A(I,L)`, `dgemm.f:250-262`, and likewise in the
//! others), a Fortran compiler's `-ffp-contract` defaults to on, and on an ISA
//! with a baseline FMA that is one rounding. Two receipts, both on aarch64:
//!
//!   * `gfortran -O2` with no flags on netlib's reference BLAS emits `fmadd`
//!     inside `dgemm`/`dsyrk`/`dgemv`/`ddot` and `fmsub` inside
//!     `dtrsm`/`dtrsv` — `dtrsv`'s inner loops contain *no* unfused multiply.
//!   * R's shipped `libRblas`, which is that same source built by R's own
//!     toolchain, disassembles the same way.
//!
//! So an un-fused port is the departure, not the fusion, and it is the same
//! departure [`super::numeric::mulsub`] documents for the simplicial path. That
//! one was caught because `L` can be compared against upstream entry for entry;
//! this one could not be, because these kernels are their own oracle. It cost a
//! measured **1.4-1.5x** across the whole kernel mix a factorization issues.
//! `rfma` keys on `target_arch`, not on `target_feature = "fma"`, for the
//! reason its own docstring gives: the reference build is a baseline build.
//!
//! Fusing changes values without changing *order*, so this contract and the
//! summation-order one above are independent and both hold. Note that the
//! linked-kernel gate below cannot see a change of this kind — it moves both
//! sides together — so an arithmetic change here has to be checked by
//! digesting the port's own output before and after.
//!
//! **The reference's `IF (X(J).NE.ZERO)` guards are not ported.** They skip a
//! source column whose multiplier is exactly zero, which is a data-dependent
//! branch in the hot loop for a case the supernodal factorization does not
//! produce; it changes nothing numerically unless the other operand is an
//! infinity or a NaN.

use rayon::prelude::*;

use crate::nmath::util::rfma;

use super::ws::Ws;

#[allow(dead_code)]
pub fn syrk_ln(n: usize, k: usize, a: &[f64], lda: usize, c: &mut [f64], ldc: usize) {
    debug_assert!(n == 0 || c.len() >= (n - 1) * ldc + n);
    syrk_ln_strip(n, k, a, lda, c, ldc, 0, n);
}

pub const SYRK_NB: usize = 8;

#[allow(clippy::too_many_arguments)]
pub fn syrk_ln_strip(
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    c: &mut [f64],
    ldc: usize,
    j0: usize,
    jn: usize,
) {
    /* The vendor gets upstream's own decomposition of the trapezoid — one
     * `dgemm` for the rows below the strip's square and one `dsyrk` for the
     * square — where the recursive halving below exists to give hea's
     * one-thread-per-call kernels a shape a threaded BLAS does not need. */
    #[cfg(vendor_blas)]
    if jn > 0 && super::blas::worth_it(2.0 * n as f64 * jn as f64 * k as f64) {
        if n > j0 + jn {
            super::blas::gemm_nt(
                n - j0 - jn,
                jn,
                k,
                &a[j0 + jn..],
                lda,
                &a[j0..],
                lda,
                &mut c[j0 + jn..],
                ldc,
            );
        }
        return super::blas::syrk_ln(jn, k, &a[j0..], lda, &mut c[j0..], ldc);
    }
    debug_assert!(k == 0 || a.len() >= (k - 1) * lda + n);
    debug_assert!(j0.is_multiple_of(SYRK_NB) && j0 + jn <= n);
    if jn == 0 {
        return;
    }

    /* rows below the strip: a full-width rectangle, no triangle in it */
    if n > j0 + jn {
        gemm_nt(
            n - j0 - jn,
            jn,
            k,
            &a[j0 + jn..],
            lda,
            &a[j0..],
            lda,
            &mut c[j0 + jn..],
            ldc,
        );
    }
    syrk_tri(k, a, lda, c, ldc, j0, j0, j0 + jn);
}

const SYRK_LEAF: usize = 64;

fn syrk_tri(
    k: usize,
    a: &[f64],
    lda: usize,
    c: &mut [f64],
    ldc: usize,
    j0: usize,
    lo: usize,
    hi: usize,
) {
    if hi - lo <= SYRK_LEAF {
        /* `C (jb:hi-1, jb:jb+NB-1)` is a plain `A * A'`, so a block column goes
         * straight through [`gemm_nt`] — including its diagonal `NB`-by-`NB`
         * square, computed whole rather than as a triangle.
         *
         * That writes the strict upper triangle of each diagonal square, which
         * a real `dsyrk ("L", ...)` promises not to touch. It is dead storage:
         * the only reader is `t_cholmod_super_numeric_worker.c:1042-1050`,
         * whose assembly loop is `for (i = j ; i < ndrow2 ; i++)` — lower only
         * — and the `dgemm` that fills the rest of `C` starts at row `ndrow1`.
         * The waste is `NB(NB-1)/2` extra dot products per block column, in
         * exchange for the whole kernel being the vectorized one. */
        let mut jb = lo;
        while jb < hi {
            let jbn = SYRK_NB.min(hi - jb);
            gemm_nt(
                hi - jb,
                jbn,
                k,
                &a[jb..],
                lda,
                &a[jb..],
                lda,
                &mut c[(jb - j0) * ldc + jb..],
                ldc,
            );
            jb += jbn;
        }
        return;
    }

    /* `hi - lo > SYRK_LEAF >= 2 * SYRK_NB`, so the rounded half is strictly
     * inside and both halves shrink */
    let mid = lo + ((hi - lo) / 2).next_multiple_of(SYRK_NB);
    debug_assert!(lo < mid && mid < hi);
    syrk_tri(k, a, lda, c, ldc, j0, lo, mid);
    gemm_nt(
        hi - mid,
        mid - lo,
        k,
        &a[mid..],
        lda,
        &a[lo..],
        lda,
        &mut c[(lo - j0) * ldc + mid..],
        ldc,
    );
    syrk_tri(k, a, lda, c, ldc, j0, mid, hi);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_nt<const MR: usize, const NR: usize, const ACC: bool, const SUB: bool>(
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut Ws<f64>,
    ic: usize,
    ldc: usize,
) {
    let mut acc = [[0.0f64; MR]; NR];
    if ACC {
        for (jj, accj) in acc.iter_mut().enumerate() {
            for (ii, x) in accj.iter_mut().enumerate() {
                *x = c[ic + ii + jj * ldc];
            }
        }
    }
    for l in 0..k {
        let (ao, bo) = (ia + l * lda, jb + l * ldb);
        let mut av = [0.0f64; MR];
        for (ii, v) in av.iter_mut().enumerate() {
            *v = a[ao + ii];
        }
        for (jj, accj) in acc.iter_mut().enumerate() {
            let bv = if SUB { -b[bo + jj] } else { b[bo + jj] };
            for (ii, x) in accj.iter_mut().enumerate() {
                *x = rfma(bv, av[ii], *x);
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &x) in accj.iter().enumerate() {
            c[ic + ii + jj * ldc] = x;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cols_nt<const NR: usize, const ACC: bool, const SUB: bool>(
    m: usize,
    k: usize,
    a: &Ws<f64>,
    ia0: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut Ws<f64>,
    ic0: usize,
    ldc: usize,
) {
    let mut i = 0;
    while i + 8 <= m {
        tile_nt::<8, NR, ACC, SUB>(k, a, ia0 + i, lda, b, jb, ldb, c, ic0 + i, ldc);
        i += 8;
    }
    if i + 4 <= m {
        tile_nt::<4, NR, ACC, SUB>(k, a, ia0 + i, lda, b, jb, ldb, c, ic0 + i, ldc);
        i += 4;
    }
    if i + 2 <= m {
        tile_nt::<2, NR, ACC, SUB>(k, a, ia0 + i, lda, b, jb, ldb, c, ic0 + i, ldc);
        i += 2;
    }
    if i < m {
        tile_nt::<1, NR, ACC, SUB>(k, a, ia0 + i, lda, b, jb, ldb, c, ic0 + i, ldc);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn block_nt<const ACC: bool>(
    m: usize,
    n: usize,
    k: usize,
    a: &Ws<f64>,
    ia0: usize,
    lda: usize,
    b: &Ws<f64>,
    jb0: usize,
    ldb: usize,
    c: &mut Ws<f64>,
    ic0: usize,
    ldc: usize,
) {
    let mut j = 0;
    while j + 4 <= n {
        cols_nt::<4, ACC, false>(m, k, a, ia0, lda, b, jb0 + j, ldb, c, ic0 + j * ldc, ldc);
        j += 4;
    }
    if j + 2 <= n {
        cols_nt::<2, ACC, false>(m, k, a, ia0, lda, b, jb0 + j, ldb, c, ic0 + j * ldc, ldc);
        j += 2;
    }
    if j < n {
        cols_nt::<1, ACC, false>(m, k, a, ia0, lda, b, jb0 + j, ldb, c, ic0 + j * ldc, ldc);
    }
}

pub fn gemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    debug_assert!(k == 0 || (a.len() >= (k - 1) * lda + m && b.len() >= (k - 1) * ldb + n));
    debug_assert!(n == 0 || c.len() >= (n - 1) * ldc + m);
    #[cfg(vendor_blas)]
    if super::blas::worth_it(2.0 * m as f64 * n as f64 * k as f64) {
        return super::blas::gemm_nt(m, n, k, a, lda, b, ldb, c, ldc);
    }
    if wants_packing(m, n, k) || strides_want_packing(m, n, k, lda, ldb) {
        gemm_nt_packed(m, n, k, a, lda, b, ldb, c, ldc);
    } else {
        gemm_nt_direct(m, n, k, a, lda, b, ldb, c, ldc);
    }
}

fn gemm_nt_direct(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    let (a, b) = (Ws::new_ref(a), Ws::new_ref(b));
    let c = Ws::new(c);
    block_nt::<false>(m, n, k, a, 0, lda, b, 0, ldb, c, 0, ldc);
}

const PACK_MR: usize = 8;
const PACK_NR: usize = 4;

const PACK_MC: usize = 256;

const PACK_NC: usize = 1024;

/// Length of the `k` block. Bounds both packed operands, and bounds the
/// micro-panels — `PACK_MR * PACK_KC * 8` and `PACK_NR * PACK_KC * 8` — at 16
/// and 8 KB, which is what keeps the innermost loop inside L1.
///
/// **Free to choose, for the same reason [`KB_SUB`] is.** Each `C (i,j)`
/// accumulates its `k` block in registers, is stored, and is reloaded by the
/// next block to carry on; an `f64` store and reload is exact, so the chain
/// `(((0 + a₀b₀) + a₁b₁) + …)` over all of `k` in index order is the same at
/// any block size, and the same as the unblocked path's.
const PACK_KC: usize = 256;

/// Below this the direct path wins: packing costs a copy of both operands, and
/// there is nothing to buy back while they stay resident.
///
/// Sized against the working set rather than the flops, because the defect it
/// avoids is a working-set one. The direct path re-streams `A` once per `PACK_NR`
/// columns of `B` and `B` once per `PACK_MR` rows of `A`, so what matters is
/// whether an operand survives between re-reads.
///
/// Swept rather than left derived, on one core with non-aliasing `lda` against a
/// build with no residency bound at all: packing costs 33% at a 120 KB working
/// set, and the cost decays to nothing at about a megabyte without ever becoming
/// a gain. So packing is not a win left on the table below that; it is a copy
/// that pays only once the operand would not have survived anyway — or once the
/// strides collide, which is a different question and
/// [`strides_want_packing`]'s.
const PACK_MIN_BYTES: usize = 1 << 20;

fn wants_packing(m: usize, n: usize, k: usize) -> bool {
    m >= PACK_MR * 4
        && n >= PACK_NR * 4
        && k >= 64
        && (m.saturating_mul(k) + n.saturating_mul(k)) * 8 >= PACK_MIN_BYTES
}

/// Bytes between two addresses that land in the same L1 set — the cache's size
/// over its associativity, 128 KB / 8 on an M-series performance core.
const L1_WAY_BYTES: usize = 16 << 10;

const L1_WAYS: usize = 8;

/// How close two columns of a `k` walk may land, in bytes, before the operands
/// stop being read where they lie.
///
/// Measured on a 2048-column `dpotrf` at `lda` = 2048 + {0, 1, 2, 4, 8}, whose
/// closest pair of columns within eight is {0, 8, 16, 32, 64} bytes apart:
/// [`gemm_sub_direct`] runs at 13.7, 14.1, 38.3, 54.6 and 55.4 GFLOP/s, and
/// [`gemm_sub_packed`] at 41.5, 38.4, 42.0, 46.0 and 45.5 — flat, because a
/// packed panel has no `lda` in it. The crossing is between 16 and 32.
const ALIAS_BYTES: usize = 32;

/// Whether a stride-`ld` walk collides on too few L1 sets to run at speed.
///
/// The columns of a `k` walk sit `ld` doubles apart, so column `q` lands
/// `q · ld · 8` bytes on — and, modulo [`L1_WAY_BYTES`], in the same set as
/// column 0 whenever that product comes back near zero. A supernode whose
/// `nsrow` happens to be 2048 puts *every* column in one set: on this machine
/// that is a 4x cliff, 13.7 GFLOP/s against the 55 the same code gets one row
/// wider.
///
/// Only the first [`L1_WAYS`] columns are examined, and that is the whole
/// argument for the bound rather than a sampling shortcut: a stride that takes
/// eight or more columns to repeat a set has spread them over eight sets, which
/// is exactly as many ways as there are. Checking further would flag `lda`
/// = 2304 (which repeats at `q = 8`) and cost it 23%, measured.
#[inline]
fn strides_alias(ld: usize) -> bool {
    let s = (ld * 8) % L1_WAY_BYTES;
    let mut d = 0;
    for _ in 1..L1_WAYS {
        d = (d + s) % L1_WAY_BYTES;
        if d.min(L1_WAY_BYTES - d) < ALIAS_BYTES {
            return true;
        }
    }
    false
}

/// Whether to pack because the strides collide, rather than because the
/// operands are large.
///
/// [`wants_packing`] is a residency test and this is not. Set aliasing bites at
/// 128 KB exactly as hard as at 43 MB, so gating it behind [`PACK_MIN_BYTES`]
/// leaves every smaller call on the cliff: on the corpus's own flop-weighted
/// median `dgemm`, `m` = 255, `n` = 149, `k` = 106, that is 27.9 GFLOP/s against
/// the 47.2 the packed path gets.
///
/// Each bound is where the measured gain crosses 1, swept against a build that
/// packs on [`strides_alias`] alone with no shape test whatever. `k > L1_WAYS`
/// is the mechanism's own floor rather than a fitted one, and the sweep confirms
/// the mechanism: a walk of eight or fewer columns has a way for each of them
/// however they collide, so there is nothing to win, and the measured gain is
/// flat at 1.00-1.03 up to exactly eight and climbs from exactly nine.
///
/// The shape tests come first so a call that could not benefit never pays for
/// the address arithmetic — the majority of calls in a sparse factorization are
/// a few columns wide.
fn strides_want_packing(m: usize, n: usize, k: usize, lda: usize, ldb: usize) -> bool {
    m >= PACK_MR * 4
        && n >= PACK_NR * 4
        && k > L1_WAYS
        && (strides_alias(lda) || strides_alias(ldb))
}

#[inline]
fn round_up(x: usize, q: usize) -> usize {
    x.div_ceil(q) * q
}

fn pack_rows(
    mb: usize,
    kb: usize,
    a: &Ws<f64>,
    i0: usize,
    l0: usize,
    lda: usize,
    out: &mut Ws<f64>,
) {
    let mut p = 0;
    let mut i = 0;
    while i < mb {
        let mr = (mb - i).min(PACK_MR);
        for l in 0..kb {
            let src = i0 + i + (l0 + l) * lda;
            let dst = p + l * PACK_MR;
            for ii in 0..mr {
                out[dst + ii] = a[src + ii];
            }
        }
        p += PACK_MR * kb;
        i += PACK_MR;
    }
}

fn pack_cols(
    nb: usize,
    kb: usize,
    b: &Ws<f64>,
    j0: usize,
    l0: usize,
    ldb: usize,
    out: &mut Ws<f64>,
) {
    let mut p = 0;
    let mut j = 0;
    while j < nb {
        let nr = (nb - j).min(PACK_NR);
        for l in 0..kb {
            let src = j0 + j + (l0 + l) * ldb;
            let dst = p + l * PACK_NR;
            for jj in 0..nr {
                out[dst + jj] = b[src + jj];
            }
        }
        p += PACK_NR * kb;
        j += PACK_NR;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn rows_packed_nt<const NR: usize, const ACC: bool, const SUB: bool>(
    mb: usize,
    kb: usize,
    ap: &Ws<f64>,
    bp: &Ws<f64>,
    pb: usize,
    c: &mut Ws<f64>,
    ic0: usize,
    ldc: usize,
) {
    let (mut i, mut pa) = (0usize, 0usize);
    while i + PACK_MR <= mb {
        tile_nt::<PACK_MR, NR, ACC, SUB>(kb, ap, pa, PACK_MR, bp, pb, PACK_NR, c, ic0 + i, ldc);
        pa += PACK_MR * kb;
        i += PACK_MR;
    }
    let rem = mb - i;
    let mut o = 0;
    if rem & 4 != 0 {
        tile_nt::<4, NR, ACC, SUB>(
            kb,
            ap,
            pa + o,
            PACK_MR,
            bp,
            pb,
            PACK_NR,
            c,
            ic0 + i + o,
            ldc,
        );
        o += 4;
    }
    if rem & 2 != 0 {
        tile_nt::<2, NR, ACC, SUB>(
            kb,
            ap,
            pa + o,
            PACK_MR,
            bp,
            pb,
            PACK_NR,
            c,
            ic0 + i + o,
            ldc,
        );
        o += 2;
    }
    if rem & 1 != 0 {
        tile_nt::<1, NR, ACC, SUB>(
            kb,
            ap,
            pa + o,
            PACK_MR,
            bp,
            pb,
            PACK_NR,
            c,
            ic0 + i + o,
            ldc,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn block_packed_nt<const ACC: bool, const SUB: bool>(
    mb: usize,
    nb: usize,
    kb: usize,
    ap: &Ws<f64>,
    bp: &Ws<f64>,
    c: &mut Ws<f64>,
    ic0: usize,
    ldc: usize,
) {
    let (mut j, mut pb) = (0usize, 0usize);
    while j + PACK_NR <= nb {
        rows_packed_nt::<PACK_NR, ACC, SUB>(mb, kb, ap, bp, pb, c, ic0 + j * ldc, ldc);
        pb += PACK_NR * kb;
        j += PACK_NR;
    }
    let rem = nb - j;
    /* the last one to three columns share a single packed panel of stride
     * `PACK_NR`, so the tail advances *within* it — `+= 2`, not `+= 2 * kb` */
    if rem & 2 != 0 {
        rows_packed_nt::<2, ACC, SUB>(mb, kb, ap, bp, pb, c, ic0 + j * ldc, ldc);
        pb += 2;
        j += 2;
    }
    if rem & 1 != 0 {
        rows_packed_nt::<1, ACC, SUB>(mb, kb, ap, bp, pb, c, ic0 + j * ldc, ldc);
    }
}

thread_local! {
    static PACK_BUF: core::cell::RefCell<(Vec<f64>, Vec<f64>)> =
        const { core::cell::RefCell::new((Vec::new(), Vec::new())) };
}

fn gemm_nt_packed(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    let (a, b) = (Ws::new_ref(a), Ws::new_ref(b));
    let c = Ws::new(c);
    PACK_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        let (ap, bp) = &mut *buf;
        ap.resize(round_up(PACK_MC, PACK_MR) * PACK_KC, 0.0);
        bp.resize(round_up(PACK_NC, PACK_NR) * PACK_KC, 0.0);
        let ap = Ws::new(ap);
        let bp = Ws::new(bp);

        let mut jc = 0;
        while jc < n {
            let nb = (n - jc).min(PACK_NC);
            let mut pc = 0;
            while pc < k {
                let kb = (k - pc).min(PACK_KC);
                pack_cols(nb, kb, b, jc, pc, ldb, bp);
                let mut ic = 0;
                while ic < m {
                    let mb = (m - ic).min(PACK_MC);
                    pack_rows(mb, kb, a, ic, pc, lda, ap);
                    let ic0 = ic + jc * ldc;
                    if pc == 0 {
                        block_packed_nt::<false, false>(mb, nb, kb, ap, bp, c, ic0, ldc);
                    } else {
                        block_packed_nt::<true, false>(mb, nb, kb, ap, bp, c, ic0, ldc);
                    }
                    ic += PACK_MC;
                }
                pc += PACK_KC;
            }
            jc += PACK_NC;
        }
    });
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_sub<const MR: usize, const NR: usize>(
    k: usize,
    x: &mut Ws<f64>,
    ia: usize,
    lda: usize,
    jb: usize,
    ldb: usize,
    ic: usize,
    ldc: usize,
) {
    let mut acc = [[0.0f64; MR]; NR];
    for (jj, accj) in acc.iter_mut().enumerate() {
        for (ii, v) in accj.iter_mut().enumerate() {
            *v = x[ic + ii + jj * ldc];
        }
    }
    for l in 0..k {
        let (ao, bo) = (ia + l * lda, jb + l * ldb);
        for (jj, accj) in acc.iter_mut().enumerate() {
            let bv = x[bo + jj];
            for (ii, v) in accj.iter_mut().enumerate() {
                *v = rfma(-bv, x[ao + ii], *v);
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &v) in accj.iter().enumerate() {
            x[ic + ii + jj * ldc] = v;
        }
    }
}

/// One `MR`-row slab of [`tile_sub`] across every column of `C`, the columns
/// grouped 4/2/1.
///
/// **This nesting is the one that matters.** The slab's `MR` rows of `A` are
/// `MR * k` doubles — 64 KB at `MR = 8` and `k = 1021`, so they fit L1 — and
/// every column group reuses them. Sweeping columns on the outside instead, as
/// a `C`-major loop would, streams the whole of `A` once per group: eight
/// passes over an operand that is megabytes wide, which on `gmm`'s panel put
/// the factorization at three quarters of the machine's memory bandwidth and
/// left it there no matter how many cores were pointed at it.
///
/// Free to reorder because the tiles are independent — each `C (i,j)` sums over
/// its own `l` ascending regardless of which tile computes it, and no tile
/// reads another's output.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn slab_sub<const MR: usize>(
    n: usize,
    k: usize,
    x: &mut Ws<f64>,
    ia: usize,
    lda: usize,
    jb: usize,
    ldb: usize,
    ic: usize,
    ldc: usize,
) {
    let mut j = 0;
    while j + 4 <= n {
        tile_sub::<MR, 4>(k, x, ia, lda, jb + j, ldb, ic + j * ldc, ldc);
        j += 4;
    }
    if j + 2 <= n {
        tile_sub::<MR, 2>(k, x, ia, lda, jb + j, ldb, ic + j * ldc, ldc);
        j += 2;
    }
    if j < n {
        tile_sub::<MR, 1>(k, x, ia, lda, jb + j, ldb, ic + j * ldc, ldc);
    }
}

/// [`tile_sub`] with the destination handed over as `NR` separate column
/// slices instead of one offset into the shared array.
///
/// **Same rounding, entry for entry.** `tile_sub` loads `C (i,j)`, then
/// subtracts `B (j,l) * A (i,l)` for `l` ascending, and does that independently
/// for every `(i,j)` in its tile. Nothing crosses between rows or columns, so
/// carving the tile up moves work between tasks without moving a rounding —
/// which is what lets [`gemm_sub_par`] hand row blocks to different threads and
/// still reproduce the serial answer bit for bit.
///
/// The indirection is paid twice per tile, not once per `l`: the destination is
/// read into `acc` before the `l` loop and written back after it, so the loop
/// itself touches only `a` and `b`, which are plain shared borrows.
///
/// The two ends move whole rows, and that is load-bearing rather than tidy.
/// Written an entry at a time, `c [j0 + jj] [i + ii]` is a bounds-checked double
/// indirection, and `MR * NR` of them at each end put a panicking edge between
/// every pair of writes into `acc`. LLVM will not promote an array that has to
/// be consistent at that many exits, so `acc` stays in memory and the `l` loop —
/// which never touches `c` at all — comes out scalar and spilling: 79 `fsub`
/// against [`tile_sub`]'s 49 `fsub.2d`, 218 loads against 73. That is a factor
/// of two on a kernel already near the machine's ceiling, and it destroys
/// threading rather than merely slowing the kernel: each worker runs at
/// 15.5 GFLOP/s where the serial path runs at 29, so two threads buy nothing.
/// Slicing `[i .. i + MR]` once per column leaves `NR` checks per end and hands
/// LLVM a fixed-size copy; the vector form comes back and the scaling with it.
///
/// `A`'s `MR` values are hoisted into `av` for the same reason [`tile_nt`] does
/// it. It measures neutral here — 0.1-1% across the `dtrsm` shapes, inside the
/// noise, so LLVM is already hoisting — but the idiom is what pins the codegen
/// this function is fragile about, so it stays.
///
/// Also the packed scattered path's tile: a packed panel is what it already
/// reads, with `lda` set to [`PACK_MR`] and `ldb` to [`PACK_NR`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_scat<const MR: usize, const NR: usize>(
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut [&mut [f64]],
    j0: usize,
    i: usize,
) {
    let mut acc = [[0.0f64; MR]; NR];
    for (jj, accj) in acc.iter_mut().enumerate() {
        accj.copy_from_slice(&c[j0 + jj][i..i + MR]);
    }
    for l in 0..k {
        let (ao, bo) = (ia + l * lda, jb + l * ldb);
        let mut av = [0.0f64; MR];
        for (ii, x) in av.iter_mut().enumerate() {
            *x = a[ao + ii];
        }
        for (jj, accj) in acc.iter_mut().enumerate() {
            let bv = -b[bo + jj];
            for (ii, v) in accj.iter_mut().enumerate() {
                *v = rfma(bv, av[ii], *v);
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        c[j0 + jj][i..i + MR].copy_from_slice(accj);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn slab_scat<const MR: usize>(
    n: usize,
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut [&mut [f64]],
    i: usize,
) {
    let mut j = 0;
    while j + 4 <= n {
        tile_scat::<MR, 4>(k, a, ia, lda, b, jb + j, ldb, c, j, i);
        j += 4;
    }
    if j + 2 <= n {
        tile_scat::<MR, 2>(k, a, ia, lda, b, jb + j, ldb, c, j, i);
        j += 2;
    }
    if j < n {
        tile_scat::<MR, 1>(k, a, ia, lda, b, jb + j, ldb, c, j, i);
    }
}

/// One task of [`gemm_sub_par`]: every column of `C`, over the rows this task
/// was handed.
///
/// Same shape as [`gemm_sub`], including its dispatch — a task packs on exactly
/// the condition the serial path packs on, so threading a call does not change
/// which kernel runs it. Omitting the aliasing arm here makes adding a thread a
/// pessimization rather than merely a missed gain: on `dtrsm` 1024×1024, whose
/// `ld` is `n + m` = 2048 and therefore aliases, the serial path packs, so a
/// non-packing parallel path runs 1.6x *slower* on two threads than the serial
/// one and recovers only 1.75x on eight. The non-aliasing neighbour scales 3.55x
/// on the same code.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "profiling", inline(never))]
fn block_scat(
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut [&mut [f64]],
) {
    let (m, n) = (c[0].len(), c.len());
    if strides_want_packing(m, n, k, lda, ldb) {
        block_scat_packed(k, a, ia, lda, b, jb, ldb, c);
    } else {
        block_scat_direct(k, a, ia, lda, b, jb, ldb, c);
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "profiling", inline(never))]
fn block_scat_direct(
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut [&mut [f64]],
) {
    let (m, n) = (c[0].len(), c.len());
    let mut l0 = 0;
    while l0 < k {
        let kb = KB_SUB.min(k - l0);
        let (ia, jb) = (ia + l0 * lda, jb + l0 * ldb);
        let mut i = 0;
        while i + 8 <= m {
            slab_scat::<8>(n, kb, a, ia + i, lda, b, jb, ldb, c, i);
            i += 8;
        }
        if i + 4 <= m {
            slab_scat::<4>(n, kb, a, ia + i, lda, b, jb, ldb, c, i);
            i += 4;
        }
        if i + 2 <= m {
            slab_scat::<2>(n, kb, a, ia + i, lda, b, jb, ldb, c, i);
            i += 2;
        }
        if i < m {
            slab_scat::<1>(n, kb, a, ia + i, lda, b, jb, ldb, c, i);
        }
        l0 += kb;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn rows_packed_scat<const NR: usize>(
    mb: usize,
    kb: usize,
    ap: &Ws<f64>,
    bp: &Ws<f64>,
    pb: usize,
    c: &mut [&mut [f64]],
    j0: usize,
    i0: usize,
) {
    let (mut i, mut pa) = (0usize, 0usize);
    while i + PACK_MR <= mb {
        tile_scat::<PACK_MR, NR>(kb, ap, pa, PACK_MR, bp, pb, PACK_NR, c, j0, i0 + i);
        pa += PACK_MR * kb;
        i += PACK_MR;
    }
    /* the last panel is packed to `PACK_MR` stride however few rows it holds,
     * so the tail walks *within* it — `pa + o`, not `pa + o * kb` */
    let rem = mb - i;
    let mut o = 0;
    if rem & 4 != 0 {
        tile_scat::<4, NR>(kb, ap, pa + o, PACK_MR, bp, pb, PACK_NR, c, j0, i0 + i + o);
        o += 4;
    }
    if rem & 2 != 0 {
        tile_scat::<2, NR>(kb, ap, pa + o, PACK_MR, bp, pb, PACK_NR, c, j0, i0 + i + o);
        o += 2;
    }
    if rem & 1 != 0 {
        tile_scat::<1, NR>(kb, ap, pa + o, PACK_MR, bp, pb, PACK_NR, c, j0, i0 + i + o);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn block_packed_scat(
    mb: usize,
    nb: usize,
    kb: usize,
    ap: &Ws<f64>,
    bp: &Ws<f64>,
    c: &mut [&mut [f64]],
    j0: usize,
    i0: usize,
) {
    let (mut j, mut pb) = (0usize, 0usize);
    while j + PACK_NR <= nb {
        rows_packed_scat::<PACK_NR>(mb, kb, ap, bp, pb, c, j0 + j, i0);
        pb += PACK_NR * kb;
        j += PACK_NR;
    }
    let rem = nb - j;
    if rem & 2 != 0 {
        rows_packed_scat::<2>(mb, kb, ap, bp, pb, c, j0 + j, i0);
        pb += 2;
        j += 2;
    }
    if rem & 1 != 0 {
        rows_packed_scat::<1>(mb, kb, ap, bp, pb, c, j0 + j, i0);
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "profiling", inline(never))]
fn block_scat_packed(
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut [&mut [f64]],
) {
    let (m, n) = (c[0].len(), c.len());
    PACK_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        let (apv, bpv) = &mut *buf;
        apv.resize(round_up(PACK_MC, PACK_MR) * PACK_KC, 0.0);
        bpv.resize(round_up(PACK_NC, PACK_NR) * PACK_KC, 0.0);
        let ap = Ws::new(apv);
        let bp = Ws::new(bpv);

        let mut jc = 0;
        while jc < n {
            let nb = (n - jc).min(PACK_NC);
            let mut pc = 0;
            while pc < k {
                let kb = (k - pc).min(PACK_KC);
                pack_cols(nb, kb, b, jb + jc, pc, ldb, bp);
                let mut i0 = 0;
                while i0 < m {
                    let mb = (m - i0).min(PACK_MC);
                    pack_rows(mb, kb, a, ia + i0, pc, lda, ap);
                    block_packed_scat(mb, nb, kb, ap, bp, c, jc, i0);
                    i0 += PACK_MC;
                }
                pc += PACK_KC;
            }
            jc += PACK_NC;
        }
    });
}

const GEMM_PAR_FLOPS: f64 = 2.0e6;

/// How many row blocks [`gemm_sub_par`] cuts per thread.
///
/// One block per thread is the wrong number on a heterogeneous machine: the
/// blocks are equal and the cores are not, so the join waits on whichever block
/// landed on an efficiency core. Over-decomposing lets rayon's work stealing
/// discover the ratio instead of the schedule assuming it. Measured on `gmm`'s
/// 1021-column supernode, 12 threads: 9.3 ms at one block per thread, 6.2 at
/// four, 6.2 at eight.
const PAR_OVER: usize = 4;

/// How many columns of `A` (and of `B`) one pass of [`gemm_sub_direct`] covers.
///
/// **Free to choose, for the reason the panel width is.** A destination entry
/// is loaded, has its sources for this `k` range subtracted in ascending order,
/// and is stored; the next range picks it up where the last left off. The
/// intermediate store and reload of an `f64` is exact, so `C (i,j)` still sees
/// `(((c - a0 b0) - a1 b1) - ...)` over all of `k` in index order, at any block
/// size.
///
/// It is not free for speed, and it is the single largest constant in this file.
/// What the block has to fit is not `B`'s doubles but the *cache lines* both
/// operands touch. A `k` block reads `kb` columns of each, and a column is a
/// short contiguous run at a stride of `ld` doubles, so it costs whole lines
/// however few of its bytes are wanted: one to two lines per column of `A`'s
/// eight-row slab and two or more per column of `B`. Sizing `B`'s doubles
/// against L1 instead — 8192 of them, 64 KB — is 256 columns at `n = 32` and,
/// counted in lines, three to four times the L1 it is supposed to fit inside.
///
/// Swept on the shapes `potrf_l` and `trsm_rlt` issue: every shape peaks at 16
/// and improves monotonically on the way down to it, so this is one optimum
/// rather than a compromise between two regimes. Below 16 the register tile runs
/// out of work to hide the reload of `C` — `MR·NR` loads and stores bracket
/// `kb·MR·NR` multiply-adds, which is 2.3 flops per load at `kb = 16` and 1.6 at
/// `kb = 4`.
///
/// Constant rather than `KC_DOUBLES / n`: it bounds how far *down a column* a
/// pass walks, and `n` does not enter that.
const KB_SUB: usize = 16;

#[allow(clippy::too_many_arguments)]
fn gemm_sub_par(
    m: usize,
    n: usize,
    k: usize,
    x: &mut Ws<f64>,
    ia: usize,
    lda: usize,
    jb: usize,
    ldb: usize,
    ic: usize,
    ldc: usize,
    nt: usize,
) {
    let (src, dst) = x.split_at_mut(ic);

    /* the columns of C.  The last one is whatever remains: the array is only
     * guaranteed out to `(n-1)*ldc + m`. */
    let mut tails: Vec<&mut [f64]> = Vec::with_capacity(n);
    let mut rest = dst;
    for j in 0..n {
        let col = if j + 1 < n {
            let (c, tail) = rest.split_at_mut(ldc);
            rest = tail;
            c
        } else {
            std::mem::take(&mut rest)
        };
        tails.push(&mut col[..m]);
    }

    /* Peel one row block off every column at a time, so the pieces come out in
     * row-block-major order and `par_chunks_mut` can hand a whole block to a
     * task without a second pass to transpose them.
     *
     * Two allocations for the call, and that matters: this runs once per panel,
     * and a `Vec` per column plus a `Vec` per block — which is what collecting
     * the columns separately and transposing them costs — was enough overhead
     * that widening the panel measurably improved thread scaling, i.e. the
     * fork/join was competing with the arithmetic. */
    let rows = m.div_ceil(nt * PAR_OVER).max(8);
    let nblk = m.div_ceil(rows);
    let mut flat: Vec<&mut [f64]> = Vec::with_capacity(nblk * n);
    for _ in 0..nblk {
        for t in tails.iter_mut() {
            let taken = std::mem::take(t);
            let (head, tail) = taken.split_at_mut(rows.min(taken.len()));
            flat.push(head);
            *t = tail;
        }
    }

    flat.par_chunks_mut(n)
        .enumerate()
        .for_each(|(b, c)| block_scat(k, src, ia + b * rows, lda, src, jb, ldb, c));
}

/// `C -= A * B'` for three sub-blocks of one array: `A` is `m`-by-`k` at `ia`,
/// `B` is `n`-by-`k` at `jb`, `C` is `m`-by-`n` at `ic`.
///
/// Two paths over the same register tile, but **not** the split [`gemm_nt`]
/// makes. `gemm_nt` packs on either condition; this one packs *only* when the
/// strides alias, because the direct path with [`KB_SUB`] behind it beats the
/// packed one at every size — 53-56 GFLOP/s against 42-47 on the same shapes.
/// What packing buys here is not locality, it is the removal of `lda` from the
/// inner loop, and only [`strides_want_packing`] can want that.
///
/// **`C` must not overlap `A` or `B`.** Both callers satisfy it the same way —
/// everything read lies in columns left of the panel and everything written in
/// the panel — and the direct path has always needed it too, since its tiles
/// interleave reads of `A` with writes of `C`. The packed path needs it one
/// step further out: it copies `A` out of the array *between* writes to `C`.
#[allow(clippy::too_many_arguments)]
fn gemm_sub(
    m: usize,
    n: usize,
    k: usize,
    x: &mut Ws<f64>,
    ia: usize,
    lda: usize,
    jb: usize,
    ldb: usize,
    ic: usize,
    ldc: usize,
) {
    if strides_want_packing(m, n, k, lda, ldb) {
        gemm_sub_packed(m, n, k, x, ia, lda, jb, ldb, ic, ldc);
    } else {
        gemm_sub_direct(m, n, k, x, ia, lda, jb, ldb, ic, ldc);
    }
}

/// [`gemm_sub`] with both operands copied into [`tile_nt`]'s panel layout.
///
/// [`gemm_nt_packed`]'s loop nest with `ACC` forced on — a destination entry is
/// always accumulated into here, where `gemm_nt` starts its first `k` block from
/// zero — and `SUB` on, which is the `-` in `C -= A B'`.
///
/// **Same rounding as [`gemm_sub_direct`], for [`PACK_KC`]'s reason.** The `pc`
/// loop ascends and each block loads `C`, subtracts its sources in ascending
/// `l`, and stores; the intermediate `f64` store and reload is exact, so
/// `C (i,j)` sees `(((c - a₀b₀) - a₁b₁) - …)` over all of `k` in index order at
/// either block size. `tile_nt::<_, _, true, true>` is `tile_sub`'s arithmetic
/// with `A` hoisted, which is pure code motion.
///
/// **Only worth running when the strides alias**, which is the whole of
/// [`gemm_sub`]'s dispatch. A packed panel is contiguous, so its rate barely
/// moves with `lda` — 38-47 GFLOP/s across the sweep in [`ALIAS_BYTES`] — where
/// the direct path swings between 13.7 and 55.7 over the same eight values. The
/// narrow `nb` is what holds the packed number at ~43 rather than
/// [`gemm_nt`]'s 54: the same shapes at `nb` = 64, 128 and 256 run at 49, 52 and
/// 54, so widening the panel is the way to raise this path's ceiling — see
/// [`panel_width`].
#[allow(clippy::too_many_arguments)]
fn gemm_sub_packed(
    m: usize,
    n: usize,
    k: usize,
    x: &mut Ws<f64>,
    ia: usize,
    lda: usize,
    jb: usize,
    ldb: usize,
    ic: usize,
    ldc: usize,
) {
    PACK_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        let (apv, bpv) = &mut *buf;
        apv.resize(round_up(PACK_MC, PACK_MR) * PACK_KC, 0.0);
        bpv.resize(round_up(PACK_NC, PACK_NR) * PACK_KC, 0.0);
        let ap = Ws::new(apv);
        let bp = Ws::new(bpv);

        let mut jc = 0;
        while jc < n {
            let nb = (n - jc).min(PACK_NC);
            let mut pc = 0;
            while pc < k {
                let kb = (k - pc).min(PACK_KC);
                pack_cols(nb, kb, x, jb + jc, pc, ldb, bp);
                let mut i0 = 0;
                while i0 < m {
                    let mb = (m - i0).min(PACK_MC);
                    pack_rows(mb, kb, x, ia + i0, pc, lda, ap);
                    let ic0 = ic + i0 + jc * ldc;
                    block_packed_nt::<true, true>(mb, nb, kb, ap, bp, x, ic0, ldc);
                    i0 += PACK_MC;
                }
                pc += PACK_KC;
            }
            jc += PACK_NC;
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn gemm_sub_direct(
    m: usize,
    n: usize,
    k: usize,
    x: &mut Ws<f64>,
    ia: usize,
    lda: usize,
    jb: usize,
    ldb: usize,
    ic: usize,
    ldc: usize,
) {
    let mut l0 = 0;
    while l0 < k {
        let kb = KB_SUB.min(k - l0);
        let (ia, jb) = (ia + l0 * lda, jb + l0 * ldb);
        let mut i = 0;
        while i + 8 <= m {
            slab_sub::<8>(n, kb, x, ia + i, lda, jb, ldb, ic + i, ldc);
            i += 8;
        }
        if i + 4 <= m {
            slab_sub::<4>(n, kb, x, ia + i, lda, jb, ldb, ic + i, ldc);
            i += 4;
        }
        if i + 2 <= m {
            slab_sub::<2>(n, kb, x, ia + i, lda, jb, ldb, ic + i, ldc);
            i += 2;
        }
        if i < m {
            slab_sub::<1>(n, kb, x, ia + i, lda, jb, ldb, ic + i, ldc);
        }
        l0 += kb;
    }
}

#[inline(always)]
fn strip_sub<const LR: usize>(
    m: usize,
    x: &mut Ws<f64>,
    yo: usize,
    t: &[f64; LR],
    xo: &[usize; LR],
) {
    /* one f64x2 pair per accumulator, four of them live at once */
    const IB: usize = 8;
    let mut i = 0;
    while i + IB <= m {
        let mut v = [0.0f64; IB];
        for (ii, vv) in v.iter_mut().enumerate() {
            *vv = x[yo + i + ii];
        }
        for (q, &tq) in t.iter().enumerate() {
            let xq = xo[q] + i;
            for (ii, vv) in v.iter_mut().enumerate() {
                *vv = rfma(-tq, x[xq + ii], *vv);
            }
        }
        for (ii, &vv) in v.iter().enumerate() {
            x[yo + i + ii] = vv;
        }
        i += IB;
    }
    while i < m {
        let mut v = x[yo + i];
        for (q, &tq) in t.iter().enumerate() {
            v = rfma(-tq, x[xo[q] + i], v);
        }
        x[yo + i] = v;
        i += 1;
    }
}

/// `A = L*L'` in place, lower triangle — `dpotrf ("L", n, a, lda)`.
///
/// Returns LAPACK's `info`: 0 on success, or the **1-based** index of the first
/// diagonal entry that came out non-positive, in which case the factorization
/// stops there. The worker reads that index directly — `L->minor = k1 + info -
/// 1` and `nscol_new = info - 1` (`:1093,1114`) — so its base and its
/// "non-positive" test are part of the contract, not an implementation detail.
///
/// This is `dpotrf ("L")`'s blocking, loop for loop. Per panel `dpotrf.f:216-232`
/// issues four calls, and the port maps onto them like this:
///
///   * `DSYRK ("L","N", JB, J-1, …)` and `DGEMM ("N","T", N-J-JB+1, JB, J-1, …)`
///     update the diagonal block and the rows below it with the same `-A A'`
///     over the same `J-1` columns, differing only in which rows they cover. One
///     [`gemm_sub`] over the whole `n-j0` rows does both — the one deviation,
///     and it is a fusion of two calls rather than a change of arithmetic, since
///     `dsyrk ("L","N")` and `dgemm ("N","T")` sum a destination entry
///     identically. It writes the diagonal block's strict upper triangle, which
///     neither reference call touches, so that triangle is saved and put back.
///   * `DPOTRF2 ("L", JB, …)` factorizes the `JB`-by-`JB` **diagonal block
///     only** → [`potf2_panel`] with `m = n = nb`.
///   * `DTRSM ("R","L","T","N", N-J-JB+1, JB, …)` solves the rows below against
///     it → [`trsm_rlt`], which is already that call.
///
/// **The rows below the diagonal block belong to `DTRSM`, not to the ladder.**
/// Running the unblocked ladder over all `n-j0` rows — as this did before —
/// computes the same `L` entry for entry, but it charges `m·nb²` flops to a
/// rank-1 loop that reloads its destination strip every few sources, where the
/// reference charges them to a blocked, splittable kernel. On `gmm`'s
/// 1021-column supernode that was 0.83 ms of unsplittable work in a 7.0 ms
/// factorization at one thread, and 0.97 of 3.6 ms at eight — the whole of
/// Amdahl's serial term, and none of it inherent.
///
/// A destination entry still sees every source once, `l` ascending, whether it
/// arrives through the block update, the ladder, or the solve, so all of this
/// blocking is a scheduling change and `L` is unchanged entry for entry.
pub fn potrf_l(n: usize, a: &mut [f64], lda: usize, nt: usize) -> i64 {
    debug_assert!(n == 0 || a.len() >= (n - 1) * lda + n);
    #[cfg(vendor_blas)]
    if super::blas::worth_it((n as f64).powi(3) / 3.0) {
        let _ = nt;
        return super::blas::potrf_l(n, a, lda);
    }
    let a = Ws::new(a);

    let mut save = [0.0f64; POTRF_NB * (POTRF_NB - 1) / 2];
    let wide = panel_width(n);
    let mut j0 = 0;
    while j0 < n {
        let nb = wide.min(n - j0);
        if j0 > 0 {
            /* `gemm_sub` writes whole rectangles, so it also covers the strict
             * upper triangle of this panel's diagonal square — which `dpotrf
             * ("L")` promises not to touch, and which the supernodal worker
             * relies on staying at the zero it left there. Nothing reads it
             * (`potf2_panel` and every later panel take the lower triangle
             * only), but `L` is compared against upstream's entry for entry, so
             * it is put back rather than argued about. */
            let mut q = 0;
            for j in 1..nb {
                for i in 0..j {
                    save[q] = a[j0 + i + (j0 + j) * lda];
                    q += 1;
                }
            }

            /* A (j0:n-1, j0:j0+nb-1) -= A (j0:n-1, 0:j0-1) * A (j0:j0+nb-1, 0:j0-1)' */
            let m = n - j0;
            if nt > 1 && 2.0 * (m * nb * j0) as f64 >= GEMM_PAR_FLOPS {
                gemm_sub_par(m, nb, j0, a, j0, lda, j0, lda, j0 + j0 * lda, lda, nt);
            } else {
                gemm_sub(m, nb, j0, a, j0, lda, j0, lda, j0 + j0 * lda, lda);
            }

            let mut q = 0;
            for j in 1..nb {
                for i in 0..j {
                    a[j0 + i + (j0 + j) * lda] = save[q];
                    q += 1;
                }
            }
        }
        /* DPOTRF2 ("L", JB, A (J,J), LDA, INFO) — the diagonal block alone */
        let info = potf2_panel(nb, nb, a, j0 + j0 * lda, lda);
        if info != 0 {
            /* the reference goes straight to `INFO = INFO + J - 1` and returns,
             * leaving this panel's DTRSM undone, so the rows below keep their
             * post-update values. The supernodal worker zeroes the whole
             * supernode and replays it on `info != 0` (`:1297`), so nothing
             * written here survives either way. */
            return j0 as i64 + info;
        }

        /* DTRSM ("R","L","T","N", N-J-JB+1, JB, 1.0, A (J,J), LDA, A (J+JB,J), LDA) */
        if j0 + nb < n {
            let (_, below) = a.split_at_mut(j0 + j0 * lda);
            trsm_rlt(n - j0 - nb, nb, below, lda, nt);
        }
        j0 += nb;
    }
    0
}

const POTRF_NB: usize = 32;

/// How many columns of an `n`-column block go through the rank-1 ladder before
/// the next `C -= A B'` catches the rest up.
///
/// **Free to choose.** A destination entry accumulates over `l` ascending
/// whether a source arrives through the block update or through the ladder, so
/// the width is a scheduling parameter and `L` is the same entry for entry at
/// any value of it — which the bit-exactness digest over the whole solve corpus
/// confirms, unchanged across 8, 16, 32 and 64.
///
/// It is not free for *speed*, and it cuts both ways. The ladder is the slow
/// half: its share of the work is roughly `3·nb/n`, so a wide panel on a small
/// block spends everything there. A narrow panel on a large block gives the
/// block update too little to do — at `nb = 8` the parallel path's tasks are
/// smaller than the fork/join that dispatches them. Measured on this machine:
/// widening 8 → 32 took `gmm`'s 1021-column supernode from 15.7 to 7.9 ms and
/// simultaneously took gridfit 320², which is thousands of small supernodes,
/// from 67.8 to 102.7. Hence the split rather than a constant.
#[inline]
fn panel_width(n: usize) -> usize {
    if n >= 256 {
        POTRF_NB
    } else {
        8
    }
}

fn potf2_panel(m: usize, n: usize, a: &mut Ws<f64>, off: usize, lda: usize) -> i64 {
    for j in 0..n {
        /* ajj = A (j,j) - A (j, 0:j-1) * A (j, 0:j-1)' */
        let mut ajj = a[off + j + j * lda];
        for l in 0..j {
            let x = a[off + j + l * lda];
            ajj = rfma(-x, x, ajj);
        }
        if !(ajj > 0.0) {
            /* also catches NaN, as dpotf2's `.LE. ZERO .OR. DISNAN` does */
            a[off + j + j * lda] = ajj;
            return j as i64 + 1;
        }
        let ajj = ajj.sqrt();
        a[off + j + j * lda] = ajj;

        if j + 1 < m {
            /* A (j+1:m-1, j) -= A (j+1:m-1, 0:j-1) * A (j, 0:j-1)' */
            let (mm, yo) = (m - (j + 1), off + (j + 1) + j * lda);
            let mut l = 0;
            while l + 4 <= j {
                let t = [
                    a[off + j + l * lda],
                    a[off + j + (l + 1) * lda],
                    a[off + j + (l + 2) * lda],
                    a[off + j + (l + 3) * lda],
                ];
                let xo = [
                    off + (j + 1) + l * lda,
                    off + (j + 1) + (l + 1) * lda,
                    off + (j + 1) + (l + 2) * lda,
                    off + (j + 1) + (l + 3) * lda,
                ];
                strip_sub::<4>(mm, a, yo, &t, &xo);
                l += 4;
            }
            while l < j {
                strip_sub::<1>(
                    mm,
                    a,
                    yo,
                    &[a[off + j + l * lda]],
                    &[off + (j + 1) + l * lda],
                );
                l += 1;
            }
            /* A (j+1:m-1, j) /= ajj */
            let r = 1.0 / ajj;
            for i in j + 1..m {
                a[off + i + j * lda] *= r;
            }
        }
    }
    0
}

pub fn trsm_rlt(m: usize, n: usize, x: &mut [f64], ld: usize, nt: usize) {
    debug_assert!(n == 0 || x.len() >= (n - 1) * ld + n + m);
    #[cfg(vendor_blas)]
    if super::blas::worth_it(m as f64 * (n as f64).powi(2)) {
        let _ = nt;
        return super::blas::trsm_rlt(m, n, x, ld);
    }
    let x = Ws::new(x);

    /* B (:, j) = (B (:, j) - B (:, 0:j-1) * A (j, 0:j-1)') / A (j,j)
     *
     * The reference skips a source column when `A (j,l)` is exactly zero.
     * That is not done here: it is a data-dependent branch in the hot loop for
     * a case the supernodal factorization does not produce, and it changes
     * nothing numerically unless the other operand is an infinity or a NaN. */
    let wide = panel_width(n);
    let mut j0 = 0;
    while j0 < n {
        let nb = wide.min(n - j0);
        if j0 > 0 {
            /* B (:, j0:j0+nb-1) -= B (:, 0:j0-1) * A (j0:j0+nb-1, 0:j0-1)' */
            if nt > 1 && 2.0 * (m * nb * j0) as f64 >= GEMM_PAR_FLOPS {
                gemm_sub_par(m, nb, j0, x, n, ld, j0, ld, n + j0 * ld, ld, nt);
            } else {
                gemm_sub(m, nb, j0, x, n, ld, j0, ld, n + j0 * ld, ld);
            }
        }
        for j in j0..j0 + nb {
            let yo = n + j * ld;
            let mut l = j0;
            while l + 4 <= j {
                let t = [
                    x[j + l * ld],
                    x[j + (l + 1) * ld],
                    x[j + (l + 2) * ld],
                    x[j + (l + 3) * ld],
                ];
                let xo = [
                    n + l * ld,
                    n + (l + 1) * ld,
                    n + (l + 2) * ld,
                    n + (l + 3) * ld,
                ];
                strip_sub::<4>(m, x, yo, &t, &xo);
                l += 4;
            }
            while l < j {
                strip_sub::<1>(m, x, yo, &[x[j + l * ld]], &[n + l * ld]);
                l += 1;
            }
            let r = 1.0 / x[j + j * ld];
            for i in 0..m {
                x[n + i + j * ld] *= r;
            }
        }
        j0 += nb;
    }
}

/* ========================================================================= */
/* === the solve kernels =================================================== */
/* ========================================================================= */

/// `x := L^{-1} x` — `dtrsv ("L", "N", "N", n, a, lda, x, 1)`.
///
/// `a` is `n`-by-`n` lower triangular with a non-unit diagonal; only its lower
/// triangle is read. Forward substitution in *axpy* form, as `dtrsv.f:151-160`
/// does it: divide, then push the solved entry down the column.
pub fn trsv_ln(n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    debug_assert!(n == 0 || (a.len() >= (n - 1) * lda + n && x.len() >= n));
    #[cfg(vendor_blas)]
    if super::blas::worth_it((n as f64).powi(2)) {
        return super::blas::trsv_ln(n, a, lda, x);
    }
    let a = Ws::new_ref(a);
    let x = Ws::new(x);

    for j in 0..n {
        let t = x[j] / a[j + j * lda];
        x[j] = t;
        for i in j + 1..n {
            x[i] = rfma(-t, a[i + j * lda], x[i]);
        }
    }
}

/// `x := L^{-T} x` — `dtrsv ("L", "C", "N", n, a, lda, x, 1)`.
///
/// Back-substitution in *dot* form, as `dtrsv.f:194-203` does it — and note
/// the inner loop runs **`i` descending** from `n-1` to `j+1`, which is the
/// reference's order and not the one [`trsm_llt`] uses for the same solve.
pub fn trsv_lt(n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    debug_assert!(n == 0 || (a.len() >= (n - 1) * lda + n && x.len() >= n));
    #[cfg(vendor_blas)]
    if super::blas::worth_it((n as f64).powi(2)) {
        return super::blas::trsv_lt(n, a, lda, x);
    }
    let a = Ws::new_ref(a);
    let x = Ws::new(x);

    for j in (0..n).rev() {
        let mut t = x[j];
        for i in (j + 1..n).rev() {
            t = rfma(-a[i + j * lda], x[i], t);
        }
        x[j] = t / a[j + j * lda];
    }
}

/// `y := y - A x` — `dgemv ("N", m, n, -1.0, a, lda, x, 1, 1.0, y, 1)`.
///
/// `a` is `m`-by-`n`, `x` has `n` entries and `y` has `m`. Column-wise axpy
/// (`dgemv.f:222-232`); `beta` is 1, so `y` is accumulated into rather than
/// overwritten. The reference forms `TEMP = ALPHA*X(J)` and adds — negating
/// first and adding is bit-identical to subtracting, since a sign flip is
/// exact.
/// Blocked in `n`: the reference re-reads and re-writes the whole of `y` once
/// per column, and taking `NR` columns together cuts that traffic by `NR`.
/// `y(i)` still takes its updates in `j`-ascending order, so the answer is
/// unchanged bit for bit — the same argument [`strip_sub`] rests on.
pub fn gemv_n(m: usize, n: usize, a: &[f64], lda: usize, x: &[f64], y: &mut [f64]) {
    debug_assert!(n == 0 || (a.len() >= (n - 1) * lda + m && x.len() >= n));
    debug_assert!(y.len() >= m);
    #[cfg(vendor_blas)]
    if super::blas::worth_it(2.0 * m as f64 * n as f64) {
        return super::blas::gemv_n(m, n, a, lda, x, y);
    }
    let (a, x) = (Ws::new_ref(a), Ws::new_ref(x));
    let y = Ws::new(y);

    let mut j = 0;
    while j + 4 <= n {
        gemv_n_nb::<4>(m, a, j, lda, x, y);
        j += 4;
    }
    if j + 2 <= n {
        gemv_n_nb::<2>(m, a, j, lda, x, y);
        j += 2;
    }
    while j < n {
        gemv_n_nb::<1>(m, a, j, lda, x, y);
        j += 1;
    }
}

#[inline(always)]
fn gemv_n_nb<const NR: usize>(
    m: usize,
    a: &Ws<f64>,
    j0: usize,
    lda: usize,
    x: &Ws<f64>,
    y: &mut Ws<f64>,
) {
    let mut t = [0.0f64; NR];
    for (jj, tj) in t.iter_mut().enumerate() {
        *tj = x[j0 + jj];
    }
    const IB: usize = 8;
    let mut i = 0;
    while i + IB <= m {
        let mut v = [0.0f64; IB];
        for (ii, vv) in v.iter_mut().enumerate() {
            *vv = y[i + ii];
        }
        for (jj, &tj) in t.iter().enumerate() {
            let ao = i + (j0 + jj) * lda;
            for (ii, vv) in v.iter_mut().enumerate() {
                *vv = rfma(-tj, a[ao + ii], *vv);
            }
        }
        for (ii, &vv) in v.iter().enumerate() {
            y[i + ii] = vv;
        }
        i += IB;
    }
    while i < m {
        let mut v = y[i];
        for (jj, &tj) in t.iter().enumerate() {
            v = rfma(-tj, a[i + (j0 + jj) * lda], v);
        }
        y[i] = v;
        i += 1;
    }
}

/// `y := y - A' x` — `dgemv ("C", m, n, -1.0, a, lda, x, 1, 1.0, y, 1)`.
///
/// `a` is `m`-by-`n`, so `x` has `m` entries and `y` has `n`: the transpose
/// swaps which of the two the leading dimension belongs to. One dot product
/// per column, `i` ascending, combined into `y` once at the end
/// (`dgemv.f:255-264`).
/// Blocked in `n` for the same reason [`gemv_n`] is, except that here it is `x`
/// that is re-read per column rather than `y` that is re-written. Each column's
/// dot product still runs `i` ascending on its own accumulator.
pub fn gemv_t(m: usize, n: usize, a: &[f64], lda: usize, x: &[f64], y: &mut [f64]) {
    debug_assert!(n == 0 || a.len() >= (n - 1) * lda + m);
    debug_assert!(x.len() >= m && y.len() >= n);
    #[cfg(vendor_blas)]
    if super::blas::worth_it(2.0 * m as f64 * n as f64) {
        return super::blas::gemv_t(m, n, a, lda, x, y);
    }
    let (a, x) = (Ws::new_ref(a), Ws::new_ref(x));
    let y = Ws::new(y);

    let mut j = 0;
    while j + 4 <= n {
        gemv_t_nb::<4>(m, a, j, lda, x, y);
        j += 4;
    }
    if j + 2 <= n {
        gemv_t_nb::<2>(m, a, j, lda, x, y);
        j += 2;
    }
    while j < n {
        gemv_t_nb::<1>(m, a, j, lda, x, y);
        j += 1;
    }
}

#[inline(always)]
fn gemv_t_nb<const NR: usize>(
    m: usize,
    a: &Ws<f64>,
    j0: usize,
    lda: usize,
    x: &Ws<f64>,
    y: &mut Ws<f64>,
) {
    let mut acc = [0.0f64; NR];
    for i in 0..m {
        let xv = x[i];
        for (jj, v) in acc.iter_mut().enumerate() {
            *v = rfma(a[i + (j0 + jj) * lda], xv, *v);
        }
    }
    for (jj, &v) in acc.iter().enumerate() {
        y[j0 + jj] -= v;
    }
}

/// `B := L^{-1} B` — `dtrsm ("L", "L", "N", "N", m, n, 1.0, a, lda, b, ldb)`.
///
/// `a` is `m`-by-`m` lower triangular with a non-unit diagonal, `b` is
/// `m`-by-`n`. Each right-hand side is the axpy forward substitution
/// [`trsv_ln`] does (`dtrsm.f:307-320`).
///
/// The reference nests `j` outside `k`; this hoists `j` inside, so one pass
/// over `a` serves all `NR` right-hand sides. The right-hand sides are
/// independent, so each `(i,j)` still takes its updates in `k`-ascending order
/// and the answer is unchanged bit for bit. Note the diagonal is a *division*
/// per entry, not a reciprocal multiply — `dtrsm ("L", …)` divides where
/// `dtrsm ("R", …)` forms `ONE/A(K,K)` first, and [`trsm_rlt`] follows the
/// other one.
pub fn trsm_lln(m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
    debug_assert!(m == 0 || a.len() >= (m - 1) * lda + m);
    debug_assert!(n == 0 || m == 0 || b.len() >= (n - 1) * ldb + m);
    #[cfg(vendor_blas)]
    if super::blas::worth_it((m as f64).powi(2) * n as f64) {
        return super::blas::trsm_lln(m, n, a, lda, b, ldb);
    }
    let a = Ws::new_ref(a);
    let b = Ws::new(b);

    let mut j = 0;
    while j + 4 <= n {
        trsm_lln_nb::<4>(m, a, lda, b, j, ldb);
        j += 4;
    }
    if j + 2 <= n {
        trsm_lln_nb::<2>(m, a, lda, b, j, ldb);
        j += 2;
    }
    while j < n {
        trsm_lln_nb::<1>(m, a, lda, b, j, ldb);
        j += 1;
    }
}

#[inline(always)]
fn trsm_lln_nb<const NR: usize>(
    m: usize,
    a: &Ws<f64>,
    lda: usize,
    b: &mut Ws<f64>,
    j0: usize,
    ldb: usize,
) {
    for k in 0..m {
        let akk = a[k + k * lda];
        let mut t = [0.0f64; NR];
        for (jj, tj) in t.iter_mut().enumerate() {
            let v = b[k + (j0 + jj) * ldb] / akk;
            b[k + (j0 + jj) * ldb] = v;
            *tj = v;
        }
        for i in k + 1..m {
            let aik = a[i + k * lda];
            for (jj, &tj) in t.iter().enumerate() {
                b[i + (j0 + jj) * ldb] = rfma(-tj, aik, b[i + (j0 + jj) * ldb]);
            }
        }
    }
}

/// `B := L^{-T} B` — `dtrsm ("L", "L", "C", "N", m, n, 1.0, a, lda, b, ldb)`.
///
/// The dot-form back substitution of `dtrsm.f:343-356`. Its inner loop runs
/// **`k` ascending**, where [`trsv_lt`]'s runs descending: netlib disagrees
/// with itself here, and since upstream picks between the two on `nrhs == 1`,
/// so must this.
///
/// `j` is hoisted inside `i` for the reason [`trsm_lln`] gives — one pass over
/// `a` per `NR` right-hand sides, with each `(i,j)`'s summation order intact.
pub fn trsm_llt(m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
    debug_assert!(m == 0 || a.len() >= (m - 1) * lda + m);
    debug_assert!(n == 0 || m == 0 || b.len() >= (n - 1) * ldb + m);
    #[cfg(vendor_blas)]
    if super::blas::worth_it((m as f64).powi(2) * n as f64) {
        return super::blas::trsm_llt(m, n, a, lda, b, ldb);
    }
    let a = Ws::new_ref(a);
    let b = Ws::new(b);

    let mut j = 0;
    while j + 4 <= n {
        trsm_llt_nb::<4>(m, a, lda, b, j, ldb);
        j += 4;
    }
    if j + 2 <= n {
        trsm_llt_nb::<2>(m, a, lda, b, j, ldb);
        j += 2;
    }
    while j < n {
        trsm_llt_nb::<1>(m, a, lda, b, j, ldb);
        j += 1;
    }
}

#[inline(always)]
fn trsm_llt_nb<const NR: usize>(
    m: usize,
    a: &Ws<f64>,
    lda: usize,
    b: &mut Ws<f64>,
    j0: usize,
    ldb: usize,
) {
    for i in (0..m).rev() {
        let mut t = [0.0f64; NR];
        for (jj, tj) in t.iter_mut().enumerate() {
            *tj = b[i + (j0 + jj) * ldb];
        }
        for k in i + 1..m {
            let aki = a[k + i * lda];
            for (jj, tj) in t.iter_mut().enumerate() {
                *tj = rfma(-aki, b[k + (j0 + jj) * ldb], *tj);
            }
        }
        let aii = a[i + i * lda];
        for (jj, &tj) in t.iter().enumerate() {
            b[i + (j0 + jj) * ldb] = tj / aii;
        }
    }
}

/// `C := C - A B` — `dgemm ("N", "N", m, n, k, -1.0, a, lda, b, ldb, 1.0, c,
/// ldc)`.
///
/// `a` is `m`-by-`k`, `b` is `k`-by-`n`, `c` is `m`-by-`n`.
///
/// `beta` is 1, so unlike [`gemm_nt`] this accumulates into whatever `c`
/// already holds. The register-blocked tile therefore *loads* `c` before the
/// `l` loop instead of starting from zero — which keeps the rounding sequence
/// `((c - t0 a0) - t1 a1) - …` exactly the reference's (`dgemm.f:250-262`),
/// where starting from zero and adding `c` at the end would not.
///
/// Blocked in `n` as well as `m`, which is the point: `n` is the number of
/// right-hand sides, and `A` is the supernode — the big operand. A tile that is
/// `NR` columns wide reads `A` once for all `NR` instead of once each, and
/// that ratio is what a solve of several right-hand sides is limited by.
pub fn gemm_nn(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    debug_assert!(k == 0 || (a.len() >= (k - 1) * lda + m && b.len() >= (n - 1) * ldb + k));
    debug_assert!(n == 0 || c.len() >= (n - 1) * ldc + m);
    #[cfg(vendor_blas)]
    if super::blas::worth_it(2.0 * m as f64 * n as f64 * k as f64) {
        return super::blas::gemm_nn(m, n, k, a, lda, b, ldb, c, ldc);
    }
    let (a, b) = (Ws::new_ref(a), Ws::new_ref(b));
    let c = Ws::new(c);

    let mut j = 0;
    while j + 4 <= n {
        cols_nn::<4>(m, k, a, lda, b, j, ldb, c, ldc);
        j += 4;
    }
    if j + 2 <= n {
        cols_nn::<2>(m, k, a, lda, b, j, ldb, c, ldc);
        j += 2;
    }
    if j < n {
        cols_nn::<1>(m, k, a, lda, b, j, ldb, c, ldc);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_nn<const MR: usize, const NR: usize>(
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    j0: usize,
    ldb: usize,
    c: &mut Ws<f64>,
    ic: usize,
    ldc: usize,
) {
    let mut acc = [[0.0f64; MR]; NR];
    for (jj, accj) in acc.iter_mut().enumerate() {
        for (ii, v) in accj.iter_mut().enumerate() {
            *v = c[ic + ii + jj * ldc];
        }
    }
    for l in 0..k {
        let ao = ia + l * lda;
        for (jj, accj) in acc.iter_mut().enumerate() {
            let t = b[l + (j0 + jj) * ldb];
            for (ii, v) in accj.iter_mut().enumerate() {
                *v = rfma(-t, a[ao + ii], *v);
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &v) in accj.iter().enumerate() {
            c[ic + ii + jj * ldc] = v;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cols_nn<const NR: usize>(
    m: usize,
    k: usize,
    a: &Ws<f64>,
    lda: usize,
    b: &Ws<f64>,
    j0: usize,
    ldb: usize,
    c: &mut Ws<f64>,
    ldc: usize,
) {
    let ic0 = j0 * ldc;
    let mut i = 0;
    while i + 8 <= m {
        tile_nn::<8, NR>(k, a, i, lda, b, j0, ldb, c, ic0 + i, ldc);
        i += 8;
    }
    if i + 4 <= m {
        tile_nn::<4, NR>(k, a, i, lda, b, j0, ldb, c, ic0 + i, ldc);
        i += 4;
    }
    if i + 2 <= m {
        tile_nn::<2, NR>(k, a, i, lda, b, j0, ldb, c, ic0 + i, ldc);
        i += 2;
    }
    if i < m {
        tile_nn::<1, NR>(k, a, i, lda, b, j0, ldb, c, ic0 + i, ldc);
    }
}

/// `C := C - A' B` — `dgemm ("C", "N", m, n, k, -1.0, a, lda, b, ldb, 1.0, c,
/// ldc)`.
///
/// `a` is `k`-by-`m` — the transpose swaps its two extents — `b` is
/// `k`-by-`n`, `c` is `m`-by-`n`.
///
/// The reference accumulates each dot product from zero and combines it with
/// `c` once, at the end (`dgemm.f:279-290`), so the accumulators here start at
/// zero rather than at `c` as [`gemm_nn`]'s do. Blocking in both directions
/// keeps that: each `(i,j)` is its own running sum over `l` ascending.
pub fn gemm_tn(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    debug_assert!(m == 0 || k == 0 || a.len() >= (m - 1) * lda + k);
    debug_assert!(n == 0 || k == 0 || b.len() >= (n - 1) * ldb + k);
    debug_assert!(n == 0 || c.len() >= (n - 1) * ldc + m);
    #[cfg(vendor_blas)]
    if super::blas::worth_it(2.0 * m as f64 * n as f64 * k as f64) {
        return super::blas::gemm_tn(m, n, k, a, lda, b, ldb, c, ldc);
    }
    let (a, b) = (Ws::new_ref(a), Ws::new_ref(b));
    let c = Ws::new(c);

    let mut j = 0;
    while j + 4 <= n {
        cols_tn::<4>(m, k, a, lda, b, j, ldb, c, ldc);
        j += 4;
    }
    if j + 2 <= n {
        cols_tn::<2>(m, k, a, lda, b, j, ldb, c, ldc);
        j += 2;
    }
    if j < n {
        cols_tn::<1>(m, k, a, lda, b, j, ldb, c, ldc);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_tn<const MR: usize, const NR: usize>(
    k: usize,
    a: &Ws<f64>,
    i0: usize,
    lda: usize,
    b: &Ws<f64>,
    j0: usize,
    ldb: usize,
    c: &mut Ws<f64>,
    ic: usize,
    ldc: usize,
) {
    let mut acc = [[0.0f64; MR]; NR];
    for l in 0..k {
        let mut av = [0.0f64; MR];
        for (ii, v) in av.iter_mut().enumerate() {
            *v = a[l + (i0 + ii) * lda];
        }
        for (jj, accj) in acc.iter_mut().enumerate() {
            let bv = b[l + (j0 + jj) * ldb];
            for (ii, v) in accj.iter_mut().enumerate() {
                *v = rfma(av[ii], bv, *v);
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &v) in accj.iter().enumerate() {
            c[ic + ii + jj * ldc] -= v;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cols_tn<const NR: usize>(
    m: usize,
    k: usize,
    a: &Ws<f64>,
    lda: usize,
    b: &Ws<f64>,
    j0: usize,
    ldb: usize,
    c: &mut Ws<f64>,
    ldc: usize,
) {
    let ic0 = j0 * ldc;
    let mut i = 0;
    while i + 4 <= m {
        tile_tn::<4, NR>(k, a, i, lda, b, j0, ldb, c, ic0 + i, ldc);
        i += 4;
    }
    if i + 2 <= m {
        tile_tn::<2, NR>(k, a, i, lda, b, j0, ldb, c, ic0 + i, ldc);
        i += 2;
    }
    if i < m {
        tile_tn::<1, NR>(k, a, i, lda, b, j0, ldb, c, ic0 + i, ldc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, ld: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut v = vec![f64::NAN; ld * cols + rows];
        for j in 0..cols {
            for i in 0..rows {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v[i + j * ld] = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            }
        }
        v
    }

    fn spd(n: usize, ld: usize, seed: u64) -> Vec<f64> {
        let a = mat(n, n, n, seed);
        let mut m = vec![f64::NAN; ld * n + n];
        for j in 0..n {
            for i in 0..n {
                let mut acc = if i == j { n as f64 } else { 0.0 };
                for l in 0..n {
                    acc += a[i + l * n] * a[j + l * n];
                }
                m[i + j * ld] = acc;
            }
        }
        m
    }

    #[test]
    fn syrk_computes_the_lower_triangle_of_a_gram_matrix() {
        for &(n, k, lda, ldc) in &[(1usize, 1usize, 3usize, 2usize), (5, 3, 9, 7), (8, 8, 8, 8)] {
            let a = mat(n, k, lda, 12345);
            let mut c = vec![f64::NAN; ldc * n + n];
            syrk_ln(n, k, &a, lda, &mut c, ldc);
            for j in 0..n {
                for i in j..n {
                    let want: f64 = (0..k).map(|l| a[i + l * lda] * a[j + l * lda]).sum();
                    assert!(
                        (c[i + j * ldc] - want).abs() < 1e-13,
                        "n={n} k={k} ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_multiplies_by_a_transpose() {
        for &(m, n, k) in &[(1usize, 1usize, 1usize), (4, 6, 3), (7, 2, 5)] {
            let (lda, ldb, ldc) = (m + 2, n + 3, m + 1);
            let a = mat(m, k, lda, 777);
            let b = mat(n, k, ldb, 999);
            let mut c = vec![f64::NAN; ldc * n + m];
            gemm_nt(m, n, k, &a, lda, &b, ldb, &mut c, ldc);
            for j in 0..n {
                for i in 0..m {
                    let want: f64 = (0..k).map(|l| a[i + l * lda] * b[j + l * ldb]).sum();
                    assert!((c[i + j * ldc] - want).abs() < 1e-13, "({i},{j})");
                }
            }
        }
    }

    #[cfg(not(vendor_blas))]
    fn dot_nt(k: usize, a: &[f64], ia: usize, lda: usize, b: &[f64], jb: usize, ldb: usize) -> f64 {
        let mut acc = 0.0f64;
        for l in 0..k {
            acc = rfma(b[jb + l * ldb], a[ia + l * lda], acc);
        }
        acc
    }

    #[cfg(not(vendor_blas))]
    #[test]
    fn gemm_is_bit_identical_across_the_blocking() {
        for &(m, n, k) in &[
            (64usize, 33usize, 70usize),
            (PACK_MC + 3, PACK_NR * 2 + 1, PACK_KC + 5),
            (PACK_MC * 2 + 7, 70, PACK_KC * 2 + 1),
            (37, PACK_NC + 6, 90),
        ] {
            let (lda, ldb, ldc) = (m + 5, n + 7, m + 3);
            let a = mat(m, k, lda, 4242);
            let b = mat(n, k, ldb, 2424);
            let mut got = vec![f64::NAN; ldc * n + m];
            gemm_nt(m, n, k, &a, lda, &b, ldb, &mut got, ldc);
            for j in 0..n {
                for i in 0..m {
                    assert_eq!(
                        got[i + j * ldc].to_bits(),
                        dot_nt(k, &a, i, lda, &b, j, ldb).to_bits(),
                        "m={m} n={n} k={k} at ({i},{j})"
                    );
                }
            }
        }
    }

    #[cfg(not(vendor_blas))]
    #[test]
    fn syrk_is_bit_identical_across_the_blocking() {
        for &(n, k) in &[
            (8usize, 5usize),
            (67, 40),
            (200, 90),
            (SYRK_LEAF * 4 + 8, 70),
        ] {
            let (lda, ldc) = (n + 4, n + 2);
            let a = mat(n, k, lda, 31337);
            for &jn in &[n, SYRK_NB, SYRK_LEAF, SYRK_LEAF + SYRK_NB] {
                let mut c = vec![f64::NAN; ldc * n + n];
                let mut j0 = 0;
                while j0 < n {
                    let w = jn.min(n - j0);
                    syrk_ln_strip(n, k, &a, lda, &mut c[j0 * ldc..], ldc, j0, w);
                    j0 += w;
                }
                for j in 0..n {
                    for i in j..n {
                        assert_eq!(
                            c[i + j * ldc].to_bits(),
                            dot_nt(k, &a, i, lda, &a, j, lda).to_bits(),
                            "n={n} k={k} jn={jn} at ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    #[cfg(not(vendor_blas))]
    #[test]
    fn packed_gemm_is_bit_identical_to_the_direct_one() {
        for &(m, n, k) in &[
            (64usize, 33usize, 70usize),
            (300, 17, 300),
            (PACK_MC + 3, PACK_NR * 2 + 1, PACK_KC + 5),
            (PACK_MC * 2 + 7, 70, PACK_KC * 2 + 1),
            (37, PACK_NC + 6, 90),
            (PACK_MR * 4, PACK_NR * 2, 64),
        ] {
            let (lda, ldb, ldc) = (m + 5, n + 7, m + 3);
            let a = mat(m, k, lda, 4242);
            let b = mat(n, k, ldb, 2424);
            let mut want = vec![f64::NAN; ldc * n + m];
            let mut got = vec![f64::NAN; ldc * n + m];
            gemm_nt_direct(m, n, k, &a, lda, &b, ldb, &mut want, ldc);
            gemm_nt_packed(m, n, k, &a, lda, &b, ldb, &mut got, ldc);
            for j in 0..n {
                for i in 0..m {
                    assert_eq!(
                        got[i + j * ldc].to_bits(),
                        want[i + j * ldc].to_bits(),
                        "m={m} n={n} k={k} at ({i},{j})"
                    );
                }
            }
        }
    }

    #[cfg(not(vendor_blas))]
    #[test]
    fn packed_gemm_sub_is_bit_identical_to_the_direct_one() {
        for &(m, n, k) in &[
            (64usize, 32usize, 70usize),
            (300, 8, 300),
            (PACK_MC + 3, PACK_NR * 2 + 1, PACK_KC + 5),
            (PACK_MC * 2 + 7, 33, PACK_KC * 2 + 1),
            (PACK_MR * 4, PACK_NR * 2, 64),
        ] {
            let ld = m + k + 6;
            let base = mat(ld, k + n, ld, 1234);
            /* A at row 0, B at row `m`, both over columns `0..k`; C at row 0 of
             * columns `k..k+n`, which no source column touches */
            let (ia, jb, ic) = (0usize, m, k * ld);
            let mut want = base.clone();
            let mut got = base.clone();
            gemm_sub_direct(m, n, k, Ws::new(&mut want), ia, ld, jb, ld, ic, ld);
            gemm_sub_packed(m, n, k, Ws::new(&mut got), ia, ld, jb, ld, ic, ld);
            for (q, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "m={m} n={n} k={k} at {q}");
            }
        }
    }

    #[cfg(not(vendor_blas))]
    #[test]
    fn the_aliasing_dispatch_does_not_move_a_rounding() {
        for &(m, n, k, ld) in &[
            (64usize, 33usize, 70usize, 512usize),
            (200, 64, 90, 1024),
            (300, 17, 40, 2048),
            (PACK_MR * 4, PACK_NR * 4, L1_WAYS + 1, 512),
        ] {
            assert!(
                strides_want_packing(m, n, k, ld, ld),
                "m={m} n={n} k={k} ld={ld}"
            );
            let a = mat(m, k, ld, 4242);
            let b = mat(n, k, ld, 2424);
            let ldc = m + 3;
            let mut got = vec![f64::NAN; ldc * n + m];
            gemm_nt(m, n, k, &a, ld, &b, ld, &mut got, ldc);
            for j in 0..n {
                for i in 0..m {
                    assert_eq!(
                        got[i + j * ldc].to_bits(),
                        dot_nt(k, &a, i, ld, &b, j, ld).to_bits(),
                        "m={m} n={n} k={k} ld={ld} at ({i},{j})"
                    );
                }
            }

            /* the same shapes through `C -= A B'`, laid out as its callers do:
             * one array whose *own* leading dimension is the aliasing one, `A`
             * at row 0 and `B` at row `m` of columns `0..k`, `C` in the panel */
            assert!(m + n <= ld);
            let base = mat(ld, k + n, ld, 1234);
            let (ia, jb, ic) = (0usize, m, k * ld);
            let mut want = base.clone();
            let mut got = base.clone();
            gemm_sub_direct(m, n, k, Ws::new(&mut want), ia, ld, jb, ld, ic, ld);
            gemm_sub(m, n, k, Ws::new(&mut got), ia, ld, jb, ld, ic, ld);
            for (q, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "m={m} n={n} k={k} ld={ld} at {q}");
            }
        }
    }

    #[cfg(not(vendor_blas))]
    #[test]
    fn the_packed_scattered_block_matches_the_direct_one() {
        for &(m, n, k, ld) in &[
            (64usize, 16usize, 20usize, 0usize),
            (40, PACK_NR * 4 + 2, 9, 0),
            (PACK_MC + 3, 33, PACK_KC + 5, 0),
            (512, 32, 16, 2048), /* an aliasing stride, which is the live case */
        ] {
            let ld = if ld == 0 { m + n + k + 6 } else { ld };
            let base = mat(ld, k + n, ld, 5150);
            let (ia, jb, ic) = (0usize, m, k * ld);
            let mut out = [base.clone(), base.clone()];
            for (q, x) in out.iter_mut().enumerate() {
                let (src, dst) = Ws::new(x).split_at_mut(ic);
                let mut cols: Vec<&mut [f64]> = Vec::with_capacity(n);
                let mut rest = dst;
                for j in 0..n {
                    let col = if j + 1 < n {
                        let (c, tail) = rest.split_at_mut(ld);
                        rest = tail;
                        c
                    } else {
                        std::mem::take(&mut rest)
                    };
                    cols.push(&mut col[..m]);
                }
                if q == 0 {
                    block_scat_direct(k, src, ia, ld, src, jb, ld, &mut cols);
                } else {
                    block_scat_packed(k, src, ia, ld, src, jb, ld, &mut cols);
                }
            }
            for (q, (&g, &w)) in out[1].iter().zip(out[0].iter()).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "m={m} n={n} k={k} ld={ld} at {q}");
            }
        }
    }

    #[cfg(not(vendor_blas))]
    #[test]
    fn the_parallel_update_packs_like_the_serial_one_at_an_aliasing_stride() {
        let (m, n, k, ld, nt) = (512usize, 32usize, 16usize, 2048usize, 2usize);
        let rows = m.div_ceil(nt * PAR_OVER).max(8);
        assert!(strides_want_packing(rows, n, k, ld, ld));
        let base = mat(ld, k + n, ld, 90210);
        let (ia, jb, ic) = (0usize, m, k * ld);
        let mut serial = base.clone();
        let mut wide = base.clone();
        gemm_sub(m, n, k, Ws::new(&mut serial), ia, ld, jb, ld, ic, ld);
        gemm_sub_par(m, n, k, Ws::new(&mut wide), ia, ld, jb, ld, ic, ld, nt);
        for (q, (&g, &w)) in wide.iter().zip(serial.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "at {q}");
        }
    }

    #[test]
    fn the_packing_bounds_sit_where_the_gain_crosses_one() {
        let al = 2048; /* aliases */
        let no = 2052; /* does not */
        /* `k`: 1.03 at eight columns, 1.15 at nine — a walk with no more
         * columns than the cache has ways cannot thrash, however they collide */
        assert!(!strides_want_packing(512, 128, L1_WAYS, al, al));
        assert!(strides_want_packing(512, 128, L1_WAYS + 1, al, al));
        /* `n`: 0.83 at eight columns of `C`, 1.08 at sixteen */
        assert!(!strides_want_packing(512, PACK_NR * 2, 64, al, al));
        assert!(strides_want_packing(512, PACK_NR * 4, 64, al, al));
        /* `m`: 0.81 at sixteen rows, 1.03 at thirty-two */
        assert!(!strides_want_packing(PACK_MR * 2, 128, 64, al, al));
        assert!(strides_want_packing(PACK_MR * 4, 128, 64, al, al));
        /* either operand alone is enough: `A` costs 10-25% and `B` 27-41% */
        assert!(!strides_want_packing(512, 128, 64, no, no));
        assert!(strides_want_packing(512, 128, 64, al, no));
        assert!(strides_want_packing(512, 128, 64, no, al));
    }

    /// The residency arm's `n` floor is the same crossing, and it has to be:
    /// both arms make the same copy, so both amortise it over the same `n /
    /// PACK_NR` re-streams. It was `PACK_NR * 2` for one release, where the
    /// copy cannot win however large the operand — and eight threads issue
    /// exactly that shape, because a narrow strip's rectangle has `n` equal to
    /// the strip width. Timed on the calls themselves, 13.87 → 10.91 ms.
    #[test]
    fn the_residency_arm_amortises_its_copy() {
        let big = 1024; /* `m` and `k` chosen so the byte test never decides */
        assert!(!wants_packing(big, PACK_NR * 2, big));
        assert!(wants_packing(big, PACK_NR * 4, big));
        /* and the byte test still has to pass on its own */
        assert!(!wants_packing(big, PACK_NR * 4, 64));
        /* the two arms agree on both shape floors */
        for n in [PACK_NR * 2, PACK_NR * 4] {
            assert_eq!(
                wants_packing(big, n, big),
                strides_want_packing(big, n, big, 2048, 2048),
                "the arms disagree at n={n}"
            );
        }
        for m in [PACK_MR * 2, PACK_MR * 4] {
            assert_eq!(
                wants_packing(m, 128, big),
                strides_want_packing(m, 128, big, 2048, 2048),
                "the arms disagree at m={m}"
            );
        }
    }

    #[test]
    fn the_aliasing_test_agrees_with_the_measured_cliff() {
        /* direct path 13.7-38.3 GFLOP/s, packed 38.4-44.5 */
        for &ld in &[2048usize, 2049, 2050, 2731, 3072, 4096] {
            assert!(strides_alias(ld), "lda={ld} should pack");
        }
        /* direct path 50.3-56.8, packed 42.0-47.3 */
        for &ld in &[
            1021usize, 1219, 1239, 1523, 2052, 2056, 2176, 2221, 2304, 2500, 2600, 2732, 2816,
            3000, 3021, 4174, 5344,
        ] {
            assert!(!strides_alias(ld), "lda={ld} should not pack");
        }
    }

    #[test]
    fn potrf_factors_and_reproduces_the_matrix() {
        for &n in &[1usize, 2, 5, 16] {
            let ld = n + 3;
            let m = spd(n, ld, 4242);
            let mut a = m.clone();
            assert_eq!(potrf_l(n, &mut a, ld, 1), 0, "n={n}");
            for j in 0..n {
                for i in j..n {
                    let want: f64 = (0..=j.min(i)).map(|l| a[i + l * ld] * a[j + l * ld]).sum();
                    assert!((want - m[i + j * ld]).abs() < 1e-10, "n={n} ({i},{j})");
                }
            }
        }
    }

    fn block(x: &[f64], m: usize, n: usize, ld: usize) -> Vec<f64> {
        (0..n)
            .flat_map(|j| x[j * ld..j * ld + m].iter().copied())
            .collect()
    }

    #[test]
    fn the_parallel_panel_update_is_bit_identical() {
        for &n in &[9usize, 33, 64, 130, 257] {
            for &ld in &[n, n + 5] {
                let m = spd(n, ld, 20260805);

                let (mut serial, mut wide) = (m.clone(), m.clone());
                assert_eq!(potrf_l(n, &mut serial, ld, 1), 0, "n={n}");
                assert_eq!(potrf_l(n, &mut wide, ld, 8), 0, "n={n}");
                assert_eq!(
                    block(&serial, n, n, ld),
                    block(&wide, n, n, ld),
                    "potrf_l moved a bit at n={n} ld={ld}"
                );

                /* trsm_rlt's own panel update, on the rows below the block:
                 * one array holding the factored `cols`-by-`cols` triangle and
                 * the `rows` rows under it, exactly as a supernode does */
                let cols = n.min(48);
                let (rows, nsrow) = (n, n + cols);
                let mut x = spd(nsrow, nsrow, 777);
                assert_eq!(potrf_l(cols, &mut x, nsrow, 1), 0);
                let (mut serial, mut wide) = (x.clone(), x.clone());
                trsm_rlt(rows, cols, &mut serial, nsrow, 1);
                trsm_rlt(rows, cols, &mut wide, nsrow, 8);
                assert_eq!(
                    block(&serial, nsrow, cols, nsrow),
                    block(&wide, nsrow, cols, nsrow),
                    "trsm_rlt moved a bit at n={n} ld={ld}"
                );
            }
        }
    }

    #[test]
    fn potrf_reports_the_column_that_is_not_positive_definite() {
        let n = 4;
        for bad in 0..n {
            let mut a = spd(n, n, 31337);
            /* make the leading minor of order bad+1 singular by zeroing the
             * whole of row/column `bad` outside its own diagonal, then
             * setting that diagonal to exactly what the pivot will subtract */
            a[bad + bad * n] = 0.0;
            for i in 0..n {
                if i != bad {
                    a[i + bad * n] = 0.0;
                    a[bad + i * n] = 0.0;
                }
            }
            assert_eq!(potrf_l(n, &mut a, n, 1), bad as i64 + 1);
        }
        /* a NaN pivot is a failure too, not a silent NaN factor */
        let mut a = spd(3, 3, 5);
        a[0] = f64::NAN;
        assert_eq!(potrf_l(3, &mut a, 3, 1), 1);
    }

    #[test]
    fn the_solve_kernels_invert_the_triangle() {
        for &n in &[1usize, 2, 5, 17] {
            let ld = n + 3;
            let mut l = spd(n, ld, 909);
            assert_eq!(potrf_l(n, &mut l, ld, 1), 0);
            let b = mat(n, 3, n, 55);

            for (tag, forward) in [("L", true), ("Lt", false)] {
                /* one right-hand side at a time, through trsv */
                let mut xs = vec![0.0; n * 3];
                for j in 0..3 {
                    let x = &mut xs[j * n..(j + 1) * n];
                    x.copy_from_slice(&b[j * n..j * n + n]);
                    if forward {
                        trsv_ln(n, &l, ld, x);
                    } else {
                        trsv_lt(n, &l, ld, x);
                    }
                    /* L x == b, i.e. sum_k L(i,k) x(k) over the triangle */
                    for i in 0..n {
                        let got: f64 = if forward {
                            (0..=i).map(|k| l[i + k * ld] * x[k]).sum()
                        } else {
                            (i..n).map(|k| l[k + i * ld] * x[k]).sum()
                        };
                        assert!(
                            (got - b[i + j * n]).abs() < 1e-9,
                            "{tag} n={n} rhs={j} row {i}"
                        );
                    }
                }
                /* all three at once, through trsm */
                let mut xm = b.clone();
                if forward {
                    trsm_lln(n, 3, &l, ld, &mut xm, n);
                } else {
                    trsm_llt(n, 3, &l, ld, &mut xm, n);
                }
                for k in 0..n * 3 {
                    assert!((xm[k] - xs[k]).abs() < 1e-9, "{tag} n={n} entry {k}");
                }
            }
        }
    }

    #[test]
    fn the_update_kernels_subtract_from_the_destination() {
        for &(m, n, k) in &[(1usize, 1usize, 1usize), (6, 3, 4), (11, 5, 9), (3, 8, 2)] {
            let (lda, ldb, ldc) = (m + 2, k + 3, m + 1);

            /* y := y - A x, A m-by-n */
            let a = mat(m, n, lda, 1);
            let x = mat(n, 1, n, 2);
            let y0 = mat(m, 1, m, 3);
            let mut y = y0.clone();
            gemv_n(m, n, &a, lda, &x, &mut y);
            for i in 0..m {
                let want = y0[i] - (0..n).map(|j| a[i + j * lda] * x[j]).sum::<f64>();
                assert!((y[i] - want).abs() < 1e-13, "gemv_n ({m},{n}) row {i}");
            }

            /* y := y - A' x, so y is n long and x is m long */
            let xt = mat(m, 1, m, 4);
            let yt0 = mat(n, 1, n, 5);
            let mut yt = yt0.clone();
            gemv_t(m, n, &a, lda, &xt, &mut yt);
            for j in 0..n {
                let want = yt0[j] - (0..m).map(|i| a[i + j * lda] * xt[i]).sum::<f64>();
                assert!((yt[j] - want).abs() < 1e-13, "gemv_t ({m},{n}) col {j}");
            }

            /* C := C - A B, A m-by-k, B k-by-n */
            let an = mat(m, k, lda, 6);
            let bn = mat(k, n, ldb, 7);
            let c0 = mat(m, n, ldc, 8);
            let mut c = c0.clone();
            gemm_nn(m, n, k, &an, lda, &bn, ldb, &mut c, ldc);
            for j in 0..n {
                for i in 0..m {
                    let want = c0[i + j * ldc]
                        - (0..k)
                            .map(|l| an[i + l * lda] * bn[l + j * ldb])
                            .sum::<f64>();
                    assert!((c[i + j * ldc] - want).abs() < 1e-13, "gemm_nn ({i},{j})");
                }
            }

            /* C := C - A' B, A k-by-m */
            let at = mat(k, m, ldb, 9);
            let mut c = c0.clone();
            gemm_tn(m, n, k, &at, ldb, &bn, ldb, &mut c, ldc);
            for j in 0..n {
                for i in 0..m {
                    let want = c0[i + j * ldc]
                        - (0..k)
                            .map(|l| at[l + i * ldb] * bn[l + j * ldb])
                            .sum::<f64>();
                    assert!((c[i + j * ldc] - want).abs() < 1e-13, "gemm_tn ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn trsm_inverts_the_triangular_factor_it_is_given() {
        for &(m, n) in &[(1usize, 1usize), (5, 3), (2, 6)] {
            let ld = m + n + 2;
            let mut x = spd(n, ld, 8080);
            assert_eq!(potrf_l(n, &mut x, ld, 1), 0);
            /* B, below the diagonal block, at the same leading dimension */
            let b0 = mat(m, n, ld, 606);
            for j in 0..n {
                for i in 0..m {
                    x[n + i + j * ld] = b0[i + j * ld];
                }
            }
            trsm_rlt(m, n, &mut x, ld, 1);
            /* B * A' == B0 again, i.e. (B A')(i,j) = sum_l B(i,l) A(j,l) with
             * A lower triangular, so l runs 0..=j */
            for j in 0..n {
                for i in 0..m {
                    let got: f64 = (0..=j).map(|l| x[n + i + l * ld] * x[j + l * ld]).sum();
                    assert!(
                        (got - b0[i + j * ld]).abs() < 1e-10,
                        "m={m} n={n} ({i},{j})"
                    );
                }
            }
        }
    }
}
