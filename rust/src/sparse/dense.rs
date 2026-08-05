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
//! **The reference's `IF (X(J).NE.ZERO)` guards are not ported.** They skip a
//! source column whose multiplier is exactly zero, which is a data-dependent
//! branch in the hot loop for a case the supernodal factorization does not
//! produce; it changes nothing numerically unless the other operand is an
//! infinity or a NaN.

use rayon::prelude::*;

use super::ws::Ws;

/// `C := A * A'`, lower triangle only — `dsyrk ("L", "N", n, k, 1.0, a, lda,
/// 0.0, c, ldc)`.
///
/// `a` is `n`-by-`k` and `c` is `n`-by-`n`; `beta` is 0, so `c`'s lower
/// triangle is overwritten rather than accumulated into. Parts of the strict
/// upper triangle are overwritten too, which a real `dsyrk ("L", ...)` would
/// not do — see the block comment below for why that is sound here.
///
/// The factorization reaches this through [`syrk_ln_strip`], which is the same
/// call over a range of block columns. This is the whole-matrix name the `dsyrk`
/// mapping above refers to, and what the parity harness's `dsyrk_` exports.
#[allow(dead_code)]
pub fn syrk_ln(n: usize, k: usize, a: &[f64], lda: usize, c: &mut [f64], ldc: usize) {
    debug_assert!(n == 0 || c.len() >= (n - 1) * ldc + n);
    syrk_ln_strip(n, k, a, lda, c, ldc, 0, n);
}

/// [`syrk_ln`]'s block width. A strip that starts on a multiple of it decomposes
/// into the same blocks the whole call would use.
pub const SYRK_NB: usize = 8;

/// Block columns `j0 .. j0+jn` of [`syrk_ln`], with `c` the strip of `C` that
/// starts at column `j0` — its own contiguous `jn * ldc` slice, so a caller can
/// hand several strips to several threads without any of them aliasing.
///
/// `j0` must be a multiple of [`SYRK_NB`]. Then the strip's blocks are exactly
/// the blocks the unsplit call makes, every element is still one `gemm_nt`
/// entry accumulated over `l` ascending, and splitting is a scheduling decision
/// rather than a numerical one.
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
    debug_assert!(k == 0 || a.len() >= (k - 1) * lda + n);
    debug_assert!(j0.is_multiple_of(SYRK_NB) && j0 + jn <= n);

    /* `C (jb:n-1, jb:jb+NB-1)` is a plain `A * A'`, so a block column of the
     * lower triangle goes straight through [`gemm_nt`]'s register-blocked
     * kernel — including its diagonal `NB`-by-`NB` square, computed whole
     * rather than as a triangle.
     *
     * That writes the strict upper triangle of each diagonal square, which a
     * real `dsyrk ("L", ...)` promises not to touch. It is dead storage: the
     * only reader is `t_cholmod_super_numeric_worker.c:1042-1050`, whose
     * assembly loop is `for (i = j ; i < ndrow2 ; i++)` — lower only — and the
     * `dgemm` that fills the rest of `C` starts at row `ndrow1`. The waste is
     * `NB(NB-1)/2` extra dot products per block column, i.e. `~NB/(2n)` of the
     * work, in exchange for the whole kernel being the vectorized one. */
    let mut jb = j0;
    while jb < j0 + jn {
        let jbn = SYRK_NB.min(j0 + jn - jb);
        gemm_nt(
            n - jb,
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
}

/// One `MR`-by-`NR` tile of `C := A * B'`, accumulated in registers.
///
/// `MR` and `NR` are `const` so the two innermost loops unroll fully and `acc`
/// stays in the register file: the `i` extent is contiguous in both `A` and
/// `C`, so `acc[jj]` is `MR/2` NEON `f64x2`s and each step of `l` is one
/// contiguous load of `A` plus `NR` broadcasts of `B`.
///
/// That is the whole optimization, and it does not move a single rounding:
/// each `C(i,j)` still accumulates over `l` ascending, exactly as the netlib
/// reference does. Writing `C` once at the end rather than `k` times is what
/// the register file buys.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_nt<const MR: usize, const NR: usize>(
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
    for l in 0..k {
        let (ao, bo) = (ia + l * lda, jb + l * ldb);
        for (jj, accj) in acc.iter_mut().enumerate() {
            let bv = b[bo + jj];
            for (ii, x) in accj.iter_mut().enumerate() {
                *x += bv * a[ao + ii];
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &x) in accj.iter().enumerate() {
            c[ic + ii + jj * ldc] = x;
        }
    }
}

/// The `m` direction of one `NR`-wide column block, tiled 8/4/2/1.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cols_nt<const NR: usize>(
    m: usize,
    k: usize,
    a: &Ws<f64>,
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
        tile_nt::<8, NR>(k, a, i, lda, b, jb, ldb, c, ic0 + i, ldc);
        i += 8;
    }
    if i + 4 <= m {
        tile_nt::<4, NR>(k, a, i, lda, b, jb, ldb, c, ic0 + i, ldc);
        i += 4;
    }
    if i + 2 <= m {
        tile_nt::<2, NR>(k, a, i, lda, b, jb, ldb, c, ic0 + i, ldc);
        i += 2;
    }
    if i < m {
        tile_nt::<1, NR>(k, a, i, lda, b, jb, ldb, c, ic0 + i, ldc);
    }
}

/// `C := A * B'` — `dgemm ("N", "C", m, n, k, 1.0, a, lda, b, ldb, 0.0, c,
/// ldc)`.
///
/// `a` is `m`-by-`k`, `b` is `n`-by-`k`, `c` is `m`-by-`n`. `beta` is 0.
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
    let (a, b) = (Ws::new_ref(a), Ws::new_ref(b));
    let c = Ws::new(c);

    let mut j = 0;
    while j + 4 <= n {
        cols_nt::<4>(m, k, a, lda, b, j, ldb, c, j * ldc, ldc);
        j += 4;
    }
    if j + 2 <= n {
        cols_nt::<2>(m, k, a, lda, b, j, ldb, c, j * ldc, ldc);
        j += 2;
    }
    if j < n {
        cols_nt::<1>(m, k, a, lda, b, j, ldb, c, j * ldc, ldc);
    }
}

/// One `MR`-by-`NR` tile of `C -= A * B'`, where `A`, `B` and `C` are three
/// disjoint sub-blocks of the *same* array.
///
/// This is the shape [`potrf_l`] and [`trsm_rlt`] are blocked around, and it is
/// the one configuration in this file that upstream never calls: LAPACK's
/// `dpotrf` and the reference `dtrsm` do the equivalent work as a long rank-1
/// ladder, one source column at a time. That ladder is load-bound — it reloads
/// its destination strip every few sources — and it is why those two ran at
/// well under half [`gemm_nt`]'s rate.
///
/// The destination is loaded into the accumulators *before* the loop, and each
/// source is subtracted in turn, `l` ascending. That is the same rounding
/// sequence [`strip_sub`] produces, so blocking moves work between the two
/// without moving a rounding: a destination entry still sees
/// `(((c - a₀b₀) - a₁b₁) - …)` over all its sources in index order, whether the
/// early ones arrive here and the late ones in the panel or not.
///
/// One array rather than three because that is how the callers hold it — a
/// supernode's diagonal block and the rows under it share one `L->x` and one
/// leading dimension, and no `&mut` split expresses that.
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
                *v -= bv * x[ao + ii];
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &v) in accj.iter().enumerate() {
            x[ic + ii + jj * ldc] = v;
        }
    }
}

/// The `m` direction of one `NR`-wide column block of [`tile_sub`], tiled
/// 8/4/2/1.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cols_sub<const NR: usize>(
    m: usize,
    k: usize,
    x: &mut Ws<f64>,
    ia0: usize,
    lda: usize,
    jb: usize,
    ldb: usize,
    ic0: usize,
    ldc: usize,
) {
    let mut i = 0;
    while i + 8 <= m {
        tile_sub::<8, NR>(k, x, ia0 + i, lda, jb, ldb, ic0 + i, ldc);
        i += 8;
    }
    if i + 4 <= m {
        tile_sub::<4, NR>(k, x, ia0 + i, lda, jb, ldb, ic0 + i, ldc);
        i += 4;
    }
    if i + 2 <= m {
        tile_sub::<2, NR>(k, x, ia0 + i, lda, jb, ldb, ic0 + i, ldc);
        i += 2;
    }
    if i < m {
        tile_sub::<1, NR>(k, x, ia0 + i, lda, jb, ldb, ic0 + i, ldc);
    }
}

/// One column of `C` and `MR` rows of it — [`tile_sub`] at `NR = 1`, with the
/// destination handed over as its own slice instead of an offset into the
/// shared array.
///
/// **Same rounding, entry for entry.** `tile_sub` loads `C (i,j)`, then
/// subtracts `B (j,l) * A (i,l)` for `l` ascending, and does that independently
/// for every `(i,j)` in its tile. Nothing crosses between rows or columns, so
/// carving the tile up moves work between tasks without moving a rounding —
/// which is what lets [`gemm_sub_par`] hand row blocks to different threads and
/// still reproduce the serial answer bit for bit.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_col<const MR: usize>(
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut [f64],
    i: usize,
) {
    let mut acc = [0.0f64; MR];
    for (ii, v) in acc.iter_mut().enumerate() {
        *v = c[i + ii];
    }
    for l in 0..k {
        let bv = b[jb + l * ldb];
        let ao = ia + l * lda;
        for (ii, v) in acc.iter_mut().enumerate() {
            *v -= bv * a[ao + ii];
        }
    }
    for (ii, &v) in acc.iter().enumerate() {
        c[i + ii] = v;
    }
}

/// The `m` direction of one column of [`tile_col`], tiled 8/4/2/1 the way
/// [`cols_sub`] tiles its own.
#[allow(clippy::too_many_arguments)]
fn col_sub(
    m: usize,
    k: usize,
    a: &Ws<f64>,
    ia: usize,
    lda: usize,
    b: &Ws<f64>,
    jb: usize,
    ldb: usize,
    c: &mut [f64],
) {
    let mut i = 0;
    while i + 8 <= m {
        tile_col::<8>(k, a, ia + i, lda, b, jb, ldb, c, i);
        i += 8;
    }
    if i + 4 <= m {
        tile_col::<4>(k, a, ia + i, lda, b, jb, ldb, c, i);
        i += 4;
    }
    if i + 2 <= m {
        tile_col::<2>(k, a, ia + i, lda, b, jb, ldb, c, i);
        i += 2;
    }
    if i < m {
        tile_col::<1>(k, a, ia + i, lda, b, jb, ldb, c, i);
    }
}

/// How many flops a `C -= A B'` must carry before [`gemm_sub_par`] is worth a
/// fork/join. Below it the serial [`gemm_sub`] runs, which is also the wider
/// kernel — it keeps `NR = 4` columns of the destination in registers, where
/// the parallel path takes one column at a time.
const GEMM_PAR_FLOPS: f64 = 2.0e6;

/// [`gemm_sub`] with the rows of `C` split across threads.
///
/// **Row blocks, not column blocks.** A row block reads only its own rows of
/// `A`, so the traffic through `A` scales with the block; splitting the other
/// way would stream all of `A` through every task, and `n` is the panel width —
/// 8 — so there would be nothing to gain for it either.
///
/// Sound without `unsafe` because of a property both callers have: **everything
/// read lies below `ic` and everything written at or above it.** `potrf_l`'s
/// `A` and `B` are columns `0..j0` and its `C` is columns `j0..j0+nb`;
/// `trsm_rlt`'s are the same shape one block down. So the array splits once at
/// `ic` into a head every task shares and a tail carved into disjoint column
/// strips, one per (row block, column) pair.
///
/// The kernel is [`tile_col`] rather than [`tile_sub`], i.e. one column of `C`
/// at a time instead of four. That costs re-reading the block's slice of `A`
/// once per column — cheap, because a row block's `A` is sized to stay in
/// cache, and it is what makes every destination piece a contiguous slice.
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

    /* the columns of C, as disjoint slices.  The last one is whatever remains:
     * the array is only guaranteed out to `(n-1)*ldc + m`. */
    let mut cols: Vec<&mut [f64]> = Vec::with_capacity(n);
    let mut rest = dst;
    for _ in 0..n - 1 {
        let (c, tail) = rest.split_at_mut(ldc);
        cols.push(c);
        rest = tail;
    }
    cols.push(rest);

    /* One task per (row block, column): `nt * n` of them, deliberately more
     * than there are threads.
     *
     * Grouping a row block's `n` columns into one task instead — `n` times
     * fewer fork/joins, and `A` read once per block rather than once per
     * column — measures *worse* on this machine, 9.5 ms against 7.9 on `gmm`'s
     * `M`, and no over-decomposition factor recovers it. The cores are not
     * interchangeable: 8 performance and 4 efficiency, so equal-sized tasks
     * put the critical path on an E-core. Small tasks let rayon's stealing
     * find that out at run time, which is worth more here than the cache
     * locality it costs. */
    let rows = m.div_ceil(nt).max(1);
    let mut jobs: Vec<(usize, usize, &mut [f64])> = Vec::with_capacity(nt * n);
    for (j, col) in cols.into_iter().enumerate() {
        let mut rest = &mut col[..m];
        let mut i0 = 0;
        while i0 < m {
            let len = rows.min(m - i0);
            let (piece, tail) = rest.split_at_mut(len);
            jobs.push((i0, j, piece));
            rest = tail;
            i0 += len;
        }
    }

    jobs.par_iter_mut()
        .for_each(|(i0, j, c)| col_sub(c.len(), k, src, ia + *i0, lda, src, jb + *j, ldb, c));
}

/// `C -= A * B'` for three sub-blocks of one array: `A` is `m`-by-`k` at `ia`,
/// `B` is `n`-by-`k` at `jb`, `C` is `m`-by-`n` at `ic`.
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
    let mut j = 0;
    while j + 4 <= n {
        cols_sub::<4>(m, k, x, ia, lda, jb + j, ldb, ic + j * ldc, ldc);
        j += 4;
    }
    if j + 2 <= n {
        cols_sub::<2>(m, k, x, ia, lda, jb + j, ldb, ic + j * ldc, ldc);
        j += 2;
    }
    if j < n {
        cols_sub::<1>(m, k, x, ia, lda, jb + j, ldb, ic + j * ldc, ldc);
    }
}

/// `x [yo .. yo+m] -= Σ_q t[q] * x [xo[q] .. xo[q]+m]`, for `q` ascending.
///
/// The rank-`LR` form of the rank-1 update both [`potrf_l`] and [`trsm_rlt`]
/// spend their time in. It exists to keep the destination strip in registers
/// across `LR` sources instead of storing and re-loading it `LR` times, and
/// because `LR` is `const` the compiler can see that.
///
/// Subtracting the sources one at a time, in order, is what makes this
/// bit-identical to the un-unrolled loop rather than merely equivalent: the
/// rounding sequence `(((y - t0 x0) - t1 x1) - t2 x2) - t3 x3` is unchanged.
/// Summing the products first and subtracting once would be faster still and
/// would move the last bit, so it is not done.
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
                *vv -= tq * x[xq + ii];
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
            v -= tq * x[xo[q] + i];
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
/// This is `dpotrf ("L")`'s blocking around the unblocked left-looking
/// [`potf2_panel`]: the columns to the left of a panel arrive through one
/// [`gemm_sub`], and only the panel's own columns go through the rank-1 ladder.
/// A destination entry still sees every source once, `l` ascending, so the
/// blocking is a scheduling change and `L` is unchanged entry for entry.
pub fn potrf_l(n: usize, a: &mut [f64], lda: usize, nt: usize) -> i64 {
    debug_assert!(n == 0 || a.len() >= (n - 1) * lda + n);
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
        let info = potf2_panel(n - j0, nb, a, j0 + j0 * lda, lda);
        if info != 0 {
            return j0 as i64 + info;
        }
        j0 += nb;
    }
    0
}

/// The widest panel [`potrf_l`] and [`trsm_rlt`] use, and the bound on
/// `potrf_l`'s save buffer. [`panel_width`] picks the actual width.
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

/// The unblocked `dpotf2 ("L")` on an `m`-by-`n` panel whose leading `n` rows
/// are its diagonal block, using only the panel's own columns as sources.
///
/// Returns `dpotf2`'s `info`, 1-based *within the panel*; [`potrf_l`] adds the
/// panel's offset, which is what `dpotrf` does.
fn potf2_panel(m: usize, n: usize, a: &mut Ws<f64>, off: usize, lda: usize) -> i64 {
    for j in 0..n {
        /* ajj = A (j,j) - A (j, 0:j-1) * A (j, 0:j-1)' */
        let mut ajj = a[off + j + j * lda];
        for l in 0..j {
            let x = a[off + j + l * lda];
            ajj -= x * x;
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

/// `B := B * A'^{-1}` with `A` lower triangular and non-unit — `dtrsm ("R",
/// "L", "C", "N", m, n, 1.0, a, lda, b, ldb)`.
///
/// `a` is `n`-by-`n`, `b` is `m`-by-`n`. The worker always passes the *same*
/// array for both, with `a` the supernode's leading `n`-by-`n` diagonal block
/// and `b` the `m` rows directly below it at the same leading dimension
/// (`Lx + psx` and `Lx + psx + nscol2`, both with `LDA = LDB = nsrow`,
/// `:1210-1216`) — so that is the signature: one block, split by row.
///
/// Blocked the same way [`potrf_l`] is, and for the same reason: the columns to
/// the left of a panel arrive through one [`gemm_sub`], the panel's own through
/// the rank-1 ladder, and every destination entry still sees its sources once
/// in index order.
pub fn trsm_rlt(m: usize, n: usize, x: &mut [f64], ld: usize, nt: usize) {
    debug_assert!(n == 0 || x.len() >= (n - 1) * ld + n + m);
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
    let a = Ws::new_ref(a);
    let x = Ws::new(x);

    for j in 0..n {
        let t = x[j] / a[j + j * lda];
        x[j] = t;
        for i in j + 1..n {
            x[i] -= t * a[i + j * lda];
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
    let a = Ws::new_ref(a);
    let x = Ws::new(x);

    for j in (0..n).rev() {
        let mut t = x[j];
        for i in (j + 1..n).rev() {
            t -= a[i + j * lda] * x[i];
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

/// [`gemv_n`] for a block of exactly `NR` columns.
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
                *vv -= tj * a[ao + ii];
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
            v -= tj * a[i + (j0 + jj) * lda];
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

/// [`gemv_t`] for a block of exactly `NR` columns.
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
            *v += a[i + (j0 + jj) * lda] * xv;
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

/// [`trsm_lln`] for a block of exactly `NR` right-hand sides.
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
                b[i + (j0 + jj) * ldb] -= tj * aik;
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

/// [`trsm_llt`] for a block of exactly `NR` right-hand sides.
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
                *tj -= aki * b[k + (j0 + jj) * ldb];
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

/// One `MR`-by-`NR` tile of `C := C - A * B`, accumulated in registers.
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
                *v -= t * a[ao + ii];
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &v) in accj.iter().enumerate() {
            c[ic + ii + jj * ldc] = v;
        }
    }
}

/// The `m` direction of one `NR`-wide column block of [`gemm_nn`], tiled 8/4/2/1.
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

/// One `MR`-by-`NR` tile of `C := C - A' * B`, accumulated in registers.
///
/// The `MR` loads of `a` are strided by `lda` — that is what a transposed
/// operand costs, and the reference pays it too. What the tile buys is that
/// each of them is used `NR` times.
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
                *v += av[ii] * bv;
            }
        }
    }
    for (jj, accj) in acc.iter().enumerate() {
        for (ii, &v) in accj.iter().enumerate() {
            c[ic + ii + jj * ldc] -= v;
        }
    }
}

/// The `m` direction of one `NR`-wide column block of [`gemm_tn`], tiled 4/2/1.
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

    /// A deterministic pseudo-random dense matrix, column-major with a leading
    /// dimension deliberately larger than the block, so a kernel that ignores
    /// `lda` fails.
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

    /// The entries of an `m`-by-`n` column-major block, skipping the padding
    /// between columns — which the helpers above leave as `NaN`, and `NaN` is
    /// never equal to itself.
    fn block(x: &[f64], m: usize, n: usize, ld: usize) -> Vec<f64> {
        (0..n)
            .flat_map(|j| x[j * ld..j * ld + m].iter().copied())
            .collect()
    }

    /// Going wide across the rows of the panel update must not move a bit.
    ///
    /// [`gemm_sub_par`] splits `C -= A B'` by rows of `C` and drops to
    /// [`tile_col`], one column of `C` at a time, where the serial path keeps
    /// four in registers. Neither changes what any destination entry
    /// accumulates or the order it accumulates it in, so `==` is the right
    /// comparison and a tolerance would be hiding something. Sizes straddle
    /// [`GEMM_PAR_FLOPS`], so the threshold is exercised from both sides, and
    /// the row tiling's 8/4/2/1 tail from several remainders.
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

    /// `info` is 1-based and names the column that failed, which is what
    /// `L->minor = k1 + info - 1` depends on.
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

    /// The two vector solves invert `L` and `L'`, and the two matrix solves
    /// agree with them column by column — to a tolerance, because netlib's own
    /// `trsv` and `trsm` sum the transposed solve in opposite directions.
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

    /// `gemv`/`gemm` in both transpositions, against the definition. Both
    /// subtract from what the destination already holds, so a kernel that
    /// overwrote it (`beta = 0`) would fail here.
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
