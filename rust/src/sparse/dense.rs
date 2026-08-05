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

use super::ws::Ws;

/// `C := A * A'`, lower triangle only — `dsyrk ("L", "N", n, k, 1.0, a, lda,
/// 0.0, c, ldc)`.
///
/// `a` is `n`-by-`k` and `c` is `n`-by-`n`; `beta` is 0, so `c`'s lower
/// triangle is overwritten rather than accumulated into. Parts of the strict
/// upper triangle are overwritten too, which a real `dsyrk ("L", ...)` would
/// not do — see the block comment below for why that is sound here.
pub fn syrk_ln(n: usize, k: usize, a: &[f64], lda: usize, c: &mut [f64], ldc: usize) {
    debug_assert!(k == 0 || a.len() >= (k - 1) * lda + n);
    debug_assert!(n == 0 || c.len() >= (n - 1) * ldc + n);

    /* `C (j0:n-1, j0:j0+NB-1)` is a plain `A * A'`, so a block column of the
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
    const NB: usize = 8;
    let mut j0 = 0;
    while j0 < n {
        let jn = NB.min(n - j0);
        gemm_nt(
            n - j0,
            jn,
            k,
            &a[j0..],
            lda,
            &a[j0..],
            lda,
            &mut c[j0 + j0 * ldc..],
            ldc,
        );
        j0 += jn;
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
/// This is the unblocked left-looking `dpotf2 ("L")`: column `j` is completed
/// from the `j-1` columns to its left before it is used.
pub fn potrf_l(n: usize, a: &mut [f64], lda: usize) -> i64 {
    debug_assert!(n == 0 || a.len() >= (n - 1) * lda + n);
    let a = Ws::new(a);

    for j in 0..n {
        /* ajj = A (j,j) - A (j, 0:j-1) * A (j, 0:j-1)' */
        let mut ajj = a[j + j * lda];
        for l in 0..j {
            let x = a[j + l * lda];
            ajj -= x * x;
        }
        if !(ajj > 0.0) {
            /* also catches NaN, as dpotf2's `.LE. ZERO .OR. DISNAN` does */
            a[j + j * lda] = ajj;
            return j as i64 + 1;
        }
        let ajj = ajj.sqrt();
        a[j + j * lda] = ajj;

        if j + 1 < n {
            /* A (j+1:n-1, j) -= A (j+1:n-1, 0:j-1) * A (j, 0:j-1)' */
            let (m, yo) = (n - (j + 1), (j + 1) + j * lda);
            let mut l = 0;
            while l + 4 <= j {
                let t = [
                    a[j + l * lda],
                    a[j + (l + 1) * lda],
                    a[j + (l + 2) * lda],
                    a[j + (l + 3) * lda],
                ];
                let xo = [
                    (j + 1) + l * lda,
                    (j + 1) + (l + 1) * lda,
                    (j + 1) + (l + 2) * lda,
                    (j + 1) + (l + 3) * lda,
                ];
                strip_sub::<4>(m, a, yo, &t, &xo);
                l += 4;
            }
            while l < j {
                strip_sub::<1>(m, a, yo, &[a[j + l * lda]], &[(j + 1) + l * lda]);
                l += 1;
            }
            /* A (j+1:n-1, j) /= ajj */
            let r = 1.0 / ajj;
            for i in j + 1..n {
                a[i + j * lda] *= r;
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
pub fn trsm_rlt(m: usize, n: usize, x: &mut [f64], ld: usize) {
    debug_assert!(n == 0 || x.len() >= (n - 1) * ld + n + m);
    let x = Ws::new(x);

    /* B (:, j) = (B (:, j) - B (:, 0:j-1) * A (j, 0:j-1)') / A (j,j)
     *
     * The reference skips a source column when `A (j,l)` is exactly zero.
     * That is not done here: it is a data-dependent branch in the hot loop for
     * a case the supernodal factorization does not produce, and it changes
     * nothing numerically unless the other operand is an infinity or a NaN. */
    for j in 0..n {
        let yo = n + j * ld;
        let mut l = 0;
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
            assert_eq!(potrf_l(n, &mut a, ld), 0, "n={n}");
            for j in 0..n {
                for i in j..n {
                    let want: f64 = (0..=j.min(i)).map(|l| a[i + l * ld] * a[j + l * ld]).sum();
                    assert!((want - m[i + j * ld]).abs() < 1e-10, "n={n} ({i},{j})");
                }
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
            assert_eq!(potrf_l(n, &mut a, n), bad as i64 + 1);
        }
        /* a NaN pivot is a failure too, not a silent NaN factor */
        let mut a = spd(3, 3, 5);
        a[0] = f64::NAN;
        assert_eq!(potrf_l(3, &mut a, 3), 1);
    }

    /// The two vector solves invert `L` and `L'`, and the two matrix solves
    /// agree with them column by column — to a tolerance, because netlib's own
    /// `trsv` and `trsm` sum the transposed solve in opposite directions.
    #[test]
    fn the_solve_kernels_invert_the_triangle() {
        for &n in &[1usize, 2, 5, 17] {
            let ld = n + 3;
            let mut l = spd(n, ld, 909);
            assert_eq!(potrf_l(n, &mut l, ld), 0);
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
            assert_eq!(potrf_l(n, &mut x, ld), 0);
            /* B, below the diagonal block, at the same leading dimension */
            let b0 = mat(m, n, ld, 606);
            for j in 0..n {
                for i in 0..m {
                    x[n + i + j * ld] = b0[i + j * ld];
                }
            }
            trsm_rlt(m, n, &mut x, ld);
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
