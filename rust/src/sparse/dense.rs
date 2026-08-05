//! The four dense kernels the supernodal factorization calls, in the exact
//! configurations it calls them in.
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
//! So these are not a BLAS. They are those four call sites, with the branches
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
//! **Summation order follows the netlib reference**, i.e. `k` ascending into
//! each `C(i,j)`. Nothing here can be bit-compared against a tuned BLAS —
//! Accelerate's blocking is not knowable — so the port is gated against
//! upstream's C *linked to these same kernels* (see the module's tests), which
//! is what isolates a structural defect from a summation-order difference.

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
