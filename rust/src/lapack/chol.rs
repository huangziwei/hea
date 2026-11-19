//! The Cholesky factorizations behind R's `chol()` (`La_chol`) with
//! `UPLO = 'U'`: `DPOTRF` (blocked, its diagonal blocks by the recursive
//! `DPOTRF2`) and, for `pivot = TRUE` and mgcv's `mgcv_chol`, the pivoted
//! `DPSTRF` (blocked, or `DPSTF2` below the block size); and `DPOTRI`, the
//! inverse from the factor that R's `chol2inv()` (`La_chol2inv`) asks for.

use super::blas::{ddot, dgemm, dgemv, dscal, dswap, dsyrk, dtrmm, dtrmv, dtrsm};
use super::{fma, EPS};

/// `ILAENV(1, 'DPOTRF', ...)`, which `DPSTRF` also asks for.
const POTRF_NB: i32 = 64;
/// `ILAENV(1, 'DTRTRI', ...)` and `ILAENV(1, 'DLAUUM', ...)`.
const TRTRI_NB: i32 = 64;
const LAUUM_NB: i32 = 64;

/// gfortran's `MAXLOC(WORK(lo:hi), 1)`: the 1-based position of the first
/// maximum, NaNs passed over unless every element is NaN.
unsafe fn maxloc(work: *const f64, lo: i32, hi: i32) -> i32 {
    let mut best = 0;
    let mut val = f64::NAN;
    for i in lo..=hi {
        let w = v!(work, i);
        if !w.is_nan() && (val.is_nan() || w > val) {
            val = w;
            best = i - lo + 1;
        }
    }
    if best == 0 {
        1
    } else {
        best
    }
}

/// The pivot search and diagonal test both routines open with: `PIV = 1..N`
/// and the largest diagonal; `None` when it is not positive (`RANK = 0`,
/// `INFO = 1`), else `(DSTOP, PVT, AJJ)`, the first column's pivot.
unsafe fn pstrf_start(
    n: i32,
    a: *mut f64,
    lda: i32,
    piv: *mut i32,
    tol: f64,
) -> Option<(f64, i32, f64)> {
    for i in 1..=n {
        v!(piv, i) = i;
    }
    let mut pvt = 1;
    let mut ajj = m!(a, lda, pvt, pvt);
    for i in 2..=n {
        if m!(a, lda, i, i) > ajj {
            pvt = i;
            ajj = m!(a, lda, pvt, pvt);
        }
    }
    if ajj <= 0.0 || ajj.is_nan() {
        return None;
    }
    let dstop = if tol < 0.0 { n as f64 * EPS * ajj } else { tol };
    Some((dstop, pvt, ajj))
}

/// Column `j`'s pivot step shared by `DPSTF2` and `DPSTRF` (upper): for
/// `j > 1` pick the largest updated diagonal from `j` on and stop if it is
/// at most `dstop` (column 1 keeps the opening scan's `first` pivot), then
/// swap it into place. Returns the pivot value, or `None` to stop.
#[allow(clippy::too_many_arguments)]
unsafe fn pstrf_pivot(
    n: i32,
    a: *mut f64,
    lda: i32,
    piv: *mut i32,
    work: *mut f64,
    j: i32,
    dstop: f64,
    first: (i32, f64),
) -> Option<f64> {
    let (mut pvt, mut ajj) = first;
    if j > 1 {
        let itemp = maxloc(work, n + j, 2 * n);
        pvt = itemp + j - 1;
        ajj = v!(work, n + pvt);
        if ajj <= dstop || ajj.is_nan() {
            m!(a, lda, j, j) = ajj;
            return None;
        }
    }
    if j != pvt {
        // Pivot OK, so can now swap pivot rows and columns
        m!(a, lda, pvt, pvt) = m!(a, lda, j, j);
        dswap(j - 1, pm!(a, lda, 1, j), 1, pm!(a, lda, 1, pvt), 1);
        if pvt < n {
            dswap(
                n - pvt,
                pm!(a, lda, j, pvt + 1),
                lda,
                pm!(a, lda, pvt, pvt + 1),
                lda,
            );
        }
        dswap(
            pvt - j - 1,
            pm!(a, lda, j, j + 1),
            lda,
            pm!(a, lda, j + 1, pvt),
            1,
        );
        // Swap dot products and PIV
        let dtemp = v!(work, j);
        v!(work, j) = v!(work, pvt);
        v!(work, pvt) = dtemp;
        let itemp = v!(piv, pvt);
        v!(piv, pvt) = v!(piv, j);
        v!(piv, j) = itemp;
    }
    Some(ajj)
}

/// `DPSTF2` with `UPLO = 'U'`: unblocked pivoted Cholesky `P' A P = U' U`,
/// stopping at the first pivot not above `DSTOP`. `work` holds `2n`.
/// Returns `(RANK, INFO)`.
pub(crate) unsafe fn dpstf2_u(
    n: i32,
    a: *mut f64,
    lda: i32,
    piv: *mut i32,
    tol: f64,
    work: *mut f64,
) -> (i32, i32) {
    if n == 0 {
        return (0, 0);
    }
    let Some((dstop, pvt0, ajj0)) = pstrf_start(n, a, lda, piv, tol) else {
        return (0, 1);
    };
    for i in 1..=n {
        v!(work, i) = 0.0;
    }
    for j in 1..=n {
        // Find pivot, test for exit, else swap rows and columns
        // Update dot products, compute possible pivots which are
        // stored in the second half of WORK
        for i in j..=n {
            if j > 1 {
                let x = m!(a, lda, j - 1, i);
                v!(work, i) = fma(x, x, v!(work, i));
            }
            v!(work, n + i) = m!(a, lda, i, i) - v!(work, i);
        }
        let Some(ajj) = pstrf_pivot(n, a, lda, piv, work, j, dstop, (pvt0, ajj0)) else {
            return (j - 1, 1);
        };
        let ajj = ajj.sqrt();
        m!(a, lda, j, j) = ajj;
        // Compute elements J+1:N of row J
        if j < n {
            dgemv(
                b'T',
                j - 1,
                n - j,
                -1.0,
                pm!(a, lda, 1, j + 1),
                lda,
                pm!(a, lda, 1, j),
                1,
                1.0,
                pm!(a, lda, j, j + 1),
                lda,
            );
            dscal(n - j, 1.0 / ajj, pm!(a, lda, j, j + 1), lda);
        }
    }
    (n, 0)
}

/// `DPSTRF` with `UPLO = 'U'`: blocked pivoted Cholesky (`DPSTF2` when the
/// block size is at least `n`). `work` holds `2n`. Returns `(RANK, INFO)`.
pub(crate) unsafe fn dpstrf_u(
    n: i32,
    a: *mut f64,
    lda: i32,
    piv: *mut i32,
    tol: f64,
    work: *mut f64,
) -> (i32, i32) {
    if n == 0 {
        return (0, 0);
    }
    let nb = POTRF_NB;
    if nb <= 1 || nb >= n {
        // Use unblocked code
        return dpstf2_u(n, a, lda, piv, tol, work);
    }
    let Some((dstop, pvt0, ajj0)) = pstrf_start(n, a, lda, piv, tol) else {
        return (0, 1);
    };
    let mut k = 1;
    while k <= n {
        // Account for last block not being NB wide
        let jb = nb.min(n - k + 1);
        // Set relevant part of first half of WORK to zero,
        // holds dot products
        for i in k..=n {
            v!(work, i) = 0.0;
        }
        let mut j = k;
        while j <= k + jb - 1 {
            // Find pivot, test for exit, else swap rows and columns
            // Update dot products, compute possible pivots which are
            // stored in the second half of WORK
            for i in j..=n {
                if j > k {
                    let x = m!(a, lda, j - 1, i);
                    v!(work, i) = fma(x, x, v!(work, i));
                }
                v!(work, n + i) = m!(a, lda, i, i) - v!(work, i);
            }
            let Some(ajj) = pstrf_pivot(n, a, lda, piv, work, j, dstop, (pvt0, ajj0)) else {
                return (j - 1, 1);
            };
            let ajj = ajj.sqrt();
            m!(a, lda, j, j) = ajj;
            // Compute elements J+1:N of row J.
            if j < n {
                dgemv(
                    b'T',
                    j - k,
                    n - j,
                    -1.0,
                    pm!(a, lda, k, j + 1),
                    lda,
                    pm!(a, lda, k, j),
                    1,
                    1.0,
                    pm!(a, lda, j, j + 1),
                    lda,
                );
                dscal(n - j, 1.0 / ajj, pm!(a, lda, j, j + 1), lda);
            }
            j += 1;
        }
        // Update trailing matrix, J already incremented
        if k + jb <= n {
            dsyrk(
                b'U',
                b'T',
                n - j + 1,
                jb,
                -1.0,
                pm!(a, lda, k, j),
                lda,
                1.0,
                pm!(a, lda, j, j),
                lda,
            );
        }
        k += nb;
    }
    (n, 0)
}

/// `DPOTRF2` with `UPLO = 'U'`: recursive Cholesky `A = U' U`, splitting
/// the columns in half. Returns `INFO`.
pub(crate) unsafe fn dpotrf2_u(n: i32, a: *mut f64, lda: i32) -> i32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        // Test for non-positive-definiteness
        if m!(a, lda, 1, 1) <= 0.0 || m!(a, lda, 1, 1).is_nan() {
            return 1;
        }
        // Factor
        m!(a, lda, 1, 1) = m!(a, lda, 1, 1).sqrt();
        return 0;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    // Factor A11
    let iinfo = dpotrf2_u(n1, a, lda);
    if iinfo != 0 {
        return iinfo;
    }
    // Update and scale A12
    dtrsm(
        b'L',
        b'U',
        b'T',
        b'N',
        n1,
        n2,
        1.0,
        a,
        lda,
        pm!(a, lda, 1, n1 + 1),
        lda,
    );
    // Update and factor A22
    dsyrk(
        b'U',
        b'T',
        n2,
        n1,
        -1.0,
        pm!(a, lda, 1, n1 + 1),
        lda,
        1.0,
        pm!(a, lda, n1 + 1, n1 + 1),
        lda,
    );
    let iinfo = dpotrf2_u(n2, pm!(a, lda, n1 + 1, n1 + 1), lda);
    if iinfo != 0 {
        return iinfo + n1;
    }
    0
}

/// `DPOTRF` with `UPLO = 'U'`: blocked Cholesky `A = U' U` (`DPOTRF2` when
/// the block size is at least `n`). Returns `INFO`.
pub(crate) unsafe fn dpotrf_u(n: i32, a: *mut f64, lda: i32) -> i32 {
    if n == 0 {
        return 0;
    }
    let nb = POTRF_NB;
    if nb <= 1 || nb >= n {
        // Use unblocked code.
        return dpotrf2_u(n, a, lda);
    }
    let mut j = 1;
    while j <= n {
        // Update and factorize the current diagonal block and test
        // for non-positive-definiteness.
        let jb = nb.min(n - j + 1);
        dsyrk(
            b'U',
            b'T',
            jb,
            j - 1,
            -1.0,
            pm!(a, lda, 1, j),
            lda,
            1.0,
            pm!(a, lda, j, j),
            lda,
        );
        let info = dpotrf2_u(jb, pm!(a, lda, j, j), lda);
        if info != 0 {
            return info + j - 1;
        }
        if j + jb <= n {
            // Compute the current block row.
            dgemm(
                b'T',
                b'N',
                jb,
                n - j - jb + 1,
                j - 1,
                -1.0,
                pm!(a, lda, 1, j),
                lda,
                pm!(a, lda, 1, j + jb),
                lda,
                1.0,
                pm!(a, lda, j, j + jb),
                lda,
            );
            dtrsm(
                b'L',
                b'U',
                b'T',
                b'N',
                jb,
                n - j - jb + 1,
                1.0,
                pm!(a, lda, j, j),
                lda,
                pm!(a, lda, j, j + jb),
                lda,
            );
        }
        j += nb;
    }
    0
}

/// `DTRTI2` with `UPLO = 'U'`, `DIAG = 'N'`: the inverse of the upper
/// triangular `a` in place, a column at a time - the reciprocal of the
/// diagonal, `DTRMV` with the inverse so far, `DSCAL` by minus the
/// reciprocal.
pub(crate) unsafe fn dtrti2_u(n: i32, a: *mut f64, lda: i32) {
    for j in 1..=n {
        m!(a, lda, j, j) = 1.0 / m!(a, lda, j, j);
        let ajj = -m!(a, lda, j, j);
        dtrmv(b'U', b'N', b'N', j - 1, a, lda, pm!(a, lda, 1, j), 1);
        dscal(j - 1, ajj, pm!(a, lda, 1, j), 1);
    }
}

/// `DTRTRI` with `UPLO = 'U'`, `DIAG = 'N'`: blocked (`DTRTI2` when the
/// block size is at least `n`). Returns `INFO` (`k > 0`: `A(k,k)` is zero).
pub(crate) unsafe fn dtrtri_u(n: i32, a: *mut f64, lda: i32) -> i32 {
    if n == 0 {
        return 0;
    }
    for info in 1..=n {
        if m!(a, lda, info, info) == 0.0 {
            return info;
        }
    }
    let nb = TRTRI_NB;
    if nb <= 1 || nb >= n {
        dtrti2_u(n, a, lda);
        return 0;
    }
    let mut j = 1;
    while j <= n {
        let jb = nb.min(n - j + 1);
        // Compute rows 1:j-1 of current block column
        dtrmm(
            b'L',
            b'U',
            b'N',
            b'N',
            j - 1,
            jb,
            1.0,
            a,
            lda,
            pm!(a, lda, 1, j),
            lda,
        );
        dtrsm(
            b'R',
            b'U',
            b'N',
            b'N',
            j - 1,
            jb,
            -1.0,
            pm!(a, lda, j, j),
            lda,
            pm!(a, lda, 1, j),
            lda,
        );
        // Compute inverse of current diagonal block
        dtrti2_u(jb, pm!(a, lda, j, j), lda);
        j += nb;
    }
    0
}

/// `DLAUU2` with `UPLO = 'U'`: `U * U'` in the upper triangle of `a`.
pub(crate) unsafe fn dlauu2_u(n: i32, a: *mut f64, lda: i32) {
    for i in 1..=n {
        let aii = m!(a, lda, i, i);
        if i < n {
            m!(a, lda, i, i) = ddot(n - i + 1, pm!(a, lda, i, i), lda, pm!(a, lda, i, i), lda);
            dgemv(
                b'N',
                i - 1,
                n - i,
                1.0,
                pm!(a, lda, 1, i + 1),
                lda,
                pm!(a, lda, i, i + 1),
                lda,
                aii,
                pm!(a, lda, 1, i),
                1,
            );
        } else {
            dscal(i, aii, pm!(a, lda, 1, i), 1);
        }
    }
}

/// `DLAUUM` with `UPLO = 'U'`: blocked (`DLAUU2` when the block size is at
/// least `n`).
pub(crate) unsafe fn dlauum_u(n: i32, a: *mut f64, lda: i32) {
    if n == 0 {
        return;
    }
    let nb = LAUUM_NB;
    if nb <= 1 || nb >= n {
        dlauu2_u(n, a, lda);
        return;
    }
    let mut i = 1;
    while i <= n {
        let ib = nb.min(n - i + 1);
        dtrmm(
            b'R',
            b'U',
            b'T',
            b'N',
            i - 1,
            ib,
            1.0,
            pm!(a, lda, i, i),
            lda,
            pm!(a, lda, 1, i),
            lda,
        );
        dlauu2_u(ib, pm!(a, lda, i, i), lda);
        if i + ib <= n {
            dgemm(
                b'N',
                b'T',
                i - 1,
                ib,
                n - i - ib + 1,
                1.0,
                pm!(a, lda, 1, i + ib),
                lda,
                pm!(a, lda, i, i + ib),
                lda,
                1.0,
                pm!(a, lda, 1, i),
                lda,
            );
            dsyrk(
                b'U',
                b'N',
                ib,
                n - i - ib + 1,
                1.0,
                pm!(a, lda, i, i + ib),
                lda,
                1.0,
                pm!(a, lda, i, i),
                lda,
            );
        }
        i += nb;
    }
}

/// `DPOTRI` with `UPLO = 'U'`: `(U'U)^{-1}` from the factor `U`, in the
/// upper triangle. Returns `INFO`.
pub(crate) unsafe fn dpotri_u(n: i32, a: *mut f64, lda: i32) -> i32 {
    let info = dtrtri_u(n, a, lda);
    if info > 0 {
        return info;
    }
    dlauum_u(n, a, lda);
    0
}
