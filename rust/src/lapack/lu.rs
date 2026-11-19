//! `DGESV`: the LU solve R's `solve()` calls (`La_solve`): `DGETRF` (panels
//! of `NB` columns, each factored by the recursive `DGETRF2`) and `DGETRS`
//! with `TRANS = 'N'`.

use super::blas::{dgemm, dscal, dtrsm, idamax};
use super::SAFMIN;

/// `ILAENV(1, 'DGETRF', ...)`.
const GETRF_NB: i32 = 64;

/// `DLASWP`: the row interchanges `ipiv(k1:k2)` (in order for `incx > 0`,
/// in reverse for `incx < 0`) applied to the `n` columns of `a`, 32
/// columns at a time.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlaswp(
    n: i32,
    a: *mut f64,
    lda: i32,
    k1: i32,
    k2: i32,
    ipiv: *const i32,
    incx: i32,
) {
    let (ix0, i1, i2, inc) = if incx > 0 {
        (k1, k1, k2, 1)
    } else if incx < 0 {
        (k1 + (k1 - k2) * incx, k2, k1, -1)
    } else {
        return;
    };
    let rows = |i1: i32, i2: i32, inc: i32| {
        let mut v = Vec::new();
        let mut i = i1;
        while (inc > 0 && i <= i2) || (inc < 0 && i >= i2) {
            v.push(i);
            i += inc;
        }
        v
    };
    let order = rows(i1, i2, inc);
    let mut n32 = (n / 32) * 32;
    if n32 != 0 {
        let mut j = 1;
        while j <= n32 {
            let mut ix = ix0;
            for &i in &order {
                let ip = v!(ipiv, ix);
                if ip != i {
                    for k in j..=(j + 31) {
                        let temp = m!(a, lda, i, k);
                        m!(a, lda, i, k) = m!(a, lda, ip, k);
                        m!(a, lda, ip, k) = temp;
                    }
                }
                ix += incx;
            }
            j += 32;
        }
    }
    if n32 != n {
        n32 += 1;
        let mut ix = ix0;
        for &i in &order {
            let ip = v!(ipiv, ix);
            if ip != i {
                for k in n32..=n {
                    let temp = m!(a, lda, i, k);
                    m!(a, lda, i, k) = m!(a, lda, ip, k);
                    m!(a, lda, ip, k) = temp;
                }
            }
            ix += incx;
        }
    }
}

/// `DGETRF2`: recursive LU with partial pivoting of the `m`-by-`n` `a`,
/// splitting the columns in half. Returns `INFO` (`i > 0`: `U(i,i)` is
/// exactly zero).
pub(crate) unsafe fn dgetrf2(m: i32, n: i32, a: *mut f64, lda: i32, ipiv: *mut i32) -> i32 {
    if m == 0 || n == 0 {
        return 0;
    }
    let mut info = 0;
    if m == 1 {
        // Use unblocked code for one row case
        // Just need to handle IPIV and INFO
        v!(ipiv, 1) = 1;
        if m!(a, lda, 1, 1) == 0.0 {
            info = 1;
        }
    } else if n == 1 {
        // Use unblocked code for one column case
        // Compute machine safe minimum
        let sfmin = SAFMIN;
        // Find pivot and test for singularity
        let i = idamax(m, a, 1);
        v!(ipiv, 1) = i;
        if m!(a, lda, i, 1) != 0.0 {
            // Apply the interchange
            if i != 1 {
                let temp = m!(a, lda, 1, 1);
                m!(a, lda, 1, 1) = m!(a, lda, i, 1);
                m!(a, lda, i, 1) = temp;
            }
            // Compute elements 2:M of the column
            if m!(a, lda, 1, 1).abs() >= sfmin {
                dscal(m - 1, 1.0 / m!(a, lda, 1, 1), pm!(a, lda, 2, 1), 1);
            } else {
                for i in 1..=(m - 1) {
                    m!(a, lda, 1 + i, 1) /= m!(a, lda, 1, 1);
                }
            }
        } else {
            info = 1;
        }
    } else {
        // Use recursive code
        let n1 = m.min(n) / 2;
        let n2 = n - n1;
        //        [ A11 ]
        // Factor [ --- ]
        //        [ A21 ]
        let iinfo = dgetrf2(m, n1, a, lda, ipiv);
        if info == 0 && iinfo > 0 {
            info = iinfo;
        }
        //                       [ A12 ]
        // Apply interchanges to [ --- ]
        //                       [ A22 ]
        dlaswp(n2, pm!(a, lda, 1, n1 + 1), lda, 1, n1, ipiv, 1);
        // Solve A12
        dtrsm(
            b'L',
            b'L',
            b'N',
            b'U',
            n1,
            n2,
            1.0,
            a,
            lda,
            pm!(a, lda, 1, n1 + 1),
            lda,
        );
        // Update A22
        dgemm(
            b'N',
            b'N',
            m - n1,
            n2,
            n1,
            -1.0,
            pm!(a, lda, n1 + 1, 1),
            lda,
            pm!(a, lda, 1, n1 + 1),
            lda,
            1.0,
            pm!(a, lda, n1 + 1, n1 + 1),
            lda,
        );
        // Factor A22
        let iinfo = dgetrf2(
            m - n1,
            n2,
            pm!(a, lda, n1 + 1, n1 + 1),
            lda,
            pv!(ipiv, n1 + 1),
        );
        // Adjust INFO and the pivot indices
        if info == 0 && iinfo > 0 {
            info = iinfo + n1;
        }
        for i in (n1 + 1)..=m.min(n) {
            v!(ipiv, i) += n1;
        }
        // Apply interchanges to A21
        dlaswp(n1, a, lda, n1 + 1, m.min(n), ipiv, 1);
    }
    info
}

/// `DGETRF`: LU with partial pivoting, `A = P L U`, in panels of `NB`
/// columns (one `DGETRF2` call when `NB` is not below `min(m, n)`).
pub(crate) unsafe fn dgetrf(m: i32, n: i32, a: *mut f64, lda: i32, ipiv: *mut i32) -> i32 {
    if m == 0 || n == 0 {
        return 0;
    }
    let nb = GETRF_NB;
    let mut info = 0;
    if nb <= 1 || nb >= m.min(n) {
        // Use unblocked code.
        return dgetrf2(m, n, a, lda, ipiv);
    }
    // Use blocked code.
    let mut j = 1;
    while j <= m.min(n) {
        let jb = (m.min(n) - j + 1).min(nb);
        // Factor diagonal and subdiagonal blocks and test for exact
        // singularity.
        let iinfo = dgetrf2(m - j + 1, jb, pm!(a, lda, j, j), lda, pv!(ipiv, j));
        // Adjust INFO and the pivot indices.
        if info == 0 && iinfo > 0 {
            info = iinfo + j - 1;
        }
        for i in j..=m.min(j + jb - 1) {
            v!(ipiv, i) += j - 1;
        }
        // Apply interchanges to columns 1:J-1.
        dlaswp(j - 1, a, lda, j, j + jb - 1, ipiv, 1);
        if j + jb <= n {
            // Apply interchanges to columns J+JB:N.
            dlaswp(
                n - j - jb + 1,
                pm!(a, lda, 1, j + jb),
                lda,
                j,
                j + jb - 1,
                ipiv,
                1,
            );
            // Compute block row of U.
            dtrsm(
                b'L',
                b'L',
                b'N',
                b'U',
                jb,
                n - j - jb + 1,
                1.0,
                pm!(a, lda, j, j),
                lda,
                pm!(a, lda, j, j + jb),
                lda,
            );
            if j + jb <= m {
                // Update trailing submatrix.
                dgemm(
                    b'N',
                    b'N',
                    m - j - jb + 1,
                    n - j - jb + 1,
                    jb,
                    -1.0,
                    pm!(a, lda, j + jb, j),
                    lda,
                    pm!(a, lda, j, j + jb),
                    lda,
                    1.0,
                    pm!(a, lda, j + jb, j + jb),
                    lda,
                );
            }
        }
        j += nb;
    }
    info
}

/// `DGETRS` with `TRANS = 'N'`: solves `A X = B` from `DGETRF`'s factors.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgetrs_n(
    n: i32,
    nrhs: i32,
    a: *const f64,
    lda: i32,
    ipiv: *const i32,
    b: *mut f64,
    ldb: i32,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    // Solve A * X = B.
    // Apply row interchanges to the right hand sides.
    dlaswp(nrhs, b, ldb, 1, n, ipiv, 1);
    // Solve L*X = B, overwriting B with X.
    dtrsm(b'L', b'L', b'N', b'U', n, nrhs, 1.0, a, lda, b, ldb);
    // Solve U*X = B, overwriting B with X.
    dtrsm(b'L', b'U', b'N', b'N', n, nrhs, 1.0, a, lda, b, ldb);
}

/// `DGESV`: `A X = B` by `DGETRF` then `DGETRS`; `a` is overwritten by
/// the factors, `b` by `X`. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgesv(
    n: i32,
    nrhs: i32,
    a: *mut f64,
    lda: i32,
    ipiv: *mut i32,
    b: *mut f64,
    ldb: i32,
) -> i32 {
    // Compute the LU factorization of A.
    let info = dgetrf(n, n, a, lda, ipiv);
    if info == 0 {
        // Solve the system A*X = B, overwriting B with X.
        dgetrs_n(n, nrhs, a, lda, ipiv, b, ldb);
    }
    info
}
