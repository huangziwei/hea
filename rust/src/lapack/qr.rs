//! `DGEQP3`: QR factorization with column pivoting, `A P = Q R`, as R's
//! `qr(x, LAPACK = TRUE)` calls it (`La_qr`: every column free, the
//! workspace its `lwork = -1` query returns). Panels of `NB` columns go
//! through `DLAQPS` while more than `NX` columns remain, the rest through
//! `DLAQP2`.

use super::blas::{dgemm, dgemv, dnrm2, dswap, idamax};
use super::gesdd::dgeqrf;
use super::ormtr::{dlarf1f_left, dormqr_l};
use super::sytrd::dlarfg;
use super::{fma, EPS};

/// `ILAENV(1, 'DGEQRF', ...)`, `ILAENV(2, ...)` and `ILAENV(3, ...)`.
const GEQRF_NB: i32 = 32;
const GEQRF_NBMIN: i32 = 2;
const GEQRF_NX: i32 = 128;

/// `DGEQP3`'s optimal workspace for an `m`-by-`n` matrix, what its
/// `lwork = -1` query returns.
pub(crate) fn dgeqp3_lwork(m: i32, n: i32) -> i32 {
    if m.min(n) == 0 {
        1
    } else {
        2 * n + (n + 1) * GEQRF_NB
    }
}

/// `DLAQP2`: unblocked QR with column pivoting of rows `offset+1:m` of the
/// `n` columns of `a`, the first `offset` rows already factored. `vn1` and
/// `vn2` hold the partial and exact column norms.
#[allow(clippy::too_many_arguments)]
unsafe fn dlaqp2(
    m: i32,
    n: i32,
    offset: i32,
    a: *mut f64,
    lda: i32,
    jpvt: *mut i32,
    tau: *mut f64,
    vn1: *mut f64,
    vn2: *mut f64,
    work: *mut f64,
) {
    let mn = (m - offset).min(n);
    let tol3z = EPS.sqrt();
    // Compute factorization.
    for i in 1..=mn {
        let offpi = offset + i;
        // Determine ith pivot column and swap if necessary.
        let pvt = (i - 1) + idamax(n - i + 1, pv!(vn1, i), 1);
        if pvt != i {
            dswap(m, pm!(a, lda, 1, pvt), 1, pm!(a, lda, 1, i), 1);
            let itemp = v!(jpvt, pvt);
            v!(jpvt, pvt) = v!(jpvt, i);
            v!(jpvt, i) = itemp;
            v!(vn1, pvt) = v!(vn1, i);
            v!(vn2, pvt) = v!(vn2, i);
        }
        // Generate elementary reflector H(i).
        v!(tau, i) = if offpi < m {
            dlarfg(
                m - offpi + 1,
                pm!(a, lda, offpi, i),
                pm!(a, lda, offpi + 1, i),
                1,
            )
        } else {
            dlarfg(1, pm!(a, lda, m, i), pm!(a, lda, m, i), 1)
        };
        if i < n {
            // Apply H(i)**T to A(offset+i:m,i+1:n) from the left.
            dlarf1f_left(
                m - offpi + 1,
                n - i,
                pm!(a, lda, offpi, i),
                1,
                v!(tau, i),
                pm!(a, lda, offpi, i + 1),
                lda,
                work,
            );
        }
        // Update partial column norms.
        for j in (i + 1)..=n {
            if v!(vn1, j) != 0.0 {
                // NOTE: The following lines follow from the analysis in
                // Lapack Working Note 176.
                let q = m!(a, lda, offpi, j).abs() / v!(vn1, j);
                let temp = fma(-q, q, 1.0).max(0.0);
                let r = v!(vn1, j) / v!(vn2, j);
                let temp2 = temp * (r * r);
                if temp2 <= tol3z {
                    if offpi < m {
                        v!(vn1, j) = dnrm2(m - offpi, pm!(a, lda, offpi + 1, j), 1);
                        v!(vn2, j) = v!(vn1, j);
                    } else {
                        v!(vn1, j) = 0.0;
                        v!(vn2, j) = 0.0;
                    }
                } else {
                    v!(vn1, j) *= temp.sqrt();
                }
            }
        }
    }
}

/// `DLAQPS`: one panel of up to `nb` pivoted Householder steps on rows
/// `offset+1:m` of the `n` columns of `a`, deferring the trailing update
/// through `F`; stops early (`kb < nb`) when a partial norm has lost its
/// accuracy, and recomputes those norms afterwards. Returns `kb`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlaqps(
    m: i32,
    n: i32,
    offset: i32,
    nb: i32,
    a: *mut f64,
    lda: i32,
    jpvt: *mut i32,
    tau: *mut f64,
    vn1: *mut f64,
    vn2: *mut f64,
    auxv: *mut f64,
    f: *mut f64,
    ldf: i32,
) -> i32 {
    let lastrk = m.min(n + offset);
    let mut lsticc = 0;
    let mut k = 0;
    let tol3z = EPS.sqrt();

    // Beginning of while loop.
    while k < nb && lsticc == 0 {
        k += 1;
        let rk = offset + k;
        // Determine ith pivot column and swap if necessary
        let pvt = (k - 1) + idamax(n - k + 1, pv!(vn1, k), 1);
        if pvt != k {
            dswap(m, pm!(a, lda, 1, pvt), 1, pm!(a, lda, 1, k), 1);
            dswap(k - 1, pm!(f, ldf, pvt, 1), ldf, pm!(f, ldf, k, 1), ldf);
            let itemp = v!(jpvt, pvt);
            v!(jpvt, pvt) = v!(jpvt, k);
            v!(jpvt, k) = itemp;
            v!(vn1, pvt) = v!(vn1, k);
            v!(vn2, pvt) = v!(vn2, k);
        }
        // Apply previous Householder reflectors to column K:
        // A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**T.
        if k > 1 {
            dgemv(
                b'N',
                m - rk + 1,
                k - 1,
                -1.0,
                pm!(a, lda, rk, 1),
                lda,
                pm!(f, ldf, k, 1),
                ldf,
                1.0,
                pm!(a, lda, rk, k),
                1,
            );
        }
        // Generate elementary reflector H(k).
        v!(tau, k) = if rk < m {
            dlarfg(m - rk + 1, pm!(a, lda, rk, k), pm!(a, lda, rk + 1, k), 1)
        } else {
            dlarfg(1, pm!(a, lda, rk, k), pm!(a, lda, rk, k), 1)
        };
        let akk = m!(a, lda, rk, k);
        m!(a, lda, rk, k) = 1.0;
        // Compute Kth column of F:
        // Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)**T*A(RK:M,K).
        if k < n {
            dgemv(
                b'T',
                m - rk + 1,
                n - k,
                v!(tau, k),
                pm!(a, lda, rk, k + 1),
                lda,
                pm!(a, lda, rk, k),
                1,
                0.0,
                pm!(f, ldf, k + 1, k),
                1,
            );
        }
        // Padding F(1:K,K) with zeros.
        for j in 1..=k {
            m!(f, ldf, j, k) = 0.0;
        }
        // Incremental updating of F:
        // F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)**T
        //             *A(RK:M,K).
        if k > 1 {
            dgemv(
                b'T',
                m - rk + 1,
                k - 1,
                -v!(tau, k),
                pm!(a, lda, rk, 1),
                lda,
                pm!(a, lda, rk, k),
                1,
                0.0,
                auxv,
                1,
            );
            dgemv(
                b'N',
                n,
                k - 1,
                1.0,
                f,
                ldf,
                auxv,
                1,
                1.0,
                pm!(f, ldf, 1, k),
                1,
            );
        }
        // Update the current row of A:
        // A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)**T.
        if k < n {
            dgemv(
                b'N',
                n - k,
                k,
                -1.0,
                pm!(f, ldf, k + 1, 1),
                ldf,
                pm!(a, lda, rk, 1),
                lda,
                1.0,
                pm!(a, lda, rk, k + 1),
                lda,
            );
        }
        // Update partial column norms.
        if rk < lastrk {
            for j in (k + 1)..=n {
                if v!(vn1, j) != 0.0 {
                    // NOTE: The following lines follow from the analysis in
                    // Lapack Working Note 176.
                    let t = m!(a, lda, rk, j).abs() / v!(vn1, j);
                    let temp = 0.0f64.max((1.0 + t) * (1.0 - t));
                    let r = v!(vn1, j) / v!(vn2, j);
                    let temp2 = temp * (r * r);
                    if temp2 <= tol3z {
                        v!(vn2, j) = lsticc as f64;
                        lsticc = j;
                    } else {
                        v!(vn1, j) *= temp.sqrt();
                    }
                }
            }
        }
        m!(a, lda, rk, k) = akk;
    }
    let kb = k;
    let rk = offset + kb;

    // Apply the block reflector to the rest of the matrix:
    // A(RK+1:M,KB+1:N) := A(RK+1:M,KB+1:N) -
    //                     A(RK+1:M,1:KB)*F(KB+1:N,1:KB)**T.
    if kb < n.min(m - offset) {
        dgemm(
            b'N',
            b'T',
            m - rk,
            n - kb,
            kb,
            -1.0,
            pm!(a, lda, rk + 1, 1),
            lda,
            pm!(f, ldf, kb + 1, 1),
            ldf,
            1.0,
            pm!(a, lda, rk + 1, kb + 1),
            lda,
        );
    }

    // Recomputation of difficult columns.
    while lsticc > 0 {
        let itemp = v!(vn2, lsticc).round() as i32;
        v!(vn1, lsticc) = dnrm2(m - rk, pm!(a, lda, rk + 1, lsticc), 1);
        // NOTE: The computation of VN1( LSTICC ) relies on the fact that
        // SNRM2 does not fail on vectors with norm below the value of
        // SQRT(DLAMCH('S'))
        v!(vn2, lsticc) = v!(vn1, lsticc);
        lsticc = itemp;
    }
    kb
}

/// `DGEQP3` with every column free (`JPVT` all zero on entry, as `La_qr`
/// passes it): `jpvt` receives the 1-based column permutation, `tau` the
/// `min(m, n)` reflector scalars, and `a` holds `R` above its diagonal and
/// the reflectors below. `work` holds `lwork` doubles.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgeqp3_free(
    m: i32,
    n: i32,
    a: *mut f64,
    lda: i32,
    jpvt: *mut i32,
    tau: *mut f64,
    work: *mut f64,
    lwork: i32,
) {
    // Move initial columns up front (none are fixed).
    for j in 1..=n {
        v!(jpvt, j) = j;
    }
    dgeqp3_free_cols(m, n, a, lda, jpvt, tau, work, lwork, 0);
}

/// `DGEQP3` as LAPACK defines it: a column whose `jpvt` entry is non-zero on
/// entry is fixed, moved to the front and factored unpivoted (`DGEQRF`, the
/// rest updated by `DORMQR('L', 'T')`); the remaining free columns are
/// pivoted by norm as in [`dgeqp3_free`]. `jpvt` receives the 1-based
/// column permutation.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgeqp3(
    m: i32,
    n: i32,
    a: *mut f64,
    lda: i32,
    jpvt: *mut i32,
    tau: *mut f64,
    work: *mut f64,
    lwork: i32,
) {
    // Move initial columns up front.
    let mut nfxd = 1;
    for j in 1..=n {
        if v!(jpvt, j) != 0 {
            if j != nfxd {
                dswap(m, pm!(a, lda, 1, j), 1, pm!(a, lda, 1, nfxd), 1);
                v!(jpvt, j) = v!(jpvt, nfxd);
                v!(jpvt, nfxd) = j;
            } else {
                v!(jpvt, j) = j;
            }
            nfxd += 1;
        } else {
            v!(jpvt, j) = j;
        }
    }
    nfxd -= 1;

    // Factorize fixed columns: the QR of the fixed columns, the remaining
    // columns updated.
    if nfxd > 0 {
        let na = m.min(nfxd);
        dgeqrf(m, na, a, lda, tau, work, lwork);
        if na < n {
            dormqr_l(
                b'T',
                m,
                n - na,
                na,
                a,
                lda,
                tau,
                pm!(a, lda, 1, na + 1),
                lda,
                work,
                lwork,
            );
        }
    }
    dgeqp3_free_cols(m, n, a, lda, jpvt, tau, work, lwork, nfxd);
}

/// `DGEQP3`'s factorization of the free columns `nfxd + 1 .. n`, the first
/// `nfxd` already factored.
#[allow(clippy::too_many_arguments)]
unsafe fn dgeqp3_free_cols(
    m: i32,
    n: i32,
    a: *mut f64,
    lda: i32,
    jpvt: *mut i32,
    tau: *mut f64,
    work: *mut f64,
    lwork: i32,
    nfxd: i32,
) {
    let minmn = m.min(n);

    // Factorize free columns
    if nfxd < minmn {
        let sm = m - nfxd;
        let sn = n - nfxd;
        let sminmn = minmn - nfxd;
        // Determine the block size.
        let mut nb = GEQRF_NB;
        let mut nbmin = 2;
        let mut nx = 0;
        if nb > 1 && nb < sminmn {
            // Determine when to cross over from blocked to unblocked code.
            nx = 0.max(GEQRF_NX);
            if nx < sminmn {
                // Determine if workspace is large enough for blocked code.
                let minws = 2 * sn + (sn + 1) * nb;
                if lwork < minws {
                    // Not enough workspace to use optimal NB: Reduce NB and
                    // determine the minimum value of NB.
                    nb = (lwork - 2 * sn) / (sn + 1);
                    nbmin = 2.max(GEQRF_NBMIN);
                }
            }
        }
        // Initialize partial column norms. The first N elements of work
        // store the exact column norms.
        for j in (nfxd + 1)..=n {
            v!(work, j) = dnrm2(sm, pm!(a, lda, nfxd + 1, j), 1);
            v!(work, n + j) = v!(work, j);
        }
        let mut j = nfxd + 1;
        if nb >= nbmin && nb < sminmn && nx < sminmn {
            // Use blocked code initially.
            let topbmn = minmn - nx;
            while j <= topbmn {
                let jb = nb.min(topbmn - j + 1);
                // Factorize JB columns among columns J:N.
                let fjb = dlaqps(
                    m,
                    n - j + 1,
                    j - 1,
                    jb,
                    pm!(a, lda, 1, j),
                    lda,
                    pv!(jpvt, j),
                    pv!(tau, j),
                    pv!(work, j),
                    pv!(work, n + j),
                    pv!(work, 2 * n + 1),
                    pv!(work, 2 * n + jb + 1),
                    n - j + 1,
                );
                j += fjb;
            }
        }
        // Use unblocked code to factor the last or only block.
        if j <= minmn {
            dlaqp2(
                m,
                n - j + 1,
                j - 1,
                pm!(a, lda, 1, j),
                lda,
                pv!(jpvt, j),
                pv!(tau, j),
                pv!(work, j),
                pv!(work, n + j),
                pv!(work, 2 * n + 1),
            );
        }
    }
}
