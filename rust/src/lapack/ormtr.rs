//! `DORMQR` with `SIDE = 'L'` (blocked by `DLARFT` + `DLARFB` when the
//! workspace allows, else `DORM2R` + `DLARF1F`): multiplication by the `Q`
//! of a QR factorization or by its transpose, and `DORMTR` for
//! `SIDE = 'L'`, `UPLO = 'L'`, `TRANS = 'N'` on top of it (the `Q` that
//! `DSYTRD` reduced with).

use super::blas::{daxpy, dcopy, dgemm, dgemv, dger, dscal, dtrmm};

/// `ILAENV(1, 'DORMQR', ...)` and `ILAENV(2, ...)`.
const ORMQR_NB: i32 = 32;
const ORMQR_NBMIN: i32 = 2;
/// `DORMQR`'s `NBMAX`, `LDT = NBMAX + 1` and `TSIZE = LDT * NBMAX`.
const NBMAX: i32 = 64;
const LDT: i32 = NBMAX + 1;
const TSIZE: i32 = LDT * NBMAX;

/// `ILADLC`: the last non-zero column of the `m`-by-`n` matrix `a`.
unsafe fn iladlc(m: i32, n: i32, a: *const f64, lda: i32) -> i32 {
    if n == 0 {
        return n;
    }
    if m!(a, lda, 1, n) != 0.0 || m!(a, lda, m, n) != 0.0 {
        return n;
    }
    let mut c = n;
    while c >= 1 {
        for i in 1..=m {
            if m!(a, lda, i, c) != 0.0 {
                return c;
            }
        }
        c -= 1;
    }
    0
}

/// `DLARF1F` with `SIDE = 'L'`: `C := H C` for `H = I - tau v v'`, `v(1)`
/// taken as 1 and never read.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlarf1f_left(
    m: i32,
    n: i32,
    v: *const f64,
    incv: i32,
    tau: f64,
    c: *mut f64,
    ldc: i32,
    work: *mut f64,
) {
    let mut lastv = 1;
    let mut lastc = 0;
    if tau != 0.0 {
        // Set up variables for scanning V. LASTV begins pointing to the end
        // of V.
        lastv = m;
        let mut i = if incv > 0 { 1 + (lastv - 1) * incv } else { 1 };
        // Look for the last non-zero row in V.
        while lastv > 1 && v!(v, i) == 0.0 {
            lastv -= 1;
            i -= incv;
        }
        // Scan for the last non-zero column in C(1:lastv,:).
        lastc = iladlc(lastv, n, c, ldc);
    }
    if lastc == 0 {
        return;
    }
    if lastv == 1 {
        dscal(lastc, 1.0 - tau, c, ldc);
    } else {
        // w(1:lastc,1) := C(2:lastv,1:lastc)**T * v(2:lastv,1)
        dgemv(
            b'T',
            lastv - 1,
            lastc,
            1.0,
            pm!(c, ldc, 2, 1),
            ldc,
            pv!(v, 1 + incv),
            incv,
            0.0,
            work,
            1,
        );
        // w(1:lastc,1) += C(1,1:lastc)**T * v(1,1) = C(1,1:lastc)**T
        daxpy(lastc, 1.0, c, ldc, work, 1);
        // C(1, 1:lastc) := C(...) - tau * v(1,1) * w(1:lastc,1)**T
        daxpy(lastc, -tau, work, 1, c, ldc);
        // C(2:lastv,1:lastc) := C(...) - tau * v(2:lastv,1)*w(1:lastc,1)**T
        dger(
            lastv - 1,
            lastc,
            -tau,
            pv!(v, 1 + incv),
            incv,
            work,
            1,
            pm!(c, ldc, 2, 1),
            ldc,
        );
    }
}

/// `DORM2R` with `SIDE = 'L'`: `C := Q C` (`trans = b'N'`, the reflectors
/// applied last to first) or `C := Q' C` (`trans = b'T'`, first to last).
#[allow(clippy::too_many_arguments)]
unsafe fn dorm2r_l(
    trans: u8,
    m: i32,
    n: i32,
    k: i32,
    a: *const f64,
    lda: i32,
    tau: *const f64,
    c: *mut f64,
    ldc: i32,
    work: *mut f64,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let ni = n;
    let jc = 1;
    let (i1, i3) = if trans == b'T' { (1, 1) } else { (k, -1) };
    let mut i = i1;
    while (1..=k).contains(&i) {
        // H(i) is applied to C(i:m,1:n)
        let mi = m - i + 1;
        let ic = i;
        dlarf1f_left(
            mi,
            ni,
            pm!(a, lda, i, i),
            1,
            v!(tau, i),
            pm!(c, ldc, ic, jc),
            ldc,
            work,
        );
        i += i3;
    }
}

/// `DLARFT` with `DIRECT = 'F'`, `STOREV = 'C'` (LAPACK 3.12's recursive
/// form): the upper triangular `T` of the block reflector
/// `H = I - V T V'` of the `k` reflectors in the columns of `v`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlarft_fc(
    n: i32,
    k: i32,
    v: *const f64,
    ldv: i32,
    tau: *const f64,
    t: *mut f64,
    ldt: i32,
) {
    if n == 0 || k == 0 {
        return;
    }
    if n == 1 || k == 1 {
        m!(t, ldt, 1, 1) = v!(tau, 1);
        return;
    }
    let l = k / 2;
    // T_{1,1}
    dlarft_fc(n, l, v, ldv, tau, t, ldt);
    // T_{2,2}
    dlarft_fc(
        n - l,
        k - l,
        pm!(v, ldv, l + 1, l + 1),
        ldv,
        pv!(tau, l + 1),
        pm!(t, ldt, l + 1, l + 1),
        ldt,
    );
    // T_{1,2} = V_{2,1}**T
    for j in 1..=l {
        for i in 1..=(k - l) {
            m!(t, ldt, j, l + i) = m!(v, ldv, l + i, j);
        }
    }
    // T_{1,2} = T_{1,2}*V_{2,2}
    dtrmm(
        b'R',
        b'L',
        b'N',
        b'U',
        l,
        k - l,
        1.0,
        pm!(v, ldv, l + 1, l + 1),
        ldv,
        pm!(t, ldt, 1, l + 1),
        ldt,
    );
    // T_{1,2} = V_{3,1}**T*V_{3,2} + T_{1,2}
    dgemm(
        b'T',
        b'N',
        l,
        k - l,
        n - k,
        1.0,
        pm!(v, ldv, k + 1, 1),
        ldv,
        pm!(v, ldv, k + 1, l + 1),
        ldv,
        1.0,
        pm!(t, ldt, 1, l + 1),
        ldt,
    );
    // T_{1,2} = -T_{1,1}*T_{1,2}
    dtrmm(
        b'L',
        b'U',
        b'N',
        b'N',
        l,
        k - l,
        -1.0,
        t,
        ldt,
        pm!(t, ldt, 1, l + 1),
        ldt,
    );
    // T_{1,2} = T_{1,2}*T_{2,2}
    dtrmm(
        b'R',
        b'U',
        b'N',
        b'N',
        l,
        k - l,
        1.0,
        pm!(t, ldt, l + 1, l + 1),
        ldt,
        pm!(t, ldt, 1, l + 1),
        ldt,
    );
}

/// `DLARFB` with `SIDE = 'L'`, `DIRECT = 'F'`, `STOREV = 'C'`: `C := H C`
/// (`trans = b'N'`) or `C := H' C` (`trans = b'T'`) for the block reflector
/// `H = I - V T V'`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlarfb_lfc(
    trans: u8,
    m: i32,
    n: i32,
    k: i32,
    v: *const f64,
    ldv: i32,
    t: *const f64,
    ldt: i32,
    c: *mut f64,
    ldc: i32,
    work: *mut f64,
    ldwork: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    let transt = if trans == b'N' { b'T' } else { b'N' };
    // Form  H * C  or  H**T * C  where  C = ( C1 )
    //                                       ( C2 )
    // W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
    // W := C1**T
    for j in 1..=k {
        dcopy(n, pm!(c, ldc, j, 1), ldc, pm!(work, ldwork, 1, j), 1);
    }
    // W := W * V1
    dtrmm(b'R', b'L', b'N', b'U', n, k, 1.0, v, ldv, work, ldwork);
    if m > k {
        // W := W + C2**T * V2
        dgemm(
            b'T',
            b'N',
            n,
            k,
            m - k,
            1.0,
            pm!(c, ldc, k + 1, 1),
            ldc,
            pm!(v, ldv, k + 1, 1),
            ldv,
            1.0,
            work,
            ldwork,
        );
    }
    // W := W * T**T  or  W * T
    dtrmm(b'R', b'U', transt, b'N', n, k, 1.0, t, ldt, work, ldwork);
    // C := C - V * W**T
    if m > k {
        // C2 := C2 - V2 * W**T
        dgemm(
            b'N',
            b'T',
            m - k,
            n,
            k,
            -1.0,
            pm!(v, ldv, k + 1, 1),
            ldv,
            work,
            ldwork,
            1.0,
            pm!(c, ldc, k + 1, 1),
            ldc,
        );
    }
    // W := W * V1**T
    dtrmm(b'R', b'L', b'T', b'U', n, k, 1.0, v, ldv, work, ldwork);
    // C1 := C1 - W**T
    for j in 1..=k {
        for i in 1..=n {
            m!(c, ldc, j, i) -= m!(work, ldwork, i, j);
        }
    }
}

/// `DORMQR`'s optimal workspace for `SIDE = 'L'` and `n` columns of `C`,
/// what its `lwork = -1` query returns.
pub(crate) fn dormqr_l_lwork(n: i32) -> i32 {
    n.max(1) * NBMAX.min(ORMQR_NB) + TSIZE
}

/// `DORMQR` with `SIDE = 'L'` and `lwork` doubles of workspace: `C := Q C`
/// (`trans = b'N'`, blocks last to first) or `C := Q' C` (`trans = b'T'`,
/// first to last), blocked with `NB = (lwork - TSIZE) / n` when the
/// workspace is short of the optimal size, and unblocked when that is below
/// `NBMIN` or not below `k`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dormqr_l(
    trans: u8,
    m: i32,
    n: i32,
    k: i32,
    a: *const f64,
    lda: i32,
    tau: *const f64,
    c: *mut f64,
    ldc: i32,
    work: *mut f64,
    lwork: i32,
) {
    let nw = n.max(1);
    let mut nb = NBMAX.min(ORMQR_NB);
    let lwkopt = nw * nb + TSIZE;
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mut nbmin = 2;
    let ldwork = nw;
    if nb > 1 && nb < k && lwork < lwkopt {
        nb = (lwork - TSIZE) / ldwork;
        nbmin = 2.max(ORMQR_NBMIN);
    }

    if nb < nbmin || nb >= k {
        // Use unblocked code
        dorm2r_l(trans, m, n, k, a, lda, tau, c, ldc, work);
    } else {
        // Use blocked code
        let iwt = 1 + nw * nb;
        let (i1, i3) = if trans == b'T' {
            (1, nb)
        } else {
            (((k - 1) / nb) * nb + 1, -nb)
        };
        let ni = n;
        let jc = 1;
        let mut i = i1;
        while (1..=k).contains(&i) {
            let ib = nb.min(k - i + 1);
            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1)
            dlarft_fc(
                m - i + 1,
                ib,
                pm!(a, lda, i, i),
                lda,
                pv!(tau, i),
                pv!(work, iwt),
                LDT,
            );
            // H is applied to C(i:m,1:n)
            let mi = m - i + 1;
            let ic = i;
            dlarfb_lfc(
                trans,
                mi,
                ni,
                ib,
                pm!(a, lda, i, i),
                lda,
                pv!(work, iwt),
                LDT,
                pm!(c, ldc, ic, jc),
                ldc,
                work,
                ldwork,
            );
            i += i3;
        }
    }
}

/// `DORMTR('L', 'L', 'N', ...)`: `C := Q C` for the `Q` of `DSYTRD` on the
/// lower triangle (`a`, `tau`), `C` being `m`-by-`n`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dormtr_lln(
    m: i32,
    n: i32,
    a: *const f64,
    lda: i32,
    tau: *const f64,
    c: *mut f64,
    ldc: i32,
    work: *mut f64,
    lwork: i32,
) {
    let nq = m;
    if m == 0 || n == 0 || nq == 1 {
        return;
    }
    let mi = m - 1;
    let ni = n;
    let (i1, i2) = (2, 1);
    dormqr_l(
        b'N',
        mi,
        ni,
        nq - 1,
        pm!(a, lda, 2, 1),
        lda,
        tau,
        pm!(c, ldc, i1, i2),
        ldc,
        work,
        lwork,
    );
}
