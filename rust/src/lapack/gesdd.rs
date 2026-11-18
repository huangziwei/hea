//! `DGESDD`: the singular value decomposition `A = U S V'` by divide and
//! conquer, as R's `La.svd` (`svd()`, `JOBZ = 'S'`, `'A'` or `'N'`) and
//! mgcv's `mgcv_svd_full` (`JOBZ = 'O'`) call it with the workspace their
//! `lwork = -1` query returns. With it the reductions it drives: QR and LQ
//! (`DGEQRF`, `DGELQF`, `DORGQR`, `DORGLQ`), the bidiagonal reduction
//! (`DGEBRD` = `DLABRD` panels + `DGEBD2`) and the application and
//! generation of its orthogonal factors (`DORMBR`, `DORGBR`). None of these
//! routines contains a fused product in R's libRlapack; the arithmetic is
//! in the vendor BLAS they call.

use super::aux::{dlacpy, dlascl_g, dlaset};
use super::blas::{daxpy, dcopy, dgemm, dgemv, dger, dscal, dtrmm};
use super::lasd::dbdsdc;
use super::ormtr::{dlarf1f_left, dlarfb_lfc, dlarft_fc, dormqr_l};
use super::sytrd::dlarfg;
use super::{lsame, PREC, SAFMIN};

/// `ILAENV(1, ...)` for `xGEQRF`, `xGELQF`, `xORGQR`, `xORGLQ`, `xORMQR`,
/// `xORMLQ`, `xORGBR` and `xGEBRD`; `ILAENV(2, ...)`; and `ILAENV(3, ...)`.
const NB: i32 = 32;
const NBMIN: i32 = 2;
const NX: i32 = 128;
/// `DORMLQ`'s `NBMAX`, `LDT = NBMAX + 1` and `TSIZE = LDT * NBMAX`.
const NBMAX: i32 = 64;
const LDT: i32 = NBMAX + 1;
const TSIZE: i32 = LDT * NBMAX;

/// `ILADLR`: the last non-zero row of the `m`-by-`n` matrix `a`.
unsafe fn iladlr(m: i32, n: i32, a: *const f64, lda: i32) -> i32 {
    if m == 0 {
        return m;
    }
    if m!(a, lda, m, 1) != 0.0 || m!(a, lda, m, n) != 0.0 {
        return m;
    }
    let mut r = 0;
    for j in 1..=n {
        let mut i = m;
        while i >= 1 && m!(a, lda, i, j) == 0.0 {
            i -= 1;
        }
        r = r.max(i);
    }
    r
}

/// `DLARF1F` with `SIDE = 'R'`: `C := C H` for `H = I - tau v v'`, `v(1)`
/// taken as 1 and never read.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarf1f_right(
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
        lastv = n;
        let mut i = if incv > 0 { 1 + (lastv - 1) * incv } else { 1 };
        while lastv > 1 && v!(v, i) == 0.0 {
            lastv -= 1;
            i -= incv;
        }
        lastc = iladlr(m, lastv, c, ldc);
    }
    if lastc == 0 {
        return;
    }
    if lastv == 1 {
        dscal(lastc, 1.0 - tau, c, 1);
    } else {
        // w(1:lastc,1) := C(1:lastc,2:lastv) * v(2:lastv,1)
        dgemv(
            b'N',
            lastc,
            lastv - 1,
            1.0,
            pm!(c, ldc, 1, 2),
            ldc,
            pv!(v, 1 + incv),
            incv,
            0.0,
            work,
            1,
        );
        // w(1:lastc,1) += C(1:lastc,1)
        daxpy(lastc, 1.0, c, 1, work, 1);
        // C(1:lastc,1) := C(...) - tau * w(1:lastc,1)
        daxpy(lastc, -tau, work, 1, c, 1);
        // C(1:lastc,2:lastv) := C(...) - tau * w(1:lastc,1) * v(2:lastv)**T
        dger(
            lastc,
            lastv - 1,
            -tau,
            work,
            1,
            pv!(v, 1 + incv),
            incv,
            pm!(c, ldc, 1, 2),
            ldc,
        );
    }
}

/// `DLARFT` with `DIRECT = 'F'`, `STOREV = 'R'` (LAPACK 3.12's recursive
/// form): the upper triangular `T` of the block reflector `H = I - V' T V`
/// of the `k` reflectors in the rows of `v`.
unsafe fn dlarft_fr(
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
    dlarft_fr(n, l, v, ldv, tau, t, ldt);
    dlarft_fr(
        n - l,
        k - l,
        pm!(v, ldv, l + 1, l + 1),
        ldv,
        pv!(tau, l + 1),
        pm!(t, ldt, l + 1, l + 1),
        ldt,
    );
    dlacpy(
        b'A',
        l,
        k - l,
        pm!(v, ldv, 1, l + 1),
        ldv,
        pm!(t, ldt, 1, l + 1),
        ldt,
    );
    dtrmm(
        b'R',
        b'U',
        b'T',
        b'U',
        l,
        k - l,
        1.0,
        pm!(v, ldv, l + 1, l + 1),
        ldv,
        pm!(t, ldt, 1, l + 1),
        ldt,
    );
    dgemm(
        b'N',
        b'T',
        l,
        k - l,
        n - k,
        1.0,
        pm!(v, ldv, 1, k + 1),
        ldv,
        pm!(v, ldv, l + 1, k + 1),
        ldv,
        1.0,
        pm!(t, ldt, 1, l + 1),
        ldt,
    );
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

/// `DLARFB` with `SIDE = 'R'`, `DIRECT = 'F'`, `STOREV = 'R'`: `C := C H`
/// (`trans = b'N'`) or `C := C H'` (`trans = b'T'`) for the block reflector
/// `H = I - V' T V`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarfb_rfr(
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
    // W := C1
    for j in 1..=k {
        dcopy(m, pm!(c, ldc, 1, j), 1, pm!(work, ldwork, 1, j), 1);
    }
    // W := W * V1**T
    dtrmm(b'R', b'U', b'T', b'U', m, k, 1.0, v, ldv, work, ldwork);
    if n > k {
        // W := W + C2 * V2**T
        dgemm(
            b'N',
            b'T',
            m,
            k,
            n - k,
            1.0,
            pm!(c, ldc, 1, k + 1),
            ldc,
            pm!(v, ldv, 1, k + 1),
            ldv,
            1.0,
            work,
            ldwork,
        );
    }
    // W := W * T  or  W * T**T
    dtrmm(b'R', b'U', trans, b'N', m, k, 1.0, t, ldt, work, ldwork);
    if n > k {
        // C2 := C2 - W * V2
        dgemm(
            b'N',
            b'N',
            m,
            n - k,
            k,
            -1.0,
            work,
            ldwork,
            pm!(v, ldv, 1, k + 1),
            ldv,
            1.0,
            pm!(c, ldc, 1, k + 1),
            ldc,
        );
    }
    // W := W * V1
    dtrmm(b'R', b'U', b'N', b'U', m, k, 1.0, v, ldv, work, ldwork);
    // C1 := C1 - W
    for j in 1..=k {
        for i in 1..=m {
            m!(c, ldc, i, j) -= m!(work, ldwork, i, j);
        }
    }
}

/// `DGEQR2`: unblocked QR of the `m`-by-`n` matrix `a`.
unsafe fn dgeqr2(m: i32, n: i32, a: *mut f64, lda: i32, tau: *mut f64, work: *mut f64) {
    let k = m.min(n);
    for i in 1..=k {
        v!(tau, i) = dlarfg(
            m - i + 1,
            pm!(a, lda, i, i),
            pm!(a, lda, (i + 1).min(m), i),
            1,
        );
        if i < n {
            dlarf1f_left(
                m - i + 1,
                n - i,
                pm!(a, lda, i, i),
                1,
                v!(tau, i),
                pm!(a, lda, i, i + 1),
                lda,
                work,
            );
        }
    }
}

/// `DGEQRF`'s optimal workspace, what its `lwork = -1` query returns.
fn dgeqrf_lwork(m: i32, n: i32) -> i32 {
    if m.min(n) == 0 {
        1
    } else {
        n * NB
    }
}

/// `DGEQRF`: QR of the `m`-by-`n` matrix `a` with `lwork` doubles of
/// workspace, in panels of `NB` columns while more than `NX` remain.
unsafe fn dgeqrf(m: i32, n: i32, a: *mut f64, lda: i32, tau: *mut f64, work: *mut f64, lwork: i32) {
    let k = m.min(n);
    if k == 0 {
        return;
    }
    let mut nb = NB;
    let mut nbmin = 2;
    let mut nx = 0;
    let mut ldwork = n;
    if nb > 1 && nb < k {
        nx = 0.max(NX);
        if nx < k {
            ldwork = n;
            let iws = ldwork * nb;
            if lwork < iws {
                nb = lwork / ldwork;
                nbmin = 2.max(NBMIN);
            }
        }
    }
    let mut i = 1;
    if nb >= nbmin && nb < k && nx < k {
        while i <= k - nx {
            let ib = (k - i + 1).min(nb);
            dgeqr2(m - i + 1, ib, pm!(a, lda, i, i), lda, pv!(tau, i), work);
            if i + ib <= n {
                dlarft_fc(
                    m - i + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    pv!(tau, i),
                    work,
                    ldwork,
                );
                dlarfb_lfc(
                    b'T',
                    m - i + 1,
                    n - i - ib + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    work,
                    ldwork,
                    pm!(a, lda, i, i + ib),
                    lda,
                    pv!(work, ib + 1),
                    ldwork,
                );
            }
            i += nb;
        }
    }
    if i <= k {
        dgeqr2(
            m - i + 1,
            n - i + 1,
            pm!(a, lda, i, i),
            lda,
            pv!(tau, i),
            work,
        );
    }
}

/// `DORG2R`: the first `n` columns of the `Q` of `k` reflectors, unblocked.
unsafe fn dorg2r(m: i32, n: i32, k: i32, a: *mut f64, lda: i32, tau: *const f64, work: *mut f64) {
    if n <= 0 {
        return;
    }
    // Initialise columns k+1:n to columns of the unit matrix.
    for j in k + 1..=n {
        for l in 1..=m {
            m!(a, lda, l, j) = 0.0;
        }
        m!(a, lda, j, j) = 1.0;
    }
    let mut i = k;
    while i >= 1 {
        // Apply H(i) to A(i:m,i:n) from the left.
        if i < n {
            dlarf1f_left(
                m - i + 1,
                n - i,
                pm!(a, lda, i, i),
                1,
                v!(tau, i),
                pm!(a, lda, i, i + 1),
                lda,
                work,
            );
        }
        if i < m {
            dscal(m - i, -v!(tau, i), pm!(a, lda, i + 1, i), 1);
        }
        m!(a, lda, i, i) = 1.0 - v!(tau, i);
        // Set A(1:i-1,i) to zero.
        for l in 1..=i - 1 {
            m!(a, lda, l, i) = 0.0;
        }
        i -= 1;
    }
}

/// `DORGQR`'s optimal workspace, what its `lwork = -1` query returns.
fn dorgqr_lwork(n: i32) -> i32 {
    n.max(1) * NB
}

/// `DORGQR`: the first `n` columns of the `Q` of `k` reflectors with
/// `lwork` doubles of workspace.
unsafe fn dorgqr(
    m: i32,
    n: i32,
    k: i32,
    a: *mut f64,
    lda: i32,
    tau: *const f64,
    work: *mut f64,
    lwork: i32,
) {
    if n <= 0 {
        return;
    }
    let mut nb = NB;
    let mut nbmin = 2;
    let mut nx = 0;
    let mut ldwork = n;
    if nb > 1 && nb < k {
        nx = 0.max(NX);
        if nx < k {
            ldwork = n;
            let iws = ldwork * nb;
            if lwork < iws {
                nb = lwork / ldwork;
                nbmin = 2.max(NBMIN);
            }
        }
    }
    let ki;
    let kk;
    if nb >= nbmin && nb < k && nx < k {
        // The last block is handled unblocked; the first kk columns are
        // handled by the blocked method.
        ki = ((k - nx - 1) / nb) * nb;
        kk = k.min(ki + nb);
        for j in kk + 1..=n {
            for i in 1..=kk {
                m!(a, lda, i, j) = 0.0;
            }
        }
    } else {
        ki = 0;
        kk = 0;
    }
    if kk < n {
        dorg2r(
            m - kk,
            n - kk,
            k - kk,
            pm!(a, lda, kk + 1, kk + 1),
            lda,
            pv!(tau, kk + 1),
            work,
        );
    }
    if kk > 0 {
        let mut i = ki + 1;
        while i >= 1 {
            let ib = nb.min(k - i + 1);
            if i + ib <= n {
                dlarft_fc(
                    m - i + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    pv!(tau, i),
                    work,
                    ldwork,
                );
                dlarfb_lfc(
                    b'N',
                    m - i + 1,
                    n - i - ib + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    work,
                    ldwork,
                    pm!(a, lda, i, i + ib),
                    lda,
                    pv!(work, ib + 1),
                    ldwork,
                );
            }
            dorg2r(m - i + 1, ib, ib, pm!(a, lda, i, i), lda, pv!(tau, i), work);
            for j in i..=i + ib - 1 {
                for l in 1..=i - 1 {
                    m!(a, lda, l, j) = 0.0;
                }
            }
            i -= nb;
        }
    }
}

/// `DGELQ2`: unblocked LQ of the `m`-by-`n` matrix `a`.
unsafe fn dgelq2(m: i32, n: i32, a: *mut f64, lda: i32, tau: *mut f64, work: *mut f64) {
    let k = m.min(n);
    for i in 1..=k {
        v!(tau, i) = dlarfg(
            n - i + 1,
            pm!(a, lda, i, i),
            pm!(a, lda, i, (i + 1).min(n)),
            lda,
        );
        if i < m {
            dlarf1f_right(
                m - i,
                n - i + 1,
                pm!(a, lda, i, i),
                lda,
                v!(tau, i),
                pm!(a, lda, i + 1, i),
                lda,
                work,
            );
        }
    }
}

/// `DGELQF`'s optimal workspace, what its `lwork = -1` query returns.
fn dgelqf_lwork(m: i32, n: i32) -> i32 {
    if m.min(n) == 0 {
        1
    } else {
        m * NB
    }
}

/// `DGELQF`: LQ of the `m`-by-`n` matrix `a` with `lwork` doubles of
/// workspace.
unsafe fn dgelqf(m: i32, n: i32, a: *mut f64, lda: i32, tau: *mut f64, work: *mut f64, lwork: i32) {
    let k = m.min(n);
    if k == 0 {
        return;
    }
    let mut nb = NB;
    let mut nbmin = 2;
    let mut nx = 0;
    let mut ldwork = m;
    if nb > 1 && nb < k {
        nx = 0.max(NX);
        if nx < k {
            ldwork = m;
            let iws = ldwork * nb;
            if lwork < iws {
                nb = lwork / ldwork;
                nbmin = 2.max(NBMIN);
            }
        }
    }
    let mut i = 1;
    if nb >= nbmin && nb < k && nx < k {
        while i <= k - nx {
            let ib = (k - i + 1).min(nb);
            dgelq2(ib, n - i + 1, pm!(a, lda, i, i), lda, pv!(tau, i), work);
            if i + ib <= m {
                dlarft_fr(
                    n - i + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    pv!(tau, i),
                    work,
                    ldwork,
                );
                dlarfb_rfr(
                    b'N',
                    m - i - ib + 1,
                    n - i + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    work,
                    ldwork,
                    pm!(a, lda, i + ib, i),
                    lda,
                    pv!(work, ib + 1),
                    ldwork,
                );
            }
            i += nb;
        }
    }
    if i <= k {
        dgelq2(
            m - i + 1,
            n - i + 1,
            pm!(a, lda, i, i),
            lda,
            pv!(tau, i),
            work,
        );
    }
}

/// `DORGL2`: the first `m` rows of the `Q` of `k` row reflectors,
/// unblocked.
unsafe fn dorgl2(m: i32, n: i32, k: i32, a: *mut f64, lda: i32, tau: *const f64, work: *mut f64) {
    if m <= 0 {
        return;
    }
    if k < m {
        // Initialise rows k+1:m to rows of the unit matrix.
        for j in 1..=n {
            for l in k + 1..=m {
                m!(a, lda, l, j) = 0.0;
            }
            if j > k && j <= m {
                m!(a, lda, j, j) = 1.0;
            }
        }
    }
    let mut i = k;
    while i >= 1 {
        // Apply H(i) to A(i:m,i:n) from the right.
        if i < n {
            if i < m {
                dlarf1f_right(
                    m - i,
                    n - i + 1,
                    pm!(a, lda, i, i),
                    lda,
                    v!(tau, i),
                    pm!(a, lda, i + 1, i),
                    lda,
                    work,
                );
            }
            dscal(n - i, -v!(tau, i), pm!(a, lda, i, i + 1), lda);
        }
        m!(a, lda, i, i) = 1.0 - v!(tau, i);
        // Set A(i,1:i-1) to zero.
        for l in 1..=i - 1 {
            m!(a, lda, i, l) = 0.0;
        }
        i -= 1;
    }
}

/// `DORGLQ`'s optimal workspace, what its `lwork = -1` query returns.
fn dorglq_lwork(m: i32) -> i32 {
    m.max(1) * NB
}

/// `DORGLQ`: the first `m` rows of the `Q` of `k` row reflectors with
/// `lwork` doubles of workspace.
unsafe fn dorglq(
    m: i32,
    n: i32,
    k: i32,
    a: *mut f64,
    lda: i32,
    tau: *const f64,
    work: *mut f64,
    lwork: i32,
) {
    if m <= 0 {
        return;
    }
    let mut nb = NB;
    let mut nbmin = 2;
    let mut nx = 0;
    let mut ldwork = m;
    if nb > 1 && nb < k {
        nx = 0.max(NX);
        if nx < k {
            ldwork = m;
            let iws = ldwork * nb;
            if lwork < iws {
                nb = lwork / ldwork;
                nbmin = 2.max(NBMIN);
            }
        }
    }
    let ki;
    let kk;
    if nb >= nbmin && nb < k && nx < k {
        ki = ((k - nx - 1) / nb) * nb;
        kk = k.min(ki + nb);
        for j in 1..=kk {
            for i in kk + 1..=m {
                m!(a, lda, i, j) = 0.0;
            }
        }
    } else {
        ki = 0;
        kk = 0;
    }
    if kk < m {
        dorgl2(
            m - kk,
            n - kk,
            k - kk,
            pm!(a, lda, kk + 1, kk + 1),
            lda,
            pv!(tau, kk + 1),
            work,
        );
    }
    if kk > 0 {
        let mut i = ki + 1;
        while i >= 1 {
            let ib = nb.min(k - i + 1);
            if i + ib <= m {
                dlarft_fr(
                    n - i + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    pv!(tau, i),
                    work,
                    ldwork,
                );
                dlarfb_rfr(
                    b'T',
                    m - i - ib + 1,
                    n - i + 1,
                    ib,
                    pm!(a, lda, i, i),
                    lda,
                    work,
                    ldwork,
                    pm!(a, lda, i + ib, i),
                    lda,
                    pv!(work, ib + 1),
                    ldwork,
                );
            }
            dorgl2(ib, n - i + 1, ib, pm!(a, lda, i, i), lda, pv!(tau, i), work);
            for j in 1..=i - 1 {
                for l in i..=i + ib - 1 {
                    m!(a, lda, l, j) = 0.0;
                }
            }
            i -= nb;
        }
    }
}

/// `DORML2` with `SIDE = 'R'`: `C := C Q` (`trans = b'N'`) or `C := C Q'`
/// (`trans = b'T'`) for the `Q` of `k` row reflectors, unblocked.
#[allow(clippy::too_many_arguments)]
unsafe fn dorml2_r(
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
    let notran = lsame(trans, b'N');
    let (i1, i3) = if !notran { (1, 1) } else { (k, -1) };
    let mi = m;
    let ic = 1;
    let mut i = i1;
    while (1..=k).contains(&i) {
        let ni = n - i + 1;
        let jc = i;
        dlarf1f_right(
            mi,
            ni,
            pm!(a, lda, i, i),
            lda,
            v!(tau, i),
            pm!(c, ldc, ic, jc),
            ldc,
            work,
        );
        i += i3;
    }
}

/// `DORMLQ` with `SIDE = 'R'` and `lwork` doubles of workspace: `C := C Q`
/// or `C := C Q'`, blocked when the workspace allows.
#[allow(clippy::too_many_arguments)]
unsafe fn dormlq_r(
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
    let notran = lsame(trans, b'N');
    let nq = n;
    let nw = m.max(1);
    let mut nb = NBMAX.min(NB);
    let lwkopt = nw * nb + TSIZE;
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mut nbmin = 2;
    let ldwork = nw;
    if nb > 1 && nb < k && lwork < lwkopt {
        nb = (lwork - TSIZE) / ldwork;
        nbmin = 2.max(NBMIN);
    }
    if nb < nbmin || nb >= k {
        dorml2_r(trans, m, n, k, a, lda, tau, c, ldc, work);
    } else {
        let iwt = 1 + nw * nb;
        let (i1, i3) = if !notran {
            (1, nb)
        } else {
            (((k - 1) / nb) * nb + 1, -nb)
        };
        let mi = m;
        let ic = 1;
        let transt = if notran { b'T' } else { b'N' };
        let mut i = i1;
        while (1..=k).contains(&i) {
            let ib = nb.min(k - i + 1);
            // The triangular factor of the block reflector H(i) .. H(i+ib-1).
            dlarft_fr(
                nq - i + 1,
                ib,
                pm!(a, lda, i, i),
                lda,
                pv!(tau, i),
                pv!(work, iwt),
                LDT,
            );
            let ni = n - i + 1;
            let jc = i;
            dlarfb_rfr(
                transt,
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

/// `DORMBR`'s optimal workspace for `SIDE = side` on an `m`-by-`n` `C`.
fn dormbr_lwork(side: u8, m: i32, n: i32) -> i32 {
    let nw = if side == b'L' { n.max(1) } else { m.max(1) };
    nw * NB
}

/// `DORMBR` in the two forms `DGESDD` uses: `VECT = 'Q'` with `SIDE = 'L'`
/// (`C := Q C` or `Q' C`, the `Q` of `DGEBRD`) and `VECT = 'P'` with
/// `SIDE = 'R'` (`C := C P'` for `trans = b'T'`, or `C P`).
#[allow(clippy::too_many_arguments)]
unsafe fn dormbr(
    vect: u8,
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
    if m == 0 || n == 0 {
        return;
    }
    if vect == b'Q' {
        let nq = m;
        if nq >= k {
            dormqr_l(trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
        } else if nq > 1 {
            dormqr_l(
                trans,
                m - 1,
                n,
                nq - 1,
                pm!(a, lda, 2, 1),
                lda,
                tau,
                pm!(c, ldc, 2, 1),
                ldc,
                work,
                lwork,
            );
        }
    } else {
        let nq = n;
        let transt = if lsame(trans, b'N') { b'T' } else { b'N' };
        if nq > k {
            dormlq_r(transt, m, n, k, a, lda, tau, c, ldc, work, lwork);
        } else if nq > 1 {
            dormlq_r(
                transt,
                m,
                n - 1,
                nq - 1,
                pm!(a, lda, 1, 2),
                lda,
                tau,
                pm!(c, ldc, 1, 2),
                ldc,
                work,
                lwork,
            );
        }
    }
}

/// `DORGBR`: the orthogonal `Q` (`vect = b'Q'`) or `P'` (`b'P'`) that
/// `DGEBRD` reduced with, generated in place in `a`.
#[allow(clippy::too_many_arguments)]
unsafe fn dorgbr(
    vect: u8,
    m: i32,
    n: i32,
    k: i32,
    a: *mut f64,
    lda: i32,
    tau: *const f64,
    work: *mut f64,
    lwork: i32,
) {
    if m == 0 || n == 0 {
        return;
    }
    if vect == b'Q' {
        if m >= k {
            dorgqr(m, n, k, a, lda, tau, work, lwork);
        } else {
            // Shift the reflectors one column to the right and set the first
            // row and column of Q to those of the unit matrix.
            let mut j = m;
            while j >= 2 {
                m!(a, lda, 1, j) = 0.0;
                for i in j + 1..=m {
                    m!(a, lda, i, j) = m!(a, lda, i, j - 1);
                }
                j -= 1;
            }
            m!(a, lda, 1, 1) = 1.0;
            for i in 2..=m {
                m!(a, lda, i, 1) = 0.0;
            }
            if m > 1 {
                dorgqr(
                    m - 1,
                    m - 1,
                    m - 1,
                    pm!(a, lda, 2, 2),
                    lda,
                    tau,
                    work,
                    lwork,
                );
            }
        }
    } else if k < n {
        dorglq(m, n, k, a, lda, tau, work, lwork);
    } else {
        // Shift the reflectors one row downward and set the first row and
        // column of P' to those of the unit matrix.
        m!(a, lda, 1, 1) = 1.0;
        for i in 2..=n {
            m!(a, lda, i, 1) = 0.0;
        }
        for j in 2..=n {
            let mut i = j - 1;
            while i >= 2 {
                m!(a, lda, i, j) = m!(a, lda, i - 1, j);
                i -= 1;
            }
            m!(a, lda, 1, j) = 0.0;
        }
        if n > 1 {
            dorglq(
                n - 1,
                n - 1,
                n - 1,
                pm!(a, lda, 2, 2),
                lda,
                tau,
                work,
                lwork,
            );
        }
    }
}

/// `DGEBD2`: unblocked reduction of the `m`-by-`n` `a` to bidiagonal form
/// `Q' A P = B`, upper for `m >= n`, lower otherwise.
#[allow(clippy::too_many_arguments)]
unsafe fn dgebd2(
    m: i32,
    n: i32,
    a: *mut f64,
    lda: i32,
    d: *mut f64,
    e: *mut f64,
    tauq: *mut f64,
    taup: *mut f64,
    work: *mut f64,
) {
    if m >= n {
        for i in 1..=n {
            // H(i) annihilates A(i+1:m,i).
            v!(tauq, i) = dlarfg(
                m - i + 1,
                pm!(a, lda, i, i),
                pm!(a, lda, (i + 1).min(m), i),
                1,
            );
            v!(d, i) = m!(a, lda, i, i);
            if i < n {
                dlarf1f_left(
                    m - i + 1,
                    n - i,
                    pm!(a, lda, i, i),
                    1,
                    v!(tauq, i),
                    pm!(a, lda, i, i + 1),
                    lda,
                    work,
                );
            }
            if i < n {
                // G(i) annihilates A(i,i+2:n).
                v!(taup, i) = dlarfg(
                    n - i,
                    pm!(a, lda, i, i + 1),
                    pm!(a, lda, i, (i + 2).min(n)),
                    lda,
                );
                v!(e, i) = m!(a, lda, i, i + 1);
                dlarf1f_right(
                    m - i,
                    n - i,
                    pm!(a, lda, i, i + 1),
                    lda,
                    v!(taup, i),
                    pm!(a, lda, i + 1, i + 1),
                    lda,
                    work,
                );
            } else {
                v!(taup, i) = 0.0;
            }
        }
    } else {
        for i in 1..=m {
            // G(i) annihilates A(i,i+1:n).
            v!(taup, i) = dlarfg(
                n - i + 1,
                pm!(a, lda, i, i),
                pm!(a, lda, i, (i + 1).min(n)),
                lda,
            );
            v!(d, i) = m!(a, lda, i, i);
            if i < m {
                dlarf1f_right(
                    m - i,
                    n - i + 1,
                    pm!(a, lda, i, i),
                    lda,
                    v!(taup, i),
                    pm!(a, lda, i + 1, i),
                    lda,
                    work,
                );
            }
            if i < m {
                // H(i) annihilates A(i+2:m,i).
                v!(tauq, i) = dlarfg(
                    m - i,
                    pm!(a, lda, i + 1, i),
                    pm!(a, lda, (i + 2).min(m), i),
                    1,
                );
                v!(e, i) = m!(a, lda, i + 1, i);
                dlarf1f_left(
                    m - i,
                    n - i,
                    pm!(a, lda, i + 1, i),
                    1,
                    v!(tauq, i),
                    pm!(a, lda, i + 1, i + 1),
                    lda,
                    work,
                );
            } else {
                v!(tauq, i) = 0.0;
            }
        }
    }
}

/// `DLABRD`: reduces the first `nb` rows and columns of `a` to bidiagonal
/// form and returns the `X` and `Y` that update the rest
/// (`A := A - V Y' - X U'`).
#[allow(clippy::too_many_arguments)]
unsafe fn dlabrd(
    m: i32,
    n: i32,
    nb: i32,
    a: *mut f64,
    lda: i32,
    d: *mut f64,
    e: *mut f64,
    tauq: *mut f64,
    taup: *mut f64,
    x: *mut f64,
    ldx: i32,
    y: *mut f64,
    ldy: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if m >= n {
        // Reduce to upper bidiagonal form.
        for i in 1..=nb {
            // Update A(i:m,i).
            dgemv(
                b'N',
                m - i + 1,
                i - 1,
                -1.0,
                pm!(a, lda, i, 1),
                lda,
                pm!(y, ldy, i, 1),
                ldy,
                1.0,
                pm!(a, lda, i, i),
                1,
            );
            dgemv(
                b'N',
                m - i + 1,
                i - 1,
                -1.0,
                pm!(x, ldx, i, 1),
                ldx,
                pm!(a, lda, 1, i),
                1,
                1.0,
                pm!(a, lda, i, i),
                1,
            );
            // Generate the reflection Q(i) to annihilate A(i+1:m,i).
            v!(tauq, i) = dlarfg(
                m - i + 1,
                pm!(a, lda, i, i),
                pm!(a, lda, (i + 1).min(m), i),
                1,
            );
            v!(d, i) = m!(a, lda, i, i);
            if i < n {
                m!(a, lda, i, i) = 1.0;
                // Compute Y(i+1:n,i).
                dgemv(
                    b'T',
                    m - i + 1,
                    n - i,
                    1.0,
                    pm!(a, lda, i, i + 1),
                    lda,
                    pm!(a, lda, i, i),
                    1,
                    0.0,
                    pm!(y, ldy, i + 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    m - i + 1,
                    i - 1,
                    1.0,
                    pm!(a, lda, i, 1),
                    lda,
                    pm!(a, lda, i, i),
                    1,
                    0.0,
                    pm!(y, ldy, 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    n - i,
                    i - 1,
                    -1.0,
                    pm!(y, ldy, i + 1, 1),
                    ldy,
                    pm!(y, ldy, 1, i),
                    1,
                    1.0,
                    pm!(y, ldy, i + 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    m - i + 1,
                    i - 1,
                    1.0,
                    pm!(x, ldx, i, 1),
                    ldx,
                    pm!(a, lda, i, i),
                    1,
                    0.0,
                    pm!(y, ldy, 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    i - 1,
                    n - i,
                    -1.0,
                    pm!(a, lda, 1, i + 1),
                    lda,
                    pm!(y, ldy, 1, i),
                    1,
                    1.0,
                    pm!(y, ldy, i + 1, i),
                    1,
                );
                dscal(n - i, v!(tauq, i), pm!(y, ldy, i + 1, i), 1);
                // Update A(i,i+1:n).
                dgemv(
                    b'N',
                    n - i,
                    i,
                    -1.0,
                    pm!(y, ldy, i + 1, 1),
                    ldy,
                    pm!(a, lda, i, 1),
                    lda,
                    1.0,
                    pm!(a, lda, i, i + 1),
                    lda,
                );
                dgemv(
                    b'T',
                    i - 1,
                    n - i,
                    -1.0,
                    pm!(a, lda, 1, i + 1),
                    lda,
                    pm!(x, ldx, i, 1),
                    ldx,
                    1.0,
                    pm!(a, lda, i, i + 1),
                    lda,
                );
                // Generate the reflection P(i) to annihilate A(i,i+2:n).
                v!(taup, i) = dlarfg(
                    n - i,
                    pm!(a, lda, i, i + 1),
                    pm!(a, lda, i, (i + 2).min(n)),
                    lda,
                );
                v!(e, i) = m!(a, lda, i, i + 1);
                m!(a, lda, i, i + 1) = 1.0;
                // Compute X(i+1:m,i).
                dgemv(
                    b'N',
                    m - i,
                    n - i,
                    1.0,
                    pm!(a, lda, i + 1, i + 1),
                    lda,
                    pm!(a, lda, i, i + 1),
                    lda,
                    0.0,
                    pm!(x, ldx, i + 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    n - i,
                    i,
                    1.0,
                    pm!(y, ldy, i + 1, 1),
                    ldy,
                    pm!(a, lda, i, i + 1),
                    lda,
                    0.0,
                    pm!(x, ldx, 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    m - i,
                    i,
                    -1.0,
                    pm!(a, lda, i + 1, 1),
                    lda,
                    pm!(x, ldx, 1, i),
                    1,
                    1.0,
                    pm!(x, ldx, i + 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    i - 1,
                    n - i,
                    1.0,
                    pm!(a, lda, 1, i + 1),
                    lda,
                    pm!(a, lda, i, i + 1),
                    lda,
                    0.0,
                    pm!(x, ldx, 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    m - i,
                    i - 1,
                    -1.0,
                    pm!(x, ldx, i + 1, 1),
                    ldx,
                    pm!(x, ldx, 1, i),
                    1,
                    1.0,
                    pm!(x, ldx, i + 1, i),
                    1,
                );
                dscal(m - i, v!(taup, i), pm!(x, ldx, i + 1, i), 1);
            }
        }
    } else {
        // Reduce to lower bidiagonal form.
        for i in 1..=nb {
            // Update A(i,i:n).
            dgemv(
                b'N',
                n - i + 1,
                i - 1,
                -1.0,
                pm!(y, ldy, i, 1),
                ldy,
                pm!(a, lda, i, 1),
                lda,
                1.0,
                pm!(a, lda, i, i),
                lda,
            );
            dgemv(
                b'T',
                i - 1,
                n - i + 1,
                -1.0,
                pm!(a, lda, 1, i),
                lda,
                pm!(x, ldx, i, 1),
                ldx,
                1.0,
                pm!(a, lda, i, i),
                lda,
            );
            // Generate the reflection P(i) to annihilate A(i,i+1:n).
            v!(taup, i) = dlarfg(
                n - i + 1,
                pm!(a, lda, i, i),
                pm!(a, lda, i, (i + 1).min(n)),
                lda,
            );
            v!(d, i) = m!(a, lda, i, i);
            if i < m {
                m!(a, lda, i, i) = 1.0;
                // Compute X(i+1:m,i).
                dgemv(
                    b'N',
                    m - i,
                    n - i + 1,
                    1.0,
                    pm!(a, lda, i + 1, i),
                    lda,
                    pm!(a, lda, i, i),
                    lda,
                    0.0,
                    pm!(x, ldx, i + 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    n - i + 1,
                    i - 1,
                    1.0,
                    pm!(y, ldy, i, 1),
                    ldy,
                    pm!(a, lda, i, i),
                    lda,
                    0.0,
                    pm!(x, ldx, 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    m - i,
                    i - 1,
                    -1.0,
                    pm!(a, lda, i + 1, 1),
                    lda,
                    pm!(x, ldx, 1, i),
                    1,
                    1.0,
                    pm!(x, ldx, i + 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    i - 1,
                    n - i + 1,
                    1.0,
                    pm!(a, lda, 1, i),
                    lda,
                    pm!(a, lda, i, i),
                    lda,
                    0.0,
                    pm!(x, ldx, 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    m - i,
                    i - 1,
                    -1.0,
                    pm!(x, ldx, i + 1, 1),
                    ldx,
                    pm!(x, ldx, 1, i),
                    1,
                    1.0,
                    pm!(x, ldx, i + 1, i),
                    1,
                );
                dscal(m - i, v!(taup, i), pm!(x, ldx, i + 1, i), 1);
                // Update A(i+1:m,i).
                dgemv(
                    b'N',
                    m - i,
                    i - 1,
                    -1.0,
                    pm!(a, lda, i + 1, 1),
                    lda,
                    pm!(y, ldy, i, 1),
                    ldy,
                    1.0,
                    pm!(a, lda, i + 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    m - i,
                    i,
                    -1.0,
                    pm!(x, ldx, i + 1, 1),
                    ldx,
                    pm!(a, lda, 1, i),
                    1,
                    1.0,
                    pm!(a, lda, i + 1, i),
                    1,
                );
                // Generate the reflection Q(i) to annihilate A(i+2:m,i).
                v!(tauq, i) = dlarfg(
                    m - i,
                    pm!(a, lda, i + 1, i),
                    pm!(a, lda, (i + 2).min(m), i),
                    1,
                );
                v!(e, i) = m!(a, lda, i + 1, i);
                m!(a, lda, i + 1, i) = 1.0;
                // Compute Y(i+1:n,i).
                dgemv(
                    b'T',
                    m - i,
                    n - i,
                    1.0,
                    pm!(a, lda, i + 1, i + 1),
                    lda,
                    pm!(a, lda, i + 1, i),
                    1,
                    0.0,
                    pm!(y, ldy, i + 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    m - i,
                    i - 1,
                    1.0,
                    pm!(a, lda, i + 1, 1),
                    lda,
                    pm!(a, lda, i + 1, i),
                    1,
                    0.0,
                    pm!(y, ldy, 1, i),
                    1,
                );
                dgemv(
                    b'N',
                    n - i,
                    i - 1,
                    -1.0,
                    pm!(y, ldy, i + 1, 1),
                    ldy,
                    pm!(y, ldy, 1, i),
                    1,
                    1.0,
                    pm!(y, ldy, i + 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    m - i,
                    i,
                    1.0,
                    pm!(x, ldx, i + 1, 1),
                    ldx,
                    pm!(a, lda, i + 1, i),
                    1,
                    0.0,
                    pm!(y, ldy, 1, i),
                    1,
                );
                dgemv(
                    b'T',
                    i,
                    n - i,
                    -1.0,
                    pm!(a, lda, 1, i + 1),
                    lda,
                    pm!(y, ldy, 1, i),
                    1,
                    1.0,
                    pm!(y, ldy, i + 1, i),
                    1,
                );
                dscal(n - i, v!(tauq, i), pm!(y, ldy, i + 1, i), 1);
            }
        }
    }
}

/// `DGEBRD`'s optimal workspace, what its `lwork = -1` query returns.
fn dgebrd_lwork(m: i32, n: i32) -> i32 {
    if m.min(n) == 0 {
        1
    } else {
        (m + n) * NB.max(1)
    }
}

/// `DGEBRD`: reduction of the `m`-by-`n` `a` to bidiagonal form with
/// `lwork` doubles of workspace, `DLABRD` panels of `NB` while more than
/// `NX` rows and columns remain, then `DGEBD2`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgebrd(
    m: i32,
    n: i32,
    a: *mut f64,
    lda: i32,
    d: *mut f64,
    e: *mut f64,
    tauq: *mut f64,
    taup: *mut f64,
    work: *mut f64,
    lwork: i32,
) {
    let minmn = m.min(n);
    if minmn == 0 {
        return;
    }
    let mut nb = NB.max(1);
    let lwkopt = (m + n) * nb;
    let ldwrkx = m;
    let ldwrky = n;
    let mut nx;
    if nb > 1 && nb < minmn {
        // The crossover point from blocked to unblocked code.
        nx = nb.max(NX);
        if nx < minmn {
            let ws = lwkopt;
            if lwork < ws {
                // Not enough workspace for the optimal NB.
                let nbmin = NBMIN;
                if lwork >= (m + n) * nbmin {
                    nb = lwork / (m + n);
                } else {
                    nb = 1;
                    nx = minmn;
                }
            }
        }
    } else {
        nx = minmn;
    }
    let mut i = 1;
    while i <= minmn - nx {
        // Reduce rows and columns i:i+nb-1 to bidiagonal form and return the
        // matrices X and Y needed to update the unreduced part.
        dlabrd(
            m - i + 1,
            n - i + 1,
            nb,
            pm!(a, lda, i, i),
            lda,
            pv!(d, i),
            pv!(e, i),
            pv!(tauq, i),
            pv!(taup, i),
            work,
            ldwrkx,
            pv!(work, ldwrkx * nb + 1),
            ldwrky,
        );
        // Update the trailing submatrix A(i+nb:m,i+nb:n): A := A - V Y' - X U'.
        dgemm(
            b'N',
            b'T',
            m - i - nb + 1,
            n - i - nb + 1,
            nb,
            -1.0,
            pm!(a, lda, i + nb, i),
            lda,
            pv!(work, ldwrkx * nb + nb + 1),
            ldwrky,
            1.0,
            pm!(a, lda, i + nb, i + nb),
            lda,
        );
        dgemm(
            b'N',
            b'N',
            m - i - nb + 1,
            n - i - nb + 1,
            nb,
            -1.0,
            pv!(work, nb + 1),
            ldwrkx,
            pm!(a, lda, i, i + nb),
            lda,
            1.0,
            pm!(a, lda, i + nb, i + nb),
            lda,
        );
        // Copy the diagonal and off-diagonal elements of B back into A.
        if m >= n {
            for j in i..=i + nb - 1 {
                m!(a, lda, j, j) = v!(d, j);
                m!(a, lda, j, j + 1) = v!(e, j);
            }
        } else {
            for j in i..=i + nb - 1 {
                m!(a, lda, j, j) = v!(d, j);
                m!(a, lda, j + 1, j) = v!(e, j);
            }
        }
        i += nb;
    }
    // Use unblocked code to reduce the remainder of the matrix.
    dgebd2(
        m - i + 1,
        n - i + 1,
        pm!(a, lda, i, i),
        lda,
        pv!(d, i),
        pv!(e, i),
        pv!(tauq, i),
        pv!(taup, i),
        work,
    );
}

/// `DLANGE('M', ...)`: the largest absolute entry of `a` (a NaN entry is
/// returned as the norm).
unsafe fn dlange_m(m: i32, n: i32, a: *const f64, lda: i32) -> f64 {
    if m.min(n) == 0 {
        return 0.0;
    }
    let mut value = 0.0f64;
    for j in 1..=n {
        for i in 1..=m {
            let temp = m!(a, lda, i, j).abs();
            if value < temp || temp.is_nan() {
                value = temp;
            }
        }
    }
    value
}

/// `DGESDD`'s optimal workspace for `jobz` on an `m`-by-`n` matrix, what
/// its `lwork = -1` query returns (`DROUNDUP_LWORK` of `MAXWRK`, exact for
/// any workspace that fits in an `i32`).
pub(crate) fn dgesdd_lwork(jobz: u8, m: i32, n: i32) -> i32 {
    let minmn = m.min(n);
    let wntqa = lsame(jobz, b'A');
    let wntqs = lsame(jobz, b'S');
    let wntqo = lsame(jobz, b'O');
    let wntqn = lsame(jobz, b'N');
    let mut minwrk = 1;
    let mut maxwrk = 1;
    let mnthr = (minmn as f64 * 11.0 / 6.0) as i32;
    if m >= n && minmn > 0 {
        let bdspac = if wntqn { 7 * n } else { 3 * n * n + 4 * n };
        let lwork_dgebrd_mn = dgebrd_lwork(m, n);
        let lwork_dgebrd_nn = dgebrd_lwork(n, n);
        let lwork_dgeqrf_mn = dgeqrf_lwork(m, n);
        let lwork_dorgqr_mm = dorgqr_lwork(m);
        let lwork_dorgqr_mn = dorgqr_lwork(n);
        let lwork_dormbr_prt_nn = dormbr_lwork(b'R', n, n);
        let lwork_dormbr_qln_nn = dormbr_lwork(b'L', n, n);
        let lwork_dormbr_qln_mn = dormbr_lwork(b'L', m, n);
        let lwork_dormbr_qln_mm = dormbr_lwork(b'L', m, m);
        if m >= mnthr {
            if wntqn {
                let mut wrkbl = n + lwork_dgeqrf_mn;
                wrkbl = wrkbl.max(3 * n + lwork_dgebrd_nn);
                maxwrk = wrkbl.max(bdspac + n);
                minwrk = bdspac + n;
            } else {
                let mut wrkbl = n + lwork_dgeqrf_mn;
                wrkbl = wrkbl.max(
                    n + if wntqa {
                        lwork_dorgqr_mm
                    } else {
                        lwork_dorgqr_mn
                    },
                );
                wrkbl = wrkbl.max(3 * n + lwork_dgebrd_nn);
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_qln_nn);
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_prt_nn);
                wrkbl = wrkbl.max(3 * n + bdspac);
                if wntqo {
                    maxwrk = wrkbl + 2 * n * n;
                    minwrk = bdspac + 2 * n * n + 3 * n;
                } else if wntqs {
                    maxwrk = wrkbl + n * n;
                    minwrk = bdspac + n * n + 3 * n;
                } else if wntqa {
                    maxwrk = wrkbl + n * n;
                    minwrk = n * n + (3 * n + bdspac).max(n + m);
                }
            }
        } else {
            let mut wrkbl = 3 * n + lwork_dgebrd_mn;
            if wntqn {
                maxwrk = wrkbl.max(3 * n + bdspac);
                minwrk = 3 * n + m.max(bdspac);
            } else if wntqo {
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_prt_nn);
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_qln_mn);
                wrkbl = wrkbl.max(3 * n + bdspac);
                maxwrk = wrkbl + m * n;
                minwrk = 3 * n + m.max(n * n + bdspac);
            } else if wntqs {
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_qln_mn);
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_prt_nn);
                maxwrk = wrkbl.max(3 * n + bdspac);
                minwrk = 3 * n + m.max(bdspac);
            } else if wntqa {
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_qln_mm);
                wrkbl = wrkbl.max(3 * n + lwork_dormbr_prt_nn);
                maxwrk = wrkbl.max(3 * n + bdspac);
                minwrk = 3 * n + m.max(bdspac);
            }
        }
    } else if minmn > 0 {
        let bdspac = if wntqn { 7 * m } else { 3 * m * m + 4 * m };
        let lwork_dgebrd_mn = dgebrd_lwork(m, n);
        let lwork_dgebrd_mm = dgebrd_lwork(m, m);
        let lwork_dgelqf_mn = dgelqf_lwork(m, n);
        let lwork_dorglq_nn = dorglq_lwork(n);
        let lwork_dorglq_mn = dorglq_lwork(m);
        let lwork_dormbr_prt_mm = dormbr_lwork(b'R', m, m);
        let lwork_dormbr_prt_mn = dormbr_lwork(b'R', m, n);
        let lwork_dormbr_prt_nn = dormbr_lwork(b'R', n, n);
        let lwork_dormbr_qln_mm = dormbr_lwork(b'L', m, m);
        if n >= mnthr {
            if wntqn {
                let mut wrkbl = m + lwork_dgelqf_mn;
                wrkbl = wrkbl.max(3 * m + lwork_dgebrd_mm);
                maxwrk = wrkbl.max(bdspac + m);
                minwrk = bdspac + m;
            } else {
                let mut wrkbl = m + lwork_dgelqf_mn;
                wrkbl = wrkbl.max(
                    m + if wntqa {
                        lwork_dorglq_nn
                    } else {
                        lwork_dorglq_mn
                    },
                );
                wrkbl = wrkbl.max(3 * m + lwork_dgebrd_mm);
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_qln_mm);
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_prt_mm);
                wrkbl = wrkbl.max(3 * m + bdspac);
                if wntqo {
                    maxwrk = wrkbl + 2 * m * m;
                    minwrk = bdspac + 2 * m * m + 3 * m;
                } else if wntqs {
                    maxwrk = wrkbl + m * m;
                    minwrk = bdspac + m * m + 3 * m;
                } else if wntqa {
                    maxwrk = wrkbl + m * m;
                    minwrk = m * m + (3 * m + bdspac).max(m + n);
                }
            }
        } else {
            let mut wrkbl = 3 * m + lwork_dgebrd_mn;
            if wntqn {
                maxwrk = wrkbl.max(3 * m + bdspac);
                minwrk = 3 * m + n.max(bdspac);
            } else if wntqo {
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_qln_mm);
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_prt_mn);
                wrkbl = wrkbl.max(3 * m + bdspac);
                maxwrk = wrkbl + m * n;
                minwrk = 3 * m + n.max(m * m + bdspac);
            } else if wntqs {
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_qln_mm);
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_prt_mn);
                maxwrk = wrkbl.max(3 * m + bdspac);
                minwrk = 3 * m + n.max(bdspac);
            } else if wntqa {
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_qln_mm);
                wrkbl = wrkbl.max(3 * m + lwork_dormbr_prt_nn);
                maxwrk = wrkbl.max(3 * m + bdspac);
                minwrk = 3 * m + n.max(bdspac);
            }
        }
    }
    maxwrk.max(minwrk)
}

/// `DGESDD`: the SVD of the `m`-by-`n` `a` by divide and conquer, with
/// `lwork` doubles of `work` and `8 min(m, n)` integers of `iwork`.
/// `jobz = b'A'`: all of `U` (`m`-by-`m`) and `V'` (`n`-by-`n`); `b'S'`:
/// their first `min(m, n)` columns and rows; `b'O'`: for `m >= n` the
/// columns of `U` overwrite `a` and `vt` holds `V'` (for `m < n` the rows of
/// `V'` overwrite `a` and `u` holds `U`); `b'N'`: the singular values only.
/// The singular values are in decreasing order in `s`. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgesdd(
    jobz: u8,
    m: i32,
    n: i32,
    a: *mut f64,
    lda: i32,
    s: *mut f64,
    u: *mut f64,
    ldu: i32,
    vt: *mut f64,
    ldvt: i32,
    work: *mut f64,
    lwork: i32,
    iwork: *mut i32,
) -> i32 {
    let minmn = m.min(n);
    let wntqa = lsame(jobz, b'A');
    let wntqs = lsame(jobz, b'S');
    let wntqas = wntqa || wntqs;
    let wntqo = lsame(jobz, b'O');
    let wntqn = lsame(jobz, b'N');
    if !(wntqa || wntqs || wntqo || wntqn) {
        return -1;
    } else if m < 0 {
        return -2;
    } else if n < 0 {
        return -3;
    } else if lda < 1.max(m) {
        return -5;
    } else if ldu < 1 || (wntqas && ldu < m) || (wntqo && m < n && ldu < m) {
        return -8;
    } else if ldvt < 1
        || (wntqa && ldvt < n)
        || (wntqs && ldvt < minmn)
        || (wntqo && m >= n && ldvt < n)
    {
        return -10;
    }
    if m == 0 || n == 0 {
        return 0;
    }
    let mnthr = (minmn as f64 * 11.0 / 6.0) as i32;
    let bdspac = if m >= n {
        if wntqn {
            7 * n
        } else {
            3 * n * n + 4 * n
        }
    } else if wntqn {
        7 * m
    } else {
        3 * m * m + 4 * m
    };

    let eps = PREC;
    let smlnum = SAFMIN.sqrt() / eps;
    let bignum = 1.0 / smlnum;
    let anrm = dlange_m(m, n, a, lda);
    if anrm.is_nan() {
        return -4;
    }
    let mut iscl = 0;
    if anrm > 0.0 && anrm < smlnum {
        iscl = 1;
        dlascl_g(anrm, smlnum, m, n, a, lda);
    } else if anrm > bignum {
        iscl = 1;
        dlascl_g(anrm, bignum, m, n, a, lda);
    }

    let mut dum = [0.0f64; 1];
    let dum = dum.as_mut_ptr();
    let mut info = 0;

    if m >= n {
        if m >= mnthr {
            if wntqn {
                // Path 1 (M >> N, JOBZ='N').
                let itau = 1;
                let mut nwork = itau + n;
                dgeqrf(
                    m,
                    n,
                    a,
                    lda,
                    pv!(work, itau),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dlaset(b'L', n - 1, n - 1, 0.0, 0.0, pm!(a, lda, 2, 1), lda);
                let ie = 1;
                let itauq = ie + n;
                let itaup = itauq + n;
                nwork = itaup + n;
                dgebrd(
                    n,
                    n,
                    a,
                    lda,
                    s,
                    pv!(work, ie),
                    pv!(work, itauq),
                    pv!(work, itaup),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                nwork = ie + n;
                info = dbdsdc(
                    b'U',
                    b'N',
                    n,
                    s,
                    pv!(work, ie),
                    dum,
                    1,
                    dum,
                    1,
                    pv!(work, nwork),
                    iwork,
                );
            } else if wntqo {
                // Path 2 (M >> N, JOBZ='O').
                let ir = 1;
                let ldwrkr = if lwork >= lda * n + n * n + 3 * n + bdspac {
                    lda
                } else {
                    (lwork - n * n - 3 * n - bdspac) / n
                };
                let itau = ir + ldwrkr * n;
                let mut nwork = itau + n;
                dgeqrf(
                    m,
                    n,
                    a,
                    lda,
                    pv!(work, itau),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dlacpy(b'U', n, n, a, lda, pv!(work, ir), ldwrkr);
                dlaset(b'L', n - 1, n - 1, 0.0, 0.0, pv!(work, ir + 1), ldwrkr);
                dorgqr(
                    m,
                    n,
                    n,
                    a,
                    lda,
                    pv!(work, itau),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                let ie = itau;
                let itauq = ie + n;
                let itaup = itauq + n;
                nwork = itaup + n;
                dgebrd(
                    n,
                    n,
                    pv!(work, ir),
                    ldwrkr,
                    s,
                    pv!(work, ie),
                    pv!(work, itauq),
                    pv!(work, itaup),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                let iu = nwork;
                nwork = iu + n * n;
                info = dbdsdc(
                    b'U',
                    b'I',
                    n,
                    s,
                    pv!(work, ie),
                    pv!(work, iu),
                    n,
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    iwork,
                );
                dormbr(
                    b'Q',
                    b'N',
                    n,
                    n,
                    n,
                    pv!(work, ir),
                    ldwrkr,
                    pv!(work, itauq),
                    pv!(work, iu),
                    n,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dormbr(
                    b'P',
                    b'T',
                    n,
                    n,
                    n,
                    pv!(work, ir),
                    ldwrkr,
                    pv!(work, itaup),
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                // Multiply Q in A by the left singular vectors of R in
                // WORK(IU), storing the result in WORK(IR) and copying to A.
                let mut i = 1;
                while i <= m {
                    let chunk = (m - i + 1).min(ldwrkr);
                    dgemm(
                        b'N',
                        b'N',
                        chunk,
                        n,
                        n,
                        1.0,
                        pm!(a, lda, i, 1),
                        lda,
                        pv!(work, iu),
                        n,
                        0.0,
                        pv!(work, ir),
                        ldwrkr,
                    );
                    dlacpy(
                        b'F',
                        chunk,
                        n,
                        pv!(work, ir),
                        ldwrkr,
                        pm!(a, lda, i, 1),
                        lda,
                    );
                    i += ldwrkr;
                }
            } else if wntqs {
                // Path 3 (M >> N, JOBZ='S').
                let ir = 1;
                let ldwrkr = n;
                let itau = ir + ldwrkr * n;
                let mut nwork = itau + n;
                dgeqrf(
                    m,
                    n,
                    a,
                    lda,
                    pv!(work, itau),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dlacpy(b'U', n, n, a, lda, pv!(work, ir), ldwrkr);
                dlaset(b'L', n - 1, n - 1, 0.0, 0.0, pv!(work, ir + 1), ldwrkr);
                dorgqr(
                    m,
                    n,
                    n,
                    a,
                    lda,
                    pv!(work, itau),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                let ie = itau;
                let itauq = ie + n;
                let itaup = itauq + n;
                nwork = itaup + n;
                dgebrd(
                    n,
                    n,
                    pv!(work, ir),
                    ldwrkr,
                    s,
                    pv!(work, ie),
                    pv!(work, itauq),
                    pv!(work, itaup),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                info = dbdsdc(
                    b'U',
                    b'I',
                    n,
                    s,
                    pv!(work, ie),
                    u,
                    ldu,
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    iwork,
                );
                dormbr(
                    b'Q',
                    b'N',
                    n,
                    n,
                    n,
                    pv!(work, ir),
                    ldwrkr,
                    pv!(work, itauq),
                    u,
                    ldu,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dormbr(
                    b'P',
                    b'T',
                    n,
                    n,
                    n,
                    pv!(work, ir),
                    ldwrkr,
                    pv!(work, itaup),
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dlacpy(b'F', n, n, u, ldu, pv!(work, ir), ldwrkr);
                dgemm(
                    b'N',
                    b'N',
                    m,
                    n,
                    n,
                    1.0,
                    a,
                    lda,
                    pv!(work, ir),
                    ldwrkr,
                    0.0,
                    u,
                    ldu,
                );
            } else if wntqa {
                // Path 4 (M >> N, JOBZ='A').
                let iu = 1;
                let ldwrku = n;
                let itau = iu + ldwrku * n;
                let mut nwork = itau + n;
                dgeqrf(
                    m,
                    n,
                    a,
                    lda,
                    pv!(work, itau),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dlacpy(b'L', m, n, a, lda, u, ldu);
                dorgqr(
                    m,
                    m,
                    n,
                    u,
                    ldu,
                    pv!(work, itau),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dlaset(b'L', n - 1, n - 1, 0.0, 0.0, pm!(a, lda, 2, 1), lda);
                let ie = itau;
                let itauq = ie + n;
                let itaup = itauq + n;
                nwork = itaup + n;
                dgebrd(
                    n,
                    n,
                    a,
                    lda,
                    s,
                    pv!(work, ie),
                    pv!(work, itauq),
                    pv!(work, itaup),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                info = dbdsdc(
                    b'U',
                    b'I',
                    n,
                    s,
                    pv!(work, ie),
                    pv!(work, iu),
                    n,
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    iwork,
                );
                dormbr(
                    b'Q',
                    b'N',
                    n,
                    n,
                    n,
                    a,
                    lda,
                    pv!(work, itauq),
                    pv!(work, iu),
                    ldwrku,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dormbr(
                    b'P',
                    b'T',
                    n,
                    n,
                    n,
                    a,
                    lda,
                    pv!(work, itaup),
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dgemm(
                    b'N',
                    b'N',
                    m,
                    n,
                    n,
                    1.0,
                    u,
                    ldu,
                    pv!(work, iu),
                    ldwrku,
                    0.0,
                    a,
                    lda,
                );
                dlacpy(b'F', m, n, a, lda, u, ldu);
            }
        } else {
            // Path 5 (M >= N, but not much larger): reduce to bidiagonal
            // form without QR decomposition.
            let ie = 1;
            let itauq = ie + n;
            let itaup = itauq + n;
            let mut nwork = itaup + n;
            dgebrd(
                m,
                n,
                a,
                lda,
                s,
                pv!(work, ie),
                pv!(work, itauq),
                pv!(work, itaup),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            if wntqn {
                info = dbdsdc(
                    b'U',
                    b'N',
                    n,
                    s,
                    pv!(work, ie),
                    dum,
                    1,
                    dum,
                    1,
                    pv!(work, nwork),
                    iwork,
                );
            } else if wntqo {
                let iu = nwork;
                let ldwrku;
                let mut ir = 0;
                let mut ldwrkr = 0;
                if lwork >= m * n + 3 * n + bdspac {
                    // WORK(IU) is M by N.
                    ldwrku = m;
                    nwork = iu + ldwrku * n;
                    dlaset(b'F', m, n, 0.0, 0.0, pv!(work, iu), ldwrku);
                } else {
                    // WORK(IU) is N by N; WORK(IR) holds the chunks.
                    ldwrku = n;
                    nwork = iu + ldwrku * n;
                    ir = nwork;
                    ldwrkr = (lwork - n * n - 3 * n) / n;
                }
                nwork = iu + ldwrku * n;
                info = dbdsdc(
                    b'U',
                    b'I',
                    n,
                    s,
                    pv!(work, ie),
                    pv!(work, iu),
                    ldwrku,
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    iwork,
                );
                dormbr(
                    b'P',
                    b'T',
                    n,
                    n,
                    n,
                    a,
                    lda,
                    pv!(work, itaup),
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                if lwork >= m * n + 3 * n + bdspac {
                    dormbr(
                        b'Q',
                        b'N',
                        m,
                        n,
                        n,
                        a,
                        lda,
                        pv!(work, itauq),
                        pv!(work, iu),
                        ldwrku,
                        pv!(work, nwork),
                        lwork - nwork + 1,
                    );
                    dlacpy(b'F', m, n, pv!(work, iu), ldwrku, a, lda);
                } else {
                    dorgbr(
                        b'Q',
                        m,
                        n,
                        n,
                        a,
                        lda,
                        pv!(work, itauq),
                        pv!(work, nwork),
                        lwork - nwork + 1,
                    );
                    let mut i = 1;
                    while i <= m {
                        let chunk = (m - i + 1).min(ldwrkr);
                        dgemm(
                            b'N',
                            b'N',
                            chunk,
                            n,
                            n,
                            1.0,
                            pm!(a, lda, i, 1),
                            lda,
                            pv!(work, iu),
                            ldwrku,
                            0.0,
                            pv!(work, ir),
                            ldwrkr,
                        );
                        dlacpy(
                            b'F',
                            chunk,
                            n,
                            pv!(work, ir),
                            ldwrkr,
                            pm!(a, lda, i, 1),
                            lda,
                        );
                        i += ldwrkr;
                    }
                }
            } else if wntqs {
                dlaset(b'F', m, n, 0.0, 0.0, u, ldu);
                info = dbdsdc(
                    b'U',
                    b'I',
                    n,
                    s,
                    pv!(work, ie),
                    u,
                    ldu,
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    iwork,
                );
                dormbr(
                    b'Q',
                    b'N',
                    m,
                    n,
                    n,
                    a,
                    lda,
                    pv!(work, itauq),
                    u,
                    ldu,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dormbr(
                    b'P',
                    b'T',
                    n,
                    n,
                    n,
                    a,
                    lda,
                    pv!(work, itaup),
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
            } else if wntqa {
                dlaset(b'F', m, m, 0.0, 0.0, u, ldu);
                info = dbdsdc(
                    b'U',
                    b'I',
                    n,
                    s,
                    pv!(work, ie),
                    u,
                    ldu,
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    iwork,
                );
                if m > n {
                    dlaset(b'F', m - n, m - n, 0.0, 1.0, pm!(u, ldu, n + 1, n + 1), ldu);
                }
                dormbr(
                    b'Q',
                    b'N',
                    m,
                    m,
                    n,
                    a,
                    lda,
                    pv!(work, itauq),
                    u,
                    ldu,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dormbr(
                    b'P',
                    b'T',
                    n,
                    n,
                    m,
                    a,
                    lda,
                    pv!(work, itaup),
                    vt,
                    ldvt,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
            }
        }
    } else if n >= mnthr {
        if wntqn {
            // Path 1t (N >> M, JOBZ='N').
            let itau = 1;
            let mut nwork = itau + m;
            dgelqf(
                m,
                n,
                a,
                lda,
                pv!(work, itau),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dlaset(b'U', m - 1, m - 1, 0.0, 0.0, pm!(a, lda, 1, 2), lda);
            let ie = 1;
            let itauq = ie + m;
            let itaup = itauq + m;
            nwork = itaup + m;
            dgebrd(
                m,
                m,
                a,
                lda,
                s,
                pv!(work, ie),
                pv!(work, itauq),
                pv!(work, itaup),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            nwork = ie + m;
            info = dbdsdc(
                b'U',
                b'N',
                m,
                s,
                pv!(work, ie),
                dum,
                1,
                dum,
                1,
                pv!(work, nwork),
                iwork,
            );
        } else if wntqo {
            // Path 2t (N >> M, JOBZ='O').
            let ivt = 1;
            let il = ivt + m * m;
            let ldwrkl = m;
            let chunk = if lwork >= m * n + m * m + 3 * m + bdspac {
                n
            } else {
                (lwork - m * m) / m
            };
            let itau = il + ldwrkl * m;
            let mut nwork = itau + m;
            dgelqf(
                m,
                n,
                a,
                lda,
                pv!(work, itau),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dlacpy(b'L', m, m, a, lda, pv!(work, il), ldwrkl);
            dlaset(b'U', m - 1, m - 1, 0.0, 0.0, pv!(work, il + ldwrkl), ldwrkl);
            dorglq(
                m,
                n,
                m,
                a,
                lda,
                pv!(work, itau),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            let ie = itau;
            let itauq = ie + m;
            let itaup = itauq + m;
            nwork = itaup + m;
            dgebrd(
                m,
                m,
                pv!(work, il),
                ldwrkl,
                s,
                pv!(work, ie),
                pv!(work, itauq),
                pv!(work, itaup),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            info = dbdsdc(
                b'U',
                b'I',
                m,
                s,
                pv!(work, ie),
                u,
                ldu,
                pv!(work, ivt),
                m,
                pv!(work, nwork),
                iwork,
            );
            dormbr(
                b'Q',
                b'N',
                m,
                m,
                m,
                pv!(work, il),
                ldwrkl,
                pv!(work, itauq),
                u,
                ldu,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dormbr(
                b'P',
                b'T',
                m,
                m,
                m,
                pv!(work, il),
                ldwrkl,
                pv!(work, itaup),
                pv!(work, ivt),
                m,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            // Multiply the right singular vectors of L in WORK(IVT) by Q in
            // A, storing the result in WORK(IL) and copying to A.
            let mut i = 1;
            while i <= n {
                let blk = (n - i + 1).min(chunk);
                dgemm(
                    b'N',
                    b'N',
                    m,
                    blk,
                    m,
                    1.0,
                    pv!(work, ivt),
                    m,
                    pm!(a, lda, 1, i),
                    lda,
                    0.0,
                    pv!(work, il),
                    ldwrkl,
                );
                dlacpy(b'F', m, blk, pv!(work, il), ldwrkl, pm!(a, lda, 1, i), lda);
                i += chunk;
            }
        } else if wntqs {
            // Path 3t (N >> M, JOBZ='S').
            let il = 1;
            let ldwrkl = m;
            let itau = il + ldwrkl * m;
            let mut nwork = itau + m;
            dgelqf(
                m,
                n,
                a,
                lda,
                pv!(work, itau),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dlacpy(b'L', m, m, a, lda, pv!(work, il), ldwrkl);
            dlaset(b'U', m - 1, m - 1, 0.0, 0.0, pv!(work, il + ldwrkl), ldwrkl);
            dorglq(
                m,
                n,
                m,
                a,
                lda,
                pv!(work, itau),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            let ie = itau;
            let itauq = ie + m;
            let itaup = itauq + m;
            nwork = itaup + m;
            dgebrd(
                m,
                m,
                pv!(work, il),
                ldwrkl,
                s,
                pv!(work, ie),
                pv!(work, itauq),
                pv!(work, itaup),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            info = dbdsdc(
                b'U',
                b'I',
                m,
                s,
                pv!(work, ie),
                u,
                ldu,
                vt,
                ldvt,
                pv!(work, nwork),
                iwork,
            );
            dormbr(
                b'Q',
                b'N',
                m,
                m,
                m,
                pv!(work, il),
                ldwrkl,
                pv!(work, itauq),
                u,
                ldu,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dormbr(
                b'P',
                b'T',
                m,
                m,
                m,
                pv!(work, il),
                ldwrkl,
                pv!(work, itaup),
                vt,
                ldvt,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dlacpy(b'F', m, m, vt, ldvt, pv!(work, il), ldwrkl);
            dgemm(
                b'N',
                b'N',
                m,
                n,
                m,
                1.0,
                pv!(work, il),
                ldwrkl,
                a,
                lda,
                0.0,
                vt,
                ldvt,
            );
        } else if wntqa {
            // Path 4t (N >> M, JOBZ='A').
            let ivt = 1;
            let ldwkvt = m;
            let itau = ivt + ldwkvt * m;
            let mut nwork = itau + m;
            dgelqf(
                m,
                n,
                a,
                lda,
                pv!(work, itau),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dlacpy(b'U', m, n, a, lda, vt, ldvt);
            dorglq(
                n,
                n,
                m,
                vt,
                ldvt,
                pv!(work, itau),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dlaset(b'U', m - 1, m - 1, 0.0, 0.0, pm!(a, lda, 1, 2), lda);
            let ie = itau;
            let itauq = ie + m;
            let itaup = itauq + m;
            nwork = itaup + m;
            dgebrd(
                m,
                m,
                a,
                lda,
                s,
                pv!(work, ie),
                pv!(work, itauq),
                pv!(work, itaup),
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            info = dbdsdc(
                b'U',
                b'I',
                m,
                s,
                pv!(work, ie),
                u,
                ldu,
                pv!(work, ivt),
                ldwkvt,
                pv!(work, nwork),
                iwork,
            );
            dormbr(
                b'Q',
                b'N',
                m,
                m,
                m,
                a,
                lda,
                pv!(work, itauq),
                u,
                ldu,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dormbr(
                b'P',
                b'T',
                m,
                m,
                m,
                a,
                lda,
                pv!(work, itaup),
                pv!(work, ivt),
                ldwkvt,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dgemm(
                b'N',
                b'N',
                m,
                n,
                m,
                1.0,
                pv!(work, ivt),
                ldwkvt,
                vt,
                ldvt,
                0.0,
                a,
                lda,
            );
            dlacpy(b'F', m, n, a, lda, vt, ldvt);
        }
    } else {
        // Path 5t (N > M, but not much larger): reduce to bidiagonal form
        // without LQ decomposition.
        let ie = 1;
        let itauq = ie + m;
        let itaup = itauq + m;
        let mut nwork = itaup + m;
        dgebrd(
            m,
            n,
            a,
            lda,
            s,
            pv!(work, ie),
            pv!(work, itauq),
            pv!(work, itaup),
            pv!(work, nwork),
            lwork - nwork + 1,
        );
        if wntqn {
            info = dbdsdc(
                b'L',
                b'N',
                m,
                s,
                pv!(work, ie),
                dum,
                1,
                dum,
                1,
                pv!(work, nwork),
                iwork,
            );
        } else if wntqo {
            let ldwkvt = m;
            let ivt = nwork;
            let mut il = 0;
            let mut chunk = 0;
            if lwork >= m * n + 3 * m + bdspac {
                // WORK(IVT) is M by N.
                dlaset(b'F', m, n, 0.0, 0.0, pv!(work, ivt), ldwkvt);
                nwork = ivt + ldwkvt * n;
            } else {
                // WORK(IVT) is M by M; WORK(IL) holds the chunks.
                nwork = ivt + ldwkvt * m;
                il = nwork;
                chunk = (lwork - m * m - 3 * m) / m;
            }
            info = dbdsdc(
                b'L',
                b'I',
                m,
                s,
                pv!(work, ie),
                u,
                ldu,
                pv!(work, ivt),
                ldwkvt,
                pv!(work, nwork),
                iwork,
            );
            dormbr(
                b'Q',
                b'N',
                m,
                m,
                n,
                a,
                lda,
                pv!(work, itauq),
                u,
                ldu,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            if lwork >= m * n + 3 * m + bdspac {
                dormbr(
                    b'P',
                    b'T',
                    m,
                    n,
                    m,
                    a,
                    lda,
                    pv!(work, itaup),
                    pv!(work, ivt),
                    ldwkvt,
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                dlacpy(b'F', m, n, pv!(work, ivt), ldwkvt, a, lda);
            } else {
                dorgbr(
                    b'P',
                    m,
                    n,
                    m,
                    a,
                    lda,
                    pv!(work, itaup),
                    pv!(work, nwork),
                    lwork - nwork + 1,
                );
                let mut i = 1;
                while i <= n {
                    let blk = (n - i + 1).min(chunk);
                    dgemm(
                        b'N',
                        b'N',
                        m,
                        blk,
                        m,
                        1.0,
                        pv!(work, ivt),
                        ldwkvt,
                        pm!(a, lda, 1, i),
                        lda,
                        0.0,
                        pv!(work, il),
                        m,
                    );
                    dlacpy(b'F', m, blk, pv!(work, il), m, pm!(a, lda, 1, i), lda);
                    i += chunk;
                }
            }
        } else if wntqs {
            dlaset(b'F', m, n, 0.0, 0.0, vt, ldvt);
            info = dbdsdc(
                b'L',
                b'I',
                m,
                s,
                pv!(work, ie),
                u,
                ldu,
                vt,
                ldvt,
                pv!(work, nwork),
                iwork,
            );
            dormbr(
                b'Q',
                b'N',
                m,
                m,
                n,
                a,
                lda,
                pv!(work, itauq),
                u,
                ldu,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dormbr(
                b'P',
                b'T',
                m,
                n,
                m,
                a,
                lda,
                pv!(work, itaup),
                vt,
                ldvt,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
        } else if wntqa {
            dlaset(b'F', n, n, 0.0, 0.0, vt, ldvt);
            info = dbdsdc(
                b'L',
                b'I',
                m,
                s,
                pv!(work, ie),
                u,
                ldu,
                vt,
                ldvt,
                pv!(work, nwork),
                iwork,
            );
            if n > m {
                dlaset(
                    b'F',
                    n - m,
                    n - m,
                    0.0,
                    1.0,
                    pm!(vt, ldvt, m + 1, m + 1),
                    ldvt,
                );
            }
            dormbr(
                b'Q',
                b'N',
                m,
                m,
                n,
                a,
                lda,
                pv!(work, itauq),
                u,
                ldu,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
            dormbr(
                b'P',
                b'T',
                n,
                n,
                m,
                a,
                lda,
                pv!(work, itaup),
                vt,
                ldvt,
                pv!(work, nwork),
                lwork - nwork + 1,
            );
        }
    }

    // Undo scaling if necessary.
    if iscl == 1 {
        if anrm > bignum {
            dlascl_g(bignum, anrm, minmn, 1, s, minmn);
        }
        if anrm < smlnum {
            dlascl_g(smlnum, anrm, minmn, 1, s, minmn);
        }
    }
    info
}
