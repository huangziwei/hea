//! `DSYTRD`: reduction of a symmetric matrix to tridiagonal form by
//! orthogonal similarity (`DSYTD2`, blocked by `DLATRD` + `DSYR2K`), and the
//! `DLARFG` reflectors it is built from. Lower-triangle storage, the
//! triangle `eigen(symmetric = TRUE)` reads.

use super::aux::dlapy2;
use super::blas::{daxpy, ddot, dgemv, dnrm2, dscal, dsymv, dsyr2, dsyr2k};
use super::{disnan, sign, EPS, SAFMIN};

/// `DLANSY('M', 'L', ...)`: the largest absolute entry of the lower
/// triangle (a NaN entry is returned as the norm).
pub(crate) unsafe fn dlansy_m_lower(n: i32, a: *const f64, lda: i32) -> f64 {
    let mut value = 0.0f64;
    for j in 1..=n {
        for i in j..=n {
            let sum = m!(a, lda, i, j).abs();
            if value < sum || disnan(sum) {
                value = sum;
            }
        }
    }
    value
}

/// `DLARFG`: the elementary reflector `H = I - tau v v'` with
/// `H [alpha; x] = [beta; 0]`, `v = [1; x]` overwriting `x` and `beta`
/// overwriting `alpha`. Returns `tau`.
pub(crate) unsafe fn dlarfg(n: i32, alpha: *mut f64, x: *mut f64, incx: i32) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut xnorm = dnrm2(n - 1, x, incx);
    if xnorm == 0.0 {
        // H = I
        return 0.0;
    }
    // general case
    let mut beta = -sign(dlapy2(*alpha, xnorm), *alpha);
    let safmin = SAFMIN / EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        // XNORM, BETA may be inaccurate; scale X and recompute them
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            dscal(n - 1, rsafmn, x, incx);
            beta *= rsafmn;
            *alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        // New BETA is at most 1, at least SAFMIN
        xnorm = dnrm2(n - 1, x, incx);
        beta = -sign(dlapy2(*alpha, xnorm), *alpha);
    }
    let tau = (beta - *alpha) / beta;
    dscal(n - 1, 1.0 / (*alpha - beta), x, incx);

    // If ALPHA is subnormal, it may lose relative accuracy
    for _ in 1..=knt {
        beta *= safmin;
    }
    *alpha = beta;
    tau
}

/// `DSYTD2` with `UPLO = 'L'`: unblocked reduction of the lower triangle
/// of `a` to tridiagonal `(d, e)`, reflectors below the subdiagonal.
pub(crate) unsafe fn dsytd2_lower(
    n: i32,
    a: *mut f64,
    lda: i32,
    d: *mut f64,
    e: *mut f64,
    tau: *mut f64,
) {
    if n <= 0 {
        return;
    }
    // Reduce the lower triangle of A
    for i in 1..=(n - 1) {
        // Generate elementary reflector H(i) = I - tau * v * v**T to
        // annihilate A(i+2:n,i)
        let taui = dlarfg(
            n - i,
            pm!(a, lda, i + 1, i),
            pm!(a, lda, (i + 2).min(n), i),
            1,
        );
        v!(e, i) = m!(a, lda, i + 1, i);

        if taui != 0.0 {
            // Apply H(i) from both sides to A(i+1:n,i+1:n)
            m!(a, lda, i + 1, i) = 1.0;

            // Compute  x := tau * A * v  storing y in TAU(i:n-1)
            dsymv(
                b'L',
                n - i,
                taui,
                pm!(a, lda, i + 1, i + 1),
                lda,
                pm!(a, lda, i + 1, i),
                1,
                0.0,
                pv!(tau, i),
                1,
            );

            // Compute  w := x - 1/2 * tau * (x**T * v) * v
            let alpha = -0.5 * taui * ddot(n - i, pv!(tau, i), 1, pm!(a, lda, i + 1, i), 1);
            daxpy(n - i, alpha, pm!(a, lda, i + 1, i), 1, pv!(tau, i), 1);

            // Apply the transformation as a rank-2 update:
            //    A := A - v * w**T - w * v**T
            dsyr2(
                b'L',
                n - i,
                -1.0,
                pm!(a, lda, i + 1, i),
                1,
                pv!(tau, i),
                1,
                pm!(a, lda, i + 1, i + 1),
                lda,
            );

            m!(a, lda, i + 1, i) = v!(e, i);
        }
        v!(d, i) = m!(a, lda, i, i);
        v!(tau, i) = taui;
    }
    v!(d, n) = m!(a, lda, n, n);
}

/// `DLATRD` with `UPLO = 'L'`: reduces the first `nb` columns of the lower
/// triangle and returns in `w` the matrix `W` of the blocked update
/// `A := A - V W' - W V'`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlatrd_lower(
    n: i32,
    nb: i32,
    a: *mut f64,
    lda: i32,
    e: *mut f64,
    tau: *mut f64,
    w: *mut f64,
    ldw: i32,
) {
    if n <= 0 {
        return;
    }
    // Reduce first NB columns of lower triangle
    for i in 1..=nb {
        // Update A(i:n,i)
        dgemv(
            b'N',
            n - i + 1,
            i - 1,
            -1.0,
            pm!(a, lda, i, 1),
            lda,
            pm!(w, ldw, i, 1),
            ldw,
            1.0,
            pm!(a, lda, i, i),
            1,
        );
        dgemv(
            b'N',
            n - i + 1,
            i - 1,
            -1.0,
            pm!(w, ldw, i, 1),
            ldw,
            pm!(a, lda, i, 1),
            lda,
            1.0,
            pm!(a, lda, i, i),
            1,
        );
        if i < n {
            // Generate elementary reflector H(i) to annihilate A(i+2:n,i)
            v!(tau, i) = dlarfg(
                n - i,
                pm!(a, lda, i + 1, i),
                pm!(a, lda, (i + 2).min(n), i),
                1,
            );
            v!(e, i) = m!(a, lda, i + 1, i);
            m!(a, lda, i + 1, i) = 1.0;

            // Compute W(i+1:n,i)
            dsymv(
                b'L',
                n - i,
                1.0,
                pm!(a, lda, i + 1, i + 1),
                lda,
                pm!(a, lda, i + 1, i),
                1,
                0.0,
                pm!(w, ldw, i + 1, i),
                1,
            );
            dgemv(
                b'T',
                n - i,
                i - 1,
                1.0,
                pm!(w, ldw, i + 1, 1),
                ldw,
                pm!(a, lda, i + 1, i),
                1,
                0.0,
                pm!(w, ldw, 1, i),
                1,
            );
            dgemv(
                b'N',
                n - i,
                i - 1,
                -1.0,
                pm!(a, lda, i + 1, 1),
                lda,
                pm!(w, ldw, 1, i),
                1,
                1.0,
                pm!(w, ldw, i + 1, i),
                1,
            );
            dgemv(
                b'T',
                n - i,
                i - 1,
                1.0,
                pm!(a, lda, i + 1, 1),
                lda,
                pm!(a, lda, i + 1, i),
                1,
                0.0,
                pm!(w, ldw, 1, i),
                1,
            );
            dgemv(
                b'N',
                n - i,
                i - 1,
                -1.0,
                pm!(w, ldw, i + 1, 1),
                ldw,
                pm!(w, ldw, 1, i),
                1,
                1.0,
                pm!(w, ldw, i + 1, i),
                1,
            );
            dscal(n - i, v!(tau, i), pm!(w, ldw, i + 1, i), 1);
            let alpha =
                -0.5 * v!(tau, i) * ddot(n - i, pm!(w, ldw, i + 1, i), 1, pm!(a, lda, i + 1, i), 1);
            daxpy(
                n - i,
                alpha,
                pm!(a, lda, i + 1, i),
                1,
                pm!(w, ldw, i + 1, i),
                1,
            );
        }
    }
}

/// `ILAENV(1, 'DSYTRD', ...)`, `ILAENV(2, ...)` and `ILAENV(3, ...)`.
const SYTRD_NB: i32 = 32;
const SYTRD_NBMIN: i32 = 2;
const SYTRD_NX: i32 = 32;

/// `DSYTRD` with `UPLO = 'L'` and `lwork` doubles of workspace in `work`:
/// blocked by `DLATRD` panels of `NB` columns (`NB` cut to `lwork / n` when
/// the workspace is short, as LAPACK does), the last block by `DSYTD2`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dsytrd_lower(
    n: i32,
    a: *mut f64,
    lda: i32,
    d: *mut f64,
    e: *mut f64,
    tau: *mut f64,
    work: *mut f64,
    lwork: i32,
) {
    if n == 0 {
        return;
    }
    let mut nb = SYTRD_NB;
    let mut nx = n;
    let mut ldwork = 1;
    if nb > 1 && nb < n {
        // Determine when to cross over from blocked to unblocked code
        // (last block is always handled by unblocked code).
        nx = nb.max(SYTRD_NX);
        if nx < n {
            // Determine if workspace is large enough for blocked code.
            ldwork = n;
            let iws = ldwork * nb;
            if lwork < iws {
                // Not enough workspace to use optimal NB: determine the
                // minimum value of NB, and reduce NB or force use of
                // unblocked code by setting NX = N.
                nb = (lwork / ldwork).max(1);
                if nb < SYTRD_NBMIN {
                    nx = n;
                }
            }
        } else {
            nx = n;
        }
    } else {
        nb = 1;
    }

    // Reduce the lower triangle of A
    let mut i = 1;
    while i <= n - nx {
        // Reduce columns i:i+nb-1 to tridiagonal form and form the matrix
        // W which is needed to update the unreduced part of the matrix
        dlatrd_lower(
            n - i + 1,
            nb,
            pm!(a, lda, i, i),
            lda,
            pv!(e, i),
            pv!(tau, i),
            work,
            ldwork,
        );

        // Update the unreduced submatrix A(i+nb:n,i+nb:n), using an update
        // of the form:  A := A - V*W**T - W*V**T
        dsyr2k(
            b'L',
            b'N',
            n - i - nb + 1,
            nb,
            -1.0,
            pm!(a, lda, i + nb, i),
            lda,
            pv!(work, nb + 1),
            ldwork,
            1.0,
            pm!(a, lda, i + nb, i + nb),
            lda,
        );

        // Copy subdiagonal elements back into A, and diagonal elements
        // into D
        for j in i..=(i + nb - 1) {
            m!(a, lda, j + 1, j) = v!(e, j);
            v!(d, j) = m!(a, lda, j, j);
        }
        i += nb;
    }

    // Use unblocked code to reduce the last or only block
    dsytd2_lower(
        n - i + 1,
        pm!(a, lda, i, i),
        lda,
        pv!(d, i),
        pv!(e, i),
        pv!(tau, i),
    );
}
