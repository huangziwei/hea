//! `DSYEVR` for `RANGE = 'A'`, `UPLO = 'L'`: all eigenvalues and,
//! optionally, eigenvectors of a symmetric matrix, as R's `La_rs` calls it
//! for `eigen(symmetric = TRUE)`: `DSYTRD`, then `DSTEMR` (`DSTERF` without
//! vectors) and `DORMTR`.

use super::blas::{dcopy, dscal, dswap};
use super::ormtr::dormtr_lln;
use super::stemr::dstemr_all;
use super::steqr::dsterf;
use super::sytrd::{dlansy_m_lower, dsytrd_lower};
use super::{PREC, SAFMIN};

/// `ILAENV(1, 'DSYTRD', ...)` and `ILAENV(1, 'DORMTR', ...)`.
const NB: i32 = 32;

/// `(LWORK, LIWORK)` that `DSYEVR`'s workspace query returns, the sizes R
/// allocates: `max((NB + 1) n, 26 n)` and `10 n`.
pub(crate) fn dsyevr_lwork(n: i32) -> (usize, usize) {
    let lwmin = (26 * n).max(1);
    let liwmin = (10 * n).max(1);
    (((NB + 1) * n).max(lwmin) as usize, liwmin as usize)
}

/// `DSYEVR('V' or 'N', 'A', 'L', ...)` with `abstol = 0`: eigenvalues
/// ascending into `w` and, when `wantz`, the eigenvectors into `z`. `a` is
/// overwritten. `work`/`iwork` hold [`dsyevr_lwork`] entries. Returns
/// `(m, INFO)`; a nonzero `INFO` from `DSTEMR` is returned as such (LAPACK
/// then falls back to `DSTEBZ`/`DSTEIN`).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dsyevr_all_lower(
    wantz: bool,
    n: i32,
    a: *mut f64,
    lda: i32,
    w: *mut f64,
    z: *mut f64,
    ldz: i32,
    isuppz: *mut i32,
    work: *mut f64,
    lwork: i32,
    iwork: *mut i32,
) -> (i32, i32) {
    let mut m = 0;
    if n == 0 {
        return (0, 0);
    }
    if n == 1 {
        m = 1;
        v!(w, 1) = m!(a, lda, 1, 1);
        if wantz {
            m!(z, ldz, 1, 1) = 1.0;
            v!(isuppz, 1) = 1;
            v!(isuppz, 2) = 1;
        }
        return (m, 0);
    }

    // Get machine constants.
    let safmin = SAFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt().min(1.0 / safmin.sqrt().sqrt());

    // Scale matrix to allowable range, if necessary.
    let mut iscale = 0;
    let mut sigma = 0.0;
    let anrm = dlansy_m_lower(n, a, lda);
    if anrm > 0.0 && anrm < rmin {
        iscale = 1;
        sigma = rmin / anrm;
    } else if anrm > rmax {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if iscale == 1 {
        for j in 1..=n {
            dscal(n - j + 1, sigma, pm!(a, lda, j, j), 1);
        }
    }

    // Initialize indices into workspaces.
    let indtau = 1;
    let indd = indtau + n;
    let inde = indd + n;
    let inddd = inde + n;
    let indee = inddd + n;
    let indwk = indee + n;
    let llwork = lwork - indwk + 1;

    // Call DSYTRD to reduce symmetric matrix to tridiagonal form.
    dsytrd_lower(
        n,
        a,
        lda,
        pv!(work, indd),
        pv!(work, inde),
        pv!(work, indtau),
        pv!(work, indwk),
        llwork,
    );

    // If all eigenvalues are desired then call DSTERF or DSTEMR and DORMTR.
    let info;
    if !wantz {
        dcopy(n, pv!(work, indd), 1, w, 1);
        dcopy(n - 1, pv!(work, inde), 1, pv!(work, indee), 1);
        info = dsterf(n, w, pv!(work, indee));
    } else {
        dcopy(n - 1, pv!(work, inde), 1, pv!(work, indee), 1);
        dcopy(n, pv!(work, indd), 1, pv!(work, inddd), 1);

        let mut tryrac = 0.0 <= 2.0 * n as f64 * eps;
        let (mm, inf) = dstemr_all(
            true,
            n,
            pv!(work, inddd),
            pv!(work, indee),
            w,
            z,
            ldz,
            isuppz,
            &mut tryrac,
            pv!(work, indwk),
            iwork,
        );
        m = mm;
        info = inf;

        // Apply orthogonal matrix used in reduction to tridiagonal form.
        if info == 0 {
            let indwkn = inde;
            let llwrkn = lwork - indwkn + 1;
            dormtr_lln(
                n,
                m,
                a,
                lda,
                pv!(work, indtau),
                z,
                ldz,
                pv!(work, indwkn),
                llwrkn,
            );
        }
    }
    if info != 0 {
        return (m, info);
    }
    // Everything worked.
    m = n;

    // If matrix was scaled, then rescale eigenvalues appropriately.
    if iscale == 1 {
        dscal(m, 1.0 / sigma, w, 1);
    }

    // If eigenvalues are not in order, then sort them, along with
    // eigenvectors. This is only done if the eigenvectors are wanted.
    if wantz {
        for j in 1..=(m - 1) {
            let mut i = 0;
            let mut tmp1 = v!(w, j);
            for jj in (j + 1)..=m {
                if v!(w, jj) < tmp1 {
                    i = jj;
                    tmp1 = v!(w, jj);
                }
            }
            if i != 0 {
                v!(w, i) = v!(w, j);
                v!(w, j) = tmp1;
                dswap(n, pm!(z, ldz, 1, i), 1, pm!(z, ldz, 1, j), 1);
            }
        }
    }
    (m, 0)
}
