//! `DLARRV`: eigenvectors of the representations `DLARRE` built, by twisted
//! factorizations (`DLAR1V`), with new representations for clusters
//! (`DLARRF`) and bisection (`DLARRB`, `DLANEG`) where Rayleigh quotient
//! correction does not converge.

use super::aux::dlaset;
use super::blas::{dcopy, dscal};
use super::{disnan, fma, PREC, SAFMIN};

/// `DLANEG`: the Sturm count of `L D L' - sigma I` (twisted at `r`).
unsafe fn dlaneg(n: i32, d: *const f64, lld: *const f64, sigma: f64, _pivmin: f64, r: i32) -> i32 {
    const BLKLEN: i32 = 128;
    let mut negcnt = 0;

    // I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
    let mut t = -sigma;
    let mut bj = 1;
    while bj <= r - 1 {
        let mut neg1 = 0;
        let bsav = t;
        for j in bj..=(bj + BLKLEN - 1).min(r - 1) {
            let dplus = v!(d, j) + t;
            if dplus < 0.0 {
                neg1 += 1;
            }
            let tmp = t / dplus;
            t = fma(tmp, v!(lld, j), -sigma);
        }
        let sawnan = disnan(t);
        // Run a slower version of the above loop if a NaN is detected.
        // A NaN should occur only with a zero pivot after an infinite
        // pivot. In that case, substituting 1 for T/DPLUS is the
        // correct limit.
        if sawnan {
            neg1 = 0;
            t = bsav;
            for j in bj..=(bj + BLKLEN - 1).min(r - 1) {
                let dplus = v!(d, j) + t;
                if dplus < 0.0 {
                    neg1 += 1;
                }
                let mut tmp = t / dplus;
                if disnan(tmp) {
                    tmp = 1.0;
                }
                t = fma(tmp, v!(lld, j), -sigma);
            }
        }
        negcnt += neg1;
        bj += BLKLEN;
    }

    // II) lower part: L D L^T - SIGMA I = U- D- U-^T
    let mut p = v!(d, n) - sigma;
    let mut bj = n - 1;
    while bj >= r {
        let mut neg2 = 0;
        let bsav = p;
        let mut j = bj;
        while j >= (bj - BLKLEN + 1).max(r) {
            let dminus = v!(lld, j) + p;
            if dminus < 0.0 {
                neg2 += 1;
            }
            let tmp = p / dminus;
            p = fma(tmp, v!(d, j), -sigma);
            j -= 1;
        }
        let sawnan = disnan(p);
        // As above, run a slower version that substitutes 1 for Inf/Inf.
        if sawnan {
            neg2 = 0;
            p = bsav;
            let mut j = bj;
            while j >= (bj - BLKLEN + 1).max(r) {
                let dminus = v!(lld, j) + p;
                if dminus < 0.0 {
                    neg2 += 1;
                }
                let mut tmp = p / dminus;
                if disnan(tmp) {
                    tmp = 1.0;
                }
                p = fma(tmp, v!(d, j), -sigma);
                j -= 1;
            }
        }
        negcnt += neg2;
        bj -= BLKLEN;
    }

    // III) Twist index
    //   T was shifted by SIGMA initially.
    let gamma = (t + sigma) + p;
    if gamma < 0.0 {
        negcnt += 1;
    }
    negcnt
}

/// `DLARRB`: bisection refinement of the eigenvalues `ifirst..=ilast` of
/// `L D L'` (`lld` the products `l(i) d(i) l(i)`), with the gaps.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarrb(
    n: i32,
    d: *const f64,
    lld: *const f64,
    ifirst: i32,
    ilast: i32,
    rtol1: f64,
    rtol2: f64,
    offset: i32,
    w: *mut f64,
    wgap: *mut f64,
    werr: *mut f64,
    work: *mut f64,
    iwork: *mut i32,
    pivmin: f64,
    spdiam: f64,
    twist: i32,
) -> i32 {
    if n <= 0 {
        return 0;
    }
    let maxitr = (((spdiam + pivmin).ln() - pivmin.ln()) / 2.0f64.ln()) as i32 + 2;
    let mnwdth = 2.0 * pivmin;

    let mut r = twist;
    if r < 1 || r > n {
        r = n;
    }

    // Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
    // The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
    // Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer
    // IWORK( 2*I-1 ) for an unconverged interval is set to the index of
    // the next unconverged interval, and is -1 or 0 for a converged
    // interval. Thus a linked list of unconverged intervals is set up.
    let mut i1 = ifirst;
    // The number of unconverged intervals
    let mut nint = 0;
    // The last unconverged interval found
    let mut prev = 0;

    let mut rgap = v!(wgap, i1 - offset);
    for i in i1..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        let mut left = v!(w, ii) - v!(werr, ii);
        let mut right = v!(w, ii) + v!(werr, ii);
        let lgap = rgap;
        rgap = v!(wgap, ii);
        let gap = lgap.min(rgap);

        // Make sure that [LEFT,RIGHT] contains the desired eigenvalue
        // Compute negcount from dstqds facto L+D+L+^T = L D L^T - LEFT
        //
        // Do while( NEGCNT(LEFT).GT.I-1 )
        let mut back = v!(werr, ii);
        loop {
            let negcnt = dlaneg(n, d, lld, left, pivmin, r);
            if negcnt > i - 1 {
                left -= back;
                back *= 2.0;
            } else {
                break;
            }
        }

        // Do while( NEGCNT(RIGHT).LT.I )
        // Compute negcount from dstqds facto L+D+L+^T = L D L^T - RIGHT
        back = v!(werr, ii);
        let negcnt;
        loop {
            let nc = dlaneg(n, d, lld, right, pivmin, r);
            if nc < i {
                right += back;
                back *= 2.0;
            } else {
                negcnt = nc;
                break;
            }
        }
        let width = 0.5 * (left - right).abs();
        let tmp = left.abs().max(right.abs());
        let cvrgd = (rtol1 * gap).max(rtol2 * tmp);
        if width <= cvrgd || width <= mnwdth {
            // This interval has already converged and does not need
            // refinement. (Note that the gaps might change through
            // refining the eigenvalues, however, they can only get
            // bigger.) Remove it from the list.
            v!(iwork, k - 1) = -1;
            // Make sure that I1 always points to the first unconverged
            // interval
            if i == i1 && i < ilast {
                i1 = i + 1;
            }
            if prev >= i1 && i <= ilast {
                v!(iwork, 2 * prev - 1) = i + 1;
            }
        } else {
            // unconverged interval found
            prev = i;
            nint += 1;
            v!(iwork, k - 1) = i + 1;
            v!(iwork, k) = negcnt;
        }
        v!(work, k - 1) = left;
        v!(work, k) = right;
    }

    // Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
    // and while (ITER.LT.MAXITR)
    let mut iter = 0;
    loop {
        let mut prev = i1 - 1;
        let mut i = i1;
        let olnint = nint;

        for _ip in 1..=olnint {
            let k = 2 * i;
            let ii = i - offset;
            let rgap = v!(wgap, ii);
            let mut lgap = rgap;
            if ii > 1 {
                lgap = v!(wgap, ii - 1);
            }
            let gap = lgap.min(rgap);
            let next = v!(iwork, k - 1);
            let left = v!(work, k - 1);
            let right = v!(work, k);
            let mid = 0.5 * (left + right);

            // semiwidth of interval
            let width = right - mid;
            let tmp = left.abs().max(right.abs());
            let cvrgd = (rtol1 * gap).max(rtol2 * tmp);
            if width <= cvrgd || width <= mnwdth || iter == maxitr {
                // reduce number of unconverged intervals
                nint -= 1;
                // Mark interval as converged.
                v!(iwork, k - 1) = 0;
                if i1 == i {
                    i1 = next;
                } else {
                    // Prev holds the last unconverged interval previously
                    // examined
                    if prev >= i1 {
                        v!(iwork, 2 * prev - 1) = next;
                    }
                }
                i = next;
                continue;
            }
            prev = i;

            // Perform one bisection step
            let negcnt = dlaneg(n, d, lld, mid, pivmin, r);
            if negcnt <= i - 1 {
                v!(work, k - 1) = mid;
            } else {
                v!(work, k) = mid;
            }
            i = next;
        }
        iter += 1;
        // do another loop if there are still unconverged intervals
        // However, in the last iteration, all intervals are accepted
        // since this is the best we can do.
        if !(nint > 0 && iter <= maxitr) {
            break;
        }
    }

    // At this point, all the intervals have converged
    for i in ifirst..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        // All intervals marked by '0' have been refined.
        if v!(iwork, k - 1) == 0 {
            v!(w, ii) = 0.5 * (v!(work, k - 1) + v!(work, k));
            v!(werr, ii) = v!(work, k) - v!(w, ii);
        }
    }

    for i in (ifirst + 1)..=ilast {
        let ii = i - offset;
        v!(wgap, ii - 1) = 0.0f64.max(v!(w, ii) - v!(werr, ii) - v!(w, ii - 1) - v!(werr, ii - 1));
    }
    0
}

/// Outputs of [`dlar1v`].
struct Lr1v {
    negcnt: i32,
    r: i32,
    nrminv: f64,
    resid: f64,
    rqcorr: f64,
}

/// `DLAR1V`: the eigenvector of `L D L' - lambda I` for an eigenvalue of
/// it, from the twisted factorization at the index minimizing `|gamma|`
/// (`r` on entry: 0 to search `b1..=bn`, else that index).
#[allow(clippy::too_many_arguments)]
unsafe fn dlar1v(
    n: i32,
    b1: i32,
    bn: i32,
    lambda: f64,
    d: *const f64,
    l: *const f64,
    ld: *const f64,
    lld: *const f64,
    pivmin: f64,
    gaptol: f64,
    z: *mut f64,
    wantnc: bool,
    r_in: i32,
    isuppz: *mut i32,
    work: *mut f64,
) -> Lr1v {
    let eps = PREC;

    let (r1, r2) = if r_in == 0 { (b1, bn) } else { (r_in, r_in) };

    // Storage for LPLUS
    let indlpl = 0;
    // Storage for UMINUS
    let indumn = n;
    let inds = 2 * n + 1;
    let indp = 3 * n + 1;

    if b1 == 1 {
        v!(work, inds) = 0.0;
    } else {
        v!(work, inds + b1 - 1) = v!(lld, b1 - 1);
    }

    // Compute the stationary transform (using the differential form) until
    // the index R2.
    let mut neg1 = 0;
    let mut s = v!(work, inds + b1 - 1) - lambda;
    for i in b1..=(r1 - 1) {
        let dplus = v!(d, i) + s;
        v!(work, indlpl + i) = v!(ld, i) / dplus;
        if dplus < 0.0 {
            neg1 += 1;
        }
        v!(work, inds + i) = s * v!(work, indlpl + i) * v!(l, i);
        s = v!(work, inds + i) - lambda;
    }
    let mut sawnan1 = disnan(s);
    if !sawnan1 {
        for i in r1..=(r2 - 1) {
            let dplus = v!(d, i) + s;
            v!(work, indlpl + i) = v!(ld, i) / dplus;
            v!(work, inds + i) = s * v!(work, indlpl + i) * v!(l, i);
            s = v!(work, inds + i) - lambda;
        }
        sawnan1 = disnan(s);
    }

    if sawnan1 {
        // Runs a slower version of the above loop if a NaN is detected
        neg1 = 0;
        s = v!(work, inds + b1 - 1) - lambda;
        for i in b1..=(r1 - 1) {
            let mut dplus = v!(d, i) + s;
            if dplus.abs() < pivmin {
                dplus = -pivmin;
            }
            v!(work, indlpl + i) = v!(ld, i) / dplus;
            if dplus < 0.0 {
                neg1 += 1;
            }
            v!(work, inds + i) = s * v!(work, indlpl + i) * v!(l, i);
            if v!(work, indlpl + i) == 0.0 {
                v!(work, inds + i) = v!(lld, i);
            }
            s = v!(work, inds + i) - lambda;
        }
        for i in r1..=(r2 - 1) {
            let mut dplus = v!(d, i) + s;
            if dplus.abs() < pivmin {
                dplus = -pivmin;
            }
            v!(work, indlpl + i) = v!(ld, i) / dplus;
            v!(work, inds + i) = s * v!(work, indlpl + i) * v!(l, i);
            if v!(work, indlpl + i) == 0.0 {
                v!(work, inds + i) = v!(lld, i);
            }
            s = v!(work, inds + i) - lambda;
        }
    }

    // Compute the progressive transform (using the differential form) until
    // the index R1
    let mut neg2 = 0;
    v!(work, indp + bn - 1) = v!(d, bn) - lambda;
    let mut i = bn - 1;
    while i >= r1 {
        let dminus = v!(lld, i) + v!(work, indp + i);
        let tmp = v!(d, i) / dminus;
        if dminus < 0.0 {
            neg2 += 1;
        }
        v!(work, indumn + i) = v!(l, i) * tmp;
        v!(work, indp + i - 1) = fma(v!(work, indp + i), tmp, -lambda);
        i -= 1;
    }
    let tmp = v!(work, indp + r1 - 1);
    let sawnan2 = disnan(tmp);

    if sawnan2 {
        // Runs a slower version of the above loop if a NaN is detected
        neg2 = 0;
        let mut i = bn - 1;
        while i >= r1 {
            let mut dminus = v!(lld, i) + v!(work, indp + i);
            if dminus.abs() < pivmin {
                dminus = -pivmin;
            }
            let tmp = v!(d, i) / dminus;
            if dminus < 0.0 {
                neg2 += 1;
            }
            v!(work, indumn + i) = v!(l, i) * tmp;
            v!(work, indp + i - 1) = fma(v!(work, indp + i), tmp, -lambda);
            if tmp == 0.0 {
                v!(work, indp + i - 1) = v!(d, i) - lambda;
            }
            i -= 1;
        }
    }

    // Find the index (from R1 to R2) of the largest (in magnitude) diagonal
    // element of the inverse
    let mut mingma = v!(work, inds + r1 - 1) + v!(work, indp + r1 - 1);
    if mingma < 0.0 {
        neg1 += 1;
    }
    let negcnt = if wantnc { neg1 + neg2 } else { -1 };
    if mingma.abs() == 0.0 {
        mingma = eps * v!(work, inds + r1 - 1);
    }
    let mut r = r1;
    for i in r1..=(r2 - 1) {
        let mut tmp = v!(work, inds + i) + v!(work, indp + i);
        if tmp == 0.0 {
            tmp = eps * v!(work, inds + i);
        }
        if tmp.abs() <= mingma.abs() {
            mingma = tmp;
            r = i + 1;
        }
    }

    // Compute the FP vector: solve N^T v = e_r
    v!(isuppz, 1) = b1;
    v!(isuppz, 2) = bn;
    v!(z, r) = 1.0;
    let mut ztz = 1.0;

    // Compute the FP vector upwards from R
    if !sawnan1 && !sawnan2 {
        let mut i = r - 1;
        while i >= b1 {
            v!(z, i) = -(v!(work, indlpl + i) * v!(z, i + 1));
            if (v!(z, i).abs() + v!(z, i + 1).abs()) * v!(ld, i).abs() < gaptol {
                v!(z, i) = 0.0;
                v!(isuppz, 1) = i + 1;
                break;
            }
            ztz = fma(v!(z, i), v!(z, i), ztz);
            i -= 1;
        }
    } else {
        // Run slower loop if NaN occurred.
        let mut i = r - 1;
        while i >= b1 {
            if v!(z, i + 1) == 0.0 {
                v!(z, i) = -(v!(ld, i + 1) / v!(ld, i)) * v!(z, i + 2);
            } else {
                v!(z, i) = -(v!(work, indlpl + i) * v!(z, i + 1));
            }
            if (v!(z, i).abs() + v!(z, i + 1).abs()) * v!(ld, i).abs() < gaptol {
                v!(z, i) = 0.0;
                v!(isuppz, 1) = i + 1;
                break;
            }
            ztz = fma(v!(z, i), v!(z, i), ztz);
            i -= 1;
        }
    }

    // Compute the FP vector downwards from R in blocks of size BLKSIZ
    if !sawnan1 && !sawnan2 {
        for i in r..=(bn - 1) {
            v!(z, i + 1) = -(v!(work, indumn + i) * v!(z, i));
            if (v!(z, i).abs() + v!(z, i + 1).abs()) * v!(ld, i).abs() < gaptol {
                v!(z, i + 1) = 0.0;
                v!(isuppz, 2) = i;
                break;
            }
            ztz = fma(v!(z, i + 1), v!(z, i + 1), ztz);
        }
    } else {
        // Run slower loop if NaN occurred.
        for i in r..=(bn - 1) {
            if v!(z, i) == 0.0 {
                v!(z, i + 1) = -(v!(ld, i - 1) / v!(ld, i)) * v!(z, i - 1);
            } else {
                v!(z, i + 1) = -(v!(work, indumn + i) * v!(z, i));
            }
            if (v!(z, i).abs() + v!(z, i + 1).abs()) * v!(ld, i).abs() < gaptol {
                v!(z, i + 1) = 0.0;
                v!(isuppz, 2) = i;
                break;
            }
            ztz = fma(v!(z, i + 1), v!(z, i + 1), ztz);
        }
    }

    // Compute quantities for convergence test
    let tmp = 1.0 / ztz;
    let nrminv = tmp.sqrt();
    let resid = mingma.abs() * nrminv;
    let rqcorr = mingma * tmp;
    Lr1v {
        negcnt,
        r,
        nrminv,
        resid,
        rqcorr,
    }
}

/// `DLARRF`: a new relatively robust representation
/// `L+ D+ L+' = L D L' - sigma I` for the cluster `clstrt..=clend`, shifted
/// to one of its ends. Returns `(sigma, INFO)`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarrf(
    n: i32,
    d: *const f64,
    l: *const f64,
    ld: *const f64,
    clstrt: i32,
    clend: i32,
    w: *const f64,
    wgap: *const f64,
    werr: *const f64,
    spdiam: f64,
    clgapl: f64,
    clgapr: f64,
    pivmin: f64,
    dplus: *mut f64,
    lplus: *mut f64,
    work: *mut f64,
) -> (f64, i32) {
    const QUART: f64 = 0.25;
    const MAXGROWTH1: f64 = 8.0;
    const MAXGROWTH2: f64 = 8.0;
    const KTRYMAX: i32 = 1;
    const SLEFT: i32 = 1;
    const SRIGHT: i32 = 2;

    if n <= 0 {
        return (0.0, 0);
    }

    let fact = (1i32 << KTRYMAX) as f64;
    let eps = PREC;
    let shift: i32;
    let mut forcer = false;
    let mut sigma = 0.0;

    // Note that we cannot guarantee that for any of the shifts tried,
    // the factorization has a small or even moderate element growth.
    // There could be Ritz values at both ends of the cluster and despite
    // backing off, there are examples where all factorizations tried
    // (in IEEE mode, allowing zero pivots & infinities) have INFINITE
    // element growth.
    // For this reason, we should use PIVMIN in this subroutine so that at
    // least the L D L^T factorization exists. It can be checked afterwards
    // whether the element growth caused bad residuals/orthogonality.

    // Decide whether the code should accept the best among all
    // representations despite large element growth or signal INFO=1
    // Setting NOFAIL to .FALSE. for quick fix for bug 113
    let nofail = false;

    // Compute the average gap length of the cluster
    let clwdth = (v!(w, clend) - v!(w, clstrt)).abs() + v!(werr, clend) + v!(werr, clstrt);
    let avgap = clwdth / (clend - clstrt) as f64;
    let mingap = clgapl.min(clgapr);
    // Initial values for shifts to both ends of cluster
    let mut lsigma = v!(w, clstrt).min(v!(w, clend)) - v!(werr, clstrt);
    let mut rsigma = v!(w, clstrt).max(v!(w, clend)) + v!(werr, clend);

    // Use a small fudge to make sure that we really shift to the outside
    lsigma = fma(-(lsigma.abs() * 4.0), eps, lsigma);
    rsigma = fma(rsigma.abs() * 4.0, eps, rsigma);

    // Compute upper bounds for how much to back off the initial shifts
    let ldmax = fma(mingap, QUART, 2.0 * pivmin);
    let rdmax = fma(mingap, QUART, 2.0 * pivmin);

    let mut ldelta = avgap.max(v!(wgap, clstrt)) / fact;
    let mut rdelta = avgap.max(v!(wgap, clend - 1)) / fact;

    // Initialize the record of the best representation found
    let s0 = SAFMIN;
    let mut smlgrowth = 1.0 / s0;
    let fail = (n - 1) as f64 * mingap / (spdiam * eps);
    let fail2 = (n - 1) as f64 * mingap / (spdiam * eps.sqrt());
    let mut bestshift = lsigma;

    // while (KTRY <= KTRYMAX)
    let mut ktry = 0;
    let growthbound = MAXGROWTH1 * spdiam;

    let mut indx = 0;
    'retry: loop {
        // 5
        let mut sawnan1 = false;
        let mut sawnan2 = false;
        // Ensure that we do not back off too much of the initial shifts
        ldelta = ldmax.min(ldelta);
        rdelta = rdmax.min(rdelta);

        // Compute the element growth when shifting to both ends of the
        // cluster accept the shift if there is no element growth at one
        // of the two ends

        // Left end
        let mut s = -lsigma;
        v!(dplus, 1) = v!(d, 1) + s;
        if v!(dplus, 1).abs() < pivmin {
            v!(dplus, 1) = -pivmin;
            // Need to set SAWNAN1 because refined RRR test should not be
            // used in this case
            sawnan1 = true;
        }
        let mut max1 = v!(dplus, 1).abs();
        for i in 1..=(n - 1) {
            v!(lplus, i) = v!(ld, i) / v!(dplus, i);
            s = fma(v!(l, i), v!(lplus, i) * s, -lsigma);
            v!(dplus, i + 1) = v!(d, i + 1) + s;
            if v!(dplus, i + 1).abs() < pivmin {
                v!(dplus, i + 1) = -pivmin;
                // Need to set SAWNAN1 because refined RRR test should not
                // be used in this case
                sawnan1 = true;
            }
            max1 = max1.max(v!(dplus, i + 1).abs());
        }
        sawnan1 = sawnan1 || disnan(max1);

        if forcer || (max1 <= growthbound && !sawnan1) {
            sigma = lsigma;
            shift = SLEFT;
            break 'retry;
        }

        // Right end
        s = -rsigma;
        v!(work, 1) = v!(d, 1) + s;
        if v!(work, 1).abs() < pivmin {
            v!(work, 1) = -pivmin;
            // Need to set SAWNAN2 because refined RRR test should not be
            // used in this case
            sawnan2 = true;
        }
        let mut max2 = v!(work, 1).abs();
        for i in 1..=(n - 1) {
            v!(work, n + i) = v!(ld, i) / v!(work, i);
            s = fma(v!(l, i), v!(work, n + i) * s, -rsigma);
            v!(work, i + 1) = v!(d, i + 1) + s;
            if v!(work, i + 1).abs() < pivmin {
                v!(work, i + 1) = -pivmin;
                // Need to set SAWNAN2 because refined RRR test should not
                // be used in this case
                sawnan2 = true;
            }
            max2 = max2.max(v!(work, i + 1).abs());
        }
        sawnan2 = sawnan2 || disnan(max2);

        if forcer || (max2 <= growthbound && !sawnan2) {
            sigma = rsigma;
            shift = SRIGHT;
            break 'retry;
        }
        // If we are at this point, both shifts led to too much element
        // growth

        // Record the better of the two shifts (provided it didn't lead to
        // NaN)
        let mut goto50 = false;
        if sawnan1 && sawnan2 {
            // both MAX1 and MAX2 are NaN
            goto50 = true;
        } else {
            if !sawnan1 {
                indx = 1;
                if max1 <= smlgrowth {
                    smlgrowth = max1;
                    bestshift = lsigma;
                }
            }
            if !sawnan2 {
                if sawnan1 || max2 <= max1 {
                    indx = 2;
                }
                if max2 <= smlgrowth {
                    smlgrowth = max2;
                    bestshift = rsigma;
                }
            }
        }

        if !goto50 {
            // If we are here, both the left and the right shift led to
            // element growth. If the element growth is moderate, then
            // we may still accept the representation, if it passes a
            // refined test for RRR. This test supposes that no NaN
            // occurred. Moreover, we use the refined RRR test only for
            // isolated clusters.
            let dorrr1 = clwdth < mingap / 128.0 && max1.min(max2) < fail2 && !sawnan1 && !sawnan2;
            let tryrrr1 = true;
            if tryrrr1 && dorrr1 {
                if indx == 1 {
                    let mut tmp = v!(dplus, n).abs();
                    let mut znm2 = 1.0;
                    let mut prod = 1.0;
                    let mut oldp = 1.0;
                    let mut i = n - 1;
                    while i >= 1 {
                        if prod <= eps {
                            prod = ((v!(dplus, i + 1) * v!(work, n + i + 1))
                                / (v!(dplus, i) * v!(work, n + i)))
                                * oldp;
                        } else {
                            prod *= v!(work, n + i).abs();
                        }
                        oldp = prod;
                        znm2 = fma(prod, prod, znm2);
                        tmp = tmp.max((v!(dplus, i) * prod).abs());
                        i -= 1;
                    }
                    let rrr1 = tmp / (spdiam * znm2.sqrt());
                    if rrr1 <= MAXGROWTH2 {
                        sigma = lsigma;
                        shift = SLEFT;
                        break 'retry;
                    }
                } else if indx == 2 {
                    let mut tmp = v!(work, n).abs();
                    let mut znm2 = 1.0;
                    let mut prod = 1.0;
                    let mut oldp = 1.0;
                    let mut i = n - 1;
                    while i >= 1 {
                        if prod <= eps {
                            prod = ((v!(work, i + 1) * v!(lplus, i + 1))
                                / (v!(work, i) * v!(lplus, i)))
                                * oldp;
                        } else {
                            prod *= v!(lplus, i).abs();
                        }
                        oldp = prod;
                        znm2 = fma(prod, prod, znm2);
                        tmp = tmp.max((v!(work, i) * prod).abs());
                        i -= 1;
                    }
                    let rrr2 = tmp / (spdiam * znm2.sqrt());
                    if rrr2 <= MAXGROWTH2 {
                        sigma = rsigma;
                        shift = SRIGHT;
                        break 'retry;
                    }
                }
            }
        }

        // 50
        if ktry < KTRYMAX {
            // If we are here, both shifts failed also the RRR test.
            // Back off to the outside
            lsigma = (lsigma - ldelta).max(lsigma - ldmax);
            rsigma = (rsigma + rdelta).min(rsigma + rdmax);
            ldelta *= 2.0;
            rdelta *= 2.0;
            ktry += 1;
            continue 'retry;
        } else {
            // None of the representations investigated satisfied our
            // criteria. Take the best one we found.
            if smlgrowth < fail || nofail {
                lsigma = bestshift;
                rsigma = bestshift;
                forcer = true;
                continue 'retry;
            } else {
                return (sigma, 1);
            }
        }
    }

    // 100
    if shift == SRIGHT {
        // store new L and D back into DPLUS, LPLUS
        dcopy(n, work, 1, dplus, 1);
        dcopy(n - 1, pv!(work, n + 1), 1, lplus, 1);
    }
    (sigma, 0)
}

/// `DLARRV` for `DOL = 1`, `DOU = M`: the eigenvectors of every
/// eigenvalue `DLARRE` found, into `z`, with their supports `isuppz`.
/// `d`, `l` hold `DLARRE`'s representation (the shift of each block in its
/// last `l`). Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlarrv(
    n: i32,
    vl: f64,
    _vu: f64,
    d: *mut f64,
    l: *mut f64,
    pivmin: f64,
    isplit: *const i32,
    m: i32,
    dol: i32,
    dou: i32,
    minrgp: f64,
    rtol1_in: f64,
    rtol2_in: f64,
    w: *mut f64,
    werr: *mut f64,
    wgap: *mut f64,
    iblock: *const i32,
    indexw: *const i32,
    gers: *const f64,
    z: *mut f64,
    ldz: i32,
    isuppz: *mut i32,
    work: *mut f64,
    iwork: *mut i32,
) -> i32 {
    const MAXITR: i32 = 10;

    // Quick return if possible
    if n <= 0 || m <= 0 {
        return 0;
    }

    // The first N entries of WORK are reserved for the eigenvalues
    let indld = n + 1;
    let indlld = 2 * n + 1;
    let indwrk = 3 * n + 1;
    let minwsize = 12 * n;

    for i in 1..=minwsize {
        v!(work, i) = 0.0;
    }

    // IWORK(IINDR+1:IINDR+N) hold the twist indices R for the factorization
    // of the eigenvectors.
    let iindr = 0;
    // IWORK(IINDC1+1:IINC2+N) are used to store the clusters of the current
    // layer and the one above.
    let iindc1 = n;
    let iindc2 = 2 * n;
    let iindwk = 3 * n + 1;

    let miniwsize = 7 * n;
    for i in 1..=miniwsize {
        v!(iwork, i) = 0;
    }

    let mut zusedl = 1;
    if dol > 1 {
        // Set lower bound for use of Z
        zusedl = dol - 1;
    }
    let mut zusedu = m;
    if dou < m {
        // Set lower bound for use of Z
        zusedu = dou + 1;
    }
    // The width of the part of Z that is used
    let zusedw = zusedu - zusedl + 1;

    dlaset(b'F', n, zusedw, 0.0, 0.0, pm!(z, ldz, 1, zusedl), ldz);

    let eps = PREC;
    let rqtol = 2.0 * eps;

    // Set expert flags for standard code.
    let tryrqc = true;

    let (rtol1, rtol2) = if dol == 1 && dou == m {
        (rtol1_in, rtol2_in)
    } else {
        // Only selected eigenpairs are computed. Since the other
        // evalues are not refined by RQ iteration, bisection has to
        // compute to full accuracy.
        (4.0 * eps, 4.0 * eps)
    };

    // The entries WBEGIN:WEND in W, WERR, WGAP correspond to the
    // desired eigenvalues. The support of the nonzero eigenvector
    // entries is contained in the interval IBEGIN:IEND.
    // Remark that if k eigenpairs are desired, then the eigenvectors
    // are stored in k contiguous columns of Z.

    let mut ibegin = 1;
    let mut wbegin = 1;
    for jblk in 1..=v!(iblock, m) {
        let iend = v!(isplit, jblk);
        let mut sigma = v!(l, iend);
        // Find the eigenvectors of the submatrix indexed IBEGIN
        // through IEND.
        let mut wend = wbegin - 1;
        while wend < m && v!(iblock, wend + 1) == jblk {
            wend += 1;
        }
        if wend < wbegin {
            ibegin = iend + 1;
            continue;
        } else if wend < dol || wbegin > dou {
            ibegin = iend + 1;
            wbegin = wend + 1;
            continue;
        }

        // Find local spectral diameter of the block
        let mut gl = v!(gers, 2 * ibegin - 1);
        let mut gu = v!(gers, 2 * ibegin);
        for i in (ibegin + 1)..=iend {
            gl = v!(gers, 2 * i - 1).min(gl);
            gu = v!(gers, 2 * i).max(gu);
        }
        let spdiam = gu - gl;

        // OLDIEN is the last index of the previous block
        let oldien = ibegin - 1;
        // Calculate the size of the current block
        let in_ = iend - ibegin + 1;
        // The number of eigenvalues in the current block
        let im = wend - wbegin + 1;

        // This is for a 1x1 block
        if ibegin == iend {
            m!(z, ldz, ibegin, wbegin) = 1.0;
            v!(isuppz, 2 * wbegin - 1) = ibegin;
            v!(isuppz, 2 * wbegin) = ibegin;
            v!(w, wbegin) += sigma;
            v!(work, wbegin) = v!(w, wbegin);
            ibegin = iend + 1;
            wbegin += 1;
            continue;
        }

        // The desired (shifted) eigenvalues are stored in W(WBEGIN:WEND)
        // Note that these can be approximations, in this case, the corresp.
        // entries of WERR give the size of the uncertainty interval.
        // The eigenvalue approximations will be refined when necessary as
        // high relative accuracy is required for the computation of the
        // corresponding eigenvectors.
        dcopy(im, pv!(w, wbegin), 1, pv!(work, wbegin), 1);

        // We store in W the eigenvalue approximations w.r.t. the original
        // matrix T.
        for i in 1..=im {
            v!(w, wbegin + i - 1) += sigma;
        }

        // NDEPTH is the current depth of the representation tree
        let mut ndepth = 0;
        // PARITY is either 1 or 0
        let mut parity = 1;
        // NCLUS is the number of clusters for the next level of the
        // representation tree, we start with NCLUS = 1 for the root
        let mut nclus = 1;
        v!(iwork, iindc1 + 1) = 1;
        v!(iwork, iindc1 + 2) = im;

        // IDONE is the number of eigenvectors already computed in the
        // current block
        let mut idone = 0;
        // loop while( IDONE.LT.IM )
        // generate the representation tree for the current block and
        // compute the eigenvectors
        while idone < im {
            // This is a crude protection against infinitely deep trees
            if ndepth > m {
                return -2;
            }
            // breadth first processing of the current level of the
            // representation tree: OLDNCL = number of clusters on current
            // level
            let oldncl = nclus;
            // reset NCLUS to count the number of child clusters
            nclus = 0;

            parity = 1 - parity;
            let (oldcls, newcls) = if parity == 0 {
                (iindc1, iindc2)
            } else {
                (iindc2, iindc1)
            };
            // Process the clusters on the current level
            for i in 1..=oldncl {
                let j0 = oldcls + 2 * i;
                // OLDFST, OLDLST = first, last index of current cluster.
                //                  cluster indices start with 1 and are
                //                  relative to WBEGIN when accessing W,
                //                  WGAP, WERR, Z
                let oldfst = v!(iwork, j0 - 1);
                let oldlst = v!(iwork, j0);
                if ndepth > 0 {
                    // Retrieve relatively robust representation (RRR) of
                    // cluster that has been computed at the previous level
                    // The RRR is stored in Z and overwritten once the
                    // eigenvectors have been computed or when the cluster
                    // is refined
                    let j = if dol == 1 && dou == m {
                        // Get representation from location of the leftmost
                        // evalue of the cluster
                        wbegin + oldfst - 1
                    } else if wbegin + oldfst - 1 < dol {
                        // Get representation from the left end of Z array
                        dol - 1
                    } else if wbegin + oldfst - 1 > dou {
                        // Get representation from the right end of Z array
                        dou
                    } else {
                        wbegin + oldfst - 1
                    };
                    dcopy(in_, pm!(z, ldz, ibegin, j), 1, pv!(d, ibegin), 1);
                    dcopy(in_ - 1, pm!(z, ldz, ibegin, j + 1), 1, pv!(l, ibegin), 1);
                    sigma = m!(z, ldz, iend, j + 1);

                    // Set the corresponding entries in Z to zero
                    dlaset(b'F', in_, 2, 0.0, 0.0, pm!(z, ldz, ibegin, j), ldz);
                }

                // Compute DL and DLL of current RRR
                for j in ibegin..=(iend - 1) {
                    let tmp = v!(d, j) * v!(l, j);
                    v!(work, indld - 1 + j) = tmp;
                    v!(work, indlld - 1 + j) = tmp * v!(l, j);
                }

                if ndepth > 0 {
                    // P and Q are index of the first and last eigenvalue
                    // to compute within the current block
                    let p = v!(indexw, wbegin - 1 + oldfst);
                    let q = v!(indexw, wbegin - 1 + oldlst);
                    // Offset for the arrays WORK, WGAP and WERR, i.e., the
                    // P-OFFSET through the Q-OFFSET elements of these
                    // arrays are to be used.
                    let offset = v!(indexw, wbegin) - 1;
                    // perform limited bisection (if necessary) to get
                    // approximate eigenvalues to the precision needed.
                    let iinfo = dlarrb(
                        in_,
                        pv!(d, ibegin),
                        pv!(work, indlld + ibegin - 1),
                        p,
                        q,
                        rtol1,
                        rtol2,
                        offset,
                        pv!(work, wbegin),
                        pv!(wgap, wbegin),
                        pv!(werr, wbegin),
                        pv!(work, indwrk),
                        pv!(iwork, iindwk),
                        pivmin,
                        spdiam,
                        in_,
                    );
                    if iinfo != 0 {
                        return -1;
                    }
                    // We also recompute the extremal gaps. W holds all
                    // eigenvalues of the unshifted matrix and must be used
                    // for computation of WGAP, the entries of WORK might
                    // stem from RRRs with different shifts. The gaps from
                    // WBEGIN-1+OLDFST to WBEGIN-1+OLDLST are correctly
                    // computed in DLARRB. However, we only allow the gaps
                    // to become greater since this is what should happen
                    // when we decrease WERR
                    if oldfst > 1 {
                        v!(wgap, wbegin + oldfst - 2) = v!(wgap, wbegin + oldfst - 2).max(
                            v!(w, wbegin + oldfst - 1)
                                - v!(werr, wbegin + oldfst - 1)
                                - v!(w, wbegin + oldfst - 2)
                                - v!(werr, wbegin + oldfst - 2),
                        );
                    }
                    if wbegin + oldlst - 1 < wend {
                        v!(wgap, wbegin + oldlst - 1) = v!(wgap, wbegin + oldlst - 1).max(
                            v!(w, wbegin + oldlst)
                                - v!(werr, wbegin + oldlst)
                                - v!(w, wbegin + oldlst - 1)
                                - v!(werr, wbegin + oldlst - 1),
                        );
                    }
                    // Each time the eigenvalues in WORK get refined, we
                    // store the newly found approximation with all shifts
                    // applied in W
                    for j in oldfst..=oldlst {
                        v!(w, wbegin + j - 1) = v!(work, wbegin + j - 1) + sigma;
                    }
                }

                // Process the current node.
                let mut newfst = oldfst;
                for j in oldfst..=oldlst {
                    let newlst;
                    if j == oldlst {
                        // we are at the right end of the cluster, this is
                        // also the boundary of the child cluster
                        newlst = j;
                    } else if v!(wgap, wbegin + j - 1) >= minrgp * v!(work, wbegin + j - 1).abs() {
                        // the right relative gap is big enough, the child
                        // cluster (NEWFST,..,NEWLST) is well separated from
                        // the following
                        newlst = j;
                    } else {
                        // inside a child cluster, the relative gap is not
                        // big enough.
                        continue;
                    }

                    // Compute size of child cluster found
                    let newsiz = newlst - newfst + 1;

                    // NEWFTT is the place in Z where the new RRR or the
                    // computed eigenvector is to be stored
                    let newftt = if dol == 1 && dou == m {
                        // Store representation at location of the leftmost
                        // evalue of the cluster
                        wbegin + newfst - 1
                    } else if wbegin + newfst - 1 < dol {
                        // Store representation at the left end of Z array
                        dol - 1
                    } else if wbegin + newfst - 1 > dou {
                        // Store representation at the right end of Z array
                        dou
                    } else {
                        wbegin + newfst - 1
                    };

                    'node: {
                        if newsiz > 1 {
                            // Current child is not a singleton but a
                            // cluster. Compute and store new representation
                            // of child.
                            //
                            // Compute left and right cluster gap.
                            //
                            // LGAP and RGAP are not computed from WORK
                            // because the eigenvalue approximations may stem
                            // from RRRs different shifts. However, W hold
                            // all eigenvalues of the unshifted matrix.
                            // Still, the entries in WGAP have to be computed
                            // from WORK since the entries in W might be of
                            // the same order so that gaps are not exhibited
                            // correctly for very close eigenvalues.
                            let lgap = if newfst == 1 {
                                0.0f64.max(v!(w, wbegin) - v!(werr, wbegin) - vl)
                            } else {
                                v!(wgap, wbegin + newfst - 2)
                            };
                            let rgap = v!(wgap, wbegin + newlst - 1);

                            // Compute left- and rightmost eigenvalue of
                            // child to high precision in order to shift as
                            // close as possible and obtain as large relative
                            // gaps as possible
                            for k in 1..=2 {
                                let p = if k == 1 {
                                    v!(indexw, wbegin - 1 + newfst)
                                } else {
                                    v!(indexw, wbegin - 1 + newlst)
                                };
                                let offset = v!(indexw, wbegin) - 1;
                                dlarrb(
                                    in_,
                                    pv!(d, ibegin),
                                    pv!(work, indlld + ibegin - 1),
                                    p,
                                    p,
                                    rqtol,
                                    rqtol,
                                    offset,
                                    pv!(work, wbegin),
                                    pv!(wgap, wbegin),
                                    pv!(werr, wbegin),
                                    pv!(work, indwrk),
                                    pv!(iwork, iindwk),
                                    pivmin,
                                    spdiam,
                                    in_,
                                );
                            }

                            if wbegin + newlst - 1 < dol || wbegin + newfst - 1 > dou {
                                // if the cluster contains no desired
                                // eigenvalues skip the computation of that
                                // branch of the rep. tree
                                //
                                // We could skip before the refinement of
                                // the extremal eigenvalues of the child,
                                // but then the representation tree could
                                // be different from the one when nothing is
                                // skipped. For this reason we skip at this
                                // place.
                                idone += newlst - newfst + 1;
                                break 'node;
                            }

                            // Compute RRR of child cluster.
                            // Note that the new RRR is stored in Z
                            //
                            // DLARRF needs LWORK = 2*N
                            let (tau, iinfo) = dlarrf(
                                in_,
                                pv!(d, ibegin),
                                pv!(l, ibegin),
                                pv!(work, indld + ibegin - 1),
                                newfst,
                                newlst,
                                pv!(work, wbegin),
                                pv!(wgap, wbegin),
                                pv!(werr, wbegin),
                                spdiam,
                                lgap,
                                rgap,
                                pivmin,
                                pm!(z, ldz, ibegin, newftt),
                                pm!(z, ldz, ibegin, newftt + 1),
                                pv!(work, indwrk),
                            );
                            if iinfo == 0 {
                                // a new RRR for the cluster was found by
                                // DLARRF update shift and store it
                                let ssigma = sigma + tau;
                                m!(z, ldz, iend, newftt + 1) = ssigma;
                                // WORK() are the midpoints and WERR() the
                                // semi-width Note that the entries in W
                                // are unchanged.
                                for k in newfst..=newlst {
                                    let wk_old = v!(work, wbegin + k - 1);
                                    v!(work, wbegin + k - 1) = wk_old - tau;
                                    let fudge = fma(
                                        3.0 * eps,
                                        wk_old.abs(),
                                        4.0 * eps * v!(work, wbegin + k - 1).abs(),
                                    );
                                    // Fudge errors
                                    v!(werr, wbegin + k - 1) += fudge;
                                    // Gaps are not fudged. Provided that
                                    // WERR is small when eigenvalues are
                                    // close, a zero gap indicates that a
                                    // new representation is needed for
                                    // resolving the cluster. A fudge could
                                    // lead to a wrong decision of
                                    // judging eigenvalues 'separated' which
                                    // in reality are not. This could have a
                                    // negative impact on the orthogonality
                                    // of the computed eigenvectors.
                                }

                                nclus += 1;
                                let k = newcls + 2 * nclus;
                                v!(iwork, k - 1) = newfst;
                                v!(iwork, k) = newlst;
                            } else {
                                return -2;
                            }
                        } else {
                            // Compute eigenvector of singleton
                            let mut iter = 0;

                            let tol = 4.0 * (in_ as f64).ln() * eps;

                            let k = newfst;
                            let windex = wbegin + k - 1;
                            let windmn = (windex - 1).max(1);
                            let windpl = (windex + 1).min(m);
                            let mut lambda = v!(work, windex);
                            // Check if eigenvector computation is to be
                            // skipped
                            let eskip = windex < dol || windex > dou;
                            let mut savgap = 0.0;
                            if !eskip {
                                let mut left = v!(work, windex) - v!(werr, windex);
                                let mut right = v!(work, windex) + v!(werr, windex);
                                let indeig = v!(indexw, windex);
                                // Note that since we compute the
                                // eigenpairs for a child, all eigenvalue
                                // approximations are w.r.t the same shift.
                                // In this case, the entries in WORK should
                                // be used for computing the gaps since
                                // they exhibit even very small differences
                                // in the eigenvalues, as opposed to the
                                // entries in W which might "look" the same.
                                let lgap = if k == 1 {
                                    // In the case RANGE='I' and with not
                                    // much initial accuracy in LAMBDA and
                                    // VL, the formula
                                    // LGAP = MAX( ZERO, (SIGMA - VL) + LAMBDA )
                                    // can lead to an overestimation of the
                                    // left gap and thus to inadequately
                                    // early RQI 'convergence'.
                                    // Prevent this by forcing a small left
                                    // gap.
                                    eps * left.abs().max(right.abs())
                                } else {
                                    v!(wgap, windmn)
                                };
                                let rgap = if k == im {
                                    // In the case RANGE='I' and with not
                                    // much initial accuracy in LAMBDA and
                                    // VU, the formula can lead to an
                                    // overestimation of the right gap and
                                    // thus to inadequately early RQI
                                    // 'convergence'.
                                    // Prevent this by forcing a small right
                                    // gap.
                                    eps * left.abs().max(right.abs())
                                } else {
                                    v!(wgap, windex)
                                };
                                let gap = lgap.min(rgap);
                                let gaptol = if k == 1 || k == im {
                                    // The eigenvector support can become
                                    // wrong because significant entries
                                    // could be cut off due to a large
                                    // GAPTOL parameter in LAR1V. Prevent
                                    // this.
                                    0.0
                                } else {
                                    gap * eps
                                };
                                let mut isupmn = in_;
                                let mut isupmx = 1;
                                // Update WGAP so that it holds the minimum
                                // gap to the left or the right. This is
                                // crucial in the case where bisection is
                                // used to ensure that the eigenvalue is
                                // refined up to the required precision.
                                // The correct value is restored afterwards.
                                savgap = v!(wgap, windex);
                                v!(wgap, windex) = gap;
                                // We want to use the Rayleigh Quotient
                                // Correction as often as possible since it
                                // converges quadratically when we are close
                                // enough to the desired eigenvalue. However,
                                // the Rayleigh Quotient can have the wrong
                                // sign and lead us away from the desired
                                // eigenvalue. In this case, the best we can
                                // do is to use bisection.
                                let mut usedbs = false;
                                let mut usedrq = false;
                                // Bisection is initially turned off unless
                                // it is forced
                                let mut needbs = !tryrqc;
                                let mut bstres = 0.0;
                                let mut bstw = 0.0;
                                let mut res;
                                loop {
                                    // 120
                                    // Check if bisection should be used to
                                    // refine eigenvalue
                                    if needbs {
                                        // Take the bisection as new iterate
                                        usedbs = true;
                                        let itmp1 = v!(iwork, iindr + windex);
                                        let offset = v!(indexw, wbegin) - 1;
                                        let iinfo = dlarrb(
                                            in_,
                                            pv!(d, ibegin),
                                            pv!(work, indlld + ibegin - 1),
                                            indeig,
                                            indeig,
                                            0.0,
                                            2.0 * eps,
                                            offset,
                                            pv!(work, wbegin),
                                            pv!(wgap, wbegin),
                                            pv!(werr, wbegin),
                                            pv!(work, indwrk),
                                            pv!(iwork, iindwk),
                                            pivmin,
                                            spdiam,
                                            itmp1,
                                        );
                                        if iinfo != 0 {
                                            return -3;
                                        }
                                        lambda = v!(work, windex);
                                        // Reset twist index from inaccurate
                                        // LAMBDA to force computation of true
                                        // MINGMA
                                        v!(iwork, iindr + windex) = 0;
                                    }
                                    // Given LAMBDA, compute the eigenvector.
                                    res = dlar1v(
                                        in_,
                                        1,
                                        in_,
                                        lambda,
                                        pv!(d, ibegin),
                                        pv!(l, ibegin),
                                        pv!(work, indld + ibegin - 1),
                                        pv!(work, indlld + ibegin - 1),
                                        pivmin,
                                        gaptol,
                                        pm!(z, ldz, ibegin, windex),
                                        !usedbs,
                                        v!(iwork, iindr + windex),
                                        pv!(isuppz, 2 * windex - 1),
                                        pv!(work, indwrk),
                                    );
                                    v!(iwork, iindr + windex) = res.r;
                                    if iter == 0 || res.resid < bstres {
                                        bstres = res.resid;
                                        bstw = lambda;
                                    }
                                    isupmn = isupmn.min(v!(isuppz, 2 * windex - 1));
                                    isupmx = isupmx.max(v!(isuppz, 2 * windex));
                                    iter += 1;

                                    // sin alpha <= |resid|/gap
                                    // Note that both the residual and the gap
                                    // are proportional to the matrix, so
                                    // ||T|| doesn't play a role in the quotient
                                    //
                                    // Convergence test for Rayleigh-Quotient
                                    // iteration (omitted if Bisection has been
                                    // used)
                                    if res.resid > tol * gap
                                        && res.rqcorr.abs() > rqtol * lambda.abs()
                                        && !usedbs
                                    {
                                        // We need to check that the RQCORR
                                        // update doesn't move the eigenvalue
                                        // away from the desired one and
                                        // towards a neighbor. -> protection
                                        // with bisection
                                        let sgndef = if indeig <= res.negcnt {
                                            // The wanted eigenvalue lies to
                                            // the left
                                            -1.0
                                        } else {
                                            // The wanted eigenvalue lies to
                                            // the right
                                            1.0
                                        };
                                        // We only use the RQCORR if it
                                        // improves the the iterate reasonably.
                                        if res.rqcorr * sgndef >= 0.0
                                            && lambda + res.rqcorr <= right
                                            && lambda + res.rqcorr >= left
                                        {
                                            usedrq = true;
                                            // Store new midpoint of
                                            // bisection interval in WORK
                                            if sgndef == 1.0 {
                                                // The current LAMBDA is on
                                                // the left of the true
                                                // eigenvalue
                                                left = lambda;
                                            } else {
                                                // The current LAMBDA is on
                                                // the right of the true
                                                // eigenvalue
                                                right = lambda;
                                            }
                                            v!(work, windex) = 0.5 * (right + left);
                                            // Take RQCORR since it has the
                                            // correct sign and improves the
                                            // iterate reasonably
                                            lambda += res.rqcorr;
                                            // Update width of error interval
                                            v!(werr, windex) = 0.5 * (right - left);
                                        } else {
                                            needbs = true;
                                        }
                                        if right - left < rqtol * lambda.abs() {
                                            // The eigenvalue is computed to
                                            // bisection accuracy compute
                                            // eigenvector and stop
                                            usedbs = true;
                                            continue;
                                        } else if iter < MAXITR {
                                            continue;
                                        } else if iter == MAXITR {
                                            needbs = true;
                                            continue;
                                        } else {
                                            return 5;
                                        }
                                    } else {
                                        let mut stp2ii = false;
                                        if usedrq && usedbs && bstres <= res.resid {
                                            lambda = bstw;
                                            stp2ii = true;
                                        }
                                        if stp2ii {
                                            // improve error angle by second
                                            // step
                                            res = dlar1v(
                                                in_,
                                                1,
                                                in_,
                                                lambda,
                                                pv!(d, ibegin),
                                                pv!(l, ibegin),
                                                pv!(work, indld + ibegin - 1),
                                                pv!(work, indlld + ibegin - 1),
                                                pivmin,
                                                gaptol,
                                                pm!(z, ldz, ibegin, windex),
                                                !usedbs,
                                                v!(iwork, iindr + windex),
                                                pv!(isuppz, 2 * windex - 1),
                                                pv!(work, indwrk),
                                            );
                                            v!(iwork, iindr + windex) = res.r;
                                        }
                                        v!(work, windex) = lambda;
                                        break;
                                    }
                                }

                                // Compute FP-vector support w.r.t. whole
                                // matrix
                                v!(isuppz, 2 * windex - 1) += oldien;
                                v!(isuppz, 2 * windex) += oldien;
                                let zfrom = v!(isuppz, 2 * windex - 1);
                                let zto = v!(isuppz, 2 * windex);
                                isupmn += oldien;
                                isupmx += oldien;
                                // Ensure vector is ok if support in the RQI
                                // has changed
                                if isupmn < zfrom {
                                    for ii in isupmn..=(zfrom - 1) {
                                        m!(z, ldz, ii, windex) = 0.0;
                                    }
                                }
                                if isupmx > zto {
                                    for ii in (zto + 1)..=isupmx {
                                        m!(z, ldz, ii, windex) = 0.0;
                                    }
                                }
                                dscal(zto - zfrom + 1, res.nrminv, pm!(z, ldz, zfrom, windex), 1);
                            }
                            // 125
                            // Update W
                            v!(w, windex) = lambda + sigma;
                            // Recompute the gaps on the left and right
                            // But only allow them to become larger and not
                            // smaller (which can only happen through
                            // "bad" cancellation and doesn't reflect the
                            // theory where the initial gaps are
                            // underestimated due to WERR being too crude.)
                            if !eskip {
                                if k > 1 {
                                    v!(wgap, windmn) = v!(wgap, windmn).max(
                                        v!(w, windex)
                                            - v!(werr, windex)
                                            - v!(w, windmn)
                                            - v!(werr, windmn),
                                    );
                                }
                                if windex < wend {
                                    v!(wgap, windex) = savgap.max(
                                        v!(w, windpl)
                                            - v!(werr, windpl)
                                            - v!(w, windex)
                                            - v!(werr, windex),
                                    );
                                }
                            }
                            idone += 1;
                        }
                    }
                    // here ends the code for the current child
                    //
                    // 139
                    // Proceed to any remaining child nodes
                    newfst = j + 1;
                }
            }
            ndepth += 1;
        }
        ibegin = iend + 1;
        wbegin = wend + 1;
    }
    0
}
