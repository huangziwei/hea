//! `DSTEMR` for `RANGE = 'A'`: all eigenpairs of a symmetric tridiagonal
//! matrix by the MRRR algorithm (`DLARRE` representations and dqds
//! eigenvalues, `DLARRV` vectors, `DLARRJ` refinement to relative accuracy).

use super::aux::{dlae2, dlaev2, dlanst_m, dlasrt};
use super::blas::{dcopy, dscal, dswap};
use super::larrv::dlarrv;
use super::lasq::dlasq2;
use super::{fma, PREC, SAFMIN};

/// `DLARRR`: whether the tridiagonal `(d, e)` warrants the relatively
/// accurate (more expensive) computation. Returns `INFO` (0 if so).
unsafe fn dlarrr(n: i32, d: *const f64, e: *const f64) -> i32 {
    const RELCOND: f64 = 0.999;
    if n <= 0 {
        return 0;
    }
    let safmin = SAFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let rmin = smlnum.sqrt();

    // Tests for relative accuracy
    let mut yesrel = true;
    let mut offdig = 0.0;
    let mut tmp = v!(d, 1).abs().sqrt();
    if tmp < rmin {
        yesrel = false;
    }
    if yesrel {
        for i in 2..=n {
            let tmp2 = v!(d, i).abs().sqrt();
            if tmp2 < rmin {
                yesrel = false;
            }
            if !yesrel {
                break;
            }
            let offdig2 = v!(e, i - 1).abs() / (tmp * tmp2);
            if offdig + offdig2 >= RELCOND {
                yesrel = false;
            }
            if !yesrel {
                break;
            }
            tmp = tmp2;
            offdig = offdig2;
        }
    }
    if yesrel {
        0
    } else {
        1
    }
}

/// `DLARRA`: splitting points of the tridiagonal where off-diagonal
/// entries are negligible (zeroed in `e` and `e2`). Returns `NSPLIT`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarra(
    n: i32,
    d: *const f64,
    e: *mut f64,
    e2: *mut f64,
    spltol: f64,
    tnrm: f64,
    isplit: *mut i32,
) -> i32 {
    let mut nsplit = 1;
    if n <= 0 {
        return nsplit;
    }
    if spltol < 0.0 {
        // Criterion based on absolute off-diagonal value
        let tmp1 = spltol.abs() * tnrm;
        for i in 1..=(n - 1) {
            let eabs = v!(e, i).abs();
            if eabs <= tmp1 {
                v!(e, i) = 0.0;
                v!(e2, i) = 0.0;
                v!(isplit, nsplit) = i;
                nsplit += 1;
            }
        }
    } else {
        // Criterion that guarantees relative accuracy
        for i in 1..=(n - 1) {
            let eabs = v!(e, i).abs();
            if eabs <= spltol * v!(d, i).abs().sqrt() * v!(d, i + 1).abs().sqrt() {
                v!(e, i) = 0.0;
                v!(e2, i) = 0.0;
                v!(isplit, nsplit) = i;
                nsplit += 1;
            }
        }
    }
    v!(isplit, nsplit) = n;
    nsplit
}

/// `DLARRC`: Sturm counts of the tridiagonal (`jobt = 'T'`) or of its
/// `L D L'` factorization (`'L'`) on `(vl, vu]`; returns
/// `(eigcnt, lcnt, rcnt)`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarrc(
    jobt: u8,
    n: i32,
    vl: f64,
    vu: f64,
    d: *const f64,
    e: *const f64,
) -> (i32, i32, i32) {
    let mut lcnt = 0;
    let mut rcnt = 0;
    if n <= 0 {
        return (0, 0, 0);
    }
    if jobt == b'T' || jobt == b't' {
        // Sturm sequence count on T
        let mut lpivot = v!(d, 1) - vl;
        let mut rpivot = v!(d, 1) - vu;
        if lpivot <= 0.0 {
            lcnt += 1;
        }
        if rpivot <= 0.0 {
            rcnt += 1;
        }
        for i in 1..=(n - 1) {
            let tmp = v!(e, i) * v!(e, i);
            lpivot = (v!(d, i + 1) - vl) - tmp / lpivot;
            rpivot = (v!(d, i + 1) - vu) - tmp / rpivot;
            if lpivot <= 0.0 {
                lcnt += 1;
            }
            if rpivot <= 0.0 {
                rcnt += 1;
            }
        }
    } else {
        // Sturm sequence count on L D L^T
        let mut sl = -vl;
        let mut su = -vu;
        for i in 1..=(n - 1) {
            let lpivot = v!(d, i) + sl;
            let rpivot = v!(d, i) + su;
            if lpivot <= 0.0 {
                lcnt += 1;
            }
            if rpivot <= 0.0 {
                rcnt += 1;
            }
            let tmp = v!(e, i) * v!(d, i) * v!(e, i);
            let tmp2 = tmp / lpivot;
            if tmp2 == 0.0 {
                sl = tmp - vl;
            } else {
                sl = fma(sl, tmp2, -vl);
            }
            let tmp2 = tmp / rpivot;
            if tmp2 == 0.0 {
                su = tmp - vu;
            } else {
                su = fma(su, tmp2, -vu);
            }
        }
        let lpivot = v!(d, n) + sl;
        let rpivot = v!(d, n) + su;
        if lpivot <= 0.0 {
            lcnt += 1;
        }
        if rpivot <= 0.0 {
            rcnt += 1;
        }
    }
    (rcnt - lcnt, lcnt, rcnt)
}

/// `DLARRK`: the `iw`-th eigenvalue of the tridiagonal (`d`, squared
/// off-diagonals `e2`) by bisection in `[gl, gu]`; returns
/// `(w, werr, info)`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarrk(
    n: i32,
    iw: i32,
    gl: f64,
    gu: f64,
    d: *const f64,
    e2: *const f64,
    pivmin: f64,
    reltol: f64,
) -> (f64, f64, i32) {
    const FUDGE: f64 = 2.0;
    if n <= 0 {
        return (0.0, 0.0, 0);
    }
    let eps = PREC;
    let tnorm = gl.abs().max(gu.abs());
    let rtoli = reltol;
    let atoli = FUDGE * 2.0 * pivmin;

    let itmax = (((tnorm + pivmin).ln() - pivmin.ln()) / 2.0f64.ln()) as i32 + 2;

    let mut info = -1;

    let t = (FUDGE * tnorm) * eps;
    let mut left = fma(-t, n as f64, gl) - atoli;
    let mut right = atoli + fma(t, n as f64, gu);
    let mut it = 0;

    loop {
        // Check if interval converged or maximum number of iterations
        // reached
        let tmp1 = (right - left).abs();
        let tmp2 = right.abs().max(left.abs());
        if tmp1 < atoli.max(pivmin).max(rtoli * tmp2) {
            info = 0;
            break;
        }
        if it > itmax {
            break;
        }

        // Count number of negative pivots for mid-point
        it += 1;
        let mid = 0.5 * (left + right);
        let mut negcnt = 0;
        let mut tmp1 = v!(d, 1) - mid;
        if tmp1.abs() < pivmin {
            tmp1 = -pivmin;
        }
        if tmp1 <= 0.0 {
            negcnt += 1;
        }

        for i in 2..=n {
            tmp1 = v!(d, i) - v!(e2, i - 1) / tmp1 - mid;
            if tmp1.abs() < pivmin {
                tmp1 = -pivmin;
            }
            if tmp1 <= 0.0 {
                negcnt += 1;
            }
        }

        if negcnt >= iw {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Converged or maximum number of iterations reached
    (0.5 * (left + right), 0.5 * (right - left).abs(), info)
}

include!("mm_table.rs");

/// `DLARUV`: `n <= 128` uniform (0, 1) numbers from the multiplicative
/// congruential generator with 48-bit state `iseed`, advanced in place.
unsafe fn dlaruv(iseed: &mut [i32; 4], n: i32, x: *mut f64) {
    const LV: i32 = 128;
    const IPW2: i32 = 4096;
    const R: f64 = 1.0 / IPW2 as f64;
    if n < 1 {
        return;
    }
    let mut i1 = iseed[0];
    let mut i2 = iseed[1];
    let mut i3 = iseed[2];
    let mut i4 = iseed[3];
    let (mut it1, mut it2, mut it3, mut it4) = (0, 0, 0, 0);
    for i in 1..=n.min(LV) {
        let mm = MM[(i - 1) as usize];
        loop {
            // Multiply the seed by i-th power of the multiplier modulo 2**48
            it4 = i4 * mm[3];
            it3 = it4 / IPW2;
            it4 -= IPW2 * it3;
            it3 = it3 + i3 * mm[3] + i4 * mm[2];
            it2 = it3 / IPW2;
            it3 -= IPW2 * it2;
            it2 = it2 + i2 * mm[3] + i3 * mm[2] + i4 * mm[1];
            it1 = it2 / IPW2;
            it2 -= IPW2 * it1;
            it1 = it1 + i1 * mm[3] + i2 * mm[2] + i3 * mm[1] + i4 * mm[0];
            it1 %= IPW2;

            // Convert 48-bit integer to a real number in the interval (0,1)
            let t = fma(it4 as f64, R, it3 as f64);
            let t = fma(t, R, it2 as f64);
            v!(x, i) = fma(t, R, it1 as f64) * R;

            if v!(x, i) == 1.0 {
                // If a real number has n bits of precision, and the first
                // n bits of the 48-bit integer above happen to be all 1
                // (which will occur about once every 2**n calls), then
                // X( I ) will be rounded to exactly 1.0. Since X( I ) is
                // not supposed to return exactly 0.0 or 1.0, the
                // statistically correct thing to do in this situation is
                // simply to iterate again.
                i1 += 2;
                i2 += 2;
                i3 += 2;
                i4 += 2;
                continue;
            }
            break;
        }
    }

    // Return final value of seed
    iseed[0] = it1;
    iseed[1] = it2;
    iseed[2] = it3;
    iseed[3] = it4;
}

/// `DLARNV` with `IDIST = 2`: `n` uniform (-1, 1) numbers into `x`.
pub(crate) unsafe fn dlarnv2(iseed: &mut [i32; 4], n: i32, x: *mut f64) {
    const LV: i32 = 128;
    let mut u = [0.0f64; LV as usize];
    let mut iv = 1;
    while iv <= n {
        let il = (LV / 2).min(n - iv + 1);
        // Call DLARUV to generate IL2 numbers from a uniform (0,1)
        // distribution (IL2 <= LV)
        dlaruv(iseed, il, u.as_mut_ptr());
        // Convert generated numbers to uniform (-1,1) distribution
        for i in 1..=il {
            v!(x, iv + i - 1) = fma(u[(i - 1) as usize], 2.0, -1.0);
        }
        iv += LV / 2;
    }
}

/// Outputs of [`dlarre`] that its caller reads back.
pub(crate) struct Lrre {
    pub vl: f64,
    pub vu: f64,
    pub nsplit: i32,
    pub m: i32,
    pub pivmin: f64,
    pub info: i32,
}

/// `DLARRE` for `RANGE = 'A'`: splits the tridiagonal, and for each block
/// chooses a shift `sigma` and the root representation `L D L'` of
/// `T - sigma I` (stored over `d`, `e`, with `sigma` in the block's last `e`),
/// then all of its eigenvalues by dqds (`DLASQ2`).
#[allow(clippy::too_many_arguments)]
unsafe fn dlarre_all(
    n: i32,
    d: *mut f64,
    e: *mut f64,
    e2: *mut f64,
    spltol: f64,
    isplit: *mut i32,
    w: *mut f64,
    werr: *mut f64,
    wgap: *mut f64,
    iblock: *mut i32,
    indexw: *mut i32,
    gers: *mut f64,
    work: *mut f64,
) -> Lrre {
    const HNDRD: f64 = 100.0;
    const PERT: f64 = 8.0;
    const FOURTH: f64 = 0.25;
    const MAXGROWTH: f64 = 64.0;
    const FUDGE: f64 = 2.0;
    const MAXTRY: i32 = 6;

    let mut out = Lrre {
        vl: 0.0,
        vu: 0.0,
        nsplit: 0,
        m: 0,
        pivmin: 0.0,
        info: 0,
    };
    if n <= 0 {
        return out;
    }

    let safmin = SAFMIN;
    let eps = PREC;

    // Set parameters
    let rtl = eps.sqrt();

    // Treat case of 1x1 matrix for quick return
    if n == 1 {
        out.m = 1;
        v!(w, 1) = v!(d, 1);
        // The computation error of the eigenvalue is zero
        v!(werr, 1) = 0.0;
        v!(wgap, 1) = 0.0;
        v!(iblock, 1) = 1;
        v!(indexw, 1) = 1;
        v!(gers, 1) = v!(d, 1);
        v!(gers, 2) = v!(d, 1);
        // store the shift for the initial RRR, which is zero in this case
        v!(e, 1) = 0.0;
        return out;
    }

    // General case: tridiagonal matrix of order > 1
    //
    // Init WERR, WGAP. Compute Gerschgorin intervals and spectral
    // diameter. Compute maximum off-diagonal entry and pivmin.
    let mut gl = v!(d, 1);
    let mut gu = v!(d, 1);
    let mut eold = 0.0;
    let mut emax = 0.0f64;
    v!(e, n) = 0.0;
    for i in 1..=n {
        v!(werr, i) = 0.0;
        v!(wgap, i) = 0.0;
        let eabs = v!(e, i).abs();
        if eabs >= emax {
            emax = eabs;
        }
        let tmp1 = eabs + eold;
        v!(gers, 2 * i - 1) = v!(d, i) - tmp1;
        gl = gl.min(v!(gers, 2 * i - 1));
        v!(gers, 2 * i) = v!(d, i) + tmp1;
        gu = gu.max(v!(gers, 2 * i));
        eold = eabs;
    }
    // The minimum pivot allowed in the Sturm sequence for T
    let pivmin = safmin * 1.0f64.max(emax * emax);
    out.pivmin = pivmin;
    // Compute spectral diameter. The Gerschgorin bounds give an estimate
    // that is wrong by at most a factor of SQRT(2)
    let spdiam = gu - gl;

    // Compute splitting points
    let nsplit = dlarra(n, d, e, e2, spltol, spdiam, isplit);
    out.nsplit = nsplit;

    // Set interval [VL,VU] that contains all eigenvalues
    let vl = gl;
    let vu = gu;
    out.vl = vl;
    out.vu = vu;

    // Loop over unreduced blocks
    let mut m = 0;
    let mut ibegin = 1;
    let mut wbegin = 1;
    for jblk in 1..=nsplit {
        let iend = v!(isplit, jblk);
        let in_ = iend - ibegin + 1;

        // 1 X 1 block
        if in_ == 1 {
            m += 1;
            v!(w, m) = v!(d, ibegin);
            v!(werr, m) = 0.0;
            // The gap for a single block doesn't matter for the later
            // algorithm and is assigned an arbitrary large value
            v!(wgap, m) = 0.0;
            v!(iblock, m) = jblk;
            v!(indexw, m) = 1;
            wbegin += 1;
            // E( IEND ) holds the shift for the initial RRR
            v!(e, iend) = 0.0;
            ibegin = iend + 1;
            continue;
        }

        // Blocks of size larger than 1x1
        //
        // E( IEND ) will hold the shift for the initial RRR, for now set
        // it =0
        v!(e, iend) = 0.0;

        // Find local outer bounds GL,GU for the block
        let mut gl = v!(d, ibegin);
        let mut gu = v!(d, ibegin);
        for i in ibegin..=iend {
            gl = v!(gers, 2 * i - 1).min(gl);
            gu = v!(gers, 2 * i).max(gu);
        }

        // Case of DQDS
        // Find approximations to the extremal eigenvalues of the block
        let (tmp, tmp1, iinfo) =
            dlarrk(in_, 1, gl, gu, pv!(d, ibegin), pv!(e2, ibegin), pivmin, rtl);
        if iinfo != 0 {
            out.info = -1;
            return out;
        }
        let x = tmp - tmp1;
        let isleft = fma(-(eps * HNDRD), x.abs(), x).max(gl);

        let (tmp, tmp1, iinfo) = dlarrk(
            in_,
            in_,
            gl,
            gu,
            pv!(d, ibegin),
            pv!(e2, ibegin),
            pivmin,
            rtl,
        );
        if iinfo != 0 {
            out.info = -1;
            return out;
        }
        let x = tmp + tmp1;
        let isrght = fma(x.abs(), eps * HNDRD, x).min(gu);
        // Improve the estimate of the spectral diameter
        let spdiam = isrght - isleft;

        // Decide whether dqds or bisection is more efficient
        let usedqd = true;
        // Use DQDS for all eigenvalues
        let indl = 1;
        let indu = in_;
        let mb = in_;
        let wend = wbegin + mb - 1;
        // Calculate gaps for the current block
        // In later stages, when representations for individual
        // eigenvalues are different, we use SIGMA = E( IEND ).
        let s1 = fma(spdiam, FOURTH, isleft);
        let s2 = fma(-spdiam, FOURTH, isrght);

        // Compute the negcount at the 1/4 and 3/4 points
        let (mut cnt1, mut cnt2) = (0, 0);
        if mb > 1 {
            let (_cnt, c1, c2) = dlarrc(b'T', in_, s1, s2, pv!(d, ibegin), pv!(e, ibegin));
            cnt1 = c1;
            cnt2 = c2;
        }

        let mut sigma;
        let sgndef;
        if mb == 1 {
            sigma = gl;
            sgndef = 1.0;
        } else if cnt1 - indl >= indu - cnt2 {
            sigma = isleft.max(gl);
            sgndef = 1.0;
        } else {
            sigma = isrght.min(gu);
            sgndef = -1.0;
        }

        // An initial SIGMA has been chosen that will be used for computing
        // T - SIGMA I = L D L^T
        // Define the increment TAU of the shift in case the initial shift
        // needs to be refined to obtain a factorization with not too much
        // element growth.
        let mut tau = fma(spdiam * eps, n as f64, 2.0 * pivmin);
        tau = tau.max(2.0 * eps * sigma.abs());

        let mut found = false;
        for idum in 1..=MAXTRY {
            // Compute L D L^T factorization of tridiagonal matrix T - sigma I.
            // Store D in WORK(1:IN), L in WORK(IN+1:2*IN), and reciprocals of
            // pivots in WORK(2*IN+1:3*IN)
            let mut dpivot = v!(d, ibegin) - sigma;
            v!(work, 1) = dpivot;
            let mut dmax = v!(work, 1).abs();
            let mut j = ibegin;
            for i in 1..=(in_ - 1) {
                v!(work, 2 * in_ + i) = 1.0 / v!(work, i);
                let tmp = v!(e, j) * v!(work, 2 * in_ + i);
                v!(work, in_ + i) = tmp;
                dpivot = fma(-v!(e, j), tmp, v!(d, j + 1) - sigma);
                v!(work, i + 1) = dpivot;
                dmax = dmax.max(dpivot.abs());
                j += 1;
            }
            // check for element growth
            let mut norep = dmax > MAXGROWTH * spdiam;
            if usedqd && !norep {
                // Ensure the definiteness of the representation
                // All entries of D (of L D L^T) must have the same sign
                for i in 1..=in_ {
                    let tmp = sgndef * v!(work, i);
                    if tmp < 0.0 {
                        norep = true;
                    }
                }
            }
            if norep {
                // Note that in the case of IRANGE=ALLRNG, we use the
                // Gerschgorin shift which makes the matrix definite. So we
                // should end up here really only in the case of
                // IRANGE = VALRNG or INDRNG.
                if idum == MAXTRY - 1 {
                    if sgndef == 1.0 {
                        // The fudged Gerschgorin shift should succeed
                        sigma = gl - FUDGE * spdiam * eps * n as f64 - FUDGE * 2.0 * pivmin;
                    } else {
                        sigma = gu + FUDGE * spdiam * eps * n as f64 + FUDGE * 2.0 * pivmin;
                    }
                } else {
                    sigma = fma(-sgndef, tau, sigma);
                    tau *= 2.0;
                }
            } else {
                // an initial RRR is found
                found = true;
                break;
            }
        }
        if !found {
            // if the program reaches this point, no base representation
            // could be found in MAXTRY iterations.
            out.info = 2;
            return out;
        }

        // At this point, we have found an initial base representation
        // T - SIGMA I = L D L^T with not too much element growth.
        // Store the shift.
        v!(e, iend) = sigma;
        // Store D and L.
        dcopy(in_, work, 1, pv!(d, ibegin), 1);
        dcopy(in_ - 1, pv!(work, in_ + 1), 1, pv!(e, ibegin), 1);

        if mb > 1 {
            // Perturb each entry of the base representation by a small
            // multiple of its value to break ties. The constant PERT is
            // chosen so that the perturbation is a few ulps.
            let mut iseed = [1i32; 4];
            dlarnv2(&mut iseed, 2 * in_ - 1, work);
            for i in 1..=(in_ - 1) {
                v!(d, ibegin + i - 1) *= fma(eps * PERT, v!(work, i), 1.0);
                v!(e, ibegin + i - 1) *= fma(eps * PERT, v!(work, in_ + i), 1.0);
            }
            v!(d, iend) *= fma(eps * 4.0, v!(work, in_), 1.0);
        }

        // Compute the required eigenvalues of L D L' by dqds: the full
        // spectrum of the block.
        let rtol = (in_ as f64).ln() * 4.0 * eps;
        let mut j = ibegin;
        for i in 1..=(in_ - 1) {
            v!(work, 2 * i - 1) = v!(d, j).abs();
            v!(work, 2 * i) = v!(e, j) * v!(e, j) * v!(work, 2 * i - 1);
            j += 1;
        }
        v!(work, 2 * in_ - 1) = v!(d, iend).abs();
        v!(work, 2 * in_) = 0.0;
        let iinfo = dlasq2(in_, work);
        if iinfo != 0 {
            // If IINFO = -5 then an index is part of a tight cluster and
            // should be changed. The index is in IWORK(1) and the gap is
            // in WORK(N+1)
            out.info = -5;
            return out;
        } else {
            // Test that all eigenvalues are positive as expected
            for i in 1..=in_ {
                if v!(work, i) < 0.0 {
                    out.info = -6;
                    return out;
                }
            }
        }
        if sgndef > 0.0 {
            for i in indl..=indu {
                m += 1;
                v!(w, m) = v!(work, in_ - i + 1);
                v!(iblock, m) = jblk;
                v!(indexw, m) = i;
            }
        } else {
            for i in indl..=indu {
                m += 1;
                v!(w, m) = -v!(work, i);
                v!(iblock, m) = jblk;
                v!(indexw, m) = i;
            }
        }

        for i in (m - mb + 1)..=m {
            // the value of RTOL below should be the tolerance in DLASQ2
            v!(werr, i) = rtol * v!(w, i).abs();
        }
        for i in (m - mb + 1)..=(m - 1) {
            // compute the right gap between the intervals
            v!(wgap, i) = 0.0f64.max(v!(w, i + 1) - v!(werr, i + 1) - (v!(w, i) + v!(werr, i)));
        }
        v!(wgap, m) = 0.0f64.max((vu - sigma) - (v!(w, m) + v!(werr, m)));

        ibegin = iend + 1;
        wbegin = wend + 1;
    }
    out.m = m;
    out
}

/// `DLARRJ`: refines the eigenvalue approximations `w(ifirst..ilast)` of
/// the tridiagonal (`d`, squared off-diagonals `e2`) by bisection until
/// their intervals are below `rtol` relative width.
#[allow(clippy::too_many_arguments)]
unsafe fn dlarrj(
    n: i32,
    d: *const f64,
    e2: *const f64,
    ifirst: i32,
    ilast: i32,
    rtol: f64,
    offset: i32,
    w: *mut f64,
    werr: *mut f64,
    work: *mut f64,
    iwork: *mut i32,
    pivmin: f64,
    spdiam: f64,
) {
    if n <= 0 {
        return;
    }
    let maxitr = (((spdiam + pivmin).ln() - pivmin.ln()) / 2.0f64.ln()) as i32 + 2;

    // Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
    // The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
    // Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer
    // IWORK( 2*I-1 ) for an unconverged interval is set to the index of
    // the next unconverged interval, and is -1 or 0 for a converged
    // interval. Thus a linked list of unconverged intervals is set up.
    let mut i1 = ifirst;
    let i2 = ilast;
    // The number of unconverged intervals
    let mut nint = 0;
    // The last unconverged interval found
    let mut prev = 0;
    for i in i1..=i2 {
        let k = 2 * i;
        let ii = i - offset;
        let mut left = v!(w, ii) - v!(werr, ii);
        let mid = v!(w, ii);
        let mut right = v!(w, ii) + v!(werr, ii);
        let width = right - mid;
        let tmp = left.abs().max(right.abs());

        // The following test prevents the test of converged intervals
        if width < rtol * tmp {
            // This interval has already converged and does not need
            // refinement. (Note that the gaps might change through
            // refining the eigenvalues, however, they can only get bigger.)
            // Remove it from the list.
            v!(iwork, k - 1) = -1;
            // Make sure that I1 always points to the first unconverged
            // interval
            if i == i1 && i < i2 {
                i1 = i + 1;
            }
            if prev >= i1 && i <= i2 {
                v!(iwork, 2 * prev - 1) = i + 1;
            }
        } else {
            // unconverged interval found
            prev = i;
            // Make sure that [LEFT,RIGHT] contains the desired eigenvalue
            //
            // Do while( CNT(LEFT).GT.I-1 )
            let mut fac = 1.0;
            let mut cnt;
            loop {
                cnt = 0;
                let s = left;
                let mut dplus = v!(d, 1) - s;
                if dplus < 0.0 {
                    cnt += 1;
                }
                for j in 2..=n {
                    dplus = v!(d, j) - s - v!(e2, j - 1) / dplus;
                    if dplus < 0.0 {
                        cnt += 1;
                    }
                }
                if cnt > i - 1 {
                    left = fma(-v!(werr, ii), fac, left);
                    fac *= 2.0;
                } else {
                    break;
                }
            }

            // Do while( CNT(RIGHT).LT.I )
            fac = 1.0;
            loop {
                cnt = 0;
                let s = right;
                let mut dplus = v!(d, 1) - s;
                if dplus < 0.0 {
                    cnt += 1;
                }
                for j in 2..=n {
                    dplus = v!(d, j) - s - v!(e2, j - 1) / dplus;
                    if dplus < 0.0 {
                        cnt += 1;
                    }
                }
                if cnt < i {
                    right = fma(v!(werr, ii), fac, right);
                    fac *= 2.0;
                } else {
                    break;
                }
            }
            nint += 1;
            v!(iwork, k - 1) = i + 1;
            v!(iwork, k) = cnt;
        }
        v!(work, k - 1) = left;
        v!(work, k) = right;
    }

    let savi1 = i1;

    // Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
    // and while (ITER.LT.MAXITR)
    let mut iter = 0;
    loop {
        let mut prev = i1 - 1;
        let mut i = i1;
        let olnint = nint;

        for _p in 1..=olnint {
            let k = 2 * i;
            let next = v!(iwork, k - 1);
            let left = v!(work, k - 1);
            let right = v!(work, k);
            let mid = 0.5 * (left + right);

            // semiwidth of interval
            let width = right - mid;
            let tmp = left.abs().max(right.abs());

            if width < rtol * tmp || iter == maxitr {
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
            let mut cnt = 0;
            let s = mid;
            let mut dplus = v!(d, 1) - s;
            if dplus < 0.0 {
                cnt += 1;
            }
            for j in 2..=n {
                dplus = v!(d, j) - s - v!(e2, j - 1) / dplus;
                if dplus < 0.0 {
                    cnt += 1;
                }
            }
            if cnt <= i - 1 {
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
    for i in savi1..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        // All intervals marked by '0' have been refined.
        if v!(iwork, k - 1) == 0 {
            v!(w, ii) = 0.5 * (v!(work, k - 1) + v!(work, k));
            v!(werr, ii) = v!(work, k) - v!(w, ii);
        }
    }
}

/// `DSTEMR` with `RANGE = 'A'`: all eigenvalues of the tridiagonal
/// `(d, e)` into `w` (ascending) and, when `wantz`, the eigenvectors into
/// `z` with their supports in `isuppz`. `e` holds `n` entries (the last is
/// workspace), `work` `18 n` doubles and `iwork` `10 n` ints. `tryrac` is
/// updated as LAPACK updates it. Returns `(m, INFO)`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dstemr_all(
    wantz: bool,
    n: i32,
    d: *mut f64,
    e: *mut f64,
    w: *mut f64,
    z: *mut f64,
    ldz: i32,
    isuppz: *mut i32,
    tryrac: &mut bool,
    work: *mut f64,
    iwork: *mut i32,
) -> (i32, i32) {
    const MINRGP: f64 = 1.0e-3;
    let mut m = 0;
    if n == 0 {
        return (0, 0);
    }

    let safmin = SAFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt().min(1.0 / safmin.sqrt().sqrt());

    if n == 1 {
        m = 1;
        v!(w, 1) = v!(d, 1);
        if wantz {
            m!(z, ldz, 1, 1) = 1.0;
            v!(isuppz, 1) = 1;
            v!(isuppz, 2) = 1;
        }
        return (m, 0);
    }

    let mut nsplit = 0;
    if n == 2 {
        let (mut r1, mut r2, cs, sn) = if !wantz {
            let (a, b) = dlae2(v!(d, 1), v!(e, 1), v!(d, 2));
            (a, b, 0.0, 0.0)
        } else {
            dlaev2(v!(d, 1), v!(e, 1), v!(d, 2))
        };
        let mut laeswap = false;
        // D/S/LAE2 and D/S/LAEV2 outputs satisfy |R1| >= |R2|. However,
        // the following code requires R1 >= R2. Hence, we correct
        // the order of R1, R2, CS, SN if R1 < R2 before further processing
        if r1 < r2 {
            v!(e, 2) = r1;
            r1 = r2;
            r2 = v!(e, 2);
            laeswap = true;
        }
        m += 1;
        v!(w, m) = r2;
        if wantz {
            if laeswap {
                m!(z, ldz, 1, m) = cs;
                m!(z, ldz, 2, m) = sn;
            } else {
                m!(z, ldz, 1, m) = -sn;
                m!(z, ldz, 2, m) = cs;
            }
            // Note: At most one of SN and CS can be zero.
            if sn != 0.0 {
                if cs != 0.0 {
                    v!(isuppz, 2 * m - 1) = 1;
                    v!(isuppz, 2 * m) = 2;
                } else {
                    v!(isuppz, 2 * m - 1) = 1;
                    v!(isuppz, 2 * m) = 1;
                }
            } else {
                v!(isuppz, 2 * m - 1) = 2;
                v!(isuppz, 2 * m) = 2;
            }
        }
        m += 1;
        v!(w, m) = r1;
        if wantz {
            if laeswap {
                m!(z, ldz, 1, m) = -sn;
                m!(z, ldz, 2, m) = cs;
            } else {
                m!(z, ldz, 1, m) = cs;
                m!(z, ldz, 2, m) = sn;
            }
            // Note: At most one of SN and CS can be zero.
            if sn != 0.0 {
                if cs != 0.0 {
                    v!(isuppz, 2 * m - 1) = 1;
                    v!(isuppz, 2 * m) = 2;
                } else {
                    v!(isuppz, 2 * m - 1) = 1;
                    v!(isuppz, 2 * m) = 1;
                }
            } else {
                v!(isuppz, 2 * m - 1) = 2;
                v!(isuppz, 2 * m) = 2;
            }
        }
    } else {
        // Continue with general N
        let indgrs = 1;
        let inderr = 2 * n + 1;
        let indgp = 3 * n + 1;
        let indd = 4 * n + 1;
        let inde2 = 5 * n + 1;
        let indwrk = 6 * n + 1;

        let iinspl = 1;
        let iindbl = n + 1;
        let iindw = 2 * n + 1;
        let iindwk = 3 * n + 1;

        // Scale matrix to allowable range, if necessary.
        // The allowable range is related to the PIVMIN parameter; see the
        // comments in DLARRD. The preference for scaling small values
        // up is heuristic; we expect users' matrices not to be close to the
        // RMAX threshold.
        let mut scale = 1.0;
        let mut tnrm = dlanst_m(n, d, e);
        if tnrm > 0.0 && tnrm < rmin {
            scale = rmin / tnrm;
        } else if tnrm > rmax {
            scale = rmax / tnrm;
        }
        if scale != 1.0 {
            dscal(n, scale, d, 1);
            dscal(n - 1, scale, e, 1);
            tnrm *= scale;
        }

        // Compute the desired eigenvalues of the tridiagonal after
        // splitting into smaller subblocks if the corresponding
        // off-diagonal elements are small
        // THRESH is the splitting parameter for DLARRE
        // A negative THRESH forces the old splitting criterion based on
        // the size of the off-diagonal. A positive THRESH switches to
        // splitting which preserves relative accuracy.
        let iinfo = if *tryrac {
            // Test whether the matrix warrants the more expensive relative
            // approach.
            dlarrr(n, d, e)
        } else {
            // The user does not care about relative accurately eigenvalues
            -1
        };
        // Set the splitting criterion
        let thresh;
        if iinfo == 0 {
            thresh = eps;
        } else {
            thresh = -eps;
            // relative accuracy is desired but T does not guarantee it
            *tryrac = false;
        }

        if *tryrac {
            // Copy original diagonal, needed to guarantee relative accuracy
            dcopy(n, d, 1, pv!(work, indd), 1);
        }
        // Store the squares of the offdiagonal values of T
        for j in 1..=(n - 1) {
            v!(work, inde2 + j - 1) = v!(e, j) * v!(e, j);
        }

        // Set the tolerance parameters for bisection
        let (rtol1, rtol2) = if !wantz {
            // DLARRE computes the eigenvalues to full precision.
            (4.0 * eps, 4.0 * eps)
        } else {
            // DLARRE computes the eigenvalues to less than full precision.
            // DLARRV will refine the eigenvalue approximations, and we can
            // need less accurate initial bisection in DLARRE.
            // Note: these settings do only affect the subset case and
            // DLARRE
            (eps.sqrt(), (eps.sqrt() * 5.0e-3).max(4.0 * eps))
        };
        let rr = dlarre_all(
            n,
            d,
            e,
            pv!(work, inde2),
            thresh,
            pv!(iwork, iinspl),
            w,
            pv!(work, inderr),
            pv!(work, indgp),
            pv!(iwork, iindbl),
            pv!(iwork, iindw),
            pv!(work, indgrs),
            pv!(work, indwrk),
        );
        if rr.info != 0 {
            return (rr.m, 10 + rr.info.abs());
        }
        m = rr.m;
        nsplit = rr.nsplit;
        let pivmin = rr.pivmin;
        // Note that if RANGE .NE. 'V', DLARRE computes bounds on the
        // desired part of the spectrum. All desired eigenvalues are
        // contained in (WL,WU]
        let (wl, wu) = (rr.vl, rr.vu);

        if wantz {
            // Compute the desired eigenvectors corresponding to the
            // computed eigenvalues
            let iinfo = dlarrv(
                n,
                wl,
                wu,
                d,
                e,
                pivmin,
                pv!(iwork, iinspl),
                m,
                1,
                m,
                MINRGP,
                rtol1,
                rtol2,
                w,
                pv!(work, inderr),
                pv!(work, indgp),
                pv!(iwork, iindbl),
                pv!(iwork, iindw),
                pv!(work, indgrs),
                z,
                ldz,
                isuppz,
                pv!(work, indwrk),
                pv!(iwork, iindwk),
            );
            if iinfo != 0 {
                return (m, 20 + iinfo.abs());
            }
        } else {
            // DLARRE computes eigenvalues of the (shifted) root
            // representation. DLARRV returns the eigenvalues of the
            // unshifted matrix. However, if the eigenvectors are not
            // desired by the user, we need to apply the corresponding
            // shifts from DLARRE to obtain the eigenvalues of the original
            // matrix.
            for j in 1..=m {
                let itmp = v!(iwork, iindbl + j - 1);
                v!(w, j) += v!(e, v!(iwork, iinspl + itmp - 1));
            }
        }

        if *tryrac {
            // Refine computed eigenvalues so that they are relatively
            // accurate with respect to the original matrix T.
            let mut ibegin = 1;
            let mut wbegin = 1;
            for jblk in 1..=v!(iwork, iindbl + m - 1) {
                let iend = v!(iwork, iinspl + jblk - 1);
                let in_ = iend - ibegin + 1;
                let mut wend = wbegin - 1;
                // check if any eigenvalues have to be refined in this block
                while wend < m && v!(iwork, iindbl + wend) == jblk {
                    wend += 1;
                }
                if wend < wbegin {
                    ibegin = iend + 1;
                    continue;
                }

                let offset = v!(iwork, iindw + wbegin - 1) - 1;
                let ifirst = v!(iwork, iindw + wbegin - 1);
                let ilast = v!(iwork, iindw + wend - 1);
                let rtol2 = 4.0 * eps;
                dlarrj(
                    in_,
                    pv!(work, indd + ibegin - 1),
                    pv!(work, inde2 + ibegin - 1),
                    ifirst,
                    ilast,
                    rtol2,
                    offset,
                    pv!(w, wbegin),
                    pv!(work, inderr + wbegin - 1),
                    pv!(work, indwrk),
                    pv!(iwork, iindwk),
                    pivmin,
                    tnrm,
                );
                ibegin = iend + 1;
                wbegin = wend + 1;
            }
        }

        // If matrix was scaled, then rescale eigenvalues appropriately.
        if scale != 1.0 {
            dscal(m, 1.0 / scale, w, 1);
        }
    }

    // If eigenvalues are not in increasing order, then sort them,
    // possibly along with eigenvectors.
    if nsplit > 1 || n == 2 {
        if !wantz {
            dlasrt(b'I', m, w);
        } else {
            for j in 1..=(m - 1) {
                let mut i = 0;
                let mut tmp = v!(w, j);
                for jj in (j + 1)..=m {
                    if v!(w, jj) < tmp {
                        i = jj;
                        tmp = v!(w, jj);
                    }
                }
                if i != 0 {
                    v!(w, i) = v!(w, j);
                    v!(w, j) = tmp;
                    dswap(n, pm!(z, ldz, 1, i), 1, pm!(z, ldz, 1, j), 1);
                    let itmp = v!(isuppz, 2 * i - 1);
                    v!(isuppz, 2 * i - 1) = v!(isuppz, 2 * j - 1);
                    v!(isuppz, 2 * j - 1) = itmp;
                    let itmp = v!(isuppz, 2 * i);
                    v!(isuppz, 2 * i) = v!(isuppz, 2 * j);
                    v!(isuppz, 2 * j) = itmp;
                }
            }
        }
    }
    (m, 0)
}
