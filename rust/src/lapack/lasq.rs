//! `DLASQ2`-`DLASQ6`: all eigenvalues of the positive definite `L D L'`
//! (in qd form) by the dqds algorithm with shifts, as `DLARRE` uses it.

use super::auxiliary::dlasrt;
use super::{disnan, fma, PREC, SAFMIN};

/// The state `DLASQ3` threads through its calls (its in/out arguments).
struct Dqds {
    i0: i32,
    n0: i32,
    pp: i32,
    dmin: f64,
    sigma: f64,
    desig: f64,
    qmax: f64,
    nfail: i32,
    iter: i32,
    ndiv: i32,
    ieee: bool,
    ttype: i32,
    dmin1: f64,
    dmin2: f64,
    dn: f64,
    dn1: f64,
    dn2: f64,
    g: f64,
    tau: f64,
}

/// `DLASQ2`: the eigenvalues of the qd array `z` (`4 n` entries: `q` and
/// `e` interleaved as `DLARRE` builds them), returned in decreasing order
/// in `z(1..n)`. Returns `INFO`.
pub(crate) unsafe fn dlasq2(n: i32, z: *mut f64) -> i32 {
    const CBIAS: f64 = 1.50;
    const HUNDRD: f64 = 100.0;

    let eps = PREC;
    let safmin = SAFMIN;
    let tol = eps * HUNDRD;
    let tol2 = tol * tol;

    if n < 0 {
        return -1;
    } else if n == 0 {
        return 0;
    } else if n == 1 {
        // 1-by-1 case.
        if v!(z, 1) < 0.0 {
            return -201;
        }
        return 0;
    } else if n == 2 {
        // 2-by-2 case.
        if v!(z, 1) < 0.0 {
            return -201;
        } else if v!(z, 2) < 0.0 {
            return -202;
        } else if v!(z, 3) < 0.0 {
            return -203;
        } else if v!(z, 3) > v!(z, 1) {
            let d = v!(z, 3);
            v!(z, 3) = v!(z, 1);
            v!(z, 1) = d;
        }
        v!(z, 5) = v!(z, 1) + v!(z, 2) + v!(z, 3);
        if v!(z, 2) > v!(z, 3) * tol2 {
            let t = 0.5 * ((v!(z, 1) - v!(z, 3)) + v!(z, 2));
            let mut s = v!(z, 3) * (v!(z, 2) / t);
            if s <= t {
                s = v!(z, 3) * (v!(z, 2) / (t * (1.0 + (1.0 + s / t).sqrt())));
            } else {
                s = v!(z, 3) * (v!(z, 2) / fma(t.sqrt(), (t + s).sqrt(), t));
            }
            let t = v!(z, 1) + (s + v!(z, 2));
            v!(z, 3) *= v!(z, 1) / t;
            v!(z, 1) = t;
        }
        v!(z, 2) = v!(z, 3);
        v!(z, 6) = v!(z, 2) + v!(z, 1);
        return 0;
    }

    // Check for negative data and compute sums of q's and e's.
    v!(z, 2 * n) = 0.0;
    let mut emin = v!(z, 2);
    let mut qmax = 0.0f64;
    let mut d = 0.0;
    let mut e = 0.0;

    let mut k = 1;
    while k <= 2 * (n - 1) {
        if v!(z, k) < 0.0 {
            return -(200 + k);
        } else if v!(z, k + 1) < 0.0 {
            return -(200 + k + 1);
        }
        d += v!(z, k);
        e += v!(z, k + 1);
        qmax = qmax.max(v!(z, k));
        emin = emin.min(v!(z, k + 1));
        k += 2;
    }
    if v!(z, 2 * n - 1) < 0.0 {
        return -(200 + 2 * n - 1);
    }
    d += v!(z, 2 * n - 1);
    qmax = qmax.max(v!(z, 2 * n - 1));

    // Check for diagonality.
    if e == 0.0 {
        for k in 2..=n {
            v!(z, k) = v!(z, 2 * k - 1);
        }
        dlasrt(b'D', n, z);
        v!(z, 2 * n - 1) = d;
        return 0;
    }

    let trace = d + e;

    // Check for zero data.
    if trace == 0.0 {
        v!(z, 2 * n - 1) = 0.0;
        return 0;
    }

    // Check whether the machine is IEEE conformable.
    let ieee = true;

    // Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
    let mut k = 2 * n;
    while k >= 2 {
        v!(z, 2 * k) = 0.0;
        v!(z, 2 * k - 1) = v!(z, k);
        v!(z, 2 * k - 2) = 0.0;
        v!(z, 2 * k - 3) = v!(z, k - 1);
        k -= 2;
    }

    let i0 = 1;
    let n0 = n;

    // Reverse the qd-array, if warranted.
    if CBIAS * v!(z, 4 * i0 - 3) < v!(z, 4 * n0 - 3) {
        let ipn4 = 4 * (i0 + n0);
        let mut i4 = 4 * i0;
        while i4 <= 2 * (i0 + n0 - 1) {
            let temp = v!(z, i4 - 3);
            v!(z, i4 - 3) = v!(z, ipn4 - i4 - 3);
            v!(z, ipn4 - i4 - 3) = temp;
            let temp = v!(z, i4 - 1);
            v!(z, i4 - 1) = v!(z, ipn4 - i4 - 5);
            v!(z, ipn4 - i4 - 5) = temp;
            i4 += 4;
        }
    }

    // Initial split checking via dqd and Li's test.
    let mut pp = 0;

    for _k in 1..=2 {
        let mut d = v!(z, 4 * n0 + pp - 3);
        let mut i4 = 4 * (n0 - 1) + pp;
        while i4 >= 4 * i0 + pp {
            if v!(z, i4 - 1) <= tol2 * d {
                v!(z, i4 - 1) = -0.0;
                d = v!(z, i4 - 3);
            } else {
                d = v!(z, i4 - 3) * (d / (d + v!(z, i4 - 1)));
            }
            i4 -= 4;
        }

        // dqd maps Z to ZZ plus Li's test.
        emin = v!(z, 4 * i0 + pp + 1);
        d = v!(z, 4 * i0 + pp - 3);
        let mut i4 = 4 * i0 + pp;
        while i4 <= 4 * (n0 - 1) + pp {
            v!(z, i4 - 2 * pp - 2) = d + v!(z, i4 - 1);
            if v!(z, i4 - 1) <= tol2 * d {
                v!(z, i4 - 1) = -0.0;
                v!(z, i4 - 2 * pp - 2) = d;
                v!(z, i4 - 2 * pp) = 0.0;
                d = v!(z, i4 + 1);
            } else if safmin * v!(z, i4 + 1) < v!(z, i4 - 2 * pp - 2)
                && safmin * v!(z, i4 - 2 * pp - 2) < v!(z, i4 + 1)
            {
                let temp = v!(z, i4 + 1) / v!(z, i4 - 2 * pp - 2);
                v!(z, i4 - 2 * pp) = v!(z, i4 - 1) * temp;
                d *= temp;
            } else {
                v!(z, i4 - 2 * pp) = v!(z, i4 + 1) * (v!(z, i4 - 1) / v!(z, i4 - 2 * pp - 2));
                d = v!(z, i4 + 1) * (d / v!(z, i4 - 2 * pp - 2));
            }
            emin = emin.min(v!(z, i4 - 2 * pp));
            i4 += 4;
        }
        v!(z, 4 * n0 - pp - 2) = d;

        // Now find qmax.
        qmax = v!(z, 4 * i0 - pp - 2);
        let mut i4 = 4 * i0 - pp + 2;
        while i4 <= 4 * n0 - pp - 2 {
            qmax = qmax.max(v!(z, i4));
            i4 += 4;
        }

        // Prepare for the next iteration on K.
        pp = 1 - pp;
    }

    // Initialise variables to pass to DLASQ3.
    let mut st = Dqds {
        i0,
        n0,
        pp,
        dmin: 0.0,
        sigma: 0.0,
        desig: 0.0,
        qmax,
        nfail: 0,
        iter: 2,
        ndiv: 2 * (n0 - i0),
        ieee,
        ttype: 0,
        dmin1: 0.0,
        dmin2: 0.0,
        dn: 0.0,
        dn1: 0.0,
        dn2: 0.0,
        g: 0.0,
        tau: 0.0,
    };

    let mut converged = false;
    for _iwhila in 1..=(n + 1) {
        if st.n0 < 1 {
            converged = true;
            break;
        }

        // While array unfinished do
        //
        // E(N0) holds the value of SIGMA when submatrix in I0:N0
        // splits from the rest of the array, but is negated.
        st.desig = 0.0;
        if st.n0 == n {
            st.sigma = 0.0;
        } else {
            st.sigma = -v!(z, 4 * st.n0 - 1);
        }
        if st.sigma < 0.0 {
            return 1;
        }

        // Find last unreduced submatrix's top index I0, find QMAX and
        // EMIN. Find Gershgorin-type bound if Q's much greater than E's.
        let mut emax = 0.0f64;
        if st.n0 > st.i0 {
            emin = v!(z, 4 * st.n0 - 5).abs();
        } else {
            emin = 0.0;
        }
        let mut qmin = v!(z, 4 * st.n0 - 3);
        st.qmax = qmin;
        let mut i4 = 4 * st.n0;
        let mut split = false;
        while i4 >= 8 {
            if v!(z, i4 - 5) <= 0.0 {
                split = true;
                break;
            }
            if qmin >= 4.0 * emax {
                qmin = qmin.min(v!(z, i4 - 3));
                emax = emax.max(v!(z, i4 - 5));
            }
            st.qmax = st.qmax.max(v!(z, i4 - 7) + v!(z, i4 - 5));
            emin = emin.min(v!(z, i4 - 5));
            i4 -= 4;
        }
        if !split {
            i4 = 4;
        }

        // 100
        st.i0 = i4 / 4;
        st.pp = 0;

        if st.n0 - st.i0 > 1 {
            let mut dee = v!(z, 4 * st.i0 - 3);
            let mut deemin = dee;
            let mut kmin = st.i0;
            let mut i4 = 4 * st.i0 + 1;
            while i4 <= 4 * st.n0 - 3 {
                dee = v!(z, i4) * (dee / (dee + v!(z, i4 - 2)));
                if dee <= deemin {
                    deemin = dee;
                    kmin = (i4 + 3) / 4;
                }
                i4 += 4;
            }
            if (kmin - st.i0) * 2 < st.n0 - kmin && deemin <= 0.5 * v!(z, 4 * st.n0 - 3) {
                let ipn4 = 4 * (st.i0 + st.n0);
                st.pp = 2;
                let mut i4 = 4 * st.i0;
                while i4 <= 2 * (st.i0 + st.n0 - 1) {
                    let temp = v!(z, i4 - 3);
                    v!(z, i4 - 3) = v!(z, ipn4 - i4 - 3);
                    v!(z, ipn4 - i4 - 3) = temp;
                    let temp = v!(z, i4 - 2);
                    v!(z, i4 - 2) = v!(z, ipn4 - i4 - 2);
                    v!(z, ipn4 - i4 - 2) = temp;
                    let temp = v!(z, i4 - 1);
                    v!(z, i4 - 1) = v!(z, ipn4 - i4 - 5);
                    v!(z, ipn4 - i4 - 5) = temp;
                    let temp = v!(z, i4);
                    v!(z, i4) = v!(z, ipn4 - i4 - 4);
                    v!(z, ipn4 - i4 - 4) = temp;
                    i4 += 4;
                }
            }
        }

        // Put -(initial shift) into DMIN.
        st.dmin = -(0.0f64.max(fma(-(2.0 * qmin.sqrt()), emax.sqrt(), qmin)));

        // Now I0:N0 is unreduced.
        // PP = 0 for ping, PP = 1 for pong.
        // PP = 2 indicates that flipping was applied to the Z array and
        //        and that the tests for deflation upon entry in DLASQ3
        //        should not be performed.
        let nbig = 100 * (st.n0 - st.i0 + 1);
        let mut inner_done = false;
        for _iwhilb in 1..=nbig {
            if st.i0 > st.n0 {
                inner_done = true;
                break;
            }

            // While submatrix unfinished take a good dqds step.
            dlasq3(&mut st, z);

            st.pp = 1 - st.pp;

            // When EMIN is very small check for splits.
            if st.pp == 0 && st.n0 - st.i0 >= 3 {
                if v!(z, 4 * st.n0) <= tol2 * st.qmax || v!(z, 4 * st.n0 - 1) <= tol2 * st.sigma {
                    let mut splt = st.i0 - 1;
                    st.qmax = v!(z, 4 * st.i0 - 3);
                    emin = v!(z, 4 * st.i0 - 1);
                    let mut oldemn = v!(z, 4 * st.i0);
                    let mut i4 = 4 * st.i0;
                    while i4 <= 4 * (st.n0 - 3) {
                        if v!(z, i4) <= tol2 * v!(z, i4 - 3) || v!(z, i4 - 1) <= tol2 * st.sigma {
                            v!(z, i4 - 1) = -st.sigma;
                            splt = i4 / 4;
                            st.qmax = 0.0;
                            emin = v!(z, i4 + 3);
                            oldemn = v!(z, i4 + 4);
                        } else {
                            st.qmax = st.qmax.max(v!(z, i4 + 1));
                            emin = emin.min(v!(z, i4 - 1));
                            oldemn = oldemn.min(v!(z, i4));
                        }
                        i4 += 4;
                    }
                    v!(z, 4 * st.n0 - 1) = emin;
                    v!(z, 4 * st.n0) = oldemn;
                    st.i0 = splt + 1;
                }
            }
        }

        if !inner_done {
            // Maximum number of iterations exceeded, restore the shift
            // SIGMA and place the new d's and e's in a qd array.
            // This might need to be done for several blocks
            let mut i1 = st.i0;
            let mut n1;
            loop {
                // 145
                let mut tempq = v!(z, 4 * st.i0 - 3);
                v!(z, 4 * st.i0 - 3) += st.sigma;
                for k in (st.i0 + 1)..=st.n0 {
                    let tempe = v!(z, 4 * k - 5);
                    v!(z, 4 * k - 5) *= tempq / v!(z, 4 * k - 7);
                    tempq = v!(z, 4 * k - 3);
                    v!(z, 4 * k - 3) = v!(z, 4 * k - 3) + st.sigma + tempe - v!(z, 4 * k - 5);
                }

                // Prepare to do this on the previous block if there is one
                if i1 > 1 {
                    n1 = i1 - 1;
                    while i1 >= 2 && v!(z, 4 * i1 - 5) >= 0.0 {
                        i1 -= 1;
                    }
                    st.sigma = -v!(z, 4 * n1 - 1);
                    continue;
                }
                break;
            }

            for k in 1..=n {
                v!(z, 2 * k - 1) = v!(z, 4 * k - 3);
                // Only the block 1..N0 is unfinished. The rest of the e's
                // must be essentially zero, although sometimes other data
                // has been stored in them.
                if k < st.n0 {
                    v!(z, 2 * k) = v!(z, 4 * k - 1);
                } else {
                    v!(z, 2 * k) = 0.0;
                }
            }
            return 2;
        }
        // end IWHILB
    }

    if !converged {
        return 3;
    }

    // Move q's to the front.
    for k in 2..=n {
        v!(z, k) = v!(z, 4 * k - 3);
    }

    // Sort and compute sum of eigenvalues.
    dlasrt(b'D', n, z);

    let mut e = 0.0;
    let mut k = n;
    while k >= 1 {
        e += v!(z, k);
        k -= 1;
    }

    // Store trace, sum(eigenvalues) and information on performance.
    v!(z, 2 * n + 1) = trace;
    v!(z, 2 * n + 2) = e;
    v!(z, 2 * n + 3) = st.iter as f64;
    v!(z, 2 * n + 4) = st.ndiv as f64 / (n * n) as f64;
    v!(z, 2 * n + 5) = HUNDRD * st.nfail as f64 / st.iter as f64;
    0
}

/// `DLASQ3`: checks for deflation, computes a shift (`DLASQ4`) and
/// applies one dqds transform (`DLASQ5`, or `DLASQ6` without shift on
/// failure).
unsafe fn dlasq3(st: &mut Dqds, z: *mut f64) {
    const CBIAS: f64 = 1.50;
    const QURTR: f64 = 0.250;
    const HUNDRD: f64 = 100.0;

    let n0in = st.n0;
    let eps = PREC;
    let tol = eps * HUNDRD;
    let tol2 = tol * tol;

    // Check for deflation.
    loop {
        // 10
        if st.n0 < st.i0 {
            return;
        }
        let nn = 4 * st.n0 + st.pp;
        let mut two_deflate = false;
        if st.n0 != st.i0 {
            if st.n0 == st.i0 + 1 {
                two_deflate = true;
            } else {
                // Check whether E(N0-1) is negligible, 1 eigenvalue.
                if v!(z, nn - 5) > tol2 * (st.sigma + v!(z, nn - 3))
                    && v!(z, nn - 2 * st.pp - 4) > tol2 * v!(z, nn - 7)
                {
                    // 30: Check whether E(N0-2) is negligible, 2 eigenvalues.
                    if v!(z, nn - 9) > tol2 * st.sigma
                        && v!(z, nn - 2 * st.pp - 8) > tol2 * v!(z, nn - 11)
                    {
                        break;
                    }
                    two_deflate = true;
                }
            }
        }
        if !two_deflate {
            // 20
            v!(z, 4 * st.n0 - 3) = v!(z, 4 * st.n0 + st.pp - 3) + st.sigma;
            st.n0 -= 1;
            continue;
        }
        // 40
        if v!(z, nn - 3) > v!(z, nn - 7) {
            let s = v!(z, nn - 3);
            v!(z, nn - 3) = v!(z, nn - 7);
            v!(z, nn - 7) = s;
        }
        let t = 0.5 * ((v!(z, nn - 7) - v!(z, nn - 3)) + v!(z, nn - 5));
        if v!(z, nn - 5) > v!(z, nn - 3) * tol2 && t != 0.0 {
            let mut s = v!(z, nn - 3) * (v!(z, nn - 5) / t);
            if s <= t {
                s = v!(z, nn - 3) * (v!(z, nn - 5) / (t * (1.0 + (1.0 + s / t).sqrt())));
            } else {
                s = v!(z, nn - 3) * (v!(z, nn - 5) / fma(t.sqrt(), (t + s).sqrt(), t));
            }
            let t = v!(z, nn - 7) + (s + v!(z, nn - 5));
            v!(z, nn - 3) *= v!(z, nn - 7) / t;
            v!(z, nn - 7) = t;
        }
        v!(z, 4 * st.n0 - 7) = v!(z, nn - 7) + st.sigma;
        v!(z, 4 * st.n0 - 3) = v!(z, nn - 3) + st.sigma;
        st.n0 -= 2;
    }

    // 50
    if st.pp == 2 {
        st.pp = 0;
    }

    // Reverse the qd-array, if warranted.
    if (st.dmin <= 0.0 || st.n0 < n0in)
        && CBIAS * v!(z, 4 * st.i0 + st.pp - 3) < v!(z, 4 * st.n0 + st.pp - 3)
    {
        let ipn4 = 4 * (st.i0 + st.n0);
        let mut j4 = 4 * st.i0;
        while j4 <= 2 * (st.i0 + st.n0 - 1) {
            let temp = v!(z, j4 - 3);
            v!(z, j4 - 3) = v!(z, ipn4 - j4 - 3);
            v!(z, ipn4 - j4 - 3) = temp;
            let temp = v!(z, j4 - 2);
            v!(z, j4 - 2) = v!(z, ipn4 - j4 - 2);
            v!(z, ipn4 - j4 - 2) = temp;
            let temp = v!(z, j4 - 1);
            v!(z, j4 - 1) = v!(z, ipn4 - j4 - 5);
            v!(z, ipn4 - j4 - 5) = temp;
            let temp = v!(z, j4);
            v!(z, j4) = v!(z, ipn4 - j4 - 4);
            v!(z, ipn4 - j4 - 4) = temp;
            j4 += 4;
        }
        if st.n0 - st.i0 <= 4 {
            v!(z, 4 * st.n0 + st.pp - 1) = v!(z, 4 * st.i0 + st.pp - 1);
            v!(z, 4 * st.n0 - st.pp) = v!(z, 4 * st.i0 - st.pp);
        }
        st.dmin2 = st.dmin2.min(v!(z, 4 * st.n0 + st.pp - 1));
        v!(z, 4 * st.n0 + st.pp - 1) = v!(z, 4 * st.n0 + st.pp - 1)
            .min(v!(z, 4 * st.i0 + st.pp - 1))
            .min(v!(z, 4 * st.i0 + st.pp + 3));
        v!(z, 4 * st.n0 - st.pp) = v!(z, 4 * st.n0 - st.pp)
            .min(v!(z, 4 * st.i0 - st.pp))
            .min(v!(z, 4 * st.i0 - st.pp + 4));
        st.qmax = st
            .qmax
            .max(v!(z, 4 * st.i0 + st.pp - 3))
            .max(v!(z, 4 * st.i0 + st.pp + 1));
        st.dmin = -0.0;
    }

    // Choose a shift.
    dlasq4(st, z, n0in);

    // Call dqds until DMIN > 0.
    loop {
        // 70
        dlasq5(st, z, eps);

        st.ndiv += st.n0 - st.i0 + 2;
        st.iter += 1;

        // Check status.
        if st.dmin >= 0.0 && st.dmin1 >= 0.0 {
            // Success.
            break;
        } else if st.dmin < 0.0
            && st.dmin1 > 0.0
            && v!(z, 4 * (st.n0 - 1) - st.pp) < tol * (st.sigma + st.dn1)
            && st.dn.abs() < tol * st.sigma
        {
            // Convergence hidden by negative DN.
            v!(z, 4 * (st.n0 - 1) - st.pp + 2) = 0.0;
            st.dmin = 0.0;
            break;
        } else if st.dmin < 0.0 {
            // TAU too big. Select new TAU and try again.
            st.nfail += 1;
            if st.ttype < -22 {
                // Failed twice. Play it safe.
                st.tau = 0.0;
            } else if st.dmin1 > 0.0 {
                // Late failure. Gives excellent shift.
                st.tau = (st.tau + st.dmin) * fma(-eps, 2.0, 1.0);
                st.ttype -= 11;
            } else {
                // Early failure. Divide by 4.
                st.tau *= QURTR;
                st.ttype -= 12;
            }
            continue;
        } else if disnan(st.dmin) {
            // NaN.
            if st.tau == 0.0 {
                dlasq6_risky(st, z);
                break;
            } else {
                st.tau = 0.0;
                continue;
            }
        } else {
            // Possible underflow. Play it safe.
            dlasq6_risky(st, z);
            break;
        }
    }

    // 90: Update shift
    if st.tau < st.sigma {
        st.desig += st.tau;
        let t = st.sigma + st.desig;
        st.desig -= t - st.sigma;
        st.sigma = t;
    } else {
        let t = st.sigma + st.tau;
        st.desig = st.sigma - (t - st.tau) + st.desig;
        st.sigma = t;
    }
}

/// `DLASQ3`'s label 80: a `DLASQ6` sweep without shift.
unsafe fn dlasq6_risky(st: &mut Dqds, z: *mut f64) {
    dlasq6(st, z);
    st.ndiv += st.n0 - st.i0 + 2;
    st.iter += 1;
    st.tau = 0.0;
}

/// `DLASQ4`: the shift for the next dqds step, from the recent `d` values.
unsafe fn dlasq4(st: &mut Dqds, z: *mut f64, n0in: i32) {
    const CNST1: f64 = 0.5630;
    const CNST2: f64 = 1.010;
    const CNST3: f64 = 1.050;
    const QURTR: f64 = 0.250;
    const THIRD: f64 = 0.3330;
    const HUNDRD: f64 = 100.0;

    let (i0, n0, pp) = (st.i0, st.n0, st.pp);
    let (dmin, dmin1, dmin2, dn, dn1, dn2) = (st.dmin, st.dmin1, st.dmin2, st.dn, st.dn1, st.dn2);

    // A negative DMIN forces the shift to take that absolute value
    // TTYPE records the type of shift.
    if dmin <= 0.0 {
        st.tau = -dmin;
        st.ttype = -1;
        return;
    }

    let nn = 4 * n0 + pp;
    let mut s = 0.0;
    if n0in == n0 {
        // No eigenvalues deflated.
        if dmin == dn || dmin == dn1 {
            let b1 = v!(z, nn - 3).sqrt() * v!(z, nn - 5).sqrt();
            let b2 = v!(z, nn - 7).sqrt() * v!(z, nn - 9).sqrt();
            let a2 = v!(z, nn - 7) + v!(z, nn - 5);

            // Cases 2 and 3.
            if dmin == dn && dmin1 == dn1 {
                let gap2 = fma(-dmin2, QURTR, dmin2 - a2);
                let gap1 = if gap2 > 0.0 && gap2 > b2 {
                    fma(-(b2 / gap2), b2, a2 - dn)
                } else {
                    a2 - dn - (b1 + b2)
                };
                if gap1 > 0.0 && gap1 > b1 {
                    s = fma(-(b1 / gap1), b1, dn).max(0.5 * dmin);
                    st.ttype = -2;
                } else {
                    s = 0.0;
                    if dn > b1 {
                        s = dn - b1;
                    }
                    if a2 > (b1 + b2) {
                        s = s.min(a2 - (b1 + b2));
                    }
                    s = s.max(THIRD * dmin);
                    st.ttype = -3;
                }
            } else {
                // Case 4.
                st.ttype = -4;
                s = QURTR * dmin;
                let gam;
                let mut a2;
                let mut b2;
                let np;
                if dmin == dn {
                    gam = dn;
                    a2 = 0.0;
                    if v!(z, nn - 5) > v!(z, nn - 7) {
                        return;
                    }
                    b2 = v!(z, nn - 5) / v!(z, nn - 7);
                    np = nn - 9;
                } else {
                    let npp = nn - 2 * pp;
                    gam = dn1;
                    if v!(z, npp - 4) > v!(z, npp - 2) {
                        return;
                    }
                    a2 = v!(z, npp - 4) / v!(z, npp - 2);
                    if v!(z, nn - 9) > v!(z, nn - 11) {
                        return;
                    }
                    b2 = v!(z, nn - 9) / v!(z, nn - 11);
                    np = nn - 13;
                }

                // Approximate contribution to norm squared from I < NN-1.
                a2 += b2;
                let mut i4 = np;
                while i4 >= 4 * i0 - 1 + pp {
                    if b2 == 0.0 {
                        break;
                    }
                    let b1 = b2;
                    if v!(z, i4) > v!(z, i4 - 2) {
                        return;
                    }
                    b2 *= v!(z, i4) / v!(z, i4 - 2);
                    a2 += b2;
                    if HUNDRD * b2.max(b1) < a2 || CNST1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 *= CNST3;

                // Rayleigh quotient residual bound.
                if a2 < CNST1 {
                    s = gam * (1.0 - a2.sqrt()) / (1.0 + a2);
                }
            }
        } else if dmin == dn2 {
            // Case 5.
            st.ttype = -5;
            s = QURTR * dmin;

            // Compute contribution to norm squared from I > NN-2.
            let np = nn - 2 * pp;
            let mut b1 = v!(z, np - 2);
            let mut b2 = v!(z, np - 6);
            let gam = dn2;
            if v!(z, np - 8) > b2 || v!(z, np - 4) > b1 {
                return;
            }
            let mut a2 = (v!(z, np - 8) / b2) * (1.0 + v!(z, np - 4) / b1);

            // Approximate contribution to norm squared from I < NN-2.
            if n0 - i0 > 2 {
                b2 = v!(z, nn - 13) / v!(z, nn - 15);
                a2 += b2;
                let mut i4 = nn - 17;
                while i4 >= 4 * i0 - 1 + pp {
                    if b2 == 0.0 {
                        break;
                    }
                    b1 = b2;
                    if v!(z, i4) > v!(z, i4 - 2) {
                        return;
                    }
                    b2 *= v!(z, i4) / v!(z, i4 - 2);
                    a2 += b2;
                    if HUNDRD * b2.max(b1) < a2 || CNST1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 *= CNST3;
            }

            if a2 < CNST1 {
                s = gam * (1.0 - a2.sqrt()) / (1.0 + a2);
            }
        } else {
            // Case 6, no information to guide us.
            if st.ttype == -6 {
                st.g = fma(1.0 - st.g, THIRD, st.g);
            } else if st.ttype == -18 {
                st.g = QURTR * THIRD;
            } else {
                st.g = QURTR;
            }
            s = st.g * dmin;
            st.ttype = -6;
        }
    } else if n0in == n0 + 1 {
        // One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
        if dmin1 == dn1 && dmin2 == dn2 {
            // Cases 7 and 8.
            st.ttype = -7;
            s = THIRD * dmin1;
            if v!(z, nn - 5) > v!(z, nn - 7) {
                return;
            }
            let mut b1 = v!(z, nn - 5) / v!(z, nn - 7);
            let mut b2 = b1;
            if b2 != 0.0 {
                let mut i4 = 4 * n0 - 9 + pp;
                while i4 >= 4 * i0 - 1 + pp {
                    let a2 = b1;
                    if v!(z, i4) > v!(z, i4 - 2) {
                        return;
                    }
                    b1 *= v!(z, i4) / v!(z, i4 - 2);
                    b2 += b1;
                    if HUNDRD * b1.max(a2) < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            // 60
            b2 = (CNST3 * b2).sqrt();
            let a2 = dmin1 / fma(b2, b2, 1.0);
            let gap2 = fma(dmin2, 0.5, -a2);
            if gap2 > 0.0 && gap2 > b2 * a2 {
                s = s.max(a2 * fma(-((a2 * CNST2) * (b2 / gap2)), b2, 1.0));
            } else {
                s = s.max(a2 * fma(-b2, CNST2, 1.0));
                st.ttype = -8;
            }
        } else {
            // Case 9.
            s = QURTR * dmin1;
            if dmin1 == dn1 {
                s = 0.5 * dmin1;
            }
            st.ttype = -9;
        }
    } else if n0in == n0 + 2 {
        // Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
        //
        // Cases 10 and 11.
        if dmin2 == dn2 && 2.0 * v!(z, nn - 5) < v!(z, nn - 7) {
            st.ttype = -10;
            s = THIRD * dmin2;
            if v!(z, nn - 5) > v!(z, nn - 7) {
                return;
            }
            let mut b1 = v!(z, nn - 5) / v!(z, nn - 7);
            let mut b2 = b1;
            if b2 != 0.0 {
                let mut i4 = 4 * n0 - 9 + pp;
                while i4 >= 4 * i0 - 1 + pp {
                    if v!(z, i4) > v!(z, i4 - 2) {
                        return;
                    }
                    b1 *= v!(z, i4) / v!(z, i4 - 2);
                    b2 += b1;
                    if HUNDRD * b1 < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            // 80
            b2 = (CNST3 * b2).sqrt();
            let a2 = dmin2 / fma(b2, b2, 1.0);
            let gap2 = fma(
                -v!(z, nn - 11).sqrt(),
                v!(z, nn - 9).sqrt(),
                v!(z, nn - 7) + v!(z, nn - 9),
            ) - a2;
            if gap2 > 0.0 && gap2 > b2 * a2 {
                s = s.max(a2 * fma(-((a2 * CNST2) * (b2 / gap2)), b2, 1.0));
            } else {
                s = s.max(a2 * fma(-b2, CNST2, 1.0));
            }
        } else {
            s = QURTR * dmin2;
            st.ttype = -11;
        }
    } else if n0in > n0 + 2 {
        // Case 12, more than two eigenvalues deflated. No information.
        s = 0.0;
        st.ttype = -12;
    }

    st.tau = s;
}

/// `DLASQ5`: one dqds transform with shift `tau` (ping-pong `pp`),
/// tracking the minimum `d`.
unsafe fn dlasq5(st: &mut Dqds, z: *mut f64, eps: f64) {
    let (i0, n0, pp) = (st.i0, st.n0, st.pp);
    if (n0 - i0 - 1) <= 0 {
        return;
    }
    let mut tau = st.tau;
    let sigma = st.sigma;

    let dthresh = eps * (sigma + tau);
    if tau < dthresh * 0.5 {
        tau = 0.0;
    }
    st.tau = tau;
    let mut j4;
    let mut j4p2;
    let mut emin;
    let mut d;
    if tau != 0.0 {
        j4 = 4 * i0 + pp - 3;
        emin = v!(z, j4 + 4);
        d = v!(z, j4) - tau;
        st.dmin = d;
        st.dmin1 = -v!(z, j4);

        if st.ieee {
            // Code for IEEE arithmetic.
            if pp == 0 {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 2) = d + v!(z, j4 - 1);
                    let temp = v!(z, j4 + 1) / v!(z, j4 - 2);
                    d = fma(d, temp, -tau);
                    st.dmin = st.dmin.min(d);
                    v!(z, j4) = v!(z, j4 - 1) * temp;
                    emin = v!(z, j4).min(emin);
                    j4 += 4;
                }
            } else {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 3) = d + v!(z, j4);
                    let temp = v!(z, j4 + 2) / v!(z, j4 - 3);
                    d = fma(d, temp, -tau);
                    st.dmin = st.dmin.min(d);
                    v!(z, j4 - 1) = v!(z, j4) * temp;
                    emin = v!(z, j4 - 1).min(emin);
                    j4 += 4;
                }
            }

            // Unroll last two steps.
            st.dn2 = d;
            st.dmin2 = st.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn2 + v!(z, j4p2);
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn1 = fma(v!(z, j4p2 + 2), st.dn2 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn1);

            st.dmin1 = st.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn1 + v!(z, j4p2);
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn = fma(v!(z, j4p2 + 2), st.dn1 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn);
        } else {
            // Code for non IEEE arithmetic.
            if pp == 0 {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 2) = d + v!(z, j4 - 1);
                    if d < 0.0 {
                        return;
                    }
                    v!(z, j4) = v!(z, j4 + 1) * (v!(z, j4 - 1) / v!(z, j4 - 2));
                    d = fma(v!(z, j4 + 1), d / v!(z, j4 - 2), -tau);
                    st.dmin = st.dmin.min(d);
                    emin = emin.min(v!(z, j4));
                    j4 += 4;
                }
            } else {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 3) = d + v!(z, j4);
                    if d < 0.0 {
                        return;
                    }
                    v!(z, j4 - 1) = v!(z, j4 + 2) * (v!(z, j4) / v!(z, j4 - 3));
                    d = fma(v!(z, j4 + 2), d / v!(z, j4 - 3), -tau);
                    st.dmin = st.dmin.min(d);
                    emin = emin.min(v!(z, j4 - 1));
                    j4 += 4;
                }
            }

            // Unroll last two steps.
            st.dn2 = d;
            st.dmin2 = st.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn2 + v!(z, j4p2);
            if st.dn2 < 0.0 {
                return;
            }
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn1 = fma(v!(z, j4p2 + 2), st.dn2 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn1);

            st.dmin1 = st.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn1 + v!(z, j4p2);
            if st.dn1 < 0.0 {
                return;
            }
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn = fma(v!(z, j4p2 + 2), st.dn1 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn);
        }
    } else {
        // This is the version that sets d's to zero if they are small
        // enough
        j4 = 4 * i0 + pp - 3;
        emin = v!(z, j4 + 4);
        d = v!(z, j4) - tau;
        st.dmin = d;
        st.dmin1 = -v!(z, j4);
        if st.ieee {
            // Code for IEEE arithmetic.
            if pp == 0 {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 2) = d + v!(z, j4 - 1);
                    let temp = v!(z, j4 + 1) / v!(z, j4 - 2);
                    d = fma(d, temp, -tau);
                    if d < dthresh {
                        d = 0.0;
                    }
                    st.dmin = st.dmin.min(d);
                    v!(z, j4) = v!(z, j4 - 1) * temp;
                    emin = v!(z, j4).min(emin);
                    j4 += 4;
                }
            } else {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 3) = d + v!(z, j4);
                    let temp = v!(z, j4 + 2) / v!(z, j4 - 3);
                    d = fma(d, temp, -tau);
                    if d < dthresh {
                        d = 0.0;
                    }
                    st.dmin = st.dmin.min(d);
                    v!(z, j4 - 1) = v!(z, j4) * temp;
                    emin = v!(z, j4 - 1).min(emin);
                    j4 += 4;
                }
            }

            // Unroll last two steps.
            st.dn2 = d;
            st.dmin2 = st.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn2 + v!(z, j4p2);
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn1 = fma(v!(z, j4p2 + 2), st.dn2 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn1);

            st.dmin1 = st.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn1 + v!(z, j4p2);
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn = fma(v!(z, j4p2 + 2), st.dn1 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn);
        } else {
            // Code for non IEEE arithmetic.
            if pp == 0 {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 2) = d + v!(z, j4 - 1);
                    if d < 0.0 {
                        return;
                    }
                    v!(z, j4) = v!(z, j4 + 1) * (v!(z, j4 - 1) / v!(z, j4 - 2));
                    d = fma(v!(z, j4 + 1), d / v!(z, j4 - 2), -tau);
                    if d < dthresh {
                        d = 0.0;
                    }
                    st.dmin = st.dmin.min(d);
                    emin = emin.min(v!(z, j4));
                    j4 += 4;
                }
            } else {
                j4 = 4 * i0;
                while j4 <= 4 * (n0 - 3) {
                    v!(z, j4 - 3) = d + v!(z, j4);
                    if d < 0.0 {
                        return;
                    }
                    v!(z, j4 - 1) = v!(z, j4 + 2) * (v!(z, j4) / v!(z, j4 - 3));
                    d = fma(v!(z, j4 + 2), d / v!(z, j4 - 3), -tau);
                    if d < dthresh {
                        d = 0.0;
                    }
                    st.dmin = st.dmin.min(d);
                    emin = emin.min(v!(z, j4 - 1));
                    j4 += 4;
                }
            }

            // Unroll last two steps.
            st.dn2 = d;
            st.dmin2 = st.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn2 + v!(z, j4p2);
            if st.dn2 < 0.0 {
                return;
            }
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn1 = fma(v!(z, j4p2 + 2), st.dn2 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn1);

            st.dmin1 = st.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            v!(z, j4 - 2) = st.dn1 + v!(z, j4p2);
            if st.dn1 < 0.0 {
                return;
            }
            v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
            st.dn = fma(v!(z, j4p2 + 2), st.dn1 / v!(z, j4 - 2), -tau);
            st.dmin = st.dmin.min(st.dn);
        }
    }

    v!(z, j4 + 2) = st.dn;
    v!(z, 4 * n0 - pp) = emin;
}

/// `DLASQ6`: one dqd transform (no shift) with safeguards against
/// underflow and overflow.
unsafe fn dlasq6(st: &mut Dqds, z: *mut f64) {
    let (i0, n0, pp) = (st.i0, st.n0, st.pp);
    if (n0 - i0 - 1) <= 0 {
        return;
    }

    let safmin = SAFMIN;
    let mut j4 = 4 * i0 + pp - 3;
    let mut emin = v!(z, j4 + 4);
    let mut d = v!(z, j4);
    let mut dmin = d;

    if pp == 0 {
        j4 = 4 * i0;
        while j4 <= 4 * (n0 - 3) {
            v!(z, j4 - 2) = d + v!(z, j4 - 1);
            if v!(z, j4 - 2) == 0.0 {
                v!(z, j4) = 0.0;
                d = v!(z, j4 + 1);
                dmin = d;
                emin = 0.0;
            } else if safmin * v!(z, j4 + 1) < v!(z, j4 - 2)
                && safmin * v!(z, j4 - 2) < v!(z, j4 + 1)
            {
                let temp = v!(z, j4 + 1) / v!(z, j4 - 2);
                v!(z, j4) = v!(z, j4 - 1) * temp;
                d *= temp;
            } else {
                v!(z, j4) = v!(z, j4 + 1) * (v!(z, j4 - 1) / v!(z, j4 - 2));
                d = v!(z, j4 + 1) * (d / v!(z, j4 - 2));
            }
            dmin = dmin.min(d);
            emin = emin.min(v!(z, j4));
            j4 += 4;
        }
    } else {
        j4 = 4 * i0;
        while j4 <= 4 * (n0 - 3) {
            v!(z, j4 - 3) = d + v!(z, j4);
            if v!(z, j4 - 3) == 0.0 {
                v!(z, j4 - 1) = 0.0;
                d = v!(z, j4 + 2);
                dmin = d;
                emin = 0.0;
            } else if safmin * v!(z, j4 + 2) < v!(z, j4 - 3)
                && safmin * v!(z, j4 - 3) < v!(z, j4 + 2)
            {
                let temp = v!(z, j4 + 2) / v!(z, j4 - 3);
                v!(z, j4 - 1) = v!(z, j4) * temp;
                d *= temp;
            } else {
                v!(z, j4 - 1) = v!(z, j4 + 2) * (v!(z, j4) / v!(z, j4 - 3));
                d = v!(z, j4 + 2) * (d / v!(z, j4 - 3));
            }
            dmin = dmin.min(d);
            emin = emin.min(v!(z, j4 - 1));
            j4 += 4;
        }
    }

    // Unroll last two steps.
    let dnm2 = d;
    let dmin2 = dmin;
    j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    v!(z, j4 - 2) = dnm2 + v!(z, j4p2);
    let dnm1;
    if v!(z, j4 - 2) == 0.0 {
        v!(z, j4) = 0.0;
        dnm1 = v!(z, j4p2 + 2);
        dmin = dnm1;
        emin = 0.0;
    } else if safmin * v!(z, j4p2 + 2) < v!(z, j4 - 2) && safmin * v!(z, j4 - 2) < v!(z, j4p2 + 2) {
        let temp = v!(z, j4p2 + 2) / v!(z, j4 - 2);
        v!(z, j4) = v!(z, j4p2) * temp;
        dnm1 = dnm2 * temp;
    } else {
        v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
        dnm1 = v!(z, j4p2 + 2) * (dnm2 / v!(z, j4 - 2));
    }
    dmin = dmin.min(dnm1);

    let dmin1 = dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    v!(z, j4 - 2) = dnm1 + v!(z, j4p2);
    let dn;
    if v!(z, j4 - 2) == 0.0 {
        v!(z, j4) = 0.0;
        dn = v!(z, j4p2 + 2);
        dmin = dn;
        emin = 0.0;
    } else if safmin * v!(z, j4p2 + 2) < v!(z, j4 - 2) && safmin * v!(z, j4 - 2) < v!(z, j4p2 + 2) {
        let temp = v!(z, j4p2 + 2) / v!(z, j4 - 2);
        v!(z, j4) = v!(z, j4p2) * temp;
        dn = dnm1 * temp;
    } else {
        v!(z, j4) = v!(z, j4p2 + 2) * (v!(z, j4p2) / v!(z, j4 - 2));
        dn = v!(z, j4p2 + 2) * (dnm1 / v!(z, j4 - 2));
    }
    dmin = dmin.min(dn);

    v!(z, j4 + 2) = dn;
    v!(z, 4 * n0 - pp) = emin;
    st.dmin = dmin;
    st.dmin1 = dmin1;
    st.dmin2 = dmin2;
    st.dn = dn;
    st.dn1 = dnm1;
    st.dn2 = dnm2;
}
