//! Divide and conquer for the bidiagonal SVD: `DBDSDC` (`COMPQ = 'N'` or
//! `'I'`, the two `DGESDD` calls), its tree driver `DLASD0`, the merge
//! `DLASD1` (deflation `DLASD2`, secular equation `DLASD3`/`DLASD4`/
//! `DLASD5`) and the subproblem tree `DLASDT`. Fused products follow
//! gfortran's GIMPLE for R's libRlapack.

use super::aux::{dlacpy, dlamrg, dlanst_m, dlapy2, dlartg, dlascl_g, dlaset, dlasr};
use super::bdsqr::dlasdq;
use super::blas::{dcopy, dgemm, dnrm2, drot, dswap};
use super::stedc::dlaed6;
use super::{fma, lsame, sign, EPS};

/// `ILAENV(9, 'DBDSDC', ...)`: the largest subproblem solved directly.
pub(crate) const SMLSIZ: i32 = 25;

const MAXIT: i32 = 400;

/// `DLASD5`: the `i`-th (1 or 2) square root of the eigenvalues of
/// `diag(d)^2 + rho z z'` for `n = 2`, into `dsigma`, with
/// `delta(j) = d(j) - dsigma` and `work(j) = d(j) + dsigma`.
unsafe fn dlasd5(
    i: i32,
    d: *const f64,
    z: *const f64,
    delta: *mut f64,
    rho: f64,
    dsigma: &mut f64,
    work: *mut f64,
) {
    let d1 = v!(d, 1);
    let d2 = v!(d, 2);
    let z1 = v!(z, 1);
    let z2 = v!(z, 2);
    let del = d2 - d1;
    let delsq = del * (d2 + d1);
    // Shared by B in every branch, so rounded.
    let rzz = rho * (z1 * z1 + z2 * z2);
    if i == 1 {
        let w = 1.0 + 4.0 * rho * (z2 * z2 / fma(d2, 3.0, d1) - z1 * z1 / fma(d1, 3.0, d2)) / del;
        if w > 0.0 {
            let b = delsq + rzz;
            let c = rho * z1 * z1 * delsq;
            let mut tau = 2.0 * c / (b + fma(b, b, -(c * 4.0)).abs().sqrt());
            tau /= d1 + fma(d1, d1, tau).sqrt();
            *dsigma = d1 + tau;
            v!(delta, 1) = -tau;
            v!(delta, 2) = del - tau;
            v!(work, 1) = fma(d1, 2.0, tau);
            v!(work, 2) = (d1 + tau) + d2;
        } else {
            let b = -delsq + rzz;
            let c = rho * z2 * z2 * delsq;
            let sq = fma(b, b, c * 4.0).sqrt();
            let mut tau = if b > 0.0 {
                -2.0 * c / (b + sq)
            } else {
                (b - sq) / 2.0
            };
            tau /= d2 + fma(d2, d2, tau).abs().sqrt();
            *dsigma = d2 + tau;
            v!(delta, 1) = -(del + tau);
            v!(delta, 2) = -tau;
            v!(work, 1) = d1 + tau + d2;
            v!(work, 2) = fma(d2, 2.0, tau);
        }
    } else {
        let b = -delsq + rzz;
        let c = rho * z2 * z2 * delsq;
        let sq = fma(b, b, c * 4.0).sqrt();
        let mut tau = if b > 0.0 {
            (b + sq) / 2.0
        } else {
            2.0 * c / (-b + sq)
        };
        tau /= d2 + fma(d2, d2, tau).sqrt();
        *dsigma = d2 + tau;
        v!(delta, 1) = -(del + tau);
        v!(delta, 2) = -tau;
        v!(work, 1) = d1 + tau + d2;
        v!(work, 2) = fma(d2, 2.0, tau);
    }
}

/// The `PSI`/`DPSI` (or `PHI`/`DPHI`) accumulation of one side of the
/// secular sum: `temp = z(j) / (work(j) delta(j))`, `s += z(j) temp` and
/// `ds += temp^2` each in one rounding, and `erretm += s`.
#[inline(always)]
unsafe fn sec_term(
    z: *const f64,
    work: *const f64,
    delta: *const f64,
    j: i32,
    s: &mut f64,
    ds: &mut f64,
    erretm: &mut f64,
) {
    let temp = v!(z, j) / (v!(work, j) * v!(delta, j));
    *s = fma(v!(z, j), temp, *s);
    *ds = fma(temp, temp, *ds);
    *erretm += *s;
}

/// `DLASD4`: the `i`-th updated singular value of the secular equation
/// `1 + rho sum_j z_j^2 / ((d_j - sigma)(d_j + sigma)) = 0` for ascending
/// `d >= 0`, into `sigma`, with `delta(j) = d(j) - sigma` and
/// `work(j) = d(j) + sigma`. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlasd4(
    n: i32,
    i: i32,
    d: *const f64,
    z: *const f64,
    delta: *mut f64,
    rho: f64,
    sigma: &mut f64,
    work: *mut f64,
) -> i32 {
    if n == 1 {
        *sigma = fma(v!(d, 1), v!(d, 1), rho * v!(z, 1) * v!(z, 1)).sqrt();
        v!(delta, 1) = 1.0;
        v!(work, 1) = 1.0;
        return 0;
    }
    if n == 2 {
        dlasd5(i, d, z, delta, rho, sigma, work);
        return 0;
    }

    let eps = EPS;
    let rhoinv = 1.0 / rho;
    let mut tau2;
    let mut info = 0;

    if i == n {
        // I = N: the root lies beyond the last pole.
        let ii = n - 1;
        let mut niter = 1;
        let dn = v!(d, n);
        // D(N)*D(N) feeds several square roots, so it is rounded.
        let dn2 = dn * dn;
        let mut temp = rho / 2.0;
        let temp1 = temp / (dn + (dn2 + temp).sqrt());
        for j in 1..=n {
            v!(work, j) = v!(d, j) + dn + temp1;
            v!(delta, j) = (v!(d, j) - dn) - temp1;
        }
        let mut psi = 0.0;
        for j in 1..=n - 2 {
            psi += v!(z, j) * v!(z, j) / (v!(delta, j) * v!(work, j));
        }
        let c = rhoinv + psi;
        let w = c
            + v!(z, ii) * v!(z, ii) / (v!(delta, ii) * v!(work, ii))
            + v!(z, n) * v!(z, n) / (v!(delta, n) * v!(work, n));

        let mut tau;
        let quad = |c: f64| -> f64 {
            let delsq = (dn - v!(d, n - 1)) * (dn + v!(d, n - 1));
            let a = fma(-c, delsq, v!(z, n - 1) * v!(z, n - 1)) + v!(z, n) * v!(z, n);
            let b = v!(z, n) * v!(z, n) * delsq;
            let sq = fma(a, a, c * (b * 4.0)).sqrt();
            if a < 0.0 {
                2.0 * b / (sq - a)
            } else {
                (a + sq) / (2.0 * c)
            }
        };
        if w <= 0.0 {
            let temp1 = (dn2 + rho).sqrt();
            temp = v!(z, n - 1) * v!(z, n - 1)
                / ((v!(d, n - 1) + temp1) * (dn - v!(d, n - 1) + rho / (dn + temp1)))
                + v!(z, n) * v!(z, n) / rho;
            if c <= temp {
                tau = rho;
            } else {
                tau2 = quad(c);
                tau = tau2 / (dn + (dn2 + tau2).sqrt());
            }
        } else {
            tau2 = quad(c);
            tau = tau2 / (dn + (dn2 + tau2).sqrt());
        }

        *sigma = dn + tau;
        for j in 1..=n {
            v!(delta, j) = (v!(d, j) - dn) - tau;
            v!(work, j) = v!(d, j) + dn + tau;
        }

        let mut dpsi = 0.0;
        let mut psi = 0.0;
        let mut erretm = 0.0;
        for j in 1..=ii {
            sec_term(z, work, delta, j, &mut psi, &mut dpsi, &mut erretm);
        }
        erretm = f64::abs(erretm);
        let zn = v!(z, n);
        let mut t = zn / (v!(delta, n) * v!(work, n));
        let mut dphi = t * t;
        // PHI = Z(N)*TEMP is contracted into each of its uses.
        erretm = fma(-zn, t, fma(fma(-zn, t, -psi), 8.0, erretm)) + rhoinv;
        let mut w = fma(zn, t, rhoinv) + psi;

        if w.abs() <= eps * erretm {
            return info;
        }

        // The first iteration step.
        niter += 1;
        let dtnsq1 = v!(work, n - 1) * v!(delta, n - 1);
        let dtnsq = v!(work, n) * v!(delta, n);
        let mut c = fma(-dphi, dtnsq, fma(-dtnsq1, dpsi, w));
        let a = fma(-(dtnsq1 * dtnsq), dpsi + dphi, (dtnsq + dtnsq1) * w);
        let b = dtnsq * dtnsq1 * w;
        if c < 0.0 {
            c = c.abs();
        }
        // SIGMA*SIGMA is rounded in this step.
        let ssq = *sigma * *sigma;
        let mut eta;
        if c == 0.0 {
            eta = rho - ssq;
        } else {
            let sq = fma(a, a, -(c * (b * 4.0))).abs().sqrt();
            if a >= 0.0 {
                eta = (a + sq) / (2.0 * c);
            } else {
                eta = 2.0 * b / (a - sq);
            }
        }
        if w * eta > 0.0 {
            eta = -w / (dpsi + dphi);
        }
        temp = eta - dtnsq;
        if temp > rho {
            eta = rho + dtnsq;
        }
        eta /= *sigma + (eta + ssq).sqrt();
        tau += eta;
        *sigma += eta;
        for j in 1..=n {
            v!(delta, j) -= eta;
            v!(work, j) += eta;
        }

        dpsi = 0.0;
        psi = 0.0;
        erretm = 0.0;
        for j in 1..=ii {
            sec_term(z, work, delta, j, &mut psi, &mut dpsi, &mut erretm);
        }
        erretm = erretm.abs();
        tau2 = v!(work, n) * v!(delta, n);
        t = zn / tau2;
        dphi = t * t;
        erretm = fma(-zn, t, fma(fma(-zn, t, -psi), 8.0, erretm)) + rhoinv;
        w = fma(zn, t, rhoinv) + psi;

        let iter = niter + 1;
        for _niter in iter..=MAXIT {
            if w.abs() <= eps * erretm {
                return info;
            }
            let dtnsq1 = v!(work, n - 1) * v!(delta, n - 1);
            let dtnsq = v!(work, n) * v!(delta, n);
            let c = fma(-dtnsq, dphi, fma(-dtnsq1, dpsi, w));
            let p = dtnsq1 * dtnsq;
            let a = fma(dtnsq + dtnsq1, w, -(p * (dpsi + dphi)));
            let b = p * w;
            let sq = fma(a, a, -(c * (b * 4.0))).abs().sqrt();
            let mut eta = if a >= 0.0 {
                (a + sq) / (2.0 * c)
            } else {
                2.0 * b / (a - sq)
            };
            if w * eta > 0.0 {
                eta = -w / (dpsi + dphi);
            }
            let temp = eta - dtnsq;
            if temp <= 0.0 {
                eta /= 2.0;
            }
            eta /= *sigma + fma(*sigma, *sigma, eta).sqrt();
            tau += eta;
            *sigma += eta;
            for j in 1..=n {
                v!(delta, j) -= eta;
                v!(work, j) += eta;
            }
            dpsi = 0.0;
            psi = 0.0;
            erretm = 0.0;
            for j in 1..=ii {
                sec_term(z, work, delta, j, &mut psi, &mut dpsi, &mut erretm);
            }
            erretm = erretm.abs();
            tau2 = v!(work, n) * v!(delta, n);
            let t = zn / tau2;
            dphi = t * t;
            erretm = fma(-zn, t, fma(fma(-zn, t, -psi), 8.0, erretm)) + rhoinv;
            w = fma(zn, t, rhoinv) + psi;
        }
        let _ = tau;
        return 1;
    }

    // 1 <= I < N: the root lies between d(i) and d(i+1).
    let mut niter = 1;
    let ip1 = i + 1;
    let di = v!(d, i);
    let dip1 = v!(d, ip1);
    let delsq = (dip1 - di) * (dip1 + di);
    let delsq2 = delsq / 2.0;
    // D(I)**2, D(IP1)**2, Z(I)**2 and Z(IP1)**2 each feed several branches,
    // so they are rounded.
    let di2 = di * di;
    let dip12 = dip1 * dip1;
    let zi2 = v!(z, i) * v!(z, i);
    let zip12 = v!(z, ip1) * v!(z, ip1);
    let sq2 = ((di2 + dip12) / 2.0).sqrt();
    let mut temp = delsq2 / (di + sq2);
    for j in 1..=n {
        v!(work, j) = v!(d, j) + di + temp;
        v!(delta, j) = (v!(d, j) - di) - temp;
    }
    let mut psi = 0.0;
    for j in 1..=i - 1 {
        psi += v!(z, j) * v!(z, j) / (v!(work, j) * v!(delta, j));
    }
    let mut phi = 0.0;
    let mut j = n;
    while j >= i + 2 {
        phi += v!(z, j) * v!(z, j) / (v!(work, j) * v!(delta, j));
        j -= 1;
    }
    let c = rhoinv + psi + phi;
    let w = c
        + v!(z, i) * v!(z, i) / (v!(work, i) * v!(delta, i))
        + v!(z, ip1) * v!(z, ip1) / (v!(work, ip1) * v!(delta, ip1));

    let mut geomavg = false;
    let orgati;
    let ii;
    let mut sglb;
    let mut sgub;
    let mut tau;
    let cdelsq = c * delsq;
    if w > 0.0 {
        // d(i)^2 < the root < (d(i)^2+d(i+1)^2)/2.
        orgati = true;
        ii = i;
        sglb = 0.0;
        sgub = delsq2 / (di + sq2);
        let a = (cdelsq + zi2) + zip12;
        let b = zi2 * delsq;
        let sq = fma(a, a, -(c * (b * 4.0))).abs().sqrt();
        tau2 = if a > 0.0 {
            2.0 * b / (a + sq)
        } else {
            (a - sq) / (2.0 * c)
        };
        tau = tau2 / (di + (di2 + tau2).sqrt());
        temp = eps.sqrt();
        if di <= temp * dip1 && v!(z, i).abs() <= temp && di > 0.0 {
            tau = (10.0 * di).min(sgub);
            geomavg = true;
        }
    } else {
        // (d(i)^2+d(i+1)^2)/2 <= the root < d(i+1)^2.
        orgati = false;
        ii = ip1;
        sglb = -delsq2 / (v!(d, ii) + sq2);
        sgub = 0.0;
        let a = (cdelsq - zi2) - zip12;
        let b = zip12 * delsq;
        let sq = fma(a, a, c * (b * 4.0)).abs().sqrt();
        tau2 = if a < 0.0 {
            2.0 * b / (a - sq)
        } else {
            -(a + sq) / (2.0 * c)
        };
        tau = tau2 / (dip1 + (dip12 + tau2).abs().sqrt());
    }

    let dii = v!(d, ii);
    *sigma = dii + tau;
    for j in 1..=n {
        v!(work, j) = v!(d, j) + dii + tau;
        v!(delta, j) = (v!(d, j) - dii) - tau;
    }
    let iim1 = ii - 1;
    let iip1 = ii + 1;

    let mut dpsi = 0.0;
    psi = 0.0;
    let mut erretm = 0.0;
    for j in 1..=iim1 {
        sec_term(z, work, delta, j, &mut psi, &mut dpsi, &mut erretm);
    }
    erretm = erretm.abs();
    let mut dphi = 0.0;
    phi = 0.0;
    let mut j = n;
    while j >= iip1 {
        sec_term(z, work, delta, j, &mut phi, &mut dphi, &mut erretm);
        j -= 1;
    }
    let mut w = rhoinv + phi + psi;

    let mut swtch3 = false;
    if orgati {
        if w < 0.0 {
            swtch3 = true;
        }
    } else if w > 0.0 {
        swtch3 = true;
    }
    if ii == 1 || ii == n {
        swtch3 = false;
    }

    let zii = v!(z, ii);
    let t = zii / (v!(work, ii) * v!(delta, ii));
    let mut dw = fma(t, t, dpsi + dphi);
    let t2 = zii * t;
    w += t2;
    erretm = fma(t2.abs(), 3.0, fma(phi - psi, 8.0, erretm) + 2.0 * rhoinv);

    if w.abs() <= eps * erretm {
        return info;
    }
    if w <= 0.0 {
        sglb = sglb.max(tau);
    } else {
        sgub = sgub.min(tau);
    }

    // The first iteration step.
    niter += 1;
    let mut eta = 0.0;
    let mut dd = [0.0f64; 3];
    let mut zz = [0.0f64; 3];
    if !swtch3 {
        let dtipsq = v!(work, ip1) * v!(delta, ip1);
        let dtisq = v!(work, i) * v!(delta, i);
        let c = if orgati {
            let q = v!(z, i) / dtisq;
            fma(delsq, q * q, fma(-dw, dtipsq, w))
        } else {
            let q = v!(z, ip1) / dtipsq;
            fma(-delsq, q * q, fma(-dw, dtisq, w))
        };
        let p = dtipsq * dtisq;
        let mut a = fma(w, dtipsq + dtisq, -(p * dw));
        let b = p * w;
        if c == 0.0 {
            if a == 0.0 {
                a = if orgati {
                    fma(dtipsq * dtipsq, dpsi + dphi, zi2)
                } else {
                    fma(dtisq * dtisq, dpsi + dphi, zip12)
                };
            }
            eta = b / a;
        } else {
            let sq = fma(a, a, -(c * (b * 4.0))).abs().sqrt();
            eta = if a <= 0.0 {
                (a - sq) / (2.0 * c)
            } else {
                2.0 * b / (a + sq)
            };
        }
    } else {
        // Interpolation with three poles.
        let dtiim = v!(work, iim1) * v!(delta, iim1);
        let dtiip = v!(work, iip1) * v!(delta, iip1);
        let temp = rhoinv + psi + phi;
        let c;
        if orgati {
            let mut temp1 = v!(z, iim1) / dtiim;
            temp1 *= temp1;
            let diim1 = v!(d, iim1);
            let diip1 = v!(d, iip1);
            c = fma(
                -((diim1 - diip1) * (diim1 + diip1)),
                temp1,
                fma(-dtiip, dpsi + dphi, temp),
            );
            zz[0] = v!(z, iim1) * v!(z, iim1);
            zz[2] = if dpsi < temp1 {
                dtiip * dtiip * dphi
            } else {
                dtiip * dtiip * ((dpsi - temp1) + dphi)
            };
        } else {
            let mut temp1 = v!(z, iip1) / dtiip;
            temp1 *= temp1;
            let diim1 = v!(d, iim1);
            let diip1 = v!(d, iip1);
            c = fma(
                -((diip1 - diim1) * (diim1 + diip1)),
                temp1,
                fma(-dtiim, dpsi + dphi, temp),
            );
            zz[0] = if dphi < temp1 {
                dtiim * dtiim * dpsi
            } else {
                dtiim * dtiim * (dpsi + (dphi - temp1))
            };
            zz[2] = v!(z, iip1) * v!(z, iip1);
        }
        zz[1] = zii * zii;
        dd[0] = dtiim;
        dd[1] = v!(delta, ii) * v!(work, ii);
        dd[2] = dtiip;
        info = dlaed6(niter, orgati, c, dd.as_ptr(), zz.as_ptr(), w, &mut eta);
        if info != 0 {
            // DLAED6 failed: fall back to the two-pole interpolation.
            swtch3 = false;
            info = 0;
            let dtipsq = v!(work, ip1) * v!(delta, ip1);
            let dtisq = v!(work, i) * v!(delta, i);
            let c = if orgati {
                let q = v!(z, i) / dtisq;
                fma(delsq, q * q, fma(-dw, dtipsq, w))
            } else {
                let q = v!(z, ip1) / dtipsq;
                fma(-delsq, q * q, fma(-dw, dtisq, w))
            };
            let p = dtipsq * dtisq;
            let mut a = fma(w, dtipsq + dtisq, -(p * dw));
            let b = p * w;
            if c == 0.0 {
                if a == 0.0 {
                    a = if orgati {
                        fma(v!(z, i), v!(z, i), dtipsq * dtipsq * (dpsi + dphi))
                    } else {
                        fma(v!(z, ip1), v!(z, ip1), dtisq * dtisq * (dpsi + dphi))
                    };
                }
                eta = b / a;
            } else {
                let sq = fma(a, a, -(c * (b * 4.0))).abs().sqrt();
                eta = if a <= 0.0 {
                    (a - sq) / (2.0 * c)
                } else {
                    2.0 * b / (a + sq)
                };
            }
        }
    }

    // The new eta must stay inside the bracket.
    if w * eta >= 0.0 {
        eta = -w / dw;
    }
    eta /= *sigma + fma(*sigma, *sigma, eta).sqrt();
    temp = tau + eta;
    if temp > sgub || temp < sglb {
        eta = if w < 0.0 {
            (sgub - tau) / 2.0
        } else {
            (sglb - tau) / 2.0
        };
        if geomavg {
            if w < 0.0 {
                if tau > 0.0 {
                    eta = (sgub * tau).sqrt() - tau;
                }
            } else if sglb > 0.0 {
                eta = (sglb * tau).sqrt() - tau;
            }
        }
    }
    let mut prew = w;
    tau += eta;
    *sigma += eta;
    for j in 1..=n {
        v!(work, j) += eta;
        v!(delta, j) -= eta;
    }

    let eval = |dpsi: &mut f64,
                psi: &mut f64,
                dphi: &mut f64,
                phi: &mut f64,
                erretm: &mut f64,
                dw: &mut f64,
                w: &mut f64| {
        *dpsi = 0.0;
        *psi = 0.0;
        *erretm = 0.0;
        for j in 1..=iim1 {
            sec_term(z, work, delta, j, psi, dpsi, erretm);
        }
        *erretm = erretm.abs();
        *dphi = 0.0;
        *phi = 0.0;
        let mut j = n;
        while j >= iip1 {
            sec_term(z, work, delta, j, phi, dphi, erretm);
            j -= 1;
        }
        let tau2 = v!(work, ii) * v!(delta, ii);
        let t = zii / tau2;
        *dw = fma(t, t, *dpsi + *dphi);
        let t2 = zii * t;
        *w = rhoinv + *phi + *psi + t2;
        *erretm = fma(t2.abs(), 3.0, 2.0 * rhoinv + fma(*phi - *psi, 8.0, *erretm));
    };
    eval(
        &mut dpsi,
        &mut psi,
        &mut dphi,
        &mut phi,
        &mut erretm,
        &mut dw,
        &mut w,
    );

    let mut swtch = false;
    if orgati {
        if -w > prew.abs() / 10.0 {
            swtch = true;
        }
    } else if w > prew.abs() / 10.0 {
        swtch = true;
    }

    // The main loop to update the values of the array DELTA.
    let iter = niter + 1;
    for niter in iter..=MAXIT {
        if w.abs() <= eps * erretm {
            return info;
        }
        if w <= 0.0 {
            sglb = sglb.max(tau);
        } else {
            sgub = sgub.min(tau);
        }

        // Two-pole interpolation, as the first step and as the fallback when
        // DLAED6 fails.
        let two_pole = |dpsi: &mut f64, dphi: &mut f64| -> f64 {
            let dtipsq = v!(work, ip1) * v!(delta, ip1);
            let dtisq = v!(work, i) * v!(delta, i);
            let c;
            if !swtch {
                c = if orgati {
                    let q = v!(z, i) / dtisq;
                    fma(delsq, q * q, fma(-dtipsq, dw, w))
                } else {
                    let q = v!(z, ip1) / dtipsq;
                    fma(-delsq, q * q, fma(-dtisq, dw, w))
                };
            } else {
                let t = zii / (v!(work, ii) * v!(delta, ii));
                // TEMP*TEMP is hoisted out of the two branches and rounded.
                let tt = t * t;
                if orgati {
                    *dpsi += tt;
                } else {
                    *dphi += tt;
                }
                c = fma(-*dphi, dtipsq, fma(-*dpsi, dtisq, w));
            }
            let p = dtipsq * dtisq;
            let mut a = fma(dtipsq + dtisq, w, -(p * dw));
            let b = p * w;
            if c == 0.0 {
                if a == 0.0 {
                    a = if !swtch {
                        if orgati {
                            fma(v!(z, i), v!(z, i), dtipsq * dtipsq * (*dpsi + *dphi))
                        } else {
                            fma(v!(z, ip1), v!(z, ip1), dtisq * dtisq * (*dpsi + *dphi))
                        }
                    } else {
                        fma(dtisq * dtisq, *dpsi, dtipsq * dtipsq * *dphi)
                    };
                }
                b / a
            } else {
                let sq = fma(a, a, -(c * (b * 4.0))).abs().sqrt();
                if a <= 0.0 {
                    (a - sq) / (2.0 * c)
                } else {
                    2.0 * b / (a + sq)
                }
            }
        };

        let mut eta;
        if !swtch3 {
            eta = two_pole(&mut dpsi, &mut dphi);
        } else {
            // Interpolation with three poles.
            let dtiim = v!(work, iim1) * v!(delta, iim1);
            let dtiip = v!(work, iip1) * v!(delta, iip1);
            let temp = rhoinv + psi + phi;
            let c;
            if swtch {
                c = fma(-dtiip, dphi, fma(-dtiim, dpsi, temp));
                zz[0] = dtiim * dtiim * dpsi;
                zz[2] = dtiip * dtiip * dphi;
            } else if orgati {
                let mut temp1 = v!(z, iim1) / dtiim;
                temp1 *= temp1;
                let diim1 = v!(d, iim1);
                let diip1 = v!(d, iip1);
                c = fma(
                    -((diim1 - diip1) * (diim1 + diip1)),
                    temp1,
                    fma(-dtiip, dpsi + dphi, temp),
                );
                zz[0] = v!(z, iim1) * v!(z, iim1);
                zz[2] = if dpsi < temp1 {
                    dtiip * dtiip * dphi
                } else {
                    dtiip * dtiip * ((dpsi - temp1) + dphi)
                };
            } else {
                let mut temp1 = v!(z, iip1) / dtiip;
                temp1 *= temp1;
                let diim1 = v!(d, iim1);
                let diip1 = v!(d, iip1);
                c = fma(
                    -((diip1 - diim1) * (diim1 + diip1)),
                    temp1,
                    fma(-dtiim, dpsi + dphi, temp),
                );
                zz[0] = if dphi < temp1 {
                    dtiim * dtiim * dpsi
                } else {
                    dtiim * dtiim * (dpsi + (dphi - temp1))
                };
                zz[2] = v!(z, iip1) * v!(z, iip1);
            }
            dd[0] = dtiim;
            dd[1] = v!(delta, ii) * v!(work, ii);
            dd[2] = dtiip;
            eta = 0.0;
            info = dlaed6(niter, orgati, c, dd.as_ptr(), zz.as_ptr(), w, &mut eta);
            if info != 0 {
                swtch3 = false;
                info = 0;
                eta = two_pole(&mut dpsi, &mut dphi);
            }
        }

        if w * eta >= 0.0 {
            eta = -w / dw;
        }
        eta /= *sigma + fma(*sigma, *sigma, eta).sqrt();
        let temp = tau + eta;
        if temp > sgub || temp < sglb {
            eta = if w < 0.0 {
                (sgub - tau) / 2.0
            } else {
                (sglb - tau) / 2.0
            };
            if geomavg {
                if w < 0.0 {
                    if tau > 0.0 {
                        eta = (sgub * tau).sqrt() - tau;
                    }
                } else if sglb > 0.0 {
                    eta = (sglb * tau).sqrt() - tau;
                }
            }
        }
        prew = w;
        tau += eta;
        *sigma += eta;
        for j in 1..=n {
            v!(work, j) += eta;
            v!(delta, j) -= eta;
        }
        eval(
            &mut dpsi,
            &mut psi,
            &mut dphi,
            &mut phi,
            &mut erretm,
            &mut dw,
            &mut w,
        );
        if w * prew > 0.0 && w.abs() > prew.abs() / 10.0 {
            swtch = !swtch;
        }
    }
    1
}

/// `DLASDT`: the computation tree of the divide and conquer on `n` rows
/// with leaves of at most `msub` rows. Returns `(lvl, nd)`.
pub(crate) unsafe fn dlasdt(
    n: i32,
    inode: *mut i32,
    ndiml: *mut i32,
    ndimr: *mut i32,
    msub: i32,
) -> (i32, i32) {
    let maxn = 1.max(n);
    let temp = (maxn as f64 / (msub + 1) as f64).ln() / 2.0f64.ln();
    let lvl = temp as i32 + 1;
    let i = n / 2;
    v!(inode, 1) = i + 1;
    v!(ndiml, 1) = i;
    v!(ndimr, 1) = n - i - 1;
    let mut il = 0;
    let mut ir = 1;
    let mut llst = 1;
    for _nlvl in 1..=lvl - 1 {
        for i in 0..=llst - 1 {
            il += 2;
            ir += 2;
            let ncrnt = llst + i;
            v!(ndiml, il) = v!(ndiml, ncrnt) / 2;
            v!(ndimr, il) = v!(ndiml, ncrnt) - v!(ndiml, il) - 1;
            v!(inode, il) = v!(inode, ncrnt) - v!(ndimr, il) - 1;
            v!(ndiml, ir) = v!(ndimr, ncrnt) / 2;
            v!(ndimr, ir) = v!(ndimr, ncrnt) - v!(ndiml, ir) - 1;
            v!(inode, ir) = v!(inode, ncrnt) + v!(ndiml, ir) + 1;
        }
        llst *= 2;
    }
    (lvl, llst * 2 - 1)
}

/// `DLASD2`: merges the two sets of singular values of `DLASD1` into one
/// sorted set and deflates it; returns the size `k` of the secular
/// equation. `coltyp(1..4)` holds the column-type counts on exit.
#[allow(clippy::too_many_arguments)]
unsafe fn dlasd2(
    nl: i32,
    nr: i32,
    sqre: i32,
    d: *mut f64,
    z: *mut f64,
    alpha: f64,
    beta: f64,
    u: *mut f64,
    ldu: i32,
    vt: *mut f64,
    ldvt: i32,
    dsigma: *mut f64,
    u2: *mut f64,
    ldu2: i32,
    vt2: *mut f64,
    ldvt2: i32,
    idxp: *mut i32,
    idx: *mut i32,
    idxc: *mut i32,
    idxq: *mut i32,
    coltyp: *mut i32,
) -> i32 {
    let n = nl + nr + 1;
    let m = n + sqre;
    let nlp1 = nl + 1;
    let nlp2 = nl + 2;

    // Generate the first part of the vector Z and move the singular values
    // in the first part of D one position backward.
    let z1 = alpha * m!(vt, ldvt, nlp1, nlp1);
    v!(z, 1) = z1;
    let mut i = nl;
    while i >= 1 {
        v!(z, i + 1) = alpha * m!(vt, ldvt, i, nlp1);
        v!(d, i + 1) = v!(d, i);
        v!(idxq, i + 1) = v!(idxq, i) + 1;
        i -= 1;
    }
    for i in nlp2..=m {
        v!(z, i) = beta * m!(vt, ldvt, i, nlp2);
    }

    // Initialize some reference arrays.
    for i in 2..=nlp1 {
        v!(coltyp, i) = 1;
    }
    for i in nlp2..=n {
        v!(coltyp, i) = 2;
    }
    for i in nlp2..=n {
        v!(idxq, i) += nlp1;
    }
    for i in 2..=n {
        let q = v!(idxq, i);
        v!(dsigma, i) = v!(d, q);
        m!(u2, ldu2, i, 1) = v!(z, q);
        v!(idxc, i) = v!(coltyp, q);
    }
    dlamrg(nl, nr, pv!(dsigma, 2), 1, 1, pv!(idx, 2));
    for i in 2..=n {
        let idxi = 1 + v!(idx, i);
        v!(d, i) = v!(dsigma, idxi);
        v!(z, i) = m!(u2, ldu2, idxi, 1);
        v!(coltyp, i) = v!(idxc, idxi);
    }

    // Calculate the allowable deflation tolerance.
    let eps = EPS;
    let mut tol = alpha.abs().max(beta.abs());
    tol = 8.0 * eps * v!(d, n).abs().max(tol);

    let mut k = 1;
    let mut k2 = n + 1;
    let mut jprev = 0;
    let mut all_deflated = false;
    for j in 2..=n {
        if v!(z, j).abs() <= tol {
            // Deflate due to small z component.
            k2 -= 1;
            v!(idxp, k2) = j;
            v!(coltyp, j) = 4;
            if j == n {
                all_deflated = true;
                break;
            }
        } else {
            jprev = j;
            break;
        }
    }
    if !all_deflated {
        let mut j = jprev;
        loop {
            j += 1;
            if j > n {
                break;
            }
            if v!(z, j).abs() <= tol {
                k2 -= 1;
                v!(idxp, k2) = j;
                v!(coltyp, j) = 4;
            } else if (v!(d, j) - v!(d, jprev)).abs() <= tol {
                // Two singular values are close: deflate one by a rotation.
                let mut s = v!(z, jprev);
                let mut c = v!(z, j);
                let tau = dlapy2(c, s);
                c /= tau;
                s = -s / tau;
                v!(z, j) = tau;
                v!(z, jprev) = 0.0;
                let mut idxjp = v!(idxq, v!(idx, jprev) + 1);
                let mut idxj = v!(idxq, v!(idx, j) + 1);
                if idxjp <= nlp1 {
                    idxjp -= 1;
                }
                if idxj <= nlp1 {
                    idxj -= 1;
                }
                drot(n, pm!(u, ldu, 1, idxjp), 1, pm!(u, ldu, 1, idxj), 1, c, s);
                drot(
                    m,
                    pm!(vt, ldvt, idxjp, 1),
                    ldvt,
                    pm!(vt, ldvt, idxj, 1),
                    ldvt,
                    c,
                    s,
                );
                if v!(coltyp, j) != v!(coltyp, jprev) {
                    v!(coltyp, j) = 3;
                }
                v!(coltyp, jprev) = 4;
                k2 -= 1;
                v!(idxp, k2) = jprev;
                jprev = j;
            } else {
                k += 1;
                m!(u2, ldu2, k, 1) = v!(z, jprev);
                v!(dsigma, k) = v!(d, jprev);
                v!(idxp, k) = jprev;
                jprev = j;
            }
        }
        // Record the last singular value.
        k += 1;
        m!(u2, ldu2, k, 1) = v!(z, jprev);
        v!(dsigma, k) = v!(d, jprev);
        v!(idxp, k) = jprev;
    }

    // Count the singular values of each column type and sort them into the
    // order DLASD3 expects.
    let mut ctot = [0i32; 4];
    for j in 2..=n {
        let ct = v!(coltyp, j);
        ctot[(ct - 1) as usize] += 1;
    }
    let mut psm = [0i32; 4];
    psm[0] = 2;
    psm[1] = 2 + ctot[0];
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    for j in 2..=n {
        let jp = v!(idxp, j);
        let ct = v!(coltyp, jp);
        v!(idxc, psm[(ct - 1) as usize]) = j;
        psm[(ct - 1) as usize] += 1;
    }
    for j in 2..=n {
        let jp = v!(idxp, j);
        v!(dsigma, j) = v!(d, jp);
        let mut idxj = v!(idxq, v!(idx, v!(idxp, v!(idxc, j))) + 1);
        if idxj <= nlp1 {
            idxj -= 1;
        }
        dcopy(n, pm!(u, ldu, 1, idxj), 1, pm!(u2, ldu2, 1, j), 1);
        dcopy(
            m,
            pm!(vt, ldvt, idxj, 1),
            ldvt,
            pm!(vt2, ldvt2, j, 1),
            ldvt2,
        );
    }

    // Determine DSIGMA(1), DSIGMA(2) and Z(1).
    v!(dsigma, 1) = 0.0;
    let hlftol = tol / 2.0;
    if v!(dsigma, 2).abs() <= hlftol {
        v!(dsigma, 2) = hlftol;
    }
    let mut c = 0.0;
    let mut s = 0.0;
    if m > n {
        v!(z, 1) = dlapy2(z1, v!(z, m));
        if v!(z, 1) <= tol {
            c = 1.0;
            s = 0.0;
            v!(z, 1) = tol;
        } else {
            c = z1 / v!(z, 1);
            s = v!(z, m) / v!(z, 1);
        }
    } else if z1.abs() <= tol {
        v!(z, 1) = tol;
    } else {
        v!(z, 1) = z1;
    }

    // Move the rest of the updating row to Z.
    dcopy(k - 1, pm!(u2, ldu2, 2, 1), 1, pv!(z, 2), 1);

    // Determine the first column of U2, the first row of VT2 and the last
    // row of VT.
    dlaset(b'A', n, 1, 0.0, 0.0, u2, ldu2);
    m!(u2, ldu2, nlp1, 1) = 1.0;
    if m > n {
        for i in 1..=nlp1 {
            m!(vt, ldvt, m, i) = -s * m!(vt, ldvt, nlp1, i);
            m!(vt2, ldvt2, 1, i) = c * m!(vt, ldvt, nlp1, i);
        }
        for i in nlp2..=m {
            m!(vt2, ldvt2, 1, i) = s * m!(vt, ldvt, m, i);
            m!(vt, ldvt, m, i) = c * m!(vt, ldvt, m, i);
        }
    } else {
        dcopy(m, pm!(vt, ldvt, nlp1, 1), ldvt, vt2, ldvt2);
    }
    if m > n {
        dcopy(m, pm!(vt, ldvt, m, 1), ldvt, pm!(vt2, ldvt2, m, 1), ldvt2);
    }

    // The deflated singular values and their vectors go back into the last
    // N - K slots of D, U and VT.
    if n > k {
        dcopy(n - k, pv!(dsigma, k + 1), 1, pv!(d, k + 1), 1);
        dlacpy(
            b'A',
            n,
            n - k,
            pm!(u2, ldu2, 1, k + 1),
            ldu2,
            pm!(u, ldu, 1, k + 1),
            ldu,
        );
        dlacpy(
            b'A',
            n - k,
            m,
            pm!(vt2, ldvt2, k + 1, 1),
            ldvt2,
            pm!(vt, ldvt, k + 1, 1),
            ldvt,
        );
    }
    // Copy CTOT into COLTYP for referencing in DLASD3.
    for j in 1..=4 {
        v!(coltyp, j) = ctot[(j - 1) as usize];
    }
    k
}

/// `DLASD3`: the singular values of the `k`-size secular equation of the
/// merge and the updated singular vectors. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlasd3(
    nl: i32,
    nr: i32,
    sqre: i32,
    k: i32,
    d: *mut f64,
    q: *mut f64,
    ldq: i32,
    dsigma: *const f64,
    u: *mut f64,
    ldu: i32,
    u2: *const f64,
    ldu2: i32,
    vt: *mut f64,
    ldvt: i32,
    vt2: *mut f64,
    ldvt2: i32,
    idxc: *const i32,
    ctot: *const i32,
    z: *mut f64,
) -> i32 {
    let n = nl + nr + 1;
    let m = n + sqre;
    let nlp1 = nl + 1;
    let nlp2 = nl + 2;

    if k == 1 {
        v!(d, 1) = v!(z, 1).abs();
        dcopy(m, vt2, ldvt2, vt, ldvt);
        if v!(z, 1) > 0.0 {
            dcopy(n, u2, 1, u, 1);
        } else {
            for i in 1..=n {
                m!(u, ldu, i, 1) = -m!(u2, ldu2, i, 1);
            }
        }
        return 0;
    }

    // Keep a copy of Z and normalize it.
    dcopy(k, z, 1, q, 1);
    let mut rho = dnrm2(k, z, 1);
    dlascl_g(rho, 1.0, k, 1, z, k);
    rho *= rho;

    // Find the new singular values.
    for j in 1..=k {
        let mut sig = 0.0;
        let info = dlasd4(
            k,
            j,
            dsigma,
            z,
            pm!(u, ldu, 1, j),
            rho,
            &mut sig,
            pm!(vt, ldvt, 1, j),
        );
        v!(d, j) = sig;
        if info != 0 {
            return info;
        }
    }

    // Compute updated Z.
    for i in 1..=k {
        v!(z, i) = m!(u, ldu, i, k) * m!(vt, ldvt, i, k);
        for j in 1..=i - 1 {
            v!(z, i) *= m!(u, ldu, i, j) * m!(vt, ldvt, i, j)
                / (v!(dsigma, i) - v!(dsigma, j))
                / (v!(dsigma, i) + v!(dsigma, j));
        }
        for j in i..=k - 1 {
            v!(z, i) *= m!(u, ldu, i, j) * m!(vt, ldvt, i, j)
                / (v!(dsigma, i) - v!(dsigma, j + 1))
                / (v!(dsigma, i) + v!(dsigma, j + 1));
        }
        v!(z, i) = sign(v!(z, i).abs().sqrt(), m!(q, ldq, i, 1));
    }

    // Left singular vectors of the modified diagonal matrix, then the
    // updated U.
    for i in 1..=k {
        m!(vt, ldvt, 1, i) = v!(z, 1) / m!(u, ldu, 1, i) / m!(vt, ldvt, 1, i);
        m!(u, ldu, 1, i) = -1.0;
        for j in 2..=k {
            m!(vt, ldvt, j, i) = v!(z, j) / m!(u, ldu, j, i) / m!(vt, ldvt, j, i);
            m!(u, ldu, j, i) = v!(dsigma, j) * m!(vt, ldvt, j, i);
        }
        let temp = dnrm2(k, pm!(u, ldu, 1, i), 1);
        m!(q, ldq, 1, i) = m!(u, ldu, 1, i) / temp;
        for j in 2..=k {
            let jc = v!(idxc, j);
            m!(q, ldq, j, i) = m!(u, ldu, jc, i) / temp;
        }
    }

    'u: {
        if k == 2 {
            dgemm(b'N', b'N', n, k, k, 1.0, u2, ldu2, q, ldq, 0.0, u, ldu);
            break 'u;
        }
        let ct1 = v!(ctot, 1);
        let ct2 = v!(ctot, 2);
        let ct3 = v!(ctot, 3);
        if ct1 > 0 {
            dgemm(
                b'N',
                b'N',
                nl,
                k,
                ct1,
                1.0,
                pm!(u2, ldu2, 1, 2),
                ldu2,
                pm!(q, ldq, 2, 1),
                ldq,
                0.0,
                u,
                ldu,
            );
            if ct3 > 0 {
                let ktemp = 2 + ct1 + ct2;
                dgemm(
                    b'N',
                    b'N',
                    nl,
                    k,
                    ct3,
                    1.0,
                    pm!(u2, ldu2, 1, ktemp),
                    ldu2,
                    pm!(q, ldq, ktemp, 1),
                    ldq,
                    1.0,
                    u,
                    ldu,
                );
            }
        } else if ct3 > 0 {
            let ktemp = 2 + ct1 + ct2;
            dgemm(
                b'N',
                b'N',
                nl,
                k,
                ct3,
                1.0,
                pm!(u2, ldu2, 1, ktemp),
                ldu2,
                pm!(q, ldq, ktemp, 1),
                ldq,
                0.0,
                u,
                ldu,
            );
        } else {
            dlacpy(b'F', nl, k, u2, ldu2, u, ldu);
        }
        dcopy(k, q, ldq, pm!(u, ldu, nlp1, 1), ldu);
        let ktemp = 2 + ct1;
        let ctemp = ct2 + ct3;
        dgemm(
            b'N',
            b'N',
            nr,
            k,
            ctemp,
            1.0,
            pm!(u2, ldu2, nlp2, ktemp),
            ldu2,
            pm!(q, ldq, ktemp, 1),
            ldq,
            0.0,
            pm!(u, ldu, nlp2, 1),
            ldu,
        );
    }

    // The right singular vectors of the modified diagonal matrix, then the
    // updated VT.
    for i in 1..=k {
        let temp = dnrm2(k, pm!(vt, ldvt, 1, i), 1);
        m!(q, ldq, i, 1) = m!(vt, ldvt, 1, i) / temp;
        for j in 2..=k {
            let jc = v!(idxc, j);
            m!(q, ldq, i, j) = m!(vt, ldvt, jc, i) / temp;
        }
    }
    if k == 2 {
        dgemm(b'N', b'N', k, m, k, 1.0, q, ldq, vt2, ldvt2, 0.0, vt, ldvt);
        return 0;
    }
    let ct1 = v!(ctot, 1);
    let ct2 = v!(ctot, 2);
    let ct3 = v!(ctot, 3);
    let mut ktemp = 1 + ct1;
    dgemm(
        b'N', b'N', k, nlp1, ktemp, 1.0, q, ldq, vt2, ldvt2, 0.0, vt, ldvt,
    );
    ktemp = 2 + ct1 + ct2;
    if ktemp <= ldvt2 {
        dgemm(
            b'N',
            b'N',
            k,
            nlp1,
            ct3,
            1.0,
            pm!(q, ldq, 1, ktemp),
            ldq,
            pm!(vt2, ldvt2, ktemp, 1),
            ldvt2,
            1.0,
            vt,
            ldvt,
        );
    }
    ktemp = ct1 + 1;
    let nrp1 = nr + sqre;
    if ktemp > 1 {
        for i in 1..=k {
            m!(q, ldq, i, ktemp) = m!(q, ldq, i, 1);
        }
        for i in nlp2..=m {
            m!(vt2, ldvt2, ktemp, i) = m!(vt2, ldvt2, 1, i);
        }
    }
    let ctemp = 1 + ct2 + ct3;
    dgemm(
        b'N',
        b'N',
        k,
        nrp1,
        ctemp,
        1.0,
        pm!(q, ldq, 1, ktemp),
        ldq,
        pm!(vt2, ldvt2, ktemp, nlp2),
        ldvt2,
        0.0,
        pm!(vt, ldvt, 1, nlp2),
        ldvt,
    );
    0
}

/// `DLASD1`: the SVD of the upper bidiagonal `[D1 alpha 0; 0 beta D2]`
/// (with an extra column when `sqre = 1`) from those of its two halves in
/// `u` and `vt`; `idxq` returns the permutation sorting `d` ascending.
/// `iwork` holds `4 n`, `work` `3 m^2 + 2 m`. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlasd1(
    nl: i32,
    nr: i32,
    sqre: i32,
    d: *mut f64,
    alpha: f64,
    beta: f64,
    u: *mut f64,
    ldu: i32,
    vt: *mut f64,
    ldvt: i32,
    idxq: *mut i32,
    iwork: *mut i32,
    work: *mut f64,
) -> i32 {
    let n = nl + nr + 1;
    let m = n + sqre;
    let ldu2 = n;
    let ldvt2 = m;
    let iz = 1;
    let isigma = iz + m;
    let iu2 = isigma + n;
    let ivt2 = iu2 + ldu2 * n;
    let iq = ivt2 + ldvt2 * m;
    let idx = 1;
    let idxc = idx + n;
    let coltyp = idxc + n;
    let idxp = coltyp + n;

    // Scale.
    let mut orgnrm = alpha.abs().max(beta.abs());
    v!(d, nl + 1) = 0.0;
    for i in 1..=n {
        if v!(d, i).abs() > orgnrm {
            orgnrm = v!(d, i).abs();
        }
    }
    dlascl_g(orgnrm, 1.0, n, 1, d, n);
    let alpha = alpha / orgnrm;
    let beta = beta / orgnrm;

    // Deflate singular values.
    let k = dlasd2(
        nl,
        nr,
        sqre,
        d,
        pv!(work, iz),
        alpha,
        beta,
        u,
        ldu,
        vt,
        ldvt,
        pv!(work, isigma),
        pv!(work, iu2),
        ldu2,
        pv!(work, ivt2),
        ldvt2,
        pv!(iwork, idxp),
        pv!(iwork, idx),
        pv!(iwork, idxc),
        idxq,
        pv!(iwork, coltyp),
    );

    // Solve the secular equation and update the singular vectors.
    let ldq = k;
    let info = dlasd3(
        nl,
        nr,
        sqre,
        k,
        d,
        pv!(work, iq),
        ldq,
        pv!(work, isigma),
        u,
        ldu,
        pv!(work, iu2),
        ldu2,
        vt,
        ldvt,
        pv!(work, ivt2),
        ldvt2,
        pv!(iwork, idxc),
        pv!(iwork, coltyp),
        pv!(work, iz),
    );
    if info != 0 {
        return info;
    }

    // Unscale and prepare the permutation for the next merge.
    dlascl_g(1.0, orgnrm, n, 1, d, n);
    let n1 = k;
    let n2 = n - k;
    dlamrg(n1, n2, d, 1, -1, idxq);
    0
}

/// `DLASD0`: the SVD of the `n`-by-`(n + sqre)` upper bidiagonal by divide
/// and conquer, `B = U S VT`, with `u` and `vt` set to the singular
/// vectors. `iwork` holds `8 n`, `work` `3 m^2 + 2 m` for `m = n + sqre`.
/// Returns `INFO`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlasd0(
    n: i32,
    sqre: i32,
    d: *mut f64,
    e: *mut f64,
    u: *mut f64,
    ldu: i32,
    vt: *mut f64,
    ldvt: i32,
    smlsiz: i32,
    iwork: *mut i32,
    work: *mut f64,
) -> i32 {
    if n < 0 {
        return -1;
    } else if !(0..=1).contains(&sqre) {
        return -2;
    }
    let m = n + sqre;
    if ldu < n {
        return -6;
    } else if ldvt < m {
        return -8;
    } else if smlsiz < 3 {
        return -9;
    }

    // If the input matrix is too small, call DLASDQ to find the SVD.
    if n <= smlsiz {
        return dlasdq(b'U', sqre, n, m, n, 0, d, e, vt, ldvt, u, ldu, u, ldu, work);
    }

    // Set up the computation tree.
    let inode = 1;
    let ndiml = inode + n;
    let ndimr = ndiml + n;
    let idxq = ndimr + n;
    let iwk = idxq + n;
    let (nlvl, nd) = dlasdt(
        n,
        pv!(iwork, inode),
        pv!(iwork, ndiml),
        pv!(iwork, ndimr),
        smlsiz,
    );

    // Solve the leaves directly.
    let ndb1 = (nd + 1) / 2;
    let ncc = 0;
    for i in ndb1..=nd {
        let i1 = i - 1;
        let ic = v!(iwork, inode + i1);
        let nl = v!(iwork, ndiml + i1);
        let nlp1 = nl + 1;
        let nr = v!(iwork, ndimr + i1);
        let nlf = ic - nl;
        let nrf = ic + 1;
        let mut sqrei = 1;
        let info = dlasdq(
            b'U',
            sqrei,
            nl,
            nlp1,
            nl,
            ncc,
            pv!(d, nlf),
            pv!(e, nlf),
            pm!(vt, ldvt, nlf, nlf),
            ldvt,
            pm!(u, ldu, nlf, nlf),
            ldu,
            pm!(u, ldu, nlf, nlf),
            ldu,
            work,
        );
        if info != 0 {
            return info;
        }
        let itemp = idxq + nlf - 2;
        for j in 1..=nl {
            v!(iwork, itemp + j) = j;
        }
        sqrei = if i == nd { sqre } else { 1 };
        let nrp1 = nr + sqrei;
        let info = dlasdq(
            b'U',
            sqrei,
            nr,
            nrp1,
            nr,
            ncc,
            pv!(d, nrf),
            pv!(e, nrf),
            pm!(vt, ldvt, nrf, nrf),
            ldvt,
            pm!(u, ldu, nrf, nrf),
            ldu,
            pm!(u, ldu, nrf, nrf),
            ldu,
            work,
        );
        if info != 0 {
            return info;
        }
        let itemp = idxq + ic;
        for j in 1..=nr {
            v!(iwork, itemp + j - 1) = j;
        }
    }

    // Merge the subproblems from the bottom of the tree up.
    let mut lvl = nlvl;
    while lvl >= 1 {
        let (lf, ll) = if lvl == 1 {
            (1, 1)
        } else {
            let lf = 1i32 << (lvl - 1);
            (lf, 2 * lf - 1)
        };
        for i in lf..=ll {
            let im1 = i - 1;
            let ic = v!(iwork, inode + im1);
            let nl = v!(iwork, ndiml + im1);
            let nr = v!(iwork, ndimr + im1);
            let nlf = ic - nl;
            let sqrei = if sqre == 0 && i == ll { sqre } else { 1 };
            let idxqc = idxq + nlf - 1;
            let alpha = v!(d, ic);
            let beta = v!(e, ic);
            let info = dlasd1(
                nl,
                nr,
                sqrei,
                pv!(d, nlf),
                alpha,
                beta,
                pm!(u, ldu, nlf, nlf),
                ldu,
                pm!(vt, ldvt, nlf, nlf),
                ldvt,
                pv!(iwork, idxqc),
                pv!(iwork, iwk),
                work,
            );
            if info != 0 {
                return info;
            }
        }
        lvl -= 1;
    }
    0
}

/// `DBDSDC` for `COMPQ = 'N'` (singular values only) or `'I'` (`u` and `vt`
/// set to the singular vectors): the SVD of the `n`-by-`n` bidiagonal with
/// diagonal `d` and off-diagonal `e`, upper (`uplo = 'U'`) or lower
/// (`'L'`), the singular values in decreasing order. `work` holds
/// `3 n^2 + 4 n` (`4 n` for `'N'`), `iwork` `8 n`. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dbdsdc(
    uplo: u8,
    compq: u8,
    n: i32,
    d: *mut f64,
    e: *mut f64,
    u: *mut f64,
    ldu: i32,
    vt: *mut f64,
    ldvt: i32,
    work: *mut f64,
    iwork: *mut i32,
) -> i32 {
    let mut iuplo = 0;
    if lsame(uplo, b'U') {
        iuplo = 1;
    }
    if lsame(uplo, b'L') {
        iuplo = 2;
    }
    let icompq = if lsame(compq, b'N') {
        0
    } else if lsame(compq, b'I') {
        2
    } else {
        -1
    };
    if iuplo == 0 {
        return -1;
    } else if icompq < 0 {
        return -2;
    } else if n < 0 {
        return -3;
    } else if ldu < 1 || (icompq == 2 && ldu < n) {
        return -7;
    } else if ldvt < 1 || (icompq == 2 && ldvt < n) {
        return -9;
    }
    if n == 0 {
        return 0;
    }
    let smlsiz = SMLSIZ;
    if n == 1 {
        if icompq == 2 {
            m!(u, ldu, 1, 1) = sign(1.0, v!(d, 1));
            m!(vt, ldvt, 1, 1) = 1.0;
        }
        v!(d, 1) = v!(d, 1).abs();
        return 0;
    }
    let nm1 = n - 1;

    // A lower bidiagonal becomes upper by rotations from the left.
    let mut wstart = 1;
    if iuplo == 2 {
        if icompq == 2 {
            wstart = 2 * n - 1;
        }
        for i in 1..=n - 1 {
            let (cs, sn, r) = dlartg(v!(d, i), v!(e, i));
            v!(d, i) = r;
            v!(e, i) = sn * v!(d, i + 1);
            v!(d, i + 1) = cs * v!(d, i + 1);
            if icompq == 2 {
                v!(work, i) = cs;
                v!(work, nm1 + i) = -sn;
            }
        }
    }

    let info;
    'sort: {
        // Singular values only.
        if icompq == 0 {
            info = dlasdq(b'U', 0, n, 0, 0, 0, d, e, vt, ldvt, u, ldu, u, ldu, work);
            break 'sort;
        }

        // A small problem goes to DLASDQ directly.
        if n <= smlsiz {
            dlaset(b'A', n, n, 0.0, 1.0, u, ldu);
            dlaset(b'A', n, n, 0.0, 1.0, vt, ldvt);
            info = dlasdq(
                b'U',
                0,
                n,
                n,
                n,
                0,
                d,
                e,
                vt,
                ldvt,
                u,
                ldu,
                u,
                ldu,
                pv!(work, wstart),
            );
            break 'sort;
        }

        dlaset(b'A', n, n, 0.0, 1.0, u, ldu);
        dlaset(b'A', n, n, 0.0, 1.0, vt, ldvt);

        // Scale.
        let orgnrm = dlanst_m(n, d, e);
        if orgnrm == 0.0 {
            return 0;
        }
        dlascl_g(orgnrm, 1.0, n, 1, d, n);
        dlascl_g(orgnrm, 1.0, nm1, 1, e, nm1);

        let eps = 0.9 * EPS;
        for i in 1..=n {
            if v!(d, i).abs() < eps {
                v!(d, i) = sign(eps, v!(d, i));
            }
        }

        // Split at negligible off-diagonals and solve each piece.
        let mut start = 1;
        let sqre = 0;
        for i in 1..=nm1 {
            if v!(e, i).abs() < eps || i == nm1 {
                let nsize;
                if i < nm1 {
                    // A subproblem with E(I) small for I < NM1.
                    nsize = i - start + 1;
                } else if v!(e, i).abs() >= eps {
                    // A subproblem with E(NM1) not too small but I = NM1.
                    nsize = n - start + 1;
                } else {
                    // A subproblem with E(NM1) small: the last singular
                    // value is solved by itself.
                    nsize = i - start + 1;
                    m!(u, ldu, n, n) = sign(1.0, v!(d, n));
                    m!(vt, ldvt, n, n) = 1.0;
                    v!(d, n) = v!(d, n).abs();
                }
                let inf = dlasd0(
                    nsize,
                    sqre,
                    pv!(d, start),
                    pv!(e, start),
                    pm!(u, ldu, start, start),
                    ldu,
                    pm!(vt, ldvt, start, start),
                    ldvt,
                    smlsiz,
                    iwork,
                    pv!(work, wstart),
                );
                if inf != 0 {
                    return inf;
                }
                start = i + 1;
            }
        }

        // Unscale.
        dlascl_g(1.0, orgnrm, n, 1, d, n);
        info = 0;
    }

    // Sort the singular values into decreasing order (selection sort, one
    // transposition per vector).
    for ii in 2..=n {
        let i = ii - 1;
        let mut kk = i;
        let mut p = v!(d, i);
        for j in ii..=n {
            if v!(d, j) > p {
                kk = j;
                p = v!(d, j);
            }
        }
        if kk != i {
            v!(d, kk) = v!(d, i);
            v!(d, i) = p;
            if icompq == 2 {
                dswap(n, pm!(u, ldu, 1, i), 1, pm!(u, ldu, 1, kk), 1);
                dswap(n, pm!(vt, ldvt, i, 1), ldvt, pm!(vt, ldvt, kk, 1), ldvt);
            }
        }
    }

    // Undo the rotation of a lower bidiagonal.
    if iuplo == 2 && icompq == 2 {
        dlasr(b'L', b'V', b'B', n, n, work, pv!(work, n), u, ldu);
    }
    info
}
