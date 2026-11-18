//! The singular values and vectors of a bidiagonal matrix by implicit QR
//! iteration: `DBDSQR` (zero-shift and shifted sweeps, `DLASQ1`'s dqds when
//! no vectors are wanted), its 2-by-2 kernels `DLAS2` and `DLASV2`, and
//! `DLASDQ`, which also takes the `(n+1)`-column bidiagonal of a
//! divide-and-conquer leaf. Fused products follow gfortran's GIMPLE for
//! R's libRlapack.

use super::aux::{dlartg, dlascl_g, dlasr, dlasrt};
use super::blas::{dcopy, drot, dscal, dswap};
use super::lasq::dlasq2;
use super::{fma, lsame, sign, EPS, PREC, SAFMIN};

/// `DLAS2`: the singular values `(ssmin, ssmax)` of the upper triangular
/// `[f g; 0 h]`.
pub(crate) fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let fa = f.abs();
    let ga = g.abs();
    let ha = h.abs();
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            let q = fhmx.min(ga) / fhmx.max(ga);
            fhmx.max(ga) * fma(q, q, 1.0).sqrt()
        };
        (0.0, ssmax)
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        // AU = (GA/FHMX)**2 is contracted into both square roots.
        let q = ga / fhmx;
        let c = 2.0 / (fma(q, q, as_ * as_).sqrt() + fma(q, q, at * at).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let p = au * as_;
            let t = au * at;
            let c = 1.0 / (fma(p, p, 1.0).sqrt() + fma(t, t, 1.0).sqrt());
            let s = (fhmn * c) * au;
            (s + s, ga / (c + c))
        }
    }
}

/// `DLASV2`: the singular value decomposition of the upper triangular
/// `[f g; 0 h]`; returns `(ssmin, ssmax, snr, csr, snl, csl)`.
pub(crate) fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let mut ssmin = 0.0;
    let mut ssmax = 0.0;
    let mut clt = 0.0;
    let mut crt = 0.0;
    let mut slt = 0.0;
    let mut srt = 0.0;
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        if ga > fa {
            pmax = 2;
            if (fa / ga) < EPS {
                gasmal = false;
                ssmax = ga;
                ssmin = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                clt = 1.0;
                slt = ht / gt;
                srt = 1.0;
                crt = ft / gt;
            }
        }
        if gasmal {
            let d = fa - ha;
            let mut l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            // TT = T*T is contracted into S.
            let s = fma(t, t, mm).sqrt();
            let r = if l == 0.0 {
                m.abs()
            } else {
                fma(l, l, mm).sqrt()
            };
            let a = 0.5 * (s + r);
            ssmin = ha / a;
            ssmax = fa * a;
            if mm == 0.0 {
                if l == 0.0 {
                    t = sign(2.0, ft) * sign(1.0, gt);
                } else {
                    t = gt / sign(d, ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = fma(t, t, 4.0).sqrt();
            crt = 2.0 / l;
            srt = t / l;
            clt = fma(srt, m, crt) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// `DLASQ1`: the singular values of the bidiagonal with diagonal `d` and
/// superdiagonal `e`, by dqds, in decreasing order in `d`. `work` holds
/// `4 n`. Returns `INFO`.
pub(crate) unsafe fn dlasq1(n: i32, d: *mut f64, e: *mut f64, work: *mut f64) -> i32 {
    if n < 0 {
        return -1;
    }
    if n == 0 {
        return 0;
    }
    if n == 1 {
        v!(d, 1) = v!(d, 1).abs();
        return 0;
    }
    if n == 2 {
        let (sigmn, sigmx) = dlas2(v!(d, 1), v!(e, 1), v!(d, 2));
        v!(d, 1) = sigmx;
        v!(d, 2) = sigmn;
        return 0;
    }
    let mut sigmx = 0.0f64;
    for i in 1..=n - 1 {
        v!(d, i) = v!(d, i).abs();
        sigmx = sigmx.max(v!(e, i).abs());
    }
    v!(d, n) = v!(d, n).abs();
    if sigmx == 0.0 {
        dlasrt(b'D', n, d);
        return 0;
    }
    for i in 1..=n {
        sigmx = sigmx.max(v!(d, i));
    }
    let scale = (PREC / SAFMIN).sqrt();
    dcopy(n, d, 1, work, 2);
    dcopy(n - 1, e, 1, pv!(work, 2), 2);
    dlascl_g(sigmx, scale, 2 * n - 1, 1, work, 2 * n - 1);
    for i in 1..=2 * n - 1 {
        v!(work, i) = v!(work, i) * v!(work, i);
    }
    v!(work, 2 * n) = 0.0;
    let info = dlasq2(n, work);
    if info == 0 {
        for i in 1..=n {
            v!(d, i) = v!(work, i).sqrt();
        }
        dlascl_g(scale, sigmx, n, 1, d, n);
    } else if info == 2 {
        for i in 1..=n {
            v!(d, i) = v!(work, 2 * i - 1).sqrt();
            v!(e, i) = v!(work, 2 * i).sqrt();
        }
        dlascl_g(scale, sigmx, n, 1, d, n);
        dlascl_g(scale, sigmx, n, 1, e, n);
    }
    info
}

/// `DBDSQR`: the SVD `B = Q S P'` of the `n`-by-`n` bidiagonal with
/// diagonal `d` and off-diagonal `e` (upper for `uplo = 'U'`, lower for
/// `'L'`): the singular values in decreasing order in `d`, `VT := P' VT`
/// (`ncvt` columns), `U := U Q` (`nru` rows) and `C := Q' C` (`ncc`
/// columns). `work` holds `4 n` (`4 (n - 1)` with vectors). Returns
/// `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dbdsqr(
    uplo: u8,
    n: i32,
    ncvt: i32,
    nru: i32,
    ncc: i32,
    d: *mut f64,
    e: *mut f64,
    vt: *mut f64,
    ldvt: i32,
    u: *mut f64,
    ldu: i32,
    c: *mut f64,
    ldc: i32,
    work: *mut f64,
) -> i32 {
    const HNDRTH: f64 = 0.01;
    const TEN: f64 = 10.0;
    const HNDRD: f64 = 100.0;
    const MEIGTH: f64 = -0.125;
    const MAXITR: i32 = 6;

    let lower = lsame(uplo, b'L');
    if !lsame(uplo, b'U') && !lower {
        return -1;
    } else if n < 0 {
        return -2;
    } else if ncvt < 0 {
        return -3;
    } else if nru < 0 {
        return -4;
    } else if ncc < 0 {
        return -5;
    } else if (ncvt == 0 && ldvt < 1) || (ncvt > 0 && ldvt < 1.max(n)) {
        return -9;
    } else if ldu < 1.max(nru) {
        return -11;
    } else if (ncc == 0 && ldc < 1) || (ncc > 0 && ldc < 1.max(n)) {
        return -13;
    }
    if n == 0 {
        return 0;
    }
    if n > 1 {
        let rotate = ncvt > 0 || nru > 0 || ncc > 0;
        if !rotate {
            let info = dlasq1(n, d, e, work);
            if info != 2 {
                return info;
            }
        }
        let nm1 = n - 1;
        let nm12 = nm1 + nm1;
        let nm13 = nm12 + nm1;
        let mut idir = 0;
        let eps = EPS;
        let unfl = SAFMIN;

        if lower {
            for i in 1..=n - 1 {
                let (cs, sn, r) = dlartg(v!(d, i), v!(e, i));
                v!(d, i) = r;
                v!(e, i) = sn * v!(d, i + 1);
                v!(d, i + 1) = cs * v!(d, i + 1);
                v!(work, i) = cs;
                v!(work, nm1 + i) = sn;
            }
            if nru > 0 {
                dlasr(b'R', b'V', b'F', nru, n, work, pv!(work, n), u, ldu);
            }
            if ncc > 0 {
                dlasr(b'L', b'V', b'F', n, ncc, work, pv!(work, n), c, ldc);
            }
        }

        // EPS**MEIGTH is a run-time pow in R's binary.
        let tolmul = TEN.max(HNDRD.min(std::hint::black_box(eps).powf(MEIGTH)));
        let tol = tolmul * eps;

        let mut smax = 0.0f64;
        for i in 1..=n {
            smax = smax.max(v!(d, i).abs());
        }
        for i in 1..=n - 1 {
            smax = smax.max(v!(e, i).abs());
        }
        let mut smin = 0.0f64;
        let thresh;
        let nf = n as f64;
        if tol >= 0.0 {
            let mut sminoa = v!(d, 1).abs();
            if sminoa != 0.0 {
                let mut mu = sminoa;
                for i in 2..=n {
                    mu = v!(d, i).abs() * (mu / (mu + v!(e, i - 1).abs()));
                    sminoa = sminoa.min(mu);
                    if sminoa == 0.0 {
                        break;
                    }
                }
            }
            sminoa /= nf.sqrt();
            thresh = (tol * sminoa).max(MAXITR as f64 * (nf * (nf * unfl)));
        } else {
            thresh = (tol.abs() * smax).max(MAXITR as f64 * (nf * (nf * unfl)));
        }

        let maxitdivn = MAXITR * n;
        let mut iterdivn = 0;
        let mut iter = -1;
        let mut oldll = -1;
        let mut oldm = -1;
        let mut m = n;

        let converged = 'sweep: loop {
            if m <= 1 {
                break true;
            }
            if iter >= n {
                iter -= n;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    break false;
                }
            }
            if tol < 0.0 && v!(d, m).abs() <= thresh {
                v!(d, m) = 0.0;
            }
            smax = v!(d, m).abs();
            let mut ll = 0;
            let mut split = false;
            for lll in 1..=m - 1 {
                ll = m - lll;
                let abss = v!(d, ll).abs();
                let abse = v!(e, ll).abs();
                if tol < 0.0 && abss <= thresh {
                    v!(d, ll) = 0.0;
                }
                if abse <= thresh {
                    split = true;
                    break;
                }
                smax = smax.max(abss).max(abse);
            }
            if split {
                v!(e, ll) = 0.0;
                if ll == m - 1 {
                    m -= 1;
                    continue 'sweep;
                }
            } else {
                ll = 0;
            }
            ll += 1;

            if ll == m - 1 {
                let (sigmn, sigmx, sinr, cosr, sinl, cosl) =
                    dlasv2(v!(d, m - 1), v!(e, m - 1), v!(d, m));
                v!(d, m - 1) = sigmx;
                v!(e, m - 1) = 0.0;
                v!(d, m) = sigmn;
                if ncvt > 0 {
                    drot(
                        ncvt,
                        pm!(vt, ldvt, m - 1, 1),
                        ldvt,
                        pm!(vt, ldvt, m, 1),
                        ldvt,
                        cosr,
                        sinr,
                    );
                }
                if nru > 0 {
                    drot(
                        nru,
                        pm!(u, ldu, 1, m - 1),
                        1,
                        pm!(u, ldu, 1, m),
                        1,
                        cosl,
                        sinl,
                    );
                }
                if ncc > 0 {
                    drot(
                        ncc,
                        pm!(c, ldc, m - 1, 1),
                        ldc,
                        pm!(c, ldc, m, 1),
                        ldc,
                        cosl,
                        sinl,
                    );
                }
                m -= 2;
                continue 'sweep;
            }

            if ll > oldm || m < oldll {
                idir = if v!(d, ll).abs() >= v!(d, m).abs() {
                    1
                } else {
                    2
                };
            }

            if idir == 1 {
                if v!(e, m - 1).abs() <= tol.abs() * v!(d, m).abs()
                    || (tol < 0.0 && v!(e, m - 1).abs() <= thresh)
                {
                    v!(e, m - 1) = 0.0;
                    continue 'sweep;
                }
                if tol >= 0.0 {
                    let mut mu = v!(d, ll).abs();
                    smin = mu;
                    for lll in ll..=m - 1 {
                        if v!(e, lll).abs() <= tol * mu {
                            v!(e, lll) = 0.0;
                            continue 'sweep;
                        }
                        mu = v!(d, lll + 1).abs() * (mu / (mu + v!(e, lll).abs()));
                        smin = smin.min(mu);
                    }
                }
            } else {
                if v!(e, ll).abs() <= tol.abs() * v!(d, ll).abs()
                    || (tol < 0.0 && v!(e, ll).abs() <= thresh)
                {
                    v!(e, ll) = 0.0;
                    continue 'sweep;
                }
                if tol >= 0.0 {
                    let mut mu = v!(d, m).abs();
                    smin = mu;
                    let mut lll = m - 1;
                    while lll >= ll {
                        if v!(e, lll).abs() <= tol * mu {
                            v!(e, lll) = 0.0;
                            continue 'sweep;
                        }
                        mu = v!(d, lll).abs() * (mu / (mu + v!(e, lll).abs()));
                        smin = smin.min(mu);
                        lll -= 1;
                    }
                }
            }
            oldll = ll;
            oldm = m;

            let mut shift;
            if tol >= 0.0 && nf * tol * (smin / smax) <= eps.max(HNDRTH * tol) {
                shift = 0.0;
            } else {
                let sll;
                if idir == 1 {
                    sll = v!(d, ll).abs();
                    shift = dlas2(v!(d, m - 1), v!(e, m - 1), v!(d, m)).0;
                } else {
                    sll = v!(d, m).abs();
                    shift = dlas2(v!(d, ll), v!(e, ll), v!(d, ll + 1)).0;
                }
                if sll > 0.0 {
                    let q = shift / sll;
                    if q * q < eps {
                        shift = 0.0;
                    }
                }
            }

            iter = iter + m - ll;

            if shift == 0.0 {
                if idir == 1 {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in ll..=m - 1 {
                        let (c1, sn, r) = dlartg(v!(d, i) * cs, v!(e, i));
                        cs = c1;
                        if i > ll {
                            v!(e, i - 1) = oldsn * r;
                        }
                        let (oc, os, di) = dlartg(oldcs * r, v!(d, i + 1) * sn);
                        oldcs = oc;
                        oldsn = os;
                        v!(d, i) = di;
                        v!(work, i - ll + 1) = cs;
                        v!(work, i - ll + 1 + nm1) = sn;
                        v!(work, i - ll + 1 + nm12) = oldcs;
                        v!(work, i - ll + 1 + nm13) = oldsn;
                    }
                    let h = v!(d, m) * cs;
                    v!(d, m) = h * oldcs;
                    v!(e, m - 1) = h * oldsn;
                    if ncvt > 0 {
                        dlasr(
                            b'L',
                            b'V',
                            b'F',
                            m - ll + 1,
                            ncvt,
                            work,
                            pv!(work, n),
                            pm!(vt, ldvt, ll, 1),
                            ldvt,
                        );
                    }
                    if nru > 0 {
                        dlasr(
                            b'R',
                            b'V',
                            b'F',
                            nru,
                            m - ll + 1,
                            pv!(work, nm12 + 1),
                            pv!(work, nm13 + 1),
                            pm!(u, ldu, 1, ll),
                            ldu,
                        );
                    }
                    if ncc > 0 {
                        dlasr(
                            b'L',
                            b'V',
                            b'F',
                            m - ll + 1,
                            ncc,
                            pv!(work, nm12 + 1),
                            pv!(work, nm13 + 1),
                            pm!(c, ldc, ll, 1),
                            ldc,
                        );
                    }
                    if v!(e, m - 1).abs() <= thresh {
                        v!(e, m - 1) = 0.0;
                    }
                } else {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    let mut i = m;
                    while i >= ll + 1 {
                        let (c1, sn, r) = dlartg(v!(d, i) * cs, v!(e, i - 1));
                        cs = c1;
                        if i < m {
                            v!(e, i) = oldsn * r;
                        }
                        let (oc, os, di) = dlartg(oldcs * r, v!(d, i - 1) * sn);
                        oldcs = oc;
                        oldsn = os;
                        v!(d, i) = di;
                        v!(work, i - ll) = cs;
                        v!(work, i - ll + nm1) = -sn;
                        v!(work, i - ll + nm12) = oldcs;
                        v!(work, i - ll + nm13) = -oldsn;
                        i -= 1;
                    }
                    let h = v!(d, ll) * cs;
                    v!(d, ll) = h * oldcs;
                    v!(e, ll) = h * oldsn;
                    if ncvt > 0 {
                        dlasr(
                            b'L',
                            b'V',
                            b'B',
                            m - ll + 1,
                            ncvt,
                            pv!(work, nm12 + 1),
                            pv!(work, nm13 + 1),
                            pm!(vt, ldvt, ll, 1),
                            ldvt,
                        );
                    }
                    if nru > 0 {
                        dlasr(
                            b'R',
                            b'V',
                            b'B',
                            nru,
                            m - ll + 1,
                            work,
                            pv!(work, n),
                            pm!(u, ldu, 1, ll),
                            ldu,
                        );
                    }
                    if ncc > 0 {
                        dlasr(
                            b'L',
                            b'V',
                            b'B',
                            m - ll + 1,
                            ncc,
                            work,
                            pv!(work, n),
                            pm!(c, ldc, ll, 1),
                            ldc,
                        );
                    }
                    if v!(e, ll).abs() <= thresh {
                        v!(e, ll) = 0.0;
                    }
                }
            } else if idir == 1 {
                let mut f = (v!(d, ll).abs() - shift) * (sign(1.0, v!(d, ll)) + shift / v!(d, ll));
                let mut g = v!(e, ll);
                for i in ll..=m - 1 {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i > ll {
                        v!(e, i - 1) = r;
                    }
                    let di = v!(d, i);
                    let ei = v!(e, i);
                    f = fma(cosr, di, sinr * ei);
                    v!(e, i) = fma(cosr, ei, -(sinr * di));
                    g = sinr * v!(d, i + 1);
                    v!(d, i + 1) = cosr * v!(d, i + 1);
                    let (cosl, sinl, r) = dlartg(f, g);
                    v!(d, i) = r;
                    let ei = v!(e, i);
                    let di1 = v!(d, i + 1);
                    f = fma(cosl, ei, sinl * di1);
                    v!(d, i + 1) = fma(cosl, di1, -(sinl * ei));
                    if i < m - 1 {
                        g = sinl * v!(e, i + 1);
                        v!(e, i + 1) = cosl * v!(e, i + 1);
                    }
                    v!(work, i - ll + 1) = cosr;
                    v!(work, i - ll + 1 + nm1) = sinr;
                    v!(work, i - ll + 1 + nm12) = cosl;
                    v!(work, i - ll + 1 + nm13) = sinl;
                }
                v!(e, m - 1) = f;
                if ncvt > 0 {
                    dlasr(
                        b'L',
                        b'V',
                        b'F',
                        m - ll + 1,
                        ncvt,
                        work,
                        pv!(work, n),
                        pm!(vt, ldvt, ll, 1),
                        ldvt,
                    );
                }
                if nru > 0 {
                    dlasr(
                        b'R',
                        b'V',
                        b'F',
                        nru,
                        m - ll + 1,
                        pv!(work, nm12 + 1),
                        pv!(work, nm13 + 1),
                        pm!(u, ldu, 1, ll),
                        ldu,
                    );
                }
                if ncc > 0 {
                    dlasr(
                        b'L',
                        b'V',
                        b'F',
                        m - ll + 1,
                        ncc,
                        pv!(work, nm12 + 1),
                        pv!(work, nm13 + 1),
                        pm!(c, ldc, ll, 1),
                        ldc,
                    );
                }
                if v!(e, m - 1).abs() <= thresh {
                    v!(e, m - 1) = 0.0;
                }
            } else {
                let mut f = (v!(d, m).abs() - shift) * (sign(1.0, v!(d, m)) + shift / v!(d, m));
                let mut g = v!(e, m - 1);
                let mut i = m;
                while i >= ll + 1 {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i < m {
                        v!(e, i) = r;
                    }
                    let di = v!(d, i);
                    let ei1 = v!(e, i - 1);
                    f = fma(cosr, di, sinr * ei1);
                    v!(e, i - 1) = fma(cosr, ei1, -(sinr * di));
                    g = sinr * v!(d, i - 1);
                    v!(d, i - 1) = cosr * v!(d, i - 1);
                    let (cosl, sinl, r) = dlartg(f, g);
                    v!(d, i) = r;
                    let ei1 = v!(e, i - 1);
                    let di1 = v!(d, i - 1);
                    f = fma(cosl, ei1, sinl * di1);
                    v!(d, i - 1) = fma(cosl, di1, -(sinl * ei1));
                    if i > ll + 1 {
                        g = sinl * v!(e, i - 2);
                        v!(e, i - 2) = cosl * v!(e, i - 2);
                    }
                    v!(work, i - ll) = cosr;
                    v!(work, i - ll + nm1) = -sinr;
                    v!(work, i - ll + nm12) = cosl;
                    v!(work, i - ll + nm13) = -sinl;
                    i -= 1;
                }
                v!(e, ll) = f;
                if v!(e, ll).abs() <= thresh {
                    v!(e, ll) = 0.0;
                }
                if ncvt > 0 {
                    dlasr(
                        b'L',
                        b'V',
                        b'B',
                        m - ll + 1,
                        ncvt,
                        pv!(work, nm12 + 1),
                        pv!(work, nm13 + 1),
                        pm!(vt, ldvt, ll, 1),
                        ldvt,
                    );
                }
                if nru > 0 {
                    dlasr(
                        b'R',
                        b'V',
                        b'B',
                        nru,
                        m - ll + 1,
                        work,
                        pv!(work, n),
                        pm!(u, ldu, 1, ll),
                        ldu,
                    );
                }
                if ncc > 0 {
                    dlasr(
                        b'L',
                        b'V',
                        b'B',
                        m - ll + 1,
                        ncc,
                        work,
                        pv!(work, n),
                        pm!(c, ldc, ll, 1),
                        ldc,
                    );
                }
            }
        };
        if !converged {
            let mut info = 0;
            for i in 1..=n - 1 {
                if v!(e, i) != 0.0 {
                    info += 1;
                }
            }
            return info;
        }
    }

    // All singular values converged: make them positive, then sort them
    // into decreasing order.
    for i in 1..=n {
        if v!(d, i) == 0.0 {
            v!(d, i) = 0.0;
        }
        if v!(d, i) < 0.0 {
            v!(d, i) = -v!(d, i);
            if ncvt > 0 {
                dscal(ncvt, -1.0, pm!(vt, ldvt, i, 1), ldvt);
            }
        }
    }
    for i in 1..=n - 1 {
        let mut isub = 1;
        let mut smin = v!(d, 1);
        for j in 2..=n + 1 - i {
            if v!(d, j) <= smin {
                isub = j;
                smin = v!(d, j);
            }
        }
        if isub != n + 1 - i {
            v!(d, isub) = v!(d, n + 1 - i);
            v!(d, n + 1 - i) = smin;
            if ncvt > 0 {
                dswap(
                    ncvt,
                    pm!(vt, ldvt, isub, 1),
                    ldvt,
                    pm!(vt, ldvt, n + 1 - i, 1),
                    ldvt,
                );
            }
            if nru > 0 {
                dswap(nru, pm!(u, ldu, 1, isub), 1, pm!(u, ldu, 1, n + 1 - i), 1);
            }
            if ncc > 0 {
                dswap(
                    ncc,
                    pm!(c, ldc, isub, 1),
                    ldc,
                    pm!(c, ldc, n + 1 - i, 1),
                    ldc,
                );
            }
        }
    }
    0
}

/// `DLASDQ`: the SVD of the bidiagonal with diagonal `d` and off-diagonal
/// `e`, upper (`uplo = 'U'`) or lower (`'L'`), square (`sqre = 0`) or with
/// one extra column (`'U'`) or row (`'L'`) (`sqre = 1`, `e(n)` used); the
/// singular values in ascending order in `d`, `VT := P' VT`, `U := U Q`,
/// `C := Q' C`. `work` holds `4 n`. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlasdq(
    uplo: u8,
    sqre: i32,
    n: i32,
    ncvt: i32,
    nru: i32,
    ncc: i32,
    d: *mut f64,
    e: *mut f64,
    vt: *mut f64,
    ldvt: i32,
    u: *mut f64,
    ldu: i32,
    c: *mut f64,
    ldc: i32,
    work: *mut f64,
) -> i32 {
    let mut iuplo = 0;
    if lsame(uplo, b'U') {
        iuplo = 1;
    }
    if lsame(uplo, b'L') {
        iuplo = 2;
    }
    if iuplo == 0 {
        return -1;
    } else if !(0..=1).contains(&sqre) {
        return -2;
    } else if n < 0 {
        return -3;
    } else if ncvt < 0 {
        return -4;
    } else if nru < 0 {
        return -5;
    } else if ncc < 0 {
        return -6;
    } else if (ncvt == 0 && ldvt < 1) || (ncvt > 0 && ldvt < 1.max(n)) {
        return -10;
    } else if ldu < 1.max(nru) {
        return -12;
    } else if (ncc == 0 && ldc < 1) || (ncc > 0 && ldc < 1.max(n)) {
        return -14;
    }
    if n == 0 {
        return 0;
    }
    let rotate = ncvt > 0 || nru > 0 || ncc > 0;
    let np1 = n + 1;
    let mut sqre1 = sqre;

    // An upper bidiagonal with an extra column becomes lower by rotations
    // from the right.
    if iuplo == 1 && sqre1 == 1 {
        for i in 1..=n - 1 {
            let (cs, sn, r) = dlartg(v!(d, i), v!(e, i));
            v!(d, i) = r;
            v!(e, i) = sn * v!(d, i + 1);
            v!(d, i + 1) = cs * v!(d, i + 1);
            if rotate {
                v!(work, i) = cs;
                v!(work, n + i) = sn;
            }
        }
        let (cs, sn, r) = dlartg(v!(d, n), v!(e, n));
        v!(d, n) = r;
        v!(e, n) = 0.0;
        if rotate {
            v!(work, n) = cs;
            v!(work, n + n) = sn;
        }
        iuplo = 2;
        sqre1 = 0;
        if ncvt > 0 {
            dlasr(b'L', b'V', b'F', np1, ncvt, work, pv!(work, np1), vt, ldvt);
        }
    }

    // A lower bidiagonal becomes upper by rotations from the left.
    if iuplo == 2 {
        for i in 1..=n - 1 {
            let (cs, sn, r) = dlartg(v!(d, i), v!(e, i));
            v!(d, i) = r;
            v!(e, i) = sn * v!(d, i + 1);
            v!(d, i + 1) = cs * v!(d, i + 1);
            if rotate {
                v!(work, i) = cs;
                v!(work, n + i) = sn;
            }
        }
        if sqre1 == 1 {
            let (cs, sn, r) = dlartg(v!(d, n), v!(e, n));
            v!(d, n) = r;
            if rotate {
                v!(work, n) = cs;
                v!(work, n + n) = sn;
            }
        }
        if nru > 0 {
            if sqre1 == 0 {
                dlasr(b'R', b'V', b'F', nru, n, work, pv!(work, np1), u, ldu);
            } else {
                dlasr(b'R', b'V', b'F', nru, np1, work, pv!(work, np1), u, ldu);
            }
        }
        if ncc > 0 {
            if sqre1 == 0 {
                dlasr(b'L', b'V', b'F', n, ncc, work, pv!(work, np1), c, ldc);
            } else {
                dlasr(b'L', b'V', b'F', np1, ncc, work, pv!(work, np1), c, ldc);
            }
        }
    }

    let info = dbdsqr(
        b'U', n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work,
    );

    // Sort the singular values into ascending order, one transposition per
    // singular vector.
    for i in 1..=n {
        let mut isub = i;
        let mut smin = v!(d, i);
        for j in i + 1..=n {
            if v!(d, j) < smin {
                isub = j;
                smin = v!(d, j);
            }
        }
        if isub != i {
            v!(d, isub) = v!(d, i);
            v!(d, i) = smin;
            if ncvt > 0 {
                dswap(
                    ncvt,
                    pm!(vt, ldvt, isub, 1),
                    ldvt,
                    pm!(vt, ldvt, i, 1),
                    ldvt,
                );
            }
            if nru > 0 {
                dswap(nru, pm!(u, ldu, 1, isub), 1, pm!(u, ldu, 1, i), 1);
            }
            if ncc > 0 {
                dswap(ncc, pm!(c, ldc, isub, 1), ldc, pm!(c, ldc, i, 1), ldc);
            }
        }
    }
    info
}
