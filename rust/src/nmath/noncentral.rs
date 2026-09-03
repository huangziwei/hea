#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::consts::{DBL_MIN, M_LN2, M_LN_SQRT_2PI};
use super::discrete::dbeta_scalar;
use super::gamma::{dgamma_scalar, pgamma_scalar, qgamma_scalar, r_dt_qiv, r_log1_exp};
use super::lgamma::lgammafn;
use super::loader::dpois_raw;
use super::norm::{dnorm5_scalar, dt0, dt1, pnorm5_scalar, qnorm5_scalar};
use super::tf::{dt_scalar, pt_scalar, qt_scalar};
use super::toms708::{bratio, lbeta_scalar, logspace_add, pbeta_scalar};

const PNCH_DBL_MIN_EXP: f64 = M_LN2 * (-1021.0); // M_LN2 * DBL_MIN_EXP
const M_SQRT_2DPI: f64 = 0.797884560802865355879892119869; // sqrt(2/pi)
const M_LN_SQRT_PI: f64 = 0.572364942924700087071713675677; // log(sqrt(pi))

#[inline]
fn pchisq(x: f64, df: f64, lower_tail: bool, log_p: bool) -> f64 {
    pgamma_scalar(x, df / 2.0, 2.0, lower_tail, log_p)
}
#[inline]
fn dchisq(x: f64, df: f64, give_log: bool) -> f64 {
    dgamma_scalar(x, df / 2.0, 2.0, give_log)
}
#[inline]
fn qchisq(p: f64, df: f64, lower_tail: bool, log_p: bool) -> f64 {
    qgamma_scalar(p, df / 2.0, 2.0, lower_tail, log_p)
}

#[inline]
fn r_dt_val(x: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        if log_p {
            x.ln()
        } else {
            x
        }
    } else if log_p {
        (-x).ln_1p()
    } else {
        0.5 - x + 0.5
    }
}

pub(crate) fn pnchisq_raw(
    x: f64,
    f: f64,
    theta: f64,
    errmax: f64,
    reltol: f64,
    itrmax: i32,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x <= 0.0 {
        if x == 0.0 && f == 0.0 {
            let l = -0.5 * theta;
            if lower_tail {
                return if log_p { l } else { l.exp() };
            }
            return if log_p { r_log1_exp(l) } else { -l.exp_m1() };
        }
        return dt0(lower_tail, log_p);
    }
    if !x.is_finite() {
        return dt1(lower_tail, log_p);
    }

    if theta < 80.0 {
        if lower_tail
            && f > 0.0
            && x.ln() < M_LN2 + 2.0 / f * (lgammafn(f / 2.0 + 1.0) + PNCH_DBL_MIN_EXP)
        {
            let lam = 0.5 * theta;
            let mut pr = -lam;
            let log_lam = lam.ln();
            let mut sum = f64::NEG_INFINITY;
            let mut sum2 = f64::NEG_INFINITY;
            let mut i: i32 = 0;
            while i < 110 {
                sum2 = logspace_add(sum2, pr);
                sum = logspace_add(sum, pr + pchisq(x, f + 2.0 * i as f64, lower_tail, true));
                if sum2 >= -1e-15 {
                    break;
                }
                i += 1;
                pr += log_lam - (i as f64).ln();
            }
            let ans = sum - sum2;
            return if log_p { ans } else { ans.exp() };
        }
        let lam = 0.5 * theta;
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut pr = (-lam).exp();
        let mut i: i32 = 0;
        while i < 110 {
            sum2 += pr;
            sum += pr * pchisq(x, f + 2.0 * i as f64, lower_tail, false);
            if sum2 >= 1.0 - 1e-15 {
                break;
            }
            i += 1;
            pr *= lam / i as f64;
        }
        let ans = sum / sum2;
        return if log_p { ans.ln() } else { ans };
    }

    let lam = 0.5 * theta;
    let mut lam_sml = -lam < PNCH_DBL_MIN_EXP;
    let mut l_lam = -1.0f64;
    let mut u;
    let mut lu;
    if lam_sml {
        u = 0.0;
        lu = -lam;
        l_lam = lam.ln();
    } else {
        u = (-lam).exp();
        lu = -1.0;
    }
    let mut v = u;
    let x2 = 0.5 * x;
    let f2 = 0.5 * f;
    let mut f_x_2n = f - x;

    let mut t = x2 - f2;
    let mut lt;
    if f2 * f64::EPSILON > 0.125 && t.abs() < f64::EPSILON.sqrt() * f2 {
        lt = (1.0 - t) * (2.0 - t / (f2 + 1.0)) - M_LN_SQRT_2PI - 0.5 * (f2 + 1.0).ln();
    } else {
        lt = f2 * x2.ln() - x2 - lgammafn(f2 + 1.0);
    }

    let mut l_x = -1.0f64;
    let mut t_sml = lt < PNCH_DBL_MIN_EXP;
    let mut term;
    let mut ans;
    if t_sml {
        if x > f + theta + 5.0 * (2.0 * (f + 2.0 * theta)).sqrt() {
            return dt1(lower_tail, log_p);
        }
        l_x = x.ln();
        ans = 0.0;
        term = 0.0;
        t = 0.0;
    } else {
        t = lt.exp();
        term = v * t;
        ans = term;
    }

    let mut n: i32 = 1;
    let mut f_2n = f + 2.0;
    f_x_2n += 2.0;
    while n <= itrmax {
        if f_x_2n > 0.0 {
            let bound = t * x / f_x_2n;
            if bound <= errmax && term <= reltol * ans {
                break;
            }
        }
        if lam_sml {
            lu += l_lam - (n as f64).ln();
            if lu >= PNCH_DBL_MIN_EXP {
                u = lu.exp();
                v = u;
                lam_sml = false;
            }
        } else {
            u *= lam / n as f64;
            v += u;
        }
        if t_sml {
            lt += l_x - f_2n.ln();
            if lt >= PNCH_DBL_MIN_EXP {
                t = lt.exp();
                t_sml = false;
            }
        } else {
            t *= x / f_2n;
        }
        if !lam_sml && !t_sml {
            term = v * t;
            ans += term;
        }
        n += 1;
        f_2n += 2.0;
        f_x_2n += 2.0;
    }
    r_dt_val(ans, lower_tail, log_p)
}

pub(crate) fn pnchisq_scalar(x: f64, df: f64, ncp: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || df.is_nan() || ncp.is_nan() {
        return x + df + ncp;
    }
    if !df.is_finite() || !ncp.is_finite() {
        return f64::NAN;
    }
    if df < 0.0 || ncp < 0.0 {
        return f64::NAN;
    }
    let mut ans = pnchisq_raw(
        x,
        df,
        ncp,
        1e-12,
        8.0 * f64::EPSILON,
        1_000_000,
        lower_tail,
        log_p,
    );
    if x <= 0.0 || x == f64::INFINITY {
        return ans;
    }
    if ncp >= 80.0 {
        if lower_tail {
            ans = ans.min(if log_p { 0.0 } else { 1.0 });
        } else if !log_p && ans < 0.0 {
            ans = 0.0;
        }
    }
    if !log_p || ans < -1e-8 {
        return ans;
    }
    ans = pnchisq_raw(
        x,
        df,
        ncp,
        1e-12,
        8.0 * f64::EPSILON,
        1_000_000,
        !lower_tail,
        false,
    );
    (-ans).ln_1p()
}

pub(crate) fn dnchisq_scalar(x: f64, df: f64, ncp: f64, give_log: bool) -> f64 {
    let eps = 5e-15;
    if x.is_nan() || df.is_nan() || ncp.is_nan() {
        return x + df + ncp;
    }
    if !df.is_finite() || !ncp.is_finite() || ncp < 0.0 || df < 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if x == 0.0 && df < 2.0 {
        return f64::INFINITY;
    }
    if ncp == 0.0 {
        return if df > 0.0 {
            dchisq(x, df, give_log)
        } else if give_log {
            f64::NEG_INFINITY
        } else {
            0.0
        };
    }
    if x == f64::INFINITY {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }

    let ncp2 = 0.5 * ncp;
    let mut imax = ((-(2.0 + df) + ((2.0 - df) * (2.0 - df) + 4.0 * ncp * x).sqrt()) / 4.0).ceil();
    if imax < 0.0 {
        imax = 0.0;
    }
    let dfmid;
    let mid;
    if imax.is_finite() {
        dfmid = df + 2.0 * imax;
        mid = dpois_raw(imax, ncp2, false) * dchisq(x, dfmid, false);
    } else {
        dfmid = 0.0; // unused (mid==0 path below)
        mid = 0.0;
    }
    if mid == 0.0 {
        if give_log || ncp > 1000.0 {
            let nl = df + ncp;
            let ic = nl / (nl + ncp);
            return dchisq(x * ic, nl * ic, give_log);
        }
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }

    let mut sum = mid;
    let mut term = mid;
    let mut dfv = dfmid;
    let mut i = imax;
    let x2 = x * ncp2;
    loop {
        i += 1.0;
        let q = x2 / i / dfv;
        dfv += 2.0;
        term *= q;
        sum += term;
        if !(q >= 1.0 || term * q > (1.0 - q) * eps || term > 1e-10 * sum) {
            break;
        }
    }
    term = mid;
    dfv = dfmid;
    i = imax;
    while i != 0.0 {
        dfv -= 2.0;
        let q = i * dfv / x2;
        i -= 1.0;
        term *= q;
        sum += term;
        if q < 1.0 && term * q <= (1.0 - q) * eps {
            break;
        }
    }
    if give_log {
        sum.ln()
    } else {
        sum
    }
}

pub(crate) fn qnchisq_scalar(p: f64, df: f64, ncp: f64, mut lower_tail: bool, log_p: bool) -> f64 {
    let accu = 1e-13;
    let racc = 4.0 * f64::EPSILON;
    let eps = 1e-11;
    let reps = 1e-10;
    if p.is_nan() || df.is_nan() || ncp.is_nan() {
        return p + df + ncp;
    }
    if !df.is_finite() {
        return f64::NAN;
    }
    if df < 0.0 || ncp < 0.0 {
        return f64::NAN;
    }
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { f64::INFINITY } else { 0.0 };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail { 0.0 } else { f64::INFINITY };
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { 0.0 } else { f64::INFINITY };
        }
        if p == 1.0 {
            return if lower_tail { f64::INFINITY } else { 0.0 };
        }
    }

    let pp0 = if log_p { p.exp() } else { p }; // R_D_qIv(p)
    if pp0 > 1.0 - f64::EPSILON {
        return if lower_tail { f64::INFINITY } else { 0.0 };
    }

    let b = (ncp * ncp) / (df + 3.0 * ncp);
    let c = (df + 3.0 * ncp) / (df + 2.0 * ncp);
    let ff = (df + 2.0 * ncp) / (c * c);
    let mut ux = b + c * qchisq(p, ff, lower_tail, log_p);
    if ux <= 0.0 {
        ux = 1.0;
    }
    let ux0 = ux;

    let pval;
    if !lower_tail && ncp >= 80.0 {
        pval = if log_p { -p.exp_m1() } else { 0.5 - p + 0.5 };
        lower_tail = true;
    } else {
        pval = pp0;
    }

    let mut pp = (1.0 - f64::EPSILON).min(pval * (1.0 + eps));
    let mut lx;
    if lower_tail {
        while ux < f64::MAX && pnchisq_raw(ux, df, ncp, eps, reps, 10000, true, false) < pp {
            ux *= 2.0;
        }
        pp = pval * (1.0 - eps);
        lx = ux0.min(f64::MAX);
        while lx > DBL_MIN && pnchisq_raw(lx, df, ncp, eps, reps, 10000, true, false) > pp {
            lx *= 0.5;
        }
    } else {
        while ux < f64::MAX && pnchisq_raw(ux, df, ncp, eps, reps, 10000, false, false) > pp {
            ux *= 2.0;
        }
        pp = pval * (1.0 - eps);
        lx = ux0.min(f64::MAX);
        while lx > DBL_MIN && pnchisq_raw(lx, df, ncp, eps, reps, 10000, false, false) < pp {
            lx *= 0.5;
        }
    }

    let mut nx;
    if lower_tail {
        loop {
            nx = 0.5 * (lx + ux);
            if pnchisq_raw(nx, df, ncp, accu, racc, 100000, true, false) > pval {
                ux = nx;
            } else {
                lx = nx;
            }
            if !((ux - lx) / nx > accu) {
                break;
            }
        }
    } else {
        loop {
            nx = 0.5 * (lx + ux);
            if pnchisq_raw(nx, df, ncp, accu, racc, 100000, false, false) < pval {
                ux = nx;
            } else {
                lx = nx;
            }
            if !((ux - lx) / nx > accu) {
                break;
            }
        }
    }
    0.5 * (ux + lx)
}

pub(crate) fn pnt_scalar(t: f64, df: f64, ncp: f64, mut lower_tail: bool, log_p: bool) -> f64 {
    let itrmax = 1000;
    let errmax = 1e-12;
    if df <= 0.0 {
        return f64::NAN;
    }
    if ncp == 0.0 {
        return pt_scalar(t, df, lower_tail, log_p);
    }
    if !t.is_finite() {
        return if t < 0.0 {
            dt0(lower_tail, log_p)
        } else {
            dt1(lower_tail, log_p)
        };
    }
    let negdel;
    let tt;
    let del;
    if t >= 0.0 {
        negdel = false;
        tt = t;
        del = ncp;
    } else {
        if ncp > 40.0 && (!log_p || !lower_tail) {
            return dt0(lower_tail, log_p);
        }
        negdel = true;
        tt = -t;
        del = -ncp;
    }

    if df > 4e5 || del * del > 2.0 * M_LN2 * 1021.0 {
        let s = 1.0 / (4.0 * df);
        return pnorm5_scalar(
            tt * (1.0 - s),
            del,
            (1.0 + tt * tt * 2.0 * s).sqrt(),
            lower_tail != negdel,
            log_p,
        );
    }

    let mut x = t * t;
    let mut rxb = df / (x + df);
    x /= x + df;
    let mut tnc;
    if x > 0.0 {
        let lambda = del * del;
        let mut p = 0.5 * (-0.5 * lambda).exp();
        if p == 0.0 {
            return dt0(lower_tail, log_p);
        }
        let mut q = M_SQRT_2DPI * p * del;
        let mut s = 0.5 - p;
        if s < 1e-7 {
            s = -0.5 * (-0.5 * lambda).exp_m1();
        }
        let mut a = 0.5;
        let b = 0.5 * df;
        rxb = rxb.powf(b);
        let albeta = M_LN_SQRT_PI + lgammafn(b) - lgammafn(0.5 + b);
        let mut xodd = pbeta_scalar(x, a, b, true, false);
        let mut godd = 2.0 * rxb * (a * x.ln() - albeta).exp();
        let bx = b * x;
        let mut xeven = if bx < f64::EPSILON { bx } else { 1.0 - rxb };
        let mut geven = bx * rxb;
        tnc = p * xodd + q * xeven;
        let mut it = 1;
        while it <= itrmax {
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2.0 * it as f64);
            q *= lambda / (2.0 * it as f64 + 1.0);
            tnc += p * xodd + q * xeven;
            s -= p;
            if s < -1e-10 {
                break;
            }
            if s <= 0.0 && it > 1 {
                break;
            }
            let errbd = 2.0 * s * (xodd - godd);
            if errbd.abs() < errmax {
                break;
            }
            it += 1;
        }
    } else {
        tnc = 0.0;
    }
    tnc += pnorm5_scalar(-del, 0.0, 1.0, true, false);
    lower_tail = lower_tail != negdel;
    r_dt_val(tnc.min(1.0), lower_tail, log_p)
}

pub(crate) fn dnt_scalar(x: f64, df: f64, ncp: f64, give_log: bool) -> f64 {
    if x.is_nan() || df.is_nan() {
        return x + df;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if ncp == 0.0 {
        return dt_scalar(x, df, give_log);
    }
    if !x.is_finite() {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if !df.is_finite() || df > 1e8 {
        return dnorm5_scalar(x, ncp, 1.0, give_log);
    }
    let u = if x.abs() > (df * f64::EPSILON).sqrt() {
        df.ln() - x.abs().ln()
            + (pnt_scalar(x * ((df + 2.0) / df).sqrt(), df + 2.0, ncp, true, false)
                - pnt_scalar(x, df, ncp, true, false))
            .abs()
            .ln()
    } else {
        lgammafn((df + 1.0) / 2.0)
            - lgammafn(df / 2.0)
            - (M_LN_SQRT_PI + 0.5 * (df.ln() + ncp * ncp))
    };
    if give_log {
        u
    } else {
        u.exp()
    }
}

pub(crate) fn qnt_scalar(p: f64, df: f64, ncp: f64, lower_tail: bool, log_p: bool) -> f64 {
    let accu = 1e-13;
    let eps = 1e-11;
    if p.is_nan() || df.is_nan() || ncp.is_nan() {
        return p + df + ncp;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if ncp == 0.0 && df >= 1.0 {
        return qt_scalar(p, df, lower_tail, log_p);
    }
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        if p == 1.0 {
            return if lower_tail {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
    }
    if !df.is_finite() {
        return qnorm5_scalar(p, ncp, 1.0, lower_tail, log_p);
    }
    let p = r_dt_qiv(p, lower_tail, log_p);
    if p > 1.0 - f64::EPSILON {
        return f64::INFINITY;
    }
    let mut pp = (1.0 - f64::EPSILON).min(p * (1.0 + eps));
    let mut ux = 1.0f64.max(ncp);
    while ux < f64::MAX && pnt_scalar(ux, df, ncp, true, false) < pp {
        ux *= 2.0;
    }
    pp = p * (1.0 - eps);
    let mut lx = (-1.0f64).min(-ncp);
    while lx > -f64::MAX && pnt_scalar(lx, df, ncp, true, false) > pp {
        lx *= 2.0;
    }
    let mut nx;
    loop {
        nx = 0.5 * (lx + ux);
        if pnt_scalar(nx, df, ncp, true, false) > p {
            ux = nx;
        } else {
            lx = nx;
        }
        if !((ux - lx) > accu * lx.abs().max(ux.abs())) {
            break;
        }
    }
    0.5 * (lx + ux)
}

fn pnbeta_raw(x: f64, o_x: f64, a: f64, b: f64, ncp: f64) -> f64 {
    let errmax = 1.0e-9;
    let itrmax = 10000;
    if ncp < 0.0 || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 || o_x > 1.0 || (x == 0.0 && o_x == 1.0) {
        return 0.0;
    }
    if x > 1.0 || o_x < 0.0 || (x == 1.0 && o_x == 0.0) {
        return 1.0;
    }
    let c = ncp / 2.0;
    let x0 = (c - 7.0 * c.sqrt()).max(0.0).floor();
    let a0 = a + x0;
    let l_beta = lbeta_scalar(a0, b);
    let (mut temp, _tmp_c, _ierr) = bratio(a0, b, x, o_x, false);
    let mut gx =
        (a0 * x.ln() + b * (if x < 0.5 { (-x).ln_1p() } else { o_x.ln() }) - l_beta - a0.ln())
            .exp();
    let mut q = if a0 > a {
        (-c + x0 * c.ln() - lgammafn(x0 + 1.0)).exp()
    } else {
        (-c).exp()
    };
    let mut sumq = 1.0 - q;
    let mut ans = q * temp;
    let mut j = x0.floor();
    loop {
        j += 1.0;
        temp -= gx;
        gx *= x * (a + b + j - 1.0) / (a + j);
        q *= c / j;
        sumq -= q;
        let ax = temp * q;
        ans += ax;
        let errbd = (temp - gx) * sumq;
        if !(errbd > errmax && j < itrmax as f64 + x0) {
            break;
        }
    }
    ans
}

fn pnbeta2(x: f64, o_x: f64, a: f64, b: f64, ncp: f64, lower_tail: bool, log_p: bool) -> f64 {
    let mut ans = pnbeta_raw(x, o_x, a, b, ncp);
    if lower_tail {
        return if log_p { ans.ln() } else { ans };
    }
    if ans > 1.0 {
        ans = 1.0;
    }
    if log_p {
        (-ans).ln_1p()
    } else {
        1.0 - ans
    }
}

pub(crate) fn pnbeta_scalar(
    x: f64,
    a: f64,
    b: f64,
    ncp: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() || ncp.is_nan() {
        return x + a + b + ncp;
    }
    if x <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    if x >= 1.0 {
        return dt1(lower_tail, log_p);
    }
    pnbeta2(x, 1.0 - x, a, b, ncp, lower_tail, log_p)
}

pub(crate) fn dnbeta_scalar(x: f64, a: f64, b: f64, ncp: f64, give_log: bool) -> f64 {
    let eps = 1.0e-15;
    if x.is_nan() || a.is_nan() || b.is_nan() || ncp.is_nan() {
        return x + a + b + ncp;
    }
    if ncp < 0.0 || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if !a.is_finite() || !b.is_finite() || !ncp.is_finite() {
        return f64::NAN;
    }
    if x < 0.0 || x > 1.0 {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if ncp == 0.0 {
        return dbeta_scalar(x, a, b, give_log);
    }
    let ncp2 = 0.5 * ncp;
    let dx2 = ncp2 * x;
    let d = 0.5 * (dx2 - a - 1.0);
    let mut dd = d * d + dx2 * (a + b) - a;
    let k_max = if dd <= 0.0 {
        0.0
    } else {
        dd = (d + dd.sqrt()).ceil();
        if dd > 0.0 {
            dd
        } else {
            0.0
        }
    };
    let mut term = dbeta_scalar(x, a + k_max, b, true);
    let mut p_k = dpois_raw(k_max, ncp2, true);
    if x == 0.0 || !term.is_finite() || !p_k.is_finite() {
        let v = p_k + term;
        return if give_log { v } else { v.exp() };
    }
    p_k += term;
    let mut sum = 1.0f64;
    term = 1.0;
    let mut k = k_max;
    while k > 0.0 && term > sum * eps {
        k -= 1.0;
        let q = (k + 1.0) * (k + a) / (k + a + b) / dx2;
        term *= q;
        sum += term;
    }
    term = 1.0;
    k = k_max;
    loop {
        let q = dx2 * (k + a + b) / (k + a) / (k + 1.0);
        k += 1.0;
        term *= q;
        sum += term;
        if !(term > sum * eps) {
            break;
        }
    }
    let v = p_k + sum.ln();
    if give_log {
        v
    } else {
        v.exp()
    }
}

pub(crate) fn qnbeta_scalar(
    p: f64,
    a: f64,
    b: f64,
    ncp: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    let accu = 1e-15;
    let eps = 1e-14;
    if p.is_nan() || a.is_nan() || b.is_nan() || ncp.is_nan() {
        return p + a + b + ncp;
    }
    if !a.is_finite() {
        return f64::NAN;
    }
    if ncp < 0.0 || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { 1.0 } else { 0.0 };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail { 0.0 } else { 1.0 };
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { 0.0 } else { 1.0 };
        }
        if p == 1.0 {
            return if lower_tail { 1.0 } else { 0.0 };
        }
    }
    let p = r_dt_qiv(p, lower_tail, log_p);
    if p > 1.0 - f64::EPSILON {
        return 1.0;
    }
    let mut pp = (1.0 - f64::EPSILON).min(p * (1.0 + eps));
    let mut ux = 0.5f64;
    while ux < 1.0 - f64::EPSILON && pnbeta_scalar(ux, a, b, ncp, true, false) < pp {
        ux = 0.5 * (1.0 + ux);
    }
    pp = p * (1.0 - eps);
    let mut lx = 0.5f64;
    while lx > DBL_MIN && pnbeta_scalar(lx, a, b, ncp, true, false) > pp {
        lx *= 0.5;
    }
    let mut nx;
    loop {
        nx = 0.5 * (lx + ux);
        if pnbeta_scalar(nx, a, b, ncp, true, false) > p {
            ux = nx;
        } else {
            lx = nx;
        }
        if !((ux - lx) / nx > accu) {
            break;
        }
    }
    0.5 * (ux + lx)
}

pub(crate) fn pnf_scalar(
    x: f64,
    df1: f64,
    df2: f64,
    ncp: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x.is_nan() || df1.is_nan() || df2.is_nan() || ncp.is_nan() {
        return x + df2 + df1 + ncp;
    }
    if df1 <= 0.0 || df2 <= 0.0 || ncp < 0.0 {
        return f64::NAN;
    }
    if !ncp.is_finite() {
        return f64::NAN;
    }
    if !df1.is_finite() && !df2.is_finite() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    if x == f64::INFINITY {
        return dt1(lower_tail, log_p);
    }
    if df2 > 1e8 {
        return pnchisq_scalar(x * df1, df1, ncp, lower_tail, log_p);
    }
    let y = (df1 / df2) * x;
    pnbeta2(
        y / (1.0 + y),
        1.0 / (1.0 + y),
        df1 / 2.0,
        df2 / 2.0,
        ncp,
        lower_tail,
        log_p,
    )
}

pub(crate) fn dnf_scalar(x: f64, df1: f64, df2: f64, ncp: f64, give_log: bool) -> f64 {
    if x.is_nan() || df1.is_nan() || df2.is_nan() || ncp.is_nan() {
        return x + df2 + df1 + ncp;
    }
    if df1 <= 0.0 || df2 <= 0.0 || ncp < 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if !ncp.is_finite() {
        return f64::NAN;
    }
    if !df1.is_finite() && !df2.is_finite() {
        if x == 1.0 {
            return f64::INFINITY;
        }
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if !df2.is_finite() {
        return df1 * dnchisq_scalar(x * df1, df1, ncp, give_log);
    }
    if df1 > 1e14 && ncp < 1e7 {
        let f = 1.0 + ncp / df1;
        let z = dgamma_scalar(1.0 / x / f, df2 / 2.0, 2.0 / df2, give_log);
        return if give_log {
            z - 2.0 * x.ln() - f.ln()
        } else {
            z / (x * x) / f
        };
    }
    let y = (df1 / df2) * x;
    let z = dnbeta_scalar(y / (1.0 + y), df1 / 2.0, df2 / 2.0, ncp, give_log);
    if give_log {
        z + df1.ln() - df2.ln() - 2.0 * y.ln_1p()
    } else {
        z * (df1 / df2) / (1.0 + y) / (1.0 + y)
    }
}

pub(crate) fn qnf_scalar(
    p: f64,
    df1: f64,
    df2: f64,
    ncp: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if p.is_nan() || df1.is_nan() || df2.is_nan() || ncp.is_nan() {
        return p + df1 + df2 + ncp;
    }
    if df1 <= 0.0 || df2 <= 0.0 || ncp < 0.0 {
        return f64::NAN;
    }
    if !ncp.is_finite() {
        return f64::NAN;
    }
    if !df1.is_finite() && !df2.is_finite() {
        return f64::NAN;
    }
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { f64::INFINITY } else { 0.0 };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail { 0.0 } else { f64::INFINITY };
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { 0.0 } else { f64::INFINITY };
        }
        if p == 1.0 {
            return if lower_tail { f64::INFINITY } else { 0.0 };
        }
    }
    if df2 > 1e8 {
        return qnchisq_scalar(p, df1, ncp, lower_tail, log_p) / df1;
    }
    let y = qnbeta_scalar(p, df1 / 2.0, df2 / 2.0, ncp, lower_tail, log_p);
    y / (1.0 - y) * (df2 / df1)
}

#[pyfunction]
#[pyo3(name = "pnchisq", signature = (x, df, ncp, lower_tail=true, log_p=false))]
pub fn pnchisq<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        df.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |x, d, n| pnchisq_scalar(x, d, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dnchisq", signature = (x, df, ncp, give_log=false))]
pub fn dnchisq<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        df.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |x, d, n| dnchisq_scalar(x, d, n, give_log),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qnchisq", signature = (p, df, ncp, lower_tail=true, log_p=false))]
pub fn qnchisq<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        p.as_slice().unwrap(),
        df.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |p, d, n| qnchisq_scalar(p, d, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "pnt", signature = (t, df, ncp, lower_tail=true, log_p=false))]
pub fn pnt<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        t.as_slice().unwrap(),
        df.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |t, d, n| pnt_scalar(t, d, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dnt", signature = (x, df, ncp, give_log=false))]
pub fn dnt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        df.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |x, d, n| dnt_scalar(x, d, n, give_log),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qnt", signature = (p, df, ncp, lower_tail=true, log_p=false))]
pub fn qnt<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        p.as_slice().unwrap(),
        df.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |p, d, n| qnt_scalar(p, d, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "pnbeta", signature = (x, a, b, ncp, lower_tail=true, log_p=false))]
pub fn pnbeta<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        x.as_slice().unwrap(),
        a.as_slice().unwrap(),
        b.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |x, a, b, n| pnbeta_scalar(x, a, b, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dnbeta", signature = (x, a, b, ncp, give_log=false))]
pub fn dnbeta<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        x.as_slice().unwrap(),
        a.as_slice().unwrap(),
        b.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |x, a, b, n| dnbeta_scalar(x, a, b, n, give_log),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qnbeta", signature = (p, a, b, ncp, lower_tail=true, log_p=false))]
pub fn qnbeta<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        p.as_slice().unwrap(),
        a.as_slice().unwrap(),
        b.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |p, a, b, n| qnbeta_scalar(p, a, b, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "pnf", signature = (x, df1, df2, ncp, lower_tail=true, log_p=false))]
pub fn pnf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    df1: PyReadonlyArray1<'py, f64>,
    df2: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        x.as_slice().unwrap(),
        df1.as_slice().unwrap(),
        df2.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |x, a, b, n| pnf_scalar(x, a, b, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dnf", signature = (x, df1, df2, ncp, give_log=false))]
pub fn dnf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    df1: PyReadonlyArray1<'py, f64>,
    df2: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        x.as_slice().unwrap(),
        df1.as_slice().unwrap(),
        df2.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |x, a, b, n| dnf_scalar(x, a, b, n, give_log),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qnf", signature = (p, df1, df2, ncp, lower_tail=true, log_p=false))]
pub fn qnf<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    df1: PyReadonlyArray1<'py, f64>,
    df2: PyReadonlyArray1<'py, f64>,
    ncp: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        p.as_slice().unwrap(),
        df1.as_slice().unwrap(),
        df2.as_slice().unwrap(),
        ncp.as_slice().unwrap(),
        |p, a, b, n| qnf_scalar(p, a, b, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}
