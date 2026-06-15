//! `pgamma` / `dgamma` / `qgamma` — R's pgamma.c (Welinder) / dgamma.c /
//! qgamma.c (AS 91 + Newton). Mirror of the `hea/R/nmath.py` gamma cluster.
//! Scalar kernels + numpy-vectorized PyO3 wrappers.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::coeffs::{LGAMMA1P_COEFFS, PPA_COEFS_A, PPA_COEFS_B};
use super::consts::{DBL_MIN, M_CUTOFF, M_LN2, PG_SCALEFACTOR};
use super::lgamma::lgammafn;
use super::loader::dpois_raw;
use super::norm::{dnorm5_scalar, dt0, dt1, pnorm5_scalar, qnorm5_scalar};

const EPS: f64 = f64::EPSILON;

// --- dpq.h helpers -----------------------------------------------------------
#[inline]
pub(crate) fn r_log1_exp(x: f64) -> f64 {
    if x > -M_LN2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}
#[inline]
pub(crate) fn r_d_log(p: f64, log_p: bool) -> f64 {
    if log_p {
        p
    } else {
        p.ln()
    }
}
#[inline]
pub(crate) fn r_d_lexp(p: f64, log_p: bool) -> f64 {
    if log_p {
        r_log1_exp(p)
    } else {
        (-p).ln_1p()
    }
}
#[inline]
pub(crate) fn r_dt_log(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_log(p, log_p)
    } else {
        r_d_lexp(p, log_p)
    }
}
#[inline]
pub(crate) fn r_dt_clog(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_lexp(p, log_p)
    } else {
        r_d_log(p, log_p)
    }
}
#[inline]
pub(crate) fn r_dt_qiv(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            p.exp()
        } else {
            -p.exp_m1()
        }
    } else if lower_tail {
        p
    } else {
        0.5 - p + 0.5
    }
}

// --- continued fraction / series helpers (pgamma.c) --------------------------
fn logcf(x: f64, i: f64, d: f64, eps: f64) -> f64 {
    let mut c1 = 2.0 * d;
    let mut c2 = i + d;
    let mut c4 = c2 + d;
    let mut a1 = c2;
    let mut b1 = i * (c2 - i * x);
    let mut b2 = d * d * x;
    let mut a2 = c4 * c2 - b2;
    b2 = c4 * b1 - i * b2;
    let sf = PG_SCALEFACTOR;
    while (a2 * b1 - a1 * b2).abs() > (eps * b1 * b2).abs() {
        let mut c3 = c2 * c2 * x;
        c2 += d;
        c4 += d;
        a1 = c4 * a2 - c3 * a1;
        b1 = c4 * b2 - c3 * b1;
        c3 = c1 * c1 * x;
        c1 += d;
        c4 += d;
        a2 = c4 * a1 - c3 * a2;
        b2 = c4 * b1 - c3 * b2;
        if b2.abs() > sf {
            a1 /= sf;
            b1 /= sf;
            a2 /= sf;
            b2 /= sf;
        } else if b2.abs() < 1.0 / sf {
            a1 *= sf;
            b1 *= sf;
            a2 *= sf;
            b2 *= sf;
        }
    }
    a2 / b2
}

pub(crate) fn log1pmx(x: f64) -> f64 {
    let min_log1_value = -0.79149064;
    if x > 1.0 || x < min_log1_value {
        return x.ln_1p() - x;
    }
    let r = x / (2.0 + x);
    let y = r * r;
    if x.abs() < 1e-2 {
        let two = 2.0;
        return r * ((((two / 9.0 * y + two / 7.0) * y + two / 5.0) * y + two / 3.0) * y - x);
    }
    let tol_logcf = 1e-14;
    r * (2.0 * y * logcf(y, 3.0, 2.0, tol_logcf) - x)
}

fn lgamma1p(a: f64) -> f64 {
    if a.abs() >= 0.5 {
        return lgammafn(a + 1.0);
    }
    let eulers_const = 0.5772156649015328606065120900824024;
    let n = 40i32;
    let c = 0.2273736845824652515226821577978691e-12;
    let tol_logcf = 1e-14;
    let mut lgam = c * logcf(-a / 2.0, (n + 2) as f64, 1.0, tol_logcf);
    for i in (0..n).rev() {
        lgam = LGAMMA1P_COEFFS[i as usize] - a * lgam;
    }
    (a * lgam - eulers_const) * a - log1pmx(a)
}

fn dpois_wrap(x_plus_1: f64, lambda: f64, give_log: bool) -> f64 {
    if !lambda.is_finite() {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if x_plus_1 > 1.0 {
        return dpois_raw(x_plus_1 - 1.0, lambda, give_log);
    }
    if lambda > (x_plus_1 - 1.0).abs() * M_CUTOFF {
        let v = -lambda - lgammafn(x_plus_1);
        return if give_log { v } else { v.exp() };
    }
    let d = dpois_raw(x_plus_1, lambda, give_log);
    if give_log {
        d + (x_plus_1 / lambda).ln()
    } else {
        d * (x_plus_1 / lambda)
    }
}

fn pgamma_smallx(x: f64, alph: f64, lower_tail: bool, log_p: bool) -> f64 {
    let mut sum = 0.0;
    let mut c = alph;
    let mut n = 0.0;
    loop {
        n += 1.0;
        c *= -x / n;
        let term = c / (alph + n);
        sum += term;
        if term.abs() <= EPS * sum.abs() {
            break;
        }
    }
    if lower_tail {
        let f1 = if log_p { sum.ln_1p() } else { 1.0 + sum };
        let f2 = if alph > 1.0 {
            let t = dpois_raw(alph, x, log_p);
            if log_p {
                t + x
            } else {
                t * x.exp()
            }
        } else if log_p {
            alph * x.ln() - lgamma1p(alph)
        } else {
            x.powf(alph) / lgamma1p(alph).exp()
        };
        if log_p {
            f1 + f2
        } else {
            f1 * f2
        }
    } else {
        let lf2 = alph * x.ln() - lgamma1p(alph);
        if log_p {
            r_log1_exp(sum.ln_1p() + lf2)
        } else {
            let f1m1 = sum;
            let f2m1 = lf2.exp_m1();
            -(f1m1 + f2m1 + f1m1 * f2m1)
        }
    }
}

fn pd_upper_series(x: f64, mut y: f64, log_p: bool) -> f64 {
    let mut term = x / y;
    let mut sum = term;
    loop {
        y += 1.0;
        term *= x / y;
        sum += term;
        if !(term > sum * EPS) {
            break;
        }
    }
    if log_p {
        sum.ln()
    } else {
        sum
    }
}

fn pd_lower_cf(y: f64, d: f64) -> f64 {
    let sf = PG_SCALEFACTOR;
    let max_it = 200000.0;
    if y == 0.0 {
        return 0.0;
    }
    let mut f0 = y / d;
    if (y - 1.0).abs() < d.abs() * EPS {
        return f0;
    }
    if f0 > 1.0 {
        f0 = 1.0;
    }
    let mut c2 = y;
    let mut c4 = d;
    let mut a1 = 0.0;
    let mut b1 = 1.0;
    let mut a2 = y;
    let mut b2 = d;
    while b2 > sf {
        a1 /= sf;
        b1 /= sf;
        a2 /= sf;
        b2 /= sf;
    }
    let mut i = 0.0;
    let mut of = -1.0;
    let mut f = 0.0;
    while i < max_it {
        i += 1.0;
        c2 -= 1.0;
        let mut c3 = i * c2;
        c4 += 2.0;
        a1 = c4 * a2 + c3 * a1;
        b1 = c4 * b2 + c3 * b1;
        i += 1.0;
        c2 -= 1.0;
        c3 = i * c2;
        c4 += 2.0;
        a2 = c4 * a1 + c3 * a2;
        b2 = c4 * b1 + c3 * b2;
        if b2 > sf {
            a1 /= sf;
            b1 /= sf;
            a2 /= sf;
            b2 /= sf;
        }
        if b2 != 0.0 {
            f = a2 / b2;
            if (f - of).abs() <= EPS * f0.max(f.abs()) {
                return f;
            }
            of = f;
        }
    }
    f
}

fn pd_lower_series(lambda: f64, mut y: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 0.0;
    while y >= 1.0 && term > sum * EPS {
        term *= y / lambda;
        sum += term;
        y -= 1.0;
    }
    if y != y.floor() {
        let f = pd_lower_cf(y, lambda + 1.0 - y);
        sum += term * f;
    }
    sum
}

fn dpnorm(mut x: f64, mut lower_tail: bool, lp: f64) -> f64 {
    if x < 0.0 {
        x = -x;
        lower_tail = !lower_tail;
    }
    if x > 10.0 && !lower_tail {
        let mut term = 1.0 / x;
        let mut sum = term;
        let x2 = x * x;
        let mut i = 1.0;
        loop {
            term *= -i / x2;
            sum += term;
            i += 2.0;
            if !(term.abs() > EPS * sum) {
                break;
            }
        }
        return 1.0 / sum;
    }
    let d = dnorm5_scalar(x, 0.0, 1.0, false);
    d / lp.exp()
}

fn ppois_asymp(x: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    let dfm = lambda - x;
    let pt_ = -log1pmx(dfm / x);
    let mut s2pt = (2.0 * x * pt_).sqrt();
    if dfm < 0.0 {
        s2pt = -s2pt;
    }
    let mut res12 = 0.0;
    let mut res1_ig = x.sqrt();
    let mut res1_term = res1_ig;
    let mut res2_ig = s2pt;
    let mut res2_term = s2pt;
    for i in 1..8 {
        res12 += res1_ig * PPA_COEFS_A[i];
        res12 += res2_ig * PPA_COEFS_B[i];
        res1_term *= pt_ / i as f64;
        res2_term *= 2.0 * pt_ / (2.0 * i as f64 + 1.0);
        res1_ig = res1_ig / x + res1_term;
        res2_ig = res2_ig / x + res2_term;
    }
    let mut elfb = x;
    let mut elfb_term = 1.0;
    for i in 1..8 {
        elfb += elfb_term * PPA_COEFS_B[i];
        elfb_term /= x;
    }
    if !lower_tail {
        elfb = -elfb;
    }
    let f = res12 / elfb;
    let np_ = pnorm5_scalar(s2pt, 0.0, 1.0, !lower_tail, log_p);
    if log_p {
        let n_d_over_p = dpnorm(s2pt, !lower_tail, np_);
        np_ + (f * n_d_over_p).ln_1p()
    } else {
        let nd = dnorm5_scalar(s2pt, 0.0, 1.0, false);
        np_ + f * nd
    }
}

pub fn pgamma_raw(x: f64, alph: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    if x >= f64::INFINITY {
        return dt1(lower_tail, log_p);
    }
    let res;
    if x < 1.0 {
        res = pgamma_smallx(x, alph, lower_tail, log_p);
    } else if x <= alph - 1.0 && x < 0.8 * (alph + 50.0) {
        let sum = pd_upper_series(x, alph, log_p);
        let d = dpois_wrap(alph, x, log_p);
        res = if !lower_tail {
            if log_p {
                r_log1_exp(d + sum)
            } else {
                1.0 - d * sum
            }
        } else if log_p {
            sum + d
        } else {
            sum * d
        };
    } else if alph - 1.0 < x && alph < 0.8 * (x + 50.0) {
        let d = dpois_wrap(alph, x, log_p);
        let sum = if alph < 1.0 {
            if x * EPS > 1.0 - alph {
                if log_p {
                    0.0
                } else {
                    1.0
                }
            } else {
                let fcf = pd_lower_cf(alph, x - (alph - 1.0)) * x / alph;
                if log_p {
                    fcf.ln()
                } else {
                    fcf
                }
            }
        } else {
            let s = pd_lower_series(x, alph - 1.0);
            if log_p {
                s.ln_1p()
            } else {
                1.0 + s
            }
        };
        res = if !lower_tail {
            if log_p {
                sum + d
            } else {
                sum * d
            }
        } else if log_p {
            r_log1_exp(d + sum)
        } else {
            1.0 - d * sum
        };
    } else {
        res = ppois_asymp(alph - 1.0, x, !lower_tail, log_p);
    }
    if !log_p && res < DBL_MIN / EPS {
        return pgamma_raw(x, alph, lower_tail, true).exp();
    }
    res
}

pub fn pgamma_scalar(mut x: f64, alph: f64, scale: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || alph.is_nan() || scale.is_nan() {
        return x + alph + scale;
    }
    if alph < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    x /= scale;
    if x.is_nan() {
        return x;
    }
    if alph == 0.0 {
        return if x <= 0.0 {
            dt0(lower_tail, log_p)
        } else {
            dt1(lower_tail, log_p)
        };
    }
    pgamma_raw(x, alph, lower_tail, log_p)
}

pub fn dgamma_scalar(x: f64, shape: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || shape.is_nan() || scale.is_nan() {
        return x + shape + scale;
    }
    if shape < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if x < 0.0 {
        return rd0;
    }
    if shape == 0.0 {
        return if x == 0.0 { f64::INFINITY } else { rd0 };
    }
    if x == 0.0 {
        if shape < 1.0 {
            return f64::INFINITY;
        }
        if shape > 1.0 {
            return rd0;
        }
        return if give_log { -scale.ln() } else { 1.0 / scale };
    }
    if shape < 1.0 {
        let pr = dpois_raw(shape, x / scale, give_log);
        if give_log {
            let sx = shape / x;
            return pr + if sx.is_finite() {
                sx.ln()
            } else {
                shape.ln() - x.ln()
            };
        }
        return pr * shape / x;
    }
    let pr = dpois_raw(shape - 1.0, x / scale, give_log);
    if give_log {
        pr - scale.ln()
    } else {
        pr / scale
    }
}

fn qchisq_appr(p: f64, nu: f64, g: f64, lower_tail: bool, log_p: bool, tol: f64) -> f64 {
    let (c7, c8, c9, c10) = (4.67, 6.66, 6.73, 13.32);
    if p.is_nan() || nu.is_nan() {
        return p + nu;
    }
    if (log_p && p > 0.0) || (!log_p && (p < 0.0 || p > 1.0)) {
        return f64::NAN;
    }
    if nu <= 0.0 {
        return f64::NAN;
    }
    let alpha = 0.5 * nu;
    let c = alpha - 1.0;
    let p1 = r_dt_log(p, lower_tail, log_p);
    if nu < (-1.24) * p1 {
        let lgam1pa = if alpha < 0.5 {
            lgamma1p(alpha)
        } else {
            alpha.ln() + g
        };
        ((lgam1pa + p1) / alpha + M_LN2).exp()
    } else if nu > 0.32 {
        let x = qnorm5_scalar(p, 0.0, 1.0, lower_tail, log_p);
        let p1b = 2.0 / (9.0 * nu);
        let mut ch = nu * (x * p1b.sqrt() + 1.0 - p1b).powf(3.0);
        if ch > 2.2 * nu + 6.0 {
            ch = -2.0 * (r_dt_clog(p, lower_tail, log_p) - c * (0.5 * ch).ln() + g);
        }
        ch
    } else {
        let mut ch = 0.4;
        let a = r_dt_clog(p, lower_tail, log_p) + g + c * M_LN2;
        loop {
            let q = ch;
            let p1c = 1.0 / (1.0 + ch * (c7 + ch));
            let p2 = ch * (c9 + ch * (c8 + ch));
            let t = -0.5 + (c7 + 2.0 * ch) * p1c - (c9 + ch * (c10 + 3.0 * ch)) / p2;
            ch -= (1.0 - (a + 0.5 * ch).exp() * p2 * p1c) / t;
            if !((q - ch).abs() > tol * ch.abs()) {
                break;
            }
        }
        ch
    }
}

pub fn qgamma_scalar(mut p: f64, alpha: f64, scale: f64, lower_tail: bool, mut log_p: bool) -> f64 {
    let eps1 = 1e-2;
    let eps2 = 5e-7;
    let eps_n = 1e-15;
    let maxit = 1000;
    let pmin = 1e-100;
    let pmax = 1.0 - 1e-14;
    let i420 = 1.0 / 420.0;
    let i2520 = 1.0 / 2520.0;
    let i5040 = 1.0 / 5040.0;
    if p.is_nan() || alpha.is_nan() || scale.is_nan() {
        return p + alpha + scale;
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
    if alpha < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if alpha == 0.0 {
        return 0.0;
    }
    let mut max_it_newton = 1;
    if alpha < 1e-10 {
        max_it_newton = 7;
    }
    let p_ = r_dt_qiv(p, lower_tail, log_p);
    let g = lgammafn(alpha);
    let mut ch = qchisq_appr(p, 2.0 * alpha, g, lower_tail, log_p, eps1);
    let mut at_end = false;
    if !ch.is_finite() {
        max_it_newton = 0;
        at_end = true;
    } else if ch < eps2 {
        max_it_newton = 20;
        at_end = true;
    } else if p_ > pmax || p_ < pmin {
        max_it_newton = 20;
        at_end = true;
    }
    if !at_end {
        let c = alpha - 1.0;
        let s6 = (120.0 + c * (346.0 + 127.0 * c)) * i5040;
        let ch0 = ch;
        for i in 1..=maxit {
            let q = ch;
            let p1 = 0.5 * ch;
            let p2 = p_ - pgamma_raw(p1, alpha, true, false);
            if !p2.is_finite() || ch <= 0.0 {
                ch = ch0;
                max_it_newton = 27;
                break;
            }
            let t = p2 * (alpha * M_LN2 + g + p1 - c * ch.ln()).exp();
            let b = t / ch;
            let a = 0.5 * t - b * c;
            let s1 = (210.0 + a * (140.0 + a * (105.0 + a * (84.0 + a * (70.0 + 60.0 * a))))) * i420;
            let s2 = (420.0 + a * (735.0 + a * (966.0 + a * (1141.0 + 1278.0 * a)))) * i2520;
            let s3 = (210.0 + a * (462.0 + a * (707.0 + 932.0 * a))) * i2520;
            let s4 = (252.0 + a * (672.0 + 1182.0 * a) + c * (294.0 + a * (889.0 + 1740.0 * a)))
                * i5040;
            let s5 = (84.0 + 2264.0 * a + c * (1175.0 + 606.0 * a)) * i2520;
            ch += t
                * (1.0 + 0.5 * t * s1
                    - b * c * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))));
            if (q - ch).abs() < eps2 * ch {
                break;
            }
            if (q - ch).abs() > 0.1 * ch {
                ch = if ch < q { 0.9 * q } else { 1.1 * q };
            }
            let _ = i;
        }
    }
    // END
    let mut x = 0.5 * scale * ch;
    if max_it_newton != 0 {
        if !log_p {
            p = p.ln();
            log_p = true;
        }
        let mut pp;
        if x == 0.0 {
            let _1_p = 1.0 + 1e-7;
            let _1_m = 1.0 - 1e-7;
            x = DBL_MIN;
            pp = pgamma_scalar(x, alpha, scale, lower_tail, log_p);
            if (lower_tail && pp > p * _1_p) || (!lower_tail && pp < p * _1_m) {
                return 0.0;
            }
        } else {
            pp = pgamma_scalar(x, alpha, scale, lower_tail, log_p);
        }
        if pp == f64::NEG_INFINITY {
            return 0.0;
        }
        let rd0 = if log_p { f64::NEG_INFINITY } else { 0.0 };
        for i in 1..=max_it_newton {
            let p1 = pp - p;
            if p1.abs() < (eps_n * p).abs() {
                break;
            }
            let gg = dgamma_scalar(x, alpha, scale, log_p);
            if gg == rd0 {
                break;
            }
            let t = if log_p { p1 * (pp - gg).exp() } else { p1 / gg };
            let t = if lower_tail { x - t } else { x + t };
            pp = pgamma_scalar(t, alpha, scale, lower_tail, log_p);
            if (pp - p).abs() > p1.abs() || (i > 1 && (pp - p).abs() == p1.abs()) {
                break;
            }
            x = t;
        }
    }
    x
}

// === PyO3 wrappers (numpy-vectorized; Python pre-broadcasts equal-length) =====

#[pyfunction]
#[pyo3(name = "pgamma", signature = (x, alph, scale, lower_tail=true, log_p=false))]
pub fn pgamma<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    alph: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let out = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        alph.as_slice().unwrap(),
        scale.as_slice().unwrap(),
        |x, a, s| pgamma_scalar(x, a, s, lower_tail, log_p),
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dgamma", signature = (x, shape, scale, give_log=false))]
pub fn dgamma<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    shape: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let out = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        shape.as_slice().unwrap(),
        scale.as_slice().unwrap(),
        |x, sh, sc| dgamma_scalar(x, sh, sc, give_log),
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qgamma", signature = (p, alpha, scale, lower_tail=true, log_p=false))]
pub fn qgamma<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let out = crate::par::map3(
        py,
        p.as_slice().unwrap(),
        alpha.as_slice().unwrap(),
        scale.as_slice().unwrap(),
        |p, a, s| qgamma_scalar(p, a, s, lower_tail, log_p),
    );
    out.into_pyarray(py)
}
