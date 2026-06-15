//! Discrete CDFs/PMFs + quantiles — R's nmath ppois.c / pbinom.c / dpois.c /
//! dbinom.c / dbeta.c / qpois.c / qbinom.c (qDiscrete_search.h). Mirror of the
//! `hea/R/nmath.py` discrete cluster.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::gamma::pgamma_scalar;
use super::loader::{dbinom_raw, dpois_raw};
use super::norm::{dt0, dt1, qnorm5_scalar};
use super::toms708::{lbeta_scalar, pbeta_scalar};
use super::util::round_half_even;

// === CDFs ====================================================================
pub(crate) fn ppois_scalar(x: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || lambda.is_nan() {
        return x + lambda;
    }
    if lambda < 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return dt0(lower_tail, log_p);
    }
    if lambda == 0.0 {
        return dt1(lower_tail, log_p);
    }
    if !x.is_finite() {
        return dt1(lower_tail, log_p);
    }
    let x = (x + 1e-7).floor();
    pgamma_scalar(lambda, x + 1.0, 1.0, !lower_tail, log_p)
}

pub(crate) fn pbinom_scalar(x: f64, n: f64, p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || n.is_nan() || p.is_nan() {
        return x + n + p;
    }
    if !n.is_finite() || !p.is_finite() {
        return f64::NAN;
    }
    let n = round_half_even(n);
    if n < 0.0 || p < 0.0 || p > 1.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return dt0(lower_tail, log_p);
    }
    let x = (x + 1e-7).floor();
    if n <= x {
        return dt1(lower_tail, log_p);
    }
    pbeta_scalar(p, x + 1.0, n - x, !lower_tail, log_p)
}

// === densities ===============================================================
fn r_nonint(x: f64) -> bool {
    (x - round_half_even(x)).abs() > 1e-9 * 1f64.max(x.abs())
}

pub(crate) fn dbinom_scalar(x: f64, n: f64, p: f64, give_log: bool) -> f64 {
    if x.is_nan() || n.is_nan() || p.is_nan() {
        return x + n + p;
    }
    if p < 0.0 || p > 1.0 || (n < 0.0 || r_nonint(n)) {
        return f64::NAN;
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if r_nonint(x) {
        return rd0;
    }
    if x < 0.0 || !x.is_finite() {
        return rd0;
    }
    let n = round_half_even(n);
    let x = round_half_even(x);
    dbinom_raw(x, n, p, 1.0 - p, give_log)
}

pub(crate) fn dpois_scalar(x: f64, lambda: f64, give_log: bool) -> f64 {
    if x.is_nan() || lambda.is_nan() {
        return x + lambda;
    }
    if lambda < 0.0 {
        return f64::NAN;
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if r_nonint(x) {
        return rd0;
    }
    if x < 0.0 || !x.is_finite() {
        return rd0;
    }
    let x = round_half_even(x);
    dpois_raw(x, lambda, give_log)
}

pub(crate) fn dbeta_scalar(x: f64, a: f64, b: f64, give_log: bool) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return x + a + b;
    }
    if a < 0.0 || b < 0.0 {
        return f64::NAN;
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if x < 0.0 || x > 1.0 {
        return rd0;
    }
    if a == 0.0 || b == 0.0 || !a.is_finite() || !b.is_finite() {
        if a == 0.0 && b == 0.0 {
            return if x == 0.0 || x == 1.0 { f64::INFINITY } else { rd0 };
        }
        if a == 0.0 || a / b == 0.0 {
            return if x == 0.0 { f64::INFINITY } else { rd0 };
        }
        if b == 0.0 || b / a == 0.0 {
            return if x == 1.0 { f64::INFINITY } else { rd0 };
        }
        return if x == 0.5 { f64::INFINITY } else { rd0 };
    }
    if x == 0.0 {
        if a > 1.0 {
            return rd0;
        }
        if a < 1.0 {
            return f64::INFINITY;
        }
    }
    if x == 1.0 {
        if b > 1.0 {
            return rd0;
        }
        if b < 1.0 {
            return f64::INFINITY;
        }
    }
    let lval = if a <= 2.0 || b <= 2.0 {
        (a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - lbeta_scalar(a, b)
    } else {
        (a + b - 1.0).ln() + dbinom_raw(a - 1.0, a + b - 2.0, x, 1.0 - x, true)
    };
    if give_log {
        lval
    } else {
        lval.exp()
    }
}

// === discrete quantile search (qDiscrete_search.h) ===========================
fn do_search<F: Fn(f64, bool, bool) -> f64>(
    mut y: f64,
    z: &mut f64,
    p: f64,
    cdf: &F,
    incr: f64,
    lower_tail: bool,
    log_p: bool,
    y_max: Option<f64>,
) -> f64 {
    let left = if lower_tail { *z >= p } else { *z < p };
    if left {
        loop {
            let mut newz = -1.0;
            if y > 0.0 {
                newz = cdf(y - incr, lower_tail, log_p);
            } else if y < 0.0 {
                y = 0.0;
            }
            if y == 0.0 || newz.is_nan() || (if lower_tail { newz < p } else { newz >= p }) {
                return y;
            }
            y = 0f64.max(y - incr);
            *z = newz;
        }
    } else {
        loop {
            let prevy = y;
            let mut newz = -1.0;
            y += incr;
            if let Some(ym) = y_max {
                if y < ym {
                    newz = cdf(y, lower_tail, log_p);
                } else if y > ym {
                    y = ym;
                }
            } else {
                newz = cdf(y, lower_tail, log_p);
            }
            if y_max == Some(y) || newz.is_nan() || (if lower_tail { newz >= p } else { newz < p })
            {
                if incr <= 1.0 {
                    *z = newz;
                    return y;
                }
                return prevy;
            }
            *z = newz;
        }
    }
}

fn q_discrete<F: Fn(f64, bool, bool) -> f64>(
    mut p: f64,
    lower_tail: bool,
    log_p: bool,
    mu: f64,
    sigma: f64,
    gamma: f64,
    cdf: &F,
    y_max: Option<f64>,
) -> f64 {
    let z = qnorm5_scalar(p, 0.0, 1.0, lower_tail, log_p);
    // `+ 0.0` normalizes -0.0 -> +0.0 to match Python's float(round(...)) (int->float),
    // which never yields signed-zero (R's nearbyint would).
    let mut y = round_half_even(mu + sigma * (z + gamma * (z * z - 1.0) / 6.0)) + 0.0;
    if let Some(ym) = y_max {
        if y > ym {
            y = ym;
        } else if y < 0.0 {
            y = 0.0;
        }
    } else if y < 0.0 {
        y = 0.0;
    }
    let mut zc = cdf(y, lower_tail, log_p);
    let pf_n = 8.0;
    let pf_l = 2.0;
    let y_large = 4096.0;
    let inc_f = 1.0 / 64.0;
    let i_shrink = 8.0;
    let rel_tol = 1e-15;
    let xf = 4.0;
    if log_p {
        let e = pf_l * f64::EPSILON;
        if lower_tail && p > -1.7976931348623157e308 {
            p *= 1.0 + e;
        } else {
            p *= 1.0 - e;
        }
    } else {
        let e = pf_n * f64::EPSILON;
        if lower_tail {
            p *= 1.0 - e;
        } else if 1.0 - p > xf * e {
            p *= 1.0 + e;
        }
    }
    if y < y_large {
        return do_search(y, &mut zc, p, cdf, 1.0, lower_tail, log_p, y_max);
    }
    let mut incr = (y * inc_f).floor();
    loop {
        let oldincr = incr;
        y = do_search(y, &mut zc, p, cdf, incr, lower_tail, log_p, y_max);
        incr = 1f64.max((incr / i_shrink).floor());
        if !(oldincr > 1.0 && incr > y * rel_tol) {
            break;
        }
    }
    y
}

pub(crate) fn qpois_scalar(p: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || lambda.is_nan() {
        return p + lambda;
    }
    if !lambda.is_finite() {
        return f64::NAN;
    }
    if lambda < 0.0 {
        return f64::NAN;
    }
    if (log_p && p > 0.0) || (!log_p && (p < 0.0 || p > 1.0)) {
        return f64::NAN;
    }
    if lambda == 0.0 {
        return 0.0;
    }
    if p == dt0(lower_tail, log_p) {
        return 0.0;
    }
    if p == dt1(lower_tail, log_p) {
        return f64::INFINITY;
    }
    let sigma = lambda.sqrt();
    let gamma = 1.0 / sigma;
    let cdf = |y: f64, lt: bool, lg: bool| ppois_scalar(y, lambda, lt, lg);
    q_discrete(p, lower_tail, log_p, lambda, sigma, gamma, &cdf, None)
}

pub(crate) fn qbinom_scalar(p: f64, n: f64, pr: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || n.is_nan() || pr.is_nan() {
        return p + n + pr;
    }
    if !n.is_finite() || !pr.is_finite() {
        return f64::NAN;
    }
    if !p.is_finite() && !log_p {
        return f64::NAN;
    }
    let n = round_half_even(n);
    if pr < 0.0 || pr > 1.0 || n < 0.0 {
        return f64::NAN;
    }
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { n } else { 0.0 };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail { 0.0 } else { n };
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { 0.0 } else { n };
        }
        if p == 1.0 {
            return if lower_tail { n } else { 0.0 };
        }
    }
    if pr == 0.0 || n == 0.0 {
        return 0.0;
    }
    if pr == 1.0 {
        return n;
    }
    let q = 1.0 - pr;
    let mu = n * pr;
    let sigma = (n * pr * q).sqrt();
    let gamma = (q - pr) / sigma;
    let cdf = |y: f64, lt: bool, lg: bool| pbinom_scalar(y, n, pr, lt, lg);
    q_discrete(p, lower_tail, log_p, mu, sigma, gamma, &cdf, Some(n))
}

// === PyO3 wrappers ===========================================================
macro_rules! wrap2 {
    ($name:literal, $fn:ident, $sc:path, ($p2:ident=$d2:literal), ($p3:ident=$d3:literal)) => {
        #[pyfunction]
        #[pyo3(name = $name, signature = (x, a, $p2=$d2, $p3=$d3))]
        pub fn $fn<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, f64>,
            a: PyReadonlyArray1<'py, f64>,
            $p2: bool,
            $p3: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            let v = crate::par::map2(py, x.as_slice().unwrap(), a.as_slice().unwrap(), |x, a| {
                $sc(x, a, $p2, $p3)
            });
            v.into_pyarray(py)
        }
    };
}

wrap2!("ppois", ppois, ppois_scalar, (lower_tail = true), (log_p = false));
wrap2!("qpois", qpois, qpois_scalar, (lower_tail = true), (log_p = false));

#[pyfunction]
#[pyo3(name = "dpois", signature = (x, lam, give_log=false))]
pub fn dpois<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    lam: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map2(py, x.as_slice().unwrap(), lam.as_slice().unwrap(), |x, l| {
        dpois_scalar(x, l, give_log)
    });
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "pbinom", signature = (x, n, p, lower_tail=true, log_p=false))]
pub fn pbinom<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        n.as_slice().unwrap(),
        p.as_slice().unwrap(),
        |x, n, p| pbinom_scalar(x, n, p, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dbinom", signature = (x, n, p, give_log=false))]
pub fn dbinom<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        n.as_slice().unwrap(),
        p.as_slice().unwrap(),
        |x, n, p| dbinom_scalar(x, n, p, give_log),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qbinom", signature = (p, n, pr, lower_tail=true, log_p=false))]
pub fn qbinom<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    pr: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        p.as_slice().unwrap(),
        n.as_slice().unwrap(),
        pr.as_slice().unwrap(),
        |p, n, pr| qbinom_scalar(p, n, pr, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dbeta", signature = (x, a, b, give_log=false))]
pub fn dbeta<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map3(
        py,
        x.as_slice().unwrap(),
        a.as_slice().unwrap(),
        b.as_slice().unwrap(),
        |x, a, b| dbeta_scalar(x, a, b, give_log),
    );
    v.into_pyarray(py)
}
