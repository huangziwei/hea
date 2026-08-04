//! Continuous "second-half" families — R's nmath dcauchy/pcauchy/qcauchy,
//! dlogis/plogis/qlogis, dlnorm/plnorm/qlnorm, dweibull/pweibull/qweibull.
//! Mirror of the `hea/R/nmath.py` scalar kernels. All closed-form (no LDOUBLE
//! series) → f64 is 0-ulp to R here.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64::consts::PI;

use super::consts::{M_1_SQRT_2PI, M_LN_SQRT_2PI};
use super::gamma::{r_dt_clog, r_log1_exp};
use super::norm::{dt0, dt1, pnorm5_scalar, qnorm5_scalar};
use super::tf::tanpi;
use super::util::rfma;

// --- dpq.h helpers not already centralised -----------------------------------
#[inline]
fn r_d_val(x: f64, log_p: bool) -> f64 {
    // R_D_val(x) = log_p ? log(x) : x
    if log_p {
        x.ln()
    } else {
        x
    }
}
#[inline]
fn r_d_clog(p: f64, log_p: bool) -> f64 {
    // R_D_Clog(p) = log_p ? log1p(-p) : (0.5 - p + 0.5)
    if log_p {
        (-p).ln_1p()
    } else {
        0.5 - p + 0.5
    }
}

/// R's `log1pexp(x) = log(1 + exp(x))` (plogis.c), overflow-safe.
#[inline]
fn log1pexp(x: f64) -> f64 {
    if x <= 18.0 {
        x.exp().ln_1p()
    } else if x > 33.3 {
        x
    } else {
        x + (-x).exp()
    }
}

/// Faithful expansion of the `R_Q_P01_boundaries(p, left, right)` macro: returns
/// `Some(boundary)` when `p` is out of range or at 0/1, else `None`.
#[inline]
fn q_p01_boundaries(p: f64, lower_tail: bool, log_p: bool, left: f64, right: f64) -> Option<f64> {
    if log_p {
        if p > 0.0 {
            return Some(f64::NAN);
        }
        if p == 0.0 {
            return Some(if lower_tail { right } else { left });
        }
        if p == f64::NEG_INFINITY {
            return Some(if lower_tail { left } else { right });
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return Some(f64::NAN);
        }
        if p == 0.0 {
            return Some(if lower_tail { left } else { right });
        }
        if p == 1.0 {
            return Some(if lower_tail { right } else { left });
        }
    }
    None
}

// === Cauchy ==================================================================
pub(crate) fn dcauchy_scalar(x: f64, location: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || location.is_nan() || scale.is_nan() {
        return x + location + scale;
    }
    if scale <= 0.0 {
        return f64::NAN;
    }
    let y = (x - location) / scale;
    if give_log {
        -(PI * scale * (1.0 + y * y)).ln()
    } else {
        1.0 / (PI * scale * (1.0 + y * y))
    }
}

pub(crate) fn pcauchy_scalar(
    mut x: f64,
    location: f64,
    scale: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x.is_nan() || location.is_nan() || scale.is_nan() {
        return x + location + scale;
    }
    if scale <= 0.0 {
        return f64::NAN;
    }
    x = (x - location) / scale;
    if x.is_nan() {
        return f64::NAN;
    }
    if !x.is_finite() {
        return if x < 0.0 {
            dt0(lower_tail, log_p)
        } else {
            dt1(lower_tail, log_p)
        };
    }
    if !lower_tail {
        x = -x;
    }
    // Installed R (no HAVE_ATANPI) uses the atan(1/x)/M_PI branch.
    if x.abs() > 1.0 {
        let y = (1.0 / x).atan() / PI;
        if x > 0.0 {
            r_d_clog(y, log_p)
        } else {
            r_d_val(-y, log_p)
        }
    } else {
        r_d_val(0.5 + x.atan() / PI, log_p)
    }
}

pub(crate) fn qcauchy_scalar(
    mut p: f64,
    location: f64,
    scale: f64,
    mut lower_tail: bool,
    log_p: bool,
) -> f64 {
    if p.is_nan() || location.is_nan() || scale.is_nan() {
        return p + location + scale;
    }
    // R_Q_P01_check(p)
    if (log_p && p > 0.0) || (!log_p && (p < 0.0 || p > 1.0)) {
        return f64::NAN;
    }
    if scale <= 0.0 || !scale.is_finite() {
        if scale == 0.0 {
            return location;
        }
        return f64::NAN;
    }
    // my_INF := location + (lower_tail ? scale : -scale) * +Inf  (original lower_tail)
    let my_inf = location + (if lower_tail { scale } else { -scale }) * f64::INFINITY;
    if log_p {
        if p > -1.0 {
            if p == 0.0 {
                return my_inf;
            }
            lower_tail = !lower_tail;
            p = -p.exp_m1();
        } else {
            p = p.exp();
        }
    } else if p > 0.5 {
        if p == 1.0 {
            return my_inf;
        }
        p = 1.0 - p;
        lower_tail = !lower_tail;
    }
    if p == 0.5 {
        return location;
    }
    if p == 0.0 {
        return location + (if lower_tail { scale } else { -scale }) * f64::NEG_INFINITY;
    }
    location + (if lower_tail { -scale } else { scale }) / tanpi(p)
}

// === Logistic ================================================================
pub(crate) fn dlogis_scalar(x: f64, location: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || location.is_nan() || scale.is_nan() {
        return x + location + scale;
    }
    if scale <= 0.0 {
        return f64::NAN;
    }
    let x = ((x - location) / scale).abs();
    let e = (-x).exp();
    let f = 1.0 + e;
    if give_log {
        -(x + (scale * f * f).ln())
    } else {
        e / (scale * f * f)
    }
}

pub(crate) fn plogis_scalar(
    x: f64,
    location: f64,
    scale: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x.is_nan() || location.is_nan() || scale.is_nan() {
        return x + location + scale;
    }
    if scale <= 0.0 {
        return f64::NAN;
    }
    let x = (x - location) / scale;
    if x.is_nan() {
        return f64::NAN;
    }
    // R_P_bounds_Inf_01(x)
    if !x.is_finite() {
        return if x > 0.0 {
            dt1(lower_tail, log_p)
        } else {
            dt0(lower_tail, log_p)
        };
    }
    if log_p {
        -log1pexp(if lower_tail { -x } else { x })
    } else {
        1.0 / (1.0 + (if lower_tail { -x } else { x }).exp())
    }
}

pub(crate) fn qlogis_scalar(
    mut p: f64,
    location: f64,
    scale: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if p.is_nan() || location.is_nan() || scale.is_nan() {
        return p + location + scale;
    }
    if let Some(b) = q_p01_boundaries(p, lower_tail, log_p, f64::NEG_INFINITY, f64::INFINITY) {
        return b;
    }
    if scale < 0.0 {
        return f64::NAN;
    }
    if scale == 0.0 {
        return location;
    }
    // p := logit(p) = log(p / (1-p))
    p = if log_p {
        if lower_tail {
            p - r_log1_exp(p)
        } else {
            r_log1_exp(p) - p
        }
    } else if lower_tail {
        (p / (1.0 - p)).ln()
    } else {
        ((1.0 - p) / p).ln()
    };
    location + scale * p
}

// === Log-normal ==============================================================
pub(crate) fn dlnorm_scalar(x: f64, meanlog: f64, sdlog: f64, give_log: bool) -> f64 {
    if x.is_nan() || meanlog.is_nan() || sdlog.is_nan() {
        return x + meanlog + sdlog;
    }
    if sdlog < 0.0 {
        return f64::NAN;
    }
    if !x.is_finite() && x.ln() == meanlog {
        return f64::NAN; // log(x) - meanlog is NaN
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if sdlog == 0.0 {
        return if x.ln() == meanlog {
            f64::INFINITY
        } else {
            rd0
        };
    }
    if x <= 0.0 {
        return rd0;
    }
    let y = (x.ln() - meanlog) / sdlog;
    if give_log {
        // R: `-(M_LN_SQRT_2PI + 0.5 * y * y + log(x * sdlog))`; clang contracts
        // `M_LN_SQRT_2PI + (0.5*y)*y` into one fmadd on arm64, so `rfma` (=
        // plain `a*b+c` on x86) is what keeps this 0-ulp to R on both arches.
        -(rfma(0.5 * y, y, M_LN_SQRT_2PI) + (x * sdlog).ln())
    } else {
        M_1_SQRT_2PI * (-0.5 * y * y).exp() / (x * sdlog)
    }
}

pub(crate) fn plnorm_scalar(
    x: f64,
    meanlog: f64,
    sdlog: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x.is_nan() || meanlog.is_nan() || sdlog.is_nan() {
        return x + meanlog + sdlog;
    }
    if sdlog < 0.0 {
        return f64::NAN;
    }
    if x > 0.0 {
        pnorm5_scalar(x.ln(), meanlog, sdlog, lower_tail, log_p)
    } else {
        dt0(lower_tail, log_p)
    }
}

pub(crate) fn qlnorm_scalar(
    p: f64,
    meanlog: f64,
    sdlog: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if p.is_nan() || meanlog.is_nan() || sdlog.is_nan() {
        return p + meanlog + sdlog;
    }
    if let Some(b) = q_p01_boundaries(p, lower_tail, log_p, 0.0, f64::INFINITY) {
        return b;
    }
    qnorm5_scalar(p, meanlog, sdlog, lower_tail, log_p).exp()
}

// === Weibull =================================================================
pub(crate) fn dweibull_scalar(x: f64, shape: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || shape.is_nan() || scale.is_nan() {
        return x + shape + scale;
    }
    if shape <= 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if x < 0.0 {
        return rd0;
    }
    if !x.is_finite() {
        return rd0;
    }
    if x == 0.0 && shape < 1.0 {
        return f64::INFINITY;
    }
    let tmp1 = (x / scale).powf(shape - 1.0);
    let tmp2 = tmp1 * (x / scale);
    if give_log {
        -tmp2 + (shape * tmp1 / scale).ln()
    } else {
        shape * tmp1 * (-tmp2).exp() / scale
    }
}

pub(crate) fn pweibull_scalar(
    x: f64,
    shape: f64,
    scale: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x.is_nan() || shape.is_nan() || scale.is_nan() {
        return x + shape + scale;
    }
    if shape <= 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    let x = -(x / scale).powf(shape);
    if lower_tail {
        if log_p {
            r_log1_exp(x)
        } else {
            -x.exp_m1()
        }
    } else if log_p {
        x
    } else {
        x.exp()
    }
}

pub(crate) fn qweibull_scalar(
    p: f64,
    shape: f64,
    scale: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if p.is_nan() || shape.is_nan() || scale.is_nan() {
        return p + shape + scale;
    }
    if shape <= 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if let Some(b) = q_p01_boundaries(p, lower_tail, log_p, 0.0, f64::INFINITY) {
        return b;
    }
    scale * (-r_dt_clog(p, lower_tail, log_p)).powf(1.0 / shape)
}

// === PyO3 wrappers ===========================================================
macro_rules! wrap_d3 {
    ($name:literal, $fn:ident, $sc:path) => {
        #[pyfunction]
        #[pyo3(name = $name, signature = (x, a, b, give_log=false))]
        pub fn $fn<'py>(
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
                |x, a, b| $sc(x, a, b, give_log),
            );
            v.into_pyarray(py)
        }
    };
}

macro_rules! wrap_pq3 {
    ($name:literal, $fn:ident, $sc:path) => {
        #[pyfunction]
        #[pyo3(name = $name, signature = (x, a, b, lower_tail=true, log_p=false))]
        pub fn $fn<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, f64>,
            a: PyReadonlyArray1<'py, f64>,
            b: PyReadonlyArray1<'py, f64>,
            lower_tail: bool,
            log_p: bool,
        ) -> Bound<'py, PyArray1<f64>> {
            let v = crate::par::map3(
                py,
                x.as_slice().unwrap(),
                a.as_slice().unwrap(),
                b.as_slice().unwrap(),
                |x, a, b| $sc(x, a, b, lower_tail, log_p),
            );
            v.into_pyarray(py)
        }
    };
}

wrap_d3!("dcauchy", dcauchy, dcauchy_scalar);
wrap_pq3!("pcauchy", pcauchy, pcauchy_scalar);
wrap_pq3!("qcauchy", qcauchy, qcauchy_scalar);
wrap_d3!("dlogis", dlogis, dlogis_scalar);
wrap_pq3!("plogis", plogis, plogis_scalar);
wrap_pq3!("qlogis", qlogis, qlogis_scalar);
wrap_d3!("dlnorm", dlnorm, dlnorm_scalar);
wrap_pq3!("plnorm", plnorm, plnorm_scalar);
wrap_pq3!("qlnorm", qlnorm, qlnorm_scalar);
wrap_d3!("dweibull", dweibull, dweibull_scalar);
wrap_pq3!("pweibull", pweibull, pweibull_scalar);
wrap_pq3!("qweibull", qweibull, qweibull_scalar);
