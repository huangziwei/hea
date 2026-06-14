//! `dexp` / `pexp` / `qexp` — R's nmath dexp.c / pexp.c / qexp.c.
//! Mirror of the `hea/R/nmath.py` exponential cluster.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::gamma::{r_dt_clog, r_log1_exp};
use super::norm::dt0;

pub(crate) fn dexp_scalar(x: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || scale.is_nan() {
        return x + scale;
    }
    if scale <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if give_log {
        (-x / scale) - scale.ln()
    } else {
        (-x / scale).exp() / scale
    }
}

pub(crate) fn pexp_scalar(x: f64, scale: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || scale.is_nan() {
        return x + scale;
    }
    if scale < 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    let x = -(x / scale);
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

pub(crate) fn qexp_scalar(p: f64, scale: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || scale.is_nan() {
        return p + scale;
    }
    if scale < 0.0 {
        return f64::NAN;
    }
    if (log_p && p > 0.0) || (!log_p && (p < 0.0 || p > 1.0)) {
        return f64::NAN;
    }
    if p == dt0(lower_tail, log_p) {
        return 0.0;
    }
    -scale * r_dt_clog(p, lower_tail, log_p)
}

#[pyfunction]
#[pyo3(name = "dexp", signature = (x, scale, give_log=false))]
pub fn dexp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (xv, sv) = (x.as_array(), scale.as_array());
    let v: Vec<f64> = (0..xv.len()).map(|i| dexp_scalar(xv[i], sv[i], give_log)).collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "pexp", signature = (x, scale, lower_tail=true, log_p=false))]
pub fn pexp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (xv, sv) = (x.as_array(), scale.as_array());
    let v: Vec<f64> = (0..xv.len())
        .map(|i| pexp_scalar(xv[i], sv[i], lower_tail, log_p))
        .collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qexp", signature = (p, scale, lower_tail=true, log_p=false))]
pub fn qexp<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (pv, sv) = (p.as_array(), scale.as_array());
    let v: Vec<f64> = (0..pv.len())
        .map(|i| qexp_scalar(pv[i], sv[i], lower_tail, log_p))
        .collect();
    v.into_pyarray(py)
}
