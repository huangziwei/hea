#![allow(dead_code)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::coeffs::{ALGMCS, GAMCS};
use super::consts::{
    M_LN_SQRT_PId2, GAM_XMAX, GAM_XMIN, GAM_XSML, LGC_XBIG, LGM_XMAX, M_LN_SQRT_2PI, NALGM, NGAM,
};
use super::loader::stirlerr;
use super::util::rfma;

const PI: f64 = std::f64::consts::PI;

pub fn chebyshev_eval(x: f64, a: &[f64], n: usize) -> f64 {
    if n < 1 || n > 1000 {
        return f64::NAN;
    }
    if x < -1.1 || x > 1.1 {
        return f64::NAN;
    }
    let twox = x * 2.0;
    let mut b2 = 0.0;
    let mut b1 = 0.0;
    let mut b0 = 0.0;
    for i in 1..=n {
        b2 = b1;
        b1 = b0;
        b0 = rfma(twox, b1, -b2) + a[n - i];
    }
    (b0 - b2) * 0.5
}

pub fn lgammacor(x: f64) -> f64 {
    if x < 10.0 {
        return f64::NAN;
    }
    if x < LGC_XBIG {
        let tmp = 10.0 / x;
        return chebyshev_eval(rfma(tmp * tmp, 2.0, -1.0), &ALGMCS, NALGM) / x;
    }
    1.0 / (x * 12.0)
}

#[cfg(target_os = "macos")]
pub fn sinpi(x: f64) -> f64 {
    extern "C" {
        fn __sinpi(x: f64) -> f64;
    }
    unsafe { __sinpi(x) }
}

#[cfg(not(target_os = "macos"))]
pub fn sinpi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let mut x = x % 2.0; // fmod(x, 2.0)
    if x <= -1.0 {
        x += 2.0;
    } else if x > 1.0 {
        x -= 2.0;
    }
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    if x == 0.5 {
        return 1.0;
    }
    if x == -0.5 {
        return -1.0;
    }
    (PI * x).sin()
}

pub fn gammafn(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 || (x < 0.0 && x == x.trunc()) {
        return f64::NAN;
    }
    let y = x.abs();
    if y <= 10.0 {
        let mut n = x as i32;
        if x < 0.0 {
            n -= 1;
        }
        let frac = x - n as f64;
        n -= 1;
        let mut value = chebyshev_eval(rfma(frac, 2.0, -1.0), &GAMCS, NGAM) + 0.9375;
        if n == 0 {
            return value;
        }
        if n < 0 {
            if frac < GAM_XSML {
                return if x > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
            }
            let nn = -n;
            for i in 0..nn {
                value /= x + i as f64;
            }
            return value;
        }
        for i in 1..=n {
            value *= frac + i as f64;
        }
        return value;
    }
    if x > GAM_XMAX {
        return f64::INFINITY;
    }
    if x < GAM_XMIN {
        return 0.0;
    }
    let value;
    if y <= 50.0 && y == y.trunc() {
        let mut v = 1.0;
        let mut i = 2i64;
        let top = y as i64;
        while i < top {
            v *= i as f64;
            i += 1;
        }
        value = v;
    } else {
        let corr = if (2.0 * y) == (2.0 * y).trunc() {
            stirlerr(y)
        } else {
            lgammacor(y)
        };
        value = (rfma(y - 0.5, y.ln(), -y) + M_LN_SQRT_2PI + corr).exp();
    }
    if x > 0.0 {
        return value;
    }
    let sinpiy = sinpi(y);
    if sinpiy == 0.0 {
        return f64::INFINITY;
    }
    -PI / (y * sinpiy * value)
}

pub fn lgammafn(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x <= 0.0 && x == x.trunc() {
        return f64::INFINITY;
    }
    let y = x.abs();
    if y < 1e-306 {
        return -y.ln();
    }
    if y <= 10.0 {
        return gammafn(x).abs().ln();
    }
    if y > LGM_XMAX {
        return f64::INFINITY;
    }
    if x > 0.0 {
        if x > 1e17 {
            return x * (x.ln() - 1.0);
        } else if x > 4934720.0 {
            return rfma(x - 0.5, x.ln(), M_LN_SQRT_2PI) - x;
        }
        return rfma(x - 0.5, x.ln(), M_LN_SQRT_2PI) - x + lgammacor(x);
    }
    let sinpiy = sinpi(y).abs();
    rfma(x - 0.5, y.ln(), M_LN_SQRT_PId2) - x - sinpiy.ln() - lgammacor(y)
}

#[pyfunction]
#[pyo3(name = "lgammafn")]
pub fn py_lgammafn<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map1(py, x.as_slice().unwrap(), lgammafn);
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "gammafn")]
pub fn py_gammafn<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map1(py, x.as_slice().unwrap(), gammafn);
    v.into_pyarray(py)
}
