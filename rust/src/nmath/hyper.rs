//! `dhyper` / `phyper` — hypergeometric density/CDF (nmath/dhyper.c, phyper.c).
//! Mirror of the `hea/R/nmath.py` hypergeometric cluster. dhyper is exact-double
//! (routes through dbinom_raw). phyper's `pdhyper` ratio sums a short converging
//! `LDOUBLE` series in R; f64 here (residual confined to that tail sum).
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::gamma::{r_dt_qiv, r_log1_exp};
use super::loader::dbinom_raw;
use super::norm::{dt0, dt1};
use super::toms708::lbeta_scalar;

#[inline]
fn r_nonint(x: f64) -> bool {
    // R_nonint: |x - nearbyint(x)| > 1e-9 * max(1, |x|); nearbyint = round-half-even
    (x - x.round_ties_even()).abs() > 1e-9 * 1.0f64.max(x.abs())
}
#[inline]
fn r_forceint(x: f64) -> f64 {
    x.round_ties_even()
}

pub(crate) fn dhyper_scalar(mut x: f64, mut r: f64, mut b: f64, mut n: f64, give_log: bool) -> f64 {
    if x.is_nan() || r.is_nan() || b.is_nan() || n.is_nan() {
        return x + r + b + n;
    }
    if (r < 0.0 || r_nonint(r)) || (b < 0.0 || r_nonint(b)) || (n < 0.0 || r_nonint(n)) || n > r + b
    {
        return f64::NAN;
    }
    if x < 0.0 {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if r_nonint(x) {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    x = r_forceint(x);
    r = r_forceint(r);
    b = r_forceint(b);
    n = r_forceint(n);
    if n < x || r < x || n - x > b {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if n == 0.0 {
        return if x == 0.0 {
            if give_log {
                0.0
            } else {
                1.0
            }
        } else if give_log {
            f64::NEG_INFINITY
        } else {
            0.0
        };
    }
    let p = n / (r + b);
    let q = (r + b - n) / (r + b);
    let p1 = dbinom_raw(x, r, p, q, give_log);
    let p2 = dbinom_raw(n - x, b, p, q, give_log);
    let p3 = dbinom_raw(n, r + b, p, q, give_log);
    if give_log {
        p1 + p2 - p3
    } else {
        p1 * p2 / p3
    }
}

fn pdhyper(mut x: f64, nr: f64, nb: f64, n: f64, log_p: bool) -> f64 {
    let mut sum = 0.0f64;
    let mut term = 1.0f64;
    while x > 0.0 && term >= f64::EPSILON * sum {
        term *= x * (nb - n + x) / (n + 1.0 - x) / (nr + 1.0 - x);
        sum += term;
        x -= 1.0;
    }
    if log_p {
        sum.ln_1p()
    } else {
        1.0 + sum
    }
}

pub(crate) fn phyper_scalar(
    mut x: f64,
    mut nr: f64,
    mut nb: f64,
    mut n: f64,
    mut lower_tail: bool,
    log_p: bool,
) -> f64 {
    if x.is_nan() || nr.is_nan() || nb.is_nan() || n.is_nan() {
        return x + nr + nb + n;
    }
    x = (x + 1e-7).floor();
    nr = r_forceint(nr);
    nb = r_forceint(nb);
    n = r_forceint(n);
    if nr < 0.0 || nb < 0.0 || !(nr + nb).is_finite() || n < 0.0 || n > nr + nb {
        return f64::NAN;
    }
    if x * (nr + nb) > n * nr {
        std::mem::swap(&mut nr, &mut nb);
        x = n - x - 1.0;
        lower_tail = !lower_tail;
    }
    if x < 0.0 || x < n - nb {
        return dt0(lower_tail, log_p);
    }
    if x >= nr || x >= n {
        return dt1(lower_tail, log_p);
    }
    let d = dhyper_scalar(x, nr, nb, n, log_p);
    if (!log_p && d == 0.0) || (log_p && d == f64::NEG_INFINITY) {
        return dt0(lower_tail, log_p);
    }
    let pd = pdhyper(x, nr, nb, n, log_p);
    if log_p {
        if lower_tail {
            d + pd
        } else {
            r_log1_exp(d + pd)
        }
    } else if lower_tail {
        d * pd
    } else {
        0.5 - d * pd + 0.5
    }
}

#[inline]
fn lfastchoose(n: f64, k: f64) -> f64 {
    // choose.c: lfastchoose(n, k) = -log(n+1) - lbeta(n-k+1, k+1)
    -(n + 1.0).ln() - lbeta_scalar(n - k + 1.0, k + 1.0)
}

pub(crate) fn qhyper_scalar(
    mut p: f64,
    mut nr: f64,
    mut nb: f64,
    mut n: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    if p.is_nan() || nr.is_nan() || nb.is_nan() || n.is_nan() {
        return p + nr + nb + n;
    }
    if !p.is_finite() || !nr.is_finite() || !nb.is_finite() || !n.is_finite() {
        return f64::NAN;
    }
    nr = r_forceint(nr);
    nb = r_forceint(nb);
    let nn = nr + nb; // N
    n = r_forceint(n);
    if nr < 0.0 || nb < 0.0 || n < 0.0 || n > nn {
        return f64::NAN;
    }
    // Find xr (= #red in sample) with phyper(xr) >= p > phyper(xr-1).
    let xstart = 0.0f64.max(n - nb);
    let xend = n.min(nr);
    // R_Q_P01_boundaries(p, xstart, xend)
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { xend } else { xstart };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail { xstart } else { xend };
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { xstart } else { xend };
        }
        if p == 1.0 {
            return if lower_tail { xend } else { xstart };
        }
    }
    let mut xr = xstart;
    let mut xb = n - xr; // #black in sample
    let small_n = nn < 1000.0; // no underflow in the product below
    let mut term = lfastchoose(nr, xr) + lfastchoose(nb, xb) - lfastchoose(nn, n);
    if small_n {
        term = term.exp();
    }
    nr -= xr;
    nb -= xb;
    if !lower_tail || log_p {
        p = r_dt_qiv(p, lower_tail, log_p);
    }
    p *= 1.0 - 1000.0 * f64::EPSILON; // was 64, but failed on FreeBSD sometimes
    let mut sum = if small_n { term } else { term.exp() };
    while sum < p && xr < xend {
        xr += 1.0;
        nb += 1.0;
        if small_n {
            term *= (nr / xr) * (xb / nb);
        } else {
            term += ((nr / xr) * (xb / nb)).ln();
        }
        sum += if small_n { term } else { term.exp() };
        xb -= 1.0;
        nr -= 1.0;
    }
    xr
}

// === PyO3 wrappers ===========================================================
#[pyfunction]
#[pyo3(name = "dhyper", signature = (x, r, b, n, give_log=false))]
pub fn dhyper<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        x.as_slice().unwrap(),
        r.as_slice().unwrap(),
        b.as_slice().unwrap(),
        n.as_slice().unwrap(),
        |x, r, b, n| dhyper_scalar(x, r, b, n, give_log),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "phyper", signature = (x, nr, nb, n, lower_tail=true, log_p=false))]
pub fn phyper<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    nr: PyReadonlyArray1<'py, f64>,
    nb: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        x.as_slice().unwrap(),
        nr.as_slice().unwrap(),
        nb.as_slice().unwrap(),
        n.as_slice().unwrap(),
        |x, nr, nb, n| phyper_scalar(x, nr, nb, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qhyper", signature = (p, nr, nb, n, lower_tail=true, log_p=false))]
pub fn qhyper<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    nr: PyReadonlyArray1<'py, f64>,
    nb: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        p.as_slice().unwrap(),
        nr.as_slice().unwrap(),
        nb.as_slice().unwrap(),
        n.as_slice().unwrap(),
        |p, nr, nb, n| qhyper_scalar(p, nr, nb, n, lower_tail, log_p),
    );
    v.into_pyarray(py)
}
