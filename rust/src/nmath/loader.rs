//! Loader's saddlepoint kernels — R's stirlerr.c / bd0.c / dpois.c / dbinom.c.
//! Mirror of the `hea/R/nmath.py` loader cluster. Scalar functions below; the
//! `#[pyfunction]` wrappers map them over numpy arrays (Python pre-broadcasts).
#![allow(dead_code)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::bd0_scale::BD0_SCALE;
use super::coeffs::{S, STIRLERR_HALVES};
use super::consts::{DBL_MIN, M_2PI, M_LN2, M_LN_2PI, M_LN_SQRT_2PI, M_SQRT_2PI, X_LRG};
use super::lgamma::lgammafn;
use super::util::{frexp, ldexp};

/// R's `stirlerr(n) = log(n!) - log(sqrt(2*pi*n)*(n/e)^n)`.
pub fn stirlerr(n: f64) -> f64 {
    if n <= 23.5 {
        let nn2 = n + n;
        if n <= 15.0 && nn2 == nn2.trunc() {
            return STIRLERR_HALVES[nn2 as usize];
        }
        if n >= 1.0 && n <= 5.25 {
            let l_n = n.ln();
            return lgammafn(n) + n * (1.0 - l_n) + (l_n - M_LN_2PI) * 0.5;
        }
        if n < 1.0 {
            return lgammafn(1.0 + n) - (n + 0.5) * n.ln() + n - M_LN_SQRT_2PI;
        }
        // 5.25 < n <= 23.5 — asymptotic series, length by threshold
        let nn = n * n;
        if n > 12.8 {
            return (S[0]-(S[1]-(S[2]-(S[3]-(S[4]-(S[5]-S[6]/nn)/nn)/nn)/nn)/nn)/nn)/n;
        }
        if n > 12.3 {
            return (S[0]-(S[1]-(S[2]-(S[3]-(S[4]-(S[5]-(S[6]-S[7]/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        }
        if n > 8.9 {
            return (S[0]-(S[1]-(S[2]-(S[3]-(S[4]-(S[5]-(S[6]-(S[7]-S[8]/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        }
        if n > 7.3 {
            return (S[0]-(S[1]-(S[2]-(S[3]-(S[4]-(S[5]-(S[6]-(S[7]-(S[8]-(S[9]-S[10]/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        }
        if n > 6.6 {
            return (S[0]-(S[1]-(S[2]-(S[3]-(S[4]-(S[5]-(S[6]-(S[7]-(S[8]-(S[9]-(S[10]-(S[11]-S[12]/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        }
        if n > 6.1 {
            return (S[0]-(S[1]-(S[2]-(S[3]-(S[4]-(S[5]-(S[6]-(S[7]-(S[8]-(S[9]-(S[10]-(S[11]-(S[12]-(S[13]-S[14]/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        }
        return (S[0]-(S[1]-(S[2]-(S[3]-(S[4]-(S[5]-(S[6]-(S[7]-(S[8]-(S[9]-(S[10]-(S[11]-(S[12]-(S[13]-(S[14]-(S[15]-S[16]/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
    }
    // n > 23.5
    let nn = n * n;
    if n > 15.7e6 {
        return S[0] / n;
    }
    if n > 6180.0 {
        return (S[0] - S[1] / nn) / n;
    }
    if n > 205.0 {
        return (S[0] - (S[1] - S[2] / nn) / nn) / n;
    }
    if n > 86.0 {
        return (S[0] - (S[1] - (S[2] - S[3] / nn) / nn) / nn) / n;
    }
    if n > 27.0 {
        return (S[0] - (S[1] - (S[2] - (S[3] - S[4] / nn) / nn) / nn) / nn) / n;
    }
    (S[0] - (S[1] - (S[2] - (S[3] - (S[4] - S[5] / nn) / nn) / nn) / nn) / nn) / n
}

/// R's `bd0(x, np) = x*log(x/np) + np - x`, accurate near `x/np == 1`.
pub fn bd0(x: f64, np_: f64) -> f64 {
    if !(x.is_finite() && np_.is_finite() && np_ != 0.0) {
        return f64::NAN;
    }
    if (x - np_).abs() < 0.1 * (x + np_) {
        // close: Taylor series
        let d = x - np_;
        let mut v = d / (x + np_);
        if d != 0.0 && v == 0.0 {
            let x_ = ldexp(x, -2);
            let n_ = ldexp(np_, -2);
            v = (x_ - n_) / (x_ + n_);
        }
        let mut s = ldexp(d, -1) * v;
        let s2 = ldexp(s, 1);
        if s2.abs() < DBL_MIN {
            return s2;
        }
        let mut ej = x * v;
        let v2 = v * v;
        let mut j = 1i64;
        loop {
            ej *= v2;
            let s_old = s;
            s += ej / (((j << 1) + 1) as f64);
            if s == s_old {
                break;
            }
            j += 1;
            if j >= 1000 {
                break;
            }
        }
        ldexp(s, 1)
    } else {
        // far: direct
        let xnp = x / np_;
        let lg = if xnp.is_finite() {
            xnp.ln()
        } else {
            x.ln() - np_.ln()
        };
        if x > np_ {
            x * (lg - 1.0) + np_
        } else {
            x * lg + np_ - x
        }
    }
}

/// R's `ebd0(x, M)` (Welinder) — returns `(yh, yl)` with `yh+yl = x*log(x/M)+(M-x)`.
pub fn ebd0(x: f64, m: f64) -> (f64, f64) {
    const SB: i32 = 10;
    let s_f = 1024.0_f64; // S = 1 << Sb
    let n_f = 128.0_f64; // N

    if x == m {
        return (0.0, 0.0);
    }
    if x == 0.0 {
        return (m, 0.0);
    }
    if m == 0.0 {
        return (f64::INFINITY, 0.0);
    }
    let mox = m / x;
    if mox == f64::INFINITY {
        return (m, 0.0);
    }
    let (r, e) = frexp(mox);
    // overflow guard: M_LN2 * (-e) > 1 + DBL_MAX/x
    if M_LN2 * (-(e as f64)) > 1.0 + f64::MAX / x {
        return (f64::INFINITY, 0.0);
    }
    let i = ((r - 0.5) * (2.0 * n_f) + 0.5).floor() as i64;
    let f = (s_f / (0.5 + (i as f64) / (2.0 * n_f)) + 0.5).floor();
    let fg = ldexp(f, -(e + SB));
    if fg == f64::INFINITY {
        return (f64::INFINITY, 0.0);
    }

    let mut lh = 0.0;
    let mut ll = 0.0;
    // ADD1(d): d1 = floor(d+0.5); lh += d1; ll += d - d1
    macro_rules! add1 {
        ($d:expr) => {{
            let d = $d;
            let d1 = (d + 0.5).floor();
            lh += d1;
            ll += d - d1;
        }};
    }

    // ADD1(-x * log1pmx((M*fg - x)/x)) — R's ebd0 uses the accurate log1pmx.
    let arg = (m * fg - x) / x;
    add1!(-x * super::gamma::log1pmx(arg));

    if fg != 1.0 {
        let iu = i as usize;
        for j in 0..4 {
            add1!(x * BD0_SCALE[iu][j]);
            add1!(-x * BD0_SCALE[0][j] * (e as f64));
            if !lh.is_finite() {
                return (f64::INFINITY, 0.0);
            }
        }
        add1!(m);
        add1!(-m * fg);
    }
    (lh, ll)
}

/// R's `pow1p(x, y) = (1+x)^y`, accurate for `|x| << 1`.
pub fn pow1p(x: f64, y: f64) -> f64 {
    if y.is_nan() {
        return if x == 0.0 { 1.0 } else { y };
    }
    if y == y.trunc() && y >= 0.0 && y <= 4.0 {
        return match y as i32 {
            0 => 1.0,
            1 => x + 1.0,
            2 => x * (x + 2.0) + 1.0,
            3 => x * (x * (x + 3.0) + 3.0) + 1.0,
            _ => x * (x * (x * (x + 4.0) + 6.0) + 4.0) + 1.0,
        };
    }
    let xp1 = x + 1.0;
    let x_ = xp1 - 1.0;
    if x_ == x || x.abs() > 0.5 || x.is_nan() {
        xp1.powf(y)
    } else {
        (y * x.ln_1p()).exp()
    }
}

/// R's `dpois_raw(x, lambda, give_log)` (dpois.c) — Loader saddlepoint w/ ebd0.
pub fn dpois_raw(x: f64, lam: f64, give_log: bool) -> f64 {
    let tiny = DBL_MIN; // np.finfo(float).tiny (smallest normal)
    if lam == 0.0 {
        let lr = if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
        return if give_log { lr } else { lr.exp() };
    }
    if !lam.is_finite() {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if x < 0.0 {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    if x <= lam * tiny {
        let lr = -lam;
        return if give_log { lr } else { lr.exp() };
    }
    if lam < x * tiny {
        let lr = if !x.is_finite() {
            f64::NEG_INFINITY
        } else {
            -lam + x * lam.ln() - lgammafn(x + 1.0)
        };
        return if give_log { lr } else { lr.exp() };
    }
    // saddlepoint
    let (yh, yl) = ebd0(x, lam);
    let yl_total = yl + stirlerr(x);
    let lrg = x >= X_LRG;
    let r = if lrg { M_SQRT_2PI * x.sqrt() } else { M_2PI * x };
    let log_correction = if lrg { r.ln() } else { 0.5 * r.ln() };
    if give_log {
        -yl_total - yh - log_correction
    } else {
        (-yl_total).exp() * (-yh).exp() / (if lrg { r } else { r.sqrt() })
    }
}

/// R's `dbinom_raw(x, n, p, q, give_log)` (dbinom.c) — Loader saddlepoint w/ bd0.
pub fn dbinom_raw(x: f64, n: f64, p: f64, q: f64, give_log: bool) -> f64 {
    if p == 0.0 {
        let lr = if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
        return if give_log { lr } else { lr.exp() };
    }
    if q == 0.0 {
        let lr = if x == n { 0.0 } else { f64::NEG_INFINITY };
        return if give_log { lr } else { lr.exp() };
    }
    if x == 0.0 {
        if n == 0.0 {
            return if give_log { 0.0 } else { 1.0 };
        }
        if give_log {
            return if p > q { n * q.ln() } else { n * (-p).ln_1p() };
        }
        return if p > q { q.powf(n) } else { pow1p(-p, n) };
    }
    if x == n {
        if give_log {
            return if p > q { n * (-q).ln_1p() } else { n * p.ln() };
        }
        return if p > q { pow1p(-q, n) } else { p.powf(n) };
    }
    if x < 0.0 || x > n {
        return if give_log { f64::NEG_INFINITY } else { 0.0 };
    }
    // saddlepoint
    let lc =
        stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    let lf = M_LN_2PI + x.ln() + (-x / n).ln_1p();
    let lr = lc - 0.5 * lf;
    if give_log {
        lr
    } else {
        lr.exp()
    }
}

// === PyO3 wrappers (numpy-vectorized; Python pre-broadcasts equal-length) =====

#[pyfunction]
#[pyo3(name = "stirlerr")]
pub fn py_stirlerr<'py>(py: Python<'py>, n: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let v: Vec<f64> = n.as_array().iter().map(|&x| stirlerr(x)).collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "bd0")]
pub fn py_bd0<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    np_: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let xv = x.as_array();
    let nv = np_.as_array();
    let v: Vec<f64> = xv.iter().zip(nv.iter()).map(|(&a, &b)| bd0(a, b)).collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "pow1p")]
pub fn py_pow1p<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let xv = x.as_array();
    let yv = y.as_array();
    let v: Vec<f64> = xv.iter().zip(yv.iter()).map(|(&a, &b)| pow1p(a, b)).collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dpois_raw", signature = (x, lam, give_log=true))]
pub fn py_dpois_raw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    lam: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let xv = x.as_array();
    let lv = lam.as_array();
    let v: Vec<f64> = xv
        .iter()
        .zip(lv.iter())
        .map(|(&a, &b)| dpois_raw(a, b, give_log))
        .collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dbinom_raw", signature = (x, n, p, q, give_log=true))]
pub fn py_dbinom_raw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let xv = x.as_array();
    let nv = n.as_array();
    let pv = p.as_array();
    let qv = q.as_array();
    let mut out = Vec::with_capacity(xv.len());
    for i in 0..xv.len() {
        out.push(dbinom_raw(xv[i], nv[i], pv[i], qv[i], give_log));
    }
    out.into_pyarray(py)
}
