//! `pt` / `pf` / `dt` / `qt` / `qf` — R's nmath pt.c / pf.c / dt.c / qt.c / qf.c.
//! Mirror of the `hea/R/nmath.py` t/F cluster. These route through pbeta /
//! pgamma / qbeta / qgamma / qnorm / dnorm / bd0 / stirlerr.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::consts::{M_LN2, M_LN_SQRT_2PI, M_SQRT2};
use super::gamma::{pgamma_scalar, qgamma_scalar, r_d_lexp, r_d_log, r_dt_qiv};
use super::loader::{bd0, stirlerr};
use super::norm::{dnorm5_scalar, dt0, dt1, pnorm5_scalar, qnorm5_scalar};
use super::qbeta::qbeta_scalar;
use super::toms708::{lbeta_scalar, pbeta_scalar};

const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934;
const M_1_PI: f64 = 0.318309886183790671537767526745;
const M_PI_2: f64 = 1.570796326794896619231321691640;
const PI: f64 = std::f64::consts::PI;

pub(crate) fn pt_scalar(x: f64, n: f64, mut lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || n.is_nan() {
        return x + n;
    }
    if n <= 0.0 {
        return f64::NAN;
    }
    if !x.is_finite() {
        return if x < 0.0 {
            dt0(lower_tail, log_p)
        } else {
            dt1(lower_tail, log_p)
        };
    }
    if !n.is_finite() {
        return pnorm5_scalar(x, 0.0, 1.0, lower_tail, log_p);
    }
    let nx = 1.0 + (x / n) * x;
    let mut val;
    if nx > 1e100 {
        let lval = -0.5 * n * (2.0 * x.abs().ln() - n.ln()) - lbeta_scalar(0.5 * n, 0.5)
            - (0.5 * n).ln();
        val = if log_p { lval } else { lval.exp() };
    } else {
        val = if n > x * x {
            pbeta_scalar(x * x / (n + x * x), 0.5, n / 2.0, false, log_p)
        } else {
            pbeta_scalar(1.0 / nx, n / 2.0, 0.5, true, log_p)
        };
    }
    if x <= 0.0 {
        lower_tail = !lower_tail;
    }
    if log_p {
        if lower_tail {
            return (-0.5 * val.exp()).ln_1p();
        }
        return val - M_LN2;
    }
    val /= 2.0;
    if lower_tail {
        0.5 - val + 0.5
    } else {
        val
    }
}

pub(crate) fn pf_scalar(x: f64, df1: f64, df2: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || df1.is_nan() || df2.is_nan() {
        return x + df2 + df1;
    }
    if df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    if x >= f64::INFINITY {
        return dt1(lower_tail, log_p);
    }
    if df2 == f64::INFINITY {
        if df1 == f64::INFINITY {
            if x < 1.0 {
                return dt0(lower_tail, log_p);
            }
            if x == 1.0 {
                return if log_p { -M_LN2 } else { 0.5 };
            }
            return dt1(lower_tail, log_p);
        }
        return pgamma_scalar(x * df1, df1 / 2.0, 2.0, lower_tail, log_p);
    }
    if df1 == f64::INFINITY {
        return pgamma_scalar(df2 / x, df2 / 2.0, 2.0, !lower_tail, log_p);
    }
    let x2 = if df1 * x > df2 {
        pbeta_scalar(df2 / (df2 + df1 * x), df2 / 2.0, df1 / 2.0, !lower_tail, log_p)
    } else {
        pbeta_scalar(df1 * x / (df2 + df1 * x), df1 / 2.0, df2 / 2.0, lower_tail, log_p)
    };
    if !x2.is_nan() {
        x2
    } else {
        f64::NAN
    }
}

pub(crate) fn dt_scalar(x: f64, n: f64, give_log: bool) -> f64 {
    if x.is_nan() || n.is_nan() {
        return x + n;
    }
    if n <= 0.0 {
        return f64::NAN;
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if !x.is_finite() {
        return rd0;
    }
    if !n.is_finite() {
        return dnorm5_scalar(x, 0.0, 1.0, give_log);
    }
    let t = -bd0(n / 2.0, (n + 1.0) / 2.0) + stirlerr((n + 1.0) / 2.0) - stirlerr(n / 2.0);
    let x2n = x * x / n;
    let mut ax = 0.0;
    let lrg_x2n = x2n > 1.0 / f64::EPSILON;
    let l_x2n;
    let u;
    if lrg_x2n {
        ax = x.abs();
        l_x2n = ax.ln() - n.ln() / 2.0;
        u = n * l_x2n;
    } else if x2n > 0.2 {
        l_x2n = (1.0 + x2n).ln() / 2.0;
        u = n * l_x2n;
    } else {
        l_x2n = x2n.ln_1p() / 2.0;
        u = -bd0(n / 2.0, (n + x * x) / 2.0) + x * x / 2.0;
    }
    if give_log {
        return t - u - (M_LN_SQRT_2PI + l_x2n);
    }
    let i_sqrt = if lrg_x2n { n.sqrt() / ax } else { (-l_x2n).exp() };
    (t - u).exp() * M_1_SQRT_2PI * i_sqrt
}

pub(crate) fn tanpi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let mut x = x % 1.0;
    if x <= -0.5 {
        x += 1.0;
    } else if x > 0.5 {
        x -= 1.0;
    }
    if x == 0.0 {
        0.0
    } else if x == 0.5 {
        f64::NAN
    } else {
        (PI * x).tan()
    }
}

pub(crate) fn qt_scalar(p: f64, ndf: f64, lower_tail: bool, log_p: bool) -> f64 {
    let eps = 1e-12;
    if p.is_nan() || ndf.is_nan() {
        return p + ndf;
    }
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { f64::INFINITY } else { f64::NEG_INFINITY };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail { f64::NEG_INFINITY } else { f64::INFINITY };
        }
    } else {
        if p < 0.0 || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        if p == 1.0 {
            return if lower_tail { f64::INFINITY } else { f64::NEG_INFINITY };
        }
    }
    if ndf <= 0.0 {
        return f64::NAN;
    }
    if ndf < 1.0 {
        let accu = 1e-13;
        let eps2 = 1e-11;
        let pv = r_dt_qiv(p, lower_tail, log_p);
        if pv > 1.0 - f64::EPSILON {
            return f64::INFINITY;
        }
        let mut pp = (1.0 - f64::EPSILON).min(pv * (1.0 + eps2));
        let mut ux = 1.0;
        while ux < f64::MAX && pt_scalar(ux, ndf, true, false) < pp {
            ux *= 2.0;
        }
        pp = pv * (1.0 - eps2);
        let mut lx = -1.0;
        while lx > -f64::MAX && pt_scalar(lx, ndf, true, false) > pp {
            lx *= 2.0;
        }
        let mut it = 0;
        let mut nx;
        loop {
            nx = 0.5 * (lx + ux);
            if pt_scalar(nx, ndf, true, false) > pv {
                ux = nx;
            } else {
                lx = nx;
            }
            it += 1;
            if !((ux - lx) / nx.abs() > accu && it < 1000) {
                break;
            }
        }
        return 0.5 * (lx + ux);
    }
    if ndf > 1e20 {
        return qnorm5_scalar(p, 0.0, 1.0, lower_tail, log_p);
    }
    let mut pp = if log_p { p.exp() } else { p };
    let neg = ((!lower_tail) || pp < 0.5) && (lower_tail || pp > 0.5);
    let is_neg_lower = lower_tail == neg;
    if neg {
        pp = 2.0
            * (if log_p {
                if lower_tail { pp } else { -p.exp_m1() }
            } else if lower_tail {
                pp
            } else {
                0.5 - p + 0.5
            });
    } else {
        pp = 2.0
            * (if log_p {
                if lower_tail { -p.exp_m1() } else { pp }
            } else if lower_tail {
                0.5 - p + 0.5
            } else {
                p
            });
    }

    let mut q;
    if (ndf - 2.0).abs() < eps {
        if pp > f64::MIN_POSITIVE {
            if 3.0 * pp < f64::EPSILON {
                q = 1.0 / pp.sqrt();
            } else if pp > 0.9 {
                q = (1.0 - pp) * (2.0 / (pp * (2.0 - pp))).sqrt();
            } else {
                q = (2.0 / (pp * (2.0 - pp)) - 2.0).sqrt();
            }
        } else if log_p {
            q = if is_neg_lower {
                (-p / 2.0).exp() / M_SQRT2
            } else {
                1.0 / (-p.exp_m1()).sqrt()
            };
        } else {
            q = f64::INFINITY;
        }
    } else if ndf < 1.0 + eps {
        if pp == 1.0 {
            q = 0.0;
        } else if pp > 0.0 {
            q = 1.0 / tanpi(pp / 2.0);
        } else if log_p {
            q = if is_neg_lower {
                M_1_PI * (-p).exp()
            } else {
                -1.0 / (PI * p.exp_m1())
            };
        } else {
            q = f64::INFINITY;
        }
    } else {
        let mut x = 0.0;
        let mut log_p2 = 0.0;
        let a = 1.0 / (ndf - 0.5);
        let b = 48.0 / (a * a);
        let mut c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
        let d = ((94.5 / (b + c) - 3.0) / b + 1.0) * (a * M_PI_2).sqrt() * ndf;
        let p_ok1 = pp > f64::MIN_POSITIVE || !log_p;
        let mut p_ok = p_ok1;
        let mut y;
        if p_ok1 {
            y = (d * pp).powf(2.0 / ndf);
            p_ok = y >= f64::EPSILON;
        } else {
            y = 0.0;
        }
        if !p_ok {
            log_p2 = if is_neg_lower {
                r_d_log(p, log_p)
            } else {
                r_d_lexp(p, log_p)
            };
            x = (d.ln() + M_LN2 + log_p2) / ndf;
            y = (2.0 * x).exp();
        }
        if (ndf < 2.1 && pp > 0.5) || y > 0.05 + a {
            if p_ok {
                x = qnorm5_scalar(0.5 * pp, 0.0, 1.0, true, false);
            } else {
                x = qnorm5_scalar(log_p2, 0.0, 1.0, lower_tail, true);
            }
            y = x * x;
            if ndf < 5.0 {
                c += 0.3 * (ndf - 4.5) * (x + 0.6);
            }
            c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c;
            y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
            y = (a * y * y).exp_m1();
            q = (ndf * y).sqrt();
        } else if !p_ok && x < -M_LN2 * 53.0 {
            q = ndf.sqrt() * (-x).exp();
        } else {
            y = ((1.0 / (((ndf + 6.0) / (ndf * y) - 0.089 * d - 0.822) * (ndf + 2.0) * 3.0)
                + 0.5 / (ndf + 4.0))
                * y
                - 1.0)
                * (ndf + 1.0)
                / (ndf + 2.0)
                + 1.0 / y;
            q = (ndf * y).sqrt();
        }
        if p_ok1 {
            let mm = (f64::MAX / 2.0).sqrt() - ndf;
            let mm = mm.abs();
            let mut it = 0;
            while it < 10 {
                it += 1;
                y = dt_scalar(q, ndf, false);
                if !(y > 0.0) {
                    break;
                }
                x = (pt_scalar(q, ndf, false, false) - pp / 2.0) / y;
                if !x.is_finite() || !(x.abs() > 1e-14 * q.abs()) {
                    break;
                }
                let ff = if q.abs() < mm {
                    q * (ndf + 1.0) / (2.0 * (q * q + ndf))
                } else {
                    (ndf + 1.0) / (2.0 * (q + ndf / q))
                };
                let del_q = x * (1.0 + x * ff);
                if del_q.is_finite() && (q + del_q).is_finite() {
                    q += del_q;
                } else if x.is_finite() && (q + x).is_finite() {
                    q += x;
                } else {
                    break;
                }
            }
        }
    }
    if neg {
        -q
    } else {
        q
    }
}

pub(crate) fn qf_scalar(p: f64, df1: f64, df2: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || df1.is_nan() || df2.is_nan() {
        return p + df1 + df2;
    }
    if df1 <= 0.0 || df2 <= 0.0 {
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
    if df1 <= df2 && df2 > 4e5 {
        if !df1.is_finite() {
            return 1.0;
        }
        return qgamma_scalar(p, df1 / 2.0, 2.0, lower_tail, log_p) / df1;
    } else if df1 > 4e5 {
        return df2 / qgamma_scalar(p, df2 / 2.0, 2.0, !lower_tail, log_p);
    }
    let res = (1.0 / qbeta_scalar(p, df2 / 2.0, df1 / 2.0, !lower_tail, log_p) - 1.0) * (df2 / df1);
    if !res.is_nan() {
        res
    } else {
        f64::NAN
    }
}

// === PyO3 wrappers ===========================================================
#[pyfunction]
#[pyo3(name = "pt", signature = (x, n, lower_tail=true, log_p=false))]
pub fn pt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (xv, nv) = (x.as_array(), n.as_array());
    let v: Vec<f64> = (0..xv.len())
        .map(|i| pt_scalar(xv[i], nv[i], lower_tail, log_p))
        .collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qt", signature = (p, ndf, lower_tail=true, log_p=false))]
pub fn qt<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    ndf: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (pv, nv) = (p.as_array(), ndf.as_array());
    let v: Vec<f64> = (0..pv.len())
        .map(|i| qt_scalar(pv[i], nv[i], lower_tail, log_p))
        .collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dt", signature = (x, n, give_log=false))]
pub fn dt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (xv, nv) = (x.as_array(), n.as_array());
    let v: Vec<f64> = (0..xv.len()).map(|i| dt_scalar(xv[i], nv[i], give_log)).collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "pf", signature = (x, df1, df2, lower_tail=true, log_p=false))]
pub fn pf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    df1: PyReadonlyArray1<'py, f64>,
    df2: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (xv, d1, d2) = (x.as_array(), df1.as_array(), df2.as_array());
    let v: Vec<f64> = (0..xv.len())
        .map(|i| pf_scalar(xv[i], d1[i], d2[i], lower_tail, log_p))
        .collect();
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qf", signature = (p, df1, df2, lower_tail=true, log_p=false))]
pub fn qf<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    df1: PyReadonlyArray1<'py, f64>,
    df2: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (pv, d1, d2) = (p.as_array(), df1.as_array(), df2.as_array());
    let v: Vec<f64> = (0..pv.len())
        .map(|i| qf_scalar(pv[i], d1[i], d2[i], lower_tail, log_p))
        .collect();
    v.into_pyarray(py)
}
