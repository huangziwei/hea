use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::coeffs::{QN_A, QN_B, QN_C, QN_D, QN_E, QN_F};
use super::consts::{M_2PI, M_LN2, M_LN_SQRT_2PI, M_SQRT2};
use super::util::{ldexp, rfma, round_half_even};

const M_SQRT_32: f64 = 5.656854249492380195206754896838; // sqrt(32)
const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934; // 1/sqrt(2pi)

const PN_A: [f64; 5] = [
    2.2352520354606839287,
    161.02823106855587881,
    1067.6894854603709582,
    18154.981253343561249,
    0.065682337918207449113,
];
const PN_B: [f64; 4] = [
    47.20258190468824187,
    976.09855173777669322,
    10260.932208618978205,
    45507.789335026729956,
];
const PN_C: [f64; 9] = [
    0.39894151208813466764,
    8.8831497943883759412,
    93.506656132177855979,
    597.27027639480026226,
    2494.5375852903726711,
    6848.1904505362823326,
    11602.651437647350124,
    9842.7148383839780218,
    1.0765576773720192317e-8,
];
const PN_D: [f64; 8] = [
    22.266688044328115691,
    235.38790178262499861,
    1519.377599407554805,
    6485.558298266760755,
    18615.571640885098091,
    34900.952721145977266,
    38912.003286093271411,
    19685.429676859990727,
];
const PN_P: [f64; 6] = [
    0.21589853405795699,
    0.1274011611602473639,
    0.022235277870649807,
    0.001421619193227893466,
    2.9112874951168792e-5,
    0.02307344176494017303,
];
const PN_Q: [f64; 5] = [
    1.28426009614491121,
    0.468238212480865118,
    0.0659881378689285515,
    0.00378239633202758244,
    7.29751555083966205e-5,
];

#[inline]
pub(crate) fn dt0(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        if log_p {
            f64::NEG_INFINITY
        } else {
            0.0
        }
    } else if log_p {
        0.0
    } else {
        1.0
    }
}

#[inline]
pub(crate) fn dt1(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        if log_p {
            0.0
        } else {
            1.0
        }
    } else if log_p {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

fn pnorm_both(x: f64, i_tail: i32, log_p: bool) -> (f64, f64) {
    if x.is_nan() {
        return (x, x);
    }
    let eps = f64::EPSILON * 0.5;
    let lower = i_tail != 1;
    let upper = i_tail != 0;
    let y = x.abs();

    if y <= 0.67448975 {
        let (xnum, xden);
        if y > eps {
            let xsq = x * x;
            let mut xn = PN_A[4] * xsq;
            let mut xd = xsq;
            for i in 0..3 {
                xn = (xn + PN_A[i]) * xsq;
                xd = (xd + PN_B[i]) * xsq;
            }
            xnum = xn;
            xden = xd;
        } else {
            xnum = 0.0;
            xden = 0.0;
        }
        let temp = x * (xnum + PN_A[3]) / (xden + PN_B[3]);
        let mut cum = f64::NAN;
        let mut ccum = f64::NAN;
        if lower {
            cum = 0.5 + temp;
        }
        if upper {
            ccum = 0.5 - temp;
        }
        if log_p {
            if lower {
                cum = cum.ln();
            }
            if upper {
                ccum = ccum.ln();
            }
        }
        return (cum, ccum);
    }

    if y <= M_SQRT_32 {
        let mut xn = PN_C[8] * y;
        let mut xd = y;
        for i in 0..7 {
            xn = (xn + PN_C[i]) * y;
            xd = (xd + PN_D[i]) * y;
        }
        let temp = (xn + PN_C[7]) / (xd + PN_D[7]);
        let xsq = (y * 16.0).trunc() / 16.0;
        let del = (y - xsq) * (y + xsq);
        let mut cum;
        let mut ccum = f64::NAN;
        if log_p {
            cum = -xsq * (xsq * 0.5) - del * 0.5 + temp.ln();
            if (lower && x > 0.0) || (upper && x <= 0.0) {
                ccum = (-((-xsq * (xsq * 0.5)).exp() * (-(del * 0.5)).exp() * temp)).ln_1p();
            }
        } else {
            cum = (-xsq * (xsq * 0.5)).exp() * (-(del * 0.5)).exp() * temp;
            ccum = 1.0 - cum;
        }
        if x > 0.0 {
            let t = cum;
            if lower {
                cum = ccum;
            }
            ccum = t;
        }
        return (cum, ccum);
    }

    if (log_p && y < 1e170)
        || (lower && -38.4674 < x && x < 8.2924)
        || (upper && -8.2924 < x && x < 38.4674)
    {
        let xsq0 = 1.0 / (x * x);
        let mut xn = PN_P[5] * xsq0;
        let mut xd = xsq0;
        for i in 0..4 {
            xn = (xn + PN_P[i]) * xsq0;
            xd = (xd + PN_Q[i]) * xsq0;
        }
        let mut temp = xsq0 * (xn + PN_P[4]) / (xd + PN_Q[4]);
        temp = (M_1_SQRT_2PI - temp) / y;
        let xsq = (x * 16.0).trunc() / 16.0;
        let del = (x - xsq) * (x + xsq);
        let mut cum;
        let mut ccum = f64::NAN;
        if log_p {
            cum = -xsq * (xsq * 0.5) - del * 0.5 + temp.ln();
            if (lower && x > 0.0) || (upper && x <= 0.0) {
                ccum = (-((-xsq * (xsq * 0.5)).exp() * (-(del * 0.5)).exp() * temp)).ln_1p();
            }
        } else {
            cum = (-xsq * (xsq * 0.5)).exp() * (-(del * 0.5)).exp() * temp;
            ccum = 1.0 - cum;
        }
        if x > 0.0 {
            let t = cum;
            if lower {
                cum = ccum;
            }
            ccum = t;
        }
        return (cum, ccum);
    }

    let rd0 = if log_p { f64::NEG_INFINITY } else { 0.0 };
    let rd1 = if log_p { 0.0 } else { 1.0 };
    if x > 0.0 {
        (rd1, rd0)
    } else {
        (rd0, rd1)
    }
}

pub fn pnorm5_scalar(x: f64, mu: f64, sigma: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || mu.is_nan() || sigma.is_nan() {
        return x + mu + sigma;
    }
    if x.is_infinite() && mu == x {
        return f64::NAN;
    }
    if sigma <= 0.0 {
        if sigma < 0.0 {
            return f64::NAN;
        }
        return if x < mu {
            dt0(lower_tail, log_p)
        } else {
            dt1(lower_tail, log_p)
        };
    }
    let p = (x - mu) / sigma;
    if p.is_infinite() {
        return if x < mu {
            dt0(lower_tail, log_p)
        } else {
            dt1(lower_tail, log_p)
        };
    }
    let (cum, ccum) = pnorm_both(p, if lower_tail { 0 } else { 1 }, log_p);
    if lower_tail {
        cum
    } else {
        ccum
    }
}

#[pyfunction]
#[pyo3(signature = (x, mu, sigma, lower_tail=true, log_p=false))]
pub fn pnorm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let xs = x.as_slice().unwrap();
    let ms = mu.as_slice().unwrap();
    let ss = sigma.as_slice().unwrap();
    let out = if ms.len() == 1 && ss.len() == 1 {
        let (m0, s0) = (ms[0], ss[0]);
        crate::par::map1(py, xs, |x| pnorm5_scalar(x, m0, s0, lower_tail, log_p))
    } else {
        crate::par::map3(py, xs, ms, ss, |x, m, s| {
            pnorm5_scalar(x, m, s, lower_tail, log_p)
        })
    };
    out.into_pyarray(py)
}

#[inline]
fn qn_horner(r: f64, c: &[f64]) -> f64 {
    let mut v = c[0];
    for &k in &c[1..] {
        v = rfma(v, r, k);
    }
    v
}

pub fn qnorm5_scalar(p: f64, mu: f64, sigma: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || mu.is_nan() || sigma.is_nan() {
        return p + mu + sigma;
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
    if sigma < 0.0 {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return mu;
    }

    let p_ = if log_p {
        if lower_tail {
            p.exp()
        } else {
            -p.exp_m1()
        }
    } else if lower_tail {
        p
    } else {
        0.5 - p + 0.5
    };
    let q = p_ - 0.5;

    if q.abs() <= 0.425 {
        let r = rfma(-q, q, 0.180625);
        let val = q * qn_horner(r, &QN_A) / qn_horner(r, &QN_B);
        return mu + sigma * val;
    }

    let lp;
    if log_p && ((lower_tail && q <= 0.0) || (!lower_tail && q > 0.0)) {
        lp = p;
    } else if q > 0.0 {
        let civ = if log_p {
            if lower_tail {
                -p.exp_m1()
            } else {
                p.exp()
            }
        } else if lower_tail {
            0.5 - p + 0.5
        } else {
            p
        };
        lp = civ.ln();
    } else {
        lp = p_.ln();
    }
    let mut r = (-lp).sqrt();

    let mut val;
    if r <= 5.0 {
        r += -1.6;
        val = qn_horner(r, &QN_C) / qn_horner(r, &QN_D);
    } else if r <= 27.0 {
        r += -5.0;
        val = qn_horner(r, &QN_E) / qn_horner(r, &QN_F);
    } else if r >= 6.4e8 {
        val = r * M_SQRT2;
    } else {
        let s2 = -ldexp(lp, 1);
        let mut x2 = s2 - (M_2PI * s2).ln();
        if r < 36000.0 {
            x2 = s2 - (M_2PI * x2).ln() - 2.0 / (2.0 + x2);
            if r < 840.0 {
                x2 =
                    s2 - (M_2PI * x2).ln() + 2.0 * (-(1.0 - 1.0 / (4.0 + x2)) / (2.0 + x2)).ln_1p();
                if r < 109.0 {
                    x2 = s2 - (M_2PI * x2).ln()
                        + 2.0
                            * (-(1.0 - (1.0 - 5.0 / (6.0 + x2)) / (4.0 + x2)) / (2.0 + x2)).ln_1p();
                    if r < 55.0 {
                        x2 = s2 - (M_2PI * x2).ln()
                            + 2.0
                                * (-(1.0
                                    - (1.0 - (5.0 - 9.0 / (8.0 + x2)) / (6.0 + x2)) / (4.0 + x2))
                                    / (2.0 + x2))
                                    .ln_1p();
                    }
                }
            }
        }
        val = x2.sqrt();
    }
    if q < 0.0 {
        val = -val;
    }
    mu + sigma * val
}

pub fn dnorm5_scalar(x: f64, mu: f64, sigma: f64, give_log: bool) -> f64 {
    if x.is_nan() || mu.is_nan() || sigma.is_nan() {
        return x + mu + sigma;
    }
    let rd0 = if give_log { f64::NEG_INFINITY } else { 0.0 };
    if sigma.is_infinite() {
        return rd0;
    }
    if x.is_infinite() && mu == x {
        return f64::NAN;
    }
    if sigma <= 0.0 {
        if sigma < 0.0 {
            return f64::NAN;
        }
        return if x == mu { f64::INFINITY } else { rd0 };
    }
    let mut x = (x - mu) / sigma;
    if x.is_infinite() {
        return rd0;
    }
    x = x.abs();
    let two_sqrt_dbl_max = 2.0 * f64::MAX.sqrt();
    if x >= two_sqrt_dbl_max {
        return rd0;
    }
    if give_log {
        // dnorm.c:52 `M_LN_SQRT_2PI + 0.5*x*x + log(sigma)`: the outer mul
        // of 0.5*x*x fuses into the first add on arm64.
        return -(rfma(0.5 * x, x, M_LN_SQRT_2PI) + sigma.ln());
    }
    if x < 5.0 {
        return M_1_SQRT_2PI * (-0.5 * x * x).exp() / sigma;
    }
    let dnorm_big = (-2.0 * M_LN2 * (-1021.0 + 1.0 - 53.0)).sqrt();
    if x > dnorm_big {
        return 0.0;
    }
    let x1 = ldexp(round_half_even(ldexp(x, 16)), -16);
    let x2 = x - x1;
    // dnorm.c:85 `(-0.5*x2 - x1)*x2`: fma(-0.5, x2, -x1); outer mul plain.
    M_1_SQRT_2PI / sigma * ((-0.5 * x1 * x1).exp() * (rfma(-0.5, x2, -x1) * x2).exp())
}

#[pyfunction]
#[pyo3(name = "qnorm", signature = (p, mu, sigma, lower_tail=true, log_p=false))]
pub fn qnorm<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let ps = p.as_slice().unwrap();
    let ms = mu.as_slice().unwrap();
    let ss = sigma.as_slice().unwrap();
    let v = if ms.len() == 1 && ss.len() == 1 {
        let (m0, s0) = (ms[0], ss[0]);
        crate::par::map1(py, ps, |p| qnorm5_scalar(p, m0, s0, lower_tail, log_p))
    } else {
        crate::par::map3(py, ps, ms, ss, |p, m, s| {
            qnorm5_scalar(p, m, s, lower_tail, log_p)
        })
    };
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "dnorm", signature = (x, mu, sigma, give_log=false))]
pub fn dnorm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    give_log: bool,
) -> Bound<'py, PyArray1<f64>> {
    let xs = x.as_slice().unwrap();
    let ms = mu.as_slice().unwrap();
    let ss = sigma.as_slice().unwrap();
    let v = if ms.len() == 1 && ss.len() == 1 {
        let (m0, s0) = (ms[0], ss[0]);
        crate::par::map1(py, xs, |x| dnorm5_scalar(x, m0, s0, give_log))
    } else {
        crate::par::map3(py, xs, ms, ss, |x, m, s| dnorm5_scalar(x, m, s, give_log))
    };
    v.into_pyarray(py)
}
