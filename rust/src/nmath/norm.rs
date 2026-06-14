//! `pnorm` — normal CDF (R's `nmath/pnorm.c`, Cody 1993).
//!
//! Line-by-line mirror of `hea/R/nmath.py`'s `_pnorm_both` / `pnorm5`. The
//! float-op order is preserved EXACTLY (it is what makes the Python 0-ulp to R):
//! do not reassociate, do not introduce FMA. `ldexp(v, k)` for power-of-two `k`
//! is written as the exact multiply/divide (`v * 16.0`, `v * 0.5`, `/ 16.0`),
//! which is bit-identical to `ldexp` for the (non-subnormal) ranges here.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// --- constants (Rmath.h) -----------------------------------------------------
const M_SQRT_32: f64 = 5.656854249492380195206754896838; // sqrt(32)
const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934; // 1/sqrt(2pi)

// --- Cody 1993 rational-approximation coefficients (mirror nmath.py) ---------
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

// R_DT_0 = lower_tail ? R_D__0 : R_D__1
#[inline]
fn dt0(lower_tail: bool, log_p: bool) -> f64 {
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

// R_DT_1 = lower_tail ? R_D__1 : R_D__0
#[inline]
fn dt1(lower_tail: bool, log_p: bool) -> f64 {
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

/// R's `pnorm_both(x, &cum, &ccum, i_tail, log_p)`. `i_tail` in {0,1} =
/// {lower, upper}. Returns `(cum, ccum)`; the non-requested entry may be NaN.
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
        // qnorm(3/4) < |x| <= sqrt(32) ~ 5.657
        let mut xn = PN_C[8] * y;
        let mut xd = y;
        for i in 0..7 {
            xn = (xn + PN_C[i]) * y;
            xd = (xd + PN_D[i]) * y;
        }
        let temp = (xn + PN_C[7]) / (xd + PN_D[7]);
        // do_del(y)
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
            // swap_tail
            let t = cum;
            if lower {
                cum = ccum;
            }
            ccum = t;
        }
        return (cum, ccum);
    }

    // |x| > sqrt(32)
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
        // do_del(x)
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
            // swap_tail
            let t = cum;
            if lower {
                cum = ccum;
            }
            ccum = t;
        }
        return (cum, ccum);
    }

    // large |x|: probs are 0 or 1
    let rd0 = if log_p { f64::NEG_INFINITY } else { 0.0 };
    let rd1 = if log_p { 0.0 } else { 1.0 };
    if x > 0.0 {
        (rd1, rd0)
    } else {
        (rd0, rd1)
    }
}

/// R's `pnorm5(x, mu, sigma, lower_tail, log_p)`, bit-exact.
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
        // sigma == 0
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

/// Vectorized `pnorm` over a 1-D f64 array. Mirrors `nmath._vec` semantics for
/// the scalar-params case (R recycling of array-vs-scalar args is handled in
/// Python before the native call).
#[pyfunction]
#[pyo3(signature = (x, mu=0.0, sigma=1.0, lower_tail=true, log_p=false))]
pub fn pnorm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    mu: f64,
    sigma: f64,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let xv = x.as_array();
    let out: Vec<f64> = xv
        .iter()
        .map(|&xi| pnorm5_scalar(xi, mu, sigma, lower_tail, log_p))
        .collect();
    out.into_pyarray(py)
}
