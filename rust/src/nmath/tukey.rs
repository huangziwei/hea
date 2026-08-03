//! `ptukey` / `qtukey` — CDF/quantile of the studentized range (nmath/ptukey.c,
//! qtukey.c). Mirror of the `hea/R/nmath.py` tukey cluster. Copenhaver-Holland
//! Gauss-Legendre quadrature (wprob) + AS 70 start + secant (qtukey).
//!
//! Only `wprob`'s inner accumulators (einsum/elsum/blb/bub) are `LDOUBLE` in R;
//! `ptukey`'s outer sum is plain `double`. Rust has no 80-bit float, so wprob's
//! accumulators are `f64` here — the residual vs R is confined to wprob.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::consts::M_LN2;
use super::lgamma::lgammafn;
use super::norm::{dt0, dt1, pnorm5_scalar};
use super::util::rfma;

const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934; // 1/sqrt(2pi)

// wprob: 12-point Gauss-Legendre nodes/weights (upper half)
const XLEG: [f64; 6] = [
    0.981560634246719250690549090149,
    0.904117256370474856678465866119,
    0.769902674194304687036893833213,
    0.587317954286617447296702418941,
    0.367831498998180193752691536644,
    0.125233408511468915472441369464,
];
const ALEG: [f64; 6] = [
    0.047175336386511827194615961485,
    0.106939325995318430960254718194,
    0.160078328543346226334652529543,
    0.203167426723065921749064455810,
    0.233492536538354808760849898925,
    0.249147045813402785000562436043,
];
// ptukey: 16-point Gauss-Legendre nodes/weights (upper half)
const XLEGQ: [f64; 8] = [
    0.989400934991649932596154173450,
    0.944575023073232576077988415535,
    0.865631202387831743880467897712,
    0.755404408355003033895101194847,
    0.617876244402643748446671764049,
    0.458016777657227386342419442984,
    0.281603550779258913230460501460,
    0.0950125098376374401853193354250,
];
const ALEGQ: [f64; 8] = [
    0.0271524594117540948517805724560,
    0.0622535239386478928628438369944,
    0.0951585116824927848099251076022,
    0.124628971255533872052476282192,
    0.149595988816576732081501730547,
    0.169156519395002538189312079030,
    0.182603415044923588866763667969,
    0.189450610455068496285396723208,
];

#[inline]
fn r_dt_val(x: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        if log_p {
            x.ln()
        } else {
            x
        }
    } else if log_p {
        (-x).ln_1p()
    } else {
        0.5 - x + 0.5
    }
}

fn wprob(w: f64, rr: f64, cc: f64) -> f64 {
    let nleg = 12;
    let ihalf = 6;
    let c1 = -30.0;
    let c2 = -50.0;
    let c3 = 60.0;
    let bb = 8.0;
    let wlar = 3.0;
    let wincr1 = 2.0;
    let wincr2 = 3.0;

    let qsqz = w * 0.5;
    if qsqz >= bb {
        return 1.0;
    }
    let mut pr_w = 2.0 * pnorm5_scalar(qsqz, 0.0, 1.0, true, false) - 1.0;
    if pr_w >= (c2 / cc).exp() {
        pr_w = pr_w.powf(cc);
    } else {
        pr_w = 0.0;
    }
    let wincr = if w > wlar { wincr1 } else { wincr2 };
    let mut blb = qsqz;
    let binc = (bb - qsqz) / wincr;
    let mut bub = blb + binc;
    let mut einsum = 0.0f64;
    let cc1 = cc - 1.0;
    for _ in 0..(wincr as i32) {
        let mut elsum = 0.0f64;
        let a = 0.5 * (bub + blb);
        let b = 0.5 * (bub - blb);
        for jj in 1..=nleg {
            let j;
            let xx;
            if ihalf < jj {
                j = (nleg - jj) + 1;
                xx = XLEG[j - 1];
            } else {
                j = jj;
                xx = -XLEG[j - 1];
            }
            let c = b * xx;
            let ac = a + c;
            let qexpo = ac * ac;
            if qexpo > c3 {
                break;
            }
            let pplus = 2.0 * pnorm5_scalar(ac, 0.0, 1.0, true, false);
            let pminus = 2.0 * pnorm5_scalar(ac, w, 1.0, true, false);
            let mut rinsum = (pplus * 0.5) - (pminus * 0.5);
            if rinsum >= (c1 / cc1).exp() {
                rinsum = (ALEG[j - 1] * (-(0.5 * qexpo)).exp()) * rinsum.powf(cc1);
                elsum += rinsum;
            }
        }
        elsum *= ((2.0 * b) * cc) * M_1_SQRT_2PI;
        einsum += elsum;
        blb = bub;
        bub += binc;
    }
    pr_w += einsum;
    if pr_w <= (c1 / rr).exp() {
        return 0.0;
    }
    pr_w = pr_w.powf(rr);
    if pr_w >= 1.0 {
        return 1.0;
    }
    pr_w
}

pub(crate) fn ptukey_scalar(
    q: f64,
    rr: f64,
    cc: f64,
    df: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    let nlegq = 16;
    let ihalfq = 8;
    let eps1 = -30.0;
    let eps2 = 1.0e-14;
    let dhaf = 100.0;
    let dquar = 800.0;
    let deigh = 5000.0;
    let dlarg = 25000.0;
    let ulen1 = 1.0;
    let ulen2 = 0.5;
    let ulen3 = 0.25;
    let ulen4 = 0.125;

    if q.is_nan() || rr.is_nan() || cc.is_nan() || df.is_nan() {
        return q + rr + cc + df;
    }
    if q <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    if df < 2.0 || rr < 1.0 || cc < 2.0 {
        return f64::NAN;
    }
    if !q.is_finite() {
        return dt1(lower_tail, log_p);
    }
    if df > dlarg {
        return r_dt_val(wprob(q, rr, cc), lower_tail, log_p);
    }

    // clang contracts the *leading* multiply of each `a*b ± c` into one fmadd
    // on arm64, so `rfma` (= plain `a*b+c` on x86) is what keeps this whole
    // quadrature 0-ulp to R on both arches.
    let f2 = df * 0.5;
    let mut f2lf = rfma(f2, df.ln(), -(df * M_LN2)) - lgammafn(f2);
    let f21 = f2 - 1.0;
    let ff4 = df * 0.25;
    let ulen: f64 = if df <= dhaf {
        ulen1
    } else if df <= dquar {
        ulen2
    } else if df <= deigh {
        ulen3
    } else {
        ulen4
    };
    f2lf += ulen.ln();
    let mut ans = 0.0f64;

    for i in 1..=50 {
        let mut otsum = 0.0f64;
        let twa1 = (2 * i - 1) as f64 * ulen;
        for jj in 1..=nlegq {
            let j;
            let t1;
            if ihalfq < jj {
                j = jj - ihalfq - 1;
                let xu1 = rfma(XLEGQ[j], ulen, twa1);
                t1 = rfma(-xu1, ff4, rfma(f21, xu1.ln(), f2lf));
            } else {
                j = jj - 1;
                t1 = rfma(
                    rfma(XLEGQ[j], ulen, -twa1),
                    ff4,
                    rfma(f21, rfma(-XLEGQ[j], ulen, twa1).ln(), f2lf),
                );
            }
            if t1 >= eps1 {
                let qsqz = if ihalfq < jj {
                    q * (rfma(XLEGQ[j], ulen, twa1) * 0.5).sqrt()
                } else {
                    // `(-(XLEGQ[j]*ulen)) + twa1`: the negation makes the LHS an
                    // fneg, not an fmul, so clang leaves this one uncontracted.
                    q * (((-(XLEGQ[j] * ulen)) + twa1) * 0.5).sqrt()
                };
                let wprb = wprob(qsqz, rr, cc);
                let rotsum = (wprb * ALEGQ[j]) * t1.exp();
                otsum += rotsum;
            }
        }
        if i as f64 * ulen >= 1.0 && otsum <= eps2 {
            break;
        }
        ans += otsum;
    }
    if ans > 1.0 {
        ans = 1.0;
    }
    r_dt_val(ans, lower_tail, log_p)
}

fn qtukey_qinv(p: f64, c: f64, v: f64) -> f64 {
    let p0 = 0.322232421088;
    let q0 = 0.0993484626060;
    let p1 = -1.0;
    let q1 = 0.588581570495;
    let p2 = -0.342242088547;
    let q2 = 0.531103462366;
    let p3 = -0.204231210125;
    let q3 = 0.103537752850;
    let p4 = -0.0000453642210148;
    let q4 = 0.0038560700634;
    let c1 = 0.8832;
    let c2 = 0.2368;
    let c3 = 1.214;
    let c4 = 1.208;
    let c5 = 1.4142;
    let vmax = 120.0;

    // Every `a*b + c` below is one fmadd in R's arm64 build (see ptukey above).
    let ps = 0.5 - 0.5 * p;
    let yi = (1.0 / (ps * ps)).ln().sqrt();
    let mut t = yi
        + rfma(rfma(rfma(rfma(yi, p4, p3), yi, p2), yi, p1), yi, p0)
            / rfma(rfma(rfma(rfma(yi, q4, q3), yi, q2), yi, q1), yi, q0);
    if v < vmax {
        t += rfma(t * t, t, t) / v / 4.0;
    }
    let mut q = rfma(-c2, t, c1);
    if v < vmax {
        q += -c3 / v + c4 * t / v;
    }
    t * rfma(q, (c - 1.0).ln(), c5)
}

pub(crate) fn qtukey_scalar(
    p: f64,
    rr: f64,
    cc: f64,
    df: f64,
    lower_tail: bool,
    log_p: bool,
) -> f64 {
    let eps = 0.0001;
    let maxiter = 50;

    if p.is_nan() || rr.is_nan() || cc.is_nan() || df.is_nan() {
        return p + rr + cc + df;
    }
    if df < 2.0 || rr < 1.0 || cc < 2.0 {
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
    let p = if log_p {
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

    let x0 = qtukey_qinv(p, cc, df);
    let mut valx0 = ptukey_scalar(x0, rr, cc, df, true, false) - p;
    let mut x1 = if valx0 > 0.0 {
        (x0 - 1.0).max(0.0)
    } else {
        x0 + 1.0
    };
    let mut valx1 = ptukey_scalar(x1, rr, cc, df, true, false) - p;
    let mut x0 = x0;

    let mut ans = 0.0;
    for _ in 1..maxiter {
        ans = x1 - ((valx1 * (x1 - x0)) / (valx1 - valx0));
        valx0 = valx1;
        x0 = x1;
        if ans < 0.0 {
            ans = 0.0;
            valx1 = -p;
        }
        valx1 = ptukey_scalar(ans, rr, cc, df, true, false) - p;
        x1 = ans;
        if (x1 - x0).abs() < eps {
            return ans;
        }
    }
    ans
}

// === PyO3 wrappers ===========================================================
#[pyfunction]
#[pyo3(name = "ptukey", signature = (q, rr, cc, df, lower_tail=true, log_p=false))]
pub fn ptukey<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    rr: PyReadonlyArray1<'py, f64>,
    cc: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        q.as_slice().unwrap(),
        rr.as_slice().unwrap(),
        cc.as_slice().unwrap(),
        df.as_slice().unwrap(),
        |q, r, c, d| ptukey_scalar(q, r, c, d, lower_tail, log_p),
    );
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "qtukey", signature = (p, rr, cc, df, lower_tail=true, log_p=false))]
pub fn qtukey<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    rr: PyReadonlyArray1<'py, f64>,
    cc: PyReadonlyArray1<'py, f64>,
    df: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let v = crate::par::map4(
        py,
        p.as_slice().unwrap(),
        rr.as_slice().unwrap(),
        cc.as_slice().unwrap(),
        df.as_slice().unwrap(),
        |p, r, c, d| qtukey_scalar(p, r, c, d, lower_tail, log_p),
    );
    v.into_pyarray(py)
}
