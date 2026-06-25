//! `psigamma` — R's nmath/polygamma.c (Amos, TOMS 610). Line-by-line mirror of
//! the `hea/R/nmath.py` psigamma cluster (which itself mirrors the C): the
//! scalar `dpsifn` specialized to the R case (kode=1, m=1) + the
//! `psigamma(x, deriv)` wrapper, behind a numpy-vectorized PyO3 entry that
//! rayon-parallelizes over the array (the kernel is per-element independent).
#![allow(dead_code)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::util::{r_pow_di, rfma};

const M_PI: f64 = 3.141592653589793238462643383280;
const M_LOG10_2: f64 = 0.301029995663981195213738894724; // R d1mach(5)
const DBL_EPSILON: f64 = 2.220446049250313080847e-16;

// Bernoulli numbers B_2k for the asymptotic expansion (polygamma.c:177-200).
const BVALUES: [f64; 22] = [
    1.00000000000000000e+00,
    -5.00000000000000000e-01,
    1.66666666666666667e-01,
    -3.33333333333333333e-02,
    2.38095238095238095e-02,
    -3.33333333333333333e-02,
    7.57575757575757576e-02,
    -2.53113553113553114e-01,
    1.16666666666666667e+00,
    -7.09215686274509804e+00,
    5.49711779448621554e+01,
    -5.29124242424242424e+02,
    6.19212318840579710e+03,
    -8.65802531135531136e+04,
    1.42551716666666667e+06,
    -2.72982310678160920e+07,
    6.01580873900642368e+08,
    -1.51163157670921569e+10,
    4.29614643061166667e+11,
    -1.37116552050883328e+13,
    4.88332318973593167e+14,
    -1.92965793419400681e+16,
];

/// `(d/dx)^n cot(x)` for n in {0..5} (polygamma.c:149-172); else NaN.
fn d_n_cot(x: f64, n: i64) -> f64 {
    match n {
        0 => x.cos() / x.sin(),
        1 => -1.0 / r_pow_di(x.sin(), 2),
        2 => 2.0 * x.cos() / r_pow_di(x.sin(), 3),
        3 => {
            let sin2 = r_pow_di(x.sin(), 2);
            -2.0 * (3.0 - 2.0 * sin2) / r_pow_di(sin2, 2)
        }
        4 => {
            let co = x.cos();
            8.0 * co * (r_pow_di(co, 2) + 2.0) / r_pow_di(x.sin(), 5)
        }
        5 => {
            let co2 = r_pow_di(x.cos(), 2);
            -8.0 * (2.0 * r_pow_di(co2, 2) + 11.0 * co2 + 2.0) / r_pow_di(x.sin(), 6)
        }
        _ => f64::NAN,
    }
}

/// R's `dpsifn(x, n, kode=1, m=1)` (polygamma.c:175-485): the single scaled
/// derivative `(-1)^(n+1)/gamma(n+1) * psi(n,x)`. Returns NaN on the C
/// `ierr != 0` exits. Only the R case (kode=1, m=1) is ported.
fn dpsifn_m1(mut x: f64, n: i64) -> f64 {
    if n < 0 {
        return f64::NAN; // ierr = 1
    }
    if x <= 0.0 {
        if x == x.round() {
            // non-positive integer
            return if n % 2 != 0 { f64::INFINITY } else { f64::NAN };
        }
        let mut ans = dpsifn_m1(1.0 - x, n); // reflection (A&S 6.4.7)
        if n > 5 {
            return f64::NAN; // ierr = 4
        }
        x = x * M_PI;
        let mut t1 = 1.0;
        let mut t2 = 1.0;
        let mut s = 1.0;
        let mut k: i64 = 0;
        let mut j: i64 = k - n;
        while j < 1 {
            // m == 1  => j < 1
            t1 *= M_PI; // t1 == pi^(k+1)
            if k >= 2 {
                t2 *= k as f64; // t2 == k!
            }
            if j >= 0 {
                // R fuses `ans + (t1/t2)*d_n_cot` to one fmadd on arm64; the
                // reflection cancels badly so a 1-ulp FMA diff amplifies.
                // `rfma` matches R per-arch (= plain `a*b+c` on x86).
                ans = s * rfma(t1 / t2, d_n_cot(x, k), ans);
            }
            k += 1;
            j += 1;
            s = -s;
        }
        return ans;
    }
    // x > 0
    let xln = x.ln();
    let lrg = 1.0 / (2.0 * DBL_EPSILON);
    if n == 0 && x * xln > lrg {
        return -xln;
    }
    if n >= 1 && x > n as f64 * lrg {
        return (-(n as f64) * xln).exp() / n as f64; // x^-n / n
    }
    let mut nx: i64 = 1021; // imin2(-i1mach(15), i1mach(16))
    let r1m5 = M_LOG10_2;
    let r1m4 = DBL_EPSILON * 0.5;
    let wdtol = r1m4.max(0.5e-18);
    let elim = 2.302 * (nx as f64 * r1m5 - 3.0);
    let rln = (r1m5 * 53.0).min(18.06); // i1mach(14) == 53
    let mut fln = rln.max(3.0) - 3.0;
    let yint = 3.50 + 0.40 * fln;
    let slope = 0.21 + fln * (0.0006038 * fln + 0.008677);
    let nn = n;
    let fn_ = n as f64;
    let mut t = (fn_ + 1.0) * xln;
    if t.abs() > elim {
        if t <= 0.0 {
            return f64::NAN; // ierr = 2 (overflow)
        }
        return 0.0; // underflow (m == 1)
    }
    if x < wdtol {
        return r_pow_di(x, -n - 1); // kode == 1: no +xln
    }
    let mut xm = yint + slope * fn_;
    let xmin = (xm as i64 + 1) as f64;
    if n != 0 {
        xm = -2.302 * rln - 0.0_f64.min(xln);
        let arg = 0.0_f64.min(xm / n as f64);
        let eps = arg.exp();
        xm = if arg.abs() < 1.0e-3 { -arg } else { 1.0 - eps };
        fln = x * xm / eps;
        xm = xmin - x;
        if xm > 7.0 && fln < 15.0 {
            // rapidly-converging series
            let nn_s = fln as i64 + 1;
            let np_ = n + 1;
            t = (-(n as f64 + 1.0) * xln).exp();
            let mut s = t;
            let mut den = x;
            for _i in 1..=nn_s {
                den += 1.0;
                s += den.powf(-(np_ as f64));
            }
            return s;
        }
    }
    let mut xdmy = x;
    let mut xdmln = xln;
    let mut xinc = 0.0;
    if x < xmin {
        nx = x as i64;
        xinc = xmin - nx as f64;
        xdmy = x + xinc;
        xdmln = xdmy.ln();
    }
    t = fn_ * xdmln;
    let mut t1 = xdmln + xdmln;
    let t2 = t + xdmln;
    let mut tk = t.abs().max(t1.abs().max(t2.abs()));
    if tk > elim {
        return 0.0; // underflow
    }
    // L10: asymptotic (Bernoulli) expansion in 1/xdmy^2
    let tss = (-t).exp();
    let tt = 0.5 / xdmy;
    t1 = tt;
    let tst = wdtol * tt;
    if nn != 0 {
        t1 = tt + 1.0 / fn_;
    }
    let rxsq = 1.0 / (xdmy * xdmy);
    let ta = 0.5 * rxsq;
    t = (fn_ + 1.0) * ta;
    let mut s = t * BVALUES[2];
    if s.abs() >= tst {
        tk = 2.0;
        for k in 4..23 {
            t = t * ((tk + fn_ + 1.0) / (tk + 1.0)) * ((tk + fn_) / (tk + 2.0)) * rxsq;
            let trm_k = t * BVALUES[k - 1];
            if trm_k.abs() < tst {
                break;
            }
            s += trm_k;
            tk += 2.0;
        }
    }
    s = (s + t1) * tss;
    if xinc != 0.0 {
        // backward recur xdmy -> x
        nx = xinc as i64;
        let np_ = nn + 1;
        if nx > 100 {
            return f64::NAN; // ierr = 3
        }
        if nn == 0 {
            for i in 1..=nx {
                s += 1.0 / (x + (nx - i) as f64); // L20 (avoids cancellation)
            }
            return s - xdmln; // L30, kode == 1
        }
        xm = xinc - 1.0;
        let mut fx = x + xm;
        for _i in 1..=nx {
            s += fx.powf(-(np_ as f64));
            xm -= 1.0;
            fx = x + xm;
        }
    }
    if fn_ == 0.0 {
        return s - xdmln; // L30, kode == 1
    }
    s
}

/// R's `psigamma(x, deriv)` (polygamma.c:499-520): the `deriv`-th derivative of
/// the digamma function; `psigamma(x, 0) == digamma(x)`.
pub fn psigamma_scalar(x: f64, deriv: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let n = deriv.round_ties_even() as i64; // R_forceint
    if n > 100 {
        return f64::NAN;
    }
    let mut ans = dpsifn_m1(x, n);
    ans = -ans; // (-1)^(0+1) gamma(1) A
    for k in 1..=n {
        ans = ans * (-(k as f64)); // (-1)^(k+1) gamma(k+1) A
    }
    ans
}

/// `psigamma(x, deriv)` over a 1-D array; `deriv` is the scalar polygamma order.
#[pyfunction]
pub fn psigamma<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    deriv: f64,
) -> Bound<'py, PyArray1<f64>> {
    let out = crate::par::map1(py, x.as_slice().unwrap(), |xi| psigamma_scalar(xi, deriv));
    out.into_pyarray(py)
}
