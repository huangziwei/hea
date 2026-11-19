//! mgcv's Tweedie density series: mechanical ports of the C `tweedious`
//! (src/misc.c:170, one `(th, rho)` for every observation, the `j` terms
//! buffered) and `tweedious2` (misc.c:513, `(th, rho)` per observation,
//! `lgamma(j + 1)` carried along the sweep), returning `log W` and its
//! derivatives in the working parameters `(rho, th)`.
//!
//! Each row is summed up from its series maximum and then down, to `eps`
//! relative to the largest term. The arithmetic is the compiled C's: clang
//! fuses `a*b + c` within an expression (and the left product of
//! `a*b + c*d`), so those are `rfma` here and nothing else is; `lgamma` is
//! libm's (as the C calls it, not R's `lgammafn`), digamma and trigamma are
//! the nmath ports.

use crate::nmath::psigamma::psigamma_scalar;
use crate::nmath::util::rfma;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

extern "C" {
    fn lgamma_r(x: f64, sign: *mut i32) -> f64;
}

/// libm `lgamma`, which mgcv's `tweedious`/`tweedious2` call (not R's
/// `lgammafn`); the reentrant form, so rows can run on several threads.
#[inline]
fn libm_lgamma(x: f64) -> f64 {
    let mut sign = 0i32;
    unsafe { lgamma_r(x, &mut sign) }
}

/// `p` and its first two derivatives in the working parameter `th`, as the
/// C computes them (misc.c:209-222 / 547-557), fused as compiled.
#[inline]
fn tw_p_th(th: f64, a: f64, b: f64) -> (f64, f64, f64) {
    if th > 0.0 {
        let exp_th = (-th).exp();
        let x = 1.0 + exp_th;
        let p = rfma(a, exp_th, b) / x;
        let x1 = x * x;
        let bma = exp_th * (b - a);
        let dpth1 = bma / x1;
        let dpth2 = rfma(a - b, exp_th, exp_th * bma) / (x * x1);
        (p, dpth1, dpth2)
    } else {
        let exp_th = th.exp();
        let x = exp_th + 1.0;
        let p = rfma(b, exp_th, a) / x;
        let x1 = x * x;
        let bma = exp_th * (b - a);
        let dpth1 = bma / x1;
        let dpth2 = rfma((a - b) * exp_th, exp_th, bma) / (x * x1);
        (p, dpth1, dpth2)
    }
}

/// The parts of `log W_j` and its derivatives that are a multiple of `j`.
struct TwBases {
    onep: f64,
    onep2: f64,
    alpha: f64,
    w_base: f64,
    wp_base: f64,
    wp2_base: f64,
}

#[inline]
fn tw_bases(p: f64, rho: f64) -> TwBases {
    let onep = 1.0 - p;
    let onep2 = onep * onep;
    let twop = 2.0 - p;
    let alpha = twop / onep;
    let w_base = rfma(alpha, (-onep).ln(), rho / onep) - twop.ln();
    let lr = (-onep).ln() + rho;
    let wp_base = lr / onep2 - alpha / onep + 1.0 / twop;
    let wp2_base =
        (lr + lr) / (onep2 * onep) - rfma(alpha, 3.0, -2.0) / onep2 + 1.0 / (twop * twop);
    TwBases {
        onep,
        onep2,
        alpha,
        w_base,
        wp_base,
        wp2_base,
    }
}

/// `tweedious`'s buffered terms for index `j` (misc.c:285-293):
/// `(wb, wb1, wp1, wp2, wpp)`.
#[inline]
fn tw_jterms(j: i64, bs: &TwBases) -> (f64, f64, f64, f64, f64) {
    let jf = j as f64;
    let nja = ((-j) as f64) * bs.alpha;
    let wb = rfma(jf, bs.w_base, -libm_lgamma(jf + 1.0)) - libm_lgamma(nja);
    let wb1 = ((-j) as f64) / bs.onep;
    let xx = jf / bs.onep2;
    let x = xx * psigamma_scalar(nja, 0.0);
    let wp1 = rfma(jf, bs.wp_base, x);
    let xx2 = psigamma_scalar(nja, 1.0) * xx * xx;
    let wp2 = rfma(jf, bs.wp2_base, (x + x) / bs.onep) - xx2;
    let wpp = jf / bs.onep2;
    (wb, wb1, wp1, wp2, wpp)
}

/// The six sums of a sweep (`wi`, `w1i`, `w2i`, `wdlogwdp`, `wdW2d2W`,
/// `dWpp`).
#[derive(Default)]
struct TwSums {
    wi: f64,
    w1i: f64,
    w2i: f64,
    wdlogwdp: f64,
    wd_w2d2w: f64,
    dwpp: f64,
}

impl TwSums {
    /// One term of the sum (misc.c:331-352), fused as compiled: the `j`
    /// products in `wp1j`/`wp2j`, the left product of the θ-chain, and the
    /// `w1i`/`w2i` updates are fused; the other three sums add rounded
    /// products.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        j: i64,
        wj: f64,
        wmax: f64,
        w1j: f64,
        wp1: f64,
        wp2: f64,
        wpp: f64,
        logy1p2: f64,
        logy1p3: f64,
        dpth1: f64,
        dpth2: f64,
        onep: f64,
    ) {
        let jf = j as f64;
        let wp1j0 = rfma(-jf, logy1p2, wp1);
        let wp2j0 = rfma(-((2 * j) as f64), logy1p3, wp2);
        let wp2j = rfma(wp1j0, dpth2, wp2j0 * dpth1 * dpth1);
        let wp1j = wp1j0 * dpth1;
        let wppj = wpp * dpth1;
        let ws = (wj - wmax).exp();
        self.wi += ws;
        self.w1i = rfma(ws, w1j, self.w1i);
        self.w2i = rfma(ws * w1j, w1j, self.w2i);
        self.wdlogwdp += ws * wp1j;
        self.wd_w2d2w += ws * rfma(wp1j, wp1j, wp2j);
        self.dwpp += ws * (wp1j * jf / onep + wppj);
    }

    /// The final transformations (misc.c:498-503), fused as compiled.
    #[inline]
    fn finish(&self, wmax: f64, row: &mut [f64]) {
        let a1 = self.w1i / self.wi;
        let a2 = self.wdlogwdp / self.wi;
        row[0] = wmax + self.wi.ln();
        row[1] = -self.w1i / self.wi;
        row[2] = rfma(-a1, a1, self.w2i / self.wi);
        row[3] = a2;
        row[4] = rfma(-a2, a2, self.wd_w2d2w / self.wi);
        row[5] = rfma(a1, a2, self.dwpp / self.wi);
    }
}

/// mgcv's buffer limit on the series length (`jal_lim`, misc.c).
const JAL_LIM: i64 = 50_000_000;

/// mgcv C `tweedious` (misc.c:170-510): the Tweedie series `log W` and its
/// derivatives in the working parameters `(rho, th)` at one `(th, rho)`
/// for every `y > 0`, columns `[w, w1, w2, w1p, w2p, w2pp]`. The terms are
/// those mgcv buffers over `j` (they depend on `j` alone), each row summed
/// up from its series maximum and then down, to `eps` relative to the
/// largest term. The second value is mgcv's `eps` on return: `-1` when a
/// series ran past the buffer limit, `-2` when an index overflowed (rows
/// from that one on are not computed).
#[pyfunction]
#[pyo3(name = "tweedious_work")]
fn tweedious_work<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    th: f64,
    rho: f64,
    a: f64,
    b: f64,
    eps: f64,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let y = y.as_slice().unwrap();
    let n = y.len();
    let mut out = vec![0.0f64; n * 6];
    if n == 0 {
        return (
            Array2::from_shape_vec((0, 6), out)
                .unwrap()
                .into_pyarray(py),
            eps,
        );
    }
    let phi = rho.exp();
    let (p, dpth1, dpth2) = tw_p_th(th, a, b);
    let log_eps = eps.ln();
    let bs = tw_bases(p, rho);
    let twop = 2.0 - p;
    let mut ymax = y[0];
    let mut ymin = y[0];
    for &yi in &y[1..] {
        if yi > ymax {
            ymax = yi;
        } else if yi < ymin {
            ymin = yi;
        }
    }
    let x_lo = ymin.powf(twop) / (phi * twop);
    let j_lo = (x_lo.floor() as i64).max(1);
    let x_hi = ymax.powf(twop) / (phi * twop);
    let j_hi = (x_hi.ceil() as i64).max(j_lo);
    let j0 = (j_lo - 1000).max(1);
    // The terms depend on `j` alone, so one table serves every row. mgcv
    // fills its buffer over `[j_lo, j_hi]` and extends it as sweeps run
    // past either end; here the table spans the sweeps of the rows with the
    // smallest and largest `y`, which reach furthest, and any other index
    // is formed on demand to the same value.
    let sweep_ends = |yi: f64| -> (i64, i64) {
        let ly = yi.ln();
        let alogy = bs.alpha * ly;
        let x = yi.powf(twop) / (phi * twop);
        let mut jm = x.floor() as i64;
        if x - (jm as f64) > 0.5 || jm < 1 {
            jm += 1;
        }
        let jalogy = (jm as f64) * alogy;
        let jm = jm.clamp(j_lo, j_hi);
        let wb_of = |j: i64| -> f64 {
            let jf = j as f64;
            rfma(jf, bs.w_base, -libm_lgamma(jf + 1.0)) - libm_lgamma(((-j) as f64) * bs.alpha)
        };
        let wmin = (wb_of(jm) - jalogy) + log_eps;
        let mut up = jm;
        while wb_of(up) - (up as f64) * alogy >= wmin && up - j0 < JAL_LIM {
            up += 1;
        }
        let mut down = jm - 1;
        while down > 1 && wb_of(down) - (down as f64) * alogy >= wmin {
            down -= 1;
        }
        (down.max(1), up)
    };
    let (t_lo, _) = sweep_ends(ymin);
    let (_, t_hi) = sweep_ends(ymax);
    let t_lo = t_lo.min(j_lo);
    let t_hi = t_hi.max(j_hi);
    let table: Vec<(f64, f64, f64, f64, f64)> = if n >= crate::par::PAR_THRESHOLD {
        py.allow_threads(|| {
            (t_lo..=t_hi)
                .into_par_iter()
                .map(|j| tw_jterms(j, &bs))
                .collect()
        })
    } else {
        (t_lo..=t_hi).map(|j| tw_jterms(j, &bs)).collect()
    };
    let term = |j: i64| -> (f64, f64, f64, f64, f64) {
        if j >= t_lo && j <= t_hi {
            table[(j - t_lo) as usize]
        } else {
            tw_jterms(j, &bs)
        }
    };

    // Each row's series location `x`; mgcv stops at the first row whose
    // index overflows and leaves the later rows.
    let xs: Vec<f64> = if n >= crate::par::PAR_THRESHOLD {
        py.allow_threads(|| {
            y.par_iter()
                .map(|&yi| yi.powf(twop) / (phi * twop))
                .collect()
        })
    } else {
        y.iter().map(|&yi| yi.powf(twop) / (phi * twop)).collect()
    };
    let mut n_ok = n;
    for (i, &x) in xs.iter().enumerate() {
        let mut jm = x.floor() as i64;
        if x - (jm as f64) > 0.5 || jm < 1 {
            jm += 1;
        }
        if ((jm as f64) - x).abs() > 1.0 {
            n_ok = i;
            break;
        }
    }
    let failed = n_ok < n;

    let fill = |i: usize, row: &mut [f64]| -> bool {
        let yi = y[i];
        let ly = yi.ln();
        let alogy = bs.alpha * ly;
        let logy1p2 = ly / bs.onep2;
        let logy1p3 = logy1p2 / bs.onep;
        let x = xs[i];
        let mut jm = x.floor() as i64;
        if x - (jm as f64) > 0.5 || jm < 1 {
            jm += 1;
        }
        let jalogy = (jm as f64) * alogy;
        let jm = jm.clamp(j_lo, j_hi);
        let wmax = term(jm).0 - jalogy;
        let wmin = wmax + log_eps;
        let mut sums = TwSums::default();
        let mut run_out = false;
        let mut j = jm;
        loop {
            let (wb, wb1, wp1, wp2, wpp) = term(j);
            let wj = wb - (j as f64) * alogy;
            sums.add(
                j, wj, wmax, wb1, wp1, wp2, wpp, logy1p2, logy1p3, dpth1, dpth2, bs.onep,
            );
            if wj < wmin {
                break;
            }
            j += 1;
            if j - j0 >= JAL_LIM {
                run_out = true;
                break;
            }
        }
        let mut j = jm - 1;
        while j >= 1 {
            let (wb, wb1, wp1, wp2, wpp) = term(j);
            let wj = wb - (j as f64) * alogy;
            sums.add(
                j, wj, wmax, wb1, wp1, wp2, wpp, logy1p2, logy1p3, dpth1, dpth2, bs.onep,
            );
            if wj < wmin {
                break;
            }
            j -= 1;
        }
        sums.finish(wmax, row);
        run_out
    };

    let run_out = if n_ok >= crate::par::PAR_THRESHOLD {
        py.allow_threads(|| {
            out[..n_ok * 6]
                .par_chunks_mut(6)
                .enumerate()
                .map(|(i, row)| fill(i, row))
                .reduce(|| false, |u, v| u || v)
        })
    } else {
        let mut any = false;
        for (i, row) in out[..n_ok * 6].chunks_mut(6).enumerate() {
            any |= fill(i, row);
        }
        any
    };
    let eps_out = if failed {
        -2.0
    } else if run_out {
        -1.0
    } else {
        eps
    };
    (
        Array2::from_shape_vec((n, 6), out)
            .unwrap()
            .into_pyarray(py),
        eps_out,
    )
}

/// mgcv C `tweedious2` (misc.c:513-661): as `tweedious` with a `(th, rho)`
/// per row and no buffer. Each row sums one sweep up from its series
/// maximum and one down, `lgamma(j + 1)` carried along by `+= log(j)` up
/// and `+= -log(j + 1)` down from a fresh `lgamma` at each sweep's start;
/// `wmax` is formed in one expression, its last product fused.
#[pyfunction]
#[pyo3(name = "tweedious_work_pv")]
#[allow(clippy::too_many_arguments)]
fn tweedious_work_pv<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    th: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    a: f64,
    b: f64,
    eps: f64,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let y = y.as_slice().unwrap();
    let th = th.as_slice().unwrap();
    let rho = rho.as_slice().unwrap();
    let n = y.len();
    let log_eps = eps.ln();
    let mut out = vec![0.0f64; n * 6];

    let mut n_ok = n;
    for i in 0..n {
        let phi = rho[i].exp();
        let (p, _, _) = tw_p_th(th[i], a, b);
        let twop = 2.0 - p;
        let x = y[i].powf(twop) / (phi * twop);
        let mut jm = x.floor() as i64;
        if x - (jm as f64) > 0.5 || jm < 1 {
            jm += 1;
        }
        if ((jm as f64) - x).abs() > 1.0 {
            n_ok = i;
            break;
        }
    }
    let failed = n_ok < n;

    let fill = |i: usize, row: &mut [f64]| -> bool {
        let phi = rho[i].exp();
        let (p, dpth1, dpth2) = tw_p_th(th[i], a, b);
        let twop = 2.0 - p;
        let x = y[i].powf(twop) / (phi * twop);
        let mut jm = x.floor() as i64;
        if x - (jm as f64) > 0.5 || jm < 1 {
            jm += 1;
        }
        let bs = tw_bases(p, rho[i]);
        let ly = y[i].ln();
        let logy1p2 = ly / bs.onep2;
        let logy1p3 = logy1p2 / bs.onep;
        let alogy = ly * bs.alpha;
        let jmf = jm as f64;
        let nja_m = ((-jm) as f64) * bs.alpha;
        let wmax = rfma(
            -jmf,
            alogy,
            rfma(jmf, bs.w_base, -libm_lgamma(jmf + 1.0)) - libm_lgamma(nja_m),
        );
        let wmin = wmax + log_eps;
        let mut sums = TwSums::default();
        let mut j = jm;
        let mut incr: i64 = 1;
        let mut lgammaj1 = libm_lgamma(jmf + 1.0);
        let mut k: i64 = 0;
        let mut too_long = false;
        loop {
            let jf = j as f64;
            let nja = ((-j) as f64) * bs.alpha;
            let wbj = rfma(jf, bs.w_base, -lgammaj1) - libm_lgamma(nja);
            let wb1j = ((-j) as f64) / bs.onep;
            let xx = jf / bs.onep2;
            let xd = xx * psigamma_scalar(nja, 0.0);
            let wp1jb = rfma(jf, bs.wp_base, xd);
            let xx2 = psigamma_scalar(nja, 1.0) * xx * xx;
            let wp2jb = rfma(jf, bs.wp2_base, (xd + xd) / bs.onep) - xx2;
            let wj = wbj - jf * alogy;
            sums.add(
                j, wj, wmax, wb1j, wp1jb, wp2jb, xx, logy1p2, logy1p3, dpth1, dpth2, bs.onep,
            );
            j += incr;
            let mut done = false;
            if incr > 0 {
                lgammaj1 += (j as f64).ln();
                if wj < wmin {
                    j = jm - 1;
                    incr = -1;
                    if j == 0 {
                        done = true;
                    }
                    lgammaj1 = libm_lgamma(j as f64 + 1.0);
                }
            } else {
                lgammaj1 += -((j + 1) as f64).ln();
                if wj < wmin || j < 1 {
                    done = true;
                }
            }
            k += 1;
            if k >= JAL_LIM {
                too_long = true;
                done = true;
            }
            if done {
                break;
            }
        }
        sums.finish(wmax, row);
        too_long
    };

    let too_long = if n_ok >= crate::par::PAR_THRESHOLD {
        py.allow_threads(|| {
            out[..n_ok * 6]
                .par_chunks_mut(6)
                .enumerate()
                .map(|(i, row)| fill(i, row))
                .reduce(|| false, |u, v| u || v)
        })
    } else {
        let mut any = false;
        for (i, row) in out[..n_ok * 6].chunks_mut(6).enumerate() {
            any |= fill(i, row);
        }
        any
    };
    let eps_out = if failed {
        -2.0
    } else if too_long {
        -1.0
    } else {
        eps
    };
    (
        Array2::from_shape_vec((n, 6), out)
            .unwrap()
            .into_pyarray(py),
        eps_out,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tweedious_work, m)?)?;
    m.add_function(wrap_pyfunction!(tweedious_work_pv, m)?)?;
    Ok(())
}

use pyo3::wrap_pyfunction;
