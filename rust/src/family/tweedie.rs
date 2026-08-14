//! Tweedie density series — mechanical port of mgcv's `tweedious` (src/misc.c:170),
//! the per-row windowed summation of the Dunn & Smyth (2005) series for
//! `log a(y, φ, p) = log Σ_{j≥1} W_j` and its working-parameter derivatives.
//!
//! mgcv builds the j-indexed terms once into a shared buffer (the special
//! functions `lgamma(j+1)`, `lgamma(-jα)`, `digamma(-jα)`, `trigamma(-jα)`
//! depend only on j for scalar p), then for each observation sweeps OUTWARD
//! from the series peak `j_max` with an early break once a term drops `LD_EPS`
//! below the peak (misc.c:285-505). The numpy port (`hea/family.py::
//! _tweedie_log_a_vec`) instead materialises a dense `(n_active, J)` matrix
//! sized to the WIDEST row and reduces along j — O(n·J) work and memory. This
//! kernel keeps mgcv's per-row sweep: O(Σ_i width_i), no `(n, J)` array, rayon
//! over the (independent) rows.
//!
//! Two kernel families share this file:
//!
//!  * `tweedie_series` (scalar p) returns the four well-conditioned `j`-moments
//!    `(log_a, E[j], Var[j], E[jψ])` plus mgcv's THREE p-parameterisation
//!    working-derivative accumulators `(E[wp1], E[wp1²+wp2], E[wp1·j/(1−p)+wpp])`
//!    under `p_j = W_j/ΣW_k` — mgcv `wdlogwdp/wi`, `wdW2d2W/wi`, `dWpp/wi`
//!    (misc.c:346-352) BEFORE the θ-chain. It serves the classic fixed-p family's
//!    p-derivative path (`_d2ls_dp`/`dls_dp`), which reapplies `dpth1/dpth2`
//!    outside. Combining `wp1²+wp2` PER TERM (the curvature, which nearly
//!    cancels) — instead of summing `E[wp1²]` and `E[wp2]` separately and
//!    subtracting — avoids the ~1e-11 catastrophic cancellation of the split.
//!
//!  * `tweedious_work` / `tweedious_work_pv` (below) are the FAITHFUL ports of
//!    the C `tweedious`/`tweedious2`: they apply the θ-chain PER TERM inside the
//!    sweep and return the SIX working-parameter log-density derivatives
//!    `[w, w1, w2, w1p, w2p, w2pp]` directly (mgcv's C outputs), consumed by
//!    `_ld_tweedie_work`. Plain ops — the C source carries no explicit fma.
//!
//! Special functions are the bundled nmath ports (`lgammafn`, `psigamma_scalar`
//! — R's own Rmath C, NOT scipy), so the series matches the arm64 R build to the
//! libm floor. In `tweedie_series`, `rfma` is used where mgcv's C fuses on arm64
//! (the `wp1²+wp2` combine + the `j·wp_base+x` builds); the reductions follow
//! mgcv's statement structure (stored intermediates NOT fused). Gates: rust-vs-R
//! (ldTweedie oracle) + rust-vs-numpy (moments 1e-8; tweedious_work bit-exact).

use crate::nmath::lgamma::lgammafn;
use crate::nmath::psigamma::psigamma_scalar;
use crate::nmath::util::rfma;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Scalar-p Tweedie series + moments (mgcv `tweedious`). All special-function
/// Builds the shared `j = 1..=J` nmath tables once (α constant), then sweeps
/// each active row. Per row, `ly[i] = log(y[i])` and the integer peak `j_int[i]`
/// (1-based) drive `log W_j = j·w_base − lgamma(j+1) − lgamma(-jα) − j·α·ly`
/// (mgcv `wb[jb] − j·alogy`, misc.c:286/331). `w_base`/`wp_base`/`wp2_base`/`onep`
/// (= 1−p) are the scalar `p`-bases (misc.c:230-232,287-293). Returns the
/// `(n_active, 7)` matrix `[log_a, E[j], Var[j], E[jψ], E[wp1], E[wp1²+wp2],
/// E[wp1·j/(1−p)+wpp]]` (the last three are mgcv's p-param working accumulators).
/// `J` is sized by the caller (curvature + right-edge grow) so the eps gate fires
/// within `[1, J]` for every row.
#[pyfunction]
#[pyo3(name = "tweedie_series")]
#[allow(clippy::too_many_arguments)]
fn tweedie_series<'py>(
    py: Python<'py>,
    ly: PyReadonlyArray1<'py, f64>,
    j_int: PyReadonlyArray1<'py, i64>,
    alpha: f64,
    w_base: PyReadonlyArray1<'py, f64>,
    wp_base: PyReadonlyArray1<'py, f64>,
    wp2_base: PyReadonlyArray1<'py, f64>,
    onep: f64,
    j_cap: i64,
    ld_eps: f64,
) -> Bound<'py, PyArray2<f64>> {
    let ly = ly.as_slice().unwrap();
    let j_int = j_int.as_slice().unwrap();
    let w_base = w_base.as_slice().unwrap();
    let wp_base = wp_base.as_slice().unwrap();
    let wp2_base = wp2_base.as_slice().unwrap();
    let n = ly.len();
    let jmax = j_cap.max(1);
    let jcount = jmax as usize;
    let onep2 = onep * onep;

    // Shared length-J special-function tables (R's Rmath via nmath, α constant).
    let mut lgam_j1 = vec![0.0f64; jcount];
    let mut lgam_nja = vec![0.0f64; jcount];
    let mut psi = vec![0.0f64; jcount];
    let mut trig = vec![0.0f64; jcount];
    for k in 0..jcount {
        let jf = (k + 1) as f64;
        lgam_j1[k] = lgammafn(jf + 1.0);
        lgam_nja[k] = lgammafn(-jf * alpha);
        psi[k] = psigamma_scalar(-jf * alpha, 0.0);
        trig[k] = psigamma_scalar(-jf * alpha, 1.0);
    }

    let mut out = vec![0.0f64; n * 7];

    let fill = |i: usize, row: &mut [f64]| {
        let lyi = ly[i];
        let wb0 = w_base[i]; // per-row: phi (hence rho/onep) may vary by weight
        let wpb = wp_base[i];
        let wp2b = wp2_base[i];
        let alogy = alpha * lyi; // mgcv alogy[i] = α·log y (misc.c:248)
        let logy1p2 = lyi / onep2; // mgcv logy1p2 = log y/(1−p)² (misc.c:243)
        let logy1p3 = logy1p2 / onep; // mgcv logy1p3 = logy1p2/(1−p) (misc.c:250)
        let mut ji = j_int[i];
        if ji < 1 {
            ji = 1;
        }
        if ji > jmax {
            ji = jmax;
        }
        // log W_j: mgcv wb[jb] − j·alogy, wb[jb] = j·w_base − lgamma(j+1) − lgamma(-jα)
        let lw = |j: i64| -> f64 {
            let k = (j - 1) as usize;
            let jf = j as f64;
            let wb = rfma(jf, wb0, -lgam_j1[k]) - lgam_nja[k];
            wb - jf * alogy
        };
        let wmax = lw(ji);
        let wmin = wmax - ld_eps;

        let mut s0 = 0.0f64; // Σ W_j (scaled by exp(-wmax)) = mgcv wi
        let mut s1 = 0.0f64; // Σ W_j·j
        let mut s2 = 0.0f64; // Σ W_j·j²
        let mut sp1 = 0.0f64; // Σ W_j·j·ψ
        let mut s_wp1 = 0.0f64; // Σ W_j·wp1  (mgcv wdlogwdp, pre-chain)
        let mut s_comb = 0.0f64; // Σ W_j·(wp1²+wp2)  (mgcv wdW2d2W, pre-chain)
        let mut s_dwpp = 0.0f64; // Σ W_j·(wp1·j/(1−p)+wpp)  (mgcv dWpp, pre-chain)

        // One term's contribution given its 1-based index j and log-weight lwj.
        // wp1/wp2/wpp are the p-parameterisation working derivatives of log W_j
        // (mgcv misc.c:289-293,333-334), before the θ-chain. `wp1²+wp2` is fused
        // (`rfma`) — it is the per-term curvature whose pieces nearly cancel, so
        // summing them separately and subtracting loses ~1e-11; combining per
        // term keeps full precision (mgcv misc.c:392).
        macro_rules! accumulate {
            ($j:expr, $lwj:expr) => {{
                let k = ($j - 1) as usize;
                let jf = $j as f64;
                let w = ($lwj - wmax).exp();
                let psij = psi[k];
                let jf_o2 = jf / onep2;
                let xj = jf_o2 * psij;
                let wp1 = rfma(jf, wpb, xj) - jf * logy1p2;
                let wp2xx = trig[k] * jf_o2 * jf_o2;
                let wp2 = rfma(jf, wp2b, 2.0 * xj / onep) - wp2xx - 2.0 * jf * logy1p3;
                let wpp = jf_o2;
                s0 += w;
                s1 += w * jf;
                s2 += w * jf * jf;
                sp1 += w * (jf * psij);
                s_wp1 += w * wp1;
                s_comb += w * rfma(wp1, wp1, wp2);
                s_dwpp += w * (wp1 * jf / onep + wpp);
            }};
        }

        // Upsweep from the peak (inclusive), then downsweep — mgcv accumulates
        // each term THEN breaks once it drops below `wmin` (misc.c:329-358,415).
        let mut j = ji;
        loop {
            let lwj = if j == ji { wmax } else { lw(j) };
            accumulate!(j, lwj);
            if lwj < wmin || j >= jmax {
                break;
            }
            j += 1;
        }
        let mut j = ji - 1;
        while j >= 1 {
            let lwj = lw(j);
            accumulate!(j, lwj);
            if lwj < wmin {
                break;
            }
            j -= 1;
        }

        let jb = s1 / s0;
        row[0] = wmax + s0.ln(); // log_a (log-sum-exp, reference-invariant)
        row[1] = jb;
        row[2] = s2 / s0 - jb * jb; // Var[j] = E[j²]−E[j]² (mgcv misc.c:499 form)
        row[3] = sp1 / s0;
        row[4] = s_wp1 / s0; // E[wp1]            (mgcv wdlogwdp/wi)
        row[5] = s_comb / s0; // E[wp1²+wp2]      (mgcv wdW2d2W/wi)
        row[6] = s_dwpp / s0; // E[wp1·j/(1−p)+wpp] (mgcv dWpp/wi)
    };

    if n >= crate::par::PAR_THRESHOLD {
        py.allow_threads(|| {
            out.par_chunks_mut(7)
                .enumerate()
                .for_each(|(i, row)| fill(i, row));
        });
    } else {
        out.chunks_mut(7)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    Array2::from_shape_vec((n, 7), out)
        .unwrap()
        .into_pyarray(py)
}

/// Scalar-p tweedious working-parameter derivatives (mgcv C `tweedious`,
/// misc.c:170-510). Unlike `tweedie_series` (which returns p-param MOMENTS and
/// lets the caller reapply the θ-chain), this applies the chain PER TERM inside
/// the sweep and returns the six working-parameter log-density derivatives
/// `[w, w1, w2, w1p, w2p, w2pp]` = `[logW, dρ, dρρ, dθ, dθθ, dθρ]` — mgcv's C
/// outputs, added to the saddle part by `_ld_tweedie_work`. `rho` is scalar
/// (single φ, buffer=TRUE path); `p`/`dpth1`/`dpth2` scalar. Bases are computed
/// internally (misc.c:227-232). Plain ops — the C source carries no explicit
/// fma (its accumulators use stored temporaries, not fused within one
/// expression). Twin of `_tweedious_work_scalar_py`.
#[pyfunction]
#[pyo3(name = "tweedious_work")]
#[allow(clippy::too_many_arguments)]
fn tweedious_work<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    rho: f64,
    p: f64,
    dpth1: f64,
    dpth2: f64,
    log_eps: f64,
    j_cap: i64,
) -> Bound<'py, PyArray2<f64>> {
    let y = y.as_slice().unwrap();
    let n = y.len();
    let phi = rho.exp();
    let onep = 1.0 - p;
    let onep2 = onep * onep;
    let twop = 2.0 - p;
    let alpha = twop / onep;
    let w_base = alpha * (p - 1.0).ln() + rho / onep - twop.ln();
    let log_neg = (-onep).ln() + rho;
    let wp_base = log_neg / onep2 - alpha / onep + 1.0 / twop;
    let wp2_base =
        2.0 * log_neg / (onep2 * onep) - (3.0 * alpha - 2.0) / onep2 + 1.0 / (twop * twop);

    let mut out = vec![0.0f64; n * 6];
    let fill = |i: usize, row: &mut [f64]| {
        let lyi = y[i].ln();
        let alogy = alpha * lyi;
        let logy1p2 = lyi / onep2;
        let logy1p3 = logy1p2 / onep;
        // locate the series maximum (misc.c:303-305)
        let x = y[i].powf(twop) / (phi * twop);
        let mut jm = x.floor();
        if x - jm > 0.5 || jm < 1.0 {
            jm += 1.0;
        }
        let jm = jm as i64;
        let jmf = jm as f64;
        let wmax = jmf * w_base - lgammafn(jmf + 1.0) - lgammafn(-jmf * alpha) - jmf * alogy;
        let wmin = wmax + log_eps;
        let mut wi = 0.0f64;
        let mut w1i = 0.0f64;
        let mut w2i = 0.0f64;
        let mut wdlogwdp = 0.0f64;
        let mut wd_w2d2w = 0.0f64;
        let mut dwpp = 0.0f64;

        // Per-term (misc.c:329-352): p-param wp1/wp2, θ-chain, accumulate.
        macro_rules! accumulate {
            ($j:expr) => {{
                let jf = $j as f64;
                let wj = jf * w_base - lgammafn(jf + 1.0) - lgammafn(-jf * alpha) - jf * alogy;
                let w1j = -jf / onep;
                let nja = -jf * alpha;
                let xx = jf / onep2;
                let xdig = xx * psigamma_scalar(nja, 0.0);
                let wp1j0 = (jf * wp_base + xdig) - jf * logy1p2;
                let wp2j0 = (jf * wp2_base + 2.0 * xdig / onep
                    - psigamma_scalar(nja, 1.0) * xx * xx)
                    - 2.0 * jf * logy1p3;
                let wp2j = wp1j0 * dpth2 + wp2j0 * dpth1 * dpth1;
                let wp1j = wp1j0 * dpth1;
                let wppj = xx * dpth1;
                let ws = (wj - wmax).exp();
                wi += ws;
                w1i += ws * w1j;
                w2i += ws * w1j * w1j;
                wdlogwdp += ws * wp1j;
                wd_w2d2w += ws * (wp1j * wp1j + wp2j);
                dwpp += ws * (wp1j * jf / onep + wppj);
                wj
            }};
        }

        let mut j = jm;
        loop {
            let wj = accumulate!(j);
            if wj < wmin {
                break;
            }
            j += 1;
            if j - jm > j_cap {
                break;
            }
        }
        let mut j = jm - 1;
        while j >= 1 {
            let wj = accumulate!(j);
            if wj < wmin {
                break;
            }
            j -= 1;
        }

        row[0] = wmax + wi.ln();
        row[1] = -w1i / wi;
        row[2] = w2i / wi - (w1i / wi) * (w1i / wi);
        row[3] = wdlogwdp / wi;
        row[4] = wd_w2d2w / wi - (wdlogwdp / wi) * (wdlogwdp / wi);
        row[5] = (w1i / wi) * (wdlogwdp / wi) + dwpp / wi;
    };

    if n >= crate::par::PAR_THRESHOLD {
        py.allow_threads(|| {
            out.par_chunks_mut(6)
                .enumerate()
                .for_each(|(i, row)| fill(i, row));
        });
    } else {
        out.chunks_mut(6)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    Array2::from_shape_vec((n, 6), out)
        .unwrap()
        .into_pyarray(py)
}

/// Vector-p tweedious working-parameter derivatives (mgcv C `tweedious2`,
/// misc.c:513-661). Per-row `rho`/`p`/`dpth1`/`dpth2`; bases recomputed per row.
/// `lgamma(j+1)` is built by RECURSION along the sweep (`+= ln(j)` up,
/// `-= ln(j+1)` down) from a fresh `lgamma` at each sweep's start — mgcv's
/// `lgammaj1` recursion (misc.c:596-644). Returns `(n, 6)` `[w,w1,w2,w1p,w2p,
/// w2pp]`. Twin of `_tweedious_work_pv_py`.
#[pyfunction]
#[pyo3(name = "tweedious_work_pv")]
#[allow(clippy::too_many_arguments)]
fn tweedious_work_pv<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    dpth1: PyReadonlyArray1<'py, f64>,
    dpth2: PyReadonlyArray1<'py, f64>,
    log_eps: f64,
    _j_cap: i64,
) -> Bound<'py, PyArray2<f64>> {
    let y = y.as_slice().unwrap();
    let rho = rho.as_slice().unwrap();
    let p = p.as_slice().unwrap();
    let dpth1 = dpth1.as_slice().unwrap();
    let dpth2 = dpth2.as_slice().unwrap();
    let n = y.len();

    let mut out = vec![0.0f64; n * 6];
    let fill = |i: usize, row: &mut [f64]| {
        let pi = p[i];
        let d1 = dpth1[i];
        let d2 = dpth2[i];
        let phi = rho[i].exp();
        let onep = 1.0 - pi;
        let onep2 = onep * onep;
        let twop = 2.0 - pi;
        let alpha = twop / onep;
        let alogy_raw = y[i].ln();
        let logy1p2 = alogy_raw / onep2;
        let logy1p3 = logy1p2 / onep;
        let alogy = alogy_raw * alpha;
        let w_base = alpha * (-onep).ln() + rho[i] / onep - twop.ln();
        let log_neg = (-onep).ln() + rho[i];
        let wp_base = log_neg / onep2 - alpha / onep + 1.0 / twop;
        let wp2_base =
            2.0 * log_neg / (onep2 * onep) - (3.0 * alpha - 2.0) / onep2 + 1.0 / (twop * twop);
        let x = y[i].powf(twop) / (phi * twop);
        let mut jm = x.floor();
        if x - jm > 0.5 || jm < 1.0 {
            jm += 1.0;
        }
        let jm = jm as i64;
        let jmf = jm as f64;
        let wmax = jmf * w_base - lgammafn(jmf + 1.0) - lgammafn(-jmf * alpha) - jmf * alogy;
        let wmin = wmax + log_eps;
        let mut wi = 0.0f64;
        let mut w1i = 0.0f64;
        let mut w2i = 0.0f64;
        let mut wdlogwdp = 0.0f64;
        let mut wd_w2d2w = 0.0f64;
        let mut dwpp = 0.0f64;

        // Single toggling sweep with the lgamma(j+1) recursion (misc.c:595-648).
        let mut j = jm;
        let mut incr: i64 = 1;
        let mut lgammaj1 = lgammafn(jmf + 1.0);
        let mut ok = false;
        while !ok {
            let jf = j as f64;
            let wbj = jf * w_base - lgammaj1 - lgammafn(-jf * alpha);
            let w1j = -jf / onep;
            let nja = -jf * alpha;
            let xx = jf / onep2;
            let xdig = xx * psigamma_scalar(nja, 0.0);
            let wp1jb = jf * wp_base + xdig;
            let wp2jb = jf * wp2_base + 2.0 * xdig / onep - psigamma_scalar(nja, 1.0) * xx * xx;
            let wj = wbj - jf * alogy;
            let wp1j0 = wp1jb - jf * logy1p2;
            let wp2j0 = wp2jb - 2.0 * jf * logy1p3;
            let wp2j = wp1j0 * d2 + wp2j0 * d1 * d1;
            let wp1j = wp1j0 * d1;
            let wppj = xx * d1;
            let ws = (wj - wmax).exp();
            wi += ws;
            w1i += ws * w1j;
            w2i += ws * w1j * w1j;
            wdlogwdp += ws * wp1j;
            wd_w2d2w += ws * (wp1j * wp1j + wp2j);
            dwpp += ws * (wp1j * jf / onep + wppj);
            j += incr;
            if incr > 0 {
                lgammaj1 += (j as f64).ln();
                if wj < wmin {
                    j = jm - 1;
                    incr = -1;
                    if j == 0 {
                        ok = true;
                    }
                    lgammaj1 = lgammafn(j as f64 + 1.0);
                }
            } else {
                lgammaj1 += -((j as f64) + 1.0).ln();
                if wj < wmin || j < 1 {
                    ok = true;
                }
            }
        }
        row[0] = wmax + wi.ln();
        row[1] = -w1i / wi;
        row[2] = w2i / wi - (w1i / wi) * (w1i / wi);
        row[3] = wdlogwdp / wi;
        row[4] = wd_w2d2w / wi - (wdlogwdp / wi) * (wdlogwdp / wi);
        row[5] = (w1i / wi) * (wdlogwdp / wi) + dwpp / wi;
    };

    if n >= crate::par::PAR_THRESHOLD {
        py.allow_threads(|| {
            out.par_chunks_mut(6)
                .enumerate()
                .for_each(|(i, row)| fill(i, row));
        });
    } else {
        out.chunks_mut(6)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    Array2::from_shape_vec((n, 6), out)
        .unwrap()
        .into_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tweedie_series, m)?)?;
    m.add_function(wrap_pyfunction!(tweedious_work, m)?)?;
    m.add_function(wrap_pyfunction!(tweedious_work_pv, m)?)?;
    Ok(())
}

use pyo3::wrap_pyfunction;
