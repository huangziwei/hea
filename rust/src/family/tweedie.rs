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
//! It returns the four well-conditioned `j`-moments `(log_a, E[j], Var[j], E[jψ])`
//! plus mgcv's THREE p-parameterisation working-derivative accumulators
//! `(E[wp1], E[wp1²+wp2], E[wp1·j/(1−p) + wpp])` under `p_j = W_j/ΣW_k`. The last
//! three are mgcv `wdlogwdp/wi`, `wdW2d2W/wi`, `dWpp/wi` (misc.c:346-352) BEFORE
//! the θ-chain (the chain factors `dpth1/dpth2` are pulled out and reapplied by
//! the caller — `_ld_tweedie_work` for θ, `_d2ls_dp` for p). Combining
//! `wp1²+wp2` PER TERM (the curvature, which nearly cancels) — instead of summing
//! `E[wp1²]` and `E[wp2]` separately and subtracting — is the whole point: the
//! moment-split lost ~1e-11 to catastrophic cancellation in the 2nd derivatives.
//!
//! Special functions are evaluated with the bundled nmath ports (`lgammafn`,
//! `psigamma_scalar` — R's own Rmath C, NOT scipy), so the series matches the
//! arm64 R build to the libm floor. `rfma` is used where mgcv's C fuses on arm64
//! (the `wp1²+wp2` curvature combine + the `j·wp_base+x` term builds); the
//! per-row reductions follow mgcv's statement structure (stored intermediates are
//! NOT fused). Gate: rust-vs-R (the new oracle arm) + rust-vs-numpy at rtol 1e-8.

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
        // summing them separately and subtracting (the old moment split) lost
        // ~1e-11; combining per term keeps full precision (mgcv misc.c:392).
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

/// Vector-p Tweedie series + working derivatives (mgcv `tweedious2`, misc.c:513).
/// Here α (and the `p`-bases) vary by row, so the special functions CANNOT be
/// shared into length-J tables — `tweedious2` recomputes `lgamma(j+1)`,
/// `lgamma(-jα)`, `digamma(-jα)`, `trigamma(-jα)` inside its per-row sweep. This
/// kernel does the same via the nmath ports (R's own Rmath C), evaluating the
/// special functions only on each row's eps-window (`O(Σ_i width_i)`). Per active
/// row: `ly[i] = log y`, the integer peak `j_int[i]` (1-based), and the row's
/// `alpha`/`w_base`/`wp_base`/`wp2_base`/`onep`(=1−p). `j_max` caps the up-sweep
/// (`_LD_J_MAX`). Same `(n_active, 7)` column order as `tweedie_series`.
#[pyfunction]
#[pyo3(name = "tweedie_series_pv")]
#[allow(clippy::too_many_arguments)]
fn tweedie_series_pv<'py>(
    py: Python<'py>,
    ly: PyReadonlyArray1<'py, f64>,
    j_int: PyReadonlyArray1<'py, i64>,
    alpha: PyReadonlyArray1<'py, f64>,
    w_base: PyReadonlyArray1<'py, f64>,
    wp_base: PyReadonlyArray1<'py, f64>,
    wp2_base: PyReadonlyArray1<'py, f64>,
    onep: PyReadonlyArray1<'py, f64>,
    ld_eps: f64,
    j_max: i64,
) -> Bound<'py, PyArray2<f64>> {
    let ly = ly.as_slice().unwrap();
    let j_int = j_int.as_slice().unwrap();
    let alpha = alpha.as_slice().unwrap();
    let w_base = w_base.as_slice().unwrap();
    let wp_base = wp_base.as_slice().unwrap();
    let wp2_base = wp2_base.as_slice().unwrap();
    let onep = onep.as_slice().unwrap();
    let n = ly.len();

    let mut out = vec![0.0f64; n * 7];

    let fill = |i: usize, row: &mut [f64]| {
        let lyi = ly[i];
        let a = alpha[i];
        let wb0 = w_base[i];
        let wpb = wp_base[i];
        let wp2b = wp2_base[i];
        let op = onep[i];
        let op2 = op * op;
        let alogy = a * lyi;
        let logy1p2 = lyi / op2;
        let logy1p3 = logy1p2 / op;
        let mut ji = j_int[i];
        if ji < 1 {
            ji = 1;
        }
        if ji > j_max {
            ji = j_max;
        }
        // log W_j = j·w_base − lgamma(j+1) − lgamma(−jα) − j·alogy, j 1-based.
        let lw = |j: i64| -> f64 {
            let jf = j as f64;
            let wb = rfma(jf, wb0, -lgammafn(jf + 1.0)) - lgammafn(-jf * a);
            wb - jf * alogy
        };
        let wmax = lw(ji);
        let wmin = wmax - ld_eps;

        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut sp1 = 0.0f64;
        let mut s_wp1 = 0.0f64;
        let mut s_comb = 0.0f64;
        let mut s_dwpp = 0.0f64;

        // One term; ψ/ψ' recomputed at −jα (α is per-row). wp1/wp2/wpp are the
        // p-param working derivatives (mgcv misc.c:289-293,333-334); `wp1²+wp2`
        // is fused (`rfma`) — the per-term curvature combine that avoids the
        // moment-split cancellation (mgcv misc.c:392).
        macro_rules! accumulate {
            ($j:expr, $lwj:expr) => {{
                let jf = $j as f64;
                let nja = -jf * a;
                let psij = psigamma_scalar(nja, 0.0);
                let trigj = psigamma_scalar(nja, 1.0);
                let w = ($lwj - wmax).exp();
                let jf_o2 = jf / op2;
                let xj = jf_o2 * psij;
                let wp1 = rfma(jf, wpb, xj) - jf * logy1p2;
                let wp2xx = trigj * jf_o2 * jf_o2;
                let wp2 = rfma(jf, wp2b, 2.0 * xj / op) - wp2xx - 2.0 * jf * logy1p3;
                let wpp = jf_o2;
                s0 += w;
                s1 += w * jf;
                s2 += w * jf * jf;
                sp1 += w * (jf * psij);
                s_wp1 += w * wp1;
                s_comb += w * rfma(wp1, wp1, wp2);
                s_dwpp += w * (wp1 * jf / op + wpp);
            }};
        }

        let mut j = ji;
        loop {
            let lwj = if j == ji { wmax } else { lw(j) };
            accumulate!(j, lwj);
            if lwj < wmin || j >= j_max {
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
        row[0] = wmax + s0.ln();
        row[1] = jb;
        row[2] = s2 / s0 - jb * jb;
        row[3] = sp1 / s0;
        row[4] = s_wp1 / s0;
        row[5] = s_comb / s0;
        row[6] = s_dwpp / s0;
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tweedie_series, m)?)?;
    m.add_function(wrap_pyfunction!(tweedie_series_pv, m)?)?;
    Ok(())
}

use pyo3::wrap_pyfunction;
