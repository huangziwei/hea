//! Tweedie density series — mechanical port of mgcv's `tweedious` (src/misc.c:170),
//! the per-row windowed summation of the Dunn & Smyth (2005) series for
//! `log a(y, φ, p) = log Σ_{j≥1} W_j` and its j-moments.
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
//! Faithful to hea's MOMENT abstraction, not mgcv's assembled working-parameter
//! derivatives: it returns the same seven quantities `_tweedie_log_a_vec` does
//! — `(log_a, E[j], Var[j], E[jψ], E[j²ψ], E[(jψ)²], E[j²ψ'])` under
//! `p_j = W_j/ΣW_k` — so every existing Python caller is unchanged. The
//! special-function TABLES are precomputed by the caller with the SAME scipy
//! routines the numpy path uses and passed in, so the only divergence from the
//! numpy oracle is the sweep's reduction order (sub-ULP; the variance uses the
//! uncentered `E[j²]−E[j]²` form, which is exactly what mgcv accumulates at
//! misc.c:499). Plain `a*b+c` (no fused multiply-add) to track numpy, not the
//! arm64-fused R build — the gate is rust-vs-numpy-oracle atol=1e-9.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Scalar-p Tweedie series + moments (mgcv `tweedious`). All special-function
/// tables are indexed by `j-1` for the 1-based series index `j = 1..=J`:
/// `lgam_j1[j-1] = lgamma(j+1)`, `lgam_nja[j-1] = lgamma(-jα)`,
/// `psi[j-1] = digamma(-jα)`, `trig[j-1] = trigamma(-jα)`. Per active row,
/// `log_z[i]` is the y/φ-dependent base (`log W_j = j·log_z − lgamma(j+1) −
/// lgamma(-jα)`) and `j_int[i]` the integer peak (1-based). Returns the
/// `(n_active, 7)` moment matrix in column order `[log_a, E[j], Var[j], E[jψ],
/// E[j²ψ], E[(jψ)²], E[j²ψ']]`. The caller has sized `J` (via curvature +
/// right-edge grow) so the eps gate provably fires within `[1, J]` for every
/// row, so the sweep never needs a table entry past `J`.
#[pyfunction]
#[pyo3(name = "tweedie_series")]
#[allow(clippy::too_many_arguments)]
fn tweedie_series<'py>(
    py: Python<'py>,
    log_z: PyReadonlyArray1<'py, f64>,
    j_int: PyReadonlyArray1<'py, i64>,
    lgam_j1: PyReadonlyArray1<'py, f64>,
    lgam_nja: PyReadonlyArray1<'py, f64>,
    psi: PyReadonlyArray1<'py, f64>,
    trig: PyReadonlyArray1<'py, f64>,
    near: i64,
    ld_eps: f64,
) -> Bound<'py, PyArray2<f64>> {
    let log_z = log_z.as_slice().unwrap();
    let j_int = j_int.as_slice().unwrap();
    let lgam_j1 = lgam_j1.as_slice().unwrap();
    let lgam_nja = lgam_nja.as_slice().unwrap();
    let psi = psi.as_slice().unwrap();
    let trig = trig.as_slice().unwrap();
    let n = log_z.len();
    let jmax = lgam_j1.len() as i64; // == J

    let mut out = vec![0.0f64; n * 7];

    let fill = |i: usize, row: &mut [f64]| {
        let lz = log_z[i];
        let mut ji = j_int[i];
        if ji < 1 {
            ji = 1;
        }
        if ji > jmax {
            ji = jmax;
        }
        // log W_j with j 1-based; the shared tables are indexed j-1.
        let lw = |j: i64| -> f64 {
            let k = (j - 1) as usize;
            (j as f64) * lz - lgam_j1[k] - lgam_nja[k]
        };
        let wmax = lw(ji);
        let wmin = wmax - ld_eps;

        let mut s0 = 0.0f64; // Σ W_j (scaled by exp(-wmax))
        let mut s1 = 0.0f64; // Σ W_j·j
        let mut s2 = 0.0f64; // Σ W_j·j²
        let mut sp1 = 0.0f64; // Σ W_j·j·ψ
        let mut sp2 = 0.0f64; // Σ W_j·j²·ψ
        let mut spp = 0.0f64; // Σ W_j·(j·ψ)²
        let mut st = 0.0f64; // Σ W_j·j²·ψ'

        // One term's contribution given its 1-based index j and log-weight lwj.
        macro_rules! accumulate {
            ($j:expr, $lwj:expr) => {{
                let k = ($j - 1) as usize;
                let jf = $j as f64;
                let w = ($lwj - wmax).exp();
                let jpsi = jf * psi[k];
                s0 += w;
                s1 += w * jf;
                s2 += w * jf * jf;
                sp1 += w * jpsi;
                sp2 += w * jf * jpsi;
                spp += w * jpsi * jpsi;
                st += w * jf * jf * trig[k];
            }};
        }

        // peak (lw == wmax ⇒ scaled weight 1)
        accumulate!(ji, wmax);
        // up-sweep: continue while within `near` of the peak OR above the eps
        // gate; break once beyond both (unimodal ⇒ all further terms smaller).
        let mut j = ji + 1;
        while j <= jmax {
            let lwj = lw(j);
            if j - ji > near && lwj < wmin {
                break;
            }
            accumulate!(j, lwj);
            j += 1;
        }
        // down-sweep toward j = 1.
        let mut j = ji - 1;
        while j >= 1 {
            let lwj = lw(j);
            if ji - j > near && lwj < wmin {
                break;
            }
            accumulate!(j, lwj);
            j -= 1;
        }

        let jb = s1 / s0;
        row[0] = wmax + s0.ln(); // log_a (log-sum-exp, reference-invariant)
        row[1] = jb;
        row[2] = s2 / s0 - jb * jb; // Var[j] = E[j²]−E[j]² (mgcv misc.c:499 form)
        row[3] = sp1 / s0;
        row[4] = sp2 / s0;
        row[5] = spp / s0;
        row[6] = st / s0;
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
    Ok(())
}

use pyo3::wrap_pyfunction;
