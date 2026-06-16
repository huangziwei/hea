//! LOESS local-regression pass — the O(n²) hot loop of hea's `loess()`.
//!
//! Each query point's local weighted polynomial fit is independent, so the whole
//! pass parallelizes with rayon. Mirrors `hea/ggplot/stats/smooth.py`
//! `_loess_local_fit` (degrees 1 and 2): k-NN tricube bandwidth, then a weighted
//! local polynomial; returns per-query `(fitted = β0, var00 = (XᵀWX)⁻¹[0,0])`.
//!
//! Numerics: we form the weighted NORMAL EQUATIONS in a SCALED monomial basis
//! `t = (x_i − xq)/h` (|t| < 1 inside the tricube window). Column scaling by
//! `D = diag(1, h, h², …)` leaves the intercept `β0` (the fitted value) and
//! `var00 = (XᵀWX)⁻¹[0,0]` mathematically UNCHANGED (because `D[0,0] = 1`), while
//! keeping the moment matrix well-conditioned — so the small solve matches the
//! Python lstsq path closely without squaring the condition number. Both hea's
//! fit and predict route through this one kernel, so fitted == predict-at-x
//! bit-for-bit regardless of the serial/parallel split.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

// Each query does O(n_data) work, so the crossover is far below the element-wise
// PAR_THRESHOLD used for the d/p/q kernels — parallelize once there are enough
// independent queries to amortize the thread dispatch.
const LOESS_PAR_MIN_QUERIES: usize = 64;

/// One local fit at `xq`. Returns `(fitted = β0, var00 = (XᵀWX)⁻¹[0,0])`,
/// matching `_loess_local_fit`'s special cases (exact ties, rank-deficient).
fn local_fit(
    xq: f64,
    x: &[f64],
    y: &[f64],
    span: f64,
    degree: usize,
    w_extra: &[f64],
    want_var: bool,
) -> (f64, f64) {
    let n = x.len();
    let p = degree + 1;
    let k = ((span * n as f64).ceil() as usize).max(p);

    // bandwidth h = k-th smallest distance (or max distance when k ≥ n)
    let mut dist: Vec<f64> = x.iter().map(|&xi| (xi - xq).abs()).collect();
    let h = if k >= n {
        dist.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else {
        *dist
            .select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap())
            .1
    };

    if h <= 0.0 {
        // exact ties at xq: weighted average of the coincident y's
        let mut sw = 0.0;
        let mut swy = 0.0;
        for i in 0..n {
            if x[i] == xq {
                sw += w_extra[i];
                swy += w_extra[i] * y[i];
            }
        }
        if sw == 0.0 {
            return (0.0, 1e300);
        }
        return (swy / sw, f64::INFINITY);
    }

    // weighted power sums in the scaled basis t = (x − xq)/h:
    //   S[j] = Σ w·tʲ     (j = 0..=2·degree)
    //   T[j] = Σ w·tʲ·y   (j = 0..=degree)
    let mut s = [0.0f64; 5];
    let mut t_rhs = [0.0f64; 3];
    let mut nz = 0usize;
    for i in 0..n {
        let delta = x[i] - xq;
        let u = delta.abs() / h;
        if u < 1.0 {
            let a = 1.0 - u * u * u;
            let tri = a * a * a; // tricube
            let w = tri * w_extra[i];
            if w > 0.0 {
                nz += 1;
            }
            let t = delta / h;
            let mut tp = 1.0;
            for j in 0..=(2 * degree) {
                s[j] += w * tp;
                if j <= degree {
                    t_rhs[j] += w * tp * y[i];
                }
                tp *= t;
            }
        }
    }
    if nz < p {
        return (f64::NAN, f64::NAN);
    }

    // M[a][b] = S[a+b]; solve M·β = T → β0, and M·z = e0 → var00 = z0 = inv[0,0]
    let mut m = [[0.0f64; 3]; 3];
    for a in 0..p {
        for b in 0..p {
            m[a][b] = s[a + b];
        }
    }
    let fitted = solve_first(&m, p, &t_rhs);
    let var00 = if want_var {
        let mut e0 = [0.0f64; 3];
        e0[0] = 1.0;
        solve_first(&m, p, &e0)
    } else {
        f64::NAN
    };
    (fitted, var00)
}

/// Solve `M·x = rhs` (p×p, p ≤ 3) by Gaussian elimination with partial
/// pivoting; return `x[0]`. `M` is the well-conditioned scaled moment matrix.
fn solve_first(m_in: &[[f64; 3]; 3], p: usize, rhs: &[f64; 3]) -> f64 {
    let mut a = *m_in;
    let mut b = *rhs;
    for col in 0..p {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..p {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let d = a[col][col];
        if d == 0.0 {
            return f64::NAN;
        }
        for r in (col + 1)..p {
            let f = a[r][col] / d;
            for c in col..p {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..p).rev() {
        let mut acc = b[i];
        for c in (i + 1)..p {
            acc -= a[i][c] * x[c];
        }
        x[i] = acc / a[i][i];
    }
    x[0]
}

/// Evaluate the local fits at every query in `xq` against data `(x, y)` with
/// robustness weights `extra_w`. Returns `(fitted, var00)` arrays. Used for both
/// the fit (queries = the data x) and predict (queries = new x).
#[pyfunction]
#[pyo3(name = "loess_eval", signature = (xq, x, y, span, degree, extra_w, want_var=true))]
pub fn loess_eval<'py>(
    py: Python<'py>,
    xq: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    span: f64,
    degree: usize,
    extra_w: PyReadonlyArray1<'py, f64>,
    want_var: bool,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let q = xq.as_slice().unwrap();
    let xs = x.as_slice().unwrap();
    let ys = y.as_slice().unwrap();
    let ws = extra_w.as_slice().unwrap();
    let nq = q.len();

    let pairs: Vec<(f64, f64)> = if nq >= LOESS_PAR_MIN_QUERIES {
        py.allow_threads(|| {
            (0..nq)
                .into_par_iter()
                .map(|i| local_fit(q[i], xs, ys, span, degree, ws, want_var))
                .collect()
        })
    } else {
        (0..nq)
            .map(|i| local_fit(q[i], xs, ys, span, degree, ws, want_var))
            .collect()
    };

    let mut fitted = Vec::with_capacity(nq);
    let mut lev = Vec::with_capacity(nq);
    for (f, v) in pairs {
        fitted.push(f);
        lev.push(v);
    }
    (fitted.into_pyarray(py), lev.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(loess_eval, m)?)?;
    Ok(())
}
