//! R's least-squares QR kernel — a mechanical port of the LINPACK routines R
//! calls from `lm.fit`/`lm.wfit` via `Cdqrls`: `dqrdc2` (R's modified pivoted
//! Householder QR, `dqrdc2.f`) + `dqrsl` (apply Q, `dqrsl.f`,
//! the `job=1110` path that `dqrls.f` uses: `Qᵀy`, coef, residuals).
//!
//! The point is *determinism + R-faithfulness* (the `chol.rs` precedent): the
//! BLAS kernels (`dnrm2`/`ddot`/`daxpy`/`dscal`) accumulate strictly in order,
//! so the rank/pivot/coef/effects are bit-identical across platform/run — unlike
//! Accelerate/OpenBLAS, whose reduction order is the source of hea's
//! BLAS-bistable rank flakes. Spec/oracle: `hea/R/linalg.py`; checked 3-way
//! (Rust ≡ Python ≡ live R) in `tests/`.
//!
//! Storage is **column-major** internally (matching the Fortran `x(ldx,p)`), so
//! a column `j` is the contiguous slice `x[j*n .. j*n+n]`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Reference-BLAS `dnrm2` — Euclidean norm via the classic scaled sum-of-squares
/// (overflow-safe, in-order → deterministic, matches R's `dnrm2`).
pub(crate) fn dnrm2(v: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for &xi in v {
        if xi != 0.0 {
            let absxi = xi.abs();
            if scale < absxi {
                let r = scale / absxi;
                ssq = 1.0 + ssq * r * r;
                scale = absxi;
            } else {
                let r = absxi / scale;
                ssq += r * r;
            }
        }
    }
    if scale == 0.0 {
        0.0
    } else {
        scale * ssq.sqrt()
    }
}

/// In-order dot product (deterministic). Iterator form so the bounds checks are
/// elided and the fold stays strictly sequential (f64 `sum` is not reassociated
/// without fast-math, so this matches R's `ddot` order — no SIMD reduction).
fn ddot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y += t*x` over equal-length slices (deterministic daxpy, bounds-check-free).
fn daxpy(t: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += t * xi;
    }
}

/// Copy a 2-D view into a fresh column-major `Vec`, as cheaply as the input
/// layout allows. When the array is already F-contiguous (the polars-origin
/// path — its memory order *is* column-major) this is a single `memcpy` with no
/// zero-fill and no per-element stride math; only a C-order/strided caller pays
/// the transpose loop.
fn to_colmajor(xv: &numpy::ndarray::ArrayView2<f64>) -> Vec<f64> {
    let (n, p) = (xv.nrows(), xv.ncols());
    if !xv.is_standard_layout() {
        if let Some(s) = xv.as_slice_memory_order() {
            if s.len() == n * p {
                return s.to_vec(); // F-contiguous: column-major already
            }
        }
    }
    let mut v = vec![0.0_f64; n * p];
    for j in 0..p {
        for i in 0..n {
            v[j * n + i] = xv[[i, j]];
        }
    }
    v
}

/// R's `dqrdc2` (lower-level): factor the column-major `n×p` matrix `x`
/// in place. Returns `(qraux, jpvt, rank)`; `jpvt` is 1-based.
fn dqrdc2(x: &mut [f64], n: usize, p: usize, tol: f64) -> (Vec<f64>, Vec<usize>, usize) {
    let mut qraux = vec![0.0_f64; p];
    let mut work1 = vec![0.0_f64; p]; // work(:,1) — original norms
    let mut work2 = vec![0.0_f64; p]; // work(:,2) — original norms, 0→1
    let mut jpvt: Vec<usize> = (1..=p).collect();

    // column norms
    if n > 0 {
        for j in 0..p {
            let nrm = dnrm2(&x[j * n..j * n + n]);
            qraux[j] = nrm;
            work1[j] = nrm;
            work2[j] = if nrm == 0.0 { 1.0 } else { nrm };
        }
    }

    let lup = n.min(p);
    let mut k = p + 1; // 1-based rank boundary
    let mut tmpcol = vec![0.0_f64; n];
    for l in 1..=lup {
        let l0 = l - 1;
        // cycle negligible columns l..p to the right
        while !(l >= k || qraux[l0] >= work2[l0] * tol) {
            // cyclic left-shift of columns l0..p-1, moving column l0 to p-1
            tmpcol.copy_from_slice(&x[l0 * n..l0 * n + n]);
            x.copy_within((l0 + 1) * n..p * n, l0 * n);
            x[(p - 1) * n..p * n].copy_from_slice(&tmpcol);
            let (isv, tsv, tt1, tt2) = (jpvt[l0], qraux[l0], work1[l0], work2[l0]);
            for j in l0..p - 1 {
                jpvt[j] = jpvt[j + 1];
                qraux[j] = qraux[j + 1];
                work1[j] = work1[j + 1];
                work2[j] = work2[j + 1];
            }
            jpvt[p - 1] = isv;
            qraux[p - 1] = tsv;
            work1[p - 1] = tt1;
            work2[p - 1] = tt2;
            k -= 1;
        }
        if l != n {
            // Householder for column l (rows l..n) — slice x(l:n, l)
            let col_l = l0 * n; // base of column l0
            let mut nrmxl = dnrm2(&x[col_l + l0..col_l + n]);
            if nrmxl != 0.0 {
                if x[col_l + l0] != 0.0 {
                    nrmxl = nrmxl.copysign(x[col_l + l0]);
                }
                let inv = 1.0 / nrmxl;
                for xi in x[col_l + l0..col_l + n].iter_mut() {
                    *xi *= inv; // dscal
                }
                x[col_l + l0] += 1.0;
                let pivot_diag = x[col_l + l0];
                // Split so the pivot column l0 (read) and the trailing columns
                // l0+1..p (written) are disjoint borrows. Each trailing column's
                // update depends ONLY on the shared pivot column `cl` (read-only)
                // — columns are mutually independent, so they run in parallel
                // (rayon) with each column's dot/daxpy kept strictly in order →
                // bit-identical regardless of thread count (deterministic). This
                // is the lever that beats R's single-threaded dqrdc2 at scale.
                let (left, right) = x.split_at_mut((l0 + 1) * n);
                let cl = &left[col_l + l0..col_l + n]; // pivot col, rows l0..n
                                                       // per-column Householder update (col rows l0..n, its qraux, work1)
                let update = |cj: &mut [f64], qx: &mut f64, w1: &mut f64| {
                    let cj = &mut cj[l0..n]; // rows l0..n of this column
                    let dot: f64 = cl.iter().zip(cj.iter()).map(|(a, b)| a * b).sum();
                    let t = -dot / pivot_diag;
                    for (b, &a) in cj.iter_mut().zip(cl.iter()) {
                        *b += t * a; // daxpy
                    }
                    if *qx != 0.0 {
                        let ratio = cj[0].abs() / *qx; // |x(l,j)|
                        let mut tt = 1.0 - ratio * ratio;
                        if tt < 0.0 {
                            tt = 0.0;
                        }
                        if tt.abs() >= 1e-6 {
                            *qx *= tt.sqrt();
                        } else {
                            *qx = dnrm2(&cj[1..]); // dnrm2(x(l+1:n, j))
                            *w1 = *qx;
                        }
                    }
                };
                let qx = &mut qraux[l..p];
                let w1 = &mut work1[l..p];
                // Parallelize only when the trailing block is big enough to beat
                // rayon's dispatch overhead (small fits stay serial).
                const PAR_MIN_ELEMS: usize = 1 << 15;
                if right.len() >= PAR_MIN_ELEMS {
                    right
                        .par_chunks_mut(n)
                        .zip(qx.par_iter_mut())
                        .zip(w1.par_iter_mut())
                        .for_each(|((cj, qx), w1)| update(cj, qx, w1));
                } else {
                    right
                        .chunks_mut(n)
                        .zip(qx.iter_mut())
                        .zip(w1.iter_mut())
                        .for_each(|((cj, qx), w1)| update(cj, qx, w1));
                }
                qraux[l0] = x[col_l + l0];
                x[col_l + l0] = -nrmxl;
            }
        }
    }
    let rank = (k - 1).min(n);
    (qraux, jpvt, rank)
}

/// LINPACK `dqrsl`, `job=1110` path (the only one `dqrls`/`Cdqrls` uses):
/// compute `qty = Qᵀy`, the coefficients `b` (length `rank`, pivoted order) and
/// the residuals `rsd`. `x` is the column-major `dqrdc2` factor (restored on
/// return — the diagonal is temporarily swapped with `qraux`).
fn dqrsl_1110(
    x: &mut [f64],
    n: usize,
    k: usize,
    qraux: &[f64],
    y: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut qty = y.to_vec();
    let mut b = vec![0.0_f64; k];
    let mut rsd = vec![0.0_f64; n];
    let ju = k.min(n.saturating_sub(1));

    if ju == 0 {
        // n == 1 special case (cqty, cb, cr)
        if n >= 1 {
            qty[0] = y[0];
            if k >= 1 {
                if x[0] != 0.0 {
                    b[0] = y[0] / x[0];
                }
                rsd[0] = 0.0;
            }
        }
        return (qty, b, rsd);
    }

    // compute Qᵀy (ascending j)
    for j in 1..=ju {
        let j0 = j - 1;
        if qraux[j0] != 0.0 {
            let col = j0 * n;
            let temp = x[col + j0];
            x[col + j0] = qraux[j0];
            let dot = ddot(&x[col + j0..col + n], &qty[j0..n]);
            let t = -dot / x[col + j0];
            daxpy(t, &x[col + j0..col + n], &mut qty[j0..n]); // x, qty distinct
            x[col + j0] = temp;
        }
    }

    // b <- qty[:k]; rsd[k:n] <- qty[k:n]; rsd[:k] <- 0
    b[..k].copy_from_slice(&qty[..k]);
    if k < n {
        rsd[k..n].copy_from_slice(&qty[k..n]);
    }
    for v in rsd.iter_mut().take(k) {
        *v = 0.0;
    }

    // back-substitute for b (descending)
    for jj in 1..=k {
        let j = k - jj + 1;
        let j0 = j - 1;
        let col = j0 * n;
        if x[col + j0] == 0.0 {
            break; // singular (info=j) — leaves b as is
        }
        b[j0] /= x[col + j0];
        if j != 1 {
            let t = -b[j0];
            for i in 0..j0 {
                b[i] += t * x[col + i]; // daxpy(j-1, t, x(1,j), b)
            }
        }
    }

    // compute rsd (descending)
    for jj in 1..=ju {
        let j0 = (ju - jj + 1) - 1;
        if qraux[j0] != 0.0 {
            let col = j0 * n;
            let temp = x[col + j0];
            x[col + j0] = qraux[j0];
            let dot = ddot(&x[col + j0..col + n], &rsd[j0..n]);
            let t = -dot / x[col + j0];
            daxpy(t, &x[col + j0..col + n], &mut rsd[j0..n]); // x, rsd distinct
            x[col + j0] = temp;
        }
    }
    (qty, b, rsd)
}

/// Full `dqrls` (single rhs): `dqrdc2` then `dqrsl` (job 1110). Returns
/// `(qr_colmajor, coef_p, residuals, effects, rank, pivot_1based, qraux)`.
#[allow(clippy::type_complexity)]
fn dqrls_impl(
    xcol: &mut [f64],
    n: usize,
    p: usize,
    y: &[f64],
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, Vec<usize>, Vec<f64>) {
    let (qraux, jpvt, rank) = dqrdc2(xcol, n, p, tol);
    let mut coef = vec![0.0_f64; p];
    let (qty, rsd) = if rank > 0 {
        let (qty, b, rsd) = dqrsl_1110(xcol, n, rank, &qraux, y);
        coef[..rank].copy_from_slice(&b[..rank]);
        (qty, rsd)
    } else {
        (y.to_vec(), y.to_vec())
    };
    (coef, rsd, qty, rank, jpvt, qraux)
}

/// `Cdqrls`-style entry: weighted-least-squares QR of `x` (n×p) against `y`.
/// Returns `(qr, coefficients, residuals, effects, rank, pivot, qraux)`; `qr` is
/// row-major `(n,p)` (the compact factor), `pivot` is 1-based. Mirrors
/// `hea.R.linalg.Cdqrls` exactly (the pure-Python oracle).
#[pyfunction]
#[pyo3(signature = (x, y, tol=1e-7))]
#[allow(clippy::type_complexity)]
pub fn dqrls<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let xv = x.as_array();
    let (n, p) = (xv.nrows(), xv.ncols());
    let yv = y.as_array();
    let mut xcol = to_colmajor(&xv);
    let yvec: Vec<f64> = yv.iter().copied().collect();

    let (coef, rsd, qty, rank, jpvt, qraux) = dqrls_impl(&mut xcol, n, p, &yvec, tol);

    // qr back to row-major (n,p)
    let mut qr = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        for i in 0..n {
            qr[[i, j]] = xcol[j * n + i];
        }
    }
    let pivot: Vec<i64> = jpvt.iter().map(|&v| v as i64).collect();
    Ok((
        qr.into_pyarray(py),
        PyArray1::from_vec(py, coef),
        PyArray1::from_vec(py, rsd),
        PyArray1::from_vec(py, qty),
        rank,
        PyArray1::from_vec(py, pivot),
        PyArray1::from_vec(py, qraux),
    ))
}

/// Rank + pivot only (R's `dqrdc2`) — for alias detection, which needs neither
/// the coefficients/effects/residuals nor the `n×p` factor. Runs only `dqrdc2`
/// (no `dqrsl`) and marshals back only the rank and the length-p pivot, so it
/// avoids the large-array copies `dqrls` does. `pivot` is 1-based.
#[pyfunction]
#[pyo3(signature = (x, tol=1e-7))]
pub fn dqrls_rank<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    tol: f64,
) -> PyResult<(usize, Bound<'py, PyArray1<i64>>)> {
    let xv = x.as_array();
    let (n, p) = (xv.nrows(), xv.ncols());
    let mut xcol = to_colmajor(&xv);
    let (_qraux, jpvt, rank) = dqrdc2(&mut xcol, n, p, tol);
    let pivot: Vec<i64> = jpvt.iter().map(|&v| v as i64).collect();
    Ok((rank, PyArray1::from_vec(py, pivot)))
}
