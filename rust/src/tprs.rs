//! Thin-plate regression spline kernel evaluation (`s(bs="tp")`, the default).
//!
//! Mirrors mgcv's `XBuild` (src/tprs.c:560, the knots≠data path): for each data
//! row `x_i`, build the length-(nu+M) vector
//!     b = [ η(‖x_i − Xu_j‖²)  for j in 0..nu | ∏_k x_i,k^{p_{l,k}}  for l in 0..M ]
//! which the caller multiplies by `UZ` to get the design row. This is the
//! n-dependent hot path (a √ over the n×nu radial table); rayon parallelises it
//! over rows — which numpy/Accelerate can't, their ufuncs are single-threaded.
//!
//! Bit-exactness: the fill is element-wise independent, so the parallel fill is
//! byte-for-byte the serial one; and the per-element arithmetic mirrors
//! `hea/formula.py::_tp_eval_X_raw` / `_tp_T` exactly, so `b` equals the numpy
//! build. The caller keeps the BLAS `b @ UZ` matmul, so the design is identical.
//!
//! FMA/contraction parity with arm64 R: tprs.c is compiled `clang -O2`, which
//! fuses single-expression `a*b+c` to `fmadd` where the ISA has baseline FMA
//! (aarch64 yes, x86-64 no). So the squared-distance accumulate `z=a-b; r+=z*z;`
//! (tprs.c:92 tpsE, :591 XBuild) is `fma(z,z,r)` on arm64 — mirrored here via
//! `rfma` (per-arch). And the null-space monomial is built by *repeated multiply*
//! `r=1; for(kk<pin) r*=xx[j];` (tprs.c:156 tpsT, :598 XBuild) — NOT `powi`, which
//! reassociates at powers ≥4 even on x86. Both fixes keep `b` 0-ulp to the numpy
//! `_rfma_vec`-fold build on every platform.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nmath::util::rfma;

/// `fast_eta` (tprs.c:61) — η given r² in `rsq`, constant in `f0`. Same
/// multiplication order as `_tp_fast_eta_vec`: d-odd is maskless (√0 = 0 gives
/// the exact 0), d-even guards log(0) via the r²≤0 early return.
#[inline]
#[allow(clippy::manual_is_multiple_of)] // mirror C/Python `d % 2 == 0` verbatim
fn fast_eta(m: usize, d: usize, rsq: f64, f0: f64) -> f64 {
    if rsq <= 0.0 {
        return 0.0;
    }
    let d2 = d / 2;
    let mut f = f0;
    if d % 2 == 0 {
        f *= rsq.ln() * 0.5;
        for _ in 0..(m - d2) {
            f *= rsq;
        }
    } else {
        for _ in 0..(m - d2 - 1) {
            f *= rsq;
        }
        f *= rsq.sqrt();
    }
    f
}

/// Build the (n, nu+M) radial+polynomial matrix `b` for the tp kernel-eval path.
/// `x_c` is (n, d) centred covariates, `xu` the (nu, d) knot grid, `poly_powers`
/// the (M, d) null-space exponents (`_tp_gen_poly_powers`). Returns `b`; the
/// caller does `b @ UZ`.
#[pyfunction]
#[pyo3(name = "tp_eval_b")]
fn tp_eval_b<'py>(
    py: Python<'py>,
    x_c: PyReadonlyArray2<'py, f64>,
    xu: PyReadonlyArray2<'py, f64>,
    m: usize,
    d: usize,
    eta0: f64,
    poly_powers: PyReadonlyArray2<'py, i64>,
) -> Bound<'py, PyArray2<f64>> {
    let x = x_c.as_array();
    let u = xu.as_array();
    let pp = poly_powers.as_array();
    let n = x.shape()[0];
    let nu = u.shape()[0];
    let mm = pp.shape()[0]; // M = null-space polynomial columns
    let ncol = nu + mm;

    // Logical-order (row-major) flat copies → contiguous inner loops regardless
    // of the caller's array layout (callers pass ascontiguousarray anyway).
    let x_flat: Vec<f64> = x.iter().copied().collect(); // (n, d)
    let xu_flat: Vec<f64> = u.iter().copied().collect(); // (nu, d)
    let pp_flat: Vec<i64> = pp.iter().copied().collect(); // (M, d)

    let mut out = vec![0.0f64; n * ncol];
    let fill = |i: usize, row: &mut [f64]| {
        let xi = &x_flat[i * d..i * d + d];
        for j in 0..nu {
            let uj = &xu_flat[j * d..j * d + d];
            let mut r = 0.0;
            for k in 0..d {
                let z = xi[k] - uj[k];
                r = rfma(z, z, r); // tprs.c:591 `z=a-b; r+=z*z;` → fmadd on arm64
            }
            row[j] = fast_eta(m, d, r, eta0);
        }
        for l in 0..mm {
            // null-space monomial ∏_k x_k^{pin[l,k]} by repeated multiply, exactly
            // as tprs.c:598 `r=1; for(j) for(kk<pin) r*=xx[j];` (NOT powi).
            let mut t = 1.0;
            for k in 0..d {
                let pk = pp_flat[l * d + k];
                for _ in 0..pk {
                    t *= xi[k];
                }
            }
            row[nu + l] = t;
        }
    };

    // Element-wise independent ⇒ parallel == serial bit-for-bit. Small n stays
    // serial (rayon split/join isn't worth it).
    if n >= 256 {
        py.allow_threads(|| {
            out.par_chunks_mut(ncol)
                .enumerate()
                .for_each(|(i, row)| fill(i, row));
        });
    } else {
        out.chunks_mut(ncol)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    Array2::from_shape_vec((n, ncol), out)
        .unwrap()
        .into_pyarray(py)
}

/// `tpsE` (tprs.c:76) — the (nu, nu) penalty/radial matrix on the knot grid,
/// `E_ij = η(‖Xu_i − Xu_j‖²)`. The n-independent knot-side cost (a √ over the
/// nu² table) that floors basis time at small n. Rayon over rows; element-wise
/// independent ⇒ bit-identical to the numpy `_tp_E`. Computes the full matrix
/// (each row independent for safe parallel writes); the diagonal is 0 (r²=0).
#[pyfunction]
#[pyo3(name = "tp_eval_E")] // Python name keeps the math-symbol E (cf. `_tp_E`)
fn tp_eval_e<'py>(
    py: Python<'py>,
    xu: PyReadonlyArray2<'py, f64>,
    m: usize,
    d: usize,
    eta0: f64,
) -> Bound<'py, PyArray2<f64>> {
    let u = xu.as_array();
    let nu = u.shape()[0];
    let xu_flat: Vec<f64> = u.iter().copied().collect(); // (nu, d) row-major
    let mut out = vec![0.0f64; nu * nu];
    let fill = |i: usize, row: &mut [f64]| {
        let xi = &xu_flat[i * d..i * d + d];
        for j in 0..nu {
            let uj = &xu_flat[j * d..j * d + d];
            let mut r = 0.0;
            for k in 0..d {
                let z = xi[k] - uj[k];
                r = rfma(z, z, r); // tprs.c:92 `x=a-b; r+=x*x;` → fmadd on arm64
            }
            row[j] = fast_eta(m, d, r, eta0);
        }
    };
    if nu >= 256 {
        py.allow_threads(|| {
            out.par_chunks_mut(nu)
                .enumerate()
                .for_each(|(i, row)| fill(i, row));
        });
    } else {
        out.chunks_mut(nu)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    Array2::from_shape_vec((nu, nu), out)
        .unwrap()
        .into_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tp_eval_b, m)?)?;
    m.add_function(wrap_pyfunction!(tp_eval_e, m)?)?;
    Ok(())
}
