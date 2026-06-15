//! Deterministic dense Cholesky — a mechanical port of reference LAPACK
//! `dpotf2` (the unblocked, "Level-2" factorization), lower variant.
//!
//! T4 spike (plan §7.3). The point is *determinism + portability*: a naive
//! in-order accumulation that gives the SAME bits on every platform/run, unlike
//! optimized BLAS (Accelerate/OpenBLAS) whose reduction order is
//! address/SIMD/thread dependent (the source of hea's BLAS-bistable test flakes,
//! [[ill-cond-gam-coef-blas-bistable]] / [[cross-fit-deviance-residual-blas-flake]]).
//!
//! Op order is matched to `dpotf2` EXACTLY so it can be bit-exact to R's
//! `chol()` in the regime where R also runs unblocked (small n, where OpenBLAS's
//! `ddot`/`dgemv` reduce to a plain in-order loop): the DDOT is summed first then
//! subtracted ONCE (`a[j][j] - dot`, not repeated `-=`), and the column scale is
//! a multiply by the reciprocal `1/ajj` (DSCAL), not a divide.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Reference `dpotf2` (lower): overwrite the lower triangle of the row-major
/// `n×n` matrix `a` with `L` s.t. `L Lᵀ == A`. Returns 0 on success, or the
/// 1-based index of the column where a non-positive pivot was found.
pub fn dpotf2_lower(a: &mut [f64], n: usize) -> i32 {
    for j in 0..n {
        // AJJ = A(j,j) - DDOT(j, row_j, row_j)  — sum first, subtract once.
        let mut dot = 0.0;
        for k in 0..j {
            dot += a[j * n + k] * a[j * n + k];
        }
        let ajj = a[j * n + j] - dot;
        if !(ajj > 0.0) {
            a[j * n + j] = ajj;
            return (j + 1) as i32;
        }
        let ajj = ajj.sqrt();
        a[j * n + j] = ajj;
        if j + 1 < n {
            let inv = 1.0 / ajj; // DSCAL multiplies by ONE/AJJ
            for i in (j + 1)..n {
                // DGEMV 'No transpose': temp = sum_k A(i,k)·A(j,k); y -= temp.
                let mut temp = 0.0;
                for k in 0..j {
                    temp += a[i * n + k] * a[j * n + k];
                }
                a[i * n + j] = (a[i * n + j] - temp) * inv;
            }
        }
    }
    0
}

/// Deterministic lower Cholesky of a symmetric `n×n` matrix (only the lower
/// triangle is read). Returns `L` (lower-triangular, upper zeroed) with
/// `L Lᵀ == A`. Raises if not positive-definite.
#[pyfunction]
pub fn chol_lower<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let av = a.as_array();
    let (nr, nc) = (av.nrows(), av.ncols());
    if nr != nc {
        return Err(PyValueError::new_err("chol_lower: matrix must be square"));
    }
    let n = nr;
    let mut buf: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..=i {
            buf[i * n + k] = av[[i, k]];
        }
    }
    let info = dpotf2_lower(&mut buf, n);
    if info != 0 {
        return Err(PyValueError::new_err(format!(
            "chol_lower: not positive definite (pivot {info})"
        )));
    }
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for k in 0..=i {
            out[[i, k]] = buf[i * n + k];
        }
    }
    Ok(out.into_pyarray(py))
}
