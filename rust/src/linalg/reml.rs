//! REML Hessian cross-products — fixed-order dense matmul for the bam
//! `discrete=FALSE` fREML path (`Sl.iftChol`, fast-REML.r:1405-1488).
//!
//! mgcv forms `db'(D+S.db)`, `D'db`, `XX·db` and `db'·XX.db` through
//! `mgcv_pmmult2`→`dgemm` (mat.c:539, 431). On a threaded BLAS
//! (Accelerate/OpenBLAS) the reduction order of those contractions is not
//! pinned, so the REML Hessian wobbles ~1 ULP from run to run and the fREML
//! Newton iteration lands at a slightly different `rho` each fit — the measured
//! source of hea's `discrete=FALSE` run-to-run nondeterminism (numpy `@` here
//! is bistable; mgcv linked against a single-threaded reference BLAS is not).
//!
//! This kernel mirrors `mgcv_mmult` (mat.c:431) — `C = op(a) op(b)` selected by
//! the `bt`/`ct` transpose flags — but accumulates each inner product STRICTLY
//! in `k`-order with separate multiply and add (Rust does not contract `a*b+c`
//! to an FMA, and the sequential `+=` is not reassociated without fast-math), so
//! the result is bit-identical across run AND platform. That is the `linalg`
//! module's determinism contract (cf. `qr::ddot`, `chol`): portable determinism,
//! not 0-ulp to an arbitrary threaded-BLAS-linked R.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// `C = op(a) @ op(b)` where `op` transposes when its flag is set (`at` for
/// `a`, `bt` for `b`) — mgcv `mgcv_mmult` (mat.c:431) `bt`/`ct` semantics.
/// `C` is `m × n` with common dimension `k`; each `C[i,j]` is the dot product
/// over `k` accumulated in index order, so the sum is deterministic and the
/// same on every platform.
#[pyfunction]
#[pyo3(name = "reml_pmmult", signature = (a, b, at, bt))]
pub fn reml_pmmult<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    at: bool,
    bt: bool,
) -> Bound<'py, PyArray2<f64>> {
    let av = a.as_array();
    let bv = b.as_array();
    let (m, ka) = if at {
        (av.ncols(), av.nrows())
    } else {
        (av.nrows(), av.ncols())
    };
    let (kb, n) = if bt {
        (bv.ncols(), bv.nrows())
    } else {
        (bv.nrows(), bv.ncols())
    };
    assert_eq!(ka, kb, "reml_pmmult: inner dimensions disagree");
    let k = ka;
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for kk in 0..k {
                let aik = if at { av[[kk, i]] } else { av[[i, kk]] };
                let bkj = if bt { bv[[j, kk]] } else { bv[[kk, j]] };
                // separate multiply + add, strictly in k-order: no BLAS
                // reduction reorder, no FMA contraction → cross-platform stable.
                acc += aik * bkj;
            }
            c[i * n + j] = acc;
        }
    }
    Array2::from_shape_vec((m, n), c).unwrap().into_pyarray(py)
}
