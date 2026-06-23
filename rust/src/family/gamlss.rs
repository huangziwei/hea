//! `gamlss.gH` coefficient-space Hessian block crossprod — the
//! `crossprod(X_i, (w·l2)·X_j)` inner product that assembles each LP-block of
//! the gamlss/lss Hessian (`hea/family.py::gamlss_gH`, a port of mgcv
//! `gamlss.gH`, gamlss.r:653-660).
//!
//! Why a kernel and not numpy `@`: an optimized BLAS GEMM tiles its output rows
//! independently, so two *bit-identical* input columns (a rank-deficient
//! duplicate covariate) can produce output rows that differ by ~1e-13. That
//! asymmetry flips gam.fit5's end-stage QR rank-check pivot tie
//! (gam.fit4.r:1172) → a different unidentifiable column dropped, *platform
//! dependently* (observed: arm64 vs Intel disagree on which duplicate `gevlss`
//! drops). R's reference path is row/col-consistent here; numpy's `@` is not.
//!
//! This kernel computes `A[r,c] = Σ_k X_i[k,r]·WX_j[k,c]` with a reduction whose
//! structure depends only on `(n, NCHUNKS(n))` — never on `(r,c)` or the thread
//! count — so identical input columns give *bit-identical* output rows/cols by
//! construction, on every platform. It is not bit-equal to numpy `@` / einsum
//! (different summation order) but agrees to ~n·eps and is checked to tolerance
//! + for the row/col property in tests/test_rs_parity.py.
//!
//! Layout contract: caller passes C-contiguous `xi` (n, p_i) and `wxj` (n, p_j)
//! (the latter already `l2_col[:,None]·X_j`). Returns `A` (p_i, p_j). The k-loop
//! is the outer reduction and the `p_i×p_j` tile an inner rank-1 update
//! (`a[r,:] += xi[k,r]·wxj[k,:]`): the tile stays hot in cache and each input
//! row is streamed exactly once (`@`-grade data reuse, unlike a dot-per-output
//! sweep that re-reads an input p times), while the `p_j`-wide axpy
//! auto-vectorises (the `&mut` tile is `noalias` vs the shared `wxj` row).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nmath::util::rfma;

/// Fixed chunk count for the parallel k-reduction: a deterministic function of
/// `n` ONLY (independent of the runtime thread pool), so the summation order —
/// and thus the result — is identical on every machine. ~2048 rows/chunk keeps
/// each chunk's rank-1 sweep cache-resident; capped at 64-way to bound the
/// partial-tile memory and the serial combine.
#[inline]
fn nchunks(n: usize) -> usize {
    (n / 2048).clamp(1, 64).min(n.max(1))
}

/// `a[..] = rfma(x, b[..], a[..])` — the `p_j`-wide rank-1 (axpy) update. `a` is
/// `&mut` (⇒ `noalias`) so LLVM vectorises the fused multiply-adds.
#[inline]
fn axpy(a: &mut [f64], x: f64, b: &[f64]) {
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        *av = rfma(x, bv, *av);
    }
}

/// `A[r,c] = Σ_k xi[k,r]·wxj[k,c]`, row/col-consistent by construction.
fn xwx(xi: &[f64], wxj: &[f64], n: usize, pi: usize, pj: usize) -> Vec<f64> {
    let pij = pi * pj;
    let nc = nchunks(n);
    // Per-chunk partial tiles, each a sequential rank-1 accumulation over its own
    // contiguous k-range; `collect` preserves chunk index order.
    let partials: Vec<Vec<f64>> = (0..nc)
        .into_par_iter()
        .map(|ci| {
            let k0 = ci * n / nc;
            let k1 = (ci + 1) * n / nc;
            let mut a = vec![0.0f64; pij];
            for k in k0..k1 {
                let xr = &xi[k * pi..k * pi + pi];
                let wc = &wxj[k * pj..k * pj + pj];
                for r in 0..pi {
                    axpy(&mut a[r * pj..r * pj + pj], xr[r], wc);
                }
            }
            a
        })
        .collect();
    // Combine partials in fixed chunk-index order (same order for every entry).
    let mut out = vec![0.0f64; pij];
    for p in &partials {
        for (o, &pv) in out.iter_mut().zip(p.iter()) {
            *o += pv;
        }
    }
    out
}

#[pyfunction]
#[pyo3(name = "gamlss_xwx")]
fn gamlss_xwx<'py>(
    py: Python<'py>,
    xi: PyReadonlyArray2<'py, f64>,
    wxj: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let n = xi.shape()[0];
    let pi = xi.shape()[1];
    let pj = wxj.shape()[1];
    // Logical-order flat copies → unit-stride inner loops regardless of layout.
    let xi_f: Vec<f64> = xi.as_array().iter().copied().collect();
    let wxj_f: Vec<f64> = wxj.as_array().iter().copied().collect();
    let a = py.allow_threads(|| xwx(&xi_f, &wxj_f, n, pi, pj));
    Array2::from_shape_vec((pi, pj), a).unwrap().into_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gamlss_xwx, m)?)?;
    Ok(())
}
