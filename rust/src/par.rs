//! Data-parallel array maps for the element-wise nmath kernels.
//!
//! The d/p/q kernels are *element-wise independent* — each output cell depends
//! only on the same-index inputs, with no cross-element reduction. That makes
//! them embarrassingly parallel and, crucially, **order-independent**: a
//! parallel map produces bit-for-bit the same `Vec<f64>` as a serial one, so
//! the 0-ulp parity gate (tests/test_rs_parity.py) stays green.
//!
//! Strategy (plan §"Closing the remaining scipy gap", lever #1):
//!   - len >= `PAR_THRESHOLD` → split across cores with rayon, under
//!     `py.allow_threads` so other Python threads run while we compute.
//!   - len <  `PAR_THRESHOLD` → plain serial map (parallel split/join overhead
//!     isn't worth it). Scalar dispatch arrives as length-1 arrays from
//!     `nmath._disp`, so scalars always take this path — zero added cost.
//!
//! Inputs are contiguous flat `f64` slices: `nmath._disp` / `_norm_rs` pass
//! `np.ascontiguousarray(a.reshape(-1))`, so `PyReadonlyArray1::as_slice()`
//! succeeds at every call site (the wrappers `.unwrap()` it).

use pyo3::prelude::*;
use rayon::prelude::*;

/// Arrays at least this long take the parallel path. The GIL-release + rayon
/// dispatch costs a few µs, so the crossover sits where serial work clears
/// that; one uniform value keeps behavior predictable across kernels and is
/// safely above the per-fit family-hook sizes that arrive as length-1 scalars.
/// Tuned against the benchmark in the plan.
pub const PAR_THRESHOLD: usize = 2048;

/// Map a unary kernel over `a`.
#[inline]
pub fn map1<F>(py: Python<'_>, a: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
        py.allow_threads(|| a.par_iter().map(|&x| f(x)).collect())
    } else {
        a.iter().map(|&x| f(x)).collect()
    }
}

/// Map a binary kernel over `(a, b)` (equal length, pre-broadcast in Python).
#[inline]
pub fn map2<F>(py: Python<'_>, a: &[f64], b: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
        py.allow_threads(|| {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| f(x, y))
                .collect()
        })
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
    }
}

/// Map a ternary kernel over `(a, b, c)`.
#[inline]
pub fn map3<F>(py: Python<'_>, a: &[f64], b: &[f64], c: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64, f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
        py.allow_threads(|| {
            a.par_iter()
                .zip(b.par_iter())
                .zip(c.par_iter())
                .map(|((&x, &y), &z)| f(x, y, z))
                .collect()
        })
    } else {
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((&x, &y), &z)| f(x, y, z))
            .collect()
    }
}

/// Map a quaternary kernel over `(a, b, c, d)`.
#[inline]
pub fn map4<F>(py: Python<'_>, a: &[f64], b: &[f64], c: &[f64], d: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64, f64, f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
        py.allow_threads(|| {
            a.par_iter()
                .zip(b.par_iter())
                .zip(c.par_iter())
                .zip(d.par_iter())
                .map(|(((&x, &y), &z), &w)| f(x, y, z, w))
                .collect()
        })
    } else {
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .zip(d.iter())
            .map(|(((&x, &y), &z), &w)| f(x, y, z, w))
            .collect()
    }
}
