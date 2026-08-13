//! Family-specific kernels for the gam.fit4/fit5 extended families — mirrors of
//! the per-family numpy code in `hea/family.py` (themselves verbatim ports of
//! mgcv's `efam.r` / `coxph.c` / `misc.c`). One submodule per family.
//!
//! Convention note: these are the kernels that are a genuine rust win — those
//! with data-dependent control flow numpy can't vectorize (`cox`'s risk-set
//! sweep) or a large temporary to eliminate (`tweedie`'s `(n, J)` series
//! matrix). Plain element-wise deviance tables (`$Dd`) are NOT here: numpy's
//! vectorized-SIMD `power`/`log` beat a scalar-libm rayon port for those.

use pyo3::prelude::*;

pub mod cox;
pub mod gamlss;
pub mod tweedie;

/// Register every family pyfunction onto the `_rs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cox::register(m)?;
    gamlss::register(m)?;
    tweedie::register(m)?;
    Ok(())
}
