//! Ports of R's `src/nmath/` probability kernels (mirror of `hea/R/nmath.py`).

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod norm;

/// Register every nmath pyfunction onto the `_native` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(norm::pnorm, m)?)?;
    Ok(())
}
