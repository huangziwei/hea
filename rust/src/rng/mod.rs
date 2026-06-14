//! R's RNG (Mersenne-Twister `set.seed` stream + nmath `r*` samplers), ported
//! from `hea/R/rng.py` (which mirrors R's `src/main/RNG.c`, `random.c`, and the
//! `src/nmath/r*.c` samplers). Exposes one stateful `#[pyclass] RsMt`.

use pyo3::prelude::*;

pub mod mt;

/// Register the RNG class onto the `_rs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<mt::RsMt>()?;
    Ok(())
}
