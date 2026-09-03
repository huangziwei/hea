//! R's RNG (Mersenne-Twister `set.seed` stream + nmath `r*` samplers), ported
//! from `hea/R/rng.py` (which mirrors R's `src/main/RNG.c`, `random.c`, and the
//! `src/nmath/r*.c` samplers). Exposes one stateful `#[pyclass] RsMt`.

use pyo3::prelude::*;

pub mod mt;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<mt::RsMt>()?;
    Ok(())
}
