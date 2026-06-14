//! `hea._native` — the compiled accelerator behind hea's pure-Python numeric
//! kernels.
//!
//! Every function here is a *line-by-line* mirror of the corresponding function
//! in `hea/R/nmath.py`, which is itself a line-by-line mirror of R's
//! `src/nmath/*.c`. The Python module is the spec AND the test oracle: since
//! `python == R` is pinned bit-for-bit (tests/test_R.py), proving
//! `native == python` (tests/test_native_parity.py) transitively proves
//! `native == R` — without needing R in CI.
//!
//! T0 (this spike) ships only `pnorm`. Tier 1 adds the rest of the d/p/q surface.

use pyo3::prelude::*;

mod nmath;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nmath::register(m)?;
    Ok(())
}
