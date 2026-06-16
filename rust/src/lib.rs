//! `hea._rs` — the compiled Rust accelerator behind hea's pure-Python numeric
//! kernels.
//!
//! Every function here is a *line-by-line* mirror of the corresponding function
//! in `hea/R/nmath.py`, which is itself a line-by-line mirror of R's
//! `src/nmath/*.c`. The Python module is the spec AND the test oracle: since
//! `python == R` is pinned bit-for-bit (tests/test_R.py), proving
//! `rs == python` (tests/test_rs_parity.py) transitively proves `rs == R` —
//! without needing R in CI.

use pyo3::prelude::*;

mod linalg;
mod loess;
mod nmath;
mod par;
mod rng;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nmath::register(m)?;
    rng::register(m)?;
    linalg::register(m)?;
    loess::register(m)?;
    Ok(())
}
