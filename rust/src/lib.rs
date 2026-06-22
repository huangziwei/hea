//! `hea._rs` — the compiled Rust accelerator behind hea's pure-Python numeric
//! kernels.
//!
//! Every function here is a *line-by-line* mirror of the corresponding pure
//! Python kernel (nmath ← `hea/R/nmath.py`, tprs ← `hea/formula.py`, etc.),
//! which is itself a line-by-line mirror of the upstream C/Fortran. The Python
//! module is the spec AND the test oracle: since `python == R` is pinned
//! bit-for-bit (tests/test_R.py), proving `rs == python`
//! (tests/test_rs_parity.py) transitively proves `rs == R` — without R in CI.

// This crate is a *verbatim* mechanical port of upstream C/Fortran/R (R nmath,
// mgcv, lme4). Idiomatic-Rust clippy lints that would make the code DIVERGE from
// its upstream source are allowed crate-wide on purpose: faithfulness to the
// source is the whole point (it keeps re-audits line-by-line and avoids
// transcription drift in a port that must stay 0-ulp to R). Two classes:
//   - source literals: R/C spell out full-precision constants (incl. their own
//     M_LN2/M_SQRT2 etc.), which we copy character-for-character rather than
//     swap for `f64::consts::*` (excessive_precision, approx_constant);
//   - C control flow: NaN-aware `!(x < y)` guards, indexed `for` loops, manual
//     element copies / assign-ops / late inits mirror the source verbatim
//     (neg_cmp_op_on_partial_ord, needless_range_loop, manual_memcpy,
//     assign_op_pattern, needless_late_init, manual_range_contains,
//     mut_range_bound, if_same_then_else, needless_return, redundant_closure,
//     type_complexity, too_many_arguments).
// Correctness is gated by the 0-ulp parity tests (tests/test_rs_parity.py), not
// by idiom. New, non-port code should still be written clippy-clean.
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::assign_op_pattern,
    clippy::needless_late_init,
    clippy::manual_range_contains,
    clippy::mut_range_bound,
    clippy::if_same_then_else,
    clippy::needless_return,
    clippy::redundant_closure,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use pyo3::prelude::*;

mod cluster;
mod family;
mod linalg;
mod loess;
mod nmath;
mod par;
mod rng;
mod tprs;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nmath::register(m)?;
    rng::register(m)?;
    linalg::register(m)?;
    loess::register(m)?;
    cluster::register(m)?;
    tprs::register(m)?;
    family::register(m)?;
    Ok(())
}
