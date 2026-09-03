#![allow(unused_assignments)]
#![allow(
    clippy::int_plus_one,
    clippy::nonminimal_bool,
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
mod discrete;
mod family;
mod fexact;
mod linalg;
mod loess;
mod nmath;
mod optimize;
mod par;
mod rng;
mod sparse;
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
    discrete::register(m)?;
    optimize::register(m)?;
    fexact::register(m)?;
    sparse::register(m)?;
    Ok(())
}
