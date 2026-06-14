//! Ports of R's `src/nmath/` probability kernels (mirror of `hea/R/nmath.py`).
//!
//! The auto-generated tables (consts.rs, coeffs.rs, bd0_scale.rs) are emitted
//! from the live Python reference (hea/R/nmath.py).

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod bd0_scale;
pub mod coeffs;
pub mod consts;
pub mod discrete;
pub mod exp;
pub mod gamma;
pub mod lgamma;
pub mod loader;
pub mod norm;
pub mod qbeta;
pub mod tf;
pub mod toms708;
pub mod util;

/// Register every nmath pyfunction onto the `_rs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // norm
    m.add_function(wrap_pyfunction!(norm::pnorm, m)?)?;
    m.add_function(wrap_pyfunction!(norm::qnorm, m)?)?;
    m.add_function(wrap_pyfunction!(norm::dnorm, m)?)?;
    // lgamma / gamma function
    m.add_function(wrap_pyfunction!(lgamma::py_lgammafn, m)?)?;
    m.add_function(wrap_pyfunction!(lgamma::py_gammafn, m)?)?;
    // loader saddlepoint kernels
    m.add_function(wrap_pyfunction!(loader::py_stirlerr, m)?)?;
    m.add_function(wrap_pyfunction!(loader::py_bd0, m)?)?;
    m.add_function(wrap_pyfunction!(loader::py_pow1p, m)?)?;
    m.add_function(wrap_pyfunction!(loader::py_dpois_raw, m)?)?;
    m.add_function(wrap_pyfunction!(loader::py_dbinom_raw, m)?)?;
    // gamma family
    m.add_function(wrap_pyfunction!(gamma::pgamma, m)?)?;
    m.add_function(wrap_pyfunction!(gamma::dgamma, m)?)?;
    m.add_function(wrap_pyfunction!(gamma::qgamma, m)?)?;
    // beta (toms708)
    m.add_function(wrap_pyfunction!(toms708::pbeta, m)?)?;
    m.add_function(wrap_pyfunction!(toms708::lbeta, m)?)?;
    m.add_function(wrap_pyfunction!(qbeta::qbeta, m)?)?;
    // discrete CDFs/PMFs/quantiles
    m.add_function(wrap_pyfunction!(discrete::ppois, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::dpois, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::qpois, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::pbinom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::dbinom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::qbinom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::dbeta, m)?)?;
    // t / F
    m.add_function(wrap_pyfunction!(tf::pt, m)?)?;
    m.add_function(wrap_pyfunction!(tf::qt, m)?)?;
    m.add_function(wrap_pyfunction!(tf::dt, m)?)?;
    m.add_function(wrap_pyfunction!(tf::pf, m)?)?;
    m.add_function(wrap_pyfunction!(tf::qf, m)?)?;
    // exponential
    m.add_function(wrap_pyfunction!(exp::dexp, m)?)?;
    m.add_function(wrap_pyfunction!(exp::pexp, m)?)?;
    m.add_function(wrap_pyfunction!(exp::qexp, m)?)?;
    Ok(())
}
