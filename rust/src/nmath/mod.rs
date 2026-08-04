//! Ports of R's `src/nmath/` probability kernels (mirror of `hea/R/nmath.py`).
//!
//! The auto-generated tables (consts.rs, coeffs.rs, bd0_scale.rs) are emitted
//! from the live Python reference (hea/R/nmath.py).

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod bd0_scale;
pub mod coeffs;
pub mod consts;
pub mod contin;
pub mod discrete;
pub mod exp;
pub mod gamma;
pub mod hyper;
pub mod lgamma;
pub mod loader;
pub mod noncentral;
pub mod norm;
pub mod psigamma;
pub mod qbeta;
pub mod tf;
pub mod toms708;
pub mod tukey;
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
    // psigamma / polygamma
    m.add_function(wrap_pyfunction!(psigamma::psigamma, m)?)?;
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
    m.add_function(wrap_pyfunction!(discrete::pnbinom_mu, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::qnbinom_mu, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::dnbinom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::dnbinom_mu, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::pnbinom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::qnbinom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::dbeta, m)?)?;
    // geometric
    m.add_function(wrap_pyfunction!(discrete::dgeom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::pgeom, m)?)?;
    m.add_function(wrap_pyfunction!(discrete::qgeom, m)?)?;
    // t / F
    m.add_function(wrap_pyfunction!(tf::pt, m)?)?;
    m.add_function(wrap_pyfunction!(tf::qt, m)?)?;
    m.add_function(wrap_pyfunction!(tf::dt, m)?)?;
    m.add_function(wrap_pyfunction!(tf::pf, m)?)?;
    m.add_function(wrap_pyfunction!(tf::qf, m)?)?;
    // noncentral chi-square / t / beta / F
    m.add_function(wrap_pyfunction!(noncentral::pnchisq, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::dnchisq, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::qnchisq, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::pnt, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::dnt, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::qnt, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::pnbeta, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::dnbeta, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::qnbeta, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::pnf, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::dnf, m)?)?;
    m.add_function(wrap_pyfunction!(noncentral::qnf, m)?)?;
    // studentized range (tukey)
    m.add_function(wrap_pyfunction!(tukey::ptukey, m)?)?;
    m.add_function(wrap_pyfunction!(tukey::qtukey, m)?)?;
    // hypergeometric
    m.add_function(wrap_pyfunction!(hyper::dhyper, m)?)?;
    m.add_function(wrap_pyfunction!(hyper::phyper, m)?)?;
    m.add_function(wrap_pyfunction!(hyper::qhyper, m)?)?;
    // exponential
    m.add_function(wrap_pyfunction!(exp::dexp, m)?)?;
    m.add_function(wrap_pyfunction!(exp::pexp, m)?)?;
    m.add_function(wrap_pyfunction!(exp::qexp, m)?)?;
    // cauchy / logistic / log-normal / weibull
    m.add_function(wrap_pyfunction!(contin::dcauchy, m)?)?;
    m.add_function(wrap_pyfunction!(contin::pcauchy, m)?)?;
    m.add_function(wrap_pyfunction!(contin::qcauchy, m)?)?;
    m.add_function(wrap_pyfunction!(contin::dlogis, m)?)?;
    m.add_function(wrap_pyfunction!(contin::plogis, m)?)?;
    m.add_function(wrap_pyfunction!(contin::qlogis, m)?)?;
    m.add_function(wrap_pyfunction!(contin::dlnorm, m)?)?;
    m.add_function(wrap_pyfunction!(contin::plnorm, m)?)?;
    m.add_function(wrap_pyfunction!(contin::qlnorm, m)?)?;
    m.add_function(wrap_pyfunction!(contin::dweibull, m)?)?;
    m.add_function(wrap_pyfunction!(contin::pweibull, m)?)?;
    m.add_function(wrap_pyfunction!(contin::qweibull, m)?)?;
    Ok(())
}
