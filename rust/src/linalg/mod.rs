use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod chol;
pub mod pls;
pub mod qr;
pub mod reml;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chol::chol_lower, m)?)?;
    m.add_function(wrap_pyfunction!(qr::dqrls, m)?)?;
    m.add_function(wrap_pyfunction!(qr::dqrls_rank, m)?)?;
    m.add_function(wrap_pyfunction!(pls::pls_fit1, m)?)?;
    m.add_function(wrap_pyfunction!(reml::reml_pmmult, m)?)?;
    Ok(())
}
