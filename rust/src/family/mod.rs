use pyo3::prelude::*;

pub mod cox;
pub mod gamlss;
pub mod tweedie;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cox::register(m)?;
    gamlss::register(m)?;
    tweedie::register(m)?;
    Ok(())
}
