use pyo3::prelude::*;

pub mod cutree;
pub mod dist;
pub mod hclust;
pub mod kmeans;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dist::register(m)?;
    hclust::register(m)?;
    kmeans::register(m)?; // Hartigan-Wong + Lloyd + MacQueen
    cutree::register(m)?;
    Ok(())
}
