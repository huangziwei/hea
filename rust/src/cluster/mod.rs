//! Base-R `stats` clustering + distance kernels — one file per verb:
//!   * `dist`   — pairwise distances (`distance.c`),
//!   * `hclust` — agglomeration + `hcass2` order transform (`hclust.f`),
//!   * `cutree` — flat labels from a cut (`hclust-utils.c`),
//!   * `kmeans` — Hartigan-Wong + Lloyd + MacQueen (`kmns.f` + `cluster_kmeans.c`).
//!
//! All are 1:1 mirrors of the pure-Python spec/oracle in `hea/R/distance.py` /
//! `hea/R/clustering.py`. `dist` parallelizes over independent pairs and `lloyd`
//! over its assignment phase (each unit's float reduction kept serial → parallel
//! == serial bit-for-bit, 0-ulp); `hclust`/`kmns`/MacQueen are inherently serial.

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
