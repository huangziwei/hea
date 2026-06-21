//! Base-R `stats` clustering + distance kernels.
//!
//! 1:1 mirrors of the pure-Python spec/oracle in `hea/R/distance.py` and
//! `hea/R/clustering.py` (themselves ports of R's `src/library/stats/src/`
//! `distance.c` / `hclust.f` / `kmns.f`). `dist` parallelizes over independent
//! pairs (rayon, each pair's column reduction kept serial → 0-ulp); `hclust` and
//! `kmns` are inherently serial NN-chain / transfer loops — never parallelized.

use pyo3::prelude::*;

pub mod dist;
pub mod hclust;
pub mod kmns;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dist::register(m)?;
    hclust::register(m)?;
    kmns::register(m)?;
    Ok(())
}
