//! Sparse Cholesky — a self-contained replacement for the `scikit-sparse`
//! (CHOLMOD) dependency, so `hea` installs from a wheel on every platform with
//! no system SuiteSparse.
//!
//! Every routine here is a mechanical port of SuiteSparse **7.6.0** — the
//! version R's `Matrix` ships, hence the one `lme4` factorizes with, which is
//! what makes "hea == lme4" a checkable claim.
//!
//! The oracle for these routines is upstream's own C at that tag, compiled and
//! driven directly — *not* `scikit-sparse`. `sksparse`'s `F.perm` is the ordering
//! `cholmod_analyze` stores, which is the fill-reducing permutation already
//! composed with a weighted etree postorder (`cholmod_analyze.c:832-845`), so
//! it is a different quantity from what `cholmod_amd` returns and only
//! coincides when that postorder is the identity.

pub mod amd;
pub mod symbolic;
#[cfg(test)]
pub mod testcorpus;
pub mod ws;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use amd::IntWidth;

/// `cholmod_amd` — the fill-reducing permutation CHOLMOD computes for a
/// symmetric matrix, before any symbolic analysis.
///
/// `indptr`/`indices` are a CSC pattern. `stype` selects the stored half the
/// way CHOLMOD's `A->stype` does: `> 0` upper, `< 0` lower; entries in the
/// other half are ignored rather than folded in. `use_long` picks which C build
/// to reproduce — `hash` is `uint32_t` in the `int32_t` build and `uint64_t` in
/// the `int64_t` one, which is observable only when `hash` overflows.
///
/// Returns `(Perm, info)`, with `Perm[k] = i` if row/column `i` of `A` is the
/// `k`th row/column of `P A P'`. `info` carries AMD's `Info` array under its
/// upstream names, plus the two derived quantities CHOLMOD reads back —
/// `lnz = n + Info[AMD_LNZ]` and `fl = Info[AMD_NDIV] + 2*Info[AMD_NMULTSUBS_LDL]
/// + n` (`cholmod_amd.c:177-180`). Both are slight upper bounds, and both are
/// what `cholmod_analyze`'s ordering trial loop ranks candidate orderings by.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, stype, dense=amd::DEFAULT_DENSE,
                    aggressive=amd::DEFAULT_AGGRESSIVE, use_long=false))]
#[allow(clippy::too_many_arguments)]
fn amd_order(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    stype: i32,
    dense: f64,
    aggressive: bool,
    use_long: bool,
) -> PyResult<(Py<PyArray1<i64>>, Py<PyDict>)> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    if stype == 0 {
        return Err(PyValueError::new_err(
            "stype must be nonzero: cholmod_amd orders a symmetric matrix, and \
             stype selects which triangle is the stored half",
        ));
    }
    let width = if use_long {
        IntWidth::I64
    } else {
        IntWidth::I32
    };
    // `cholmod_amd` validates the pattern itself — that check is what licenses
    // its kernels to index without re-checking, so it cannot be hoisted here.
    let (perm, info) = py
        .allow_threads(|| {
            let mut work = amd::Work::new(n);
            amd::cholmod_amd(
                n,
                indptr,
                indices,
                stype,
                dense,
                aggressive,
                width,
                &mut work.all(),
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("AMD_LNZ", info.lnz)?;
    d.set_item("AMD_NDIV", info.ndiv)?;
    d.set_item("AMD_NMULTSUBS_LDL", info.nms_ldl)?;
    d.set_item("AMD_NMULTSUBS_LU", info.nms_lu)?;
    d.set_item("AMD_NDENSE", info.ndense)?;
    d.set_item("AMD_DMAX", info.dmax)?;
    d.set_item("AMD_NCMPA", info.ncmpa)?;
    d.set_item("anz", info.anz)?;
    d.set_item("lnz", info.lnz(n))?;
    d.set_item("fl", info.fl(n))?;
    Ok((perm.into_pyarray(py).unbind(), d.unbind()))
}

/// `cholmod_analyze` for a symmetric matrix — the fill-reducing ordering, the
/// elimination tree, its weighted postordering, and the column counts of `L`.
///
/// `indptr`/`indices` are a CSC pattern and `stype` selects the stored half the
/// way CHOLMOD's `A->stype` does. `ordering` is `"amd"` or `"natural"`;
/// `default_strategy` says whether `Common->nmethods` was left at its default
/// 0, which is the only setting under which upstream would consider METIS.
///
/// Returns `perm`, `colcount`, `parent` and `post`, all in the final ordering —
/// i.e. with the weighted postorder already composed in, which is what makes
/// `perm` the same quantity as `scikit-sparse`'s `F.perm` (and different from
/// `amd_order`'s output). `fl`/`lnz` are the exact counts from
/// `cholmod_rowcolcounts`, not AMD's estimates; `metis_would_be_tried` reports
/// whether upstream's default strategy would have gone on to try METIS, which
/// is the one place this port can diverge from a CHOLMOD built with the
/// Partition module.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, stype, ordering="amd", default_strategy=true,
                    use_long=false))]
fn analyze(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    stype: i32,
    ordering: &str,
    default_strategy: bool,
    use_long: bool,
) -> PyResult<Py<PyDict>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let order = match ordering {
        "amd" => symbolic::Ordering::Amd,
        "natural" => symbolic::Ordering::Natural,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'amd' or 'natural', not {other:?}"
            )));
        }
    };
    let width = if use_long {
        IntWidth::I64
    } else {
        IntWidth::I32
    };
    let s = py
        .allow_threads(|| {
            symbolic::analyze(n, indptr, indices, stype, order, default_strategy, width)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("perm", s.perm.into_pyarray(py))?;
    d.set_item("colcount", s.colcount.into_pyarray(py))?;
    d.set_item("parent", s.parent.into_pyarray(py))?;
    d.set_item("post", s.post.into_pyarray(py))?;
    d.set_item("fl", s.fl)?;
    d.set_item("lnz", s.lnz)?;
    d.set_item("anz", s.anz)?;
    d.set_item(
        "ordering",
        match s.ordering {
            symbolic::Ordering::Amd => "amd",
            symbolic::Ordering::Natural => "natural",
            symbolic::Ordering::Postordered => "postordered",
        },
    )?;
    d.set_item("metis_would_be_tried", s.metis_would_be_tried)?;
    if let Some(info) = s.amd {
        /* AMD's own estimates, which are what the METIS decision is taken on
         * — deliberately not the exact counts above */
        d.set_item("amd_fl", info.fl(n))?;
        d.set_item("amd_lnz", info.lnz(n))?;
        d.set_item("amd_anz", info.anz)?;
    }
    Ok(d.unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(amd_order, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    Ok(())
}
