//! `CHOLMOD/Partition/cholmod_metis.c` — CHOLMOD's wrapper around
//! `METIS_NodeND`.
//!
//! It is a thin shell around [`super::metis`]: build `B = A + A'` with both
//! halves and no diagonal, hand METIS the pattern, keep `perm`. Everything else
//! in the file is the three workarounds below and an optional postorder that
//! `cholmod_analyze` never asks for (`cholmod_analyze.c:664` passes
//! `postorder = FALSE`, "because it will be later, below").

use super::amd::copy_sym_to_unsym;
use super::metis::metis_nodend;
use super::ws::CscError;

/// `Common->metis_nswitch` / `metis_dswitch`
/// (`Utility/t_cholmod_defaults.c:51-52`).
///
/// The comment at `cholmod_metis.c:719-733` is worth keeping: METIS 4.0.1 seg
/// faulted on one matrix of order 3005 with 66% density, and the workaround is
/// to return the identity for anything that dense. It has never been retested
/// against 5.1.0, and it fires before `METIS_NodeND` is ever called, so it is
/// part of the ordering CHOLMOD produces whether or not the bug still exists.
const METIS_NSWITCH: i64 = 3000;
const METIS_DSWITCH: f64 = 0.66;

/// `cholmod_metis` (`cholmod_metis.c:557-846`) with `postorder = FALSE`.
///
/// `indptr`/`indices` are a CSC pattern and `stype` selects the stored half the
/// way CHOLMOD's `A->stype` does. Returns `(Perm, anz)`, where `anz` is what
/// upstream leaves in `Common->anz` — `nz/2 + n`, the entries in the lower
/// triangle of `B` including the diagonal, which the trial loop's break check
/// compares against.
pub fn cholmod_metis(
    n: usize,
    indptr: &[i64],
    indices: &[i64],
    stype: i32,
) -> Result<(Vec<i64>, f64), CscError> {
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }

    // B = A+A', upper and lower parts present, no diagonal. `cholmod_copy`'s
    // `mode = -1` and `mode = -2` build the identical pattern — they differ
    // only in how much slack `cnzmax` leaves (`t_cholmod_copy.c:248`) — so this
    // is the same construction `cholmod_amd` uses.
    // `copy_sym_to_unsym` returns the *allocated* length, which carries AMD's
    // elbow room; `Bp[n]` is the entry count METIS is given.
    let (bp, bi, _nzmax) = copy_sym_to_unsym(n, indptr, indices, stype)?;
    let nz = bp[n] as usize;

    let anz = (nz / 2 + n) as f64;

    let identity = if nz == 0 {
        // "The matrix has no off-diagonal entries. METIS_NodeND fails in this
        // case, so avoid using it. The best permutation is identity anyway."
        true
    } else {
        let d = nz as f64 / (n as f64 * n as f64);
        n as i64 > METIS_NSWITCH && d > METIS_DSWITCH
    };
    // `metis_memory_ok` is the third workaround and it is a no-op at the
    // default `Common->metis_memory = 0.0`, which returns TRUE without
    // attempting anything (`cholmod_metis.c:...`).

    if identity {
        return Ok(((0..n as i64).collect(), anz));
    }

    let (perm, _iperm) = metis_nodend(n as i64, &bp, &bi).expect(
        "SetupCtrl (METIS_OP_OMETIS, NULL, 1, 3) cannot fail: CheckParams passes on the defaults",
    );
    Ok((perm, anz))
}
