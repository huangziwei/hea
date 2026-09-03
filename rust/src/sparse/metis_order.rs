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
/// Per `cholmod_metis.c:719-733`, CHOLMOD returns the identity for anything at
/// or above this order and density rather than calling METIS. The guard fires
/// before `METIS_NodeND` is reached, so it is part of the ordering CHOLMOD
/// produces and has to be mirrored to stay bit-identical.
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
    ncol: usize,
    indptr: &[i64],
    indices: &[i64],
    stype: i32,
) -> Result<(Vec<i64>, f64), CscError> {
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }

    // B = A+A' for a symmetric A, or A*A' for `stype == 0` (`:630-641`), in
    // both cases with the diagonal removed. `cholmod_copy`'s `mode = -1` and
    // `mode = -2` build the identical pattern — they differ only in how much
    // slack `cnzmax` leaves (`t_cholmod_copy.c:248`) — so the symmetric arm is
    // the same construction `cholmod_amd` uses. The unsymmetric arm asks `aat`
    // for `mode = -1`, where `cholmod_amd` asks for `-2`; same reason, same
    // pattern.
    // `copy_sym_to_unsym` returns the *allocated* length, which carries AMD's
    // elbow room; `Bp[n]` is the entry count METIS is given.
    let (bp, bi) = if stype == 0 {
        super::ws::validate_csc_rect(n, ncol, indptr, indices)?;
        let a = super::symbolic::Sparse {
            nrow: n,
            n: ncol,
            p: std::borrow::Cow::Borrowed(indptr),
            i: std::borrow::Cow::Borrowed(indices),
            x: std::borrow::Cow::Borrowed(&[]),
            numeric: false,
            stype: 0,
            sorted: true,
        };
        let b = super::aat::aat(&a, -1);
        (b.p.into_owned(), b.i.into_owned())
    } else {
        let (bp, bi, _nzmax) = copy_sym_to_unsym(n, indptr, indices, stype)?;
        (bp, bi)
    };
    let nz = bp[n] as usize;

    let anz = (nz / 2 + n) as f64;

    let identity = if nz == 0 {
        true
    } else {
        let d = nz as f64 / (n as f64 * n as f64);
        n as i64 > METIS_NSWITCH && d > METIS_DSWITCH
    };

    if identity {
        return Ok(((0..n as i64).collect(), anz));
    }

    let (perm, _iperm) = metis_nodend(n as i64, &bp, &bi).expect(
        "SetupCtrl (METIS_OP_OMETIS, NULL, 1, 3) cannot fail: CheckParams passes on the defaults",
    );
    Ok((perm, anz))
}
