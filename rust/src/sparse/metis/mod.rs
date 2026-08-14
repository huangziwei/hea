//! METIS 5.1.0's nested-dissection ordering, `METIS_NodeND`.
//!
//! Mechanical port of the METIS that SuiteSparse 7.6.0 vendors and CHOLMOD
//! compiles into itself — CHOLMOD's vendored `SuiteSparse_metis`, driven
//! through `CHOLMOD/Partition/cholmod_metis_wrapper.c`. One module per upstream
//! `.c` file, so a citation is a file name plus a line.
//!
//! This fills the METIS slot in `cholmod_analyze`'s ordering trial loop. On a
//! 3.4M-row system where AMD and NATURAL both do badly, METIS gives 380.5M
//! `nnz(L)` against NATURAL's 523.0M — 3.0× on the numeric factorization.
//!
//! Only what `METIS_NodeND` can actually reach is ported. The k-way partitioner
//! (`kwayfm.c`, `kwayrefine.c`, `minconn.c`, `contig.c`) links into upstream's
//! library but no path from `METIS_NodeND` enters it.

pub mod balance;
pub mod bucketsort;
pub mod coarsen;
pub mod compress;
pub mod ctrl;
pub mod fm;
pub mod gklib;
pub mod graph;
pub mod initpart;
pub mod mcutil;
pub mod mmd;
pub mod ometis;
pub mod pqueue;
pub mod refine;
pub mod rng;
pub mod separator;
pub mod sfm;
pub mod srefine;
#[cfg(test)]
mod tests;
pub mod wspace;

pub use ometis::metis_nodend;

/// `idx_t` — `metis.h:34` sets `IDXTYPEWIDTH 64` in the SuiteSparse build, and
/// CHOLMOD relies on it matching its own `Int` so it can hand METIS its column
/// pointers without a copy (`cholmod_metis.c:669-676`).
pub type Idx = i64;

/// `real_t` — `metis.h:44` sets `REALTYPEWIDTH 32`. Used for the load-imbalance
/// predicates. Rebuilding the reference with `-ffp-contract=off` changes no
/// permutation on the gate corpus, so plain non-contracting `f32` is faithful
/// here.
pub type Real = f32;

/// `iabs` — `metis.h:118-123` binds it to `SuiteSparse_metis_abs64`, which
/// takes an `int64_t`.
///
/// Spelled as a function rather than `Idx::abs` because that is what it is: at
/// several call sites the argument is a `real_t` *expression*
/// (`balance.c:16`'s `ntpwgts[0]*tvwgt[0] - pwgts[0]`), so the C truncates it
/// to `int64_t` before taking the absolute value. Reading `iabs` as an
/// arithmetic-on-integers macro there would compute a different number.
#[inline]
pub fn iabs(x: Idx) -> Idx {
    if x < 0 {
        -x
    } else {
        x
    }
}
