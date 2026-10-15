//! `libmetis/wspace.c` — where METIS's scratch arrays come from.
//!
//! Upstream keeps one bump-allocated core inside `ctrl_t` (`gk_mcoreCreate`,
//! `gk_mcoreMalloc`) with a push/pop stack: `WCOREPUSH` marks, `iwspacemalloc`
//! carves, `WCOREPOP` releases everything since the mark. It is an allocator,
//! not an algorithm — and the port measured that nothing depends on what it
//! hands back before it is written: upstream driven with five different fill
//! bytes in place of the bump core gives an identical permutation on all 23
//! corpus matrices.
//!
//! So the port carves scratch as owned `Vec`s at the same points, and Rust's
//! scopes are `WCOREPUSH`/`WCOREPOP`. That is the one mechanical deviation in
//! this module, and it is what keeps `ctrl` from being mutably borrowed by its
//! own workspace for the lifetime of every kernel.
//!
//! The two differences it introduces are both in the safe direction: the memory
//! comes back zeroed rather than uninitialized, and a large request cannot
//! silently fall out of the core into `gk_malloc` because there is no core to
//! overflow.
//!
//! `AllocateRefinementWorkSpace`, `cnbrpool*` and `vnbrpool*` are k-way-only and
//! not ported; `METIS_NodeND` has no path to them.

use super::gklib::Ikv;
use super::{Idx, Real};

/// `iwspacemalloc` (`wspace.c:134-137`).
#[inline]
pub fn iwspacemalloc(n: Idx) -> Vec<Idx> {
    vec![0; n.max(0) as usize]
}

/// `rwspacemalloc` (`wspace.c:143-146`).
#[inline]
#[allow(dead_code)]
pub fn rwspacemalloc(n: Idx) -> Vec<Real> {
    vec![0.0; n.max(0) as usize]
}

/// `ikvwspacemalloc` (`wspace.c:152-155`).
#[inline]
pub fn ikvwspacemalloc(n: Idx) -> Vec<Ikv> {
    vec![Ikv::default(); n.max(0) as usize]
}

#[inline]
pub fn iset_wspace(n: Idx, val: Idx) -> Vec<Idx> {
    vec![val; n.max(0) as usize]
}
