//! `libmetis/bucketsort.c`.

use super::gklib::makecsr;
use super::wspace::iset_wspace;
use super::Idx;

/// `BucketSortKeysInc` (`bucketsort.c:23-42`) — counting sort of `0..=max`
/// keys, returning a permutation.
///
/// `tperm` supplies the order ties are visited in, so equal keys come out in
/// `tperm` order. `Match_SHEM` feeds it a random `tperm`, which is where the
/// matching's randomization enters.
pub fn bucket_sort_keys_inc(n: Idx, max: Idx, keys: &[Idx], tperm: &[Idx], perm: &mut [Idx]) {
    let mut counts = iset_wspace(max + 2, 0);

    for i in 0..n as usize {
        counts[keys[i] as usize] += 1;
    }
    makecsr((max + 1) as usize, &mut counts);

    for ii in 0..n as usize {
        let i = tperm[ii] as usize;
        perm[counts[keys[i] as usize] as usize] = i as Idx;
        counts[keys[i] as usize] += 1;
    }
}
