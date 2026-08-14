//! `libmetis/mcutil.c` — the one function of it `METIS_NodeND` reaches.
//!
//! The other fourteen are multi-constraint vector predicates (`ivecle`,
//! `BetterVBalance`, …) that only the `ncon > 1` arms call.

use super::graph::Graph;
use super::{Idx, Real};

/// `ComputeLoadImbalanceDiff` (`mcutil.c:...`) — how far past `ubvec` the worst
/// partition is, in `real_t`.
pub fn compute_load_imbalance_diff(
    graph: &Graph,
    nparts: Idx,
    pijbm: &[Real],
    ubvec: &[Real],
) -> Real {
    let ncon = graph.ncon as usize;

    let mut max: Real = -1.0;
    for i in 0..ncon {
        for j in 0..nparts as usize {
            // `pwgts` is idx_t and `pijbm` real_t, so the product is a float.
            let cur = graph.pwgts[j * ncon + i] as Real * pijbm[j * ncon + i] - ubvec[i];
            if cur > max {
                max = cur;
            }
        }
    }
    max
}
