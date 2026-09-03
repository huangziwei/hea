use super::graph::Graph;
use super::{Idx, Real};

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
            let cur = graph.pwgts[j * ncon + i] as Real * pijbm[j * ncon + i] - ubvec[i];
            if cur > max {
                max = cur;
            }
        }
    }
    max
}
