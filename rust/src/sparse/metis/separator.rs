//! `libmetis/separator.c`.
//!
//! Only `ConstructSeparator` is ported. `ConstructMinCoverSeparator` is the one
//! function in the reachable files with no caller anywhere in METIS 5.1.0.

use super::ctrl::Ctrl;
use super::graph::Graph;
use super::sfm::{fm_2way_node_refine_1sided, fm_2way_node_refine_2sided};
use super::srefine::{allocate_2way_node_partition_memory, compute_2way_node_partition_params};
use crate::sparse::ws::Ws;

/// `ConstructSeparator` (`separator.c:21-...`) — promote the boundary of an
/// edge bisection into a vertex separator, then refine it.
pub fn construct_separator(ctrl: &mut Ctrl, graph: &mut Graph) {
    let nvtxs = graph.nvtxs as usize;
    let nbnd = graph.nbnd as usize;

    let mut r#where = graph.r#where[..nvtxs].to_vec();

    {
        let (xadj, bndind) = (Ws::new_ref(&graph.xadj), Ws::new_ref(&graph.bndind));
        let w = Ws::new(&mut r#where);
        for i in 0..nbnd {
            let j = bndind[i];
            if xadj[j + 1] - xadj[j] > 0 {
                w[j] = 2;
            }
        }
    }

    graph.free_rdata();

    allocate_2way_node_partition_memory(graph);
    graph.r#where[..nvtxs].copy_from_slice(&r#where[..nvtxs]);

    compute_2way_node_partition_params(graph);

    fm_2way_node_refine_2sided(ctrl, graph, 1);
    fm_2way_node_refine_1sided(ctrl, graph, 4);
}
