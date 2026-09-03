use super::super::ws::Ws;
use super::ctrl::{Ctrl, METIS_RTYPE_SEP1SIDED, METIS_RTYPE_SEP2SIDED};
use super::graph::{bnd_insert, Graph};
use super::sfm::{fm_2way_node_balance, fm_2way_node_refine_1sided, fm_2way_node_refine_2sided};
use super::Idx;

/// `Refine2WayNode` (`srefine.c:21-61`).
///
/// `base` is `orggraph`; the coarsest level is `levels.len() - 1`. On return the
/// stack has been popped back to `base`.
pub fn refine_2way_node(ctrl: &mut Ctrl, levels: &mut Vec<Graph>, base: usize) {
    if levels.len() - 1 == base {
        compute_2way_node_partition_params(&mut levels[base]);
        return;
    }

    while levels.len() - 1 > base {
        let cur = levels.len() - 2;
        project_2way_node_partition(levels, cur);

        fm_2way_node_balance(ctrl, &mut levels[cur]);

        if ctrl.rtype == METIS_RTYPE_SEP2SIDED {
            fm_2way_node_refine_2sided(ctrl, &mut levels[cur], ctrl.niter);
        } else if ctrl.rtype == METIS_RTYPE_SEP1SIDED {
            fm_2way_node_refine_1sided(ctrl, &mut levels[cur], ctrl.niter);
        } else {
            unreachable!("CheckParams rejects any other rtype");
        }
    }
}

/// `Allocate2WayNodePartitionMemory` (`srefine.c:67-79`).
pub fn allocate_2way_node_partition_memory(graph: &mut Graph) {
    let nvtxs = graph.nvtxs as usize;

    graph.pwgts = vec![0; 3];
    graph.r#where = vec![0; nvtxs];
    graph.bndptr = vec![0; nvtxs];
    graph.bndind = vec![0; nvtxs];
    graph.nrinfo = vec![Default::default(); nvtxs];
}

/// `Compute2WayNodePartitionParams` (`srefine.c:85-131`).
pub fn compute_2way_node_partition_params(graph: &mut Graph) {
    let Graph {
        xadj,
        adjncy,
        vwgt,
        r#where,
        pwgts,
        bndptr,
        bndind,
        nrinfo,
        mincut: g_mincut,
        nbnd: g_nbnd,
        ..
    } = graph;
    let xadj = Ws::new_ref(xadj);
    let adjncy = Ws::new_ref(adjncy);
    let vwgt = Ws::new_ref(vwgt);
    let r#where = Ws::new(r#where);
    let pwgts = Ws::new(pwgts);
    let bndptr = Ws::new(bndptr);
    let bndind = Ws::new(bndind);
    let nvtxs = graph.nvtxs as usize;

    for i in 0..3usize {
        pwgts[i] = 0;
    }
    for i in 0..nvtxs {
        bndptr[i] = -1;
    }

    let mut nbnd: Idx = 0;
    for i in 0..nvtxs {
        let me = r#where[i] as usize;
        pwgts[me] += vwgt[i];

        if me == 2 {
            bnd_insert(&mut nbnd, bndind, bndptr, i);

            let mut ed = [0 as Idx; 2];
            for j in xadj[i] as usize..xadj[i + 1] as usize {
                let other = r#where[adjncy[j] as usize];
                if other != 2 {
                    ed[other as usize] += vwgt[adjncy[j] as usize];
                }
            }
            nrinfo[i].edegrees = ed;
        }
    }

    *g_mincut = pwgts[2];
    *g_nbnd = nbnd;
}

/// `Project2WayNodePartition` (`srefine.c:137-...`) — lift the coarse
/// `where` through `cmap`, then drop the coarse graph.
pub fn project_2way_node_partition(levels: &mut Vec<Graph>, cur: usize) {
    debug_assert_eq!(levels.len(), cur + 2);
    let (fine, coarse) = levels.split_at_mut(cur + 1);
    let graph = &mut fine[cur];
    let cgraph = &coarse[0];

    let nvtxs = graph.nvtxs as usize;
    allocate_2way_node_partition_memory(graph);

    for i in 0..nvtxs {
        graph.r#where[i] = cgraph.r#where[graph.cmap[i] as usize];
    }

    levels.pop();
    compute_2way_node_partition_params(&mut levels[cur]);
}
