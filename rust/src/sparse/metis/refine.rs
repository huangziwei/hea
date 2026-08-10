//! `libmetis/refine.c` — the two edge-bisection setup routines.
//!
//! `Refine2Way` and `Project2WayPartition` are the uncoarsening driver for the
//! *edge* objective; `METIS_NodeND` never enters them, because its initial
//! partition is computed on the coarsest graph and it uncoarsens through
//! `Refine2WayNode` (`srefine.c`) instead.

use super::super::ws::Ws;
use super::graph::{bnd_insert, Graph};

/// `Allocate2WayPartitionMemory` (`refine.c:...`).
pub fn allocate_2way_partition_memory(graph: &mut Graph) {
    let nvtxs = graph.nvtxs as usize;
    let ncon = graph.ncon as usize;

    graph.pwgts = vec![0; 2 * ncon];
    graph.r#where = vec![0; nvtxs];
    graph.bndptr = vec![0; nvtxs];
    graph.bndind = vec![0; nvtxs];
    graph.id = vec![0; nvtxs];
    graph.ed = vec![0; nvtxs];
}

/// `Compute2WayPartitionParams` (`refine.c:...`), `ncon == 1`.
pub fn compute_2way_partition_params(graph: &mut Graph) {
    // The C's prologue, taken through `Ws`. Every subscript is one the
    // algorithm produced itself, so the bound is walked in `cargo test` and
    // elided here (`sparse::ws`, `metis::tests`).
    let Graph {
        xadj,
        adjncy,
        adjwgt,
        vwgt,
        r#where,
        id,
        ed,
        pwgts,
        bndptr,
        bndind,
        mincut: g_mincut,
        nbnd: g_nbnd,
        ..
    } = graph;
    let xadj = Ws::new_ref(xadj);
    let adjncy = Ws::new_ref(adjncy);
    let adjwgt = Ws::new_ref(adjwgt);
    let vwgt = Ws::new_ref(vwgt);
    let r#where = Ws::new(r#where);
    let id = Ws::new(id);
    let ed = Ws::new(ed);
    let pwgts = Ws::new(pwgts);
    let bndptr = Ws::new(bndptr);
    let bndind = Ws::new(bndind);
    let nvtxs = graph.nvtxs as usize;
    let ncon = graph.ncon as usize;

    // `iset (2*ncon, 0, graph->pwgts)` — a prefix, not the whole array:
    // `GrowBisectionNode` allocates three slots and only two are the edge
    // bisection's.
    for i in 0..2 * ncon {
        pwgts[i] = 0;
    }
    for i in 0..nvtxs {
        bndptr[i] = -1;
    }

    for i in 0..nvtxs {
        pwgts[r#where[i] as usize] += vwgt[i];
    }

    let mut nbnd = 0;
    let mut mincut = 0;
    for i in 0..nvtxs {
        let istart = xadj[i] as usize;
        let iend = xadj[i + 1] as usize;

        let me = r#where[i];
        let mut tid = 0;
        let mut ted = 0;

        for j in istart..iend {
            if me == r#where[adjncy[j] as usize] {
                tid += adjwgt[j];
            } else {
                ted += adjwgt[j];
            }
        }
        id[i] = tid;
        ed[i] = ted;

        if ted > 0 || istart == iend {
            bnd_insert(&mut nbnd, bndind, bndptr, i);
            mincut += ted;
        }
    }

    *g_mincut = mincut / 2;
    *g_nbnd = nbnd;
}
