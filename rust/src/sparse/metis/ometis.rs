use super::super::ws::Ws;
use super::coarsen::{coarsen_graph, coarsen_graph_nlevels};
use super::compress::compress_graph;
use super::ctrl::{Ctrl, OpType};
use super::graph::Graph;
use super::initpart::init_separator;
use super::mmd::genmmd;
use super::srefine::{compute_2way_node_partition_params, refine_2way_node};
use super::wspace::iwspacemalloc;
use super::Idx;

/// `defs.h:52` — below this many vertices a subgraph is ordered by minimum
/// degree instead of being dissected further.
const MMDSWITCH: Idx = 120;
/// `defs.h:45`.
const LARGENIPARTS: Idx = 7;

/// `METIS_NodeND` (`ometis.c:43-172`).
///
/// `xadj`/`adjncy` are the symmetric pattern with both halves and no diagonal.
/// Returns `(perm, iperm)` with METIS's own meaning: `A'[i] = A[perm[i]]` and
/// `A[i] = A'[iperm[i]]`.
pub fn metis_nodend(nvtxs: Idx, xadj: &[Idx], adjncy: &[Idx]) -> Option<(Vec<Idx>, Vec<Idx>)> {
    let mut ctrl = Ctrl::setup(OpType::Ometis, None, 1, 3)?;

    let mut perm = vec![0 as Idx; nvtxs as usize];
    let mut iperm = vec![0 as Idx; nvtxs as usize];

    let mut cptr = Vec::new();
    let mut cind = Vec::new();
    let mut graph = None;
    let mut nnvtxs = 0;

    if ctrl.compress != 0 {
        cptr = iwspacemalloc(nvtxs + 1);
        cind = iwspacemalloc(nvtxs);

        graph = compress_graph(nvtxs, xadj, adjncy, None, &mut cptr, &mut cind);
        match &graph {
            None => ctrl.compress = 0,
            Some(g) => {
                nnvtxs = g.nvtxs;
                ctrl.cfactor = (1.0 * nvtxs as f64 / nnvtxs as f64) as super::Real;
                if ctrl.cfactor > 1.5 && ctrl.nseps == 1 {
                    ctrl.nseps = 2;
                }
            }
        }
    }

    let graph = match graph {
        Some(g) => g,
        None => Graph::setup(nvtxs, 1, xadj, adjncy, None),
    };

    if ctrl.ccorder != 0 {
        unimplemented!("ctrl->ccorder is 0 by default and cholmod_metis never sets it");
    }
    let n = graph.nvtxs;
    mlevel_nested_dissection(&mut ctrl, graph, &mut iperm, n);

    if ctrl.compress != 0 {
        for i in 0..nnvtxs as usize {
            perm[iperm[i] as usize] = i as Idx;
        }
        let mut l = 0;
        for ii in 0..nnvtxs as usize {
            let i = perm[ii] as usize;
            for j in cptr[i]..cptr[i + 1] {
                iperm[cind[j as usize] as usize] = l;
                l += 1;
            }
        }
    }

    for i in 0..nvtxs as usize {
        perm[iperm[i] as usize] = i as Idx;
    }

    Some((perm, iperm))
}

/// `MlevelNestedDissection` (`ometis.c:181-219`).
///
/// Takes the graph by value: upstream frees it right after the split, and the
/// two halves recurse independently.
fn mlevel_nested_dissection(ctrl: &mut Ctrl, graph: Graph, order: &mut [Idx], lastvtx: Idx) {
    let mut levels = vec![graph];
    mlevel_node_bisection_multiple(ctrl, &mut levels);
    let graph = levels.pop().unwrap();

    let mut lastvtx = lastvtx;
    for i in 0..graph.nbnd as usize {
        lastvtx -= 1;
        order[graph.label[graph.bndind[i] as usize] as usize] = lastvtx;
    }

    let (lgraph, rgraph) = split_graph_order(&graph);
    drop(graph);

    let rnvtxs = rgraph.nvtxs;
    if lgraph.nvtxs > MMDSWITCH && lgraph.nedges > 0 {
        mlevel_nested_dissection(ctrl, lgraph, order, lastvtx - rnvtxs);
    } else {
        mmd_order(lgraph, order, lastvtx - rnvtxs);
    }
    if rgraph.nvtxs > MMDSWITCH && rgraph.nedges > 0 {
        mlevel_nested_dissection(ctrl, rgraph, order, lastvtx);
    } else {
        mmd_order(rgraph, order, lastvtx);
    }
}

/// `MlevelNodeBisectionMultiple` (`ometis.c:294-338`) — `nseps` independent
/// tri-sections, keeping the best.
fn mlevel_node_bisection_multiple(ctrl: &mut Ctrl, levels: &mut Vec<Graph>) {
    let small = if ctrl.compress != 0 { 1000 } else { 2000 };
    if ctrl.nseps == 1 || levels[0].nvtxs < small {
        mlevel_node_bisection_l2(ctrl, levels, 0, LARGENIPARTS);
        return;
    }

    let nvtxs = levels[0].nvtxs as usize;
    let mut bestwhere = iwspacemalloc(levels[0].nvtxs);

    let mut mincut = levels[0].tvwgt[0];
    for i in 0..ctrl.nseps {
        mlevel_node_bisection_l2(ctrl, levels, 0, LARGENIPARTS);

        if i == 0 || levels[0].mincut < mincut {
            mincut = levels[0].mincut;
            if i < ctrl.nseps - 1 {
                bestwhere[..nvtxs].copy_from_slice(&levels[0].r#where[..nvtxs]);
            }
        }

        if mincut == 0 {
            break;
        }

        if i < ctrl.nseps - 1 {
            levels[0].free_rdata();
        }
    }

    if mincut != levels[0].mincut {
        levels[0].r#where[..nvtxs].copy_from_slice(&bestwhere[..nvtxs]);
        compute_2way_node_partition_params(&mut levels[0]);
    }
}

/// `MlevelNodeBisectionL2` (`ometis.c:345-388`) — coarsen four levels, run L1
/// five times on the result, then uncoarsen the winner all the way back.
fn mlevel_node_bisection_l2(ctrl: &mut Ctrl, levels: &mut Vec<Graph>, base: usize, niparts: Idx) {
    if levels[base].nvtxs < 5000 {
        mlevel_node_bisection_l1(ctrl, levels, base, niparts);
        return;
    }

    let nruns = 5;
    ctrl.coarsen_to = Idx::max(100, levels[base].nvtxs / 30);

    let c = coarsen_graph_nlevels(ctrl, levels, base, 4);

    let cnvtxs = levels[c].nvtxs as usize;
    let mut bestwhere = iwspacemalloc(levels[c].nvtxs);

    let mut mincut = levels[base].tvwgt[0];
    for i in 0..nruns {
        mlevel_node_bisection_l1(ctrl, levels, c, (0.7 * niparts as f64) as Idx);

        if i == 0 || levels[c].mincut < mincut {
            mincut = levels[c].mincut;
            if i < nruns - 1 {
                bestwhere[..cnvtxs].copy_from_slice(&levels[c].r#where[..cnvtxs]);
            }
        }

        if mincut == 0 {
            break;
        }

        if i < nruns - 1 {
            levels[c].free_rdata();
        }
    }

    if mincut != levels[c].mincut {
        levels[c].r#where[..cnvtxs].copy_from_slice(&bestwhere[..cnvtxs]);
    }

    refine_2way_node(ctrl, levels, base);
}

/// `MlevelNodeBisectionL1` (`ometis.c:394-410`) — one full multilevel
/// tri-section.
fn mlevel_node_bisection_l1(ctrl: &mut Ctrl, levels: &mut Vec<Graph>, base: usize, niparts: Idx) {
    ctrl.coarsen_to = (levels[base].nvtxs / 8).clamp(40, 100);

    let c = coarsen_graph(ctrl, levels, base);

    let niparts = Idx::max(
        1,
        if levels[c].nvtxs <= ctrl.coarsen_to {
            niparts / 2
        } else {
            niparts
        },
    );
    init_separator(ctrl, &mut levels[c], niparts);

    refine_2way_node(ctrl, levels, base);
}

/// `SplitGraphOrder` (`ometis.c:421-529`) — split a tri-sected graph into its
/// left and right halves, dropping the separator.
///
/// "This function relies on the fact that adjwgt is all equal to 1."
fn split_graph_order(graph: &Graph) -> (Graph, Graph) {
    let nvtxs = graph.nvtxs as usize;

    let xadj = Ws::new_ref(&graph.xadj);
    let adjncy = Ws::new_ref(&graph.adjncy);
    let vwgt = Ws::new_ref(&graph.vwgt);
    let label = Ws::new_ref(&graph.label);
    let g_where = Ws::new_ref(&graph.r#where);
    let bndind = Ws::new_ref(&graph.bndind);

    let mut rename_ = iwspacemalloc(graph.nvtxs);
    let rename = Ws::new(&mut rename_);

    let mut snvtxs = [0 as Idx; 3];
    let mut snedges = [0 as Idx; 3];
    for i in 0..nvtxs {
        let k = g_where[i] as usize;
        rename[i] = snvtxs[k];
        snvtxs[k] += 1;
        snedges[k] += xadj[i + 1] - xadj[i];
    }

    let mut sgraph = [
        graph.setup_split(snvtxs[0], snedges[0]),
        graph.setup_split(snvtxs[1], snedges[1]),
    ];

    let mut bndptr_ = graph.bndptr[..nvtxs].to_vec();
    let bndptr = Ws::new(&mut bndptr_);
    for ii in 0..graph.nbnd as usize {
        let i = bndind[ii];
        for j in xadj[i]..xadj[i + 1] {
            bndptr[adjncy[j]] = 1;
        }
    }

    let mut snvtxs = [0usize; 2];
    let mut snedges = [0usize; 2];
    sgraph[0].xadj[0] = 0;
    sgraph[1].xadj[0] = 0;
    for i in 0..nvtxs {
        let mypart = g_where[i];
        if mypart == 2 {
            continue;
        }
        let mypart = mypart as usize;

        let istart = xadj[i] as usize;
        let iend = xadj[i + 1] as usize;
        if bndptr[i] == -1 {
            let s = &mut sgraph[mypart];
            s.adjncy[snedges[mypart]..snedges[mypart] + (iend - istart)]
                .copy_from_slice(&graph.adjncy[istart..iend]);
            snedges[mypart] += iend - istart;
        } else {
            let sadjncy = Ws::new(&mut sgraph[mypart].adjncy);
            let mut l = snedges[mypart];
            for j in istart..iend {
                let k = adjncy[j];
                if g_where[k] == mypart as Idx {
                    sadjncy[l] = k;
                    l += 1;
                }
            }
            snedges[mypart] = l;
        }

        let s = &mut sgraph[mypart];
        s.vwgt[snvtxs[mypart]] = vwgt[i];
        s.label[snvtxs[mypart]] = label[i];
        snvtxs[mypart] += 1;
        s.xadj[snvtxs[mypart]] = snedges[mypart] as Idx;
    }

    for mypart in 0..2 {
        let iend = snedges[mypart];
        let s = &mut sgraph[mypart];
        for a in s.adjwgt.iter_mut().take(iend) {
            *a = 1;
        }
        let sadjncy = Ws::new(&mut s.adjncy);
        for i in 0..iend {
            sadjncy[i] = rename[sadjncy[i]];
        }
    }

    let [mut lgraph, mut rgraph] = sgraph;
    lgraph.nvtxs = snvtxs[0] as Idx;
    lgraph.nedges = snedges[0] as Idx;
    rgraph.nvtxs = snvtxs[1] as Idx;
    rgraph.nedges = snedges[1] as Idx;

    lgraph.setup_tvwgt();
    rgraph.setup_tvwgt();

    (lgraph, rgraph)
}

/// `MMDOrder` (`ometis.c:654-701`) — order a leaf subgraph by minimum degree,
/// numbering downwards from `lastvtx`.
///
/// Upstream shifts `graph->xadj`/`adjncy` to 1-based in place and shifts them
/// back afterwards, because `genmmd` is Fortran-derived; here the shifted copies
/// are built explicitly, which is also what lets `genmmd` destroy `adjncy`
/// without touching the caller's.
fn mmd_order(graph: Graph, order: &mut [Idx], lastvtx: Idx) {
    let nvtxs = graph.nvtxs;
    let n = nvtxs as usize;

    let mut xadj1 = vec![0 as Idx; n + 2];
    for i in 0..=n {
        xadj1[i + 1] = graph.xadj[i] + 1;
    }
    let nedges = graph.xadj[n] as usize;
    let mut adjncy1 = vec![0 as Idx; nedges + 1];
    for j in 0..nedges {
        adjncy1[j + 1] = graph.adjncy[j] + 1;
    }

    let w = (nvtxs + 5) as usize + 1;
    let mut perm = vec![0 as Idx; w];
    let mut iperm = vec![0 as Idx; w];
    let mut head = vec![0 as Idx; w];
    let mut qsize = vec![0 as Idx; w];
    let mut list = vec![0 as Idx; w];
    let mut marker = vec![0 as Idx; w];
    let mut nofsub: Idx = 0;

    genmmd(
        nvtxs,
        Ws::new_ref(&xadj1),
        Ws::new(&mut adjncy1),
        Ws::new(&mut iperm),
        Ws::new(&mut perm),
        1,
        Ws::new(&mut head),
        Ws::new(&mut qsize),
        Ws::new(&mut list),
        Ws::new(&mut marker),
        Idx::MAX,
        &mut nofsub,
    );

    let firstvtx = lastvtx - nvtxs;
    for i in 0..n {
        order[graph.label[i] as usize] = firstvtx + iperm[i + 1] - 1;
    }
}
