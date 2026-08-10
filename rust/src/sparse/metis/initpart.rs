//! `libmetis/initpart.c` — the initial bisection of the coarsest graph.
//!
//! With the default `iptype` (`METIS_IPTYPE_EDGE`) the separator is reached
//! indirectly: bisect by edge cut, then promote the boundary
//! (`separator.c`'s `ConstructSeparator`). `METIS_IPTYPE_NODE` goes straight
//! there through `GrowBisectionNode`. `Init2WayPartition`, the multi-constraint
//! bisections and `GrowBisectionNode2` belong to the k-way and recursive entry
//! points.

use super::super::ws::Ws;
use super::balance::balance_2way;
use super::ctrl::{Ctrl, METIS_IPTYPE_EDGE, METIS_IPTYPE_NODE};
use super::fm::fm_2way_refine;
use super::graph::Graph;
use super::refine::{allocate_2way_partition_memory, compute_2way_partition_params};
use super::separator::construct_separator;
use super::sfm::{fm_2way_node_refine_1sided, fm_2way_node_refine_2sided};
use super::srefine::compute_2way_node_partition_params;
use super::wspace::iwspacemalloc;
use super::{Idx, Real};

/// `InitSeparator` (`initpart.c:...`).
pub fn init_separator(ctrl: &mut Ctrl, graph: &mut Graph, niparts: Idx) {
    let ntpwgts: [Real; 2] = [0.5, 0.5];

    // "this is required for the cut-based part of the refinement"
    let (invtvwgt, ncon) = (graph.invtvwgt.clone(), graph.ncon);
    ctrl.setup_2way_bal_multipliers(&invtvwgt, ncon, &ntpwgts);

    if ctrl.iptype == METIS_IPTYPE_EDGE {
        if graph.nedges == 0 {
            random_bisection(ctrl, graph, &ntpwgts, niparts);
        } else {
            grow_bisection(ctrl, graph, &ntpwgts, niparts);
        }
        compute_2way_partition_params(graph);
        construct_separator(ctrl, graph);
    } else if ctrl.iptype == METIS_IPTYPE_NODE {
        grow_bisection_node(ctrl, graph, &ntpwgts, niparts);
    } else {
        unreachable!("CheckParams rejects any other iptype");
    }
}

/// `RandomBisection` (`initpart.c:...`) — for graphs with no edges at all.
pub fn random_bisection(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2], niparts: Idx) {
    let nvtxs = graph.nvtxs;

    allocate_2way_partition_memory(graph);

    let mut bestwhere = iwspacemalloc(nvtxs);
    let mut perm = iwspacemalloc(nvtxs);

    let zeromaxpwgt = (ctrl.ubfactors[0] * graph.tvwgt[0] as Real * ntpwgts[0]) as Idx;

    let mut bestcut: Idx = 0;
    for inbfs in 0..niparts {
        for w in graph.r#where.iter_mut() {
            *w = 1;
        }

        if inbfs > 0 {
            ctrl.rng.irand_array_permute(nvtxs, &mut perm, nvtxs / 2, 1);
            let mut pwgts = [0 as Idx, graph.tvwgt[0]];

            for ii in 0..nvtxs as usize {
                let i = perm[ii] as usize;
                if pwgts[0] + graph.vwgt[i] < zeromaxpwgt {
                    graph.r#where[i] = 0;
                    pwgts[0] += graph.vwgt[i];
                    pwgts[1] -= graph.vwgt[i];
                    if pwgts[0] > zeromaxpwgt {
                        break;
                    }
                }
            }
        }

        compute_2way_partition_params(graph);
        balance_2way(ctrl, graph, ntpwgts);
        fm_2way_refine(ctrl, graph, ntpwgts, 4);

        if inbfs == 0 || bestcut > graph.mincut {
            bestcut = graph.mincut;
            bestwhere[..nvtxs as usize].copy_from_slice(&graph.r#where[..nvtxs as usize]);
            if bestcut == 0 {
                break;
            }
        }
    }

    graph.mincut = bestcut;
    graph.r#where[..nvtxs as usize].copy_from_slice(&bestwhere[..nvtxs as usize]);
}

/// `GrowBisection` (`initpart.c:...`) — BFS from a random seed until one side
/// is heavy enough, then balance and FM-refine. Repeated `niparts` times.
pub fn grow_bisection(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2], niparts: Idx) {
    let nvtxs = graph.nvtxs;

    allocate_2way_partition_memory(graph);

    let mut bestwhere = iwspacemalloc(nvtxs);
    let mut queue = iwspacemalloc(nvtxs);
    let mut touched = iwspacemalloc(nvtxs);

    // Note the two are computed in different precisions upstream: `onemaxpwgt`
    // is an all-`real_t` chain, `oneminpwgt` starts from the double `1.0`.
    let onemaxpwgt = (ctrl.ubfactors[0] * graph.tvwgt[0] as Real * ntpwgts[1]) as Idx;
    let oneminpwgt =
        ((1.0f64 / ctrl.ubfactors[0] as f64) * graph.tvwgt[0] as f64 * ntpwgts[1] as f64) as Idx;

    let mut bestcut: Idx = 0;
    for inbfs in 0..niparts {
        for w in graph.r#where.iter_mut() {
            *w = 1;
        }
        for t in touched.iter_mut() {
            *t = 0;
        }

        let mut pwgts = [0 as Idx, graph.tvwgt[0]];

        queue[0] = ctrl.rng.irand_in_range(nvtxs);
        touched[queue[0] as usize] = 1;
        let mut first = 0usize;
        let mut last = 1usize;
        let mut nleft = nvtxs - 1;
        let mut drain = false;

        loop {
            if first == last {
                // Empty queue: the graph is disconnected.
                if nleft == 0 || drain {
                    break;
                }

                let mut k = ctrl.rng.irand_in_range(nleft);
                let mut i = 0usize;
                while i < nvtxs as usize {
                    if touched[i] == 0 {
                        if k == 0 {
                            break;
                        }
                        k -= 1;
                    }
                    i += 1;
                }

                queue[0] = i as Idx;
                touched[i] = 1;
                first = 0;
                last = 1;
                nleft -= 1;
            }

            let i = queue[first] as usize;
            first += 1;
            if pwgts[0] > 0 && pwgts[1] - graph.vwgt[i] < oneminpwgt {
                drain = true;
                continue;
            }

            graph.r#where[i] = 0;
            pwgts[0] += graph.vwgt[i];
            pwgts[1] -= graph.vwgt[i];
            if pwgts[1] <= onemaxpwgt {
                break;
            }

            drain = false;
            let (xadj, adjncy) = (Ws::new_ref(&graph.xadj), Ws::new_ref(&graph.adjncy));
            let touched = Ws::new(&mut touched);
            let queue = Ws::new(&mut queue);
            for j in xadj[i]..xadj[i + 1] {
                let k = adjncy[j];
                if touched[k] == 0 {
                    queue[last] = k;
                    last += 1;
                    touched[k] = 1;
                    nleft -= 1;
                }
            }
        }

        // Bad limiting cases.
        if pwgts[1] == 0 {
            let r = ctrl.rng.irand_in_range(nvtxs) as usize;
            graph.r#where[r] = 1;
        }
        if pwgts[0] == 0 {
            let r = ctrl.rng.irand_in_range(nvtxs) as usize;
            graph.r#where[r] = 0;
        }

        compute_2way_partition_params(graph);
        balance_2way(ctrl, graph, ntpwgts);
        fm_2way_refine(ctrl, graph, ntpwgts, ctrl.niter);

        if inbfs == 0 || bestcut > graph.mincut {
            bestcut = graph.mincut;
            bestwhere[..nvtxs as usize].copy_from_slice(&graph.r#where[..nvtxs as usize]);
            if bestcut == 0 {
                break;
            }
        }
    }

    graph.mincut = bestcut;
    graph.r#where[..nvtxs as usize].copy_from_slice(&bestwhere[..nvtxs as usize]);
}

/// `GrowBisectionNode` (`initpart.c:...`) — the `METIS_IPTYPE_NODE` variant,
/// which builds the separator itself instead of going through
/// `ConstructSeparator`.
pub fn grow_bisection_node(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2], niparts: Idx) {
    let nvtxs = graph.nvtxs;

    let mut bestwhere = iwspacemalloc(nvtxs);
    let mut queue = iwspacemalloc(nvtxs);
    let mut touched = iwspacemalloc(nvtxs);

    let onemaxpwgt = (ctrl.ubfactors[0] * graph.tvwgt[0] as Real * 0.5) as Idx;
    let oneminpwgt = ((1.0f64 / ctrl.ubfactors[0] as f64) * graph.tvwgt[0] as f64 * 0.5) as Idx;

    // "Allocate sufficient memory for both edge and node"
    graph.pwgts = vec![0; 3];
    graph.r#where = vec![0; nvtxs as usize];
    graph.bndptr = vec![0; nvtxs as usize];
    graph.bndind = vec![0; nvtxs as usize];
    graph.id = vec![0; nvtxs as usize];
    graph.ed = vec![0; nvtxs as usize];
    graph.nrinfo = vec![Default::default(); nvtxs as usize];

    let mut bestcut: Idx = 0;
    for inbfs in 0..niparts {
        for w in graph.r#where.iter_mut() {
            *w = 1;
        }
        for t in touched.iter_mut() {
            *t = 0;
        }

        let mut pwgts = [0 as Idx, graph.tvwgt[0]];

        queue[0] = ctrl.rng.irand_in_range(nvtxs);
        touched[queue[0] as usize] = 1;
        let mut first = 0usize;
        let mut last = 1usize;
        let mut nleft = nvtxs - 1;
        let mut drain = false;

        loop {
            if first == last {
                if nleft == 0 || drain {
                    break;
                }

                let mut k = ctrl.rng.irand_in_range(nleft);
                let mut i = 0usize;
                while i < nvtxs as usize {
                    if touched[i] == 0 {
                        if k == 0 {
                            break;
                        }
                        k -= 1;
                    }
                    i += 1;
                }

                queue[0] = i as Idx;
                touched[i] = 1;
                first = 0;
                last = 1;
                nleft -= 1;
            }

            let i = queue[first] as usize;
            first += 1;
            if pwgts[1] - graph.vwgt[i] < oneminpwgt {
                drain = true;
                continue;
            }

            graph.r#where[i] = 0;
            pwgts[0] += graph.vwgt[i];
            pwgts[1] -= graph.vwgt[i];
            if pwgts[1] <= onemaxpwgt {
                break;
            }

            drain = false;
            for j in graph.xadj[i] as usize..graph.xadj[i + 1] as usize {
                let k = graph.adjncy[j] as usize;
                if touched[k] == 0 {
                    queue[last] = k as Idx;
                    last += 1;
                    touched[k] = 1;
                    nleft -= 1;
                }
            }
        }

        compute_2way_partition_params(graph);
        balance_2way(ctrl, graph, ntpwgts);
        fm_2way_refine(ctrl, graph, ntpwgts, 4);

        // Construct and refine the vertex separator.
        for i in 0..graph.nbnd as usize {
            let j = graph.bndind[i] as usize;
            if graph.xadj[j + 1] - graph.xadj[j] > 0 {
                graph.r#where[j] = 2;
            }
        }

        compute_2way_node_partition_params(graph);
        fm_2way_node_refine_2sided(ctrl, graph, 1);
        fm_2way_node_refine_1sided(ctrl, graph, 4);

        if inbfs == 0 || bestcut > graph.mincut {
            bestcut = graph.mincut;
            bestwhere[..nvtxs as usize].copy_from_slice(&graph.r#where[..nvtxs as usize]);
        }
    }

    graph.mincut = bestcut;
    graph.r#where[..nvtxs as usize].copy_from_slice(&bestwhere[..nvtxs as usize]);
}
