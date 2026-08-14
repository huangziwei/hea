//! `libmetis/balance.c` — moving vertices across an edge bisection purely to
//! meet the balance constraint, before FM optimises the cut.
//!
//! `McGeneral2WayBalance` is the `ncon > 1` arm and is unreachable here.

use super::super::ws::Ws;
use super::ctrl::Ctrl;
use super::graph::{bnd_delete, bnd_insert, Graph};
use super::mcutil::compute_load_imbalance_diff;
use super::pqueue::rpq_create;
use super::wspace::iwspacemalloc;
use super::{iabs, Idx, Real};

/// `Balance2Way` (`balance.c:16-38`).
pub fn balance_2way(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2]) {
    if compute_load_imbalance_diff(graph, 2, &ctrl.pijbm, &ctrl.ubfactors) <= 0.0 {
        return;
    }

    // `iabs (ntpwgts[0]*graph->tvwgt[0] - graph->pwgts[0])`: the argument is a
    // `real_t` expression and `iabs` takes an `int64_t`, so it truncates first.
    let diff = (graph.tvwgt[0] as Real * ntpwgts[0]) - graph.pwgts[0] as Real;
    if iabs(diff as Idx) < 3 * graph.tvwgt[0] / graph.nvtxs {
        return;
    }

    if graph.nbnd > 0 {
        bnd_2way_balance(ctrl, graph, ntpwgts);
    } else {
        general_2way_balance(ctrl, graph, ntpwgts);
    }
}

/// `Bnd2WayBalance` (`balance.c:44-...`) — only boundary vertices are
/// candidates, which is the common case.
pub fn bnd_2way_balance(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2]) {
    // The C's prologue, taken through `Ws` — `xadj = graph->xadj; ...`. Every
    // subscript below is one the algorithm produced itself, so the bound is
    // walked in `cargo test` and elided here (`sparse::ws`, `metis::tests`).
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
    let nvtxs = graph.nvtxs;

    let mut moved = iwspacemalloc(nvtxs);
    let mut perm = iwspacemalloc(nvtxs);

    let mut tpwgts = [0 as Idx; 2];
    tpwgts[0] = (graph.tvwgt[0] as Real * ntpwgts[0]) as Idx;
    tpwgts[1] = graph.tvwgt[0] - tpwgts[0];
    let mindiff = iabs(tpwgts[0] - pwgts[0]);
    let from = if pwgts[0] < tpwgts[0] { 1usize } else { 0usize };
    let to = (from + 1) % 2;

    let mut queue = rpq_create(nvtxs as usize);

    for m in moved.iter_mut() {
        *m = -1;
    }

    let mut nbnd = *g_nbnd;
    ctrl.rng.irand_array_permute(nbnd, &mut perm, nbnd / 5, 1);
    for ii in 0..nbnd as usize {
        let i = perm[ii] as usize;
        let v = bndind[i] as usize;
        if r#where[v] == from as Idx && vwgt[v] <= mindiff {
            queue.insert(v as Idx, (ed[v] - id[v]) as Real);
        }
    }

    let mut mincut = *g_mincut;
    for nswaps in 0..nvtxs {
        let higain = queue.get_top();
        if higain == -1 {
            break;
        }
        let higain = higain as usize;

        if pwgts[to] + vwgt[higain] > tpwgts[to] {
            break;
        }

        mincut -= ed[higain] - id[higain];
        pwgts[to] += vwgt[higain];
        pwgts[from] -= vwgt[higain];

        r#where[higain] = to as Idx;
        moved[higain] = nswaps;

        // SWAP (id[higain], ed[higain], tmp)
        std::mem::swap(&mut id[higain], &mut ed[higain]);
        if ed[higain] == 0 && xadj[higain] < xadj[higain + 1] {
            bnd_delete(&mut nbnd, bndind, bndptr, higain);
        }

        for j in xadj[higain] as usize..xadj[higain + 1] as usize {
            let k = adjncy[j] as usize;
            let kwgt = if to as Idx == r#where[k] {
                adjwgt[j]
            } else {
                -adjwgt[j]
            };
            id[k] += kwgt;
            ed[k] -= kwgt;

            let eligible = moved[k] == -1 && r#where[k] == from as Idx && vwgt[k] <= mindiff;
            if bndptr[k] != -1 {
                if ed[k] == 0 {
                    bnd_delete(&mut nbnd, bndind, bndptr, k);
                    if eligible {
                        queue.delete(k as Idx);
                    }
                } else if eligible {
                    queue.update(k as Idx, (ed[k] - id[k]) as Real);
                }
            } else if ed[k] > 0 {
                bnd_insert(&mut nbnd, bndind, bndptr, k);
                if eligible {
                    queue.insert(k as Idx, (ed[k] - id[k]) as Real);
                }
            }
        }
    }

    *g_mincut = mincut;
    *g_nbnd = nbnd;
}

/// `General2WayBalance` (`balance.c:...`) — every vertex of the heavy side is a
/// candidate. Reached only when the bisection has no boundary at all.
pub fn general_2way_balance(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2]) {
    // The C's prologue, taken through `Ws` — `xadj = graph->xadj; ...`. Every
    // subscript below is one the algorithm produced itself, so the bound is
    // walked in `cargo test` and elided here (`sparse::ws`, `metis::tests`).
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
    let nvtxs = graph.nvtxs;

    let mut moved = iwspacemalloc(nvtxs);
    let mut perm = iwspacemalloc(nvtxs);

    let mut tpwgts = [0 as Idx; 2];
    tpwgts[0] = (graph.tvwgt[0] as Real * ntpwgts[0]) as Idx;
    tpwgts[1] = graph.tvwgt[0] - tpwgts[0];
    let mindiff = iabs(tpwgts[0] - pwgts[0]);
    let from = if pwgts[0] < tpwgts[0] { 1usize } else { 0usize };
    let to = (from + 1) % 2;

    let mut queue = rpq_create(nvtxs as usize);

    for m in moved.iter_mut() {
        *m = -1;
    }

    ctrl.rng.irand_array_permute(nvtxs, &mut perm, nvtxs / 5, 1);
    for ii in 0..nvtxs as usize {
        let i = perm[ii] as usize;
        if r#where[i] == from as Idx && vwgt[i] <= mindiff {
            queue.insert(i as Idx, (ed[i] - id[i]) as Real);
        }
    }

    let mut mincut = *g_mincut;
    let mut nbnd = *g_nbnd;
    for nswaps in 0..nvtxs {
        let higain = queue.get_top();
        if higain == -1 {
            break;
        }
        let higain = higain as usize;

        if pwgts[to] + vwgt[higain] > tpwgts[to] {
            break;
        }

        mincut -= ed[higain] - id[higain];
        pwgts[to] += vwgt[higain];
        pwgts[from] -= vwgt[higain];

        r#where[higain] = to as Idx;
        moved[higain] = nswaps;

        // SWAP (id[higain], ed[higain], tmp)
        std::mem::swap(&mut id[higain], &mut ed[higain]);
        if ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain + 1] {
            bnd_delete(&mut nbnd, bndind, bndptr, higain);
        }
        if ed[higain] > 0 && bndptr[higain] == -1 {
            bnd_insert(&mut nbnd, bndind, bndptr, higain);
        }

        for j in xadj[higain] as usize..xadj[higain + 1] as usize {
            let k = adjncy[j] as usize;

            let kwgt = if to as Idx == r#where[k] {
                adjwgt[j]
            } else {
                -adjwgt[j]
            };
            id[k] += kwgt;
            ed[k] -= kwgt;

            if moved[k] == -1 && r#where[k] == from as Idx && vwgt[k] <= mindiff {
                queue.update(k as Idx, (ed[k] - id[k]) as Real);
            }

            if ed[k] == 0 && bndptr[k] != -1 {
                bnd_delete(&mut nbnd, bndind, bndptr, k);
            } else if ed[k] > 0 && bndptr[k] == -1 {
                bnd_insert(&mut nbnd, bndind, bndptr, k);
            }
        }
    }

    *g_mincut = mincut;
    *g_nbnd = nbnd;
}
