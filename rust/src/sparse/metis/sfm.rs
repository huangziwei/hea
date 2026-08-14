//! `libmetis/sfm.c` — FM refinement of a *vertex separator*, the kernel this
//! whole module exists to run. 99% of it executes on every ordering.
//!
//! `where[i]` is 0, 1 or 2, with 2 meaning "in the separator"; `pwgts[2]` is
//! therefore the separator weight and also the objective. `rinfo[i].edegrees[s]`
//! is the weight vertex `i` has on side `s`, maintained incrementally.
//!
//! `moved[k]` carries three kinds of value in the two-sided kernel: `-1` for
//! untouched, `nswaps >= 0` for moved this pass, and `-(2 + side)` for "pulled
//! into the separator and queued on `side` only". Reading it as a plain
//! moved/not-moved flag silently drops half the queue updates.

use super::super::ws::Ws;
use super::ctrl::Ctrl;
use super::graph::{bnd_delete, bnd_insert, Graph, NrInfo};
use super::pqueue::rpq_create;
use super::wspace::iwspacemalloc;
use super::{iabs, Idx, Real};

/// `FM_2WayNodeRefine2Sided` (`sfm.c:21-252`).
pub fn fm_2way_node_refine_2sided(ctrl: &mut Ctrl, graph: &mut Graph, niter: Idx) {
    // The C's prologue, taken through `Ws`: `xadj = graph->xadj; adjncy =
    // graph->adjncy; ...`. Every subscript below is one the algorithm produced
    // itself — an `adjncy` entry, a `bndind` slot, a `where` value in 0..=2 —
    // so the bound is walked in `cargo test` and elided here (`sparse::ws`,
    // `metis::tests`).
    let Graph {
        xadj,
        adjncy,
        vwgt,
        bndind,
        bndptr,
        r#where,
        pwgts,
        nrinfo,
        mincut: g_mincut,
        nbnd: g_nbnd,
        ..
    } = graph;
    let xadj = Ws::new_ref(xadj);
    let adjncy = Ws::new_ref(adjncy);
    let vwgt = Ws::new_ref(vwgt);
    let bndind = Ws::new(bndind);
    let bndptr = Ws::new(bndptr);
    let r#where = Ws::new(r#where);
    let pwgts = Ws::new(pwgts);
    let rinfo: &mut [NrInfo] = nrinfo;
    let nvtxs = graph.nvtxs;

    let mut queues = [rpq_create(nvtxs as usize), rpq_create(nvtxs as usize)];

    let mut moved = iwspacemalloc(nvtxs);
    let mut swaps = iwspacemalloc(nvtxs);
    let mut mptr = iwspacemalloc(nvtxs + 1);
    let mut mind = iwspacemalloc(2 * nvtxs);

    // `mult = 0.5*ctrl->ubfactors[0]` is a double expression stored into a
    // `real_t`; the product with the idx_t sum below is then a float, before
    // the explicit truncation back to `idx_t`.
    let mult: Real = (0.5 * ctrl.ubfactors[0] as f64) as Real;
    let badmaxpwgt = (mult * (pwgts[0] + pwgts[1] + pwgts[2]) as Real) as Idx;

    for pass in 0..niter {
        for m in moved.iter_mut() {
            *m = -1;
        }
        queues[0].reset();
        queues[1].reset();

        let mut mincutorder: Idx = -1;
        let initcut = *g_mincut;
        let mut mincut = *g_mincut;
        let mut nbnd = *g_nbnd;

        // "use the swaps array in place of the traditional perm array"
        ctrl.rng.irand_array_permute(nbnd, &mut swaps, nbnd, 1);
        for ii in 0..nbnd as usize {
            let i = bndind[swaps[ii] as usize] as usize;
            queues[0].insert(i as Idx, (vwgt[i] - rinfo[i].edegrees[1]) as Real);
            queues[1].insert(i as Idx, (vwgt[i] - rinfo[i].edegrees[0]) as Real);
        }

        let limit = if ctrl.compress != 0 {
            Idx::min(5 * nbnd, 400)
        } else {
            Idx::min(2 * nbnd, 300)
        };

        mptr[0] = 0;
        let mut nmind: Idx = 0;
        let mut mindiff = iabs(pwgts[0] - pwgts[1]);
        // Every path that reaches a move reassigns `to` from the queue tops,
        // so this initial value is dead here as it is upstream; it is kept
        // because it is what `sfm.c:87` writes.
        #[allow(unused_assignments)]
        let mut to = if pwgts[0] < pwgts[1] { 0usize } else { 1usize };

        let mut nswaps: Idx = 0;
        while nswaps < nvtxs {
            let u = [queues[0].see_top_val(), queues[1].see_top_val()];
            if u[0] != -1 && u[1] != -1 {
                let g = [
                    vwgt[u[0] as usize] - rinfo[u[0] as usize].edegrees[1],
                    vwgt[u[1] as usize] - rinfo[u[1] as usize].edegrees[0],
                ];

                to = if g[0] > g[1] {
                    0
                } else if g[0] < g[1] {
                    1
                } else {
                    (pass % 2) as usize
                };

                if pwgts[to] + vwgt[u[to] as usize] > badmaxpwgt {
                    to = (to + 1) % 2;
                }
            } else if u[0] == -1 && u[1] == -1 {
                break;
            } else if u[0] != -1 && pwgts[0] + vwgt[u[0] as usize] <= badmaxpwgt {
                to = 0;
            } else if u[1] != -1 && pwgts[1] + vwgt[u[1] as usize] <= badmaxpwgt {
                to = 1;
            } else {
                break;
            }

            let other = (to + 1) % 2;

            let higain = queues[to].get_top() as usize;
            if moved[higain] == -1 {
                // it was in the separator originally, so it is on both queues
                queues[other].delete(higain as Idx);
            }

            // guard against over-running `mind`
            if nmind + xadj[higain + 1] - xadj[higain] >= 2 * nvtxs - 1 {
                break;
            }

            pwgts[2] -= vwgt[higain] - rinfo[higain].edegrees[other];

            let newdiff =
                iabs(pwgts[to] + vwgt[higain] - (pwgts[other] - rinfo[higain].edegrees[other]));
            if pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff) {
                mincut = pwgts[2];
                mincutorder = nswaps;
                mindiff = newdiff;
            } else if nswaps - mincutorder > 2 * limit
                || (nswaps - mincutorder > limit && (pwgts[2] as f64) > 1.10 * mincut as f64)
            {
                pwgts[2] += vwgt[higain] - rinfo[higain].edegrees[other];
                break;
            }

            bnd_delete(&mut nbnd, bndind, bndptr, higain);
            pwgts[to] += vwgt[higain];
            r#where[higain] = to as Idx;
            moved[higain] = nswaps;
            swaps[nswaps as usize] = higain as Idx;

            for j in xadj[higain] as usize..xadj[higain + 1] as usize {
                let k = adjncy[j] as usize;
                if r#where[k] == 2 {
                    let oldgain = vwgt[k] - rinfo[k].edegrees[to];
                    rinfo[k].edegrees[to] += vwgt[higain];
                    if moved[k] == -1 || moved[k] == -(2 + other as Idx) {
                        queues[other].update(k as Idx, (oldgain - vwgt[higain]) as Real);
                    }
                } else if r#where[k] == other as Idx {
                    // pulled into the separator
                    bnd_insert(&mut nbnd, bndind, bndptr, k);

                    mind[nmind as usize] = k as Idx;
                    nmind += 1;
                    r#where[k] = 2;
                    pwgts[other] -= vwgt[k];

                    let mut ed = [0 as Idx; 2];
                    for jj in xadj[k] as usize..xadj[k + 1] as usize {
                        let kk = adjncy[jj] as usize;
                        if r#where[kk] != 2 {
                            ed[r#where[kk] as usize] += vwgt[kk];
                        } else {
                            let oldgain = vwgt[kk] - rinfo[kk].edegrees[other];
                            rinfo[kk].edegrees[other] -= vwgt[k];
                            if moved[kk] == -1 || moved[kk] == -(2 + to as Idx) {
                                queues[to].update(kk as Idx, (oldgain + vwgt[k]) as Real);
                            }
                        }
                    }
                    rinfo[k].edegrees = ed;

                    // "Insert the new vertex into the priority queue. Only one side!"
                    if moved[k] == -1 {
                        queues[to].insert(k as Idx, (vwgt[k] - ed[other]) as Real);
                        moved[k] = -(2 + to as Idx);
                    }
                }
            }
            mptr[(nswaps + 1) as usize] = nmind;
            nswaps += 1;
        }

        // Roll back to the best separator seen.
        nswaps -= 1;
        while nswaps > mincutorder {
            let higain = swaps[nswaps as usize] as usize;

            let to = r#where[higain] as usize;
            let other = (to + 1) % 2;
            pwgts[2] += vwgt[higain];
            pwgts[to] -= vwgt[higain];
            r#where[higain] = 2;
            bnd_insert(&mut nbnd, bndind, bndptr, higain);

            let mut ed = [0 as Idx; 2];
            for j in xadj[higain] as usize..xadj[higain + 1] as usize {
                let k = adjncy[j] as usize;
                if r#where[k] == 2 {
                    rinfo[k].edegrees[to] -= vwgt[higain];
                } else {
                    ed[r#where[k] as usize] += vwgt[k];
                }
            }
            rinfo[higain].edegrees = ed;

            // Push nodes back out of the separator.
            for j in mptr[nswaps as usize]..mptr[(nswaps + 1) as usize] {
                let k = mind[j as usize] as usize;
                r#where[k] = other as Idx;
                pwgts[other] += vwgt[k];
                pwgts[2] -= vwgt[k];
                bnd_delete(&mut nbnd, bndind, bndptr, k);
                for jj in xadj[k] as usize..xadj[k + 1] as usize {
                    let kk = adjncy[jj] as usize;
                    if r#where[kk] == 2 {
                        rinfo[kk].edegrees[other] += vwgt[k];
                    }
                }
            }
            nswaps -= 1;
        }

        *g_mincut = mincut;
        *g_nbnd = nbnd;

        if mincutorder == -1 || mincut >= initcut {
            break;
        }
    }
}

/// `FM_2WayNodeRefine1Sided` (`sfm.c:261-460`) — the same refinement split into
/// two sub-passes, each allowing moves to one side only. The default `rtype`.
pub fn fm_2way_node_refine_1sided(ctrl: &mut Ctrl, graph: &mut Graph, niter: Idx) {
    // The C's prologue, taken through `Ws`: `xadj = graph->xadj; adjncy =
    // graph->adjncy; ...`. Every subscript below is one the algorithm produced
    // itself — an `adjncy` entry, a `bndind` slot, a `where` value in 0..=2 —
    // so the bound is walked in `cargo test` and elided here (`sparse::ws`,
    // `metis::tests`).
    let Graph {
        xadj,
        adjncy,
        vwgt,
        bndind,
        bndptr,
        r#where,
        pwgts,
        nrinfo,
        mincut: g_mincut,
        nbnd: g_nbnd,
        ..
    } = graph;
    let xadj = Ws::new_ref(xadj);
    let adjncy = Ws::new_ref(adjncy);
    let vwgt = Ws::new_ref(vwgt);
    let bndind = Ws::new(bndind);
    let bndptr = Ws::new(bndptr);
    let r#where = Ws::new(r#where);
    let pwgts = Ws::new(pwgts);
    let rinfo: &mut [NrInfo] = nrinfo;
    let nvtxs = graph.nvtxs;

    let mut queue = rpq_create(nvtxs as usize);

    let mut swaps = iwspacemalloc(nvtxs);
    let mut mptr = iwspacemalloc(nvtxs + 1);
    let mut mind = iwspacemalloc(2 * nvtxs);

    let mult: Real = (0.5 * ctrl.ubfactors[0] as f64) as Real;
    let badmaxpwgt = (mult * (pwgts[0] + pwgts[1] + pwgts[2]) as Real) as Idx;

    let mut to = if pwgts[0] < pwgts[1] { 1usize } else { 0usize };
    for pass in 0..2 * niter {
        let other = to;
        to = (to + 1) % 2;

        queue.reset();

        let mut mincutorder: Idx = -1;
        let initcut = *g_mincut;
        let mut mincut = *g_mincut;
        let mut nbnd = *g_nbnd;

        ctrl.rng.irand_array_permute(nbnd, &mut swaps, nbnd, 1);
        for ii in 0..nbnd as usize {
            let i = bndind[swaps[ii] as usize] as usize;
            queue.insert(i as Idx, (vwgt[i] - rinfo[i].edegrees[other]) as Real);
        }

        let limit = if ctrl.compress != 0 {
            Idx::min(5 * nbnd, 500)
        } else {
            Idx::min(3 * nbnd, 300)
        };

        mptr[0] = 0;
        let mut nmind: Idx = 0;
        let mut mindiff = iabs(pwgts[0] - pwgts[1]);

        let mut nswaps: Idx = 0;
        while nswaps < nvtxs {
            let higain = queue.get_top();
            if higain == -1 {
                break;
            }
            let higain = higain as usize;

            if nmind + xadj[higain + 1] - xadj[higain] >= 2 * nvtxs - 1 {
                break;
            }
            if pwgts[to] + vwgt[higain] > badmaxpwgt {
                break;
            }

            pwgts[2] -= vwgt[higain] - rinfo[higain].edegrees[other];

            let newdiff =
                iabs(pwgts[to] + vwgt[higain] - (pwgts[other] - rinfo[higain].edegrees[other]));
            if pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff) {
                mincut = pwgts[2];
                mincutorder = nswaps;
                mindiff = newdiff;
            } else if nswaps - mincutorder > 3 * limit
                || (nswaps - mincutorder > limit && (pwgts[2] as f64) > 1.10 * mincut as f64)
            {
                pwgts[2] += vwgt[higain] - rinfo[higain].edegrees[other];
                break;
            }

            bnd_delete(&mut nbnd, bndind, bndptr, higain);
            pwgts[to] += vwgt[higain];
            r#where[higain] = to as Idx;
            swaps[nswaps as usize] = higain as Idx;

            for j in xadj[higain] as usize..xadj[higain + 1] as usize {
                let k = adjncy[j] as usize;

                if r#where[k] == 2 {
                    rinfo[k].edegrees[to] += vwgt[higain];
                } else if r#where[k] == other as Idx {
                    bnd_insert(&mut nbnd, bndind, bndptr, k);

                    mind[nmind as usize] = k as Idx;
                    nmind += 1;
                    r#where[k] = 2;
                    pwgts[other] -= vwgt[k];

                    let mut ed = [0 as Idx; 2];
                    let iend = xadj[k + 1] as usize;
                    for jj in xadj[k] as usize..iend {
                        let kk = adjncy[jj] as usize;
                        if r#where[kk] != 2 {
                            ed[r#where[kk] as usize] += vwgt[kk];
                        } else {
                            rinfo[kk].edegrees[other] -= vwgt[k];
                            // one-sided, so kk cannot have been moved yet
                            queue.update(kk as Idx, (vwgt[kk] - rinfo[kk].edegrees[other]) as Real);
                        }
                    }
                    rinfo[k].edegrees = ed;

                    queue.insert(k as Idx, (vwgt[k] - ed[other]) as Real);
                }
            }
            mptr[(nswaps + 1) as usize] = nmind;
            nswaps += 1;
        }

        nswaps -= 1;
        while nswaps > mincutorder {
            let higain = swaps[nswaps as usize] as usize;

            pwgts[2] += vwgt[higain];
            pwgts[to] -= vwgt[higain];
            r#where[higain] = 2;
            bnd_insert(&mut nbnd, bndind, bndptr, higain);

            let mut ed = [0 as Idx; 2];
            for j in xadj[higain] as usize..xadj[higain + 1] as usize {
                let k = adjncy[j] as usize;
                if r#where[k] == 2 {
                    rinfo[k].edegrees[to] -= vwgt[higain];
                } else {
                    ed[r#where[k] as usize] += vwgt[k];
                }
            }
            rinfo[higain].edegrees = ed;

            for j in mptr[nswaps as usize]..mptr[(nswaps + 1) as usize] {
                let k = mind[j as usize] as usize;
                r#where[k] = other as Idx;
                pwgts[other] += vwgt[k];
                pwgts[2] -= vwgt[k];
                bnd_delete(&mut nbnd, bndind, bndptr, k);
                let iend = xadj[k + 1] as usize;
                for jj in xadj[k] as usize..iend {
                    let kk = adjncy[jj] as usize;
                    if r#where[kk] == 2 {
                        rinfo[kk].edegrees[other] += vwgt[k];
                    }
                }
            }
            nswaps -= 1;
        }

        *g_mincut = mincut;
        *g_nbnd = nbnd;

        if pass % 2 == 1 && (mincutorder == -1 || mincut >= initcut) {
            break;
        }
    }
}

/// `FM_2WayNodeBalance` (`sfm.c:466-...`) — moves separator vertices to the
/// light side until the two sides are balanced, ignoring the cut.
pub fn fm_2way_node_balance(ctrl: &mut Ctrl, graph: &mut Graph) {
    // The C's prologue, taken through `Ws`: `xadj = graph->xadj; adjncy =
    // graph->adjncy; ...`. Every subscript below is one the algorithm produced
    // itself — an `adjncy` entry, a `bndind` slot, a `where` value in 0..=2 —
    // so the bound is walked in `cargo test` and elided here (`sparse::ws`,
    // `metis::tests`).
    let Graph {
        xadj,
        adjncy,
        vwgt,
        bndind,
        bndptr,
        r#where,
        pwgts,
        nrinfo,
        mincut: g_mincut,
        nbnd: g_nbnd,
        ..
    } = graph;
    let xadj = Ws::new_ref(xadj);
    let adjncy = Ws::new_ref(adjncy);
    let vwgt = Ws::new_ref(vwgt);
    let bndind = Ws::new(bndind);
    let bndptr = Ws::new(bndptr);
    let r#where = Ws::new(r#where);
    let pwgts = Ws::new(pwgts);
    let rinfo: &mut [NrInfo] = nrinfo;
    let nvtxs = graph.nvtxs;

    let mult: Real = (0.5 * ctrl.ubfactors[0] as f64) as Real;

    let mut badmaxpwgt = (mult * (pwgts[0] + pwgts[1]) as Real) as Idx;
    if Idx::max(pwgts[0], pwgts[1]) < badmaxpwgt {
        return;
    }
    if iabs(pwgts[0] - pwgts[1]) < 3 * graph.tvwgt[0] / nvtxs {
        return;
    }

    let to = if pwgts[0] < pwgts[1] { 0usize } else { 1usize };
    let other = (to + 1) % 2;

    let mut queue = rpq_create(nvtxs as usize);

    let mut perm = iwspacemalloc(nvtxs);
    let mut moved = super::wspace::iset_wspace(nvtxs, -1);

    let mut nbnd = *g_nbnd;
    ctrl.rng.irand_array_permute(nbnd, &mut perm, nbnd, 1);
    for ii in 0..nbnd as usize {
        let i = bndind[perm[ii] as usize] as usize;
        queue.insert(i as Idx, (vwgt[i] - rinfo[i].edegrees[other]) as Real);
    }

    for _nswaps in 0..nvtxs {
        let higain = queue.get_top();
        if higain == -1 {
            break;
        }
        let higain = higain as usize;

        moved[higain] = 1;

        let gain = vwgt[higain] - rinfo[higain].edegrees[other];
        badmaxpwgt = (mult * (pwgts[0] + pwgts[1]) as Real) as Idx;

        if pwgts[to] > pwgts[other] {
            break;
        }
        if gain < 0 && pwgts[other] < badmaxpwgt {
            break;
        }
        if pwgts[to] + vwgt[higain] > badmaxpwgt {
            continue;
        }

        pwgts[2] -= gain;

        bnd_delete(&mut nbnd, bndind, bndptr, higain);
        pwgts[to] += vwgt[higain];
        r#where[higain] = to as Idx;

        for j in xadj[higain] as usize..xadj[higain + 1] as usize {
            let k = adjncy[j] as usize;
            if r#where[k] == 2 {
                rinfo[k].edegrees[to] += vwgt[higain];
            } else if r#where[k] == other as Idx {
                bnd_insert(&mut nbnd, bndind, bndptr, k);

                r#where[k] = 2;
                pwgts[other] -= vwgt[k];

                let mut ed = [0 as Idx; 2];
                for jj in xadj[k] as usize..xadj[k + 1] as usize {
                    let kk = adjncy[jj] as usize;
                    if r#where[kk] != 2 {
                        ed[r#where[kk] as usize] += vwgt[kk];
                    } else {
                        let oldgain = vwgt[kk] - rinfo[kk].edegrees[other];
                        rinfo[kk].edegrees[other] -= vwgt[k];
                        if moved[kk] == -1 {
                            queue.update(kk as Idx, (oldgain + vwgt[k]) as Real);
                        }
                    }
                }
                rinfo[k].edegrees = ed;

                queue.insert(k as Idx, (vwgt[k] - ed[other]) as Real);
            }
        }
    }

    *g_mincut = pwgts[2usize];
    *g_nbnd = nbnd;
}
