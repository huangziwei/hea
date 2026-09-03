use super::super::ws::Ws;
use super::ctrl::Ctrl;
use super::graph::{bnd_delete, bnd_insert, Graph};
use super::pqueue::rpq_create;
use super::wspace::iwspacemalloc;
use super::{iabs, Idx, Real};

/// `FM_2WayRefine` (`fm.c:19-25`).
pub fn fm_2way_refine(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2], niter: Idx) {
    fm_2way_cut_refine(ctrl, graph, ntpwgts, niter);
}

/// `FM_2WayCutRefine` (`fm.c:31-235`).
pub fn fm_2way_cut_refine(ctrl: &mut Ctrl, graph: &mut Graph, ntpwgts: &[Real; 2], niter: Idx) {
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
    let mut swaps = iwspacemalloc(nvtxs);
    let mut perm = iwspacemalloc(nvtxs);

    let mut tpwgts = [0 as Idx; 2];
    tpwgts[0] = (graph.tvwgt[0] as Real * ntpwgts[0]) as Idx;
    tpwgts[1] = graph.tvwgt[0] - tpwgts[0];

    let limit = (0.01 * nvtxs as f64).clamp(15.0, 100.0) as Idx;
    let psum: Idx = pwgts[0] + pwgts[1];
    let avgvwgt = Idx::min(psum / 20, 2 * psum / nvtxs);

    let mut queues = [rpq_create(nvtxs as usize), rpq_create(nvtxs as usize)];

    let origdiff = iabs(tpwgts[0] - pwgts[0]);
    for m in moved.iter_mut() {
        *m = -1;
    }

    for _pass in 0..niter {
        queues[0].reset();
        queues[1].reset();

        let mut mincutorder: Idx = -1;
        let initcut = *g_mincut;
        let mut mincut = *g_mincut;
        let mut newcut = *g_mincut;
        let mut mindiff = iabs(tpwgts[0] - pwgts[0]);

        let mut nbnd = *g_nbnd;
        ctrl.rng.irand_array_permute(nbnd, &mut perm, nbnd, 1);
        for ii in 0..nbnd as usize {
            let i = perm[ii] as usize;
            let v = bndind[i] as usize;
            queues[r#where[v] as usize].insert(v as Idx, (ed[v] - id[v]) as Real);
        }

        let mut nswaps: Idx = 0;
        while nswaps < nvtxs {
            let from = if tpwgts[0] - pwgts[0] < tpwgts[1] - pwgts[1] {
                0usize
            } else {
                1usize
            };
            let to = (from + 1) % 2;

            let higain = queues[from].get_top();
            if higain == -1 {
                break;
            }
            let higain = higain as usize;

            newcut -= ed[higain] - id[higain];
            pwgts[to] += vwgt[higain];
            pwgts[from] -= vwgt[higain];

            if (newcut < mincut && iabs(tpwgts[0] - pwgts[0]) <= origdiff + avgvwgt)
                || (newcut == mincut && iabs(tpwgts[0] - pwgts[0]) < mindiff)
            {
                mincut = newcut;
                mindiff = iabs(tpwgts[0] - pwgts[0]);
                mincutorder = nswaps;
            } else if nswaps - mincutorder > limit {
                #[allow(unused_assignments)]
                {
                    newcut += ed[higain] - id[higain];
                }
                pwgts[from] += vwgt[higain];
                pwgts[to] -= vwgt[higain];
                break;
            }

            r#where[higain] = to as Idx;
            moved[higain] = nswaps;
            swaps[nswaps as usize] = higain as Idx;

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

                if bndptr[k] != -1 {
                    if ed[k] == 0 {
                        bnd_delete(&mut nbnd, bndind, bndptr, k);
                        if moved[k] == -1 {
                            queues[r#where[k] as usize].delete(k as Idx);
                        }
                    } else if moved[k] == -1 {
                        queues[r#where[k] as usize].update(k as Idx, (ed[k] - id[k]) as Real);
                    }
                } else if ed[k] > 0 {
                    bnd_insert(&mut nbnd, bndind, bndptr, k);
                    if moved[k] == -1 {
                        queues[r#where[k] as usize].insert(k as Idx, (ed[k] - id[k]) as Real);
                    }
                }
            }

            nswaps += 1;
        }

        for i in 0..nswaps as usize {
            moved[swaps[i] as usize] = -1;
        }
        nswaps -= 1;
        while nswaps > mincutorder {
            let higain = swaps[nswaps as usize] as usize;

            let to = (r#where[higain] + 1) % 2;
            r#where[higain] = to;
            std::mem::swap(&mut id[higain], &mut ed[higain]);
            if ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain + 1] {
                bnd_delete(&mut nbnd, bndind, bndptr, higain);
            } else if ed[higain] > 0 && bndptr[higain] == -1 {
                bnd_insert(&mut nbnd, bndind, bndptr, higain);
            }

            pwgts[to] += vwgt[higain];
            pwgts[(to + 1) % 2] -= vwgt[higain];
            for j in xadj[higain] as usize..xadj[higain + 1] as usize {
                let k = adjncy[j] as usize;

                let kwgt = if to == r#where[k] {
                    adjwgt[j]
                } else {
                    -adjwgt[j]
                };
                id[k] += kwgt;
                ed[k] -= kwgt;

                if bndptr[k] != -1 && ed[k] == 0 {
                    bnd_delete(&mut nbnd, bndind, bndptr, k);
                }
                if bndptr[k] == -1 && ed[k] > 0 {
                    bnd_insert(&mut nbnd, bndind, bndptr, k);
                }
            }
            nswaps -= 1;
        }

        *g_mincut = mincut;
        *g_nbnd = nbnd;

        if mincutorder <= 0 || mincut == initcut {
            break;
        }
    }
}
