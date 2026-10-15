use super::super::ws::Ws;
use super::bucketsort::bucket_sort_keys_inc;
use super::ctrl::{Ctrl, METIS_CTYPE_RM, METIS_CTYPE_SHEM};
use super::gklib::{ikvsorti, makecsr, shiftcsr, Ikv};
use super::graph::Graph;
use super::wspace::{ikvwspacemalloc, iset_wspace, iwspacemalloc};
use super::Idx;

/// `defs.h:43`.
const UNMATCHED: Idx = -1;
/// `defs.h:48` — "node reduction between successive coarsening levels".
const COARSEN_FRACTION: f64 = 0.85;
/// `coarsen.c:14` — "the fraction of unmatched vertices that triggers 2-hop".
const UNMATCHEDFOR2HOP: f64 = 0.10;
/// `defs.h:23` — `(1 << 11) - 1`.
const HTLENGTH: Idx = (1 << 11) - 1;

/// `CoarsenGraph` (`coarsen.c:22-76`). Returns the index of the coarsest level.
pub fn coarsen_graph(ctrl: &mut Ctrl, levels: &mut Vec<Graph>, base: usize) -> usize {
    let mut eqewgts = equal_edge_weights(&levels[base]);

    for i in 0..levels[base].ncon as usize {
        ctrl.maxvwgt[i] = (1.5 * levels[base].tvwgt[i] as f64 / ctrl.coarsen_to as f64) as Idx;
    }

    let mut cur = base;
    loop {
        if levels[cur].cmap.is_empty() {
            levels[cur].cmap = vec![0; levels[cur].nvtxs as usize];
        }
        match_level(ctrl, levels, cur, eqewgts);
        cur += 1;
        eqewgts = false;

        let (g, finer) = (&levels[cur], &levels[cur - 1]);
        if !(g.nvtxs > ctrl.coarsen_to
            && (g.nvtxs as f64) < COARSEN_FRACTION * finer.nvtxs as f64
            && g.nedges > g.nvtxs / 2)
        {
            break;
        }
    }
    cur
}

/// `CoarsenGraphNlevels` (`coarsen.c:83-140`) — the same loop bounded by a level
/// count, with the termination test moved to the top of the body.
pub fn coarsen_graph_nlevels(
    ctrl: &mut Ctrl,
    levels: &mut Vec<Graph>,
    base: usize,
    nlevels: Idx,
) -> usize {
    let mut eqewgts = equal_edge_weights(&levels[base]);

    for i in 0..levels[base].ncon as usize {
        ctrl.maxvwgt[i] = (1.5 * levels[base].tvwgt[i] as f64 / ctrl.coarsen_to as f64) as Idx;
    }

    let mut cur = base;
    for _ in 0..nlevels {
        if levels[cur].cmap.is_empty() {
            levels[cur].cmap = vec![0; levels[cur].nvtxs as usize];
        }
        match_level(ctrl, levels, cur, eqewgts);
        cur += 1;
        eqewgts = false;

        let (g, finer) = (&levels[cur], &levels[cur - 1]);
        if g.nvtxs < ctrl.coarsen_to
            || (g.nvtxs as f64) > COARSEN_FRACTION * finer.nvtxs as f64
            || g.nedges < g.nvtxs / 2
        {
            break;
        }
    }
    cur
}

fn equal_edge_weights(graph: &Graph) -> bool {
    for i in 1..graph.nedges as usize {
        if graph.adjwgt[0] != graph.adjwgt[i] {
            return false;
        }
    }
    true
}

fn match_level(ctrl: &mut Ctrl, levels: &mut Vec<Graph>, cur: usize, eqewgts: bool) {
    if ctrl.ctype == METIS_CTYPE_RM {
        match_rm(ctrl, levels, cur);
    } else if ctrl.ctype == METIS_CTYPE_SHEM {
        if eqewgts || levels[cur].nedges == 0 {
            match_rm(ctrl, levels, cur);
        } else {
            match_shem(ctrl, levels, cur);
        }
    } else {
        unreachable!("CheckParams rejects any other ctype");
    }
}

/// `Match_RM` (`coarsen.c:147-262`) — match with a random unmatched neighbour.
fn match_rm(ctrl: &mut Ctrl, levels: &mut Vec<Graph>, cur: usize) -> Idx {
    let nvtxs = levels[cur].nvtxs;
    let mut nunmatched: usize = 0;

    let mut r#match = iset_wspace(nvtxs, UNMATCHED);
    let mut perm = iwspacemalloc(nvtxs);

    ctrl.rng.irand_array_permute(nvtxs, &mut perm, nvtxs / 8, 1);

    let cnvtxs = {
        let g = &mut levels[cur];
        let Graph {
            xadj,
            adjncy,
            vwgt,
            cmap,
            ..
        } = g;
        let xadj = Ws::new_ref(xadj);
        let adjncy = Ws::new_ref(adjncy);
        let vwgt = Ws::new_ref(vwgt);
        let cmap = Ws::new(cmap);
        let r#match = Ws::new(&mut r#match);
        let perm = Ws::new_ref(&perm);
        let maxvwgt = &ctrl.maxvwgt;
        let mut cnvtxs = 0;
        let mut last_unmatched = 0;

        for pi in 0..nvtxs as usize {
            let i = perm[pi] as usize;

            if r#match[i] != UNMATCHED {
                continue;
            }
            let mut maxidx = i as Idx;

            if vwgt[i] < maxvwgt[0] {
                if xadj[i] == xadj[i + 1] {
                    last_unmatched = last_unmatched.max(pi) + 1;
                    while last_unmatched < nvtxs as usize {
                        let j = perm[last_unmatched];
                        if r#match[j] == UNMATCHED {
                            maxidx = j;
                            break;
                        }
                        last_unmatched += 1;
                    }
                } else {
                    for j in xadj[i] as usize..xadj[i + 1] as usize {
                        let k = adjncy[j];
                        if r#match[k] == UNMATCHED && vwgt[i] + vwgt[k] <= maxvwgt[0] {
                            maxidx = k;
                            break;
                        }
                    }
                    if maxidx == i as Idx && 3 * vwgt[i] < maxvwgt[0] {
                        nunmatched += 1;
                        maxidx = UNMATCHED;
                    }
                }
            }

            if maxidx != UNMATCHED {
                cmap[i] = cnvtxs;
                cmap[maxidx] = cnvtxs;
                cnvtxs += 1;
                r#match[i] = maxidx;
                r#match[maxidx] = i as Idx;
            }
        }
        cnvtxs
    };

    if ctrl.no2hop == 0 && nunmatched as f64 > UNMATCHEDFOR2HOP * nvtxs as f64 {
        let _ = match_2hop(
            ctrl,
            &mut levels[cur],
            &perm,
            &mut r#match,
            cnvtxs,
            nunmatched,
        );
    }

    let cnvtxs = renumber_matched(&mut levels[cur], &mut r#match);

    create_coarse_graph(ctrl, levels, cur, cnvtxs, &r#match);
    cnvtxs
}

/// `Match_SHEM` (`coarsen.c:269-406`) — heavy-edge matching, visiting vertices
/// by increasing (capped) degree so every vertex gets a chance.
fn match_shem(ctrl: &mut Ctrl, levels: &mut Vec<Graph>, cur: usize) -> Idx {
    let nvtxs = levels[cur].nvtxs;
    let mut nunmatched: usize = 0;

    let mut r#match = iset_wspace(nvtxs, UNMATCHED);
    let mut perm = iwspacemalloc(nvtxs);
    let mut tperm = iwspacemalloc(nvtxs);
    let mut degrees = iwspacemalloc(nvtxs);

    ctrl.rng
        .irand_array_permute(nvtxs, &mut tperm, nvtxs / 8, 1);

    let avgdegree = {
        let g = &levels[cur];
        (0.7 * (g.xadj[nvtxs as usize] / nvtxs) as f64) as Idx
    };
    {
        let g = &levels[cur];
        for i in 0..nvtxs as usize {
            let d = g.xadj[i + 1] - g.xadj[i];
            degrees[i] = if d > avgdegree { avgdegree } else { d };
        }
    }
    bucket_sort_keys_inc(nvtxs, avgdegree, &degrees, &tperm, &mut perm);

    let cnvtxs = {
        let g = &mut levels[cur];
        let Graph {
            xadj,
            adjncy,
            adjwgt,
            vwgt,
            cmap,
            ..
        } = g;
        let xadj = Ws::new_ref(xadj);
        let adjncy = Ws::new_ref(adjncy);
        let adjwgt = Ws::new_ref(adjwgt);
        let vwgt = Ws::new_ref(vwgt);
        let cmap = Ws::new(cmap);
        let r#match = Ws::new(&mut r#match);
        let perm = Ws::new_ref(&perm);
        let maxvwgt = &ctrl.maxvwgt;
        let mut cnvtxs = 0;
        let mut last_unmatched = 0;

        for pi in 0..nvtxs as usize {
            let i = perm[pi] as usize;

            if r#match[i] != UNMATCHED {
                continue;
            }
            let mut maxidx = i as Idx;
            let mut maxwgt = -1;

            if vwgt[i] < maxvwgt[0] {
                if xadj[i] == xadj[i + 1] {
                    last_unmatched = last_unmatched.max(pi) + 1;
                    while last_unmatched < nvtxs as usize {
                        let j = perm[last_unmatched];
                        if r#match[j] == UNMATCHED {
                            maxidx = j;
                            break;
                        }
                        last_unmatched += 1;
                    }
                } else {
                    for j in xadj[i] as usize..xadj[i + 1] as usize {
                        let k = adjncy[j];
                        if r#match[k] == UNMATCHED
                            && maxwgt < adjwgt[j]
                            && vwgt[i] + vwgt[k] <= maxvwgt[0]
                        {
                            maxidx = k;
                            maxwgt = adjwgt[j];
                        }
                    }
                    if maxidx == i as Idx && 3 * vwgt[i] < maxvwgt[0] {
                        nunmatched += 1;
                        maxidx = UNMATCHED;
                    }
                }
            }

            if maxidx != UNMATCHED {
                cmap[i] = cnvtxs;
                cmap[maxidx] = cnvtxs;
                cnvtxs += 1;
                r#match[i] = maxidx;
                r#match[maxidx] = i as Idx;
            }
        }
        cnvtxs
    };

    if ctrl.no2hop == 0 && nunmatched as f64 > UNMATCHEDFOR2HOP * nvtxs as f64 {
        let _ = match_2hop(
            ctrl,
            &mut levels[cur],
            &perm,
            &mut r#match,
            cnvtxs,
            nunmatched,
        );
    }

    let cnvtxs = renumber_matched(&mut levels[cur], &mut r#match);

    create_coarse_graph(ctrl, levels, cur, cnvtxs, &r#match);
    cnvtxs
}

/// "match the final unmatched vertices with themselves and reorder the vertices
/// of the coarse graph for memory-friendly contraction" — the tail both
/// matchers share (`coarsen.c:249-259`).
fn renumber_matched(graph: &mut Graph, r#match: &mut [Idx]) -> Idx {
    let mut cnvtxs = 0;
    for i in 0..graph.nvtxs as usize {
        if r#match[i] == UNMATCHED {
            r#match[i] = i as Idx;
            graph.cmap[i] = cnvtxs;
            cnvtxs += 1;
        } else if (i as Idx) <= r#match[i] {
            graph.cmap[i] = cnvtxs;
            graph.cmap[r#match[i] as usize] = cnvtxs;
            cnvtxs += 1;
        }
    }
    cnvtxs
}

/// `Match_2Hop` (`coarsen.c:411-423`).
fn match_2hop(
    ctrl: &Ctrl,
    graph: &mut Graph,
    perm: &[Idx],
    r#match: &mut [Idx],
    mut cnvtxs: Idx,
    mut nunmatched: usize,
) -> Idx {
    let _ = ctrl;
    cnvtxs = match_2hop_any(graph, perm, r#match, cnvtxs, &mut nunmatched, 2);
    cnvtxs = match_2hop_all(graph, perm, r#match, cnvtxs, &mut nunmatched, 64);
    if nunmatched as f64 > 1.5 * UNMATCHEDFOR2HOP * graph.nvtxs as f64 {
        cnvtxs = match_2hop_any(graph, perm, r#match, cnvtxs, &mut nunmatched, 3);
    }
    if nunmatched as f64 > 2.0 * UNMATCHEDFOR2HOP * graph.nvtxs as f64 {
        cnvtxs = match_2hop_any(
            graph,
            perm,
            r#match,
            cnvtxs,
            &mut nunmatched,
            graph.nvtxs as usize,
        );
    }
    cnvtxs
}

/// `Match_2HopAny` (`coarsen.c:434-511`) — pair up low-degree unmatched
/// vertices that share any neighbour, via an inverted index.
fn match_2hop_any(
    graph: &mut Graph,
    perm: &[Idx],
    r#match: &mut [Idx],
    mut cnvtxs: Idx,
    r_nunmatched: &mut usize,
    maxdegree: usize,
) -> Idx {
    let nvtxs = graph.nvtxs;
    let mut nunmatched = *r_nunmatched;

    let mut colptr = iset_wspace(nvtxs + 1, 0);
    for i in 0..nvtxs as usize {
        if r#match[i] == UNMATCHED && ((graph.xadj[i + 1] - graph.xadj[i]) as usize) < maxdegree {
            for j in graph.xadj[i] as usize..graph.xadj[i + 1] as usize {
                colptr[graph.adjncy[j] as usize] += 1;
            }
        }
    }
    makecsr(nvtxs as usize, &mut colptr);

    let mut rowind = iwspacemalloc(colptr[nvtxs as usize]);
    for pi in 0..nvtxs as usize {
        let i = perm[pi] as usize;
        if r#match[i] == UNMATCHED && ((graph.xadj[i + 1] - graph.xadj[i]) as usize) < maxdegree {
            for j in graph.xadj[i] as usize..graph.xadj[i + 1] as usize {
                let c = graph.adjncy[j] as usize;
                rowind[colptr[c] as usize] = i as Idx;
                colptr[c] += 1;
            }
        }
    }
    shiftcsr(nvtxs as usize, &mut colptr);

    for pi in 0..nvtxs as usize {
        let i = perm[pi] as usize;
        if colptr[i + 1] - colptr[i] < 2 {
            continue;
        }

        let mut jj = colptr[i + 1];
        let mut j = colptr[i];
        while j < jj {
            if r#match[rowind[j as usize] as usize] == UNMATCHED {
                jj -= 1;
                while jj > j {
                    if r#match[rowind[jj as usize] as usize] == UNMATCHED {
                        let (a, b) = (rowind[j as usize], rowind[jj as usize]);
                        graph.cmap[a as usize] = cnvtxs;
                        graph.cmap[b as usize] = cnvtxs;
                        cnvtxs += 1;
                        r#match[a as usize] = b;
                        r#match[b as usize] = a;
                        nunmatched -= 2;
                        break;
                    }
                    jj -= 1;
                }
            }
            j += 1;
        }
    }

    *r_nunmatched = nunmatched;
    cnvtxs
}

/// `Match_2HopAll` (`coarsen.c:521-609`) — collapse vertices with *identical*
/// adjacency lists, found by hashing the list into an `ikv_t` key and sorting.
fn match_2hop_all(
    graph: &mut Graph,
    perm: &[Idx],
    r#match: &mut [Idx],
    mut cnvtxs: Idx,
    r_nunmatched: &mut usize,
    maxdegree: usize,
) -> Idx {
    let nvtxs = graph.nvtxs;
    let mut nunmatched = *r_nunmatched;
    let mask = (Idx::MAX as u64 / maxdegree as u64) as Idx;

    let mut keys = ikvwspacemalloc(nunmatched as Idx);
    let mut ncand = 0usize;
    for pi in 0..nvtxs as usize {
        let i = perm[pi] as usize;
        let idegree = graph.xadj[i + 1] - graph.xadj[i];
        if r#match[i] == UNMATCHED && idegree > 1 && (idegree as usize) < maxdegree {
            let mut k: Idx = 0;
            for j in graph.xadj[i] as usize..graph.xadj[i + 1] as usize {
                k += graph.adjncy[j] % mask;
            }
            keys[ncand] = Ikv {
                key: ((k % mask) as u64).wrapping_mul(maxdegree as u64) as Idx + idegree,
                val: i as Idx,
            };
            ncand += 1;
        }
    }
    ikvsorti(ncand, &mut keys);

    let mut mark = iset_wspace(nvtxs, 0);
    for pi in 0..ncand {
        let i = keys[pi].val as usize;
        if r#match[i] != UNMATCHED {
            continue;
        }

        for j in graph.xadj[i] as usize..graph.xadj[i + 1] as usize {
            mark[graph.adjncy[j] as usize] = i as Idx;
        }

        for pk in pi + 1..ncand {
            let k = keys[pk].val as usize;
            if r#match[k] != UNMATCHED {
                continue;
            }
            if keys[pi].key != keys[pk].key {
                break;
            }
            if graph.xadj[i + 1] - graph.xadj[i] != graph.xadj[k + 1] - graph.xadj[k] {
                break;
            }

            let mut jj = graph.xadj[k];
            while jj < graph.xadj[k + 1] {
                if mark[graph.adjncy[jj as usize] as usize] != i as Idx {
                    break;
                }
                jj += 1;
            }
            if jj == graph.xadj[k + 1] {
                graph.cmap[i] = cnvtxs;
                graph.cmap[k] = cnvtxs;
                cnvtxs += 1;
                r#match[i] = k as Idx;
                r#match[k] = i as Idx;
                nunmatched -= 2;
                break;
            }
        }
    }

    *r_nunmatched = nunmatched;
    cnvtxs
}

/// `SetupCoarseGraph` (`coarsen.c:1093-1118`) — pushed onto the level stack
/// rather than linked through `coarser`/`finer`.
fn setup_coarse_graph(graph: &Graph, cnvtxs: Idx) -> Graph {
    let mut c = Graph::new();
    c.nvtxs = cnvtxs;
    c.ncon = graph.ncon;

    c.xadj = vec![0; (cnvtxs + 1) as usize];
    c.adjncy = vec![0; graph.nedges as usize];
    c.adjwgt = vec![0; graph.nedges as usize];
    c.vwgt = vec![0; (c.ncon * cnvtxs) as usize];
    c.tvwgt = vec![0; c.ncon as usize];
    c.invtvwgt = vec![0.0; c.ncon as usize];
    c
}

/// `CreateCoarseGraph` (`coarsen.c:621-789`) — the masked hash-table version,
/// with `CreateCoarseGraphNoMask` as its fallback.
fn create_coarse_graph(
    ctrl: &Ctrl,
    levels: &mut Vec<Graph>,
    cur: usize,
    cnvtxs: Idx,
    r#match: &[Idx],
) {
    let _ = ctrl;
    let mask = HTLENGTH;

    let use_mask = {
        let g = &levels[cur];
        if cnvtxs < 2 * mask || g.nedges / g.nvtxs > mask / 20 {
            false
        } else {
            (0..g.nvtxs as usize).all(|v| g.xadj[v + 1] - g.xadj[v] <= (mask >> 3))
        }
    };

    if !use_mask {
        create_coarse_graph_nomask(levels, cur, cnvtxs, r#match);
        return;
    }

    let mut cgraph = setup_coarse_graph(&levels[cur], cnvtxs);
    let g = &levels[cur];
    let ncon = g.ncon as usize;

    let xadj = Ws::new_ref(&g.xadj);
    let adjncy = Ws::new_ref(&g.adjncy);
    let adjwgt = Ws::new_ref(&g.adjwgt);
    let vwgt = Ws::new_ref(&g.vwgt);
    let cmap = Ws::new_ref(&g.cmap);
    let cxadj = Ws::new(&mut cgraph.xadj);
    let cvwgt = Ws::new(&mut cgraph.vwgt);
    let cadjncy = Ws::new(&mut cgraph.adjncy);
    let cadjwgt = Ws::new(&mut cgraph.adjwgt);

    let mut htable_ = iwspacemalloc(mask + 1);
    for h in htable_.iter_mut().take((cnvtxs + 1).min(mask + 1) as usize) {
        *h = -1;
    }
    let htable = Ws::new(&mut htable_);

    let mut cv = 0usize; // the C reuses `cnvtxs` as the running coarse index
    let mut cnedges = 0usize;
    cxadj[0usize] = 0;

    for v in 0..g.nvtxs as usize {
        let u = r#match[v];
        if u < v as Idx {
            continue;
        }
        let u = u as usize;

        cvwgt[cv * ncon] = vwgt[v];

        let mut nedges = 0usize;
        for j in xadj[v] as usize..xadj[v + 1] as usize {
            let k = cmap[adjncy[j] as usize];
            let kk = k & mask;
            let m = htable[kk];
            if m == -1 {
                cadjncy[cnedges + nedges] = k;
                cadjwgt[cnedges + nedges] = adjwgt[j];
                htable[kk] = nedges as Idx;
                nedges += 1;
            } else if cadjncy[cnedges + m as usize] == k {
                cadjwgt[cnedges + m as usize] += adjwgt[j];
            } else {
                let mut jj = 0usize;
                while jj < nedges {
                    if cadjncy[cnedges + jj] == k {
                        cadjwgt[cnedges + jj] += adjwgt[j];
                        break;
                    }
                    jj += 1;
                }
                if jj == nedges {
                    cadjncy[cnedges + nedges] = k;
                    cadjwgt[cnedges + nedges] = adjwgt[j];
                    nedges += 1;
                }
            }
        }

        if v != u {
            cvwgt[cv * ncon] += vwgt[u];

            for j in xadj[u] as usize..xadj[u + 1] as usize {
                let k = cmap[adjncy[j] as usize];
                let kk = k & mask;
                let m = htable[kk];
                if m == -1 {
                    cadjncy[cnedges + nedges] = k;
                    cadjwgt[cnedges + nedges] = adjwgt[j];
                    htable[kk] = nedges as Idx;
                    nedges += 1;
                } else if cadjncy[cnedges + m as usize] == k {
                    cadjwgt[cnedges + m as usize] += adjwgt[j];
                } else {
                    let mut jj = 0usize;
                    while jj < nedges {
                        if cadjncy[cnedges + jj] == k {
                            cadjwgt[cnedges + jj] += adjwgt[j];
                            break;
                        }
                        jj += 1;
                    }
                    if jj == nedges {
                        cadjncy[cnedges + nedges] = k;
                        cadjwgt[cnedges + nedges] = adjwgt[j];
                        nedges += 1;
                    }
                }
            }

            let mut jj = htable[cv as Idx & mask];
            if jj >= 0 && cadjncy[cnedges + jj as usize] != cv as Idx {
                jj = 0;
                while (jj as usize) < nedges {
                    if cadjncy[cnedges + jj as usize] == cv as Idx {
                        break;
                    }
                    jj += 1;
                }
            }
            if jj >= 0 && (jj as usize) < nedges && cadjncy[cnedges + jj as usize] == cv as Idx {
                nedges -= 1;
                cadjncy[cnedges + jj as usize] = cadjncy[cnedges + nedges];
                cadjwgt[cnedges + jj as usize] = cadjwgt[cnedges + nedges];
            }
        }

        for j in 0..nedges {
            htable[(cadjncy[cnedges + j] & mask) as usize] = -1;
        }
        htable[cv as Idx & mask] = -1;

        cnedges += nedges;
        cv += 1;
        cxadj[cv] = cnedges as Idx;
    }

    finish_coarse_graph(levels, cur, cgraph, cnedges);
}

/// `CreateCoarseGraphNoMask` (`coarsen.c:796-921`) — a full `cnvtxs`-length
/// hash table, so there are no collisions to scan past.
fn create_coarse_graph_nomask(levels: &mut Vec<Graph>, cur: usize, cnvtxs: Idx, r#match: &[Idx]) {
    let mut cgraph = setup_coarse_graph(&levels[cur], cnvtxs);
    let g = &levels[cur];
    let ncon = g.ncon as usize;

    let xadj = Ws::new_ref(&g.xadj);
    let adjncy = Ws::new_ref(&g.adjncy);
    let adjwgt = Ws::new_ref(&g.adjwgt);
    let vwgt = Ws::new_ref(&g.vwgt);
    let cmap = Ws::new_ref(&g.cmap);
    let cxadj = Ws::new(&mut cgraph.xadj);
    let cvwgt = Ws::new(&mut cgraph.vwgt);
    let cadjncy = Ws::new(&mut cgraph.adjncy);
    let cadjwgt = Ws::new(&mut cgraph.adjwgt);
    let mut htable_ = iset_wspace(cnvtxs, -1);
    let htable = Ws::new(&mut htable_);

    let mut cv = 0usize;
    let mut cnedges = 0usize;
    cxadj[0usize] = 0;

    for v in 0..g.nvtxs as usize {
        let u = r#match[v];
        if u < v as Idx {
            continue;
        }
        let u = u as usize;

        cvwgt[cv * ncon] = vwgt[v];

        let mut nedges = 0usize;
        for j in xadj[v]..xadj[v + 1] {
            let k = cmap[adjncy[j]];
            let m = htable[k];
            if m == -1 {
                cadjncy[cnedges + nedges] = k;
                cadjwgt[cnedges + nedges] = adjwgt[j];
                htable[k] = nedges as Idx;
                nedges += 1;
            } else {
                cadjwgt[cnedges + m as usize] += adjwgt[j];
            }
        }

        if v != u {
            cvwgt[cv * ncon] += vwgt[u];

            for j in xadj[u]..xadj[u + 1] {
                let k = cmap[adjncy[j]];
                let m = htable[k];
                if m == -1 {
                    cadjncy[cnedges + nedges] = k;
                    cadjwgt[cnedges + nedges] = adjwgt[j];
                    htable[k] = nedges as Idx;
                    nedges += 1;
                } else {
                    cadjwgt[cnedges + m as usize] += adjwgt[j];
                }
            }

            let j = htable[cv];
            if j != -1 {
                nedges -= 1;
                cadjncy[cnedges + j as usize] = cadjncy[cnedges + nedges];
                cadjwgt[cnedges + j as usize] = cadjwgt[cnedges + nedges];
                htable[cv] = -1;
            }
        }

        for j in 0..nedges {
            htable[cadjncy[cnedges + j]] = -1;
        }

        cnedges += nedges;
        cv += 1;
        cxadj[cv] = cnedges as Idx;
    }

    finish_coarse_graph(levels, cur, cgraph, cnedges);
}

/// The tail both contractions share: record `nedges`, recompute `tvwgt`, then
/// `ReAdjustMemory` (`coarsen.c:1074-1081`).
fn finish_coarse_graph(levels: &mut Vec<Graph>, cur: usize, mut cgraph: Graph, cnedges: usize) {
    cgraph.nedges = cnedges as Idx;

    for j in 0..cgraph.ncon as usize {
        cgraph.tvwgt[j] = super::graph::isum_strided(cgraph.nvtxs, &cgraph.vwgt[j..], cgraph.ncon);
        let d = if cgraph.tvwgt[j] > 0 {
            cgraph.tvwgt[j]
        } else {
            1
        };
        cgraph.invtvwgt[j] = (1.0f64 / d as f64) as super::Real;
    }

    let shrink = cgraph.nedges > 10000 && (cgraph.nedges as f64) < 0.9 * levels[cur].nedges as f64;
    cgraph.adjncy.truncate(cnedges);
    cgraph.adjwgt.truncate(cnedges);
    if shrink {
        cgraph.adjncy.shrink_to_fit();
        cgraph.adjwgt.shrink_to_fit();
    }

    debug_assert_eq!(levels.len(), cur + 1);
    levels.push(cgraph);
}
