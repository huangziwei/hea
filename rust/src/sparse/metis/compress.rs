use super::super::ws::Ws;
use super::gklib::{ikvsorti, Ikv};
use super::graph::Graph;
use super::Idx;

/// `defs.h:50`.
const COMPRESSION_FRACTION: f64 = 0.85;

pub fn compress_graph(
    nvtxs: Idx,
    xadj: &[Idx],
    adjncy: &[Idx],
    vwgt: Option<&[Idx]>,
    cptr: &mut [Idx],
    cind: &mut [Idx],
) -> Option<Graph> {
    let n = nvtxs as usize;

    let xadj = Ws::new_ref(xadj);
    let adjncy = Ws::new_ref(adjncy);
    let cptr = Ws::new(cptr);
    let cind = Ws::new(cind);

    let mut mark_ = vec![-1 as Idx; n];
    let mut map_ = vec![-1 as Idx; n];
    let mark = Ws::new(&mut mark_);
    let map = Ws::new(&mut map_);
    let mut keys = vec![Ikv::default(); n];

    for i in 0..n {
        let mut k: Idx = 0;
        for j in xadj[i] as usize..xadj[i + 1] as usize {
            k += adjncy[j];
        }
        keys[i].key = k + i as Idx; // "Add the diagonal entry as well"
        keys[i].val = i as Idx;
    }

    ikvsorti(n, &mut keys);

    let mut l: Idx = 0;
    cptr[0usize] = 0;
    let mut cnvtxs: Idx = 0;
    for i in 0..n {
        let ii = keys[i].val as usize;
        if map[ii] != -1 {
            continue;
        }

        mark[ii] = i as Idx; // "Add the diagonal entry"
        for j in xadj[ii] as usize..xadj[ii + 1] as usize {
            mark[adjncy[j] as usize] = i as Idx;
        }

        map[ii] = cnvtxs;
        cind[l] = ii as Idx;
        l += 1;

        for j in i + 1..n {
            let iii = keys[j].val as usize;

            if keys[i].key != keys[j].key || xadj[ii + 1] - xadj[ii] != xadj[iii + 1] - xadj[iii] {
                break;
            }

            if map[iii] == -1 {
                let mut jj = xadj[iii];
                while jj < xadj[iii + 1] {
                    if mark[adjncy[jj] as usize] != i as Idx {
                        break;
                    }
                    jj += 1;
                }

                if jj == xadj[iii + 1] {
                    map[iii] = cnvtxs;
                    cind[l] = iii as Idx;
                    l += 1;
                }
            }
        }

        cnvtxs += 1;
        cptr[cnvtxs] = l;
    }

    if (cnvtxs as f64) >= COMPRESSION_FRACTION * nvtxs as f64 {
        return None;
    }

    let mut graph = Graph::new();
    let cn = cnvtxs as usize;

    let mut cnedges: Idx = 0;
    for i in 0..cn {
        let ii = cind[cptr[i] as usize] as usize;
        cnedges += xadj[ii + 1] - xadj[ii];
    }

    graph.xadj = vec![0; cn + 1];
    graph.vwgt = vec![0; cn];
    graph.adjncy = vec![0; cnedges as usize];
    graph.adjwgt = vec![1; cnedges as usize];

    for i in 0..n {
        mark[i] = -1;
    }
    let mut l: Idx = 0;
    graph.xadj[0] = 0;
    for i in 0..cn {
        mark[i] = i as Idx; // "Remove any diagonal entries in the compressed graph"
        for j in cptr[i]..cptr[i + 1] {
            let ii = cind[j] as usize;

            graph.vwgt[i] += match vwgt {
                None => 1,
                Some(v) => v[ii],
            };

            for jj in xadj[ii] as usize..xadj[ii + 1] as usize {
                let k = map[adjncy[jj] as usize];
                if mark[k] != i as Idx {
                    mark[k] = i as Idx;
                    graph.adjncy[l as usize] = k;
                    l += 1;
                }
            }
        }
        graph.xadj[i + 1] = l;
    }

    graph.nvtxs = cnvtxs;
    graph.nedges = l;
    graph.ncon = 1;

    graph.adjncy.truncate(l as usize);
    graph.adjwgt.truncate(l as usize);

    graph.setup_tvwgt();
    graph.setup_label();

    Some(graph)
}
