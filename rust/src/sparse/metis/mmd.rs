//! `libmetis/mmd.c` — multiple minimum degree, used to order the leaves of the
//! separator tree once a subgraph drops below `MMDSWITCH` vertices.
//!
//! Upstream is a transliteration of SPARSPAK Fortran: 1-based arrays obtained by
//! decrementing the pointers (`xadj--; adjncy--; …` at the top of `genmmd`), and
//! control flow carried by `goto`. Both survive here rather than being
//! rewritten, because the algorithm's output depends on visit order in ways that
//! are not locally obvious:
//!
//! * **1-based indexing is explicit.** Every array below is passed with slot 0
//!   unused, so `xadj[node]` here is `xadj[node]` there. `MMDOrder` builds the
//!   shifted copies. They arrive as [`Ws`], because every subscript in this file
//!   is an `idx_t` the algorithm itself produced — a linked-list successor, a
//!   degree bucket, a negated element id — and a checked access on each would
//!   be most of the cost of the ordering (`sparse::ws`).
//! * **The `goto` graph becomes a state machine** in `mmdupd`, because its
//!   back-edges land in the middle of the body (`n2200` jumps to either `n900`
//!   or `n1600`) and no arrangement of loops reproduces that. The simpler
//!   back-edges in `mmdelm` and `genmmd` are labelled loops.

use super::super::ws::Ws;
use super::Idx;

/// `genmmd` (`mmd.c:53-150`).
///
/// `xadj`/`adjncy` are 1-based (slot 0 unused, values already incremented);
/// `adjncy` is destroyed. The workspace arrays are indexed `1..=neqns+5`.
#[allow(clippy::too_many_arguments)]
pub fn genmmd(
    neqns: Idx,
    xadj: &Ws,
    adjncy: &mut Ws,
    invp: &mut Ws,
    perm: &mut Ws,
    delta: Idx,
    head: &mut Ws,
    qsize: &mut Ws,
    list: &mut Ws,
    marker: &mut Ws,
    maxint: Idx,
    ncsub: &mut Idx,
) {
    if neqns <= 0 {
        return;
    }

    *ncsub = 0;
    mmdint(neqns, xadj, head, invp, perm, qsize, list, marker);

    // 'num' counts the number of ordered nodes plus 1.
    let mut num: Idx = 1;

    // eliminate all isolated nodes
    let mut nextmd = head[1usize];
    while nextmd > 0 {
        let mdeg_node = nextmd;
        nextmd = invp[mdeg_node];
        marker[mdeg_node] = maxint;
        invp[mdeg_node] = -num;
        num += 1;
    }

    if num <= neqns {
        let mut tag: Idx = 1;
        head[1usize] = 0;
        let mut mdeg: Idx = 2;

        'n1000: loop {
            while head[mdeg] <= 0 {
                mdeg += 1;
            }

            // 'mdlmt' governs when a degree update is performed
            let mdlmt = mdeg + delta;
            let mut ehead: Idx = 0;

            'n900: loop {
                let mut mdeg_node = head[mdeg];
                while mdeg_node <= 0 {
                    mdeg += 1;
                    if mdeg > mdlmt {
                        break 'n900;
                    }
                    mdeg_node = head[mdeg];
                }

                // remove 'mdeg_node' from the degree structure
                let nextmd = invp[mdeg_node];
                head[mdeg] = nextmd;
                if nextmd > 0 {
                    perm[nextmd] = -mdeg;
                }
                invp[mdeg_node] = -num;
                *ncsub += mdeg + qsize[mdeg_node] - 2;
                if num + qsize[mdeg_node] > neqns {
                    break 'n1000;
                }

                // eliminate 'mdeg_node' and transform the quotient graph
                tag += 1;
                if tag >= maxint {
                    tag = 1;
                    for i in 1..=neqns {
                        if marker[i] < maxint {
                            marker[i] = 0;
                        }
                    }
                }

                mmdelm(
                    mdeg_node, xadj, adjncy, head, invp, perm, qsize, list, marker, maxint, tag,
                );

                num += qsize[mdeg_node];
                list[mdeg_node] = ehead;
                ehead = mdeg_node;
                if delta >= 0 {
                    continue 'n900;
                }
                break 'n900;
            }

            // n900: update the degrees of the affected nodes
            if num > neqns {
                break 'n1000;
            }
            mmdupd(
                ehead, neqns, xadj, adjncy, delta, &mut mdeg, head, invp, perm, qsize, list,
                marker, maxint, &mut tag,
            );
        }
    }

    mmdnum(neqns, perm, invp, qsize);
}

/// `mmdint` (`mmd.c:300-...`).
#[allow(clippy::too_many_arguments)]
fn mmdint(
    neqns: Idx,
    xadj: &Ws,
    head: &mut Ws,
    forward: &mut Ws,
    backward: &mut Ws,
    qsize: &mut Ws,
    list: &mut Ws,
    marker: &mut Ws,
) {
    for node in 1..=neqns {
        head[node] = 0;
        qsize[node] = 1;
        marker[node] = 0;
        list[node] = 0;
    }

    for node in 1..=neqns {
        // the commented-out "+ 1" is upstream's; leaving it out is the George
        // modification the comment names.
        let mut ndeg = xadj[node + 1] - xadj[node];
        if ndeg == 0 {
            ndeg = 1;
        }
        let fnode = head[ndeg];
        forward[node] = fnode;
        head[ndeg] = node;
        if fnode > 0 {
            backward[fnode] = node;
        }
        backward[node] = -ndeg;
    }
}

/// `mmdelm` (`mmd.c:171-289`) — eliminate `mdeg_node` and update the quotient
/// graph in place.
#[allow(clippy::too_many_arguments)]
fn mmdelm(
    mdeg_node: Idx,
    xadj: &Ws,
    adjncy: &mut Ws,
    head: &mut Ws,
    forward: &mut Ws,
    backward: &mut Ws,
    qsize: &mut Ws,
    list: &mut Ws,
    marker: &mut Ws,
    maxint: Idx,
    tag: Idx,
) {
    // find the reachable set of 'mdeg_node'
    marker[mdeg_node] = tag;
    let istart = xadj[mdeg_node];
    let istop = xadj[(mdeg_node + 1) as usize] - 1;

    let mut element: Idx = 0;
    let mut rloc = istart;
    let mut rlmt = istop;
    let mut i = istart;
    while i <= istop {
        let nabor = adjncy[i];
        if nabor == 0 {
            break;
        }
        if marker[nabor] < tag {
            marker[nabor] = tag;
            if forward[nabor] < 0 {
                list[nabor] = element;
                element = nabor;
            } else {
                adjncy[rloc] = nabor;
                rloc += 1;
            }
        }
        i += 1;
    }

    // merge with reachable nodes from generalized elements
    while element > 0 {
        adjncy[rlmt] = -element;
        let mut link = element;

        'n400: loop {
            let jstart = xadj[link];
            let jstop = xadj[(link + 1) as usize] - 1;
            let mut j = jstart;
            while j <= jstop {
                let node = adjncy[j];
                link = -node;
                if node < 0 {
                    continue 'n400;
                }
                if node == 0 {
                    break;
                }
                if marker[node] < tag && forward[node] >= 0 {
                    marker[node] = tag;
                    // use storage from eliminated nodes if necessary
                    while rloc >= rlmt {
                        let l = -adjncy[rlmt];
                        rloc = xadj[l];
                        rlmt = xadj[(l + 1) as usize] - 1;
                    }
                    adjncy[rloc] = node;
                    rloc += 1;
                }
                j += 1;
            }
            break;
        }
        element = list[element];
    }
    if rloc <= rlmt {
        adjncy[rloc] = 0;
    }

    // for each node in the reachable set
    let mut link = mdeg_node;
    'n1100: loop {
        let istart = xadj[link];
        let istop = xadj[(link + 1) as usize] - 1;
        let mut i = istart;
        while i <= istop {
            let rnode = adjncy[i];
            link = -rnode;
            if rnode < 0 {
                continue 'n1100;
            }
            if rnode == 0 {
                return;
            }

            // 'rnode' is in the degree list structure: remove it
            let pvnode = backward[rnode];
            if pvnode != 0 && pvnode != -maxint {
                let nxnode = forward[rnode];
                if nxnode > 0 {
                    backward[nxnode] = pvnode;
                }
                if pvnode > 0 {
                    forward[pvnode] = nxnode;
                }
                let npv = -pvnode;
                if pvnode < 0 {
                    head[npv] = nxnode;
                }
            }

            // purge inactive quotient nabors of 'rnode'
            let jstart = xadj[rnode];
            let jstop = xadj[(rnode + 1) as usize] - 1;
            let mut xqnbr = jstart;
            let mut j = jstart;
            while j <= jstop {
                let nabor = adjncy[j];
                if nabor == 0 {
                    break;
                }
                if marker[nabor] < tag {
                    adjncy[xqnbr] = nabor;
                    xqnbr += 1;
                }
                j += 1;
            }

            let nqnbrs = xqnbr - jstart;
            if nqnbrs <= 0 {
                // merge 'rnode' with 'mdeg_node'
                qsize[mdeg_node] += qsize[rnode];
                qsize[rnode] = 0;
                marker[rnode] = maxint;
                forward[rnode] = -mdeg_node;
                backward[rnode] = -maxint;
            } else {
                // flag 'rnode' for a degree update
                forward[rnode] = nqnbrs + 1;
                backward[rnode] = 0;
                adjncy[xqnbr] = mdeg_node;
                xqnbr += 1;
                if xqnbr <= jstop {
                    adjncy[xqnbr] = 0;
                }
            }
            i += 1;
        }
        return;
    }
}

/// `mmdnum` (`mmd.c:...`) — turn the merged-tree encoding in `invp` into the
/// final permutation pair.
fn mmdnum(neqns: Idx, perm: &mut Ws, invp: &mut Ws, qsize: &Ws) {
    for node in 1..=neqns {
        let nqsize = qsize[node];
        if nqsize <= 0 {
            perm[node] = invp[node];
        }
        if nqsize > 0 {
            perm[node] = -invp[node];
        }
    }

    for node in 1..=neqns {
        if perm[node] <= 0 {
            // trace the merged tree to its root
            let mut father = node;
            while perm[father] <= 0 {
                father = -perm[father];
            }

            let root = father;
            let num = perm[root] + 1;
            invp[node] = -num;
            perm[root] = num;

            // shorten the merged tree
            let mut father = node;
            let mut nextf = -perm[father];
            while nextf > 0 {
                perm[father] = -root;
                father = nextf;
                nextf = -perm[father];
            }
        }
    }

    for node in 1..=neqns {
        let num = -invp[node];
        invp[node] = num;
        perm[num] = node;
    }
}

/// The labels `mmdupd` jumps between (`mmd.c:...`).
enum S {
    N100,
    N900,
    N1500,
    N1600,
    N2100,
    N2200,
    N2300,
}

/// `mmdupd` (`mmd.c:...`) — recompute the degrees of everything touched by the
/// last elimination step.
// The C declares `q2head`, `qxhead`, `deg0`, `enode`, `iq2`, `deg` and `link`
// uninitialized at function scope; the state machine always enters at `N100`,
// which writes them, so their Rust initializers are never read.
#[allow(clippy::too_many_arguments, unused_assignments)]
fn mmdupd(
    ehead: Idx,
    neqns: Idx,
    xadj: &Ws,
    adjncy: &Ws,
    delta: Idx,
    mdeg: &mut Idx,
    head: &mut Ws,
    forward: &mut Ws,
    backward: &mut Ws,
    qsize: &mut Ws,
    list: &mut Ws,
    marker: &mut Ws,
    maxint: Idx,
    tag: &mut Idx,
) {
    let mdeg0 = *mdeg + delta;
    let mut element = ehead;

    let mut mtag: Idx = 0;
    let mut q2head: Idx = 0;
    let mut qxhead: Idx = 0;
    let mut deg0: Idx = 0;
    let mut enode: Idx = 0;
    let mut iq2: Idx = 1;
    let mut deg: Idx = 0;
    let mut link: Idx = 0;

    let mut state = S::N100;
    loop {
        match state {
            S::N100 => {
                if element <= 0 {
                    return;
                }

                // reset the tag value if necessary
                mtag = *tag + mdeg0;
                if mtag >= maxint {
                    *tag = 1;
                    for i in 1..=neqns {
                        if marker[i] < maxint {
                            marker[i] = 0;
                        }
                    }
                    mtag = *tag + mdeg0;
                }

                // split the nodes of this element into a two-nabor list
                // (q2head) and an everything-else list (qxhead)
                q2head = 0;
                qxhead = 0;
                deg0 = 0;
                link = element;

                'n400: loop {
                    let istart = xadj[link];
                    let istop = xadj[(link + 1) as usize] - 1;
                    let mut i = istart;
                    while i <= istop {
                        let en = adjncy[i];
                        link = -en;
                        if en < 0 {
                            continue 'n400;
                        }
                        if en == 0 {
                            break;
                        }
                        if qsize[en] != 0 {
                            deg0 += qsize[en];
                            marker[en] = mtag;

                            if backward[en] == 0 {
                                if forward[en] != 2 {
                                    list[en] = qxhead;
                                    qxhead = en;
                                } else {
                                    list[en] = q2head;
                                    q2head = en;
                                }
                            }
                        }
                        i += 1;
                    }
                    break;
                }

                enode = q2head;
                iq2 = 1;
                state = S::N900;
            }

            S::N900 => {
                if enode <= 0 {
                    state = S::N1500;
                    continue;
                }
                if backward[enode] != 0 {
                    state = S::N2200;
                    continue;
                }
                *tag += 1;
                deg = deg0;

                // identify the other adjacent element nabor
                let istart = xadj[enode];
                let mut nabor = adjncy[istart];
                if nabor == element {
                    nabor = adjncy[(istart + 1) as usize];
                }
                link = nabor;
                if forward[nabor] >= 0 {
                    // uneliminated: increase the degree count
                    deg += qsize[nabor];
                    state = S::N2100;
                    continue;
                }

                // the nabor is eliminated: walk the 2nd element. Both exits
                // -- `node == 0` and falling off the list -- go to n2100.
                'n1000: loop {
                    let istart = xadj[link];
                    let istop = xadj[(link + 1) as usize] - 1;
                    let mut i = istart;
                    while i <= istop {
                        let node = adjncy[i];
                        link = -node;
                        if node != enode {
                            if node < 0 {
                                continue 'n1000;
                            }
                            if node == 0 {
                                break;
                            }
                            if qsize[node] != 0 {
                                if marker[node] < *tag {
                                    marker[node] = *tag;
                                    deg += qsize[node];
                                } else if backward[node] == 0 {
                                    if forward[node] == 2 {
                                        // indistinguishable from 'enode': merge
                                        qsize[enode] += qsize[node];
                                        qsize[node] = 0;
                                        marker[node] = maxint;
                                        forward[node] = -enode;
                                        backward[node] = -maxint;
                                    } else {
                                        // "'node' is outmatched by 'enode'":
                                        // upstream re-tests `backward[node]
                                        // == 0` here, inside the branch that
                                        // already established it.
                                        backward[node] = -maxint;
                                    }
                                }
                            }
                        }
                        i += 1;
                    }
                    break;
                }
                state = S::N2100;
            }

            S::N1500 => {
                enode = qxhead;
                iq2 = 0;
                state = S::N1600;
            }

            S::N1600 => {
                if enode <= 0 {
                    state = S::N2300;
                    continue;
                }
                if backward[enode] != 0 {
                    state = S::N2200;
                    continue;
                }
                *tag += 1;
                deg = deg0;

                // for each unmarked nabor of 'enode'
                let istart = xadj[enode];
                let istop = xadj[(enode + 1) as usize] - 1;
                let mut i = istart;
                while i <= istop {
                    let nabor = adjncy[i];
                    if nabor == 0 {
                        break;
                    }
                    if marker[nabor] < *tag {
                        marker[nabor] = *tag;
                        link = nabor;
                        if forward[nabor] >= 0 {
                            deg += qsize[nabor];
                        } else {
                            'n1700: loop {
                                let jstart = xadj[link];
                                let jstop = xadj[(link + 1) as usize] - 1;
                                let mut j = jstart;
                                while j <= jstop {
                                    let node = adjncy[j];
                                    link = -node;
                                    if node < 0 {
                                        continue 'n1700;
                                    }
                                    if node == 0 {
                                        break;
                                    }
                                    if marker[node] < *tag {
                                        marker[node] = *tag;
                                        deg += qsize[node];
                                    }
                                    j += 1;
                                }
                                break;
                            }
                        }
                    }
                    i += 1;
                }
                state = S::N2100;
            }

            S::N2100 => {
                // update the external degree of 'enode', and '*mdeg'
                deg = deg - qsize[enode] + 1;
                let fnode = head[deg];
                forward[enode] = fnode;
                backward[enode] = -deg;
                if fnode > 0 {
                    backward[fnode] = enode;
                }
                head[deg] = enode;
                if deg < *mdeg {
                    *mdeg = deg;
                }
                state = S::N2200;
            }

            S::N2200 => {
                enode = list[enode];
                state = if iq2 == 1 { S::N900 } else { S::N1600 };
            }

            S::N2300 => {
                *tag = mtag;
                element = list[element];
                state = S::N100;
            }
        }
    }
}
