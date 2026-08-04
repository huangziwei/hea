//! AMD — approximate minimum degree ordering.
//!
//! Mechanical port of SuiteSparse 7.6.0 (AMD 3.3.1), the version R's Matrix
//! ships and therefore the one lme4 factorizes with:
//!
//!   * `AMD/Source/amd_2.c`          → [`amd_2`]
//!   * `AMD/Source/amd_postorder.c`  → [`postorder`]
//!   * `AMD/Source/amd_post_tree.c`  → [`post_tree`]
//!
//! plus the CHOLMOD-side input construction, because `cholmod_amd` does not go
//! through `amd_order`/`amd_1`/`amd_aat` at all — it builds `C = A+A'` itself
//! and calls `amd_2` directly (`CHOLMOD/Cholesky/cholmod_amd.c:139-172`, with
//! `C` from `CHOLMOD/Utility/t_cholmod_copy.c`'s symmetric→unsymmetric branch
//! at `:186-323`):
//!
//!   * `cholmod_copy (A, 0, -2)`     → [`copy_sym_to_unsym`]
//!   * `cholmod_amd`                 → [`cholmod_amd`]
//!
//! `Int` in the C is `int32_t` or `int64_t` depending on the build, and `UInt`
//! (the type of `hash`) follows it. The two differ in observable behaviour only
//! where `hash` overflows, so the width is carried explicitly in [`IntWidth`]
//! rather than baked in: everything else runs in `i64`.

/// `amd_internal.h:71`.
const EMPTY: i64 = -1;

/// `amd_internal.h:72` — `FLIP(i)` is `(-(i)-2)`.
#[inline]
fn flip(i: i64) -> i64 {
    -i - 2
}

/// `amd.h:332-333` — `AMD_DEFAULT_DENSE` / `AMD_DEFAULT_AGGRESSIVE`. CHOLMOD
/// passes exactly these: `Common->method[k].prune_dense = 10.0` and
/// `.aggressive = TRUE` (`Utility/t_cholmod_defaults.c:80,82`).
pub const DEFAULT_DENSE: f64 = 10.0;
pub const DEFAULT_AGGRESSIVE: bool = true;

/// Which C build of AMD to reproduce. `hash` is `UInt`, i.e. `uint32_t` in the
/// `int32_t` build and `uint64_t` in the `int64_t` build (`amd_internal.h:111`,
/// `:133`), so it wraps at a different modulus; and `wbig = Int_MAX - n`
/// (`amd_2.c:631`) sets how often `clear_flag` renormalizes `W`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntWidth {
    I32,
    I64,
}

impl IntWidth {
    #[inline]
    fn hash_mask(self) -> u64 {
        match self {
            IntWidth::I32 => u32::MAX as u64,
            IntWidth::I64 => u64::MAX,
        }
    }

    #[inline]
    fn int_max(self) -> i64 {
        match self {
            IntWidth::I32 => i32::MAX as i64,
            IntWidth::I64 => i64::MAX,
        }
    }
}

/// The subset of AMD's `Info` array that CHOLMOD reads back
/// (`cholmod_amd.c:177-180`), plus the statistics `amd_2.c` computes alongside.
#[derive(Clone, Copy, Debug, Default)]
pub struct AmdInfo {
    /// `Info [AMD_LNZ]` — nz in L excluding the diagonal.
    pub lnz: f64,
    /// `Info [AMD_NDIV]`.
    pub ndiv: f64,
    /// `Info [AMD_NMULTSUBS_LDL]`.
    pub nms_ldl: f64,
    /// `Info [AMD_NMULTSUBS_LU]`.
    pub nms_lu: f64,
    /// `Info [AMD_NDENSE]`.
    pub ndense: f64,
    /// `Info [AMD_DMAX]`.
    pub dmax: f64,
    /// `Info [AMD_NCMPA]` — garbage collections.
    pub ncmpa: f64,
}

/* ========================================================================= */
/* === clear_flag ========================================================== */
/* ========================================================================= */

/// `amd_2.c:22-35`.
fn clear_flag(wflg: i64, wbig: i64, w: &mut [i64], n: i64) -> i64 {
    if wflg < 2 || wflg >= wbig {
        for x in 0..n as usize {
            if w[x] != 0 {
                w[x] = 1;
            }
        }
        return 2;
    }
    /*  at this point, W [0..n-1] < wflg holds */
    wflg
}

/* ========================================================================= */
/* === AMD_post_tree ======================================================= */
/* ========================================================================= */

/// `amd_post_tree.c:15-120` — the non-recursive version, using an explicit
/// stack (the recursive one at `:39-52` is `#if 0`'d out upstream).
fn post_tree(
    root: i64,
    k_in: i64,
    child: &mut [i64],
    sibling: &[i64],
    order: &mut [i64],
    stack: &mut [i64],
) -> i64 {
    let mut k = k_in;

    /* push root on the stack */
    let mut head: i64 = 0;
    stack[0] = root;

    while head >= 0 {
        /* get head of stack */
        let i = stack[head as usize];

        if child[i as usize] != EMPTY {
            /* the children of i are not yet ordered */
            /* push each child onto the stack in reverse order */
            /* so that small ones at the head of the list get popped first */
            /* and the biggest one at the end of the list gets popped last */
            let mut f = child[i as usize];
            while f != EMPTY {
                head += 1;
                f = sibling[f as usize];
            }
            let mut h = head;
            let mut f = child[i as usize];
            while f != EMPTY {
                stack[h as usize] = f;
                h -= 1;
                f = sibling[f as usize];
            }

            /* delete child list so that i gets ordered next time we see it */
            child[i as usize] = EMPTY;
        } else {
            /* the children of i (if there were any) are already ordered */
            /* remove i from the stack and order it.  Front i is kth front */
            head -= 1;
            order[i as usize] = k;
            k += 1;
        }
    }
    k
}

/* ========================================================================= */
/* === AMD_postorder ======================================================= */
/* ========================================================================= */

/// `amd_postorder.c:15-206`.
#[allow(clippy::too_many_arguments)]
fn postorder(
    nn: i64,
    parent: &[i64],
    nv: &[i64],
    fsize: &[i64],
    order: &mut [i64],
    child: &mut [i64],
    sibling: &mut [i64],
    stack: &mut [i64],
) {
    for j in 0..nn as usize {
        child[j] = EMPTY;
        sibling[j] = EMPTY;
    }

    /* --------------------------------------------------------------------- */
    /* place the children in link lists - bigger elements tend to be last */
    /* --------------------------------------------------------------------- */

    for j in (0..nn).rev() {
        if nv[j as usize] > 0 {
            /* this is an element */
            let par = parent[j as usize];
            if par != EMPTY {
                /* place the element in link list of the children its parent */
                /* bigger elements will tend to be at the end of the list */
                sibling[j as usize] = child[par as usize];
                child[par as usize] = j;
            }
        }
    }

    /* --------------------------------------------------------------------- */
    /* place the largest child last in the list of children for each node */
    /* --------------------------------------------------------------------- */

    for i in 0..nn as usize {
        if nv[i] > 0 && child[i] != EMPTY {
            /* find the biggest element in the child list */
            let mut fprev = EMPTY;
            let mut maxfrsize = EMPTY;
            let mut bigfprev = EMPTY;
            let mut bigf = EMPTY;
            let mut f = child[i];
            while f != EMPTY {
                let frsize = fsize[f as usize];
                if frsize >= maxfrsize {
                    /* this is the biggest seen so far */
                    maxfrsize = frsize;
                    bigfprev = fprev;
                    bigf = f;
                }
                fprev = f;
                f = sibling[f as usize];
            }

            let fnext = sibling[bigf as usize];

            if fnext != EMPTY {
                /* if fnext is EMPTY then bigf is already at the end of list */

                if bigfprev == EMPTY {
                    /* delete bigf from the element of the list */
                    child[i] = fnext;
                } else {
                    /* delete bigf from the middle of the list */
                    sibling[bigfprev as usize] = fnext;
                }

                /* put bigf at the end of the list */
                sibling[bigf as usize] = EMPTY;
                sibling[fprev as usize] = bigf;
            }
        }
    }

    /* --------------------------------------------------------------------- */
    /* postorder the assembly tree */
    /* --------------------------------------------------------------------- */

    for i in 0..nn as usize {
        order[i] = EMPTY;
    }

    let mut k: i64 = 0;

    for i in 0..nn {
        if parent[i as usize] == EMPTY && nv[i as usize] > 0 {
            k = post_tree(i, k, child, sibling, order, stack);
        }
    }
}

/* ========================================================================= */
/* === AMD_2 =============================================================== */
/* ========================================================================= */

/// `amd_2.c:42-1842`.
///
/// `Pe`/`Iw`/`Len` hold the matrix on input (see `amd_2.c:237-330`); `Last` is
/// the output permutation and `Next` the output inverse permutation. `Nv` holds
/// the supernode sizes and `Elen` the column counts of L on output.
#[allow(clippy::too_many_arguments)]
pub fn amd_2(
    n: i64,
    pe: &mut [i64],
    iw: &mut [i64],
    len: &mut [i64],
    iwlen: i64,
    pfree_in: i64,
    nv: &mut [i64],
    next: &mut [i64],
    last: &mut [i64],
    head: &mut [i64],
    elen: &mut [i64],
    degree: &mut [i64],
    w: &mut [i64],
    control_dense: f64,
    control_aggressive: bool,
    width: IntWidth,
) -> AmdInfo {
    let mut pfree = pfree_in;
    let hash_mask = width.hash_mask();

    /* ===================================================================== */
    /*  INITIALIZATIONS */
    /* ===================================================================== */

    /* initialize output statistics */
    let mut lnz: f64 = 0.0;
    let mut ndiv: f64 = 0.0;
    let mut nms_lu: f64 = 0.0;
    let mut nms_ldl: f64 = 0.0;
    let mut dmax: f64 = 1.0;
    let mut me: i64;

    let mut mindeg: i64 = 0;
    let mut ncmpa: i64 = 0;
    let mut nel: i64 = 0;
    let mut lemax: i64 = 0;

    /* get control parameters */
    let alpha = control_dense;
    let aggressive = control_aggressive;

    /* Note: if alpha is NaN, this is undefined: */
    let mut dense: i64 = if alpha < 0.0 {
        /* only remove completely dense rows/columns */
        n - 2
    } else {
        (alpha * (n as f64).sqrt()) as i64
    };
    dense = dense.max(16);
    dense = dense.min(n);

    for i in 0..n as usize {
        last[i] = EMPTY;
        head[i] = EMPTY;
        next[i] = EMPTY;
        nv[i] = 1;
        w[i] = 1;
        elen[i] = 0;
        degree[i] = len[i];
    }

    /* initialize wflg */
    let wbig = width.int_max() - n;
    let mut wflg = clear_flag(0, wbig, w, n);

    /* --------------------------------------------------------------------- */
    /* initialize degree lists and eliminate dense and empty rows */
    /* --------------------------------------------------------------------- */

    let mut ndense: i64 = 0;

    for i in 0..n {
        let deg = degree[i as usize];
        if deg == 0 {
            /* -------------------------------------------------------------
             * we have a variable that can be eliminated at once because
             * there is no off-diagonal non-zero in its row.  Note that
             * Nv [i] = 1 for an empty variable i.  It is treated just
             * the same as an eliminated element i.
             * ------------------------------------------------------------- */

            elen[i as usize] = flip(1);
            nel += 1;
            pe[i as usize] = EMPTY;
            w[i as usize] = 0;
        } else if deg > dense {
            /* -------------------------------------------------------------
             * Dense variables are not treated as elements, but as unordered,
             * non-principal variables that have no parent.  They do not take
             * part in the postorder, since Nv [i] = 0.  Note that the Fortran
             * version does not have this option.
             * ------------------------------------------------------------- */

            ndense += 1;
            nv[i as usize] = 0; /* do not postorder this node */
            elen[i as usize] = EMPTY;
            nel += 1;
            pe[i as usize] = EMPTY;
        } else {
            /* -------------------------------------------------------------
             * place i in the degree list corresponding to its degree
             * ------------------------------------------------------------- */

            let inext = head[deg as usize];
            if inext != EMPTY {
                last[inext as usize] = i;
            }
            next[i as usize] = inext;
            head[deg as usize] = i;
        }
    }

    /* ===================================================================== */
    /* WHILE (selecting pivots) DO */
    /* ===================================================================== */

    while nel < n {
        /* ================================================================= */
        /* GET PIVOT OF MINIMUM DEGREE */
        /* ================================================================= */

        /* ----------------------------------------------------------------- */
        /* find next supervariable for elimination */
        /* ----------------------------------------------------------------- */

        let mut deg = mindeg;
        me = EMPTY;
        while deg < n {
            me = head[deg as usize];
            if me != EMPTY {
                break;
            }
            deg += 1;
        }
        mindeg = deg;

        /* ----------------------------------------------------------------- */
        /* remove chosen variable from link list */
        /* ----------------------------------------------------------------- */

        let inext = next[me as usize];
        if inext != EMPTY {
            last[inext as usize] = EMPTY;
        }
        head[deg as usize] = inext;

        /* ----------------------------------------------------------------- */
        /* me represents the elimination of pivots nel to nel+Nv[me]-1. */
        /* place me itself as the first in this set. */
        /* ----------------------------------------------------------------- */

        let elenme = elen[me as usize];
        let mut nvpiv = nv[me as usize];
        nel += nvpiv;

        /* ================================================================= */
        /* CONSTRUCT NEW ELEMENT */
        /* ================================================================= */

        /* flag the variable "me" as being in Lme by negating Nv [me] */
        nv[me as usize] = -nvpiv;
        let mut degme: i64 = 0;

        let pme1: i64;
        let pme2: i64;

        if elenme == 0 {
            /* ------------------------------------------------------------- */
            /* construct the new element in place */
            /* ------------------------------------------------------------- */

            pme1 = pe[me as usize];
            let mut pme2_v = pme1 - 1;

            let mut p = pme1;
            while p <= pme1 + len[me as usize] - 1 {
                let i = iw[p as usize];
                let nvi = nv[i as usize];
                if nvi > 0 {
                    /* ----------------------------------------------------- */
                    /* i is a principal variable not yet placed in Lme. */
                    /* store i in new list */
                    /* ----------------------------------------------------- */

                    /* flag i as being in Lme by negating Nv [i] */
                    degme += nvi;
                    nv[i as usize] = -nvi;
                    pme2_v += 1;
                    iw[pme2_v as usize] = i;

                    /* ----------------------------------------------------- */
                    /* remove variable i from degree list. */
                    /* ----------------------------------------------------- */

                    let ilast = last[i as usize];
                    let inext = next[i as usize];
                    if inext != EMPTY {
                        last[inext as usize] = ilast;
                    }
                    if ilast != EMPTY {
                        next[ilast as usize] = inext;
                    } else {
                        /* i is at the head of the degree list */
                        head[degree[i as usize] as usize] = inext;
                    }
                }
                p += 1;
            }
            pme2 = pme2_v;
        } else {
            /* ------------------------------------------------------------- */
            /* construct the new element in empty space, Iw [pfree ...] */
            /* ------------------------------------------------------------- */

            let mut p = pe[me as usize];
            let mut pme1_v = pfree;
            let slenme = len[me as usize] - elenme;

            for knt1 in 1..=(elenme + 1) {
                let e;
                let mut pj;
                let ln;

                if knt1 > elenme {
                    /* search the supervariables in me. */
                    e = me;
                    pj = p;
                    ln = slenme;
                } else {
                    /* search the elements in me. */
                    e = iw[p as usize];
                    p += 1;
                    pj = pe[e as usize];
                    ln = len[e as usize];
                }

                /* ---------------------------------------------------------
                 * search for different supervariables and add them to the
                 * new list, compressing when necessary. this loop is
                 * executed once for each element in the list and once for
                 * all the supervariables in the list.
                 * --------------------------------------------------------- */

                for knt2 in 1..=ln {
                    let i = iw[pj as usize];
                    pj += 1;
                    let nvi = nv[i as usize];

                    if nvi > 0 {
                        /* ------------------------------------------------- */
                        /* compress Iw, if necessary */
                        /* ------------------------------------------------- */

                        if pfree >= iwlen {
                            /* prepare for compressing Iw by adjusting pointers
                             * and lengths so that the lists being searched in
                             * the inner and outer loops contain only the
                             * remaining entries. */

                            pe[me as usize] = p;
                            len[me as usize] -= knt1;
                            /* check if nothing left of supervariable me */
                            if len[me as usize] == 0 {
                                pe[me as usize] = EMPTY;
                            }
                            pe[e as usize] = pj;
                            len[e as usize] = ln - knt2;
                            /* nothing left of element e */
                            if len[e as usize] == 0 {
                                pe[e as usize] = EMPTY;
                            }

                            ncmpa += 1; /* one more garbage collection */

                            /* store first entry of each object in Pe */
                            /* FLIP the first entry in each object */
                            for j in 0..n as usize {
                                let pn = pe[j];
                                if pn >= 0 {
                                    pe[j] = iw[pn as usize];
                                    iw[pn as usize] = flip(j as i64);
                                }
                            }

                            /* psrc/pdst point to source/destination */
                            let mut psrc: i64 = 0;
                            let mut pdst: i64 = 0;
                            let pend = pme1_v - 1;

                            while psrc <= pend {
                                /* search for next FLIP'd entry */
                                let j = flip(iw[psrc as usize]);
                                psrc += 1;
                                if j >= 0 {
                                    iw[pdst as usize] = pe[j as usize];
                                    pe[j as usize] = pdst;
                                    pdst += 1;
                                    let lenj = len[j as usize];
                                    /* copy from source to destination */
                                    for _knt3 in 0..=(lenj - 2) {
                                        iw[pdst as usize] = iw[psrc as usize];
                                        pdst += 1;
                                        psrc += 1;
                                    }
                                }
                            }

                            /* move the new partially-constructed element */
                            let p1 = pdst;
                            for psrc in pme1_v..=(pfree - 1) {
                                iw[pdst as usize] = iw[psrc as usize];
                                pdst += 1;
                            }
                            pme1_v = p1;
                            pfree = pdst;
                            pj = pe[e as usize];
                            p = pe[me as usize];
                        }

                        /* ------------------------------------------------- */
                        /* i is a principal variable not yet placed in Lme */
                        /* store i in new list */
                        /* ------------------------------------------------- */

                        /* flag i as being in Lme by negating Nv [i] */
                        degme += nvi;
                        nv[i as usize] = -nvi;
                        iw[pfree as usize] = i;
                        pfree += 1;

                        /* ------------------------------------------------- */
                        /* remove variable i from degree link list */
                        /* ------------------------------------------------- */

                        let ilast = last[i as usize];
                        let inext = next[i as usize];
                        if inext != EMPTY {
                            last[inext as usize] = ilast;
                        }
                        if ilast != EMPTY {
                            next[ilast as usize] = inext;
                        } else {
                            /* i is at the head of the degree list */
                            head[degree[i as usize] as usize] = inext;
                        }
                    }
                }

                if e != me {
                    /* set tree pointer and flag to indicate element e is
                     * absorbed into new element me (the parent of e is me) */
                    pe[e as usize] = flip(me);
                    w[e as usize] = 0;
                }
            }

            pme1 = pme1_v;
            pme2 = pfree - 1;
        }

        /* ----------------------------------------------------------------- */
        /* me has now been converted into an element in Iw [pme1..pme2] */
        /* ----------------------------------------------------------------- */

        /* degme holds the external degree of new element */
        degree[me as usize] = degme;
        pe[me as usize] = pme1;
        len[me as usize] = pme2 - pme1 + 1;

        elen[me as usize] = flip(nvpiv + degme);
        /* FLIP (Elen (me)) is now the degree of pivot (including
         * diagonal part). */

        /* ----------------------------------------------------------------- */
        /* make sure that wflg is not too large. */
        /* ----------------------------------------------------------------- */

        /* With the current value of wflg, wflg+n must not cause integer
         * overflow */

        wflg = clear_flag(wflg, wbig, w, n);

        /* ================================================================= */
        /* COMPUTE (W [e] - wflg) = |Le\Lme| FOR ALL ELEMENTS */
        /* ================================================================= */

        for pme in pme1..=pme2 {
            let i = iw[pme as usize];
            let eln = elen[i as usize];
            if eln > 0 {
                /* note that Nv [i] has been negated to denote i in Lme: */
                let nvi = -nv[i as usize];
                let wnvi = wflg - nvi;
                for p in pe[i as usize]..=(pe[i as usize] + eln - 1) {
                    let e = iw[p as usize];
                    let mut we = w[e as usize];
                    if we >= wflg {
                        /* unabsorbed element e has been seen in this loop */
                        we -= nvi;
                    } else if we != 0 {
                        /* e is an unabsorbed element */
                        /* this is the first we have seen e in all of Scan 1 */
                        we = degree[e as usize] + wnvi;
                    }
                    w[e as usize] = we;
                }
            }
        }

        /* ================================================================= */
        /* DEGREE UPDATE AND ELEMENT ABSORPTION */
        /* ================================================================= */

        for pme in pme1..=pme2 {
            let i = iw[pme as usize];
            let p1 = pe[i as usize];
            let p2 = p1 + elen[i as usize] - 1;
            let mut pn = p1;
            let mut hash: u64 = 0;
            let mut deg: i64 = 0;

            /* ------------------------------------------------------------- */
            /* scan the element list associated with supervariable i */
            /* ------------------------------------------------------------- */

            /* UMFPACK/MA38-style approximate degree: */
            if aggressive {
                for p in p1..=p2 {
                    let e = iw[p as usize];
                    let we = w[e as usize];
                    if we != 0 {
                        /* e is an unabsorbed element */
                        /* dext = | Le \ Lme | */
                        let dext = we - wflg;
                        if dext > 0 {
                            deg += dext;
                            iw[pn as usize] = e;
                            pn += 1;
                            hash = hash.wrapping_add(e as u64) & hash_mask;
                        } else {
                            /* external degree of e is zero, absorb e into me*/
                            pe[e as usize] = flip(me);
                            w[e as usize] = 0;
                        }
                    }
                }
            } else {
                for p in p1..=p2 {
                    let e = iw[p as usize];
                    let we = w[e as usize];
                    if we != 0 {
                        /* e is an unabsorbed element */
                        let dext = we - wflg;
                        deg += dext;
                        iw[pn as usize] = e;
                        pn += 1;
                        hash = hash.wrapping_add(e as u64) & hash_mask;
                    }
                }
            }

            /* count the number of elements in i (including me): */
            elen[i as usize] = pn - p1 + 1;

            /* ------------------------------------------------------------- */
            /* scan the supervariables in the list associated with i */
            /* ------------------------------------------------------------- */

            /* The bulk of the AMD run time is typically spent in this loop,
             * particularly if the matrix has many dense rows that are not
             * removed prior to ordering. */
            let p3 = pn;
            let p4 = p1 + len[i as usize];
            for p in (p2 + 1)..p4 {
                let j = iw[p as usize];
                let nvj = nv[j as usize];
                if nvj > 0 {
                    /* j is unabsorbed, and not in Lme. */
                    /* add to degree and add to new list */
                    deg += nvj;
                    iw[pn as usize] = j;
                    pn += 1;
                    hash = hash.wrapping_add(j as u64) & hash_mask;
                }
            }

            /* ------------------------------------------------------------- */
            /* update the degree and check for mass elimination */
            /* ------------------------------------------------------------- */

            if elen[i as usize] == 1 && p3 == pn {
                /* --------------------------------------------------------- */
                /* mass elimination */
                /* --------------------------------------------------------- */

                /* There is nothing left of this node except for an edge to
                 * the current pivot element.  Elen [i] is 1, and there are
                 * no variables adjacent to node i.  Absorb i into the
                 * current pivot element, me. */

                pe[i as usize] = flip(me);
                let nvi = -nv[i as usize];
                degme -= nvi;
                nvpiv += nvi;
                nel += nvi;
                nv[i as usize] = 0;
                elen[i as usize] = EMPTY;
            } else {
                /* --------------------------------------------------------- */
                /* update the upper-bound degree of i */
                /* --------------------------------------------------------- */

                /* the following degree does not yet include the size
                 * of the current element, which is added later: */

                degree[i as usize] = degree[i as usize].min(deg);

                /* --------------------------------------------------------- */
                /* add me to the list for i */
                /* --------------------------------------------------------- */

                /* move first supervariable to end of list */
                iw[pn as usize] = iw[p3 as usize];
                /* move first element to end of element part of list */
                iw[p3 as usize] = iw[p1 as usize];
                /* add new element, me, to front of list. */
                iw[p1 as usize] = me;
                /* store the new length of the list in Len [i] */
                len[i as usize] = pn - p1 + 1;

                /* --------------------------------------------------------- */
                /* place in hash bucket.  Save hash key of i in Last [i]. */
                /* --------------------------------------------------------- */

                /* NOTE: this can fail if hash is negative, because the ANSI C
                 * standard does not define a % b when a and/or b are negative.
                 * That's why hash is defined as an unsigned Int, to avoid this
                 * problem. */
                let hash = (hash % (n as u64)) as i64;

                /* if the Hhead array is not used: */
                let j = head[hash as usize];
                if j <= EMPTY {
                    /* degree list is empty, hash head is FLIP (j) */
                    next[i as usize] = flip(j);
                    head[hash as usize] = flip(i);
                } else {
                    /* degree list is not empty, use Last [Head [hash]] as
                     * hash head. */
                    next[i as usize] = last[j as usize];
                    last[j as usize] = i;
                }

                last[i as usize] = hash;
            }
        }

        degree[me as usize] = degme;

        /* ----------------------------------------------------------------- */
        /* Clear the counter array, W [...], by incrementing wflg. */
        /* ----------------------------------------------------------------- */

        /* make sure that wflg+n does not cause integer overflow */
        lemax = lemax.max(degme);
        wflg += lemax;
        wflg = clear_flag(wflg, wbig, w, n);
        /*  at this point, W [0..n-1] < wflg holds */

        /* ================================================================= */
        /* SUPERVARIABLE DETECTION */
        /* ================================================================= */

        for pme in pme1..=pme2 {
            let i0 = iw[pme as usize];
            if nv[i0 as usize] < 0 {
                /* i is a principal variable in Lme */

                /* ---------------------------------------------------------
                 * examine all hash buckets with 2 or more variables.  We do
                 * this by examing all unique hash keys for supervariables in
                 * the pattern Lme of the current element, me
                 * --------------------------------------------------------- */

                /* let i = head of hash bucket, and empty the hash bucket */
                let hash = last[i0 as usize];

                /* if Hhead array is not used: */
                let j0 = head[hash as usize];
                let mut i = if j0 == EMPTY {
                    /* hash bucket and degree list are both empty */
                    EMPTY
                } else if j0 < EMPTY {
                    /* degree list is empty */
                    head[hash as usize] = EMPTY;
                    flip(j0)
                } else {
                    /* degree list is not empty, restore Last [j] of head j */
                    let iv = last[j0 as usize];
                    last[j0 as usize] = EMPTY;
                    iv
                };

                while i != EMPTY && next[i as usize] != EMPTY {
                    /* -----------------------------------------------------
                     * this bucket has one or more variables following i.
                     * scan all of them to see if i can absorb any entries
                     * that follow i in hash bucket.  Scatter i into w.
                     * ----------------------------------------------------- */

                    let ln = len[i as usize];
                    let eln = elen[i as usize];
                    /* do not flag the first element in the list (me) */
                    for p in (pe[i as usize] + 1)..=(pe[i as usize] + ln - 1) {
                        w[iw[p as usize] as usize] = wflg;
                    }

                    /* ----------------------------------------------------- */
                    /* scan every other entry j following i in bucket */
                    /* ----------------------------------------------------- */

                    let mut jlast = i;
                    let mut j = next[i as usize];

                    while j != EMPTY {
                        /* ------------------------------------------------- */
                        /* check if j and i have identical nonzero pattern */
                        /* ------------------------------------------------- */

                        /* check if i and j have the same Len and Elen */
                        let mut ok = (len[j as usize] == ln) && (elen[j as usize] == eln);
                        /* skip the first element in the list (me) */
                        let mut p = pe[j as usize] + 1;
                        while ok && p <= pe[j as usize] + ln - 1 {
                            if w[iw[p as usize] as usize] != wflg {
                                ok = false;
                            }
                            p += 1;
                        }
                        if ok {
                            /* --------------------------------------------- */
                            /* found it!  j can be absorbed into i */
                            /* --------------------------------------------- */

                            pe[j as usize] = flip(i);
                            /* both Nv [i] and Nv [j] are negated since they */
                            /* are in Lme, and the absolute values of each */
                            /* are the number of variables in i and j: */
                            nv[i as usize] += nv[j as usize];
                            nv[j as usize] = 0;
                            elen[j as usize] = EMPTY;
                            /* delete j from hash bucket */
                            j = next[j as usize];
                            next[jlast as usize] = j;
                        } else {
                            /* j cannot be absorbed into i */
                            jlast = j;
                            j = next[j as usize];
                        }
                    }

                    /* -----------------------------------------------------
                     * no more variables can be absorbed into i
                     * go to next i in bucket and clear flag array
                     * ----------------------------------------------------- */

                    wflg += 1;
                    i = next[i as usize];
                }
            }
        }

        /* ================================================================= */
        /* RESTORE DEGREE LISTS AND REMOVE NONPRINCIPAL SUPERVARIABLES FROM */
        /* ELEMENT */
        /* ================================================================= */

        let mut p = pme1;
        let nleft = n - nel;
        for pme in pme1..=pme2 {
            let i = iw[pme as usize];
            let nvi = -nv[i as usize];
            if nvi > 0 {
                /* i is a principal variable in Lme */
                /* restore Nv [i] to signify that i is principal */
                nv[i as usize] = nvi;

                /* --------------------------------------------------------- */
                /* compute the external degree (add size of current element) */
                /* --------------------------------------------------------- */

                let mut deg = degree[i as usize] + degme - nvi;
                deg = deg.min(nleft - nvi);

                /* --------------------------------------------------------- */
                /* place the supervariable at the head of the degree list */
                /* --------------------------------------------------------- */

                let inext = head[deg as usize];
                if inext != EMPTY {
                    last[inext as usize] = i;
                }
                next[i as usize] = inext;
                last[i as usize] = EMPTY;
                head[deg as usize] = i;

                /* --------------------------------------------------------- */
                /* save the new degree, and find the minimum degree */
                /* --------------------------------------------------------- */

                mindeg = mindeg.min(deg);
                degree[i as usize] = deg;

                /* --------------------------------------------------------- */
                /* place the supervariable in the element pattern */
                /* --------------------------------------------------------- */

                iw[p as usize] = i;
                p += 1;
            }
        }

        /* ================================================================= */
        /* FINALIZE THE NEW ELEMENT */
        /* ================================================================= */

        nv[me as usize] = nvpiv;
        /* save the length of the list for the new element me */
        len[me as usize] = p - pme1;
        if len[me as usize] == 0 {
            /* there is nothing left of the current pivot element */
            /* it is a root of the assembly tree */
            pe[me as usize] = EMPTY;
            w[me as usize] = 0;
        }
        if elenme != 0 {
            /* element was not constructed in place: deallocate part of */
            /* it since newly nonprincipal variables may have been removed */
            pfree = p;
        }

        /* The new element has nvpiv pivots and the size of the contribution
         * block for a multifrontal method is degme-by-degme, not including
         * the "dense" rows/columns.  If the "dense" rows/columns are included,
         * the frontal matrix is no larger than
         * (degme+ndense)-by-(degme+ndense).
         */

        {
            let f = nvpiv as f64;
            let r = (degme + ndense) as f64;
            dmax = dmax.max(f + r);

            /* number of nonzeros in L (excluding the diagonal) */
            let lnzme = f * r + (f - 1.0) * f / 2.0;
            lnz += lnzme;

            /* number of divide operations for LDL' and for LU */
            ndiv += lnzme;

            /* number of multiply-subtract pairs for LU */
            let s = f * r * r + r * (f - 1.0) * f + (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;
            nms_lu += s;

            /* number of multiply-subtract pairs for LDL' */
            nms_ldl += (s + lnzme) / 2.0;
        }
    }

    /* ===================================================================== */
    /* DONE SELECTING PIVOTS */
    /* ===================================================================== */

    let info = {
        /* count the work to factorize the ndense-by-ndense submatrix */
        let f = ndense as f64;
        dmax = dmax.max(ndense as f64);

        /* number of nonzeros in L (excluding the diagonal) */
        let lnzme = (f - 1.0) * f / 2.0;
        lnz += lnzme;

        /* number of divide operations for LDL' and for LU */
        ndiv += lnzme;

        /* number of multiply-subtract pairs for LU */
        let s = (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;
        nms_lu += s;

        /* number of multiply-subtract pairs for LDL' */
        nms_ldl += (s + lnzme) / 2.0;

        AmdInfo {
            lnz,
            ndiv,
            nms_ldl,
            nms_lu,
            ndense: ndense as f64,
            dmax,
            ncmpa: ncmpa as f64,
        }
    };

    /* ===================================================================== */
    /* POST-ORDERING */
    /* ===================================================================== */

    /* restore Pe */
    for i in 0..n as usize {
        pe[i] = flip(pe[i]);
    }

    /* restore Elen, for output information, and for postordering */
    for i in 0..n as usize {
        elen[i] = flip(elen[i]);
    }

    /* Now the parent of j is Pe [j], or EMPTY if j is a root.  Elen [e] > 0
     * is the size of element e.  Elen [i] is EMPTY for unordered variable i. */

    /* ===================================================================== */
    /* compress the paths of the variables */
    /* ===================================================================== */

    for i in 0..n {
        if nv[i as usize] == 0 {
            /* -------------------------------------------------------------
             * i is an un-ordered row.  Traverse the tree from i until
             * reaching an element, e.  The element, e, was the principal
             * supervariable of i and all nodes in the path from i to when e
             * was selected as pivot.
             * ------------------------------------------------------------- */

            let mut j = pe[i as usize];
            if j == EMPTY {
                /* Skip a dense variable.  It has no parent. */
                continue;
            }

            /* while (j is a variable) */
            while nv[j as usize] == 0 {
                j = pe[j as usize];
            }
            /* got to an element e */
            let e = j;

            /* -------------------------------------------------------------
             * traverse the path again from i to e, and compress the path
             * (all nodes point to e).  Path compression allows this code to
             * compute in O(n) time.
             * ------------------------------------------------------------- */

            let mut j = i;
            /* while (j is a variable) */
            while nv[j as usize] == 0 {
                let jnext = pe[j as usize];
                pe[j as usize] = e;
                j = jnext;
            }
        }
    }

    /* ===================================================================== */
    /* postorder the assembly tree */
    /* ===================================================================== */

    postorder(n, pe, nv, elen, w, head, next, last);

    /* ===================================================================== */
    /* compute output permutation and inverse permutation */
    /* ===================================================================== */

    /* W [e] = k means that element e is the kth element in the new
     * order.  e is in the range 0 to n-1, and k is in the range 0 to
     * the number of elements.  Use Head for inverse order. */

    for k in 0..n as usize {
        head[k] = EMPTY;
        next[k] = EMPTY;
    }
    for e in 0..n as usize {
        let k = w[e];
        if k != EMPTY {
            head[k as usize] = e as i64;
        }
    }

    /* construct output inverse permutation in Next,
     * and permutation in Last */
    let mut nel: i64 = 0;
    for k in 0..n as usize {
        let e = head[k];
        if e == EMPTY {
            break;
        }
        next[e as usize] = nel;
        nel += nv[e as usize];
    }

    /* order non-principal variables (dense, & those merged into supervar's) */
    for i in 0..n as usize {
        if nv[i] == 0 {
            let e = pe[i];
            if e != EMPTY {
                /* This is an unordered variable that was merged
                 * into element e via supernode detection or mass
                 * elimination of i when e became the pivot element.
                 * Place i in order just before e. */
                next[i] = next[e as usize];
                next[e as usize] += 1;
            } else {
                /* This is a dense unordered variable, with no parent.
                 * Place it last in the output order. */
                next[i] = nel;
                nel += 1;
            }
        }
    }

    for i in 0..n as usize {
        let k = next[i];
        last[k as usize] = i as i64;
    }

    info
}

/* ========================================================================= */
/* === CHOLMOD's input construction ======================================== */
/* ========================================================================= */

/// `C = A+A'`, pattern only, diagonal removed, with `nnz(C)/2 + n` elbow room —
/// `cholmod_copy (A, 0, -2)` restricted to its symmetric→unsymmetric branch
/// (`Utility/t_cholmod_copy.c:186-323` plus `t_cholmod_copy_worker.c:59-135`).
///
/// `a_indptr`/`a_indices` is the CSC pattern of `A`; `stype > 0` means the
/// upper triangle is the stored half and `stype < 0` the lower, exactly as
/// CHOLMOD's `A->stype`. Entries in the ignored half are skipped, not folded
/// in — that is what makes this a copy rather than an addition.
///
/// Returns `(Cp, Ci, nzmax)`; `Ci` is allocated to `nzmax` (`cnz + cnz/2 + n`)
/// because `amd_2` uses the tail as workspace, and `Cp[n] == cnz`.
pub fn copy_sym_to_unsym(
    n: usize,
    a_indptr: &[i64],
    a_indices: &[i64],
    stype: i32,
) -> (Vec<i64>, Vec<i64>, usize) {
    let up = stype > 0;
    let lo = stype < 0;

    /* ------------------------------------------------------------------ */
    /* count entries in each column of C */
    /* ------------------------------------------------------------------ */

    let mut wj = vec![0i64; n];
    let mut cnz: usize = 0;

    for j in 0..n {
        for p in a_indptr[j]..a_indptr[j + 1] {
            let i = a_indices[p as usize];
            if i == j as i64 {
                /* diagonal entry A(i,i): ignore_diag is always true here */
                continue;
            } else if (up && i < j as i64) || (lo && i > j as i64) {
                /* A(i,j) is placed in both upper and lower part of C */
                wj[j] += 1;
                wj[i as usize] += 1;
                cnz += 2;
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* allocate C with mode == -2 elbow room */
    /* ------------------------------------------------------------------ */

    let cnzmax = cnz + (cnz / 2 + n);

    /* ------------------------------------------------------------------ */
    /* Cp = cumsum of column counts (Wj), and then copy Cp back to Wj */
    /* ------------------------------------------------------------------ */

    let mut cp = vec![0i64; n + 1];
    let mut acc: i64 = 0;
    for j in 0..n {
        cp[j] = acc;
        acc += wj[j];
    }
    cp[n] = acc;
    wj[..n].copy_from_slice(&cp[..n]);

    /* ------------------------------------------------------------------ */
    /* construct C */
    /* ------------------------------------------------------------------ */

    let mut ci = vec![0i64; cnzmax];
    for j in 0..n {
        for p in a_indptr[j]..a_indptr[j + 1] {
            let i = a_indices[p as usize];
            /* skip entries in the half that isn't stored */
            if (up && i > j as i64) || (lo && i < j as i64) {
                continue;
            }
            /* the diagonal is dropped: `keep_diag` is false for mode -2 */
            if i == j as i64 {
                continue;
            }
            /* place A(i,j) in C(:,j) and A(i,j)' in C(:,i) */
            let q = wj[j];
            wj[j] += 1;
            ci[q as usize] = i;
            let q = wj[i as usize];
            wj[i as usize] += 1;
            ci[q as usize] = j as i64;
        }
    }

    (cp, ci, cnzmax)
}

/// `cholmod_amd` (`Cholesky/cholmod_amd.c:44-194`) for a symmetric `A`.
///
/// Returns `Perm` with `Perm[k] = i` if row/column `i` of `A` is the `k`th
/// row/column of `P A P'`, plus the `Info` entries CHOLMOD reads back.
pub fn cholmod_amd(
    n: usize,
    a_indptr: &[i64],
    a_indices: &[i64],
    stype: i32,
    control_dense: f64,
    control_aggressive: bool,
    width: IntWidth,
) -> (Vec<i64>, AmdInfo) {
    if n == 0 {
        return (Vec::new(), AmdInfo::default());
    }

    /* construct the input matrix for AMD */
    let (mut cp, mut ci, cnzmax) = copy_sym_to_unsym(n, a_indptr, a_indices, stype);

    let mut len = vec![0i64; n];
    for j in 0..n {
        len[j] = cp[j + 1] - cp[j];
    }
    let cnz = cp[n];

    /* order C using AMD */
    let mut nv = vec![0i64; n];
    let mut next = vec![0i64; n];
    let mut perm = vec![0i64; n];
    let mut head = vec![0i64; n];
    let mut elen = vec![0i64; n];
    let mut degree = vec![0i64; n];
    let mut wi = vec![0i64; n];

    // `cholmod_amd` hands `amd_2` the column pointers as `Pe`; `amd_2` writes
    // through them, so the (n+1)-length `Cp` is passed as its leading n.
    let info = amd_2(
        n as i64,
        &mut cp[..n],
        &mut ci,
        &mut len,
        cnzmax as i64,
        cnz,
        &mut nv,
        &mut next,
        &mut perm,
        &mut head,
        &mut elen,
        &mut degree,
        &mut wi,
        control_dense,
        control_aggressive,
        width,
    );

    (perm, info)
}
