//! Symbolic analysis — the elimination tree, its postordering, the column
//! counts of `L`, and the driver that composes them with the fill-reducing
//! ordering.
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `CHOLMOD/Utility/t_cholmod_transpose_sym.c`  → [`transpose_sym`]
//!   * `CHOLMOD/Utility/t_cholmod_ptranspose.c`     → [`ptranspose`]
//!   * `CHOLMOD/Cholesky/cholmod_etree.c`           → [`etree`]
//!   * `CHOLMOD/Cholesky/cholmod_postorder.c`       → [`postorder`]
//!   * `CHOLMOD/Cholesky/cholmod_rowcolcounts.c`    → [`rowcolcounts`]
//!   * `CHOLMOD/Cholesky/cholmod_analyze.c`         → [`analyze_ordering`], [`analyze`]
//!
//! These six files are byte-identical between v7.6.0 and v7.12.2 apart from
//! `cholmod_analyze.c` saving and restoring `Common->try_catch` instead of
//! clearing it, which is error-reporting state and changes no output. So the
//! system CHOLMOD a `scikit-sparse` wheel links is a valid oracle here, and
//! `F.perm` is directly comparable to what [`analyze`] returns — unlike at the
//! AMD stage, where it is a different quantity (see [`analyze`]).
//!
//! **Scope: `A` is symmetric.** `cholmod_analyze` also handles `stype == 0`,
//! where it analyzes `AA'` — that needs `cholmod_aat`, `transpose_unsym` and
//! the `fset` machinery, and no consumer of this crate asks for it, so the
//! `stype == 0` branches of `etree`/`rowcolcounts` are not ported and
//! [`SymbolicError::Unsymmetric`] is returned instead.

use super::amd::{self, AmdInfo, IntWidth, DEFAULT_AGGRESSIVE, DEFAULT_DENSE};
use super::metis_order;
use super::ws::{columns_are_sorted, validate_csc, CscError, Work, WorkRef, Ws, EMPTY};

/// Why an analysis could not be performed.
#[derive(Debug)]
pub enum SymbolicError {
    /// The input pattern was malformed.
    Csc(CscError),
    /// `A->stype == 0`; see the module docs.
    Unsymmetric,
    /// `cholmod_postorder` returned fewer than `n` nodes, which means the
    /// elimination tree it was handed was not a tree
    /// (`cholmod_analyze.c:337-341` turns this into `CHOLMOD_INVALID`).
    NotATree { got: usize, want: usize },
}

impl From<CscError> for SymbolicError {
    fn from(e: CscError) -> Self {
        SymbolicError::Csc(e)
    }
}

impl core::fmt::Display for SymbolicError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SymbolicError::Csc(e) => write!(f, "{e}"),
            SymbolicError::Unsymmetric => write!(
                f,
                "stype must be nonzero: this port analyzes LL' = A for a \
                 symmetric A, not LL' = AA'"
            ),
            SymbolicError::NotATree { got, want } => write!(
                f,
                "the elimination tree postordered to {got} nodes, not {want}: \
                 the parent array is not a tree"
            ),
        }
    }
}

/// A `cholmod_sparse`, restricted to the two xtypes this crate builds:
/// `CHOLMOD_PATTERN` (`x` empty) and `CHOLMOD_REAL` + `CHOLMOD_DOUBLE`.
///
/// Always packed with `nz == NULL`: that is what `cholmod_allocate_sparse`
/// produces, and what arrives from scipy. The upstream templates select their
/// `pend` with `#ifdef PACKED` at compile time, so the unpacked arms are a
/// different instantiation rather than a branch, and are simply not built here.
pub struct Sparse {
    pub n: usize,
    /// `A->p`, size `n + 1`.
    pub p: Vec<i64>,
    /// `A->i`, size `p[n]`.
    pub i: Vec<i64>,
    /// `A->x`, size `p[n]`, and empty when [`Sparse::numeric`] is false.
    pub x: Vec<f64>,
    /// `A->xtype != CHOLMOD_PATTERN`. A field rather than `!x.is_empty()`,
    /// because upstream's discriminator is `A->xtype` and a real matrix with no
    /// entries still has a non-NULL `A->x`; deriving it would make an empty
    /// matrix un-factorizable.
    pub numeric: bool,
    /// `A->stype`: `> 0` upper triangle stored, `< 0` lower.
    pub stype: i32,
    /// `A->sorted`: row indices ascend within every column. A field the caller
    /// sets and CHOLMOD trusts; `cholmod_rowfac` is the one routine here that
    /// reads it, and reads it per entry (`cholmod_rowfac.c:149`).
    pub sorted: bool,
}

/* ========================================================================= */
/* === cholmod_transpose_sym =============================================== */
/* ========================================================================= */

/// `C = A'` or `C = A(p,p)'` — `cholmod_transpose_sym`
/// (`t_cholmod_transpose_sym.c:71-244`).
///
/// `C->stype = -A->stype`, so this is how the upper form and the lower form of
/// the same matrix are obtained from one another, permuted or not. `C` is
/// sorted only when `Perm` is `NULL` (`:240`); nothing downstream needs sorted
/// columns, and both `cholmod_etree` and `cholmod_rowcolcounts` say so.
///
/// `values` is upstream's `mode >= 1`, forced off when `A` is pattern-only
/// (`:96-100`). Real values make `mode = 1` and `mode = 2` the same thing, so
/// the conjugating instantiations are not built.
///
/// `Perm` is trusted here: the upstream validity check (`:134-148`) is a
/// property of what produced it, and every caller in this module passes either
/// `None` or a permutation this crate just computed.
///
/// `Wi` and `Pinv` are `Iwork [0..n)` and `Iwork [n..2n)`, as upstream slices
/// them (`:126,136`); only `C` itself is allocated.
pub fn transpose_sym(
    a: &Sparse,
    values: bool,
    perm: Option<&[i64]>,
    work: &mut WorkRef<'_>,
) -> Sparse {
    let n = a.n;
    /* C = A(p,p)' A is symmetric, or C=A' where A is any matrix */
    let cnz = a.p[n] as usize;
    let values = values && a.numeric;

    let mut cp = vec![0i64; n + 1];
    let mut ci = vec![0i64; cnz];
    let mut cx = vec![0.0f64; if values { cnz } else { 0 }];

    let ap = Ws::new_ref(&a.p);
    let ai = Ws::new_ref(&a.i[..cnz]);
    let ax = Ws::new_ref(if values { &a.x[..cnz] } else { &[] });

    let (wi_buf, pinv_buf) = work.scratch2(n);
    wi_buf.fill(0);

    /* compute Pinv and make sure Perm is valid */
    let pinv: Option<&Ws> = match perm {
        None => None,
        Some(perm) => {
            pinv_buf.fill(EMPTY);
            let pinv = Ws::new(pinv_buf);
            for (k, &i) in perm.iter().enumerate() {
                pinv[i] = k as i64;
            }
            Some(Ws::new_ref(pinv_buf))
        }
    };

    /* Count the # of entries in each column of C, then walk it again to place
     * them. `Wi [i]++` sits outside the `#ifdef NUMERIC` in the upstream
     * template, so the two passes are the same walk with the writes to `Ci`
     * compiled out — and *compiled* out is the point: the counting pass is the
     * template included directly into `cholmod_transpose_sym` with `NUMERIC`
     * undefined (`:158`), the fill pass is the same template inside the worker
     * with it defined, so neither body carries a test. `VALUES` is the same
     * story one level up: upstream dispatches on `C->xtype` to a worker where
     * `ASSIGN` is either a copy or nothing at all (`:177-234`). */
    macro_rules! pass {
        ($numeric:literal, $values:literal) => {{
            let wi = Ws::new(wi_buf);
            let ci = Ws::new(&mut ci);
            let cx = Ws::new(&mut cx);
            match (pinv, a.stype < 0) {
                (None, true) => {
                    transpose_unpermuted::<true, $numeric, $values>(n, ap, ai, ax, wi, ci, cx)
                }
                (None, false) => {
                    transpose_unpermuted::<false, $numeric, $values>(n, ap, ai, ax, wi, ci, cx)
                }
                (Some(pinv), true) => {
                    transpose_permuted::<true, $numeric, $values>(n, ap, ai, ax, pinv, wi, ci, cx)
                }
                (Some(pinv), false) => {
                    transpose_permuted::<false, $numeric, $values>(n, ap, ai, ax, pinv, wi, ci, cx)
                }
            }
        }};
    }

    pass!(false, false);

    /* compute the column pointers of C: cumsum (C->p, Wi, n), then Wi = C->p */
    let mut acc = 0i64;
    for k in 0..n {
        cp[k] = acc;
        acc += wi_buf[k];
    }
    cp[n] = acc;
    wi_buf.copy_from_slice(&cp[..n]);

    /* compute the pattern and values of C */
    if values {
        pass!(true, true);
    } else {
        pass!(true, false);
    }

    /* Entries in the half of A that is not stored were skipped, so C holds
     * fewer than nnz(A) of them; upstream leaves the slack as C->nzmax. */
    ci.truncate(cp[n] as usize);
    if values {
        cx.truncate(cp[n] as usize);
    }

    Sparse {
        n,
        p: cp,
        i: ci,
        x: cx,
        numeric: values,
        stype: -a.stype.signum(),
        /* `C->sorted = (Perm == NULL)` (`t_cholmod_transpose_sym.c:240`): the
         * unpermuted walk visits j ascending so each column of C comes out
         * ordered, the permuted one does not. */
        sorted: perm.is_none(),
    }
}

/// `C = A'` — `t_cholmod_transpose_sym_unpermuted.c`.
///
/// `LO`, `NUMERIC` and `VALUES` are the template's `#ifdef LO` / `#ifdef
/// NUMERIC` and its xtype, so they are const: upstream instantiates this body
/// once per combination rather than testing any of them per entry.
#[inline]
fn transpose_unpermuted<const LO: bool, const NUMERIC: bool, const VALUES: bool>(
    n: usize,
    ap: &Ws,
    ai: &Ws,
    ax: &Ws<f64>,
    wi: &mut Ws,
    ci: &mut Ws,
    cx: &mut Ws<f64>,
) {
    for j in 0..n as i64 {
        let (pa, paend) = (ap[j], ap[j + 1]);
        for (k, &i) in ai.range(pa, paend).iter().enumerate() {
            /* get A(i,j) */
            if LO {
                /* A is symmetric lower, C is symmetric upper */
                if i < j {
                    continue;
                }
            } else {
                /* A is symmetric upper, C is symmetric lower */
                if i > j {
                    continue;
                }
            }
            /* C(j,i) = A(i,j) */
            let pc = wi[i];
            wi[i] += 1;
            if NUMERIC {
                if VALUES {
                    cx[pc] = ax[pa + k as i64];
                }
                ci[pc] = j;
            }
        }
    }
}

/// `C = A(p,p)'` — `t_cholmod_transpose_sym_permuted.c`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn transpose_permuted<const LO: bool, const NUMERIC: bool, const VALUES: bool>(
    n: usize,
    ap: &Ws,
    ai: &Ws,
    ax: &Ws<f64>,
    pinv: &Ws,
    wi: &mut Ws,
    ci: &mut Ws,
    cx: &mut Ws<f64>,
) {
    for jold in 0..n as i64 {
        let jnew = pinv[jold];
        let (pa, paend) = (ap[jold], ap[jold + 1]);
        for (k, &iold) in ai.range(pa, paend).iter().enumerate() {
            /* get A(iold,jold) */
            let inew = pinv[iold];
            let flip = if LO {
                /* A is symmetric lower, C is symmetric upper */
                if iold < jold {
                    continue;
                }
                inew > jnew
            } else {
                /* A is symmetric upper, C is symmetric lower */
                if iold > jold {
                    continue;
                }
                inew < jnew
            };
            if flip {
                /* C(jnew,inew) = A(iold,jold) */
                let pc = wi[inew];
                wi[inew] += 1;
                if NUMERIC {
                    if VALUES {
                        cx[pc] = ax[pa + k as i64];
                    }
                    ci[pc] = jnew;
                }
            } else {
                /* C(inew,jnew) = A(iold,jold) */
                let pc = wi[jnew];
                wi[jnew] += 1;
                if NUMERIC {
                    if VALUES {
                        cx[pc] = ax[pa + k as i64];
                    }
                    ci[pc] = inew;
                }
            }
        }
    }
}

/// `cholmod_ptranspose` restricted to a symmetric `A`
/// (`t_cholmod_ptranspose.c:25-115`). For `A->stype != 0` it is exactly
/// [`transpose_sym`]; the `fset` counting above it only runs when `stype == 0`.
#[inline]
pub fn ptranspose(
    a: &Sparse,
    values: bool,
    perm: Option<&[i64]>,
    work: &mut WorkRef<'_>,
) -> Sparse {
    transpose_sym(a, values, perm, work)
}

/// `permute_matrices` (`cholmod_analyze.c:161-286`) for a symmetric `A`,
/// returning only `S`.
///
/// `S` is the permuted matrix `P A P'` with one triangle stored, in column
/// form. Every consumer needs it and they do not agree on which triangle:
/// `cholmod_rowfac` and `cholmod_super_symbolic` want `triu`, the supernodal
/// numeric factorization wants `tril` (`cholmod_factorize.c:216-238`), so
/// `lower` picks. Getting there costs one `ptranspose` or two, because a
/// transpose flips the stored triangle *and* is the only thing that applies the
/// permutation:
///
/// | | natural | permuted |
/// |---|---|---|
/// | already the wanted triangle | `S = A` | 1 |
/// | the other triangle | 1 | 2 |
///
/// `None` means `S` is `A` itself, which upstream expresses by aliasing the
/// pointer and leaving `A1`/`A2` `NULL`.
///
/// `F` is not returned. In the symmetric case it is used only by
/// `cholmod_rowcolcounts` — the `do_rowcolcounts` argument exists to skip
/// building it (`:167-169`) — and [`analyze`] holds its own.
///
/// `values` covers both of upstream's numeric modes: the simplicial branch
/// asks for `ptranspose (A, 1, ...)` and the supernodal one for
/// `ptranspose (A, 2, ...)`, and mode 2 only differs by conjugating, which for
/// `CHOLMOD_REAL` is nothing.
pub fn permute_sym(
    a: &Sparse,
    ordering: Ordering,
    perm: &[i64],
    values: bool,
    lower: bool,
    work: &mut WorkRef<'_>,
) -> Option<Sparse> {
    /* `A` is stored in the wanted triangle already iff (stype > 0) != lower */
    let as_wanted = (a.stype > 0) != lower;
    if ordering == Ordering::Natural {
        if as_wanted {
            /* S = A */
            None
        } else {
            /* S = A' */
            Some(ptranspose(a, values, None, work))
        }
    } else if as_wanted {
        /* the permuted transpose lands in the other triangle, so transpose
         * once more to come back: F = A (p,p)' and S = F' */
        let f = ptranspose(a, values, Some(perm), work);
        Some(ptranspose(&f, values, None, work))
    } else {
        /* one transpose both permutes and lands in the wanted triangle.  This
         * is the fastest option for factorizing a permuted matrix. */
        Some(ptranspose(a, values, Some(perm), work))
    }
}

/* ========================================================================= */
/* === cholmod_etree ======================================================= */
/* ========================================================================= */

/// `cholmod_etree.c:42-73` — walk the path from `k` to the root, compressing
/// it, and record `(k, i)` if that path ends without reaching `i`.
#[inline]
fn update_etree(mut k: i64, i: i64, parent: &mut Ws, ancestor: &mut Ws) {
    loop {
        /* traverse the path from k to the root of the tree */
        let a = ancestor[k];
        if a == i {
            /* final ancestor reached; no change to tree */
            return;
        }
        /* perform path compression */
        ancestor[k] = i;
        if a == EMPTY {
            /* final ancestor undefined; this is a new edge in the tree */
            parent[k] = i;
            return;
        }
        /* traverse up to the ancestor of k */
        k = a;
    }
}

/// The elimination tree of a symmetric `A` — `cholmod_etree.c:81-221`.
///
/// Only the upper triangular part of `A` is used, so `A->stype` must be
/// positive: upstream rejects the lower form outright (`:215`, "symmetric lower
/// not supported"), because the algorithm needs the columns of `triu(A)` and
/// the lower form stores its transpose.
pub fn etree(
    a: &Sparse,
    parent_buf: &mut [i64],
    work: &mut WorkRef<'_>,
) -> Result<(), SymbolicError> {
    if a.stype <= 0 {
        return Err(SymbolicError::Unsymmetric);
    }
    let ncol = a.n;
    /* Ancestor = Iwork [0..n) */
    let (ancestor_buf, _) = work.scratch2(ncol);

    let ap = Ws::new_ref(&a.p);
    let ai = Ws::new_ref(&a.i);
    let parent = Ws::new(parent_buf);
    let ancestor = Ws::new(ancestor_buf);
    for j in 0..ncol {
        parent[j] = EMPTY;
        ancestor[j] = EMPTY;
    }

    /* symmetric (upper) case: compute etree (A) */
    for j in 0..ncol as i64 {
        /* for each row i in column j of triu(A), excluding the diagonal */
        for &i in ai.range(ap[j], ap[j + 1]) {
            if i < j {
                update_etree(i, j, parent, ancestor);
            }
        }
    }
    Ok(())
}

/* ========================================================================= */
/* === cholmod_postorder =================================================== */
/* ========================================================================= */

/// `cholmod_postorder.c:60-97`, the non-recursive DFS. Returns the new `k`.
fn dfs(p: i64, mut k: i64, post: &mut Ws, head: &mut Ws, next: &Ws, pstack: &mut Ws) -> i64 {
    /* put the root node on the stack */
    pstack[0usize] = p;
    let mut phead: i64 = 0;

    /* while the stack is not empty, do: */
    while phead >= 0 {
        /* grab the node p from top of the stack and get its youngest child j */
        let p = pstack[phead];
        let j = head[p];
        if j == EMPTY {
            /* all children of p ordered.  remove p from stack and order it */
            phead -= 1;
            post[k] = p;
            k += 1;
        } else {
            /* leave p on the stack.  Start a DFS at child node j by putting
             * j on the stack and removing j from the list of children of p. */
            head[p] = next[j];
            phead += 1;
            pstack[phead] = j;
        }
    }
    k
}

/// Postorder a tree — `cholmod_postorder.c:137-286`. Returns `Post` and the
/// number of nodes ordered, which is `n` iff `Parent` really is a tree.
///
/// With `weight`, the children of a node are visited in increasing order of
/// their weight; without it, in increasing order of node number. Both are
/// needed: `analyze_ordering` postorders unweighted because
/// `cholmod_rowcolcounts` requires it, and `cholmod_analyze` then postorders
/// again weighted by the column counts, which is what makes the stored
/// permutation what it is.
pub fn postorder(
    parent: &[i64],
    n: usize,
    weight: Option<&[i64]>,
    post_buf: &mut [i64],
    work: &mut WorkRef<'_>,
) -> usize {
    /* Next = Iwork [0..n), Pstack = Iwork [n..2n); Head is Common's, size n+1
     * and all EMPTY on entry */
    let (next_buf, pstack_buf) = work.iwork[..2 * n].split_at_mut(n);
    let head_buf = &mut work.head;

    let parent = Ws::new_ref(parent);
    let post = Ws::new(post_buf);
    let head = Ws::new(head_buf);
    let pstack = Ws::new(pstack_buf);
    let n_i = n as i64;

    /* construct a link list of children for each node */
    match weight {
        None => {
            let next = Ws::new(next_buf);
            /* in reverse order so children are in ascending order in each list */
            for j in (0..n_i).rev() {
                let p = parent[j];
                if p >= 0 && p < n_i {
                    /* add j to the list of children for node p */
                    next[j] = head[p];
                    head[p] = j;
                }
            }
            /* Head [p] = j if j is the youngest (least-numbered) child of p
             * Next [j1] = j2 if j2 is the next-oldest sibling of j1 */
        }
        Some(weight) => {
            /* First, construct a set of link lists according to Weight.
             *
             * Whead [w] = j if node j is the first node in bucket w.
             * Next [j1] = j2 if node j2 follows j1 in a link list. */
            let weight = Ws::new_ref(weight);
            /* upstream aliases Whead onto Pstack, which the DFS has not
             * reached yet (`cholmod_postorder.c:223`) */
            let whead = &mut *pstack;
            let next = Ws::new(next_buf);

            for w in 0..n_i {
                whead[w] = EMPTY;
            }
            /* do in forward order, so nodes that ties are ordered by node index */
            for j in 0..n_i {
                let p = parent[j];
                if p >= 0 && p < n_i {
                    let w = weight[j].max(0).min(n_i - 1);
                    /* place node j at the head of link list for weight w */
                    next[j] = whead[w];
                    whead[w] = j;
                }
            }

            /* traverse weight buckets, placing each node in its parent's list */
            for w in (0..n_i).rev() {
                let mut j = whead[w];
                while j != EMPTY {
                    let nextj = next[j];
                    /* put node j in the link list of its parent */
                    let p = parent[j];
                    next[j] = head[p];
                    head[p] = j;
                    j = nextj;
                }
            }

            /* Whead no longer needed
             * Head [p] = j if j is the lightest child of p
             * Next [j1] = j2 if j2 is the next-heaviest sibling of j1 */
        }
    }

    /* start a DFS at each root node of the etree */
    let next = Ws::new_ref(next_buf);
    let mut k = 0i64;
    for j in 0..n_i {
        if parent[j] == EMPTY {
            /* j is the root of a tree; start a DFS here */
            k = dfs(j, k, post, head, next, pstack);
        }
    }

    /* this would normally be EMPTY already, unless Parent is invalid — and it
     * has to be, because Head is shared with whatever runs next.  Upstream
     * clears Head [0..n-1] here; Head [n] is never written, the DFS only ever
     * indexes it by a node. */
    head_buf[..n].fill(EMPTY);
    k as usize
}

/* ========================================================================= */
/* === cholmod_rowcolcounts ================================================ */
/* ========================================================================= */

/// `cholmod_rowcolcounts.c:57-78` — initial work for the `k`th node.
#[inline]
fn initialize_node(k: i64, post: &Ws, parent: &Ws, colcount: &mut Ws, prevnbr: &mut Ws) -> i64 {
    /* determine p, the kth node in the postordered etree */
    let p = post[k];
    /* adjust the weight if p is not a root of the etree */
    let par = parent[p];
    if par != EMPTY {
        colcount[par] -= 1;
    }
    /* flag node p to exclude self edges (p,p) */
    prevnbr[p] = k;
    p
}

/// Workspace and outputs of [`rowcolcounts`], grouped so the argument list
/// stays the shape the C's is.
struct RowColWork<'a> {
    first: &'a mut Ws,
    level: &'a mut Ws,
    prevnbr: &'a mut Ws,
    prevleaf: &'a mut Ws,
    setparent: &'a mut Ws,
    colcount: &'a mut Ws,
}

/// `cholmod_rowcolcounts.c:87-157` — edge `(p,u)` is being processed, where
/// `p < u` is a descendant of its ancestor `u` in the etree.
///
/// `ROWCOUNT` is the C's `RowCount != NULL` test. Upstream leaves it a runtime
/// null check here, but it is loop-invariant across the entire call, so this
/// port hoists it into a const chosen once in [`sym_pass`] rather than testing
/// it in the innermost loop of the whole analysis. Same instantiation either
/// way — `analyze_ordering` always passes `NULL` (`cholmod_analyze.c:349`); the
/// MATLAB `symbfact` interface is the only upstream caller that does not.
#[inline]
fn process_edge<const ROWCOUNT: bool>(
    p: i64,
    u: i64,
    k: i64,
    w: &mut RowColWork<'_>,
    rowcount: &mut [i64],
) {
    if w.first[p] > w.prevnbr[u] {
        /* p is a leaf of the subtree of u */
        w.colcount[p] += 1;
        let prevleaf = w.prevleaf[u];
        let q;
        if prevleaf == EMPTY {
            /* p is the first leaf of subtree of u; RowCount will be incremented
             * by the length of the path in the etree from p up to u. */
            q = u;
        } else {
            /* q = FIND (prevleaf): find the root q of the
             * SetParent tree containing prevleaf */
            let mut r = prevleaf;
            while r != w.setparent[r] {
                r = w.setparent[r];
            }
            q = r;
            /* the root q has been found; re-traverse the path and
             * perform path compression */
            let mut s = prevleaf;
            while s != q {
                let sparent = w.setparent[s];
                w.setparent[s] = q;
                s = sparent;
            }
            /* adjust the RowCount and ColCount; RowCount will be incremented by
             * the length of the path from p to the SetParent root q, and
             * decrement the ColCount of q by one. */
            w.colcount[q] -= 1;
        }
        if ROWCOUNT {
            /* if RowCount is being computed, increment it by the length of
             * the path from p to q */
            rowcount[u as usize] += w.level[p] - w.level[q];
        }
        /* p is a leaf of the subtree of u, so mark PrevLeaf [u] to be p */
        w.prevleaf[u] = p;
    }
    /* flag u has having been processed at step k */
    w.prevnbr[u] = k;
}

/// The symmetric branch of `cholmod_rowcolcounts` (`:414-446`), returning
/// `Common->anz`.
///
/// Split out only so that `ROWCOUNT` — the C's `RowCount != NULL`, which is
/// loop-invariant across the whole call — is chosen once here rather than per
/// edge. See [`process_edge`].
#[inline]
fn sym_pass<const ROWCOUNT: bool>(
    nrow: usize,
    ap: &Ws,
    ai: &Ws,
    post: &Ws,
    parent: &Ws,
    w: &mut RowColWork<'_>,
    rowcount: &mut [i64],
) -> i64 {
    let mut anz = nrow as i64;
    for k in 0..nrow as i64 {
        /* j is the kth node in the postordered etree */
        let j = initialize_node(k, post, parent, w.colcount, w.prevnbr);

        /* for all nonzeros A(i,j) below the diagonal, in column j of A */
        for &i in ai.range(ap[j], ap[j + 1]) {
            if i > j {
                /* j is a descendant of i in etree(A) */
                anz += 1;
                process_edge::<ROWCOUNT>(j, i, k, w, rowcount);
            }
        }
        /* update SetParent: UNION (j, Parent [j]) */
        finalize_node(j, parent, w.setparent);
    }
    anz
}

/// `cholmod_rowcolcounts.c:163-176` — `UNION (p, Parent [p])`.
#[inline]
fn finalize_node(p: i64, parent: &Ws, setparent: &mut Ws) {
    /* all nodes in the SetParent tree rooted at p now have as their final
     * root the node Parent [p].  This computes UNION (p, Parent [p]) */
    if parent[p] != EMPTY {
        setparent[p] = parent[p];
    }
}

/// What `cholmod_rowcolcounts` leaves in `Common` besides its array outputs.
#[derive(Debug, Clone, Copy, Default)]
pub struct RowColCounts {
    /// `Common->anz` — entries in `triu(A)` including the diagonal
    /// (`:445`). Overwrites whatever `cholmod_amd` left there.
    pub anz: f64,
    /// `Common->lnz` — `Σ ColCount[j]`, i.e. nnz(L) including the diagonal.
    pub lnz: f64,
    /// `Common->fl` — `Σ ColCount[j]²` (`:517-524`).
    pub fl: f64,
}

/// Row and column counts of `L` where `LL' = A` — `cholmod_rowcolcounts.c`.
///
/// `A` must be symmetric **lower** (`stype < 0`): upstream rejects the upper
/// form (`:223-228`), which is the mirror image of [`etree`]'s requirement.
/// `parent` and `post` are the outputs of [`etree`] and an unweighted
/// [`postorder`] of it.
///
/// Fills `colcount`, and `first`/`level` — `Level` is the caller's workspace
/// upstream but is computed unconditionally, and `cholmod_analyze` reuses both
/// buffers afterwards. `row_count`, when given, gets nnz per row of `L`
/// including the diagonal; `cholmod_analyze` never asks for it.
#[allow(clippy::too_many_arguments)]
pub fn rowcolcounts(
    a: &Sparse,
    parent: &[i64],
    post: &[i64],
    row_count: Option<&mut [i64]>,
    colcount: &mut [i64],
    first: &mut [i64],
    level: &mut [i64],
    work: &mut WorkRef<'_>,
) -> Result<RowColCounts, SymbolicError> {
    if a.stype >= 0 {
        return Err(SymbolicError::Unsymmetric);
    }
    let nrow = a.n;

    /* SetParent = Iwork [0..n), PrevNbr = Iwork [n..2n), PrevLeaf = Flag */
    let (setparent_buf, prevnbr_buf) = work.iwork[..2 * nrow].split_at_mut(nrow);
    let prevleaf_buf = &mut work.flag;

    let ap = Ws::new_ref(&a.p);
    let ai = Ws::new_ref(&a.i);
    let parent_w = Ws::new_ref(parent);
    let post_w = Ws::new_ref(post);

    let (mut empty, has_rows) = (Vec::new(), row_count.is_some());
    let rows = row_count.unwrap_or(&mut empty);

    let mut w = RowColWork {
        first: Ws::new(first),
        level: Ws::new(level),
        prevnbr: Ws::new(prevnbr_buf),
        prevleaf: Ws::new(prevleaf_buf),
        setparent: Ws::new(setparent_buf),
        colcount: Ws::new(colcount),
    };

    /* find the first descendant and level of each node in the tree.
     * First [i] = k if the postordering of first descendent of node i is k
     * Level [i] = length of path from node i to the root (Level [root] = 0) */
    for i in 0..nrow {
        w.first[i] = EMPTY;
    }

    /* postorder traversal of the etree */
    for k in 0..nrow as i64 {
        /* node i of the etree is the kth node in the postordered etree */
        let i = post_w[k];

        /* i is a leaf if First [i] is still EMPTY.
         * ColCount [i] starts at 1 if i is a leaf, zero otherwise */
        w.colcount[i] = if w.first[i] == EMPTY { 1 } else { 0 };

        /* traverse the path from node i to the root, stopping if we find a
         * node r whose First [r] is already defined. */
        let mut len = 0i64;
        let mut r = i;
        while r != EMPTY && w.first[r] == EMPTY {
            w.first[r] = k;
            len += 1;
            r = parent_w[r];
        }
        if r == EMPTY {
            /* we hit a root node, the level of which is zero */
            len -= 1;
        } else {
            /* we stopped at node r, where Level [r] is already defined */
            len += w.level[r];
        }
        /* re-traverse the path from node i to r; set the level of each node */
        let mut s = i;
        while s != r {
            w.level[s] = len;
            len -= 1;
            s = parent_w[s];
        }
    }

    /* compute the row counts and node weights */
    if has_rows {
        for x in rows.iter_mut() {
            *x = 1;
        }
    }
    for i in 0..nrow as i64 {
        w.prevleaf[i] = EMPTY;
        w.prevnbr[i] = EMPTY;
        w.setparent[i] = i; /* every node is in its own set, by itself */
    }

    /* symmetric case: LL' = A.
     * also determine the number of entries in triu(A) */
    let anz = if has_rows {
        sym_pass::<true>(nrow, ap, ai, post_w, parent_w, &mut w, rows)
    } else {
        sym_pass::<false>(nrow, ap, ai, post_w, parent_w, &mut w, rows)
    };

    /* finish computing the column counts */
    for j in 0..nrow as i64 {
        let par = parent_w[j];
        if par != EMPTY {
            /* add the ColCount of j to its parent */
            let c = w.colcount[j];
            w.colcount[par] += c;
        }
    }

    /* clear workspace: Flag is shared, and PrevLeaf left it dirty */
    work.reset_flag();

    /* flop count and nnz(L) for subsequent LL' numerical factorization.
     * use double to avoid integer overflow.  lnz cannot be NaN. */
    let (mut lnz, mut fl) = (0.0f64, 0.0f64);
    for j in 0..nrow {
        let ff = colcount[j] as f64;
        lnz += ff;
        fl += ff * ff;
    }

    Ok(RowColCounts {
        anz: anz as f64,
        lnz,
        fl,
    })
}

/* ========================================================================= */
/* === cholmod_analyze ===================================================== */
/* ========================================================================= */

/// Which fill-reducing ordering [`analyze`] should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// `CHOLMOD_NATURAL` — no permutation. The weighted postorder still runs,
    /// so the stored permutation is the postorder itself and upstream relabels
    /// the result `CHOLMOD_POSTORDERED` (`cholmod_analyze.c:875-878`).
    Natural,
    /// `CHOLMOD_AMD`.
    Amd,
    /// `CHOLMOD_METIS` — `METIS_NodeND` through
    /// [`super::metis_order::cholmod_metis`].
    Metis,
    /// `CHOLMOD_POSTORDERED` — what [`Ordering::Natural`] becomes once the
    /// weighted postorder has been composed in. Never requested; it is only an
    /// output. The distinction is load-bearing: `permute_matrices` tests
    /// `ordering == CHOLMOD_NATURAL` to decide whether to apply `L->Perm`, so a
    /// factor left labelled natural would be re-analyzed unpermuted.
    Postordered,
}

/// Which orderings [`analyze`] tries — upstream's `Common->nmethods` together
/// with the `Common->method[]` it implies (`cholmod_analyze.c:432-472`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Method {
    /// `Common->nmethods == 0`, CHOLMOD's own default: try every ordering in
    /// [`DEFAULT_METHODS`] and keep the one with the smallest `lnz`.
    #[default]
    Default,
    /// `Common->nmethods == 1` — this ordering and no other. Upstream's
    /// `default_strategy` is then false, so the METIS break check at
    /// `cholmod_analyze.c:767-781` never runs and
    /// [`Symbolic::metis_would_be_tried`] stays false.
    Pinned(Ordering),
}

/// The default strategy's method list — now exactly upstream's.
///
/// `cholmod_analyze.c:452-455` sets `{CHOLMOD_GIVEN, CHOLMOD_AMD,
/// CHOLMOD_METIS}` (or `NESDIS` for `Common->default_nesdis`), and
/// `CHOLMOD_GIVEN` is skipped whenever `UserPerm` is `NULL` (`:609-613`). This
/// port has no user-permutation argument, so its effective list is the same
/// two methods upstream runs.
///
/// This slot used to hold `CHOLMOD_NATURAL`, because the port had no METIS.
/// That substitution was the one documented deviation in the candidate set, and
/// it was not free: on the SAC conformal system at `conformal_jump = 1` it cost
/// 523.0M `nnz(L)` against METIS's 380.5M, i.e. 3.0x on the numeric
/// factorization. Natural's own case — `hea.models.gmm`'s crossed
/// random-effects matrices, where it beat AMD outright — is now covered by
/// METIS, which beats both there.
pub const DEFAULT_METHODS: [Ordering; 2] = [Ordering::Amd, Ordering::Metis];

/// What `permute_matrices` hands back (`cholmod_analyze.c:172-176`).
///
/// Upstream fills four out-parameters, but only `A1` and `A2` are matrices it
/// allocates: `S` and `F` are *aliases*, each pointing at `A`, `A1` or `A2`.
/// Keeping that distinction is the whole point of the type — on the natural
/// path one of `S`/`F` is `A` itself, so only one transpose gets built.
struct Permuted {
    a1: Option<Sparse>,
    a2: Option<Sparse>,
    /// `S`, the upper form `cholmod_etree` needs.
    s: Handle,
    /// `F`, the lower form `cholmod_rowcolcounts` needs.
    f: Handle,
}

/// Which of the three matrices an alias points at.
#[derive(Clone, Copy)]
enum Handle {
    A,
    A1,
    A2,
}

impl Permuted {
    fn get<'a>(&'a self, a: &'a Sparse, h: Handle) -> &'a Sparse {
        match h {
            Handle::A => a,
            Handle::A1 => self.a1.as_ref().expect("A1 was not built"),
            Handle::A2 => self.a2.as_ref().expect("A2 was not built"),
        }
    }
}

/// `cholmod_analyze` transposes for the pattern alone — every `ptranspose`
/// inside `permute_matrices` passes `mode = 0` (`cholmod_analyze.c:199,211,...`).
const PATTERN: bool = false;

/// `permute_matrices` (`cholmod_analyze.c:161-280`) for a symmetric `A`, with
/// `do_rowcolcounts` true — this port's only caller always wants the column
/// counts, so the `F`-not-needed arm is never taken.
fn permute_matrices(
    a: &Sparse,
    ordering: Ordering,
    perm: &[i64],
    work: &mut WorkRef<'_>,
) -> Permuted {
    let mut p = Permuted {
        a1: None,
        a2: None,
        s: Handle::A,
        f: Handle::A,
    };
    if matches!(ordering, Ordering::Natural) {
        /* natural ordering of A */
        if a.stype < 0 {
            /* symmetric lower case: A already in lower form, so S=A' */
            p.a2 = Some(ptranspose(a, PATTERN, None, work));
            p.f = Handle::A;
            p.s = Handle::A2;
        } else {
            /* symmetric upper case: F = pattern of triu (A)', S = A */
            p.a1 = Some(ptranspose(a, PATTERN, None, work));
            p.f = Handle::A1;
            p.s = Handle::A;
        }
    } else {
        /* A is permuted */
        if a.stype < 0 {
            /* symmetric lower case: S = tril (A (p,p))' and F = S' */
            p.a2 = Some(ptranspose(a, PATTERN, Some(perm), work));
            p.s = Handle::A2;
            p.a1 = Some(ptranspose(p.a2.as_ref().unwrap(), PATTERN, None, work));
            p.f = Handle::A1;
        } else {
            /* symmetric upper case: F = triu (A (p,p))' and S = F' */
            p.a1 = Some(ptranspose(a, PATTERN, Some(perm), work));
            p.f = Handle::A1;
            p.a2 = Some(ptranspose(p.a1.as_ref().unwrap(), PATTERN, None, work));
            p.s = Handle::A2;
        }
    }
    p
}

/// The elimination tree, its postordering and the column counts of `L`, for a
/// matrix and a permutation of it — `cholmod_analyze_ordering`
/// (`cholmod_analyze.c:297-357`).
///
/// `first` and `level` are `cholmod_postorder`'s workspace upstream; they are
/// taken by reference here for the same reason, because `cholmod_analyze`
/// reuses them for the composition afterwards.
pub fn analyze_ordering(
    a: &Sparse,
    ordering: Ordering,
    perm: &[i64],
    parent: &mut [i64],
    post: &mut [i64],
    colcount: &mut [i64],
    first: &mut [i64],
    level: &mut [i64],
    work: &mut WorkRef<'_>,
) -> Result<RowColCounts, SymbolicError> {
    let n = a.n;
    /* permute A according to Perm */
    let p = permute_matrices(a, ordering, perm, work);
    let (s, f) = (p.get(a, p.s), p.get(a, p.f));

    /* find etree of S (symmetric upper/lower case) */
    etree(s, parent, work)?;

    /* postorder the etree (required by cholmod_rowcolcounts) */
    let k = postorder(parent, n, None, post, work);
    if k != n {
        return Err(SymbolicError::NotATree { got: k, want: n });
    }

    /* analyze LL'=S */
    rowcolcounts(f, parent, post, None, colcount, first, level, work)
}

/// What [`analyze`] computed. The three arrays are in the *final* ordering,
/// i.e. after the weighted postorder has been composed into them.
#[derive(Debug, Clone)]
pub struct Symbolic {
    /// `L->Perm` — the fill-reducing ordering composed with the weighted
    /// postorder. This is `scikit-sparse`'s `F.perm`.
    pub perm: Vec<i64>,
    /// `L->ColCount` — nnz in each column of `L`, including the diagonal.
    pub colcount: Vec<i64>,
    /// `Lparent` — the elimination tree. Not stored in `L` upstream; kept
    /// because the supernodal analysis needs it.
    pub parent: Vec<i64>,
    /// The weighted postorder that was composed in, `EMPTY`-free and a
    /// permutation, or all zeros if it was skipped.
    pub post: Vec<i64>,
    /// `Common->fl` and `Common->lnz` after the analysis: `Σ ColCount[j]²` and
    /// `Σ ColCount[j]`. `cholmod_analyze` picks supernodal over simplicial when
    /// `fl / lnz >= 40`.
    pub fl: f64,
    pub lnz: f64,
    /// `Common->anz` — nnz of the stored triangle of the permuted `A`,
    /// diagonal included.
    pub anz: f64,
    /// The ordering the trial loop selected, after upstream's relabel of
    /// [`Ordering::Natural`] to [`Ordering::Postordered`] — `L->ordering`.
    pub ordering: Ordering,
    /// AMD's own statistics, whenever AMD *ran* — which under
    /// [`Method::Default`] is always, even on a matrix where another method
    /// won. They are the estimates the METIS break check is taken on, and
    /// deliberately not the exact counts in `fl`/`lnz`.
    pub amd: Option<AmdInfo>,
    /// Whether the default strategy went on past AMD —
    /// `!((fl < 500*lnz) || (lnz < 5*anz))` on AMD's estimates
    /// (`cholmod_analyze.c:767-781`). Upstream's third method is METIS and this
    /// port's is natural ([`DEFAULT_METHODS`]), so this doubles as the flag for
    /// where the two candidate sets can differ. False whenever
    /// `Common->nmethods` is 1, i.e. whenever the caller pinned the ordering.
    pub metis_would_be_tried: bool,
}

/// `cholmod_analyze` for a symmetric `A` — `cholmod_analyze.c:384-909`.
///
/// Runs the requested ordering, computes the elimination tree, postorders it,
/// counts the columns of `L`, then postorders the tree a second time weighted
/// by those counts and composes that into all three outputs. The composition
/// (`:828-880`) is not cosmetic: it is what makes the stored permutation differ
/// from the ordering routine's raw output, and therefore what makes
/// [`Symbolic::perm`] comparable to `scikit-sparse`'s `F.perm` where
/// [`amd::cholmod_amd`]'s is not.
///
/// `method` is upstream's `Common->nmethods`: [`Method::Default`] runs the
/// trial loop over [`DEFAULT_METHODS`] and keeps the smallest `lnz`,
/// [`Method::Pinned`] runs exactly one ordering.
pub fn analyze(
    n: usize,
    indptr: &[i64],
    indices: &[i64],
    stype: i32,
    method: Method,
    width: IntWidth,
) -> Result<Symbolic, SymbolicError> {
    if stype == 0 {
        return Err(SymbolicError::Unsymmetric);
    }
    let nz = validate_csc(n, indptr, indices)?;
    let a = Sparse {
        n,
        p: indptr[..n + 1].to_vec(),
        i: indices[..nz].to_vec(),
        x: Vec::new(),
        numeric: false,
        stype,
        sorted: columns_are_sorted(n, indptr, indices),
    };
    analyze_sparse(&a, method, width)
}

/// [`analyze`] for a matrix the caller already holds.
///
/// The two exist separately because a caller that goes on to factorize has the
/// matrix in hand and should not pay to have it copied twice: `cholmod_analyze`
/// takes the same `cholmod_sparse *A` that `cholmod_factorize` does.
pub fn analyze_sparse(
    a: &Sparse,
    method: Method,
    width: IntWidth,
) -> Result<Symbolic, SymbolicError> {
    if a.stype == 0 {
        return Err(SymbolicError::Unsymmetric);
    }
    let n = a.n;

    /* allocate workspace.  Note: enough space needs to be allocated here so
     * that routines called by cholmod_analyze do not reallocate the space. */
    let mut work = Work::new(n);

    /* Upstream carves First and Level out of the last 4n Ints of Iwork because
     * the kernels below use only the first 2n (`:515-520`).  The ordering
     * routines may use all 6n, and here they run *inside* the method loop, so
     * these get buffers of their own instead — strictly more workspace, and
     * trivially disjoint from what the kernels scratch in. */
    let mut first = vec![0i64; n];
    let mut level = vec![0i64; n];

    /* the candidate the loop is working on: upstream's Perm/Parent/ColCount
     * workspace, against the Lperm/Lparent/Lcolcount that hold the best so
     * far.  Post is workspace for both — the composition below recomputes it
     * from the winner's tree. */
    let mut perm = vec![0i64; n];
    let mut parent = vec![EMPTY; n];
    let mut colcount = vec![0i64; n];
    let mut post = vec![0i64; n];

    let mut lperm = vec![0i64; n];
    let mut lparent = vec![EMPTY; n];
    let mut lcolcount = vec![0i64; n];

    let default_strategy = matches!(method, Method::Default);
    let pinned: [Ordering; 1];
    let methods: &[Ordering] = match method {
        Method::Default => &DEFAULT_METHODS,
        Method::Pinned(o) => {
            pinned = [o];
            &pinned
        }
    };

    /* Common->selected, Common->method[].fl/lnz and lnz_best.  amd_ran is not
     * upstream's — it keeps AMD's estimates reportable even when another
     * method wins, since they are what the break check was taken on. */
    let mut selected = None;
    let mut lnz_best = 0.0;
    let mut skip_best = false;
    let mut best_ordering = Ordering::Natural;
    let mut best_fl = 0.0;
    let mut amd_ran = None;
    let mut anz = 0.0;
    let mut metis_would_be_tried = false;
    let mut failure = None;

    /* try all the requested ordering options — cholmod_analyze.c:554-782.
     * Upstream's trailing `method == nmethods` iteration is the AMD backup, and
     * it is unreachable here: it needs `amd_backup`, which the default strategy
     * clears at `:456` and which a pinned AMD or NATURAL leaves false at
     * `:467-471`. */
    for (m, &ordering) in methods.iter().enumerate() {
        let mut skip_analysis = false;

        /* find the fill-reducing permutation.  The ordering routines may use
         * all 6n of Iwork, since nothing else's contents are needed across
         * them. */
        match ordering {
            Ordering::Natural | Ordering::Postordered => {
                for (k, p) in perm.iter_mut().enumerate() {
                    *p = k as i64;
                }
            }
            Ordering::Amd => {
                match amd::cholmod_amd(
                    n,
                    &a.p,
                    &a.i,
                    a.stype,
                    DEFAULT_DENSE,
                    DEFAULT_AGGRESSIVE,
                    width,
                    &mut work.all(),
                ) {
                    Ok((p, info)) => {
                        perm.copy_from_slice(&p);
                        anz = info.anz;
                        amd_ran = Some(info);
                    }
                    Err(e) => {
                        /* method failed; clear status and try the next
                         * (`:703-709`) */
                        failure.get_or_insert(SymbolicError::from(e));
                        continue;
                    }
                }
                skip_analysis = true;
            }
            Ordering::Metis => match metis_order::cholmod_metis(n, &a.p, &a.i, a.stype) {
                Ok((p, _anz)) => perm.copy_from_slice(&p),
                Err(e) => {
                    failure.get_or_insert(SymbolicError::from(e));
                    continue;
                }
            },
        }

        /* analyze the ordering.  AMD is exempt: cholmod_amd has already left
         * its own fl/lnz estimates in Common, and the exact counts are wanted
         * only for the ordering that wins (`:715-725`, `:814-822`). */
        let (fl, lnz) = if skip_analysis {
            let info = amd_ran.expect("cholmod_amd sets its Info before returning");
            (info.fl(n), info.lnz(n))
        } else {
            match analyze_ordering(
                a,
                ordering,
                &perm,
                &mut parent,
                &mut post,
                &mut colcount,
                &mut first,
                &mut level,
                &mut work.all(),
            ) {
                Ok(counts) => {
                    anz = counts.anz;
                    (counts.fl, counts.lnz)
                }
                Err(e) => {
                    failure.get_or_insert(e);
                    continue;
                }
            }
        };

        /* pick the best method — fl.pt. compare, but lnz can never be NaN
         * (`:731-761`) */
        if selected.is_none() || lnz < lnz_best {
            selected = Some(m);
            best_ordering = ordering;
            lnz_best = lnz;
            best_fl = fl;
            lperm.copy_from_slice(&perm);
            /* save the results of analyze_ordering, if it was called */
            skip_best = skip_analysis;
            if !skip_analysis {
                lcolcount.copy_from_slice(&colcount);
                lparent.copy_from_slice(&parent);
            }
        }

        /* determine if the third method is to be skipped (`:763-781`).  AMD
         * found an ordering with less than 500 flops per nonzero in L, or one
         * with a fill-in ratio of less than 5?  Then it is unlikely another
         * method will do better.  All three terms are AMD's own, cholmod_amd
         * being the only thing that has run. */
        if default_strategy && ordering == Ordering::Amd {
            metis_would_be_tried = !((fl < 500.0 * lnz) || (lnz < 5.0 * anz));
            if !metis_would_be_tried {
                break;
            }
        }
    }

    if selected.is_none() {
        /* all methods failed (`:787-804`).  Upstream reports the worst status
         * any of them set; the only method it can skip *without* an error is
         * CHOLMOD_GIVEN with a NULL UserPerm, which this port does not have,
         * so something always recorded a reason. */
        return Err(failure.expect("a method was skipped without recording why"));
    }

    /* do the analysis for AMD, if skipped (`:806-822`).  This overwrites
     * Common->fl and Common->lnz with the exact counts, so the supernodal
     * switch downstream never sees AMD's estimate. */
    let mut ordering = best_ordering;
    if skip_best {
        let counts = analyze_ordering(
            a,
            ordering,
            &lperm,
            &mut lparent,
            &mut post,
            &mut lcolcount,
            &mut first,
            &mut level,
            &mut work.all(),
        )?;
        anz = counts.anz;
        best_fl = counts.fl;
        lnz_best = counts.lnz;
    }

    /* postorder the etree, weighted by the column counts, and combine the
     * fill-reducing ordering with it */
    let mut w = work.all();
    let (first, level) = (&mut first[..], &mut level[..]);
    let k = postorder(&lparent, n, Some(&lcolcount), &mut post, &mut w);
    if k == n {
        /* use First and Level as workspace */
        let (wi, invpost) = (first, level);

        for k in 0..n {
            wi[k] = lperm[post[k] as usize];
        }
        lperm.copy_from_slice(&wi[..n]);

        for k in 0..n {
            wi[k] = lcolcount[post[k] as usize];
        }
        lcolcount.copy_from_slice(&wi[..n]);

        for k in 0..n {
            invpost[post[k] as usize] = k as i64;
        }

        /* updated Lparent needed only for supernodal case */
        for newchild in 0..n {
            let oldchild = post[newchild] as usize;
            let oldparent = lparent[oldchild];
            wi[newchild] = if oldparent == EMPTY {
                EMPTY
            } else {
                invpost[oldparent as usize]
            };
        }
        lparent.copy_from_slice(&wi[..n]);

        /* L is now postordered, no longer in natural ordering */
        if ordering == Ordering::Natural {
            ordering = Ordering::Postordered;
        }
    }

    Ok(Symbolic {
        perm: lperm,
        colcount: lcolcount,
        parent: lparent,
        post,
        fl: best_fl,
        lnz: lnz_best,
        anz,
        ordering,
        amd: amd_ran,
        metis_would_be_tried,
    })
}

/* ========================================================================= */
/* === tests =============================================================== */
/* ========================================================================= */

#[cfg(test)]
mod tests {
    //! Memory safety and structural invariants, run against
    //! [`crate::sparse::testcorpus`] in a build where [`Ws`] still checks its
    //! bounds. Agreement with upstream's C is checked from the Python suite.

    use super::*;
    use crate::sparse::testcorpus::{corpus, triangle_csc};

    fn analyzed(n: usize, edges: &[(usize, usize)], lower: bool, o: Ordering) -> Symbolic {
        by_method(n, edges, lower, Method::Pinned(o))
    }

    fn by_method(n: usize, edges: &[(usize, usize)], lower: bool, m: Method) -> Symbolic {
        let (indptr, indices) = triangle_csc(n, edges, lower);
        let stype = if lower { -1 } else { 1 };
        analyze(n, &indptr, &indices, stype, m, IntWidth::I64)
            .unwrap_or_else(|e| panic!("n = {n}, stype {stype}: {e}"))
    }

    fn assert_is_permutation(p: &[i64], n: usize, what: &str) {
        assert_eq!(p.len(), n, "{what}: wrong length");
        let mut seen = vec![false; n];
        for &k in p {
            assert!(k >= 0 && (k as usize) < n, "{what}: {k} is out of range");
            assert!(!seen[k as usize], "{what}: repeats {k}");
            seen[k as usize] = true;
        }
    }

    /// The whole point of the module: every subscript these kernels form, under
    /// a build where [`Ws`] still checks them, for both triangles and both
    /// orderings.
    #[test]
    fn analyze_never_indexes_out_of_bounds() {
        for (name, n, edges) in corpus() {
            for lower in [true, false] {
                for o in [Ordering::Natural, Ordering::Amd] {
                    let s = analyzed(n, &edges, lower, o);
                    assert_is_permutation(&s.perm, n, name);
                    assert_is_permutation(&s.post, n, name);

                    /* an elimination tree points strictly upwards once it is
                     * postordered, which is what makes the numeric factorization
                     * a single left-to-right sweep */
                    for j in 0..n {
                        let p = s.parent[j];
                        assert!(
                            p == EMPTY || (p > j as i64 && (p as usize) < n),
                            "{name}: parent[{j}] = {p} is not above {j}"
                        );
                        assert!(
                            s.colcount[j] >= 1 && s.colcount[j] <= (n - j) as i64,
                            "{name}: colcount[{j}] = {} is impossible",
                            s.colcount[j]
                        );
                    }
                    assert_eq!(s.lnz, s.colcount.iter().sum::<i64>() as f64);
                }
            }
        }
    }

    /// `stype` picks which triangle is stored, not which matrix it is, so the
    /// analysis of a matrix must not depend on it.
    #[test]
    fn both_triangles_of_one_matrix_analyze_alike() {
        for (name, n, edges) in corpus() {
            for o in [Ordering::Natural, Ordering::Amd] {
                let lo = analyzed(n, &edges, true, o);
                let up = analyzed(n, &edges, false, o);
                assert_eq!(lo.perm, up.perm, "{name}: perm differs by stype");
                assert_eq!(
                    lo.colcount, up.colcount,
                    "{name}: colcount differs by stype"
                );
                assert_eq!(lo.parent, up.parent, "{name}: parent differs by stype");
                assert_eq!(lo.anz, up.anz, "{name}: anz differs by stype");
            }
        }
    }

    /// The trial loop selects, it does not invent: whatever it returns is
    /// exactly what pinning the ordering it reports would have returned, and
    /// it is never worse than pinning AMD.
    ///
    /// The two halves are one claim each. `metis_would_be_tried` decides
    /// whether the loop looked past AMD at all, so when it is false the answer
    /// must be AMD's *bit for bit* — that is what keeps the default cheap on
    /// the matrices upstream would not have looked past AMD on either. When it
    /// is true, METIS was tried, and the selection rule is smallest `lnz`.
    #[test]
    fn the_trial_loop_selects_and_never_invents() {
        for (name, n, edges) in corpus() {
            let best = by_method(n, &edges, true, Method::Default);
            let amd = analyzed(n, &edges, true, Ordering::Amd);

            let same_as = match best.ordering {
                Ordering::Amd => &amd,
                Ordering::Metis => &analyzed(n, &edges, true, Ordering::Metis),
                /* natural is relabelled postordered on the way out */
                Ordering::Postordered => &analyzed(n, &edges, true, Ordering::Natural),
                Ordering::Natural => panic!("{name}: natural was not relabelled"),
            };
            assert_eq!(best.perm, same_as.perm, "{name}: perm is not the winner's");
            assert_eq!(best.colcount, same_as.colcount, "{name}: colcount");
            assert_eq!(best.parent, same_as.parent, "{name}: parent");
            assert_eq!(best.lnz, same_as.lnz, "{name}: lnz");

            if best.metis_would_be_tried {
                assert!(
                    best.lnz <= amd.lnz,
                    "{name}: selected lnz {} is worse than AMD's {}",
                    best.lnz,
                    amd.lnz
                );
            } else {
                assert_eq!(
                    best.ordering,
                    Ordering::Amd,
                    "{name}: the loop broke after AMD but did not select it"
                );
            }
        }
    }

    /// Pinning is `Common->nmethods == 1`, and upstream takes the break check
    /// only under the default strategy (`cholmod_analyze.c:767`) — so a pinned
    /// run reports no verdict on it rather than a stale one.
    #[test]
    fn pinning_an_ordering_takes_no_metis_decision() {
        for (name, n, edges) in corpus() {
            for o in [Ordering::Natural, Ordering::Amd] {
                let s = analyzed(n, &edges, true, o);
                assert!(!s.metis_would_be_tried, "{name}: pinned {o:?} took one");
            }
        }
    }

    /// `transpose_sym` is an involution up to `stype`, and the permuted form
    /// has to agree with permuting the unpermuted one. Run with values, so the
    /// `VALUES` instantiation is walked under `debug_assertions` too — that is
    /// what licenses its unchecked indexing.
    #[test]
    fn transpose_sym_round_trips() {
        for (name, n, edges) in corpus() {
            let (indptr, indices) = triangle_csc(n, &edges, true);
            /* a value that identifies its own (i,j), so a misplaced entry is
             * a mismatch rather than a coincidence */
            let x: Vec<f64> = (0..indices.len()).map(|p| p as f64 + 0.5).collect();
            let a = Sparse {
                n,
                p: indptr,
                i: indices,
                x,
                numeric: true,
                stype: -1,
                sorted: true,
            };
            let mut work = Work::new(n);
            let t = transpose_sym(&a, true, None, &mut work.all());
            assert_eq!(t.stype, 1, "{name}");
            assert_eq!(t.x.len(), t.i.len(), "{name}");
            let tt = transpose_sym(&t, true, None, &mut work.all());
            assert_eq!(tt.stype, -1, "{name}");
            assert_eq!((&tt.p, &tt.i, &tt.x), (&a.p, &a.i, &a.x), "{name}");

            /* the permuted transpose is unsorted, so compare as sets */
            let p: Vec<i64> = (0..n as i64).rev().collect();
            let mut got = transpose_sym(&a, true, Some(&p), &mut work.all());
            assert_eq!(got.p[n], a.p[n], "{name}: permuting changed nnz");
            for j in 0..n {
                let (lo, hi) = (got.p[j] as usize, got.p[j + 1] as usize);
                got.i[lo..hi].sort_unstable();
                /* C->stype is +1, and an upper-stored column holds rows <= j */
                for &i in &got.i[lo..hi] {
                    assert!(i <= j as i64, "{name}: C({i},{j}) is not in the upper half");
                }
            }
            /* a value-carrying transpose moves the same multiset of values */
            let (mut before, mut after) = (a.x.clone(), got.x.clone());
            before.sort_by(f64::total_cmp);
            after.sort_by(f64::total_cmp);
            assert_eq!(before, after, "{name}: values were not carried");
        }
    }

    /// The row counts are only reachable through the `ROWCOUNT` instantiation
    /// that `cholmod_analyze` never asks for, so exercise it here: nnz(L) has
    /// to come out the same counted by row as by column.
    #[test]
    fn row_counts_and_column_counts_agree() {
        for (name, n, edges) in corpus() {
            let (indptr, indices) = triangle_csc(n, &edges, false);
            let a = Sparse {
                n,
                p: indptr,
                i: indices,
                x: Vec::new(),
                numeric: false,
                stype: 1,
                sorted: true,
            };
            let mut work = Work::new(n);
            let f = transpose_sym(&a, PATTERN, None, &mut work.all());
            let mut parent = vec![EMPTY; n];
            etree(&a, &mut parent, &mut work.all()).unwrap();
            let mut post = vec![0i64; n];
            assert_eq!(
                postorder(&parent, n, None, &mut post, &mut work.all()),
                n,
                "{name}"
            );

            let mut rows = vec![0i64; n];
            let (mut cols, mut first, mut level) = (vec![0i64; n], vec![0i64; n], vec![0i64; n]);
            let c = rowcolcounts(
                &f,
                &parent,
                &post,
                Some(&mut rows),
                &mut cols,
                &mut first,
                &mut level,
                &mut work.all(),
            )
            .unwrap();
            assert_eq!(
                rows.iter().sum::<i64>(),
                cols.iter().sum::<i64>(),
                "{name}: nnz(L) by row != by column"
            );
            assert_eq!(c.lnz, cols.iter().sum::<i64>() as f64, "{name}");
            /* the diagonal is in every row and column */
            assert!(rows.iter().all(|&r| r >= 1), "{name}");
        }
    }

    /// `stype == 0` is out of scope rather than silently mis-analyzed, and the
    /// two kernels that require a particular triangle say so.
    #[test]
    fn unsupported_stypes_are_rejected() {
        assert!(matches!(
            analyze(
                1,
                &[0, 1],
                &[0],
                0,
                Method::Pinned(Ordering::Amd),
                IntWidth::I64
            ),
            Err(SymbolicError::Unsymmetric)
        ));
        let lower = Sparse {
            n: 1,
            p: vec![0, 1],
            i: vec![0],
            x: Vec::new(),
            numeric: false,
            stype: -1,
            sorted: true,
        };
        let mut work = Work::new(1);
        let mut par = vec![EMPTY; 1];
        assert!(matches!(
            etree(&lower, &mut par, &mut work.all()),
            Err(SymbolicError::Unsymmetric)
        ));
        let upper = Sparse { stype: 1, ..lower };
        let (mut c, mut f, mut l) = (vec![0i64; 1], vec![0i64; 1], vec![0i64; 1]);
        assert!(matches!(
            rowcolcounts(
                &upper,
                &[EMPTY],
                &[0],
                None,
                &mut c,
                &mut f,
                &mut l,
                &mut work.all()
            ),
            Err(SymbolicError::Unsymmetric)
        ));
    }
}
