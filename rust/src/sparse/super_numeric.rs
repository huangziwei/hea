//! Supernodal numeric factorization — the `LL'` that does its arithmetic in
//! dense blocks.
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `CHOLMOD/Supernodal/t_cholmod_super_numeric_worker.c` → [`worker`]
//!   * `CHOLMOD/Supernodal/cholmod_super_numeric.c`          → [`super_numeric`]
//!   * `CHOLMOD/Cholesky/cholmod_factorize.c:172-266`        → [`super_factorize`]
//!
//! **Scope.** `CHOLMOD_REAL` + `CHOLMOD_DOUBLE` and `A->stype != 0`, i.e.
//! upstream's `rd_cholmod_super_numeric_worker` with `F` unused. The other five
//! instantiations (`{real, complex, zomplex} × {double, single}`,
//! `cholmod_super_numeric.c:77-92`) are not built, the same way one of twelve
//! `rowfac` instantiations is built in [`super::numeric`]. The GPU arms are
//! `#if defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE)` and this crate has no
//! CUDA build, so `useGPU` is a compile-time zero and they are not branches
//! here: `Previous`, `gpu_reorder_descendants` and the large/small descendant
//! interleaving do not exist. The OpenMP `#pragma`s are `#ifdef _OPENMP` and
//! parallelize loops whose results do not depend on the order, so dropping them
//! changes nothing but speed.
//!
//! **`S` is the *lower* triangle here**, not the upper one the simplicial
//! `rowfac` takes. `cholmod_factorize_p`'s supernodal branch builds
//! `S = tril (P A P')` (`:216-238`) where its simplicial branch builds
//! `triu`; the worker's `if (i >= k)` filter (`:455`) is what depends on it,
//! and an upper `S` would pass that filter only on the diagonal and factorize
//! the wrong matrix in silence. [`super_factorize`] is what guarantees it.
//!
//! **Supernodal is `LL'` only** (`cholmod_super_numeric.c:239`), so there is no
//! `D` and no `dbound`, and a non-positive pivot is a failure rather than
//! something to clamp.
//!
//! # Two drivers over one supernode kernel
//!
//! Upstream walks the supernodes in index order in one loop. This port keeps
//! that loop — [`worker`], the mechanical port — and adds a second driver,
//! [`par_numeric`], which factorizes independent subtrees of the supernodal
//! elimination tree concurrently. Both call the same [`node_numeric`] for the
//! whole of what one supernode does, so the arithmetic and its order are the
//! same in both and `L->x` comes out bit for bit identical; what differs is
//! only *when* each supernode runs.
//!
//! Two facts make the parallel driver possible, and both are checked rather
//! than assumed:
//!
//! * **The descendant lists are symbolic.** The link-list walk reads only
//!   `L->s`, `L->pi`, `L->px` and `L->super`, so [`Plan`] can compute every
//!   supernode's ordered update list up front. Upstream cannot hoist it because
//!   the walk advances `Lpos [d]` as it reads it, but nothing in it depends on a
//!   value.
//! * **A subtree is a contiguous slice of `L->x`.** `cholmod_analyze`
//!   postorders the elimination tree, so the subtree rooted at supernode `s` is
//!   the index interval `first [s] ..= s`, and `L->px` is a prefix sum — so
//!   handing a subtree its own `&mut [f64]` is `split_at_mut`, not an aliasing
//!   argument. [`Tree::build`] verifies the interval property and refuses the
//!   parallel path if it ever fails to hold.
//!
//! A non-positive pivot sends the whole factorization back through [`worker`]
//! from the top, which re-does its own walk and its `repeat_supernode` replay.
//! That is exact rather than approximate: the first failing supernode has the
//! same inputs under either driver, so "the parallel run found no failure" and
//! "the serial run finds no failure" are the same statement.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as Atomic};
use std::sync::Mutex;

use rayon::prelude::*;

use super::dense::{gemm_nt, potrf_l, syrk_ln_strip, trsm_rlt, SYRK_NB};
use super::numeric::NumericError;
use super::super_symbolic::SuperSymbolic;
use super::symbolic::{permute_sym, Ordering, Sparse, Symbolic};
use super::ws::{Work, Ws, EMPTY};

/// A supernodal `cholmod_factor`: the symbolic pattern plus `L->x`.
///
/// `L->is_ll` is not a field — it is always true here, because
/// `cholmod_super_numeric` sets it unconditionally and supernodal `LDL'` does
/// not exist. `L->is_super` is likewise always true.
#[derive(Debug, Clone)]
pub struct SuperFactor {
    pub n: usize,
    /// `L->Perm` and `L->ColCount`, carried over from the analysis.
    pub perm: Vec<i64>,
    pub colcount: Vec<i64>,
    /// `L->ordering`.
    pub ordering: Ordering,
    /// `L->nsuper`, `L->super`, `L->pi`, `L->px`, `L->s` and the two workspace
    /// bounds.
    pub sym: SuperSymbolic,
    /// `L->x`, `L->xsize` doubles: supernode `s` is the `nsrow`-by-`nscol`
    /// column-major block at `x [px[s] .. px[s+1])`.
    pub x: Vec<f64>,
    /// `L->minor`: `n` if `A` was positive definite, otherwise the column where
    /// it stopped being so.
    pub minor: usize,
    /// `L->xtype != CHOLMOD_PATTERN`: [`SuperFactor::x`] is allocated.
    pub numeric: bool,
    /// Not upstream's: what the tree driver needs, derived from [`Self::sym`]
    /// on the first numeric factorization.
    schedule: Schedule,
}

impl SuperFactor {
    /// The supernodal symbolic factor, as `cholmod_analyze` returns it: the
    /// pattern, with no numeric part yet.
    pub fn new(s: &Symbolic, sym: SuperSymbolic) -> SuperFactor {
        SuperFactor {
            n: sym.n,
            perm: s.perm.clone(),
            colcount: s.colcount.clone(),
            ordering: s.ordering,
            sym,
            x: Vec::new(),
            minor: s.perm.len(),
            numeric: false,
            schedule: Schedule::default(),
        }
    }
}

/// The elimination tree and the update lists, built once and reused by every
/// later numeric factorization of the same `L`.
///
/// This is a function of the symbolic factor and nothing else, which is why it
/// is a field of `L` rather than of [`SuperWork`]: in the workspace it would
/// need a key saying which factor it described, and no cheap key is a proof.
/// Here there is nothing to key — the only way to invalidate it is to modify
/// `L`'s symbolic fields in place between factorizations, which is already
/// outside CHOLMOD's contract.
///
/// It is worth caching rather than rebuilding: it is 4-10 % of the numeric
/// factorization on the benchmark matrices, and repeat factorization against one
/// analysis is the workload this port exists for.
#[derive(Debug, Clone, Default)]
struct Schedule {
    plan: Plan,
    tree: Tree,
    /// Whether the subtrees are index intervals, i.e. whether [`par_numeric`]
    /// may be used at all.
    postordered: bool,
    built: bool,
}

impl Schedule {
    fn ensure(&mut self, sym: &SuperSymbolic, supermap: &Ws) {
        if self.built {
            return;
        }
        self.plan.build(sym, supermap);
        self.postordered = self.tree.build(sym, supermap, &self.plan);
        self.built = true;
    }
}

//------------------------------------------------------------------------------
// per-supernode scratch
//------------------------------------------------------------------------------

/// The scratch one supernode needs while it is being factorized.
///
/// Upstream keeps all four of these in `Common`, because its supernode loop is
/// serial: `Map` is `Common->Flag` (`t_cholmod_super_numeric_worker.c:213`),
/// `RelativeMap` is a slice of `Common->Iwork`, and `C` is the one
/// `cholmod_allocate_dense` per call (`cholmod_super_numeric.c:245`). Here they
/// are per *task*, handed out by [`WorkPool`], because two supernodes can be in
/// flight at once.
///
/// `map` is `EMPTY` when a `TaskWork` is created and holds whatever the last
/// supernode to use it wrote afterwards — the same staleness upstream has, and
/// with the same consequence. `Map [i]` is read for a row `i` of `A`'s column
/// `k` that may not be a row of the supernode; for an `A` whose pattern is
/// contained in `L`'s, which is the only case upstream defines ("the numeric
/// factorization of `A` is undefined" otherwise, `:452-454`), every such `i` is
/// a row of the supernode and the entry was just written.
#[derive(Debug)]
struct TaskWork {
    /// `Map`, size `n`.
    map: Vec<i64>,
    /// `RelativeMap`, size `n`.
    relative_map: Vec<i64>,
    /// `C`, grown to whatever this task's largest update needs rather than to
    /// `L->maxcsize`: the big `C`s belong to the supernodes near the root, and
    /// sizing every task's buffer for those would cost `nthreads` times the
    /// largest one.
    c: Vec<f64>,
    /// One `C` per concurrently-computed descendant; see [`apply_updates`].
    bufs: Vec<Vec<f64>>,
}

impl TaskWork {
    fn new(n: usize) -> TaskWork {
        TaskWork {
            map: vec![EMPTY; n],
            relative_map: vec![0; n],
            c: Vec::new(),
            bufs: Vec::new(),
        }
    }
}

/// The free list [`TaskWork`]s are taken from and returned to.
///
/// A fixed array indexed by thread would not be sound: a rayon worker that
/// blocks on a join can steal another task, so more tasks can be *in progress*
/// than there are threads. The free list simply allocates one more when it is
/// empty, so the pool grows to whatever concurrency actually occurred and then
/// stays there across refactorizations. The lock is held for a `pop` or a
/// `push` and never across any work.
#[derive(Debug, Default)]
struct WorkPool {
    n: usize,
    free: Mutex<Vec<TaskWork>>,
}

impl WorkPool {
    /// Size the pool for an `n`-by-`n` factor, discarding it if `n` changed.
    fn ensure(&mut self, n: usize) {
        if self.n != n {
            self.n = n;
            self.free
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
        }
    }

    fn take(&self) -> TaskWork {
        self.free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_else(|| TaskWork::new(self.n))
    }

    fn give(&self, w: TaskWork) {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).push(w);
    }
}

/// What the last factorization did, so that "the parallel path ran" is a number
/// rather than an assumption.
#[derive(Debug, Default)]
struct Counters {
    /// Updates computed on the batched arm of [`apply_updates`].
    wide: AtomicUsize,
    /// Supernodes whose children were forked rather than walked in order.
    forked: AtomicUsize,
}

/// The `cholmod_dense *C` workspace `cholmod_super_numeric` allocates per call
/// (`:245`), kept across calls instead — plus the thresholds the two parallel
/// arms use.
///
/// A caller refactorizing the same pattern repeatedly should hold one of these,
/// exactly as it holds one [`Work`]. The elimination tree and the update lists
/// are *not* here: they describe one particular symbolic factor, so they live in
/// it (see [`Schedule`]) and need no key saying which.
#[derive(Debug)]
pub struct SuperWork {
    pool: WorkPool,
    /// [`worker`]'s scratch for the update list of the supernode it is on. The
    /// tree driver reads `L`'s [`Schedule`] instead.
    descs: Vec<Desc>,
    counters: Counters,
    /// The batch size, in flops, at which the updates of one supernode go
    /// wide — [`PAR_FLOPS`], except in [`SuperWork::pinned`].
    par_flops: f64,
    /// The subtree size, in flops, at which a supernode's children are forked
    /// rather than run in index order — [`TREE_FLOPS`], except in
    /// [`SuperWork::pinned`].
    tree_flops: f64,
    /// Take [`worker`]'s fused link-list walk rather than [`par_numeric`]. Set
    /// only by [`SuperWork::pinned`]; the fallback itself does not need it,
    /// since it is reached by [`par_numeric`] returning `false`.
    force_serial: bool,
}

impl Default for SuperWork {
    fn default() -> SuperWork {
        SuperWork {
            pool: WorkPool::default(),
            descs: Vec::new(),
            counters: Counters::default(),
            par_flops: PAR_FLOPS,
            tree_flops: TREE_FLOPS,
            force_serial: false,
        }
    }
}

impl SuperWork {
    pub fn new() -> SuperWork {
        SuperWork::default()
    }

    /// The same workspace with both thresholds and the driver pinned, which is
    /// how the corpus tests drive every arm rather than whichever one the
    /// defaults happen to select on a small matrix. All four combinations have
    /// to produce the same `L->x`, bit for bit.
    #[cfg(test)]
    pub(super) fn pinned(par_flops: f64, tree_flops: f64, force_serial: bool) -> SuperWork {
        SuperWork {
            par_flops,
            tree_flops,
            force_serial,
            ..SuperWork::default()
        }
    }

    /// Updates computed on the batched arm, and supernodes that forked.
    #[cfg(test)]
    pub(super) fn counts(&self) -> (usize, usize) {
        (
            self.counters.wide.load(Atomic::Relaxed),
            self.counters.forked.load(Atomic::Relaxed),
        )
    }
}

//------------------------------------------------------------------------------
// the update lists
//------------------------------------------------------------------------------

/// One pending descendant update of the supernode being factorized, read out of
/// the link list before any of them is computed.
///
/// The geometry is a function of `Lpos [d]` at the moment `s` is reached, and
/// the same walk that reads it advances it — so upstream's one fused loop can
/// never have more than one update in flight. Separating the walk from the
/// arithmetic is what makes a batch available. It is the *only* thing that
/// changes: the two kernel calls are the same calls with the same arguments,
/// and the order the results are assembled into `L` is still the link list's.
#[derive(Clone, Copy, Debug)]
struct Desc {
    /// `pdx1` — the first row of `d` that affects `s`, in `L->x`.
    pdx1: i64,
    /// `pdi1` — the same row, in `L->s`.
    pdi1: i64,
    /// `ndrow` — supernode `d`'s leading dimension.
    ndrow: i64,
    /// `ndcol` — the columns of `d`, i.e. the update's `K`.
    ndcol: i64,
    /// `C` is `ndrow2`-by-`ndrow1`, its first `ndrow1` rows triangular.
    ndrow1: i64,
    ndrow2: i64,
}

impl Desc {
    /// The `C` this update writes into.
    #[inline]
    fn csize(&self) -> usize {
        (self.ndrow2 * self.ndrow1) as usize
    }

    /// What the two kernels will do, for the batching and forking decisions
    /// only — not a flop count anyone reports.
    #[inline]
    fn flops(&self) -> f64 {
        2.0 * self.ndrow2 as f64 * self.ndrow1 as f64 * self.ndcol as f64
    }
}

/// Every supernode's pending updates, in the link list's order, computed before
/// any arithmetic runs.
///
/// This is `t_cholmod_super_numeric_worker.c:829-1050` with the arithmetic
/// removed. It reads only the symbolic factor, so the list — and therefore the
/// order every update is assembled in — is fixed by the analysis rather than by
/// whichever thread reaches a supernode first.
///
/// [`worker`] keeps its own fused walk rather than reading this, because on a
/// non-positive-definite matrix upstream *replays* the walk for the failing
/// supernode (`repeat_supernode`, `:301-318`) and the plan assumes success.
#[derive(Debug, Clone, Default)]
struct Plan {
    /// `descs [dptr [s] .. dptr [s+1])` are supernode `s`'s updates.
    dptr: Vec<usize>,
    descs: Vec<Desc>,
    /// `Head`, `Next` and `Lpos`, private to the walk.
    head: Vec<i64>,
    next: Vec<i64>,
    lpos: Vec<i64>,
}

impl Plan {
    #[inline]
    fn of(&self, s: usize) -> &[Desc] {
        &self.descs[self.dptr[s]..self.dptr[s + 1]]
    }

    fn build(&mut self, sym: &SuperSymbolic, supermap: &Ws) {
        let nsuper = sym.nsuper;
        self.dptr.clear();
        self.dptr.reserve(nsuper + 1);
        self.descs.clear();
        self.head.clear();
        self.head.resize(nsuper, EMPTY);
        self.next.clear();
        self.next.resize(nsuper, EMPTY);
        self.lpos.clear();
        self.lpos.resize(nsuper, 0);

        let ls = Ws::new_ref(&sym.s);
        let lpi = Ws::new_ref(&sym.pi);
        let lpx = Ws::new_ref(&sym.px);
        let sup = Ws::new_ref(&sym.sup);
        let head = Ws::new(&mut self.head);
        let next = Ws::new(&mut self.next);
        let lpos = Ws::new(&mut self.lpos);

        for s in 0..nsuper {
            self.dptr.push(self.descs.len());

            let k1 = sup[s];
            let k2 = sup[s + 1];
            let nscol = k2 - k1;
            let psi = lpi[s];
            let psend = lpi[s + 1];
            let nsrow = psend - psi;

            let mut dnext = head[s];
            while dnext != EMPTY {
                let d = dnext;

                /* get the size of supernode d */
                let kd1 = sup[d]; /* d contains cols kd1 to kd2-1 of L */
                let kd2 = sup[d + 1];
                let ndcol = kd2 - kd1; /* # of columns in all of d */
                let pdi = lpi[d]; /* pointer to first row of d in Ls */
                let pdx = lpx[d]; /* pointer to first row of d in Lx */
                let pdend = lpi[d + 1]; /* pointer just past last row of d in Ls */
                let ndrow = pdend - pdi; /* # rows in all of d */

                /* find the range of rows of d that affect rows k1 to k2-1 of s */
                let p = lpos[d]; /* offset of 1st row of d affecting s */
                let pdi1 = pdi + p; /* ptr to 1st row of d affecting s in Ls */
                let pdx1 = pdx + p; /* ptr to 1st row of d affecting s in Lx */

                /* there must be at least one row remaining in d to update s */
                debug_assert!(pdi1 < pdend && ls[pdi1] >= k1 && ls[pdi1] < k2);

                let mut pdi2 = pdi1;
                while pdi2 < pdend && ls[pdi2] < k2 {
                    pdi2 += 1;
                }
                let ndrow1 = pdi2 - pdi1; /* # rows in first part of d */
                let ndrow2 = pdend - pdi1; /* # rows in remaining d */

                /* rows Ls [pdi1 ... pdi2-1] are in the range k1 to k2-1.  Since
                 * d affects s, this set cannot be empty. */
                debug_assert!(pdi1 < pdi2 && pdi2 <= pdend);
                debug_assert!(ndrow2 * ndrow1 <= sym.maxcsize as i64);
                debug_assert!(ndrow2 - ndrow1 >= 0);

                self.descs.push(Desc {
                    pdx1,
                    pdi1,
                    ndrow,
                    ndcol,
                    ndrow1,
                    ndrow2,
                });

                /* prepare this supernode d for its next ancestor */
                dnext = next[d];
                lpos[d] = pdi2 - pdi;
                if lpos[d] < ndrow {
                    /* place d in the link list of its next ancestor */
                    let dancestor = supermap[ls[pdi2]];
                    debug_assert!(dancestor > s as i64 && dancestor < nsuper as i64);
                    next[d] = head[dancestor];
                    head[dancestor] = d;
                }
            }

            /* place this supernode in the link list of its parent */
            if nsrow - nscol > 0 {
                lpos[s] = nscol;
                let sparent = supermap[ls[psi + nscol]];
                debug_assert!(sparent > s as i64 && sparent < nsuper as i64);
                next[s] = head[sparent];
                head[sparent] = s as i64;
            }
            head[s] = EMPTY;
        }
        self.dptr.push(self.descs.len());
    }
}

//------------------------------------------------------------------------------
// the supernodal elimination tree
//------------------------------------------------------------------------------

/// The supernodal elimination tree, and the two things scheduling needs from
/// it: which index interval each subtree is, and how much work is in it.
///
/// `parent [s]` is the supernode holding `Ls [psi + nscol]`, the first row of
/// `s` below its own columns — which is exactly the ancestor
/// `t_cholmod_super_numeric_worker.c:1195` links `s` into, so the tree is the
/// assembly tree and not a second structure that has to agree with it.
#[derive(Debug, Clone, Default)]
struct Tree {
    parent: Vec<i64>,
    /// Children of `s`, ascending, as CSR.
    cptr: Vec<usize>,
    child: Vec<i64>,
    /// The lowest-numbered supernode in `s`'s subtree. Postorder makes the
    /// subtree the interval `first [s] ..= s`.
    first: Vec<usize>,
    /// Flops `s` performs, and flops its whole subtree performs. An estimate
    /// for the forking decision, nothing more: the `syrk` half of each update
    /// is counted as a full `gemm`.
    work: Vec<f64>,
    subwork: Vec<f64>,
    /// The roots, ascending. The tree is a forest whenever `A` is reducible.
    roots: Vec<usize>,
}

impl Tree {
    #[inline]
    fn kids(&self, s: usize) -> &[i64] {
        &self.child[self.cptr[s]..self.cptr[s + 1]]
    }

    /// Build the tree, and report whether the subtrees really are index
    /// intervals.
    ///
    /// They are whenever the elimination tree is postordered, which
    /// `cholmod_analyze_p2` guarantees — but "guaranteed upstream" is not a
    /// receipt for a `split_at_mut`, so it is checked here and the caller takes
    /// [`worker`] if it ever fails.
    fn build(&mut self, sym: &SuperSymbolic, supermap: &Ws, plan: &Plan) -> bool {
        let nsuper = sym.nsuper;
        let ls = Ws::new_ref(&sym.s);
        let lpi = Ws::new_ref(&sym.pi);
        let sup = Ws::new_ref(&sym.sup);

        self.parent.clear();
        self.parent.resize(nsuper, EMPTY);
        self.first.clear();
        self.first.extend(0..nsuper);
        self.work.clear();
        self.work.resize(nsuper, 0.0);
        self.subwork.clear();
        self.subwork.resize(nsuper, 0.0);
        self.roots.clear();
        self.cptr.clear();
        self.cptr.resize(nsuper + 2, 0);
        self.child.clear();
        self.child.resize(nsuper, EMPTY);

        for s in 0..nsuper {
            let nscol = sup[s + 1] - sup[s];
            let nsrow = lpi[s + 1] - lpi[s];
            self.parent[s] = if nsrow > nscol {
                supermap[ls[lpi[s] + nscol]]
            } else {
                EMPTY
            };
            let upd: f64 = plan.of(s).iter().map(Desc::flops).sum();
            let (nscol, nsrow) = (nscol as f64, nsrow as f64);
            self.work[s] = upd + nscol * nscol * nscol / 3.0 + (nsrow - nscol) * nscol * nscol;
        }

        /* children, ascending, by counting sort on the parent */
        for s in 0..nsuper {
            let p = self.parent[s];
            if p != EMPTY {
                self.cptr[p as usize + 2] += 1;
            }
        }
        for s in 0..nsuper {
            self.cptr[s + 2] += self.cptr[s + 1];
        }
        for s in 0..nsuper {
            let p = self.parent[s];
            if p != EMPTY {
                let q = self.cptr[p as usize + 1];
                self.child[q] = s as i64;
                self.cptr[p as usize + 1] += 1;
            } else {
                self.roots.push(s);
            }
        }
        self.cptr.truncate(nsuper + 1);

        /* `parent [s] > s` always, so one ascending pass finishes every
         * subtree before its parent needs it */
        let mut size = vec![1usize; nsuper];
        let mut interval = true;
        for s in 0..nsuper {
            self.subwork[s] += self.work[s];
            interval &= size[s] == s - self.first[s] + 1;
            let p = self.parent[s];
            if p != EMPTY {
                let p = p as usize;
                self.first[p] = self.first[p].min(self.first[s]);
                size[p] += size[s];
                self.subwork[p] += self.subwork[s];
            }
        }
        interval
    }
}

//------------------------------------------------------------------------------
// the dense update of one supernode
//------------------------------------------------------------------------------

/// At most this many `C` buffers are live at once, and at most this many
/// doubles across them. The batch is bounded by its scratch, not only by the
/// thread count: a wide batch of wide updates would otherwise hold tens of
/// megabytes that the serial path never allocates.
const BATCH_MAX: usize = 64;
const BATCH_DOUBLES: usize = 4 << 20;

/// Below this, a batch of updates is not worth a fork and a join.
///
/// The kernels run at tens of GF/s, so this is tens of microseconds of work
/// against a join that costs a few — and it has to be a *batch* total rather
/// than a per-update one, because the batch is what rayon splits.
const PAR_FLOPS: f64 = 5.0e5;

/// Below this much work under a supernode, its children are walked in index
/// order rather than forked.
const TREE_FLOPS: f64 = 1.0e6;

/// A valve, not a tuning knob: past this nesting the children are walked in
/// index order whatever their weight, so a pathological tree cannot recurse the
/// stack away. Measured nesting on the benchmark corpus is 9-23.
const MAX_FORK_DEPTH: u32 = 48;

/// Roughly what one strip of one update should be worth. Small enough that the
/// largest update in a batch is not the batch's floor, large enough that the
/// strip is worth being a task.
const STRIP_FLOPS: f64 = 1.0e5;

/// How many columns of one update's `C` go in a strip — [`STRIP_FLOPS`] worth,
/// rounded up to [`SYRK_NB`] so the strip's block columns are the ones the
/// unsplit call would use.
fn strip_width(g: &Desc) -> usize {
    let per_col = 2.0 * g.ndrow2 as f64 * g.ndcol as f64;
    let cols = (STRIP_FLOPS / per_col).ceil().max(1.0) as usize;
    cols.div_ceil(SYRK_NB) * SYRK_NB
}

/// Columns `j0 .. j0+jn` of `C1 = L1*L1'` stacked over `C2 = L2*L1'` —
/// `:1002-1035`, restricted to a strip.
///
/// `lx` is `L->x [base .. psx)`: every descendant of `s` is stored below `psx`
/// and, when `s` is inside a subtree being factorized on its own, at or above
/// that subtree's `base`. So the truncation is what lets an update read `L`
/// while the assembly writes the supernode — two disjoint slices of one array
/// rather than an aliasing argument. `c` is likewise the strip's own slice: a
/// column of `C` is contiguous, so the strips of one `C` are disjoint slices
/// too, and neither split needs an aliasing argument to be sound.
///
/// A strip changes which thread computes an entry and nothing else: the `gemm`
/// calls are the same calls on the same block columns, so every entry is still
/// accumulated over `l` ascending in one place.
fn update_strip(lx: &[f64], base: i64, g: &Desc, j0: usize, jn: usize, c: &mut [f64]) {
    let (ndrow, ndcol) = (g.ndrow as usize, g.ndcol as usize);
    let (ndrow1, ndrow2) = (g.ndrow1 as usize, g.ndrow2 as usize);
    let pdx1 = (g.pdx1 - base) as usize;

    /* C1 = L1*L1' */
    syrk_ln_strip(
        ndrow1, /* N: L1 is ndrow1-by-ndcol */
        ndcol,  /* K */
        &lx[pdx1..],
        ndrow, /* A, LDA: L1, ndrow */
        c,
        ndrow2, /* C, LDC: C1 */
        j0,
        jn,
    );

    /* C2 = L2*L1' */
    if ndrow2 > ndrow1 {
        gemm_nt(
            ndrow2 - ndrow1, /* M */
            jn,              /* N */
            ndcol,           /* K */
            &lx[pdx1 + ndrow1..],
            ndrow, /* A, LDA: L2 */
            &lx[pdx1 + j0..],
            ndrow, /* B, LDB: L1 */
            &mut c[ndrow1..],
            ndrow2, /* C, LDC: C2 */
        );
    }
}

/// The whole of one descendant's `C`, i.e. [`update_strip`] over every column.
fn update_c(lx: &[f64], base: i64, g: &Desc, c: &mut [f64]) {
    update_strip(lx, base, g, 0, g.ndrow1 as usize, c);
}

/// `Common->chunk` (`Utility/t_cholmod_start.c:44`) — the work one thread must
/// be given before a second is worth adding.
const CHUNK: f64 = 128000.0;

/// `cholmod_nthreads (work, Common)` (`Include/cholmod_internal.h:566-587`).
///
/// `floor (work / chunk)`, clamped to `[1, nthreads_max]`. Deliberately
/// conservative — most of the loops upstream guards with it never reach two
/// threads at these shapes, and that is upstream's answer, not an oversight to
/// correct. Porting the *rule* rather than a threshold of our own is the point:
/// hea's own `PAR_FLOPS`/`TREE_FLOPS`/`STRIP_FLOPS` gate the tree and strip
/// splits, which upstream does not have at all, but a loop upstream *does*
/// thread is gated the way upstream gates it.
#[inline]
fn nthreads(work: f64, max: usize) -> usize {
    (work.max(1.0) / CHUNK).floor().clamp(1.0, max as f64) as usize
}

/// The relative map and the scatter of one `C` into the supernode — `:1037-1050`.
///
/// `sx` is `L->x` from `psx` on, so upstream's `psx + RelativeMap [j] * nsrow`
/// loses its `psx`.
///
/// **Both loops are `#pragma omp parallel for` upstream** (`:897` over `ndrow2`
/// with `if (ndrow2 > 64)`, `:915` over `ndrow1` with `if (ndrow1 > 64)`), and
/// the `j` loop is safe to split for a reason worth stating rather than
/// re-deriving: `px = RelativeMap [j] * nsrow`, so **iteration `j` writes column
/// `RelativeMap [j]` and nothing else**. `RelativeMap` is injective — `Map` is a
/// bijection from `s`'s rows onto `0 .. nsrow`, and `Ls [pdi1 ..]` are distinct
/// — so distinct `j` touch distinct columns and no entry is written twice.
/// Nothing accumulates across `j`, so there is no summation order to preserve
/// and the split is bit-exact by construction, not by argument.
///
/// It is in fact strictly *increasing*, since `Ls` is ascending within a
/// supernode and `Map` is order-preserving on it. That is what lets the columns
/// be carved out of `sx` in one forward pass with `chunks_mut`, with no `unsafe`
/// and no second lookup.
///
/// This is only about the `j` loop of *one* descendant. Two descendants can hit
/// the same entry of `s`, so [`apply_updates`] still assembles them one at a
/// time in link-list order — see its docstring. The two were conflated once, and
/// the outer constraint was wrongly inherited by the inner loop.
fn assemble(
    g: &Desc,
    c: &[f64],
    sx: &mut [f64],
    nsrow: i64,
    ls: &Ws,
    map: &Ws,
    relative_map: &mut Ws,
    nt: usize,
) {
    /* construct relative map to supernode s */
    for i in 0..g.ndrow2 {
        relative_map[i] = map[ls[g.pdi1 + i]];
        debug_assert!(relative_map[i] >= 0 && relative_map[i] < nsrow);
    }

    /* work = ndcol * ndrow2 (`:911`), and `if (ndrow1 > 64)` */
    let work = g.ndcol as f64 * g.ndrow2 as f64;
    if g.ndrow1 > 64 && nthreads(work, nt) > 1 {
        assemble_par(g, c, sx, nsrow, relative_map);
        return;
    }

    let sx = Ws::new(sx);
    let cw = Ws::new_ref(c);
    for j in 0..g.ndrow1 {
        /* cols k1:k2-1 */
        let px = relative_map[j] * nsrow;
        for i in j..g.ndrow2 {
            /* rows k1:n-1 */
            let q = px + relative_map[i];
            sx[q] -= cw[i + g.ndrow2 * j];
        }
    }
}

/// [`assemble`]'s `j` loop with the columns split across threads.
///
/// Same entries, same values, same single `-=` per entry — only the thread that
/// performs it changes. The `i` loop starts at `j`, so the columns are unequal
/// (the first does `ndrow2` rows, the last does one); `par_iter_mut` steals, so
/// they do not need to be equal.
fn assemble_par(g: &Desc, c: &[f64], sx: &mut [f64], nsrow: i64, relative_map: &Ws) {
    /* `RelativeMap` is strictly increasing, so one forward pass over the
     * supernode's columns picks out the ones this descendant writes. */
    let mut cols: Vec<&mut [f64]> = Vec::with_capacity(g.ndrow1 as usize);
    let mut rest = sx;
    let mut at = 0i64;
    for j in 0..g.ndrow1 {
        let want = relative_map[j];
        debug_assert!(want >= at, "RelativeMap must ascend");
        let skip = ((want - at) * nsrow) as usize;
        let (_, tail) = rest.split_at_mut(skip);
        let (col, tail) = tail.split_at_mut(nsrow as usize);
        cols.push(col);
        rest = tail;
        at = want + 1;
    }

    let cw = Ws::new_ref(c);
    cols.par_iter_mut().enumerate().for_each(|(j, col)| {
        let col = Ws::new(col);
        let j = j as i64;
        for i in j..g.ndrow2 {
            col[relative_map[i]] -= cw[i + g.ndrow2 * j];
        }
    });
}

/// Apply every pending update of one supernode, in the link list's order.
///
/// The updates are independent — each reads a different descendant's block of
/// `L` and writes its own `C` — but their assemblies are not, because two
/// descendants can hit the same entry of the supernode. So `C` is what goes
/// wide and *this* loop stays serial and in order, which is what keeps `L->x`
/// bit-for-bit what the one-at-a-time loop produces. It is also why the batch
/// is a batch: the buffers are live simultaneously, so their total size is
/// capped.
///
/// **That constraint stops here.** Inside one [`assemble`] the column loop is
/// disjoint and upstream threads it; this docstring used to be read as
/// "assembly is serial", which quietly made an un-ported `#pragma omp` look
/// like a deliberate bit-exactness decision.
///
/// The parallel and serial arms differ only in *which* `C` buffer each update
/// gets. Both call [`update_c`] and [`assemble`] with the same arguments.
fn apply_updates(
    ctx: &Ctx,
    descs: &[Desc],
    lower: &[f64],
    base: i64,
    sx: &mut [f64],
    nsrow: i64,
    tw: &mut TaskWork,
) {
    let TaskWork {
        map,
        relative_map,
        c,
        bufs,
    } = tw;
    let map = Ws::new(map);
    let relative_map = Ws::new(relative_map);

    let mut i0 = 0;
    while i0 < descs.len() {
        let (mut i1, mut size, mut flops) = (i0, 0usize, 0.0);
        while i1 < descs.len() && i1 - i0 < BATCH_MAX && size < BATCH_DOUBLES {
            size += descs[i1].csize();
            flops += descs[i1].flops();
            i1 += 1;
        }
        let batch = &descs[i0..i1];

        if batch.len() > 1 && flops >= ctx.par_flops {
            if bufs.len() < batch.len() {
                bufs.resize_with(batch.len(), Vec::new);
            }
            for (buf, g) in bufs.iter_mut().zip(batch) {
                if buf.len() < g.csize() {
                    buf.resize(g.csize(), 0.0);
                }
            }
            {
                /* One task per strip rather than per descendant: a supernode's
                 * descendants differ in size by orders of magnitude, and a
                 * batch cannot finish before its largest member does. */
                let mut tasks: Vec<(Desc, usize, usize, &mut [f64])> = Vec::new();
                for (buf, g) in bufs[..batch.len()].iter_mut().zip(batch) {
                    let (ndrow1, ndrow2) = (g.ndrow1 as usize, g.ndrow2 as usize);
                    let width = strip_width(g);
                    let mut rest = &mut buf[..ndrow1 * ndrow2];
                    let mut j0 = 0;
                    while j0 < ndrow1 {
                        let jn = width.min(ndrow1 - j0);
                        let (strip, tail) = rest.split_at_mut(jn * ndrow2);
                        tasks.push((*g, j0, jn, strip));
                        rest = tail;
                        j0 += jn;
                    }
                }
                tasks
                    .par_iter_mut()
                    .for_each(|(g, j0, jn, c)| update_strip(lower, base, g, *j0, *jn, c));
            }
            ctx.counters.wide.fetch_add(batch.len(), Atomic::Relaxed);
            for (buf, g) in bufs.iter().zip(batch) {
                assemble(g, buf, sx, nsrow, ctx.ls, map, relative_map, ctx.nthreads);
            }
        } else {
            for g in batch {
                if c.len() < g.csize() {
                    c.resize(g.csize(), 0.0);
                }
                update_c(lower, base, g, c);
                assemble(g, c, sx, nsrow, ctx.ls, map, relative_map, ctx.nthreads);
            }
        }
        i0 = i1;
    }
}

//------------------------------------------------------------------------------
// one supernode
//------------------------------------------------------------------------------

/// Everything read-only that factorizing a supernode needs, so that both
/// drivers hand [`node_numeric`] the same thing.
struct Ctx<'a> {
    ap: &'a Ws,
    ai: &'a Ws,
    ax: &'a Ws<f64>,
    ls: &'a Ws,
    lpi: &'a Ws,
    lpx: &'a Ws,
    sup: &'a Ws,
    beta: f64,
    n: usize,
    nsuper: usize,
    xsize: usize,
    par_flops: f64,
    tree_flops: f64,
    /// Whether rayon has more than one thread to fork onto.
    threads: bool,
    /// `rayon::current_num_threads()`, passed to the dense panel kernels so a
    /// supernode big enough to be worth it can go wide on its own factorization
    /// — the flops of a crossed random-effects `M` sit almost entirely in one
    /// supernode, where the tree driver has nothing to spread.
    nthreads: usize,
    counters: &'a Counters,
}

/// Everything supernode `s` does to its own block, given that its descendants'
/// blocks are final — `t_cholmod_super_numeric_worker.c:757-1250` less the link
/// list bookkeeping, which is the only part the two drivers do differently.
///
/// `own` is `L->x [px[s] .. px[s+1])`, `lower` is `L->x [base .. px[s])`, and
/// every descendant of `s` lies in `lower`. Returns `dpotrf`'s `info`, having
/// stopped before the triangular solve if it is nonzero.
#[allow(clippy::too_many_arguments)]
fn node_numeric(
    ctx: &Ctx,
    s: usize,
    descs: &[Desc],
    lower: &[f64],
    base: i64,
    own: &mut [f64],
    tw: &mut TaskWork,
    nscol2: i64,
    repeat_supernode: bool,
) -> i64 {
    let k1 = ctx.sup[s]; /* s contains columns k1 to k2-1 of L */
    let k2 = ctx.sup[s + 1];
    let nscol = k2 - k1; /* # of columns in all of s */
    let psi = ctx.lpi[s]; /* pointer to first row of s in Ls */
    let psend = ctx.lpi[s + 1]; /* pointer just past last row of s in Ls */
    let nsrow = psend - psi; /* # of rows in all of s */
    debug_assert_eq!(own.len(), (nsrow * nscol) as usize);

    //--------------------------------------------------------------------------
    // zero the supernode s
    //--------------------------------------------------------------------------

    own.fill(0.0);

    //--------------------------------------------------------------------------
    // construct the scattered Map for supernode s
    //--------------------------------------------------------------------------

    /* If row i is the kth row in s, then Map [i] = k.  Similarly, if
     * column j is the kth column in s, then  Map [j] = k. */
    {
        let map = Ws::new(&mut tw.map);
        for k in 0..nsrow {
            map[ctx.ls[psi + k]] = k;
        }
    }

    //--------------------------------------------------------------------------
    // copy matrix into supernode s (lower triangular part only)
    //--------------------------------------------------------------------------

    {
        let map = Ws::new_ref(&tw.map);
        let sx = Ws::new(own);
        for k in k1..k2 {
            /* copy the kth column of A into the supernode */
            for p in ctx.ap[k]..ctx.ap[k + 1] {
                /* row i of L is located in row Map [i] of s */
                let i = ctx.ai[p];
                if i >= k {
                    /* If the test is false, the numeric factorization of A
                     * is undefined.  The test does not detect all invalid
                     * entries, only some of them. */
                    let imap = map[i];
                    if imap >= 0 && imap < nsrow {
                        /* Lx [Map [i] + pk] = Ax [p] */
                        sx[imap + (k - k1) * nsrow] = ctx.ax[p];
                    }
                }
            }
        }

        /* add beta to the diagonal of the supernode, if nonzero */
        if ctx.beta != 0.0 {
            let mut pk = 0;
            for _ in k1..k2 {
                sx[pk] += ctx.beta;
                pk += nsrow + 1; /* advance to the next diagonal entry */
            }
        }
    }

    //--------------------------------------------------------------------------
    // update supernode s with each pending descendant d
    //--------------------------------------------------------------------------

    apply_updates(ctx, descs, lower, base, own, nsrow, tw);

    //--------------------------------------------------------------------------
    // factorize diagonal block of supernode s in LL'
    //--------------------------------------------------------------------------

    let mut info = potrf_l(
        nscol2 as usize, /* N: nscol2 */
        own,
        nsrow as usize, /* A, LDA: L1, nsrow */
        ctx.nthreads,
    );

    /* if the matrix is not positive definite, the supernode is repeated
     * and only its first nscol_new columns are kept */
    if repeat_supernode {
        /* zero out the rest of this supernode */
        info = 0;
        own[(nsrow * nscol2) as usize..(nsrow * nscol) as usize].fill(0.0);
    }

    if info != 0 {
        return info;
    }

    //--------------------------------------------------------------------------
    // compute the subdiagonal block
    //--------------------------------------------------------------------------

    let nsrow2 = nsrow - nscol2;
    if nsrow2 > 0 {
        /* The current supernode is columns k1 to k2-1 of L.  Let L1 be the
         * diagonal block (factorized by dpotrf/zpotrf above; rows/cols
         * k1:k2-1), and L2 be rows k2:n-1 and columns k1:k2-1 of L.  The
         * triangular system to solve is L2*L1' = S2, where S2 is
         * overwritten with L2.  More precisely, L2 = S2 / L1' is computed. */
        trsm_rlt(
            nsrow2 as usize, /* M: L2 is nsrow2-by-nscol2 */
            nscol2 as usize, /* N */
            own,
            nsrow as usize, /* A/B, LDA/LDB: L1 and L2, nsrow */
            ctx.nthreads,
        );
    }
    0
}

//------------------------------------------------------------------------------
// driver 1: the elimination tree
//------------------------------------------------------------------------------

/// Factorize supernode `t`, which lives inside the slice `lx` covering
/// `L->x [base .. )`.
fn node(
    ctx: &Ctx,
    plan: &Plan,
    failed: &AtomicBool,
    t: usize,
    base: i64,
    lx: &mut [f64],
    tw: &mut TaskWork,
) {
    if failed.load(Atomic::Relaxed) {
        return;
    }
    let psx = ctx.lpx[t];
    let (lower, tail) = lx.split_at_mut((psx - base) as usize);
    let own = &mut tail[..(ctx.lpx[t + 1] - psx) as usize];
    let nscol = ctx.sup[t + 1] - ctx.sup[t];
    if node_numeric(ctx, t, plan.of(t), lower, base, own, tw, nscol, false) != 0 {
        failed.store(true, Atomic::Relaxed);
    }
}

/// Factorize the subtree rooted at supernode `s`, which owns exactly the slice
/// `lx` = `L->x [base .. px[s+1])`.
///
/// Either the children are forked — each gets its own `split_at_mut` piece of
/// `lx`, and `s` itself is the join — or the whole subtree runs in index order.
/// The second arm is not a special case of the first: postorder makes the
/// subtree the interval `first [s] ..= s`, so ascending index order is a
/// topological order, and running it iteratively is what keeps a long chain of
/// supernodes off the stack.
#[allow(clippy::too_many_arguments)]
fn subtree(
    ctx: &Ctx,
    tree: &Tree,
    plan: &Plan,
    pool: &WorkPool,
    failed: &AtomicBool,
    s: usize,
    base: i64,
    lx: &mut [f64],
    depth: u32,
) {
    let kids = tree.kids(s);
    let fork = ctx.threads
        && depth < MAX_FORK_DEPTH
        && kids.len() > 1
        && tree.subwork[s] - tree.work[s] >= ctx.tree_flops;

    if !fork {
        let mut tw = pool.take();
        for t in tree.first[s]..=s {
            node(ctx, plan, failed, t, base, lx, &mut tw);
        }
        pool.give(tw);
        return;
    }

    {
        let mut jobs: Vec<(usize, i64, &mut [f64])> = Vec::with_capacity(kids.len());
        let mut rest = &mut lx[..];
        let mut cur = base;
        for &c in kids {
            let c = c as usize;
            let end = ctx.lpx[c + 1];
            let (piece, tail) = rest.split_at_mut((end - cur) as usize);
            jobs.push((c, cur, piece));
            rest = tail;
            cur = end;
        }
        debug_assert_eq!(cur, ctx.lpx[s]);
        jobs.par_iter_mut()
            .for_each(|(c, b, sl)| subtree(ctx, tree, plan, pool, failed, *c, *b, sl, depth + 1));
    }
    ctx.counters.forked.fetch_add(1, Atomic::Relaxed);

    let mut tw = pool.take();
    node(ctx, plan, failed, s, base, lx, &mut tw);
    pool.give(tw);
}

/// Factorize every supernode, subtrees concurrently. Returns `false` if any
/// supernode was not positive definite, in which case `L->x` is garbage and the
/// caller re-runs [`worker`], which is the only path that implements upstream's
/// `repeat_supernode` replay.
fn par_numeric(ctx: &Ctx, tree: &Tree, plan: &Plan, pool: &WorkPool, lx: &mut [f64]) -> bool {
    let failed = AtomicBool::new(false);
    {
        let mut jobs: Vec<(usize, i64, &mut [f64])> = Vec::with_capacity(tree.roots.len());
        let mut rest = &mut lx[..];
        let mut cur = 0i64;
        for &r in &tree.roots {
            let end = ctx.lpx[r + 1];
            let (piece, tail) = rest.split_at_mut((end - cur) as usize);
            jobs.push((r, cur, piece));
            rest = tail;
            cur = end;
        }
        /* `L->xsize` is `max (1, px [nsuper])` upstream (`:675-676`), so the
         * roots cover `px [nsuper]` and not necessarily all of `L->x`. */
        debug_assert_eq!(cur, ctx.lpx[ctx.nsuper]);
        jobs.par_iter_mut()
            .for_each(|(r, b, sl)| subtree(ctx, tree, plan, pool, &failed, *r, *b, sl, 0));
    }
    !failed.load(Atomic::Relaxed)
}

//------------------------------------------------------------------------------
// driver 2: upstream's supernode loop
//------------------------------------------------------------------------------

/// `t_cholmod_super_numeric_worker.c:133-1250`, real/double, `stype != 0`, no
/// GPU.
///
/// Returns `false` only where upstream returns `Common->status < CHOLMOD_OK`,
/// which for this instantiation means BLAS integer overflow — impossible with
/// `Int = int64_t` and the leading dimensions bounded by `n`, so it is folded
/// into the `Result` of [`super_numeric`] rather than tracked per call.
/// "Not positive definite" is *not* a failure: it is reported through
/// `L->minor`, as in [`super::numeric::rowfac`].
///
/// This is both the reference the tree driver is checked against and the path a
/// non-positive-definite matrix takes, so it keeps its own fused link-list walk
/// rather than reading [`Plan`]: the walk is what `repeat_supernode` replays.
#[allow(clippy::too_many_arguments)]
fn worker(
    ctx: &Ctx,
    lx: &mut [f64],
    minor: &mut usize,
    tw: &mut TaskWork,
    supermap: &Ws,
    next: &mut Ws,
    lpos: &mut Ws,
    next_save: &mut Ws,
    lpos_save: &mut Ws,
    head: &mut Ws,
    descs: &mut Vec<Desc>,
    quick_return_if_not_posdef: bool,
) {
    let nsuper = ctx.nsuper;

    /* clear the Map so that changes in the pattern of A can be detected */
    tw.map.fill(EMPTY);

    /* If the matrix is not positive definite, the supernode s containing the
     * first zero or negative diagonal entry of L is repeated (but factorized
     * only up to just before the problematic diagonal entry). The purpose is
     * to provide MATLAB with [R,p]=chol(A) ; columns 1 to p-1 of L=R' are
     * required, where L(p,p) is the problematic diagonal entry.  The
     * repeat_supernode flag tells us whether this is the repeated supernode.
     * Once supernode s is repeated, the factorization is terminated. */
    let mut repeat_supernode = false;
    let mut nscol_new: i64 = 0;

    //--------------------------------------------------------------------------
    // supernodal numerical factorization
    //--------------------------------------------------------------------------

    let mut s: usize = 0;
    while s < nsuper {
        //----------------------------------------------------------------------
        // get the size of supernode s
        //----------------------------------------------------------------------

        let k1 = ctx.sup[s]; /* s contains columns k1 to k2-1 of L */
        let k2 = ctx.sup[s + 1];
        let nscol = k2 - k1; /* # of columns in all of s */
        let psi = ctx.lpi[s]; /* pointer to first row of s in Ls */
        let psx = ctx.lpx[s]; /* pointer to first row of s in Lx */
        let psend = ctx.lpi[s + 1]; /* pointer just past last row of s in Ls */
        let nsrow = psend - psi; /* # of rows in all of s */

        //----------------------------------------------------------------------
        // save/restore the list of supernodes
        //----------------------------------------------------------------------

        if !repeat_supernode {
            /* Save the list of pending descendants in case s is not positive
             * definite.  Also save Lpos for each descendant d, so that we can
             * find which part of d is used to update s. */
            let mut d = head[s];
            while d != EMPTY {
                lpos_save[d] = lpos[d];
                next_save[d] = next[d];
                d = next[d];
            }
        } else {
            /* restore Lpos from prior failed supernode */
            let mut d = head[s];
            while d != EMPTY {
                lpos[d] = lpos_save[d];
                next[d] = next_save[d];
                d = next[d];
            }
        }

        //----------------------------------------------------------------------
        // read the list of pending descendants d of supernode s
        //----------------------------------------------------------------------

        /* Upstream walks the link list and does the arithmetic in one loop.
         * Here the walk comes first and only records what each update is, so
         * that the updates — which are independent, unlike the walk, which
         * advances Lpos and re-links d as it goes — can be computed together.
         * Nothing in the walk depends on the arithmetic: the geometry is read
         * before Lpos [d] is advanced, exactly as it is upstream, and the
         * ancestor a descendant is re-linked into is always past s.  It is the
         * same walk [`Plan::build`] does for every supernode at once, kept
         * here because only this path replays it. */
        descs.clear();
        let mut dnext = head[s];
        while dnext != EMPTY {
            let d = dnext;

            /* get the size of supernode d */
            let kd1 = ctx.sup[d]; /* d contains cols kd1 to kd2-1 of L */
            let kd2 = ctx.sup[d + 1];
            let ndcol = kd2 - kd1; /* # of columns in all of d */
            let pdi = ctx.lpi[d]; /* pointer to first row of d in Ls */
            let pdx = ctx.lpx[d]; /* pointer to first row of d in Lx */
            let pdend = ctx.lpi[d + 1]; /* pointer just past last row of d in Ls */
            let ndrow = pdend - pdi; /* # rows in all of d */

            /* find the range of rows of d that affect rows k1 to k2-1 of s */
            let p = lpos[d]; /* offset of 1st row of d affecting s */
            let pdi1 = pdi + p; /* ptr to 1st row of d affecting s in Ls */
            let pdx1 = pdx + p; /* ptr to 1st row of d affecting s in Lx */

            /* there must be at least one row remaining in d to update s */
            debug_assert!(pdi1 < pdend && ctx.ls[pdi1] >= k1 && ctx.ls[pdi1] < k2);

            let mut pdi2 = pdi1;
            while pdi2 < pdend && ctx.ls[pdi2] < k2 {
                pdi2 += 1;
            }
            let ndrow1 = pdi2 - pdi1; /* # rows in first part of d */
            let ndrow2 = pdend - pdi1; /* # rows in remaining d */

            /* rows Ls [pdi1 ... pdi2-1] are in the range k1 to k2-1.  Since d
             * affects s, this set cannot be empty. */
            debug_assert!(pdi1 < pdi2 && pdi2 <= pdend);

            /* C = L (k1:n-1, kd1:kd2-1) * L (k1:k2-1, kd1:kd2-1)', except that
             * k1:n-1 refers to all of the rows in L, but many of the rows are
             * all zero.  Supernode d holds columns kd1 to kd2-1 of L.  Nonzero
             * rows in the range k1:k2-1 are in the list Ls [pdi1 ... pdi2-1],
             * of size ndrow1.  Nonzero rows in the range k2:n-1 are in the list
             * Ls [pdi2 ... pdend], of size ndrow2.  Let
             * L1 = L (Ls [pdi1 ... pdi2-1], kd1:kd2-1), and let
             * L2 = L (Ls [pdi2 ... pdend],  kd1:kd2-1).  C is ndrow2-by-ndrow1.
             * Let C1 be the first ndrow1 rows of C and let C2 be the last
             * ndrow2-ndrow1 rows of C.  Only the lower triangular part of C1
             * needs to be computed since C1 is symmetric. */

            debug_assert!(ndrow2 - ndrow1 >= 0);

            descs.push(Desc {
                pdx1,
                pdi1,
                ndrow,
                ndcol,
                ndrow1,
                ndrow2,
            });

            /* prepare this supernode d for its next ancestor */
            dnext = next[d];

            if !repeat_supernode {
                /* If node s is being repeated, Head [dancestor] has already
                 * been cleared (set to EMPTY).  It must remain EMPTY.  The
                 * dancestor will not be factorized since the factorization
                 * terminates at node s. */
                lpos[d] = pdi2 - pdi;
                if lpos[d] < ndrow {
                    /* place d in the link list of its next ancestor */
                    let dancestor = supermap[ctx.ls[pdi2]];
                    debug_assert!(dancestor > s as i64 && dancestor < nsuper as i64);
                    next[d] = head[dancestor];
                    head[dancestor] = d;
                }
            }
        }

        //----------------------------------------------------------------------
        // zero, assemble, update and factorize supernode s
        //----------------------------------------------------------------------

        let nscol2 = if repeat_supernode { nscol_new } else { nscol };
        let info = {
            let (lower, tail) = lx.split_at_mut(psx as usize);
            let own = &mut tail[..(nsrow * nscol) as usize];
            node_numeric(ctx, s, descs, lower, 0, own, tw, nscol2, repeat_supernode)
        };

        //----------------------------------------------------------------------
        // check if the matrix is not positive definite
        //----------------------------------------------------------------------

        if info != 0 {
            /* Matrix is not positive definite.  dpotrf/zpotrf do NOT report an
             * error if the diagonal of L has NaN's, only if it has a zero. */
            *minor = (k1 + info - 1) as usize;

            /* clear the link lists of all subsequent supernodes */
            for ss in s + 1..nsuper {
                head[ss] = EMPTY;
            }

            /* zero this supernode, and all remaining supernodes */
            lx[psx as usize..ctx.xsize].fill(0.0);

            /* If L->minor is zero, then it contains no data, and the
             * factorization is complete (a 1-by-1 matrix, say). */
            if info == 1 || quick_return_if_not_posdef {
                /* If the first column of supernode s contains a zero or
                 * negative diagonal entry, then it is already properly set to
                 * zero.  Also, info will be 1 if integer overflow occured in
                 * the BLAS. */
                head[s] = EMPTY;
                return;
            }

            /* Repeat supernode s, but only factorize it up to but not
             * including the column containing the problematic diagonal entry. */
            repeat_supernode = true;
            nscol_new = info - 1;
            continue;
        }

        //----------------------------------------------------------------------
        // prepare supernode s for its parent
        //----------------------------------------------------------------------

        if nsrow - nscol2 > 0 && !repeat_supernode {
            /* Place this supernode in the link list of its parent. */
            lpos[s] = nscol;
            let sparent = supermap[ctx.ls[psi + nscol]];
            debug_assert!(sparent != s as i64 && sparent > s as i64);
            debug_assert!(sparent < nsuper as i64);
            next[s] = head[sparent];
            head[sparent] = s as i64;
        }

        head[s] = EMPTY; /* link list for supernode s no longer needed */

        if repeat_supernode {
            /* matrix is not positive definite; finished clean-up for supernode
             * containing negative diagonal */
            return;
        }

        s += 1;
    }

    /* success; matrix is positive definite */
    *minor = ctx.n;
}

//------------------------------------------------------------------------------
// entry points
//------------------------------------------------------------------------------

/// `cholmod_super_numeric` (`cholmod_super_numeric.c:96-337`) for a symmetric
/// `A`.
///
/// `a` must be the **lower** triangle of the already-permuted matrix — see the
/// module docs. [`super_factorize`] is the entry point that arranges that;
/// this one is upstream's, taking `S` ready-made.
///
/// `Common->Flag` is *not* borrowed as `Map` here, as it is at
/// `t_cholmod_super_numeric_worker.c:213`: `Map` is per-task (see [`TaskWork`])
/// because two supernodes can be in flight. So `Flag` is untouched, and the
/// `CLEAR_FLAG` this returns through upstream (`:331-332`) has nothing to
/// clear.
pub fn super_numeric(
    a: &Sparse,
    beta: f64,
    l: &mut SuperFactor,
    work: &mut Work,
    cwork: &mut SuperWork,
) -> Result<(), NumericError> {
    let n = l.n;
    let nsuper = l.sym.nsuper;

    if a.stype == 0 {
        return Err(NumericError::Invalid(
            "stype must be nonzero: this port factorizes LL' = A for a \
             symmetric A, not LL' = AA'",
        ));
    }
    if a.n != n {
        return Err(NumericError::Invalid("dimensions of A and L do not match"));
    }
    if !a.numeric {
        return Err(NumericError::Invalid("A has no numeric values"));
    }

    /* allocate workspace in Common: w = 2*nrow + 5*nsuper.  Map and
     * RelativeMap are per-task here (see TaskWork), so the two n-sized slots
     * upstream carves for them go unused; the request is left at upstream's
     * size because Work is shared and only ever grows. */
    work.ensure_iwork(2 * n + 5 * nsuper);
    cwork.pool.ensure(n);

    /* get the current factor L and allocate numerical part, if needed */
    if !l.numeric {
        /* convert to supernodal numeric by allocating L->x */
        l.x = vec![0.0; l.sym.xsize];
        l.numeric = true;
    }
    /* supernodal LDL' is not supported: L->is_ll is always TRUE */

    let Work { iwork, head, .. } = work;

    /* SuperMap: size n; then four size-nsuper arrays.  `Previous` is the
     * fifth, and is read only by the GPU path. */
    let (supermap, rest) = iwork[..2 * n + 5 * nsuper].split_at_mut(n);
    let (_relative_map, rest) = rest.split_at_mut(n);
    let (next, rest) = rest.split_at_mut(nsuper);
    let (lpos, rest) = rest.split_at_mut(nsuper);
    let (next_save, rest) = rest.split_at_mut(nsuper);
    let (lpos_save, _) = rest.split_at_mut(nsuper);

    let supermap = Ws::new(supermap);
    /* Head: size n+1, only Head [0..nsuper-1] used */
    let head = Ws::new(head);

    /* find the mapping of nodes to relaxed supernodes:
     * SuperMap [k] = s if column k is contained in supernode s */
    {
        let sup = Ws::new_ref(&l.sym.sup);
        for s in 0..nsuper {
            for k in sup[s]..sup[s + 1] {
                supermap[k] = s as i64;
            }
        }
    }

    let SuperWork {
        pool,
        descs,
        counters,
        par_flops,
        tree_flops,
        force_serial,
    } = cwork;
    counters.wide.store(0, Atomic::Relaxed);
    counters.forked.store(0, Atomic::Relaxed);

    l.schedule.ensure(&l.sym, supermap);

    /* Inside [`numeric_pool`], so `current_num_threads` is the pool that will
     * actually run this and not the process-wide one — `ctx.nthreads` is what
     * the panel kernels split against, so reading it outside would size every
     * fork for a width the workers do not have. */
    let mut run = || {
        let ctx = Ctx {
            ap: Ws::new_ref(&a.p),
            ai: Ws::new_ref(&a.i),
            ax: Ws::new_ref(&a.x),
            ls: Ws::new_ref(&l.sym.s),
            lpi: Ws::new_ref(&l.sym.pi),
            lpx: Ws::new_ref(&l.sym.px),
            sup: Ws::new_ref(&l.sym.sup),
            beta,
            n,
            nsuper,
            xsize: l.sym.xsize,
            par_flops: *par_flops,
            tree_flops: *tree_flops,
            threads: rayon::current_num_threads() > 1,
            nthreads: rayon::current_num_threads(),
            counters,
        };

        let sched = &l.schedule;
        if sched.postordered
            && !*force_serial
            && par_numeric(&ctx, &sched.tree, &sched.plan, pool, &mut l.x)
        {
            l.minor = n;
            return;
        }

        /* Either the tree could not be used, or a supernode was not positive
         * definite and L->x is garbage.  Upstream's loop rewrites every entry of
         * L->x before reading it, so re-running it from the top is a clean slate
         * and not a repair. */
        let mut tw = pool.take();
        worker(
            &ctx,
            &mut l.x,
            &mut l.minor,
            &mut tw,
            supermap,
            Ws::new(next),
            Ws::new(lpos),
            Ws::new(next_save),
            Ws::new(lpos_save),
            head,
            descs,
            QUICK_RETURN_IF_NOT_POSDEF,
        );
        pool.give(tw);
    };
    match numeric_pool() {
        Some(p) => p.install(run),
        None => run(),
    }
    Ok(())
}

/// `Common->quick_return_if_not_posdef` (`t_cholmod_defaults.c:48`).
const QUICK_RETURN_IF_NOT_POSDEF: bool = false;

/// The pool the numeric factorization runs in — the *performance* cores, not
/// every logical CPU.
///
/// `rayon`'s default pool is one worker per logical CPU, which on a
/// heterogeneous machine means workers on cores several times slower than the
/// rest. This factorization forks over the elimination tree and again inside a
/// supernode's panel, and both joins wait on their slowest task, so a worker on
/// a slow core does not add its share — it sets the pace. Measured on an M4 Pro
/// (8 performance + 4 efficiency cores), refactorize against thread count:
///
/// | threads       | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
/// |---------------|---|---|---|---|---|----|----|
/// | gridfit 110²  | 1.00 | 1.70 | 2.10 | **2.34** | 2.17 | 1.99 | 1.90 |
/// | gridfit 320²  | 1.00 | 1.71 | 2.87 | 3.36 | **3.89** | 3.80 | 3.67 |
/// | pywarper AtA  | 1.00 | 1.73 | 2.56 | 2.82 | 2.83 | **2.92** | 2.66 |
///
/// Every one of them peaks at or below the performance-core count and is worse
/// at 12 than at 8 — by 6% on the two large ones and 23% on the small one. So
/// the extra four workers are not merely low-yield, they cost.
///
/// **A private pool, not a resize of the global one.** `hea` is a library and
/// the global pool belongs to the process; the nmath element-wise maps use it
/// and want every core, since those are equal-sized independent chunks with no
/// join in the middle. Only this one caller has the problem, so only this one
/// caller gets the smaller pool.
///
/// `RAYON_NUM_THREADS` still wins: when it is set, the count is left at
/// `rayon`'s default so an explicit request is honoured. Anywhere the query
/// does not apply — a homogeneous machine, or any target but macOS — this
/// returns `None` and the global pool is used unchanged. x86-64 is *not* capped
/// to physical cores here: that would be a guess about hyperthreading, and
/// there is no measurement behind it.
fn numeric_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        if std::env::var_os("RAYON_NUM_THREADS").is_some() {
            return None;
        }
        let nt = perf_cores()?;
        if nt >= rayon::current_num_threads() || nt < 2 {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .thread_name(|i| format!("hea-sparse-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// `sysctl hw.perflevel0.physicalcpu` — the number of performance cores.
///
/// Level 0 is the fastest cluster on Apple Silicon; the key is absent on Intel
/// Macs, which is the "homogeneous, leave it alone" answer.
#[cfg(target_os = "macos")]
fn perf_cores() -> Option<usize> {
    let mut v: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: `sysctlbyname` writes at most `len` bytes into `v`, and `len` is
    // `size_of::<i32>()`; the name is a NUL-terminated literal.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut v).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && v > 0).then_some(v as usize)
}

#[cfg(not(target_os = "macos"))]
fn perf_cores() -> Option<usize> {
    None
}

/// `cholmod_factorize_p`'s supernodal branch (`cholmod_factorize.c:172-266`),
/// for a symmetric `A`.
///
/// Permutes `A` into the *lower* form the supernodal worker needs — which is
/// the mirror of what the simplicial [`super::numeric::factorize`] builds — and
/// factorizes. `Common->final_asis` is true at its default, so the
/// `change_factor` back to simplicial (`:263-267`) is a no-op and is not
/// ported: a caller wanting a simplicial `L` should not have asked for a
/// supernodal factor.
pub fn super_factorize(
    a: &Sparse,
    beta: f64,
    l: &mut SuperFactor,
    work: &mut Work,
    cwork: &mut SuperWork,
) -> Result<(), NumericError> {
    if a.stype == 0 {
        return Err(NumericError::Invalid(
            "stype must be nonzero: this port factorizes LL' = A for a \
             symmetric A, not LL' = AA'",
        ));
    }
    if a.n != l.n {
        return Err(NumericError::Invalid("dimensions of A and L do not match"));
    }

    /* S = tril (P A P').  `ptranspose (A, 2, ...)` is the conjugate transpose,
     * which for CHOLMOD_REAL is the same array transpose mode 1 gives. */
    const VALUES: bool = true;
    const LOWER: bool = true;
    let s = permute_sym(a, l.ordering, &l.perm, VALUES, LOWER, &mut work.all());
    super_numeric(s.as_ref().unwrap_or(a), beta, l, work, cwork)
}

/* ========================================================================= */

#[cfg(test)]
mod tests {
    use super::super::amd::IntWidth;
    use super::super::super_symbolic::{super_symbolic, Relax};
    use super::super::symbolic::{analyze_sparse, permute_sym, Method, Sparse};
    use super::super::testcorpus::{corpus, spd_triangle};
    use super::super::ws::{columns_are_sorted, Work};
    use super::*;

    /// [`assemble_par`] writes exactly what [`assemble`]'s serial `j` loop
    /// writes, bit for bit.
    ///
    /// Worth a direct test rather than corpus coverage: upstream's gate
    /// (`ndrow1 > 64` and `cholmod_nthreads (ndcol * ndrow2) > 1`, i.e.
    /// `ndcol * ndrow2 >= 256000`) fires on **2 of 11446** descendant updates
    /// across the largest matrix in `dev/`, and on none at all in the smaller
    /// ones — so the factorization gates cannot be relied on to reach this arm.
    #[test]
    fn the_parallel_assembly_writes_what_the_serial_one_writes() {
        for &(ndrow1, ndrow2, ndcol) in &[(1i64, 1i64, 1i64), (5, 9, 3), (70, 300, 900)] {
            /* `RelativeMap` is derived from `Map ∘ Ls`; feed it a strictly
             * increasing map with gaps, which is the shape it really has. */
            let nsrow = 2 * ndrow2 + 3;
            let ls: Vec<i64> = (0..ndrow2).map(|i| 2 * i + 1).collect();
            let mut map = vec![EMPTY; (2 * ndrow2 + 3) as usize];
            for (i, &r) in ls.iter().enumerate() {
                map[r as usize] = 2 * i as i64 + 1;
            }
            let g = Desc {
                pdx1: 0,
                pdi1: 0,
                ndrow: ndrow2,
                ndcol,
                ndrow1,
                ndrow2,
            };
            let c: Vec<f64> = (0..g.csize())
                .map(|i| ((i % 97) as f64 - 48.0) / 7.0)
                .collect();
            let base: Vec<f64> = (0..(nsrow * (2 * ndrow2 + 3)) as usize)
                .map(|i| ((i % 31) as f64) / 3.0)
                .collect();

            let mut ser = base.clone();
            let mut rm = vec![0i64; ndrow2 as usize];
            assemble(
                &g,
                &c,
                &mut ser,
                nsrow,
                Ws::new_ref(&ls),
                Ws::new_ref(&map),
                Ws::new(&mut rm),
                1, /* nt = 1 keeps the serial arm regardless of the gate */
            );

            let mut par = base.clone();
            let mut rm2 = vec![0i64; ndrow2 as usize];
            {
                let rmw = Ws::new(&mut rm2);
                for i in 0..ndrow2 {
                    rmw[i] = map[ls[i as usize] as usize];
                }
                assemble_par(&g, &c, &mut par, nsrow, rmw);
            }

            assert_eq!(ser, par, "shape {ndrow1}x{ndrow2}x{ndcol}");
        }
    }

    /// One corpus matrix, scaled by `scale`, and the symbolic factor and
    /// workspace `mod.rs` would hand the numeric path.
    fn setup(
        n: usize,
        edges: &[(usize, usize)],
        ordering: Ordering,
        scale: f64,
    ) -> (Sparse, SuperFactor, Work) {
        let (p, i, v) = spd_triangle(n, edges, false);
        let a = Sparse {
            n,
            p: p.clone(),
            i: i.clone(),
            x: v.iter().map(|x| x * scale).collect(),
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(n, &p, &i),
        };
        let s = analyze_sparse(&a, Method::Pinned(ordering), IntWidth::I64).unwrap();
        let mut w = Work::new(n);
        let a2 = permute_sym(&a, s.ordering, &s.perm, false, false, &mut w.all());
        let sym = super_symbolic(
            a2.as_ref().unwrap_or(&a),
            &s.parent,
            &s.colcount,
            &Relax::default(),
            &mut w,
        )
        .unwrap();
        let l = SuperFactor::new(&s, sym);
        (a, l, w)
    }

    /// One corpus matrix, factorized supernodally with both thresholds and the
    /// driver pinned. Returns the factor, how many updates took the batched arm
    /// and how many supernodes forked.
    fn factor(
        n: usize,
        edges: &[(usize, usize)],
        ordering: Ordering,
        par_flops: f64,
        tree_flops: f64,
        force_serial: bool,
    ) -> (SuperFactor, usize, usize) {
        let (a, mut l, mut w) = setup(n, edges, ordering, 1.0);
        let mut cw = SuperWork::pinned(par_flops, tree_flops, force_serial);
        super_factorize(&a, 0.0, &mut l, &mut w, &mut cw).unwrap();
        let (wide, forked) = cw.counts();
        (l, wide, forked)
    }

    /// Every arm has to produce the same `L->x`, entry for entry — not to a
    /// tolerance.
    ///
    /// That is the whole claim both parallel levers rest on. Computing several
    /// `C`s at once reorders nothing, because each reads a different descendant
    /// and they are assembled into the supernode in the link list's order
    /// either way; factorizing two subtrees at once reorders nothing, because a
    /// supernode's inputs are its descendants' finished blocks and its update
    /// list was fixed by [`Plan`] before any arithmetic ran. The thresholds are
    /// pinned rather than left at their defaults so every arm is exercised on
    /// every corpus matrix, and the counters are asserted so no arm can pass
    /// vacuously.
    #[test]
    fn going_wide_does_not_move_a_rounding() {
        let (mut batched, mut forks) = (0, 0);
        for (name, n, edges) in corpus() {
            for ordering in [Ordering::Amd, Ordering::Natural] {
                /* upstream's fused link-list walk, one supernode at a time */
                let (base, none, _) = factor(n, &edges, ordering, f64::INFINITY, 0.0, true);
                assert_eq!(none, 0, "{name} went wide with the batching disabled");

                for (par, tree) in [
                    (f64::INFINITY, f64::INFINITY),
                    (0.0, f64::INFINITY),
                    (f64::INFINITY, 0.0),
                    (0.0, 0.0),
                ] {
                    let (got, wide, forked) = factor(n, &edges, ordering, par, tree, false);
                    assert_eq!(base.minor, got.minor, "{name} minor at {par}/{tree}");
                    assert_eq!(base.x, got.x, "{name} L->x at {par}/{tree}");
                    batched += wide;
                    forks += forked;
                }
            }
        }
        assert!(
            batched > 0,
            "the batched arm never ran: the test proves nothing"
        );
        assert!(
            forks > 0,
            "no supernode ever forked: the test proves nothing"
        );
    }

    /// A refactorization reuses the [`Schedule`] the first one built, so it has
    /// to produce exactly what a fresh factor does.
    ///
    /// This is the workload the cache exists for and the one nothing else here
    /// covers: every other test factorizes each `L` once, so a stale or
    /// half-rebuilt schedule would pass them all and be wrong only in
    /// production. The second matrix is the first scaled, so the pattern — and
    /// therefore the schedule — is shared while every value differs.
    #[test]
    fn refactorizing_reuses_the_schedule_without_changing_the_answer() {
        for (name, n, edges) in corpus() {
            for ordering in [Ordering::Amd, Ordering::Natural] {
                let (a1, mut l, mut w) = setup(n, &edges, ordering, 1.0);
                let (a3, mut fresh, _) = setup(n, &edges, ordering, 3.0);
                let mut cw = SuperWork::new();

                super_factorize(&a1, 0.0, &mut l, &mut w, &mut cw).unwrap();
                super_factorize(&a3, 0.0, &mut l, &mut w, &mut cw).unwrap();
                super_factorize(&a3, 0.0, &mut fresh, &mut w, &mut cw).unwrap();

                assert_eq!(fresh.minor, l.minor, "{name} minor");
                assert_eq!(fresh.x, l.x, "{name} L->x");
            }
        }
    }
}
