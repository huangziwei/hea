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
//! **Memory contract.** `A` is borrowed and nothing here keeps it; `L->x` is
//! allocated once per factor and reused by every refactorization. The one
//! deviation is the per-supernode scratch: upstream has exactly one `Map`
//! (aliased onto `Common->Flag`), one `RelativeMap` (a slice of `Common->Iwork`)
//! and one `C` (`cholmod_super_numeric.c:245`), which is correct **only because
//! its supernode loop is serial**. Here they come from a [`WorkPool`], one set
//! per concurrently-running task, and the pool retains one set per worker plus
//! nothing over [`C_KEEP_DOUBLES`]. On a 3.4M-row system that is 210 MB of
//! `Map` held and up to `nthreads × maxcsize` of `C` in flight against
//! upstream's one. That is the memory cost of the parallelism.
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
use crate::nmath::util::rfma;

#[derive(Debug, Clone)]
pub struct SuperFactor {
    pub n: usize,
    pub perm: Vec<i64>,
    pub colcount: Vec<i64>,
    pub ordering: Ordering,
    pub sym: SuperSymbolic,
    pub x: Vec<f64>,
    pub minor: usize,
    pub numeric: bool,
    schedule: Schedule,
}

impl SuperFactor {
    /// The supernodal symbolic factor, as `cholmod_analyze` returns it: the
    /// pattern, with no numeric part yet.
    ///
    /// Takes the [`Symbolic`] **by value**, because upstream's analysis has no
    /// separate object to take it from: `Lperm = L->Perm` at
    /// `cholmod_analyze.c:542`, so the trial loop's winner *is* the factor's
    /// ordering and no copy exists. A caller that needs the analysis afterwards
    /// clones it and can see what that costs — `2n` int64s, 52 MiB on a
    /// 3.4M-row system.
    pub fn new(s: Symbolic, sym: SuperSymbolic) -> SuperFactor {
        SuperFactor {
            n: sym.n,
            minor: s.perm.len(),
            perm: s.perm,
            colcount: s.colcount,
            ordering: s.ordering,
            sym,
            x: Vec::new(),
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
    map: Vec<i64>,
    /// `RelativeMap`. Upstream's is a size-`n` slice of the shared `Iwork`;
    /// here there is one per task, and it is indexed `0 .. ndrow2`, so it is
    /// sized `L->maxesize` instead — `ndrow2` counts a descendant's rows
    /// outside its own columns, which is what `maxesize` bounds. At `n` it was
    /// 27 MB per task on a 3.4M-row system, for 33 KB of use.
    relative_map: Vec<i64>,
    c: Vec<f64>,
    bufs: Vec<Vec<f64>>,
}

impl TaskWork {
    fn new(n: usize, esize: usize) -> TaskWork {
        TaskWork {
            map: vec![EMPTY; n],
            relative_map: vec![0; esize],
            c: Vec::new(),
            bufs: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
struct WorkPool {
    n: usize,
    esize: usize,
    keep: usize,
    free: Mutex<Vec<TaskWork>>,
}

impl WorkPool {
    fn ensure(&mut self, n: usize, esize: usize, keep: usize) {
        self.keep = keep;
        if self.n != n || self.esize != esize {
            self.n = n;
            self.esize = esize;
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
            .unwrap_or_else(|| TaskWork::new(self.n, self.esize))
    }

    /// Return a workspace, keeping only what is worth keeping.
    ///
    /// Two separate hoards, and both were real on a 3.4M-row system. The big
    /// `C`s belong to the supernodes near the root and only a couple of tasks
    /// are inside one at a time, but every task that had *ever* held one kept
    /// it — ~13 × 139 MB. And the pool itself grows to whatever concurrency the
    /// joins reached, so `Map` alone (size `n`, 27 MB here) was held that many
    /// times over for the rest of the process. Re-allocating either is
    /// `alloc_zeroed`, so the pages come from the kernel already zero and cost
    /// the same first-touch faults the writes would take anyway.
    ///
    /// The cap is on what is *retained*, never on what is handed out: `take`
    /// still allocates on demand, so a burst of concurrency cannot deadlock
    /// against it.
    fn give(&self, mut w: TaskWork) {
        if w.c.len() > C_KEEP_DOUBLES {
            w.c = Vec::new();
        }
        for b in w.bufs.iter_mut() {
            if b.len() > C_KEEP_DOUBLES {
                *b = Vec::new();
            }
        }
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if free.len() < self.keep {
            free.push(w);
        }
    }
}

const C_KEEP_DOUBLES: usize = 1 << 16;

#[derive(Debug, Default)]
struct Counters {
    wide: AtomicUsize,
    forked: AtomicUsize,
}

#[derive(Debug)]
pub struct SuperWork {
    pool: WorkPool,
    descs: Vec<Desc>,
    counters: Counters,
    par_flops: f64,
    tree_flops: f64,
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

    #[cfg(all(test, not(vendor_blas)))]
    pub(super) fn pinned(par_flops: f64, tree_flops: f64, force_serial: bool) -> SuperWork {
        SuperWork {
            par_flops,
            tree_flops,
            force_serial,
            ..SuperWork::default()
        }
    }

    #[cfg(all(test, not(vendor_blas)))]
    pub(super) fn counts(&self) -> (usize, usize) {
        (
            self.counters.wide.load(Atomic::Relaxed),
            self.counters.forked.load(Atomic::Relaxed),
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Desc {
    pdx1: i64,
    pdi1: i64,
    ndrow: i64,
    ndcol: i64,
    ndrow1: i64,
    ndrow2: i64,
}

impl Desc {
    #[inline]
    fn csize(&self) -> usize {
        (self.ndrow2 * self.ndrow1) as usize
    }

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
/// supernode (`repeat_supernode`, `:301-318`) and [`Plan`] assumes success.
#[derive(Debug, Clone, Default)]
struct Plan {
    dptr: Vec<usize>,
    descs: Vec<Desc>,
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
    cptr: Vec<usize>,
    child: Vec<i64>,
    first: Vec<usize>,
    work: Vec<f64>,
    subwork: Vec<f64>,
    roots: Vec<usize>,
}

impl Tree {
    #[inline]
    fn kids(&self, s: usize) -> &[i64] {
        &self.child[self.cptr[s]..self.cptr[s + 1]]
    }

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

const BATCH_MAX: usize = 64;
const BATCH_DOUBLES: usize = 4 << 20;

/// Below this, a batch of updates is not worth a fork and a join.
///
/// The kernels run at tens of GF/s, so this is tens of microseconds of work
/// against a join that costs a few — and it has to be a *batch* total rather
/// than a per-update one, because the batch is what rayon splits.
///
/// This arm is two builds, and they want the same value for the same reason:
/// the portable kernels are one thread per call, and so is OpenBLAS here, since
/// [`super::blas::init`] pins it to one inside hea's pool. Either way this fork
/// *is* the parallelism, which is why the number is four orders of magnitude
/// below the `accelerate` one below.
///
/// Raising it trades the two regimes against each other, which is why it stays
/// where it is. Small systems gain a few percent of core from batches that no
/// longer fork and so no longer pay for the join; multi-million-row systems lose
/// several percent of the wall clock, and the wall clock at scale outranks core
/// on a 35 ms factorization. Both backends on this arm read the same shape.
#[cfg(not(accelerate))]
const PAR_FLOPS: f64 = 5.0e5;

/// Four orders of magnitude higher when the BLAS threads the call itself, which
/// is the same predicate [`strip_width`] splits on and for the same reason: with
/// Accelerate underneath, forking the descendants of one supernode puts a second
/// scheduler on top of the vendor's own, and it is pure loss on both axes.
/// Raising it to here costs nothing on the wall clock — it improves — and drops
/// the CPU by 1.4-2.2x. Multi-million-row systems neither gain nor object: their
/// own updates are far above any threshold in this range and fork regardless.
///
/// Without a threaded BLAS this fork *is* the parallelism and raising it is a
/// 1.75x wall regression, which is why the two arms differ.
#[cfg(accelerate)]
const PAR_FLOPS: f64 = 1.0e9;

const TREE_FLOPS: f64 = 1.0e6;

const MAX_FORK_DEPTH: u32 = 48;

const STRIP_FLOPS: f64 = 1.0e5;

const STRIP_MIN_COLS: usize = 128;

const STRIP_RESIDENT_BYTES: usize = 2 << 20;

const STRIP_OVER: usize = 4;

const STRIP_TASKS: usize = 16;

fn strip_width(g: &Desc, nt: usize, batch_flops: f64) -> usize {
    let ndrow1 = g.ndrow1 as usize;
    /* A threaded BLAS wants upstream's whole call, and the split exists only to
     * give a one-thread-per-call kernel a shape. That is a statement about the
     * *library*, not about the feature: Accelerate threads, so it gets the whole
     * call, but OpenBLAS is pinned to one thread here — see `blas::init` — and
     * the pool needs the tasks. What Accelerate does *not* do is stand down
     * inside a pool: measured, it takes 1.06-1.30 threads' worth of CPU at
     * `RAYON_NUM_THREADS=1`, and pinning it with `VECLIB_MAXIMUM_THREADS=1`
     * costs the 3.4M-row system 12% of its wall clock for 21% of its CPU. That
     * trade is the caller's to make, so it stays an environment variable. */
    if nt <= 1 || cfg!(accelerate) {
        return ndrow1.max(1);
    }
    let per_col = 2.0 * g.ndrow2 as f64 * g.ndcol as f64;
    /* Relative to the batch and floored at the absolute budget, so a
     * flop-heavy batch gets *wider* strips rather than more of them, and a
     * batch below the crossing falls back to the absolute budget entry for
     * entry. See `STRIP_TASKS`. */
    let budget = (batch_flops / (STRIP_TASKS * nt) as f64).max(STRIP_FLOPS);
    let mut cols = (budget / per_col).ceil().max(1.0) as usize;
    if 8 * g.ndrow2 as usize * g.ndcol as usize > STRIP_RESIDENT_BYTES {
        cols = cols
            .max(STRIP_MIN_COLS)
            .max(ndrow1.div_ceil(nt * STRIP_OVER));
    }
    cols.div_ceil(SYRK_NB) * SYRK_NB
}

#[cfg_attr(feature = "profiling", inline(never))]
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

#[cfg_attr(feature = "profiling", inline(never))]
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

/// Make `buf` at least `need` long, discarding what is in it.
///
/// A fresh allocation and not `Vec::resize`, because nothing in the old
/// contents is wanted: `update_c` writes every entry `assemble` reads, and
/// `resize` would `realloc` — copying the stale bytes — and then `memset` the
/// tail. `vec![0.0; n]` goes through `alloc_zeroed`, so for a `C` in the tens of
/// megabytes the pages arrive already zero and are faulted in by the writes that
/// were going to happen anyway. Growing a 139 MB `C` a supernode at a time was
/// 12% of the factorization's busy samples in `__bzero`/`memmove`/`memset`.
#[inline]
fn grow_c(buf: &mut Vec<f64>, need: usize) {
    if buf.len() < need {
        *buf = vec![0.0; need];
    }
}

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

        if ctx.threads && flops >= ctx.par_flops {
            /* One `C` per descendant, and for a lone one the serial path's own
             * buffer — a descendant whose `C` is already over `BATCH_DOUBLES`
             * is a batch of one, and it is exactly the descendant with the most
             * flops in it. Giving that case the serial arm left the largest
             * updates in the factorization single-threaded. */
            let bufs: &mut [Vec<f64>] = if batch.len() == 1 {
                core::slice::from_mut(c)
            } else {
                if bufs.len() < batch.len() {
                    bufs.resize_with(batch.len(), Vec::new);
                }
                &mut bufs[..batch.len()]
            };
            for (buf, g) in bufs.iter_mut().zip(batch) {
                grow_c(buf, g.csize());
            }
            {
                /* One task per strip rather than per descendant: a supernode's
                 * descendants differ in size by orders of magnitude, and a
                 * batch cannot finish before its largest member does. */
                let mut tasks: Vec<(Desc, usize, usize, &mut [f64])> = Vec::new();
                for (buf, g) in bufs.iter_mut().zip(batch) {
                    let (ndrow1, ndrow2) = (g.ndrow1 as usize, g.ndrow2 as usize);
                    let width = strip_width(g, ctx.nthreads, flops);
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
                grow_c(c, g.csize());
                update_c(lower, base, g, c);
                assemble(g, c, sx, nsrow, ctx.ls, map, relative_map, ctx.nthreads);
            }
        }
        i0 = i1;
    }
}

struct Ctx<'a> {
    ap: &'a Ws,
    ai: &'a Ws,
    ax: &'a Ws<f64>,
    /// `F` — `A'` in column form — present exactly when `A->stype == 0`, where
    /// the supernode is assembled from `A*F` rather than from `A` itself
    /// (`t_cholmod_super_numeric_worker.c:441-500`).
    f: Option<(&'a Ws, &'a Ws, &'a Ws<f64>)>,
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
    threads: bool,
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

    own.fill(0.0);

    /* If row i is the kth row in s, then Map [i] = k.  Similarly, if
     * column j is the kth column in s, then  Map [j] = k.
     *
     * Upstream wraps this in `#pragma omp parallel for if (nsrow > 128)` with
     * `cholmod_nthreads (nsrow)` (`:387-397`), which hands out a second thread
     * only once `nsrow` reaches twice `Common->chunk`. Serial here because
     * that is two orders of magnitude wider than the widest supernode these
     * factorizations build, so a parallel arm would be unreachable — and an
     * unreachable arm is one no gate can check. The same applies to the copy
     * below and to `assemble`'s relative map. */
    {
        let map = Ws::new(&mut tw.map);
        for k in 0..nsrow {
            map[ctx.ls[psi + k]] = k;
        }
    }

    /* Upstream's `#pragma omp parallel for if (k2-k1 > 64)` (`:437-439`) takes
     * its thread count from the nonzeros this column block reads, not from the
     * column count, and needs twice `Common->chunk` of them. Serial here for
     * the reason given at the `Map` build above. */
    {
        let map = Ws::new_ref(&tw.map);
        let sx = Ws::new(own);
        for k in k1..k2 {
            match ctx.f {
                None => {
                    /* copy the kth column of A into the supernode */
                    for p in ctx.ap[k]..ctx.ap[k + 1] {
                        /* row i of L is located in row Map [i] of s */
                        let i = ctx.ai[p];
                        if i >= k {
                            /* If the test is false, the numeric factorization
                             * of A is undefined.  The test does not detect all
                             * invalid entries, only some of them. */
                            let imap = map[i];
                            if imap >= 0 && imap < nsrow {
                                /* Lx [Map [i] + pk] = Ax [p] */
                                sx[imap + (k - k1) * nsrow] = ctx.ax[p];
                            }
                        }
                    }
                }
                Some((fp, fi, fx)) => {
                    /* copy the kth column of A*F into the supernode */
                    for pf in fp[k]..fp[k + 1] {
                        let j = fi[pf];
                        let fjk = fx[pf];
                        for p in ctx.ap[j]..ctx.ap[j + 1] {
                            let i = ctx.ai[p];
                            if i >= k {
                                /* see the discussion of imap above */
                                let imap = map[i];
                                if imap >= 0 && imap < nsrow {
                                    /* Lx [Map [i] + pk] += Ax [p] * fjk */
                                    let q = imap + (k - k1) * nsrow;
                                    sx[q] = rfma(ctx.ax[p], fjk, sx[q]);
                                }
                            }
                        }
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

    apply_updates(ctx, descs, lower, base, own, nsrow, tw);

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
    let worth =
        ctx.threads && depth < MAX_FORK_DEPTH && tree.subwork[s] - tree.work[s] >= ctx.tree_flops;

    /* follow single-child links down to the first branch */
    let mut top = s;
    if worth {
        while tree.kids(top).len() == 1 {
            let c = tree.kids(top)[0] as usize;
            debug_assert_eq!(
                c + 1,
                top,
                "postorder makes a single child the previous index"
            );
            top = c;
        }
    }
    let kids = tree.kids(top);

    if !worth || kids.len() < 2 {
        let mut tw = pool.take();
        for t in tree.first[s]..=s {
            node(ctx, plan, failed, t, base, lx, &mut tw);
        }
        pool.give(tw);
        return;
    }

    {
        /* the fork only reaches `px [top+1]`; the reborrow ends with this
         * block, so the chain below gets `lx` whole again */
        let head = &mut lx[..(ctx.lpx[top + 1] - base) as usize];
        let mut jobs: Vec<(usize, i64, &mut [f64])> = Vec::with_capacity(kids.len());
        let mut rest = &mut head[..];
        let mut cur = base;
        for &c in kids {
            let c = c as usize;
            let end = ctx.lpx[c + 1];
            let (piece, tail) = rest.split_at_mut((end - cur) as usize);
            jobs.push((c, cur, piece));
            rest = tail;
            cur = end;
        }
        debug_assert_eq!(cur, ctx.lpx[top]);
        jobs.par_iter_mut()
            .for_each(|(c, b, sl)| subtree(ctx, tree, plan, pool, failed, *c, *b, sl, depth + 1));
    }
    ctx.counters.forked.fetch_add(1, Atomic::Relaxed);

    /* `top`, then the chain that led down to it — ascending index order, which
     * postorder makes a topological one */
    let mut tw = pool.take();
    for t in top..=s {
        node(ctx, plan, failed, t, base, lx, &mut tw);
    }
    pool.give(tw);
}

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

    let mut s: usize = 0;
    while s < nsuper {
        let k1 = ctx.sup[s]; /* s contains columns k1 to k2-1 of L */
        let k2 = ctx.sup[s + 1];
        let nscol = k2 - k1; /* # of columns in all of s */
        let psi = ctx.lpi[s]; /* pointer to first row of s in Ls */
        let psx = ctx.lpx[s]; /* pointer to first row of s in Lx */
        let psend = ctx.lpi[s + 1]; /* pointer just past last row of s in Ls */
        let nsrow = psend - psi; /* # of rows in all of s */

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

        let nscol2 = if repeat_supernode { nscol_new } else { nscol };
        let info = {
            let (lower, tail) = lx.split_at_mut(psx as usize);
            let own = &mut tail[..(nsrow * nscol) as usize];
            node_numeric(ctx, s, descs, lower, 0, own, tw, nscol2, repeat_supernode)
        };

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
    f: Option<&Sparse>,
    beta: f64,
    l: &mut SuperFactor,
    work: &mut Work,
    cwork: &mut SuperWork,
) -> Result<(), NumericError> {
    let n = l.n;
    let nsuper = l.sym.nsuper;

    if (a.stype == 0) != f.is_some() {
        return Err(NumericError::Invalid(
            "super_numeric needs F exactly when A->stype is zero",
        ));
    }
    if a.nrow != n {
        return Err(NumericError::Invalid("dimensions of A and L do not match"));
    }
    if !a.numeric {
        return Err(NumericError::Invalid("A has no numeric values"));
    }

    #[cfg(vendor_blas)]
    super::blas::init();

    /* allocate workspace in Common: w = 2*nrow + 5*nsuper.  Map and
     * RelativeMap are per-task here (see TaskWork), so the two n-sized slots
     * upstream carves for them go unused; the request is left at upstream's
     * size because Work is shared and only ever grows. */
    work.allocate(n, 2 * n + 5 * nsuper, 0);
    /* One workspace per worker is what a refactorization can reuse without
     * hoarding: more than that only ever exist because a worker blocked on a
     * join stole another task, and those are transient. */
    cwork
        .pool
        .ensure(n, l.sym.maxesize.max(1), rayon::current_num_threads());

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
            f: f.map(|f| (Ws::new_ref(&f.p), Ws::new_ref(&f.i), Ws::new_ref(&f.x))),
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
    if a.nrow != l.n {
        return Err(NumericError::Invalid("dimensions of A and L do not match"));
    }

    /* S = tril (P A P').  `ptranspose (A, 2, ...)` is the conjugate transpose,
     * which for CHOLMOD_REAL is the same array transpose mode 1 gives.
     *
     * For `stype == 0` there is no triangle to take: `S` is `A` itself under a
     * natural ordering and `F'` under a permuted one, with `F = A(p,:)'` in
     * both cases the second operand the assembly multiplies through
     * (`cholmod_factorize.c:201-250`). */
    const VALUES: bool = true;
    const LOWER: bool = true;
    if a.stype != 0 {
        let s = permute_sym(a, l.ordering, &l.perm, VALUES, LOWER, &mut work.all());
        return super_numeric(s.as_ref().unwrap_or(a), None, beta, l, work, cwork);
    }
    let f = super::symbolic::transpose_unsym(
        a,
        VALUES,
        (!matches!(l.ordering, Ordering::Natural)).then_some(&l.perm[..]),
    );
    if matches!(l.ordering, Ordering::Natural) {
        super_numeric(a, Some(&f), beta, l, work, cwork)
    } else {
        let s = super::symbolic::transpose_unsym(&f, VALUES, None);
        super_numeric(&s, Some(&f), beta, l, work, cwork)
    }
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

    fn setup(
        n: usize,
        edges: &[(usize, usize)],
        ordering: Ordering,
        scale: f64,
    ) -> (Sparse<'static>, SuperFactor, Work) {
        let (p, i, v) = spd_triangle(n, edges, false);
        let a = Sparse {
            nrow: n,
            n,
            p: p.clone().into(),
            i: i.clone().into(),
            x: v.iter().map(|x| x * scale).collect(),
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(n, &p, &i),
        };
        let mut w = Work::new(n);
        let s = analyze_sparse(&a, Method::Pinned(ordering), IntWidth::I64, &mut w).unwrap();
        let a2 = permute_sym(&a, s.ordering, &s.perm, false, false, &mut w.all());
        let sym = super_symbolic(
            a2.as_ref().unwrap_or(&a),
            None,
            &s.parent,
            &s.colcount,
            &Relax::default(),
            &mut w,
        )
        .unwrap();
        let l = SuperFactor::new(s, sym);
        (a, l, w)
    }

    #[cfg(not(vendor_blas))]
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

    #[cfg(not(vendor_blas))]
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
