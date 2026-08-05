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
        }
    }
}

/// The `cholmod_dense *C` workspace `cholmod_super_numeric` allocates per call
/// (`:245`), kept across calls instead — plus the extra copies of it the
/// batched update path needs.
///
/// `c` is `L->maxcsize` doubles, which is a property of the *symbolic* factor —
/// so a caller refactorizing the same pattern repeatedly should hold one of
/// these, exactly as it holds one [`Work`]. `pool` is the same buffer once per
/// concurrently-computed descendant; see [`apply_updates`].
#[derive(Debug)]
pub struct SuperWork {
    c: Vec<f64>,
    pool: Vec<Vec<f64>>,
    descs: Vec<Desc>,
    /// The batch size, in flops, at which the updates go wide — [`PAR_FLOPS`],
    /// except in [`SuperWork::with_par_flops`].
    par_flops: f64,
    /// How many updates the last factorization computed on the batched arm, so
    /// that "the parallel path ran" is a number rather than an assumption.
    wide: usize,
}

impl Default for SuperWork {
    fn default() -> SuperWork {
        SuperWork {
            c: Vec::new(),
            pool: Vec::new(),
            descs: Vec::new(),
            par_flops: PAR_FLOPS,
            wide: 0,
        }
    }
}

impl SuperWork {
    pub fn new() -> SuperWork {
        SuperWork::default()
    }

    /// The same workspace with the batching threshold overridden, which is how
    /// the corpus test drives both arms of [`apply_updates`]: `f64::INFINITY`
    /// forces the one-at-a-time loop and `0.0` forces the batched one, and the
    /// two have to agree bit for bit.
    #[cfg(test)]
    pub(super) fn with_par_flops(par_flops: f64) -> SuperWork {
        SuperWork {
            par_flops,
            ..SuperWork::default()
        }
    }

    /// Grow to `maxcsize`, never shrink — `cholmod_allocate_dense` is a fresh
    /// allocation per call upstream, but only its size matters.
    fn ensure(&mut self, maxcsize: usize) {
        if self.c.len() < maxcsize {
            self.c.resize(maxcsize, 0.0);
        }
    }
}

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

    /// What the two kernels will do, for the batching decision only — not a
    /// flop count anyone reports.
    #[inline]
    fn flops(&self) -> f64 {
        2.0 * self.ndrow2 as f64 * self.ndrow1 as f64 * self.ndcol as f64
    }
}

/// At most this many `C` buffers are live at once, and at most this many
/// doubles across them. The batch is bounded by its scratch, not only by the
/// thread count: a wide batch of wide updates would otherwise hold tens of
/// megabytes that the serial path never allocates.
const BATCH_MAX: usize = 64;
const BATCH_DOUBLES: usize = 4 << 20;

/// Below this, a batch is not worth a fork and a join.
///
/// The kernels run at tens of GF/s, so this is tens of microseconds of work
/// against a join that costs a few — and it has to be a *batch* total rather
/// than a per-update one, because the batch is what rayon splits.
const PAR_FLOPS: f64 = 5.0e5;

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
/// `lx` is `L->x` truncated at `psx`. Every descendant of `s` is stored below
/// that point, so the truncation is what lets a batch read `L` while the
/// assembly writes the supernode: the two halves are disjoint slices of one
/// array rather than an aliasing argument. `c` is likewise the strip's own
/// slice — a column of `C` is contiguous, so the strips of one `C` are disjoint
/// slices too, and neither split needs an aliasing argument to be sound.
///
/// A strip changes which thread computes an entry and nothing else: the `gemm`
/// calls are the same calls on the same block columns, so every entry is still
/// accumulated over `l` ascending in one place.
fn update_strip(lx: &[f64], g: &Desc, j0: usize, jn: usize, c: &mut [f64]) {
    let (pdx1, ndrow, ndcol) = (g.pdx1 as usize, g.ndrow as usize, g.ndcol as usize);
    let (ndrow1, ndrow2) = (g.ndrow1 as usize, g.ndrow2 as usize);

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
fn update_c(lx: &[f64], g: &Desc, c: &mut [f64]) {
    update_strip(lx, g, 0, g.ndrow1 as usize, c);
}

/// The relative map and the scatter of one `C` into the supernode — `:1037-1050`.
///
/// `sx` is `L->x` from `psx` on, so upstream's `psx + RelativeMap [j] * nsrow`
/// loses its `psx`.
fn assemble(
    g: &Desc,
    c: &[f64],
    sx: &mut [f64],
    nsrow: i64,
    ls: &Ws,
    map: &Ws,
    relative_map: &mut Ws,
) {
    /* construct relative map to supernode s */
    for i in 0..g.ndrow2 {
        relative_map[i] = map[ls[g.pdi1 + i]];
        debug_assert!(relative_map[i] >= 0 && relative_map[i] < nsrow);
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

/// Apply every pending update of one supernode, in the link list's order.
///
/// The updates are independent — each reads a different descendant's block of
/// `L` and writes its own `C` — but their assemblies are not, because two
/// descendants can hit the same entry of the supernode. So `C` is what goes
/// wide and the assembly stays serial and in order, which is what keeps `L->x`
/// bit-for-bit what the one-at-a-time loop produces. It is also why the batch
/// is a batch: the buffers are live simultaneously, so their total size is
/// capped.
///
/// The parallel and serial arms differ only in *which* `C` buffer each update
/// gets. Both call [`update_c`] and [`assemble`] with the same arguments.
#[allow(clippy::too_many_arguments)]
fn apply_updates(
    descs: &[Desc],
    lower: &[f64],
    sx: &mut [f64],
    nsrow: i64,
    c: &mut [f64],
    pool: &mut Vec<Vec<f64>>,
    par_flops: f64,
    wide: &mut usize,
    ls: &Ws,
    map: &Ws,
    relative_map: &mut Ws,
) {
    let mut i0 = 0;
    while i0 < descs.len() {
        let (mut i1, mut size, mut flops) = (i0, 0usize, 0.0);
        while i1 < descs.len() && i1 - i0 < BATCH_MAX && size < BATCH_DOUBLES {
            size += descs[i1].csize();
            flops += descs[i1].flops();
            i1 += 1;
        }
        let batch = &descs[i0..i1];

        if batch.len() > 1 && flops >= par_flops {
            if pool.len() < batch.len() {
                pool.resize_with(batch.len(), Vec::new);
            }
            for (buf, g) in pool.iter_mut().zip(batch) {
                if buf.len() < g.csize() {
                    buf.resize(g.csize(), 0.0);
                }
            }
            {
                /* One task per strip rather than per descendant: a supernode's
                 * descendants differ in size by orders of magnitude, and a
                 * batch cannot finish before its largest member does. */
                let mut tasks: Vec<(Desc, usize, usize, &mut [f64])> = Vec::new();
                for (buf, g) in pool[..batch.len()].iter_mut().zip(batch) {
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
                    .for_each(|(g, j0, jn, c)| update_strip(lower, g, *j0, *jn, c));
            }
            *wide += batch.len();
            for (buf, g) in pool.iter().zip(batch) {
                assemble(g, buf, sx, nsrow, ls, map, relative_map);
            }
        } else {
            for g in batch {
                debug_assert!(c.len() >= g.csize());
                update_c(lower, g, c);
                assemble(g, c, sx, nsrow, ls, map, relative_map);
            }
        }
        i0 = i1;
    }
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
#[allow(clippy::too_many_arguments)]
fn worker(
    a: &Sparse,
    beta: f64,
    l: &mut SuperFactor,
    cwork: &mut SuperWork,
    supermap: &mut Ws,
    relative_map: &mut Ws,
    next: &mut Ws,
    lpos: &mut Ws,
    next_save: &mut Ws,
    lpos_save: &mut Ws,
    map: &mut Ws,
    head: &mut Ws,
    quick_return_if_not_posdef: bool,
) {
    let nsuper = l.sym.nsuper;
    let n = l.n;

    let SuperWork {
        c,
        pool,
        descs,
        par_flops,
        wide,
    } = cwork;
    let par_flops = *par_flops;
    *wide = 0;

    let ap = Ws::new_ref(&a.p);
    let ai = Ws::new_ref(&a.i);
    let ax = Ws::new_ref(&a.x);
    let ls = Ws::new_ref(&l.sym.s);
    let lpi = Ws::new_ref(&l.sym.pi);
    let lpx = Ws::new_ref(&l.sym.px);
    let sup = Ws::new_ref(&l.sym.sup);

    /* clear the Map so that changes in the pattern of A can be detected */
    map.fill(EMPTY);

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

        let k1 = sup[s]; /* s contains columns k1 to k2-1 of L */
        let k2 = sup[s + 1];
        let nscol = k2 - k1; /* # of columns in all of s */
        let psi = lpi[s]; /* pointer to first row of s in Ls */
        let psx = lpx[s]; /* pointer to first row of s in Lx */
        let psend = lpi[s + 1]; /* pointer just past last row of s in Ls */
        let nsrow = psend - psi; /* # of rows in all of s */

        //----------------------------------------------------------------------
        // zero the supernode s
        //----------------------------------------------------------------------

        {
            let lx = Ws::new(&mut l.x);
            let pend = psx + nsrow * nscol; /* s is nsrow-by-nscol */
            for p in psx..pend {
                lx[p] = 0.0;
            }
        }

        //----------------------------------------------------------------------
        // construct the scattered Map for supernode s
        //----------------------------------------------------------------------

        /* If row i is the kth row in s, then Map [i] = k.  Similarly, if
         * column j is the kth column in s, then  Map [j] = k. */
        for k in 0..nsrow {
            map[ls[psi + k]] = k;
        }

        //----------------------------------------------------------------------
        // copy matrix into supernode s (lower triangular part only)
        //----------------------------------------------------------------------

        {
            let lx = Ws::new(&mut l.x);
            for k in k1..k2 {
                /* copy the kth column of A into the supernode */
                for p in ap[k]..ap[k + 1] {
                    /* row i of L is located in row Map [i] of s */
                    let i = ai[p];
                    if i >= k {
                        /* If the test is false, the numeric factorization of A
                         * is undefined.  The test does not detect all invalid
                         * entries, only some of them. */
                        let imap = map[i];
                        if imap >= 0 && imap < nsrow {
                            /* Lx [Map [i] + pk] = Ax [p] */
                            lx[imap + (psx + (k - k1) * nsrow)] = ax[p];
                        }
                    }
                }
            }

            /* add beta to the diagonal of the supernode, if nonzero */
            if beta != 0.0 {
                let mut pk = psx;
                for _ in k1..k2 {
                    lx[pk] += beta;
                    pk += nsrow + 1; /* advance to the next diagonal entry */
                }
            }
        }

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
        // update supernode s with each pending descendant d
        //----------------------------------------------------------------------

        /* Upstream walks the link list and does the arithmetic in one loop.
         * Here the walk comes first and only records what each update is, so
         * that the updates — which are independent, unlike the walk, which
         * advances Lpos and re-links d as it goes — can be computed together.
         * Nothing in the walk depends on the arithmetic: the geometry is read
         * before Lpos [d] is advanced, exactly as it is upstream, and the
         * ancestor a descendant is re-linked into is always past s. */
        descs.clear();
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

            debug_assert!(ndrow2 * ndrow1 <= l.sym.maxcsize as i64);
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
                    let dancestor = supermap[ls[pdi2]];
                    debug_assert!(dancestor > s as i64 && dancestor < nsuper as i64);
                    next[d] = head[dancestor];
                    head[dancestor] = d;
                }
            }
        }

        {
            /* Every descendant of s lives below psx in L->x, so the split is
             * what lets the updates read L while the assembly writes s. */
            let (lower, sx) = l.x.split_at_mut(psx as usize);
            apply_updates(
                descs,
                lower,
                sx,
                nsrow,
                c,
                pool,
                par_flops,
                wide,
                ls,
                map,
                relative_map,
            );
        }

        //----------------------------------------------------------------------
        // factorize diagonal block of supernode s in LL'
        //----------------------------------------------------------------------

        let nscol2 = if repeat_supernode { nscol_new } else { nscol };
        let mut info = potrf_l(
            nscol2 as usize, /* N: nscol2 */
            &mut l.x[psx as usize..],
            nsrow as usize, /* A, LDA: L1, nsrow */
        );

        /* if the matrix is not positive definite, the supernode is repeated
         * and only its first nscol_new columns are kept */
        if repeat_supernode {
            /* zero out the rest of this supernode */
            info = 0;
            let lx = Ws::new(&mut l.x);
            for p in psx + nsrow * nscol_new..psx + nsrow * nscol {
                lx[p] = 0.0;
            }
        }

        //----------------------------------------------------------------------
        // check if the matrix is not positive definite
        //----------------------------------------------------------------------

        if info != 0 {
            /* Matrix is not positive definite.  dpotrf/zpotrf do NOT report an
             * error if the diagonal of L has NaN's, only if it has a zero. */
            l.minor = (k1 + info - 1) as usize;

            /* clear the link lists of all subsequent supernodes */
            for ss in s + 1..nsuper {
                head[ss] = EMPTY;
            }

            /* zero this supernode, and all remaining supernodes */
            {
                let lx = Ws::new(&mut l.x);
                for p in psx..l.sym.xsize as i64 {
                    lx[p] = 0.0;
                }
            }

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
        // compute the subdiagonal block and prepare supernode for its parent
        //----------------------------------------------------------------------

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
                &mut l.x[psx as usize..],
                nsrow as usize, /* A/B, LDA/LDB: L1 and L2, nsrow */
            );

            if !repeat_supernode {
                /* Place this supernode in the link list of its parent. */
                lpos[s] = nscol;
                let sparent = supermap[ls[psi + nscol]];
                debug_assert!(sparent != s as i64 && sparent > s as i64);
                debug_assert!(sparent < nsuper as i64);
                next[s] = head[sparent];
                head[sparent] = s as i64;
            }
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
    l.minor = n;
}

/// `cholmod_super_numeric` (`cholmod_super_numeric.c:96-337`) for a symmetric
/// `A`.
///
/// `a` must be the **lower** triangle of the already-permuted matrix — see the
/// module docs. [`super_factorize`] is the entry point that arranges that;
/// this one is upstream's, taking `S` ready-made.
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

    /* allocate workspace in Common: w = 2*nrow + 5*nsuper */
    work.ensure_iwork(2 * n + 5 * nsuper);
    cwork.ensure(l.sym.maxcsize);

    /* get the current factor L and allocate numerical part, if needed */
    if !l.numeric {
        /* convert to supernodal numeric by allocating L->x */
        l.x = vec![0.0; l.sym.xsize];
        l.numeric = true;
    }
    /* supernodal LDL' is not supported: L->is_ll is always TRUE */

    let Work {
        iwork, flag, head, ..
    } = work;

    /* SuperMap: size n; RelativeMap: size n; then four size-nsuper arrays.
     * `Previous` is the fifth, and is read only by the GPU path. */
    let (supermap, rest) = iwork[..2 * n + 5 * nsuper].split_at_mut(n);
    let (relative_map, rest) = rest.split_at_mut(n);
    let (next, rest) = rest.split_at_mut(nsuper);
    let (lpos, rest) = rest.split_at_mut(nsuper);
    let (next_save, rest) = rest.split_at_mut(nsuper);
    let (lpos_save, _) = rest.split_at_mut(nsuper);

    let supermap = Ws::new(supermap);
    /* Map: size n, use Flag as workspace for Map array */
    let map = Ws::new(flag);
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

    worker(
        a,
        beta,
        l,
        cwork,
        supermap,
        Ws::new(relative_map),
        Ws::new(next),
        Ws::new(lpos),
        Ws::new(next_save),
        Ws::new(lpos_save),
        map,
        head,
        QUICK_RETURN_IF_NOT_POSDEF,
    );

    /* Flag array was used as workspace, clear it */
    work.reset_flag();
    Ok(())
}

/// `Common->quick_return_if_not_posdef` (`t_cholmod_defaults.c:48`).
const QUICK_RETURN_IF_NOT_POSDEF: bool = false;

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
    use super::super::symbolic::{analyze_sparse, permute_sym, Sparse};
    use super::super::testcorpus::{corpus, spd_triangle};
    use super::super::ws::{columns_are_sorted, Work};
    use super::*;

    /// One corpus matrix, factorized supernodally with the batching threshold
    /// pinned, the way `mod.rs` drives it. Returns the factor and how many
    /// updates took the batched arm.
    fn factor(
        n: usize,
        edges: &[(usize, usize)],
        ordering: Ordering,
        par_flops: f64,
    ) -> (SuperFactor, usize) {
        let (p, i, v) = spd_triangle(n, edges, false);
        let a = Sparse {
            n,
            p: p.clone(),
            i: i.clone(),
            x: v.clone(),
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(n, &p, &i),
        };
        let s = analyze_sparse(&a, ordering, true, IntWidth::I64).unwrap();
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
        let mut l = SuperFactor::new(&s, sym);
        let mut cw = SuperWork::with_par_flops(par_flops);
        super_factorize(&a, 0.0, &mut l, &mut w, &mut cw).unwrap();
        (l, cw.wide)
    }

    /// The two arms of [`apply_updates`] have to produce the same `L->x`, entry
    /// for entry — not to a tolerance.
    ///
    /// That is the whole claim the batching rests on: computing several `C`s at
    /// once reorders nothing, because each one reads a different descendant and
    /// they are assembled into the supernode in the link list's order either
    /// way. `f64::INFINITY` pins the one-at-a-time loop and `0.0` pins the
    /// batched one, so both are exercised on every corpus matrix rather than
    /// whichever one the default threshold happens to select.
    #[test]
    fn batching_the_updates_does_not_move_a_rounding() {
        let mut batched = 0;
        for (name, n, edges) in corpus() {
            for ordering in [Ordering::Amd, Ordering::Natural] {
                let (serial, none) = factor(n, &edges, ordering, f64::INFINITY);
                let (wide, some) = factor(n, &edges, ordering, 0.0);
                assert_eq!(none, 0, "{name} went wide with the batching disabled");
                assert_eq!(serial.minor, wide.minor, "{name} minor");
                assert_eq!(serial.x, wide.x, "{name} L->x");
                batched += some;
            }
        }
        assert!(
            batched > 0,
            "the batched arm never ran: the test proves nothing"
        );
    }
}
