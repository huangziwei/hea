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

use super::dense::{gemm_nt, potrf_l, syrk_ln, trsm_rlt};
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
/// (`:245`), kept across calls instead.
///
/// It is `L->maxcsize` doubles, which is a property of the *symbolic* factor —
/// so a caller refactorizing the same pattern repeatedly should hold one of
/// these, exactly as it holds one [`Work`].
#[derive(Debug, Default)]
pub struct SuperWork {
    c: Vec<f64>,
}

impl SuperWork {
    pub fn new() -> SuperWork {
        SuperWork { c: Vec::new() }
    }

    /// Grow to `maxcsize`, never shrink — `cholmod_allocate_dense` is a fresh
    /// allocation per call upstream, but only its size matters.
    fn ensure(&mut self, maxcsize: usize) {
        if self.c.len() < maxcsize {
            self.c.resize(maxcsize, 0.0);
        }
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
    c: &mut [f64],
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

            //------------------------------------------------------------------
            // construct the update matrix C for this supernode d
            //------------------------------------------------------------------

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

            let ndrow3 = ndrow2 - ndrow1; /* number of rows of C2 */
            debug_assert!(ndrow3 >= 0);

            /* C1 = L1*L1' */
            syrk_ln(
                ndrow1 as usize, /* N: L1 is ndrow1-by-ndcol */
                ndcol as usize,  /* K */
                &l.x[pdx1 as usize..],
                ndrow as usize, /* A, LDA: L1, ndrow */
                c,
                ndrow2 as usize, /* C, LDC: C1 */
            );

            /* C2 = L2*L1' */
            if ndrow3 > 0 {
                gemm_nt(
                    ndrow3 as usize, /* M */
                    ndrow1 as usize, /* N */
                    ndcol as usize,  /* K */
                    &l.x[(pdx1 + ndrow1) as usize..],
                    ndrow as usize, /* A, LDA: L2 */
                    &l.x[pdx1 as usize..],
                    ndrow as usize, /* B, LDB: L1 */
                    &mut c[ndrow1 as usize..],
                    ndrow2 as usize, /* C, LDC: C2 */
                );
            }

            /* construct relative map to supernode s */
            for i in 0..ndrow2 {
                relative_map[i] = map[ls[pdi1 + i]];
                debug_assert!(relative_map[i] >= 0 && relative_map[i] < nsrow);
            }

            /* assemble C into supernode s using the relative map */
            {
                let lx = Ws::new(&mut l.x);
                let cw = Ws::new_ref(c);
                for j in 0..ndrow1 {
                    /* cols k1:k2-1 */
                    let px = psx + relative_map[j] * nsrow;
                    for i in j..ndrow2 {
                        /* rows k1:n-1 */
                        let q = px + relative_map[i];
                        lx[q] -= cw[i + ndrow2 * j];
                    }
                }
            }

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
        &mut cwork.c,
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
