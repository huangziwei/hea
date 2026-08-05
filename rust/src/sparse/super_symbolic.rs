//! Supernodal symbolic analysis — supernode detection, relaxed amalgamation,
//! and the nonzero pattern of the supernodal `L`.
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `CHOLMOD/Supernodal/cholmod_super_symbolic.c` → [`super_symbolic`], [`subtree`]
//!   * `CHOLMOD/Cholesky/cholmod_analyze.c:886-901`  → [`auto_supernodal`]
//!
//! **Scope: `A->stype > 0`.** Upstream rejects `stype < 0` outright ("symmetric
//! lower not supported", `:189-194`) and handles `stype == 0` by analyzing
//! `A*F`, which needs the `fset` machinery [`super::symbolic`] does not build
//! either; that branch is not ported and [`SuperError::Unsymmetric`] is returned
//! instead. `for_whom` is fixed at `CHOLMOD_ANALYZE_FOR_CHOLESKY`, so `L->px`
//! and `L->maxcsize` are always computed and the SPQR-only arms (`:737-740`,
//! `:892-893`) collapse. The GPU arms are `#ifdef CHOLMOD_HAS_CUDA` and this
//! crate has no CUDA build, so they are not branches to port: with
//! `L->useGPU == 0` neither the supernode-splitting test at `:419-422` nor the
//! merge veto at `:576-586` exists.
//!
//! **This routine reads `A` only through its pattern**, and it needs the *same*
//! `A` the numeric factorization will get — upstream's `S`, i.e. `triu(A(p,p))`
//! in column form, which [`super::numeric::factorize`] builds with the same two
//! transposes. It does not permute anything itself and does not look at
//! `L->Perm`.

use super::symbolic::Sparse;
use super::ws::{clear_flag, Work, Ws, EMPTY};

/// `Common->supernodal_switch` (`t_cholmod_defaults.c:43`).
pub const DEFAULT_SUPERNODAL_SWITCH: f64 = 40.0;

/// `cholmod_analyze`'s supernodal-vs-simplicial choice at
/// `Common->supernodal == CHOLMOD_AUTO`, which is the default
/// (`cholmod_analyze.c:887-890`).
///
/// `fl` and `lnz` are [`super::symbolic::Symbolic`]'s, i.e. the flop count and
/// nnz(`L`) the column counts imply. Their ratio is the average number of flops
/// per entry of `L`: below the switch the supernodes would be too small for
/// dense kernels to pay for themselves.
#[inline]
pub fn auto_supernodal(fl: f64, lnz: f64, supernodal_switch: f64) -> bool {
    lnz > 0.0 && (fl / lnz) >= supernodal_switch
}

/// `Common->nrelax` and `Common->zrelax` (`t_cholmod_defaults.c:54-59`) — the
/// thresholds that decide whether two adjacent supernodes are merged even
/// though that stores explicit zeros.
///
/// Supernodes `s` and `s+1` merge if the merged column count `ns` is at most
/// `nrelax[0]`, or if the merge adds no zeros at all, or if one of the three
/// `(ns, z)` pairs below admits it, where `z` is the resulting fraction of
/// explicit zeros (`cholmod.h:467-478`).
#[derive(Clone, Copy, Debug)]
pub struct Relax {
    pub nrelax: [i64; 3],
    pub zrelax: [f64; 3],
}

impl Default for Relax {
    fn default() -> Relax {
        Relax {
            nrelax: [4, 16, 48],
            zrelax: [0.8, 0.1, 0.05],
        }
    }
}

/// Why a supernodal analysis could not be performed.
#[derive(Debug)]
pub enum SuperError {
    /// Upstream's own `CHOLMOD_INVALID` (`:192`).
    Invalid(&'static str),
    /// `A->stype == 0`: upstream would analyze `A*F`, which this port does not
    /// build.
    Unsymmetric,
    /// `L->ssize` or `L->xsize` would not fit in an `Int` (`:674-677`).
    TooLarge,
}

impl core::fmt::Display for SuperError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SuperError::Invalid(m) => write!(f, "{m}"),
            SuperError::Unsymmetric => write!(
                f,
                "stype must be positive: this port analyzes LL' = A for a \
                 symmetric A, not LL' = AA'"
            ),
            SuperError::TooLarge => write!(f, "supernodal L is too large to index"),
        }
    }
}

/// The supernodal part of a `cholmod_factor`, as `cholmod_super_symbolic`
/// leaves it: `L->nsuper`, `L->super`, `L->pi`, `L->px`, `L->s`, `L->maxcsize`
/// and `L->maxesize`.
///
/// The numeric part (`L->x`, `L->xsize` doubles) is not allocated here, which
/// is the point of the split: one symbolic analysis is reused across every
/// factorization of a matrix with the same pattern.
#[derive(Debug, Clone)]
pub struct SuperSymbolic {
    pub n: usize,
    /// `L->nsuper`, the number of supernodes after relaxed amalgamation.
    pub nsuper: usize,
    /// `L->super`, size `nsuper+1`. Supernode `s` holds columns
    /// `super[s] .. super[s+1]-1` of `L`.
    pub sup: Vec<i64>,
    /// `L->pi`, size `nsuper+1`. The row indices of supernode `s` are
    /// `s[pi[s] .. pi[s+1]-1]`, in increasing order, the first `nscol` of them
    /// being the supernode's own columns.
    pub pi: Vec<i64>,
    /// `L->px`, size `nsuper+1`. Supernode `s` occupies `x[px[s] .. px[s+1]-1]`
    /// as an `nsrow`-by-`nscol` dense column-major block.
    pub px: Vec<i64>,
    /// `L->s`, size [`SuperSymbolic::ssize`].
    pub s: Vec<i64>,
    /// `L->ssize` and `L->xsize` — `s.len()`, and the number of doubles the
    /// numeric factorization will need. Both are at least 1, as upstream
    /// forces (`:675-676`).
    pub ssize: usize,
    pub xsize: usize,
    /// `L->maxcsize`, the largest update block any supernode contributes to an
    /// ancestor: how big the numeric factorization's `C` workspace has to be.
    pub maxcsize: usize,
    /// `L->maxesize`, the largest number of row indices a supernode has outside
    /// its own columns: the column dimension of the supernodal solve's `E`
    /// workspace.
    pub maxesize: usize,
}

/// Traverse the kth row subtree from the nonzeros in `A (0:k1-1,k)` and add the
/// new entries found to the pattern of the kth row of `L`. The current
/// supernode `s` contains the diagonal block `k1:k2-1`, so it can be skipped.
///
/// If `A` is sorted, then the total time taken by this function is proportional
/// to the number of nonzeros in the strictly block upper triangular part of
/// `A`, plus the number of entries in the strictly block lower triangular part
/// of the supernodal part of `L`.
///
/// Only adds column indices corresponding to the leading columns of each
/// relaxed supernode.
///
/// Upstream takes `j` and `k` separately because the unsymmetric case calls it
/// once per nonzero `F(j,k)`; in the symmetric case `j == k` always (`:80`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn subtree(
    /* inputs, not modified: */
    k: i64, /* j = k for symmetric case */
    ap: &Ws,
    ai: &Ws,
    supermap: &Ws,
    sparent: &Ws,
    mark: i64,
    sorted: bool, /* true if the columns of A are sorted */
    k1: i64,      /* only consider A (0:k1-1,k) */
    /* input/output: */
    flag: &mut Ws,
    ls: &mut Ws,
    lpi2: &mut Ws,
) {
    let j = k;
    /* `Anz == NULL`: a [`Sparse`] is always packed */
    for &i in ai.range(ap[j], ap[j + 1]) {
        if i < k1 {
            /* (i,k) is an entry in the upper triangular part of A: A(i,k) is
             * nonzero.
             *
             * Column i is in supernode si = SuperMap [i].  Follow path from si
             * to root of supernodal etree, stopping at the first flagged
             * supernode.  The root of the row subtree is supernode SuperMap[k],
             * which is flagged already. This traversal will stop there, or it
             * might stop earlier if supernodes have been flagged by previous
             * calls to this routine for the same k. */
            let mut si = supermap[i];
            while flag[si] < mark {
                debug_assert!(si <= supermap[k]);
                ls[lpi2[si]] = k;
                lpi2[si] += 1;
                flag[si] = mark;
                si = sparent[si];
            }
        } else if sorted {
            break;
        }
    }
}

/// `cholmod_super_symbolic` (`cholmod_super_symbolic.c:147-938`), for a
/// symmetric `A`.
///
/// `a` is upstream's `S` — `triu(A(p,p))` in column form, packed, sorted or
/// not. `parent` and `colcount` are the simplicial elimination tree and column
/// counts [`super::symbolic::analyze`] returns, in the same final ordering; the
/// C takes them as `Parent` and `L->ColCount` and says so at `:32-33`.
///
/// Upstream writes its result into the `cholmod_factor` it is given, converting
/// it from simplicial symbolic to supernodal symbolic in place
/// (`change_factor (CHOLMOD_PATTERN, TRUE, TRUE, TRUE, TRUE, L)`, `:686`). Here
/// it is returned instead: nothing in the simplicial factor survives that
/// conversion except `Perm` and `ColCount`, which the caller already holds.
pub fn super_symbolic(
    a: &Sparse,
    parent: &[i64],
    colcount: &[i64],
    relax: &Relax,
    work: &mut Work,
) -> Result<SuperSymbolic, SuperError> {
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if a.stype < 0 {
        /* invalid symmetry; symmetric lower form not supported */
        return Err(SuperError::Invalid("symmetric lower not supported"));
    }
    if a.stype == 0 {
        /* F must be present in the unsymmetric case */
        return Err(SuperError::Unsymmetric);
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    let n = a.n;
    let ap = Ws::new_ref(&a.p);
    let ai = Ws::new_ref(&a.i);
    let parent = Ws::new_ref(parent);
    let colcount = Ws::new_ref(colcount);

    let [nrelax0, nrelax1, nrelax2] = relax.nrelax;
    let [zrelax0, zrelax1, zrelax2] = relax.zrelax.map(|z| if z.is_nan() { 0.0 } else { z });

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    /* Sparent, Snz, and Merged could be allocated later, of size nfsuper */
    let Work {
        iwork,
        flag,
        head,
        mark,
        ..
    } = work;
    let (wi, rest) = iwork[..5 * n].split_at_mut(n); /* size n */
    let (wj, rest) = rest.split_at_mut(n); /* size n */
    let (sparent, rest) = rest.split_at_mut(n); /* size nfsuper <= n [ */
    let (snz, merged) = rest.split_at_mut(n); /* size nfsuper <= n [ */
    let (wi, wj) = (Ws::new(wi), Ws::new(wj));
    let (sparent, snz, merged) = (Ws::new(sparent), Ws::new(snz), Ws::new(merged));
    let flag = Ws::new(flag); /* size n */
    let head: &mut [i64] = head; /* size n+1 */

    //--------------------------------------------------------------------------
    // find the fundamental supernodes
    //--------------------------------------------------------------------------

    /* count the number of children of each node, using Wi [ */
    wi.fill(0);
    for j in 0..n {
        let p = parent[j];
        if p != EMPTY {
            wi[p] += 1;
        }
    }

    /* use Head [0..nfsuper] as workspace for Super list ( */
    let fsuper = Ws::new(&mut *head);

    /* column 0 always starts a new supernode */
    let mut nfsuper: i64 = if n == 0 { 0 } else { 1 }; /* number of fundamental supernodes */
    fsuper[0usize] = 0;

    for j in 1..n as i64 {
        /* check if j starts new supernode, or in the same supernode as j-1 */
        if parent[j - 1] != j                       /* parent of j-1 is not j */
            || (colcount[j - 1] != colcount[j] + 1) /* j-1 not subset of j */
            || wi[j] > 1
        /* j has more than one child */
        {
            /* j is the leading node of a supernode */
            fsuper[nfsuper] = j;
            nfsuper += 1;
        }
    }
    fsuper[nfsuper] = n as i64;

    /* contents of Wi no longer needed for child count ] */

    let nscol = &mut *wi; /* use Wi as size-nfsuper workspace for Nscol [ */

    //--------------------------------------------------------------------------
    // find the mapping of fundamental nodes to supernodes
    //--------------------------------------------------------------------------

    let supermap = &mut *wj; /* use Wj as workspace for SuperMap [ */

    /* SuperMap [k] = s if column k is contained in supernode s */
    for s in 0..nfsuper {
        for k in fsuper[s]..fsuper[s + 1] {
            supermap[k] = s;
        }
    }

    //--------------------------------------------------------------------------
    // construct the fundamental supernodal etree
    //--------------------------------------------------------------------------

    for s in 0..nfsuper {
        let j = fsuper[s + 1] - 1; /* last node in supernode s */
        let p = parent[j]; /* parent of last node */
        sparent[s] = if p == EMPTY { EMPTY } else { supermap[p] };
    }

    /* contents of Wj no longer needed as workspace for SuperMap ]
     * SuperMap will be recomputed below, for the relaxed supernodes. */

    let zeros = &mut *wj; /* use Wj for Zeros, workspace of size nfsuper [ */

    //--------------------------------------------------------------------------
    // relaxed amalgamation
    //--------------------------------------------------------------------------

    for s in 0..nfsuper {
        merged[s] = EMPTY; /* s not merged into another */
        nscol[s] = fsuper[s + 1] - fsuper[s]; /* # of columns in s */
        zeros[s] = 0; /* # of zero entries in s */
        debug_assert!(s <= fsuper[s]);
        snz[s] = colcount[fsuper[s]]; /* # of entries in leading col of s */
    }

    for s in (0..nfsuper - 1).rev() {
        /* should supernodes s and s+1 merge into a new node s? */

        if sparent[s] == EMPTY {
            /* s is a root, no merge with s+1 */
            continue;
        }

        /* find the current parent of s (perform path compression as needed) */
        let mut ss = sparent[s];
        while merged[ss] != EMPTY {
            ss = merged[ss];
        }
        let sparent_s = ss;

        /* ss is the current parent of s */
        let mut ss = sparent[s];
        while merged[ss] != EMPTY {
            /* ss is dead, merged into snext */
            let snext = merged[ss];
            merged[ss] = sparent_s;
            ss = snext;
        }

        /* if s+1 is not the current parent of s, do not merge */
        if sparent_s != s + 1 {
            continue;
        }

        let nscol0 = nscol[s]; /* # of columns in s */
        let nscol1 = nscol[s + 1]; /* # of columns in s+1 */
        let ns = nscol0 + nscol1;

        let mut totzeros = zeros[s + 1]; /* current # of zeros in s+1 */
        let lnz1 = snz[s + 1] as f64; /* # entries in leading column of s+1 */

        /* determine if supernodes s and s+1 should merge */
        let merge = if ns <= nrelax0 {
            /* ns is tiny, so go ahead and merge */
            true
        } else {
            /* use double to avoid integer overflow; approximations are OK */
            let lnz0 = snz[s] as f64; /* # entries in leading column of s */
            let xnewzeros = nscol0 as f64 * (lnz1 + nscol0 as f64 - lnz0);

            /* use Int for the final update of Zeros [s] below */
            let newzeros = nscol0 * (snz[s + 1] + nscol0 - snz[s]);
            debug_assert_eq!(newzeros as f64, xnewzeros);

            if xnewzeros == 0.0 {
                /* no new zeros, so go ahead and merge */
                true
            } else {
                /* # of zeros if merged */
                let xtotzeros = totzeros as f64 + xnewzeros;

                /* xtotsize: total size of merged supernode, if merged: */
                let xns = ns as f64;
                let xtotsize = (xns * (xns + 1.0) / 2.0) + xns * (lnz1 - nscol1 as f64);
                let z = xtotzeros / xtotsize;

                /* use Int for the final update of Zeros [s] below */
                totzeros += newzeros;

                /* do not merge if supernode would become too big
                 * (Int overflow).  Continue computing; not (yet) an error.
                 * fl.pt. compare, but no NaN's can occur here */
                ((ns <= nrelax1 && z < zrelax0) || (ns <= nrelax2 && z < zrelax1) || (z < zrelax2))
                    && (xtotsize < (i64::MAX as f64) / (size_of::<f64>() as f64))
            }
        };

        if merge {
            zeros[s] = totzeros;
            merged[s + 1] = s;
            snz[s] = nscol0 + snz[s + 1];
            nscol[s] += nscol[s + 1];
        }
    }

    /* contents of Wj no longer needed for Zeros ]
     * contents of Wi no longer needed for Nscol ]
     * contents of Sparent no longer needed (recomputed below) */

    //--------------------------------------------------------------------------
    // construct the relaxed supernode list
    //--------------------------------------------------------------------------

    let mut nsuper: i64 = 0;
    for s in 0..nfsuper {
        if merged[s] == EMPTY {
            /* live supernode */
            fsuper[nsuper] = fsuper[s];
            snz[nsuper] = snz[s];
            nsuper += 1;
        }
    }
    fsuper[nsuper] = n as i64;
    let nsuper = nsuper as usize;

    /* Merged no longer needed ] */

    //--------------------------------------------------------------------------
    // find the mapping of relaxed nodes to supernodes
    //--------------------------------------------------------------------------

    let supermap = &mut *wj; /* use Wj as workspace for SuperMap { */

    /* SuperMap [k] = s if column k is contained in supernode s */
    for s in 0..nsuper {
        for k in fsuper[s]..fsuper[s + 1] {
            supermap[k] = s as i64;
        }
    }

    //--------------------------------------------------------------------------
    // construct the relaxed supernodal etree
    //--------------------------------------------------------------------------

    for s in 0..nsuper {
        let j = fsuper[s + 1] - 1; /* last node in supernode s */
        let p = parent[j]; /* parent of last node */
        sparent[s] = if p == EMPTY { EMPTY } else { supermap[p] };
    }

    //--------------------------------------------------------------------------
    // determine the size of L->s and L->x
    //--------------------------------------------------------------------------

    /* do the computations in 64-bits to guard against integer overflow.
     * Upstream threads an `ok` flag through `add_size_t`/`mult_uint64_t` and
     * keeps going; saturating here is the same predicate, because both are
     * compared against Int_max and nothing else reads them. */
    let mut ssize: u64 = 0;
    let mut xsize: u64 = 0;
    for s in 0..nsuper {
        let nscol = (fsuper[s + 1] - fsuper[s]) as u64;
        let nsrow = snz[s] as u64;
        ssize = ssize.saturating_add(nsrow);
        xsize = xsize.saturating_add(nscol.saturating_mul(nsrow));
    }

    if !(ssize < i64::MAX as u64 && xsize < i64::MAX as u64) {
        /* upstream returns FALSE with L left a valid simplicial symbolic
         * factor; here there is nothing to leave behind */
        return Err(SuperError::TooLarge);
    }
    let ssize = ssize.max(1) as usize;
    let xsize = xsize.max(1) as usize;

    //--------------------------------------------------------------------------
    // allocate L (all except real part L->x)
    //--------------------------------------------------------------------------

    let mut lsuper = vec![0i64; nsuper + 1];
    let mut lpi = vec![0i64; nsuper + 1];
    let mut lpx = vec![0i64; nsuper + 1];
    let mut ls = vec![0i64; ssize];
    ls[0] = 0; /* flag for cholmod_check_factor; supernodes are defined */

    /* copy the list of relaxed supernodes into the final list in L */
    lsuper.copy_from_slice(&head[..nsuper + 1]);

    /* Head no longer needed as workspace for fundamental Super list ) */

    let maxcsize;
    let maxesize;
    {
        /* Super is now the list of relaxed supernodes */
        let sup = Ws::new_ref(&lsuper);
        let (lpi, lpx, ls) = (Ws::new(&mut lpi), Ws::new(&mut lpx), Ws::new(&mut ls));

        //----------------------------------------------------------------------
        // construct column pointers of relaxed supernodal pattern (L->pi)
        //----------------------------------------------------------------------

        let mut p: i64 = 0;
        for s in 0..nsuper {
            lpi[s] = p;
            p += snz[s];
        }
        lpi[nsuper] = p;
        debug_assert_eq!(ssize, p.max(1) as usize);

        //----------------------------------------------------------------------
        // construct pointers for supernodal values (L->px)
        //----------------------------------------------------------------------

        /* `Lpx [0] = 123456`, upstream's "ignore Lpx" marker for
         * `cholmod_check_factor`, is the non-GPU QR case only (`:735-742`) */
        let mut p: i64 = 0;
        for s in 0..nsuper {
            let nscol = sup[s + 1] - sup[s]; /* number of columns in s */
            let nsrow = snz[s]; /* # of rows, incl triangular part */
            lpx[s] = p; /* pointer to numerical part of s */
            p += nscol * nsrow;
        }
        lpx[nsuper] = p;
        debug_assert_eq!(xsize, p.max(1) as usize);

        /* Snz no longer needed ] */

        //----------------------------------------------------------------------
        // symbolic analysis to construct the relaxed supernodal pattern (L->s)
        //----------------------------------------------------------------------

        let lpi2 = &mut *wi; /* copy Lpi into Lpi2, using Wi as workspace for Lpi2 [ */
        for s in 0..nsuper {
            lpi2[s] = lpi[s];
        }

        let asorted = a.sorted;

        for s in 0..nsuper {
            /* sth supernode is in columns k1 to k2-1.
             * compute nonzero pattern of L (k1:k2-1,:). */

            /* place rows k1 to k2-1 in leading column of supernode s */
            let k1 = sup[s];
            let k2 = sup[s + 1];
            for k in k1..k2 {
                ls[lpi2[s]] = k;
                lpi2[s] += 1;
            }

            /* compute nonzero pattern each row k1 to k2-1 */
            for k in k1..k2 {
                /* compute row k of L.  In the symmetric case, the pattern of
                 * L(k,:) is the set of nodes reachable in the supernodal etree
                 * from any row i in the nonzero pattern of A(0:k,k). */

                /* clear the Flag array and mark the current supernode */
                clear_flag(flag, mark);
                flag[s] = *mark;
                debug_assert_eq!(s as i64, supermap[k]);

                /* traverse the row subtree for each nonzero in A */
                subtree(
                    k, ap, ai, supermap, sparent, *mark, asorted, k1, flag, ls, lpi2,
                );
            }
        }

        debug_assert!((0..nsuper).all(|s| lpi2[s] == lpi[s + 1]));

        /* contents of Wi no longer needed for Lpi2 ]
         * Sparent no longer needed ] */

        //----------------------------------------------------------------------
        // determine the largest update matrix (L->maxcsize)
        //----------------------------------------------------------------------

        /* The csize for a supernode is the size of its largest contribution to
         * a subsequent ancestor supernode; maxcsize is the largest of those
         * over the whole matrix. maxesize is the largest number of row indices
         * a supernode has below its own columns. */

        let mut mcsize: i64 = 1;
        let mut mesize: i64 = 1;

        for d in 0..nsuper {
            let nscol = sup[d + 1] - sup[d];
            let mut p = lpi[d] + nscol;
            let mut plast = p;
            let pend = lpi[d + 1];
            let esize = pend - p;
            mesize = mesize.max(esize);
            let mut slast = if p == pend { EMPTY } else { supermap[ls[p]] };
            while p <= pend {
                let s = if p == pend { EMPTY } else { supermap[ls[p]] };
                if s != slast {
                    /* row i is the start of a new supernode */
                    let ndrow1 = p - plast;
                    let ndrow2 = pend - plast;
                    let csize = ndrow2 * ndrow1;
                    mcsize = mcsize.max(csize);
                    plast = p;
                    slast = s;
                }
                p += 1;
            }
        }

        /* Wj no longer needed for SuperMap } */

        maxcsize = mcsize as usize;
        maxesize = mesize as usize;
    }

    //--------------------------------------------------------------------------
    // supernodal symbolic factorization is complete
    //--------------------------------------------------------------------------

    /* FREE_WORKSPACE */
    clear_flag(flag, mark);
    head[..=nfsuper as usize].fill(EMPTY);

    Ok(SuperSymbolic {
        n,
        nsuper,
        sup: lsuper,
        pi: lpi,
        px: lpx,
        s: ls,
        ssize,
        xsize,
        maxcsize,
        maxesize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::amd::IntWidth;
    use crate::sparse::symbolic::{analyze_sparse, permute_sym, Ordering, Symbolic};
    use crate::sparse::testcorpus::{corpus, triangle_csc};
    use crate::sparse::ws::Work;

    /// The corpus, analyzed and then handed to [`super_symbolic`] exactly as
    /// `cholmod_analyze`'s supernodal branch would (`:892-898`).
    fn analyses() -> Vec<(String, Symbolic, SuperSymbolic)> {
        let mut out = Vec::new();
        for (name, n, edges) in corpus() {
            for stype in [1i32, -1] {
                for order in [Ordering::Amd, Ordering::Natural] {
                    let (p, i) = triangle_csc(n, &edges, stype < 0);
                    let a = Sparse {
                        n,
                        p,
                        i,
                        x: Vec::new(),
                        numeric: false,
                        stype,
                        sorted: true,
                    };
                    let s = analyze_sparse(&a, order, true, IntWidth::I64).unwrap();
                    let mut work = Work::new(n);
                    let a2 = permute_sym(&a, s.ordering, &s.perm, false, false, &mut work.all());
                    let ss = super_symbolic(
                        a2.as_ref().unwrap_or(&a),
                        &s.parent,
                        &s.colcount,
                        &Relax::default(),
                        &mut work,
                    )
                    .unwrap();
                    out.push((format!("{name} stype={stype} {order:?}"), s, ss));
                }
            }
        }
        out
    }

    /// The debug-build sweep the [`Ws`] contract asks for: every index this
    /// kernel computes from data is bounds-checked under `cargo test`, so
    /// walking the whole corpus here is what licenses the unchecked accesses in
    /// release.
    #[test]
    fn the_supernodal_pattern_is_self_consistent() {
        for (tag, _, ss) in analyses() {
            let n = ss.n as i64;

            /* the supernodes partition the columns, in order */
            assert_eq!(ss.sup.len(), ss.nsuper + 1, "{tag}");
            assert_eq!(ss.sup[0], 0, "{tag}");
            assert_eq!(ss.sup[ss.nsuper], n, "{tag}");
            for s in 0..ss.nsuper {
                assert!(ss.sup[s] < ss.sup[s + 1], "{tag}: supernode {s} is empty");
            }

            /* pi and px are the running sums their loops make them */
            assert_eq!(ss.pi[0], 0, "{tag}");
            assert_eq!(ss.px[0], 0, "{tag}");
            assert_eq!(ss.pi[ss.nsuper].max(1) as usize, ss.ssize, "{tag}");
            assert_eq!(ss.px[ss.nsuper].max(1) as usize, ss.xsize, "{tag}");
            assert_eq!(ss.s.len(), ss.ssize, "{tag}");
            assert!(ss.maxcsize >= 1 && ss.maxesize >= 1, "{tag}");

            for s in 0..ss.nsuper {
                let (k1, k2) = (ss.sup[s], ss.sup[s + 1]);
                let nscol = k2 - k1;
                let rows = &ss.s[ss.pi[s] as usize..ss.pi[s + 1] as usize];
                let nsrow = rows.len() as i64;

                /* the numeric block is nsrow-by-nscol dense */
                assert_eq!(ss.px[s + 1] - ss.px[s], nscol * nsrow, "{tag} s={s}");

                /* the leading nscol rows are the supernode's own columns; the
                 * rest are strictly below them and strictly increasing */
                assert!(nsrow >= nscol, "{tag} s={s}");
                for (t, &r) in rows.iter().enumerate() {
                    if (t as i64) < nscol {
                        assert_eq!(r, k1 + t as i64, "{tag} s={s} row {t}");
                    } else {
                        assert!(r > rows[t - 1] && r < n, "{tag} s={s} row {t}");
                    }
                }

                assert!((nsrow - nscol) as usize <= ss.maxesize, "{tag} s={s}");
            }
        }
    }

    /// The property that makes a dense block valid at all: every column the
    /// supernode holds has its simplicial pattern inside the supernode's row
    /// list. Relaxed amalgamation only ever *adds* rows, so `ColCount` bounds
    /// what is left after the leading `k - k1` rows are skipped.
    #[test]
    fn each_supernode_covers_the_columns_it_holds() {
        for (tag, s, ss) in analyses() {
            for t in 0..ss.nsuper {
                let k1 = ss.sup[t] as usize;
                let nsrow = ss.pi[t + 1] - ss.pi[t];
                for k in k1..ss.sup[t + 1] as usize {
                    assert!(
                        s.colcount[k] <= nsrow - (k - k1) as i64,
                        "{tag} supernode {t} column {k}"
                    );
                }
            }
        }
    }

    /// The workspace is left as every kernel in this module promises to leave
    /// it — `Flag` cleared and `Head` all `EMPTY` — so the next user of the
    /// same [`Work`] sees what `cholmod_allocate_work` would have given it.
    /// That is `FREE_WORKSPACE` (`:130-139`), and the numeric factorization
    /// takes the same `Work` straight after.
    #[test]
    fn the_workspace_is_restored() {
        let (name, n, edges) = corpus().into_iter().find(|c| c.0 == "arrow-300").unwrap();
        let (p, i) = triangle_csc(n, &edges, false);
        let a = Sparse {
            n,
            p,
            i,
            x: Vec::new(),
            numeric: false,
            stype: 1,
            sorted: true,
        };
        let s = analyze_sparse(&a, Ordering::Amd, true, IntWidth::I64).unwrap();
        let mut work = Work::new(n);
        let a2 = permute_sym(&a, s.ordering, &s.perm, false, false, &mut work.all());
        assert!(
            work.is_pristine(),
            "{name}: analyze left the workspace dirty"
        );
        super_symbolic(
            a2.as_ref().unwrap_or(&a),
            &s.parent,
            &s.colcount,
            &Relax::default(),
            &mut work,
        )
        .unwrap();
        assert!(work.is_pristine(), "{name}: super_symbolic left it dirty");
    }

    /// A lower-triangular `A` is upstream's own `CHOLMOD_INVALID` (`:189-194`);
    /// `stype == 0` is this port's scope boundary.
    #[test]
    fn the_wrong_stype_is_rejected() {
        for stype in [-1i32, 0] {
            let a = Sparse {
                n: 1,
                p: vec![0, 1],
                i: vec![0],
                x: Vec::new(),
                numeric: false,
                stype,
                sorted: true,
            };
            let mut work = Work::new(1);
            let e = super_symbolic(&a, &[EMPTY], &[1], &Relax::default(), &mut work);
            match (&e, stype) {
                (Err(SuperError::Invalid(_)), -1) | (Err(SuperError::Unsymmetric), 0) => {}
                _ => panic!("stype {stype} gave {e:?}"),
            }
        }
    }

    /// `n == 0`: `nfsuper` and `nsuper` are both 0, and the two sizes are
    /// floored at 1 rather than left empty (`:675-676`).
    #[test]
    fn an_empty_matrix_analyzes_to_no_supernodes() {
        let a = Sparse {
            n: 0,
            p: vec![0],
            i: Vec::new(),
            x: Vec::new(),
            numeric: false,
            stype: 1,
            sorted: true,
        };
        let mut work = Work::new(0);
        let ss = super_symbolic(&a, &[], &[], &Relax::default(), &mut work).unwrap();
        assert_eq!(ss.nsuper, 0);
        assert_eq!(ss.sup, vec![0]);
        assert_eq!((ss.ssize, ss.xsize), (1, 1));
        assert_eq!((ss.maxcsize, ss.maxesize), (1, 1));
    }

    /// `Common->supernodal == CHOLMOD_AUTO`'s predicate, at both ends and at
    /// the switch itself.
    #[test]
    fn the_auto_switch_is_the_flops_per_entry_ratio() {
        let sw = DEFAULT_SUPERNODAL_SWITCH;
        assert!(
            !auto_supernodal(1e9, 0.0, sw),
            "lnz == 0 is never supernodal"
        );
        assert!(!auto_supernodal(39.0, 1.0, sw));
        assert!(auto_supernodal(40.0, 1.0, sw), "the switch is inclusive");
        assert!(auto_supernodal(4000.0, 50.0, sw));
    }
}
