//! Numeric factorization — the up-looking simplicial `LDL'`/`LL'`, and the
//! `cholmod_factor` object it fills.
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `CHOLMOD/Cholesky/cholmod_rowfac.c`            → [`rowfac`] (the `SUBTREE` macro)
//!   * `CHOLMOD/Cholesky/t_cholmod_rowfac_worker.c`   → [`rowfac`]
//!   * `CHOLMOD/Cholesky/cholmod_factorize.c`         → [`factorize`] (its simplicial branch)
//!   * `CHOLMOD/Utility/t_cholmod_change_factor.c`    → [`Factor::change_factor`]
//!   * `CHOLMOD/Utility/t_cholmod_reallocate_column.c`→ [`Factor::reallocate_column`]
//!   * `CHOLMOD/Utility/t_cholmod_reallocate_factor.c`→ [`Factor::reallocate_factor`]
//!   * `CHOLMOD/Utility/t_cholmod_pack_factor.c`      → [`Factor::pack`]
//!   * `CHOLMOD/Utility/t_cholmod_bound.c`            → [`Params::dbound_of`]
//!
//! **Scope.** `A->stype > 0` (upper), `CHOLMOD_REAL` + `CHOLMOD_DOUBLE`, and
//! the no-`mask` instantiation. Upstream compiles the worker twelve times
//! (`{real, complex, zomplex} x {double, single} x {mask, no mask}`,
//! `cholmod_rowfac.c:159-194`); the other eleven are not built here, the same
//! way `stype == 0` is not built in [`super::symbolic`]. `mask`/`RLinkUp` exist
//! only for LPDASA (`cholmod_rowfac.c:610,637`) and are unreachable through
//! `cholmod_factorize`.
//!
//! The `Int` width is upstream's `int64_t` build, i.e. `cholmod_l_rowfac`.
//! Nothing here is width-observable the way AMD's `hash` is — the only place
//! `Int_max` appears is [`grow_l`]'s clamp, which no reachable problem size
//! comes near.

use crate::nmath::util::rfma;

use super::symbolic::{permute_sym, Ordering, Sparse, Symbolic};
use super::ws::{Work, WorkRef, Ws, EMPTY};

/// Why a factorization could not be performed. Not positive definite is *not*
/// one of these: upstream reports that through `L->minor < n` and carries on
/// (`t_cholmod_rowfac_worker.c:430-434`), and so does this.
#[derive(Debug)]
pub enum NumericError {
    /// `A->stype <= 0`, or `A` and `L` disagree on `n`.
    Invalid(&'static str),
    /// `L` would need more entries than an `Int` can address
    /// (`t_cholmod_change_factor.c:602`).
    TooLarge,
}

impl core::fmt::Display for NumericError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NumericError::Invalid(m) => write!(f, "{m}"),
            NumericError::TooLarge => write!(f, "problem too large"),
        }
    }
}

/* ========================================================================= */
/* === Common ============================================================== */
/* ========================================================================= */

/// The `cholmod_common` fields these routines read, at their
/// `cholmod_defaults` values (`t_cholmod_defaults.c:28-40`).
///
/// `grow0`/`grow1`/`grow2` decide how much slack each column of `L` and `L`
/// itself are given; `dbound` is the floor `LDL'` clamps `|D(j,j)|` to, off at
/// its default of 0. They are grouped rather than passed loose because that is
/// what they are upstream — state on `Common`, shared by
/// `change_factor`/`reallocate_column`/`pack_factor`, and mutated across a call
/// by [`factorize`] exactly as `cholmod_factorize_p` mutates it (`:386-394`).
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub grow0: f64,
    pub grow1: f64,
    pub grow2: i64,
    pub dbound: f64,
    /// `Common->final_ll`: factorize to `LL'` rather than `LDL'`.
    pub final_ll: bool,
    /// `Common->final_asis`: leave `L` in whatever form `rowfac` produced.
    /// True at its default, which is what makes the `final_pack` /
    /// `final_monotonic` conversion below a no-op.
    pub final_asis: bool,
    pub final_pack: bool,
    pub final_monotonic: bool,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            grow0: 1.2,
            grow1: 1.2,
            grow2: 5,
            dbound: 0.0,
            final_ll: false,
            final_asis: true,
            final_pack: true,
            final_monotonic: true,
        }
    }
}

impl Params {
    /// `cholmod_dbound` (`t_cholmod_bound.c:23-78`). Only reached when
    /// `dbound > 0`, which is `use_bound` in the worker.
    #[inline]
    fn dbound_of(&self, djj: f64) -> f64 {
        if djj.is_nan() {
            /* no change if D(j,j) is NaN */
            return djj;
        }
        let (bound, hit) = if djj >= 0.0 {
            /* D(j,j) is positive: check if djj in range [0,Common->bound] */
            (self.dbound, djj < self.dbound)
        } else {
            /* D(j,j) is negative: check if djj in range [-Common->bound,0] */
            (-self.dbound, djj > -self.dbound)
        };
        if hit {
            bound
        } else {
            djj
        }
    }
}

/* ========================================================================= */
/* === cholmod_factor ====================================================== */
/* ========================================================================= */

/// A `cholmod_factor`, simplicial only.
///
/// `xtype` is `CHOLMOD_PATTERN` while [`Factor::p`] and the rest are empty
/// (upstream's "simplicial symbolic": just `Perm` and `ColCount`), and
/// `CHOLMOD_REAL` once [`Factor::change_factor`] has allocated them. There is
/// no supernodal form here — `L->is_super` is always false, so the fields that
/// only exist for it (`super`, `pi`, `px`, `s`, `maxcsize`, `maxesize`) are not
/// carried.
///
/// `L->nzmax` is not a field: in this port it is `i.len()`, which cannot drift
/// from the allocation the way a separately tracked count can.
#[derive(Debug, Clone)]
pub struct Factor {
    pub n: usize,
    /// `L->Perm`, the fill-reducing ordering `cholmod_analyze` chose.
    pub perm: Vec<i64>,
    /// `L->ColCount`, the exact nnz of each column of `L` under that ordering.
    pub colcount: Vec<i64>,
    /// `L->ordering`.
    pub ordering: Ordering,
    /// `L->is_ll`: `LL'` if set, `LDL'` if not.
    pub is_ll: bool,
    /// `L->is_monotonic`: the columns appear in `L->i` in order `0..n-1`.
    pub is_monotonic: bool,
    /// `L->minor`. `n` if the factorization succeeded; otherwise the first
    /// column at which `A` was found not to be positive definite.
    pub minor: usize,
    /// `L->xtype != CHOLMOD_PATTERN`: the arrays below are allocated.
    pub numeric: bool,
    /// `L->p`, size `n+1`. Column `j` starts at `p[j]`; it *ends* at
    /// `p[j] + nz[j]`, which is not `p[j+1]` unless `L` is packed.
    pub p: Vec<i64>,
    /// `L->nz`, size `n`: entries currently in column `j`, diagonal included.
    pub nz: Vec<i64>,
    /// `L->prev`, size `n+2`.
    pub prev: Vec<i64>,
    /// `L->next`, size `n+2`. The doubly-linked list of columns in the order
    /// they occupy `L->i`, with head `n+1` and tail `n`.
    pub next: Vec<i64>,
    /// `L->i`, size `L->nzmax`.
    pub i: Vec<i64>,
    /// `L->x`, size `L->nzmax`.
    pub x: Vec<f64>,
}

impl Factor {
    /// `cholmod_alloc_factor` (`t_cholmod_alloc_factor.c:26-97`) followed by
    /// what `cholmod_analyze` writes into it: the simplicial symbolic factor.
    pub fn from_symbolic(s: &Symbolic) -> Factor {
        Factor {
            n: s.perm.len(),
            perm: s.perm.clone(),
            colcount: s.colcount.clone(),
            ordering: s.ordering,
            /* calloc'd header: LDL', not supernodal */
            is_ll: false,
            is_monotonic: true,
            minor: s.perm.len(),
            numeric: false,
            p: Vec::new(),
            nz: Vec::new(),
            prev: Vec::new(),
            next: Vec::new(),
            i: Vec::new(),
            x: Vec::new(),
        }
    }

    /// `L->nzmax`.
    #[inline]
    pub fn nzmax(&self) -> usize {
        self.i.len()
    }

    /* The C caches `Lp`, `Lnz`, `Li`, `Lx` in locals and walks them as raw
     * pointers, refreshing them after `reallocate_column` moves the arrays.
     * Rust cannot hold those across a `&mut self` call, so each is taken as a
     * momentary borrow instead — same generated access, and the refresh the C
     * has to remember to do falls out of it. */

    #[inline(always)]
    fn wp(&self) -> &Ws {
        Ws::new_ref(&self.p)
    }

    #[inline(always)]
    fn wnz(&self) -> &Ws {
        Ws::new_ref(&self.nz)
    }

    #[inline(always)]
    fn wnext(&self) -> &Ws {
        Ws::new_ref(&self.next)
    }

    #[inline(always)]
    fn wxv(&self) -> &Ws<f64> {
        Ws::new_ref(&self.x)
    }

    #[inline(always)]
    fn wi_mut(&mut self) -> &mut Ws {
        Ws::new(&mut self.i)
    }

    #[inline(always)]
    fn wx_mut(&mut self) -> &mut Ws<f64> {
        Ws::new(&mut self.x)
    }

    #[inline(always)]
    fn wnz_mut(&mut self) -> &mut Ws {
        Ws::new(&mut self.nz)
    }

    /// `natural_list` (`t_cholmod_change_factor.c:193-223`): link the columns
    /// in order `0, 1, ... n-1`, with the head at `n+1` and the tail at `n`.
    fn natural_list(&mut self) {
        let n = self.n as i64;
        let (next, prev) = (Ws::new(&mut self.next), Ws::new(&mut self.prev));

        /* create the head node */
        let head = n + 1;
        next[head] = 0;
        prev[head] = EMPTY;

        /* create the tail node */
        let tail = n;
        next[tail] = EMPTY;
        prev[tail] = n - 1;

        /* link columns 0 to n-1 in increasing order: 0, 1, 2, ... n-1 */
        for j in 0..n {
            next[j] = j + 1;
            prev[j] = j - 1;
        }

        /* the prev node of the first column 0 is n+1 */
        prev[0i64] = head;

        /* the columns appear in order 0, 1, 2, ... n-1 in the link list */
        self.is_monotonic = true;
    }

    /// `alloc_simplicial_num` (`t_cholmod_change_factor.c:277-334`): the four
    /// size-`n` arrays, not `L->(i,x)`.
    fn alloc_simplicial_num(&mut self) {
        let n = self.n;
        self.p = vec![0; n + 1];
        self.nz = vec![0; n];
        self.prev = vec![0; n + 2];
        self.next = vec![0; n + 2];
        self.natural_list();
    }

    /// `simplicial_sym_to_simplicial_num` (`t_cholmod_change_factor.c:455-649`)
    /// — allocate `L` from `ColCount` and set it to the identity.
    ///
    /// `packed` is upstream's `> 0` / `== 0`; its third case (`< 0`, "allocate
    /// but do not initialize") exists only for `super_num_to_simplicial_num`,
    /// which this port has no supernodal form to reach.
    fn simplicial_sym_to_simplicial_num(
        &mut self,
        to_ll: bool,
        packed: bool,
        params: &Params,
    ) -> Result<(), NumericError> {
        self.alloc_simplicial_num();

        let n = self.n as i64;
        let mut ok = true;
        let mut lnz = 0i64;

        if packed {
            /* initialize the packed LL' or LDL' case (L is identity) */
            let mut j = 0;
            while ok && j < n {
                /* ensure ColCount [j] is in the range 1 to n-j */
                let len = self.colcount[j as usize].max(1).min(n - j);
                lnz += len;
                ok = lnz >= 0;
                j += 1;
            }
            /* each column L(:,j) holds a single diagonal entry */
            for j in 0..=n as usize {
                self.p[j] = j as i64;
            }
            self.nz.fill(1);
        } else {
            /* initialize the unpacked LL' or LDL' case (L is identity);
             * slack space will be added to L below */
            let grow0 = if params.grow0.is_nan() {
                1.0
            } else {
                params.grow0
            };
            let grow1 = if params.grow1.is_nan() {
                1.0
            } else {
                params.grow1
            };
            let grow2 = params.grow2 as f64;
            let grow = grow0 >= 1.0 && grow1 >= 1.0 && grow2 > 0.0;

            let mut j = 0;
            while ok && j < n {
                /* log the start of L(:,j), containing a single entry */
                self.p[j as usize] = lnz;
                self.nz[j as usize] = 1;

                /* ensure ColCount [j] is in the range 1 to n-j */
                let mut len = self.colcount[j as usize].max(1).min(n - j);

                /* add some slack space to L(:,j) */
                if grow {
                    len = grow_column(len, grow1, grow2, n - j);
                }
                lnz += len;
                ok = lnz >= 0;
                j += 1;
            }

            /* add slack space at the end of L */
            if ok {
                self.p[n as usize] = lnz;
                if grow {
                    lnz = grow_l(lnz, grow0, n);
                }
            }
        }

        if !ok {
            return Err(NumericError::TooLarge);
        }

        /* allocate L->i and L->x with the new xtype and existing dtype */
        let lnz = lnz.max(1) as usize;
        self.i = vec![0; lnz];
        self.x = vec![0.0; lnz];
        self.numeric = true;
        self.minor = self.n;

        /* set L to the identity matrix (change_factor_1_worker, `:19-56`) */
        for j in 0..self.n {
            let p = self.p[j];
            self.i[p as usize] = j as i64;
            self.x[p as usize] = 1.0;
        }

        self.is_ll = to_ll;
        Ok(())
    }

    /// `cholmod_change_factor` (`t_cholmod_change_factor.c:1122-1292`) for the
    /// two conversions this port can reach: simplicial symbolic → simplicial
    /// numeric, and simplicial numeric → simplicial numeric.
    ///
    /// The supernodal arms and the "convert to `CHOLMOD_PATTERN`" arm are not
    /// built: there is no supernodal factor here, and nothing in
    /// `cholmod_factorize`'s simplicial path throws `L` away.
    pub fn change_factor(
        &mut self,
        to_ll: bool,
        to_packed: bool,
        to_monotonic: bool,
        params: &Params,
    ) -> Result<(), NumericError> {
        if !self.numeric {
            /* convert simplicial symbolic to simplicial numeric (L=D=I) */
            self.simplicial_sym_to_simplicial_num(to_ll, to_packed, params)
        } else {
            /* change a simplicial numeric factor: change LL' to LDL', LDL' to
             * LL', or leave as-is.  pack the columns of L, or leave as-is.
             * Ensure the columns are monotonic, or leave as-is. */
            self.change_simplicial_num(to_ll, to_packed, to_monotonic, params)
        }
    }

    /// `change_simplicial_num` (`t_cholmod_change_factor.c:673-869`).
    fn change_simplicial_num(
        &mut self,
        to_ll: bool,
        to_packed: bool,
        to_monotonic: bool,
        params: &Params,
    ) -> Result<(), NumericError> {
        let out_of_place = (to_packed || to_monotonic) && !self.is_monotonic;
        let make_ll = to_ll && !self.is_ll;
        let make_ldl = !to_ll && self.is_ll;

        let n = self.n as i64;
        let grow0 = if params.grow0.is_nan() {
            1.0
        } else {
            params.grow0
        };
        let grow1 = if params.grow1.is_nan() {
            1.0
        } else {
            params.grow1
        };
        let grow2 = params.grow2 as f64;

        let mut grow = false;
        let mut lnz = 0i64;
        let (mut i2, mut x2) = (Vec::new(), Vec::new());

        if out_of_place {
            /* The columns of L are out of order (not monotonic), but L is being
             * changed to being either monotonic, or packed, or both.  Thus, L
             * needs to be resized, in newly allocated space. */
            if !to_packed {
                grow = grow0 >= 1.0 && grow1 >= 1.0 && grow2 > 0.0;
            }

            let mut ok = true;
            let mut j = 0;
            while ok && j < n {
                let mut len = self.nz[j as usize];
                if grow {
                    len = grow_column(len, grow1, grow2, n - j);
                }
                lnz += len;
                ok = lnz >= 0;
                j += 1;
            }
            if !ok {
                return Err(NumericError::TooLarge);
            }

            /* add additional space at the end of L, if requested */
            if grow {
                lnz = grow_l(lnz, grow0, n);
            }
            let cap = lnz.max(1) as usize;
            i2 = vec![0i64; cap];
            x2 = vec![0.0f64; cap];
        }

        if out_of_place {
            self.change_factor_2::<OUT_OF_PLACE>(
                &mut i2, &mut x2, grow, grow1, grow2, make_ll, make_ldl,
            );
        } else if to_packed {
            self.change_factor_2::<TO_PACKED>(
                &mut i2, &mut x2, grow, grow1, grow2, make_ll, make_ldl,
            );
        } else {
            self.change_factor_2::<IN_PLACE>(
                &mut i2, &mut x2, grow, grow1, grow2, make_ll, make_ldl,
            );
        }

        self.is_ll = to_ll;

        if out_of_place {
            /* free the old space and move the new space into L */
            self.i = i2;
            self.x = x2;
            /* revise the link list (columns 0 to n-1 now in natural order) */
            self.natural_list();
        }
        Ok(())
    }

    /// `t_cholmod_change_factor_2_template.c` — convert `L` to/from `LL'` and
    /// `LDL'`, in place, packed in place, or into new space.
    ///
    /// `MODE` is the template's three `#include`s of the same body under
    /// `OUT_OF_PLACE` / `TO_PACKED` / `IN_PLACE` (`_2_worker.c:84,99,114`), so
    /// it is const: each is a separate function upstream, and the `#ifndef
    /// IN_PLACE` guards vanish rather than becoming branches.
    #[allow(clippy::too_many_arguments)]
    fn change_factor_2<const MODE: u8>(
        &mut self,
        i2: &mut [i64],
        x2: &mut [f64],
        grow: bool,
        grow1: f64,
        grow2: f64,
        make_ll: bool,
        make_ldl: bool,
    ) {
        let n = self.n as i64;
        if make_ll {
            /* will be set below to the min j where D(j,j) <= 0 */
            self.minor = self.n;
        }
        let mut pnew = 0i64;

        for j in 0..n {
            let p = self.p[j as usize];
            let mut len = self.nz[j as usize];
            let p_new = if MODE == IN_PLACE { p } else { pnew };

            if make_ll {
                /* convert an LDL' factorization to LL' */
                let djj = self.x[p as usize];
                if djj <= 0.0 {
                    /* The matrix is not positive-definite and cannot be
                     * converted to LL'.  The column L(:,j) is moved to its new
                     * space but not numerically modified so it can be converted
                     * back to a valid LDL' factorization. */
                    if self.minor == self.n {
                        self.minor = j as usize;
                    }
                    if MODE != IN_PLACE {
                        self.move_entries::<MODE>(i2, x2, p_new, p, len);
                    }
                } else {
                    /* L(j,j) = sqrt (D(j,j)) */
                    let ljj = djj.sqrt();
                    if MODE != IN_PLACE {
                        write_i::<MODE>(&mut self.i, i2, p_new, j);
                    }
                    write_x::<MODE>(&mut self.x, x2, p_new, ljj);

                    /* L(j+1:n,j) = L(j+1:n,j) * L(j,j) */
                    for k in 1..len {
                        if MODE != IN_PLACE {
                            let v = self.i[(p + k) as usize];
                            write_i::<MODE>(&mut self.i, i2, p_new + k, v);
                        }
                        let v = self.x[(p + k) as usize] * ljj;
                        write_x::<MODE>(&mut self.x, x2, p_new + k, v);
                    }
                }
            } else if make_ldl {
                /* convert an LL' factorization to LDL' */
                let ljj = self.x[p as usize];
                if ljj <= 0.0 {
                    /* do not modify L(:,j), just copy it to its new place */
                    if MODE != IN_PLACE {
                        self.move_entries::<MODE>(i2, x2, p_new, p, len);
                    }
                } else {
                    /* D(j,j) = L(j,j)^2 */
                    if MODE != IN_PLACE {
                        write_i::<MODE>(&mut self.i, i2, p_new, j);
                    }
                    write_x::<MODE>(&mut self.x, x2, p_new, ljj * ljj);

                    /* L(j+1:n) = L(j+1:n) / L(j,j) */
                    for k in 1..len {
                        if MODE != IN_PLACE {
                            let v = self.i[(p + k) as usize];
                            write_i::<MODE>(&mut self.i, i2, p_new + k, v);
                        }
                        let v = self.x[(p + k) as usize] / ljj;
                        write_x::<MODE>(&mut self.x, x2, p_new + k, v);
                    }
                }
            } else {
                /* factorization remains as is, but space may be revised */
                if MODE == IN_PLACE {
                    continue;
                }
                if MODE != TO_PACKED || p_new < p {
                    self.move_entries::<MODE>(i2, x2, p_new, p, len);
                    self.p[j as usize] = p_new;
                }
                if MODE == OUT_OF_PLACE && grow {
                    len = grow_column(len, grow1, grow2, n - j);
                }
                pnew = p_new + len;
                continue;
            }

            /* grow column L(:,j) if requested, and advance to column j+1 */
            if MODE != IN_PLACE {
                self.p[j as usize] = p_new;
                if MODE == OUT_OF_PLACE && grow {
                    len = grow_column(len, grow1, grow2, n - j);
                }
                pnew = p_new + len;
            }
        }

        /* log the end of the last column */
        if MODE != IN_PLACE {
            self.p[self.n] = pnew;
        }
    }

    /// The `for (k = 0 ; k < len ; k++) { Li_NEW [p_NEW+k] = Li [p+k] ; ... }`
    /// that every arm of the template above shares.
    #[inline]
    fn move_entries<const MODE: u8>(
        &mut self,
        i2: &mut [i64],
        x2: &mut [f64],
        pdest: i64,
        psrc: i64,
        len: i64,
    ) {
        for k in 0..len {
            let (iv, xv) = (self.i[(psrc + k) as usize], self.x[(psrc + k) as usize]);
            write_i::<MODE>(&mut self.i, i2, pdest + k, iv);
            write_x::<MODE>(&mut self.x, x2, pdest + k, xv);
        }
    }

    /// `cholmod_reallocate_factor` (`t_cholmod_reallocate_factor.c:16-56`):
    /// change the max # of nonzeros `L` can hold.
    fn reallocate_factor(&mut self, nznew: usize) {
        /* ensure L can hold at least 1 entry */
        let nznew = nznew.max(1);
        self.i.resize(nznew, 0);
        self.x.resize(nznew, 0.0);
    }

    /// `cholmod_pack_factor` (`t_cholmod_pack_factor.c:54-115` and its worker
    /// `:17-91`): squeeze the gaps between columns down to `grow2` slack each,
    /// leaving all the free space at the tail of `L->i` and `L->x`.
    ///
    /// The columns are walked through the link list, not `0..n`: they are in
    /// whatever order `reallocate_column` left them in.
    pub fn pack(&mut self, params: &Params) {
        if !self.numeric {
            /* nothing to do */
            return;
        }
        let n = self.n as i64;
        let slack = params.grow2;

        /* first column in the list is Lnext [n+1] */
        let mut j = self.next[(n + 1) as usize];
        /* next column can move to pnew */
        let mut pnew = 0i64;

        while j != n {
            /* get column j, entries currently in Li,Lx [pold...pold+lnzj-1] */
            let pold = self.p[j as usize];
            let lnzj = self.nz[j as usize];

            /* pack column j, if possible */
            if pnew < pold {
                for k in 0..lnzj {
                    self.i[(pnew + k) as usize] = self.i[(pold + k) as usize];
                    self.x[(pnew + k) as usize] = self.x[(pold + k) as usize];
                }
                /* log the new position of the first entry of L(:,j) */
                self.p[j as usize] = pnew;
            }

            /* add some empty space at the end of column j */
            let desired_space = lnzj + slack;
            let max_space = n - j;
            let total_space = desired_space.min(max_space);

            /* next column will move to position pnew, if possible */
            let jnext = self.next[j as usize];
            let pnext = self.p[jnext as usize];
            let pthis = self.p[j as usize] + total_space;
            pnew = pthis.min(pnext);
            j = jnext;
        }
    }

    /// `cholmod_reallocate_column` (`t_cholmod_reallocate_column.c:52-205`):
    /// grow `L(:,j)` by moving it to the end of `L`, growing `L` itself first
    /// if there is no room there.
    ///
    /// Returns `false` only on the upstream error paths, which here means
    /// nothing — Rust's allocator aborts rather than returning null — so it is
    /// infallible in practice and kept `bool`-shaped only where upstream reads
    /// the result.
    fn reallocate_column(&mut self, j: i64, need: i64, params: &Params) {
        let n = self.n as i64;

        /* ensure need is in range 1:(n-j) and add slack space */
        let need = need.max(1);
        let slack =
            (params.grow1.max(1.0) * (need as f64) + params.grow2 as f64).min((n - j) as f64);
        let need = (need.max(slack as i64)).max(1).min(n - j);

        /* quick return if L(:,j) already big enough */
        let already_have = self.p[self.next[j as usize] as usize] - self.p[j as usize];
        if already_have >= need {
            return;
        }

        /* check if enough space at the end of L->i and L->x */
        let tail = n;
        let new_nzmax_required = need + self.p[tail as usize];
        if new_nzmax_required > self.nzmax() as i64 {
            /* out of space in L, so grow the entire factor to lnznew space */
            let grow0 = if params.grow0.is_nan() || params.grow0 < 1.2 {
                1.2
            } else {
                params.grow0
            };
            let xnz = grow0 * ((self.nzmax() as f64) + (need as f64) + 1.0);
            let lnznew = if xnz > usize::MAX as f64 {
                usize::MAX
            } else {
                xnz as usize
            };
            self.reallocate_factor(lnznew);

            /* repack all columns so each column has some slack space */
            self.pack(params);
        }

        /* move j to the end of the list */
        self.is_monotonic = false;
        let (jprev, jnext) = (self.prev[j as usize], self.next[j as usize]);
        self.next[jprev as usize] = jnext;
        self.prev[jnext as usize] = jprev;
        let tprev = self.prev[tail as usize];
        self.next[tprev as usize] = j;
        self.prev[j as usize] = tprev;
        self.next[j as usize] = tail;
        self.prev[tail as usize] = j;

        /* add space to L(:,j), now at the end of L */
        let psrc = self.p[j as usize];
        let pdest = self.p[tail as usize];
        self.p[j as usize] = pdest;
        self.p[tail as usize] += need;

        /* move L(:,j) to its new space at the end of L
         * (t_cholmod_reallocate_column_worker.c:13-42) */
        let len = self.nz[j as usize];
        for k in 0..len {
            self.i[(pdest + k) as usize] = self.i[(psrc + k) as usize];
            self.x[(pdest + k) as usize] = self.x[(psrc + k) as usize];
        }
    }
}

/// `t_cholmod_change_factor_2_template.c`'s `#ifdef OUT_OF_PLACE`.
const OUT_OF_PLACE: u8 = 0;
/// its `#ifdef TO_PACKED`.
const TO_PACKED: u8 = 1;
/// its `#ifdef IN_PLACE`.
const IN_PLACE: u8 = 2;

/// `Li_NEW [p] = v`, where `Li_NEW` is `Li2` out of place and `Li` otherwise.
#[inline]
fn write_i<const MODE: u8>(li: &mut [i64], li2: &mut [i64], p: i64, v: i64) {
    if MODE == OUT_OF_PLACE {
        li2[p as usize] = v;
    } else {
        li[p as usize] = v;
    }
}

/// `Lx_NEW [p] = v`.
#[inline]
fn write_x<const MODE: u8>(lx: &mut [f64], lx2: &mut [f64], p: i64, v: f64) {
    if MODE == OUT_OF_PLACE {
        lx2[p as usize] = v;
    } else {
        lx[p as usize] = v;
    }
}

/// `grow_column` (`t_cholmod_change_factor.c:116-125`).
#[inline]
fn grow_column(len: i64, grow1: f64, grow2: f64, maxlen: i64) -> i64 {
    let mut xlen = len as f64;
    xlen = grow1 * xlen + grow2;
    xlen = xlen.min(maxlen as f64);
    let len = xlen as i64;
    len.max(1).min(maxlen)
}

/// `grow_L` (`t_cholmod_change_factor.c:131-141`).
#[inline]
fn grow_l(lnz: i64, grow0: f64, n: i64) -> i64 {
    let mut xlnz = lnz as f64;
    xlnz *= grow0;
    xlnz = xlnz.min(i64::MAX as f64);
    let d = n as f64;
    let d = (d * d + d) / 2.0;
    xlnz = xlnz.min(d);
    lnz.max(xlnz as i64)
}

/* ========================================================================= */
/* === cholmod_rowfac ====================================================== */
/* ========================================================================= */

/// `cholmod_rowfac` for the symmetric-upper, real, double, no-`mask`
/// instantiation — `t_cholmod_rowfac_worker.c:19-455`.
///
/// Computes `L*D*L' = beta*I + A` (or `L*L'`) one **row** at a time, up-looking:
/// for each `k`, the pattern of row `k` of `L` is the `k`th row subtree of the
/// elimination tree, gathered by [`subtree`], and its values come from a sparse
/// triangular solve against the columns of `L` already built.
///
/// `A` must be symmetric-upper; only its diagonal and upper triangle are read.
/// A full factorization is `kstart = 0, kend = n`, which additionally resets
/// `L->nz` and `L->minor` (`:126-139`) so the same `L` can be refactorized.
///
/// Not positive definite is *not* an error: `L->minor` is set to the offending
/// column and the remaining rows are computed with a zero diagonal, which is
/// what makes `L->minor` meaningful to the caller.
///
/// Returns `Common->rowfacfl`, the flop count.
pub fn rowfac(
    a: &Sparse,
    beta: f64,
    kstart: usize,
    kend: usize,
    l: &mut Factor,
    params: &Params,
    work: &mut WorkRef<'_>,
) -> Result<f64, NumericError> {
    let n = a.n;
    if a.stype <= 0 {
        return Err(NumericError::Invalid(
            "rowfac needs the upper triangle of a symmetric A: \
             stype <= 0 is not supported",
        ));
    }
    if !a.numeric {
        return Err(NumericError::Invalid(
            "a pattern-only matrix cannot be factorized",
        ));
    }
    if n != l.n {
        return Err(NumericError::Invalid("dimensions of A and L do not match"));
    }
    if kend > l.n {
        return Err(NumericError::Invalid("kend invalid"));
    }

    let mut fl = 0.0f64;
    let use_bound = params.dbound > 0.0;
    let is_ll = l.is_ll;

    /* get the current factors L (and D for LDL'); allocate space if needed */
    if !l.numeric {
        /* L is symbolic only; allocate and initialize L (and D for LDL') */
        l.change_factor(is_ll, false, true, params)?;
    } else if kstart == 0 && kend == n {
        /* refactorization; reset L->nz and L->minor to restart factorization */
        l.minor = n;
        l.nz.fill(1);
    }

    /* `sorted` is a field of `cholmod_sparse` that upstream tests per entry
     * rather than templating on, so this does too. It matters only when `A`
     * carries entries in the half it does not claim to store: for a genuinely
     * upper `A` every entry of column k already has i <= k, so the early break
     * is never reached either way. */
    let sorted = a.sorted;
    let ap = Ws::new_ref(&a.p);
    let ai = Ws::new_ref(&a.i);
    let ax = Ws::new_ref(&a.x);

    /* get workspace (`t_cholmod_rowfac_worker.c:164-170`). Taken once, outside
     * the row loop, because that is where the C takes it: re-slicing per row
     * would add a bounds check and a reborrow to every one of the n
     * iterations. */
    let stack = Ws::new(&mut work.iwork[..n]);
    let flag = Ws::new(&mut work.flag[..n]);
    let wx = Ws::new(&mut work.xwork[..n]);
    let mark = &mut *work.mark;

    let n_i = n as i64;
    let kend = kend as i64;

    for k in kstart as i64..kend {
        /* ------------------------------------------------------------------ */
        /* compute pattern of kth row of L and scatter kth input column        */
        /* ------------------------------------------------------------------ */

        /* do not include diagonal entry in Stack */
        flag[k] = *mark;

        /* Stack is empty; scatter kth col of triu (beta*I+A), get pattern
         * L(k,:) */
        let top = subtree(
            k,
            ap[k],
            ap[k + 1],
            sorted,
            n_i,
            *mark,
            ai,
            ax,
            l,
            stack,
            flag,
            wx,
        );

        /* nonzero pattern of kth row of L is now in Stack [top..n-1].
         * Flag [Stack [top..n-1]] is equal to mark, but no longer needed.
         * `CLEAR_FLAG` (`cholmod_types.h:49-57`) rewrites Flag only if the
         * counter overflowed, which is what keeps this O(1) rather than O(n). */
        *mark += 1;
        if *mark <= 0 {
            *mark = 0;
            flag.fill(EMPTY);
        }

        /* ------------------------------------------------------------------ */
        /* compute kth row of L and store in column form                       */
        /* ------------------------------------------------------------------ */

        /* dk = W [k] + beta */
        let mut dk = wx[k] + beta;
        /* W [k] = 0.0 */
        wx[k] = 0.0;

        for s in top..n_i {
            /* get i for each nonzero entry L(k,i) */
            let i = stack[s];

            /* y = W [i] ; W [i] = 0.0 */
            let mut y = wx[i];
            wx[i] = 0.0;

            let lnz = l.wnz()[i];
            let mut p = l.wp()[i];
            let pend = p + lnz;

            /* di = Lx [p] ; the diagonal entry L or D(i,i), which is real */
            let di = l.wxv()[p];

            let lx: f64;
            if i >= l.minor as i64 || di == 0.0 {
                /* For the LL' factorization, L(i,i) is zero.  For the LDL',
                 * D(i,i) is zero.  Skip column i of L, and set L(k,i) = 0. */
                lx = 0.0;
                p = pend;
            } else if is_ll {
                fl += 2.0 * ((pend - p - 1) as f64) + 3.0;
                /* forward solve using L (i:(k-1),i); divide by L(i,i), which
                 * must be real and nonzero */
                y /= di;
                axpy(&l.i, &l.x, p + 1, pend, y, wx);
                p = pend;
                /* do not scale L; compute dot product for L(k,k) */
                lx = y;
                /* d -= conj(y) * y */
                dk = mulsub(dk, y, y);
            } else {
                fl += 2.0 * ((pend - p - 1) as f64) + 3.0;
                /* forward solve using D (i,i) and L ((i+1):(k-1),i) */
                axpy(&l.i, &l.x, p + 1, pend, y, wx);
                p = pend;
                /* Scale L (k,0:k-1) for LDL' factorization, compute D (k,k) */
                lx = y / di;
                dk = mulsub(dk, lx, y);
            }

            /* determine if column i of L can hold the new L(k,i) entry */
            if p >= l.wp()[l.wnext()[i]] {
                /* column i needs to grow */
                l.reallocate_column(i, lnz + 1, params);
                /* contents of L->p changed */
                p = l.wp()[i] + lnz;
            }

            /* store L (k,i) in the column form matrix of L */
            l.wi_mut()[p] = k;
            l.wx_mut()[p] = lx;
            l.wnz_mut()[i] += 1;
        }

        /* ------------------------------------------------------------------ */
        /* ensure abs (d) >= bound if dbound is given, and store it in L       */
        /* ------------------------------------------------------------------ */

        let p = l.wp()[k];
        l.wi_mut()[p] = k;

        if k >= l.minor as i64 {
            /* the matrix is already not positive definite */
            dk = 0.0;
        } else if use_bound {
            /* modify the diagonal to force LL' or LDL' to exist */
            dk = params.dbound_of(if is_ll { dk.abs() } else { dk });
        } else if if is_ll { dk <= 0.0 } else { dk == 0.0 } {
            /* the matrix has just been found to be not positive definite */
            dk = 0.0;
            l.minor = k as usize;
        }

        if is_ll {
            /* this is counted as one flop, below */
            dk = dk.sqrt();
        }

        /* Lx [p] = D(k,k) = d */
        l.wx_mut()[p] = dk;
    }

    if is_ll {
        /* count sqrt's */
        fl += (kend - kstart as i64).max(0) as f64;
    }
    Ok(fl)
}

/// `SUBTREE` (`cholmod_rowfac.c:127-153`) with `SCATTER` set to `W [i] = Ax [p]`
/// and `PARENT(i)` read off `L` itself (`t_cholmod_rowfac_worker.c:200,208`).
///
/// Walks `A(:,k)`, and from each entry `i <= k` climbs the elimination tree to
/// the first already-visited node, pushing the path. Each path is then copied
/// down to the top of the stack, so `Stack [top..n-1]` ends up in topological
/// order — which is what makes the triangular solve above it valid.
///
/// Returns the new `top`. `len` and `top` grow toward each other in the same
/// `Iwork`; they cannot meet, because the two regions together hold only the
/// nodes marked at this `k`, of which there are at most `n`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn subtree(
    k: i64,
    p: i64,
    pend: i64,
    sorted: bool,
    mut top: i64,
    mark: i64,
    ai: &Ws,
    ax: &Ws<f64>,
    l: &Factor,
    stack: &mut Ws,
    flag: &mut Ws,
    wx: &mut Ws<f64>,
) -> i64 {
    let (lp, lnz, li) = (Ws::new_ref(&l.p), Ws::new_ref(&l.nz), Ws::new_ref(&l.i));

    for pa in p..pend {
        let i = ai[pa];
        if i <= k {
            /* scatter the column of A into Wx */
            wx[i] = ax[pa];
            /* start at node i and traverse up the subtree, stop at node k */
            let mut len = 0i64;
            let mut i = i;
            while i < k && i != EMPTY && flag[i] < mark {
                /* L(k,i) is nonzero, and seen for the first time */
                stack[len] = i;
                len += 1;
                /* mark i as visited */
                flag[i] = mark;
                /* traverse up the etree to parent */
                i = if lnz[i] > 1 { li[lp[i] + 1] } else { EMPTY };
            }
            /* move the path down to the bottom of the stack */
            while len > 0 {
                top -= 1;
                len -= 1;
                let v = stack[len];
                stack[top] = v;
            }
        } else if sorted {
            break;
        }
    }
    top
}

/// `x - a*b`, contracted the way a C compiler contracts it.
///
/// `R_MULTSUB` is `x [p] -= ax [q] * bx [r]` (`cholmod_internal.h:253`), and
/// `-ffp-contract` defaults to on, so a C compiler with an FMA instruction
/// fuses that into one rounding. The shipped `libcholmod` is the receipt:
/// `cholmod_l_rowfac`'s inner loop disassembles to `fmsub d0, d0, d13, d1`, and
/// its two dot-product accumulators to `fmsub d8, d13, d13, d8` (`LLDOT`) and
/// `fmsub d8, d13, d0, d8` (the `LDL'` branch). Fusing moves the result by up
/// to an ulp and roughly half the entries of `L` take the difference, so this
/// is the whole gap between an un-fused port and upstream — not a rounding
/// footnote.
///
/// Which is why this defers to [`rfma`], the crate's one contraction policy:
/// fuse on `aarch64`, stay plain everywhere else. That policy was set for R's
/// nmath and holds here for the same reason — the reference build is a `clang
/// -O2` with no `-ffp-contract` flag, so it fuses exactly where the ISA has a
/// baseline FMA. Keying on `target_feature = "fma"` instead would be *less*
/// faithful, not more: it fires only when someone builds with `-C
/// target-cpu=native`, and the CHOLMOD they are being compared against is a
/// baseline x86-64 build that does not fuse.
#[inline(always)]
pub(super) fn mulsub(x: f64, a: f64, b: f64) -> f64 {
    /* negating a is exact, so this is `x - a*b` under one rounding */
    rfma(-a, b, x)
}

/// `for (p++ ; p < pend ; p++) W [Li [p]] -= Lx [p] * y`, the inner loop of the
/// sparse triangular solve.
///
/// Handed the two columns as slices rather than indexed per iteration, for the
/// reason [`Ws::range`] documents.
#[inline]
fn axpy(li: &[i64], lx: &[f64], p: i64, pend: i64, y: f64, wx: &mut Ws<f64>) {
    let (li, lx) = (Ws::new_ref(li), Ws::new_ref(lx));
    for (&i, &v) in li.range(p, pend).iter().zip(lx.range(p, pend)) {
        wx[i] = mulsub(wx[i], v, y);
    }
}

/* ========================================================================= */
/* === cholmod_factorize =================================================== */
/* ========================================================================= */

/// `cholmod_factorize_p`'s simplicial branch (`cholmod_factorize.c:297-408`),
/// for a symmetric `A`.
///
/// Permutes `A` into the upper form `rowfac` needs — which for a permuted
/// ordering means two transposes, because `ptranspose` of an upper matrix gives
/// the lower form and only the second gives back the upper one — then factorizes.
///
/// `Common->grow2` is set to 0 for the duration of the *first* factorization
/// (`:388-392`), which is what makes each column of `L` come out exactly
/// `ColCount [j]` long; without it the identity `L` is allocated with slack it
/// would never use.
pub fn factorize(
    a: &Sparse,
    beta: f64,
    l: &mut Factor,
    params: &Params,
    work: &mut Work,
) -> Result<f64, NumericError> {
    let nrow = a.n;
    if a.stype == 0 {
        return Err(NumericError::Invalid(
            "stype must be nonzero: this port factorizes LL' = A for a \
             symmetric A, not LL' = AA'",
        ));
    }
    if nrow != l.n {
        return Err(NumericError::Invalid("dimensions of A and L do not match"));
    }

    /* Permute the input matrix A if necessary.  cholmod_rowfac requires
     * triu(A) in column form for the symmetric case. */
    const VALUES: bool = true;
    const UPPER: bool = false;
    let a2 = permute_sym(a, l.ordering, &l.perm, VALUES, UPPER, &mut work.all());
    let s: &Sparse = a2.as_ref().unwrap_or(a);

    /* factorize beta*I+S */
    let mut facparams = *params;
    l.is_ll = params.final_ll;
    if !l.numeric && params.final_pack {
        /* allocate a factor with exactly the space required */
        facparams.grow2 = 0;
    }
    let fl = rowfac(s, beta, 0, nrow, l, &facparams, &mut work.all())?;
    /* Common->grow2 = grow2 — restored before anything else reads it */

    /* convert to final form, if requested */
    if !params.final_asis {
        l.change_factor(l.is_ll, params.final_pack, params.final_monotonic, params)?;
    }
    Ok(fl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::amd::IntWidth;
    use crate::sparse::symbolic::{analyze, Symbolic};
    use crate::sparse::testcorpus::{corpus, spd_triangle, Lcg};
    use crate::sparse::ws::Work;

    /// `A`, its symbolic analysis, and the factor it produces.
    fn setup(n: usize, edges: &[(usize, usize)], stype: i32) -> (Sparse, Symbolic) {
        let (p, i, x) = spd_triangle(n, edges, stype < 0);
        let s = analyze(n, &p, &i, stype, Ordering::Amd, true, IntWidth::I64)
            .expect("the corpus is well-formed");
        (
            Sparse {
                n,
                p,
                i,
                x,
                numeric: true,
                stype,
                sorted: true,
            },
            s,
        )
    }

    /// Dense `L` and `D` from the unpacked column form, so the reconstruction
    /// below reads the same entries `cholmod_solve` would.
    fn dense_ldl(l: &Factor) -> (Vec<f64>, Vec<f64>) {
        let n = l.n;
        let mut ld = vec![0.0f64; n * n];
        let mut d = vec![0.0f64; n];
        for j in 0..n {
            let p = l.p[j] as usize;
            d[j] = l.x[p];
            ld[j * n + j] = if l.is_ll { l.x[p] } else { 1.0 };
            for k in 1..l.nz[j] as usize {
                ld[j * n + l.i[p + k] as usize] = l.x[p + k];
            }
        }
        (ld, d)
    }

    /// `max |L*D*L'*v - P A P'*v|` over a few pseudo-random `v`, relative to
    /// `max |P A P'*v|`. A matvec rather than a dense product: the corpus runs
    /// to n = 1000 and this is a debug build, so an O(n^3) check would dominate
    /// the whole test suite's runtime to catch nothing extra.
    fn residual(a: &Sparse, l: &Factor) -> f64 {
        let n = l.n;
        let mut pinv = vec![0i64; n];
        for (k, &i) in l.perm.iter().enumerate() {
            pinv[i as usize] = k as i64;
        }
        let mut rng = Lcg(0xD15EA5E);
        let mut worst = 0.0f64;
        for _ in 0..4 {
            let v: Vec<f64> = (0..n)
                .map(|_| (rng.below(2000) as f64) / 1000.0 - 1.0)
                .collect();

            /* t = L'v, u = D t, z = L u */
            let mut t = vec![0.0f64; n];
            for j in 0..n {
                let p = l.p[j] as usize;
                let mut s = if l.is_ll { l.x[p] * v[j] } else { v[j] };
                for k in 1..l.nz[j] as usize {
                    s += l.x[p + k] * v[l.i[p + k] as usize];
                }
                t[j] = if l.is_ll { s } else { s * l.x[p] };
            }
            let mut z = vec![0.0f64; n];
            for j in 0..n {
                let p = l.p[j] as usize;
                z[j] += if l.is_ll { l.x[p] * t[j] } else { t[j] };
                for k in 1..l.nz[j] as usize {
                    z[l.i[p + k] as usize] += l.x[p + k] * t[j];
                }
            }

            /* w = (P A P') v, from the stored triangle */
            let mut w = vec![0.0f64; n];
            for j in 0..n {
                for p in a.p[j] as usize..a.p[j + 1] as usize {
                    let i = a.i[p] as usize;
                    let (pi, pj) = (pinv[i] as usize, pinv[j] as usize);
                    w[pi] += a.x[p] * v[pj];
                    if i != j {
                        w[pj] += a.x[p] * v[pi];
                    }
                }
            }

            let scale = w.iter().fold(1e-30f64, |m, x| m.max(x.abs()));
            for k in 0..n {
                worst = worst.max((z[k] - w[k]).abs() / scale);
            }
        }
        worst
    }

    /// The corpus, factorized both ways round, in a debug build — which is
    /// what makes [`Ws`]'s elided bounds checks evidence rather than a claim.
    #[test]
    fn factorize_never_indexes_out_of_bounds() {
        for (name, n, edges) in corpus() {
            for stype in [1i32, -1] {
                for final_ll in [false, true] {
                    let (a, s) = setup(n, &edges, stype);
                    let mut l = Factor::from_symbolic(&s);
                    let mut work = Work::new(n);
                    let params = Params {
                        final_ll,
                        ..Params::default()
                    };
                    factorize(&a, 0.0, &mut l, &params, &mut work).unwrap();
                    assert_eq!(l.minor, n, "{name}: diagonally dominant A went indefinite");
                    assert_eq!(l.is_ll, final_ll, "{name}");
                    /* the factorization has to be one, not merely in range */
                    let r = residual(&a, &l);
                    assert!(r < 1e-9, "{name}: residual {r:e}");
                    /* every column holds its diagonal first, and ColCount is
                     * exact, so `final_pack` sized each one to the entry */
                    for j in 0..n {
                        assert_eq!(l.i[l.p[j] as usize], j as i64, "{name}: col {j}");
                        assert_eq!(l.nz[j], l.colcount[j], "{name}: col {j}");
                    }
                }
            }
        }
    }

    /// A factor analyzed for one pattern and then handed a denser one is the
    /// only way `reallocate_column` — and through it `reallocate_factor` and
    /// `pack_factor` — is reached, since a first factorization sizes every
    /// column to its exact `ColCount`.
    #[test]
    fn a_denser_matrix_grows_the_columns_it_needs() {
        for (name, n, edges) in corpus() {
            if n < 8 {
                continue;
            }
            /* analyze a strict subset of the edges, factorize all of them */
            let sparse_edges: Vec<(usize, usize)> =
                edges.iter().copied().step_by(3).collect::<Vec<_>>();
            let (thin_p, thin_i, _) = spd_triangle(n, &sparse_edges, false);
            let s = analyze(n, &thin_p, &thin_i, 1, Ordering::Amd, true, IntWidth::I64).unwrap();
            let (a, _) = setup(n, &edges, 1);
            let mut l = Factor::from_symbolic(&s);
            let mut work = Work::new(n);
            factorize(&a, 0.0, &mut l, &Params::default(), &mut work).unwrap();
            assert_eq!(l.minor, n, "{name}");
            let r = residual(&a, &l);
            assert!(r < 1e-9, "{name}: residual {r:e}");

            /* and the non-monotonic factor that leaves behind converts back to
             * a packed monotonic one without changing a value */
            let (before, dbefore) = dense_ldl(&l);
            l.change_factor(false, true, true, &Params::default())
                .unwrap();
            assert!(l.is_monotonic, "{name}");
            for j in 0..n {
                assert_eq!(l.p[j + 1] - l.p[j], l.nz[j], "{name}: col {j} not packed");
            }
            let (after, dafter) = dense_ldl(&l);
            assert_eq!(
                (before, dbefore),
                (after, dafter),
                "{name}: packing moved a value"
            );
        }
    }

    /// `LDL' -> LL' -> LDL'` is not the identity in floating point, but it is
    /// on the pattern, and the values have to come back to within a few ulps.
    #[test]
    fn converting_between_ll_and_ldl_preserves_the_factorization() {
        for (name, n, edges) in corpus() {
            let (a, s) = setup(n, &edges, 1);
            let mut l = Factor::from_symbolic(&s);
            let mut work = Work::new(n);
            factorize(&a, 0.0, &mut l, &Params::default(), &mut work).unwrap();
            let before = dense_ldl(&l);
            l.change_factor(true, false, false, &Params::default())
                .unwrap();
            assert!(l.is_ll, "{name}");
            assert!(residual(&a, &l) < 1e-9, "{name}: LL' residual");
            l.change_factor(false, false, false, &Params::default())
                .unwrap();
            assert!(!l.is_ll, "{name}");
            let after = dense_ldl(&l);
            for (x, y) in before.0.iter().zip(&after.0) {
                assert!(
                    (x - y).abs() <= 8.0 * f64::EPSILON * x.abs().max(1e-300),
                    "{name}"
                );
            }
        }
    }

    /// Not positive definite is reported through `L->minor`, and the rows past
    /// it are still computed rather than left as garbage.
    #[test]
    fn an_indefinite_matrix_reports_where_it_failed() {
        /* [[1,2],[2,1]] is symmetric with eigenvalues 3 and -1 */
        let a = Sparse {
            n: 2,
            p: vec![0, 1, 3],
            i: vec![0, 0, 1],
            x: vec![1.0, 2.0, 1.0],
            numeric: true,
            stype: 1,
            sorted: true,
        };
        let s = analyze(2, &a.p, &a.i, 1, Ordering::Natural, true, IntWidth::I64).unwrap();
        let mut l = Factor::from_symbolic(&s);
        let mut work = Work::new(2);
        /* LDL' only fails on an exactly zero pivot, so this one succeeds with a
         * negative D — that is the documented behaviour, not a miss */
        factorize(&a, 0.0, &mut l, &Params::default(), &mut work).unwrap();
        assert_eq!(l.minor, 2);
        assert_eq!(l.x[l.p[1] as usize], -3.0);

        /* LL' cannot represent it, and stops at the offending column */
        let mut l = Factor::from_symbolic(&s);
        let params = Params {
            final_ll: true,
            ..Params::default()
        };
        factorize(&a, 0.0, &mut l, &params, &mut work).unwrap();
        assert_eq!(l.minor, 1);
    }

    /// `beta` shifts the diagonal without touching the pattern.
    #[test]
    fn beta_shifts_the_diagonal() {
        let (a, s) = setup(60, &corpus()[4].2, 1);
        let mut work = Work::new(60);
        let mut plain = Factor::from_symbolic(&s);
        factorize(&a, 0.0, &mut plain, &Params::default(), &mut work).unwrap();
        let mut shifted = Factor::from_symbolic(&s);
        factorize(&a, 2.5, &mut shifted, &Params::default(), &mut work).unwrap();
        assert_eq!(plain.nz, shifted.nz);
        assert_eq!(plain.i, shifted.i);
        /* every pivot grew, because A + 2.5 I is more dominant than A */
        for j in 0..60 {
            let (p, q) = (plain.p[j] as usize, shifted.p[j] as usize);
            assert!(shifted.x[q] > plain.x[p], "col {j}");
        }
    }

    /// The scope limits are rejected rather than silently mis-answered.
    #[test]
    fn unsupported_inputs_are_rejected() {
        let mut work = Work::new(1);
        let pattern_only = Sparse {
            n: 1,
            p: vec![0, 1],
            i: vec![0],
            x: Vec::new(),
            numeric: false,
            stype: 1,
            sorted: true,
        };
        let s = analyze(1, &[0, 1], &[0], 1, Ordering::Natural, true, IntWidth::I64).unwrap();
        let mut l = Factor::from_symbolic(&s);
        assert!(matches!(
            rowfac(
                &pattern_only,
                0.0,
                0,
                1,
                &mut l,
                &Params::default(),
                &mut work.all()
            ),
            Err(NumericError::Invalid(_))
        ));
        let lower = Sparse {
            x: vec![1.0],
            numeric: true,
            stype: -1,
            ..pattern_only
        };
        assert!(matches!(
            rowfac(
                &lower,
                0.0,
                0,
                1,
                &mut l,
                &Params::default(),
                &mut work.all()
            ),
            Err(NumericError::Invalid(_))
        ));
    }
}
