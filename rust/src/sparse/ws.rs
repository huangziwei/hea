//! The workspace primitive every kernel in this module indexes through, and
//! the input validation that licenses it.
//!
//! SuiteSparse's symbolic kernels are a dense mesh of `Iw [Pe [e]]`-style
//! indirections whose subscripts are values the algorithm itself wrote into
//! another workspace array a few lines earlier. Nothing in those data flows is
//! visible to the optimizer, so a plain `slice[i as usize]` costs a
//! compare-and-branch on every access — a measured 1.2-1.4x against the C,
//! which walks raw pointers. [`Ws`] is how the port gets that back, and
//! [`validate_csc`] is what makes it sound.

/// A workspace array indexed by `Int`, the way the C indexes it.
///
/// The bound is checked under `debug_assertions` and elided otherwise. That is
/// only worth anything if the debug build is actually exercised, so every
/// module using `Ws` keeps a `#[cfg(test)]` corpus that walks its branches —
/// `cargo test` is a debug build, so each of those asserts fires on every index
/// the corpus touches.
///
/// `T` defaults to `Int`, which is what the symbolic kernels index; the numeric
/// ones subscript `Common->Xwork` and `L->x` the same way, and get the same
/// treatment through `Ws<f64>`.
#[repr(transparent)]
pub struct Ws<T = i64>([T]);

impl<T> Ws<T> {
    /// Borrow a workspace slice. `#[inline(always)]` because the whole point is
    /// that this compiles to nothing.
    #[inline(always)]
    pub(super) fn new(s: &mut [T]) -> &mut Ws<T> {
        // SAFETY: `Ws` is `#[repr(transparent)]` over `[T]`.
        unsafe { &mut *(s as *mut [T] as *mut Ws<T>) }
    }

    #[inline(always)]
    pub(super) fn new_ref(s: &[T]) -> &Ws<T> {
        // SAFETY: `Ws` is `#[repr(transparent)]` over `[T]`.
        unsafe { &*(s as *const [T] as *const Ws<T>) }
    }

    #[inline(always)]
    pub(super) fn split_at_mut(&mut self, mid: usize) -> (&Ws<T>, &mut [T]) {
        let (lo, hi) = self.0.split_at_mut(mid);
        (Ws::new_ref(lo), hi)
    }

    #[inline]
    pub(super) fn fill(&mut self, v: T)
    where
        T: Copy,
    {
        self.0.fill(v);
    }

    #[inline(always)]
    pub(super) fn range(&self, lo: i64, hi: i64) -> &[T] {
        debug_assert!(
            lo >= 0 && hi >= lo && (hi as usize) <= self.0.len(),
            "Ws range {lo}..{hi} out of order or out of range for len {}",
            self.0.len()
        );
        // SAFETY: `hi >= lo` holds because `validate_csc` rejects a
        // non-monotone `indptr`, and `hi <= len` because it rejects an
        // `indptr[n]` past the end of `indices`.
        unsafe { self.0.get_unchecked(lo as usize..hi as usize) }
    }
}

macro_rules! ws_index {
    ($t:ty) => {
        impl<T> core::ops::Index<$t> for Ws<T> {
            type Output = T;
            #[inline(always)]
            fn index(&self, i: $t) -> &T {
                debug_assert!(
                    i as i64 >= 0 && (i as usize) < self.0.len(),
                    "Ws index {i} out of range for len {}",
                    self.0.len()
                );
                // SAFETY: checked in debug builds, which is where the corpus runs.
                unsafe { self.0.get_unchecked(i as usize) }
            }
        }

        impl<T> core::ops::IndexMut<$t> for Ws<T> {
            #[inline(always)]
            fn index_mut(&mut self, i: $t) -> &mut T {
                debug_assert!(
                    i as i64 >= 0 && (i as usize) < self.0.len(),
                    "Ws index {i} out of range for len {}",
                    self.0.len()
                );
                // SAFETY: checked in debug builds, which is where the corpus runs.
                unsafe { self.0.get_unchecked_mut(i as usize) }
            }
        }
    };
}

ws_index!(i64);
ws_index!(usize);
ws_index!(i32);

pub const EMPTY: i64 = -1;

/// `cholmod_clear_flag` (`t_cholmod_clear_flag.c:34-49`), taking the two pieces
/// of [`Work`] it touches rather than the whole of one.
///
/// [`super::super_symbolic`] needs it while `Head` is spoken for as the
/// fundamental supernode list, so it cannot go through a [`WorkRef`].
#[inline]
pub(super) fn clear_flag(flag: &mut Ws, mark: &mut i64) {
    *mark += 1;
    if *mark <= 0 {
        *mark = 0;
        flag.fill(EMPTY);
    }
}

/// `Common`'s persistent workspace — `cholmod_alloc_work`
/// (`Utility/t_cholmod_alloc_work.c:50-90`).
///
/// CHOLMOD allocates this **once** per `cholmod_analyze` and every routine it
/// calls carves its scratch out of it; `cholmod_analyze.c:483-484` says so
/// outright ("enough space needs to be allocated here so that routines called
/// by cholmod_analyze do not reallocate the space") and sets
/// `no_workspace_reallocate` to enforce it.
///
/// That is not a housekeeping detail, and it was measured rather than assumed.
/// With the same driver source and the same `amd_2` object code, allocating the
/// workspace per call instead of reusing it costs **1.06-1.18x** on these
/// matrices: a freshly mapped block is first-touch page faults that the reused
/// one has already paid, and `calloc`'s zero pages are lazy precisely so the
/// cost lands there instead. Pooling brought the driver onto `cholmod_amd` to
/// within noise (2.973 vs 3.049 ms, 0.890 vs 0.843, 3.702 vs 3.617).
///
/// Held as owned buffers by [`Work`] and handed to the kernels as a
/// [`WorkRef`], which is what lets `cholmod_analyze` keep `First` and `Level`
/// live across calls that are meanwhile scribbling on `Iwork [0..2n)` — the C
/// gets that for free from raw pointers into one block.
pub struct Work {
    /// `Common->Iwork`, `6*n` for the symmetric analyze path. Uninitialized on
    /// allocation, like the C's `malloc`.
    ///
    /// Only the ordering routines may use more than the first `2n`
    /// (`cholmod_analyze.c:509-514`): the rest holds `Parent`, `First`,
    /// `Level` and `Post` across their calls.
    pub(super) iwork: Vec<i64>,
    pub(super) flag: Vec<i64>,
    pub(super) head: Vec<i64>,
    pub(super) xwork: Vec<f64>,
    /// `Common->mark`. `Flag [i] < mark` is the invariant every user of `Flag`
    /// relies on; bumping `mark` is how a kernel invalidates the whole array in
    /// O(1) instead of rewriting it (`cholmod_types.h:49-57`).
    pub(super) mark: i64,
}

impl Work {
    pub fn new(n: usize) -> Work {
        let mut w = Work {
            iwork: Vec::new(),
            flag: Vec::new(),
            head: Vec::new(),
            xwork: Vec::new(),
            /* `Common->mark = 0` accompanies a fresh `Flag` (`:71`) */
            mark: 0,
        };
        w.allocate(n, 6 * n, n);
        w
    }

    /// The whole workspace. Only the ordering routines may take this
    /// (`cholmod_analyze.c:511-514`); everything else uses `Iwork [0..2n)` and
    /// leaves the last `4n` to `Parent`/`First`/`Level`/`Post`.
    pub(super) fn all(&mut self) -> WorkRef<'_> {
        WorkRef {
            iwork: &mut self.iwork,
            flag: &mut self.flag,
            head: &mut self.head,
            xwork: &mut self.xwork,
            mark: &mut self.mark,
        }
    }

    /// `cholmod_allocate_work (nrow, iworksize, xworksize, Common)` —
    /// `t_cholmod_alloc_work.c:50-109`. Each of the four arrays grows if it is
    /// short and is left alone otherwise; **none of them ever shrinks**, which
    /// is why one workspace serves a whole session and why its size is the
    /// union of what every routine asked for.
    ///
    /// Growing *discards* the contents, as the C's free-then-malloc does —
    /// which is why `cholmod_super_numeric` fills `SuperMap` only after calling
    /// it (`:266-287`) and says so at
    /// `t_cholmod_super_numeric_worker.c:204-208`.
    pub(super) fn allocate(&mut self, nrow: usize, iworksize: usize, xworksize: usize) {
        /* the C's `MAX (1, nrow)`, `MAX (1, iworksize)` and `MAX (2,
         * xworksize)` (`:54,80,95`) */
        let nrow = nrow.max(1);
        if nrow > self.flag.len() {
            self.flag = vec![EMPTY; nrow];
            self.head = vec![EMPTY; nrow + 1];
            self.mark = 0;
        }
        if iworksize.max(1) > self.iwork.len() {
            self.iwork = vec![0; iworksize.max(1)];
        }
        if xworksize.max(2) > self.xwork.len() {
            /* `alloc_work` zeroes Xwork once, at allocation (`:106-108`) */
            self.xwork = vec![0.0; xworksize.max(2)];
        }
    }

    /// `Iwork` split the way `cholmod_analyze_p2` splits it
    /// (`cholmod_analyze.c:509-526`): `Parent`, `First`, `Level` and `Post` are
    /// the **last `4n`**, and everything the analysis calls between them gets a
    /// [`WorkRef`] over the first `2n`.
    ///
    /// The split is what makes the four survive calls that are meanwhile
    /// scratching in `Iwork [0..2n)` — the C gets that for free from raw
    /// pointers into one block, and Rust needs to be shown the disjointness.
    ///
    /// **The ordering routines are the exception and must not take this.** AMD
    /// uses all `6n` (`amd.rs`'s `Degree`/`Wi`/`Len`/`Nv`/`Next`/`Elen`), which
    /// upstream permits precisely because none of the four is needed across
    /// such a call (`:511-514`). Re-split afterwards; every one of the four is
    /// written before it is read again.
    ///
    /// `uncol` is `A->ncol` for an `stype == 0` matrix and zero otherwise:
    /// upstream puts the four at `Iwork + 2n + uncol` (`:515`), because the
    /// unsymmetric arms of `cholmod_etree` and `cholmod_rowcolcounts` need that
    /// much scratch rather than `2n`.
    pub(super) fn split_analyze(&mut self, n: usize, uncol: usize) -> Analyze<'_> {
        let (scratch, tail) = self.iwork.split_at_mut(2 * n + uncol);
        let (parent, tail) = tail.split_at_mut(n);
        let (first, tail) = tail.split_at_mut(n);
        let (level, post) = tail.split_at_mut(n);
        Analyze {
            work: WorkRef {
                iwork: scratch,
                flag: &mut self.flag,
                head: &mut self.head,
                xwork: &mut self.xwork,
                mark: &mut self.mark,
            },
            parent,
            first,
            level,
            post: &mut post[..n],
        }
    }

    /// The between-users invariant every kernel here promises to restore:
    /// `Flag [i] < mark` for all `i`, `Head` all `EMPTY`, `Xwork` all zero
    /// (`cholmod_internal.h:225-238`). Upstream checks it with
    /// `cholmod_dump_work`, under `#ifndef NDEBUG` — same idea, same build.
    #[cfg(test)]
    pub(super) fn is_pristine(&self) -> bool {
        self.flag.iter().all(|&f| f < self.mark)
            && self.head.iter().all(|&h| h == EMPTY)
            && self.xwork.iter().all(|&x| x == 0.0)
    }
}

/// [`Work::split_analyze`]'s four `n`-vectors and the scratch that goes with
/// them — `Work4n` in `cholmod_analyze.c:516-521`.
pub(super) struct Analyze<'a> {
    pub(super) work: WorkRef<'a>,
    pub(super) parent: &'a mut [i64],
    pub(super) first: &'a mut [i64],
    pub(super) level: &'a mut [i64],
    pub(super) post: &'a mut [i64],
}

pub struct WorkRef<'a> {
    pub(super) iwork: &'a mut [i64],
    pub(super) flag: &'a mut [i64],
    pub(super) head: &'a mut [i64],
    pub(super) xwork: &'a mut [f64],
    pub(super) mark: &'a mut i64,
}

impl WorkRef<'_> {
    pub(super) fn scratch2(&mut self, n: usize) -> (&mut [i64], &mut [i64]) {
        self.iwork[..2 * n].split_at_mut(n)
    }

    /// `cholmod_clear_flag` (`t_cholmod_clear_flag.c:34-49`) — bump `mark`, and
    /// rewrite `Flag` only if that overflowed.
    #[inline]
    pub(super) fn clear_flag(&mut self) {
        clear_flag(Ws::new(self.flag), self.mark);
    }

    /// `Common->mark = EMPTY ; CLEAR_FLAG (Common)`, the two lines that
    /// unconditionally reset `Flag` to all-`EMPTY`
    /// (`cholmod_rowcolcounts.c:504-505`). Spelled out rather than folded into
    /// a `fill`, because the resulting `mark` is 0 and not `EMPTY`, and the
    /// next kernel to use `Flag` reads it.
    #[inline]
    pub(super) fn reset_flag(&mut self) {
        *self.mark = EMPTY;
        self.clear_flag();
    }
}

#[derive(Debug)]
pub enum CscError {
    IndptrLen { got: usize, want: usize },
    IndptrNotMonotone { at: usize, prev: i64, got: i64 },
    IndptrPastEnd { nz: i64, len: usize },
    RowOutOfRange { at: usize, got: i64, n: usize },
}

impl core::fmt::Display for CscError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            CscError::IndptrLen { got, want } => {
                write!(f, "indptr has length {got}, expected n+1 = {want}")
            }
            CscError::IndptrNotMonotone { at, prev, got } => write!(
                f,
                "indptr must be non-negative and non-decreasing: \
                 indptr[{at}] = {got} follows {prev}"
            ),
            CscError::IndptrPastEnd { nz, len } => {
                write!(f, "indptr[n] = {nz} is out of range for {len} row indices")
            }
            CscError::RowOutOfRange { at, got, n } => write!(
                f,
                "row index {got} at position {at} is out of range for n = {n}"
            ),
        }
    }
}

pub fn validate_csc(n: usize, indptr: &[i64], indices: &[i64]) -> Result<usize, CscError> {
    validate_csc_rect(n, n, indptr, indices)
}

pub fn validate_csc_rect(
    nrow: usize,
    ncol: usize,
    indptr: &[i64],
    indices: &[i64],
) -> Result<usize, CscError> {
    let n = ncol;
    if indptr.len() != n + 1 {
        return Err(CscError::IndptrLen {
            got: indptr.len(),
            want: n + 1,
        });
    }
    let mut prev = 0i64;
    for (j, &p) in indptr.iter().enumerate() {
        if p < prev {
            return Err(CscError::IndptrNotMonotone {
                at: j,
                prev,
                got: p,
            });
        }
        prev = p;
    }
    let nz = indptr[n];
    if nz as usize > indices.len() {
        return Err(CscError::IndptrPastEnd {
            nz,
            len: indices.len(),
        });
    }
    let nz = nz as usize;

    let (mut lo, mut hi) = (0i64, -1i64);
    for &i in &indices[..nz] {
        lo = lo.min(i);
        hi = hi.max(i);
    }
    if lo < 0 || hi >= nrow as i64 {
        let (at, got) = indices[..nz]
            .iter()
            .enumerate()
            .find(|(_, &i)| i < 0 || i >= nrow as i64)
            .map(|(p, &i)| (p, i))
            .expect("the min/max fold only fails when some index is out of range");
        return Err(CscError::RowOutOfRange { at, got, n: nrow });
    }
    Ok(nz)
}

/// `A->sorted` — whether the row indices ascend within every column.
///
/// CHOLMOD takes this as a claim from the caller rather than checking it, but
/// the caller here is numpy, which does not carry the flag through a raw array.
/// One O(nnz) pass to establish it is cheap next to what reads it, and getting
/// it wrong is silent: `cholmod_rowfac` stops scanning a column at the first
/// entry below the diagonal when it is set (`cholmod_rowfac.c:149-152`).
pub fn columns_are_sorted(n: usize, indptr: &[i64], indices: &[i64]) -> bool {
    (0..n).all(|j| {
        let (lo, hi) = (indptr[j] as usize, indptr[j + 1] as usize);
        indices[lo..hi].windows(2).all(|w| w[0] < w[1])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "out of range")]
    #[cfg_attr(not(debug_assertions), ignore = "the bound is compiled out")]
    fn ws_still_checks_its_bound_under_cfg_test() {
        let mut buf = [0i64; 4];
        let ws = Ws::new(&mut buf);
        std::hint::black_box(ws[10i64]);
    }

    #[test]
    fn validate_csc_rejects_malformed_patterns() {
        assert!(matches!(
            validate_csc(3, &[0, 1, 2], &[0, 1, 2]),
            Err(CscError::IndptrLen { got: 3, want: 4 })
        ));
        /* a non-monotone indptr is what would otherwise let the column loop
         * walk off the end of `indices` */
        assert!(matches!(
            validate_csc(3, &[0, 2, 1, 3], &[0, 1, 2]),
            Err(CscError::IndptrNotMonotone { at: 2, .. })
        ));
        assert!(matches!(
            validate_csc(3, &[-1, 0, 1, 2], &[0, 1, 2]),
            Err(CscError::IndptrNotMonotone { at: 0, .. })
        ));
        assert!(matches!(
            validate_csc(2, &[0, 1, 9], &[0, 1]),
            Err(CscError::IndptrPastEnd { nz: 9, len: 2 })
        ));
        assert!(matches!(
            validate_csc(3, &[0, 1, 2, 3], &[0, 7, 2]),
            Err(CscError::RowOutOfRange {
                at: 1,
                got: 7,
                n: 3
            })
        ));
        assert!(matches!(
            validate_csc(3, &[0, 1, 2, 3], &[0, -1, 2]),
            Err(CscError::RowOutOfRange {
                at: 1,
                got: -1,
                n: 3
            })
        ));
        /* an empty pattern at n = 0 is well-formed, not a row-out-of-range */
        assert!(matches!(validate_csc(0, &[0], &[]), Ok(0)));
        assert!(matches!(validate_csc(3, &[0, 1, 2, 3], &[0, 1, 2]), Ok(3)));
    }
}
