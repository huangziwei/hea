//! Selected inverse — the entries of `A⁻¹` that lie on the factor's pattern.
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `MATLAB_Tools/sparseinv/sparseinv.c`     → [`sweep`]
//!   * `MATLAB_Tools/sparseinv/sparseinv.m`     → [`selected_inverse`]'s setup
//!
//! Takahashi's equations compute `Z(i,j) = (A⁻¹)(i,j)` for every `(i,j)` in
//! `pattern(L + L')` without forming `A⁻¹`. Every diagonal entry is in that
//! pattern, so `diag(A⁻¹)` comes out in full — which is the standard error, the
//! hat diagonal, and the exact `tr(A⁻¹B)` for any `B` whose pattern is
//! contained in `L + L'`.
//!
//! **Scope.** Real, double, simplicial, and `LDL'` — see the factor-form note
//! below. The upstream is written for a general `(L+I) D (U+I)` from an LU, and
//! its symmetric case sets `U = L'`; only that case is built here, since every
//! hea factor is a Cholesky. The `.m` driver's LU branch, its `Z = (Z+Z')/2`
//! symmetrization and its permutation handling are the caller's, not this
//! module's.
//!
//! **Cost is a factorization's worth of work, run at a scalar rate.** The
//! sweep dot-products each column against itself, so the work is
//! `Σ_j |L_j|²` — the factorization's flop count, not `O(nnz(L))`, and on a
//! large system those differ by more than two orders of magnitude. It then
//! does that work in one thread with no blocking, where the supernodal
//! factorization has both, so the wall clock is another two orders above the
//! work ratio. [`sweep`] returns upstream's own flop count so a caller can
//! check what it paid.
//!
//! The lever, if this is ever made faster: [`sweep`]'s two arithmetic loops
//! visit exactly the same number of entries, and the back-substitution costs
//! most of the time anyway, because each `z[k]` it writes feeds the next
//! iteration's gather and the chain stalls on FMA latency. The updates are
//! mutually independent. Fusing the loops — see [`sweep`] — buys what can be
//! bought without reassociating a sum; past that, every option changes the
//! arithmetic.
//!
//! **Memory is roughly two `L`s.** `Z` holds `pattern(L + L')`, i.e.
//! `2·nnz(L) − n` entries with both an index and a value, where `L` holds
//! `nnz(L)` of each.
//!
//! ## What upstream wants, and how hea's factor maps onto it
//!
//! `sparseinv` takes `L` with a **unit diagonal** and `D` separately, i.e. the
//! `LDL'` form. CHOLMOD's simplicial `LDL'` stores `D(j,j)` in the diagonal
//! slot of each column and the unit-`L` values off it, so `d[j]` is
//! `x[p[j]]` and the rest of `x` passes through untouched. An `LL'` factor has
//! to be converted first; [`selected_inverse`] rejects it rather than
//! converting, because the conversion needs `Params` and belongs with the
//! caller that has one.
//!
//! The diagonal slot is never read as a value in either role. In `L`'s role
//! `Lmunch[k]` walks column `k` from the bottom and only consumes an entry
//! whose row is `j`, and every `k` the sweep reaches satisfies `k < j`, so the
//! diagonal's row index `k` never matches. In `U`'s role it is the first entry
//! of the column and [`sweep`] starts past it.
//!
//! `U` is `L'` **stored by row**, which for `U = L'` is the same three arrays
//! as `L` stored by column: row `k` of `U` is column `k` of `L`. Upstream's mex
//! driver makes this explicit by passing `U'` in compressed-column form. So
//! there is no transpose to build and no second set of arrays.
//!
//! Upstream indexes columns as `Lp[k] .. Lp[k+1]`, which assumes a packed
//! factor. hea's columns run `p[k] .. p[k] + nz[k]` and are not packed in
//! general, so every `Lp[k+1]` here is `p[k] + nz[k]`.
//!
//! **Both `x -= a*b` sites go through [`mulsub`]**, the crate's contraction
//! policy, for the reason that function documents: the reference is a
//! `clang -O2` build with no `-ffp-contract` flag, so it fuses wherever the ISA
//! has a baseline FMA, and a plain `-=` here is a different number in the last
//! bit on `aarch64`. Written without it, this sweep disagreed with upstream's
//! own C by 1-2 ULP on 35 of 42 corpus cases while matching its pattern
//! exactly.

use super::numeric::{mulsub, Factor};

/// Why a selected inverse could not be computed.
#[derive(Debug, PartialEq, Eq)]
pub enum SpinvError {
    /// The factor has no numeric values.
    NotNumeric,
    /// The factor is `LL'`; upstream's recursion wants `LDL'`.
    NotLdl,
    /// `A` was not positive definite, so `L` is only valid up to `minor`.
    NotPositiveDefinite(usize),
    /// A column of `Z` has no diagonal entry — `sparseinv.c:82`'s `return -1`.
    NoDiagonal(usize),
}

impl std::fmt::Display for SpinvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpinvError::NotNumeric => write!(f, "factor has no numeric values"),
            SpinvError::NotLdl => write!(f, "selected inverse wants an LDL' factor, not LL'"),
            SpinvError::NotPositiveDefinite(k) => write!(
                f,
                "matrix is not positive definite: the leading minor of order {} is not",
                k + 1
            ),
            SpinvError::NoDiagonal(j) => {
                write!(
                    f,
                    "column {j} of the selected inverse has no diagonal entry"
                )
            }
        }
    }
}

/// `Z`, the selected inverse, on `pattern(L + L')` in compressed-column form.
///
/// In the factor's own ordering: `L` factors `P A P'`, so `Z` is the selected
/// inverse of `P A P'` and the caller unpermutes.
#[derive(Debug, Clone)]
pub struct Selected {
    pub n: usize,
    /// Column pointers, size `n + 1`. Packed, so column `j` is `p[j]..p[j+1]`.
    pub p: Vec<i64>,
    /// Row indices, ascending within each column.
    pub i: Vec<i64>,
    /// Values.
    pub x: Vec<f64>,
    /// `Zdiagp`: where column `j`'s diagonal entry sits in [`Selected::i`].
    pub diagp: Vec<i64>,
}

impl Selected {
    /// `diag(Z)`, in the factor's ordering.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.n)
            .map(|j| self.x[self.diagp[j] as usize])
            .collect()
    }
}

/// Nanoseconds spent in each of [`sweep`]'s three phases.
///
/// All zero unless the crate is built with the `profiling` feature. The clock
/// is read once per phase per column rather than per entry, so a sweep that is
/// being measured does the same work at the same rate as one that is not.
#[derive(Debug, Default, Clone, Copy)]
pub struct Phases {
    /// Scattering column `j` of `Z` into the dense workspace.
    pub scatter: u64,
    /// The recurrence itself: the back-substitution for the strictly-upper
    /// entries of column `j`, fused with the left-looking updates it feeds.
    pub recurrence: u64,
    /// Gathering column `j` back out and clearing the workspace.
    pub gather: u64,
}

#[cfg(feature = "profiling")]
#[inline]
fn tick() -> Option<std::time::Instant> {
    Some(std::time::Instant::now())
}

#[cfg(not(feature = "profiling"))]
#[inline]
fn tick() -> Option<std::time::Instant> {
    None
}

#[inline]
fn tock(t: Option<std::time::Instant>, acc: &mut u64) {
    if let Some(t) = t {
        *acc += t.elapsed().as_nanos() as u64;
    }
}

/// Column `j` of the factor as a half-open range into `L->i` / `L->x`.
#[inline]
fn col(l: &Factor, j: usize) -> (usize, usize) {
    let b = l.p[j] as usize;
    (b, b + l.nz[j] as usize)
}

/// `pattern(L + L')`, with each column's diagonal located.
///
/// Column `j` is row `j` of `L` without its diagonal, then column `j` of `L`.
/// Row `j` of `L` holds column indices `i ≤ j` and column `j` of `L` holds row
/// indices `i ≥ j`, both ascending, so the concatenation is ascending and the
/// diagonal sits exactly at the join. That is upstream's requirement — `Zi`
/// sorted, diagonal present — met by construction rather than by a sort.
fn z_pattern(l: &Factor) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let n = l.n;

    /* Entries in row j of L, diagonal included: the length of column j of L'. */
    let mut rowcount = vec![0i64; n];
    for j in 0..n {
        let (b, e) = col(l, j);
        for p in b..e {
            rowcount[l.i[p] as usize] += 1;
        }
    }

    let mut zp = vec![0i64; n + 1];
    for j in 0..n {
        let strict_upper = (rowcount[j] - 1).max(0);
        zp[j + 1] = zp[j] + strict_upper + l.nz[j];
    }
    let mut zi = vec![0i64; zp[n] as usize];

    /* The strictly-upper half, filled by walking L's columns in order so each
     * row's entries land ascending. `next[j]` ends at the join, which is the
     * diagonal's position. */
    let mut next: Vec<i64> = zp[..n].to_vec();
    for j in 0..n {
        let (b, e) = col(l, j);
        for p in b..e {
            let r = l.i[p] as usize;
            if r > j {
                zi[next[r] as usize] = j as i64;
                next[r] += 1;
            }
        }
    }

    /* The diagonal and below: column j of L, verbatim. */
    let diagp = next.clone();
    for j in 0..n {
        let (b, e) = col(l, j);
        let mut q = next[j] as usize;
        for p in b..e {
            zi[q] = l.i[p];
            q += 1;
        }
    }

    (zp, zi, diagp)
}

/// `sparseinv` (`sparseinv.c:28-169`), specialized to `U = L'`.
///
/// Returns the flop count upstream returns. `z`, `zdiagp` and `lmunch` are the
/// three size-`n` workspaces of the same names; `z` is zero on entry and is
/// restored to zero before returning, so it may be reused across calls.
#[allow(clippy::too_many_arguments)]
fn sweep(
    n: usize,
    lp: &[i64],
    lnz: &[i64],
    li: &[i64],
    lx: &[f64],
    d: &[f64],
    zp: &[i64],
    zi: &[i64],
    zx: &mut [f64],
    z: &mut [f64],
    zdiagp: &mut [i64],
    lmunch: &mut [i64],
    ph: &mut Phases,
) -> Result<i64, SpinvError> {
    let mut flops: i64 = n as i64;

    /* clear the numerical values of Z */
    let znz = zp[n] as usize;
    for p in 0..znz {
        zx[p] = 0.0;
    }

    /* find the diagonal of Z and initialize it */
    for j in 0..n {
        let mut pdiag: i64 = -1;
        let mut p = zp[j] as usize;
        while p < zp[j + 1] as usize && pdiag == -1 {
            if zi[p] == j as i64 {
                pdiag = p as i64;
                zx[p] = 1.0 / d[j];
            }
            p += 1;
        }
        zdiagp[j] = pdiag;
        if pdiag == -1 {
            return Err(SpinvError::NoDiagonal(j));
        }
    }

    /* Lmunch [k] points to the last entry in column k of L */
    for k in 0..n {
        lmunch[k] = lp[k] + lnz[k] - 1;
    }

    for j in (0..n).rev() {
        /* scatter Z (:,j) into z workspace; only the lower triangular part is
         * needed, since the upper triangular part is all zero */
        let t = tick();
        for p in zdiagp[j] as usize..zp[j + 1] as usize {
            z[zi[p] as usize] = zx[p];
        }
        tock(t, &mut ph.scatter);

        /* Upstream runs two separate `p` loops over the strictly-upper pattern
         * of column j — first every `Z(k,j)`, then every left-looking update —
         * and they are fused here into one descending walk. The fusion is
         * exact rather than approximate: the update for `k` reads `z` only at
         * rows `≥ k` of column `k` of `L`, and the dot loop has written every
         * `z[k']` for `k' ≥ k` by the time it reaches `k`, so both loops see
         * the same values in the same order however they are interleaved. The
         * two also walk the same row list — `zi[zdiagp[k]..]` is column `k` of
         * `L` verbatim — so fusing loads `z[i]` once for both.
         *
         * The reason to do it: the dot is a back-substitution whose `z[k]`
         * feeds the next iteration's gather, so consecutive dots are one long
         * dependency chain and stall on FMA latency, while the updates are
         * mutually independent. Interleaving gives the chain something to
         * overlap with. Neither sum is reassociated, so the answer is
         * unchanged bit for bit.
         *
         * The two are one timed phase because of the fusion, not despite it:
         * splitting them would need a clock read per entry of `L` rather than
         * per column, which costs a fifth of the sweep and would be measuring
         * the measurement. */
        let t = tick();
        let mut p = zdiagp[j] - 1;
        while p >= zp[j] {
            let k = zi[p as usize] as usize;

            /* Z (k,j) = - U (k,k+1:n) * Z (k+1:n,j) */
            let mut zkj = 0.0;
            let (ub, ue) = (lp[k] as usize, (lp[k] + lnz[k]) as usize);
            flops += (ue - ub) as i64;
            /* Upstream tests `i > k` per entry to "skip the diagonal of U, if
             * present", because its `U` comes from an LU where it may not be.
             * Column k of an `LDL'` factor is always the D slot at row k
             * followed by rows > k ascending, so the test is false exactly
             * once and true for every other entry. Dropping the first entry
             * visits the same entries in the same order. */
            if ue > ub {
                debug_assert_eq!(
                    li[ub], k as i64,
                    "column {k} of L must open on its diagonal"
                );
                for (&v, &i) in lx[ub + 1..ue].iter().zip(&li[ub + 1..ue]) {
                    zkj = mulsub(zkj, v, z[i as usize]);
                }
            }
            z[k] = zkj;

            /* left-looking update to the lower triangular part of Z.
             * ljk = L (j,k) */
            if lmunch[k] >= lp[k] && li[lmunch[k] as usize] == j as i64 {
                let ljk = lx[lmunch[k] as usize];
                lmunch[k] -= 1;

                /* Z (k+1:n,k) = Z (k+1:n,k) - Z (k+1:n,j) * L (j,k) */
                flops += zp[k + 1] - zdiagp[k];
                let (zb, ze) = (zdiagp[k] as usize, zp[k + 1] as usize);
                for (v, &i) in zx[zb..ze].iter_mut().zip(&zi[zb..ze]) {
                    *v = mulsub(*v, z[i as usize], ljk);
                }
            }
            p -= 1;
        }
        tock(t, &mut ph.recurrence);

        /* gather Z (:,j) back from z workspace */
        let t = tick();
        for p in zp[j] as usize..zp[j + 1] as usize {
            let i = zi[p] as usize;
            zx[p] = z[i];
            z[i] = 0.0;
        }
        tock(t, &mut ph.gather);
    }

    Ok(flops)
}

/// The selected inverse of the matrix `l` factors, in `l`'s own ordering.
///
/// `l` must be a numeric simplicial `LDL'` factor. Returns `Z`, upstream's
/// flop count, and where the time went — the last being all zero unless the
/// crate is built with the `profiling` feature.
pub fn selected_inverse(l: &Factor) -> Result<(Selected, i64, Phases), SpinvError> {
    if !l.numeric {
        return Err(SpinvError::NotNumeric);
    }
    if l.is_ll {
        return Err(SpinvError::NotLdl);
    }
    if l.minor < l.n {
        return Err(SpinvError::NotPositiveDefinite(l.minor));
    }

    let n = l.n;
    let (zp, zi, _) = z_pattern(l);
    let mut zx = vec![0.0f64; zp[n] as usize];

    /* d [j] is D (j,j), which an LDL' factor keeps in the diagonal slot of
     * column j — the slot upstream's L never reads. */
    let d: Vec<f64> = (0..n).map(|j| l.x[l.p[j] as usize]).collect();

    let mut z = vec![0.0f64; n];
    let mut zdiagp = vec![0i64; n];
    let mut lmunch = vec![0i64; n];
    let mut ph = Phases::default();

    let flops = sweep(
        n,
        &l.p,
        &l.nz,
        &l.i,
        &l.x,
        &d,
        &zp,
        &zi,
        &mut zx,
        &mut z,
        &mut zdiagp,
        &mut lmunch,
        &mut ph,
    )?;

    Ok((
        Selected {
            n,
            p: zp,
            i: zi,
            x: zx,
            diagp: zdiagp,
        },
        flops,
        ph,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::symbolic::{analyze_sparse, Method, Ordering, Sparse};
    use super::super::testcorpus::{corpus, spd_triangle};
    use super::super::ws::{columns_are_sorted, Work};
    use super::super::{numeric, testcorpus};
    use super::*;

    /// An `LDL'` factor of a corpus matrix, and the triangle it came from.
    fn factor(n: usize, edges: &[(usize, usize)]) -> (Factor, Vec<i64>, Vec<i64>, Vec<f64>) {
        let (p, i, v) = spd_triangle(n, edges, false);
        let a = Sparse {
            nrow: n,
            n,
            p: p.clone().into(),
            i: i.clone().into(),
            x: v.clone().into(),
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(n, &p, &i),
        };
        let mut work = Work::new(n);
        let s = analyze_sparse(
            &a,
            Method::Pinned(Ordering::Amd),
            super::super::amd::IntWidth::I64,
            &mut work,
        )
        .unwrap();
        let mut l = Factor::from_symbolic(s);
        let params = numeric::Params {
            final_ll: false,
            ..numeric::Params::default()
        };
        let mut w = Work::new(n);
        numeric::factorize(&a, 0.0, &mut l, &params, &mut w).unwrap();
        (l, p, i, v)
    }

    /// `inv(P A P')` by Gaussian elimination, for the small cases.
    fn dense_inverse(n: usize, p: &[i64], i: &[i64], v: &[f64], perm: &[i64]) -> Vec<f64> {
        /* A as dense, symmetric, from the stored upper triangle */
        let mut a = vec![0.0f64; n * n];
        for j in 0..n {
            for q in p[j] as usize..p[j + 1] as usize {
                let r = i[q] as usize;
                a[r + j * n] = v[q];
                a[j + r * n] = v[q];
            }
        }
        /* permute to P A P' */
        let mut m = vec![0.0f64; n * n];
        for c in 0..n {
            for r in 0..n {
                m[r + c * n] = a[perm[r] as usize + perm[c] as usize * n];
            }
        }
        /* invert by Gauss-Jordan, partial pivoting */
        let mut inv = vec![0.0f64; n * n];
        for j in 0..n {
            inv[j + j * n] = 1.0;
        }
        for k in 0..n {
            let mut piv = k;
            for r in k + 1..n {
                if m[r + k * n].abs() > m[piv + k * n].abs() {
                    piv = r;
                }
            }
            for c in 0..n {
                m.swap(k + c * n, piv + c * n);
                inv.swap(k + c * n, piv + c * n);
            }
            let d = m[k + k * n];
            for c in 0..n {
                m[k + c * n] /= d;
                inv[k + c * n] /= d;
            }
            for r in 0..n {
                if r != k {
                    let f = m[r + k * n];
                    if f != 0.0 {
                        for c in 0..n {
                            m[r + c * n] -= f * m[k + c * n];
                            inv[r + c * n] -= f * inv[k + c * n];
                        }
                    }
                }
            }
        }
        inv
    }

    #[test]
    fn the_selected_entries_are_the_inverses_entries() {
        for (name, n, edges) in corpus() {
            if n == 0 || n > 120 {
                continue;
            }
            let (l, p, i, v) = factor(n, &edges);
            let (z, _, _) = selected_inverse(&l).unwrap();
            let inv = dense_inverse(n, &p, &i, &v, &l.perm);
            for j in 0..n {
                for q in z.p[j] as usize..z.p[j + 1] as usize {
                    let r = z.i[q] as usize;
                    let want = inv[r + j * n];
                    assert!(
                        (z.x[q] - want).abs() <= 1e-9 * want.abs().max(1e-6),
                        "{name}: Z[{r},{j}] = {} want {want}",
                        z.x[q]
                    );
                }
            }
        }
    }

    #[test]
    fn the_diagonal_is_always_reached() {
        for (name, n, edges) in corpus() {
            if n == 0 || n > 400 {
                continue;
            }
            let (l, _, _, _) = factor(n, &edges);
            let (z, _, _) = selected_inverse(&l).unwrap();
            assert_eq!(z.diagonal().len(), n, "{name}");
            for j in 0..n {
                assert_eq!(z.i[z.diagp[j] as usize], j as i64, "{name}: column {j}");
            }
        }
    }

    #[test]
    fn the_pattern_is_l_plus_l_transpose_and_sorted() {
        for (name, n, edges) in corpus() {
            if n == 0 || n > 400 {
                continue;
            }
            let (l, _, _, _) = factor(n, &edges);
            let (zp, zi, _) = z_pattern(&l);
            let nnz_l: i64 = l.nz.iter().sum();
            assert_eq!(zp[n], 2 * nnz_l - n as i64, "{name}");
            for j in 0..n {
                for q in zp[j] as usize + 1..zp[j + 1] as usize {
                    assert!(zi[q] > zi[q - 1], "{name}: column {j} not ascending");
                }
            }
            /* every entry of L is in Z, and so is its transpose */
            for j in 0..n {
                let (b, e) = col(&l, j);
                for q in b..e {
                    let r = l.i[q];
                    assert!(
                        zi[zp[j] as usize..zp[j + 1] as usize].contains(&r),
                        "{name}"
                    );
                    assert!(
                        zi[zp[r as usize] as usize..zp[r as usize + 1] as usize]
                            .contains(&(j as i64)),
                        "{name}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_flop_count_is_a_factorizations_not_a_pass() {
        /* `Σ_j |L_j|²` rather than `nnz(L)` — the thing the cost model warns
         * about. On a banded matrix the ratio is small but must exceed 1. */
        let (_, n, edges) = corpus()
            .into_iter()
            .find(|(name, n, _)| *name == "banded" && *n == 400)
            .unwrap();
        let (l, _, _, _) = factor(n, &edges);
        let (_, flops, _) = selected_inverse(&l).unwrap();
        let nnz_l: i64 = l.nz.iter().sum();
        assert!(flops > nnz_l, "flops {flops} vs nnz(L) {nnz_l}");
    }

    #[test]
    fn an_ll_factor_is_refused() {
        let (p, i, v) = spd_triangle(20, &[(0, 1), (1, 2), (2, 3)], false);
        let a = Sparse {
            nrow: 20,
            n: 20,
            p: p.clone().into(),
            i: i.clone().into(),
            x: v.clone().into(),
            numeric: true,
            stype: 1,
            sorted: columns_are_sorted(20, &p, &i),
        };
        let mut work = Work::new(20);
        let s = analyze_sparse(
            &a,
            Method::Pinned(Ordering::Amd),
            super::super::amd::IntWidth::I64,
            &mut work,
        )
        .unwrap();
        let mut l = Factor::from_symbolic(s);
        let params = numeric::Params {
            final_ll: true,
            ..numeric::Params::default()
        };
        let mut w = Work::new(20);
        numeric::factorize(&a, 0.0, &mut l, &params, &mut w).unwrap();
        assert_eq!(selected_inverse(&l).unwrap_err(), SpinvError::NotLdl);
    }

    #[test]
    fn every_subscript_stays_in_bounds_on_the_whole_corpus() {
        /* Debug build, so every index the sweep forms is checked by rustc. */
        for (name, n, edges) in corpus() {
            if n == 0 {
                continue;
            }
            let (l, _, _, _) = factor(n, &edges);
            let (z, flops, _) = selected_inverse(&l).unwrap();
            assert!(flops >= n as i64, "{name}");
            assert_eq!(z.x.len(), z.i.len(), "{name}");
        }
        let _ = testcorpus::Lcg(1);
    }
}
