//! `C = A*A'` — the pattern the `stype == 0` ordering is computed from.
//!
//! Mechanical port of SuiteSparse 7.6.0:
//!
//!   * `CHOLMOD/Utility/t_cholmod_aat.c`         → [`aat`]
//!   * `CHOLMOD/Utility/t_cholmod_aat_worker.c`  → [`aat`]'s second pass
//!
//! `cholmod_analyze` on an unsymmetric `A` factorizes `A A'` without ever
//! forming its values: the explicit product exists only so a fill-reducing
//! ordering has a pattern to work on. `cholmod_amd` is where that happens —
//! `C = CHOLMOD(aat) (A, fset, fsize, -2, Common)` at `cholmod_amd.c:122` —
//! and the numeric factorization then consumes `A` and `A'` directly.
//!
//! **Scope: pattern only, `fset == NULL`.** Upstream builds seven numeric
//! instantiations and a pattern one; only the pattern one is reachable from
//! here, because the only caller wants an ordering. The `A(:,f)` column-subset
//! variant has no consumer either — `cholmod_analyze` passes `fset` straight
//! through from its own argument, and nothing in this crate supplies one.
//!
//! `mode` therefore ranges over the non-positive half of upstream's:
//!
//! | mode | meaning |
//! |---|---|
//! | `0` | pattern, keeping the diagonal |
//! | `-1` | pattern, removing the diagonal |
//! | `-2` | as `-1`, plus 50% + `n` of elbow room for AMD |
//!
//! **`C` is not sorted**, which is upstream's choice too
//! (`allocate_sparse (..., /* C is not sorted: */ FALSE, ...)`): each column is
//! filled in the order the scatter meets its entries. `cholmod_amd` does not
//! care, and neither does anything else that reads a pattern.

use std::borrow::Cow;

use super::symbolic::{transpose_unsym, Sparse};

/// `cholmod_aat` for a pattern `C = A A'`.
///
/// `A` must have `stype == 0`. Returns an `n`-by-`n` pattern matrix, `n` being
/// `A->nrow`, with `stype == 0` — upstream sets `C->stype = 0` and leaves the
/// caller to read it as the symmetric thing it is.
///
/// The two passes are upstream's and are kept apart for its reason: the first
/// counts `nnz(C)` so `C` can be allocated once, and it marks with
/// `jmark = -j-2` — always negative, and different for every column — so the
/// same `W` serves as "seen in this column" without being cleared per column.
/// The second pass re-uses `W` as "position of row `i` within the current
/// column of `C`", which is why it is left holding negatives on entry.
pub fn aat(a: &Sparse, mode: i32) -> Sparse<'static> {
    debug_assert!(a.stype == 0, "cholmod_aat wants an unsymmetric A");
    let mode = mode.clamp(-2, 0);
    let ignore_diag = mode < 0;

    let n = a.nrow;

    /* F = A', pattern only: `mode` is non-positive here, so upstream's
     * `ptranspose (A, mode, NULL, fset, fsize)` is a pattern transpose. */
    let f = transpose_unsym(a, false, None);

    /* W [i] != jmark means row i has not yet been seen in C(:,j) */
    let mut w = vec![-1i64; n];

    /* cnz = nnz (C) */
    let mut cnz: usize = 0;
    for j in 0..n {
        let jmark = -(j as i64) - 2;
        for pf in f.p[j] as usize..f.p[j + 1] as usize {
            let t = f.i[pf] as usize;
            for p in a.p[t] as usize..a.p[t + 1] as usize {
                let i = a.i[p] as usize;
                if ignore_diag && i == j {
                    continue;
                }
                if w[i] != jmark {
                    w[i] = jmark;
                    cnz += 1;
                }
            }
        }
    }

    let nzmax = if mode == -2 { cnz + cnz / 2 + n } else { cnz };

    let mut cp = vec![0i64; n + 1];
    let mut ci = vec![0i64; nzmax];

    /* C = A*A'. W is all negative on entry, which is what makes
     * `pi < pc_start` the "not yet in this column" test. */
    let mut pc: usize = 0;
    for j in 0..n {
        let pc_start = pc as i64;
        cp[j] = pc as i64;
        for pf in f.p[j] as usize..f.p[j + 1] as usize {
            let t = f.i[pf] as usize;
            for p in a.p[t] as usize..a.p[t + 1] as usize {
                let i = a.i[p] as usize;
                if ignore_diag && i == j {
                    continue;
                }
                if w[i] < pc_start {
                    /* C(i,j) is a new entry; log its position */
                    ci[pc] = i as i64;
                    w[i] = pc as i64;
                    pc += 1;
                }
                /* else C(i,j) already exists at position W[i]; with no values
                 * to accumulate there is nothing to do. */
            }
        }
    }
    cp[n] = pc as i64;
    ci.truncate(pc);

    Sparse {
        nrow: n,
        n,
        p: Cow::Owned(cp),
        i: Cow::Owned(ci),
        x: Cow::Owned(Vec::new()),
        numeric: false,
        stype: 0,
        sorted: false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testcorpus::corpus;
    use super::*;

    /// A rectangular pattern with `nrow` rows and `ncol` columns, built from an
    /// edge list so the corpus can be reused for a shape it was not written
    /// for.
    fn rect(nrow: usize, ncol: usize, edges: &[(usize, usize)]) -> Sparse<'static> {
        let mut cols: Vec<Vec<i64>> = vec![Vec::new(); ncol];
        for &(i, j) in edges {
            let (i, j) = (i % nrow, j % ncol);
            if !cols[j].contains(&(i as i64)) {
                cols[j].push(i as i64);
            }
        }
        let mut p = vec![0i64; ncol + 1];
        let mut idx = Vec::new();
        for j in 0..ncol {
            cols[j].sort_unstable();
            idx.extend_from_slice(&cols[j]);
            p[j + 1] = idx.len() as i64;
        }
        Sparse {
            nrow,
            n: ncol,
            p: Cow::Owned(p),
            i: Cow::Owned(idx),
            x: Cow::Owned(Vec::new()),
            numeric: false,
            stype: 0,
            sorted: true,
        }
    }

    /// `A A'` as a dense boolean matrix, the definition rather than the port.
    fn dense_aat(a: &Sparse) -> Vec<Vec<bool>> {
        let n = a.nrow;
        let mut out = vec![vec![false; n]; n];
        for t in 0..a.n {
            let rows: Vec<usize> = (a.p[t] as usize..a.p[t + 1] as usize)
                .map(|p| a.i[p] as usize)
                .collect();
            for &i in &rows {
                for &j in &rows {
                    out[i][j] = true;
                }
            }
        }
        out
    }

    fn as_dense(c: &Sparse) -> Vec<Vec<bool>> {
        let n = c.nrow;
        let mut out = vec![vec![false; n]; n];
        for j in 0..c.n {
            for p in c.p[j] as usize..c.p[j + 1] as usize {
                out[c.i[p] as usize][j] = true;
            }
        }
        out
    }

    #[test]
    fn the_pattern_is_a_times_a_transpose() {
        for (name, n, edges) in corpus() {
            if n == 0 {
                continue;
            }
            for &(nr, nc) in &[(n, n), (n, n / 2 + 1), (n / 2 + 1, n)] {
                if nr == 0 || nc == 0 {
                    continue;
                }
                let a = rect(nr, nc, &edges);
                let c = aat(&a, 0);
                let want = dense_aat(&a);
                assert_eq!(as_dense(&c), want, "{name} {nr}x{nc}");
            }
        }
    }

    #[test]
    fn the_negative_modes_drop_the_diagonal() {
        for (name, n, edges) in corpus() {
            if n == 0 {
                continue;
            }
            let a = rect(n, n / 2 + 1, &edges);
            for mode in [-1, -2] {
                let c = aat(&a, mode);
                for j in 0..c.n {
                    for p in c.p[j] as usize..c.p[j + 1] as usize {
                        assert_ne!(c.i[p], j as i64, "{name} mode {mode}");
                    }
                }
                /* off the diagonal it agrees with mode 0 */
                let full = as_dense(&aat(&a, 0));
                let got = as_dense(&c);
                for i in 0..c.nrow {
                    for j in 0..c.n {
                        if i != j {
                            assert_eq!(got[i][j], full[i][j], "{name} mode {mode}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_column_lists_each_row_once() {
        for (_, n, edges) in corpus() {
            if n == 0 {
                continue;
            }
            let c = aat(&rect(n, n / 2 + 1, &edges), 0);
            for j in 0..c.n {
                let mut rows: Vec<i64> = c.i[c.p[j] as usize..c.p[j + 1] as usize].to_vec();
                let before = rows.len();
                rows.sort_unstable();
                rows.dedup();
                assert_eq!(rows.len(), before);
            }
        }
    }

    #[test]
    fn the_result_is_symmetric() {
        for (name, n, edges) in corpus() {
            if n == 0 {
                continue;
            }
            let c = aat(&rect(n, n / 2 + 1, &edges), 0);
            let d = as_dense(&c);
            for i in 0..c.nrow {
                for j in 0..c.n {
                    assert_eq!(d[i][j], d[j][i], "{name} at ({i},{j})");
                }
            }
        }
    }
}
