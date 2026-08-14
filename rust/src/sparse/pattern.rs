//! Laying one matrix's values out on a wider, fixed pattern.
//!
//! Unlike the rest of this directory this is not a port of anything: it backs
//! `hea.sparse.PatternPlan`, which is hea's own API. It is here because the
//! pattern it merges against is a CSC factor input and the callers are the
//! same ones.
//!
//! The operation is a **merge**, not a search. Both patterns are CSC with row
//! indices ascending within each column, so walking the two in step visits
//! every entry once. The Python it replaces reduced each pattern to one
//! `col * nrow + row` key per entry and ran `searchsorted`, which is correct
//! but costs a `nnz log nnz` and an 8-byte key array held for the plan's whole
//! lifetime.

/// Scatter `B`'s values onto the pattern `(ap, ai)`, column by column.
///
/// `out` is the caller's, of length `ai.len()`, and is written in full — every
/// slot the merge does not reach is zeroed, so it need not arrive clear.
///
/// Returns how many of `B`'s entries are **not** in the pattern. A nonzero
/// count means the containment the caller assumed does not hold and `out` is
/// not `B`'s values; the caller decides what to do about it, because the
/// useful error message names the plan.
///
/// A repeated row index within one column of `B` writes its slot twice and the
/// last value wins, which is what the `searchsorted` form did with the same
/// input. Canonical CSC has no repeats.
pub fn scatter(
    ncol: usize,
    ap: &[i64],
    ai: &[i64],
    bp: &[i64],
    bi: &[i64],
    bx: &[f64],
    out: &mut [f64],
) -> usize {
    out.fill(0.0);
    let mut missing = 0usize;
    for j in 0..ncol {
        let mut q = ap[j] as usize;
        let qend = ap[j + 1] as usize;
        for p in bp[j] as usize..bp[j + 1] as usize {
            let row = bi[p];
            while q < qend && ai[q] < row {
                q += 1;
            }
            if q < qend && ai[q] == row {
                out[q] = bx[p];
            } else {
                missing += 1;
            }
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(indptr, indices)` of a dense-ish pattern given as column row-lists.
    fn csc(cols: &[&[i64]]) -> (Vec<i64>, Vec<i64>) {
        let mut p = vec![0i64];
        let mut i = Vec::new();
        for c in cols {
            i.extend_from_slice(c);
            p.push(i.len() as i64);
        }
        (p, i)
    }

    #[test]
    fn a_contained_matrix_lands_on_its_own_slots() {
        let (ap, ai) = csc(&[&[0, 1, 3], &[2, 4]]);
        let (bp, bi) = csc(&[&[1, 3], &[4]]);
        let bx = vec![7.0, 8.0, 9.0];
        let mut out = vec![f64::NAN; ai.len()];
        assert_eq!(scatter(2, &ap, &ai, &bp, &bi, &bx, &mut out), 0);
        assert_eq!(out, vec![0.0, 7.0, 8.0, 0.0, 9.0]);
    }

    #[test]
    fn an_entry_outside_the_pattern_is_counted_not_written() {
        let (ap, ai) = csc(&[&[0, 3], &[2]]);
        /* row 1 in column 0 and row 5 in column 1 are both absent, and 5 is
         * past the column's last row, which is the case a clamped
         * `searchsorted` had to reject by comparison rather than by index. */
        let (bp, bi) = csc(&[&[0, 1], &[5]]);
        let bx = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; ai.len()];
        assert_eq!(scatter(2, &ap, &ai, &bp, &bi, &bx, &mut out), 2);
        assert_eq!(out, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn out_is_cleared_even_where_nothing_lands() {
        let (ap, ai) = csc(&[&[0, 1], &[0]]);
        let (bp, bi) = csc(&[&[], &[]]);
        let mut out = vec![5.0, 6.0, 7.0];
        assert_eq!(scatter(2, &ap, &ai, &bp, &bi, &[], &mut out), 0);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_columns_on_either_side_are_skipped() {
        let (ap, ai) = csc(&[&[], &[0, 2], &[]]);
        let (bp, bi) = csc(&[&[], &[2], &[]]);
        let bx = vec![4.0];
        let mut out = vec![0.0; ai.len()];
        assert_eq!(scatter(3, &ap, &ai, &bp, &bi, &bx, &mut out), 0);
        assert_eq!(out, vec![0.0, 4.0]);
    }

    #[test]
    fn the_whole_pattern_can_be_filled() {
        let (ap, ai) = csc(&[&[0, 1], &[0, 1]]);
        let bx = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        assert_eq!(scatter(2, &ap, &ai, &ap, &ai, &bx, &mut out), 0);
        assert_eq!(out, bx);
    }
}
