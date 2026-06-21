//! Agglomerative hierarchical clustering — the O(n²) NN-chain core of `hclust()`.
//!
//! A line-by-line mirror of `hea/R/clustering.py::_hclust_fortran`, which is a
//! 1:1 port of R's `src/library/stats/src/hclust.f` (`SUBROUTINE HCLUST`). The
//! merge sequence has data-dependent tie-breaking and the Lance-Williams update
//! mutates the dissimilarity vector in place, so this is INHERENTLY SERIAL —
//! never parallelize (it would reorder merges and break `$merge`/`$height`
//! parity), exactly like `rng/mt.rs` and `linalg/chol.rs`.
//!
//! Contract with the Python seam (`clustering.py`): returns the agglomeration
//! arrays `(ia, ib, crit)` as the `[1..=n]` slices of the 1-based Fortran arrays
//! (entries `1..n-1` are the merges, entry `n` is the unused trailing 0); the
//! seam prepends the unused leading `0` to recover the full 1-based arrays that
//! `_hcass2` consumes — so `[0, *rs] == _hclust_fortran(...)` bit-for-bit.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// Matches `inf = 1.0e300` in _hclust_fortran (NOT f64::INFINITY): the running
// "minimum dissimilarity" sentinel, mirrored so comparisons agree bit-for-bit.
const INF: f64 = 1.0e300;

/// `(ia, ib, crit)` for an `n`-point clustering with packed dissimilarities
/// `diss` (length `n*(n-1)/2`), method code `iopt` (1..8), and `members`.
#[pyfunction]
#[pyo3(name = "hclust", signature = (n, diss, iopt, members))]
pub fn hclust<'py>(
    py: Python<'py>,
    n: usize,
    diss: PyReadonlyArray1<'py, f64>,
    iopt: usize,
    members: PyReadonlyArray1<'py, f64>,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let diss = diss.as_slice().unwrap();
    let memb0 = members.as_slice().unwrap();

    let (ia_out, ib_out, crit_out) = py.allow_threads(|| {
        let length = n * (n - 1) / 2;

        // 1-based working arrays (index 0 unused), mirroring the Fortran decls.
        let mut d = vec![0.0f64; length + 1];
        for i in 1..=length {
            d[i] = diss[i - 1];
        }
        let mut ia = vec![0i64; n + 1];
        let mut ib = vec![0i64; n + 1];
        let mut crit = vec![0.0f64; n + 1];
        let mut membr = vec![0.0f64; n + 1];
        for i in 1..=n {
            membr[i] = memb0[i - 1];
        }
        let mut nn = vec![0usize; n + 1];
        let mut disnn = vec![0.0f64; n + 1];
        let mut flag = vec![false; n + 1];

        let ioffst = |i: usize, j: usize| -> usize { j + (i - 1) * n - i * (i + 1) / 2 };

        // persistent locals (carry across iterations, as in Fortran)
        let (mut im, mut jj, mut jm): (usize, usize, usize) = (0, 0, 0);

        for i in 1..=n {
            flag[i] = true;
        }
        let mut ncl = n;

        let isward = iopt == 1 || iopt == 8;
        if iopt == 8 {
            // Ward "D2": cluster on squared distances
            for i in 1..=length {
                d[i] = d[i] * d[i];
            }
        }

        // initial nearest-neighbour list (NN to the RIGHT of i)
        for i in 1..n {
            let mut dmin = INF;
            for j in (i + 1)..=n {
                let ind = ioffst(i, j);
                if dmin > d[ind] {
                    dmin = d[ind];
                    jm = j;
                }
            }
            nn[i] = jm;
            disnn[i] = dmin;
        }

        loop {
            // least dissimilarity among the current NNs
            let mut dmin = INF;
            for i in 1..n {
                if flag[i] && disnn[i] < dmin {
                    dmin = disnn[i];
                    im = i;
                    jm = nn[i];
                }
            }
            ncl -= 1;

            let i2 = im.min(jm);
            let j2 = im.max(jm);
            ia[n - ncl] = i2 as i64;
            ib[n - ncl] = j2 as i64;
            if iopt == 8 {
                dmin = dmin.sqrt();
            }
            crit[n - ncl] = dmin;
            flag[j2] = false;

            // update dissimilarities from the new cluster
            let mut dmin_nn = INF;
            for k in 1..=n {
                if flag[k] && k != i2 {
                    let ind1 = if i2 < k { ioffst(i2, k) } else { ioffst(k, i2) };
                    let ind2 = if j2 < k { ioffst(j2, k) } else { ioffst(k, j2) };
                    let d12 = d[ioffst(i2, j2)];

                    if isward {
                        d[ind1] = (membr[i2] + membr[k]) * d[ind1]
                            + (membr[j2] + membr[k]) * d[ind2]
                            - membr[k] * d12;
                        d[ind1] /= membr[i2] + membr[j2] + membr[k];
                    } else if iopt == 2 {
                        d[ind1] = d[ind1].min(d[ind2]);
                    } else if iopt == 3 {
                        d[ind1] = d[ind1].max(d[ind2]);
                    } else if iopt == 4 {
                        d[ind1] = (membr[i2] * d[ind1] + membr[j2] * d[ind2])
                            / (membr[i2] + membr[j2]);
                    } else if iopt == 5 {
                        d[ind1] = (d[ind1] + d[ind2]) / 2.0;
                    } else if iopt == 6 {
                        d[ind1] = ((d[ind1] + d[ind2]) - d12 / 2.0) / 2.0;
                    } else if iopt == 7 {
                        d[ind1] = (membr[i2] * d[ind1] + membr[j2] * d[ind2]
                            - membr[i2] * membr[j2] * d12 / (membr[i2] + membr[j2]))
                            / (membr[i2] + membr[j2]);
                    }

                    if i2 < k {
                        if d[ind1] < dmin_nn {
                            dmin_nn = d[ind1];
                            jj = k;
                        }
                    } else if d[ind1] < disnn[k] {
                        // i2 > k: keep correct NNs for non-monotone methods
                        disnn[k] = d[ind1];
                        nn[k] = i2;
                    }
                }
            }
            membr[i2] += membr[j2];
            disnn[i2] = dmin_nn;
            nn[i2] = jj;

            // rebuild the NN list where it pointed at the merged pair
            for i in 1..n {
                if flag[i] && (nn[i] == i2 || nn[i] == j2) {
                    let mut dmin_r = INF;
                    for j in (i + 1)..=n {
                        if flag[j] {
                            let ind = ioffst(i, j);
                            if d[ind] < dmin_r {
                                dmin_r = d[ind];
                                jj = j;
                            }
                        }
                    }
                    nn[i] = jj;
                    disnn[i] = dmin_r;
                }
            }

            if ncl > 1 {
                continue;
            }
            break;
        }

        (ia[1..=n].to_vec(), ib[1..=n].to_vec(), crit[1..=n].to_vec())
    });

    (
        ia_out.into_pyarray(py),
        ib_out.into_pyarray(py),
        crit_out.into_pyarray(py),
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hclust, m)?)?;
    Ok(())
}
