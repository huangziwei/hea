//! bam `discrete=TRUE` cross-product kernel — the smooth×smooth raw `X'WX`
//! block.
//!
//! Mirrors `hea/models/bam.py::_smooth_smooth_block` / `_wbar_contract`, itself
//! a port of mgcv `XWXijs` (src/discrete.c:1672). For a term pair, decompose
//! each (possibly tensor / matrix-argument) term into (non-final row-tensor) ⊗
//! (final marginal); for each sub-block `(r,c)` accumulate the final-marginal
//! weight table `W̄[a,b] = Σ_{s,t,rows} w·dXi_r·dXj_c·[K_i=a][K_j=b]` then
//! contract `Xd_im' W̄ Xd_jm`. The `n×p` design is never formed.
//!
//! Three branches, mirroring mgcv's path selection (and the numpy spec):
//!  * `dense` — accumulate the full `m_im×m_jm` W̄, then `Xim'(W̄ Xjm)`. Used
//!    for `acc_w` (`n > m_im·m_jm`, 1801) AND the !acc_w large-p `indReduce`
//!    branch (`min(p)>15`, 1884/1922) when W̄ fits `XWX_DENSE_MSIZE_CAP` —
//!    collapsing the `(K_i,K_j)` duplicates the way indReduce's hash does.
//!  * `rfac` / `!rfac` — form only the smaller factor `C = W̄ Xjm` /
//!    `D = W̄' Xim` by direct per-row accumulation (1924-2006), for the !acc_w
//!    small-p case or marginals past the cap (memory-safe fallback).
//!
//! The Python kernel is the spec and the test oracle (tests/test_rs_parity.py
//! pins `rs == python` to a tight tolerance — the two sum W̄ in different orders,
//! so not 0-ulp); this runs the `(r,c)×(s,t)×rows` accumulation in one tight
//! pass instead of numpy's per-(s,t) `bincount` loop, which is call-overhead
//! bound on the signal-regression cases (large summation count `s_i·s_j`).
//!
//! Layout contract (caller passes C-contiguous): `xim` (m_im, p_im) and `xjm`
//! (m_jm, p_jm) are the final-marginal bases; `ki` (s_i, n) / `kj` (s_j, n) the
//! final-marginal index columns; `tti` (s_i, nd_i, n) / `ttj` (s_j, nd_j, n) the
//! truncated row-tensors (`tti[s, r, :]` contiguous so the inner row loop is
//! unit-stride); `w` (n,) the diagonal weight; `woff` (n-1,) the AR1
//! tridiagonal off-diagonal (empty ⇒ plain `diag(w)`, no super/sub scatters).
//! Returns the raw block (nd_i·p_im, nd_j·p_jm). For `diag_term` the term is its
//! own pair: only sub-blocks `c ≥ r` are formed and `(c,r)` is the transpose.

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nmath::util::rfma;

/// Largest dense `W̄` (m_im·m_jm) materialised on the !acc_w `indReduce` branch
/// before falling back to the per-column factor path (~32 MB f64). MUST match
/// `_XWX_DENSE_MSIZE_CAP` in `hea/models/bam.py`.
const XWX_DENSE_MSIZE_CAP: usize = 4_000_000;

#[pyfunction]
#[pyo3(name = "xwx_smooth_block")]
fn xwx_smooth_block<'py>(
    py: Python<'py>,
    xim: PyReadonlyArray2<'py, f64>,
    xjm: PyReadonlyArray2<'py, f64>,
    ki: PyReadonlyArray2<'py, i64>,
    kj: PyReadonlyArray2<'py, i64>,
    tti: PyReadonlyArray3<'py, f64>,
    ttj: PyReadonlyArray3<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    woff: PyReadonlyArray1<'py, f64>,
    diag_term: bool,
) -> Bound<'py, PyArray2<f64>> {
    let mim = xim.shape()[0];
    let pim = xim.shape()[1];
    let mjm = xjm.shape()[0];
    let pjm = xjm.shape()[1];
    let si = ki.shape()[0];
    let n = ki.shape()[1];
    let sj = kj.shape()[0];
    let ndi = tti.shape()[1];
    let ndj = ttj.shape()[1];

    // Logical-order flat copies → unit-stride inner loops regardless of layout.
    let xim_f: Vec<f64> = xim.as_array().iter().copied().collect(); // (mim, pim)
    let xjm_f: Vec<f64> = xjm.as_array().iter().copied().collect(); // (mjm, pjm)
    let ki_f: Vec<i64> = ki.as_array().iter().copied().collect(); // (si, n)
    let kj_f: Vec<i64> = kj.as_array().iter().copied().collect(); // (sj, n)
    let tti_f: Vec<f64> = tti.as_array().iter().copied().collect(); // (si, ndi, n)
    let ttj_f: Vec<f64> = ttj.as_array().iter().copied().collect(); // (sj, ndj, n)
    let w_f: Vec<f64> = w.as_array().iter().copied().collect();
    // AR1 tridiagonal off-diagonal (length n-1); empty ⇒ plain diag(w) weight.
    let woff_f: Vec<f64> = woff.as_array().iter().copied().collect();
    let tri = !woff_f.is_empty();

    let msize = mim * mjm;
    // s_i·s_j summation sets, ×3 for the AR1 tri scatters (diag+super+sub) so this
    // matches `len(Ki_list)` in the numpy `_wbar_contract` spec.
    let nst = if tri { 3 * si * sj } else { si * sj };
    // mgcv acc_w = (n > mjm*mim), strict (discrete.c:1801) OR the !acc_w large-p
    // `indReduce` branch (1884/1922): both collapse the (K_i,K_j) duplicates into
    // the dense W̄ then contract once — cheaper than the per-column factor
    // accumulation when the table fits. Two !acc_w guards so W̄ never costs more
    // than the factor path: `msize ≤ CAP` (memory) and `msize ≤ 16·nst·n` (the W̄
    // scan stays under the per-column factor work). MUST match the
    // `_XWX_DENSE_MSIZE_CAP`/`min(p)>15`/`16·nst·n` gate in bam.py so the numpy
    // spec and this kernel take the same branch (`rs == python`).
    let dense =
        n > msize || (pim.min(pjm) > 15 && msize <= XWX_DENSE_MSIZE_CAP && msize <= 16 * nst * n);
    let rfac = pjm <= pim; // form C (m_im×p_jm) else D (m_jm×p_im)
    let nrow = ndi * pim;
    let ncol = ndj * pjm;

    // One sub-block (r,c) → its p_im×p_jm cross-product, row-major. Read-only
    // over the shared flat inputs ⇒ the (r,c) map is embarrassingly parallel.
    let compute_sub = |r: usize, c: usize| -> Vec<f64> {
        let mut sub = vec![0.0f64; pim * pjm];
        // Deposit every W̄ entry the (r,c) sub-block needs: the diagonal weight
        // `w` for every row, plus — when `tri` — the AR1 super/sub couplings
        // `w_off` (mgcv XWXijs tri branches, discrete.c:1843-1880; super then sub
        // then diag per row 0..n-2, then the final-row diag). `deposit(a,b,v)`
        // adds `v` to W̄[a,b] in whatever factored form the active branch holds.
        let accumulate = |deposit: &mut dyn FnMut(usize, usize, f64)| {
            for s in 0..si {
                let ki_s = &ki_f[s * n..s * n + n];
                let tti_sr = &tti_f[(s * ndi + r) * n..(s * ndi + r) * n + n];
                for t in 0..sj {
                    let kj_t = &kj_f[t * n..t * n + n];
                    let ttj_tc = &ttj_f[(t * ndj + c) * n..(t * ndj + c) * n + n];
                    if tri {
                        for row in 0..n - 1 {
                            let a = ki_s[row] as usize;
                            let a1 = ki_s[row + 1] as usize;
                            let b = kj_t[row] as usize;
                            let b1 = kj_t[row + 1] as usize;
                            // super: (K_i[l], K_j[l+1]) += w_off·dXi[l]·dXj[l+1]
                            deposit(a, b1, woff_f[row] * tti_sr[row] * ttj_tc[row + 1]);
                            // sub:   (K_i[l+1], K_j[l]) += w_off·dXi[l+1]·dXj[l]
                            deposit(a1, b, woff_f[row] * tti_sr[row + 1] * ttj_tc[row]);
                            // diag:  (K_i[l], K_j[l])   += w·dXi[l]·dXj[l]
                            deposit(a, b, w_f[row] * tti_sr[row] * ttj_tc[row]);
                        }
                        let row = n - 1;
                        deposit(
                            ki_s[row] as usize,
                            kj_t[row] as usize,
                            w_f[row] * tti_sr[row] * ttj_tc[row],
                        );
                    } else {
                        for row in 0..n {
                            deposit(
                                ki_s[row] as usize,
                                kj_t[row] as usize,
                                w_f[row] * tti_sr[row] * ttj_tc[row],
                            );
                        }
                    }
                }
            }
        };
        if dense {
            let mut wbar = vec![0.0f64; msize];
            accumulate(&mut |a, b, v| wbar[a * mjm + b] += v);
            // sub = Xim' W̄ Xjm  (via tmp = W̄ Xjm), iterating only nonzero W̄.
            let mut tmp = vec![0.0f64; mim * pjm];
            for a in 0..mim {
                for b in 0..mjm {
                    let wv = wbar[a * mjm + b];
                    if wv != 0.0 {
                        for bj in 0..pjm {
                            tmp[a * pjm + bj] += wv * xjm_f[b * pjm + bj];
                        }
                    }
                }
            }
            for a in 0..mim {
                for ai in 0..pim {
                    let xv = xim_f[a * pim + ai];
                    if xv != 0.0 {
                        for bj in 0..pjm {
                            sub[ai * pjm + bj] += xv * tmp[a * pjm + bj];
                        }
                    }
                }
            }
        } else if rfac {
            let mut cfac = vec![0.0f64; mim * pjm];
            accumulate(&mut |a, b, v| {
                for bj in 0..pjm {
                    cfac[a * pjm + bj] += v * xjm_f[b * pjm + bj];
                }
            });
            for a in 0..mim {
                for ai in 0..pim {
                    let xv = xim_f[a * pim + ai];
                    if xv != 0.0 {
                        for bj in 0..pjm {
                            sub[ai * pjm + bj] += xv * cfac[a * pjm + bj];
                        }
                    }
                }
            }
        } else {
            let mut dfac = vec![0.0f64; mjm * pim];
            accumulate(&mut |a, b, v| {
                for ai in 0..pim {
                    dfac[b * pim + ai] += v * xim_f[a * pim + ai];
                }
            });
            // sub = D' Xjm  (D is m_jm×p_im)
            for b in 0..mjm {
                for ai in 0..pim {
                    let dv = dfac[b * pim + ai];
                    if dv != 0.0 {
                        for bj in 0..pjm {
                            sub[ai * pjm + bj] += dv * xjm_f[b * pjm + bj];
                        }
                    }
                }
            }
        }
        sub
    };

    let block = py.allow_threads(|| {
        // Enumerate the upper (or full) sub-block tasks, compute in parallel,
        // then assemble — disjoint writes, so no races on the output.
        let mut tasks: Vec<(usize, usize)> = Vec::new();
        for r in 0..ndi {
            let c0 = if diag_term { r } else { 0 };
            for c in c0..ndj {
                tasks.push((r, c));
            }
        }
        let subs: Vec<Vec<f64>> = if tasks.len() > 1 {
            tasks.par_iter().map(|&(r, c)| compute_sub(r, c)).collect()
        } else {
            tasks.iter().map(|&(r, c)| compute_sub(r, c)).collect()
        };
        let mut block = vec![0.0f64; nrow * ncol];
        for (&(r, c), sub) in tasks.iter().zip(subs.iter()) {
            for ai in 0..pim {
                for bj in 0..pjm {
                    block[(r * pim + ai) * ncol + (c * pjm + bj)] = sub[ai * pjm + bj];
                }
            }
            if diag_term && c > r {
                // (c,r) sub-block is the transpose (same term ⇒ p_im==p_jm)
                for ai in 0..pim {
                    for bj in 0..pjm {
                        block[(c * pim + bj) * ncol + (r * pjm + ai)] = sub[ai * pjm + bj];
                    }
                }
            }
        }
        block
    });
    Array2::from_shape_vec((nrow, ncol), block)
        .unwrap()
        .into_pyarray(py)
}

/// `rwMatrix` (src/misc.c:710-748; R wrapper bam.r:18-29) — recombine the rows of
/// the (n, p) matrix `x` per `stop`/`row`/`w`. Forward (`trans=false`): output row
/// `i` is `Σ_{j ∈ seg(i)} w[j]·x[row[j], :]`, the segment `seg(i) = start..stop[i]+1`
/// with `start = stop[i-1]+1` (`stop[-1] = -1`). Transpose (`trans=true`): the
/// scatter adjoint `out[row[j], :] += w[j]·x[i, :]` over the same `(i, j)` pairs,
/// `out` zero at outset (misc.c:730).
///
/// Both fold left-to-right in mgcv's i-outer/j-inner (== global k-ascending) order,
/// fusing each `*X1p += weight * *Xp` (misc.c:742) to `fma` on arm64 via `rfma`. So
/// this is 0-ulp to live arm64 R (verified vs `mgcv:::rwMatrix`), where the numpy
/// `np.add.at`/`reduceat` fallback in `_rw_matrix` pre-rounds the `weight·x`
/// products and so diverges ≤1 ulp (sub-floor). The single O(K·p) native pass also
/// avoids the fallback's `np.add.at` scatter — the AR1 X'Wy hot path.
///
/// `stop` (length n) and `row` (length K = Σ segment lengths) arrive 0-based (the
/// Python wrapper subtracts mgcv's 1-based R indices). Indexing is logical, so the
/// column layout of `x` is irrelevant — each output column is an independent fold.
#[pyfunction]
#[pyo3(name = "rw_matrix")]
fn rw_matrix<'py>(
    py: Python<'py>,
    stop: PyReadonlyArray1<'py, i64>,
    row: PyReadonlyArray1<'py, i64>,
    w: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    trans: bool,
) -> Bound<'py, PyArray2<f64>> {
    let n = x.shape()[0];
    let p = x.shape()[1];
    let stop_f: Vec<i64> = stop.as_array().iter().copied().collect();
    let row_f: Vec<i64> = row.as_array().iter().copied().collect();
    let w_f: Vec<f64> = w.as_array().iter().copied().collect();
    let x_f: Vec<f64> = x.as_array().iter().copied().collect(); // (n, p) row-major
    let mut out = vec![0.0f64; n * p];
    // misc.c:731-745 verbatim: i outer over output rows, j inner over the segment's
    // input rows; `start` advances to `stop[i]+1` each row. x_f (src) and out (dst)
    // are separate buffers, so no aliasing across the borrow.
    let mut start: i64 = 0;
    for i in 0..n {
        let end = stop_f[i] + 1;
        let mut j = start;
        while j < end {
            let jj = j as usize;
            let weight = w_f[jj];
            let rj = row_f[jj] as usize;
            // forward: out[i] += w·x[row[j]];  trans: out[row[j]] += w·x[i]
            let (src_row, dst_row) = if trans { (i, rj) } else { (rj, i) };
            let src = &x_f[src_row * p..src_row * p + p];
            let dst = &mut out[dst_row * p..dst_row * p + p];
            for c in 0..p {
                dst[c] = rfma(weight, src[c], dst[c]); // misc.c:742 → fmadd on arm64
            }
            j += 1;
        }
        start = end;
    }
    Array2::from_shape_vec((n, p), out)
        .unwrap()
        .into_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xwx_smooth_block, m)?)?;
    m.add_function(wrap_pyfunction!(rw_matrix, m)?)?;
    Ok(())
}
