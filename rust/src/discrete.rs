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
//! unit-stride); `w` (n,). Returns the raw block (nd_i·p_im, nd_j·p_jm). For
//! `diag_term` the term is its own pair: only sub-blocks `c ≥ r` are formed and
//! `(c,r)` is the transpose.

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

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

    let msize = mim * mjm;
    let nst = si * sj; // summation-convention sets (same as bam.py len(Ki_list))
    // mgcv acc_w = (n > mjm*mim), strict (discrete.c:1801) OR the !acc_w large-p
    // `indReduce` branch (1884/1922): both collapse the (K_i,K_j) duplicates into
    // the dense W̄ then contract once — cheaper than the per-column factor
    // accumulation when the table fits. Two !acc_w guards so W̄ never costs more
    // than the factor path: `msize ≤ CAP` (memory) and `msize ≤ 16·nst·n` (the W̄
    // scan stays under the per-column factor work). MUST match the
    // `_XWX_DENSE_MSIZE_CAP`/`min(p)>15`/`16·nst·n` gate in bam.py so the numpy
    // spec and this kernel take the same branch (`rs == python`).
    let dense = n > msize
        || (pim.min(pjm) > 15 && msize <= XWX_DENSE_MSIZE_CAP && msize <= 16 * nst * n);
    let rfac = pjm <= pim; // form C (m_im×p_jm) else D (m_jm×p_im)
    let nrow = ndi * pim;
    let ncol = ndj * pjm;

    // One sub-block (r,c) → its p_im×p_jm cross-product, row-major. Read-only
    // over the shared flat inputs ⇒ the (r,c) map is embarrassingly parallel.
    let compute_sub = |r: usize, c: usize| -> Vec<f64> {
        let mut sub = vec![0.0f64; pim * pjm];
        if dense {
            let mut wbar = vec![0.0f64; msize];
            for s in 0..si {
                let ki_s = &ki_f[s * n..s * n + n];
                let tti_sr = &tti_f[(s * ndi + r) * n..(s * ndi + r) * n + n];
                for t in 0..sj {
                    let kj_t = &kj_f[t * n..t * n + n];
                    let ttj_tc = &ttj_f[(t * ndj + c) * n..(t * ndj + c) * n + n];
                    for row in 0..n {
                        let v = w_f[row] * tti_sr[row] * ttj_tc[row];
                        wbar[(ki_s[row] as usize) * mjm + kj_t[row] as usize] += v;
                    }
                }
            }
            // sub = Xim' W̄ Xjm  (via tmp = W̄ Xjm)
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
            for s in 0..si {
                let ki_s = &ki_f[s * n..s * n + n];
                let tti_sr = &tti_f[(s * ndi + r) * n..(s * ndi + r) * n + n];
                for t in 0..sj {
                    let kj_t = &kj_f[t * n..t * n + n];
                    let ttj_tc = &ttj_f[(t * ndj + c) * n..(t * ndj + c) * n + n];
                    for row in 0..n {
                        let v = w_f[row] * tti_sr[row] * ttj_tc[row];
                        let a = ki_s[row] as usize;
                        let b = kj_t[row] as usize;
                        for bj in 0..pjm {
                            cfac[a * pjm + bj] += v * xjm_f[b * pjm + bj];
                        }
                    }
                }
            }
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
            for s in 0..si {
                let ki_s = &ki_f[s * n..s * n + n];
                let tti_sr = &tti_f[(s * ndi + r) * n..(s * ndi + r) * n + n];
                for t in 0..sj {
                    let kj_t = &kj_f[t * n..t * n + n];
                    let ttj_tc = &ttj_f[(t * ndj + c) * n..(t * ndj + c) * n + n];
                    for row in 0..n {
                        let v = w_f[row] * tti_sr[row] * ttj_tc[row];
                        let a = ki_s[row] as usize;
                        let b = kj_t[row] as usize;
                        for ai in 0..pim {
                            dfac[b * pim + ai] += v * xim_f[a * pim + ai];
                        }
                    }
                }
            }
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xwx_smooth_block, m)?)?;
    Ok(())
}
