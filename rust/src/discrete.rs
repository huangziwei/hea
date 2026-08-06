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

/// Row-scan inputs shared by every `(r,c)` sub-block of one term pair.
struct Acc<'a> {
    si: usize,
    sj: usize,
    ndi: usize,
    ndj: usize,
    n: usize,
    ki: &'a [i64],
    kj: &'a [i64],
    tti: &'a [f64],
    ttj: &'a [f64],
    w: &'a [f64],
    woff: &'a [f64],
    tri: bool,
}

impl<'a> Acc<'a> {
    /// Term i's `s`th index row and its `r`th truncated row-tensor column
    /// (empty ⇒ mgcv `tensi == 0`).
    #[inline]
    fn rows_i(&self, s: usize, r: usize) -> (&'a [i64], &'a [f64]) {
        let n = self.n;
        let tt = if self.tti.is_empty() {
            &[][..]
        } else {
            &self.tti[(s * self.ndi + r) * n..(s * self.ndi + r) * n + n]
        };
        (&self.ki[s * n..s * n + n], tt)
    }

    /// Term j's `t`th index row and its `c`th truncated row-tensor column.
    #[inline]
    fn rows_j(&self, t: usize, c: usize) -> (&'a [i64], &'a [f64]) {
        let n = self.n;
        let tt = if self.ttj.is_empty() {
            &[][..]
        } else {
            &self.ttj[(t * self.ndj + c) * n..(t * self.ndj + c) * n + n]
        };
        (&self.kj[t * n..t * n + n], tt)
    }
}

/// Deposit every W̄ entry the `(r,c)` sub-block needs: the diagonal weight `w`
/// for every row, plus — when `tri` — the AR1 super/sub couplings `w_off` (mgcv
/// XWXijs tri branches, discrete.c:1843-1880; super then sub then diag per row
/// `0..n-2`, then the final-row diag). `deposit(a, b, v)` adds `v` to `W̄[a,b]`
/// in whatever factored form the caller's branch holds.
///
/// Generic (not `&mut dyn FnMut`) so the deposit body inlines into the row loop:
/// mgcv hand-writes this loop once per branch rather than calling through a
/// function pointer (discrete.c:1924-2006), and an indirect call per row costs
/// more than the deposit itself on the `p ≈ 10` blocks that dominate a
/// factor-`by` fit.
///
/// `tti`/`ttj` with an empty last axis mean mgcv's `tensi`/`tensj == 0`: that
/// term is a singleton, its truncated row-tensor is identically 1, and the
/// factor drops out of the product entirely. mgcv keeps four hand-written loops
/// for the four `(tensi, tensj)` combinations (:1927-1962); `scatter` is
/// monomorphised per combination, which is the same thing.
#[inline]
fn accumulate_wbar<F: FnMut(usize, usize, f64)>(a: &Acc<'_>, r: usize, c: usize, mut deposit: F) {
    let n = a.n;
    let tens_i = !a.tti.is_empty();
    let tens_j = !a.ttj.is_empty();
    for s in 0..a.si {
        let ki_s = &a.ki[s * n..s * n + n];
        let tti_sr: &[f64] = if tens_i {
            &a.tti[(s * a.ndi + r) * n..(s * a.ndi + r) * n + n]
        } else {
            &[]
        };
        for t in 0..a.sj {
            let kj_t = &a.kj[t * n..t * n + n];
            let ttj_tc: &[f64] = if tens_j {
                &a.ttj[(t * a.ndj + c) * n..(t * a.ndj + c) * n + n]
            } else {
                &[]
            };
            if a.tri {
                // AR1: keep the general form — `tri` is rare and the extra
                // per-row branch is dwarfed by the three scatters it guards.
                let gi = |row: usize| if tens_i { tti_sr[row] } else { 1.0 };
                let gj = |row: usize| if tens_j { ttj_tc[row] } else { 1.0 };
                for row in 0..n - 1 {
                    let i0 = ki_s[row] as usize;
                    let i1 = ki_s[row + 1] as usize;
                    let j0 = kj_t[row] as usize;
                    let j1 = kj_t[row + 1] as usize;
                    // super: (K_i[l], K_j[l+1]) += w_off·dXi[l]·dXj[l+1]
                    deposit(i0, j1, a.woff[row] * gi(row) * gj(row + 1));
                    // sub:   (K_i[l+1], K_j[l]) += w_off·dXi[l+1]·dXj[l]
                    deposit(i1, j0, a.woff[row] * gi(row + 1) * gj(row));
                    // diag:  (K_i[l], K_j[l])   += w·dXi[l]·dXj[l]
                    deposit(i0, j0, a.w[row] * gi(row) * gj(row));
                }
                let row = n - 1;
                deposit(
                    ki_s[row] as usize,
                    kj_t[row] as usize,
                    a.w[row] * gi(row) * gj(row),
                );
            } else {
                // The four mgcv branches, monomorphised. `w` alone for the
                // singleton×singleton case — no all-ones factor streamed.
                match (tens_i, tens_j) {
                    (true, true) => scatter(n, ki_s, kj_t, &mut deposit, |row| {
                        a.w[row] * tti_sr[row] * ttj_tc[row]
                    }),
                    (true, false) => {
                        scatter(n, ki_s, kj_t, &mut deposit, |row| a.w[row] * tti_sr[row])
                    }
                    (false, true) => {
                        scatter(n, ki_s, kj_t, &mut deposit, |row| a.w[row] * ttj_tc[row])
                    }
                    (false, false) => scatter(n, ki_s, kj_t, &mut deposit, |row| a.w[row]),
                }
            }
        }
    }
}

/// One `(K_i[row], K_j[row]) += weight(row)` pass — the row loop of mgcv's
/// direct accumulation. Generic in `weight` so each `(tensi, tensj)` case
/// compiles to its own straight-line loop.
#[inline(always)]
fn scatter<D: FnMut(usize, usize, f64), W: Fn(usize) -> f64>(
    n: usize,
    ki_s: &[i64],
    kj_t: &[i64],
    deposit: &mut D,
    weight: W,
) {
    for row in 0..n {
        deposit(ki_s[row] as usize, kj_t[row] as usize, weight(row));
    }
}

/// mgcv's direct factor accumulation, column-outer / row-inner
/// (discrete.c:1925-1963 for `C`, :1965-2005 for `D`) — the non-`tri` case.
///
/// `fac` is the `m_dst × p` factor in COLUMN-major order and `xsrc` the source
/// marginal likewise, so the inner loop is exactly mgcv's
/// `Cq[*Kik] += *p0 * Xj[*Kjk]`: one contiguous `m`-length column of each is
/// live per pass (8 KB apiece at `m = 1000`), against a row-major layout's
/// `p`-wide read-modify-write at a random `a` spanning the whole `m·p` array.
/// Ordering is untouched — for a fixed `(a, q)` the rows still accumulate in
/// `(s, t, row)`-ascending order — so this is bit-for-bit the row-major result.
///
/// `kd`/`ks` are the destination/source index rows, `tt_d`/`tt_s` the
/// corresponding truncated row-tensor columns (empty ⇒ mgcv `tens* == 0`).
/// # Safety
///
/// Every `kd[row] < m_dst` and `ks_[row] < m_src` for `row < n`, and `w`,
/// `tt_d`/`tt_s` (when non-empty) are all at least `n` long. `xwx_smooth_block`
/// establishes this once per call via [`indices_in_range`], before the loop —
/// the indices are data (bin numbers), so the check cannot be hoisted by the
/// compiler and would otherwise cost two compare-and-branches per row, per
/// pass. That is the whole gap against mgcv's C on this loop.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn direct_factor(
    fac: &mut [f64],
    xsrc_cm: &[f64],
    m_dst: usize,
    m_src: usize,
    p: usize,
    n: usize,
    kd: &[i64],
    ks_: &[i64],
    w: &[f64],
    tt_d: &[f64],
    tt_s: &[f64],
) {
    debug_assert_eq!(fac.len(), m_dst * p);
    debug_assert_eq!(xsrc_cm.len(), m_src * p);
    debug_assert!(w.len() >= n);
    debug_assert!(kd.len() >= n && ks_.len() >= n);
    let tens_d = !tt_d.is_empty();
    let tens_s = !tt_s.is_empty();
    debug_assert!(!tens_d || tt_d.len() >= n);
    debug_assert!(!tens_s || tt_s.len() >= n);
    for q in 0..p {
        let cq = &mut fac[q * m_dst..q * m_dst + m_dst];
        let xq = &xsrc_cm[q * m_src..q * m_src + m_src];
        // SAFETY: caller's contract — indices in range, buffers ≥ n.
        // The four mgcv `(tensi, tensj)` branches, monomorphised: a singleton's
        // truncated row-tensor is identically 1 and drops out entirely.
        unsafe {
            match (tens_d, tens_s) {
                (true, true) => {
                    for row in 0..n {
                        let v = *w.get_unchecked(row)
                            * *tt_d.get_unchecked(row)
                            * *tt_s.get_unchecked(row)
                            * *xq.get_unchecked(*ks_.get_unchecked(row) as usize);
                        *cq.get_unchecked_mut(*kd.get_unchecked(row) as usize) += v;
                    }
                }
                (true, false) => {
                    for row in 0..n {
                        let v = *w.get_unchecked(row)
                            * *tt_d.get_unchecked(row)
                            * *xq.get_unchecked(*ks_.get_unchecked(row) as usize);
                        *cq.get_unchecked_mut(*kd.get_unchecked(row) as usize) += v;
                    }
                }
                (false, true) => {
                    for row in 0..n {
                        let v = *w.get_unchecked(row)
                            * *tt_s.get_unchecked(row)
                            * *xq.get_unchecked(*ks_.get_unchecked(row) as usize);
                        *cq.get_unchecked_mut(*kd.get_unchecked(row) as usize) += v;
                    }
                }
                (false, false) => {
                    for row in 0..n {
                        let v = *w.get_unchecked(row)
                            * *xq.get_unchecked(*ks_.get_unchecked(row) as usize);
                        *cq.get_unchecked_mut(*kd.get_unchecked(row) as usize) += v;
                    }
                }
            }
        }
    }
}

/// True when every entry of `idx` is in `0..m` — the one-shot bounds proof for
/// [`direct_factor`]. Vacuously true for an empty row (a term with no summation
/// sets contributes no rows to scan).
fn indices_in_range(idx: &[i64], m: usize) -> bool {
    let hi = m as i64;
    idx.iter().all(|&v| v >= 0 && v < hi)
}

/// Borrow a C-contiguous numpy buffer, copying only if it is not (the caller
/// passes `np.ascontiguousarray`, so the copy is a correctness fallback, never
/// the hot path).
///
/// This is not a micro-optimisation: `ki`/`kj`/`w` are `n`-long, so eagerly
/// collecting them cost ~2.6 MB of copying per block at `n = 110k` — measured
/// at 0.53 ms of the 0.99 ms a block took, i.e. more than the arithmetic. mgcv
/// reads `k` and `w` straight out of the caller's arrays (`Ki = k + (ks[im]+s)*n`),
/// and so must this.
macro_rules! borrow_flat {
    ($arr:expr, $ty:ty) => {
        match $arr.as_slice() {
            Ok(s) => std::borrow::Cow::Borrowed(s),
            Err(_) => {
                std::borrow::Cow::Owned($arr.as_array().iter().copied().collect::<Vec<$ty>>())
            }
        }
    };
}

/// `wb[K[row]] += v[row]` (or `+= v[row]·u[row]`) in one fused pass — mgcv's
/// weighted bin accumulation, used by the `XWXijs` simple diagonal branch
/// (`wb[K[kk]] += w[kk]`, discrete.c:1786) and by `singleXty` (:327).
///
/// The numpy spelling is `np.bincount(K, weights=v, minlength=m)`, which costs
/// 0.118 ms at `n = 110k` against 0.04 ms here: bincount cannot fuse the `v·u`
/// product (forcing an `n`-long temporary) and pays a bounds check per row that
/// the one-shot [`indices_in_range`] proof removes.
#[pyfunction]
#[pyo3(name = "bin_accum")]
fn bin_accum<'py>(
    py: Python<'py>,
    k: PyReadonlyArray1<'py, i64>,
    v: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
    m: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let k_f = borrow_flat!(k, i64);
    let v_f = borrow_flat!(v, f64);
    let u_f = borrow_flat!(u, f64);
    let n = k_f.len();
    if v_f.len() < n || (!u_f.is_empty() && u_f.len() < n) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bin_accum: weight arrays shorter than the index array",
        ));
    }
    if !indices_in_range(&k_f, m) {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "bin_accum: index out of range for {m} bins"
        )));
    }
    let out = py.allow_threads(|| {
        let mut wb = vec![0.0f64; m];
        if m == 1 {
            // Single bin ⇒ every row lands on `wb[0]`. Written as a scatter
            // (the general branch below) the accumulator round-trips through
            // memory each row, so the loop runs at store-to-load-forward +
            // FP-add latency instead of FP-add latency alone — measured 2×
            // at n = 110k, and the intercept marginal hits it on every fit.
            // A register accumulator adds the SAME values in the SAME
            // row-ascending order, so the result is bit-identical.
            let mut s = 0.0f64;
            if u_f.is_empty() {
                for row in 0..n {
                    s += v_f[row];
                }
            } else {
                for row in 0..n {
                    s += v_f[row] * u_f[row];
                }
            }
            wb[0] = s;
            return wb;
        }
        // SAFETY: `indices_in_range` proved every k < m, and the length check
        // above covers v/u. Accumulation stays row-ascending, as in mgcv.
        unsafe {
            if u_f.is_empty() {
                for row in 0..n {
                    *wb.get_unchecked_mut(*k_f.get_unchecked(row) as usize) +=
                        *v_f.get_unchecked(row);
                }
            } else {
                for row in 0..n {
                    *wb.get_unchecked_mut(*k_f.get_unchecked(row) as usize) +=
                        *v_f.get_unchecked(row) * *u_f.get_unchecked(row);
                }
            }
        }
        wb
    });
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// Column-major (`m`, `p`) copy of a row-major buffer — the layout mgcv's
/// marginal bases already have (`Xj = X + off[jm] + q * mjm`).
fn to_col_major(src: &[f64], m: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * p];
    for a in 0..m {
        for q in 0..p {
            out[q * m + a] = src[a * p + q];
        }
    }
    out
}

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
    bounds_checked: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mim = xim.shape()[0];
    let pim = xim.shape()[1];
    let mjm = xjm.shape()[0];
    let pjm = xjm.shape()[1];
    let si = ki.shape()[0];
    let n = ki.shape()[1];
    let sj = kj.shape()[0];
    let ndi = tti.shape()[1];
    let ndj = ttj.shape()[1];

    // Borrowed in logical order → unit-stride inner loops, no per-call copy of
    // the n-long index/weight rows.
    let xim_f = borrow_flat!(xim, f64); // (mim, pim)
    let xjm_f = borrow_flat!(xjm, f64); // (mjm, pjm)
    let ki_f = borrow_flat!(ki, i64); // (si, n)
    let kj_f = borrow_flat!(kj, i64); // (sj, n)
    let tti_f = borrow_flat!(tti, f64); // (si, ndi, n)
    let ttj_f = borrow_flat!(ttj, f64); // (sj, ndj, n)
    let w_f = borrow_flat!(w, f64);
    // AR1 tridiagonal off-diagonal (length n-1); empty ⇒ plain diag(w) weight.
    let woff_f = borrow_flat!(woff, f64);
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
    // Column-major marginals for the direct-factor branches: mgcv's own layout
    // (`Xj = X + off[jm] + q*mjm`), so one contiguous m-length column is live
    // per accumulation pass. Skipped on the dense branch, which reads xjm/xim
    // row-major.
    let (xim_cm, xjm_cm) = if dense {
        (Vec::new(), Vec::new())
    } else {
        (
            to_col_major(&xim_f, mim, pim),
            to_col_major(&xjm_f, mjm, pjm),
        )
    };
    // Bounds proof for `direct_factor`'s unchecked gathers/scatters. The
    // indices are data (bin numbers), so the compiler cannot hoist a per-access
    // check out of the row loop; proving the whole row set once instead removes
    // two compare-and-branches from every row of every pass.
    //
    // `bounds_checked` says the caller already proved it. That matters because
    // the scan is a second read of `2n` i64 — 0.077 ms of a 0.647 ms block at
    // n = 110k — while the index rows are FIXED for a design's lifetime: they
    // depend only on `k` and the term's marginal, not on the weights. The
    // caller caches the proof per (term, marginal) and pays it once per fit
    // instead of once per block per PIRLS iteration. mgcv's C never checks.
    if !dense && !bounds_checked && !(indices_in_range(&ki_f, mim) && indices_in_range(&kj_f, mjm))
    {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "xwx_smooth_block: discrete index out of range for marginals \
             ({mim}, {mjm}) — k columns and Xd are inconsistent"
        )));
    }

    // One sub-block (r,c) → its p_im×p_jm cross-product, row-major. Read-only
    // over the shared flat inputs ⇒ the (r,c) map is embarrassingly parallel.
    let compute_sub = |r: usize, c: usize| -> Vec<f64> {
        let mut sub = vec![0.0f64; pim * pjm];
        // Shared row-scan parameters; `accumulate_wbar` is generic over the
        // deposit closure so each branch's body inlines into the row loop.
        let acc = Acc {
            si,
            sj,
            ndi,
            ndj,
            n,
            ki: &ki_f,
            kj: &kj_f,
            tti: &tti_f,
            ttj: &ttj_f,
            w: &w_f,
            woff: &woff_f,
            tri,
        };
        if dense {
            let mut wbar = vec![0.0f64; msize];
            accumulate_wbar(&acc, r, c, |a, b, v| wbar[a * mjm + b] += v);
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
            // C = W̄ Xjm, m_im×p_jm, COLUMN-major — mgcv discrete.c:1925-1963.
            let mut cfac = vec![0.0f64; mim * pjm];
            if tri {
                // AR1 keeps the generic three-scatter row pass; `tri` is rare
                // and the row-major-style deposit is dwarfed by the scatters.
                accumulate_wbar(&acc, r, c, |a, b, v| {
                    for bj in 0..pjm {
                        cfac[bj * mim + a] += v * xjm_cm[bj * mjm + b];
                    }
                });
            } else {
                for s in 0..si {
                    let (ki_s, tti_sr) = acc.rows_i(s, r);
                    for t in 0..sj {
                        let (kj_t, ttj_tc) = acc.rows_j(t, c);
                        // SAFETY: `unchecked_ok` verified ki < mim and
                        // kj < mjm above; w/tt rows are all length n.
                        unsafe {
                            direct_factor(
                                &mut cfac, &xjm_cm, mim, mjm, pjm, n, ki_s, kj_t, &w_f, tti_sr,
                                ttj_tc,
                            );
                        }
                    }
                }
            }
            // sub = Xim' C, both column-major ⇒ contiguous dot products, and
            // each sub[ai,bj] still folds over `a` ascending as before.
            for bj in 0..pjm {
                let cq = &cfac[bj * mim..bj * mim + mim];
                for ai in 0..pim {
                    let xi = &xim_cm[ai * mim..ai * mim + mim];
                    let mut x = 0.0;
                    for a in 0..mim {
                        x += xi[a] * cq[a];
                    }
                    sub[ai * pjm + bj] = x;
                }
            }
        } else {
            // Mirror image: D = W̄' Xim, m_jm×p_im, column-major (mgcv :1965).
            let mut dfac = vec![0.0f64; mjm * pim];
            if tri {
                accumulate_wbar(&acc, r, c, |a, b, v| {
                    for ai in 0..pim {
                        dfac[ai * mjm + b] += v * xim_cm[ai * mim + a];
                    }
                });
            } else {
                for s in 0..si {
                    let (ki_s, tti_sr) = acc.rows_i(s, r);
                    for t in 0..sj {
                        let (kj_t, ttj_tc) = acc.rows_j(t, c);
                        // destination is indexed by K_j, source read at K_i
                        // SAFETY: as above, with the roles swapped.
                        unsafe {
                            direct_factor(
                                &mut dfac, &xim_cm, mjm, mim, pim, n, kj_t, ki_s, &w_f, ttj_tc,
                                tti_sr,
                            );
                        }
                    }
                }
            }
            // sub = D' Xjm
            for ai in 0..pim {
                let dq = &dfac[ai * mjm..ai * mjm + mjm];
                for bj in 0..pjm {
                    let xj = &xjm_cm[bj * mjm..bj * mjm + mjm];
                    let mut x = 0.0;
                    for b in 0..mjm {
                        x += dq[b] * xj[b];
                    }
                    sub[ai * pjm + bj] = x;
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
    Ok(Array2::from_shape_vec((nrow, ncol), block)
        .unwrap()
        .into_pyarray(py))
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
    m.add_function(wrap_pyfunction!(bin_accum, m)?)?;
    m.add_function(wrap_pyfunction!(rw_matrix, m)?)?;
    Ok(())
}
