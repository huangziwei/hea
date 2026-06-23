//! `pls_fit1` — mgcv's penalized least-squares inner solve (gdi.c:2895), the
//! per-PIRLS-iteration kernel. A mechanical port of the algorithm hea's
//! `gam._pls_qr` runs, restructured so it never forms the orthogonal factor Q:
//!
//!   minimise ‖√W(z − Xβ)‖² + β'Sλβ ,   E'E = Sλ
//!
//! Since `[√|W|X; E] = QR`, every quantity the solve needs is available from `R`
//! alone — the rhs projection is `Q₁'(√|W|z) = R⁻ᵀX'Wz` (mgcv's own use_wy-stable
//! identity, gdi.c:3132) and, for the negative-Newton-weight correction, the
//! negative rows of the data-block orthogonal factor are `IQ = √|w[neg]|·X[neg]
//! ·R⁻¹`, so `IQ'IQ = R⁻ᵀ(X[neg]'diag|w_neg|X[neg])R⁻¹` — a p×p eigenproblem.
//!
//! `R` is obtained by a row-blocked **TSQR**: each row block of `√|W|X` is QR'd
//! in parallel (rayon) to a p×p factor `R_b` with `R_b'R_b = (block)'(block)`,
//! then `[R_0; …; R_{k-1}; E]` is QR'd once to the final `R` with
//! `R'R = X'|W|X + Sλ`. Block count is a pure function of `n` (not the thread
//! count) and every reduction is strictly in order, so `R` is bit-identical
//! across runs/threads. Accelerate's `dgeqrf` is single-threaded for tall-skinny
//! shapes; the block parallelism beats it AND removes the per-call Python glue
//! (≈15 numpy↔C transitions) that dominates the pure-Python path.
//!
//! Everything returned (β, the Cholesky factor of X'WX+Sλ, log|X'WX+Sλ|) is
//! invariant to the QR convention, so this matches mgcv's own `pls_fit1`
//! (gdi.c:2895, which factors via LAPACK `dgeqrf` over Accelerate BLAS) to the
//! LAPACK floor — β to ~1e-15, the factor/log|·| to ~1e-14 — and declines the
//! same indefinite X'WX+Sλ that mgcv signals with `n<0`. It is NOT 0-ulp vs R:
//! the row-blocked TSQR is not LAPACK's panel factorization and the BLAS leaf
//! reductions are SIMD, so 0-ulp is unreachable in scalar Rust (same floor as
//! the qr/discrete-contraction kernels). `tests/test_rs_parity.py` gates this
//! 3-way: `test_pls_fit1_parity` (rs == numpy `_pls_qr`) and
//! `test_pls_fit1_matches_r` (rs == live `.C(C_pls_fit1)` β/penalty + R
//! `chol`/`determinant` of the penalized Hessian).

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::qr::dnrm2;

/// In-order dot of two equal-length slices (deterministic, no SIMD reassoc).
fn ddot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 2-D ndarray element strides `(row, col)` as `usize`, or `None` if either is
/// negative (a reversed view — caller materializes C-order instead).
fn strides_usize(s: &[isize]) -> Option<(usize, usize)> {
    if s.len() == 2 && s[0] >= 0 && s[1] >= 0 {
        Some((s[0] as usize, s[1] as usize))
    } else {
        None
    }
}

/// Non-pivoted Householder QR of the column-major `m×p` matrix `a`, in place.
/// On return the upper triangle of `a` (column-major) holds `R`; the reflectors
/// below the diagonal are left in place but unused. Mirrors the per-column
/// Householder of R's `dqrdc2` (without the pivoting) so the BLAS-1 ops
/// accumulate in the same order.
fn house_qr(a: &mut [f64], m: usize, p: usize) {
    let lup = m.min(p);
    for l in 0..lup {
        let col = l * m;
        let mut nrmxl = dnrm2(&a[col + l..col + m]);
        if nrmxl == 0.0 {
            continue;
        }
        if a[col + l] != 0.0 {
            nrmxl = nrmxl.copysign(a[col + l]);
        }
        let inv = 1.0 / nrmxl;
        for xi in a[col + l..col + m].iter_mut() {
            *xi *= inv;
        }
        a[col + l] += 1.0;
        let piv = a[col + l];
        // Update the trailing columns l+1..p against the pivot column.
        let (left, right) = a.split_at_mut((l + 1) * m);
        let cl = &left[col + l..col + m];
        for cj in right.chunks_mut(m) {
            let cj = &mut cj[l..m];
            let dot = ddot(cl, cj);
            let t = -dot / piv;
            for (b, &av) in cj.iter_mut().zip(cl.iter()) {
                *b += t * av;
            }
        }
        a[col + l] = -nrmxl; // R diagonal
    }
}

/// Extract the `p×p` upper-triangular `R` (row-major) from a column-major matrix
/// `a` already factored in place by [`house_qr`]. Rows ≥ `valid` are zero (a
/// block with fewer rows than columns is rank-deficient — its trailing rows of
/// `R` are zero, which contributes nothing when stacked).
fn extract_r(a: &[f64], m: usize, p: usize, valid: usize) -> Vec<f64> {
    let mut r = vec![0.0_f64; p * p];
    for i in 0..valid.min(p) {
        for j in i..p {
            r[i * p + j] = a[j * m + i];
        }
    }
    r
}

/// Block count for the data TSQR — a pure function of `n` (so `R` is
/// thread-count-independent). Serial below a threshold where rayon dispatch
/// would cost more than it saves.
fn n_blocks(n: usize) -> usize {
    if n < 1024 {
        1
    } else {
        (n / 256).min(16)
    }
}

/// `R` of `[√|W|X; E]` via row-blocked TSQR, returned as a `p×p` row-major
/// upper-triangular matrix (`R'R = X'|W|X + E'E`). `x`/`e` are accessed by
/// element strides (`X[i,j] = x[i·xrs + j·xcs]`), so a C- or F-contiguous design
/// is read in place with no transpose copy.
#[allow(clippy::too_many_arguments)]
fn tsqr_r(
    x: &[f64], xrs: usize, xcs: usize, sqw: &[f64], n: usize, p: usize,
    e: &[f64], ne: usize, ers: usize, ecs: usize,
) -> Vec<f64> {
    let k = n_blocks(n);
    if k == 1 {
        // Small n: a single Householder QR of the full [√|W|X; E], matching
        // LAPACK dgeqrf's structure exactly (one QR, not the TSQR two-stage), so
        // the factor is as accurate as the numpy path on ill-conditioned designs
        // — the two-stage's extra rounding of R_b would cost ~1 digit at κ≈1e10.
        let rows = n + ne;
        let mut a = vec![0.0_f64; rows * p];
        for j in 0..p {
            for i in 0..n {
                a[j * rows + i] = sqw[i] * x[i * xrs + j * xcs];
            }
            for i in 0..ne {
                a[j * rows + n + i] = e[i * ers + j * ecs];
            }
        }
        house_qr(&mut a, rows, p);
        return extract_r(&a, rows, p, rows);
    }
    let bs = n.div_ceil(k);
    // Per-block: gather+scale the block into a column-major buffer and QR it to
    // a p×p R_b, written into a flat (k·p×p) buffer (no per-block heap Vec /
    // collect). Blocks are independent → rayon; each block's reduction is
    // in-order so R_b is deterministic across thread counts.
    let mut r_blocks = vec![0.0_f64; k * p * p];
    r_blocks
        .par_chunks_mut(p * p)
        .enumerate()
        .for_each(|(b, rb)| {
            let r0 = b * bs;
            let r1 = ((b + 1) * bs).min(n);
            if r0 >= r1 {
                return;
            }
            let br = r1 - r0;
            let mut blk = vec![0.0_f64; br * p];
            // j-outer / i-inner: for the F-contiguous design (xcs=n, xrs=1) both
            // the read x[i+j·n] and the write blk[j·br+ii] are contiguous in i —
            // the i-outer order strides reads by n (cache-hostile) and was the
            // reason the TSQR did not beat dgeqrf.
            for j in 0..p {
                let xoff = j * xcs;
                let boff = j * br;
                for (ii, i) in (r0..r1).enumerate() {
                    blk[boff + ii] = sqw[i] * x[i * xrs + xoff];
                }
            }
            house_qr(&mut blk, br, p);
            let valid = br.min(p);
            for i in 0..valid {
                for j in i..p {
                    rb[i * p + j] = blk[j * br + i];
                }
            }
        });
    // Stack [R_0; …; R_{k-1}; E] column-major, then one final QR → R.
    let rows = k * p + ne;
    let mut stack = vec![0.0_f64; rows * p];
    for b in 0..k {
        for i in 0..p {
            for j in 0..p {
                stack[j * rows + b * p + i] = r_blocks[b * p * p + i * p + j];
            }
        }
    }
    for i in 0..ne {
        for j in 0..p {
            stack[j * rows + k * p + i] = e[i * ers + j * ecs];
        }
    }
    house_qr(&mut stack, rows, p);
    extract_r(&stack, rows, p, rows)
}

/// Solve `R x = b` (back-substitution); `R` is `p×p` row-major upper. In place
/// on `b` (length p). Returns false if a zero pivot is hit.
fn solve_r(r: &[f64], p: usize, b: &mut [f64]) -> bool {
    for i in (0..p).rev() {
        let mut s = b[i];
        for j in i + 1..p {
            s -= r[i * p + j] * b[j];
        }
        let d = r[i * p + i];
        if d == 0.0 {
            return false;
        }
        b[i] = s / d;
    }
    true
}

/// Solve `Rᵀ x = b` (forward-substitution); `R` is `p×p` row-major upper. In
/// place on `b` (length p). Returns false if a zero pivot is hit.
fn solve_rt(r: &[f64], p: usize, b: &mut [f64]) -> bool {
    for i in 0..p {
        let mut s = b[i];
        for j in 0..i {
            s -= r[j * p + i] * b[j];
        }
        let d = r[i * p + i];
        if d == 0.0 {
            return false;
        }
        b[i] = s / d;
    }
    true
}

/// Symmetric eigendecomposition of a `p×p` matrix (row-major, symmetric) by
/// cyclic Jacobi. Returns `(evals, V)` with `V` row-major and column `j` the
/// eigenvector for `evals[j]` (i.e. `A = V diag(evals) Vᵀ`). Order/sign are
/// unspecified — the caller's correction `V(I−2D²)⁻¹Vᵀ` is invariant to both.
/// Accurate to the floor for the small (p≈10–40) gram matrices here.
fn jacobi_eigh(a_in: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0_f64; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    if p == 1 {
        return (vec![a[0]], v);
    }
    let off = |m: &[f64]| -> f64 {
        let mut s = 0.0;
        for i in 0..p {
            for j in i + 1..p {
                s += m[i * p + j] * m[i * p + j];
            }
        }
        s
    };
    let mut norm = 0.0;
    for x in a.iter() {
        norm += x * x;
    }
    let thresh = 1e-30 * norm;
    for _sweep in 0..60 {
        if off(&a) <= thresh {
            break;
        }
        for q in 0..p {
            for r in q + 1..p {
                let apq = a[q * p + r];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[r * p + r] - a[q * p + q]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols q,r of A.
                for k in 0..p {
                    let akq = a[k * p + q];
                    let akr = a[k * p + r];
                    a[k * p + q] = c * akq - s * akr;
                    a[k * p + r] = s * akq + c * akr;
                }
                for k in 0..p {
                    let aqk = a[q * p + k];
                    let ark = a[r * p + k];
                    a[q * p + k] = c * aqk - s * ark;
                    a[r * p + k] = s * aqk + c * ark;
                }
                // Accumulate the rotation into V.
                for k in 0..p {
                    let vkq = v[k * p + q];
                    let vkr = v[k * p + r];
                    v[k * p + q] = c * vkq - s * vkr;
                    v[k * p + r] = s * vkq + c * vkr;
                }
            }
        }
    }
    let evals: Vec<f64> = (0..p).map(|i| a[i * p + i]).collect();
    (evals, v)
}

/// Sign-normalize a `p×p` row-major upper-triangular factor to the unique
/// Cholesky factor (positive diagonal): flip the sign of each row whose
/// diagonal is negative.
fn sign_normalize(r: &mut [f64], p: usize) {
    for i in 0..p {
        if r[i * p + i] < 0.0 {
            for j in i..p {
                r[i * p + j] = -r[i * p + j];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pls_core(
    x: &[f64],
    xrs: usize,
    xcs: usize,
    w: &[f64],
    n: usize,
    p: usize,
    e: &[f64],
    ne: usize,
    ers: usize,
    ecs: usize,
    z: &[f64],
    xtwz: &[f64],
    use_xtwz: bool,
) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let sqw: Vec<f64> = w.iter().map(|&wi| wi.abs().sqrt()).collect();
    let mut r = tsqr_r(x, xrs, xcs, &sqw, n, p, e, ne, ers, ecs);
    if !r.iter().all(|x| x.is_finite()) {
        return None;
    }
    for i in 0..p {
        if r[i * p + i] == 0.0 {
            return None;
        }
    }
    let log_abs_diag: f64 = (0..p).map(|i| r[i * p + i].abs().ln()).sum();

    // rhs X'Wz: either supplied (use_wy mode) or X'(w⊙z) (w signed).
    let mut xwz = vec![0.0_f64; p];
    if use_xtwz {
        xwz.copy_from_slice(&xtwz[..p]);
    } else {
        // X'(w⊙z): j-outer scalar accumulate so the F-contiguous column
        // x[·+j·n] is streamed contiguously (cache-friendly), unlike i-outer.
        let wz: Vec<f64> = (0..n).map(|i| w[i] * z[i]).collect();
        for j in 0..p {
            let xoff = j * xcs;
            let mut s = 0.0;
            for i in 0..n {
                s += wz[i] * x[i * xrs + xoff];
            }
            xwz[j] = s;
        }
    }

    let any_neg = w.iter().any(|&wi| wi < 0.0);
    if !any_neg {
        // β = R⁻¹ R⁻ᵀ X'Wz ; log|X'WX+Sλ| = 2 Σ log|R_ii|.
        let mut beta = xwz;
        if !solve_rt(&r, p, &mut beta) || !solve_r(&r, p, &mut beta) {
            return None;
        }
        if !beta.iter().all(|x| x.is_finite()) {
            return None;
        }
        sign_normalize(&mut r, p);
        return Some((beta, r, 2.0 * log_abs_diag));
    }

    // --- negative Newton weights: eigen determinant correction ----------
    // G = X[neg]'diag|w_neg|X[neg] (in-order), Z = R⁻ᵀ G R⁻¹, eigh(Z).
    let mut g = vec![0.0_f64; p * p];
    for i in 0..n {
        if w[i] < 0.0 {
            let aw = -w[i];
            let base = i * xrs;
            for a in 0..p {
                let v = aw * x[base + a * xcs];
                for b in a..p {
                    g[a * p + b] += v * x[base + b * xcs];
                }
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            g[a * p + b] = g[b * p + a];
        }
    }
    // Y = R⁻ᵀ G (columnwise forward solve).
    let mut y = g;
    for j in 0..p {
        let mut col: Vec<f64> = (0..p).map(|i| y[i * p + j]).collect();
        if !solve_rt(&r, p, &mut col) {
            return None;
        }
        for i in 0..p {
            y[i * p + j] = col[i];
        }
    }
    // Z = R⁻ᵀ Yᵀ, then symmetrize → IQ'IQ.
    let mut zt = vec![0.0_f64; p * p];
    for j in 0..p {
        let mut col: Vec<f64> = (0..p).map(|i| y[j * p + i]).collect(); // row j of Y = col j of Yᵀ
        if !solve_rt(&r, p, &mut col) {
            return None;
        }
        for i in 0..p {
            zt[i * p + j] = col[i];
        }
    }
    let mut zsym = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            zsym[i * p + j] = 0.5 * (zt[j * p + i] + zt[i * p + j]);
        }
    }
    let (evals, vmat) = jacobi_eigh(&zsym, p);
    let mut d2 = vec![0.0_f64; p];
    for i in 0..p {
        d2[i] = 1.0 - 2.0 * evals[i];
        if d2[i] <= 0.0 {
            return None;
        }
    }
    // c = Vᵀ (R⁻ᵀ X'Wz).
    let mut t1 = xwz;
    if !solve_rt(&r, p, &mut t1) {
        return None;
    }
    let mut c = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..p {
            s += vmat[i * p + j] * t1[i]; // (Vᵀ)_{j,i} = V_{i,j}
        }
        c[j] = s;
    }
    // β = R⁻¹ ( V (c / d2) ).
    for j in 0..p {
        c[j] /= d2[j];
    }
    let mut beta = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = 0.0;
        for j in 0..p {
            s += vmat[i * p + j] * c[j];
        }
        beta[i] = s;
    }
    if !solve_r(&r, p, &mut beta) || !beta.iter().all(|x| x.is_finite()) {
        return None;
    }
    // M = diag(√d2) (Vᵀ R) ; R_corr = QR(M).R  (unique Cholesky factor).
    let sqrt_d2: Vec<f64> = d2.iter().map(|&v| v.sqrt()).collect();
    // VtR[i,j] = Σ_k V_{k,i} R_{k,j}.
    let mut m_cm = vec![0.0_f64; p * p]; // column-major for house_qr
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..p {
                s += vmat[k * p + i] * r[k * p + j];
            }
            m_cm[j * p + i] = sqrt_d2[i] * s;
        }
    }
    house_qr(&mut m_cm, p, p);
    let mut r_corr = extract_r(&m_cm, p, p, p);
    if !r_corr.iter().all(|x| x.is_finite()) {
        return None;
    }
    for i in 0..p {
        if r_corr[i * p + i] == 0.0 {
            return None;
        }
    }
    let log_det = 2.0 * log_abs_diag + d2.iter().map(|&v| v.ln()).sum::<f64>();
    sign_normalize(&mut r_corr, p);
    Some((beta, r_corr, log_det))
}

/// PyO3 entry: returns `(ok, beta, R, log_det)`. On `ok=False` the penalized
/// Hessian was indefinite/singular (caller retries with Fisher weights, as
/// gam.fit3.r:341 does); `beta`/`R` are then empty.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, w, e, z, xtwz, use_xtwz))]
pub fn pls_fit1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    xtwz: PyReadonlyArray1<'py, f64>,
    use_xtwz: bool,
) -> PyResult<(bool, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, f64)> {
    let xv = x.as_array();
    let (n, p) = (xv.nrows(), xv.ncols());
    let ev = e.as_array();
    let ne = ev.nrows();
    // Read the 2-D design/penalty in place by element strides — the hot-path
    // arrays are F-contiguous (column-major, hea builds them that way) but may
    // also be C-contiguous; either is borrowed with no transpose copy. Only a
    // non-contiguous / negatively-strided view (rare) is materialized C-order.
    let xowned;
    let (x_s, xrs, xcs): (&[f64], usize, usize) = match (
        xv.as_slice_memory_order(),
        strides_usize(xv.strides()),
    ) {
        (Some(s), Some((rs, cs))) => (s, rs, cs),
        _ => {
            xowned = xv.iter().copied().collect::<Vec<f64>>();
            (&xowned, p, 1)
        }
    };
    let eowned;
    let (e_s, ers, ecs): (&[f64], usize, usize) = match (
        ev.as_slice_memory_order(),
        strides_usize(ev.strides()),
    ) {
        (Some(s), Some((rs, cs))) => (s, rs, cs),
        _ => {
            eowned = ev.iter().copied().collect::<Vec<f64>>();
            (&eowned, p, 1)
        }
    };
    let wvec: Vec<f64> = w.as_array().iter().copied().collect();
    let zvec: Vec<f64> = z.as_array().iter().copied().collect();
    let xtwzvec: Vec<f64> = xtwz.as_array().iter().copied().collect();

    let res = py.allow_threads(|| {
        pls_core(
            x_s, xrs, xcs, &wvec, n, p, e_s, ne, ers, ecs, &zvec, &xtwzvec,
            use_xtwz,
        )
    });
    match res {
        Some((beta, r, log_det)) => {
            let rmat = PyArray2::from_vec2(
                py,
                &(0..p)
                    .map(|i| r[i * p..i * p + p].to_vec())
                    .collect::<Vec<_>>(),
            )?;
            Ok((true, PyArray1::from_vec(py, beta), rmat, log_det))
        }
        None => {
            let empty1 = PyArray1::from_vec(py, Vec::<f64>::new());
            let empty2 = PyArray2::from_vec2(py, &[Vec::<f64>::new()])?;
            Ok((false, empty1, empty2, f64::NAN))
        }
    }
}
