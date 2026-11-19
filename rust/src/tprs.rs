//! Thin-plate regression spline kernel evaluation (`s(bs="tp")`, the default).
//!
//! Mirrors mgcv's `XBuild` (src/tprs.c:560, the knots≠data path): for each data
//! row `x_i`, build the length-(nu+M) vector
//!     b = [ η(‖x_i − Xu_j‖²)  for j in 0..nu | ∏_k x_i,k^{p_{l,k}}  for l in 0..M ]
//! which the caller multiplies by `UZ` to get the design row. This is the
//! n-dependent hot path (a √ over the n×nu radial table); rayon parallelises it
//! over rows — which numpy/Accelerate can't, their ufuncs are single-threaded.
//!
//! Bit-exactness: the fill is element-wise independent, so the parallel fill is
//! byte-for-byte the serial one; and the per-element arithmetic mirrors
//! `hea/formula.py::_tp_eval_X_raw` / `_tp_T` exactly, so `b` equals the numpy
//! build. The caller keeps the BLAS `b @ UZ` matmul, so the design is identical.
//!
//! FMA/contraction parity with arm64 R: tprs.c is compiled `clang -O2`, which
//! fuses single-expression `a*b+c` to `fmadd` where the ISA has baseline FMA
//! (aarch64 yes, x86-64 no). So the squared-distance accumulate `z=a-b; r+=z*z;`
//! (tprs.c:92 tpsE, :591 XBuild) is `fma(z,z,r)` on arm64 — mirrored here via
//! `rfma` (per-arch). And the null-space monomial is built by *repeated multiply*
//! `r=1; for(kk<pin) r*=xx[j];` (tprs.c:156 tpsT, :598 XBuild) — NOT `powi`, which
//! reassociates at powers ≥4 even on x86. Both fixes keep `b` 0-ulp to the numpy
//! `_rfma_vec`-fold build on every platform.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nmath::util::rfma;

/// `fast_eta` (tprs.c:61) — η given r² in `rsq`, constant in `f0`. Same
/// multiplication order as `_tp_fast_eta_vec`: d-odd is maskless (√0 = 0 gives
/// the exact 0), d-even guards log(0) via the r²≤0 early return.
#[inline]
#[allow(clippy::manual_is_multiple_of)] // mirror C/Python `d % 2 == 0` verbatim
fn fast_eta(m: usize, d: usize, rsq: f64, f0: f64) -> f64 {
    if rsq <= 0.0 {
        return 0.0;
    }
    let d2 = d / 2;
    let mut f = f0;
    if d % 2 == 0 {
        f *= rsq.ln() * 0.5;
        for _ in 0..(m - d2) {
            f *= rsq;
        }
    } else {
        for _ in 0..(m - d2 - 1) {
            f *= rsq;
        }
        f *= rsq.sqrt();
    }
    f
}

#[pyfunction]
#[pyo3(name = "tp_eval_b")]
fn tp_eval_b<'py>(
    py: Python<'py>,
    x_c: PyReadonlyArray2<'py, f64>,
    xu: PyReadonlyArray2<'py, f64>,
    m: usize,
    d: usize,
    eta0: f64,
    poly_powers: PyReadonlyArray2<'py, i64>,
) -> Bound<'py, PyArray2<f64>> {
    let x = x_c.as_array();
    let u = xu.as_array();
    let pp = poly_powers.as_array();
    let n = x.shape()[0];
    let nu = u.shape()[0];
    let mm = pp.shape()[0]; // M = null-space polynomial columns
    let ncol = nu + mm;

    let x_flat: Vec<f64> = x.iter().copied().collect(); // (n, d)
    let xu_flat: Vec<f64> = u.iter().copied().collect(); // (nu, d)
    let pp_flat: Vec<i64> = pp.iter().copied().collect(); // (M, d)

    let mut out = vec![0.0f64; n * ncol];
    let fill = |i: usize, row: &mut [f64]| {
        let xi = &x_flat[i * d..i * d + d];
        for j in 0..nu {
            let uj = &xu_flat[j * d..j * d + d];
            let mut r = 0.0;
            for k in 0..d {
                let z = xi[k] - uj[k];
                r = rfma(z, z, r); // tprs.c:591 `z=a-b; r+=z*z;` → fmadd on arm64
            }
            row[j] = fast_eta(m, d, r, eta0);
        }
        for l in 0..mm {
            // null-space monomial ∏_k x_k^{pin[l,k]} by repeated multiply, exactly
            // as tprs.c:598 `r=1; for(j) for(kk<pin) r*=xx[j];` (NOT powi).
            let mut t = 1.0;
            for k in 0..d {
                let pk = pp_flat[l * d + k];
                for _ in 0..pk {
                    t *= xi[k];
                }
            }
            row[nu + l] = t;
        }
    };

    if n >= 256 {
        py.allow_threads(|| {
            out.par_chunks_mut(ncol)
                .enumerate()
                .for_each(|(i, row)| fill(i, row));
        });
    } else {
        out.chunks_mut(ncol)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    Array2::from_shape_vec((n, ncol), out)
        .unwrap()
        .into_pyarray(py)
}

/// `tpsE` (tprs.c:76) — the (nu, nu) penalty/radial matrix on the knot grid,
/// `E_ij = η(‖Xu_i − Xu_j‖²)`. The n-independent knot-side cost (a √ over the
/// nu² table) that floors basis time at small n. Rayon over rows; element-wise
/// independent ⇒ bit-identical to the numpy `_tp_E`. Computes the full matrix
/// (each row independent for safe parallel writes); the diagonal is 0 (r²=0).
#[pyfunction]
#[pyo3(name = "tp_eval_E")] // Python name keeps the math-symbol E (cf. `_tp_E`)
fn tp_eval_e<'py>(
    py: Python<'py>,
    xu: PyReadonlyArray2<'py, f64>,
    m: usize,
    d: usize,
    eta0: f64,
) -> Bound<'py, PyArray2<f64>> {
    let u = xu.as_array();
    let nu = u.shape()[0];
    let xu_flat: Vec<f64> = u.iter().copied().collect(); // (nu, d) row-major
    let mut out = vec![0.0f64; nu * nu];
    let fill = |i: usize, row: &mut [f64]| {
        let xi = &xu_flat[i * d..i * d + d];
        for j in 0..nu {
            let uj = &xu_flat[j * d..j * d + d];
            let mut r = 0.0;
            for k in 0..d {
                let z = xi[k] - uj[k];
                r = rfma(z, z, r); // tprs.c:92 `x=a-b; r+=x*x;` → fmadd on arm64
            }
            row[j] = fast_eta(m, d, r, eta0);
        }
    };
    if nu >= 256 {
        py.allow_threads(|| {
            out.par_chunks_mut(nu)
                .enumerate()
                .for_each(|(i, row)| fill(i, row));
        });
    } else {
        out.chunks_mut(nu)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    Array2::from_shape_vec((nu, nu), out)
        .unwrap()
        .into_pyarray(py)
}

/// Column sums of squares of `X` (n, k) as `tprs_setup` forms them before
/// rescaling each column to unit rms (tprs.c:494): `w=0; for (j<n) w +=
/// X[j][i]*X[j][i];` -- row order, fused to `fmadd` on arm64 like the
/// distance accumulate above.
#[pyfunction]
fn tp_col_sumsq<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let a = x.as_array();
    let (n, k) = (a.shape()[0], a.shape()[1]);
    let mut out = vec![0.0f64; k];
    for (i, o) in out.iter_mut().enumerate() {
        let mut w = 0.0;
        for j in 0..n {
            let v = a[[j, i]];
            w = rfma(v, v, w);
        }
        *o = w;
    }
    out.into_pyarray(py)
}

/// The tp design block `[eta(|x_i - Xu_j|) | T(x_i)] UZ` row by row, as
/// tprs.c builds it (`tprs_setup` :446-470, `predict_tprs` :585-605): each
/// row's radial and polynomial values into one reused buffer `b` (the values
/// `tp_eval_b` computes), then `dgemv('T')` for `(UZ)'b`. Rows run in
/// parallel, each worker with its own `b`, so the n-by-kk matrix of `b`s is
/// never formed.
#[cfg(vendor_blas)]
#[pyfunction]
#[pyo3(name = "tp_eval_x")]
#[allow(clippy::too_many_arguments)]
fn tp_eval_x<'py>(
    py: Python<'py>,
    x_c: PyReadonlyArray2<'py, f64>,
    xu: PyReadonlyArray2<'py, f64>,
    m: usize,
    d: usize,
    eta0: f64,
    poly_powers: PyReadonlyArray2<'py, i64>,
    uz: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (x, u, pp, uv) = (
        x_c.as_array(),
        xu.as_array(),
        poly_powers.as_array(),
        uz.as_array(),
    );
    let (n, nu, mm) = (x.shape()[0], u.shape()[0], pp.shape()[0]);
    let kk = nu + mm;
    if uv.shape()[0] != kk {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tp_eval_x: UZ must have one row per knot and polynomial term",
        ));
    }
    let k = uv.shape()[1];
    let x_flat: Vec<f64> = x.iter().copied().collect();
    let xu_flat: Vec<f64> = u.iter().copied().collect();
    let pp_flat: Vec<i64> = pp.iter().copied().collect();
    let mut uzc = vec![0.0f64; kk * k];
    for j in 0..k {
        for i in 0..kk {
            uzc[i + j * kk] = uv[[i, j]];
        }
    }
    let fill = |i: usize, b: &mut [f64]| {
        let xi = &x_flat[i * d..i * d + d];
        for j in 0..nu {
            let uj = &xu_flat[j * d..j * d + d];
            let mut r = 0.0;
            for kd in 0..d {
                let z = xi[kd] - uj[kd];
                r = rfma(z, z, r);
            }
            b[j] = fast_eta(m, d, r, eta0);
        }
        for l in 0..mm {
            let mut t = 1.0;
            for kd in 0..d {
                for _ in 0..pp_flat[l * d + kd] {
                    t *= xi[kd];
                }
            }
            b[nu + l] = t;
        }
    };
    let mut out = vec![0.0f64; n * k];
    if k > 0 {
        py.allow_threads(|| {
            out.par_chunks_mut(k).enumerate().for_each_init(
                || vec![0.0f64; kk],
                |b, (i, row)| {
                    fill(i, b);
                    crate::sparse::blas::gemv_t_set(kk, k, &uzc, kk, b, row);
                },
            );
        });
    }
    Ok(Array2::from_shape_vec((n, k), out)
        .unwrap()
        .into_pyarray(py))
}

/// `(UZ)'b` for every row `b` of `B` (n, kk): one BLAS `dgemv('T', kk, k,
/// 1.0, UZ, kk, b, 1, 0.0, a, 1)` per row, the call tprs.c makes in
/// `tprs_setup` (:466) and `predict_tprs` (:603). A single `B UZ` `dgemm`
/// rounds differently from those row calls, so the rows are issued one by
/// one, in parallel since each is independent.
#[cfg(vendor_blas)]
#[pyfunction]
fn tp_uz_rows<'py>(
    py: Python<'py>,
    b: PyReadonlyArray2<'py, f64>,
    uz: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (bv, uv) = (b.as_array(), uz.as_array());
    let (n, kk) = (bv.shape()[0], bv.shape()[1]);
    if uv.shape()[0] != kk {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tp_uz_rows: B and UZ disagree on the inner dimension",
        ));
    }
    let k = uv.shape()[1];
    let mut uzc = vec![0.0f64; kk * k];
    for j in 0..k {
        for i in 0..kk {
            uzc[i + j * kk] = uv[[i, j]];
        }
    }
    let brows: Vec<f64> = bv.iter().copied().collect();
    let mut out = vec![0.0f64; n * k];
    if k > 0 {
        py.allow_threads(|| {
            out.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
                crate::sparse::blas::gemv_t_set(kk, k, &uzc, kk, &brows[i * kk..(i + 1) * kk], row);
            });
        });
    }
    Ok(Array2::from_shape_vec((n, k), out)
        .unwrap()
        .into_pyarray(py))
}

/// R's ``qr.qty(qr, t(Y))`` as LINPACK ``dqrqty`` computes it, returned as
/// ``t(.)``: ``dqrsl`` with ``job = 1000`` on each row of ``Y`` (n columns),
/// which is a column of ``t(Y)``. For ``j < min(k, n - 1)`` with
/// ``qraux[j] != 0``, ``x(j,j)`` is set to ``qraux(j)`` for the step and
/// ``qty(j:) += t x(j:, j)`` with ``t = -ddot(x(j:, j), qty(j:)) / x(j,j)``.
/// ``x`` is the factor ``qr_t`` (``t(qr)``, so ``x`` column-major in its
/// rows) and ``qty`` the rows of ``Y``, in a buffer laid out as the copy of
/// ``t(Y)`` that ``.Fortran`` passes, so each row sits where ``dqrsl``
/// addresses it: the vendor ``ddot`` rounds by its second operand's
/// alignment. Columns are
/// independent, so each worker updates its own copy of ``x``.
#[cfg(vendor_blas)]
#[pyfunction]
fn dqrqty_rows<'py>(
    py: Python<'py>,
    qr_t: PyReadonlyArray2<'py, f64>,
    qraux: numpy::PyReadonlyArray1<'py, f64>,
    k: usize,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    use crate::lapack::blas::{daxpy, ddot};
    let (xv, qv, yv) = (qr_t.as_array(), qraux.as_array(), y.as_array());
    let (kq, n) = (xv.shape()[0], xv.shape()[1]);
    let ny = yv.shape()[0];
    if yv.shape()[1] != n || qv.len() < kq.min(k) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dqrqty_rows: Y rows must match nrow(qr), qraux must cover k",
        ));
    }
    let x: Vec<f64> = xv.iter().copied().collect();
    let qa: Vec<f64> = qv.iter().copied().collect();
    let ycopy: Vec<f64> = yv.iter().copied().collect();
    let mut qty_buf = crate::ralign::RVec::from_slice(&ycopy);
    let ju = k.min(kq).min(n.saturating_sub(1));
    if n > 0 && ju > 0 {
        let out = qty_buf.as_mut_slice();
        py.allow_threads(|| {
            out.par_chunks_mut(n).for_each_init(
                || x.clone(),
                |xl, qty| {
                    for j in 0..ju {
                        if qa[j] == 0.0 {
                            continue;
                        }
                        let dj = j * n + j;
                        let temp = xl[dj];
                        xl[dj] = qa[j];
                        let len = (n - j) as i32;
                        // SAFETY: `x(j:, j)` and `qty(j:)` both hold `n - j`
                        // entries of their own buffers, which do not alias.
                        unsafe {
                            let t =
                                -ddot(len, xl.as_ptr().add(dj), 1, qty.as_ptr().add(j), 1) / xl[dj];
                            daxpy(len, t, xl.as_ptr().add(dj), 1, qty.as_mut_ptr().add(j), 1);
                        }
                        xl[dj] = temp;
                    }
                },
            );
        });
    }
    Ok(Array2::from_shape_vec((ny, n), qty_buf.as_slice().to_vec())
        .unwrap()
        .into_pyarray(py))
}

/// mgcv matrix.c `matmult(C, A, B, tA=1, tB=0)`, `C = A'B`: `C` zeroed,
/// then for each row `k` of `A` in order `C[i][j] += A[k][i] B[k][j]`,
/// fused, so every entry is an in-order fused sum over `k`.
#[pyfunction]
#[pyo3(name = "tp_matmult_ta")]
fn tp_matmult_ta<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (av, bv) = (a.as_array(), b.as_array());
    let (nk, p) = (av.shape()[0], av.shape()[1]);
    if bv.shape()[0] != nk {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tp_matmult_ta: A and B must have the same number of rows",
        ));
    }
    let q = bv.shape()[1];
    let mut c = vec![0.0f64; p * q];
    for k in 0..nk {
        for i in 0..p {
            let aki = av[[k, i]];
            for j in 0..q {
                c[i * q + j] = rfma(aki, bv[[k, j]], c[i * q + j]);
            }
        }
    }
    Ok(Array2::from_shape_vec((p, q), c).unwrap().into_pyarray(py))
}

/// mgcv's `Rlanczos` (mat.c:3637-3880) on one thread: symmetric Lanczos
/// with full re-orthogonalization for the `m` largest and `lm` smallest
/// eigenpairs of the symmetric `a`, or with `lm < 0` the `m` of largest
/// magnitude (tprs.c:408 calls it so, `tol = DBL_EPSILON^0.7`). `z = A q_j`
/// is BLAS `dsymv` on the upper triangle and the two Gram-Schmidt passes
/// `ddot`/`daxpy`, as mgcv issues them; the plain C loops (the start
/// vector's norm, `a_j = q_j'z`, `b_j = |z|`, the three-term update and the
/// Ritz sums) run in index order with the contraction R's arm64 build
/// applies (`rfma`), so `z - a_j q_j - b_{j-1} q_{j-1}` is `z - fma(a_j, q_j,
/// b_{j-1} q_{j-1})`. Each convergence check solves the tridiagonal as
/// `mgcv_trisymeig` does (mat.c:3579): `dstedc('I')`, then the eigenpairs
/// reversed into descending order. Returns `(D, U)`.
#[cfg(vendor_blas)]
#[pyfunction]
#[pyo3(name = "tp_rlanczos")]
fn tp_rlanczos<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    m: i64,
    lm: i64,
    tol: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    use crate::lapack::stedc::{dstedc, dstedc_lwork};
    use crate::sparse::blas;

    let shape = a.as_array().shape().to_vec();
    let n = shape[0];
    if shape[1] != n || n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tp_rlanczos: A must be a non-empty square matrix",
        ));
    }
    // `a` is symmetric, so its row-major buffer is also its column-major one
    let amat: &[f64] = a.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("tp_rlanczos: A must be C-contiguous")
    })?;
    let res = py.allow_threads(|| {
        let biggest = lm < 0;
        let mut m = m.max(0) as usize;
        let mut lm = if biggest { 0 } else { lm as usize };
        let mut f_check = ((m + lm) / 2).max(10);
        let kk = (n / 10).max(1);
        if kk < f_check {
            f_check = kk;
        }
        let mut q: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        let mut q0 = vec![0.0f64; n];
        let (mut jran, ia, ic, im) = (1u64, 106u64, 1283u64, 6075u64);
        let mut bt = 0.0f64;
        for v in q0.iter_mut() {
            jran = (jran * ia + ic) % im;
            *v = jran as f64 / im as f64 - 0.5;
            bt = rfma(*v, *v, bt);
        }
        bt = bt.sqrt();
        for v in q0.iter_mut() {
            *v /= bt;
        }
        q.push(q0);
        let mut av_ = vec![0.0f64; n];
        let mut bv = vec![0.0f64; n];
        let mut err = vec![1e300f64; n];
        let mut z = vec![0.0f64; n];
        let mut d: Vec<f64> = Vec::new();
        let mut v: Vec<f64> = Vec::new();
        let mut vlength = 0usize;
        let mut j = 0usize;
        while j < n {
            blas::symv_u_set(n, amat, n, &q[j], &mut z);
            let mut xx = 0.0f64;
            for (qi, zi) in q[j].iter().zip(z.iter()) {
                xx = rfma(*qi, *zi, xx);
            }
            av_[j] = xx;
            if j == 0 {
                for (zi, qi) in z.iter_mut().zip(q[j].iter()) {
                    *zi = rfma(-xx, *qi, *zi);
                }
            } else {
                let yy = bv[j - 1];
                for i in 0..n {
                    z[i] -= rfma(xx, q[j][i], yy * q[j - 1][i]);
                }
                for _rep in 0..2 {
                    for qi in q.iter().take(j + 1) {
                        let c = -blas::dot(&z, qi);
                        blas::axpy(c, qi, &mut z);
                    }
                }
            }
            let mut ss = 0.0f64;
            for zi in z.iter() {
                ss = rfma(*zi, *zi, ss);
            }
            bv[j] = ss.sqrt();
            if j < n - 1 {
                let bj = bv[j];
                q.push(z.iter().map(|zi| zi / bj).collect());
            }
            if (j >= m + lm && j.is_multiple_of(f_check)) || j == n - 1 {
                let k2 = j + 1;
                let mut dd = av_[..k2].to_vec();
                let mut gg = vec![0.0f64; k2];
                gg[..j].copy_from_slice(&bv[..j]);
                let mut vv = vec![0.0f64; k2 * k2];
                let (lwork, liwork) = dstedc_lwork(k2 as i32);
                let mut work = vec![0.0f64; lwork.max(1)];
                let mut iwork = vec![0i32; liwork.max(1)];
                // SAFETY: every array is sized as DSTEDC requires for COMPZ =
                // 'I' (`dstedc_lwork`), and none of them alias.
                unsafe {
                    dstedc(
                        b'I',
                        k2 as i32,
                        dd.as_mut_ptr(),
                        gg.as_mut_ptr(),
                        vv.as_mut_ptr(),
                        k2 as i32,
                        work.as_mut_ptr(),
                        iwork.as_mut_ptr(),
                    );
                }
                for i in 0..k2 / 2 {
                    dd.swap(i, k2 - i - 1);
                    for r in 0..k2 {
                        vv.swap(k2 * i + r, k2 * (k2 - i - 1) + r);
                    }
                }
                d = dd;
                v = vv;
                vlength = k2;
                let norm_tj = d[0].abs().max(d[j].abs());
                for k in 0..=j {
                    err[k] = (bv[j] * v[k * vlength + j]).abs();
                }
                if j >= m + lm {
                    let max_err = norm_tj * tol;
                    if biggest {
                        let (mut pi, mut ni) = (0usize, 0usize);
                        let mut converged = true;
                        while pi + ni < m {
                            if d[pi].abs() >= d[j - ni].abs() {
                                if err[pi] > max_err {
                                    converged = false;
                                    break;
                                }
                                pi += 1;
                            } else {
                                // mat.c tests err[ni] here, not err[j - ni]
                                if err[ni] > max_err {
                                    converged = false;
                                    break;
                                }
                                ni += 1;
                            }
                        }
                        if converged {
                            m = pi;
                            lm = ni;
                            j += 1;
                            break;
                        }
                    } else {
                        let mut ok = err[..m].iter().all(|e| *e <= max_err);
                        for e in err[(j + 1 - lm.min(j + 1))..=j].iter() {
                            if *e > max_err {
                                ok = false;
                            }
                        }
                        if ok {
                            j += 1;
                            break;
                        }
                    }
                }
            }
            j += 1;
        }
        if vlength == 0 {
            return None;
        }
        let ncol = m + lm;
        let mut dout = vec![0.0f64; ncol];
        let mut u = vec![0.0f64; n * ncol];
        for k in 0..ncol {
            let col = if k < m { k } else { j - (lm + m - k) };
            dout[k] = d[col];
            let uk = &mut u[k * n..(k + 1) * n];
            for (l, ql) in q.iter().enumerate().take(j) {
                let xx = v[l + col * vlength];
                for (ui, qi) in uk.iter_mut().zip(ql.iter()) {
                    *ui = rfma(*qi, xx, *ui);
                }
            }
        }
        Some((dout, u, ncol))
    });
    let (dout, u, ncol) = res.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("tp_rlanczos: no eigen-decomposition was formed")
    })?;
    let mut out = Array2::<f64>::zeros((n, ncol));
    for k in 0..ncol {
        for i in 0..n {
            out[[i, k]] = u[i + k * n];
        }
    }
    Ok((dout.into_pyarray(py), out.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tp_matmult_ta, m)?)?;
    #[cfg(vendor_blas)]
    m.add_function(wrap_pyfunction!(dqrqty_rows, m)?)?;
    #[cfg(vendor_blas)]
    m.add_function(wrap_pyfunction!(tp_uz_rows, m)?)?;
    #[cfg(vendor_blas)]
    m.add_function(wrap_pyfunction!(tp_eval_x, m)?)?;
    #[cfg(vendor_blas)]
    m.add_function(wrap_pyfunction!(tp_rlanczos, m)?)?;
    m.add_function(wrap_pyfunction!(tp_col_sumsq, m)?)?;
    m.add_function(wrap_pyfunction!(tp_eval_b, m)?)?;
    m.add_function(wrap_pyfunction!(tp_eval_e, m)?)?;
    Ok(())
}
