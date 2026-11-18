use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ralign::RVec;

pub(crate) fn dnrm2(v: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for &xi in v {
        if xi != 0.0 {
            let absxi = xi.abs();
            if scale < absxi {
                let r = scale / absxi;
                ssq = 1.0 + ssq * r * r;
                scale = absxi;
            } else {
                let r = absxi / scale;
                ssq += r * r;
            }
        }
    }
    if scale == 0.0 {
        0.0
    } else {
        scale * ssq.sqrt()
    }
}

/// The BLAS calls `dqrdc2.f` and `dqrsl.f` make (`dnrm2`, `ddot`, `daxpy`),
/// issued to the vendor library R's BLAS is, on sub-vectors at the
/// addresses the Fortran uses, and the norm downdate `1 - q**2` as one
/// fused multiply-subtract, as R's arm64 build computes it.
#[cfg(vendor_blas)]
mod rblas {
    use crate::lapack::blas;

    pub(super) fn nrm2(v: &[f64]) -> f64 {
        // SAFETY: `v` holds `v.len()` contiguous doubles.
        unsafe { blas::dnrm2(v.len() as i32, v.as_ptr(), 1) }
    }

    pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: both slices hold `a.len()` contiguous doubles.
        unsafe { blas::ddot(a.len() as i32, a.as_ptr(), 1, b.as_ptr(), 1) }
    }

    pub(super) fn axpy(t: f64, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), y.len());
        // SAFETY: both slices hold `x.len()` contiguous doubles and do not
        // overlap (`y` is borrowed mutably).
        unsafe { blas::daxpy(x.len() as i32, t, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
    }

    pub(super) fn one_minus_sq(q: f64) -> f64 {
        crate::lapack::fma(-q, q, 1.0)
    }
}

/// Portable stand-ins for the BLAS calls when no vendor BLAS is linked.
#[cfg(not(vendor_blas))]
mod rblas {
    pub(super) use super::dnrm2 as nrm2;

    pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    pub(super) fn axpy(t: f64, x: &[f64], y: &mut [f64]) {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi += t * xi;
        }
    }

    pub(super) fn one_minus_sq(q: f64) -> f64 {
        1.0 - q * q
    }
}

fn to_colmajor(xv: &numpy::ndarray::ArrayView2<f64>) -> Vec<f64> {
    let (n, p) = (xv.nrows(), xv.ncols());
    if !xv.is_standard_layout() {
        if let Some(s) = xv.as_slice_memory_order() {
            if s.len() == n * p {
                return s.to_vec(); // F-contiguous: column-major already
            }
        }
    }
    let mut v = vec![0.0_f64; n * p];
    for j in 0..p {
        for i in 0..n {
            v[j * n + i] = xv[[i, j]];
        }
    }
    v
}

fn dqrdc2(x: &mut [f64], n: usize, p: usize, tol: f64) -> (Vec<f64>, Vec<usize>, usize) {
    let mut qraux = vec![0.0_f64; p];
    let mut work1 = vec![0.0_f64; p]; // work(:,1) — original norms
    let mut work2 = vec![0.0_f64; p]; // work(:,2) — original norms, 0→1
    let mut jpvt: Vec<usize> = (1..=p).collect();

    if n > 0 {
        for j in 0..p {
            let nrm = rblas::nrm2(&x[j * n..j * n + n]);
            qraux[j] = nrm;
            work1[j] = nrm;
            work2[j] = if nrm == 0.0 { 1.0 } else { nrm };
        }
    }

    let lup = n.min(p);
    let mut k = p + 1; // 1-based rank boundary
    let mut tmpcol = vec![0.0_f64; n];
    for l in 1..=lup {
        let l0 = l - 1;
        while !(l >= k || qraux[l0] >= work2[l0] * tol) {
            tmpcol.copy_from_slice(&x[l0 * n..l0 * n + n]);
            x.copy_within((l0 + 1) * n..p * n, l0 * n);
            x[(p - 1) * n..p * n].copy_from_slice(&tmpcol);
            let (isv, tsv, tt1, tt2) = (jpvt[l0], qraux[l0], work1[l0], work2[l0]);
            for j in l0..p - 1 {
                jpvt[j] = jpvt[j + 1];
                qraux[j] = qraux[j + 1];
                work1[j] = work1[j + 1];
                work2[j] = work2[j + 1];
            }
            jpvt[p - 1] = isv;
            qraux[p - 1] = tsv;
            work1[p - 1] = tt1;
            work2[p - 1] = tt2;
            k -= 1;
        }
        if l != n {
            let col_l = l0 * n; // base of column l0
            let mut nrmxl = rblas::nrm2(&x[col_l + l0..col_l + n]);
            if nrmxl != 0.0 {
                if x[col_l + l0] != 0.0 {
                    nrmxl = nrmxl.copysign(x[col_l + l0]);
                }
                let inv = 1.0 / nrmxl;
                for xi in x[col_l + l0..col_l + n].iter_mut() {
                    *xi *= inv; // dscal
                }
                x[col_l + l0] += 1.0;
                let pivot_diag = x[col_l + l0];
                let (left, right) = x.split_at_mut((l0 + 1) * n);
                let cl = &left[col_l + l0..col_l + n]; // pivot col, rows l0..n
                let update = |cj: &mut [f64], qx: &mut f64, w1: &mut f64| {
                    let cj = &mut cj[l0..n]; // rows l0..n of this column
                    let t = -rblas::dot(cl, cj) / pivot_diag;
                    rblas::axpy(t, cl, cj);
                    if *qx != 0.0 {
                        let ratio = cj[0].abs() / *qx; // |x(l,j)|
                        let mut tt = rblas::one_minus_sq(ratio);
                        if tt < 0.0 {
                            tt = 0.0;
                        }
                        if tt.abs() >= 1e-6 {
                            *qx *= tt.sqrt();
                        } else {
                            *qx = rblas::nrm2(&cj[1..]); // dnrm2(x(l+1:n, j))
                            *w1 = *qx;
                        }
                    }
                };
                let qx = &mut qraux[l..p];
                let w1 = &mut work1[l..p];
                const PAR_MIN_ELEMS: usize = 1 << 15;
                if right.len() >= PAR_MIN_ELEMS {
                    right
                        .par_chunks_mut(n)
                        .zip(qx.par_iter_mut())
                        .zip(w1.par_iter_mut())
                        .for_each(|((cj, qx), w1)| update(cj, qx, w1));
                } else {
                    right
                        .chunks_mut(n)
                        .zip(qx.iter_mut())
                        .zip(w1.iter_mut())
                        .for_each(|((cj, qx), w1)| update(cj, qx, w1));
                }
                qraux[l0] = x[col_l + l0];
                x[col_l + l0] = -nrmxl;
            }
        }
    }
    let rank = (k - 1).min(n);
    (qraux, jpvt, rank)
}

/// `dqrsl` with `job = 1110`, as `dqrls.f` calls it: `qty` (holding `y` on
/// entry), the `k` coefficients in `b` and the residuals in `rsd`, written
/// into the caller's buffers.
fn dqrsl_1110(
    x: &mut [f64],
    n: usize,
    k: usize,
    qraux: &[f64],
    qty: &mut [f64],
    b: &mut [f64],
    rsd: &mut [f64],
) {
    let y0 = qty[0];
    let ju = k.min(n.saturating_sub(1));

    if ju == 0 {
        if n >= 1 && k >= 1 {
            if x[0] != 0.0 {
                b[0] = y0 / x[0];
            }
            rsd[0] = 0.0;
        }
        return;
    }

    for j in 1..=ju {
        let j0 = j - 1;
        if qraux[j0] != 0.0 {
            let col = j0 * n;
            let temp = x[col + j0];
            x[col + j0] = qraux[j0];
            let t = -rblas::dot(&x[col + j0..col + n], &qty[j0..n]) / x[col + j0];
            rblas::axpy(t, &x[col + j0..col + n], &mut qty[j0..n]);
            x[col + j0] = temp;
        }
    }

    b[..k].copy_from_slice(&qty[..k]);
    if k < n {
        rsd[k..n].copy_from_slice(&qty[k..n]);
    }
    for v in rsd.iter_mut().take(k) {
        *v = 0.0;
    }

    for jj in 1..=k {
        let j = k - jj + 1;
        let j0 = j - 1;
        let col = j0 * n;
        if x[col + j0] == 0.0 {
            break; // singular (info=j) — leaves b as is
        }
        b[j0] /= x[col + j0];
        if j != 1 {
            let t = -b[j0];
            rblas::axpy(t, &x[col..col + j0], &mut b[..j0]);
        }
    }

    for jj in 1..=ju {
        let j0 = (ju - jj + 1) - 1;
        if qraux[j0] != 0.0 {
            let col = j0 * n;
            let temp = x[col + j0];
            x[col + j0] = qraux[j0];
            let t = -rblas::dot(&x[col + j0..col + n], &rsd[j0..n]) / x[col + j0];
            rblas::axpy(t, &x[col + j0..col + n], &mut rsd[j0..n]);
            x[col + j0] = temp;
        }
    }
}

#[allow(clippy::type_complexity)]
fn dqrls_impl(
    xcol: &mut [f64],
    n: usize,
    p: usize,
    y: &[f64],
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, Vec<usize>, Vec<f64>) {
    // Cdqrls's coefficients, residuals and effects vectors, as R lays them out.
    let (qraux, jpvt, rank) = dqrdc2(xcol, n, p, tol);
    let mut coef = RVec::zeros(p);
    let mut qty = RVec::from_slice(y);
    let mut rsd = RVec::from_slice(y);
    if rank > 0 {
        rsd.as_mut_slice().fill(0.0);
        dqrsl_1110(
            xcol,
            n,
            rank,
            &qraux,
            qty.as_mut_slice(),
            coef.as_mut_slice(),
            rsd.as_mut_slice(),
        );
    }
    (
        coef.as_slice().to_vec(),
        rsd.as_slice().to_vec(),
        qty.as_slice().to_vec(),
        rank,
        jpvt,
        qraux,
    )
}

#[pyfunction]
#[pyo3(signature = (x, y, tol=1e-7))]
#[allow(clippy::type_complexity)]
pub fn dqrls<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let xv = x.as_array();
    let (n, p) = (xv.nrows(), xv.ncols());
    let yv = y.as_array();
    let mut xr = RVec::from_slice(&to_colmajor(&xv));
    let xcol = xr.as_mut_slice();
    let yvec: Vec<f64> = yv.iter().copied().collect();

    let (coef, rsd, qty, rank, jpvt, qraux) = dqrls_impl(xcol, n, p, &yvec, tol);

    let mut qr = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        for i in 0..n {
            qr[[i, j]] = xcol[j * n + i];
        }
    }
    let pivot: Vec<i64> = jpvt.iter().map(|&v| v as i64).collect();
    Ok((
        qr.into_pyarray(py),
        PyArray1::from_vec(py, coef),
        PyArray1::from_vec(py, rsd),
        PyArray1::from_vec(py, qty),
        rank,
        PyArray1::from_vec(py, pivot),
        PyArray1::from_vec(py, qraux),
    ))
}

#[pyfunction]
#[pyo3(signature = (x, tol=1e-7))]
pub fn dqrls_rank<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    tol: f64,
) -> PyResult<(usize, Bound<'py, PyArray1<i64>>)> {
    let xv = x.as_array();
    let (n, p) = (xv.nrows(), xv.ncols());
    let mut xr = RVec::from_slice(&to_colmajor(&xv));
    let (_qraux, jpvt, rank) = dqrdc2(xr.as_mut_slice(), n, p, tol);
    let pivot: Vec<i64> = jpvt.iter().map(|&v| v as i64).collect();
    Ok((rank, PyArray1::from_vec(py, pivot)))
}
