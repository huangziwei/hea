//! Python entry points for [`crate::lapack`], R's bundled LAPACK.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::lapack::chol::{dpotrf_u, dpotri_u, dpstrf_u};
use crate::lapack::gesdd::{dgesdd, dgesdd_lwork};
use crate::lapack::lu::dgesv;
use crate::lapack::ormtr::{dormqr_l, dormqr_l_lwork};
use crate::lapack::qr::{dgeqp3, dgeqp3_free, dgeqp3_lwork};
use crate::lapack::stedc::{dstedc, dstedc_lwork};
use crate::lapack::syevr::{dsyevr_all_lower, dsyevr_lwork};
use crate::ralign::RVec;

/// The columns of `rows` (one matrix column per row, `nrow` long) laid out
/// column-major at leading dimension `ld`.
fn col_major(rows: &numpy::ndarray::ArrayView2<'_, f64>, nrow: usize, ld: usize) -> Vec<f64> {
    let ncol = rows.shape()[0];
    let mut x = vec![0.0f64; ld * ncol.max(1)];
    for j in 0..ncol {
        for i in 0..nrow {
            x[i + j * ld] = rows[[j, i]];
        }
    }
    x
}

/// The `n`-by-`n` matrix whose columns are the rows of `at`, in an `RVec`
/// at R's vector layout (`r_layout`) or on a 16-byte boundary.
fn square_rvec(
    at: &numpy::ndarray::ArrayView2<'_, f64>,
    r_layout: bool,
) -> PyResult<(RVec, usize)> {
    let n = at.shape()[0];
    if at.shape()[1] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the matrix must be square",
        ));
    }
    let mut a = if r_layout {
        RVec::zeros(n * n)
    } else {
        RVec::zeros_aligned(n * n)
    };
    let s = a.as_mut_slice();
    for j in 0..n {
        for i in 0..n {
            s[i + j * n] = at[[j, i]];
        }
    }
    Ok((a, n))
}

fn rows_of(a: &RVec, n: usize) -> Array2<f64> {
    let s = a.as_slice();
    Array2::from_shape_fn((n, n), |(j, i)| s[i + j * n])
}

/// LAPACK `dpstrf('U')` with tolerance `tol` on a copy of the symmetric
/// matrix (only its upper triangle is read), as R's `chol(pivot = TRUE)`
/// (`r_layout`) or mgcv's `mgcv_chol` call it. Returns `(ut, piv, rank,
/// info)`: the factor's columns as the rows of `ut`, untouched below the
/// diagonal, the 1-based pivots, the rank and `INFO`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn r_dpstrf<'py>(
    py: Python<'py>,
    at: PyReadonlyArray2<'py, f64>,
    tol: f64,
    r_layout: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i32>>,
    i32,
    i32,
)> {
    let (mut a, n) = square_rvec(&at.as_array(), r_layout)?;
    let mut piv = vec![0i32; n.max(1)];
    let mut work = vec![0.0f64; 2 * n.max(1)];
    let (rank, info) = py.allow_threads(|| {
        // SAFETY: `a` holds `n`-by-`n` at leading dimension `n`, `piv` `n`
        // and `work` `2n`; none alias.
        unsafe {
            dpstrf_u(
                n as i32,
                a.as_mut_slice().as_mut_ptr(),
                n.max(1) as i32,
                piv.as_mut_ptr(),
                tol,
                work.as_mut_ptr(),
            )
        }
    });
    piv.truncate(n);
    Ok((
        rows_of(&a, n).into_pyarray(py),
        piv.into_pyarray(py),
        rank,
        info,
    ))
}

/// LAPACK `dpotrf('U')` on a copy of the symmetric matrix laid out as R's
/// `chol()` lays it (`La_chol`). Returns `(ut, info)`, the factor's columns
/// as the rows of `ut`.
#[pyfunction]
fn r_dpotrf<'py>(
    py: Python<'py>,
    at: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, i32)> {
    let (mut a, n) = square_rvec(&at.as_array(), true)?;
    let info = py.allow_threads(|| {
        // SAFETY: `a` holds `n`-by-`n` at leading dimension `n`.
        unsafe { dpotrf_u(n as i32, a.as_mut_slice().as_mut_ptr(), n.max(1) as i32) }
    });
    Ok((rows_of(&a, n).into_pyarray(py), info))
}

/// R's `chol2inv(x, size = n)` (`La_chol2inv`): LAPACK `dpotri('U')` on a
/// copy of the `n`-by-`n` upper triangle laid out where R allocates it, the
/// result mirrored to the lower triangle. `at` holds the matrix's columns as
/// its rows. Returns `(inverse, info)`.
#[pyfunction]
fn r_dpotri<'py>(
    py: Python<'py>,
    at: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, i32)> {
    let (mut a, n) = square_rvec(&at.as_array(), true)?;
    let info = py.allow_threads(|| {
        // SAFETY: `a` holds `n`-by-`n` at leading dimension `n`.
        unsafe { dpotri_u(n as i32, a.as_mut_slice().as_mut_ptr(), n.max(1) as i32) }
    });
    let s = a.as_mut_slice();
    for j in 0..n {
        for i in j + 1..n {
            s[i + j * n] = s[j + i * n];
        }
    }
    Ok((rows_of(&a, n).into_pyarray(py), info))
}

/// LAPACK `dstedc('I')` on the symmetric tridiagonal matrix with diagonal
/// `d` and off-diagonal `e`, the call in mgcv's `mgcv_trisymeig`. Returns
/// `(w, zt, info)`: the eigenvalues ascending and the eigenvectors as the
/// rows of `zt`.
#[pyfunction]
fn r_dstedc<'py>(
    py: Python<'py>,
    d: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, i32)> {
    let mut d = d.as_array().to_vec();
    let n = d.len();
    let e_in = e.as_array();
    if n > 0 && e_in.len() + 1 < n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_dstedc: e must hold n - 1 off-diagonal entries",
        ));
    }
    let mut e = vec![0.0f64; n.max(1)];
    for i in 0..n.saturating_sub(1) {
        e[i] = e_in[i];
    }
    let ldz = n.max(1);
    let mut z = vec![0.0f64; ldz * n.max(1)];
    let (lwork, liwork) = dstedc_lwork(n as i32);
    let mut work = vec![0.0f64; lwork.max(1)];
    let mut iwork = vec![0i32; liwork.max(1)];
    let info = py.allow_threads(|| {
        // SAFETY: every array is sized as DSTEDC's documentation requires for
        // COMPZ = 'I' (`dstedc_lwork`), and none of them alias.
        unsafe {
            dstedc(
                b'I',
                n as i32,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                z.as_mut_ptr(),
                ldz as i32,
                work.as_mut_ptr(),
                iwork.as_mut_ptr(),
            )
        }
    });
    z.truncate(n * n);
    let zt = Array2::from_shape_vec((n, n), z).unwrap();
    Ok((d.into_pyarray(py), zt.into_pyarray(py), info))
}

/// LAPACK `dsyevr('V' or 'N', 'A', 'L', ...)` with `abstol = 0` and the
/// workspace its query returns, the call base R's `La_rs` makes for
/// `eigen(symmetric = TRUE)` (the lower triangle of `a` is read). Returns
/// `(w, zt, info)`: the eigenvalues ascending and, unless `only_values`, the
/// eigenvectors as the rows of `zt`.
#[pyfunction]
fn r_dsyevr<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    only_values: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, i32)> {
    let av = a.as_array();
    let n = av.shape()[0];
    if av.shape()[1] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_dsyevr: the matrix must be square",
        ));
    }
    let ld = n.max(1);
    let mut x = vec![0.0f64; ld * n.max(1)];
    for j in 0..n {
        for i in 0..n {
            x[i + j * ld] = av[[i, j]];
        }
    }
    let (lwork, liwork) = dsyevr_lwork(n as i32);
    let mut w = vec![0.0f64; n.max(1)];
    let mut z = vec![0.0f64; ld * n.max(1)];
    let mut isuppz = vec![0i32; 2 * n.max(1)];
    let mut work = vec![0.0f64; lwork];
    let mut iwork = vec![0i32; liwork];
    let (_m, info) = py.allow_threads(|| {
        // SAFETY: `x` and `z` are `n`-by-`n` at leading dimension `ld`, `w`
        // holds `n`, `isuppz` `2 n`, and `work`/`iwork` the sizes DSYEVR's
        // workspace query returns; none of them alias.
        unsafe {
            dsyevr_all_lower(
                !only_values,
                n as i32,
                x.as_mut_ptr(),
                ld as i32,
                w.as_mut_ptr(),
                z.as_mut_ptr(),
                ld as i32,
                isuppz.as_mut_ptr(),
                work.as_mut_ptr(),
                lwork as i32,
                iwork.as_mut_ptr(),
            )
        }
    });
    w.truncate(n);
    let zt = if only_values {
        Array2::zeros((0, n))
    } else {
        z.truncate(n * n);
        Array2::from_shape_vec((n, n), z).unwrap()
    };
    Ok((w.into_pyarray(py), zt.into_pyarray(py), info))
}

/// LAPACK `dgeqp3` with every column free and the workspace its query
/// returns, base R's `La_qr` for `qr(x, LAPACK = TRUE)`; `at` holds the
/// columns of `x` as its rows. Returns `(qrt, tau, jpvt)`: the factored
/// matrix with its columns as the rows of `qrt`, the `min(m, n)` reflector
/// scalars and the 1-based column pivot.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn r_dgeqp3<'py>(
    py: Python<'py>,
    at: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let av = at.as_array();
    let (n, m) = (av.shape()[0], av.shape()[1]);
    let ld = m.max(1);
    let mut x = col_major(&av, m, ld);
    let k = m.min(n);
    let mut tau = vec![0.0f64; k.max(1)];
    let mut jpvt = vec![0i32; n.max(1)];
    let lwork = dgeqp3_lwork(m as i32, n as i32);
    let mut work = vec![0.0f64; lwork.max(1) as usize];
    py.allow_threads(|| {
        // SAFETY: `x` is `m`-by-`n` at leading dimension `ld`, `tau` holds
        // `min(m, n)`, `jpvt` `n`, and `work` the `lwork` doubles DGEQP3's
        // query returns; none of them alias.
        unsafe {
            dgeqp3_free(
                m as i32,
                n as i32,
                x.as_mut_ptr(),
                ld as i32,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                lwork,
            )
        }
    });
    x.truncate(ld * n);
    tau.truncate(k);
    jpvt.truncate(n);
    let qrt = Array2::from_shape_vec((n, ld), x).unwrap();
    Ok((
        qrt.into_pyarray(py),
        tau.into_pyarray(py),
        jpvt.into_pyarray(py),
    ))
}

/// LAPACK `dgeqp3` as defined, with the workspace its query returns: a
/// column whose `jpvt` entry is non-zero on entry is fixed (moved to the
/// front, factored unpivoted), the rest pivoted by norm - mgcv's `mgcv_qr`,
/// whose callers may leave earlier pivots in `jpvt`. `at` holds the columns
/// of `x` as its rows. Returns `(qrt, tau, jpvt)` as [`r_dgeqp3`].
#[pyfunction]
#[allow(clippy::type_complexity)]
fn r_dgeqp3_jpvt<'py>(
    py: Python<'py>,
    at: PyReadonlyArray2<'py, f64>,
    jpvt_in: PyReadonlyArray1<'py, i32>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let av = at.as_array();
    let (n, m) = (av.shape()[0], av.shape()[1]);
    if jpvt_in.as_array().len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_dgeqp3_jpvt: jpvt needs one entry per column",
        ));
    }
    let ld = m.max(1);
    let mut x = col_major(&av, m, ld);
    let k = m.min(n);
    let mut tau = vec![0.0f64; k.max(1)];
    let mut jpvt = vec![0i32; n.max(1)];
    jpvt[..n].copy_from_slice(&jpvt_in.as_array().to_vec());
    let lwork = dgeqp3_lwork(m as i32, n as i32);
    let mut work = vec![0.0f64; lwork.max(1) as usize];
    py.allow_threads(|| {
        // SAFETY: `x` is `m`-by-`n` at leading dimension `ld`, `tau` holds
        // `min(m, n)`, `jpvt` `n`, and `work` the `lwork` doubles DGEQP3's
        // query returns; none of them alias.
        unsafe {
            dgeqp3(
                m as i32,
                n as i32,
                x.as_mut_ptr(),
                ld as i32,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                lwork,
            )
        }
    });
    x.truncate(ld * n);
    tau.truncate(k);
    jpvt.truncate(n);
    let qrt = Array2::from_shape_vec((n, ld), x).unwrap();
    Ok((
        qrt.into_pyarray(py),
        tau.into_pyarray(py),
        jpvt.into_pyarray(py),
    ))
}

/// LAPACK `dormqr('L', 'T' or 'N', ...)` with the workspace its query
/// returns, base R's `qr_qy_real` for `qr.qty` (`trans`) and `qr.qy` of a
/// LAPACK QR: `C := Q' C` or `C := Q C` for the `Q` whose first
/// `len(tau)` reflectors are the columns of `qrt`'s rows. `ct` holds the
/// columns of `C` as its rows; the result has the same layout.
#[pyfunction]
fn r_dormqr_l<'py>(
    py: Python<'py>,
    qrt: PyReadonlyArray2<'py, f64>,
    tau: PyReadonlyArray1<'py, f64>,
    ct: PyReadonlyArray2<'py, f64>,
    trans: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let qv = qrt.as_array();
    let cv = ct.as_array();
    let m = qv.shape()[1];
    let tau = tau.as_array().to_vec();
    let k = tau.len();
    let nrhs = cv.shape()[0];
    if cv.shape()[1] != m {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "right-hand side should have {m} not {} rows",
            cv.shape()[1]
        )));
    }
    if k > qv.shape()[0] || k > m {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_dormqr_l: more reflectors than columns or rows",
        ));
    }
    let ld = m.max(1);
    let a = col_major(&qv, m, ld);
    let mut c = col_major(&cv, m, ld);
    let lwork = dormqr_l_lwork(nrhs as i32);
    let mut work = vec![0.0f64; lwork as usize];
    py.allow_threads(|| {
        // SAFETY: `a` holds the reflectors (`m` rows at leading dimension
        // `ld`, at least `k` columns), `c` is `m`-by-`nrhs` at `ld`, `tau`
        // holds `k`, and `work` the `lwork` doubles DORMQR's query returns;
        // none of them alias.
        unsafe {
            dormqr_l(
                if trans { b'T' } else { b'N' },
                m as i32,
                nrhs as i32,
                k as i32,
                a.as_ptr(),
                ld as i32,
                tau.as_ptr(),
                c.as_mut_ptr(),
                ld as i32,
                work.as_mut_ptr(),
                lwork,
            )
        }
    });
    c.truncate(ld * nrhs);
    let out = Array2::from_shape_vec((nrhs, ld), c).unwrap();
    Ok(out.into_pyarray(py))
}

/// LAPACK `dgesv`, base R's `La_solve` for `solve(a, b)`: `at` and `bt`
/// hold the columns of `a` (square) and `b` as their rows. Returns
/// `(xt, lut, ipiv, info)`: the solution and the LU factors in the same
/// layout, the 1-based row pivots and `INFO`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn r_dgesv<'py>(
    py: Python<'py>,
    at: PyReadonlyArray2<'py, f64>,
    bt: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i32>>,
    i32,
)> {
    let av = at.as_array();
    let bv = bt.as_array();
    let n = av.shape()[0];
    if av.shape()[1] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_dgesv: the matrix must be square",
        ));
    }
    if bv.shape()[1] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_dgesv: b must have as many rows as a",
        ));
    }
    let nrhs = bv.shape()[0];
    let ld = n.max(1);
    let mut a = col_major(&av, n, ld);
    let mut b = col_major(&bv, n, ld);
    let mut ipiv = vec![0i32; n.max(1)];
    let info = py.allow_threads(|| {
        // SAFETY: `a` is `n`-by-`n` and `b` `n`-by-`nrhs`, both at leading
        // dimension `ld`; `ipiv` holds `n`; none of them alias.
        unsafe {
            dgesv(
                n as i32,
                nrhs as i32,
                a.as_mut_ptr(),
                ld as i32,
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                ld as i32,
            )
        }
    });
    a.truncate(ld * n);
    b.truncate(ld * nrhs);
    ipiv.truncate(n);
    let lut = Array2::from_shape_vec((n, ld), a).unwrap();
    let xt = Array2::from_shape_vec((nrhs, ld), b).unwrap();
    Ok((
        xt.into_pyarray(py),
        lut.into_pyarray(py),
        ipiv.into_pyarray(py),
        info,
    ))
}

/// LAPACK `dgesdd` with the workspace its `lwork = -1` query returns, as
/// base R's `La_svd` calls it (`jobz` 'N', 'S' or 'A', every buffer an R
/// vector) or mgcv's `mgcv_svd_full` (`jobz` 'O', `ldu = m`, the workspace
/// from `calloc`: `calloc_work`). `at` holds the columns of the `m`-by-`n`
/// `x` as its rows. Returns `(s, ut, vtt, xt, info)`: the singular values;
/// `U` (`m`-by-`k`) and `V'` (`r`-by-`n`) with their columns as the rows of
/// `ut` and `vtt` (`k`, `r` as `jobz` sets them, 0 when not formed); and
/// `x` as `dgesdd` leaves it (for 'O' with `m >= n`, the columns of `U`),
/// columns as rows.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn r_dgesdd<'py>(
    py: Python<'py>,
    at: PyReadonlyArray2<'py, f64>,
    jobz: &str,
    calloc_work: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    i32,
)> {
    let av = at.as_array();
    let (n, m) = (av.shape()[0], av.shape()[1]);
    let mn = m.min(n);
    let jb = match jobz {
        "N" | "S" | "O" | "A" => jobz.as_bytes()[0],
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "jobz must be one of 'N', 'S', 'O', 'A'",
            ))
        }
    };
    // (rows, columns) of U and V' as dgesdd fills them.
    let (ldu, ucol, ldvt, vtcol) = match jb {
        b'S' => (m, mn, mn, n),
        b'A' => (m, m, n, n),
        b'O' if m >= n => (m, 0, n, n),
        b'O' => (m, m, 1, 0),
        _ => (1, 0, 1, 0),
    };
    let lda = m.max(1);
    let mut x = RVec::zeros(lda * n);
    {
        let xs = x.as_mut_slice();
        for j in 0..n {
            for i in 0..m {
                xs[i + j * lda] = av[[j, i]];
            }
        }
    }
    let mut s = RVec::zeros(mn);
    let mut u = RVec::zeros((ldu.max(1) * ucol).max(1));
    let mut vt = RVec::zeros((ldvt.max(1) * vtcol).max(1));
    let lwork = dgesdd_lwork(jb, m as i32, n as i32).max(1);
    let mut work = if calloc_work {
        RVec::zeros_aligned(lwork as usize)
    } else {
        RVec::zeros(lwork as usize)
    };
    let mut iwork = vec![0i32; 8 * mn.max(1)];
    let info = py.allow_threads(|| {
        // SAFETY: `x` is `m`-by-`n` at leading dimension `lda`, `s` holds
        // `min(m, n)`, `u` and `vt` the `ldu`-by-`ucol` and `ldvt`-by-`vtcol`
        // matrices JOBZ asks for (one element when unreferenced), `work` the
        // `lwork` doubles DGESDD's query returns and `iwork` `8 min(m, n)`.
        unsafe {
            dgesdd(
                jb,
                m as i32,
                n as i32,
                x.as_mut_slice().as_mut_ptr(),
                lda as i32,
                s.as_mut_slice().as_mut_ptr(),
                u.as_mut_slice().as_mut_ptr(),
                ldu.max(1) as i32,
                vt.as_mut_slice().as_mut_ptr(),
                ldvt.max(1) as i32,
                work.as_mut_slice().as_mut_ptr(),
                lwork,
                iwork.as_mut_ptr(),
            )
        }
    });
    let us = u.as_slice();
    let ut = Array2::from_shape_fn((ucol, m), |(j, i)| us[i + j * ldu.max(1)]);
    let vs = vt.as_slice();
    let vtt = Array2::from_shape_fn((vtcol, ldvt), |(j, i)| vs[i + j * ldvt.max(1)]);
    let xs = x.as_slice();
    let xt = Array2::from_shape_fn((n, m), |(j, i)| xs[i + j * lda]);
    Ok((
        s.as_slice().to_vec().into_pyarray(py),
        ut.into_pyarray(py),
        vtt.into_pyarray(py),
        xt.into_pyarray(py),
        info,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(r_dstedc, m)?)?;
    m.add_function(wrap_pyfunction!(r_dsyevr, m)?)?;
    m.add_function(wrap_pyfunction!(r_dgeqp3, m)?)?;
    m.add_function(wrap_pyfunction!(r_dgeqp3_jpvt, m)?)?;
    m.add_function(wrap_pyfunction!(r_dpstrf, m)?)?;
    m.add_function(wrap_pyfunction!(r_dpotrf, m)?)?;
    m.add_function(wrap_pyfunction!(r_dpotri, m)?)?;
    m.add_function(wrap_pyfunction!(r_dormqr_l, m)?)?;
    m.add_function(wrap_pyfunction!(r_dgesv, m)?)?;
    m.add_function(wrap_pyfunction!(r_dgesdd, m)?)?;
    Ok(())
}
