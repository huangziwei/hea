//! R-optimizer kernels (`nlm`'s UNCMIN and `optim`'s L-BFGS-B 2.3) —
//! the compiled path behind `hea/R/optimize.py`. The Python modules
//! (`hea/R/uncmin.py`, `hea/R/lbfgsb.py`) are the spec and the test
//! oracle; the R-level semantics (nlm's function-value cache, msg
//! bits, optim's fnscale/parscale) stay in Python — only the numeric
//! driver loops move here, with the objective called back into Python
//! per evaluation.

mod lbfgsb;
mod linpack;
mod uncmin;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Adapter: Python callables → the uncmin `Obj` trait. `fcn(x)->float`,
/// `d1fcn(x)->seq[n]`, `d2fcn(x, a)->None` (fills the n×n lower
/// triangle; `a` is passed as a fresh 1-D column-major buffer and
/// copied back).
struct PyObj<'py> {
    py: Python<'py>,
    fcn: PyObject,
    d1fcn: PyObject,
    d2fcn: Option<PyObject>,
}

impl uncmin::Obj for PyObj<'_> {
    fn fcn(&mut self, x: &[f64]) -> PyResult<f64> {
        let xa = x.to_vec().into_pyarray(self.py);
        let args = PyTuple::new(self.py, [xa])?;
        self.fcn.call1(self.py, args)?.extract::<f64>(self.py)
    }

    fn d1fcn(&mut self, x: &[f64], g: &mut [f64]) -> PyResult<()> {
        let xa = x.to_vec().into_pyarray(self.py);
        let args = PyTuple::new(self.py, [xa])?;
        let res = self.d1fcn.call1(self.py, args)?;
        let v: Vec<f64> = res.extract(self.py)?;
        g[..v.len()].copy_from_slice(&v);
        Ok(())
    }

    fn d2fcn(&mut self, x: &[f64], a: &mut [f64], nr: usize) -> PyResult<()> {
        let d2 = self.d2fcn.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("d2fcn requested but not supplied")
        })?;
        let n = x.len();
        let xa = x.to_vec().into_pyarray(self.py);
        let args = PyTuple::new(self.py, [xa])?;
        let res = d2.call1(self.py, args)?;
        let v: Vec<f64> = res.extract(self.py)?; // n*n column-major
        for j in 0..n {
            for i in j..n {
                a[i + j * nr] = v[i + j * n];
            }
        }
        Ok(())
    }
}

/// uncmin `optif9` with Python-callback objective. Returns
/// (xpls, fpls, gpls, itrmcd, itncnt, msg).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x0, fcn, d1fcn, d2fcn, typsiz, fscale, method, iexp, msg,
                    ndigit, itnlim, iagflg, iahflg, dlt, gradtl, stepmx, steptl))]
pub fn optif9(
    py: Python<'_>,
    x0: PyReadonlyArray1<f64>,
    fcn: PyObject,
    d1fcn: PyObject,
    d2fcn: Option<PyObject>,
    typsiz: PyReadonlyArray1<f64>,
    fscale: f64,
    method: i32,
    iexp: i32,
    msg: i32,
    ndigit: i32,
    itnlim: i32,
    iagflg: i32,
    iahflg: i32,
    dlt: f64,
    gradtl: f64,
    stepmx: f64,
    steptl: f64,
) -> PyResult<(Py<PyArray1<f64>>, f64, Py<PyArray1<f64>>, i32, i32, i32)> {
    let mut x = x0.as_slice()?.to_vec();
    let mut ts = typsiz.as_slice()?.to_vec();
    let n = x.len();
    let mut xpls = vec![0.0; n];
    let mut gpls = vec![0.0; n];
    let mut obj = PyObj {
        py,
        fcn,
        d1fcn,
        d2fcn,
    };
    let (fpls, itrmcd, itncnt, msg_out) = uncmin::optdrv(
        n, &mut x, &mut obj, &mut ts, fscale, method, iexp, msg, ndigit, itnlim, iagflg, iahflg,
        dlt, gradtl, stepmx, steptl, &mut xpls, &mut gpls,
    )?;
    Ok((
        xpls.into_pyarray(py).unbind(),
        fpls,
        gpls.into_pyarray(py).unbind(),
        itrmcd,
        itncnt,
        msg_out,
    ))
}

/// uncmin `fdhess` (nlm's hessian=TRUE). Returns the n×n Hessian upper
/// triangle filled, flat column-major.
#[pyfunction]
#[pyo3(signature = (x, fval, fcn, ndigit, typsiz))]
pub fn uncmin_fdhess(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    fval: f64,
    fcn: PyObject,
    ndigit: i32,
    typsiz: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let mut xv = x.as_slice()?.to_vec();
    let ts = typsiz.as_slice()?.to_vec();
    let n = xv.len();
    let mut h = vec![0.0; n * n];
    let mut obj = PyObj {
        py,
        fcn,
        d1fcn: py.None(),
        d2fcn: None,
    };
    uncmin::fdhess(n, &mut xv, fval, &mut obj, &mut h, n, ndigit, &ts)?;
    Ok(h.into_pyarray(py).unbind())
}

struct PyLbObj<'py> {
    py: Python<'py>,
    fminfn: PyObject,
    fmingr: PyObject,
}

impl lbfgsb::Objective for PyLbObj<'_> {
    fn f(&mut self, x: &[f64]) -> PyResult<f64> {
        let xa = x.to_vec().into_pyarray(self.py);
        let args = PyTuple::new(self.py, [xa])?;
        self.fminfn.call1(self.py, args)?.extract::<f64>(self.py)
    }

    fn grad(&mut self, x: &[f64], g: &mut [f64]) -> PyResult<()> {
        let xa = x.to_vec().into_pyarray(self.py);
        let args = PyTuple::new(self.py, [xa])?;
        let res = self.fmingr.call1(self.py, args)?;
        let v: Vec<f64> = res.extract(self.py)?;
        g[..v.len()].copy_from_slice(&v);
        Ok(())
    }
}

/// R's `lbfgsb()` driver with Python-callback objective (`fminfn(x)`
/// returns the scaled f; `fmingr(x)` RETURNS the scaled gradient).
/// Returns (x, Fmin, fail, fncount, grcount, msg).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (n, m, x0, lo, up, nbd, fminfn, fmingr, factr, pgtol, maxit))]
pub fn lbfgsb_drive(
    py: Python<'_>,
    n: usize,
    m: usize,
    x0: PyReadonlyArray1<f64>,
    lo: PyReadonlyArray1<f64>,
    up: PyReadonlyArray1<f64>,
    nbd: PyReadonlyArray1<i64>,
    fminfn: PyObject,
    fmingr: PyObject,
    factr: f64,
    pgtol: f64,
    maxit: i32,
) -> PyResult<(Py<PyArray1<f64>>, f64, i32, i32, i32, String)> {
    let mut x = x0.as_slice()?.to_vec();
    let lov = lo.as_slice()?.to_vec();
    let upv = up.as_slice()?.to_vec();
    let nbdv: Vec<i32> = nbd.as_slice()?.iter().map(|&v| v as i32).collect();
    let mut obj = PyLbObj { py, fminfn, fmingr };
    let (fmin, fail, fncount, grcount, msg) = lbfgsb::lbfgsb_drive(
        n, m, &mut x, &lov, &upv, &nbdv, &mut obj, factr, pgtol, maxit,
    )?;
    Ok((
        x.into_pyarray(py).unbind(),
        fmin,
        fail,
        fncount,
        grcount,
        msg,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optif9, m)?)?;
    m.add_function(wrap_pyfunction!(uncmin_fdhess, m)?)?;
    m.add_function(wrap_pyfunction!(lbfgsb_drive, m)?)?;
    Ok(())
}
