use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn dpotf2_lower(a: &mut [f64], n: usize) -> i32 {
    for j in 0..n {
        let mut dot = 0.0;
        for k in 0..j {
            dot += a[j * n + k] * a[j * n + k];
        }
        let ajj = a[j * n + j] - dot;
        if !(ajj > 0.0) {
            a[j * n + j] = ajj;
            return (j + 1) as i32;
        }
        let ajj = ajj.sqrt();
        a[j * n + j] = ajj;
        if j + 1 < n {
            let inv = 1.0 / ajj; // DSCAL multiplies by ONE/AJJ
            for i in (j + 1)..n {
                let mut temp = 0.0;
                for k in 0..j {
                    temp += a[i * n + k] * a[j * n + k];
                }
                a[i * n + j] = (a[i * n + j] - temp) * inv;
            }
        }
    }
    0
}

#[pyfunction]
pub fn chol_lower<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let av = a.as_array();
    let (nr, nc) = (av.nrows(), av.ncols());
    if nr != nc {
        return Err(PyValueError::new_err("chol_lower: matrix must be square"));
    }
    let n = nr;
    let mut buf: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..=i {
            buf[i * n + k] = av[[i, k]];
        }
    }
    let info = dpotf2_lower(&mut buf, n);
    if info != 0 {
        return Err(PyValueError::new_err(format!(
            "chol_lower: not positive definite (pivot {info})"
        )));
    }
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for k in 0..=i {
            out[[i, k]] = buf[i * n + k];
        }
    }
    Ok(out.into_pyarray(py))
}
