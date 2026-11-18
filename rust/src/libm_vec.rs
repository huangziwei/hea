//! Elementwise C-library math over `f64` arrays: the platform libm's `exp`,
//! `log`, `pow`, ... called once per element, as R's arithmetic calls them.
//!
//! numpy's float64 ufuncs pick an implementation by the CPU they run on: on
//! x86-64 its AVX2 and AVX-512 kernels for `exp`, `log`, `power` and others
//! are not libm's and round differently in the last bit, while R calls the C
//! library on every machine. These functions call the C symbols themselves
//! (Rust's own `asinh` is not libm's, so nothing goes through `f64` methods).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Below this many elements one thread is faster: one C call costs a few
/// nanoseconds, and handing a slice to the pool tens of microseconds.
const PAR_MIN: usize = 1 << 15;

fn map1(py: Python<'_>, a: &[f64], f: impl Fn(f64) -> f64 + Sync + Send) -> Vec<f64> {
    if a.len() >= PAR_MIN {
        py.allow_threads(|| a.par_iter().map(|&x| f(x)).collect())
    } else {
        a.iter().map(|&x| f(x)).collect()
    }
}

fn map2(
    py: Python<'_>,
    a: &[f64],
    b: &[f64],
    f: impl Fn(f64, f64) -> f64 + Sync + Send,
) -> Vec<f64> {
    if a.len() >= PAR_MIN {
        py.allow_threads(|| {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| f(x, y))
                .collect()
        })
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
    }
}

mod c {
    extern "C" {
        pub fn exp(x: f64) -> f64;
        pub fn log(x: f64) -> f64;
        pub fn log1p(x: f64) -> f64;
        pub fn expm1(x: f64) -> f64;
        pub fn log2(x: f64) -> f64;
        pub fn log10(x: f64) -> f64;
        pub fn sin(x: f64) -> f64;
        pub fn cos(x: f64) -> f64;
        pub fn tan(x: f64) -> f64;
        pub fn sinh(x: f64) -> f64;
        pub fn cosh(x: f64) -> f64;
        pub fn tanh(x: f64) -> f64;
        pub fn atan(x: f64) -> f64;
        pub fn acos(x: f64) -> f64;
        pub fn asin(x: f64) -> f64;
        pub fn asinh(x: f64) -> f64;
        pub fn pow(x: f64, y: f64) -> f64;
        pub fn atan2(y: f64, x: f64) -> f64;
        pub fn hypot(x: f64, y: f64) -> f64;
    }
}

fn slice<'a>(a: &'a PyReadonlyArrayDyn<'_, f64>) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("libm: the array must be C-contiguous")
    })
}

macro_rules! unary {
    ($py_name:ident, $scalar_name:ident, $c:ident) => {
        #[pyfunction]
        pub fn $py_name<'py>(
            py: Python<'py>,
            x: PyReadonlyArrayDyn<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let x = slice(&x)?;
            // SAFETY: a C library function of one double, defined for every input.
            Ok(map1(py, x, |v| unsafe { c::$c(v) }).into_pyarray(py))
        }

        #[pyfunction]
        pub fn $scalar_name(x: f64) -> f64 {
            // SAFETY: as above.
            unsafe { c::$c(x) }
        }
    };
}

macro_rules! binary {
    ($py_name:ident, $scalar_name:ident, $c:ident) => {
        #[pyfunction]
        pub fn $py_name<'py>(
            py: Python<'py>,
            x: PyReadonlyArrayDyn<'py, f64>,
            y: PyReadonlyArrayDyn<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let (x, y) = (slice(&x)?, slice(&y)?);
            if x.len() != y.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "libm: length mismatch",
                ));
            }
            // SAFETY: a C library function of two doubles, defined for every input.
            Ok(map2(py, x, y, |a, b| unsafe { c::$c(a, b) }).into_pyarray(py))
        }

        #[pyfunction]
        pub fn $scalar_name(x: f64, y: f64) -> f64 {
            // SAFETY: as above.
            unsafe { c::$c(x, y) }
        }
    };
}

unary!(libm_exp, libm_exp1, exp);
unary!(libm_log, libm_log1, log);
unary!(libm_log1p, libm_log1p1, log1p);
unary!(libm_expm1, libm_expm11, expm1);
unary!(libm_log2, libm_log21, log2);
unary!(libm_log10, libm_log101, log10);
unary!(libm_sin, libm_sin1, sin);
unary!(libm_cos, libm_cos1, cos);
unary!(libm_tan, libm_tan1, tan);
unary!(libm_sinh, libm_sinh1, sinh);
unary!(libm_cosh, libm_cosh1, cosh);
unary!(libm_tanh, libm_tanh1, tanh);
unary!(libm_atan, libm_atan1, atan);
unary!(libm_acos, libm_acos1, acos);
unary!(libm_asin, libm_asin1, asin);
unary!(libm_asinh, libm_asinh1, asinh);
binary!(libm_pow, libm_pow1, pow);
binary!(libm_atan2, libm_atan21, atan2);
binary!(libm_hypot, libm_hypot1, hypot);

/// `pow(x[i], y)`: an array of bases and one exponent.
#[pyfunction]
pub fn libm_pow_ys<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = slice(&x)?;
    // SAFETY: `pow` is defined for every pair of doubles.
    Ok(map1(py, x, |v| unsafe { c::pow(v, y) }).into_pyarray(py))
}

/// `pow(x, y[i])`: one base and an array of exponents.
#[pyfunction]
pub fn libm_pow_xs<'py>(
    py: Python<'py>,
    x: f64,
    y: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = slice(&y)?;
    // SAFETY: as above.
    Ok(map1(py, y, |v| unsafe { c::pow(x, v) }).into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(libm_exp, m)?,
        wrap_pyfunction!(libm_exp1, m)?,
        wrap_pyfunction!(libm_log, m)?,
        wrap_pyfunction!(libm_log1, m)?,
        wrap_pyfunction!(libm_log1p, m)?,
        wrap_pyfunction!(libm_log1p1, m)?,
        wrap_pyfunction!(libm_expm1, m)?,
        wrap_pyfunction!(libm_expm11, m)?,
        wrap_pyfunction!(libm_log2, m)?,
        wrap_pyfunction!(libm_log21, m)?,
        wrap_pyfunction!(libm_log10, m)?,
        wrap_pyfunction!(libm_log101, m)?,
        wrap_pyfunction!(libm_sin, m)?,
        wrap_pyfunction!(libm_sin1, m)?,
        wrap_pyfunction!(libm_cos, m)?,
        wrap_pyfunction!(libm_cos1, m)?,
        wrap_pyfunction!(libm_tan, m)?,
        wrap_pyfunction!(libm_tan1, m)?,
        wrap_pyfunction!(libm_sinh, m)?,
        wrap_pyfunction!(libm_sinh1, m)?,
        wrap_pyfunction!(libm_cosh, m)?,
        wrap_pyfunction!(libm_cosh1, m)?,
        wrap_pyfunction!(libm_tanh, m)?,
        wrap_pyfunction!(libm_tanh1, m)?,
        wrap_pyfunction!(libm_atan, m)?,
        wrap_pyfunction!(libm_atan1, m)?,
        wrap_pyfunction!(libm_acos, m)?,
        wrap_pyfunction!(libm_acos1, m)?,
        wrap_pyfunction!(libm_asin, m)?,
        wrap_pyfunction!(libm_asin1, m)?,
        wrap_pyfunction!(libm_asinh, m)?,
        wrap_pyfunction!(libm_asinh1, m)?,
        wrap_pyfunction!(libm_pow, m)?,
        wrap_pyfunction!(libm_pow1, m)?,
        wrap_pyfunction!(libm_atan2, m)?,
        wrap_pyfunction!(libm_atan21, m)?,
        wrap_pyfunction!(libm_hypot, m)?,
        wrap_pyfunction!(libm_hypot1, m)?,
        wrap_pyfunction!(libm_pow_ys, m)?,
        wrap_pyfunction!(libm_pow_xs, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
