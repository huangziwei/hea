use crate::nmath::util::rfma;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

const DIST_PAR_MIN_PAIRS: usize = 64;

const DBL_MAX: f64 = f64::MAX;
const DBL_MIN: f64 = f64::MIN_POSITIVE;

#[inline]
fn r_euclidean(x: &[f64], nc: usize, i1: usize, i2: usize) -> f64 {
    let (mut p1, mut p2) = (i1 * nc, i2 * nc);
    let mut dist = 0.0;
    let mut count = 0usize;
    for _ in 0..nc {
        let (a, b) = (x[p1], x[p2]);
        if !a.is_nan() && !b.is_nan() {
            let dev = a - b;
            if !dev.is_nan() {
                dist = rfma(dev, dev, dist);
                count += 1;
            }
        }
        p1 += 1;
        p2 += 1;
    }
    if count == 0 {
        return f64::NAN;
    }
    if count != nc {
        dist /= (count as f64) / (nc as f64);
    }
    dist.sqrt()
}

#[inline]
fn r_maximum(x: &[f64], nc: usize, i1: usize, i2: usize) -> f64 {
    let (mut p1, mut p2) = (i1 * nc, i2 * nc);
    let mut dist = -DBL_MAX;
    let mut count = 0usize;
    for _ in 0..nc {
        let (a, b) = (x[p1], x[p2]);
        if !a.is_nan() && !b.is_nan() {
            let dev = (a - b).abs();
            if !dev.is_nan() {
                if dev > dist {
                    dist = dev;
                }
                count += 1;
            }
        }
        p1 += 1;
        p2 += 1;
    }
    if count == 0 {
        return f64::NAN;
    }
    dist
}

#[inline]
fn r_manhattan(x: &[f64], nc: usize, i1: usize, i2: usize) -> f64 {
    let (mut p1, mut p2) = (i1 * nc, i2 * nc);
    let mut dist = 0.0;
    let mut count = 0usize;
    for _ in 0..nc {
        let (a, b) = (x[p1], x[p2]);
        if !a.is_nan() && !b.is_nan() {
            let dev = (a - b).abs();
            if !dev.is_nan() {
                dist += dev;
                count += 1;
            }
        }
        p1 += 1;
        p2 += 1;
    }
    if count == 0 {
        return f64::NAN;
    }
    if count != nc {
        dist /= (count as f64) / (nc as f64);
    }
    dist
}

#[inline]
fn r_canberra(x: &[f64], nc: usize, i1: usize, i2: usize) -> f64 {
    let (mut p1, mut p2) = (i1 * nc, i2 * nc);
    let mut dist = 0.0;
    let mut count = 0usize;
    for _ in 0..nc {
        let (a, b) = (x[p1], x[p2]);
        if !a.is_nan() && !b.is_nan() {
            let sum = a.abs() + b.abs();
            let diff = (a - b).abs();
            if sum > DBL_MIN || diff > DBL_MIN {
                let mut dev = diff / sum;
                if !dev.is_nan()
                    || (!diff.is_finite() && diff == sum && {
                        dev = 1.0;
                        true
                    })
                {
                    dist += dev;
                    count += 1;
                }
            }
        }
        p1 += 1;
        p2 += 1;
    }
    if count == 0 {
        return f64::NAN;
    }
    if count != nc {
        dist /= (count as f64) / (nc as f64);
    }
    dist
}

#[inline]
fn r_dist_binary(x: &[f64], nc: usize, i1: usize, i2: usize, nonfinite: &AtomicBool) -> f64 {
    let (mut p1, mut p2) = (i1 * nc, i2 * nc);
    let mut total = 0i64;
    let mut count = 0i64;
    let mut dist = 0i64;
    for _ in 0..nc {
        let (a, b) = (x[p1], x[p2]);
        if !a.is_nan() && !b.is_nan() {
            if !(a.is_finite() && b.is_finite()) {
                nonfinite.store(true, Ordering::Relaxed); // warn after the map
            } else {
                if a != 0.0 || b != 0.0 {
                    count += 1;
                    if !(a != 0.0 && b != 0.0) {
                        dist += 1;
                    }
                }
                total += 1;
            }
        }
        p1 += 1;
        p2 += 1;
    }
    if total == 0 {
        return f64::NAN;
    }
    if count == 0 {
        return 0.0;
    }
    (dist as f64) / (count as f64)
}

#[inline]
fn r_pow_nonneg(x: f64, y: f64) -> f64 {
    if y == 2.0 {
        return x * x;
    }
    if y == 3.0 {
        return if x <= 11.0 { x * x * x } else { x.powf(3.0) };
    }
    if y == 4.0 {
        return if x <= 11.0 {
            x * x * x * x
        } else {
            x.powf(4.0)
        };
    }
    x.powf(y)
}

#[inline]
fn r_minkowski(x: &[f64], nc: usize, i1: usize, i2: usize, p: f64) -> f64 {
    let (mut p1, mut p2) = (i1 * nc, i2 * nc);
    let mut dist = 0.0;
    let mut count = 0usize;
    for _ in 0..nc {
        let (a, b) = (x[p1], x[p2]);
        if !a.is_nan() && !b.is_nan() {
            let dev = a - b;
            if !dev.is_nan() {
                dist += r_pow_nonneg(dev.abs(), p);
                count += 1;
            }
        }
        p1 += 1;
        p2 += 1;
    }
    if count == 0 {
        return f64::NAN;
    }
    if count != nc {
        dist /= (count as f64) / (nc as f64);
    }
    r_pow_nonneg(dist, 1.0 / p)
}

#[pyfunction]
#[pyo3(name = "cdist", signature = (x, method, p))]
pub fn cdist<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    method: usize,
    p: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let view = x.as_array();
    let nr = view.shape()[0];
    let nc = view.shape()[1];
    let xs: &[f64] = x.as_slice()?; // C-contiguous: row-major flat

    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(nr * nr.saturating_sub(1) / 2);
    for j in 0..nr {
        for i in (j + 1)..nr {
            pairs.push((i, j));
        }
    }
    let m = pairs.len();
    let nonfinite = AtomicBool::new(false);

    let kernel = |&(i, j): &(usize, usize)| -> f64 {
        match method {
            0 => r_euclidean(xs, nc, i, j),
            1 => r_maximum(xs, nc, i, j),
            2 => r_manhattan(xs, nc, i, j),
            3 => r_canberra(xs, nc, i, j),
            4 => r_dist_binary(xs, nc, i, j, &nonfinite),
            5 => r_minkowski(xs, nc, i, j, p),
            _ => f64::NAN,
        }
    };

    let data: Vec<f64> = if m >= DIST_PAR_MIN_PAIRS {
        py.allow_threads(|| pairs.par_iter().map(kernel).collect())
    } else {
        pairs.iter().map(kernel).collect()
    };

    if nonfinite.load(Ordering::Relaxed) {
        let warnings = py.import("warnings")?;
        warnings.call_method1("warn", ("treating non-finite values as NA",))?;
    }

    Ok(data.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cdist, m)?)?;
    Ok(())
}
