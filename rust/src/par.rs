use pyo3::prelude::*;
use rayon::prelude::*;

pub const PAR_THRESHOLD: usize = 2048;

#[inline]
pub fn map1<F>(py: Python<'_>, a: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
        py.allow_threads(|| a.par_iter().map(|&x| f(x)).collect())
    } else {
        a.iter().map(|&x| f(x)).collect()
    }
}

#[inline]
pub fn map2<F>(py: Python<'_>, a: &[f64], b: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
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

#[inline]
pub fn map3<F>(py: Python<'_>, a: &[f64], b: &[f64], c: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64, f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
        py.allow_threads(|| {
            a.par_iter()
                .zip(b.par_iter())
                .zip(c.par_iter())
                .map(|((&x, &y), &z)| f(x, y, z))
                .collect()
        })
    } else {
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((&x, &y), &z)| f(x, y, z))
            .collect()
    }
}

#[inline]
pub fn map4<F>(py: Python<'_>, a: &[f64], b: &[f64], c: &[f64], d: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64, f64, f64) -> f64 + Sync + Send,
{
    if a.len() >= PAR_THRESHOLD {
        py.allow_threads(|| {
            a.par_iter()
                .zip(b.par_iter())
                .zip(c.par_iter())
                .zip(d.par_iter())
                .map(|(((&x, &y), &z), &w)| f(x, y, z, w))
                .collect()
        })
    } else {
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .zip(d.iter())
            .map(|(((&x, &y), &z), &w)| f(x, y, z, w))
            .collect()
    }
}

#[inline]
pub fn map_index<F>(py: Python<'_>, n: usize, f: F) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync + Send,
{
    if n >= PAR_THRESHOLD {
        py.allow_threads(|| (0..n).into_par_iter().map(|i| f(i)).collect())
    } else {
        (0..n).map(|i| f(i)).collect()
    }
}
