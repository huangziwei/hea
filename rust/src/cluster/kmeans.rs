//! k-means — ALL three of R's algorithms in one module (R splits them across
//! `kmeans_kmns.f` (Hartigan-Wong, Fortran) and `cluster_kmeans.c` (Lloyd /
//! MacQueen, C) — a historical language artifact we don't mirror; they're all
//! `kmeans()`):
//!   * `kmns`      — Hartigan-Wong (mirrors `_kmns`; OPTRA/QTRAN, serial),
//!   * `lloyd`     — Lloyd / Forgy (mirrors `_kmeans_lloyd`),
//!   * `macqueen`  — MacQueen     (mirrors `_kmeans_macqueen`).
//!
//! All float reductions (per-point distance, centroid accumulation, WSS) are
//! kept SEQUENTIAL in the same order as the C/Fortran/Python, so parity is
//! 0-ulp. Lloyd's assignment phase (independent per point, `argmin` only — no
//! cross-point float reduction) is parallelized with rayon (parallel == serial
//! bit-for-bit); HW and MacQueen's incremental refinement are inherently serial.
//! Empty clusters divide by zero → ±inf/NaN exactly as the numpy path (IEEE).

use crate::nmath::util::rfma;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

const KM_PAR_MIN: usize = 256;
const BIG: f64 = 1.0e30; // Hartigan-Wong "infinity" sentinel (matches _kmns)

/// Σ (a−b)² over equal-length contiguous rows, accumulated SEQUENTIALLY (same
/// order as the C/Fortran/Python `for j: dd += (x-c)^2`) → 0-ulp. Slices give the
/// compiler the length, so no per-element index math / bounds check, and the
/// row-major access is contiguous (cache-friendlier than R's column-major
/// `X(I,J)` strided-by-M access).
#[inline]
fn sqdist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// The two closest centres (1-based, `ic1`/`ic2` with `dt[ic1] <= dt[ic2]`) to
/// 0-based point `i0` — the Hartigan-Wong initial assignment. Independent per
/// point (reads x, cen; no shared state), so the init map is rayon-parallel
/// while staying 0-ulp.
#[inline]
fn two_closest(x: &[f64], cen: &[f64], p: usize, k: usize, i0: usize) -> (usize, usize) {
    let xrow = &x[i0 * p..i0 * p + p];
    let (mut ic1, mut ic2) = (1usize, 2usize);
    let mut dt = [sqdist(xrow, &cen[0..p]), sqdist(xrow, &cen[p..2 * p])];
    if dt[0] > dt[1] {
        ic1 = 2;
        ic2 = 1;
        dt.swap(0, 1);
    }
    for ell in 3..=k {
        let crow = &cen[(ell - 1) * p..(ell - 1) * p + p];
        let mut db = 0.0;
        let mut skip = false;
        for c in 0..p {
            let dc = xrow[c] - crow[c];
            db += dc * dc;
            if db >= dt[1] {
                skip = true;
                break;
            }
        }
        if skip {
            continue;
        }
        if db >= dt[0] {
            dt[1] = db;
            ic2 = ell;
        } else {
            dt[1] = dt[0];
            ic2 = ic1;
            dt[0] = db;
            ic1 = ell;
        }
    }
    (ic1, ic2)
}

/// 0-based index of the nearest centre to point `i` (ties → smallest index,
/// matching the strict `dd < best` in the C/Python).
#[inline]
fn nearest(x: &[f64], cen: &[f64], p: usize, k: usize, i: usize) -> usize {
    let mut best = f64::INFINITY;
    let mut inew = 0usize;
    let xi = &x[i * p..i * p + p];
    for j in 0..k {
        let cj = &cen[j * p..j * p + p];
        let mut dd = 0.0;
        for c in 0..p {
            let tmp = xi[c] - cj[c];
            dd += tmp * tmp;
        }
        if dd < best {
            best = dd;
            inew = j;
        }
    }
    inew
}

/// Per-cluster within sum-of-squares (`_kmeans_wss`): sequential over points
/// (i-order) and dims (c-order) to match the pure-Python accumulation.
fn wss_of(x: &[f64], cen: &[f64], cl: &[i64], n: usize, p: usize, k: usize) -> Vec<f64> {
    let mut wss = vec![0.0f64; k];
    for i in 0..n {
        let it = (cl[i] - 1) as usize;
        let xi = &x[i * p..i * p + p];
        let ci = &cen[it * p..it * p + p];
        for c in 0..p {
            let tmp = xi[c] - ci[c];
            // R fuses `wss += d*d` to fmadd on arm64; `rfma` mirrors per-arch.
            wss[it] = rfma(tmp, tmp, wss[it]);
        }
    }
    wss
}

/// Recompute centres as cluster means (the Lloyd / MacQueen-init step): zero,
/// accumulate points in i-order, divide. Sequential accumulation (0-ulp).
fn recompute_centres(x: &[f64], cen: &mut [f64], nc: &mut [i64], cl: &[i64], n: usize, p: usize, k: usize) {
    cen.iter_mut().for_each(|v| *v = 0.0);
    nc.iter_mut().for_each(|v| *v = 0);
    for i in 0..n {
        let it = (cl[i] - 1) as usize;
        nc[it] += 1;
        let xi = &x[i * p..i * p + p];
        let ci = &mut cen[it * p..it * p + p];
        for c in 0..p {
            ci[c] += xi[c];
        }
    }
    for j in 0..k {
        let aa = nc[j] as f64;
        for c in 0..p {
            cen[j * p + c] /= aa;
        }
    }
}

#[inline]
fn assign_all(py: Python<'_>, x: &[f64], cen: &[f64], n: usize, p: usize, k: usize) -> Vec<usize> {
    if n >= KM_PAR_MIN {
        py.allow_threads(|| (0..n).into_par_iter().map(|i| nearest(x, cen, p, k, i)).collect())
    } else {
        (0..n).map(|i| nearest(x, cen, p, k, i)).collect()
    }
}

type KmOut<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    i64,
);

fn finish<'py>(py: Python<'py>, cl: Vec<i64>, cen: Vec<f64>, nc: Vec<i64>, wss: Vec<f64>, it: i64) -> KmOut<'py> {
    (
        cl.into_pyarray(py),
        cen.into_pyarray(py),
        nc.into_pyarray(py),
        wss.into_pyarray(py),
        it,
    )
}

/// Lloyd's algorithm. Returns `(cl, cen_flat, nc, wss, iter)`.
#[pyfunction]
#[pyo3(name = "lloyd", signature = (x, centers, k, maxiter))]
pub fn lloyd<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    k: usize,
    maxiter: usize,
) -> KmOut<'py> {
    let xs = x.as_slice().unwrap();
    let n = x.shape()[0];
    let p = x.shape()[1];
    let mut cen = centers.as_slice().unwrap().to_vec();
    let mut cl = vec![-1i64; n];
    let mut nc = vec![0i64; k];
    let mut broke = false;
    let mut iteration = 0usize;
    for it in 0..maxiter {
        iteration = it;
        let newcl = assign_all(py, xs, &cen, n, p, k);
        let mut updated = false;
        for i in 0..n {
            let inew = newcl[i] as i64 + 1;
            if cl[i] != inew {
                updated = true;
                cl[i] = inew;
            }
        }
        if !updated {
            broke = true;
            break;
        }
        recompute_centres(xs, &mut cen, &mut nc, &cl, n, p, k);
    }
    let c_iter = if broke { iteration } else { maxiter };
    let wss = wss_of(xs, &cen, &cl, n, p, k);
    finish(py, cl, cen, nc, wss, c_iter as i64 + 1)
}

/// MacQueen's algorithm. Returns `(cl, cen_flat, nc, wss, iter)`.
#[pyfunction]
#[pyo3(name = "macqueen", signature = (x, centers, k, maxiter))]
pub fn macqueen<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    k: usize,
    maxiter: usize,
) -> KmOut<'py> {
    let xs = x.as_slice().unwrap();
    let n = x.shape()[0];
    let p = x.shape()[1];
    let mut cen = centers.as_slice().unwrap().to_vec();
    let mut nc = vec![0i64; k];

    // initial nearest-centre assignment + centroids
    let init = assign_all(py, xs, &cen, n, p, k);
    let mut cl: Vec<i64> = init.iter().map(|&j| j as i64 + 1).collect();
    recompute_centres(xs, &mut cen, &mut nc, &cl, n, p, k);

    // incremental refinement (inherently sequential: each transfer shifts cen)
    let mut broke = false;
    let mut iteration = 0usize;
    for it in 0..maxiter {
        iteration = it;
        let mut updated = false;
        for i in 0..n {
            let inew = nearest(xs, &cen, p, k, i);
            let iold = (cl[i] - 1) as usize;
            if iold != inew {
                updated = true;
                cl[i] = inew as i64 + 1;
                nc[iold] -= 1;
                nc[inew] += 1;
                let aold = nc[iold] as f64;
                let anew = nc[inew] as f64;
                for c in 0..p {
                    let xic = xs[i * p + c];
                    cen[iold * p + c] += (cen[iold * p + c] - xic) / aold;
                    cen[inew * p + c] += (xic - cen[inew * p + c]) / anew;
                }
            }
        }
        if !updated {
            broke = true;
            break;
        }
    }
    let c_iter = if broke { iteration } else { maxiter };
    let wss = wss_of(xs, &cen, &cl, n, p, k);
    finish(py, cl, cen, nc, wss, c_iter as i64 + 1)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kmns, m)?)?;
    m.add_function(wrap_pyfunction!(lloyd, m)?)?;
    m.add_function(wrap_pyfunction!(macqueen, m)?)?;
    Ok(())
}

// --------------------------------------------------------------------------- #
// Hartigan-Wong (kmns.f) — consolidated here from the former kmns.rs.
// --------------------------------------------------------------------------- #
struct Hw<'a> {
    x: &'a [f64], // m*p, row-major: x(i,j) = x[(i-1)*p + (j-1)]
    m: usize,
    p: usize,
    k: usize,
    cen: Vec<f64>, // k*p, row-major (mutated)
    ic1: Vec<usize>,
    ic2: Vec<usize>,
    d: Vec<f64>,
    nc: Vec<i64>,
    ncp: Vec<i64>,
    an1: Vec<f64>,
    an2: Vec<f64>,
    live: Vec<i64>,
    itran: Vec<i64>,
    imaxqtr: i64,
}

impl<'a> Hw<'a> {
    #[inline]
    fn xv(&self, i: usize, j: usize) -> f64 {
        self.x[(i - 1) * self.p + (j - 1)]
    }
    #[inline]
    fn cv(&self, ell: usize, j: usize) -> f64 {
        self.cen[(ell - 1) * self.p + (j - 1)]
    }
    #[inline]
    fn cset(&mut self, ell: usize, j: usize, v: f64) {
        self.cen[(ell - 1) * self.p + (j - 1)] = v;
    }

    /// OPTRA — optimal-transfer stage. Returns the updated `indx`.
    fn optra(&mut self, mut indx: usize) -> usize {
        let (m, p, k) = (self.m, self.p, self.k);
        for ell in 1..=k {
            if self.itran[ell] == 1 {
                self.live[ell] = (m + 1) as i64;
            }
        }
        for i in 1..=m {
            indx += 1;
            let l1 = self.ic1[i];
            let mut l2 = self.ic2[i];
            let ll = l2;
            if self.nc[l1] != 1 {
                let xb = (i - 1) * p;
                let xrow = &self.x[xb..xb + p];
                if self.ncp[l1] != 0 {
                    let de = sqdist(xrow, &self.cen[(l1 - 1) * p..(l1 - 1) * p + p]);
                    self.d[i] = de * self.an1[l1];
                }
                let da = sqdist(xrow, &self.cen[(l2 - 1) * p..(l2 - 1) * p + p]);
                let mut r2 = da * self.an2[l2];
                for ell in 1..=k {
                    if ((i as i64) >= self.live[l1] && (i as i64) >= self.live[ell])
                        || ell == l1
                        || ell == ll
                    {
                        continue;
                    }
                    let rr = r2 / self.an2[ell];
                    let crow = &self.cen[(ell - 1) * p..(ell - 1) * p + p];
                    let mut dc = 0.0;
                    let mut skip = false;
                    for c in 0..p {
                        let dd = xrow[c] - crow[c];
                        dc += dd * dd;
                        if dc >= rr {
                            skip = true;
                            break;
                        }
                    }
                    if skip {
                        continue;
                    }
                    r2 = dc * self.an2[ell];
                    l2 = ell;
                }
                if r2 >= self.d[i] {
                    self.ic2[i] = l2;
                } else {
                    indx = 0;
                    self.live[l1] = (m + i) as i64;
                    self.live[l2] = (m + i) as i64;
                    self.ncp[l1] = i as i64;
                    self.ncp[l2] = i as i64;
                    let al1 = self.nc[l1] as f64;
                    let alw = al1 - 1.0;
                    let al2 = self.nc[l2] as f64;
                    let alt = al2 + 1.0;
                    let (xb, cb1, cb2) = ((i - 1) * p, (l1 - 1) * p, (l2 - 1) * p);
                    for c in 0..p {
                        let xic = self.x[xb + c];
                        self.cen[cb1 + c] = (self.cen[cb1 + c] * al1 - xic) / alw;
                        self.cen[cb2 + c] = (self.cen[cb2 + c] * al2 + xic) / alt;
                    }
                    self.nc[l1] -= 1;
                    self.nc[l2] += 1;
                    self.an2[l1] = alw / al1;
                    self.an1[l1] = BIG;
                    if alw > 1.0 {
                        self.an1[l1] = alw / (alw - 1.0);
                    }
                    self.an1[l2] = alt / al2;
                    self.an2[l2] = alt / (alt + 1.0);
                    self.ic1[i] = l2;
                    self.ic2[i] = l1;
                }
            }
            if indx == m {
                return indx;
            }
        }
        for ell in 1..=k {
            self.itran[ell] = 0;
            self.live[ell] -= m as i64;
        }
        indx
    }

    /// QTRAN — quick-transfer stage. Returns the updated `indx`.
    fn qtran(&mut self, mut indx: usize) -> usize {
        let (m, p) = (self.m, self.p);
        let mut icoun: usize = 0;
        let mut istep: i64 = 0;
        loop {
            for i in 1..=m {
                icoun += 1;
                istep += 1;
                if istep >= self.imaxqtr {
                    self.imaxqtr = -1;
                    return indx;
                }
                let l1 = self.ic1[i];
                let l2 = self.ic2[i];
                if self.nc[l1] != 1 {
                    let xb = (i - 1) * p;
                    let xrow = &self.x[xb..xb + p];
                    if istep <= self.ncp[l1] {
                        let da = sqdist(xrow, &self.cen[(l1 - 1) * p..(l1 - 1) * p + p]);
                        self.d[i] = da * self.an1[l1];
                    }
                    if istep < self.ncp[l1] || istep < self.ncp[l2] {
                        let r2 = self.d[i] / self.an2[l2];
                        let crow = &self.cen[(l2 - 1) * p..(l2 - 1) * p + p];
                        let mut dd = 0.0;
                        let mut skip = false;
                        for c in 0..p {
                            let de = xrow[c] - crow[c];
                            dd += de * de;
                            if dd >= r2 {
                                skip = true;
                                break;
                            }
                        }
                        if !skip {
                            icoun = 0;
                            indx = 0;
                            self.itran[l1] = 1;
                            self.itran[l2] = 1;
                            self.ncp[l1] = istep + m as i64;
                            self.ncp[l2] = istep + m as i64;
                            let al1 = self.nc[l1] as f64;
                            let alw = al1 - 1.0;
                            let al2 = self.nc[l2] as f64;
                            let alt = al2 + 1.0;
                            let (xb, cb1, cb2) = ((i - 1) * p, (l1 - 1) * p, (l2 - 1) * p);
                            for c in 0..p {
                                let xic = self.x[xb + c];
                                self.cen[cb1 + c] = (self.cen[cb1 + c] * al1 - xic) / alw;
                                self.cen[cb2 + c] = (self.cen[cb2 + c] * al2 + xic) / alt;
                            }
                            self.nc[l1] -= 1;
                            self.nc[l2] += 1;
                            self.an2[l1] = alw / al1;
                            self.an1[l1] = BIG;
                            if alw > 1.0 {
                                self.an1[l1] = alw / (alw - 1.0);
                            }
                            self.an1[l2] = alt / al2;
                            self.an2[l2] = alt / (alt + 1.0);
                            self.ic1[i] = l2;
                            self.ic2[i] = l1;
                        }
                    }
                }
                if icoun == m {
                    return indx;
                }
            }
            // GO TO 10: repeat the sweep
        }
    }
}

/// Hartigan-Wong k-means for data `x` (`m x p`) and initial `centers` (`k x p`).
#[pyfunction]
#[pyo3(name = "kmns", signature = (x, centers, k, iter_max))]
#[allow(clippy::type_complexity)]
pub fn kmns<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    k: usize,
    iter_max: usize,
) -> (
    i64,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    i64,
) {
    let xs = x.as_slice().unwrap();
    let m = x.shape()[0];
    let p = x.shape()[1];
    let cen0 = centers.as_slice().unwrap();

    let empty_i = || Vec::<i64>::new().into_pyarray(py);
    let empty_f = || Vec::<f64>::new().into_pyarray(py);

    if k <= 1 || k >= m {
        return (3, empty_i(), empty_f(), empty_i(), empty_f(), 0);
    }

    let (ifault, cluster, cen_flat, nc_out, wss, iter_ret) = py.allow_threads(|| {
        let mut hw = Hw {
            x: xs,
            m,
            p,
            k,
            cen: cen0.to_vec(),
            ic1: vec![0usize; m + 1],
            ic2: vec![0usize; m + 1],
            d: vec![0.0f64; m + 1],
            nc: vec![0i64; k + 1],
            ncp: vec![0i64; k + 1],
            an1: vec![0.0f64; k + 1],
            an2: vec![0.0f64; k + 1],
            live: vec![0i64; k + 1],
            itran: vec![0i64; k + 1],
            imaxqtr: (50 * m as i64).min(2147483647),
        };

        // two closest centres IC1, IC2 for each point. Independent per point, so
        // the map is rayon-parallel (the one parallelizable HW phase; OPTRA/QTRAN
        // below are inherently serial). Parallel == serial bit-for-bit (0-ulp).
        let pairs: Vec<(usize, usize)> = if m >= KM_PAR_MIN {
            (0..m)
                .into_par_iter()
                .map(|i0| two_closest(xs, &hw.cen, p, k, i0))
                .collect()
        } else {
            (0..m).map(|i0| two_closest(xs, &hw.cen, p, k, i0)).collect()
        };
        for i in 1..=m {
            hw.ic1[i] = pairs[i - 1].0;
            hw.ic2[i] = pairs[i - 1].1;
        }

        // update centres to cluster means; sizes NC; an1/an2
        for ell in 1..=k {
            hw.nc[ell] = 0;
            for j in 1..=p {
                hw.cset(ell, j, 0.0);
            }
        }
        for i in 1..=m {
            let ell = hw.ic1[i];
            hw.nc[ell] += 1;
            for j in 1..=p {
                let v = hw.cv(ell, j) + hw.xv(i, j);
                hw.cset(ell, j, v);
            }
        }
        for ell in 1..=k {
            if hw.nc[ell] == 0 {
                return (1i64, Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0i64);
            }
            let aa = hw.nc[ell] as f64;
            for j in 1..=p {
                let v = hw.cv(ell, j) / aa;
                hw.cset(ell, j, v);
            }
            hw.an2[ell] = aa / (aa + 1.0);
            hw.an1[ell] = BIG;
            if aa > 1.0 {
                hw.an1[ell] = aa / (aa - 1.0);
            }
            hw.itran[ell] = 1;
            hw.ncp[ell] = -1;
        }

        // OPTRA / QTRAN iterations
        let mut indx = 0usize;
        let mut iter_ret = iter_max + 1;
        let mut ifault = 2i64;
        for ij in 1..=iter_max {
            indx = hw.optra(indx);
            if indx == m {
                iter_ret = ij;
                ifault = 0;
                break;
            }
            indx = hw.qtran(indx);
            if hw.imaxqtr < 0 {
                ifault = 4;
                iter_ret = ij;
                break;
            }
            if k == 2 {
                iter_ret = ij;
                ifault = 0;
                break;
            }
            for ell in 1..=k {
                hw.ncp[ell] = 0;
            }
        }

        // within-cluster sum of squares (recompute centres as the means)
        let mut wss = vec![0.0f64; k + 1];
        for ell in 1..=k {
            for j in 1..=p {
                hw.cset(ell, j, 0.0);
            }
        }
        for i in 1..=m {
            let ii = hw.ic1[i];
            for j in 1..=p {
                let v = hw.cv(ii, j) + hw.xv(i, j);
                hw.cset(ii, j, v);
            }
        }
        for j in 1..=p {
            for ell in 1..=k {
                let v = hw.cv(ell, j) / (hw.nc[ell] as f64);
                hw.cset(ell, j, v);
            }
            for i in 1..=m {
                let ii = hw.ic1[i];
                let da = hw.xv(i, j) - hw.cv(ii, j);
                wss[ii] = rfma(da, da, wss[ii]);
            }
        }

        let cluster: Vec<i64> = (1..=m).map(|i| hw.ic1[i] as i64).collect();
        let nc_out: Vec<i64> = hw.nc[1..=k].to_vec();
        let wss_out: Vec<f64> = wss[1..=k].to_vec();
        (ifault, cluster, hw.cen, nc_out, wss_out, iter_ret as i64)
    });

    (
        ifault,
        cluster.into_pyarray(py),
        cen_flat.into_pyarray(py),
        nc_out.into_pyarray(py),
        wss.into_pyarray(py),
        iter_ret,
    )
}
