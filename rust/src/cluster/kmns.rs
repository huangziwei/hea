//! k-means, Hartigan-Wong — the OPTRA/QTRAN transfer loops of `kmeans()`.
//!
//! A line-by-line mirror of `hea/R/clustering.py::_kmns` (itself a 1:1 port of
//! R's `src/library/stats/src/kmns.f` — `KMNS` + `OPTRA` + `QTRAN`). The optimal-
//! and quick-transfer stages mutate the assignment/centre state in place with
//! data-dependent control flow, so this is INHERENTLY SERIAL — never parallelize
//! (like `rng/mt.rs` / `linalg/chol.rs`). All bookkeeping arrays are 1-based
//! (index 0 unused), matching the Fortran/Python.
//!
//! Returns `(ifault, cluster, centers_flat, nc, wss, iter)`; the Python seam
//! reshapes `centers_flat` to `(k, p)` and assembles the result dict. `ifault`
//! 1 (empty cluster) / 3 (k out of range) come back with empty arrays, exactly
//! as `_kmns` returns `{"ifault": ...}`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

const BIG: f64 = 1.0e30;

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
                if self.ncp[l1] != 0 {
                    let mut de = 0.0;
                    for j in 1..=p {
                        let df = self.xv(i, j) - self.cv(l1, j);
                        de += df * df;
                    }
                    self.d[i] = de * self.an1[l1];
                }
                let mut da = 0.0;
                for j in 1..=p {
                    let db = self.xv(i, j) - self.cv(l2, j);
                    da += db * db;
                }
                let mut r2 = da * self.an2[l2];
                for ell in 1..=k {
                    if ((i as i64) >= self.live[l1] && (i as i64) >= self.live[ell])
                        || ell == l1
                        || ell == ll
                    {
                        continue;
                    }
                    let rr = r2 / self.an2[ell];
                    let mut dc = 0.0;
                    let mut skip = false;
                    for j in 1..=p {
                        let dd = self.xv(i, j) - self.cv(ell, j);
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
                    for j in 1..=p {
                        let c1 = (self.cv(l1, j) * al1 - self.xv(i, j)) / alw;
                        let c2 = (self.cv(l2, j) * al2 + self.xv(i, j)) / alt;
                        self.cset(l1, j, c1);
                        self.cset(l2, j, c2);
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
                    if istep <= self.ncp[l1] {
                        let mut da = 0.0;
                        for j in 1..=p {
                            let db = self.xv(i, j) - self.cv(l1, j);
                            da += db * db;
                        }
                        self.d[i] = da * self.an1[l1];
                    }
                    if istep < self.ncp[l1] || istep < self.ncp[l2] {
                        let r2 = self.d[i] / self.an2[l2];
                        let mut dd = 0.0;
                        let mut skip = false;
                        for j in 1..=p {
                            let de = self.xv(i, j) - self.cv(l2, j);
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
                            for j in 1..=p {
                                let c1 = (self.cv(l1, j) * al1 - self.xv(i, j)) / alw;
                                let c2 = (self.cv(l2, j) * al2 + self.xv(i, j)) / alt;
                                self.cset(l1, j, c1);
                                self.cset(l2, j, c2);
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

        // two closest centres IC1, IC2 for each point
        for i in 1..=m {
            hw.ic1[i] = 1;
            hw.ic2[i] = 2;
            let mut dt = [0.0f64; 2];
            for il in 1..=2usize {
                dt[il - 1] = 0.0;
                for j in 1..=p {
                    let da = hw.xv(i, j) - hw.cv(il, j);
                    dt[il - 1] += da * da;
                }
            }
            if dt[0] > dt[1] {
                hw.ic1[i] = 2;
                hw.ic2[i] = 1;
                dt.swap(0, 1);
            }
            for ell in 3..=k {
                let mut db = 0.0;
                let mut skip = false;
                for j in 1..=p {
                    let dc = hw.xv(i, j) - hw.cv(ell, j);
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
                    hw.ic2[i] = ell;
                } else {
                    dt[1] = dt[0];
                    hw.ic2[i] = hw.ic1[i];
                    dt[0] = db;
                    hw.ic1[i] = ell;
                }
            }
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
                wss[ii] += da * da;
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kmns, m)?)?;
    Ok(())
}
