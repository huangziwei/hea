//! Dennis-Schnabel UNCMIN (`R nlm`'s optimizer) — mirrors
//! `hea/R/uncmin.py` (the spec and test oracle) line by line, which is
//! itself a port of R 4.6.0 `src/appl/uncmin.c` with the clang
//! `-ffp-contract` fusions of R-as-built mirrored via `rfma` (see the
//! Python module docstring for the disassembly receipts and the ctypes
//! oracle validation against R's compiled `optif9`).
//!
//! Matrices are flat column-major `a[i + j*n]` (the f2c layout). The
//! objective is a `PyResult`-returning trait so Python exceptions from
//! the callbacks propagate out of the driver exactly like R errors.

use pyo3::prelude::*;

use super::linpack::{ddot, dnrm2, dscal, dtrsl};
use crate::nmath::util::rfma;

const DBL_EPSILON: f64 = f64::EPSILON;
const DBL_MAX: f64 = f64::MAX;

pub trait Obj {
    fn fcn(&mut self, x: &[f64]) -> PyResult<f64>;
    fn d1fcn(&mut self, x: &[f64], g: &mut [f64]) -> PyResult<()>;
    fn d2fcn(&mut self, x: &[f64], a: &mut [f64], nr: usize) -> PyResult<()>;
}

fn fmax2(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

fn fmin2(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

/// uncmin.c `fdhess` (:50) — used by `nlm(hessian=TRUE)`.
pub fn fdhess(
    n: usize,
    x: &mut [f64],
    fval: f64,
    obj: &mut dyn Obj,
    h: &mut [f64],
    nfd: usize,
    ndigit: i32,
    typx: &[f64],
) -> PyResult<()> {
    let mut step = vec![0.0; n];
    let mut f = vec![0.0; n];
    let eta = 10f64.powf(-(ndigit as f64) / 3.0);
    for i in 0..n {
        step[i] = eta * fmax2(x[i], typx[i]);
        if typx[i] < 0.0 {
            step[i] = -step[i];
        }
        let tempi = x[i];
        x[i] = tempi + step[i];
        step[i] = x[i] - tempi;
        f[i] = obj.fcn(x)?;
        x[i] = tempi;
    }
    for i in 0..n {
        let tempi = x[i];
        x[i] = tempi + step[i] * 2.0;
        let fii = obj.fcn(x)?;
        h[i + i * nfd] = ((fval - f[i]) + (fii - f[i])) / (step[i] * step[i]);
        x[i] = tempi + step[i];
        for j in (i + 1)..n {
            let tempj = x[j];
            x[j] = tempj + step[j];
            let fij = obj.fcn(x)?;
            h[i + j * nfd] = ((fval - f[i]) + (fij - f[j])) / (step[i] * step[j]);
            x[j] = tempj;
        }
        x[i] = tempi;
    }
    Ok(())
}

/// uncmin.c `mvmltl` (:132): y = L x (clang-contracted accumulation).
fn mvmltl(n: usize, a: &[f64], x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..=i {
            s = rfma(a[i + j * n], x[j], s);
        }
        y[i] = s;
    }
}

/// uncmin.c `mvmltu` (:160): y = L' x — F77 ddot down each column.
fn mvmltu(n: usize, a: &[f64], x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        y[i] = ddot(n - i, a, i + i * n, 1, x, i, 1);
    }
}

/// uncmin.c `mvmlts` (:184): y = A x, symmetric A in the lower triangle.
fn mvmlts(n: usize, a: &[f64], x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..=i {
            s = rfma(a[i + j * n], x[j], s);
        }
        for j in (i + 1)..n {
            s = rfma(a[j + i * n], x[j], s);
        }
        y[i] = s;
    }
}

/// uncmin.c `lltslv` (:215): solve LL'x = b (dtrsl jobs 0 then 10).
fn lltslv(n: usize, a: &[f64], x: &mut [f64], b: &[f64]) {
    x[..n].copy_from_slice(&b[..n]);
    dtrsl(a, 0, n, n, x, 0, 1, 0);
    dtrsl(a, 0, n, n, x, 0, 1, 10);
}

/// uncmin.c `choldc` (:242): perturbed Cholesky of a+D; returns addmax.
fn choldc(n: usize, a: &mut [f64], diagmx: f64, tol: f64) -> f64 {
    let mut addmax = 0.0;
    let aminl = (diagmx * tol).sqrt();
    let amnlsq = aminl * aminl;
    for i in 0..n {
        for j in 0..i {
            let mut s = 0.0;
            for k in 0..j {
                s = rfma(a[i + k * n], a[j + k * n], s);
            }
            a[i + j * n] = (a[i + j * n] - s) / a[j + j * n];
        }
        let mut s = 0.0;
        for k in 0..i {
            s = rfma(a[i + k * n], a[i + k * n], s);
        }
        let tmp1 = a[i + i * n] - s;
        if tmp1 >= amnlsq {
            a[i + i * n] = tmp1.sqrt();
        } else {
            let mut offmax = 0.0;
            for j in 0..i {
                let tmp2 = a[i + j * n].abs();
                if offmax < tmp2 {
                    offmax = tmp2;
                }
            }
            if offmax <= amnlsq {
                offmax = amnlsq;
            }
            a[i + i * n] = offmax.sqrt();
            let tmp2 = offmax - tmp1;
            if addmax < tmp2 {
                addmax = tmp2;
            }
        }
    }
    addmax
}

/// uncmin.c `qraux1` (:323): swap rows i, i+1 over columns i..n-1.
fn qraux1(n: usize, r: &mut [f64], i: usize) {
    for j in i..n {
        r.swap(i + j * n, i + 1 + j * n);
    }
}

/// uncmin.c `qraux2` (:347): Jacobi rotation, first product fused.
fn qraux2(n: usize, r: &mut [f64], i: usize, a: f64, b: f64) {
    let den = a.hypot(b);
    let c = a / den;
    let s = b / den;
    for j in i..n {
        let y = r[i + j * n];
        let z = r[i + 1 + j * n];
        r[i + j * n] = rfma(c, y, -(s * z));
        r[i + 1 + j * n] = rfma(s, y, c * z);
    }
}

/// uncmin.c `qrupdt` (:382): rank-1 QR update.
fn qrupdt(n: usize, a: &mut [f64], u: &mut [f64], v: &[f64]) {
    let mut k = n - 1;
    while k > 0 && u[k] == 0.0 {
        k -= 1;
    }
    let mut ii = k;
    while ii > 0 {
        let i = ii - 1;
        if u[i] == 0.0 {
            qraux1(n, a, i);
            u[i] = u[ii];
        } else {
            qraux2(n, a, i, u[i], -u[ii]);
            u[i] = u[i].hypot(u[ii]);
        }
        ii = i;
    }
    for j in 0..n {
        a[j * n] = rfma(u[0], v[j], a[j * n]);
    }
    for i in 0..k {
        if a[i + i * n] == 0.0 {
            qraux1(n, a, i);
        } else {
            let t1 = a[i + i * n];
            let t2 = -a[i + 1 + i * n];
            qraux2(n, a, i, t1, t2);
        }
    }
}

/// uncmin.c `tregup` (:444): trust-region accept/update (methods 2-3,
/// unreachable from R's nlm; kept in plain reference order like the
/// Python spec). Returns (dlt, iretcd, fplsp, fpls, mxtake).
#[allow(clippy::too_many_arguments)]
fn tregup(
    n: usize,
    x: &[f64],
    f: f64,
    g: &[f64],
    a: &[f64],
    obj: &mut dyn Obj,
    sc: &[f64],
    sx: &[f64],
    nwtake: bool,
    stepmx: f64,
    steptl: f64,
    mut dlt: f64,
    mut iretcd: i32,
    xplsp: &mut [f64],
    mut fplsp: f64,
    xpls: &mut [f64],
    method: i32,
    udiag: &[f64],
) -> PyResult<(f64, i32, f64, f64, bool)> {
    let mut mxtake = false;
    for i in 0..n {
        xpls[i] = x[i] + sc[i];
    }
    let mut fpls = obj.fcn(xpls)?;
    let dltf = fpls - f;
    let slp = ddot(n, g, 0, 1, sc, 0, 1);
    if iretcd == 3 && (fpls >= fplsp || dltf > slp * 1e-4) {
        iretcd = 0;
        xpls[..n].copy_from_slice(&xplsp[..n]);
        fpls = fplsp;
        dlt *= 0.5;
    } else if dltf > slp * 1e-4 {
        let mut rln = 0.0;
        for i in 0..n {
            let temp1 = sc[i].abs() / fmax2(xpls[i].abs(), 1.0 / sx[i]);
            if rln < temp1 {
                rln = temp1;
            }
        }
        if rln < steptl {
            iretcd = 1;
        } else {
            iretcd = 2;
            let dltmp = -slp * dlt / ((dltf - slp) * 2.0);
            if dltmp < dlt * 0.1 {
                dlt *= 0.1;
            } else {
                dlt = dltmp;
            }
        }
    } else {
        let mut dltfp = 0.0;
        if method == 2 {
            for i in 0..n {
                let mut temp1 = 0.0;
                for j in i..n {
                    temp1 += a[j + i * n] * sc[j];
                }
                dltfp += temp1 * temp1;
            }
        } else {
            for i in 0..n {
                dltfp += udiag[i] * sc[i] * sc[i];
                let mut temp1 = 0.0;
                for j in (i + 1)..n {
                    temp1 += a[i + j * n] * sc[i] * sc[j];
                }
                dltfp += temp1 * 2.0;
            }
        }
        dltfp = slp + dltfp / 2.0;
        if iretcd != 2 && (dltfp - dltf).abs() <= dltf.abs() * 0.1 && nwtake && dlt <= stepmx * 0.99
        {
            iretcd = 3;
            xplsp[..n].copy_from_slice(&xpls[..n]);
            fplsp = fpls;
            dlt = fmin2(dlt * 2.0, stepmx);
        } else {
            iretcd = 0;
            if dlt > stepmx * 0.99 {
                mxtake = true;
            }
            if dltf >= dltfp * 0.1 {
                dlt *= 0.5;
            } else if dltf <= dltfp * 0.75 {
                dlt = fmin2(dlt * 2.0, stepmx);
            }
        }
    }
    Ok((dlt, iretcd, fplsp, fpls, mxtake))
}

/// uncmin.c `lnsrch` (:614): backtracking line search (method 1) with
/// the R-as-built fusions. Returns (fpls, iretcd, mxtake).
#[allow(clippy::too_many_arguments)]
fn lnsrch(
    n: usize,
    x: &[f64],
    f: f64,
    g: &[f64],
    p: &mut [f64],
    xpls: &mut [f64],
    obj: &mut dyn Obj,
    stepmx: f64,
    steptl: f64,
    sx: &[f64],
) -> PyResult<(f64, i32, bool)> {
    let mut firstback = true;
    let mut pfpls = 0.0;
    let mut plmbda = 0.0;
    let mut temp1 = 0.0;
    for i in 0..n {
        temp1 = rfma(sx[i] * sx[i] * p[i], p[i], temp1);
    }
    let mut sln = temp1.sqrt();
    if sln > stepmx {
        dscal(n, stepmx / sln, p, 0);
        sln = stepmx;
    }
    let slp = ddot(n, g, 0, 1, p, 0, 1);
    let mut rln = 0.0;
    for i in 0..n {
        let t = p[i].abs() / fmax2(x[i].abs(), 1.0 / sx[i]);
        if rln < t {
            rln = t;
        }
    }
    let rmnlmb = steptl / rln;
    let mut lam = 1.0;
    let mut mxtake = false;
    let mut iretcd = 2;
    let mut fpls;
    let mut tlmbda = 0.0;
    loop {
        for i in 0..n {
            xpls[i] = rfma(lam, p[i], x[i]);
        }
        fpls = obj.fcn(xpls)?;
        if fpls <= rfma(slp * 1e-4, lam, f) {
            iretcd = 0;
            if lam == 1.0 && sln > stepmx * 0.99 {
                mxtake = true;
            }
            return Ok((fpls, iretcd, mxtake));
        }
        if lam < rmnlmb {
            iretcd = 1;
            return Ok((fpls, iretcd, mxtake));
        }
        if fpls >= DBL_MAX {
            lam *= 0.1;
            firstback = true;
        } else {
            if firstback {
                tlmbda = -lam * slp / ((fpls - f - slp) * 2.0);
                firstback = false;
            } else {
                let t1 = rfma(-lam, slp, fpls - f);
                let t2 = rfma(-plmbda, slp, pfpls - f);
                let t3 = 1.0 / (lam - plmbda);
                let a3 = 3.0 * t3 * (t1 / (lam * lam) - t2 / (plmbda * plmbda));
                let b = t3 * (t2 * lam / (plmbda * plmbda) - t1 * plmbda / (lam * lam));
                let disc = rfma(b, b, -(a3 * slp));
                if disc > b * b {
                    tlmbda = (-b + if a3 < 0.0 { -disc.sqrt() } else { disc.sqrt() }) / a3;
                } else {
                    tlmbda = (-b + if a3 < 0.0 { disc.sqrt() } else { -disc.sqrt() }) / a3;
                }
                if tlmbda > lam * 0.5 {
                    tlmbda = lam * 0.5;
                }
            }
            plmbda = lam;
            pfpls = fpls;
            if tlmbda < lam * 0.1 {
                lam *= 0.1;
            } else {
                lam = tlmbda;
            }
        }
        if iretcd <= 1 {
            return Ok((fpls, iretcd, mxtake));
        }
    }
}

/// uncmin.c `dog_1step` (:742) — method 2, plain reference order.
#[allow(clippy::too_many_arguments)]
fn dog_1step(
    n: usize,
    g: &[f64],
    a: &[f64],
    p: &[f64],
    sx: &[f64],
    rnwtln: f64,
    mut dlt: f64,
    mut fstdog: bool,
    ssd: &mut [f64],
    v: &mut [f64],
    mut cln: f64,
    mut eta: f64,
    sc: &mut [f64],
    stepmx: f64,
) -> (f64, bool, bool, f64, f64) {
    let nwtake = rnwtln <= dlt;
    if nwtake {
        sc[..n].copy_from_slice(&p[..n]);
        dlt = rnwtln;
        return (dlt, nwtake, fstdog, cln, eta);
    }
    if fstdog {
        fstdog = false;
        let mut alpha = 0.0;
        for i in 0..n {
            alpha += g[i] * g[i] / (sx[i] * sx[i]);
        }
        let mut bet = 0.0;
        for i in 0..n {
            let mut tmp = 0.0;
            for j in i..n {
                tmp += a[j + i * n] * g[j] / (sx[j] * sx[j]);
            }
            bet += tmp * tmp;
        }
        for i in 0..n {
            ssd[i] = -(alpha / bet) * g[i] / sx[i];
        }
        cln = alpha * alpha.sqrt() / bet;
        eta = 0.8 * alpha * alpha / (-bet * ddot(n, g, 0, 1, p, 0, 1)) + 0.2;
        for i in 0..n {
            v[i] = eta * sx[i] * p[i] - ssd[i];
        }
        if dlt == -1.0 {
            dlt = fmin2(cln, stepmx);
        }
    }
    if eta * rnwtln <= dlt {
        for i in 0..n {
            sc[i] = dlt / rnwtln * p[i];
        }
    } else if cln >= dlt {
        for i in 0..n {
            sc[i] = dlt / cln * ssd[i] / sx[i];
        }
    } else {
        let dot1 = ddot(n, v, 0, 1, ssd, 0, 1);
        let dot2 = ddot(n, v, 0, 1, v, 0, 1);
        let alam = (-dot1 + (dot1 * dot1 - dot2 * (cln * cln - dlt * dlt)).sqrt()) / dot2;
        for i in 0..n {
            sc[i] = (ssd[i] + alam * v[i]) / sx[i];
        }
    }
    (dlt, nwtake, fstdog, cln, eta)
}

/// uncmin.c `dogdrv` (:840) — method 2 driver.
#[allow(clippy::too_many_arguments)]
fn dogdrv(
    n: usize,
    x: &[f64],
    f: f64,
    g: &[f64],
    a: &[f64],
    p: &[f64],
    xpls: &mut [f64],
    obj: &mut dyn Obj,
    sx: &[f64],
    stepmx: f64,
    steptl: f64,
    mut dlt: f64,
) -> PyResult<(f64, f64, i32, bool)> {
    let mut ssd = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut xplsp = vec![0.0; n];
    let mut sc = vec![0.0; n];
    let mut fplsp = 0.0;
    let mut cln = 0.0;
    let mut eta = 0.0;
    let mut tmp = 0.0;
    for i in 0..n {
        tmp += sx[i] * sx[i] * p[i] * p[i];
    }
    let rnwtln = tmp.sqrt();
    let mut iretcd = 4;
    let mut fstdog = true;
    let mut fpls = 0.0;
    let mut mxtake = false;
    while iretcd > 1 {
        let nwtake;
        (dlt, nwtake, fstdog, cln, eta) = dog_1step(
            n, g, a, p, sx, rnwtln, dlt, fstdog, &mut ssd, &mut v, cln, eta, &mut sc, stepmx,
        );
        let r = tregup(
            n, x, f, g, a, obj, &sc, sx, nwtake, stepmx, steptl, dlt, iretcd, &mut xplsp, fplsp,
            xpls, 2, &ssd,
        )?;
        (dlt, iretcd, fplsp, fpls, mxtake) = r;
    }
    Ok((fpls, dlt, iretcd, mxtake))
}

/// uncmin.c `hook_1step` (:908) — method 3, plain reference order.
#[allow(clippy::too_many_arguments)]
fn hook_1step(
    n: usize,
    g: &[f64],
    a: &mut [f64],
    udiag: &[f64],
    p: &[f64],
    sx: &[f64],
    rnwtln: f64,
    mut dlt: f64,
    mut amu: f64,
    dltp: f64,
    mut phi: f64,
    mut phip0: f64,
    mut fstime: bool,
    sc: &mut [f64],
    wrk0: &mut [f64],
    epsm: f64,
) -> (f64, f64, f64, f64, bool, bool) {
    let hi = 1.5;
    let alo = 0.75;
    let nwtake = rnwtln <= hi * dlt;
    if nwtake {
        sc[..n].copy_from_slice(&p[..n]);
        dlt = fmin2(dlt, rnwtln);
        amu = 0.0;
        return (dlt, amu, phi, phip0, fstime, nwtake);
    }
    if amu > 0.0 {
        amu -= (phi + dltp) * (dltp - dlt + phi) / (dlt * phip0);
    }
    phi = rnwtln - dlt;
    if fstime {
        for i in 0..n {
            wrk0[i] = sx[i] * sx[i] * p[i];
        }
        dtrsl(a, 0, n, n, wrk0, 0, 1, 0);
        let temp1 = dnrm2(n, wrk0, 0);
        phip0 = -(temp1 * temp1) / rnwtln;
        fstime = false;
    }
    let mut phip = phip0;
    let mut amulo = -phi / phip;
    let mut amuup = 0.0;
    for i in 0..n {
        amuup += g[i] * g[i] / (sx[i] * sx[i]);
    }
    amuup = amuup.sqrt() / dlt;
    loop {
        if amu < amulo || amu > amuup {
            amu = fmax2((amulo * amuup).sqrt(), amuup * 0.001);
        }
        for i in 0..n {
            a[i + i * n] = udiag[i] + amu * sx[i] * sx[i];
            for j in 0..i {
                a[i + j * n] = a[j + i * n];
            }
        }
        choldc(n, a, 0.0, epsm.sqrt());
        for i in 0..n {
            wrk0[i] = -g[i];
        }
        {
            let b: Vec<f64> = wrk0[..n].to_vec();
            lltslv(n, a, sc, &b);
        }
        let mut stepln = 0.0;
        for i in 0..n {
            stepln += sx[i] * sx[i] * sc[i] * sc[i];
        }
        let stepln = stepln.sqrt();
        phi = stepln - dlt;
        for i in 0..n {
            wrk0[i] = sx[i] * sx[i] * sc[i];
        }
        dtrsl(a, 0, n, n, wrk0, 0, 1, 0);
        let temp1 = dnrm2(n, wrk0, 0);
        phip = -(temp1 * temp1) / stepln;
        if (alo * dlt <= stepln && stepln <= hi * dlt) || (amuup - amulo > 0.0) {
            break;
        }
        let temp1 = (amu - phi) / phip;
        amulo = fmax2(amulo, temp1);
        if phi < 0.0 {
            amuup = fmin2(amuup, amu);
        }
        amu -= stepln * phi / (dlt * phip);
    }
    (dlt, amu, phi, phip0, fstime, nwtake)
}

/// uncmin.c `hookdrv` (:1047) — method 3 driver.
#[allow(clippy::too_many_arguments)]
fn hookdrv(
    n: usize,
    x: &[f64],
    f: f64,
    g: &[f64],
    a: &mut [f64],
    udiag: &[f64],
    p: &[f64],
    xpls: &mut [f64],
    obj: &mut dyn Obj,
    sx: &[f64],
    stepmx: f64,
    steptl: f64,
    mut dlt: f64,
    mut amu: f64,
    mut dltp: f64,
    mut phi: f64,
    mut phip0: f64,
    epsm: f64,
    itncnt: i32,
) -> PyResult<(f64, f64, i32, bool, f64, f64, f64, f64)> {
    let mut sc = vec![0.0; n];
    let mut xplsp = vec![0.0; n];
    let mut wrk0 = vec![0.0; n];
    let mut fplsp = 0.0;
    let mut tmp = 0.0;
    for i in 0..n {
        tmp += sx[i] * sx[i] * p[i] * p[i];
    }
    let rnwtln = tmp.sqrt();
    if itncnt == 1 {
        amu = 0.0;
        if dlt == -1.0 {
            let mut alpha = 0.0;
            for i in 0..n {
                alpha += g[i] * g[i] / (sx[i] * sx[i]);
            }
            let mut bet = 0.0;
            for i in 0..n {
                let mut t = 0.0;
                for j in i..n {
                    t += a[j + i * n] * g[j] / (sx[j] * sx[j]);
                }
                bet += t * t;
            }
            dlt = alpha * alpha.sqrt() / bet;
            if dlt > stepmx {
                dlt = stepmx;
            }
        }
    }
    let mut iretcd = 4;
    let mut fstime = true;
    let mut fpls = 0.0;
    let mut mxtake = false;
    while iretcd > 1 {
        let nwtake;
        (dlt, amu, phi, phip0, fstime, nwtake) = hook_1step(
            n, g, a, udiag, p, sx, rnwtln, dlt, amu, dltp, phi, phip0, fstime, &mut sc, &mut wrk0,
            epsm,
        );
        dltp = dlt;
        let r = tregup(
            n, x, f, g, a, obj, &sc, sx, nwtake, stepmx, steptl, dlt, iretcd, &mut xplsp, fplsp,
            xpls, 3, udiag,
        )?;
        (dlt, iretcd, fplsp, fpls, mxtake) = r;
    }
    Ok((fpls, dlt, iretcd, mxtake, amu, dltp, phi, phip0))
}

/// uncmin.c `secunf` (:1147): unfactored BFGS update (method 3).
#[allow(clippy::too_many_arguments)]
fn secunf(
    n: usize,
    x: &[f64],
    g: &[f64],
    a: &mut [f64],
    udiag: &[f64],
    xpls: &[f64],
    gpls: &[f64],
    epsm: f64,
    itncnt: i32,
    rnf: f64,
    iagflg: i32,
    mut noupdt: bool,
) -> bool {
    let mut s = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut t = vec![0.0; n];
    for i in 0..n {
        a[i + i * n] = udiag[i];
        for j in 0..i {
            a[i + j * n] = a[j + i * n];
        }
    }
    noupdt = itncnt == 1;
    for i in 0..n {
        s[i] = xpls[i] - x[i];
        y[i] = gpls[i] - g[i];
    }
    let den1 = ddot(n, &s, 0, 1, &y, 0, 1);
    let snorm2 = dnrm2(n, &s, 0);
    let ynrm2 = dnrm2(n, &y, 0);
    if den1 < epsm.sqrt() * snorm2 * ynrm2 {
        return noupdt;
    }
    mvmlts(n, a, &s, &mut t);
    let mut den2 = ddot(n, &s, 0, 1, &t, 0, 1);
    if noupdt {
        let gam = den1 / den2;
        den2 *= gam;
        for j in 0..n {
            t[j] *= gam;
            for i in j..n {
                a[i + j * n] *= gam;
            }
        }
        noupdt = false;
    }
    let mut skpupd = true;
    for i in 0..n {
        let mut tol = rnf * fmax2(g[i].abs(), gpls[i].abs());
        if iagflg == 0 {
            tol /= rnf.sqrt();
        }
        if (y[i] - t[i]).abs() >= tol {
            skpupd = false;
            break;
        }
    }
    if skpupd {
        return noupdt;
    }
    for j in 0..n {
        for i in j..n {
            a[i + j * n] += y[i] * y[j] / den1 - t[i] * t[j] / den2;
        }
    }
    noupdt
}

/// uncmin.c `secfac` (:1241): factored BFGS update (methods 1-2).
#[allow(clippy::too_many_arguments)]
fn secfac(
    n: usize,
    x: &[f64],
    g: &[f64],
    a: &mut [f64],
    xpls: &[f64],
    gpls: &[f64],
    epsm: f64,
    itncnt: i32,
    rnf: f64,
    iagflg: i32,
    mut noupdt: bool,
) -> bool {
    let mut s = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut w = vec![0.0; n];
    noupdt = itncnt == 1;
    for i in 0..n {
        s[i] = xpls[i] - x[i];
        y[i] = gpls[i] - g[i];
    }
    let den1 = ddot(n, &s, 0, 1, &y, 0, 1);
    let snorm2 = dnrm2(n, &s, 0);
    let ynrm2 = dnrm2(n, &y, 0);
    if den1 < epsm.sqrt() * snorm2 * ynrm2 {
        return noupdt;
    }
    mvmltu(n, a, &s, &mut u);
    let mut den2 = ddot(n, &u, 0, 1, &u, 0, 1);
    let mut alp = (den1 / den2).sqrt();
    if noupdt {
        for j in 0..n {
            u[j] = alp * u[j];
            for i in j..n {
                a[i + j * n] *= alp;
            }
        }
        noupdt = false;
        den2 = den1;
        alp = 1.0;
    }
    mvmltl(n, a, &u, &mut w);
    let reltol = if iagflg == 0 { rnf.sqrt() } else { rnf };
    let mut skpupd = true;
    for i in 0..n {
        skpupd = (y[i] - w[i]).abs() < reltol * fmax2(g[i].abs(), gpls[i].abs());
        if !skpupd {
            break;
        }
    }
    if skpupd {
        return noupdt;
    }
    for i in 0..n {
        w[i] = rfma(-alp, w[i], y[i]);
    }
    alp /= den1;
    for i in 0..n {
        u[i] *= alp;
    }
    for i in 1..n {
        for j in 0..i {
            a[j + i * n] = a[i + j * n];
            a[i + j * n] = 0.0;
        }
    }
    qrupdt(n, a, &mut u, &w);
    for i in 1..n {
        for j in 0..i {
            a[i + j * n] = a[j + i * n];
        }
    }
    noupdt
}

/// uncmin.c `chlhsn` (:1361): safely-PD LL' of the model Hessian.
fn chlhsn(n: usize, a: &mut [f64], epsm: f64, sx: &[f64], udiag: &mut [f64]) {
    for j in 0..n {
        for i in j..n {
            a[i + j * n] /= sx[i] * sx[j];
        }
    }
    let tol = epsm.sqrt();
    let mut diagmx = a[0];
    let mut diagmn = a[0];
    if n > 1 {
        for i in 1..n {
            let tmp = a[i + i * n];
            if diagmn > tmp {
                diagmn = tmp;
            }
            if diagmx < tmp {
                diagmx = tmp;
            }
        }
    }
    let posmax = fmax2(diagmx, 0.0);
    if diagmn <= posmax * tol {
        let mut amu = rfma(tol, posmax - diagmn, -diagmn);
        if amu == 0.0 {
            let mut offmax = 0.0;
            for i in 1..n {
                for j in 0..i {
                    let tmp = a[i + j * n].abs();
                    if offmax < tmp {
                        offmax = tmp;
                    }
                }
            }
            amu = if offmax == 0.0 {
                1.0
            } else {
                offmax * (tol + 1.0)
            };
        }
        for i in 0..n {
            a[i + i * n] += amu;
        }
        diagmx += amu;
    }
    for i in 0..n {
        udiag[i] = a[i + i * n];
        for j in 0..i {
            a[j + i * n] = a[i + j * n];
        }
    }
    let addmax = choldc(n, a, diagmx, tol);
    if addmax > 0.0 {
        for i in 0..n {
            a[i + i * n] = udiag[i];
            for j in 0..i {
                a[i + j * n] = a[j + i * n];
            }
        }
        let mut evmin = 0.0;
        let mut evmax = a[0];
        for i in 0..n {
            let mut offrow = 0.0;
            for j in 0..i {
                offrow += a[i + j * n].abs();
            }
            for j in (i + 1)..n {
                offrow += a[j + i * n].abs();
            }
            let tmp = a[i + i * n] - offrow;
            if evmin > tmp {
                evmin = tmp;
            }
            let tmp = a[i + i * n] + offrow;
            if evmax < tmp {
                evmax = tmp;
            }
        }
        let sdd = rfma(tol, evmax - evmin, -evmin);
        let amu = fmin2(sdd, addmax);
        for i in 0..n {
            a[i + i * n] += amu;
            udiag[i] = a[i + i * n];
        }
        choldc(n, a, 0.0, tol);
    }
    for j in 0..n {
        for i in j..n {
            a[i + j * n] *= sx[i];
        }
        for i in 0..j {
            a[i + j * n] *= sx[i] * sx[j];
        }
        udiag[j] *= sx[j] * sx[j];
    }
}

/// uncmin.c `hsnint` (:1539): initial Hessian for secant updates.
fn hsnint(n: usize, a: &mut [f64], sx: &[f64], method: i32) {
    for i in 0..n {
        a[i + i * n] = if method == 3 { sx[i] * sx[i] } else { sx[i] };
        for j in 0..i {
            a[i + j * n] = 0.0;
        }
    }
}

/// uncmin.c `fstofd` (:1567): forward-difference derivative columns.
/// `m == 1` estimates a gradient row from `fcn`; `m == n` a Hessian
/// from `d1fcn`; `xpls` is perturbed and restored in place.
#[allow(clippy::too_many_arguments)]
fn fstofd(
    m: usize,
    n: usize,
    xpls: &mut [f64],
    obj: &mut dyn Obj,
    grad_mode: bool,
    fpls: &[f64],
    a: &mut [f64],
    nr: usize,
    sx: &[f64],
    rnoise: f64,
    icase: i32,
) -> PyResult<()> {
    let mut fhat = vec![0.0; m];
    for j in 0..n {
        let stepsz = rnoise.sqrt() * fmax2(xpls[j].abs(), 1.0 / sx[j]);
        let xtmpj = xpls[j];
        xpls[j] = xtmpj + stepsz;
        if grad_mode {
            obj.d1fcn(xpls, &mut fhat)?;
        } else {
            fhat[0] = obj.fcn(xpls)?;
        }
        xpls[j] = xtmpj;
        for i in 0..m {
            a[i + j * nr] = (fhat[i] - fpls[i]) / stepsz;
        }
    }
    if icase == 3 && n > 1 {
        for i in 1..m {
            for j in 0..i {
                a[i + j * nr] = (a[i + j * nr] + a[j + i * nr]) / 2.0;
            }
        }
    }
    Ok(())
}

/// uncmin.c `fstocd` (:1648): central-difference gradient.
fn fstocd(
    n: usize,
    x: &mut [f64],
    obj: &mut dyn Obj,
    sx: &[f64],
    rnoise: f64,
    g: &mut [f64],
) -> PyResult<()> {
    for i in 0..n {
        let xtempi = x[i];
        let stepi = rnoise.powf(1.0 / 3.0) * fmax2(xtempi.abs(), 1.0 / sx[i]);
        x[i] = xtempi + stepi;
        let fplus = obj.fcn(x)?;
        x[i] = xtempi - stepi;
        let fminus = obj.fcn(x)?;
        x[i] = xtempi;
        g[i] = (fplus - fminus) / (stepi * 2.0);
    }
    Ok(())
}

/// uncmin.c `sndofd` (:1686): second-order FD Hessian (no gradient).
fn sndofd(
    n: usize,
    xpls: &mut [f64],
    obj: &mut dyn Obj,
    fpls: f64,
    a: &mut [f64],
    sx: &[f64],
    rnoise: f64,
) -> PyResult<()> {
    let mut stepsz = vec![0.0; n];
    let mut anbr = vec![0.0; n];
    for i in 0..n {
        let xtmpi = xpls[i];
        stepsz[i] = rnoise.powf(1.0 / 3.0) * fmax2(xtmpi.abs(), 1.0 / sx[i]);
        xpls[i] = xtmpi + stepsz[i];
        anbr[i] = obj.fcn(xpls)?;
        xpls[i] = xtmpi;
    }
    for i in 0..n {
        let xtmpi = xpls[i];
        xpls[i] = xtmpi + stepsz[i] * 2.0;
        let fhat = obj.fcn(xpls)?;
        a[i + i * n] = ((fpls - anbr[i]) + (fhat - anbr[i])) / (stepsz[i] * stepsz[i]);
        if i == 0 {
            xpls[i] = xtmpi;
            continue;
        }
        xpls[i] = xtmpi + stepsz[i];
        for j in 0..i {
            let xtmpj = xpls[j];
            xpls[j] = xtmpj + stepsz[j];
            let fhat = obj.fcn(xpls)?;
            a[i + j * n] = ((fpls - anbr[i]) + (fhat - anbr[j])) / (stepsz[i] * stepsz[j]);
            xpls[j] = xtmpj;
        }
        xpls[i] = xtmpi;
    }
    Ok(())
}

/// uncmin.c `grdchk` (:1760): analytic-vs-FD gradient check.
#[allow(clippy::too_many_arguments)]
fn grdchk(
    n: usize,
    x: &mut [f64],
    obj: &mut dyn Obj,
    f: f64,
    g: &[f64],
    typsiz: &[f64],
    sx: &[f64],
    fscale: f64,
    rnf: f64,
    analtl: f64,
    msg: i32,
) -> PyResult<i32> {
    let mut wrk1 = vec![0.0; n];
    let fpls = [f];
    fstofd(1, n, x, obj, false, &fpls, &mut wrk1, 1, sx, rnf, 1)?;
    for i in 0..n {
        let gs = fmax2(f.abs(), fscale) / fmax2(x[i].abs(), typsiz[i]);
        if (g[i] - wrk1[i]).abs() > fmax2(g[i].abs(), gs) * analtl {
            return Ok(-21);
        }
    }
    Ok(msg)
}

/// uncmin.c `heschk` (:1804): analytic-vs-FD Hessian check.
#[allow(clippy::too_many_arguments)]
fn heschk(
    n: usize,
    x: &mut [f64],
    obj: &mut dyn Obj,
    f: f64,
    g: &mut [f64],
    a: &mut [f64],
    typsiz: &[f64],
    sx: &[f64],
    rnf: f64,
    analtl: f64,
    iagflg: i32,
    msg: i32,
) -> PyResult<i32> {
    let mut udiag = vec![0.0; n];
    if iagflg != 0 {
        let gc = g.to_vec();
        fstofd(n, n, x, obj, true, &gc, a, n, sx, rnf, 3)?;
    } else {
        sndofd(n, x, obj, f, a, sx, rnf)?;
    }
    for j in 0..n {
        udiag[j] = a[j + j * n];
        for i in (j + 1)..n {
            a[j + i * n] = a[i + j * n];
        }
    }
    obj.d2fcn(x, a, n)?;
    for j in 0..n {
        let hs = fmax2(g[j].abs(), 1.0) / fmax2(x[j].abs(), typsiz[j]);
        if (a[j + j * n] - udiag[j]).abs() > fmax2(udiag[j].abs(), hs) * analtl {
            return Ok(-22);
        }
        for i in (j + 1)..n {
            let temp1 = a[i + j * n];
            let temp2 = (temp1 - a[j + i * n]).abs();
            if temp2 > fmax2(temp1.abs(), hs) * analtl {
                return Ok(-22);
            }
        }
    }
    Ok(msg)
}

/// uncmin.c `opt_stop` (:1884). Returns (itrmcd, icscmx).
#[allow(clippy::too_many_arguments)]
fn opt_stop(
    n: usize,
    xpls: &[f64],
    fpls: f64,
    gpls: &[f64],
    x: &[f64],
    itncnt: i32,
    mut icscmx: i32,
    gradtl: f64,
    steptl: f64,
    sx: &[f64],
    fscale: f64,
    itnlim: i32,
    iretcd: i32,
    mxtake: bool,
) -> (i32, i32) {
    if iretcd == 1 {
        return (3, icscmx);
    }
    let d = fmax2(fpls.abs(), fscale);
    let mut rgx = 0.0;
    for i in 0..n {
        let relgrd = gpls[i].abs() * fmax2(xpls[i].abs(), 1.0 / sx[i]) / d;
        if rgx < relgrd {
            rgx = relgrd;
        }
    }
    let mut jtrmcd = 1;
    if rgx > gradtl {
        if itncnt == 0 {
            return (0, icscmx);
        }
        let mut rsx = 0.0;
        for i in 0..n {
            let relstp = (xpls[i] - x[i]).abs() / fmax2(xpls[i].abs(), 1.0 / sx[i]);
            if rsx < relstp {
                rsx = relstp;
            }
        }
        jtrmcd = 2;
        if rsx > steptl {
            jtrmcd = 4;
            if itncnt < itnlim {
                if !mxtake {
                    icscmx = 0;
                    return (0, icscmx);
                }
                icscmx += 1;
                if icscmx < 5 {
                    return (0, icscmx);
                }
                jtrmcd = 5;
            }
        }
    }
    (jtrmcd, icscmx)
}

/// uncmin.c `optchk` (:1973). Mutates typsiz/sx; returns the reset
/// scalars, msg < 0 on input error.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn optchk(
    n: usize,
    x: &[f64],
    typsiz: &mut [f64],
    sx: &mut [f64],
    mut fscale: f64,
    gradtl: f64,
    itnlim: i32,
    mut ndigit: i32,
    epsm: f64,
    mut dlt: f64,
    mut method: i32,
    mut iexp: i32,
    mut iagflg: i32,
    mut iahflg: i32,
    mut stepmx: f64,
    msg: i32,
) -> (f64, i32, i32, f64, i32, i32, i32, i32, f64, i32) {
    if !(1..=3).contains(&method) {
        method = 1;
    }
    if iagflg != 1 {
        iagflg = 0;
    }
    if iahflg != 1 {
        iahflg = 0;
    }
    if iexp != 0 {
        iexp = 1;
    }
    if (msg / 2) % 2 == 1 && iagflg == 0 {
        return (
            fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -6,
        );
    }
    if (msg / 4) % 2 == 1 && iahflg == 0 {
        return (
            fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -7,
        );
    }
    if n == 0 {
        return (
            fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -1,
        );
    }
    if n == 1 && msg % 2 == 0 {
        return (
            fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -2,
        );
    }
    for i in 0..n {
        if typsiz[i] == 0.0 {
            typsiz[i] = 1.0;
        } else if typsiz[i] < 0.0 {
            typsiz[i] = -typsiz[i];
        }
        sx[i] = 1.0 / typsiz[i];
    }
    if stepmx <= 0.0 {
        let mut stpsiz = 0.0;
        for i in 0..n {
            stpsiz = rfma(x[i] * x[i] * sx[i], sx[i], stpsiz);
        }
        stepmx = 1000.0 * fmax2(stpsiz.sqrt(), 1.0);
    }
    if fscale == 0.0 {
        fscale = 1.0;
    } else if fscale < 0.0 {
        fscale = -fscale;
    }
    if gradtl < 0.0 {
        return (
            fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -3,
        );
    }
    if itnlim <= 0 {
        return (
            fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -4,
        );
    }
    if ndigit == 0 {
        return (
            fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -5,
        );
    }
    if ndigit < 0 {
        ndigit = (-epsm.log10()) as i32;
    }
    if dlt <= 0.0 {
        dlt = -1.0;
    } else if dlt > stepmx {
        dlt = stepmx;
    }
    (
        fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, msg,
    )
}

/// uncmin.c `optdrv` (:2166). Returns (fpls, itrmcd, itncnt, msg).
#[allow(clippy::too_many_arguments)]
pub fn optdrv(
    n: usize,
    x: &mut [f64],
    obj: &mut dyn Obj,
    typsiz: &mut [f64],
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
    xpls: &mut [f64],
    gpls: &mut [f64],
) -> PyResult<(f64, i32, i32, i32)> {
    let mut a = vec![0.0; n * n];
    let mut udiag = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut sx = vec![0.0; n];
    let mut wrk1 = vec![0.0; n];
    let mut itncnt = 0;
    let epsm = DBL_EPSILON;
    let (fscale, itnlim, ndigit, mut dlt, method, iexp, mut iagflg, iahflg, stepmx, mut msg) =
        optchk(
            n, x, typsiz, &mut sx, fscale, gradtl, itnlim, ndigit, epsm, dlt, method, iexp, iagflg,
            iahflg, stepmx, msg,
        );
    if msg < 0 {
        return Ok((0.0, 0, itncnt, msg));
    }
    let rnf = fmax2(10f64.powf(-(ndigit as f64)), epsm);
    let analtl = fmax2(0.1, rnf.sqrt());
    let mut f = obj.fcn(x)?;
    if iagflg == 0 {
        let fv = [f];
        fstofd(1, n, x, obj, false, &fv, &mut g, 1, &sx, rnf, 1)?;
    } else {
        obj.d1fcn(x, &mut g)?;
        if (msg / 2) % 2 == 0 {
            msg = grdchk(n, x, obj, f, &g, typsiz, &sx, fscale, rnf, analtl, msg)?;
            if msg < 0 {
                return Ok((f, 0, itncnt, msg));
            }
        }
    }
    let iretcd = -1;
    let (mut itrmcd, mut icscmx) = opt_stop(
        n, x, f, &g, &wrk1, itncnt, 0, gradtl, steptl, &sx, fscale, itnlim, iretcd, false,
    );
    if itrmcd != 0 {
        // immediate convergence: optdrv_end's itrmcd-3 reset
        let fpls = f;
        xpls[..n].copy_from_slice(&x[..n]);
        gpls[..n].copy_from_slice(&g[..n]);
        return Ok((fpls, itrmcd, itncnt, 0));
    }
    if iexp != 0 {
        hsnint(n, &mut a, &sx, method);
    } else if iahflg == 0 {
        if iagflg != 0 {
            let gc = g.to_vec();
            fstofd(n, n, x, obj, true, &gc, &mut a, n, &sx, rnf, 3)?;
        } else {
            sndofd(n, x, obj, f, &mut a, &sx, rnf)?;
        }
    } else if (msg / 4) % 2 == 1 {
        obj.d2fcn(x, &mut a, n)?;
    } else {
        msg = heschk(
            n, x, obj, f, &mut g, &mut a, typsiz, &sx, rnf, analtl, iagflg, msg,
        )?;
        if msg < 0 {
            return Ok((f, 0, itncnt, msg));
        }
    }

    let mut fpls = 0.0;
    let mut mxtake;
    let mut noupdt = false;
    let (mut dltsav, mut dlpsav, mut phisav, mut amusav, mut phpsav) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut dltp, mut phi, mut phip0, mut amu) = (0.0, 0.0, 0.0, 0.0);
    let mut iretcd;
    loop {
        itncnt += 1;
        if !(iexp != 0 && method != 3) {
            chlhsn(n, &mut a, epsm, &sx, &mut udiag);
        }
        loop {
            // L105: solve for newton step ap = -g
            for i in 0..n {
                wrk1[i] = -g[i];
            }
            {
                let b: Vec<f64> = wrk1[..n].to_vec();
                lltslv(n, &a, &mut p, &b);
            }
            if iagflg == 0 && method != 1 {
                dltsav = dlt;
                if method != 2 {
                    amusav = amu;
                    dlpsav = dltp;
                    phisav = phi;
                    phpsav = phip0;
                }
            }
            match method {
                1 => {
                    let r = lnsrch(n, x, f, &g, &mut p, xpls, obj, stepmx, steptl, &sx)?;
                    (fpls, iretcd, mxtake) = r;
                }
                2 => {
                    let r = dogdrv(n, x, f, &g, &a, &p, xpls, obj, &sx, stepmx, steptl, dlt)?;
                    (fpls, dlt, iretcd, mxtake) = r;
                }
                _ => {
                    let r = hookdrv(
                        n, x, f, &g, &mut a, &udiag, &p, xpls, obj, &sx, stepmx, steptl, dlt, amu,
                        dltp, phi, phip0, epsm, itncnt,
                    )?;
                    (fpls, dlt, iretcd, mxtake, amu, dltp, phi, phip0) = r;
                }
            }
            if iretcd == 1 && iagflg == 0 {
                iagflg = -1;
                fstocd(n, x, obj, &sx, rnf, &mut g)?;
                if method == 1 {
                    continue;
                }
                dlt = dltsav;
                if method == 2 {
                    continue;
                }
                amu = amusav;
                dltp = dlpsav;
                phi = phisav;
                phip0 = phpsav;
                chlhsn(n, &mut a, epsm, &sx, &mut udiag);
                continue;
            }
            break;
        }
        for i in 0..n {
            p[i] = xpls[i] - x[i];
        }
        match iagflg {
            -1 => fstocd(n, xpls, obj, &sx, rnf, gpls)?,
            0 => {
                let fv = [fpls];
                fstofd(1, n, xpls, obj, false, &fv, gpls, 1, &sx, rnf, 1)?;
            }
            _ => obj.d1fcn(xpls, gpls)?,
        }
        let r = opt_stop(
            n, xpls, fpls, gpls, x, itncnt, icscmx, gradtl, steptl, &sx, fscale, itnlim, iretcd,
            mxtake,
        );
        itrmcd = r.0;
        icscmx = r.1;
        if itrmcd != 0 {
            break;
        }
        if iexp != 0 {
            if method == 3 {
                noupdt = secunf(
                    n, x, &g, &mut a, &udiag, xpls, gpls, epsm, itncnt, rnf, iagflg, noupdt,
                );
            } else {
                noupdt = secfac(
                    n, x, &g, &mut a, xpls, gpls, epsm, itncnt, rnf, iagflg, noupdt,
                );
            }
        } else if iahflg == 0 {
            if iagflg != 0 {
                let gc = gpls.to_vec();
                fstofd(n, n, xpls, obj, true, &gc, &mut a, n, &sx, rnf, 3)?;
            } else {
                sndofd(n, xpls, obj, fpls, &mut a, &sx, rnf)?;
            }
        } else {
            obj.d2fcn(xpls, &mut a, n)?;
        }
        f = fpls;
        x[..n].copy_from_slice(&xpls[..n]);
        g[..n].copy_from_slice(&gpls[..n]);
    }
    // optdrv_end
    if itrmcd == 3 {
        fpls = f;
        xpls[..n].copy_from_slice(&x[..n]);
        gpls[..n].copy_from_slice(&g[..n]);
    }
    Ok((fpls, itrmcd, itncnt, 0))
}
