//! mgcv's discrete `X'WX`: `XWXd0` (discrete.c:2056-2273) driving the
//! per-block `XWXijs` (discrete.c:1672-2030), with `tensorXj`, `Ztb`,
//! `indReduce`, `row_squash` and `up2lo`, computed as mgcv.so does on arm64.
//! A scalar `acc += a*b*...*z` keeps the leading products rounded and fuses
//! the last multiply into the add; in `a*b + c*d` the left product is the
//! fused one. The "simplest" diagonal block's contiguous reduction
//! `x += wb*Xj*Xi` multiplies eight terms at a time and adds them in order,
//! fusing only the remainder. The cross products are BLAS `dgemm` calls to
//! the vendor library, as R's are.
//!
//! Layout (mgcv's): `x` is `unlist(X)`, each marginal column-major and
//! `m[q] x p[q]`; `k` is the `n x nk` index matrix column-major and 0-based;
//! `ks` is `nx x 2` column-major (start, end) into the columns of `k`; `ts`
//! (0-based) and `dt` give each term's first marginal and count; `v` is
//! `unlist(v)` and `qc` the constraint codes. The result is the
//! `ptot x ptot` matrix, column-major.

use crate::lapack::blas::dgemm;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// `x = 0; for kk: x += a[kk]*b[kk]*c[kk]` as clang vectorizes it: eight
/// rounded products `(a*b)*c` added in order, the `len % 8` tail fused.
fn strict8_triple(a: &[f64], b: &[f64], c: &[f64], len: usize) -> f64 {
    let mut x = 0.0f64;
    let nv = len & !7usize;
    let mut kk = 0;
    while kk < nv {
        for q in 0..8 {
            x += a[kk + q] * b[kk + q] * c[kk + q];
        }
        kk += 8;
    }
    while kk < len {
        x = fma(a[kk] * b[kk], c[kk], x);
        kk += 1;
    }
    x
}

/// `SMinihash`: the 256 hash generator words.
fn sm_inihash() -> [u64; 256] {
    let mut ht = [0u64; 256];
    let mut h: u64 = 0x987564BACF987454;
    for slot in ht.iter_mut() {
        for _ in 0..31 {
            h ^= h >> 7;
            h ^= h << 11;
            h ^= h >> 10;
        }
        *slot = h;
    }
    ht
}

struct Ctx<'a> {
    x: &'a [f64],
    k: &'a [i32],
    ks: &'a [i32],
    m: &'a [i32],
    p: &'a [i32],
    ts: &'a [i32],
    dt: &'a [i32],
    nx: usize,
    n: usize,
    off: Vec<usize>,
    /// Diagonal of the weight matrix (with AR1, the tridiagonal's leading
    /// diagonal formed as `XWXd0` does).
    w: Vec<f64>,
    /// AR1 super-diagonal then sub-diagonal (`n - 1` each); empty otherwise.
    ws: Vec<f64>,
    tri: bool,
    ht: [u64; 256],
}

impl Ctx<'_> {
    fn kcol(&self, c: usize) -> &[i32] {
        &self.k[c * self.n..(c + 1) * self.n]
    }
    fn ks0(&self, q: usize) -> usize {
        self.ks[q] as usize
    }
    fn ksn(&self, q: usize) -> usize {
        (self.ks[q + self.nx] - self.ks[q]) as usize
    }
    fn wl(&self) -> &[f64] {
        &self.ws[self.n - 1..]
    }
}

/// `tensorXj`: multiply `xj` elementwise by column `j` of the row tensor
/// product of `dt` marginals starting at marginal `q0`, using index columns
/// `ks[q0 + i] + koff`.
fn tensor_xj(cx: &Ctx, xj: &mut [f64], q0: usize, dt: usize, j: usize, koff: usize) {
    let n = cx.n;
    let mut q: usize = 1;
    for i in 0..dt {
        q *= cx.p[q0 + i] as usize;
    }
    let mut jp = j;
    for i in 0..dt {
        let qi = q0 + i;
        q /= cx.p[qi] as usize;
        let l = jp / q;
        jp %= q;
        let mi = cx.m[qi] as usize;
        let col = &cx.x[cx.off[qi] + mi * l..cx.off[qi] + mi * (l + 1)];
        let kp = cx.kcol(cx.ks0(qi) + koff);
        for kk in 0..n {
            xj[kk] *= col[kp[kk] as usize];
        }
    }
}

/// `indReduce` with `buffer = 1`: accumulate the sparse weight matrix
/// (`w[l]` to `(ka[l], kb[l])`, and with `tri` the super/sub weights to the
/// off-diagonal positions) through mgcv's hash table, read it out in slot
/// order, then `C += W A` (`trans` false) or `C += W' A`. `C` is `rc x cc`,
/// `A` is `ra x cc`, both column-major.
#[allow(clippy::too_many_arguments)]
fn ind_reduce(
    ht: &[u64; 256],
    ka: &[i32],
    kb: &[i32],
    w: &[f64],
    ws: &[f64],
    wl: &[f64],
    tri: bool,
    n: usize,
    c: &mut [f64],
    a: &[f64],
    rc: usize,
    cc: usize,
    ra: usize,
    trans: bool,
) {
    struct El {
        i: i32,
        j: i32,
        w: f64,
        next: usize,
    }
    const NIL: usize = usize::MAX;
    let mut sm = vec![NIL; n];
    let mut stack: Vec<El> = Vec::with_capacity(if tri { 3 * n } else { n });
    let mut ntri = if tri { 3 } else { 1 };
    for l in 0..n {
        let mut t = 0;
        while t < ntri {
            let (i, j, wij) = if t == 0 {
                (ka[l], kb[l], w[l])
            } else if t == 1 {
                (ka[l], kb[l + 1], ws[l])
            } else {
                if l + 2 == n {
                    ntri = 1;
                }
                (ka[l + 1], kb[l], wl[l])
            };
            let mut key = [0u8; 8];
            key[..4].copy_from_slice(&i.to_ne_bytes());
            key[4..].copy_from_slice(&j.to_ne_bytes());
            let mut h: u64 = 0x99EFB145DAA48450;
            for &b in &key {
                h = h.wrapping_mul(7664345821815920749) ^ ht[b as usize];
            }
            let slot = (h % n as u64) as usize;
            let mut e = sm[slot];
            while e != NIL {
                if stack[e].i == i && stack[e].j == j {
                    stack[e].w += wij;
                    break;
                }
                e = stack[e].next;
            }
            if e == NIL {
                stack.push(El {
                    i,
                    j,
                    w: wij,
                    next: sm[slot],
                });
                sm[slot] = stack.len() - 1;
            }
            t += 1;
        }
    }
    let mut ko: Vec<usize> = Vec::new();
    let mut lo: Vec<usize> = Vec::new();
    let mut wo: Vec<f64> = Vec::new();
    for &head in &sm {
        let mut e = head;
        while e != NIL {
            ko.push(stack[e].i as usize);
            lo.push(stack[e].j as usize);
            wo.push(stack[e].w);
            e = stack[e].next;
        }
    }
    for q in 0..cc {
        let cq = &mut c[q * rc..(q + 1) * rc];
        let aq = &a[q * ra..(q + 1) * ra];
        if trans {
            for e in 0..wo.len() {
                cq[lo[e]] = fma(aq[ko[e]], wo[e], cq[lo[e]]);
            }
        } else {
            for e in 0..wo.len() {
                cq[ko[e]] = fma(aq[lo[e]], wo[e], cq[ko[e]]);
            }
        }
    }
}

/// `XWXijs`: sub-block `(r, c)` of the term block `(i, j)` of `X'WX`,
/// written at `xwx[base..]` with leading dimension `nxwx`.
#[allow(clippy::too_many_arguments)]
fn xwxijs(
    cx: &Ctx,
    xwx: &mut [f64],
    base: usize,
    nxwx: usize,
    i: usize,
    j: usize,
    r: usize,
    c: usize,
) {
    let n = cx.n;
    let tri = cx.tri;
    let w = &cx.w[..];
    let ts = |t: usize| cx.ts[t] as usize;
    let dtt = |t: usize| cx.dt[t] as usize;
    let si = cx.ksn(ts(i));
    let im = ts(i) + dtt(i) - 1;
    let mim = cx.m[im] as usize;
    // row-tensor work columns, needed only by tensor terms
    let tens = cx.dt[i] > 1 || cx.dt[j] > 1;
    let mut dxi = if tens { vec![1.0f64; n] } else { Vec::new() };
    let mut dxj = if tens { vec![1.0f64; n] } else { Vec::new() };
    let at = |row: usize, col: usize| base + row + nxwx * col;

    if dtt(i) == 1 && dtt(j) == 1 && cx.m[ts(i)] as usize == n && cx.m[ts(j)] as usize == n {
        // both stored matrices have n rows
        let jm = ts(j);
        let mjm = cx.m[jm] as usize;
        let pim = cx.p[im] as usize;
        let pjm = cx.p[jm] as usize;
        let sj = cx.ksn(ts(j));
        for ii in 0..pim {
            for jj in 0..pjm {
                xwx[at(ii, jj)] = 0.0;
            }
        }
        for s in 0..si {
            for t in 0..sj {
                let ki = cx.kcol(cx.ks0(im) + s);
                let kj = cx.kcol(cx.ks0(jm) + t);
                for ii in 0..pim {
                    let xi = &cx.x[cx.off[im] + mim * ii..];
                    let jj0 = if i == j { ii } else { 0 };
                    for jj in jj0..pjm {
                        let xj = &cx.x[cx.off[jm] + mjm * jj..];
                        let x = if tri {
                            let ws = &cx.ws[..];
                            let wl = cx.wl();
                            let ki_ = |q: usize| ki[q] as usize;
                            let kj_ = |q: usize| kj[q] as usize;
                            let mut x = xi[ki_(0)] * fma(xj[kj_(0)], w[0], xj[kj_(1)] * ws[0]);
                            for q in 1..n - 1 {
                                let s2 = fma(wl[q - 1], xj[kj_(q - 1)], w[q] * xj[kj_(q)]);
                                let s3 = fma(ws[q], xj[kj_(q + 1)], s2);
                                x = fma(xi[ki_(q)], s3, x);
                            }
                            let q = n - 1;
                            let s2 = fma(wl[q - 1], xj[kj_(q - 1)], w[q] * xj[kj_(q)]);
                            fma(xi[ki_(q)], s2, x)
                        } else {
                            let mut x = 0.0f64;
                            for q in 0..n {
                                x = fma(w[q] * xi[ki[q] as usize], xj[kj[q] as usize], x);
                            }
                            x
                        };
                        if i == j {
                            let v = xwx[at(ii, jj)] + x;
                            xwx[at(ii, jj)] = v;
                            xwx[at(jj, ii)] = v;
                        } else {
                            xwx[at(ii, jj)] += x;
                        }
                    }
                }
            }
        }
    } else if !tri && i == j && si == 1 {
        // simplest setup: accumulate the diagonal weights
        let mut wb = vec![0.0f64; mim];
        if dtt(i) > 1 {
            tensor_xj(cx, &mut dxi, ts(i), dtt(i) - 1, r, 0);
            if r != c {
                tensor_xj(cx, &mut dxj, ts(i), dtt(i) - 1, c, 0);
            }
        }
        let kf = cx.kcol(cx.ks0(im));
        if dtt(i) > 1 {
            let dxjr: &[f64] = if r != c { &dxj } else { &dxi };
            for kk in 0..n {
                let b = kf[kk] as usize;
                wb[b] = fma(dxi[kk] * dxjr[kk], w[kk], wb[b]);
            }
        } else {
            // SAFETY: indices checked by `xwxd0_py`; `wb` holds `mim`.
            unsafe {
                let (wp, kp, wbp) = (w.as_ptr(), kf.as_ptr(), wb.as_mut_ptr());
                for kk in 0..n {
                    *wbp.add(*kp.add(kk) as usize) += *wp.add(kk);
                }
            }
        }
        let xt = &cx.x[cx.off[im]..];
        let pim = cx.p[im] as usize;
        let col = |q: usize| &xt[mim * q..mim * (q + 1)];
        if r != c {
            for jj in 0..pim {
                for ii in 0..pim {
                    let x = strict8_triple(&wb, col(jj), col(ii), mim);
                    xwx[at(c * pim + jj, r * pim + ii)] = x;
                    xwx[at(r * pim + ii, c * pim + jj)] = x;
                }
            }
        } else {
            for ii in 0..pim {
                for jj in ii..pim {
                    let x = strict8_triple(&wb, col(jj), col(ii), mim);
                    xwx[at(r * pim + ii, c * pim + jj)] = x;
                    xwx[at(r * pim + jj, c * pim + ii)] = x;
                    xwx[base + (r * pim + ii) * nxwx + c * pim + jj] = x;
                    xwx[base + (r * pim + jj) * nxwx + c * pim + ii] = x;
                }
            }
        }
    } else {
        general_block(cx, xwx, base, nxwx, i, j, r, c, &mut dxi, &mut dxj);
    }
}

/// The general branch of `XWXijs` (discrete.c:1796-2028).
#[allow(clippy::too_many_arguments)]
fn general_block(
    cx: &Ctx,
    xwx: &mut [f64],
    base: usize,
    nxwx: usize,
    i: usize,
    j: usize,
    r: usize,
    c: usize,
    dxi: &mut [f64],
    dxj: &mut [f64],
) {
    let n = cx.n;
    let tri = cx.tri;
    let w = &cx.w[..];
    let ts = |t: usize| cx.ts[t] as usize;
    let dtt = |t: usize| cx.dt[t] as usize;
    let si = cx.ksn(ts(i));
    let sj = cx.ksn(ts(j));
    let im = ts(i) + dtt(i) - 1;
    let jm = ts(j) + dtt(j) - 1;
    let mim = cx.m[im] as usize;
    let mjm = cx.m[jm] as usize;
    let pim = cx.p[im] as usize;
    let pjm = cx.p[jm] as usize;
    let ddti = dtt(i) - 1;
    let ddtj = dtt(j) - 1;
    let tensi = ddti > 0;
    let tensj = ddtj > 0;
    let acc_w = n > mjm * mim;
    let (pim_i, pjm_i, mim_i, mjm_i, n_i) =
        (pim as i64, pjm as i64, mim as i64, mjm as i64, n as i64);
    let rfac = if acc_w {
        !(pim_i * mim_i * mjm_i + pim_i * pjm_i * mjm_i
            > mim_i * mjm_i * pjm_i + pim_i * pjm_i * mim_i)
    } else {
        let mut ii: i64 = if tensi { 2 } else { 1 };
        if tensj {
            ii += 1;
        }
        let alpha: i64 = if tri { ii * 3 + 3 } else { ii + 1 };
        let a = alpha * si as i64 * sj as i64 * n_i;
        let mut rf = !(a * pim_i + mjm_i * pim_i * pjm_i < a * pjm_i + mim_i * pim_i * pjm_i);
        if mim == n {
            rf = false;
        } else if mjm == n {
            rf = true;
        }
        rf
    };
    let mut wbar = if acc_w {
        vec![0.0f64; mim * mjm]
    } else {
        Vec::new()
    };
    let mut cmat = if rfac {
        vec![0.0f64; mim * pjm]
    } else {
        Vec::new()
    };
    let mut dmat = if rfac {
        Vec::new()
    } else {
        vec![0.0f64; mjm * pim]
    };
    let ws_all = &cx.ws[..];
    let wl_all: &[f64] = if tri { cx.wl() } else { &[] };
    let xi_all = &cx.x[cx.off[im]..cx.off[im] + mim * pim];
    let xj_all = &cx.x[cx.off[jm]..cx.off[jm] + mjm * pjm];

    for s in 0..si {
        for t in 0..sj {
            if tensi {
                dxi.iter_mut().for_each(|v| *v = 1.0);
                tensor_xj(cx, dxi, ts(i), ddti, r, s);
            }
            if tensj {
                dxj.iter_mut().for_each(|v| *v = 1.0);
                tensor_xj(cx, dxj, ts(j), ddtj, c, t);
            }
            let ki = cx.kcol(cx.ks0(im) + s);
            let kj = cx.kcol(cx.ks0(jm) + t);
            let kiu = |q: usize| ki[q] as usize;
            let kju = |q: usize| kj[q] as usize;
            if acc_w {
                let wi = |a: usize, b: usize| a + mim * b;
                if tri {
                    for q in 0..n - 1 {
                        let (sp, wp, lp) = (ws_all[q], w[q], wl_all[q]);
                        let (vs, vl, vw) = match (tensi, tensj) {
                            (true, true) => (
                                sp * dxi[q] * dxj[q + 1],
                                lp * dxi[q + 1] * dxj[q],
                                wp * dxi[q] * dxj[q],
                            ),
                            (true, false) => (sp * dxi[q], lp * dxi[q + 1], wp * dxi[q]),
                            (false, true) => (sp * dxj[q + 1], lp * dxj[q], wp * dxj[q]),
                            (false, false) => (sp, lp, wp),
                        };
                        let a1 = wi(kiu(q), kju(q + 1));
                        let a2 = wi(kiu(q + 1), kju(q));
                        let a3 = wi(kiu(q), kju(q));
                        match (tensi, tensj) {
                            (true, true) => {
                                wbar[a1] = fma(sp * dxi[q], dxj[q + 1], wbar[a1]);
                                wbar[a2] = fma(lp * dxi[q + 1], dxj[q], wbar[a2]);
                                wbar[a3] = fma(wp * dxi[q], dxj[q], wbar[a3]);
                            }
                            (false, false) => {
                                wbar[a1] += vs;
                                wbar[a2] += vl;
                                wbar[a3] += vw;
                            }
                            _ => {
                                let (fs, gs, fl, gl, fw, gw) = match (tensi, tensj) {
                                    (true, false) => (sp, dxi[q], lp, dxi[q + 1], wp, dxi[q]),
                                    _ => (sp, dxj[q + 1], lp, dxj[q], wp, dxj[q]),
                                };
                                wbar[a1] = fma(fs, gs, wbar[a1]);
                                wbar[a2] = fma(fl, gl, wbar[a2]);
                                wbar[a3] = fma(fw, gw, wbar[a3]);
                            }
                        }
                    }
                    let q = n - 1;
                    let a3 = wi(kiu(q), kju(q));
                    let wp = w[q];
                    wbar[a3] = match (tensi, tensj) {
                        (true, true) => fma(wp * dxi[q], dxj[q], wbar[a3]),
                        (true, false) => fma(wp, dxi[q], wbar[a3]),
                        (false, true) => fma(wp, dxj[q], wbar[a3]),
                        (false, false) => wbar[a3] + wp,
                    };
                } else if !tensi && !tensj {
                    // SAFETY: `xwxd0_py` checked every index against its
                    // marginal's rows, so `a < mim * mjm`, and all three
                    // arrays hold `n` entries.
                    unsafe {
                        let (wp, kip, kjp, wb) =
                            (w.as_ptr(), ki.as_ptr(), kj.as_ptr(), wbar.as_mut_ptr());
                        for q in 0..n {
                            let a = *kip.add(q) as usize + mim * *kjp.add(q) as usize;
                            *wb.add(a) += *wp.add(q);
                        }
                    }
                } else {
                    for q in 0..n {
                        let a = wi(kiu(q), kju(q));
                        wbar[a] = match (tensi, tensj) {
                            (true, true) => fma(w[q] * dxi[q], dxj[q], wbar[a]),
                            (true, false) => fma(w[q], dxi[q], wbar[a]),
                            (false, true) => fma(w[q], dxj[q], wbar[a]),
                            (false, false) => wbar[a] + w[q],
                        };
                    }
                }
            } else if (rfac && pjm > 15) || (!rfac && pim > 15) {
                // sparse accumulation of \bar W through indReduce
                let mut wi = vec![0.0f64; n];
                let mut wsi = vec![0.0f64; if tri { n } else { 0 }];
                let mut wli = vec![0.0f64; if tri { n } else { 0 }];
                if tri {
                    for q in 0..n - 1 {
                        match (tensi, tensj) {
                            (true, true) => {
                                wi[q] = w[q] * dxi[q] * dxj[q];
                                wsi[q] = ws_all[q] * dxi[q] * dxj[q + 1];
                                wli[q] = wl_all[q] * dxi[q + 1] * dxj[q];
                            }
                            (true, false) => {
                                wi[q] = w[q] * dxi[q];
                                wsi[q] = ws_all[q] * dxi[q];
                                wli[q] = wl_all[q] * dxi[q + 1];
                            }
                            (false, true) => {
                                wi[q] = w[q] * dxj[q];
                                wsi[q] = ws_all[q] * dxj[q + 1];
                                wli[q] = wl_all[q] * dxj[q];
                            }
                            (false, false) => {
                                wi[q] = w[q];
                                wsi[q] = ws_all[q];
                                wli[q] = wl_all[q];
                            }
                        }
                    }
                    let q = n - 1;
                    wi[q] = match (tensi, tensj) {
                        (true, true) => w[q] * dxi[q] * dxj[q],
                        (true, false) => w[q] * dxi[q],
                        (false, true) => w[q] * dxj[q],
                        (false, false) => w[q],
                    };
                } else {
                    for q in 0..n {
                        wi[q] = match (tensi, tensj) {
                            (true, true) => w[q] * dxi[q] * dxj[q],
                            (true, false) => w[q] * dxi[q],
                            (false, true) => w[q] * dxj[q],
                            (false, false) => w[q],
                        };
                    }
                }
                if rfac {
                    ind_reduce(
                        &cx.ht, ki, kj, &wi, &wsi, &wli, tri, n, &mut cmat, xj_all, mim, pjm, mjm,
                        false,
                    );
                } else {
                    ind_reduce(
                        &cx.ht, ki, kj, &wi, &wsi, &wli, tri, n, &mut dmat, xi_all, mjm, pim, mim,
                        true,
                    );
                }
            } else if rfac {
                // direct accumulation of C = \bar W X_j
                for q in 0..pjm {
                    let xj = &xj_all[q * mjm..(q + 1) * mjm];
                    let cq = &mut cmat[q * mim..(q + 1) * mim];
                    if tri {
                        for l in 0..n - 1 {
                            let (a, a1) = (kiu(l), kiu(l + 1));
                            let (b, b1) = (kju(l), kju(l + 1));
                            match (tensi, tensj) {
                                (true, true) => {
                                    let inner =
                                        fma(ws_all[l] * dxj[l + 1], xj[b1], w[l] * dxj[l] * xj[b]);
                                    cq[a] = fma(dxi[l], inner, cq[a]);
                                    cq[a1] = fma(wl_all[l] * dxi[l + 1] * dxj[l], xj[b], cq[a1]);
                                }
                                (true, false) => {
                                    let inner = fma(ws_all[l], xj[b1], w[l] * xj[b]);
                                    cq[a] = fma(dxi[l], inner, cq[a]);
                                    cq[a1] = fma(wl_all[l] * dxi[l + 1], xj[b], cq[a1]);
                                }
                                (false, true) => {
                                    let s0 =
                                        fma(ws_all[l] * dxj[l + 1], xj[b1], w[l] * dxj[l] * xj[b]);
                                    cq[a] += s0;
                                    cq[a1] = fma(wl_all[l] * dxj[l], xj[b], cq[a1]);
                                }
                                (false, false) => {
                                    let s0 = fma(ws_all[l], xj[b1], w[l] * xj[b]);
                                    cq[a] += s0;
                                    cq[a1] = fma(wl_all[l], xj[b], cq[a1]);
                                }
                            }
                        }
                        let l = n - 1;
                        let (a, b) = (kiu(l), kju(l));
                        cq[a] = match (tensi, tensj) {
                            (true, true) => fma(w[l] * dxi[l] * dxj[l], xj[b], cq[a]),
                            (true, false) => fma(dxi[l] * w[l], xj[b], cq[a]),
                            (false, true) => fma(w[l] * dxj[l], xj[b], cq[a]),
                            (false, false) => fma(w[l], xj[b], cq[a]),
                        };
                    } else if !tensi && !tensj {
                        // SAFETY: indices checked by `xwxd0_py`; `cq` holds
                        // `mim` and `xj` `mjm` entries.
                        unsafe {
                            let (wp, kip, kjp) = (w.as_ptr(), ki.as_ptr(), kj.as_ptr());
                            let (xjp, cqp) = (xj.as_ptr(), cq.as_mut_ptr());
                            for l in 0..n {
                                let a = *kip.add(l) as usize;
                                let b = *kjp.add(l) as usize;
                                *cqp.add(a) = fma(*wp.add(l), *xjp.add(b), *cqp.add(a));
                            }
                        }
                    } else {
                        for l in 0..n {
                            let (a, b) = (kiu(l), kju(l));
                            cq[a] = match (tensi, tensj) {
                                (true, true) => fma(w[l] * dxi[l] * dxj[l], xj[b], cq[a]),
                                (true, false) => fma(w[l] * dxi[l], xj[b], cq[a]),
                                (false, true) => fma(w[l] * dxj[l], xj[b], cq[a]),
                                (false, false) => fma(w[l], xj[b], cq[a]),
                            };
                        }
                    }
                }
            } else {
                // direct accumulation of D = \bar W' X_i
                for q in 0..pim {
                    let xi = &xi_all[q * mim..(q + 1) * mim];
                    let dq = &mut dmat[q * mjm..(q + 1) * mjm];
                    if tri {
                        for l in 0..n - 1 {
                            let (a, a1) = (kiu(l), kiu(l + 1));
                            let (b, b1) = (kju(l), kju(l + 1));
                            match (tensi, tensj) {
                                (true, true) => {
                                    let inner =
                                        fma(dxi[l + 1] * wl_all[l], xi[a1], dxi[l] * w[l] * xi[a]);
                                    dq[b] = fma(dxj[l], inner, dq[b]);
                                    dq[b1] = fma(ws_all[l] * dxi[l] * dxj[l + 1], xi[a], dq[b1]);
                                }
                                (true, false) => {
                                    let s0 =
                                        fma(wl_all[l] * dxi[l + 1], xi[a1], w[l] * dxi[l] * xi[a]);
                                    dq[b] += s0;
                                    dq[b1] = fma(ws_all[l] * dxi[l], xi[a], dq[b1]);
                                }
                                (false, true) => {
                                    let inner = fma(wl_all[l], xi[a1], w[l] * xi[a]);
                                    dq[b] = fma(dxj[l], inner, dq[b]);
                                    dq[b1] = fma(ws_all[l] * dxj[l + 1], xi[a], dq[b1]);
                                }
                                (false, false) => {
                                    let s0 = fma(wl_all[l], xi[a1], w[l] * xi[a]);
                                    dq[b] += s0;
                                    dq[b1] = fma(ws_all[l], xi[a], dq[b1]);
                                }
                            }
                        }
                        let l = n - 1;
                        let (a, b) = (kiu(l), kju(l));
                        dq[b] = match (tensi, tensj) {
                            (true, true) => fma(dxj[l] * dxi[l] * w[l], xi[a], dq[b]),
                            (true, false) => fma(w[l] * dxi[l], xi[a], dq[b]),
                            (false, true) => fma(dxj[l] * w[l], xi[a], dq[b]),
                            (false, false) => fma(w[l], xi[a], dq[b]),
                        };
                    } else if !tensi && !tensj {
                        // SAFETY: indices checked by `xwxd0_py`; `dq` holds
                        // `mjm` and `xi` `mim` entries.
                        unsafe {
                            let (wp, kip, kjp) = (w.as_ptr(), ki.as_ptr(), kj.as_ptr());
                            let (xip, dqp) = (xi.as_ptr(), dq.as_mut_ptr());
                            for l in 0..n {
                                let a = *kip.add(l) as usize;
                                let b = *kjp.add(l) as usize;
                                *dqp.add(b) = fma(*wp.add(l), *xip.add(a), *dqp.add(b));
                            }
                        }
                    } else {
                        for l in 0..n {
                            let (a, b) = (kiu(l), kju(l));
                            dq[b] = match (tensi, tensj) {
                                (true, true) => fma(w[l] * dxi[l] * dxj[l], xi[a], dq[b]),
                                (true, false) => fma(w[l] * dxi[l], xi[a], dq[b]),
                                (false, true) => fma(w[l] * dxj[l], xi[a], dq[b]),
                                (false, false) => fma(w[l], xi[a], dq[b]),
                            };
                        }
                    }
                }
            }
        }
    }
    // SAFETY: every matrix passed below is column-major with the stated
    // leading dimension and holds at least the rows x columns dgemm reads.
    unsafe {
        if acc_w {
            if rfac {
                dgemm(
                    b'N',
                    b'N',
                    mim as i32,
                    pjm as i32,
                    mjm as i32,
                    1.0,
                    wbar.as_ptr(),
                    mim as i32,
                    xj_all.as_ptr(),
                    mjm as i32,
                    0.0,
                    cmat.as_mut_ptr(),
                    mim as i32,
                );
            } else {
                dgemm(
                    b'T',
                    b'N',
                    mjm as i32,
                    pim as i32,
                    mim as i32,
                    1.0,
                    wbar.as_ptr(),
                    mim as i32,
                    xi_all.as_ptr(),
                    mim as i32,
                    0.0,
                    dmat.as_mut_ptr(),
                    mjm as i32,
                );
            }
        }
        let out = xwx.as_mut_ptr().add(base + r * pim + c * pjm * nxwx);
        if rfac {
            dgemm(
                b'T',
                b'N',
                pim as i32,
                pjm as i32,
                mim as i32,
                1.0,
                xi_all.as_ptr(),
                mim as i32,
                cmat.as_ptr(),
                mim as i32,
                0.0,
                out,
                nxwx as i32,
            );
        } else {
            dgemm(
                b'T',
                b'N',
                pim as i32,
                pjm as i32,
                mjm as i32,
                1.0,
                dmat.as_ptr(),
                mjm as i32,
                xj_all.as_ptr(),
                mjm as i32,
                0.0,
                out,
                nxwx as i32,
            );
        }
    }
}

/// `Ztb`: `b1 = Z' b0` for the `p` elements of `b0` spaced `di` apart,
/// written `di` apart from `o1`; `b0` starts at `o0` of the same array.
/// Householder (`qc > 0`): `(I - vv') b0` with the first element dropped;
/// `qc < 0`: the Kronecker product of sum-to-zero contrasts.
fn ztb(a: &mut [f64], o1: usize, o0: usize, v: &[f64], qc: i32, di: usize, p: usize) {
    if qc > 0 {
        let mut x = 0.0f64;
        for l in 0..p {
            x = fma(a[o0 + l * di], v[l], x);
        }
        for l in 1..p {
            a[o1 + (l - 1) * di] = fma(-v[l], x, a[o0 + l * di]);
        }
    } else if qc < 0 {
        let mut w0: Vec<f64> = (0..p).map(|l| a[o0 + l * di]).collect();
        let mut w1 = vec![0.0f64; p];
        let mm = v[0].round() as usize;
        let mut pp = p;
        for i in 0..mm {
            pp /= v[i + 1].round() as usize;
        }
        let mut k1 = p;
        for kk in 0..=mm {
            let (mk, mk1) = if kk < mm {
                let mk = v[kk + 1].round() as usize;
                (mk, mk - 1)
            } else {
                (pp, pp)
            };
            let p1m = k1 / mk;
            let mut q = 0;
            for i in 0..p1m {
                let z = if kk < mm { w0[i + (mk - 1) * p1m] } else { 0.0 };
                for jj in 0..mk1 {
                    w1[q] = w0[i + jj * p1m] - z;
                    q += 1;
                }
            }
            if kk < mm {
                k1 -= p1m;
            }
            std::mem::swap(&mut w0, &mut w1);
        }
        for l in 0..k1 {
            a[o1 + l * di] = w0[l];
        }
    }
}

/// R's `revsort` (sort.c): `a` into descending order by heapsort, `ib`
/// carried alongside.
fn revsort(a: &mut [f64], ib: &mut [usize]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    // 1-based, as R writes it
    let mut l = (n >> 1) + 1;
    let mut ir = n;
    loop {
        let (ra, ii);
        if l > 1 {
            l -= 1;
            ra = a[l - 1];
            ii = ib[l - 1];
        } else {
            ra = a[ir - 1];
            ii = ib[ir - 1];
            a[ir - 1] = a[0];
            ib[ir - 1] = ib[0];
            ir -= 1;
            if ir == 1 {
                a[0] = ra;
                ib[0] = ii;
                return;
            }
        }
        let mut i = l;
        let mut j = l << 1;
        while j <= ir {
            if j < ir && a[j - 1] > a[j] {
                j += 1;
            }
            if ra > a[j - 1] {
                a[i - 1] = a[j - 1];
                ib[i - 1] = ib[j - 1];
                i = j;
                j += i;
            } else {
                j = ir + 1;
            }
        }
        a[i - 1] = ra;
        ib[i - 1] = ii;
    }
}

/// mgcv's `XWXd1`: the `X'WX` blocks of the row terms `rs` against the
/// column terms `cs` (0-based), as a compact `ptot x ctot` matrix
/// (column-major), each constrained as `XWXd0` does. With `symmetric` (one
/// side given; `cs` is then `rs`) a single-index diagonal block forms only
/// its upper sub-blocks and the result is mirrored; otherwise every
/// sub-block is formed. mgcv squashes `ptot` columns out of the row
/// padding, which leaves later columns in place when the column terms are
/// wider; all `ctot` are squashed here.
#[allow(clippy::too_many_arguments)]
fn xwxd1(
    x: &[f64],
    w_in: &[f64],
    k: &[i32],
    ks: &[i32],
    m: &[i32],
    p: &[i32],
    ts: &[i32],
    dt: &[i32],
    v: &[f64],
    qc: &[i32],
    ar_weights: &[f64],
    rs: &[usize],
    cs: &[usize],
    symmetric: bool,
) -> (Vec<f64>, usize, usize) {
    let nx = m.len();
    let nt = ts.len();
    let n = w_in.len();
    let mut pt = vec![0usize; nt];
    let mut pd = vec![0usize; nt];
    let mut off = vec![0usize; nx + 1];
    let mut voff = vec![0usize; nt + 1];
    let mut q = 0;
    for i in 0..nt {
        for jj in 0..dt[i] as usize {
            if jj == dt[i] as usize - 1 {
                pd[i] = p[q] as usize;
            }
            off[q + 1] = off[q] + p[q] as usize * m[q] as usize;
            pt[i] = if jj == 0 {
                p[q] as usize
            } else {
                pt[i] * p[q] as usize
            };
            q += 1;
        }
        voff[i + 1] = if qc[i] == 0 {
            voff[i]
        } else if qc[i] > 0 {
            voff[i] + pt[i]
        } else {
            voff[i] + v[voff[i]].round() as usize + 2
        };
    }
    let constrained = |i: usize| -> usize {
        if qc[i] == 0 {
            pt[i]
        } else if qc[i] > 0 {
            pt[i] - 1
        } else {
            let si = v[voff[i]].round() as usize;
            pt[i] - v[voff[i] + si + 1].round() as usize
        }
    };
    // where each selected term starts, constrained and not, in the row set
    // and in the column set
    let mut tpsr = vec![0usize; nt + 1];
    let mut tpsur = vec![0usize; nt + 1];
    let (mut kk, mut ii) = (0usize, 0usize);
    for &r in rs {
        tpsr[r] = kk;
        tpsur[r] = ii;
        kk += constrained(r);
        ii += pt[r];
    }
    let (ptot, nxwx) = (kk, ii);
    let mut tpsc = vec![0usize; nt + 1];
    let mut tpsuc = vec![0usize; nt + 1];
    let (mut kk, mut ii) = (0usize, 0usize);
    for &c in cs {
        tpsc[c] = kk;
        tpsuc[c] = ii;
        kk += constrained(c);
        ii += pt[c];
    }
    let (ctot, ncu) = (kk, ii);
    let tri = !ar_weights.is_empty();
    let (w, ws) = if tri {
        let sw: Vec<f64> = w_in.iter().map(|v| v.sqrt()).collect();
        let mut ws = vec![0.0f64; 2 * n - 2];
        for i in 0..n - 1 {
            let val = ar_weights[2 * i + 1] * ar_weights[2 * i + 2] * sw[i + 1] * sw[i];
            ws[i] = val;
            ws[i + n - 1] = val;
        }
        let mut wd = sw.clone();
        for i in 0..n - 1 {
            let a = fma(
                ar_weights[2 * i + 1],
                ar_weights[2 * i + 1],
                ar_weights[2 * i] * ar_weights[2 * i],
            );
            wd[i] *= a * sw[i];
        }
        let i = n - 1;
        wd[i] *= sw[i] * ar_weights[2 * i] * ar_weights[2 * i];
        (wd, ws)
    } else {
        (w_in.to_vec(), Vec::new())
    };
    let cx = Ctx {
        x,
        k,
        ks,
        m,
        p,
        ts,
        dt,
        nx,
        n,
        off,
        w,
        ws,
        tri,
        ht: sm_inihash(),
    };
    // the main blocks, their sub-block counts and rough costs; the
    // sub-blocks are formed most costly first (R's revsort), the order
    // deciding which of two overlapping mirrored writes survives
    let (mut bl_r, mut bl_c, mut sb, mut bl_cost) =
        (Vec::new(), Vec::new(), vec![0usize], Vec::new());
    for (ri, &r) in rs.iter().enumerate() {
        let c0 = if symmetric { ri } else { 0 };
        for &c in &cs[c0..] {
            bl_r.push(r);
            bl_c.push(c);
            let si = cx.ksn(ts[r] as usize);
            let cnt = if symmetric && r == c && si == 1 && !tri {
                let i = pt[r] / pd[r];
                i * (i + 1) / 2
            } else {
                (pt[r] / pd[r]) * (pt[c] / pd[c])
            };
            sb.push(sb[sb.len() - 1] + cnt);
            let mrc = m[r].wrapping_mul(m[c]);
            let cost = if (mrc as i64) < n as i64 {
                let a = mrc as f64 * pd[c] as f64;
                let x = mrc as f64 * pd[r] as f64;
                if x < a {
                    x
                } else {
                    a
                }
            } else {
                let a = n as f64 * pd[c] as f64;
                let x = n as f64 * pd[r] as f64;
                if x < a {
                    x
                } else {
                    a
                }
            };
            bl_cost.push(cost);
        }
    }
    let nsub = sb[sb.len() - 1];
    let mut owner = vec![0usize; nsub];
    let mut cost = vec![0.0f64; nsub];
    let mut order: Vec<usize> = (0..nsub).collect();
    let mut kb = 0;
    for i in 0..nsub {
        while i >= sb[kb + 1] {
            kb += 1;
        }
        owner[i] = kb;
        cost[i] = bl_cost[kb];
    }
    revsort(&mut cost, &mut order);
    // Sub-blocks write only inside their main block, and only those of one
    // main block can overwrite each other: each main block is formed on its
    // own, its sub-blocks in their sorted order, the blocks in parallel.
    let mut per_block: Vec<Vec<usize>> = vec![Vec::new(); bl_r.len()];
    for &kk in &order {
        per_block[owner[kk]].push(kk);
    }
    let blocks: Vec<Vec<f64>> = per_block
        .par_iter()
        .enumerate()
        .map(|(kb, subs)| {
            let (rb, cb) = (bl_r[kb], bl_c[kb]);
            let ld = pt[rb];
            let mut blk = vec![0.0f64; (pt[rb] * pt[cb]).max(1)];
            let rt = pt[rb] / pd[rb];
            let ct = pt[cb] / pd[cb];
            for &kk in subs {
                let mut i = kk - sb[kb];
                let (r, c) = if symmetric && sb[kb + 1] - sb[kb] < rt * ct {
                    let mut r = 0;
                    while i >= rt - r {
                        i -= rt - r;
                        r += 1;
                    }
                    (r, i + r)
                } else {
                    (i / ct, i % ct)
                };
                xwxijs(&cx, &mut blk, 0, ld, rb, cb, r, c);
            }
            blk
        })
        .collect();
    let mut xwx = vec![0.0f64; (nxwx * ncu.max(nxwx)).max(1)];
    for (kb, blk) in blocks.iter().enumerate() {
        let (rb, cb) = (bl_r[kb], bl_c[kb]);
        let ld = pt[rb];
        let base = tpsur[rb] + nxwx * tpsuc[cb];
        for jj in 0..pt[cb] {
            xwx[base + jj * nxwx..base + jj * nxwx + ld]
                .copy_from_slice(&blk[jj * ld..jj * ld + ld]);
        }
    }
    for (ri, &r) in rs.iter().enumerate() {
        let c0 = if symmetric { ri } else { 0 };
        for &c in &cs[c0..] {
            let pa;
            if dt[r] > 1 && qc[r] != 0 {
                for jj in 0..pt[c] {
                    let o0 = tpsur[r] + (tpsuc[c] + jj) * nxwx;
                    let o1 = tpsr[r] + (tpsuc[c] + jj) * nxwx;
                    ztb(&mut xwx, o1, o0, &v[voff[r]..], qc[r], 1, pt[r]);
                }
                pa = pt[r] - 1;
            } else {
                pa = pt[r];
                if tpsur[r] != tpsr[r] {
                    for jj in 0..pt[c] {
                        let o0 = tpsur[r] + (tpsuc[c] + jj) * nxwx;
                        let o1 = tpsr[r] + (tpsuc[c] + jj) * nxwx;
                        for l in 0..pt[r] {
                            xwx[o1 + l] = xwx[o0 + l];
                        }
                    }
                }
            }
            if dt[c] > 1 && qc[c] != 0 {
                for jj in 0..pa {
                    let o0 = tpsr[r] + jj + tpsuc[c] * nxwx;
                    let o1 = tpsr[r] + jj + tpsc[c] * nxwx;
                    ztb(&mut xwx, o1, o0, &v[voff[c]..], qc[c], nxwx, pt[c]);
                }
            } else if tpsuc[c] != tpsc[c] {
                for jj in 0..pa {
                    let o0 = tpsr[r] + jj + tpsuc[c] * nxwx;
                    let o1 = tpsr[r] + jj + tpsc[c] * nxwx;
                    for l in 0..pt[c] {
                        xwx[o1 + l * nxwx] = xwx[o0 + l * nxwx];
                    }
                }
            }
        }
    }
    let mut out = vec![0.0f64; ptot * ctot];
    for cc in 0..ctot {
        for rr in 0..ptot {
            out[rr + cc * ptot] = xwx[rr + cc * nxwx];
        }
    }
    if symmetric {
        for i in 0..ptot {
            for jj in i + 1..ptot {
                out[jj + i * ptot] = out[i + jj * ptot];
            }
        }
    }
    (out, ptot, ctot)
}

/// mgcv's `XWXd1` (the `lt`/`rt` path of `XWXd`): the blocks of `X'WX`
/// for the row terms `rs` against the column terms `cs` (0-based), an
/// empty side meaning the other, as a `ptot x ctot` matrix (column-major).
#[pyfunction]
#[pyo3(name = "xwxd1")]
#[allow(clippy::too_many_arguments)]
fn xwxd1_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, i32>,
    ks: PyReadonlyArray1<'py, i32>,
    m: PyReadonlyArray1<'py, i32>,
    p: PyReadonlyArray1<'py, i32>,
    ts: PyReadonlyArray1<'py, i32>,
    dt: PyReadonlyArray1<'py, i32>,
    v: PyReadonlyArray1<'py, f64>,
    qc: PyReadonlyArray1<'py, i32>,
    ar_weights: PyReadonlyArray1<'py, f64>,
    rs: Vec<usize>,
    cs: Vec<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, usize)> {
    let (x, w, k, ks) = (x.as_slice()?, w.as_slice()?, k.as_slice()?, ks.as_slice()?);
    let (m, p, ts, dt) = (m.as_slice()?, p.as_slice()?, ts.as_slice()?, dt.as_slice()?);
    let (v, qc, arw) = (v.as_slice()?, qc.as_slice()?, ar_weights.as_slice()?);
    let n = w.len();
    let nx = m.len();
    let nt = ts.len();
    if ks.len() != 2 * nx || p.len() != nx || dt.len() != nt || qc.len() != nt {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xwxd1: inconsistent dimensions",
        ));
    }
    if rs.is_empty() && cs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xwxd1: give the row terms, the column terms or both",
        ));
    }
    if rs.iter().chain(cs.iter()).any(|&t| t >= nt) {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "xwxd1: no such term",
        ));
    }
    for q in 0..nx {
        let (c0, c1) = (ks[q], ks[q + nx]);
        if c0 < 0 || c1 < c0 || (c1 as usize) * n > k.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "xwxd1: bad index columns",
            ));
        }
        let col = &k[c0 as usize * n..c1 as usize * n];
        let (lo, hi) = col
            .iter()
            .fold((i32::MAX, i32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        if !col.is_empty() && (lo < 0 || hi >= m[q]) {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "xwxd1: an index exceeds its marginal's rows",
            ));
        }
    }
    let (rs, cs, symmetric) = if rs.is_empty() {
        (cs.clone(), cs, true)
    } else if cs.is_empty() {
        (rs.clone(), rs, true)
    } else {
        (rs, cs, false)
    };
    let (out, ptot, ctot) =
        py.allow_threads(|| xwxd1(x, w, k, ks, m, p, ts, dt, v, qc, arw, &rs, &cs, symmetric));
    Ok((PyArray1::from_vec(py, out), ptot, ctot))
}

/// mgcv's `XWXd0`: the full `X'WX` (`ptot x ptot`, column-major).
#[pyfunction]
#[pyo3(name = "xwxd0")]
#[allow(clippy::too_many_arguments)]
fn xwxd0_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, i32>,
    ks: PyReadonlyArray1<'py, i32>,
    m: PyReadonlyArray1<'py, i32>,
    p: PyReadonlyArray1<'py, i32>,
    ts: PyReadonlyArray1<'py, i32>,
    dt: PyReadonlyArray1<'py, i32>,
    v: PyReadonlyArray1<'py, f64>,
    qc: PyReadonlyArray1<'py, i32>,
    ar_weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    let (x, w, k, ks) = (x.as_slice()?, w.as_slice()?, k.as_slice()?, ks.as_slice()?);
    let (m, p, ts, dt) = (m.as_slice()?, p.as_slice()?, ts.as_slice()?, dt.as_slice()?);
    let (v, qc, arw) = (v.as_slice()?, qc.as_slice()?, ar_weights.as_slice()?);
    let n = w.len();
    let nx = m.len();
    if ks.len() != 2 * nx || p.len() != nx || ts.len() != dt.len() || ts.len() != qc.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xwxd0: inconsistent dimensions",
        ));
    }
    for q in 0..nx {
        let (c0, c1) = (ks[q], ks[q + nx]);
        if c0 < 0 || c1 < c0 || (c1 as usize) * n > k.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "xwxd0: bad index columns",
            ));
        }
        let col = &k[c0 as usize * n..c1 as usize * n];
        let (lo, hi) = col
            .iter()
            .fold((i32::MAX, i32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        if !col.is_empty() && (lo < 0 || hi >= m[q]) {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "xwxd0: an index exceeds its marginal's rows",
            ));
        }
    }
    let (out, ptot) = py.allow_threads(|| xwxd0(x, w, k, ks, m, p, ts, dt, v, qc, arw));
    Ok((PyArray1::from_vec(py, out), ptot))
}

#[allow(clippy::too_many_arguments)]
fn xwxd0(
    x: &[f64],
    w_in: &[f64],
    k: &[i32],
    ks: &[i32],
    m: &[i32],
    p: &[i32],
    ts: &[i32],
    dt: &[i32],
    v: &[f64],
    qc: &[i32],
    ar_weights: &[f64],
) -> (Vec<f64>, usize) {
    let nx = m.len();
    let nt = ts.len();
    let n = w_in.len();
    let mut pt = vec![0usize; nt];
    let mut pd = vec![0usize; nt];
    let mut off = vec![0usize; nx + 1];
    let mut voff = vec![0usize; nt + 1];
    let mut tps = vec![0usize; nt + 1];
    let mut tpsu = vec![0usize; nt + 1];
    let mut q = 0;
    for i in 0..nt {
        for jj in 0..dt[i] as usize {
            if jj == dt[i] as usize - 1 {
                pd[i] = p[q] as usize;
            }
            off[q + 1] = off[q] + p[q] as usize * m[q] as usize;
            pt[i] = if jj == 0 {
                p[q] as usize
            } else {
                pt[i] * p[q] as usize
            };
            q += 1;
        }
        voff[i + 1] = if qc[i] == 0 {
            voff[i]
        } else if qc[i] > 0 {
            voff[i] + pt[i]
        } else {
            voff[i] + v[voff[i]].round() as usize + 2
        };
        tps[i + 1] = if qc[i] == 0 {
            tps[i] + pt[i]
        } else if qc[i] > 0 {
            tps[i] + pt[i] - 1
        } else {
            let si = v[voff[i]].round() as usize;
            tps[i] + pt[i] - v[voff[i] + si + 1].round() as usize
        };
        tpsu[i + 1] = tpsu[i] + pt[i];
    }
    let ptot = tps[nt];
    let nxwx = tpsu[nt];
    let tri = !ar_weights.is_empty();
    let (w, ws) = if tri {
        let sw: Vec<f64> = w_in.iter().map(|v| v.sqrt()).collect();
        let mut ws = vec![0.0f64; 2 * n - 2];
        for i in 0..n - 1 {
            let val = ar_weights[2 * i + 1] * ar_weights[2 * i + 2] * sw[i + 1] * sw[i];
            ws[i] = val;
            ws[i + n - 1] = val;
        }
        let mut wd = sw.clone();
        for i in 0..n - 1 {
            let a = fma(
                ar_weights[2 * i + 1],
                ar_weights[2 * i + 1],
                ar_weights[2 * i] * ar_weights[2 * i],
            );
            wd[i] *= a * sw[i];
        }
        let i = n - 1;
        wd[i] *= sw[i] * ar_weights[2 * i] * ar_weights[2 * i];
        (wd, ws)
    } else {
        (w_in.to_vec(), Vec::new())
    };
    let cx = Ctx {
        x,
        k,
        ks,
        m,
        p,
        ts,
        dt,
        nx,
        n,
        off,
        w,
        ws,
        tri,
        ht: sm_inihash(),
    };
    let mut xwx = vec![0.0f64; nxwx * nxwx];
    for rb in 0..nt {
        for cb in rb..nt {
            let si = cx.ksn(ts[rb] as usize);
            let rt = pt[rb] / pd[rb];
            let ct = pt[cb] / pd[cb];
            let base = tpsu[rb] + nxwx * tpsu[cb];
            if rb == cb && si == 1 && !tri {
                for r in 0..rt {
                    for c in r..rt {
                        xwxijs(&cx, &mut xwx, base, nxwx, rb, cb, r, c);
                    }
                }
            } else {
                for r in 0..rt {
                    for c in 0..ct {
                        xwxijs(&cx, &mut xwx, base, nxwx, rb, cb, r, c);
                    }
                }
            }
        }
    }
    for r in 0..nt {
        for c in r..nt {
            let pa;
            if dt[r] > 1 && qc[r] != 0 {
                for jj in 0..pt[c] {
                    let o0 = tpsu[r] + (tpsu[c] + jj) * nxwx;
                    let o1 = tps[r] + (tpsu[c] + jj) * nxwx;
                    ztb(&mut xwx, o1, o0, &v[voff[r]..], qc[r], 1, pt[r]);
                }
                pa = pt[r] - 1;
            } else {
                pa = pt[r];
                if tpsu[r] != tps[r] {
                    for jj in 0..pt[c] {
                        let o0 = tpsu[r] + (tpsu[c] + jj) * nxwx;
                        let o1 = tps[r] + (tpsu[c] + jj) * nxwx;
                        for l in 0..pt[r] {
                            xwx[o1 + l] = xwx[o0 + l];
                        }
                    }
                }
            }
            if dt[c] > 1 && qc[c] != 0 {
                for jj in 0..pa {
                    let o0 = tps[r] + jj + tpsu[c] * nxwx;
                    let o1 = tps[r] + jj + tps[c] * nxwx;
                    ztb(&mut xwx, o1, o0, &v[voff[c]..], qc[c], nxwx, pt[c]);
                }
            } else if tpsu[c] != tps[c] {
                for jj in 0..pa {
                    let o0 = tps[r] + jj + tpsu[c] * nxwx;
                    let o1 = tps[r] + jj + tps[c] * nxwx;
                    for l in 0..pt[c] {
                        xwx[o1 + l * nxwx] = xwx[o0 + l * nxwx];
                    }
                }
            }
        }
    }
    let mut out = vec![0.0f64; ptot * ptot];
    for cc in 0..ptot {
        for rr in 0..ptot {
            out[rr + cc * ptot] = xwx[rr + cc * nxwx];
        }
    }
    for i in 0..ptot {
        for jj in i + 1..ptot {
            out[jj + i * ptot] = out[i + jj * ptot];
        }
    }
    (out, ptot)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xwxd0_py, m)?)?;
    m.add_function(wrap_pyfunction!(xwxd1_py, m)?)?;
    Ok(())
}
