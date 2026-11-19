//! mgcv's `gdi1` (gdi.c): at a converged P-IRLS fit, the coefficients and
//! their derivatives with respect to the log smoothing parameters (by the
//! implicit function theorem), the derivatives of the deviance, the Pearson
//! statistic, `tr(A)`, and the REML or ML log determinant terms, as
//! gam.fit3 uses them to form its score. Called on the stable
//! reparameterization: `E` is a square root of the total penalty, `Es` its
//! well scaled version for rank determination, and `rS` the penalty square
//! roots side by side.
//!
//! The arithmetic is that of R's arm64 mgcv.so (clang). Besides the rules
//! [`crate::mgcv_c`] states, the per-observation weight derivatives,
//! Pearson terms and log determinant sums follow its compiled contractions,
//! set out where each is formed.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::lapack::fma;
use crate::mgcv_c::{
    diag_abt, dot_s8, dot_s8_from, drop_cols, drop_rows, frobenius_norm, get_rpqr, get_xtmx,
    get_xtwx, get_xtx, mgcv_backsolve, mgcv_forwardsolve, mgcv_mmult, mgcv_pqrqy, mgcv_qr,
    mgcv_qrqy_left, mgcv_svd_full, pivoter, r_cond, rc_prod, rinv, undrop_rows,
};

/// gdi.c's `multSk`: `y = S_k x` for the `q`-by-`xcol` `x`, through the
/// `k`-th square root in `rs` (`q`-by-`rsncol[k]`, after the earlier ones).
#[allow(clippy::too_many_arguments)]
unsafe fn mult_sk(
    y: *mut f64,
    x: *const f64,
    xcol: i32,
    k: usize,
    rs: *const f64,
    rsncol: &[i32],
    q: i32,
    work: *mut f64,
) {
    let off: usize = rsncol[..k].iter().map(|&c| q as usize * c as usize).sum();
    let rsk = rs.add(off);
    let nc = rsncol[k];
    mgcv_mmult(work, rsk, x, true, false, nc, xcol, q);
    mgcv_mmult(y, rsk, work, false, false, q, xcol, nc);
}

/// gdi.c's `applyP`: `y = P x` for `P = R^{-1}`, or `R^{-1} V` when there
/// are negative weights (`vt` holding `V'`); `x` is `r`-by-`c` (`right`:
/// `c`-by-`r`, and `y = x P`). `R` is the upper triangle of the `nr`-row
/// `rm`.
#[allow(clippy::too_many_arguments)]
unsafe fn apply_p(
    y: *mut f64,
    x: *const f64,
    rm: *const f64,
    vt: *const f64,
    neg_w: bool,
    nr: i32,
    r: i32,
    c: i32,
    right: bool,
) {
    if neg_w {
        let mut x1 = vec![0.0f64; (r as usize * c as usize).max(1)];
        if right {
            mgcv_backsolve(rm, nr, r, x, x1.as_mut_ptr(), c, true);
            mgcv_mmult(y, x1.as_ptr(), vt, false, true, c, r, r);
        } else {
            mgcv_mmult(x1.as_mut_ptr(), vt, x, true, false, r, c, r);
            mgcv_backsolve(rm, nr, r, x1.as_ptr(), y, c, false);
        }
    } else {
        mgcv_backsolve(rm, nr, r, x, y, c, right);
    }
}

/// gdi.c's `applyPt`: `y = P' x`, the transpose of [`apply_p`].
#[allow(clippy::too_many_arguments)]
unsafe fn apply_pt(
    y: *mut f64,
    x: *const f64,
    rm: *const f64,
    vt: *const f64,
    neg_w: bool,
    nr: i32,
    r: i32,
    c: i32,
    right: bool,
) {
    if neg_w {
        let mut x1 = vec![0.0f64; (r as usize * c as usize).max(1)];
        if right {
            mgcv_mmult(x1.as_mut_ptr(), x, vt, false, false, c, r, r);
            mgcv_forwardsolve(rm, nr, r, x1.as_ptr(), y, c, true);
        } else {
            mgcv_forwardsolve(rm, nr, r, x, x1.as_mut_ptr(), c, false);
            mgcv_mmult(y, vt, x1.as_ptr(), false, false, r, c, r);
        }
    } else {
        mgcv_forwardsolve(rm, nr, r, x, y, c, right);
    }
}

/// gdi.c's `ift1`: the first (`b1`, `r`-by-`m`) and second (`b2`, one
/// column per `k >= i` pair) derivatives of the coefficients with respect
/// to the log smoothing parameters by the implicit function theorem, and
/// those of the linear predictor (`eta1`, `eta2`).
#[allow(clippy::too_many_arguments)]
unsafe fn ift1(
    rm: *const f64,
    vt: *const f64,
    x: *const f64,
    rs: *const f64,
    beta: *const f64,
    sp: *const f64,
    dwdeta: *const f64,
    b1: *mut f64,
    b2: *mut f64,
    eta1: *mut f64,
    eta2: *mut f64,
    n: i32,
    r: i32,
    m: i32,
    rsncol: &[i32],
    deriv2: bool,
    neg_w: bool,
    nr: i32,
) {
    let (nu, ru, mu) = (n as usize, r as usize, m as usize);
    let nw = nu.max(ru).max(1);
    let mut work = vec![0.0f64; nw];
    let mut work1 = vec![0.0f64; nw];
    let mut skb = vec![0.0f64; ru.max(1)];
    let n_2d = (mu * (1 + mu)) / 2;
    for i in 0..mu {
        mult_sk(
            skb.as_mut_ptr(),
            beta,
            1,
            i,
            rs,
            rsncol,
            r,
            work.as_mut_ptr(),
        );
        let s = -*sp.add(i);
        for v in skb[..ru].iter_mut() {
            *v *= s;
        }
        apply_pt(
            work.as_mut_ptr(),
            skb.as_ptr(),
            rm,
            vt,
            neg_w,
            nr,
            r,
            1,
            false,
        );
        apply_p(
            b1.add(i * ru),
            work.as_ptr(),
            rm,
            vt,
            neg_w,
            nr,
            r,
            1,
            false,
        );
    }
    mgcv_mmult(eta1, x, b1, false, false, n, m, r);

    if deriv2 {
        let mut pp = b2;
        for i in 0..mu {
            for k in i..mu {
                let p0 = eta1.add(nu * i);
                let p1 = eta1.add(nu * k);
                for j in 0..nu {
                    work[j] = (-*p0.add(j) * *p1.add(j)) * *dwdeta.add(j);
                }
                mgcv_mmult(skb.as_mut_ptr(), x, work.as_ptr(), true, false, r, 1, n);
                mult_sk(
                    work.as_mut_ptr(),
                    b1.add(k * ru),
                    1,
                    i,
                    rs,
                    rsncol,
                    r,
                    work1.as_mut_ptr(),
                );
                let si = *sp.add(i);
                for j in 0..ru {
                    skb[j] = fma(-si, work[j], skb[j]);
                }
                mult_sk(
                    work.as_mut_ptr(),
                    b1.add(i * ru),
                    1,
                    k,
                    rs,
                    rsncol,
                    r,
                    work1.as_mut_ptr(),
                );
                let sk = *sp.add(k);
                for j in 0..ru {
                    skb[j] = fma(-sk, work[j], skb[j]);
                }
                apply_pt(
                    work.as_mut_ptr(),
                    skb.as_ptr(),
                    rm,
                    vt,
                    neg_w,
                    nr,
                    r,
                    1,
                    false,
                );
                apply_p(pp, work.as_ptr(), rm, vt, neg_w, nr, r, 1, false);
                if i == k {
                    for j in 0..ru {
                        *pp.add(j) += *b1.add(i * ru + j);
                    }
                }
                pp = pp.add(ru);
            }
        }
        mgcv_mmult(eta2, x, b2, false, false, n, n_2d as i32, r);
    }
}

/// gdi.c's `get_bSb`: `beta' S beta` and, for `deriv > 0`, its first
/// (`bsb1`) and second (`bsb2`) derivatives with respect to the `m0`
/// likelihood parameters (zero) and the `m` log smoothing parameters.
/// `b2` holds one column per `k >= m` pair. `beta' S beta` accumulates
/// through the output pointer: 8-wide from 12 terms on.
#[allow(clippy::too_many_arguments)]
unsafe fn get_bsb(
    bsb: &mut f64,
    bsb1: *mut f64,
    bsb2: *mut f64,
    sp: *const f64,
    e: *const f64,
    rs: *const f64,
    rsncol: &[i32],
    enrow: i32,
    q: i32,
    m: i32,
    m0: i32,
    beta: *const f64,
    b1: *const f64,
    mut b2: *const f64,
    deriv: i32,
) {
    let (qu, mu, m0u) = (q as usize, m as usize, m0 as usize);
    let mut jw = qu;
    for &c in &rsncol[..mu] {
        jw = jw.max(c as usize);
    }
    jw += m0u;
    let mut work = vec![0.0f64; jw.max(1)];
    let mut sb = vec![0.0f64; qu.max(1)];
    mgcv_mmult(work.as_mut_ptr(), e, beta, false, false, enrow, 1, q);
    mgcv_mmult(sb.as_mut_ptr(), e, work.as_ptr(), true, false, q, 1, enrow);
    *bsb = dot_s8_from(std::slice::from_raw_parts(beta, qu), &sb[..qu], 12);
    if deriv <= 0 {
        return;
    }

    let mut work1 = vec![0.0f64; jw.max(1)];
    let mut skb = vec![0.0f64; (mu * qu).max(1)];
    let mut rsoff = 0usize;
    let beta_s = std::slice::from_raw_parts(beta, qu);
    for i in 0..mu {
        let nc = rsncol[i];
        mgcv_mmult(
            work.as_mut_ptr(),
            rs.add(rsoff),
            beta,
            true,
            false,
            nc,
            1,
            q,
        );
        let s = *sp.add(i);
        for v in work[..nc as usize].iter_mut() {
            *v *= s;
        }
        let p1 = skb.as_mut_ptr().add(i * qu);
        mgcv_mmult(p1, rs.add(rsoff), work.as_ptr(), false, false, q, 1, nc);
        rsoff += qu * nc as usize;
        *bsb1.add(i + m0u) = dot_s8(beta_s, std::slice::from_raw_parts(p1, qu));
    }
    for i in 0..m0u {
        *bsb1.add(i) = 0.0;
    }
    let mtot = mu + m0u;

    if deriv > 1 {
        for mm in 0..mtot {
            mgcv_mmult(
                work1.as_mut_ptr(),
                e,
                b1.add(mm * qu),
                false,
                false,
                enrow,
                1,
                q,
            );
            mgcv_mmult(
                work.as_mut_ptr(),
                e,
                work1.as_ptr(),
                true,
                false,
                q,
                1,
                enrow,
            );
            for k in mm..mtot {
                let km = k * mtot + mm;
                let mk = mm * mtot + k;
                let xx = dot_s8(std::slice::from_raw_parts(b2, qu), &sb[..qu]);
                b2 = b2.add(qu);
                *bsb2.add(km) = xx + xx;
                let xx = dot_s8(&work[..qu], std::slice::from_raw_parts(b1.add(k * qu), qu));
                *bsb2.add(km) = fma(xx, 2.0, *bsb2.add(km));
                if k >= m0u {
                    let xx = dot_s8(
                        std::slice::from_raw_parts(b1.add(mm * qu), qu),
                        &skb[(k - m0u) * qu..(k - m0u) * qu + qu],
                    );
                    *bsb2.add(km) = fma(xx, 2.0, *bsb2.add(km));
                }
                if mm >= m0u {
                    let xx = dot_s8(
                        std::slice::from_raw_parts(b1.add(k * qu), qu),
                        &skb[(mm - m0u) * qu..(mm - m0u) * qu + qu],
                    );
                    *bsb2.add(km) = fma(xx, 2.0, *bsb2.add(km));
                }
                if k == mm {
                    *bsb2.add(km) += *bsb1.add(k);
                } else {
                    *bsb2.add(mk) = *bsb2.add(km);
                }
            }
        }
    }

    mgcv_mmult(
        work.as_mut_ptr(),
        b1,
        sb.as_ptr(),
        true,
        false,
        mtot as i32,
        1,
        q,
    );
    for i in 0..mtot {
        *bsb1.add(i) = fma(2.0, work[i], *bsb1.add(i));
    }
}

/// gdi.c's `pearson2`: the Pearson statistic and its first and second
/// derivatives with respect to the log smoothing parameters, through its
/// derivatives with respect to the linear predictor.
#[allow(clippy::too_many_arguments)]
unsafe fn pearson2(
    p: &mut f64,
    p1: *mut f64,
    p2: *mut f64,
    y: *const f64,
    mu: *const f64,
    v: *const f64,
    v1: *const f64,
    v2: *const f64,
    g1: *const f64,
    g2: *const f64,
    p_weights: *const f64,
    eta1: *const f64,
    eta2: *const f64,
    n: usize,
    m: usize,
    deriv: bool,
    deriv2: bool,
) {
    let n_2d = if deriv2 { (m * (1 + m)) / 2 } else { 0 };
    let mut pe1 = if deriv { vec![0.0f64; n] } else { Vec::new() };
    let mut pi1 = if deriv {
        vec![0.0f64; n * m]
    } else {
        Vec::new()
    };
    let mut pe2 = if deriv2 { vec![0.0f64; n] } else { Vec::new() };
    let mut v2w = if deriv2 { vec![0.0f64; n] } else { Vec::new() };
    let mut pi2 = if deriv2 {
        vec![0.0f64; n_2d * n]
    } else {
        Vec::new()
    };
    *p = 0.0;
    for i in 0..n {
        let resid = *y.add(i) - *mu.add(i);
        let xx = (resid * *p_weights.add(i)) / *v.add(i);
        *p = fma(xx, resid, *p);
        if deriv {
            let g1i = *g1.add(i);
            let v1i = *v1.add(i);
            pe1[i] = -(xx * fma(resid, v1i, 2.0)) / g1i;
            if deriv2 {
                let t = fma(
                    2.0 * xx,
                    v1i,
                    (*p_weights.add(i) + *p_weights.add(i)) / *v.add(i),
                );
                let t = fma(-(pe1[i] * v1i), g1i, t);
                let t = fma(-(xx * resid), fma(-v1i, v1i, *v2.add(i)), t);
                pe2[i] = t / (g1i * g1i) - (*g2.add(i) * pe1[i]) / g1i;
            }
        }
    }

    if deriv {
        rc_prod(pi1.as_mut_ptr(), pe1.as_ptr(), eta1, m as i32, n as i32);
        if deriv2 {
            rc_prod(pi2.as_mut_ptr(), pe1.as_ptr(), eta2, n_2d as i32, n as i32);
            let mut pp = 0usize;
            for mm in 0..m {
                for k in mm..m {
                    rc_prod(
                        pe1.as_mut_ptr(),
                        eta1.add(n * mm),
                        eta1.add(n * k),
                        1,
                        n as i32,
                    );
                    rc_prod(v2w.as_mut_ptr(), pe2.as_ptr(), pe1.as_ptr(), 1, n as i32);
                    for i in 0..n {
                        pi2[pp] += v2w[i];
                        pp += 1;
                    }
                }
            }
        }
        let mut pp = 0usize;
        for k in 0..m {
            let mut xx = 0.0;
            for _ in 0..n {
                xx += pi1[pp];
                pp += 1;
            }
            *p1.add(k) = xx;
        }
        if deriv2 {
            let mut pp = 0usize;
            for mm in 0..m {
                for k in mm..m {
                    let mut xx = 0.0;
                    for _ in 0..n {
                        xx += pi2[pp];
                        pp += 1;
                    }
                    *p2.add(k * m + mm) = xx;
                    *p2.add(mm * m + k) = xx;
                }
            }
        }
    }
}

/// gdi.c's `get_ddetXWXpS`: the first (`det1`) and second (`det2`)
/// derivatives of `log|X'WX + S|` with respect to the `m0` likelihood
/// parameters and `m` log smoothing parameters, from `P` (`q`-by-`r`), `K`
/// (`n`-by-`r`) and the weight derivative ratios `tk`, `tkm`.
#[allow(clippy::too_many_arguments)]
unsafe fn get_ddet_xwxps(
    det1: *mut f64,
    det2: *mut f64,
    p: *const f64,
    k_: *const f64,
    sp: *const f64,
    rs: *const f64,
    rsncol: &[i32],
    tk: *const f64,
    tkm: *const f64,
    n: i32,
    q: i32,
    r: i32,
    m: i32,
    m0: i32,
    deriv: i32,
) {
    if deriv == 0 {
        return;
    }
    let (nu, qu, ru, mu, m0u) = (n as usize, q as usize, r as usize, m as usize, m0 as usize);
    let mtot = m0u + mu;
    let deriv2 = deriv == 2;
    let mut diag_kkt = vec![0.0f64; nu.max(1)];
    diag_abt(diag_kkt.as_mut_ptr(), k_, k_, n, r);
    let work_n = nu.max(ru).max(1);
    let mut work = vec![0.0f64; work_n];
    let mut kt_tk = if deriv2 {
        vec![0.0f64; (ru * ru * mtot).max(1)]
    } else {
        Vec::new()
    };
    if deriv2 {
        for k in 0..mtot {
            get_xtwx(
                kt_tk.as_mut_ptr().add(k * ru * ru),
                k_,
                tk.add(k * nu),
                n,
                r,
            );
        }
    }
    mgcv_mmult(det1, tk, diag_kkt.as_ptr(), true, false, mtot as i32, 1, n);

    let mut max_col = qu;
    for &c in &rsncol[..mu] {
        max_col = max_col.max(c as usize);
    }
    let mut ptr_sm = vec![0.0f64; (ru * max_col).max(1)];
    let mut tr_ptsp = vec![0.0f64; mu.max(1)];
    let mut ptsp = if deriv2 {
        vec![0.0f64; (mu * ru * ru).max(1)]
    } else {
        Vec::new()
    };
    let mut rsoff = vec![0usize; mu.max(1)];
    for mm in 1..mu {
        rsoff[mm] = rsoff[mm - 1] + rsncol[mm - 1] as usize;
    }
    for mm in 0..mu {
        let nc = rsncol[mm];
        mgcv_mmult(
            ptr_sm.as_mut_ptr(),
            p,
            rs.add(rsoff[mm] * qu),
            true,
            false,
            r,
            nc,
            q,
        );
        tr_ptsp[mm] =
            *sp.add(mm) * diag_abt(work.as_mut_ptr(), ptr_sm.as_ptr(), ptr_sm.as_ptr(), r, nc);
        *det1.add(mm + m0u) += tr_ptsp[mm];
        if deriv2 {
            mgcv_mmult(
                ptsp.as_mut_ptr().add(mm * ru * ru),
                ptr_sm.as_ptr(),
                ptr_sm.as_ptr(),
                false,
                true,
                r,
                r,
                nc,
            );
        }
    }

    if deriv2 {
        let rr = ru * ru;
        for mm in 0..mtot {
            let mut ptkm = if mm == 0 {
                tkm
            } else {
                tkm.add((mm * mtot - (mm * (mm - 1)) / 2) * nu)
            };
            for k in mm..mtot {
                let km = k * mtot + mm;
                let mk = mm * mtot + k;
                let xx = dot_s8(std::slice::from_raw_parts(ptkm, nu), &diag_kkt[..nu]);
                ptkm = ptkm.add(nu);
                *det2.add(km) = xx;
                *det2.add(km) -= diag_abt(
                    work.as_mut_ptr(),
                    kt_tk.as_ptr().add(k * rr),
                    kt_tk.as_ptr().add(mm * rr),
                    r,
                    r,
                );
                if k >= m0u && k == mm {
                    *det2.add(km) += tr_ptsp[mm - m0u];
                }
                if mm >= m0u {
                    let tr = diag_abt(
                        work.as_mut_ptr(),
                        kt_tk.as_ptr().add(k * rr),
                        ptsp.as_ptr().add((mm - m0u) * rr),
                        r,
                        r,
                    );
                    *det2.add(km) = fma(-*sp.add(mm - m0u), tr, *det2.add(km));
                }
                if k >= m0u {
                    let tr = diag_abt(
                        work.as_mut_ptr(),
                        kt_tk.as_ptr().add(mm * rr),
                        ptsp.as_ptr().add((k - m0u) * rr),
                        r,
                        r,
                    );
                    *det2.add(km) = fma(-*sp.add(k - m0u), tr, *det2.add(km));
                }
                if k >= m0u && mm >= m0u {
                    let s = *sp.add(mm - m0u) * *sp.add(k - m0u);
                    let tr = diag_abt(
                        work.as_mut_ptr(),
                        ptsp.as_ptr().add((k - m0u) * rr),
                        ptsp.as_ptr().add((mm - m0u) * rr),
                        r,
                        r,
                    );
                    *det2.add(km) = fma(-s, tr, *det2.add(km));
                }
                *det2.add(mk) = *det2.add(km);
            }
        }
    }
}

/// gdi.c's `get_trA2`: `tr(A)` for the influence matrix `A = K K'` (signed
/// by the weights when some are negative), and for `deriv > 0` its first and
/// second derivatives with respect to the log smoothing parameters.
#[allow(clippy::too_many_arguments)]
unsafe fn get_tra2(
    tr_a: &mut f64,
    tr_a1: *mut f64,
    tr_a2: *mut f64,
    p: *const f64,
    k_: *const f64,
    sp: *const f64,
    rs: *const f64,
    rsncol: &[i32],
    tk: *const f64,
    mut tkm: *const f64,
    w: *const f64,
    n: i32,
    q: i32,
    r: i32,
    m: i32,
    deriv: i32,
) {
    let (nu, qu, ru, mu) = (n as usize, q as usize, r as usize, m as usize);
    let deriv2 = deriv == 2;
    let mut ip = vec![0.0f64; nu.max(1)];
    let mut neg_w = false;
    for i in 0..nu {
        if *w.add(i) < 0.0 {
            ip[i] = -1.0;
            neg_w = true;
        } else {
            ip[i] = 1.0;
        }
    }
    let mut diag_kkt = vec![0.0f64; nu.max(1)];
    *tr_a = diag_abt(diag_kkt.as_mut_ptr(), k_, k_, n, r);
    if neg_w {
        // accumulated through the output pointer: 8-wide from 34 terms on
        *tr_a = dot_s8_from(&ip[..nu], &diag_kkt[..nu], 34);
    }
    if deriv == 0 {
        return;
    }

    let n_work = nu.max(ru).max(1);
    let mut work = vec![0.0f64; n_work];
    let mut ktk = vec![0.0f64; (ru * ru).max(1)];
    let mut ipk = vec![0.0f64; (ru * nu).max(1)];
    if neg_w {
        for j in 0..ru {
            for i in 0..nu {
                ipk[i + nu * j] = ip[i] * *k_.add(i + nu * j);
            }
        }
    } else {
        std::ptr::copy_nonoverlapping(k_, ipk.as_mut_ptr(), nu * ru);
    }
    mgcv_mmult(ktk.as_mut_ptr(), k_, ipk.as_ptr(), true, false, r, r, n);
    drop(ipk);
    let mut kktk = vec![0.0f64; (nu * ru).max(1)];
    mgcv_mmult(kktk.as_mut_ptr(), k_, ktk.as_ptr(), false, false, n, r, r);
    let mut diag_kktkkt = vec![0.0f64; nu.max(1)];
    diag_abt(diag_kktkkt.as_mut_ptr(), kktk.as_ptr(), k_, n, r);

    let rr = ru * ru;
    let (mut kt_tk, mut kt_tkktk) = if deriv2 {
        (
            vec![0.0f64; (rr * mu).max(1)],
            vec![0.0f64; (rr * mu).max(1)],
        )
    } else {
        (Vec::new(), Vec::new())
    };
    if deriv2 {
        for k in 0..mu {
            get_xtwx(kt_tk.as_mut_ptr().add(k * rr), k_, tk.add(k * nu), n, r);
            mgcv_mmult(
                kt_tkktk.as_mut_ptr().add(k * rr),
                kt_tk.as_ptr().add(k * rr),
                ktk.as_ptr(),
                false,
                false,
                r,
                r,
                r,
            );
        }
    }

    mgcv_mmult(tr_a1, tk, diag_kkt.as_ptr(), true, false, m, 1, n);
    mgcv_mmult(
        work.as_mut_ptr(),
        tk,
        diag_kktkkt.as_ptr(),
        true,
        false,
        m,
        1,
        n,
    );
    for i in 0..mu {
        *tr_a1.add(i) += -work[i];
    }

    if deriv2 {
        let diff: Vec<f64> = (0..nu).map(|i| diag_kkt[i] - diag_kktkkt[i]).collect();
        for mm in 0..mu {
            for k in mm..mu {
                let km = k * mu + mm;
                let mk = mm * mu + k;
                let xx = dot_s8(std::slice::from_raw_parts(tkm, nu), &diff);
                tkm = tkm.add(nu);
                *tr_a2.add(km) = xx;
                let tr = diag_abt(
                    work.as_mut_ptr(),
                    kt_tk.as_ptr().add(k * rr),
                    kt_tk.as_ptr().add(mm * rr),
                    r,
                    r,
                );
                *tr_a2.add(km) = fma(-2.0, tr, *tr_a2.add(km));
                let tr = diag_abt(
                    work.as_mut_ptr(),
                    kt_tk.as_ptr().add(k * rr),
                    kt_tkktk.as_ptr().add(mm * rr),
                    r,
                    r,
                );
                *tr_a2.add(km) += tr + tr;
                *tr_a2.add(mk) = *tr_a2.add(km);
            }
        }
    }
    drop(kt_tkktk);
    drop(kt_tk);

    let mut ptr_sm = vec![0.0f64; (ru * qu).max(1)];
    let mut kptr_sm = vec![0.0f64; (nu * qu).max(1)];
    let mut diag_kptspkt = vec![0.0f64; (nu * mu).max(1)];
    let (mut ptsp, mut ptspktk, mut ktkptr_sm, mut kktkptr_sm, mut diag_kptspktkkt) = if deriv2 {
        (
            vec![0.0f64; (mu * rr).max(1)],
            vec![0.0f64; (mu * rr).max(1)],
            vec![0.0f64; (ru * qu).max(1)],
            vec![0.0f64; (nu * qu).max(1)],
            vec![0.0f64; (nu * mu).max(1)],
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };
    let mut rsoff = vec![0usize; mu.max(1)];
    for mm in 1..mu {
        rsoff[mm] = rsoff[mm - 1] + rsncol[mm - 1] as usize;
    }
    for mm in 0..mu {
        let nc = rsncol[mm];
        mgcv_mmult(
            ptr_sm.as_mut_ptr(),
            p,
            rs.add(rsoff[mm] * qu),
            true,
            false,
            r,
            nc,
            q,
        );
        mgcv_mmult(
            kptr_sm.as_mut_ptr(),
            k_,
            ptr_sm.as_ptr(),
            false,
            false,
            n,
            nc,
            r,
        );
        if deriv2 {
            mgcv_mmult(
                ktkptr_sm.as_mut_ptr(),
                ktk.as_ptr(),
                ptr_sm.as_ptr(),
                false,
                false,
                r,
                nc,
                r,
            );
            mgcv_mmult(
                ptsp.as_mut_ptr().add(mm * rr),
                ptr_sm.as_ptr(),
                ptr_sm.as_ptr(),
                false,
                true,
                r,
                r,
                nc,
            );
            mgcv_mmult(
                kktkptr_sm.as_mut_ptr(),
                kktk.as_ptr(),
                ptr_sm.as_ptr(),
                false,
                false,
                n,
                nc,
                r,
            );
            mgcv_mmult(
                ptspktk.as_mut_ptr().add(mm * rr),
                ptr_sm.as_ptr(),
                ktkptr_sm.as_ptr(),
                false,
                true,
                r,
                r,
                nc,
            );
            diag_abt(
                diag_kptspktkkt.as_mut_ptr().add(mm * nu),
                kptr_sm.as_ptr(),
                kktkptr_sm.as_ptr(),
                n,
                nc,
            );
        }
        let mut xx = *sp.add(mm)
            * diag_abt(
                diag_kptspkt.as_mut_ptr().add(mm * nu),
                kptr_sm.as_ptr(),
                kptr_sm.as_ptr(),
                n,
                nc,
            );
        if neg_w {
            xx = dot_s8(&diag_kptspkt[mm * nu..mm * nu + nu], &ip[..nu]);
            xx *= *sp.add(mm);
        }
        *tr_a1.add(mm) -= xx;
        if deriv2 {
            *tr_a2.add(mm * mu + mm) -= xx;
        }
    }
    if !deriv2 {
        return;
    }
    for mm in 0..mu {
        for k in mm..mu {
            let km = k * mu + mm;
            let mk = mm * mu + k;
            let tk_k = std::slice::from_raw_parts(tk.add(k * nu), nu);
            let tk_m = std::slice::from_raw_parts(tk.add(mm * nu), nu);
            let (spm, spk) = (*sp.add(mm), *sp.add(k));
            let xx = dot_s8(&diag_kptspktkkt[mm * nu..mm * nu + nu], tk_k);
            *tr_a2.add(km) = fma(spm + spm, xx, *tr_a2.add(km));
            let xx = dot_s8(&diag_kptspktkkt[k * nu..k * nu + nu], tk_m);
            *tr_a2.add(km) = fma(spk + spk, xx, *tr_a2.add(km));
            let xx = dot_s8(&diag_kptspkt[mm * nu..mm * nu + nu], tk_k);
            *tr_a2.add(km) = fma(-spm, xx, *tr_a2.add(km));
            let xx = dot_s8(&diag_kptspkt[k * nu..k * nu + nu], tk_m);
            *tr_a2.add(km) = fma(-spk, xx, *tr_a2.add(km));
            let tr = diag_abt(
                work.as_mut_ptr(),
                ptsp.as_ptr().add(mm * rr),
                ptspktk.as_ptr().add(k * rr),
                r,
                r,
            );
            *tr_a2.add(km) = fma((spk + spk) * spm, tr, *tr_a2.add(km));
            *tr_a2.add(mk) = *tr_a2.add(km);
        }
    }
}

/// `1 - 2 d^2` for each singular value of the negative-weight rows,
/// `(1 - 2 d^2)^{-1/2}` (zero when not positive) left in `d`, and
/// `log|I - 2 D^2|` returned; `work` (when given) receives the square
/// roots.
unsafe fn neg_w_scaling(d: &mut [f64], mut work: Option<&mut [f64]>) -> f64 {
    let mut ldet = 0.0;
    for i in 0..d.len() {
        let v = fma(-2.0 * d[i], d[i], 1.0);
        if v <= 0.0 {
            d[i] = 0.0;
            if let Some(w) = work.as_deref_mut() {
                w[i] = 0.0;
            }
        } else {
            ldet += v.ln();
            let s = v.sqrt();
            if let Some(w) = work.as_deref_mut() {
                w[i] = s;
            }
            d[i] = 1.0 / s;
        }
    }
    ldet
}

/// gdi.c's `MLpenalty1`: `log|X'WX + S|` projected into the penalty range
/// space, as ML needs it, with its derivatives through
/// [`get_ddet_xwxps`]. Works on the rank-truncated, pivoted problem:
/// `nulli` marks null space parameters (positive), `rm` is the `q`-by-`q`
/// QR factor and `qm1` its `n`-by-`q` orthogonal factor. `rs` is
/// overwritten.
#[allow(clippy::too_many_arguments)]
unsafe fn ml_penalty1(
    det1: *mut f64,
    det2: *mut f64,
    tk: *const f64,
    tkm: *const f64,
    nulli: &[f64],
    rm: *const f64,
    qm1: *const f64,
    nind: &[i32],
    sp: *const f64,
    rs: *mut f64,
    rsncol: &[i32],
    q: i32,
    n: i32,
    m: i32,
    m0: i32,
    deriv: i32,
    x: *const f64,
    type1: bool,
) -> f64 {
    let (qu, nu) = (q as usize, n as usize);
    let neg_w = nind.len();
    let drop: Vec<i32> = (0..qu)
        .filter(|&i| nulli[i] > 0.0)
        .map(|i| i as i32)
        .collect();
    let n_drop = drop.len();
    let scs: i32 = rsncol[..m as usize].iter().sum();
    let qm = qu - n_drop;
    let qmi = qm as i32;

    let mut ru1 = vec![0.0f64; (qu * qu).max(1)];
    std::ptr::copy_nonoverlapping(rm, ru1.as_mut_ptr(), qu * qu);
    drop_cols(ru1.as_mut_ptr(), q, q, &drop);
    let mut tau = vec![0.0f64; qm.max(1)];
    let mut pivot = vec![0i32; qm.max(1)];
    mgcv_qr(
        ru1.as_mut_ptr(),
        q,
        qmi,
        pivot.as_mut_ptr(),
        tau.as_mut_ptr(),
    );
    let mut ri = vec![0.0f64; (qm * qm).max(1)];
    rinv(ri.as_mut_ptr(), ru1.as_ptr(), qmi, q, qmi);
    // the new Q factor for type 0 or negative weights, else X without the
    // null space columns
    let mut qb = vec![0.0f64; (qu * qm).max(1)];
    let mut xu1 = Vec::new();
    if !type1 || neg_w > 0 {
        for i in 0..qm {
            qb[i * qu + i] = 1.0;
        }
        mgcv_qrqy_left(
            qb.as_mut_ptr(),
            ru1.as_ptr(),
            tau.as_ptr(),
            q,
            qmi,
            qmi,
            false,
        );
    } else {
        xu1 = vec![0.0f64; (nu * qu).max(1)];
        std::ptr::copy_nonoverlapping(x, xu1.as_mut_ptr(), nu * qu);
        drop_cols(xu1.as_mut_ptr(), n, q, &drop);
    }

    let mut k_ = vec![0.0f64; (nu * qm).max(1)];
    let mut p = vec![0.0f64; (qm * qm).max(1)];
    let mut ldet_i2d = 0.0;
    if neg_w > 0 {
        let k = (qu + 1).max(neg_w);
        let mut iq = vec![0.0f64; k * qu];
        for i in 0..neg_w {
            for j in 0..qu {
                iq[i + k * j] = *qm1.add(nind[i] as usize + nu * j);
            }
        }
        let mut iqq = vec![0.0f64; (k * qm).max(1)];
        mgcv_mmult(
            iqq.as_mut_ptr(),
            iq.as_ptr(),
            qb.as_ptr(),
            false,
            false,
            k as i32,
            qmi,
            q,
        );
        let mut vt = vec![0.0f64; (qm * qm).max(1)];
        let mut d = vec![0.0f64; qm];
        mgcv_svd_full(
            iqq.as_mut_ptr(),
            vt.as_mut_ptr(),
            d.as_mut_ptr(),
            k as i32,
            qmi,
        );
        ldet_i2d = neg_w_scaling(&mut d, None);
        for i in 0..qm {
            for j in 0..qm {
                vt[j + qm * i] *= d[j];
            }
        }
        mgcv_mmult(
            p.as_mut_ptr(),
            ri.as_ptr(),
            vt.as_ptr(),
            false,
            true,
            qmi,
            qmi,
            qmi,
        );
        if type1 {
            // mgcv forms K from XU1 here, which it allocated only without
            // negative weights
            return f64::NAN;
        }
        let mut work = vec![0.0f64; (qu * qm).max(1)];
        mgcv_mmult(
            work.as_mut_ptr(),
            qb.as_ptr(),
            vt.as_ptr(),
            false,
            true,
            q,
            qmi,
            qmi,
        );
        mgcv_mmult(k_.as_mut_ptr(), qm1, work.as_ptr(), false, false, n, qmi, q);
    } else {
        p[..qm * qm].copy_from_slice(&ri[..qm * qm]);
        if type1 {
            mgcv_mmult(
                k_.as_mut_ptr(),
                xu1.as_ptr(),
                p.as_ptr(),
                false,
                false,
                n,
                qmi,
                qmi,
            );
        } else {
            mgcv_mmult(k_.as_mut_ptr(), qm1, qb.as_ptr(), false, false, n, qmi, q);
        }
    }

    let mut ldet = 0.0;
    for i in 0..qm {
        ldet += ru1[i + i * qu].abs().ln();
    }
    ldet += ldet;
    ldet += ldet_i2d;

    drop_rows(rs, q, scs, &drop);
    pivoter(rs, qmi, scs, pivot.as_ptr(), false, false);
    if deriv != 0 {
        get_ddet_xwxps(
            det1,
            det2,
            p.as_ptr(),
            k_.as_ptr(),
            sp,
            rs,
            rsncol,
            tk,
            tkm,
            n,
            qmi,
            qmi,
            m,
            m0,
            deriv,
        );
    }
    ldet
}

/// What [`gdi_pk`] leaves for the rest of `gdi1`.
struct Pk {
    /// `[R; E]` after the final QR (`nr` rows), its factor `R` in the upper
    /// triangle.
    r: Vec<f64>,
    nr: usize,
    rank: usize,
    /// The final QR's pivots (0-based).
    pivot1: Vec<i32>,
    /// The columns dropped as unidentifiable, ascending.
    drop: Vec<i32>,
    /// `+1` for a parameter in the penalty null space, `-1` otherwise.
    nulli: Vec<f64>,
    /// `X'WX` from the first QR, pivoted (`rank`-by-`rank`), when second
    /// derivatives are wanted.
    dev_hess: Vec<f64>,
    /// `Rh` with `Rh'Rh = X'WX + S`.
    rh: Vec<f64>,
    /// `P` (`rank`-by-`rank`) and `K` (`n`-by-`rank`) of Wood (2008).
    p: Vec<f64>,
    k: Vec<f64>,
    /// `(I - 2 D^2)^{-1/2} V'` for negative weights.
    vt: Vec<f64>,
    /// The pivoted coefficients.
    pktz: Vec<f64>,
    /// `Q1 = Qf Qs[1:q, ]`, `n`-by-`rank`.
    q1: Vec<f64>,
    ldet_xwxs: f64,
}

/// gdi.c's `gdiPK`: the QR of `sqrt|W| X`, the rank of the penalized
/// problem from the well scaled `[R; Es]`, the unidentifiable columns
/// dropped from `x`, `e` and the rows of `rs`, the QR of `[R; E]`, and `P`,
/// `K`, `Rh` and the coefficients, corrected for negative weights. With
/// `type1` (gdi2's type 1, for zero weights) `z` holds `Wz` and `K = XP`,
/// the coefficients `P K'Wz`.
#[allow(clippy::too_many_arguments)]
unsafe fn gdi_pk(
    x: *mut f64,
    e: *mut f64,
    es: *const f64,
    rs: *mut f64,
    z: *const f64,
    raw: &[f64],
    nind: &[i32],
    n: usize,
    q: usize,
    mp: usize,
    enrow: usize,
    deriv2: bool,
    scs: usize,
    reml: i32,
    rank_tol: f64,
    type1: bool,
) -> Pk {
    let neg_w = nind.len();
    let (ni, qi, ei) = (n as i32, q as i32, enrow as i32);
    // type 0: z = [sqrt(|W|)z, 0]; type 1: z holds Wz on entry
    let mut zz = vec![0.0f64; n.max(1)];
    if !type1 {
        for i in 0..n {
            zz[i] = *z.add(i) * raw[i];
        }
        for &k in nind {
            zz[k as usize] = -zz[k as usize];
        }
    }
    let mut wx = vec![0.0f64; (n * q).max(1)];
    for j in 0..q {
        for i in 0..n {
            wx[i + n * j] = raw[i] * *x.add(i + n * j);
        }
    }
    let mut tau = vec![0.0f64; q.max(1)];
    let mut pivot = vec![0i32; q.max(1)];
    mgcv_qr(
        wx.as_mut_ptr(),
        ni,
        qi,
        pivot.as_mut_ptr(),
        tau.as_mut_ptr(),
    );

    let rr = q.min(n);
    let rri = rr as i32;
    let mut r1 = vec![0.0f64; (rr * q).max(1)];
    get_rpqr(r1.as_mut_ptr(), wx.as_ptr(), ni, qi, rri);
    pivoter(r1.as_mut_ptr(), rri, qi, pivot.as_ptr(), true, true);

    let rnorm = frobenius_norm(r1.as_ptr(), rri, qi);
    let enorm = frobenius_norm(es, ei, qi);
    let nr = rr + enrow;
    let nri = nr as i32;
    let mut r = vec![0.0f64; (nr * q).max(1)];
    for j in 0..q {
        for i in 0..rr {
            r[i + nr * j] = r1[i + rr * j] / rnorm;
        }
        for i in 0..enrow {
            r[i + rr + nr * j] = *es.add(i + enrow * j) / enorm;
        }
    }
    let mut tau1 = vec![0.0f64; q.max(1)];
    let mut pivot1 = vec![0i32; q.max(1)];
    mgcv_qr(
        r.as_mut_ptr(),
        nri,
        qi,
        pivot1.as_mut_ptr(),
        tau1.as_mut_ptr(),
    );
    let mut rank = q.min(nr);
    let mut rcond = r_cond(r.as_ptr(), nri, rank as i32);
    while rank_tol * rcond > 1.0 {
        rank -= 1;
        rcond = r_cond(r.as_ptr(), nri, rank as i32);
    }
    let ranki = rank as i32;

    let mut nulli = vec![0.0f64; q.max(1)];
    for (i, v) in nulli[..q].iter_mut().enumerate() {
        *v = if i < q - mp { -1.0 } else { 1.0 };
    }
    let n_drop = q - rank;
    let mut drop: Vec<i32> = pivot1[rank..rank + n_drop].to_vec();
    drop.sort_unstable();
    if n_drop > 0 {
        drop_cols(r1.as_mut_ptr(), rri, qi, &drop);
        drop_cols(e, ei, qi, &drop);
        drop_cols(x, ni, qi, &drop);
        drop_rows(rs, qi, scs as i32, &drop);
        drop_rows(nulli.as_mut_ptr(), qi, 1, &drop);
    }

    for j in 0..rank {
        for i in 0..rr {
            r[i + nr * j] = r1[i + rr * j];
        }
        for i in 0..enrow {
            r[i + rr + nr * j] = *e.add(i + enrow * j);
        }
    }
    mgcv_qr(
        r.as_mut_ptr(),
        nri,
        ranki,
        pivot1.as_mut_ptr(),
        tau1.as_mut_ptr(),
    );
    pivoter(nulli.as_mut_ptr(), ranki, 1, pivot1.as_ptr(), false, false);

    let mut dev_hess = Vec::new();
    if deriv2 {
        pivoter(r1.as_mut_ptr(), rri, ranki, pivot1.as_ptr(), true, false);
        dev_hess = vec![0.0f64; (q * q).max(1)];
        get_xtx(dev_hess.as_mut_ptr(), r1.as_ptr(), rri, ranki);
    }

    // Q1 = Qf Qs[1:q, ]: needed for the negative-w correction and for type 0
    let mut q1 = vec![0.0f64; (n * q).max(1)];
    if neg_w > 0 || !type1 {
        let mut qm = vec![0.0f64; (nr * rank).max(1)];
        for i in 0..rank {
            qm[i * nr + i] = 1.0;
        }
        mgcv_qrqy_left(
            qm.as_mut_ptr(),
            r.as_ptr(),
            tau1.as_ptr(),
            nri,
            ranki,
            ranki,
            false,
        );
        for i in 0..rr {
            for j in 0..rank {
                q1[i + rr * j] = qm[i + nr * j];
            }
        }
        mgcv_pqrqy(
            q1.as_mut_ptr(),
            wx.as_ptr(),
            tau.as_ptr(),
            ni,
            rri,
            ranki,
            false,
        );
    }

    let mut ri = vec![0.0f64; (rank * rank).max(1)];
    rinv(ri.as_mut_ptr(), r.as_ptr(), ranki, nri, ranki);

    let mut p = vec![0.0f64; (q * q).max(1)];
    let mut k_ = vec![0.0f64; (n * q).max(1)];
    let mut rh = vec![0.0f64; (q * q).max(1)];
    let mut vt = Vec::new();
    let mut ldet_i2d = 0.0;
    if neg_w > 0 {
        vt = vec![0.0f64; (q * q).max(1)];
        let k = (rank + 1).max(neg_w);
        let mut iq = vec![0.0f64; k * rank];
        for i in 0..neg_w {
            for j in 0..rank {
                iq[i + k * j] = q1[nind[i] as usize + n * j];
            }
        }
        let mut d = vec![0.0f64; rank];
        mgcv_svd_full(
            iq.as_mut_ptr(),
            vt.as_mut_ptr(),
            d.as_mut_ptr(),
            k as i32,
            ranki,
        );
        if deriv2 {
            // P temporarily holds D V'R, R upper triangular
            for j in 0..rank {
                let p1 = p.as_mut_ptr().add(j * rank);
                let rj = *r.as_ptr().add(j * nr);
                for i in 0..rank {
                    *p1.add(i) = vt[i] * rj;
                }
                for kk in 1..=j {
                    let rkj = r[kk + j * nr];
                    for i in 0..rank {
                        *p1.add(i) = fma(vt[i + rank * kk], rkj, *p1.add(i));
                    }
                }
            }
            for j in 0..rank {
                for i in 0..rank {
                    p[i + rank * j] *= d[i];
                }
            }
            get_xtx(k_.as_mut_ptr(), p.as_ptr(), ranki, ranki);
            for i in 0..rank * rank {
                dev_hess[i] = fma(-2.0, k_[i], dev_hess[i]);
            }
        }
        let mut work = vec![0.0f64; rank];
        ldet_i2d = neg_w_scaling(&mut d, Some(&mut work));
        // the upper triangle of R into P
        for i in 0..rank {
            for kk in 0..rank {
                p[kk + rank * i] = if kk <= i { r[kk + i * nr] } else { 0.0 };
            }
        }
        mgcv_mmult(
            rh.as_mut_ptr(),
            vt.as_ptr(),
            p.as_ptr(),
            false,
            false,
            ranki,
            ranki,
            ranki,
        );
        for i in 0..rank {
            for j in 0..rank {
                rh[j + rank * i] *= work[j];
            }
        }
        for i in 0..rank {
            for j in 0..rank {
                vt[j + rank * i] *= d[j];
            }
        }
        mgcv_mmult(
            p.as_mut_ptr(),
            ri.as_ptr(),
            vt.as_ptr(),
            false,
            true,
            ranki,
            ranki,
            ranki,
        );
        if !type1 {
            mgcv_mmult(
                k_.as_mut_ptr(),
                q1.as_ptr(),
                vt.as_ptr(),
                false,
                true,
                ni,
                ranki,
                ranki,
            );
        }
    } else {
        if !type1 {
            k_[..n * rank].copy_from_slice(&q1[..n * rank]);
        }
        p[..rank * rank].copy_from_slice(&ri[..rank * rank]);
        for i in 0..rank {
            for kk in 0..rank {
                rh[kk + rank * i] = if kk <= i { r[kk + i * nr] } else { 0.0 };
            }
        }
    }

    let mut ldet_xwxs = 0.0;
    if reml > 0 {
        for i in 0..rank {
            ldet_xwxs += r[i + i * nr].abs().ln();
        }
        ldet_xwxs += ldet_xwxs;
        ldet_xwxs += ldet_i2d;
    }

    pivoter(rs, ranki, scs as i32, pivot1.as_ptr(), false, false);
    pivoter(e, ei, ranki, pivot1.as_ptr(), true, false);
    pivoter(x, ni, ranki, pivot1.as_ptr(), true, false);

    let vtp = if neg_w > 0 {
        vt.as_ptr()
    } else {
        std::ptr::null()
    };
    let mut work = vec![0.0f64; 3 * q.max(1)];
    let mut pktz = vec![0.0f64; q.max(1)];
    if type1 {
        // K = X P, then P K'Wz
        apply_p(
            k_.as_mut_ptr(),
            x,
            r.as_ptr(),
            vtp,
            neg_w > 0,
            nri,
            ranki,
            ni,
            true,
        );
        mgcv_mmult(work.as_mut_ptr(), k_.as_ptr(), z, true, false, ranki, 1, ni);
        apply_p(
            pktz.as_mut_ptr(),
            work.as_ptr(),
            r.as_ptr(),
            vtp,
            neg_w > 0,
            nri,
            ranki,
            1,
            false,
        );
        return Pk {
            r,
            nr,
            rank,
            pivot1,
            drop,
            nulli,
            dev_hess,
            rh,
            p,
            k: k_,
            vt,
            pktz,
            q1,
            ldet_xwxs,
        };
    }
    mgcv_mmult(
        work.as_mut_ptr(),
        k_.as_ptr(),
        zz.as_ptr(),
        true,
        false,
        ranki,
        1,
        ni,
    );
    mgcv_mmult(
        work.as_mut_ptr().add(2 * q),
        q1.as_ptr(),
        zz.as_ptr(),
        true,
        false,
        ranki,
        1,
        ni,
    );
    for i in 0..n {
        zz[i] = raw[i] * raw[i] * *z.add(i);
    }
    for &k in nind {
        zz[k as usize] = -zz[k as usize];
    }
    mgcv_mmult(
        work.as_mut_ptr().add(q),
        x,
        zz.as_ptr(),
        true,
        false,
        ranki,
        1,
        ni,
    );
    let (mut norm1, mut norm2) = (0.0f64, 0.0f64);
    for i in 0..rank {
        let mut xx = dot_s8(&r[nr * i..nr * i + i + 1], &work[2 * q..2 * q + i + 1]);
        xx -= work[i + q];
        norm1 = fma(xx, xx, norm1);
        norm2 = fma(work[i + q], work[i + q], norm2);
    }
    if norm1 > rank_tol * norm2 {
        apply_pt(
            zz.as_mut_ptr(),
            work.as_ptr().add(q),
            r.as_ptr(),
            vtp,
            neg_w > 0,
            nri,
            ranki,
            1,
            false,
        );
        apply_p(
            pktz.as_mut_ptr(),
            zz.as_ptr(),
            r.as_ptr(),
            vtp,
            neg_w > 0,
            nri,
            ranki,
            1,
            false,
        );
    } else {
        apply_p(
            pktz.as_mut_ptr(),
            work.as_ptr(),
            r.as_ptr(),
            vtp,
            neg_w > 0,
            nri,
            ranki,
            1,
            false,
        );
    }

    Pk {
        r,
        nr,
        rank,
        pivot1,
        drop,
        nulli,
        dev_hess,
        rh,
        p,
        k: k_,
        vt,
        pktz,
        q1,
        ldet_xwxs,
    }
}

/// Everything `.C(C_gdi1, ...)` takes, owned; the outputs are written back
/// in place, as the C writes through its pointer arguments.
pub(crate) struct Gdi1Args {
    pub x: Vec<f64>,
    pub e: Vec<f64>,
    pub es: Vec<f64>,
    pub rs: Vec<f64>,
    pub sp: Vec<f64>,
    pub z: Vec<f64>,
    pub w: Vec<f64>,
    pub wf: Vec<f64>,
    pub alpha: Vec<f64>,
    pub mu: Vec<f64>,
    pub y: Vec<f64>,
    pub p_weights: Vec<f64>,
    pub g1: Vec<f64>,
    pub g2: Vec<f64>,
    pub g3: Vec<f64>,
    pub g4: Vec<f64>,
    pub v0: Vec<f64>,
    pub v1: Vec<f64>,
    pub v2: Vec<f64>,
    pub v3: Vec<f64>,
    pub beta: Vec<f64>,
    pub b1: Vec<f64>,
    pub w1: Vec<f64>,
    pub d1: Vec<f64>,
    pub d2: Vec<f64>,
    pub p0: f64,
    pub p1: Vec<f64>,
    pub p2: Vec<f64>,
    pub tr_a: f64,
    pub tr_a1: Vec<f64>,
    pub tr_a2: Vec<f64>,
    pub rv: Vec<f64>,
    pub rank_tol: f64,
    pub conv_tol: f64,
    pub rank_est: i32,
    pub n: usize,
    pub q: usize,
    pub m: usize,
    pub mp: usize,
    pub enrow: usize,
    pub rsncol: Vec<i32>,
    pub deriv: i32,
    pub reml: i32,
    pub fisher: bool,
    pub dvkk: Vec<f64>,
}

/// gdi.c's `gdi1`, one thread.
pub(crate) unsafe fn gdi1(a: &mut Gdi1Args) {
    let (n, q, m, mp, enrow) = (a.n, a.q, a.m, a.mp, a.enrow);
    let (ni, mi) = (n as i32, m as i32);
    let deriv = a.deriv;
    let deriv2 = deriv == 2;
    let reml = a.reml;
    let fisher = a.fisher;
    let scs: usize = a.rsncol[..m].iter().map(|&c| c as usize).sum();
    let rsncol = a.rsncol.clone();

    let mut raw = vec![0.0f64; n.max(1)];
    let mut nind = Vec::new();
    for i in 0..n {
        if a.w[i] < 0.0 {
            nind.push(i as i32);
            raw[i] = (-a.w[i]).sqrt();
        } else {
            raw[i] = a.w[i].sqrt();
        }
    }
    let neg_w = !nind.is_empty();

    let mut pk = gdi_pk(
        a.x.as_mut_ptr(),
        a.e.as_mut_ptr(),
        a.es.as_ptr(),
        a.rs.as_mut_ptr(),
        a.z.as_ptr(),
        &raw,
        &nind,
        n,
        q,
        mp,
        enrow,
        deriv2,
        scs,
        reml,
        a.rank_tol,
        false,
    );
    let rank = pk.rank;
    let ranki = rank as i32;
    let nr = pk.nr;
    let vtp = if neg_w {
        pk.vt.as_ptr()
    } else {
        std::ptr::null()
    };

    let mut n_work = (4 * n + 2 * q) * m + 2 * n;
    n_work = n_work.max(5 * q).max((m * (1 + m)) / 2 * n);
    let mut work = vec![0.0f64; n_work.max(1)];

    let n_2d = (m * (1 + m)) / 2;
    let mut b2 = Vec::new();
    let mut eta1 = Vec::new();
    let mut eta2 = Vec::new();
    let mut tk = Vec::new();
    let mut tkm = Vec::new();
    let mut w2 = Vec::new();
    let mut a1 = Vec::new();
    let mut a2 = Vec::new();
    let mut af1 = Vec::new();
    let mut af2 = Vec::new();
    let mut dev_grad = Vec::new();
    let mut tfk = Vec::new();
    let mut tfkm = Vec::new();
    if deriv != 0 {
        b2 = vec![0.0f64; (rank * n_2d).max(1)];
        eta1 = vec![0.0f64; (n * m).max(1)];
        tk = vec![0.0f64; (n * m).max(1)];
        eta2 = vec![0.0f64; (n * n_2d).max(1)];
        tkm = vec![0.0f64; (n * n_2d).max(1)];
        w2 = vec![0.0f64; (n * n_2d).max(1)];
        a1 = vec![0.0f64; n.max(1)];
        a2 = vec![0.0f64; n.max(1)];
        if fisher {
            for i in 0..n {
                let (w, v1, g1, g2) = (a.w[i], a.v1[i], a.g1[i], a.g2[i]);
                a1[i] = -(w * fma(g2, 2.0, v1)) / g1;
            }
            for i in 0..n {
                let (w, v1, g1, g2) = (a.w[i], a.v1[i], a.g1[i], a.g2[i]);
                let t = fma(-2.0 * g2, g2, fma(a.g3[i], 2.0, fma(-v1, v1, a.v2[i])));
                a2[i] = fma(a1[i], a1[i] / w - g2 / g1, -(w * t) / (g1 * g1));
            }
        } else {
            let mut alpha1 = vec![0.0f64; n];
            let mut alpha2 = vec![0.0f64; n];
            for i in 0..n {
                let (v1, v2, g2, g3) = (a.v1[i], a.v2[i], a.g2[i], a.g3[i]);
                let xx = fma(-g2, g2, fma(-v1, v1, v2) + g3);
                let ymu = a.y[i] - a.mu[i];
                alpha1[i] = fma(ymu, xx, -(v1 + g2)) / a.alpha[i];
                let mut t = fma(v1 * -3.0, v2, a.v3[i]);
                t = fma(v1 * (v1 + v1), v1, t);
                t = a.g4[i] + t;
                t = fma(g3 * -3.0, g2, t);
                t = fma(g2 * (g2 + g2), g2, t);
                alpha2[i] = fma(xx, -2.0, ymu * t) / a.alpha[i];
            }
            for i in 0..n {
                a1[i] = (a.w[i] * fma(a.g2[i], -2.0, alpha1[i] - a.v1[i])) / a.g1[i];
            }
            for i in 0..n {
                let (w, v1, g1, g2) = (a.w[i], a.v1[i], a.g1[i], a.g2[i]);
                let t = fma(alpha1[i], alpha1[i], -alpha2[i]) + a.v2[i];
                let t = fma(-2.0 * g2, g2, fma(a.g3[i], 2.0, fma(-v1, v1, t)));
                a2[i] = fma(a1[i], a1[i] / w - g2 / g1, -(w * t) / (g1 * g1));
            }
            if reml == 0 {
                af1 = vec![0.0f64; n];
                af2 = vec![0.0f64; n];
                for i in 0..n {
                    let (wf, v1, g1, g2) = (a.wf[i], a.v1[i], a.g1[i], a.g2[i]);
                    af1[i] = -(wf * fma(g2, 2.0, v1)) / g1;
                }
                for i in 0..n {
                    let (wf, v1, g1, g2) = (a.wf[i], a.v1[i], a.g1[i], a.g2[i]);
                    let t = fma(-2.0 * g2, g2, fma(a.g3[i], 2.0, fma(-v1, v1, a.v2[i])));
                    af2[i] = fma(af1[i], af1[i] / wf - g2 / g1, -(wf * t) / (g1 * g1));
                }
            }
        }

        for i in 0..n {
            work[i] = ((a.p_weights[i] * -2.0) * (a.y[i] - a.mu[i])) / (a.v0[i] * a.g1[i]);
        }
        dev_grad = vec![0.0f64; rank.max(1)];
        mgcv_mmult(
            dev_grad.as_mut_ptr(),
            a.x.as_ptr(),
            work.as_ptr(),
            true,
            false,
            ranki,
            1,
            ni,
        );
        if deriv2 {
            for v in pk.dev_hess[..rank * rank].iter_mut() {
                *v += *v;
            }
        }

        ift1(
            pk.r.as_ptr(),
            vtp,
            a.x.as_ptr(),
            a.rs.as_ptr(),
            pk.pktz.as_ptr(),
            a.sp.as_ptr(),
            a1.as_ptr(),
            a.b1.as_mut_ptr(),
            b2.as_mut_ptr(),
            eta1.as_mut_ptr(),
            eta2.as_mut_ptr(),
            ni,
            ranki,
            mi,
            &rsncol,
            deriv2,
            neg_w,
            nr as i32,
        );
        if m > 0 {
            mgcv_mmult(
                work.as_mut_ptr(),
                pk.rh.as_ptr(),
                a.b1.as_ptr(),
                false,
                false,
                ranki,
                mi,
                ranki,
            );
            get_xtx(a.dvkk.as_mut_ptr(), work.as_ptr(), ranki, mi);
        }

        let v1p = work.as_mut_ptr();
        let v2p = work.as_mut_ptr().add(n * m);
        rc_prod(a.w1.as_mut_ptr(), a1.as_ptr(), eta1.as_ptr(), mi, ni);
        if deriv2 {
            rc_prod(w2.as_mut_ptr(), a1.as_ptr(), eta2.as_ptr(), n_2d as i32, ni);
            let mut pw2 = 0usize;
            for mm in 0..m {
                for k in mm..m {
                    rc_prod(
                        v1p,
                        eta1.as_ptr().add(n * mm),
                        eta1.as_ptr().add(n * k),
                        1,
                        ni,
                    );
                    rc_prod(v2p, a2.as_ptr(), v1p, 1, ni);
                    for i in 0..n {
                        w2[pw2] += *v2p.add(i);
                        pw2 += 1;
                    }
                }
            }
        }
        let mut wi: Vec<f64> = (0..n).map(|i| 1.0 / a.w[i].abs()).collect();
        rc_prod(tk.as_mut_ptr(), wi.as_ptr(), a.w1.as_ptr(), mi, ni);
        if deriv2 {
            rc_prod(tkm.as_mut_ptr(), wi.as_ptr(), w2.as_ptr(), n_2d as i32, ni);
        }
        if reml == 0 && !fisher {
            rc_prod(a.w1.as_mut_ptr(), af1.as_ptr(), eta1.as_ptr(), mi, ni);
            tfk = vec![0.0f64; (n * m).max(1)];
            tfkm = vec![0.0f64; (n * n_2d).max(1)];
            if deriv2 {
                rc_prod(
                    w2.as_mut_ptr(),
                    af1.as_ptr(),
                    eta2.as_ptr(),
                    n_2d as i32,
                    ni,
                );
                let mut pw2 = 0usize;
                for mm in 0..m {
                    for k in mm..m {
                        rc_prod(
                            v1p,
                            eta1.as_ptr().add(n * mm),
                            eta1.as_ptr().add(n * k),
                            1,
                            ni,
                        );
                        rc_prod(v2p, af2.as_ptr(), v1p, 1, ni);
                        for i in 0..n {
                            w2[pw2] += *v2p.add(i);
                            pw2 += 1;
                        }
                    }
                }
            }
            for i in 0..n {
                wi[i] = 1.0 / a.wf[i];
            }
            rc_prod(tfk.as_mut_ptr(), wi.as_ptr(), a.w1.as_ptr(), mi, ni);
            if deriv2 {
                rc_prod(tfkm.as_mut_ptr(), wi.as_ptr(), w2.as_ptr(), n_2d as i32, ni);
            }
        }

        mgcv_mmult(
            a.d1.as_mut_ptr(),
            a.b1.as_ptr(),
            dev_grad.as_ptr(),
            true,
            false,
            mi,
            1,
            ranki,
        );
        if deriv2 {
            get_xtmx(
                a.d2.as_mut_ptr(),
                a.b1.as_ptr(),
                pk.dev_hess.as_ptr(),
                ranki,
                mi,
            );
            let mut pb2 = 0usize;
            for mm in 0..m {
                for k in mm..m {
                    let xx = dot_s8(&dev_grad[..rank], &b2[pb2..pb2 + rank]);
                    pb2 += rank;
                    a.d2[k * m + mm] += xx;
                    a.d2[mm * m + k] = a.d2[k * m + mm];
                }
            }
        }
    }

    if reml < 0 {
        // R kept as rank-by-rank for MLpenalty1
        let (mut p0, mut p1) = (0usize, 0usize);
        for j in 0..rank {
            for i in 0..=j {
                pk.r[p0] = pk.r[p1 + i];
                p0 += 1;
            }
            for _ in (j + 1)..rank {
                pk.r[p0] = 0.0;
                p0 += 1;
            }
            p1 += nr;
        }
    }

    let mut bsb = 0.0;
    if reml != 0 {
        get_bsb(
            &mut bsb,
            a.tr_a1.as_mut_ptr(),
            a.tr_a2.as_mut_ptr(),
            a.sp.as_ptr(),
            a.e.as_ptr(),
            a.rs.as_ptr(),
            &rsncol,
            enrow as i32,
            ranki,
            mi,
            0,
            pk.pktz.as_ptr(),
            a.b1.as_ptr(),
            b2.as_ptr(),
            deriv,
        );
        if deriv != 0 {
            for i in 0..m {
                a.d1[i] += a.tr_a1[i];
                if deriv2 {
                    for j in 0..m {
                        a.d2[i * m + j] += a.tr_a2[i * m + j];
                    }
                }
            }
        }
    }

    let mut p0v = 0.0;
    pearson2(
        &mut p0v,
        a.p1.as_mut_ptr(),
        a.p2.as_mut_ptr(),
        a.y.as_ptr(),
        a.mu.as_ptr(),
        a.v0.as_ptr(),
        a.v1.as_ptr(),
        a.v2.as_ptr(),
        a.g1.as_ptr(),
        a.g2.as_ptr(),
        a.p_weights.as_ptr(),
        eta1.as_ptr(),
        eta2.as_ptr(),
        n,
        m,
        deriv != 0,
        deriv2,
    );
    a.p0 = p0v;
    if reml != 0 {
        let j = (n - mp) as f64;
        a.p0 /= j;
        if deriv != 0 {
            for v in a.p1[..m].iter_mut() {
                *v /= j;
            }
        }
        if deriv > 1 {
            for v in a.p2[..m * m].iter_mut() {
                *v /= j;
            }
        }
    }

    if deriv != 0 {
        for j in (0..m).rev() {
            for i in 0..rank {
                a.beta[pk.pivot1[i] as usize] = a.b1[rank * j + i];
            }
            undrop_rows(a.beta.as_mut_ptr(), q as i32, 1, &pk.drop);
            for i in 0..q {
                a.b1[q * j + i] = a.beta[i];
            }
        }
    }
    for i in 0..rank {
        a.beta[pk.pivot1[i] as usize] = pk.pktz[i];
    }
    undrop_rows(a.beta.as_mut_ptr(), q as i32, 1, &pk.drop);

    let mut reml_penalty = 0.0;
    if reml > 0 {
        reml_penalty = pk.ldet_xwxs;
        get_ddet_xwxps(
            a.tr_a1.as_mut_ptr(),
            a.tr_a2.as_mut_ptr(),
            pk.p.as_ptr(),
            pk.k.as_ptr(),
            a.sp.as_ptr(),
            a.rs.as_ptr(),
            &rsncol,
            tk.as_ptr(),
            tkm.as_ptr(),
            ni,
            ranki,
            ranki,
            mi,
            0,
            deriv,
        );
    }
    if reml < 0 {
        reml_penalty = ml_penalty1(
            a.tr_a1.as_mut_ptr(),
            a.tr_a2.as_mut_ptr(),
            tk.as_ptr(),
            tkm.as_ptr(),
            &pk.nulli,
            pk.r.as_ptr(),
            pk.q1.as_ptr(),
            &nind,
            a.sp.as_ptr(),
            a.rs.as_mut_ptr(),
            &rsncol,
            ranki,
            ni,
            mi,
            0,
            deriv,
            a.x.as_ptr(),
            false,
        );
    }

    let mut pivot = Vec::new();
    let (wfp, tfkp, tfkmp) = if fisher {
        (a.w.as_ptr(), tk.as_ptr(), tkm.as_ptr())
    } else {
        let nr2 = n + enrow;
        let mut wx = vec![0.0f64; (nr2 * rank).max(1)];
        for i in 0..n {
            a.w[i] = a.wf[i].sqrt();
        }
        for i in 0..rank {
            for kk in 0..n {
                wx[kk + nr2 * i] = a.x[kk + n * i] * a.w[kk];
            }
            for j in 0..enrow {
                wx[n + j + nr2 * i] = a.e[j + enrow * i];
            }
        }
        pivot = vec![0i32; rank.max(1)];
        let mut tau = vec![0.0f64; rank.max(1)];
        mgcv_qr(
            wx.as_mut_ptr(),
            nr2 as i32,
            ranki,
            pivot.as_mut_ptr(),
            tau.as_mut_ptr(),
        );
        let mut r1 = vec![0.0f64; (rank * rank).max(1)];
        get_rpqr(r1.as_mut_ptr(), wx.as_ptr(), nr2 as i32, ranki, ranki);
        rinv(pk.p.as_mut_ptr(), r1.as_ptr(), ranki, ranki, ranki);
        let mut qm = vec![0.0f64; (nr2 * rank).max(1)];
        for i in 0..rank {
            qm[i * rank + i] = 1.0;
        }
        mgcv_pqrqy(
            qm.as_mut_ptr(),
            wx.as_ptr(),
            tau.as_ptr(),
            nr2 as i32,
            ranki,
            ranki,
            false,
        );
        for j in 0..rank {
            for i in 0..n {
                pk.k[i + n * j] = qm[i + nr2 * j];
            }
        }
        if deriv != 0 {
            pivoter(
                a.rs.as_mut_ptr(),
                ranki,
                scs as i32,
                pivot.as_ptr(),
                false,
                false,
            );
        }
        (a.wf.as_ptr(), tfk.as_ptr(), tfkm.as_ptr())
    };

    let tra_deriv = if reml != 0 { 0 } else { deriv };
    let mut tr_a = 0.0;
    get_tra2(
        &mut tr_a,
        a.tr_a1.as_mut_ptr(),
        a.tr_a2.as_mut_ptr(),
        pk.p.as_ptr(),
        pk.k.as_ptr(),
        a.sp.as_ptr(),
        a.rs.as_ptr(),
        &rsncol,
        tfkp,
        tfkmp,
        wfp,
        ni,
        ranki,
        ranki,
        mi,
        tra_deriv,
    );
    a.tr_a = tr_a;

    if !fisher {
        pivoter(pk.p.as_mut_ptr(), ranki, ranki, pivot.as_ptr(), false, true);
    }
    for i in 0..rank {
        for j in 0..rank {
            a.rv[pk.pivot1[j] as usize + i * rank] = pk.p[j + rank * i];
        }
    }
    undrop_rows(a.rv.as_mut_ptr(), q as i32, ranki, &pk.drop);
    for v in a.rv[q * rank..q * q].iter_mut() {
        *v = 0.0;
    }
    a.x[..rank * n].copy_from_slice(&pk.k[..rank * n]);
    for v in a.x[rank * n..q * n].iter_mut() {
        *v = 0.0;
    }
    a.rank_est = ranki;
    if reml != 0 {
        a.rank_tol = reml_penalty;
        a.conv_tol = bsb;
    }
    a.deriv = 0;
}

/// gdi.c's `pls_fit1`: the penalized least squares fit of the pseudodata
/// in `y` (`n` used; at least `max(n, q)` long) with weights `w`, some
/// possibly negative (full Newton), to the `n`-by-`q` `x` under the penalty
/// square root `e` (`re`-by-`q`; its unidentifiable columns are dropped in
/// place), the well scaled `es` deciding the rank. Leaves the linear
/// predictor in `eta`, the penalty in `penalty` and the coefficients in the
/// first `q` entries of `y`. Returns false, mgcv's `n = -1`, when
/// `X'WX + E'E` is not positive definite. With `use_wy`, or when `R'Q'Wz`
/// drifts from `X'Wz` (in `wy`), the coefficients come from `X'Wz`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn pls_fit1(
    y: &mut [f64],
    x: &[f64],
    w: &[f64],
    wy: &[f64],
    e: &mut [f64],
    es: &[f64],
    n: usize,
    q: usize,
    re: usize,
    eta: &mut [f64],
    penalty: &mut f64,
    rank_tol: f64,
    use_wy: &mut bool,
) -> bool {
    let (ni, qi, rei) = (n as i32, q as i32, re as i32);
    let rr = q.min(n);
    let nr = rr + re;
    let nz = n.max(nr);
    let (rri, nri) = (rr as i32, nr as i32);
    let mut z = vec![0.0f64; nz.max(1)];
    let mut raw = vec![0.0f64; n.max(1)];
    let mut nind: Vec<usize> = Vec::new();
    for i in 0..n {
        if w[i] < 0.0 {
            nind.push(i);
            raw[i] = (-w[i]).sqrt();
        } else {
            raw[i] = w[i].sqrt();
        }
    }
    let neg_w = nind.len();
    for i in 0..n {
        z[i] = y[i] * raw[i];
    }
    for &k in &nind {
        z[k] = -z[k];
    }
    let mut wx = vec![0.0f64; (n * q).max(1)];
    for j in 0..q {
        for i in 0..n {
            wx[i + n * j] = x[i + n * j] * raw[i];
        }
    }
    let mut tau = vec![0.0f64; q.max(1)];
    let mut pivot = vec![0i32; q.max(1)];
    mgcv_qr(
        wx.as_mut_ptr(),
        ni,
        qi,
        pivot.as_mut_ptr(),
        tau.as_mut_ptr(),
    );
    let mut r1 = vec![0.0f64; (rr * q).max(1)];
    get_rpqr(r1.as_mut_ptr(), wx.as_ptr(), ni, qi, rri);
    pivoter(r1.as_mut_ptr(), rri, qi, pivot.as_ptr(), true, true);

    let rnorm = frobenius_norm(r1.as_ptr(), rri, qi);
    let enorm = frobenius_norm(es.as_ptr(), rei, qi);
    let mut r = vec![0.0f64; (q * nr).max(1)];
    for j in 0..q {
        for i in 0..rr {
            r[i + nr * j] = r1[i + rr * j] / rnorm;
        }
        for i in 0..re {
            r[i + rr + nr * j] = es[i + re * j] / enorm;
        }
    }
    let mut tau1 = vec![0.0f64; q.max(1)];
    let mut pivot1 = vec![0i32; q.max(1)];
    mgcv_qr(
        r.as_mut_ptr(),
        nri,
        qi,
        pivot1.as_mut_ptr(),
        tau1.as_mut_ptr(),
    );
    let mut rank = q.min(nr);
    let mut rcond = r_cond(r.as_ptr(), nri, rank as i32);
    while rank_tol * rcond > 1.0 {
        rank -= 1;
        rcond = r_cond(r.as_ptr(), nri, rank as i32);
    }
    let ranki = rank as i32;
    let mut drop: Vec<i32> = pivot1[rank..q].to_vec();
    drop.sort_unstable();
    if !drop.is_empty() {
        drop_cols(r1.as_mut_ptr(), rri, qi, &drop);
        drop_cols(e.as_mut_ptr(), rei, qi, &drop);
    }
    for j in 0..rank {
        for i in 0..rr {
            r[i + nr * j] = r1[i + rr * j];
        }
        for i in 0..re {
            r[i + rr + nr * j] = e[i + re * j];
        }
    }
    mgcv_qr(
        r.as_mut_ptr(),
        nri,
        ranki,
        pivot1.as_mut_ptr(),
        tau1.as_mut_ptr(),
    );

    let mut vt = Vec::new();
    let mut d = Vec::new();
    if neg_w > 0 {
        let mut qm = vec![0.0f64; (nr * rank).max(1)];
        for i in 0..rank {
            qm[i * nr + i] = 1.0;
        }
        mgcv_qrqy_left(
            qm.as_mut_ptr(),
            r.as_ptr(),
            tau1.as_ptr(),
            nri,
            ranki,
            ranki,
            false,
        );
        let mut q1 = vec![0.0f64; (n * rank).max(1)];
        for i in 0..rr {
            for j in 0..rank {
                q1[i + rr * j] = qm[i + nr * j];
            }
        }
        mgcv_pqrqy(
            q1.as_mut_ptr(),
            wx.as_ptr(),
            tau.as_ptr(),
            ni,
            rri,
            ranki,
            false,
        );
        let k = (rank + 1).max(neg_w);
        let mut iq = vec![0.0f64; k * rank];
        for (i, &row) in nind.iter().enumerate() {
            for j in 0..rank {
                iq[i + k * j] = q1[row + n * j];
            }
        }
        vt = vec![0.0f64; (rank * rank).max(1)];
        d = vec![0.0f64; rank];
        mgcv_svd_full(
            iq.as_mut_ptr(),
            vt.as_mut_ptr(),
            d.as_mut_ptr(),
            k as i32,
            ranki,
        );
        for di in d.iter_mut() {
            let v = fma(-2.0 * *di, *di, 1.0);
            if v < -rank_tol {
                return false;
            }
            *di = if v <= 0.0 { 0.0 } else { 1.0 / v };
        }
    }

    let mut work = vec![0.0f64; (4 * q).max(1)];
    if !*use_wy {
        mgcv_pqrqy(z.as_mut_ptr(), wx.as_ptr(), tau.as_ptr(), ni, rri, 1, true);
        for v in z[rank..nz].iter_mut() {
            *v = 0.0;
        }
        mgcv_qrqy_left(
            z.as_mut_ptr(),
            r.as_ptr(),
            tau1.as_ptr(),
            nri,
            1,
            ranki,
            true,
        );
        for v in z[rank..nz].iter_mut() {
            *v = 0.0;
        }
        for i in 0..rank {
            y[i] = z[i];
            work[i + q] = z[i];
        }
        if neg_w > 0 {
            mgcv_mmult(
                work.as_mut_ptr(),
                vt.as_ptr(),
                z.as_ptr(),
                false,
                false,
                ranki,
                1,
                ranki,
            );
            for i in 0..rank {
                work[i] *= d[i];
            }
            mgcv_mmult(
                z.as_mut_ptr(),
                vt.as_ptr(),
                work.as_ptr(),
                true,
                false,
                ranki,
                1,
                ranki,
            );
            y[..rank].copy_from_slice(&z[..rank]);
        }
        mgcv_qrqy_left(
            z.as_mut_ptr(),
            r.as_ptr(),
            tau1.as_ptr(),
            nri,
            1,
            ranki,
            false,
        );
        *penalty = 0.0;
        for i in rr..nr {
            *penalty = fma(z[i], z[i], *penalty);
        }
        for v in z[rank..n].iter_mut() {
            *v = 0.0;
        }
        mgcv_pqrqy(z.as_mut_ptr(), wx.as_ptr(), tau.as_ptr(), ni, rri, 1, false);
        for i in 0..n {
            eta[i] = z[i] / raw[i];
        }
    }

    mgcv_mmult(
        work.as_mut_ptr(),
        x.as_ptr(),
        wy.as_ptr(),
        true,
        false,
        qi,
        1,
        ni,
    );
    drop_rows(work.as_mut_ptr(), qi, 1, &drop);
    pivoter(work.as_mut_ptr(), ranki, 1, pivot1.as_ptr(), false, false);
    if !*use_wy {
        let (mut zz, mut zz1) = (0.0f64, 0.0f64);
        for i in 0..rank {
            let mut xx = dot_s8(&r[nr * i..nr * i + i + 1], &work[q..q + i + 1]);
            xx -= work[i];
            zz1 = fma(xx, xx, zz1);
            zz = fma(work[i], work[i], zz);
        }
        if zz1 > rank_tol * zz {
            *use_wy = true;
        }
    }
    if *use_wy {
        for k in 0..rank {
            let xx = dot_s8(&r[nr * k..nr * k + k], &z[..k]);
            z[k] = (work[k] - xx) / r[k + nr * k];
        }
        if neg_w > 0 {
            mgcv_mmult(
                work.as_mut_ptr(),
                vt.as_ptr(),
                z.as_ptr(),
                false,
                false,
                ranki,
                1,
                ranki,
            );
            for i in 0..rank {
                work[i] *= d[i];
            }
            mgcv_mmult(
                z.as_mut_ptr(),
                vt.as_ptr(),
                work.as_ptr(),
                true,
                false,
                ranki,
                1,
                ranki,
            );
        }
        y[..rank].copy_from_slice(&z[..rank]);
    }
    for k in (0..rank).rev() {
        let mut xx = 0.0;
        for j in (k + 1)..rank {
            xx = fma(r[k + nr * j], z[j], xx);
        }
        z[k] = (y[k] - xx) / r[k + nr * k];
    }
    for i in 0..rank {
        y[pivot1[i] as usize] = z[i];
    }
    undrop_rows(y.as_mut_ptr(), qi, 1, &drop);
    if *use_wy {
        mgcv_mmult(
            eta.as_mut_ptr(),
            x.as_ptr(),
            y.as_ptr(),
            false,
            false,
            ni,
            1,
            qi,
        );
        mgcv_mmult(
            work.as_mut_ptr(),
            e.as_ptr(),
            y.as_ptr(),
            false,
            false,
            rei,
            1,
            qi,
        );
        *penalty = 0.0;
        for v in &work[..re] {
            *penalty = fma(*v, *v, *penalty);
        }
    }
    true
}

/// `.C(C_pls_fit1, ...)` as gam.fit3 calls it: `args` maps each argument
/// name (`y`, `X`, `w`, `wy`, `E`, `Es`, `n`, `q`, `rE`, `eta`, `penalty`,
/// `rank.tol`, `nt`, `use.wy`) to its value, matrices column-major and
/// flattened; returns the same names with the values the call leaves.
#[pyfunction]
fn pls_fit1_c<'py>(py: Python<'py>, args: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let n = scalar_of(args, "n")? as usize;
    let q = scalar_of(args, "q")? as usize;
    let re = scalar_of(args, "rE")? as usize;
    let mut y = vec_of(args, "y")?;
    // X is only read: borrow it where it starts on a 16-byte boundary, as
    // R's `.C` copy does (Accelerate's ddot rounds by its operands'
    // alignment), else copy it into a fresh allocation that does
    let x_arr = args
        .get_item("X")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("X"))?;
    let x_ro = x_arr.extract::<PyReadonlyArray1<'_, f64>>().ok();
    let x_borrowed: Option<&[f64]> = x_ro
        .as_ref()
        .and_then(|a| a.as_slice().ok())
        .filter(|sl| (sl.as_ptr() as usize).is_multiple_of(16));
    let x_owned: Vec<f64> = if x_borrowed.is_none() {
        vec_of(args, "X")?
    } else {
        Vec::new()
    };
    let x: &[f64] = x_borrowed.unwrap_or(&x_owned);
    let w = vec_of(args, "w")?;
    let wy = vec_of(args, "wy")?;
    let mut e = vec_of(args, "E")?;
    let es = vec_of(args, "Es")?;
    let mut eta = vec_of(args, "eta")?;
    let mut penalty = scalar_of(args, "penalty")?;
    let rank_tol = scalar_of(args, "rank.tol")?;
    let mut use_wy = scalar_of(args, "use.wy")? != 0.0;
    let checks = [
        ("y", y.len() >= n.max(q)),
        ("X", x.len() >= n * q),
        ("w", w.len() >= n),
        ("wy", wy.len() >= n),
        ("E", e.len() >= re * q),
        ("Es", es.len() >= re * q),
        ("eta", eta.len() >= n),
    ];
    for (name, ok) in checks {
        if !ok {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} is too short"
            )));
        }
    }
    let ok = py.allow_threads(|| unsafe {
        pls_fit1(
            &mut y,
            x,
            &w,
            &wy,
            &mut e,
            &es,
            n,
            q,
            re,
            &mut eta,
            &mut penalty,
            rank_tol,
            &mut use_wy,
        )
    });
    let out = PyDict::new(py);
    out.set_item("y", y.into_pyarray(py))?;
    out.set_item("E", e.into_pyarray(py))?;
    out.set_item("eta", eta.into_pyarray(py))?;
    out.set_item("penalty", vec![penalty].into_pyarray(py))?;
    out.set_item("n", vec![if ok { n as f64 } else { -1.0 }].into_pyarray(py))?;
    out.set_item(
        "use.wy",
        vec![if use_wy { 1.0 } else { 0.0 }].into_pyarray(py),
    )?;
    Ok(out)
}

/// The values under `key`: a contiguous float64 array copied in one go, or
/// any sequence of numbers.
fn vec_of(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let v = d
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?;
    if let Ok(a) = v.extract::<PyReadonlyArray1<'_, f64>>() {
        if let Ok(sl) = a.as_slice() {
            return Ok(sl.to_vec());
        }
    }
    v.extract::<Vec<f64>>()
}

fn scalar_of(d: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    let v = vec_of(d, key)?;
    v.first()
        .copied()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("{key} is empty")))
}

/// `.C(C_gdi1, ...)` as gam.fit3 calls it: `args` maps each argument name
/// gam.fit3 uses (`X`, `E`, `Eb`, `rS`, ..., `dVkk`) to its value, matrices
/// column-major and flattened; returns the same names with the values the
/// call leaves.
#[pyfunction]
fn gdi1_c<'py>(py: Python<'py>, args: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let n = scalar_of(args, "n")? as usize;
    let q = scalar_of(args, "p")? as usize;
    let m = scalar_of(args, "M")? as usize;
    let mut a = Gdi1Args {
        x: vec_of(args, "X")?,
        e: vec_of(args, "E")?,
        es: vec_of(args, "Eb")?,
        rs: vec_of(args, "rS")?,
        sp: vec_of(args, "sp")?,
        z: vec_of(args, "z")?,
        w: vec_of(args, "w")?,
        wf: vec_of(args, "wf")?,
        alpha: vec_of(args, "alpha")?,
        mu: vec_of(args, "mu")?,
        y: vec_of(args, "y")?,
        p_weights: vec_of(args, "p.weights")?,
        g1: vec_of(args, "g1")?,
        g2: vec_of(args, "g2")?,
        g3: vec_of(args, "g3")?,
        g4: vec_of(args, "g4")?,
        v0: vec_of(args, "V0")?,
        v1: vec_of(args, "V1")?,
        v2: vec_of(args, "V2")?,
        v3: vec_of(args, "V3")?,
        beta: vec_of(args, "beta")?,
        b1: vec_of(args, "b1")?,
        w1: vec_of(args, "w1")?,
        d1: vec_of(args, "D1")?,
        d2: vec_of(args, "D2")?,
        p0: scalar_of(args, "P")?,
        p1: vec_of(args, "P1")?,
        p2: vec_of(args, "P2")?,
        tr_a: scalar_of(args, "trA")?,
        tr_a1: vec_of(args, "trA1")?,
        tr_a2: vec_of(args, "trA2")?,
        rv: vec_of(args, "rV")?,
        rank_tol: scalar_of(args, "rank.tol")?,
        conv_tol: scalar_of(args, "conv.tol")?,
        rank_est: scalar_of(args, "rank.est")? as i32,
        n,
        q,
        m,
        mp: scalar_of(args, "Mp")? as usize,
        enrow: scalar_of(args, "Enrow")? as usize,
        rsncol: vec_of(args, "rSncol")?.iter().map(|&v| v as i32).collect(),
        deriv: scalar_of(args, "deriv")? as i32,
        reml: scalar_of(args, "REML")? as i32,
        fisher: scalar_of(args, "fisher")? != 0.0,
        dvkk: vec_of(args, "dVkk")?,
    };
    let checks = [
        ("X", a.x.len() >= n * q),
        ("rV", a.rv.len() >= q * q),
        ("b1", a.b1.len() >= q * m),
        ("w1", a.w1.len() >= n * m),
        ("D2", a.d2.len() >= m * m),
        ("P2", a.p2.len() >= m * m),
        ("trA2", a.tr_a2.len() >= m * m),
        ("dVkk", a.dvkk.len() >= m * m),
        ("rSncol", a.rsncol.len() >= m),
    ];
    for (name, ok) in checks {
        if !ok {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} is too short"
            )));
        }
    }
    py.allow_threads(|| unsafe { gdi1(&mut a) });
    let out = PyDict::new(py);
    out.set_item("X", a.x.into_pyarray(py))?;
    out.set_item("E", a.e.into_pyarray(py))?;
    out.set_item("rS", a.rs.into_pyarray(py))?;
    out.set_item("w", a.w.into_pyarray(py))?;
    out.set_item("beta", a.beta.into_pyarray(py))?;
    out.set_item("b1", a.b1.into_pyarray(py))?;
    out.set_item("w1", a.w1.into_pyarray(py))?;
    out.set_item("D1", a.d1.into_pyarray(py))?;
    out.set_item("D2", a.d2.into_pyarray(py))?;
    out.set_item("P", vec![a.p0].into_pyarray(py))?;
    out.set_item("P1", a.p1.into_pyarray(py))?;
    out.set_item("P2", a.p2.into_pyarray(py))?;
    out.set_item("trA", vec![a.tr_a].into_pyarray(py))?;
    out.set_item("trA1", a.tr_a1.into_pyarray(py))?;
    out.set_item("trA2", a.tr_a2.into_pyarray(py))?;
    out.set_item("rV", a.rv.into_pyarray(py))?;
    out.set_item("rank.tol", vec![a.rank_tol].into_pyarray(py))?;
    out.set_item("conv.tol", vec![a.conv_tol].into_pyarray(py))?;
    out.set_item("rank.est", vec![a.rank_est as f64].into_pyarray(py))?;
    out.set_item("deriv", vec![a.deriv as f64].into_pyarray(py))?;
    out.set_item("dVkk", a.dvkk.into_pyarray(py))?;
    Ok(out)
}

/// mgcv's `trBtAB` as compiled inline in `get_stableS` (gdi.c): for each
/// column `j` of `b` (`n`-by-`m`, column-major) and `k` of `a` (`n`-by-`n`),
/// `tr += a[i,k] b[i,j] b[k,j]` over `i`, the first `8 floor(n/8)` terms
/// (both products rounded) added in order, the rest fused.
fn tr_bt_ab(a: &[f64], b: &[f64], n: usize, m: usize) -> f64 {
    let nv = if n >= 8 { n / 8 * 8 } else { 0 };
    let mut tr = 0.0;
    for j in 0..m {
        let bj = &b[j * n..(j + 1) * n];
        for k in 0..n {
            let x = bj[k];
            let ak = &a[k * n..(k + 1) * n];
            for i in 0..nv {
                tr += (ak[i] * bj[i]) * x;
            }
            for i in nv..n {
                tr = fma(ak[i] * bj[i], x, tr);
            }
        }
    }
    tr
}

/// `trBtAB(a, b)` for numpy matrices of any layout.
#[pyfunction]
#[pyo3(name = "tr_bt_ab")]
fn tr_bt_ab_py(a: PyReadonlyArray2<'_, f64>, b: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    let (a, b) = (a.as_array(), b.as_array());
    let (n, m) = b.dim();
    if a.dim() != (n, n) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a must be square with as many rows as b",
        ));
    }
    let ac: Vec<f64> = (0..n * n).map(|t| a[[t % n, t / n]]).collect();
    let bc: Vec<f64> = (0..n * m).map(|t| b[[t % n, t / n]]).collect();
    Ok(tr_bt_ab(&ac, &bc, n, m))
}

/// `s + sum a[i] b[i]` accumulated in a local as clang vectorizes it: for
/// `n >= 8` the first `8 floor(n/8)` products rounded and added in order,
/// the rest fused.
fn dot_s8_acc(mut s: f64, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let nv = if n >= 8 { n / 8 * 8 } else { 0 };
    for i in 0..nv {
        s += a[i] * b[i];
    }
    for i in nv..n {
        s = fma(a[i], b[i], s);
    }
    s
}

/// gdi.c's `ift2`: by the implicit function theorem the first (`b1`, `r`
/// by `n_theta + m`, the likelihood parameters first) and second (`b2`, one
/// column per `k >= i` pair) derivatives of the coefficients with respect
/// to the extra likelihood parameters and the log smoothing parameters, and
/// those of the linear predictor (`eta1`, `eta2`).
#[allow(clippy::too_many_arguments)]
unsafe fn ift2(
    rm: *const f64,
    vt: *const f64,
    x: *const f64,
    rs: *const f64,
    beta: *const f64,
    sp: *const f64,
    det_th: *const f64,
    det2_th: *const f64,
    det3: *const f64,
    det_th2: *const f64,
    b1: *mut f64,
    b2: *mut f64,
    eta1: *mut f64,
    eta2: *mut f64,
    n: i32,
    r: i32,
    m: i32,
    n_theta: i32,
    rsncol: &[i32],
    deriv2: bool,
    neg_w: bool,
    nr: i32,
) {
    let (nu, ru, mu, nth) = (n as usize, r as usize, m as usize, n_theta as usize);
    let nw = nu.max(ru).max(1);
    let mut work = vec![0.0f64; nw];
    let mut work1 = vec![0.0f64; nw];
    let mut db_th = vec![0.0f64; ru.max(1)];
    let ntot = mu + nth;
    let n_2d = (ntot * (1 + ntot)) / 2;
    for i in 0..ntot {
        if i < nth {
            mgcv_mmult(
                db_th.as_mut_ptr(),
                x,
                det_th.add(i * nu),
                true,
                false,
                r,
                1,
                n,
            );
            for v in db_th[..ru].iter_mut() {
                *v *= -0.5;
            }
        } else {
            mult_sk(
                db_th.as_mut_ptr(),
                beta,
                1,
                i - nth,
                rs,
                rsncol,
                r,
                work.as_mut_ptr(),
            );
            let s = -*sp.add(i - nth);
            for v in db_th[..ru].iter_mut() {
                *v *= s;
            }
        }
        apply_pt(
            work.as_mut_ptr(),
            db_th.as_ptr(),
            rm,
            vt,
            neg_w,
            nr,
            r,
            1,
            false,
        );
        apply_p(
            b1.add(i * ru),
            work.as_ptr(),
            rm,
            vt,
            neg_w,
            nr,
            r,
            1,
            false,
        );
    }
    mgcv_mmult(eta1, x, b1, false, false, n, ntot as i32, r);

    if deriv2 {
        let mut pp = b2;
        let mut kk = 0usize;
        for i in 0..ntot {
            for k in i..ntot {
                let p0 = eta1.add(nu * i);
                let p1 = eta1.add(nu * k);
                for j in 0..nu {
                    work[j] = (-*p0.add(j) * *p1.add(j)) * *det3.add(j);
                }
                mgcv_mmult(db_th.as_mut_ptr(), x, work.as_ptr(), true, false, r, 1, n);
                if k < nth {
                    let q0 = det2_th.add(nu * k);
                    for j in 0..nu {
                        work[j] = *q0.add(j) * *p0.add(j);
                    }
                    mgcv_mmult(work1.as_mut_ptr(), x, work.as_ptr(), true, false, r, 1, n);
                } else {
                    mult_sk(
                        work1.as_mut_ptr(),
                        b1.add(i * ru),
                        1,
                        k - nth,
                        rs,
                        rsncol,
                        r,
                        work.as_mut_ptr(),
                    );
                    let s2 = *sp.add(k - nth) * 2.0;
                    for v in work1[..ru].iter_mut() {
                        *v *= s2;
                    }
                }
                for j in 0..ru {
                    db_th[j] -= work1[j];
                }
                if i < nth {
                    let q0 = det2_th.add(nu * i);
                    for j in 0..nu {
                        work[j] = *q0.add(j) * *p1.add(j);
                    }
                    mgcv_mmult(work1.as_mut_ptr(), x, work.as_ptr(), true, false, r, 1, n);
                } else {
                    mult_sk(
                        work1.as_mut_ptr(),
                        b1.add(k * ru),
                        1,
                        i - nth,
                        rs,
                        rsncol,
                        r,
                        work.as_mut_ptr(),
                    );
                    let s2 = *sp.add(i - nth) * 2.0;
                    for v in work1[..ru].iter_mut() {
                        *v *= s2;
                    }
                }
                for j in 0..ru {
                    db_th[j] -= work1[j];
                }
                if i < nth && k < nth {
                    mgcv_mmult(
                        work.as_mut_ptr(),
                        x,
                        det_th2.add(nu * kk),
                        true,
                        false,
                        r,
                        1,
                        n,
                    );
                    for j in 0..ru {
                        db_th[j] -= work[j];
                    }
                    kk += 1;
                } else if i == k {
                    mult_sk(
                        work1.as_mut_ptr(),
                        beta,
                        1,
                        i - nth,
                        rs,
                        rsncol,
                        r,
                        work.as_mut_ptr(),
                    );
                    let si = *sp.add(i - nth);
                    for j in 0..ru {
                        db_th[j] -= (work1[j] * si) * 2.0;
                    }
                }
                for v in db_th[..ru].iter_mut() {
                    *v *= 0.5;
                }
                apply_pt(
                    work.as_mut_ptr(),
                    db_th.as_ptr(),
                    rm,
                    vt,
                    neg_w,
                    nr,
                    r,
                    1,
                    false,
                );
                apply_p(pp, work.as_ptr(), rm, vt, neg_w, nr, r, 1, false);
                pp = pp.add(ru);
            }
        }
        mgcv_mmult(eta2, x, b2, false, false, n, n_2d as i32, r);
    }
}

/// gdi2's arguments as gam.fit4's `.C(C_gdi2, ...)` passes them, the
/// arrays it writes included.
pub(crate) struct Gdi2Args {
    pub x: Vec<f64>,
    pub e: Vec<f64>,
    pub es: Vec<f64>,
    pub rs: Vec<f64>,
    pub sp: Vec<f64>,
    pub z: Vec<f64>,
    pub w: Vec<f64>,
    pub wz: Vec<f64>,
    pub wf: Vec<f64>,
    pub dth: Vec<f64>,
    pub det: Vec<f64>,
    pub det2: Vec<f64>,
    pub dth2: Vec<f64>,
    pub det_th: Vec<f64>,
    pub det2_th: Vec<f64>,
    pub det3: Vec<f64>,
    pub det_th2: Vec<f64>,
    pub det4: Vec<f64>,
    pub det3_th: Vec<f64>,
    pub det2_th2: Vec<f64>,
    pub beta: Vec<f64>,
    pub b1: Vec<f64>,
    pub w1: Vec<f64>,
    pub d1: Vec<f64>,
    pub d2: Vec<f64>,
    pub p0: f64,
    pub p1: Vec<f64>,
    pub p2: Vec<f64>,
    pub ldet: f64,
    pub ldet1: Vec<f64>,
    pub ldet2: Vec<f64>,
    pub rv: Vec<f64>,
    pub rank_tol: f64,
    pub rank_est: i32,
    pub n: usize,
    pub q: usize,
    pub m: usize,
    pub n_theta: usize,
    pub mp: usize,
    pub enrow: usize,
    pub rsncol: Vec<i32>,
    pub deriv: i32,
    pub type_: i32,
    pub dvkk: Vec<f64>,
}

/// gdi.c's `gdi2`, one thread: at the converged coefficients of an
/// extended family's penalized IRLS, the coefficients, their derivatives
/// and the deviance's and penalty's with respect to the likelihood
/// parameters and the log smoothing parameters, log|X'WX + S| (REML) or
/// its ML version (`ldet < 0` on entry) and their derivatives, `rV` and `K`
/// (in `x`) from the Fisher weights.
pub(crate) unsafe fn gdi2(a: &mut Gdi2Args) {
    let (n, q, m, nth, mp, enrow) = (a.n, a.q, a.m, a.n_theta, a.mp, a.enrow);
    let (ni, mi, nthi) = (n as i32, m as i32, nth as i32);
    let ml = a.ldet < 0.0;
    let deriv = a.deriv;
    let deriv2 = deriv == 2;
    let scs: usize = a.rsncol[..m].iter().map(|&c| c as usize).sum();
    let rsncol = a.rsncol.clone();

    let mut raw = vec![0.0f64; n.max(1)];
    let mut nind = Vec::new();
    for i in 0..n {
        if a.w[i] < 0.0 {
            nind.push(i as i32);
            raw[i] = (-a.w[i]).sqrt();
        } else if a.w[i] > 0.0 {
            raw[i] = a.w[i].sqrt();
        } else {
            a.type_ = 1;
        }
    }
    let neg_w = !nind.is_empty();
    let type1 = a.type_ == 1;
    let zp = if type1 { a.wz.as_ptr() } else { a.z.as_ptr() };
    let mut pk = gdi_pk(
        a.x.as_mut_ptr(),
        a.e.as_mut_ptr(),
        a.es.as_ptr(),
        a.rs.as_mut_ptr(),
        zp,
        &raw,
        &nind,
        n,
        q,
        mp,
        enrow,
        deriv2,
        scs,
        1,
        a.rank_tol,
        type1,
    );
    a.ldet = pk.ldet_xwxs;
    let rank = pk.rank;
    let ranki = rank as i32;
    let nr = pk.nr;
    let vtp = if neg_w {
        pk.vt.as_ptr()
    } else {
        std::ptr::null()
    };

    let mut n_work = (4 * n + 2 * q) * m + 2 * n;
    n_work = n_work.max(5 * q).max((m * (1 + m)) / 2 * n);
    let mut work = vec![0.0f64; n_work.max(1)];
    let ntot = m + nth;
    let n_2d = (ntot * (1 + ntot)) / 2;
    let mut eta1 = Vec::new();
    let mut b2 = Vec::new();
    let mut eta2 = Vec::new();
    let mut w2 = Vec::new();
    let mut tk = Vec::new();
    let mut tkm = Vec::new();
    if deriv != 0 {
        eta1 = vec![0.0f64; (n * ntot).max(1)];
        if deriv2 {
            b2 = vec![0.0f64; (rank * n_2d).max(1)];
            eta2 = vec![0.0f64; (n * n_2d).max(1)];
        }
        ift2(
            pk.r.as_ptr(),
            vtp,
            a.x.as_ptr(),
            a.rs.as_ptr(),
            pk.pktz.as_ptr(),
            a.sp.as_ptr(),
            a.det_th.as_ptr(),
            a.det2_th.as_ptr(),
            a.det3.as_ptr(),
            a.det_th2.as_ptr(),
            a.b1.as_mut_ptr(),
            b2.as_mut_ptr(),
            eta1.as_mut_ptr(),
            eta2.as_mut_ptr(),
            ni,
            ranki,
            mi,
            nthi,
            &rsncol,
            deriv2,
            neg_w,
            nr as i32,
        );
        if m > 0 {
            mgcv_mmult(
                work.as_mut_ptr(),
                pk.rh.as_ptr(),
                a.b1.as_ptr().add(nth * rank),
                false,
                false,
                ranki,
                mi,
                ranki,
            );
            get_xtx(a.dvkk.as_mut_ptr(), work.as_ptr(), ranki, mi);
        }
        // the deviance's gradient: the likelihood parameters' entries carry
        // its direct dependence Dth
        for i in 0..nth {
            let mut acc = 0.0;
            for j in 0..n {
                acc += fma(eta1[n * i + j], a.det[j], a.dth[n * i + j]);
            }
            a.d1[i] = acc;
        }
        for i in nth..ntot {
            a.d1[i] = dot_s8_from(&eta1[n * i..n * i + n], &a.det[..n], 12);
        }
        if deriv2 {
            let mut p0 = 0usize;
            let mut p6 = 0usize;
            for i in 0..ntot {
                for k in i..ntot {
                    let mut xx = 0.0;
                    for j in 0..n {
                        xx += fma(
                            a.det2[j] * eta1[n * i + j],
                            eta1[n * k + j],
                            a.det[j] * eta2[p0],
                        );
                        p0 += 1;
                    }
                    if k < nth {
                        for _ in 0..n {
                            xx += a.dth2[p6];
                            p6 += 1;
                        }
                    }
                    if i < nth {
                        xx = dot_s8_acc(xx, &a.det_th[n * i..n * i + n], &eta1[n * k..n * k + n]);
                    }
                    if k < nth {
                        xx = dot_s8_acc(xx, &a.det_th[n * k..n * k + n], &eta1[n * i..n * i + n]);
                    }
                    a.d2[i + ntot * k] = xx;
                    a.d2[k + ntot * i] = xx;
                }
            }
        }

        // the weights' derivatives
        let mut p0 = 0usize;
        let mut p3 = 0usize;
        for i in 0..ntot {
            if i < nth {
                for j in 0..n {
                    a.w1[p0] = fma(a.det3[j], eta1[p0], a.det2_th[p3]) * 0.5;
                    p0 += 1;
                    p3 += 1;
                }
            } else {
                for j in 0..n {
                    a.w1[p0] = (a.det3[j] * eta1[p0]) * 0.5;
                    p0 += 1;
                }
            }
        }
        if deriv2 {
            w2 = vec![0.0f64; (n * n_2d).max(1)];
            let mut p0 = 0usize;
            let mut p1 = 0usize;
            for i in 0..ntot {
                for k in i..ntot {
                    let st = p0;
                    for j in 0..n {
                        w2[p0] = fma(
                            a.det4[j] * eta1[n * i + j],
                            eta1[n * k + j],
                            a.det3[j] * eta2[p0],
                        );
                        p0 += 1;
                    }
                    if i < nth {
                        for j in 0..n {
                            w2[st + j] = fma(a.det3_th[n * i + j], eta1[n * k + j], w2[st + j]);
                        }
                    }
                    if k < nth {
                        for j in 0..n {
                            w2[st + j] = fma(a.det3_th[n * k + j], eta1[n * i + j], w2[st + j]);
                        }
                        if i < nth {
                            for j in 0..n {
                                w2[st + j] += a.det2_th2[p1];
                                p1 += 1;
                            }
                        }
                    }
                    for v in w2[st..st + n].iter_mut() {
                        *v *= 0.5;
                    }
                }
            }
        }
        if !type1 {
            let wi: Vec<f64> = (0..n).map(|i| 1.0 / a.w[i].abs()).collect();
            tk = vec![0.0f64; (n * ntot).max(1)];
            rc_prod(tk.as_mut_ptr(), wi.as_ptr(), a.w1.as_ptr(), ntot as i32, ni);
            if deriv2 {
                tkm = vec![0.0f64; (n * n_2d).max(1)];
                rc_prod(tkm.as_mut_ptr(), wi.as_ptr(), w2.as_ptr(), n_2d as i32, ni);
            }
        } else {
            tk = a.w1.clone();
            tkm = w2.clone();
        }
    }

    if ml {
        // R kept rank by rank
        let mut p0 = 0usize;
        let mut p1 = 0usize;
        for j in 0..rank {
            for i in 0..=j {
                pk.r[p0] = pk.r[p1 + i];
                p0 += 1;
            }
            for _ in j + 1..rank {
                pk.r[p0] = 0.0;
                p0 += 1;
            }
            p1 += nr;
        }
    }
    let mut bsb = 0.0;
    get_bsb(
        &mut bsb,
        a.p1.as_mut_ptr(),
        a.p2.as_mut_ptr(),
        a.sp.as_ptr(),
        a.e.as_ptr(),
        a.rs.as_ptr(),
        &rsncol,
        enrow as i32,
        ranki,
        mi,
        nthi,
        pk.pktz.as_ptr(),
        a.b1.as_ptr(),
        b2.as_ptr(),
        deriv,
    );
    a.p0 = bsb;
    let tkp = if tk.is_empty() {
        std::ptr::null()
    } else {
        tk.as_ptr()
    };
    let tkmp = if tkm.is_empty() {
        std::ptr::null()
    } else {
        tkm.as_ptr()
    };
    if ml {
        a.ldet = ml_penalty1(
            a.ldet1.as_mut_ptr(),
            a.ldet2.as_mut_ptr(),
            tkp,
            tkmp,
            &pk.nulli,
            pk.r.as_ptr(),
            pk.q1.as_ptr(),
            &nind,
            a.sp.as_ptr(),
            a.rs.as_mut_ptr(),
            &rsncol,
            ranki,
            ni,
            mi,
            nthi,
            deriv,
            a.x.as_ptr(),
            type1,
        );
    } else {
        get_ddet_xwxps(
            a.ldet1.as_mut_ptr(),
            a.ldet2.as_mut_ptr(),
            pk.p.as_ptr(),
            pk.k.as_ptr(),
            a.sp.as_ptr(),
            a.rs.as_ptr(),
            &rsncol,
            tkp,
            tkmp,
            ni,
            ranki,
            ranki,
            mi,
            nthi,
            deriv,
        );
    }

    if deriv != 0 {
        for j in (0..ntot).rev() {
            for i in 0..rank {
                a.beta[pk.pivot1[i] as usize] = a.b1[rank * j + i];
            }
            undrop_rows(a.beta.as_mut_ptr(), q as i32, 1, &pk.drop);
            for i in 0..q {
                a.b1[q * j + i] = a.beta[i];
            }
        }
    }
    for i in 0..rank {
        a.beta[pk.pivot1[i] as usize] = pk.pktz[i];
    }
    undrop_rows(a.beta.as_mut_ptr(), q as i32, 1, &pk.drop);

    // rV and K from sqrt(wf) X augmented with E
    let nr2 = n + enrow;
    let mut wx = vec![0.0f64; (nr2 * rank).max(1)];
    for i in 0..n {
        a.w[i] = a.wf[i].sqrt();
    }
    for i in 0..rank {
        for kk in 0..n {
            wx[kk + nr2 * i] = a.x[kk + n * i] * a.w[kk];
        }
        for j in 0..enrow {
            wx[n + j + nr2 * i] = a.e[j + enrow * i];
        }
    }
    let mut pivot = vec![0i32; rank.max(1)];
    let mut tau = vec![0.0f64; rank.max(1)];
    mgcv_qr(
        wx.as_mut_ptr(),
        nr2 as i32,
        ranki,
        pivot.as_mut_ptr(),
        tau.as_mut_ptr(),
    );
    let mut r1 = vec![0.0f64; (rank * rank).max(1)];
    get_rpqr(r1.as_mut_ptr(), wx.as_ptr(), nr2 as i32, ranki, ranki);
    rinv(pk.p.as_mut_ptr(), r1.as_ptr(), ranki, ranki, ranki);
    let mut qm = vec![0.0f64; (nr2 * rank).max(1)];
    for i in 0..rank {
        qm[i * rank + i] = 1.0;
    }
    mgcv_pqrqy(
        qm.as_mut_ptr(),
        wx.as_ptr(),
        tau.as_ptr(),
        nr2 as i32,
        ranki,
        ranki,
        false,
    );
    for j in 0..rank {
        for i in 0..n {
            pk.k[i + n * j] = qm[i + nr2 * j];
        }
    }
    pivoter(pk.p.as_mut_ptr(), ranki, ranki, pivot.as_ptr(), false, true);
    for i in 0..rank {
        for j in 0..rank {
            a.rv[pk.pivot1[j] as usize + i * rank] = pk.p[j + rank * i];
        }
    }
    undrop_rows(a.rv.as_mut_ptr(), q as i32, ranki, &pk.drop);
    for v in a.rv[q * rank..q * q].iter_mut() {
        *v = 0.0;
    }
    a.x[..rank * n].copy_from_slice(&pk.k[..rank * n]);
    for v in a.x[rank * n..q * n].iter_mut() {
        *v = 0.0;
    }
    a.rank_est = ranki;
}

/// `.C(C_gdi2, ...)` as gam.fit4 calls it: `args` maps each argument name
/// (`X`, `E`, `Es`, `rS`, ..., `dVkk`) to its value, matrices column-major
/// and flattened; returns the same names with the values the call leaves.
#[pyfunction]
fn gdi2_c<'py>(py: Python<'py>, args: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let n = scalar_of(args, "n")? as usize;
    let q = scalar_of(args, "q")? as usize;
    let m = scalar_of(args, "M")? as usize;
    let n_theta = scalar_of(args, "n.theta")? as usize;
    let mut a = Gdi2Args {
        x: vec_of(args, "X")?,
        e: vec_of(args, "E")?,
        es: vec_of(args, "Es")?,
        rs: vec_of(args, "rS")?,
        sp: vec_of(args, "sp")?,
        z: vec_of(args, "z")?,
        w: vec_of(args, "w")?,
        wz: vec_of(args, "wz")?,
        wf: vec_of(args, "wf")?,
        dth: vec_of(args, "Dth")?,
        det: vec_of(args, "Det")?,
        det2: vec_of(args, "Det2")?,
        dth2: vec_of(args, "Dth2")?,
        det_th: vec_of(args, "Det.th")?,
        det2_th: vec_of(args, "Det2.th")?,
        det3: vec_of(args, "Det3")?,
        det_th2: vec_of(args, "Det.th2")?,
        det4: vec_of(args, "Det4")?,
        det3_th: vec_of(args, "Det3.th")?,
        det2_th2: vec_of(args, "Deta2.th2")?,
        beta: vec_of(args, "beta")?,
        b1: vec_of(args, "b1")?,
        w1: vec_of(args, "w1")?,
        d1: vec_of(args, "D1")?,
        d2: vec_of(args, "D2")?,
        p0: scalar_of(args, "P")?,
        p1: vec_of(args, "P1")?,
        p2: vec_of(args, "P2")?,
        ldet: scalar_of(args, "ldet")?,
        ldet1: vec_of(args, "ldet1")?,
        ldet2: vec_of(args, "ldet2")?,
        rv: vec_of(args, "rV")?,
        rank_tol: scalar_of(args, "rank.tol")?,
        rank_est: scalar_of(args, "rank.est")? as i32,
        n,
        q,
        m,
        n_theta,
        mp: scalar_of(args, "Mp")? as usize,
        enrow: scalar_of(args, "Enrow")? as usize,
        rsncol: vec_of(args, "rSncol")?.iter().map(|&v| v as i32).collect(),
        deriv: scalar_of(args, "deriv")? as i32,
        type_: scalar_of(args, "type")? as i32,
        dvkk: vec_of(args, "dVkk")?,
    };
    let ntot = m + n_theta;
    let checks = [
        ("X", a.x.len() >= n * q),
        ("rV", a.rv.len() >= q * q),
        ("b1", a.b1.len() >= q * ntot),
        ("w1", a.w1.len() >= n * ntot),
        ("D2", a.d2.len() >= ntot * ntot),
        ("P2", a.p2.len() >= ntot * ntot),
        ("ldet2", a.ldet2.len() >= ntot * ntot),
        ("dVkk", a.dvkk.len() >= m * m),
        ("rSncol", a.rsncol.len() >= m),
    ];
    for (name, ok) in checks {
        if !ok {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} is too short"
            )));
        }
    }
    py.allow_threads(|| unsafe { gdi2(&mut a) });
    let out = PyDict::new(py);
    out.set_item("X", a.x.into_pyarray(py))?;
    out.set_item("E", a.e.into_pyarray(py))?;
    out.set_item("rS", a.rs.into_pyarray(py))?;
    out.set_item("w", a.w.into_pyarray(py))?;
    out.set_item("beta", a.beta.into_pyarray(py))?;
    out.set_item("b1", a.b1.into_pyarray(py))?;
    out.set_item("w1", a.w1.into_pyarray(py))?;
    out.set_item("D1", a.d1.into_pyarray(py))?;
    out.set_item("D2", a.d2.into_pyarray(py))?;
    out.set_item("P", vec![a.p0].into_pyarray(py))?;
    out.set_item("P1", a.p1.into_pyarray(py))?;
    out.set_item("P2", a.p2.into_pyarray(py))?;
    out.set_item("ldet", vec![a.ldet].into_pyarray(py))?;
    out.set_item("ldet1", a.ldet1.into_pyarray(py))?;
    out.set_item("ldet2", a.ldet2.into_pyarray(py))?;
    out.set_item("rV", a.rv.into_pyarray(py))?;
    out.set_item("rank.est", vec![a.rank_est as f64].into_pyarray(py))?;
    out.set_item("type", vec![a.type_ as f64].into_pyarray(py))?;
    out.set_item("dVkk", a.dvkk.into_pyarray(py))?;
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tr_bt_ab_py, m)?)?;
    m.add_function(wrap_pyfunction!(gdi1_c, m)?)?;
    m.add_function(wrap_pyfunction!(gdi2_c, m)?)?;
    m.add_function(wrap_pyfunction!(pls_fit1_c, m)?)?;
    Ok(())
}
