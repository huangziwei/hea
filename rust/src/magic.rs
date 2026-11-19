//! mgcv's C `magic` (magic.c): GCV/UBRE smoothing parameter selection for a
//! penalized least squares problem, by Newton's method backed by steepest
//! descent with step halving, on the QR-reduced problem; `fit_magic` fits
//! at given smoothing parameters through the SVD of `[R; sqrt(S)]` and
//! `magic_gH` returns the score's gradient and Hessian.
//!
//! The arithmetic is that of R's arm64 mgcv.so (clang): a dot product clang
//! vectorizes (2 lanes, 4 registers) adds the first `8 floor(n/8)` products
//! rounded and in order and fuses the rest, a sum of squares it unrolls by
//! four does the same in fours, and every other `x += a*b` is one fused
//! multiply-add. The BLAS and LAPACK calls are the ports of R's bundled
//! LAPACK on the vendor BLAS.

use numpy::ndarray::ShapeBuilder;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::lapack::chol::dpstrf_u;
use crate::lapack::fma;
use crate::lapack::ormtr::{dormqr_l, dormqr_l_lwork};
use crate::lapack::qr::{dgeqp3_free, dgeqp3_lwork};
use crate::lapack::syevr::{dsyevr_all_lower, dsyevr_lwork};
use crate::mgcv_c::{dot_s8, get_xtx, mgcv_mmult, mgcv_svd_full, sumsq_s4};

/// mgcv's C `mroot`: the pivoted Cholesky factor of the symmetric `n`-by-`n`
/// `a` (`mgcv_chol`: `dpstrf('U')` at its default tolerance, then the
/// subdiagonal zeroed), unpivoted and cut to its first `rank` rows: on exit
/// `a` holds `B` (`rank`-by-`n`, leading dimension `rank`) with
/// `B'B = a`. `rank <= 0` takes the numerical rank. Returns the rank.
unsafe fn mroot(a: &mut [f64], rank_in: i32, n: usize) -> usize {
    let mut piv = vec![0i32; n];
    let mut work = vec![0.0f64; 2 * n];
    let (erank, _info) = dpstrf_u(
        n as i32,
        a.as_mut_ptr(),
        n as i32,
        piv.as_mut_ptr(),
        -1.0,
        work.as_mut_ptr(),
    );
    for j in 0..n {
        for i in j + 1..n {
            a[i + j * n] = 0.0;
        }
    }
    let rank = (if rank_in <= 0 { erank } else { rank_in }) as usize;
    let mut b = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..=i {
            b[k + i * n] = a[k + i * n];
            a[k + i * n] = 0.0;
        }
    }
    for i in 0..n {
        let j = (piv[i] - 1) as usize;
        for k in 0..=i {
            a[k + j * n] = b[k + i * n];
        }
    }
    let mut pi = 0;
    for i in 0..n {
        for k in 0..rank {
            a[pi] = a[k + i * n];
            pi += 1;
        }
    }
    rank
}

/// What `fit_magic` leaves for `magic_gH` and the driver.
struct Fit {
    y1: Vec<f64>,
    u1: Vec<f64>,
    v: Vec<f64>,
    d: Vec<f64>,
    b: Vec<f64>,
    score: f64,
    norm: f64,
    delta: f64,
    rank: usize,
}

/// The problem `fit_magic` solves at each set of smoothing parameters.
struct Problem<'a> {
    /// The pivoted QR of the `n`-by-`q` model matrix (`dgeqp3`).
    x: &'a [f64],
    n: usize,
    q: usize,
    /// The `S_i = sqrt(S_i) sqrt(S_i)'`, pivoted, each `q`-by-`q`.
    si: &'a [Vec<f64>],
    /// The pivoted fixed penalty.
    h: Option<&'a [f64]>,
    gamma: f64,
    gcv: bool,
    rank_tol: f64,
    yy: f64,
    y0: &'a [f64],
    norm_const: f64,
    n_score: i32,
}

/// `fit_magic`: the fit at log smoothing parameters `sp`, from the SVD of
/// the QR factor stacked on a square root of `H + sum_i exp(sp_i) S_i`,
/// truncated at `rank_tol` times the largest singular value; the GCV (with
/// the scale estimate) or UBRE score.
unsafe fn fit_magic(pr: &Problem, sp: &[f64], scale: &mut f64, f: &mut Fit) {
    let (n, q) = (pr.n, pr.q);
    let m = pr.si.len();
    let mut st = vec![0.0f64; q * q];
    if let Some(h) = pr.h {
        st.copy_from_slice(h);
    }
    for k in 0..m {
        let xx = sp[k].exp();
        for (s, &p) in st.iter_mut().zip(pr.si[k].iter()) {
            *s = fma(p, xx, *s);
        }
    }
    let rank_s = if m > 0 || pr.h.is_some() {
        mroot(&mut st, -1, q)
    } else {
        0
    };
    // The augmented R: getRpqr's upper triangle over the penalty root.
    let r = rank_s + q;
    let mut rm = vec![0.0f64; r * q];
    for i in 0..q.min(r) {
        for j in 0..q {
            rm[i + r * j] = if i > j { 0.0 } else { pr.x[i + n * j] };
        }
    }
    for j in 0..q {
        for i in q..r {
            rm[i + r * j] = st[(i - q) + rank_s * j];
        }
    }
    let mut vt = vec![0.0f64; q * q];
    mgcv_svd_full(
        rm.as_mut_ptr(),
        vt.as_mut_ptr(),
        f.d.as_mut_ptr(),
        r as i32,
        q as i32,
    );
    let mut rank = q;
    let xx = f.d[0] * pr.rank_tol;
    while f.d[rank - 1] < xx {
        rank -= 1;
    }
    for i in 0..q {
        for j in 0..rank {
            f.v[i + q * j] = vt[j + q * i];
        }
    }
    for i in 0..q {
        for j in 0..rank {
            f.u1[i + q * j] = rm[i + r * j];
        }
    }
    for i in 0..rank {
        f.y1[i] = dot_s8(&f.u1[i * q..i * q + q], &pr.y0[..q]);
    }
    let y_ay = sumsq_s4(&f.y1[..rank]);
    for i in 0..q {
        let mut s = 0.0;
        for j in 0..rank {
            s = fma(f.u1[i + q * j], f.y1[j], s);
        }
        f.b[i] = s;
    }
    let y_aay = sumsq_s4(&f.b[..q]);
    let mut norm = fma(y_ay, -2.0, pr.yy) + y_aay;
    if norm < 0.0 {
        norm = 0.0;
    }
    let tr_a = sumsq_s4(&f.u1[..q * rank]);
    let a: Vec<f64> = (0..rank).map(|i| f.y1[i] / f.d[i]).collect();
    for i in 0..q {
        let mut s = 0.0;
        for j in 0..rank {
            s = fma(f.v[i + q * j], a[j], s);
        }
        f.b[i] = s;
    }
    let nn = pr.n_score as f64;
    let xx = fma(-pr.gamma, tr_a, nn);
    f.delta = xx;
    let rss = norm + pr.norm_const;
    if pr.gcv {
        f.score = rss * nn / (xx * xx);
        *scale = rss / (nn - tr_a);
    } else {
        f.score = *scale + fma((*scale * -2.0) / nn, xx, rss / nn);
    }
    f.norm = norm;
    f.rank = rank;
}

/// `magic_gH`'s workspace and results.
struct Deriv {
    u1u1: Vec<f64>,
    mm: Vec<Vec<f64>>,
    kk: Vec<Vec<f64>>,
    vs: Vec<f64>,
    my: Vec<Vec<f64>>,
    ky: Vec<Vec<f64>>,
    yk: Vec<Vec<f64>>,
    hess: Vec<Vec<f64>>,
    grad: Vec<f64>,
    dnorm: Vec<f64>,
    ddelta: Vec<f64>,
    d2norm: Vec<Vec<f64>>,
    d2delta: Vec<Vec<f64>>,
}

impl Deriv {
    fn new(q: usize, m: usize) -> Self {
        Deriv {
            u1u1: vec![0.0; q * q],
            mm: vec![vec![0.0; q * q]; m],
            kk: vec![vec![0.0; q * q]; m],
            vs: vec![0.0; q * q],
            my: vec![vec![0.0; q]; m],
            ky: vec![vec![0.0; q]; m],
            yk: vec![vec![0.0; q]; m],
            hess: vec![vec![0.0; m]; m],
            grad: vec![0.0; m],
            dnorm: vec![0.0; m],
            ddelta: vec![0.0; m],
            d2norm: vec![vec![0.0; m]; m],
            d2delta: vec![vec![0.0; m]; m],
        }
    }
}

/// `magic_gH`: the gradient and Hessian of the GCV or UBRE score with
/// respect to the log smoothing parameters `sp`, from `fit_magic`'s SVD.
/// `s` holds the pivoted penalty roots side by side (`q` rows, `cs[i]`
/// columns each, starting at column `cucs[i]`).
#[allow(clippy::too_many_arguments)]
unsafe fn magic_gh(
    g: &mut Deriv,
    f: &Fit,
    sp: &[f64],
    s: &[f64],
    q: usize,
    cs: &[i32],
    cucs: &[usize],
    gcv: bool,
    gamma: f64,
    scale: f64,
    n: i32,
    norm_const: f64,
) {
    let rank = f.rank;
    let m = cs.len();
    get_xtx(g.u1u1.as_mut_ptr(), f.u1.as_ptr(), q as i32, rank as i32);
    for i in 0..m {
        let ci = cs[i];
        let p = s.as_ptr().add(cucs[i] * q);
        // V'S_i^0.5 (rank by cS[i]), then D^{-1} V'S_i^0.5.
        mgcv_mmult(
            g.vs.as_mut_ptr(),
            f.v.as_ptr(),
            p,
            true,
            false,
            rank as i32,
            ci,
            q as i32,
        );
        for j in 0..ci as usize {
            for k in 0..rank {
                g.vs[k + j * rank] /= f.d[k];
            }
        }
        mgcv_mmult(
            g.mm[i].as_mut_ptr(),
            g.vs.as_ptr(),
            g.u1u1.as_ptr(),
            true,
            false,
            ci,
            rank as i32,
            rank as i32,
        );
        mgcv_mmult(
            g.kk[i].as_mut_ptr(),
            g.vs.as_ptr(),
            g.mm[i].as_ptr(),
            false,
            false,
            rank as i32,
            rank as i32,
            ci,
        );
        mgcv_mmult(
            g.mm[i].as_mut_ptr(),
            g.vs.as_ptr(),
            g.vs.as_ptr(),
            false,
            true,
            rank as i32,
            rank as i32,
            ci,
        );
        for k in 0..rank {
            g.my[i][k] = dot_s8(&f.y1[..rank], &g.mm[i][k * rank..k * rank + rank]);
        }
        for k in 0..rank {
            g.yk[i][k] = dot_s8(&f.y1[..rank], &g.kk[i][k * rank..k * rank + rank]);
        }
        for k in 0..rank {
            let mut x = 0.0;
            for c in 0..rank {
                x = fma(f.y1[c], g.kk[i][k + c * rank], x);
            }
            g.ky[i][k] = x;
        }
    }
    let nv8 = if rank >= 8 { rank / 8 * 8 } else { 0 };
    for i in 0..m {
        let mut xx = 0.0;
        for k in 0..rank {
            xx += g.kk[i][k * (rank + 1)];
        }
        g.ddelta[i] = gamma * xx * sp[i].exp();
        for j in 0..=i {
            let xx = dot_s8(&g.mm[j][..rank * rank], &g.kk[i][..rank * rank]);
            let v = gamma * -2.0 * (sp[i] + sp[j]).exp() * xx;
            g.d2delta[j][i] = v;
            g.d2delta[i][j] = v;
        }
        g.d2delta[i][i] += g.ddelta[i];
        let mut xx = 0.0;
        for k in 0..nv8 {
            xx += f.y1[k] * (g.my[i][k] - g.ky[i][k]);
        }
        for k in nv8..rank {
            xx = fma(f.y1[k], g.my[i][k] - g.ky[i][k], xx);
        }
        let e = sp[i].exp();
        g.dnorm[i] = (e + e) * xx;
        for j in 0..=i {
            let mut xx = 0.0;
            for k in 0..rank {
                let (p1, p2, p3, p4, p) =
                    (g.my[i][k], g.my[j][k], g.ky[i][k], g.ky[j][k], g.yk[i][k]);
                let t = fma(p1, p4, p2 * p3);
                let t = fma(p1 * -2.0, p2, t);
                xx += fma(p, p2, t);
            }
            let v = (xx + xx) * (sp[i] + sp[j]).exp();
            g.d2norm[i][j] = v;
            g.d2norm[j][i] = v;
        }
        g.d2norm[i][i] += g.dnorm[i];
    }
    let nf = n as f64;
    if gcv {
        let norm = f.norm + norm_const;
        let delta = f.delta;
        let xx = nf / (delta * delta);
        let xx1 = (xx + xx) * norm / delta;
        let x1 = xx * -2.0 / delta;
        let x2 = xx1 * 3.0 / delta;
        for i in 0..m {
            g.grad[i] = fma(xx, g.dnorm[i], g.ddelta[i] * -xx1);
            for j in 0..=i {
                let sd = fma(g.ddelta[j], g.dnorm[i], g.ddelta[i] * g.dnorm[j]);
                let h = fma(x1, sd, xx * g.d2norm[i][j]);
                let h = fma(x2 * g.ddelta[i], g.ddelta[j], h);
                let h = fma(-xx1, g.d2delta[i][j], h);
                g.hess[i][j] = h;
                g.hess[j][i] = h;
            }
        }
    } else {
        for i in 0..m {
            g.grad[i] = fma(scale * -2.0, g.ddelta[i], g.dnorm[i]) / nf;
            for j in 0..=i {
                let h = fma(scale * -2.0, g.d2delta[i][j], g.d2norm[i][j]) / nf;
                g.hess[i][j] = h;
                g.hess[j][i] = h;
            }
        }
    }
}

/// `magic`'s results, as the R wrapper reads them back.
pub(crate) struct MagicOut {
    pub b: Vec<f64>,
    pub sp: Vec<f64>,
    pub score: f64,
    pub scale: f64,
    pub rank: usize,
    pub converged: bool,
    pub hess_pos_def: bool,
    pub iter: i32,
    pub score_calls: i32,
    pub rms_grad: f64,
    /// `q`-by-`rank`, unpivoted.
    pub rv: Vec<f64>,
    /// The unpivoted QR factor, `q`-by-`q`.
    pub r: Vec<f64>,
}

/// `sp = L sp0 + lsp0` (or `sp = sp0` without `L`).
unsafe fn map_sp(sp: &mut [f64], l: Option<&[f64]>, sp0: &[f64], lsp0: &[f64]) {
    let m = sp.len();
    match l {
        Some(l) => {
            let mp = sp0.len();
            sp.iter_mut().for_each(|v| *v = 0.0);
            if mp > 0 {
                mgcv_mmult(
                    sp.as_mut_ptr(),
                    l.as_ptr(),
                    sp0.as_ptr(),
                    false,
                    false,
                    m as i32,
                    1,
                    mp as i32,
                );
            }
            for i in 0..m {
                sp[i] += lsp0[i];
            }
        }
        None => sp.copy_from_slice(&sp0[..m]),
    }
}

/// `magic` (magic.c): GCV (`gcv`) or UBRE smoothing parameter selection for
/// `||y - Xb||^2 + b'Hb + sum_i exp(sp_i) b'S_i b` with `sp = L log(sp0) +
/// lsp0`. `x` is `n`-by-`q`; `s` holds the penalty roots side by side
/// (`cs[i]` columns each); `sp0` with any entry `<= 0` starts from
/// `log(def_sp)`. `mp < 0`: no `L` (`sp = log(sp0)`).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn magic(
    y: &[f64],
    x: &mut [f64],
    n: usize,
    q: usize,
    sp0_in: &[f64],
    def_sp_in: &[f64],
    s_in: &[f64],
    h_in: Option<&[f64]>,
    l: Option<&[f64]>,
    lsp0: &[f64],
    gamma: f64,
    scale_in: f64,
    gcv: bool,
    max_half: i32,
    cs: &[i32],
    rank_tol: f64,
    tol: f64,
    norm_const: f64,
    n_score: i32,
) -> Result<MagicOut, &'static str> {
    let m = cs.len();
    let l_exists = l.is_some();
    let mp = if l_exists { sp0_in.len() } else { m };
    let mut scale = scale_in;

    // The pivoted QR of X (mgcv_pqr with one thread: dgeqp3, every column free).
    let mut tau = vec![0.0f64; q.max(1)];
    let mut pivot = vec![0i32; q.max(1)];
    let lw = dgeqp3_lwork(n as i32, q as i32).max(1);
    let mut work = vec![0.0f64; lw as usize];
    dgeqp3_free(
        n as i32,
        q as i32,
        x.as_mut_ptr(),
        n as i32,
        pivot.as_mut_ptr(),
        tau.as_mut_ptr(),
        work.as_mut_ptr(),
        lw,
    );
    let pivot: Vec<usize> = pivot[..q].iter().map(|&p| (p - 1) as usize).collect();
    let mut cucs = vec![0usize; m];
    for i in 1..m {
        cucs[i] = cucs[i - 1] + cs[i - 1] as usize;
    }
    let scs: usize = cs.iter().map(|&c| c as usize).sum();

    // Pivot the rows of the penalty roots and of H.
    let mut s = s_in[..q * scs].to_vec();
    let mut tmp = vec![0.0f64; q];
    for col in 0..scs {
        for i in 0..q {
            tmp[i] = s[pivot[i] + col * q];
        }
        s[col * q..col * q + q].copy_from_slice(&tmp);
    }
    let h = h_in.map(|h0| {
        let mut h = h0[..q * q].to_vec();
        for j in 0..q {
            for i in 0..q {
                tmp[i] = h[pivot[i] + q * j];
            }
            for i in 0..q {
                h[i + q * j] = tmp[i];
            }
        }
        for i in 0..q {
            for j in 0..q {
                tmp[j] = h[i + pivot[j] * q];
            }
            for j in 0..q {
                h[i + q * j] = tmp[j];
            }
        }
        h
    });

    // y0 = Q'y (its first q entries are Q_1'y), and y'y.
    let mut y0 = y[..n].to_vec();
    let lw = dormqr_l_lwork(1).max(1);
    let mut work = vec![0.0f64; lw as usize];
    dormqr_l(
        b'T',
        n as i32,
        1,
        q as i32,
        x.as_ptr(),
        n as i32,
        tau.as_ptr(),
        y0.as_mut_ptr(),
        n as i32,
        work.as_mut_ptr(),
        lw,
    );
    let yy = sumsq_s4(&y[..n]);

    // The S_i from their roots.
    let mut si = vec![vec![0.0f64; q * q]; m];
    for k in 0..m {
        let p = s.as_ptr().add(cucs[k] * q);
        mgcv_mmult(
            si[k].as_mut_ptr(),
            p,
            p,
            false,
            true,
            q as i32,
            q as i32,
            cs[k],
        );
    }

    let mut sp0 = sp0_in[..mp].to_vec();
    let mut def_sp = def_sp_in[..mp].to_vec();
    let autoinit = sp0.iter().any(|&v| v <= 0.0);
    let def_supplied = !def_sp.iter().any(|&v| v <= 0.0);
    if l_exists && !def_supplied {
        return Err("magic requires smoothing parameter starting values if L supplied");
    }

    // The R factor, for a norm of X and for the output.
    let mut rq = vec![0.0f64; q * q];
    for i in 0..q {
        for j in 0..q {
            rq[i + q * j] = if i > j { 0.0 } else { x[i + n * j] };
        }
    }
    if m > 0 && !def_supplied {
        let mut xms = 0.0;
        for j in 0..q {
            for i in 0..=j {
                let v = rq[i + q * j];
                xms = fma(v, v, xms);
            }
        }
        xms /= (n * q) as f64;
        for i in 0..m {
            let mut x = 0.0;
            for k in 0..q {
                x += si[i][k * (q + 1)];
            }
            let r_sms = x / (q as i32 * cs[i]) as f64;
            def_sp[i] = xms / r_sms;
        }
    }
    for i in 0..mp {
        sp0[i] = if autoinit {
            def_sp[i].ln()
        } else {
            sp0[i].ln()
        };
    }
    let mut sp = vec![0.0f64; m];
    map_sp(&mut sp, l, &sp0, lsp0);

    let mut f = Fit {
        y1: vec![0.0; q],
        u1: vec![0.0; q * q],
        v: vec![0.0; q * q],
        d: vec![0.0; q],
        b: vec![0.0; q],
        score: 0.0,
        norm: 0.0,
        delta: 0.0,
        rank: 0,
    };
    let pr = Problem {
        x,
        n,
        q,
        si: &si,
        h: h.as_deref(),
        gamma,
        gcv,
        rank_tol,
        yy,
        y0: &y0,
        norm_const,
        n_score,
    };
    fit_magic(&pr, &sp, &mut scale, &mut f);
    let mut fit_call = 1;
    let mut g = Deriv::new(q, m);
    let mut grad = vec![0.0f64; mp];
    let lt_grad = |grad: &mut [f64], g1: &[f64]| {
        if let Some(l) = l {
            mgcv_mmult(
                grad.as_mut_ptr(),
                l.as_ptr(),
                g1.as_ptr(),
                true,
                false,
                mp as i32,
                1,
                m as i32,
            );
        } else {
            grad.copy_from_slice(&g1[..mp]);
        }
    };

    // Reset to the default any smoothing parameter the score is flat in.
    if mp > 0 && !autoinit {
        magic_gh(
            &mut g, &f, &sp, &s, q, cs, &cucs, gcv, gamma, scale, n_score, norm_const,
        );
        let xx = 1e-4 * (1.0 + f.score.abs());
        let mut ok = true;
        lt_grad(&mut grad, &g.grad);
        for i in 0..mp {
            if grad[i].abs() < xx {
                sp0[i] = def_sp[i].ln();
                ok = false;
            }
        }
        map_sp(&mut sp, l, &sp0, lsp0);
        if !ok {
            fit_magic(&pr, &sp, &mut scale, &mut f);
            fit_call += 1;
        }
    }

    let mut min_score = f.score;
    let mut sd_step = vec![0.0f64; mp];
    let mut n_step = vec![0.0f64; mp];
    let mut use_sd = false;
    let mut iter = 0;
    let mut step_fail = false;
    let mut msg = 0.0;
    let mut d_score = 1e10;

    if autoinit && !def_supplied {
        // Second guesses: scale rank(S_i) / b'S_i b.
        let mut col = 0;
        for k in 0..m {
            let ck = cs[k] as usize;
            for j in 0..ck {
                f.d[j] = dot_s8(&s[(col + j) * q..(col + j) * q + q], &f.b[..q]);
            }
            col += ck;
            let mut xx = 0.0;
            for j in 0..ck {
                xx = fma(f.d[j], f.d[j], xx);
            }
            sd_step[k] = (scale * cs[k] as f64 / xx).ln() - sp[k];
        }
        use_sd = true;
    }

    if mp > 0 {
        let mut converged = false;
        let mut nsp = vec![0.0f64; mp];
        let mut u = vec![0.0f64; m * m];
        while !converged {
            iter += 1;
            if iter > 400 {
                return Err(
                    "magic, the gcv/ubre optimizer, failed to converge after 400 iterations.",
                );
            }
            let mut ok = iter > 1 || (autoinit && !def_supplied);
            let mut try_ = 0;
            let mut step_sd = use_sd;
            while ok {
                try_ += 1;
                if try_ == 4 && !use_sd {
                    use_sd = true;
                    step_sd = true;
                }
                {
                    let step = if step_sd { &sd_step } else { &n_step };
                    for i in 0..mp {
                        nsp[i] = sp0[i] + step[i];
                    }
                }
                map_sp(&mut sp, l, &nsp, lsp0);
                fit_magic(&pr, &sp, &mut scale, &mut f);
                fit_call += 1;
                let step = if step_sd { &mut sd_step } else { &mut n_step };
                if f.score < min_score {
                    ok = false;
                    d_score = min_score - f.score;
                    min_score = f.score;
                    sp0.copy_from_slice(&nsp);
                } else {
                    for v in step.iter_mut() {
                        *v /= 2.0;
                    }
                }
                if try_ == max_half - 1 && ok {
                    for v in step.iter_mut() {
                        *v = 0.0;
                    }
                }
                if try_ == max_half {
                    ok = false;
                }
            }
            if iter > 3 {
                converged = true;
                if d_score > tol * (1.0 + min_score) {
                    converged = false;
                }
                let mut xx = 0.0;
                for &gv in &grad {
                    xx = fma(gv, gv, xx);
                }
                xx = xx.sqrt();
                if xx > tol.powf(1.0 / 3.0) * (1.0 + min_score.abs()) {
                    converged = false;
                }
                if try_ == max_half {
                    converged = true;
                }
                if converged {
                    msg = (xx * xx / mp as f64).sqrt();
                    if try_ == max_half {
                        step_fail = true;
                    }
                }
            }

            // The derivatives at the new point, and the search directions.
            map_sp(&mut sp, l, &sp0, lsp0);
            magic_gh(
                &mut g, &f, &sp, &s, q, cs, &cucs, gcv, gamma, scale, n_score, norm_const,
            );
            for i in 0..m {
                for j in 0..m {
                    u[i + m * j] = g.hess[i][j];
                }
            }
            lt_grad(&mut grad, &g.grad);
            let mut uh = if let Some(l) = l {
                let mut u0 = vec![0.0f64; m * mp];
                mgcv_mmult(
                    u0.as_mut_ptr(),
                    l.as_ptr(),
                    u.as_ptr(),
                    true,
                    false,
                    mp as i32,
                    m as i32,
                    m as i32,
                );
                let mut uh = vec![0.0f64; mp * mp];
                mgcv_mmult(
                    uh.as_mut_ptr(),
                    u0.as_ptr(),
                    l.as_ptr(),
                    false,
                    false,
                    mp as i32,
                    mp as i32,
                    m as i32,
                );
                uh
            } else {
                u.clone()
            };
            let ev = mgcv_symeig_asc(&mut uh, mp);
            use_sd = ev.iter().any(|&e| e < 0.0);
            if !use_sd {
                for i in 0..mp {
                    sd_step[i] = dot_s8(&uh[mp * i..mp * i + mp], &grad) / ev[i];
                }
                for i in 0..mp {
                    let mut x = 0.0;
                    for j in 0..mp {
                        x = fma(uh[i + mp * j], sd_step[j], x);
                    }
                    n_step[i] = -x;
                }
                let mut xx = n_step[0].abs();
                for v in &n_step[1..] {
                    if v.abs() > xx {
                        xx = v.abs();
                    }
                }
                if xx > 5.0 {
                    let xx = 5.0 / xx;
                    for v in n_step.iter_mut() {
                        *v *= xx;
                    }
                }
            }
            let mut xx = grad[0].abs();
            for v in &grad[1..] {
                if xx < v.abs() {
                    xx = v.abs();
                }
            }
            for i in 0..mp {
                sd_step[i] = -grad[i] / xx;
            }
        }

        // Check that no smoothing parameter's optimum is at +/- infinity.
        for k in 0..mp {
            let mut ok = 5;
            let xx = 2.0;
            let sign = if grad[k] < 0.0 { 1.0 } else { -1.0 };
            while ok > 0 {
                sp0[k] += sign * xx;
                map_sp(&mut sp, l, &sp0, lsp0);
                ok -= 1;
                fit_magic(&pr, &sp, &mut scale, &mut f);
                if f.score < min_score {
                    min_score = f.score;
                } else {
                    ok = 0;
                    sp0[k] += -sign * xx;
                }
            }
        }
        map_sp(&mut sp, l, &sp0, lsp0);
        fit_magic(&pr, &sp, &mut scale, &mut f);
    }

    // rV = V D^{-1}, unpivoted; the parameters unpivoted.
    let rank = f.rank;
    for j in 0..rank {
        for i in 0..q {
            f.v[i + q * j] /= f.d[j];
        }
    }
    let mut rv = vec![0.0f64; q * rank];
    for col in 0..rank {
        for i in 0..q {
            rv[pivot[i] + q * col] = f.v[i + q * col];
        }
    }
    let mut b = vec![0.0f64; q];
    for i in 0..q {
        b[pivot[i]] = f.b[i];
    }
    let sp_out: Vec<f64> = sp0.iter().map(|v| v.exp()).collect();
    // The QR factor with its columns unpivoted.
    let mut r = vec![0.0f64; q * q];
    for i in 0..q {
        for j in i..q {
            r[i + q * pivot[j]] = rq[i + q * j];
        }
    }
    Ok(MagicOut {
        b,
        sp: sp_out,
        score: f.score,
        scale,
        rank,
        converged: !step_fail,
        hess_pos_def: !use_sd,
        iter,
        score_calls: fit_call,
        rms_grad: msg,
        rv,
        r,
    })
}

/// mgcv's `mgcv_symeig` with `dsyevr`: the eigenvalues of the symmetric
/// `n`-by-`n` `a` in ascending order, its eigenvectors over `a`.
unsafe fn mgcv_symeig_asc(a: &mut [f64], n: usize) -> Vec<f64> {
    let (lwork, liwork) = dsyevr_lwork(n as i32);
    let mut w = vec![0.0f64; n.max(1)];
    let mut z = vec![0.0f64; (n * n).max(1)];
    let mut isuppz = vec![0i32; 2 * n.max(1)];
    let mut work = vec![0.0f64; lwork.max(1)];
    let mut iwork = vec![0i32; liwork.max(1)];
    dsyevr_all_lower(
        true,
        n as i32,
        a.as_mut_ptr(),
        n.max(1) as i32,
        w.as_mut_ptr(),
        z.as_mut_ptr(),
        n.max(1) as i32,
        isuppz.as_mut_ptr(),
        work.as_mut_ptr(),
        lwork as i32,
        iwork.as_mut_ptr(),
    );
    a[..n * n].copy_from_slice(&z[..n * n]);
    w.truncate(n);
    w
}

/// A column-major copy of a 2-d array.
fn col_major(a: &numpy::ndarray::ArrayView2<'_, f64>) -> Vec<f64> {
    let (r, c) = (a.shape()[0], a.shape()[1]);
    let mut v = vec![0.0f64; r * c];
    for j in 0..c {
        for i in 0..r {
            v[i + r * j] = a[[i, j]];
        }
    }
    v
}

/// mgcv's C `magic` on the `n`-by-`q` `x`, as mgcv's R `magic()` calls it
/// with one thread: see [`magic`]. `s` (`q` rows) holds the penalty roots
/// side by side, `cs[i]` columns each; `h` is the fixed penalty and `l` the
/// `m`-by-`mp` map, `None` when absent. Returns `(b, sp, score, scale,
/// info, rms_grad, rv, r)` with `info = (rank, converged, hess_pos_def,
/// iter, score_calls)`, `rv` (`q`-by-`rank`) and the unpivoted QR factor
/// `r` (`q`-by-`q`).
#[pyfunction]
#[pyo3(signature = (y, x, sp0, def_sp, s, h, l, lsp0, gamma, scale, gcv, max_half, cs, rank_tol, tol, norm_const, n_score))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn magic_c<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    sp0: PyReadonlyArray1<'py, f64>,
    def_sp: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray2<'py, f64>,
    h: Option<PyReadonlyArray2<'py, f64>>,
    l: Option<PyReadonlyArray2<'py, f64>>,
    lsp0: PyReadonlyArray1<'py, f64>,
    gamma: f64,
    scale: f64,
    gcv: bool,
    max_half: i32,
    cs: Vec<i32>,
    rank_tol: f64,
    tol: f64,
    norm_const: f64,
    n_score: i32,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    (usize, bool, bool, i32, i32),
    f64,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let xv = x.as_array();
    let (n, q) = (xv.shape()[0], xv.shape()[1]);
    let mut xc = col_major(&xv);
    let sv = s.as_array();
    let scs: usize = cs.iter().map(|&c| c.max(0) as usize).sum();
    if sv.shape()[0] != q || sv.shape()[1] != scs {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s must be q by sum(cs)",
        ));
    }
    let sc = col_major(&sv);
    let hc = h.map(|h| col_major(&h.as_array()));
    let lc = l.map(|l| col_major(&l.as_array()));
    let y = y.as_slice()?.to_vec();
    let sp0 = sp0.as_slice()?.to_vec();
    let def_sp = def_sp.as_slice()?.to_vec();
    let lsp0 = lsp0.as_slice()?.to_vec();
    let out = py.allow_threads(|| {
        // SAFETY: `xc` is n-by-q, `sc` q-by-sum(cs), `hc` q-by-q and `lc`
        // m-by-mp, the dimensions `magic` reads.
        unsafe {
            magic(
                &y,
                &mut xc,
                n,
                q,
                &sp0,
                &def_sp,
                &sc,
                hc.as_deref(),
                lc.as_deref(),
                &lsp0,
                gamma,
                scale,
                gcv,
                max_half,
                &cs,
                rank_tol,
                tol,
                norm_const,
                n_score,
            )
        }
    });
    let out = out.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let rank = out.rank;
    let rv = numpy::ndarray::Array2::from_shape_vec((q, rank).f(), out.rv).unwrap();
    let r = numpy::ndarray::Array2::from_shape_vec((q, q).f(), out.r).unwrap();
    Ok((
        out.b.into_pyarray(py),
        out.sp.into_pyarray(py),
        out.score,
        out.scale,
        (
            rank,
            out.converged,
            out.hess_pos_def,
            out.iter,
            out.score_calls,
        ),
        out.rms_grad,
        rv.into_pyarray(py),
        r.into_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(magic_c, m)?)?;
    Ok(())
}
