//! mgcv's small C helpers as R's arm64 mgcv.so computes them, shared by the
//! ports of its C routines: clang's in-order 8-wide dot product and 4-wide
//! sum of squares, the BLAS- and LAPACK-backed products, solves and
//! factorizations of mat.c, and gdi.c's matrix bookkeeping (pivoting,
//! dropping and restoring rows and columns, `R^{-1}`, the condition
//! estimate).
//!
//! clang fuses every `x += a*b` and `x -= a*b`. A reduction over contiguous
//! memory it vectorizes, 8 products a pass rounded and added in order, the
//! remainder fused ([`dot_s8`]); one over a strided operand it keeps scalar
//! and fused.

use crate::lapack::blas::{dgemm, dgemv, dsyrk, dtrsm};
use crate::lapack::fma;
use crate::lapack::gesdd::{dgesdd, dgesdd_lwork};
use crate::lapack::ormtr::{dormqr_l, dormqr_l_lwork};
use crate::lapack::qr::{dgeqp3, dgeqp3_lwork};

/// `sum a[i] b[i]` as clang vectorizes it: for `n >= 8` the first
/// `8 floor(n/8)` products rounded and added in order, the rest fused.
pub(crate) fn dot_s8(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let nv = if n >= 8 { n / 8 * 8 } else { 0 };
    let mut s = 0.0;
    for i in 0..nv {
        s += a[i] * b[i];
    }
    for i in nv..n {
        s = fma(a[i], b[i], s);
    }
    s
}

/// `sum a[i] b[i]` accumulated through memory clang cannot prove distinct
/// from `a` and `b`: below `min_n` terms every product fused into the sum;
/// from `min_n` on (after its run-time overlap check passes) as
/// [`dot_s8`].
pub(crate) fn dot_s8_from(a: &[f64], b: &[f64], min_n: usize) -> f64 {
    if a.len() >= min_n {
        return dot_s8(a, b);
    }
    let mut s = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        s = fma(x, y, s);
    }
    s
}

/// `sum a[i]^2` as clang unrolls it: for `n > 3` the first `4 floor(n/4)`
/// squares rounded and added in order, the rest fused.
pub(crate) fn sumsq_s4(a: &[f64]) -> f64 {
    let n = a.len();
    let nv = if n > 3 { n / 4 * 4 } else { 0 };
    let mut s = 0.0;
    for &x in &a[..nv] {
        s += x * x;
    }
    for &x in &a[nv..] {
        s = fma(x, x, s);
    }
    s
}

/// mgcv's `getXtX`: `X'X` for the `r`-by-`c` `x` by `dsyrk('L', 'T')`, the
/// upper triangle copied from the lower.
pub(crate) unsafe fn get_xtx(xtx: *mut f64, x: *const f64, r: i32, c: i32) {
    dsyrk(b'L', b'T', c, r, 1.0, x, r, 0.0, xtx, c);
    let c = c as usize;
    for i in 0..c {
        for j in 0..i {
            *xtx.add(j + i * c) = *xtx.add(i + j * c);
        }
    }
}

/// mgcv's `getXXt`: `X X'` for the `r`-by-`c` `x` by `dsyrk('L', 'N')`, the
/// upper triangle copied from the lower.
pub(crate) unsafe fn get_xxt(xxt: *mut f64, x: *const f64, r: i32, c: i32) {
    dsyrk(b'L', b'N', r, c, 1.0, x, r, 0.0, xxt, r);
    let r = r as usize;
    for i in 0..r {
        for j in 0..i {
            *xxt.add(j + i * r) = *xxt.add(i + j * r);
        }
    }
}

/// mgcv's `mgcv_mmult`: `A = op(B) op(C)` (`r`-by-`c`, common dimension
/// `n`) by one `dgemm`, or by `getXtX`/`getXXt` when `B` and `C` are the
/// same matrix and the product is its cross product.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mgcv_mmult(
    a: *mut f64,
    b: *const f64,
    c: *const f64,
    bt: bool,
    ct: bool,
    r: i32,
    cc: i32,
    n: i32,
) {
    if r <= 0 || cc <= 0 || n <= 0 {
        return;
    }
    if std::ptr::eq(b, c) {
        if bt && !ct && r == cc {
            get_xtx(a, b, n, r);
            return;
        } else if ct && !bt && r == cc {
            get_xxt(a, b, cc, n);
            return;
        }
    }
    let (ta, lda) = if bt { (b'T', n) } else { (b'N', r) };
    let (tb, ldb) = if ct { (b'T', cc) } else { (b'N', n) };
    dgemm(ta, tb, r, cc, n, 1.0, b, lda, c, ldb, 0.0, a, r);
}

/// mgcv's `mgcv_svd_full`: `dgesdd` with `JOBZ = 'O'` on the `r`-by-`c`
/// `x` (`r >= c`): `U` over `x`, `V'` in `vt`, the singular values in `d`.
pub(crate) unsafe fn mgcv_svd_full(x: *mut f64, vt: *mut f64, d: *mut f64, r: i32, c: i32) {
    let lwork = dgesdd_lwork(b'O', r, c).max(1);
    let mut work = vec![0.0f64; lwork as usize];
    let mut iwork = vec![0i32; 8 * (c.max(1) as usize)];
    let mut udum = [0.0f64; 1];
    dgesdd(
        b'O',
        r,
        c,
        x,
        r,
        d,
        udum.as_mut_ptr(),
        r,
        vt,
        c,
        work.as_mut_ptr(),
        lwork,
        iwork.as_mut_ptr(),
    );
}

/// mgcv's `mgcv_qr`: `dgeqp3` on the `r`-by-`c` `x` with the workspace its
/// query returns, the pivots then made 0-based. `pivot` goes in as `JPVT`:
/// an entry non-zero on entry fixes that column (gdiPK's second call passes
/// the first call's pivots).
pub(crate) unsafe fn mgcv_qr(x: *mut f64, r: i32, c: i32, pivot: *mut i32, tau: *mut f64) {
    let lwork = dgeqp3_lwork(r, c);
    let mut work = vec![0.0f64; lwork.max(1) as usize];
    dgeqp3(r, c, x, r, pivot, tau, work.as_mut_ptr(), lwork);
    for i in 0..c as usize {
        *pivot.add(i) -= 1;
    }
}

/// mgcv's `mgcv_qrqy` from the left: `b` (`r`-by-`c`) times the `Q` of the
/// `k` reflectors in `a` and `tau` (`tp`: its transpose), by `dormqr` with
/// the workspace its query returns.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mgcv_qrqy_left(
    b: *mut f64,
    a: *const f64,
    tau: *const f64,
    r: i32,
    c: i32,
    k: i32,
    tp: bool,
) {
    let lwork = dormqr_l_lwork(c);
    let mut work = vec![0.0f64; lwork.max(1) as usize];
    let trans = if tp { b'T' } else { b'N' };
    dormqr_l(trans, r, c, k, a, r, tau, b, r, work.as_mut_ptr(), lwork);
}

/// mgcv's `mgcv_pqrqy` with one thread: `b`, physically `r`-by-`cb`, holds
/// a `c`-by-`cb` matrix on entry when `tp` is false (spread to `r` rows
/// first) and on exit when it is true (packed back); the `Q` of `mgcv_pqr`
/// applied from the left.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mgcv_pqrqy(
    b: *mut f64,
    a: *const f64,
    tau: *const f64,
    r: i32,
    c: i32,
    cb: i32,
    tp: bool,
) {
    let (ru, cu, cbu) = (r as usize, c as usize, cb as usize);
    if !tp {
        // work down the columns from the end, moving each into place and
        // clearing what it leaves
        let mut x0 = (ru * cbu) as isize - 1;
        let mut x1 = (cu * cbu) as isize - 1;
        for _ in 0..cbu {
            x0 -= (ru - cu) as isize;
            for _ in 0..cu {
                *b.offset(x0) = *b.offset(x1);
                if x0 != x1 {
                    *b.offset(x1) = 0.0;
                }
                x0 -= 1;
                x1 -= 1;
            }
        }
    }
    mgcv_qrqy_left(b, a, tau, r, cb, c, tp);
    if tp {
        let (mut x0, mut x1) = (0usize, 0usize);
        for _ in 0..cbu {
            for _ in 0..cu {
                *b.add(x0) = *b.add(x1);
                x0 += 1;
                x1 += 1;
            }
            x1 += ru - cu;
        }
    }
}

/// mgcv's `getRpqr`: the `c`-by-`c` upper triangle of the `r`-by-`c` QR in
/// `x` into the `rr`-row `rm`, zeros below the diagonal.
pub(crate) unsafe fn get_rpqr(rm: *mut f64, x: *const f64, r: i32, c: i32, rr: i32) {
    let (r, c, rr) = (r as usize, c as usize, rr as usize);
    let rows = c.min(rr);
    for i in 0..rows {
        for j in 0..c {
            *rm.add(i + rr * j) = if i > j { 0.0 } else { *x.add(i + r * j) };
        }
    }
}

/// mgcv's `mgcv_backsolve`: `C = R^{-1} B` (`right`: `C = B R^{-1}`) for
/// the `c`-by-`c` upper triangle of the `r`-row `R`, `B` `c`-by-`bc`
/// (`bc`-by-`c`), by `dtrsm`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mgcv_backsolve(
    rm: *const f64,
    r: i32,
    c: i32,
    b: *const f64,
    cm: *mut f64,
    bc: i32,
    right: bool,
) {
    let (side, m, n) = if right { (b'R', bc, c) } else { (b'L', c, bc) };
    std::ptr::copy_nonoverlapping(b, cm, (bc as usize) * (c as usize));
    dtrsm(side, b'U', b'N', b'N', m, n, 1.0, rm, r, cm, m);
}

/// mgcv's `mgcv_forwardsolve`: `C = R^{-T} B` (`right`: `C = B R^{-T}`),
/// as [`mgcv_backsolve`] with `R` transposed.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mgcv_forwardsolve(
    rm: *const f64,
    r: i32,
    c: i32,
    b: *const f64,
    cm: *mut f64,
    bc: i32,
    right: bool,
) {
    let (side, m, n) = if right { (b'R', bc, c) } else { (b'L', c, bc) };
    std::ptr::copy_nonoverlapping(b, cm, (bc as usize) * (c as usize));
    dtrsm(side, b'U', b'T', b'N', m, n, 1.0, rm, r, cm, m);
}

/// mgcv's `getXtWX`: `X'WX` for the `r`-by-`c` `x` and `W = diag(w)`, a
/// column at a time (`dgemv('T')` of `X[, 1:i]` on `w * X[, i]`), the upper
/// triangle copied to the lower.
pub(crate) unsafe fn get_xtwx(xtwx: *mut f64, x: *const f64, w: *const f64, r: i32, c: i32) {
    let (ru, cu) = (r as usize, c as usize);
    let mut work = vec![0.0f64; ru.max(1)];
    let w2 = xtwx;
    let mut xx = 0.0;
    for i in 0..cu {
        for k in 0..ru {
            work[k] = *x.add(k + ru * i) * *w.add(k);
        }
        let j = (i + 1) as i32;
        dgemv(b'T', r, j, 1.0, x, r, work.as_ptr(), 1, 0.0, w2, 1);
        if i == 0 {
            xx = *w2;
        } else {
            for j in 0..=i {
                *xtwx.add(i * cu + j) = *w2.add(j);
            }
        }
    }
    if ru * cu > 0 {
        *xtwx = xx;
    }
    for i in 0..cu {
        for j in 0..i {
            *xtwx.add(j * cu + i) = *xtwx.add(i * cu + j);
        }
    }
}

/// mgcv's `getXtMX`: `X'MX` for the `r`-by-`c` `x` and symmetric `r`-by-`r`
/// `m`: `M X[, i]` a fused column sweep, each entry an 8-wide dot.
pub(crate) unsafe fn get_xtmx(xtmx: *mut f64, x: *const f64, m: *const f64, r: i32, c: i32) {
    let (ru, cu) = (r as usize, c as usize);
    let mut work = vec![0.0f64; ru];
    for i in 0..cu {
        let xi = x.add(ru * i);
        let x0 = *xi;
        for k in 0..ru {
            work[k] = x0 * *m.add(k);
        }
        for j in 1..ru {
            let xj = *xi.add(j);
            let mj = m.add(ru * j);
            for k in 0..ru {
                work[k] = fma(xj, *mj.add(k), work[k]);
            }
        }
        for j in 0..=i {
            let xx = dot_s8(&work, std::slice::from_raw_parts(x.add(ru * j), ru));
            *xtmx.add(i * cu + j) = xx;
            *xtmx.add(j * cu + i) = xx;
        }
    }
}

/// gdi.c's `pivoter`: pivot (`reverse`: unpivot) the rows, or with `col`
/// the columns, of the `r`-by-`c` `x` by the 0-based `pivot`; pivoted
/// element `i` is unpivoted element `pivot[i]`.
pub(crate) unsafe fn pivoter(
    x: *mut f64,
    r: i32,
    c: i32,
    pivot: *const i32,
    col: bool,
    reverse: bool,
) {
    let (ru, cu) = (r as usize, c as usize);
    if col {
        let mut dum = vec![0.0f64; cu];
        for i in 0..ru {
            if reverse {
                for j in 0..cu {
                    dum[*pivot.add(j) as usize] = *x.add(i + ru * j);
                }
            } else {
                for j in 0..cu {
                    dum[j] = *x.add(i + ru * (*pivot.add(j) as usize));
                }
            }
            for j in 0..cu {
                *x.add(i + ru * j) = dum[j];
            }
        }
    } else {
        let mut dum = vec![0.0f64; ru];
        for j in 0..cu {
            let p = x.add(ru * j);
            if reverse {
                for i in 0..ru {
                    dum[*pivot.add(i) as usize] = *p.add(i);
                }
            } else {
                for i in 0..ru {
                    dum[i] = *p.add(*pivot.add(i) as usize);
                }
            }
            for i in 0..ru {
                *p.add(i) = dum[i];
            }
        }
    }
}

/// gdi.c's `drop_cols`: remove the columns `drop` (ascending) of the
/// `r`-by-`c` `x`, packing the rest to the front.
pub(crate) unsafe fn drop_cols(x: *mut f64, r: i32, c: i32, drop: &[i32]) {
    let n_drop = drop.len();
    let r = r as usize;
    for k in 0..n_drop {
        let j = drop[k] as usize - k;
        let j0 = drop[k] as usize + 1;
        let j1 = if k < n_drop - 1 {
            drop[k + 1] as usize
        } else {
            c as usize
        };
        let (mut p, mut p1, p2) = (j * r, j0 * r, j1 * r);
        while p1 < p2 {
            *x.add(p) = *x.add(p1);
            p += 1;
            p1 += 1;
        }
    }
}

/// gdi.c's `drop_rows`: remove the rows `drop` (ascending) of the
/// `r`-by-`c` `x`, packing it as `(r - n_drop)`-by-`c`.
pub(crate) unsafe fn drop_rows(x: *mut f64, r: i32, c: i32, drop: &[i32]) {
    let n_drop = drop.len();
    if n_drop == 0 {
        return;
    }
    let (r, c) = (r as usize, c as usize);
    let (mut xo, mut xs) = (0usize, 0usize);
    for _ in 0..c {
        for _ in 0..drop[0] as usize {
            *x.add(xo) = *x.add(xs);
            xo += 1;
            xs += 1;
        }
        xs += 1;
        for k in 1..n_drop {
            for _ in (drop[k - 1] as usize + 1)..drop[k] as usize {
                *x.add(xo) = *x.add(xs);
                xo += 1;
                xs += 1;
            }
            xs += 1;
        }
        for _ in (drop[n_drop - 1] as usize + 1)..r {
            *x.add(xo) = *x.add(xs);
            xo += 1;
            xs += 1;
        }
    }
}

/// gdi.c's `undrop_rows`: spread the `(r - n_drop)`-by-`c` `x` back to
/// `r`-by-`c` with zero rows at `drop` (ascending).
pub(crate) unsafe fn undrop_rows(x: *mut f64, r: i32, c: i32, drop: &[i32]) {
    let n_drop = drop.len();
    if n_drop == 0 {
        return;
    }
    let (r, c) = (r as isize, c as isize);
    let mut xs = (r - n_drop as isize) * c - 1;
    let mut xo = r * c - 1;
    for _ in 0..c {
        let mut i = r - 1;
        while i > drop[n_drop - 1] as isize {
            *x.offset(xo) = *x.offset(xs);
            xo -= 1;
            xs -= 1;
            i -= 1;
        }
        *x.offset(xo) = 0.0;
        xo -= 1;
        for k in (1..n_drop).rev() {
            let mut i = drop[k] as isize - 1;
            while i > drop[k - 1] as isize {
                *x.offset(xo) = *x.offset(xs);
                xo -= 1;
                xs -= 1;
                i -= 1;
            }
            *x.offset(xo) = 0.0;
            xo -= 1;
        }
        let mut i = drop[0] as isize - 1;
        while i >= 0 {
            *x.offset(xo) = *x.offset(xs);
            xo -= 1;
            xs -= 1;
            i -= 1;
        }
    }
}

/// gdi.c's `Rinv`: the inverse of the `c`-by-`c` upper triangle of the
/// `r`-row `rm` into the top of the `ri`-row `out`, by columns of back
/// substitution. The inner sum runs along a row of `rm`, strided, so
/// clang keeps it scalar and fused.
pub(crate) unsafe fn rinv(out: *mut f64, rm: *const f64, c: i32, r: i32, ri: i32) {
    let (c, r, ri) = (c as usize, r as usize, ri as usize);
    for i in 0..c {
        let rc = out.add(ri * i);
        let mut eye = 1.0;
        for k in (0..=i).rev() {
            let mut xx = 0.0;
            for j in (k + 1)..=i {
                xx = fma(*rm.add(k + j * r), *rc.add(j), xx);
            }
            *rc.add(k) = (eye - xx) / *rm.add(k + k * r);
            eye = 0.0;
        }
        for k in (i + 1)..c {
            *rc.add(k) = 0.0;
        }
    }
}

/// gdi.c's `rc_prod`: `y = diag(z) x` for the `n`-row `x` of `xcol`
/// columns.
pub(crate) unsafe fn rc_prod(y: *mut f64, z: *const f64, x: *const f64, xcol: i32, n: i32) {
    let n = n as usize;
    for j in 0..xcol as usize {
        for i in 0..n {
            *y.add(i + n * j) = *z.add(i) * *x.add(i + n * j);
        }
    }
}

/// gdi.c's `frobenius_norm`: `sqrt(sum x^2)` over the `r`-by-`c` `x`.
pub(crate) unsafe fn frobenius_norm(x: *const f64, r: i32, c: i32) -> f64 {
    let n = (r as usize) * (c as usize);
    sumsq_s4(std::slice::from_raw_parts(x, n)).sqrt()
}

/// gdi.c's `diagABt`: `d = diag(A B')` for `r`-by-`c` `A` and `B` (the
/// first column's products rounded, the rest fused in), returning its sum.
pub(crate) unsafe fn diag_abt(d: *mut f64, a: *const f64, b: *const f64, r: i32, c: i32) -> f64 {
    let (r, c) = (r as usize, c as usize);
    let mut tr = 0.0;
    if c > 0 {
        for i in 0..r {
            *d.add(i) = *a.add(i) * *b.add(i);
        }
        for j in 1..c {
            for i in 0..r {
                *d.add(i) = fma(*a.add(i + r * j), *b.add(i + r * j), *d.add(i));
            }
        }
        for i in 0..r {
            tr += *d.add(i);
        }
    }
    tr
}

/// gdi.c's `R_cond`: the Cline, Moler, Stewart and Wilkinson estimate of
/// the condition number of the `c`-by-`c` upper triangle of the `r`-row
/// `rm`.
pub(crate) unsafe fn r_cond(rm: *const f64, r: i32, c: i32) -> f64 {
    let (r, c) = (r as usize, c as usize);
    let mut pp = vec![0.0f64; c];
    let mut pm = vec![0.0f64; c];
    let mut y = vec![0.0f64; c];
    let mut p = vec![0.0f64; c];
    let (mut y_inf, mut r_inf) = (0.0f64, 0.0f64);
    for k in (0..c).rev() {
        let rkk = *rm.add(k + r * k);
        let yp = (1.0 - p[k]) / rkk;
        let ym = (-1.0 - p[k]) / rkk;
        let mut pp_norm = 0.0;
        for i in 0..k {
            pp[i] = fma(*rm.add(i + r * k), yp, p[i]);
            pp_norm += pp[i].abs();
        }
        let mut pm_norm = 0.0;
        for i in 0..k {
            pm[i] = fma(*rm.add(i + r * k), ym, p[i]);
            pm_norm += pm[i].abs();
        }
        if yp.abs() + pp_norm >= ym.abs() + pm_norm {
            y[k] = yp;
            p[..k].copy_from_slice(&pp[..k]);
        } else {
            y[k] = ym;
            p[..k].copy_from_slice(&pm[..k]);
        }
        let kappa = y[k].abs();
        if kappa > y_inf {
            y_inf = kappa;
        }
    }
    for i in 0..c {
        let mut kappa = 0.0;
        for j in i..c {
            kappa += (*rm.add(i + r * j)).abs();
        }
        if kappa > r_inf {
            r_inf = kappa;
        }
    }
    r_inf * y_inf
}
