//! BLAS level-1 / LINPACK kernels for the R-optimizer ports, mirroring
//! `hea/R/_linpack.py` (the spec and test oracle) line by line — which in
//! turn emulates the BLAS R actually links per platform (Accelerate on
//! macOS/arm64, plain-ordered reference BLAS elsewhere; see the Python
//! module docstring for the probed n≤4 exactness boundary) and R's
//! gfortran-compiled `dtrsl.f`/`dpofa.f`.
//!
//! Matrices are flat column-major with an explicit leading dimension
//! (`a[i + j*lda]`), exactly like the f2c sources; the Python spec's
//! `[row, col]` numpy indexing is the same layout viewed 2-D.

use crate::nmath::util::rfma;

/// R links Accelerate only on macOS, and its n=4 ddot pair tree was
/// probed on arm64; on reference-BLAS platforms ddot is sequential at
/// every n. Mirrors `_linpack._ACCEL_PAIR4`.
const ACCEL_PAIR4: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// `_ddot`: sequential everywhere, except the probed Accelerate pair
/// tree at n = 4 on darwin/arm64.
pub fn ddot(n: usize, dx: &[f64], ox: usize, incx: usize, dy: &[f64], oy: usize, incy: usize) -> f64 {
    if ACCEL_PAIR4 && n == 4 {
        let s0 = dx[ox] * dy[oy];
        let s1 = dx[ox + incx] * dy[oy + incy];
        let s2 = dx[ox + 2 * incx] * dy[oy + 2 * incy];
        let s3 = dx[ox + 3 * incx] * dy[oy + 3 * incy];
        return (s0 + s2) + (s1 + s3);
    }
    let mut dtemp = 0.0;
    for i in 0..n {
        dtemp += dx[ox + i * incx] * dy[oy + i * incy];
    }
    dtemp
}

/// `_dnrm2`: sqrt of the `ddot` self-product (Accelerate n ≤ 2 exact).
pub fn dnrm2(n: usize, x: &[f64], ox: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    ddot(n, x, ox, 1, x, ox, 1).sqrt()
}

/// `_dscal`: elementwise scale in place.
pub fn dscal(n: usize, da: f64, dx: &mut [f64], ox: usize) {
    for i in 0..n {
        dx[ox + i] = da * dx[ox + i];
    }
}

/// `_daxpy`: Accelerate fuses per element (`rfma` keeps the per-arch
/// R-parity policy).
pub fn daxpy(n: usize, da: f64, dx: &[f64], ox: usize, incx: usize, dy: &mut [f64], oy: usize) {
    if n == 0 || da == 0.0 {
        return;
    }
    for i in 0..n {
        dy[oy + i] = rfma(da, dx[ox + i * incx], dy[oy + i]);
    }
}

/// `daxpy` where both vectors live in the same buffer (cauchy's `wa`
/// blocks). `split_at_mut` can't express overlapping-free aliasing at
/// distinct offsets generically, so index arithmetic does it.
pub fn daxpy_same(n: usize, da: f64, wa: &mut [f64], ox: usize, oy: usize) {
    if n == 0 || da == 0.0 {
        return;
    }
    for i in 0..n {
        wa[oy + i] = rfma(da, wa[ox + i], wa[oy + i]);
    }
}

/// `_dcopy`: forward copy.
pub fn dcopy(n: usize, dx: &[f64], ox: usize, dy: &mut [f64], oy: usize) {
    for i in 0..n {
        dy[oy + i] = dx[ox + i];
    }
}

/// `dcopy` within one buffer (matupd/formk shifts) — forward loop like
/// the Fortran.
pub fn dcopy_same(n: usize, buf: &mut [f64], ox: usize, oy: usize) {
    for i in 0..n {
        buf[oy + i] = buf[ox + i];
    }
}

/// LINPACK `dtrsl` on the n×n leading block of the column-major matrix
/// `t` (leading dimension `lda`, starting at flat offset `t0`), solving
/// into `b[ob..]` (stride `incb` — column solves inside `formk` pass a
/// matrix column). job: 0 lower / 1 upper / 10 lower-T / 11 upper-T.
pub fn dtrsl(
    t: &[f64],
    t0: usize,
    lda: usize,
    n: usize,
    b: &mut [f64],
    ob: usize,
    incb: usize,
    job: i32,
) -> i32 {
    for info in 1..=n {
        if t[t0 + (info - 1) + (info - 1) * lda] == 0.0 {
            return info as i32;
        }
    }
    let mut kase = 1;
    if job % 10 != 0 {
        kase = 2;
    }
    if (job % 100) / 10 != 0 {
        kase += 2;
    }
    let bi = |i: usize| ob + i * incb;
    match kase {
        1 => {
            // solve t*x=b, t lower triangular (daxpy sweeps)
            b[bi(0)] /= t[t0];
            for j in 2..=n {
                let temp = -b[bi(j - 2)];
                if temp != 0.0 {
                    for i in 0..(n - j + 1) {
                        b[bi(j - 1 + i)] =
                            rfma(temp, t[t0 + (j - 1 + i) + (j - 2) * lda], b[bi(j - 1 + i)]);
                    }
                }
                b[bi(j - 1)] /= t[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        2 => {
            // solve t*x=b, t upper triangular
            b[bi(n - 1)] /= t[t0 + (n - 1) + (n - 1) * lda];
            for jj in 2..=n {
                let j = n - jj + 1;
                let temp = -b[bi(j)];
                if temp != 0.0 {
                    for i in 0..j {
                        b[bi(i)] = rfma(temp, t[t0 + i + j * lda], b[bi(i)]);
                    }
                }
                b[bi(j - 1)] /= t[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        3 => {
            // solve trans(t)*x=b, t lower triangular (ddot form)
            b[bi(n - 1)] /= t[t0 + (n - 1) + (n - 1) * lda];
            for jj in 2..=n {
                let j = n - jj + 1;
                let mut dt = 0.0;
                if ACCEL_PAIR4 && jj - 1 == 4 {
                    let s0 = t[t0 + j + (j - 1) * lda] * b[bi(j)];
                    let s1 = t[t0 + j + 1 + (j - 1) * lda] * b[bi(j + 1)];
                    let s2 = t[t0 + j + 2 + (j - 1) * lda] * b[bi(j + 2)];
                    let s3 = t[t0 + j + 3 + (j - 1) * lda] * b[bi(j + 3)];
                    dt = (s0 + s2) + (s1 + s3);
                } else {
                    for i in 0..(jj - 1) {
                        dt += t[t0 + (j + i) + (j - 1) * lda] * b[bi(j + i)];
                    }
                }
                b[bi(j - 1)] -= dt;
                b[bi(j - 1)] /= t[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        _ => {
            // solve trans(t)*x=b, t upper triangular
            b[bi(0)] /= t[t0];
            for j in 2..=n {
                let mut dt = 0.0;
                if ACCEL_PAIR4 && j - 1 == 4 {
                    let s0 = t[t0 + (j - 1) * lda] * b[bi(0)];
                    let s1 = t[t0 + 1 + (j - 1) * lda] * b[bi(1)];
                    let s2 = t[t0 + 2 + (j - 1) * lda] * b[bi(2)];
                    let s3 = t[t0 + 3 + (j - 1) * lda] * b[bi(3)];
                    dt = (s0 + s2) + (s1 + s3);
                } else {
                    for i in 0..(j - 1) {
                        dt += t[t0 + i + (j - 1) * lda] * b[bi(i)];
                    }
                }
                b[bi(j - 1)] -= dt;
                b[bi(j - 1)] /= t[t0 + (j - 1) + (j - 1) * lda];
            }
        }
    }
    0
}

/// `ddot` where both vectors live in one buffer (formk's wn columns).
pub fn ddot_same(n: usize, buf: &[f64], ox: usize, oy: usize) -> f64 {
    if ACCEL_PAIR4 && n == 4 {
        let s0 = buf[ox] * buf[oy];
        let s1 = buf[ox + 1] * buf[oy + 1];
        let s2 = buf[ox + 2] * buf[oy + 2];
        let s3 = buf[ox + 3] * buf[oy + 3];
        return (s0 + s2) + (s1 + s3);
    }
    let mut dtemp = 0.0;
    for i in 0..n {
        dtemp += buf[ox + i] * buf[oy + i];
    }
    dtemp
}

/// `dtrsl` where the right-hand side is a column of the SAME buffer as
/// the triangular matrix (formk solves into wn's own columns). One
/// mutable borrow; index arithmetic replicates the C aliasing exactly
/// (the solved column is disjoint from the referenced triangle).
pub fn dtrsl_same(buf: &mut [f64], t0: usize, lda: usize, n: usize, ob: usize, job: i32) -> i32 {
    for info in 1..=n {
        if buf[t0 + (info - 1) + (info - 1) * lda] == 0.0 {
            return info as i32;
        }
    }
    let mut kase = 1;
    if job % 10 != 0 {
        kase = 2;
    }
    if (job % 100) / 10 != 0 {
        kase += 2;
    }
    match kase {
        1 => {
            buf[ob] /= buf[t0];
            for j in 2..=n {
                let temp = -buf[ob + j - 2];
                if temp != 0.0 {
                    for i in 0..(n - j + 1) {
                        buf[ob + j - 1 + i] =
                            rfma(temp, buf[t0 + (j - 1 + i) + (j - 2) * lda], buf[ob + j - 1 + i]);
                    }
                }
                buf[ob + j - 1] /= buf[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        2 => {
            buf[ob + n - 1] /= buf[t0 + (n - 1) + (n - 1) * lda];
            for jj in 2..=n {
                let j = n - jj + 1;
                let temp = -buf[ob + j];
                if temp != 0.0 {
                    for i in 0..j {
                        buf[ob + i] = rfma(temp, buf[t0 + i + j * lda], buf[ob + i]);
                    }
                }
                buf[ob + j - 1] /= buf[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        3 => {
            buf[ob + n - 1] /= buf[t0 + (n - 1) + (n - 1) * lda];
            for jj in 2..=n {
                let j = n - jj + 1;
                let mut dt = 0.0;
                if ACCEL_PAIR4 && jj - 1 == 4 {
                    let s0 = buf[t0 + j + (j - 1) * lda] * buf[ob + j];
                    let s1 = buf[t0 + j + 1 + (j - 1) * lda] * buf[ob + j + 1];
                    let s2 = buf[t0 + j + 2 + (j - 1) * lda] * buf[ob + j + 2];
                    let s3 = buf[t0 + j + 3 + (j - 1) * lda] * buf[ob + j + 3];
                    dt = (s0 + s2) + (s1 + s3);
                } else {
                    for i in 0..(jj - 1) {
                        dt += buf[t0 + (j + i) + (j - 1) * lda] * buf[ob + j + i];
                    }
                }
                buf[ob + j - 1] -= dt;
                buf[ob + j - 1] /= buf[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        _ => {
            buf[ob] /= buf[t0];
            for j in 2..=n {
                let mut dt = 0.0;
                if ACCEL_PAIR4 && j - 1 == 4 {
                    let s0 = buf[t0 + (j - 1) * lda] * buf[ob];
                    let s1 = buf[t0 + 1 + (j - 1) * lda] * buf[ob + 1];
                    let s2 = buf[t0 + 2 + (j - 1) * lda] * buf[ob + 2];
                    let s3 = buf[t0 + 3 + (j - 1) * lda] * buf[ob + 3];
                    dt = (s0 + s2) + (s1 + s3);
                } else {
                    for i in 0..(j - 1) {
                        dt += buf[t0 + i + (j - 1) * lda] * buf[ob + i];
                    }
                }
                buf[ob + j - 1] -= dt;
                buf[ob + j - 1] /= buf[t0 + (j - 1) + (j - 1) * lda];
            }
        }
    }
    0
}

/// LINPACK `dpofa` (R's 2002 tolerance) on the n×n leading block at
/// flat offset `a0` with leading dimension `lda`; upper triangle
/// overwritten with r (a = r'r). `s = s + t*t` is gfortran-contracted.
pub fn dpofa(a: &mut [f64], a0: usize, lda: usize, n: usize) -> i32 {
    let eps = 1e-14;
    for j in 1..=n {
        let mut s = 0.0;
        for k in 1..=(j - 1) {
            let mut dt = 0.0;
            if ACCEL_PAIR4 && k - 1 == 4 {
                let s0 = a[a0 + (k - 1) * lda] * a[a0 + (j - 1) * lda];
                let s1 = a[a0 + 1 + (k - 1) * lda] * a[a0 + 1 + (j - 1) * lda];
                let s2 = a[a0 + 2 + (k - 1) * lda] * a[a0 + 2 + (j - 1) * lda];
                let s3 = a[a0 + 3 + (k - 1) * lda] * a[a0 + 3 + (j - 1) * lda];
                dt = (s0 + s2) + (s1 + s3);
            } else {
                for i in 0..(k - 1) {
                    dt += a[a0 + i + (k - 1) * lda] * a[a0 + i + (j - 1) * lda];
                }
            }
            let mut t = a[a0 + (k - 1) + (j - 1) * lda] - dt;
            t /= a[a0 + (k - 1) + (k - 1) * lda];
            a[a0 + (k - 1) + (j - 1) * lda] = t;
            s = rfma(t, t, s);
        }
        let s = a[a0 + (j - 1) + (j - 1) * lda] - s;
        if s <= eps * a[a0 + (j - 1) + (j - 1) * lda].abs() {
            return j as i32;
        }
        a[a0 + (j - 1) + (j - 1) * lda] = s.sqrt();
    }
    0
}
