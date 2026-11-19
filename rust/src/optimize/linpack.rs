use crate::nmath::util::rfma;

const ACCEL_PAIR4: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

pub fn ddot(
    n: usize,
    dx: &[f64],
    ox: usize,
    incx: usize,
    dy: &[f64],
    oy: usize,
    incy: usize,
) -> f64 {
    if ACCEL_PAIR4 && n == 4 {
        let s0 = dx[ox] * dy[oy];
        let s1 = dx[ox + incx] * dy[oy + incy];
        let s2 = dx[ox + 2 * incx] * dy[oy + 2 * incy];
        let s3 = dx[ox + 3 * incx] * dy[oy + 3 * incy];
        return (s0 + s2) + (s1 + s3);
    }
    // from five terms on Accelerate's ddot rounds by its own reduction and
    // by where its second operand starts, so the call is Accelerate's own
    #[cfg(accelerate)]
    if n >= 5 && incx == 1 && incy == 1 {
        return crate::sparse::blas::dot(&dx[ox..ox + n], &dy[oy..oy + n]);
    }
    let mut dtemp = 0.0;
    for i in 0..n {
        dtemp += dx[ox + i * incx] * dy[oy + i * incy];
    }
    dtemp
}

pub fn dnrm2(n: usize, x: &[f64], ox: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    ddot(n, x, ox, 1, x, ox, 1).sqrt()
}

pub fn dscal(n: usize, da: f64, dx: &mut [f64], ox: usize) {
    for i in 0..n {
        dx[ox + i] = da * dx[ox + i];
    }
}

pub fn daxpy(n: usize, da: f64, dx: &[f64], ox: usize, incx: usize, dy: &mut [f64], oy: usize) {
    if n == 0 || da == 0.0 {
        return;
    }
    for i in 0..n {
        dy[oy + i] = rfma(da, dx[ox + i * incx], dy[oy + i]);
    }
}

pub fn daxpy_same(n: usize, da: f64, wa: &mut [f64], ox: usize, oy: usize) {
    if n == 0 || da == 0.0 {
        return;
    }
    for i in 0..n {
        wa[oy + i] = rfma(da, wa[ox + i], wa[oy + i]);
    }
}

pub fn dcopy(n: usize, dx: &[f64], ox: usize, dy: &mut [f64], oy: usize) {
    for i in 0..n {
        dy[oy + i] = dx[ox + i];
    }
}

pub fn dcopy_same(n: usize, buf: &mut [f64], ox: usize, oy: usize) {
    for i in 0..n {
        buf[oy + i] = buf[ox + i];
    }
}

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
            b[bi(n - 1)] /= t[t0 + (n - 1) + (n - 1) * lda];
            for jj in 2..=n {
                let j = n - jj + 1;
                let dt = ddot(jj - 1, t, t0 + j + (j - 1) * lda, 1, b, bi(j), incb);
                b[bi(j - 1)] -= dt;
                b[bi(j - 1)] /= t[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        _ => {
            b[bi(0)] /= t[t0];
            for j in 2..=n {
                let dt = ddot(j - 1, t, t0 + (j - 1) * lda, 1, b, bi(0), incb);
                b[bi(j - 1)] -= dt;
                b[bi(j - 1)] /= t[t0 + (j - 1) + (j - 1) * lda];
            }
        }
    }
    0
}

pub fn ddot_same(n: usize, buf: &[f64], ox: usize, oy: usize) -> f64 {
    ddot(n, buf, ox, 1, buf, oy, 1)
}

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
                        buf[ob + j - 1 + i] = rfma(
                            temp,
                            buf[t0 + (j - 1 + i) + (j - 2) * lda],
                            buf[ob + j - 1 + i],
                        );
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
                let dt = ddot(jj - 1, buf, t0 + j + (j - 1) * lda, 1, buf, ob + j, 1);
                buf[ob + j - 1] -= dt;
                buf[ob + j - 1] /= buf[t0 + (j - 1) + (j - 1) * lda];
            }
        }
        _ => {
            buf[ob] /= buf[t0];
            for j in 2..=n {
                let dt = ddot(j - 1, buf, t0 + (j - 1) * lda, 1, buf, ob, 1);
                buf[ob + j - 1] -= dt;
                buf[ob + j - 1] /= buf[t0 + (j - 1) + (j - 1) * lda];
            }
        }
    }
    0
}

pub fn dpofa(a: &mut [f64], a0: usize, lda: usize, n: usize) -> i32 {
    let eps = 1e-14;
    for j in 1..=n {
        let mut s = 0.0;
        for k in 1..=(j - 1) {
            let dt = ddot(k - 1, a, a0 + (k - 1) * lda, 1, a, a0 + (j - 1) * lda, 1);
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
