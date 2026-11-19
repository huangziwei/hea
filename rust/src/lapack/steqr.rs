//! `DSTEQR` (implicit QL/QR on a symmetric tridiagonal matrix) and `DSTERF`
//! (its root-free eigenvalue-only variant, Pal-Walker-Kahan).

use super::auxiliary::{dlae2, dlaev2, dlanst_m, dlapy2, dlartg, dlascl_g, dlaset, dlasr, dlasrt};
use super::blas::dswap;
use super::{fma, lsame, sign, EPS, SAFMIN};

const MAXIT: i32 = 30;

/// `DSTEQR`: eigenvalues of the tridiagonal `(d, e)` into `d` (ascending)
/// and, for `compz = 'V'` or `'I'`, the eigenvectors into `z`. `work` holds
/// `2*(n-1)` doubles when vectors are wanted. Returns `INFO`.
pub(crate) unsafe fn dsteqr(
    compz: u8,
    n: i32,
    d: *mut f64,
    e: *mut f64,
    z: *mut f64,
    ldz: i32,
    work: *mut f64,
) -> i32 {
    let mut info = 0;
    let icompz = if lsame(compz, b'N') {
        0
    } else if lsame(compz, b'V') {
        1
    } else if lsame(compz, b'I') {
        2
    } else {
        -1
    };
    if icompz < 0 {
        return -1;
    } else if n < 0 {
        return -2;
    } else if ldz < 1 || (icompz > 0 && ldz < n.max(1)) {
        return -6;
    }
    if n == 0 {
        return 0;
    }
    if n == 1 {
        if icompz == 2 {
            m!(z, ldz, 1, 1) = 1.0;
        }
        return 0;
    }

    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;

    if icompz == 2 {
        dlaset(b'F', n, n, 0.0, 1.0, z, ldz);
    }

    let nmaxit = n * MAXIT;
    let mut jtot = 0;

    let mut l1 = 1;
    let nm1 = n - 1;

    'outer: loop {
        // 10
        if l1 > n {
            break 'outer;
        }
        if l1 > 1 {
            v!(e, l1 - 1) = 0.0;
        }
        let mut m = n;
        if l1 <= nm1 {
            let mut mm = l1;
            while mm <= nm1 {
                let tst = v!(e, mm).abs();
                if tst == 0.0 {
                    m = mm;
                    break;
                }
                if tst <= (v!(d, mm).abs().sqrt() * v!(d, mm + 1).abs().sqrt()) * eps {
                    v!(e, mm) = 0.0;
                    m = mm;
                    break;
                }
                mm += 1;
            }
        }
        // 30
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue 'outer;
        }

        let anorm = dlanst_m(lend - l + 1, pv!(d, l), pv!(e, l));
        let mut iscale = 0;
        if anorm == 0.0 {
            continue 'outer;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl_g(anorm, ssfmax, lend - l + 1, 1, pv!(d, l), n);
            dlascl_g(anorm, ssfmax, lend - l, 1, pv!(e, l), n);
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl_g(anorm, ssfmin, lend - l + 1, 1, pv!(d, l), n);
            dlascl_g(anorm, ssfmin, lend - l, 1, pv!(e, l), n);
        }

        if v!(d, lend).abs() < v!(d, l).abs() {
            lend = lsv;
            l = lendsv;
        }

        if lend > l {
            // QL iteration: look for a small subdiagonal element.
            loop {
                // 40
                let mut m = lend;
                if l != lend {
                    let lendm1 = lend - 1;
                    let mut mm = l;
                    while mm <= lendm1 {
                        let tst = v!(e, mm).abs() * v!(e, mm).abs();
                        if tst <= fma(eps2 * v!(d, mm).abs(), v!(d, mm + 1).abs(), safmin) {
                            m = mm;
                            break;
                        }
                        mm += 1;
                    }
                }
                // 60
                if m < lend {
                    v!(e, m) = 0.0;
                }
                let mut p = v!(d, l);
                if m == l {
                    // 80: eigenvalue found.
                    v!(d, l) = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let (rt1, rt2);
                    if icompz > 0 {
                        let (r1, r2, c, s) = dlaev2(v!(d, l), v!(e, l), v!(d, l + 1));
                        rt1 = r1;
                        rt2 = r2;
                        v!(work, l) = c;
                        v!(work, n - 1 + l) = s;
                        dlasr(
                            b'R',
                            b'V',
                            b'B',
                            n,
                            2,
                            pv!(work, l),
                            pv!(work, n - 1 + l),
                            pm!(z, ldz, 1, l),
                            ldz,
                        );
                    } else {
                        let (r1, r2) = dlae2(v!(d, l), v!(e, l), v!(d, l + 1));
                        rt1 = r1;
                        rt2 = r2;
                    }
                    v!(d, l) = rt1;
                    v!(d, l + 1) = rt2;
                    v!(e, l) = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;

                // Form shift.
                let mut g = (v!(d, l + 1) - p) / (2.0 * v!(e, l));
                let mut r = dlapy2(g, 1.0);
                g = v!(d, m) - p + (v!(e, l) / (g + sign(r, g)));

                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;

                let mm1 = m - 1;
                let mut i = mm1;
                while i >= l {
                    let f = s * v!(e, i);
                    let b = c * v!(e, i);
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    r = rr;
                    if i != m - 1 {
                        v!(e, i + 1) = r;
                    }
                    g = v!(d, i + 1) - p;
                    r = fma(v!(d, i) - g, s, 2.0 * c * b);
                    p = s * r;
                    v!(d, i + 1) = g + p;
                    g = fma(c, r, -b);

                    if icompz > 0 {
                        v!(work, i) = c;
                        v!(work, n - 1 + i) = -s;
                    }
                    i -= 1;
                }

                if icompz > 0 {
                    let mm = m - l + 1;
                    dlasr(
                        b'R',
                        b'V',
                        b'B',
                        n,
                        mm,
                        pv!(work, l),
                        pv!(work, n - 1 + l),
                        pm!(z, ldz, 1, l),
                        ldz,
                    );
                }

                v!(d, l) -= p;
                v!(e, l) = g;
            }
        } else {
            // QR iteration: look for a small superdiagonal element.
            loop {
                // 90
                let mut m = lend;
                if l != lend {
                    let lendp1 = lend + 1;
                    let mut mm = l;
                    while mm >= lendp1 {
                        let tst = v!(e, mm - 1).abs() * v!(e, mm - 1).abs();
                        if tst <= fma(eps2 * v!(d, mm).abs(), v!(d, mm - 1).abs(), safmin) {
                            m = mm;
                            break;
                        }
                        mm -= 1;
                    }
                }
                // 110
                if m > lend {
                    v!(e, m - 1) = 0.0;
                }
                let mut p = v!(d, l);
                if m == l {
                    // 130: eigenvalue found.
                    v!(d, l) = p;
                    l -= 1;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if m == l - 1 {
                    let (rt1, rt2);
                    if icompz > 0 {
                        let (r1, r2, c, s) = dlaev2(v!(d, l - 1), v!(e, l - 1), v!(d, l));
                        rt1 = r1;
                        rt2 = r2;
                        v!(work, m) = c;
                        v!(work, n - 1 + m) = s;
                        dlasr(
                            b'R',
                            b'V',
                            b'F',
                            n,
                            2,
                            pv!(work, m),
                            pv!(work, n - 1 + m),
                            pm!(z, ldz, 1, l - 1),
                            ldz,
                        );
                    } else {
                        let (r1, r2) = dlae2(v!(d, l - 1), v!(e, l - 1), v!(d, l));
                        rt1 = r1;
                        rt2 = r2;
                    }
                    v!(d, l - 1) = rt1;
                    v!(d, l) = rt2;
                    v!(e, l - 1) = 0.0;
                    l -= 2;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;

                // Form shift.
                let mut g = (v!(d, l - 1) - p) / (2.0 * v!(e, l - 1));
                let mut r = dlapy2(g, 1.0);
                g = v!(d, m) - p + (v!(e, l - 1) / (g + sign(r, g)));

                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;

                let lm1 = l - 1;
                for i in m..=lm1 {
                    let f = s * v!(e, i);
                    let b = c * v!(e, i);
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    r = rr;
                    if i != m {
                        v!(e, i - 1) = r;
                    }
                    g = v!(d, i) - p;
                    r = fma(v!(d, i + 1) - g, s, 2.0 * c * b);
                    p = s * r;
                    v!(d, i) = g + p;
                    g = fma(c, r, -b);

                    if icompz > 0 {
                        v!(work, i) = c;
                        v!(work, n - 1 + i) = s;
                    }
                }

                if icompz > 0 {
                    let mm = l - m + 1;
                    dlasr(
                        b'R',
                        b'V',
                        b'F',
                        n,
                        mm,
                        pv!(work, m),
                        pv!(work, n - 1 + m),
                        pm!(z, ldz, 1, m),
                        ldz,
                    );
                }

                v!(d, l) -= p;
                v!(e, lm1) = g;
            }
        }

        // 140: undo scaling if necessary.
        if iscale == 1 {
            dlascl_g(ssfmax, anorm, lendsv - lsv + 1, 1, pv!(d, lsv), n);
            dlascl_g(ssfmax, anorm, lendsv - lsv, 1, pv!(e, lsv), n);
        } else if iscale == 2 {
            dlascl_g(ssfmin, anorm, lendsv - lsv + 1, 1, pv!(d, lsv), n);
            dlascl_g(ssfmin, anorm, lendsv - lsv, 1, pv!(e, lsv), n);
        }

        // Check for no convergence to an eigenvalue after a total of
        // n*MAXIT iterations.
        if jtot < nmaxit {
            continue 'outer;
        }
        for i in 1..=(n - 1) {
            if v!(e, i) != 0.0 {
                info += 1;
            }
        }
        return info;
    }

    // 160: order eigenvalues and eigenvectors.
    if icompz == 0 {
        dlasrt(b'I', n, d);
    } else {
        for ii in 2..=n {
            let i = ii - 1;
            let mut k = i;
            let mut p = v!(d, i);
            for j in ii..=n {
                if v!(d, j) < p {
                    k = j;
                    p = v!(d, j);
                }
            }
            if k != i {
                v!(d, k) = v!(d, i);
                v!(d, i) = p;
                dswap(n, pm!(z, ldz, 1, i), 1, pm!(z, ldz, 1, k), 1);
            }
        }
    }
    info
}

/// `DSTERF`: all eigenvalues of the tridiagonal `(d, e)` into `d`
/// (ascending), by the root-free QL/QR variant. Returns `INFO`.
pub(crate) unsafe fn dsterf(n: i32, d: *mut f64, e: *mut f64) -> i32 {
    let mut info = 0;
    if n < 0 {
        return -1;
    }
    if n <= 1 {
        return 0;
    }

    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;

    let nmaxit = n * MAXIT;
    let mut sigma;
    let mut jtot = 0;

    let mut l1 = 1;

    'outer: loop {
        // 10
        if l1 > n {
            break 'outer;
        }
        if l1 > 1 {
            v!(e, l1 - 1) = 0.0;
        }
        let mut m = n;
        let mut mm = l1;
        while mm <= n - 1 {
            if v!(e, mm).abs() <= (v!(d, mm).abs().sqrt() * v!(d, mm + 1).abs().sqrt()) * eps {
                v!(e, mm) = 0.0;
                m = mm;
                break;
            }
            mm += 1;
        }
        // 30
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue 'outer;
        }

        let anorm = dlanst_m(lend - l + 1, pv!(d, l), pv!(e, l));
        let mut iscale = 0;
        if anorm == 0.0 {
            continue 'outer;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl_g(anorm, ssfmax, lend - l + 1, 1, pv!(d, l), n);
            dlascl_g(anorm, ssfmax, lend - l, 1, pv!(e, l), n);
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl_g(anorm, ssfmin, lend - l + 1, 1, pv!(d, l), n);
            dlascl_g(anorm, ssfmin, lend - l, 1, pv!(e, l), n);
        }

        for i in l..=(lend - 1) {
            v!(e, i) = v!(e, i) * v!(e, i);
        }

        if v!(d, lend).abs() < v!(d, l).abs() {
            lend = lsv;
            l = lendsv;
        }

        if lend >= l {
            // QL iteration.
            loop {
                // 50
                let mut m = lend;
                if l != lend {
                    let mut mm = l;
                    while mm <= lend - 1 {
                        if v!(e, mm).abs() <= eps2 * (v!(d, mm) * v!(d, mm + 1)).abs() {
                            m = mm;
                            break;
                        }
                        mm += 1;
                    }
                }
                // 70
                if m < lend {
                    v!(e, m) = 0.0;
                }
                let mut p = v!(d, l);
                if m == l {
                    // 90
                    v!(d, l) = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let rte = v!(e, l).sqrt();
                    let (rt1, rt2) = dlae2(v!(d, l), rte, v!(d, l + 1));
                    v!(d, l) = rt1;
                    v!(d, l + 1) = rt2;
                    v!(e, l) = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;

                // Form shift.
                let rte = v!(e, l).sqrt();
                sigma = (v!(d, l + 1) - p) / (2.0 * rte);
                let r = dlapy2(sigma, 1.0);
                sigma = p - (rte / (sigma + sign(r, sigma)));

                let mut c = 1.0;
                let mut s = 0.0;
                let mut gamma = v!(d, m) - sigma;
                p = gamma * gamma;

                let mut i = m - 1;
                while i >= l {
                    let bb = v!(e, i);
                    let r = p + bb;
                    if i != m - 1 {
                        v!(e, i + 1) = s * r;
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = v!(d, i);
                    gamma = fma(c, alpha - sigma, -(s * oldgam));
                    v!(d, i + 1) = oldgam + (alpha - gamma);
                    if c != 0.0 {
                        p = (gamma * gamma) / c;
                    } else {
                        p = oldc * bb;
                    }
                    i -= 1;
                }

                v!(e, l) = s * p;
                v!(d, l) = sigma + gamma;
            }
        } else {
            // QR iteration.
            loop {
                // 100
                let mut m = lend;
                let mut mm = l;
                while mm >= lend + 1 {
                    if v!(e, mm - 1).abs() <= eps2 * (v!(d, mm) * v!(d, mm - 1)).abs() {
                        m = mm;
                        break;
                    }
                    mm -= 1;
                }
                // 120
                if m > lend {
                    v!(e, m - 1) = 0.0;
                }
                let mut p = v!(d, l);
                if m == l {
                    // 140
                    v!(d, l) = p;
                    l -= 1;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if m == l - 1 {
                    let rte = v!(e, l - 1).sqrt();
                    let (rt1, rt2) = dlae2(v!(d, l), rte, v!(d, l - 1));
                    v!(d, l) = rt1;
                    v!(d, l - 1) = rt2;
                    v!(e, l - 1) = 0.0;
                    l -= 2;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;

                // Form shift.
                let rte = v!(e, l - 1).sqrt();
                sigma = (v!(d, l - 1) - p) / (2.0 * rte);
                let r = dlapy2(sigma, 1.0);
                sigma = p - (rte / (sigma + sign(r, sigma)));

                let mut c = 1.0;
                let mut s = 0.0;
                let mut gamma = v!(d, m) - sigma;
                p = gamma * gamma;

                for i in m..=(l - 1) {
                    let bb = v!(e, i);
                    let r = p + bb;
                    if i != m {
                        v!(e, i - 1) = s * r;
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = v!(d, i + 1);
                    gamma = fma(c, alpha - sigma, -(s * oldgam));
                    v!(d, i) = oldgam + (alpha - gamma);
                    if c != 0.0 {
                        p = (gamma * gamma) / c;
                    } else {
                        p = oldc * bb;
                    }
                }

                v!(e, l - 1) = s * p;
                v!(d, l) = sigma + gamma;
            }
        }

        // 150: undo scaling if necessary.
        if iscale == 1 {
            dlascl_g(ssfmax, anorm, lendsv - lsv + 1, 1, pv!(d, lsv), n);
        }
        if iscale == 2 {
            dlascl_g(ssfmin, anorm, lendsv - lsv + 1, 1, pv!(d, lsv), n);
        }

        if jtot < nmaxit {
            continue 'outer;
        }
        for i in 1..=(n - 1) {
            if v!(e, i) != 0.0 {
                info += 1;
            }
        }
        return info;
    }

    // 170
    dlasrt(b'I', n, d);
    info
}
