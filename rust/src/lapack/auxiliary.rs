//! LAPACK's small auxiliary routines.

use super::{disnan, fma, lsame, sign, HUGEVAL, SAFMIN};

/// `DLAPY2`: `sqrt(x**2 + y**2)` without unnecessary overflow.
pub(crate) fn dlapy2(x: f64, y: f64) -> f64 {
    let x_is_nan = disnan(x);
    let y_is_nan = disnan(y);
    let mut r = 0.0;
    if x_is_nan {
        r = x;
    }
    if y_is_nan {
        r = y;
    }
    if !(x_is_nan || y_is_nan) {
        let xabs = x.abs();
        let yabs = y.abs();
        let w = xabs.max(yabs);
        let z = xabs.min(yabs);
        if z == 0.0 || w > HUGEVAL {
            r = w;
        } else {
            let q = z / w;
            r = w * fma(q, q, 1.0).sqrt();
        }
    }
    r
}

/// `DLAE2`: eigenvalues of `[[a, b], [b, c]]`, `rt1` of larger magnitude.
pub(crate) fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let q = ab / adf;
        adf * fma(q, q, 1.0).sqrt()
    } else if adf < ab {
        let q = adf / ab;
        ab * fma(q, q, 1.0).sqrt()
    } else {
        ab * 2.0f64.sqrt()
    };
    let (rt1, rt2);
    if sm < 0.0 {
        rt1 = 0.5 * (sm - rt);
        rt2 = fma(acmx / rt1, acmn, -((b / rt1) * b));
    } else if sm > 0.0 {
        rt1 = 0.5 * (sm + rt);
        rt2 = fma(acmx / rt1, acmn, -((b / rt1) * b));
    } else {
        rt1 = 0.5 * rt;
        rt2 = -0.5 * rt;
    }
    (rt1, rt2)
}

/// `DLAEV2`: eigendecomposition of `[[a, b], [b, c]]`; returns
/// `(rt1, rt2, cs1, sn1)` with `(cs1, sn1)` the unit eigenvector of `rt1`.
pub(crate) fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let q = ab / adf;
        adf * fma(q, q, 1.0).sqrt()
    } else if adf < ab {
        let q = adf / ab;
        ab * fma(q, q, 1.0).sqrt()
    } else {
        ab * 2.0f64.sqrt()
    };
    let (rt1, rt2, sgn1);
    if sm < 0.0 {
        rt1 = 0.5 * (sm - rt);
        sgn1 = -1;
        rt2 = fma(acmx / rt1, acmn, -((b / rt1) * b));
    } else if sm > 0.0 {
        rt1 = 0.5 * (sm + rt);
        sgn1 = 1;
        rt2 = fma(acmx / rt1, acmn, -((b / rt1) * b));
    } else {
        rt1 = 0.5 * rt;
        rt2 = -0.5 * rt;
        sgn1 = 1;
    }
    let (cs, sgn2) = if df >= 0.0 {
        (df + rt, 1)
    } else {
        (df - rt, -1)
    };
    let acs = cs.abs();
    let (mut cs1, mut sn1);
    if acs > ab {
        let ct = -tb / cs;
        sn1 = 1.0 / fma(ct, ct, 1.0).sqrt();
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1.0 / fma(tn, tn, 1.0).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

/// `DLARTG` (LAPACK 3.12, `la_xlartg`): the plane rotation `(c, s)` with
/// `[c s; -s c] [f; g] = [r; 0]`; returns `(c, s, r)`.
pub(crate) fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    let safmax = 1.0 / SAFMIN;
    let rtmin = SAFMIN.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let f1 = f.abs();
    let g1 = g.abs();
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, sign(1.0, g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = fma(f, f, g * g).sqrt();
        let c = f1 / d;
        let r = sign(d, f);
        let s = g / r;
        (c, s, r)
    } else {
        let u = safmax.min(SAFMIN.max(f1).max(g1));
        let fs = f / u;
        let gs = g / u;
        let d = fma(fs, fs, gs * gs).sqrt();
        let c = fs.abs() / d;
        let r = sign(d, f);
        let s = gs / r;
        (c, s, r * u)
    }
}

/// `DLASCL` for `TYPE = 'G'`: multiplies the `m`-by-`n` matrix `a` by
/// `cto/cfrom` in steps that cannot over- or underflow.
pub(crate) unsafe fn dlascl_g(cfrom: f64, cto: f64, m: i32, n: i32, a: *mut f64, lda: i32) {
    if n == 0 || m == 0 {
        return;
    }
    let smlnum = SAFMIN;
    let bignum = 1.0 / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let cfrom1 = cfromc * smlnum;
        let mul;
        let done;
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        for j in 1..=n {
            for i in 1..=m {
                m!(a, lda, i, j) *= mul;
            }
        }
        if done {
            return;
        }
    }
}

/// `DLASET`: off-diagonal entries of the `uplo` part set to `alpha`, the
/// diagonal to `beta`.
pub(crate) unsafe fn dlaset(
    uplo: u8,
    m: i32,
    n: i32,
    alpha: f64,
    beta: f64,
    a: *mut f64,
    lda: i32,
) {
    if lsame(uplo, b'U') {
        for j in 2..=n {
            for i in 1..=(j - 1).min(m) {
                m!(a, lda, i, j) = alpha;
            }
        }
    } else if lsame(uplo, b'L') {
        for j in 1..=m.min(n) {
            for i in (j + 1)..=m {
                m!(a, lda, i, j) = alpha;
            }
        }
    } else {
        for j in 1..=n {
            for i in 1..=m {
                m!(a, lda, i, j) = alpha;
            }
        }
    }
    for i in 1..=m.min(n) {
        m!(a, lda, i, i) = beta;
    }
}

/// `DLACPY`: copies the `uplo` part of `a` into `b`.
pub(crate) unsafe fn dlacpy(
    uplo: u8,
    m: i32,
    n: i32,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    if lsame(uplo, b'U') {
        for j in 1..=n {
            for i in 1..=j.min(m) {
                m!(b, ldb, i, j) = m!(a, lda, i, j);
            }
        }
    } else if lsame(uplo, b'L') {
        for j in 1..=n {
            for i in j..=m {
                m!(b, ldb, i, j) = m!(a, lda, i, j);
            }
        }
    } else {
        for j in 1..=n {
            for i in 1..=m {
                m!(b, ldb, i, j) = m!(a, lda, i, j);
            }
        }
    }
}

/// `DLANST('M', ...)`: the largest absolute entry of the symmetric
/// tridiagonal matrix with diagonal `d` and off-diagonal `e`.
pub(crate) unsafe fn dlanst_m(n: i32, d: *const f64, e: *const f64) -> f64 {
    if n <= 0 {
        return 0.0;
    }
    let mut anorm = v!(d, n).abs();
    for i in 1..=(n - 1) {
        let sum = v!(d, i).abs();
        if anorm < sum || disnan(sum) {
            anorm = sum;
        }
        let sum = v!(e, i).abs();
        if anorm < sum || disnan(sum) {
            anorm = sum;
        }
    }
    anorm
}

/// `DLASRT`: sorts `d` increasingly (`id = 'I'`) or decreasingly (`'D'`),
/// by LAPACK's quicksort with insertion sort below 20 entries.
pub(crate) unsafe fn dlasrt(id: u8, n: i32, d: *mut f64) {
    const SELECT: i32 = 20;
    let dir = if lsame(id, b'D') {
        0
    } else if lsame(id, b'I') {
        1
    } else {
        -1
    };
    if dir == -1 || n <= 1 {
        return;
    }
    let mut stack = [[0i32; 2]; 32];
    let mut stkpnt: i32 = 1;
    stack[0][0] = 1;
    stack[0][1] = n;
    loop {
        let start = stack[(stkpnt - 1) as usize][0];
        let endd = stack[(stkpnt - 1) as usize][1];
        stkpnt -= 1;
        if endd - start <= SELECT && endd - start > 0 {
            if dir == 0 {
                for i in (start + 1)..=endd {
                    let mut j = i;
                    while j >= start + 1 {
                        if v!(d, j) > v!(d, j - 1) {
                            let dmnmx = v!(d, j);
                            v!(d, j) = v!(d, j - 1);
                            v!(d, j - 1) = dmnmx;
                        } else {
                            break;
                        }
                        j -= 1;
                    }
                }
            } else {
                for i in (start + 1)..=endd {
                    let mut j = i;
                    while j >= start + 1 {
                        if v!(d, j) < v!(d, j - 1) {
                            let dmnmx = v!(d, j);
                            v!(d, j) = v!(d, j - 1);
                            v!(d, j - 1) = dmnmx;
                        } else {
                            break;
                        }
                        j -= 1;
                    }
                }
            }
        } else if endd - start > SELECT {
            let d1 = v!(d, start);
            let d2 = v!(d, endd);
            let i = (start + endd) / 2;
            let d3 = v!(d, i);
            let dmnmx = if d1 < d2 {
                if d3 < d1 {
                    d1
                } else if d3 < d2 {
                    d3
                } else {
                    d2
                }
            } else if d3 < d2 {
                d2
            } else if d3 < d1 {
                d3
            } else {
                d1
            };
            let mut i = start - 1;
            let mut j = endd + 1;
            if dir == 0 {
                loop {
                    j -= 1;
                    while v!(d, j) < dmnmx {
                        j -= 1;
                    }
                    i += 1;
                    while v!(d, i) > dmnmx {
                        i += 1;
                    }
                    if i < j {
                        let tmp = v!(d, i);
                        v!(d, i) = v!(d, j);
                        v!(d, j) = tmp;
                    } else {
                        break;
                    }
                }
            } else {
                loop {
                    j -= 1;
                    while v!(d, j) > dmnmx {
                        j -= 1;
                    }
                    i += 1;
                    while v!(d, i) < dmnmx {
                        i += 1;
                    }
                    if i < j {
                        let tmp = v!(d, i);
                        v!(d, i) = v!(d, j);
                        v!(d, j) = tmp;
                    } else {
                        break;
                    }
                }
            }
            stkpnt += 1;
            let (first, second) = if j - start > endd - j - 1 {
                ([start, j], [j + 1, endd])
            } else {
                ([j + 1, endd], [start, j])
            };
            stack[(stkpnt - 1) as usize] = first;
            stkpnt += 1;
            stack[(stkpnt - 1) as usize] = second;
        }
        if stkpnt <= 0 {
            return;
        }
    }
}

/// `DLAMRG`: the permutation `index` merging two sorted runs of `a` (`n1`
/// entries then `n2`, each read forwards when its stride is positive) into
/// one ascending list.
pub(crate) unsafe fn dlamrg(
    n1: i32,
    n2: i32,
    a: *const f64,
    dtrd1: i32,
    dtrd2: i32,
    index: *mut i32,
) {
    let mut n1sv = n1;
    let mut n2sv = n2;
    let mut ind1 = if dtrd1 > 0 { 1 } else { n1 };
    let mut ind2 = if dtrd2 > 0 { 1 + n1 } else { n1 + n2 };
    let mut i = 1;
    while n1sv > 0 && n2sv > 0 {
        if v!(a, ind1) <= v!(a, ind2) {
            v!(index, i) = ind1;
            i += 1;
            ind1 += dtrd1;
            n1sv -= 1;
        } else {
            v!(index, i) = ind2;
            i += 1;
            ind2 += dtrd2;
            n2sv -= 1;
        }
    }
    if n1sv == 0 {
        for _ in 1..=n2sv {
            v!(index, i) = ind2;
            i += 1;
            ind2 += dtrd2;
        }
    } else {
        for _ in 1..=n1sv {
            v!(index, i) = ind1;
            i += 1;
            ind1 += dtrd1;
        }
    }
}

/// `DLASR`: applies the sequence of plane rotations `(c(k), s(k))` to `a`
/// from the left or right, pivot `'V'` (variable), `'T'` (top) or `'B'`
/// (bottom), forward or backward.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlasr(
    side: u8,
    pivot: u8,
    direct: u8,
    m: i32,
    n: i32,
    c: *const f64,
    s: *const f64,
    a: *mut f64,
    lda: i32,
) {
    if m == 0 || n == 0 {
        return;
    }
    let fwd = lsame(direct, b'F');
    let span = |dim: i32| {
        let (first, last) = if lsame(pivot, b'T') {
            (2, dim)
        } else {
            (1, dim - 1)
        };
        if fwd {
            (first, 1, last + 1)
        } else {
            (last, -1, first - 1)
        }
    };
    if lsame(side, b'L') {
        let (mut j, step, stop) = span(m);
        while j != stop {
            if lsame(pivot, b'V') {
                let ctemp = v!(c, j);
                let stemp = v!(s, j);
                if ctemp != 1.0 || stemp != 0.0 {
                    for i in 1..=n {
                        let temp = m!(a, lda, j + 1, i);
                        let aji = m!(a, lda, j, i);
                        m!(a, lda, j + 1, i) = fma(ctemp, temp, -(stemp * aji));
                        m!(a, lda, j, i) = fma(stemp, temp, ctemp * aji);
                    }
                }
            } else if lsame(pivot, b'T') {
                let ctemp = v!(c, j - 1);
                let stemp = v!(s, j - 1);
                if ctemp != 1.0 || stemp != 0.0 {
                    for i in 1..=n {
                        let temp = m!(a, lda, j, i);
                        let a1i = m!(a, lda, 1, i);
                        m!(a, lda, j, i) = fma(ctemp, temp, -(stemp * a1i));
                        m!(a, lda, 1, i) = fma(stemp, temp, ctemp * a1i);
                    }
                }
            } else if lsame(pivot, b'B') {
                let ctemp = v!(c, j);
                let stemp = v!(s, j);
                if ctemp != 1.0 || stemp != 0.0 {
                    for i in 1..=n {
                        let temp = m!(a, lda, j, i);
                        let ami = m!(a, lda, m, i);
                        m!(a, lda, j, i) = fma(ami, stemp, ctemp * temp);
                        m!(a, lda, m, i) = fma(ami, ctemp, -(stemp * temp));
                    }
                }
            }
            j += step;
        }
    } else if lsame(side, b'R') {
        let (mut j, step, stop) = span(n);
        while j != stop {
            if lsame(pivot, b'V') {
                let ctemp = v!(c, j);
                let stemp = v!(s, j);
                if ctemp != 1.0 || stemp != 0.0 {
                    for i in 1..=m {
                        let temp = m!(a, lda, i, j + 1);
                        let aij = m!(a, lda, i, j);
                        m!(a, lda, i, j + 1) = fma(ctemp, temp, -(stemp * aij));
                        m!(a, lda, i, j) = fma(stemp, temp, ctemp * aij);
                    }
                }
            } else if lsame(pivot, b'T') {
                let ctemp = v!(c, j - 1);
                let stemp = v!(s, j - 1);
                if ctemp != 1.0 || stemp != 0.0 {
                    for i in 1..=m {
                        let temp = m!(a, lda, i, j);
                        let ai1 = m!(a, lda, i, 1);
                        m!(a, lda, i, j) = fma(ctemp, temp, -(stemp * ai1));
                        m!(a, lda, i, 1) = fma(stemp, temp, ctemp * ai1);
                    }
                }
            } else if lsame(pivot, b'B') {
                let ctemp = v!(c, j);
                let stemp = v!(s, j);
                if ctemp != 1.0 || stemp != 0.0 {
                    for i in 1..=m {
                        let temp = m!(a, lda, i, j);
                        let ain = m!(a, lda, i, n);
                        m!(a, lda, i, j) = fma(ain, stemp, ctemp * temp);
                        m!(a, lda, i, n) = fma(ain, ctemp, -(stemp * temp));
                    }
                }
            }
            j += step;
        }
    }
}
