//! `DSTEDC`: eigenpairs of a symmetric tridiagonal matrix by Cuppen's divide
//! and conquer (`DLAED0`-`DLAED6`), with `DSTEQR` below `SMLSIZ`.

use super::auxiliary::{dlacpy, dlamrg, dlanst_m, dlapy2, dlascl_g, dlaset};
use super::blas::{dcopy, dgemm, dnrm2, drot, dscal, dswap, idamax};
use super::steqr::{dsteqr, dsterf};
use super::{fma, lsame, sign, EPS, SAFMIN};

/// `ILAENV(9, 'DSTEDC', ...)`: the largest subproblem solved by `DSTEQR`.
pub(crate) const SMLSIZ: i32 = 25;

/// `DSTEDC` for `compz = 'N'` or `'I'`: eigenvalues of the tridiagonal
/// `(d, e)` into `d` (ascending) and, for `'I'`, its eigenvectors into `z`.
/// `work`/`iwork` must hold [`dstedc_lwork`] doubles and ints. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dstedc(
    compz: u8,
    n: i32,
    d: *mut f64,
    e: *mut f64,
    z: *mut f64,
    ldz: i32,
    work: *mut f64,
    iwork: *mut i32,
) -> i32 {
    let icompz = if lsame(compz, b'N') {
        0
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
        if icompz != 0 {
            m!(z, ldz, 1, 1) = 1.0;
        }
        return 0;
    }

    if icompz == 0 {
        return dsterf(n, d, e);
    }

    if n <= SMLSIZ {
        return dsteqr(compz, n, d, e, z, ldz, work);
    }

    // ICOMPZ = 2.
    let storez = 1;
    dlaset(b'F', n, n, 0.0, 1.0, z, ldz);

    let orgnrm = dlanst_m(n, d, e);
    if orgnrm == 0.0 {
        return 0;
    }

    let eps = EPS;

    let mut start = 1;
    while start <= n {
        // Let FINISH be the position of the next subdiagonal entry such that
        // E(FINISH) <= TINY, or FINISH = N if there is none.
        let mut finish = start;
        while finish < n {
            let tiny = eps * v!(d, finish).abs().sqrt() * v!(d, finish + 1).abs().sqrt();
            if v!(e, finish).abs() > tiny {
                finish += 1;
            } else {
                break;
            }
        }

        let m = finish - start + 1;
        if m == 1 {
            start = finish + 1;
            continue;
        }
        if m > SMLSIZ {
            let orgnrm = dlanst_m(m, pv!(d, start), pv!(e, start));
            dlascl_g(orgnrm, 1.0, m, 1, pv!(d, start), m);
            dlascl_g(orgnrm, 1.0, m - 1, 1, pv!(e, start), m - 1);

            let strtrw = start;
            let info = dlaed0(
                n,
                m,
                pv!(d, start),
                pv!(e, start),
                pm!(z, ldz, strtrw, start),
                ldz,
                pv!(work, storez),
                iwork,
            );
            if info != 0 {
                return (info / (m + 1) + start - 1) * (n + 1) + info % (m + 1) + start - 1;
            }

            dlascl_g(1.0, orgnrm, m, 1, pv!(d, start), m);
        } else {
            let info = dsteqr(
                b'I',
                m,
                pv!(d, start),
                pv!(e, start),
                pm!(z, ldz, start, start),
                ldz,
                work,
            );
            if info != 0 {
                return start * (n + 1) + finish;
            }
        }

        start = finish + 1;
    }

    // Selection sort, to minimize swaps of eigenvectors.
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
    0
}

/// `(LWORK, LIWORK)` that `DSTEDC('I', N, ...)` asks for.
pub(crate) fn dstedc_lwork(n: i32) -> (usize, usize) {
    if n <= 1 {
        (1, 1)
    } else if n <= SMLSIZ {
        ((2 * (n - 1)) as usize, 1)
    } else {
        ((1 + 4 * n + n * n) as usize, (3 + 5 * n) as usize)
    }
}

/// `DLAED0` for `ICOMPQ = 2`: eigenpairs of the `n`-by-`n` tridiagonal
/// `(d, e)` into `d` and `q` by recursive halving.
#[allow(clippy::too_many_arguments)]
unsafe fn dlaed0(
    _qsiz: i32,
    n: i32,
    d: *mut f64,
    e: *mut f64,
    q: *mut f64,
    ldq: i32,
    work: *mut f64,
    iwork: *mut i32,
) -> i32 {
    if n == 0 {
        return 0;
    }
    let smlsiz = SMLSIZ;

    // Determine the size and placement of the submatrices, and save in the
    // leading elements of IWORK.
    v!(iwork, 1) = n;
    let mut subpbs = 1;
    while v!(iwork, subpbs) > smlsiz {
        let mut j = subpbs;
        while j >= 1 {
            v!(iwork, 2 * j) = (v!(iwork, j) + 1) / 2;
            v!(iwork, 2 * j - 1) = v!(iwork, j) / 2;
            j -= 1;
        }
        subpbs *= 2;
    }
    for j in 2..=subpbs {
        v!(iwork, j) += v!(iwork, j - 1);
    }

    // Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
    // using rank-1 modifications (cuts).
    let spm1 = subpbs - 1;
    for i in 1..=spm1 {
        let submat = v!(iwork, i) + 1;
        let smm1 = submat - 1;
        v!(d, smm1) -= v!(e, smm1).abs();
        v!(d, submat) -= v!(e, smm1).abs();
    }

    let indxq = 4 * n + 3;

    // Solve each submatrix eigenproblem at the bottom of the divide and
    // conquer tree.
    let mut submat: i32;
    let mut matsiz: i32;
    let mut info;
    for i in 0..=spm1 {
        if i == 0 {
            submat = 1;
            matsiz = v!(iwork, 1);
        } else {
            submat = v!(iwork, i) + 1;
            matsiz = v!(iwork, i + 1) - v!(iwork, i);
        }
        info = dsteqr(
            b'I',
            matsiz,
            pv!(d, submat),
            pv!(e, submat),
            pm!(q, ldq, submat, submat),
            ldq,
            work,
        );
        if info != 0 {
            return submat * (n + 1) + submat + matsiz - 1;
        }
        let mut k = 1;
        for j in submat..=v!(iwork, i + 1) {
            v!(iwork, indxq + j) = k;
            k += 1;
        }
    }

    // Successively merge eigensystems of adjacent submatrices into the
    // eigensystem for the corresponding larger matrix.
    while subpbs > 1 {
        let spm2 = subpbs - 2;
        let mut i = 0;
        while i <= spm2 {
            let msd2;
            if i == 0 {
                submat = 1;
                matsiz = v!(iwork, 2);
                msd2 = v!(iwork, 1);
            } else {
                submat = v!(iwork, i) + 1;
                matsiz = v!(iwork, i + 2) - v!(iwork, i);
                msd2 = matsiz / 2;
            }
            info = dlaed1(
                matsiz,
                pv!(d, submat),
                pm!(q, ldq, submat, submat),
                ldq,
                pv!(iwork, indxq + submat),
                v!(e, submat + msd2 - 1),
                msd2,
                work,
                pv!(iwork, subpbs + 1),
            );
            if info != 0 {
                return submat * (n + 1) + submat + matsiz - 1;
            }
            v!(iwork, i / 2 + 1) = v!(iwork, i + 2);
            i += 2;
        }
        subpbs /= 2;
    }

    // Re-merge the eigenvalues/vectors which were deflated at the final
    // merge step.
    for i in 1..=n {
        let j = v!(iwork, indxq + i);
        v!(work, i) = v!(d, j);
        dcopy(n, pm!(q, ldq, 1, j), 1, pv!(work, n * i + 1), 1);
    }
    dcopy(n, work, 1, d, 1);
    dlacpy(b'A', n, n, pv!(work, n + 1), n, q, ldq);
    0
}

/// `DLAED1`: merges two adjacent eigensystems through the rank-one cut
/// `rho` at `cutpnt`.
#[allow(clippy::too_many_arguments)]
unsafe fn dlaed1(
    n: i32,
    d: *mut f64,
    q: *mut f64,
    ldq: i32,
    indxq: *mut i32,
    mut rho: f64,
    cutpnt: i32,
    work: *mut f64,
    iwork: *mut i32,
) -> i32 {
    if n == 0 {
        return 0;
    }
    // The following values are integer pointers which indicate the portion
    // of the workspace used by a particular array in DLAED2 and DLAED3.
    let iz = 1;
    let idlmda = iz + n;
    let iw = idlmda + n;
    let iq2 = iw + n;

    let indx = 1;
    let indxc = indx + n;
    let coltyp = indxc + n;
    let indxp = coltyp + n;

    // Form the z-vector which consists of the last row of Q_1 and the first
    // row of Q_2.
    dcopy(cutpnt, pm!(q, ldq, cutpnt, 1), ldq, pv!(work, iz), 1);
    let zpp1 = cutpnt + 1;
    dcopy(
        n - cutpnt,
        pm!(q, ldq, zpp1, zpp1),
        ldq,
        pv!(work, iz + cutpnt),
        1,
    );

    // Deflate eigenvalues.
    let mut k = 0;
    let mut info = dlaed2(
        &mut k,
        n,
        cutpnt,
        d,
        q,
        ldq,
        indxq,
        &mut rho,
        pv!(work, iz),
        pv!(work, idlmda),
        pv!(work, iw),
        pv!(work, iq2),
        pv!(iwork, indx),
        pv!(iwork, indxc),
        pv!(iwork, indxp),
        pv!(iwork, coltyp),
    );
    if info != 0 {
        return info;
    }

    // Solve Secular Equation.
    if k != 0 {
        let is = (v!(iwork, coltyp) + v!(iwork, coltyp + 1)) * cutpnt
            + (v!(iwork, coltyp + 1) + v!(iwork, coltyp + 2)) * (n - cutpnt)
            + iq2;
        info = dlaed3(
            k,
            n,
            cutpnt,
            d,
            q,
            ldq,
            rho,
            pv!(work, idlmda),
            pv!(work, iq2),
            pv!(iwork, indxc),
            pv!(iwork, coltyp),
            pv!(work, iw),
            pv!(work, is),
        );
        if info != 0 {
            return info;
        }

        // Prepare the INDXQ sorting permutation.
        let n1 = k;
        let n2 = n - k;
        dlamrg(n1, n2, d, 1, -1, indxq);
    } else {
        for i in 1..=n {
            v!(indxq, i) = i;
        }
    }
    0
}

/// `DLAED2`: deflation of the merged system (small `z` entries, close
/// eigenvalue pairs rotated together), and the column grouping `DLAED3`
/// multiplies with.
#[allow(clippy::too_many_arguments)]
unsafe fn dlaed2(
    k: &mut i32,
    n: i32,
    n1: i32,
    d: *mut f64,
    q: *mut f64,
    ldq: i32,
    indxq: *mut i32,
    rho: &mut f64,
    z: *mut f64,
    dlambda: *mut f64,
    w: *mut f64,
    q2: *mut f64,
    indx: *mut i32,
    indxc: *mut i32,
    indxp: *mut i32,
    coltyp: *mut i32,
) -> i32 {
    if n == 0 {
        return 0;
    }
    let mut ctot = [0i32; 4];
    let mut psm = [0i32; 4];

    let n2 = n - n1;
    let n1p1 = n1 + 1;

    if *rho < 0.0 {
        dscal(n2, -1.0, pv!(z, n1p1), 1);
    }

    // Normalize z so that norm(z) = 1. Since z is the concatenation of two
    // normalized vectors, norm2(z) = sqrt(2).
    let t = 1.0 / 2.0f64.sqrt();
    dscal(n, t, z, 1);

    // RHO = ABS( norm(z)**2 * RHO )
    *rho = (2.0 * *rho).abs();

    // Sort the eigenvalues into increasing order.
    for i in n1p1..=n {
        v!(indxq, i) += n1;
    }

    // re-integrate the deflated parts from the last pass
    for i in 1..=n {
        v!(dlambda, i) = v!(d, v!(indxq, i));
    }
    dlamrg(n1, n2, dlambda, 1, 1, indxc);
    for i in 1..=n {
        v!(indx, i) = v!(indxq, v!(indxc, i));
    }

    // Calculate the allowable deflation tolerance.
    let imax = idamax(n, z, 1);
    let jmax = idamax(n, d, 1);
    let eps = EPS;
    let tol = 8.0 * eps * v!(d, jmax).abs().max(v!(z, imax).abs());

    // If the rank-1 modifier is small enough, no more needs to be done
    // except to reorganize Q so that its columns correspond with the
    // elements in D.
    if *rho * v!(z, imax).abs() <= tol {
        *k = 0;
        let mut iq2 = 1;
        for j in 1..=n {
            let i = v!(indx, j);
            dcopy(n, pm!(q, ldq, 1, i), 1, pv!(q2, iq2), 1);
            v!(dlambda, j) = v!(d, i);
            iq2 += n;
        }
        dlacpy(b'A', n, n, q2, n, q, ldq);
        dcopy(n, dlambda, 1, d, 1);
        return 0;
    }

    // If there are multiple eigenvalues then the problem deflates. Here the
    // number of equal eigenvalues are found. As each equal eigenvalue is
    // found, an elementary reflector is computed to rotate the
    // corresponding eigensubspace so that the corresponding components of
    // Z are zero in this new basis.
    for i in 1..=n1 {
        v!(coltyp, i) = 1;
    }
    for i in n1p1..=n {
        v!(coltyp, i) = 3;
    }

    *k = 0;
    let mut k2 = n + 1;
    let mut pj = 0;
    let mut j = 1;
    let mut found = false;
    while j <= n {
        let nj = v!(indx, j);
        if *rho * v!(z, nj).abs() <= tol {
            // Deflate due to small z component.
            k2 -= 1;
            v!(coltyp, nj) = 4;
            v!(indxp, k2) = nj;
            if j == n {
                break;
            }
        } else {
            pj = nj;
            found = true;
            break;
        }
        j += 1;
    }
    if found {
        loop {
            // 80
            j += 1;
            if j > n {
                break;
            }
            let nj = v!(indx, j);
            if *rho * v!(z, nj).abs() <= tol {
                // Deflate due to small z component.
                k2 -= 1;
                v!(coltyp, nj) = 4;
                v!(indxp, k2) = nj;
            } else {
                // Check if eigenvalues are close enough to allow deflation.
                let mut s = v!(z, pj);
                let mut c = v!(z, nj);

                // Find sqrt(a**2+b**2) without overflow or destructive
                // underflow.
                let tau = dlapy2(c, s);
                let t = v!(d, nj) - v!(d, pj);
                c /= tau;
                s = -s / tau;
                if (t * c * s).abs() <= tol {
                    // Deflation is possible.
                    v!(z, nj) = tau;
                    v!(z, pj) = 0.0;
                    if v!(coltyp, nj) != v!(coltyp, pj) {
                        v!(coltyp, nj) = 2;
                    }
                    v!(coltyp, pj) = 4;
                    drot(n, pm!(q, ldq, 1, pj), 1, pm!(q, ldq, 1, nj), 1, c, s);
                    let c2 = c * c;
                    let s2 = s * s;
                    let t = fma(v!(d, pj), c2, v!(d, nj) * s2);
                    v!(d, nj) = fma(v!(d, pj), s2, v!(d, nj) * c2);
                    v!(d, pj) = t;
                    k2 -= 1;
                    let mut i = 1;
                    loop {
                        if k2 + i <= n {
                            if v!(d, pj) < v!(d, v!(indxp, k2 + i)) {
                                v!(indxp, k2 + i - 1) = v!(indxp, k2 + i);
                                v!(indxp, k2 + i) = pj;
                                i += 1;
                                continue;
                            } else {
                                v!(indxp, k2 + i - 1) = pj;
                            }
                        } else {
                            v!(indxp, k2 + i - 1) = pj;
                        }
                        break;
                    }
                    pj = nj;
                } else {
                    *k += 1;
                    v!(dlambda, *k) = v!(d, pj);
                    v!(w, *k) = v!(z, pj);
                    v!(indxp, *k) = pj;
                    pj = nj;
                }
            }
        }
        // 100: record the last eigenvalue.
        *k += 1;
        v!(dlambda, *k) = v!(d, pj);
        v!(w, *k) = v!(z, pj);
        v!(indxp, *k) = pj;
    }

    // Count up the total number of the various types of columns, then form
    // a permutation which positions the four column types into four
    // uniform groups (although one or more of these groups may be empty).
    for jj in 1..=4 {
        ctot[jj - 1] = 0;
    }
    for jj in 1..=n {
        let ct = v!(coltyp, jj);
        ctot[(ct - 1) as usize] += 1;
    }

    // PSM(*) = Position in SubMatrix (of types 1 through 4)
    psm[0] = 1;
    psm[1] = 1 + ctot[0];
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    *k = n - ctot[3];

    // Fill out the INDXC array so that the permutation which it induces
    // will place all type-1 columns first, all type-2 columns next, then
    // all type-3's, and finally all type-4's.
    for jj in 1..=n {
        let js = v!(indxp, jj);
        let ct = v!(coltyp, js) as usize;
        v!(indx, psm[ct - 1]) = js;
        v!(indxc, psm[ct - 1]) = jj;
        psm[ct - 1] += 1;
    }

    // Sort the eigenvalues and corresponding eigenvectors into DLAMBDA and
    // Q2 respectively. The eigenvalues/vectors which were not deflated go
    // into the first K slots of DLAMBDA and Q2 respectively, while those
    // which were deflated go into the last N - K slots.
    let mut i = 1;
    let mut iq1 = 1;
    let mut iq2 = 1 + (ctot[0] + ctot[1]) * n1;
    for _ in 1..=ctot[0] {
        let js = v!(indx, i);
        dcopy(n1, pm!(q, ldq, 1, js), 1, pv!(q2, iq1), 1);
        v!(z, i) = v!(d, js);
        i += 1;
        iq1 += n1;
    }

    for _ in 1..=ctot[1] {
        let js = v!(indx, i);
        dcopy(n1, pm!(q, ldq, 1, js), 1, pv!(q2, iq1), 1);
        dcopy(n2, pm!(q, ldq, n1 + 1, js), 1, pv!(q2, iq2), 1);
        v!(z, i) = v!(d, js);
        i += 1;
        iq1 += n1;
        iq2 += n2;
    }

    for _ in 1..=ctot[2] {
        let js = v!(indx, i);
        dcopy(n2, pm!(q, ldq, n1 + 1, js), 1, pv!(q2, iq2), 1);
        v!(z, i) = v!(d, js);
        i += 1;
        iq2 += n2;
    }

    iq1 = iq2;
    for _ in 1..=ctot[3] {
        let js = v!(indx, i);
        dcopy(n, pm!(q, ldq, 1, js), 1, pv!(q2, iq2), 1);
        iq2 += n;
        v!(z, i) = v!(d, js);
        i += 1;
    }

    // The deflated eigenvalues and their corresponding vectors go back
    // into the last N - K slots of D and Q respectively.
    if *k < n {
        dlacpy(
            b'A',
            n,
            ctot[3],
            pv!(q2, iq1),
            n,
            pm!(q, ldq, 1, *k + 1),
            ldq,
        );
        dcopy(n - *k, pv!(z, *k + 1), 1, pv!(d, *k + 1), 1);
    }

    // Copy CTOT into COLTYP for referencing in DLAED3.
    for jj in 1..=4 {
        v!(coltyp, jj) = ctot[(jj - 1) as usize];
    }
    0
}

/// `DLAED3`: roots of the secular equation, the eigenvectors of the
/// deflated rank-one update (Gu-Eisenstat), and their back-multiplication.
#[allow(clippy::too_many_arguments)]
unsafe fn dlaed3(
    k: i32,
    n: i32,
    n1: i32,
    d: *mut f64,
    q: *mut f64,
    ldq: i32,
    rho: f64,
    dlambda: *mut f64,
    q2: *mut f64,
    indx: *mut i32,
    ctot: *mut i32,
    w: *mut f64,
    s: *mut f64,
) -> i32 {
    if k == 0 {
        return 0;
    }

    for j in 1..=k {
        let info = dlaed4(k, j, dlambda, w, pm!(q, ldq, 1, j), rho, &mut v!(d, j));
        // If the zero finder fails, the computation is terminated.
        if info != 0 {
            return info;
        }
    }

    if k == 1 {
        // nothing more for the secular vectors
    } else if k == 2 {
        for j in 1..=k {
            v!(w, 1) = m!(q, ldq, 1, j);
            v!(w, 2) = m!(q, ldq, 2, j);
            let ii = v!(indx, 1);
            m!(q, ldq, 1, j) = v!(w, ii);
            let ii = v!(indx, 2);
            m!(q, ldq, 2, j) = v!(w, ii);
        }
    } else {
        // Compute updated W.
        dcopy(k, w, 1, s, 1);

        // Initialize W(I) = Q(I,I)
        dcopy(k, q, ldq + 1, w, 1);
        for j in 1..=k {
            for i in 1..j {
                v!(w, i) *= m!(q, ldq, i, j) / (v!(dlambda, i) - v!(dlambda, j));
            }
            for i in (j + 1)..=k {
                v!(w, i) *= m!(q, ldq, i, j) / (v!(dlambda, i) - v!(dlambda, j));
            }
        }
        for i in 1..=k {
            v!(w, i) = sign((-v!(w, i)).sqrt(), v!(s, i));
        }

        // Compute eigenvectors of the modified rank-1 modification.
        for j in 1..=k {
            for i in 1..=k {
                v!(s, i) = v!(w, i) / m!(q, ldq, i, j);
            }
            let temp = dnrm2(k, s, 1);
            for i in 1..=k {
                let ii = v!(indx, i);
                m!(q, ldq, i, j) = v!(s, ii) / temp;
            }
        }
    }

    // Compute the updated eigenvectors.
    let n2 = n - n1;
    let n12 = v!(ctot, 1) + v!(ctot, 2);
    let n23 = v!(ctot, 2) + v!(ctot, 3);

    dlacpy(b'A', n23, k, pm!(q, ldq, v!(ctot, 1) + 1, 1), ldq, s, n23);
    let iq2 = n1 * n12 + 1;
    if n23 != 0 {
        dgemm(
            b'N',
            b'N',
            n2,
            k,
            n23,
            1.0,
            pv!(q2, iq2),
            n2,
            s,
            n23,
            0.0,
            pm!(q, ldq, n1 + 1, 1),
            ldq,
        );
    } else {
        dlaset(b'A', n2, k, 0.0, 0.0, pm!(q, ldq, n1 + 1, 1), ldq);
    }

    dlacpy(b'A', n12, k, q, ldq, s, n12);
    if n12 != 0 {
        dgemm(b'N', b'N', n1, k, n12, 1.0, q2, n1, s, n12, 0.0, q, ldq);
    } else {
        dlaset(b'A', n1, k, 0.0, 0.0, pm!(q, ldq, 1, 1), ldq);
    }
    0
}

const MAXIT4: i32 = 30;

/// `DLAED4`: the `i`-th root of the secular equation
/// `1/rho + sum_j z_j^2 / (d_j - lambda) = 0` (`d` ascending, `n >= 1`),
/// into `dlam`, with `delta(j) = d(j) - lambda` (or the eigenvector for
/// `n <= 2`). Returns `INFO`.
pub(crate) unsafe fn dlaed4(
    n: i32,
    i: i32,
    d: *const f64,
    z: *const f64,
    delta: *mut f64,
    rho: f64,
    dlam: &mut f64,
) -> i32 {
    // Quick return for N=1 and 2.
    if n == 1 {
        // Presumably, I=1 upon entry
        *dlam = fma(v!(z, 1), rho * v!(z, 1), v!(d, 1));
        v!(delta, 1) = 1.0;
        return 0;
    }
    if n == 2 {
        dlaed5(i, d, z, delta, rho, dlam);
        return 0;
    }

    let eps = EPS;
    let rhoinv = 1.0 / rho;
    let mut info = 0;

    if i == n {
        // The case I = N.
        let ii = n - 1;
        let mut niter = 1;

        // Calculate initial guess.
        let midpt = rho / 2.0;

        // If ||Z||_2 is not one, then TEMP should be set to
        // RHO * ||Z||_2^2 / TWO
        for j in 1..=n {
            v!(delta, j) = (v!(d, j) - v!(d, i)) - midpt;
        }

        let mut psi = 0.0;
        for j in 1..=(n - 2) {
            psi += v!(z, j) * v!(z, j) / v!(delta, j);
        }

        let c = rhoinv + psi;
        let w = c + v!(z, ii) * v!(z, ii) / v!(delta, ii) + v!(z, n) * v!(z, n) / v!(delta, n);

        let mut tau;
        let mut dltlb;
        let mut dltub;
        if w <= 0.0 {
            let temp = v!(z, n - 1) * v!(z, n - 1) / (v!(d, n) - v!(d, n - 1) + rho)
                + v!(z, n) * v!(z, n) / rho;
            if c <= temp {
                tau = rho;
            } else {
                let del = v!(d, n) - v!(d, n - 1);
                let znsq = v!(z, n) * v!(z, n);
                let a = fma(-c, del, v!(z, n - 1) * v!(z, n - 1)) + znsq;
                let b = znsq * del;
                let sq = fma(a, a, c * (b * 4.0)).sqrt();
                if a < 0.0 {
                    tau = (2.0 * b) / (sq - a);
                } else {
                    tau = (a + sq) / (2.0 * c);
                }
            }

            // It can be proved that
            //     D(N)+RHO/2 <= LAMBDA(N) < D(N)+TAU <= D(N)+RHO
            dltlb = midpt;
            dltub = rho;
        } else {
            let del = v!(d, n) - v!(d, n - 1);
            let znsq = v!(z, n) * v!(z, n);
            let a = fma(-c, del, v!(z, n - 1) * v!(z, n - 1)) + znsq;
            let b = znsq * del;
            let sq = fma(a, a, c * (b * 4.0)).sqrt();
            if a < 0.0 {
                tau = (2.0 * b) / (sq - a);
            } else {
                tau = (a + sq) / (2.0 * c);
            }

            // It can be proved that
            //     D(N) < D(N)+TAU < LAMBDA(N) < D(N)+RHO/2
            dltlb = 0.0;
            dltub = midpt;
        }

        for j in 1..=n {
            v!(delta, j) = (v!(d, j) - v!(d, i)) - tau;
        }

        // Evaluate PSI and the derivative DPSI
        let mut dpsi = 0.0;
        let mut psi = 0.0;
        let mut erretm = 0.0;
        for j in 1..=ii {
            let temp = v!(z, j) / v!(delta, j);
            psi = fma(v!(z, j), temp, psi);
            dpsi = fma(temp, temp, dpsi);
            erretm += psi;
        }
        erretm = erretm.abs();

        // Evaluate PHI and the derivative DPHI
        let temp = v!(z, n) / v!(delta, n);
        let mut phi = v!(z, n) * temp;
        let mut dphi = temp * temp;
        erretm = fma(
            tau.abs(),
            dpsi + dphi,
            (fma(-phi - psi, 8.0, erretm) - phi) + rhoinv,
        );

        let mut w = (rhoinv + phi) + psi;

        // Test for convergence
        if w.abs() <= eps * erretm {
            *dlam = v!(d, i) + tau;
            return info;
        }

        if w <= 0.0 {
            dltlb = dltlb.max(tau);
        } else {
            dltub = dltub.min(tau);
        }

        // Calculate the new step
        niter += 1;
        let mut c = fma(-dphi, v!(delta, n), fma(-v!(delta, n - 1), dpsi, w));
        let dd = v!(delta, n - 1) * v!(delta, n);
        let a = fma(-(dpsi + dphi), dd, w * (v!(delta, n - 1) + v!(delta, n)));
        let b = dd * w;
        if c < 0.0 {
            c = c.abs();
        }
        let mut eta;
        if c == 0.0 {
            eta = -w / (dpsi + dphi);
        } else if a >= 0.0 {
            eta = (a + fma(a, a, -(c * (b * 4.0))).abs().sqrt()) / (2.0 * c);
        } else {
            eta = (2.0 * b) / (a - fma(a, a, -(c * (b * 4.0))).abs().sqrt());
        }

        // Note, eta should be positive if w is negative, and eta should be
        // negative otherwise. However, if for some reason caused by roundoff,
        // eta*w > 0, we simply use one Newton step instead. This way will
        // guarantee eta*w < 0.
        if w * eta > 0.0 {
            eta = -w / (dpsi + dphi);
        }
        let temp = tau + eta;
        if temp > dltub || temp < dltlb {
            if w < 0.0 {
                eta = (dltub - tau) / 2.0;
            } else {
                eta = (dltlb - tau) / 2.0;
            }
        }
        for j in 1..=n {
            v!(delta, j) -= eta;
        }

        tau += eta;

        // Evaluate PSI and the derivative DPSI
        dpsi = 0.0;
        psi = 0.0;
        erretm = 0.0;
        for j in 1..=ii {
            let temp = v!(z, j) / v!(delta, j);
            psi = fma(v!(z, j), temp, psi);
            dpsi = fma(temp, temp, dpsi);
            erretm += psi;
        }
        erretm = erretm.abs();

        // Evaluate PHI and the derivative DPHI
        let temp = v!(z, n) / v!(delta, n);
        phi = v!(z, n) * temp;
        dphi = temp * temp;
        erretm = fma(
            tau.abs(),
            dpsi + dphi,
            (fma(-phi - psi, 8.0, erretm) - phi) + rhoinv,
        );

        w = (rhoinv + phi) + psi;

        // Main loop to update the values of the array DELTA
        let iter = niter + 1;
        for _niter in iter..=MAXIT4 {
            // Test for convergence
            if w.abs() <= eps * erretm {
                *dlam = v!(d, i) + tau;
                return info;
            }

            if w <= 0.0 {
                dltlb = dltlb.max(tau);
            } else {
                dltub = dltub.min(tau);
            }

            // Calculate the new step
            let c = fma(-dphi, v!(delta, n), fma(-v!(delta, n - 1), dpsi, w));
            let dd = v!(delta, n - 1) * v!(delta, n);
            let a = fma(v!(delta, n - 1) + v!(delta, n), w, -(dd * (dpsi + dphi)));
            let b = dd * w;
            if a >= 0.0 {
                eta = (a + fma(a, a, -(c * (b * 4.0))).abs().sqrt()) / (2.0 * c);
            } else {
                eta = (2.0 * b) / (a - fma(a, a, -(c * (b * 4.0))).abs().sqrt());
            }

            if w * eta > 0.0 {
                eta = -w / (dpsi + dphi);
            }
            let temp = tau + eta;
            if temp > dltub || temp < dltlb {
                if w < 0.0 {
                    eta = (dltub - tau) / 2.0;
                } else {
                    eta = (dltlb - tau) / 2.0;
                }
            }
            for j in 1..=n {
                v!(delta, j) -= eta;
            }

            tau += eta;

            // Evaluate PSI and the derivative DPSI
            dpsi = 0.0;
            psi = 0.0;
            erretm = 0.0;
            for j in 1..=ii {
                let temp = v!(z, j) / v!(delta, j);
                psi = fma(v!(z, j), temp, psi);
                dpsi = fma(temp, temp, dpsi);
                erretm += psi;
            }
            erretm = erretm.abs();

            // Evaluate PHI and the derivative DPHI; PHI = Z(N)*TEMP is fused
            // into each of its three uses here.
            let zn = v!(z, n);
            let temp = zn / v!(delta, n);
            dphi = temp * temp;
            erretm = fma(
                tau.abs(),
                dphi + dpsi,
                fma(-zn, temp, fma(fma(-zn, temp, -psi), 8.0, erretm)) + rhoinv,
            );

            w = fma(zn, temp, rhoinv) + psi;
        }

        // Return with INFO = 1, NITER = MAXIT and not converged
        info = 1;
        *dlam = v!(d, i) + tau;
        info
    } else {
        // The case for I < N
        let mut niter = 1;
        let ip1 = i + 1;

        // Calculate initial guess
        let del = v!(d, ip1) - v!(d, i);
        let midpt = del / 2.0;
        for j in 1..=n {
            v!(delta, j) = (v!(d, j) - v!(d, i)) - midpt;
        }

        let mut psi = 0.0;
        for j in 1..i {
            psi += v!(z, j) * v!(z, j) / v!(delta, j);
        }

        let mut phi = 0.0;
        let mut j = n;
        while j >= i + 2 {
            phi += v!(z, j) * v!(z, j) / v!(delta, j);
            j -= 1;
        }
        let c = rhoinv + psi + phi;
        let zisq = v!(z, i) * v!(z, i);
        let zip1sq = v!(z, ip1) * v!(z, ip1);
        let w = c + zisq / v!(delta, i) + zip1sq / v!(delta, ip1);
        let cdel = c * del;

        let orgati;
        let mut tau;
        let mut dltlb;
        let mut dltub;
        if w > 0.0 {
            // d(i)< the ith eigenvalue < (d(i)+d(i+1))/2
            //
            // We choose d(i) as origin.
            orgati = true;
            let a = cdel + zisq + zip1sq;
            let b = zisq * del;
            let sq = fma(a, a, -(c * (b * 4.0))).abs().sqrt();
            if a > 0.0 {
                tau = (2.0 * b) / (a + sq);
            } else {
                tau = (a - sq) / (2.0 * c);
            }
            dltlb = 0.0;
            dltub = midpt;
        } else {
            // (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)
            //
            // We choose d(i+1) as origin.
            orgati = false;
            let a = cdel - zisq - zip1sq;
            let b = zip1sq * del;
            let sq = fma(a, a, c * (b * 4.0)).abs().sqrt();
            if a < 0.0 {
                tau = (2.0 * b) / (a - sq);
            } else {
                tau = -(a + sq) / (2.0 * c);
            }
            dltlb = -midpt;
            dltub = 0.0;
        }

        if orgati {
            for j in 1..=n {
                v!(delta, j) = (v!(d, j) - v!(d, i)) - tau;
            }
        } else {
            for j in 1..=n {
                v!(delta, j) = (v!(d, j) - v!(d, ip1)) - tau;
            }
        }
        let ii = if orgati { i } else { i + 1 };
        let iim1 = ii - 1;
        let iip1 = ii + 1;

        // Evaluate PSI and the derivative DPSI
        let mut dpsi = 0.0;
        let mut psi = 0.0;
        let mut erretm = 0.0;
        for j in 1..=iim1 {
            let temp = v!(z, j) / v!(delta, j);
            psi = fma(v!(z, j), temp, psi);
            dpsi = fma(temp, temp, dpsi);
            erretm += psi;
        }
        erretm = erretm.abs();

        // Evaluate PHI and the derivative DPHI
        let mut dphi = 0.0;
        let mut phi = 0.0;
        let mut j = n;
        while j >= iip1 {
            let temp = v!(z, j) / v!(delta, j);
            phi = fma(v!(z, j), temp, phi);
            dphi = fma(temp, temp, dphi);
            erretm += phi;
            j -= 1;
        }

        let mut w = rhoinv + phi + psi;

        // W is the value of the secular function with its ii-th element
        // removed.
        let mut swtch3 = false;
        if orgati {
            if w < 0.0 {
                swtch3 = true;
            }
        } else if w > 0.0 {
            swtch3 = true;
        }
        if ii == 1 || ii == n {
            swtch3 = false;
        }

        let temp = v!(z, ii) / v!(delta, ii);
        let mut dw = fma(temp, temp, dpsi + dphi);
        let temp = v!(z, ii) * temp;
        w += temp;
        erretm = fma(
            tau.abs(),
            dw,
            fma(temp.abs(), 3.0, 2.0 * rhoinv + fma(phi - psi, 8.0, erretm)),
        );

        // Test for convergence
        if w.abs() <= eps * erretm {
            if orgati {
                *dlam = v!(d, i) + tau;
            } else {
                *dlam = v!(d, ip1) + tau;
            }
            return info;
        }

        if w <= 0.0 {
            dltlb = dltlb.max(tau);
        } else {
            dltub = dltub.min(tau);
        }

        // Calculate the new step
        niter += 1;
        let mut eta = 0.0;
        let mut zz = [0.0f64; 3];
        if !swtch3 {
            let c = if orgati {
                let q = v!(z, i) / v!(delta, i);
                fma(-(v!(d, i) - v!(d, ip1)), q * q, fma(-dw, v!(delta, ip1), w))
            } else {
                let q = v!(z, ip1) / v!(delta, ip1);
                fma(-(v!(d, ip1) - v!(d, i)), q * q, fma(-dw, v!(delta, i), w))
            };
            let dd = v!(delta, i) * v!(delta, ip1);
            let mut a = fma(v!(delta, i) + v!(delta, ip1), w, -(dd * dw));
            let b = dd * w;
            if c == 0.0 {
                if a == 0.0 {
                    if orgati {
                        a = fma(v!(delta, ip1) * v!(delta, ip1), dpsi + dphi, zisq);
                    } else {
                        a = fma(v!(delta, i) * v!(delta, i), dpsi + dphi, zip1sq);
                    }
                }
                eta = b / a;
            } else if a <= 0.0 {
                eta = (a - fma(a, a, -(c * (b * 4.0))).abs().sqrt()) / (2.0 * c);
            } else {
                eta = (2.0 * b) / (a + fma(a, a, -(c * (b * 4.0))).abs().sqrt());
            }
        } else {
            // Interpolation using THREE most relevant poles
            let temp = rhoinv + psi + phi;
            let c;
            if orgati {
                let temp1 = v!(z, iim1) / v!(delta, iim1);
                let temp1 = temp1 * temp1;
                c = fma(
                    -(v!(d, iim1) - v!(d, iip1)),
                    temp1,
                    fma(-v!(delta, iip1), dpsi + dphi, temp),
                );
                zz[0] = v!(z, iim1) * v!(z, iim1);
                zz[2] = v!(delta, iip1) * v!(delta, iip1) * ((dpsi - temp1) + dphi);
            } else {
                let temp1 = v!(z, iip1) / v!(delta, iip1);
                let temp1 = temp1 * temp1;
                c = fma(
                    -(v!(d, iip1) - v!(d, iim1)),
                    temp1,
                    fma(-v!(delta, iim1), dpsi + dphi, temp),
                );
                zz[0] = v!(delta, iim1) * v!(delta, iim1) * (dpsi + (dphi - temp1));
                zz[2] = v!(z, iip1) * v!(z, iip1);
            }
            zz[1] = v!(z, ii) * v!(z, ii);
            info = dlaed6(niter, orgati, c, pv!(delta, iim1), zz.as_ptr(), w, &mut eta);
            if info != 0 {
                return info;
            }
        }

        // Note, eta should be positive if w is negative, and eta should be
        // negative otherwise. However, if for some reason caused by roundoff,
        // eta*w > 0, we simply use one Newton step instead. This way will
        // guarantee eta*w < 0.
        if w * eta >= 0.0 {
            eta = -w / dw;
        }
        let temp = tau + eta;
        if temp > dltub || temp < dltlb {
            if w < 0.0 {
                eta = (dltub - tau) / 2.0;
            } else {
                eta = (dltlb - tau) / 2.0;
            }
        }

        let mut prew = w;

        for j in 1..=n {
            v!(delta, j) -= eta;
        }

        // Evaluate PSI and the derivative DPSI
        dpsi = 0.0;
        psi = 0.0;
        erretm = 0.0;
        for j in 1..=iim1 {
            let temp = v!(z, j) / v!(delta, j);
            psi = fma(v!(z, j), temp, psi);
            dpsi = fma(temp, temp, dpsi);
            erretm += psi;
        }
        erretm = erretm.abs();

        // Evaluate PHI and the derivative DPHI
        dphi = 0.0;
        phi = 0.0;
        let mut j = n;
        while j >= iip1 {
            let temp = v!(z, j) / v!(delta, j);
            phi = fma(v!(z, j), temp, phi);
            dphi = fma(temp, temp, dphi);
            erretm += phi;
            j -= 1;
        }

        let temp = v!(z, ii) / v!(delta, ii);
        dw = fma(temp, temp, dpsi + dphi);
        let temp = v!(z, ii) * temp;
        w = rhoinv + phi + psi + temp;
        erretm = fma(
            (tau + eta).abs(),
            dw,
            fma(temp.abs(), 3.0, fma(phi - psi, 8.0, erretm) + 2.0 * rhoinv),
        );

        let mut swtch = false;
        if orgati {
            if -w > prew.abs() / 10.0 {
                swtch = true;
            }
        } else if w > prew.abs() / 10.0 {
            swtch = true;
        }

        tau += eta;

        // Main loop to update the values of the array DELTA
        let iter = niter + 1;
        for niter in iter..=MAXIT4 {
            // Test for convergence
            if w.abs() <= eps * erretm {
                if orgati {
                    *dlam = v!(d, i) + tau;
                } else {
                    *dlam = v!(d, ip1) + tau;
                }
                return info;
            }

            if w <= 0.0 {
                dltlb = dltlb.max(tau);
            } else {
                dltub = dltub.min(tau);
            }

            // Calculate the new step
            if !swtch3 {
                let c = if !swtch {
                    if orgati {
                        let q = v!(z, i) / v!(delta, i);
                        fma(-(v!(d, i) - v!(d, ip1)), q * q, fma(-dw, v!(delta, ip1), w))
                    } else {
                        let q = v!(z, ip1) / v!(delta, ip1);
                        fma(-(v!(d, ip1) - v!(d, i)), q * q, fma(-dw, v!(delta, i), w))
                    }
                } else {
                    let temp = v!(z, ii) / v!(delta, ii);
                    if orgati {
                        dpsi += temp * temp;
                    } else {
                        dphi += temp * temp;
                    }
                    fma(-dphi, v!(delta, ip1), fma(-dpsi, v!(delta, i), w))
                };
                let dd = v!(delta, i) * v!(delta, ip1);
                let mut a = fma(v!(delta, i) + v!(delta, ip1), w, -(dd * dw));
                let b = dd * w;
                if c == 0.0 {
                    if a == 0.0 {
                        if !swtch {
                            if orgati {
                                a = fma(
                                    v!(z, i),
                                    v!(z, i),
                                    v!(delta, ip1) * v!(delta, ip1) * (dpsi + dphi),
                                );
                            } else {
                                a = fma(
                                    v!(z, ip1),
                                    v!(z, ip1),
                                    v!(delta, i) * v!(delta, i) * (dpsi + dphi),
                                );
                            }
                        } else {
                            a = fma(
                                v!(delta, i) * v!(delta, i),
                                dpsi,
                                v!(delta, ip1) * v!(delta, ip1) * dphi,
                            );
                        }
                    }
                    eta = b / a;
                } else if a <= 0.0 {
                    eta = (a - fma(a, a, -(c * (b * 4.0))).abs().sqrt()) / (2.0 * c);
                } else {
                    eta = (2.0 * b) / (a + fma(a, a, -(c * (b * 4.0))).abs().sqrt());
                }
            } else {
                // Interpolation using THREE most relevant poles
                let temp = rhoinv + psi + phi;
                let c;
                if swtch {
                    c = fma(-dphi, v!(delta, iip1), fma(-dpsi, v!(delta, iim1), temp));
                    zz[0] = v!(delta, iim1) * v!(delta, iim1) * dpsi;
                    zz[2] = v!(delta, iip1) * v!(delta, iip1) * dphi;
                } else if orgati {
                    let temp1 = v!(z, iim1) / v!(delta, iim1);
                    let temp1 = temp1 * temp1;
                    c = fma(
                        -(v!(d, iim1) - v!(d, iip1)),
                        temp1,
                        fma(-v!(delta, iip1), dpsi + dphi, temp),
                    );
                    zz[0] = v!(z, iim1) * v!(z, iim1);
                    zz[2] = v!(delta, iip1) * v!(delta, iip1) * ((dpsi - temp1) + dphi);
                } else {
                    let temp1 = v!(z, iip1) / v!(delta, iip1);
                    let temp1 = temp1 * temp1;
                    c = fma(
                        -(v!(d, iip1) - v!(d, iim1)),
                        temp1,
                        fma(-v!(delta, iim1), dpsi + dphi, temp),
                    );
                    zz[0] = v!(delta, iim1) * v!(delta, iim1) * (dpsi + (dphi - temp1));
                    zz[2] = v!(z, iip1) * v!(z, iip1);
                }
                info = dlaed6(niter, orgati, c, pv!(delta, iim1), zz.as_ptr(), w, &mut eta);
                if info != 0 {
                    return info;
                }
            }

            // Note, eta should be positive if w is negative, and eta should
            // be negative otherwise. However, if for some reason caused by
            // roundoff, eta*w > 0, we simply use one Newton step instead.
            // This way will guarantee eta*w < 0.
            if w * eta >= 0.0 {
                eta = -w / dw;
            }
            let temp = tau + eta;
            if temp > dltub || temp < dltlb {
                if w < 0.0 {
                    eta = (dltub - tau) / 2.0;
                } else {
                    eta = (dltlb - tau) / 2.0;
                }
            }

            for j in 1..=n {
                v!(delta, j) -= eta;
            }

            tau += eta;
            prew = w;

            // Evaluate PSI and the derivative DPSI
            dpsi = 0.0;
            psi = 0.0;
            erretm = 0.0;
            for j in 1..=iim1 {
                let temp = v!(z, j) / v!(delta, j);
                psi = fma(v!(z, j), temp, psi);
                dpsi = fma(temp, temp, dpsi);
                erretm += psi;
            }
            erretm = erretm.abs();

            // Evaluate PHI and the derivative DPHI
            dphi = 0.0;
            phi = 0.0;
            let mut j = n;
            while j >= iip1 {
                let temp = v!(z, j) / v!(delta, j);
                phi = fma(v!(z, j), temp, phi);
                dphi = fma(temp, temp, dphi);
                erretm += phi;
                j -= 1;
            }

            let temp = v!(z, ii) / v!(delta, ii);
            dw = fma(temp, temp, dpsi + dphi);
            let temp = v!(z, ii) * temp;
            w = rhoinv + phi + psi + temp;
            erretm = fma(
                tau.abs(),
                dw,
                fma(temp.abs(), 3.0, fma(phi - psi, 8.0, erretm) + 2.0 * rhoinv),
            );
            if w * prew > 0.0 && w.abs() > prew.abs() / 10.0 {
                swtch = !swtch;
            }
        }

        // Return with INFO = 1, NITER = MAXIT and not converged
        info = 1;
        if orgati {
            *dlam = v!(d, i) + tau;
        } else {
            *dlam = v!(d, ip1) + tau;
        }
        info
    }
}

/// `DLAED5`: the `i`-th eigenvalue (and normalized `delta`) of the 2-by-2
/// rank-one update `diag(d) + rho z z'`.
unsafe fn dlaed5(i: i32, d: *const f64, z: *const f64, delta: *mut f64, rho: f64, dlam: &mut f64) {
    let del = v!(d, 2) - v!(d, 1);
    let z1sq = v!(z, 1) * v!(z, 1);
    let z2sq = v!(z, 2) * v!(z, 2);
    let rz = rho * (z1sq + z2sq);
    if i == 1 {
        let w = 1.0 + 2.0 * rho * (z2sq - z1sq) / del;
        if w > 0.0 {
            let b = del + rz;
            let c = rho * v!(z, 1) * v!(z, 1) * del;

            // B > ZERO, always
            let tau = 2.0 * c / (b + fma(b, b, -(c * 4.0)).abs().sqrt());
            *dlam = v!(d, 1) + tau;
            v!(delta, 1) = -v!(z, 1) / tau;
            v!(delta, 2) = v!(z, 2) / (del - tau);
        } else {
            let b = -del + rz;
            let c = rho * v!(z, 2) * v!(z, 2) * del;
            let tau = if b > 0.0 {
                -2.0 * c / (b + fma(b, b, c * 4.0).sqrt())
            } else {
                (b - fma(b, b, c * 4.0).sqrt()) / 2.0
            };
            *dlam = v!(d, 2) + tau;
            v!(delta, 1) = -v!(z, 1) / (del + tau);
            v!(delta, 2) = -v!(z, 2) / tau;
        }
        let temp = fma(v!(delta, 1), v!(delta, 1), v!(delta, 2) * v!(delta, 2)).sqrt();
        v!(delta, 1) /= temp;
        v!(delta, 2) /= temp;
    } else {
        // Now I=2
        let b = -del + rz;
        let c = rho * v!(z, 2) * v!(z, 2) * del;
        let tau = if b > 0.0 {
            (b + fma(b, b, c * 4.0).sqrt()) / 2.0
        } else {
            2.0 * c / (-b + fma(b, b, c * 4.0).sqrt())
        };
        *dlam = v!(d, 2) + tau;
        v!(delta, 1) = -v!(z, 1) / (del + tau);
        v!(delta, 2) = -v!(z, 2) / tau;
        let temp = fma(v!(delta, 1), v!(delta, 1), v!(delta, 2) * v!(delta, 2)).sqrt();
        v!(delta, 1) /= temp;
        v!(delta, 2) /= temp;
    }
}

const MAXIT6: i32 = 40;

/// `DLAED6`: one root of the three-pole secular equation that `DLAED4`
/// interpolates with, into `tau`. Returns `INFO`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlaed6(
    kniter: i32,
    orgati: bool,
    rho: f64,
    d: *const f64,
    z: *const f64,
    finit: f64,
    tau: &mut f64,
) -> i32 {
    let mut info = 0;

    let (mut lbd, mut ubd) = if orgati {
        (v!(d, 2), v!(d, 3))
    } else {
        (v!(d, 1), v!(d, 2))
    };
    if finit < 0.0 {
        lbd = 0.0;
    } else {
        ubd = 0.0;
    }

    let niter = 1;
    *tau = 0.0;
    if kniter == 2 {
        let (mut a, mut b, mut c);
        if orgati {
            c = rho + v!(z, 1) / fma(-(v!(d, 3) - v!(d, 2)), 0.5, v!(d, 1) - v!(d, 2));
            a = fma(v!(d, 3) + v!(d, 2), c, v!(z, 2)) + v!(z, 3);
            b = fma(
                v!(z, 3),
                v!(d, 2),
                fma(c * v!(d, 2), v!(d, 3), v!(z, 2) * v!(d, 3)),
            );
        } else {
            c = rho + v!(z, 3) / fma(-(v!(d, 1) - v!(d, 2)), 0.5, v!(d, 3) - v!(d, 2));
            a = fma(v!(d, 1) + v!(d, 2), c, v!(z, 1)) + v!(z, 2);
            b = fma(
                v!(z, 2),
                v!(d, 1),
                fma(v!(d, 1) * c, v!(d, 2), v!(z, 1) * v!(d, 2)),
            );
        }
        let temp = a.abs().max(b.abs()).max(c.abs());
        a /= temp;
        b /= temp;
        c /= temp;
        if c == 0.0 {
            *tau = b / a;
        } else if a <= 0.0 {
            *tau = (a - fma(a, a, -(c * (b * 4.0))).abs().sqrt()) / (2.0 * c);
        } else {
            *tau = 2.0 * b / (a + fma(a, a, -(c * (b * 4.0))).abs().sqrt());
        }
        if *tau < lbd || *tau > ubd {
            *tau = (lbd + ubd) / 2.0;
        }
        if v!(d, 1) == *tau || v!(d, 2) == *tau || v!(d, 3) == *tau {
            *tau = 0.0;
        } else {
            let t = *tau;
            let temp = finit
                + t * v!(z, 1) / (v!(d, 1) * (v!(d, 1) - t))
                + t * v!(z, 2) / (v!(d, 2) * (v!(d, 2) - t))
                + t * v!(z, 3) / (v!(d, 3) * (v!(d, 3) - t));
            if temp <= 0.0 {
                lbd = t;
            } else {
                ubd = t;
            }
            if finit.abs() <= temp.abs() {
                *tau = 0.0;
            }
        }
    }

    // get machine parameters for possible scaling to avoid overflow
    //
    // modified by Sven: parameters SMALL1, SMINV1, SMALL2, SMINV2, EPS are
    // not SAVEd anymore between one call to the others but recomputed at
    // each call
    let eps = EPS;
    let base = 2.0f64;
    let small1 = base.powi(((SAFMIN.ln() / base.ln()) / 3.0) as i32);
    let sminv1 = 1.0 / small1;
    let small2 = small1 * small1;
    let sminv2 = sminv1 * sminv1;

    // Determine if scaling of inputs necessary to avoid overflow when
    // computing 1/TEMP**3
    let temp = if orgati {
        (v!(d, 2) - *tau).abs().min((v!(d, 3) - *tau).abs())
    } else {
        (v!(d, 1) - *tau).abs().min((v!(d, 2) - *tau).abs())
    };
    let mut scale = false;
    let mut sclinv = 0.0;
    let mut dscale = [0.0f64; 3];
    let mut zscale = [0.0f64; 3];
    if temp <= small1 {
        scale = true;
        let sclfac;
        if temp <= small2 {
            // Scale up by power of radix nearest 1/SAFMIN**(2/3)
            sclfac = sminv2;
            sclinv = small2;
        } else {
            // Scale up by power of radix nearest 1/SAFMIN**(1/3)
            sclfac = sminv1;
            sclinv = small1;
        }

        // Scaling up safe because D, Z, TAU scaled elsewhere to be O(1)
        for i in 0..3 {
            dscale[i] = v!(d, i as i32 + 1) * sclfac;
            zscale[i] = v!(z, i as i32 + 1) * sclfac;
        }
        *tau *= sclfac;
        lbd *= sclfac;
        ubd *= sclfac;
    } else {
        // Copy D and Z to DSCALE and ZSCALE
        for i in 0..3 {
            dscale[i] = v!(d, i as i32 + 1);
            zscale[i] = v!(z, i as i32 + 1);
        }
    }

    let mut fc = 0.0;
    let mut df = 0.0;
    let mut ddf = 0.0;
    for i in 0..3 {
        let temp = 1.0 / (dscale[i] - *tau);
        let temp1 = zscale[i] * temp;
        let temp2 = temp1 * temp;
        fc += temp1 / dscale[i];
        df += temp2;
        ddf = fma(temp, temp2, ddf);
    }
    let mut f = fma(fc, *tau, finit);

    'done: {
        if f.abs() <= 0.0 {
            break 'done;
        }
        if f <= 0.0 {
            lbd = *tau;
        } else {
            ubd = *tau;
        }

        // Iteration begins -- Use Gragg-Thornton-Warner cubic convergent
        // scheme
        //
        // It is not hard to see that
        //
        //       1) Iterations will go up monotonically
        //          if FINIT < 0;
        //
        //       2) Iterations will go down monotonically
        //          if FINIT > 0.
        let iter = niter + 1;

        for _niter in iter..=MAXIT6 {
            let (temp1, temp2) = if orgati {
                (dscale[1] - *tau, dscale[2] - *tau)
            } else {
                (dscale[0] - *tau, dscale[1] - *tau)
            };
            let t12 = temp1 * temp2;
            let mut a = fma(temp1 + temp2, f, -(t12 * df));
            let mut b = t12 * f;
            let mut c = fma(t12, ddf, fma(-(temp1 + temp2), df, f));
            let temp = a.abs().max(b.abs()).max(c.abs());
            a /= temp;
            b /= temp;
            c /= temp;
            let mut eta;
            if c == 0.0 {
                eta = b / a;
            } else if a <= 0.0 {
                eta = (a - fma(a, a, -(c * (b * 4.0))).abs().sqrt()) / (2.0 * c);
            } else {
                eta = 2.0 * b / (a + fma(a, a, -(c * (b * 4.0))).abs().sqrt());
            }
            if f * eta >= 0.0 {
                eta = -f / df;
            }

            *tau += eta;
            if *tau < lbd || *tau > ubd {
                *tau = (lbd + ubd) / 2.0;
            }

            fc = 0.0;
            let mut erretm = 0.0;
            df = 0.0;
            ddf = 0.0;
            for i in 0..3 {
                if dscale[i] - *tau != 0.0 {
                    let temp = 1.0 / (dscale[i] - *tau);
                    let temp1 = zscale[i] * temp;
                    let temp2 = temp1 * temp;
                    let temp4 = temp1 / dscale[i];
                    fc += temp4;
                    erretm += temp4.abs();
                    df += temp2;
                    ddf = fma(temp, temp2, ddf);
                } else {
                    break 'done;
                }
            }
            f = fma(fc, *tau, finit);
            erretm = fma(fma(erretm, tau.abs(), finit.abs()), 8.0, df * tau.abs());
            if f.abs() <= 4.0 * eps * erretm || (ubd - lbd) <= 4.0 * eps * tau.abs() {
                break 'done;
            }
            if f <= 0.0 {
                lbd = *tau;
            } else {
                ubd = *tau;
            }
        }
        info = 1;
    }

    // Undo scaling
    if scale {
        *tau *= sclinv;
    }
    info
}
