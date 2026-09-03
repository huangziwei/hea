use pyo3::prelude::*;

use super::linpack::{
    daxpy, daxpy_same, dcopy, dcopy_same, ddot, ddot_same, dpofa, dscal, dtrsl, dtrsl_same,
};
use crate::nmath::util::rfma;

const DBL_EPSILON: f64 = f64::EPSILON;

pub trait Objective {
    fn f(&mut self, x: &[f64]) -> PyResult<f64>;
    fn grad(&mut self, x: &[f64], g: &mut [f64]) -> PyResult<()>;
}

pub struct State {
    ws: Vec<f64>,
    wy: Vec<f64>,
    sy: Vec<f64>,
    ss: Vec<f64>,
    wt: Vec<f64>,
    wn: Vec<f64>,
    snd: Vec<f64>,
    z: Vec<f64>,
    r: Vec<f64>,
    d: Vec<f64>,
    t: Vec<f64>,
    wa: Vec<f64>,
    indx: Vec<usize>,
    iwhere: Vec<i32>,
    indx2: Vec<usize>,
    prjctd: bool,
    cnstnd: bool,
    boxed: bool,
    updatd: bool,
    iback: i32,
    head: usize,
    col: usize,
    itail: usize,
    iter: i32,
    iupdat: usize,
    nint: i32,
    nfgv: i32,
    info: i32,
    ifun: i32,
    iword: i32,
    nfree: usize,
    nact: usize,
    ileave: usize,
    nenter: usize,
    nintol: i32,
    nskip: i32,
    theta: f64,
    fold: f64,
    tol: f64,
    dnorm: f64,
    epsmch: f64,
    gd: f64,
    stpmx: f64,
    sbgnrm: f64,
    stp: f64,
    gdold: f64,
    dtd: f64,
    rr: f64,
    dr: f64,
    xstep: f64,
    csave: String,
    ls: Dcsrch,
    isave13: i32,
}

impl State {
    pub fn new(n: usize, m: usize) -> Self {
        State {
            ws: vec![0.0; n * m],
            wy: vec![0.0; n * m],
            sy: vec![0.0; m * m],
            ss: vec![0.0; m * m],
            wt: vec![0.0; m * m],
            wn: vec![0.0; 4 * m * m],
            snd: vec![0.0; 4 * m * m],
            z: vec![0.0; n],
            r: vec![0.0; n],
            d: vec![0.0; n],
            t: vec![0.0; n],
            wa: vec![0.0; 8 * m],
            indx: vec![0; n],
            iwhere: vec![0; n],
            indx2: vec![0; n],
            prjctd: false,
            cnstnd: false,
            boxed: false,
            updatd: false,
            iback: 0,
            head: 1,
            col: 0,
            itail: 0,
            iter: 0,
            iupdat: 0,
            nint: 0,
            nfgv: 0,
            info: 0,
            ifun: 0,
            iword: 0,
            nfree: n,
            nact: 0,
            ileave: 0,
            nenter: 0,
            nintol: 0,
            nskip: 0,
            theta: 1.0,
            fold: 0.0,
            tol: 0.0,
            dnorm: 0.0,
            epsmch: DBL_EPSILON,
            gd: 0.0,
            stpmx: 0.0,
            sbgnrm: 0.0,
            stp: 0.0,
            gdold: 0.0,
            dtd: 0.0,
            rr: 0.0,
            dr: 0.0,
            xstep: 0.0,
            csave: String::new(),
            ls: Dcsrch::default(),
            isave13: 0,
        }
    }
}

fn active(
    n: usize,
    lo: &[f64],
    up: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    iwhere: &mut [i32],
    st: &mut State,
) {
    st.prjctd = false;
    st.cnstnd = false;
    st.boxed = true;
    for i in 0..n {
        if nbd[i] > 0 {
            if nbd[i] <= 2 && x[i] <= lo[i] {
                if x[i] < lo[i] {
                    st.prjctd = true;
                    x[i] = lo[i];
                }
            } else if nbd[i] >= 2 && x[i] >= up[i] && x[i] > up[i] {
                st.prjctd = true;
                x[i] = up[i];
            }
        }
    }
    for i in 0..n {
        if nbd[i] != 2 {
            st.boxed = false;
        }
        if nbd[i] == 0 {
            iwhere[i] = -1;
        } else {
            st.cnstnd = true;
            if nbd[i] == 2 && up[i] - lo[i] <= 0.0 {
                iwhere[i] = 3;
            } else {
                iwhere[i] = 0;
            }
        }
    }
}

fn bmv(m: usize, sy: &[f64], wt: &[f64], col: usize, wa: &mut [f64], ov: usize, op: usize) -> i32 {
    if col == 0 {
        return 0;
    }
    wa[op + col] = wa[ov + col];
    for i in 2..=col {
        let i2 = col + i;
        let mut s = 0.0;
        for k in 1..(i) {
            s += sy[(i - 1) + (k - 1) * m] * wa[ov + k - 1] / sy[(k - 1) + (k - 1) * m];
        }
        wa[op + i2 - 1] = wa[ov + i2 - 1] + s;
    }
    {
        let (wt_ro, b) = (wt, &mut wa[op + col..op + 2 * col]);
        let info = dtrsl(wt_ro, 0, m, col, b, 0, 1, 11);
        if info != 0 {
            return info;
        }
    }
    for i in 1..=col {
        wa[op + i - 1] = wa[ov + i - 1] / sy[(i - 1) + (i - 1) * m].sqrt();
    }
    {
        let b = &mut wa[op + col..op + 2 * col];
        let info = dtrsl(wt, 0, m, col, b, 0, 1, 1);
        if info != 0 {
            return info;
        }
    }
    for i in 1..=col {
        wa[op + i - 1] = -wa[op + i - 1] / sy[(i - 1) + (i - 1) * m].sqrt();
    }
    for i in 1..=col {
        let mut s = 0.0;
        for k in (i + 1)..=col {
            s += sy[(k - 1) + (i - 1) * m] * wa[op + col + k - 1] / sy[(i - 1) + (i - 1) * m];
        }
        wa[op + i - 1] += s;
    }
    0
}

fn hpsolb(n: usize, t: &mut [f64], iorder: &mut [usize], iheap: i32) {
    if iheap == 0 {
        for k in 2..=n {
            let ddum = t[k - 1];
            let indxin = iorder[k - 1];
            let mut i = k;
            while i > 1 {
                let j = i / 2;
                if ddum < t[j - 1] {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                } else {
                    break;
                }
            }
            t[i - 1] = ddum;
            iorder[i - 1] = indxin;
        }
    }
    if n > 1 {
        let mut i = 1;
        let out = t[0];
        let indxou = iorder[0];
        let ddum = t[n - 1];
        let indxin = iorder[n - 1];
        loop {
            let mut j = i + i;
            if j <= n - 1 {
                if t[j] < t[j - 1] {
                    j += 1;
                }
                if t[j - 1] < ddum {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                    continue;
                }
            }
            break;
        }
        t[i - 1] = ddum;
        iorder[i - 1] = indxin;
        t[n - 1] = out;
        iorder[n - 1] = indxou;
    }
}

#[allow(clippy::too_many_arguments)]
fn cauchy(
    n: usize,
    x: &[f64],
    lo: &[f64],
    up: &[f64],
    nbd: &[i32],
    g: &[f64],
    iorder: &mut [usize],
    iwhere: &mut [i32],
    t: &mut [f64],
    d: &mut [f64],
    xcp: &mut [f64],
    m: usize,
    wy: &[f64],
    ws: &[f64],
    sy: &[f64],
    wt: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    wa: &mut [f64],
    sbgnrm: f64,
    epsmch: f64,
) -> (i32, i32) {
    let (op, oc, owbp, ov) = (0usize, 2 * m, 4 * m, 6 * m);
    if sbgnrm <= 0.0 {
        dcopy(n, x, 0, xcp, 0);
        return (0, 0);
    }
    let mut bnded = true;
    let mut nfree = n + 1;
    let mut nbreak = 0usize;
    let mut ibkmin = 0usize;
    let mut bkmin = 0.0;
    let col2 = 2 * col;
    let mut f1 = 0.0;
    for i in 0..col2 {
        wa[op + i] = 0.0;
    }
    let mut tl = 0.0;
    let mut tu = 0.0;
    for i in 1..=n {
        let neggi = -g[i - 1];
        if iwhere[i - 1] != 3 && iwhere[i - 1] != -1 {
            if nbd[i - 1] <= 2 {
                tl = x[i - 1] - lo[i - 1];
            }
            if nbd[i - 1] >= 2 {
                tu = up[i - 1] - x[i - 1];
            }
            let xlower = nbd[i - 1] <= 2 && tl <= 0.0;
            let xupper = nbd[i - 1] >= 2 && tu <= 0.0;
            iwhere[i - 1] = 0;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i - 1] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i - 1] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i - 1] = -3;
            }
        }
        let mut pointr = head;
        if iwhere[i - 1] != 0 && iwhere[i - 1] != -1 {
            d[i - 1] = 0.0;
        } else {
            d[i - 1] = neggi;
            f1 = rfma(-neggi, neggi, f1);
            for j in 1..=col {
                wa[op + j - 1] = rfma(wy[(i - 1) + (pointr - 1) * n], neggi, wa[op + j - 1]);
                wa[op + col + j - 1] =
                    rfma(ws[(i - 1) + (pointr - 1) * n], neggi, wa[op + col + j - 1]);
                pointr = pointr % m + 1;
            }
            if nbd[i - 1] <= 2 && nbd[i - 1] != 0 && neggi < 0.0 {
                nbreak += 1;
                iorder[nbreak - 1] = i;
                t[nbreak - 1] = tl / (-neggi);
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else if nbd[i - 1] >= 2 && neggi > 0.0 {
                nbreak += 1;
                iorder[nbreak - 1] = i;
                t[nbreak - 1] = tu / neggi;
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else {
                nfree -= 1;
                iorder[nfree - 1] = i;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }
    if theta != 1.0 {
        dscal(col, theta, wa, op + col);
    }
    dcopy(n, x, 0, xcp, 0);
    if nbreak == 0 && nfree == n + 1 {
        return (0, 0);
    }
    for j in 0..col2 {
        wa[oc + j] = 0.0;
    }
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        let info = bmv(m, sy, wt, col, wa, op, ov);
        if info != 0 {
            return (0, info);
        }
        f2 -= ddot(col2, wa, ov, 1, wa, op, 1);
    }
    let mut dtm = -f1 / f2;
    let mut tsum = 0.0;
    let mut nint = 1;
    if nbreak != 0 {
        let mut nleft = nbreak;
        let mut iter = 1;
        let mut tj = 0.0;
        loop {
            let tj0 = tj;
            let ibp;
            if iter == 1 {
                tj = bkmin;
                ibp = iorder[ibkmin - 1];
            } else {
                if iter == 2 && ibkmin != nbreak {
                    t[ibkmin - 1] = t[nbreak - 1];
                    iorder[ibkmin - 1] = iorder[nbreak - 1];
                }
                hpsolb(nleft, t, iorder, iter - 2);
                tj = t[nleft - 1];
                ibp = iorder[nleft - 1];
            }
            let dt = tj - tj0;
            if dtm < dt {
                break;
            }
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let dibp = d[ibp - 1];
            d[ibp - 1] = 0.0;
            let zibp;
            if dibp > 0.0 {
                zibp = up[ibp - 1] - x[ibp - 1];
                xcp[ibp - 1] = up[ibp - 1];
                iwhere[ibp - 1] = 2;
            } else {
                zibp = lo[ibp - 1] - x[ibp - 1];
                xcp[ibp - 1] = lo[ibp - 1];
                iwhere[ibp - 1] = 1;
            }
            if nleft == 0 && nbreak == n {
                dtm = dt;
                if col > 0 {
                    daxpy_same(col2, dtm, wa, op, oc);
                }
                return (nint, 0);
            }
            nint += 1;
            let dibp2 = dibp * dibp;
            f1 += rfma(-(theta * dibp), zibp, rfma(dt, f2, dibp2));
            f2 = rfma(-theta, dibp2, f2);
            if col > 0 {
                daxpy_same(col2, dt, wa, op, oc);
                let mut pointr = head;
                for j in 1..=col {
                    wa[owbp + j - 1] = wy[(ibp - 1) + (pointr - 1) * n];
                    wa[owbp + col + j - 1] = theta * ws[(ibp - 1) + (pointr - 1) * n];
                    pointr = pointr % m + 1;
                }
                let info = bmv(m, sy, wt, col, wa, owbp, ov);
                if info != 0 {
                    return (nint, info);
                }
                let wmc = ddot(col2, wa, oc, 1, wa, ov, 1);
                let wmp = ddot(col2, wa, op, 1, wa, ov, 1);
                let wmw = ddot(col2, wa, owbp, 1, wa, ov, 1);
                daxpy_same(col2, -dibp, wa, owbp, op);
                f1 = rfma(dibp, wmc, f1);
                f2 += rfma(2.0 * dibp, wmp, -(dibp2 * wmw));
            }
            if f2 < epsmch * f2_org {
                f2 = epsmch * f2_org;
            }
            if nleft > 0 {
                dtm = -f1 / f2;
                continue;
            } else if bnded {
                f1 = 0.0;
                f2 = 0.0;
                dtm = 0.0;
            } else {
                dtm = -f1 / f2;
            }
            break;
        }
    }
    if dtm <= 0.0 {
        dtm = 0.0;
    }
    tsum += dtm;
    daxpy(n, tsum, d, 0, 1, xcp, 0);
    if col > 0 {
        daxpy_same(col2, dtm, wa, op, oc);
    }
    (nint, 0)
}

#[allow(clippy::too_many_arguments)]
fn cmprlb(
    n: usize,
    m: usize,
    x: &[f64],
    g: &[f64],
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    wt: &[f64],
    z: &[f64],
    r: &mut [f64],
    wa: &mut [f64],
    indx: &[usize],
    theta: f64,
    col: usize,
    head: usize,
    nfree: usize,
    cnstnd: bool,
) -> i32 {
    if !cnstnd && col > 0 {
        for i in 0..n {
            r[i] = -g[i];
        }
        return 0;
    }
    for i in 1..=nfree {
        let k = indx[i - 1];
        r[i - 1] = rfma(-theta, z[k - 1] - x[k - 1], -g[k - 1]);
    }
    let info = bmv(m, sy, wt, col, wa, 2 * m, 0);
    if info != 0 {
        return -8;
    }
    let mut pointr = head;
    for j in 1..=col {
        let a1 = wa[j - 1];
        let a2 = theta * wa[col + j - 1];
        for i in 1..=nfree {
            let k = indx[i - 1];
            r[i - 1] += rfma(
                wy[(k - 1) + (pointr - 1) * n],
                a1,
                ws[(k - 1) + (pointr - 1) * n] * a2,
            );
        }
        pointr = pointr % m + 1;
    }
    0
}

fn errclb(n: usize, m: usize, factr: f64, lo: &[f64], up: &[f64], nbd: &[i32]) -> (String, i32) {
    let mut task = String::new();
    let mut info = 0;
    if n == 0 {
        task = "ERROR: N .LE. 0".into();
    }
    if m == 0 {
        task = "ERROR: M .LE. 0".into();
    }
    if factr < 0.0 {
        task = "ERROR: FACTR .LT. 0".into();
    }
    for i in 1..=n {
        if nbd[i - 1] < 0 || nbd[i - 1] > 3 {
            task = "ERROR: INVALID NBD".into();
            info = -6;
        }
        if nbd[i - 1] == 2 && lo[i - 1] > up[i - 1] {
            task = "ERROR: NO FEASIBLE SOLUTION".into();
            info = -7;
        }
    }
    (task, info)
}

#[allow(clippy::too_many_arguments)]
fn formk(
    n: usize,
    nsub: usize,
    ind: &[usize],
    nenter: usize,
    ileave: usize,
    indx2: &[usize],
    iupdat: usize,
    updatd: bool,
    wn: &mut [f64],
    wn1: &mut [f64],
    m: usize,
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    theta: f64,
    col: usize,
    head: usize,
) -> i32 {
    let m2 = 2 * m;
    let upcl;
    if updatd {
        if iupdat > m {
            for jy in 1..m {
                let js = m + jy;
                let i2 = m - jy;
                dcopy_same(i2, wn1, jy + jy * m2, (jy - 1) + (jy - 1) * m2);
                dcopy_same(i2, wn1, js + js * m2, (js - 1) + (js - 1) * m2);
                dcopy_same(m - 1, wn1, (m + 1) + jy * m2, m + (jy - 1) * m2);
            }
        }
        let pbegin = 1;
        let pend = nsub;
        let dbegin = nsub + 1;
        let dend = n;
        let iy = col;
        let mut is_ = m + col;
        let mut ipntr = head + col - 1;
        if ipntr > m {
            ipntr -= m;
        }
        let mut jpntr = head;
        for jy in 1..=col {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            for k in pbegin..=pend {
                let k1 = ind[k - 1];
                temp1 = rfma(
                    wy[(k1 - 1) + (ipntr - 1) * n],
                    wy[(k1 - 1) + (jpntr - 1) * n],
                    temp1,
                );
            }
            for k in dbegin..=dend {
                let k1 = ind[k - 1];
                temp2 = rfma(
                    ws[(k1 - 1) + (ipntr - 1) * n],
                    ws[(k1 - 1) + (jpntr - 1) * n],
                    temp2,
                );
                temp3 = rfma(
                    ws[(k1 - 1) + (ipntr - 1) * n],
                    wy[(k1 - 1) + (jpntr - 1) * n],
                    temp3,
                );
            }
            wn1[(iy - 1) + (jy - 1) * m2] = temp1;
            wn1[(is_ - 1) + (js - 1) * m2] = temp2;
            wn1[(is_ - 1) + (jy - 1) * m2] = temp3;
            jpntr = jpntr % m + 1;
        }
        let jy = col;
        let mut jpntr = head + col - 1;
        if jpntr > m {
            jpntr -= m;
        }
        let mut ipntr = head;
        for i in 1..=col {
            is_ = m + i;
            let mut temp3 = 0.0;
            for k in pbegin..=pend {
                let k1 = ind[k - 1];
                temp3 = rfma(
                    ws[(k1 - 1) + (ipntr - 1) * n],
                    wy[(k1 - 1) + (jpntr - 1) * n],
                    temp3,
                );
            }
            ipntr = ipntr % m + 1;
            wn1[(is_ - 1) + (jy - 1) * m2] = temp3;
        }
        upcl = col - 1;
    } else {
        upcl = col;
    }
    let mut ipntr = head;
    for iy in 1..=upcl {
        let is_ = m + iy;
        let mut jpntr = head;
        for jy in 1..=iy {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            let mut temp4 = 0.0;
            for k in 1..=nenter {
                let k1 = indx2[k - 1];
                temp1 = rfma(
                    wy[(k1 - 1) + (ipntr - 1) * n],
                    wy[(k1 - 1) + (jpntr - 1) * n],
                    temp1,
                );
                temp2 = rfma(
                    ws[(k1 - 1) + (ipntr - 1) * n],
                    ws[(k1 - 1) + (jpntr - 1) * n],
                    temp2,
                );
            }
            for k in ileave..=n {
                let k1 = indx2[k - 1];
                temp3 = rfma(
                    wy[(k1 - 1) + (ipntr - 1) * n],
                    wy[(k1 - 1) + (jpntr - 1) * n],
                    temp3,
                );
                temp4 = rfma(
                    ws[(k1 - 1) + (ipntr - 1) * n],
                    ws[(k1 - 1) + (jpntr - 1) * n],
                    temp4,
                );
            }
            wn1[(iy - 1) + (jy - 1) * m2] = wn1[(iy - 1) + (jy - 1) * m2] + temp1 - temp3;
            wn1[(is_ - 1) + (js - 1) * m2] = wn1[(is_ - 1) + (js - 1) * m2] - temp2 + temp4;
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    let mut ipntr = head;
    for is_ in (m + 1)..=(m + upcl) {
        let mut jpntr = head;
        for jy in 1..=upcl {
            let mut temp1 = 0.0;
            let mut temp3 = 0.0;
            for k in 1..=nenter {
                let k1 = indx2[k - 1];
                temp1 = rfma(
                    ws[(k1 - 1) + (ipntr - 1) * n],
                    wy[(k1 - 1) + (jpntr - 1) * n],
                    temp1,
                );
            }
            for k in ileave..=n {
                let k1 = indx2[k - 1];
                temp3 = rfma(
                    ws[(k1 - 1) + (ipntr - 1) * n],
                    wy[(k1 - 1) + (jpntr - 1) * n],
                    temp3,
                );
            }
            if is_ <= jy + m {
                wn1[(is_ - 1) + (jy - 1) * m2] += temp1 - temp3;
            } else {
                wn1[(is_ - 1) + (jy - 1) * m2] += -temp1 + temp3;
            }
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    for iy in 1..=col {
        let is_ = col + iy;
        let is1 = m + iy;
        for jy in 1..=iy {
            let js = col + jy;
            let js1 = m + jy;
            wn[(jy - 1) + (iy - 1) * m2] = wn1[(iy - 1) + (jy - 1) * m2] / theta;
            wn[(js - 1) + (is_ - 1) * m2] = wn1[(is1 - 1) + (js1 - 1) * m2] * theta;
        }
        for jy in 1..iy {
            wn[(jy - 1) + (is_ - 1) * m2] = -wn1[(is1 - 1) + (jy - 1) * m2];
        }
        for jy in iy..=col {
            wn[(jy - 1) + (is_ - 1) * m2] = wn1[(is1 - 1) + (jy - 1) * m2];
        }
        wn[(iy - 1) + (iy - 1) * m2] += sy[(iy - 1) + (iy - 1) * m];
    }
    let info = dpofa(wn, 0, m2, col);
    if info != 0 {
        return -1;
    }
    let col2 = 2 * col;
    for js in (col + 1)..=col2 {
        dtrsl_same(wn, 0, m2, col, (js - 1) * m2, 11);
    }
    for is_ in (col + 1)..=col2 {
        for js in is_..=col2 {
            wn[(is_ - 1) + (js - 1) * m2] += ddot_same(col, wn, (is_ - 1) * m2, (js - 1) * m2);
        }
    }
    let info = dpofa(wn, col + col * m2, m2, col);
    if info != 0 {
        return -2;
    }
    0
}

fn formt(m: usize, wt: &mut [f64], sy: &[f64], ss: &[f64], col: usize, theta: f64) -> i32 {
    for j in 1..=col {
        wt[(j - 1) * m] = theta * ss[(j - 1) * m];
    }
    for i in 2..=col {
        for j in i..=col {
            let k1 = i.min(j) - 1;
            let mut ddum = 0.0;
            for k in 1..=k1 {
                ddum += sy[(i - 1) + (k - 1) * m] * sy[(j - 1) + (k - 1) * m]
                    / sy[(k - 1) + (k - 1) * m];
            }
            wt[(i - 1) + (j - 1) * m] = rfma(theta, ss[(i - 1) + (j - 1) * m], ddum);
        }
    }
    let info = dpofa(wt, 0, m, col);
    if info != 0 {
        return -3;
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn freev(
    n: usize,
    mut nfree: usize,
    indx: &mut [usize],
    indx2: &mut [usize],
    iwhere: &[i32],
    updatd: bool,
    cnstnd: bool,
    iter: i32,
) -> (usize, usize, usize, bool) {
    let mut nenter = 0usize;
    let mut ileave = n + 1;
    if iter > 0 && cnstnd {
        for i in 1..=nfree {
            let k = indx[i - 1];
            if iwhere[k - 1] > 0 {
                ileave -= 1;
                indx2[ileave - 1] = k;
            }
        }
        for i in (nfree + 1)..=n {
            let k = indx[i - 1];
            if iwhere[k - 1] <= 0 {
                nenter += 1;
                indx2[nenter - 1] = k;
            }
        }
    }
    let wrk = (ileave < n + 1) || (nenter > 0) || updatd;
    nfree = 0;
    let mut iact = n + 1;
    for i in 1..=n {
        if iwhere[i - 1] <= 0 {
            nfree += 1;
            indx[nfree - 1] = i;
        } else {
            iact -= 1;
            indx[iact - 1] = i;
        }
    }
    (nfree, nenter, ileave, wrk)
}

#[derive(Default)]
pub struct Dcsrch {
    brackt: bool,
    stage: i32,
    finit: f64,
    ginit: f64,
    gtest: f64,
    width: f64,
    width1: f64,
    stx: f64,
    fx: f64,
    gx: f64,
    sty: f64,
    fy: f64,
    gy: f64,
    stmin: f64,
    stmax: f64,
}

#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let d1 = theta / s;
        let mut gamm = s * rfma(d1, d1, -(*dx / s * (dp / s))).sqrt();
        if *stp < *stx {
            gamm = -gamm;
        }
        let p = gamm - *dx + theta;
        let q = gamm - *dx + gamm + dp;
        let r = p / q;
        let stpc = rfma(r, *stp - *stx, *stx);
        let stpq = rfma(
            *dx / ((*fx - fp) / (*stp - *stx) + *dx) / 2.0,
            *stp - *stx,
            *stx,
        );
        if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / 2.0;
        }
        *brackt = true;
    } else if sgnd < 0.0 {
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let d1 = theta / s;
        let mut gamm = s * rfma(d1, d1, -(*dx / s * (dp / s))).sqrt();
        if *stp > *stx {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm - dp + gamm + *dx;
        let r = p / q;
        let stpc = rfma(r, *stx - *stp, *stp);
        let stpq = rfma(dp / (dp - *dx), *stx - *stp, *stp);
        if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let t1 = theta / s;
        let d1 = rfma(t1, t1, -(*dx / s * (dp / s)));
        let gamm0 = if d1 < 0.0 { 0.0 } else { s * d1.sqrt() };
        let gamm = if *stp > *stx { -gamm0 } else { gamm0 };
        let p = gamm - dp + theta;
        let q = gamm + (*dx - dp) + gamm;
        let r = p / q;
        let stpc = if r < 0.0 && gamm != 0.0 {
            rfma(r, *stx - *stp, *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = rfma(dp / (dp - *dx), *stx - *stp, *stp);
        if *brackt {
            let mut f = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            let d1 = rfma(*sty - *stp, 0.66, *stp);
            if *stp > *stx {
                f = d1.min(f);
            } else {
                f = d1.max(f);
            }
            stpf = f;
        } else {
            let mut f = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            f = stpmax.min(f);
            f = stpmin.max(f);
            stpf = f;
        }
    } else if *brackt {
        let theta = (fp - *fy) * 3.0 / (*sty - *stp) + *dy + dp;
        let s = theta.abs().max(dy.abs()).max(dp.abs());
        let d1 = theta / s;
        let mut gamm = s * rfma(d1, d1, -(*dy / s * (dp / s))).sqrt();
        if *stp > *sty {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm - dp + gamm + *dy;
        let r = p / q;
        stpf = rfma(r, *sty - *stp, *stp);
    } else if *stp > *stx {
        stpf = stpmax;
    } else {
        stpf = stpmin;
    }
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

#[allow(clippy::too_many_arguments)]
fn dcsrch(
    f: f64,
    g: f64,
    stp: &mut f64,
    ftol: f64,
    gtol: f64,
    xtol: f64,
    stpmin: f64,
    stpmax: f64,
    task: &str,
    ls: &mut Dcsrch,
) -> String {
    if task.starts_with("START") {
        if *stp < stpmin {
            return "ERROR: STP .LT. STPMIN".into();
        }
        if *stp > stpmax {
            return "ERROR: STP .GT. STPMAX".into();
        }
        if g >= 0.0 {
            return "ERROR: INITIAL G .GE. ZERO".into();
        }
        if ftol < 0.0 {
            return "ERROR: FTOL .LT. ZERO".into();
        }
        if gtol < 0.0 {
            return "ERROR: GTOL .LT. ZERO".into();
        }
        if xtol < 0.0 {
            return "ERROR: XTOL .LT. ZERO".into();
        }
        if stpmin < 0.0 {
            return "ERROR: STPMIN .LT. ZERO".into();
        }
        if stpmax < stpmin {
            return "ERROR: STPMAX .LT. STPMIN".into();
        }
        ls.brackt = false;
        ls.stage = 1;
        ls.finit = f;
        ls.ginit = g;
        ls.gtest = ftol * g;
        ls.width = stpmax - stpmin;
        ls.width1 = (stpmax - stpmin) / 0.5;
        ls.stx = 0.0;
        ls.fx = f;
        ls.gx = g;
        ls.sty = 0.0;
        ls.fy = f;
        ls.gy = g;
        ls.stmin = 0.0;
        ls.stmax = *stp + *stp * 4.0;
        return "FG".into();
    }
    let mut task_out = task.to_string();
    let ftest = rfma(*stp, ls.gtest, ls.finit);
    if ls.stage == 1 && f <= ftest && g >= 0.0 {
        ls.stage = 2;
    }
    if ls.brackt && (*stp <= ls.stmin || *stp >= ls.stmax) {
        task_out = "WARNING: ROUNDING ERRORS PREVENT PROGRESS".into();
    }
    if ls.brackt && ls.stmax - ls.stmin <= xtol * ls.stmax {
        task_out = "WARNING: XTOL TEST SATISFIED".into();
    }
    if *stp == stpmax && f <= ftest && g <= ls.gtest {
        task_out = "WARNING: STP = STPMAX".into();
    }
    if *stp == stpmin && (f > ftest || g >= ls.gtest) {
        task_out = "WARNING: STP = STPMIN".into();
    }
    if f <= ftest && g.abs() <= gtol * (-ls.ginit) {
        task_out = "CONVERGENCE".into();
    }
    if task_out.starts_with("WARN") || task_out.starts_with("CONV") {
        return task_out;
    }
    if ls.stage == 1 && f <= ls.fx && f > ftest {
        let fm = rfma(-*stp, ls.gtest, f);
        let mut fxm = rfma(-ls.stx, ls.gtest, ls.fx);
        let mut fym = rfma(-ls.sty, ls.gtest, ls.fy);
        let gm = g - ls.gtest;
        let mut gxm = ls.gx - ls.gtest;
        let mut gym = ls.gy - ls.gtest;
        dcstep(
            &mut ls.stx,
            &mut fxm,
            &mut gxm,
            &mut ls.sty,
            &mut fym,
            &mut gym,
            stp,
            fm,
            gm,
            &mut ls.brackt,
            ls.stmin,
            ls.stmax,
        );
        ls.fx = rfma(ls.stx, ls.gtest, fxm);
        ls.fy = rfma(ls.sty, ls.gtest, fym);
        ls.gx = gxm + ls.gtest;
        ls.gy = gym + ls.gtest;
    } else {
        let (mut fx, mut gx, mut fy, mut gy) = (ls.fx, ls.gx, ls.fy, ls.gy);
        dcstep(
            &mut ls.stx,
            &mut fx,
            &mut gx,
            &mut ls.sty,
            &mut fy,
            &mut gy,
            stp,
            f,
            g,
            &mut ls.brackt,
            ls.stmin,
            ls.stmax,
        );
        ls.fx = fx;
        ls.gx = gx;
        ls.fy = fy;
        ls.gy = gy;
    }
    if ls.brackt {
        if (ls.sty - ls.stx).abs() >= ls.width1 * 0.66 {
            *stp = ls.stx + (ls.sty - ls.stx) * 0.5;
        }
        ls.width1 = ls.width;
        ls.width = (ls.sty - ls.stx).abs();
    }
    if ls.brackt {
        ls.stmin = ls.stx.min(ls.sty);
        ls.stmax = ls.stx.max(ls.sty);
    } else {
        ls.stmin = rfma(*stp - ls.stx, 1.1, *stp);
        ls.stmax = rfma(*stp - ls.stx, 4.0, *stp);
    }
    if *stp < stpmin {
        *stp = stpmin;
    }
    if *stp > stpmax {
        *stp = stpmax;
    }
    if (ls.brackt && (*stp <= ls.stmin || *stp >= ls.stmax))
        || (ls.brackt && ls.stmax - ls.stmin <= xtol * ls.stmax)
    {
        *stp = ls.stx;
    }
    "FG".into()
}

#[allow(clippy::too_many_arguments)]
fn lnsrlb(
    n: usize,
    lo: &[f64],
    up: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    f: f64,
    g: &[f64],
    d: &[f64],
    r: &mut [f64],
    t: &mut [f64],
    z: &[f64],
    task: &str,
    st: &mut State,
) -> (f64, String) {
    let ftol = 0.001;
    let gtol = 0.9;
    let xtol = 0.1;
    let stpmin = 0.0;
    let mut task_out = task.to_string();
    if !task.starts_with("FG_LN") {
        st.dtd = ddot(n, d, 0, 1, d, 0, 1);
        st.dnorm = st.dtd.sqrt();
        st.stpmx = 1e10;
        if st.cnstnd {
            if st.iter == 0 {
                st.stpmx = 1.0;
            } else {
                for i in 1..=n {
                    let a1 = d[i - 1];
                    if nbd[i - 1] != 0 {
                        if a1 < 0.0 && nbd[i - 1] <= 2 {
                            let a2 = lo[i - 1] - x[i - 1];
                            if a2 >= 0.0 {
                                st.stpmx = 0.0;
                            } else if a1 * st.stpmx < a2 {
                                st.stpmx = a2 / a1;
                            }
                        } else if a1 > 0.0 && nbd[i - 1] >= 2 {
                            let a2 = up[i - 1] - x[i - 1];
                            if a2 <= 0.0 {
                                st.stpmx = 0.0;
                            } else if a1 * st.stpmx > a2 {
                                st.stpmx = a2 / a1;
                            }
                        }
                    }
                }
            }
        }
        if st.iter == 0 && !st.boxed {
            st.stp = (1.0 / st.dnorm).min(st.stpmx);
        } else {
            st.stp = 1.0;
        }
        dcopy(n, x, 0, t, 0);
        dcopy(n, g, 0, r, 0);
        st.fold = f;
        st.ifun = 0;
        st.iback = 0;
        st.csave = "START".into();
    }
    st.gd = ddot(n, g, 0, 1, d, 0, 1);
    if st.ifun == 0 {
        st.gdold = st.gd;
        if st.gd >= 0.0 {
            st.info = -4;
            return (f, task_out);
        }
    }
    let csave_in = st.csave.clone();
    let mut stp = st.stp;
    st.csave = dcsrch(
        f, st.gd, &mut stp, ftol, gtol, xtol, stpmin, st.stpmx, &csave_in, &mut st.ls,
    );
    st.stp = stp;
    st.xstep = st.stp * st.dnorm;
    if !st.csave.starts_with("CONV") && !st.csave.starts_with("WARN") {
        task_out = "FG_LNSRCH".into();
        st.ifun += 1;
        st.nfgv += 1;
        st.iback = st.ifun - 1;
        if st.stp == 1.0 {
            dcopy(n, z, 0, x, 0);
        } else {
            for i in 1..=n {
                x[i - 1] = rfma(st.stp, d[i - 1], t[i - 1]);
            }
        }
    } else {
        task_out = "NEW_X".into();
    }
    (f, task_out)
}

#[allow(clippy::too_many_arguments)]
fn matupd(
    n: usize,
    m: usize,
    ws: &mut [f64],
    wy: &mut [f64],
    sy: &mut [f64],
    ss: &mut [f64],
    d: &[f64],
    r: &[f64],
    st: &mut State,
) {
    if st.iupdat <= m {
        st.col = st.iupdat;
        st.itail = (st.head + st.iupdat - 2) % m + 1;
    } else {
        st.itail = st.itail % m + 1;
        st.head = st.head % m + 1;
    }
    dcopy(n, d, 0, ws, (st.itail - 1) * n);
    dcopy(n, r, 0, wy, (st.itail - 1) * n);
    st.theta = st.rr / st.dr;
    if st.iupdat > m {
        for j in 1..st.col {
            dcopy_same(j, ss, 1 + j * m, (j - 1) * m);
            let i2 = st.col - j;
            dcopy_same(i2, sy, j + j * m, (j - 1) + (j - 1) * m);
        }
    }
    let mut pointr = st.head;
    let col = st.col;
    for j in 1..col {
        sy[(col - 1) + (j - 1) * m] = ddot(n, d, 0, 1, wy, (pointr - 1) * n, 1);
        ss[(j - 1) + (col - 1) * m] = ddot(n, ws, (pointr - 1) * n, 1, d, 0, 1);
        pointr = pointr % m + 1;
    }
    if st.stp == 1.0 {
        ss[(col - 1) + (col - 1) * m] = st.dtd;
    } else {
        ss[(col - 1) + (col - 1) * m] = st.stp * st.stp * st.dtd;
    }
    sy[(col - 1) + (col - 1) * m] = st.dr;
}

fn projgr(n: usize, lo: &[f64], up: &[f64], nbd: &[i32], x: &[f64], g: &[f64]) -> f64 {
    let mut sbgnrm = 0.0;
    for i in 0..n {
        let mut gi = g[i];
        if nbd[i] != 0 {
            if gi < 0.0 {
                if nbd[i] >= 2 {
                    let d1 = x[i] - up[i];
                    if gi < d1 {
                        gi = d1;
                    }
                }
            } else if nbd[i] <= 2 {
                let d1 = x[i] - lo[i];
                if gi > d1 {
                    gi = d1;
                }
            }
        }
        if sbgnrm < gi.abs() {
            sbgnrm = gi.abs();
        }
    }
    sbgnrm
}

#[allow(clippy::too_many_arguments)]
fn subsm(
    n: usize,
    m: usize,
    nsub: usize,
    ind: &[usize],
    lo: &[f64],
    up: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    d: &mut [f64],
    ws: &[f64],
    wy: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    wv: &mut [f64],
    wn: &[f64],
) -> (i32, i32) {
    if nsub == 0 {
        return (0, 0);
    }
    let m2 = 2 * m;
    let mut pointr = head;
    for i in 1..=col {
        let mut temp1 = 0.0;
        let mut temp2 = 0.0;
        for j in 1..=nsub {
            let k = ind[j - 1];
            temp1 = rfma(wy[(k - 1) + (pointr - 1) * n], d[j - 1], temp1);
            temp2 = rfma(ws[(k - 1) + (pointr - 1) * n], d[j - 1], temp2);
        }
        wv[i - 1] = temp1;
        wv[col + i - 1] = theta * temp2;
        pointr = pointr % m + 1;
    }
    let col2 = 2 * col;
    let info = dtrsl(wn, 0, m2, col2, wv, 0, 1, 11);
    if info != 0 {
        return (0, info);
    }
    for i in 1..=col {
        wv[i - 1] = -wv[i - 1];
    }
    let info = dtrsl(wn, 0, m2, col2, wv, 0, 1, 1);
    if info != 0 {
        return (0, info);
    }
    let mut pointr = head;
    for jy in 1..=col {
        let js = col + jy;
        for i in 1..=nsub {
            let k = ind[i - 1];
            d[i - 1] += rfma(
                ws[(k - 1) + (pointr - 1) * n],
                wv[js - 1],
                wy[(k - 1) + (pointr - 1) * n] * wv[jy - 1] / theta,
            );
        }
        pointr = pointr % m + 1;
    }
    for i in 1..=nsub {
        d[i - 1] /= theta;
    }
    let mut alpha = 1.0;
    let mut temp1 = alpha;
    let mut ibd = 0usize;
    for i in 1..=nsub {
        let k = ind[i - 1];
        let dk = d[i - 1];
        if nbd[k - 1] != 0 {
            if dk < 0.0 && nbd[k - 1] <= 2 {
                let temp2 = lo[k - 1] - x[k - 1];
                if temp2 >= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha < temp2 {
                    temp1 = temp2 / dk;
                }
            } else if dk > 0.0 && nbd[k - 1] >= 2 {
                let temp2 = up[k - 1] - x[k - 1];
                if temp2 <= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha > temp2 {
                    temp1 = temp2 / dk;
                }
            }
            if temp1 < alpha {
                alpha = temp1;
                ibd = i;
            }
        }
    }
    if alpha < 1.0 {
        let dk = d[ibd - 1];
        let k = ind[ibd - 1];
        if dk > 0.0 {
            x[k - 1] = up[k - 1];
            d[ibd - 1] = 0.0;
        } else if dk < 0.0 {
            x[k - 1] = lo[k - 1];
            d[ibd - 1] = 0.0;
        }
    }
    for i in 1..=nsub {
        let k = ind[i - 1];
        x[k - 1] = rfma(alpha, d[i - 1], x[k - 1]);
    }
    let iword = if alpha < 1.0 { 1 } else { 0 };
    (iword, 0)
}

#[allow(clippy::too_many_arguments)]
pub fn mainlb(
    n: usize,
    m: usize,
    x: &mut [f64],
    lo: &[f64],
    up: &[f64],
    nbd: &[i32],
    mut f: f64,
    g: &mut [f64],
    factr: f64,
    pgtol: f64,
    task: &str,
    st: &mut State,
) -> (f64, String) {
    let mut task = task.to_string();
    enum Jump {
        L111,
        L222,
        L333,
        L555,
        L666,
        L777,
        Stop,
    }
    let mut jump;
    if task.starts_with("START") {
        st.epsmch = DBL_EPSILON;
        st.fold = 0.0;
        st.dnorm = 0.0;
        st.gd = 0.0;
        st.sbgnrm = 0.0;
        st.stp = 0.0;
        st.xstep = 0.0;
        st.stpmx = 0.0;
        st.gdold = 0.0;
        st.dtd = 0.0;
        st.col = 0;
        st.head = 1;
        st.theta = 1.0;
        st.iupdat = 0;
        st.updatd = false;
        st.iback = 0;
        st.itail = 0;
        st.ifun = 0;
        st.iword = 0;
        st.nact = 0;
        st.ileave = 0;
        st.nenter = 0;
        st.iter = 0;
        st.nfgv = 0;
        st.nint = 0;
        st.nintol = 0;
        st.nskip = 0;
        st.nfree = n;
        st.tol = factr * st.epsmch;
        st.info = 0;
        st.ls = Dcsrch::default();
        st.csave = String::new();
        let (etask, einfo) = errclb(n, m, factr, lo, up, nbd);
        if !etask.is_empty() {
            st.info = einfo;
            st.isave13 = st.nfgv;
            return (f, etask);
        }
        {
            let mut iwhere = std::mem::take(&mut st.iwhere);
            active(n, lo, up, nbd, x, &mut iwhere, st);
            st.iwhere = iwhere;
        }
        task = "FG_START".into();
        st.isave13 = st.nfgv;
        return (f, task);
    } else if task.starts_with("FG_LN") {
        jump = Jump::L666;
    } else if task.starts_with("NEW_X") {
        jump = Jump::L777;
    } else if task.starts_with("FG_ST") {
        jump = Jump::L111;
    } else if task.starts_with("STOP") {
        if task.len() >= 9 && &task[6..9] == "CPU" {
            let t = st.t.clone();
            dcopy(n, &t, 0, x, 0);
            let r = st.r.clone();
            dcopy(n, &r, 0, g, 0);
            f = st.fold;
        }
        st.isave13 = st.nfgv;
        return (f, task);
    } else {
        task = "FG_START".into();
        st.isave13 = st.nfgv;
        return (f, task);
    }
    let mut wrk = false;
    loop {
        match jump {
            Jump::L111 => {
                st.nfgv = 1;
                st.sbgnrm = projgr(n, lo, up, nbd, x, g);
                if st.sbgnrm <= pgtol {
                    task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL".into();
                    jump = Jump::Stop;
                    continue;
                }
                jump = Jump::L222;
            }
            Jump::L222 => {
                st.iword = -1;
                if !st.cnstnd && st.col > 0 {
                    let z = std::mem::take(&mut st.z);
                    let mut z = z;
                    dcopy(n, x, 0, &mut z, 0);
                    st.z = z;
                    wrk = st.updatd;
                    st.nint = 0;
                    jump = Jump::L333;
                    continue;
                }
                let mut iorder = std::mem::take(&mut st.indx2);
                let mut iwhere = std::mem::take(&mut st.iwhere);
                let mut t = std::mem::take(&mut st.t);
                let mut d = std::mem::take(&mut st.d);
                let mut z = std::mem::take(&mut st.z);
                let mut wa = std::mem::take(&mut st.wa);
                let (nint, info) = cauchy(
                    n,
                    x,
                    lo,
                    up,
                    nbd,
                    g,
                    &mut iorder,
                    &mut iwhere,
                    &mut t,
                    &mut d,
                    &mut z,
                    m,
                    &st.wy,
                    &st.ws,
                    &st.sy,
                    &st.wt,
                    st.theta,
                    st.col,
                    st.head,
                    &mut wa,
                    st.sbgnrm,
                    st.epsmch,
                );
                st.indx2 = iorder;
                st.iwhere = iwhere;
                st.t = t;
                st.d = d;
                st.z = z;
                st.wa = wa;
                st.nint = nint;
                st.info = info;
                if st.info != 0 {
                    st.info = 0;
                    st.col = 0;
                    st.head = 1;
                    st.theta = 1.0;
                    st.iupdat = 0;
                    st.updatd = false;
                    jump = Jump::L222;
                    continue;
                }
                st.nintol += st.nint;
                let mut indx = std::mem::take(&mut st.indx);
                let mut indx2 = std::mem::take(&mut st.indx2);
                let (nfree, nenter, ileave, w) = freev(
                    n, st.nfree, &mut indx, &mut indx2, &st.iwhere, st.updatd, st.cnstnd, st.iter,
                );
                st.indx = indx;
                st.indx2 = indx2;
                st.nfree = nfree;
                st.nenter = nenter;
                st.ileave = ileave;
                wrk = w;
                st.nact = n - st.nfree;
                jump = Jump::L333;
            }
            Jump::L333 => {
                if st.nfree == 0 || st.col == 0 {
                    jump = Jump::L555;
                    continue;
                }
                if wrk {
                    let mut wn = std::mem::take(&mut st.wn);
                    let mut snd = std::mem::take(&mut st.snd);
                    st.info = formk(
                        n, st.nfree, &st.indx, st.nenter, st.ileave, &st.indx2, st.iupdat,
                        st.updatd, &mut wn, &mut snd, m, &st.ws, &st.wy, &st.sy, st.theta, st.col,
                        st.head,
                    );
                    st.wn = wn;
                    st.snd = snd;
                }
                if st.info != 0 {
                    st.info = 0;
                    st.col = 0;
                    st.head = 1;
                    st.theta = 1.0;
                    st.iupdat = 0;
                    st.updatd = false;
                    jump = Jump::L222;
                    continue;
                }
                {
                    let mut r = std::mem::take(&mut st.r);
                    let mut wa = std::mem::take(&mut st.wa);
                    st.info = cmprlb(
                        n, m, x, g, &st.ws, &st.wy, &st.sy, &st.wt, &st.z, &mut r, &mut wa,
                        &st.indx, st.theta, st.col, st.head, st.nfree, st.cnstnd,
                    );
                    st.r = r;
                    st.wa = wa;
                }
                if st.info == 0 {
                    let mut z = std::mem::take(&mut st.z);
                    let mut r = std::mem::take(&mut st.r);
                    let mut wa = std::mem::take(&mut st.wa);
                    let (iword, info) = subsm(
                        n, m, st.nfree, &st.indx, lo, up, nbd, &mut z, &mut r, &st.ws, &st.wy,
                        st.theta, st.col, st.head, &mut wa, &st.wn,
                    );
                    st.z = z;
                    st.r = r;
                    st.wa = wa;
                    st.iword = iword;
                    st.info = info;
                }
                if st.info != 0 {
                    st.info = 0;
                    st.col = 0;
                    st.head = 1;
                    st.theta = 1.0;
                    st.iupdat = 0;
                    st.updatd = false;
                    jump = Jump::L222;
                    continue;
                }
                jump = Jump::L555;
            }
            Jump::L555 => {
                for i in 1..=n {
                    st.d[i - 1] = st.z[i - 1] - x[i - 1];
                }
                jump = Jump::L666;
            }
            Jump::L666 => {
                let d = std::mem::take(&mut st.d);
                let z = std::mem::take(&mut st.z);
                let mut r = std::mem::take(&mut st.r);
                let mut t = std::mem::take(&mut st.t);
                let (fnew, tnew) =
                    lnsrlb(n, lo, up, nbd, x, f, g, &d, &mut r, &mut t, &z, &task, st);
                st.d = d;
                st.z = z;
                st.r = r;
                st.t = t;
                f = fnew;
                task = tnew;
                if st.info != 0 || st.iback >= 20 {
                    let t = st.t.clone();
                    dcopy(n, &t, 0, x, 0);
                    let r = st.r.clone();
                    dcopy(n, &r, 0, g, 0);
                    f = st.fold;
                    if st.col == 0 {
                        if st.info == 0 {
                            st.info = -9;
                            st.nfgv -= 1;
                            st.ifun -= 1;
                            st.iback -= 1;
                        }
                        task = "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH".into();
                        st.iter += 1;
                        jump = Jump::Stop;
                        continue;
                    } else {
                        if st.info == 0 {
                            st.nfgv -= 1;
                        }
                        st.info = 0;
                        st.col = 0;
                        st.head = 1;
                        st.theta = 1.0;
                        st.iupdat = 0;
                        st.updatd = false;
                        task = "RESTART_FROM_LNSRCH".into();
                        jump = Jump::L222;
                        continue;
                    }
                } else if task.starts_with("FG_LN") {
                    jump = Jump::Stop;
                    continue;
                } else {
                    st.iter += 1;
                    st.sbgnrm = projgr(n, lo, up, nbd, x, g);
                    jump = Jump::Stop;
                    continue;
                }
            }
            Jump::L777 => {
                if st.sbgnrm <= pgtol {
                    task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL".into();
                    jump = Jump::Stop;
                    continue;
                }
                let ddum = st.fold.abs().max(f.abs()).max(1.0);
                if st.fold - f <= st.tol * ddum {
                    task = "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH".into();
                    if st.iback >= 10 {
                        st.info = -5;
                    }
                    jump = Jump::Stop;
                    continue;
                }
                for i in 1..=n {
                    st.r[i - 1] = g[i - 1] - st.r[i - 1];
                }
                st.rr = ddot(n, &st.r, 0, 1, &st.r, 0, 1);
                let ddum2;
                if st.stp == 1.0 {
                    st.dr = st.gd - st.gdold;
                    ddum2 = -st.gdold;
                } else {
                    st.dr = (st.gd - st.gdold) * st.stp;
                    let stp = st.stp;
                    dscal(n, stp, &mut st.d, 0);
                    ddum2 = -st.gdold * st.stp;
                }
                if st.dr <= st.epsmch * ddum2 {
                    st.nskip += 1;
                    st.updatd = false;
                    jump = Jump::L222;
                    continue;
                }
                st.updatd = true;
                st.iupdat += 1;
                {
                    let mut ws = std::mem::take(&mut st.ws);
                    let mut wy = std::mem::take(&mut st.wy);
                    let mut sy = std::mem::take(&mut st.sy);
                    let mut ss = std::mem::take(&mut st.ss);
                    let d = st.d.clone();
                    let r = st.r.clone();
                    matupd(n, m, &mut ws, &mut wy, &mut sy, &mut ss, &d, &r, st);
                    st.ws = ws;
                    st.wy = wy;
                    st.sy = sy;
                    st.ss = ss;
                }
                {
                    let mut wt = std::mem::take(&mut st.wt);
                    st.info = formt(m, &mut wt, &st.sy, &st.ss, st.col, st.theta);
                    st.wt = wt;
                }
                if st.info != 0 {
                    st.info = 0;
                    st.col = 0;
                    st.head = 1;
                    st.theta = 1.0;
                    st.iupdat = 0;
                    st.updatd = false;
                }
                jump = Jump::L222;
            }
            Jump::Stop => break,
        }
    }
    st.isave13 = st.nfgv;
    (f, task)
}

/// R's `lbfgsb()` driver (src/appl/optim.c:642). Mutates `x`; returns
/// (Fmin, fail, fncount, grcount, msg).
#[allow(clippy::too_many_arguments)]
pub fn lbfgsb_drive(
    n: usize,
    m: usize,
    x: &mut [f64],
    lo: &[f64],
    up: &[f64],
    nbd: &[i32],
    obj: &mut dyn Objective,
    factr: f64,
    pgtol: f64,
    maxit: i32,
) -> PyResult<(f64, i32, i32, i32, String)> {
    if n == 0 {
        let fmin = obj.f(up)?;
        return Ok((fmin, 0, 1, 0, "NOTHING TO DO".into()));
    }
    let mut fail = 0;
    let mut g = vec![0.0; n];
    let mut st = State::new(n, m);
    let mut task: String = "START".into();
    let mut f = 0.0;
    let mut iter = 0;
    loop {
        let (fnew, tnew) = mainlb(
            n, m, x, lo, up, nbd, f, &mut g, factr, pgtol, &task, &mut st,
        );
        f = fnew;
        task = tnew;
        if task.starts_with("FG") {
            f = obj.f(x)?;
            if !f.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "L-BFGS-B needs finite values of 'fn'",
                ));
            }
            obj.grad(x, &mut g)?;
        } else if task.starts_with("NEW_X") {
            iter += 1;
            if iter > maxit {
                fail = 1;
                break;
            }
        } else if task.starts_with("WARN") {
            fail = 51;
            break;
        } else if task.starts_with("CONV") {
            break;
        } else {
            fail = 52;
            break;
        }
    }
    let fncount = st.isave13;
    Ok((f, fail, fncount, fncount, task))
}
