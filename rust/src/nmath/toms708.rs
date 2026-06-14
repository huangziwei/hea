//! `pbeta` / `lbeta` — R's nmath/toms708.c (Morris ALGORITHM 708 `bratio`) +
//! pbeta.c / lbeta.c. Mirror of the `hea/R/nmath.py` toms708 cluster. Scalar
//! kernels + numpy-vectorized PyO3 wrappers. Float-op order preserved exactly.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::consts::{DBL_MIN, M_LN2, M_LN_SQRT_2PI};
use super::lgamma::{gammafn, lgammacor, lgammafn};
use super::norm::{dt0, dt1};
use super::util::ldexp;

const TOMS_EPS: f64 = 2.220446049250313e-16;
const M_SQRT_PI: f64 = 1.772453850905516027298167483341;

const ERF_A: [f64; 5] = [
    7.7105849500132e-5, -0.00133733772997339, 0.0323076579225834,
    0.0479137145607681, 0.128379167095513,
];
const ERF_B: [f64; 3] = [0.00301048631703895, 0.0538971687740286, 0.375795757275549];
const ERF_P: [f64; 8] = [
    -1.36864857382717e-7, 0.564195517478974, 7.21175825088309, 43.1622272220567,
    152.98928504694, 339.320816734344, 451.918953711873, 300.459261020162,
];
const ERF_Q: [f64; 8] = [
    1.0, 12.7827273196294, 77.0001529352295, 277.585444743988, 638.980264465631,
    931.35409485061, 790.950925327898, 300.459260956983,
];
const ERF_R: [f64; 5] = [
    2.10144126479064, 26.2370141675169, 21.3688200555087, 4.6580782871847,
    0.282094791773523,
];
const ERF_S: [f64; 4] = [94.153775055546, 187.11481179959, 99.0191814623914, 18.0124575948747];
const ERF_C: f64 = 0.564189583547756;

const PSI_P1: [f64; 7] = [
    0.0089538502298197, 4.77762828042627, 142.441585084029, 1186.45200713425,
    3633.51846806499, 4138.10161269013, 1305.60269827897,
];
const PSI_Q1: [f64; 6] = [
    44.8452573429826, 520.752771467162, 2210.0079924783, 3641.27349079381,
    1908.310765963, 6.91091682714533e-6,
];
const PSI_P2: [f64; 4] = [-2.12940445131011, -7.01677227766759, -4.48616543918019, -0.648157123766197];
const PSI_Q2: [f64; 4] = [32.2703493791143, 89.2920700481861, 54.6117738103215, 7.77788548522962];

const BCORR_C: [f64; 6] = [
    0.0833333333333333, -0.00277777777760991, 7.9365066682539e-4,
    -5.9520293135187e-4, 8.37308034031215e-4, -0.00165322962780713,
];

fn exparg(which: i32) -> f64 {
    let lnb = 0.69314718055995;
    let m = if which == 0 { 1024.0 } else { (-1021 - 1) as f64 };
    m * lnb * 0.99999
}

fn esum(mu: f64, x: f64, give_log: bool) -> f64 {
    if give_log {
        return x + mu;
    }
    let w;
    if x > 0.0 {
        if mu > 0.0 {
            return mu.exp() * x.exp();
        }
        w = mu + x;
        if w < 0.0 {
            return mu.exp() * x.exp();
        }
    } else {
        if mu < 0.0 {
            return mu.exp() * x.exp();
        }
        w = mu + x;
        if w > 0.0 {
            return mu.exp() * x.exp();
        }
    }
    w.exp()
}

fn rexpm1(x: f64) -> f64 {
    let p1 = 9.14041914819518e-10;
    let p2 = 0.0238082361044469;
    let q1 = -0.499999999085958;
    let q2 = 0.107141568980644;
    let q3 = -0.0119041179760821;
    let q4 = 5.95130811860248e-4;
    if x.abs() <= 0.15 {
        return x * (((p2 * x + p1) * x + 1.0)
            / ((((q4 * x + q3) * x + q2) * x + q1) * x + 1.0));
    }
    let w = x.exp();
    if x > 0.0 {
        w * (0.5 - 1.0 / w + 0.5)
    } else {
        w - 0.5 - 0.5
    }
}

fn alnrel(a: f64) -> f64 {
    if a.abs() > 0.375 {
        return (1.0 + a).ln();
    }
    let p1 = -1.29418923021993;
    let p2 = 0.405303492862024;
    let p3 = -0.0178874546012214;
    let q1 = -1.62752256355323;
    let q2 = 0.747811014037616;
    let q3 = -0.0845104217945565;
    let t = a / (a + 2.0);
    let t2 = t * t;
    let w = (((p3 * t2 + p2) * t2 + p1) * t2 + 1.0)
        / (((q3 * t2 + q2) * t2 + q1) * t2 + 1.0);
    t * 2.0 * w
}

fn rlog1(x: f64) -> f64 {
    let a = 0.0566749439387324;
    let b = 0.0456512608815524;
    let p0 = 0.333333333333333;
    let p1 = -0.224696413112536;
    let p2 = 0.00620886815375787;
    let q1 = -1.27408923933623;
    let q2 = 0.354508718369557;
    if x < -0.39 || x > 0.57 {
        let w = x + 0.5 + 0.5;
        return x - w.ln();
    }
    let h;
    let w1;
    if x < -0.18 {
        h = (x + 0.3) / 0.7;
        w1 = a - h * 0.3;
    } else if x > 0.18 {
        h = x * 0.75 - 0.25;
        w1 = b + h / 3.0;
    } else {
        h = x;
        w1 = 0.0;
    }
    let r = h / (h + 2.0);
    let t = r * r;
    let w = ((p2 * t + p1) * t + p0) / ((q2 * t + q1) * t + 1.0);
    t * 2.0 * (1.0 / (1.0 - r) - r * w) + w1
}

fn erf_(x: f64) -> f64 {
    let (a, b, p, q, r, s) = (&ERF_A, &ERF_B, &ERF_P, &ERF_Q, &ERF_R, &ERF_S);
    let ax = x.abs();
    if ax <= 0.5 {
        let t = x * x;
        let top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.0;
        let bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        return x * (top / bot);
    }
    if ax <= 4.0 {
        let top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5])
            * ax + p[6]) * ax + p[7];
        let bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5])
            * ax + q[6]) * ax + q[7];
        let rr = 0.5 - (-x * x).exp() * top / bot + 0.5;
        return if x < 0.0 { -rr } else { rr };
    }
    if ax >= 5.8 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let x2 = x * x;
    let t = 1.0 / x2;
    let top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
    let bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
    let t = (ERF_C - top / (x2 * bot)) / ax;
    let rr = 0.5 - (-x2).exp() * t + 0.5;
    if x < 0.0 { -rr } else { rr }
}

fn erfc1(ind: i32, x: f64) -> f64 {
    let (a, b, p, q, r, s) = (&ERF_A, &ERF_B, &ERF_P, &ERF_Q, &ERF_R, &ERF_S);
    let ax = x.abs();
    let mut ret;
    if ax <= 0.5 {
        let t = x * x;
        let top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.0;
        let bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        ret = 0.5 - x * (top / bot) + 0.5;
        if ind != 0 {
            ret = t.exp() * ret;
        }
        return ret;
    }
    if ax <= 4.0 {
        let top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5])
            * ax + p[6]) * ax + p[7];
        let bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5])
            * ax + q[6]) * ax + q[7];
        ret = top / bot;
    } else {
        if x <= -5.6 {
            ret = 2.0;
            if ind != 0 {
                ret = (x * x).exp() * 2.0;
            }
            return ret;
        }
        if ind == 0 && (x > 100.0 || x * x > -exparg(1)) {
            return 0.0;
        }
        let t = 1.0 / (x * x);
        let top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
        let bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
        ret = (ERF_C - t * top / bot) / ax;
    }
    if ind != 0 {
        if x < 0.0 {
            ret = (x * x).exp() * 2.0 - ret;
        }
    } else {
        let w = x * x;
        let t = w;
        let e = w - t;
        ret = (0.5 - e + 0.5) * (-t).exp() * ret;
        if x < 0.0 {
            ret = 2.0 - ret;
        }
    }
    ret
}

fn gam1(a: f64) -> f64 {
    let d = a - 0.5;
    let t = if d > 0.0 { d - 0.5 } else { a };
    if t < 0.0 {
        let r = [
            -0.422784335098468, -0.771330383816272, -0.244757765222226, 0.118378989872749,
            9.30357293360349e-4, -0.0118290993445146, 0.00223047661158249, 2.66505979058923e-4,
            -1.32674909766242e-4,
        ];
        let s1 = 0.273076135303957;
        let s2 = 0.0559398236957378;
        let top = (((((((r[8] * t + r[7]) * t + r[6]) * t + r[5]) * t + r[4]) * t + r[3]) * t
            + r[2]) * t + r[1]) * t + r[0];
        let bot = (s2 * t + s1) * t + 1.0;
        let w = top / bot;
        if d > 0.0 { t * w / a } else { a * (w + 0.5 + 0.5) }
    } else if t == 0.0 {
        0.0
    } else {
        let p = [
            0.577215664901533, -0.409078193005776, -0.230975380857675, 0.0597275330452234,
            0.0076696818164949, -0.00514889771323592, 5.89597428611429e-4,
        ];
        let q = [
            1.0, 0.427569613095214, 0.158451672430138, 0.0261132021441447, 0.00423244297896961,
        ];
        let top = (((((p[6] * t + p[5]) * t + p[4]) * t + p[3]) * t + p[2]) * t + p[1]) * t + p[0];
        let bot = (((q[4] * t + q[3]) * t + q[2]) * t + q[1]) * t + 1.0;
        let w = top / bot;
        if d > 0.0 { t / a * (w - 0.5 - 0.5) } else { a * w }
    }
}

fn gamln1(a: f64) -> f64 {
    if a < 0.6 {
        let p0 = 0.577215664901533;
        let p1 = 0.844203922187225;
        let p2 = -0.168860593646662;
        let p3 = -0.780427615533591;
        let p4 = -0.402055799310489;
        let p5 = -0.0673562214325671;
        let p6 = -0.00271935708322958;
        let q1 = 2.88743195473681;
        let q2 = 3.12755088914843;
        let q3 = 1.56875193295039;
        let q4 = 0.361951990101499;
        let q5 = 0.0325038868253937;
        let q6 = 6.67465618796164e-4;
        let w = ((((((p6 * a + p5) * a + p4) * a + p3) * a + p2) * a + p1) * a + p0)
            / ((((((q6 * a + q5) * a + q4) * a + q3) * a + q2) * a + q1) * a + 1.0);
        return -a * w;
    }
    let r0 = 0.422784335098467;
    let r1 = 0.848044614534529;
    let r2 = 0.565221050691933;
    let r3 = 0.156513060486551;
    let r4 = 0.017050248402265;
    let r5 = 4.97958207639485e-4;
    let s1 = 1.24313399877507;
    let s2 = 0.548042109832463;
    let s3 = 0.10155218743983;
    let s4 = 0.00713309612391;
    let s5 = 1.16165475989616e-4;
    let x = a - 0.5 - 0.5;
    let w = (((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x + r0)
        / (((((s5 * x + s4) * x + s3) * x + s2) * x + s1) * x + 1.0);
    x * w
}

fn psi(x: f64) -> f64 {
    let mut x = x;
    let piov4 = 0.785398163397448;
    let dx0 = 1.461632144968362341262659542325721325;
    let (p1, q1, p2, q2) = (&PSI_P1, &PSI_Q1, &PSI_P2, &PSI_Q2);
    let mut xmax1 = 2147483647.0_f64;
    let d2 = 0.5 / (0.5 * f64::EPSILON);
    if xmax1 > d2 {
        xmax1 = d2;
    }
    let xsmall = 1e-9;
    let mut aug = 0.0;
    if x < 0.5 {
        if x.abs() <= xsmall {
            if x == 0.0 {
                return 0.0;
            }
            aug = -1.0 / x;
        } else {
            let mut w = -x;
            let mut sgn = piov4;
            if w <= 0.0 {
                w = -w;
                sgn = -sgn;
            }
            if w >= xmax1 {
                return 0.0;
            }
            let mut nq = w as i64;
            w -= nq as f64;
            nq = (w * 4.0) as i64;
            w = (w - nq as f64 * 0.25) * 4.0;
            let n1 = nq / 2;
            if n1 + n1 != nq {
                w = 1.0 - w;
            }
            let z = piov4 * w;
            let m1 = n1 / 2;
            if m1 + m1 != n1 {
                sgn = -sgn;
            }
            let n = (nq + 1) / 2;
            let mut m = n / 2;
            m += m;
            if m == n {
                if z == 0.0 {
                    return 0.0;
                }
                aug = sgn * (z.cos() / z.sin() * 4.0);
            } else {
                aug = sgn * (z.sin() / z.cos() * 4.0);
            }
        }
        x = 1.0 - x;
    }
    if x <= 3.0 {
        let mut den = x;
        let mut upper = p1[0] * x;
        for i in 1..6 {
            den = (den + q1[i - 1]) * x;
            upper = (upper + p1[i]) * x;
        }
        den = (upper + p1[6]) / (den + q1[5]);
        let xmx0 = x - dx0;
        return den * xmx0 + aug;
    }
    if x < xmax1 {
        let w = 1.0 / (x * x);
        let mut den = w;
        let mut upper = p2[0] * w;
        for i in 1..4 {
            den = (den + q2[i - 1]) * w;
            upper = (upper + p2[i]) * w;
        }
        aug = upper / (den + q2[3]) - 0.5 / x + aug;
    }
    aug + x.ln()
}

fn bcorr(a0: f64, b0: f64) -> f64 {
    let [c0, c1, c2, c3, c4, c5] = BCORR_C;
    let a = a0.min(b0);
    let b = a0.max(b0);
    let h = a / b;
    let c = h / (h + 1.0);
    let x = 1.0 / (h + 1.0);
    let x2 = x * x;
    let s3 = x + x2 + 1.0;
    let s5 = x + x2 * s3 + 1.0;
    let s7 = x + x2 * s5 + 1.0;
    let s9 = x + x2 * s7 + 1.0;
    let s11 = x + x2 * s9 + 1.0;
    let mut t = 1.0 / b;
    t *= t;
    let mut w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0;
    w *= c / b;
    let mut t = 1.0 / a;
    t *= t;
    (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a + w
}

fn algdiv(a: f64, b: f64) -> f64 {
    let [c0, c1, c2, c3, c4, c5] = BCORR_C;
    let (h, c, x, d);
    if a > b {
        h = b / a;
        c = 1.0 / (h + 1.0);
        x = h / (h + 1.0);
        d = a + (b - 0.5);
    } else {
        h = a / b;
        c = h / (h + 1.0);
        x = 1.0 / (h + 1.0);
        d = b + (a - 0.5);
    }
    let x2 = x * x;
    let s3 = x + x2 + 1.0;
    let s5 = x + x2 * s3 + 1.0;
    let s7 = x + x2 * s5 + 1.0;
    let s9 = x + x2 * s7 + 1.0;
    let s11 = x + x2 * s9 + 1.0;
    let t = 1.0 / (b * b);
    let mut w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0;
    w *= c / b;
    let u = d * alnrel(a / b);
    let v = a * (b.ln() - 1.0);
    if u > v { w - v - u } else { w - u - v }
}

fn gamln(a: f64) -> f64 {
    let [c0, c1, c2, c3, c4, c5] = BCORR_C;
    let d = 0.418938533204673;
    if a <= 0.8 {
        return gamln1(a) - a.ln();
    } else if a <= 2.25 {
        return gamln1(a - 0.5 - 0.5);
    } else if a < 10.0 {
        let n = (a - 1.25) as i64;
        let mut t = a;
        let mut w = 1.0;
        for _ in 1..=n {
            t += -1.0;
            w *= t;
        }
        return gamln1(t - 1.0) + w.ln();
    }
    let t = 1.0 / (a * a);
    let w = (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a;
    d + w + (a - 0.5) * (a.ln() - 1.0)
}

fn gsumln(a: f64, b: f64) -> f64 {
    let x = a + b - 2.0;
    if x <= 0.25 {
        return gamln1(x + 1.0);
    }
    if x <= 1.25 {
        return gamln1(x) + alnrel(x);
    }
    gamln1(x - 1.0) + (x * (x + 1.0)).ln()
}

fn betaln(a0: f64, b0: f64) -> f64 {
    let mut a = a0.min(b0);
    let mut b = a0.max(b0);
    if a < 8.0 {
        if a < 1.0 {
            if b < 8.0 {
                return gamln(a) + (gamln(b) - gamln(a + b));
            }
            return gamln(a) + algdiv(a, b);
        }
        let mut w = 0.0;
        let mut skip_to_40 = false;
        if a < 2.0 {
            if b <= 2.0 {
                return gamln(a) + gamln(b) - gsumln(a, b);
            }
            if b < 8.0 {
                w = 0.0;
                skip_to_40 = true;
            } else {
                return gamln(a) + algdiv(a, b);
            }
        }
        if !skip_to_40 {
            if b <= 1e3 {
                let n = (a - 1.0) as i64;
                let mut ww = 1.0;
                for _ in 1..=n {
                    a += -1.0;
                    let h = a / b;
                    ww *= h / (h + 1.0);
                }
                w = ww.ln();
                if b >= 8.0 {
                    return w + gamln(a) + algdiv(a, b);
                }
            } else {
                let n = (a - 1.0) as i64;
                let mut ww = 1.0;
                for _ in 1..=n {
                    a += -1.0;
                    ww *= a / (a / b + 1.0);
                }
                return ww.ln() - (n as f64) * b.ln() + (gamln(a) + algdiv(a, b));
            }
        }
        // L40
        let n = (b - 1.0) as i64;
        let mut z = 1.0;
        for _ in 1..=n {
            b += -1.0;
            z *= b / (a + b);
        }
        return w + z.ln() + (gamln(a) + (gamln(b) - gsumln(a, b)));
    }
    let e = 0.918938533204673;
    let w = bcorr(a, b);
    let h = a / b;
    let u = -(a - 0.5) * (h / (h + 1.0)).ln();
    let v = b * alnrel(h);
    if u > v {
        b.ln() * -0.5 + e + w - v - u
    } else {
        b.ln() * -0.5 + e + w - u - v
    }
}

fn fpser(a: f64, b: f64, x: f64, eps: f64, log_p: bool) -> f64 {
    let mut ans;
    if log_p {
        ans = a * x.ln();
    } else if a > eps * 0.001 {
        let t = a * x.ln();
        if t < exparg(1) {
            return 0.0;
        }
        ans = t.exp();
    } else {
        ans = 1.0;
    }
    if log_p {
        ans += b.ln() - a.ln();
    } else {
        ans *= b / a;
    }
    let tol = eps / a;
    let mut an = a + 1.0;
    let mut t = x;
    let mut s = t / an;
    loop {
        an += 1.0;
        t = x * t;
        let c = t / an;
        s += c;
        if !(c.abs() > tol) {
            break;
        }
    }
    if log_p {
        ans += (a * s).ln_1p();
    } else {
        ans *= a * s + 1.0;
    }
    ans
}

fn apser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    let g = 0.577215664901533;
    let bx = b * x;
    let mut t = x - bx;
    let c;
    if b * eps <= 0.02 {
        c = x.ln() + psi(b) + g + t;
    } else {
        c = bx.ln() + g + t;
    }
    let tol = eps * 5.0 * c.abs();
    let mut j = 1.0;
    let mut s = 0.0;
    loop {
        j += 1.0;
        t *= x - bx / j;
        let aj = t / j;
        s += aj;
        if !(aj.abs() > tol) {
            break;
        }
    }
    -a * (c + s)
}

fn bpser(a: f64, b: f64, x: f64, eps: f64, log_p: bool) -> f64 {
    let rd0 = if log_p { f64::NEG_INFINITY } else { 0.0 };
    if x == 0.0 {
        return rd0;
    }
    let a0 = a.min(b);
    let mut ans;
    if a0 >= 1.0 {
        let z = a * x.ln() - betaln(a, b);
        ans = if log_p { z - a.ln() } else { z.exp() / a };
    } else {
        let mut b0 = a.max(b);
        if b0 < 8.0 {
            if b0 <= 1.0 {
                ans = if log_p { a * x.ln() } else { x.powf(a) };
                if !log_p && ans == 0.0 {
                    return ans;
                }
                let apb = a + b;
                let z;
                if apb > 1.0 {
                    let u = a + b - 1.0;
                    z = (gam1(u) + 1.0) / apb;
                } else {
                    z = gam1(apb) + 1.0;
                }
                let c = (gam1(a) + 1.0) * (gam1(b) + 1.0) / z;
                if log_p {
                    ans += (c * (b / apb)).ln();
                } else {
                    ans *= c * (b / apb);
                }
            } else {
                let mut u = gamln1(a0);
                let m = (b0 - 1.0) as i64;
                if m >= 1 {
                    let mut c = 1.0;
                    for _ in 1..=m {
                        b0 += -1.0;
                        c *= b0 / (a0 + b0);
                    }
                    u += c.ln();
                }
                let z = a * x.ln() - u;
                b0 += -1.0;
                let apb = a0 + b0;
                let t;
                if apb > 1.0 {
                    let u2 = a0 + b0 - 1.0;
                    t = (gam1(u2) + 1.0) / apb;
                } else {
                    t = gam1(apb) + 1.0;
                }
                if log_p {
                    ans = z + (a0 / a).ln() + gam1(b0).ln_1p() - t.ln();
                } else {
                    ans = z.exp() * (a0 / a) * (gam1(b0) + 1.0) / t;
                }
            }
        } else {
            let u = gamln1(a0) + algdiv(a0, b0);
            let z = a * x.ln() - u;
            ans = if log_p { z + (a0 / a).ln() } else { a0 / a * z.exp() };
        }
    }
    if ans == rd0 || (!log_p && a <= eps * 0.1) {
        return ans;
    }
    let tol = eps / a;
    let mut n = 0.0;
    let mut sum = 0.0;
    let mut c = 1.0;
    let mut w;
    loop {
        n += 1.0;
        c *= (0.5 - b / n + 0.5) * x;
        w = c / (a + n);
        sum += w;
        if !(n < 1e7 && w.abs() > tol) {
            break;
        }
    }
    if log_p {
        if a * sum > -1.0 {
            ans += (a * sum).ln_1p();
        } else {
            ans = f64::NEG_INFINITY;
        }
    } else if a * sum > -1.0 {
        ans *= a * sum + 1.0;
    } else {
        ans = 0.0;
    }
    ans
}

fn bup(a: f64, b: f64, x: f64, y: f64, n: i32, eps: f64, give_log: bool) -> f64 {
    let apb = a + b;
    let ap1 = a + 1.0;
    let mut d;
    let mu;
    if n > 1 && a >= 1.0 && apb >= ap1 * 1.1 {
        let mut muv = exparg(1).abs() as i64;
        let k = exparg(0) as i64;
        if muv > k {
            muv = k;
        }
        mu = muv;
        d = (-(mu as f64)).exp();
    } else {
        mu = 0;
        d = 1.0;
    }
    let mut ret = if give_log {
        brcmp1(mu as f64, a, b, x, y, true) - a.ln()
    } else {
        brcmp1(mu as f64, a, b, x, y, false) / a
    };
    if n == 1 || (give_log && ret == f64::NEG_INFINITY) || (!give_log && ret == 0.0) {
        return ret;
    }
    let nm1 = n - 1;
    let mut w = d;
    let mut k = 0i32;
    if b > 1.0 {
        if y > 1e-4 {
            let r = (b - 1.0) * x / y - a;
            if r >= 1.0 {
                k = if r < nm1 as f64 { r as i32 } else { nm1 };
            }
        } else {
            k = nm1;
        }
        for i in 0..k {
            let ll = i as f64;
            d *= (apb + ll) / (ap1 + ll) * x;
            w += d;
        }
    }
    for i in k..nm1 {
        let ll = i as f64;
        d *= (apb + ll) / (ap1 + ll) * x;
        w += d;
        if d <= eps * w {
            break;
        }
    }
    if give_log {
        ret += w.ln();
    } else {
        ret *= w;
    }
    ret
}

fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64, eps: f64, log_p: bool) -> f64 {
    if !lambda.is_finite() {
        return f64::NAN;
    }
    let brc = brcomp(a, b, x, y, log_p);
    if brc.is_nan() {
        return f64::NAN;
    }
    if !log_p && brc == 0.0 {
        return 0.0;
    }
    let c = lambda + 1.0;
    let c0 = b / a;
    let c1 = 1.0 / a + 1.0;
    let yp1 = y + 1.0;
    let mut n = 0.0;
    let mut p = 1.0;
    let mut s = a + 1.0;
    let mut an = 0.0;
    let mut bn = 1.0;
    let mut anp1 = 1.0;
    let mut bnp1 = c / c1;
    let mut r = c1 / c;
    let mut r0;
    let max_it = 1000;
    while n < max_it as f64 {
        n += 1.0;
        let mut w = n * x * (b - n);
        let rescale = !w.is_finite();
        if rescale {
            w = n * x * ldexp(b - n, -20);
        }
        let t = n / a;
        let mut e = a / s;
        let alpha = p * (p + c0) * e * e * (w * x);
        e = (t + 1.0) / (c1 + t + t);
        let beta = w / s
            + if rescale {
                ldexp(n + e * (c + n * yp1), -20)
            } else {
                n + e * (c + n * yp1)
            };
        p = t + 1.0;
        s += 2.0;
        let mut tt = alpha * an + beta * anp1;
        an = anp1;
        anp1 = tt;
        tt = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = tt;
        r0 = r;
        r = anp1 / bnp1;
        if (r - r0).abs() <= eps * r {
            break;
        }
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
    }
    if log_p { brc + r.ln() } else { brc * r }
}

fn brcomp(a: f64, b: f64, x: f64, y: f64, log_p: bool) -> f64 {
    let rd0 = if log_p { f64::NEG_INFINITY } else { 0.0 };
    if x == 0.0 || y == 0.0 {
        return rd0;
    }
    let a0 = a.min(b);
    if a0 < 8.0 {
        let (lnx, lny);
        if x <= 0.375 {
            lnx = x.ln();
            lny = alnrel(-x);
        } else if y > 0.375 {
            lnx = x.ln();
            lny = y.ln();
        } else {
            lnx = alnrel(-y);
            lny = y.ln();
        }
        let mut z = a * lnx + b * lny;
        if a0 >= 1.0 {
            z -= betaln(a, b);
            return if log_p { z } else { z.exp() };
        }
        let mut b0 = a.max(b);
        if b0 >= 8.0 {
            let u = gamln1(a0) + algdiv(a0, b0);
            return if log_p { a0.ln() + (z - u) } else { a0 * (z - u).exp() };
        }
        if b0 <= 1.0 {
            let e_z = if log_p { z } else { z.exp() };
            if !log_p && e_z == 0.0 {
                return 0.0;
            }
            let apb = a + b;
            let zz;
            if apb > 1.0 {
                zz = (gam1(apb - 1.0) + 1.0) / apb;
            } else {
                zz = gam1(apb) + 1.0;
            }
            let c = (gam1(a) + 1.0) * (gam1(b) + 1.0) / zz;
            return if log_p {
                e_z + (a0 * c).ln() - (a0 / b0).ln_1p()
            } else {
                e_z * (a0 * c) / (a0 / b0 + 1.0)
            };
        }
        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i64;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 1..=n {
                b0 += -1.0;
                c *= b0 / (a0 + b0);
            }
            u = c.ln() + u;
        }
        z -= u;
        b0 += -1.0;
        let apb = a0 + b0;
        let t;
        if apb > 1.0 {
            let u2 = a0 + b0 - 1.0;
            t = (gam1(u2) + 1.0) / apb;
        } else {
            t = gam1(apb) + 1.0;
        }
        if log_p {
            a0.ln() + z + gam1(b0).ln_1p() - t.ln()
        } else {
            a0 * z.exp() * (gam1(b0) + 1.0) / t
        }
    } else {
        let const__: f64 = 0.398942280401433;
        let apb = a + b;
        let lambda = if apb.is_finite() {
            if a <= b { a - apb * x } else { apb * y - b }
        } else {
            a * y - b * x
        };
        let (h, x0, y0);
        if a <= b {
            h = a / b;
            x0 = h / (h + 1.0);
            y0 = 1.0 / (h + 1.0);
        } else {
            h = b / a;
            x0 = 1.0 / (h + 1.0);
            y0 = h / (h + 1.0);
        }
        let mut e = -lambda / a;
        let u = if e.abs() > 0.6 { e - (x / x0).ln() } else { rlog1(e) };
        e = lambda / b;
        let v = if e.abs() <= 0.6 { rlog1(e) } else { e - (y / y0).ln() };
        let z = if log_p { -(a * u + b * v) } else { (-(a * u + b * v)).exp() };
        if log_p {
            -M_LN_SQRT_2PI + 0.5 * (b * x0).ln() + z - bcorr(a, b)
        } else {
            const__ * (b * x0).sqrt() * z * (-bcorr(a, b)).exp()
        }
    }
}

fn brcmp1(mu: f64, a: f64, b: f64, x: f64, y: f64, give_log: bool) -> f64 {
    let a0 = a.min(b);
    if a0 < 8.0 {
        let (lnx, lny);
        if x <= 0.375 {
            lnx = x.ln();
            lny = alnrel(-x);
        } else if y > 0.375 {
            lnx = x.ln();
            lny = y.ln();
        } else {
            lnx = alnrel(-y);
            lny = y.ln();
        }
        let mut z = a * lnx + b * lny;
        if a0 >= 1.0 {
            z -= betaln(a, b);
            return esum(mu, z, give_log);
        }
        let mut b0 = a.max(b);
        if b0 >= 8.0 {
            let u = gamln1(a0) + algdiv(a0, b0);
            return if give_log {
                a0.ln() + esum(mu, z - u, true)
            } else {
                a0 * esum(mu, z - u, false)
            };
        } else if b0 <= 1.0 {
            let ans = esum(mu, z, give_log);
            if ans == (if give_log { f64::NEG_INFINITY } else { 0.0 }) {
                return ans;
            }
            let apb = a + b;
            let zz;
            if apb > 1.0 {
                zz = (gam1(apb - 1.0) + 1.0) / apb;
            } else {
                zz = gam1(apb) + 1.0;
            }
            let c = if give_log {
                gam1(a).ln_1p() + gam1(b).ln_1p() - zz.ln()
            } else {
                (gam1(a) + 1.0) * (gam1(b) + 1.0) / zz
            };
            return if give_log {
                ans + a0.ln() + c - (a0 / b0).ln_1p()
            } else {
                ans * (a0 * c) / (a0 / b0 + 1.0)
            };
        }
        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i64;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 1..=n {
                b0 += -1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }
        z -= u;
        b0 += -1.0;
        let apb = a0 + b0;
        let t;
        if apb > 1.0 {
            t = (gam1(apb - 1.0) + 1.0) / apb;
        } else {
            t = gam1(apb) + 1.0;
        }
        if give_log {
            a0.ln() + esum(mu, z, true) + gam1(b0).ln_1p() - t.ln()
        } else {
            a0 * esum(mu, z, false) * (gam1(b0) + 1.0) / t
        }
    } else {
        let const__: f64 = 0.398942280401433;
        let apb = a + b;
        let lambda = if apb.is_finite() {
            if a <= b { a - apb * x } else { apb * y - b }
        } else {
            a * y - b * x
        };
        let (h, x0, y0);
        if a > b {
            h = b / a;
            x0 = 1.0 / (h + 1.0);
            y0 = h / (h + 1.0);
        } else {
            h = a / b;
            x0 = h / (h + 1.0);
            y0 = 1.0 / (h + 1.0);
        }
        let lx0 = -(b / a).ln_1p();
        let mut e = -lambda / a;
        let u = if e.abs() > 0.6 { e - (x / x0).ln() } else { rlog1(e) };
        e = lambda / b;
        let v = if e.abs() > 0.6 { e - (y / y0).ln() } else { rlog1(e) };
        let z = esum(mu, -(a * u + b * v), give_log);
        if give_log {
            const__.ln() + (b.ln() + lx0) / 2.0 + z - bcorr(a, b)
        } else {
            const__ * (b * x0).sqrt() * z * (-bcorr(a, b)).exp()
        }
    }
}

fn grat_r(a: f64, x: f64, log_r: f64, eps: f64) -> f64 {
    if a * x == 0.0 {
        return if x <= a { (-log_r).exp() } else { 0.0 };
    } else if a == 0.5 {
        if x < 0.25 {
            let p = erf_(x.sqrt());
            return (0.5 - p + 0.5) * (-log_r).exp();
        }
        let sx = x.sqrt();
        return erfc1(1, sx) / sx * M_SQRT_PI;
    } else if x < 1.1 {
        let mut an = 3.0;
        let mut c = x;
        let mut sum = x / (a + 3.0);
        let tol = eps * 0.1 / (a + 1.0);
        loop {
            an += 1.0;
            c *= -(x / an);
            let t = c / (a + an);
            sum += t;
            if !(t.abs() > tol) {
                break;
            }
        }
        let j = a * x * ((sum / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0));
        let z = a * x.ln();
        let h = gam1(a);
        let g = h + 1.0;
        if (x >= 0.25 && (a < x / 2.59)) || (z > -0.13394) {
            let ll = rexpm1(z);
            let q = ((ll + 0.5 + 0.5) * j - ll) * g - h;
            if q <= 0.0 { 0.0 } else { q * (-log_r).exp() }
        } else {
            let p = z.exp() * g * (0.5 - j + 0.5);
            (0.5 - p + 0.5) * (-log_r).exp()
        }
    } else {
        let mut a2n_1 = 1.0;
        let mut a2n = 1.0;
        let mut b2n_1 = x;
        let mut b2n = x + (1.0 - a);
        let mut c = 1.0;
        let mut an0;
        loop {
            a2n_1 = x * a2n + c * a2n_1;
            b2n_1 = x * b2n + c * b2n_1;
            let am0 = a2n_1 / b2n_1;
            c += 1.0;
            let c_a = c - a;
            a2n = a2n_1 + c_a * a2n;
            b2n = b2n_1 + c_a * b2n;
            an0 = a2n / b2n;
            if !((an0 - am0).abs() >= eps * an0) {
                break;
            }
        }
        an0
    }
}

fn logspace_add(logx: f64, logy: f64) -> f64 {
    logx.max(logy) + (-(logx - logy).abs()).exp().ln_1p()
}

fn bgrat(a: f64, b: f64, x: f64, y: f64, mut w: f64, eps: f64, log_w: bool) -> (f64, i32) {
    let n_terms = 30usize;
    let mut c = [0.0f64; 30];
    let mut d = [0.0f64; 30];
    let bm1 = b - 0.5 - 0.5;
    let nu = a + bm1 * 0.5;
    let lnx = if y > 0.375 { x.ln() } else { alnrel(-y) };
    let z = -nu * lnx;
    if b * z == 0.0 {
        return (w, 1);
    }
    let log_r = b.ln() + gam1(b).ln_1p() + b * z.ln() + nu * lnx;
    let log_u = log_r - (algdiv(b, a) + b * nu.ln());
    let u = log_u.exp();
    if log_u == f64::NEG_INFINITY {
        return (w, 2);
    }
    let u_0 = u == 0.0;
    let ll = if log_w {
        if w == f64::NEG_INFINITY { 0.0 } else { (w - log_u).exp() }
    } else if w == 0.0 {
        0.0
    } else {
        (w.ln() - log_u).exp()
    };
    let q_r = grat_r(b, z, log_r, eps);
    let v = 0.25 / (nu * nu);
    let t2 = lnx * 0.25 * lnx;
    let mut j = q_r;
    let mut sum = j;
    let mut t = 1.0;
    let mut cn = 1.0;
    let mut n2 = 0.0;
    let mut ierr = 0;
    for n in 1..=n_terms {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        let nm1 = n - 1;
        c[nm1] = cn;
        let mut s = 0.0;
        if n > 1 {
            let mut coef = b - n as f64;
            for i in 1..=nm1 {
                s += coef * c[i - 1] * d[nm1 - i];
                coef += b;
            }
        }
        d[nm1] = bm1 * cn + s / n as f64;
        let dj = d[nm1] * j;
        sum += dj;
        if sum <= 0.0 {
            return (w, 3);
        }
        if dj.abs() <= eps * (sum + ll) {
            ierr = 0;
            break;
        } else if n == n_terms {
            ierr = 4;
        }
    }
    if log_w {
        w = logspace_add(w, log_u + sum.ln());
    } else {
        w += if u_0 { (log_u + sum.ln()).exp() } else { u * sum };
    }
    (w, ierr)
}

fn basym(a: f64, b: f64, lambda: f64, eps: f64, log_p: bool) -> f64 {
    let num_it = 20usize;
    let e0 = 1.12837916709551;
    let e1 = 0.353553390593274;
    let ln_e0 = 0.120782237635245;
    let mut a0 = [0.0f64; 21];
    let mut b0 = [0.0f64; 21];
    let mut c = [0.0f64; 21];
    let mut d = [0.0f64; 21];
    let f = a * rlog1(-lambda / a) + b * rlog1(lambda / b);
    let t;
    if log_p {
        t = -f;
    } else {
        t = (-f).exp();
        if t == 0.0 {
            return 0.0;
        }
    }
    let z0 = f.sqrt();
    let z = z0 / e1 * 0.5;
    let z2 = f + f;
    let (h, r0, r1, w0);
    if a < b {
        h = a / b;
        r0 = 1.0 / (h + 1.0);
        r1 = (b - a) / b;
        w0 = 1.0 / (a * (h + 1.0)).sqrt();
    } else {
        h = b / a;
        r0 = 1.0 / (h + 1.0);
        r1 = (b - a) / a;
        w0 = 1.0 / (b * (h + 1.0)).sqrt();
    }
    a0[0] = r1 * 0.66666666666666663;
    c[0] = a0[0] * -0.5;
    d[0] = -c[0];
    let mut j0 = 0.5 / e0 * erfc1(1, z0);
    let mut j1 = e1;
    let mut sum = j0 + d[0] * w0 * j1;
    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let mut w = w0;
    let mut znm1 = z;
    let mut zn = z2;
    let mut n = 2usize;
    while n <= num_it {
        hn *= h2;
        a0[n - 1] = r0 * 2.0 * (h * hn + 1.0) / (n as f64 + 2.0);
        let np1 = n + 1;
        s += hn;
        a0[np1 - 1] = r1 * 2.0 * s / (n as f64 + 3.0);
        for i in n..=np1 {
            let r = (i as f64 + 1.0) * -0.5;
            b0[0] = r * a0[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for jj in 1..m {
                    let mmj = m - jj;
                    bsum += (jj as f64 * r - mmj as f64) * a0[jj - 1] * b0[mmj - 1];
                }
                b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
            }
            c[i - 1] = b0[i - 1] / (i as f64 + 1.0);
            let mut dsum = 0.0;
            for jj in 1..i {
                dsum += d[i - jj - 1] * c[jj - 1];
            }
            d[i - 1] = -(dsum + c[i - 1]);
        }
        j0 = e1 * znm1 + (n as f64 - 1.0) * j0;
        j1 = e1 * zn + n as f64 * j1;
        znm1 = z2 * znm1;
        zn = z2 * zn;
        w *= w0;
        let t0 = d[n - 1] * w * j0;
        w *= w0;
        let t1 = d[np1 - 1] * w * j1;
        sum += t0 + t1;
        if t0.abs() + t1.abs() <= eps * sum {
            break;
        }
        n += 2;
    }
    if log_p {
        return ln_e0 + t - bcorr(a, b) + sum.ln();
    }
    let u = (-bcorr(a, b)).exp();
    e0 * t * u * sum
}

#[inline]
/// `R_Log1_Exp` as redefined *inside* toms708.c (its lines 46-47 `#undef` the
/// dpq.h macro and re-`#define` it to call the file-local `rexpm1` instead of
/// libm `expm1`). Every `R_Log1_Exp` reached from `bratio` is this variant; it
/// differs from `gamma::r_log1_exp` by ~1 ulp on the `x > -M_LN2` branch, which
/// the `log_p` beta tails expose.
fn log1_exp_rexpm1(x: f64) -> f64 {
    if x > -M_LN2 {
        (-rexpm1(x)).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

fn br_end(wv: f64, w1v: f64, do_swap: bool, ierr: i32) -> (f64, f64, i32) {
    if do_swap {
        (w1v, wv, ierr)
    } else {
        (wv, w1v, ierr)
    }
}

fn br_end_from_w1(w1v: f64, do_swap: bool, log_p: bool, ierr: i32) -> (f64, f64, i32) {
    let (w, w1);
    if log_p {
        w = (-w1v).ln_1p();
        w1 = w1v.ln();
    } else {
        w = 0.5 - w1v + 0.5;
        w1 = w1v;
    }
    br_end(w, w1, do_swap, ierr)
}

fn br_end_from_w(wv: f64, do_swap: bool, log_p: bool, ierr: i32) -> (f64, f64, i32) {
    let (w, w1);
    if log_p {
        w1 = (-wv).ln_1p();
        w = wv.ln();
    } else {
        w = wv;
        w1 = 0.5 - wv + 0.5;
    }
    br_end(w, w1, do_swap, ierr)
}

fn br_end_from_w1_log(w1v: f64, do_swap: bool, log_p: bool, ierr: i32) -> (f64, f64, i32) {
    let (w, w1);
    if log_p {
        w = log1_exp_rexpm1(w1v);
        w1 = w1v;
    } else {
        w = -(w1v.exp_m1());
        w1 = w1v.exp();
    }
    br_end(w, w1, do_swap, ierr)
}

fn bratio(a: f64, b: f64, x: f64, y: f64, log_p: bool) -> (f64, f64, i32) {
    let rd0 = if log_p { f64::NEG_INFINITY } else { 0.0 };
    let rd1 = if log_p { 0.0 } else { 1.0 };
    let mut eps = TOMS_EPS;
    let w = rd0;
    let w1 = rd0;
    if x.is_nan() || y.is_nan() || a.is_nan() || b.is_nan() {
        return (w, w1, 9);
    }
    if a < 0.0 || b < 0.0 {
        return (w, w1, 1);
    }
    if a == 0.0 && b == 0.0 {
        return (w, w1, 2);
    }
    if x < 0.0 || x > 1.0 {
        return (w, w1, 3);
    }
    if y < 0.0 || y > 1.0 {
        return (w, w1, 4);
    }
    let z = x + y - 0.5 - 0.5;
    if z.abs() > eps * 3.0 {
        return (w, w1, 5);
    }
    if x == 0.0 {
        return if a == 0.0 { (w, w1, 6) } else { (rd0, rd1, 0) };
    }
    if y == 0.0 {
        return if b == 0.0 { (w, w1, 7) } else { (rd1, rd0, 0) };
    }
    if a == 0.0 {
        return (rd1, rd0, 0);
    }
    if b == 0.0 {
        return (rd0, rd1, 0);
    }
    eps = eps.max(1e-15);
    let a_lt_b = a < b;
    if (if a_lt_b { b } else { a }) < eps * 0.001 {
        let (wv, w1v);
        if log_p {
            if a_lt_b {
                wv = (-a / (a + b)).ln_1p();
                w1v = (a / (a + b)).ln();
            } else {
                wv = (b / (a + b)).ln();
                w1v = (-b / (a + b)).ln_1p();
            }
        } else {
            wv = b / (a + b);
            w1v = a / (a + b);
        }
        return (wv, w1v, 0);
    }

    let mut ierr = 0;

    if a.min(b) <= 1.0 {
        let do_swap = x > 0.5;
        let (a0, x0, mut b0, y0);
        if do_swap {
            a0 = b;
            x0 = y;
            b0 = a;
            y0 = x;
        } else {
            a0 = a;
            x0 = x;
            b0 = b;
            y0 = y;
        }
        if b0 < eps.min(eps * a0) {
            let wv = fpser(a0, b0, x0, eps, log_p);
            let w1v = if log_p { log1_exp_rexpm1(wv) } else { 0.5 - wv + 0.5 };
            return br_end(wv, w1v, do_swap, ierr);
        }
        if a0 < eps.min(eps * b0) && b0 * x0 <= 1.0 {
            let w1v = apser(a0, b0, x0, eps);
            return br_end_from_w1(w1v, do_swap, log_p, ierr);
        }
        let mut did_bup = false;
        let mut go_bpser_w = false;
        let mut go_bpser_w1 = false;
        let mut do_l131 = false;
        let mut n = 20;
        if a0.max(b0) > 1.0 {
            if b0 <= 1.0 {
                go_bpser_w = true;
            } else if x0 >= 0.29 {
                go_bpser_w1 = true;
            } else if x0 < 0.1 && (x0 * b0).powf(a0) <= 0.7 {
                go_bpser_w = true;
            } else if b0 > 15.0 {
                do_l131 = true;
            }
        } else if a0 >= 0.2_f64.min(b0) {
            go_bpser_w = true;
        } else if x0.powf(a0) <= 0.9 {
            go_bpser_w = true;
        } else if x0 >= 0.3 {
            go_bpser_w1 = true;
        }
        if go_bpser_w {
            let wv = bpser(a0, b0, x0, eps, log_p);
            let w1v = if log_p { log1_exp_rexpm1(wv) } else { 0.5 - wv + 0.5 };
            return br_end(wv, w1v, do_swap, ierr);
        }
        if go_bpser_w1 {
            let w1v = bpser(b0, a0, y0, eps, log_p);
            let wv = if log_p { log1_exp_rexpm1(w1v) } else { 0.5 - w1v + 0.5 };
            return br_end(wv, w1v, do_swap, ierr);
        }
        let mut w1v;
        if do_l131 {
            w1v = 0.0;
        } else {
            n = 20;
            w1v = bup(b0, a0, y0, x0, n, eps, false);
            did_bup = true;
            b0 += n as f64;
        }
        // L131
        let (w1r, ierr1) = bgrat(b0, a0, y0, x0, w1v, 15.0 * eps, false);
        w1v = w1r;
        if w1v == 0.0 || (0.0 < w1v && w1v < DBL_MIN) {
            if did_bup {
                w1v = bup(b0 - n as f64, a0, y0, x0, n, eps, true);
            } else {
                w1v = f64::NEG_INFINITY;
            }
            let (w1r2, ierr1b) = bgrat(b0, a0, y0, x0, w1v, 15.0 * eps, true);
            if ierr1b != 0 {
                ierr = 10 + ierr1b;
            }
            return br_end_from_w1_log(w1r2, do_swap, log_p, ierr);
        }
        if ierr1 != 0 {
            ierr = 10 + ierr1;
        }
        br_end_from_w1(w1v, do_swap, log_p, ierr)
    } else {
        let lambda = if (a + b).is_finite() {
            if a > b { (a + b) * y - b } else { a - (a + b) * x }
        } else {
            a * y - b * x
        };
        let do_swap = lambda < 0.0;
        let mut lam = lambda;
        let (a0, x0, mut b0, y0);
        if do_swap {
            lam = -lambda;
            a0 = b;
            x0 = y;
            b0 = a;
            y0 = x;
        } else {
            a0 = a;
            x0 = x;
            b0 = b;
            y0 = y;
        }
        let mut a0m = a0;
        let mut go_bpser_w = false;
        let mut go_bfrac = false;
        let mut go_l140 = false;
        if b0 < 40.0 {
            if b0 * x0 <= 0.7 || (log_p && lam > 650.0) {
                go_bpser_w = true;
            } else {
                go_l140 = true;
            }
        } else if a0m > b0 {
            if b0 <= 100.0 || lam > b0 * 0.03 {
                go_bfrac = true;
            }
        } else if a0m <= 100.0 {
            go_bfrac = true;
        } else if lam > a0m * 0.03 {
            go_bfrac = true;
        }
        if go_bpser_w {
            let wv = bpser(a0m, b0, x0, eps, log_p);
            let w1v = if log_p { log1_exp_rexpm1(wv) } else { 0.5 - wv + 0.5 };
            return br_end(wv, w1v, do_swap, ierr);
        }
        if go_bfrac {
            let wv = bfrac(a0m, b0, x0, y0, lam, eps * 15.0, log_p);
            let w1v = if log_p { log1_exp_rexpm1(wv) } else { 0.5 - wv + 0.5 };
            return br_end(wv, w1v, do_swap, ierr);
        }
        if go_l140 {
            let mut nn = b0 as i64;
            b0 -= nn as f64;
            if b0 == 0.0 {
                nn -= 1;
                b0 = 1.0;
            }
            let mut wv = bup(b0, a0m, y0, x0, nn as i32, eps, false);
            if wv < DBL_MIN && log_p {
                b0 += nn as f64;
                let wv2 = bpser(a0m, b0, x0, eps, log_p);
                let w1v = if log_p { log1_exp_rexpm1(wv2) } else { 0.5 - wv2 + 0.5 };
                return br_end(wv2, w1v, do_swap, ierr);
            }
            if x0 <= 0.7 {
                wv += bpser(a0m, b0, x0, eps, false);
                return br_end_from_w(wv, do_swap, log_p, ierr);
            }
            if a0m <= 15.0 {
                let n2 = 20;
                wv += bup(a0m, b0, x0, y0, n2, eps, false);
                a0m += n2 as f64;
            }
            let (wv3, ierr1) = bgrat(a0m, b0, x0, y0, wv, 15.0 * eps, false);
            if ierr1 != 0 {
                ierr = 10 + ierr1;
            }
            return br_end_from_w(wv3, do_swap, log_p, ierr);
        }
        // L180 — basym
        let wv = basym(a0m, b0, lam, eps * 100.0, log_p);
        let w1v = if log_p { log1_exp_rexpm1(wv) } else { 0.5 - wv + 0.5 };
        br_end(wv, w1v, do_swap, ierr)
    }
}

pub(crate) fn pbeta_raw(x: f64, a: f64, b: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x >= 1.0 {
        return dt1(lower_tail, log_p);
    }
    if a == 0.0 || b == 0.0 || !a.is_finite() || !b.is_finite() {
        if a == 0.0 && b == 0.0 {
            return if log_p { -M_LN2 } else { 0.5 };
        }
        if a == 0.0 || a / b == 0.0 {
            return dt1(lower_tail, log_p);
        }
        if b == 0.0 || b / a == 0.0 {
            return dt0(lower_tail, log_p);
        }
        return if x < 0.5 {
            dt0(lower_tail, log_p)
        } else {
            dt1(lower_tail, log_p)
        };
    }
    if x <= 0.0 {
        return dt0(lower_tail, log_p);
    }
    let x1 = 0.5 - x + 0.5;
    let (w, wc, _ierr) = bratio(a, b, x, x1, log_p);
    if lower_tail { w } else { wc }
}

pub(crate) fn pbeta_scalar(x: f64, a: f64, b: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return x + a + b;
    }
    if a < 0.0 || b < 0.0 {
        return f64::NAN;
    }
    pbeta_raw(x, a, b, lower_tail, log_p)
}

pub(crate) fn lbeta_scalar(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return a + b;
    }
    let mut p = a;
    let mut q = a;
    if b < p {
        p = b;
    }
    if b > q {
        q = b;
    }
    if p < 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if !q.is_finite() {
        return f64::NEG_INFINITY;
    }
    if p >= 10.0 {
        let corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q);
        return q.ln() * -0.5 + M_LN_SQRT_2PI + corr + (p - 0.5) * (p / (p + q)).ln()
            + q * (-p / (p + q)).ln_1p();
    } else if q >= 10.0 {
        let corr = lgammacor(q) - lgammacor(p + q);
        return lgammafn(p) + corr + p - p * (p + q).ln() + (q - 0.5) * (-p / (p + q)).ln_1p();
    }
    if p < 1e-306 {
        return lgammafn(p) + (lgammafn(q) - lgammafn(p + q));
    }
    (gammafn(p) * (gammafn(q) / gammafn(p + q))).ln()
}

// === PyO3 wrappers ===========================================================

#[pyfunction]
#[pyo3(name = "pbeta", signature = (x, a, b, lower_tail=true, log_p=false))]
pub fn pbeta<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let (xv, av, bv) = (x.as_array(), a.as_array(), b.as_array());
    let mut out = Vec::with_capacity(xv.len());
    for i in 0..xv.len() {
        out.push(pbeta_scalar(xv[i], av[i], bv[i], lower_tail, log_p));
    }
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "lbeta", signature = (a, b))]
pub fn lbeta<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let (av, bv) = (a.as_array(), b.as_array());
    let mut out = Vec::with_capacity(av.len());
    for i in 0..av.len() {
        out.push(lbeta_scalar(av[i], bv[i]));
    }
    out.into_pyarray(py)
}
