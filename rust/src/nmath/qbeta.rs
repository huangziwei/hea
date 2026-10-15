#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)] // loop-carried locals mirror the C `goto` state vars

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::consts::{DBL_MIN, M_LN2};
use super::gamma::{r_dt_clog, r_dt_log, r_dt_qiv, r_log1_exp};
use super::norm::{dt0, dt1};
use super::toms708::{lbeta_scalar, pbeta_raw};
use super::util::rfma;

const DBL_VERY_MIN: f64 = 2.2250738585072014e-308 / 4.0;
const DBL_LOG_V_MIN: f64 = M_LN2 * (-1021.0 - 2.0);

#[inline]
fn r_dt_civ(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            -p.exp_m1()
        } else {
            p.exp()
        }
    } else if lower_tail {
        0.5 - p + 0.5
    } else {
        p
    }
}

#[inline]
fn r_pow_di3(x: f64) -> f64 {
    x * (x * x)
}

#[inline]
fn clog(x: f64) -> f64 {
    if x > 0.0 {
        x.ln()
    } else if x == 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::NAN
    }
}

pub(crate) fn qbeta_raw(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> (f64, f64) {
    let log_q_cut: f64 = -5.0;
    let n_n = 4;
    let give_log_q = false;
    let mut use_log_x = give_log_q;
    let fpu: f64 = 3e-308;
    let acu_min: f64 = 1e-300;
    let p_lo: f64 = fpu;
    let p_hi: f64 = 1.0 - 2.22e-16;
    let (const1, const2, const3, const4) = (2.30753, 0.27061, 0.99229, 0.04481);

    if alpha == dt0(lower_tail, log_p) {
        return (0.0, 1.0);
    }
    if alpha == dt1(lower_tail, log_p) {
        return (1.0, 0.0);
    }
    if (log_p && alpha > 0.0) || (!log_p && (alpha < 0.0 || alpha > 1.0)) {
        return (f64::NAN, f64::NAN);
    }
    if p == 0.0 || q == 0.0 || !p.is_finite() || !q.is_finite() {
        let rdh = if log_p { -M_LN2 } else { 0.5 };
        if p == 0.0 && q == 0.0 {
            if alpha < rdh {
                return (0.0, 1.0);
            }
            if alpha > rdh {
                return (1.0, 0.0);
            }
            return (0.5, 0.5);
        } else if p == 0.0 || p / q == 0.0 {
            return (0.0, 1.0);
        } else if q == 0.0 || q / p == 0.0 {
            return (1.0, 0.0);
        }
        return (0.5, 0.5);
    }

    let p_ = r_dt_qiv(alpha, lower_tail, log_p);
    let logbeta = lbeta_scalar(p, q);
    let mut swap_tail = p_ > 0.5;
    let log_eps_c = M_LN2 * (1.0 - 53.0);

    let mut y = -1.0;
    let mut u_n = 1.0;
    let mut add_n_step = true;
    let mut n_maybe_swaps = 0;
    let mut goto_return = false;
    let mut converged = false;
    let mut tx = 0.0;
    let mut a = 0.0;
    let mut la = 0.0;
    let mut pp = 0.0;
    let mut qq = 0.0;
    let mut u = 0.0;
    let mut xinbta = 0.0;

    loop {
        if swap_tail {
            a = r_dt_civ(alpha, lower_tail, log_p);
            la = r_dt_clog(alpha, lower_tail, log_p);
            pp = q;
            qq = p;
        } else {
            a = p_;
            la = r_dt_log(alpha, lower_tail, log_p);
            pp = p;
            qq = q;
        }
        n_maybe_swaps += 1;
        let acu = acu_min.max(10f64.powf(-13.0 - 2.5 / (pp * pp) - 0.5 / (a * a)));
        let u0 = (la + pp.ln() + logbeta) / pp;
        let mut rp = pp * (1.0 - qq) / (pp + 1.0);
        let mut t = 0.2;
        let u0_maybe = M_LN2 * (-1021.0) < u0 && u0 < -0.01;
        u_n = 1.0;
        let mut skip_init = false;
        if u0_maybe
            && u0
                < rfma(
                    t,
                    log_eps_c,
                    -clog((pp * (1.0 - qq) * (2.0 - qq) / (2.0 * (pp + 2.0))).abs()),
                ) / 2.0
        {
            rp *= u0.exp();
            u = if rp > -1.0 { u0 - rp.ln_1p() / pp } else { u0 };
            tx = u.exp();
            xinbta = tx;
            use_log_x = true;
            skip_init = true;
        }

        if !skip_init {
            let mut r = (-2.0 * la).sqrt();
            y = r - rfma(const2, r, const1) / rfma(rfma(const4, r, const3), r, 1.0);
            if pp > 1.0 && qq > 1.0 {
                r = rfma(y, y, -3.0) / 6.0;
                let s = 1.0 / (pp + pp - 1.0);
                t = 1.0 / (qq + qq - 1.0);
                let h = 2.0 / (s + t);
                let w = rfma(
                    -(t - s),
                    r + 5.0 / 6.0 - 2.0 / (3.0 * h),
                    y * (h + r).sqrt() / h,
                );
                if w > 300.0 {
                    t = w + w + qq.ln() - pp.ln();
                    u = if t <= 18.0 {
                        -(t.exp()).ln_1p()
                    } else {
                        -t - (-t).exp()
                    };
                    xinbta = u.exp();
                } else {
                    xinbta = pp / rfma(qq, (w + w).exp(), pp);
                    u = -(qq / pp * (w + w).exp()).ln_1p();
                }
            } else {
                r = qq + qq;
                t = 1.0 / (3.0 * qq.sqrt());
                t = r * r_pow_di3(rfma(t, -t + y, 1.0));
                let s = rfma(4.0, pp, r) - 2.0;
                if t == 0.0 || (t < 0.0 && s >= t) {
                    let l1ma = if swap_tail {
                        r_dt_log(alpha, lower_tail, log_p)
                    } else {
                        r_dt_clog(alpha, lower_tail, log_p)
                    };
                    let xx = (l1ma + qq.ln() + logbeta) / qq;
                    if xx <= 0.0 {
                        xinbta = -xx.exp_m1();
                        u = r_log1_exp(xx);
                    } else {
                        let r_ = rp * u0.exp();
                        u = if r_ > -1.0 { u0 - r_.ln_1p() / pp } else { u0 };
                        xinbta = u.exp();
                    }
                } else {
                    t = s / t;
                    if t <= 1.0 {
                        u = u0;
                        xinbta = u.exp();
                    } else {
                        xinbta = 1.0 - 2.0 / (t + 1.0);
                        u = (-2.0 / (t + 1.0)).ln_1p();
                    }
                }
            }

            if (swap_tail && u >= -log_q_cut.exp())
                || (!swap_tail && u >= -(4.0 * log_q_cut).exp() && pp / qq < 1000.0)
            {
                swap_tail = !swap_tail;
                if swap_tail {
                    a = r_dt_civ(alpha, lower_tail, log_p);
                    la = r_dt_clog(alpha, lower_tail, log_p);
                    pp = q;
                    qq = p;
                } else {
                    a = p_;
                    la = r_dt_log(alpha, lower_tail, log_p);
                    pp = p;
                    qq = q;
                }
                u = r_log1_exp(u);
                xinbta = u.exp();
            }

            if !use_log_x {
                use_log_x = u < log_q_cut;
            }
            let bad_u = !u.is_finite();
            let bad_init = bad_u || xinbta > p_hi;
            tx = xinbta;
            if bad_u || u < log_q_cut {
                let w = pbeta_raw(DBL_VERY_MIN, pp, qq, true, log_p);
                if w > (if log_p { la } else { a }) {
                    if log_p || (w - a).abs() < (0.0 - a).abs() {
                        tx = DBL_VERY_MIN;
                        u_n = DBL_LOG_V_MIN;
                    } else {
                        tx = 0.0;
                        u_n = f64::NEG_INFINITY;
                    }
                    use_log_x = log_p;
                    add_n_step = false;
                    goto_return = true;
                } else if u < DBL_LOG_V_MIN {
                    u = DBL_LOG_V_MIN;
                    xinbta = DBL_VERY_MIN;
                }
            }
            if !goto_return && bad_init && !(use_log_x && tx > 0.0) {
                if u == f64::NEG_INFINITY {
                    u = M_LN2 * (-1021.0);
                    xinbta = DBL_MIN;
                } else {
                    xinbta = if xinbta > 1.1 {
                        0.5
                    } else if xinbta < p_lo {
                        u.exp()
                    } else {
                        p_hi
                    };
                    if bad_u {
                        u = xinbta.ln();
                    }
                }
            }
        }

        if goto_return {
            break;
        }

        let r = 1.0 - pp;
        let t = 1.0 - qq;
        let mut wprev: f64 = 0.0;
        let mut prev: f64 = 1.0;
        let mut adj: f64 = 1.0;
        let mut jump_swap = false;
        if use_log_x {
            for i_pb in 0..1000 {
                y = pbeta_raw(xinbta, pp, qq, true, true);
                let w = if y == f64::NEG_INFINITY {
                    0.0
                } else {
                    (y - la) * rfma(t, r_log1_exp(u), rfma(r, u, y - u + logbeta)).exp()
                };
                if !w.is_finite() {
                    if n_maybe_swaps <= 1 {
                        jump_swap = true;
                        break;
                    }
                    return (f64::NAN, f64::NAN);
                }
                if i_pb >= n_n && w * wprev <= 0.0 {
                    prev = adj.abs().max(fpu);
                }
                let mut g = 1.0;
                for _ in 0..1000 {
                    adj = g * w;
                    if adj.abs() < prev {
                        u_n = u - adj;
                        if u_n <= 0.0 {
                            if prev <= acu || w.abs() <= acu {
                                converged = true;
                            }
                            break;
                        }
                    }
                    g /= 3.0;
                }
                if converged {
                    break;
                }
                let dd = adj.abs().min((u_n - u).abs());
                if dd <= 4e-16 * (u_n + u).abs() {
                    converged = true;
                    break;
                }
                u = u_n;
                xinbta = u.exp();
                wprev = w;
            }
        } else {
            for i_pb in 0..1000 {
                y = pbeta_raw(xinbta, pp, qq, true, log_p);
                let w = if log_p {
                    (y - la) * rfma(t, (-xinbta).ln_1p(), rfma(r, xinbta.ln(), y + logbeta)).exp()
                } else {
                    (y - a) * rfma(t, (-xinbta).ln_1p(), rfma(r, xinbta.ln(), logbeta)).exp()
                };
                if !w.is_finite() {
                    if n_maybe_swaps <= 2 {
                        if !log_p && n_maybe_swaps == 2 {
                            use_log_x = true;
                        }
                        if !log_p || n_maybe_swaps <= 1 {
                            jump_swap = true;
                            break;
                        }
                    }
                    return (f64::NAN, f64::NAN);
                }
                if i_pb >= n_n && w * wprev <= 0.0 {
                    prev = adj.abs().max(fpu);
                }
                let mut g = 1.0;
                for _ in 0..1000 {
                    adj = g * w;
                    if i_pb < n_n || adj.abs() < prev {
                        tx = xinbta - adj;
                        if (0.0..=1.0).contains(&tx) {
                            if prev <= acu || w.abs() <= acu {
                                converged = true;
                                break;
                            }
                            if tx != 0.0 && tx != 1.0 {
                                break;
                            }
                        }
                    }
                    g /= 3.0;
                }
                if converged {
                    break;
                }
                if (tx - xinbta).abs() <= 4e-16 * (tx + xinbta) {
                    converged = true;
                    break;
                }
                xinbta = tx;
                if tx == 0.0 {
                    break;
                }
                wprev = w;
            }
        }
        if jump_swap {
            continue;
        }
        break;
    }

    if !goto_return {
        let log_ = log_p || use_log_x;
        if (log_ && y == f64::NEG_INFINITY) || (!log_ && y == 0.0) {
            let w = pbeta_raw(DBL_VERY_MIN, pp, qq, true, log_);
            if log_ || (w - a).abs() <= (y - a).abs() {
                tx = DBL_VERY_MIN;
                u_n = DBL_LOG_V_MIN;
            }
            add_n_step = false;
        }
    }
    let r = 1.0 - pp;
    let t = 1.0 - qq;
    if use_log_x {
        if add_n_step {
            if u_n != 1.0 {
                xinbta = u_n.exp();
            }
            y = pbeta_raw(xinbta, pp, qq, true, log_p);
            let w = if log_p {
                (y - la) * rfma(t, (-xinbta).ln_1p(), rfma(r, xinbta.ln(), y + logbeta)).exp()
            } else {
                (y - a) * rfma(t, (-xinbta).ln_1p(), rfma(r, xinbta.ln(), logbeta)).exp()
            };
            tx = if w.is_finite() { xinbta - w } else { xinbta };
        } else if swap_tail {
            return (-u_n.exp_m1(), u_n.exp());
        } else {
            return (u_n.exp(), -u_n.exp_m1());
        }
    }
    if swap_tail {
        (1.0 - tx, tx)
    } else {
        (tx, 1.0 - tx)
    }
}

pub(crate) fn qbeta_scalar(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || q.is_nan() || alpha.is_nan() {
        return p + q + alpha;
    }
    if p < 0.0 || q < 0.0 {
        return f64::NAN;
    }
    qbeta_raw(alpha, p, q, lower_tail, log_p).0
}

#[pyfunction]
#[pyo3(name = "qbeta", signature = (alpha, p, q, lower_tail=true, log_p=false))]
pub fn qbeta<'py>(
    py: Python<'py>,
    alpha: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    lower_tail: bool,
    log_p: bool,
) -> Bound<'py, PyArray1<f64>> {
    let out = crate::par::map3(
        py,
        alpha.as_slice().unwrap(),
        p.as_slice().unwrap(),
        q.as_slice().unwrap(),
        |al, p, q| qbeta_scalar(al, p, q, lower_tail, log_p),
    );
    out.into_pyarray(py)
}
