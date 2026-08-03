//! Cox proportional-hazards partial-likelihood kernel — mechanical port of
//! mgcv's `coxlpl` (src/coxph.c:141), the `deriv<=0` path (log partial
//! likelihood `lpl`, its coefficient gradient `g`, and Hessian `H`).
//!
//! mgcv sweeps the rows back in time keeping RUNNING risk-set accumulators —
//! `gamma_p` (scalar), `b_p` (p-vector), `A_p` (p×p) — and emits each unique
//! event time's contribution to `lpl`/`g`/`H` as it goes. The numpy port
//! (`hea/family.py::_coxlpl`) instead materialises the full (n,p,p) `gXX` and
//! its cumulative sum, an O(n·p²) temporary that dominates the cox fit (it is
//! re-evaluated on every line-search likelihood call). This kernel keeps the
//! single pass: O(p²) memory, no (n,p,p) array.
//!
//! Rows arrive sorted into DESCENDING time (the caller's `np.argsort(-time)`),
//! with `r[i]` the 0-based unique-time group of row `i` (non-decreasing) and
//! `nt` the number of groups. The accumulation order AND the per-statement
//! multiply/divide order + FMA contraction mirror the C verbatim: coxph.c is
//! `clang -O2`, which fuses single-expression `a*b+c` to `fmadd` on arm64, so
//! every `acc += a*b` (the `b_p`/`A_p`/`d1*` accumulators) and `c - a*b` (the
//! `lpl`/`g`/`H` emits) uses `rfma` (per-arch). The Hessian emit keeps C's
//! mult-THEN-div order `-dr*A_p/g + dr*b_k*b_m/(g*g)` — NOT a pre-divided
//! `inv=dr/g` (which reassociates the division and diverges on *both* arches).
//! Result equals mgcv bit-for-bit up to libm `exp`/`log`; it differs sub-ULP
//! from the numpy port only because numpy reduces pairwise, not sequentially.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::nmath::util::rfma;

/// `coxlpl` `deriv<0`: the log partial likelihood `lpl` only — the
/// line-search evaluation. The C guards `b_p`/`A_p`/`g`/`H` behind
/// `*deriv>=0`, so this is just the running `gamma_p` sweep (coxph.c:207-321),
/// O(n) and X-free. `eta` (n,), `d` (n,) event indicator, `r` (n,) 0-based
/// group per row, `nt` groups.
#[pyfunction]
#[pyo3(name = "cox_l")]
fn cox_l(
    eta: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray1<'_, i64>,
    r: PyReadonlyArray1<'_, i64>,
    nt: usize,
) -> f64 {
    let eta = eta.as_slice().unwrap();
    let d = d.as_slice().unwrap();
    let r = r.as_slice().unwrap();
    let n = eta.len();
    let mut lpl = 0.0f64;
    let mut gamma_p = 0.0f64;
    let mut i = 0usize;
    for j in 0..nt {
        let mut eta_sum = 0.0f64;
        let mut dr = 0.0f64;
        while i < n && r[i] == j as i64 {
            gamma_p += eta[i].exp();
            if d[i] == 1 {
                dr += 1.0;
                eta_sum += eta[i];
            }
            i += 1;
        }
        lpl += rfma(-dr, gamma_p.ln(), eta_sum); // eta_sum - dr*log(g) (coxph.c:321)
    }
    lpl
}

/// `coxlpl` with `deriv<0` semantics extended to also return `g`,`H` (i.e. the
/// C `deriv==0` branch): `(lpl, g[p], H[p,p])`. `eta` (n,), `x` (n,p) row-major,
/// `d` (n,) event indicator, `r` (n,) 0-based group per row, `nt` groups.
#[pyfunction]
#[pyo3(name = "cox_lpl0")]
fn cox_lpl0<'py>(
    py: Python<'py>,
    eta: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray1<'py, i64>,
    r: PyReadonlyArray1<'py, i64>,
    nt: usize,
) -> (f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let eta = eta.as_slice().unwrap();
    let (n, p) = {
        let a = x.as_array();
        (a.shape()[0], a.shape()[1])
    };
    let x_flat = x.as_slice().unwrap(); // (n,p) row-major (caller: ascontiguousarray)
    let d = d.as_slice().unwrap();
    let r = r.as_slice().unwrap();

    // gamma[i] = exp(eta[i])  (coxph.c:207)
    let gamma: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut lpl = 0.0f64;
    let mut gamma_p = 0.0f64;
    let mut b_p = vec![0.0f64; p];
    let mut a_p = vec![0.0f64; p * p]; // upper triangle (k<=m), row-major
    let mut g = vec![0.0f64; p];
    let mut h = vec![0.0f64; p * p];

    let mut i = 0usize;
    for j in 0..nt {
        // work back in time (coxph.c:266)
        let mut eta_sum = 0.0f64;
        let mut dr = 0.0f64;
        while i < n && r[i] == j as i64 {
            let gi = gamma[i];
            gamma_p += gi;
            let xi = &x_flat[i * p..i * p + p];
            let is_ev = d[i] == 1;
            if is_ev {
                dr += 1.0;
                eta_sum += eta[i];
            }
            for k in 0..p {
                b_p[k] = rfma(gi, xi[k], b_p[k]); // b_p[k] += gamma[i]*X (coxph.c:275)
            }
            if is_ev {
                for k in 0..p {
                    g[k] += xi[k];
                }
            }
            for k in 0..p {
                let xik = gi * xi[k]; // rounded gamma[i]*X[k] (coxph.c:278)
                for m in k..p {
                    a_p[k * p + m] = rfma(xik, xi[m], a_p[k * p + m]); // += ·X[m]
                }
            }
            i += 1;
        }
        // emit this event time's contribution (coxph.c:321-327)
        lpl += rfma(-dr, gamma_p.ln(), eta_sum); // eta_sum - dr*log(g)
        let inv = dr / gamma_p; // dr/gamma_p (coxph.c:323 g only)
        for k in 0..p {
            g[k] = rfma(-inv, b_p[k], g[k]); // g[k] += -dr/g*b_p[k]
        }
        for k in 0..p {
            for m in k..p {
                // coxph.c:325-326: mult-THEN-div, NOT a pre-divided inv (which
                // reassociates the division → diverges on both arches).
                h[k * p + m] +=
                    -dr * a_p[k * p + m] / gamma_p + dr * b_p[k] * b_p[m] / (gamma_p * gamma_p);
            }
        }
    }
    // symmetrize H (coxph.c:372)
    for k in 0..p {
        for m in 0..k {
            h[k * p + m] = h[m * p + k];
        }
    }

    (
        lpl,
        g.into_pyarray(py),
        Array2::from_shape_vec((p, p), h).unwrap().into_pyarray(py),
    )
}

/// `coxlpl` deriv 1/2: additionally the per-ρ first derivatives of `H`,
/// `d1H` (p,p,M). `d1gamma` (n,M) = (X·d1beta)·gamma is precomputed by the
/// caller (a cheap BLAS matmul); rust does the single-pass accumulation of the
/// running `d1b_p`(p,M), `d1gamma_p`(M), `d1A_p`(p,p,M) — never the (n,p,p,M)
/// `d1A_p` temporary the numpy port builds (coxph.c:282-340).
#[pyfunction]
#[pyo3(name = "cox_lpl_d1")]
fn cox_lpl_d1<'py>(
    py: Python<'py>,
    eta: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray1<'py, i64>,
    r: PyReadonlyArray1<'py, i64>,
    nt: usize,
    d1gamma: PyReadonlyArray2<'py, f64>,
) -> (
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
) {
    let eta = eta.as_slice().unwrap();
    let (n, p) = {
        let a = x.as_array();
        (a.shape()[0], a.shape()[1])
    };
    let x_flat = x.as_slice().unwrap(); // (n,p) row-major
    let d = d.as_slice().unwrap();
    let r = r.as_slice().unwrap();
    let mm = d1gamma.as_array().shape()[1]; // M = number of smoothing parameters
    let d1g = d1gamma.as_slice().unwrap(); // (n,M) row-major

    let gamma: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut lpl = 0.0f64;
    let mut gamma_p = 0.0f64;
    let mut b_p = vec![0.0f64; p];
    let mut a_p = vec![0.0f64; p * p]; // upper tri (k<=l), row-major k*p+l
    let mut g = vec![0.0f64; p];
    let mut h = vec![0.0f64; p * p];
    // first-derivative running accumulators
    let mut d1gamma_p = vec![0.0f64; mm];
    let mut d1b_p = vec![0.0f64; p * mm]; // [k*M+m]
    let mut d1a_p = vec![0.0f64; mm * p * p]; // [m*p*p + k*p + l], upper tri k<=l
    let mut d1h = vec![0.0f64; p * p * mm]; // output (p,p,M): [(k*p+l)*M + m]

    let mut i = 0usize;
    for j in 0..nt {
        let mut eta_sum = 0.0f64;
        let mut dr = 0.0f64;
        while i < n && r[i] == j as i64 {
            let gi = gamma[i];
            gamma_p += gi;
            let xi = &x_flat[i * p..i * p + p];
            let d1i = &d1g[i * mm..i * mm + mm];
            let is_ev = d[i] == 1;
            if is_ev {
                dr += 1.0;
                eta_sum += eta[i];
            }
            for k in 0..p {
                b_p[k] = rfma(gi, xi[k], b_p[k]); // += gamma[i]*X (coxph.c:275)
            }
            if is_ev {
                for k in 0..p {
                    g[k] += xi[k];
                }
            }
            for k in 0..p {
                let xik = gi * xi[k]; // rounded gamma[i]*X[k] (coxph.c:278)
                for l in k..p {
                    a_p[k * p + l] = rfma(xik, xi[l], a_p[k * p + l]);
                }
            }
            // first derivatives (coxph.c:282-288)
            for m in 0..mm {
                d1gamma_p[m] += d1i[m];
            }
            for m in 0..mm {
                let xx = d1i[m];
                for k in 0..p {
                    d1b_p[k * mm + m] = rfma(xx, xi[k], d1b_p[k * mm + m]);
                }
            }
            // first derivatives of A_p (coxph.c:301-306)
            for m in 0..mm {
                let xx = d1i[m];
                let base = m * p * p;
                for k in 0..p {
                    let xxk = xx * xi[k]; // rounded d1gamma*X[k]
                    for l in k..p {
                        d1a_p[base + k * p + l] = rfma(xxk, xi[l], d1a_p[base + k * p + l]);
                    }
                }
            }
            i += 1;
        }
        lpl += rfma(-dr, gamma_p.ln(), eta_sum); // eta_sum - dr*log(g)
        let inv = dr / gamma_p; // dr/gamma_p (coxph.c:323 g only)
        for k in 0..p {
            g[k] = rfma(-inv, b_p[k], g[k]); // g[k] += -dr/g*b_p[k]
        }
        for k in 0..p {
            for l in k..p {
                // coxph.c:325-326: mult-THEN-div, NOT a pre-divided inv.
                h[k * p + l] +=
                    -dr * a_p[k * p + l] / gamma_p + dr * b_p[k] * b_p[l] / (gamma_p * gamma_p);
            }
        }
        // first derivatives of H (coxph.c:337-338). Prefactors match C's
        // mult/div order; the multi-product body uses the EXACT clang -O2 arm64
        // fma tree (rule: at `P_left ± P_right` fuse the left product, round the
        // right as the addend; chain the rest) — verified bit-for-bit against the
        // compiled coxph.c emit (4000/4000 random inputs).
        let xx0 = dr / gamma_p;
        for m in 0..mm {
            let xx = d1gamma_p[m] * xx0 / gamma_p;
            let xx1 = xx0 / gamma_p;
            let xx2 = xx1 * 2.0 * d1gamma_p[m] / gamma_p;
            let base = m * p * p;
            for k in 0..p {
                for l in k..p {
                    // d1b[k,m]*b_l + b_k*d1b[l,m]  (fuse left product)
                    let pin = rfma(d1b_p[k * mm + m], b_p[l], b_p[k] * d1b_p[l * mm + m]);
                    let mut v = rfma(xx1, pin, -(xx2 * b_p[k] * b_p[l])); // xx1*pin - xx2*b_k*b_l
                    v = rfma(xx, a_p[k * p + l], v); // + xx*A_kl
                    v = rfma(-xx0, d1a_p[base + k * p + l], v); // - xx0*d1A
                    d1h[(k * p + l) * mm + m] += v;
                }
            }
        }
    }
    // symmetrize H (coxph.c:372) and each d1H slice (coxph.c:373-377)
    for k in 0..p {
        for l in 0..k {
            h[k * p + l] = h[l * p + k];
        }
    }
    for m in 0..mm {
        for k in 0..p {
            for l in 0..k {
                d1h[(k * p + l) * mm + m] = d1h[(l * p + k) * mm + m];
            }
        }
    }

    (
        lpl,
        g.into_pyarray(py),
        Array2::from_shape_vec((p, p), h).unwrap().into_pyarray(py),
        Array3::from_shape_vec((p, p, mm), d1h)
            .unwrap()
            .into_pyarray(py),
    )
}

/// `coxlpl` deriv 3, leading-diagonal second derivatives `d2H` (p, nhh) — the
/// eigenbasis `trHid2H` accumulation. `xp` is the eigenbasis design, `d1gamma`
/// (n,M) and `d2gamma` (n,nhh) the precomputed gamma derivatives; rust keeps
/// only the diagonals of `A_p`/`d1A_p` plus `b_p`/`d1b_p`/`d2b_p`/`d2ldA_p`
/// running, never the (n,nhh,p) temporaries (coxph.c:290-368). The caller does
/// `trHid2H = (d2H * dvec[:,None]).sum(0)`.
#[pyfunction]
#[pyo3(name = "cox_d2h")]
fn cox_d2h<'py>(
    py: Python<'py>,
    xp: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray1<'py, i64>,
    r: PyReadonlyArray1<'py, i64>,
    nt: usize,
    eta: PyReadonlyArray1<'py, f64>,
    d1gamma: PyReadonlyArray2<'py, f64>,
    d2gamma: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let (n, p) = {
        let a = xp.as_array();
        (a.shape()[0], a.shape()[1])
    };
    let x_flat = xp.as_slice().unwrap(); // (n,p) row-major
    let d = d.as_slice().unwrap();
    let r = r.as_slice().unwrap();
    let eta = eta.as_slice().unwrap();
    let mm = d1gamma.as_array().shape()[1];
    let d1g = d1gamma.as_slice().unwrap(); // (n,M)
    let nhh = d2gamma.as_array().shape()[1]; // M*(M+1)/2
    let d2g = d2gamma.as_slice().unwrap(); // (n,nhh)

    let gamma: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut gamma_p = 0.0f64;
    let mut b_p = vec![0.0f64; p];
    let mut adiag = vec![0.0f64; p]; // A_p[l,l]
    let mut d1gamma_p = vec![0.0f64; mm];
    let mut d1b_p = vec![0.0f64; p * mm]; // [l*M+m]
    let mut d1adiag = vec![0.0f64; p * mm]; // d1A_p[l,l,m] = [l*M+m]
    let mut d2gamma_p = vec![0.0f64; nhh];
    let mut d2b_p = vec![0.0f64; p * nhh]; // [l*nhh+off]
    let mut d2lda_p = vec![0.0f64; p * nhh]; // [l*nhh+off]
    let mut d2h = vec![0.0f64; p * nhh]; // output (p,nhh): [l*nhh+off]

    let mut i = 0usize;
    for j in 0..nt {
        let mut dr = 0.0f64;
        while i < n && r[i] == j as i64 {
            let gi = gamma[i];
            gamma_p += gi;
            let xi = &x_flat[i * p..i * p + p];
            let d1i = &d1g[i * mm..i * mm + mm];
            let d2i = &d2g[i * nhh..i * nhh + nhh];
            if d[i] == 1 {
                dr += 1.0;
            }
            for l in 0..p {
                b_p[l] = rfma(gi, xi[l], b_p[l]); // += gamma[i]*X[l] (coxph.c:275)
                let t = gi * xi[l]; // rounded gamma[i]*X[l]; A_p[l,l] += t*X[l] (:278)
                adiag[l] = rfma(t, xi[l], adiag[l]);
            }
            for m in 0..mm {
                d1gamma_p[m] += d1i[m];
                let xx = d1i[m];
                for l in 0..p {
                    d1b_p[l * mm + m] = rfma(xx, xi[l], d1b_p[l * mm + m]); // :284
                    let t = xx * xi[l]; // d1A_p[l,l,m] += t*X[l] (:303)
                    d1adiag[l * mm + m] = rfma(t, xi[l], d1adiag[l * mm + m]);
                }
            }
            for off in 0..nhh {
                d2gamma_p[off] += d2i[off];
                let xx = d2i[off];
                for l in 0..p {
                    d2b_p[l * nhh + off] = rfma(xx, xi[l], d2b_p[l * nhh + off]); // :293
                    let t = xx * xi[l]; // d2ldA_p[l,off] += t*X[l] (:365)
                    d2lda_p[l * nhh + off] = rfma(t, xi[l], d2lda_p[l * nhh + off]);
                }
            }
            i += 1;
        }
        // emit second derivatives of leading diagonal of H (coxph.c:341-368)
        let xx = dr / gamma_p;
        let xx0 = xx / gamma_p; // dr/gamma_p^2
        let xx1 = xx0 / gamma_p; // dr/gamma_p^3
        let xx2 = xx1 / gamma_p; // dr/gamma_p^4
        let mut off = 0usize;
        for m in 0..mm {
            let xx3 = -2.0 * xx1 * d1gamma_p[m];
            for k in m..mm {
                for l in 0..p {
                    // coxph.c:351-362 d2H emit, in the EXACT clang -O2 arm64 fma
                    // tree (same left-fuse/right-round rule as d1H) — verified
                    // bit-for-bit against the compiled emit (6000/6000 inputs).
                    let bl = b_p[l];
                    // A_p[l,l]*d1gamma[k] + 2*d1b[l,k]*b_l
                    let inner1 = rfma(adiag[l], d1gamma_p[k], (2.0 * d1b_p[l * mm + k]) * bl);
                    // 5-product group, left-assoc fused
                    let mut big5 =
                        rfma(d1adiag[l * mm + m], d1gamma_p[k], adiag[l] * d2gamma_p[off]);
                    big5 = rfma(d2b_p[l * nhh + off], bl, big5);
                    big5 = rfma(2.0 * d1b_p[l * mm + k], d1b_p[l * mm + m], big5);
                    big5 = rfma(bl, d2b_p[l * nhh + off], big5);
                    // 2*d1b[l,m]*b_l*d1gamma[k] + b_l*b_l*d2gamma[off]
                    let inner6 = rfma(
                        (2.0 * d1b_p[l * mm + m]) * bl,
                        d1gamma_p[k],
                        (bl * bl) * d2gamma_p[off],
                    );
                    let mut v = rfma(xx3, inner1, xx0 * big5); // T1 fused, T2 rounded
                    v = rfma(xx0 * d1gamma_p[m], d1adiag[l * mm + k], v); // + xx0*d1gamma[m]*d1A[l,l,k]
                    v = rfma(-xx, d2lda_p[l * nhh + off], v); // - xx*d2ldA
                    let t5 = (((6.0 * xx2) * d1gamma_p[m]) * bl) * bl;
                    v = rfma(t5, d1gamma_p[k], v); // + 6*xx2*d1gamma[m]*b_l^2*d1gamma[k]
                    v = rfma(-(2.0 * xx1), inner6, v); // - 2*xx1*inner6
                    d2h[l * nhh + off] += v;
                }
                off += 1;
            }
        }
    }

    Array2::from_shape_vec((p, nhh), d2h)
        .unwrap()
        .into_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cox_l, m)?)?;
    m.add_function(wrap_pyfunction!(cox_lpl0, m)?)?;
    m.add_function(wrap_pyfunction!(cox_lpl_d1, m)?)?;
    m.add_function(wrap_pyfunction!(cox_d2h, m)?)?;
    Ok(())
}

use pyo3::wrap_pyfunction;
