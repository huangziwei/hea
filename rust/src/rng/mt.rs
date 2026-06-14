//! `RsMt` — R's Mersenne-Twister RNG (`set.seed` stream) + the nmath `r*`
//! samplers, as a single stateful PyO3 class.
//!
//! Line-by-line mirror of `hea/R/rng.py`'s `RMersenneTwister`, which is itself a
//! mirror of R's `src/main/RNG.c` / `src/main/random.c` and `src/nmath/{snorm,
//! sexp,rpois,rbinom,rgamma,rbeta}.c`. The Python class is the spec AND the
//! fallback/oracle: `python == R` is pinned, so `rs == python` (and the live-R
//! gate) transitively pins `rs == R`.
//!
//! The whole point of this port is to run the rejection-sampling *loops* in Rust
//! so the family `$rd` hooks don't pay Python per-draw overhead. The stream is
//! inherently serial — NEVER parallelize draws (would reorder them).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::nmath::norm::qnorm5_scalar;
use crate::nmath::util::r_pow_di;

// --- MT period parameters (RNG.c:650-654) -----------------------------------
const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;
// MT_genrand scale factor (RNG.c:726) — IEEE-754 nearest-double to 2^-32.
const INV_2P32: f64 = 2.3283064365386963e-10;
// fixup() boundary epsilon (RNG.c:86) — R's i2_32m1.
const I2_32M1: f64 = 2.328306437080797e-10;
// rbeta overflow guard: expmax = DBL_MAX_EXP * M_LN2, and DBL_MAX.
const EXPMAX: f64 = 1024.0 * 0.6931471805599453;
const DBL_MAX: f64 = f64::MAX;
const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934;
const BIG: f64 = 134217728.0; // 2^27 (snorm.c INVERSION)

/// R's default RNG after `set.seed(seed)`, bit-exact and platform-independent.
#[pyclass]
pub struct RsMt {
    mt: [u32; N],
    buf: [f64; N],
    pos: usize,
}

// --- internal (non-Python) helpers ------------------------------------------
impl RsMt {
    /// One MT19937 twist of the 624-word state (R's `MT_genrand` block
    /// generation, in place), then temper+fixup all 624 words into `buf`.
    /// Block-tempering equals R's per-draw tempering bit-for-bit (tempering is a
    /// pure function of the stored word).
    fn refill(&mut self) {
        {
            let mt = &mut self.mt;
            for kk in 0..(N - M) {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            }
            for kk in (N - M)..(N - 1) {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk - (N - M)] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            }
            let y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
        }
        // Tempering (RNG.c:720-723) + fixup to the open interval (0, 1).
        for i in 0..N {
            let mut t = self.mt[i];
            t ^= t >> 11;
            t ^= (t << 7) & 0x9D2C_5680;
            t ^= (t << 15) & 0xEFC6_0000;
            t ^= t >> 18;
            let mut u = t as f64 * INV_2P32;
            if u <= 0.0 {
                u = 0.5 * I2_32M1;
            }
            if 1.0 - u <= 0.0 {
                u = 1.0 - 0.5 * I2_32M1;
            }
            self.buf[i] = u;
        }
        self.pos = 0;
    }

    /// One uniform from the open interval (0, 1) — R's `unif_rand` (MT case).
    fn next_unif(&mut self) -> f64 {
        if self.pos >= N {
            self.refill();
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        v
    }

    /// `n` uniforms, drawn as one bulk pull off the same buffer (identical
    /// values + refill boundaries to `n` scalar pulls — mirror of rng.py).
    fn unif_vec(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        let mut filled = 0;
        while filled < n {
            if self.pos >= N {
                self.refill();
            }
            let take = std::cmp::min(n - filled, N - self.pos);
            out.extend_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            filled += take;
        }
        out
    }

    /// One standard normal via R's Inversion: `qnorm((floor(2^27 u1) + u2)/2^27)`
    /// (snorm.c INVERSION). Consumes two uniforms.
    fn next_norm(&mut self) -> f64 {
        let u1 = self.next_unif();
        let u1 = (BIG * u1).trunc() + self.next_unif();
        qnorm5_scalar(u1 / BIG, 0.0, 1.0, true, false)
    }

    /// R's `exp_rand` (standard exponential) — sexp.c (Ahrens-Dieter).
    fn next_exp(&mut self) -> f64 {
        let q = EXP_Q[0];
        let mut a = 0.0;
        let mut u = self.next_unif();
        while u <= 0.0 || u >= 1.0 {
            u = self.next_unif();
        }
        loop {
            u += u;
            if u > 1.0 {
                break;
            }
            a += q;
        }
        u -= 1.0;
        if u <= q {
            return a + u;
        }
        let mut i = 0usize;
        let mut ustar = self.next_unif();
        let mut umin = ustar;
        loop {
            ustar = self.next_unif();
            if umin > ustar {
                umin = ustar;
            }
            i += 1;
            if u <= EXP_Q[i] {
                break;
            }
        }
        a + umin * q
    }

    /// R's `R_unif_index(dn)` (REJECTION) — integer in [0, dn).
    fn next_unif_index(&mut self, dn: i64) -> i64 {
        if dn <= 0 {
            return 0;
        }
        let bits = (dn as f64).log2().ceil() as i32;
        loop {
            let dv = self.rbits(bits);
            if dv < dn {
                return dv;
            }
        }
    }

    /// rbits (RNG.c:879-889): a random non-negative integer < 2^bits, 16 bits per
    /// `unif_rand` draw.
    fn rbits(&mut self, bits: i32) -> i64 {
        let mut v: u64 = 0;
        let mut nb = 0i32;
        while nb <= bits {
            let v1 = (self.next_unif() * 65536.0).floor() as u64;
            v = 65536 * v + v1;
            nb += 16;
        }
        (v & ((1u64 << bits) - 1)) as i64
    }
}

// Cheng's v/w step shared by rbeta's BB and BC branches (rbeta.c).
fn beta_vw(aa: f64, u1: f64, beta: f64) -> (f64, f64) {
    let v = beta * (u1 / (1.0 - u1)).ln();
    let w = if v <= EXPMAX {
        let w = aa * v.exp();
        if !w.is_finite() {
            DBL_MAX
        } else {
            w
        }
    } else {
        DBL_MAX
    };
    (v, w)
}

#[pymethods]
impl RsMt {
    #[new]
    fn new(seed: i64) -> Self {
        let mut s = RsMt {
            mt: [0u32; N],
            buf: [0.0; N],
            pos: N,
        };
        s.set_seed(seed);
        s
    }

    /// R's `set.seed(seed)` (Mersenne-Twister kind): 50× LCG warm-up, then 625
    /// further LCG draws; the first (the `mti` slot) is discarded and `mti` is
    /// set to N so the first draw regenerates the state (FixupSeeds initial).
    fn set_seed(&mut self, seed: i64) {
        let mut s = seed as u32;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        for j in 0..625usize {
            s = s.wrapping_mul(69069).wrapping_add(1);
            if j >= 1 {
                self.mt[j - 1] = s;
            }
        }
        self.pos = N; // mti = 624 >= N ⇒ first draw refills
    }

    /// One `runif()` draw.
    fn unif_rand(&mut self) -> f64 {
        self.next_unif()
    }

    /// A length-`n` `runif()` array (same stream as `n` scalar draws).
    fn unif_rand_n<'py>(&mut self, py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f64>> {
        self.unif_vec(n).into_pyarray(py)
    }

    /// One standard-normal `norm_rand()` (Inversion).
    fn norm_rand(&mut self) -> f64 {
        self.next_norm()
    }

    /// `n` standard normals, drawing the 2n Inversion uniforms in one batch
    /// (same stream as 2n scalar draws; bit-identical to the per-draw path).
    fn rnorm_n<'py>(&mut self, py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f64>> {
        let u = self.unif_vec(2 * n);
        let out: Vec<f64> = (0..n)
            .map(|i| {
                let comb = (BIG * u[2 * i]).trunc() + u[2 * i + 1];
                qnorm5_scalar(comb / BIG, 0.0, 1.0, true, false)
            })
            .collect();
        out.into_pyarray(py)
    }

    /// R's `exp_rand` (standard exponential).
    fn exp_rand(&mut self) -> f64 {
        self.next_exp()
    }

    /// `R_unif_index(dn)` (REJECTION) — integer in [0, dn).
    fn unif_index(&mut self, dn: i64) -> i64 {
        self.next_unif_index(dn)
    }

    /// R's `sample(1:n, k, replace=)` as 0-based indices (do_sample, random.c).
    fn sample_int<'py>(
        &mut self,
        py: Python<'py>,
        n: i64,
        k: i64,
        replace: bool,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        if replace {
            let out: Vec<i64> = (0..k).map(|_| self.next_unif_index(n)).collect();
            return Ok(out.into_pyarray(py));
        }
        if k < 0 || k > n {
            return Err(PyValueError::new_err(format!("k={k} not in [0, n={n}]")));
        }
        let mut pool: Vec<i64> = (0..n).collect();
        let mut out: Vec<i64> = Vec::with_capacity(k as usize);
        let mut m = n;
        for _ in 0..k {
            let j = self.next_unif_index(m) as usize;
            out.push(pool[j]);
            m -= 1;
            pool[j] = pool[m as usize];
        }
        Ok(out.into_pyarray(py))
    }

    /// R's `rpois(mu)` — rpois.c. Inversion for mu<10 (1 uniform, CDF walk),
    /// transformed rejection (Ahrens-Dieter PD) for mu>=10.
    fn rpois(&mut self, mu: f64) -> f64 {
        if mu <= 0.0 {
            return 0.0;
        }
        if mu < 10.0 {
            // inversion (consumes 1 uniform)
            let u = self.next_unif();
            let p0 = (-mu).exp();
            let mut p = p0;
            let mut q = p0;
            if u <= p0 {
                return 0.0;
            }
            let mut k = 0i64;
            while k < 35 {
                k += 1;
                p *= mu / k as f64;
                q += p;
                if u <= q {
                    return k as f64;
                }
            }
            loop {
                k += 1;
                p *= mu / k as f64;
                q += p;
                if u <= q || p == 0.0 {
                    return k as f64;
                }
            }
        }
        // big mu (>= 10): Ahrens-Dieter (1982) "PD" algorithm — rpois.c.
        const A0: f64 = -0.5;
        const A1: f64 = 0.3333333;
        const A2: f64 = -0.2500068;
        const A3: f64 = 0.2000118;
        const A4: f64 = -0.1661269;
        const A5: f64 = 0.1421878;
        const A6: f64 = -0.1384794;
        const A7: f64 = 0.1250060;
        const FACT: [f64; 10] = [1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.];
        const ONE_7: f64 = 0.1428571428571428571;
        const ONE_12: f64 = 0.0833333333333333333;
        const ONE_24: f64 = 0.0416666666666666667;
        let s = mu.sqrt();
        let d = 6.0 * mu * mu;
        let big_l = (mu - 1.1484).floor();
        let omega = M_1_SQRT_2PI / s;
        let b1 = ONE_24 / mu;
        let b2 = 0.3 * b1 * b1;
        let c3 = ONE_7 * b1 * b2;
        let c2 = b2 - 15.0 * c3;
        let c1 = b1 - 6.0 * b2 + 45.0 * c3;
        let c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3;
        let c = 0.1069 / mu;

        let step_f = |pois: f64, fk: f64, difmuk: f64| -> (f64, f64, f64, f64) {
            let (px, py);
            if pois < 10.0 {
                px = -mu;
                py = mu.powf(pois) / FACT[pois as usize];
            } else {
                let mut del = ONE_12 / fk;
                del = del * (1.0 - 4.8 * del * del);
                let v = difmuk / fk;
                if v.abs() <= 0.25 {
                    px = fk * v * v
                        * (((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1) * v
                            + A0)
                        - del;
                } else {
                    px = fk * (1.0 + v).ln() - difmuk - del;
                }
                py = M_1_SQRT_2PI / fk.sqrt();
            }
            let x = (0.5 - difmuk) / s;
            let xx = x * x;
            let fx = -0.5 * xx;
            let fy = omega * (((c3 * xx + c2) * xx + c1) * xx + c0);
            (px, py, fx, fy)
        };

        // Step N — normal candidate (immediate / squeeze acceptance)
        let g = mu + s * self.next_norm();
        if g >= 0.0 {
            let pois = g.floor();
            if pois >= big_l {
                return pois;
            }
            let fk = pois;
            let difmuk = mu - fk;
            let u = self.next_unif();
            if d * u >= difmuk * difmuk * difmuk {
                return pois;
            }
            let (px, py, fx, fy) = step_f(pois, fk, difmuk);
            if fy - u * fy <= py * (px - fx).exp() {
                return pois;
            }
        }
        // Step E — exponential candidates
        loop {
            let e = self.next_exp();
            let u = 2.0 * self.next_unif() - 1.0;
            let t = 1.8 + e.copysign(u);
            if t <= -0.6744 {
                continue;
            }
            let pois = (mu + s * t).floor();
            let fk = pois;
            let difmuk = mu - fk;
            let (px, py, fx, fy) = step_f(pois, fk, difmuk);
            if c * u.abs() <= py * (px + e).exp() - fy * (fx + e).exp() {
                return pois;
            }
        }
    }

    /// R's `rbinom(size, prob)` — rbinom.c. Inversion (BINV) for n·min(p,1-p)<30,
    /// BTPE rejection otherwise. `qn = R_pow_di(q, n)` (NOT libm pow).
    #[pyo3(signature = (size, prob))]
    fn rbinom(&mut self, size: f64, prob: f64) -> f64 {
        let n = crate::nmath::util::round_half_even(size) as i64;
        if n == 0 || prob <= 0.0 {
            return 0.0;
        }
        if prob >= 1.0 {
            return n as f64;
        }
        let p = prob.min(1.0 - prob);
        let q = 1.0 - p;
        let np_ = n as f64 * p;
        if np_ < 30.0 {
            // inversion (BINV)
            let qn = r_pow_di(q, n);
            let r = p / q;
            let g = r * (n as f64 + 1.0);
            loop {
                let mut ix = 0i64;
                let mut f = qn;
                let mut u = self.next_unif();
                loop {
                    if u < f {
                        return if prob <= 0.5 { ix as f64 } else { (n - ix) as f64 };
                    }
                    if ix > 110 {
                        break;
                    }
                    u -= f;
                    ix += 1;
                    f *= g / ix as f64 - r;
                }
            }
        }
        // BTPE (Kachitvichyanukul & Schmeiser)
        let ffm = np_ + p;
        let m = ffm as i64;
        let fm = m as f64;
        let npq = np_ * q;
        let p1 = (2.195 * npq.sqrt() - 4.6 * q).floor() + 0.5;
        let xm = fm + 0.5;
        let xl = xm - p1;
        let xr = xm + p1;
        let c = 0.134 + 20.5 / (15.3 + fm);
        let mut al = (ffm - xl) / (ffm - xl * p);
        let xll = al * (1.0 + 0.5 * al);
        al = (xr - ffm) / (xr * q);
        let xlr = al * (1.0 + 0.5 * al);
        let p2 = p1 * (1.0 + c + c);
        let p3 = p2 + c / xll;
        let p4 = p3 + c / xlr;
        loop {
            let u = self.next_unif() * p4;
            let mut v = self.next_unif();
            let ix: i64;
            if u <= p1 {
                let ixv = (xm - p1 * v + u) as i64;
                return if prob <= 0.5 { ixv as f64 } else { (n - ixv) as f64 };
            }
            if u <= p2 {
                let x = xl + (u - p1) / c;
                v = v * c + 1.0 - (xm - x).abs() / p1;
                if v > 1.0 || v <= 0.0 {
                    continue;
                }
                ix = x as i64;
            } else if u <= p3 {
                ix = (xl + v.ln() / xll) as i64;
                if ix < 0 {
                    continue;
                }
                v = v * (u - p2) * xll;
            } else {
                ix = (xr - v.ln() / xlr) as i64;
                if ix > n {
                    continue;
                }
                v = v * (u - p3) * xlr;
            }
            let k = (ix - m).abs();
            if k <= 20 || k as f64 >= npq / 2.0 - 1.0 {
                let mut f = 1.0;
                let r = p / q;
                let g = (n as f64 + 1.0) * r;
                if m < ix {
                    for i in (m + 1)..=ix {
                        f *= g / i as f64 - r;
                    }
                } else if m > ix {
                    for i in (ix + 1)..=m {
                        f /= g / i as f64 - r;
                    }
                }
                if v <= f {
                    return if prob <= 0.5 { ix as f64 } else { (n - ix) as f64 };
                }
                continue;
            }
            let amaxp = (k as f64 / npq)
                * ((k as f64 * (k as f64 / 3.0 + 0.625) + 0.1666666666666) / npq + 0.5);
            let ynorm = -(k as f64) * k as f64 / (2.0 * npq);
            let alv = v.ln();
            if alv < ynorm - amaxp {
                return if prob <= 0.5 { ix as f64 } else { (n - ix) as f64 };
            }
            if alv > ynorm + amaxp {
                continue;
            }
            let x1 = (ix + 1) as f64;
            let f1 = fm + 1.0;
            let z = (n + 1) as f64 - fm;
            let w = (n - ix) as f64 + 1.0;
            let z2 = z * z;
            let x2 = x1 * x1;
            let f2 = f1 * f1;
            let w2 = w * w;
            let t = xm * (f1 / x1).ln()
                + (n as f64 - fm + 0.5) * (z / w).ln()
                + (ix - m) as f64 * (w * p / (x1 * q)).ln()
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0;
            if alv <= t {
                return if prob <= 0.5 { ix as f64 } else { (n - ix) as f64 };
            }
        }
    }

    /// R's `rgamma(shape, scale=)` — rgamma.c (GD for a>=1, GS for a<1).
    #[pyo3(signature = (shape, scale=1.0))]
    fn rgamma(&mut self, shape: f64, scale: f64) -> f64 {
        let a = shape;
        if a < 1.0 {
            // GS algorithm
            if a == 0.0 {
                return 0.0;
            }
            let e1 = 0.36787944117144232159; // exp(-1)
            let e = 1.0 + e1 * a;
            loop {
                let p = e * self.next_unif();
                if p >= 1.0 {
                    let x = -((e - p) / a).ln();
                    if self.next_exp() >= (1.0 - a) * x.ln() {
                        return scale * x;
                    }
                } else {
                    let x = (p.ln() / a).exp();
                    if self.next_exp() >= x {
                        return scale * x;
                    }
                }
            }
        }
        // GD algorithm (a >= 1)
        let sqrt32 = 5.656854;
        let s2 = a - 0.5;
        let s = s2.sqrt();
        let d = sqrt32 - 12.0 * s;
        let t0 = self.next_norm();
        let x0 = s + 0.5 * t0;
        let ret = x0 * x0;
        if t0 >= 0.0 {
            return scale * ret;
        }
        let u0 = self.next_unif();
        if d * u0 <= t0 * t0 * t0 {
            return scale * ret;
        }
        let r = 1.0 / a;
        let q0 = ((((((GA_Q[6] * r + GA_Q[5]) * r + GA_Q[4]) * r + GA_Q[3]) * r + GA_Q[2]) * r
            + GA_Q[1])
            * r
            + GA_Q[0])
            * r;
        let (b, si, c);
        if a <= 3.686 {
            b = 0.463 + s + 0.178 * s2;
            si = 1.235;
            c = 0.195 / s - 0.079 + 0.16 * s;
        } else if a <= 13.022 {
            b = 1.654 + 0.0076 * s2;
            si = 1.68 / s + 0.275;
            c = 0.062 / s + 0.024;
        } else {
            b = 1.77;
            si = 0.75;
            c = 0.1515 / s;
        }
        if x0 > 0.0 {
            let v = t0 / (s + s);
            let qq = if v.abs() <= 0.25 {
                q0 + 0.5 * t0 * t0
                    * ((((((GA_A[6] * v + GA_A[5]) * v + GA_A[4]) * v + GA_A[3]) * v + GA_A[2]) * v
                        + GA_A[1])
                        * v
                        + GA_A[0])
                    * v
            } else {
                q0 - s * t0 + 0.25 * t0 * t0 + (s2 + s2) * (1.0 + v).ln()
            };
            if (1.0 - u0).ln() <= qq {
                return scale * ret;
            }
        }
        loop {
            let e = self.next_exp();
            let u = 2.0 * self.next_unif() - 1.0;
            let t = b + (si * e).copysign(u);
            if t >= -0.71874483771719 {
                let v = t / (s + s);
                let qq = if v.abs() <= 0.25 {
                    q0 + 0.5 * t * t
                        * ((((((GA_A[6] * v + GA_A[5]) * v + GA_A[4]) * v + GA_A[3]) * v + GA_A[2])
                            * v
                            + GA_A[1])
                            * v
                            + GA_A[0])
                        * v
                } else {
                    q0 - s * t + 0.25 * t * t + (s2 + s2) * (1.0 + v).ln()
                };
                if qq > 0.0 {
                    let w = qq.exp_m1();
                    if c * u.abs() <= w * (e - 0.5 * t * t).exp() {
                        let x = s + 0.5 * t;
                        return scale * x * x;
                    }
                }
            }
        }
    }

    /// R's `rnbinom(size, mu=)` — Poisson-Gamma mixture (rnbinom.c).
    fn rnbinom(&mut self, size: f64, mu: f64) -> f64 {
        if mu <= 0.0 {
            return 0.0;
        }
        let g = self.rgamma(size, mu / size);
        self.rpois(g)
    }

    /// R's `rchisq(df, ncp=0)` — central rgamma(df/2, 2); noncentral via rpois.
    #[pyo3(signature = (df, ncp=0.0))]
    fn rchisq(&mut self, df: f64, ncp: f64) -> f64 {
        if ncp == 0.0 {
            return if df == 0.0 {
                0.0
            } else {
                self.rgamma(df / 2.0, 2.0)
            };
        }
        let r = self.rpois(ncp / 2.0);
        let mut out = if r > 0.0 { self.rgamma(r, 2.0) } else { 0.0 };
        if df > 0.0 {
            out += self.rgamma(df / 2.0, 2.0);
        }
        out
    }

    /// R's central `rt(df)` = norm_rand() / sqrt(rchisq(df)/df) (rt.c).
    fn rt(&mut self, df: f64) -> f64 {
        if !df.is_finite() {
            return self.next_norm();
        }
        self.next_norm() / (self.rchisq(df, 0.0) / df).sqrt()
    }

    /// R's central `rf(df1, df2)` = (rchisq(df1)/df1)/(rchisq(df2)/df2) (rf.c).
    fn rf(&mut self, df1: f64, df2: f64) -> f64 {
        let v1 = if df1.is_finite() {
            self.rchisq(df1, 0.0) / df1
        } else {
            1.0
        };
        let v2 = if df2.is_finite() {
            self.rchisq(df2, 0.0) / df2
        } else {
            1.0
        };
        v1 / v2
    }

    /// R's `rbeta(aa, bb)` — Cheng's BB (min>1) / BC (min<=1) algorithm (rbeta.c).
    fn rbeta(&mut self, aa: f64, bb: f64) -> PyResult<f64> {
        if aa < 0.0 || bb < 0.0 {
            return Err(PyValueError::new_err("rbeta: shapes must be >= 0"));
        }
        if aa == 0.0 && bb == 0.0 {
            return Ok(if self.next_unif() < 0.5 { 0.0 } else { 1.0 });
        }
        let a = if aa < bb { aa } else { bb }; // min(aa, bb)
        let b = if aa < bb { bb } else { aa }; // max(aa, bb)
        let alpha = a + b;
        if a <= 1.0 {
            // --- Algorithm BC ---
            let beta = 1.0 / a;
            let delta = 1.0 + b - a;
            let k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta - 0.777778);
            let k2 = 0.25 + (0.5 + 0.25 / delta) * a;
            let w = loop {
                let u1 = self.next_unif();
                let u2 = self.next_unif();
                let z;
                if u1 < 0.5 {
                    let y = u1 * u2;
                    z = u1 * y; // u1*(u1*u2)
                    if 0.25 * u2 + z - y >= k1 {
                        continue;
                    }
                } else {
                    z = u1 * u1 * u2; // (u1*u1)*u2
                    if z <= 0.25 {
                        let (_v, ww) = beta_vw(b, u1, beta);
                        break ww;
                    }
                    if z >= k2 {
                        continue;
                    }
                }
                let (v, ww) = beta_vw(b, u1, beta);
                if alpha * ((alpha / (a + ww)).ln() + v) - 1.3862944 >= z.ln() {
                    break ww;
                }
            };
            return Ok(if aa == a { a / (a + w) } else { w / (a + w) });
        }
        // --- Algorithm BB ---
        let beta = ((alpha - 2.0) / (2.0 * a * b - alpha)).sqrt();
        let gamma = a + 1.0 / beta;
        let w = loop {
            let u1 = self.next_unif();
            let u2 = self.next_unif();
            let (v, ww) = beta_vw(a, u1, beta);
            let z = u1 * u1 * u2;
            let r = gamma * v - 1.3862944;
            let s = a + r - ww;
            if s + 2.609438 >= 5.0 * z {
                break ww;
            }
            let t = z.ln();
            if s > t {
                break ww;
            }
            if r + alpha * (alpha / (b + ww)).ln() >= t {
                break ww;
            }
        };
        Ok(if aa != a { b / (b + w) } else { w / (b + w) })
    }

    // --- batch samplers: the whole per-element loop runs in Rust, so the family
    // `$rd` hooks (via RGenerator) pay ONE Python↔Rust crossing per vector
    // instead of n. Each element is bit-identical to the scalar method; the draw
    // order (and thus the stream) is the same as n scalar calls. ---

    /// `rpois` over an array of means.
    fn rpois_n<'py>(
        &mut self,
        py: Python<'py>,
        mu: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mu = mu.as_array();
        let out: Vec<f64> = (0..mu.len()).map(|i| self.rpois(mu[i])).collect();
        out.into_pyarray(py)
    }

    /// `rbinom` over arrays of (size, prob).
    fn rbinom_n<'py>(
        &mut self,
        py: Python<'py>,
        size: PyReadonlyArray1<'py, f64>,
        prob: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let (sz, pr) = (size.as_array(), prob.as_array());
        let out: Vec<f64> = (0..sz.len()).map(|i| self.rbinom(sz[i], pr[i])).collect();
        out.into_pyarray(py)
    }

    /// `rgamma` over arrays of (shape, scale).
    fn rgamma_n<'py>(
        &mut self,
        py: Python<'py>,
        shape: PyReadonlyArray1<'py, f64>,
        scale: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let (sh, sc) = (shape.as_array(), scale.as_array());
        let out: Vec<f64> = (0..sh.len()).map(|i| self.rgamma(sh[i], sc[i])).collect();
        out.into_pyarray(py)
    }

    /// central `rt` over an array of df.
    fn rt_n<'py>(
        &mut self,
        py: Python<'py>,
        df: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let df = df.as_array();
        let out: Vec<f64> = (0..df.len()).map(|i| self.rt(df[i])).collect();
        out.into_pyarray(py)
    }

    /// central `rf` over arrays of (df1, df2).
    fn rf_n<'py>(
        &mut self,
        py: Python<'py>,
        df1: PyReadonlyArray1<'py, f64>,
        df2: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let (a, b) = (df1.as_array(), df2.as_array());
        let out: Vec<f64> = (0..a.len()).map(|i| self.rf(a[i], b[i])).collect();
        out.into_pyarray(py)
    }

    /// `rchisq` over arrays of (df, ncp) — each element a full `rnchisq`.
    fn rchisq_n<'py>(
        &mut self,
        py: Python<'py>,
        df: PyReadonlyArray1<'py, f64>,
        ncp: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let (d, c) = (df.as_array(), ncp.as_array());
        let out: Vec<f64> = (0..d.len()).map(|i| self.rchisq(d[i], c[i])).collect();
        out.into_pyarray(py)
    }

    /// `rnbinom` over arrays of (size, mu).
    fn rnbinom_n<'py>(
        &mut self,
        py: Python<'py>,
        size: PyReadonlyArray1<'py, f64>,
        mu: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let (s, m) = (size.as_array(), mu.as_array());
        let out: Vec<f64> = (0..s.len()).map(|i| self.rnbinom(s[i], m[i])).collect();
        out.into_pyarray(py)
    }

    /// `n` standard exponentials (`exp_rand`).
    fn exp_rand_n<'py>(&mut self, py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f64>> {
        let out: Vec<f64> = (0..n).map(|_| self.next_exp()).collect();
        out.into_pyarray(py)
    }

    /// `rbeta` over arrays of (aa, bb).
    fn rbeta_n<'py>(
        &mut self,
        py: Python<'py>,
        aa: PyReadonlyArray1<'py, f64>,
        bb: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (a, b) = (aa.as_array(), bb.as_array());
        let mut out: Vec<f64> = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            out.push(self.rbeta(a[i], b[i])?);
        }
        Ok(out.into_pyarray(py))
    }
}

// rgamma coefficients (rgamma.c) — q1..q7 and a1..a7.
const GA_Q: [f64; 7] = [
    0.04166669, 0.02083148, 0.00801191, 0.00144121, -7.388e-5, 2.4511e-4, 2.424e-4,
];
const GA_A: [f64; 7] = [
    0.3333333, -0.250003, 0.2000062, -0.1662921, 0.1423657, -0.1367177, 0.1233795,
];

// cumulative ln(2)^k / k! — sexp.c (rng.py `_EXP_Q`)
const EXP_Q: [f64; 16] = [
    0.6931471805599453,
    0.9333736875190459,
    0.9888777961838675,
    0.9984959252914960040,
    0.9998292811061389,
    0.9999833164100727,
    0.9999985691438767,
    0.9999998906925558,
    0.9999999924734159,
    0.9999999995283275,
    0.9999999999728814,
    0.9999999999985598,
    0.9999999999999289,
    0.9999999999999968,
    0.9999999999999999,
    1.0000000000000000,
];
