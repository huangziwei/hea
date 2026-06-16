//! Exact float utilities used by the gamma/loader kernels.
//!
//! `frexp`/`ldexp` are pure bit-manipulation (no rounding), so they match C's
//! `frexp`/`ldexp` — and therefore numpy's `np.frexp`/`np.ldexp` — on every
//! platform. Mirrors musl. `round_half_even` matches R's `nearbyint` / Python
//! `round` (round-half-to-even), which R uses for `R_forceint`.
#![allow(dead_code)]

/// R-parity fused multiply-add (`a*b + c`), matching how CRAN R's nmath is
/// compiled on the *current* arch.
///
/// R is built `clang -O2` with no `-ffp-contract` flag, so clang's default
/// (`=on`) fuses `a*b + c` written within a single C expression to an `fmadd` —
/// but only where the ISA has baseline FMA: **true on aarch64, false on generic
/// x86-64**. So to stay 0-ulp to the *live* R on whatever machine the gate runs,
/// hea must fuse on aarch64 and stay plain (two roundings) on x86-64. R itself is
/// NOT bit-identical across the two arches for these polynomial kernels — that is
/// the root of the Apple-Silicon parity drift — so this match must be per-arch.
///
/// On x86-64 this is literally `a*b + c` (the pre-FMA-fix code that was already
/// green vs R on Intel), so switching a kernel to `rfma` is a no-op there.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn rfma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn rfma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

/// C `frexp`: `(m, e)` with `x = m * 2^e`, `m` in `[0.5, 1)`; for 0/inf/nan
/// returns `(x, 0)`. Matches `np.frexp`.
pub fn frexp(x: f64) -> (f64, i32) {
    let mut bits = x.to_bits();
    let ee = ((bits >> 52) & 0x7ff) as i32;
    if ee == 0 {
        if x != 0.0 {
            // subnormal: scale by 2^64 and recurse
            let (m, e) = frexp(x * f64::from_bits(0x43f0000000000000));
            return (m, e - 64);
        }
        return (x, 0);
    } else if ee == 0x7ff {
        return (x, 0); // inf / nan
    }
    let e = ee - 0x3fe;
    bits &= 0x800fffffffffffff;
    bits |= 0x3fe0000000000000;
    (f64::from_bits(bits), e)
}

/// C `ldexp`/`scalbn`: `x * 2^n`, exact. Matches `np.ldexp`.
pub fn ldexp(x: f64, mut n: i32) -> f64 {
    let mut y = x;
    let p1023 = f64::from_bits(0x7fe0000000000000); // 2^1023
    let p_1022 = f64::from_bits(0x0010000000000000); // 2^-1022
    let p53 = f64::from_bits(0x4340000000000000); // 2^53
    if n > 1023 {
        y *= p1023;
        n -= 1023;
        if n > 1023 {
            y *= p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // make final n < -53 to avoid double rounding (musl trick)
        y *= p_1022 * p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= p_1022 * p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

/// Round half to even (banker's rounding) — R's `nearbyint` default mode and
/// Python's `round` for floats. (Rust's `f64::round` is half-away-from-zero.)
#[inline]
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// R's `R_pow_di(x, n)` (src/nmath/../main/arithmetic.c): integer power by
/// repeated squaring. This is deliberately NOT libm `powf` — R uses it (e.g.
/// `rbinom`'s `qn = q^n`) and the two differ by up to hundreds of ulp, which can
/// flip a rejection-sampling result. Bit-exact mirror of R's loop.
pub fn r_pow_di(mut x: f64, mut n: i64) -> f64 {
    let mut pow = 1.0;
    if x.is_nan() {
        return x;
    }
    if n != 0 {
        if !x.is_finite() {
            return x.powf(n as f64); // R: R_pow(x, (double)n)
        }
        let is_neg = n < 0;
        if is_neg {
            n = -n;
        }
        loop {
            if n & 1 == 1 {
                pow *= x;
            }
            n >>= 1;
            if n != 0 {
                x *= x;
            } else {
                break;
            }
        }
        if is_neg {
            pow = 1.0 / pow;
        }
    }
    pow
}
