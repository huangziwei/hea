//! Exact float utilities used by the gamma/loader kernels.
//!
//! `frexp`/`ldexp` are pure bit-manipulation (no rounding), so they match C's
//! `frexp`/`ldexp` — and therefore numpy's `np.frexp`/`np.ldexp` — on every
//! platform. Mirrors musl. `round_half_even` matches R's `nearbyint` / Python
//! `round` (round-half-to-even), which R uses for `R_forceint`.
#![allow(dead_code)]

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
