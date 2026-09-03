#![allow(dead_code)]

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

pub fn frexp(x: f64) -> (f64, i32) {
    let mut bits = x.to_bits();
    let ee = ((bits >> 52) & 0x7ff) as i32;
    if ee == 0 {
        if x != 0.0 {
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

#[inline]
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

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
