"""R's random number generator — a bit-exact port.

Reproduces R >= 3.6 defaults: ``RNGkind("Mersenne-Twister", normal.kind =
"Inversion", sample.kind = "Rejection")``. ``RMersenneTwister(seed)``
matches ``set.seed(seed)`` and then draws the identical stream R would,
so R results that depend on ``runif()`` / ``sample()`` can be pinned
bit-for-bit from Python.

Direct port of R's ``src/main/RNG.c`` and ``src/main/random.c``:

* ``RNG_Init`` — 50× LCG warm-up of the user seed, then 625 further LCG
  draws fill ``i_seed[0..624]``; ``FixupSeeds(initial=TRUE)`` overwrites
  ``i_seed[0]`` (the ``mti`` slot) with N=624 so the first draw
  regenerates the whole state.
* ``MT_genrand`` — MT19937 twist + R's tempering masks, scaled by the
  IEEE-754 nearest-double to 2⁻³². The twist here is vectorized over
  the 624-word block (tempering is a pure function of the stored word,
  so block-tempering equals R's per-draw tempering bit-for-bit).
* ``unif_rand`` — the MT output through R's ``fixup`` (keeps draws
  strictly inside (0, 1)).
* ``R_unif_index`` (``Sample_kind = REJECTION``) — ``bits =
  ceil(log2(dn))``, draw ``rbits(bits)`` until < dn.
* ``sample(n, k, replace=FALSE)`` — ``do_sample``'s shrinking-pool walk;
  ``replace=TRUE`` — one ``R_unif_index(n)`` per draw.

Index results are 0-based (R returns 1-based; callers adjust).

In-tree consumers: tp/ds/sos knot subsampling (``temp.seed(1)`` +
``sample()``, smooth.r:1286 via hea.formula), qq.gam's ``sample(U)``
randomization and k.check's permutation null (hea.models.gam), and
bam's ``compress.df`` / ``discrete.mf`` (``temp.seed(8547)``,
hea.models.bam).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

__all__ = ["RMersenneTwister"]

# Period parameters (RNG.c:646-650).
_N = 624
_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER = np.uint32(0x80000000)
_LOWER = np.uint32(0x7FFFFFFF)
# MT_genrand scale factor (RNG.c:722) — IEEE-754 nearest-double to 2^-32.
_INV_2P32 = 2.3283064365386963e-10
# fixup() boundary epsilon (RNG.c:86) — R's i2_32m1, a distinct constant
# (1/(2^32-1)-ish); only reachable when a tempered word is exactly 0.
_I2_32M1 = 2.328306437080797e-10


class RMersenneTwister:
    """R's default RNG after ``set.seed(seed)``, bit-exact and
    platform-independent."""

    __slots__ = ("_mt", "_buf", "_pos")

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """R's ``set.seed(seed)`` (Mersenne-Twister kind)."""
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        fills = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            fills[j] = s
        # fills[0] is i_seed[0] (the mti slot) which FixupSeeds discards;
        # the remaining 624 words are the MT state.
        self._mt = fills[1:].copy()
        self._buf = np.empty(0)
        self._pos = 0

    def _refill(self) -> None:
        # One MT19937 twist of the 624-word state, vectorized in
        # dependency-free slices. Wrap-around: the last word's y pairs
        # old mt[623] with the freshly updated mt[0] (the C loop has
        # already overwritten it by then).
        mt = self._mt
        y = (mt[:_N - 1] & _UPPER) | (mt[1:] & _LOWER)
        mag = np.where((y & np.uint32(1)) != 0, _MATRIX_A, np.uint32(0))
        yshift = (y >> np.uint32(1)) ^ mag
        new = np.empty(_N, dtype=np.uint32)
        new[:_N - _M] = mt[_M:] ^ yshift[:_N - _M]
        new[_N - _M:2 * (_N - _M)] = (new[:_N - _M]
                                      ^ yshift[_N - _M:2 * (_N - _M)])
        new[2 * (_N - _M):_N - 1] = (new[_N - _M:_M - 1]
                                     ^ yshift[2 * (_N - _M):])
        y_last = (mt[_N - 1] & _UPPER) | (new[0] & _LOWER)
        mag_last = _MATRIX_A if (int(y_last) & 1) else np.uint32(0)
        new[_N - 1] = new[_M - 1] ^ (y_last >> np.uint32(1)) ^ mag_last
        self._mt = new
        # Tempering (RNG.c:716-719) + fixup to the open interval (0, 1).
        t = new.copy()
        t ^= t >> np.uint32(11)
        t ^= (t << np.uint32(7)) & np.uint32(0x9D2C5680)
        t ^= (t << np.uint32(15)) & np.uint32(0xEFC60000)
        t ^= t >> np.uint32(18)
        u = t.astype(np.float64) * _INV_2P32
        u = np.where(u <= 0.0, 0.5 * _I2_32M1, u)
        u = np.where(1.0 - u <= 0.0, 1.0 - 0.5 * _I2_32M1, u)
        self._buf = u
        self._pos = 0

    def unif_rand(self, n: int | None = None):
        """R's ``runif`` stream: one draw (``n=None``) or a length-``n``
        array — consuming the identical sequence either way."""
        if n is None:
            if self._pos >= self._buf.size:
                self._refill()
            v = float(self._buf[self._pos])
            self._pos += 1
            return v
        out = np.empty(int(n))
        for i in range(int(n)):
            out[i] = self.unif_rand()
        return out

    def norm_rand(self) -> float:
        """R's ``norm_rand`` with ``normal.kind = "Inversion"`` (R's default).

        One standard normal = ``qnorm`` of two combined uniforms for 53-bit
        precision (``snorm.c`` INVERSION case): ``u = floor(2^27·u1) + u2``,
        return ``qnorm(u / 2^27)``. So each normal consumes **two**
        ``unif_rand`` draws. The MT uniforms are bit-exact to R; ``qnorm`` here
        is SciPy's ``ndtri`` (agrees with R's Wichura AS-241 to ~1e-12).
        """
        big = 134217728.0  # 2^27
        u1 = self.unif_rand()
        u1 = float(int(big * u1)) + self.unif_rand()
        return float(ndtri(u1 / big))

    def rnorm(self, n: int | None = None, mean: float = 0.0, sd: float = 1.0):
        """R's ``rnorm(n, mean, sd)`` on R's MT stream (Inversion normals).

        ``n=None`` returns one draw; otherwise a length-``n`` array consuming
        the identical sequence. Use this (not numpy / ``hea.R.rnorm``) whenever
        the draws must line up with R's ``set.seed(); rnorm()``.
        """
        if n is None:
            return mean + sd * self.norm_rand()
        return mean + sd * np.array(
            [self.norm_rand() for _ in range(int(n))]
        )

    def _rbits(self, bits: int) -> int:
        # rbits (RNG.c:875-885): 16 bits per unif_rand draw.
        v = 0
        nb = 0
        while nb <= bits:
            v1 = int(np.floor(self.unif_rand() * 65536.0))
            v = 65536 * v + v1
            nb += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """``R_unif_index(dn)``, ``Sample_kind = REJECTION`` — an integer
        in [0, dn). R computes ``bits = ceil(log2(dn))`` in C-double
        arithmetic; kept literally for fidelity."""
        if dn <= 0:
            return 0
        bits = int(np.ceil(np.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    def sample_int(self, n: int, k: int, replace: bool = False) -> np.ndarray:
        """R's ``sample(1:n, k, replace=)`` as 0-based indices.

        Without replacement: ``do_sample``'s shrinking-pool walk
        (src/main/random.c). With replacement: independent
        ``R_unif_index(n)`` per draw."""
        if replace:
            out = np.empty(k, dtype=np.int64)
            for i in range(k):
                out[i] = self.unif_index(n)
            return out
        if k < 0 or k > n:
            raise ValueError(f"k={k} not in [0, n={n}]")
        pool = np.arange(n, dtype=np.int64)
        out = np.empty(k, dtype=np.int64)
        m = n
        for i in range(k):
            j = self.unif_index(m)
            out[i] = pool[j]
            m -= 1
            pool[j] = pool[m]
        return out

    # bam-era aliases, kept for the established call sites.
    def sample_no_replace(self, n: int, k: int) -> np.ndarray:
        return self.sample_int(n, k)

    def sample_replace(self, n: int, k: int) -> np.ndarray:
        return self.sample_int(n, k, replace=True)

    def permute(self, x: np.ndarray) -> np.ndarray:
        """R's ``sample(x)`` — a full permutation of ``x``."""
        x = np.asarray(x)
        return x[self.sample_int(len(x), len(x))]

    # ------------------------------------------------------------------
    # Family samplers — ports of R's nmath ``r*`` on this bit-exact stream
    # (sexp.c / rpois.c / rbinom.c / rgamma.c). Used by simulate.merMod.
    # ------------------------------------------------------------------

    # cumulative ln(2)^k / k! — R sexp.c
    _EXP_Q = (
        0.6931471805599453, 0.9333736875190459, 0.9888777961838675,
        0.9984959252914960040, 0.9998292811061389, 0.9999833164100727,
        0.9999985691438767, 0.9999998906925558, 0.9999999924734159,
        0.9999999995283275, 0.9999999999728814, 0.9999999999985598,
        0.9999999999999289, 0.9999999999999968, 0.9999999999999999,
        1.0000000000000000,
    )

    def exp_rand(self) -> float:
        """R's ``exp_rand`` (standard exponential) — sexp.c (Ahrens-Dieter)."""
        q = self._EXP_Q
        a = 0.0
        u = self.unif_rand()
        while u <= 0.0 or u >= 1.0:
            u = self.unif_rand()
        while True:
            u += u
            if u > 1.0:
                break
            a += q[0]
        u -= 1.0
        if u <= q[0]:
            return a + u
        i = 0
        ustar = self.unif_rand()
        umin = ustar
        while True:
            ustar = self.unif_rand()
            if umin > ustar:
                umin = ustar
            i += 1
            if u <= q[i]:
                break
        return a + umin * q[0]

    def rpois(self, mu: float) -> float:
        """R's ``rpois(mu)`` — rpois.c. Inversion for ``mu < 10`` (one uniform
        per draw, CDF walk), transformed rejection (PTRS) for ``mu >= 10``."""
        if mu <= 0.0:
            return 0.0
        if mu < 10.0:                      # inversion (consumes 1 uniform)
            u = self.unif_rand()
            p = q = p0 = math.exp(-mu)
            if u <= p0:
                return 0.0
            k = 0
            while k < 35:
                k += 1
                p *= mu / k
                q += p
                if u <= q:
                    return float(k)
            # tail: keep walking (rare)
            while True:
                k += 1
                p *= mu / k
                q += p
                if u <= q or p == 0.0:
                    return float(k)
        # big mu (>= 10): Ahrens-Dieter (1982) "PD" algorithm — R rpois.c.
        M_1_SQRT_2PI = 0.398942280401432677939946059934
        a0, a1, a2, a3 = -0.5, 0.3333333, -0.2500068, 0.2000118
        a4, a5, a6, a7 = -0.1661269, 0.1421878, -0.1384794, 0.1250060
        fact = (1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.)
        one_7, one_12, one_24 = (0.1428571428571428571,
                                 0.0833333333333333333, 0.0416666666666666667)
        s = math.sqrt(mu)
        d = 6.0 * mu * mu
        big_l = math.floor(mu - 1.1484)
        omega = M_1_SQRT_2PI / s
        b1 = one_24 / mu
        b2 = 0.3 * b1 * b1
        c3 = one_7 * b1 * b2
        c2 = b2 - 15.0 * c3
        c1 = b1 - 6.0 * b2 + 45.0 * c3
        c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3
        c = 0.1069 / mu

        def step_f(pois, fk, difmuk):
            if pois < 10:
                px = -mu
                py = mu ** pois / fact[int(pois)]
            else:
                del_ = one_12 / fk
                del_ = del_ * (1.0 - 4.8 * del_ * del_)
                v = difmuk / fk
                if abs(v) <= 0.25:
                    px = fk * v * v * (((((((a7 * v + a6) * v + a5) * v + a4) * v
                          + a3) * v + a2) * v + a1) * v + a0) - del_
                else:
                    px = fk * math.log(1.0 + v) - difmuk - del_
                py = M_1_SQRT_2PI / math.sqrt(fk)
            x = (0.5 - difmuk) / s
            xx = x * x
            fx = -0.5 * xx
            fy = omega * (((c3 * xx + c2) * xx + c1) * xx + c0)
            return px, py, fx, fy

        # Step N — normal candidate (immediate / squeeze acceptance)
        g = mu + s * self.norm_rand()
        if g >= 0.0:
            pois = math.floor(g)
            if pois >= big_l:
                return float(pois)
            fk = float(pois)
            difmuk = mu - fk
            u = self.unif_rand()
            if d * u >= difmuk * difmuk * difmuk:
                return float(pois)
            px, py, fx, fy = step_f(pois, fk, difmuk)
            if fy - u * fy <= py * math.exp(px - fx):
                return float(pois)
        # Step E — exponential candidates
        while True:
            e = self.exp_rand()
            u = 2.0 * self.unif_rand() - 1.0
            t = 1.8 + math.copysign(e, u)
            if t <= -0.6744:
                continue
            pois = math.floor(mu + s * t)
            fk = float(pois)
            difmuk = mu - fk
            px, py, fx, fy = step_f(pois, fk, difmuk)
            if c * abs(u) <= py * math.exp(px + e) - fy * math.exp(fx + e):
                return float(pois)

    def rbinom(self, size: int, prob: float) -> float:
        """R's ``rbinom(size, prob)`` — rbinom.c. Inversion for small
        ``size·min(p,1-p)``, BTPE rejection otherwise."""
        n = int(round(size))
        if n == 0 or prob <= 0.0:
            return 0.0
        if prob >= 1.0:
            return float(n)
        p = min(prob, 1.0 - prob)
        q = 1.0 - p
        np_ = n * p
        if np_ < 30.0:                     # inversion (BINV)
            qn = q ** n
            r = p / q
            g = r * (n + 1)
            while True:
                ix = 0
                f = qn
                u = self.unif_rand()
                while True:
                    if u < f:
                        return float(ix if prob <= 0.5 else n - ix)
                    if ix > 110:
                        break
                    u -= f
                    ix += 1
                    f *= (g / ix - r)
        # BTPE (Kachitvichyanukul & Schmeiser)
        ffm = np_ + p
        m = int(ffm)
        fm = m
        npq = np_ * q
        p1 = math.floor(2.195 * math.sqrt(npq) - 4.6 * q) + 0.5
        xm = fm + 0.5
        xl = xm - p1
        xr = xm + p1
        c = 0.134 + 20.5 / (15.3 + fm)
        al = (ffm - xl) / (ffm - xl * p)
        xll = al * (1.0 + 0.5 * al)
        al = (xr - ffm) / (xr * q)
        xlr = al * (1.0 + 0.5 * al)
        p2 = p1 * (1.0 + c + c)
        p3 = p2 + c / xll
        p4 = p3 + c / xlr
        while True:
            u = self.unif_rand() * p4
            v = self.unif_rand()
            if u <= p1:
                ix = int(xm - p1 * v + u)
                return float(ix if prob <= 0.5 else n - ix)
            if u <= p2:
                x = xl + (u - p1) / c
                v = v * c + 1.0 - abs(xm - x) / p1
                if v > 1.0 or v <= 0.0:
                    continue
                ix = int(x)
            elif u <= p3:
                ix = int(xl + math.log(v) / xll)
                if ix < 0:
                    continue
                v = v * (u - p2) * xll
            else:
                ix = int(xr - math.log(v) / xlr)
                if ix > n:
                    continue
                v = v * (u - p3) * xlr
            k = abs(ix - m)
            if k <= 20 or k >= npq / 2 - 1:
                f = 1.0
                r = p / q
                g = (n + 1) * r
                if m < ix:
                    for i in range(m + 1, ix + 1):
                        f *= (g / i - r)
                elif m > ix:
                    for i in range(ix + 1, m + 1):
                        f /= (g / i - r)
                if v <= f:
                    return float(ix if prob <= 0.5 else n - ix)
                continue
            amaxp = (k / npq) * ((k * (k / 3.0 + 0.625) + 0.1666666666666) / npq + 0.5)
            ynorm = -k * k / (2.0 * npq)
            alv = math.log(v)
            if alv < ynorm - amaxp:
                return float(ix if prob <= 0.5 else n - ix)
            if alv > ynorm + amaxp:
                continue
            x1 = ix + 1
            f1 = fm + 1.0
            z = n + 1 - fm
            w = n - ix + 1.0
            z2 = z * z
            x2 = x1 * x1
            f2 = f1 * f1
            w2 = w * w
            t = (xm * math.log(f1 / x1) + (n - m + 0.5) * math.log(z / w)
                 + (ix - m) * math.log(w * p / (x1 * q))
                 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0
                 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0
                 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0
                 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0)
            if alv <= t:
                return float(ix if prob <= 0.5 else n - ix)

    # rgamma coefficients (R rgamma.c)
    _GA_Q = (0.04166669, 0.02083148, 0.00801191, 0.00144121,
             -7.388e-5, 2.4511e-4, 2.424e-4)
    _GA_A = (0.3333333, -0.250003, 0.2000062, -0.1662921,
             0.1423657, -0.1367177, 0.1233795)

    def rgamma(self, shape: float, scale: float = 1.0) -> float:
        """R's ``rgamma(shape, scale=)`` — rgamma.c (GD for a>=1, GS for a<1)."""
        a = float(shape)
        if a < 1.0:                        # GS algorithm
            if a == 0.0:
                return 0.0
            e1 = 0.36787944117144232159    # exp(-1)
            e = 1.0 + e1 * a
            while True:
                p = e * self.unif_rand()
                if p >= 1.0:
                    x = -math.log((e - p) / a)
                    if self.exp_rand() >= (1.0 - a) * math.log(x):
                        break
                else:
                    x = math.exp(math.log(p) / a)
                    if self.exp_rand() >= x:
                        break
            return scale * x
        # GD algorithm (a >= 1)
        sqrt32 = 5.656854
        s2 = a - 0.5
        s = math.sqrt(s2)
        d = sqrt32 - 12.0 * s
        t = self.norm_rand()
        x = s + 0.5 * t
        ret = x * x
        if t >= 0.0:
            return scale * ret
        u = self.unif_rand()
        if d * u <= t * t * t:
            return scale * ret
        r = 1.0 / a
        q = self._GA_Q
        q0 = ((((((q[6] * r + q[5]) * r + q[4]) * r + q[3]) * r + q[2]) * r
               + q[1]) * r + q[0]) * r
        if a <= 3.686:
            b = 0.463 + s + 0.178 * s2
            si = 1.235
            c = 0.195 / s - 0.079 + 0.16 * s
        elif a <= 13.022:
            b = 1.654 + 0.0076 * s2
            si = 1.68 / s + 0.275
            c = 0.062 / s + 0.024
        else:
            b = 1.77
            si = 0.75
            c = 0.1515 / s
        aa = self._GA_A
        if x > 0.0:
            v = t / (s + s)
            if abs(v) <= 0.25:
                qq = q0 + 0.5 * t * t * ((((((aa[6] * v + aa[5]) * v + aa[4]) * v
                     + aa[3]) * v + aa[2]) * v + aa[1]) * v + aa[0]) * v
            else:
                qq = q0 - s * t + 0.25 * t * t + (s2 + s2) * math.log(1.0 + v)
            if math.log(1.0 - u) <= qq:
                return scale * ret
        while True:
            e = self.exp_rand()
            u = 2.0 * self.unif_rand() - 1.0
            t = b + math.copysign(si * e, u)
            if t >= -0.71874483771719:
                v = t / (s + s)
                if abs(v) <= 0.25:
                    qq = q0 + 0.5 * t * t * ((((((aa[6] * v + aa[5]) * v + aa[4]) * v
                         + aa[3]) * v + aa[2]) * v + aa[1]) * v + aa[0]) * v
                else:
                    qq = q0 - s * t + 0.25 * t * t + (s2 + s2) * math.log(1.0 + v)
                if qq > 0.0:
                    w = math.expm1(qq)
                    if c * abs(u) <= w * math.exp(e - 0.5 * t * t):
                        break
        x = s + 0.5 * t
        return scale * x * x

    def rnbinom(self, size: float, mu: float) -> float:
        """R's ``rnbinom(size, mu=)`` — a Poisson-Gamma mixture
        ``rpois(rgamma(size, scale=mu/size))`` (rnbinom.c)."""
        if mu <= 0.0:
            return 0.0
        return self.rpois(self.rgamma(size, scale=mu / size))
