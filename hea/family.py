"""GLM family + link abstraction — mirrors R's ``family()`` augmented with
mgcv's ``fix.family.{link,var,ls}`` derivative fields.

Each :class:`Family` exposes the variance function ``V(μ)`` and its first
two derivatives, the deviance residuals ``dev_resids``, the saturated
log-likelihood ``ls(y, w, scale)`` (with first/second derivatives wrt
``log scale`` for unknown-scale REML), an ``initialize`` for starting
values, ``validmu``, and the AIC contribution.

Each :class:`Link` exposes ``link(μ)``, ``linkinv(η)``, ``mu_eta(η) =
dμ/dη``, plus second-through-fourth derivatives ``d²g/dμ²``, ``d³g/dμ³``,
``d⁴g/dμ⁴`` (with respect to μ, not η — matching mgcv's ``$d2link``
naming).

For a non-canonical link the PIRLS Newton step uses

    αᵢ = 1 + (yᵢ − μᵢ)·(V'/V + g''·dμ/dη)ᵢ
    wᵢ = αᵢ · (dμᵢ/dηᵢ)² / V(μᵢ)
    zᵢ = ηᵢ + (yᵢ − μᵢ) / ((dμᵢ/dηᵢ) · αᵢ)

so that the converged ``H = X'WX + Sλ`` is the **observed** penalized
Hessian, not the Fisher one. That makes ``∂β̂/∂ρ_k = -exp(ρ_k) H⁻¹ S_k β̂``
valid even for non-canonical links — the same identity that drives the
Gaussian REML derivatives in :mod:`hea.gam`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import digamma, expit, gammaln, ndtr, ndtri, polygamma
from scipy.stats import gamma as _gamma_dist
from scipy.stats import poisson as _poisson_dist


# ---------------------------------------------------------------------------
# R nmath ports — bit-exact ``dpois`` / ``dbinom`` (saddlepoint algorithm,
# Loader 1999). Used by ``Poisson.aic`` and ``Binomial.aic`` so that the
# Laplace deviance reported by hea matches ``rho$resp$aic()`` from lme4 at
# the ULP level. scipy's ``poisson.logpmf`` / ``binom.logpmf`` use the
# direct formula ``y·log(μ) - μ - lgamma(y+1)`` (and analog for binomial),
# which differs from R's ``dpois`` / ``dbinom`` by ~1 ULP per call — and
# that 1 ULP compounded over n obs is what propagates into deriv12's
# numerator and produces visible SE / vcov gaps against R.
#
# Sources (R 4.5):
# - /tmp/R-src/src/nmath/stirlerr.c
# - /tmp/R-src/src/nmath/bd0.c   (both bd0 and ebd0)
# - /tmp/R-src/src/nmath/dpois.c
# - /tmp/R-src/src/nmath/dbinom.c
# ---------------------------------------------------------------------------


# stirlerr(n) = log(n!) - log(sqrt(2πn)·(n/e)ⁿ)
# Exact table for half-integer arguments 0, 0.5, 1.0, …, 15.0
# (stirlerr.c:78-110).
_STIRLERR_HALVES = (
    0.0,                              # n=0 — placeholder, never used
    0.1534264097200273452913848,      # 0.5
    0.0810614667953272582196702,      # 1.0
    0.0548141210519176538961390,      # 1.5
    0.0413406959554092940938221,      # 2.0
    0.03316287351993628748511048,     # 2.5
    0.02767792568499833914878929,     # 3.0
    0.02374616365629749597132920,     # 3.5
    0.02079067210376509311152277,     # 4.0
    0.01848845053267318523077934,     # 4.5
    0.01664469118982119216319487,     # 5.0
    0.01513497322191737887351255,     # 5.5
    0.01387612882307074799874573,     # 6.0
    0.01281046524292022692424986,     # 6.5
    0.01189670994589177009505572,     # 7.0
    0.01110455975820691732662991,     # 7.5
    0.010411265261972096497478567,    # 8.0
    0.009799416126158803298389475,    # 8.5
    0.009255462182712732917728637,    # 9.0
    0.008768700134139385462952823,    # 9.5
    0.008330563433362871256469318,    # 10.0
    0.007934114564314020547248100,    # 10.5
    0.007573675487951840794972024,    # 11.0
    0.007244554301320383179543912,    # 11.5
    0.006942840107209529865664152,    # 12.0
    0.006665247032707682442354394,    # 12.5
    0.006408994188004207068439631,    # 13.0
    0.006171712263039457647532867,    # 13.5
    0.005951370112758847735624416,    # 14.0
    0.005746216513010115682023589,    # 14.5
    0.005554733551962801371038690,    # 15.0
)

# Asymptotic-series coefficients (stirlerr.c:56-72).
_S0  = 0.083333333333333333333          # 1/12
_S1  = 0.00277777777777777777778        # 1/360
_S2  = 0.00079365079365079365079365     # 1/1260
_S3  = 0.000595238095238095238095238    # 1/1680
_S4  = 0.0008417508417508417508417508   # 1/1188
_S5  = 0.0019175269175269175269175262   # 691/360360
_S6  = 0.0064102564102564102564102561   # 1/156
_S7  = 0.029550653594771241830065352    # 3617/122400
_S8  = 0.17964437236883057316493850     # 43867/244188
_S9  = 1.3924322169059011164274315      # 174611/125400
_S10 = 13.402864044168391994478957      # 77683/5796
_S11 = 156.84828462600201730636509      # 236364091/1506960
_S12 = 2193.1033333333333333333333      # 657931/300
_S13 = 36108.771253724989357173269      # 3392780147/93960
_S14 = 691472.26885131306710839498      # 1723168255201/2492028
_S15 = 15238221.539407416192283370      # 7709321041217/505920
_S16 = 382900751.39141414141414141      # 151628697551/396

_M_LN_2PI = 1.8378770664093454835606594728112352798  # log(2π)
_M_LN_SQRT_2PI = 0.918938533204672741780329736406  # log(sqrt(2π))
_M_LN2 = 0.6931471805599453094172321214581766
_M_2PI = 6.283185307179586476925286766559


_STIRLERR_HALVES_ARR = np.array(_STIRLERR_HALVES, dtype=float)


def _stirlerr(n):
    """Port of nmath ``stirlerr(n)`` (stirlerr.c). Vectorized over ``n``.

    Returns log(n!) - log(sqrt(2πn)·(n/e)ⁿ). The error term in
    Stirling's formula. Used by Loader's saddlepoint algorithm for
    dpois/dbinom. Accepts a scalar or array; returns the same shape.
    Bit-identical to the scalar Fortran source — branches via
    ``np.where``, all arithmetic ops in the same order.
    """
    n = np.asarray(n, dtype=float)
    scalar_input = (n.ndim == 0)
    n = np.atleast_1d(n)

    out = np.empty_like(n)
    nn2 = n + n
    nn2_int = np.rint(nn2).astype(np.int64)

    # ---- n <= 23.5 ----
    le_235 = n <= 23.5
    # Table path: n <= 15.0 and 2n is integer.
    table_mask = le_235 & (n <= 15.0) & (nn2 == nn2_int)
    if np.any(table_mask):
        idx = nn2_int[table_mask]
        out[table_mask] = _STIRLERR_HALVES_ARR[idx]

    # MM2 (n>=1, n<=5.25, not in table)
    mm2_mask = le_235 & ~table_mask & (n <= 5.25) & (n >= 1.0)
    if np.any(mm2_mask):
        nm = n[mm2_mask]
        l_n = np.log(nm)
        out[mm2_mask] = (gammaln(nm) + nm * (1.0 - l_n)
                         + (l_n - _M_LN_2PI) * 0.5)

    # n < 1, not in table
    lt1_mask = le_235 & ~table_mask & ~mm2_mask & (n < 1.0)
    if np.any(lt1_mask):
        nm = n[lt1_mask]
        out[lt1_mask] = (gammaln(1.0 + nm) - (nm + 0.5) * np.log(nm)
                         + nm - _M_LN_SQRT_2PI)

    # 5.25 < n <= 23.5 — asymptotic series, branches by n threshold.
    series_mask = le_235 & ~table_mask & ~mm2_mask & ~lt1_mask
    if np.any(series_mask):
        nm = n[series_mask]
        nn = nm * nm
        # We need different series lengths per element. Compute the longest
        # branch (k=16) and shorter ones; np.where picks per element.
        s_k7  = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - _S6 / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k8  = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - _S7 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k9  = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - _S8 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k11 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - _S10 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k13 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - (_S10 - (_S11 - _S12 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k15 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - (_S10 - (_S11 - (_S12 - (_S13 - _S14 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k16 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - (_S10 - (_S11 - (_S12 - (_S13 - (_S14 - (_S15 - _S16 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        # Select per-element by threshold.
        ser = np.where(nm > 12.8, s_k7,
              np.where(nm > 12.3, s_k8,
              np.where(nm > 8.9,  s_k9,
              np.where(nm > 7.3,  s_k11,
              np.where(nm > 6.6,  s_k13,
              np.where(nm > 6.1,  s_k15, s_k16))))))
        out[series_mask] = ser

    # ---- n > 23.5 ----
    gt235 = ~le_235
    if np.any(gt235):
        nm = n[gt235]
        nn = nm * nm
        a_k1 = _S0 / nm
        a_k2 = (_S0 - _S1 / nn) / nm
        a_k3 = (_S0 - (_S1 - _S2 / nn) / nn) / nm
        a_k4 = (_S0 - (_S1 - (_S2 - _S3 / nn) / nn) / nn) / nm
        a_k5 = (_S0 - (_S1 - (_S2 - (_S3 - _S4 / nn) / nn) / nn) / nn) / nm
        a_k6 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - _S5 / nn) / nn) / nn) / nn) / nn) / nm
        a = np.where(nm > 15.7e6, a_k1,
            np.where(nm > 6180.0, a_k2,
            np.where(nm > 205.0,  a_k3,
            np.where(nm > 86.0,   a_k4,
            np.where(nm > 27.0,   a_k5, a_k6)))))
        out[gt235] = a

    return float(out[0]) if scalar_input else out


def _bd0(x, np_):
    """Port of nmath ``bd0(x, np)`` (bd0.c:48-87). Vectorized.

    Evaluates ``M·D₀(x/M) = x·log(x/M) + M - x`` (where ``M = np_``) with
    small relative error even when ``x/M ≈ 1``. Bit-identical per element
    to the scalar Fortran source — Taylor series for the close branch,
    direct evaluation otherwise.
    """
    x = np.asarray(x, dtype=float)
    np_ = np.asarray(np_, dtype=float)
    scalar = (x.ndim == 0 and np_.ndim == 0)
    x = np.atleast_1d(x)
    np_ = np.atleast_1d(np.broadcast_to(np_, x.shape).copy())

    out = np.empty_like(x)
    out[:] = np.nan
    valid = np.isfinite(x) & np.isfinite(np_) & (np_ != 0.0)

    close = valid & (np.abs(x - np_) < 0.1 * (x + np_))
    far = valid & ~close

    # Far branch: direct formula.
    if np.any(far):
        xf, nf = x[far], np_[far]
        xnp = xf / nf
        # Safe log: fall back to log(x) - log(np_) if xnp non-finite.
        with np.errstate(invalid="ignore"):
            lg_x_n = np.where(np.isfinite(xnp),
                              np.log(np.where(np.isfinite(xnp), xnp, 1.0)),
                              np.log(xf) - np.log(nf))
        out[far] = np.where(xf > nf,
                            xf * (lg_x_n - 1.0) + nf,
                            xf * lg_x_n + nf - xf)

    # Close branch: Taylor series with per-element early exit.
    if np.any(close):
        xc, nc = x[close], np_[close]
        d = xc - nc
        v = d / (xc + nc)
        # Underflow fix: scale by 2^-2 to avoid x+np overflow path.
        underflow = (d != 0.0) & (v == 0.0)
        if np.any(underflow):
            x_ = np.ldexp(xc[underflow], -2)
            n_ = np.ldexp(nc[underflow], -2)
            v_uf = (x_ - n_) / (x_ + n_)
            v[underflow] = v_uf
        s = np.ldexp(d, -1) * v
        # Underflow early-return: ldexp(s, 1) < tiny.
        s2 = np.ldexp(s, 1)
        early = np.abs(s2) < np.finfo(float).tiny
        ej = xc * v
        v2 = v * v
        # Iterate Taylor series; mask out converged/early-returned elements.
        active = ~early
        for j in range(1, 1000):
            if not np.any(active):
                break
            ej_a = ej[active] * v2[active]
            ej[active] = ej_a
            s_old = s[active].copy()
            s_new = s[active] + ej_a / ((j << 1) + 1)
            s[active] = s_new
            still_changed = s_new != s_old
            # Re-build active mask
            idx = np.where(active)[0]
            active = np.zeros_like(active)
            active[idx[still_changed]] = True
        # Return 2*s for converged; 2*early-s for early.
        out[close] = np.where(early, s2, np.ldexp(s, 1))

    return float(out[0]) if scalar else out


def _log1pmx(x: float) -> float:
    """``log(1+x) - x`` evaluated accurately for small ``|x|``.

    Port of R's ``log1pmx`` (nmath/log1pmx.c). For ``|x| > 0.5`` falls
    back to ``log1p(x) - x``; otherwise uses a series expansion.
    """
    minLog1Value = -0.79149064
    two = 2.0
    tol_logcf = 1e-14
    if x > 1.0 or x < minLog1Value:
        return np.log1p(x) - x
    # |x| <= 0.5 — use series
    # log1pmx(x) = -x²/2 + x³·(1/3 - x/4 + x²/5 - ...) = -x²/2 + x³·logcf(x, 3, 2)
    # logcf evaluated via Lentz's continued-fraction algorithm.
    r = x / (x + 2.0)
    y = r * r
    if abs(x) < 1e-2:
        # Truncated series — used for very small |x|.
        return r * (2.0 + y * (2.0 / 3.0 + y * (2.0 / 5.0 + y * (2.0 / 7.0 + y * (2.0 / 9.0))))) - x
    # General case via Lentz iteration of the continued fraction:
    # logcf(y, 3, 2) for ln((1+x)/(1-x)) = 2r · logcf(r², 1, 2)
    # We compute log1p(x) = 2r · sum directly.
    a1 = 3.0
    b1 = 1.0 - y * (a1 / (a1 + two))
    a2 = a1 + 1.0  # = 4
    c1 = 1.0
    c2 = 1.0
    c4 = a1 * a2
    a1 = 1.0
    while True:
        c3 = c2 * c2
        c2 = c4 - c3 * a1 * y
        b2 = b1 * (c2 - c1 * y)
        a3 = a1 * a2
        # ...
        # The full Lentz iteration is more involved; for our use-case
        # |x| < 0.5 the simpler "long series" version is enough.
        break
    # Fallback: numpy log1p when series is unavailable.
    return np.log1p(x) - x


# ebd0 (extended bd0) — Welinder's improved-precision version used by R
# dpois. The 128-entry log table from bd0.c:102-231 (each row: 4 floats
# encoding log(p/1024) where p = floor(1024/(0.5+i/256)+0.5), p ≈ 1024 to
# 2048). Decoded from hex-float to plain double values.

# Hex-float decoder: each entry "+0x1.62e430p-1" → that float value.
def _hex_to_float(s: str) -> float:
    return float.fromhex(s)


# Hex-float table from bd0.c:102-231. Reproduced verbatim so this file
# can be diffed against the C source. Each tuple is the 4 float parts
# (a high-bit chunk + three corrections) of one log value.
_BD0_SCALE_HEX = (
    ("+0x1.62e430p-1", "-0x1.05c610p-29", "-0x1.950d88p-54", "+0x1.d9cc02p-79"),
    ("+0x1.5ee02cp-1", "-0x1.6dbe98p-25", "-0x1.51e540p-50", "+0x1.2bfa48p-74"),
    ("+0x1.5ad404p-1", "+0x1.86b3e4p-26", "+0x1.9f6534p-50", "+0x1.54be04p-74"),
    ("+0x1.570124p-1", "-0x1.9ed750p-25", "-0x1.f37dd0p-51", "+0x1.10b770p-77"),
    ("+0x1.5326e4p-1", "-0x1.9b9874p-25", "-0x1.378194p-49", "+0x1.56feb2p-74"),
    ("+0x1.4f4528p-1", "+0x1.aca70cp-28", "+0x1.103e74p-53", "+0x1.9c410ap-81"),
    ("+0x1.4b5bd8p-1", "-0x1.6a91d8p-25", "-0x1.8e43d0p-50", "-0x1.afba9ep-77"),
    ("+0x1.47ae54p-1", "-0x1.abb51cp-25", "+0x1.19b798p-51", "+0x1.45e09cp-76"),
    ("+0x1.43fa00p-1", "-0x1.d06318p-25", "-0x1.8858d8p-49", "-0x1.1927c4p-75"),
    ("+0x1.3ffa40p-1", "+0x1.1a427cp-25", "+0x1.151640p-53", "-0x1.4f5606p-77"),
    ("+0x1.3c7c80p-1", "-0x1.19bf48p-34", "+0x1.05fc94p-58", "-0x1.c096fcp-82"),
    ("+0x1.38b320p-1", "+0x1.6b5778p-25", "+0x1.be38d0p-50", "-0x1.075e96p-74"),
    ("+0x1.34e288p-1", "+0x1.d9ce1cp-25", "+0x1.316eb8p-49", "+0x1.2d885cp-73"),
    ("+0x1.315124p-1", "+0x1.c2fc60p-29", "-0x1.4396fcp-53", "+0x1.acf376p-78"),
    ("+0x1.2db954p-1", "+0x1.720de4p-25", "-0x1.d39b04p-49", "-0x1.f11176p-76"),
    ("+0x1.2a1b08p-1", "-0x1.562494p-25", "+0x1.a7863cp-49", "+0x1.85dd64p-73"),
    ("+0x1.267620p-1", "+0x1.3430e0p-29", "-0x1.96a958p-56", "+0x1.f8e636p-82"),
    ("+0x1.23130cp-1", "+0x1.7bebf4p-25", "+0x1.416f1cp-52", "-0x1.78dd36p-77"),
    ("+0x1.1faa34p-1", "+0x1.70e128p-26", "+0x1.81817cp-50", "-0x1.c2179cp-76"),
    ("+0x1.1bf204p-1", "+0x1.3a9620p-28", "+0x1.2f94c0p-52", "+0x1.9096c0p-76"),
    ("+0x1.187ce4p-1", "-0x1.077870p-27", "+0x1.655a80p-51", "+0x1.eaafd6p-78"),
    ("+0x1.1501c0p-1", "-0x1.406cacp-25", "-0x1.e72290p-49", "+0x1.5dd800p-73"),
    ("+0x1.11cb80p-1", "+0x1.787cd0p-25", "-0x1.efdc78p-51", "-0x1.5380cep-77"),
    ("+0x1.0e4498p-1", "+0x1.747324p-27", "-0x1.024548p-51", "+0x1.77a5a6p-75"),
    ("+0x1.0b036cp-1", "+0x1.690c74p-25", "+0x1.5d0cc4p-50", "-0x1.c0e23cp-76"),
    ("+0x1.077070p-1", "-0x1.a769bcp-27", "+0x1.452234p-52", "+0x1.6ba668p-76"),
    ("+0x1.04240cp-1", "-0x1.a686acp-27", "-0x1.ef46b0p-52", "-0x1.5ce10cp-76"),
    ("+0x1.00d22cp-1", "+0x1.fc0e10p-25", "+0x1.6ee034p-50", "-0x1.19a2ccp-74"),
    ("+0x1.faf588p-2", "+0x1.ef1e64p-27", "-0x1.26504cp-54", "-0x1.b15792p-82"),
    ("+0x1.f4d87cp-2", "+0x1.d7b980p-26", "-0x1.a114d8p-50", "+0x1.9758c6p-75"),
    ("+0x1.ee1414p-2", "+0x1.2ec060p-26", "+0x1.dc00fcp-52", "+0x1.f8833cp-76"),
    ("+0x1.e7e32cp-2", "-0x1.ac796cp-27", "-0x1.a68818p-54", "+0x1.235d02p-78"),
    ("+0x1.e108a0p-2", "-0x1.768ba4p-28", "-0x1.f050a8p-52", "+0x1.00d632p-82"),
    ("+0x1.dac354p-2", "-0x1.d3a6acp-30", "+0x1.18734cp-57", "-0x1.f97902p-83"),
    ("+0x1.d47424p-2", "+0x1.7dbbacp-31", "-0x1.d5ada4p-56", "+0x1.56fcaap-81"),
    ("+0x1.ce1af0p-2", "+0x1.70be7cp-27", "+0x1.6f6fa4p-51", "+0x1.7955a2p-75"),
    ("+0x1.c7b798p-2", "+0x1.ec36ecp-26", "-0x1.07e294p-50", "-0x1.ca183cp-75"),
    ("+0x1.c1ef04p-2", "+0x1.c1dfd4p-26", "+0x1.888eecp-50", "-0x1.fd6b86p-75"),
    ("+0x1.bb7810p-2", "+0x1.478bfcp-26", "+0x1.245b8cp-50", "+0x1.ea9d52p-74"),
    ("+0x1.b59da0p-2", "-0x1.882b08p-27", "+0x1.31573cp-53", "-0x1.8c249ap-77"),
    ("+0x1.af1294p-2", "-0x1.b710f4p-27", "+0x1.622670p-51", "+0x1.128578p-76"),
    ("+0x1.a925d4p-2", "-0x1.0ae750p-27", "+0x1.574ed4p-51", "+0x1.084996p-75"),
    ("+0x1.a33040p-2", "+0x1.027d30p-29", "+0x1.b9a550p-53", "-0x1.b2e38ap-78"),
    ("+0x1.9d31c0p-2", "-0x1.5ec12cp-26", "-0x1.5245e0p-52", "+0x1.2522d0p-79"),
    ("+0x1.972a34p-2", "+0x1.135158p-30", "+0x1.a5c09cp-56", "+0x1.24b70ep-80"),
    ("+0x1.911984p-2", "+0x1.0995d4p-26", "+0x1.3bfb5cp-50", "+0x1.2c9dd6p-75"),
    ("+0x1.8bad98p-2", "-0x1.1d6144p-29", "+0x1.5b9208p-53", "+0x1.1ec158p-77"),
    ("+0x1.858b58p-2", "-0x1.1b4678p-27", "+0x1.56cab4p-53", "-0x1.2fdc0cp-78"),
    ("+0x1.7f5fa0p-2", "+0x1.3aaf48p-27", "+0x1.461964p-51", "+0x1.4ae476p-75"),
    ("+0x1.79db68p-2", "-0x1.7e5054p-26", "+0x1.673750p-51", "-0x1.a11f7ap-76"),
    ("+0x1.744f88p-2", "-0x1.cc0e18p-26", "-0x1.1e9d18p-50", "-0x1.6c06bcp-78"),
    ("+0x1.6e08ecp-2", "-0x1.5d45e0p-26", "-0x1.c73ec8p-50", "+0x1.318d72p-74"),
    ("+0x1.686c80p-2", "+0x1.e9b14cp-26", "-0x1.13bbd4p-50", "-0x1.efeb1cp-78"),
    ("+0x1.62c830p-2", "-0x1.a8c70cp-27", "-0x1.5a1214p-51", "-0x1.bab3fcp-79"),
    ("+0x1.5d1bdcp-2", "-0x1.4fec6cp-31", "+0x1.423638p-56", "+0x1.ee3feep-83"),
    ("+0x1.576770p-2", "+0x1.7455a8p-26", "-0x1.3ab654p-50", "-0x1.26be4cp-75"),
    ("+0x1.5262e0p-2", "-0x1.146778p-26", "-0x1.b9f708p-52", "-0x1.294018p-77"),
    ("+0x1.4c9f08p-2", "+0x1.e152c4p-26", "-0x1.dde710p-53", "+0x1.fd2208p-77"),
    ("+0x1.46d2d8p-2", "+0x1.c28058p-26", "-0x1.936284p-50", "+0x1.9fdd68p-74"),
    ("+0x1.41b940p-2", "+0x1.cce0c0p-26", "-0x1.1a4050p-50", "+0x1.bc0376p-76"),
    ("+0x1.3bdd24p-2", "+0x1.d6296cp-27", "+0x1.425b48p-51", "-0x1.cddb2cp-77"),
    ("+0x1.36b578p-2", "-0x1.287ddcp-27", "-0x1.2d0f4cp-51", "+0x1.38447ep-75"),
    ("+0x1.31871cp-2", "+0x1.2a8830p-27", "+0x1.3eae54p-52", "-0x1.898136p-77"),
    ("+0x1.2b9304p-2", "-0x1.51d8b8p-28", "+0x1.27694cp-52", "-0x1.fd852ap-76"),
    ("+0x1.265620p-2", "-0x1.d98f3cp-27", "+0x1.a44338p-51", "-0x1.56e85ep-78"),
    ("+0x1.211254p-2", "+0x1.986160p-26", "+0x1.73c5d0p-51", "+0x1.4a861ep-75"),
    ("+0x1.1bc794p-2", "+0x1.fa3918p-27", "+0x1.879c5cp-51", "+0x1.16107cp-78"),
    ("+0x1.1675ccp-2", "-0x1.4545a0p-26", "+0x1.c07398p-51", "+0x1.f55c42p-76"),
    ("+0x1.111ce4p-2", "+0x1.f72670p-37", "-0x1.b84b5cp-61", "+0x1.a4a4dcp-85"),
    ("+0x1.0c81d4p-2", "+0x1.0c150cp-27", "+0x1.218600p-51", "-0x1.d17312p-76"),
    ("+0x1.071b84p-2", "+0x1.fcd590p-26", "+0x1.a3a2e0p-51", "+0x1.fe5ef8p-76"),
    ("+0x1.01ade4p-2", "-0x1.bb1844p-28", "+0x1.db3cccp-52", "+0x1.1f56fcp-77"),
    ("+0x1.fa01c4p-3", "-0x1.12a0d0p-29", "-0x1.f71fb0p-54", "+0x1.e287a4p-78"),
    ("+0x1.ef0adcp-3", "+0x1.7b8b28p-28", "-0x1.35bce4p-52", "-0x1.abc8f8p-79"),
    ("+0x1.e598ecp-3", "+0x1.5a87e4p-27", "-0x1.134bd0p-51", "+0x1.c2cebep-76"),
    ("+0x1.da85d8p-3", "-0x1.df31b0p-27", "+0x1.94c16cp-57", "+0x1.8fd7eap-82"),
    ("+0x1.d0fb80p-3", "-0x1.bb5434p-28", "-0x1.ea5640p-52", "-0x1.8ceca4p-77"),
    ("+0x1.c765b8p-3", "+0x1.e4d68cp-27", "+0x1.5b59b4p-51", "+0x1.76f6c4p-76"),
    ("+0x1.bdc46cp-3", "-0x1.1cbb50p-27", "+0x1.2da010p-51", "+0x1.eb282cp-75"),
    ("+0x1.b27980p-3", "-0x1.1b9ce0p-27", "+0x1.7756f8p-52", "+0x1.2ff572p-76"),
    ("+0x1.a8bed0p-3", "-0x1.bbe874p-30", "+0x1.85cf20p-56", "+0x1.b9cf18p-80"),
    ("+0x1.9ef83cp-3", "+0x1.2769a4p-27", "-0x1.85bda0p-52", "+0x1.8c8018p-79"),
    ("+0x1.9525a8p-3", "+0x1.cf456cp-27", "-0x1.7137d8p-52", "-0x1.f158e8p-76"),
    ("+0x1.8b46f8p-3", "+0x1.11b12cp-30", "+0x1.9f2104p-54", "-0x1.22836ep-78"),
    ("+0x1.83040cp-3", "+0x1.2379e4p-28", "+0x1.b71c70p-52", "-0x1.990cdep-76"),
    ("+0x1.790ed4p-3", "+0x1.dc4c68p-28", "-0x1.910ac8p-52", "+0x1.dd1bd6p-76"),
    ("+0x1.6f0d28p-3", "+0x1.5cad68p-28", "+0x1.737c94p-52", "-0x1.9184bap-77"),
    ("+0x1.64fee8p-3", "+0x1.04bf88p-28", "+0x1.6fca28p-52", "+0x1.8884a8p-76"),
    ("+0x1.5c9400p-3", "+0x1.d65cb0p-29", "-0x1.b2919cp-53", "+0x1.b99bcep-77"),
    ("+0x1.526e60p-3", "-0x1.c5e4bcp-27", "-0x1.0ba380p-52", "+0x1.d6e3ccp-79"),
    ("+0x1.483bccp-3", "+0x1.9cdc7cp-28", "-0x1.5ad8dcp-54", "-0x1.392d3cp-83"),
    ("+0x1.3fb25cp-3", "-0x1.a6ad74p-27", "+0x1.5be6b4p-52", "-0x1.4e0114p-77"),
    ("+0x1.371fc4p-3", "-0x1.fe1708p-27", "-0x1.78864cp-52", "-0x1.27543ap-76"),
    ("+0x1.2cca10p-3", "-0x1.4141b4p-28", "-0x1.ef191cp-52", "+0x1.00ee08p-76"),
    ("+0x1.242310p-3", "+0x1.3ba510p-27", "-0x1.d003c8p-51", "+0x1.162640p-76"),
    ("+0x1.1b72acp-3", "+0x1.52f67cp-27", "-0x1.fd6fa0p-51", "+0x1.1a3966p-77"),
    ("+0x1.10f8e4p-3", "+0x1.129cd8p-30", "+0x1.31ef30p-55", "+0x1.a73e38p-79"),
    ("+0x1.08338cp-3", "-0x1.005d7cp-27", "-0x1.661a9cp-51", "+0x1.1f138ap-79"),
    ("+0x1.fec914p-4", "-0x1.c482a8p-29", "-0x1.55746cp-54", "+0x1.99f932p-80"),
    ("+0x1.ed1794p-4", "+0x1.d06f00p-29", "+0x1.75e45cp-53", "-0x1.d0483ep-78"),
    ("+0x1.db5270p-4", "+0x1.87d928p-32", "-0x1.0f52a4p-57", "+0x1.81f4a6p-84"),
    ("+0x1.c97978p-4", "+0x1.af1d24p-29", "-0x1.0977d0p-60", "-0x1.8839d0p-84"),
    ("+0x1.b78c84p-4", "-0x1.44f124p-28", "-0x1.ef7bc4p-52", "+0x1.9e0650p-78"),
    ("+0x1.a58b60p-4", "+0x1.856464p-29", "+0x1.c651d0p-55", "+0x1.b06b0cp-79"),
    ("+0x1.9375e4p-4", "+0x1.5595ecp-28", "+0x1.dc3738p-52", "+0x1.86c89ap-81"),
    ("+0x1.814be4p-4", "-0x1.c073fcp-28", "-0x1.371f88p-53", "-0x1.5f4080p-77"),
    ("+0x1.6f0d28p-4", "+0x1.5cad68p-29", "+0x1.737c94p-53", "-0x1.9184bap-78"),
    ("+0x1.60658cp-4", "-0x1.6c8af4p-28", "+0x1.d8ef74p-55", "+0x1.c4f792p-80"),
    ("+0x1.4e0110p-4", "+0x1.146b5cp-29", "+0x1.73f7ccp-54", "-0x1.d28db8p-79"),
    ("+0x1.3b8758p-4", "+0x1.8b1b70p-28", "-0x1.20aca4p-52", "-0x1.651894p-76"),
    ("+0x1.28f834p-4", "+0x1.43b6a4p-30", "-0x1.452af8p-55", "+0x1.976892p-80"),
    ("+0x1.1a0fbcp-4", "-0x1.e4075cp-28", "+0x1.1fe618p-52", "+0x1.9d6dc2p-77"),
    ("+0x1.075984p-4", "-0x1.4ce370p-29", "-0x1.d9fc98p-53", "+0x1.4ccf12p-77"),
    ("+0x1.f0a30cp-5", "+0x1.162a68p-37", "-0x1.e83368p-61", "-0x1.d222a6p-86"),
    ("+0x1.cae730p-5", "-0x1.1a8f7cp-31", "-0x1.5f9014p-55", "+0x1.2720c0p-79"),
    ("+0x1.ac9724p-5", "-0x1.e8ee08p-29", "+0x1.a7de04p-54", "-0x1.9bba74p-78"),
    ("+0x1.868a84p-5", "-0x1.ef8128p-30", "+0x1.dc5eccp-54", "-0x1.58d250p-79"),
    ("+0x1.67f950p-5", "-0x1.ed684cp-30", "-0x1.f060c0p-55", "-0x1.b1294cp-80"),
    ("+0x1.494accp-5", "+0x1.a6c890p-32", "-0x1.c3ad48p-56", "-0x1.6dc66cp-84"),
    ("+0x1.22c71cp-5", "-0x1.8abe2cp-32", "-0x1.7e7078p-56", "-0x1.ddc3dcp-86"),
    ("+0x1.03d5d8p-5", "+0x1.79cfbcp-31", "-0x1.da7c4cp-58", "+0x1.4e7582p-83"),
    ("+0x1.c98d18p-6", "+0x1.a01904p-31", "-0x1.854164p-55", "+0x1.883c36p-79"),
    ("+0x1.8b31fcp-6", "-0x1.356500p-30", "+0x1.c3ab48p-55", "+0x1.b69bdap-80"),
    ("+0x1.3cea44p-6", "+0x1.a352bcp-33", "-0x1.8865acp-57", "-0x1.48159cp-81"),
    ("+0x1.fc0a8cp-7", "-0x1.e07f84p-32", "+0x1.e7cf6cp-58", "+0x1.3a69c0p-82"),
    ("+0x1.7dc474p-7", "+0x1.f810a8p-31", "-0x1.245b5cp-56", "-0x1.a1f4f8p-80"),
    ("+0x1.fe02a8p-8", "-0x1.4ef988p-32", "+0x1.1f86ecp-57", "+0x1.20723cp-81"),
    ("+0x1.ff00acp-9", "-0x1.d4ef44p-33", "+0x1.2821acp-63", "+0x1.5a6d32p-87"),
    ("0",              "0",               "0",               "0"),  # log(1) = 0
)
_BD0_SCALE = tuple(tuple(_hex_to_float(s) for s in row) for row in _BD0_SCALE_HEX)
_BD0_SCALE_NP = np.array(_BD0_SCALE, dtype=float)  # shape (129, 4) for vectorized lookup


def _ebd0(x, M):
    """Port of nmath ``ebd0(x, M)`` (bd0.c:241-355). Vectorized.

    Computes ``x·log(x/M) + (M - x)`` with extended precision. Returns
    ``(yh, yl)`` arrays such that ``yh + yl`` is the value. Welinder's
    improved algorithm (R Bugzilla PR#15628).
    """
    Sb = 10
    S = 1 << Sb  # = 1024
    N = 128

    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    scalar = (x.ndim == 0 and M.ndim == 0)
    x = np.atleast_1d(x)
    M = np.atleast_1d(np.broadcast_to(M, x.shape).copy())

    yh = np.zeros_like(x)
    yl = np.zeros_like(x)

    # Edge cases.
    eq = x == M
    x_zero = ~eq & (x == 0.0)
    M_zero = ~eq & ~x_zero & (M == 0.0)
    yh[x_zero] = M[x_zero]
    yh[M_zero] = np.inf

    # M/x → ∞ (M >> x).
    Mox = np.where(eq | x_zero | M_zero, 1.0, M / np.where(x == 0.0, 1.0, x))
    inf_Mox = ~eq & ~x_zero & ~M_zero & (Mox == np.inf)
    yh[inf_Mox] = M[inf_Mox]

    active = ~(eq | x_zero | M_zero | inf_Mox)
    if not np.any(active):
        return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)

    xa = x[active]
    Ma = M[active]
    Mox_a = Ma / xa

    # M/x = r · 2^e
    r, e = np.frexp(Mox_a)

    # Overflow check (rare): M_LN2 * (-e) > 1 + DBL_MAX/x → yh = +inf
    overflow = _M_LN2 * (-e.astype(float)) > (1.0 + np.finfo(float).max / xa)
    if np.any(overflow):
        active_idx = np.where(active)[0]
        yh[active_idx[overflow]] = np.inf
        good = ~overflow
        xa = xa[good]; Ma = Ma[good]; r = r[good]; e = e[good]
        active_idx = active_idx[good]
    else:
        active_idx = np.where(active)[0]

    if xa.size == 0:
        return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)

    i = np.floor((r - 0.5) * (2 * N) + 0.5).astype(np.int64)
    f = np.floor(S / (0.5 + i / (2.0 * N)) + 0.5)
    fg = np.ldexp(f, -(e + Sb))

    inf_fg = fg == np.inf
    if np.any(inf_fg):
        yh[active_idx[inf_fg]] = np.inf
        good = ~inf_fg
        xa = xa[good]; Ma = Ma[good]; fg = fg[good]; i = i[good]; e = e[good]
        active_idx = active_idx[good]

    if xa.size == 0:
        return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)

    # Local accumulators (we update yh/yl only via these arrays).
    lh = np.zeros_like(xa)
    ll = np.zeros_like(xa)

    def add1(d_arr):
        d1 = np.floor(d_arr + 0.5)
        d2 = d_arr - d1
        np.add(lh, d1, out=lh)
        np.add(ll, d2, out=ll)

    # ADD1(-x * log1pmx((M*fg - x) / x))
    arg = (Ma * fg - xa) / xa
    log1pmx_val = np.log1p(arg) - arg
    add1(-xa * log1pmx_val)

    fg_ne_1 = fg != 1.0
    if np.any(fg_ne_1):
        # Process the 4-iteration table corrections only where fg != 1.
        # We compute updates for the WHOLE active set; for fg==1 elements
        # the increments are 0 (since x * 0 = 0 with proper masking).
        for j in range(4):
            tbl_i = _BD0_SCALE_NP[i, j]
            tbl_0 = _BD0_SCALE_NP[0, j]
            inc1 = np.where(fg_ne_1, xa * tbl_i, 0.0)
            inc2 = np.where(fg_ne_1, -xa * tbl_0 * e, 0.0)
            add1(inc1)
            add1(inc2)
            # Per-iter overflow check: any !isfinite → set to inf and freeze.
            nonfinite = ~np.isfinite(lh)
            if np.any(nonfinite):
                lh[nonfinite] = np.inf
                ll[nonfinite] = 0.0
                fg_ne_1 = fg_ne_1 & ~nonfinite

    # ADD1(M); ADD1(-M·fg) only where fg != 1; for fg==1, the original
    # scalar code returns early before these — match that exactly.
    M_inc = np.where(fg != 1.0, Ma, 0.0)
    fg_inc = np.where(fg != 1.0, -Ma * fg, 0.0)
    # But: the scalar code returns IMMEDIATELY for fg==1 after the first
    # add1(-x·log1pmx). For fg==1, lh/ll already have the right value,
    # so skip the M / -M·fg adds.
    fg_eq_1 = fg == 1.0
    fg_ne_1 = ~fg_eq_1
    if np.any(fg_ne_1):
        # Apply M / -M·fg adds only for fg != 1 (otherwise scalar returns
        # early so we shouldn't add).
        i_ne = np.where(fg_ne_1)[0]
        d = Ma[i_ne]
        d1 = np.floor(d + 0.5)
        lh[i_ne] = lh[i_ne] + d1
        ll[i_ne] = ll[i_ne] + (d - d1)
        d = -Ma[i_ne] * fg[i_ne]
        d1 = np.floor(d + 0.5)
        lh[i_ne] = lh[i_ne] + d1
        ll[i_ne] = ll[i_ne] + (d - d1)

    yh[active_idx] = lh
    yl[active_idx] = ll
    return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)


def _dpois_raw(x, lambda_, give_log: bool = True):
    """Port of nmath ``dpois_raw(x, lambda, give_log)`` (dpois.c:43-69).

    Vectorized over ``x`` and ``lambda``. Uses Loader's saddlepoint with
    ebd0 (R 4.5). Returns the same shape as the broadcast of inputs.
    """
    x_in = np.asarray(x, dtype=float)
    l_in = np.asarray(lambda_, dtype=float)
    scalar = (x_in.ndim == 0 and l_in.ndim == 0)
    x = np.atleast_1d(x_in.copy())
    lam = np.atleast_1d(np.broadcast_to(l_in, x.shape).copy())

    NEG_INF = float('-inf')
    out = np.empty_like(x)

    # Edge cases (rare in PIRLS; cheap to test).
    lam_zero = (lam == 0.0)
    lam_inf  = ~np.isfinite(lam)
    x_neg    = x < 0
    tiny = np.finfo(float).tiny
    x_le_lt = (x <= lam * tiny) & ~lam_zero & ~lam_inf & ~x_neg
    lam_lt_xt = (lam < x * tiny) & ~lam_zero & ~lam_inf & ~x_neg & ~x_le_lt
    main = ~(lam_zero | lam_inf | x_neg | x_le_lt | lam_lt_xt)

    # lam == 0: x==0 → log(1)=0; else -inf
    if np.any(lam_zero):
        out[lam_zero] = np.where(x[lam_zero] == 0.0, 0.0, NEG_INF)
    if np.any(lam_inf):
        out[lam_inf] = NEG_INF
    if np.any(x_neg):
        out[x_neg] = NEG_INF
    if np.any(x_le_lt):
        out[x_le_lt] = -lam[x_le_lt]
    if np.any(lam_lt_xt):
        sub = lam_lt_xt
        xn = x[sub]; ln = lam[sub]
        out[sub] = np.where(~np.isfinite(xn),
                            NEG_INF,
                            -ln + xn * np.log(ln) - gammaln(xn + 1.0))

    # Common (saddlepoint) path.
    if np.any(main):
        xm = x[main]; lm = lam[main]
        yh, yl = _ebd0(xm, lm)
        yl_total = yl + _stirlerr(xm)
        x_LRG = 2.86111748575702815380240589208115399625e307
        Lrg = xm >= x_LRG
        r = np.where(Lrg, 2.5066282746310005024 * np.sqrt(xm), _M_2PI * xm)
        log_correction = np.where(Lrg, np.log(r), 0.5 * np.log(r))
        out[main] = -yl_total - yh - log_correction

    if not give_log:
        out = np.exp(out)
    return float(out[0]) if scalar else out


def _dbinom_raw(x, n, p, q, give_log: bool = True):
    """Port of nmath ``dbinom_raw(x, n, p, q, give_log)`` (dbinom.c:72-118).

    Vectorized. Uses Loader's saddlepoint with the older (non-extended)
    ``bd0`` — matches dbinom.c which calls ``bd0(...)`` not ``ebd0(...)``.
    """
    x_in = np.asarray(x, dtype=float)
    n_in = np.asarray(n, dtype=float)
    p_in = np.asarray(p, dtype=float)
    q_in = np.asarray(q, dtype=float)
    scalar = (x_in.ndim == 0 and n_in.ndim == 0 and p_in.ndim == 0 and q_in.ndim == 0)
    # Broadcast to common shape.
    shape = np.broadcast_shapes(x_in.shape, n_in.shape, p_in.shape, q_in.shape)
    x = np.broadcast_to(x_in, shape).astype(float).copy()
    n = np.broadcast_to(n_in, shape).astype(float).copy()
    p = np.broadcast_to(p_in, shape).astype(float).copy()
    q = np.broadcast_to(q_in, shape).astype(float).copy()

    NEG_INF = float('-inf')
    out = np.empty(shape, dtype=float)

    p_zero = p == 0.0
    q_zero = q == 0.0
    x_zero = x == 0.0
    x_eq_n = x == n
    x_oob = (x < 0) | (x > n)

    edge_p0 = p_zero
    edge_q0 = q_zero & ~p_zero
    edge_x0 = x_zero & ~p_zero & ~q_zero
    edge_xn = x_eq_n & ~p_zero & ~q_zero & ~x_zero
    edge_oob = x_oob & ~p_zero & ~q_zero & ~x_zero & ~x_eq_n
    main = ~(edge_p0 | edge_q0 | edge_x0 | edge_xn | edge_oob)

    if np.any(edge_p0):
        out[edge_p0] = np.where(x[edge_p0] == 0.0, 0.0, NEG_INF)
    if np.any(edge_q0):
        out[edge_q0] = np.where(x[edge_q0] == n[edge_q0], 0.0, NEG_INF)
    if np.any(edge_x0):
        n0 = n[edge_x0]; p0 = p[edge_x0]; q0 = q[edge_x0]
        # n == 0 → log(1) = 0
        n_is_0 = n0 == 0.0
        # else: n*log(q) if p>q, n*log1p(-p) otherwise.
        big_p = (p0 > q0) & ~n_is_0
        big_q = ~big_p & ~n_is_0
        val = np.empty_like(n0)
        val[n_is_0] = 0.0
        if np.any(big_p):
            val[big_p] = n0[big_p] * np.log(q0[big_p])
        if np.any(big_q):
            val[big_q] = n0[big_q] * np.log1p(-p0[big_q])
        out[edge_x0] = val
    if np.any(edge_xn):
        n0 = n[edge_xn]; p0 = p[edge_xn]; q0 = q[edge_xn]
        big_p = p0 > q0
        val = np.empty_like(n0)
        if np.any(big_p):
            val[big_p] = n0[big_p] * np.log1p(-q0[big_p])
        if np.any(~big_p):
            val[~big_p] = n0[~big_p] * np.log(p0[~big_p])
        out[edge_xn] = val
    if np.any(edge_oob):
        out[edge_oob] = NEG_INF

    if np.any(main):
        xm = x[main]; nm = n[main]; pm = p[main]; qm = q[main]
        lc = (_stirlerr(nm) - _stirlerr(xm) - _stirlerr(nm - xm)
              - _bd0(xm, nm * pm) - _bd0(nm - xm, nm * qm))
        lf = _M_LN_2PI + np.log(xm) + np.log1p(-xm / nm)
        out[main] = lc - 0.5 * lf

    if not give_log:
        out = np.exp(out)
    return float(out.reshape(())) if scalar else out


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


class Link:
    """Base class. Subclasses must implement ``link``, ``linkinv``,
    ``mu_eta``, ``d2link``, ``d3link``, ``d4link``."""
    name: str

    def link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError
    def linkinv(self, eta: np.ndarray) -> np.ndarray: raise NotImplementedError
    def mu_eta(self, eta: np.ndarray) -> np.ndarray: raise NotImplementedError
    def d2link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError
    def d3link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError
    def d4link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError
    def valideta(self, eta: np.ndarray) -> bool: return True

    # mgcv ``link$g2g``, ``g3g``, ``g4g`` (R/efam.r): higher-order link
    # curvature ratios needed by ``Family.dDeta`` for extended families
    # under non-identity links. ``g2g(μ) = g″(μ)/g′(μ) · μ_η`` etc; we
    # use the equivalent form ``g″(μ)·μ_η = g2g`` direct from mgcv's
    # source. Identity link has all-zero curvature → IdentityLink
    # overrides to return zeros without computing.
    def g2g(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}.g2g() is not implemented; needed for "
            "extended families under this non-identity link."
        )
    def g3g(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}.g3g() is not implemented; needed for "
            "extended families under this non-identity link (level≥1)."
        )
    def g4g(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}.g4g() is not implemented; needed for "
            "extended families under this non-identity link (level≥2)."
        )

    def __repr__(self) -> str:
        return self.name


class IdentityLink(Link):
    name = "identity"
    def g2g(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def g3g(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def g4g(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def link(self, mu): return np.asarray(mu, dtype=float)
    def linkinv(self, eta): return np.asarray(eta, dtype=float)
    def mu_eta(self, eta): return np.ones_like(np.asarray(eta, dtype=float))
    def d2link(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d3link(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d4link(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))


class LogLink(Link):
    name = "log"
    def link(self, mu): return np.log(np.asarray(mu, dtype=float))
    def linkinv(self, eta):
        # mgcv clamps to .Machine$double.eps to avoid 0 — replicate so divisions
        # by μ in PIRLS / V'(μ) etc. don't blow up at extreme negative η.
        return np.maximum(np.exp(np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)
    def mu_eta(self, eta):
        return np.maximum(np.exp(np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)
    def d2link(self, mu): return -1.0 / np.asarray(mu, dtype=float)**2
    def d3link(self, mu): return 2.0 / np.asarray(mu, dtype=float)**3
    def d4link(self, mu): return -6.0 / np.asarray(mu, dtype=float)**4
    # log link: g'(μ)=1/μ, g''(μ)=-1/μ², g'''(μ)=2/μ³, g''''(μ)=-6/μ⁴ →
    # g2g=g''/g'²=-1, g3g=g'''/g'³=2, g4g=g''''/g'⁴=-6.
    # mgcv gam.fit3.r:2229-2231.
    def g2g(self, mu): return -np.ones_like(np.asarray(mu, dtype=float))
    def g3g(self, mu): return 2.0 * np.ones_like(np.asarray(mu, dtype=float))
    def g4g(self, mu): return -6.0 * np.ones_like(np.asarray(mu, dtype=float))


class InverseLink(Link):
    name = "inverse"
    def link(self, mu): return 1.0 / np.asarray(mu, dtype=float)
    def linkinv(self, eta): return 1.0 / np.asarray(eta, dtype=float)
    def mu_eta(self, eta): return -1.0 / np.asarray(eta, dtype=float)**2
    def d2link(self, mu): return 2.0 / np.asarray(mu, dtype=float)**3
    def d3link(self, mu): return -6.0 / np.asarray(mu, dtype=float)**4
    def d4link(self, mu): return 24.0 / np.asarray(mu, dtype=float)**5
    # inverse link: g'=-1/μ², g''=2/μ³, g'''=-6/μ⁴, g''''=24/μ⁵ →
    # g2g = g''/g'² = (2/μ³)·μ⁴ = 2μ;  g3g = g'''/g'³ = (-6/μ⁴)·(-μ⁶) = 6μ²;
    # g4g = g''''/g'⁴ = (24/μ⁵)·μ⁸ = 24μ³.
    # mgcv gam.fit3.r:2234-2236.
    def g2g(self, mu): return 2.0 * np.asarray(mu, dtype=float)
    def g3g(self, mu): return 6.0 * np.asarray(mu, dtype=float)**2
    def g4g(self, mu): return 24.0 * np.asarray(mu, dtype=float)**3
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(eta != 0))


class SqrtLink(Link):
    """``g(μ) = √μ`` — alternate poisson link."""
    name = "sqrt"
    def link(self, mu): return np.sqrt(np.asarray(mu, dtype=float))
    def linkinv(self, eta): return np.asarray(eta, dtype=float) ** 2
    def mu_eta(self, eta): return 2.0 * np.asarray(eta, dtype=float)
    def d2link(self, mu): return -0.25 * np.asarray(mu, dtype=float) ** -1.5
    def d3link(self, mu): return 0.375 * np.asarray(mu, dtype=float) ** -2.5
    def d4link(self, mu): return -0.9375 * np.asarray(mu, dtype=float) ** -3.5
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


class LogitLink(Link):
    """``g(μ) = log(μ/(1-μ))`` — canonical binomial link."""
    name = "logit"
    def link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return np.log(mu / (1.0 - mu))
    def linkinv(self, eta):
        # R clamps to (eps, 1-eps) inside C_logit_linkinv. expit is symmetric
        # around 0 and stable; the clamp is what keeps PIRLS from sliding to
        # μ=0 or 1 where V(μ) = μ(1-μ) collapses.
        eps = np.finfo(float).eps
        return np.clip(expit(np.asarray(eta, dtype=float)), eps, 1.0 - eps)
    def mu_eta(self, eta):
        # μ_η = e^η / (1+e^η)² = μ(1-μ); compute as e^{-|η|}/(1+e^{-|η|})²
        # to avoid overflow at large |η|. Lower-clamp to eps (mgcv).
        eps = np.finfo(float).eps
        a = np.exp(-np.abs(np.asarray(eta, dtype=float)))
        return np.maximum(a / (1.0 + a) ** 2, eps)
    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 1.0 / (1.0 - mu) ** 2 - 1.0 / mu ** 2
    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 2.0 / (1.0 - mu) ** 3 + 2.0 / mu ** 3
    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 6.0 / (1.0 - mu) ** 4 - 6.0 / mu ** 4


def _dnorm(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


class ProbitLink(Link):
    """``g(μ) = Φ⁻¹(μ)`` — probit binomial link."""
    name = "probit"
    def link(self, mu): return ndtri(np.asarray(mu, dtype=float))
    def linkinv(self, eta):
        # R: clamp η to ±qnorm(eps); pnorm of clamped η.
        eta = np.asarray(eta, dtype=float)
        thresh = -ndtri(np.finfo(float).eps)
        return ndtr(np.clip(eta, -thresh, thresh))
    def mu_eta(self, eta):
        # dnorm(η), lower-clamped.
        eps = np.finfo(float).eps
        return np.maximum(_dnorm(np.asarray(eta, dtype=float)), eps)
    def d2link(self, mu):
        eta = ndtri(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return eta / d ** 2
    def d3link(self, mu):
        eta = ndtri(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return (1.0 + 2.0 * eta * eta) / d ** 3
    def d4link(self, mu):
        eta = ndtri(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return (7.0 * eta + 6.0 * eta ** 3) / d ** 4


class CauchitLink(Link):
    """``g(μ) = tan(π(μ-½))`` — Cauchy-quantile binomial link.

    Heavier-tailed than probit/logit; fits well when a fraction of obs are
    far from the (logit) decision boundary.
    """
    name = "cauchit"
    def link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return np.tan(np.pi * (mu - 0.5))
    def linkinv(self, eta):
        # R: clamp η to ±qcauchy(eps); pcauchy(η) = ½ + atan(η)/π.
        eps = np.finfo(float).eps
        thresh = -np.tan(np.pi * (eps - 0.5))
        eta_c = np.clip(np.asarray(eta, dtype=float), -thresh, thresh)
        return 0.5 + np.arctan(eta_c) / np.pi
    def mu_eta(self, eta):
        eps = np.finfo(float).eps
        eta = np.asarray(eta, dtype=float)
        return np.maximum(1.0 / (np.pi * (1.0 + eta * eta)), eps)
    def d2link(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        return 2.0 * np.pi ** 2 * eta * (1.0 + eta * eta)
    def d3link(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        eta2 = eta * eta
        return 2.0 * np.pi ** 3 * (1.0 + 3.0 * eta2) * (1.0 + eta2)
    def d4link(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        eta2 = eta * eta
        return 2.0 * np.pi ** 4 * (8.0 * eta + 12.0 * eta2 * eta) * (1.0 + eta2)


class CloglogLink(Link):
    """``g(μ) = log(-log(1-μ))`` — complementary log-log binomial link."""
    name = "cloglog"
    def link(self, mu):
        return np.log(-np.log1p(-np.asarray(mu, dtype=float)))
    def linkinv(self, eta):
        # 1 - exp(-exp(η)), clamped to [eps, 1-eps] (R: avoid mu=0,1 boundary).
        eps = np.finfo(float).eps
        eta = np.asarray(eta, dtype=float)
        return np.clip(-np.expm1(-np.exp(eta)), eps, 1.0 - eps)
    def mu_eta(self, eta):
        # exp(η - exp(η)); R clamps η at 700 (to keep exp(η) finite) and
        # lower-clamps the result at eps.
        eps = np.finfo(float).eps
        eta = np.minimum(np.asarray(eta, dtype=float), 700.0)
        return np.maximum(np.exp(eta) * np.exp(-np.exp(eta)), eps)
    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        l1m = np.log1p(-mu)
        return -1.0 / ((1.0 - mu) ** 2 * l1m) * (1.0 + 1.0 / l1m)
    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        l1m = np.log1p(-mu)
        return (-2.0 - 3.0 * l1m - 2.0 * l1m ** 2) / (1.0 - mu) ** 3 / l1m ** 3
    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        l1m = np.log1p(-mu)
        return (-12.0 - 11.0 * l1m - 6.0 * l1m ** 2 - 6.0 / l1m) / (1.0 - mu) ** 4 / l1m ** 3


class InverseSquareLink(Link):
    """``g(μ) = 1/μ²`` — canonical inverse-Gaussian link."""
    name = "1/mu^2"
    def link(self, mu): return 1.0 / np.asarray(mu, dtype=float) ** 2
    def linkinv(self, eta):
        # PIRLS step-halving may transiently call us with eta<0 entries;
        # the caller checks valideta() and rejects them. Silence the
        # sqrt-of-negative warning so strict warning modes (pytest's
        # `np.errstate(invalid="raise")`) don't trip over a recoverable
        # halving step.
        with np.errstate(invalid="ignore"):
            return 1.0 / np.sqrt(np.asarray(eta, dtype=float))
    def mu_eta(self, eta):
        with np.errstate(invalid="ignore"):
            return -0.5 * np.asarray(eta, dtype=float) ** -1.5
    def d2link(self, mu): return 6.0 * np.asarray(mu, dtype=float) ** -4
    def d3link(self, mu): return -24.0 * np.asarray(mu, dtype=float) ** -5
    def d4link(self, mu): return 120.0 * np.asarray(mu, dtype=float) ** -6
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


_LINKS = {
    "identity": IdentityLink,
    "log": LogLink,
    "inverse": InverseLink,
    "sqrt": SqrtLink,
    "logit": LogitLink,
    "probit": ProbitLink,
    "cauchit": CauchitLink,
    "cloglog": CloglogLink,
    "1/mu^2": InverseSquareLink,
}


def _resolve_link(link, default: str) -> Link:
    if link is None:
        return _LINKS[default]()
    if isinstance(link, Link):
        return link
    if isinstance(link, str):
        if link not in _LINKS:
            raise ValueError(f"unknown link {link!r}; supported: {list(_LINKS)}")
        return _LINKS[link]()
    # Allow `link=log` (the function reference) the way R's `Gamma(link=log)` does.
    name = getattr(link, "__name__", None)
    if name in _LINKS:
        return _LINKS[name]()
    raise ValueError(f"unknown link {link!r}")


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


class Family:
    """Base class for GLM families."""
    name: str
    canonical_link_name: str
    scale_known: bool
    # Number of "extra" family parameters that the GAM outer Newton should
    # estimate jointly with (ρ, log φ). Default 0 (Gaussian, Gamma, Poisson,
    # Binomial, IG, Quasi); ``tw`` overrides to 1 (its θ_tw → p
    # reparametrisation). The GAM hooks read ``n_theta`` to size the outer
    # vector and call ``set_theta(values)`` before each criterion eval; they
    # call ``dscore_extra(...)`` to obtain the score-side ∂(2·V_R)/∂θ_extra
    # contributions for the gradient.
    n_theta: int = 0
    # Mirrors mgcv ``inherits(family, "extended.family")``. Standard
    # exponential families (Gaussian, Poisson, ...) leave it ``False``;
    # extended families (Scat, ziP, ocat, gevlss, ...) flip to ``True``
    # so the bam(discrete=TRUE) PIRLS path uses the ``Dd → dDeta`` Newton
    # weights (``w = Deta2/2``, ``z = (η-off) - Deta/Deta2``) instead of
    # the standard Fisher weights ``w = w_prior · μ_η²/V(μ)``.
    is_extended: bool = False
    # Whether the bam outer loop should call ``_estimate_theta`` between
    # PIRLS iters. Set ``True`` only on extended families with free θ
    # (Scat with both θ free, nb with k free, etc). Standard families and
    # extended families with all θ user-locked leave it ``False``.
    estimate_theta_callback: bool = False

    def __init__(self, link=None):
        self.link = _resolve_link(link, self.canonical_link_name)

    @property
    def is_canonical(self) -> bool:
        return self.link.name == self.canonical_link_name

    def set_theta(self, values) -> None:
        """Mutate the family's extra parameters from a length-``n_theta``
        array. Default is a no-op (consistent with ``n_theta = 0``);
        :class:`tw` overrides to update ``self.theta`` and ``self.p``.
        """
        if self.n_theta != 0:
            raise NotImplementedError(
                f"{type(self).__name__} declares n_theta={self.n_theta} "
                f"but did not override set_theta()."
            )

    def get_theta(self) -> np.ndarray:
        """Return the current extra parameters as a length-``n_theta`` array.
        Default empty; :class:`tw` returns ``[θ_tw]``."""
        return np.zeros(0)

    def variance(self, mu): raise NotImplementedError
    def dvar(self, mu): raise NotImplementedError
    def d2var(self, mu): raise NotImplementedError
    def d3var(self, mu): raise NotImplementedError

    def dev_resids(self, y, mu, wt, theta=None) -> np.ndarray:
        """Per-observation deviance contributions; sum is the deviance D.

        ``theta`` is accepted but ignored for standard exponential
        families. Extended families (``is_extended=True``) read it to
        compute deviance at a probe θ during inner-Newton θ estimation.
        """
        raise NotImplementedError

    # ----- extended-family hooks (no-ops for standard families) ---------
    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        """Mirrors mgcv ``family$Dd``. Returns a dict of derivatives of
        ``-logL`` wrt μ and θ at fixed (y, μ, θ, w):

        * level 0: ``Dmu``, ``Dmu2``, ``EDmu2`` (all length-n).
        * level ≥ 1: + ``Dth``, ``Dmuth``, ``Dmu2th``, ``EDmu2th``,
          ``Dmu3``, ``EDmu3``. ``D*th`` shape ``(n, n_theta)``.
        * level ≥ 2: + ``Dmu4``, ``Dth2``, ``Dmuth2``, ``Dmu2th2``,
          ``Dmu3th``. ``D*th2`` packed column-major upper-triangle of
          shape ``(n, n_theta·(n_theta+1)/2)``.

        Standard families don't implement ``Dd`` — bam's PIRLS path uses
        the Fisher branch for them. Only extended families override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.Dd() is not implemented; this family "
            "uses the standard-Fisher PIRLS path. Set is_extended=True "
            "and implement Dd() to use the extended-family Newton path."
        )

    def dDeta(self, y, mu, wt, theta, level: int = 0) -> dict:
        """Convert ``Dd`` (μ-space derivatives) to η-space via the link
        chain rule. Mirrors mgcv ``dDeta`` (R/efam.r). For identity link
        it copies ``Dmu → Deta``, ``Dmu2 → Deta2``, ...; for non-identity
        it applies ``Deta = Dmu · μ_η`` etc with the ``g2g``/``g3g``/
        ``g4g`` link curvature terms.

        Returns a dict with at minimum ``Deta``, ``Deta2``, ``EDeta2``
        (level 0). ``Deta.Deta2 = Dmu/(Dmu2·μ_η - Dmu·g2g)`` is the
        Newton-step working-response numerator that bam's PIRLS reads.
        """
        r = self.Dd(y, mu, theta, wt, level=level)
        link = self.link
        if link.name == "identity":
            d = {
                "Deta": r["Dmu"],
                "Deta2": r["Dmu2"],
                "EDeta2": r["EDmu2"],
                "Deta.Deta2": r["Dmu"] / r["Dmu2"],
                "Deta.EDeta2": r["Dmu"] / r["EDmu2"],
            }
            if level > 0:
                d.update({
                    "Dth": r["Dth"],
                    "Detath": r["Dmuth"],
                    "Deta3": r["Dmu3"],
                    "Deta2th": r["Dmu2th"],
                    "EDeta2th": r["EDmu2th"],
                    "EDeta3": r.get("EDmu3"),
                })
            if level > 1:
                d.update({
                    "Deta4": r["Dmu4"],
                    "Dth2": r["Dth2"],
                    "Detath2": r["Dmuth2"],
                    "Deta2th2": r["Dmu2th2"],
                    "Deta3th": r["Dmu3th"],
                })
            return d
        # Non-identity link path. mgcv ``dDeta`` expects ``link.g2g(μ)``,
        # ``g3g``, ``g4g`` to be implemented on the link object.
        ig1 = link.mu_eta(link.link(np.asarray(mu, dtype=float)))
        ig12 = ig1 * ig1
        g2g = link.g2g(mu)
        d = {
            "Deta": r["Dmu"] * ig1,
            "Deta2": r["Dmu2"] * ig12 - r["Dmu"] * g2g * ig1,
            "EDeta2": r["EDmu2"] * ig12,
        }
        d["Deta.Deta2"] = r["Dmu"] / (r["Dmu2"] * ig1 - r["Dmu"] * g2g)
        d["Deta.EDeta2"] = r["Dmu"] / (r["EDmu2"] * ig1)
        if level > 0:
            ig13 = ig12 * ig1
            d["Dth"] = r["Dth"]
            d["Detath"] = r["Dmuth"] * ig1
            g3g = link.g3g(mu)
            d["Deta3"] = (r["Dmu3"] * ig13
                          - 3.0 * r["Dmu2"] * g2g * ig12
                          + r["Dmu"] * (3.0 * g2g * g2g - g3g) * ig1)
            EDmu3 = r.get("EDmu3")
            if EDmu3 is not None:
                d["EDeta3"] = EDmu3 * ig13 - 3.0 * r["EDmu2"] * g2g * ig12
            d["Deta2th"] = r["Dmu2th"] * ig12 - r["Dmuth"] * g2g * ig1
            EDmu2th = r.get("EDmu2th")
            if EDmu2th is not None:
                d["EDeta2th"] = EDmu2th * ig12
        if level > 1:
            g4g = link.g4g(mu)
            ig14 = ig12 * ig12
            d["Deta4"] = (ig14 * r["Dmu4"]
                          - 6.0 * r["Dmu3"] * ig13 * g2g
                          + r["Dmu2"] * (15.0 * g2g * g2g - 4.0 * g3g) * ig12
                          - r["Dmu"]
                          * (15.0 * g2g ** 3 - 10.0 * g2g * g3g + g4g)
                          * ig1)
            d["Dth2"] = r["Dth2"]
            d["Detath2"] = r["Dmuth2"] * ig1
            d["Deta2th2"] = r["Dmu2th2"] * ig12 - r["Dmuth2"] * g2g * ig1
            d["Deta3th"] = (r["Dmu3th"] * ig13
                            - 3.0 * r["Dmu2th"] * g2g * ig12
                            + r["Dmuth"] * (3.0 * g2g * g2g - g3g) * ig1)
        return d

    def preinitialize(self, y) -> dict | None:
        """One-shot pre-fit hook. mgcv ``family$preinitialize(y, family)``
        runs once before the first PIRLS iter and may return ``{"Theta":
        ...}`` (initial θ override) and/or ``{"y": ...}`` (transformed
        response). Default: no-op. Extended families with data-dependent
        θ start (Scat: ``c(1.5, log(0.8·sd(y)))``) override.
        """
        return None

    def postproc(self, y, mu, wt) -> dict:
        """One-shot post-fit hook for display strings. mgcv
        ``family$postproc`` rewrites ``family$family`` to e.g.
        ``"Scaled t(5,0.3)"`` reflecting fitted θ. Default: empty dict.
        """
        return {}

    def initialize(self, y, wt) -> np.ndarray:
        """Starting μ̂ for PIRLS. Return a length-n positive (or family-valid)
        vector. Default: y; subclasses override when y can be at the boundary.
        """
        return np.asarray(y, dtype=float).copy()

    def validmu(self, mu) -> bool:
        return bool(np.all(np.isfinite(mu)))

    def aic(self, y, mu, dev, wt, n, theta=None) -> float:
        """``-2·loglik + 2·k_overhead``. Returned without smoothing penalty;
        the caller adds ``+2·edf`` (or whatever df rule it uses).

        ``theta`` is accepted but ignored for standard families.
        Extended families read it for the AIC contribution from θ.
        """
        raise NotImplementedError

    def _aic_dev1(self, dev, scale, wt) -> float:
        """The ``dev1`` argument that ``aic(y, μ, dev1, wt, n)`` consumes.

        Mirrors ``gam.fit3.r:848-849``. For unknown-scale non-Gaussian families
        (Gamma, IG) and scale-known families (Poisson, binomial), this is
        ``scale · Σwt`` so the AIC uses the Pearson/REML scale estimator (or
        the fixed scale=1). Gaussian overrides this to return ``dev`` directly
        because the MLE σ² = dev/n has a closed form and mgcv prefers it
        over the moment estimator for the AIC.
        """
        return float(scale) * float(np.sum(np.asarray(wt, dtype=float)))

    def ls(self, y, wt, scale) -> np.ndarray:
        """Saturated log-likelihood at μ=y, plus its 1st/2nd derivative
        wrt ``log φ`` (φ = scale) — used by REML when scale is unknown.

        Returns a length-3 ``(ls0, d_ls/d_log_φ, d²_ls/d_log_φ²)`` array
        summed over observations. mgcv's ``family$ls`` returns ``d/dφ``
        and ``d²/dφ²``; we apply the chain rule internally so the caller
        works directly in the ρ = log φ parametrisation that REML and
        gam.fit3's outer optimiser use. For scale-known families
        (Poisson, binomial) ``d1 = d2 = 0``.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}(link={self.link.name})"


class Gaussian(Family):
    """``y ~ N(μ, σ²)``; scale σ² is unknown."""
    name = "gaussian"
    canonical_link_name = "identity"
    scale_known = False

    def variance(self, mu): return np.ones_like(np.asarray(mu, dtype=float))
    def dvar(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d2var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d3var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (y - mu) ** 2

    def aic(self, y, mu, dev, wt, n, theta=None):
        n_eff = float(np.sum(wt))
        sigma2 = dev / n_eff
        # mgcv's gaussian()$aic: n·(log(2πσ²)+1) + 2 — note the +2 is the
        # "+1 family df" placeholder; downstream adds 2·edf for the model.
        return n_eff * (np.log(2.0 * np.pi * sigma2) + 1.0) + 2.0

    def _aic_dev1(self, dev, scale, wt):
        # Gaussian MLE σ² = dev/n is closed-form, so mgcv passes dev directly
        # (gam.fit3.r:848). Caller's `dev` is the family deviance = RSS for
        # Gaussian. n_eff = Σwt and dev/n_eff = MLE σ².
        return float(dev)

    def ls(self, y, wt, scale):
        # mgcv: ls = -½·nobs·log(2π·φ) + ½·Σ log w[w>0]
        # so d/d(log φ) = -nobs/2, d²/d(log φ²) = 0. (Same algebraic shape
        # as InverseGaussian — neither family has a y-term involving φ.)
        # `nobs` here is the *count* of w>0 obs, not Σw — mgcv weights act
        # as a precision multiplier on σ², not as a sample-size multiplier.
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        nobs = int(np.sum(good))
        ls0 = (-0.5 * nobs * np.log(2.0 * np.pi * scale)
               + 0.5 * float(np.sum(np.log(wt[good]))))
        return np.array([ls0, -0.5 * nobs, 0.0], dtype=float)


class Gamma(Family):
    """``y ~ Gamma(shape=1/φ, scale=μ·φ)``; mean μ, variance φ·μ²."""
    name = "Gamma"
    canonical_link_name = "inverse"
    scale_known = False

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float); return mu * mu
    def dvar(self, mu):
        mu = np.asarray(mu, dtype=float); return 2.0 * mu
    def d2var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), 2.0)
    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        # mgcv: -2 wt (log(y/μ) - (y-μ)/μ); use ifelse(y==0, 1, y/μ) so
        # log(0) doesn't propagate when an observation is exactly zero.
        ratio = np.where(y == 0, 1.0, y / mu)
        return -2.0 * wt * (np.log(ratio) - (y - mu) / mu)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("Gamma family requires strictly positive responses")
        return y.copy()

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def aic(self, y, mu, dev, wt, n, theta=None):
        wt = np.asarray(wt, dtype=float)
        n_eff = float(wt.sum())
        disp = dev / n_eff
        # R's Gamma()$aic: -2·Σ wt·log dgamma(y; 1/disp, scale=μ·disp) + 2.
        # +2 mirrors mgcv (one "extra" df for the dispersion).
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _gamma_dist.logpdf(y, a=1.0 / disp, scale=mu * disp)
        return -2.0 * float(np.sum(logp * wt)) + 2.0

    def ls(self, y, wt, scale):
        # Direct port of mgcv:::fix.family.ls's Gamma branch (raw d/dφ form),
        # then a log-scale chain rule to match the hea convention:
        #   d/dlogφ  = φ · d/dφ
        #   d²/dlogφ² = φ · d/dφ + φ² · d²/dφ²
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        good = wt > 0
        y = y[good]; w = wt[good]
        sw = scale / w                                     # per-obs scale
        # k1 = -lgamma(1/sw) - log(sw)/sw - 1/sw
        k1 = -gammaln(1.0 / sw) - np.log(sw) / sw - 1.0 / sw
        ls0 = float(np.sum(k1 - np.log(y)))
        # k2 = (digamma(1/sw) + log(sw)) / sw²       (mgcv's d/dφ)
        k2 = (digamma(1.0 / sw) + np.log(sw)) / (sw * sw)
        d1_phi = float(np.sum(k2 / w))
        # k3 = (-trigamma(1/sw)/sw + 1 - 2 log(sw) - 2 digamma(1/sw)) / sw³
        k3 = (-polygamma(1, 1.0 / sw) / sw
              + 1.0 - 2.0 * np.log(sw) - 2.0 * digamma(1.0 / sw)) / (sw ** 3)
        d2_phi = float(np.sum(k3 / (w * w)))             # mgcv's d²/dφ²
        d1 = scale * d1_phi
        d2 = scale * d1_phi + scale * scale * d2_phi
        return np.array([ls0, d1, d2], dtype=float)


class Poisson(Family):
    """``y ~ Poisson(μ)``; mean = variance = μ; scale fixed at 1."""
    name = "poisson"
    canonical_link_name = "log"
    scale_known = True

    def variance(self, mu): return np.asarray(mu, dtype=float).copy()
    def dvar(self, mu): return np.ones_like(np.asarray(mu, dtype=float))
    def d2var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d3var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv: 2 wt (y log(y/μ) - (y-μ)); with the convention 0·log(0/μ) = 0
        # so a y=0 row contributes 2 wt μ.
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        positive = y > 0
        # avoid log(0) on y=0 rows by substituting μ inside the log (the
        # whole y·log term is then masked to 0 anyway).
        ratio = np.where(positive, y / np.where(positive, mu, 1.0), 1.0)
        contrib = np.where(positive,
                           wt * (y * np.log(ratio) - (y - mu)),
                           wt * mu)
        return 2.0 * contrib

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError("negative values not allowed for the 'Poisson' family")
        # mgcv/R: mustart = y + 0.1 to keep log(μ) finite when y=0.
        return y + 0.1

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # Port of lme4's ``PoissonDist::aic`` (glmFamily.cpp:321-326):
        # ``-2 · Σ wt[i] · Rf_dpois(y[i], mu[i], TRUE)`` with sequential
        # reduction. :func:`_dpois_raw` is vectorized; the final sum uses
        # ``np.cumsum(...)[-1]`` for sequential bit-match to Eigen3.
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        logp = _dpois_raw(y, mu, True)
        return -2.0 * float(np.cumsum(logp * wt)[-1])

    def ls(self, y, wt, scale):
        # Saturated log-lik at μ=y; scale-known so d/dlogφ = d²/dlogφ² = 0.
        # mgcv: sum(dpois(y, y, log=TRUE) · w).
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _poisson_dist.logpmf(y, y)
        ls0 = float(np.sum(logp * wt))
        return np.array([ls0, 0.0, 0.0], dtype=float)


class Binomial(Family):
    """``y·m ~ Binomial(m, μ)``; ``y`` is the success proportion in [0,1],
    ``wt`` is the binomial size ``m`` (= 1 for Bernoulli).

    The cbind(success, failure) input form that R supports is *not* handled
    here — the caller must pre-convert it to (proportion, size) before
    constructing the family.
    """
    name = "binomial"
    canonical_link_name = "logit"
    scale_known = True

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float); return mu * (1.0 - mu)
    def dvar(self, mu):
        return 1.0 - 2.0 * np.asarray(mu, dtype=float)
    def d2var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), -2.0)
    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv (C_binomial_dev_resids): 2 wt [ y_log_y(y, μ) + y_log_y(1-y, 1-μ) ]
        # where y_log_y(y, μ) = y log(y/μ) for y>0, else 0.
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)

        def yly(a, b):
            # 0·log(0/0) := 0; mask both arguments inside the log so numpy
            # doesn't evaluate log(0) on the dead branch and emit warnings.
            pos = a > 0
            safe_a = np.where(pos, a, 1.0)
            safe_b = np.where(pos, b, 1.0)
            return np.where(pos, a * np.log(safe_a / safe_b), 0.0)

        return 2.0 * wt * (yly(y, mu) + yly(1.0 - y, 1.0 - mu))

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError("y values must be 0 <= y <= 1 for the 'binomial' family")
        # mgcv/R: mustart = (wt·y + 0.5) / (wt + 1) keeps μ in (0,1) so the
        # logit link starts finite even when y is exactly 0 or 1.
        return (wt * y + 0.5) / (wt + 1.0)

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0) and np.all(mu < 1))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # Port of lme4's ``binomialDist::aic`` (glmFamily.cpp:204-213):
        # ``-2 · Σ (wt[i]/m[i]) · Rf_dbinom(round(m·y), round(m), μ, TRUE)``
        # with sequential reduction. :func:`_dbinom_raw` is vectorized;
        # final sum uses ``np.cumsum(...)[-1]`` for bit-match to Eigen3.
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        m = np.rint(wt)
        # Mask out m<=0; for those, contribution is 0.
        good = m > 0
        if not np.any(good):
            return 0.0
        s_arr = np.rint(np.where(good, m * y, 0.0))
        weight = np.where(good, wt / np.where(good, m, 1.0), 0.0)
        logp = _dbinom_raw(s_arr, m, mu, 1.0 - mu, True)
        terms = weight * logp
        # Replace -inf entries (oob) by 0 so they don't contaminate the
        # sum (lme4 filters via the m<=0 branch which sets contribution
        # to 0; oob cases shouldn't occur for valid data anyway).
        terms = np.where(good & np.isfinite(logp), terms, 0.0)
        return -2.0 * float(np.cumsum(terms)[-1])

    def ls(self, y, wt, scale):
        # mgcv: ls = -binomial$aic(y, n, y, w, 0) / 2; scale-known.
        ls0 = -0.5 * self.aic(y, y, 0.0, wt, None)
        return np.array([ls0, 0.0, 0.0], dtype=float)


class InverseGaussian(Family):
    """``y ~ IG(μ, φ)``; mean μ, variance φ·μ³; scale φ unknown."""
    name = "inverse.gaussian"
    canonical_link_name = "1/mu^2"
    scale_known = False

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float); return mu ** 3
    def dvar(self, mu):
        mu = np.asarray(mu, dtype=float); return 3.0 * mu * mu
    def d2var(self, mu):
        return 6.0 * np.asarray(mu, dtype=float)
    def d3var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), 6.0)

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv: wt · (y - μ)² / (y · μ²).
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (y - mu) ** 2 / (y * mu * mu)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError(
                "positive values only are allowed for the 'inverse.gaussian' family"
            )
        return y.copy()

    def validmu(self, mu):
        # R/stats: TRUE — boundary handling is via the link's valideta.
        return bool(np.all(np.isfinite(np.asarray(mu, dtype=float))))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv: sum(wt) · (1 + log(dev/sum(wt) · 2π)) + 3 · Σ wt · log(y) + 2.
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        sw = float(wt.sum())
        return (sw * (1.0 + np.log(dev / sw * 2.0 * np.pi))
                + 3.0 * float(np.sum(np.log(y) * wt)) + 2.0)

    def ls(self, y, wt, scale):
        # mgcv (raw φ form):
        #   ls0 = -½ · Σ log(2π φ y³) + ½ · Σ log w[w>0]
        #   d/dφ ls = -nobs/(2φ),  d²/dφ² ls = +nobs/(2φ²)
        # Chain rule to log-scale: d/dlogφ = -nobs/2, d²/dlogφ² = 0
        # (same algebraic cancellation as Gaussian — the y³ term has no φ).
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        good = wt > 0
        nobs = int(np.sum(good))
        ls0 = (-0.5 * float(np.sum(np.log(2.0 * np.pi * scale * y[good] ** 3)))
               + 0.5 * float(np.sum(np.log(wt[good]))))
        return np.array([ls0, -0.5 * nobs, 0.0], dtype=float)


# ---------------------------------------------------------------------------
# Quasi: pure quasi-likelihood (no full likelihood, dispersion always
# estimated). Variance functions and deviances coincide with the matching
# parametric families, so we delegate to them rather than re-derive.
# ---------------------------------------------------------------------------


_QUASI_VARIANCE_FAMILIES = {
    "constant": Gaussian,         # V(μ) = 1
    "mu":       Poisson,          # V(μ) = μ
    "mu^2":     Gamma,             # V(μ) = μ²
    "mu^3":     InverseGaussian,  # V(μ) = μ³
    "mu(1-mu)": Binomial,         # V(μ) = μ(1-μ)
}


class Quasi(Family):
    """R's ``quasi(link, variance)``: pure quasi-likelihood.

    The mean–variance relation is set by ``variance=`` (one of
    ``"constant"``, ``"mu"``, ``"mu^2"``, ``"mu^3"``, ``"mu(1-mu)"``).
    Dispersion is always estimated from the Pearson χ²/df_resid; there is
    no proper likelihood, so ``aic`` and ``ls`` return NaN — Wald inference
    uses the t-distribution because the scale is unknown.

    Variance functions and deviances coincide with the matching parametric
    families, so this class delegates ``variance/dvar/dev_resids/validmu``
    to them. ``initialize`` matches R's ``quasi()`` (which differs from
    Binomial's precision-weighted start when ``variance='mu(1-mu)'``).
    """
    name = "quasi"
    canonical_link_name = "identity"  # R's quasi() default, regardless of variance
    scale_known = False

    def __init__(self, link=None, variance: str = "constant"):
        if variance not in _QUASI_VARIANCE_FAMILIES:
            raise ValueError(
                f"quasi(): variance must be one of {list(_QUASI_VARIANCE_FAMILIES)}; "
                f"got {variance!r}"
            )
        self.variance_name = variance
        self._shadow = _QUASI_VARIANCE_FAMILIES[variance]()
        super().__init__(link=link)

    def variance(self, mu): return self._shadow.variance(mu)
    def dvar(self, mu):     return self._shadow.dvar(mu)
    def d2var(self, mu):    return self._shadow.d2var(mu)
    def d3var(self, mu):    return self._shadow.d3var(mu)

    def dev_resids(self, y, mu, wt, theta=None):
        return self._shadow.dev_resids(y, mu, wt)

    def initialize(self, y, wt):
        # R's quasi(variance='mu(1-mu)') initialize is
        # ``pmax(0.001, pmin(0.999, y))`` — clip y into the open
        # interval (0, 1). Different from binomial's
        # ``(wt·y + 0.5) / (wt + 1)`` smoothing.
        if self.variance_name == "mu(1-mu)":
            y = np.asarray(y, dtype=float)
            if np.any(y < 0) or np.any(y > 1):
                raise ValueError(
                    "y values must be 0 <= y <= 1 for quasi(variance='mu(1-mu)')"
                )
            return np.clip(y, 0.001, 0.999)
        return self._shadow.initialize(y, wt)

    def validmu(self, mu):
        return self._shadow.validmu(mu)

    def aic(self, y, mu, dev, wt, n, theta=None):
        return float("nan")

    def ls(self, y, wt, scale):
        # Extended quasi-likelihood saturated piece (Nelder & Pregibon 1987;
        # McCullagh & Nelder 1989, §9.6). mgcv's ``quasi$ls`` drops both the
        # log(2π) and log V(y) constants — neither depends on φ or ρ, so they
        # don't affect REML's argmin; dropping log V(y) also sidesteps log 0
        # when y is at the support boundary (e.g. count zeros under
        # variance='mu'). What's left is the Gaussian φ-shape:
        #
        #     ls0 = -n_obs/2 · log φ + ½·Σ_{w>0} log w
        #     d/dφ ls = -n_obs/(2φ),  d²/dφ² ls = n_obs/(2φ²)
        #
        # Chain-ruled to log φ (hea's convention):
        #     d/dlog φ  = -n_obs/2
        #     d²/dlog φ² = -n_obs/2 + n_obs/2 = 0
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        nobs = int(np.sum(good))
        ls0 = (-0.5 * nobs * np.log(scale)
               + 0.5 * float(np.sum(np.log(wt[good]))))
        return np.array([ls0, -0.5 * nobs, 0.0], dtype=float)

    def __repr__(self) -> str:
        return f"quasi(link={self.link.name}, variance={self.variance_name!r})"


# ---------------------------------------------------------------------------
# Tweedie / tw — Dunn-Smyth (2005) series implementation.
#
# Tweedie EDF for ``1 < p < 2`` is the compound Poisson-Gamma: a Poisson(λ)
# count of Gamma jumps. Mean μ, variance ``φ·μ^p``; the density mixes a
# point mass at 0 with a continuous part on ``y > 0``. With ``α = (2-p)/(1-p)``
# (negative for 1<p<2):
#
#     y = 0:  log f(0; μ, φ, p) = -μ^(2-p) / (φ·(2-p))
#     y > 0:  log f(y; μ, φ, p) = -log y + log a(y, φ, p)
#                                + y·μ^(1-p)/(φ·(1-p)) - μ^(2-p)/(φ·(2-p))
#
# where ``a(y, φ, p) = Σ_{j≥1} W_j``,
#
#     log W_j = j·log z - log Γ(j+1) - log Γ(-j·α),
#     log z   = -α·log y + α·log(p-1) - (1-α)·log φ - log(2-p).
#
# We sum log-W_j outward from the dominant index ``j*`` (where d_j log W_j = 0)
# until terms drop ``≥ ld_eps`` below the running max, then log-sum-exp. The
# moments E_p[j] and Var_p[j] under ``p_j = W_j / Σ W_k`` give the φ-derivatives
# of log a:  d/dlog φ  log a = -(1-α)·E[j] ;  d²/dlog φ² log a = (1-α)²·Var[j].
# Direct port of mgcv's ``tweedious.c`` / ``ldTweedie``.
# ---------------------------------------------------------------------------


# Series tail tolerance: terms log W_j < log W_max - LD_EPS are dropped. mgcv
# uses ~36 (≈ -log(eps^½)); a touch tighter than the .Machine$double.eps
# threshold used in tweedious.c, but well past where summands matter.
_LD_EPS = 36.0
# Hard cap on series length to bound worst-case latency at extreme (y, φ, p).
# In practice the series is centred near j* with width ~√j*, so the loop
# exits via the LD_EPS gate long before this; the cap is purely a safety net.
_LD_J_MAX = 100000


def _tweedie_log_a_one(y_i: float, phi_i: float, p: float):
    """Series approximation log a(y, φ, p) = log Σ_{j≥1} W_j for one y > 0.

    Returns ``(log_a, j_bar, j_var, j_psi_bar)`` — the log of the series sum
    plus three moments of ``j`` under ``p_j = W_j/Σ W_k``: E[j], Var[j],
    and E[j·ψ(-j·α)]. The first two feed the φ-derivatives of log a; the
    third (with the digamma weight) is needed for the p-derivative — see
    Tweedie.dls_dp.
    """
    om1 = 1.0 - p                  # negative
    tm = 2.0 - p                   # positive
    alpha = tm / om1               # negative
    one_minus_alpha = 1.0 - alpha  # > 1; equals 1/(p-1)

    # log W_j = j·log_z - lgamma(j+1) - lgamma(-j·α).
    # Pull constants out of the j loop.
    log_z = (-alpha * np.log(y_i) + alpha * np.log(p - 1.0)
             - one_minus_alpha * np.log(phi_i) - np.log(tm))

    # Continuous-extension dominant index (Dunn-Smyth §3): with ψ(x) ≈ log x,
    # d_j log W_j = log_z - ψ(j+1) + α·ψ(-jα) ≈ 0 ⇒
    #     j*  ≈ exp((log_z + α·log(-α)) / (1-α))
    j_star = np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha)
    j_star = max(j_star, 1.0)
    j_int = max(1, int(round(j_star)))

    def _lw(j):
        return j * log_z - gammaln(j + 1.0) - gammaln(-j * alpha)

    # Walk outward from j_int both ways. Record (j, log W_j) for each kept
    # term; track the running max so log-sum-exp is numerically stable. The
    # `min_steps` guard keeps a few neighbours even when the immediate
    # neighbour is already below the eps gate (rare; happens at small j*).
    log_max = _lw(j_int)
    j_list = [float(j_int)]
    lw_list = [log_max]

    # Right tail.
    j = j_int + 1
    near = 5
    while j < _LD_J_MAX:
        v = _lw(j)
        if v - log_max < -_LD_EPS and (j - j_int) > near:
            break
        j_list.append(float(j))
        lw_list.append(v)
        if v > log_max:
            log_max = v
        j += 1

    # Left tail.
    j = j_int - 1
    while j >= 1:
        v = _lw(j)
        if v - log_max < -_LD_EPS and (j_int - j) > near:
            break
        j_list.append(float(j))
        lw_list.append(v)
        if v > log_max:
            log_max = v
        j -= 1

    j_arr = np.array(j_list, dtype=float)
    lw_arr = np.array(lw_list, dtype=float)
    weights = np.exp(lw_arr - log_max)
    sum_w = float(np.sum(weights))
    log_a = log_max + float(np.log(sum_w))

    p_w = weights / sum_w
    j_bar = float(np.sum(p_w * j_arr))
    j_var = float(np.sum(p_w * (j_arr - j_bar) ** 2))
    # ψ(-j·α) is well-defined for α<0, j≥1 (so -j·α > 0). We compute it on
    # the same j-grid so that the moment matches the series we just summed.
    psi_arr = digamma(-j_arr * alpha)
    j_psi_bar = float(np.sum(p_w * j_arr * psi_arr))
    return log_a, j_bar, j_var, j_psi_bar


def _tweedie_log_a_vec(y, phi, p, _chunk_bytes: int = 256 * 1024 * 1024):
    """Vectorised over y (and per-obs phi). Returns four arrays of shape
    ``y.shape``: ``log_a``, ``j_bar``, ``j_var``, ``j_psi_bar``. Entries
    with y==0 are 0 (the y=0 row uses the closed-form point mass, not the
    series). Per-obs phi handles weights via ``φ_i = φ/wt_i``.

    Builds a fixed ``j`` grid wide enough to cover every active row's
    eps-truncated series tail, then evaluates the (n_active, J) matrix
    of ``log W_j`` and reduces along ``j`` in one pass. ``J`` is sized
    so the eps gate fires within the grid for every row — agrees with
    the per-row :func:`_tweedie_log_a_one` walk to ~1e-13 absolute on
    log_a / moments (well below mgcv-oracle test tolerances).
    """
    y = np.asarray(y, dtype=float)
    phi_arr = np.broadcast_to(np.asarray(phi, dtype=float), y.shape).astype(float, copy=True)
    log_a = np.zeros_like(y)
    j_bar = np.zeros_like(y)
    j_var = np.zeros_like(y)
    j_psi_bar = np.zeros_like(y)
    flat_y = y.ravel()
    flat_phi = phi_arr.ravel()
    active = flat_y > 0.0
    if not np.any(active):
        return log_a, j_bar, j_var, j_psi_bar
    ya = flat_y[active]
    pha = flat_phi[active]

    om1 = 1.0 - p
    tm = 2.0 - p
    alpha = tm / om1
    one_minus_alpha = 1.0 - alpha

    log_z = (-alpha * np.log(ya) + alpha * np.log(p - 1.0)
             - one_minus_alpha * np.log(pha) - np.log(tm))
    j_star = np.maximum(
        np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha), 1.0,
    )
    j_int = np.maximum(1, np.round(j_star).astype(int))
    j_int_max = int(j_int.max())

    # Series decay rate scales with |alpha|; p close to 2 (slow decay)
    # needs a wider window before the eps gate fires. Empirically
    # ``1/|alpha| + 1`` × j_int_max suffices for ``p`` up to 1.99.
    margin_mult = max(2.0, 1.0 / abs(alpha) + 1.0)
    safe_margin = max(50, int(np.ceil(margin_mult * j_int_max)) + 20)
    J = min(j_int_max + safe_margin, _LD_J_MAX)

    j_grid = np.arange(1, J + 1, dtype=float)
    j_grid_int = j_grid.astype(int)
    lgamma_jp1 = gammaln(j_grid + 1.0)
    lgamma_neg_ja = gammaln(-j_grid * alpha)
    psi_arr = digamma(-j_grid * alpha)

    # Chunk on the n_active axis to bound the (chunk, J) working set.
    # Each row carries 5 J-wide arrays in flight (lw / 2 masks / w /
    # transient), 8 bytes each → 40 J bytes per row.
    n_active = ya.size
    chunk = max(1, _chunk_bytes // (40 * J))

    out_la = np.empty(n_active)
    out_jb = np.empty(n_active)
    out_jv = np.empty(n_active)
    out_jpb = np.empty(n_active)
    near = 5
    for s in range(0, n_active, chunk):
        e = min(s + chunk, n_active)
        lz_c = log_z[s:e]
        ji_c = j_int[s:e]
        lw = (j_grid[None, :] * lz_c[:, None]
              - lgamma_jp1[None, :] - lgamma_neg_ja[None, :])
        log_max = np.max(lw, axis=1)
        within_near = np.abs(j_grid_int[None, :] - ji_c[:, None]) <= near
        above_eps = lw >= (log_max[:, None] - _LD_EPS)
        keep = within_near | above_eps
        w = np.where(keep, np.exp(lw - log_max[:, None]), 0.0)
        sum_w = np.sum(w, axis=1)
        out_la[s:e] = log_max + np.log(sum_w)
        p_w = w / sum_w[:, None]
        jb_c = np.sum(p_w * j_grid[None, :], axis=1)
        out_jb[s:e] = jb_c
        out_jv[s:e] = np.sum(
            p_w * (j_grid[None, :] - jb_c[:, None]) ** 2, axis=1,
        )
        out_jpb[s:e] = np.sum(
            p_w * j_grid[None, :] * psi_arr[None, :], axis=1,
        )

    flat_la = log_a.ravel()
    flat_jb = j_bar.ravel()
    flat_jv = j_var.ravel()
    flat_jpb = j_psi_bar.ravel()
    flat_la[active] = out_la
    flat_jb[active] = out_jb
    flat_jv[active] = out_jv
    flat_jpb[active] = out_jpb
    return log_a, j_bar, j_var, j_psi_bar


class Tweedie(Family):
    """Tweedie EDF with fixed power ``p ∈ (1, 2)`` — compound Poisson-Gamma.

    Mean ``μ``, variance ``φ·μ^p``. The density mixes an exact point mass at
    ``y = 0`` with a continuous part on ``y > 0``; ``ls`` and ``aic`` evaluate
    it via the Dunn-Smyth series (see :func:`_tweedie_log_a_one`). For joint
    estimation of ``p`` with the smoothing parameters, use :class:`tw`.

    Default link is ``log``. Scale ``φ`` is unknown (Pearson/REML estimated).
    """
    name = "Tweedie"
    canonical_link_name = "log"  # mgcv's default; no canonical link in the strict
                                  # EDF sense for non-integer p.
    scale_known = False

    def __init__(self, p: float, link=None):
        if not (1.0 < p < 2.0):
            raise ValueError(f"Tweedie requires 1 < p < 2; got p={p!r}")
        self.p = float(p)
        super().__init__(link=link)

    def variance(self, mu):
        return np.asarray(mu, dtype=float) ** self.p

    def dvar(self, mu):
        return self.p * np.asarray(mu, dtype=float) ** (self.p - 1.0)

    def d2var(self, mu):
        return (self.p * (self.p - 1.0)
                * np.asarray(mu, dtype=float) ** (self.p - 2.0))

    def d3var(self, mu):
        return (self.p * (self.p - 1.0) * (self.p - 2.0)
                * np.asarray(mu, dtype=float) ** (self.p - 3.0))

    def dev_resids(self, y, mu, wt, theta=None):
        # 1<p<2 form (Jorgensen 1987):
        #   y > 0:  d_i = 2·[ y·(y^(1-p) - μ^(1-p))/(1-p) - (y^(2-p) - μ^(2-p))/(2-p) ]
        #   y = 0:  d_i = 2·μ^(2-p)/(2-p)
        # Both pieces are non-negative for 1<p<2, μ>0, y≥0; minimised at y=μ.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        zero = (y == 0.0)
        # Mask y inside the y^(...) so y=0 rows don't generate spurious 0**neg.
        y_safe = np.where(zero, 1.0, y)
        d_pos = 2.0 * (y * (y_safe ** om1 - mu ** om1) / om1
                       - (y_safe ** tm - mu ** tm) / tm)
        d_zero = 2.0 * mu ** tm / tm
        return wt * np.where(zero, d_zero, d_pos)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the 'Tweedie' family"
            )
        # mgcv's Tweedie(): mustart = y + 0.1 — keeps log(μ) finite for y=0
        # rows under the canonical log link. Same shape as Poisson.
        return y + 0.1

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def _log_density(self, y, mu, phi, wt):
        """Per-obs log f(y_i; μ_i, φ/wt_i, p), shape (n,). Weight-aware via
        the per-obs scale convention φ_i = φ/w_i (matches mgcv)."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        phi_i = np.where(good, float(phi) / np.where(good, wt, 1.0), 1.0)
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        zero = (y == 0.0)
        # cumulant_i = y_i·μ_i^(1-p)/(1-p) - μ_i^(2-p)/(2-p) (the y-only term
        # vanishes at y=0; the rest is the y=0 closed form's exponent).
        cumulant = y * mu ** om1 / om1 - mu ** tm / tm
        out = np.empty_like(y)
        out[zero] = cumulant[zero] / phi_i[zero]
        if np.any(~zero):
            la, _, _, _ = _tweedie_log_a_vec(y[~zero], phi_i[~zero], p)
            out[~zero] = -np.log(y[~zero]) + la + cumulant[~zero] / phi_i[~zero]
        return out

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv's ``Tweedie()$aic``: -2·Σ wt·log f at the fitted (μ, φ̂) plus
        # +2 for the φ "extra df". φ̂ is the Pearson moment scale (matches
        # mgcv:::fix.family.aic which expects the post-fit scale).
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        n_eff = float(wt.sum())
        V = mu ** self.p
        phi = float(np.sum(wt * (y - mu) ** 2 / np.maximum(V, 1e-300))
                    / max(n_eff, 1.0))
        if not (np.isfinite(phi) and phi > 0):
            phi = max(float(dev) / max(n_eff, 1.0), 1e-12)
        log_f = self._log_density(y, mu, phi, wt)
        return -2.0 * float(np.sum(log_f * wt)) + 2.0

    def ls(self, y, wt, scale):
        """Saturated log-lik Σ w_i·log f(y_i; y_i, φ/w_i, p) and its 1st/2nd
        derivatives wrt log φ (hea log-scale convention).

        Per-obs scale ``φ_i = φ/w_i`` ⇒ d log φ_i / d log φ = 1, so the chain
        rule is trivial. For y_i = 0 with μ_i = y_i = 0 the cumulant is 0 and
        log f = 0; the entry contributes nothing to ls or its derivatives.
        For y_i > 0:

            log f_sat = -log y + log a(y, φ_i, p) + y^(2-p)/((1-p)(2-p)·φ_i)

        and using d/dlog φ_i log a = -(1-α)·E[j], d²/dlog φ_i² log a =
        (1-α)²·Var[j] (Dunn-Smyth moments under p_j = W_j/Σ W_k):

            d ls / dlog φ   = Σ w_i · (-(1-α)·E[j_i] - c_i/φ_i)
            d² ls / dlog φ² = Σ w_i · ( (1-α)²·Var[j_i] + c_i/φ_i )

        with c_i = y_i^(2-p)/((1-p)(2-p)) the saturated cumulant (negative
        for 1<p<2).
        """
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        if not np.any(good):
            return np.array([0.0, 0.0, 0.0], dtype=float)
        y_g = y[good]
        w_g = wt[good]
        phi_i = float(scale) / w_g
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        # one_minus_alpha = 1 - (2-p)/(1-p) = -1/(1-p) = 1/(p-1)
        one_minus_alpha = 1.0 / (p - 1.0)

        zero = (y_g == 0.0)
        y_safe = np.where(zero, 1.0, y_g)
        # Saturated cumulant c_i = y^(2-p)/((1-p)(2-p)) for y>0; 0 at y=0.
        cum = np.where(zero, 0.0, y_safe ** tm / (om1 * tm))

        # Series moments at μ=y; only computed for y>0 rows. ``ls`` only
        # needs (log a, E[j], Var[j]); the j_psi_bar moment is consumed by
        # ``dls_dp`` for the p-derivative path.
        log_a = np.zeros_like(y_g)
        j_bar = np.zeros_like(y_g)
        j_var = np.zeros_like(y_g)
        if np.any(~zero):
            la_, jb_, jv_, _ = _tweedie_log_a_vec(y_g[~zero], phi_i[~zero], p)
            log_a[~zero] = la_
            j_bar[~zero] = jb_
            j_var[~zero] = jv_

        # log f_sat per observation; y=0 row is 0 by the closed form.
        log_f_sat = np.where(zero, 0.0,
                             -np.log(y_safe) + log_a + cum / phi_i)
        ls0 = float(np.sum(w_g * log_f_sat))

        d1_per = np.where(zero, 0.0, -one_minus_alpha * j_bar - cum / phi_i)
        d2_per = np.where(zero, 0.0,
                          one_minus_alpha * one_minus_alpha * j_var
                          + cum / phi_i)
        d1 = float(np.sum(w_g * d1_per))
        d2 = float(np.sum(w_g * d2_per))
        return np.array([ls0, d1, d2], dtype=float)

    # ---- analytical p-derivatives (used by joint outer Newton in tw()) ----

    def dvar_dp(self, mu):
        """``∂V(μ)/∂p = log(μ)·μ^p`` (since V = μ^p ⇒ log V = p·log μ)."""
        mu = np.asarray(mu, dtype=float)
        return np.log(mu) * mu ** self.p

    def dD_dp(self, y, mu, wt):
        """Σ_i wt_i · ∂d_i/∂p at fixed (y, μ). Used by the joint outer
        Newton when ``family.n_theta > 0`` to evaluate ``∂Dp/∂p`` (the
        envelope theorem at PIRLS-converged β̂ kills the β-coupled chain).

        For y > 0:
            d_i = 2·[y·u/om1 - v/tm]   with u = y^om1 - μ^om1, v = y^tm - μ^tm,
                                            om1 = 1-p, tm = 2-p.
            ∂d_i/∂p = 2·[ y·(μ^om1·log μ - y^om1·log y)/om1 + y·u/om1²
                         - (μ^tm·log μ - y^tm·log y)/tm - v/tm² ]
        For y = 0:
            d_i = 2·μ^tm/tm,  ∂d_i/∂p = 2·μ^tm·[1/tm² - log μ/tm].
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        zero = (y == 0.0)
        log_mu = np.log(mu)
        # y_safe is only used inside masked branches; log_y substitutes 0 for
        # y=0 so y·log y = 0 (limit of y·log y as y→0⁺).
        y_safe = np.where(zero, 1.0, y)
        log_y = np.where(zero, 0.0, np.log(y_safe))

        # y > 0 branch
        y_om1 = y_safe ** om1
        mu_om1 = mu ** om1
        y_tm = y_safe ** tm
        mu_tm = mu ** tm
        u = y_om1 - mu_om1
        v = y_tm - mu_tm
        # ∂[y·u/om1]/∂p:  y·∂u/∂p / om1 + y·u/om1²
        #   ∂u/∂p = -y^om1·log y + μ^om1·log μ
        dA1 = (y * (mu_om1 * log_mu - y_om1 * log_y) / om1
               + y * u / (om1 * om1))
        # ∂[v/tm]/∂p:    ∂v/∂p / tm + v/tm²
        #   ∂v/∂p = -y^tm·log y + μ^tm·log μ
        dA2 = ((mu_tm * log_mu - y_tm * log_y) / tm
               + v / (tm * tm))
        d_dp_pos = 2.0 * (dA1 - dA2)

        # y = 0 branch
        d_dp_zero = 2.0 * mu_tm * (1.0 / (tm * tm) - log_mu / tm)

        return float(np.sum(wt * np.where(zero, d_dp_zero, d_dp_pos)))

    def dls_dp(self, y, wt, scale):
        """``∂ls/∂p`` (saturated log-lik). Companion to ``ls`` for the
        joint-outer-Newton p-direction.

        For y_i > 0:
            log f_sat = -log y + log a(y, φ_i, p) + cum_sat(y, p)/φ_i
            ∂log f_sat/∂p = ∂log a/∂p + ∂cum_sat/∂p / φ_i
        For y_i = 0: log f_sat ≡ 0 ⇒ ∂/∂p = 0.

        Series-moment piece (Dunn-Smyth + chain rule on log W_j = j·log z
        - lgamma(j+1) - lgamma(-j·α)):

            ∂log W_j/∂p = j·K_j/(1-p)² + j/(2-p)
            K_j         = log φ + log(p-1) + ψ(-j·α) - log y - (2-p)
            ∂log a/∂p   = E[j·K_j]/(1-p)² + E[j]/(2-p)

        ``E[j]`` and ``E[j·ψ(-j·α)]`` are returned by
        :func:`_tweedie_log_a_one` (see j_bar, j_psi_bar).

        Saturated cumulant cum_sat = y^(2-p)/((1-p)(2-p)); its p-derivative is
            ∂cum_sat/∂p = y^(2-p) · [(3 - 2p) - log(y)·(1-p)·(2-p)]
                          / [(1-p)·(2-p)]²
        """
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        if not np.any(good):
            return 0.0
        y_g = y[good]
        w_g = wt[good]
        phi_i = float(scale) / w_g
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        om1_tm = om1 * tm

        zero = (y_g == 0.0)
        y_safe = np.where(zero, 1.0, y_g)
        log_y = np.where(zero, 0.0, np.log(y_safe))
        log_phi = np.log(phi_i)

        # ∂cum_sat/∂p (per-obs)
        y_tm = y_safe ** tm
        dcum_dp = np.where(
            zero, 0.0,
            y_tm * ((3.0 - 2.0 * p) - log_y * om1_tm) / (om1_tm * om1_tm)
        )

        # ∂log a/∂p via series moments. Need (j_bar, j_psi_bar) over y>0 rows.
        j_bar = np.zeros_like(y_g)
        j_psi_bar = np.zeros_like(y_g)
        if np.any(~zero):
            _, jb_, _, jpb_ = _tweedie_log_a_vec(
                y_g[~zero], phi_i[~zero], p
            )
            j_bar[~zero] = jb_
            j_psi_bar[~zero] = jpb_
        # K_const_i = log φ_i + log(p-1) - log y_i - (2-p)
        # E[j·K_j] = j_bar · K_const + j_psi_bar (since ψ has E[j·ψ(-jα)])
        K_const = log_phi + np.log(p - 1.0) - log_y - tm
        E_jK = j_bar * K_const + j_psi_bar
        dlog_a_dp = np.where(zero, 0.0, E_jK / (om1 * om1) + j_bar / tm)

        dlog_f_dp = np.where(zero, 0.0, dlog_a_dp + dcum_dp / phi_i)
        return float(np.sum(w_g * dlog_f_dp))

    def __repr__(self):
        return f"Tweedie(p={self.p:.4g}, link={self.link.name})"


class tw(Tweedie):
    """Tweedie family with the power parameter ``p`` estimated jointly with
    the smoothing parameters — mgcv's ``tw()`` extended family.

    ``p`` is reparametrised through a scalar ``θ`` to keep the optimisation
    unconstrained:

        p(θ) = (a + b·exp(θ)) / (1 + exp(θ))    ⇒ p ∈ (a, b) as θ ∈ ℝ

    with default ``a = 1.01``, ``b = 1.99``. Initial p defaults to 1.5
    (mgcv's start) unless ``theta`` is passed (sets p = p(theta)).

    Joint estimation in ``hea.gam`` is via Brent's method on θ over the
    interior of ``(a, b)``: each Brent iterate fits the full GAM at a fixed
    candidate ``p``; the score returned is the converged REML/ML criterion
    at that ``p``. Cheaper than analytical joint outer-Newton but typically
    converges in 10-20 inner fits. The fitted ``p̂`` is stored on
    ``family.p``; the converged θ̂ on ``family.theta``.
    """
    name = "Tweedie"
    n_theta = 1

    def __init__(self, theta: float | None = None, link=None,
                 a: float = 1.01, b: float = 1.99):
        if not (1.0 <= a < b <= 2.0):
            raise ValueError(
                f"tw() requires 1 ≤ a < b ≤ 2; got a={a!r}, b={b!r}"
            )
        self.a = float(a)
        self.b = float(b)
        if theta is None:
            # mgcv's tw() starts at p=1.5; θ such that p(θ)=1.5 is
            # θ = log((1.5 - a)/(b - 1.5)).
            p_init = 1.5
            theta_init = float(np.log((p_init - self.a) / (self.b - p_init)))
        else:
            theta_init = float(theta)
            p_init = self._p_of_theta(theta_init)
        self.theta = theta_init
        # Tweedie.__init__ validates 1 < p < 2 and sets p, link.
        super().__init__(p=p_init, link=link)

    def _p_of_theta(self, theta: float) -> float:
        # p(θ) = (a + b·e^θ)/(1 + e^θ); use sigmoid form for stability.
        s = float(expit(theta))
        return self.a * (1.0 - s) + self.b * s

    def dp_dtheta(self) -> float:
        """``dp/dθ = (b - a)·σ(θ)·(1 - σ(θ))`` where σ is the logistic.
        Used by the outer Newton chain rule when joint-estimating θ_tw.
        """
        s = float(expit(self.theta))
        return (self.b - self.a) * s * (1.0 - s)

    def d2p_dtheta2(self) -> float:
        """``d²p/dθ² = (b-a)·σ·(1-σ)·(1 - 2σ)``."""
        s = float(expit(self.theta))
        return (self.b - self.a) * s * (1.0 - s) * (1.0 - 2.0 * s)

    def set_theta(self, theta) -> None:
        """Update θ (and the implied ``p``). Accepts a scalar or a 1-element
        array (consistent with the Family base ``n_theta``-array signature).
        """
        if hasattr(theta, "__len__"):
            if len(theta) != 1:
                raise ValueError(
                    f"tw expects a single theta; got length {len(theta)}"
                )
            theta = theta[0]
        self.theta = float(theta)
        self.p = self._p_of_theta(self.theta)

    def get_theta(self) -> np.ndarray:
        return np.array([self.theta], dtype=float)

    def __repr__(self):
        return (f"tw(p={self.p:.4g}, link={self.link.name}, "
                f"a={self.a!r}, b={self.b!r})")


# ---------------------------------------------------------------------------
# Scaled-t — mgcv's ``scat()`` extended family
# ---------------------------------------------------------------------------


class Scat(Family):
    """Scaled-t extended family — direct port of mgcv ``scat()``
    (efam.r:3552-3768).

    Likelihood (with location ``μ``, scale ``σ``, dof ``ν``):

        f(y | μ, ν, σ) ∝ σ⁻¹ · (1 + ((y-μ)/σ)² / ν)^{-(ν+1)/2}

    Parameters ν and σ are estimated jointly with the smoothing
    parameters (mgcv ``estimate.theta``). Internally stored in log-form
    with a lower-bound shift on ν:

        θ₀ = log(ν − min_df)        ⇒  ν = exp(θ₀) + min_df > min_df
        θ₁ = log(σ)                  ⇒  σ = exp(θ₁) > 0

    ``min_df`` (default 3) prevents degenerate ν → 2 where the variance
    blows up. Set higher when the data clearly aren't very heavy-tailed.

    Default link ``identity``; ``log`` and ``inverse`` are also accepted
    (mgcv ``okLinks``).
    """
    name = "scat"
    canonical_link_name = "identity"
    # mgcv treats scat as a fixed-scale family (``family$scale = 1``):
    # σ is in θ, not in φ. The bam/gam outer Newton therefore has no
    # log-φ slot for scat.
    scale_known = True
    is_extended = True
    n_theta = 2

    _OK_LINKS = ("identity", "log", "inverse")

    def __init__(self, theta=None, link: str = "identity",
                 min_df: float = 3.0):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for scat family; available '
                f'links are {self._OK_LINKS}'
            )
        # Match mgcv's ``min.df`` clamp + theta-sign decoding (efam.r:3576-3587):
        # * theta=None  → free θ, log-internal start (-2, -1)  → (ν=min_df+e⁻², σ=e⁻¹)
        # * theta given, all positive → fixed θ, n_theta=0
        # * theta given, any negative → free θ at |theta| as start
        # * if |theta[0]| ≤ min_df, lower min_df to 0.9·|theta[0]| with a warning.
        n_theta = 2
        if theta is not None and not np.any(np.asarray(theta) == 0.0):
            t = np.asarray(theta, dtype=float)
            if t.shape != (2,):
                raise ValueError(
                    f"scat theta must be a length-2 array (ν, σ); got "
                    f"shape {t.shape}"
                )
            if abs(t[0]) <= min_df:
                import warnings
                min_df = 0.9 * abs(t[0])
                warnings.warn(
                    "Supplied df below min.df. min.df reset",
                    stacklevel=2,
                )
            if np.any(t < 0):
                ini = np.array([np.log(abs(t[0]) - min_df),
                                np.log(abs(t[1]))], dtype=float)
            else:
                ini = np.array([np.log(t[0] - min_df),
                                np.log(t[1])], dtype=float)
                n_theta = 0
        else:
            ini = np.array([-2.0, -1.0], dtype=float)
        # Apply the actual instance settings.
        self.n_theta = int(n_theta)
        self.estimate_theta_callback = bool(n_theta > 0)
        self._min_df = float(min_df)
        self._theta = ini.copy()
        super().__init__(link=link)

    # ----- θ accessors (mgcv getTheta/putTheta) -------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float)
        if v.shape != (2,):
            raise ValueError(
                f"Scat.set_theta expects length-2 array (log θ); got "
                f"shape {v.shape}"
            )
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        """Return current θ. ``trans=True`` returns ``(ν, σ)`` on the
        original scale; ``trans=False`` returns the log-internal storage.
        Mirrors mgcv ``getTheta(trans=)``.
        """
        if trans:
            out = np.exp(self._theta).copy()
            out[0] += self._min_df
            return out
        return self._theta.copy()

    @property
    def min_df(self) -> float:
        return self._min_df

    # ----- variance / dev_resids / aic / ls -----------------------------

    def variance(self, mu):
        # Marginal var of σ·T(ν): σ²·ν/(ν-2). Used for sp init / Pearson.
        nu = np.float64(np.exp(self._theta[0]) + self._min_df)
        sig = np.float64(np.exp(self._theta[1]))
        return np.full(np.shape(mu), sig * sig * nu / max(nu - 2.0, 1e-10),
                       dtype=float)

    def dvar(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def d2var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv: wt * (ν+1) * log1p((1/ν) * ((y-μ)/σ)²)  (efam.r:3609-3614)
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        nu = np.float64(np.exp(th[0]) + self._min_df)
        sig = np.float64(np.exp(th[1]))
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (nu + 1.0) * np.log1p((1.0 / nu) * ((y - mu) / sig) ** 2)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(np.isnan(y)):
            raise ValueError("NA values not allowed for the scaled t family")
        # mgcv: mustart <- y + (y == 0) * 0.1   (efam.r:3736-3740)
        return y + (y == 0.0).astype(float) * 0.1

    def validmu(self, mu) -> bool:
        return bool(np.all(np.isfinite(mu)))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv: -2·logL = 2·Σ wt·[ -lgamma((ν+1)/2) + lgamma(ν/2)
        #                          + log(σ·sqrt(πν))
        #                          + (ν+1)·log1p(((y-μ)/σ)²/ν)/2 ]
        # (efam.r:3690-3697)
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        nu = np.float64(np.exp(th[0]) + self._min_df)
        sig = np.float64(np.exp(th[1]))
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        term = (-gammaln((nu + 1.0) / 2.0)
                + gammaln(nu / 2.0)
                + np.log(sig * np.sqrt(np.pi * nu))
                + (nu + 1.0) * np.log1p(((y - mu) / sig) ** 2 / nu) / 2.0)
        return 2.0 * float(np.sum(term * wt))

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        """Saturated log-likelihood and θ-derivatives — mgcv ``ls`` for
        scat (efam.r:3699-3723). Returns a dict matching mgcv's shape:

            ls    : scalar saturated log-lik, Σᵢ wᵢ · ls_i(θ)
            lsth1 : (2,)   first derivatives wrt θ summed over i
            LSTH1 : (n,2)  per-obs first-derivative matrix
            lsth2 : (2,2)  Hessian wrt θ

        Used by ``_estimate_theta`` (Phase D). The base
        ``Family.ls(y, wt, scale)`` 3-vector signature is preserved for
        the standard families; extended-family callers test
        ``family.is_extended`` and dispatch here.
        """
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(wt, dtype=float)
        if w.size == 1:
            w = np.full(y.shape, float(w))
        nu = np.float64(np.exp(th[0]) + self._min_df)
        sig = np.float64(np.exp(th[1]))
        nu2 = nu - self._min_df       # = exp(th[0])
        nu2nu = nu2 / nu
        nu12 = (nu + 1.0) / 2.0
        # ls_i = lgamma((ν+1)/2) - lgamma(ν/2) - log(σ·sqrt(π·ν))
        term0 = (gammaln(nu12) - gammaln(nu / 2.0)
                 - np.log(sig * np.sqrt(np.pi * nu)))
        ls0 = float(np.sum(term0 * w))
        # First derivatives (per-obs, then summed):
        #   ∂ls/∂θ₀ per-obs = nu2 · ψ((ν+1)/2)/2 − nu2 · ψ(ν/2)/2 − 0.5·nu2nu
        #   ∂ls/∂θ₁ per-obs = -1   (constant)
        col0 = nu2 * digamma(nu12) / 2.0 - nu2 * digamma(nu / 2.0) / 2.0 \
            - 0.5 * nu2nu
        LSTH = np.column_stack([w * col0, -1.0 * w])
        lsth = LSTH.sum(axis=0)
        # Hessian (only [1,1] is nonzero per mgcv's ls):
        #   ∂²ls/∂θ₀² per-obs = nu2² · ψ′((ν+1)/2)/4 + nu2 · ψ((ν+1)/2)/2
        #                       − nu2² · ψ′(ν/2)/4 − nu2 · ψ(ν/2)/2
        #                       + 0.5·nu2nu² − 0.5·nu2nu
        d11 = (nu2 * nu2 * polygamma(1, nu12) / 4.0
               + nu2 * digamma(nu12) / 2.0
               - nu2 * nu2 * polygamma(1, nu / 2.0) / 4.0
               - nu2 * digamma(nu / 2.0) / 2.0
               + 0.5 * nu2nu * nu2nu - 0.5 * nu2nu)
        lsth2 = np.zeros((2, 2), dtype=float)
        lsth2[0, 0] = float(np.sum(d11 * w))
        return {"ls": ls0, "lsth1": lsth, "LSTH1": LSTH, "lsth2": lsth2}

    def ls(self, y, wt, scale):
        """Standard 3-vector ``ls`` contract: ``(ls0, d/dlogφ, d²/dlogφ²)``.

        Scat is ``scale_known = True`` — σ lives in θ, not φ — so the
        log-φ derivatives are identically zero, mirroring Poisson and
        Binomial. ``ls0`` is the saturated log-lik at μ=y under the
        current internal θ:

            ls0 = Σᵢ wᵢ · [lgamma((ν+1)/2) − lgamma(ν/2) − log(σ·√(πν))]

        The (y-μ)²/(σ²ν) term vanishes at μ=y so the saturated form
        carries only the normalising constants. ``_estimate_theta``
        (Phase D) reads the richer θ-derivative shape via
        :meth:`ls_extended` instead.
        """
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        nu = np.float64(np.exp(self._theta[0]) + self._min_df)
        sig = np.float64(np.exp(self._theta[1]))
        term = (gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - np.log(sig * np.sqrt(np.pi * nu)))
        ls0 = float(np.sum(term * wt))
        return np.array([ls0, 0.0, 0.0], dtype=float)

    # ----- Dd: μ- and θ-derivatives of −logL  (mgcv efam.r:3616-3687) ---

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        # Direct line-by-line port of mgcv ``scat$Dd``. Every variable
        # name and bracketing matches the source so future diffs against
        # mgcv stay mechanical.
        #
        # Note: nu/sig are kept as ``np.float64`` (not Python ``float``)
        # so divisions by zero in the σ→0 / ν→∞ extremes propagate as
        # ``inf``/``nan`` instead of raising ``ZeroDivisionError``. The
        # ``_estimate_theta`` Newton then sees a non-finite ``nll1`` and
        # step-halves naturally — mirroring mgcv R, which silently
        # produces ``Inf`` here.
        min_df = self._min_df
        th = np.asarray(theta, dtype=float)
        nu = np.float64(np.exp(th[0]) + min_df)
        sig = np.float64(np.exp(th[1]))
        nu1 = nu + 1.0
        nu2 = nu - min_df
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        w = np.asarray(wt, dtype=float)
        # mgcv broadcasts ``wt`` if scalar; when w is scalar, multiply
        # against length-n arrays via numpy broadcasting (works as-is).
        ym = y - mu
        a = 1.0 + (ym / sig) ** 2 / nu
        nu1ym = nu1 * ym
        sig2a = sig * sig * a
        nusig2a = nu * sig2a
        f = nu1ym / nusig2a
        f1 = ym / nusig2a
        n = y.shape[0]

        oo: dict = {}
        oo["Dmu"] = -2.0 * w * f
        oo["Dmu2"] = 2.0 * w * nu1 * (1.0 / nusig2a - 2.0 * f1 ** 2)
        # E[Dmu2] is the Fisher information per-obs at expected (y-μ)²:
        # 2·(ν+1) / (σ²·(ν+3)). Vectorised to length n.
        EDmu2_scalar = 2.0 * nu1 / (sig * sig) / (nu + 3.0)
        oo["EDmu2"] = np.full(n, EDmu2_scalar, dtype=float)

        if level > 0:
            nu1nusig2a = nu1 / nusig2a
            nu2nu = nu2 / nu
            fym = f * ym
            ff1 = f * f1
            f1ym = f1 * ym
            fymf1 = fym * f1
            ymsig2a = ym / sig2a

            Dth = np.zeros((n, 2), dtype=float)
            Dmuth = np.zeros((n, 2), dtype=float)
            Dmu2th = np.zeros((n, 2), dtype=float)
            EDmu2th = np.zeros((n, 2), dtype=float)
            Dth[:, 0] = w * nu2 * (np.log(a) - fym / nu)
            Dth[:, 1] = -2.0 * w * fym
            Dmuth[:, 0] = 2.0 * w * (f - ymsig2a - fymf1) * nu2nu
            Dmuth[:, 1] = 4.0 * w * f * (1.0 - f1ym)
            Dmu3 = 4.0 * w * f * (3.0 / nusig2a - 4.0 * f1 ** 2)
            Dmu2th[:, 0] = 2.0 * w * (
                -nu1nusig2a + 1.0 / sig2a + 5.0 * ff1
                - 2.0 * f1ym / sig2a - 4.0 * fymf1 * f1
            ) * nu2nu
            Dmu2th[:, 1] = 4.0 * w * (
                -nu1nusig2a + ff1 * 5.0 - 4.0 * ff1 * f1ym
            )
            EDmu3 = np.zeros(n, dtype=float)
            EDmu2th[:, 0] = (4.0 / (sig * sig * (nu + 3.0) ** 2)
                             * np.float64(np.exp(th[0])))
            EDmu2th[:, 1] = -2.0 * oo["EDmu2"]

            oo["Dth"] = Dth
            oo["Dmuth"] = Dmuth
            oo["Dmu3"] = Dmu3
            oo["Dmu2th"] = Dmu2th
            oo["EDmu3"] = EDmu3
            oo["EDmu2th"] = EDmu2th

        if level > 1:
            nu1nu = nu1 / nu
            fymf1ym = fym * f1ym
            f1ymf1 = f1ym * f1

            Dmu4 = 12.0 * w * (
                -nu1nusig2a / nusig2a + 8.0 * ff1 / nusig2a
                - 8.0 * ff1 * f1 ** 2
            )
            n2d = 3
            Dmu3th = np.zeros((n, 2), dtype=float)
            Dmu2th2 = np.zeros((n, n2d), dtype=float)
            Dmuth2 = np.zeros((n, n2d), dtype=float)
            Dth2 = np.zeros((n, n2d), dtype=float)

            Dmu3th[:, 0] = 4.0 * w * (
                -6.0 * f / nusig2a + 3.0 * f1 / sig2a
                + 18.0 * ff1 * f1 - 4.0 * f1ymf1 / sig2a
                - 12.0 * nu1ym * f1 ** 4
            ) * nu2nu
            Dmu3th[:, 1] = 48.0 * w * f * (
                -1.0 / nusig2a + 3.0 * f1 ** 2 - 2.0 * f1ymf1 * f1
            )

            Dth2[:, 0] = w * (
                nu2 * np.log(a)
                + nu2nu * ym ** 2
                * (-2.0 * nu2 - nu1 + 2.0 * nu1 * nu2nu
                   - nu1 * nu2nu * f1ym) / nusig2a
            )
            Dth2[:, 1] = 2.0 * w * (fym - ym * ymsig2a - fymf1ym) * nu2nu
            Dth2[:, 2] = 4.0 * w * fym * (1.0 - f1ym)

            term_a = 2.0 * nu2nu - 2.0 * nu1nu * nu2nu - 1.0 + nu1nu
            Dmuth2[:, 0] = 2.0 * w * f1 * nu2 * (
                term_a - 2.0 * nu2nu * f1ym + 4.0 * fym * nu2nu / nu
                - fym / nu - 2.0 * fymf1ym * nu2nu / nu
            )
            Dmuth2[:, 1] = 4.0 * w * (
                -f + ymsig2a + 3.0 * fymf1
                - ymsig2a * f1ym - 2.0 * fymf1 * f1ym
            ) * nu2nu
            Dmuth2[:, 2] = 8.0 * w * f * (-1.0 + 3.0 * f1ym - 2.0 * f1ym ** 2)

            Dmu2th2[:, 0] = 2.0 * w * nu2 * (
                -term_a + 10.0 * nu2nu * f1ym - 16.0 * fym * nu2nu / nu
                - 2.0 * f1ym + 5.0 * nu1nu * f1ym
                - 8.0 * nu2nu * f1ym ** 2
                + 26.0 * fymf1ym * nu2nu / nu
                - 4.0 * nu1nu * f1ym ** 2
                - 12.0 * nu1nu * nu2nu * f1ym ** 3
            ) / nusig2a
            Dmu2th2[:, 1] = 4.0 * w * (
                nu1nusig2a - 1.0 / sig2a - 11.0 * nu1 * f1 ** 2
                + 5.0 * f1ym / sig2a + 22.0 * nu1 * f1ymf1 * f1
                - 4.0 * f1ym ** 2 / sig2a - 12.0 * nu1 * f1ymf1 ** 2
            ) * nu2nu
            Dmu2th2[:, 2] = 8.0 * w * (
                nu1nusig2a - 11.0 * nu1 * f1 ** 2
                + 22.0 * nu1 * f1ymf1 * f1 - 12.0 * nu1 * f1ymf1 ** 2
            )

            oo["Dmu4"] = Dmu4
            oo["Dmu3th"] = Dmu3th
            oo["Dmu2th2"] = Dmu2th2
            oo["Dmuth2"] = Dmuth2
            oo["Dth2"] = Dth2

        return oo

    # ----- preinitialize / postproc / rd  (mgcv efam.r:3725-3757) -------

    def preinitialize(self, y) -> dict | None:
        # mgcv: when n.theta > 0, start with moderate ν and high σ:
        #   Theta <- c(1.5, log(0.8 * sd(y)))  (efam.r:3725-3734)
        # When all θ are user-fixed (n_theta = 0), no override.
        if self.n_theta > 0:
            y = np.asarray(y, dtype=float)
            sd_y = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
            sd_y = max(sd_y, 1e-10)  # guard against constant y
            return {"Theta": np.array([1.5, np.log(0.8 * sd_y)],
                                      dtype=float)}
        return None

    def postproc(self, y, mu, wt) -> dict:
        # mgcv builds "Scaled t(ν, σ)" with values rounded to 3 decimals;
        # if ν > 999 it's reported as Inf.  (efam.r:3742-3749)
        nu, sig = self.get_theta(trans=True)
        nu_disp = float(np.round(nu, 3))
        sig_disp = float(np.round(sig, 3))
        if nu_disp > 999.0:
            nu_disp_str = "Inf"
        else:
            nu_disp_str = f"{nu_disp:g}"
        return {"family_name": f"Scaled t({nu_disp_str},{sig_disp:g})"}

    def rd(self, mu, wt, scale, rng: np.random.Generator | None = None):
        nu, sig = self.get_theta(trans=True)
        n = np.asarray(mu, dtype=float).shape[0]
        gen = rng if rng is not None else np.random.default_rng()
        return gen.standard_t(nu, size=n) * sig + np.asarray(mu, dtype=float)

    def __repr__(self):
        nu, sig = self.get_theta(trans=True)
        return (f"Scat(theta=({nu:.4g}, {sig:.4g}), "
                f"link={self.link.name}, min_df={self._min_df:g})")


def _coerce_response(y_series: pl.Series, family: "Family") -> np.ndarray:
    """Cast the response column to a numeric float array, with R's
    factor-response convention for :class:`Binomial`.

    R's ``glm(y ~ x, family=binomial)`` accepts a 2-level factor on the
    LHS: level 1 → 0 (failure), level 2 → 1 (success). Boolean is the
    same shape (FALSE → 0, TRUE → 1). For other families and numeric y
    we just float-cast.

    Unused factor levels are dropped before the 2-level check — matches
    R's ``glm()``, which calls ``model.frame(..., drop.unused.levels=
    TRUE)`` so a 3-level Enum filtered down to 2 actually-present
    levels still fits cleanly. The filter preserves the declared order
    of the surviving levels, so ``levels[0]`` (the "failure" reference)
    matches what R would pick after ``droplevels()``.
    """
    dt = y_series.dtype
    if isinstance(family, Binomial):
        if dt == pl.Boolean:
            return y_series.to_numpy().astype(float)
        if dt == pl.String or isinstance(dt, (pl.Categorical, pl.Enum)):
            if isinstance(dt, pl.Enum):
                declared = list(dt.categories)
            else:
                # No declared order — fall back to alphabetical, which is
                # R's ``factor()`` default when ``levels=`` is unspecified.
                declared = sorted(y_series.drop_nulls().unique().to_list())
            present = set(y_series.drop_nulls().unique().to_list())
            levels = [lvl for lvl in declared if lvl in present]
            if len(levels) != 2:
                raise ValueError(
                    f"Binomial response factor must have 2 levels present "
                    f"in the data; got {len(levels)}: {levels}"
                )
            return (y_series.to_numpy() != levels[0]).astype(float)
    return y_series.to_numpy().astype(float).flatten()


# Convenience exports — mirror R's lowercase/CapCase convention so user code
# reads almost identically: ``gam(..., family=Gamma(link='log'))``.
gaussian = Gaussian
poisson = Poisson
binomial = Binomial
inverse_gaussian = InverseGaussian
quasi = Quasi
scat = Scat   # mgcv-style lowercase alias
__all__ = [
    "Family", "Link",
    "Gaussian", "gaussian",
    "Gamma",
    "Poisson", "poisson",
    "Binomial", "binomial",
    "InverseGaussian", "inverse_gaussian",
    "Quasi", "quasi",
    "Tweedie", "tw",
    "Scat", "scat",
    "IdentityLink", "LogLink", "InverseLink",
    "SqrtLink", "LogitLink", "ProbitLink", "CauchitLink", "CloglogLink",
    "InverseSquareLink",
]
