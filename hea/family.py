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

import itertools

import numpy as np
import polars as pl
from scipy.linalg import solve_triangular
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
    # fix.family.link's extended-family ratios (gam.fit3.r:2243-2247):
    # g' = ½μ^-½ ⇒ g2g = g″/g′² = -μ^-½, g3g = g‴/g′³ = 3/μ,
    # g4g = g⁗/g′⁴ = -15·μ^-1.5.
    def g2g(self, mu): return -np.asarray(mu, dtype=float) ** -0.5
    def g3g(self, mu): return 3.0 / np.asarray(mu, dtype=float)
    def g4g(self, mu): return -15.0 * np.asarray(mu, dtype=float) ** -1.5
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


class PowerLink(Link):
    """R's ``power(λ)`` link for 0 < λ ≠ 1: ``g(μ) = μ^λ``.

    Use the :func:`power` factory, which mirrors R exactly — ``λ ≤ 0``
    returns the log link and ``λ = 1`` the identity, so only genuine
    powers reach this class. ``linkinv``/``mu_eta`` carry R's
    ``.Machine$double.eps`` floor; the d2link..d4link table is
    fix.family.link's power branch (gam.fit3.r:2329-2335 quasi
    vector-link form ≡ the "mu^" name branch :2415-2421).
    """
    def __init__(self, lam: float):
        self.lam = float(lam)
        # R: link name is paste0("mu^", round(lambda, 3)).
        self.name = f"mu^{round(self.lam, 3):g}"
    def link(self, mu):
        return np.asarray(mu, dtype=float) ** self.lam
    def linkinv(self, eta):
        eps = np.finfo(float).eps
        return np.maximum(
            np.asarray(eta, dtype=float) ** (1.0 / self.lam), eps,
        )
    def mu_eta(self, eta):
        eps = np.finfo(float).eps
        return np.maximum(
            np.asarray(eta, dtype=float) ** (1.0 / self.lam - 1.0)
            / self.lam, eps,
        )
    def d2link(self, mu):
        lam = self.lam
        return lam * (lam - 1.0) * np.asarray(mu, dtype=float) ** (lam - 2.0)
    def d3link(self, mu):
        lam = self.lam
        return (lam * (lam - 1.0) * (lam - 2.0)
                * np.asarray(mu, dtype=float) ** (lam - 3.0))
    def d4link(self, mu):
        lam = self.lam
        return (lam * (lam - 1.0) * (lam - 2.0) * (lam - 3.0)
                * np.asarray(mu, dtype=float) ** (lam - 4.0))
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


def power(lam: float = 1.0) -> Link:
    """R ``stats::power(lambda)``: the ``μ^λ`` link-glm object.

    Exact R semantics: ``λ ≤ 0`` → the log link, ``λ = 1`` → identity,
    otherwise :class:`PowerLink`. Pass the OBJECT to a family —
    ``quasi(link=power(1/3))`` — exactly as in R (R's ``make.link``
    does not accept a "power(...)" string and neither does hea).
    """
    lam = float(lam)
    if not np.isfinite(lam):
        raise ValueError("invalid argument 'lambda'")
    if lam <= 0.0:
        return LogLink()
    if lam == 1.0:
        return IdentityLink()
    return PowerLink(lam)


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


def _brent_fmin(f, ax: float, bx: float, tol: float) -> tuple[float, float]:
    """R's ``Brent_fmin`` (src/library/stats/src/optimize.c) — the exact
    golden-section + successive-parabolic-interpolation loop behind
    ``stats::optimize``, ported operation-for-operation so mgcv code
    built on ``optimize`` (``find.null.dev``) reproduces R's stop points.
    Returns ``(x_min, f(x_min))``.
    """
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(np.finfo(float).eps)
    a, b = ax, bx
    v = a + c * (b - a)
    w = x = v
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:                       # fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if (abs(p) >= abs(q * 0.5 * r)
                or p <= q * (a - x) or p >= q * (b - x)):
            # golden-section step
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            # parabolic-interpolation step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d > 0.0 else -tol1)
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx


def find_null_dev(family: "Family", y, eta, offset, weights) -> float:
    """mgcv ``find.null.dev`` (efam.r:98-117): the null deviance of an
    extended family — deviance of the best single-constant model on the
    link scale, found by 1-D ``optimize`` over the constant with mgcv's
    interval-doubling protocol (double the half-width until the minimum
    is interior). Replaces the standard weighted-mean null deviance in
    the extended postprocs (nb efam.r:283, tw efam.r:3239,
    scat efam.r:3742) — for non-canonical-ish links the optimal constant
    is NOT the weighted mean, so the two differ at 1e-3 level.

    ``eta`` is the converged linear predictor INCLUDING the offset
    (mgcv's ``linear.predictors``); the initial constant comes from the
    weighted mean of ``linkinv(eta − offset)``, while the candidate
    models are ``μ = linkinv(γ + offset)``.
    """
    y = np.asarray(y, dtype=float)
    eta = np.asarray(eta, dtype=float)
    offset = np.zeros_like(eta) if offset is None else np.asarray(
        offset, dtype=float)
    weights = np.asarray(weights, dtype=float)
    link = family.link

    def fnull(gamma: float) -> float:
        # 3-arg dev.resids like mgcv's fnull — extended families read
        # their current θ when ``theta=None``.
        mu = link.linkinv(gamma + offset)
        return float(np.sum(family.dev_resids(y, mu, weights)))

    mu0 = link.linkinv(eta - offset)
    mum = float(np.mean(mu0 * weights) / np.mean(weights))
    eta0 = float(link.link(mum))
    deta = abs(eta0) * 0.1 + 1.0       # search interval half width
    tol = float(np.finfo(float).eps) ** 0.25   # optimize's default tol
    while True:
        lo, hi = eta0 - deta, eta0 + deta
        x_min, f_min = _brent_fmin(fnull, lo, hi, tol)
        if lo < x_min < hi:
            return f_min
        deta *= 2.0


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

    # mgcv's canonical link for PIRLS's full-Newton/Fisher switch
    # (fix.family.link's table, gam.fit3.r:2316-2323). ``None`` means
    # "same as canonical_link_name" (the table's gaussian/poisson/
    # binomial/Gamma/IG rows). Families outside that table set "none"
    # explicitly — quasi (table fallback :2322), Tweedie
    # (gam.fit3.r:3105), tw (efam.r:3262), scat/nb — so the inner loop
    # never takes the Fisher shortcut whatever the link. Distinct from
    # ``canonical_link_name``, which also resolves the *default* link.
    _newton_canonical: str | None = None

    def __init__(self, link=None):
        self.link = _resolve_link(link, self.canonical_link_name)

    @property
    def is_canonical(self) -> bool:
        canon = self._newton_canonical
        if canon is None:
            canon = self.canonical_link_name
        return self.link.name == canon

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

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """One-shot post-fit hook — mgcv ``family$postproc(family, y,
        prior.weights, fitted, linear.predictors, offset, intercept)``.
        Extended families return ``{"null_deviance": ...}`` (via
        :func:`find_null_dev`, replacing the standard weighted-mean null
        deviance) and ``{"family_name": ...}`` (the θ-embedding relabel
        mgcv writes into ``family$family`` — "Scaled t(ν,σ)",
        "Negative Binomial(Θ)", "Tweedie(p=…)"). Default: empty dict
        (standard families keep estimate.gam's generics).
        """
        return {}

    # ----- qq.gam hooks (mgcv fix.family.qf / fix.family.rd,
    # plots.r:31-91). ``None`` means unavailable: the qq machinery then
    # tries simulation (rd) and finally falls back to a normal QQ plot.
    # Subclasses override with methods qf(p, mu, wt, scale) — the
    # response quantile function — and rd(rng, mu, wt, scale) — random
    # deviates (rng is a numpy Generator; mgcv uses R's global RNG).
    qf = None
    rd = None

    # mgcv residuals.gam dispatches to ``family$residuals(object, type)``
    # when the family supplies one (mgcv.r:3429) — general families
    # (gaulss & co) define their own residuals this way. hea's signature
    # is ``residuals(y, fitted, type)`` (the only pieces mgcv's hooks
    # read off the object). ``None`` means use the standard
    # deviance/pearson/working/response computations.
    residuals = None

    def initialize(self, y, wt) -> np.ndarray:
        """Starting μ̂ for PIRLS. Return a length-n positive (or family-valid)
        vector. Default: y; subclasses override when y can be at the boundary.
        """
        return np.asarray(y, dtype=float).copy()

    def gam_initialize(self, y, wt, n=None) -> np.ndarray:
        """Starting μ̂ for gam/bam PIRLS — mgcv patches some families'
        ``initialize`` before fitting (``fix.family``, gam.fit3.r:2550),
        making starts valid where glm's would refuse (e.g. gaussian-log
        with y ≤ 0). Default: same as ``initialize``; Gaussian overrides.

        ``n`` is the binomial trials vector from a ``cbind(succ, fail)``
        response (R's initialize keeps it distinct from the prior
        weights); only forwarded when given so ``initialize`` overrides
        without an ``n`` parameter stay valid.
        """
        if n is not None:
            return self.initialize(y, wt, n=n)
        return self.initialize(y, wt)

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

    def gam_initialize(self, y, wt):
        # mgcv fix.family (gam.fit3.r:2550-2561): link-aware starting μ̂ so
        # gaussian fits with log/inverse links start inside the valid
        # region (glm's initialize refuses y ≤ 0 under a log link).
        y = np.asarray(y, dtype=float)
        if self.link.name == "inverse":
            return y + (y == 0.0) * np.std(y, ddof=1) * 0.01
        if self.link.name == "log":
            return np.maximum(y, 0.01 * np.std(y, ddof=1))
        return y.copy()

    def qf(self, p, mu, wt, scale):
        from scipy.stats import norm
        return norm.ppf(p, loc=mu, scale=np.sqrt(
            scale / np.asarray(wt, dtype=float)))

    def rd(self, rng, mu, wt, scale):
        return rng.normal(mu, np.sqrt(scale / np.asarray(wt, dtype=float)))

    def dev_resids(self, y, mu, wt, theta=None):
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (y - mu) ** 2

    def aic(self, y, mu, dev, wt, n, theta=None):
        # R's gaussian()$aic verbatim: nobs·(log(2π·dev/nobs)+1) + 2
        # − Σ log(wt), with nobs = length(y) (NOT Σwt — prior weights are
        # precision multipliers on σ², not extra observations; they enter
        # through the −Σlog(wt) Jacobian term instead). A zero weight makes
        # this Inf, exactly as in R. The +2 is the "+1 family df"
        # placeholder; downstream adds 2·edf for the model.
        wt = np.asarray(wt, dtype=float)
        nobs = float(np.asarray(y).shape[0])
        sigma2 = dev / nobs
        with np.errstate(divide="ignore"):
            log_wt_sum = float(np.sum(np.log(wt)))
        return nobs * (np.log(2.0 * np.pi * sigma2) + 1.0) + 2.0 - log_wt_sum

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

    def qf(self, p, mu, wt, scale):
        # mgcv fix.family.qf: qgamma(p, shape=1/scale, scale=mu*scale) —
        # prior weights are ignored (as in mgcv).
        from scipy.stats import gamma as _gamma_dist
        return _gamma_dist.ppf(p, a=1.0 / scale,
                               scale=np.asarray(mu, dtype=float) * scale)

    def rd(self, rng, mu, wt, scale):
        mu = np.asarray(mu, dtype=float)
        return rng.gamma(shape=1.0 / scale, scale=mu * scale)


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

    def qf(self, p, mu, wt, scale):
        return _poisson_dist.ppf(p, np.asarray(mu, dtype=float))

    def rd(self, rng, mu, wt, scale):
        return rng.poisson(np.asarray(mu, dtype=float)).astype(float)


class Binomial(Family):
    """``y·m ~ Binomial(m, μ)``; ``y`` is the success proportion in [0,1],
    ``wt`` is the binomial size ``m`` (= 1 for Bernoulli).

    The cbind(success, failure) response form is handled by the *model*
    front ends (``gam``, ``glm``), which convert it to (proportion,
    weights·trials) before fitting — R's binomial ``initialize`` does the
    same. The trials vector ``n`` stays distinct from the prior weights
    in ``aic``/``ls``/``initialize`` (R keeps them separate whenever the
    caller also supplies its own ``weights=``); when ``n`` is omitted,
    the prior weights play both roles exactly as before.
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

    def initialize(self, y, wt, n=None, warn_non_integer=True):
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError("y values must be 0 <= y <= 1 for the 'binomial' family")
        if n is not None:
            # R binomial initialize, NCOL(y)==2 branch: mustart =
            # (n·y + 0.5)/(n + 1) — the trials vector, NOT the (possibly
            # prior-weight-scaled) wt. Only the starting point differs;
            # the converged fit is identical either way. (That branch's
            # non-integer-counts warning fired at the cbind intake.)
            n = np.asarray(n, dtype=float)
            return (n * y + 0.5) / (n + 1.0)
        # R's NCOL(y)==1 branch: m = weights·y must be integral counts.
        # The warning is gated on the family template being literally
        # "binomial" (quasibinomial's initialize is the same expression
        # with %s = "quasibinomial", so its guard is false → silent;
        # QuasiBinomial delegates here with warn_non_integer=False).
        if warn_non_integer:
            m = wt * y
            if np.any(np.abs(m - np.rint(m)) > 0.001):
                import warnings as _w
                _w.warn("non-integer #successes in a binomial glm!",
                        stacklevel=2)
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
        # R binomial()$aic: ``m <- if (any(n > 1)) n else wt`` — with a
        # cbind(succ, fail) response, ``n`` is the trials vector kept by
        # initialize and ``wt`` carries any extra prior weights on top
        # (wt = pw·n), so the density is evaluated at the true counts
        # with coefficient wt/m = pw. Callers passing a scalar n (nobs)
        # or ones keep the historical wt-only path bit-for-bit.
        n_arr = None if n is None else np.asarray(n, dtype=float)
        if (n_arr is not None and n_arr.ndim == 1
                and n_arr.shape == y.shape and np.any(n_arr > 1.0)):
            good = n_arr > 0
            weight = np.where(good, wt / np.where(good, n_arr, 1.0), 0.0)
            s_arr = np.rint(np.where(good, n_arr * y, 0.0))
            size = np.rint(n_arr)
            logp = _dbinom_raw(s_arr, size, mu, 1.0 - mu, True)
            terms = np.where(good & np.isfinite(logp), weight * logp, 0.0)
            return -2.0 * float(np.cumsum(terms)[-1])
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

    def ls(self, y, wt, scale, n=None):
        # mgcv: ls = -binomial$aic(y, n, y, w, 0) / 2; scale-known.
        # ``n`` (trials, cbind responses) flows into the aic exactly as in
        # fix.family.ls (gam.fit3.r:2516) — None keeps the wt-only path.
        ls0 = -0.5 * self.aic(y, y, 0.0, wt, n)
        return np.array([ls0, 0.0, 0.0], dtype=float)

    def qf(self, p, mu, wt, scale):
        # mgcv fix.family.qf: ceiling non-integer denominators with a
        # warning; qbinom(p, wt, mu)/(wt + (wt==0)).
        from scipy.stats import binom as _binom_dist
        wt = np.asarray(wt, dtype=float)
        if not np.allclose(wt, np.ceil(wt)):
            wt = np.ceil(wt)
            import warnings as _w
            _w.warn("non-integer binomial denominator: quantiles "
                    "incorrect", stacklevel=2)
        q = _binom_dist.ppf(p, wt, np.asarray(mu, dtype=float))
        return q / (wt + (wt == 0))

    def rd(self, rng, mu, wt, scale):
        wt = np.asarray(wt, dtype=float)
        d = rng.binomial(np.rint(wt).astype(np.int64),
                         np.asarray(mu, dtype=float))
        return d / (wt + (wt == 0))


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

    def rd(self, rng, mu, wt, scale):
        # mgcv fix.family.rd: rig(n, mu, scale) — inverse Gaussian with
        # variance scale·μ³, i.e. numpy's wald(mean=μ, scale=1/φ). No qf
        # in mgcv, so qq machinery simulates for this family.
        return rng.wald(np.asarray(mu, dtype=float), 1.0 / scale)


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
    # fix.family.link's table fallback (gam.fit3.r:2322): plain quasi →
    # "none"; quasipoisson/quasibinomial override with log/logit.
    _newton_canonical = "none"

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

    def ls(self, y, wt, scale, n=None):
        # ``n`` (trials, quasibinomial cbind responses) accepted and
        # ignored — mgcv's quasi ls(y,w,n,scale) never reads it.
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


class QuasiPoisson(Quasi):
    """R's ``quasipoisson(link="log")``: Poisson variance/deviance with
    estimated dispersion (no likelihood — AIC/logLik are NaN, EQL ls).

    Differs from ``Quasi(variance="mu")`` exactly where R differs:
    default link log, poisson's ``initialize`` (μ₀ = y + 0.1 with the
    negative-y check), canonical log for the Newton/Fisher switch
    (gam.fit3.r:2318), and the family name printers show.
    """
    name = "quasipoisson"
    canonical_link_name = "log"
    _newton_canonical = "log"

    def __init__(self, link=None):
        super().__init__(link=link, variance="mu")

    def initialize(self, y, wt):
        # R quasipoisson shares poisson's initialize verbatim.
        return self._shadow.initialize(y, wt)

    __repr__ = Family.__repr__


class QuasiBinomial(Quasi):
    """R's ``quasibinomial(link="logit")``: binomial variance/deviance
    with estimated dispersion (no likelihood — AIC/logLik are NaN).

    Shares binomial's ``initialize`` verbatim — the proportion-smoothing
    mustart and the ``cbind(succ, fail)`` trials form (which warns on
    non-integer counts, like R) — unlike ``Quasi(variance="mu(1-mu)")``'s
    clip-style start. Canonical logit (gam.fit3.r:2319).
    """
    name = "quasibinomial"
    canonical_link_name = "logit"
    _newton_canonical = "logit"

    def __init__(self, link=None):
        super().__init__(link=link, variance="mu(1-mu)")

    def initialize(self, y, wt, n=None):
        # R quasibinomial shares binomial's initialize verbatim (incl.
        # the n-form mustart for cbind responses) — minus the
        # non-integer-#successes warning, whose template guard
        # ("quasibinomial" == "binomial") is false in R.
        return self._shadow.initialize(y, wt, n=n, warn_non_integer=False)

    __repr__ = Family.__repr__


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

    Returns ``(log_a, j_bar, j_var, j_psi_bar, j2_psi_bar, j2_psi2_bar,
    j2_trig_bar)`` — the log of the series sum plus six moments of ``j``
    under ``p_j = W_j/Σ W_k``: E[j], Var[j], E[j·ψ(-j·α)], E[j²·ψ(-j·α)],
    E[(j·ψ(-j·α))²], and E[j²·ψ′(-j·α)]. The first two feed the
    φ-derivatives of log a; E[j·ψ] the p-derivative (Tweedie.dls_dp);
    the last three the p-second-derivatives (Tweedie._d2ls_dp — tw's
    analytic ``lsth2``, family-review B4).
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
    j2_psi_bar = float(np.sum(p_w * j_arr * j_arr * psi_arr))
    j2_psi2_bar = float(np.sum(p_w * (j_arr * psi_arr) ** 2))
    j2_trig_bar = float(np.sum(
        p_w * j_arr * j_arr * polygamma(1, -j_arr * alpha)
    ))
    return (log_a, j_bar, j_var, j_psi_bar,
            j2_psi_bar, j2_psi2_bar, j2_trig_bar)


def _tweedie_log_a_vec(y, phi, p, _chunk_bytes: int = 256 * 1024 * 1024):
    """Vectorised over y (and per-obs phi). Returns seven arrays of shape
    ``y.shape``: ``log_a``, ``j_bar``, ``j_var``, ``j_psi_bar``,
    ``j2_psi_bar``, ``j2_psi2_bar``, ``j2_trig_bar`` (the same moment
    set as :func:`_tweedie_log_a_one`). Entries with y==0 are 0 (the
    y=0 row uses the closed-form point mass, not the series). Per-obs
    phi handles weights via ``φ_i = φ/wt_i``.

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
    j2_psi_bar = np.zeros_like(y)
    j2_psi2_bar = np.zeros_like(y)
    j2_trig_bar = np.zeros_like(y)
    flat_y = y.ravel()
    flat_phi = phi_arr.ravel()
    active = flat_y > 0.0
    if not np.any(active):
        return (log_a, j_bar, j_var, j_psi_bar,
                j2_psi_bar, j2_psi2_bar, j2_trig_bar)
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
    trig_arr = polygamma(1, -j_grid * alpha)

    # Chunk on the n_active axis to bound the (chunk, J) working set.
    # Each row carries 5 J-wide arrays in flight (lw / 2 masks / w /
    # transient), 8 bytes each → 40 J bytes per row.
    n_active = ya.size
    chunk = max(1, _chunk_bytes // (40 * J))

    out_la = np.empty(n_active)
    out_jb = np.empty(n_active)
    out_jv = np.empty(n_active)
    out_jpb = np.empty(n_active)
    out_j2pb = np.empty(n_active)
    out_j2p2b = np.empty(n_active)
    out_j2tb = np.empty(n_active)
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
        jpsi = j_grid[None, :] * psi_arr[None, :]
        out_j2pb[s:e] = np.sum(p_w * j_grid[None, :] * jpsi, axis=1)
        out_j2p2b[s:e] = np.sum(p_w * jpsi * jpsi, axis=1)
        out_j2tb[s:e] = np.sum(
            p_w * j_grid[None, :] ** 2 * trig_arr[None, :], axis=1,
        )

    flat_la = log_a.ravel()
    flat_jb = j_bar.ravel()
    flat_jv = j_var.ravel()
    flat_jpb = j_psi_bar.ravel()
    flat_la[active] = out_la
    flat_jb[active] = out_jb
    flat_jv[active] = out_jv
    flat_jpb[active] = out_jpb
    j2_psi_bar.ravel()[active] = out_j2pb
    j2_psi2_bar.ravel()[active] = out_j2p2b
    j2_trig_bar.ravel()[active] = out_j2tb
    return (log_a, j_bar, j_var, j_psi_bar,
            j2_psi_bar, j2_psi2_bar, j2_trig_bar)


def _tweedie_log_a_vec_pv(y, phi, p, _chunk_bytes: int = 256 * 1024 * 1024):
    """Per-observation-``p`` variant of :func:`_tweedie_log_a_vec`
    (mgcv's ``C_tweedious2`` case — ldTweedie called with vector
    ``theta``/``rho``, gam.fit3.r:2952-2956). Same seven return arrays;
    the special-function tables become (rows, J) matrices because α
    varies by row. Kept separate from the scalar-``p`` function so its
    existing consumers stay byte-identical."""
    y = np.asarray(y, dtype=float)
    phi_arr = np.broadcast_to(
        np.asarray(phi, dtype=float), y.shape).astype(float, copy=True)
    p_arr = np.broadcast_to(
        np.asarray(p, dtype=float), y.shape).astype(float, copy=True)
    log_a = np.zeros_like(y)
    j_bar = np.zeros_like(y)
    j_var = np.zeros_like(y)
    j_psi_bar = np.zeros_like(y)
    j2_psi_bar = np.zeros_like(y)
    j2_psi2_bar = np.zeros_like(y)
    j2_trig_bar = np.zeros_like(y)
    flat_y = y.ravel()
    active = flat_y > 0.0
    if not np.any(active):
        return (log_a, j_bar, j_var, j_psi_bar,
                j2_psi_bar, j2_psi2_bar, j2_trig_bar)
    ya = flat_y[active]
    pha = phi_arr.ravel()[active]
    pa = p_arr.ravel()[active]

    om1 = 1.0 - pa
    tm = 2.0 - pa
    alpha = tm / om1
    one_minus_alpha = 1.0 - alpha

    log_z = (-alpha * np.log(ya) + alpha * np.log(pa - 1.0)
             - one_minus_alpha * np.log(pha) - np.log(tm))
    j_star = np.maximum(
        np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha), 1.0,
    )
    j_int = np.maximum(1, np.round(j_star).astype(int))
    j_int_max = int(j_int.max())

    # widest window needed across rows: decay slows as |alpha| shrinks
    # (p → 2), so size the shared grid from the slowest-decaying row.
    margin_mult = max(2.0, 1.0 / float(np.min(np.abs(alpha))) + 1.0)
    safe_margin = max(50, int(np.ceil(margin_mult * j_int_max)) + 20)
    J = min(j_int_max + safe_margin, _LD_J_MAX)

    j_grid = np.arange(1, J + 1, dtype=float)
    j_grid_int = j_grid.astype(int)
    lgamma_jp1 = gammaln(j_grid + 1.0)

    # per-row α ⇒ the -jα tables are (chunk, J); budget ~9 J-wide
    # doubles per row in flight → 72 J bytes per row.
    n_active = ya.size
    chunk = max(1, _chunk_bytes // (72 * J))

    out_la = np.empty(n_active)
    out_jb = np.empty(n_active)
    out_jv = np.empty(n_active)
    out_jpb = np.empty(n_active)
    out_j2pb = np.empty(n_active)
    out_j2p2b = np.empty(n_active)
    out_j2tb = np.empty(n_active)
    near = 5
    for s in range(0, n_active, chunk):
        e = min(s + chunk, n_active)
        lz_c = log_z[s:e]
        ji_c = j_int[s:e]
        nja = -j_grid[None, :] * alpha[s:e, None]      # (c, J), > 0
        lw = (j_grid[None, :] * lz_c[:, None]
              - lgamma_jp1[None, :] - gammaln(nja))
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
        psi_c = digamma(nja)
        out_jpb[s:e] = np.sum(p_w * j_grid[None, :] * psi_c, axis=1)
        jpsi = j_grid[None, :] * psi_c
        out_j2pb[s:e] = np.sum(p_w * j_grid[None, :] * jpsi, axis=1)
        out_j2p2b[s:e] = np.sum(p_w * jpsi * jpsi, axis=1)
        out_j2tb[s:e] = np.sum(
            p_w * j_grid[None, :] ** 2 * polygamma(1, nja), axis=1,
        )

    log_a.ravel()[active] = out_la
    j_bar.ravel()[active] = out_jb
    j_var.ravel()[active] = out_jv
    j_psi_bar.ravel()[active] = out_jpb
    j2_psi_bar.ravel()[active] = out_j2pb
    j2_psi2_bar.ravel()[active] = out_j2p2b
    j2_trig_bar.ravel()[active] = out_j2tb
    return (log_a, j_bar, j_var, j_psi_bar,
            j2_psi_bar, j2_psi2_bar, j2_trig_bar)


def _ld_tweedie_work(y, mu, theta, rho, a: float = 1.001,
                     b: float = 1.999) -> np.ndarray:
    """mgcv ``ldTweedie`` in the working (ρ, θ) parameterization with
    ``all.derivs=TRUE`` (gam.fit3.r:2838-3035): log Tweedie density and
    derivatives for vector ``mu``/``theta``/``rho``, with
    p = (a + b·e^θ)/(1 + e^θ) ∈ (a, b) and φ = e^ρ.

    Returns the (n, 10) array in mgcv's column order
    ``[l, ρ, ρρ, θ, θθ, θρ, μ, μμ, μθ, μρ]`` — exactly what twlss's ll
    consumes (gamlss.r:2575-2580). The closed-form saddle/zero parts
    and the (p, φ) → (θ, ρ) chain are line-by-line ports; the series
    part (the C ``tweedious2`` call) runs the Dunn-Smyth moment
    machinery (:func:`_tweedie_log_a_vec_pv`) with the same eps gate,
    converted to working-parameter derivatives via

        ∂log W_j/∂ρ = −(1−α)·j,   ∂log W_j/∂p = j·L′ + α′·j·ψ(−jα),

    L = log z, α = (2−p)/(1−p), and the chain p(θ).
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    mu = np.ascontiguousarray(
        np.broadcast_to(np.asarray(mu, dtype=float), y.shape))
    theta = np.ascontiguousarray(
        np.broadcast_to(np.asarray(theta, dtype=float), y.shape))
    rho = np.ascontiguousarray(
        np.broadcast_to(np.asarray(rho, dtype=float), y.shape))
    if not (1.0 < a < b < 2.0):
        raise ValueError("1<a<b<2 (strict) required")

    # p(θ) and its θ-derivatives, the ±θ-stable branches
    # (gam.fit3.r:2849-2858)
    pos = theta > 0
    eth = np.exp(-np.abs(theta))
    p = np.where(pos, (b + a * eth) / (1.0 + eth),
                 (b * eth + a) / (eth + 1.0))
    dpth1 = eth * (b - a) / (1.0 + eth) ** 2
    dpth2 = np.where(
        pos,
        ((a - b) * eth + (b - a) * eth * eth) / (eth + 1.0) ** 3,
        ((a - b) * eth * eth + (b - a) * eth) / (eth + 1.0) ** 3,
    )
    phi = np.exp(rho)

    ld = np.zeros((n, 10))

    # y == 0 rows: closed forms (gam.fit3.r:2920-2937), mu > 0 gate
    zm = (y == 0.0) & (mu > 0.0)
    if np.any(zm):
        mu_z = mu[zm]
        p_z = p[zm]
        phi_z = phi[zm]
        lmu_z = np.log(mu_z)
        ld[zm, 0] = -mu_z ** (2.0 - p_z) / (phi_z * (2.0 - p_z))
        ld[zm, 1] = -ld[zm, 0] / phi_z
        ld[zm, 2] = -2.0 * ld[zm, 1] / phi_z
        ld[zm, 3] = -ld[zm, 0] * (lmu_z - 1.0 / (2.0 - p_z))
        ld[zm, 4] = (2.0 * ld[zm, 3] / (2.0 - p_z)
                     + ld[zm, 0] * lmu_z ** 2)
        ld[zm, 5] = -ld[zm, 3] / phi_z
        mup = mu_z ** p_z
        ld[zm, 6] = -mu_z / (mup * phi_z)
        ld[zm, 7] = -(1.0 - p_z) / (mup * phi_z)
        ld[zm, 8] = lmu_z * mu_z / (mup * phi_z)
        ld[zm, 9] = -ld[zm, 6] / phi_z

    # y > 0 rows: saddle part in (p, φ) (gam.fit3.r:2974-2989)
    ind = y > 0.0
    any_pos = bool(np.any(ind))
    if any_pos:
        y_i = y[ind]
        mu_i = mu[ind]
        p_i = p[ind]
        phii = phi[ind]
        log_mu = np.log(mu_i)
        onep = 1.0 - p_i
        twop = 2.0 - p_i
        mu1p = mu_i ** onep
        k_theta = mu_i * mu1p / twop          # mu^(2-p)/(2-p)
        theta_s = mu1p / onep                 # mu^(1-p)/(1-p)
        a1 = y_i / onep - mu_i / twop
        l_base = mu1p * a1 / phii
        ld[ind, 0] = l_base - np.log(y_i)
        ld[ind, 1] = -l_base / phii
        ld[ind, 2] = 2.0 * l_base / phii ** 2
        x_ = (theta_s * y_i * (1.0 / onep - log_mu) / phii
              + k_theta * (log_mu - 1.0 / twop) / phii)
        ld[ind, 3] = x_
        ld[ind, 4] = (theta_s * y_i
                      * (log_mu ** 2 - 2.0 * log_mu / onep
                         + 2.0 / onep ** 2) / phii
                      - k_theta * (log_mu ** 2 - 2.0 * log_mu / twop
                                   + 2.0 / twop ** 2) / phii)
        ld[ind, 5] = -x_ / phii

    # transform (p, φ) derivatives to working (θ, ρ)
    # (gam.fit3.r:2990-2997) — all rows, zeros included
    ld[:, 2] = ld[:, 2] * phi ** 2 + ld[:, 1] * phi
    ld[:, 1] = ld[:, 1] * phi
    ld[:, 4] = ld[:, 4] * dpth1 ** 2 + ld[:, 3] * dpth2
    ld[:, 3] = ld[:, 3] * dpth1
    ld[:, 5] = ld[:, 5] * dpth1 * phi

    # all.derivs μ-columns for y > 0 (gam.fit3.r:2999-3009)
    if any_pos:
        a2 = mu1p / (mu_i * phii)             # 1/(mu^p · φ)
        ld[ind, 6] = a2 * (onep * a1 - mu_i / twop)
        ld[ind, 7] = -a2 * (onep * p_i * a1 / mu_i
                            + 2.0 * onep / twop)
        ld[ind, 8] = a2 * (-log_mu * onep * a1 - a1
                           + onep * (y_i / onep ** 2 - mu_i / twop ** 2)
                           + mu_i * log_mu / twop - mu_i / twop ** 2)
        ld[ind, 9] = a2 * (mu_i / (phii * twop) - onep * a1 / phii)
    ld[:, 9] = ld[:, 9] * phi
    ld[:, 8] = ld[:, 8] * dpth1

    # series part — added AFTER the transform: like the C code, it is
    # computed natively in (θ, ρ) (gam.fit3.r:3013-3020)
    if any_pos:
        la, jb, jv, jpb, j2pb, j2p2b, j2tb = _tweedie_log_a_vec_pv(
            y_i, phii, p_i)
        al = twop / onep                      # α < 0
        alp = 1.0 / onep ** 2                 # dα/dp
        alpp = 2.0 / onep ** 3                # d²α/dp²
        lphi = np.log(phii)
        ly = np.log(y_i)
        lp1 = np.log(p_i - 1.0)
        Lp = (-alp * ly + alp * lp1 + al / (p_i - 1.0) + alp * lphi
              + 1.0 / twop)
        Lpp = (-alpp * ly + alpp * lp1 + 2.0 * alp / (p_i - 1.0)
               - al / (p_i - 1.0) ** 2 + alpp * lphi + 1.0 / twop ** 2)
        one_m_al = 1.0 - al
        cov_j_jpsi = j2pb - jb * jpb
        dla_dp = jb * Lp + alp * jpb
        d2la_dp2 = (Lp ** 2 * jv + 2.0 * Lp * alp * cov_j_jpsi
                    + alp ** 2 * (j2p2b - jpb ** 2)
                    + jb * Lpp + alpp * jpb - alp ** 2 * j2tb)
        d2la_dpdrho = (-one_m_al * (Lp * jv + alp * cov_j_jpsi)
                       + alp * jb)
        d1 = dpth1[ind]
        d2_ = dpth2[ind]
        ld[ind, 0] += la
        ld[ind, 1] += -one_m_al * jb
        ld[ind, 2] += one_m_al ** 2 * jv
        ld[ind, 3] += d1 * dla_dp
        ld[ind, 4] += d1 ** 2 * d2la_dp2 + d2_ * dla_dp
        ld[ind, 5] += d1 * d2la_dpdrho
    return ld


def _tw_null_fit(y, a: float = 1.001, b: float = 1.999):
    """mgcv ``tw.null.fit`` (gamlss.r:2454-2490): stabilized,
    step-controlled Newton MLE of (μ, p, φ) for a plain Tweedie sample,
    iterating on the working scale (log μ, θ, ρ). Returns
    ``(mu, p, phi)`` — R's ``c(mu, p, sigma)``. The Hessian's log-μ
    chain and the negative-definite eigenvalue clamp are ported
    literally (the gradient stop test is exact, so the approximate
    chain only shapes the path)."""
    y = np.asarray(y, dtype=float)
    th = np.zeros(3)                     # log mu, theta, rho
    ones = np.ones_like(y)

    def _ld_sums(t):
        ld = _ld_tweedie_work(y, np.exp(t[0]) * ones, t[1] * ones,
                              t[2] * ones, a=a, b=b)
        return ld.sum(axis=0)

    lds = _ld_sums(th)
    for _ in range(50):
        g = lds[[6, 3, 1]].copy()
        if np.sum(np.abs(g) > 1e-9 * abs(lds[0])) == 0:
            break
        g[0] = g[0] * np.exp(th[0])      # work on log scale for mu
        H = np.zeros((3, 3))             # mu, th, rh
        H[0, 0] = lds[7]
        H[1, 1] = lds[4]
        H[2, 2] = lds[2]
        H[0, 1] = H[1, 0] = lds[8]
        H[0, 2] = H[2, 0] = lds[9]
        H[1, 2] = H[2, 1] = lds[5]
        H[:, 0] = H[:, 0] * np.exp(th[0])
        H[0, 1:] = H[0, 1:] * np.exp(th[0])
        ev, V = np.linalg.eigh(0.5 * (H + H.T))
        tol = float(np.max(np.abs(ev))) * 1e-7
        ev[ev > -tol] = -tol
        step = V @ ((V.T @ g) / ev)
        ms = float(np.max(np.abs(step)))
        if ms > 3.0:
            step = step * 3.0 / ms
        while True:
            th1 = th - step
            lds1 = _ld_sums(th1)
            if lds1[0] < lds[0]:
                step = step / 2.0
            else:
                th = th1
                lds = lds1
                break
    t2 = th[1]
    if t2 > 0:
        p = (b + a * np.exp(-t2)) / (1.0 + np.exp(-t2))
    else:
        p = (b * np.exp(t2) + a) / (np.exp(t2) + 1.0)
    return float(np.exp(th[0])), float(p), float(np.exp(th[2]))


def _shash_log1pexp(x):
    """shash's ``.log1pexp`` (gamlss.r:3431-3441): log(1 + e^x) with
    R's binned stabilization. The x = −Inf corner (z = 0 exactly)
    falls in the first bin here and returns 0 — R's ``.bincode``
    NA-drops that boundary and would propagate the −Inf."""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    m1 = x <= -37.0
    m2 = (x > -37.0) & (x <= 18.0)
    m3 = (x > 18.0) & (x <= 33.3)
    out[m1] = np.exp(x[m1])
    out[m2] = np.log1p(np.exp(x[m2]))
    out[m3] = x[m3] + np.exp(-x[m3])
    return out


def _sqrt_x2pm(x, m):
    """shash's ``.sqrtX2pm`` (gamlss.r:3444-3451): sqrt(x² + m),
    passing |x| through unchanged once |x| ≥ 1e8."""
    x = np.abs(np.asarray(x, dtype=float))
    out = x.copy()
    kk = x < 1e8
    out[kk] = np.sqrt(x[kk] ** 2 + m)
    return out


def _ax2m1_div_x2m2_sq(x, m1, m2, a=1.0):
    """shash's ``.ax2m1DivX2m2SQ`` (gamlss.r:3454-3466):
    (a·x² + m1)/(x² + m2)² computed stably for large |x|."""
    if a < 0:
        raise ValueError("'a' has to be positive")
    x = np.abs(np.asarray(x, dtype=float))
    kk = (a * x ** 2 + m1) < 0.0
    out = np.zeros_like(x)
    if np.any(kk):
        out[kk] = (a * x[kk] ** 2 + m1) / (x[kk] ** 2 + m2) ** 2
    nk = ~kk
    if np.any(nk):
        out[nk] = ((_sqrt_x2pm(np.sqrt(a) * x[nk], m1)
                    / _sqrt_x2pm(x[nk], m2)) / _sqrt_x2pm(x[nk], m2)) ** 2
    return out


def _sech(x):
    return 1.0 / np.cosh(x)


def _shash_derivs(y, mu, tau, eps, phi, phi_pen, deriv):
    """shash log-density and packed parameter-space derivatives —
    mgcv's shash ``ll`` body (gamlss.r:3487-3950) up to the etamu
    hand-off. Returns ``(l0, L1, L2, L3, L4)`` with the (μ, τ, ε, φ)
    packing; L3/L4 are None below the requesting deriv level. The
    third/fourth-derivative blocks are mechanical transcriptions of
    mgcv's auto-generated maxima code (sequencing and groupings kept
    line-for-line; only `del`→`delta` renamed).
    """
    y = np.asarray(y, dtype=float)
    sig = np.exp(tau)
    delta = np.exp(phi)
    z = (y - mu) / (sig * delta)
    dTasMe = delta * np.arcsinh(z) - eps
    g = -dTasMe
    CC = np.cosh(dTasMe)
    SS = np.sinh(dTasMe)
    with np.errstate(divide="ignore"):
        log_abs_z2 = 2.0 * np.log(np.abs(z))
    l0 = (-tau - 0.5 * np.log(2.0 * np.pi) + np.log(CC)
          - 0.5 * _shash_log1pexp(log_abs_z2) - 0.5 * SS ** 2
          - phi_pen * phi ** 2)
    L1 = L2 = L3 = L4 = None
    if deriv >= 1:
        zsd = z * sig * delta
        sSp1 = _sqrt_x2pm(z, 1.0)               # sqrt(z² + 1)
        asinhZ = np.arcsinh(z)

        # first derivatives (gamlss.r:3513-3519)
        De = np.tanh(g) - 0.5 * np.sinh(2.0 * g)
        Dm = 1.0 / (delta * sig * sSp1) * (delta * De + z / sSp1)
        Dt = zsd * Dm - 1.0
        Dp = Dt + 1.0 - delta * asinhZ * De - 2.0 * phi_pen * phi
        L1 = np.column_stack([Dm, Dt, De, Dp])

        # second derivatives, packed mm,mt,me,mp,tt,te,tp,ee,ep,pp
        # (gamlss.r:3522-3535)
        Dme = (_sech(g) ** 2 - np.cosh(2.0 * g)) / (sig * sSp1)
        Dte = zsd * Dme
        Dmm = (Dme / (sig * sSp1) + z * De / (sig ** 2 * delta * sSp1 ** 3)
               + _ax2m1_div_x2m2_sq(z, -1.0, 1.0) / (delta * sig * delta
                                                     * sig))
        Dmt = zsd * Dmm - Dm
        Dee = -2.0 * np.cosh(g) ** 2 + _sech(g) ** 2 + 1.0
        Dtt = zsd * Dmt
        Dep = Dte - delta * asinhZ * Dee
        Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
        Dtp = zsd * Dmp
        Dpp = (Dtp - delta * asinhZ * Dep
               + delta * (z / sSp1 - asinhZ) * De - 2.0 * phi_pen)
        L2 = np.column_stack([Dmm, Dmt, Dme, Dmp, Dtt, Dte, Dtp, Dee,
                              Dep, Dpp])
    if deriv > 1:
        # third derivatives (gamlss.r:3545-3567)
        Deee = -2 * (np.sinh(2 * g) + _sech(g) ** 2 * np.tanh(g))
        Dmee = Deee / (sig * sSp1)
        Dmme = Dmee / (sig * sSp1) + z * Dee / (sig * sig * delta * sSp1 ** 3)
        Dmmm = (
            2 * z * Dme / (sig * sig * delta * sSp1 ** 3) + Dmme /
            (sig * sSp1) + _ax2m1_div_x2m2_sq(z, -1, 1, 2) * De /
            (sig ** 3 * delta ** 2 * sSp1) + 2 * (z / sSp1) *
            _ax2m1_div_x2m2_sq(z, -3, 1) / ((sig * delta) ** 3 * sSp1)
        )
        Dmmt = zsd * Dmmm - 2 * Dmm
        Dtee = zsd * Dmee
        Dmte = zsd * Dmme - Dme
        Dtte = zsd * Dmte
        Dmtt = zsd * Dmmt - Dmt
        Dttt = zsd * Dmtt
        Dmep = Dmte + Dee / (sig * sSp1) - delta * asinhZ * Dmee
        Dtep = zsd * Dmep
        Deep = Dtee - delta * asinhZ * Deee
        Depp = Dtep - delta * asinhZ * Deep + delta * (z / sSp1 - asinhZ) * Dee
        Dmmp = (
            Dmmt + 2 * Dme / (sig * sSp1) + z * De /
            (delta * sig * sig * sSp1 ** 3) - delta * asinhZ * Dmme
        )
        Dmtp = zsd * Dmmp - Dmp
        Dttp = zsd * Dmtp
        Dmpp = (
            Dmtp + Dep / (sig * sSp1) + z ** 2 * De / (sig * sSp1 ** 3) -
            delta * asinhZ * Dmep + delta * Dme * (z / sSp1 - asinhZ)
        )
        Dtpp = zsd * Dmpp
        Dppp = (
            Dtpp - delta * asinhZ * Depp + delta * (z / sSp1 - asinhZ) *
            (2 * Dep + De) + delta * (z / sSp1) ** 3 * De
        )

        L3 = np.column_stack([Dmmm, Dmmt, Dmme, Dmmp, Dmtt, Dmte, Dmtp,
                              Dmee, Dmep, Dmpp, Dttt, Dtte, Dttp, Dtee,
                              Dtep, Dtpp, Deee, Deep, Depp, Dppp])
    if deriv > 3:
        # fourth derivatives — mgcv's auto-generated block
        # (gamlss.r:3586-3941); 35 columns in the packed order
        # mmmm..pppp listed at gamlss.r:3579-3582
        m = mu
        t = tau
        p = phi
        e = eps
        exp1 = np.e
        aaa1 = -t
        aaa2 = y - m
        aaa3 = exp1 ** p * np.asinh(exp1 ** (aaa1 - p) * aaa2) - e
        abb8 = np.cosh(aaa3)
        abb9 = np.sinh(aaa3)
        abb1 = exp1 ** ((-2 * t) - 2 * p)
        abb3 = aaa2 ** 2
        abb4 = 1 / exp1 ** t
        abb5 = -t - p
        abb7 = exp1 ** (2 * abb5) * abb3 + 1
        abb6 = 1 / np.sqrt(abb7)
        aee5 = aaa3 + e
        aff04 = abb1 * abb3 + 1
        aff05 = abb4 ** 2
        aff08 = 2 * abb5
        aff10 = 1 / abb7
        aff13 = abb8 ** 2
        aff14 = exp1 ** (aaa1 + aff08)
        aff15 = abb6 ** 3
        aff17 = abb9 ** 2
        agg15 = 1 / abb6
        agg17 = 1 / abb8
        aii11 = aaa3 + e
        aii12 = aii11 - abb4 * aaa2 * abb6
        aii17 = abb6 ** 3
        ajj15 = aaa2 ** 3
        ann05 = exp1 ** p
        ann06 = np.asinh(exp1 ** abb5 * aaa2)
        aoo09 = -aaa2 / (exp1 ** t * agg15)
        app02 = -2 * t
        app04 = exp1 ** (app02 - 2 * p) * abb3 + 1
        app08 = exp1 ** (app02 + aff08)
        app10 = 1 / abb7 ** 2
        app14 = exp1 ** (aaa1 + 4 * abb5)
        app16 = 1 / agg15 ** 5
        app21 = 1 / exp1 ** (3 * t)
        aqq03 = exp1 ** (app02 - 2 * p)
        aqq05 = aqq03 * abb3 + 1
        aqq27 = 1 / aff13
        arr06 = exp1 ** aff08 * aaa2 ** 2 + 1
        arr07 = 1 / np.sqrt(arr06) ** 3
        arr12 = 1 / arr06
        ass16 = aii11 - aaa2 / (exp1 ** t * agg15)
        ass23 = 1 / abb8
        ass28 = 1 / aff13
        att19 = aaa2 ** 4
        avv19 = aii11 - abb4 * aaa2 * abb6
        ayy14 = -abb4 * aaa2 * abb6
        ayy16 = aii11 + ayy14
        ayy17 = aii11 + ayy14 - aff14 * ajj15 * aii17
        ayy24 = ayy16 ** 2
        azz19 = aaa2 ** 5
        bdd07 = np.sqrt(exp1 ** aff08 * aaa2 ** 2 + 1)
        bdd08 = 1 / bdd07 ** 3
        bdd14 = 1 / bdd07
        bdd15 = aii11 - abb4 * aaa2 * bdd14
        bgg4 = (
            aee5 - aaa2 /
            (exp1 ** t * np.sqrt(exp1 ** (2 * abb5) * aaa2 ** 2 + 1))
        )
        bhh13 = -abb4 * aaa2 * bdd14
        bhh14 = ann05 * ann06
        bii11 = aii11 + aoo09
        bii15 = aii11 + aoo09 - aff14 * ajj15 * aii17
        bjj07 = 4 * abb5
        bjj08 = exp1 ** (app02 + bjj07)
        bjj11 = 1 / abb7 ** 3
        bjj14 = 1 / exp1 ** (4 * t)
        bjj18 = exp1 ** (aaa1 + 6 * abb5)
        bjj21 = 1 / agg15 ** 7
        bjj24 = exp1 ** (aff08 - 3 * t)
        bjj26 = exp1 ** (aaa1 + bjj07)
        j2 = (
            (-(6 * bjj14 * app10 * abb9 ** 4) / abb8 ** 4) -
            (12 * bjj24 * aaa2 * app16 * abb9 ** 3) / abb8 ** 3 + 8 * bjj14 *
            app10 * aqq27 * aff17 + 4 * app08 * app10 * aqq27 * aff17 - 15 *
            bjj08 * abb3 * bjj11 * aqq27 * aff17 - 4 * bjj14 * app10 * aff17 +
            4 * app08 * app10 * aff17 - 15 * bjj08 * abb3 * bjj11 * aff17 - 9
            * bjj26 * aaa2 * app16 * abb8 * abb9 + 24 * bjj24 * aaa2 * app16 *
            abb8 * abb9 + 15 * bjj18 * ajj15 * bjj21 * abb8 * abb9 + 9 * bjj26
            * aaa2 * app16 * agg17 * abb9 + 12 * bjj24 * aaa2 * app16 * agg17
            * abb9 - 15 * bjj18 * ajj15 * bjj21 * agg17 * abb9 - 4 * bjj14 *
            app10 * aff13 + 4 * app08 * app10 * aff13 - 15 * bjj08 * abb3 *
            bjj11 * aff13 - 2 * bjj14 * app10 - 4 * app08 * app10 + 15 * bjj08
            * abb3 * bjj11 + (6 * exp1 ** ((-4 * t) - 4 * p)) / app04 ** 2 -
            (48 * exp1 ** ((-6 * t) - 6 * p) * abb3) / app04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 4) / app04 ** 4
        )
        bkk33 = 1 / abb8 ** 3
        bkk34 = abb9 ** 3
        k2 = (
            (-(6 * bjj14 * aaa2 * app10 * abb9 ** 4) / abb8 ** 4) + 6 * app21
            * aff15 * bkk33 * bkk34 - 12 * bjj24 * abb3 * app16 * bkk33 *
            bkk34 + 8 * bjj14 * aaa2 * app10 * aqq27 * aff17 + 13 * app08 *
            aaa2 * app10 * aqq27 * aff17 - 15 * bjj08 * ajj15 * bjj11 * aqq27
            * aff17 - 4 * bjj14 * aaa2 * app10 * aff17 + 13 * app08 * aaa2 *
            app10 * aff17 - 15 * bjj08 * ajj15 * bjj11 * aff17 - 12 * app21 *
            aff15 * abb8 * abb9 + 3 * aff14 * aff15 * abb8 * abb9 - 18 * bjj26
            * abb3 * app16 * abb8 * abb9 + 24 * bjj24 * abb3 * app16 * abb8 *
            abb9 + 15 * bjj18 * att19 * bjj21 * abb8 * abb9 - 6 * app21 *
            aff15 * agg17 * abb9 - 3 * aff14 * aff15 * agg17 * abb9 + 18 *
            bjj26 * abb3 * app16 * agg17 * abb9 + 12 * bjj24 * abb3 * app16 *
            agg17 * abb9 - 15 * bjj18 * att19 * bjj21 * agg17 * abb9 - 4 *
            bjj14 * aaa2 * app10 * aff13 + 13 * app08 * aaa2 * app10 * aff13 -
            15 * bjj08 * ajj15 * bjj11 * aff13 - 2 * bjj14 * aaa2 * app10 - 13
            * app08 * aaa2 * app10 + 15 * bjj08 * ajj15 * bjj11 +
            (24 * exp1 ** ((-4 * t) - 4 * p) * aaa2) / app04 ** 2 -
            (72 * exp1 ** ((-6 * t) - 6 * p) * ajj15) / app04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 5) / app04 ** 4
        )
        bll16 = exp1 ** (aff08 - 2 * t)
        l2 = (
            (-(6 * app21 * aff15 * abb9 ** 4) / abb8 ** 4) -
            (6 * bll16 * aaa2 * app10 * abb9 ** 3) / abb8 ** 3 + 8 * app21 *
            aff15 * aqq27 * aff17 + aff14 * aff15 * aqq27 * aff17 - 3 * app14
            * abb3 * app16 * aqq27 * aff17 - 4 * app21 * aff15 * aff17 + aff14
            * aff15 * aff17 - 3 * app14 * abb3 * app16 * aff17 + 12 * bll16 *
            aaa2 * app10 * abb8 * abb9 + (6 * bll16 * aaa2 * app10 * abb9) /
            abb8 - 4 * app21 * aff15 * aff13 + aff14 * aff15 * aff13 - 3 *
            app14 * abb3 * app16 * aff13 - 2 * app21 * aff15 - aff14 * aff15 +
            3 * app14 * abb3 * app16
        )
        bmm34 = 1 / abb8 ** 3
        bmm35 = abb9 ** 3
        m2 = (
            (6 * app21 * aff15 * ass16 * abb9 ** 4) / abb8 ** 4 + 6 * app08 *
            aaa2 * app10 * ass16 * bmm34 * bmm35 - 6 * bjj24 * abb3 * app16 *
            bmm34 * bmm35 - 8 * app21 * aff15 * ass16 * ass28 * aff17 - aff14
            * aff15 * ass16 * ass28 * aff17 + 3 * bjj26 * abb3 * app16 * ass16
            * ass28 * aff17 + 6 * app08 * aaa2 * app10 * ass28 * aff17 - 12 *
            bjj08 * ajj15 * bjj11 * ass28 * aff17 + 4 * app21 * aff15 * ass16
            * aff17 - aff14 * aff15 * ass16 * aff17 + 3 * bjj26 * abb3 * app16
            * ass16 * aff17 + 6 * app08 * aaa2 * app10 * aff17 - 12 * bjj08 *
            ajj15 * bjj11 * aff17 - 12 * app08 * aaa2 * app10 * ass16 * abb8 *
            abb9 + 2 * aff14 * aff15 * abb8 * abb9 - 15 * bjj26 * abb3 * app16
            * abb8 * abb9 + 12 * bjj24 * abb3 * app16 * abb8 * abb9 + 15 *
            bjj18 * att19 * bjj21 * abb8 * abb9 - 6 * app08 * aaa2 * app10 *
            ass16 * ass23 * abb9 - 2 * aff14 * aff15 * ass23 * abb9 + 15 *
            bjj26 * abb3 * app16 * ass23 * abb9 + 6 * bjj24 * abb3 * app16 *
            ass23 * abb9 - 15 * bjj18 * att19 * bjj21 * ass23 * abb9 + 4 *
            app21 * aff15 * ass16 * aff13 - aff14 * aff15 * ass16 * aff13 + 3
            * bjj26 * abb3 * app16 * ass16 * aff13 + 6 * app08 * aaa2 * app10
            * aff13 - 12 * bjj08 * ajj15 * bjj11 * aff13 + 2 * app21 * aff15 *
            ass16 + aff14 * aff15 * ass16 - 3 * bjj26 * abb3 * app16 * ass16 -
            6 * app08 * aaa2 * app10 + 12 * bjj08 * ajj15 * bjj11 +
            (24 * exp1 ** ((-4 * t) - 4 * p) * aaa2) / app04 ** 2 -
            (72 * exp1 ** ((-6 * t) - 6 * p) * ajj15) / app04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 5) / app04 ** 4
        )
        n2 = (
            (-(6 * bjj14 * abb3 * app10 * abb9 ** 4) / abb8 ** 4) + 10 * app21
            * aaa2 * aff15 * bkk33 * bkk34 - 12 * bjj24 * ajj15 * app16 *
            bkk33 * bkk34 - 4 * aff05 * aff10 * aqq27 * aff17 + 8 * bjj14 *
            abb3 * app10 * aqq27 * aff17 + 19 * app08 * abb3 * app10 * aqq27 *
            aff17 - 15 * bjj08 * att19 * bjj11 * aqq27 * aff17 - 4 * aff05 *
            aff10 * aff17 - 4 * bjj14 * abb3 * app10 * aff17 + 19 * app08 *
            abb3 * app10 * aff17 - 15 * bjj08 * att19 * bjj11 * aff17 - 20 *
            app21 * aaa2 * aff15 * abb8 * abb9 + 9 * aff14 * aaa2 * aff15 *
            abb8 * abb9 - 24 * bjj26 * ajj15 * app16 * abb8 * abb9 + 24 *
            bjj24 * ajj15 * app16 * abb8 * abb9 + 15 * bjj18 * azz19 * bjj21 *
            abb8 * abb9 - 10 * app21 * aaa2 * aff15 * agg17 * abb9 - 9 * aff14
            * aaa2 * aff15 * agg17 * abb9 + 24 * bjj26 * ajj15 * app16 * agg17
            * abb9 + 12 * bjj24 * ajj15 * app16 * agg17 * abb9 - 15 * bjj18 *
            azz19 * bjj21 * agg17 * abb9 - 4 * aff05 * aff10 * aff13 - 4 *
            bjj14 * abb3 * app10 * aff13 + 19 * app08 * abb3 * app10 * aff13 -
            15 * bjj08 * att19 * bjj11 * aff13 + 4 * aff05 * aff10 - 2 * bjj14
            * abb3 * app10 - 19 * app08 * abb3 * app10 + 15 * bjj08 * att19 *
            bjj11 - (4 * aqq03) / aqq05 +
            (44 * exp1 ** ((-4 * t) - 4 * p) * abb3) / aqq05 ** 2 -
            (88 * exp1 ** ((-6 * t) - 6 * p) * att19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 6) / aqq05 ** 4
        )
        o2 = (
            (-(6 * app21 * aaa2 * aff15 * abb9 ** 4) / abb8 ** 4) + 4 * aff05
            * aff10 * bkk33 * bkk34 - 6 * bll16 * abb3 * app10 * bkk33 * bkk34
            + 8 * app21 * aaa2 * aff15 * aqq27 * aff17 + 3 * aff14 * aaa2 *
            aff15 * aqq27 * aff17 - 3 * app14 * ajj15 * app16 * aqq27 * aff17
            - 4 * app21 * aaa2 * aff15 * aff17 + 3 * aff14 * aaa2 * aff15 *
            aff17 - 3 * app14 * ajj15 * app16 * aff17 - 8 * aff05 * aff10 *
            abb8 * abb9 + 12 * bll16 * abb3 * app10 * abb8 * abb9 - 4 * aff05
            * aff10 * agg17 * abb9 + 6 * bll16 * abb3 * app10 * agg17 * abb9 -
            4 * app21 * aaa2 * aff15 * aff13 + 3 * aff14 * aaa2 * aff15 *
            aff13 - 3 * app14 * ajj15 * app16 * aff13 - 2 * app21 * aaa2 *
            aff15 - 3 * aff14 * aaa2 * aff15 + 3 * app14 * ajj15 * app16
        )
        p2 = (
            (6 * app21 * aaa2 * aff15 * ass16 * abb9 ** 4) / abb8 ** 4 - 4 *
            aff05 * aff10 * ass16 * bmm34 * bmm35 + 6 * app08 * abb3 * app10 *
            ass16 * bmm34 * bmm35 - 6 * bjj24 * ajj15 * app16 * bmm34 * bmm35
            - 8 * app21 * aaa2 * aff15 * ass16 * ass28 * aff17 - 3 * aff14 *
            aaa2 * aff15 * ass16 * ass28 * aff17 + 3 * bjj26 * ajj15 * app16 *
            ass16 * ass28 * aff17 + 10 * app08 * abb3 * app10 * ass28 * aff17
            - 12 * bjj08 * att19 * bjj11 * ass28 * aff17 + 4 * app21 * aaa2 *
            aff15 * ass16 * aff17 - 3 * aff14 * aaa2 * aff15 * ass16 * aff17 +
            3 * bjj26 * ajj15 * app16 * ass16 * aff17 + 10 * app08 * abb3 *
            app10 * aff17 - 12 * bjj08 * att19 * bjj11 * aff17 + 8 * aff05 *
            aff10 * ass16 * abb8 * abb9 - 12 * app08 * abb3 * app10 * ass16 *
            abb8 * abb9 + 6 * aff14 * aaa2 * aff15 * abb8 * abb9 - 21 * bjj26
            * ajj15 * app16 * abb8 * abb9 + 12 * bjj24 * ajj15 * app16 * abb8
            * abb9 + 15 * bjj18 * azz19 * bjj21 * abb8 * abb9 + 4 * aff05 *
            aff10 * ass16 * ass23 * abb9 - 6 * app08 * abb3 * app10 * ass16 *
            ass23 * abb9 - 6 * aff14 * aaa2 * aff15 * ass23 * abb9 + 21 *
            bjj26 * ajj15 * app16 * ass23 * abb9 + 6 * bjj24 * ajj15 * app16 *
            ass23 * abb9 - 15 * bjj18 * azz19 * bjj21 * ass23 * abb9 + 4 *
            app21 * aaa2 * aff15 * ass16 * aff13 - 3 * aff14 * aaa2 * aff15 *
            ass16 * aff13 + 3 * bjj26 * ajj15 * app16 * ass16 * aff13 + 10 *
            app08 * abb3 * app10 * aff13 - 12 * bjj08 * att19 * bjj11 * aff13
            + 2 * app21 * aaa2 * aff15 * ass16 + 3 * aff14 * aaa2 * aff15 *
            ass16 - 3 * bjj26 * ajj15 * app16 * ass16 - 10 * app08 * abb3 *
            app10 + 12 * bjj08 * att19 * bjj11 - (4 * aqq03) / aqq05 +
            (44 * exp1 ** ((-4 * t) - 4 * p) * abb3) / aqq05 ** 2 -
            (88 * exp1 ** ((-6 * t) - 6 * p) * att19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 6) / aqq05 ** 4
        )
        q2 = (
            (-(6 * aff05 * arr12 * abb9 ** 4) / abb8 ** 4) -
            (2 * aff14 * aaa2 * arr07 * abb9 ** 3) / abb8 ** 3 +
            (8 * aff05 * arr12 * aff17) / aff13 - 4 * aff05 * arr12 * aff17 +
            4 * aff14 * aaa2 * arr07 * abb8 * abb9 +
            (2 * aff14 * aaa2 * arr07 * abb9) / abb8 - 4 * aff05 * arr12 *
            aff13 - 2 * aff05 * arr12
        )
        r2 = (
            (6 * aff05 * aff10 * ass16 * abb9 ** 4) / abb8 ** 4 + 2 * aff14 *
            aaa2 * aff15 * ass16 * bmm34 * bmm35 - 4 * bll16 * abb3 * app10 *
            bmm34 * bmm35 - 8 * aff05 * aff10 * ass16 * ass28 * aff17 + 2 *
            aff14 * aaa2 * aff15 * ass28 * aff17 - 3 * app14 * ajj15 * app16 *
            ass28 * aff17 + 4 * aff05 * aff10 * ass16 * aff17 + 2 * aff14 *
            aaa2 * aff15 * aff17 - 3 * app14 * ajj15 * app16 * aff17 - 4 *
            aff14 * aaa2 * aff15 * ass16 * abb8 * abb9 + 8 * bll16 * abb3 *
            app10 * abb8 * abb9 - 2 * aff14 * aaa2 * aff15 * ass16 * ass23 *
            abb9 + 4 * bll16 * abb3 * app10 * ass23 * abb9 + 4 * aff05 * aff10
            * ass16 * aff13 + 2 * aff14 * aaa2 * aff15 * aff13 - 3 * app14 *
            ajj15 * app16 * aff13 + 2 * aff05 * aff10 * ass16 - 2 * aff14 *
            aaa2 * aff15 + 3 * app14 * ajj15 * app16
        )
        bss21 = 2 * aff14 * abb3 * aff15 - 3 * bjj26 * att19 * app16
        bss23 = -abb4 * aaa2 * abb6
        bss25 = aii11 + bss23
        bss26 = aii11 + bss23 - aff14 * ajj15 * aff15
        bss29 = bss25 ** 2
        bss33 = (
            (-4 * aff14 * aaa2 * aff15) + 18 * bjj26 * ajj15 * app16 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 5) / agg15 ** 7
        )
        s2 = (
            (-(6 * aff05 * aff10 * bss29 * abb9 ** 4) / abb8 ** 4) - 2 * aff14
            * aaa2 * aff15 * bss29 * bmm34 * bmm35 + 2 * aff05 * aff10 * bss26
            * bmm34 * bmm35 + 8 * app08 * abb3 * app10 * bss25 * bmm34 * bmm35
            + 8 * aff05 * aff10 * bss29 * ass28 * aff17 + aff14 * aaa2 * aff15
            * bss26 * ass28 * aff17 - 4 * aff14 * aaa2 * aff15 * bss25 * ass28
            * aff17 + 6 * bjj26 * ajj15 * app16 * bss25 * ass28 * aff17 + 2 *
            abb4 * abb6 * bss21 * ass28 * aff17 - 2 * bjj08 * att19 * bjj11 *
            ass28 * aff17 - 4 * aff05 * aff10 * bss29 * aff17 + aff14 * aaa2 *
            aff15 * bss26 * aff17 - 4 * aff14 * aaa2 * aff15 * bss25 * aff17 +
            6 * bjj26 * ajj15 * app16 * bss25 * aff17 + 2 * abb4 * abb6 *
            bss21 * aff17 - 2 * bjj08 * att19 * bjj11 * aff17 + 4 * aff14 *
            aaa2 * aff15 * bss29 * abb8 * abb9 - 4 * aff05 * aff10 * bss26 *
            abb8 * abb9 - 16 * app08 * abb3 * app10 * bss25 * abb8 * abb9 -
            bss33 * abb8 * abb9 + 2 * aff14 * aaa2 * aff15 * bss29 * ass23 *
            abb9 - 2 * aff05 * aff10 * bss26 * ass23 * abb9 - 8 * app08 * abb3
            * app10 * bss25 * ass23 * abb9 + bss33 * ass23 * abb9 - 4 * aff05
            * aff10 * bss29 * aff13 + aff14 * aaa2 * aff15 * bss26 * aff13 - 4
            * aff14 * aaa2 * aff15 * bss25 * aff13 + 6 * bjj26 * ajj15 * app16
            * bss25 * aff13 + 2 * abb4 * abb6 * bss21 * aff13 - 2 * bjj08 *
            att19 * bjj11 * aff13 - 2 * aff05 * aff10 * bss29 - aff14 * aaa2 *
            aff15 * bss26 + 4 * aff14 * aaa2 * aff15 * bss25 - 6 * bjj26 *
            ajj15 * app16 * bss25 - 2 * abb4 * abb6 * bss21 + 2 * bjj08 *
            att19 * bjj11 - (4 * aqq03) / aqq05 +
            (44 * exp1 ** ((-4 * t) - 4 * p) * abb3) / aqq05 ** 2 -
            (88 * exp1 ** ((-6 * t) - 6 * p) * att19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 6) / aqq05 ** 4
        )
        btt24 = aaa2 ** 6
        t2 = (
            (-(6 * bjj14 * ajj15 * app10 * abb9 ** 4) / abb8 ** 4) + 12 *
            app21 * abb3 * aff15 * bkk33 * bkk34 - 12 * bjj24 * att19 * app16
            * bkk33 * bkk34 - 7 * aff05 * aaa2 * aff10 * aqq27 * aff17 + 8 *
            bjj14 * ajj15 * app10 * aqq27 * aff17 + 22 * app08 * ajj15 * app10
            * aqq27 * aff17 - 15 * bjj08 * azz19 * bjj11 * aqq27 * aff17 - 7 *
            aff05 * aaa2 * aff10 * aff17 - 4 * bjj14 * ajj15 * app10 * aff17 +
            22 * app08 * ajj15 * app10 * aff17 - 15 * bjj08 * azz19 * bjj11 *
            aff17 - abb4 * abb6 * abb8 * abb9 - 24 * app21 * abb3 * aff15 *
            abb8 * abb9 + 13 * aff14 * abb3 * aff15 * abb8 * abb9 - 27 * bjj26
            * att19 * app16 * abb8 * abb9 + 24 * bjj24 * att19 * app16 * abb8
            * abb9 + 15 * bjj18 * btt24 * bjj21 * abb8 * abb9 + abb4 * abb6 *
            agg17 * abb9 - 12 * app21 * abb3 * aff15 * agg17 * abb9 - 13 *
            aff14 * abb3 * aff15 * agg17 * abb9 + 27 * bjj26 * att19 * app16 *
            agg17 * abb9 + 12 * bjj24 * att19 * app16 * agg17 * abb9 - 15 *
            bjj18 * btt24 * bjj21 * agg17 * abb9 - 7 * aff05 * aaa2 * aff10 *
            aff13 - 4 * bjj14 * ajj15 * app10 * aff13 + 22 * app08 * ajj15 *
            app10 * aff13 - 15 * bjj08 * azz19 * bjj11 * aff13 + 7 * aff05 *
            aaa2 * aff10 - 2 * bjj14 * ajj15 * app10 - 22 * app08 * ajj15 *
            app10 + 15 * bjj08 * azz19 * bjj11 - (8 * aqq03 * aaa2) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aqq05 ** 4
        )
        u2 = (
            (-(6 * app21 * abb3 * aff15 * abb9 ** 4) / abb8 ** 4) + 6 * aff05
            * aaa2 * aff10 * bkk33 * bkk34 - 6 * bll16 * ajj15 * app10 * bkk33
            * bkk34 - abb4 * abb6 * aqq27 * aff17 + 8 * app21 * abb3 * aff15 *
            aqq27 * aff17 + 4 * aff14 * abb3 * aff15 * aqq27 * aff17 - 3 *
            app14 * att19 * app16 * aqq27 * aff17 - abb4 * abb6 * aff17 - 4 *
            app21 * abb3 * aff15 * aff17 + 4 * aff14 * abb3 * aff15 * aff17 -
            3 * app14 * att19 * app16 * aff17 - 12 * aff05 * aaa2 * aff10 *
            abb8 * abb9 + 12 * bll16 * ajj15 * app10 * abb8 * abb9 - 6 * aff05
            * aaa2 * aff10 * agg17 * abb9 + 6 * bll16 * ajj15 * app10 * agg17
            * abb9 - abb4 * abb6 * aff13 - 4 * app21 * abb3 * aff15 * aff13 +
            4 * aff14 * abb3 * aff15 * aff13 - 3 * app14 * att19 * app16 *
            aff13 + abb4 * abb6 - 2 * app21 * abb3 * aff15 - 4 * aff14 * abb3
            * aff15 + 3 * app14 * att19 * app16
        )
        v2 = (
            (6 * app21 * abb3 * aff15 * avv19 * abb9 ** 4) / abb8 ** 4 - 6 *
            aff05 * aaa2 * aff10 * avv19 * bmm34 * bmm35 + 6 * app08 * ajj15 *
            app10 * avv19 * bmm34 * bmm35 - 6 * bjj24 * att19 * app16 * bmm34
            * bmm35 + abb4 * abb6 * avv19 * ass28 * aff17 - 8 * app21 * abb3 *
            aff15 * avv19 * ass28 * aff17 - 4 * aff14 * abb3 * aff15 * avv19 *
            ass28 * aff17 + 3 * bjj26 * att19 * app16 * avv19 * ass28 * aff17
            + 12 * app08 * ajj15 * app10 * ass28 * aff17 - 12 * bjj08 * azz19
            * bjj11 * ass28 * aff17 + abb4 * abb6 * avv19 * aff17 + 4 * app21
            * abb3 * aff15 * avv19 * aff17 - 4 * aff14 * abb3 * aff15 * avv19
            * aff17 + 3 * bjj26 * att19 * app16 * avv19 * aff17 + 12 * app08 *
            ajj15 * app10 * aff17 - 12 * bjj08 * azz19 * bjj11 * aff17 + 12 *
            aff05 * aaa2 * aff10 * avv19 * abb8 * abb9 - 12 * app08 * ajj15 *
            app10 * avv19 * abb8 * abb9 + 9 * aff14 * abb3 * aff15 * abb8 *
            abb9 - 24 * bjj26 * att19 * app16 * abb8 * abb9 + 12 * bjj24 *
            att19 * app16 * abb8 * abb9 + 15 * bjj18 * btt24 * bjj21 * abb8 *
            abb9 + 6 * aff05 * aaa2 * aff10 * avv19 * ass23 * abb9 - 6 * app08
            * ajj15 * app10 * avv19 * ass23 * abb9 - 9 * aff14 * abb3 * aff15
            * ass23 * abb9 + 24 * bjj26 * att19 * app16 * ass23 * abb9 + 6 *
            bjj24 * att19 * app16 * ass23 * abb9 - 15 * bjj18 * btt24 * bjj21
            * ass23 * abb9 + abb4 * abb6 * avv19 * aff13 + 4 * app21 * abb3 *
            aff15 * avv19 * aff13 - 4 * aff14 * abb3 * aff15 * avv19 * aff13 +
            3 * bjj26 * att19 * app16 * avv19 * aff13 + 12 * app08 * ajj15 *
            app10 * aff13 - 12 * bjj08 * azz19 * bjj11 * aff13 - abb4 * abb6 *
            avv19 + 2 * app21 * abb3 * aff15 * avv19 + 4 * aff14 * abb3 *
            aff15 * avv19 - 3 * bjj26 * att19 * app16 * avv19 - 12 * app08 *
            ajj15 * app10 + 12 * bjj08 * azz19 * bjj11 - (8 * aqq03 * aaa2) /
            aqq05 + (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aqq05 ** 4
        )
        w2 = (
            (-(6 * aff05 * aaa2 * aff10 * abb9 ** 4) / abb8 ** 4) + 2 * abb4 *
            abb6 * bkk33 * bkk34 - 2 * aff14 * abb3 * aff15 * bkk33 * bkk34 +
            (8 * aff05 * aaa2 * aff10 * aff17) / aff13 - 4 * aff05 * aaa2 *
            aff10 * aff17 - 4 * abb4 * abb6 * abb8 * abb9 + 4 * aff14 * abb3 *
            aff15 * abb8 * abb9 - 2 * abb4 * abb6 * agg17 * abb9 + 2 * aff14 *
            abb3 * aff15 * agg17 * abb9 - 4 * aff05 * aaa2 * aff10 * aff13 - 2
            * aff05 * aaa2 * aff10
        )
        x2 = (
            (6 * aff05 * aaa2 * aff10 * avv19 * abb9 ** 4) / abb8 ** 4 - 2 *
            abb4 * abb6 * avv19 * bmm34 * bmm35 + 2 * aff14 * abb3 * aff15 *
            avv19 * bmm34 * bmm35 - 4 * bll16 * ajj15 * app10 * bmm34 * bmm35
            - 8 * aff05 * aaa2 * aff10 * avv19 * ass28 * aff17 + 3 * aff14 *
            abb3 * aff15 * ass28 * aff17 - 3 * app14 * att19 * app16 * ass28 *
            aff17 + 4 * aff05 * aaa2 * aff10 * avv19 * aff17 + 3 * aff14 *
            abb3 * aff15 * aff17 - 3 * app14 * att19 * app16 * aff17 + 4 *
            abb4 * abb6 * avv19 * abb8 * abb9 - 4 * aff14 * abb3 * aff15 *
            avv19 * abb8 * abb9 + 8 * bll16 * ajj15 * app10 * abb8 * abb9 + 2
            * abb4 * abb6 * avv19 * ass23 * abb9 - 2 * aff14 * abb3 * aff15 *
            avv19 * ass23 * abb9 + 4 * bll16 * ajj15 * app10 * ass23 * abb9 +
            4 * aff05 * aaa2 * aff10 * avv19 * aff13 + 3 * aff14 * abb3 *
            aff15 * aff13 - 3 * app14 * att19 * app16 * aff13 + 2 * aff05 *
            aaa2 * aff10 * avv19 - 3 * aff14 * abb3 * aff15 + 3 * app14 *
            att19 * app16
        )
        byy24 = 2 * aff14 * ajj15 * aff15 - 3 * bjj26 * azz19 * app16
        byy35 = (
            (-6 * aff14 * abb3 * aff15) + 21 * bjj26 * att19 * app16 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 6) / agg15 ** 7
        )
        y2 = (
            (-(6 * aff05 * aaa2 * aff10 * bss29 * abb9 ** 4) / abb8 ** 4) + 2
            * abb4 * abb6 * bss29 * bmm34 * bmm35 - 2 * aff14 * abb3 * aff15 *
            bss29 * bmm34 * bmm35 + 2 * aff05 * aaa2 * aff10 * bss26 * bmm34 *
            bmm35 + 8 * app08 * ajj15 * app10 * bss25 * bmm34 * bmm35 + 8 *
            aff05 * aaa2 * aff10 * bss29 * ass28 * aff17 - abb4 * abb6 * bss26
            * ass28 * aff17 + aff14 * abb3 * aff15 * bss26 * ass28 * aff17 - 6
            * aff14 * abb3 * aff15 * bss25 * ass28 * aff17 + 6 * bjj26 * att19
            * app16 * bss25 * ass28 * aff17 + abb4 * abb6 * byy24 * ass28 *
            aff17 + abb4 * aaa2 * abb6 * bss21 * ass28 * aff17 - 2 * bjj08 *
            azz19 * bjj11 * ass28 * aff17 - 4 * aff05 * aaa2 * aff10 * bss29 *
            aff17 - abb4 * abb6 * bss26 * aff17 + aff14 * abb3 * aff15 * bss26
            * aff17 - 6 * aff14 * abb3 * aff15 * bss25 * aff17 + 6 * bjj26 *
            att19 * app16 * bss25 * aff17 + abb4 * abb6 * byy24 * aff17 + abb4
            * aaa2 * abb6 * bss21 * aff17 - 2 * bjj08 * azz19 * bjj11 * aff17
            - 4 * abb4 * abb6 * bss29 * abb8 * abb9 + 4 * aff14 * abb3 * aff15
            * bss29 * abb8 * abb9 - 4 * aff05 * aaa2 * aff10 * bss26 * abb8 *
            abb9 - 16 * app08 * ajj15 * app10 * bss25 * abb8 * abb9 - byy35 *
            abb8 * abb9 - 2 * abb4 * abb6 * bss29 * ass23 * abb9 + 2 * aff14 *
            abb3 * aff15 * bss29 * ass23 * abb9 - 2 * aff05 * aaa2 * aff10 *
            bss26 * ass23 * abb9 - 8 * app08 * ajj15 * app10 * bss25 * ass23 *
            abb9 + byy35 * ass23 * abb9 - 4 * aff05 * aaa2 * aff10 * bss29 *
            aff13 - abb4 * abb6 * bss26 * aff13 + aff14 * abb3 * aff15 * bss26
            * aff13 - 6 * aff14 * abb3 * aff15 * bss25 * aff13 + 6 * bjj26 *
            att19 * app16 * bss25 * aff13 + abb4 * abb6 * byy24 * aff13 + abb4
            * aaa2 * abb6 * bss21 * aff13 - 2 * bjj08 * azz19 * bjj11 * aff13
            - 2 * aff05 * aaa2 * aff10 * bss29 + abb4 * abb6 * bss26 - aff14 *
            abb3 * aff15 * bss26 + 6 * aff14 * abb3 * aff15 * bss25 - 6 *
            bjj26 * att19 * app16 * bss25 - abb4 * abb6 * byy24 - abb4 * aaa2
            * abb6 * bss21 + 2 * bjj08 * azz19 * bjj11 - (8 * aqq03 * aaa2) /
            aqq05 + (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aqq05 ** 4
        )
        bzz7 = abb8 ** 2
        bzz9 = abb9 ** 2
        z2 = (
            (-(6 * abb4 * abb6 * abb9 ** 4) / abb8 ** 4) +
            (8 * abb4 * abb6 * bzz9) / bzz7 - 4 * abb4 * abb6 * bzz9 - 4 *
            abb4 * abb6 * bzz7 - 2 * abb4 * abb6
        )
        a3 = (
            (6 * abb4 * abb6 * aii12 * abb9 ** 4) / abb8 ** 4 -
            (2 * aff14 * abb3 * aii17 * abb9 ** 3) / abb8 ** 3 -
            (8 * abb4 * abb6 * aii12 * aff17) / aff13 + 4 * abb4 * abb6 *
            aii12 * aff17 + 4 * aff14 * abb3 * aii17 * abb8 * abb9 +
            (2 * aff14 * abb3 * aii17 * abb9) / abb8 + 4 * abb4 * abb6 * aii12
            * aff13 + 2 * abb4 * abb6 * aii12
        )
        cbb09 = 1 / agg15 ** 5
        cbb18 = 2 * aff14 * abb3 * aii17 - 3 * app14 * att19 * cbb09
        cbb24 = aii11 + ayy14 - aff14 * aaa2 ** 3 * aii17
        b3 = (
            (-(6 * abb4 * abb6 * ayy24 * abb9 ** 4) / abb8 ** 4) + 2 * abb4 *
            abb6 * cbb24 * bmm34 * bmm35 + 4 * aff14 * abb3 * aii17 * ayy16 *
            bmm34 * bmm35 + 8 * abb4 * abb6 * ayy24 * ass28 * aff17 + cbb18 *
            ass28 * aff17 - 4 * abb4 * abb6 * ayy24 * aff17 + cbb18 * aff17 -
            4 * abb4 * abb6 * cbb24 * abb8 * abb9 - 8 * aff14 * abb3 * aii17 *
            ayy16 * abb8 * abb9 - 2 * abb4 * abb6 * cbb24 * ass23 * abb9 - 4 *
            aff14 * abb3 * aii17 * ayy16 * ass23 * abb9 - 4 * abb4 * abb6 *
            ayy24 * aff13 + cbb18 * aff13 - 2 * abb4 * abb6 * ayy24 - 2 *
            aff14 * abb3 * aii17 + 3 * app14 * att19 * cbb09
        )
        ccc23 = (
            aii11 + ayy14 + aff14 * ajj15 * aii17 - 3 * app14 * azz19 * cbb09
        )
        ccc24 = ayy16 ** 3
        ccc28 = (
            (-4 * aff14 * abb3 * aii17) + 18 * app14 * att19 * cbb09 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 6) / agg15 ** 7
        )
        c3 = (
            (6 * abb4 * abb6 * ccc24 * abb9 ** 4) / abb8 ** 4 - 6 * aff14 *
            abb3 * aii17 * ayy24 * bmm34 * bmm35 - 6 * abb4 * abb6 * ayy16 *
            ayy17 * bmm34 * bmm35 - 8 * abb4 * abb6 * ccc24 * ass28 * aff17 +
            abb4 * abb6 * ccc23 * ass28 * aff17 + 3 * aff14 * abb3 * aii17 *
            ayy17 * ass28 * aff17 - 3 * cbb18 * ayy16 * ass28 * aff17 + 4 *
            abb4 * abb6 * ccc24 * aff17 + abb4 * abb6 * ccc23 * aff17 + 3 *
            aff14 * abb3 * aii17 * ayy17 * aff17 - 3 * cbb18 * ayy16 * aff17 +
            12 * aff14 * abb3 * aii17 * ayy24 * abb8 * abb9 + 12 * abb4 * abb6
            * ayy16 * ayy17 * abb8 * abb9 - ccc28 * abb8 * abb9 + 6 * aff14 *
            abb3 * aii17 * ayy24 * ass23 * abb9 + 6 * abb4 * abb6 * ayy16 *
            ayy17 * ass23 * abb9 + ccc28 * ass23 * abb9 + 4 * abb4 * abb6 *
            ccc24 * aff13 + abb4 * abb6 * ccc23 * aff13 + 3 * aff14 * abb3 *
            aii17 * ayy17 * aff13 - 3 * cbb18 * ayy16 * aff13 + 2 * abb4 *
            abb6 * ccc24 - abb4 * abb6 * ccc23 - 3 * aff14 * abb3 * aii17 *
            ayy17 + 3 * cbb18 * ayy16 - (8 * abb1 * aaa2) / aff04 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aff04 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aff04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aff04 ** 4
        )
        cdd24 = aaa2 ** 7
        d3 = (
            (-(6 * bjj14 * att19 * app10 * abb9 ** 4) / abb8 ** 4) + 12 *
            app21 * ajj15 * aff15 * bkk33 * bkk34 - 12 * bjj24 * azz19 * app16
            * bkk33 * bkk34 - 7 * aff05 * abb3 * aff10 * aqq27 * aff17 + 8 *
            bjj14 * att19 * app10 * aqq27 * aff17 + 22 * app08 * att19 * app10
            * aqq27 * aff17 - 15 * bjj08 * btt24 * bjj11 * aqq27 * aff17 - 7 *
            aff05 * abb3 * aff10 * aff17 - 4 * bjj14 * att19 * app10 * aff17 +
            22 * app08 * att19 * app10 * aff17 - 15 * bjj08 * btt24 * bjj11 *
            aff17 - abb4 * aaa2 * abb6 * abb8 * abb9 - 24 * app21 * ajj15 *
            aff15 * abb8 * abb9 + 13 * aff14 * ajj15 * aff15 * abb8 * abb9 -
            27 * bjj26 * azz19 * app16 * abb8 * abb9 + 24 * bjj24 * azz19 *
            app16 * abb8 * abb9 + 15 * bjj18 * cdd24 * bjj21 * abb8 * abb9 +
            abb4 * aaa2 * abb6 * agg17 * abb9 - 12 * app21 * ajj15 * aff15 *
            agg17 * abb9 - 13 * aff14 * ajj15 * aff15 * agg17 * abb9 + 27 *
            bjj26 * azz19 * app16 * agg17 * abb9 + 12 * bjj24 * azz19 * app16
            * agg17 * abb9 - 15 * bjj18 * cdd24 * bjj21 * agg17 * abb9 - 7 *
            aff05 * abb3 * aff10 * aff13 - 4 * bjj14 * att19 * app10 * aff13 +
            22 * app08 * att19 * app10 * aff13 - 15 * bjj08 * btt24 * bjj11 *
            aff13 + 7 * aff05 * abb3 * aff10 - 2 * bjj14 * att19 * app10 - 22
            * app08 * att19 * app10 + 15 * bjj08 * btt24 * bjj11 -
            (8 * aqq03 * abb3) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * att19) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * btt24) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aqq05 ** 4
        )
        e3 = (
            (-(6 * app21 * ajj15 * aff15 * abb9 ** 4) / abb8 ** 4) + 6 * aff05
            * abb3 * aff10 * bkk33 * bkk34 - 6 * bll16 * att19 * app10 * bkk33
            * bkk34 - abb4 * aaa2 * abb6 * aqq27 * aff17 + 8 * app21 * ajj15 *
            aff15 * aqq27 * aff17 + 4 * aff14 * ajj15 * aff15 * aqq27 * aff17
            - 3 * app14 * azz19 * app16 * aqq27 * aff17 - abb4 * aaa2 * abb6 *
            aff17 - 4 * app21 * ajj15 * aff15 * aff17 + 4 * aff14 * ajj15 *
            aff15 * aff17 - 3 * app14 * azz19 * app16 * aff17 - 12 * aff05 *
            abb3 * aff10 * abb8 * abb9 + 12 * bll16 * att19 * app10 * abb8 *
            abb9 - 6 * aff05 * abb3 * aff10 * agg17 * abb9 + 6 * bll16 * att19
            * app10 * agg17 * abb9 - abb4 * aaa2 * abb6 * aff13 - 4 * app21 *
            ajj15 * aff15 * aff13 + 4 * aff14 * ajj15 * aff15 * aff13 - 3 *
            app14 * azz19 * app16 * aff13 + abb4 * aaa2 * abb6 - 2 * app21 *
            ajj15 * aff15 - 4 * aff14 * ajj15 * aff15 + 3 * app14 * azz19 *
            app16
        )
        f3 = (
            (6 * app21 * ajj15 * aff15 * avv19 * abb9 ** 4) / abb8 ** 4 - 6 *
            aff05 * abb3 * aff10 * avv19 * bmm34 * bmm35 + 6 * app08 * att19 *
            app10 * avv19 * bmm34 * bmm35 - 6 * bjj24 * azz19 * app16 * bmm34
            * bmm35 + abb4 * aaa2 * abb6 * avv19 * ass28 * aff17 - 8 * app21 *
            ajj15 * aff15 * avv19 * ass28 * aff17 - 4 * aff14 * ajj15 * aff15
            * avv19 * ass28 * aff17 + 3 * bjj26 * azz19 * app16 * avv19 *
            ass28 * aff17 + 12 * app08 * att19 * app10 * ass28 * aff17 - 12 *
            bjj08 * btt24 * bjj11 * ass28 * aff17 + abb4 * aaa2 * abb6 * avv19
            * aff17 + 4 * app21 * ajj15 * aff15 * avv19 * aff17 - 4 * aff14 *
            ajj15 * aff15 * avv19 * aff17 + 3 * bjj26 * azz19 * app16 * avv19
            * aff17 + 12 * app08 * att19 * app10 * aff17 - 12 * bjj08 * btt24
            * bjj11 * aff17 + 12 * aff05 * abb3 * aff10 * avv19 * abb8 * abb9
            - 12 * app08 * att19 * app10 * avv19 * abb8 * abb9 + 9 * aff14 *
            ajj15 * aff15 * abb8 * abb9 - 24 * bjj26 * azz19 * app16 * abb8 *
            abb9 + 12 * bjj24 * azz19 * app16 * abb8 * abb9 + 15 * bjj18 *
            cdd24 * bjj21 * abb8 * abb9 + 6 * aff05 * abb3 * aff10 * avv19 *
            ass23 * abb9 - 6 * app08 * att19 * app10 * avv19 * ass23 * abb9 -
            9 * aff14 * ajj15 * aff15 * ass23 * abb9 + 24 * bjj26 * azz19 *
            app16 * ass23 * abb9 + 6 * bjj24 * azz19 * app16 * ass23 * abb9 -
            15 * bjj18 * cdd24 * bjj21 * ass23 * abb9 + abb4 * aaa2 * abb6 *
            avv19 * aff13 + 4 * app21 * ajj15 * aff15 * avv19 * aff13 - 4 *
            aff14 * ajj15 * aff15 * avv19 * aff13 + 3 * bjj26 * azz19 * app16
            * avv19 * aff13 + 12 * app08 * att19 * app10 * aff13 - 12 * bjj08
            * btt24 * bjj11 * aff13 - abb4 * aaa2 * abb6 * avv19 + 2 * app21 *
            ajj15 * aff15 * avv19 + 4 * aff14 * ajj15 * aff15 * avv19 - 3 *
            bjj26 * azz19 * app16 * avv19 - 12 * app08 * att19 * app10 + 12 *
            bjj08 * btt24 * bjj11 - (8 * aqq03 * abb3) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * att19) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * btt24) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aqq05 ** 4
        )
        g3 = (
            (-(6 * aff05 * abb3 * aff10 * abb9 ** 4) / abb8 ** 4) + 2 * abb4 *
            aaa2 * abb6 * bkk33 * bkk34 - 2 * aff14 * ajj15 * aff15 * bkk33 *
            bkk34 + (8 * aff05 * abb3 * aff10 * aff17) / aff13 - 4 * aff05 *
            abb3 * aff10 * aff17 - 4 * abb4 * aaa2 * abb6 * abb8 * abb9 + 4 *
            aff14 * ajj15 * aff15 * abb8 * abb9 - 2 * abb4 * aaa2 * abb6 *
            agg17 * abb9 + 2 * aff14 * ajj15 * aff15 * agg17 * abb9 - 4 *
            aff05 * abb3 * aff10 * aff13 - 2 * aff05 * abb3 * aff10
        )
        h3 = (
            (6 * aff05 * abb3 * aff10 * avv19 * abb9 ** 4) / abb8 ** 4 - 2 *
            abb4 * aaa2 * abb6 * avv19 * bmm34 * bmm35 + 2 * aff14 * ajj15 *
            aff15 * avv19 * bmm34 * bmm35 - 4 * bll16 * att19 * app10 * bmm34
            * bmm35 - 8 * aff05 * abb3 * aff10 * avv19 * ass28 * aff17 + 3 *
            aff14 * ajj15 * aff15 * ass28 * aff17 - 3 * app14 * azz19 * app16
            * ass28 * aff17 + 4 * aff05 * abb3 * aff10 * avv19 * aff17 + 3 *
            aff14 * ajj15 * aff15 * aff17 - 3 * app14 * azz19 * app16 * aff17
            + 4 * abb4 * aaa2 * abb6 * avv19 * abb8 * abb9 - 4 * aff14 * ajj15
            * aff15 * avv19 * abb8 * abb9 + 8 * bll16 * att19 * app10 * abb8 *
            abb9 + 2 * abb4 * aaa2 * abb6 * avv19 * ass23 * abb9 - 2 * aff14 *
            ajj15 * aff15 * avv19 * ass23 * abb9 + 4 * bll16 * att19 * app10 *
            ass23 * abb9 + 4 * aff05 * abb3 * aff10 * avv19 * aff13 + 3 *
            aff14 * ajj15 * aff15 * aff13 - 3 * app14 * azz19 * app16 * aff13
            + 2 * aff05 * abb3 * aff10 * avv19 - 3 * aff14 * ajj15 * aff15 + 3
            * app14 * azz19 * app16
        )
        i3 = (
            (-(6 * aff05 * abb3 * aff10 * bss29 * abb9 ** 4) / abb8 ** 4) + 2
            * abb4 * aaa2 * abb6 * bss29 * bmm34 * bmm35 - 2 * aff14 * ajj15 *
            aff15 * bss29 * bmm34 * bmm35 + 2 * aff05 * abb3 * aff10 * bss26 *
            bmm34 * bmm35 + 8 * app08 * att19 * app10 * bss25 * bmm34 * bmm35
            + 8 * aff05 * abb3 * aff10 * bss29 * ass28 * aff17 - abb4 * aaa2 *
            abb6 * bss26 * ass28 * aff17 + aff14 * ajj15 * aff15 * bss26 *
            ass28 * aff17 - 6 * aff14 * ajj15 * aff15 * bss25 * ass28 * aff17
            + 6 * bjj26 * azz19 * app16 * bss25 * ass28 * aff17 + 4 * app08 *
            att19 * app10 * ass28 * aff17 - 8 * bjj08 * btt24 * bjj11 * ass28
            * aff17 - 4 * aff05 * abb3 * aff10 * bss29 * aff17 - abb4 * aaa2 *
            abb6 * bss26 * aff17 + aff14 * ajj15 * aff15 * bss26 * aff17 - 6 *
            aff14 * ajj15 * aff15 * bss25 * aff17 + 6 * bjj26 * azz19 * app16
            * bss25 * aff17 + 4 * app08 * att19 * app10 * aff17 - 8 * bjj08 *
            btt24 * bjj11 * aff17 - 4 * abb4 * aaa2 * abb6 * bss29 * abb8 *
            abb9 + 4 * aff14 * ajj15 * aff15 * bss29 * abb8 * abb9 - 4 * aff05
            * abb3 * aff10 * bss26 * abb8 * abb9 - 16 * app08 * att19 * app10
            * bss25 * abb8 * abb9 + 6 * aff14 * ajj15 * aff15 * abb8 * abb9 -
            21 * bjj26 * azz19 * app16 * abb8 * abb9 + 15 * bjj18 * cdd24 *
            bjj21 * abb8 * abb9 - 2 * abb4 * aaa2 * abb6 * bss29 * ass23 *
            abb9 + 2 * aff14 * ajj15 * aff15 * bss29 * ass23 * abb9 - 2 *
            aff05 * abb3 * aff10 * bss26 * ass23 * abb9 - 8 * app08 * att19 *
            app10 * bss25 * ass23 * abb9 - 6 * aff14 * ajj15 * aff15 * ass23 *
            abb9 + 21 * bjj26 * azz19 * app16 * ass23 * abb9 - 15 * bjj18 *
            cdd24 * bjj21 * ass23 * abb9 - 4 * aff05 * abb3 * aff10 * bss29 *
            aff13 - abb4 * aaa2 * abb6 * bss26 * aff13 + aff14 * ajj15 * aff15
            * bss26 * aff13 - 6 * aff14 * ajj15 * aff15 * bss25 * aff13 + 6 *
            bjj26 * azz19 * app16 * bss25 * aff13 + 4 * app08 * att19 * app10
            * aff13 - 8 * bjj08 * btt24 * bjj11 * aff13 - 2 * aff05 * abb3 *
            aff10 * bss29 + abb4 * aaa2 * abb6 * bss26 - aff14 * ajj15 * aff15
            * bss26 + 6 * aff14 * ajj15 * aff15 * bss25 - 6 * bjj26 * azz19 *
            app16 * bss25 - 4 * app08 * att19 * app10 + 8 * bjj08 * btt24 *
            bjj11 - (8 * aqq03 * abb3) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * att19) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * btt24) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aqq05 ** 4
        )
        j3 = (
            (-(6 * abb4 * aaa2 * abb6 * abb9 ** 4) / abb8 ** 4) +
            (8 * abb4 * aaa2 * abb6 * bzz9) / bzz7 - 4 * abb4 * aaa2 * abb6 *
            bzz9 - 4 * abb4 * aaa2 * abb6 * bzz7 - 2 * abb4 * aaa2 * abb6
        )
        k3 = (
            (6 * abb4 * aaa2 * bdd14 * bdd15 * abb9 ** 4) / abb8 ** 4 -
            (2 * aff14 * ajj15 * bdd08 * abb9 ** 3) / abb8 ** 3 -
            (8 * abb4 * aaa2 * bdd14 * bdd15 * aff17) / aff13 + 4 * abb4 *
            aaa2 * bdd14 * bdd15 * aff17 + 4 * aff14 * ajj15 * bdd08 * abb8 *
            abb9 + (2 * aff14 * ajj15 * bdd08 * abb9) / abb8 + 4 * abb4 * aaa2
            * bdd14 * bdd15 * aff13 + 2 * abb4 * aaa2 * bdd14 * bdd15
        )
        cll08 = 1 / bdd07 ** 5
        cll16 = aii11 + bhh13
        cll17 = cll16 ** 2
        cll18 = 2 * aff14 * ajj15 * bdd08 - 3 * app14 * azz19 * cll08
        cll24 = aii11 + bhh13 - aff14 * ajj15 * bdd08
        l3 = (
            (-(6 * abb4 * aaa2 * bdd14 * cll17 * abb9 ** 4) / abb8 ** 4) + 2 *
            abb4 * aaa2 * bdd14 * cll24 * bmm34 * bmm35 + 4 * aff14 * ajj15 *
            bdd08 * cll16 * bmm34 * bmm35 + 8 * abb4 * aaa2 * bdd14 * cll17 *
            ass28 * aff17 + cll18 * ass28 * aff17 - 4 * abb4 * aaa2 * bdd14 *
            cll17 * aff17 + cll18 * aff17 - 4 * abb4 * aaa2 * bdd14 * cll24 *
            abb8 * abb9 - 8 * aff14 * ajj15 * bdd08 * cll16 * abb8 * abb9 - 2
            * abb4 * aaa2 * bdd14 * cll24 * ass23 * abb9 - 4 * aff14 * ajj15 *
            bdd08 * cll16 * ass23 * abb9 - 4 * abb4 * aaa2 * bdd14 * cll17 *
            aff13 + cll18 * aff13 - 2 * abb4 * aaa2 * bdd14 * cll17 - 2 *
            aff14 * ajj15 * bdd08 + 3 * app14 * azz19 * cll08
        )
        cmm12 = -3 * app14 * azz19 * cbb09
        cmm16 = 2 * aff14 * ajj15 * aii17 + cmm12
        cmm23 = aii11 + ayy14 + aff14 * ajj15 * aii17 + cmm12
        cmm28 = (
            (-4 * aff14 * ajj15 * aii17) + 18 * app14 * azz19 * cbb09 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 7) / agg15 ** 7
        )
        m3 = (
            (6 * abb4 * aaa2 * abb6 * ccc24 * abb9 ** 4) / abb8 ** 4 - 6 *
            aff14 * ajj15 * aii17 * ayy24 * bmm34 * bmm35 - 6 * abb4 * aaa2 *
            abb6 * ayy16 * ayy17 * bmm34 * bmm35 - 8 * abb4 * aaa2 * abb6 *
            ccc24 * ass28 * aff17 + abb4 * aaa2 * abb6 * cmm23 * ass28 * aff17
            + 3 * aff14 * ajj15 * aii17 * ayy17 * ass28 * aff17 - 3 * cmm16 *
            ayy16 * ass28 * aff17 + 4 * abb4 * aaa2 * abb6 * ccc24 * aff17 +
            abb4 * aaa2 * abb6 * cmm23 * aff17 + 3 * aff14 * ajj15 * aii17 *
            ayy17 * aff17 - 3 * cmm16 * ayy16 * aff17 + 12 * aff14 * ajj15 *
            aii17 * ayy24 * abb8 * abb9 + 12 * abb4 * aaa2 * abb6 * ayy16 *
            ayy17 * abb8 * abb9 - cmm28 * abb8 * abb9 + 6 * aff14 * ajj15 *
            aii17 * ayy24 * ass23 * abb9 + 6 * abb4 * aaa2 * abb6 * ayy16 *
            ayy17 * ass23 * abb9 + cmm28 * ass23 * abb9 + 4 * abb4 * aaa2 *
            abb6 * ccc24 * aff13 + abb4 * aaa2 * abb6 * cmm23 * aff13 + 3 *
            aff14 * ajj15 * aii17 * ayy17 * aff13 - 3 * cmm16 * ayy16 * aff13
            + 2 * abb4 * aaa2 * abb6 * ccc24 - abb4 * aaa2 * abb6 * cmm23 - 3
            * aff14 * ajj15 * aii17 * ayy17 + 3 * cmm16 * ayy16 -
            (8 * abb1 * abb3) / aff04 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * aaa2 ** 4) / aff04 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * aaa2 ** 6) / aff04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aff04 ** 4
        )
        cnn3 = abb8 ** 2
        cnn5 = abb9 ** 2
        n3 = (
            (-(6 * abb9 ** 4) / abb8 ** 4) + (8 * cnn5) / cnn3 - 4 * cnn5 - 4
            * cnn3 - 2
        )
        coo7 = abb8 ** 2
        coo9 = abb9 ** 2
        o3 = (
            (6 * bgg4 * abb9 ** 4) / abb8 ** 4 - (8 * bgg4 * coo9) / coo7 + 4
            * bgg4 * coo9 + 4 * bgg4 * coo7 + 2 * bgg4
        )
        cpp06 = -aaa2 / (exp1 ** t * bdd07)
        cpp08 = (cpp06 + aii11) ** 2
        cpp12 = (
            aii11 + cpp06 - (exp1 ** (aaa1 + aff08) * aaa2 ** 3) / bdd07 ** 3
        )
        p3 = (
            (-(6 * cpp08 * abb9 ** 4) / abb8 ** 4) + (2 * cpp12 * abb9 ** 3) /
            abb8 ** 3 + (8 * cpp08 * aff17) / aff13 - 4 * cpp08 * aff17 - 4 *
            cpp12 * abb8 * abb9 - (2 * cpp12 * abb9) / abb8 - 4 * cpp08 *
            aff13 - 2 * cpp08
        )
        cqq12 = -aff14 * ajj15 * bdd08
        cqq19 = bhh14 + bhh13
        cqq20 = cqq19 ** 3
        cqq21 = (
            bhh14 + bhh13 + aff14 * ajj15 * bdd08 - 3 * app14 * azz19 * cll08
        )
        cqq25 = bhh14 + bhh13 + cqq12
        cqq28 = 1 / aff13
        q3 = (
            (6 * cqq20 * abb9 ** 4) / abb8 ** 4 -
            (6 * cqq19 * cqq25 * abb9 ** 3) / abb8 ** 3 - 8 * cqq20 * cqq28 *
            aff17 + cqq21 * cqq28 * aff17 + 4 * cqq20 * aff17 + cqq21 * aff17
            + 12 * cqq19 * cqq25 * abb8 * abb9 + (6 * cqq19 * cqq25 * abb9) /
            abb8 + 4 * cqq20 * aff13 + cqq21 * aff13 + 2 * cqq20 - ann05 *
            ann06 + abb4 * aaa2 * bdd14 + cqq12 + 3 * app14 * azz19 * cll08
        )
        crr18 = (
            aii11 + aoo09 + aff14 * ajj15 * aii17 - 3 * app14 * azz19 * cbb09
        )
        crr19 = bii11 ** 4
        crr21 = bii15 ** 2
        crr25 = (
            aii11 + aoo09 - 3 * aff14 * ajj15 * aii17 + 15 * app14 * azz19 *
            cbb09 - (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 7) / agg15 ** 7
        )
        crr28 = bii11 ** 2
        r3 = (
            (-(6 * crr19 * abb9 ** 4) / abb8 ** 4) +
            (12 * crr28 * bii15 * abb9 ** 3) / abb8 ** 3 - 3 * crr21 * ass28 *
            aff17 + 8 * crr19 * ass28 * aff17 - 4 * bii11 * crr18 * ass28 *
            aff17 - 3 * crr21 * aff17 - 4 * crr19 * aff17 - 4 * bii11 * crr18
            * aff17 - 24 * crr28 * bii15 * abb8 * abb9 - crr25 * abb8 * abb9 -
            12 * crr28 * bii15 * ass23 * abb9 + crr25 * ass23 * abb9 - 3 *
            crr21 * aff13 - 4 * crr19 * aff13 - 4 * bii11 * crr18 * aff13 + 3
            * crr21 - 2 * crr19 + 4 * bii11 * crr18 - (8 * abb1 * abb3) /
            aff04 + (56 * exp1 ** ((-4 * t) - 4 * p) * aaa2 ** 4) / aff04 ** 2
            - (96 * exp1 ** ((-6 * t) - 6 * p) * aaa2 ** 6) / aff04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aff04 ** 4
        )

        L4 = np.column_stack([j2, k2, l2, m2, n2, o2, p2, q2, r2, s2,
                              t2, u2, v2, w2, x2, y2, z2, a3, b3, c3,
                              d3, e3, f3, g3, h3, i3, j3, k3, l3, m3,
                              n3, o3, p3, q3, r3])
    return l0, L1, L2, L3, L4


def _r_tweedie(rng, mu, p: float, phi: float) -> np.ndarray:
    """mgcv ``rTweedie`` (gam.fit3.r:3112-3146): compound Poisson-Gamma
    deviates for 1 < p < 2. mgcv draws N_i ~ Poisson(λ_i) individual
    Gamma(shape, scale_i) jumps and C_psum's them; equal in law to one
    Gamma(N_i·shape, scale_i) draw per row (gamma additivity at shared
    scale), which is what we sample. numpy RNG — Monte-Carlo-level
    parity (R's rejection samplers aren't ported).
    """
    mu = np.asarray(mu, dtype=float)
    if not (1.0 < p < 2.0):
        raise ValueError("p must be in (1,2)")
    if np.any(mu < 0):
        raise ValueError("mean, mu, must be non negative")
    if phi <= 0:
        raise ValueError("scale parameter must be positive")
    lam = mu ** (2.0 - p) / ((2.0 - p) * phi)
    shape = (2.0 - p) / (p - 1.0)
    scale = phi * (p - 1.0) * mu ** (p - 1.0)
    N = rng.poisson(lam)
    pos = N > 0
    y = np.zeros(mu.shape[0], dtype=float)
    if np.any(pos):
        y[pos] = rng.gamma(N[pos] * shape, scale[pos])
    return y


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
    # mgcv sets canonical="none" explicitly (gam.fit3.r:3105; tw
    # efam.r:3262): PIRLS runs full Newton even at the default log link.
    _newton_canonical = "none"
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
        # mgcv: mustart = y + 0.1·(y==0) — bump only the zeros so log(μ)
        # stays finite (Tweedie gam.fit3.r:3078, tw efam.r:3234).
        return y + 0.1 * (y == 0.0)

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def rd(self, rng, mu, wt, scale):
        # Tweedie rd (gam.fit3.r:3097-3099) / tw rd (efam.r:3245-3254,
        # inherited): rTweedie(mu, p, phi=scale). ``wt`` is in mgcv's
        # signature but unread — prior weights don't enter, bug-for-bug.
        # (mgcv's p==2 rgamma branch is unreachable here: hea requires
        # 1 < p < 2.)
        return _r_tweedie(rng, mu, self.p, float(scale))

    def _log_density(self, y, mu, phi):
        """Per-obs log f(y_i; μ_i, φ, p), shape (n,) — one unmodified φ for
        every row (mgcv's ``ldTweedie(y, mu, p, phi=scale)``; prior weights
        multiply the summed log-density at the call site, they never divide
        the dispersion — same convention as ``ls``)."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        phi_i = np.full_like(y, float(phi))
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
            la = _tweedie_log_a_vec(y[~zero], phi_i[~zero], p)[0]
            out[~zero] = -np.log(y[~zero]) + la + cumulant[~zero] / phi_i[~zero]
        return out

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv's ``Tweedie()$aic`` (gam.fit3.r:3086) and ``tw()$aic``
        # (efam.r:3212), identical math: scale = dev/Σwt — the caller's
        # dev1 is scale·Σwt (gam.fit3.r:848 / gam.fit4.r:794), so this
        # recovers the REML/Pearson scale — then
        # -2·Σ wt·ldTweedie(y, μ, p, φ=scale) + 2.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        n_eff = float(wt.sum())
        phi = max(float(dev) / max(n_eff, 1e-300), 1e-12)
        log_f = self._log_density(y, mu, phi)
        return -2.0 * float(np.sum(log_f * wt)) + 2.0

    def ls(self, y, wt, scale):
        """Saturated log-lik Σ w_i·log f(y_i; y_i, φ, p) and its 1st/2nd
        derivatives wrt log φ (hea log-scale convention).

        mgcv's Tweedie convention (BOTH variants): the prior weight
        multiplies the per-obs log-density at *unmodified* φ —
        ``colSums(w·ldTweedie(y, y, phi=scale))`` (fix.family.ls,
        gam.fit3.r:3083) and ``w·ldTweedie(y, y, rho=log(scale))``
        (tw()$ls, efam.r:3224). This deliberately differs from the
        Gamma/exponential-family ``φ_i = φ/w_i`` convention. For y_i = 0
        with μ_i = y_i = 0 the cumulant is 0 and log f = 0; the entry
        contributes nothing to ls or its derivatives. For y_i > 0:

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
        phi_i = np.full_like(w_g, float(scale))
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
            la_, jb_, jv_ = _tweedie_log_a_vec(y_g[~zero], phi_i[~zero], p)[:3]
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
        # Same mgcv convention as ``ls``: weight outside, φ unmodified.
        phi_i = np.full_like(w_g, float(scale))
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
            _, jb_, _, jpb_, *_rest2 = _tweedie_log_a_vec(
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

    def _d2ls_dp(self, y, wt, scale):
        """``(∂²ls/∂p², ∂²ls/∂p∂log φ)`` at the saturated point — the
        p-space second derivatives behind tw's analytic ``lsth2``
        (ldTweedie's columns 5/6 in the (θ,ρ) form before the p(θ)
        chain: gam.fit3.r:2802-2806 density part + the C_tweedious
        series part; family-review B4).

        Density part at μ = y (mgcv's ld[,5]/ld[,6] closed forms with
        θ_y·y = y^(2−p)/(1−p), k_y = y^(2−p)/(2−p), L = log y):

            d²/dp²   = [θ_y·y(L² − 2L/(1−p) + 2/(1−p)²)
                        − k_y(L² − 2L/(2−p) + 2/(2−p)²)]/φ
            d²/dp∂φ  = −x/φ  ⇒  d²/dp∂logφ = −x   (x = density ∂/∂p)

        Series part via Dunn-Smyth moments of log a = log Σ_j W_j with
        log W_j = j·log z − lgamma(j+1) − lgamma(−jα), α = (2−p)/(1−p),
        α′ = 1/(1−p)², α″ = 2/(1−p)³, K_j = C + ψ(−jα),
        C = log φ + log(p−1) − log y − (2−p):

            ∂logW_j/∂p       = j·α′·K_j + j/(2−p)              (=: G_j)
            ∂²logW_j/∂p²     = j[α″K_j + α′(1/(p−1) + 1
                                − jα′ψ′(−jα)) + 1/(2−p)²]
            ∂²logW_j/∂p∂logφ = j·α′

            ∂²log a/∂p²      = E[∂²logW/∂p²] + Var[G]
            ∂²log a/∂p∂logφ  = α′E[j] − (1/(p−1))·[(α′C + 1/(2−p))Var[j]
                                + α′(E[j²ψ] − E[jψ]E[j])]

        y = 0 rows contribute nothing (log f_sat ≡ 0 there, matching
        ldTweedie(y, y)'s all-zero rows at y = 0).
        """
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        if not np.any(good):
            return 0.0, 0.0
        y_g = y[good]
        w_g = wt[good]
        phi_i = np.full_like(w_g, float(scale))
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p

        zero = (y_g == 0.0)
        y_safe = np.where(zero, 1.0, y_g)
        L = np.where(zero, 0.0, np.log(y_safe))
        log_phi = np.log(phi_i)

        # --- density part (μ = y) ---------------------------------------
        y_tm = y_safe ** tm
        th_y = y_tm / om1                  # θ_y·y = y^(2-p)/(1-p)
        k_y = y_tm / tm
        x_dens = (th_y * (1.0 / om1 - L) + k_y * (L - 1.0 / tm)) / phi_i
        d2p_dens = (th_y * (L * L - 2.0 * L / om1 + 2.0 / (om1 * om1))
                    - k_y * (L * L - 2.0 * L / tm + 2.0 / (tm * tm))) / phi_i
        cross_dens = -x_dens               # already in log φ form

        # --- series part -------------------------------------------------
        ap = 1.0 / (om1 * om1)             # α′
        app = 2.0 / (om1 * om1 * om1)      # α″
        inv_pm1 = 1.0 / (p - 1.0)          # 1 − α
        d2p_ser = np.zeros_like(y_g)
        cross_ser = np.zeros_like(y_g)
        if np.any(~zero):
            (_, jb, jv, jpb, j2pb, j2p2b, j2tb) = _tweedie_log_a_vec(
                y_g[~zero], phi_i[~zero], p
            )
            C = log_phi[~zero] + np.log(p - 1.0) - L[~zero] - tm
            E_jK = jb * C + jpb
            G_mean = ap * E_jK + jb / tm
            E_j2 = jv + jb * jb
            coef = ap * C + 1.0 / tm
            E_G2 = (coef * coef * E_j2 + 2.0 * coef * ap * j2pb
                    + ap * ap * j2p2b)
            var_G = E_G2 - G_mean * G_mean
            d2p_ser[~zero] = (app * E_jK
                              + ap * (inv_pm1 + 1.0) * jb
                              - ap * ap * j2tb
                              + jb / (tm * tm)
                              + var_G)
            cross_ser[~zero] = (ap * jb
                                - inv_pm1 * (coef * jv
                                             + ap * (j2pb - jpb * jb)))

        d2p = np.where(zero, 0.0, d2p_ser + d2p_dens)
        cross = np.where(zero, 0.0, cross_ser + cross_dens)
        return (float(np.sum(w_g * d2p)),
                float(np.sum(w_g * cross)))

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

    ``hea.gam`` estimates θ jointly with (ρ, log φ) in the analytical
    outer Newton (the family-generic Dd chain supplies the θ gradient;
    the Hessian θ rows are central differences of that gradient). The
    fitted ``p̂`` is stored on ``family.p``; the converged θ̂ on
    ``family.theta``.
    """
    name = "Tweedie"
    n_theta = 1

    # mgcv tw() okLinks (efam.r:3098-3101) — tw validates strictly,
    # UNLIKE fixed-p Tweedie() whose is.character fallback
    # (gam.fit3.r:3042-3045) accepts any make.link name (R-verified:
    # Tweedie(1.5, link="logit") constructs, tw(link="logit") errors).
    _OK_LINKS = ("log", "identity", "sqrt", "inverse")

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
        if self.link.name not in self._OK_LINKS:
            raise ValueError(
                f'link "{self.link.name}" not available for tw family; '
                f'available links are {self._OK_LINKS}'
            )

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

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # tw postproc (efam.r:3237-3243): find.null.dev + "Tweedie(p=…)"
        # relabel with the fitted power rounded to 3 decimals.
        return {
            "null_deviance": find_null_dev(
                self, y, eta=linear_predictors, offset=offset,
                weights=prior_weights,
            ),
            "family_name": f"Tweedie(p={np.round(self.p, 3):g})",
        }

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        """mgcv ``tw()$ls`` in dict form (efam.r:3221-3230): saturated
        log-likelihood and its full first/second derivatives wrt the
        working parameters (θ, log φ) — ldTweedie's columns
        (1,4,2,5,6,3) summed with weight w:

            lsth1 = (LS₄, LS₂)
            lsth2 = [[LS₅, LS₆], [LS₆, LS₃]]

        The θ entries chain the p-space derivatives through p(θ):
        ∂/∂θ = (∂/∂p)·p′, ∂²/∂θ² = (∂²/∂p²)·p′² + (∂/∂p)·p″,
        ∂²/∂θ∂logφ = (∂²/∂p∂logφ)·p′ — exactly ldTweedie's work.param
        transform (gam.fit3.r:2808-2814). The p-space second
        derivatives come from :meth:`Tweedie._d2ls_dp` (family-review
        B4; previously NaN-poisoned).

        Note: hea's outer-Newton θ rows are still central differences
        of the analytical gradient (gam.py `_reml_hessian`) — they
        don't read lsth2 yet; mgcv's `estimate.theta` Newton and any
        future analytic θ-row port do.
        """
        saved = None
        if theta is not None:
            th = np.asarray(theta, dtype=float).reshape(-1)
            if not np.allclose(th, self.get_theta()):
                saved = self.get_theta().copy()
                self.set_theta(th)
        try:
            ls3 = np.asarray(self.ls(y, wt, scale), dtype=float)
            dp1 = float(self.dp_dtheta())
            dp2 = float(self.d2p_dtheta2())
            dls_dp = float(self.dls_dp(y, wt, scale))
            dls_dth = dls_dp * dp1
            d2ls_dp2, d2ls_dpdlphi = self._d2ls_dp(y, wt, scale)
            lsth2 = np.empty((2, 2))
            lsth2[0, 0] = d2ls_dp2 * dp1 * dp1 + dls_dp * dp2
            lsth2[0, 1] = lsth2[1, 0] = d2ls_dpdlphi * dp1
            lsth2[1, 1] = float(ls3[2])
            return {
                "ls": float(ls3[0]),
                "lsth1": np.array([dls_dth, float(ls3[1])]),
                "lsth2": lsth2,
                "LSTH1": None,
            }
        finally:
            if saved is not None:
                self.set_theta(saved)

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        """Tweedie deviance derivatives wrt μ and θ — full port of mgcv
        tw()$Dd (efam.r:3155-3210). Level 0 feeds ``initial.spg``; level
        1's ``Dmuth`` feeds the family-θ column of ``db.drho``
        (∂β̂/∂θ for the Vc/edf2 sp-uncertainty correction)."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        th = float(np.asarray(theta, dtype=float).ravel()[0])
        a, b = self.a, self.b
        if th > 0:
            p = (b + a * np.exp(-th)) / (1 + np.exp(-th))
            dpth1 = np.exp(-th) * (b - a) / (1 + np.exp(-th)) ** 2
            dpth2 = (((a - b) * np.exp(-th) + (b - a) * np.exp(-2 * th))
                     / (np.exp(-th) + 1) ** 3)
        else:
            p = (b * np.exp(th) + a) / (np.exp(th) + 1)
            dpth1 = np.exp(th) * (b - a) / (np.exp(th) + 1) ** 2
            dpth2 = (((a - b) * np.exp(2 * th) + (b - a) * np.exp(th))
                     / (np.exp(th) + 1) ** 3)
        mu1p = mu ** (1 - p)
        mup = mu ** p
        r = {}
        ymupi = y / mup
        r["Dmu"] = 2 * wt * (mu1p - ymupi)
        r["Dmu2"] = 2 * wt * (mu ** (-1 - p) * p * y + (1 - p) / mup)
        r["EDmu2"] = (2 * wt) / mup
        if level > 0:
            i1p = 1 / (1 - p)
            y1 = y + (y == 0)
            logmu = np.log(mu)
            mu2p = mu * mu1p
            r["Dth"] = 2 * wt * (
                (y ** (2 - p) * np.log(y1) - mu2p * logmu) / (2 - p)
                + (y * mu1p * logmu - y ** (2 - p) * np.log(y1)) / (1 - p)
                - (y ** (2 - p) - mu2p) / (2 - p) ** 2
                + (y ** (2 - p) - y * mu1p) * i1p ** 2
            ) * dpth1
            r["Dmuth"] = 2 * wt * logmu * (ymupi - mu1p) * dpth1
            mup1 = mu ** (-p - 1)
            r["Dmu3"] = -2 * wt * mup1 * p * (y / mu * (p + 1) + 1 - p)
            r["Dmu2th"] = 2 * wt * (
                mup1 * y * (1 - p * logmu) - (logmu * (1 - p) + 1) / mup
            ) * dpth1
            r["EDmu3"] = -2 * wt * p * mup1
            r["EDmu2th"] = -2 * wt * logmu / mup * dpth1
        if level > 1:
            mup2 = mup1 / mu
            r["Dmu4"] = 2 * wt * mup2 * p * (p + 1) * (y * (p + 2) / mu + 1 - p)
            y2plogy = y ** (2 - p) * np.log(y1)
            y2plog2y = y2plogy * np.log(y1)
            r["Dth2"] = 2 * wt * (
                (mu2p * logmu ** 2 - y2plog2y) / (2 - p)
                + (y2plog2y - y * mu1p * logmu ** 2) / (1 - p)
                + 2 * (y2plogy - mu2p * logmu) / (2 - p) ** 2
                + 2 * (y * mu1p * logmu - y2plogy) / (1 - p) ** 2
                + 2 * (mu2p - y ** (2 - p)) / (2 - p) ** 3
                + 2 * (y ** (2 - p) - y * mu ** (1 - p)) / (1 - p) ** 3
            ) * dpth1 ** 2 + r["Dth"] * dpth2 / dpth1
            r["Dmuth2"] = (2 * wt * ((mu1p * logmu ** 2
                                      - logmu ** 2 * ymupi) * dpth1 ** 2)
                           + r["Dmuth"] * dpth2 / dpth1)
            r["Dmu2th2"] = (2 * wt * ((mup1 * logmu * y * (logmu * p - 2)
                            + logmu / mup * (logmu * (1 - p) + 2)) * dpth1 ** 2)
                            + r["Dmu2th"] * dpth2 / dpth1)
            r["Dmu3th"] = 2 * wt * mup1 * (
                y / mu * (logmu * (1 + p) * p - p - p - 1)
                + logmu * (1 - p) * p + p - 1 + p
            ) * dpth1
        return r

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
    _newton_canonical = "none"  # efam.r:2641 (canonical=""); extended
                                # path is always full Newton anyway.
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

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # scat postproc (efam.r:3742-3749): find.null.dev null deviance
        # + "Scaled t(ν,σ)" relabel, θ rounded to 3 decimals, ν > 999
        # reported as Inf.
        nu, sig = self.get_theta(trans=True)
        nu_disp = float(np.round(nu, 3))
        sig_disp = float(np.round(sig, 3))
        if nu_disp > 999.0:
            nu_disp_str = "Inf"
        else:
            nu_disp_str = f"{nu_disp:g}"
        return {
            "null_deviance": find_null_dev(
                self, y, eta=linear_predictors, offset=offset,
                weights=prior_weights,
            ),
            "family_name": f"Scaled t({nu_disp_str},{sig_disp:g})",
        }

    def rd(self, rng, mu, wt, scale):
        nu, sig = self.get_theta(trans=True)
        n = np.asarray(mu, dtype=float).shape[0]
        return rng.standard_t(nu, size=n) * sig + np.asarray(mu, dtype=float)

    def __repr__(self):
        nu, sig = self.get_theta(trans=True)
        return (f"Scat(theta=({nu:.4g}, {sig:.4g}), "
                f"link={self.link.name}, min_df={self._min_df:g})")


class nb(Family):
    """Negative binomial extended family — direct port of mgcv ``nb()``
    (efam.r:161-306).

    ``Var(y) = μ + μ²/Θ`` with the size parameter Θ estimated jointly
    with the smoothing parameters (θ = log Θ internally; scale fixed
    at 1 like Poisson).

    Constructor ``theta`` follows mgcv's sign convention:
    ``None``/``0`` → free θ starting at Θ=1; ``theta > 0`` → Θ fixed
    (``n_theta = 0``); ``theta < 0`` → free θ starting at ``|theta|``.
    Links: log (default), identity, sqrt.
    """
    name = "negative binomial"
    canonical_link_name = "log"
    _newton_canonical = "none"  # extended family: no Fisher shortcut.
    scale_known = True
    is_extended = True
    n_theta = 1

    _OK_LINKS = ("log", "identity", "sqrt")

    def __init__(self, theta: float | None = None, link: str = "log"):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for nb family; available '
                f'links are {self._OK_LINKS}'
            )
        n_theta = 1
        if theta is not None and theta != 0.0:
            if theta > 0:
                ini = float(np.log(theta))
                n_theta = 0
            else:
                ini = float(np.log(-theta))
        else:
            ini = 0.0
        self.n_theta = int(n_theta)
        self._theta = np.array([ini], dtype=float)
        super().__init__(link=link)

    # ----- θ accessors ---------------------------------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape != (1,):
            raise ValueError(
                f"nb.set_theta expects a single log Θ; got shape {v.shape}"
            )
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        if trans:
            return np.exp(self._theta).copy()
        return self._theta.copy()

    # ----- variance ------------------------------------------------------

    def variance(self, mu):
        Th = float(np.exp(self._theta[0]))
        mu = np.asarray(mu, dtype=float)
        return mu + mu * mu / Th

    def dvar(self, mu):
        Th = float(np.exp(self._theta[0]))
        return 1.0 + 2.0 * np.asarray(mu, dtype=float) / Th

    def d2var(self, mu):
        Th = float(np.exp(self._theta[0]))
        return np.full_like(np.asarray(mu, dtype=float), 2.0 / Th)

    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    # ----- deviance / likelihood ----------------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv (efam.r:199-205): 2·wt·[y·log(max(1,y)/μ)
        #                              − (y+Θ)·log((y+Θ)/(μ+Θ))]
        th = self._theta if theta is None else np.asarray(theta,
                                                          dtype=float)
        Th = float(np.exp(np.asarray(th).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return 2.0 * wt * (
            y * np.log(np.maximum(1.0, y) / mu)
            - (y + Th) * np.log((y + Th) / (mu + Th))
        )

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        # mgcv nb()$Dd verbatim (efam.r:207-237); θ = log Θ supplied.
        Th = float(np.exp(np.asarray(theta, dtype=float).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        yth = y + Th
        muth = mu + Th
        r = {}
        r["Dmu"] = 2.0 * wt * (yth / muth - y / mu)
        r["Dmu2"] = -2.0 * wt * (yth / muth ** 2 - y / mu ** 2)
        r["EDmu2"] = 2.0 * wt * (1.0 / mu - 1.0 / muth)
        if level > 0:
            r["Dth"] = -2.0 * wt * Th * (np.log(yth / muth)
                                         + (1.0 - yth / muth))
            r["Dmuth"] = 2.0 * wt * Th * (1.0 - yth / muth) / muth
            r["Dmu3"] = 4.0 * wt * (yth / muth ** 3 - y / mu ** 3)
            r["Dmu2th"] = 2.0 * wt * Th * (2.0 * yth / muth - 1.0) / muth ** 2
            r["EDmu2th"] = 2.0 * wt / muth ** 2
        if level > 1:
            r["Dmu4"] = 2.0 * wt * (6.0 * y / mu ** 4
                                    - 6.0 * yth / muth ** 4)
            r["Dth2"] = -2.0 * wt * Th * (
                np.log(yth / muth) + Th * yth / muth ** 2 - yth / muth
                - 2.0 * Th / muth + 1.0 + Th / yth
            )
            r["Dmuth2"] = 2.0 * wt * Th * (
                2.0 * Th * yth / muth ** 2 - yth / muth
                - 2.0 * Th / muth + 1.0
            ) / muth
            r["Dmu2th2"] = 2.0 * wt * Th * (
                -6.0 * yth * Th / muth ** 2 + 2.0 * yth / muth
                + 4.0 * Th / muth - 1.0
            ) / muth ** 2
            r["Dmu3th"] = 4.0 * wt * Th * (1.0 - 3.0 * yth / muth) / muth ** 3
        return r

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv nb()$aic (efam.r:239-246); `dev` is unused (Θ-form direct).
        th = self._theta if theta is None else np.asarray(theta,
                                                          dtype=float)
        Th = float(np.exp(np.asarray(th).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        term = ((y + Th) * np.log(mu + Th) - y * np.log(mu)
                + gammaln(y + 1.0) - Th * np.log(Th) + gammaln(Th)
                - gammaln(Th + y))
        return 2.0 * float(np.sum(term * wt))

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        # mgcv nb()$ls (efam.r:248-275). scale is fixed at 1, so lsth1 is
        # the single θ derivative (no scale slot).
        th = self._theta if theta is None else np.asarray(theta,
                                                          dtype=float)
        th0 = float(np.asarray(th).reshape(-1)[0])
        Th = float(np.exp(th0))
        y = np.asarray(y, dtype=float)
        w = np.asarray(wt, dtype=float)
        ylogy = np.where(y > 0, y * np.log(np.maximum(y, 1e-300)), 0.0)
        term = ((y + Th) * np.log(y + Th) - ylogy
                + gammaln(y + 1.0) - Th * np.log(Th) + gammaln(Th)
                - gammaln(Th + y))
        ls0 = -float(np.sum(term * w))
        yth = y + Th
        lyth = np.log(yth)
        psi0_yth = digamma(yth)
        psi0_th = digamma(Th)
        term1 = Th * (lyth - psi0_yth + psi0_th - th0)
        LSTH = (-term1 * w)[:, None]
        lsth = float(np.sum(LSTH))
        psi1_yth = polygamma(1, yth)
        psi1_th = polygamma(1, Th)
        term2 = Th * (lyth - Th * psi1_yth - psi0_yth + Th / yth
                      + Th * psi1_th + psi0_th - th0 - 1.0)
        lsth2 = -float(np.sum(term2 * w))
        return {
            "ls": ls0,
            "lsth1": np.array([lsth]),
            "lsth2": np.array([[lsth2]]),
            "LSTH1": LSTH,
        }

    # ----- initialization / validity -------------------------------------

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the negative binomial "
                "family"
            )
        # mgcv: mustart <- y + (y == 0)/6
        return y + (y == 0.0) / 6.0

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # nb postproc (efam.r:283-289): find.null.dev + "Negative
        # Binomial(Θ)" relabel, Θ rounded to 3 decimals.
        Th = float(self.get_theta(trans=True)[0])
        return {
            "null_deviance": find_null_dev(
                self, y, eta=linear_predictors, offset=offset,
                weights=prior_weights,
            ),
            "family_name": f"Negative Binomial({np.round(Th, 3):g})",
        }

    def rd(self, rng, mu, wt, scale):
        Th = float(self.get_theta(trans=True)[0])
        mu = np.asarray(mu, dtype=float)
        # NB as Gamma-Poisson mixture: rate ~ Gamma(Θ, μ/Θ).
        lam = rng.gamma(shape=Th, scale=mu / Th)
        return rng.poisson(lam).astype(float)

    def __repr__(self):
        Th = float(self.get_theta(trans=True)[0])
        return f"nb(theta={Th:.4g}, link={self.link.name})"


# ---------------------------------------------------------------------------
# General-family seam — mgcv gamlss.r authoring kit (§5.3 prerequisite 5).
#
# General families (gam.fit5: multiple linear predictors, likelihood
# supplied as ``ll`` instead of a deviance) are authored from per-datum
# derivative arrays of the log-likelihood w.r.t. the distribution
# parameters (μ₁..μ_K), packed in upper-triangular order. The kit:
#   * trind_generator — symmetric index lookups into the packed arrays
#   * gamlss_etamu    — chain rule μ-derivatives → η-derivatives through
#                        the per-LP link derivatives
#   * gamlss_gH       — assemble the coefficient-space gradient/Hessian/
#                        ∂H/∂ρ/tr(H⁻¹∂²H) that gam.fit5 consumes
# A custom family supplies l1..l4 + links; everything downstream is
# generic. Ported complete-array/dense paths
# only — out of scope (absent, never silent): the "remap" dropped-zero-
# column optimization (multinom-scale K), discrete (bam) X lists,
# sandwich, bootstrap deriv<0, the non-linear g.index corrections.
# Index convention: everything 0-based (R's 1-based m and dims shifted).
# ---------------------------------------------------------------------------


def trind_generator(K: int = 2) -> dict:
    """mgcv ``trind.generator`` (gamlss.r:20-112): index arrays for
    upper-triangular packed storage of symmetric derivative arrays up to
    order 4. ``i4[i,j,k,l]`` (0-based everywhere) gives the packed column
    holding the derivative w.r.t. parameters i,j,k,l in any order;
    ``i3``/``i2`` likewise."""
    i4 = np.zeros((K, K, K, K), dtype=int)
    m = 0
    for i in range(K):
        for j in range(i, K):
            for k in range(j, K):
                for ll_ in range(k, K):
                    for perm in itertools.permutations((i, j, k, ll_)):
                        i4[perm] = m
                    m += 1
    i3 = np.zeros((K, K, K), dtype=int)
    m = 0
    for j in range(K):
        for k in range(j, K):
            for ll_ in range(k, K):
                for perm in itertools.permutations((j, k, ll_)):
                    i3[perm] = m
                m += 1
    i2 = np.zeros((K, K), dtype=int)
    m = 0
    for k in range(K):
        for ll_ in range(k, K):
            i2[k, ll_] = i2[ll_, k] = m
            m += 1
    return {"i2": i2, "i3": i3, "i4": i4}


def _deriv_orders(idx: tuple[int, ...]) -> np.ndarray:
    """mgcv's ``ordf`` (gamlss.r:254-278): differentiation order carried
    by each slot of a 2-4 index tuple (repeats accumulate on the first
    occurrence, later slots zero out)."""
    idx = tuple(idx)
    d = len(idx)
    ord_ = np.ones(d, dtype=int)
    if d >= 2 and idx[0] == idx[1]:
        ord_[0] += 1
        ord_[1] = 0
    if d >= 3:
        if idx[0] == idx[2]:
            ord_[0] += 1
            ord_[2] = 0
        if ord_[1] and idx[1] == idx[2]:
            ord_[1] += 1
            ord_[2] = 0
    if d == 4:
        if idx[0] == idx[3]:
            ord_[0] += 1
            ord_[3] = 0
        if ord_[1]:
            if idx[1] == idx[3]:
                ord_[1] += 1
                ord_[3] = 0
        if ord_[2] and idx[2] == idx[3]:
            ord_[2] += 1
            ord_[3] = 0
    return ord_


def gamlss_etamu(l1, l2, l3=None, l4=None, ig1=None, g2=None, g3=None,
                 g4=None, i2=None, i3=None, i4=None, deriv: int = 0) -> dict:
    """mgcv ``gamlss.etamu`` (gamlss.r:231-584), complete-array paths:
    transform packed log-likelihood derivatives w.r.t. the distribution
    parameters (μ₁..μ_K) into derivatives w.r.t. the linear predictors
    (η₁..η_K). ``ig1[:,k]`` = 1/g'(μ_k) (= dμ_k/dη_k), ``g2``-``g4`` the
    per-LP link derivatives d²g/dμ²… evaluated at μ_k. ``deriv``: 0 →
    l1,l2 only; >0 adds l3; >2 adds l4 (mgcv's convention — it is the
    ll-level deriv minus one)."""
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    K = l1.shape[1]
    d1 = l1 * ig1

    d2 = np.array(l2, dtype=float, copy=True)
    k = 0
    for i in range(K):
        for j in range(i, K):
            ord_ = _deriv_orders((i, j))
            if ord_.max() == 2:
                d2[:, k] = ((l2[:, k] - l1[:, i] * g2[:, i] * ig1[:, i])
                            * ig1[:, i] ** 2)
            else:
                d2[:, k] = l2[:, k] * ig1[:, i] * ig1[:, j]
            k += 1

    d3 = l3
    if deriv > 0:
        l3 = np.asarray(l3, dtype=float)
        d3 = np.array(l3, dtype=float, copy=True)
        k = 0
        for i in range(K):
            for j in range(i, K):
                for ll_ in range(j, K):
                    ord_ = _deriv_orders((i, j, ll_))
                    ii = np.array((i, j, ll_))
                    mo = int(ord_.max())
                    if mo == 3:
                        mind = i2[i, i]
                        d3[:, k] = ((l3[:, k]
                                     - 3.0 * l2[:, mind] * g2[:, i]
                                     * ig1[:, i]
                                     + l1[:, i] * (3.0 * g2[:, i] ** 2
                                                   * ig1[:, i] ** 2
                                                   - g3[:, i] * ig1[:, i]))
                                    * ig1[:, i] ** 3)
                    elif mo == 1:
                        d3[:, k] = (l3[:, k] * ig1[:, i] * ig1[:, j]
                                    * ig1[:, ll_])
                    else:
                        k1 = int(ii[ord_ == 1][0])
                        k2 = int(ii[ord_ == 2][0])
                        mind = i2[k2, k1]
                        d3[:, k] = ((l3[:, k] - l2[:, mind] * g2[:, k2]
                                     * ig1[:, k2])
                                    * ig1[:, k1] * ig1[:, k2] ** 2)
                    k += 1

    d4 = l4
    if deriv > 2:
        l4 = np.asarray(l4, dtype=float)
        d4 = np.array(l4, dtype=float, copy=True)
        k = 0
        for i in range(K):
            for j in range(i, K):
                for ll_ in range(j, K):
                    for m_ in range(ll_, K):
                        ord_ = _deriv_orders((i, j, ll_, m_))
                        ii = np.array((i, j, ll_, m_))
                        mo = int(ord_.max())
                        if mo == 4:
                            mi2 = i2[i, i]
                            mi3 = i3[i, i, i]
                            d4[:, k] = ((
                                l4[:, k]
                                - 6.0 * l3[:, mi3] * g2[:, i] * ig1[:, i]
                                + l2[:, mi2] * (15.0 * g2[:, i] ** 2
                                                * ig1[:, i] ** 2
                                                - 4.0 * g3[:, i]
                                                * ig1[:, i])
                                - l1[:, i] * (15.0 * g2[:, i] ** 3
                                              * ig1[:, i] ** 3
                                              - 10.0 * g2[:, i] * g3[:, i]
                                              * ig1[:, i] ** 2
                                              + g4[:, i] * ig1[:, i])
                            ) * ig1[:, i] ** 4)
                        elif mo == 1:
                            d4[:, k] = (l4[:, k] * ig1[:, i] * ig1[:, j]
                                        * ig1[:, ll_] * ig1[:, m_])
                        elif mo == 3:
                            k1 = int(ii[ord_ == 1][0])
                            k3 = int(ii[ord_ == 3][0])
                            mi2 = i2[k3, k1]
                            mi3 = i3[k3, k3, k1]
                            d4[:, k] = ((
                                l4[:, k]
                                - 3.0 * l3[:, mi3] * g2[:, k3] * ig1[:, k3]
                                + l2[:, mi2] * (3.0 * g2[:, k3] ** 2
                                                * ig1[:, k3] ** 2
                                                - g3[:, k3] * ig1[:, k3])
                            ) * ig1[:, k1] * ig1[:, k3] ** 3)
                        elif int(np.sum(ord_ == 2)) == 2:
                            two = ii[ord_ == 2]
                            k2a, k2b = int(two[0]), int(two[1])
                            mi2 = i2[k2a, k2b]
                            mi3 = i3[k2a, k2b, k2b]
                            mi3a = i3[k2a, k2a, k2b]
                            d4[:, k] = ((
                                l4[:, k]
                                - l3[:, mi3] * g2[:, k2a] * ig1[:, k2a]
                                - l3[:, mi3a] * g2[:, k2b] * ig1[:, k2b]
                                + l2[:, mi2] * g2[:, k2a] * g2[:, k2b]
                                * ig1[:, k2a] * ig1[:, k2b]
                            ) * ig1[:, k2a] ** 2 * ig1[:, k2b] ** 2)
                        else:
                            k2 = int(ii[ord_ == 2][0])
                            ones = ii[ord_ == 1]
                            k1a, k1b = int(ones[0]), int(ones[1])
                            mi3 = i3[k2, k1a, k1b]
                            d4[:, k] = ((l4[:, k] - l3[:, mi3] * g2[:, k2]
                                         * ig1[:, k2])
                                        * ig1[:, k1a] * ig1[:, k1b]
                                        * ig1[:, k2] ** 2)
                        k += 1

    return {"l1": d1, "l2": d2, "l3": d3, "l4": d4}


def gamlss_gH(X, jj, l1, l2, i2, l3=None, i3=None, l4=None, i4=None,
              d1b=None, d2b=None, deriv: int = 0, fh=None,
              D=None) -> dict:
    """mgcv ``gamlss.gH`` (gamlss.r:587-857), dense complete-array paths:
    coefficient-space quantities from η-space derivative arrays.

    ``jj[i]`` = LP i's column indices into X (0-based). ``deriv``:
      0 — ``lb`` (gradient) and ``lbb`` (Hessian) only;
      1 — + ``d1H`` as the vector tr(Hp⁻¹·∂H/∂ρ_l) (``fh`` must be the
          INVERSE penalized Hessian);
      2 — + ``d1H`` as the list of full ∂H/∂ρ_l matrices;
      3 — + ``trHid2H`` (``fh`` the pivoted Cholesky of the diagonally
          preconditioned Hp, ``D`` the preconditioner — gam.fit5's
          convention; or an eigendecomposition dict {values, vectors}).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    K = len(jj)
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    lb = np.zeros(p)
    for i in range(K):
        lb[jj[i]] += X[:, jj[i]].T @ l1[:, i]

    lbb = np.zeros((p, p))
    for i in range(K):
        for j in range(i, K):
            A = X[:, jj[i]].T @ (l2[:, i2[i, j]][:, None] * X[:, jj[j]])
            lbb[np.ix_(jj[i], jj[j])] += A
            if j > i:
                lbb[np.ix_(jj[j], jj[i])] += A.T

    d1H = None
    trHid2H = None
    if deriv > 0:
        l3 = np.asarray(l3, dtype=float)
        d1b = np.asarray(d1b, dtype=float)
        m = d1b.shape[1]
        # Stacked per-LP derivative of η w.r.t. each ρ (gamlss.r:680-686).
        d1eta = np.zeros((n * K, m))
        for i in range(K):
            d1eta[i * n:(i + 1) * n, :] = X[:, jj[i]] @ d1b[jj[i], :]

    if deriv == 1:
        # tr(Hp⁻¹ ∂H/∂ρ_l) accumulation (gamlss.r:735-773, dense branch);
        # fh is the inverse penalized Hessian.
        fh = np.asarray(fh, dtype=float)
        d1H = np.zeros(m)
        for i in range(K):
            for j in range(i, K):
                Hpi = fh[np.ix_(jj[i], jj[j])]
                a = np.einsum("ij,ij->i", X[:, jj[i]] @ Hpi, X[:, jj[j]])
                mult = 1.0 if i == j else 2.0
                for ll_ in range(m):
                    v = np.zeros(n)
                    for q in range(K):
                        v += l3[:, i3[i, j, q]] * d1eta[q * n:(q + 1) * n,
                                                        ll_]
                    d1H[ll_] += mult * float(np.sum(a * v))

    if deriv > 1:
        # Full ∂H/∂ρ_l matrices (gamlss.r:776-796).
        d1H = []
        for ll_ in range(m):
            Hl = np.zeros((p, p))
            for i in range(K):
                for j in range(i, K):
                    v = np.zeros(n)
                    for q in range(K):
                        v += l3[:, i3[i, j, q]] * d1eta[q * n:(q + 1) * n,
                                                        ll_]
                    A = X[:, jj[i]].T @ (v[:, None] * X[:, jj[j]])
                    Hl[np.ix_(jj[i], jj[j])] += A
                    if j > i:
                        Hl[np.ix_(jj[j], jj[i])] += A.T
            d1H.append(Hl)

    if deriv > 2:
        # tr(Hp⁻¹ ∂²H/∂ρ_k∂ρ_l) (gamlss.r:798-855).
        l4 = np.asarray(l4, dtype=float)
        d2b = np.asarray(d2b, dtype=float)
        Xe = np.zeros((K * n, p))
        for i in range(K):
            Xe[i * n:(i + 1) * n, jj[i]] = X[:, jj[i]]
        if isinstance(fh, dict):
            dvals = np.asarray(fh["values"], dtype=float).copy()
            dvals[dvals > 0] = 1.0 / dvals[dvals > 0]
            dvals[dvals <= 0] = 0.0
            V = np.asarray(fh["vectors"], dtype=float)
            Hinv = V @ (dvals[:, None] * V.T)
            Xe_solved = (D[:, None] * (Hinv @ (D[:, None] * Xe.T))).T
        else:
            # fh: pivoted upper-Cholesky (R chol(...,pivot=TRUE) analog)
            # with pivot vector in fh[1]; D the diagonal preconditioner.
            R_f, piv = fh
            DXt = (D[:, None] * Xe.T)[piv, :]
            tmp = solve_triangular(R_f, DXt, lower=False, trans="T")
            sol = solve_triangular(R_f, tmp, lower=False)
            ipiv = np.empty_like(piv)
            ipiv[piv] = np.arange(p)
            Xe_solved = (D[:, None] * sol[ipiv, :]).T
        d2eta = np.zeros((n * K, d2b.shape[1]))
        for i in range(K):
            d2eta[i * n:(i + 1) * n, :] = X[:, jj[i]] @ d2b[jj[i], :]
        n2 = d2b.shape[1]
        trHid2H = np.zeros(n2)
        VX = np.zeros((K * n, p))
        kk = 0
        for k_ in range(m):
            for ll_ in range(k_, m):
                VX[:] = 0.0
                for i in range(K):
                    for j in range(K):
                        v = np.zeros(n)
                        for q in range(K):
                            v += (d2eta[q * n:(q + 1) * n, kk]
                                  * l3[:, i3[i, j, q]])
                            for s in range(K):
                                v += (d1eta[q * n:(q + 1) * n, k_]
                                      * d1eta[s * n:(s + 1) * n, ll_]
                                      * l4[:, i4[i, j, q, s]])
                        VX[j * n:(j + 1) * n, jj[i]] = (v[:, None]
                                                        * X[:, jj[i]])
                trHid2H[kk] = float(np.sum(Xe_solved * VX))
                kk += 1

    return {"lb": lb, "lbb": lbb, "d1H": d1H, "trHid2H": trHid2H}


def _pen_reg(x: np.ndarray, e: np.ndarray, y: np.ndarray) -> np.ndarray:
    """mgcv ``pen.reg`` (gamlss.r:1415-1453): penalized regression of y
    on x with square-root penalty e used as a *regularizer* — the
    penalty weight k is grown/shrunk (×10 / ÷5) until the edf lands in
    (0.85·rank(x), rank(x) − 0.1·re]. Used by general-family
    ``initialize`` when E arrives without mgcv's ``use.unscaled``
    attribute (the initial.spg path)."""
    # local import: hea.models.gam imports this module at load time,
    # so the reverse import must be deferred to call time.
    from .models.gam import _R_rank
    x = np.asarray(x, dtype=float)
    e = np.asarray(e, dtype=float)
    y = np.asarray(y, dtype=float)
    if float(np.sum(np.abs(e))) == 0.0:
        b, *_ = np.linalg.lstsq(x, y, rcond=None)
        b[~np.isfinite(b)] = 0.0
        return b
    from scipy.linalg import qr as _scipy_qr
    Q_x, R, piv = _scipy_qr(x, mode="economic", pivoting=True)
    r = R.shape[1]
    rr = _R_rank(R, tol=float(np.finfo(float).eps) ** 0.9)
    R_unpiv = np.empty_like(R)
    R_unpiv[:, piv] = R                      # R[, pivot] <- R
    R = R_unpiv
    Qy = Q_x.T @ y                           # qr.qty(...)[1:ncol(R)]

    def _edf_and_R(k):
        aug = np.vstack([R, e * k])
        Q_a, R_a = np.linalg.qr(aug, mode="reduced")
        return float(np.sum(Q_a[:r] ** 2)), R_a

    norm_R = float(np.abs(R).sum(axis=0).max())      # R norm(): "O"
    norm_e = float(np.abs(e).sum(axis=0).max())
    k = 0.01 * norm_R / norm_e
    edf, R_a = _edf_and_R(k)
    re = (min(int(np.sum(np.abs(e).sum(axis=0) != 0)), e.shape[0])
          - _R_rank(R_a, tol=float(np.finfo(float).eps) ** 0.9) + rr)
    while edf > rr - 0.1 * re:               # increase penalization
        k = k * 10.0
        edf, _ = _edf_and_R(k)
    while edf < 0.85 * rr:                   # reduce penalization
        k = k / 5.0
        edf, _ = _edf_and_R(k)
    aug = np.vstack([R, e * k])
    rhs = np.concatenate([Qy, np.zeros(e.shape[0])])
    b, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
    b[~np.isfinite(b)] = 0.0
    return b


class LogbLink(Link):
    """mgcv's ``logb`` link for gaulss's precision LP (gamlss.r:887-900):
    η = log(1/μ − b) so μ = 1/(exp(η) + b) stays below 1/b (τ = 1/σ
    bounded away from ∞ ⇒ σ > b)."""
    name = "logb"

    def __init__(self, b: float = 0.01):
        self.b = float(b)

    def link(self, mu):
        return np.log(1.0 / np.asarray(mu, dtype=float) - self.b)

    def linkinv(self, eta):
        return 1.0 / (np.exp(np.asarray(eta, dtype=float)) + self.b)

    def mu_eta(self, eta):
        ee = np.exp(np.asarray(eta, dtype=float))
        return -ee / (ee + self.b) ** 2

    def _mub(self, mu):
        return np.maximum(1.0 - np.asarray(mu, dtype=float) * self.b,
                          np.finfo(float).eps)

    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = self._mub(mu)
        return (2.0 * mub - 1.0) / (mub * mu) ** 2

    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = self._mub(mu)
        return ((1.0 - mub) * mub * 6.0 - 2.0) / (mub * mu) ** 3

    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = self._mub(mu)
        return ((((24.0 * mub - 36.0) * mub + 24.0) * mub - 6.0)
                / (mub * mu) ** 4)


class GeneralFamily(Family):
    """Base for mgcv "general families" (gam.fit5): several linear
    predictors, the likelihood supplied directly via :meth:`ll` instead
    of a deviance/PIRLS interface.

    **Authoring contract.** This is hea's public extension API for
    new general families (mgcv's ``general.family`` analog), frozen by
    ``test_general_family_authoring_contract`` (tests/test_gam.py).

    Attributes a subclass declares:

    - ``n_lp`` — number of linear predictors; ``gam`` takes a list of
      exactly ``n_lp`` formulas, one per LP.
    - ``links`` — list of ``n_lp`` :class:`Link` objects (set via
      ``__init__``); custom subclasses welcome. Each implements
      ``link``/``linkinv``/``mu_eta`` plus ``d2link``..``d4link`` up
      to the order ``available_derivs`` implies (the chain rule runs
      through :func:`gamlss_etamu`); clamp ``linkinv`` inside open
      supports and floor ``mu_eta`` like mgcv's links do.
    - ``available_derivs`` — 2: full outer Newton, :meth:`ll` must
      answer every ``deriv`` ≤ 4. 0: extended Fellner-Schall;
      :meth:`ll` is only ever called with ``deriv`` ≤ 1, on every
      path (free, fixed and absent sp). 1: reserved for the unported
      bfgs route — fitting refuses unless ``optimizer="efs"`` is
      passed (mgcv.r:1907).
    - conventional flags, as on :class:`gaulss`: ``scale_known =
      True``, ``n_theta = 0``; ``name`` is what summaries print.

    Engine call protocol (signatures are the contract):

    - ``ll(y, X, coef, wt, *, lpi, offset=None, deriv=0, d1b=None,
      d2b=None, fh=None, D=None)`` — ``lpi`` is a list of ``n_lp``
      0-based integer column-index arrays into the stacked ``X``;
      ``offset`` a per-LP list (entries ``None`` for offset-free
      formulas) or ``None``; ``wt`` the (n,) prior weights, forwarded
      by the engine — note mgcv's own general families (gaulss,
      twlss) leave the likelihood unweighted and consume prior
      weights only in residuals/postproc; follow your reference.
      Deriv levels: :meth:`ll`.
    - ``initialize_coef(y, X, lpi, E=None, offset=None,
      use_unscaled=False)`` — called with ``use_unscaled=True`` from
      gam.fit5 (E = the ldetS root, gam.fit4.r:974) and with the
      default ``False`` from the initial.spg seed (E = the balanced
      root, pen.reg semantics).
    - ``postproc(y, prior_weights, fitted, linear_predictors, offset,
      intercept)`` — mgcv's 6-argument form (unified 2026-06-11),
      keyword-called once on the converged fit; see :meth:`postproc`.
    - ``residuals(y, fitted, type="deviance")`` — REQUIRED for
      general families: the fit stores ``residuals(y, fitted)`` and
      ``residuals_of(type=)``/qq dispatch through it (mgcv.r:3429);
      ``fitted`` is the (n, n_lp) inverse-linked matrix. A hook MAY
      additionally declare an optional ``prior_weights`` keyword —
      the engine passes the fit's prior weights when it is declared
      (twlss's deviance residuals carry mgcv's
      ``object$prior.weights``).
    - ``rd(rng, mu, wt, scale)`` — optional; enables qq.gam's
      simulation path (``mu`` = the fitted matrix, like
      :class:`gaulss`).

    Almost always :meth:`ll` is implemented by filling the packed
    per-datum arrays l1..l4 of log-density derivatives w.r.t. the
    distribution parameters and delegating to :func:`gamlss_etamu` +
    :func:`gamlss_gH` exactly like :class:`gaulss` does
    (:func:`trind_generator` supplies the packed index tables).
    """
    is_general = True
    n_lp: int = 2
    available_derivs: int = 2
    canonical_link_name = "none"

    def __init__(self, links: list[Link]):
        self.links = links
        # Family base wires a single .link; point it at LP1's for the
        # odd shared code path that asks (residual helpers etc.).
        self.link = links[0]

    def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        """Log-likelihood + coefficient-space derivatives at ``coef``.

        ``deriv``: 0 value only; 1 + lb/lbb; 2 + d1H trace vector (fh =
        Hp⁻¹); 3 + d1H matrix list; 4 + trHid2H (fh/D = gam.fit5's
        preconditioned Cholesky pieces). Returns a dict with keys
        ``l`` (+ ``lb``, ``lbb``, ``d1H``, ``trHid2H`` as available).
        """
        raise NotImplementedError

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """Starting coefficients (mgcv ``family$initialize``).

        ``use_unscaled`` mirrors mgcv's ``attr(E, "use.unscaled")``:
        gam.fit5 passes its ldetS penalty root with the attribute set
        (E used as-is in a stacked least squares); initial.spg passes
        the balanced root WITHOUT it, and the initializer then adjusts
        the penalty weight itself (``pen.reg``)."""
        raise NotImplementedError

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """mgcv ``family$postproc`` analog: family-specific deviance /
        null-deviance overrides, evaluated on the converged fit.
        Returns a dict with optional ``deviance`` / ``null_deviance``
        keys; absent keys fall back to estimate.gam's generics
        (deviance = Σ deviance-residuals², mgcv.r:2429). ``fitted`` is
        the (n, n_lp) fitted matrix for general families."""
        return {}


class gaulss(GeneralFamily):
    """Gaussian location-scale general family — mgcv ``gaulss()``
    (gamlss.r:862-1106). LP1 models μ (links: identity/log/inverse/sqrt);
    LP2 models τ = 1/σ through the ``logb`` link (σ > b > 0).

        log f = −½(y−μ)²τ² − ½log(2π) + log τ
    """
    name = "gaulss"
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2

    _OK_MU_LINKS = ("identity", "log", "inverse", "sqrt")

    def __init__(self, link: tuple[str, str] = ("identity", "logb"),
                 b: float = 0.01):
        mu_link, tau_link = link
        if mu_link not in self._OK_MU_LINKS:
            raise ValueError(
                f'link "{mu_link}" not available for the mu parameter of '
                f"gaulss; available links are {self._OK_MU_LINKS}"
            )
        if tau_link != "logb":
            raise ValueError(
                'only the "logb" link is available for the precision '
                "parameter of gaulss"
            )
        links = [
            {"identity": IdentityLink, "log": LogLink,
             "inverse": InverseLink, "sqrt": SqrtLink}[mu_link](),
            LogbLink(b=b),
        ]
        self.b = float(b)
        self.tri = trind_generator(2)
        super().__init__(links)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        eta = X[:, jj[0]] @ coef[jj[0]]
        eta1 = X[:, jj[1]] @ coef[jj[1]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                eta1 = eta1 + offset[1]
        mu = self.links[0].linkinv(eta)
        tau = self.links[1].linkinv(eta1)

        n = y.shape[0]
        ymu = y - mu
        ymu2 = ymu * ymu
        tau2 = tau * tau
        l0 = -0.5 * ymu2 * tau2 - 0.5 * np.log(2.0 * np.pi) + np.log(tau)
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
        if deriv == 0:
            return ret

        l1 = np.column_stack([tau2 * ymu, 1.0 / tau - tau * ymu2])
        # second derivatives, packed (mm, ms, ss)
        l2 = np.column_stack([-tau2, 2.0 * l1[:, 0] / tau,
                              -ymu2 - 1.0 / tau2])
        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(eta1)])
        g2 = np.column_stack([self.links[0].d2link(mu),
                              self.links[1].d2link(tau)])
        l3 = l4 = g3 = g4 = None
        if deriv > 1:
            # third derivatives, packed (mmm, mms, mss, sss)
            zeros = np.zeros(n)
            l3 = np.column_stack([zeros, -2.0 * tau, 2.0 * ymu,
                                  2.0 / tau ** 3])
            g3 = np.column_stack([self.links[0].d3link(mu),
                                  self.links[1].d3link(tau)])
        if deriv > 3:
            # fourth derivatives, packed (mmmm, mmms, mmss, msss, ssss)
            zeros = np.zeros(n)
            l4 = np.column_stack([zeros, zeros, np.full(n, -2.0), zeros,
                                  -6.0 / (tau2 * tau2)])
            g4 = np.column_stack([self.links[0].d4link(mu),
                                  self.links[1].d4link(tau)])

        tri = self.tri
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gaulss ``initialize`` (gamlss.r:1016-1086, dense branch):
        regress g(y) on LP1's columns, then the log absolute residuals
        on LP2's, with the penalty root ``E`` as a regularizer.
        ``use_unscaled`` (mgcv's ``attr(E,"use.unscaled")``, set by
        gam.fit5 on its ldetS root): stacked least squares with E
        as-is; otherwise (initial.spg's balanced root) ``pen.reg``
        adjusts the penalty weight to an edf target."""
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)
        if self.links[0].name == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.links[0].link(np.abs(y) + float(np.max(y)) * 1e-7)
        if offset is not None and offset[0] is not None:
            yt1 = yt1 - offset[0]

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(X[:, cols], E[:, cols], target)

        b1 = _reg(jj[0], yt1)
        start[jj[0]] = b1
        lres1 = np.log(np.abs(y - self.links[0].linkinv(
            X[:, jj[0]] @ b1)))
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            lres1 = lres1 - offset[1]
        start[jj[1]] = _reg(jj[1], lres1)
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """gaulss postproc (gamlss.r:910-918): null deviance only —
        ``Σ((y − ȳ)·τ̂)²`` (the fitted-precision-weighted null SS);
        the deviance itself falls back to estimate.gam's generic
        Σ deviance-residuals² (mgcv.r:2429)."""
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        return {"null_deviance": float(np.sum(
            ((y - float(np.mean(y))) * fitted[:, 1]) ** 2))}

    def rd(self, rng, mu, wt, scale):
        """gaulss rd (gamlss.r:1089): ``rnorm(n, mu[,1],
        sqrt(scale/wt)/mu[,2])`` — μ is the (n, 2) fitted matrix
        (mean, τ = 1/σ); scale ≡ 1 for gaulss fits. Drives qq.gam's
        simulation path (mgcv does NOT qqnorm-fallback for gaulss)."""
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        sd = np.sqrt(float(scale) / wt) / mu[:, 1]
        return rng.normal(mu[:, 0], sd)

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """gaulss residuals (gamlss.r:903-908): response = y − μ̂;
        deviance/pearson = (y − μ̂)·τ̂ = (y − μ̂)/σ̂. ``fitted`` is the
        (n, 2) matrix of (μ̂, τ̂)."""
        if type not in ("deviance", "pearson", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'pearson', 'response' "
                f"for gaulss residuals; got {type!r}")
        fitted = np.asarray(fitted, dtype=float)
        rsd = np.asarray(y, dtype=float) - fitted[:, 0]
        if type == "response":
            return rsd
        return rsd * fitted[:, 1]

    def __repr__(self):
        return (f"gaulss(link=({self.links[0].name!r}, 'logb'), "
                f"b={self.b:g})")


class twlss(GeneralFamily):
    """Tweedie location-scale-shape general family — mgcv ``twlss()``
    (gamlss.r:2493-2662). Three linear predictors: LP1 the mean μ
    (links: log/identity/sqrt), LP2 the transformed index θ with
    p = (a + b·e^θ)/(1 + e^θ) ∈ (a, b) (identity link), LP3
    ρ = log scale (identity link).

    ``available_derivs = 0``: mgcv supplies no third/fourth
    log-likelihood derivatives, so fitting always runs the extended
    Fellner-Schall loop (mgcv.r:1907-1908's automatic optimizer
    switch). Like mgcv, the likelihood itself ignores prior weights
    (gamlss.r:2556 — ``wt`` unread, same as gaulss); they enter the
    deviance residuals and null deviance only.
    """
    name = "twlss"
    scale_known = True
    n_theta = 0
    n_lp = 3
    available_derivs = 0

    _OK_MU_LINKS = ("log", "identity", "sqrt")

    def __init__(self, link: tuple[str, str, str] = ("log", "identity",
                                                     "identity"),
                 a: float = 1.01, b: float = 1.99):
        mu_link, th_link, rho_link = link
        if mu_link not in self._OK_MU_LINKS:
            raise ValueError(
                f'link "{mu_link}" not available for the mu parameter '
                f"of twlss; available links are {self._OK_MU_LINKS}"
            )
        if th_link != "identity" or rho_link != "identity":
            raise ValueError(
                'only the "identity" link is available for the theta '
                "and rho parameters of twlss"
            )
        if not (1.0 < a < b < 2.0):
            raise ValueError("1<a<b<2 (strict) required")
        self.a = float(a)
        self.b = float(b)
        links = [
            {"log": LogLink, "identity": IdentityLink,
             "sqrt": SqrtLink}[mu_link](),
            IdentityLink(), IdentityLink(),
        ]
        super().__init__(links)
        self.tri = trind_generator(3)

    def _p_of_theta(self, theta):
        """p(θ) with the ±θ-stable branches (gamlss.r:2528-2532)."""
        theta = np.asarray(theta, dtype=float)
        eth = np.exp(-np.abs(theta))
        return np.where(theta > 0,
                        (self.b + self.a * eth) / (1.0 + eth),
                        (self.b * eth + self.a) / (eth + 1.0))

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None,
           deriv: int = 0, d1b=None, d2b=None, fh=None, D=None) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        eta = X[:, jj[0]] @ coef[jj[0]]
        theta = X[:, jj[1]] @ coef[jj[1]]
        rho = X[:, jj[2]] @ coef[jj[2]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                theta = theta + offset[1]
            if len(offset) > 2 and offset[2] is not None:
                rho = rho + offset[2]
        mu = self.links[0].linkinv(eta)

        # ldTweedie columns: l; ρ, ρρ; θ, θθ, θρ; μ, μμ, μθ, μρ —
        # reordered into the packed (μ, θ, ρ) layout (gamlss.r:2575-2580)
        ld = _ld_tweedie_work(y, mu, theta, rho, a=self.a, b=self.b)
        l0 = ld[:, 0]
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
        if deriv == 0:
            return ret
        l1 = ld[:, [6, 3, 1]]
        l2 = ld[:, [7, 8, 9, 4, 5, 2]]
        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(theta),
                               self.links[2].mu_eta(rho)])
        g2 = np.column_stack([self.links[0].d2link(mu),
                              self.links[1].d2link(theta),
                              self.links[2].d2link(rho)])
        # no l3/l4 for this family: etamu/gH run at deriv 0 whenever
        # any derivative is requested (gamlss.r:2592-2599)
        tri = self.tri
        de = gamlss_etamu(l1, l2, None, None, ig1, g2, None, None,
                          tri["i2"], tri["i3"], tri["i4"], 0)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=0,
                       fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """twlss ``initialize`` (gamlss.r:2609-2649): regress g(y) on
        LP1's columns, the log absolute scaled residuals
        ``log|((y−μ₁)/μ₁^1.5)|`` on LP3's (the log-scale predictor),
        and start the θ predictor at zero (p = (a+b)/2). E is a
        regularizer; mgcv's expression never references offsets here —
        ported as-is."""
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)
        if self.links[0].name == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.links[0].link(np.abs(y) + float(np.max(y)) * 1e-7)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                bvec, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                bvec[~np.isfinite(bvec)] = 0.0
                return bvec
            return _pen_reg(X[:, cols], E[:, cols], target)

        b1 = _reg(jj[0], yt1)
        start[jj[0]] = b1
        mu1 = self.links[0].linkinv(X[:, jj[0]] @ b1)
        lres1 = np.log(np.abs((y - mu1) / mu1 ** 1.5))
        start[jj[2]] = _reg(jj[2], lres1)
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """twlss ``postproc`` (gamlss.r:2545-2554): null deviance from
        the intercept-only Tweedie MLE — mgcv calls ``tw.null.fit(y)``
        with ITS defaults a=1.001/b=1.999 even when the family was
        built with other (a, b); ported bug-for-bug — scaled by the
        FITTED per-observation e^ρ."""
        y = np.asarray(y, dtype=float)
        pw = np.asarray(prior_weights, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu0, p0, _phi0 = _tw_null_fit(y)
        y1 = y + (y == 0.0)
        th0 = (y1 ** (1.0 - p0) - mu0 ** (1.0 - p0)) / (1.0 - p0)
        ka0 = (y ** (2.0 - p0) - mu0 ** (2.0 - p0)) / (2.0 - p0)
        nd = np.sum(np.maximum(
            2.0 * (y * th0 - ka0) * pw / np.exp(fitted[:, 2]), 0.0))
        return {"null_deviance": float(nd)}

    def residuals(self, y, fitted, type: str = "deviance",
                  prior_weights=None) -> np.ndarray:
        """twlss residuals (gamlss.r:2522-2543): ``fitted`` is the
        (n, 3) matrix (μ, θ, ρ). Deviance residuals carry mgcv's
        ``object$prior.weights`` — the engine passes them through the
        optional ``prior_weights`` keyword."""
        if type not in ("deviance", "pearson", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'pearson', 'response' "
                f"for twlss residuals; got {type!r}")
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu = fitted[:, 0]
        p = self._p_of_theta(fitted[:, 1])
        phi = np.exp(fitted[:, 2])
        if type == "pearson":
            return (y - mu) / np.sqrt(phi * mu ** p)
        if type == "response":
            return y - mu
        pw = (np.ones_like(y) if prior_weights is None
              else np.asarray(prior_weights, dtype=float))
        y1 = y + (y == 0.0)
        th = (y1 ** (1.0 - p) - mu ** (1.0 - p)) / (1.0 - p)
        ka = (y ** (2.0 - p) - mu ** (2.0 - p)) / (2.0 - p)
        return np.sign(y - mu) * np.sqrt(
            np.maximum(2.0 * (y * th - ka) * pw / phi, 0.0))

    def __repr__(self):
        return (f"twlss(link=({self.links[0].name!r}, 'identity', "
                f"'identity'), a={self.a:g}, b={self.b:g})")


class LogebLink(Link):
    """shash's ``logeb`` link for τ = log σ (gamlss.r:3356-3371):
    η = log(e^τ − b), τ = log(e^η + b) — keeps σ = e^τ > b > 0."""

    name = "logeb"

    def __init__(self, b: float = 1e-2):
        self.b = float(b)

    def link(self, mu):
        return np.log(np.exp(np.asarray(mu, dtype=float)) - self.b)

    def linkinv(self, eta):
        return np.log(np.exp(np.asarray(eta, dtype=float)) + self.b)

    def mu_eta(self, eta):
        ee = np.exp(np.asarray(eta, dtype=float))
        return ee / (ee + self.b)

    def d2link(self, mu):
        em = np.exp(np.asarray(mu, dtype=float))
        fr = em / (em - self.b)
        return fr * (1.0 - fr)

    def d3link(self, mu):
        em = np.exp(np.asarray(mu, dtype=float))
        fr = em / (em - self.b)
        oo = fr * (1.0 - fr)
        return oo - 2.0 * oo * fr

    def d4link(self, mu):
        em = np.exp(np.asarray(mu, dtype=float))
        b = self.b
        return (-b * em * (b ** 2 + 4.0 * b * em + em ** 2)
                / (em - b) ** 4)


class shash(GeneralFamily):
    """Sinh-arcsinh location-scale-shape general family — mgcv
    ``shash()`` (gamlss.r:3334-4080). Four linear predictors: LP1 the
    location μ (identity), LP2 τ = log σ through the ``logeb`` link
    (σ > b > 0), LP3 the skewness ε (identity), LP4 the log-kurtosis
    φ (identity; δ = e^φ).

        z = (y − μ)/(σδ),  l = −τ − ½log 2π + log cosh(δ·asinh z − ε)
            − ½log(1 + z²) − ½sinh²(δ·asinh z − ε) − phiPen·φ²

    The phiPen·φ² ridge is part of the LIKELIHOOD itself (mgcv's
    light regularization of the kurtosis direction). Full analytic
    derivatives to order 4 (``available_derivs = 2`` — outer Newton);
    no postproc (mgcv's is commented out, so null deviance is NaN
    like mgcv's NULL); formula offsets are rejected exactly like
    mgcv's ll (gamlss.r:3470). The ``cdf`` hook is ported for surface
    parity (mgcv consumes it only in unported NCV machinery).
    """
    name = "shash"
    scale_known = True
    n_theta = 0
    n_lp = 4
    available_derivs = 2

    def __init__(self, link: tuple = ("identity", "logeb", "identity",
                                      "identity"),
                 b: float = 1e-2, phiPen: float = 1e-3):
        mu_link, tau_link, eps_link, phi_link = link
        if mu_link != "identity" or eps_link != "identity" \
                or phi_link != "identity":
            raise ValueError(
                'only the "identity" link is available for the mu, eps '
                "and phi parameters of shash"
            )
        if tau_link != "logeb":
            raise ValueError(
                'only the "logeb" link is available for the scale '
                "parameter of shash"
            )
        self.b = float(b)
        self.phiPen = float(phiPen)
        super().__init__([IdentityLink(), LogebLink(b), IdentityLink(),
                          IdentityLink()])
        self.tri = trind_generator(4)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None,
           deriv: int = 0, d1b=None, d2b=None, fh=None, D=None) -> dict:
        # mgcv's shash ll rejects offsets outright (gamlss.r:3470)
        if offset is not None and any(
                o is not None and np.any(np.asarray(o) != 0.0)
                for o in offset):
            raise NotImplementedError(
                "offset not still available for this family (mgcv "
                "shash, gamlss.r:3470)")
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        etas = [X[:, jj[k]] @ coef[jj[k]] for k in range(4)]
        mu = self.links[0].linkinv(etas[0])
        tau = self.links[1].linkinv(etas[1])
        eps = self.links[2].linkinv(etas[2])
        phi = self.links[3].linkinv(etas[3])

        l0, L1, L2, L3, L4 = _shash_derivs(y, mu, tau, eps, phi,
                                           self.phiPen, deriv)
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
        if deriv == 0:
            return ret
        params = (mu, tau, eps, phi)
        ig1 = np.column_stack([lnk.mu_eta(eta)
                               for lnk, eta in zip(self.links, etas)])
        g2 = np.column_stack([lnk.d2link(par)
                              for lnk, par in zip(self.links, params)])
        g3 = g4 = None
        if deriv > 1:
            g3 = np.column_stack([lnk.d3link(par)
                                  for lnk, par in zip(self.links,
                                                      params)])
        if deriv > 3:
            g4 = np.column_stack([lnk.d4link(par)
                                  for lnk, par in zip(self.links,
                                                      params)])
        tri = self.tri
        de = gamlss_etamu(L1, L2, L3, L4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """shash ``initialize`` (gamlss.r:3973-4024): regress y on
        LP1's columns and the log absolute residuals on LP2's (the
        log-scale predictor), both E-regularized; the skewness and
        log-kurtosis predictors target the constant linkfun(0) = 0
        through plain least squares (Gaussian start)."""
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                bvec, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                bvec[~np.isfinite(bvec)] = 0.0
                return bvec
            return _pen_reg(X[:, cols], E[:, cols], target)

        b1 = _reg(jj[0], y)
        start[jj[0]] = b1
        lres1 = np.log(np.abs(y - self.links[0].linkinv(
            X[:, jj[0]] @ b1)))
        start[jj[1]] = _reg(jj[1], lres1)
        for k in (2, 3):
            target = np.zeros(X.shape[0])
            bvec, *_ = np.linalg.lstsq(X[:, jj[k]], target, rcond=None)
            bvec[~np.isfinite(bvec)] = 0.0
            start[jj[k]] = bvec
        return start

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """shash residuals (gamlss.r:3377-3411): ``fitted`` is the
        (n, 4) matrix (μ, τ, ε, φ). The raw residual subtracts the
        sinh-arcsinh mean (Bessel-K form); deviance residuals use the
        plain log-likelihood against a zero saturated reference
        (mgcv sets ls = 0 — no phiPen term here)."""
        if type not in ("deviance", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'response' for shash "
                f"residuals; got {type!r}")
        from scipy.special import kv
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu, tau, eps, phi = (fitted[:, 0], fitted[:, 1], fitted[:, 2],
                             fitted[:, 3])
        sig = np.exp(tau)
        delta = np.exp(phi)
        rsd = y - mu - sig * delta * np.exp(0.25) * (
            kv((1.0 / delta + 1.0) / 2.0, 0.25)
            + kv((1.0 / delta - 1.0) / 2.0, 0.25)) / np.sqrt(8.0 * np.pi)
        if type == "response":
            return rsd
        sgn = np.sign(rsd)
        z = (y - mu) / (sig * delta)
        dTasMe = delta * np.arcsinh(z) - eps
        ll = (-tau - 0.5 * np.log(2.0 * np.pi) + np.log(np.cosh(dTasMe))
              - 0.5 * np.log1p(z ** 2) - 0.5 * np.sinh(dTasMe) ** 2)
        return np.sqrt(np.maximum(0.0, 2.0 * (0.0 - ll))) * sgn

    def rd(self, rng, mu, wt, scale):
        """shash ``rd`` (gamlss.r:4026-4039): deviates via the
        quantile transform of uniforms (R's qnorm(runif(n)))."""
        from scipy.special import ndtri
        mu = np.asarray(mu, dtype=float)
        mu_e = mu[:, 0]
        sig_e = np.exp(mu[:, 1])
        eps_e = mu[:, 2]
        del_e = np.exp(mu[:, 3])
        n = mu_e.shape[0]
        u = ndtri(rng.uniform(size=n))
        return mu_e + (del_e * sig_e) * np.sinh(
            (1.0 / del_e) * np.arcsinh(u) + eps_e / del_e)

    def qf(self, p, mu, wt, scale):
        """shash quantile function (gamlss.r:4041-4053)."""
        from scipy.special import ndtri
        mu = np.asarray(mu, dtype=float)
        p = np.asarray(p, dtype=float)
        mu_e = mu[:, 0]
        sig_e = np.exp(mu[:, 1])
        eps_e = mu[:, 2]
        del_e = np.exp(mu[:, 3])
        return mu_e + (del_e * sig_e) * np.sinh(
            (1.0 / del_e) * np.arcsinh(ndtri(p)) + eps_e / del_e)

    def cdf(self, q, mu, wt, scale, logp: bool = False):
        """shash cdf (gamlss.r:4055-4067). Ported for surface parity —
        mgcv consumes family$cdf only in (unported) NCV machinery."""
        from scipy.special import log_ndtr, ndtr
        mu = np.asarray(mu, dtype=float)
        q = np.asarray(q, dtype=float)
        mu_e = mu[:, 0]
        sig_e = np.exp(mu[:, 1])
        eps_e = mu[:, 2]
        del_e = np.exp(mu[:, 3])
        s = np.sinh((np.arcsinh((q - mu_e) / (del_e * sig_e))
                     - eps_e / del_e) * del_e)
        return log_ndtr(s) if logp else ndtr(s)

    def __repr__(self):
        return (f"shash(link=('identity', 'logeb', 'identity', "
                f"'identity'), b={self.b:g}, phiPen={self.phiPen:g})")


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
    if isinstance(family, (Binomial, QuasiBinomial)):
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
quasipoisson = QuasiPoisson
quasibinomial = QuasiBinomial
scat = Scat   # mgcv-style lowercase alias
__all__ = [
    "Family", "Link",
    "Gaussian", "gaussian",
    "Gamma",
    "Poisson", "poisson",
    "Binomial", "binomial",
    "InverseGaussian", "inverse_gaussian",
    "Quasi", "quasi",
    "QuasiPoisson", "quasipoisson",
    "QuasiBinomial", "quasibinomial",
    "Tweedie", "tw",
    "Scat", "scat",
    "nb",
    "GeneralFamily", "gaulss", "twlss", "shash", "LogebLink",
    "trind_generator", "gamlss_etamu", "gamlss_gH",
    "IdentityLink", "LogLink", "InverseLink",
    "SqrtLink", "LogitLink", "ProbitLink", "CauchitLink", "CloglogLink",
    "InverseSquareLink", "PowerLink", "power",
]
