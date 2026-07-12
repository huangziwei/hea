"""R's d* / p* / q* / r* distribution surface, plus ``set.seed`` and ``sample``.

The **central** d/p/q functions (normal, t, F, chi-square, gamma, beta,
binomial, Poisson, exponential) route through :mod:`hea.R.nmath` — bit-exact
ports of R's ``src/nmath/`` C kernels (Cody pnorm, Wichura qnorm, Welinder
pgamma, TOMS 708 pbeta, Loader dpois/dbinom, AS 91/109 qgamma/qbeta) — so
``hea.R`` d/p/q is 0-ulp to R, not scipy's ~1-3 ulp approximations. ``unif`` is
an exact closed-form port of nmath/{dunif,punif,qunif}.c (no special functions).
The **non-central** variants (``ncp != 0``), the studentized range
(``ptukey``/``qtukey``) and the hypergeometric (``dhyper``/``phyper``) are also
ported from nmath (dnt/pnt/qnt, dnf/pnf/qnf, dnchisq/pnchisq/qnchisq, pnbeta,
ptukey/qtukey, dhyper/phyper). The pure-Python reference is 0-ulp to R (except a
≤1-ulp residual on the ncp>=80 far-lower-tail of pnchisq). When the ``hea._rs``
extension is built these route through a Rust **f64** fast path (~100-2500x
faster — at or near R's C speed): R accumulates these AS-275/AS-226/AS-243/
Copenhaver-Holland series in 80-bit ``LDOUBLE`` and Rust std has no 80-bit
float, so the f64 result matches R to a few ulp (extreme tails degrade more),
not bit-for-bit. Absent the extension, the 0-ulp Python path runs unchanged.
No ``scipy.stats`` in the d/p/q path.

Conventions kept from R: ``lower_tail`` (R's ``lower.tail``) and ``log_p`` /
``log`` for p* / q* / d*; ``ncp`` non-centrality where applicable; ``lambda_``
for Poisson (R's ``lambda`` is a Python keyword). ``df`` PDF is intentionally
omitted — ``df`` is too common as a DataFrame variable; use
``scipy.stats.f.pdf`` directly when you need it.
"""
from __future__ import annotations

import math

import numpy as np

from . import nmath as _nm
from ._shared import NamedVector
from .rng import RMersenneTwister

# --- Process-global R Mersenne-Twister backing the r* / sample surface --------
# R keeps ONE global RNG; ``set_seed`` (R's ``set.seed``) reseeds it and every
# ported r* family draws from it, so ``set_seed(k); runif(n)`` is bit-exact to
# R's ``set.seed(k); runif(n)``. All of runif / rnorm / rexp / rpois / rbinom /
# rgamma / rbeta / rchisq / rt / rf and ``sample`` (incl. weighted) route through
# this one R MT stream — bit-exact, no scipy/numpy RNG in the r* path.
_R_RNG: RMersenneTwister | None = None


def _r_rng() -> RMersenneTwister:
    """The process-global R MT stream. Lazily seeded from the clock when
    ``set_seed`` was never called (mirrors R seeding from time-of-day), so draws
    are non-deterministic by default but always routed through R's algorithm."""
    global _R_RNG
    if _R_RNG is None:
        import time
        _R_RNG = RMersenneTwister(int(time.time()) & 0x7FFFFFFF)
    return _R_RNG


def _recycle(p, n: int) -> np.ndarray:
    """R-style recycling of a parameter to length ``n`` (scalar → repeat,
    length-n → as-is, else tile like R's recycling rule)."""
    arr = np.asarray(p, dtype=float).ravel()
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    return np.resize(arr, n)


# normal
def dnorm(x, mean=0, sd=1, log=False):
    """R's ``dnorm`` — bit-exact via the ported ``dnorm5`` (nmath/dnorm.c)."""
    if np.ndim(x) == 0 and np.ndim(mean) == 0 and np.ndim(sd) == 0:
        return _nm.dnorm5(float(x), float(mean), float(sd), log)
    return _nm.dnorm5_vec(x, mean, sd, log)


def pnorm(q, mean=0, sd=1, lower_tail=True, log_p=False):
    """R's ``pnorm`` — bit-exact via the ported ``pnorm5`` (nmath/pnorm.c, Cody)."""
    if np.ndim(q) == 0 and np.ndim(mean) == 0 and np.ndim(sd) == 0:
        return _nm.pnorm5(float(q), float(mean), float(sd), lower_tail, log_p)
    return _nm.pnorm5_vec(q, mean, sd, lower_tail, log_p)


def qnorm(p, mean=0, sd=1, lower_tail=True, log_p=False):
    """R's ``qnorm`` — bit-exact via the ported ``qnorm5`` (nmath/qnorm.c, AS-241)."""
    if np.ndim(p) == 0 and np.ndim(mean) == 0 and np.ndim(sd) == 0:
        return _nm.qnorm5(float(p), float(mean), float(sd), lower_tail, log_p)
    return _nm.qnorm5_vec(p, mean, sd, lower_tail, log_p)


def rnorm(n, mean=0, sd=1):
    """R: ``rnorm(n, mean=0, sd=1)`` — n samples from Normal(mean, sd).

    Eagerly returns a numpy array when ``n`` is an int. When ``n`` is a
    polars ``Expr`` (e.g. ``rnorm(length(col("x")))`` inside a tibble
    / with_columns), returns an Expr that produces N random normals
    at evaluation time — N resolves against the receiver's row count.
    """
    import polars as pl
    if isinstance(n, pl.Expr):
        # Lazy per-row generation inside a tibble — draws from R's stream in
        # row order (reproducible and R-exact per draw; the in-frame generation
        # order is not something R itself defines).
        rng = _r_rng()
        return pl.int_range(0, n).map_elements(
            lambda _: mean + sd * rng.norm_rand(),
            return_dtype=pl.Float64,
        )
    return _r_rng().rnorm(int(n), mean=mean, sd=sd)


# Student's t  (df = degrees of freedom, ncp = non-centrality)
def dt(x, df, ncp=0, log=False):
    """R's ``dt`` — bit-exact via ported dt / dnt (nmath/dt.c, dnt.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("dt", _nm.dt, [x, df], (log,))
    return _nm._disp("dnt", _nm.dnt, [x, df, ncp], (log,))


def pt(q, df, ncp=0, lower_tail=True, log_p=False):
    """R's ``pt`` — bit-exact via ported pt / pnt (nmath/pt.c, pnt.c AS 243)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("pt", _nm.pt, [q, df], (lower_tail, log_p))
    return _nm._disp("pnt", _nm.pnt, [q, df, ncp], (lower_tail, log_p))


def qt(p, df, ncp=0, lower_tail=True, log_p=False):
    """R's ``qt`` — bit-exact via ported qt / qnt (nmath/qt.c, qnt.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("qt", _nm.qt, [p, df], (lower_tail, log_p))
    return _nm._disp("qnt", _nm.qnt, [p, df, ncp], (lower_tail, log_p))


def rt(n, df, ncp=0):
    """R: ``rt(n, df, ncp=0)`` on R's MT stream (bit-exact). Central: per-element
    ``norm_rand()/sqrt(rchisq(df)/df)``. Noncentral: R's block form
    ``rnorm(n, ncp)/sqrt(rchisq(n, df)/df)``."""
    rng = _r_rng()
    nn = int(n)
    if np.all(np.asarray(ncp) == 0):
        return rng.rt_n(_recycle(df, nn))
    num = rng.rnorm(nn, mean=_recycle(ncp, nn))
    dfv = _recycle(df, nn)
    den = rng.rchisq_n(dfv, np.zeros(nn))
    return num / np.sqrt(den / dfv)


# F  (df() PDF intentionally omitted — clashes with `df` variable name)
def pf(q, df1, df2, ncp=0, lower_tail=True, log_p=False):
    """R's ``pf`` — bit-exact via ported pf / pnf (nmath/pf.c, pnf.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("pf", _nm.pf, [q, df1, df2], (lower_tail, log_p))
    return _nm._disp("pnf", _nm.pnf, [q, df1, df2, ncp], (lower_tail, log_p))


def qf(p, df1, df2, ncp=0, lower_tail=True, log_p=False):
    """R's ``qf`` — bit-exact via ported qf / qnf (nmath/qf.c, qnf.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("qf", _nm.qf, [p, df1, df2], (lower_tail, log_p))
    return _nm._disp("qnf", _nm.qnf, [p, df1, df2, ncp], (lower_tail, log_p))


def rf(n, df1, df2, ncp=0):
    """R: ``rf(n, df1, df2, ncp=0)`` on R's MT stream (bit-exact). Central:
    per-element. Noncentral: R's block form
    ``(rchisq(n, df1, ncp)/df1)/(rchisq(n, df2)/df2)``."""
    rng = _r_rng()
    nn = int(n)
    d1 = _recycle(df1, nn)
    d2 = _recycle(df2, nn)
    if np.all(np.asarray(ncp) == 0):
        return rng.rf_n(d1, d2)
    ncpv = _recycle(ncp, nn)
    num = rng.rchisq_n(d1, ncpv) / d1
    den = rng.rchisq_n(d2, np.zeros(nn)) / d2
    return num / den


# chi-squared  (ncp != 0 via ported nmath dnchisq/pnchisq/qnchisq — bit-exact to
# R except a ≤1-ulp residual on the ncp>=80 far-lower-tail, where numpy's long-
# double exp differs from the system expl R links; still far tighter than scipy.)
def dchisq(x, df, ncp=0):
    # central chi-square = gamma(shape=df/2, scale=2) — bit-exact via nmath.
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("dgamma", _nm.dgamma, [x, np.asarray(df, float) / 2.0, 2.0],
                         (False,))
    return _nm._disp("dnchisq", _nm.dnchisq, [x, df, ncp], (False,))


def pchisq(q, df, ncp=0, lower_tail=True):
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("pgamma", _nm.pgamma, [q, np.asarray(df, float) / 2.0, 2.0],
                         (lower_tail, False))
    return _nm._disp("pnchisq", _nm.pnchisq, [q, df, ncp], (lower_tail, False))


def qchisq(p, df, ncp=0, lower_tail=True):
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("qgamma", _nm.qgamma, [p, np.asarray(df, float) / 2.0, 2.0],
                         (lower_tail, False))
    return _nm._disp("qnchisq", _nm.qnchisq, [p, df, ncp], (lower_tail, False))


def rchisq(n, df, ncp=0):
    """R: ``rchisq(n, df, ncp=0)`` — per-element ``rnchisq`` on R's MT stream
    (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    dfv = _recycle(df, nn)
    ncpv = _recycle(ncp, nn)
    return rng.rchisq_n(dfv, ncpv)


# binomial
def dbinom(x, size, prob, log=False):
    """R's ``dbinom`` — bit-exact via ported dbinom (nmath/dbinom.c)."""
    return _nm._disp("dbinom", _nm.dbinom, [x, size, prob], (log,))


def pbinom(q, size, prob, lower_tail=True, log_p=False):
    """R's ``pbinom`` — bit-exact via ported pbinom (nmath/pbinom.c -> pbeta)."""
    return _nm._disp("pbinom", _nm.pbinom, [q, size, prob], (lower_tail, log_p))


def qbinom(p, size, prob, lower_tail=True, log_p=False):
    """R's ``qbinom`` — bit-exact via ported qbinom (nmath discrete search)."""
    return _nm._disp("qbinom", _nm.qbinom, [p, size, prob], (lower_tail, log_p))


def rbinom(n, size, prob):
    """R: ``rbinom(n, size, prob)`` — on R's global MT stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    sz = _recycle(size, nn)
    pr = _recycle(prob, nn)
    return rng.rbinom_n(sz, pr).astype(np.int64)


# poisson  (R uses `lambda`, a Python keyword → spelled `lambda_`)
def dpois(x, lambda_, log=False):
    """R's ``dpois`` — bit-exact via ported dpois (nmath/dpois.c, Loader)."""
    return _nm._disp("dpois", _nm.dpois, [x, lambda_], (log,))


def ppois(q, lambda_, lower_tail=True, log_p=False):
    """R's ``ppois`` — bit-exact via ported ppois (nmath/ppois.c -> pgamma)."""
    return _nm._disp("ppois", _nm.ppois, [q, lambda_], (lower_tail, log_p))


def qpois(p, lambda_, lower_tail=True, log_p=False):
    """R's ``qpois`` — bit-exact via ported qpois (nmath discrete search)."""
    return _nm._disp("qpois", _nm.qpois, [p, lambda_], (lower_tail, log_p))


def rpois(n, lambda_):
    """R: ``rpois(n, lambda)`` — on R's global MT stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    lam = _recycle(lambda_, nn)
    return rng.rpois_n(lam).astype(np.int64)


# uniform  (exact closed form — R nmath/{dunif,punif,qunif}.c, no special functions)
def _scalar_in(*xs) -> bool:
    return all(np.ndim(x) == 0 for x in xs)


def dunif(x, min=0, max=1, log=False):
    """R's ``dunif`` — exact port of nmath/dunif.c."""
    x = np.asarray(x, float)
    a = np.asarray(min, float)
    b = np.asarray(max, float)
    inside = (a <= x) & (x <= b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(inside, -np.log(b - a), -np.inf) if log \
            else np.where(inside, 1.0 / (b - a), 0.0)
    out = np.asarray(out, float)
    out = np.where((b <= a) | np.isnan(x) | np.isnan(a) | np.isnan(b), np.nan, out)
    return float(out) if _scalar_in(x, a, b) else out


def punif(q, min=0, max=1, lower_tail=True, log_p=False):
    """R's ``punif`` — exact port of nmath/punif.c."""
    q = np.asarray(q, float)
    a = np.asarray(min, float)
    b = np.asarray(max, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.clip((q - a) / (b - a), 0.0, 1.0)  # x>=b -> 1, x<=a -> 0
    p = lower if lower_tail else 1.0 - lower
    if log_p:
        with np.errstate(divide="ignore"):
            p = np.log(p)
    p = np.asarray(p, float)
    bad = (b < a) | ~np.isfinite(a) | ~np.isfinite(b) \
        | np.isnan(q) | np.isnan(a) | np.isnan(b)
    p = np.where(bad, np.nan, p)
    return float(p) if _scalar_in(q, a, b) else p


def qunif(p, min=0, max=1, lower_tail=True, log_p=False):
    """R's ``qunif`` — exact port of nmath/qunif.c."""
    p = np.asarray(p, float)
    a = np.asarray(min, float)
    b = np.asarray(max, float)
    # R_DT_qIv: map (lower_tail, log_p) back to the lower-tail identity prob.
    if log_p:
        pv = np.exp(p) if lower_tail else -np.expm1(p)
    else:
        pv = p if lower_tail else 1.0 - p
    out = np.asarray(a + pv * (b - a), float)
    # R_Q_P01_check: probability out of [0,1] (identity scale) -> NaN.
    p01_bad = (pv < 0.0) | (pv > 1.0)
    bad = p01_bad | (b < a) | ~np.isfinite(a) | ~np.isfinite(b) \
        | np.isnan(p) | np.isnan(a) | np.isnan(b)
    out = np.where(bad, np.nan, out)
    return float(out) if _scalar_in(p, a, b) else out


def runif(n, min=0, max=1):
    """R: ``runif(n, min=0, max=1)`` — on R's global MT stream (bit-exact)."""
    u = _r_rng().unif_rand(int(n))
    return min + (max - min) * u


# exponential  (R: rate = 1/scale)
def dexp(x, rate=1, log=False):
    """R's ``dexp`` — bit-exact via ported dexp (nmath/dexp.c)."""
    return _nm._disp("dexp", _nm.dexp, [x, 1.0 / np.asarray(rate, float)], (log,))


def pexp(q, rate=1, lower_tail=True, log_p=False):
    """R's ``pexp`` — bit-exact via ported pexp (nmath/pexp.c)."""
    return _nm._disp("pexp", _nm.pexp, [q, 1.0 / np.asarray(rate, float)],
                     (lower_tail, log_p))


def qexp(p, rate=1, lower_tail=True, log_p=False):
    """R's ``qexp`` — bit-exact via ported qexp (nmath/qexp.c)."""
    return _nm._disp("qexp", _nm.qexp, [p, 1.0 / np.asarray(rate, float)],
                     (lower_tail, log_p))


def rexp(n, rate=1):
    """R: ``rexp(n, rate=1)`` — ``exp_rand()/rate`` on R's MT stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    rt_ = _recycle(rate, nn)
    return rng.exp_rand_n(nn) / rt_


# gamma  (R: shape, rate; ``scale`` overrides if given)
def dgamma(x, shape, rate=1, scale=None, log=False):
    """R's ``dgamma`` — bit-exact via the ported ``dgamma`` (nmath/dgamma.c)."""
    if scale is None:
        scale = 1 / rate
    return _nm._disp("dgamma", _nm.dgamma, [x, shape, scale], (log,))


def pgamma(q, shape, rate=1, scale=None, lower_tail=True, log_p=False):
    """R's ``pgamma`` — bit-exact via the ported ``pgamma`` (nmath/pgamma.c)."""
    if scale is None:
        scale = 1 / rate
    return _nm._disp("pgamma", _nm.pgamma, [q, shape, scale], (lower_tail, log_p))


def qgamma(p, shape, rate=1, scale=None, lower_tail=True, log_p=False):
    """R's ``qgamma`` — bit-exact via the ported ``qgamma`` (nmath/qgamma.c)."""
    if scale is None:
        scale = 1 / rate
    return _nm._disp("qgamma", _nm.qgamma, [p, shape, scale], (lower_tail, log_p))


def rgamma(n, shape, rate=1, scale=None):
    """R: ``rgamma(n, shape, rate=1, scale=1/rate)`` — R's MT stream (bit-exact)."""
    if scale is None:
        scale = 1 / rate
    rng = _r_rng()
    nn = int(n)
    sh = _recycle(shape, nn)
    sc = _recycle(scale, nn)
    return rng.rgamma_n(sh, sc)


# beta
def dbeta(x, shape1, shape2, log=False):
    """R's ``dbeta`` — bit-exact via ported dbeta (nmath/dbeta.c)."""
    return _nm._disp("dbeta", _nm.dbeta, [x, shape1, shape2], (log,))


def pbeta(q, shape1, shape2, lower_tail=True, log_p=False):
    """R's ``pbeta`` — bit-exact via ported pbeta (nmath/toms708 bratio)."""
    return _nm._disp("pbeta", _nm.pbeta, [q, shape1, shape2], (lower_tail, log_p))


def qbeta(p, shape1, shape2, lower_tail=True, log_p=False):
    """R's ``qbeta`` — bit-exact via ported qbeta (nmath/qbeta.c, AS 109)."""
    return _nm._disp("qbeta", _nm.qbeta, [p, shape1, shape2], (lower_tail, log_p))


def rbeta(n, shape1, shape2):
    """R: ``rbeta(n, shape1, shape2)`` — Cheng's BB/BC algorithm on R's MT
    stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    s1 = _recycle(shape1, nn)
    s2 = _recycle(shape2, nn)
    return rng.rbeta_n(s1, s2)


# cauchy
def dcauchy(x, location=0, scale=1, log=False):
    """R's ``dcauchy`` — Cauchy density (nmath/dcauchy.c); bit-exact."""
    return _nm._disp("dcauchy", _nm.dcauchy, [x, location, scale], (log,))


def pcauchy(q, location=0, scale=1, lower_tail=True, log_p=False):
    """R's ``pcauchy`` — Cauchy CDF (nmath/pcauchy.c); bit-exact."""
    return _nm._disp("pcauchy", _nm.pcauchy, [q, location, scale],
                     (lower_tail, log_p))


def qcauchy(p, location=0, scale=1, lower_tail=True, log_p=False):
    """R's ``qcauchy`` — Cauchy quantile (nmath/qcauchy.c); bit-exact."""
    return _nm._disp("qcauchy", _nm.qcauchy, [p, location, scale],
                     (lower_tail, log_p))


def rcauchy(n, location=0, scale=1):
    """R: ``rcauchy(n, location=0, scale=1)`` — ``location + scale*tan(pi*U)``
    on R's MT stream (bit-exact for finite location, scale > 0)."""
    rng = _r_rng()
    nn = int(n)
    loc = _recycle(location, nn)
    sc = _recycle(scale, nn)
    u = np.asarray(rng.unif_rand(nn), dtype=float)
    return loc + sc * np.tan(np.pi * u)


# logistic
def dlogis(x, location=0, scale=1, log=False):
    """R's ``dlogis`` — logistic density (nmath/dlogis.c); bit-exact."""
    return _nm._disp("dlogis", _nm.dlogis, [x, location, scale], (log,))


def plogis(q, location=0, scale=1, lower_tail=True, log_p=False):
    """R's ``plogis`` — logistic CDF (nmath/plogis.c); bit-exact."""
    return _nm._disp("plogis", _nm.plogis, [q, location, scale],
                     (lower_tail, log_p))


def qlogis(p, location=0, scale=1, lower_tail=True, log_p=False):
    """R's ``qlogis`` — logistic quantile (nmath/qlogis.c); bit-exact."""
    return _nm._disp("qlogis", _nm.qlogis, [p, location, scale],
                     (lower_tail, log_p))


def rlogis(n, location=0, scale=1):
    """R: ``rlogis(n, location=0, scale=1)`` — ``location + scale*log(U/(1-U))``
    on R's MT stream (bit-exact for finite location, scale > 0)."""
    rng = _r_rng()
    nn = int(n)
    loc = _recycle(location, nn)
    sc = _recycle(scale, nn)
    u = np.asarray(rng.unif_rand(nn), dtype=float)
    return loc + sc * np.log(u / (1. - u))


# log-normal
def dlnorm(x, meanlog=0, sdlog=1, log=False):
    """R's ``dlnorm`` — log-normal density (nmath/dlnorm.c); bit-exact."""
    return _nm._disp("dlnorm", _nm.dlnorm, [x, meanlog, sdlog], (log,))


def plnorm(q, meanlog=0, sdlog=1, lower_tail=True, log_p=False):
    """R's ``plnorm`` — log-normal CDF (nmath/plnorm.c → pnorm); bit-exact."""
    return _nm._disp("plnorm", _nm.plnorm, [q, meanlog, sdlog],
                     (lower_tail, log_p))


def qlnorm(p, meanlog=0, sdlog=1, lower_tail=True, log_p=False):
    """R's ``qlnorm`` — log-normal quantile (nmath/qlnorm.c → qnorm); bit-exact."""
    return _nm._disp("qlnorm", _nm.qlnorm, [p, meanlog, sdlog],
                     (lower_tail, log_p))


def rlnorm(n, meanlog=0, sdlog=1):
    """R: ``rlnorm(n, meanlog=0, sdlog=1)`` — ``exp(rnorm(meanlog, sdlog))`` on
    R's MT stream (bit-exact)."""
    return np.exp(rnorm(n, meanlog, sdlog))


# weibull
def dweibull(x, shape, scale=1, log=False):
    """R's ``dweibull`` — Weibull density (nmath/dweibull.c); bit-exact."""
    return _nm._disp("dweibull", _nm.dweibull, [x, shape, scale], (log,))


def pweibull(q, shape, scale=1, lower_tail=True, log_p=False):
    """R's ``pweibull`` — Weibull CDF (nmath/pweibull.c); bit-exact."""
    return _nm._disp("pweibull", _nm.pweibull, [q, shape, scale],
                     (lower_tail, log_p))


def qweibull(p, shape, scale=1, lower_tail=True, log_p=False):
    """R's ``qweibull`` — Weibull quantile (nmath/qweibull.c); bit-exact."""
    return _nm._disp("qweibull", _nm.qweibull, [p, shape, scale],
                     (lower_tail, log_p))


def rweibull(n, shape, scale=1):
    """R: ``rweibull(n, shape, scale=1)`` — ``scale*(-log(U))^(1/shape)`` on R's
    MT stream (bit-exact for shape, scale > 0)."""
    rng = _r_rng()
    nn = int(n)
    sh = _recycle(shape, nn)
    sc = _recycle(scale, nn)
    u = np.asarray(rng.unif_rand(nn), dtype=float)
    return sc * np.power(-np.log(u), 1.0 / sh)


# geometric  (R: dgeom(x, prob) — Pr(X=x) = prob*(1-prob)^x, x = 0,1,2,...)
def dgeom(x, prob, log=False):
    """R's ``dgeom`` — geometric density (nmath/dgeom.c); bit-exact."""
    return _nm._disp("dgeom", _nm.dgeom, [x, prob], (log,))


def pgeom(q, prob, lower_tail=True, log_p=False):
    """R's ``pgeom`` — geometric CDF (nmath/pgeom.c); bit-exact."""
    return _nm._disp("pgeom", _nm.pgeom, [q, prob], (lower_tail, log_p))


def qgeom(p, prob, lower_tail=True, log_p=False):
    """R's ``qgeom`` — geometric quantile (nmath/qgeom.c); bit-exact."""
    return _nm._disp("qgeom", _nm.qgeom, [p, prob], (lower_tail, log_p))


def rgeom(n, prob):
    """R: ``rgeom(n, prob)`` — ``rpois(exp_rand()*(1-p)/p)`` on R's MT stream
    (bit-exact). Per-element interleaved exp/pois draws match R's rgeom.c."""
    rng = _r_rng()
    nn = int(n)
    pr = _recycle(prob, nn)
    out = np.empty(nn)
    for i in range(nn):
        p = float(pr[i])
        if not np.isfinite(p) or p <= 0 or p > 1:
            out[i] = np.nan
        else:
            out[i] = rng.rpois(rng.exp_rand() * ((1 - p) / p))
    return out


# Wilcoxon signed-rank distribution (exact; nmath/signrank.c)
def dsignrank(x, n, log=False):
    """R's ``dsignrank`` — density of the Wilcoxon signed-rank statistic."""
    return _nm._vec(lambda xx, nn: _nm.dsignrank(xx, nn, log), x, n)


def psignrank(q, n, lower_tail=True, log_p=False):
    """R's ``psignrank`` — CDF of the Wilcoxon signed-rank statistic."""
    return _nm._vec(lambda qq, nn: _nm.psignrank(qq, nn, lower_tail, log_p), q, n)


def qsignrank(p, n, lower_tail=True, log_p=False):
    """R's ``qsignrank`` — quantile of the Wilcoxon signed-rank statistic."""
    return _nm._vec(lambda pp, nn: _nm.qsignrank(pp, nn, lower_tail, log_p), p, n)


# Wilcoxon rank-sum (Mann-Whitney) distribution (exact; nmath/wilcox.c)
def dwilcox(x, m, n, log=False):
    """R's ``dwilcox`` — density of the Wilcoxon rank-sum (Mann-Whitney) stat."""
    return _nm._vec(lambda xx, mm, nn: _nm.dwilcox(xx, mm, nn, log), x, m, n)


def pwilcox(q, m, n, lower_tail=True, log_p=False):
    """R's ``pwilcox`` — CDF of the Wilcoxon rank-sum (Mann-Whitney) statistic."""
    return _nm._vec(
        lambda qq, mm, nn: _nm.pwilcox(qq, mm, nn, lower_tail, log_p), q, m, n)


def qwilcox(p, m, n, lower_tail=True, log_p=False):
    """R's ``qwilcox`` — quantile of the Wilcoxon rank-sum (Mann-Whitney) stat."""
    return _nm._vec(
        lambda pp, mm, nn: _nm.qwilcox(pp, mm, nn, lower_tail, log_p), p, m, n)


# hypergeometric  (R: dhyper(x, m, n, k) — m white, n black, k drawn)
def dhyper(x, m, n, k, log=False):
    """R's ``dhyper`` — hypergeometric density (nmath/dhyper.c); R-parity."""
    return _nm._disp("dhyper", _nm.dhyper, [x, m, n, k], (log,))


def phyper(q, m, n, k, lower_tail=True, log_p=False):
    """R's ``phyper`` — hypergeometric CDF (nmath/phyper.c); R-parity."""
    return _nm._disp("phyper", _nm.phyper, [q, m, n, k], (lower_tail, log_p))


def qhyper(p, m, n, k, lower_tail=True, log_p=False):
    """R's ``qhyper`` — hypergeometric quantile (nmath/qhyper.c); bit-exact."""
    return _nm._disp("qhyper", _nm.qhyper, [p, m, n, k], (lower_tail, log_p))


# negative binomial  (R accepts EITHER prob OR mu; mu → the (size, mu) kernels)
def dnbinom(x, size, prob=None, mu=None, log=False):
    """R's ``dnbinom(x, size, prob | mu)`` (nmath/dnbinom.c); bit-exact."""
    if mu is not None:
        return _nm._disp("dnbinom_mu", _nm.dnbinom_mu, [x, size, mu], (log,))
    return _nm._disp("dnbinom", _nm.dnbinom, [x, size, prob], (log,))


def pnbinom(q, size, prob=None, mu=None, lower_tail=True, log_p=False):
    """R's ``pnbinom(q, size, prob | mu)`` (nmath/pnbinom.c → pbeta); bit-exact."""
    if mu is not None:
        return _nm._disp("pnbinom_mu", _nm.pnbinom_mu, [q, size, mu],
                         (lower_tail, log_p))
    return _nm._disp("pnbinom", _nm.pnbinom, [q, size, prob], (lower_tail, log_p))


def qnbinom(p, size, prob=None, mu=None, lower_tail=True, log_p=False):
    """R's ``qnbinom(p, size, prob | mu)`` (nmath/qnbinom.c); bit-exact."""
    if mu is not None:
        return _nm._disp("qnbinom_mu", _nm.qnbinom_mu, [p, size, mu],
                         (lower_tail, log_p))
    return _nm._disp("qnbinom", _nm.qnbinom, [p, size, prob], (lower_tail, log_p))


# studentized range (Tukey)  — R exposes only the CDF / quantile (no d*/r*)
def ptukey(q, nmeans, df, nranges=1, lower_tail=True, log_p=False):
    """R's ``ptukey`` — CDF of the studentized range distribution.

    ``nmeans`` is the number of groups/treatments, ``df`` the error degrees of
    freedom, ``nranges`` the number of independent ranges (default 1). R-parity
    via the ported nmath ``ptukey`` (Copenhaver-Holland Gauss-Legendre
    quadrature); see the module docstring on the Rust f64 fast path."""
    return _nm._disp("ptukey", _nm.ptukey, [q, nranges, nmeans, df],
                     (lower_tail, log_p))


def qtukey(p, nmeans, df, nranges=1, lower_tail=True, log_p=False):
    """R's ``qtukey`` — quantile of the studentized range distribution.

    Inverse of :func:`ptukey` (secant iteration off an AS 70 start value);
    R-parity via the ported nmath ``qtukey``."""
    return _nm._disp("qtukey", _nm.qtukey, [p, nranges, nmeans, df],
                     (lower_tail, log_p))


def set_seed(seed):
    """R: ``set.seed()`` — seed the process-global R Mersenne-Twister stream.

    Every ``hea.R`` random draw (``runif`` / ``rnorm`` / ``sample`` / ``rpois`` /
    ``rgamma`` / ``rbinom`` / ``rexp`` / ``rchisq`` / ``rt`` / ``rf`` / ``rbeta``)
    routes through this one stream, so ``set_seed(k); <draw>`` is **bit-exact** to
    R's ``set.seed(k); <draw>``. For the low-level stream object see
    :class:`hea.R.rng.RMersenneTwister`.
    """
    global _R_RNG
    _R_RNG = RMersenneTwister(int(seed))


def sample(x, size=None, replace=False, prob=None):
    """R: ``sample()`` — random permutation or draw.

    Forms:

    - ``sample(x)`` where ``x`` is a vector → permute ``x``.
    - ``sample(n)`` where ``n`` is a scalar int → permute ``1:n``
      (R's "convenience" form).
    - ``sample(x, size)`` → draw ``size`` without replacement.
    - ``sample(x, size, replace=True)`` → with replacement.
    - ``sample(x, size, prob=p)`` → weighted draw.

    Names from a :class:`hea.NamedVector` are preserved through
    permutation / draw.
    """

    if isinstance(x, NamedVector):
        names = x.names
        values = x.values
    elif isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        names = None
        values = np.arange(1, int(x) + 1)
    else:
        names = None
        values = np.asarray(x).ravel()

    n = len(values)
    if size is None:
        size = n
    if prob is not None:
        idx = _r_rng().sample_prob(np.asarray(prob, dtype=float),
                                   int(size), replace=replace)
    else:
        idx = _r_rng().sample_int(n, int(size), replace=replace)
    if names is not None:
        return NamedVector([names[i] for i in idx], values[idx])
    return values[idx]


# ======================================================================
# Combinatorial / multivariate distributions (2nd-half Tier-1 add-ons).
# The r* generators draw from the process-global R MT stream (bit-exact to
# set.seed); the closed-form pbirthday/qbirthday/dmultinom are pure R ports.
# ======================================================================

# negative binomial variates (R: rnbinom(n, size, prob | mu); rnbinom.c)
def rnbinom(n, size, prob=None, mu=None):
    """R: ``rnbinom(n, size, prob | mu)`` — negative-binomial variates,
    ``rpois(rgamma(size, ·))`` on R's MT stream (bit-exact). Supply ``prob`` or
    ``mu`` (mu is the ``rnbinom(size, mu)`` parameterisation)."""
    rng = _r_rng()
    nn = int(n)
    sz = _recycle(size, nn)
    if mu is not None:
        mm = _recycle(mu, nn)
        return np.array([rng.rnbinom(float(sz[i]), float(mm[i]))
                         for i in range(nn)])
    pr = _recycle(prob, nn)
    return np.array([rng.rnbinom_prob(float(sz[i]), float(pr[i]))
                     for i in range(nn)])


# hypergeometric variates (R: rhyper(nn, m, n, k); rhyper.c, H2PE)
def rhyper(nn, m, n, k):
    """R: ``rhyper(nn, m, n, k)`` — ``nn`` hypergeometric variates: white balls
    drawn when ``k`` are taken from ``m`` white + ``n`` black, on R's MT stream
    (rhyper.c, Kachitvichyanukul-Schmeiser H2PE); bit-exact."""
    rng = _r_rng()
    ln = int(nn)
    ms = _recycle(m, ln)
    ns = _recycle(n, ln)
    ks = _recycle(k, ln)
    return np.array([rng.rhyper(float(ms[i]), float(ns[i]), float(ks[i]))
                     for i in range(ln)])


# Wilcoxon signed-rank + rank-sum variates (signrank.c / wilcox.c)
def rsignrank(nn, n):
    """R: ``rsignrank(nn, n)`` — ``nn`` Wilcoxon signed-rank variates for a
    sample of size ``n``, on R's MT stream (signrank.c); bit-exact."""
    rng = _r_rng()
    m = int(nn)
    ns = _recycle(n, m)
    return np.array([rng.rsignrank(float(ns[i])) for i in range(m)])


def rwilcox(nn, m, n):
    """R: ``rwilcox(nn, m, n)`` — ``nn`` Wilcoxon rank-sum variates for samples
    of size ``m`` and ``n``, on R's MT stream (wilcox.c); bit-exact."""
    rng = _r_rng()
    ln = int(nn)
    ms = _recycle(m, ln)
    ns = _recycle(n, ln)
    return np.array([rng.rwilcox(float(ms[i]), float(ns[i])) for i in range(ln)])


# multinomial variates (R: rmultinom(n, size, prob); rmultinom.c)
def rmultinom(n, size, prob):
    """R: ``rmultinom(n, size, prob)`` — a (K x n) integer matrix of independent
    Multinomial(size, prob) columns, on R's MT stream (rmultinom.c); bit-exact.
    ``prob`` is normalised via FixupProb (as R does)."""
    return _r_rng().rmultinom(int(n), int(size), prob)


# random 2-way contingency tables (R: r2dtable(n, r, c); rcont.c, AS 159)
def r2dtable(n, r, c):
    """R: ``r2dtable(n, r, c)`` — ``n`` random 2-way tables with fixed row
    (``r``) and column (``c``) margins, on R's MT stream (rcont.c, AS 159);
    bit-exact. Returns a list of integer matrices."""
    r = np.asarray(r, dtype=np.int64).ravel()
    c = np.asarray(c, dtype=np.int64).ravel()
    if r.size <= 1 or np.any(r < 0):
        raise ValueError("invalid argument 'r'")
    if c.size <= 1 or np.any(c < 0):
        raise ValueError("invalid argument 'c'")
    if int(r.sum()) != int(c.sum()):
        raise ValueError("arguments 'r' and 'c' must have the same sums")
    ntotal = int(r.sum())
    fact = [0.0] * (ntotal + 1)
    for i in range(1, ntotal + 1):
        fact[i] = _nm._lgammafn(float(i + 1))
    rng = _r_rng()
    return [np.array(rng.rcont2(r.tolist(), c.tolist(), ntotal, fact),
                     dtype=np.int64)
            for _ in range(int(n))]


# Wishart matrices (R: rWishart(n, df, Sigma); rWishart.c, Bartlett)
def rWishart(n, df, Sigma):
    """R: ``rWishart(n, df, Sigma)`` — ``n`` draws from Wishart(df, Sigma) on R's
    MT stream (rWishart.c, Bartlett decomposition). Returns a (p, p, n) array.
    The RNG stream is bit-exact; the Cholesky/crossprod carry platform-BLAS
    rounding (a few ulp vs R's reference BLAS, not a port discrepancy)."""
    from scipy.linalg import cholesky
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("'Sigma' must be a square, real matrix")
    p = Sigma.shape[0]
    nn = int(n)
    if nn <= 0:
        nn = 1
    r_chol = cholesky(Sigma, lower=False)          # dpotrf "U": R'R = Sigma
    rng = _r_rng()
    out = np.empty((p, p, nn), dtype=float)
    for j in range(nn):
        a = rng.std_rwishart_factor(float(df), p)  # upper-tri Bartlett factor
        m = a @ r_chol                             # tmp = A · R  (dtrmm)
        out[:, :, j] = m.T @ m                     # crossprod (dsyrk)
    return out


# --- closed-form (no RNG): birthday problem + multinomial density -----------
def pbirthday(n, classes=365, coincident=2):
    """R's ``pbirthday(n, classes, coincident)`` (birthday.R) — probability of a
    coincidence of at least ``coincident`` among ``n`` items over ``classes``
    equiprobable categories (Diaconis-Mosteller); bit-exact scalar."""
    k = coincident
    c = classes
    if k < 2:
        return 1.0
    if k == 2:
        acc = np.longdouble(1.0)                    # R's prod → long double
        for i in range(int(n)):
            acc *= np.longdouble((c - i) / c)
        return float(1.0 - acc)
    if k > n:
        return 0.0
    if n > c * (k - 1):
        return 1.0
    lhs = n * math.exp(-n / (c * k)) / (1 - n / (c * (k + 1))) ** (1.0 / k)
    lxx = k * math.log(lhs) - (k - 1) * math.log(c) - _nm._lgammafn(k + 1)
    return float(-math.expm1(-math.exp(lxx)))


def qbirthday(prob=0.5, classes=365, coincident=2):
    """R's ``qbirthday(prob, classes, coincident)`` (birthday.R) — smallest ``n``
    with ``pbirthday(n) >= prob`` (crude Diaconis-Mosteller inversion, then a
    linear search); returns an int, bit-exact to R."""
    k = coincident
    c = classes
    p = prob
    if p <= 0:
        return 1
    if p >= 1:
        return int(c * (k - 1) + 1)
    nn = math.exp(((k - 1) * math.log(c) + _nm._lgammafn(k + 1)
                   + math.log(-math.log1p(-p))) / k)
    nn = math.ceil(nn)
    if pbirthday(nn, c, k) < prob:
        nn += 1
        while pbirthday(nn, c, k) < prob:
            nn += 1
    elif pbirthday(nn - 1, c, k) >= prob:
        nn -= 1
        while pbirthday(nn - 1, c, k) >= prob:
            nn -= 1
    return int(nn)


def dmultinom(x, size=None, prob=None, log=False):
    """R's ``dmultinom(x, size=None, prob, log=False)`` (distn.R) — the
    multinomial pmf at count vector ``x``; bit-exact (log-gamma via nmath,
    R's long-double sum)."""
    x = np.asarray(x, dtype=float).ravel()
    prob = np.asarray(prob, dtype=float).ravel()
    if x.size != prob.size:
        raise ValueError("x[] and prob[] must be equal length vectors.")
    if not np.all(np.isfinite(prob)) or np.any(prob < 0):
        raise ValueError("probabilities must be finite, non-negative and not all 0")
    s = float(prob.sum())
    if s == 0:
        raise ValueError("probabilities must be finite, non-negative and not all 0")
    prob = prob / s
    x = np.floor(x + 0.5).astype(np.int64)         # as.integer(x + 0.5)
    if np.any(x < 0):
        raise ValueError("'x' must be non-negative")
    total = int(x.sum())
    if size is None:
        size = total
    elif size != total:
        raise ValueError("size != sum(x), i.e. one is wrong")
    i0 = prob == 0
    if i0.any():
        if np.any(x[i0] != 0):
            return -math.inf if log else 0.0
        if i0.all():
            return 0.0 if log else 1.0
        x = x[~i0]
        prob = prob[~i0]
    acc = np.longdouble(0.0)                        # R: sum(...) in long double
    for xi, pi in zip(x, prob):
        acc += np.longdouble(xi * math.log(pi) - _nm._lgammafn(float(xi) + 1))
    r = _nm._lgammafn(float(size) + 1) + float(acc)
    return r if log else math.exp(r)
