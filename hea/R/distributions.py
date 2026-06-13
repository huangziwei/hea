"""R's d* / p* / q* / r* distribution wrappers around scipy.stats, plus
``set.seed`` and ``sample``.

Conventions kept from R: ``lower_tail`` (R's ``lower.tail``) for p* / q*;
``ncp`` non-centrality where applicable; ``lambda_`` for Poisson (R's
``lambda`` is a Python keyword). ``df`` PDF is intentionally omitted —
``df`` is too common as a DataFrame variable; use ``scipy.stats.f.pdf``
directly when you need it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _sps

from ._shared import NamedVector
from .rng import RMersenneTwister

# --- Process-global R Mersenne-Twister backing the r* / sample surface --------
# R keeps ONE global RNG; ``set_seed`` (R's ``set.seed``) reseeds it and the
# ported r* families draw from it, so ``set_seed(k); runif(n)`` is bit-exact to
# R's ``set.seed(k); runif(n)``. Families R hasn't been ported for (rt / rbeta /
# rchisq / rf, weighted ``sample``) stay on scipy/numpy — reproducible under
# ``set_seed`` but not R-bit-exact (documented per-function).
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
def dnorm(x, mean=0, sd=1):
    return _sps.norm.pdf(x, loc=mean, scale=sd)


def pnorm(q, mean=0, sd=1, lower_tail=True):
    p = _sps.norm.cdf(q, loc=mean, scale=sd)
    return p if lower_tail else 1 - p


def qnorm(p, mean=0, sd=1, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.norm.ppf(p, loc=mean, scale=sd)


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
def dt(x, df, ncp=0):
    if ncp == 0:
        return _sps.t.pdf(x, df=df)
    return _sps.nct.pdf(x, df=df, nc=ncp)


def pt(q, df, ncp=0, lower_tail=True):
    if ncp == 0:
        p = _sps.t.cdf(q, df=df)
    else:
        p = _sps.nct.cdf(q, df=df, nc=ncp)
    return p if lower_tail else 1 - p


def qt(p, df, ncp=0, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    if ncp == 0:
        return _sps.t.ppf(p, df=df)
    return _sps.nct.ppf(p, df=df, nc=ncp)


def rt(n, df, ncp=0):
    """R: ``rt(n, df, ncp=0)`` on R's MT stream (bit-exact). Central: per-element
    ``norm_rand()/sqrt(rchisq(df)/df)``. Noncentral: R's block form
    ``rnorm(n, ncp)/sqrt(rchisq(n, df)/df)``."""
    rng = _r_rng()
    nn = int(n)
    if np.all(np.asarray(ncp) == 0):
        dfv = _recycle(df, nn)
        return np.array([rng.rt(float(dfv[i])) for i in range(nn)])
    num = rng.rnorm(nn, mean=_recycle(ncp, nn))
    dfv = _recycle(df, nn)
    den = np.array([rng.rchisq(float(dfv[i])) for i in range(nn)])
    return num / np.sqrt(den / dfv)


# F  (df() PDF intentionally omitted — clashes with `df` variable name)
def pf(q, df1, df2, ncp=0, lower_tail=True):
    if ncp == 0:
        p = _sps.f.cdf(q, df1, df2)
    else:
        p = _sps.ncf.cdf(q, df1, df2, nc=ncp)
    return p if lower_tail else 1 - p


def qf(p, df1, df2, ncp=0, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    if ncp == 0:
        return _sps.f.ppf(p, df1, df2)
    return _sps.ncf.ppf(p, df1, df2, nc=ncp)


def rf(n, df1, df2, ncp=0):
    """R: ``rf(n, df1, df2, ncp=0)`` on R's MT stream (bit-exact). Central:
    per-element. Noncentral: R's block form
    ``(rchisq(n, df1, ncp)/df1)/(rchisq(n, df2)/df2)``."""
    rng = _r_rng()
    nn = int(n)
    d1 = _recycle(df1, nn)
    d2 = _recycle(df2, nn)
    if np.all(np.asarray(ncp) == 0):
        return np.array([rng.rf(float(d1[i]), float(d2[i])) for i in range(nn)])
    ncpv = _recycle(ncp, nn)
    num = np.array([rng.rchisq(float(d1[i]), float(ncpv[i]))
                    for i in range(nn)]) / d1
    den = np.array([rng.rchisq(float(d2[i])) for i in range(nn)]) / d2
    return num / den


# chi-squared
def dchisq(x, df, ncp=0):
    if ncp == 0:
        return _sps.chi2.pdf(x, df=df)
    return _sps.ncx2.pdf(x, df=df, nc=ncp)


def pchisq(q, df, ncp=0, lower_tail=True):
    if ncp == 0:
        p = _sps.chi2.cdf(q, df=df)
    else:
        p = _sps.ncx2.cdf(q, df=df, nc=ncp)
    return p if lower_tail else 1 - p


def qchisq(p, df, ncp=0, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    if ncp == 0:
        return _sps.chi2.ppf(p, df=df)
    return _sps.ncx2.ppf(p, df=df, nc=ncp)


def rchisq(n, df, ncp=0):
    """R: ``rchisq(n, df, ncp=0)`` — per-element ``rnchisq`` on R's MT stream
    (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    dfv = _recycle(df, nn)
    ncpv = _recycle(ncp, nn)
    return np.array([rng.rchisq(float(dfv[i]), float(ncpv[i]))
                     for i in range(nn)])


# binomial
def dbinom(x, size, prob):
    return _sps.binom.pmf(x, size, prob)


def pbinom(q, size, prob, lower_tail=True):
    p = _sps.binom.cdf(q, size, prob)
    return p if lower_tail else 1 - p


def qbinom(p, size, prob, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.binom.ppf(p, size, prob)


def rbinom(n, size, prob):
    """R: ``rbinom(n, size, prob)`` — on R's global MT stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    sz = _recycle(size, nn)
    pr = _recycle(prob, nn)
    return np.array([rng.rbinom(int(round(sz[i])), float(pr[i]))
                     for i in range(nn)], dtype=np.int64)


# poisson  (R uses `lambda`, a Python keyword → spelled `lambda_`)
def dpois(x, lambda_):
    return _sps.poisson.pmf(x, mu=lambda_)


def ppois(q, lambda_, lower_tail=True):
    p = _sps.poisson.cdf(q, mu=lambda_)
    return p if lower_tail else 1 - p


def qpois(p, lambda_, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.poisson.ppf(p, mu=lambda_)


def rpois(n, lambda_):
    """R: ``rpois(n, lambda)`` — on R's global MT stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    lam = _recycle(lambda_, nn)
    return np.array([rng.rpois(float(lam[i])) for i in range(nn)],
                    dtype=np.int64)


# uniform
def dunif(x, min=0, max=1):
    return _sps.uniform.pdf(x, loc=min, scale=max - min)


def punif(q, min=0, max=1, lower_tail=True):
    p = _sps.uniform.cdf(q, loc=min, scale=max - min)
    return p if lower_tail else 1 - p


def qunif(p, min=0, max=1, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.uniform.ppf(p, loc=min, scale=max - min)


def runif(n, min=0, max=1):
    """R: ``runif(n, min=0, max=1)`` — on R's global MT stream (bit-exact)."""
    u = _r_rng().unif_rand(int(n))
    return min + (max - min) * u


# exponential  (R: rate = 1/scale)
def dexp(x, rate=1):
    return _sps.expon.pdf(x, scale=1 / rate)


def pexp(q, rate=1, lower_tail=True):
    p = _sps.expon.cdf(q, scale=1 / rate)
    return p if lower_tail else 1 - p


def qexp(p, rate=1, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.expon.ppf(p, scale=1 / rate)


def rexp(n, rate=1):
    """R: ``rexp(n, rate=1)`` — ``exp_rand()/rate`` on R's MT stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    rt_ = _recycle(rate, nn)
    return np.array([rng.exp_rand() / rt_[i] for i in range(nn)])


# gamma  (R: shape, rate; ``scale`` overrides if given)
def dgamma(x, shape, rate=1, scale=None):
    if scale is None:
        scale = 1 / rate
    return _sps.gamma.pdf(x, a=shape, scale=scale)


def pgamma(q, shape, rate=1, scale=None, lower_tail=True):
    if scale is None:
        scale = 1 / rate
    p = _sps.gamma.cdf(q, a=shape, scale=scale)
    return p if lower_tail else 1 - p


def qgamma(p, shape, rate=1, scale=None, lower_tail=True):
    if scale is None:
        scale = 1 / rate
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.gamma.ppf(p, a=shape, scale=scale)


def rgamma(n, shape, rate=1, scale=None):
    """R: ``rgamma(n, shape, rate=1, scale=1/rate)`` — R's MT stream (bit-exact)."""
    if scale is None:
        scale = 1 / rate
    rng = _r_rng()
    nn = int(n)
    sh = _recycle(shape, nn)
    sc = _recycle(scale, nn)
    return np.array([rng.rgamma(float(sh[i]), scale=float(sc[i]))
                     for i in range(nn)])


# beta
def dbeta(x, shape1, shape2):
    return _sps.beta.pdf(x, a=shape1, b=shape2)


def pbeta(q, shape1, shape2, lower_tail=True):
    p = _sps.beta.cdf(q, a=shape1, b=shape2)
    return p if lower_tail else 1 - p


def qbeta(p, shape1, shape2, lower_tail=True):
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.beta.ppf(p, a=shape1, b=shape2)


def rbeta(n, shape1, shape2):
    """R: ``rbeta(n, shape1, shape2)`` — Cheng's BB/BC algorithm on R's MT
    stream (bit-exact)."""
    rng = _r_rng()
    nn = int(n)
    s1 = _recycle(shape1, nn)
    s2 = _recycle(shape2, nn)
    return np.array([rng.rbeta(float(s1[i]), float(s2[i])) for i in range(nn)])


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
