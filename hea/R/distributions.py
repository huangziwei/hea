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
        return pl.int_range(0, n).map_elements(
            lambda _: _sps.norm.rvs(loc=mean, scale=sd),
            return_dtype=pl.Float64,
        )
    return _sps.norm.rvs(loc=mean, scale=sd, size=int(n))


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
    if ncp == 0:
        return _sps.t.rvs(df=df, size=int(n))
    return _sps.nct.rvs(df=df, nc=ncp, size=int(n))


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
    if ncp == 0:
        return _sps.f.rvs(df1, df2, size=int(n))
    return _sps.ncf.rvs(df1, df2, nc=ncp, size=int(n))


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
    if ncp == 0:
        return _sps.chi2.rvs(df=df, size=int(n))
    return _sps.ncx2.rvs(df=df, nc=ncp, size=int(n))


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
    return _sps.binom.rvs(n=size, p=prob, size=int(n))


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
    return _sps.poisson.rvs(mu=lambda_, size=int(n))


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
    return _sps.uniform.rvs(loc=min, scale=max - min, size=int(n))


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
    return _sps.expon.rvs(scale=1 / rate, size=int(n))


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
    if scale is None:
        scale = 1 / rate
    return _sps.gamma.rvs(a=shape, scale=scale, size=int(n))


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
    return _sps.beta.rvs(a=shape1, b=shape2, size=int(n))


def set_seed(seed):
    """R: ``set.seed()`` — seeds numpy's global RNG (used by ``r*`` here).

    For bit-exact reproduction of R's RNG (mgcv parity), see
    :class:`hea.models.bam.RMersenneTwister`. This wrapper is for
    ordinary reproducibility.
    """
    np.random.seed(int(seed))


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
    idx = np.random.choice(n, size=int(size), replace=replace,
                           p=(np.asarray(prob, dtype=float) if prob is not None else None))
    if names is not None:
        return NamedVector([names[i] for i in idx], values[idx])
    return values[idx]
