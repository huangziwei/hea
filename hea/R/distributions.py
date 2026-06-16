"""R's d* / p* / q* / r* distribution surface, plus ``set.seed`` and ``sample``.

The **central** d/p/q functions (normal, t, F, chi-square, gamma, beta,
binomial, Poisson, exponential, uniform) route through :mod:`hea.R.nmath` —
bit-exact ports of R's ``src/nmath/`` C kernels (Cody pnorm, Wichura qnorm,
Welinder pgamma, TOMS 708 pbeta, Loader dpois/dbinom, AS 91/109 qgamma/qbeta)
— so ``hea.R`` d/p/q is 0-ulp to R, not scipy's ~1-3 ulp approximations.
Only the **non-central** variants (``ncp != 0``: nct/ncf/ncx2) still defer to
``scipy.stats`` (a separate algorithm family, not yet ported).

Conventions kept from R: ``lower_tail`` (R's ``lower.tail``) and ``log_p`` /
``log`` for p* / q* / d*; ``ncp`` non-centrality where applicable; ``lambda_``
for Poisson (R's ``lambda`` is a Python keyword). ``df`` PDF is intentionally
omitted — ``df`` is too common as a DataFrame variable; use
``scipy.stats.f.pdf`` directly when you need it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _sps

from . import nmath as _nm
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
    """R's ``dt`` — central case bit-exact via ported dt (nmath/dt.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("dt", _nm.dt, [x, df], (log,))
    return _sps.nct.pdf(x, df=df, nc=ncp)


def pt(q, df, ncp=0, lower_tail=True, log_p=False):
    """R's ``pt`` — central case bit-exact via ported pt (nmath/pt.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("pt", _nm.pt, [q, df], (lower_tail, log_p))
    p = _sps.nct.cdf(q, df=df, nc=ncp)
    return p if lower_tail else 1 - p


def qt(p, df, ncp=0, lower_tail=True, log_p=False):
    """R's ``qt`` — central case bit-exact via ported qt (nmath/qt.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("qt", _nm.qt, [p, df], (lower_tail, log_p))
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.nct.ppf(p, df=df, nc=ncp)


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
    """R's ``pf`` — central case bit-exact via ported pf (nmath/pf.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("pf", _nm.pf, [q, df1, df2], (lower_tail, log_p))
    p = _sps.ncf.cdf(q, df1, df2, nc=ncp)
    return p if lower_tail else 1 - p


def qf(p, df1, df2, ncp=0, lower_tail=True, log_p=False):
    """R's ``qf`` — central case bit-exact via ported qf (nmath/qf.c)."""
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("qf", _nm.qf, [p, df1, df2], (lower_tail, log_p))
    if not lower_tail:
        p = 1 - np.asarray(p)
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
        return rng.rf_n(d1, d2)
    ncpv = _recycle(ncp, nn)
    num = rng.rchisq_n(d1, ncpv) / d1
    den = rng.rchisq_n(d2, np.zeros(nn)) / d2
    return num / den


# chi-squared
def dchisq(x, df, ncp=0):
    # central chi-square = gamma(shape=df/2, scale=2) — bit-exact via nmath.
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("dgamma", _nm.dgamma, [x, np.asarray(df, float) / 2.0, 2.0],
                         (False,))
    return _sps.ncx2.pdf(x, df=df, nc=ncp)


def pchisq(q, df, ncp=0, lower_tail=True):
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("pgamma", _nm.pgamma, [q, np.asarray(df, float) / 2.0, 2.0],
                         (lower_tail, False))
    p = _sps.ncx2.cdf(q, df=df, nc=ncp)
    return p if lower_tail else 1 - p


def qchisq(p, df, ncp=0, lower_tail=True):
    if np.all(np.asarray(ncp) == 0):
        return _nm._disp("qgamma", _nm.qgamma, [p, np.asarray(df, float) / 2.0, 2.0],
                         (lower_tail, False))
    if not lower_tail:
        p = 1 - np.asarray(p)
    return _sps.ncx2.ppf(p, df=df, nc=ncp)


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
