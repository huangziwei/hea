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
