"""R-like free-function API for hea.

Designed for ``from hea.R import *`` so R muscle memory works directly:
``head(df)``, ``pnorm(1.96)``, ``mean(x)``, ``sd(x)``, ``factor(s)``, etc.

Design rules
------------
* **No builtin shadowing.** Names that would clobber Python builtins on
  ``import *`` (``range``, ``min``, ``max``, ``sum``, ``round``, ``abs``,
  ``format``, ``print``, ``list``, ``dict``, ``set``, ``type``, ``len``,
  ``filter``, ``map``, ``zip``, ``sorted``, ``reversed``, ``all``, ``any``,
  …) are intentionally NOT exported. Use numpy / Python equivalents.
* **Polars name collisions are OK.** No one does ``from polars import *``,
  so ``head``, ``mean``, ``var``, ``filter``, ``sort`` etc. are safe to
  redefine here.
* **R's ``c()`` is skipped.** A single-letter glob would clobber loop
  variables. Use ``np.array([...])`` or a Python list.
* **R's ``df()`` (PDF of the F distribution) is skipped.** ``df`` is too
  commonly used as a DataFrame variable. Use ``scipy.stats.f.pdf``;
  ``pf`` / ``qf`` / ``rf`` are exposed for CDF / quantile / random.
* **Indexing functions return 0-based indices** (``which``, ``which_max``,
  ``which_min``, ``order``) to match Python conventions. ``seq_len`` /
  ``seq_along`` / ``seq(n)`` keep R's 1-based values, since R users
  reach for them as values (e.g. ``1:n`` printouts), not as indices.
* **R parameter names preserved where possible.** ``mean=`` / ``sd=`` /
  ``df=`` / ``shape=`` / ``rate=`` / ``prob=``. R's ``lower.tail``
  becomes ``lower_tail``. R's ``na.rm`` becomes ``na_rm``. R's
  ``lambda=`` becomes ``lambda_=`` (Python keyword).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as _sps

from .data import factor

__all__ = [
    # shape / preview
    "head", "tail", "nrow", "ncol", "dim", "length",
    "names", "colnames", "summary", "glimpse",
    "complete_cases", "na_omit",
    # vector helpers
    "seq", "seq_len", "seq_along",
    "rev", "sort", "order",
    "which", "which_max", "which_min",
    "cumsum", "cumprod", "cummax", "cummin", "diff",
    "unique", "duplicated", "tabulate",
    # reductions (R defaults: sd/var use N-1)
    "mean", "median", "var", "sd", "quantile", "cor", "cov",
    # coercion / predicates
    "as_numeric", "as_integer", "as_character", "as_logical",
    "is_na", "is_null", "is_finite", "is_numeric", "is_factor",
    "factor", "levels", "nlevels",
    # distributions: d/p/q/r families
    "dnorm", "pnorm", "qnorm", "rnorm",
    "dt", "pt", "qt", "rt",
    "pf", "qf", "rf",
    "dchisq", "pchisq", "qchisq", "rchisq",
    "dbinom", "pbinom", "qbinom", "rbinom",
    "dpois", "ppois", "qpois", "rpois",
    "dunif", "punif", "qunif", "runif",
    "dexp", "pexp", "qexp", "rexp",
    "dgamma", "pgamma", "qgamma", "rgamma",
    "dbeta", "pbeta", "qbeta", "rbeta",
    "set_seed",
    # model generics (lm / glm / gam / bam / lme)
    "coef", "coefficients", "fixef", "ranef",
    "resid", "residuals", "fitted", "fitted_values",
    "predict", "confint", "vcov",
    "logLik", "deviance", "nobs", "df_residual",
    "formula", "model_matrix", "model_frame",
    "AIC", "BIC",
]


# ---- shape / preview ------------------------------------------------

def head(x, n=6):
    """R: first ``n`` rows / elements. Dispatches to ``x.head(n)`` if defined."""
    if hasattr(x, "head"):
        return x.head(n)
    return list(x)[:n]


def tail(x, n=6):
    """R: last ``n`` rows / elements. Dispatches to ``x.tail(n)`` if defined."""
    if hasattr(x, "tail"):
        return x.tail(n)
    seq_ = list(x)
    return seq_[-n:] if n > 0 else seq_[:n]


def nrow(df):
    """R: number of rows."""
    if isinstance(df, pl.DataFrame):
        return df.height
    if hasattr(df, "shape") and df.shape:
        return df.shape[0]
    return len(df)


def ncol(df):
    """R: number of columns."""
    if isinstance(df, pl.DataFrame):
        return df.width
    if hasattr(df, "shape") and len(df.shape) > 1:
        return df.shape[1]
    return 1


def dim(df):
    """R: ``(nrow, ncol)``."""
    return (nrow(df), ncol(df))


def length(x):
    """R: ``length()`` — elements for a vector, columns for a data.frame."""
    if isinstance(x, pl.DataFrame):
        return x.width
    return len(x)


def colnames(df):
    """R: column names of a data.frame."""
    return list(df.columns)


def names(x):
    """R: names of a data.frame (cols), Series (its name), or dict (keys)."""
    if isinstance(x, pl.DataFrame):
        return list(x.columns)
    if isinstance(x, pl.Series):
        return x.name
    if isinstance(x, dict):
        return list(x.keys())
    return None


def summary(x, **kwargs):
    """R: ``summary()`` — dispatches to ``x.summary(**kwargs)``.

    Works on hea models (``lm``/``glm``/``gam``/``lme``/``bam``) and on
    ``hea.DataFrame``. For raw arrays / Series, wrap first:
    ``hea.tbl(pl.DataFrame({"x": arr})).summary()``.
    """
    if hasattr(x, "summary"):
        return x.summary(**kwargs)
    raise TypeError(
        f"summary(): {type(x).__name__} has no .summary() method"
    )


def glimpse(df, **kwargs):
    """dplyr: ``glimpse()`` — wide preview. Dispatches to ``.glimpse()``."""
    if hasattr(df, "glimpse"):
        return df.glimpse(**kwargs)
    raise TypeError(
        f"glimpse(): {type(df).__name__} has no .glimpse() method"
    )


def complete_cases(df):
    """R: boolean vector — True for rows with no NA / null."""
    if isinstance(df, pl.DataFrame):
        return ~df.select(
            pl.any_horizontal(pl.all().is_null())
        ).to_series()
    if isinstance(df, pl.Series):
        return ~df.is_null()
    arr = np.asarray(df)
    if arr.dtype.kind not in "fc":
        n = arr.shape[0] if arr.ndim else 1
        return np.ones(n, dtype=bool)
    if arr.ndim == 1:
        return ~np.isnan(arr)
    return ~np.isnan(arr).any(axis=tuple(range(1, arr.ndim)))


def na_omit(df):
    """R: drop rows / elements with any NA / null."""
    if isinstance(df, (pl.DataFrame, pl.Series)):
        return df.drop_nulls()
    arr = np.asarray(df)
    if arr.dtype.kind not in "fc":
        return arr
    if arr.ndim == 1:
        return arr[~np.isnan(arr)]
    return arr[~np.isnan(arr).any(axis=tuple(range(1, arr.ndim)))]


# ---- vector helpers -------------------------------------------------

def seq(*args, by=None, length_out=None, along_with=None):
    """R: ``seq()`` — flexible sequence constructor.

    Supports the common R call shapes:

    * ``seq(n)`` → ``1, 2, …, n`` (1-based, matches R)
    * ``seq(from, to)`` → ``from, from+1, …, to`` (inclusive)
    * ``seq(from, to, by=step)`` → step-spaced, inclusive
    * ``seq(from, to, length_out=n)`` → ``n`` evenly spaced
    * ``seq(along_with=x)`` → ``1, …, len(x)``
    """
    if along_with is not None:
        return np.arange(1, len(along_with) + 1)
    if length_out is not None:
        if len(args) == 0:
            return np.arange(1, int(length_out) + 1)
        if len(args) == 1:
            return np.linspace(1, args[0], int(length_out))
        return np.linspace(args[0], args[1], int(length_out))
    if len(args) == 0:
        raise ValueError("seq(): need at least one positional argument")
    if len(args) == 1:
        return np.arange(1, int(args[0]) + 1)
    start, stop = args[0], args[1]
    step = by if by is not None else (1 if stop >= start else -1)
    n_steps = int(np.floor((stop - start) / step + 1e-10)) + 1
    return start + np.arange(n_steps) * step


def seq_len(n):
    """R: ``seq_len(n)`` → ``1, 2, …, n`` (1-based, matches R)."""
    return np.arange(1, int(n) + 1)


def seq_along(x):
    """R: ``seq_along(x)`` → ``1, 2, …, len(x)`` (1-based, matches R)."""
    return np.arange(1, len(x) + 1)


def rev(x):
    """R: reverse element order."""
    if isinstance(x, (pl.Series, pl.DataFrame)):
        return x.reverse()
    if isinstance(x, list):
        return x[::-1]
    return np.asarray(x)[::-1]


def sort(x, decreasing=False):
    """R: sort. ``decreasing=True`` matches R's keyword."""
    if isinstance(x, pl.Series):
        return x.sort(descending=decreasing)
    if isinstance(x, pl.DataFrame):
        raise TypeError(
            "sort(DataFrame): pass a column name to .sort() instead"
        )
    arr = np.sort(np.asarray(x))
    return arr[::-1] if decreasing else arr


def order(*args, decreasing=False):
    """R: ``order()`` — permutation that sorts. Multi-key supported.

    Returns 0-based indices (Python convention; R returns 1-based).
    """
    if len(args) == 1:
        idx = np.argsort(np.asarray(args[0]), kind="stable")
    else:
        keys = [np.asarray(a) for a in args]
        idx = np.lexsort(list(reversed(keys)))
    return idx[::-1].copy() if decreasing else idx


def which(cond):
    """R: ``which()`` — indices where cond is True. 0-based."""
    return np.flatnonzero(np.asarray(cond))


def which_max(x):
    """R: ``which.max()`` — index of first max. 0-based."""
    return int(np.argmax(np.asarray(x)))


def which_min(x):
    """R: ``which.min()`` — index of first min. 0-based."""
    return int(np.argmin(np.asarray(x)))


def cumsum(x):
    return np.cumsum(np.asarray(x))


def cumprod(x):
    return np.cumprod(np.asarray(x))


def cummax(x):
    return np.maximum.accumulate(np.asarray(x))


def cummin(x):
    return np.minimum.accumulate(np.asarray(x))


def diff(x, lag=1, differences=1):
    """R: ``diff()`` — lagged and iterated differences."""
    arr = np.asarray(x)
    for _ in range(int(differences)):
        arr = arr[lag:] - arr[:-lag]
    return arr


def unique(x):
    """R: unique values, preserving order of first occurrence."""
    if isinstance(x, (pl.Series, pl.DataFrame)):
        return x.unique(maintain_order=True)
    arr = np.asarray(x)
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def duplicated(x):
    """R: ``duplicated()`` — True for the 2nd+ occurrence of each value."""
    if isinstance(x, pl.Series):
        return ~x.is_first_distinct()
    arr = np.asarray(x)
    seen: set = set()
    out = np.zeros(arr.shape[0] if arr.ndim else 1, dtype=bool)
    for i, v in enumerate(arr.tolist() if arr.ndim else [arr.item()]):
        if v in seen:
            out[i] = True
        else:
            seen.add(v)
    return out


def tabulate(x, nbins=None):
    """R: ``tabulate()`` — counts of integer values in ``1..nbins``.

    1-based to match R: ``tabulate([1, 2, 2, 3])`` → ``[1, 2, 1]``.
    """
    arr = np.asarray(x, dtype=int)
    if nbins is None:
        nbins = int(arr.max()) if arr.size else 0
    if nbins == 0:
        return np.zeros(0, dtype=int)
    return np.bincount(arr - 1, minlength=int(nbins))[:int(nbins)]


# ---- reductions (R defaults) ----------------------------------------

def mean(x, na_rm=False):
    arr = np.asarray(x, dtype=float)
    if na_rm:
        return float(np.nanmean(arr))
    return float(np.mean(arr))


def median(x, na_rm=False):
    arr = np.asarray(x, dtype=float)
    if na_rm:
        return float(np.nanmedian(arr))
    return float(np.median(arr))


def var(x, y=None, na_rm=False):
    """R: variance with N-1 denominator. ``var(x, y)`` returns covariance."""
    if y is not None:
        return cov(x, y, na_rm=na_rm)
    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    return float(np.var(arr, ddof=1))


def sd(x, na_rm=False):
    """R: standard deviation, N-1 denominator (matches R)."""
    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    return float(np.std(arr, ddof=1))


def quantile(x, probs=(0, 0.25, 0.5, 0.75, 1.0), na_rm=False):
    """R: ``quantile()`` — numpy default is linear interpolation, ≈ R type 7."""
    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    return np.quantile(arr, probs)


def cor(x, y=None, na_rm=False):
    """R: Pearson correlation. ``cor(matrix)`` or ``cor(x, y)``."""
    if y is None:
        arr = np.asarray(x, dtype=float)
        if na_rm:
            arr = arr[~np.isnan(arr).any(axis=1)]
        return np.corrcoef(arr, rowvar=False)
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if na_rm:
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
    return float(np.corrcoef(a, b)[0, 1])


def cov(x, y=None, na_rm=False):
    """R: sample covariance, N-1 denominator."""
    if y is None:
        arr = np.asarray(x, dtype=float)
        if na_rm:
            arr = arr[~np.isnan(arr).any(axis=1)]
        return np.cov(arr, rowvar=False, ddof=1)
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if na_rm:
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
    return float(np.cov(a, b, ddof=1)[0, 1])


# ---- coercion / predicates ------------------------------------------

def as_numeric(x):
    if isinstance(x, pl.Series):
        return x.cast(pl.Float64)
    return np.asarray(x, dtype=float)


def as_integer(x):
    if isinstance(x, pl.Series):
        return x.cast(pl.Int64)
    return np.asarray(x, dtype=np.int64)


def as_character(x):
    if isinstance(x, pl.Series):
        return x.cast(pl.Utf8)
    return np.asarray(x, dtype=str)


def as_logical(x):
    if isinstance(x, pl.Series):
        return x.cast(pl.Boolean)
    return np.asarray(x, dtype=bool)


def is_na(x):
    """R: ``is.na()`` — element-wise NaN / null."""
    if isinstance(x, (pl.Series, pl.DataFrame)):
        return x.is_null()
    arr = np.asarray(x)
    if arr.dtype.kind in "fc":
        return np.isnan(arr)
    if arr.dtype == object:
        return np.array(
            [v is None for v in arr.flat]
        ).reshape(arr.shape)
    return np.zeros(arr.shape, dtype=bool)


def is_null(x):
    """R: ``is.null()`` — True iff x is None (Python's NULL)."""
    return x is None


def is_finite(x):
    """R: ``is.finite()`` — element-wise finiteness."""
    return np.isfinite(np.asarray(x, dtype=float))


def is_numeric(x):
    """R: ``is.numeric()``."""
    if isinstance(x, pl.Series):
        return x.dtype.is_numeric()
    if isinstance(x, np.ndarray):
        return x.dtype.kind in "fiu"
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float, np.integer, np.floating))


def is_factor(x):
    """R: ``is.factor()`` — True for ``pl.Enum`` / ``pl.Categorical`` columns."""
    if isinstance(x, pl.Series):
        return isinstance(x.dtype, (pl.Enum, pl.Categorical))
    return False


def levels(x):
    """R: ``levels()`` — categories of a factor / Enum, in storage order."""
    if isinstance(x, pl.Series):
        if isinstance(x.dtype, pl.Enum):
            return x.dtype.categories.to_list()
        if isinstance(x.dtype, pl.Categorical):
            return x.cat.get_categories().to_list()
    return None


def nlevels(x):
    """R: ``nlevels()`` — number of factor categories."""
    lv = levels(x)
    return len(lv) if lv is not None else 0


# ---- distributions (scipy wrappers) ---------------------------------
#
# Each family follows R's argument names where possible. Scalars in →
# scalars out; arrays in → arrays out. ``lower_tail=False`` returns
# ``1 - cdf`` (and the equivalent inverse for q*).

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
    ``hea._r_random``. This wrapper is for ordinary reproducibility.
    """
    np.random.seed(int(seed))


# ---- model generics -------------------------------------------------
#
# Free-function dispatch over hea's fitted model objects (``lm``, ``glm``,
# ``gam``, ``bam``, ``lme``). Pure duck typing — no model-class imports
# needed. Where R has multiple aliases (``coef``/``coefficients``,
# ``resid``/``residuals``, ``fitted``/``fitted.values``), we expose both.

def _bhat_to_dict(model) -> dict:
    if not hasattr(model, "bhat") or not isinstance(model.bhat, pl.DataFrame):
        raise TypeError(
            f"{model.__class__.__name__} has no .bhat DataFrame"
        )
    return dict(zip(model.bhat.columns, model.bhat.row(0)))


def coef(model):
    """R: ``coef()`` — coefficients as ``{name: estimate}``.

    Works for ``lm`` / ``glm`` / ``gam`` / ``bam`` / ``lme``. For ``lme``
    this returns FIXED effects only (= R's ``fixef(m)``); R's
    ``coef.lmerMod`` returns per-group BLUPs which hea doesn't compute
    in the same shape — use ``fixef()`` + ``ranef()`` to assemble.
    """
    return _bhat_to_dict(model)


def coefficients(model):
    """R alias for :func:`coef`."""
    return coef(model)


def fixef(model):
    """R: ``fixef()`` — fixed-effect coefficients (lme).

    For non-mixed models, identical to :func:`coef`.
    """
    return coef(model)


def ranef(model):
    """R: ``ranef()`` — random effects (lme only)."""
    if hasattr(model, "ranef"):
        return model.ranef
    raise TypeError(
        f"ranef(): {model.__class__.__name__} has no random effects"
    )


def resid(model, type=None):
    """R: ``resid()`` / ``residuals()`` — residuals as 1D ``ndarray``.

    For ``glm`` / ``gam`` / ``bam``, ``type`` selects among
    ``{"deviance"`` (default, matches R), ``"pearson"``, ``"working"``,
    ``"response"}``. ``lm`` and ``lme`` only have response residuals;
    pass ``type=None`` or ``"response"`` (anything else raises).
    """
    if hasattr(model, "residuals_of"):
        return model.residuals_of(type or "deviance")
    if type not in (None, "response"):
        raise ValueError(
            f"resid(): type={type!r} not supported for "
            f"{model.__class__.__name__} (only 'response' / None)"
        )
    r = getattr(model, "residuals", None)
    if isinstance(r, pl.DataFrame):
        return r.to_series().to_numpy()
    if isinstance(r, np.ndarray):
        return r
    if isinstance(r, pl.Series):
        return r.to_numpy()
    raise TypeError(
        f"resid(): {model.__class__.__name__} has no usable residuals"
    )


def residuals(model, type=None):
    """R alias for :func:`resid`."""
    return resid(model, type)


def fitted(model):
    """R: ``fitted()`` — fitted values as 1D ``ndarray``.

    For lm/glm this is the response-scale prediction (μ̂); for gam/lme
    same. Equivalent to ``model.predict()`` on the training data.
    """
    fv = getattr(model, "fitted_values", None)
    if fv is not None:
        return np.asarray(fv)
    f = getattr(model, "fitted", None)
    if f is not None and not callable(f):
        return np.asarray(f)
    yh = getattr(model, "yhat", None)
    if isinstance(yh, pl.DataFrame):
        col = "Fitted" if "Fitted" in yh.columns else yh.columns[0]
        return yh[col].to_numpy()
    if isinstance(yh, np.ndarray):
        return yh
    raise TypeError(
        f"fitted(): {model.__class__.__name__} has no fitted values"
    )


def fitted_values(model):
    """R alias for :func:`fitted`."""
    return fitted(model)


def predict(model, *args, **kwargs):
    """R: ``predict()`` — dispatches to ``model.predict(...)``.

    Forwards positional and keyword arguments untouched, so
    ``predict(m, newdata, interval="confidence")`` works exactly like
    the bound method.
    """
    if not hasattr(model, "predict"):
        raise TypeError(
            f"predict(): {model.__class__.__name__} has no .predict()"
        )
    return model.predict(*args, **kwargs)


def confint(model, level=0.95):
    """R: ``confint()`` — confidence intervals for the coefficients.

    Returns a polars DataFrame with one row per coefficient.

    For ``lm``, ``level`` is honored exactly (refits CIs at
    ``alpha = 1 - level``). For other models, only ``level=0.95``
    is wired here — use the model's own method for other levels.
    """
    if level == 0.95 and hasattr(model, "ci_bhat"):
        return model.ci_bhat
    if hasattr(model, "compute_ci_bhat"):
        return model.compute_ci_bhat(alpha=1 - level)
    raise NotImplementedError(
        f"confint(): level={level} not supported for "
        f"{model.__class__.__name__}"
    )


def vcov(model):
    """R: ``vcov()`` — variance-covariance matrix of the coefficients.

    Return type varies by model: lm/glm return ``ndarray`` (``V_bhat``);
    gam/bam return ``ndarray`` (``Vp``, the Bayesian posterior); lme
    returns a polars ``DataFrame`` (``vcov_beta``, fixed effects only).
    """
    if hasattr(model, "vcov_beta"):  # lme
        return model.vcov_beta
    if hasattr(model, "Vp"):  # gam / bam (Bayesian posterior)
        return model.Vp
    if hasattr(model, "V_bhat"):  # lm / glm
        return model.V_bhat
    raise TypeError(
        f"vcov(): {model.__class__.__name__} not supported"
    )


def logLik(model):
    """R: ``logLik()`` — model log-likelihood.

    For REML-fit ``lme`` (no plain ``loglike``), returns the REML
    log-likelihood ``-REML_criterion / 2``, matching ``logLik.lmerMod``.
    """
    if hasattr(model, "loglike"):
        return float(model.loglike)
    if hasattr(model, "REML_criterion"):
        return -float(model.REML_criterion) / 2.0
    raise TypeError(
        f"logLik(): {model.__class__.__name__} has no log-likelihood"
    )


def deviance(model):
    """R: ``deviance()`` — model deviance.

    For ``lm`` (no Gaussian deviance attribute), returns ``rss`` —
    matches ``deviance.lm = sum(residuals^2)``.
    """
    if hasattr(model, "deviance") and not callable(model.deviance):
        return float(model.deviance)
    if hasattr(model, "rss"):  # lm
        return float(model.rss)
    raise TypeError(
        f"deviance(): {model.__class__.__name__} has no deviance"
    )


def nobs(model):
    """R: ``nobs()`` — number of observations used to fit."""
    return int(model.n)


def df_residual(model):
    """R: ``df.residual()`` — residual degrees of freedom."""
    for attr in ("df_residual", "df_residuals", "df_resid"):
        v = getattr(model, attr, None)
        if v is not None:
            return float(v)
    raise TypeError(
        f"df_residual(): {model.__class__.__name__} has no residual df"
    )


def formula(model):
    """R: ``formula()`` — extract the model formula (string)."""
    return model.formula


def model_matrix(model):
    """R: ``model.matrix()`` — design matrix used at fit time.

    Returns a polars DataFrame; columns are the named design columns
    (intercept, dummy-coded factor levels, spline bases, …). R returns
    an unnamed numeric matrix; we keep the names attached.
    """
    if hasattr(model, "X"):
        return model.X
    raise TypeError(
        f"model_matrix(): {model.__class__.__name__} has no design matrix"
    )


def model_frame(model):
    """R: ``model.frame()`` — original data passed at fit time."""
    return model.data


def AIC(*models):
    """R: ``AIC()`` — scalar for one model, comparison table for many.

    With one argument, returns ``model.AIC`` as a float. With two or
    more, returns a polars DataFrame with row labels recovered from the
    caller's variable names (R-style), plus columns ``df`` and ``AIC``.

    Note: ``hea.AIC`` (without the ``from hea.R import *``) prints the
    table and returns ``None``. This R-style version always returns.
    """
    if not models:
        raise TypeError("AIC(): need at least one model")
    if len(models) == 1:
        return float(models[0].AIC)
    import inspect
    from .compare import _caller_names
    names = _caller_names(models, inspect.currentframe().f_back)
    return pl.DataFrame({
        "":    names,
        "df":  [m.npar for m in models],
        "AIC": [float(m.AIC) for m in models],
    })


def BIC(*models):
    """R: ``BIC()`` — scalar for one model, comparison table for many.

    Same convention as :func:`AIC`.
    """
    if not models:
        raise TypeError("BIC(): need at least one model")
    if len(models) == 1:
        return float(models[0].BIC)
    import inspect
    from .compare import _caller_names
    names = _caller_names(models, inspect.currentframe().f_back)
    return pl.DataFrame({
        "":    names,
        "df":  [m.npar for m in models],
        "BIC": [float(m.BIC) for m in models],
    })
