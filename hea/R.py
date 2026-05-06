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
* **Sequence and indexing functions are 0-based.** ``which``, ``which_max``,
  ``which_min``, ``order`` and the one-arg ``seq(n)`` / ``seq_len`` /
  ``seq_along`` all match Python conventions. For R's ``1:n`` muscle
  memory, write ``seq(1, n)`` explicitly — the two-arg ``seq(start, stop)``
  form is still inclusive on both ends, so ``seq(1, 5)`` gives
  ``[1, 2, 3, 4, 5]``.
* **R parameter names preserved where possible.** ``mean=`` / ``sd=`` /
  ``df=`` / ``shape=`` / ``rate=`` / ``prob=``. R's ``lower.tail``
  becomes ``lower_tail``. R's ``na.rm`` becomes ``na_rm``. R's
  ``lambda=`` becomes ``lambda_=`` (Python keyword).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

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
    "cut", "findInterval",
    # contingency tables
    "table", "xtabs", "prop_table", "addmargins",
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
    # rank helpers (Lindeløv-style "tests as lm" notebook)
    "rank", "signed_rank",
    # hypothesis tests (return HTest, R's ``htest`` print-shape)
    "HTest", "AnovaTable",
    "t_test", "wilcox_test", "cor_test", "kruskal_test", "chisq_test",
    "fisher_test", "prop_test", "binom_test", "var_test", "bartlett_test",
    "shapiro_test", "ks_test", "mcnemar_test", "friedman_test",
    "aov",
    # model generics (lm / glm / gam / bam / lme)
    "coef", "coefficients", "fixef", "ranef",
    "resid", "residuals", "fitted", "fitted_values",
    "predict", "confint", "vcov",
    "logLik", "deviance", "nobs", "df_residual",
    "formula", "model_matrix", "model_frame",
    "AIC", "BIC",
    "update", "terms", "Terms",
    # regression diagnostics
    "hatvalues", "rstandard", "rstudent",
    "cooks_distance", "dffits", "dfbetas", "influence",
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

    The one-argument and ``along_with`` forms are 0-based to match Python
    indexing — different from R's 1-based defaults. The two-argument
    ``seq(from, to)`` form keeps R's inclusive-on-both-ends semantics, so
    explicit ``seq(1, n)`` gives ``[1, 2, …, n]``.

    Call shapes:

    * ``seq(n)`` → ``0, 1, …, n-1`` (Python; for R's ``1:n`` use ``seq(1, n)``)
    * ``seq(from, to)`` → ``from, from+1, …, to`` (inclusive, R-faithful)
    * ``seq(from, to, by=step)`` → step-spaced, inclusive
    * ``seq(from, to, length_out=n)`` → ``n`` evenly spaced
    * ``seq(along_with=x)`` → ``0, 1, …, len(x)-1``
    """
    if along_with is not None:
        return np.arange(len(along_with))
    if length_out is not None:
        if len(args) == 0:
            return np.arange(int(length_out))
        if len(args) == 1:
            return np.linspace(0, args[0], int(length_out))
        return np.linspace(args[0], args[1], int(length_out))
    if len(args) == 0:
        raise ValueError("seq(): need at least one positional argument")
    if len(args) == 1:
        return np.arange(int(args[0]))
    start, stop = args[0], args[1]
    step = by if by is not None else (1 if stop >= start else -1)
    n_steps = int(np.floor((stop - start) / step + 1e-10)) + 1
    return start + np.arange(n_steps) * step


def seq_len(n):
    """R: ``seq_len(n)`` → ``0, 1, …, n-1`` (0-based; differs from R's ``1:n``)."""
    return np.arange(int(n))


def seq_along(x):
    """R: ``seq_along(x)`` → ``0, 1, …, len(x)-1`` (0-based; differs from R)."""
    return np.arange(len(x))


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


def cut(x, breaks, *, labels=None, right=True, include_lowest=False):
    """R: ``cut()`` — bin a numeric vector into intervals (a factor).

    Parameters
    ----------
    x : array-like
        Numeric vector to bin.
    breaks : int or array-like
        Number of equal-width bins (R's scalar form), or an explicit
        strictly-increasing sequence of cut points.
    labels : list, False, or None
        ``None`` (default): auto-generate ``"(a,b]"`` / ``"[a,b)"``-style
        labels. A list: custom labels (length must equal ``len(breaks)-1``).
        ``False``: return integer codes instead of a factor (1-based to
        match R; ``NaN`` for out-of-range).
    right : bool, default True
        If True, bins are right-closed ``(a, b]`` (R's default). If False,
        left-closed ``[a, b)``.
    include_lowest : bool, default False
        If True, include the boundary value in the lowest bin (right=True)
        or in the highest bin (right=False), matching R's ``include.lowest``.

    Returns
    -------
    pl.Series
        ``pl.Enum`` factor with the bin labels, or a numpy ``float64`` array
        of 1-based integer codes when ``labels=False``.
    """
    arr = np.asarray(x, dtype=float)
    if isinstance(breaks, (int, np.integer)) and not isinstance(breaks, bool):
        n = int(breaks)
        if n < 1:
            raise ValueError("cut(): need at least 1 bin")
        if arr.size == 0:
            raise ValueError("cut(): empty input with scalar breaks")
        rng = float(arr.max()) - float(arr.min())
        eps = 0.001 * rng if rng > 0 else 0.001
        breaks = np.linspace(arr.min() - eps, arr.max() + eps, n + 1)
    breaks_arr = np.asarray(breaks, dtype=float)
    if not (np.diff(breaks_arr) > 0).all():
        raise ValueError("cut(): breaks must be strictly increasing")
    n_bins = len(breaks_arr) - 1
    if n_bins < 1:
        raise ValueError("cut(): need at least 2 break points")

    idx = np.digitize(arr, breaks_arr, right=right) - 1
    if include_lowest:
        if right:
            idx = np.where(arr == breaks_arr[0], 0, idx)
        else:
            idx = np.where(arr == breaks_arr[-1], n_bins - 1, idx)
    out_of_range = (idx < 0) | (idx >= n_bins)

    if labels is False:
        codes = (idx + 1).astype(float)
        codes[out_of_range] = np.nan
        return codes

    if labels is None:
        if right:
            lab_list = [
                f"({_fmt(breaks_arr[i])},{_fmt(breaks_arr[i + 1])}]"
                for i in range(n_bins)
            ]
            if include_lowest:
                lab_list[0] = (
                    f"[{_fmt(breaks_arr[0])},{_fmt(breaks_arr[1])}]"
                )
        else:
            lab_list = [
                f"[{_fmt(breaks_arr[i])},{_fmt(breaks_arr[i + 1])})"
                for i in range(n_bins)
            ]
            if include_lowest:
                lab_list[-1] = (
                    f"[{_fmt(breaks_arr[-2])},{_fmt(breaks_arr[-1])}]"
                )
    else:
        lab_list = [str(la) for la in labels]
        if len(lab_list) != n_bins:
            raise ValueError(
                f"cut(): {len(lab_list)} labels but {n_bins} bins"
            )

    result = [
        None if out_of_range[i] else lab_list[int(idx[i])]
        for i in range(len(arr))
    ]
    return pl.Series(result, dtype=pl.Utf8).cast(pl.Enum(lab_list))


def findInterval(
    x,
    vec,
    *,
    rightmost_closed: bool = False,
    all_inside: bool = False,
    left_open: bool = False,
):
    """R: ``findInterval()`` — for each x, the index of its enclosing
    interval in a sorted ``vec``.

    Returns ``i`` in ``0..len(vec)`` (same convention as R, treating the
    return as an index into ``vec``):

    * ``i = 0``        → x is below ``vec[0]``
    * ``vec[i-1] ≤ x < vec[i]`` (default, ``left_open=False``)
    * ``i = len(vec)`` → x is at or above ``vec[-1]``
    """
    arr = np.asarray(x, dtype=float)
    vec_arr = np.asarray(vec, dtype=float)
    if not (np.diff(vec_arr) >= 0).all():
        raise ValueError("findInterval(): vec must be non-decreasing")
    side = "left" if left_open else "right"
    idx = np.searchsorted(vec_arr, arr, side=side)
    if rightmost_closed:
        idx = np.where(arr == vec_arr[-1], len(vec_arr) - 1, idx)
    if all_inside:
        idx = np.clip(idx, 1, len(vec_arr) - 1)
    return idx


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


# ---- contingency tables ---------------------------------------------
#
# ``table`` / ``xtabs`` build frequency tables; ``prop_table`` and
# ``addmargins`` decorate them. We return polars DataFrames rather than
# R's "table" object — the first column carries the row labels, with
# remaining columns one per level of the second variable.

def table(x, y=None, *, dnn=None):
    """R: ``table()`` — 1- or 2-way frequency table.

    1-way: returns a 2-column DataFrame ``(value, n)`` sorted by value.
    2-way: returns a wide DataFrame whose first column is the row label
    (named ``""`` by default; pass ``dnn=("row", "col")`` for R-style
    dimension names) and remaining columns are the levels of ``y``.

    Nulls are dropped (R's ``useNA="no"`` default).
    """
    if y is None:
        s = pl.Series(x).drop_nulls().cast(pl.Utf8)
        out = (
            pl.DataFrame({"value": s})
            .group_by("value").len(name="n")
            .sort("value")
        )
        if dnn is not None:
            out = out.rename({"value": str(dnn[0])})
        return out

    df = pl.DataFrame({
        "__x__": pl.Series(x).cast(pl.Utf8),
        "__y__": pl.Series(y).cast(pl.Utf8),
    }).drop_nulls()
    counts = df.group_by(["__x__", "__y__"]).len(name="n")
    pivot = (
        counts.pivot(values="n", index="__x__", on="__y__")
        .fill_null(0)
        .sort("__x__")
    )
    # sort columns alphabetically so output is reproducible (polars'
    # pivot uses encounter order, which depends on group_by ordering)
    label_col = "__x__"
    other_cols = sorted(c for c in pivot.columns if c != label_col)
    pivot = pivot.select([label_col, *other_cols])
    new_label = str(dnn[0]) if dnn is not None else ""
    return pivot.rename({label_col: new_label})


def xtabs(formula: str, data: pl.DataFrame):
    """R: ``xtabs()`` — formula-based contingency table.

    Currently supports the count form ``~ a`` or ``~ a + b`` (no LHS).
    The weighted form ``w ~ a + b`` (sum of ``w`` per cell) is not yet
    wired.
    """
    if "~" not in formula:
        raise ValueError("xtabs(): formula must contain '~'")
    lhs, rhs = formula.split("~", 1)
    if lhs.strip():
        raise NotImplementedError(
            "xtabs(): weighted form (with LHS) not yet supported"
        )
    cols = [p.strip() for p in rhs.split("+")]
    if len(cols) == 1:
        return table(data[cols[0]])
    if len(cols) == 2:
        return table(data[cols[0]], data[cols[1]], dnn=(cols[0], cols[1]))
    raise NotImplementedError(
        "xtabs(): 3+ way tables not supported in v1"
    )


def prop_table(tbl, margin=None):
    """R: ``prop.table()`` — convert counts to proportions.

    ``margin=None`` divides by the grand total (default).
    ``margin=1`` divides each row by its sum (row-conditional).
    ``margin=2`` divides each column by its sum (column-conditional).

    Accepts the polars DataFrame produced by :func:`table` /
    :func:`xtabs`, or a plain numpy 2-D array.
    """
    if isinstance(tbl, pl.DataFrame):
        # 1-way table from this module: cols are exactly ['value', 'n'].
        if tbl.columns == ["value", "n"]:
            n = tbl["n"].cast(pl.Float64).to_numpy()
            return tbl.with_columns(pl.Series("n", n / n.sum()))
        # 2-way: first col carries row labels; rest are counts.
        label_col = tbl.columns[0]
        count_cols = tbl.columns[1:]
        mat = tbl.select(count_cols).to_numpy().astype(float)
        mat = _apply_margin_proportion(mat, margin)
        return pl.DataFrame(
            {label_col: tbl[label_col], **{c: mat[:, i] for i, c in enumerate(count_cols)}}
        )
    arr = np.asarray(tbl, dtype=float)
    return _apply_margin_proportion(arr, margin)


def _apply_margin_proportion(mat: np.ndarray, margin) -> np.ndarray:
    if margin is None:
        total = mat.sum()
        if total == 0:
            return mat.copy()
        return mat / total
    if margin == 1:
        rs = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rs > 0, mat / rs, 0.0)
    if margin == 2:
        cs = mat.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cs > 0, mat / cs, 0.0)
    raise ValueError("prop_table(): margin must be None, 1, or 2")


def addmargins(tbl, margin=None):
    """R: ``addmargins()`` — append ``Sum`` rows / columns to a table.

    ``margin=None``: add both a column-of-row-sums and a row-of-column-sums
    (plus the grand total at the corner). ``margin=1``: only the column-sum
    row. ``margin=2``: only the row-sum column.
    """
    if not isinstance(tbl, pl.DataFrame):
        raise NotImplementedError(
            "addmargins(): only polars DataFrame inputs supported in v1"
        )
    if tbl.columns == ["value", "n"]:
        # 1-way: append a "Sum" row, cast `n` to keep types consistent
        n_total = float(tbl["n"].sum())
        n_cast = tbl["n"].cast(pl.Float64)
        return pl.concat([
            tbl.with_columns(n_cast),
            pl.DataFrame({"value": ["Sum"], "n": [n_total]}),
        ])
    label_col = tbl.columns[0]
    count_cols = tbl.columns[1:]
    mat = tbl.select(count_cols).to_numpy().astype(float)
    margins = (1, 2) if margin is None else (margin,)
    # Cast count columns to Float64 so the appended Sum row (float) lines up.
    new_tbl = tbl.with_columns(*(pl.col(c).cast(pl.Float64) for c in count_cols))
    if 2 in margins:
        new_tbl = new_tbl.with_columns(
            pl.Series("Sum", mat.sum(axis=1))
        )
    if 1 in margins:
        sum_row: dict = {label_col: "Sum"}
        col_sums = mat.sum(axis=0)
        for i, c in enumerate(count_cols):
            sum_row[c] = float(col_sums[i])
        if 2 in margins:
            sum_row["Sum"] = float(mat.sum())
        new_tbl = pl.concat([new_tbl, pl.DataFrame([sum_row])])
    return new_tbl


# ---- result containers ----------------------------------------------
#
# ``HTest`` and ``AnovaTable`` mirror R's ``htest`` and ``anova`` print
# objects. The ``__repr__`` blocks reproduce ``stats:::print.htest`` /
# ``print.anova`` line layouts so two paths through the notebook (named
# test vs ``lm()`` / ``glm()``) print comparable output. Living here
# alongside the test functions keeps the result type and its consumers
# in one file.

@dataclass
class HTest:
    """R's ``htest`` class as a Python dataclass.

    Mirrors ``stats:::print.htest``: ``method`` is the title, ``statistic``
    the named scalar, ``parameter`` the df line, plus optional p-value,
    CI, point ``estimate``, and ``alternative``. ``data_name`` is the
    "data:" label R prints before the stats.
    """

    method: str
    statistic: dict = field(default_factory=dict)
    parameter: dict = field(default_factory=dict)
    p_value: Optional[float] = None
    conf_int: Optional[tuple] = None
    estimate: dict = field(default_factory=dict)
    null_value: Optional[Union[float, dict]] = None
    alternative: str = "two.sided"
    data_name: str = ""
    conf_level: float = 0.95

    def __repr__(self) -> str:
        out = ["", f"\t{self.method}", ""]
        if self.data_name:
            out.append(f"data:  {self.data_name}")
        bits = []
        for k, v in self.statistic.items():
            bits.append(f"{k} = {_fmt(v)}")
        for k, v in self.parameter.items():
            bits.append(f"{k} = {_fmt(v)}")
        if self.p_value is not None:
            bits.append(f"p-value = {_fmt_pval(self.p_value)}")
        if bits:
            out.append(", ".join(bits))
        if self.alternative:
            null = self.null_value
            tail = "not equal to"
            if self.alternative == "greater":
                tail = "greater than"
            elif self.alternative == "less":
                tail = "less than"
            if isinstance(null, dict):
                null_str = ", ".join(f"{k} = {_fmt(v)}" for k, v in null.items())
                out.append(f"alternative hypothesis: true {null_str.split(' = ')[0]} is {tail} {null_str.split(' = ')[1]}")
            elif null is not None:
                # name from estimate keys when possible
                nm = next(iter(self.estimate.keys()), "value")
                out.append(f"alternative hypothesis: true {nm} is {tail} {_fmt(null)}")
        if self.conf_int is not None:
            out.append(f"{int(self.conf_level * 100)} percent confidence interval:")
            out.append(f" {_fmt(self.conf_int[0])} {_fmt(self.conf_int[1])}")
        if self.estimate:
            out.append("sample estimates:")
            keys = "  ".join(f"{k}" for k in self.estimate)
            vals = "  ".join(f"{_fmt(v)}" for v in self.estimate.values())
            out.append(keys)
            out.append(vals)
        return "\n".join(out) + "\n"


@dataclass
class AnovaTable:
    """R-style ``Anova`` / ``anova`` table (Type-II by default for ``aov``).

    Stored as a list of rows (term, df, sum_sq, mean_sq, F, p) plus a
    Residuals row. ``__repr__`` formats it close to R's printout.
    """

    response: str
    rows: list  # list of dicts: term, df, sum_sq, mean_sq, F, p
    residual_df: int
    residual_ss: float
    type: str = "II"

    def __repr__(self) -> str:
        out = [f"Anova Table (Type {self.type} tests)", "",
               f"Response: {self.response}",
               f"{'':<12}{'Sum Sq':>10}{'Df':>4}{'F value':>10}{'Pr(>F)':>12}"]
        for r in self.rows:
            out.append(
                f"{r['term']:<12}{_fmt(r['sum_sq']):>10}{r['df']:>4}"
                f"{_fmt(r['F']):>10}{_fmt_pval(r['p']):>12}"
            )
        out.append(
            f"{'Residuals':<12}{_fmt(self.residual_ss):>10}{self.residual_df:>4}"
        )
        return "\n".join(out)


# ---- formatters & private helpers (used by HTest / AnovaTable) ------

def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    fx = float(x)
    if not np.isfinite(fx):
        return str(fx)
    ax = abs(fx)
    if ax != 0 and (ax < 1e-4 or ax >= 1e5):
        return f"{fx:.5g}"
    return f"{fx:.5g}"


def _fmt_pval(p: float) -> str:
    if p is None:
        return ""
    if p < 2.2e-16:
        return "< 2.2e-16"
    return _fmt(p)


def _as_array(x) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(float)
    return np.asarray(x, dtype=float)


# ---- rank helpers (used by Wilcoxon/Spearman/Lindeløv constructions) -

def rank(x) -> np.ndarray:
    """R's ``rank()`` with ``ties.method = "average"`` (R's default).

    Returns a float array so downstream lm() formulas treat it as numeric.
    """
    return _sps.rankdata(_as_array(x), method="average")


def signed_rank(x) -> np.ndarray:
    """Lindeløv's ``signed_rank = function(x) sign(x) * rank(abs(x))``.

    Used to turn Wilcoxon signed-rank into an intercept-only ``lm``.
    """
    arr = _as_array(x)
    return np.sign(arr) * _sps.rankdata(np.abs(arr), method="average")


# ---- hypothesis tests -----------------------------------------------
#
# Every function returns an :class:`HTest`. R parameter names are
# preserved where possible: ``alternative`` ∈ {"two.sided", "greater",
# "less"}, ``conf_level=0.95``, ``correct=`` (continuity correction)
# where applicable. ``mu`` / ``p`` / ``ratio`` carry their R meanings.

def t_test(
    x,
    y=None,
    *,
    paired: bool = False,
    var_equal: bool = True,
    mu: float = 0.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``t.test``.

    - ``y=None``                     → one-sample t-test on ``x``.
    - ``y`` given, ``paired=True``   → paired t-test on ``x - y``.
    - ``y`` given, ``var_equal=True``→ Student's two-sample (pooled var).
    - ``y`` given, ``var_equal=False``→ Welch's two-sample.

    Always uses Student-t CI on the appropriate df.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    if y is None:
        res = _sps.ttest_1samp(x, mu, alternative=alt)
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="One Sample t-test",
            statistic={"t": float(res.statistic)},
            parameter={"df": float(res.df)},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"mean of x": float(np.mean(x))},
            null_value=mu,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x",
        )
    y = _as_array(y)
    if paired:
        d = x - y
        res = _sps.ttest_1samp(d, mu, alternative=alt)
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="Paired t-test",
            statistic={"t": float(res.statistic)},
            parameter={"df": float(res.df)},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"mean of the differences": float(np.mean(d))},
            null_value=mu,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and y",
        )
    res = _sps.ttest_ind(x, y, equal_var=var_equal, alternative=alt)
    ci = res.confidence_interval(conf_level)
    method = "Two Sample t-test" if var_equal else "Welch Two Sample t-test"
    return HTest(
        method=method,
        statistic={"t": float(res.statistic)},
        parameter={"df": float(res.df)},
        p_value=float(res.pvalue),
        conf_int=(float(ci.low), float(ci.high)),
        estimate={"mean of x": float(np.mean(x)), "mean of y": float(np.mean(y))},
        null_value=mu,
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and y",
    )


def wilcox_test(
    x,
    y=None,
    *,
    paired: bool = False,
    alternative: str = "two.sided",
    correct: bool = True,
) -> HTest:
    """R's ``wilcox.test``.

    Defaults to continuity correction (``correct=True``). One-sample and
    paired branches use ``scipy.stats.wilcoxon``; the two-sample branch
    uses ``mannwhitneyu`` (R's "Wilcoxon rank-sum" with W statistic).
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    if y is None:
        res = _sps.wilcoxon(x, alternative=alt, correction=correct, zero_method="wilcox")
        return HTest(
            method="Wilcoxon signed rank test"
            + (" with continuity correction" if correct else ""),
            statistic={"V": float(res.statistic)},
            p_value=float(res.pvalue),
            null_value=0.0,
            alternative=alternative,
            data_name="x",
        )
    y = _as_array(y)
    if paired:
        res = _sps.wilcoxon(x, y, alternative=alt, correction=correct, zero_method="wilcox")
        return HTest(
            method="Wilcoxon signed rank test"
            + (" with continuity correction" if correct else ""),
            statistic={"V": float(res.statistic)},
            p_value=float(res.pvalue),
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    res = _sps.mannwhitneyu(
        x, y, alternative=alt, use_continuity=correct, method="asymptotic"
    )
    return HTest(
        method="Wilcoxon rank sum test"
        + (" with continuity correction" if correct else ""),
        statistic={"W": float(res.statistic)},
        p_value=float(res.pvalue),
        null_value=0.0,
        alternative=alternative,
        data_name="x and y",
    )


def cor_test(
    x,
    y,
    *,
    method: str = "pearson",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``cor.test`` with ``method`` in {pearson, spearman, kendall}.

    For Pearson, we report ``t``, df = n-2, and Fisher-z CI. Spearman
    reports ``S`` (rank-sum statistic R's ``cor.test`` shows). Kendall
    reports ``z``.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    y = _as_array(y)
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length")
    n = len(x)
    if method == "pearson":
        res = _sps.pearsonr(x, y, alternative=alt)
        r = float(res.statistic)
        df = n - 2
        t = r * np.sqrt(df / max(1 - r * r, 1e-300))
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="Pearson's product-moment correlation",
            statistic={"t": t},
            parameter={"df": df},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"cor": r},
            null_value=0.0,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and y",
        )
    if method == "spearman":
        res = _sps.spearmanr(x, y, alternative=alt)
        rho = float(res.statistic)
        # R reports S = sum((rank(x) - rank(y))^2) for the Spearman test
        S = float(np.sum((_sps.rankdata(x) - _sps.rankdata(y)) ** 2))
        return HTest(
            method="Spearman's rank correlation rho",
            statistic={"S": S},
            p_value=float(res.pvalue),
            estimate={"rho": rho},
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    if method == "kendall":
        res = _sps.kendalltau(x, y, alternative=alt)
        return HTest(
            method="Kendall's rank correlation tau",
            statistic={"z": float(res.statistic)},
            p_value=float(res.pvalue),
            estimate={"tau": float(res.statistic)},
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    raise ValueError(f"unknown method: {method}")


def kruskal_test(formula: str, data: pl.DataFrame) -> HTest:
    """R's ``kruskal.test(y ~ group, data)``.

    Only the formula form is supported here — that's what the notebook
    uses. The numeric LHS is grouped by the RHS factor and passed to
    ``scipy.stats.kruskal``.
    """
    if "~" not in formula:
        raise ValueError("formula must look like 'y ~ group'")
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    groups = [
        data.filter(pl.col(rhs) == g)[lhs].to_numpy().astype(float)
        for g in data[rhs].unique().to_list()
    ]
    res = _sps.kruskal(*groups)
    return HTest(
        method="Kruskal-Wallis rank sum test",
        statistic={"Kruskal-Wallis chi-squared": float(res.statistic)},
        parameter={"df": int(len(groups) - 1)},
        p_value=float(res.pvalue),
        alternative="",
        data_name=f"{lhs} by {rhs}",
    )


def chisq_test(
    x,
    y=None,
    *,
    p=None,
    correct: bool = True,
) -> HTest:
    """R's ``chisq.test``.

    - 1-D ``x`` (and no ``y``)         → goodness-of-fit against ``p`` (uniform if None).
    - 2-D ``x`` (matrix or 2-D array)  → contingency-table test.
    - 1-D ``x`` and 1-D ``y``          → contingency on ``crosstab(x, y)``.
    """
    arr = np.asarray(x)
    if y is not None:
        tbl = _crosstab(x, y)
        return _chisq_table(tbl, correct=correct, name="x and y")
    if arr.ndim == 2:
        return _chisq_table(arr, correct=correct, name="x")
    # goodness of fit
    counts = arr.astype(float)
    if p is None:
        p = np.full_like(counts, 1.0 / len(counts))
    p = np.asarray(p, dtype=float)
    expected = counts.sum() * p
    stat = float(np.sum((counts - expected) ** 2 / expected))
    df = len(counts) - 1
    pval = float(_sps.chi2.sf(stat, df))
    return HTest(
        method="Chi-squared test for given probabilities",
        statistic={"X-squared": stat},
        parameter={"df": df},
        p_value=pval,
        alternative="",
        data_name="x",
    )


def _chisq_table(tbl: np.ndarray, *, correct: bool, name: str) -> HTest:
    res = _sps.chi2_contingency(tbl, correction=(correct and tbl.shape == (2, 2)))
    return HTest(
        method="Pearson's Chi-squared test"
        + (" with Yates' continuity correction" if (correct and tbl.shape == (2, 2)) else ""),
        statistic={"X-squared": float(res.statistic)},
        parameter={"df": int(res.dof)},
        p_value=float(res.pvalue),
        alternative="",
        data_name=name,
    )


def _crosstab(x, y) -> np.ndarray:
    """Build a 2-way contingency table from two 1-D vectors (utf8-cast).

    Internal columns use ``__x__`` / ``__y__`` so user data containing
    string values like ``"x"`` / ``"y"`` doesn't collide with the index
    column name once ``pivot`` spreads the levels of ``y`` into columns.
    """
    x_ser = pl.Series("__x__", x).cast(pl.Utf8)
    y_ser = pl.Series("__y__", y).cast(pl.Utf8)
    return (
        pl.DataFrame({"__x__": x_ser, "__y__": y_ser})
        .group_by(["__x__", "__y__"]).len()
        .pivot(values="len", index="__x__", on="__y__")
        .fill_null(0)
        .drop("__x__")
        .to_numpy()
    )


def fisher_test(
    x,
    y=None,
    *,
    alternative: str = "two.sided",
) -> HTest:
    """R's ``fisher.test`` — Fisher's exact test for a 2×2 contingency table.

    ``x`` may be a 2×2 array/matrix or a 1-D vector paired with ``y``.
    Larger tables (R's Monte-Carlo simulation branch) are not supported.
    Returns the odds ratio as the point estimate; CI is omitted (R uses
    inverse non-central hypergeometric, not yet wired).
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    if y is not None:
        tbl = _crosstab(x, y)
        name = "x and y"
    else:
        tbl = np.asarray(x)
        name = "x"
    if tbl.shape != (2, 2):
        raise NotImplementedError(
            f"fisher_test(): only 2x2 tables supported (got {tbl.shape})"
        )
    res = _sps.fisher_exact(tbl, alternative=alt)
    odds = float(res.statistic)
    return HTest(
        method="Fisher's Exact Test for Count Data",
        p_value=float(res.pvalue),
        estimate={"odds ratio": odds},
        null_value=1.0,
        alternative=alternative,
        data_name=name,
    )


def prop_test(
    x,
    n=None,
    *,
    p=None,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
) -> HTest:
    """R's ``prop.test`` — chi-squared test on proportions.

    Supports 1-sample (``length(x)==1``, requires ``n``; ``p`` defaults
    to 0.5) and 2-sample equality (``length(x)==2``, ``p=None``). The
    k-sample (k > 2) and ``p`` vectors with ``length>1`` are not yet
    wired.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=int))
    if n is None:
        raise ValueError("prop_test(): n must be provided")
    n_arr = np.atleast_1d(np.asarray(n, dtype=int))
    if x_arr.shape != n_arr.shape:
        raise ValueError("prop_test(): x and n must have the same length")
    k = len(x_arr)
    estimates = {f"prop {i+1}": float(x_arr[i] / n_arr[i]) for i in range(k)}

    if k == 1:
        p_null = 0.5 if p is None else float(np.asarray(p))
        x0, n0 = int(x_arr[0]), int(n_arr[0])
        diff = abs(x0 / n0 - p_null)
        if correct:
            diff = max(diff - 0.5 / n0, 0.0)
        if p_null in (0.0, 1.0):
            stat = float("inf") if diff > 0 else 0.0
        else:
            stat = (diff ** 2) / (p_null * (1 - p_null) / n0)
        df = 1
        pval = float(_sps.chi2.sf(stat, df))
        return HTest(
            method="1-sample test for given proportion"
            + (" with continuity correction" if correct else ""),
            statistic={"X-squared": stat},
            parameter={"df": df},
            p_value=pval,
            estimate={"p": x0 / n0},
            null_value=p_null,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and n",
        )
    if k >= 2 and p is None:
        # k-sample equality of proportions: (k × 2) chi-squared. R applies
        # Yates' continuity correction only for the 2×2 case; for k > 2
        # the correction is silently dropped.
        tbl = np.array([
            [int(x_arr[i]), int(n_arr[i] - x_arr[i])]
            for i in range(k)
        ])
        use_correction = correct and k == 2
        res = _sps.chi2_contingency(tbl, correction=use_correction)
        suffix = " with continuity correction" if use_correction else ""
        if k == 2:
            method = f"2-sample test for equality of proportions{suffix}"
        else:
            method = f"{k}-sample test for equality of proportions{suffix}"
        return HTest(
            method=method,
            statistic={"X-squared": float(res.statistic)},
            parameter={"df": int(res.dof)},
            p_value=float(res.pvalue),
            estimate=estimates,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x out of n",
        )
    raise NotImplementedError(
        "prop_test(): k > 1 with explicit ``p`` (vector hypothesis) "
        "not yet wired"
    )


def binom_test(
    x,
    n=None,
    *,
    p: float = 0.5,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``binom.test`` — exact binomial test for one proportion.

    ``x`` is the success count, or a length-2 ``(successes, failures)``
    vector. ``n`` is the total trials (omitted when ``x`` already has
    both counts).
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    if n is None:
        x_arr = np.asarray(x, dtype=int)
        if x_arr.shape != (2,):
            raise ValueError(
                "binom_test(): n must be provided unless x = (succ, fail)"
            )
        x_succ = int(x_arr[0])
        n = int(x_arr.sum())
    else:
        x_succ = int(x)
        n = int(n)
    res = _sps.binomtest(x_succ, n, float(p), alternative=alt)
    ci = res.proportion_ci(confidence_level=conf_level, method="exact")
    return HTest(
        method="Exact binomial test",
        statistic={"number of successes": x_succ},
        parameter={"number of trials": n},
        p_value=float(res.pvalue),
        conf_int=(float(ci.low), float(ci.high)),
        estimate={"probability of success": x_succ / n},
        null_value=float(p),
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and n",
    )


def var_test(
    x,
    y,
    *,
    ratio: float = 1.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``var.test`` — F-test for equal variances of two samples.

    ``F = (var(x) / var(y)) / ratio``; df = ``(n_x - 1, n_y - 1)``.
    CI is for the variance ratio at the requested confidence level.
    """
    x_arr = _as_array(x)
    y_arr = _as_array(y)
    n1, n2 = len(x_arr), len(y_arr)
    df1, df2 = n1 - 1, n2 - 1
    var_x = float(np.var(x_arr, ddof=1))
    var_y = float(np.var(y_arr, ddof=1))
    if var_y <= 0:
        raise ValueError("var_test(): var(y) must be positive")
    F = (var_x / var_y) / float(ratio)

    if alternative == "two.sided":
        p = 2 * min(_sps.f.cdf(F, df1, df2), _sps.f.sf(F, df1, df2))
    elif alternative == "less":
        p = float(_sps.f.cdf(F, df1, df2))
    elif alternative == "greater":
        p = float(_sps.f.sf(F, df1, df2))
    else:
        raise ValueError(f"var_test(): unknown alternative {alternative!r}")

    alpha = 1 - conf_level
    if alternative == "two.sided":
        lo = F / _sps.f.ppf(1 - alpha / 2, df1, df2)
        hi = F / _sps.f.ppf(alpha / 2, df1, df2)
    elif alternative == "less":
        lo = 0.0
        hi = F / _sps.f.ppf(alpha, df1, df2)
    else:  # greater
        lo = F / _sps.f.ppf(1 - alpha, df1, df2)
        hi = float("inf")

    return HTest(
        method="F test to compare two variances",
        statistic={"F": F},
        parameter={"num df": df1, "denom df": df2},
        p_value=float(p),
        conf_int=(float(lo), float(hi)),
        estimate={"ratio of variances": var_x / var_y},
        null_value=float(ratio),
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and y",
    )


def bartlett_test(x, g) -> HTest:
    """R's ``bartlett.test(x, g)`` — Bartlett's test for equal variances.

    ``x`` is the values vector; ``g`` is the parallel group label vector.
    Returns the K² statistic with ``k - 1`` degrees of freedom.
    """
    x_arr = _as_array(x)
    g_arr = np.asarray(g)
    if x_arr.shape != g_arr.shape:
        raise ValueError("bartlett_test(): x and g must have the same length")
    groups = [x_arr[g_arr == val] for val in np.unique(g_arr)]
    if len(groups) < 2:
        raise ValueError("bartlett_test(): need at least 2 groups")
    res = _sps.bartlett(*groups)
    return HTest(
        method="Bartlett test of homogeneity of variances",
        statistic={"Bartlett's K-squared": float(res.statistic)},
        parameter={"df": int(len(groups) - 1)},
        p_value=float(res.pvalue),
        alternative="",
        data_name="x by g",
    )


def shapiro_test(x) -> HTest:
    """R's ``shapiro.test`` — Shapiro-Wilk normality test."""
    x_arr = _as_array(x)
    res = _sps.shapiro(x_arr)
    return HTest(
        method="Shapiro-Wilk normality test",
        statistic={"W": float(res.statistic)},
        p_value=float(res.pvalue),
        alternative="",
        data_name="x",
    )


def ks_test(
    x,
    y,
    *,
    alternative: str = "two.sided",
) -> HTest:
    """R's ``ks.test`` — Kolmogorov-Smirnov test.

    ``y`` is either a second sample (two-sample test) or a string naming
    a scipy distribution (one-sample goodness-of-fit). R uses names like
    ``"pnorm"``; we accept either ``"pnorm"`` or scipy's ``"norm"``.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x_arr = _as_array(x)
    if isinstance(y, str):
        dist_name = y[1:] if y.startswith("p") and len(y) > 1 else y
        res = _sps.kstest(x_arr, dist_name, alternative=alt)
        method = "One-sample Kolmogorov-Smirnov test"
        data_name = "x"
    else:
        y_arr = _as_array(y)
        res = _sps.ks_2samp(x_arr, y_arr, alternative=alt)
        method = "Two-sample Kolmogorov-Smirnov test"
        data_name = "x and y"
    return HTest(
        method=method,
        statistic={"D": float(res.statistic)},
        p_value=float(res.pvalue),
        alternative=alternative,
        data_name=data_name,
    )


def mcnemar_test(x, y=None, *, correct: bool = True) -> HTest:
    """R's ``mcnemar.test`` — McNemar's chi-squared test on a 2×2 table.

    ``x`` is the 2×2 table or a 1-D vector paired with ``y``. With
    ``correct=True`` (R's default), uses the Yates continuity correction
    ``(|b - c| - 1)² / (b + c)``.
    """
    if y is not None:
        tbl = _crosstab(x, y)
    else:
        tbl = np.asarray(x)
    if tbl.shape != (2, 2):
        raise ValueError(
            f"mcnemar_test(): table must be 2x2 (got {tbl.shape})"
        )
    b = float(tbl[0, 1])
    c = float(tbl[1, 0])
    if b + c == 0:
        stat = 0.0
    elif correct:
        diff = max(abs(b - c) - 1, 0.0)
        stat = diff ** 2 / (b + c)
    else:
        stat = (b - c) ** 2 / (b + c)
    pval = float(_sps.chi2.sf(stat, 1))
    return HTest(
        method="McNemar's Chi-squared test"
        + (" with continuity correction" if correct else ""),
        statistic={"McNemar's chi-squared": stat},
        parameter={"df": 1},
        p_value=pval,
        alternative="",
        data_name="x" if y is None else "x and y",
    )


def friedman_test(y, groups, blocks) -> HTest:
    """R's ``friedman.test(y, groups, blocks)`` — Friedman rank-sum test.

    ``y`` is the value vector, ``groups`` and ``blocks`` are parallel
    label vectors. The data is reshaped into ``(blocks × groups)`` wide
    form before being passed to ``scipy.stats.friedmanchisquare``.
    """
    y_arr = _as_array(y)
    g_arr = np.asarray(groups)
    b_arr = np.asarray(blocks)
    if not (y_arr.shape == g_arr.shape == b_arr.shape):
        raise ValueError(
            "friedman_test(): y, groups, blocks must have the same length"
        )
    # Internal column names use ``__y__`` / ``__g__`` / ``__b__`` so user
    # group/block labels equal to "y" or "g" or "b" don't collide with
    # the temp column names after pivot.
    df = pl.DataFrame({
        "__y__": y_arr,
        "__g__": pl.Series(g_arr).cast(pl.Utf8),
        "__b__": pl.Series(b_arr).cast(pl.Utf8),
    })
    wide = df.pivot(values="__y__", index="__b__", on="__g__")
    cols = [c for c in wide.columns if c != "__b__"]
    samples = [wide[c].to_numpy().astype(float) for c in cols]
    res = _sps.friedmanchisquare(*samples)
    return HTest(
        method="Friedman rank sum test",
        statistic={"Friedman chi-squared": float(res.statistic)},
        parameter={"df": int(len(samples) - 1)},
        p_value=float(res.pvalue),
        alternative="",
        data_name="y, groups and blocks",
    )


def aov(formula: str, data: pl.DataFrame, *, type: str = "II") -> AnovaTable:
    """R's ``aov`` followed by ``car::Anova(..., type='II')``.

    Computes Type-II sums of squares by dropping one top-level term at a
    time and comparing ``ΔRSS``. Works for either form the notebook uses:
    factor formulas (``value ~ group``) or explicit-dummy formulas
    (``value ~ 1 + group_b + group_c``) — both go through ``hea.lm``,
    so the term grouping comes from the formula's own ``term_labels``.
    """
    from .lm import lm  # local import to avoid circular at package load

    fit = lm(formula, data)
    term_labels = list(fit._expanded.term_labels)
    rss_full = float(fit.rss)
    df_full = int(fit.df_residuals)

    lhs = formula.split("~", 1)[0]
    rows = []
    for term in term_labels:
        kept = [t for t in term_labels if t != term]
        reduced_rhs = " + ".join(["1"] + kept) if kept else "1"
        sub_formula = f"{lhs} ~ {reduced_rhs}"
        sub = lm(sub_formula, data)
        ss = float(sub.rss - rss_full)
        df_term = int(sub.df_residuals - df_full)
        F = (ss / df_term) / (rss_full / df_full) if df_term > 0 else None
        p = float(_sps.f.sf(F, df_term, df_full)) if F is not None else None
        rows.append(
            {
                "term": term,
                "df": df_term,
                "sum_sq": ss,
                "mean_sq": ss / df_term if df_term else None,
                "F": F,
                "p": p,
            }
        )
    return AnovaTable(
        response=fit.y.name,
        rows=rows,
        residual_df=df_full,
        residual_ss=rss_full,
        type=type,
    )


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


@dataclass
class Terms:
    """Lightweight stand-in for R's ``terms`` object.

    R's ``terms`` carries a factor matrix and many attributes; we expose
    only what hea actually keeps around: the formula string, the
    response (LHS) variable name, and the top-level term labels (the
    same list ``aov`` / ``anova`` use to build their tables).
    """

    formula: str
    response: str
    term_labels: list

    def __repr__(self) -> str:
        return (
            f"Terms(formula={self.formula!r}, response={self.response!r}, "
            f"term_labels={self.term_labels!r})"
        )


def terms(model) -> Terms:
    """R: ``terms()`` — formula structure summary.

    Returns a :class:`Terms` with the formula string, response name, and
    top-level term labels. Less than R's full terms object (no factor
    matrix, no order vector) but enough to drive things like ``anova``
    table titles or to round-trip a formula via ``update``.
    """
    f = model.formula
    if "~" not in f:
        raise ValueError(f"terms(): bad formula on {model.__class__.__name__}")
    lhs, rhs = f.split("~", 1)
    response = lhs.strip()
    if hasattr(model, "_expanded") and hasattr(model._expanded, "term_labels"):
        labels = list(model._expanded.term_labels)
    else:
        labels = [t.strip() for t in rhs.split("+") if t.strip()]
    return Terms(formula=f, response=response, term_labels=labels)


def update(model, formula, **kwargs):
    """R: ``update()`` — refit on the same data with a new formula.

    Two formula forms supported:

    * **Full formula** (e.g. ``"y ~ x1 + x2"``) — used verbatim.
    * **Delta formula** with R's ``.`` placeholder (e.g.
      ``". ~ . + x3"`` or ``"log(y) ~ . - x1"``). On each side of
      ``~``, ``.`` is substituted with the corresponding side of the
      original ``model.formula`` wrapped in parentheses, so terms can
      be added or removed without retyping.

    ``family`` is auto-forwarded for glm/gam/bam so ``update(m, "y ~ z")``
    keeps the original family. Other constructor kwargs (``weights``,
    ``method``, ``REML``, ``offset``, …) are NOT auto-forwarded — pass
    them explicitly via ``**kwargs`` if you need them carried over.
    """
    f = formula.strip()
    if "~" not in f:
        raise ValueError(f"update(): formula must contain '~'; got {f!r}")
    if "." in f:
        old_lhs, old_rhs = (s.strip() for s in model.formula.split("~", 1))
        new_lhs, new_rhs = (s.strip() for s in f.split("~", 1))
        if new_lhs == ".":
            new_lhs = old_lhs
        elif "." in new_lhs:
            new_lhs = new_lhs.replace(".", f"({old_lhs})")
        if new_rhs == ".":
            new_rhs = old_rhs
        elif "." in new_rhs:
            new_rhs = new_rhs.replace(".", f"({old_rhs})")
        f = f"{new_lhs} ~ {new_rhs}"
    cls = type(model)
    if hasattr(model, "family") and "family" not in kwargs:
        kwargs["family"] = model.family
    return cls(f, model.data, **kwargs)


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


# ---- regression diagnostics -----------------------------------------
#
# Most diagnostics are defined in terms of three primitives that hea
# already caches at fit time: ``leverage`` (h_ii), the standardized
# residuals (``std_residuals`` for lm; ``std_dev_residuals`` /
# ``std_pearson_residuals`` for glm/gam/bam), and the cross-product
# inverse ``XtXinv`` (lm). The closed-form deletion diagnostics
# (``rstudent`` / ``dffits`` / ``dfbetas`` / ``influence``) currently
# implement the lm formulas; glm/gam variants use a different jackknife
# approximation in R and are deferred to a later pass.

def hatvalues(model):
    """R: ``hatvalues()`` — leverage ``h_ii`` (hat-matrix diagonal)."""
    if hasattr(model, "leverage"):
        return np.asarray(model.leverage)
    raise TypeError(
        f"hatvalues(): {model.__class__.__name__} has no leverage"
    )


def rstandard(model, type=None):
    """R: ``rstandard()`` — internally studentized residuals.

    For ``glm`` / ``gam`` / ``bam``, ``type`` selects between
    ``"deviance"`` (default, matches ``rstandard.glm``) and ``"pearson"``.
    For ``lm``, only one form exists (Gaussian) and ``type`` is ignored.
    """
    if type == "pearson" and hasattr(model, "std_pearson_residuals"):
        return np.asarray(model.std_pearson_residuals)
    if type not in (None, "deviance", "pearson"):
        raise ValueError(
            f"rstandard(): type={type!r} not recognized "
            "(use 'deviance' or 'pearson')"
        )
    if hasattr(model, "std_dev_residuals"):
        return np.asarray(model.std_dev_residuals)
    if hasattr(model, "std_residuals"):
        return np.asarray(model.std_residuals)
    raise TypeError(
        f"rstandard(): {model.__class__.__name__} has no standardized residuals"
    )


def _loo_sigma_lm(model) -> np.ndarray:
    """Leave-one-out σ estimates ``σ_(-i)`` for an unweighted ``lm``.

    Uses the closed form
    ``σ_(-i)^2 = (RSS - e_i^2 / (1 - h_i)) / (n - p - 1)``.
    Raises if the fit was weighted or if ``n - p - 1 ≤ 0``.
    """
    if getattr(model, "weights", None) is not None:
        raise NotImplementedError(
            "deletion diagnostics for weighted lm not implemented yet"
        )
    e = model.residuals.to_series().to_numpy()
    h = np.asarray(model.leverage)
    rss = float(model.rss)
    n = int(model.n)
    p = int(model.p)
    df_loo = n - p - 1
    if df_loo <= 0:
        raise ValueError(
            "deletion diagnostics need n - p - 1 > 0; "
            f"got n={n}, p={p}"
        )
    one_minus_h = np.clip(1 - h, 1e-12, None)
    rss_loo = rss - (e ** 2) / one_minus_h
    return np.sqrt(np.maximum(rss_loo, 0.0) / df_loo)


def _xtwxinv_glm_gam(model) -> np.ndarray:
    """Return the cached ``(X'WX + S)^-1`` for glm/gam/bam (penalty included).

    Derived from the model's vcov: ``V_bhat = dispersion · (X'WX)^-1`` for
    glm; ``Vp = scale · (X'WX + S)^-1`` for gam/bam.
    """
    if hasattr(model, "Vp"):  # gam / bam
        return np.asarray(model.Vp) / float(model.scale)
    if hasattr(model, "V_bhat"):  # glm
        return np.asarray(model.V_bhat) / float(model.dispersion)
    raise AttributeError(
        f"{model.__class__.__name__}: no Vp / V_bhat for jackknife inputs"
    )


def _loo_sigma_glm_gam(model) -> np.ndarray:
    """Leave-one-out σ estimates for glm/gam/bam.

    Known-scale families (Binomial, Poisson, …) return ``ones`` since
    R's ``influence.glm`` fixes σ at 1. Unknown-scale families use the
    same closed form as ``lm``, swapping RSS for total deviance:
    ``σ_(-i)^2 = (deviance - d_i^2 / (1 - h_i)) / (n - p - 1)`` where
    ``d_i`` is the raw deviance residual (so ``d_i^2`` equals the per-
    observation deviance contribution).
    """
    h = np.asarray(model.leverage)
    if model.family.scale_known:
        return np.ones_like(h)
    n = int(model.n)
    p = int(model.p)
    df_loo = n - p - 1
    if df_loo <= 0:
        raise ValueError(
            f"deletion diagnostics need n - p - 1 > 0; got n={n}, p={p}"
        )
    d = np.asarray(model.residuals_of("deviance"))
    one_minus_h = np.clip(1 - h, 1e-12, None)
    sigma_sq = (float(model.deviance) - d ** 2 / one_minus_h) / df_loo
    return np.sqrt(np.maximum(sigma_sq, 0.0))


def _design_full(model) -> np.ndarray:
    """Return the full design matrix as an ndarray.

    For ``gam`` / ``bam``, ``model.X`` only carries the parametric
    columns; the full penalised design (parametric + spline bases) is
    stashed privately as ``_X_full``.
    """
    if hasattr(model, "_X_full"):
        return np.asarray(model._X_full, dtype=float)
    return model.X.to_numpy().astype(float)


def _irls_inputs(model) -> dict:
    """Inputs for closed-form glm/gam jackknife diagnostics.

    Returns a dict with:

    * ``X`` — full design matrix (``n × p``) as ndarray
    * ``XtWXinv`` — penalised cross-product inverse, ``Vp/scale`` or
      ``V_bhat/dispersion``
    * ``w_irls`` — IRLS working weights, recovered from leverage via
      ``h_i = w_i · x_i' (X'WX)^{-1} x_i``
    * ``working_resid`` — ``(y - μ) / g'(μ)`` (R's ``glm$residuals``)
    * ``h`` — leverage diagonal
    * ``sigma_loo`` — leave-one-out σ
    """
    h = np.asarray(model.leverage)
    X = _design_full(model)
    XtWXinv = _xtwxinv_glm_gam(model)

    hX = X @ XtWXinv
    quad = (hX * X).sum(axis=1)
    safe_quad = np.where(quad > 0, quad, 1.0)
    w_irls = h / safe_quad

    mu = np.asarray(model.fitted_values, dtype=float)
    eta = np.asarray(model.linear_predictors, dtype=float)
    y_arr = (
        model.y.to_numpy().astype(float)
        if isinstance(model.y, pl.Series)
        else np.asarray(model.y, dtype=float)
    )
    mu_eta = np.asarray(model.family.link.mu_eta(eta), dtype=float)
    safe_mu_eta = np.where(mu_eta != 0, mu_eta, 1.0)
    working_resid = (y_arr - mu) / safe_mu_eta

    sigma_loo = _loo_sigma_glm_gam(model)

    return {
        "X": X,
        "XtWXinv": XtWXinv,
        "w_irls": w_irls,
        "working_resid": working_resid,
        "h": h,
        "sigma_loo": sigma_loo,
    }


def rstudent(model):
    """R: ``rstudent()`` — externally studentized residuals.

    For ``lm`` (Gaussian), uses the closed form
    ``r_i^* = r_i · √((n-p-1) / (n-p - r_i^2))`` derived from the
    leave-one-out σ estimate.

    For ``glm`` / ``gam`` / ``bam``, follows R's ``rstudent.glm`` —
    the Williams (1987) likelihood residual:
    ``r_i = sign(d_i) · √(d_i² + p_i² · h_i / (1-h_i)) / (σ_(-i) · √(1-h_i))``
    where ``d_i`` and ``p_i`` are raw deviance and Pearson residuals.
    Known-scale families fix ``σ_(-i) = 1``.
    """
    # lm path — closed form on internally studentized residuals
    if hasattr(model, "std_residuals"):
        if getattr(model, "weights", None) is not None:
            raise NotImplementedError(
                "rstudent(): weighted lm not implemented yet"
            )
        r = np.asarray(model.std_residuals)
        n = int(model.n)
        p = int(model.p)
        df_resid = n - p
        if df_resid - 1 <= 0:
            raise ValueError(
                f"rstudent(): need n - p - 1 > 0; got n={n}, p={p}"
            )
        return r * np.sqrt(
            (df_resid - 1) / np.clip(df_resid - r ** 2, 1e-12, None)
        )

    # glm / gam / bam path — Williams' likelihood residual
    if not hasattr(model, "residuals_of"):
        raise TypeError(
            f"rstudent(): {model.__class__.__name__} not supported"
        )
    h = np.asarray(model.leverage)
    one_minus_h = np.clip(1 - h, 1e-12, None)
    d = np.asarray(model.residuals_of("deviance"))
    pe = np.asarray(model.residuals_of("pearson"))
    likelihood_r = np.sign(d) * np.sqrt(d ** 2 + (pe ** 2) * h / one_minus_h)
    sigma_loo = _loo_sigma_glm_gam(model)
    return likelihood_r / (sigma_loo * np.sqrt(one_minus_h))


def cooks_distance(model):
    """R: ``cooks.distance()`` — Cook's distance for each observation.

    Uses the unified formula
    ``D_i = r_i^2 · h_i / ((1 - h_i) · p)`` where ``r_i`` is the
    standardized residual (deviance for glm/gam/bam, ordinary for lm)
    and ``p`` is the effective parameter count. R's ``cooks.distance.lm``
    uses ``model.p``; ``cooks.distance.glm`` uses ``sum(hat)`` — we
    follow that split to match R numerically.
    """
    h = hatvalues(model)
    one_minus_h = np.clip(1 - h, 1e-12, None)
    if hasattr(model, "std_pearson_residuals"):  # glm / gam / bam
        r = np.asarray(model.std_pearson_residuals)
        p = float(np.sum(h))  # matches R's cooks.distance.glm
    elif hasattr(model, "std_residuals"):  # lm
        r = np.asarray(model.std_residuals)
        p = float(model.p)
    else:
        raise TypeError(
            f"cooks_distance(): {model.__class__.__name__} not supported"
        )
    if p <= 0:
        raise ValueError("cooks_distance(): effective parameter count is zero")
    return r ** 2 * h / (one_minus_h * p)


def dffits(model):
    """R: ``dffits()``.

    For ``lm``, uses ``DFFITS_i = r_i^* · √(h_i / (1 - h_i))``.
    For ``glm`` / ``gam`` / ``bam``, follows ``stats:::dffits`` exactly:
    ``DFFITS_i = p_i · √(h_i) / (σ_(-i) · (1 - h_i))`` where ``p_i`` is
    the raw response-scale Pearson residual.
    """
    if hasattr(model, "std_residuals"):  # lm
        rs = rstudent(model)
        h = hatvalues(model)
        return rs * np.sqrt(h / np.clip(1 - h, 1e-12, None))

    if not hasattr(model, "residuals_of"):
        raise TypeError(
            f"dffits(): {model.__class__.__name__} not supported"
        )
    h = np.asarray(model.leverage)
    one_minus_h = np.clip(1 - h, 1e-12, None)
    pe = np.asarray(model.residuals_of("pearson"))
    sigma_loo = _loo_sigma_glm_gam(model)
    return pe * np.sqrt(h) / (sigma_loo * one_minus_h)


def dfbetas(model):
    """R: ``dfbetas()`` — standardized leave-one-out coefficient changes.

    Returns an ``n × p`` polars DataFrame whose columns are the design
    columns (``(Intercept)``, predictors, …). Element ``[i, j]`` is the
    change in ``β̂_j`` when observation ``i`` is dropped, scaled by
    ``σ_(-i) · √(diag((X'X)^{-1})_j)``.

    For ``lm``: closed form using ``XtXinv``. For ``glm`` / ``gam`` /
    ``bam``: IRLS closed form using ``Vp/scale`` (or ``V_bhat/dispersion``)
    and IRLS working weights recovered from ``leverage``.
    """
    # lm path
    if hasattr(model, "XtXinv"):
        if getattr(model, "weights", None) is not None:
            raise NotImplementedError(
                "dfbetas(): weighted lm not implemented yet"
            )
        X = model.X.to_numpy().astype(float)
        XtXinv = np.asarray(model.XtXinv)
        e = model.residuals.to_series().to_numpy()
        h = hatvalues(model)
        one_minus_h = np.clip(1 - h, 1e-12, None)
        sigma_loo = _loo_sigma_lm(model)
        delta = (X @ XtXinv) * (e / one_minus_h)[:, None]
        sd_j = np.sqrt(np.diag(XtXinv))
        sd_j = np.where(sd_j > 0, sd_j, 1.0)
        out = delta / (sigma_loo[:, None] * sd_j[None, :])
        return pl.DataFrame(
            {col: out[:, i] for i, col in enumerate(model.column_names)}
        )

    # glm / gam / bam path — IRLS closed form
    if not hasattr(model, "residuals_of"):
        raise TypeError(
            f"dfbetas(): {model.__class__.__name__} not supported"
        )
    inputs = _irls_inputs(model)
    X, XtWXinv = inputs["X"], inputs["XtWXinv"]
    w_irls = inputs["w_irls"]
    working_resid = inputs["working_resid"]
    h = inputs["h"]
    sigma_loo = inputs["sigma_loo"]
    one_minus_h = np.clip(1 - h, 1e-12, None)

    # IRLS leave-one-out:
    # β̂ - β̂_(-i) = (X'WX)^{-1} · X_i · w_i · z_i / (1 - h_i)
    # where z_i is the working residual.
    delta = (X @ XtWXinv) * (w_irls * working_resid / one_minus_h)[:, None]
    sd_j = np.sqrt(np.diag(XtWXinv))
    sd_j = np.where(sd_j > 0, sd_j, 1.0)
    out = delta / (sigma_loo[:, None] * sd_j[None, :])
    return pl.DataFrame(
        {col: out[:, i] for i, col in enumerate(model.column_names)}
    )


def influence(model):
    """R: ``influence()`` / ``lm.influence()`` — deletion diagnostics bundle.

    Returns a dict mirroring R's ``lm.influence(do.coef=TRUE)``:

    * ``hat`` — leverage ``h_ii`` (ndarray, length ``n``)
    * ``sigma`` — leave-one-out σ estimates ``σ_(-i)`` (ndarray, length ``n``)
    * ``coefficients`` — leave-one-out coefficient *deltas*
      ``β̂ - β̂_(-i)`` as an ``n × p`` DataFrame named like the design
    * ``residuals`` — for ``lm``, response residuals; for ``glm`` /
      ``gam`` / ``bam``, working residuals (matches R's
      ``influence.glm`` "wt.res")
    """
    # lm path
    if hasattr(model, "XtXinv"):
        if getattr(model, "weights", None) is not None:
            raise NotImplementedError(
                "influence(): weighted lm not implemented yet"
            )
        X = model.X.to_numpy().astype(float)
        XtXinv = np.asarray(model.XtXinv)
        e = model.residuals.to_series().to_numpy()
        h = np.asarray(model.leverage)
        one_minus_h = np.clip(1 - h, 1e-12, None)
        delta = (X @ XtXinv) * (e / one_minus_h)[:, None]
        return {
            "hat": h,
            "sigma": _loo_sigma_lm(model),
            "coefficients": pl.DataFrame(
                {col: delta[:, i] for i, col in enumerate(model.column_names)}
            ),
            "residuals": e,
        }

    # glm / gam / bam path — IRLS closed form
    if not hasattr(model, "residuals_of"):
        raise TypeError(
            f"influence(): {model.__class__.__name__} not supported"
        )
    inputs = _irls_inputs(model)
    X, XtWXinv = inputs["X"], inputs["XtWXinv"]
    w_irls = inputs["w_irls"]
    working_resid = inputs["working_resid"]
    h = inputs["h"]
    sigma_loo = inputs["sigma_loo"]
    one_minus_h = np.clip(1 - h, 1e-12, None)
    delta = (X @ XtWXinv) * (w_irls * working_resid / one_minus_h)[:, None]
    return {
        "hat": h,
        "sigma": sigma_loo,
        "coefficients": pl.DataFrame(
            {col: delta[:, i] for i, col in enumerate(model.column_names)}
        ),
        "residuals": working_resid,
    }
