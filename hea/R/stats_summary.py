"""R-shaped summary reductions: ``mean`` / ``median`` / ``var`` / ``sd``,
``quantile`` / ``IQR``, ``cor`` / ``cov``.

All default ``na_rm=True`` (diverges from R's ``na.rm=FALSE`` to match
polars' null-skip convention — hea's house default).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def mean(x, na_rm=True):
    """R: ``mean()`` — arithmetic mean.

    ``na_rm=True`` is hea's default — diverges from R's ``na.rm=FALSE``
    to match polars' null-skip convention (and what users almost always
    want). Pass ``na_rm=False`` to recover R's strict NA-in → NA-out
    behavior.

    Dispatches: ``pl.Expr`` → ``pl.Expr`` (scalar that broadcasts inside
    ``mutate``); ``pl.Series`` → Python scalar; list / ndarray → float.
    """
    if isinstance(x, pl.Expr):
        if na_rm:
            return x.mean()
        return pl.when(x.is_null().any()).then(None).otherwise(x.mean())
    if isinstance(x, pl.Series):
        if not na_rm and x.null_count() > 0:
            return None
        return x.mean()
    arr = np.asarray(x, dtype=float)
    if na_rm:
        return float(np.nanmean(arr))
    return float(np.mean(arr))


def median(x, na_rm=True):
    """R: ``median()`` — 50th percentile.

    ``na_rm=True`` default — see :func:`mean` for the rationale.
    """
    if isinstance(x, pl.Expr):
        if na_rm:
            return x.median()
        return pl.when(x.is_null().any()).then(None).otherwise(x.median())
    if isinstance(x, pl.Series):
        if not na_rm and x.null_count() > 0:
            return None
        return x.median()
    arr = np.asarray(x, dtype=float)
    if na_rm:
        return float(np.nanmedian(arr))
    return float(np.median(arr))


def var(x, y=None, na_rm=True):
    """R: variance with N-1 denominator. ``var(x, y)`` returns covariance.

    ``na_rm=True`` default — see :func:`mean` for the rationale.
    Dispatches like :func:`mean` for the unary form. The binary form
    delegates to :func:`cov` (currently eager-only — no clean polars
    top-level for a 2-vector covariance).
    """
    if y is not None:
        return cov(x, y, na_rm=na_rm)
    if isinstance(x, pl.Expr):
        if na_rm:
            return x.var(ddof=1)
        return pl.when(x.is_null().any()).then(None).otherwise(x.var(ddof=1))
    if isinstance(x, pl.Series):
        if not na_rm and x.null_count() > 0:
            return None
        return x.var(ddof=1)
    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    return float(np.var(arr, ddof=1))


def sd(x, na_rm=True):
    """R: standard deviation, N-1 denominator.

    ``na_rm=True`` default — see :func:`mean` for the rationale.
    """
    if isinstance(x, pl.Expr):
        if na_rm:
            return x.std(ddof=1)
        return pl.when(x.is_null().any()).then(None).otherwise(x.std(ddof=1))
    if isinstance(x, pl.Series):
        if not na_rm and x.null_count() > 0:
            return None
        return x.std(ddof=1)
    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    return float(np.std(arr, ddof=1))


# R's ``quantile(..., type=k)`` maps to numpy's ``method=`` argument.
# Indexed 1-9 to match R; index 0 is unused.
_R_QUANTILE_METHOD = (
    None,                          # 0 — unused (R types are 1..9)
    "inverted_cdf",                 # 1
    "averaged_inverted_cdf",        # 2
    "closest_observation",          # 3
    "interpolated_inverted_cdf",    # 4
    "hazen",                        # 5
    "weibull",                      # 6
    "linear",                       # 7 — R default
    "median_unbiased",              # 8
    "normal_unbiased",              # 9
)


def IQR(x, na_rm=True, type=7):
    """R: ``IQR()`` — interquartile range, ``Q3 - Q1``.

    Mirrors ``stats::IQR(x, na.rm, type = 7)`` — but ``na_rm`` defaults
    to ``True`` (matches the rest of hea's R-shaped API; diverges from
    R's ``na.rm=FALSE``). The eager path supports all nine R quantile
    types (mapped to numpy's ``method=`` keyword). The polars (Expr /
    Series) path supports only ``type=7`` — polars' ``Expr.quantile``
    exposes ``"linear"`` interpolation only.

    Parameters
    ----------
    x : str | pl.Expr | pl.Series | list | tuple | ndarray
        Column name (resolved to ``pl.col(name)``), or any vector-shape
        input.
    na_rm : bool, default True
        Drop nulls / NaN before computing. Pass ``False`` for R's
        strict NA-in → null-out behavior (hea is more graceful than R,
        which raises in that case).
    type : int in 1..9, default 7
        R's quantile algorithm. ``type=7`` is the dplyr/R default.

    Returns
    -------
    pl.Expr (for Expr/str input — broadcast inside ``mutate``),
    Python float (for Series / list / ndarray; ``None`` if input has NA
    and ``na_rm=False``).
    """
    if not (1 <= int(type) <= 9):
        raise ValueError(f"IQR(type={type}): expected an integer in 1..9.")

    # String column-name shorthand (polars convention; lets ``IQR("col")``
    # work the same way ``pl.quantile("col", 0.5)`` does).
    if isinstance(x, str):
        x = pl.col(x)

    if isinstance(x, pl.Expr):
        if int(type) != 7:
            raise NotImplementedError(
                f"IQR(Expr, type={type}): polars only supports linear "
                "interpolation (type=7) for Expr/Series. Materialize the "
                "column (.to_list() or .to_numpy()) for other types."
            )
        diff = (
            x.quantile(0.75, interpolation="linear")
            - x.quantile(0.25, interpolation="linear")
        )
        if na_rm:
            return diff
        return pl.when(x.is_null().any()).then(None).otherwise(diff)

    if isinstance(x, pl.Series):
        if int(type) != 7:
            raise NotImplementedError(
                f"IQR(Series, type={type}): see IQR(Expr) docstring."
            )
        if not na_rm and x.null_count() > 0:
            return None
        return (
            x.quantile(0.75, interpolation="linear")
            - x.quantile(0.25, interpolation="linear")
        )

    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    method = _R_QUANTILE_METHOD[int(type)]
    q1, q3 = np.quantile(arr, [0.25, 0.75], method=method)
    return float(q3 - q1)


def quantile(x, probs=(0, 0.25, 0.5, 0.75, 1.0), na_rm=True):
    """R: ``quantile()`` — linear interpolation, R type 7.

    ``na_rm=True`` default — see :func:`mean` for the rationale.
    For ``pl.Expr`` / ``pl.Series`` inputs, ``probs`` must be a scalar
    (polars has no native batch-quantile expression). List-probs goes
    through the eager numpy path.
    """
    is_scalar = np.isscalar(probs)
    if isinstance(x, pl.Expr):
        if not is_scalar:
            raise TypeError(
                "quantile(): list-of-probs only supported for eager inputs; "
                "for Expr input, call quantile(col, p) with a scalar p"
            )
        q = x.quantile(probs, interpolation="linear")
        if na_rm:
            return q
        return pl.when(x.is_null().any()).then(None).otherwise(q)
    if isinstance(x, pl.Series) and is_scalar:
        if not na_rm and x.null_count() > 0:
            return None
        return x.quantile(probs, interpolation="linear")
    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    return np.quantile(arr, probs)


def cor(x, y=None, na_rm=True):
    """R: Pearson correlation. ``cor(matrix)`` or ``cor(x, y)``.

    ``na_rm=True`` default — see :func:`mean` for the rationale.
    For ``pl.Expr`` / ``pl.Series`` inputs in the binary form, dispatches
    to ``pl.corr``. The matrix form (``y=None``) is eager-only — pass a
    2D ndarray.
    """
    if y is None:
        arr = np.asarray(x, dtype=float)
        if na_rm:
            arr = arr[~np.isnan(arr).any(axis=1)]
        return np.corrcoef(arr, rowvar=False)
    # Binary form. Expr/Series dispatch routes to pl.corr.
    if isinstance(x, (pl.Expr, pl.Series)) or isinstance(y, (pl.Expr, pl.Series)):
        a = x if isinstance(x, pl.Expr) else (
            x.to_frame().to_series() if isinstance(x, pl.Series) else pl.Series(x)
        )
        b = y if isinstance(y, pl.Expr) else (
            y.to_frame().to_series() if isinstance(y, pl.Series) else pl.Series(y)
        )
        # na_rm: drop pairs where either is null
        if na_rm and isinstance(a, pl.Expr) and isinstance(b, pl.Expr):
            mask = ~(a.is_null() | b.is_null())
            a, b = a.filter(mask), b.filter(mask)
        return pl.corr(a, b, method="pearson")
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if na_rm:
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
    return float(np.corrcoef(a, b)[0, 1])


def cov(x, y=None, na_rm=True):
    """R: sample covariance, N-1 denominator. Currently eager-only —
    polars has no top-level covariance expression. For an Expr-context
    covariance use ``((x - x.mean()) * (y - y.mean())).sum() / (n - 1)``.

    ``na_rm=True`` default — see :func:`mean` for the rationale.
    """
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
