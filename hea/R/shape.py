"""R's data-frame shape / preview verbs: ``head``, ``tail``, ``nrow`` /
``ncol`` / ``dim`` / ``length``, ``names`` / ``colnames``, ``summary``,
``complete.cases``, ``na.omit``.
"""
from __future__ import annotations

import numpy as np
import polars as pl


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

    Special case: emmeans contrasts DataFrames (``rem.contrasts``) route
    to ``hea.emmeans.summary_emmgrid_contrasts`` so R's
    ``summary(rem$contrasts, infer=TRUE, adjust="bonferroni")`` works
    without the caller knowing which package the table came from.
    """
    if isinstance(x, pl.DataFrame) and {"contrast", "estimate", "t.ratio"} <= set(x.columns):
        from .emmeans import summary_emmgrid_contrasts
        return summary_emmgrid_contrasts(x, **kwargs)
    if hasattr(x, "summary"):
        return x.summary(**kwargs)
    raise TypeError(
        f"summary(): {type(x).__name__} has no .summary() method"
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
