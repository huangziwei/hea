"""R's element-wise predicates: ``is.na`` / ``is.null`` / ``is.finite`` /
``is.numeric``. (``is.factor`` lives in :mod:`hea.R.factor` alongside
``factor`` / ``ordered`` / ``levels``.)
"""
from __future__ import annotations

import numpy as np
import polars as pl


def is_na(x):
    """R: ``is.na()`` — element-wise NaN / null."""
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.is_null()
    if isinstance(x, pl.DataFrame):
        # polars DataFrame has no top-level .is_null(); per-column is the
        # idiom. Result is a same-shape DataFrame of booleans.
        return x.select(pl.all().is_null())
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
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.is_finite()
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
