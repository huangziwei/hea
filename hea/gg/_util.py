"""Small helpers for hea.gg — color/shape parsing, group computation."""

from __future__ import annotations

import numpy as np
import polars as pl


def to_series(x, length: int, name: str = "value") -> pl.Series:
    """Coerce a scalar / array / Series into a polars Series of given length."""
    if isinstance(x, pl.Series):
        if len(x) != length:
            raise ValueError(f"length mismatch: aes value has {len(x)} rows, data has {length}")
        return x.alias(name)
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = np.repeat(arr, length)
    if len(arr) != length:
        raise ValueError(f"length mismatch: aes value has {len(arr)} rows, data has {length}")
    return pl.Series(name=name, values=arr)
