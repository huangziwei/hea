"""Small helpers for hea.gg — color/shape parsing, group computation."""

from __future__ import annotations

import re

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


_GREY_RX = re.compile(r"^gr(a|e)y(\d{1,3})$", re.IGNORECASE)


def r_color(c):
    """Translate an R-flavoured colour name into something matplotlib accepts.

    Handles the ``grey0``..``grey100`` / ``gray0``..``gray100`` family that R
    exposes programmatically (matplotlib only takes ``"gray"`` ≡ 50%). Other
    names pass through unchanged — matplotlib already knows ``"red"``,
    ``"black"``, hex codes, RGBA tuples, etc.
    """
    if c is None:
        return None
    if not isinstance(c, str):
        return c
    m = _GREY_RX.match(c)
    if m:
        n = int(m.group(2))
        n = max(0, min(100, n))
        return f"{n / 100:.4f}"
    return c
