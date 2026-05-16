"""R's type coercion functions: ``as.numeric`` / ``as.integer`` /
``as.character`` / ``as.logical`` / ``as.Date``.

Each dispatches on input type (Expr/Series → polars cast, else numpy
``asarray(..., dtype=...)`` ). ``as_Date`` is an alias for ``as_date``
(R uses a dot we can't spell in Python).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def as_numeric(x):
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.cast(pl.Float64)
    return np.asarray(x, dtype=float)


def as_integer(x):
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.cast(pl.Int64)
    return np.asarray(x, dtype=np.int64)


def as_character(x):
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.cast(pl.Utf8)
    return np.asarray(x, dtype=str)


def as_logical(x):
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.cast(pl.Boolean)
    return np.asarray(x, dtype=bool)


def as_date(x, format=None):
    """R/lubridate's ``as_date()`` — coerce strings to ``Date``.

    Works on polars expressions (typical use inside ``mutate()``) and on
    eager Series; both delegate to ``str.to_date``. Pass ``format=`` for
    a strptime pattern; omit for ISO-8601 auto-detect (matches
    lubridate's default).

    Examples
    --------
    >>> import hea
    >>> from hea import col
    >>> from hea.R import as_date
    >>> hea.DataFrame({"s": ["2024-01-15"]}).mutate(d=as_date(col("s")))  # doctest: +SKIP

    Already-``Date`` inputs pass through (cast to ``pl.Date``).
    Numeric / epoch-day input isn't supported here — use
    ``pl.from_epoch(x, time_unit='d')`` for that.
    """
    if isinstance(x, pl.Expr):
        return x.str.to_date(format=format)
    if isinstance(x, pl.Series):
        if x.dtype == pl.Date:
            return x
        if x.dtype == pl.Datetime:
            return x.dt.date()
        return x.str.to_date(format=format)
    return pl.Series(x).str.to_date(format=format)


# R's base spelling (``as.Date``) — Python can't have a dot, so the
# convention here is the camel-cased underscore form. Same function.
as_Date = as_date
