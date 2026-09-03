"""Polars API shims shared across hea's subpackages.

A leaf module: imports nothing but ``polars``, so ``hea.formula``,
``hea.tidy``, ``hea.ggplot``, ``hea.plot``, ``hea.R`` and ``hea.translate``
can all reach it without an import cycle.
"""

from __future__ import annotations

import polars as pl

__all__ = ["cat_pool"]


def cat_pool(s: pl.Series) -> pl.Series:
    """Category pool of a Categorical/Enum series, as a ``pl.Series``.

    ``dtype.categories`` is two different types depending on the dtype:
    an ``Enum`` carries its declared levels as a ``pl.Series`` already,
    while a ``Categorical`` carries a ``Categories`` pool object that has
    to be materialized. Both are returned here as a plain Series, indexed
    so that ``pool[code]`` inverts ``s.to_physical()``.

    Note this is the *pool*, not the values present: for a Categorical the
    pool is process-global and holds every string any Categorical column
    has interned in this process. Callers that want R's ``levels()`` must
    still restrict to the codes the column actually uses.
    """
    cats = s.dtype.categories
    return cats if isinstance(cats, pl.Series) else cats.to_series()
