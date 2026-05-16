"""R's functional iteration: ``tapply`` (group apply) and ``sapply``
(simplify-apply). Both produce :class:`~hea.R.NamedVector` to mirror R's
"named numeric vector" return shape.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ._shared import NamedVector


def tapply(X, INDEX, FUN, *args, **kwargs):
    """R: ``tapply(X, INDEX, FUN, ...)`` — apply ``FUN`` to subsets of
    ``X`` grouped by the levels of ``INDEX``.

    Returns a :class:`hea.NamedVector` keyed by the unique levels of
    ``INDEX``, matching R's "named numeric vector" return shape.
    """

    x = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    idx = INDEX.to_numpy() if hasattr(INDEX, "to_numpy") else np.asarray(INDEX)
    levels = sorted(set(idx.tolist()), key=str)
    names = [str(lvl) for lvl in levels]
    values = [FUN(x[idx == lvl], *args, **kwargs) for lvl in levels]
    return NamedVector(names, values)


def sapply(X, FUN, *args, **kwargs):
    """R: ``sapply()`` — apply ``FUN`` to each element of ``X`` and
    simplify the result.

    - Scalar results → 1-D numpy array.
    - Same-shape vector results → 2-D matrix (columns = X positions,
      matching R's ``sapply`` convention).
    - Mixed → list (R's ``lapply`` fallback).
    """
    # Iterate respecting R's "elements of X." For a vector/list, it's
    # the elements. For a polars Series, the values. For a NamedVector,
    # the values too (names are dropped on the FUN inputs but propagated
    # to the column labels of the result matrix).

    if isinstance(X, NamedVector):
        names = X.names
        items = list(X.values)
    elif isinstance(X, pl.Series):
        names = X.to_list()
        items = list(X.to_list())
    elif isinstance(X, dict):
        names = list(X.keys())
        items = list(X.values())
    else:
        items = list(X)
        names = [str(x) for x in items]

    results = [FUN(it, *args, **kwargs) for it in items]
    if not results:
        return np.asarray([])

    if all(np.isscalar(r) or (isinstance(r, np.ndarray) and r.ndim == 0) for r in results):
        return np.asarray(results)
    arrs = [np.asarray(r).ravel() for r in results]
    sizes = {len(a) for a in arrs}
    if len(sizes) == 1:
        # Matrix: columns are X positions (R's convention).
        return np.stack(arrs, axis=1)
    return results  # fall back to a Python list (R's ``lapply`` shape)
