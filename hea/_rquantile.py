"""R's ``quantile(x, probs)`` for the default ``type = 7``
(``quantile.default``, R's stats/R/quantile.R), as R evaluates it.

A leaf module, so :mod:`hea.formula` can place knots with it without importing
:mod:`hea.R`.
"""

from __future__ import annotations

import numpy as np


def r_quantile7(x, probs) -> np.ndarray:
    """``quantile(x, probs, names = FALSE)``, type 7, for finite ``x``:
    ``index <- 1 + (n - 1) * probs``, ``qs <- x[floor(index)]`` of the sorted
    ``x``, and where ``index`` is fractional and the bracketing order
    statistics differ, ``(1 - h) * qs + h * x[ceiling(index)]`` with
    ``h = index - floor(index)``, each operation rounded on its own."""
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    p = np.atleast_1d(np.asarray(probs, dtype=float))
    p = np.maximum(0.0, np.minimum(1.0, p))
    n = xs.size
    index = 1.0 + float(max(n - 1, 0)) * p
    lo = np.floor(index)
    hi = np.ceil(index)
    qs = xs[lo.astype(np.int64) - 1].copy()
    x_hi = xs[hi.astype(np.int64) - 1]
    i = (index > lo) & (x_hi != qs)
    h = (index - lo)[i]
    qs[i] = (1.0 - h) * qs[i] + h * x_hi[i]
    return qs
