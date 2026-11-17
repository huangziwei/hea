"""The per-arch R-parity fused multiply-add.

R's C sources are compiled with clang on arm64, which contracts a
single-expression ``a*b + c`` into one fused ``fmadd`` (one rounding); on
x86-64 the default build has no FMA and rounds twice. ``_rfma`` reproduces
that per architecture for scalars, ``_rfma_vec`` elementwise for arrays, and
``_rfma_dot`` for the in-order dot product of a plain C loop
``s = 0; for (i) s += x[i]*y[i];``.

A leaf module, so :mod:`hea.formula` and :mod:`hea.models.bam` can use it
without importing :mod:`hea.R` (which imports both of them).
:mod:`hea.R._shared` re-exports the names for the ``hea.R`` modules.
"""

from __future__ import annotations

import math
import platform

import numpy as np

from ._dispatch import rs_fn

_R_FMA = platform.machine().lower() in ("arm64", "aarch64") and hasattr(math, "fma")
_rfma_vec_rs = rs_fn("rfma_vec")
_rfma_dot_rs = rs_fn("rfma_dot")

if _R_FMA:

    def _rfma(a, b, c):
        try:
            return math.fma(a, b, c)
        except (OverflowError, ValueError):
            return a * b + c

    _rfma_ufunc = np.frompyfunc(_rfma, 3, 1)

    def _rfma_vec(a, b, c):
        if _rfma_vec_rs is not None:
            a, b, c = np.broadcast_arrays(
                np.asarray(a, dtype=np.float64),
                np.asarray(b, dtype=np.float64),
                np.asarray(c, dtype=np.float64),
            )
            shape = a.shape
            out = _rfma_vec_rs(
                np.ascontiguousarray(a).ravel(),
                np.ascontiguousarray(b).ravel(),
                np.ascontiguousarray(c).ravel(),
            )
            return np.asarray(out).reshape(shape)
        return _rfma_ufunc(a, b, c).astype(np.float64)

else:

    def _rfma(a, b, c):
        return a * b + c

    def _rfma_vec(a, b, c):
        return a * b + c


def _rfma_dot(x, y) -> float:
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    if _rfma_dot_rs is not None:
        return float(_rfma_dot_rs(x, y))
    s = 0.0
    for xi, yi in zip(x.tolist(), y.tolist()):
        s = _rfma(xi, yi, s)
    return s
