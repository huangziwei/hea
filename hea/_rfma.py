"""The per-arch R-parity fused multiply-add, and nothing else.

A leaf on purpose. This lived in :mod:`hea.R._shared`, which put every
consumer behind ``hea/R/__init__.py`` -- and two of them, :mod:`hea.formula`
and :mod:`hea.models.bam`, are not in ``hea.R`` at all. That was a real import
cycle (``hea.formula`` -> ``hea.R`` -> ``hea.R.factor`` -> ``hea.formula``),
survivable only while ``hea/__init__.py`` happened to import the two in a
working order; a lazy top level does not, and it surfaced immediately. It was
also import weight: a three-line numeric primitive was dragging in the whole
base-R namespace.

:mod:`hea.R._shared` re-exports both names, so every ``from ._shared import
_rfma`` inside ``hea.R`` is unchanged.
"""

from __future__ import annotations

import math
import platform

import numpy as np

_R_FMA = platform.machine().lower() in ("arm64", "aarch64") and hasattr(math, "fma")
if _R_FMA:

    def _rfma(a, b, c):
        try:
            return math.fma(a, b, c)
        except (OverflowError, ValueError):
            return a * b + c

    _rfma_ufunc = np.frompyfunc(_rfma, 3, 1)

    def _rfma_vec(a, b, c):
        return _rfma_ufunc(a, b, c).astype(np.float64)
else:

    def _rfma(a, b, c):
        return a * b + c

    def _rfma_vec(a, b, c):
        return a * b + c
