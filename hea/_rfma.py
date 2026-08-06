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

# --- Per-arch R-parity fused multiply-add (mirrors rust ``nmath::util::rfma``) -
# R's nmath/stats C is built ``clang -O2`` (no -ffp-contract flag); clang's
# default fuses ``a*b + c`` within one C expression to ``fmadd`` ONLY where the
# ISA has baseline FMA (aarch64: yes; generic x86-64: no). To stay 0-ulp to the
# *live* R on this machine, fuse on arm64 and stay plain (two roundings) on
# x86-64 — where it is byte-identical to the pre-FMA code already green vs R on
# Intel, so switching a kernel to ``_rfma`` is a no-op there. numpy has no fma
# ufunc, so the vectorized path loops ``math.fma`` via frompyfunc (correct;
# slower — the pure-Python oracle/fallback, where correctness > speed).
_R_FMA = platform.machine().lower() in ("arm64", "aarch64") and hasattr(math, "fma")
if _R_FMA:

    def _rfma(a, b, c):
        # C99 fma never raises: overflow -> +-Inf, invalid (Inf*0) -> NaN.
        # math.fma raises OverflowError/ValueError there instead; the plain
        # expression reproduces C's Inf/NaN results exactly (fused vs unfused
        # rounding only differs for finite results).
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
