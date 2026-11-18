"""libm functions that R's compiled C calls directly and Python's ``math``
module does not provide as libm: ``lgamma`` (``math.lgamma`` is CPython's own
approximation, which differs from libm in the last bits).

``lgamma_r`` is the reentrant form; it returns the same value as ``lgamma``
without writing the global ``signgam``.
"""

from __future__ import annotations

import ctypes
import ctypes.util

_libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
_lgamma_r = _libm.lgamma_r
_lgamma_r.restype = ctypes.c_double
_lgamma_r.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]


def libm_lgamma(x: float) -> float:
    """libm ``lgamma(x)``."""
    sign = ctypes.c_int(0)
    return _lgamma_r(float(x), ctypes.byref(sign))
