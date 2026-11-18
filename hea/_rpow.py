"""R's double operators that are not plain IEEE arithmetic: ``x ^ y``
(``R_POW`` / ``R_pow``) and ``x %% y`` (``myfmod``), from
src/main/arithmetic.c.

``R_pow`` is not libm ``pow``: ``y == 2`` is ``x*x`` for every ``x``, and for
``-11 <= x <= 11`` the exponents 3 and 4 are the sequential products
``x*x*x`` and ``x*x*x*x``. Those round differently from ``pow`` in the last
bit, and numpy's ``**`` makes neither split. ``x == 1`` or ``y == 0`` give 1
and ``x == 0`` is handled before ``pow`` is called; everything else finite is
``pow``. ``_rpow`` applies this elementwise for a scalar exponent.

``myfmod`` is not ``fmod`` or numpy's ``mod``: it reduces ``x1 - floor(q)*x2``
and then once more by ``floor(tmp/x2)``, and on arm64 each of those is one
fused multiply-subtract. ``_r_mod`` applies it elementwise.

A leaf module, so :mod:`hea.formula` and :mod:`hea.family` can both use it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from ._rfma import _rfma_vec


def _rpow_int(x, k):
    """R's ``x ^ k`` for ``k`` in 2, 3, 4: ``x*x`` for every ``x`` at
    ``k == 2``; for 3 and 4 the sequential product where
    ``-11 <= x <= 11`` and libm ``pow`` elsewhere."""
    x = np.asarray(x, dtype=float)
    if k == 2:
        return x * x
    seq = x * x * x if k == 3 else x * x * x * x
    fast = (x >= -11.0) & (x <= 11.0)
    if np.all(fast):
        return seq

    def _pow1(v):
        try:
            return math.pow(v, float(k))
        except OverflowError:
            return math.copysign(math.inf, v) if k % 2 else math.inf

    out = np.where(fast, seq, 0.0).reshape(-1)
    xf = x.reshape(-1)
    for i in np.flatnonzero(~fast.reshape(-1)):
        out[i] = _pow1(float(xf[i]))
    return out.reshape(x.shape)


def _r_pow_scalar(x, p):
    """Scalar ``R_pow(x, p)``: the ``x == 1`` / ``p == 0`` / ``x == 0``
    cases ahead of libm ``pow``, which overflows to a signed infinity and
    returns NaN for a negative base with a fractional exponent, as C does."""
    if p == 2.0:
        return x * x
    if x == 1.0 or p == 0.0:
        return 1.0
    if x == 0.0:
        if p > 0.0:
            return 0.0
        if p < 0.0:
            return math.inf
        return p  # NaN
    if -11.0 <= x <= 11.0 and p == 3.0:
        return x * x * x
    if -11.0 <= x <= 11.0 and p == 4.0:
        return x * x * x * x
    if math.isfinite(x) and math.isfinite(p):
        try:
            return math.pow(x, p)
        except OverflowError:
            return math.inf if (x > 0 or p % 2 == 0) else -math.inf
        except ValueError:
            return math.nan
    if math.isnan(x) or math.isnan(p):
        return x + p
    if not math.isfinite(x):
        if x > 0:
            return 0.0 if p < 0 else math.inf
        if math.isfinite(p) and p == math.floor(p):
            return 0.0 if p < 0 else (x if math.fmod(p, 2.0) != 0 else -x)
    if not math.isfinite(p) and x >= 0:
        if p > 0:
            return math.inf if x >= 1 else 0.0
        return math.inf if x < 1 else 0.0
    return math.nan


def _rpow(x, p):
    """R's elementwise ``x ^ p`` for a scalar ``p``. Finite bases go to
    libm ``pow`` in one vectorized call (numpy's ``power`` is ``pow`` per
    element except at ``p = 0.5`` and ``p = -1``, where it takes ``sqrt``
    and the reciprocal, so those exponents go element by element), with R's
    ``x == 1`` and ``x == 0`` cases applied first; any other element takes
    the scalar ``R_pow`` path."""
    p = float(p)
    if p in (2.0, 3.0, 4.0):
        return _rpow_int(x, int(p))
    x = np.asarray(x, dtype=float)
    if not math.isfinite(p) or p in (0.0, 0.5, -1.0):
        out = np.empty(x.shape)
        of = out.ravel()
        for i, v in enumerate(x.ravel()):
            of[i] = _r_pow_scalar(float(v), p)
        return out
    with np.errstate(all="ignore"):
        out = np.power(x, p)
    out = np.where(x == 1.0, 1.0, out)
    out = np.where(x == 0.0, 0.0 if p > 0.0 else math.inf, out)
    odd = ~np.isfinite(x)
    if odd.any():
        idx = np.flatnonzero(odd.ravel())
        of = out.reshape(-1)
        xf = x.reshape(-1)
        for i in idx:
            of[i] = _r_pow_scalar(float(xf[i]), p)
    return out


_C_EPS = float(np.finfo(float).eps)


def _r_mod(x1, x2):
    """R's ``x1 %% x2`` on doubles (``myfmod``), elementwise: NaN for
    ``x2 == 0``; for ``|x2| > 1/eps`` and finite ``|x1| <= |x2|`` the exact
    result by sign; otherwise ``tmp = x1 - floor(q)*x2`` with ``q = x1/x2``,
    then ``tmp - floor(tmp/x2)*x2``, warning as R does when ``q`` is too large
    for the result to be accurate."""
    x1, x2 = np.broadcast_arrays(
        np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    )
    shape = x1.shape
    x1 = x1.ravel()
    x2 = x2.ravel()
    out = np.empty(x1.shape)
    zero = x2 == 0.0
    big = (
        ~zero & (np.abs(x2) * _C_EPS > 1) & np.isfinite(x1) & (np.abs(x1) <= np.abs(x2))
    )
    if big.any():
        a, b = x1[big], x2[big]
        differ = ((a < 0) & (b > 0)) | ((b < 0) & (a > 0))
        out[big] = np.where(np.abs(a) == np.abs(b), 0.0, np.where(differ, a + b, a))
    gen = ~zero & ~big
    if gen.any():
        a, b = x1[gen], x2[gen]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            q = a / b
            if np.any(np.isfinite(q) & (np.abs(q) * _C_EPS > 1)):
                warnings.warn(
                    "probable complete loss of accuracy in modulus", stacklevel=2
                )
            tmp = _rfma_vec(-np.floor(q), b, a)
            out[gen] = _rfma_vec(-np.floor(tmp / b), b, tmp)
    out[zero] = np.nan
    return out.reshape(shape)
