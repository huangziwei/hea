"""R's in-order ``LDOUBLE`` accumulation, the primitive under base R's
``mean``, ``colSums``, ``colMeans``, ``rowSums`` and ``rowMeans``.

R sums a double vector with a plain loop, ``LDOUBLE s = 0.0; for (k) s += x[k];``
(``real_mean`` in summary.c, ``do_colsum`` in array.c). numpy's ``sum`` and
``mean`` reduce pairwise, which rounds differently in the last bits; those
bits decide eigenvector signs downstream (the thin-plate basis is built from
covariates centred on ``mean(x)``). ``np.cumsum`` is a left-to-right loop, and
``np.longdouble`` is the C ``long double`` R's ``LDOUBLE`` names on the same
platform (80-bit on x86-64 Linux, double on arm64 macOS).

A leaf module: :mod:`hea.formula` and :mod:`hea.R` both import it, and
:mod:`hea.R` imports :mod:`hea.formula`.
"""

from __future__ import annotations

import math

import numpy as np

_BLOCK = 1 << 20


def _ld_sum_rows(xl: np.ndarray) -> np.ndarray:
    """Column sums of a 2-D ``long double`` array, each column accumulated
    over rows in order from ``0.0``. Row blocks are chained by seeding each
    block's ``cumsum`` with the running sum, which keeps the order intact
    while bounding the scratch memory."""
    n, p = xl.shape
    s = np.zeros(p, dtype=np.longdouble)
    if n == 0 or p == 0:
        return s
    step = max(1, _BLOCK // p)
    for a in range(0, n, step):
        blk = np.concatenate([s[None, :], xl[a : a + step]], axis=0)
        s = np.cumsum(blk, axis=0)[-1]
    return s


def _ld_sum(xl: np.ndarray) -> np.longdouble:
    """``LDOUBLE s = 0.0; for (k) s += x[k];`` over a 1-D ``long double`` array."""
    return _ld_sum_rows(np.asarray(xl, dtype=np.longdouble).reshape(-1, 1))[0]


def r_sum(x) -> float:
    """R's ``sum(x)`` for a double vector or matrix: the elements in R's
    (column-major) order accumulated left to right in ``LDOUBLE``."""
    x = np.asarray(x, dtype=float)
    return float(_ld_sum(x.ravel(order="F")))


def r_mean(x) -> float:
    """``real_mean`` (R summary.c), the body of ``mean.default`` for a double
    vector: the in-order sum divided by n, then the in-order sum of the
    residuals from it, divided by n, added back. If the first sum overflows,
    each term is divided by n (in double) before summing, and the residual
    correction is accumulated pre-divided too. n = 0 gives NaN, as in R."""
    xd = np.asarray(x, dtype=float).ravel()
    n = xd.size
    xl = xd.astype(np.longdouble)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        s = _ld_sum(xl)
        finite_s = math.isfinite(float(s))
        if finite_s:
            s = s / n
        else:
            s = _ld_sum((xd / n).astype(np.longdouble))
        if finite_s and math.isfinite(float(s)):
            t = _ld_sum(xl - s)
            s = s + t / n
        elif math.isfinite(float(s)):
            t = _ld_sum((xl - s) / n)
            s = s + t
    return float(s)


def r_colsums(x: np.ndarray, na_rm: bool = False, mean: bool = False) -> np.ndarray:
    """``do_colsum`` (R array.c) for ``colSums`` / ``colMeans`` on a double
    matrix: each column summed in order in ``LDOUBLE``; with ``na_rm`` NaNs
    are skipped and not counted; ``colMeans`` divides by that count (NaN when
    it is zero)."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    xl = a.astype(np.longdouble)
    if na_rm:
        miss = np.isnan(a)
        xl[
            miss
        ] = (
            -0.0
        )  # the additive identity: skipping a term and adding -0.0 agree bit-for-bit
        cnt = (~miss).sum(axis=0)
    else:
        cnt = np.full(a.shape[1], a.shape[0])
    s = _ld_sum_rows(xl)
    if mean:
        with np.errstate(invalid="ignore", divide="ignore"):
            s = s / cnt
    return s.astype(float)


def r_rowsums(x: np.ndarray, na_rm: bool = False, mean: bool = False) -> np.ndarray:
    """``do_colsum`` (R array.c) for ``rowSums`` / ``rowMeans`` on a double
    matrix: an ``LDOUBLE`` accumulator per row, adding the columns in order;
    ``rowMeans`` divides by p, or by the per-row count of non-NaN entries
    when ``na_rm``."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    xl = a.astype(np.longdouble)
    if na_rm:
        miss = np.isnan(a)
        xl[miss] = -0.0
        cnt = (~miss).sum(axis=1)
    else:
        cnt = np.full(a.shape[0], a.shape[1])
    s = _ld_sum_rows(np.ascontiguousarray(xl.T))
    if mean:
        with np.errstate(invalid="ignore", divide="ignore"):
            s = s / cnt
    return s.astype(float)


def r_var(x) -> float:
    """R's ``var(x)`` for a double vector without NA: stats' C ``cov`` on its
    ``"everything"`` path (``cov_na_1``). The mean is ``MEAN_``'s two passes
    in ``LDOUBLE`` (the sum over n, then the in-order sum of the residuals
    from it over n added back); the variance accumulates
    ``(LDOUBLE)(x[k] - m) * (x[k] - m)`` over ``n - 1``. Where ``LDOUBLE``
    is ``double`` and the build fuses (arm64 macOS), clang forms eight
    products at a time rounded and adds them in order, fusing only the last
    ``n % 8`` terms (every term when n < 8); elsewhere the loop is a plain
    ``long double`` accumulation. n < 2 gives NaN, as R's NA."""
    from ._rfma import _R_FMA, _rfma

    xd = np.asarray(x, dtype=float).ravel()
    n = xd.size
    if n < 2:
        return float("nan")
    xl = xd.astype(np.longdouble)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        tmp = _ld_sum(xl) / n
        if math.isfinite(float(tmp)):
            tmp = tmp + _ld_sum(xl - tmp) / n
        m = float(tmp)
        d = xd - m
        if _R_FMA and np.finfo(np.longdouble).bits == 64:
            nv = n - n % 8 if n >= 8 else 0
            s = float(_ld_sum(d[:nv] * d[:nv])) if nv else 0.0
            for v in d[nv:].tolist():
                s = _rfma(v, v, s)
            return s / (n - 1)
        dl = d.astype(np.longdouble)
        return float(_ld_sum(dl * dl) / (n - 1))


def r_all_equal(target, current, tolerance: float | None = None) -> bool:
    """``isTRUE(all.equal(target, current, tolerance))`` for two plain double
    vectors (``all.equal.numeric``, ``scale = NULL``, ``countEQ = FALSE``):
    the entries equal in both (or NA in both) are set aside, and the rest
    compared by their mean absolute difference, relative to the mean
    absolute target unless that is not above ``tolerance``. The default
    tolerance is ``sqrt(.Machine$double.eps)``."""
    if tolerance is None:
        tolerance = math.sqrt(float(np.finfo(float).eps))
    t = np.asarray(target, dtype=float).ravel()
    c = np.asarray(current, dtype=float).ravel()
    if t.size != c.size:
        return False
    out = np.isnan(t)
    if np.any(out != np.isnan(c)):
        return False
    out = out | (t == c)
    if np.all(out):
        return True
    keep = ~out
    t = t[keep]
    c = c[keep]
    N = t.size
    scale = r_sum(np.abs(t) / N)
    if not (math.isfinite(scale) and scale > tolerance):
        scale = 1.0
    xy = r_sum(np.abs(t - c) / (N * scale))
    return not (math.isnan(xy) or xy > tolerance)
