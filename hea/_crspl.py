"""mgcv's cubic regression spline basis (``bs = "cr"``): the C ``getFS``
and ``crspl`` (mgcv.c) as compiled for arm64, with R's bundled LAPACK
``dptsv`` (``dpttrf`` then ``dptts2``) for ``B^{-1} D``.

clang fuses the left product of every ``a*b + c*d`` in ``getFS`` and
``crspl`` (``fma(a, b, c*d)``; the three-term rows nest a second ``fma``
for the last product); gfortran 14.2 fuses ``D(I+1) - E(I)*EI`` in
``dpttrf`` and both updates of ``dptts2``. Matrices are indexed
``[row, col]`` as mgcv's column-major arrays.
"""

from __future__ import annotations

import numpy as np

from ._rfma import _rfma, _rfma_vec


def _dptsv(d: np.ndarray, e: np.ndarray, b: np.ndarray) -> np.ndarray:
    """LAPACK ``DPTSV``: ``A X = B`` for the symmetric positive definite
    tridiagonal ``A`` with diagonal ``d`` and off-diagonal ``e``. ``DPTTRF``
    factors ``A = L D L'``; when it meets a non-positive pivot ``DPTSV``
    returns ``B`` unsolved, as here. ``DPTTS2`` then solves every column
    (``N = 1`` scales by ``1/D(1)``, ``DSCAL``)."""
    d = np.array(d, dtype=float)
    e = np.array(e, dtype=float)
    b = np.array(b, dtype=float)
    n = d.size
    for i in range(n - 1):
        if d[i] <= 0.0:
            return b
        ei = e[i]
        e[i] = ei / d[i]
        d[i + 1] = _rfma(-e[i], ei, d[i + 1])
    if n == 0 or d[n - 1] <= 0.0:
        return b
    if n == 1:
        b[0] = b[0] * (1.0 / d[0])
        return b
    for i in range(1, n):
        b[i] = _rfma_vec(-b[i - 1], e[i - 1], b[i])
    b[n - 1] = b[n - 1] / d[n - 1]
    for i in range(n - 2, -1, -1):
        b[i] = _rfma_vec(-b[i + 1], e[i], b[i] / d[i])
    return b


def get_fs(xk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``getFS``: for ascending knots ``xk`` (``n >= 3``), the wiggliness
    penalty ``S`` and ``F`` (``F[k, j]`` is the second derivative at knot
    ``j`` per unit function value at knot ``k``; columns 0 and ``n-1`` are
    zero), both ``n x n``."""
    x = np.asarray(xk, dtype=float)
    n = x.size
    n2 = n - 2
    h = x[1:] - x[:-1]
    D = np.zeros((n2, n))
    for i in range(n2):
        D[i, i] = 1 / h[i]
        D[i, i + 2] = 1 / h[i + 1]
        D[i, i + 1] = -D[i, i] - D[i, i + 2]
    ldB = (h[:n2] + h[1 : n2 + 1]) / 3
    sdB = h[1:n2] / 6
    Dp = _dptsv(ldB, sdB, D)
    F = np.zeros((n, n))
    F[:, 1 : n - 1] = Dp.T
    S = np.empty((n, n))
    a = 1 / h[0]
    S[0] = Dp[0] * a
    if n > 3:
        a = -1 / h[0] - 1 / h[1]
        b = 1 / h[1]
        S[1] = _rfma_vec(Dp[0], a, b * Dp[1])
        for j in range(2, n2):
            a = 1 / h[j - 1]
            c = 1 / h[j]
            b = -a - c
            S[j] = _rfma_vec(Dp[j], c, _rfma_vec(Dp[j - 2], a, b * Dp[j - 1]))
        a = 1 / h[n2 - 1]
        b = -1 / h[n2 - 1] - 1 / h[n2]
        S[n2] = _rfma_vec(Dp[n2 - 2], a, b * Dp[n2 - 1])
    else:
        a = -1 / h[0] - 1 / h[1]
        S[1] = Dp[0] * a
    a = 1 / h[n2]
    S[n - 1] = Dp[n2 - 1] * a
    return S, F


def crspl(
    x: np.ndarray, xk: np.ndarray, F: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """``crspl``: the ``len(x) x nk`` model matrix of the cubic regression
    spline with knots ``xk``. Without ``F`` it is formed by ``get_fs`` and
    ``S`` returned with it (``Fsupplied = 0``); with ``F`` (prediction)
    ``S`` is ``None``. ``x`` lies in knot interval ``j`` when
    ``xk[j] < x <= xk[j+1]`` (``j = 0`` at ``xk[0]``); outside the knots the
    basis is extended linearly, and above the last knot mgcv's loop reads
    ``F[0, nk-1]`` (zero) for every ``F[k, nk-1]``."""
    xk = np.asarray(xk, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    nk = xk.size
    S = None
    if F is None:
        S, F = get_fs(xk)
    kmin, kmax = xk[0], xk[nk - 1]
    X = np.zeros((x.size, nk))
    below = x < kmin
    above = x > kmax
    mid = ~(below | above)

    xi = x[mid]
    j = np.clip(np.searchsorted(xk, xi, side="left") - 1, 0, nk - 2)
    xj = xk[j]
    xj1 = xk[j + 1]
    h = xj1 - xj
    ajm = xj1 - xi
    ajp = xi - xj
    cjm = (ajm * (ajm * ajm / h - h)) / 6
    cjp = (ajp * (ajp * ajp / h - h)) / 6
    Xm = _rfma_vec(cjm[:, None], F[:, j].T, cjp[:, None] * F[:, j + 1].T)
    rows = np.arange(xi.size)
    Xm[rows, j] = Xm[rows, j] + ajm / h
    Xm[rows, j + 1] = Xm[rows, j + 1] + ajp / h
    X[mid] = Xm

    if below.any():
        xi = x[below]
        h = xk[1] - kmin
        xik = xi - kmin
        t = -(xik * h)
        cjm = t / 3
        cjp = t / 6
        Xb = _rfma_vec(cjm[:, None], F[:, 0][None, :], cjp[:, None] * F[:, 1][None, :])
        Xb[:, 0] = Xb[:, 0] + (1 - xik / h)
        Xb[:, 1] = Xb[:, 1] + xik / h
        X[below] = Xb

    if above.any():
        xi = x[above]
        h = kmax - xk[nk - 2]
        xik = xi - kmax
        t = xik * h
        cjm = t / 6
        cjp = t / 3
        Xa = _rfma_vec(cjm[:, None], F[:, nk - 2][None, :], cjp[:, None] * F[0, nk - 1])
        Xa[:, nk - 2] = Xa[:, nk - 2] - xik / h
        Xa[:, nk - 1] = Xa[:, nk - 1] + (1 + xik / h)
        X[above] = Xa
    return X, S, F
