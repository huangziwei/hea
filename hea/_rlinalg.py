"""Base R's dense linear algebra on doubles, issuing R's own BLAS/LAPACK calls.

``r_matprod`` is ``%*%`` (``matprod`` in R's array.c). With R's default ``matprod`` option, a product whose operands are all finite
is one BLAS call: ``dgemv('N')`` when ``y`` has one column, ``dgemv('T')`` on
``y`` when ``x`` has one row, ``dgemm('N', 'N')`` otherwise. Operands holding
NaN or Inf go to ``simple_matprod``, a plain triple loop. numpy's ``@`` reaches
the same library through different calls (row-major ``cblas`` layouts), and
those round differently, so a port of R code that multiplies matrices calls
this instead. scipy's BLAS is the library R links on this platform.

``r_eigen_sym`` is ``eigen(symmetric = TRUE)`` and ``r_backsolve`` is
``backsolve`` / ``forwardsolve``.

A leaf module, importable from :mod:`hea.formula` without :mod:`hea.R`.
"""

from __future__ import annotations

import numpy as np

from ._rfma import _rfma


def r_matprod(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``x %*% y`` for 2-D double arrays, R's ``matprod``."""
    from scipy.linalg.blas import dgemm, dgemv

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nrx, ncx = x.shape
    nry, ncy = y.shape
    if ncx != nry:
        raise ValueError("non-conformable arguments")
    if nrx == 0 or ncx == 0 or nry == 0 or ncy == 0:
        return np.zeros((nrx, ncy))
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        z = np.empty((nrx, ncy))
        for i in range(nrx):
            for k in range(ncy):
                s = 0.0
                for j in range(ncx):
                    s = _rfma(float(x[i, j]), float(y[j, k]), s)
                z[i, k] = s
        return z
    if ncy == 1:
        return np.asarray(dgemv(1.0, np.asfortranarray(x), y[:, 0])).reshape(nrx, 1)
    if nrx == 1:
        return np.asarray(dgemv(1.0, np.asfortranarray(y), x[0, :], trans=1)).reshape(
            1, ncy
        )
    return np.asarray(dgemm(1.0, np.asfortranarray(x), np.asfortranarray(y)))


def r_eigen_sym(A: np.ndarray, only_values: bool = False):
    """R ``eigen(A, symmetric=TRUE)``: LAPACK ``dsyevr`` on the lower
    triangle with ``abstol = 0`` (R's ``La_rs``; mgcv's ``mgcv_symeig``
    with ``use_dsyevd = 0`` makes the same call), eigenvalues in
    descending order and the vectors reordered to match. The routine
    decides each eigenvector's sign."""
    from scipy.linalg import eigh as _sla_eigh

    if only_values:
        w = _sla_eigh(A, lower=True, eigvals_only=True, driver="evr")
        return w[::-1].copy()
    w, V = _sla_eigh(A, lower=True, driver="evr")
    return w[::-1].copy(), np.ascontiguousarray(V[:, ::-1])


def r_backsolve(
    r: np.ndarray, x: np.ndarray, upper: bool = True, transpose: bool = False
) -> np.ndarray:
    """R ``backsolve(r, x, upper.tri = upper, transpose = transpose)`` with
    ``k = ncol(r)`` (``do_backsolve``): the first k rows of x copied, then one
    BLAS ``dtrsm('L', uplo, trans, 'N')``. ``forwardsolve(l, x)`` is
    ``upper=False``."""
    from scipy.linalg.blas import dtrsm

    r = np.asarray(r, dtype=float)
    x = np.asarray(x, dtype=float)
    vec = x.ndim == 1
    b = x.reshape(-1, 1) if vec else x
    k = r.shape[1]
    if np.any(np.diag(r)[:k] == 0.0):
        first = int(np.flatnonzero(np.diag(r)[:k] == 0.0)[0]) + 1
        raise np.linalg.LinAlgError(
            f"singular matrix in 'backsolve'. First zero in diagonal [{first}]"
        )
    ans = np.asfortranarray(b[:k, :].copy())
    if ans.size:
        ans = dtrsm(
            1.0,
            np.asfortranarray(r),
            ans,
            side=0,
            lower=0 if upper else 1,
            trans_a=1 if transpose else 0,
            diag=0,
        )
    ans = np.asarray(ans)
    return ans[:, 0] if vec else ans
