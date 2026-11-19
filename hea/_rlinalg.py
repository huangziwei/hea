"""Base R's dense linear algebra on doubles, issuing R's own BLAS/LAPACK calls.

``r_matprod`` is ``%*%`` (``matprod`` in R's array.c). With R's default ``matprod`` option, a product whose operands are all finite
is one BLAS call: ``dgemv('N')`` when ``y`` has one column, ``dgemv('T')`` on
``y`` when ``x`` has one row, ``dgemm('N', 'N')`` otherwise. Operands holding
NaN or Inf go to ``simple_matprod``, a plain triple loop. numpy's ``@`` reaches
the same library through different calls (row-major ``cblas`` layouts), and
those round differently, so a port of R code that multiplies matrices calls
this instead. scipy's BLAS is the library R links on this platform.

``r_eigen_sym`` is ``eigen(symmetric = TRUE)``, ``r_backsolve`` is
``backsolve`` / ``forwardsolve``, ``r_norm`` is ``norm`` and ``r_solve`` is
``solve``. ``r_qr`` / ``r_qr_qy`` / ``r_qr_coef`` are ``qr`` and its methods
for R's default LINPACK QR (``dqrdc2``, ``dqrsl``), ``r_qr_lapack`` /
``r_qr_qy_lapack`` the same for ``qr(x, LAPACK = TRUE)`` (``dgeqp3``,
``dormqr``), and ``r_qr_R`` is ``qr.R`` for either. ``r_svd`` is ``svd``
(``La.svd``'s ``dgesdd``) and ``mgcv_svd_full`` mgcv's C SVD.

A leaf module, importable from :mod:`hea.formula` without :mod:`hea.R`.
"""

from __future__ import annotations

import math

import numpy as np

from ._dispatch import rs_fn
from ._rfma import _rfma, _rfma_dot

_r_dsyevr_rs = rs_fn("r_dsyevr")
_r_dgeqp3_rs = rs_fn("r_dgeqp3")
_r_dgeqp3_jpvt_rs = rs_fn("r_dgeqp3_jpvt")
_r_dormqr_rs = rs_fn("r_dormqr_l")
_r_dgesv_rs = rs_fn("r_dgesv")
_r_dpstrf_rs = rs_fn("r_dpstrf")
_r_dpotrf_rs = rs_fn("r_dpotrf")
_r_dpotri_rs = rs_fn("r_dpotri")
_r_dgesdd_rs = rs_fn("r_dgesdd")


def r_may_have_nan_or_inf(a: np.ndarray) -> bool:
    """R's ``mayHaveNaNOrInf`` (array.c:643) on ``a``'s column-major data:
    an odd first element not finite, or the sum of any later pair (``x[i] +
    x[i+1]`` from ``i = n & 1``, in steps of two) not finite, so two huge
    finite values whose sum overflows count too. A pair can overflow only
    once an entry reaches 2^1023."""
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        return True
    if a.size < 2 or float(np.abs(a).max()) < 2.0**1023:
        return False
    v = a.ravel(order="F")
    i0 = v.size & 1
    with np.errstate(over="ignore"):
        s = v[i0::2] + v[i0 + 1 :: 2]
    return not bool(np.isfinite(s).all())


def r_matprod_safe(a: np.ndarray) -> bool:
    """Whether every arrangement of ``a``'s entries - ``a`` itself, a subset
    of its rows, its transpose - passes :func:`r_may_have_nan_or_inf`: all
    finite and none at or above 2^1023, so no pair can overflow."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return True
    return bool(np.isfinite(a).all()) and float(np.abs(a).max()) < 2.0**1023


def r_matprod(x: np.ndarray, y: np.ndarray, *, x_finite: bool = False) -> np.ndarray:
    """``x %*% y`` for 2-D double arrays, R's ``matprod``: the BLAS unless
    ``mayHaveNaNOrInf`` flags either operand, then the plain loop.
    ``x_finite`` says the caller already knows that test passes for ``x``,
    sparing the scan of a large ``x`` multiplied repeatedly."""
    from scipy.linalg.blas import dgemm, dgemv

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nrx, ncx = x.shape
    nry, ncy = y.shape
    if ncx != nry:
        raise ValueError("non-conformable arguments")
    if nrx == 0 or ncx == 0 or nry == 0 or ncy == 0:
        return np.zeros((nrx, ncy))
    if (not x_finite and r_may_have_nan_or_inf(x)) or r_may_have_nan_or_inf(y):
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


# Blue's scaling constants for doubles (LAPACK la_constants.f90).
_TSML = 2.0**-511
_TBIG = 2.0**486
_SSML = 2.0**537
_SBIG = 2.0**-538


def _dlassq(x: np.ndarray, scale: float, sumsq: float) -> tuple[float, float]:
    """LAPACK 3.12 ``DLASSQ`` (dlassq.f90, Blue's algorithm): ``(scale,
    sumsq)`` updated so ``scale**2 * sumsq`` gains ``sum(x**2)``, with big,
    small and mid-range entries accumulated apart (small ones only until the
    first big one) and the incoming sum merged into the matching
    accumulator. As R's gfortran 14.2 build compiles it, every accumulation,
    each ``scale * (scale * sumsq)`` merge, ``(amed*sbig)*sbig`` and
    ``1 + (ymin/ymax)**2`` are fused multiply-adds."""
    if math.isnan(scale) or math.isnan(sumsq):
        return scale, sumsq
    if sumsq == 0.0:
        scale = 1.0
    if scale == 0.0:
        scale = 1.0
        sumsq = 0.0
    ax = np.abs(np.asarray(x, dtype=float).ravel())
    if ax.size == 0:
        return scale, sumsq
    big = ax > _TBIG
    small = ~big & (ax < _TSML)
    med = ~big & ~small
    first_big = np.flatnonzero(big)
    notbig = first_big.size == 0
    cut = ax.size if notbig else int(first_big[0])
    tb = ax[big] * _SBIG
    abig = _rfma_dot(tb, tb) if tb.size else 0.0
    ts = ax[:cut][small[:cut]] * _SSML
    asml = _rfma_dot(ts, ts) if ts.size else 0.0
    tm = ax[med]
    amed = _rfma_dot(tm, tm) if tm.size else 0.0
    # Put the existing sum of squares into one of the accumulators
    if sumsq > 0.0:
        a = scale * math.sqrt(sumsq)
        if a > _TBIG:
            if scale > 1.0:
                scale = scale * _SBIG
                abig = _rfma(scale, scale * sumsq, abig)
            else:
                abig = _rfma(scale, scale * (_SBIG * (_SBIG * sumsq)), abig)
        elif a < _TSML:
            if notbig:
                if scale < 1.0:
                    scale = scale * _SSML
                    asml = _rfma(scale, scale * sumsq, asml)
                else:
                    asml = _rfma(scale, scale * (_SSML * (_SSML * sumsq)), asml)
        else:
            amed = _rfma(scale, scale * sumsq, amed)
    # Combine abig and amed or amed and asml if more than one accumulator
    # was used.
    if abig > 0.0:
        if amed > 0.0 or math.isnan(amed):
            abig = _rfma(amed * _SBIG, _SBIG, abig)
        return 1.0 / _SBIG, abig
    if asml > 0.0:
        if amed > 0.0 or math.isnan(amed):
            amed = math.sqrt(amed)
            asml = math.sqrt(asml) / _SSML
            ymin, ymax = (amed, asml) if asml > amed else (asml, amed)
            q = ymin / ymax
            return 1.0, (ymax * ymax) * _rfma(q, q, 1.0)
        return 1.0 / _SSML, asml
    return 1.0, amed


def r_crossprod(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """R ``crossprod(x)`` / ``crossprod(x, y)`` (array.c): ``t(x) %*% x`` by
    BLAS ``dsyrk('U', 'T')`` with the lower triangle copied from the upper
    (``symcrossprod``), ``t(x) %*% y`` by ``dgemv('T')`` when one side has a
    single column and ``dgemm('T', 'N')`` otherwise. Operands holding NaN
    or Inf go to the plain loop ``sum += x[j,i] * y[j,k]``, fused on arm64."""
    from scipy.linalg.blas import dgemm, dgemv, dsyrk

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    yy = x if y is None else np.asarray(y, dtype=float)
    if yy.ndim == 1:
        yy = yy.reshape(-1, 1)
    nrx, ncx = x.shape
    nry, ncy = yy.shape
    if nrx != nry:
        raise ValueError("non-conformable arguments")
    if nrx == 0 or ncx == 0 or ncy == 0:
        return np.zeros((ncx, ncy))
    if not (np.isfinite(x).all() and np.isfinite(yy).all()):
        z = np.empty((ncx, ncy))
        for i in range(ncx):
            for k in range(ncy):
                acc = 0.0
                for j in range(nrx):
                    acc = _rfma(float(x[j, i]), float(yy[j, k]), acc)
                z[i, k] = acc
        return z
    xf = np.asfortranarray(x)
    if y is None:
        z = np.array(dsyrk(1.0, xf, trans=1, lower=0), dtype=float)
        lo = np.tril_indices(ncx, -1)
        z[lo] = z.T[lo]
        return z
    if ncy == 1:
        return np.asarray(dgemv(1.0, xf, yy[:, 0], trans=1)).reshape(ncx, 1)
    if ncx == 1:
        return np.asarray(dgemv(1.0, np.asfortranarray(yy), x[:, 0], trans=1)).reshape(
            1, ncy
        )
    return np.asarray(dgemm(1.0, xf, np.asfortranarray(yy), trans_a=1))


def r_tcrossprod(x: np.ndarray) -> np.ndarray:
    """R ``tcrossprod(x)`` (array.c ``symtcrossprod``): ``x %*% t(x)`` by
    BLAS ``dsyrk('U', 'N')`` with the lower triangle copied from the upper.
    An operand holding NaN or Inf goes to the plain loop ``sum += x[i,j] *
    x[k,j]``, fused on arm64."""
    from scipy.linalg.blas import dsyrk

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    nr, nc = x.shape
    if nr == 0 or nc == 0:
        return np.zeros((nr, nr))
    if not np.isfinite(x).all():
        z = np.empty((nr, nr))
        for i in range(nr):
            for k in range(nr):
                acc = 0.0
                for j in range(nc):
                    acc = _rfma(float(x[i, j]), float(x[k, j]), acc)
                z[i, k] = acc
        return z
    z = np.array(dsyrk(1.0, np.asfortranarray(x), trans=0, lower=0), dtype=float)
    lo = np.tril_indices(nr, -1)
    z[lo] = z.T[lo]
    return z


def r_norm(x: np.ndarray, type: str = "O") -> float:
    """R ``norm(x, type)``: LAPACK ``dlange``. For ``"O"`` (``"1"``), ``"I"``
    and ``"M"`` its sums run in its loop order (each column top to bottom
    for the one norm, each row across the columns in turn for the infinity
    norm), so a cumulative sum reproduces them, and a NaN sum is returned as
    the norm, as its ``DISNAN`` test does. ``"F"`` (``"E"``) is ``DLASSQ``
    over the columns in turn, ``scale * sqrt(sumsq)``."""
    a = np.abs(np.asarray(x, dtype=float))
    if a.ndim != 2:
        a = a.reshape(-1, 1)
    if a.size == 0:
        return 0.0
    t = type.upper()
    if t in ("F", "E"):
        scale, sumsq = 0.0, 1.0
        for j in range(a.shape[1]):
            scale, sumsq = _dlassq(a[:, j], scale, sumsq)
        return scale * math.sqrt(sumsq)
    if t in ("O", "1"):
        sums = np.cumsum(a, axis=0)[-1]
    elif t == "I":
        sums = np.cumsum(a, axis=1)[:, -1]
    elif t == "M":
        sums = a.ravel(order="F")
    else:
        raise ValueError(f"norm type {type!r} is not ported")
    # dlange's ``IF (VALUE.LT.TEMP .OR. DISNAN(TEMP)) VALUE = TEMP`` from 0:
    # NaN once any sum is NaN, else the largest
    if np.isnan(sums).any():
        return float("nan")
    return max(0.0, float(sums.max()))


def r_eigen_sym(A: np.ndarray, only_values: bool = False):
    """R ``eigen(A, symmetric=TRUE)``: LAPACK ``dsyevr`` on the lower
    triangle with ``abstol = 0`` (R's ``La_rs``; mgcv's ``mgcv_symeig``
    with ``use_dsyevd = 0`` makes the same call), eigenvalues in
    descending order and the vectors reordered to match. The routine
    decides each eigenvector's sign.

    The solve is the Rust port of R's bundled LAPACK when the extension is
    present, which reproduces R's ``dsyevr`` bit for bit. Without it, or when
    its MRRR stage fails (where LAPACK falls back to bisection and inverse
    iteration), scipy's ``dsyevr`` from whatever LAPACK scipy links.
    """
    A = np.asarray(A, dtype=float)
    if _r_dsyevr_rs is not None:
        w, zt, info = _r_dsyevr_rs(np.ascontiguousarray(A), bool(only_values))
        if info == 0:
            if only_values:
                return np.asarray(w)[::-1].copy()
            return np.asarray(w)[::-1].copy(), np.ascontiguousarray(
                np.asarray(zt).T[:, ::-1]
            )
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


def r_solve(a, b=None, tol: float = float(np.finfo(float).eps)) -> np.ndarray:
    """R ``solve(a, b)`` (``solve.default`` then ``La_solve``): LAPACK
    ``dgesv`` on copies of ``a`` and ``b`` (``b`` defaulting to the
    identity, which gives the inverse), then, when every factor is finite
    and ``tol > 0``, the reciprocal condition number of ``a`` from
    ``dgecon`` on the LU factors with ``anorm = norm(a, "1")``, an error
    below ``tol``. A vector ``b`` gives a vector. The solve is the Rust port
    of R's bundled LAPACK when present, else scipy's ``dgesv``; the
    condition estimate is scipy's ``dgecon`` on those factors."""
    from scipy.linalg.lapack import dgecon

    A = np.asarray(a, dtype=float)
    if A.ndim != 2:
        raise ValueError("'a' must be a numeric matrix")
    n = A.shape[0]
    if n == 0:
        raise ValueError("'a' is 0-diml")
    if A.shape[1] != n:
        raise ValueError(f"'a' ({n} x {A.shape[1]}) must be square")
    vec = False
    if b is None:
        B = np.eye(n)
    else:
        B = np.asarray(b, dtype=float)
        vec = B.ndim == 1
        if vec:
            B = B.reshape(-1, 1)
        if B.shape[0] != n:
            raise ValueError(
                f"'b' ({B.shape[0]} x {B.shape[1]}) must be compatible with 'a' ({n} x {n})"
            )
    if _r_dgesv_rs is not None:
        xt, lut, _ipiv, info = _r_dgesv_rs(
            np.ascontiguousarray(A.T), np.ascontiguousarray(B.T)
        )
        X = np.asarray(xt)[:, :n].T
        LU = np.asarray(lut)[:, :n].T
    else:
        from scipy.linalg.lapack import dgesv

        LU, _ipiv, X, info = dgesv(np.asfortranarray(A), np.asfortranarray(B))
    if info < 0:
        raise ValueError(f"argument {-info} of Lapack routine dgesv had invalid value")
    if info > 0:
        raise np.linalg.LinAlgError(
            f"Lapack routine dgesv: system is exactly singular: U[{info},{info}] = 0"
        )
    if tol > 0 and np.all(np.isfinite(LU)):
        rcond, _info = dgecon(np.asfortranarray(LU), r_norm(A, "O"), norm="1")
        if rcond < tol:
            raise np.linalg.LinAlgError(
                f"system is computationally singular: reciprocal condition number = {rcond:g}"
            )
    X = np.ascontiguousarray(X)
    return X[:, 0] if vec else X


def r_qr(x: np.ndarray, tol: float = 1e-7) -> dict:
    """R ``qr(x, tol)`` with the default LINPACK method: ``dqrdc2`` on a
    copy of ``x``. Returns R's list ``qr``, ``rank``, ``qraux``, ``pivot``
    (1-based)."""
    from .R.linalg import dqrdc2

    qr, qraux, jpvt, rank = dqrdc2(np.asarray(x, dtype=float), tol)
    return {
        "qr": qr,
        "rank": int(rank),
        "qraux": np.asarray(qraux),
        "pivot": np.asarray(jpvt),
    }


def r_qr_qy(q: dict, y, transpose: bool = False) -> np.ndarray:
    """R ``qr.qy(q, y)`` (``transpose=False``, ``dqrqy``) or ``qr.qty(q, y)``
    (``transpose=True``, ``dqrqty``) for a LINPACK QR: ``dqrsl`` on each
    column of ``y`` with ``k = q$rank`` reflectors, written into a copy of
    ``y`` laid out as the one ``.Fortran`` passes. The result has the shape
    of ``y``."""
    from .R.linalg import dqrsl, r_vector_like

    qr = np.asarray(q["qr"], dtype=float)
    n = qr.shape[0]
    Y = np.asarray(y, dtype=float)
    vec = Y.ndim == 1
    Y2 = Y.reshape(-1, 1) if vec else Y
    if Y2.shape[0] != n:
        raise ValueError("'qr' and 'y' must have the same number of rows")
    k = int(q["rank"])
    out = r_vector_like(Y2.shape)
    out[...] = Y2
    for j in range(Y2.shape[1]):
        if transpose:
            dqrsl(qr, n, k, q["qraux"], Y2[:, j], 1000, qty=out[:, j])
        else:
            dqrsl(qr, n, k, q["qraux"], Y2[:, j], 10000, qy=out[:, j])
    out = np.ascontiguousarray(out)
    return out[:, 0] if vec else out


def r_qr_fitted(q: dict, y) -> np.ndarray:
    """R ``qr.fitted(q, y)`` for a LINPACK QR: ``dqrxb``, i.e. ``dqrsl`` job 1
    on each column of ``y`` with ``k = q$rank`` - ``Q'y``, its entries past
    ``k`` zeroed, and ``Q`` applied back. The result has the shape of
    ``y``."""
    from .R.linalg import dqrsl, r_vector_like

    qr = np.asarray(q["qr"], dtype=float)
    n = qr.shape[0]
    Y = np.asarray(y, dtype=float)
    vec = Y.ndim == 1
    Y2 = Y.reshape(-1, 1) if vec else Y
    if Y2.shape[0] != n:
        raise ValueError("'qr' and 'y' must have the same number of rows")
    k = int(q["rank"])
    out = r_vector_like(Y2.shape)
    out[...] = Y2
    for j in range(Y2.shape[1]):
        dqrsl(qr, n, k, q["qraux"], Y2[:, j], 1, xb=out[:, j])
    out = np.ascontiguousarray(out)
    return out[:, 0] if vec else out


def r_qr_coef(q: dict, y) -> np.ndarray:
    """R ``qr.coef(q, y)`` for a LINPACK QR: ``dqrcf`` (``dqrsl`` job 100
    on each column of ``y`` with ``k = q$rank`` reflectors); for
    ``k < ncol(q$qr)`` the ``k`` coefficients go to the pivot positions
    ``q$pivot[1:k]`` and the rest are NaN (R's NA). A vector ``y`` gives a
    vector."""
    from .R.linalg import dqrsl, r_vector_like

    qr = np.asarray(q["qr"], dtype=float)
    n, p = qr.shape
    k = int(q["rank"])
    Y = np.asarray(y, dtype=float)
    vec = Y.ndim == 1
    Y2 = Y.reshape(-1, 1) if vec else Y
    ny = Y2.shape[1]
    if p == 0:
        return np.zeros(0) if vec else np.zeros((0, ny))
    if k == 0:
        out = np.full((p, ny), np.nan)
        return out[:, 0] if vec else out
    if Y2.shape[0] != n:
        raise ValueError("'qr' and 'y' must have the same number of rows")
    ybuf = r_vector_like(Y2.shape)
    ybuf[...] = Y2
    b = r_vector_like((k, ny))
    for j in range(ny):
        col = ybuf[:, j]
        *_, info = dqrsl(qr, n, k, q["qraux"], col, 100, qty=col, b=b[:, j])
        if info:
            raise np.linalg.LinAlgError("exact singularity in 'qr.coef'")
    b = np.ascontiguousarray(b)
    if k < p:
        coef = np.full((p, ny), np.nan)
        coef[np.asarray(q["pivot"])[:k] - 1] = b
    else:
        coef = b
    return coef[:, 0] if vec else coef


def r_qr_lapack(x: np.ndarray) -> dict:
    """R ``qr(x, LAPACK = TRUE)`` (``La_qr``): LAPACK ``dgeqp3`` on a copy of
    ``x`` with every column free. Returns R's list: ``qr`` (``R`` on and
    above the diagonal, the Householder vectors below), ``rank``
    (``min(dim(x))``, as R reports it for a LAPACK QR), ``qraux`` (the
    reflector scalars) and ``pivot`` (1-based). The Rust port of R's
    bundled LAPACK when present, else scipy's ``dgeqp3``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("'a' must be a numeric matrix")
    m, n = x.shape
    if _r_dgeqp3_rs is not None:
        qrt, tau, jpvt = _r_dgeqp3_rs(np.ascontiguousarray(x.T))
        qr = np.asarray(qrt)[:, :m].T
    else:
        from scipy.linalg.lapack import dgeqp3

        qr, jpvt, tau, _work, _info = dgeqp3(np.asfortranarray(x))
    return {
        "qr": qr,
        "rank": min(m, n),
        "qraux": np.asarray(tau),
        "pivot": np.asarray(jpvt, dtype=np.int64),
    }


def r_dgeqp3(a: np.ndarray, jpvt=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LAPACK ``dgeqp3`` on the ``m``-by-``n`` ``a`` as defined, with the
    workspace its query returns: a column whose ``jpvt`` entry is non-zero
    on entry is fixed - moved to the front in its order and factored
    unpivoted (``dgeqrf``, the other columns updated by ``dormqr('L',
    'T')``) - and the rest are pivoted by norm. Returns ``(qr, tau, jpvt)``:
    the factored ``m``-by-``n`` matrix, the ``min(m, n)`` reflector scalars
    and the 1-based column permutation. R's LAPACK through the extension;
    without it the same driver over SciPy's LAPACK."""
    a = np.asarray(a, dtype=float)
    m, n = a.shape
    jp = (
        np.zeros(n, dtype=np.int32)
        if jpvt is None
        else np.asarray(jpvt).astype(np.int32)
    )
    if _r_dgeqp3_jpvt_rs is not None:
        qrt, tau, jpo = _r_dgeqp3_jpvt_rs(
            np.ascontiguousarray(a.T), np.ascontiguousarray(jp)
        )
        qr = np.asarray(qrt)[:, :m].T.copy(order="F")
        return qr, np.asarray(tau), np.asarray(jpo, dtype=np.int64)
    from scipy.linalg.lapack import dgeqp3, dgeqrf, dormqr

    A = np.array(a, order="F", copy=True)
    jout = np.zeros(n, dtype=np.int64)
    # move the fixed columns up front, in their order
    nfxd = 0
    for j in range(n):
        if jp[j] != 0:
            if j != nfxd:
                A[:, [j, nfxd]] = A[:, [nfxd, j]]
                jout[j] = jout[nfxd]
                jout[nfxd] = j + 1
            else:
                jout[j] = j + 1
            nfxd += 1
        else:
            jout[j] = j + 1
    minmn = min(m, n)
    tau = np.zeros(minmn)
    if nfxd > 0:
        na = min(m, nfxd)
        qr, tq, _work, _info = dgeqrf(np.asfortranarray(A[:, :na]))
        A[:, :na] = qr
        tau[:na] = tq
        if na < n:
            lw = int(dormqr("L", "T", A[:, :na], tau[:na], A[:, na:], -1)[1][0])
            A[:, na:] = dormqr("L", "T", A[:, :na], tau[:na], A[:, na:], max(lw, 1))[0]
    if nfxd < minmn:
        sub, jsub, tsub, _work, _info = dgeqp3(np.asfortranarray(A[nfxd:, nfxd:]))
        perm = np.asarray(jsub, dtype=np.int64) - 1
        A[:nfxd, nfxd:] = A[:nfxd, nfxd:][:, perm]
        A[nfxd:, nfxd:] = sub
        jout[nfxd:] = jout[nfxd:][perm]
        tau[nfxd:minmn] = np.asarray(tsub)[: minmn - nfxd]
    return A, tau, jout


def r_dormqr_left(a: np.ndarray, tau, b: np.ndarray, trans: bool) -> np.ndarray:
    """LAPACK ``dormqr('L', 'T' or 'N', ...)`` with the workspace its query
    returns: ``Q' B`` (``trans``) or ``Q B`` for the ``m``-by-``c`` ``b`` and
    the ``Q`` of the ``len(tau)`` reflectors in the columns of the
    ``m``-row ``a``. R's LAPACK through the extension, else SciPy's."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    tau = np.asarray(tau, dtype=float)
    k = tau.size
    if _r_dormqr_rs is not None:
        ct = _r_dormqr_rs(
            np.ascontiguousarray(a[:, :k].T),
            np.ascontiguousarray(tau),
            np.ascontiguousarray(b.T),
            bool(trans),
        )
        return np.asarray(ct).T.copy(order="F")
    from scipy.linalg.lapack import dormqr

    t = "T" if trans else "N"
    A = np.asfortranarray(a[:, :k])
    B = np.asfortranarray(b)
    lw = int(dormqr("L", t, A, tau, B, -1)[1][0])
    return np.asarray(dormqr("L", t, A, tau, B, max(lw, 1))[0])


def r_qr_qy_lapack(q: dict, y, transpose: bool = False) -> np.ndarray:
    """R ``qr.qy(q, y)`` (``transpose=False``) or ``qr.qty(q, y)``
    (``transpose=True``) for a LAPACK QR (``qr_qy_real``): LAPACK
    ``dormqr('L', 'N' or 'T')`` with the reflectors of ``q`` and the
    workspace its query returns. ``y`` is taken ``as.matrix``; the result
    is a matrix with ``nrow(q$qr)`` rows."""
    qr = np.asarray(q["qr"], dtype=float)
    tau = np.ascontiguousarray(q["qraux"], dtype=float)
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    m = qr.shape[0]
    if Y.shape[0] != m:
        raise ValueError(f"right-hand side should have {m} not {Y.shape[0]} rows")
    if _r_dormqr_rs is not None:
        ct = _r_dormqr_rs(
            np.ascontiguousarray(qr.T), tau, np.ascontiguousarray(Y.T), bool(transpose)
        )
        return np.ascontiguousarray(np.asarray(ct)[:, :m].T)
    from scipy.linalg.lapack import dormqr

    lwork = max(1, Y.shape[1]) * 32 + 65 * 64
    cq, _work, _info = dormqr(
        "L",
        "T" if transpose else "N",
        np.asfortranarray(qr),
        tau,
        np.asfortranarray(Y),
        lwork,
    )
    return np.asarray(cq)


def r_qr_R(q: dict) -> np.ndarray:
    """R ``qr.R(q)``: the first ``min(dim(q$qr))`` rows of ``q$qr`` with the
    entries below the diagonal set to zero."""
    qr = np.asarray(q["qr"], dtype=float)
    return np.triu(qr[: min(qr.shape), :])


def _dgesdd(x: np.ndarray, jobz: str, calloc_work: bool):
    """LAPACK ``dgesdd`` on a copy of ``x`` with the workspace its query
    returns: the Rust port of R's bundled LAPACK when present, else scipy's.
    Returns ``(d, u, vt, x_out, info)`` (``u``/``vt`` as ``jobz`` forms
    them)."""
    m, n = x.shape
    if _r_dgesdd_rs is not None:
        d, ut, vtt, xt, info = _r_dgesdd_rs(
            np.ascontiguousarray(x.T), jobz, bool(calloc_work)
        )
        return (
            np.asarray(d),
            np.ascontiguousarray(np.asarray(ut).T),
            np.ascontiguousarray(np.asarray(vtt).T),
            np.ascontiguousarray(np.asarray(xt).T),
            int(info),
        )
    from scipy.linalg.lapack import dgesdd

    if jobz == "N":
        _u, d, _vt, info = dgesdd(x, compute_uv=0)
        return np.asarray(d), np.zeros((m, 0)), np.zeros((0, n)), x.copy(), int(info)
    if jobz == "O":
        u, d, vt, info = dgesdd(x, compute_uv=1, full_matrices=0)
        return np.asarray(d), np.asarray(u), np.asarray(vt), np.asarray(u), int(info)
    u, d, vt, info = dgesdd(x, compute_uv=1, full_matrices=int(jobz == "A"))
    return np.asarray(d), np.asarray(u), np.asarray(vt), x.copy(), int(info)


def r_svd(x: np.ndarray, nu: int | None = None, nv: int | None = None) -> dict:
    """R ``svd(x, nu, nv)``: ``La.svd``'s LAPACK ``dgesdd`` with ``jobu``
    ``'N'`` when no vectors are asked for, ``'S'`` when at most
    ``min(dim(x))`` of each are, else ``'A'``, then ``u`` truncated to ``nu``
    columns and ``v = t(vt)`` to ``nv``. Returns ``{"d", "u", "v"}``, the
    vectors only when asked for."""
    x = np.asarray(x, dtype=float)
    if x.ndim < 2:
        x = x.reshape(-1, 1)
    if not np.all(np.isfinite(x)):
        raise ValueError("infinite or missing values in 'x'")
    n, p = x.shape
    if n == 0 or p == 0:
        raise ValueError("a dimension is zero")
    k = min(n, p)
    nu = k if nu is None else int(nu)
    nv = k if nv is None else int(nv)
    if nu or nv:
        jobu = "S" if nu <= k and nv <= k else "A"
        nu0, nv0 = (k, k) if jobu == "S" else (n, p)
    else:
        jobu = "N"
        nu0 = nv0 = 0
    d, u, vt, _x, info = _dgesdd(x, jobu, False)
    if info != 0:
        raise np.linalg.LinAlgError(f"error code {info} from Lapack routine 'dgesdd'")
    res = {"d": d}
    if nu:
        res["u"] = u[:, : min(n, nu)] if nu < nu0 else u
    if nv:
        vt = vt[: min(p, nv), :] if nv < nv0 else vt
        res["v"] = np.ascontiguousarray(vt.T)
    return res


def mgcv_svd_full(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mgcv's C ``mgcv_svd_full`` (mat.c): LAPACK ``dgesdd`` with ``JOBZ =
    'O'`` on the ``r``-by-``c`` ``x`` (``r >= c``), its workspace from
    ``calloc``. Returns ``(u, d, vt)``: ``U`` (``r``-by-``c``, written over
    ``x``), the singular values and ``V'`` (``c``-by-``c``)."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] < x.shape[1]:
        raise ValueError("mgcv_svd_full needs nrow(x) >= ncol(x)")
    d, _u, vt, xo, info = _dgesdd(x, "O", True)
    if info != 0:
        raise np.linalg.LinAlgError(f"error code {info} from Lapack routine 'dgesdd'")
    return xo, d, vt


def r_chol2inv(x: np.ndarray, size: int | None = None) -> np.ndarray:
    """R ``chol2inv(x, size)`` (``La_chol2inv``): the inverse of ``R'R`` from
    the upper triangle of the leading ``size x size`` block of ``x``, by
    R's bundled LAPACK ``dpotri('U')`` (the Rust port) and mirrored to the
    lower triangle; scipy's ``dpotri`` when the extension is absent."""
    x = np.asarray(x, dtype=float)
    if x.ndim < 2:
        x = x.reshape(1, 1)
    n = x.shape[1] if size is None else int(size)
    a = np.triu(x[:n, :n])
    if _r_dpotri_rs is not None:
        inv, info = _r_dpotri_rs(np.ascontiguousarray(a.T))
        inv = np.asarray(inv)
    else:
        from scipy.linalg.lapack import dpotri

        inv, info = dpotri(np.asfortranarray(a), lower=0)
        inv = np.triu(inv) + np.triu(inv, 1).T
    if info > 0:
        raise np.linalg.LinAlgError(
            f"element ({info}, {info}) is zero, so the inverse cannot be computed"
        )
    return np.ascontiguousarray(inv)


def r_chol(
    x: np.ndarray, pivot: bool = False, tol: float = -1.0, *, r_layout: bool = True
):
    """R ``chol(x, pivot, tol)`` (``La_chol``): on a copy of ``x`` with its
    lower triangle zeroed, LAPACK ``dpotrf('U')`` (an error unless positive
    definite) or, for ``pivot``, ``dpstrf('U')`` with ``tol``, after which
    the upper triangle past the rank is zeroed. Returns the upper factor;
    with ``pivot`` also the 1-based pivots and the rank (R warns, and its
    callers here silence it, when the rank is short). ``r_layout=False``
    places the copy as mgcv's C callers' ``calloc`` buffers lie (the
    ``mgcv_chol`` call), leaving the zeroing to the caller."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("'a' must be a square matrix")
    n = a.shape[0]
    if n == 0:
        raise ValueError("'a' must have dims > 0")
    u = np.triu(a)
    if not pivot:
        if _r_dpotrf_rs is not None:
            ut, info = _r_dpotrf_rs(np.ascontiguousarray(u.T))
            R = np.array(np.asarray(ut).T)
        else:
            from scipy.linalg.lapack import dpotrf

            R, info = dpotrf(u, lower=0, clean=0)
        if info > 0:
            raise ValueError(f"the leading minor of order {info} is not positive")
        return R
    if _r_dpstrf_rs is not None:
        ut, piv, rank, _info = _r_dpstrf_rs(
            np.ascontiguousarray(u.T), float(tol), bool(r_layout)
        )
        R = np.array(np.asarray(ut).T)
        piv = np.asarray(piv, dtype=np.int64)
    else:
        from scipy.linalg.lapack import dpstrf

        R, piv, rank, _info = dpstrf(u, tol=float(tol), lower=0)
        piv = np.asarray(piv, dtype=np.int64)
    rank = int(rank)
    if r_layout:
        for j in range(rank, n):
            R[rank : j + 1, j] = 0.0
    return R, piv, rank
