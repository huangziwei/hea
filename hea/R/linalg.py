"""R's least-squares QR kernel — a mechanical port of the LINPACK routines R
calls from ``lm.fit`` / ``lm.wfit`` via ``Cdqrls`` (``stats/src/lm.c``):

  * ``dqrdc2`` — R's *modification* of LINPACK ``dqrdc`` (``dqrdc2.f``):
    Householder QR with a limited column-pivoting strategy that defers
    near-dependent columns to the right (so sequential 1-df effects are well
    defined) and reports the rank.
  * ``dqrsl``  — LINPACK ``dqrsl`` (``dqrsl.f``): applies the factored Q
    to compute ``Qᵀy`` (effects), the coefficients ``b`` and residuals ``rsd``.
  * ``dqrls``  — the ``dqrdc2`` + ``dqrsl`` wrapper (``dqrls.f``).
  * ``Cdqrls`` — the ``lm.c`` entry returning R's ``$qr/$coefficients/$residuals/
    $effects/$rank/$pivot/$qraux/$tol/$pivoted``.
  * ``kappa``  — base R's ``kappa.default`` at its defaults: ``dqrdc2`` then
    LINPACK ``dtrco`` on the triangular factor.

This is the **pure-Python spec + oracle** for the Rust port (``hea/_rs`` →
``dqrls``); it mirrors the Fortran line-for-line (pivot order, rank rule, the
1e-6 norm-recompute branch) so Rust ≡ Python ≡ live R is checkable. Ported
against the real sources, not reverse-engineered. ``pivot`` is returned 1-based
(R parity).
"""

from __future__ import annotations

import math

import numpy as np

from .._dispatch import rs_fn
from .._rfma import _rfma

__all__ = ["Cdqrls", "dqrdc2", "dqrls", "dqrls_rank", "dqrsl", "kappa"]

_rs_dqrls = rs_fn("dqrls")
_rs_dqrls_rank = rs_fn("dqrls_rank")


def dqrls_rank(x: np.ndarray, tol: float = 1e-7):
    """``(rank, pivot)`` from R's ``dqrdc2`` only — for alias/rank detection,
    where the coefficients/effects/QR aren't needed. Rust active path (lean: no
    big-array marshalling), pure-Python ``dqrls`` as the fallback/oracle.
    ``pivot`` is 1-based."""
    x = np.asarray(x, dtype=float)
    if _rs_dqrls_rank is not None:
        rank, pivot = _rs_dqrls_rank(np.asfortranarray(x, dtype=float), tol)
        return int(rank), np.asarray(pivot)
    _, _, _, _, k, jpvt, _ = dqrls(x.copy(), np.zeros(x.shape[0]), tol)
    return int(k), np.asarray(jpvt)


def dqrdc2(x: np.ndarray, tol: float = 1e-7):
    """Mechanical port of R's ``dqrdc2.f`` (LINPACK ``dqrdc`` with R's
    limited column pivoting). ``x`` is ``(n, p)`` (a copy is overwritten).
    Returns ``(qr, qraux, jpvt, rank)``: ``qr`` holds R in its upper triangle
    and the Householder vectors below; ``qraux`` the reflector scalars;
    ``jpvt`` the 1-based pivot (column j of the output is column ``jpvt[j]``
    of the input); ``rank`` the number of independent columns.

    The column norms, dot products and updates are the BLAS calls the
    Fortran makes (``dnrm2``, ``ddot``, ``daxpy``) through scipy's BLAS, the
    library R links on this platform; ``dscal`` by ``1/nrmxl`` is the
    elementwise product it computes; the norm downdate ``1 - (|x_lj|/
    qraux_j)**2`` is a single fused ``fmsub`` in R's arm64 build. Columns are
    kept contiguous, as in R's column-major storage."""
    from scipy.linalg.blas import daxpy, ddot, dnrm2

    qr = np.array(x, dtype=float, order="F", copy=True)
    n, p = qr.shape
    qraux = np.zeros(p)
    work = np.zeros((p, 2))
    jpvt = np.arange(1, p + 1)  # 1-based column indices (R 'pivot')

    if n > 0:
        for j in range(p):
            qraux[j] = dnrm2(qr[:, j])
            work[j, 0] = qraux[j]
            work[j, 1] = qraux[j] if qraux[j] != 0.0 else 1.0

    lup = min(n, p)
    k = p + 1  # Fortran 'k' (1-based rank boundary)
    for l in range(1, lup + 1):
        l0 = l - 1
        while not (l >= k or qraux[l0] >= work[l0, 1] * tol):
            tcol = qr[:, l0].copy()
            qr[:, l0 : p - 1] = qr[:, l0 + 1 : p]
            qr[:, p - 1] = tcol
            isv, tsv = jpvt[l0], qraux[l0]
            tt_sv, ttt_sv = work[l0, 0], work[l0, 1]
            jpvt[l0 : p - 1] = jpvt[l0 + 1 : p]
            qraux[l0 : p - 1] = qraux[l0 + 1 : p]
            work[l0 : p - 1, 0] = work[l0 + 1 : p, 0]
            work[l0 : p - 1, 1] = work[l0 + 1 : p, 1]
            jpvt[p - 1], qraux[p - 1] = isv, tsv
            work[p - 1, 0], work[p - 1, 1] = tt_sv, ttt_sv
            k -= 1
        if l != n:
            nrmxl = dnrm2(qr[l0:n, l0])
            if nrmxl != 0.0:
                if qr[l0, l0] != 0.0:
                    nrmxl = math.copysign(nrmxl, qr[l0, l0])
                qr[l0:n, l0] = qr[l0:n, l0] * (1.0 / nrmxl)
                qr[l0, l0] = 1.0 + qr[l0, l0]
                xl = qr[l0:n, l0]
                for j in range(l, p):  # Fortran j = l+1..p
                    t = -ddot(xl, qr[l0:n, j]) / qr[l0, l0]
                    qr[l0:n, j] = daxpy(xl, qr[l0:n, j].copy(), a=t)
                    if qraux[j] != 0.0:
                        q = abs(qr[l0, j]) / qraux[j]
                        tt = _rfma(-q, q, 1.0)  # 1 - q**2, one fmsub in R's arm64 build
                        tt = max(tt, 0.0)
                        if abs(tt) >= 1e-6:
                            qraux[j] = qraux[j] * math.sqrt(tt)
                        else:
                            qraux[j] = dnrm2(qr[l0 + 1 : n, j])
                            work[j, 0] = qraux[j]
                qraux[l0] = qr[l0, l0]
                qr[l0, l0] = -nrmxl
    k = min(k - 1, n)
    return np.ascontiguousarray(qr), qraux, jpvt, k


def kappa(z: np.ndarray) -> float:
    """R ``kappa(z)`` with ``kappa.default``'s defaults (``exact = FALSE``,
    ``method = "qr"``, ``norm = NULL``): ``kappa.qr(qr(z))``. That is R's
    LINPACK ``qr`` (``dqrdc2``, ``tol = 1e-7``) of ``z``, or of ``t(z)`` when
    ``z`` is wide, then ``.kappa_tri`` on the leading square block of the
    triangular factor with its ``LINPACK = TRUE`` default:
    ``1/dtrco(R, job = 1)``, the reciprocal of LINPACK's 1-norm
    reciprocal-condition estimate. An exactly singular factor gives
    ``rcond = 0`` and so ``Inf``, as in R."""
    from ._linpack import _dtrco

    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or 0 in z.shape:
        return 0.0
    if z.shape[0] < z.shape[1]:
        z = z.T
    qr, _qraux, _jpvt, _rank = dqrdc2(z, tol=1e-7)
    p = qr.shape[1]
    R = np.triu(qr[:p, :p])
    rcond = _dtrco(R, p, 1)
    return math.inf if rcond == 0.0 else 1.0 / rcond


def _job_flags(job: int):
    return (
        job // 10000 != 0,  # cqy
        job % 10000 != 0,  # cqty
        (job % 1000) // 100 != 0,  # cb
        (job % 100) // 10 != 0,  # cr
        job % 10 != 0,  # cxb
    )


def dqrsl(qr: np.ndarray, n: int, k: int, qraux: np.ndarray, y: np.ndarray, job: int):
    """Mechanical port of ``dqrsl.f`` — apply the ``dqrdc2`` factorization.
    Returns ``(qy, qty, b, rsd, xb, info)``; only the parts selected by ``job``
    (decimal ``abcde``: a=qy, b/c/d/e⇒qty, c=b, d=rsd, e=xb) are meaningful.
    Each reflector is applied with the Fortran's BLAS ``ddot`` and ``daxpy``
    through scipy's BLAS, on a column-major copy of ``qr``."""
    from scipy.linalg.blas import daxpy, ddot

    qr = np.array(qr, dtype=float, order="F", copy=True)
    cqy, cqty, cb, cr, cxb = _job_flags(job)
    info = 0
    y = np.asarray(y, dtype=float).reshape(-1)
    qy = np.zeros(n)
    qty = np.zeros(n)
    b = np.zeros(k)
    rsd = np.zeros(n)
    xb = np.zeros(n)
    ju = min(k, n - 1)

    if ju == 0:  # special action when n == 1
        if cqy:
            qy[0] = y[0]
        if cqty:
            qty[0] = y[0]
        if cxb:
            xb[0] = y[0]
        if cb:
            if qr[0, 0] == 0.0:
                info = 1
            else:
                b[0] = y[0] / qr[0, 0]
        if cr:
            rsd[0] = 0.0
        return qy, qty, b, rsd, xb, info

    if cqy:
        qy[:] = y
    if cqty:
        qty[:] = y
    if cqy:  # compute Q·y (descending j)
        for jj in range(1, ju + 1):
            j0 = (ju - jj + 1) - 1
            if qraux[j0] != 0.0:
                temp = qr[j0, j0]
                qr[j0, j0] = qraux[j0]
                t = -ddot(qr[j0:n, j0], qy[j0:n]) / qr[j0, j0]
                qy[j0:n] = daxpy(qr[j0:n, j0], qy[j0:n].copy(), a=t)
                qr[j0, j0] = temp
    if cqty:  # compute Qᵀ·y (ascending j)
        for j in range(1, ju + 1):
            j0 = j - 1
            if qraux[j0] != 0.0:
                temp = qr[j0, j0]
                qr[j0, j0] = qraux[j0]
                t = -ddot(qr[j0:n, j0], qty[j0:n]) / qr[j0, j0]
                qty[j0:n] = daxpy(qr[j0:n, j0], qty[j0:n].copy(), a=t)
                qr[j0, j0] = temp

    if cb:
        b[:k] = qty[:k]
    if cxb:
        xb[:k] = qty[:k]
    if cr and k < n:
        rsd[k:n] = qty[k:n]
    if cxb and k < n:
        xb[k:n] = 0.0
    if cr:
        rsd[:k] = 0.0

    if cb:  # back-substitute for b
        for jj in range(1, k + 1):
            j = k - jj + 1
            j0 = j - 1
            if qr[j0, j0] == 0.0:
                info = j
                break
            b[j0] = b[j0] / qr[j0, j0]
            if j != 1:
                t = -b[j0]
                b[:j0] = daxpy(qr[:j0, j0], b[:j0].copy(), a=t)

    if cr or cxb:  # compute rsd / xb (descending)
        for jj in range(1, ju + 1):
            j0 = (ju - jj + 1) - 1
            if qraux[j0] != 0.0:
                temp = qr[j0, j0]
                qr[j0, j0] = qraux[j0]
                if cr:
                    t = -ddot(qr[j0:n, j0], rsd[j0:n]) / qr[j0, j0]
                    rsd[j0:n] = daxpy(qr[j0:n, j0], rsd[j0:n].copy(), a=t)
                if cxb:
                    t = -ddot(qr[j0:n, j0], xb[j0:n]) / qr[j0, j0]
                    xb[j0:n] = daxpy(qr[j0:n, j0], xb[j0:n].copy(), a=t)
                qr[j0, j0] = temp
    return qy, qty, b, rsd, xb, info


def dqrls(x: np.ndarray, y: np.ndarray, tol: float = 1e-7):
    """Mechanical port of ``dqrls.f`` (single rhs). Returns
    ``(qr, coef, residuals, effects, rank, jpvt, qraux)`` — ``coef`` is length p
    in *pivoted* order, unused (rank+1..p) entries zeroed; ``effects`` is ``Qᵀy``."""
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    y = np.asarray(y, dtype=float).reshape(-1)
    qr, qraux, jpvt, k = dqrdc2(x, tol)
    coef = np.zeros(p)
    if k > 0:
        _, qty, b_k, rsd, _, _ = dqrsl(qr, n, k, qraux, y, 1110)
        coef[:k] = b_k[:k]
    else:  # k == 0: rsd = y, qty = y
        qty = y.copy()
        rsd = y.copy()
    return qr, coef, rsd, qty, k, jpvt, qraux


def Cdqrls(x: np.ndarray, y: np.ndarray, tol: float = 1e-7) -> dict:
    """Port of ``Cdqrls`` (``stats/src/lm.c``) — the entry ``lm.fit`` calls.
    Returns R's list as a dict: ``qr`` (compact factor), ``coefficients`` (pivoted,
    unused→0), ``residuals``, ``effects`` (``Qᵀy``), ``rank``, ``pivot`` (1-based),
    ``qraux``, ``tol``, ``pivoted``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("'x' is not a matrix")
    n, p = x.shape
    y = np.asarray(y, dtype=float).reshape(-1)
    if n and y.shape[0] % n != 0:
        raise ValueError("dimensions of 'x' and 'y' do not match")
    if not np.all(np.isfinite(x)):
        raise ValueError("NA/NaN/Inf in 'x'")
    if not np.all(np.isfinite(y)):
        raise ValueError("NA/NaN/Inf in 'y'")
    if _rs_dqrls is not None:  # Rust active path (pure-Py = oracle)
        qr, coef, rsd, qty, k, jpvt, qraux = _rs_dqrls(
            np.asfortranarray(x, dtype=float), np.asarray(y, dtype=float), tol
        )
        k = int(k)
    else:
        qr, coef, rsd, qty, k, jpvt, qraux = dqrls(x.copy(), y, tol)
    pivoted = not np.array_equal(jpvt, np.arange(1, p + 1))
    return {
        "qr": qr,
        "coefficients": coef,
        "residuals": rsd,
        "effects": qty,
        "rank": int(k),
        "pivot": jpvt,
        "qraux": qraux,
        "tol": float(tol),
        "pivoted": bool(pivoted),
    }
