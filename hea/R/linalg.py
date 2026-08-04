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

This is the **pure-Python spec + oracle** for the Rust port (``hea/_rs`` →
``dqrls``); it mirrors the Fortran line-for-line (pivot order, rank rule, the
1e-6 norm-recompute branch) so Rust ≡ Python ≡ live R is checkable per
[[rng-rust-port-t2]]. Per [[mechanical-port-never-guess]]: ported against the
real sources, not reverse-engineered. ``pivot`` is returned 1-based (R parity).
"""

from __future__ import annotations

import math

import numpy as np

from .._dispatch import rs_fn

__all__ = ["dqrdc2", "dqrsl", "dqrls", "Cdqrls", "dqrls_rank"]

_rs_dqrls = rs_fn("dqrls")
_rs_dqrls_rank = rs_fn("dqrls_rank")


def dqrls_rank(x: np.ndarray, tol: float = 1e-7):
    """``(rank, pivot)`` from R's ``dqrdc2`` only — for alias/rank detection,
    where the coefficients/effects/QR aren't needed. Rust active path (lean: no
    big-array marshalling), pure-Python ``dqrls`` as the fallback/oracle.
    ``pivot`` is 1-based."""
    x = np.asarray(x, dtype=float)
    if _rs_dqrls_rank is not None:
        # The kernel works column-major, so hand it a Fortran-order array — for
        # polars-origin designs (already F-order) this is a no-op; forcing C with
        # ascontiguousarray would add a needless F→C transpose (then a second
        # transpose inside Rust). asfortranarray only copies for a C-order caller.
        rank, pivot = _rs_dqrls_rank(np.asfortranarray(x, dtype=float), tol)
        return int(rank), np.asarray(pivot)
    _, _, _, _, k, jpvt, _ = dqrls(x.copy(), np.zeros(x.shape[0]), tol)
    return int(k), np.asarray(jpvt)


def dqrdc2(x: np.ndarray, tol: float = 1e-7):
    """Mechanical port of ``dqrdc2.f``. ``x`` is ``(n, p)`` (overwritten on a
    copy here). Returns ``(qr, qraux, jpvt, rank)``: ``qr`` holds R in its upper
    triangle and the Householder vectors below; ``qraux`` the reflector scalars;
    ``jpvt`` the 1-based pivot (column j of the output is column ``jpvt[j]`` of
    the input); ``rank`` the number of independent columns."""
    qr = np.array(x, dtype=float, copy=True)
    n, p = qr.shape
    qraux = np.zeros(p)
    work = np.zeros((p, 2))
    jpvt = np.arange(1, p + 1)  # 1-based column indices (R 'pivot')

    # compute the norms of the columns of x
    if n > 0:
        for j in range(p):
            qraux[j] = np.linalg.norm(qr[:, j])  # dnrm2(n, x(1,j))
            work[j, 0] = qraux[j]
            work[j, 1] = qraux[j] if qraux[j] != 0.0 else 1.0

    # Householder reduction of x
    lup = min(n, p)
    k = p + 1  # Fortran 'k' (1-based rank boundary)
    for l in range(1, lup + 1):  # noqa: E741 — Fortran 'l' (1-based), kept for port fidelity
        l0 = l - 1
        # cycle columns l..p left-to-right until one has non-negligible norm;
        # a column is negligible if its norm fell below tol·(original norm).
        while not (l >= k or qraux[l0] >= work[l0, 1] * tol):
            # cyclic left-shift of columns l..p, moving column l to position p
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
            # Householder transformation for column l (rows l..n)
            nrmxl = np.linalg.norm(qr[l0:n, l0])  # dnrm2(n-l+1, x(l,l))
            if nrmxl != 0.0:
                if qr[l0, l0] != 0.0:
                    nrmxl = math.copysign(nrmxl, qr[l0, l0])
                qr[l0:n, l0] /= nrmxl  # dscal(1/nrmxl)
                qr[l0, l0] += 1.0
                # apply the transformation to the remaining columns + update norms
                for j in range(l, p):  # Fortran j = l+1..p
                    t = -np.dot(qr[l0:n, l0], qr[l0:n, j]) / qr[l0, l0]
                    qr[l0:n, j] += t * qr[l0:n, l0]  # daxpy
                    if qraux[j] != 0.0:
                        tt = 1.0 - (abs(qr[l0, j]) / qraux[j]) ** 2
                        tt = max(tt, 0.0)
                        # re-compute the norm if the reduction was large (BDR 9/99)
                        if abs(tt) >= 1e-6:
                            qraux[j] = qraux[j] * math.sqrt(tt)
                        else:
                            qraux[j] = np.linalg.norm(qr[l0 + 1 : n, j])  # dnrm2(n-l)
                            work[j, 0] = qraux[j]
                qraux[l0] = qr[l0, l0]
                qr[l0, l0] = -nrmxl
    k = min(k - 1, n)
    return qr, qraux, jpvt, k


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
    (decimal ``abcde``: a=qy, b/c/d/e⇒qty, c=b, d=rsd, e=xb) are meaningful."""
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
                t = -np.dot(qr[j0:n, j0], qy[j0:n]) / qr[j0, j0]
                qy[j0:n] += t * qr[j0:n, j0]
                qr[j0, j0] = temp
    if cqty:  # compute Qᵀ·y (ascending j)
        for j in range(1, ju + 1):
            j0 = j - 1
            if qraux[j0] != 0.0:
                temp = qr[j0, j0]
                qr[j0, j0] = qraux[j0]
                t = -np.dot(qr[j0:n, j0], qty[j0:n]) / qr[j0, j0]
                qty[j0:n] += t * qr[j0:n, j0]
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
                b[:j0] += t * qr[:j0, j0]  # daxpy(j-1, t, x(1,j), b)

    if cr or cxb:  # compute rsd / xb (descending)
        for jj in range(1, ju + 1):
            j0 = (ju - jj + 1) - 1
            if qraux[j0] != 0.0:
                temp = qr[j0, j0]
                qr[j0, j0] = qraux[j0]
                if cr:
                    t = -np.dot(qr[j0:n, j0], rsd[j0:n]) / qr[j0, j0]
                    rsd[j0:n] += t * qr[j0:n, j0]
                if cxb:
                    t = -np.dot(qr[j0:n, j0], xb[j0:n]) / qr[j0, j0]
                    xb[j0:n] += t * qr[j0:n, j0]
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
        # F-order X so the column-major kernel copies contiguously (no transpose);
        # no-op for polars-origin (F) designs. y is 1-D (layout-agnostic).
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
