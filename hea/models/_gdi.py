"""mgcv's ``gdi1``, ``gdi2`` and ``pls_fit1`` (gdi.c) as R's arm64 mgcv.so
computes them: the Python twin of the extension's ``gdi.rs``, which gam.fit3
(:mod:`hea.models._fit3`) and gam.fit4 (:mod:`hea.models._fit4`) call when
the extension is absent.

At a converged penalized IRLS fit, gdi1 forms the coefficients and their
derivatives with respect to the log smoothing parameters (by the implicit
function theorem), the derivatives of the deviance, the Pearson statistic,
``tr(A)`` and the REML or ML log determinant terms; gdi2 does the same for an
extended family, its likelihood parameters included; pls_fit1 is the
penalized least squares step of the IRLS. All work on the stable
reparameterization: ``E`` is a square root of the total penalty, ``Es`` its
well scaled version for rank determination and ``rS`` the penalty square
roots side by side.

Matrices are flat, column-major arrays and the routines write through the
arrays they are handed, as in :mod:`hea.models._mgcv_c`, whose rules for
clang's contractions and vectorized reductions hold throughout. The
per-observation weight derivatives, Pearson terms and log determinant sums
follow the compiled contractions set out where each is formed. The ``*_c``
functions take and return the argument lists of ``.C(C_gdi1, ...)``,
``.C(C_gdi2, ...)`` and ``.C(C_pls_fit1, ...)`` as gam.fit3 and gam.fit4
pass them, matrices column-major and flattened.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from .._rfma import _rfma, _rfma_vec
from ._mgcv_c import (
    _inorder,
    diag_abt,
    dot_s8,
    dot_s8_acc,
    dot_s8_from,
    drop_cols,
    drop_rows,
    frobenius_norm,
    get_rpqr,
    get_xtmx,
    get_xtwx,
    get_xtx,
    mat,
    mgcv_backsolve,
    mgcv_forwardsolve,
    mgcv_mmult,
    mgcv_pqrqy,
    mgcv_qr,
    mgcv_qrqy_left,
    mgcv_svd_full,
    pivoter,
    r_cond,
    rc_prod,
    rinv,
    undrop_rows,
)

_EMPTY = np.zeros(0)


def mult_sk(y, x, xcol, k, rs, rsncol, q, work) -> None:
    """gdi.c's ``multSk``: ``y = S_k x`` for the ``q``-by-``xcol`` ``x``,
    through the ``k``-th square root in ``rs`` (``q``-by-``rsncol[k]``,
    after the earlier ones)."""
    off = sum(q * c for c in rsncol[:k])
    rsk = rs[off:]
    nc = rsncol[k]
    mgcv_mmult(work, rsk, x, True, False, nc, xcol, q)
    mgcv_mmult(y, rsk, work, False, False, q, xcol, nc)


def apply_p(y, x, rm, vt, neg_w, nr, r, c, right) -> None:
    """gdi.c's ``applyP``: ``y = P x`` for ``P = R^{-1}``, or ``R^{-1} V``
    when there are negative weights (``vt`` holding ``V'``); ``x`` is
    ``r``-by-``c`` (``right``: ``c``-by-``r``, and ``y = x P``). ``R`` is the
    upper triangle of the ``nr``-row ``rm``."""
    if neg_w:
        x1 = np.zeros(max(r * c, 1))
        if right:
            mgcv_backsolve(rm, nr, r, x, x1, c, True)
            mgcv_mmult(y, x1, vt, False, True, c, r, r)
        else:
            mgcv_mmult(x1, vt, x, True, False, r, c, r)
            mgcv_backsolve(rm, nr, r, x1, y, c, False)
    else:
        mgcv_backsolve(rm, nr, r, x, y, c, right)


def apply_pt(y, x, rm, vt, neg_w, nr, r, c, right) -> None:
    """gdi.c's ``applyPt``: ``y = P' x``, the transpose of
    :func:`apply_p`."""
    if neg_w:
        x1 = np.zeros(max(r * c, 1))
        if right:
            mgcv_mmult(x1, x, vt, False, False, c, r, r)
            mgcv_forwardsolve(rm, nr, r, x1, y, c, True)
        else:
            mgcv_forwardsolve(rm, nr, r, x, x1, c, False)
            mgcv_mmult(y, vt, x1, False, False, r, c, r)
    else:
        mgcv_forwardsolve(rm, nr, r, x, y, c, right)


def ift1(
    rm, vt, x, rs, beta, sp, dwdeta, b1, b2, eta1, eta2, n, r, m, rsncol, deriv2,
    neg_w, nr,
) -> None:  # fmt: skip
    """gdi.c's ``ift1``: the first (``b1``, ``r``-by-``m``) and second
    (``b2``, one column per ``k >= i`` pair) derivatives of the coefficients
    with respect to the log smoothing parameters by the implicit function
    theorem, and those of the linear predictor (``eta1``, ``eta2``)."""
    nw = max(n, r, 1)
    work = np.zeros(nw)
    work1 = np.zeros(nw)
    skb = np.zeros(max(r, 1))
    n_2d = (m * (1 + m)) // 2
    for i in range(m):
        mult_sk(skb, beta, 1, i, rs, rsncol, r, work)
        skb[:r] *= -sp[i]
        apply_pt(work, skb, rm, vt, neg_w, nr, r, 1, False)
        apply_p(b1[i * r :], work, rm, vt, neg_w, nr, r, 1, False)
    mgcv_mmult(eta1, x, b1, False, False, n, m, r)

    if deriv2:
        pp = 0
        for i in range(m):
            for k in range(i, m):
                p0 = eta1[n * i : n * i + n]
                p1 = eta1[n * k : n * k + n]
                work[:n] = (-p0 * p1) * dwdeta[:n]
                mgcv_mmult(skb, x, work, True, False, r, 1, n)
                mult_sk(work, b1[k * r :], 1, i, rs, rsncol, r, work1)
                skb[:r] = _rfma_vec(-sp[i], work[:r], skb[:r])
                mult_sk(work, b1[i * r :], 1, k, rs, rsncol, r, work1)
                skb[:r] = _rfma_vec(-sp[k], work[:r], skb[:r])
                apply_pt(work, skb, rm, vt, neg_w, nr, r, 1, False)
                apply_p(b2[pp:], work, rm, vt, neg_w, nr, r, 1, False)
                if i == k:
                    b2[pp : pp + r] += b1[i * r : i * r + r]
                pp += r
        mgcv_mmult(eta2, x, b2, False, False, n, n_2d, r)


def get_bsb(
    bsb1, bsb2, sp, e, rs, rsncol, enrow, q, m, m0, beta, b1, b2, deriv
) -> float:
    """gdi.c's ``get_bSb``: returns ``beta' S beta`` and, for ``deriv >
    0``, writes its first (``bsb1``) and second (``bsb2``) derivatives with
    respect to the ``m0`` likelihood parameters (zero) and the ``m`` log
    smoothing parameters. ``b2`` holds one column per ``k >= m`` pair.
    ``beta' S beta`` accumulates through the output pointer: 8-wide from 12
    terms on."""
    jw = q
    for c in rsncol[:m]:
        jw = max(jw, c)
    jw += m0
    work = np.zeros(max(jw, 1))
    sb = np.zeros(max(q, 1))
    mgcv_mmult(work, e, beta, False, False, enrow, 1, q)
    mgcv_mmult(sb, e, work, True, False, q, 1, enrow)
    bsb = dot_s8_from(beta[:q], sb[:q], 12)
    if deriv <= 0:
        return bsb

    work1 = np.zeros(max(jw, 1))
    skb = np.zeros(max(m * q, 1))
    rsoff = 0
    for i in range(m):
        nc = rsncol[i]
        mgcv_mmult(work, rs[rsoff:], beta, True, False, nc, 1, q)
        work[:nc] *= sp[i]
        p1 = skb[i * q :]
        mgcv_mmult(p1, rs[rsoff:], work, False, False, q, 1, nc)
        rsoff += q * nc
        bsb1[i + m0] = dot_s8(beta[:q], p1[:q])
    bsb1[:m0] = 0.0
    mtot = m + m0

    if deriv > 1:
        pb2 = 0
        for mm in range(mtot):
            mgcv_mmult(work1, e, b1[mm * q :], False, False, enrow, 1, q)
            mgcv_mmult(work, e, work1, True, False, q, 1, enrow)
            for k in range(mm, mtot):
                km = k * mtot + mm
                mk = mm * mtot + k
                xx = dot_s8(b2[pb2 : pb2 + q], sb[:q])
                pb2 += q
                bsb2[km] = xx + xx
                xx = dot_s8(work[:q], b1[k * q : k * q + q])
                bsb2[km] = _rfma(xx, 2.0, bsb2[km])
                if k >= m0:
                    o = (k - m0) * q
                    xx = dot_s8(b1[mm * q : mm * q + q], skb[o : o + q])
                    bsb2[km] = _rfma(xx, 2.0, bsb2[km])
                if mm >= m0:
                    o = (mm - m0) * q
                    xx = dot_s8(b1[k * q : k * q + q], skb[o : o + q])
                    bsb2[km] = _rfma(xx, 2.0, bsb2[km])
                if k == mm:
                    bsb2[km] += bsb1[k]
                else:
                    bsb2[mk] = bsb2[km]

    mgcv_mmult(work, b1, sb, True, False, mtot, 1, q)
    for i in range(mtot):
        bsb1[i] = _rfma(2.0, work[i], bsb1[i])
    return bsb


def pearson2(
    p1, p2, y, mu, v, v1, v2, g1, g2, p_weights, eta1, eta2, n, m, deriv, deriv2
) -> float:
    """gdi.c's ``pearson2``: returns the Pearson statistic and writes its
    first and second derivatives with respect to the log smoothing
    parameters, through its derivatives with respect to the linear
    predictor."""
    n_2d = (m * (1 + m)) // 2 if deriv2 else 0
    resid = y[:n] - mu[:n]
    xx = (resid * p_weights[:n]) / v[:n]
    p = 0.0
    for a, b in zip(xx.tolist(), resid.tolist()):
        p = _rfma(a, b, p)
    if not deriv:
        return p
    g1n, v1n = g1[:n], v1[:n]
    pe1 = -(xx * _rfma_vec(resid, v1n, 2.0)) / g1n
    if deriv2:
        t = _rfma_vec(2.0 * xx, v1n, (p_weights[:n] + p_weights[:n]) / v[:n])
        t = _rfma_vec(-(pe1 * v1n), g1n, t)
        t = _rfma_vec(-(xx * resid), _rfma_vec(-v1n, v1n, v2[:n]), t)
        pe2 = t / (g1n * g1n) - (g2[:n] * pe1) / g1n

    pi1 = np.zeros(n * m)
    rc_prod(pi1, pe1, eta1, m, n)
    if deriv2:
        pi2 = np.zeros(n_2d * n)
        v2w = np.zeros(n)
        rc_prod(pi2, pe1, eta2, n_2d, n)
        pp = 0
        for mm in range(m):
            for k in range(mm, m):
                rc_prod(pe1, eta1[n * mm :], eta1[n * k :], 1, n)
                rc_prod(v2w, pe2, pe1, 1, n)
                pi2[pp : pp + n] += v2w
                pp += n
    for k in range(m):
        p1[k] = _inorder(pi1[n * k : n * k + n])
    if deriv2:
        pp = 0
        for mm in range(m):
            for k in range(mm, m):
                xx2 = _inorder(pi2[pp : pp + n])
                pp += n
                p2[k * m + mm] = xx2
                p2[mm * m + k] = xx2
    return p


def get_ddet_xwxps(
    det1, det2, p, k_, sp, rs, rsncol, tk, tkm, n, q, r, m, m0, deriv
) -> None:
    """gdi.c's ``get_ddetXWXpS``: the first (``det1``) and second (``det2``)
    derivatives of ``log|X'WX + S|`` with respect to the ``m0`` likelihood
    parameters and ``m`` log smoothing parameters, from ``P``
    (``q``-by-``r``), ``K`` (``n``-by-``r``) and the weight derivative
    ratios ``tk``, ``tkm``."""
    if deriv == 0:
        return
    mtot = m0 + m
    deriv2 = deriv == 2
    diag_kkt = np.zeros(max(n, 1))
    diag_abt(diag_kkt, k_, k_, n, r)
    work = np.zeros(max(n, r, 1))
    rr = r * r
    kt_tk = np.zeros(max(rr * mtot, 1)) if deriv2 else _EMPTY
    if deriv2:
        for k in range(mtot):
            get_xtwx(kt_tk[k * rr :], k_, tk[k * n :], n, r)
    mgcv_mmult(det1, tk, diag_kkt, True, False, mtot, 1, n)

    max_col = q
    for c in rsncol[:m]:
        max_col = max(max_col, c)
    ptr_sm = np.zeros(max(r * max_col, 1))
    tr_ptsp = np.zeros(max(m, 1))
    ptsp = np.zeros(max(m * rr, 1)) if deriv2 else _EMPTY
    rsoff = [0] * max(m, 1)
    for mm in range(1, m):
        rsoff[mm] = rsoff[mm - 1] + rsncol[mm - 1]
    for mm in range(m):
        nc = rsncol[mm]
        mgcv_mmult(ptr_sm, p, rs[rsoff[mm] * q :], True, False, r, nc, q)
        tr_ptsp[mm] = sp[mm] * diag_abt(work, ptr_sm, ptr_sm, r, nc)
        det1[mm + m0] += tr_ptsp[mm]
        if deriv2:
            mgcv_mmult(ptsp[mm * rr :], ptr_sm, ptr_sm, False, True, r, r, nc)

    if deriv2:
        for mm in range(mtot):
            ptkm = 0 if mm == 0 else (mm * mtot - (mm * (mm - 1)) // 2) * n
            for k in range(mm, mtot):
                km = k * mtot + mm
                mk = mm * mtot + k
                xx = dot_s8(tkm[ptkm : ptkm + n], diag_kkt[:n])
                ptkm += n
                det2[km] = xx
                det2[km] -= diag_abt(work, kt_tk[k * rr :], kt_tk[mm * rr :], r, r)
                if k >= m0 and k == mm:
                    det2[km] += tr_ptsp[mm - m0]
                if mm >= m0:
                    tr = diag_abt(work, kt_tk[k * rr :], ptsp[(mm - m0) * rr :], r, r)
                    det2[km] = _rfma(-sp[mm - m0], tr, det2[km])
                if k >= m0:
                    tr = diag_abt(work, kt_tk[mm * rr :], ptsp[(k - m0) * rr :], r, r)
                    det2[km] = _rfma(-sp[k - m0], tr, det2[km])
                if k >= m0 and mm >= m0:
                    s = sp[mm - m0] * sp[k - m0]
                    tr = diag_abt(
                        work, ptsp[(k - m0) * rr :], ptsp[(mm - m0) * rr :], r, r
                    )
                    det2[km] = _rfma(-s, tr, det2[km])
                det2[mk] = det2[km]


def get_tra2(
    tr_a1, tr_a2, p, k_, sp, rs, rsncol, tk, tkm, w, n, q, r, m, deriv
) -> float:
    """gdi.c's ``get_trA2``: returns ``tr(A)`` for the influence matrix ``A
    = K K'`` (signed by the weights when some are negative) and, for ``deriv
    > 0``, writes its first and second derivatives with respect to the log
    smoothing parameters."""
    deriv2 = deriv == 2
    wn = w[:n]
    ip = np.where(wn < 0.0, -1.0, 1.0)
    neg_w = bool(np.any(wn < 0.0))
    diag_kkt = np.zeros(max(n, 1))
    tr_a = diag_abt(diag_kkt, k_, k_, n, r)
    if neg_w:
        # accumulated through the output pointer: 8-wide from 34 terms on
        tr_a = dot_s8_from(ip, diag_kkt[:n], 34)
    if deriv == 0:
        return tr_a

    work = np.zeros(max(n, r, 1))
    ktk = np.zeros(max(r * r, 1))
    ipk = np.zeros(max(r * n, 1))
    if neg_w:
        ipk[: n * r] = (ip[:, None] * mat(k_, n, r)).ravel(order="F")
    else:
        ipk[: n * r] = k_[: n * r]
    mgcv_mmult(ktk, k_, ipk, True, False, r, r, n)
    del ipk
    kktk = np.zeros(max(n * r, 1))
    mgcv_mmult(kktk, k_, ktk, False, False, n, r, r)
    diag_kktkkt = np.zeros(max(n, 1))
    diag_abt(diag_kktkkt, kktk, k_, n, r)

    rr = r * r
    kt_tk = np.zeros(max(rr * m, 1)) if deriv2 else _EMPTY
    kt_tkktk = np.zeros(max(rr * m, 1)) if deriv2 else _EMPTY
    if deriv2:
        for k in range(m):
            get_xtwx(kt_tk[k * rr :], k_, tk[k * n :], n, r)
            mgcv_mmult(kt_tkktk[k * rr :], kt_tk[k * rr :], ktk, False, False, r, r, r)

    mgcv_mmult(tr_a1, tk, diag_kkt, True, False, m, 1, n)
    mgcv_mmult(work, tk, diag_kktkkt, True, False, m, 1, n)
    for i in range(m):
        tr_a1[i] += -work[i]

    if deriv2:
        diff = diag_kkt[:n] - diag_kktkkt[:n]
        ptkm = 0
        for mm in range(m):
            for k in range(mm, m):
                km = k * m + mm
                mk = mm * m + k
                xx = dot_s8(tkm[ptkm : ptkm + n], diff)
                ptkm += n
                tr_a2[km] = xx
                tr = diag_abt(work, kt_tk[k * rr :], kt_tk[mm * rr :], r, r)
                tr_a2[km] = _rfma(-2.0, tr, tr_a2[km])
                tr = diag_abt(work, kt_tk[k * rr :], kt_tkktk[mm * rr :], r, r)
                tr_a2[km] += tr + tr
                tr_a2[mk] = tr_a2[km]
    del kt_tkktk, kt_tk

    ptr_sm = np.zeros(max(r * q, 1))
    kptr_sm = np.zeros(max(n * q, 1))
    diag_kptspkt = np.zeros(max(n * m, 1))
    if deriv2:
        ptsp = np.zeros(max(m * rr, 1))
        ptspktk = np.zeros(max(m * rr, 1))
        ktkptr_sm = np.zeros(max(r * q, 1))
        kktkptr_sm = np.zeros(max(n * q, 1))
        diag_kptspktkkt = np.zeros(max(n * m, 1))
    rsoff = [0] * max(m, 1)
    for mm in range(1, m):
        rsoff[mm] = rsoff[mm - 1] + rsncol[mm - 1]
    for mm in range(m):
        nc = rsncol[mm]
        mgcv_mmult(ptr_sm, p, rs[rsoff[mm] * q :], True, False, r, nc, q)
        mgcv_mmult(kptr_sm, k_, ptr_sm, False, False, n, nc, r)
        if deriv2:
            mgcv_mmult(ktkptr_sm, ktk, ptr_sm, False, False, r, nc, r)
            mgcv_mmult(ptsp[mm * rr :], ptr_sm, ptr_sm, False, True, r, r, nc)
            mgcv_mmult(kktkptr_sm, kktk, ptr_sm, False, False, n, nc, r)
            mgcv_mmult(ptspktk[mm * rr :], ptr_sm, ktkptr_sm, False, True, r, r, nc)
            diag_abt(diag_kptspktkkt[mm * n :], kptr_sm, kktkptr_sm, n, nc)
        xx = sp[mm] * diag_abt(diag_kptspkt[mm * n :], kptr_sm, kptr_sm, n, nc)
        if neg_w:
            xx = dot_s8(diag_kptspkt[mm * n : mm * n + n], ip)
            xx *= sp[mm]
        tr_a1[mm] -= xx
        if deriv2:
            tr_a2[mm * m + mm] -= xx
    if not deriv2:
        return tr_a
    for mm in range(m):
        for k in range(mm, m):
            km = k * m + mm
            mk = mm * m + k
            tk_k = tk[k * n : k * n + n]
            tk_m = tk[mm * n : mm * n + n]
            spm, spk = sp[mm], sp[k]
            xx = dot_s8(diag_kptspktkkt[mm * n : mm * n + n], tk_k)
            tr_a2[km] = _rfma(spm + spm, xx, tr_a2[km])
            xx = dot_s8(diag_kptspktkkt[k * n : k * n + n], tk_m)
            tr_a2[km] = _rfma(spk + spk, xx, tr_a2[km])
            xx = dot_s8(diag_kptspkt[mm * n : mm * n + n], tk_k)
            tr_a2[km] = _rfma(-spm, xx, tr_a2[km])
            xx = dot_s8(diag_kptspkt[k * n : k * n + n], tk_m)
            tr_a2[km] = _rfma(-spk, xx, tr_a2[km])
            tr = diag_abt(work, ptsp[mm * rr :], ptspktk[k * rr :], r, r)
            tr_a2[km] = _rfma((spk + spk) * spm, tr, tr_a2[km])
            tr_a2[mk] = tr_a2[km]
    return tr_a


def neg_w_scaling(d: np.ndarray, work: np.ndarray | None) -> float:
    """``1 - 2 d^2`` for each singular value of the negative-weight rows,
    ``(1 - 2 d^2)^{-1/2}`` (zero when not positive) left in ``d``, and
    ``log|I - 2 D^2|`` returned; ``work`` (when given) receives the square
    roots."""
    ldet = 0.0
    for i in range(d.size):
        di = float(d[i])
        v = _rfma(-2.0 * di, di, 1.0)
        if v <= 0.0:
            d[i] = 0.0
            if work is not None:
                work[i] = 0.0
        else:
            ldet += math.log(v)
            s = math.sqrt(v)
            if work is not None:
                work[i] = s
            d[i] = 1.0 / s
    return ldet


def ml_penalty1(
    det1, det2, tk, tkm, nulli, rm, qm1, nind, sp, rs, rsncol, q, n, m, m0, deriv, x,
    type1,
) -> float:  # fmt: skip
    """gdi.c's ``MLpenalty1``: ``log|X'WX + S|`` projected into the penalty
    range space, as ML needs it, with its derivatives through
    :func:`get_ddet_xwxps`. Works on the rank-truncated, pivoted problem:
    ``nulli`` marks null space parameters (positive), ``rm`` is the
    ``q``-by-``q`` QR factor and ``qm1`` its ``n``-by-``q`` orthogonal
    factor. ``rs`` is overwritten."""
    neg_w = len(nind)
    drop = [i for i in range(q) if nulli[i] > 0.0]
    scs = sum(rsncol[:m])
    qm = q - len(drop)

    ru1 = np.zeros(max(q * q, 1))
    ru1[: q * q] = rm[: q * q]
    drop_cols(ru1, q, q, drop)
    tau = np.zeros(max(qm, 1))
    pivot = np.zeros(max(qm, 1), dtype=np.int32)
    mgcv_qr(ru1, q, qm, pivot, tau)
    ri = np.zeros(max(qm * qm, 1))
    rinv(ri, ru1, qm, q, qm)
    # the new Q factor for type 0 or negative weights, else X without the
    # null space columns
    qb = np.zeros(max(q * qm, 1))
    xu1 = _EMPTY
    if not type1 or neg_w > 0:
        for i in range(qm):
            qb[i * q + i] = 1.0
        mgcv_qrqy_left(qb, ru1, tau, q, qm, qm, False)
    else:
        xu1 = np.zeros(max(n * q, 1))
        xu1[: n * q] = x[: n * q]
        drop_cols(xu1, n, q, drop)

    k_ = np.zeros(max(n * qm, 1))
    p = np.zeros(max(qm * qm, 1))
    ldet_i2d = 0.0
    if neg_w > 0:
        k = max(q + 1, neg_w)
        iq = np.zeros(k * q)
        mat(iq, k, q)[:neg_w, :] = mat(qm1, n, q)[nind, :]
        iqq = np.zeros(max(k * qm, 1))
        mgcv_mmult(iqq, iq, qb, False, False, k, qm, q)
        vt = np.zeros(max(qm * qm, 1))
        d = np.zeros(qm)
        mgcv_svd_full(iqq, vt, d, k, qm)
        ldet_i2d = neg_w_scaling(d, None)
        mat(vt, qm, qm)[...] *= d[:, None]
        mgcv_mmult(p, ri, vt, False, True, qm, qm, qm)
        if type1:
            # mgcv forms K from XU1 here, which it allocated only without
            # negative weights
            return math.nan
        work = np.zeros(max(q * qm, 1))
        mgcv_mmult(work, qb, vt, False, True, q, qm, qm)
        mgcv_mmult(k_, qm1, work, False, False, n, qm, q)
    else:
        p[: qm * qm] = ri[: qm * qm]
        if type1:
            mgcv_mmult(k_, xu1, p, False, False, n, qm, qm)
        else:
            mgcv_mmult(k_, qm1, qb, False, False, n, qm, q)

    ldet = 0.0
    for i in range(qm):
        ldet += math.log(abs(float(ru1[i + i * q])))
    ldet += ldet
    ldet += ldet_i2d

    drop_rows(rs, q, scs, drop)
    pivoter(rs, qm, scs, pivot, False, False)
    if deriv != 0:
        get_ddet_xwxps(
            det1, det2, p, k_, sp, rs, rsncol, tk, tkm, n, qm, qm, m, m0, deriv
        )
    return ldet


def gdi_pk(
    x, e, es, rs, z, raw, nind, n, q, mp, enrow, deriv2, scs, reml, rank_tol, type1
) -> SimpleNamespace:
    """gdi.c's ``gdiPK``: the QR of ``sqrt|W| X``, the rank of the penalized
    problem from the well scaled ``[R; Es]``, the unidentifiable columns
    dropped from ``x``, ``e`` and the rows of ``rs``, the QR of ``[R; E]``,
    and ``P``, ``K``, ``Rh`` and the coefficients, corrected for negative
    weights. With ``type1`` (gdi2's type 1, for zero weights) ``z`` holds
    ``Wz`` and ``K = XP``, the coefficients ``P K'Wz``.

    Returns what it leaves for the rest of gdi1 and gdi2: ``r`` (``[R; E]``
    after the final QR, ``nr`` rows, its factor ``R`` in the upper
    triangle), ``rank``, ``pivot1`` (the final QR's 0-based pivots),
    ``drop`` (the columns dropped as unidentifiable, ascending), ``nulli``
    (``+1`` for a parameter in the penalty null space, ``-1`` otherwise),
    ``dev_hess`` (``X'WX`` from the first QR, pivoted, when second
    derivatives are wanted), ``rh`` (``Rh'Rh = X'WX + S``), ``p`` and ``k``
    (``P`` and ``K`` of Wood, 2008), ``vt`` (``(I - 2 D^2)^{-1/2} V'`` for
    negative weights), ``pktz`` (the pivoted coefficients), ``q1`` (``Qf
    Qs[1:q, ]``) and ``ldet_xwxs``."""
    neg_w = len(nind)
    # type 0: z = [sqrt(|W|)z, 0]; type 1: z holds Wz on entry
    zz = np.zeros(max(n, 1))
    if not type1:
        zz[:n] = z[:n] * raw[:n]
        for k in nind:
            zz[k] = -zz[k]
    wx = np.zeros(max(n * q, 1))
    wx[: n * q] = (raw[:n, None] * mat(x, n, q)).ravel(order="F")
    tau = np.zeros(max(q, 1))
    pivot = np.zeros(max(q, 1), dtype=np.int32)
    mgcv_qr(wx, n, q, pivot, tau)

    rr = min(q, n)
    r1 = np.zeros(max(rr * q, 1))
    get_rpqr(r1, wx, n, q, rr)
    pivoter(r1, rr, q, pivot, True, True)

    rnorm = frobenius_norm(r1, rr, q)
    enorm = frobenius_norm(es, enrow, q)
    nr = rr + enrow
    r = np.zeros(max(nr * q, 1))
    R = mat(r, nr, q)
    R[:rr, :] = mat(r1, rr, q) / rnorm
    R[rr:, :] = mat(es, enrow, q) / enorm
    tau1 = np.zeros(max(q, 1))
    pivot1 = np.zeros(max(q, 1), dtype=np.int32)
    mgcv_qr(r, nr, q, pivot1, tau1)
    rank = min(q, nr)
    rcond = r_cond(r, nr, rank)
    while rank_tol * rcond > 1.0:
        rank -= 1
        rcond = r_cond(r, nr, rank)

    nulli = np.zeros(max(q, 1))
    nulli[:q] = np.where(np.arange(q) < q - mp, -1.0, 1.0)
    n_drop = q - rank
    drop = sorted(int(v) for v in pivot1[rank : rank + n_drop])
    if n_drop > 0:
        drop_cols(r1, rr, q, drop)
        drop_cols(e, enrow, q, drop)
        drop_cols(x, n, q, drop)
        drop_rows(rs, q, scs, drop)
        drop_rows(nulli, q, 1, drop)

    R = mat(r, nr, rank)
    R[:rr, :] = mat(r1, rr, rank)
    R[rr:, :] = mat(e, enrow, rank)
    mgcv_qr(r, nr, rank, pivot1, tau1)
    pivoter(nulli, rank, 1, pivot1, False, False)

    dev_hess = _EMPTY
    if deriv2:
        pivoter(r1, rr, rank, pivot1, True, False)
        dev_hess = np.zeros(max(q * q, 1))
        get_xtx(dev_hess, r1, rr, rank)

    # Q1 = Qf Qs[1:q, ]: needed for the negative-w correction and for type 0
    q1 = np.zeros(max(n * q, 1))
    if neg_w > 0 or not type1:
        qm = np.zeros(max(nr * rank, 1))
        for i in range(rank):
            qm[i * nr + i] = 1.0
        mgcv_qrqy_left(qm, r, tau1, nr, rank, rank, False)
        mat(q1, rr, rank)[...] = mat(qm, nr, rank)[:rr, :]
        mgcv_pqrqy(q1, wx, tau, n, rr, rank, False)

    ri = np.zeros(max(rank * rank, 1))
    rinv(ri, r, rank, nr, rank)

    p = np.zeros(max(q * q, 1))
    k_ = np.zeros(max(n * q, 1))
    rh = np.zeros(max(q * q, 1))
    vt = _EMPTY
    ldet_i2d = 0.0
    if neg_w > 0:
        vt = np.zeros(max(q * q, 1))
        k = max(rank + 1, neg_w)
        iq = np.zeros(k * rank)
        mat(iq, k, rank)[:neg_w, :] = mat(q1, n, rank)[nind, :]
        d = np.zeros(rank)
        mgcv_svd_full(iq, vt, d, k, rank)
        if deriv2:
            # P temporarily holds D V'R, R upper triangular
            for j in range(rank):
                p1 = p[j * rank : j * rank + rank]
                p1[:] = vt[:rank] * r[j * nr]
                for kk in range(1, j + 1):
                    p1[:] = _rfma_vec(
                        vt[rank * kk : rank * kk + rank], r[kk + j * nr], p1
                    )
            mat(p, rank, rank)[...] *= d[:, None]
            get_xtx(k_, p, rank, rank)
            dev_hess[: rank * rank] = _rfma_vec(
                -2.0, k_[: rank * rank], dev_hess[: rank * rank]
            )
        work = np.zeros(rank)
        ldet_i2d = neg_w_scaling(d, work)
        # the upper triangle of R into P
        mat(p, rank, rank)[...] = np.triu(mat(r, nr, rank)[:rank, :])
        mgcv_mmult(rh, vt, p, False, False, rank, rank, rank)
        mat(rh, rank, rank)[...] *= work[:, None]
        mat(vt, rank, rank)[...] *= d[:, None]
        mgcv_mmult(p, ri, vt, False, True, rank, rank, rank)
        if not type1:
            mgcv_mmult(k_, q1, vt, False, True, n, rank, rank)
    else:
        if not type1:
            k_[: n * rank] = q1[: n * rank]
        p[: rank * rank] = ri[: rank * rank]
        mat(rh, rank, rank)[...] = np.triu(mat(r, nr, rank)[:rank, :])

    ldet_xwxs = 0.0
    if reml > 0:
        for i in range(rank):
            ldet_xwxs += math.log(abs(float(r[i + i * nr])))
        ldet_xwxs += ldet_xwxs
        ldet_xwxs += ldet_i2d

    pivoter(rs, rank, scs, pivot1, False, False)
    pivoter(e, enrow, rank, pivot1, True, False)
    pivoter(x, n, rank, pivot1, True, False)

    vtp = vt if neg_w > 0 else None
    work = np.zeros(3 * max(q, 1))
    pktz = np.zeros(max(q, 1))
    pk = SimpleNamespace(
        r=r, nr=nr, rank=rank, pivot1=pivot1, drop=drop, nulli=nulli,
        dev_hess=dev_hess, rh=rh, p=p, k=k_, vt=vt, pktz=pktz, q1=q1,
        ldet_xwxs=ldet_xwxs,
    )  # fmt: skip
    if type1:
        # K = X P, then P K'Wz
        apply_p(k_, x, r, vtp, neg_w > 0, nr, rank, n, True)
        mgcv_mmult(work, k_, z, True, False, rank, 1, n)
        apply_p(pktz, work, r, vtp, neg_w > 0, nr, rank, 1, False)
        return pk
    mgcv_mmult(work, k_, zz, True, False, rank, 1, n)
    mgcv_mmult(work[2 * q :], q1, zz, True, False, rank, 1, n)
    zz[:n] = raw[:n] * raw[:n] * z[:n]
    for k in nind:
        zz[k] = -zz[k]
    mgcv_mmult(work[q:], x, zz, True, False, rank, 1, n)
    norm1 = norm2 = 0.0
    for i in range(rank):
        xx = dot_s8(r[nr * i : nr * i + i + 1], work[2 * q : 2 * q + i + 1])
        xx -= work[i + q]
        norm1 = _rfma(xx, xx, norm1)
        norm2 = _rfma(work[i + q], work[i + q], norm2)
    if norm1 > rank_tol * norm2:
        apply_pt(zz, work[q:], r, vtp, neg_w > 0, nr, rank, 1, False)
        apply_p(pktz, zz, r, vtp, neg_w > 0, nr, rank, 1, False)
    else:
        apply_p(pktz, work, r, vtp, neg_w > 0, nr, rank, 1, False)
    return pk


def _raw_weights(w: np.ndarray, n: int) -> tuple[np.ndarray, list[int]]:
    """``sqrt|w|`` (``n`` used, at least one long) and the rows whose weight
    is negative."""
    raw = np.zeros(max(n, 1))
    wn = w[:n]
    raw[:n] = np.sqrt(np.where(wn < 0.0, -wn, wn))
    return raw, np.flatnonzero(wn < 0.0).tolist()


def gdi1(a: SimpleNamespace) -> None:
    """gdi.c's ``gdi1``, one thread, on ``a``'s fields (the arguments of
    ``.C(C_gdi1, ...)``, the arrays it writes included), which it updates in
    place as the C writes through its pointer arguments."""
    n, q, m, mp, enrow = a.n, a.q, a.m, a.mp, a.enrow
    deriv = a.deriv
    deriv2 = deriv == 2
    reml = a.reml
    fisher = a.fisher
    rsncol = list(a.rsncol)
    scs = sum(rsncol[:m])

    raw, nind = _raw_weights(a.w, n)
    neg_w = len(nind) > 0

    pk = gdi_pk(
        a.x, a.e, a.es, a.rs, a.z, raw, nind, n, q, mp, enrow, deriv2, scs, reml,
        a.rank_tol, False,
    )  # fmt: skip
    rank = pk.rank
    nr = pk.nr
    vtp = pk.vt if neg_w else None

    n_work = max((4 * n + 2 * q) * m + 2 * n, 5 * q, (m * (1 + m)) // 2 * n)
    work = np.zeros(max(n_work, 1))

    n_2d = (m * (1 + m)) // 2
    b2 = eta1 = eta2 = tk = tkm = w2 = a1 = a2 = af1 = af2 = _EMPTY
    dev_grad = tfk = tfkm = _EMPTY
    if deriv != 0:
        b2 = np.zeros(max(rank * n_2d, 1))
        eta1 = np.zeros(max(n * m, 1))
        tk = np.zeros(max(n * m, 1))
        eta2 = np.zeros(max(n * n_2d, 1))
        tkm = np.zeros(max(n * n_2d, 1))
        w2 = np.zeros(max(n * n_2d, 1))
        a1 = np.zeros(max(n, 1))
        a2 = np.zeros(max(n, 1))
        w, v1, g1, g2 = a.w[:n], a.v1[:n], a.g1[:n], a.g2[:n]
        v2, g3 = a.v2[:n], a.g3[:n]
        if fisher:
            a1[:n] = -(w * _rfma_vec(g2, 2.0, v1)) / g1
            t = _rfma_vec(-2.0 * g2, g2, _rfma_vec(g3, 2.0, _rfma_vec(-v1, v1, v2)))
            a2[:n] = _rfma_vec(a1[:n], a1[:n] / w - g2 / g1, -(w * t) / (g1 * g1))
        else:
            xx = _rfma_vec(-g2, g2, _rfma_vec(-v1, v1, v2) + g3)
            ymu = a.y[:n] - a.mu[:n]
            alpha1 = _rfma_vec(ymu, xx, -(v1 + g2)) / a.alpha[:n]
            t = _rfma_vec(v1 * -3.0, v2, a.v3[:n])
            t = _rfma_vec(v1 * (v1 + v1), v1, t)
            t = a.g4[:n] + t
            t = _rfma_vec(g3 * -3.0, g2, t)
            t = _rfma_vec(g2 * (g2 + g2), g2, t)
            alpha2 = _rfma_vec(xx, -2.0, ymu * t) / a.alpha[:n]
            a1[:n] = (w * _rfma_vec(g2, -2.0, alpha1 - v1)) / g1
            t = _rfma_vec(alpha1, alpha1, -alpha2) + v2
            t = _rfma_vec(-2.0 * g2, g2, _rfma_vec(g3, 2.0, _rfma_vec(-v1, v1, t)))
            a2[:n] = _rfma_vec(a1[:n], a1[:n] / w - g2 / g1, -(w * t) / (g1 * g1))
            if reml == 0:
                wf = a.wf[:n]
                af1 = -(wf * _rfma_vec(g2, 2.0, v1)) / g1
                t = _rfma_vec(-2.0 * g2, g2, _rfma_vec(g3, 2.0, _rfma_vec(-v1, v1, v2)))
                af2 = _rfma_vec(af1, af1 / wf - g2 / g1, -(wf * t) / (g1 * g1))

        work[:n] = ((a.p_weights[:n] * -2.0) * (a.y[:n] - a.mu[:n])) / (
            a.v0[:n] * a.g1[:n]
        )
        dev_grad = np.zeros(max(rank, 1))
        mgcv_mmult(dev_grad, a.x, work, True, False, rank, 1, n)
        if deriv2:
            pk.dev_hess[: rank * rank] += pk.dev_hess[: rank * rank]

        ift1(
            pk.r, vtp, a.x, a.rs, pk.pktz, a.sp, a1, a.b1, b2, eta1, eta2, n, rank,
            m, rsncol, deriv2, neg_w, nr,
        )  # fmt: skip
        if m > 0:
            mgcv_mmult(work, pk.rh, a.b1, False, False, rank, m, rank)
            get_xtx(a.dvkk, work, rank, m)

        v1p = work
        v2p = work[n * m :]
        rc_prod(a.w1, a1, eta1, m, n)
        if deriv2:
            rc_prod(w2, a1, eta2, n_2d, n)
            pw2 = 0
            for mm in range(m):
                for k in range(mm, m):
                    rc_prod(v1p, eta1[n * mm :], eta1[n * k :], 1, n)
                    rc_prod(v2p, a2, v1p, 1, n)
                    w2[pw2 : pw2 + n] += v2p[:n]
                    pw2 += n
        wi = 1.0 / np.abs(a.w[:n])
        rc_prod(tk, wi, a.w1, m, n)
        if deriv2:
            rc_prod(tkm, wi, w2, n_2d, n)
        if reml == 0 and not fisher:
            rc_prod(a.w1, af1, eta1, m, n)
            tfk = np.zeros(max(n * m, 1))
            tfkm = np.zeros(max(n * n_2d, 1))
            if deriv2:
                rc_prod(w2, af1, eta2, n_2d, n)
                pw2 = 0
                for mm in range(m):
                    for k in range(mm, m):
                        rc_prod(v1p, eta1[n * mm :], eta1[n * k :], 1, n)
                        rc_prod(v2p, af2, v1p, 1, n)
                        w2[pw2 : pw2 + n] += v2p[:n]
                        pw2 += n
            wi = 1.0 / a.wf[:n]
            rc_prod(tfk, wi, a.w1, m, n)
            if deriv2:
                rc_prod(tfkm, wi, w2, n_2d, n)

        mgcv_mmult(a.d1, a.b1, dev_grad, True, False, m, 1, rank)
        if deriv2:
            get_xtmx(a.d2, a.b1, pk.dev_hess, rank, m)
            pb2 = 0
            for mm in range(m):
                for k in range(mm, m):
                    xx = dot_s8(dev_grad[:rank], b2[pb2 : pb2 + rank])
                    pb2 += rank
                    a.d2[k * m + mm] += xx
                    a.d2[mm * m + k] = a.d2[k * m + mm]

    if reml < 0:
        # R kept as rank-by-rank for MLpenalty1
        pk.r[: rank * rank] = np.triu(mat(pk.r, nr, rank)[:rank, :]).ravel(order="F")

    bsb = 0.0
    if reml != 0:
        bsb = get_bsb(
            a.tr_a1, a.tr_a2, a.sp, a.e, a.rs, rsncol, enrow, rank, m, 0, pk.pktz,
            a.b1, b2, deriv,
        )  # fmt: skip
        if deriv != 0:
            for i in range(m):
                a.d1[i] += a.tr_a1[i]
                if deriv2:
                    for j in range(m):
                        a.d2[i * m + j] += a.tr_a2[i * m + j]

    a.p0 = pearson2(
        a.p1, a.p2, a.y, a.mu, a.v0, a.v1, a.v2, a.g1, a.g2, a.p_weights, eta1, eta2,
        n, m, deriv != 0, deriv2,
    )  # fmt: skip
    if reml != 0:
        j = float(n - mp)
        a.p0 /= j
        if deriv != 0:
            a.p1[:m] /= j
        if deriv > 1:
            a.p2[: m * m] /= j

    piv = pk.pivot1[:rank]
    if deriv != 0:
        for j in range(m - 1, -1, -1):
            a.beta[piv] = a.b1[rank * j : rank * j + rank]
            undrop_rows(a.beta, q, 1, pk.drop)
            a.b1[q * j : q * j + q] = a.beta[:q]
    a.beta[piv] = pk.pktz[:rank]
    undrop_rows(a.beta, q, 1, pk.drop)

    reml_penalty = 0.0
    if reml > 0:
        reml_penalty = pk.ldet_xwxs
        get_ddet_xwxps(
            a.tr_a1, a.tr_a2, pk.p, pk.k, a.sp, a.rs, rsncol, tk, tkm, n, rank, rank,
            m, 0, deriv,
        )  # fmt: skip
    if reml < 0:
        reml_penalty = ml_penalty1(
            a.tr_a1, a.tr_a2, tk, tkm, pk.nulli, pk.r, pk.q1, nind, a.sp, a.rs,
            rsncol, rank, n, m, 0, deriv, a.x, False,
        )  # fmt: skip

    pivot = np.zeros(0, dtype=np.int32)
    if fisher:
        wfp, tfkp, tfkmp = a.w, tk, tkm
    else:
        nr2 = n + enrow
        wx = np.zeros(max(nr2 * rank, 1))
        a.w[:n] = np.sqrt(a.wf[:n])
        WX = mat(wx, nr2, rank)
        WX[:n, :] = mat(a.x, n, rank) * a.w[:n, None]
        WX[n:, :] = mat(a.e, enrow, rank)
        pivot = np.zeros(max(rank, 1), dtype=np.int32)
        tau = np.zeros(max(rank, 1))
        mgcv_qr(wx, nr2, rank, pivot, tau)
        r1 = np.zeros(max(rank * rank, 1))
        get_rpqr(r1, wx, nr2, rank, rank)
        rinv(pk.p, r1, rank, rank, rank)
        qm = np.zeros(max(nr2 * rank, 1))
        for i in range(rank):
            qm[i * rank + i] = 1.0
        mgcv_pqrqy(qm, wx, tau, nr2, rank, rank, False)
        mat(pk.k, n, rank)[...] = mat(qm, nr2, rank)[:n, :]
        if deriv != 0:
            pivoter(a.rs, rank, scs, pivot, False, False)
        wfp, tfkp, tfkmp = a.wf, tfk, tfkm

    tra_deriv = 0 if reml != 0 else deriv
    a.tr_a = get_tra2(
        a.tr_a1, a.tr_a2, pk.p, pk.k, a.sp, a.rs, rsncol, tfkp, tfkmp, wfp, n, rank,
        rank, m, tra_deriv,
    )  # fmt: skip

    if not fisher:
        pivoter(pk.p, rank, rank, pivot, False, True)
    _rv_and_k(a, pk, n, q, rank)
    if reml != 0:
        a.rank_tol = reml_penalty
        a.conv_tol = bsb
    a.deriv = 0


def _rv_and_k(a: SimpleNamespace, pk: SimpleNamespace, n: int, q: int, rank: int):
    """``rV`` (``P`` unpivoted, the dropped rows restored as zeros) and
    ``K`` into ``a.rv`` and ``a.x``, and the rank into ``a.rank_est``."""
    mat(a.rv, rank, rank)[pk.pivot1[:rank], :] = mat(pk.p, rank, rank)
    undrop_rows(a.rv, q, rank, pk.drop)
    a.rv[q * rank : q * q] = 0.0
    a.x[: rank * n] = pk.k[: rank * n]
    a.x[rank * n : q * n] = 0.0
    a.rank_est = rank


def pls_fit1(y, x, w, wy, e, es, n, q, re, eta, penalty, rank_tol, use_wy):
    """gdi.c's ``pls_fit1``: the penalized least squares fit of the
    pseudodata in ``y`` (``n`` used; at least ``max(n, q)`` long) with
    weights ``w``, some possibly negative (full Newton), to the
    ``n``-by-``q`` ``x`` under the penalty square root ``e`` (``re``-by-``q``;
    its unidentifiable columns are dropped in place), the well scaled ``es``
    deciding the rank. Leaves the linear predictor in ``eta`` and the
    coefficients in the first ``q`` entries of ``y``, and returns ``(ok,
    penalty, use_wy)``: ``ok`` false, mgcv's ``n = -1``, when ``X'WX + E'E``
    is not positive definite. With ``use_wy``, or when ``R'Q'Wz`` drifts from
    ``X'Wz`` (in ``wy``), the coefficients come from ``X'Wz``."""
    rr = min(q, n)
    nr = rr + re
    nz = max(n, nr)
    z = np.zeros(max(nz, 1))
    raw, nind = _raw_weights(w, n)
    neg_w = len(nind)
    z[:n] = y[:n] * raw[:n]
    for k in nind:
        z[k] = -z[k]
    wx = np.zeros(max(n * q, 1))
    wx[: n * q] = (mat(x, n, q) * raw[:n, None]).ravel(order="F")
    tau = np.zeros(max(q, 1))
    pivot = np.zeros(max(q, 1), dtype=np.int32)
    mgcv_qr(wx, n, q, pivot, tau)
    r1 = np.zeros(max(rr * q, 1))
    get_rpqr(r1, wx, n, q, rr)
    pivoter(r1, rr, q, pivot, True, True)

    rnorm = frobenius_norm(r1, rr, q)
    enorm = frobenius_norm(es, re, q)
    r = np.zeros(max(q * nr, 1))
    R = mat(r, nr, q)
    R[:rr, :] = mat(r1, rr, q) / rnorm
    R[rr:, :] = mat(es, re, q) / enorm
    tau1 = np.zeros(max(q, 1))
    pivot1 = np.zeros(max(q, 1), dtype=np.int32)
    mgcv_qr(r, nr, q, pivot1, tau1)
    rank = min(q, nr)
    rcond = r_cond(r, nr, rank)
    while rank_tol * rcond > 1.0:
        rank -= 1
        rcond = r_cond(r, nr, rank)
    drop = sorted(int(v) for v in pivot1[rank:q])
    if drop:
        drop_cols(r1, rr, q, drop)
        drop_cols(e, re, q, drop)
    R = mat(r, nr, rank)
    R[:rr, :] = mat(r1, rr, rank)
    R[rr:, :] = mat(e, re, rank)
    mgcv_qr(r, nr, rank, pivot1, tau1)

    vt = d = _EMPTY
    if neg_w > 0:
        qm = np.zeros(max(nr * rank, 1))
        for i in range(rank):
            qm[i * nr + i] = 1.0
        mgcv_qrqy_left(qm, r, tau1, nr, rank, rank, False)
        q1 = np.zeros(max(n * rank, 1))
        mat(q1, rr, rank)[...] = mat(qm, nr, rank)[:rr, :]
        mgcv_pqrqy(q1, wx, tau, n, rr, rank, False)
        k = max(rank + 1, neg_w)
        iq = np.zeros(k * rank)
        mat(iq, k, rank)[:neg_w, :] = mat(q1, n, rank)[nind, :]
        vt = np.zeros(max(rank * rank, 1))
        d = np.zeros(rank)
        mgcv_svd_full(iq, vt, d, k, rank)
        for i in range(rank):
            di = float(d[i])
            v = _rfma(-2.0 * di, di, 1.0)
            if v < -rank_tol:
                return False, penalty, use_wy
            d[i] = 0.0 if v <= 0.0 else 1.0 / v

    work = np.zeros(max(4 * q, 1))
    if not use_wy:
        mgcv_pqrqy(z, wx, tau, n, rr, 1, True)
        z[rank:nz] = 0.0
        mgcv_qrqy_left(z, r, tau1, nr, 1, rank, True)
        z[rank:nz] = 0.0
        y[:rank] = z[:rank]
        work[q : q + rank] = z[:rank]
        if neg_w > 0:
            mgcv_mmult(work, vt, z, False, False, rank, 1, rank)
            work[:rank] *= d[:rank]
            mgcv_mmult(z, vt, work, True, False, rank, 1, rank)
            y[:rank] = z[:rank]
        mgcv_qrqy_left(z, r, tau1, nr, 1, rank, False)
        penalty = 0.0
        for v in z[rr:nr].tolist():
            penalty = _rfma(v, v, penalty)
        z[rank:n] = 0.0
        mgcv_pqrqy(z, wx, tau, n, rr, 1, False)
        eta[:n] = z[:n] / raw[:n]

    mgcv_mmult(work, x, wy, True, False, q, 1, n)
    drop_rows(work, q, 1, drop)
    pivoter(work, rank, 1, pivot1, False, False)
    if not use_wy:
        zz = zz1 = 0.0
        for i in range(rank):
            xx = dot_s8(r[nr * i : nr * i + i + 1], work[q : q + i + 1])
            xx -= work[i]
            zz1 = _rfma(xx, xx, zz1)
            zz = _rfma(work[i], work[i], zz)
        if zz1 > rank_tol * zz:
            use_wy = True
    if use_wy:
        for k in range(rank):
            xx = dot_s8(r[nr * k : nr * k + k], z[:k])
            z[k] = (work[k] - xx) / r[k + nr * k]
        if neg_w > 0:
            mgcv_mmult(work, vt, z, False, False, rank, 1, rank)
            work[:rank] *= d[:rank]
            mgcv_mmult(z, vt, work, True, False, rank, 1, rank)
        y[:rank] = z[:rank]
    for k in range(rank - 1, -1, -1):
        xx = 0.0
        for j in range(k + 1, rank):
            xx = _rfma(float(r[k + nr * j]), float(z[j]), xx)
        z[k] = (y[k] - xx) / r[k + nr * k]
    y[pivot1[:rank]] = z[:rank]
    undrop_rows(y, q, 1, drop)
    if use_wy:
        mgcv_mmult(eta, x, y, False, False, n, 1, q)
        mgcv_mmult(work, e, y, False, False, re, 1, q)
        penalty = 0.0
        for v in work[:re].tolist():
            penalty = _rfma(v, v, penalty)
    return True, penalty, use_wy


def _vec(args: dict, key: str) -> np.ndarray:
    """A fresh float64 copy of the values under ``key`` (a 1-D array or any
    sequence of numbers, a scalar as one value)."""
    v = np.array(args[key], dtype=np.float64, copy=True, ndmin=1)
    if v.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional (column-major, flattened)")
    return v


def _scalar(args: dict, key: str) -> float:
    v = _vec(args, key)
    if v.size == 0:
        raise ValueError(f"{key} is empty")
    return float(v[0])


def _check(checks) -> None:
    for name, ok in checks:
        if not ok:
            raise ValueError(f"{name} is too short")


def pls_fit1_c(args: dict) -> dict:
    """``.C(C_pls_fit1, ...)`` as gam.fit3 calls it: ``args`` maps each
    argument name (``y``, ``X``, ``w``, ``wy``, ``E``, ``Es``, ``n``, ``q``,
    ``rE``, ``eta``, ``penalty``, ``rank.tol``, ``nt``, ``use.wy``) to its
    value, matrices column-major and flattened; returns the names the call
    writes with the values it leaves."""
    n = int(_scalar(args, "n"))
    q = int(_scalar(args, "q"))
    re = int(_scalar(args, "rE"))
    y = _vec(args, "y")
    # X is only read: used as given when it starts on a 16-byte boundary,
    # as R's .C copy does (Accelerate's ddot rounds by its operands'
    # alignment), else copied into a fresh allocation that does
    x = np.asarray(args["X"], dtype=np.float64)
    if x.ndim != 1 or not x.flags.c_contiguous or x.ctypes.data % 16:
        x = _vec(args, "X")
    w = _vec(args, "w")
    wy = _vec(args, "wy")
    e = _vec(args, "E")
    es = _vec(args, "Es")
    eta = _vec(args, "eta")
    penalty = _scalar(args, "penalty")
    rank_tol = _scalar(args, "rank.tol")
    use_wy = _scalar(args, "use.wy") != 0.0
    _check(
        [
            ("y", y.size >= max(n, q)),
            ("X", x.size >= n * q),
            ("w", w.size >= n),
            ("wy", wy.size >= n),
            ("E", e.size >= re * q),
            ("Es", es.size >= re * q),
            ("eta", eta.size >= n),
        ]
    )
    with np.errstate(all="ignore"):
        ok, penalty, use_wy = pls_fit1(
            y, x, w, wy, e, es, n, q, re, eta, penalty, rank_tol, use_wy
        )
    return {
        "y": y,
        "E": e,
        "eta": eta,
        "penalty": np.array([penalty]),
        "n": np.array([float(n) if ok else -1.0]),
        "use.wy": np.array([1.0 if use_wy else 0.0]),
    }


def gdi1_c(args: dict) -> dict:
    """``.C(C_gdi1, ...)`` as gam.fit3 calls it: ``args`` maps each argument
    name gam.fit3 uses (``X``, ``E``, ``Eb``, ``rS``, ..., ``dVkk``) to its
    value, matrices column-major and flattened; returns the names the call
    writes with the values it leaves."""
    n = int(_scalar(args, "n"))
    q = int(_scalar(args, "p"))
    m = int(_scalar(args, "M"))
    a = SimpleNamespace(
        x=_vec(args, "X"), e=_vec(args, "E"), es=_vec(args, "Eb"),
        rs=_vec(args, "rS"), sp=_vec(args, "sp"), z=_vec(args, "z"),
        w=_vec(args, "w"), wf=_vec(args, "wf"), alpha=_vec(args, "alpha"),
        mu=_vec(args, "mu"), y=_vec(args, "y"), p_weights=_vec(args, "p.weights"),
        g1=_vec(args, "g1"), g2=_vec(args, "g2"), g3=_vec(args, "g3"),
        g4=_vec(args, "g4"), v0=_vec(args, "V0"), v1=_vec(args, "V1"),
        v2=_vec(args, "V2"), v3=_vec(args, "V3"), beta=_vec(args, "beta"),
        b1=_vec(args, "b1"), w1=_vec(args, "w1"), d1=_vec(args, "D1"),
        d2=_vec(args, "D2"), p0=_scalar(args, "P"), p1=_vec(args, "P1"),
        p2=_vec(args, "P2"), tr_a=_scalar(args, "trA"), tr_a1=_vec(args, "trA1"),
        tr_a2=_vec(args, "trA2"), rv=_vec(args, "rV"),
        rank_tol=_scalar(args, "rank.tol"), conv_tol=_scalar(args, "conv.tol"),
        rank_est=int(_scalar(args, "rank.est")), n=n, q=q, m=m,
        mp=int(_scalar(args, "Mp")), enrow=int(_scalar(args, "Enrow")),
        rsncol=[int(v) for v in _vec(args, "rSncol")],
        deriv=int(_scalar(args, "deriv")), reml=int(_scalar(args, "REML")),
        fisher=_scalar(args, "fisher") != 0.0, dvkk=_vec(args, "dVkk"),
    )  # fmt: skip
    _check(
        [
            ("X", a.x.size >= n * q),
            ("rV", a.rv.size >= q * q),
            ("b1", a.b1.size >= q * m),
            ("w1", a.w1.size >= n * m),
            ("D2", a.d2.size >= m * m),
            ("P2", a.p2.size >= m * m),
            ("trA2", a.tr_a2.size >= m * m),
            ("dVkk", a.dvkk.size >= m * m),
            ("rSncol", len(a.rsncol) >= m),
        ]
    )
    with np.errstate(all="ignore"):
        gdi1(a)
    return {
        "X": a.x, "E": a.e, "rS": a.rs, "w": a.w, "beta": a.beta, "b1": a.b1,
        "w1": a.w1, "D1": a.d1, "D2": a.d2, "P": np.array([a.p0]), "P1": a.p1,
        "P2": a.p2, "trA": np.array([a.tr_a]), "trA1": a.tr_a1, "trA2": a.tr_a2,
        "rV": a.rv, "rank.tol": np.array([a.rank_tol]),
        "conv.tol": np.array([a.conv_tol]),
        "rank.est": np.array([float(a.rank_est)]),
        "deriv": np.array([float(a.deriv)]), "dVkk": a.dvkk,
    }  # fmt: skip


def ift2(
    rm, vt, x, rs, beta, sp, det_th, det2_th, det3, det_th2, b1, b2, eta1, eta2, n,
    r, m, n_theta, rsncol, deriv2, neg_w, nr,
) -> None:  # fmt: skip
    """gdi.c's ``ift2``: by the implicit function theorem the first (``b1``,
    ``r`` by ``n_theta + m``, the likelihood parameters first) and second
    (``b2``, one column per ``k >= i`` pair) derivatives of the coefficients
    with respect to the extra likelihood parameters and the log smoothing
    parameters, and those of the linear predictor (``eta1``, ``eta2``)."""
    nth = n_theta
    nw = max(n, r, 1)
    work = np.zeros(nw)
    work1 = np.zeros(nw)
    db_th = np.zeros(max(r, 1))
    ntot = m + nth
    n_2d = (ntot * (1 + ntot)) // 2
    for i in range(ntot):
        if i < nth:
            mgcv_mmult(db_th, x, det_th[i * n :], True, False, r, 1, n)
            db_th[:r] *= -0.5
        else:
            mult_sk(db_th, beta, 1, i - nth, rs, rsncol, r, work)
            db_th[:r] *= -sp[i - nth]
        apply_pt(work, db_th, rm, vt, neg_w, nr, r, 1, False)
        apply_p(b1[i * r :], work, rm, vt, neg_w, nr, r, 1, False)
    mgcv_mmult(eta1, x, b1, False, False, n, ntot, r)

    if deriv2:
        pp = 0
        kk = 0
        for i in range(ntot):
            for k in range(i, ntot):
                p0 = eta1[n * i : n * i + n]
                p1 = eta1[n * k : n * k + n]
                work[:n] = (-p0 * p1) * det3[:n]
                mgcv_mmult(db_th, x, work, True, False, r, 1, n)
                if k < nth:
                    work[:n] = det2_th[n * k : n * k + n] * p0
                    mgcv_mmult(work1, x, work, True, False, r, 1, n)
                else:
                    mult_sk(work1, b1[i * r :], 1, k - nth, rs, rsncol, r, work)
                    work1[:r] *= sp[k - nth] * 2.0
                db_th[:r] -= work1[:r]
                if i < nth:
                    work[:n] = det2_th[n * i : n * i + n] * p1
                    mgcv_mmult(work1, x, work, True, False, r, 1, n)
                else:
                    mult_sk(work1, b1[k * r :], 1, i - nth, rs, rsncol, r, work)
                    work1[:r] *= sp[i - nth] * 2.0
                db_th[:r] -= work1[:r]
                if i < nth and k < nth:
                    mgcv_mmult(work, x, det_th2[n * kk :], True, False, r, 1, n)
                    db_th[:r] -= work[:r]
                    kk += 1
                elif i == k:
                    mult_sk(work1, beta, 1, i - nth, rs, rsncol, r, work)
                    db_th[:r] -= (work1[:r] * sp[i - nth]) * 2.0
                db_th[:r] *= 0.5
                apply_pt(work, db_th, rm, vt, neg_w, nr, r, 1, False)
                apply_p(b2[pp:], work, rm, vt, neg_w, nr, r, 1, False)
                pp += r
        mgcv_mmult(eta2, x, b2, False, False, n, n_2d, r)


def gdi2(a: SimpleNamespace) -> None:
    """gdi.c's ``gdi2``, one thread: at the converged coefficients of an
    extended family's penalized IRLS, the coefficients, their derivatives
    and the deviance's and penalty's with respect to the likelihood
    parameters and the log smoothing parameters, log|X'WX + S| (REML) or its
    ML version (``ldet < 0`` on entry) and their derivatives, ``rV`` and
    ``K`` (in ``x``) from the Fisher weights; on ``a``'s fields, updated in
    place."""
    n, q, m, nth, mp, enrow = a.n, a.q, a.m, a.n_theta, a.mp, a.enrow
    ml = a.ldet < 0.0
    deriv = a.deriv
    deriv2 = deriv == 2
    rsncol = list(a.rsncol)
    scs = sum(rsncol[:m])

    raw = np.zeros(max(n, 1))
    wn = a.w[:n]
    neg, pos = wn < 0.0, wn > 0.0
    raw[:n][neg] = np.sqrt(-wn[neg])
    raw[:n][pos] = np.sqrt(wn[pos])
    if not np.all(neg | pos):
        a.type_ = 1
    nind = np.flatnonzero(neg).tolist()
    neg_w = len(nind) > 0
    type1 = a.type_ == 1
    zp = a.wz if type1 else a.z
    pk = gdi_pk(
        a.x, a.e, a.es, a.rs, zp, raw, nind, n, q, mp, enrow, deriv2, scs, 1,
        a.rank_tol, type1,
    )  # fmt: skip
    a.ldet = pk.ldet_xwxs
    rank = pk.rank
    nr = pk.nr
    vtp = pk.vt if neg_w else None

    n_work = max((4 * n + 2 * q) * m + 2 * n, 5 * q, (m * (1 + m)) // 2 * n)
    work = np.zeros(max(n_work, 1))
    ntot = m + nth
    n_2d = (ntot * (1 + ntot)) // 2
    eta1 = b2 = eta2 = w2 = tk = tkm = _EMPTY
    if deriv != 0:
        eta1 = np.zeros(max(n * ntot, 1))
        if deriv2:
            b2 = np.zeros(max(rank * n_2d, 1))
            eta2 = np.zeros(max(n * n_2d, 1))
        ift2(
            pk.r, vtp, a.x, a.rs, pk.pktz, a.sp, a.det_th, a.det2_th, a.det3,
            a.det_th2, a.b1, b2, eta1, eta2, n, rank, m, nth, rsncol, deriv2, neg_w,
            nr,
        )  # fmt: skip
        if m > 0:
            mgcv_mmult(work, pk.rh, a.b1[nth * rank :], False, False, rank, m, rank)
            get_xtx(a.dvkk, work, rank, m)
        # the deviance's gradient: the likelihood parameters' entries carry
        # its direct dependence Dth
        det = a.det[:n]
        for i in range(nth):
            a.d1[i] = _inorder(
                _rfma_vec(eta1[n * i : n * i + n], det, a.dth[n * i : n * i + n])
            )
        for i in range(nth, ntot):
            a.d1[i] = dot_s8_from(eta1[n * i : n * i + n], det, 12)
        if deriv2:
            p0 = 0
            p6 = 0
            for i in range(ntot):
                for k in range(i, ntot):
                    xx = _inorder(
                        _rfma_vec(
                            a.det2[:n] * eta1[n * i : n * i + n],
                            eta1[n * k : n * k + n],
                            det * eta2[p0 : p0 + n],
                        )
                    )
                    p0 += n
                    if k < nth:
                        xx = _inorder(a.dth2[p6 : p6 + n], xx)
                        p6 += n
                    if i < nth:
                        xx = dot_s8_acc(
                            xx, a.det_th[n * i : n * i + n], eta1[n * k : n * k + n]
                        )
                    if k < nth:
                        xx = dot_s8_acc(
                            xx, a.det_th[n * k : n * k + n], eta1[n * i : n * i + n]
                        )
                    a.d2[i + ntot * k] = xx
                    a.d2[k + ntot * i] = xx

        # the weights' derivatives
        det3 = a.det3[:n]
        for i in range(ntot):
            o = n * i
            if i < nth:
                a.w1[o : o + n] = (
                    _rfma_vec(det3, eta1[o : o + n], a.det2_th[o : o + n]) * 0.5
                )
            else:
                a.w1[o : o + n] = (det3 * eta1[o : o + n]) * 0.5
        if deriv2:
            w2 = np.zeros(max(n * n_2d, 1))
            st = 0
            p1 = 0
            for i in range(ntot):
                for k in range(i, ntot):
                    e1i = eta1[n * i : n * i + n]
                    e1k = eta1[n * k : n * k + n]
                    v = _rfma_vec(a.det4[:n] * e1i, e1k, det3 * eta2[st : st + n])
                    if i < nth:
                        v = _rfma_vec(a.det3_th[n * i : n * i + n], e1k, v)
                    if k < nth:
                        v = _rfma_vec(a.det3_th[n * k : n * k + n], e1i, v)
                        if i < nth:
                            v = v + a.det2_th2[p1 : p1 + n]
                            p1 += n
                    w2[st : st + n] = v * 0.5
                    st += n
        if not type1:
            wi = 1.0 / np.abs(a.w[:n])
            tk = np.zeros(max(n * ntot, 1))
            rc_prod(tk, wi, a.w1, ntot, n)
            if deriv2:
                tkm = np.zeros(max(n * n_2d, 1))
                rc_prod(tkm, wi, w2, n_2d, n)
        else:
            tk = a.w1.copy()
            tkm = w2.copy()

    if ml:
        # R kept rank by rank
        pk.r[: rank * rank] = np.triu(mat(pk.r, nr, rank)[:rank, :]).ravel(order="F")
    a.p0 = get_bsb(
        a.p1, a.p2, a.sp, a.e, a.rs, rsncol, enrow, rank, m, nth, pk.pktz, a.b1, b2,
        deriv,
    )  # fmt: skip
    if ml:
        a.ldet = ml_penalty1(
            a.ldet1, a.ldet2, tk, tkm, pk.nulli, pk.r, pk.q1, nind, a.sp, a.rs,
            rsncol, rank, n, m, nth, deriv, a.x, type1,
        )  # fmt: skip
    else:
        get_ddet_xwxps(
            a.ldet1, a.ldet2, pk.p, pk.k, a.sp, a.rs, rsncol, tk, tkm, n, rank, rank,
            m, nth, deriv,
        )  # fmt: skip

    piv = pk.pivot1[:rank]
    if deriv != 0:
        for j in range(ntot - 1, -1, -1):
            a.beta[piv] = a.b1[rank * j : rank * j + rank]
            undrop_rows(a.beta, q, 1, pk.drop)
            a.b1[q * j : q * j + q] = a.beta[:q]
    a.beta[piv] = pk.pktz[:rank]
    undrop_rows(a.beta, q, 1, pk.drop)

    # rV and K from sqrt(wf) X augmented with E
    nr2 = n + enrow
    wx = np.zeros(max(nr2 * rank, 1))
    a.w[:n] = np.sqrt(a.wf[:n])
    WX = mat(wx, nr2, rank)
    WX[:n, :] = mat(a.x, n, rank) * a.w[:n, None]
    WX[n:, :] = mat(a.e, enrow, rank)
    pivot = np.zeros(max(rank, 1), dtype=np.int32)
    tau = np.zeros(max(rank, 1))
    mgcv_qr(wx, nr2, rank, pivot, tau)
    r1 = np.zeros(max(rank * rank, 1))
    get_rpqr(r1, wx, nr2, rank, rank)
    rinv(pk.p, r1, rank, rank, rank)
    qm = np.zeros(max(nr2 * rank, 1))
    for i in range(rank):
        qm[i * rank + i] = 1.0
    mgcv_pqrqy(qm, wx, tau, nr2, rank, rank, False)
    mat(pk.k, n, rank)[...] = mat(qm, nr2, rank)[:n, :]
    pivoter(pk.p, rank, rank, pivot, False, True)
    _rv_and_k(a, pk, n, q, rank)


def gdi2_c(args: dict) -> dict:
    """``.C(C_gdi2, ...)`` as gam.fit4 calls it: ``args`` maps each argument
    name (``X``, ``E``, ``Es``, ``rS``, ..., ``dVkk``) to its value,
    matrices column-major and flattened; returns the names the call writes
    with the values it leaves."""
    n = int(_scalar(args, "n"))
    q = int(_scalar(args, "q"))
    m = int(_scalar(args, "M"))
    n_theta = int(_scalar(args, "n.theta"))
    a = SimpleNamespace(
        x=_vec(args, "X"), e=_vec(args, "E"), es=_vec(args, "Es"),
        rs=_vec(args, "rS"), sp=_vec(args, "sp"), z=_vec(args, "z"),
        w=_vec(args, "w"), wz=_vec(args, "wz"), wf=_vec(args, "wf"),
        dth=_vec(args, "Dth"), det=_vec(args, "Det"), det2=_vec(args, "Det2"),
        dth2=_vec(args, "Dth2"), det_th=_vec(args, "Det.th"),
        det2_th=_vec(args, "Det2.th"), det3=_vec(args, "Det3"),
        det_th2=_vec(args, "Det.th2"), det4=_vec(args, "Det4"),
        det3_th=_vec(args, "Det3.th"), det2_th2=_vec(args, "Deta2.th2"),
        beta=_vec(args, "beta"), b1=_vec(args, "b1"), w1=_vec(args, "w1"),
        d1=_vec(args, "D1"), d2=_vec(args, "D2"), p0=_scalar(args, "P"),
        p1=_vec(args, "P1"), p2=_vec(args, "P2"), ldet=_scalar(args, "ldet"),
        ldet1=_vec(args, "ldet1"), ldet2=_vec(args, "ldet2"), rv=_vec(args, "rV"),
        rank_tol=_scalar(args, "rank.tol"), rank_est=int(_scalar(args, "rank.est")),
        n=n, q=q, m=m, n_theta=n_theta, mp=int(_scalar(args, "Mp")),
        enrow=int(_scalar(args, "Enrow")),
        rsncol=[int(v) for v in _vec(args, "rSncol")],
        deriv=int(_scalar(args, "deriv")), type_=int(_scalar(args, "type")),
        dvkk=_vec(args, "dVkk"),
    )  # fmt: skip
    ntot = m + n_theta
    _check(
        [
            ("X", a.x.size >= n * q),
            ("rV", a.rv.size >= q * q),
            ("b1", a.b1.size >= q * ntot),
            ("w1", a.w1.size >= n * ntot),
            ("D2", a.d2.size >= ntot * ntot),
            ("P2", a.p2.size >= ntot * ntot),
            ("ldet2", a.ldet2.size >= ntot * ntot),
            ("dVkk", a.dvkk.size >= m * m),
            ("rSncol", len(a.rsncol) >= m),
        ]
    )
    with np.errstate(all="ignore"):
        gdi2(a)
    return {
        "X": a.x, "E": a.e, "rS": a.rs, "w": a.w, "beta": a.beta, "b1": a.b1,
        "w1": a.w1, "D1": a.d1, "D2": a.d2, "P": np.array([a.p0]), "P1": a.p1,
        "P2": a.p2, "ldet": np.array([a.ldet]), "ldet1": a.ldet1, "ldet2": a.ldet2,
        "rV": a.rv, "rank.est": np.array([float(a.rank_est)]),
        "type": np.array([float(a.type_)]), "dVkk": a.dvkk,
    }  # fmt: skip
