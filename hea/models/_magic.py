"""mgcv's ``magic``: GCV/UBRE smoothing parameter selection for a penalized
least squares problem with several quadratic penalties.

``magic`` is mgcv's R wrapper (mgcv.r): starting values by ``initial.sp``,
penalty square roots by ``mroot``, linear constraints absorbed by a QR of
``t(C)``, weights, zero padding of a short design, then the C engine.
``magic_post_proc`` is ``magic.post.proc``: the covariance matrices, edf
and leverages from the engine's ``rV``.

The engine is mgcv's C ``magic`` (magic.c). With the Rust extension it is
the port in ``rust/src/magic.rs``, which reproduces R's arm64 mgcv.so bit
for bit; without it, ``_magic_c_py`` follows the same C line by line on
the LAPACK of :mod:`hea._rlinalg`.
"""

from __future__ import annotations

import math

import numpy as np

from .. import _libm as libm
from .._dispatch import rs_fn
from .._rlinalg import (
    mgcv_svd_full,
    r_chol,
    r_crossprod,
    r_eigen_sym,
    r_matprod,
    r_qr,
    r_qr_lapack,
    r_qr_qy,
    r_qr_qy_lapack,
    r_qr_R,
    r_tcrossprod,
)
from .._rsum import r_colsums, r_rowsums

_magic_c_rs = rs_fn("magic_c")


# ---------------------------------------------------------------------------
# The C engine (magic.c) without the extension.
# ---------------------------------------------------------------------------


def _mroot_c(A: np.ndarray) -> np.ndarray:
    """mgcv's C ``mroot`` at its numerical rank: ``mgcv_chol``
    (``dpstrf('U')`` at LAPACK's default tolerance, the subdiagonal zeroed),
    the factor unpivoted and cut to its first ``rank`` rows. Returns ``B``
    (``rank``-by-``n``) with ``B'B = A``."""
    n = A.shape[0]
    R, piv, rank = r_chol(A, pivot=True, tol=-1.0, r_layout=False)
    B = np.zeros((rank, n))
    B[:, np.asarray(piv) - 1] = np.triu(R)[:rank]
    return B


class _MagicFit:
    """``fit_magic``'s results: ``y1 = U1'Q1'y``, ``U1`` and ``V``
    (``q``-by-``rank``), the singular values ``d``, the coefficients ``b``
    (pivoted), the score, ``||y - Ay||^2`` (``norm``), ``n - gamma tr(A)``
    (``delta``) and the rank."""

    __slots__ = ("U1", "V", "b", "d", "delta", "norm", "rank", "score", "y1")


def _fit_magic(P: dict, sp: np.ndarray, scale: float) -> tuple[_MagicFit, float]:
    """magic.c ``fit_magic``: the fit at log smoothing parameters ``sp``
    from the SVD of the QR factor stacked on a square root of ``H + sum_i
    exp(sp_i) S_i``, truncated at ``rank_tol`` times the largest singular
    value, with the GCV score and scale estimate or the UBRE score at
    ``scale``. Returns the fit and the scale."""
    q = P["q"]
    m = len(P["Si"])
    St = P["H"].copy() if P["H"] is not None else np.zeros((q, q))
    for k in range(m):
        St = St + P["Si"][k] * math.exp(sp[k])
    root = _mroot_c(St) if (m > 0 or P["H"] is not None) else np.zeros((0, q))
    U, d, Vt = mgcv_svd_full(np.vstack([P["R"], root]))
    rank = q
    xx = d[0] * P["rank_tol"]
    while d[rank - 1] < xx:
        rank -= 1
    f = _MagicFit()
    f.V = np.ascontiguousarray(Vt[:rank].T)
    f.U1 = np.ascontiguousarray(U[:q, :rank])
    f.d = d
    f.y1 = f.U1.T @ P["y0"][:q]
    yAy = float(f.y1 @ f.y1)
    b = f.U1 @ f.y1
    yAAy = float(b @ b)
    norm = P["yy"] - 2 * yAy + yAAy
    norm = max(norm, 0.0)
    trA = float(np.sum(f.U1 * f.U1))
    f.b = f.V @ (f.y1 / d[:rank])
    n = float(P["n_score"])
    xx = n - P["gamma"] * trA
    f.delta = xx
    rss = norm + P["norm_const"]
    if P["gcv"]:
        f.score = n * rss / (xx * xx)
        scale = rss / (n - trA)
    else:
        f.score = rss / n - 2 * scale / n * xx + scale
    f.norm = norm
    f.rank = rank
    return f, scale


def _magic_gH(P: dict, f: _MagicFit, sp: np.ndarray, scale: float):
    """magic.c ``magic_gH``: the gradient and Hessian of the GCV or UBRE
    score with respect to the log smoothing parameters, from
    ``fit_magic``'s SVD."""
    rank = f.rank
    m = len(P["Si"])
    gamma, n = P["gamma"], float(P["n_score"])
    y1, d = f.y1, f.d[:rank]
    U1U1 = f.U1.T @ f.U1
    M, K, My, Ky, yK = [], [], [], [], []
    for i in range(m):
        c0 = P["cucS"][i]
        VS = (f.V.T @ P["S"][:, c0 : c0 + P["cS"][i]]) / d[:, None]
        Mi = VS.T @ U1U1
        Ki = VS @ Mi
        Mi = VS @ VS.T
        M.append(Mi)
        K.append(Ki)
        My.append(Mi.T @ y1)
        yK.append(Ki.T @ y1)
        Ky.append(Ki @ y1)
    ddelta = np.zeros(m)
    dnorm = np.zeros(m)
    d2delta = np.zeros((m, m))
    d2norm = np.zeros((m, m))
    for i in range(m):
        ddelta[i] = gamma * float(np.trace(K[i])) * math.exp(sp[i])
        for j in range(i + 1):
            xx = float(np.sum(M[j] * K[i]))
            d2delta[i, j] = d2delta[j, i] = -gamma * 2 * math.exp(sp[i] + sp[j]) * xx
        d2delta[i, i] += ddelta[i]
        dnorm[i] = 2 * math.exp(sp[i]) * float(y1 @ (My[i] - Ky[i]))
        for j in range(i + 1):
            xx = float(
                np.sum(
                    My[i] * Ky[j] + My[j] * Ky[i] - 2 * My[i] * My[j] + yK[i] * My[j]
                )
            )
            d2norm[i, j] = d2norm[j, i] = xx * 2 * math.exp(sp[i] + sp[j])
        d2norm[i, i] += dnorm[i]
    grad = np.zeros(m)
    hess = np.zeros((m, m))
    if P["gcv"]:
        norm = f.norm + P["norm_const"]
        delta = f.delta
        xx = n / (delta * delta)
        xx1 = xx * 2 * norm / delta
        x1 = -2 * xx / delta
        x2 = 3 * xx1 / delta
        for i in range(m):
            grad[i] = xx * dnorm[i] - xx1 * ddelta[i]
            for j in range(i + 1):
                hess[i, j] = hess[j, i] = (
                    x1 * (ddelta[j] * dnorm[i] + ddelta[i] * dnorm[j])
                    + xx * d2norm[i, j]
                    + x2 * ddelta[i] * ddelta[j]
                    - xx1 * d2delta[i, j]
                )
    else:
        for i in range(m):
            grad[i] = (dnorm[i] - 2 * scale * ddelta[i]) / n
            for j in range(i + 1):
                hess[i, j] = hess[j, i] = (d2norm[i, j] - 2 * scale * d2delta[i, j]) / n
    return grad, hess


def _magic_c_py(
    y,
    X,
    sp0,
    def_sp,
    S,
    H,
    L,
    lsp0,
    gamma,
    scale,
    gcv,
    max_half,
    cS,
    rank_tol,
    tol,
    norm_const,
    n_score,
):
    """magic.c ``magic`` on the LAPACK of :mod:`hea._rlinalg`: same inputs
    and outputs as the Rust ``magic_c``."""
    n, q = X.shape
    m = len(cS)
    L_exists = L is not None
    mp = len(sp0) if L_exists else m
    qrx = r_qr_lapack(X)
    pivot = np.asarray(qrx["pivot"], dtype=np.int64) - 1
    cucS = [0] * m
    for i in range(1, m):
        cucS[i] = cucS[i - 1] + cS[i - 1]
    S = np.asarray(S, dtype=float)[pivot, :]
    if H is not None:
        H = np.asarray(H, dtype=float)[np.ix_(pivot, pivot)]
    y0 = r_qr_qy_lapack(qrx, y, transpose=True)[:, 0]
    yy = float(np.dot(y, y))
    Si = [S[:, c0 : c0 + c] @ S[:, c0 : c0 + c].T for c0, c in zip(cucS, cS)]
    sp0 = np.array(sp0[:mp], dtype=float)
    def_sp = np.array(def_sp[:mp], dtype=float)
    autoinit = bool(np.any(sp0 <= 0.0))
    def_supplied = not bool(np.any(def_sp <= 0.0))
    if L_exists and not def_supplied:
        raise ValueError(
            "magic requires smoothing parameter starting values if L supplied"
        )
    R = np.triu(np.asarray(qrx["qr"])[:q, :q])
    if m > 0 and not def_supplied:
        Xms = float(np.sum(np.triu(R) ** 2)) / (n * q)
        for i in range(m):
            rSms = float(np.trace(Si[i])) / (q * cS[i])
            def_sp[i] = Xms / rSms
    sp0 = libm.log(def_sp) if autoinit else libm.log(sp0)

    def map_sp(s0):
        if L_exists:
            out = L @ s0 if mp > 0 else np.zeros(m)
            return out + lsp0
        return s0.copy()

    sp = map_sp(sp0)
    P = {
        "q": q,
        "R": R,
        "Si": Si,
        "S": S,
        "cS": list(cS),
        "cucS": cucS,
        "H": H,
        "rank_tol": rank_tol,
        "y0": y0,
        "yy": yy,
        "gamma": gamma,
        "gcv": gcv,
        "norm_const": norm_const,
        "n_score": n_score,
    }
    f, scale = _fit_magic(P, sp, scale)
    fit_call = 1
    grad = np.zeros(mp)

    def lt_grad(g1):
        return L.T @ g1 if L_exists else g1.copy()

    if mp > 0 and not autoinit:
        g1, _hess = _magic_gH(P, f, sp, scale)
        xx = 1e-4 * (1 + abs(f.score))
        grad = lt_grad(g1)
        flat = np.abs(grad) < xx
        sp0[flat] = libm.log(def_sp[flat])
        sp = map_sp(sp0)
        if np.any(flat):
            f, scale = _fit_magic(P, sp, scale)
            fit_call += 1

    min_score = f.score
    sd_step = np.zeros(mp)
    n_step = np.zeros(mp)
    use_sd = False
    it = 0
    step_fail = False
    msg = 0.0
    d_score = 1e10
    if autoinit and not def_supplied:
        for k in range(m):
            Sk = S[:, cucS[k] : cucS[k] + cS[k]]
            xx = float(np.sum((Sk.T @ f.b) ** 2))
            sd_step[k] = math.log(scale * cS[k] / xx) - sp[k]
        use_sd = True

    if mp > 0:
        converged = False
        while not converged:
            it += 1
            if it > 400:
                raise RuntimeError(
                    "magic, the gcv/ubre optimizer, failed to converge after 400 iterations."
                )
            ok = it > 1 or (autoinit and not def_supplied)
            try_ = 0
            step = sd_step if use_sd else n_step
            while ok:
                try_ += 1
                if try_ == 4 and not use_sd:
                    use_sd = True
                    step = sd_step
                nsp = sp0 + step
                sp = map_sp(nsp)
                f, scale = _fit_magic(P, sp, scale)
                fit_call += 1
                if f.score < min_score:
                    ok = False
                    d_score = min_score - f.score
                    min_score = f.score
                    sp0 = nsp
                else:
                    step /= 2
                if try_ == max_half - 1 and ok:
                    step[:] = 0.0
                if try_ == max_half:
                    ok = False
            if it > 3:
                converged = True
                if d_score > tol * (1 + min_score):
                    converged = False
                xx = math.sqrt(float(grad @ grad))
                if xx > libm.pow_op(tol, (1 / 3.0)) * (1 + abs(min_score)):
                    converged = False
                if try_ == max_half:
                    converged = True
                if converged:
                    msg = math.sqrt(xx * xx / mp)
                    if try_ == max_half:
                        step_fail = True
            sp = map_sp(sp0)
            g1, hess = _magic_gH(P, f, sp, scale)
            grad = lt_grad(g1)
            u = L.T @ hess @ L if L_exists else hess
            ev, U = r_eigen_sym(u)
            ev, U = ev[::-1].copy(), np.ascontiguousarray(U[:, ::-1])
            use_sd = bool(np.any(ev < 0.0))
            if not use_sd:
                sd = (U.T @ grad) / ev
                n_step = -(U @ sd)
                xx = float(np.max(np.abs(n_step)))
                if xx > 5.0:
                    n_step = n_step * (5.0 / xx)
            sd_step = -grad / float(np.max(np.abs(grad)))
        for k in range(mp):
            ok = 5
            xx = 2.0
            sign = 1.0 if grad[k] < 0.0 else -1.0
            while ok:
                sp0[k] += sign * xx
                sp = map_sp(sp0)
                ok -= 1
                f, scale = _fit_magic(P, sp, scale)
                if f.score < min_score:
                    min_score = f.score
                else:
                    ok = 0
                    sp0[k] += -sign * xx
        sp = map_sp(sp0)
        f, scale = _fit_magic(P, sp, scale)

    rank = f.rank
    rV = np.zeros((q, rank))
    rV[pivot, :] = f.V / f.d[:rank]
    b = np.zeros(q)
    b[pivot] = f.b
    Rout = np.zeros((q, q))
    Rout[:, pivot] = R
    info = (rank, not step_fail, not use_sd, it, fit_call)
    return b, libm.exp(sp0), f.score, scale, info, msg, rV, Rout


def _magic_c(
    y,
    X,
    sp0,
    def_sp,
    S,
    H,
    L,
    lsp0,
    gamma,
    scale,
    gcv,
    max_half,
    cS,
    rank_tol,
    tol,
    norm_const,
    n_score,
):
    """mgcv's C ``magic`` as the R wrapper's ``.C(C_magic, ...)`` call makes
    it (one thread): returns ``(b, sp, score, scale, info, rms_grad, rV,
    R)`` with ``info = (rank, converged, hess_pos_def, iter,
    score_calls)``."""
    args = (
        np.ascontiguousarray(y, dtype=float),
        np.ascontiguousarray(X, dtype=float),
        np.ascontiguousarray(sp0, dtype=float),
        np.ascontiguousarray(def_sp, dtype=float),
        np.ascontiguousarray(S, dtype=float),
        None if H is None else np.ascontiguousarray(H, dtype=float),
        None if L is None else np.ascontiguousarray(L, dtype=float),
        np.ascontiguousarray(lsp0, dtype=float),
        float(gamma),
        float(scale),
        bool(gcv),
        int(max_half),
        [int(c) for c in cS],
        float(rank_tol),
        float(tol),
        float(norm_const),
        int(n_score),
    )
    if _magic_c_rs is not None:
        b, sp, score, scale, info, rms, rV, R = _magic_c_rs(*args)
        return (
            np.asarray(b),
            np.asarray(sp),
            score,
            scale,
            info,
            rms,
            np.asarray(rV),
            np.asarray(R),
        )
    return _magic_c_py(*args)


# ---------------------------------------------------------------------------
# The R wrapper (mgcv.r).
# ---------------------------------------------------------------------------


def magic(
    y,
    X,
    sp,
    slots,
    *,
    L=None,
    lsp0=None,
    rank=None,
    H=None,
    C=None,
    w=None,
    gamma: float = 1.0,
    scale: float = 1.0,
    gcv: bool = True,
    ridge_parameter: float | None = None,
    control: dict | None = None,
    extra_rss: float = 0.0,
    n_score: int | None = None,
) -> dict:
    """mgcv ``magic(y, X, sp, S, off, L, lsp0, rank, H, C, w, gamma, scale,
    gcv, ridge.parameter, control, extra.rss, n.score)``. The penalties come
    as ``slots`` (``S`` and the 0-based first column ``col_start``, mgcv's
    ``S`` and ``off``); ``rank`` is mgcv's ``rank`` (``None``: square roots
    by ``mroot(S, method = "svd")``, else ``mroot(S, rank, "chol")``); ``C``
    is a constraint matrix (rows are constraints) and ``w`` the square root
    weights (a vector) or a weight matrix. Any non-positive ``sp`` entry
    starts from ``initial.sp``.

    Returns R's list: ``b``, ``scale``, ``score``, ``sp``, ``sp_full``
    (``exp(L %*% log(sp))``), ``R`` (the unpivoted QR factor of the weighted
    design), ``rV`` (``cov(b) = rV rV' scale``) and ``gcv_info``
    (``full_rank``, ``rank``, ``fully_converged``, ``hess_pos_def``,
    ``iter``, ``score_calls``, ``rms_grad``)."""
    from .gam import _initial_sp, _lm_on_L, _mroot

    ctrl = {
        "tol": 1e-6,
        "step_half": 25,
        "rank_tol": libm.pow_op(float(np.finfo(float).eps), 0.5),
    }
    if control:
        ctrl.update({k: v for k, v in control.items() if v is not None})
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if n_score is None:
        n_score = y.size
    sp = np.asarray(sp, dtype=float).ravel()
    n_p = len(slots)
    n_b = X.shape[1]
    def_sp = _initial_sp(X, slots) if n_p else sp.copy()

    if L is not None:
        L = np.asarray(L, dtype=float)
        if L.ndim != 2:
            raise ValueError("L must be a matrix.")
        if L.shape[0] < L.shape[1]:
            raise ValueError("L must have at least as many rows as columns.")
        if L.shape[0] != n_p or L.shape[1] != sp.size:
            raise ValueError("L has inconsistent dimensions.")
        if lsp0 is None:
            lsp0 = np.zeros(L.shape[0])
        if L.shape[1]:
            def_sp = libm.exp(_lm_on_L(libm.log(def_sp), L, lsp0))

    S = []
    for i, slot in enumerate(slots):
        B = (
            _mroot(slot.S, method="svd")
            if rank is None
            else _mroot(slot.S, rank=int(rank[i]), method="chol")
        )
        Ri = np.zeros((n_b, B.shape[1]))
        Ri[slot.col_start : slot.col_start + B.shape[0], :] = B
        S.append(Ri)

    n_con = 0
    ns_qr = None
    if C is not None:
        C = np.asarray(C, dtype=float)
        n_con = C.shape[0]
        if n_con > 0:
            # The null space of C is the last n.b - n.con columns of Q from
            # qr(t(C)); with no rows Q is the identity.
            ns_qr = r_qr(np.ascontiguousarray(C.T))
            X = np.ascontiguousarray(
                r_qr_qy(ns_qr, np.ascontiguousarray(X.T), transpose=True).T[
                    :, n_con:n_b
                ]
            )
            for i in range(n_p):
                Si = r_qr_qy(ns_qr, S[i], transpose=True)[n_con:n_b, :]
                if Si.shape[1] > Si.shape[0]:
                    Si = np.ascontiguousarray(
                        r_qr_R(r_qr(np.ascontiguousarray(Si.T))).T
                    )
                S[i] = Si
            if H is not None:
                H = r_qr_qy(ns_qr, np.asarray(H, dtype=float), transpose=True)[
                    n_con:n_b, :
                ]
                H = np.ascontiguousarray(
                    r_qr_qy(ns_qr, np.ascontiguousarray(H.T), transpose=True)[
                        n_con:n_b, :
                    ].T
                )
        full_rank = n_b - n_con
    else:
        full_rank = n_b

    if w is not None:
        w = np.asarray(w, dtype=float)
        if w.ndim == 2:
            if w.shape[0] != w.shape[1] or w.shape[1] != X.shape[0]:
                raise ValueError("dimensions of supplied w wrong.")
            y = r_matprod(w, y.reshape(-1, 1))[:, 0]
            X = r_matprod(w, X)
        else:
            if y.size != w.size:
                raise ValueError("w different length from y!")
            y = y * w
            X = w.ravel()[:, None] * X

    cS = [Si.shape[1] for Si in S]
    q = X.shape[1]
    Sall = np.concatenate(S, axis=1) if n_p else np.zeros((q, 0))
    rdef = X.shape[1] - X.shape[0]
    if rdef > 0:
        X = np.vstack([X, np.zeros((rdef, X.shape[1]))])
        y = np.concatenate([y, np.zeros(rdef)])

    if ridge_parameter is not None and ridge_parameter > 0:
        H = (
            np.diag(np.full(q, float(ridge_parameter)))
            if H is None
            else H + np.diag(np.full(q, float(ridge_parameter)))
        )
    L_full = np.eye(n_p) if L is None else L
    if lsp0 is None:
        lsp0 = np.zeros(L_full.shape[0])
    lsp0 = np.asarray(lsp0, dtype=float)

    b, sp_out, score, scale_out, info, rms, rV, R = _magic_c(
        y,
        X,
        sp,
        def_sp,
        Sall,
        H,
        L,
        lsp0,
        gamma,
        scale,
        gcv,
        ctrl["step_half"],
        cS,
        ctrl["rank_tol"],
        ctrl["tol"],
        extra_rss,
        n_score,
    )
    sp_full = libm.exp(r_matprod(L_full, libm.log(sp_out).reshape(-1, 1))[:, 0])
    rank_out = int(info[0])
    res = {
        "b": np.asarray(b, dtype=float),
        "scale": float(scale_out),
        "score": float(score),
        "sp": np.asarray(sp_out, dtype=float),
        "sp_full": sp_full,
        "R": np.asarray(R, dtype=float),
        "rV": np.asarray(rV, dtype=float)[:, :rank_out],
        "gcv_info": {
            "full_rank": full_rank,
            "rank": rank_out,
            "fully_converged": bool(info[1]),
            "hess_pos_def": bool(info[2]),
            "iter": int(info[3]),
            "score_calls": int(info[4]),
            "rms_grad": float(rms),
        },
    }
    if ns_qr is not None:
        bb = np.concatenate([np.zeros(n_con), res["b"]])
        res["b"] = r_qr_qy(ns_qr, bb)
        bm = np.zeros((n_b, res["rV"].shape[1]))
        bm[n_con:n_b, :] = res["rV"]
        res["rV"] = r_qr_qy(ns_qr, bm)
    return res


def magic_post_proc(X, obj: dict, w=None) -> dict:
    """mgcv ``magic.post.proc(X, object, w)``: from ``magic``'s ``rV``, the
    Bayesian covariance ``Vb = rV rV' scale``, the frequentist ``Ve``, the
    per-coefficient ``edf`` (``diag(V X'WX)``), ``edf1 = 2 edf -
    diag(F F)`` with ``F = V X'WX``, and the leverages ``hat``. ``w`` is a
    weight vector (``W = diag(w)``) or a matrix square root."""
    X = np.asarray(X, dtype=float)
    V = r_tcrossprod(np.asarray(obj["rV"], dtype=float))
    if w is not None:
        w = np.asarray(w, dtype=float)
        if w.ndim == 2:
            WX = X = r_matprod(w, X)
        else:
            WX = w.ravel()[:, None] * X
    else:
        WX = X
    M = r_matprod(WX, V)
    XWX = r_crossprod(np.asarray(obj["R"], dtype=float))
    F = Ve = r_matprod(V, XWX)
    edf1 = r_rowsums(np.ascontiguousarray(Ve.T) * Ve)
    Ve = r_matprod(Ve, V) * obj["scale"]
    B = X * M
    hat = r_rowsums(B)
    edf = r_colsums(B)
    Vb = V * obj["scale"]
    return {"Ve": Ve, "Vb": Vb, "hat": hat, "edf": edf, "edf1": 2 * edf - edf1, "F": F}
