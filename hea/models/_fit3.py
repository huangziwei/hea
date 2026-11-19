"""mgcv's ``gam.fit3`` (gam.fit3.r:67-857) for exponential families: one
evaluation of a smoothness selection score. The penalized IRLS runs in
``gam.reparam``'s stable basis (``x T``, ``T = U1 blockdiag(Qs, I)``), each
step by mgcv's ``pls_fit1``; at convergence ``gdi1`` gives the coefficient
derivatives and the score's ingredients, which are assembled into the REML,
ML, P-REML, GCV, UBRE or GACV score and its first two derivatives as
gam.fit3 does. Matrix products are R's (``r_matprod``: the BLAS call R's
``matprod`` makes) and sums R's (``r_sum``, ``r_mean``), so each value is
mgcv's."""

from __future__ import annotations

import math

import numpy as np

from .._dispatch import rs_fn
from .._rlinalg import r_matprod, r_matprod_safe
from .._rsum import r_mean, r_sum

_pls_fit1 = rs_fn("pls_fit1_c")
_gdi1 = rs_fn("gdi1_c")
_EPS = float(np.finfo(float).eps)


def _mv(a: np.ndarray, v: np.ndarray, a_finite: bool = False) -> np.ndarray:
    """``a %*% v`` for a vector ``v`` (``a_finite``: ``a`` is known to pass
    matprod's NaN/Inf screen, :func:`r_matprod_safe`)."""
    return r_matprod(a, np.asarray(v, dtype=float).reshape(-1, 1), x_finite=a_finite)[
        :, 0
    ]


def _quad(v: np.ndarray, s: np.ndarray) -> float:
    """``t(v) %*% s %*% v``."""
    row = np.asarray(v, dtype=float).reshape(1, -1)
    return float(r_matprod(r_matprod(row, s), row.T.copy())[0, 0])


def _dev(family, y, mu, weights) -> float:
    """``sum(dev.resids(y, mu, weights))``; a mean outside the family's
    range gives NaN, as in R, which the caller's step halving handles."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return r_sum(family.dev_resids(y, mu, weights))


def _f(v) -> np.ndarray:
    return np.ascontiguousarray(np.ravel(np.asarray(v, dtype=float), order="F"))


def _rows(a: np.ndarray, good: np.ndarray) -> np.ndarray:
    """``a[good, ]``, column-major so that :func:`_f` passes it on without
    a copy; ``a`` itself when every row is kept."""
    if good.all():
        return a
    return np.asfortranarray(a[good])


def gam_fit3(
    x,
    y,
    sp,
    Eb,
    UrS,
    *,
    family,
    weights=None,
    start=None,
    etastart=None,
    mustart=None,
    offset=None,
    U1=None,
    Mp=-1,
    epsilon=1e-7,
    maxit=200,
    scale_est="fletcher",
    intercept=True,
    deriv=2,
    gamma=1.0,
    scale=1.0,
    score_type="REML",
    null_coef=None,
    pearson_extra=0.0,
    dev_extra=0.0,
    n_true=-1.0,
    binom_n=None,
    x_finite=False,
    cache=None,
) -> dict:
    """mgcv ``gam.fit3`` at the log smoothing parameters ``sp`` (with log
    scale last when ``scale <= 0`` and the score is REML/ML/EFS). ``Eb``,
    ``UrS``, ``U1`` and ``Mp`` are gam.setup's (totalPenaltySpace and the
    penalty roots in its range space, the fixed penalty's root last).
    Returns gam.fit3's list: the coefficients (original basis), fit, score
    and derivatives, ``rV``, ``db_drho``, ``K`` and the rest.

    ``x_finite`` says ``x`` is known to hold no NaN or infinity. ``cache``,
    a dict kept across the calls of one fit with the same ``y``,
    ``weights``, ``offset`` and family, holds what does not depend on
    ``sp`` (``family$ls`` at a given scale, ``sum(weights)``, the null
    deviance) so it is computed once."""
    from .gam import _gam_reparam

    if _pls_fit1 is None or _gdi1 is None:
        raise RuntimeError("gam_fit3 needs the compiled pls_fit1 and gdi1")
    link = family.link
    warn: list[str] = []
    fisher = bool(family.is_canonical)
    scale = float(scale)
    scale_known = scale > 0
    sp = np.asarray(sp, dtype=float).ravel().copy()
    if not scale_known and score_type in ("REML", "ML", "EFS"):
        scale = math.exp(sp[-1])
        sp = sp[:-1]
    if deriv not in (0, 1, 2):
        raise ValueError("unsupported order of differentiation requested of gam.fit3")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nSp = sp.size
    deriv_sp = 0 if nSp == 0 else deriv
    rank_tol = _EPS * 100
    q = x.shape[1]
    nobs = y.shape[0]
    weights = np.ones(nobs) if weights is None else np.asarray(weights, dtype=float)
    offset = np.zeros(nobs) if offset is None else np.asarray(offset, dtype=float)
    null_coef = np.zeros(q) if null_coef is None else np.asarray(null_coef, dtype=float)
    U1 = np.eye(q) if U1 is None else np.asarray(U1, dtype=float)
    Mp = int(Mp)

    if len(UrS):
        grderiv = (
            1
            if score_type == "EFS"
            else deriv * int(score_type in ("REML", "ML", "P-REML", "P-ML"))
        )
        rp = _gam_reparam(list(UrS), sp, grderiv, fixed_penalty=len(UrS) > nSp)
        Qs = rp["Qs"]
        nq = Qs.shape[1]
        T = np.eye(q)
        T[:nq, :nq] = Qs
        T = r_matprod(U1, T)
        null_coef = _mv(np.ascontiguousarray(T.T), null_coef)
        if start is not None:
            start = _mv(np.ascontiguousarray(T.T), start)
        x = r_matprod(x, T, x_finite=x_finite)
        rS = [np.vstack([r, np.zeros((Mp, r.shape[1]))]) for r in rp["rS"]]
        Eb = r_matprod(np.asarray(Eb, dtype=float), T)
        rows_E = q - Mp
        E0 = rp["E"]
        Sr = np.hstack([E0, np.zeros((E0.shape[0], Mp))])
        S0 = rp["S"]
        St = np.zeros((q, q))
        St[: S0.shape[0], : S0.shape[1]] = S0
        det, det1, det2 = rp["det"], rp["det1"], rp["det2"]
        fixed_penalty = len(UrS) > nSp
    else:
        grderiv = 0
        T = np.eye(q)
        St = np.zeros((q, q))
        rS = []
        rows_E = 0
        Eb = np.zeros((0, q))
        Sr = np.zeros((0, q))
        det, det1, det2 = 0.0, np.zeros(0), np.zeros((0, 0))
        fixed_penalty = False
    coef = np.zeros(q)
    x_fin = r_matprod_safe(x)

    if score_type == "EFS":
        score_type = "REML"
        deriv = 0

    conv = False
    n_true = float(n_true)
    if n_true <= 0:
        n_true = float(nobs)

    if mustart is None:
        mustart = (
            family.gam_initialize(y, weights)
            if binom_n is None
            else family.gam_initialize(y, weights, n=binom_n)
        )
    else:
        mustart = np.asarray(mustart, dtype=float)
    strictly_additive = family.name == "gaussian" and link.name == "identity"

    out_P = None
    trA = None
    GCV = GCV1 = GCV2 = GACV = GACV1 = GACV2 = UBRE = UBRE1 = UBRE2 = None
    REML = REML1 = REML2 = None
    reml_Dp = ml_pen = None

    if etastart is not None:
        eta = np.asarray(etastart, dtype=float)
    elif start is not None:
        if start.size != q:
            raise ValueError(f"Length of start should equal {q}")
        eta = offset + (x[:, 0] * start[0] if q == 1 else _mv(x, start, x_fin))
    else:
        eta = link.link(mustart)
    mu = link.linkinv(eta)

    boundary = False
    coefold = null_coef
    etaold = null_eta = _mv(x, null_coef, x_fin) + offset
    old_pdev = _dev(family, y, link.linkinv(null_eta), weights) + _quad(null_coef, St)
    ii = 0
    while not (family.validmu(mu) and link.valideta(eta)):
        ii += 1
        if ii > 20:
            raise ValueError("Can't find valid starting values: please specify some")
        if start is not None:
            start = start * 0.9 + coefold * 0.1
        eta = 0.9 * eta + 0.1 * etaold
        mu = link.linkinv(eta)
    zg = np.zeros(max(x.shape))
    good = weights > 0
    z = w = None
    for it in range(1, int(maxit) + 1):
        good = weights > 0
        var_val = family.variance(mu)
        varmu = var_val[good]
        if np.any(np.isnan(varmu)):
            raise FloatingPointError("NAs in V(mu)")
        if np.any(varmu == 0):
            raise FloatingPointError("0s in V(mu)")
        mu_eta_val = link.mu_eta(eta)
        if np.any(np.isnan(mu_eta_val[good])):
            raise FloatingPointError("NAs in d(mu)/d(eta)")
        good = (weights > 0) & (mu_eta_val != 0)
        if not np.any(good):
            conv = False
            warn.append(f"gam.fit3 no observations informative at iteration {it}")
            break
        mevg = mu_eta_val[good]
        mug = mu[good]
        yg = y[good]
        weg = weights[good]
        var_mug = var_val[good]
        if fisher:
            z = (eta - offset)[good] + (yg - mug) / mevg
            w = (weg * (mevg * mevg)) / var_mug
        else:
            c = yg - mug
            alpha = 1 + c * (family.dvar(mug) / var_mug + link.d2link(mug) * mevg)
            alpha[alpha == 0] = _EPS
            z = (eta - offset)[good] + (yg - mug) / (mevg * alpha)
            w = ((weg * alpha) * (mevg * mevg)) / var_mug
        ng = int(good.sum())
        zg[:ng] = z
        xg = _rows(x, good)
        oo = _pls(zg, xg, w, z, Sr, Eb, ng, q, rows_E, rank_tol)
        if not fisher and oo["n"] < 0:
            z = (eta - offset)[good] + (yg - mug) / mevg
            w = (weg * (mevg * mevg)) / var_mug
            ng = int(good.sum())
            zg[:ng] = z
            oo = _pls(zg, xg, w, z, Sr, Eb, ng, q, rows_E, rank_tol)
        start = oo["y"][:q].copy()
        penalty = float(oo["penalty"][0])
        eta = _mv(x, start, x_fin)
        if np.any(~np.isfinite(start)):
            conv = False
            warn.append(f"gam.fit3 non-finite coefficients at iteration {it}")
            break
        eta = eta + offset
        mu = link.linkinv(eta)
        dev = _dev(family, y, mu, weights)
        boundary = False
        if not math.isfinite(dev):
            warn.append("gam.fit3 step size truncated due to divergence")
            ii = 1
            while not math.isfinite(dev):
                if ii > maxit:
                    raise FloatingPointError("inner loop 1; can't correct step size")
                ii += 1
                start = (start + coefold) / 2
                eta = (eta + etaold) / 2
                mu = link.linkinv(eta)
                dev = _dev(family, y, mu, weights)
            boundary = True
            penalty = _quad(start, St)
        if not (link.valideta(eta) and family.validmu(mu)):
            warn.append("gam.fit3 step size truncated: out of bounds")
            ii = 1
            while not (link.valideta(eta) and family.validmu(mu)):
                if ii > maxit:
                    raise FloatingPointError("inner loop 2; can't correct step size")
                ii += 1
                start = (start + coefold) / 2
                eta = (eta + etaold) / 2
                mu = link.linkinv(eta)
            boundary = True
            penalty = _quad(start, St)
            dev = _dev(family, y, mu, weights)
        pdev = dev + penalty
        div_thresh = 10 * (0.1 + abs(old_pdev)) * math.sqrt(_EPS)
        if pdev - old_pdev > div_thresh:
            ii = 1
            if it == 1:
                etaold = null_eta
                coefold = null_coef
            while pdev - old_pdev > div_thresh:
                if ii > 100:
                    raise FloatingPointError("inner loop 3; can't correct step size")
                ii += 1
                start = (start + coefold) / 2
                eta = (eta + etaold) / 2
                mu = link.linkinv(eta)
                dev = _dev(family, y, mu, weights)
                pdev = dev + _quad(start, St)
        if strictly_additive:
            conv = True
            coef = start
            break
        if abs(pdev - old_pdev) < epsilon * (abs(scale) + abs(pdev)):
            xs = _mv(x, start, x_fin)
            grad = 2 * _mv(
                np.asfortranarray(_rows(x, good).T), w * (xs[good] - z), x_fin
            ) + 2 * _mv(St, start)
            if np.max(np.abs(grad)) > epsilon * (abs(pdev) + abs(scale)):
                old_pdev = pdev
                coef = coefold = start
                etaold = eta
            else:
                conv = True
                coef = start
                etaold = eta
                break
        else:
            old_pdev = pdev
            coef = coefold = start
            etaold = eta

    wdr = family.dev_resids(y, mu, weights)
    dev = r_sum(wdr)

    good = weights > 0
    var_val = family.variance(mu)
    varmu = var_val[good]
    if np.any(np.isnan(varmu)):
        raise FloatingPointError("NAs in V(mu)")
    if np.any(varmu == 0):
        raise FloatingPointError("0s in V(mu)")
    mu_eta_val = link.mu_eta(eta)
    if np.any(np.isnan(mu_eta_val[good])):
        raise FloatingPointError("NAs in d(mu)/d(eta)")
    good = (weights > 0) & (mu_eta_val != 0)
    mevg = mu_eta_val[good]
    mug = mu[good]
    yg = y[good]
    weg = weights[good]
    etag = eta[good]
    var_mug = var_val[good]
    if fisher:
        z = (eta - offset)[good] + (yg - mug) / mevg
        w = (weg * (mevg * mevg)) / var_mug
        alpha = wf = np.zeros(1)
    else:
        c = yg - mug
        alpha = 1 + c * (family.dvar(mug) / var_mug + link.d2link(mug) * mevg)
        alpha[alpha == 0] = _EPS
        z = (eta - offset)[good] + (yg - mug) / (mevg * alpha)
        wf = (weg * (mevg * mevg)) / var_mug
        w = wf * alpha
    g1 = 1 / mevg
    g2 = link.d2link(mug)
    g3 = link.d3link(mug)
    V = family.variance(mug)
    V1 = family.dvar(mug)
    V2 = family.d2var(mug)
    if fisher:
        g4 = np.zeros_like(g1)
        V3 = np.zeros_like(g1)
    else:
        g4 = link.d4link(mug)
        V3 = family.d3var(mug)
    g2 = g2 / g1
    g3 = g3 / g1
    g4 = g4 / g1
    V1 = V1 / V
    V2 = V2 / V
    V3 = V3 / V

    out_P1 = out_D1 = np.zeros(nSp)
    out_P2 = out_D2 = np.zeros((nSp, nSp))
    trA1 = np.zeros(nSp)
    trA2 = np.zeros((nSp, nSp))
    reml_flag = 0
    reml_ind = 0
    if score_type in ("REML", "P-REML", "NCV"):
        reml_flag, reml_ind = 1, 1
    elif score_type in ("ML", "P-ML"):
        reml_flag, reml_ind = -1, 0
    if score_type == "NCV":
        raise NotImplementedError("gam.fit3's NCV score is not ported")
    REML = float(reml_flag)
    rSncol = [r.shape[1] for r in rS] if reml_flag == 0 else [u.shape[1] for u in UrS]
    nz = z.size
    oo = _gdi1(
        {
            "X": _f(_rows(x, good)),
            "E": _f(Sr),
            "Eb": _f(Eb),
            "rS": np.concatenate([_f(r) for r in rS]) if rS else np.zeros(1),
            "sp": np.exp(sp) if nSp else np.zeros(1),
            "z": _f(z),
            "w": _f(w),
            "wf": _f(wf),
            "alpha": _f(alpha),
            "mu": _f(mug),
            "eta": _f(etag),
            "y": _f(yg),
            "p.weights": _f(weg),
            "g1": _f(g1),
            "g2": _f(g2),
            "g3": _f(g3),
            "g4": _f(g4),
            "V0": _f(V),
            "V1": _f(V1),
            "V2": _f(V2),
            "V3": _f(V3),
            "beta": _f(coef),
            "b1": np.zeros(max(nSp * q, 1)),
            "w1": np.zeros(max(nSp * nz, 1)),
            "D1": np.zeros(max(nSp, 1)),
            "D2": np.zeros(max(nSp * nSp, 1)),
            "P": np.zeros(1),
            "P1": np.zeros(max(nSp, 1)),
            "P2": np.zeros(max(nSp * nSp, 1)),
            "trA": np.zeros(1),
            "trA1": np.zeros(max(nSp, 1)),
            "trA2": np.zeros(max(nSp * nSp, 1)),
            "rV": np.zeros(q * q),
            "rank.tol": np.array([rank_tol]),
            "conv.tol": np.array([float(epsilon)]),
            "rank.est": np.array([1.0]),
            "n": np.array([float(nz)]),
            "p": np.array([float(q)]),
            "M": np.array([float(nSp)]),
            "Mp": np.array([float(Mp)]),
            "Enrow": np.array([float(rows_E)]),
            "rSncol": np.array(rSncol if rSncol else [0], dtype=float),
            "deriv": np.array([float(deriv_sp)]),
            "REML": np.array([float(reml_flag)]),
            "fisher": np.array([1.0 if fisher else 0.0]),
            "fixed.penalty": np.array([1.0 if fixed_penalty else 0.0]),
            "nthreads": np.array([1.0]),
            "dVkk": np.zeros(max(nSp * nSp, 1)),
        }
    )
    db_drho = (
        np.asarray(oo["b1"])[: q * nSp].reshape((q, nSp), order="F") if deriv else None
    )
    dw_drho = (
        np.asarray(oo["w1"])[: nz * nSp].reshape((nz, nSp), order="F")
        if deriv
        else None
    )
    rV = np.asarray(oo["rV"]).reshape((q, q), order="F")
    K_good = np.asarray(oo["X"])[: nz * q].reshape((nz, q), order="F")
    if nz == nobs:
        Kmat = K_good
    else:
        Kmat = np.zeros((nobs, q))
        Kmat[good] = K_good
    coef = np.asarray(oo["beta"])[:q].copy()
    eta = _mv(x, coef, x_fin) + offset
    mu = link.linkinv(eta)
    if not (family.validmu(mu) and link.valideta(eta)):
        coef = start
        eta = etaold
        mu = link.linkinv(eta)
    trA = float(oo["trA"][0])
    oo_D1 = np.asarray(oo["D1"])[:nSp]
    oo_D2 = np.asarray(oo["D2"])[: nSp * nSp].reshape((nSp, nSp), order="F")
    oo_trA1 = np.asarray(oo["trA1"])[:nSp]
    oo_trA2 = np.asarray(oo["trA2"])[: nSp * nSp].reshape((nSp, nSp), order="F")
    oo_P = float(oo["P"][0])
    oo_P1 = np.asarray(oo["P1"])[:nSp]
    oo_P2 = np.asarray(oo["P2"])[: nSp * nSp].reshape((nSp, nSp), order="F")
    oo_rank_tol = float(oo["rank.tol"][0])
    oo_conv_tol = float(oo["conv.tol"][0])

    if scale_est in ("pearson", "fletcher"):
        pearson = r_sum(weights * ((y - mu) * (y - mu)) / family.variance(mu))
        scale_est_v = (pearson + dev_extra) / (n_true - trA)
        if scale_est == "fletcher":
            s_bar = max(-0.9, r_mean(family.dvar(mu) * (y - mu) / family.variance(mu)))
            if math.isfinite(s_bar):
                scale_est_v = scale_est_v / (1 + s_bar)
    else:
        scale_est_v = (dev + dev_extra) / (n_true - trA)

    reml_scale = math.nan
    if score_type in ("REML", "ML"):
        ls = _ls(family, y, weights, scale, binom_n, cache) * n_true / nobs
        Dp = dev + oo_conv_tol + dev_extra
        REML = (
            (Dp / (2 * scale) - ls[0]) / gamma
            + oo_rank_tol / 2
            - det / 2
            - reml_ind * (Mp / 2 * (math.log(2 * math.pi * scale) - math.log(gamma)))
        )
        reml_Dp = Dp / (2 * scale)
        ml_pen = (
            oo_conv_tol / (2 * scale * gamma)
            + oo_rank_tol / 2
            - det / 2
            - reml_ind * (Mp / 2 * (math.log(2 * math.pi * scale) - math.log(gamma)))
        )
        if deriv:
            REML1 = oo_D1 / (2 * scale * gamma) + oo_trA1 / 2 - det1 / 2
            if deriv == 2:
                REML2 = (oo_D2 / (scale * gamma) + oo_trA2 - det2) / 2
                if not np.all(np.isfinite(REML2)):
                    raise FloatingPointError(
                        "Non finite derivatives. Try decreasing fit tolerance! "
                        "See `epsilon' in `gam.contol'"
                    )
        if not scale_known and deriv:
            dlr_dlphi = (-Dp / (2 * scale) - ls[1] * scale) / gamma - Mp / 2 * reml_ind
            d2lr_d2lphi = (
                Dp / (2 * scale) - ls[2] * (scale * scale) - ls[1] * scale
            ) / gamma
            d2lr_dspphi = -oo_D1 / (2 * scale * gamma)
            REML1 = np.append(REML1, dlr_dlphi)
            if deriv == 2:
                REML2 = np.vstack([REML2, d2lr_dspphi])
                REML2 = np.column_stack([REML2, np.append(d2lr_dspphi, d2lr_d2lphi)])
        reml_scale = scale
    elif score_type in ("P-REML", "P-ML"):
        reml_scale = phi = (oo_P * (nobs - Mp) + pearson_extra) / (n_true - Mp)
        oo_P1 = oo_P1 * (nobs - Mp) / (n_true - Mp)
        oo_P2 = oo_P2 * (nobs - Mp) / (n_true - Mp)
        ls = _ls(family, y, weights, phi, binom_n, cache) * n_true / nobs
        Dp = dev + oo_conv_tol + dev_extra
        K = oo_rank_tol / 2 - det / 2
        REML = (
            (Dp / (2 * phi) - ls[0])
            + K
            - Mp / 2 * (math.log(2 * math.pi * phi)) * reml_ind
        )
        reml_Dp = Dp / (2 * phi)
        if deriv:
            phi1 = oo_P1
            Dp1 = oo_D1
            K1 = oo_trA1 / 2 - det1 / 2
            REML1 = (
                Dp1 / (2 * phi)
                - phi1 * (Dp / (2 * phi * phi) + Mp / (2 * phi) * reml_ind + ls[1])
                + K1
            )
            if deriv == 2:
                phi2 = oo_P2
                Dp2 = oo_D2
                K2 = oo_trA2 / 2 - det2 / 2
                REML2 = (
                    Dp2 / (2 * phi)
                    - (np.outer(Dp1, phi1) + np.outer(phi1, Dp1)) / (2 * phi * phi)
                    + (Dp / (phi * phi * phi) - ls[2] + Mp / (2 * phi * phi) * reml_ind)
                    * np.outer(phi1, phi1)
                    - (Dp / (2 * phi * phi) + ls[1] + Mp / (2 * phi) * reml_ind) * phi2
                    + K2
                )
    else:
        out_P = P = oo_P
        delta = nobs - gamma * trA
        delta_2 = delta * delta
        GCV = nobs * dev / delta_2
        GACV = dev / nobs + P * 2 * gamma * trA / (delta * nobs)
        UBRE = dev / nobs - 2 * delta * scale / nobs + scale
        if deriv:
            trA1 = oo_trA1
            out_D1 = D1 = oo_D1
            out_P1 = P1 = oo_P1
            if not (
                np.all(np.isfinite(D1))
                and np.all(np.isfinite(P1))
                and np.all(np.isfinite(trA1))
            ):
                raise FloatingPointError(
                    "Non-finite derivatives. Try decreasing fit tolerance! "
                    "See `epsilon' in `gam.contol'"
                )
            delta_3 = delta * delta_2
            GCV1 = nobs * D1 / delta_2 + 2 * nobs * dev * trA1 * gamma / delta_3
            GACV1 = (
                D1 / nobs
                + 2 * P / delta_2 * trA1
                + 2 * gamma * trA * P1 / (delta * nobs)
            )
            UBRE1 = D1 / nobs + gamma * trA1 * 2 * scale / nobs
            if deriv == 2:
                trA2 = oo_trA2
                out_D2 = D2 = oo_D2
                out_P2 = P2 = oo_P2
                if not (
                    np.all(np.isfinite(D2))
                    and np.all(np.isfinite(P2))
                    and np.all(np.isfinite(trA2))
                ):
                    raise FloatingPointError(
                        "Non-finite derivatives. Try decreasing fit tolerance! "
                        "See `epsilon' in `gam.contol'"
                    )
                GCV2 = np.outer(trA1, D1)
                GCV2 = (
                    (GCV2 + GCV2.T) * gamma * 2 * nobs / delta_3
                    + 6
                    * nobs
                    * dev
                    * np.outer(trA1, trA1)
                    * gamma
                    * gamma
                    / (delta_2 * delta_2)
                    + nobs * D2 / delta_2
                    + 2 * nobs * dev * gamma * trA2 / delta_3
                )
                GACV2 = (
                    D2 / nobs
                    + np.outer(trA1, trA1) * 4 * P / delta_3
                    + 2 * P * trA2 / delta_2
                    + 2 * np.outer(trA1, P1) / delta_2
                    + 2
                    * np.outer(P1, trA1)
                    * (1 / (delta * nobs) + trA / (nobs * delta_2))
                    + 2 * trA * P2 / (delta * nobs)
                )
                GACV2 = (GACV2 + GACV2.T) * 0.5
                UBRE2 = D2 / nobs + 2 * gamma * trA2 * scale / nobs

    if not conv:
        warn.append("gam.fit3 algorithm did not converge")
    if boundary:
        warn.append("gam.fit3 algorithm stopped at boundary value")
    eps10 = 10 * _EPS
    if family.name == "binomial" and (np.any(mu > 1 - eps10) or np.any(mu < eps10)):
        warn.append("gam.fit3 fitted probabilities numerically 0 or 1 occurred")
    if family.name == "poisson" and np.any(mu < eps10):
        warn.append("gam.fit3 fitted rates numerically 0 occurred")

    residuals = np.full(nobs, np.nan)
    residuals[good] = z - (eta - offset)[good]
    coef = _mv(T, coef)
    rV = r_matprod(T, rV)
    ww = np.zeros(nobs)
    wt = np.zeros(nobs)
    if fisher:
        wt[good] = w
        ww = wt
    else:
        wt[good] = wf
        ww[good] = w
    if deriv:
        db_drho = r_matprod(T, db_drho)
        if dw_drho.shape[0] != nobs:
            w1 = dw_drho
            dw_drho = np.zeros((nobs, w1.shape[1]))
            dw_drho[good] = w1
    if cache is not None and "nulldev" in cache:
        sumw, nulldev = cache["sumw"], cache["nulldev"]
    else:
        sumw = r_sum(weights)
        wtdmu = r_sum(weights * y) / sumw if intercept else link.linkinv(offset)
        nulldev = r_sum(family.dev_resids(y, np.broadcast_to(wtdmu, y.shape), weights))
        if cache is not None:
            cache["sumw"], cache["nulldev"] = sumw, nulldev
    if scale_known:
        dev1 = scale * sumw
    elif family.name == "gaussian":
        dev1 = dev
    elif math.isnan(reml_scale):
        dev1 = scale_est_v * sumw
    else:
        dev1 = reml_scale * sumw
    aic_model = family.aic(
        y, mu, dev1, weights, binom_n if binom_n is not None else nobs
    )

    return {
        "coefficients": coef,
        "residuals": residuals,
        "fitted.values": mu,
        "linear.predictors": eta,
        "deviance": dev,
        "null.deviance": nulldev,
        "iter": it,
        "weights": wt,
        "working.weights": ww,
        "z": z,
        "converged": conv,
        "boundary": boundary,
        "D1": out_D1,
        "D2": out_D2,
        "P": out_P,
        "P1": out_P1,
        "P2": out_P2,
        "trA": trA,
        "trA1": trA1,
        "trA2": trA2,
        "GCV": GCV,
        "GCV1": GCV1,
        "GCV2": GCV2,
        "GACV": GACV,
        "GACV1": GACV1,
        "GACV2": GACV2,
        "UBRE": UBRE,
        "UBRE1": UBRE1,
        "UBRE2": UBRE2,
        "REML": REML,
        "REML1": REML1,
        "REML2": REML2,
        "REML_Dp": reml_Dp,
        "ml_pen": ml_pen,
        "rV": rV,
        "db.drho": db_drho,
        "dw.drho": dw_drho,
        "dVkk": np.asarray(oo["dVkk"])[: nSp * nSp].reshape((nSp, nSp), order="F"),
        "ldetS1": det1 if grderiv else 0.0,
        "scale.est": scale_est_v,
        "reml.scale": reml_scale,
        "aic": aic_model,
        "rank": int(oo["rank.est"][0]),
        "K": Kmat,
        "T": T,
        "warn": warn,
    }


def _ls(family, y, weights, scale, binom_n, cache) -> np.ndarray:
    """``family$ls(y, weights, n, scale)``, from ``cache`` when it already
    holds the value at this scale."""
    key = ("ls", float(scale))
    if cache is not None and key in cache:
        return cache[key]
    out = np.asarray(family.ls_phi(y, weights, scale, n=binom_n), dtype=float)
    if cache is not None:
        cache[key] = out
    return out


def _pls(zg, xg, w, z, Sr, Eb, ng, q, rows_E, rank_tol) -> dict:
    """gam.fit3's ``.C(C_pls_fit1, ...)`` (gam.fit3.r:334): returns the
    coefficients in ``y``, the penalty, and ``n < 0`` when the penalized
    Hessian is indefinite."""
    oo = _pls_fit1(
        {
            "y": _f(zg),
            "X": _f(xg),
            "w": _f(w),
            "wy": _f(w * z),
            "E": _f(Sr),
            "Es": _f(Eb),
            "n": np.array([float(ng)]),
            "q": np.array([float(q)]),
            "rE": np.array([float(rows_E)]),
            "eta": _f(z),
            "penalty": np.array([1.0]),
            "rank.tol": np.array([rank_tol]),
            "nt": np.array([1.0]),
            "use.wy": np.array([0.0]),
        }
    )
    return {
        "y": np.asarray(oo["y"]),
        "penalty": np.asarray(oo["penalty"]),
        "n": float(oo["n"][0]),
    }


def vb_corr(R, L, lsp0, S, off, rho, Vr, nth=0, scale_est=False):
    """mgcv ``Vb.corr`` (gam.fit3.r:869-952) with ``w = NULL``: ``R`` is the
    root of the unpenalized Hessian, so the weights' derivatives are taken
    as zero and each ``dH`` is the penalty's alone. Rebuilds ``H =
    crossprod(R) + sum(lambda_i S_i)``, differentiates its Cholesky factor
    (``dchol``), inverts each derivative (``-R^{-1} dR R^{-1}``) and returns
    ``vcorr`` of them against ``Vr``, unscaled; 0 when ``H`` has no Cholesky
    factor or there is nothing to correct."""
    from .._rlinalg import r_backsolve, r_chol, r_crossprod
    from .gam import _mgcv_dchol, _mgcv_vcorr

    M = len(off)
    rho = np.asarray(rho, dtype=float)
    Vr = np.asarray(Vr, dtype=float)
    if scale_est:
        rho = rho[:-1]
        if L is not None:
            L = L[:-1, :-1]
        Vr = Vr[:-1, :-1]
    if lsp0 is None:
        lsp0 = np.zeros(rho.size) if L is None else np.zeros(L.shape[0])
    lsp0 = np.asarray(lsp0, dtype=float)
    if L is None:
        lam = np.exp(rho + lsp0[: rho.size])
    else:
        lam = np.exp(r_matprod(L, rho.reshape(-1, 1))[:, 0] + lsp0[: L.shape[0]])
    H = r_crossprod(R)
    for i in range(M):
        a = int(off[i])
        b = a + S[i].shape[1]
        H[a:b, a:b] = H[a:b, a:b] + lam[i + nth] * S[i]
    try:
        Rc = r_chol(H)
    except ValueError:
        return 0.0
    dH = []
    for i in range(lam.size):
        dHi = H * 0
        if i >= nth:
            a = int(off[i - nth])
            b = a + S[i - nth].shape[1]
            dHi[a:b, a:b] = dHi[a:b, a:b] + lam[i] * S[i - nth]
        dH.append(dHi)
    if L is not None:
        dH1, dH = dH, []
        for j in range(rho.size):
            acc = None
            for i in range(L.shape[0]):
                if L[i, j] != 0.0:
                    acc = dH1[i] * L[i, j] if acc is None else acc + dH1[i] * L[i, j]
            if acc is not None:
                dH.append(acc)
    if not dH:
        return 0.0
    Rt = np.ascontiguousarray(Rc.T)
    dR = []
    for dHk in dH:
        dRk = _mgcv_dchol(dHk, Rc)
        inner = np.ascontiguousarray(r_backsolve(Rc, dRk).T)
        dR.append(-np.ascontiguousarray(r_backsolve(Rt, inner, upper=False).T))
    return _mgcv_vcorr(dR, Vr)


def gam_fit3_post_proc(
    X,
    L,
    lsp0,
    S,
    off,
    obj,
    *,
    scale,
    scale_estimated,
    method,
    sp,
    hess,
    edge=None,
    nth=0,
) -> dict:
    """mgcv ``gam.fit3.post.proc`` (gam.fit3.r:955-1052) on gam.fit3's last
    fit ``obj``: the Bayesian covariance ``Vb = rV rV' scale``, the edf
    matrix ``F = rV K' sqrt(w) X`` and its diagonal, ``edf1``, ``Ve``, the
    leverages, the pivoted-QR factor ``R`` of ``sqrt(w) X`` with its columns
    restored and, for (RE)ML (``reml.scale`` known and ``db.drho`` present),
    the smoothing-parameter-uncertainty corrected ``Vc`` and ``edf2`` from
    the outer Hessian ``hess`` at the working log smoothing parameters
    (``sp`` is ``object$sp``, with the log scale last when estimated). With
    ``edge`` (newton's edge correction: ``hess1``, ``db.drho1``,
    ``dw.drho1``, ``lsp1``) ``Vc`` is recomputed there with the smaller
    prior, as mgcv's second pass does. ``nth`` is an extended family's
    number of estimated parameters, first in ``sp``."""
    from .._rlinalg import r_crossprod, r_qr_lapack, r_qr_R
    from .._rsum import r_rowsums
    from .gam import _mgcv_mmult, _r_eigen_sym

    if scale_estimated:
        scale = float(obj["scale.est"])
    rV = np.asarray(obj["rV"], dtype=float)
    K = np.asarray(obj["K"], dtype=float)
    Vb = r_matprod(rV, np.ascontiguousarray(rV.T)) * scale
    PKt = _mgcv_mmult(rV, K, 0, 1)
    WX = np.sqrt(np.asarray(obj["weights"], dtype=float))[:, None] * X
    F = _mgcv_mmult(PKt, WX, 0, 0)
    edf = np.diag(F).copy()
    edf1 = 2 * edf - r_rowsums(F.T * F)
    Ve = r_matprod(F, Vb)
    hat = r_rowsums(K * K)
    qrx = r_qr_lapack(WX)
    R0 = r_qr_R(qrx)
    R = np.empty_like(R0)
    R[:, np.asarray(qrx["pivot"]) - 1] = R0
    Vc = edf2 = V_sp = None
    if not math.isnan(float(obj["reml.scale"])) and obj["db.drho"] is not None:
        for k in (1, 2) if edge is not None else (1,):
            if k == 1:
                db = np.asarray(obj["db.drho"], dtype=float)
                lsp = np.log(np.asarray(sp, dtype=float))
                h = np.asarray(hess, dtype=float)
            else:
                db = np.asarray(edge["db.drho1"], dtype=float)
                lsp = np.asarray(edge["lsp1"], dtype=float)
                h = np.asarray(edge["hess1"], dtype=float)
            M = db.shape[1]
            if L is not None:
                db = r_matprod(db, np.ascontiguousarray(L[:M, :]))
                M = db.shape[1]
            ev_w, ev_V = _r_eigen_sym(h)
            d = np.array(ev_w, dtype=float)
            ind = d <= 0
            d[ind] = 0
            d[~ind] = 1 / np.sqrt(d[~ind])
            rV2 = np.ascontiguousarray((d[:, None] * ev_V.T)[:, :M])
            Vc1 = r_crossprod(r_matprod(rV2, np.ascontiguousarray(db.T)))
            d = np.array(ev_w, dtype=float)
            d[ind] = 0
            d = 1 / np.sqrt(d + 1 / 10) if k == 1 else 1 / np.sqrt(d + 1e-7)
            Vr = r_crossprod(d[:, None] * ev_V.T)
            drop_scale = scale_estimated and method not in ("P-REML", "P-ML")
            Vc2 = scale * vb_corr(R, L, lsp0, S, off, lsp, Vr, nth, drop_scale)
            Vc = Vb + Vc1 + Vc2
            if k == 1:
                edf2 = r_rowsums(Vc * r_crossprod(R)) / scale
                if r_sum(edf2) > r_sum(edf1):
                    edf2 = edf1
        V_sp = Vr
    return {
        "Vp": Vb,
        "Ve": Ve,
        "V.sp": V_sp,
        "edf": edf,
        "edf1": edf1,
        "edf2": edf2,
        "hat": hat,
        "F": F,
        "R": R,
        "Vc": Vc,
    }


def efsudr3(g3, lsp, family, *, lspmax=15.0, efs_tol=0.1) -> dict:
    """mgcv ``efsudr`` (gam.fit4.r:822-938): the extended Fellner-Schall
    iteration over the log smoothing parameters (an extended family's
    estimated parameters first, the log scale last when estimated), each
    fit gam.fit3 (or for an extended family gam.fit4, ``g3["fit4"]``) at
    deriv 0 with ``scoreType = "EFS"`` and ``gamma = 1``. As mgcv calls it, gam.fit3
    gets no ``null.coef`` (zeros), the first fit neither the user's
    ``mustart`` nor ``etastart`` and the later ones the user's
    ``mustart``; ``fit$scale`` and ``fit$dev`` are gam.fit3's ``scale.est``
    and ``deviance`` (R's partial matching). ``g3`` holds gam.fit3's fixed
    arguments. Returns the last fit with ``sp`` (``exp(lsp)``), ``iter`` and
    ``outer.info``."""
    UrS = g3["UrS"]
    nsp = len(UrS)
    fit4 = bool(g3.get("fit4"))
    nth = int(g3.get("nth", 0)) if fit4 else 0
    spi = slice(nth, nth + nsp)
    lsp = np.asarray(lsp, dtype=float).copy()
    estimate_scale = lsp.size > nth + nsp
    lsp[spi] = lsp[spi] + 2.5
    mult = 1.0

    def fit_at(sp, start, mustart):
        # the fit gets its own copy: lsp's scale entry is updated after
        if fit4:
            from ._fit4 import gam_fit4

            return gam_fit4(
                g3["X"],
                g3["y"],
                np.array(sp, dtype=float),
                g3["Eb"],
                UrS,
                family=family,
                weights=g3["weights"],
                start=start,
                mustart=mustart,
                offset=g3["offset"],
                U1=g3["U1"],
                Mp=g3["Mp"],
                epsilon=g3["epsilon"],
                maxit=g3["maxit"],
                deriv=0,
                gamma=1.0,
                scale=g3["scale"],
                score_type="EFS",
                null_coef=np.zeros(np.shape(g3["X"])[1]),
                use_wz=g3["use_wz"],
            )
        return gam_fit3(
            g3["X"],
            g3["y"],
            np.array(sp, dtype=float),
            g3["Eb"],
            UrS,
            family=family,
            weights=g3["weights"],
            start=start,
            mustart=mustart,
            offset=g3["offset"],
            U1=g3["U1"],
            Mp=g3["Mp"],
            epsilon=g3["epsilon"],
            maxit=g3["maxit"],
            scale_est=g3["scale_est"],
            intercept=True,
            deriv=0,
            gamma=1.0,
            scale=g3["scale"],
            score_type="EFS",
            null_coef=np.zeros(np.shape(g3["X"])[1]),
            n_true=g3["n_true"],
            binom_n=g3["binom_n"],
            x_finite=g3["x_finite"],
            cache=g3["cache"],
        )

    def theta_into(v):
        # the family's parameters as gam.fit4 left them
        if nth:
            v[:nth] = np.asarray(family.get_theta(), dtype=float)[:nth]

    mustart = g3["mustart"]
    fit = fit_at(lsp, g3["start"], None)
    theta_into(lsp)
    if estimate_scale:
        lsp[-1] = math.log(fit["scale.est"])
    U1 = np.asarray(g3["U1"], dtype=float)
    n, q = np.shape(g3["X"])
    Y = np.ascontiguousarray(U1[:, : q - int(g3["Mp"])])
    Yt = np.ascontiguousarray(Y.T)
    score_hist = np.zeros(200)
    bSb = np.zeros(nsp)
    trVS = np.zeros(nsp)
    old_dev = None
    it = 0
    for it in range(1, 201):
        start = np.asarray(fit["coefficients"], dtype=float)
        Yb = _mv(Yt, start)
        rV = np.ascontiguousarray(np.asarray(fit["rV"], dtype=float).T)
        rVY = r_matprod(rV, Y)
        for i in range(nsp):
            xx = r_matprod(Yb.reshape(1, -1), UrS[i])
            bSb[i] = r_sum(xx * xx)
            xx = r_matprod(rVY, UrS[i])
            trVS[i] = r_sum(xx * xx)
        fit_scale = fit["scale.est"]
        if fit4 and estimate_scale:
            # an extended family's scale corrected for edf (gam.fit4.r:869)
            edf = q - r_sum(trVS * np.exp(lsp[spi]))
            fit_scale = fit_scale * n / (n - edf)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.maximum(
                0, np.asarray(fit["ldetS1"], dtype=float) * np.exp(-lsp[spi]) - trVS
            )
            phi = fit_scale if estimate_scale else g3["scale"]
            r = a / np.maximum(0, bSb) * phi
        r[(a == 0) & (bSb == 0)] = 1
        r[~np.isfinite(r)] = 1e6
        lsp1 = lsp.copy()
        lsp1[spi] = np.minimum(lsp[spi] + np.log(r) * mult, lspmax)
        max_step = float(np.max(np.abs(lsp1 - lsp)))
        old_reml = fit["REML"]
        fit = fit_at(lsp1, start, mustart)
        theta_into(lsp1)
        if estimate_scale:
            lsp1[-1] = math.log(fit["scale.est"])
        if fit["REML"] <= old_reml:
            if max_step < 0.05:
                lsp2 = lsp.copy()
                lsp2[spi] = np.minimum(lsp[spi] + np.log(r) * mult * 2, lspmax)
                fit2 = fit_at(lsp2, start, mustart)
                theta_into(lsp2)
                if estimate_scale:
                    lsp2[-1] = math.log(fit["scale.est"])
                if fit2["REML"] < fit["REML"]:
                    fit, lsp = fit2, lsp2
                    mult = mult * 2
                else:
                    lsp = lsp1
            else:
                lsp = lsp1
        else:
            while fit["REML"] > old_reml and mult > 1:
                mult = mult / 2
                lsp1 = lsp.copy()
                lsp1[spi] = np.minimum(lsp[spi] + np.log(r) * mult, lspmax)
                fit = fit_at(lsp1, start, mustart)
                theta_into(lsp1)
                if estimate_scale:
                    lsp1[-1] = math.log(fit["scale.est"])
            lsp = lsp1
            if mult < 1:
                mult = 1.0
        score_hist[it - 1] = fit["REML"]
        if (
            it > 3
            and max_step < 0.05
            and float(np.max(np.abs(np.diff(score_hist[it - 4 : it])))) < efs_tol
        ):
            break
        if it == 1:
            old_dev = fit["deviance"]
        else:
            if abs(old_dev - fit["deviance"]) < 100 * g3["epsilon"] * abs(
                fit["deviance"]
            ):
                break
            old_dev = fit["deviance"]
    fit["sp"] = np.exp(lsp)
    fit["lsp"] = lsp
    fit["iter"] = it
    fit["outer.info"] = {
        "iter": it,
        "score_hist": score_hist[:it].copy(),
        "conv": "iteration limit reached" if it == 200 else "full convergence",
    }
    fit["score"] = float(fit["REML"])
    return fit
