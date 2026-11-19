"""mgcv's ``gam.fit4`` (gam.fit4.r:240-820) for extended families: one
evaluation of the REML or ML score. The penalized IRLS runs on the
deviance's derivatives with respect to the linear predictor (``dDeta``) in
``gam.reparam``'s stable basis, each step by mgcv's ``pls_fit1``; at
convergence ``gdi2`` gives the coefficient derivatives with respect to the
family's parameters and the log smoothing parameters, which are assembled
into the score and its first two derivatives as gam.fit4 does. Matrix
products are R's (``r_matprod``) and sums R's (``r_sum``), so each value is
mgcv's."""

from __future__ import annotations

import math

import numpy as np

from .._dispatch import rs_fn
from .._rlinalg import r_matprod, r_matprod_safe
from .._rsum import r_rowsums, r_sum
from ._fit3 import _f, _mv, _quad, _rows

_pls_fit1 = rs_fn("pls_fit1_c")
_gdi2 = rs_fn("gdi2_c")
_EPS = float(np.finfo(float).eps)


def _dev(family, y, mu, weights) -> float:
    """``sum(dev.resids(y, mu, weights, theta))`` at the family's theta."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return r_sum(family.dev_resids(y, mu, weights))


def _dd(family, y, mu, weights, theta, deriv) -> dict:
    """mgcv's ``dDeta(y, mu, weights, theta, family, deriv)`` with its
    ``good``: the observations whose derivatives (of the order asked for,
    the likelihood parameters' row sums included) are all finite."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        d = dict(family.dDeta(y, mu, weights, theta, level=deriv))
    fin = np.isfinite
    good = fin(d["Deta"]) & fin(d["Deta2"])
    nth = np.asarray(theta).size

    def rs(key):
        v = np.asarray(d[key], dtype=float)
        return fin(r_rowsums(v)) if v.ndim == 2 else fin(v)

    if deriv > 0:
        if nth > 1:
            good = good & rs("Dth") & rs("Detath") & rs("Deta2th") & fin(d["Deta3"])
        else:
            good = good & fin(d["Deta3"])
            if nth:
                good = good & rs("Dth") & rs("Detath") & rs("Deta2th")
        if deriv > 1:
            if nth > 1:
                good = (
                    good
                    & rs("Dth2")
                    & rs("Detath2")
                    & rs("Deta2th2")
                    & rs("Deta3th")
                    & fin(d["Deta4"])
                )
            else:
                good = good & fin(d["Deta4"])
                if nth:
                    good = (
                        good
                        & rs("Dth2")
                        & rs("Detath2")
                        & rs("Deta2th2")
                        & rs("Deta3th")
                    )
    d["good"] = good
    return d


def _pls4(zg, xg, w, wz, z, Sr, Eb, ng, q, rows_E, rank_tol, use_wy) -> dict:
    """gam.fit4's ``.C(C_pls_fit1, ...)`` (gam.fit4.r:385): the weights
    ``w`` may be negative, ``wy`` is ``Wz``, used when ``use_wy``."""
    oo = _pls_fit1(
        {
            "y": _f(zg),
            "X": _f(xg),
            "w": _f(w),
            "wy": _f(wz),
            "E": _f(Sr),
            "Es": _f(Eb),
            "n": np.array([float(ng)]),
            "q": np.array([float(q)]),
            "rE": np.array([float(rows_E)]),
            "eta": _f(z),
            "penalty": np.array([1.0]),
            "rank.tol": np.array([rank_tol]),
            "nt": np.array([1.0]),
            "use.wy": np.array([1.0 if use_wy else 0.0]),
        }
    )
    return {
        "y": np.asarray(oo["y"]),
        "penalty": float(np.asarray(oo["penalty"])[0]),
        "n": float(oo["n"][0]),
    }


def gam_fit4(
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
    deriv=2,
    gamma=1.0,
    scale=1.0,
    score_type="REML",
    null_coef=None,
    use_wz=False,
) -> dict:
    """mgcv ``gam.fit4`` at ``sp``: the family's estimated parameters first
    (``family.n_theta`` of them, put into the family), the log smoothing
    parameters, and the log scale last when ``scale <= 0``. ``Eb``,
    ``UrS``, ``U1`` and ``Mp`` are gam.setup's. Returns gam.fit4's list:
    the coefficients (original basis), fit, REML/ML score and derivatives
    (with respect to the family's estimated parameters, the log smoothing
    parameters and the log scale), ``rV``, ``db.drho``, ``dw.drho``, ``K``
    and the rest."""
    from .gam import _gam_reparam

    if _pls_fit1 is None or _gdi2 is None:
        raise RuntimeError("gam_fit4 needs the compiled pls_fit1 and gdi2")
    if score_type not in ("REML", "ML", "EFS"):
        raise NotImplementedError(f"gam.fit4's {score_type} score is not ported")
    efs = score_type == "EFS"
    # mgcv's family$scale: -1 when the family's scale is estimated
    family_scale = None if family.scale_known else -1.0
    link = family.link
    warn: list[str] = []
    sp = np.asarray(sp, dtype=float).ravel().copy()
    n_th = int(family.n_theta)
    if n_th > 0:
        theta = sp[:n_th].copy()
        family.set_theta(theta)
        sp = sp[n_th:]
    else:
        theta = np.asarray(family.get_theta(), dtype=float).ravel().copy()
    scale = float(scale)
    scale_known = scale > 0
    if not scale_known:
        scale = math.exp(sp[-1])
        sp = sp[:-1]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nSp = sp.size
    rank_tol = _EPS * 100
    q = x.shape[1]
    nobs = y.shape[0]
    weights = np.ones(nobs) if weights is None else np.asarray(weights, dtype=float)
    offset = np.zeros(nobs) if offset is None else np.asarray(offset, dtype=float)
    null_coef = np.zeros(q) if null_coef is None else np.asarray(null_coef, dtype=float)
    U1 = np.eye(q) if U1 is None else np.asarray(U1, dtype=float)
    Mp = int(Mp)

    if len(UrS):
        grderiv = 1 if efs else deriv
        rp = _gam_reparam(list(UrS), sp, grderiv, fixed_penalty=len(UrS) > nSp)
        Qs = rp["Qs"]
        nq = Qs.shape[1]
        T = np.eye(q)
        T[:nq, :nq] = Qs
        T = r_matprod(U1, T)
        null_coef = _mv(np.ascontiguousarray(T.T), null_coef)
        if start is not None:
            start = _mv(np.ascontiguousarray(T.T), start)
        x = r_matprod(x, T)
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
    del fixed_penalty
    x_fin = r_matprod_safe(x)

    null_eta = _mv(x, null_coef, x_fin) + offset
    mukeep = None if mustart is None else np.asarray(mustart, dtype=float)
    mustart = np.asarray(family.gam_initialize(y, weights), dtype=float)
    old_pdev = _dev(family, y, link.linkinv(null_eta), weights) + _quad(null_coef, St)
    if start is not None:
        pdev = _dev(
            family, y, link.linkinv(_mv(x, start, x_fin) + offset), weights
        ) + _quad(start, St)
        if pdev > old_pdev:
            start = mukeep = etastart = None
    coefold = null_coef
    etaold = _mv(x, coefold, x_fin) + offset
    if mukeep is not None:
        mustart = mukeep

    if etastart is not None:
        eta = np.asarray(etastart, dtype=float)
    elif start is not None:
        if start.size != q:
            raise ValueError(f"Length of start should equal {q}")
        coefold = start
        etaold = offset + (x[:, 0] * start[0] if q == 1 else _mv(x, start, x_fin))
        eta = etaold
    else:
        eta = link.link(mustart)
    mu = link.linkinv(eta)
    conv = boundary = False
    dd = _dd(family, y, mu, weights, theta, 0)
    w = dd["Deta2"] * 0.5
    with np.errstate(invalid="ignore", over="ignore"):
        wz = w * (eta - offset) - 0.5 * dd["Deta"]
        z = (eta - offset) - dd["Deta.Deta2"]
    good = np.isfinite(z) & np.isfinite(w)
    zg = np.zeros(max(x.shape))
    coef = start
    it = 0
    for it in range(1, int(maxit) + 1):
        if not good.all():
            use_wy = True
            good = np.isfinite(w) & np.isfinite(wz)
            z = np.where(np.isfinite(z), z, 0.0)
        else:
            use_wy = bool(use_wz)
        if not good.any():
            raise FloatingPointError("no good data in iteration")
        ng = int(good.sum())
        zg[:ng] = z[good]
        oo = _pls4(
            zg,
            _rows(x, good),
            w[good],
            wz[good],
            z,
            Sr,
            Eb,
            ng,
            q,
            rows_E,
            rank_tol,
            use_wy,
        )
        posdef = oo["n"] >= 0
        if not posdef:
            with np.errstate(invalid="ignore"):
                good = np.isfinite(dd["Deta2"])
                good[good] = dd["Deta2"][good] > 0
            w = np.where(good, w, 0.0)
            with np.errstate(invalid="ignore", over="ignore"):
                wz = w * (eta - offset) - 0.5 * dd["Deta"]
                z = (eta - offset) - dd["Deta.Deta2"]
            good = np.isfinite(z) & np.isfinite(w)
            if not good.all():
                use_wy = True
                good = np.isfinite(w) & np.isfinite(wz)
                z = np.where(np.isfinite(z), z, 0.0)
            else:
                use_wy = bool(use_wz)
            ng = int(good.sum())
            zg[:ng] = z[good]
            oo = _pls4(
                zg,
                _rows(x, good),
                w[good],
                wz[good],
                z,
                Sr,
                Eb,
                ng,
                q,
                rows_E,
                rank_tol,
                use_wy,
            )
        start = oo["y"][:q].copy()
        penalty = oo["penalty"]
        eta = _mv(x, start, x_fin)
        if np.any(~np.isfinite(start)):
            warn.append(f" gam.fit4 non-finite coefficients at iteration  {it}")
            return {"REML": float("nan"), "warn": warn}
        eta = eta + offset
        mu = link.linkinv(eta)
        dev = _dev(family, y, mu, weights)
        if not math.isfinite(dev):
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
            ii = 1
            while not (link.valideta(eta) and family.validmu(mu)):
                if ii > maxit:
                    raise FloatingPointError("inner loop 2; can't correct step size")
                ii += 1
                start = (start + coefold) / 2
                eta = (eta + etaold) / 2
                mu = link.linkinv(eta)
            boundary = True
            dev = _dev(family, y, mu, weights)
            penalty = _quad(start, St)
        pdev = dev + penalty
        div_thresh = 10 * (0.1 + abs(old_pdev)) * _EPS**0.5
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
                penalty = _quad(start, St)
                pdev = dev + penalty
        if efs and n_th > 0:
            # the family's parameters at the accepted means (gam.fit4.r:507)
            from .bam import _estimate_theta

            scale1 = family_scale if family_scale is not None else scale
            theta = _estimate_theta(family, y, mu, scale=scale1, wt=weights, tol=1e-7)
            if family_scale is not None and family_scale < 0:
                scale = math.exp(theta[n_th])
                theta = theta[:n_th]
            family.set_theta(theta)
        dd = _dd(family, y, mu, weights, theta, 0)
        w = dd["Deta2"] * 0.5
        with np.errstate(invalid="ignore", over="ignore"):
            wz = w * (eta - offset) - 0.5 * dd["Deta"]
            z = (eta - offset) - dd["Deta.Deta2"]
        good = np.isfinite(z) & np.isfinite(w)
        if posdef and abs(pdev - old_pdev) / (0.1 + abs(pdev)) < epsilon:
            xs = _mv(x, start, x_fin)
            grad = 2 * _mv(
                np.asfortranarray(_rows(x, good).T),
                w[good] * xs[good] - wz[good],
                x_fin,
            ) + 2 * _mv(St, start)
            if np.max(np.abs(grad)) > epsilon * (abs(pdev) + scale):
                old_pdev = pdev
                coef = coefold = start
                etaold = eta
            else:
                conv = True
                coef = start
                break
        else:
            old_pdev = pdev
            coef = coefold = start
            etaold = eta
        if efs and n_th > 0:
            dev = _dev(family, y, mu, weights)
            old_pdev = pdev = dev + penalty

    coef = _mv(T, coef)
    if efs:
        score_type = "REML"
    dd = _dd(family, y, mu, weights, theta, deriv)
    w = dd["Deta2"] * 0.5
    with np.errstate(invalid="ignore", over="ignore"):
        z = (eta - offset) - dd["Deta.Deta2"]
        wf = np.maximum(0, dd["EDeta2"] * 0.5)
        wz = w * (eta - offset) - 0.5 * dd["Deta"]
    good = np.isfinite(wz) & np.isfinite(w) & dd["good"]
    if not good.any():
        raise FloatingPointError("not enough finite derivatives")
    residuals = z - (eta - offset)
    residuals = np.where(np.isfinite(residuals), residuals, np.nan)
    z = np.where(np.isfinite(z), z, 0.0)
    nth = theta.size
    ntot = nth + nSp
    rSncol = [u.shape[1] for u in UrS]

    def col(key, k=None):
        v = np.asarray(dd[key], dtype=float)
        if v.ndim == 1:
            v = v[:, None]
        v = v[good]
        return _f(v) if k is None else v[:, k]

    if not good.all():
        z, w, wz, wf = z[good], w[good], wz[good], wf[good]
    gdi_type = (
        1
        if (
            np.any(np.abs(w) < np.finfo(float).tiny * 1e20)
            or not np.all(np.isfinite(z))
        )
        else 0
    )
    nz = z.size
    zero = np.zeros(1)
    oo = _gdi2(
        {
            "X": _f(_rows(x, good)),
            "E": _f(Sr),
            "Es": _f(Eb),
            "rS": np.concatenate([_f(r) for r in rS]) if rS else zero,
            "U1": _f(U1),
            "sp": np.exp(sp) if nSp else zero,
            "theta": _f(theta) if nth else zero,
            "z": _f(z),
            "w": _f(w),
            "wz": _f(wz),
            "wf": _f(wf),
            "Dth": col("Dth") if deriv and nth else zero,
            "Det": col("Deta"),
            "Det2": col("Deta2"),
            "Dth2": col("Dth2") if deriv > 1 and nth else zero,
            "Det.th": col("Detath") if deriv and nth else zero,
            "Det2.th": col("Deta2th") if deriv and nth else zero,
            "Det3": col("Deta3") if deriv else zero,
            "Det.th2": col("Detath2") if deriv > 1 and nth else zero,
            "Det4": col("Deta4") if deriv > 1 else zero,
            "Det3.th": col("Deta3th") if deriv > 1 and nth else zero,
            "Deta2.th2": col("Deta2th2") if deriv > 1 and nth else zero,
            "beta": _f(coef),
            "b1": np.zeros(max(ntot * q, 1)),
            "w1": np.zeros(max(ntot * nz, 1)),
            "D1": np.zeros(max(ntot, 1)),
            "D2": np.zeros(max(ntot * ntot, 1)),
            "P": np.zeros(1),
            "P1": np.zeros(max(ntot, 1)),
            "P2": np.zeros(max(ntot * ntot, 1)),
            "ldet": np.array([1.0 - 2.0 * (score_type == "ML")]),
            "ldet1": np.zeros(max(ntot, 1)),
            "ldet2": np.zeros(max(ntot * ntot, 1)),
            "rV": np.zeros(q * q),
            "rank.tol": np.array([_EPS**0.75]),
            "rank.est": np.zeros(1),
            "n": np.array([float(nz)]),
            "q": np.array([float(q)]),
            "M": np.array([float(nSp)]),
            "n.theta": np.array([float(nth)]),
            "Mp": np.array([float(Mp)]),
            "Enrow": np.array([float(rows_E)]),
            "rSncol": np.array(rSncol if rSncol else [0], dtype=float),
            "deriv": np.array([float(deriv)]),
            "fixed.penalty": np.array([1.0 if len(UrS) > nSp else 0.0]),
            "nt": np.array([1.0]),
            "type": np.array([float(gdi_type)]),
            "dVkk": np.zeros(max(nSp * nSp, 1)),
        }
    )
    rV = np.asarray(oo["rV"]).reshape((q, q), order="F")
    db_drho = (
        np.asarray(oo["b1"])[: q * ntot].reshape((q, ntot), order="F")
        if deriv
        else None
    )
    dw_drho = (
        np.asarray(oo["w1"])[: nz * ntot].reshape((nz, ntot), order="F")
        if deriv
        else None
    )
    K_good = np.asarray(oo["X"])[: nz * q].reshape((nz, q), order="F")
    if nz == nobs:
        Kmat = K_good
    else:
        Kmat = np.zeros((nobs, q))
        Kmat[good] = K_good

    D1 = np.asarray(oo["D1"])[:ntot]
    D2 = np.asarray(oo["D2"])[: ntot * ntot].reshape((ntot, ntot), order="F")
    ldet2 = (
        np.asarray(oo["ldet2"])[: ntot * ntot].reshape((ntot, ntot), order="F").copy()
    )
    bSb2 = np.asarray(oo["P2"])[: ntot * ntot].reshape((ntot, ntot), order="F")
    P0 = float(oo["P"][0])
    P1 = np.asarray(oo["P1"])[:ntot]
    ldet = float(oo["ldet"][0])
    ls = family.ls_extended(y, weights, theta=theta, scale=scale)
    lsth1_all = np.atleast_1d(np.asarray(ls["lsth1"], dtype=float))
    lsth2_all = np.atleast_2d(np.asarray(ls["lsth2"], dtype=float))
    lsth1 = lsth1_all[:nth] if nth else np.zeros(0)
    lsth2 = lsth2_all[:nth, :nth] if nth else np.zeros((0, 0))
    reml_ind = 1.0 if score_type == "REML" else 0.0
    REML = (
        ((dev + P0) / (2 * scale) - float(ls["ls"])) / gamma
        + (ldet - det) / 2
        - reml_ind * Mp * (math.log(2 * math.pi * scale) / 2 - math.log(gamma) / 2)
    )
    ml_pen = (
        P0 / (2 * scale * gamma)
        + (ldet - det) / 2
        - reml_ind * Mp * (math.log(2 * math.pi * scale) / 2 - math.log(gamma) / 2)
    )
    REML1 = REML2 = None
    if deriv:
        dt1 = np.asarray(oo["ldet1"])[:ntot].copy()
        if nSp:
            dt1[nth:] = dt1[nth:] - det1
        REML1 = (
            (D1 + P1) / (2 * scale) - np.concatenate([lsth1, np.zeros(nSp)])
        ) / gamma + dt1 / 2
        if deriv > 1:
            ls2 = D2 * 0
            if nth:
                ls2[:nth, :nth] = lsth2
            if nSp:
                ldet2[nth:, nth:] = ldet2[nth:, nth:] - det2
            REML2 = ((D2 + bSb2) / (2 * scale) - ls2) / gamma + ldet2 / 2
    if not scale_known and deriv:
        Dp = dev + P0
        dlr_dlphi = (-Dp / (2 * scale) - lsth1_all[nth]) / gamma - reml_ind * Mp / 2
        d2lr_d2lphi = (Dp / (2 * scale) - lsth2_all[nth, nth]) / gamma
        d2lr_dspphi = -(D1 + P1) / (2 * scale * gamma)
        if nth:
            d2lr_dspphi[:nth] = d2lr_dspphi[:nth] - lsth2_all[nth, :nth] / gamma
        REML1 = np.append(REML1, dlr_dlphi)
        if deriv == 2:
            REML2 = np.vstack([REML2, d2lr_dspphi])
            REML2 = np.column_stack([REML2, np.append(d2lr_dspphi, d2lr_d2lphi)])

    if deriv:
        db_drho = r_matprod(T, db_drho)
    if deriv > 0 and n_th == 0 and nth > 0:
        REML1 = REML1[nth:]
        if deriv > 1:
            REML2 = REML2[nth:, nth:]
        db_drho = db_drho[:, nth:]

    sumw = r_sum(weights)
    wtdmu = r_sum(weights * y) / sumw
    nulldev = _dev(family, y, np.full(nobs, wtdmu), weights)
    wt = np.zeros(nobs)
    ww = np.zeros(nobs)
    wt[good] = wf
    ww[good] = w
    if deriv and dw_drho.shape[0] != nobs:
        w1 = dw_drho
        dw_drho = np.zeros((nobs, w1.shape[1]))
        dw_drho[good] = w1
    aic_model = float(family.aic(y, mu, scale * sumw, weights, nobs, theta=theta))

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
        "REML": REML,
        "REML1": REML1,
        "REML2": REML2,
        "ml_pen": ml_pen,
        "rV": r_matprod(T, rV),
        "db.drho": db_drho,
        "dw.drho": dw_drho,
        "scale.est": scale,
        "reml.scale": scale,
        "aic": aic_model,
        "rank": int(oo["rank.est"][0]),
        "K": Kmat,
        "dVkk": np.asarray(oo["dVkk"])[: nSp * nSp].reshape((nSp, nSp), order="F"),
        "ldetS1": det1 if grderiv else 0.0,
        "theta": theta.copy(),
        "warn": warn,
    }
