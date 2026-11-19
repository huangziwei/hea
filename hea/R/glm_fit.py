"""R's ``glm.fit`` (stats glm.R): iteratively reweighted least squares, each
step a LINPACK least squares fit (``Cdqrls``) of the working response on
the weighted model matrix."""

from __future__ import annotations

import warnings

import numpy as np

from .._rlinalg import r_matprod
from .._rsum import r_sum
from .linalg import Cdqrls


def glm_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    family,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    start: np.ndarray | None = None,
    etastart: np.ndarray | None = None,
    mustart: np.ndarray | None = None,
    epsilon: float = 1e-8,
    maxit: int = 25,
    intercept: bool = True,
    binom_n: np.ndarray | None = None,
) -> dict:
    """``glm.fit(x, y, weights, start, etastart, mustart, offset, family,
    control = glm.control(epsilon, maxit), intercept)`` for a model matrix
    with at least one column. The starting linear predictor is
    ``etastart``, else ``offset + x %*% start`` (``start`` also being the
    first step's fallback), else the link of ``mustart``, else of the
    family's ``initialize``; then
    IRLS steps until ``|dev - devold| / (0.1 + |dev|) < epsilon``, halving a
    step towards the previous coefficients while the deviance is not finite
    or the linear predictor or means are invalid. ``binom_n`` is binomial's
    trials vector (R's ``n`` from ``initialize`` for a two-column
    response), which its ``initialize`` and ``aic`` take.

    Returns R's list: ``coefficients`` (NA for the aliased ones),
    ``residuals`` (working), ``fitted``, ``effects``, ``R``, ``rank``, ``qr``
    (``Cdqrls``'s ``qr``, ``rank``, ``qraux``, ``pivot``, ``tol``),
    ``linear_predictors``, ``deviance``, ``aic``, ``null_deviance``,
    ``iter``, ``weights`` (working), ``prior_weights``, ``df_residual``,
    ``df_null``, ``converged`` and ``boundary``, with R's warnings."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nobs = y.size
    nvars = x.shape[1]
    weights = np.ones(nobs) if weights is None else np.asarray(weights, dtype=float)
    offset = np.zeros(nobs) if offset is None else np.asarray(offset, dtype=float)
    link = family.link
    init_kw = {} if binom_n is None else {"n": binom_n}
    mu_init = family.initialize(y, weights, **init_kw)
    if mustart is not None:
        mu_init = np.asarray(mustart, dtype=float)
    coefold = None
    if etastart is not None:
        eta = np.asarray(etastart, dtype=float)
    elif start is not None:
        start = np.asarray(start, dtype=float)
        if start.size != nvars:
            raise ValueError(
                f"length of 'start' should equal {nvars} and correspond to "
                "initial coefs"
            )
        coefold = start
        xs = (
            x[:, 0] * start[0]
            if nvars == 1
            else r_matprod(x, start.reshape(-1, 1))[:, 0]
        )
        eta = offset + xs
    else:
        eta = link.link(mu_init)
    mu = link.linkinv(eta)
    if not (family.validmu(mu) and link.valideta(eta)):
        raise ValueError("cannot find valid starting values: please specify some")
    devold = r_sum(family.dev_resids(y, mu, weights))
    boundary = conv = False
    start = np.zeros(nvars)
    coef = None
    fit = None
    good = weights > 0
    w = np.zeros(0)
    dev = devold
    it = 0
    for it in range(1, maxit + 1):
        good = weights > 0
        varmu = family.variance(mu)[good]
        if np.any(np.isnan(varmu)):
            raise ValueError("NAs in V(mu)")
        if np.any(varmu == 0):
            raise ValueError("0s in V(mu)")
        mu_eta_val = link.mu_eta(eta)
        if np.any(np.isnan(mu_eta_val[good])):
            raise ValueError("NAs in d(mu)/d(eta)")
        good = (weights > 0) & (mu_eta_val != 0)
        if not np.any(good):
            conv = False
            warnings.warn(
                f"no observations informative at iteration {it}", stacklevel=2
            )
            break
        z = (eta - offset)[good] + (y - mu)[good] / mu_eta_val[good]
        w = np.sqrt((weights[good] * mu_eta_val[good] ** 2) / family.variance(mu)[good])
        fit = Cdqrls(x[good] * w[:, None], z * w, min(1e-7, epsilon / 1000))
        if not np.all(np.isfinite(fit["coefficients"])):
            conv = False
            warnings.warn(f"non-finite coefficients at iteration {it}", stacklevel=2)
            break
        if nobs < fit["rank"]:
            raise ValueError(
                f"X matrix has rank {fit['rank']}, but only {nobs} observations"
            )
        start = start.copy()
        start[np.asarray(fit["pivot"]) - 1] = fit["coefficients"]
        eta = r_matprod(x, start.reshape(-1, 1))[:, 0] + offset
        mu = link.linkinv(eta)
        dev = r_sum(family.dev_resids(y, mu, weights))
        boundary = False
        if not np.isfinite(dev):
            if coefold is None:
                raise ValueError(
                    "no valid set of coefficients has been found: please supply "
                    "starting values"
                )
            warnings.warn("step size truncated due to divergence", stacklevel=2)
            ii = 1
            while not np.isfinite(dev):
                if ii > maxit:
                    raise ValueError("inner loop 1; cannot correct step size")
                ii += 1
                start = (start + coefold) / 2
                eta = r_matprod(x, start.reshape(-1, 1))[:, 0] + offset
                mu = link.linkinv(eta)
                dev = r_sum(family.dev_resids(y, mu, weights))
            boundary = True
        if not (link.valideta(eta) and family.validmu(mu)):
            if coefold is None:
                raise ValueError(
                    "no valid set of coefficients has been found: please supply "
                    "starting values"
                )
            warnings.warn("step size truncated: out of bounds", stacklevel=2)
            ii = 1
            while not (link.valideta(eta) and family.validmu(mu)):
                if ii > maxit:
                    raise ValueError("inner loop 2; cannot correct step size")
                ii += 1
                start = (start + coefold) / 2
                eta = r_matprod(x, start.reshape(-1, 1))[:, 0] + offset
                mu = link.linkinv(eta)
            boundary = True
            dev = r_sum(family.dev_resids(y, mu, weights))
        if abs(dev - devold) / (0.1 + abs(dev)) < epsilon:
            conv = True
            coef = start
            break
        devold = dev
        coef = coefold = start
    if not conv:
        warnings.warn("glm.fit: algorithm did not converge", stacklevel=2)
    if boundary:
        warnings.warn("glm.fit: algorithm stopped at boundary value", stacklevel=2)
    eps = 10 * float(np.finfo(float).eps)
    if family.name == "binomial" and (np.any(mu > 1 - eps) or np.any(mu < eps)):
        warnings.warn(
            "glm.fit: fitted probabilities numerically 0 or 1 occurred", stacklevel=2
        )
    if family.name == "poisson" and np.any(mu < eps):
        warnings.warn("glm.fit: fitted rates numerically 0 occurred", stacklevel=2)
    if fit is None:
        raise ValueError("glm.fit: no informative observations")
    rank = int(fit["rank"])
    pivot = np.asarray(fit["pivot"], dtype=np.int64)
    coef = np.array(coef, dtype=float)
    if rank < nvars:
        coef[pivot[rank:] - 1] = np.nan
    residuals = (y - mu) / link.mu_eta(eta)
    qr = np.asarray(fit["qr"], dtype=float)
    nr = min(int(np.sum(good)), nvars)
    if nr < nvars:
        Rmat = np.eye(nvars)
        Rmat[:nr, :nvars] = qr[:nr, :nvars]
    else:
        Rmat = qr[:nvars, :nvars].copy()
    Rmat = np.triu(Rmat)
    wt = np.zeros(nobs)
    wt[good] = w**2
    wtdmu = r_sum(weights * y) / r_sum(weights) if intercept else link.linkinv(offset)
    nulldev = r_sum(family.dev_resids(y, np.broadcast_to(wtdmu, y.shape), weights))
    n_ok = nobs - int(np.sum(weights == 0))
    aic_n = nobs if binom_n is None else binom_n
    aic_model = family.aic(y, mu, dev, weights, aic_n) + 2 * rank
    return {
        "coefficients": coef,
        "residuals": residuals,
        "fitted": mu,
        "effects": np.asarray(fit["effects"], dtype=float),
        "R": Rmat,
        "rank": rank,
        "qr": {
            "qr": qr,
            "rank": rank,
            "qraux": np.asarray(fit["qraux"], dtype=float),
            "pivot": pivot,
            "tol": min(1e-7, epsilon / 1000),
        },
        "linear_predictors": eta,
        "deviance": float(dev),
        "aic": float(aic_model),
        "null_deviance": float(nulldev),
        "iter": it,
        "weights": wt,
        "prior_weights": weights,
        "df_residual": n_ok - rank,
        "df_null": n_ok - int(intercept),
        "converged": conv,
        "boundary": boundary,
    }
