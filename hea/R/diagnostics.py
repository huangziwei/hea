"""R's regression-diagnostics family: ``hatvalues``, ``rstandard``,
``rstudent``, ``cooks.distance``, ``dffits``, ``dfbetas``, ``influence``.

Most diagnostics are defined in terms of three primitives that hea
already caches at fit time: ``leverage`` (h_ii), the standardized
residuals (``std_residuals`` for lm; ``std_dev_residuals`` /
``std_pearson_residuals`` for glm/gam/bam), and the cross-product
inverse ``XtXinv`` (lm). The closed-form deletion diagnostics follow
R's ``rstudent`` / ``dffits`` / ``dfbetas`` / ``influence`` formulas:
the lm path uses ``XtXinv``; the glm / gam / bam path uses
``Vp/scale`` (or ``V_bhat/dispersion``) and IRLS working weights
recovered from leverage.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def hatvalues(model):
    """R: ``hatvalues()`` — leverage ``h_ii`` (hat-matrix diagonal)."""
    if model.__class__.__name__ == "gmm":  # mixed model
        return np.asarray(model.hatvalues())
    if hasattr(model, "leverage"):
        return np.asarray(model.leverage)
    raise TypeError(f"hatvalues(): {model.__class__.__name__} has no leverage")


def rstandard(model, type=None):
    """R: ``rstandard()`` — internally studentized residuals.

    For ``glm`` / ``gam`` / ``bam``, ``type`` selects between
    ``"deviance"`` (default, matches ``rstandard.glm``) and ``"pearson"``.
    For ``lm``, only one form exists (Gaussian) and ``type`` is ignored.
    """
    if type == "pearson" and hasattr(model, "std_pearson_residuals"):
        return np.asarray(model.std_pearson_residuals)
    if type not in (None, "deviance", "pearson"):
        raise ValueError(
            f"rstandard(): type={type!r} not recognized (use 'deviance' or 'pearson')"
        )
    if hasattr(model, "std_dev_residuals"):
        return np.asarray(model.std_dev_residuals)
    if hasattr(model, "std_residuals"):
        return np.asarray(model.std_residuals)
    raise TypeError(
        f"rstandard(): {model.__class__.__name__} has no standardized residuals"
    )


def _loo_sigma_lm(model) -> np.ndarray:
    """Leave-one-out σ estimates ``σ_(-i)`` for ``lm`` (weighted-aware)."""
    e = model.residuals.to_series().to_numpy()
    h = np.asarray(model.leverage)
    n = int(model.n)
    p = int(model.p)
    df_loo = n - p - 1
    if df_loo <= 0:
        raise ValueError(f"deletion diagnostics need n - p - 1 > 0; got n={n}, p={p}")
    weights = getattr(model, "weights", None)
    if weights is None:
        w = np.ones_like(e)
    else:
        w = np.asarray(weights, dtype=float)
    weighted_rss = float(np.sum(w * e * e))
    one_minus_h = np.clip(1 - h, 1e-12, None)
    rss_loo = weighted_rss - (w * e * e) / one_minus_h
    return np.sqrt(np.maximum(rss_loo, 0.0) / df_loo)


def _lm_weights_array(model) -> np.ndarray:
    """Return the weights as an ``ndarray`` of length ``n`` (all-ones if unweighted)."""
    e = model.residuals.to_series().to_numpy()
    weights = getattr(model, "weights", None)
    if weights is None:
        return np.ones_like(e)
    return np.asarray(weights, dtype=float)


def _xtwxinv_glm_gam(model) -> np.ndarray:
    """Return the cached ``(X'WX + S)^-1`` for glm/gam/bam (penalty included)."""
    if hasattr(model, "Vp"):  # gam / bam
        return np.asarray(model.Vp) / float(model.scale)
    if hasattr(model, "V_bhat"):  # glm
        return np.asarray(model.V_bhat) / float(model.dispersion)
    raise AttributeError(
        f"{model.__class__.__name__}: no Vp / V_bhat for jackknife inputs"
    )


def _loo_sigma_glm_gam(model) -> np.ndarray:
    """Leave-one-out σ estimates for glm/gam/bam."""
    h = np.asarray(model.leverage)
    if model.family.scale_known:
        return np.ones_like(h)
    n = int(model.n)
    p = int(model.p)
    df_loo = n - p - 1
    if df_loo <= 0:
        raise ValueError(f"deletion diagnostics need n - p - 1 > 0; got n={n}, p={p}")
    d = np.asarray(model.residuals_of("deviance"))
    one_minus_h = np.clip(1 - h, 1e-12, None)
    sigma_sq = (float(model.deviance) - d**2 / one_minus_h) / df_loo
    return np.sqrt(np.maximum(sigma_sq, 0.0))


def _design_full(model) -> np.ndarray:
    """Return the full design matrix as an ndarray."""
    if hasattr(model, "_X_full"):
        return np.asarray(model._X_full, dtype=float)
    return model.X.to_numpy().astype(float)


def _irls_inputs(model) -> dict:
    """Inputs for closed-form glm/gam jackknife diagnostics."""
    h = np.asarray(model.leverage)
    X = _design_full(model)
    XtWXinv = _xtwxinv_glm_gam(model)

    hX = X @ XtWXinv
    quad = (hX * X).sum(axis=1)
    safe_quad = np.where(quad > 0, quad, 1.0)
    w_irls = h / safe_quad

    mu = np.asarray(model.fitted_values, dtype=float)
    eta = np.asarray(model.linear_predictors, dtype=float)
    y_arr = (
        model.y.to_numpy().astype(float)
        if isinstance(model.y, pl.Series)
        else np.asarray(model.y, dtype=float)
    )
    mu_eta = np.asarray(model.family.link.mu_eta(eta), dtype=float)
    safe_mu_eta = np.where(mu_eta != 0, mu_eta, 1.0)
    working_resid = (y_arr - mu) / safe_mu_eta

    sigma_loo = _loo_sigma_glm_gam(model)

    return {
        "X": X,
        "XtWXinv": XtWXinv,
        "w_irls": w_irls,
        "working_resid": working_resid,
        "h": h,
        "sigma_loo": sigma_loo,
    }


def rstudent(model):
    """R: ``rstudent()`` — externally studentized residuals.

    For ``lm`` (Gaussian), uses the closed form
    ``r_i^* = r_i · √((n-p-1) / (n-p - r_i^2))`` derived from the
    leave-one-out σ estimate.

    For ``glm`` / ``gam`` / ``bam``, follows R's ``rstudent.glm`` —
    the Williams (1987) likelihood residual:
    ``r_i = sign(d_i) · √(d_i² + p_i² · h_i / (1-h_i)) / (σ_(-i) · √(1-h_i))``
    where ``d_i`` and ``p_i`` are raw deviance and Pearson residuals.
    Known-scale families fix ``σ_(-i) = 1``.
    """
    if model.__class__.__name__ == "gmm":  # mixed model
        return np.asarray(model.rstudent())
    if hasattr(model, "std_residuals"):
        e = model.residuals.to_series().to_numpy()
        h = np.asarray(model.leverage)
        sigma_loo = _loo_sigma_lm(model)
        sqrt_w = np.sqrt(_lm_weights_array(model))
        one_minus_h = np.clip(1 - h, 1e-12, None)
        return e * sqrt_w / (sigma_loo * np.sqrt(one_minus_h))

    if not hasattr(model, "residuals_of"):
        raise TypeError(f"rstudent(): {model.__class__.__name__} not supported")
    h = np.asarray(model.leverage)
    one_minus_h = np.clip(1 - h, 1e-12, None)
    d = np.asarray(model.residuals_of("deviance"))
    pe = np.asarray(model.residuals_of("pearson"))
    likelihood_r = np.sign(d) * np.sqrt(d**2 + (pe**2) * h / one_minus_h)
    sigma_loo = _loo_sigma_glm_gam(model)
    return likelihood_r / (sigma_loo * np.sqrt(one_minus_h))


def cooks_distance(model):
    """R: ``cooks.distance()`` — Cook's distance for each observation.

    Uses the unified formula
    ``D_i = r_i^2 · h_i / ((1 - h_i) · p)`` where ``r_i`` is the
    standardized residual (deviance for glm/gam/bam, ordinary for lm)
    and ``p`` is the effective parameter count. R's ``cooks.distance.lm``
    uses ``model.p``; ``cooks.distance.glm`` uses ``sum(hat)`` — we
    follow that split to match R numerically.
    """
    if model.__class__.__name__ in ("gmm", "Influence"):
        return np.asarray(model.cooks_distance())
    h = hatvalues(model)
    one_minus_h = np.clip(1 - h, 1e-12, None)
    if hasattr(model, "std_pearson_residuals"):  # glm / gam / bam
        r = np.asarray(model.std_pearson_residuals)
        p = float(np.sum(h))  # matches R's cooks.distance.glm
    elif hasattr(model, "std_residuals"):  # lm
        r = np.asarray(model.std_residuals)
        p = float(model.p)
    else:
        raise TypeError(f"cooks_distance(): {model.__class__.__name__} not supported")
    if p <= 0:
        raise ValueError("cooks_distance(): effective parameter count is zero")
    return r**2 * h / (one_minus_h * p)


def dffits(model):
    """R: ``dffits()``.

    For ``lm``, uses ``DFFITS_i = r_i^* · √(h_i / (1 - h_i))``.
    For ``glm`` / ``gam`` / ``bam``, follows ``stats:::dffits`` exactly:
    ``DFFITS_i = p_i · √(h_i) / (σ_(-i) · (1 - h_i))`` where ``p_i`` is
    the raw response-scale Pearson residual.
    """
    if hasattr(model, "std_residuals"):  # lm
        e = model.residuals.to_series().to_numpy()
        h = np.asarray(model.leverage)
        sigma_loo = _loo_sigma_lm(model)
        sqrt_w = np.sqrt(_lm_weights_array(model))
        one_minus_h = np.clip(1 - h, 1e-12, None)
        return e * sqrt_w * np.sqrt(h) / (sigma_loo * one_minus_h)

    if not hasattr(model, "residuals_of"):
        raise TypeError(f"dffits(): {model.__class__.__name__} not supported")
    h = np.asarray(model.leverage)
    one_minus_h = np.clip(1 - h, 1e-12, None)
    pe = np.asarray(model.residuals_of("pearson"))
    sigma_loo = _loo_sigma_glm_gam(model)
    return pe * np.sqrt(h) / (sigma_loo * one_minus_h)


def dfbetas(model):
    """R: ``dfbetas()`` — standardized leave-one-out coefficient changes.

    Returns an ``n × p`` polars DataFrame whose columns are the design
    columns (``(Intercept)``, predictors, …). Element ``[i, j]`` is the
    change in ``β̂_j`` when observation ``i`` is dropped, scaled by
    ``σ_(-i) · √(diag((X'X)^{-1})_j)``.

    For ``lm``: closed form using ``XtXinv``. For ``glm`` / ``gam`` /
    ``bam``: IRLS closed form using ``Vp/scale`` (or ``V_bhat/dispersion``)
    and IRLS working weights recovered from ``leverage``.
    """
    if model.__class__.__name__ == "Influence":  # deletion object
        return model.dfbetas()
    if hasattr(model, "XtXinv"):
        X = model.X.to_numpy().astype(float)
        XtXinv = np.asarray(model.XtXinv)
        e = model.residuals.to_series().to_numpy()
        h = hatvalues(model)
        one_minus_h = np.clip(1 - h, 1e-12, None)
        w = _lm_weights_array(model)
        sigma_loo = _loo_sigma_lm(model)
        delta = (X @ XtXinv) * (w * e / one_minus_h)[:, None]
        sd_j = np.sqrt(np.diag(XtXinv))
        sd_j = np.where(sd_j > 0, sd_j, 1.0)
        out = delta / (sigma_loo[:, None] * sd_j[None, :])
        return pl.DataFrame(
            {col: out[:, i] for i, col in enumerate(model.column_names)}
        )

    if not hasattr(model, "residuals_of"):
        raise TypeError(f"dfbetas(): {model.__class__.__name__} not supported")
    inputs = _irls_inputs(model)
    X, XtWXinv = inputs["X"], inputs["XtWXinv"]
    w_irls = inputs["w_irls"]
    working_resid = inputs["working_resid"]
    h = inputs["h"]
    sigma_loo = inputs["sigma_loo"]
    one_minus_h = np.clip(1 - h, 1e-12, None)

    delta = (X @ XtWXinv) * (w_irls * working_resid / one_minus_h)[:, None]
    sd_j = np.sqrt(np.diag(XtWXinv))
    sd_j = np.where(sd_j > 0, sd_j, 1.0)
    out = delta / (sigma_loo[:, None] * sd_j[None, :])
    return pl.DataFrame({col: out[:, i] for i, col in enumerate(model.column_names)})


def dfbeta(model):
    """R: ``dfbeta()`` — *unstandardized* leave-one-out coefficient changes.

    Returns an ``n × p`` polars DataFrame; element ``[i, j]`` is
    ``β̂_j − β̂_j(−i)``, the raw change in coefficient ``j`` when observation
    ``i`` is dropped. This is exactly :func:`dfbetas` *before* the
    ``σ_(−i)·√diag((X'X)⁻¹)`` standardization. lm (closed form via ``XtXinv``)
    and glm/gam/bam (IRLS closed form), mirroring ``dfbetas``.

    For an :class:`hea.models.gmm.Influence` deletion object, returns lme4's
    ``dfbeta.influence.merMod`` (``β̂[-i] − β̂``, the *deleted minus full*
    convention — opposite sign to ``dfbeta.lm``).
    """
    if model.__class__.__name__ == "Influence":  # deletion object
        return model.dfbeta()
    if hasattr(model, "XtXinv"):  # lm
        X = model.X.to_numpy().astype(float)
        XtXinv = np.asarray(model.XtXinv)
        e = model.residuals.to_series().to_numpy()
        one_minus_h = np.clip(1 - hatvalues(model), 1e-12, None)
        w = _lm_weights_array(model)
        delta = (X @ XtXinv) * (w * e / one_minus_h)[:, None]
        return pl.DataFrame(
            {col: delta[:, i] for i, col in enumerate(model.column_names)}
        )

    if not hasattr(model, "residuals_of"):
        raise TypeError(f"dfbeta(): {model.__class__.__name__} not supported")
    inputs = _irls_inputs(model)
    X, XtWXinv = inputs["X"], inputs["XtWXinv"]
    one_minus_h = np.clip(1 - inputs["h"], 1e-12, None)
    delta = (X @ XtWXinv) * (inputs["w_irls"] * inputs["working_resid"] / one_minus_h)[
        :, None
    ]
    return pl.DataFrame({col: delta[:, i] for i, col in enumerate(model.column_names)})


def influence(model, groups=None, **kwargs):
    """R: ``influence()`` / ``lm.influence()`` — deletion diagnostics bundle.

    For a ``gmm`` (mixed model) this is ``influence.merMod``: each observation
    (``groups=None``) or grouping-factor level (``groups="<factor>"``) is
    deleted and the model refit, returning an :class:`hea.models.gmm.Influence`
    object (``.dfbeta()`` / ``.dfbetas()`` / ``.cooks_distance()``).

    For ``lm`` / ``glm`` / ``gam`` / ``bam`` returns a dict mirroring R's
    ``lm.influence(do.coef=TRUE)``:

    * ``hat`` — leverage ``h_ii`` (ndarray, length ``n``)
    * ``sigma`` — leave-one-out σ estimates ``σ_(-i)`` (ndarray, length ``n``)
    * ``coefficients`` — leave-one-out coefficient *deltas*
      ``β̂ - β̂_(-i)`` as an ``n × p`` DataFrame named like the design
    * ``residuals`` — for ``lm``, response residuals; for ``glm`` /
      ``gam`` / ``bam``, working residuals (matches R's
      ``influence.glm`` "wt.res")
    """
    if model.__class__.__name__ == "gmm":  # → Influence object
        return model.influence(groups=groups, **kwargs)
    if hasattr(model, "XtXinv"):
        X = model.X.to_numpy().astype(float)
        XtXinv = np.asarray(model.XtXinv)
        e = model.residuals.to_series().to_numpy()
        h = np.asarray(model.leverage)
        one_minus_h = np.clip(1 - h, 1e-12, None)
        w = _lm_weights_array(model)
        delta = (X @ XtXinv) * (w * e / one_minus_h)[:, None]
        return {
            "hat": h,
            "sigma": _loo_sigma_lm(model),
            "coefficients": pl.DataFrame(
                {col: delta[:, i] for i, col in enumerate(model.column_names)}
            ),
            "residuals": e,
        }

    if not hasattr(model, "residuals_of"):
        raise TypeError(f"influence(): {model.__class__.__name__} not supported")
    inputs = _irls_inputs(model)
    X, XtWXinv = inputs["X"], inputs["XtWXinv"]
    w_irls = inputs["w_irls"]
    working_resid = inputs["working_resid"]
    h = inputs["h"]
    sigma_loo = inputs["sigma_loo"]
    one_minus_h = np.clip(1 - h, 1e-12, None)
    delta = (X @ XtWXinv) * (w_irls * working_resid / one_minus_h)[:, None]
    return {
        "hat": h,
        "sigma": sigma_loo,
        "coefficients": pl.DataFrame(
            {col: delta[:, i] for i, col in enumerate(model.column_names)}
        ),
        "residuals": working_resid,
    }
