"""Free-function dispatch over hea's fitted model objects
(``lm`` / ``glm`` / ``gam`` / ``bam`` / ``gmm``).

Pure duck typing — no model-class imports needed at module load. Where R
has multiple aliases (``coef`` / ``coefficients``, ``resid`` / ``residuals``,
``fitted`` / ``fitted.values``), both are exposed.

Also hosts the formula-update bookkeeping (``terms`` / :class:`Terms`,
``update``, ``_merge_formula_vars_from_caller``) and the single-model
``AIC`` / ``BIC`` accessors plus their multi-model comparison-table forms.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import polars as pl

from ._shared import NamedVector, _caller_names


def _bhat_to_named_vector(model):
    """Build a ``NamedVector`` from a fitted model's ``.bhat`` row."""

    if not hasattr(model, "bhat") or not isinstance(model.bhat, pl.DataFrame):
        raise TypeError(f"{model.__class__.__name__} has no .bhat DataFrame")
    return NamedVector(model.bhat.columns, model.bhat.row(0))


def coef(model):
    """R: ``coef()`` — model coefficients.

    For ``lm`` / ``glm`` / ``gam`` / ``bam`` returns a
    :class:`hea.R.NamedVector` of the fitted coefficients — 0-based positional
    indexing (``coef(m)[0]`` is the first coefficient), name lookup
    (``coef(m)["x"]``), and elementwise arithmetic.

    For a ``gmm`` (mixed model) returns ``lme4::coef.merMod`` — a dict of
    per-group coefficient frames (the fixed effects plus the matching
    random-effect BLUP at each level of every grouping factor). Use
    :func:`fixef` for the fixed effects alone.
    """
    if model.__class__.__name__ == "gmm":
        return model.coef()
    return _bhat_to_named_vector(model)


def coefficients(model):
    """R alias for :func:`coef`."""
    return coef(model)


def fixef(model):
    """R: ``fixef()`` — fixed-effect coefficients as a named numeric vector.

    For a ``gmm`` these are the fixed effects β̂ only (unlike :func:`coef`,
    which adds the per-group BLUPs); for non-mixed models the two coincide.
    """
    return _bhat_to_named_vector(model)


def ranef(model, condVar=False, postVar=False, drop=False, whichel=None):
    """R: ``ranef(model, condVar=, postVar=, drop=, whichel=)`` — random
    effects (gmm only). ``postVar=True`` attaches the full per-level
    conditional-covariance arrays under ``.postVar``; ``drop=`` returns a
    level-named vector for scalar bars; ``whichel=`` selects grouping factors."""
    if hasattr(model, "ranef"):
        return model.ranef(condVar=condVar, postVar=postVar, drop=drop, whichel=whichel)
    raise TypeError(f"ranef(): {model.__class__.__name__} has no random effects")


def refitML(model):
    """R: ``refitML()`` — refit a REML-fitted LMM by ML (lme4's
    ``refitML.merMod``).

    A model already fitted by ML is returned unchanged — that covers any
    GLMM (``glmer`` is ML by construction, via the Laplace approximation) and
    an LMM fit with ``REML=False``. A REML LMM is refit with ``REML=False`` so
    its likelihood is comparable across models with different fixed effects
    (the reason ``anova`` / AIC need ML).
    """
    from ..models.gmm import gmm

    if not isinstance(model, gmm):
        raise TypeError(
            "refitML(): only mixed models (gmm) have a REML/ML distinction; "
            f"got {model.__class__.__name__}"
        )
    if not model.REML:
        return model  # already ML (ML-LMM or any GLMM)
    return gmm(model.formula, model.data, family=model.family, REML=False)


def refit(model, newresp=None):
    """R: ``refit()`` — refit a mixed model, optionally to a new response
    (lme4's ``refit.merMod``).

    With ``newresp=None`` the same model is re-fit (a fresh run of the
    optimizer). With a numeric ``newresp`` of length ``n`` the response column
    is replaced and the model is refit keeping the same formula, family, REML
    flag, and random-effect structure — the building block for parametric
    bootstrap / ``simulate`` (Phases 10–11). The response must be a bare data
    column (``cbind()`` / transformed LHS isn't supported yet).
    """
    from ..models.gmm import gmm

    if not isinstance(model, gmm):
        raise TypeError(
            f"refit(): only mixed models (gmm) are supported; "
            f"got {model.__class__.__name__}"
        )
    data = model.data
    if newresp is not None:
        resp = np.asarray(newresp, dtype=float).ravel()
        if resp.shape != (model.n,):
            raise ValueError(
                f"refit(): newresp must have length {model.n}; got {resp.shape}"
            )
        lhs = model.formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise NotImplementedError(
                f"refit(newresp=): response {lhs!r} is not a bare data column "
                f"(cbind() / transformed LHS not supported yet)"
            )
        data = data.with_columns(pl.Series(lhs, resp))
    return gmm(model.formula, data, family=model.family, REML=model.REML)


def resid(model, type=None, scaled=False):
    """R: ``resid()`` / ``residuals()`` — residuals as 1D ``ndarray``.

    For ``glm`` / ``gam`` / ``bam``, ``type`` selects among
    ``{"deviance"`` (default, matches R), ``"pearson"``, ``"working"``,
    ``"response"}``. For ``lm`` (matching ``residuals.lm``):
    ``"response"`` / ``"working"`` are the raw residuals and
    ``"pearson"`` / ``"deviance"`` are the weighted residuals ``√wᵢ·rᵢ``
    (equal to raw when unweighted). ``"partial"`` returns the
    component-plus-residual *matrix* (one column per RHS term,
    ``r + predict(type="terms")``) as a 2-D frame — see
    :func:`_lm_partial_residuals`.

    For a ``gmm`` (mixed model) ``type`` selects the same four scales
    (``residuals.merMod``); the default is ``"response"`` for an LMM and
    ``"deviance"`` for a GLMM (lme4's defaults). On the LMM path all four
    collapse to ``y − μ``. ``scaled=True`` divides by σ̂ (lme4's
    ``residuals(., scaled=TRUE)``) and is mixed-model only.
    """
    is_gmm = model.__class__.__name__ == "gmm"
    if scaled and not is_gmm:
        raise TypeError("resid(): scaled= is only supported for mixed models (gmm)")
    if hasattr(model, "residuals_of"):
        if type is None:
            is_glmm_fn = getattr(model, "_is_glmm", None)
            is_lmm = is_gmm and is_glmm_fn is not None and not is_glmm_fn()
            type = "response" if is_lmm else "deviance"
        if is_gmm:
            arr = np.asarray(model.residuals_of(type, scaled=scaled))
            pad = getattr(model, "_na_pad", None)  # na.exclude → pad to full len
            return pad(arr) if pad is not None else arr
        return model.residuals_of(type)
    r = getattr(model, "residuals", None)
    if isinstance(r, pl.DataFrame):
        raw = r.to_series().to_numpy()
    elif isinstance(r, np.ndarray):
        raw = r
    elif isinstance(r, pl.Series):
        raw = r.to_numpy()
    else:
        raise TypeError(f"resid(): {model.__class__.__name__} has no usable residuals")
    is_lm = hasattr(model, "_w")  # lm carries the prior-weight vector
    if type in (None, "response") or (is_lm and type == "working"):
        arr = raw
    elif is_lm and type in ("pearson", "deviance"):
        arr = raw * np.sqrt(model._w)
    elif is_lm and type == "partial":
        return _lm_partial_residuals(model, raw)
    else:
        allowed = (
            "'response' / None, 'working', 'pearson', 'deviance', 'partial'"
            if is_lm
            else "'response' / None"
        )
        raise ValueError(
            f"resid(): type={type!r} not supported for "
            f"{model.__class__.__name__} (only {allowed})"
        )
    pad = getattr(model, "_na_pad", None)
    return pad(arr) if pad is not None else arr


def residuals(model, type=None, scaled=False):
    """R alias for :func:`resid`."""
    return resid(model, type, scaled=scaled)


def _lm_partial_residuals(model, raw):
    """R: ``residuals.lm(type="partial")`` — component-plus-residual matrix."""
    terms = model._predict_terms()  # centered per-term contributions
    pad = getattr(model, "_na_pad", None)
    cols = {}
    for c in terms.columns:
        partial = raw + terms[c].to_numpy().astype(float)
        cols[c] = pad(partial) if pad is not None else partial
    from ..tidy import DataFrame as _DF  # local: avoid import cycle

    out = _DF(cols)
    out.constant = getattr(terms, "constant", 0.0)
    return out


def fitted(model):
    """R: ``fitted()`` — fitted values as 1D ``ndarray``.

    For lm/glm this is the response-scale prediction (μ̂); for gam/gmm
    same. Equivalent to ``model.predict()`` on the training data.
    """
    fv = getattr(model, "fitted_values", None)
    if fv is not None:
        arr = np.asarray(fv)
    else:
        f = getattr(model, "fitted", None)
        if f is not None and not callable(f):
            arr = np.asarray(f)
        else:
            yh = getattr(model, "yhat", None)
            if isinstance(yh, pl.DataFrame):
                col = "fit" if "fit" in yh.columns else yh.columns[0]
                arr = yh[col].to_numpy()
            elif isinstance(yh, np.ndarray):
                arr = yh
            else:
                raise TypeError(
                    f"fitted(): {model.__class__.__name__} has no fitted values"
                )
    pad = getattr(model, "_na_pad", None)
    return pad(arr) if pad is not None else arr


def fitted_values(model):
    """R alias for :func:`fitted`."""
    return fitted(model)


def predict(model, *args, **kwargs):
    """R: ``predict()`` — dispatches to ``model.predict(...)``.

    Forwards positional and keyword arguments untouched, so
    ``predict(m, newdata, interval="confidence")`` works exactly like
    the bound method.
    """
    if not hasattr(model, "predict"):
        raise TypeError(f"predict(): {model.__class__.__name__} has no .predict()")
    return model.predict(*args, **kwargs)


def confint(model, level=0.95, **kwargs):
    """R: ``confint()`` — confidence intervals.

    Returns a polars DataFrame with one row per parameter.

    Dispatch:

    * ``gmm`` (mixed model) / profile objects — defer to their ``.confint``.
      For ``gmm`` the lme4 keyword surface passes through: ``method`` ∈
      ``{"profile"(default),"Wald","boot"}``, plus ``parm``/``nsim``/
      ``boot_type``/``seed``/``FUN`` (see :meth:`hea.models.gmm.confint`).
    * ``lm`` — exact CIs at ``alpha = 1 - level`` via ``compute_ci_bhat``.
    * Other model types — return ``model.ci_bhat`` when ``level=0.95``;
      otherwise raise.
    """
    if (
        hasattr(model, "confint")
        and not hasattr(model, "ci_bhat")
        and not hasattr(model, "compute_ci_bhat")
    ):
        return model.confint(level=level, **kwargs)
    if not kwargs and level == 0.95 and hasattr(model, "ci_bhat"):
        return model.ci_bhat
    if not kwargs and hasattr(model, "compute_ci_bhat"):
        return model.compute_ci_bhat(alpha=1 - level)
    raise NotImplementedError(
        f"confint(): level={level} not supported for {model.__class__.__name__}"
    )


def profile(model, **kwargs):
    """R: ``profile.merMod`` — profile-likelihood object for a ``gmm`` fit.

    Thin delegator to :meth:`hea.models.gmm.profile`; the result's
    ``.confint(level=)`` inverts each ζ-curve. Raises for non-``gmm`` models.
    """
    if not hasattr(model, "profile"):
        raise TypeError(
            f"profile(): {model.__class__.__name__} has no profile method "
            f"(only mixed models / gmm)"
        )
    return model.profile(**kwargs)


def bootMer(x, FUN, **kwargs):
    """R: ``bootMer(x, FUN, ...)`` — parametric bootstrap of a ``gmm`` fit.

    Delegates to :meth:`hea.models.gmm.bootMer` (simulate → refit → ``FUN``).
    Keyword args mirror lme4: ``nsim``/``seed``/``use_u``/``type``/
    ``parallel``/``ncpus``. Returns a :class:`hea.models.gmm.BootMer` whose
    ``.confint(type=...)`` gives perc/basic/norm intervals.
    """
    if not hasattr(x, "bootMer"):
        raise TypeError(f"bootMer(): {x.__class__.__name__} is not a mixed model (gmm)")
    return x.bootMer(FUN, **kwargs)


def vcov(model, correlation=False, full=False, use_hessian=None):
    """R: ``vcov()`` — variance-covariance matrix of the coefficients.

    Return type varies by model: lm/glm return ``ndarray`` (``V_bhat``);
    gam/bam return ``ndarray`` (``Vp``, the Bayesian posterior); gmm
    returns a polars ``DataFrame`` (fixed effects only). For a gmm,
    ``correlation=True`` returns the correlation matrix and ``full=True`` the
    joint ``[b̂; β̂]`` conditional covariance (``vcov.merMod`` forms); both are
    mixed-model only.
    """
    if model.__class__.__name__ == "gmm":
        return model.vcov(correlation=correlation, full=full, use_hessian=use_hessian)
    if correlation or full:
        raise TypeError(
            "vcov(): correlation=/full= are only supported for mixed models (gmm)"
        )
    if hasattr(model, "Vp"):  # gam / bam (Bayesian posterior)
        return model.Vp
    if hasattr(model, "V_bhat"):  # lm / glm
        return model.V_bhat
    raise TypeError(f"vcov(): {model.__class__.__name__} not supported")


def logLik(model, REML=None):
    """R: ``logLik()`` — model log-likelihood.

    For a ``gmm`` (mixed model) defers to ``logLik.merMod``: ``REML=None``
    uses the fit's own criterion, while ``REML=True``/``False`` recomputes the
    other criterion at the fitted θ̂ (no refit). ``REML=`` is only meaningful
    for a ``gmm``.
    """
    if model.__class__.__name__ == "gmm":
        return model.logLik(REML=REML)
    if REML is not None:
        raise TypeError("logLik(): REML= is only meaningful for mixed models (gmm)")
    if hasattr(model, "loglike"):
        return float(model.loglike)
    if hasattr(model, "REML_criterion"):
        return -float(model.REML_criterion) / 2.0
    raise TypeError(f"logLik(): {model.__class__.__name__} has no log-likelihood")


def _require_gmm(model, fn):
    """Raise unless ``model`` is a ``gmm`` — for the merMod-only generics."""
    if model.__class__.__name__ != "gmm":
        raise TypeError(
            f"{fn}(): only mixed models (gmm) are supported; "
            f"got {model.__class__.__name__}"
        )


def VarCorr(model):
    """R: ``VarCorr()`` — estimated random-effect (co)variances of a mixed
    model. Returns the :class:`hea.models.gmm.VarCorr` object (per-bar
    covariance with stddev / correlation views and residual SD ``sc``; prints
    in lme4's Groups / Name / Std.Dev. / Corr layout)."""
    _require_gmm(model, "VarCorr")
    return model.VarCorr()


def getME(model, name):
    """R: ``getME(object, name)`` — extract a named component of a fitted mixed
    model (design matrices, θ/β, Λ/L, dims, …). See
    :meth:`hea.models.gmm.gmm.getME` for the supported names."""
    _require_gmm(model, "getME")
    return model.getME(name)


def isREML(model):
    """R: ``isREML()`` — ``True`` only for a REML-fit LMM (a GLMM or ML LMM is
    ``False``)."""
    _require_gmm(model, "isREML")
    return model.isREML()


def isLMM(model):
    """R: ``isLMM()`` — ``True`` for a linear mixed model (Gaussian/identity)."""
    _require_gmm(model, "isLMM")
    return model.isLMM()


def isGLMM(model):
    """R: ``isGLMM()`` — ``True`` for a generalized linear mixed model."""
    _require_gmm(model, "isGLMM")
    return model.isGLMM()


def isNLMM(model):
    """R: ``isNLMM()`` — ``True`` for a nonlinear mixed model (always ``False``
    in hea; there is no NLMM path)."""
    _require_gmm(model, "isNLMM")
    return model.isNLMM()


def isSingular(model, tol=1e-4):
    """R: ``isSingular(x, tol=1e-4)`` — ``True`` if the fit sits on the boundary
    of the feasible θ region (a variance driven to ~0 or a ±1 correlation)."""
    _require_gmm(model, "isSingular")
    return model.isSingular(tol=tol)


def getData(model):
    """R: ``getData()`` — the data frame the mixed model was fit to."""
    _require_gmm(model, "getData")
    return model.getData()


def extractAIC(model, scale=0, k=2):
    """R: ``extractAIC()`` — ``(edf, AIC)`` for a mixed model, with
    ``AIC = -2·logLik + k·edf`` on the fit's own criterion (``extractAIC.merMod``).
    Only mixed models are handled here (lm/glm use the Mallows-style formula
    inside :mod:`hea.R.model_selection`)."""
    _require_gmm(model, "extractAIC")
    return model.extractAIC(scale=scale, k=k)


def rePCA(model):
    """R: ``rePCA()`` — principal-component SDs of each grouping factor's
    relative random-effect covariance (a degeneracy / over-parameterization
    diagnostic; a near-zero component flags a singular RE term)."""
    _require_gmm(model, "rePCA")
    return model.rePCA()


def deviance(model):
    """R: ``deviance()`` — model deviance.

    For ``lm`` (no Gaussian deviance attribute), returns ``rss`` —
    matches ``deviance.lm = sum(residuals^2)``.
    """
    if hasattr(model, "deviance") and not callable(model.deviance):
        return float(model.deviance)
    if hasattr(model, "rss"):  # lm
        return float(model.rss)
    raise TypeError(f"deviance(): {model.__class__.__name__} has no deviance")


def nobs(model):
    """R: ``nobs()`` — number of observations used to fit.

    For a weighted ``lm``, R's ``nobs.lm`` counts only non-zero-weight rows
    (``sum(w != 0)``) — ``model._n_eff`` already holds that count (and equals
    ``n`` when unweighted or all weights are positive). Models without an
    ``_n_eff`` attribute fall back to the full row count ``n``.
    """
    n_eff = getattr(model, "_n_eff", None)
    return int(n_eff if n_eff is not None else model.n)


def weights(model):
    """R: ``weights()`` — the prior weights used in fitting.

    Returns the (subset / NA-aligned) weight vector as an ``ndarray``, or
    ``None`` for an unweighted ``lm``/``gmm`` fit — matching R's
    ``weights(m)`` (``NULL`` when no weights were supplied). For ``glm`` the
    prior weights (ones by default) are returned.
    """
    for attr in ("weights", "_prior_w", "prior_weights"):
        w = getattr(model, attr, None)
        if w is not None:
            return np.asarray(w)
    return None


def effects(model):
    """R: ``effects()`` — orthogonal single-degree-of-freedom effects ``Q'y``.

    A length-``n`` vector from the fit's QR: the first ``rank`` entries are the
    named regression effects, the remainder the residual effects (names beyond
    the rank are ``""``, as in R). Read off the ``effects`` component the fit
    already stored (``Qᵀ(√w·y)`` on the offset-adjusted response, ``lm.wfit``).
    lm only.
    """
    eff = getattr(model, "effects", None)
    if eff is None or not hasattr(model, "column_names"):
        raise TypeError(
            f"effects(): {model.__class__.__name__} has no QR/effects component"
        )
    eff = np.asarray(eff, dtype=float).reshape(-1)
    pad = len(eff) - len(model.column_names)
    names = list(model.column_names) + [""] * pad
    return NamedVector(names, eff)


def simulate(model, nsim=1, seed=None, **kwargs):
    """R: ``simulate`` — draw ``nsim`` response vectors from the fitted model.

    For a ``gmm`` (mixed model) this delegates to
    :meth:`hea.models.gmm.simulate` — ``simulate.merMod`` (RE draws then
    per-family draws, ``use_u=`` for conditional simulation). For ``lm`` /
    gaussian it's ``simulate.lm``: ``ŷ + N(0, σ²)`` with
    ``σ² = deviance/df.residual`` (per-row ``σ²/wᵢ`` for a weighted fit).

    Pass ``seed=`` to reproduce R's ``set.seed(seed); simulate(object, nsim)``
    bit-for-bit — draws come from hea's R Mersenne-Twister
    (:class:`hea.R.rng.RMersenneTwister`), **not** numpy. ``seed=None`` picks a
    fresh seed. Returns a polars DataFrame with columns ``sim_1`` … ``sim_{nsim}``.
    """
    from .rng import RMersenneTwister

    if hasattr(model, "simulate") and model.__class__.__name__ == "gmm":
        return model.simulate(nsim=nsim, seed=seed, **kwargs)

    ftd = np.asarray(fitted(model), dtype=float).reshape(-1)
    n = ftd.shape[0]
    var0 = float(deviance(model)) / float(df_residual(model))
    w = weights(model)
    sd_vec = (
        np.full(n, np.sqrt(var0))
        if w is None
        else np.sqrt(var0 / np.asarray(w, dtype=float))
    )
    if seed is None:
        import random

        seed = random.Random().randint(0, 2**31 - 1)
    rng = RMersenneTwister(int(seed))
    z = rng.rnorm(int(n * nsim))
    draws = z * np.tile(sd_vec, int(nsim))
    return pl.DataFrame(
        {f"sim_{i + 1}": ftd + draws[i * n : (i + 1) * n] for i in range(int(nsim))}
    )


def df_residual(model):
    """R: ``df.residual()`` — residual degrees of freedom."""
    for attr in ("df_residual", "df_residuals", "df_resid"):
        v = getattr(model, attr, None)
        if v is not None:
            return float(v)
    raise TypeError(f"df_residual(): {model.__class__.__name__} has no residual df")


def formula(model):
    """R: ``formula()`` — extract the model formula (string)."""
    return model.formula


def model_matrix(model, data=None):
    """R: ``model.matrix(model_or_formula, data=df)`` — design matrix.

    Two forms:

    - ``model_matrix(fitted_model)`` — return the design matrix already
      stored on the fitted model.
    - ``model_matrix(formula_str, data=df)`` — build a design matrix
      from the formula against ``df``. Mirrors R's bare-formula form.

    Returns a polars DataFrame; columns are the named design columns
    (intercept, dummy-coded factor levels, spline bases, …). R returns
    an unnamed numeric matrix; we keep the names attached.
    """
    if hasattr(model, "X"):
        return model.X
    if isinstance(model, str) and data is not None:
        from ..formula import prepare_design

        design = prepare_design(model, data)
        return design.X
    raise TypeError(
        f"model_matrix(): {model.__class__.__name__} has no design matrix; "
        f"for the formula form pass data= explicitly."
    )


def model_frame(model):
    """R: ``model.frame()`` — original data passed at fit time."""
    return model.data


@dataclass
class Terms:
    """Lightweight stand-in for R's ``terms`` object.

    R's ``terms`` carries a factor matrix and many attributes; we expose
    only what hea actually keeps around: the formula string, the
    response (LHS) variable name, and the top-level term labels (the
    same list ``aov`` / ``anova`` use to build their tables).
    """

    formula: str
    response: str
    term_labels: list

    def __repr__(self) -> str:
        return (
            f"Terms(formula={self.formula!r}, response={self.response!r}, "
            f"term_labels={self.term_labels!r})"
        )


def terms(model) -> Terms:
    """R: ``terms()`` — formula structure summary.

    Returns a :class:`Terms` with the formula string, response name, and
    top-level term labels. Less than R's full terms object (no factor
    matrix, no order vector) but enough to drive things like ``anova``
    table titles or to round-trip a formula via ``update``.
    """
    f = model.formula
    if "~" not in f:
        raise ValueError(f"terms(): bad formula on {model.__class__.__name__}")
    lhs, rhs = f.split("~", 1)
    response = lhs.strip()
    if hasattr(model, "_expanded") and hasattr(model._expanded, "term_labels"):
        labels = list(model._expanded.term_labels)
    else:
        labels = [t.strip() for t in rhs.split("+") if t.strip()]
    return Terms(formula=f, response=response, term_labels=labels)


def variable_names(model, full=False):
    """R: ``variable.names()`` — the model's coefficient (design-column) names.

    ``full=False`` (default) returns the estimable/kept columns (the fit's
    rank); ``full=True`` returns every design column including aliased ones
    (R's ``dimnames(qr$qr)[[2]]``).
    """
    if full and hasattr(model, "_full_names"):
        return list(model._full_names)
    return list(model.column_names)


def labels(model):
    """R: ``labels.lm()`` — the top-level term labels (RHS terms, no intercept)."""
    return list(terms(model).term_labels)


def case_names(model, full=False):
    """R: ``case.names()`` — observation labels as strings.

    0-based row indices (hea's convention; R uses 1-based). ``full=False``
    (default) drops zero-weight rows for a weighted fit — R parity, the rows
    that actually entered the fit — while ``full=True`` keeps every row.
    """
    n = int(model.n)
    names = [str(i) for i in range(n)]
    if full:
        return names
    w = weights(model)
    if w is None:
        return names
    w = np.asarray(w)
    return [names[i] for i in range(n) if w[i] != 0]


_UPDATE_AUTO_FORWARD = ("family", "method", "weights", "REML")


def update(model, formula=None, **kwargs):
    """R: ``update()`` — refit with a new formula and/or different args.

    ``formula`` is optional, matching R's ``update(object, formula. = .)``
    default: when omitted, the original ``model.formula`` is reused
    verbatim, so ``update(fm, REML=False)`` just refits with one knob
    changed. When supplied, two forms are recognised:

    * **Full formula** (e.g. ``"y ~ x1 + x2"``) — used verbatim.
    * **Delta formula** with R's ``.`` placeholder (e.g.
      ``". ~ . + x3"`` or ``"log(y) ~ . - x1"``). On each side of
      ``~``, ``.`` is substituted with the corresponding side of the
      original ``model.formula`` wrapped in parentheses, so terms can
      be added or removed without retyping.

    Constructor kwargs auto-forwarded (when the model class accepts the
    name AND the model exposes a non-``None`` public attribute):
    ``family`` (glm/gam/bam), ``method`` (lm/gam/bam), ``weights`` (lm),
    ``REML`` (gmm). User-supplied ``**kwargs`` always override the
    auto-forward. Anything not on this list (``offset``, ``sp``,
    ``select``, ``control``, …) must be passed explicitly if needed —
    forwarding ``sp`` for example would tie the new fit's smoothing
    parameters to the old formula's smooth structure.
    """
    if formula is None:
        f = model.formula
    else:
        f = formula.strip()
        if "~" not in f:
            raise ValueError(f"update(): formula must contain '~'; got {f!r}")
        if "." in f:
            old_lhs, old_rhs = (s.strip() for s in model.formula.split("~", 1))
            new_lhs, new_rhs = (s.strip() for s in f.split("~", 1))
            if new_lhs == ".":
                new_lhs = old_lhs
            elif "." in new_lhs:
                new_lhs = new_lhs.replace(".", f"({old_lhs})")
            if new_rhs == ".":
                new_rhs = old_rhs
            elif "." in new_rhs:
                new_rhs = new_rhs.replace(".", f"({old_rhs})")
            f = f"{new_lhs} ~ {new_rhs}"
    cls = type(model)
    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set()
    for name in _UPDATE_AUTO_FORWARD:
        if name in kwargs or name not in accepted:
            continue
        v = getattr(model, name, None)
        if v is None or callable(v):
            continue
        kwargs[name] = v
    data = _merge_formula_vars_from_caller(f, model.data, inspect.currentframe().f_back)
    return cls(f, data, **kwargs)


def _merge_formula_vars_from_caller(
    formula: str, data: pl.DataFrame, frame
) -> pl.DataFrame:
    """Find identifier-shaped names in ``formula`` that aren't columns of
    ``data``; pull each from ``frame``'s locals/globals if a length-match
    vector is bound there; return ``data`` augmented with those columns.
    """
    import re

    if frame is None:
        return data
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_.]*\b", formula))
    missing = [n for n in names if n not in data.columns and "." not in n]
    if not missing:
        return data
    ns = {**frame.f_globals, **frame.f_locals}
    add: dict[str, list] = {}
    n_rows = data.height
    for name in missing:
        if name not in ns:
            continue
        val = ns[name]
        try:
            arr = np.asarray(val).ravel()
        except Exception:  # noqa: BLE001, S112
            continue
        if arr.size == n_rows:
            add[name] = arr.tolist()
    if not add:
        return data
    return data.with_columns([pl.Series(k, v) for k, v in add.items()])


def AIC(*models):
    """R: ``AIC()`` — scalar for one model, comparison table for many.

    With one argument, returns ``model.AIC`` as a float. With two or
    more, returns a polars DataFrame with row labels recovered from the
    caller's variable names (R-style), plus columns ``df`` and ``AIC``.

    Note: ``hea.AIC`` (without the ``from hea.R import *``) prints the
    table and returns ``None``. This R-style version always returns.
    """
    if not models:
        raise TypeError("AIC(): need at least one model")
    if len(models) == 1:
        return float(models[0].AIC)
    names = _caller_names(models, inspect.currentframe().f_back)
    return pl.DataFrame(
        {
            "": names,
            "df": [m.npar for m in models],
            "AIC": [float(m.AIC) for m in models],
        }
    )


def BIC(*models):
    """R: ``BIC()`` — scalar for one model, comparison table for many.

    Same convention as :func:`AIC`.
    """
    if not models:
        raise TypeError("BIC(): need at least one model")
    if len(models) == 1:
        return float(models[0].BIC)
    names = _caller_names(models, inspect.currentframe().f_back)
    return pl.DataFrame(
        {
            "": names,
            "df": [m.npar for m in models],
            "BIC": [float(m.BIC) for m in models],
        }
    )
