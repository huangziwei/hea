"""Free-function dispatch over hea's fitted model objects
(``lm`` / ``glm`` / ``gam`` / ``bam`` / ``lme``).

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
        raise TypeError(
            f"{model.__class__.__name__} has no .bhat DataFrame"
        )
    return NamedVector(model.bhat.columns, model.bhat.row(0))


def coef(model):
    """R: ``coef()`` — coefficients as a named numeric vector.

    Works for ``lm`` / ``glm`` / ``gam`` / ``bam`` / ``lme``. For ``lme``
    this returns FIXED effects only (= R's ``fixef(m)``); R's
    ``coef.lmerMod`` returns per-group BLUPs which hea doesn't compute
    in the same shape — use ``fixef()`` + ``ranef()`` to assemble.

    Returns :class:`hea.R.NamedVector` — supports 0-based positional
    indexing (``coef(m)[0]`` is the first coefficient), name lookup
    (``coef(m)["x"]``), and elementwise arithmetic.
    """
    return _bhat_to_named_vector(model)


def coefficients(model):
    """R alias for :func:`coef`."""
    return coef(model)


def fixef(model):
    """R: ``fixef()`` — fixed-effect coefficients (lme).

    For non-mixed models, identical to :func:`coef`.
    """
    return coef(model)


def ranef(model):
    """R: ``ranef()`` — random effects (lme only)."""
    if hasattr(model, "ranef"):
        return model.ranef
    raise TypeError(
        f"ranef(): {model.__class__.__name__} has no random effects"
    )


def resid(model, type=None):
    """R: ``resid()`` / ``residuals()`` — residuals as 1D ``ndarray``.

    For ``glm`` / ``gam`` / ``bam``, ``type`` selects among
    ``{"deviance"`` (default, matches R), ``"pearson"``, ``"working"``,
    ``"response"}``. ``lm`` and ``lme`` only have response residuals;
    pass ``type=None`` or ``"response"`` (anything else raises).
    """
    if hasattr(model, "residuals_of"):
        return model.residuals_of(type or "deviance")
    if type not in (None, "response"):
        raise ValueError(
            f"resid(): type={type!r} not supported for "
            f"{model.__class__.__name__} (only 'response' / None)"
        )
    r = getattr(model, "residuals", None)
    if isinstance(r, pl.DataFrame):
        return r.to_series().to_numpy()
    if isinstance(r, np.ndarray):
        return r
    if isinstance(r, pl.Series):
        return r.to_numpy()
    raise TypeError(
        f"resid(): {model.__class__.__name__} has no usable residuals"
    )


def residuals(model, type=None):
    """R alias for :func:`resid`."""
    return resid(model, type)


def fitted(model):
    """R: ``fitted()`` — fitted values as 1D ``ndarray``.

    For lm/glm this is the response-scale prediction (μ̂); for gam/lme
    same. Equivalent to ``model.predict()`` on the training data.
    """
    fv = getattr(model, "fitted_values", None)
    if fv is not None:
        return np.asarray(fv)
    f = getattr(model, "fitted", None)
    if f is not None and not callable(f):
        return np.asarray(f)
    yh = getattr(model, "yhat", None)
    if isinstance(yh, pl.DataFrame):
        col = "fit" if "fit" in yh.columns else yh.columns[0]
        return yh[col].to_numpy()
    if isinstance(yh, np.ndarray):
        return yh
    raise TypeError(
        f"fitted(): {model.__class__.__name__} has no fitted values"
    )


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
        raise TypeError(
            f"predict(): {model.__class__.__name__} has no .predict()"
        )
    return model.predict(*args, **kwargs)


def confint(model, level=0.95):
    """R: ``confint()`` — confidence intervals.

    Returns a polars DataFrame with one row per parameter.

    Dispatch:

    * Profile-likelihood objects (``lme.profile()`` output) — defer to
      :meth:`hea.lme.Profile.confint`, which inverts each ζ-curve at
      ``±Φ⁻¹((1+level)/2)``. This is the lme4 ``confint(profile(fm))``
      workflow.
    * ``lm`` — exact CIs at ``alpha = 1 - level`` via
      ``compute_ci_bhat``.
    * Other model types — return ``model.ci_bhat`` when ``level=0.95``;
      otherwise raise.
    """
    # Profile objects expose their own ``confint`` — use it. Mirrors R's
    # S3 ``confint.profile`` dispatch.
    if hasattr(model, "confint") and not hasattr(model, "ci_bhat") \
            and not hasattr(model, "compute_ci_bhat"):
        return model.confint(level=level)
    if level == 0.95 and hasattr(model, "ci_bhat"):
        return model.ci_bhat
    if hasattr(model, "compute_ci_bhat"):
        return model.compute_ci_bhat(alpha=1 - level)
    raise NotImplementedError(
        f"confint(): level={level} not supported for "
        f"{model.__class__.__name__}"
    )


def vcov(model):
    """R: ``vcov()`` — variance-covariance matrix of the coefficients.

    Return type varies by model: lm/glm return ``ndarray`` (``V_bhat``);
    gam/bam return ``ndarray`` (``Vp``, the Bayesian posterior); lme
    returns a polars ``DataFrame`` (``vcov_beta``, fixed effects only).
    """
    if hasattr(model, "vcov_beta"):  # lme
        return model.vcov_beta
    if hasattr(model, "Vp"):  # gam / bam (Bayesian posterior)
        return model.Vp
    if hasattr(model, "V_bhat"):  # lm / glm
        return model.V_bhat
    raise TypeError(
        f"vcov(): {model.__class__.__name__} not supported"
    )


def logLik(model):
    """R: ``logLik()`` — model log-likelihood.

    For REML-fit ``lme`` (no plain ``loglike``), returns the REML
    log-likelihood ``-REML_criterion / 2``, matching ``logLik.lmerMod``.
    """
    if hasattr(model, "loglike"):
        return float(model.loglike)
    if hasattr(model, "REML_criterion"):
        return -float(model.REML_criterion) / 2.0
    raise TypeError(
        f"logLik(): {model.__class__.__name__} has no log-likelihood"
    )


def deviance(model):
    """R: ``deviance()`` — model deviance.

    For ``lm`` (no Gaussian deviance attribute), returns ``rss`` —
    matches ``deviance.lm = sum(residuals^2)``.
    """
    if hasattr(model, "deviance") and not callable(model.deviance):
        return float(model.deviance)
    if hasattr(model, "rss"):  # lm
        return float(model.rss)
    raise TypeError(
        f"deviance(): {model.__class__.__name__} has no deviance"
    )


def nobs(model):
    """R: ``nobs()`` — number of observations used to fit."""
    return int(model.n)


def df_residual(model):
    """R: ``df.residual()`` — residual degrees of freedom."""
    for attr in ("df_residual", "df_residuals", "df_resid"):
        v = getattr(model, attr, None)
        if v is not None:
            return float(v)
    raise TypeError(
        f"df_residual(): {model.__class__.__name__} has no residual df"
    )


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
        # Formula form: import locally to avoid circular import at module load.
        from ..design import prepare_design

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
    ``REML`` (lme). User-supplied ``**kwargs`` always override the
    auto-forward. Anything not on this list (``offset``, ``sp``,
    ``select``, ``control``, …) must be passed explicitly if needed —
    forwarding ``sp`` for example would tie the new fit's smoothing
    parameters to the old formula's smooth structure.
    """
    if formula is None:
        # R's default `formula. = .` — reuse the original verbatim.
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
    # If the new formula references names that aren't in ``model.data``,
    # look them up in the caller's frame (R's ``update()`` evaluates the
    # formula in the parent environment, so locally-computed vectors
    # like ``ab`` in ``update(m, .~. + ab)`` are picked up automatically).
    data = _merge_formula_vars_from_caller(f, model.data, inspect.currentframe().f_back)
    return cls(f, data, **kwargs)


def _merge_formula_vars_from_caller(formula: str, data: pl.DataFrame, frame) -> pl.DataFrame:
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
        except Exception:
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
    return pl.DataFrame({
        "":    names,
        "df":  [m.npar for m in models],
        "AIC": [float(m.AIC) for m in models],
    })


def BIC(*models):
    """R: ``BIC()`` — scalar for one model, comparison table for many.

    Same convention as :func:`AIC`.
    """
    if not models:
        raise TypeError("BIC(): need at least one model")
    if len(models) == 1:
        return float(models[0].BIC)
    names = _caller_names(models, inspect.currentframe().f_back)
    return pl.DataFrame({
        "":    names,
        "df":  [m.npar for m in models],
        "BIC": [float(m.BIC) for m in models],
    })
