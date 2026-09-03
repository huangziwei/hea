"""R's model-comparison verbs: ``anova``, ``add1`` / ``drop1`` / ``step``.

Implements the per-model-family dispatch: ``_anova_lm`` /
``_anova_lm_single`` for ordinary least squares, ``_anova_glm`` /
``_anova_glm_table`` for generalized linear models (with auto-pick of F
vs Chisq tests), ``_anova_gam`` / ``_anova_gam_single`` /
``_anova_gam_table`` for mgcv-style GAM fits (uses ``edf1`` residual df),
and ``_anova_gmm`` for the LRT comparison between nested ``gmm`` fits
(with silent ML refit when REML inputs are passed).

``step()`` minimizes the Mallows-style extractAIC formula (see
``_extract_aic_lm`` / ``_step_aic``). ``add1`` / ``drop1`` respect
*marginality* via ``_drop_scope`` and ``_add_scope`` so interactions
don't get dropped while main effects remain (and vice versa).
"""

from __future__ import annotations

import inspect
import itertools

import numpy as np
import polars as pl

from ..formula import deparse
from ..models.gam import gam
from ..models.glm import glm
from ..models.gmm import gmm
from ..models.lm import lm
from ..utils import (
    _dig_tst,
    format_df,
    format_pval,
    format_signif,
    significance_code,
)
from . import distributions as _dist
from ._shared import _caller_names


def anova(
    *models,
    test: str | None = None,
    freq: bool = False,
    dispersion: float | None = None,
):
    """Compare nested fits, or decompose a single fit by Type-I SS.

    - One ``lm`` → sequential (Type I) ANOVA table, splitting the model's
      total SS into incremental contributions per RHS term in formula
      order. Mirrors R's ``anova(m)`` for a single ``lm``.
    - Multiple ``lm`` fits → F-test ANOVA table (incremental for 3+).
    - Multiple ``glm`` fits → analysis-of-deviance table (incremental for
      3+); ``test=`` selects the test statistic (see below).
    - Multiple ``gmm`` fits → likelihood-ratio test (lme4-style, incremental
      for 3+). REML fits are internally refit by ML before the LRT.

    Parameters
    ----------
    test : {"Chisq", "LRT", "F", "Rao", None}, optional
        Only meaningful for ``glm`` comparisons. ``None`` (default) auto-
        picks ``"Chisq"`` for scale-known families (Poisson, Binomial) and
        ``"F"`` for unknown-scale (Gaussian, Gamma, IG), matching R's
        ``anova.glm`` recommendation. ``"LRT"`` is an alias for ``"Chisq"``.
        ``"Rao"`` (score test) is not implemented yet. For ``lm`` and ``gmm``
        the test is fixed (always F / Chisq LRT respectively); passing
        ``test=`` for those raises.
    freq, dispersion : single-``gam`` form only
        mgcv's ``anova.gam(object, dispersion=, freq=)`` passthrough to
        the summary tables: ``freq=True`` uses the frequentist ``Ve``
        for the parametric Terms table; ``dispersion=`` overrides the
        scale (known-scale Chi.sq forms throughout).

    For multi-model calls rows are sorted by parameter count (smaller
    model first), matching R's ``anova``. Row labels are recovered from
    the caller's variable names (R-style); falls back to ``model i`` for
    unbound or aliased arguments, preserving *input* order.
    """
    if len(models) == 0:
        raise TypeError("anova(): need at least one model")
    if (freq or dispersion is not None) and not (
        len(models) == 1 and isinstance(models[0], gam)
    ):
        raise TypeError("anova(): freq=/dispersion= apply to the single-gam form only")
    if len(models) == 1:
        m = models[0]
        if isinstance(m, gam):
            if test is not None:
                raise TypeError("anova(gam): test= is not accepted")
            return _anova_gam_single(m, freq=freq, dispersion=dispersion)
        if isinstance(m, lm) and not isinstance(m, glm):
            if test is not None:
                raise TypeError("anova(lm): test= is not accepted (always F)")
            return _anova_lm_single(m)
        if isinstance(m, gmm):
            if test is not None:
                raise TypeError("anova(gmm): test= is not accepted (sequential F)")
            return _anova_gmm_single(m)
        raise TypeError(
            "anova(m): single-model form supports lm, gam and gmm only "
            f"(got {type(m).__name__})"
        )
    labels = _caller_names(models, inspect.currentframe().f_back)
    if all(isinstance(m, gmm) for m in models):
        if test is not None and test.upper() not in ("CHISQ", "LRT"):
            raise ValueError(
                f"anova(gmm): only test='Chisq'/'LRT' (the default LRT) "
                f"is supported, got {test!r}"
            )
        return _anova_gmm(*models, labels=labels)
    if all(isinstance(m, gam) for m in models):
        return _anova_gam(*models, labels=labels, test=test)
    if all(isinstance(m, glm) for m in models):
        return _anova_glm(*models, labels=labels, test=test)
    if all(isinstance(m, lm) for m in models):
        if test is not None:
            raise TypeError("anova(lm): test= is not accepted (always F)")
        return _anova_lm(*models, labels=labels)
    raise TypeError("anova(): all models must be the same type (lm, glm, gam, or gmm)")


def drop1(model, *, test: str | None = None, k: float = 2.0):
    """Single-term deletions, R's ``drop1.lm`` / ``drop1.glm`` / ``drop1.merMod``.

    For each non-intercept term in ``model``, refits with that term
    removed and prints a one-row-per-term table comparing each reduced
    fit to the full model (the ``<none>`` row).

    Conventions match R:

    * ``lm``: AIC column uses ``extractAIC``'s Mallows-style formula
      ``n*log(RSS/n) + k*p``, **not** ``AIC.lm`` — drop1 uses this so
      the column is comparable across nested fits without the constant
      offset that ``AIC.lm`` carries. ``test="F"`` adds F-statistic and
      p-value columns.
    * ``glm``: AIC column is the standard ``glm.AIC`` (already on the
      same scale across nested fits). ``test="F"`` (typical for
      unknown-scale) or ``test="Chisq"``/``"LRT"`` (any family) add the
      test columns. The Chisq stat label flips between ``"LRT"`` (raw
      Δdev — appropriate when ``dispersion=1``) and ``"scaled dev."``
      (Δdev/dispersion_full — what mgcv/R uses for unknown-scale),
      matching ``drop1.glm`` exactly.
    * ``gmm``: refits without each droppable fixed-effect term (random-effect
      bars / offsets preserved), comparing by the Laplace-deviance LRT.
      Columns ``npar`` (Δnpar) / ``AIC`` (``-2logL + k·npar``) and, with a
      test, ``LRT`` / ``Pr(Chi)`` — matching ``drop1.merMod``. ``test`` accepts
      only ``"Chisq"``/``"LRT"`` or ``None`` (no F test for GLMMs).

    Parameters
    ----------
    test : {None, "F", "Chisq", "LRT", "Rao"}
        ``None`` (default) prints just the no-test columns. ``"LRT"``
        is an alias for ``"Chisq"``. ``"Rao"`` is not implemented yet.
    k : float
        Penalty multiplier for the AIC parameter count. ``k=log(n)``
        gives BIC. Only used by the lm path; glm's AIC is family-derived.
        Matches R's ``drop1.lm(..., k=)``.
    """
    if isinstance(model, gam):
        raise NotImplementedError(
            "drop1(gam): not implemented yet — mgcv's drop1.gam has "
            "smoothing-parameter caveats we haven't ported."
        )
    if isinstance(model, gmm):
        return _drop1_gmm(model, test=test, k=k)
    if isinstance(model, glm):
        return _drop1_glm(model, test=test, k=k)
    if isinstance(model, lm):
        return _drop1_lm(model, test=test, k=k)
    raise TypeError(f"drop1(): unsupported model type {type(model).__name__}")


def _drop_scope(terms) -> list[int]:
    """Indices of terms that respect *marginality* — R's ``drop.scope``."""
    factor_sets = [frozenset(t.label.split(":")) for t in terms]
    keep: list[int] = []
    for i, fi in enumerate(factor_sets):
        contained = any(
            j != i and fi < fj  # strict subset → ``fi`` is "marginal"
            for j, fj in enumerate(factor_sets)
        )
        if not contained:
            keep.append(i)
    return keep


def _refit_kwargs(m, target_n: int) -> dict:
    """Constructor kwargs for refitting ``m``'s type on a frame of
    ``target_n`` rows.
    """
    if isinstance(m, glm):
        n_orig = len(m._design_data)
        is_default = np.array_equal(m._prior_w, np.ones(n_orig))
        if is_default:
            return {"family": m.family, "weights": None}
        if target_n != n_orig:
            raise ValueError(
                "step/drop1/add1 refit: row count changed (likely due to "
                "NAs in some predictors) and the original fit had explicit "
                "weights. Pre-filter NA rows before fitting to avoid this."
            )
        return {"family": m.family, "weights": m._prior_w}
    if isinstance(m, lm):
        if m.weights is None:
            return {"weights": None, "method": m.method}
        if target_n != m.n:
            raise ValueError(
                "step/drop1/add1 refit: row count changed (likely due to "
                "NAs in some predictors) and the original fit had explicit "
                "weights. Pre-filter NA rows before fitting to avoid this."
            )
        return {"weights": m.weights, "method": m.method}
    raise TypeError(f"_refit_kwargs: unsupported {type(m).__name__}")


def _add_scope(current_terms, upper_terms) -> list[int]:
    """Indices into ``upper_terms`` of terms addable to ``current_terms``,
    respecting marginality. R's ``add.scope``.
    """
    cur_factor_sets = {frozenset(t.label.split(":")) for t in current_terms}
    cur_labels = {t.label for t in current_terms}
    addable: list[int] = []
    for i, t in enumerate(upper_terms):
        if t.label in cur_labels:
            continue
        ft = frozenset(t.label.split(":"))
        if all(
            frozenset(combo) in cur_factor_sets
            for size in range(1, len(ft))
            for combo in itertools.combinations(ft, size)
        ):
            addable.append(i)
    return addable


def add1(model, scope, *, test: str | None = None, k: float = 2.0):
    """Single-term additions, R's ``add1.lm`` / ``add1.glm``.

    For each term in ``scope`` that isn't already in ``model`` and that
    respects marginality (``a:b`` requires ``a`` and ``b`` already
    present), refits with that term added and prints a one-row-per-term
    table comparing each augmented fit to the current model
    (the ``<none>`` row).

    Mirror image of :func:`drop1` — the F denominator and Chisq stat
    use the *augmented* model's residual mean deviance and the *current*
    model's dispersion, respectively. AIC is recalibrated the same way
    drop1 does (holds dispersion fixed to make the column comparable).

    Parameters
    ----------
    scope : str
        RHS-only formula giving the upper-bound model (e.g.
        ``"x1 + x2 + x3"`` or ``"(x1+x2)*x3"``). LHS comes from ``model``.
    test : {None, "F", "Chisq", "LRT", "Rao"}
        Same surface as :func:`drop1`. ``None`` (default) prints just
        Df / Sum of Sq (or Deviance) / RSS / AIC; passing a test adds
        stat and p-value columns.
    k : float
        AIC penalty multiplier. Default 2.0 (AIC); pass ``log(n)`` for BIC.
    """
    if isinstance(model, gam):
        raise NotImplementedError(
            "add1(gam): not implemented yet — mgcv's add1.gam has "
            "smoothing-parameter caveats we haven't ported."
        )
    if isinstance(model, gmm):
        raise NotImplementedError("add1(gmm): not implemented yet.")

    lhs = model.formula.split("~", 1)[0].strip()
    upper_formula = f"{lhs} ~ {scope}"

    if isinstance(model, glm):
        upper_model = glm(
            upper_formula,
            model.data,
            family=model.family,
            weights=None,
        )
        return _add1_glm(
            model,
            upper_model._expanded.terms,
            common_data=upper_model._design_data,
            test=test,
            k=k,
        )
    if isinstance(model, lm):
        upper_model = lm(
            upper_formula,
            model.data,
            weights=None,
            method=model.method,
        )
        return _add1_lm(
            model,
            upper_model._expanded.terms,
            common_data=upper_model._design_data,
            test=test,
            k=k,
        )
    raise TypeError(f"add1(): unsupported model type {type(model).__name__}")


def _add1_lm(m: lm, upper_terms, *, common_data, test: str | None, k: float):
    """Refit-with-each-term-added implementation behind ``add1(lm)``."""
    if test is not None and test.upper() != "F":
        raise ValueError(f"add1(lm): test must be 'F' or None; got {test!r}")
    use_F = test is not None

    cur_terms = m._expanded.terms
    add_indices = _add_scope(cur_terms, upper_terms)
    if not add_indices:
        raise ValueError("add1(): no terms in scope for adding to model")

    kw = _refit_kwargs(m, len(common_data))
    if len(common_data) != m.n:
        m = lm(m.formula, common_data, **kw)

    lhs = m.formula.split("~", 1)[0].strip()
    intercept_str = "1" if m._expanded.intercept else "0"
    n = m.n
    rss_full = m.rss
    df_full = m.df_residuals

    df_col: list[int | None] = [None]
    sos_col: list[float | None] = [None]
    rss_col: list[float] = [round(rss_full, 4)]
    aic_col: list[float] = [round(_extract_aic_lm(rss_full, df_full, n, k), 4)]
    f_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]

    cur_labels = [t.label for t in cur_terms]
    for i in add_indices:
        t = upper_terms[i]
        new_labels = cur_labels + [t.label]
        sub_rhs = " + ".join(new_labels)
        sub_formula = f"{lhs} ~ {intercept_str} + {sub_rhs}"
        m_aug = lm(sub_formula, common_data, **kw)
        d_df = df_full - m_aug.df_residuals  # positive (params gained)
        d_rss = rss_full - m_aug.rss  # positive (rss reduction)

        df_col.append(d_df)
        sos_col.append(round(d_rss, 4))
        rss_col.append(round(m_aug.rss, 4))
        aic_col.append(round(_extract_aic_lm(m_aug.rss, m_aug.df_residuals, n, k), 4))
        if use_F and d_df > 0:
            mse_aug = m_aug.rss / m_aug.df_residuals
            fstat = (d_rss / d_df) / mse_aug
            p = float(_dist.pf(fstat, d_df, m_aug.df_residuals, lower_tail=False))
            f_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            f_col.append(None)
            p_col.append(None)
            sig_col.append("")

    cols: dict[str, list] = {
        "": ["<none>"] + [upper_terms[i].label for i in add_indices],
        "Df": df_col,
        "Sum of Sq": sos_col,
        "RSS": rss_col,
        "AIC": aic_col,
    }
    if use_F:
        cols["F value"] = f_col
        cols["Pr(>F)"] = p_col
        cols[" "] = sig_col

    print(f"Single term additions\n\nModel:\n{m.formula}\n")
    print(format_df(pl.DataFrame(cols)))
    if use_F:
        print("---")
        print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _add1_glm(m: glm, upper_terms, *, common_data, test: str | None, k: float):
    """Refit-with-each-term-added implementation behind ``add1(glm)``."""
    fam = m.family
    if test is None:
        kind = None
    else:
        t_norm = test.upper()
        if t_norm in ("CHISQ", "LRT"):
            kind = "Chisq"
        elif t_norm == "F":
            kind = "F"
        elif t_norm == "RAO":
            raise NotImplementedError(
                "add1(glm, test='Rao'): score test not implemented yet"
            )
        else:
            raise ValueError(
                f"add1(glm): test must be 'F', 'Chisq'/'LRT', 'Rao', or None; "
                f"got {test!r}"
            )

    cur_terms = m._expanded.terms
    add_indices = _add_scope(cur_terms, upper_terms)
    if not add_indices:
        raise ValueError("add1(): no terms in scope for adding to model")

    kw = _refit_kwargs(m, len(common_data))
    if len(common_data) != m.n:
        m = glm(m.formula, common_data, **kw)

    lhs = m.formula.split("~", 1)[0].strip()
    intercept_str = "1" if m._expanded.intercept else "0"
    dev_full = m.deviance
    df_full = m.df_residual
    n = m.n
    edf_full = n - df_full
    aic_full_table = m.AIC + (k - 2.0) * edf_full
    disp_cur = float(m.dispersion)

    df_col: list[int | None] = [None]
    dev_col: list[float] = [round(dev_full, 4)]
    aic_col: list[float] = [round(aic_full_table, 4)]
    stat_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]

    def _delta_loglik(dev_aug: float) -> float:
        """``loglik_aug - loglik_cur`` in R's drop1/add1 sign convention."""
        if fam.name == "gaussian":
            return n * float(np.log(dev_full / dev_aug))
        return (dev_full - dev_aug) / disp_cur

    cur_labels = [t.label for t in cur_terms]
    for i in add_indices:
        t = upper_terms[i]
        new_labels = cur_labels + [t.label]
        sub_rhs = " + ".join(new_labels)
        sub_formula = f"{lhs} ~ {intercept_str} + {sub_rhs}"
        m_aug = glm(sub_formula, common_data, **kw)
        d_df = df_full - m_aug.df_residual  # positive
        d_dev = dev_full - m_aug.deviance  # positive
        d_loglik = _delta_loglik(m_aug.deviance) if d_df > 0 else 0.0

        df_col.append(d_df)
        dev_col.append(round(m_aug.deviance, 4))
        aic_col.append(round(aic_full_table - d_loglik + k * d_df, 4))
        if kind == "F" and d_df > 0:
            rms_aug = m_aug.deviance / m_aug.df_residual
            fstat = (d_dev / d_df) / rms_aug
            p = float(_dist.pf(fstat, d_df, m_aug.df_residual, lower_tail=False))
            stat_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        elif kind == "Chisq" and d_df > 0:
            stat = d_loglik
            p = float(_dist.pchisq(stat, d_df, lower_tail=False))
            stat_col.append(round(stat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            stat_col.append(None)
            p_col.append(None)
            sig_col.append("")

    cols: dict[str, list] = {
        "": ["<none>"] + [upper_terms[i].label for i in add_indices],
        "Df": df_col,
        "Deviance": dev_col,
        "AIC": aic_col,
    }
    if kind == "F":
        cols["F value"] = stat_col
        cols["Pr(>F)"] = p_col
        cols[" "] = sig_col
    elif kind == "Chisq":
        stat_lbl = "LRT" if fam.scale_known else "scaled dev."
        cols[stat_lbl] = stat_col
        cols["Pr(>Chi)"] = p_col
        cols[" "] = sig_col

    print(f"Single term additions\n\nModel:\n{m.formula}\n")
    print(format_df(pl.DataFrame(cols)))
    if kind:
        print("---")
        print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _step_aic(model, k: float) -> float:
    """The AIC R's ``step()`` minimizes — extractAIC formula."""
    if isinstance(model, glm):
        edf = model.n - model.df_residual
        return model.AIC + (k - 2.0) * edf
    if isinstance(model, lm):
        return _extract_aic_lm(model.rss, model.df_residuals, model.n, k)
    raise TypeError(f"_step_aic: unsupported {type(model).__name__}")


def step(
    model,
    *,
    scope: str | dict | None = None,
    direction: str = "both",
    trace: bool = True,
    k: float = 2.0,
    steps: int = 1000,
):
    """Stepwise model selection — R's ``step()``.

    Iteratively considers single-term drops and/or adds, picks the move
    that minimizes the Mallows-style AIC (``extractAIC`` — see
    :func:`_step_aic`), and stops when no move improves AIC or
    ``steps`` iterations have elapsed. Mirrors the algorithm and
    formula choices of R's ``stats::step``.

    Parameters
    ----------
    scope : None | str | dict, optional
        Search bounds.
        - ``None`` (default): lower = ``~1`` (intercept-only), upper =
          current formula. With ``direction="both"`` this is effectively
          backward elimination — the typical "shrink a big model" use.
        - ``str``: RHS-only formula treated as the upper bound; lower
          defaults to ``~1``.
        - ``dict``: ``{"lower": "...", "upper": "..."}`` — explicit
          bounds for either or both sides.
    direction : {"both", "backward", "forward"}
        Which moves to consider at each step.
    trace : bool
        Print each step's drop/add table and the chosen move (R-style).
    k : float
        AIC penalty multiplier. ``2`` for AIC, ``log(n)`` for BIC.
    steps : int
        Hard cap on iterations.

    Returns
    -------
    The final fitted model.
    """
    if isinstance(model, gam):
        raise NotImplementedError("step(gam): not implemented yet")
    if isinstance(model, gmm):
        raise NotImplementedError("step(gmm): not implemented yet")
    if not isinstance(model, (lm, glm)):
        raise TypeError(f"step(): unsupported model type {type(model).__name__}")

    direction = direction.lower()
    if direction not in ("both", "backward", "forward"):
        raise ValueError(
            f"step(): direction must be 'both', 'backward', or 'forward'; "
            f"got {direction!r}"
        )

    is_glm = isinstance(model, glm)
    lhs = model.formula.split("~", 1)[0].strip()
    cur_rhs = model.formula.split("~", 1)[1].strip()
    intercept_str = "1" if model._expanded.intercept else "0"

    if scope is None:
        lower_rhs = intercept_str
        upper_rhs = cur_rhs
    elif isinstance(scope, str):
        lower_rhs = intercept_str
        upper_rhs = scope
    elif isinstance(scope, dict):
        lower_rhs = scope.get("lower", intercept_str)
        upper_rhs = scope.get("upper", cur_rhs)
    else:
        raise TypeError(
            "step(): scope must be None, str (upper formula RHS), "
            "or dict {'lower': ..., 'upper': ...}"
        )

    if is_glm:
        upper_model = glm(
            f"{lhs} ~ {upper_rhs}",
            model.data,
            family=model.family,
            weights=None,
        )
        lower_model = glm(
            f"{lhs} ~ {lower_rhs}",
            model.data,
            family=model.family,
            weights=None,
        )
    else:
        upper_model = lm(
            f"{lhs} ~ {upper_rhs}",
            model.data,
            weights=None,
            method=model.method,
        )
        lower_model = lm(
            f"{lhs} ~ {lower_rhs}",
            model.data,
            weights=None,
            method=model.method,
        )
    upper_terms = upper_model._expanded.terms
    lower_label_set = {t.label for t in lower_model._expanded.terms}

    common_data = upper_model._design_data
    kw = _refit_kwargs(model, len(common_data))

    def _refit(formula: str):
        if is_glm:
            return glm(formula, common_data, **kw)
        return lm(formula, common_data, **kw)

    if len(common_data) != model.n:
        current = _refit(model.formula)
    else:
        current = model
    cur_aic = _step_aic(current, k)

    if trace:
        print(f"Start:  AIC={cur_aic:.2f}")
        print(current.formula)

    for _ in range(steps):
        cur_terms = current._expanded.terms
        candidates: list[tuple[str, object, float]] = []

        if direction in ("backward", "both"):
            for j in _drop_scope(cur_terms):
                t = cur_terms[j]
                if t.label in lower_label_set:
                    continue
                rest = [cur_terms[i].label for i in range(len(cur_terms)) if i != j]
                sub_rhs_str = " + ".join(rest)
                sub_formula = (
                    f"{lhs} ~ {intercept_str} + {sub_rhs_str}"
                    if sub_rhs_str
                    else f"{lhs} ~ {intercept_str}"
                )
                sub = _refit(sub_formula)
                candidates.append(("- " + t.label, sub, _step_aic(sub, k)))

        if direction in ("forward", "both"):
            for i in _add_scope(cur_terms, upper_terms):
                t = upper_terms[i]
                new_labels = [tt.label for tt in cur_terms] + [t.label]
                sub_rhs_str = " + ".join(new_labels)
                sub_formula = f"{lhs} ~ {intercept_str} + {sub_rhs_str}"
                sub = _refit(sub_formula)
                candidates.append(("+ " + t.label, sub, _step_aic(sub, k)))

        if trace:
            _print_step_trace(current, cur_aic, candidates, is_glm)

        if not candidates:
            break

        candidates.sort(key=lambda c: c[2])
        _, best_sub, best_aic = candidates[0]
        if best_aic >= cur_aic:
            break  # no improvement — stop

        current = best_sub
        cur_aic = best_aic
        if trace:
            print(f"\nStep:  AIC={cur_aic:.2f}")
            print(current.formula)

    return current


def _print_step_trace(current, cur_aic: float, candidates, is_glm: bool):
    """R-style step trace: ``<none>`` + each candidate, sorted by AIC."""
    rows: list[tuple] = []
    if is_glm:
        rows.append(("<none>", None, current.deviance, cur_aic))
        for label, sub, aic in candidates:
            df_diff = abs(sub.df_residual - current.df_residual)
            rows.append((label, df_diff, sub.deviance, aic))
        rows.sort(key=lambda r: r[3])
        df_table = pl.DataFrame(
            {
                "": [r[0] for r in rows],
                "Df": [r[1] for r in rows],
                "Deviance": [round(r[2], 4) for r in rows],
                "AIC": [round(r[3], 2) for r in rows],
            }
        )
    else:
        rows.append(("<none>", None, None, current.rss, cur_aic))
        for label, sub, aic in candidates:
            df_diff = abs(sub.df_residuals - current.df_residuals)
            d_rss = abs(sub.rss - current.rss)
            rows.append((label, df_diff, d_rss, sub.rss, aic))
        rows.sort(key=lambda r: r[4])
        df_table = pl.DataFrame(
            {
                "": [r[0] for r in rows],
                "Df": [r[1] for r in rows],
                "Sum of Sq": [
                    round(r[2], 4) if r[2] is not None else None for r in rows
                ],
                "RSS": [round(r[3], 4) for r in rows],
                "AIC": [round(r[4], 2) for r in rows],
            }
        )
    print()
    print(format_df(df_table))


def _extract_aic_lm(rss: float, df_residuals: int, n: int, k: float) -> float:
    """R's ``extractAIC.lm`` — Mallows-style ``n*log(RSS/n) + k*p``."""
    p = n - df_residuals
    if rss <= 0:
        return float("-inf")
    return n * float(np.log(rss / n)) + k * p


def _drop1_lm(m: lm, *, test: str | None, k: float):
    """Refit-without-each-term implementation behind ``drop1(lm)``."""
    if test is not None and test.upper() != "F":
        raise ValueError(f"drop1(lm): test must be 'F' or None; got {test!r}")
    use_F = test is not None

    terms = m._expanded.terms
    if not terms:
        raise TypeError("drop1(): need at least one RHS term to drop")

    lhs = m.formula.split("~", 1)[0].strip()
    intercept_str = "1" if m._expanded.intercept else "0"
    n = m.n
    rss_full = m.rss
    df_full = m.df_residuals
    mse_full = rss_full / df_full

    scope = _drop_scope(terms)
    common_data = m._design_data
    kw = _refit_kwargs(m, len(common_data))

    df_col: list[int | None] = [None]
    sos_col: list[float | None] = [None]
    rss_col: list[float] = [round(rss_full, 4)]
    aic_col: list[float] = [round(_extract_aic_lm(rss_full, df_full, n, k), 4)]
    f_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]

    for j in scope:
        rest = [terms[i].label for i in range(len(terms)) if i != j]
        sub_rhs = " + ".join(rest) if rest else ""
        sub_formula = (
            f"{lhs} ~ {intercept_str} + {sub_rhs}"
            if sub_rhs
            else f"{lhs} ~ {intercept_str}"
        )
        m_sub = lm(sub_formula, common_data, **kw)
        d_df = m_sub.df_residuals - df_full
        d_rss = m_sub.rss - rss_full

        df_col.append(d_df)
        sos_col.append(round(d_rss, 4))
        rss_col.append(round(m_sub.rss, 4))
        aic_col.append(round(_extract_aic_lm(m_sub.rss, m_sub.df_residuals, n, k), 4))
        if use_F and d_df > 0:
            fstat = (d_rss / d_df) / mse_full
            p = float(_dist.pf(fstat, d_df, df_full, lower_tail=False))
            f_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            f_col.append(None)
            p_col.append(None)
            sig_col.append("")

    cols: dict[str, list] = {
        "": ["<none>"] + [terms[j].label for j in scope],
        "Df": df_col,
        "Sum of Sq": sos_col,
        "RSS": rss_col,
        "AIC": aic_col,
    }
    if use_F:
        cols["F value"] = f_col
        cols["Pr(>F)"] = p_col
        cols[" "] = sig_col

    print(f"Single term deletions\n\nModel:\n{m.formula}\n")
    print(format_df(pl.DataFrame(cols)))
    if use_F:
        print("---")
        print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _drop1_glm(m: glm, *, test: str | None, k: float):
    """Refit-without-each-term implementation behind ``drop1(glm)``."""
    fam = m.family
    if test is None:
        kind = None
    else:
        t_norm = test.upper()
        if t_norm in ("CHISQ", "LRT"):
            kind = "Chisq"
        elif t_norm == "F":
            kind = "F"
        elif t_norm == "RAO":
            raise NotImplementedError(
                "drop1(glm, test='Rao'): score test not implemented yet"
            )
        else:
            raise ValueError(
                f"drop1(glm): test must be 'F', 'Chisq'/'LRT', 'Rao', or None; "
                f"got {test!r}"
            )

    terms = m._expanded.terms
    if not terms:
        raise TypeError("drop1(): need at least one RHS term to drop")

    lhs = m.formula.split("~", 1)[0].strip()
    intercept_str = "1" if m._expanded.intercept else "0"
    dev_full = m.deviance
    df_full = m.df_residual
    disp_full = float(m.dispersion)
    n = m.n
    edf_full = n - df_full
    aic_full_table = m.AIC + (k - 2.0) * edf_full

    scope = _drop_scope(terms)
    common_data = m._design_data
    kw = _refit_kwargs(m, len(common_data))

    df_col: list[int | None] = [None]
    dev_col: list[float] = [round(dev_full, 4)]
    aic_col: list[float] = [round(aic_full_table, 4)]
    stat_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]

    def _delta_loglik(dev_drop: float) -> float:
        """R's drop1.glm "loglik diff" between the dropped and full fit."""
        if fam.name == "gaussian":
            return n * float(np.log(dev_drop / dev_full))
        return (dev_drop - dev_full) / disp_full

    for j in scope:
        rest = [terms[i].label for i in range(len(terms)) if i != j]
        sub_rhs = " + ".join(rest) if rest else ""
        sub_formula = (
            f"{lhs} ~ {intercept_str} + {sub_rhs}"
            if sub_rhs
            else f"{lhs} ~ {intercept_str}"
        )
        m_sub = glm(sub_formula, common_data, **kw)
        d_df = m_sub.df_residual - df_full
        d_dev = m_sub.deviance - dev_full
        d_loglik = _delta_loglik(m_sub.deviance) if d_df > 0 else 0.0

        df_col.append(d_df)
        dev_col.append(round(m_sub.deviance, 4))
        aic_col.append(round(aic_full_table + d_loglik - k * d_df, 4))
        if kind == "F" and d_df > 0:
            rms_full = dev_full / df_full
            fstat = (d_dev / d_df) / rms_full
            p = float(_dist.pf(fstat, d_df, df_full, lower_tail=False))
            stat_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        elif kind == "Chisq" and d_df > 0:
            stat = d_loglik
            p = float(_dist.pchisq(stat, d_df, lower_tail=False))
            stat_col.append(round(stat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            stat_col.append(None)
            p_col.append(None)
            sig_col.append("")

    cols: dict[str, list] = {
        "": ["<none>"] + [terms[j].label for j in scope],
        "Df": df_col,
        "Deviance": dev_col,
        "AIC": aic_col,
    }
    if kind == "F":
        cols["F value"] = stat_col
        cols["Pr(>F)"] = p_col
        cols[" "] = sig_col
    elif kind == "Chisq":
        stat_lbl = "LRT" if fam.scale_known else "scaled dev."
        cols[stat_lbl] = stat_col
        cols["Pr(>Chi)"] = p_col
        cols[" "] = sig_col

    print(f"Single term deletions\n\nModel:\n{m.formula}\n")
    print(format_df(pl.DataFrame(cols)))
    if kind:
        print("---")
        print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _drop1_gmm(model, *, test, k):
    """Single fixed-term deletions for a ``gmm`` fit — lme4's ``drop1.merMod``."""
    if test is not None and test.upper() not in ("CHISQ", "LRT"):
        raise ValueError(
            f"drop1(gmm): test must be 'Chisq'/'LRT' or None; got {test!r}"
        )
    do_test = test is not None
    if model.REML:
        print("refitting model(s) with ML (instead of REML)")
        m = gmm(model.formula, model.data, REML=False)
    else:
        m = model

    terms = m._expanded.terms
    if not terms:
        raise TypeError("drop1(gmm): need at least one fixed-effect term to drop")
    lhs = m.formula.split("~", 1)[0].strip()
    intercept_str = "1" if m._expanded.intercept else "0"
    keep_tail = [f"({deparse(b)})" for b in m._expanded.bars]
    keep_tail += [f"offset({deparse(o)})" for o in m._expanded.offsets]

    def _dev(mm):
        return float(getattr(mm, "deviance_laplace", mm.deviance))

    def _aic_table(dev, npar):
        return dev + k * npar

    dev_full = _dev(m)
    npar_full = m.npar
    fam = m.family
    scope = _drop_scope(terms)

    labels = ["<none>"]
    npar_col: list[int | None] = [None]
    aic_col: list[float] = [round(_aic_table(dev_full, npar_full), 1)]
    lrt_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]

    for j in scope:
        rest = [terms[i].label for i in range(len(terms)) if i != j]
        rhs_parts = [intercept_str] + rest + keep_tail
        sub_formula = f"{lhs} ~ " + " + ".join(rhs_parts)
        m_sub = gmm(sub_formula, m.data, family=fam, REML=False)
        d_df = npar_full - m_sub.npar
        lrt = _dev(m_sub) - dev_full

        labels.append(terms[j].label)
        npar_col.append(d_df)
        aic_col.append(round(_aic_table(_dev(m_sub), m_sub.npar), 1))
        if do_test and d_df > 0:
            p = float(_dist.pchisq(lrt, d_df, lower_tail=False))
            lrt_col.append(round(lrt, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            lrt_col.append(None)
            p_col.append(None)
            sig_col.append("")

    cols: dict[str, list] = {"": labels, "npar": npar_col, "AIC": aic_col}
    if do_test:
        cols["LRT"] = lrt_col
        cols["Pr(Chi)"] = p_col
        cols[" "] = sig_col

    print(f"Single term deletions\n\nModel:\n{m.formula}\n")
    print(format_df(pl.DataFrame(cols)))
    if do_test:
        print("---")
        print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_lm(*models, labels: list[str]):
    """F-test ANOVA table comparing nested ``lm`` fits."""
    order = sorted(
        range(len(models)), key=lambda i: models[i].df_residuals, reverse=True
    )

    dfs = [models[i].df_residuals for i in order]
    rss = [models[i].rss for i in order]
    mse_full = rss[-1] / dfs[-1]

    df_col: list[int | None] = [None]
    sos_col: list[float | None] = [None]
    f_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]
    for k in range(1, len(order)):
        d_df = dfs[k - 1] - dfs[k]
        d_rss = rss[k - 1] - rss[k]
        if d_df <= 0:
            df_col.append(d_df)
            sos_col.append(round(d_rss, 3))
            f_col.append(None)
            p_col.append(None)
            sig_col.append("")
            continue
        fstat = (d_rss / d_df) / mse_full
        p = float(_dist.pf(fstat, d_df, dfs[-1], lower_tail=False))
        df_col.append(d_df)
        sos_col.append(round(d_rss, 3))
        f_col.append(round(fstat, 3))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame(
        {
            "": [labels[i] for i in order],
            "Res.Df": dfs,
            "RSS": [round(r, 3) for r in rss],
            "Df": df_col,
            "Sum of Sq": sos_col,
            "F": f_col,
            "Pr(>F)": p_col,
            " ": sig_col,
        }
    )

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_lm_single(m: lm):
    """Sequential (Type I) ANOVA — R's ``anova.lm(m)`` for a single fit."""
    terms = m._expanded.terms
    if not terms:
        raise TypeError(
            "anova(m): single-model form needs at least one RHS term "
            "(got an intercept-only model)"
        )

    lhs = m.formula.split("~", 1)[0].strip()

    e = np.asarray(m._residuals_arr, dtype=float)
    w = (
        np.asarray(m._w, dtype=float)
        if getattr(m, "_w", None) is not None
        else np.ones_like(e)
    )
    ssr = float(np.sum(w * e * e))
    dfr = m.df_residuals
    mse_full = ssr / dfr

    comp = np.asarray(m.effects, dtype=float)[: m.rank]
    asgn = np.array([m._col_assign[c] for c in m.column_names])

    uniq = [a for a in sorted(set(asgn.tolist())) if a != 0]
    labels: list[str] = []
    df_col: list[int] = []
    sos_col: list[float] = []
    ms_col: list[float] = []
    f_col: list[float | None] = []
    p_col: list[float | None] = []
    sig_col: list[str] = []
    for a in uniq:
        idx = np.where(asgn == a)[0]
        d_df = int(idx.size)
        d_rss = float(np.sum(comp[idx] ** 2))
        ms = d_rss / d_df
        fstat = ms / mse_full
        p = float(_dist.pf(fstat, d_df, dfr, lower_tail=False))
        labels.append(terms[a - 1].label)
        df_col.append(d_df)
        sos_col.append(round(d_rss, 4))
        ms_col.append(round(ms, 4))
        f_col.append(round(fstat, 4))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])
    labels.append("Residuals")
    df_col.append(dfr)
    sos_col.append(round(ssr, 4))
    ms_col.append(round(mse_full, 4))
    f_col.append(None)
    p_col.append(None)
    sig_col.append("")

    docstring = "Analysis of Variance Table\n\n"
    docstring += f"Response: {lhs}\n"

    df_ = pl.DataFrame(
        {
            "": labels,
            "Df": df_col,
            "Sum Sq": sos_col,
            "Mean Sq": ms_col,
            "F value": f_col,
            "Pr(>F)": p_col,
            " ": sig_col,
        }
    )

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_gam_single(m: gam, freq: bool = False, dispersion: float | None = None):
    """``anova.gam``-style single-model output: parametric Terms table
    plus the smooth significance table. Mirrors mgcv's ``anova.gam`` for
    a single fit (which omits the lm-coefficient details that
    ``summary.gam`` prints).
    (mgcv.r:4153).
    """
    digits = 4
    est_disp = (not m.family.scale_known) and dispersion is None

    out = []
    out.append("")
    out.append(f"Family: {m.family.name}")
    out.append(f"Link function: {m.family.link.name}")
    out.append("")
    out.append(f"Formula: {m.formula}")
    out.append("")

    rows = m._pterms_rows(freq=freq, dispersion=dispersion)
    if rows:
        stat_col = "F" if est_disp else "Chi.sq"
        sig = significance_code([r[3] for r in rows])
        tbl = pl.DataFrame(
            {
                "": [r[0] for r in rows],
                "df": [r[1] for r in rows],
                stat_col: format_signif([r[2] for r in rows], digits=digits),
                "p-value": format_pval([r[3] for r in rows], digits=_dig_tst(digits)),
                " ": sig,
            }
        )
        out.append("Parametric Terms:")
        out.append(
            format_df(
                tbl,
                align={c: "right" for c in ("df", stat_col, "p-value")},
            )
        )
        out.append("---")
        out.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        out.append("")

    if m._blocks:
        sm_rows = m._smooth_significance_rows(dispersion=dispersion)
        sig_smooth = significance_code([r[4] for r in sm_rows])
        stat_col = "F" if est_disp else "Chi.sq"
        sm_tbl = pl.DataFrame(
            {
                "": [r[0] for r in sm_rows],
                "edf": format_signif([r[1] for r in sm_rows], digits=digits),
                "Ref.df": format_signif([r[2] for r in sm_rows], digits=digits),
                stat_col: format_signif([r[3] for r in sm_rows], digits=digits),
                "p-value": format_pval(
                    [r[4] for r in sm_rows], digits=_dig_tst(digits)
                ),
                " ": sig_smooth,
            }
        )
        out.append("Approximate significance of smooth terms:")
        out.append(
            format_df(
                sm_tbl,
                align={c: "right" for c in ("edf", "Ref.df", stat_col, "p-value")},
            )
        )
        out.append("---")
        out.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    print("\n".join(out))


def _anova_gam_rdf(g: gam) -> float:
    """mgcv-style residual df for a ``gam`` in a multi-model anova table."""
    n = g.n
    edf1_sum = float(np.sum(g.edf1))
    edf2 = getattr(g, "edf2", None)
    if edf2 is not None and not np.allclose(edf2, g.edf1):
        edf_sum = float(np.sum(g.edf))
        edf2_sum = float(np.sum(edf2))
        dfc = edf2_sum - edf_sum
    else:
        dfc = 0.0
    return n - edf1_sum - dfc


def _anova_gam(*models: gam, labels: list[str], test: str | None = None):
    """Approximate F / Chisq deviance table for nested ``gam`` fits."""
    df_, docstring = _anova_gam_table(*models, labels=labels, test=test)
    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_gam_table(*models: gam, labels: list[str], test: str | None = None):
    """Pure builder for the multi-model ``anova(gam, ...)`` table."""
    fam0 = models[0].family
    if not all(
        type(m.family) is type(fam0) and m.family.link.name == fam0.link.name
        for m in models
    ):
        raise ValueError("anova(): all gam fits must share family and link")

    if test is None:
        test = "Chisq" if fam0.scale_known else "F"
    else:
        t_norm = test.upper()
        if t_norm == "LRT":
            test = "Chisq"
        elif t_norm == "RAO":
            raise NotImplementedError(
                "anova(gam, test='Rao'): score test not implemented yet"
            )
        elif t_norm == "CHISQ":
            test = "Chisq"
        elif t_norm == "F":
            test = "F"
        else:
            raise ValueError(
                f"anova(gam): test must be 'Chisq', 'LRT', 'F', 'Rao', or None; "
                f"got {test!r}"
            )

    rdfs = [_anova_gam_rdf(m) for m in models]
    devs = [float(m.deviance) for m in models]

    order = sorted(range(len(models)), key=lambda i: rdfs[i], reverse=True)
    rdfs_sorted = [rdfs[i] for i in order]
    devs_sorted = [devs[i] for i in order]
    full = models[order[-1]]
    disp_full = float(full.scale) if not fam0.scale_known else 1.0
    rdf_full = rdfs_sorted[-1]

    df_col: list[float | None] = [None]
    dev_col: list[float | None] = [None]
    stat_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]
    for k in range(1, len(order)):
        d_df = rdfs_sorted[k - 1] - rdfs_sorted[k]
        d_dev = devs_sorted[k - 1] - devs_sorted[k]
        if d_df <= 0:
            df_col.append(round(d_df, 4))
            dev_col.append(round(d_dev, 4))
            stat_col.append(None)
            p_col.append(None)
            sig_col.append("")
            continue
        if test == "Chisq":
            stat = d_dev / disp_full
            p = float(_dist.pchisq(stat, d_df, lower_tail=False))
        else:  # "F"
            stat = (d_dev / d_df) / disp_full
            p = float(_dist.pf(stat, d_df, rdf_full, lower_tail=False))
        df_col.append(round(d_df, 4))
        dev_col.append(round(d_dev, 4))
        stat_col.append(round(stat, 4))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Deviance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    cols: dict[str, list] = {
        "": [labels[i] for i in order],
        "Resid. Df": [round(r, 4) for r in rdfs_sorted],
        "Resid. Dev": [round(d, 4) for d in devs_sorted],
        "Df": df_col,
        "Deviance": dev_col,
    }
    if test == "F":
        cols["F"] = stat_col
        cols["Pr(>F)"] = p_col
    else:
        cols["Pr(>Chi)"] = p_col
    cols[" "] = sig_col

    return pl.DataFrame(cols), docstring


def _anova_glm(*models, labels: list[str], test: str | None = None):
    """``anova.glm``-style deviance table for nested ``glm`` fits."""
    df_, docstring = _anova_glm_table(*models, labels=labels, test=test)
    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_glm_table(*models, labels: list[str], test: str | None = None):
    """Pure builder for the ``anova(glm,...)`` table."""
    fam0 = models[0].family
    if not all(
        type(m.family) is type(fam0) and m.family.link.name == fam0.link.name
        for m in models
    ):
        raise ValueError("anova(): all glm fits must share family and link")

    if test is None:
        test = "Chisq" if fam0.scale_known else "F"
    else:
        t_norm = test.upper()
        if t_norm == "LRT":
            test = "Chisq"
        elif t_norm == "RAO":
            raise NotImplementedError(
                "anova(glm, test='Rao'): score test not implemented yet"
            )
        elif t_norm == "CHISQ":
            test = "Chisq"
        elif t_norm == "F":
            test = "F"
        else:
            raise ValueError(
                f"anova(glm): test must be 'Chisq', 'LRT', 'F', 'Rao', or None; "
                f"got {test!r}"
            )

    order = sorted(
        range(len(models)), key=lambda i: models[i].df_residuals, reverse=True
    )
    dfs = [models[i].df_residual for i in order]
    devs = [models[i].deviance for i in order]
    full = models[order[-1]]
    disp_full = float(full.dispersion)
    df_full = int(full.df_residual)

    df_col: list[int | None] = [None]
    dev_col: list[float | None] = [None]
    stat_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]
    for k in range(1, len(order)):
        d_df = dfs[k - 1] - dfs[k]
        d_dev = devs[k - 1] - devs[k]
        if d_df <= 0:
            df_col.append(d_df)
            dev_col.append(round(d_dev, 4))
            stat_col.append(None)
            p_col.append(None)
            sig_col.append("")
            continue
        if test == "Chisq":
            stat = d_dev / disp_full
            p = float(_dist.pchisq(stat, d_df, lower_tail=False))
        else:
            stat = (d_dev / d_df) / disp_full
            p = float(_dist.pf(stat, d_df, df_full, lower_tail=False))
        df_col.append(d_df)
        dev_col.append(round(d_dev, 4))
        stat_col.append(round(stat, 4))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Deviance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    stat_lbl = "F" if test == "F" else "Deviance"
    p_lbl = "Pr(>F)" if test == "F" else "Pr(>Chi)"

    df_ = pl.DataFrame(
        {
            "": [labels[i] for i in order],
            "Resid. Df": dfs,
            "Resid. Dev": [round(d, 4) for d in devs],
            "Df": df_col,
            "Deviance": dev_col,
            stat_lbl: stat_col,
            p_lbl: p_col,
            " ": sig_col,
        }
    )
    return df_, docstring


def _anova_gmm_single(model):
    """Single-model Type-I (sequential) fixed-effect F-table — lme4's
    ``anova.merMod``.
    """
    from ..formula import materialize

    _, assign = materialize(model._expanded, model.data, return_assign=True)
    RX, _ = model._getme_rx_rzx()
    effects = RX @ np.asarray(model._beta, dtype=float).ravel()
    sigma2 = float(model.sigma_squared)
    rows, npar, ss_col, ms_col, f_col = [], [], [], [], []
    for i, lbl in enumerate(model._expanded.term_labels, start=1):
        cols = [j for j, a in enumerate(assign) if a == i]
        if not cols:
            continue
        ss = float(np.sum(effects[cols] ** 2))
        df = len(cols)
        rows.append(lbl)
        npar.append(df)
        ss_col.append(ss)
        ms_col.append(ss / df)
        f_col.append((ss / df) / sigma2)
    return pl.DataFrame(
        {
            "": rows,
            "npar": npar,
            "Sum Sq": ss_col,
            "Mean Sq": ms_col,
            "F value": f_col,
        }
    )


def _anova_gmm(*models, labels: list[str]):
    """Likelihood-ratio test for nested ``gmm`` fits (lme4-style)."""
    refit = any(m.REML for m in models)
    models = tuple(
        (gmm(m.formula, m.data, REML=False) if m.REML else m) for m in models
    )
    if refit:
        print("refitting model(s) with ML (instead of REML)")
    order = sorted(range(len(models)), key=lambda i: models[i].npar)

    npar_col: list[int] = []
    aic_col: list[float] = []
    bic_col: list[float] = []
    ll_col: list[float] = []
    dev_col: list[float] = []
    chi_col: list[float | None] = []
    dfc_col: list[int | None] = []
    p_col: list[float | None] = []
    sig_col: list[str] = []
    for k, idx in enumerate(order):
        m = models[idx]
        npar_col.append(m.npar)
        aic_col.append(round(m.AIC, 1))
        bic_col.append(round(m.BIC, 1))
        ll_col.append(round(m.loglike, 1))
        dev_val = float(getattr(m, "deviance_laplace", m.deviance))
        dev_col.append(round(dev_val, 1))
        if k == 0:
            chi_col.append(None)
            dfc_col.append(None)
            p_col.append(None)
            sig_col.append("")
            continue
        prev = models[order[k - 1]]
        prev_dev = float(getattr(prev, "deviance_laplace", prev.deviance))
        chisq = max(0.0, prev_dev - dev_val)
        d_df = m.npar - prev.npar
        p = (
            float(_dist.pchisq(chisq, d_df, lower_tail=False))
            if d_df > 0
            else float("nan")
        )
        chi_col.append(round(chisq, 4))
        dfc_col.append(d_df)
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table (likelihood ratio test)\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame(
        {
            "": [labels[i] for i in order],
            "npar": npar_col,
            "AIC": aic_col,
            "BIC": bic_col,
            "logLik": ll_col,
            "-2*log(L)": dev_col,
            "Chisq": chi_col,
            "Df": dfc_col,
            "Pr(>Chisq)": p_col,
            " ": sig_col,
        }
    )

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
