"""Model-comparison helpers: ``anova``, ``AIC``, ``BIC``.

Lives above ``lm`` and ``lme`` in the import graph so both can be compared
here without creating a cycle. ``lm.py`` and ``lme.py`` stay unaware of
each other.
"""

from __future__ import annotations

import inspect
import itertools

import numpy as np
import polars as pl
from scipy.stats import chi2, f

from .gam import gam
from .glm import glm
from .lm import lm
from .lme import lme
from .utils import _dig_tst, format_df, format_pval, format_signif, significance_code

__all__ = ["anova", "AIC", "BIC", "drop1", "add1", "step"]


def _caller_names(models, frame, fallback: str = "model") -> list[str]:
    """Recover caller-bound variable names for ``models``, like R's
    ``match.call``. Walks ``frame``'s locals + globals; falls back to
    ``f"{fallback} {i}"`` when a model has no unique binding (e.g.
    passed as an expression or aliased to multiple names).
    """
    if frame is None:
        return [f"{fallback} {i}" for i in range(len(models))]
    scope = {**frame.f_globals, **frame.f_locals}
    by_id: dict[int, list[str]] = {}
    for name, val in scope.items():
        if name.startswith("_"):
            continue
        by_id.setdefault(id(val), []).append(name)
    out = []
    for i, m in enumerate(models):
        names = by_id.get(id(m), [])
        out.append(names[0] if len(names) == 1 else f"{fallback} {i}")
    return out


def AIC(*models) -> None:
    """Print an AIC comparison table for one or more fitted models.

    Each model must expose ``.AIC`` and ``.npar``. Row labels are
    recovered from the caller's variable names (R-style); falls back
    to ``model i`` for unbound or aliased arguments.
    """
    names = _caller_names(models, inspect.currentframe().f_back)
    rows = pl.DataFrame({
        "":    names,
        "df":  [m.npar    for m in models],
        "AIC": [round(m.AIC, 2) for m in models],
    })
    print(format_df(rows))


def BIC(*models) -> None:
    """Print a BIC comparison table for one or more fitted models.

    Each model must expose ``.BIC`` and ``.npar``. Row labels are
    recovered from the caller's variable names (R-style); falls back
    to ``model i`` for unbound or aliased arguments.
    """
    names = _caller_names(models, inspect.currentframe().f_back)
    rows = pl.DataFrame({
        "":    names,
        "df":  [m.npar    for m in models],
        "BIC": [round(m.BIC, 2) for m in models],
    })
    print(format_df(rows))


def anova(*models, test: str | None = None):
    """Compare nested fits, or decompose a single fit by Type-I SS.

    - One ``lm`` → sequential (Type I) ANOVA table, splitting the model's
      total SS into incremental contributions per RHS term in formula
      order. Mirrors R's ``anova(m)`` for a single ``lm``.
    - Multiple ``lm`` fits → F-test ANOVA table (incremental for 3+).
    - Multiple ``glm`` fits → analysis-of-deviance table (incremental for
      3+); ``test=`` selects the test statistic (see below).
    - Multiple ``lme`` fits → likelihood-ratio test (lme4-style, incremental
      for 3+). REML fits are internally refit by ML before the LRT.

    Parameters
    ----------
    test : {"Chisq", "LRT", "F", "Rao", None}, optional
        Only meaningful for ``glm`` comparisons. ``None`` (default) auto-
        picks ``"Chisq"`` for scale-known families (Poisson, Binomial) and
        ``"F"`` for unknown-scale (Gaussian, Gamma, IG), matching R's
        ``anova.glm`` recommendation. ``"LRT"`` is an alias for ``"Chisq"``.
        ``"Rao"`` (score test) is not implemented yet. For ``lm`` and ``lme``
        the test is fixed (always F / Chisq LRT respectively); passing
        ``test=`` for those raises.

    For multi-model calls rows are sorted by parameter count (smaller
    model first), matching R's ``anova``. Row labels are recovered from
    the caller's variable names (R-style); falls back to ``model i`` for
    unbound or aliased arguments, preserving *input* order.
    """
    if len(models) == 0:
        raise TypeError("anova(): need at least one model")
    if len(models) == 1:
        m = models[0]
        if isinstance(m, gam):
            if test is not None:
                raise TypeError("anova(gam): test= is not accepted")
            return _anova_gam_single(m)
        if isinstance(m, lm) and not isinstance(m, glm):
            if test is not None:
                raise TypeError("anova(lm): test= is not accepted (always F)")
            return _anova_lm_single(m)
        raise TypeError(
            "anova(m): single-model form supports lm and gam only "
            f"(got {type(m).__name__})"
        )
    labels = _caller_names(models, inspect.currentframe().f_back)
    if all(isinstance(m, lme) for m in models):
        if test is not None and test.upper() not in ("CHISQ", "LRT"):
            raise ValueError(
                f"anova(lme): only test='Chisq'/'LRT' (the default LRT) "
                f"is supported, got {test!r}"
            )
        return _anova_lme(*models, labels=labels)
    if all(isinstance(m, gam) for m in models):
        return _anova_gam(*models, labels=labels, test=test)
    # glm before lm: glm is not an lm subclass, but the isinstance order
    # would still matter if it ever became one. Keep the explicit branch.
    if all(isinstance(m, glm) for m in models):
        return _anova_glm(*models, labels=labels, test=test)
    if all(isinstance(m, lm) for m in models):
        if test is not None:
            raise TypeError("anova(lm): test= is not accepted (always F)")
        return _anova_lm(*models, labels=labels)
    raise TypeError(
        "anova(): all models must be the same type (lm, glm, gam, or lme)"
    )


def drop1(model, *, test: str | None = None, k: float = 2.0):
    """Single-term deletions, R's ``drop1.lm`` / ``drop1.glm``.

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
    if isinstance(model, lme):
        raise NotImplementedError("drop1(lme): not implemented yet.")
    # glm before lm: glm is not an lm subclass, but order matters if
    # that ever changes (mirrors anova()'s dispatch order).
    if isinstance(model, glm):
        return _drop1_glm(model, test=test, k=k)
    if isinstance(model, lm):
        return _drop1_lm(model, test=test, k=k)
    raise TypeError(
        f"drop1(): unsupported model type {type(model).__name__}"
    )


def _drop_scope(terms) -> list[int]:
    """Indices of terms that respect *marginality* — R's ``drop.scope``.

    A term is droppable iff its factor set is not a strict subset of any
    other term's factor set. So given ``cpergore + usage + cpergore:usage``,
    neither ``cpergore`` nor ``usage`` is droppable (they're both
    contained in the interaction); only ``cpergore:usage`` is. Without
    this filter, ``drop1`` would happily compute Δrss for "drop the
    main effect while keeping the interaction" — which is what R's
    ``drop1`` deliberately avoids, and which the Faraway book example
    on ``gavote`` shows R skipping over.
    """
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


def _add_scope(current_terms, upper_terms) -> list[int]:
    """Indices into ``upper_terms`` of terms addable to ``current_terms``,
    respecting marginality. R's ``add.scope``.

    A term is addable iff (a) it isn't already in ``current_terms`` and
    (b) every strict non-empty subset of its factor set is the factor
    set of some term currently in ``current_terms``. The second clause
    is what blocks ``add a:b`` while ``a`` or ``b`` is missing — the
    interaction can only be added once both main effects exist.
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
    if isinstance(model, lme):
        raise NotImplementedError("add1(lme): not implemented yet.")

    lhs = model.formula.split("~", 1)[0].strip()
    upper_formula = f"{lhs} ~ {scope}"

    if isinstance(model, glm):
        upper_model = glm(
            upper_formula, model.data, family=model.family,
            weights=model._prior_w,
        )
        return _add1_glm(model, upper_model._expanded.terms, test=test, k=k)
    if isinstance(model, lm):
        upper_model = lm(
            upper_formula, model.data, weights=model.weights, method=model.method,
        )
        return _add1_lm(model, upper_model._expanded.terms, test=test, k=k)
    raise TypeError(f"add1(): unsupported model type {type(model).__name__}")


def _add1_lm(m: lm, upper_terms, *, test: str | None, k: float):
    """Refit-with-each-term-added implementation behind ``add1(lm)``."""
    if test is not None and test.upper() != "F":
        raise ValueError(
            f"add1(lm): test must be 'F' or None; got {test!r}"
        )
    use_F = test is not None

    cur_terms = m._expanded.terms
    add_indices = _add_scope(cur_terms, upper_terms)
    if not add_indices:
        raise ValueError("add1(): no terms in scope for adding to model")

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
        m_aug = lm(sub_formula, m.data, weights=m.weights, method=m.method)
        d_df = df_full - m_aug.df_residuals  # positive (params gained)
        d_rss = rss_full - m_aug.rss          # positive (rss reduction)

        df_col.append(d_df)
        sos_col.append(round(d_rss, 4))
        rss_col.append(round(m_aug.rss, 4))
        aic_col.append(
            round(_extract_aic_lm(m_aug.rss, m_aug.df_residuals, n, k), 4)
        )
        if use_F and d_df > 0:
            # F denom is the *augmented* model's MSE — mirror of drop1's
            # rule (denom is always the bigger model's residual MS).
            mse_aug = m_aug.rss / m_aug.df_residuals
            fstat = (d_rss / d_df) / mse_aug
            p = float(f.sf(fstat, d_df, m_aug.df_residuals))
            f_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            f_col.append(None); p_col.append(None); sig_col.append("")

    cols: dict[str, list] = {
        "":          ["<none>"] + [upper_terms[i].label for i in add_indices],
        "Df":        df_col,
        "Sum of Sq": sos_col,
        "RSS":       rss_col,
        "AIC":       aic_col,
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


def _add1_glm(m: glm, upper_terms, *, test: str | None, k: float):
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
        """``loglik_aug - loglik_cur`` in R's drop1/add1 sign convention.

        For Gaussian (σ unknown) this is ``n*log(dev_cur/dev_aug)`` —
        the proper σ-unknown LRT, positive when the augmented model
        fits better. For non-Gaussian it's ``Δdev / dispersion_cur``
        (R uses the *current* model's Pearson dispersion as the
        normalizing scale, not the augmented model's).
        """
        if fam.name == "gaussian":
            return n * float(np.log(dev_full / dev_aug))
        return (dev_full - dev_aug) / disp_cur

    cur_labels = [t.label for t in cur_terms]
    for i in add_indices:
        t = upper_terms[i]
        new_labels = cur_labels + [t.label]
        sub_rhs = " + ".join(new_labels)
        sub_formula = f"{lhs} ~ {intercept_str} + {sub_rhs}"
        m_aug = glm(sub_formula, m.data, family=fam, weights=m._prior_w)
        d_df = df_full - m_aug.df_residual    # positive
        d_dev = dev_full - m_aug.deviance     # positive
        d_loglik = _delta_loglik(m_aug.deviance) if d_df > 0 else 0.0

        df_col.append(d_df)
        dev_col.append(round(m_aug.deviance, 4))
        # Recalibrated AIC, mirror image of drop1: aic_aug holds
        # dispersion fixed at the current model's value, so AICs are
        # comparable across additions. ``aic_cur - Δloglik + k*Δdf``.
        aic_col.append(round(aic_full_table - d_loglik + k * d_df, 4))
        if kind == "F" and d_df > 0:
            # F denom is the *augmented* model's residual mean deviance,
            # mirror of drop1's "current model's residual mean deviance".
            rms_aug = m_aug.deviance / m_aug.df_residual
            fstat = (d_dev / d_df) / rms_aug
            p = float(f.sf(fstat, d_df, m_aug.df_residual))
            stat_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        elif kind == "Chisq" and d_df > 0:
            stat = d_loglik
            p = float(chi2.sf(stat, d_df))
            stat_col.append(round(stat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            stat_col.append(None); p_col.append(None); sig_col.append("")

    cols: dict[str, list] = {
        "":         ["<none>"] + [upper_terms[i].label for i in add_indices],
        "Df":       df_col,
        "Deviance": dev_col,
        "AIC":      aic_col,
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
    """The AIC R's ``step()`` minimizes — extractAIC formula.

    For lm it's ``n*log(RSS/n) + k*p`` (Mallows-style); for glm it's
    ``glm.AIC + (k-2)*edf``. Both reduce to standard AIC at ``k=2``;
    differ at ``k=log(n)`` (BIC) or other custom penalties.
    """
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
    if isinstance(model, lme):
        raise NotImplementedError("step(lme): not implemented yet")
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

    def _refit(formula: str):
        if is_glm:
            return glm(
                formula, model.data, family=model.family,
                weights=model._prior_w,
            )
        return lm(formula, model.data, weights=model.weights, method=model.method)

    upper_model = _refit(f"{lhs} ~ {upper_rhs}")
    upper_terms = upper_model._expanded.terms
    lower_model = _refit(f"{lhs} ~ {lower_rhs}")
    lower_label_set = {t.label for t in lower_model._expanded.terms}

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
                rest = [
                    cur_terms[i].label for i in range(len(cur_terms)) if i != j
                ]
                sub_rhs_str = " + ".join(rest)
                sub_formula = (
                    f"{lhs} ~ {intercept_str} + {sub_rhs_str}" if sub_rhs_str
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
    """R-style step trace: ``<none>`` + each candidate, sorted by AIC.

    Df is shown as the *absolute* change (the prefix ``-``/``+`` carries
    the direction). For lm we show ``Sum of Sq`` + ``RSS``; for glm
    just ``Deviance`` (R's convention). AIC is rounded to 2 decimals,
    matching R's printed output.
    """
    rows: list[tuple] = []
    if is_glm:
        rows.append(("<none>", None, current.deviance, cur_aic))
        for label, sub, aic in candidates:
            df_diff = abs(sub.df_residual - current.df_residual)
            rows.append((label, df_diff, sub.deviance, aic))
        rows.sort(key=lambda r: r[3])
        df_table = pl.DataFrame({
            "":         [r[0] for r in rows],
            "Df":       [r[1] for r in rows],
            "Deviance": [round(r[2], 4) for r in rows],
            "AIC":      [round(r[3], 2) for r in rows],
        })
    else:
        rows.append(("<none>", None, None, current.rss, cur_aic))
        for label, sub, aic in candidates:
            df_diff = abs(sub.df_residuals - current.df_residuals)
            d_rss = abs(sub.rss - current.rss)
            rows.append((label, df_diff, d_rss, sub.rss, aic))
        rows.sort(key=lambda r: r[4])
        df_table = pl.DataFrame({
            "":          [r[0] for r in rows],
            "Df":        [r[1] for r in rows],
            "Sum of Sq": [round(r[2], 4) if r[2] is not None else None for r in rows],
            "RSS":       [round(r[3], 4) for r in rows],
            "AIC":       [round(r[4], 2) for r in rows],
        })
    print()
    print(format_df(df_table))


def _extract_aic_lm(rss: float, df_residuals: int, n: int, k: float) -> float:
    """R's ``extractAIC.lm`` — Mallows-style ``n*log(RSS/n) + k*p``.

    Differs from ``AIC.lm`` (the logLik-based formula) by a constant
    that depends only on ``n``, so the differences across nested fits
    in a drop1 table are the same either way — we use this form to
    match R's printed values directly.
    """
    p = n - df_residuals
    if rss <= 0:
        return float("-inf")
    return n * float(np.log(rss / n)) + k * p


def _drop1_lm(m: lm, *, test: str | None, k: float):
    """Refit-without-each-term implementation behind ``drop1(lm)``."""
    if test is not None and test.upper() != "F":
        raise ValueError(
            f"drop1(lm): test must be 'F' or None; got {test!r}"
        )
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

    df_col: list[int | None] = [None]
    sos_col: list[float | None] = [None]
    rss_col: list[float] = [round(rss_full, 4)]
    aic_col: list[float] = [round(_extract_aic_lm(rss_full, df_full, n, k), 4)]
    f_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]

    for j in scope:
        t = terms[j]
        rest = [terms[i].label for i in range(len(terms)) if i != j]
        sub_rhs = " + ".join(rest) if rest else ""
        sub_formula = (
            f"{lhs} ~ {intercept_str} + {sub_rhs}" if sub_rhs
            else f"{lhs} ~ {intercept_str}"
        )
        m_sub = lm(sub_formula, m.data, weights=m.weights, method=m.method)
        d_df = m_sub.df_residuals - df_full
        d_rss = m_sub.rss - rss_full

        df_col.append(d_df)
        sos_col.append(round(d_rss, 4))
        rss_col.append(round(m_sub.rss, 4))
        aic_col.append(round(_extract_aic_lm(m_sub.rss, m_sub.df_residuals, n, k), 4))
        if use_F and d_df > 0:
            fstat = (d_rss / d_df) / mse_full
            p = float(f.sf(fstat, d_df, df_full))
            f_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            f_col.append(None); p_col.append(None); sig_col.append("")

    cols: dict[str, list] = {
        "":          ["<none>"] + [terms[j].label for j in scope],
        "Df":        df_col,
        "Sum of Sq": sos_col,
        "RSS":       rss_col,
        "AIC":       aic_col,
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
    # R's ``drop1.glm`` uses ``extractAIC(full)[2]`` for the <none> row
    # and shifts every other row by the same constant. extractAIC.glm
    # gives ``$aic + (k-2)*edf``, equivalent to standard glm AIC at k=2.
    edf_full = n - df_full
    aic_full_table = m.AIC + (k - 2.0) * edf_full

    scope = _drop_scope(terms)

    df_col: list[int | None] = [None]
    dev_col: list[float] = [round(dev_full, 4)]
    aic_col: list[float] = [round(aic_full_table, 4)]
    stat_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]

    def _delta_loglik(dev_drop: float) -> float:
        """R's drop1.glm "loglik diff" between the dropped and full fit.

        For Gaussian (σ unknown) this is the profile-likelihood form
        ``n*log(dev_drop/dev_full)``. For other families it's
        ``Δdev/dispersion_full`` (which equals Δdev when dispersion=1
        for scale-known Poisson/Binomial). Driving both the Chisq stat
        column ("scaled dev." / "LRT") and the per-row AIC shift.
        """
        if fam.name == "gaussian":
            return n * float(np.log(dev_drop / dev_full))
        return (dev_drop - dev_full) / disp_full

    for j in scope:
        t = terms[j]
        rest = [terms[i].label for i in range(len(terms)) if i != j]
        sub_rhs = " + ".join(rest) if rest else ""
        sub_formula = (
            f"{lhs} ~ {intercept_str} + {sub_rhs}" if sub_rhs
            else f"{lhs} ~ {intercept_str}"
        )
        m_sub = glm(sub_formula, m.data, family=fam, weights=m._prior_w)
        d_df = m_sub.df_residual - df_full
        d_dev = m_sub.deviance - dev_full
        d_loglik = _delta_loglik(m_sub.deviance) if d_df > 0 else 0.0

        df_col.append(d_df)
        dev_col.append(round(m_sub.deviance, 4))
        # Recalibrated AIC matches R: aic_full_table + Δloglik - k*Δdf.
        # Holds dispersion fixed at the full model's value across all
        # rows, so AICs are directly comparable across drops (which is
        # the whole point of drop1's table). For Gaussian this happens
        # to coincide with the standard glm.AIC, but for Gamma/IG the
        # standard AIC re-estimates dispersion per fit and would shift
        # the dropped rows non-uniformly.
        aic_col.append(round(aic_full_table + d_loglik - k * d_df, 4))
        if kind == "F" and d_df > 0:
            # R's ``drop1.glm`` F-denominator is ``dev_full / df_full``
            # (residual *mean deviance*), not the Pearson dispersion
            # ``summary(m)$dispersion`` that ``anova.glm`` uses. The two
            # coincide for Gaussian but differ for Gamma/IG/etc; matches
            # R's behavior (which warns "F test assumes 'quasi' family"
            # for Poisson/Binomial since ``dev_full/df_full`` is then a
            # quasi-likelihood-style scale rather than 1).
            rms_full = dev_full / df_full
            fstat = (d_dev / d_df) / rms_full
            p = float(f.sf(fstat, d_df, df_full))
            stat_col.append(round(fstat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        elif kind == "Chisq" and d_df > 0:
            stat = d_loglik
            p = float(chi2.sf(stat, d_df))
            stat_col.append(round(stat, 4))
            p_col.append(float(f"{p:.4g}"))
            sig_col.append(significance_code([p])[0])
        else:
            stat_col.append(None); p_col.append(None); sig_col.append("")

    cols: dict[str, list] = {
        "":         ["<none>"] + [terms[j].label for j in scope],
        "Df":       df_col,
        "Deviance": dev_col,
        "AIC":      aic_col,
    }
    if kind == "F":
        cols["F value"] = stat_col
        cols["Pr(>F)"] = p_col
        cols[" "] = sig_col
    elif kind == "Chisq":
        # R's drop1.glm flips the column name on scale-known-ness.
        stat_lbl = "LRT" if fam.scale_known else "scaled dev."
        cols[stat_lbl] = stat_col
        cols["Pr(>Chi)"] = p_col
        cols[" "] = sig_col

    print(f"Single term deletions\n\nModel:\n{m.formula}\n")
    print(format_df(pl.DataFrame(cols)))
    if kind:
        print("---")
        print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_lm(*models, labels: list[str]):
    """F-test ANOVA table comparing nested ``lm`` fits."""
    # Sort ascending by npar (= descending by df_residuals, matching R).
    order = sorted(range(len(models)), key=lambda i: models[i].df_residuals,
                   reverse=True)

    dfs  = [models[i].df_residuals for i in order]
    rss  = [models[i].rss           for i in order]
    # R uses the largest (least-constrained) model's MSE as the F denom.
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
            df_col.append(d_df); sos_col.append(round(d_rss, 3))
            f_col.append(None); p_col.append(None); sig_col.append("")
            continue
        fstat = (d_rss / d_df) / mse_full
        p = float(f.sf(fstat, d_df, dfs[-1]))
        df_col.append(d_df)
        sos_col.append(round(d_rss, 3))
        f_col.append(round(fstat, 3))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame({
        "":          [labels[i] for i in order],
        "Res.Df":    dfs,
        "RSS":       [round(r, 3) for r in rss],
        "Df":        df_col,
        "Sum of Sq": sos_col,
        "F":         f_col,
        "Pr(>F)":    p_col,
        " ":         sig_col,
    })

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_lm_single(m: lm):
    """Sequential (Type I) ANOVA — R's ``anova.lm(m)`` for a single fit.

    Refits the model with terms added one at a time in formula order,
    attributing each step's drop in RSS to that term. F = MS_term /
    MS_residual_full, p = upper-tail F. R uses QR-incremental SS, which
    is bit-equivalent for full-rank designs; refitting is conceptually
    simpler and reuses hea's existing rank-deficiency handling.
    """
    terms = m._expanded.terms
    if not terms:
        raise TypeError(
            "anova(m): single-model form needs at least one RHS term "
            "(got an intercept-only model)"
        )

    lhs = m.formula.split("~", 1)[0].strip()
    intercept_str = "1" if m._expanded.intercept else "0"

    def cumulative_formula(k: int) -> str:
        if k == 0:
            return f"{lhs} ~ {intercept_str}"
        rhs = " + ".join(t.label for t in terms[:k])
        return f"{lhs} ~ {intercept_str} + {rhs}"

    rss_chain: list[float] = []
    df_chain: list[int] = []
    for k in range(len(terms)):
        m_k = lm(cumulative_formula(k), m.data,
                 weights=m.weights, method=m.method)
        rss_chain.append(m_k.rss)
        df_chain.append(m_k.df_residuals)
    # Last entry = the original full model — reuse its values directly to
    # avoid a redundant refit and any floating-point drift from re-solving.
    rss_chain.append(m.rss)
    df_chain.append(m.df_residuals)

    mse_full = m.rss / m.df_residuals

    df_col: list[int] = []
    sos_col: list[float] = []
    ms_col: list[float] = []
    f_col: list[float | None] = []
    p_col: list[float | None] = []
    sig_col: list[str] = []
    for i, t in enumerate(terms):
        d_df = df_chain[i] - df_chain[i + 1]
        d_rss = rss_chain[i] - rss_chain[i + 1]
        if d_df <= 0:
            df_col.append(d_df); sos_col.append(round(d_rss, 4))
            ms_col.append(float("nan"))
            f_col.append(None); p_col.append(None); sig_col.append("")
            continue
        ms = d_rss / d_df
        fstat = ms / mse_full
        p = float(f.sf(fstat, d_df, m.df_residuals))
        df_col.append(d_df); sos_col.append(round(d_rss, 4))
        ms_col.append(round(ms, 4))
        f_col.append(round(fstat, 4))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])
    # Residuals row
    df_col.append(m.df_residuals); sos_col.append(round(m.rss, 4))
    ms_col.append(round(mse_full, 4))
    f_col.append(None); p_col.append(None); sig_col.append("")

    docstring = "Analysis of Variance Table\n\n"
    docstring += f"Response: {lhs}\n"

    df_ = pl.DataFrame({
        "":         [t.label for t in terms] + ["Residuals"],
        "Df":       df_col,
        "Sum Sq":   sos_col,
        "Mean Sq":  ms_col,
        "F value":  f_col,
        "Pr(>F)":   p_col,
        " ":        sig_col,
    })

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_gam_single(m: gam):
    """``anova.gam``-style single-model output: parametric Terms F-table
    plus the smooth significance table. Mirrors mgcv's ``anova.gam`` for
    a single fit (which omits the lm-coefficient details that
    ``summary.gam`` prints).

    Per-term parametric F is the joint Wald test
    ``F = β_t' Vp_t⁻¹ β_t / k_t``, with ``k_t`` = number of model-matrix
    columns the term contributes (1 for a numeric, ``L−1`` for an
    ``L``-level factor). Term→column mapping is by name prefix — works
    for the common factor / numeric / simple-interaction cases.
    """
    digits = 4

    out = []
    out.append("")
    out.append(f"Family: {m.family.name}")
    out.append(f"Link function: {m.family.link.name}")
    out.append("")
    out.append(f"Formula: {m.formula}")
    out.append("")

    # ---- Parametric Terms (per-term joint Wald F) -----------------------
    # Collect non-intercept parametric columns. Intercept is excluded — mgcv
    # follows the same convention in anova.gam's pTerms.table.
    cols = m.parametric_columns
    col_idx = {c: i for i, c in enumerate(cols)}
    used = {"(Intercept)"} if "(Intercept)" in col_idx else set()

    rows: list[tuple[str, int, float, float]] = []
    if m._expanded.terms:
        for term in m._expanded.terms:
            label = term.label
            # Match: a column belongs to ``term`` if it equals the label
            # exactly (numeric term) or starts with the label (factor /
            # interaction). Pick the longest label match per column to
            # avoid e.g. ``Hclass`` claiming ``Hclassmedium:Girth`` when
            # the interaction term ``Hclass:Girth`` exists.
            term_cols = [
                c for c in cols
                if c not in used and (c == label or c.startswith(label))
            ]
            if not term_cols:
                continue
            used.update(term_cols)
            idx = np.array([col_idx[c] for c in term_cols], dtype=int)
            beta_t = m._beta[idx]
            Vp_t = m.Vp[np.ix_(idx, idx)]
            k = len(idx)
            try:
                solved = np.linalg.solve(Vp_t, beta_t)
                F_stat = float(beta_t @ solved) / k
            except np.linalg.LinAlgError:
                F_stat = float("nan")
            df_resid = float(m.df_residuals)
            if np.isfinite(F_stat) and df_resid > 0:
                p_val = float(f.sf(F_stat, k, df_resid))
            else:
                p_val = float("nan")
            rows.append((label, k, F_stat, p_val))

    if rows:
        sig = significance_code([r[3] for r in rows])
        tbl = pl.DataFrame({
            "":        [r[0] for r in rows],
            "df":      [r[1] for r in rows],
            "F":       format_signif([r[2] for r in rows], digits=digits),
            "p-value": format_pval([r[3] for r in rows],
                                   digits=_dig_tst(digits)),
            " ":       sig,
        })
        out.append("Parametric Terms:")
        out.append(format_df(
            tbl,
            align={c: "right" for c in ("df", "F", "p-value")},
        ))
        out.append("---")
        out.append(
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )
        out.append("")

    # ---- Smooth significance table (same logic as gam.summary) ----------
    if m._blocks:
        from scipy.stats import f as _f_dist
        rows_label: list[str] = []
        rows_edf: list[float] = []
        rows_refdf: list[float] = []
        rows_F: list[float] = []
        rows_p: list[float] = []
        for b, (a, bcol) in zip(m._blocks, m._block_col_ranges):
            beta_b = m._beta[a:bcol]
            Vp_b = m.Vp[a:bcol, a:bcol]
            X_b = m._X_full[:, a:bcol]
            edf_b = float(m.edf[a:bcol].sum())
            edf1_b = (
                float(m.edf1[a:bcol].sum())
                if hasattr(m, "edf1") else edf_b
            )
            p_b = bcol - a
            rank = float(min(p_b, edf1_b))
            Tr, ref_df = m._test_stat_type0(X_b, Vp_b, beta_b, rank)
            F_smooth = Tr / max(ref_df, 1e-8)
            p_smooth = (
                float(_f_dist.sf(F_smooth, ref_df, m.df_residuals))
                if m.df_residuals > 0 else float("nan")
            )
            rows_label.append(b.label)
            rows_edf.append(edf_b)
            rows_refdf.append(edf1_b)
            rows_F.append(F_smooth)
            rows_p.append(p_smooth)
        sig_smooth = significance_code(rows_p)
        sm_tbl = pl.DataFrame({
            "":        rows_label,
            "edf":     format_signif(rows_edf, digits=digits),
            "Ref.df":  format_signif(rows_refdf, digits=digits),
            "F":       format_signif(rows_F, digits=digits),
            "p-value": format_pval(rows_p, digits=_dig_tst(digits)),
            " ":       sig_smooth,
        })
        out.append("Approximate significance of smooth terms:")
        out.append(format_df(
            sm_tbl,
            align={c: "right" for c in ("edf", "Ref.df", "F", "p-value")},
        ))
        out.append("---")
        out.append(
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )

    print("\n".join(out))


def _anova_gam_rdf(g: gam) -> float:
    """mgcv-style residual df for a ``gam`` in a multi-model anova table.

    mgcv's ``anova.gam`` *overrides* each fit's ``df.residual`` before
    handing the list to ``stats::anova.glmlist``: it uses ``edf1`` (not
    ``edf``) — the "1-step" effective df designed for hypothesis testing
    — minus a smoothing-parameter-uncertainty correction ``dfc`` when
    the fit carries a separate ``edf2`` (REML with unconditional
    correction). For default GCV fits, ``edf2`` is ``NULL`` and
    ``dfc = 0``, so the formula reduces to ``n - sum(edf1)``.

    hea's gam always exposes ``edf2``, but sets it equal to ``edf1`` as
    the no-op sentinel when the unconditional correction wasn't
    computed — we detect that with ``allclose`` and zero out ``dfc``,
    matching mgcv's NULL branch numerically.
    """
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
    """Approximate F / Chisq deviance table for nested ``gam`` fits.

    Mirrors mgcv's ``anova.gam`` for multiple gam objects: the residual
    df uses ``edf1`` (see ``_anova_gam_rdf``), the F denominator is the
    largest model's ``scale`` (mgcv's ``sig2``), and the test selection
    follows the same auto-pick rule as ``anova.glm`` — ``Chisq`` for
    scale-known families (Poisson/Binomial), ``F`` for unknown-scale
    (Gaussian/Gamma/IG).
    """
    df_, docstring = _anova_gam_table(*models, labels=labels, test=test)
    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_gam_table(*models: gam, labels: list[str], test: str | None = None):
    """Pure builder for the multi-model ``anova(gam, ...)`` table.

    Returns ``(df, docstring)``. See ``_anova_gam`` for semantics.
    """
    fam0 = models[0].family
    if not all(type(m.family) is type(fam0) and
               m.family.link.name == fam0.link.name for m in models):
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

    # Sort ascending by npar (= descending by edf1-residual df), matching
    # mgcv. Smallest model first; full model last.
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
            stat_col.append(None); p_col.append(None); sig_col.append("")
            continue
        if test == "Chisq":
            stat = d_dev / disp_full
            p = float(chi2.sf(stat, d_df))
        else:  # "F"
            stat = (d_dev / d_df) / disp_full
            p = float(f.sf(stat, d_df, rdf_full))
        df_col.append(round(d_df, 4))
        dev_col.append(round(d_dev, 4))
        stat_col.append(round(stat, 4))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Deviance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    cols: dict[str, list] = {
        "":           [labels[i] for i in order],
        "Resid. Df":  [round(r, 4) for r in rdfs_sorted],
        "Resid. Dev": [round(d, 4) for d in devs_sorted],
        "Df":         df_col,
        "Deviance":   dev_col,
    }
    if test == "F":
        cols["F"] = stat_col
        cols["Pr(>F)"] = p_col
    else:
        cols["Pr(>Chi)"] = p_col
    cols[" "] = sig_col

    return pl.DataFrame(cols), docstring


def _anova_glm(*models, labels: list[str], test: str | None = None):
    """``anova.glm``-style deviance table for nested ``glm`` fits.

    With ``test=None`` we auto-pick (matches R's recommendation):
    - scale-known families (Poisson, Binomial) → ``Chisq`` LRT on Δdev.
    - unknown-scale families (Gaussian, Gamma, IG) → ``F``.

    Override via ``test=``:
    - ``"Chisq"`` / ``"LRT"`` (alias) → ``Δdev / dispersion_full ~ χ²(Δdf)``.
      For scale-known families ``dispersion_full = 1`` so this is just Δdev,
      matching the auto-pick. For unknown-scale, the division is the
      asymptotic chi-square test (R's ``anova.glm`` does the same).
    - ``"F"`` → ``F = (Δdev / Δdf) / dispersion_full`` against ``F(Δdf,
      df_residual_full)``. Allowed for scale-known families too (R does)
      though the chi-square version is preferred.
    - ``"Rao"`` → score test, not implemented yet.

    Three-or-more models are walked incrementally (row k vs row k-1 after
    sorting by ``df_residuals`` descending, matching ``_anova_lm``).
    """
    df_, docstring = _anova_glm_table(*models, labels=labels, test=test)
    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_glm_table(*models, labels: list[str], test: str | None = None):
    """Pure builder for the ``anova(glm,...)`` table.

    Returns ``(df, docstring)``. Used by ``_anova_glm`` (which prints) and
    by tests that need to inspect column values directly. See ``_anova_glm``
    for the semantics of ``test=``.
    """
    fam0 = models[0].family
    if not all(type(m.family) is type(fam0) and
               m.family.link.name == fam0.link.name for m in models):
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

    # Sort ascending by npar (= descending by df_residuals), matching R.
    order = sorted(range(len(models)), key=lambda i: models[i].df_residuals,
                   reverse=True)
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
            df_col.append(d_df); dev_col.append(round(d_dev, 4))
            stat_col.append(None); p_col.append(None); sig_col.append("")
            continue
        if test == "Chisq":
            # disp_full == 1 for scale-known families (Poisson/Binomial),
            # so this matches the canonical LRT there. For unknown-scale
            # it's the asymptotic χ² test on the rescaled deviance — same
            # formula R uses when `test="Chisq"` is passed for Gaussian/
            # Gamma/IG fits.
            stat = d_dev / disp_full
            p = float(chi2.sf(stat, d_df))
        else:
            stat = (d_dev / d_df) / disp_full
            p = float(f.sf(stat, d_df, df_full))
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

    df_ = pl.DataFrame({
        "":           [labels[i] for i in order],
        "Resid. Df":  dfs,
        "Resid. Dev": [round(d, 4) for d in devs],
        "Df":         df_col,
        "Deviance":   dev_col,
        stat_lbl:     stat_col,
        p_lbl:        p_col,
        " ":          sig_col,
    })
    return df_, docstring


def _anova_lme(*models, labels: list[str]):
    """Likelihood-ratio test for nested ``lme`` fits (lme4-style)."""
    # LRT requires ML; silently refit any REML inputs.
    refit = any(m.REML for m in models)
    models = tuple(
        (lme(m.formula, m.data, REML=False) if m.REML else m) for m in models
    )
    if refit:
        print("refitting model(s) with ML (instead of REML)")
    # Sort ascending by npar, preserving original indices for row labels.
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
        aic_col.append(round(m.AIC, 4))
        bic_col.append(round(m.BIC, 4))
        ll_col.append(round(m.loglike, 4))
        dev_col.append(round(m.deviance, 4))
        if k == 0:
            chi_col.append(None); dfc_col.append(None); p_col.append(None); sig_col.append("")
            continue
        prev = models[order[k - 1]]
        chisq = prev.deviance - m.deviance
        d_df = m.npar - prev.npar
        p = float(chi2.sf(chisq, d_df)) if d_df > 0 else float("nan")
        chi_col.append(round(chisq, 4))
        dfc_col.append(d_df)
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table (likelihood ratio test)\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame({
        "":           [labels[i] for i in order],
        "npar":       npar_col,
        "AIC":        aic_col,
        "BIC":        bic_col,
        "logLik":     ll_col,
        "deviance":   dev_col,
        "Chisq":      chi_col,
        "Df":         dfc_col,
        "Pr(>Chisq)": p_col,
        " ":          sig_col,
    })

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
