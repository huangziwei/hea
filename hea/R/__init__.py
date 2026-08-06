"""R-like free-function API for hea.

Designed for ``from hea.R import *`` so R muscle memory works directly:
``head(df)``, ``pnorm(1.96)``, ``mean(x)``, ``sd(x)``, ``factor(s)``, etc.

Design rules
------------
* **No builtin shadowing.** Names that would clobber Python builtins on
  ``import *`` (``range``, ``min``, ``max``, ``sum``, ``round``, ``abs``,
  ``format``, ``print``, ``list``, ``dict``, ``set``, ``type``, ``len``,
  ``filter``, ``map``, ``zip``, ``sorted``, ``reversed``, ``all``, ``any``,
  …) are intentionally NOT exported. Use numpy / Python equivalents.
* **Polars name collisions are OK.** No one does ``from polars import *``,
  so ``head``, ``mean``, ``var``, ``filter``, ``sort`` etc. are safe to
  redefine here.
* **R's ``c()`` is skipped.** A single-letter glob would clobber loop
  variables. Use ``np.array([...])`` or a Python list.
* **R's ``df()`` (PDF of the F distribution) is skipped.** ``df`` is too
  commonly used as a DataFrame variable. Use ``scipy.stats.f.pdf``;
  ``pf`` / ``qf`` / ``rf`` are exposed for CDF / quantile / random.
* **Sequence and indexing functions are 0-based.** ``which``, ``which_max``,
  ``which_min``, ``order`` and the one-arg ``seq(n)`` / ``seq_len`` /
  ``seq_along`` all match Python conventions. For R's ``1:n`` muscle
  memory, write ``seq(1, n)`` explicitly — the two-arg ``seq(start, stop)``
  form is still inclusive on both ends, so ``seq(1, 5)`` gives
  ``[1, 2, 3, 4, 5]``.
* **R parameter names preserved where possible.** ``mean=`` / ``sd=`` /
  ``df=`` / ``shape=`` / ``rate=`` / ``prob=``. R's ``lower.tail``
  becomes ``lower_tail``. R's ``na.rm`` becomes ``na_rm``. R's
  ``lambda=`` becomes ``lambda_=`` (Python keyword).

Implementation
--------------
This package is the result of splitting the legacy ``hea/R.py`` (one
6k-line file) into base-R-shaped divisions:

* :mod:`hea.R.factor` — ``factor``, ``ordered``, ``levels`` / ``nlevels``,
  ``interaction``, ``is_factor``.
* :mod:`hea.R.shape` — ``head`` / ``tail``, ``nrow`` / ``ncol`` / ``dim``,
  ``names`` / ``colnames``, ``summary``, ``complete_cases`` / ``na_omit``.
* :mod:`hea.R.plotmath` — ``cat``, ``quote`` (plotmath → mathtext).
* :mod:`hea.R.vector` — ``seq*``, ``rev`` / ``sort`` / ``order``,
  ``which*``, ``cum*``, ``diff``, ``unique`` / ``duplicated``,
  ``tabulate``, ``cut`` / ``findInterval``.
* :mod:`hea.R.stats_summary` — R-default reductions (``mean`` / ``median``
  / ``var`` / ``sd`` / ``IQR`` / ``quantile`` / ``cor`` / ``cov``).
* :mod:`hea.R.math` — elementwise math + constants ``pi`` / ``LETTERS``
  / ``letters``.
* :mod:`hea.R.matrix` — ``rowSums`` / ``colSums``, ``apply``, ``rbind``
  / ``cbind`` / ``sweep`` / ``expand_grid`` / ``matrix`` / ``rep``,
  ``R_range`` / ``R_round``.
* :mod:`hea.R.coerce` — ``as_numeric`` / ``as_integer`` /
  ``as_character`` / ``as_logical`` / ``as_date``.
* :mod:`hea.R.predicates` — ``is_na`` / ``is_null`` / ``is_finite`` /
  ``is_numeric``.
* :mod:`hea.R.distributions` — ``dnorm`` / ``pnorm`` / ``qnorm`` /
  ``rnorm`` and the other d/p/q/r families, ``set_seed`` / ``sample``.
* :mod:`hea.R.functional` — ``tapply`` / ``sapply``.
* :mod:`hea.R.tables` — ``table`` / ``xtabs`` / ``prop_table`` /
  ``addmargins``.
* :mod:`hea.R.htest` — :class:`HTest` / :class:`AnovaTable` containers,
  ``rank`` / ``signed_rank``, every ``*_test`` hypothesis test, ``aov``.
* :mod:`hea.R.model_generics` — ``coef`` / ``predict`` / ``residuals`` /
  ``fitted`` / ``vcov`` / ``logLik`` / ``deviance`` / ``nobs`` /
  ``df_residual`` / ``formula`` / ``model_matrix`` / ``model_frame``,
  ``terms`` / :class:`Terms`, ``update``, ``AIC`` / ``BIC``.
* :mod:`hea.R.diagnostics` — ``hatvalues`` / ``rstandard`` / ``rstudent``
  / ``cooks_distance`` / ``dffits`` / ``dfbetas`` / ``influence``.
* :mod:`hea.R.model_selection` — ``anova`` / ``add1`` / ``drop1`` /
  ``step``.
"""

from __future__ import annotations

#: Every name this namespace re-exports, mapped to the sub-module that defines
#: it, so nothing is imported until it is touched. Eager re-export made a
#: *cycle*: `hea.family` needs `hea.R.nmath`, importing it runs this file, and
#: this file used to pull `htest` and `model_selection`, which import
#: `hea.models` -- whose `bam` imports `hea.family` straight back. It survived
#: only while `hea/__init__.py` happened to load `hea.R` before `hea.family`.
#: It is also weight: nothing in the base-R namespace needs `bam`/`gam`/`glm`
#: at import time, and `hea.R.nmath` is a numeric leaf that should not cost
#: them.
_EXPORTS = {
    # _shared
    "NamedVector": ("_shared", "NamedVector"),
    # factor
    "_LazyFactor": ("factor", "_LazyFactor"),
    "factor": ("factor", "factor"),
    "fct": ("factor", "fct"),
    "interaction": ("factor", "interaction"),
    "is_factor": ("factor", "is_factor"),
    "levels": ("factor", "levels"),
    "nlevels": ("factor", "nlevels"),
    "ordered": ("factor", "ordered"),
    # shape
    "colnames": ("shape", "colnames"),
    "complete_cases": ("shape", "complete_cases"),
    "dim": ("shape", "dim"),
    "head": ("shape", "head"),
    "length": ("shape", "length"),
    "na_omit": ("shape", "na_omit"),
    "names": ("shape", "names"),
    "ncol": ("shape", "ncol"),
    "nrow": ("shape", "nrow"),
    "summary": ("shape", "summary"),
    "tail": ("shape", "tail"),
    # plotmath
    "cat": ("plotmath", "cat"),
    "quote": ("plotmath", "quote"),
    # vector
    "cummax": ("vector", "cummax"),
    "cummin": ("vector", "cummin"),
    "cumprod": ("vector", "cumprod"),
    "cumsum": ("vector", "cumsum"),
    "cut": ("vector", "cut"),
    "diff": ("vector", "diff"),
    "duplicated": ("vector", "duplicated"),
    "findInterval": ("vector", "findInterval"),
    "order": ("vector", "order"),
    "rev": ("vector", "rev"),
    "seq": ("vector", "seq"),
    "seq_along": ("vector", "seq_along"),
    "seq_len": ("vector", "seq_len"),
    "sort": ("vector", "sort"),
    "tabulate": ("vector", "tabulate"),
    "unique": ("vector", "unique"),
    "which": ("vector", "which"),
    "which_max": ("vector", "which_max"),
    "which_min": ("vector", "which_min"),
    # stats_summary
    "IQR": ("stats_summary", "IQR"),
    "cor": ("stats_summary", "cor"),
    "cov": ("stats_summary", "cov"),
    "mean": ("stats_summary", "mean"),
    "median": ("stats_summary", "median"),
    "quantile": ("stats_summary", "quantile"),
    "sd": ("stats_summary", "sd"),
    "var": ("stats_summary", "var"),
    # math
    "LETTERS": ("math", "LETTERS"),
    "abs": ("math", "abs"),
    "acos": ("math", "acos"),
    "asin": ("math", "asin"),
    "atan": ("math", "atan"),
    "atan2": ("math", "atan2"),
    "ceiling": ("math", "ceiling"),
    "cos": ("math", "cos"),
    "exp": ("math", "exp"),
    "expm1": ("math", "expm1"),
    "floor": ("math", "floor"),
    "letters": ("math", "letters"),
    "log": ("math", "log"),
    "log10": ("math", "log10"),
    "log1p": ("math", "log1p"),
    "log2": ("math", "log2"),
    "pi": ("math", "pi"),
    "round": ("math", "round"),
    "sign": ("math", "sign"),
    "sin": ("math", "sin"),
    "sqrt": ("math", "sqrt"),
    "tan": ("math", "tan"),
    "trunc": ("math", "trunc"),
    # matrix
    "R_range": ("matrix", "R_range"),
    "R_round": ("matrix", "R_round"),
    "apply": ("matrix", "apply"),
    "cbind": ("matrix", "cbind"),
    "colMeans": ("matrix", "colMeans"),
    "colSums": ("matrix", "colSums"),
    "expand_grid": ("matrix", "expand_grid"),
    "matrix": ("matrix", "matrix"),
    "rbind": ("matrix", "rbind"),
    "rep": ("matrix", "rep"),
    "rowMeans": ("matrix", "rowMeans"),
    "rowSums": ("matrix", "rowSums"),
    "sweep": ("matrix", "sweep"),
    # coerce
    "as_Date": ("coerce", "as_Date"),
    "as_character": ("coerce", "as_character"),
    "as_date": ("coerce", "as_date"),
    "as_integer": ("coerce", "as_integer"),
    "as_logical": ("coerce", "as_logical"),
    "as_numeric": ("coerce", "as_numeric"),
    # predicates
    "is_finite": ("predicates", "is_finite"),
    "is_na": ("predicates", "is_na"),
    "is_null": ("predicates", "is_null"),
    "is_numeric": ("predicates", "is_numeric"),
    # distance
    "Dist": ("distance", "Dist"),
    "as_dist": ("distance", "as_dist"),
    "as_matrix_dist": ("distance", "as_matrix_dist"),
    "cmdscale": ("distance", "cmdscale"),
    "dist": ("distance", "dist"),
    "format_dist": ("distance", "format_dist"),
    "labels_dist": ("distance", "labels_dist"),
    "mahalanobis": ("distance", "mahalanobis"),
    "print_dist": ("distance", "print_dist"),
    # clustering
    "Dendrogram": ("clustering", "Dendrogram"),
    "Hclust": ("clustering", "Hclust"),
    "Kmeans": ("clustering", "Kmeans"),
    "as_dendrogram": ("clustering", "as_dendrogram"),
    "as_hclust": ("clustering", "as_hclust"),
    "cophenetic": ("clustering", "cophenetic"),
    "cophenetic_dendrogram": ("clustering", "cophenetic_dendrogram"),
    "cut_dendrogram": ("clustering", "cut_dendrogram"),
    "cutree": ("clustering", "cutree"),
    "dendrapply": ("clustering", "dendrapply"),
    "fitted_kmeans": ("clustering", "fitted_kmeans"),
    "hclust": ("clustering", "hclust"),
    "is_leaf": ("clustering", "is_leaf"),
    "kmeans": ("clustering", "kmeans"),
    "labels_dendrogram": ("clustering", "labels_dendrogram"),
    "merge_dendrogram": ("clustering", "merge_dendrogram"),
    "midcache_dendrogram": ("clustering", "midcache_dendrogram"),
    "nleaves": ("clustering", "nleaves"),
    "nobs_dendrogram": ("clustering", "nobs_dendrogram"),
    "order_dendrogram": ("clustering", "order_dendrogram"),
    "print_dendrogram": ("clustering", "print_dendrogram"),
    "print_hclust": ("clustering", "print_hclust"),
    "print_kmeans": ("clustering", "print_kmeans"),
    "reorder": ("clustering", "reorder"),
    "reorder_dendrogram": ("clustering", "reorder_dendrogram"),
    "rev_dendrogram": ("clustering", "rev_dendrogram"),
    "str_dendrogram": ("clustering", "str_dendrogram"),
    # distributions
    "dbeta": ("distributions", "dbeta"),
    "dbinom": ("distributions", "dbinom"),
    "dcauchy": ("distributions", "dcauchy"),
    "dchisq": ("distributions", "dchisq"),
    "dexp": ("distributions", "dexp"),
    "dgamma": ("distributions", "dgamma"),
    "dgeom": ("distributions", "dgeom"),
    "dhyper": ("distributions", "dhyper"),
    "dlnorm": ("distributions", "dlnorm"),
    "dlogis": ("distributions", "dlogis"),
    "dmultinom": ("distributions", "dmultinom"),
    "dnbinom": ("distributions", "dnbinom"),
    "dnorm": ("distributions", "dnorm"),
    "dpois": ("distributions", "dpois"),
    "dsignrank": ("distributions", "dsignrank"),
    "dt": ("distributions", "dt"),
    "dunif": ("distributions", "dunif"),
    "dweibull": ("distributions", "dweibull"),
    "dwilcox": ("distributions", "dwilcox"),
    "pbeta": ("distributions", "pbeta"),
    "pbinom": ("distributions", "pbinom"),
    "pbirthday": ("distributions", "pbirthday"),
    "pcauchy": ("distributions", "pcauchy"),
    "pchisq": ("distributions", "pchisq"),
    "pexp": ("distributions", "pexp"),
    "pf": ("distributions", "pf"),
    "pgamma": ("distributions", "pgamma"),
    "pgeom": ("distributions", "pgeom"),
    "phyper": ("distributions", "phyper"),
    "plnorm": ("distributions", "plnorm"),
    "plogis": ("distributions", "plogis"),
    "pnbinom": ("distributions", "pnbinom"),
    "pnorm": ("distributions", "pnorm"),
    "ppois": ("distributions", "ppois"),
    "psignrank": ("distributions", "psignrank"),
    "pt": ("distributions", "pt"),
    "ptukey": ("distributions", "ptukey"),
    "punif": ("distributions", "punif"),
    "pweibull": ("distributions", "pweibull"),
    "pwilcox": ("distributions", "pwilcox"),
    "qbeta": ("distributions", "qbeta"),
    "qbinom": ("distributions", "qbinom"),
    "qbirthday": ("distributions", "qbirthday"),
    "qcauchy": ("distributions", "qcauchy"),
    "qchisq": ("distributions", "qchisq"),
    "qexp": ("distributions", "qexp"),
    "qf": ("distributions", "qf"),
    "qgamma": ("distributions", "qgamma"),
    "qgeom": ("distributions", "qgeom"),
    "qhyper": ("distributions", "qhyper"),
    "qlnorm": ("distributions", "qlnorm"),
    "qlogis": ("distributions", "qlogis"),
    "qnbinom": ("distributions", "qnbinom"),
    "qnorm": ("distributions", "qnorm"),
    "qpois": ("distributions", "qpois"),
    "qsignrank": ("distributions", "qsignrank"),
    "qt": ("distributions", "qt"),
    "qtukey": ("distributions", "qtukey"),
    "qunif": ("distributions", "qunif"),
    "qweibull": ("distributions", "qweibull"),
    "qwilcox": ("distributions", "qwilcox"),
    "r2dtable": ("distributions", "r2dtable"),
    "rWishart": ("distributions", "rWishart"),
    "rbeta": ("distributions", "rbeta"),
    "rbinom": ("distributions", "rbinom"),
    "rcauchy": ("distributions", "rcauchy"),
    "rchisq": ("distributions", "rchisq"),
    "rexp": ("distributions", "rexp"),
    "rf": ("distributions", "rf"),
    "rgamma": ("distributions", "rgamma"),
    "rgeom": ("distributions", "rgeom"),
    "rhyper": ("distributions", "rhyper"),
    "rlnorm": ("distributions", "rlnorm"),
    "rlogis": ("distributions", "rlogis"),
    "rmultinom": ("distributions", "rmultinom"),
    "rnbinom": ("distributions", "rnbinom"),
    "rnorm": ("distributions", "rnorm"),
    "rpois": ("distributions", "rpois"),
    "rsignrank": ("distributions", "rsignrank"),
    "rt": ("distributions", "rt"),
    "runif": ("distributions", "runif"),
    "rweibull": ("distributions", "rweibull"),
    "rwilcox": ("distributions", "rwilcox"),
    "sample": ("distributions", "sample"),
    "set_seed": ("distributions", "set_seed"),
    # functional
    "sapply": ("functional", "sapply"),
    "tapply": ("functional", "tapply"),
    # emmeans
    "EmmGrid": ("emmeans", "EmmGrid"),
    "emmeans": ("emmeans", "emmeans"),
    "summary_emmgrid_contrasts": ("emmeans", "summary_emmgrid_contrasts"),
    # tables
    "addmargins": ("tables", "addmargins"),
    "prop_table": ("tables", "prop_table"),
    "table": ("tables", "table"),
    "xtabs": ("tables", "xtabs"),
    # htest
    "AnovaTable": ("htest", "AnovaTable"),
    "HTest": ("htest", "HTest"),
    "PairwiseHTest": ("htest", "PairwiseHTest"),
    "PowerHTest": ("htest", "PowerHTest"),
    "aov": ("htest", "aov"),
    "ansari_test": ("htest", "ansari_test"),
    "bartlett_test": ("htest", "bartlett_test"),
    "binom_test": ("htest", "binom_test"),
    "chisq_test": ("htest", "chisq_test"),
    "cor_test": ("htest", "cor_test"),
    "fisher_test": ("htest", "fisher_test"),
    "fligner_test": ("htest", "fligner_test"),
    "friedman_test": ("htest", "friedman_test"),
    "kruskal_test": ("htest", "kruskal_test"),
    "ks_test": ("htest", "ks_test"),
    "mantelhaen_test": ("htest", "mantelhaen_test"),
    "mcnemar_test": ("htest", "mcnemar_test"),
    "mood_test": ("htest", "mood_test"),
    "oneway_test": ("htest", "oneway_test"),
    "p_adjust": ("htest", "p_adjust"),
    "p_adjust_methods": ("htest", "p_adjust_methods"),
    "pairwise_prop_test": ("htest", "pairwise_prop_test"),
    "pairwise_t_test": ("htest", "pairwise_t_test"),
    "pairwise_wilcox_test": ("htest", "pairwise_wilcox_test"),
    "poisson_test": ("htest", "poisson_test"),
    "power_anova_test": ("htest", "power_anova_test"),
    "power_prop_test": ("htest", "power_prop_test"),
    "power_t_test": ("htest", "power_t_test"),
    "prop_test": ("htest", "prop_test"),
    "prop_trend_test": ("htest", "prop_trend_test"),
    "quade_test": ("htest", "quade_test"),
    "psmirnov": ("htest", "psmirnov"),
    "qsmirnov": ("htest", "qsmirnov"),
    "rank": ("htest", "rank"),
    "rsmirnov": ("htest", "rsmirnov"),
    "shapiro_test": ("htest", "shapiro_test"),
    "signed_rank": ("htest", "signed_rank"),
    "t_test": ("htest", "t_test"),
    "var_test": ("htest", "var_test"),
    "wilcox_test": ("htest", "wilcox_test"),
    # model_generics
    "AIC": ("model_generics", "AIC"),
    "BIC": ("model_generics", "BIC"),
    "Terms": ("model_generics", "Terms"),
    "bootMer": ("model_generics", "bootMer"),
    "coef": ("model_generics", "coef"),
    "coefficients": ("model_generics", "coefficients"),
    "confint": ("model_generics", "confint"),
    "deviance": ("model_generics", "deviance"),
    "df_residual": ("model_generics", "df_residual"),
    "effects": ("model_generics", "effects"),
    "fitted": ("model_generics", "fitted"),
    "fitted_values": ("model_generics", "fitted_values"),
    "fixef": ("model_generics", "fixef"),
    "formula": ("model_generics", "formula"),
    "profile": ("model_generics", "profile"),
    "simulate": ("model_generics", "simulate"),
    "logLik": ("model_generics", "logLik"),
    "model_frame": ("model_generics", "model_frame"),
    "model_matrix": ("model_generics", "model_matrix"),
    "nobs": ("model_generics", "nobs"),
    "predict": ("model_generics", "predict"),
    "ranef": ("model_generics", "ranef"),
    "refit": ("model_generics", "refit"),
    "refitML": ("model_generics", "refitML"),
    "case_names": ("model_generics", "case_names"),
    "labels": ("model_generics", "labels"),
    "resid": ("model_generics", "resid"),
    "residuals": ("model_generics", "residuals"),
    "terms": ("model_generics", "terms"),
    "update": ("model_generics", "update"),
    "variable_names": ("model_generics", "variable_names"),
    "vcov": ("model_generics", "vcov"),
    "weights": ("model_generics", "weights"),
    "VarCorr": ("model_generics", "VarCorr"),
    "getME": ("model_generics", "getME"),
    "getData": ("model_generics", "getData"),
    "isREML": ("model_generics", "isREML"),
    "isLMM": ("model_generics", "isLMM"),
    "isGLMM": ("model_generics", "isGLMM"),
    "isNLMM": ("model_generics", "isNLMM"),
    "isSingular": ("model_generics", "isSingular"),
    "extractAIC": ("model_generics", "extractAIC"),
    "rePCA": ("model_generics", "rePCA"),
    # formula_helpers
    "DF2formula": ("formula_helpers", "DF2formula"),
    "MFclass": ("formula_helpers", "MFclass"),
    "NAAction": ("formula_helpers", "NAAction"),
    "Poly": ("formula_helpers", "Poly"),
    "as_formula": ("formula_helpers", "as_formula"),
    "delete_response": ("formula_helpers", "delete_response"),
    "drop_terms": ("formula_helpers", "drop_terms"),
    "get_all_vars": ("formula_helpers", "get_all_vars"),
    "na_action": ("formula_helpers", "na_action"),
    "na_exclude": ("formula_helpers", "na_exclude"),
    "na_fail": ("formula_helpers", "na_fail"),
    "na_pass": ("formula_helpers", "na_pass"),
    "napredict": ("formula_helpers", "napredict"),
    "naprint": ("formula_helpers", "naprint"),
    "naresid": ("formula_helpers", "naresid"),
    "poly": ("formula_helpers", "poly"),
    "polym": ("formula_helpers", "polym"),
    "predict_poly": ("formula_helpers", "predict_poly"),
    "reformulate": ("formula_helpers", "reformulate"),
    "update_formula": ("formula_helpers", "update_formula"),
    # diagnostics
    "cooks_distance": ("diagnostics", "cooks_distance"),
    "dfbeta": ("diagnostics", "dfbeta"),
    "dfbetas": ("diagnostics", "dfbetas"),
    "dffits": ("diagnostics", "dffits"),
    "hatvalues": ("diagnostics", "hatvalues"),
    "influence": ("diagnostics", "influence"),
    "rstandard": ("diagnostics", "rstandard"),
    "rstudent": ("diagnostics", "rstudent"),
    # lm_aov_extras
    "Infl": ("lm_aov_extras", "Infl"),
    "cov2cor": ("lm_aov_extras", "cov2cor"),
    "covratio": ("lm_aov_extras", "covratio"),
    "influence_measures": ("lm_aov_extras", "influence_measures"),
    "ls_diag": ("lm_aov_extras", "ls_diag"),
    "ls_print": ("lm_aov_extras", "ls_print"),
    "lsfit": ("lm_aov_extras", "lsfit"),
    "replications": ("lm_aov_extras", "replications"),
    "sigma": ("lm_aov_extras", "sigma"),
    "weighted_residuals": ("lm_aov_extras", "weighted_residuals"),
    # rng
    "RMersenneTwister": ("rng", "RMersenneTwister"),
    # model_selection
    "add1": ("model_selection", "add1"),
    "anova": ("model_selection", "anova"),
    "drop1": ("model_selection", "drop1"),
    "step": ("model_selection", "step"),
    "_anova_gam_rdf": ("model_selection", "_anova_gam_rdf"),
    "_anova_gam_table": ("model_selection", "_anova_gam_table"),
    "_anova_glm_table": ("model_selection", "_anova_glm_table"),
    "_drop1_lm": ("model_selection", "_drop1_lm"),
    "_extract_aic_lm": ("model_selection", "_extract_aic_lm"),
}


#: Names that are both a sub-module here *and* a function this namespace
#: re-exports -- ``hea.R.factor``, ``hea.R.matrix``, ``hea.R.emmeans``. Importing
#: a sub-module binds it into the parent namespace, so it shadows the function.
#: Eagerly that never showed, because the ``from .factor import (...)`` line ran
#: after the sub-module import and the function won; lazily, whoever touches the
#: module first wins, and resolving *any* export from ``.factor`` (``fct``, say)
#: was enough to leave ``hea.R.factor`` pointing at the module. Binding all three
#: eagerly would cost 177 ms against this file's 16, so the shadow is undone at
#: the point it is cast instead.
_SHADOWED = frozenset({"emmeans", "factor", "matrix"})


def __getattr__(name: str):
    """Resolve a re-exported name, or a sub-module, on first access — PEP 562."""
    from importlib import import_module

    entry = _EXPORTS.get(name)
    if entry is not None:
        mod = import_module(f".{entry[0]}", __name__)
        value = getattr(mod, entry[1])
        if entry[0] in _SHADOWED:
            globals()[entry[0]] = getattr(mod, _EXPORTS[entry[0]][1])
    else:
        try:
            value = import_module(f".{name}", __name__)
        except ImportError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_EXPORTS))


def ts(data, start: float = 1.0, frequency: float = 1.0):
    """R: ``ts(data, start, frequency)`` — construct a time series.

    Thin wrapper around :func:`hea.tidy.dataframe.ts` (lazy import to
    avoid the load-order cycle). See that docstring for details.
    """
    from ..tidy.dataframe import ts as _ts

    return _ts(data, start=start, frequency=frequency)


__all__ = [
    # base I/O
    "cat",
    # plotmath
    "quote",
    # shape / preview
    "head",
    "tail",
    "nrow",
    "ncol",
    "dim",
    "length",
    "names",
    "colnames",
    "summary",
    "complete_cases",
    "na_omit",
    # vector helpers
    "seq",
    "seq_len",
    "seq_along",
    "rev",
    "sort",
    "order",
    "which",
    "which_max",
    "which_min",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "diff",
    "unique",
    "duplicated",
    "tabulate",
    "cut",
    "findInterval",
    # contingency tables
    "table",
    "xtabs",
    "prop_table",
    "addmargins",
    # reductions (R defaults: sd/var use N-1)
    "mean",
    "median",
    "var",
    "sd",
    "quantile",
    "IQR",
    "cor",
    "cov",
    # base-R constants
    "LETTERS",
    "letters",
    # elementwise math (R: vectorized scalar functions)
    # Note: ``abs`` and ``round`` exist as module attributes but are NOT
    # exported — they collide with Python builtins and the translator
    # treats the R names as builtins (the builtin handles scalars / Series
    # / ndarrays via __abs__ / __round__).
    "pi",
    "sqrt",
    "exp",
    "log",
    "log2",
    "log10",
    "log1p",
    "expm1",
    "sign",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "floor",
    "ceiling",
    "trunc",
    # vector primitives — R's rep() flattens nested list-of-vector inputs
    # (the translator emits ``c(scalar, vec)`` as a Python list literal).
    "rep",
    # matrix / frame utilities (R: base matrix ops)
    "rowSums",
    "colSums",
    "rowMeans",
    "colMeans",
    "apply",
    "rbind",
    "cbind",
    "sweep",
    "expand_grid",
    "matrix",
    "R_range",
    "R_round",
    # coercion / predicates
    "as_numeric",
    "as_integer",
    "as_character",
    "as_logical",
    "as_date",
    "as_Date",
    "is_na",
    "is_null",
    "is_finite",
    "is_numeric",
    "is_factor",
    "factor",
    "fct",
    "ordered",
    "interaction",
    "levels",
    "nlevels",
    # distance layer (base-R stats: dist / as.dist / as.matrix.dist + accessors)
    "Dist",
    "dist",
    "as_dist",
    "as_matrix_dist",
    "format_dist",
    "labels_dist",
    "print_dist",
    "mahalanobis",
    "cmdscale",
    # clustering (base-R stats: hclust + tree objects)
    "Hclust",
    "hclust",
    "as_hclust",
    "cutree",
    "cophenetic",
    "print_hclust",
    # dendrogram subsystem (base-R stats: dendrogram.R, non-graphics)
    "Dendrogram",
    "as_dendrogram",
    "cophenetic_dendrogram",
    "cut_dendrogram",
    "dendrapply",
    "is_leaf",
    "labels_dendrogram",
    "merge_dendrogram",
    "midcache_dendrogram",
    "nleaves",
    "nobs_dendrogram",
    "order_dendrogram",
    "print_dendrogram",
    "reorder",
    "reorder_dendrogram",
    "rev_dendrogram",
    "str_dendrogram",
    # k-means (base-R stats: Hartigan-Wong / Lloyd / Forgy / MacQueen)
    "Kmeans",
    "kmeans",
    "fitted_kmeans",
    "print_kmeans",
    # distributions: d/p/q/r families
    "dnorm",
    "pnorm",
    "qnorm",
    "rnorm",
    "dt",
    "pt",
    "qt",
    "rt",
    "pf",
    "qf",
    "rf",
    "dchisq",
    "pchisq",
    "qchisq",
    "rchisq",
    "dbinom",
    "pbinom",
    "qbinom",
    "rbinom",
    "dpois",
    "ppois",
    "qpois",
    "rpois",
    "dunif",
    "punif",
    "qunif",
    "runif",
    "ptukey",
    "qtukey",
    "dsignrank",
    "psignrank",
    "qsignrank",
    "rsignrank",
    "dwilcox",
    "pwilcox",
    "qwilcox",
    "rwilcox",
    "dhyper",
    "phyper",
    "qhyper",
    "rhyper",
    "dnbinom",
    "pnbinom",
    "qnbinom",
    "rnbinom",
    "dexp",
    "pexp",
    "qexp",
    "rexp",
    "dgamma",
    "pgamma",
    "qgamma",
    "rgamma",
    "dbeta",
    "pbeta",
    "qbeta",
    "rbeta",
    "dcauchy",
    "pcauchy",
    "qcauchy",
    "rcauchy",
    "dlogis",
    "plogis",
    "qlogis",
    "rlogis",
    "dlnorm",
    "plnorm",
    "qlnorm",
    "rlnorm",
    "dweibull",
    "pweibull",
    "qweibull",
    "rweibull",
    "dgeom",
    "pgeom",
    "qgeom",
    "rgeom",
    # combinatorial / multivariate + Smirnov distribution surface
    "dmultinom",
    "rmultinom",
    "pbirthday",
    "qbirthday",
    "r2dtable",
    "rWishart",
    "psmirnov",
    "qsmirnov",
    "rsmirnov",
    "set_seed",
    "sample",
    "sapply",
    "tapply",
    # rank helpers (Lindeløv-style "tests as lm" notebook)
    "rank",
    "signed_rank",
    # hypothesis tests (return HTest, R's ``htest`` print-shape)
    "HTest",
    "AnovaTable",
    "t_test",
    "wilcox_test",
    "cor_test",
    "kruskal_test",
    "chisq_test",
    "fisher_test",
    "prop_test",
    "binom_test",
    "var_test",
    "bartlett_test",
    "shapiro_test",
    "ks_test",
    "mcnemar_test",
    "friedman_test",
    "aov",
    "p_adjust",
    "p_adjust_methods",
    "oneway_test",
    "fligner_test",
    "mood_test",
    "quade_test",
    "poisson_test",
    "prop_trend_test",
    "mantelhaen_test",
    "ansari_test",
    "pairwise_t_test",
    "pairwise_wilcox_test",
    "pairwise_prop_test",
    "PairwiseHTest",
    "power_t_test",
    "power_prop_test",
    "power_anova_test",
    "PowerHTest",
    # model generics (lm / glm / gam / bam / gmm)
    "coef",
    "coefficients",
    "fixef",
    "ranef",
    "refit",
    "refitML",
    "resid",
    "residuals",
    "fitted",
    "fitted_values",
    "predict",
    "confint",
    "vcov",
    "profile",
    "bootMer",
    "logLik",
    "deviance",
    "nobs",
    "weights",
    "df_residual",
    "effects",
    "simulate",
    "formula",
    "model_matrix",
    "model_frame",
    "AIC",
    "BIC",
    "anova",
    "add1",
    "drop1",
    "step",
    "update",
    "terms",
    "Terms",
    # lme4 merMod accessors (VarCorr / getME / predicates / getData / rePCA)
    "VarCorr",
    "getME",
    "getData",
    "extractAIC",
    "rePCA",
    "isREML",
    "isLMM",
    "isGLMM",
    "isNLMM",
    "isSingular",
    # formula / model-frame helpers (models.R, nafns.R, contr.poly.R)
    "reformulate",
    "as_formula",
    "update_formula",
    "delete_response",
    "drop_terms",
    "DF2formula",
    "get_all_vars",
    "na_pass",
    "na_fail",
    "na_exclude",
    "na_action",
    "naresid",
    "napredict",
    "naprint",
    "NAAction",
    "MFclass",
    "poly",
    "polym",
    "predict_poly",
    "Poly",
    # regression diagnostics
    "hatvalues",
    "rstandard",
    "rstudent",
    "cooks_distance",
    "dffits",
    "dfbeta",
    "dfbetas",
    "influence",
    "variable_names",
    "case_names",
    "labels",
    # lm / aov extras
    "sigma",
    "cov2cor",
    "weighted_residuals",
    "covratio",
    "influence_measures",
    "Infl",
    "lsfit",
    "ls_diag",
    "ls_print",
    "replications",
    # time series construction (R's ts())
    "ts",
    # emmeans (CRAN port, parked here until the surface grows)
    "EmmGrid",
    "emmeans",
    "summary_emmgrid_contrasts",
]
