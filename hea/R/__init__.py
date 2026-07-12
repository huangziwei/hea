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

# Private shared helpers (imported by other R/ submodules but also useful
# as ``hea.R.NamedVector`` for callers building their own named vectors).
from ._shared import NamedVector

# Factors
from .factor import (
    _LazyFactor,
    factor,
    fct,
    interaction,
    is_factor,
    levels,
    nlevels,
    ordered,
)

# Data-frame shape / preview
from .shape import (
    colnames,
    complete_cases,
    dim,
    head,
    length,
    na_omit,
    names,
    ncol,
    nrow,
    summary,
    tail,
)

# Plotmath + cat
from .plotmath import cat, quote

# Vector helpers
from .vector import (
    cummax,
    cummin,
    cumprod,
    cumsum,
    cut,
    diff,
    duplicated,
    findInterval,
    order,
    rev,
    seq,
    seq_along,
    seq_len,
    sort,
    tabulate,
    unique,
    which,
    which_max,
    which_min,
)

# R-shaped summary reductions
from .stats_summary import (
    IQR,
    cor,
    cov,
    mean,
    median,
    quantile,
    sd,
    var,
)

# Constants + elementwise math
from .math import (
    LETTERS,
    abs,
    acos,
    asin,
    atan,
    atan2,
    ceiling,
    cos,
    exp,
    expm1,
    floor,
    letters,
    log,
    log10,
    log1p,
    log2,
    pi,
    round,
    sign,
    sin,
    sqrt,
    tan,
    trunc,
)

# Matrix / frame ops
from .matrix import (
    R_range,
    R_round,
    apply,
    cbind,
    colMeans,
    colSums,
    expand_grid,
    matrix,
    rbind,
    rep,
    rowMeans,
    rowSums,
    sweep,
)

# Coercion
from .coerce import (
    as_Date,
    as_character,
    as_date,
    as_integer,
    as_logical,
    as_numeric,
)

# Predicates
from .predicates import (
    is_finite,
    is_na,
    is_null,
    is_numeric,
)

# Distance layer (base-R stats: dist / as.dist / as.matrix.dist + accessors).
# The clustering algorithms that consume a ``Dist`` (hclust/cutree/…) land in
# ``hea.R.clustering`` and import one-way from here.
from .distance import (
    Dist,
    as_dist,
    as_matrix_dist,
    cmdscale,
    dist,
    format_dist,
    labels_dist,
    mahalanobis,
    print_dist,
)

# Clustering algorithms (base-R stats: hclust + tree objects). Imports the
# distance layer one-way (acyclic).
from .clustering import (
    Dendrogram,
    Hclust,
    Kmeans,
    as_dendrogram,
    as_hclust,
    cophenetic,
    cophenetic_dendrogram,
    cut_dendrogram,
    cutree,
    dendrapply,
    fitted_kmeans,
    hclust,
    is_leaf,
    kmeans,
    labels_dendrogram,
    merge_dendrogram,
    midcache_dendrogram,
    nleaves,
    nobs_dendrogram,
    order_dendrogram,
    print_dendrogram,
    print_hclust,
    print_kmeans,
    reorder,
    reorder_dendrogram,
    rev_dendrogram,
    str_dendrogram,
)

# Distributions
from .distributions import (
    dbeta,
    dbinom,
    dcauchy,
    dchisq,
    dexp,
    dgamma,
    dgeom,
    dhyper,
    dlnorm,
    dlogis,
    dmultinom,
    dnbinom,
    dnorm,
    dpois,
    dsignrank,
    dt,
    dunif,
    dweibull,
    dwilcox,
    pbeta,
    pbinom,
    pbirthday,
    pcauchy,
    pchisq,
    pexp,
    pf,
    pgamma,
    pgeom,
    phyper,
    plnorm,
    plogis,
    pnbinom,
    pnorm,
    ppois,
    psignrank,
    pt,
    ptukey,
    punif,
    pweibull,
    pwilcox,
    qbeta,
    qbinom,
    qbirthday,
    qcauchy,
    qchisq,
    qexp,
    qf,
    qgamma,
    qgeom,
    qhyper,
    qlnorm,
    qlogis,
    qnbinom,
    qnorm,
    qpois,
    qsignrank,
    qt,
    qtukey,
    qunif,
    qweibull,
    qwilcox,
    r2dtable,
    rWishart,
    rbeta,
    rbinom,
    rcauchy,
    rchisq,
    rexp,
    rf,
    rgamma,
    rgeom,
    rhyper,
    rlnorm,
    rlogis,
    rmultinom,
    rnbinom,
    rnorm,
    rpois,
    rsignrank,
    rt,
    runif,
    rweibull,
    rwilcox,
    sample,
    set_seed,
)

# Functional iteration
from .functional import sapply, tapply

# emmeans — small CRAN port; promoted out of R/ once the surface grows.
from .emmeans import EmmGrid, emmeans, summary_emmgrid_contrasts

# Contingency tables
from .tables import addmargins, prop_table, table, xtabs

# Hypothesis tests + containers + rank helpers
from .htest import (
    AnovaTable,
    HTest,
    aov,
    bartlett_test,
    binom_test,
    chisq_test,
    cor_test,
    fisher_test,
    friedman_test,
    kruskal_test,
    ks_test,
    mcnemar_test,
    prop_test,
    psmirnov,
    qsmirnov,
    rank,
    rsmirnov,
    shapiro_test,
    signed_rank,
    t_test,
    var_test,
    wilcox_test,
)

# Model generics
from .model_generics import (
    AIC,
    BIC,
    Terms,
    bootMer,
    coef,
    coefficients,
    confint,
    deviance,
    df_residual,
    effects,
    fitted,
    fitted_values,
    fixef,
    formula,
    profile,
    simulate,
    logLik,
    model_frame,
    model_matrix,
    nobs,
    predict,
    ranef,
    refit,
    refitML,
    case_names,
    labels,
    resid,
    residuals,
    terms,
    update,
    variable_names,
    vcov,
    weights,
    # lme4 merMod accessor surface (Tier 2 lmer-parity additions)
    VarCorr,
    getME,
    getData,
    isREML,
    isLMM,
    isGLMM,
    isNLMM,
    isSingular,
    extractAIC,
    rePCA,
)

# Regression diagnostics
from .diagnostics import (
    cooks_distance,
    dfbeta,
    dfbetas,
    dffits,
    hatvalues,
    influence,
    rstandard,
    rstudent,
)

# R's RNG (Mersenne-Twister / Inversion / Rejection), bit-exact —
# set.seed/runif/sample parity for pinning RNG-dependent R results.
from .rng import RMersenneTwister

# Model comparison / selection
from .model_selection import add1, anova, drop1, step

# Private model-selection helpers exposed for white-box tests
# (``tests/test_compare.py`` imports these directly from ``hea.R``).
from .model_selection import (
    _anova_gam_rdf,
    _anova_gam_table,
    _anova_glm_table,
    _drop1_lm,
    _extract_aic_lm,
)

# Time series construction — mirrors R's ``ts()``. The constructor
# lives in ``hea.tidy.dataframe`` (next to ``DataFrame``, whose
# ``_ts_meta`` attribute it stamps); a lazy wrapper here makes
# ``hea.R.ts`` reachable without the circular import that a top-level
# ``from ..tidy.dataframe import ts`` would trigger (``hea.tidy.dataframe``
# is mid-load when ``hea.R.__init__`` is imported).
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
    "head", "tail", "nrow", "ncol", "dim", "length",
    "names", "colnames", "summary",
    "complete_cases", "na_omit",
    # vector helpers
    "seq", "seq_len", "seq_along",
    "rev", "sort", "order",
    "which", "which_max", "which_min",
    "cumsum", "cumprod", "cummax", "cummin", "diff",
    "unique", "duplicated", "tabulate",
    "cut", "findInterval",
    # contingency tables
    "table", "xtabs", "prop_table", "addmargins",
    # reductions (R defaults: sd/var use N-1)
    "mean", "median", "var", "sd", "quantile", "IQR", "cor", "cov",
    # base-R constants
    "LETTERS", "letters",
    # elementwise math (R: vectorized scalar functions)
    # Note: ``abs`` and ``round`` exist as module attributes but are NOT
    # exported — they collide with Python builtins and the translator
    # treats the R names as builtins (the builtin handles scalars / Series
    # / ndarrays via __abs__ / __round__).
    "pi",
    "sqrt", "exp", "log", "log2", "log10", "log1p", "expm1", "sign",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "floor", "ceiling", "trunc",
    # vector primitives — R's rep() flattens nested list-of-vector inputs
    # (the translator emits ``c(scalar, vec)`` as a Python list literal).
    "rep",
    # matrix / frame utilities (R: base matrix ops)
    "rowSums", "colSums", "rowMeans", "colMeans",
    "apply", "rbind", "cbind", "sweep", "expand_grid", "matrix",
    "R_range", "R_round",
    # coercion / predicates
    "as_numeric", "as_integer", "as_character", "as_logical",
    "as_date", "as_Date",
    "is_na", "is_null", "is_finite", "is_numeric", "is_factor",
    "factor", "fct", "ordered", "interaction", "levels", "nlevels",
    # distance layer (base-R stats: dist / as.dist / as.matrix.dist + accessors)
    "Dist", "dist", "as_dist", "as_matrix_dist",
    "format_dist", "labels_dist", "print_dist", "mahalanobis", "cmdscale",
    # clustering (base-R stats: hclust + tree objects)
    "Hclust", "hclust", "as_hclust", "cutree", "cophenetic", "print_hclust",
    # dendrogram subsystem (base-R stats: dendrogram.R, non-graphics)
    "Dendrogram", "as_dendrogram", "cophenetic_dendrogram",
    "cut_dendrogram", "dendrapply", "is_leaf", "labels_dendrogram",
    "merge_dendrogram", "midcache_dendrogram", "nleaves", "nobs_dendrogram",
    "order_dendrogram", "print_dendrogram", "reorder", "reorder_dendrogram",
    "rev_dendrogram", "str_dendrogram",
    # k-means (base-R stats: Hartigan-Wong / Lloyd / Forgy / MacQueen)
    "Kmeans", "kmeans", "fitted_kmeans", "print_kmeans",
    # distributions: d/p/q/r families
    "dnorm", "pnorm", "qnorm", "rnorm",
    "dt", "pt", "qt", "rt",
    "pf", "qf", "rf",
    "dchisq", "pchisq", "qchisq", "rchisq",
    "dbinom", "pbinom", "qbinom", "rbinom",
    "dpois", "ppois", "qpois", "rpois",
    "dunif", "punif", "qunif", "runif",
    "ptukey", "qtukey",
    "dsignrank", "psignrank", "qsignrank", "rsignrank",
    "dwilcox", "pwilcox", "qwilcox", "rwilcox",
    "dhyper", "phyper", "qhyper", "rhyper",
    "dnbinom", "pnbinom", "qnbinom", "rnbinom",
    "dexp", "pexp", "qexp", "rexp",
    "dgamma", "pgamma", "qgamma", "rgamma",
    "dbeta", "pbeta", "qbeta", "rbeta",
    "dcauchy", "pcauchy", "qcauchy", "rcauchy",
    "dlogis", "plogis", "qlogis", "rlogis",
    "dlnorm", "plnorm", "qlnorm", "rlnorm",
    "dweibull", "pweibull", "qweibull", "rweibull",
    "dgeom", "pgeom", "qgeom", "rgeom",
    # combinatorial / multivariate + Smirnov distribution surface
    "dmultinom", "rmultinom", "pbirthday", "qbirthday", "r2dtable", "rWishart",
    "psmirnov", "qsmirnov", "rsmirnov",
    "set_seed", "sample", "sapply", "tapply",
    # rank helpers (Lindeløv-style "tests as lm" notebook)
    "rank", "signed_rank",
    # hypothesis tests (return HTest, R's ``htest`` print-shape)
    "HTest", "AnovaTable",
    "t_test", "wilcox_test", "cor_test", "kruskal_test", "chisq_test",
    "fisher_test", "prop_test", "binom_test", "var_test", "bartlett_test",
    "shapiro_test", "ks_test", "mcnemar_test", "friedman_test",
    "aov",
    # model generics (lm / glm / gam / bam / gmm)
    "coef", "coefficients", "fixef", "ranef", "refit", "refitML",
    "resid", "residuals", "fitted", "fitted_values",
    "predict", "confint", "vcov", "profile", "bootMer",
    "logLik", "deviance", "nobs", "weights", "df_residual",
    "effects", "simulate",
    "formula", "model_matrix", "model_frame",
    "AIC", "BIC", "anova", "add1", "drop1", "step",
    "update", "terms", "Terms",
    # lme4 merMod accessors (VarCorr / getME / predicates / getData / rePCA)
    "VarCorr", "getME", "getData", "extractAIC", "rePCA",
    "isREML", "isLMM", "isGLMM", "isNLMM", "isSingular",
    # regression diagnostics
    "hatvalues", "rstandard", "rstudent",
    "cooks_distance", "dffits", "dfbeta", "dfbetas", "influence",
    "variable_names", "case_names", "labels",
    # time series construction (R's ts())
    "ts",
    # emmeans (CRAN port, parked here until the surface grows)
    "EmmGrid", "emmeans", "summary_emmgrid_contrasts",
]
