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
* :mod:`hea.R.lubridate` — ``today`` / ``now``, ``ymd`` / ``mdy`` /
  ``dmy`` + their ``_hms`` / ``_hm`` variants.
* :mod:`hea.R.stringr` — ``str_*`` family.
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

# lubridate
from .lubridate import (
    dmy,
    dmy_hm,
    dmy_hms,
    mdy,
    mdy_hm,
    mdy_hms,
    now,
    today,
    ymd,
    ymd_hm,
    ymd_hms,
)

# stringr
from .stringr import (
    str_c,
    str_count,
    str_detect,
    str_equal,
    str_flatten,
    str_glue,
    str_length,
    str_sort,
    str_sub,
    str_to_lower,
    str_to_title,
    str_to_upper,
    str_view,
    str_view_all,
)

# Distributions
from .distributions import (
    dbeta,
    dbinom,
    dchisq,
    dexp,
    dgamma,
    dnorm,
    dpois,
    dt,
    dunif,
    pbeta,
    pbinom,
    pchisq,
    pexp,
    pf,
    pgamma,
    pnorm,
    ppois,
    pt,
    punif,
    qbeta,
    qbinom,
    qchisq,
    qexp,
    qf,
    qgamma,
    qnorm,
    qpois,
    qt,
    qunif,
    rbeta,
    rbinom,
    rchisq,
    rexp,
    rf,
    rgamma,
    rnorm,
    rpois,
    rt,
    runif,
    sample,
    set_seed,
)

# Functional iteration
from .functional import sapply, tapply

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
    rank,
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
    coef,
    coefficients,
    confint,
    deviance,
    df_residual,
    fitted,
    fitted_values,
    fixef,
    formula,
    logLik,
    model_frame,
    model_matrix,
    nobs,
    predict,
    ranef,
    resid,
    residuals,
    terms,
    update,
    vcov,
)

# Regression diagnostics
from .diagnostics import (
    cooks_distance,
    dfbetas,
    dffits,
    hatvalues,
    influence,
    rstandard,
    rstudent,
)

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
    # lubridate: clock primitives + parsers
    "today", "now",
    "ymd", "mdy", "dmy",
    "ymd_hms", "ymd_hm", "mdy_hms", "mdy_hm", "dmy_hms", "dmy_hm",
    # stringr: regex-debug pretty-printers (no-op-like)
    "str_view", "str_view_all",
    # stringr: core string ops
    "str_c", "str_glue", "str_flatten", "str_length", "str_sub",
    "str_to_upper", "str_to_lower", "str_to_title",
    "str_sort", "str_equal", "str_detect", "str_count",
    "is_na", "is_null", "is_finite", "is_numeric", "is_factor",
    "factor", "fct", "ordered", "interaction", "levels", "nlevels",
    # distributions: d/p/q/r families
    "dnorm", "pnorm", "qnorm", "rnorm",
    "dt", "pt", "qt", "rt",
    "pf", "qf", "rf",
    "dchisq", "pchisq", "qchisq", "rchisq",
    "dbinom", "pbinom", "qbinom", "rbinom",
    "dpois", "ppois", "qpois", "rpois",
    "dunif", "punif", "qunif", "runif",
    "dexp", "pexp", "qexp", "rexp",
    "dgamma", "pgamma", "qgamma", "rgamma",
    "dbeta", "pbeta", "qbeta", "rbeta",
    "set_seed", "sample", "sapply", "tapply",
    # rank helpers (Lindeløv-style "tests as lm" notebook)
    "rank", "signed_rank",
    # hypothesis tests (return HTest, R's ``htest`` print-shape)
    "HTest", "AnovaTable",
    "t_test", "wilcox_test", "cor_test", "kruskal_test", "chisq_test",
    "fisher_test", "prop_test", "binom_test", "var_test", "bartlett_test",
    "shapiro_test", "ks_test", "mcnemar_test", "friedman_test",
    "aov",
    # model generics (lm / glm / gam / bam / lme)
    "coef", "coefficients", "fixef", "ranef",
    "resid", "residuals", "fitted", "fitted_values",
    "predict", "confint", "vcov",
    "logLik", "deviance", "nobs", "df_residual",
    "formula", "model_matrix", "model_frame",
    "AIC", "BIC", "anova", "add1", "drop1", "step",
    "update", "terms", "Terms",
    # regression diagnostics
    "hatvalues", "rstandard", "rstudent",
    "cooks_distance", "dffits", "dfbetas", "influence",
]
