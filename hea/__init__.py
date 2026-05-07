"""hea — R/tidyverse-shaped statistical modeling on polars.

The ``hea`` namespace is a strict superset of the ``polars`` namespace:
everything in ``polars.__all__`` is reachable through ``hea.*`` (with
``hea.DataFrame``, ``hea.LazyFrame``, ``hea.Series`` pointing at the hea
subclasses, and constructors / I/O readers wrapped to return them). So
``import hea`` is the canonical import — users don't need
``import polars as pl`` alongside.
"""

# 1. Star-import polars first so all of pl.__all__ becomes hea.*. The
#    explicit hea-specific imports below override DataFrame, LazyFrame,
#    Series with the hea subclasses; the rest pass through unchanged.
from polars import *  # noqa: F401, F403

# 2. Sub-namespaces. polars exposes these as importable but they're not
#    in pl.__all__ as star-import targets; we re-export by alias.
from polars import api, exceptions, plugins, selectors  # noqa: F401

# 3. hea-specific imports (statistical modeling + tidyverse verbs).
from .compare import AIC, BIC, add1, anova, drop1, step
from .family import (
    Binomial,
    Family,
    Gamma,
    Gaussian,
    InverseGaussian,
    Poisson,
    Quasi,
    Tweedie,
    binomial,
    gaussian,
    inverse_gaussian,
    poisson,
    quasi,
    tw,
)
from .bam import bam
from .gam import gam
from .glm import glm
from .lm import lm
from .lme import lme
from .R import (
    aov,
    bartlett_test,
    binom_test,
    chisq_test,
    cor_test,
    fisher_test,
    friedman_test,
    if_else,
    kruskal_test,
    ks_test,
    mcnemar_test,
    parse_number,
    prop_test,
    rank,
    shapiro_test,
    signed_rank,
    t_test,
    var_test,
    wilcox_test,
)
from .data import data, factor
# These re-bindings shadow the star-imported pl.DataFrame / pl.LazyFrame
# / pl.Series. Listed in _HEA_OVERRIDES below.
from .dataframe import DataFrame, GroupBy, LazyFrame, Series, desc, tbl
from . import ggplot
from . import plot
from . import R


# 4. Wrap polars factories (constructors + I/O) so they return hea
#    subclasses. ``_rewrap`` handles polymorphic returns
#    (DataFrame / LazyFrame / Series / list-thereof / pass-through).
import functools as _functools
import polars as _pl


def _rewrap(obj):
    if isinstance(obj, _pl.DataFrame) and not isinstance(obj, DataFrame):
        return DataFrame._from_pydf(obj._df)
    if isinstance(obj, _pl.LazyFrame) and not isinstance(obj, LazyFrame):
        return LazyFrame._from_pyldf(obj._ldf)
    if isinstance(obj, _pl.Series) and not isinstance(obj, Series):
        return Series._from_pyseries(obj._s)
    if isinstance(obj, list):
        return [_rewrap(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_rewrap(x) for x in obj)
    return obj


def _wrap_factory(name: str):
    pl_func = getattr(_pl, name)

    @_functools.wraps(pl_func)
    def wrapper(*args, **kwargs):
        return _rewrap(pl_func(*args, **kwargs))

    return wrapper


# DataFrame-returning constructors and eager I/O readers. ``read_ipc_schema``,
# ``read_parquet_metadata``, ``read_parquet_schema`` return dict[str, DataType]
# and don't need wrapping.
_DF_FACTORIES = (
    "concat",
    "from_arrow",
    "from_dataframe",
    "from_dict",
    "from_dicts",
    "from_numpy",
    "from_pandas",
    "from_records",
    "from_repr",
    "from_torch",
    "json_normalize",
    "read_avro",
    "read_clipboard",
    "read_csv",
    "read_csv_batched",
    "read_database",
    "read_database_uri",
    "read_delta",
    "read_excel",
    "read_ipc",
    "read_ipc_stream",
    "read_json",
    "read_lines",
    "read_ndjson",
    "read_ods",
    "read_parquet",
)

# LazyFrame-returning scanners.
_LF_FACTORIES = (
    "scan_csv",
    "scan_delta",
    "scan_iceberg",
    "scan_ipc",
    "scan_lines",
    "scan_ndjson",
    "scan_parquet",
    "scan_pyarrow_dataset",
)

# Polymorphic — could return DataFrame, LazyFrame, list, etc. ``_rewrap``
# handles each case.
_POLY_FACTORIES = (
    "align_frames",
    "collect_all",
    "from_epoch",
    "merge_sorted",
    "union",
)

for _name in (*_DF_FACTORIES, *_LF_FACTORIES, *_POLY_FACTORIES):
    if hasattr(_pl, _name):
        globals()[_name] = _wrap_factory(_name)


# 5. _HEA_OVERRIDES — every name where hea shadows a name from
#    ``polars.__all__`` must be listed here with a one-line reason. The
#    invariant below fails if (a) hea shadows a polars name not listed,
#    or (b) a listed name isn't actually shadowed (stale entry, e.g.
#    polars renamed the function on a version bump).
_HEA_OVERRIDES = {
    "DataFrame": "subclass with tidyverse verbs",
    "LazyFrame": "subclass that re-wraps on collect",
    "Series": "subclass with hea-aware overrides",
}
for _n in (*_DF_FACTORIES, *_LF_FACTORIES, *_POLY_FACTORIES):
    if hasattr(_pl, _n):
        _HEA_OVERRIDES[_n] = "wraps result as hea subclass"

_pl_all = set(_pl.__all__)
_actual_shadows = {
    n for n in _pl_all
    if n in globals() and globals()[n] is not getattr(_pl, n)
}
_unexpected = _actual_shadows - set(_HEA_OVERRIDES)
_stale = set(_HEA_OVERRIDES) - _actual_shadows
if _unexpected:
    raise RuntimeError(
        f"hea shadows polars names not in _HEA_OVERRIDES: {sorted(_unexpected)}. "
        "Add to _HEA_OVERRIDES with a one-line reason, or stop shadowing them."
    )
if _stale:
    raise RuntimeError(
        f"_HEA_OVERRIDES lists names hea no longer shadows: {sorted(_stale)}. "
        "Polars may have renamed/removed them on a version bump."
    )


# Cleanup loop variables (the underscored module helpers _pl, _rewrap,
# _wrap_factory, _HEA_OVERRIDES are kept — _rewrap closures reference _pl
# at call time, and the others are useful as module internals).
del _name, _n, _pl_all, _actual_shadows, _unexpected, _stale
