"""Tidyverse-flavored verbs as a thin subclass of ``pl.DataFrame``.

Ports two *R for Data Science* chapters onto polars:

* Chapter 3 — data transformation: ``filter``, ``arrange``,
  ``distinct``, ``mutate``, ``select``, ``rename``, ``relocate``,
  ``group_by``, ``summarize``, ``slice_*``, ``count``, ``ungroup``.
* Chapter 5 — tidy data: ``pivot_longer`` (with ``names_sep`` /
  ``names_pattern`` / the ``.value`` sentinel), ``pivot_wider``,
  ``pull``.

Design choices:

* Subclass, not wrap. ``DataFrame(pl.DataFrame)`` preserves IS-A so any
  function that already accepts a polars DataFrame (including hea's own
  ``lm`` / ``lme`` / ``glm`` / ``gam``) keeps working. Native polars
  methods on the subclass return plain ``pl.DataFrame``; tidyverse
  methods always return our subclass via ``_wrap``.
* Polars expressions pass through unchanged. ``filter(pl.col("x") > 1)``
  is the recommended call form. We don't try to support bare-name NSE.
* ``mutate`` / ``summarize`` accept ``**kwargs`` so the right-hand side
  is auto-aliased to the kwarg name — the boilerplate fix that motivates
  this module. Positional polars expressions are also accepted.
* ``group_by`` returns a ``GroupBy`` wrapper exposing the verbs that
  make sense on a grouped frame. Persistent grouping across arbitrary
  verbs (filter, arrange, …) is intentionally NOT modeled — too much
  state for too little gain — but ``GroupBy.mutate`` does the windowed
  ``.over(group_cols)`` translation since chapter 3 explicitly contrasts
  it with ``summarize`` (exercise 6f).

Implementation
--------------
This package is the result of splitting the legacy ``hea/tidy.py`` (one
4.5k-line file) into divisions:

* :mod:`hea.tidy.dataframe` — the :class:`DataFrame` subclass (~1450
  lines; all the tidyverse verb methods).
* :mod:`hea.tidy.series` — :class:`Series` / :class:`LazyFrame`
  subclasses, plus the install hooks that patch ``pl.Expr`` /
  ``pl.Series`` / the lazy GroupBy class so subclass identity survives
  through chains.
* :mod:`hea.tidy.groupby` — the :class:`GroupBy` wrapper.
* :mod:`hea.tidy.summary` — :class:`Summary` print object + per-dtype
  block builders.
* :mod:`hea.tidy.basics` — small standalone utilities (``tbl``, ``desc``,
  ``exclude``, ``n`` / ``n_distinct``, ``if_else`` / ``case_when``,
  ``glimpse``).
* :mod:`hea.tidy.strings` — ``str_*`` family + readr's ``parse_number`` /
  ``parse_double`` + stringr's ``str_wrap``.
* :mod:`hea.tidy.dates` — lubridate parsers (``ymd`` / ``mdy`` / ``dmy``
  + the ``_hms`` / ``_hm`` variants, plus ``today`` / ``now``).
* :mod:`hea.tidy.factors` — forcats's ``fct_*`` family.
* :mod:`hea.tidy.binning` — ggplot2-style ``cut_width`` / ``cut_interval``
  / ``cut_number``.
* :mod:`hea.tidy.window` — dplyr's rank / window / cumulative /
  positional helpers (``row_number``, ``min_rank``, ``lag``, ``lead``,
  ``cummean``, ``first``, ``last``, ``nth``, ``consecutive_id``, …).
* :mod:`hea.tidy.joins` — ``closest`` / ``overlaps`` / ``within`` /
  ``join_by`` for non-equi joins.
* :mod:`hea.tidy._shared` — private cross-file helpers
  (``_resolve_lazy_factors``, ``_kwargs_to_exprs``, name cleaning,
  :class:`_TidyRange` / ``cols_between``).
"""
from __future__ import annotations

# Order matters: _shared and basics are foundational; dataframe pulls them
# in; series uses dataframe; groupby uses both. Verb files don't depend on
# the class hierarchy so they slot in anywhere.
from .basics import (
    _Desc,
    case_when,
    desc,
    exclude,
    glimpse,
    if_else,
    n,
    n_distinct,
    tbl,
)
from ._shared import (
    _apply_groups,
    _check_groups,
    _clean_one_name,
    _disambiguate_clean_names,
    _kwargs_to_exprs,
    _resolve_anchor,
    _resolve_lazy_factors,
    _split_arrange,
    _TidyRange,
    cols_between,
)

# Class hierarchy (these install module-level side effects: pl.Series and
# pl.Expr get patched with hea-aware wrappers).
from .dataframe import DataFrame
from .series import (
    LazyFrame,
    Series,
    _HeaLazyGroupBy,
    _install_df_series_overrides,
    _install_expr_is_na_alias,
    _install_expr_r_aliases,
    _install_is_in_mixed_list_support,
    _install_lazy_groupby_overrides,
    _install_series_subclass_overrides,
)
from .groupby import GroupBy
from .summary import Summary

# Free-function tidyverse verbs (split by functional area).
from .binning import cut_interval, cut_number, cut_width
from .dates import (
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
from .factors import (
    fct_collapse,
    fct_infreq,
    fct_lump_lowfreq,
    fct_lump_n,
    fct_recode,
    fct_relevel,
    fct_reorder,
    fct_reorder2,
    fct_rev,
)
from .joins import closest, join_by, overlaps, within
from .strings import (
    parse_double,
    parse_number,
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
    str_wrap,
)
from .window import (
    between,
    consecutive_id,
    cumall,
    cumany,
    cume_dist,
    cummean,
    dense_rank,
    first,
    lag,
    last,
    lead,
    min_rank,
    na_if,
    near,
    nth,
    ntile,
    percent_rank,
    row_number,
)


# ---- polars passthrough ----------------------------------------------
#
# Names a tidy pipeline reaches for. Expression builders (``col``, ``lit``,
# ``when``), basic combinators, row-wise reducers, plus a couple of
# typing/schema classes. Anything more esoteric stays in ``polars`` and
# is NOT re-exported. Dtypes live in :mod:`hea.dtypes`; I/O factories
# in :mod:`hea.io`.

from polars import (  # noqa: F401, E402
    Expr,
    Schema,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    concat_list,
    concat_str,
    lit,
    max_horizontal,
    mean_horizontal,
    min_horizontal,
    sum_horizontal,
    when,
)


# ---- polars DataFrame constructors + multi-frame combinators ---------
#
# ``hea.tidy`` is the frame namespace, period — so ``concat``,
# ``from_dict``, ``from_pandas``, etc. live here, not in ``hea.io``.
# Each is wrapped so the result is the hea subclass.

import functools as _functools  # noqa: E402
import polars as _pl_tidy  # noqa: E402


def _rewrap(obj):
    if isinstance(obj, _pl_tidy.DataFrame) and not isinstance(obj, DataFrame):
        return DataFrame._from_pydf(obj._df)
    if isinstance(obj, _pl_tidy.LazyFrame) and not isinstance(obj, LazyFrame):
        return LazyFrame._from_pyldf(obj._ldf)
    if isinstance(obj, _pl_tidy.Series) and not isinstance(obj, Series):
        return Series._from_pyseries(obj._s)
    if isinstance(obj, list):
        return [_rewrap(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_rewrap(x) for x in obj)
    return obj


def _wrap_factory(name: str):
    pl_func = getattr(_pl_tidy, name)

    @_functools.wraps(pl_func)
    def wrapper(*args, **kwargs):
        return _rewrap(pl_func(*args, **kwargs))

    return wrapper


# DataFrame-returning constructors (``from_*``, ``json_normalize``) plus
# multi-frame combinators (``concat``, ``align_frames``, ``merge_sorted``,
# ``union``, ``collect_all``, ``from_epoch``). All wrap-then-return.
_TIDY_FACTORIES = (
    "align_frames",
    "collect_all",
    "concat",
    "from_arrow",
    "from_dataframe",
    "from_dict",
    "from_dicts",
    "from_epoch",
    "from_numpy",
    "from_pandas",
    "from_records",
    "from_repr",
    "from_torch",
    "json_normalize",
    "merge_sorted",
    "union",
)

for _name in _TIDY_FACTORIES:
    if hasattr(_pl_tidy, _name):
        globals()[_name] = _wrap_factory(_name)

del _name


__all__ = [
    # core classes
    "DataFrame", "GroupBy", "LazyFrame", "Series", "Summary",
    # dplyr verbs / mutate helpers
    "case_when", "desc", "if_else", "n", "n_distinct", "tbl",
    # dplyr rank family
    "row_number", "min_rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
    # dplyr window / numeric helpers
    "lag", "lead", "between", "na_if", "near",
    # dplyr positional pickers (shadow polars pl.first / pl.last / pl.nth)
    "first", "last", "nth",
    # dplyr cumulative + run-length
    "cummean", "cumall", "cumany", "consecutive_id",
    # dplyr two-table verb helpers (chapter 19)
    "join_by", "closest", "overlaps", "within",
    # readr / stringr / tibble
    "parse_double", "parse_number", "str_wrap", "glimpse",
]
