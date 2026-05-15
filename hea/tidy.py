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
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import shutil
import textwrap
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
import polars as pl
from polars.lazyframe.group_by import LazyGroupBy as _PlLazyGroupBy
from scipy import stats as _sps

from .R import cut as _R_cut

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


def tbl(obj):
    """Re-wrap a plain polars container as the corresponding hea subclass.

    Rarely needed in normal code: Phase 1-5 of the subclass-coverage work
    means every operation hea exposes already returns the right subclass.
    ``tbl`` is the documented escape hatch for the remaining cases — e.g.
    when an external library hands you a ``pl.DataFrame`` and you want
    to chain hea methods on it without copying data:

    >>> import polars as pl, hea
    >>> raw = pl.DataFrame({"x": [1, 2, 3]})  # plain polars
    >>> hea.tbl(raw).filter(pl.col("x") > 1)  # hea subclass
    """
    if isinstance(obj, DataFrame):
        return obj
    if isinstance(obj, pl.DataFrame):
        return DataFrame._from_pydf(obj._df)
    if isinstance(obj, LazyFrame):
        return obj
    if isinstance(obj, pl.LazyFrame):
        return LazyFrame._from_pyldf(obj._ldf)
    if isinstance(obj, Series):
        return obj
    if isinstance(obj, pl.Series):
        return Series._from_pyseries(obj._s)
    raise TypeError(
        f"tbl(): expected pl.DataFrame / pl.LazyFrame / pl.Series, got {type(obj).__name__}"
    )


class _Desc:
    """Marker for descending sort, produced by ``desc("colname")``."""

    __slots__ = ("col",)

    def __init__(self, col: str):
        self.col = col


def desc(col: Any) -> Any:
    """Reverse the sort order of a column or vector — mirrors dplyr's ``desc()``.

    Two call shapes:

    * ``desc("name")`` returns a ``_Desc`` marker that ``arrange()``
      recognizes: ``df.arrange("a", desc("b"))`` sorts by ``a``
      ascending then ``b`` descending.
    * ``desc(values)`` negates the values, matching R's
      ``-xtfrm(x)`` definition. Useful with verbs that take a vector
      directly, e.g. ``min_rank(desc(x))`` gives the descending
      min-rank.

    Type-in / type-out for the value form:
        * ``pl.Expr`` → negated ``pl.Expr``
        * ``pl.Series`` → negated ``pl.Series``
        * list / tuple / ndarray → negated ``np.ndarray`` (float)
    """
    if isinstance(col, str):
        return _Desc(col)
    if isinstance(col, (pl.Expr, pl.Series)):
        return -col
    import numpy as np
    return -np.asarray(col, dtype=float)


def exclude(*columns: Any) -> pl.Expr:
    """Like :func:`polars.exclude`, but also accepts a :class:`DataFrame`
    (uses ``.columns``), :class:`Series` (uses ``.name``), or list/tuple
    thereof. Lets ``df.select(hea.exclude(df["year":"day"]))`` mirror the
    positive form ``df.select(df["year":"day"])``.
    """
    flat: list[Any] = []
    for c in columns:
        if isinstance(c, (list, tuple)):
            flat.extend(c)
        elif isinstance(c, pl.DataFrame):
            flat.extend(c.columns)
        elif isinstance(c, pl.Series):
            flat.append(c.name)
        else:
            flat.append(c)
    return pl.exclude(flat)


# dplyr's ``n()`` — row-count expression for ``mutate`` / ``summarize``.
# Aliased to ``pl.len`` so ``from hea import n`` doesn't shadow the builtin
# ``len`` (which ``from hea import len`` would, since ``hea.len`` is
# ``polars.len`` via the star-import in __init__.py).
n = pl.len

# dplyr's ``n_distinct()`` — polars exposes the same operation as
# ``n_unique``. Both names route to the same Expr; ``n_unique`` is also
# reachable as ``hea.n_unique`` via the polars star-import.
n_distinct = pl.n_unique


# ---- conditionals (dplyr) -------------------------------------------

def if_else(condition, true_value, false_value, missing=None) -> pl.Expr:
    """dplyr's ``if_else()`` — vectorized conditional.

    Wraps ``pl.when(condition).then(true_value).otherwise(false_value)``
    with one dplyr-shaped twist: a null in ``condition`` produces
    ``missing`` (default ``None`` → null), matching dplyr's ``NA in →
    NA out``. Polars' raw ``when/then/otherwise`` instead routes nulls
    through the otherwise branch — use ``pl.when(...)`` directly if
    that's what you want.

    Returns a ``pl.Expr``. Use inside ``mutate`` / ``select`` / any
    polars verb. For Series-on-Series eager evaluation, materialize
    via ``df.select(if_else(...))`` or use ``Series.zip_with``.

    Parameters
    ----------
    condition : pl.Expr | pl.Series | bool
        Boolean predicate.
    true_value, false_value : pl.Expr | pl.Series | scalar
        Values for True / False entries. Bare scalars are auto-lifted
        via ``pl.lit`` by polars' ``when/then`` machinery.
    missing : scalar, optional
        Value emitted when ``condition`` is null. Defaults to ``None``
        (null), matching dplyr's ``NA`` default.
    """
    # Polars' .then("x") interprets a bare string as a column name; dplyr's
    # if_else treats strings as literals. Lift any non-Expr non-Series value
    # to pl.lit so "5" stays "5".
    def _lit(v):
        return v if isinstance(v, (pl.Expr, pl.Series)) else pl.lit(v)

    t, f = _lit(true_value), _lit(false_value)
    if isinstance(condition, (pl.Expr, pl.Series)):
        return (
            pl.when(condition.is_null()).then(pl.lit(missing))
            .when(condition).then(t)
            .otherwise(f)
        )
    return pl.when(condition).then(t).otherwise(f)


def case_when(*pairs, default=None) -> pl.Expr:
    """dplyr's ``case_when()`` — multi-branch vectorized conditional.

    Each pair is ``(condition, value)``. The result for each row is the
    ``value`` of the first pair whose ``condition`` is True; rows matching
    no condition take ``default``. Mirrors dplyr's ``case_when()`` —
    Python has no ``cond ~ value`` formula syntax, so pass tuples instead.

    Bare-string ``value``s are lifted to ``pl.lit`` (matching dplyr's
    "strings are values" convention). Polars' raw ``pl.when(...).then("x")``
    would interpret ``"x"`` as a column reference.

    Null conditions fall through to the next branch (and ultimately to
    ``default``), matching dplyr 1.1+. Use :func:`if_else` if you want
    null-in → null-out instead.

    Parameters
    ----------
    *pairs : tuple[condition, value]
        Each ``condition`` is a boolean ``pl.Expr`` / ``pl.Series``;
        each ``value`` is a scalar / ``pl.Expr`` / ``pl.Series``.
    default : scalar | pl.Expr, optional
        Value for rows matching no condition. Defaults to ``None`` (null),
        matching dplyr's ``.default = NA``.

    Examples
    --------
    >>> import hea
    >>> from hea import case_when, col
    >>> df = hea.DataFrame({"drv": ["f", "r", "4", "f"]})
    >>> df.mutate(label=case_when(
    ...     (col("drv") == "f", "front-wheel drive"),
    ...     (col("drv") == "r", "rear-wheel drive"),
    ...     (col("drv") == "4", "4-wheel drive"),
    ... ))  # doctest: +SKIP
    """
    if not pairs:
        raise TypeError(
            "case_when() requires at least one (condition, value) pair"
        )

    def _lit(v):
        return v if isinstance(v, (pl.Expr, pl.Series)) else pl.lit(v)

    expr = None
    for i, pair in enumerate(pairs):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise TypeError(
                f"case_when() pair {i} must be a (condition, value) "
                f"tuple, got {pair!r}"
            )
        cond, val = pair
        val = _lit(val)
        expr = pl.when(cond).then(val) if expr is None else expr.when(cond).then(val)
    return expr.otherwise(_lit(default))


# ---- readr parsers --------------------------------------------------

def parse_number(x):
    """readr's ``parse_number()`` — pull the first number out of a string column.

    Strips comma thousand-separators, then extracts the first signed
    integer or decimal via ``(-?\\d+(?:\\.\\d+)?)`` and casts to
    ``Float64`` with ``strict=False`` (unparseable → null). Handles
    currency symbols, trailing units, and mixed text the same way
    readr does (``"$1,234.56"`` → ``1234.56``, ``"30 yo"`` → ``30``,
    ``"five"`` → null). Locale-specific thousand/decimal separators
    aren't supported — US-style only.

    Type-in / type-out: ``pl.Series`` → ``pl.Series``; ``pl.Expr`` →
    ``pl.Expr``; list / tuple / ndarray → ``list`` (with ``None`` for
    unparseable entries).
    """
    array_like = not isinstance(x, (pl.Series, pl.Expr))
    if array_like:
        x = pl.Series(x, dtype=pl.Utf8)
    out = (
        x.cast(pl.Utf8)
        .str.replace_all(",", "")
        .str.extract(r"(-?\d+(?:\.\d+)?)")
        .cast(pl.Float64, strict=False)
    )
    return out.to_list() if array_like else out


def parse_double(x):
    """readr's ``parse_double()`` — strict floating-point parser.

    Unlike :func:`parse_number`, this does *not* strip currency symbols
    or extract numbers from mixed text — the whole string must be a
    valid double, otherwise the value becomes null. ``"1.234"`` →
    ``1.234``; ``"$1.99"``, ``"1,234"``, ``"abc"`` → null.

    Type-in / type-out: ``pl.Series`` → ``pl.Series``; ``pl.Expr`` →
    ``pl.Expr``; list / tuple / ndarray → ``list`` (with ``None`` for
    unparseable entries).
    """
    array_like = not isinstance(x, (pl.Series, pl.Expr))
    if array_like:
        x = pl.Series(x, dtype=pl.Utf8)
    out = x.cast(pl.Utf8).cast(pl.Float64, strict=False)
    return out.to_list() if array_like else out


# ---- stringr --------------------------------------------------------

def str_wrap(string, width=80, indent=0, exdent=0, whitespace_only=True):
    """stringr's ``str_wrap()`` — wrap text to a fixed line width.

    Wraps each input string to lines no longer than ``width`` characters,
    breaking on whitespace by default. Mirrors stringr's defaults
    (``width=80``, ``whitespace_only=TRUE``); ``indent`` / ``exdent`` add
    spaces to the first / subsequent lines.

    Accepts a single string or an iterable of strings; returns the same
    shape. Built on Python's :mod:`textwrap` — no R-style pipe, but
    ``hea.str_wrap("...", width=30)`` does what you want.

    Parameters
    ----------
    string : str | Iterable[str]
        Text to wrap. ``None`` entries pass through unchanged.
    width : int, default 80
        Maximum line length (characters).
    indent : int, default 0
        Spaces prepended to the first line of each string.
    exdent : int, default 0
        Spaces prepended to subsequent lines.
    whitespace_only : bool, default True
        Only break at whitespace; never split a word or hyphenated token.
    """
    def _wrap_one(s):
        if s is None:
            return None
        return textwrap.fill(
            str(s),
            width=int(width),
            initial_indent=" " * int(indent),
            subsequent_indent=" " * int(exdent),
            break_long_words=not whitespace_only,
            break_on_hyphens=not whitespace_only,
        )

    if isinstance(string, str):
        return _wrap_one(string)
    return [_wrap_one(s) for s in string]


# ---- forcats --------------------------------------------------------
#
# Reorder ``pl.Enum`` level sets — categorical aesthetics (x ticks,
# legend keys, boxplot groups, …) display in level order, so reordering
# levels reorders the display. Each ``fct_*`` returns a callable
# ``data -> Series`` so the operation can fold into ``mutate`` /
# ``select`` (resolved by ``_resolve_lazy_factors``) or be supplied
# inline to ggplot ``aes(...)`` (resolved by the build pipeline). The
# callable carries ``__hea_label__`` / ``__hea_aes_source__`` so axis
# labels resolve to the source column name, not ``"<function>"``.
#
# Composability — ``fct_rev`` and ``fct_relevel`` accept either a
# column name OR another ``fct_*`` callable, so
# ``fct_rev(fct_infreq("g"))`` translates R's
# ``g |> fct_infreq() |> fct_rev()``. Aggregators
# (``fct_reorder`` / ``fct_reorder2`` / ``fct_infreq``) need the
# original column for grouping and only accept a column name.

# Type for fct_* args that accept either a column name or a chained
# fct_* callable (so ``fct_rev(fct_infreq("g"))`` translates R's pipe).
ColInput = Union[str, Callable[[pl.DataFrame], pl.Series]]


def _label_callable(fn: Callable, label: str) -> Callable:
    """Tag ``fn`` so the renderer / aes_source can pull a label from it."""
    fn.__hea_label__ = label
    fn.__hea_aes_source__ = label
    return fn


def _chain_label(col: ColInput) -> str:
    """Resolve the source-column label for a string OR a chained fct_* callable.

    Used so ``fct_rev(fct_infreq("marital"))`` still labels its axis
    ``"marital"`` — the outer tag propagates from the inner.
    """
    if callable(col):
        return getattr(col, "__hea_label__", "_chain")
    return col


def _resolve_col_input(col: ColInput, data: pl.DataFrame) -> pl.Series:
    """Get the input Series for a string col-name OR a chained callable.

    Lets ``fct_rev`` / ``fct_relevel`` accept either a column name
    (the existing contract) or the output of another ``fct_*`` call
    (so R's ``marital |> fct_infreq() |> fct_rev()`` translates to
    ``fct_rev(fct_infreq("marital"))``).
    """
    if callable(col):
        return col(data)
    return data[col]


def _append_unseen_enum_levels(s: pl.Series, levels: list[str]) -> list[str]:
    """Append Enum categories absent from the data so reordering keeps R parity.

    ``fct_reorder`` / ``fct_infreq`` derive their level order from a
    group_by, which only iterates values actually present in ``s``. R's
    ``fct_reorder`` keeps unobserved factor levels — their NA aggregate
    sorts to the end — so we append them in original Enum order to match.
    """
    if not isinstance(s.dtype, pl.Enum):
        return levels
    seen = set(levels)
    return levels + [lvl for lvl in s.dtype.categories.to_list() if lvl not in seen]


def fct_reorder(col: str, by: str, fn="median", *, desc: bool = False) -> Callable:
    """Reorder ``col``'s factor levels by aggregating ``by`` per level.

    Mirrors R's ``forcats::fct_reorder``. ``fn`` is the name of any
    :class:`polars.Series` aggregation method (``"median"``, ``"mean"``,
    ``"sum"``, ``"min"``, ``"max"``, ``"std"``, ``"count"``, …) or a
    callable ``Series -> scalar``. ``desc=True`` reverses the order so
    the largest aggregate appears first.
    """
    def reorder(data: pl.DataFrame) -> pl.Series:
        if isinstance(fn, str):
            agg_expr = getattr(pl.col(by), fn)()
            ordered = (
                data.lazy()
                .group_by(col, maintain_order=False)
                .agg(agg_expr.alias("_agg"))
                .sort("_agg", descending=desc)
                .collect()
            )
            levels = [str(v) for v in ordered[col].to_list() if v is not None]
        else:
            # Python callable: aggregate per-group in Python so users can
            # pass arbitrary scalar reducers (e.g. ``lambda s: s.quantile(0.9)``).
            # ``.lazy()`` routes through polars' native group_by — hea's
            # subclassed DataFrame exposes a different group_by API.
            grouped = (
                data.lazy()
                .group_by(col, maintain_order=False)
                .agg(pl.col(by).alias("_vals"))
                .collect()
            )
            rows = []
            for row in grouped.iter_rows(named=True):
                level = row[col]
                if level is None:
                    continue
                rows.append((str(level), fn(pl.Series(row["_vals"]))))
            rows.sort(key=lambda r: r[1], reverse=desc)
            levels = [r[0] for r in rows]
        levels = _append_unseen_enum_levels(data[col], levels)
        return data[col].cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(reorder, col)


def fct_reorder2(col: str, x: str, y: str, *, desc: bool = True) -> Callable:
    """Reorder ``col``'s levels by ``y`` at each level's largest ``x``.

    Mirrors R's ``forcats::fct_reorder2``. Designed for the line-plot
    legend-ordering case: a level's rank is the ``y`` value at its
    largest ``x``, so the legend order matches where lines end up at
    the right of the plot. ``desc=True`` (default, like forcats) puts
    the highest end-value first.
    """
    def reorder(data: pl.DataFrame) -> pl.Series:
        ordered = (
            data.lazy()
            .filter(pl.col(x).is_not_null())
            .sort(x)
            .group_by(col, maintain_order=False)
            .agg(pl.col(y).last().alias("_y_at_max_x"))
            .sort("_y_at_max_x", descending=desc, nulls_last=True)
            .collect()
        )
        levels = [str(v) for v in ordered[col].to_list() if v is not None]
        levels = _append_unseen_enum_levels(data[col], levels)
        return data[col].cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(reorder, col)


def fct_rev(col: ColInput) -> Callable:
    """Reverse the level order of ``col``.

    ``col`` is either a column name OR the output of another ``fct_*``
    call — so ``fct_rev(fct_infreq("g"))`` translates R's
    ``g |> fct_infreq() |> fct_rev()``. For Enum inputs, reverses the
    existing level order; for plain string columns, sorts alphabetically
    descending — symmetric with the ascending order ScaleOrdinal would
    otherwise pick.
    """
    def rev(data: pl.DataFrame) -> pl.Series:
        s = _resolve_col_input(col, data)
        if isinstance(s.dtype, pl.Enum):
            levels = list(s.dtype.categories)[::-1]
        else:
            levels = sorted(
                {str(v) for v in s.drop_nulls().to_list()},
                reverse=True,
            )
        return s.cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(rev, _chain_label(col))


def fct_relevel(col: ColInput, *levels: str) -> Callable:
    """Move ``levels`` to the front of ``col``'s factor levels.

    ``col`` is either a column name OR the output of another ``fct_*``
    call (composable with ``fct_infreq`` / ``fct_reorder`` / ``fct_rev``).
    Levels not listed keep their existing relative order behind the
    promoted ones (Enum order if the input is an Enum, alphabetical
    otherwise). Mirrors R's ``forcats::fct_relevel``.
    """
    promoted = [str(lvl) for lvl in levels]

    def relevel(data: pl.DataFrame) -> pl.Series:
        s = _resolve_col_input(col, data)
        if isinstance(s.dtype, pl.Enum):
            existing = list(s.dtype.categories)
        else:
            existing = sorted({str(v) for v in s.drop_nulls().to_list()})
        promoted_set = set(promoted)
        rest = [lvl for lvl in existing if lvl not in promoted_set]
        ordered = promoted + rest
        return s.cast(pl.Utf8).cast(pl.Enum(ordered))

    return _label_callable(relevel, _chain_label(col))


def fct_infreq(col: str) -> Callable:
    """Order ``col``'s factor levels by frequency, descending.

    Mirrors R's ``forcats::fct_infreq``. Most-common level first;
    ties broken by first-encountered order in the data.
    """
    def infreq(data: pl.DataFrame) -> pl.Series:
        # ``.lazy()`` routes through polars' native group_by API; hea's
        # subclassed DataFrame has its own group_by signature.
        counts = (
            data.lazy()
            .group_by(col, maintain_order=True)
            .agg(pl.len().alias("_n"))
            .sort("_n", descending=True)
            .collect()
        )
        levels = [str(v) for v in counts[col].to_list() if v is not None]
        levels = _append_unseen_enum_levels(data[col], levels)
        return data[col].cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(infreq, col)


def _input_levels(s: pl.Series) -> list[str]:
    """Return the canonical level list for ``s`` — Enum categories if
    available, otherwise sorted unique non-null string values. Shared by
    the rename / collapse / lump family so they all use the same source.
    """
    if isinstance(s.dtype, pl.Enum):
        return list(s.dtype.categories)
    return sorted({str(v) for v in s.drop_nulls().to_list()})


def fct_recode(col: str, **renames) -> Callable:
    """Rename (and optionally merge) factor levels via ``new=old`` kwargs.

    Mirrors R's ``forcats::fct_recode``. Kwarg direction matches R
    (``new = old``). Values can be either a single string (1:1 rename)
    or a list/tuple of strings (many:1 merge — Python's equivalent of
    R's repeated-keyword trick, since dict keys can't repeat). Use
    ``**{}`` for keys that aren't valid Python identifiers::

        fct_recode("partyid", **{
            "Republican, strong":    "Strong republican",   # 1:1
            "Republican, weak":      "Not str republican",
            "Other":                 ["No answer", "Don't know", "Other party"],  # many:1
        })

    Unmentioned levels keep their original name and position. For
    sweeping all unmentioned levels into a single bucket, use
    ``fct_collapse(..., other_level=)``.
    """
    if not renames:
        raise ValueError("fct_recode(): pass at least one new=old rename.")
    # Build a flat {old: new} mapping; list/tuple values map every entry.
    old_to_new: dict[str, str] = {}
    for new, old in renames.items():
        if isinstance(old, str):
            old_to_new[old] = new
        elif isinstance(old, (list, tuple)):
            for lvl in old:
                if not isinstance(lvl, str):
                    raise TypeError(
                        f"fct_recode(): {new!r} list contains non-string "
                        f"{lvl!r}."
                    )
                old_to_new[lvl] = new
        else:
            raise TypeError(
                f"fct_recode(): {new!r} maps to {type(old).__name__}; "
                "expected str or list/tuple of str."
            )

    def recode(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        # Preserve original order, replace renamed in place, dedupe
        # so multi-old → one-new doesn't create duplicate level entries.
        new_levels: list[str] = []
        seen: set[str] = set()
        for lvl in old_levels:
            mapped = old_to_new.get(lvl, lvl)
            if mapped not in seen:
                new_levels.append(mapped)
                seen.add(mapped)
        return s.cast(pl.Utf8).replace(old_to_new).cast(pl.Enum(new_levels))

    return _label_callable(recode, col)


def fct_collapse(
    col: str, *, other_level: str | None = None, **groups: list
) -> Callable:
    """Merge many old levels into one new level via ``new=[old, ...]`` kwargs.

    Mirrors R's ``forcats::fct_collapse``. Each kwarg maps a new level
    to a list/tuple of old levels that should map to it. Levels not in
    any group keep their original name (with ``other_level=None``) or
    are lumped into ``other_level`` if set. Result level order: declared
    new levels in kwarg order, then either ``other_level`` (if set) or
    any remaining original levels in their original order. Use ``**{}``
    for keys that aren't valid Python identifiers.
    """
    if not groups:
        raise ValueError("fct_collapse(): pass at least one new=[old,...] group.")
    for new, olds in groups.items():
        if not isinstance(olds, (list, tuple)):
            raise TypeError(
                f"fct_collapse(): {new!r} must map to a list/tuple of "
                f"old level names; got {type(olds).__name__}. For 1:1 "
                "rename use fct_recode."
            )
    # Flat {old: new} for the polars replace step.
    old_to_new = {str(old): new for new, olds in groups.items() for old in olds}

    def collapse(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        # Sweep originals into other_level (if set).
        if other_level is not None:
            for lvl in old_levels:
                if lvl not in old_to_new:
                    old_to_new[lvl] = other_level
        # Build the new level list.
        new_levels: list[str] = list(groups.keys())
        if other_level is not None:
            if other_level not in new_levels:
                new_levels.append(other_level)
        else:
            for lvl in old_levels:
                mapped = old_to_new.get(lvl, lvl)
                if mapped not in new_levels:
                    new_levels.append(mapped)
        return s.cast(pl.Utf8).replace(old_to_new).cast(pl.Enum(new_levels))

    return _label_callable(collapse, col)


def _lump_apply(
    s: pl.Series, lumped: set, kept_in_order: list[str], other_level: str
) -> pl.Series:
    """Shared cast: remap ``lumped`` levels to ``other_level``, return Enum.

    Level order = ``kept_in_order`` (original-factor-order of the kept set)
    with ``other_level`` appended unless it's already a kept name (in
    which case the lumped levels merge into the existing one and the
    Enum doesn't grow).
    """
    if not lumped:
        return s.cast(pl.Utf8).cast(pl.Enum(kept_in_order))
    new_levels = list(kept_in_order)
    if other_level not in new_levels:
        new_levels.append(other_level)
    old_to_new = {lvl: other_level for lvl in lumped}
    return s.cast(pl.Utf8).replace(old_to_new).cast(pl.Enum(new_levels))


def fct_lump_n(col: str, n: int, *, other_level: str = "Other") -> Callable:
    """Keep the top ``n`` levels by count; lump the rest into ``other_level``.

    Mirrors R's ``forcats::fct_lump_n``. When a tie spans the cutoff,
    all tied levels are kept (matches R's default ``ties.method="min"``).
    Kept levels are returned in their original factor-level order, then
    ``other_level`` appended. If ``other_level`` is already a level in
    the data, the lumped values merge into it.
    """
    if n < 0:
        raise ValueError(f"fct_lump_n(): n must be >= 0, got {n}.")

    def lump(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        # Per-level counts. value_counts() honors null; we drop nulls so
        # null doesn't compete for a top-n slot.
        vc = (
            s.cast(pl.Utf8)
            .drop_nulls()
            .value_counts()
            .sort("count", descending=True)
        )
        present = vc.height
        if present <= n:
            return s.cast(pl.Utf8).cast(pl.Enum(old_levels))
        cutoff = int(vc["count"][n - 1]) if n > 0 else (int(vc["count"].max()) + 1)
        kept: set[str] = set()
        for r in vc.iter_rows(named=True):
            if r["count"] >= cutoff:
                kept.add(str(r[s.name]))
        kept_in_order = [lvl for lvl in old_levels if lvl in kept]
        lumped = {lvl for lvl in old_levels if lvl not in kept}
        return _lump_apply(s, lumped, kept_in_order, other_level)

    return _label_callable(lump, col)


def fct_lump_lowfreq(col: str, *, other_level: str = "Other") -> Callable:
    """Lump levels into ``other_level`` so it stays the smallest level.

    Mirrors R's ``forcats::fct_lump_lowfreq``. Walks the level counts in
    descending order; a level is KEPT while its count exceeds the sum of
    all smaller-count levels (the "in_smallest" rule from the forcats
    source). Once that fails, every smaller level gets lumped. Kept
    levels preserve their original factor-level order.
    """
    def lump(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        vc = (
            s.cast(pl.Utf8)
            .drop_nulls()
            .value_counts()
            .sort("count", descending=True)
        )
        names = [str(v) for v in vc[s.name].to_list()]
        counts = vc["count"].to_list()
        # forcats::lump_cutoff — index where lumping starts (Python 0-based).
        # Levels at indices >= cutoff_idx get lumped.
        left = sum(counts)
        cutoff_idx = len(counts)
        for i, c in enumerate(counts):
            left -= c
            if c > left:
                cutoff_idx = i + 1
                break
        lumped = set(names[cutoff_idx:])
        kept_in_order = [lvl for lvl in old_levels if lvl not in lumped]
        return _lump_apply(s, lumped, kept_in_order, other_level)

    return _label_callable(lump, col)


# ---- ggplot2 binning (cut_*) ----------------------------------------
#
# Bin a continuous variable into a factor for discrete
# grouping/colouring. Algorithms mirror ``ggplot2::cut_width`` /
# ``cut_interval`` / ``cut_number`` (file ``R/bin.R`` in ggplot2); the
# actual binning delegates to :func:`hea.R.cut` so labels follow R's
# ``"(a,b]"``-style formatting. Accept eager input (Series / ndarray)
# or a column reference (string or ``pl.Expr``); the reference form
# returns a callable that resolves against layer data.

def _cut_resolve(x, data: pl.DataFrame | None = None) -> np.ndarray:
    """Materialize ``x`` as a 1-d float numpy array.

    String / ``pl.Expr`` need a DataFrame context (the lazy form used
    inside ``aes()``); eager forms (Series / ndarray / list) resolve
    immediately.
    """
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(float)
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, str):
        if data is None:
            raise ValueError(
                f"cut_*: column reference {x!r} needs a DataFrame "
                f"context — use this form inside aes(), or pass a "
                f"Series for eager use."
            )
        return data[x].to_numpy().astype(float)
    if isinstance(x, pl.Expr):
        if data is None:
            raise ValueError(
                "cut_*: polars expression needs a DataFrame context "
                "— use this form inside aes(), or pass a Series for "
                "eager use."
            )
        return data.select(x).to_series().to_numpy().astype(float)
    return np.asarray(x, dtype=float)


def _cut_maybe_lazy(x, eager_fn):
    """Run ``eager_fn`` now if ``x`` is concrete; otherwise return a
    closure for the build pipeline to invoke against the layer data."""
    if isinstance(x, (str, pl.Expr)):
        def _lazy(data):
            return eager_fn(_cut_resolve(x, data))
        return _lazy
    return eager_fn(_cut_resolve(x))


def cut_width(x, width, *, center=None, boundary=None, closed="right"):
    """Bin numeric ``x`` into intervals of width ``width``.

    Mirrors ``ggplot2::cut_width``. Default boundary = ``width / 2``,
    so a call like ``cut_width(carat, 0.1)`` produces breaks at
    ``..., 0.05, 0.15, 0.25, ...`` (each bin centered at a tenth).

    Parameters
    ----------
    x : Series, ndarray, str, or pl.Expr
        Numeric data to bin. A string column name or polars expression
        defers binning until the build pipeline supplies a DataFrame.
    width : float
        Bin width.
    center : float, optional
        Center of one bin. Mutually exclusive with ``boundary``.
    boundary : float, optional
        Position of one breakpoint. Mutually exclusive with ``center``.
    closed : {"right", "left"}, default ``"right"``
        Side of each interval that is closed (the side that includes
        its endpoint).

    Returns
    -------
    pl.Series (Enum) | callable
        Series of bin labels, or a callable awaiting a DataFrame when
        ``x`` is a column reference.
    """
    if center is not None and boundary is not None:
        raise ValueError(
            "cut_width: only one of `center` and `boundary` may be specified"
        )

    def _eager(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return pl.Series([], dtype=pl.Utf8)
        b = boundary
        if b is None:
            b = (center - width / 2) if center is not None else width / 2
        x_min = float(finite.min())
        x_max = float(finite.max())
        # Shift origin so one breakpoint coincides with ``b`` modulo ``width``.
        shift = float(np.floor((x_min - b) / width))
        origin = b + shift * width
        # Pad one extra slot on the right so x_max always falls inside a bin
        # regardless of which side is closed (matches ggplot2's
        # ``seq(min_x, max(range) + width, width)``).
        n_bins = max(int(np.ceil((x_max - origin) / width)), 1)
        breaks = origin + np.arange(n_bins + 2) * width
        return _R_cut(arr, breaks, right=(closed == "right"), include_lowest=True)

    return _cut_maybe_lazy(x, _eager)


def cut_interval(x, n=None, length=None, *, closed="right"):
    """Cut ``x`` into ``n`` intervals of equal length, or intervals of
    given ``length``. Exactly one of ``n`` and ``length`` must be set.

    Mirrors ``ggplot2::cut_interval``.
    """
    if (n is None) == (length is None):
        raise ValueError(
            "cut_interval: specify exactly one of `n` and `length`"
        )

    def _eager(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return pl.Series([], dtype=pl.Utf8)
        x_min = float(finite.min())
        x_max = float(finite.max())
        if length is not None:
            # ``fullseq``: round endpoints out to multiples of ``length``.
            start = float(np.floor(x_min / length) * length)
            end = float(np.ceil(x_max / length) * length)
            n_bins = max(int(round((end - start) / length)), 1)
            breaks = start + np.arange(n_bins + 1) * length
            if breaks[-1] < x_max:
                breaks = np.concatenate([breaks, [breaks[-1] + length]])
        else:
            breaks = np.linspace(x_min, x_max, n + 1)
        return _R_cut(arr, breaks, right=(closed == "right"), include_lowest=True)

    return _cut_maybe_lazy(x, _eager)


def cut_number(x, n, *, closed="right"):
    """Cut ``x`` into ``n`` intervals containing equal counts.

    Quantile-based bin edges. Raises if there isn't enough variation
    in the data to produce ``n`` distinct breaks (ggplot2 raises the
    same way).
    """
    def _eager(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return pl.Series([], dtype=pl.Utf8)
        probs = np.linspace(0.0, 1.0, n + 1)
        breaks = np.quantile(finite, probs)
        if np.any(np.diff(breaks) <= 0):
            raise ValueError(
                f"cut_number: insufficient data values to produce {n} bins"
            )
        return _R_cut(arr, breaks, right=(closed == "right"), include_lowest=True)

    return _cut_maybe_lazy(x, _eager)


# ---- tibble ---------------------------------------------------------

def glimpse(df, **kwargs):
    """dplyr: ``glimpse()`` — wide preview. Dispatches to ``.glimpse()``."""
    if hasattr(df, "glimpse"):
        return df.glimpse(**kwargs)
    raise TypeError(
        f"glimpse(): {type(df).__name__} has no .glimpse() method"
    )


# ---- rank family (dplyr) --------------------------------------------

def _as_array(x) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(float)
    return np.asarray(x, dtype=float)


def _rankdata_with_nan(arr: np.ndarray, method: str) -> np.ndarray:
    """``scipy.stats.rankdata`` wrapped to preserve NaN (dplyr's NA → NA)."""
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isnan(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if mask.any():
        out[mask] = _sps.rankdata(arr[mask], method=method)
    return out


def _eager_rank_out(x, arr: np.ndarray):
    """Wrap an ndarray rank result based on the original input type.

    For Python list / tuple input, return a :class:`pl.Series` with NaN
    converted to null — so ``mutate(rn=min_rank(x))`` ends up with a
    proper polars null column instead of a literal NaN value. For
    ndarray input, return the ndarray unchanged (preserves the
    lm/Wilcoxon contract used by :func:`hea.R.rank` / :func:`hea.R.signed_rank`).
    """
    if isinstance(x, (list, tuple)):
        return pl.Series(arr, nan_to_null=True)
    return arr


def row_number(x=None):
    """dplyr's ``row_number()`` — 0-based row position, or ordinal rank.

    Two call shapes:

    * ``row_number()`` (no args) returns the 0-based row position as a
      polars expression, suitable for use inside ``mutate()`` /
      ``select()``. (R / dplyr's ``row_number()`` is 1-based; hea
      follows Python indexing.)
    * ``row_number(x)`` returns the 0-based ordinal rank of ``x`` (ties
      broken by first appearance). Dispatches on input like
      :func:`min_rank`.

    Examples
    --------
    >>> import hea
    >>> from hea.tidy import row_number
    >>> hea.DataFrame({"x": [10, 20, 30]}).mutate(id=row_number())  # doctest: +SKIP
    """
    if x is None:
        return pl.int_range(0, pl.len())
    if isinstance(x, pl.Expr):
        return x.rank("ordinal") - 1
    if isinstance(x, pl.Series):
        return x.rank("ordinal") - 1
    return _eager_rank_out(x, _rankdata_with_nan(_as_array(x), method="ordinal") - 1)


def min_rank(x):
    """dplyr's ``min_rank()`` — 0-based ranks; ties get the smallest rank,
    next rank skipped.

    ``min_rank([1, 5, 5, 17, 22, None])`` → ``Series[0, 1, 1, 3, 4, null]``.
    Dispatches on input like :func:`row_number`; NA / null propagates.
    R / dplyr's ``min_rank()`` starts at 1; hea follows Python indexing.
    """
    if isinstance(x, pl.Expr):
        return x.rank("min") - 1
    if isinstance(x, pl.Series):
        return x.rank("min") - 1
    return _eager_rank_out(x, _rankdata_with_nan(_as_array(x), method="min") - 1)


def dense_rank(x):
    """dplyr's ``dense_rank()`` — 0-based; like :func:`min_rank` but no gaps
    after ties.

    ``dense_rank([1, 5, 5, 17, 22, None])`` → ``Series[0, 1, 1, 2, 3, null]``.
    R / dplyr's ``dense_rank()`` starts at 1; hea follows Python indexing.
    """
    if isinstance(x, pl.Expr):
        return x.rank("dense") - 1
    if isinstance(x, pl.Series):
        return x.rank("dense") - 1
    return _eager_rank_out(x, _rankdata_with_nan(_as_array(x), method="dense") - 1)


def percent_rank(x):
    """dplyr's ``percent_rank()`` — ``(min_rank(x) - 1) / (n - 1)``.

    ``n`` is the non-null count. Returns 0 for the minimum and 1 for the
    maximum; NA / null propagates. ``NaN`` if there's only one non-null value
    (division by zero, matches R).
    """
    if isinstance(x, pl.Expr):
        return (x.rank("min") - 1) / (x.count() - 1)
    if isinstance(x, pl.Series):
        return (x.rank("min") - 1) / (x.count() - 1)
    arr = _as_array(x)
    n = int((~np.isnan(arr)).sum())
    return _eager_rank_out(x, (_rankdata_with_nan(arr, method="min") - 1) / (n - 1))


def cume_dist(x):
    """dplyr's ``cume_dist()`` — cumulative distribution: ``rank("max") / n``.

    ``n`` is the non-null count. Returns the proportion of values ≤ each
    entry; NA / null propagates.
    """
    if isinstance(x, pl.Expr):
        return x.rank("max") / x.count()
    if isinstance(x, pl.Series):
        return x.rank("max") / x.count()
    arr = _as_array(x)
    n = int((~np.isnan(arr)).sum())
    return _eager_rank_out(x, _rankdata_with_nan(arr, method="max") / n)


def ntile(x, n):
    """dplyr's ``ntile(x, n)`` — bucket ``x`` into ``n`` roughly-equal groups.

    Uses ordinal rank, so ties may end up in different buckets. Where ``n``
    doesn't divide the non-null count evenly, the first ``count % n`` buckets
    get one extra element. Bucket labels are 0-based: ``ntile(range(10), 4)``
    → ``[0,0,0,1,1,1,2,2,3,3]`` (sizes 3, 3, 2, 2). NA / null propagates.
    R / dplyr's ``ntile()`` is 1-based; hea follows Python indexing.
    """
    if isinstance(x, pl.Expr):
        r = x.rank("ordinal")
        count = x.count()
        n_larger = count % n
        larger_size = (count + n - 1) // n
        smaller_size = count // n
        threshold = larger_size * n_larger
        return (
            pl.when(r <= threshold)
            .then((r + larger_size - 1) // larger_size)
            .otherwise(
                (r - threshold + smaller_size - 1) // smaller_size + n_larger
            )
        ) - 1
    if isinstance(x, pl.Series):
        r = x.rank("ordinal")
        count = x.count()
        if count == 0:
            return pl.Series([None] * len(x), dtype=pl.UInt32)
        n_larger = count % n
        larger_size = -(-count // n)
        smaller_size = count // n
        threshold = larger_size * n_larger
        upper = (r + larger_size - 1) // larger_size
        lower = (r - threshold + smaller_size - 1) // smaller_size + n_larger
        cond = (r <= threshold).to_numpy()
        return pl.Series(np.where(cond, upper.to_numpy(), lower.to_numpy()) - 1)
    arr = _as_array(x)
    mask = ~np.isnan(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if mask.any():
        ordinal = _sps.rankdata(arr[mask], method="ordinal").astype(np.int64)
        count = int(mask.sum())
        n_larger = count % n
        larger_size = -(-count // n)
        smaller_size = count // n
        threshold = larger_size * n_larger
        upper = (ordinal + larger_size - 1) // larger_size
        lower = (ordinal - threshold + smaller_size - 1) // smaller_size + n_larger
        out[mask] = np.where(ordinal <= threshold, upper, lower) - 1
    return _eager_rank_out(x, out)


# ---- window / mutate helpers (dplyr) --------------------------------

def lag(x, n=1, default=None, order_by=None):
    """dplyr's ``lag()`` — value ``n`` positions before each entry.

    ``lag([2, 5, 11, 11, 19, 35])`` → ``[NA, 2, 5, 11, 11, 19]``. Entries
    with no predecessor (the first ``n``) get ``default`` (``None`` →
    null/NA, matching dplyr's default).

    ``order_by`` reorders the input by another vector before computing
    the lag, then restores the original positions. Use when rows aren't
    already in chronological order:

    >>> df.mutate(prev=hea.lag(pl.col("x"), order_by="t"))  # doctest: +SKIP

    Inside ``group_by() %>% mutate()`` the lag is per-group automatically
    (polars / dplyr's ``mutate`` handles the grouping).

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    return _lag_lead(x, int(n), default, order_by)


def lead(x, n=1, default=None, order_by=None):
    """dplyr's ``lead()`` — value ``n`` positions after each entry.

    ``lead([2, 5, 11, 11, 19, 35])`` → ``[5, 11, 11, 19, 35, NA]``. Mirror
    of :func:`lag` — see that docstring for full arguments.
    """
    return _lag_lead(x, -int(n), default, order_by)


def _lag_lead(x, k, default, order_by):
    """Signed shift: ``k > 0`` lags, ``k < 0`` leads. Shared by lag/lead."""
    if isinstance(x, pl.Expr):
        if order_by is None:
            return x.shift(k, fill_value=default)
        ob = order_by if isinstance(order_by, pl.Expr) else pl.col(order_by)
        inv = ob.arg_sort().arg_sort()
        return x.sort_by(ob).shift(k, fill_value=default).gather(inv)

    is_series = isinstance(x, pl.Series)
    is_ndarray = isinstance(x, np.ndarray)

    if order_by is not None:
        ob_arr = (
            order_by.to_numpy() if isinstance(order_by, pl.Series)
            else np.asarray(order_by)
        )
        order = np.argsort(ob_arr, kind="stable")
        inv = np.argsort(order, kind="stable")
        x_arr = (x.to_numpy() if is_series else np.asarray(x))[order]
        s = pl.Series(x_arr)
    else:
        s = x if is_series else pl.Series(x)

    out = s.shift(k, fill_value=default)

    if order_by is not None:
        out = out.gather(pl.Series(inv))

    if is_series:
        return out
    if is_ndarray:
        return out.to_numpy()
    return out  # list/tuple → pl.Series, matching min_rank/dense_rank pattern


def between(x, left, right):
    """dplyr's ``between(x, left, right)`` — ``left <= x <= right`` (both inclusive).

    NA / null in ``x`` propagates. Wraps polars' ``Expr.between`` /
    ``Series.is_between`` with the dplyr name and a top-level dispatch.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        return x.is_between(left, right, closed="both")
    if isinstance(x, pl.Series):
        return x.is_between(left, right, closed="both")
    is_ndarray = isinstance(x, np.ndarray)
    s = pl.Series(x)
    out = s.is_between(left, right, closed="both")
    return out.to_numpy() if is_ndarray else out


def na_if(x, y):
    """dplyr's ``na_if(x, y)`` — replace value ``y`` in ``x`` with NA / null.

    ``na_if([1, 0, 3, 0], 0)`` → ``[1, NA, 3, NA]``. Useful for cleaning
    sentinel codes (empty strings, ``-99``, …) into proper missing values.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        return pl.when(x == y).then(None).otherwise(x)
    if isinstance(x, pl.Series):
        return x.set(x == y, None)
    is_ndarray = isinstance(x, np.ndarray)
    s = pl.Series(x)
    out = s.set(s == y, None)
    return out.to_numpy() if is_ndarray else out


def near(x, y, tol=1.5e-8):
    """dplyr's ``near(x, y, tol)`` — approximate equality, ``|x - y| < tol``.

    Default tolerance matches dplyr: ``sqrt(.Machine$double.eps)`` ≈ ``1.49e-8``.
    Element-wise; ``y`` may be a scalar or same-length vector.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        return (x - y).abs() < tol
    if isinstance(x, pl.Series):
        return (x - y).abs() < tol
    is_ndarray = isinstance(x, np.ndarray)
    # Scalar shortcut — dplyr's near(1, 1+1e-10) returns a length-1 logical;
    # in Python, returning a bool is the natural shape for scalar inputs.
    if np.isscalar(x) and np.isscalar(y):
        return bool(abs(float(x) - float(y)) < tol)
    arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float) if not np.isscalar(y) else y
    out = np.abs(arr - y_arr) < tol
    return out if is_ndarray else pl.Series(out)


# ---- cumulative helpers (dplyr) -------------------------------------

def cummean(x):
    """dplyr's ``cummean()`` — cumulative mean.

    ``cummean([1, 2, 3, 4, 5])`` → ``[1, 1.5, 2, 2.5, 3]``. NA / null
    propagates from the first missing value onward (matches dplyr's
    ``cumsum(x) / seq_along(x)`` definition).

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        csum = x.cum_sum()
        denom = pl.int_range(1, pl.len() + 1)
        has_na = x.is_null().cum_max()
        return pl.when(has_na).then(None).otherwise(csum / denom)
    is_series = isinstance(x, pl.Series)
    is_ndarray = isinstance(x, np.ndarray)
    arr = np.asarray(x.to_list() if is_series else x, dtype=float)
    csum = np.cumsum(arr)
    out = csum / np.arange(1, len(arr) + 1, dtype=float)
    if is_ndarray:
        return out
    return pl.Series(out, nan_to_null=True)


def cumall(x):
    """dplyr's ``cumall()`` — cumulative all (logical AND).

    TRUE while every value so far has been TRUE. ``FALSE`` absorbs
    everything after it; ``NA`` propagates only until a ``FALSE`` is
    seen (``FALSE`` takes precedence over ``NA``).

    ``cumall([T, T, F, T])`` → ``[T, T, F, F]``;
    ``cumall([T, NA, T])`` → ``[T, NA, NA]``;
    ``cumall([F, NA])`` → ``[F, F]``.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        # has_false = any FALSE so far; has_na = any NA so far
        has_false = (x == False).fill_null(False).cum_max()  # noqa: E712
        has_na = x.is_null().cum_max()
        return (
            pl.when(has_false).then(False)
            .when(has_na).then(None)
            .otherwise(True)
        )
    return _cumall_cumany_eager(x, all_=True)


def cumany(x):
    """dplyr's ``cumany()`` — cumulative any (logical OR).

    FALSE until the first TRUE, then TRUE forever (TRUE absorbs).
    ``NA`` propagates only until a ``TRUE`` is seen.

    ``cumany([F, F, T, F])`` → ``[F, F, T, T]``;
    ``cumany([F, NA, F])`` → ``[F, NA, NA]``;
    ``cumany([T, NA])`` → ``[T, T]``.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        has_true = x.fill_null(False).cum_max()
        has_na = x.is_null().cum_max()
        return (
            pl.when(has_true).then(True)
            .when(has_na).then(None)
            .otherwise(False)
        )
    return _cumall_cumany_eager(x, all_=False)


def _cumall_cumany_eager(x, all_):
    """Shared eager loop for ``cumall`` (all_=True) and ``cumany`` (all_=False).

    Returns a ``pl.Series`` of Boolean (with null) for Series / list /
    tuple input, or an object ndarray for ndarray input.
    """
    is_series = isinstance(x, pl.Series)
    is_ndarray = isinstance(x, np.ndarray)
    src = x.to_list() if is_series else list(x)
    if all_:
        absorb, default_state = False, True   # FALSE absorbs; start TRUE
    else:
        absorb, default_state = True, False   # TRUE absorbs; start FALSE
    state = default_state
    out = []
    for v in src:
        if state is absorb:
            out.append(absorb)
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            state = None
            out.append(None)
            continue
        v_bool = bool(v)
        if v_bool is absorb:
            state = absorb
            out.append(absorb)
        else:
            # non-absorbing: state is True (cumall) or False (cumany), or None
            out.append(state)
    if is_ndarray:
        return np.asarray(out, dtype=object)
    return pl.Series(out, dtype=pl.Boolean)


# ---- positional element pickers (dplyr) -----------------------------

def first(x, default=None, order_by=None, na_rm=True):
    """dplyr's ``first()`` — first non-null element of ``x``.

    ``na_rm=True`` is hea's default (matches the rest of the R-shaped
    API; diverges from dplyr's ``na_rm=FALSE``). Pass ``na_rm=False``
    to get the literal first row, even if it's null. Returns ``default``
    if ``x`` is empty (or, with the default ``na_rm=True``, has no
    non-null entries). ``order_by`` reorders ``x`` before picking.

    Shadows polars' top-level ``pl.first`` (which is a *column* selector,
    not an *element* picker) — the dplyr shape is what you want inside
    ``mutate``:

    >>> df.mutate(diff=pl.col("time") - hea.lag(  # doctest: +SKIP
    ...     pl.col("time"), default=hea.first(pl.col("time"))
    ... ))

    polars' first-column selector remains accessible as ``pl.first``;
    inside a polars Expr, ``pl.col("x").first()`` is the equivalent
    method shape.

    Type-in / type-out: ``pl.Expr`` → scalar ``pl.Expr`` (broadcasts
    inside ``mutate``); ``pl.Series`` / list / tuple / ndarray → Python
    scalar.
    """
    return _first_last_nth(x, 0, default, order_by, na_rm)


def last(x, default=None, order_by=None, na_rm=True):
    """dplyr's ``last()`` — last non-null element of ``x``. Mirror of
    :func:`first`.

    ``na_rm=True`` is hea's default — see :func:`first` for the rationale.
    Shadows polars' top-level ``pl.last``; use ``pl.col("x").last()`` for
    the polars method shape (which returns the literal last row,
    equivalent to ``hea.last(x, na_rm=False)``).
    """
    return _first_last_nth(x, -1, default, order_by, na_rm)


def nth(x, n, order_by=None, default=None, na_rm=True):
    """dplyr's ``nth(x, n)`` — n-th element, 0-based.

    ``nth(x, 0)`` is the first, ``nth(x, 1)`` the second. Negative ``n``
    counts from the end: ``nth(x, -1)`` is the last, ``nth(x, -2)`` the
    second-to-last. Out-of-bounds returns ``default``. R / dplyr's
    ``nth()`` is 1-based; hea follows Python indexing.

    ``na_rm=True`` is hea's default — null entries don't consume an
    index slot (so ``nth([1, None, 3, 4], 1)`` returns ``3``). Pass
    ``na_rm=False`` to count literal row positions. A ``None`` *value*
    at index ``n`` (when ``na_rm=False``) is returned as-is — ``default``
    only fires on OOB.

    Shadows polars' top-level ``pl.nth``. Mirror of :func:`first` for
    the dispatch matrix.
    """
    return _first_last_nth(x, int(n), default, order_by, na_rm)


def _first_last_nth(x, k, default, order_by, na_rm):
    """Shared logic. ``k`` is 0-based: 0 = first, -1 = last, 1 = second…
    """
    if isinstance(x, pl.Expr):
        return _first_last_nth_expr(x, k, default, order_by, na_rm)
    return _first_last_nth_eager(x, k, default, order_by, na_rm)


def _first_last_nth_expr(x_expr, k, default, order_by, na_rm):
    src = x_expr
    if order_by is not None:
        ob = order_by if isinstance(order_by, pl.Expr) else pl.col(order_by)
        src = src.sort_by(ob)
    if na_rm:
        src = src.drop_nulls()
    # polars' ``slice`` handles negative offsets (from end); slice(-1, 1)
    # is the last element, slice(0, 1) the first. ``.first()`` on a
    # 0-length slice (OOB) yields null — no ComputeError, unlike
    # ``.gather()``.
    val = src.slice(k, 1).first()
    if default is None:
        return val
    # OOB if ``|k| > len`` for negative k, or ``k >= len`` for non-negative.
    need_len = -k if k < 0 else k + 1
    in_bounds = src.len() >= need_len
    return pl.when(in_bounds).then(val).otherwise(pl.lit(default))


def _first_last_nth_eager(x, k, default, order_by, na_rm):
    if isinstance(x, pl.Series):
        arr = x.to_list()
    elif isinstance(x, np.ndarray):
        arr = x.tolist()
    else:
        arr = list(x)
    if order_by is not None:
        ob_list = (
            order_by.to_list() if isinstance(order_by, pl.Series)
            else list(order_by)
        )
        order = sorted(range(len(arr)), key=lambda i: ob_list[i])
        arr = [arr[i] for i in order]
    if na_rm:
        arr = [
            v for v in arr
            if not (v is None or (isinstance(v, float) and np.isnan(v)))
        ]
    idx = k if k >= 0 else len(arr) + k
    if 0 <= idx < len(arr):
        return arr[idx]
    return default


# ---- runs / consecutive identity (dplyr) ----------------------------

def consecutive_id(*args):
    """dplyr's ``consecutive_id()`` — 0-based id for each run of consecutive
    equal values.

    Returns 0 for the first row, then increments each time *any* of the
    inputs changes from the previous row. With multiple inputs, treats
    them as a tuple — the id increments when the tuple changes.

    ``consecutive_id([1, 1, 2, 2, 2, 1, 1])`` → ``[0, 0, 1, 1, 1, 2, 2]``.
    ``consecutive_id(["a","a","b","a"], [1,1,1,1])`` → ``[0, 0, 1, 2]``.

    Thin wrapper around polars' ``Expr.rle_id`` (also 0-based). R /
    dplyr's ``consecutive_id()`` is 1-based; hea follows Python indexing.

    Type-in / type-out: all-Expr / string → ``pl.Expr``; eager inputs
    follow the first arg's type (Series / list → Series; ndarray →
    ndarray).
    """
    if not args:
        raise TypeError("consecutive_id() requires at least one argument")

    # Pure-Expr / column-name path → return an Expr suitable for mutate().
    if all(isinstance(a, (pl.Expr, str)) for a in args):
        exprs = [a if isinstance(a, pl.Expr) else pl.col(a) for a in args]
        if len(exprs) == 1:
            return exprs[0].rle_id()
        return pl.struct(exprs).rle_id()

    # Eager path.
    first = args[0]
    if len(args) == 1:
        if isinstance(first, pl.Series):
            return first.rle_id()
        is_ndarray = isinstance(first, np.ndarray)
        out = pl.Series(first).rle_id()
        return out.to_numpy() if is_ndarray else out

    # Multiple eager args — combine into a tiny frame, struct-then-rle.
    cols = {}
    for i, a in enumerate(args):
        name = f"__c{i}"
        cols[name] = a.to_list() if isinstance(a, pl.Series) else list(a)
    df = pl.DataFrame(cols)
    out = df.select(pl.struct(pl.all()).rle_id()).to_series().rename("")
    is_ndarray = isinstance(first, np.ndarray)
    return out.to_numpy() if is_ndarray else out


# =============================================================================
# join_by() and friends — dplyr-style two-table verb specifications.
# =============================================================================

# Map polars BinaryExpr ops (from Expr.meta.serialize) to a readable name.
_BIN_OPS = {"Eq", "Lt", "LtEq", "Gt", "GtEq"}

# Map a "left <op> right" inequality op to the polars join_asof strategy that
# delivers dplyr's closest() semantics.
_CLOSEST_STRATEGY = {
    "GtEq": "backward",  # left >= right: pick largest right ≤ left
    "Gt":   "backward",
    "LtEq": "forward",   # left <= right: pick smallest right ≥ left
    "Lt":   "forward",
    "Eq":   "nearest",
}


def _extract_col_name(arg: Any) -> str:
    """Pull a single column name from a string or ``col(name)`` expression."""
    if isinstance(arg, str):
        return arg
    if isinstance(arg, pl.Expr):
        names = arg.meta.root_names()
        if len(names) != 1:
            raise ValueError(
                f"join_by helper: expected a single column reference, got {names!r}."
            )
        return names[0]
    raise TypeError(
        f"join_by helper: expected str or col(name), got {type(arg).__name__}."
    )


def _parse_join_binary(expr: pl.Expr) -> tuple[str, str, str] | None:
    """Parse ``col(L) <op> col(R)`` → ``(op, L, R)``; return ``None`` otherwise.

    Reads polars' JSON serialization so we can read off the operator (which
    polars doesn't otherwise expose on the public Expr API).
    """
    import json
    if not isinstance(expr, pl.Expr):
        return None
    try:
        tree = json.loads(expr.meta.serialize(format="json"))
    except Exception:
        return None
    if not isinstance(tree, dict) or "BinaryExpr" not in tree:
        return None
    bx = tree["BinaryExpr"]
    left = bx.get("left"); right = bx.get("right"); op = bx.get("op")
    if op not in _BIN_OPS:
        return None
    if not (isinstance(left, dict) and isinstance(right, dict)):
        return None
    if "Column" not in left or "Column" not in right:
        return None
    return op, left["Column"], right["Column"]


@dataclass(frozen=True)
class _Closest:
    """Marker that asks ``join_by`` to route through ``join_asof``."""
    op: str    # one of _CLOSEST_STRATEGY's keys
    left: str
    right: str


def closest(expr: pl.Expr) -> _Closest:
    """Inside :func:`join_by`: take only the closest matching right row.

    Maps to polars' :meth:`pl.DataFrame.join_asof`:

    - ``closest(col('x') >= col('y'))`` — backward asof (largest ``y`` ≤ ``x``).
    - ``closest(col('x') <= col('y'))`` — forward asof (smallest ``y`` ≥ ``x``).
    - ``closest(col('x') == col('y'))`` — nearest asof.

    Mirrors dplyr's ``join_by(closest(...))`` rolling join.
    """
    parsed = _parse_join_binary(expr)
    if parsed is None:
        raise ValueError(
            "closest(): expected a binary inequality between two column "
            "references, e.g. closest(col('x') >= col('y'))."
        )
    return _Closest(*parsed)


def overlaps(x_lower: Any, x_upper: Any, y_lower: Any, y_upper: Any) -> pl.Expr:
    """Inside :func:`join_by`: rows where ``[x_lower, x_upper]`` overlaps ``[y_lower, y_upper]``.

    Equivalent to ``(x_lower <= y_upper) & (y_lower <= x_upper)``. By
    convention the first two arguments are left-side columns and the last
    two are right-side. Accepts column-name strings or ``col(name)``
    expressions.

    Mirrors dplyr's ``join_by(overlaps(...))`` overlap join.
    """
    xl = pl.col(_extract_col_name(x_lower))
    xu = pl.col(_extract_col_name(x_upper))
    yl = pl.col(_extract_col_name(y_lower))
    yu = pl.col(_extract_col_name(y_upper))
    return (xl <= yu) & (yl <= xu)


def within(x_lower: Any, x_upper: Any, y_lower: Any, y_upper: Any) -> pl.Expr:
    """Inside :func:`join_by`: rows where ``[x_lower, x_upper]`` is contained in ``[y_lower, y_upper]``.

    Equivalent to ``(x_lower >= y_lower) & (x_upper <= y_upper)``. By
    convention the first two arguments are left-side columns and the last
    two are right-side.

    Mirrors dplyr's ``join_by(within(...))``.
    """
    xl = pl.col(_extract_col_name(x_lower))
    xu = pl.col(_extract_col_name(x_upper))
    yl = pl.col(_extract_col_name(y_lower))
    yu = pl.col(_extract_col_name(y_upper))
    return (xl >= yl) & (xu <= yu)


@dataclass
class _JoinBy:
    """Normalized join specification produced by :func:`join_by`."""
    # Parallel lists: equi-key columns on the left and right.
    equi_left: list[str] = field(default_factory=list)
    equi_right: list[str] = field(default_factory=list)
    # Non-equi inequalities as ``(polars_op, left_col, right_col)``. Kept
    # separately so we can rewrite right-side columns under suffix
    # disambiguation when needed.
    ineqs: list[tuple[str, str, str]] = field(default_factory=list)
    # Free-form predicate exprs (e.g. from overlaps / within / between).
    exprs: list[pl.Expr] = field(default_factory=list)
    # Rolling spec — at most one closest() per call.
    asof: _Closest | None = None


def join_by(*args: Any) -> _JoinBy:
    """Specify how two tables match for a join — dplyr's ``join_by()``.

    Accepts any combination of:

    - Strings — ``join_by("tailnum")``: same column on both sides.
    - Polars equality — ``join_by(col("dest") == col("faa"))``: rename on
      the right side.
    - Polars inequality — ``join_by(col("id") < col("id"))``: non-equi
      (routes through :meth:`pl.DataFrame.join_where`).
    - :func:`closest` — ``join_by(closest(col("x") >= col("y")))``: rolling
      join (routes through :meth:`pl.DataFrame.join_asof`).
    - :func:`between` (already exported) — pass its return value directly
      for ``y_lower <= x <= y_upper`` predicates inside a join.
    - :func:`overlaps`, :func:`within` — interval-relationship predicates.

    Multiple arguments are conjoined. ``closest()`` may appear at most
    once and combines only with equi-key arguments.
    """
    spec = _JoinBy()
    for a in args:
        _consume_join_by_arg(spec, a)
    return spec


def _consume_join_by_arg(spec: _JoinBy, a: Any) -> None:
    if isinstance(a, str):
        spec.equi_left.append(a); spec.equi_right.append(a); return
    if isinstance(a, _Closest):
        if spec.asof is not None:
            raise ValueError(
                "join_by(): only one closest() condition is supported per call."
            )
        spec.asof = a; return
    if isinstance(a, pl.Expr):
        parsed = _parse_join_binary(a)
        if parsed is not None:
            op, L, R = parsed
            if op == "Eq":
                spec.equi_left.append(L); spec.equi_right.append(R); return
            spec.ineqs.append((op, L, R)); return
        spec.exprs.append(a); return
    raise TypeError(
        f"join_by(): unsupported arg type {type(a).__name__}. "
        "Pass strings, col() comparisons, closest(), between(), "
        "overlaps(), or within()."
    )


# Map "Lt"/"LtEq"/"Gt"/"GtEq" → corresponding Expr builder for non-equi
# predicates (used to reconstitute inequalities with suffixed right refs).
_INEQ_BUILDERS: dict[str, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    "Lt":   lambda l, r: l < r,
    "LtEq": lambda l, r: l <= r,
    "Gt":   lambda l, r: l > r,
    "GtEq": lambda l, r: l >= r,
}


def _numeric_supertype(a: Any, b: Any) -> Any | None:
    """Common numeric dtype that ``a`` and ``b`` both cast to, or ``None``.

    Mirrors dplyr's implicit coercion of numeric join keys: integer +
    float promotes to float; integer + integer promotes to the wider
    signed integer. Non-numeric mismatches (string vs date, etc.)
    return ``None`` and let polars raise its native ``SchemaError``.
    """
    if a == b:
        return a
    if not (a.is_numeric() and b.is_numeric()):
        return None
    if a.is_float() or b.is_float():
        return pl.Float64
    # Both integer kinds: promote to Int64 (widest signed). Hea doesn't
    # try to preserve unsigned-ness — dplyr/R doesn't have it either.
    return pl.Int64


def _align_equi_key_types(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast equi-join key columns to a common supertype where they mismatch.

    Polars rejects equi joins on mismatched dtypes; dplyr/R coerces
    numeric keys (e.g. ``flights.year: integer`` ↔ ``planes.year: integer``
    with ``NA`` works in R because integer is nullable). We auto-cast
    numeric pairs to a common supertype; non-numeric mismatches fall
    through and surface polars' native error.
    """
    left_cast: list[pl.Expr] = []
    right_cast: list[pl.Expr] = []
    for lk, rk in zip(left_keys, right_keys):
        lt = left.schema[lk]
        rt = right.schema[rk]
        if lt == rt:
            continue
        super_t = _numeric_supertype(lt, rt)
        if super_t is None:
            continue
        if lt != super_t:
            left_cast.append(pl.col(lk).cast(super_t))
        if rt != super_t:
            right_cast.append(pl.col(rk).cast(super_t))
    if left_cast:
        left = pl.DataFrame.with_columns(left, left_cast)
    if right_cast:
        right = pl.DataFrame.with_columns(right, right_cast)
    return left, right


def _emit_natural_join_message(shared: list[str]) -> None:
    """Print dplyr's ``Joining with `by = join_by(...)``` info message.

    Goes to stderr to match R's ``message()`` channel — visible in
    Jupyter and REPL output but doesn't pollute stdout-piped scripts.
    """
    import sys
    quoted = ", ".join(repr(c) for c in shared)
    print(f"Joining with `by = join_by({quoted})`", file=sys.stderr)


_CLEAN_NAMES_REPLACE = (
    ("'", ""),
    ('"', ""),
    ("%", "_percent_"),
    ("#", "_number_"),
)
_CLEAN_NAMES_CAMEL_HEAD = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CLEAN_NAMES_CAMEL_TAIL = re.compile(r"([a-z])([A-Z])")
_CLEAN_NAMES_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def _clean_one_name(name: str) -> str:
    """Snake-case a single column name. Mirrors ``janitor::make_clean_names``
    with default options (case='snake', transliterations='Latin-ASCII').
    Disambiguation of duplicates is handled by the caller.
    """
    if not isinstance(name, str):
        name = str(name)
    for find, repl in _CLEAN_NAMES_REPLACE:
        name = name.replace(find, repl)
    # Latin-ASCII transliteration: decompose to base + combining marks,
    # then drop the combining marks (é → e, ñ → n, ü → u, …).
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # camelCase boundaries: ABCDef → ABC_Def, then aB → a_B. Digit↔upper
    # is intentionally not split (janitor: x1Test → x1test, not x1_test).
    name = _CLEAN_NAMES_CAMEL_HEAD.sub(r"\1_\2", name)
    name = _CLEAN_NAMES_CAMEL_TAIL.sub(r"\1_\2", name)
    name = _CLEAN_NAMES_NON_ALNUM.sub("_", name)
    name = name.strip("_").lower()
    if not name:
        name = "x"
    if name[0].isdigit():
        name = "x" + name
    return name


def _disambiguate_clean_names(names: list[str]) -> list[str]:
    """Resolve collisions in a cleaned-name list by appending ``_2``,
    ``_3``, … (janitor's behavior — first dup is ``_2``, not ``_1``).
    """
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
            continue
        k = 2
        while f"{n}_{k}" in seen:
            k += 1
        cand = f"{n}_{k}"
        out.append(cand)
        seen.add(cand)
    return out


def _split_arrange(cols: tuple) -> tuple[list[str], list[bool]]:
    """Split ``arrange`` args into (column names, descending flags)."""
    names: list[str] = []
    desc_flags: list[bool] = []
    for c in cols:
        if isinstance(c, _Desc):
            names.append(c.col)
            desc_flags.append(True)
        else:
            names.append(c)
            desc_flags.append(False)
    return names, desc_flags


def _resolve_anchor(
    anchor: str | int,
    ref_columns: list[str],
    *,
    after: bool = False,
    verb: str = "mutate",
) -> int:
    """Convert a ``_before`` / ``_after`` anchor to an insertion index.

    Accepts either a column name or a **1-indexed** position (matching
    dplyr's ``.before = 1`` semantics — "before the first column").
    """
    if isinstance(anchor, bool):  # bool is an int subclass; reject explicitly
        raise TypeError(f"{verb}(): _before/_after must be a column name or position, not bool.")
    if isinstance(anchor, int):
        n = len(ref_columns)
        if not (1 <= anchor <= n):
            raise ValueError(
                f"{verb}(): position {anchor} out of range for {n} column(s)."
            )
        idx = anchor - 1
    elif anchor in ref_columns:
        idx = ref_columns.index(anchor)
    else:
        raise ValueError(f"{verb}(): column {anchor!r} not in frame.")
    if after:
        idx += 1
    return idx


_VALID_GROUPS = ("drop", "drop_last", "keep", "rowwise")


def _check_groups(_groups: str) -> None:
    """Validate dplyr-style ``_groups`` arg for ``summarize``."""
    if _groups not in _VALID_GROUPS:
        raise ValueError(
            f"summarize(_groups={_groups!r}): expected one of "
            f"{list(_VALID_GROUPS)!r} (matches dplyr's .groups argument)."
        )


def _apply_groups(out_df, by_cols: list, _groups: str):
    """Resolve dplyr-style ``_groups`` on a summarize result.

    Return type depends on ``_groups`` — analogous to dplyr where the
    result is either an ungrouped tibble or a grouped one (tracked via
    metadata; hea tracks it via the GroupBy wrapper type instead).

    - ``"drop"`` → plain :class:`DataFrame` (polars' natural behavior).
    - ``"drop_last"`` → :class:`GroupBy` on ``by_cols[:-1]``; if only one
      group var existed, collapses to ungrouped (matches dplyr).
    - ``"keep"`` → :class:`GroupBy` on all ``by_cols``.
    - ``"rowwise"`` → :class:`GroupBy` on all ``by_cols``. After a
      ``summarize``, each output row is unique by those columns, so
      "each row is its own group" — operationally equivalent to dplyr's
      rowwise. polars expressions are already row-vectorized, so there's
      no further behavioral distinction inside ``mutate`` downstream.
    """
    if _groups == "drop":
        return out_df
    if _groups == "drop_last":
        remaining = list(by_cols[:-1])
        if not remaining:
            return out_df
        return GroupBy(out_df, remaining, {"maintain_order": True})
    # "keep" or "rowwise" — group on all original by cols
    return GroupBy(out_df, list(by_cols), {"maintain_order": True})


def _kwargs_to_exprs(args: tuple, kwargs: dict) -> list[pl.Expr]:
    """Translate ``(*args, **kwargs)`` of a verb into a list of polars exprs.

    Positional args pass through. Keyword args ``name=expr`` get
    ``.alias(name)`` so the kwarg name becomes the output column.
    """
    exprs: list[Any] = list(args)
    for name, expr in kwargs.items():
        if isinstance(expr, (pl.Expr, pl.Series)):
            exprs.append(expr.alias(name))
        else:
            # bare scalar / list — broadcast as a literal column
            exprs.append(pl.lit(expr).alias(name))
    return exprs


def _resolve_lazy_factors(
    df: pl.DataFrame, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    """Replace deferred factor placeholders with concrete expressions / series.

    Two placeholder shapes are resolved here:

    * ``_LazyFactor`` — returned by ``factor("col")`` / ``factor(pl.col(...))``
      for str/Expr inputs because the Enum's level set has to be detected
      from the actual data (polars expressions can't ``.to_list()``
      mid-pipeline). Resolved to a ``pl.Expr``.
    * Tagged callables — ``fct_reorder("col", "by")`` and friends (and
      the ``cut_*`` binning helpers) carry ``__hea_aes_source__`` so the
      ggplot build pipeline can invoke them. The same shape works in
      ``mutate`` / ``select``: call with the frame, get back a Series.

    Verbs that own a materialized frame (``mutate``, ``select``,
    ``GroupBy.mutate``) call this pre-pass so downstream code only
    sees real expressions / series. For kwargs the kwarg name is
    offered as a fallback column name when a ``_LazyFactor`` was
    built from a ``pl.Expr`` without an output_name.
    """
    from .R import _LazyFactor

    def _resolve(v, fallback_name=None):
        if isinstance(v, _LazyFactor):
            return v._resolve(df, fallback_name=fallback_name)
        if callable(v) and hasattr(v, "__hea_aes_source__"):
            return v(df)
        return v

    new_args = tuple(_resolve(a) for a in args)
    new_kwargs = {k: _resolve(v, fallback_name=k) for k, v in kwargs.items()}
    return new_args, new_kwargs


class _TidyRange:
    """Frame-less placeholder for dplyr's ``a:b`` column range.

    The translator emits ``cols_between('a', 'b')`` for R's ``select(a:b)``
    style ranges. ``~`` flips the placeholder to its complement so
    ``select(!(a:b))`` → ``~cols_between('a', 'b')``. The receiver verb's
    tidy-select resolver expands against the frame at call time.

    The polars-native form ``~df["a":"b"]`` continues to work via
    :meth:`DataFrame.__invert__` — this class only covers the path where
    the receiver isn't available syntactically (i.e., inside arg lists
    of a method call, which is exactly where the translator emits).
    """

    __slots__ = ("start", "stop", "exclude")

    def __init__(self, start: str, stop: str, *, exclude: bool = False):
        self.start = start
        self.stop = stop
        self.exclude = exclude

    def __invert__(self) -> "_TidyRange":
        return _TidyRange(self.start, self.stop, exclude=not self.exclude)

    def __repr__(self) -> str:
        prefix = "!" if self.exclude else ""
        return f"{prefix}cols_between({self.start!r}, {self.stop!r})"

    def resolve(self, frame: "DataFrame") -> list[str]:
        """Expand against ``frame``: list of column names (or their
        complement when ``exclude=True``)."""
        cols = frame.cols_between(self.start, self.stop)
        if not self.exclude:
            return cols
        skip = set(cols)
        return [c for c in frame.columns if c not in skip]


def cols_between(start: str, stop: str) -> _TidyRange:
    """tidy-select column range for use without a frame in hand.

    Pass to ``select`` / ``relocate`` / ``pivot_longer`` / ``fill`` /
    anywhere that accepts tidy-select cols. ``~cols_between('a', 'b')``
    is the negated form.

    The :class:`DataFrame` method of the same name (``df.cols_between(
    'a', 'b')``) is the eager equivalent — both routes expand to the
    same list of column names.
    """
    return _TidyRange(start, stop)


class DataFrame(pl.DataFrame):
    """``pl.DataFrame`` with tidyverse-named methods.

    Closed under polars operations: every method that returns a
    DataFrame/LazyFrame/Series returns the corresponding hea subclass.
    Native polars methods (``with_columns``, ``sort``, ``join``, …)
    propagate the subclass through ``self._from_pydf(...)`` automatically;
    the few methods that bypass that route (``describe``, ``corr``,
    ``unstack``, ``sql``, ``match_to_schema``, plus the lazy round-trip
    via ``lazy()`` / ``collect()``) are explicitly re-wrapped below.
    Series-returning methods (``get_column``, ``__getitem__``, the
    ``*_horizontal`` family, …) are wrapped via
    :func:`_install_df_series_overrides`.
    """

    # ---- internal -----------------------------------------------------

    def _wrap(self, df: pl.DataFrame) -> "DataFrame":
        """Re-wrap a polars result as the same subclass as ``self``."""
        return type(self)._from_pydf(df._df)

    def __invert__(self) -> pl.Expr:
        """``~df`` → ``pl.exclude(df.columns)``. Lets the slice form
        ``flights.select(~flights["year":"day"])`` mirror dplyr's
        ``select(!year:day)``: ``~`` is Python's stand-in for ``!`` (which
        can't be overloaded as a prefix operator), and polars already
        uses ``~`` for selector negation.
        """
        return pl.exclude(self.columns)

    # ---- row verbs ----------------------------------------------------

    def filter(self, *predicates: Any, **constraints: Any) -> "DataFrame":
        """Keep rows matching ``predicates``. Polars ``filter`` semantics."""
        return super().filter(*predicates, **constraints)

    def arrange(self, *cols: Any) -> "DataFrame":
        """Sort rows. Wrap a column in ``desc()`` for descending order.

        Nulls **and NaN** sort to the end regardless of direction (dplyr
        default). Polars' ``sort`` would put nulls at the front and rank
        NaN as the largest value (so ``arrange(desc(x))`` would surface
        NaN at the top). Use ``df.sort(...)`` for the polars default.
        """
        names, desc_flags = _split_arrange(cols)
        # NaN-as-largest is what makes polars diverge from dplyr; coerce
        # NaN → null per-column so ``nulls_last=True`` covers both.
        # Done as a sort-key expression so the underlying values aren't
        # actually rewritten in the output frame.
        keys: list[Any] = []
        for n in names:
            dtype = self.schema.get(n)
            if dtype in (pl.Float32, pl.Float64):
                keys.append(pl.col(n).fill_nan(None))
            else:
                keys.append(n)
        return self._wrap(
            super().sort(keys, descending=desc_flags, nulls_last=True)
        )

    def distinct(self, *cols: str, keep_all: bool = False) -> "DataFrame":
        """Keep unique rows.

        With no args, dedupes on all columns. With ``cols``, returns the
        unique combinations of those columns and **drops the others** —
        matches dplyr's default. Pass ``keep_all=True`` to retain the
        other columns (dplyr's ``.keep_all = TRUE``); their values come
        from the first row of each unique combination.
        """
        if not cols:
            return super().unique(maintain_order=True)
        subset = list(cols)
        out = super().unique(subset=subset, maintain_order=True)
        if not keep_all:
            out = out.select(subset)
        return self._wrap(out)

    # ---- column verbs -------------------------------------------------

    def mutate(
        self,
        *args: pl.Expr,
        _before: str | int | None = None,
        _after: str | int | None = None,
        _keep: str = "all",
        _by: str | list[str] | None = None,
        **kwargs: pl.Expr,
    ) -> "DataFrame":
        """Add or modify columns.

        Equivalent to ``df.with_columns(...)`` but with kwarg auto-alias:
        ``mutate(speed=pl.col("distance") / pl.col("time"))`` becomes
        ``with_columns((pl.col("distance") / pl.col("time")).alias("speed"))``.

        Parameters
        ----------
        _before, _after : str | int | None
            Place new columns before / after the anchor. Anchor can be
            a column name OR a 1-indexed position (``_before=1`` means
            "before the first column", matching dplyr). Mutually
            exclusive.
        _keep : {"all", "used", "unused", "none"}
            Which existing columns to retain alongside the new ones.
            ``"all"`` (default) keeps every existing column. ``"used"``
            keeps only the originals referenced by the new expressions
            (plus the new columns themselves). ``"unused"`` keeps the
            originals NOT referenced. ``"none"`` keeps only the new
            columns (plus any ``_by`` grouping columns).
        _by : str | list[str] | None
            Per-call grouping. Wraps each new expression in
            ``.over(_by)`` so values are computed within each group.
        """
        if _before is not None and _after is not None:
            raise ValueError("mutate(): pass either _before= or _after=, not both.")
        if _keep not in {"all", "none", "used", "unused"}:
            raise ValueError(
                f"mutate(): _keep must be one of 'all', 'none', 'used', 'unused'; got {_keep!r}"
            )

        args, kwargs = _resolve_lazy_factors(self, args, kwargs)
        exprs = _kwargs_to_exprs(args, kwargs)
        if _by is not None:
            by = [_by] if isinstance(_by, str) else list(_by)
            exprs = [e.over(by) for e in exprs]

        # dplyr mutate is *sequential* — later expressions see earlier ones.
        # Polars' ``with_columns(*exprs)`` evaluates in parallel, so we
        # chain one expression at a time. For typical mutate calls (a
        # handful of exprs) the overhead is negligible.
        out: pl.DataFrame = self
        for e in exprs:
            out = pl.DataFrame.with_columns(out, e)

        # Names of newly produced columns (last alias wins, matching with_columns).
        new_names: list[str] = []
        for e in exprs:
            try:
                meta = e.meta
                name = meta.output_name()
            except Exception:
                continue
            if name and name not in new_names:
                new_names.append(name)

        if _keep != "all":
            originals = list(self.columns)  # before with_columns
            new_set = set(new_names)
            if _keep == "used" or _keep == "unused":
                # Find originals referenced by any new expression.
                referenced: set[str] = set()
                for e in exprs:
                    try:
                        for r in e.meta.root_names():
                            if r in originals and r not in new_set:
                                referenced.add(r)
                    except Exception:
                        # Some expressions (e.g., literals) may not expose
                        # root_names; treat as referencing nothing.
                        pass
                if _keep == "used":
                    keep_originals = [c for c in originals if c in referenced]
                else:  # unused
                    keep_originals = [c for c in originals if c not in referenced]
            else:  # none
                keep_originals = []

            if _by is not None:
                by_cols = [_by] if isinstance(_by, str) else list(_by)
                for b in by_cols:
                    if b in out.columns and b not in keep_originals and b not in new_set:
                        keep_originals.append(b)

            keep_cols = keep_originals + [c for c in new_names if c in out.columns]
            out = out.select(keep_cols)

        if _before is not None or _after is not None:
            anchor = _before if _before is not None else _after
            other = [c for c in out.columns if c not in new_names]
            idx = _resolve_anchor(
                anchor, other, after=_after is not None, verb="mutate"
            )
            ordered = other[:idx] + [c for c in new_names if c in out.columns] + other[idx:]
            out = out.select(ordered)

        return self._wrap(out)

    def cols_between(self, start: str, end: str) -> list[str]:
        """Column names from ``start`` through ``end`` inclusive.

        Tidyverse equivalent of ``select(start:end)``. Returns a list of
        names (in their original order) so you can splat into ``select``,
        pass to ``pl.exclude`` for the negated form, or compose freely.

        Examples::

            flights.select(flights.cols_between("year", "day"))
            flights.select(pl.exclude(flights.cols_between("year", "day")))
        """
        cols = list(self.columns)
        if start not in cols:
            raise ValueError(f"cols_between(): {start!r} not in frame.")
        if end not in cols:
            raise ValueError(f"cols_between(): {end!r} not in frame.")
        i, j = cols.index(start), cols.index(end)
        if i > j:
            i, j = j, i
        return cols[i : j + 1]

    def select(self, *cols: Any, **named: Any) -> "DataFrame":
        """Subset columns. Accepts column names, polars selectors, exprs.

        A list/tuple positional arg is flattened one level so you can
        pass the result of :meth:`cols_between` directly:
        ``df.select(df.cols_between("a", "c"))``.

        A :class:`DataFrame` positional arg expands to its column names,
        and a :class:`Series` arg expands to ``.name``. This makes the
        slice form ``df.select(df["year":"day"])`` work as the closest
        Python analog to dplyr's ``select(year:day)``.

        Keyword args rename inline. Both forms work:
        ``select(tail_num="tailnum")`` (dplyr-style, string is treated
        as the source column name) and
        ``select(speed=pl.col("velocity") * 2)`` (expression).
        Non-string non-expression values become literal columns
        (rare in select, but matches mutate semantics).
        """
        from .R import _LazyFactor

        flat: list[Any] = []
        for c in cols:
            if isinstance(c, (list, tuple)):
                flat.extend(c)
            elif isinstance(c, pl.DataFrame):
                flat.extend(c.columns)
            elif isinstance(c, pl.Series):
                flat.append(c.name)
            elif isinstance(c, _TidyRange):
                # dplyr's ``select(year:day)`` column range — translator
                # emits ``cols_between('year', 'day')``. ``~cols_between``
                # gives the complement (``select(!(a:b))``).
                flat.extend(c.resolve(self))
            else:
                flat.append(c)
        exprs: list[Any] = []
        for c in flat:
            if isinstance(c, _LazyFactor):
                exprs.append(c._resolve(self))
            else:
                exprs.append(c)
        for new_name, src in named.items():
            if isinstance(src, _LazyFactor):
                exprs.append(src._resolve(self, fallback_name=new_name).alias(new_name))
            elif isinstance(src, str):
                exprs.append(pl.col(src).alias(new_name))
            elif isinstance(src, pl.Expr):
                exprs.append(src.alias(new_name))
            else:
                exprs.append(pl.lit(src).alias(new_name))
        return super().select(exprs)

    def drop(self, *cols: Any, strict: bool = True) -> "DataFrame":
        """Drop columns. Symmetric with :meth:`select`: accepts column
        names, lists/tuples, polars selectors, a :class:`DataFrame` (uses
        ``.columns``), or a :class:`Series` (uses ``.name``). The slice
        form ``df.drop(df["year":"day"])`` is the negated counterpart of
        ``df.select(df["year":"day"])``.
        """
        flat: list[Any] = []
        for c in cols:
            if isinstance(c, (list, tuple)):
                flat.extend(c)
            elif isinstance(c, pl.DataFrame):
                flat.extend(c.columns)
            elif isinstance(c, pl.Series):
                flat.append(c.name)
            elif isinstance(c, _TidyRange):
                flat.extend(c.resolve(self))
            else:
                flat.append(c)
        return self._wrap(super().drop(*flat, strict=strict))

    def rename(self, mapping: dict | None = None, /, **kwargs: str) -> "DataFrame":
        """Rename columns. Accepts a dict (polars-style) or kwargs.

        Tidyverse uses ``new = old`` (kwargs); polars uses ``{old: new}``
        (dict). Both work here:
        ``rename(speed="velocity")`` and ``rename({"velocity": "speed"})``
        are equivalent.
        """
        if mapping is None and not kwargs:
            return self
        if mapping is not None and kwargs:
            raise ValueError("rename(): pass either a dict or kwargs, not both.")
        if mapping is not None:
            return super().rename(mapping)
        # kwargs: new=old → {old: new}
        return super().rename({old: new for new, old in kwargs.items()})

    def clean_names(self) -> "DataFrame":
        """Snake_case all column names — janitor's ``clean_names()``.

        Lowercases, splits camelCase (``mealPlan`` → ``meal_plan``,
        ``ABCDef`` → ``abc_def``), strips Latin diacritics
        (``café`` → ``cafe``), replaces ``%`` with ``percent``, ``#``
        with ``number``, drops apostrophes/quotes, collapses other
        non-alphanumerics into a single ``_``, prepends ``x`` to
        names that would start with a digit (``100m`` → ``x100m``)
        or be empty, and resolves collisions with ``_2``, ``_3``, …
        suffixes (so the first duplicate is ``_2``, not ``_1`` —
        matches janitor).
        """
        cleaned = _disambiguate_clean_names(
            [_clean_one_name(c) for c in self.columns]
        )
        return self._wrap(super().rename(dict(zip(self.columns, cleaned))))

    def relocate(
        self,
        *cols: Any,
        _before: str | int | None = None,
        _after: str | int | None = None,
    ) -> "DataFrame":
        """Move columns to a new position.

        Without ``_before`` / ``_after``, moves ``cols`` to the front
        (dplyr default). Anchors are mutually exclusive and accept a
        column name or a 1-indexed position.

        Each ``cols`` argument can be a column name, a list/tuple of
        names (e.g. from :meth:`cols_between`), or a polars selector
        (e.g. ``pl.selectors.starts_with("arr")``). The order of
        moved columns matches their order in the frame.
        """
        import polars.selectors as cs

        if _before is not None and _after is not None:
            raise ValueError("relocate(): pass either _before= or _after=, not both.")

        moving: list[str] = []
        for c in cols:
            if isinstance(c, str):
                moving.append(c)
            elif isinstance(c, (list, tuple)):
                moving.extend(c)
            elif isinstance(c, pl.DataFrame):
                moving.extend(c.columns)
            elif isinstance(c, pl.Series):
                moving.append(c.name)
            elif isinstance(c, _TidyRange):
                # dplyr's ``relocate(year:dep_time, ...)`` column-range.
                moving.extend(c.resolve(self))
            elif cs.is_selector(c):
                moving.extend(cs.expand_selector(self, c))
            else:
                raise TypeError(
                    f"relocate(): unsupported argument {type(c).__name__}; "
                    "pass column names, lists of names, or polars selectors."
                )
        # Dedupe while preserving first-seen order — selectors can overlap.
        seen: set[str] = set()
        moving = [c for c in moving if not (c in seen or seen.add(c))]
        if not moving:
            return self._wrap(self)
        for c in moving:
            if c not in self.columns:
                raise ValueError(f"relocate(): column {c!r} not in frame.")
        # Preserve the columns' original frame order, not the input order
        # (dplyr behavior: ``relocate(c, a)`` moves them but keeps a-before-c
        # if that's how they appear in the frame).
        moving = [c for c in self.columns if c in moving]
        rest = [c for c in self.columns if c not in moving]
        if _before is None and _after is None:
            ordered = moving + rest
        else:
            anchor = _before if _before is not None else _after
            if isinstance(anchor, str) and anchor in moving:
                raise ValueError(f"relocate(): anchor {anchor!r} is also being moved.")
            idx = _resolve_anchor(
                anchor, rest, after=_after is not None, verb="relocate"
            )
            ordered = rest[:idx] + moving + rest[idx:]
        return super().select(ordered)

    # ---- groups -------------------------------------------------------

    def group_by(self, *cols: Any, **kwargs: Any) -> "GroupBy":
        """Begin a grouped operation. Returns a :class:`GroupBy` wrapper.

        Positional ``cols`` are existing column names (or polars Exprs)
        to group on. Keyword arguments ``name=expr`` define **new**
        columns to be materialized and grouped on, mirroring dplyr's
        ``group_by(hour = sched_dep_time %/% 100)``.

        ``maintain_order=True`` is the default (R-tibble behavior); pass
        ``maintain_order=False`` for polars' default. To define a
        derived column literally named ``maintain_order``, pass the
        polars option as ``_maintain_order=``.
        """
        pl_kwargs: dict[str, Any] = {}
        derived: dict[str, Any] = {}
        # _maintain_order is the explicit option (escape hatch); when
        # passed, bare ``maintain_order`` is reclaimed as a derived
        # column name.
        if "_maintain_order" in kwargs:
            pl_kwargs["maintain_order"] = kwargs["_maintain_order"]
            derived = {k: v for k, v in kwargs.items() if k != "_maintain_order"}
        else:
            for k, v in kwargs.items():
                if k == "maintain_order":
                    pl_kwargs["maintain_order"] = v
                else:
                    derived[k] = v
        pl_kwargs.setdefault("maintain_order", True)

        df: DataFrame = self
        if derived:
            df = self.with_columns(*[
                expr.alias(name) if isinstance(expr, pl.Expr)
                else pl.lit(expr).alias(name)
                for name, expr in derived.items()
            ])
            cols = (*cols, *derived.keys())

        if not cols:
            raise ValueError(
                "group_by(): pass at least one column or derived column."
            )
        return GroupBy(df, list(cols), pl_kwargs)

    def summarize(
        self,
        *args: pl.Expr,
        _by: str | list[str] | None = None,
        _groups: str = "drop",
        **kwargs: pl.Expr,
    ) -> "DataFrame":
        """Reduce the frame to one row per group (or one row total).

        Without ``_by`` and without prior ``group_by``, collapses the
        whole frame to a single row (matches dplyr).
        ``summarize(_by="g", x=pl.col("x").mean())`` is the per-call
        grouping form from dplyr 1.1.0.

        ``_groups`` mirrors dplyr's ``.groups``:

        - ``"drop"`` (default): ungrouped :class:`DataFrame`.
        - ``"drop_last"``: :class:`GroupBy` on all but the last group var
          (or DataFrame if only one group var existed).
        - ``"keep"``: :class:`GroupBy` on all group vars.
        - ``"rowwise"``: :class:`GroupBy` on all group vars (each row is
          its own group after summarize).
        """
        _check_groups(_groups)
        exprs = _kwargs_to_exprs(args, kwargs)
        if _by is None:
            # Single row from the whole frame — no groups to operate on.
            if _groups not in ("drop", "drop_last"):
                raise ValueError(
                    f"summarize(_groups={_groups!r}): no groups to "
                    "preserve — call after group_by(...) or pass _by=."
                )
            return super().select(exprs)
        by = [_by] if isinstance(_by, str) else list(_by)
        result = self._wrap(
            super().group_by(by, maintain_order=True).agg(exprs)
        )
        return _apply_groups(result, by, _groups)

    summarise = summarize  # British spelling, like dplyr.

    def ungroup(self) -> "DataFrame":
        """No-op on a flat DataFrame. Provided for symmetry with :meth:`GroupBy.ungroup`."""
        return self

    def count(
        self,
        *cols: str,
        wt: str | None = None,
        sort: bool = False,
        name: str = "n",
        **kwargs: Any,
    ) -> "DataFrame":
        """Count rows per combination of ``cols``.

        Equivalent to ``group_by(*cols).summarize(n=pl.len())``. Without
        ``cols``, returns a one-row frame with the total. ``sort=True``
        orders the result by count, descending — matches dplyr.

        ``wt=<col>`` switches from row-counting to summing that column
        per group (dplyr's weighted-count: ``count(x, wt = n)`` becomes
        ``group_by(x).summarize(n = sum(n))``). Kwargs of the form
        ``col_name=col_string`` get treated as additional grouping
        columns — R's dplyr style ``count(length = str_length(name), wt = n)``
        passes the kwargs through as the additional mutate-then-group.
        """
        # ``count(name=value)`` in R/dplyr is a mutate-then-count: each kwarg
        # adds a derived column to group by. We accept that shape but only
        # for kwargs whose value is an Expr or a string column reference.
        derived: dict[str, Any] = {}
        for k, v in kwargs.items():
            derived[k] = v
        agg_expr = pl.col(wt).sum().alias(name) if wt is not None else pl.len().alias(name)
        base: pl.DataFrame = self
        if derived:
            base = pl.DataFrame.with_columns(
                base,
                [(v if isinstance(v, pl.Expr) else pl.col(v)).alias(k) for k, v in derived.items()],
            )
            group_cols = [*cols, *derived.keys()]
        else:
            group_cols = list(cols)
        if not group_cols:
            scalar = (
                base.select(pl.col(wt).sum().alias(name))
                if wt is not None
                else pl.DataFrame({name: [self.height]})
            )
            return self._wrap(scalar)
        out = (
            pl.DataFrame.group_by(base, group_cols, maintain_order=True)
            .agg(agg_expr)
        )
        if sort:
            out = out.sort(name, descending=True)
        return self._wrap(out)

    # ---- slice family (ungrouped; grouped versions live on GroupBy) ---

    def slice_head(self, n: int = 1) -> "DataFrame":
        return super().head(n)

    def slice_tail(self, n: int = 1) -> "DataFrame":
        return super().tail(n)

    def slice_min(
        self,
        col: str,
        n: int = 1,
        with_ties: bool = True,
    ) -> "DataFrame":
        """Rows with the smallest ``n`` values of ``col``.

        Matches dplyr semantics: nulls sort to the end, so they only
        appear in the result when there aren't enough non-null rows to
        fill ``n``. With ``with_ties=True`` (dplyr default), rows tied
        with the n-th value are all kept.
        """
        return self._slice_extreme(col, n, with_ties, descending=False)

    def slice_max(
        self,
        col: str,
        n: int = 1,
        with_ties: bool = True,
    ) -> "DataFrame":
        """Rows with the largest ``n`` values of ``col``."""
        return self._slice_extreme(col, n, with_ties, descending=True)

    def _slice_extreme(
        self,
        col: str,
        n: int,
        with_ties: bool,
        *,
        descending: bool,
    ) -> "DataFrame":
        """Shared implementation for slice_min / slice_max.

        Sort by ``col`` (NAs last regardless of direction), then take
        the first ``n`` rows. If ``with_ties``, also keep any rows tied
        with the n-th value, using null-aware equality so an all-NA
        group still keeps its NA rows.
        """
        sorted_df = super().sort(col, descending=descending, nulls_last=True)
        if with_ties and sorted_df.height:
            # n-th value (1-indexed). ``slice(n-1, 1).first()`` returns
            # null if the group has < n rows, which makes the
            # eq_missing comparison correctly include nulls only when
            # they're tied with the cutoff.
            nth = pl.col(col).slice(n - 1, 1).first()
            pos = pl.int_range(0, pl.len())
            out = sorted_df.filter(
                (pos < n) | pl.col(col).eq_missing(nth)
            )
        else:
            out = sorted_df.head(n)
        return self._wrap(out)

    def slice_sample(
        self,
        n: int | None = None,
        prop: float | None = None,
        replace: bool = False,
        seed: int | None = None,
    ) -> "DataFrame":
        """Random rows. Pass ``n=`` for a count or ``prop=`` for a fraction."""
        if (n is None) == (prop is None):
            raise ValueError("slice_sample(): pass exactly one of n= or prop=.")
        return self._wrap(
            super().sample(n=n, fraction=prop, with_replacement=replace, seed=seed)
        )

    # ---- joins (chapter 19) ------------------------------------------

    def inner_join(
        self,
        other: pl.DataFrame,
        by: Any = None,
        *,
        suffix: tuple[str, str] = (".x", ".y"),
        keep: bool = False,
        na_matches: str = "na",
        unmatched: str = "drop",
        relationship: Any = None,
        multiple: str | None = None,
    ) -> "DataFrame":
        """Keep rows that have a match in both tables.

        See :meth:`left_join` for shared parameter semantics. ``unmatched``
        ("drop" default / "error") + ``relationship`` / ``multiple`` are
        dplyr-1.1 hooks for asserting cardinality / forbidding unmatched
        rows; ``"error"`` raises if either side has unmatched rows.
        """
        out = self._do_join(other, by, "inner", suffix, keep, na_matches)
        if unmatched == "error" and out.height < self.height:
            raise ValueError(
                f"inner_join(unmatched='error'): {self.height - out.height} "
                f"of {self.height} left rows had no match."
            )
        return out

    def left_join(
        self,
        other: pl.DataFrame,
        by: Any = None,
        *,
        suffix: tuple[str, str] = (".x", ".y"),
        keep: bool = False,
        na_matches: str = "na",
    ) -> "DataFrame":
        """Keep every row in ``self``; right-side columns come from matching rows.

        Parameters
        ----------
        other
            Right-hand table.
        by
            How to match. Default ``None`` does a natural join on shared
            column names (dplyr behaviour). Pass a string / list of strings
            for shared keys, a ``{left: right}`` dict to rename, or a
            :func:`join_by` spec for non-equi / rolling / overlap joins.
        suffix
            Two-element tuple ``(left_suffix, right_suffix)`` applied to
            non-key columns that collide on both sides. Defaults to dplyr's
            ``(".x", ".y")``.
        keep
            ``False`` (default) drops the right-side key after the match,
            mirroring dplyr's default. ``True`` keeps both key columns.
        na_matches
            ``"na"`` (default) treats nulls as equal during matching;
            ``"never"`` treats every null as distinct.
        """
        return self._do_join(other, by, "left", suffix, keep, na_matches)

    def right_join(
        self,
        other: pl.DataFrame,
        by: Any = None,
        *,
        suffix: tuple[str, str] = (".x", ".y"),
        keep: bool = False,
        na_matches: str = "na",
    ) -> "DataFrame":
        """Keep every row in ``other``; left-side columns come from matching rows.

        See :meth:`left_join` for parameter semantics.
        """
        return self._do_join(other, by, "right", suffix, keep, na_matches)

    def full_join(
        self,
        other: pl.DataFrame,
        by: Any = None,
        *,
        suffix: tuple[str, str] = (".x", ".y"),
        keep: bool = False,
        na_matches: str = "na",
    ) -> "DataFrame":
        """Keep every row from both tables; non-matching rows get nulls.

        See :meth:`left_join` for parameter semantics.
        """
        return self._do_join(other, by, "full", suffix, keep, na_matches)

    def semi_join(
        self,
        other: pl.DataFrame,
        by: Any = None,
        *,
        na_matches: str = "na",
    ) -> "DataFrame":
        """Keep rows in ``self`` that have a match in ``other``; drop the rest.

        Filtering join — no right-side columns are added. See
        :meth:`left_join` for ``by`` and ``na_matches`` semantics.
        """
        return self._do_join(
            other, by, "semi", suffix=(".x", ".y"), keep=False, na_matches=na_matches
        )

    def anti_join(
        self,
        other: pl.DataFrame,
        by: Any = None,
        *,
        na_matches: str = "na",
    ) -> "DataFrame":
        """Keep rows in ``self`` that have *no* match in ``other``.

        Filtering join — no right-side columns are added. See
        :meth:`left_join` for ``by`` and ``na_matches`` semantics.
        """
        return self._do_join(
            other, by, "anti", suffix=(".x", ".y"), keep=False, na_matches=na_matches
        )

    def cross_join(
        self,
        other: pl.DataFrame,
        *,
        suffix: tuple[str, str] = (".x", ".y"),
    ) -> "DataFrame":
        """Cartesian product — every row of ``self`` paired with every row of ``other``."""
        return self._do_join(
            other, by=None, how="cross", suffix=suffix, keep=False, na_matches="na",
        )

    # ---- join internals -----------------------------------------------

    def _do_join(
        self,
        other: pl.DataFrame,
        by: Any,
        how: str,
        suffix: tuple[str, str],
        keep: bool,
        na_matches: str,
    ) -> "DataFrame":
        if na_matches not in ("na", "never"):
            raise ValueError(
                f"na_matches: expected 'na' or 'never', got {na_matches!r}."
            )
        if not isinstance(suffix, tuple) or len(suffix) != 2:
            raise TypeError(
                "suffix=: expected a 2-tuple (left_suffix, right_suffix)."
            )
        spec = self._normalize_join_by(by, other, how=how)
        if spec.asof is not None:
            return self._do_asof(other, spec, how, suffix, keep, na_matches)
        if spec.ineqs or spec.exprs:
            return self._do_where(other, spec, how, suffix, keep)
        return self._do_equi(other, spec, how, suffix, keep, na_matches)

    def _normalize_join_by(
        self, by: Any, other: pl.DataFrame, *, how: str
    ) -> _JoinBy:
        """Resolve ``by=`` into a :class:`_JoinBy`."""
        if how == "cross":
            return _JoinBy()  # no keys for cross
        if by is None:
            shared = [c for c in self.columns if c in other.columns]
            if not shared:
                raise ValueError(
                    "No shared column names to join on. Pass by= or "
                    "join_by(...) to specify keys."
                )
            # Mirror dplyr's natural-join info message so it's obvious
            # when the chosen key set isn't the one the user wanted
            # (e.g. accidentally joining flights to planes on year).
            _emit_natural_join_message(shared)
            return _JoinBy(equi_left=list(shared), equi_right=list(shared))
        if isinstance(by, _JoinBy):
            return by
        if isinstance(by, str):
            return _JoinBy(equi_left=[by], equi_right=[by])
        if isinstance(by, (list, tuple)):
            if not all(isinstance(x, str) for x in by):
                raise TypeError(
                    "by=: list/tuple must contain only column-name strings."
                )
            return _JoinBy(equi_left=list(by), equi_right=list(by))
        if isinstance(by, dict):
            return _JoinBy(equi_left=list(by.keys()), equi_right=list(by.values()))
        raise TypeError(
            f"by=: unsupported type {type(by).__name__}. Pass a string, "
            "list of strings, dict, or join_by(...)."
        )

    def _do_equi(
        self,
        other: pl.DataFrame,
        spec: _JoinBy,
        how: str,
        suffix: tuple[str, str],
        keep: bool,
        na_matches: str,
    ) -> "DataFrame":
        nulls_equal = na_matches == "na"
        # Use a placeholder suffix polars won't collide with, then rename
        # to dplyr's two-sided convention.
        polars_suffix = "__hea_r__"
        # ``coalesce`` mirrors dplyr's ``keep``: keep=False merges shared
        # keys, keep=True leaves both. polars' default behaves differently
        # across how= variants, so we pin it explicitly.
        coalesce = not keep

        if how == "cross":
            out = pl.DataFrame.join(
                self, other, how="cross", suffix=polars_suffix,
            )
            return self._apply_dplyr_suffix(out, suffix, polars_suffix)

        # Coerce numeric type mismatches on key columns to a common
        # supertype (matches dplyr's implicit coercion). Polars would
        # otherwise raise SchemaError on i64-vs-f64 keys, etc.
        left_aligned, right_aligned = _align_equi_key_types(
            self, other, spec.equi_left, spec.equi_right
        )

        same_name = [
            l for l, r in zip(spec.equi_left, spec.equi_right) if l == r
        ]
        diff_name = [
            (l, r) for l, r in zip(spec.equi_left, spec.equi_right) if l != r
        ]
        if not diff_name:
            out = pl.DataFrame.join(
                left_aligned, right_aligned,
                on=same_name, how=how,
                suffix=polars_suffix,
                nulls_equal=nulls_equal,
                coalesce=coalesce,
            )
        else:
            out = pl.DataFrame.join(
                left_aligned, right_aligned,
                left_on=spec.equi_left,
                right_on=spec.equi_right,
                how=how,
                suffix=polars_suffix,
                nulls_equal=nulls_equal,
                coalesce=coalesce,
            )
        return self._apply_dplyr_suffix(out, suffix, polars_suffix)

    def _do_where(
        self,
        other: pl.DataFrame,
        spec: _JoinBy,
        how: str,
        suffix: tuple[str, str],
        keep: bool,
    ) -> "DataFrame":
        if how != "inner":
            raise NotImplementedError(
                f"{how}_join with non-equi conditions is not supported yet — "
                "polars' join_where backs only inner joins. Use inner_join "
                "or open an issue for left/right/full non-equi support."
            )
        # Rename right-side columns that collide with the left so polars'
        # name resolution inside the predicate is unambiguous. We rewrite
        # inequalities to point at the renamed names; free-form exprs
        # (overlaps / within / between) are passed through and rely on
        # non-colliding names (true for every r4ds ch19 example).
        polars_suffix = "__hea_r__"
        rename_map = {
            c: f"{c}{polars_suffix}"
            for c in other.columns if c in self.columns
        }
        right = other.rename(rename_map) if rename_map else other
        preds: list[pl.Expr] = []
        for op, L, R in spec.ineqs:
            R_renamed = rename_map.get(R, R)
            preds.append(_INEQ_BUILDERS[op](pl.col(L), pl.col(R_renamed)))
        # Equi conditions inside an otherwise non-equi join also go via
        # join_where (it accepts equality predicates too).
        for L, R in zip(spec.equi_left, spec.equi_right):
            R_renamed = rename_map.get(R, R)
            preds.append(pl.col(L) == pl.col(R_renamed))
        preds.extend(spec.exprs)
        if not preds:
            raise ValueError("_do_where(): no predicates to apply.")
        out = pl.DataFrame.join_where(self, right, *preds, suffix=polars_suffix)
        return self._apply_dplyr_suffix(out, suffix, polars_suffix)

    def _do_asof(
        self,
        other: pl.DataFrame,
        spec: _JoinBy,
        how: str,
        suffix: tuple[str, str],
        keep: bool,
        na_matches: str,
    ) -> "DataFrame":
        if how not in ("left", "inner", "anti", "semi"):
            raise NotImplementedError(
                f"{how}_join with closest() is not supported — only "
                "left/inner/anti/semi route through join_asof."
            )
        if spec.ineqs or spec.exprs:
            raise NotImplementedError(
                "join_by(closest(...)) does not yet combine with extra "
                "non-equi predicates. Use equi keys only, or open an issue."
            )
        polars_suffix = "__hea_r__"
        strategy = _CLOSEST_STRATEGY[spec.asof.op]
        # Coerce numeric type mismatches on the asof key and any equi
        # grouping keys (same rule as the equi-join path).
        asof_keys_left = list(spec.equi_left) + [spec.asof.left]
        asof_keys_right = list(spec.equi_right) + [spec.asof.right]
        left_aligned, right_aligned = _align_equi_key_types(
            self, other, asof_keys_left, asof_keys_right
        )
        # polars' join_asof requires both frames sorted on the asof key
        # (and by the equi-grouping keys, if any). dplyr preserves left
        # row order on the way out — we tag rows, sort, asof, then
        # restore the original order. Equi keys map to by_left/by_right.
        idx_col = "__hea_idx__"
        left_sorted = pl.DataFrame.with_row_index(left_aligned, idx_col)
        left_sorted = left_sorted.sort(asof_keys_left)
        right_sorted = right_aligned.sort(asof_keys_right)
        kwargs: dict[str, Any] = dict(
            left_on=spec.asof.left,
            right_on=spec.asof.right,
            strategy=strategy,
            suffix=polars_suffix,
            coalesce=not keep,
        )
        if spec.equi_left:
            kwargs["by_left"] = spec.equi_left
            kwargs["by_right"] = spec.equi_right
            # polars can't verify sortedness once `by` groups are
            # involved; we already sorted, so skip the check.
            kwargs["check_sortedness"] = False
        out = pl.DataFrame.join_asof(left_sorted, right_sorted, **kwargs)
        # asof always returns one row per left row (left-join shape);
        # the asof key column on the right tells us whether each row
        # matched (null = no match found).
        right_key_col = spec.asof.right
        # ``coalesce`` collapsed the right key into the left when ``not keep``.
        match_marker = (
            spec.asof.right
            if right_key_col in out.columns
            else (right_key_col + polars_suffix
                  if (right_key_col + polars_suffix) in out.columns
                  else spec.asof.left)
        )
        if how == "inner":
            out = out.filter(pl.col(match_marker).is_not_null())
        elif how == "anti":
            # Keep only unmatched left rows; project back to left's columns.
            out = out.filter(pl.col(match_marker).is_null())
            out = out.select([c for c in out.columns if c in self.columns or c == idx_col])
        elif how == "semi":
            out = out.filter(pl.col(match_marker).is_not_null())
            out = out.select([c for c in out.columns if c in self.columns or c == idx_col])
        # Restore left's original row order and drop the index tag.
        out = out.sort(idx_col).drop(idx_col)
        return self._apply_dplyr_suffix(out, suffix, polars_suffix)

    def _apply_dplyr_suffix(
        self,
        out: pl.DataFrame,
        suffix: tuple[str, str],
        polars_suffix: str,
    ) -> "DataFrame":
        """Rewrite polars' single-sided ``suffix`` into dplyr's two-sided form.

        Polars adds ``polars_suffix`` only to right-side columns that
        collide. dplyr renames both: the left column gets ``suffix[0]``
        and the right column gets ``suffix[1]``. We translate one to the
        other; if both suffix elements are empty no renaming happens.
        """
        ls, rs = suffix
        rename: dict[str, str] = {}
        for name in out.columns:
            if name.endswith(polars_suffix):
                base = name[: -len(polars_suffix)]
                # The corresponding un-suffixed left column should exist;
                # rename both sides.
                if base in out.columns:
                    if ls:
                        rename[base] = f"{base}{ls}"
                    if rs:
                        rename[name] = f"{base}{rs}"
                    elif ls:
                        # No right suffix: strip the polars marker so the
                        # right col becomes ``base`` — but ``base`` is the
                        # left col, so we'd collide. Skip in this edge case.
                        pass
                else:
                    # Right-only — just strip the polars marker.
                    rename[name] = base
        if rename:
            out = out.rename(rename)
        return self._wrap(out)

    # ---- pivots / pull (chapter 5) -----------------------------------

    def _resolve_cols(self, cols: Any) -> list[str]:
        """Turn a ``cols`` argument into a flat list of existing column names.

        Accepts: a string (single name), a list/tuple of strings /
        selectors / exprs, a polars selector, a polars expression
        (e.g. ``pl.exclude("a")`` — resolved against the frame), or
        ``None`` (returns ``[]``).
        """
        import polars.selectors as cs

        def expand_one(c: Any) -> list[str]:
            if isinstance(c, str):
                return [c]
            if cs.is_selector(c):
                return list(cs.expand_selector(self, c))
            if isinstance(c, pl.Series):
                return [c.name]
            if isinstance(c, pl.DataFrame):
                return list(c.columns)
            if isinstance(c, _TidyRange):
                return c.resolve(self)
            if isinstance(c, pl.Expr):
                # ``pl.exclude(...)`` and similar non-selector exprs:
                # resolve by asking polars which columns they cover.
                return list(pl.DataFrame.select(self, c).columns)
            raise TypeError(f"unsupported cols element: {type(c).__name__}")

        if cols is None:
            return []
        if isinstance(cols, (list, tuple)):
            out: list[str] = []
            for c in cols:
                out.extend(expand_one(c))
            return out
        return expand_one(cols)

    def fill(
        self,
        *cols: Any,
        direction: str = "down",
    ) -> "DataFrame":
        """tidyr: ``fill(.data, ..., .direction)`` — replace NA in each
        column by carrying neighboring non-NA values forward / backward.

        ``cols`` accepts the same tidy-select shapes :meth:`select` does
        (bare names, ``selectors.starts_with(...)``, ``everything()``,
        etc.). With no columns, all columns get filled.

        ``direction`` matches R: ``"down"`` (forward-fill, the default),
        ``"up"`` (backward), ``"downup"`` (forward then backward to
        cover leading NAs), ``"updown"``.
        """
        # No cols → everything.
        names = self._resolve_cols(list(cols)) if cols else list(self.columns)
        if direction == "down":
            strategies = ("forward",)
        elif direction == "up":
            strategies = ("backward",)
        elif direction == "downup":
            strategies = ("forward", "backward")
        elif direction == "updown":
            strategies = ("backward", "forward")
        else:
            raise ValueError(
                f"fill(): direction must be 'down' / 'up' / 'downup' / "
                f"'updown'; got {direction!r}"
            )
        exprs: list[pl.Expr] = []
        for c in names:
            e = pl.col(c)
            for strat in strategies:
                e = e.fill_null(strategy=strat)
            exprs.append(e)
        return self._wrap(pl.DataFrame.with_columns(self, exprs))

    def pivot_longer(
        self,
        cols: Any,
        *,
        names_to: str | list[str] = "name",
        values_to: str = "value",
        names_prefix: str | None = None,
        names_sep: str | None = None,
        names_pattern: str | None = None,
        values_drop_na: bool = False,
    ) -> "DataFrame":
        """Wide → long reshape — tidyr's ``pivot_longer``.

        Parameters
        ----------
        cols
            Columns to pivot. Accepts a list of names, a polars selector
            (e.g. ``pl.selectors.starts_with("wk")``), or the result of
            :meth:`cols_between`.
        names_to
            Name of the new column that will hold the pivoted column
            names. Pass a list to split each name into multiple new
            columns — requires ``names_sep`` or ``names_pattern``.
            Use the special string ``".value"`` in the list to indicate
            that piece becomes the output column name (the chapter-5
            ``household`` example).
        values_to
            Name for the new value column. Ignored when ``".value"`` is
            in ``names_to`` (the original values get spread back across
            the .value-derived columns).
        names_prefix
            Regex prefix to strip from each name before splitting (e.g.
            ``"wk"`` to turn ``"wk1"`` into ``"1"``).
        names_sep, names_pattern
            How to split each pivoted name when ``names_to`` is a list.
            Mutually exclusive. ``names_sep`` is a literal separator
            string passed to :meth:`polars.Expr.str.split_exact`;
            ``names_pattern`` is a regex passed to
            :meth:`polars.Expr.str.extract_groups` whose capture
            groups become the new columns.
        values_drop_na
            Drop rows where the pivoted value is null. Useful when the
            wide form has padding nulls (e.g. billboard's wk60–wk76).
        """
        on = self._resolve_cols(cols)
        if not on:
            raise ValueError("pivot_longer(): cols resolved to no columns.")
        index = [c for c in self.columns if c not in on]

        # Tag each input row with its original position so we can sort
        # the result back into row-major order (dplyr's default — all
        # weeks of song 1, then all of song 2, …). Polars' ``unpivot``
        # outputs column-major (all rows of wk1, then all of wk2, …).
        ROW_IDX = "__pivot_longer_row_idx__"
        with_idx = pl.DataFrame.with_row_index(self, name=ROW_IDX)

        # Step 1: unpivot to (index..., ROW_IDX, __name__, __value__).
        long = pl.DataFrame.unpivot(
            with_idx,
            on=on,
            index=[*index, ROW_IDX],
            variable_name="__name__",
            value_name="__value__",
        )

        # Step 2: drop padding nulls if requested.
        if values_drop_na:
            long = long.filter(pl.col("__value__").is_not_null())

        # Step 3: strip prefix.
        if names_prefix is not None:
            long = long.with_columns(
                pl.col("__name__").str.replace(f"^{names_prefix}", "")
            )

        # Normalize names_to.
        names_to_list = [names_to] if isinstance(names_to, str) else list(names_to)

        # Step 4: simple (single-name, no .value) case — just rename.
        if len(names_to_list) == 1 and names_to_list[0] != ".value":
            out = (
                long.rename(
                    {"__name__": names_to_list[0], "__value__": values_to}
                )
                .sort(ROW_IDX)
                .drop(ROW_IDX)
            )
            return self._wrap(out)

        # Step 5: split __name__ into pieces.
        if names_sep is not None and names_pattern is not None:
            raise ValueError(
                "pivot_longer(): pass either names_sep or names_pattern, not both."
            )
        if names_sep is None and names_pattern is None:
            raise ValueError(
                "pivot_longer(): names_to has multiple elements (or includes "
                "'.value'); set names_sep= or names_pattern=."
            )

        n_pieces = len(names_to_list)
        if names_sep is not None:
            long = long.with_columns(
                pl.col("__name__")
                .str.split_exact(names_sep, n_pieces - 1)
                .alias("__parts__")
            ).unnest("__parts__")
        else:
            long = long.with_columns(
                pl.col("__name__")
                .str.extract_groups(names_pattern)
                .alias("__parts__")
            ).unnest("__parts__")

        long = long.drop("__name__")

        # Identify the piece columns (everything new) and rename them.
        kept = set(index) | {"__value__", ROW_IDX}
        piece_cols = [c for c in long.columns if c not in kept]
        if len(piece_cols) != n_pieces:
            raise ValueError(
                f"pivot_longer(): expected {n_pieces} pieces from name split, "
                f"got {len(piece_cols)} ({piece_cols!r}). "
                "Check names_sep / names_pattern matches the column names."
            )
        long = long.rename(dict(zip(piece_cols, names_to_list)))

        # Step 6: handle the .value sentinel — pivot wider on that piece.
        if ".value" in names_to_list:
            non_value = [n for n in names_to_list if n != ".value"]
            out = long.pivot(
                on=".value",
                index=[*index, ROW_IDX, *non_value],
                values="__value__",
            )
            out = out.sort(ROW_IDX).drop(ROW_IDX)
            return self._wrap(out)

        out = (
            long.rename({"__value__": values_to})
            .sort(ROW_IDX)
            .drop(ROW_IDX)
        )
        return self._wrap(out)

    def pivot_wider(
        self,
        *,
        id_cols: Any = None,
        names_from: str | list[str] = "name",
        values_from: str | list[str] = "value",
        values_fill: Any = None,
        names_prefix: str = "",
        names_sep: str = "_",
    ) -> "DataFrame":
        """Long → wide reshape — tidyr's ``pivot_wider``.

        Parameters
        ----------
        id_cols
            Columns that uniquely identify each output row. Defaults to
            all columns not in ``names_from`` or ``values_from`` (matches
            tidyr).
        names_from
            Column(s) whose unique values become new column names. Pass
            a list to combine multiple columns; combined with
            ``names_sep``.
        values_from
            Column(s) whose values fill the new columns.
        values_fill
            Replace null cells with this value. Single value applied to
            every new column.
        names_prefix
            String to prepend to every new column name.
        names_sep
            Separator used when ``names_from`` has multiple columns
            (also used by polars to format struct-style compound names).
        """
        names_from_list = (
            [names_from] if isinstance(names_from, str) else list(names_from)
        )
        values_from_list = (
            [values_from] if isinstance(values_from, str) else list(values_from)
        )
        if id_cols is None:
            excluded = set(names_from_list) | set(values_from_list)
            id_list = [c for c in self.columns if c not in excluded]
        else:
            id_list = self._resolve_cols(id_cols)

        try:
            out = pl.DataFrame.pivot(
                self,
                on=names_from_list,
                index=id_list,
                values=values_from_list,
                aggregate_function=None,
                separator=names_sep,
            )
        except pl.exceptions.ComputeError as e:
            # Duplicates in ``names_from`` × ``id_cols`` — tidyr defaults
            # to producing list-columns + a warning in this case. Match
            # that behavior instead of halting the script.
            if "expected no or a single value" not in str(e):
                raise
            import warnings
            warnings.warn(
                "pivot_wider(): values are not uniquely identified; output "
                "will contain list-columns. Run "
                "`df.group_by(id_cols + names_from).summarize(n=n()).filter(col('n') > 1)` "
                "to find the duplicates.",
                stacklevel=2,
            )
            out = pl.DataFrame.pivot(
                self,
                on=names_from_list,
                index=id_list,
                values=values_from_list,
                aggregate_function=pl.element().implode(),
                separator=names_sep,
            )

        new_cols = [c for c in out.columns if c not in id_list]
        if names_prefix:
            out = out.rename({c: names_prefix + c for c in new_cols})
            new_cols = [names_prefix + c for c in new_cols]
        if values_fill is not None:
            out = out.with_columns(
                [pl.col(c).fill_null(values_fill) for c in new_cols]
            )
        return self._wrap(out)

    def pull(self, col: str | int | None = None) -> pl.Series:
        """Extract a single column as a polars ``Series``.

        Without ``col`` returns the last column (dplyr default), so
        ``df |> distinct(x) |> pull()`` works. Pass a column name or a
        1-indexed position; negative positions count from the right.
        """
        if col is None:
            return self.to_series(self.width - 1)
        if isinstance(col, int):
            idx = col - 1 if col > 0 else self.width + col
            return self.to_series(idx)
        return self.get_column(col)

    # ---- ggplot entry point -------------------------------------------

    def ggplot(self, mapping=None, **aes_kwargs):
        """Start a ggplot from this frame.

        Three equivalent forms::

            df.ggplot(aes(x="a", y="b", color="c")) + geom_point()
            df.ggplot(x="a", y="b", color="c") + geom_point()  # kwarg sugar
            df.ggplot(aes(x="a"), color="c")  # mix; kwargs override

        Direct kwargs are folded into the mapping (Wilkinson's
        variable-to-aesthetic binding without the ``aes()`` wrapper).
        ``aes()`` is still the right choice for layer-level overrides
        and ``after_stat``/``after_scale`` markers.

        Captures the caller's frame and passes it through as ``_env=`` so
        aes-expressions resolve user-defined helpers correctly even when
        called via this wrapper. See ``hea/ggplot/core.py:ggplot.__init__``
        for why.
        """
        # Lazy import: ggplot package depends on dataframe.py at module load,
        # so importing it at top-level would create a cycle.
        import inspect

        from hea.ggplot.core import ggplot as _ggplot
        from hea.plot.dispatch import _frame_env

        env = _frame_env(inspect.currentframe().f_back)
        return _ggplot(self, mapping, _env=env, **aes_kwargs)

    # ---- lazy frame ---------------------------------------------------

    def lazy(self) -> "LazyFrame":
        """Start a lazy query; returns a hea.LazyFrame.

        Overrides ``pl.DataFrame.lazy`` (which would return ``pl.LazyFrame``
        via ``wrap_ldf`` and lose subclass identity through the eager-via-lazy
        round-trip used by ``with_columns``/``sort``/``join``/etc.).
        """
        return LazyFrame._from_pyldf(self._df.lazy())

    # ---- subclass-preserving overrides --------------------------------
    #
    # The methods below build a fresh ``pl.DataFrame`` internally rather than
    # routing through ``self._from_pydf``. We override each to re-wrap so the
    # subclass is preserved.

    def describe(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._wrap(super().describe(*args, **kwargs))

    def corr(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._wrap(super().corr(*args, **kwargs))

    def unstack(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._wrap(super().unstack(*args, **kwargs))

    def sql(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._wrap(super().sql(*args, **kwargs))

    def match_to_schema(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._wrap(super().match_to_schema(*args, **kwargs))

    # ---- summary ------------------------------------------------------

    def summary(
        self,
        *,
        maxsum: int = 7,
        digits: int = 4,
        width: int | None = None,
    ) -> "Summary":
        """R-style per-column summary, mirroring ``summary(data.frame)``.

        Complements :meth:`describe` (polars' wide-format numeric stats)
        with R's per-column, dtype-dispatched view:

        * Numeric — Min, 1st Qu, Median, Mean, 3rd Qu, Max.
        * Boolean — Mode, FALSE count, TRUE count.
        * Enum / Categorical — top ``maxsum`` levels with counts; the
          remaining levels collapse into ``(Other)``. Empty levels show
          a count of 0 (matches R's ``summary.factor``).
        * String — Length, Class, Mode (R's ``character`` summary).
        * Date / Datetime / Time — six-stat summary, formatted as dates.

        An ``NA's : N`` row is appended whenever the column has nulls.

        Parameters
        ----------
        maxsum
            Maximum number of distinct levels to enumerate for factor-like
            columns before pooling the rest into ``(Other)``. Matches R's
            ``summary.data.frame`` default of 7.
        digits
            Significant digits for non-integer numeric values, applied
            R-style: integer-valued stats (``Min``/``Max``/etc on integer
            columns) print verbatim; non-integer values get ``signif`` at
            ``digits`` and the block aligns to common decimals.
        width
            Override the terminal width used to wrap blocks. Defaults to
            the current terminal width.

        Returns
        -------
        Summary
            A small value object whose ``__repr__`` lays the blocks out
            side-by-side. Use :attr:`Summary.blocks` to access the raw
            ``(label, value)`` entries programmatically.
        """
        blocks: list[_SummaryBlock] = []
        for col in self.columns:
            blocks.append(
                _summary_block(col, self.get_column(col), maxsum=maxsum, digits=digits)
            )
        return Summary(blocks, width=width)


class Series(pl.Series):
    """``pl.Series`` that preserves the hea subclass through chains.

    Two leak shapes need handling:

    * **Direct methods** (`head`, `slice`, `cast`, `clone`, …) already
      route through ``self._from_pyseries(...)`` and propagate the
      subclass automatically.
    * **Expression-dispatched methods** (``unique``, ``drop_nulls``,
      ``shift``, ``top_k``, ``sample``, ``gather_every``, ``not_``, the
      ``rolling_*`` family, the trig family, etc. — 116 total) have empty
      function bodies on ``pl.Series`` and are auto-wrapped by polars's
      ``call_expr`` decorator. Internally they go through
      ``wrap_s(self._s)`` (`polars/series/utils.py:99`) which hardcodes
      ``pl.Series._from_pyseries`` — losing subclass identity. Plus two
      explicit ``wrap_s`` sites: ``set`` and ``shrink_dtype``.

    We install thin wrappers for every leaky method below at class-def
    time. New polars releases that add expr-dispatched methods are picked
    up automatically by ``_install_series_subclass_overrides``.
    """

    def __invert__(self):
        """``~s`` → ``pl.exclude(s.name)``, mirroring ``~df``'s
        column-exclusion semantics for the single-column case (so
        ``df.select(~df["x"])`` matches ``df.select(~df["x":"y"])``).
        Boolean Series fall through to polars's logical-NOT so filter
        masks like ``df.filter(~mask)`` keep working.
        """
        if self.dtype == pl.Boolean:
            return super().__invert__()
        return pl.exclude(self.name)

    def to_frame(self, name: str | None = None) -> "DataFrame":
        out = super().to_frame(name) if name is not None else super().to_frame()
        return DataFrame._from_pydf(out._df)

    def to_dummies(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return DataFrame._from_pydf(super().to_dummies(*args, **kwargs)._df)

    def value_counts(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return DataFrame._from_pydf(super().value_counts(*args, **kwargs)._df)

    def hist(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return DataFrame._from_pydf(super().hist(*args, **kwargs)._df)

    def is_close(self, *args: Any, **kwargs: Any) -> "Series":
        # Bypasses self._from_pyseries; rewrap.
        out = super().is_close(*args, **kwargs)
        return type(self)._from_pyseries(out._s)

    def _is_ordered_factor(self) -> bool:
        """True when this Series should print levels with ``<`` separators
        (R's ordered-factor display). Two sources, in order:

        1. Local marker set by ``factor(..., ordered=True)`` /
           ``ordered()`` — covers unnamed Series (bare-list inputs).
           Lost on derived ops; that's fine since this is a print-time
           cosmetic.
        2. The ``_ORDERED_COLS_CV`` contextvar — covers named columns
           registered for poly contrasts in model fitting, so the same
           ordered-ness flows to ``df["col"]`` views.
        """
        if getattr(self, "_hea_ordered", False):
            return True
        if self.name:
            from .formula import _ORDERED_COLS_CV
            return self.name in _ORDERED_COLS_CV.get()
        return False

    def __str__(self) -> str:
        base = super().__str__()
        if isinstance(self.dtype, pl.Enum):
            sep = " < " if self._is_ordered_factor() else " "
            return base + "\nLevels: " + sep.join(self.dtype.categories.to_list())
        return base

    def _repr_html_(self) -> str:
        base = super()._repr_html_()
        if isinstance(self.dtype, pl.Enum):
            sep = " &lt; " if self._is_ordered_factor() else " "
            levels_html = (
                f"<small>Levels: {sep.join(self.dtype.categories.to_list())}</small>"
            )
            stripped = base.rstrip()
            if stripped.endswith("</div>"):
                return stripped[: -len("</div>")] + levels_html + "</div>"
            return base + levels_html
        return base


def _install_series_subclass_overrides() -> None:
    """Install hea.Series-aware wrappers for every method on pl.Series that
    bypasses ``self._from_pyseries`` (i.e. routes through ``wrap_s``).

    Runs once at module-import time. Picks up future polars expr-dispatched
    additions automatically — no maintenance treadmill on version bumps.
    """
    from polars.series.utils import _is_empty_method, _undecorated

    def _make_wrapper(meth_name: str):
        pl_method = getattr(pl.Series, meth_name)

        def wrapper(self, *args: Any, **kwargs: Any):
            out = pl_method(self, *args, **kwargs)
            if isinstance(out, pl.Series) and not isinstance(out, Series):
                return type(self)._from_pyseries(out._s)
            return out

        wrapper.__name__ = meth_name
        wrapper.__qualname__ = f"Series.{meth_name}"
        wrapper.__doc__ = pl_method.__doc__
        return wrapper

    # All expr-dispatched methods (auto-discovered).
    leaky_names: list[str] = []
    for name in dir(pl.Series):
        if name.startswith("_"):
            continue
        attr = pl.Series.__dict__.get(name)
        if attr is None or not hasattr(attr, "__wrapped__"):
            continue
        if _is_empty_method(_undecorated(attr)):
            leaky_names.append(name)

    # Plus the two explicit wrap_s sites in polars/series/series.py.
    leaky_names += ["set", "shrink_dtype"]

    for name in leaky_names:
        setattr(Series, name, _make_wrapper(name))


_install_series_subclass_overrides()


# DataFrame methods that return ``pl.Series``. Re-wrap as ``hea.Series`` so
# chains like ``df.get_column("x").to_frame()`` stay in hea-land.
_DF_SERIES_RETURNING = (
    "drop_in_place",
    "fold",
    "get_column",
    "hash_rows",
    "is_duplicated",
    "is_unique",
    "max_horizontal",
    "mean_horizontal",
    "min_horizontal",
    "sum_horizontal",
    "to_series",
    "to_struct",
)


def _install_df_series_overrides() -> None:
    def _make(meth_name: str):
        pl_method = getattr(pl.DataFrame, meth_name)

        def wrapper(self, *args: Any, **kwargs: Any):
            out = pl_method(self, *args, **kwargs)
            if isinstance(out, pl.Series) and not isinstance(out, Series):
                return Series._from_pyseries(out._s)
            return out

        wrapper.__name__ = meth_name
        wrapper.__qualname__ = f"DataFrame.{meth_name}"
        wrapper.__doc__ = pl_method.__doc__
        return wrapper

    for name in _DF_SERIES_RETURNING:
        setattr(DataFrame, name, _make(name))

    # ``__getitem__`` is polymorphic (Series for str key, DataFrame for slice,
    # row tuple for int) — handle each branch.
    pl_getitem = pl.DataFrame.__getitem__

    def __getitem__(self, item):
        out = pl_getitem(self, item)
        if isinstance(out, pl.Series) and not isinstance(out, Series):
            return Series._from_pyseries(out._s)
        if isinstance(out, pl.DataFrame) and not isinstance(out, DataFrame):
            return DataFrame._from_pydf(out._df)
        return out

    __getitem__.__doc__ = pl_getitem.__doc__
    DataFrame.__getitem__ = __getitem__


_install_df_series_overrides()


def _install_is_in_mixed_list_support() -> None:
    """Teach ``pl.Expr.is_in`` to accept Python lists that mix literals
    and ``Expr`` values.

    Polars' built-in ``is_in`` tries to coerce the ``other`` list into a
    homogeneous ``Series``; a list like ``[1, col("r").max()]`` errors
    with ``failed to determine supertype of i64 and object``. The
    dplyr-faithful translation of ``r %in% c(1, max(r))`` is
    ``col("r").is_in([1, col("r").max()])``, so we patch ``is_in``:
    when ``other`` contains any ``pl.Expr``, we expand into an OR-chain
    (``(self == v0) | (self == v1) | …``), which polars evaluates row-
    wise without dtype headaches. All-literal lists pass through to the
    original ``is_in`` unchanged.

    Series-side eager ``is_in`` is left alone — mixing an Expr into an
    eager membership test has no column to bind against, so polars'
    original error is the right answer there.
    """
    _orig_expr_is_in = pl.Expr.is_in

    def wrapper(self, other, *args, **kwargs):
        if isinstance(other, (list, tuple)) and any(
            isinstance(v, pl.Expr) for v in other
        ):
            nulls_equal = kwargs.get("nulls_equal", False)
            result = None
            for v in other:
                rhs = v if isinstance(v, pl.Expr) else pl.lit(v)
                cmp = self.eq_missing(rhs) if nulls_equal else self == rhs
                result = cmp if result is None else (result | cmp)
            return result
        return _orig_expr_is_in(self, other, *args, **kwargs)

    wrapper.__name__ = "is_in"
    wrapper.__qualname__ = "Expr.is_in (hea-patched)"
    wrapper.__doc__ = (_orig_expr_is_in.__doc__ or "") + (
        "\n\nhea extension: accepts a Python list mixing literals and "
        "``pl.Expr`` values. Mixed lists are expanded to an OR-chain of "
        "``self == v`` comparisons, matching R's ``x %in% c(1, max(x))``."
    )
    pl.Expr.is_in = wrapper


_install_is_in_mixed_list_support()


def _install_expr_is_na_alias() -> None:
    """Alias ``pl.Expr.is_na`` to ``is_null`` so R-translated code that
    emits ``col("x").is_na()`` works.

    Polars named its null-check ``is_null`` (``is_nan`` is the float-NaN
    one); the R-to-Python translator emits the R spelling. Without this
    alias, ``Expr`` raises ``AttributeError: 'Expr' object has no
    attribute 'is_na'``.
    """
    if not hasattr(pl.Expr, "is_na"):
        pl.Expr.is_na = pl.Expr.is_null


_install_expr_is_na_alias()


def _install_expr_r_aliases() -> None:
    """Alias R/dplyr spellings of cumulative ops on ``pl.Expr``.

    Polars renamed ``cumsum`` → ``cum_sum``, ``cummax`` → ``cum_max``,
    ``cummin`` → ``cum_min``, ``cumprod`` → ``cum_prod`` somewhere
    around v1.0. R / dplyr keep the un-underscored spellings; the
    R-to-Python translator emits R names. Without these aliases,
    ``col('x').cumsum()`` raises AttributeError on current polars.
    """
    aliases = {
        "cumsum":  "cum_sum",
        "cummax":  "cum_max",
        "cummin":  "cum_min",
        "cumprod": "cum_prod",
    }
    for r_name, polars_name in aliases.items():
        if not hasattr(pl.Expr, r_name) and hasattr(pl.Expr, polars_name):
            setattr(pl.Expr, r_name, getattr(pl.Expr, polars_name))


_install_expr_r_aliases()


class LazyFrame(pl.LazyFrame):
    """``pl.LazyFrame`` that re-wraps materialized results as ``hea.DataFrame``.

    Mostly empty — polars LazyFrame methods route through
    ``self._from_pyldf(...)``, which respects the subclass, so chains
    (``.filter(...).with_columns(...).join(...)``) propagate
    ``hea.LazyFrame`` automatically. The overrides below cover the
    handful of methods that bypass ``self._from_pyldf`` (calling
    ``wrap_ldf`` / ``wrap_df`` instead) — including the eager-via-lazy
    leak point at `polars/lazyframe/frame.py:2510` (collect).
    """

    def _wrap(self, lf: pl.LazyFrame) -> "LazyFrame":
        return type(self)._from_pyldf(lf._ldf)

    def collect(self, *args: Any, **kwargs: Any):
        out = super().collect(*args, **kwargs)
        if isinstance(out, pl.DataFrame):
            return DataFrame._from_pydf(out._df)
        # background=True path — polars returns InProcessQuery whose
        # .fetch() / .fetch_blocking() still uses pl.DataFrame. Rare
        # enough to leave un-wrapped for now (allowlisted).
        return out

    def describe(self, *args: Any, **kwargs: Any) -> "DataFrame":
        # Despite living on LazyFrame, describe() materializes — returns DataFrame.
        return DataFrame._from_pydf(super().describe(*args, **kwargs)._df)

    def match_to_schema(self, *args: Any, **kwargs: Any) -> "LazyFrame":
        return self._wrap(super().match_to_schema(*args, **kwargs))

    def sql(self, *args: Any, **kwargs: Any) -> "LazyFrame":
        return self._wrap(super().sql(*args, **kwargs))

    def group_by(self, *args: Any, **kwargs: Any) -> "_HeaLazyGroupBy":
        return _HeaLazyGroupBy(super().group_by(*args, **kwargs).lgb)

    def group_by_dynamic(self, *args: Any, **kwargs: Any) -> "_HeaLazyGroupBy":
        return _HeaLazyGroupBy(super().group_by_dynamic(*args, **kwargs).lgb)

    def rolling(self, *args: Any, **kwargs: Any) -> "_HeaLazyGroupBy":
        return _HeaLazyGroupBy(super().rolling(*args, **kwargs).lgb)


class _HeaLazyGroupBy(_PlLazyGroupBy):
    """Subclass of polars's ``LazyGroupBy`` that re-wraps every LazyFrame
    return as ``hea.LazyFrame``.

    polars's ``LazyGroupBy.agg`` (and ``head``/``tail``/``sum``/etc.) all
    use ``wrap_ldf(...)`` (`polars/lazyframe/group_by.py:194,263,…`) which
    hardcodes ``pl.LazyFrame``. We auto-wrap every LazyFrame-returning
    method via ``_install_lazy_groupby_overrides`` below so that
    ``df.lazy().group_by('g').agg(...)`` chains stay in hea-land.

    Private (leading underscore) — only reachable via ``LazyFrame.group_by``,
    not part of the public API surface.
    """


def _install_lazy_groupby_overrides() -> None:
    def _make(meth_name: str):
        pl_method = getattr(_PlLazyGroupBy, meth_name)

        def wrapper(self, *args: Any, **kwargs: Any):
            out = pl_method(self, *args, **kwargs)
            if isinstance(out, pl.LazyFrame) and not isinstance(out, LazyFrame):
                return LazyFrame._from_pyldf(out._ldf)
            return out

        wrapper.__name__ = meth_name
        wrapper.__qualname__ = f"_HeaLazyGroupBy.{meth_name}"
        wrapper.__doc__ = pl_method.__doc__
        return wrapper

    for name in dir(_PlLazyGroupBy):
        if name.startswith("_"):
            continue
        attr = getattr(_PlLazyGroupBy, name, None)
        if not callable(attr):
            continue
        setattr(_HeaLazyGroupBy, name, _make(name))


_install_lazy_groupby_overrides()


class GroupBy:
    """Grouped DataFrame returned by :meth:`DataFrame.group_by`.

    Lazy: holds a reference to the parent frame plus the grouping
    columns; doesn't materialize per-group state. Each verb dispatches
    to the appropriate polars idiom (``agg`` for summarize,
    ``.over(...)`` for windowed mutate, ``group_by(...).head/tail`` for
    slice, etc.).
    """

    __slots__ = ("_df", "_by", "_kwargs")

    def __init__(self, df: DataFrame, by: list, kwargs: dict):
        self._df = df
        self._by = by
        self._kwargs = kwargs

    @property
    def df(self) -> DataFrame:
        """The underlying ungrouped frame."""
        return self._df

    @property
    def groups(self) -> list:
        """The grouping column(s)."""
        return list(self._by)

    def ungroup(self) -> DataFrame:
        """Drop the grouping; return the underlying DataFrame."""
        return self._df

    def ggplot(self, mapping=None, **aes_kwargs):
        """Plot the underlying frame; grouping has no plot-side meaning.

        dplyr's grouping is a per-row scope for verbs, not a plot
        partition — ggplot2 handles its own faceting / aesthetics — so
        ``df.group_by(...).mutate(...).ggplot(...)`` should work just
        like ``df.group_by(...).mutate(...).ungroup().ggplot(...)``,
        matching ``ggplot(grouped_tibble, aes(...))`` in R.
        """
        return self._df.ggplot(mapping, **aes_kwargs)

    # ---- collapsing verbs --------------------------------------------

    def summarize(
        self, *args: pl.Expr, _groups: str = "drop", **kwargs: pl.Expr,
    ):
        """One row per group.

        ``_groups`` mirrors dplyr's ``.groups`` — see
        :meth:`DataFrame.summarize` for the full table. Return type
        depends on ``_groups``: :class:`DataFrame` for ``"drop"`` (and
        ``"drop_last"`` with one group var); :class:`GroupBy` otherwise.
        """
        _check_groups(_groups)
        exprs = _kwargs_to_exprs(args, kwargs)
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        result = self._df._wrap(gb.agg(exprs))
        return _apply_groups(result, self._by, _groups)

    summarise = summarize

    def count(self, sort: bool = False, name: str = "n") -> DataFrame:
        """Row count per group. Equivalent to ``summarize(n=pl.len())``."""
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        out = gb.agg(pl.len().alias(name))
        if sort:
            out = out.sort(name, descending=True)
        return self._df._wrap(out)

    # ---- windowed verbs (preserve grouping like dplyr) ---------------

    def mutate(self, *args: pl.Expr, **kwargs: pl.Expr) -> "GroupBy":
        """Add columns whose values are computed within each group.

        Each expression is wrapped in ``.over(group_cols)`` — matches
        dplyr's ``group_by(g) |> mutate(x = mean(x))`` (windowed) and
        contrasts with ``summarize`` (collapsing). Result keeps the
        original row count AND the grouping, so chained
        ``filter`` / ``arrange`` / ``select`` see the same groups.
        Call ``.ungroup()`` to drop back to a plain DataFrame.
        """
        args, kwargs = _resolve_lazy_factors(self._df, args, kwargs)
        exprs = _kwargs_to_exprs(args, kwargs)
        windowed = [e.over(self._by) for e in exprs]
        new_df = self._df._wrap(
            pl.DataFrame.with_columns(self._df, windowed)
        )
        return GroupBy(new_df, self._by, self._kwargs)

    def filter(self, *predicates) -> "GroupBy":
        """Per-group filter — reductions in the predicate become per-group.

        dplyr's ``group_by(g) |> filter(r == max(r))`` keeps rows where
        ``r`` equals the per-group max of ``r``. hea implements this by
        evaluating the combined predicate inside ``.over(group_cols)``,
        so any aggregate sub-expression (``.max()`` / ``.mean()`` /
        ``.sum()`` …) is scoped to the group.

        Don't pre-wrap your predicate with ``.over(...)`` — let GroupBy
        do it. If you want explicit windowing, call ``.ungroup()`` first
        and use the polars-style ``df.filter(expr.over(by))``.
        """
        if not predicates:
            raise TypeError("filter() requires at least one predicate")
        combined = predicates[0]
        for p in predicates[1:]:
            combined = combined & p
        masked = combined.over(self._by)
        new_df = self._df._wrap(pl.DataFrame.filter(self._df, masked))
        return GroupBy(new_df, self._by, self._kwargs)

    def arrange(self, *cols: Any) -> "GroupBy":
        """Sort the underlying frame and preserve grouping.

        Mirrors dplyr 1.0+'s default ``arrange()`` on grouped tibbles:
        sorts globally (ignoring groups). Use this verb for the
        downstream chain to remain grouped; call ``.ungroup().arrange()``
        if you want a flat DataFrame back.
        """
        return GroupBy(self._df.arrange(*cols), self._by, self._kwargs)

    sort = arrange  # polars name

    def select(self, *cols: Any, **named: Any) -> "GroupBy":
        """Select columns, always keeping the grouping vars (dplyr behavior).

        If any group var isn't named in the selection, it's prepended
        automatically — mirrors dplyr's "grouping variables are always
        retained" rule.
        """
        new_df = self._df.select(*cols, **named)
        missing = [g for g in self._by if g not in new_df.columns]
        if missing:
            new_df = self._df.select(*missing, *cols, **named)
        return GroupBy(new_df, self._by, self._kwargs)

    def transmute(self, *args: pl.Expr, **kwargs: pl.Expr) -> "GroupBy":
        """``mutate`` that drops unmentioned columns. Group vars are kept."""
        args, kwargs = _resolve_lazy_factors(self._df, args, kwargs)
        exprs = _kwargs_to_exprs(args, kwargs)
        windowed = [e.over(self._by) for e in exprs]
        # Build the new frame: group cols + the new exprs.
        new_df = self._df._wrap(
            pl.DataFrame.select(
                self._df, *[pl.col(g) for g in self._by], *windowed,
            )
        )
        return GroupBy(new_df, self._by, self._kwargs)

    def distinct(self, *cols: str, keep_all: bool = False) -> "GroupBy":
        """Per-group distinct — the grouping vars are part of the key.

        ``df.group_by(g).distinct(x)`` returns one row per ``(g, x)`` —
        matching dplyr's behavior where grouping vars are always included
        in the distinctness key.
        """
        key_cols = list(self._by) + list(cols)
        new_df = self._df.distinct(*key_cols, keep_all=keep_all)
        return GroupBy(new_df, self._by, self._kwargs)

    def rename(self, mapping: dict | None = None, /, **kwargs: str) -> "GroupBy":
        """Rename columns; group vars follow the rename automatically."""
        new_df = self._df.rename(mapping, **kwargs)
        # Build the old→new map to update self._by
        if mapping is not None:
            old_to_new = dict(mapping)
        else:
            old_to_new = {old: new for new, old in kwargs.items()}
        new_by = [old_to_new.get(g, g) for g in self._by]
        return GroupBy(new_df, new_by, self._kwargs)

    def relocate(self, *args, **kwargs) -> "GroupBy":
        """Reorder columns. Group vars unchanged."""
        return GroupBy(self._df.relocate(*args, **kwargs), self._by, self._kwargs)

    def drop(self, *cols: Any, strict: bool = True) -> "GroupBy":
        """Drop columns. Refuses to drop a grouping variable — call
        ``.ungroup().drop(...)`` if that's really what you want."""
        # Resolve cols to names if possible
        bad = [c for c in cols if isinstance(c, str) and c in self._by]
        if bad:
            raise ValueError(
                f"drop(): cannot drop grouping variable(s) {bad}. "
                "Call .ungroup() first."
            )
        return GroupBy(self._df.drop(*cols, strict=strict), self._by, self._kwargs)

    # ---- slice family per group (preserve grouping) ------------------

    def slice_head(self, n: int = 1) -> "GroupBy":
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        new_df = self._df._wrap(gb.head(n))
        return GroupBy(new_df, self._by, self._kwargs)

    def slice_tail(self, n: int = 1) -> "GroupBy":
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        new_df = self._df._wrap(gb.tail(n))
        return GroupBy(new_df, self._by, self._kwargs)

    def slice_min(
        self,
        col: str,
        n: int = 1,
        with_ties: bool = True,
    ) -> "GroupBy":
        return self._slice_extreme(col, n, with_ties, descending=False)

    def slice_max(
        self,
        col: str,
        n: int = 1,
        with_ties: bool = True,
    ) -> "GroupBy":
        return self._slice_extreme(col, n, with_ties, descending=True)

    def _slice_extreme(
        self,
        col: str,
        n: int,
        with_ties: bool,
        *,
        descending: bool,
    ) -> "GroupBy":
        """Per-group slice_min / slice_max with dplyr-faithful null handling.

        Nulls sort to the end within each group; they're kept only when
        a group has fewer than ``n`` non-null rows. With_ties extends
        the cutoff via null-aware equality so all-NA groups don't get
        silently dropped.
        """
        sort_cols = self._by + [col]
        sort_desc = [False] * len(self._by) + [descending]
        # ``nulls_last`` accepts a per-column list. Set False for group
        # keys (they have no nulls in the typical case, and we want the
        # default ordering) and True for the value column.
        nulls_last = [False] * len(self._by) + [True]
        sorted_df = pl.DataFrame.sort(
            self._df, sort_cols, descending=sort_desc, nulls_last=nulls_last
        )
        if with_ties:
            pos = pl.int_range(0, pl.len()).over(self._by)
            # n-th value within each group; ``.slice(n-1, 1).first()``
            # yields null when the group has < n rows, which is what
            # we want — eq_missing then matches NA-tied rows too.
            nth = pl.col(col).slice(n - 1, 1).first().over(self._by)
            out = pl.DataFrame.filter(
                sorted_df,
                (pos < n) | pl.col(col).eq_missing(nth),
            )
        else:
            gb = pl.DataFrame.group_by(sorted_df, self._by, **self._kwargs)
            out = gb.head(n)
        return GroupBy(self._df._wrap(out), self._by, self._kwargs)

    def slice_sample(
        self,
        n: int | None = None,
        prop: float | None = None,
        replace: bool = False,
        seed: int | None = None,
    ) -> "GroupBy":
        if (n is None) == (prop is None):
            raise ValueError("slice_sample(): pass exactly one of n= or prop=.")
        # No native per-group sample on polars GroupBy; approximate with
        # `int_range over` + filter for n=, and per-group sampling via agg
        # for prop=.
        if n is not None:
            shuffled = (
                pl.int_range(0, pl.len())
                .shuffle(seed=seed)
                .over(self._by)
            )
            out = pl.DataFrame.filter(self._df, shuffled < n)
        else:
            gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
            cols = [c for c in self._df.columns if c not in self._by]
            out = (
                gb.agg(
                    [pl.col(c).sample(fraction=prop, with_replacement=replace, seed=seed)
                     for c in cols]
                )
                .explode(cols)
            )
        return GroupBy(self._df._wrap(out), self._by, self._kwargs)

    # ---- DataFrame-passthrough ---------------------------------------
    #
    # GroupBy is a grouped *view* of a DataFrame. Read-only frame access
    # (column subscript, ``.columns``, ``.height``, ``.dtypes``, …) goes
    # straight through to the underlying frame — there's no per-group
    # semantics to apply. Verbs that DO have per-group semantics
    # (filter / arrange / select / …) are overridden above.

    def __getitem__(self, key):
        """Subscript the underlying frame. Returns a ``Series`` for
        column-name access, a ``DataFrame`` for row slicing. Grouping
        information is dropped — wrap the result in ``.group_by(...)``
        if you need it back."""
        return self._df[key]

    def __len__(self) -> int:
        return len(self._df)

    @property
    def height(self) -> int:
        return self._df.height

    @property
    def width(self) -> int:
        return self._df.width

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def dtypes(self) -> list:
        return list(self._df.dtypes)

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    @property
    def schema(self):
        return self._df.schema

    # ---- representation ----------------------------------------------
    #
    # Match dplyr: a grouped tibble prints as a tibble plus a "Groups:"
    # header. The data is the data — grouping is just metadata, not a
    # different display.

    def _n_groups(self) -> int:
        return self._df.select(pl.struct(self._by).n_unique()).item()

    def __repr__(self) -> str:
        by_str = ", ".join(self._by)
        return (
            f"# Groups: {by_str} [{self._n_groups()}]\n"
            f"{self._df!r}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_html_(self) -> str:
        """Notebook / Jupyter display — frame's HTML plus a Groups: banner."""
        by_str = ", ".join(self._by)
        banner = (
            f'<small>Groups: {by_str} [{self._n_groups()}]</small>'
        )
        # polars DataFrame exposes _repr_html_; delegate.
        inner = self._df._repr_html_() if hasattr(self._df, "_repr_html_") else (
            f"<pre>{self._df!r}</pre>"
        )
        return banner + inner


# ---------------------------------------------------------------------
# R-style summary
# ---------------------------------------------------------------------


@dataclass(slots=True)
class _SummaryBlock:
    """Per-column block of an R-style summary: a header (column name)
    plus a list of ``(label, value)`` rows."""

    name: str
    entries: list[tuple[str, str]] = field(default_factory=list)

    def render(self) -> list[str]:
        """Lay the block out as a list of equal-width strings.

        First element is the centered column name; subsequent elements
        are ``label:value`` rows with labels left-aligned and values
        right-aligned to the widest within the block.
        """
        if self.entries:
            lw = max(len(lbl) for lbl, _ in self.entries)
            vw = max(len(val) for _, val in self.entries)
            body = [f"{lbl:<{lw}}:{val:>{vw}}" for lbl, val in self.entries]
        else:
            body = []
        body_w = max((len(s) for s in body), default=0)
        block_w = max(body_w, len(self.name))
        head = f"{self.name:^{block_w}}"
        body = [f"{s:<{block_w}}" for s in body]
        return [head, *body]


class Summary:
    """R-style ``summary()`` of a DataFrame.

    Returned by :meth:`DataFrame.summary`. Holds one :class:`_SummaryBlock`
    per column; ``__repr__`` packs blocks side-by-side, wrapping to fit
    the terminal width. Display-only: for programmatic per-column stats
    use :meth:`DataFrame.describe` or polars expressions directly.
    """

    __slots__ = ("blocks", "width")

    def __init__(self, blocks: list[_SummaryBlock], width: int | None = None):
        self.blocks = blocks
        self.width = width

    def __repr__(self) -> str:
        w = self.width
        if w is None:
            try:
                w = shutil.get_terminal_size((80, 20)).columns
            except Exception:
                w = 80
        return _render_summary(self.blocks, w)


def _render_summary(blocks: list[_SummaryBlock], width: int) -> str:
    """Pack rendered blocks into rows that fit ``width`` columns."""
    rendered = [b.render() for b in blocks]
    sep = "  "
    rows: list[list[list[str]]] = []
    cur: list[list[str]] = []
    cur_w = 0
    for r in rendered:
        bw = len(r[0])
        added = bw if not cur else len(sep) + bw
        if cur and cur_w + added > width:
            rows.append(cur)
            cur = [r]
            cur_w = bw
        else:
            cur.append(r)
            cur_w += added
    if cur:
        rows.append(cur)

    out: list[str] = []
    for ri, row in enumerate(rows):
        if ri > 0:
            out.append("")
        height = max(len(r) for r in row)
        for i in range(height):
            cells = [
                r[i] if i < len(r) else " " * len(r[0])
                for r in row
            ]
            out.append(sep.join(cells).rstrip())
    return "\n".join(out)


def _summary_block(
    name: str,
    s: pl.Series,
    *,
    maxsum: int,
    digits: int,
) -> _SummaryBlock:
    """Build a single column's summary block, dispatching on dtype."""
    dtype = s.dtype
    n_null = s.null_count()

    if dtype.is_numeric():
        entries = _numeric_entries(s, digits)
        nas_inline = False
    elif dtype == pl.Boolean:
        entries = _boolean_entries(s)
        nas_inline = False
    elif isinstance(dtype, (pl.Enum, pl.Categorical)):
        # _factor_entries appends NA's itself so it can reserve a slot
        # in the maxsum budget (matches R's summary.factor).
        entries = _factor_entries(s, maxsum=maxsum, n_null=n_null)
        nas_inline = True
    elif dtype == pl.String:
        entries = _string_entries(s)
        nas_inline = False
    elif dtype in (pl.Date, pl.Datetime, pl.Time):
        entries = _temporal_entries(s)
        nas_inline = False
    else:
        # Lists, structs, objects — fall back to length/class.
        entries = [
            ("Length", str(s.len())),
            ("Class", str(dtype)),
            ("Mode", str(dtype)),
        ]
        nas_inline = False

    if not nas_inline and n_null > 0:
        entries.append(("NA's", str(n_null)))
    return _SummaryBlock(name, entries)


_NUMERIC_LABELS = ("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")


def _numeric_entries(s: pl.Series, digits: int) -> list[tuple[str, str]]:
    """Six-number summary using R's ``quantile`` type 7 (linear)."""
    s_clean = s.drop_nulls()
    if s_clean.is_empty():
        return list(zip(_NUMERIC_LABELS, ["NA"] * 6))
    stats = [
        s_clean.min(),
        s_clean.quantile(0.25, interpolation="linear"),
        s_clean.median(),
        s_clean.mean(),
        s_clean.quantile(0.75, interpolation="linear"),
        s_clean.max(),
    ]
    formatted = _format_numeric_stats([float(v) for v in stats], digits)
    return list(zip(_NUMERIC_LABELS, formatted))


def _signif_round(x: float, digits: int) -> float:
    """Round ``x`` to ``digits`` significant figures."""
    if x == 0:
        return 0.0
    return round(x, digits - int(math.floor(math.log10(abs(x)))) - 1)


def _format_numeric_stats(values: list[float], digits: int) -> list[str]:
    """Format six-number stats matching R's ``format.default`` semantics.

    Two-stage: ``signif`` each value at ``digits`` to determine whether
    decimals are needed at all, then format each **original** value
    (not the signif version) at the resulting common decimal count via
    fixed-point rounding. This is why ``Mean = 16331.025`` in an
    integer-magnitude column prints as ``16331`` rather than the signif
    rounded ``16330``: the column's signif values are all
    integer-valued, so D = 0, and the original is rounded to integer.
    """
    signifs = [0.0 if v == 0 else _signif_round(v, digits) for v in values]

    if all(v == int(v) for v in signifs):
        return [str(int(round(v))) for v in values]

    max_dec = 0
    for v in signifs:
        if v == 0 or v == int(v):
            continue
        int_digits = int(math.floor(math.log10(abs(v)))) + 1
        if int_digits >= digits:
            continue
        max_dec = max(max_dec, digits - int_digits)
    if max_dec == 0:
        return [str(int(round(v))) for v in values]
    return [f"{v:.{max_dec}f}" for v in values]


def _boolean_entries(s: pl.Series) -> list[tuple[str, str]]:
    """``Mode: logical`` plus FALSE / TRUE counts (matches R)."""
    s_clean = s.drop_nulls()
    n_true = int(s_clean.sum())
    n_false = int(s_clean.len() - n_true)
    return [("Mode", "logical"), ("FALSE", str(n_false)), ("TRUE", str(n_true))]


def _factor_entries(
    s: pl.Series,
    *,
    maxsum: int,
    n_null: int,
) -> list[tuple[str, str]]:
    """Factor counts in level order, with ``(Other)`` collapse and NA's row.

    Matches R's ``summary.factor``: when more levels exist than slots,
    the (maxsum-1) most populous levels are kept and the rest pooled as
    ``(Other)``; an ``NA's`` slot is reserved when the column has nulls.
    """
    slots = maxsum - 1 if n_null > 0 else maxsum

    counts = s.value_counts(sort=True)
    levels = counts[:, 0].to_list()
    nums = counts[:, 1].to_list()
    counted: dict[Any, int] = {
        lvl: int(n) for lvl, n in zip(levels, nums) if lvl is not None
    }

    if isinstance(s.dtype, (pl.Enum, pl.Categorical)):
        all_levels = s.cat.get_categories().to_list()
    else:
        all_levels = sorted(counted.keys())

    pairs = [(lvl, counted.get(lvl, 0)) for lvl in all_levels]

    if len(pairs) > slots:
        # Keep the (slots-1) most populous; pool the rest as "(Other)".
        ranked = sorted(pairs, key=lambda x: -x[1])
        keep_set = {lvl for lvl, _ in ranked[: slots - 1]}
        keep = [(lvl, n) for lvl, n in pairs if lvl in keep_set]
        other = sum(n for lvl, n in pairs if lvl not in keep_set)
        pairs = keep + [("(Other)", other)]

    entries = [(str(lvl), str(n)) for lvl, n in pairs]
    if n_null > 0:
        entries.append(("NA's", str(n_null)))
    return entries


def _string_entries(s: pl.Series) -> list[tuple[str, str]]:
    """R's character summary: just shape, no per-value counts."""
    return [
        ("Length", str(s.len())),
        ("Class", "character"),
        ("Mode", "character"),
    ]


def _temporal_entries(s: pl.Series) -> list[tuple[str, str]]:
    """Six-stat summary on Date / Datetime / Time, formatted as strings.

    For ``Date`` input, polars promotes quantile / mean / median to
    ``Datetime``; we drop the time component to keep the block visually
    consistent (and matches R's ``summary`` on dates returning dates).
    """
    s_clean = s.drop_nulls()
    if s_clean.is_empty():
        return list(zip(_NUMERIC_LABELS, ["NA"] * 6))
    stats = [
        s_clean.min(),
        s_clean.quantile(0.25, interpolation="linear"),
        s_clean.median(),
        s_clean.mean(),
        s_clean.quantile(0.75, interpolation="linear"),
        s_clean.max(),
    ]
    if s.dtype == pl.Date:
        stats = [
            v.date() if isinstance(v, _dt.datetime) else v
            for v in stats
        ]
    formatted = ["NA" if v is None else str(v) for v in stats]
    return list(zip(_NUMERIC_LABELS, formatted))


# ---------------------------------------------------------------------------
# Polars passthrough — only the names a tidy pipeline reaches for.
#
# Expression builders (``col``, ``lit``, ``when``), basic combinators
# (``coalesce``, ``concat_str``, ``concat_list``), the row-wise reducers
# (``min_horizontal`` family), plus a couple of typing/schema classes
# (``Expr``, ``Schema``).
#
# Anything more esoteric — Polars' SQL bridge, GPU engine, range/repeat
# constructors, system config — stays in ``polars`` and is not re-exported.
# Add a name here only when a tidy-flavored example actually needs it.
#
# Dtypes live in :mod:`hea.dtypes`; I/O factories in :mod:`hea.io`.
# ---------------------------------------------------------------------------
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
