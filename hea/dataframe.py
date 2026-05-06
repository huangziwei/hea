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
import shutil
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy as _PlLazyGroupBy

__all__ = [
    "DataFrame",
    "GroupBy",
    "LazyFrame",
    "Series",
    "Summary",
    "desc",
    "tbl",
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
    """Marker for descending sort, produced by ``desc()``."""

    __slots__ = ("col",)

    def __init__(self, col: str):
        self.col = col


def desc(col: str) -> _Desc:
    """Mark a column for descending sort inside ``arrange()``.

    Mirrors dplyr's ``desc()``: ``df.arrange("a", desc("b"))`` sorts by
    ``a`` ascending then ``b`` descending.
    """
    return _Desc(col)


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


def _kwargs_to_exprs(args: tuple, kwargs: dict) -> list[pl.Expr]:
    """Translate ``(*args, **kwargs)`` of a verb into a list of polars exprs.

    Positional args pass through. Keyword args ``name=expr`` get
    ``.alias(name)`` so the kwarg name becomes the output column.
    """
    exprs: list[Any] = list(args)
    for name, expr in kwargs.items():
        if isinstance(expr, pl.Expr):
            exprs.append(expr.alias(name))
        else:
            # bare scalar / list — broadcast as a literal column
            exprs.append(pl.lit(expr).alias(name))
    return exprs


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

    # ---- row verbs ----------------------------------------------------

    def filter(self, *predicates: Any, **constraints: Any) -> "DataFrame":
        """Keep rows matching ``predicates``. Polars ``filter`` semantics."""
        return super().filter(*predicates, **constraints)

    def arrange(self, *cols: Any) -> "DataFrame":
        """Sort rows. Wrap a column in ``desc()`` for descending order.

        Nulls are sorted to the **end** (dplyr default), regardless of
        ascending/descending. This deviates from polars' ``sort``, which
        puts nulls at the front. Use ``df.sort(...)`` directly for the
        polars default.
        """
        names, desc_flags = _split_arrange(cols)
        return self._wrap(
            super().sort(names, descending=desc_flags, nulls_last=True)
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

        Keyword args rename inline. Both forms work:
        ``select(tail_num="tailnum")`` (dplyr-style, string is treated
        as the source column name) and
        ``select(speed=pl.col("velocity") * 2)`` (expression).
        Non-string non-expression values become literal columns
        (rare in select, but matches mutate semantics).
        """
        flat: list[Any] = []
        for c in cols:
            if isinstance(c, (list, tuple)):
                flat.extend(c)
            else:
                flat.append(c)
        exprs: list[Any] = list(flat)
        for new_name, src in named.items():
            if isinstance(src, str):
                exprs.append(pl.col(src).alias(new_name))
            elif isinstance(src, pl.Expr):
                exprs.append(src.alias(new_name))
            else:
                exprs.append(pl.lit(src).alias(new_name))
        return super().select(exprs)

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

        ``maintain_order=True`` is the default (R-tibble behavior); pass
        ``maintain_order=False`` for polars' default.
        """
        kwargs.setdefault("maintain_order", True)
        if not cols:
            raise ValueError("group_by(): pass at least one column.")
        return GroupBy(self, list(cols), kwargs)

    def summarize(
        self,
        *args: pl.Expr,
        _by: str | list[str] | None = None,
        **kwargs: pl.Expr,
    ) -> "DataFrame":
        """Reduce the frame to one row per group (or one row total).

        Without ``_by`` and without prior ``group_by``, collapses the
        whole frame to a single row (matches dplyr).
        ``summarize(_by="g", x=pl.col("x").mean())`` is the per-call
        grouping form from dplyr 1.1.0.
        """
        exprs = _kwargs_to_exprs(args, kwargs)
        if _by is None:
            # Single row from the whole frame.
            return super().select(exprs)
        by = [_by] if isinstance(_by, str) else list(_by)
        return self._wrap(
            super().group_by(by, maintain_order=True).agg(exprs)
        )

    summarise = summarize  # British spelling, like dplyr.

    def ungroup(self) -> "DataFrame":
        """No-op on a flat DataFrame. Provided for symmetry with :meth:`GroupBy.ungroup`."""
        return self

    def count(
        self,
        *cols: str,
        sort: bool = False,
        name: str = "n",
    ) -> "DataFrame":
        """Count rows per combination of ``cols``.

        Equivalent to ``group_by(*cols).summarize(n=pl.len())``. Without
        ``cols``, returns a one-row frame with the total. ``sort=True``
        orders the result by count, descending — matches dplyr.
        """
        if not cols:
            return self._wrap(pl.DataFrame({name: [self.height]}))
        out = (
            super()
            .group_by(list(cols), maintain_order=True)
            .agg(pl.len().alias(name))
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

        out = pl.DataFrame.pivot(
            self,
            on=names_from_list,
            index=id_list,
            values=values_from_list,
            aggregate_function=None,
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

    def ggplot(self, mapping=None):
        """Start a ggplot from this frame: ``df.ggplot(aes("x", "y")) + geom_point()``.

        Equivalent to ``ggplot(self, mapping)`` but pairs naturally with
        the tidyverse chain (``df.filter(...).mutate(...).ggplot(aes())``).
        Layer addition is still via ``+`` (and via fluent methods on the
        returned ``ggplot``, once Phase B's auto-install lands).

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
        return _ggplot(self, mapping, _env=env)

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

    # ---- collapsing verbs --------------------------------------------

    def summarize(self, *args: pl.Expr, **kwargs: pl.Expr) -> DataFrame:
        """One row per group."""
        exprs = _kwargs_to_exprs(args, kwargs)
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        return self._df._wrap(gb.agg(exprs))

    summarise = summarize

    def count(self, sort: bool = False, name: str = "n") -> DataFrame:
        """Row count per group. Equivalent to ``summarize(n=pl.len())``."""
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        out = gb.agg(pl.len().alias(name))
        if sort:
            out = out.sort(name, descending=True)
        return self._df._wrap(out)

    # ---- windowed mutate ---------------------------------------------

    def mutate(self, *args: pl.Expr, **kwargs: pl.Expr) -> DataFrame:
        """Add columns whose values are computed within each group.

        Each expression is wrapped in ``.over(group_cols)`` — matches
        dplyr's ``group_by(g) |> mutate(x = mean(x))`` (windowed) and
        contrasts with ``summarize`` (collapsing). Result keeps the
        original row count.
        """
        exprs = _kwargs_to_exprs(args, kwargs)
        windowed = [e.over(self._by) for e in exprs]
        return self._df._wrap(
            pl.DataFrame.with_columns(self._df, windowed)
        )

    # ---- slice family per group --------------------------------------

    def slice_head(self, n: int = 1) -> DataFrame:
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        return self._df._wrap(gb.head(n))

    def slice_tail(self, n: int = 1) -> DataFrame:
        gb = pl.DataFrame.group_by(self._df, self._by, **self._kwargs)
        return self._df._wrap(gb.tail(n))

    def slice_min(
        self,
        col: str,
        n: int = 1,
        with_ties: bool = True,
    ) -> DataFrame:
        return self._slice_extreme(col, n, with_ties, descending=False)

    def slice_max(
        self,
        col: str,
        n: int = 1,
        with_ties: bool = True,
    ) -> DataFrame:
        return self._slice_extreme(col, n, with_ties, descending=True)

    def _slice_extreme(
        self,
        col: str,
        n: int,
        with_ties: bool,
        *,
        descending: bool,
    ) -> DataFrame:
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
        return self._df._wrap(out)

    def slice_sample(
        self,
        n: int | None = None,
        prop: float | None = None,
        replace: bool = False,
        seed: int | None = None,
    ) -> DataFrame:
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
        return self._df._wrap(out)

    # ---- representation ----------------------------------------------

    def __repr__(self) -> str:
        n_groups = self._df.select(pl.struct(self._by).n_unique()).item()
        return f"<GroupBy by={self._by!r} groups={n_groups} rows={self._df.height}>"


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
