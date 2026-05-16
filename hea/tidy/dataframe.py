"""The ``hea.DataFrame`` subclass — a thin ``pl.DataFrame`` lift with
tidyverse-named methods (``filter`` / ``arrange`` / ``distinct`` /
``mutate`` / ``select`` / ``rename`` / ``relocate`` / ``group_by`` /
``summarize`` / ``slice_*`` / ``count`` / ``ungroup``, plus
``pivot_longer`` / ``pivot_wider`` / ``pull`` from chapter 5).

Closed under polars operations: every method that returns a
DataFrame / LazyFrame / Series returns the corresponding hea subclass.
Native polars methods propagate via ``self._from_pydf(...)``; the
operations that bypass that route (``describe``, ``corr``, ``unstack``,
``sql``, ``match_to_schema``, plus the lazy round-trip via ``lazy()`` /
``collect()``) are explicitly re-wrapped. Series-returning methods
(``get_column``, ``__getitem__``, the ``*_horizontal`` family, …) are
wrapped via the install hooks in :mod:`hea.tidy.series`.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    from .groupby import GroupBy
    from .series import LazyFrame
    from .summary import Summary

import numpy as np
import polars as pl
from scipy import stats as _sps

from ..R import cut as _R_cut
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
)
from .basics import _Desc
from .joins import (
    _BIN_OPS,
    _CLOSEST_STRATEGY,
    _INEQ_BUILDERS,
    _JoinBy,
    _align_equi_key_types,
    _emit_natural_join_message,
)


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

    # ---- repr ---------------------------------------------------------

    def __repr__(self) -> str:
        base = super().__repr__()
        m = getattr(self, "_ts_meta", None)
        if m is None:
            return base
        return _format_ts_header(m) + "\n" + base

    def __str__(self) -> str:
        base = super().__str__()
        m = getattr(self, "_ts_meta", None)
        if m is None:
            return base
        return _format_ts_header(m) + "\n" + base

    def _repr_html_(self, **kwargs) -> str | None:
        base = super()._repr_html_(**kwargs)
        m = getattr(self, "_ts_meta", None)
        if m is None or base is None:
            return base
        return _format_ts_header_html(m) + base

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
        from ..R import _LazyFactor

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
        from .groupby import GroupBy
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

        # ts-marked frame: default x=value (single-vector default,
        # matching ``hist(Nile)``). Users targeting the time-axis line
        # plot pass aes explicitly: ``Nile.ggplot(x="time", y="value")``
        # — same explicitness ggplot2 in R requires for any chart.
        # Driven by the explicit ``_ts_meta`` flag, never by column-name
        # inference, so user-built ``DataFrame({"time": …, "value": …})``
        # is not affected.
        if getattr(self, "_ts_meta", None) is not None:
            mapped_x = ("x" in aes_kwargs) or (
                mapping is not None and "x" in mapping)
            if not mapped_x:
                aes_kwargs["x"] = "value"

        env = _frame_env(inspect.currentframe().f_back)
        return _ggplot(self, mapping, _env=env, **aes_kwargs)

    # ---- lazy frame ---------------------------------------------------

    def lazy(self) -> "LazyFrame":
        """Start a lazy query; returns a hea.LazyFrame.

        Overrides ``pl.DataFrame.lazy`` (which would return ``pl.LazyFrame``
        via ``wrap_ldf`` and lose subclass identity through the eager-via-lazy
        round-trip used by ``with_columns``/``sort``/``join``/etc.).
        """
        from .series import LazyFrame
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
        from .summary import Summary, _summary_block, _SummaryBlock
        # ts-marked frame: hea.data() and hea.R.ts() set ``_ts_meta`` on
        # frames that R would classify as ``ts``. R's ``summary(Nile)``
        # (via the ts class) summarizes only the values; we mirror that
        # by summarizing just the ``value`` column and skipping the
        # year-index ``time``. Never inferred from column names — a
        # user-built ``DataFrame({"time": …, "value": …})`` falls through
        # to the normal per-column path.
        if getattr(self, "_ts_meta", None) is not None:
            cols_to_summarize = ["value"]
        else:
            cols_to_summarize = list(self.columns)
        blocks: list[_SummaryBlock] = []
        for col in cols_to_summarize:
            blocks.append(
                _summary_block(col, self.get_column(col), maxsum=maxsum, digits=digits)
            )
        return Summary(blocks, width=width)

    # ---- time series --------------------------------------------------

    def as_ts(self, *, start: float | None = None,
              frequency: float = 1.0) -> "DataFrame":
        """Mark this frame as a time series (R's ``as.ts()``).

        The frame must already be a 2-column ``(time, value)`` shape —
        :func:`hea.R.ts` is the constructor for the from-scratch case.
        ``start`` defaults to the first row's ``time`` value; ``end`` is
        derived from the last row; ``frequency`` defaults to 1 (annual).

        Returns a new DataFrame (same data, ``_ts_meta`` set). The flag
        is not propagated by ``filter`` / ``select`` / etc. — mirrors R,
        where ``Nile[1:10]`` drops the ``ts`` class.
        """
        if list(self.columns) != ["time", "value"]:
            raise ValueError(
                f"as_ts(): frame must have columns ['time', 'value'], "
                f"got {self.columns}. Use hea.R.ts(values, start=, "
                f"frequency=) for the from-scratch case."
            )
        if start is None:
            start = float(self["time"][0])
        end = float(self["time"][-1])
        out = self._wrap(self)
        out._ts_meta = TsMeta(start=start, end=end, frequency=float(frequency))
        return out

    def drop_ts(self) -> "DataFrame":
        """Strip the ts marker — inverse of :meth:`as_ts`. Mirrors R's
        ``unclass(Nile)`` / ``as.data.frame(Nile)``.

        Returns a new DataFrame with the same columns but ``_ts_meta``
        absent, so subsequent ``hist`` / ``plot`` / ``summary`` calls
        treat it as a plain 2-column frame. No-op if the frame isn't
        ts-marked.
        """
        return self._wrap(self)


def _fmt_ts_num(v: float) -> str:
    """Format a ts metadata value like R: integers without decimal,
    floats with their natural decimal repr."""
    iv = int(v)
    return str(iv) if float(iv) == v else repr(v)


def _format_ts_header(m: "TsMeta") -> str:
    """R-style ``print.ts`` header — three lines above the data block."""
    return (
        "Time Series:\n"
        f"Start = {_fmt_ts_num(m.start)}\n"
        f"End = {_fmt_ts_num(m.end)}\n"
        f"Frequency = {_fmt_ts_num(m.frequency)}"
    )


def _format_ts_header_html(m: "TsMeta") -> str:
    return (
        "<p><b>Time Series</b><br/>"
        f"Start = {_fmt_ts_num(m.start)}<br/>"
        f"End = {_fmt_ts_num(m.end)}<br/>"
        f"Frequency = {_fmt_ts_num(m.frequency)}</p>"
    )


@dataclass(frozen=True, slots=True)
class TsMeta:
    """R-style ``tsp`` triple — (start, end, frequency) — carried as the
    ``_ts_meta`` attribute on hea DataFrames marked as a time series.

    Set by :func:`hea.R.ts` (from-scratch construction), :meth:`DataFrame.as_ts`
    (post-hoc marking), and :func:`hea.data` for the known R ``ts``
    datasets. Consulted by base-graphics plotters and
    :meth:`DataFrame.summary` to dispatch as R does for the ``ts`` class.
    Not propagated through ``_wrap``: subset / filter / select / join
    drop it, mirroring R's "indexing a ts returns a vector".
    """
    start: float
    end: float
    frequency: float


def ts(data, start: float = 1.0, frequency: float = 1.0) -> "DataFrame":
    """R: ``ts(data, start, frequency)`` — construct a time series.

    Returns a hea ``DataFrame`` with columns ``("time", "value")``,
    synthesized as ``time = start + i / frequency`` for ``i in 0..n-1``,
    and ``_ts_meta`` set so base-graphics plotters and ``summary`` route
    via R's ``ts`` dispatch.

    Parameters
    ----------
    data
        Numeric vector (1-D ndarray, list, hea/polars Series, or a
        1-column DataFrame). NA / null values are preserved (R doesn't
        drop them at ``ts`` construction either).
    start
        Time of the first observation. Defaults to 1 like R.
    frequency
        Observations per unit of time. R defaults: 1 (annual),
        12 (monthly), 4 (quarterly), 52 (weekly).

    Examples
    --------
    >>> Nile = hea.R.ts(values, start=1871, frequency=1)
    >>> hist(Nile)   # dispatches as a numeric vector via the flag
    """
    if isinstance(data, pl.DataFrame):
        if data.width != 1:
            raise ValueError(
                f"ts(): DataFrame input must have exactly one column, "
                f"got {data.columns}. Drop the time column or call "
                f"as_ts() on an existing (time, value) frame."
            )
        values_series = data[data.columns[0]]
    elif isinstance(data, pl.Series):
        values_series = data
    else:
        values_series = pl.Series("value", np.asarray(data))

    n = values_series.len()
    step = 1.0 / float(frequency)
    time_arr = np.asarray([start + i * step for i in range(n)], dtype=float)
    end = float(time_arr[-1]) if n else float(start)
    out = DataFrame({"time": time_arr, "value": values_series.rename("value")})
    out._ts_meta = TsMeta(start=float(start), end=end, frequency=float(frequency))
    return out

