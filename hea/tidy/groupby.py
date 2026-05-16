"""The :class:`GroupBy` wrapper returned by :meth:`DataFrame.group_by`.

Lazy by design: holds a reference to the parent frame plus the grouping
columns; doesn't materialize per-group state. Each verb dispatches to
the appropriate polars idiom (``agg`` for summarize, ``.over(...)`` for
windowed mutate, ``group_by(...).head/tail`` for slice, etc.).
"""
from __future__ import annotations

from typing import Any

import polars as pl

from ._shared import (
    _apply_groups,
    _check_groups,
    _kwargs_to_exprs,
    _resolve_lazy_factors,
    _split_arrange,
)
from .dataframe import DataFrame


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


