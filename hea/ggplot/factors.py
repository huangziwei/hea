"""Factor (Enum) reordering helpers — the ggplot2 / forcats idiom.

Factors in polars are :class:`polars.Enum` columns: their level *order*
drives the order in which categorical aesthetics (x ticks, legend keys,
boxplot groups, …) appear. ``fct_reorder`` and friends produce a new
Enum-typed Series whose level order has been computed from the data —
without mutating the source DataFrame. Use them inside :func:`aes`:

    df.ggplot(x=fct_reorder("class", "hwy", "median"), y="hwy").geom_boxplot()

Each ``fct_*`` function returns a *callable* that the build pipeline
evaluates against the layer data — same machinery that handles
``aes(x=lambda d: ...)``. The callable carries ``__hea_label__`` so
the axis label and legend title still resolve to the source column
name (``"class"`` here), not ``"<function>"``.
"""

from __future__ import annotations

from typing import Callable

import polars as pl


def _label_callable(fn: Callable, label: str) -> Callable:
    """Tag ``fn`` so the renderer / aes_source can pull a label from it."""
    fn.__hea_label__ = label
    fn.__hea_aes_source__ = label
    return fn


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


def fct_rev(col: str) -> Callable:
    """Reverse the level order of ``col``.

    For Enum columns, reverses the existing level order. For plain
    string columns, sorts alphabetically descending — symmetric with
    the default ascending order ScaleOrdinal would otherwise pick.
    """
    def rev(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        if isinstance(s.dtype, pl.Enum):
            levels = list(s.dtype.categories)[::-1]
        else:
            levels = sorted(
                {str(v) for v in s.drop_nulls().to_list()},
                reverse=True,
            )
        return s.cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(rev, col)


def fct_relevel(col: str, *levels: str) -> Callable:
    """Move ``levels`` to the front of ``col``'s factor levels.

    Levels not listed keep their existing relative order behind the
    promoted ones (Enum order if the column is an Enum, alphabetical
    otherwise). Mirrors R's ``forcats::fct_relevel``.
    """
    promoted = [str(lvl) for lvl in levels]

    def relevel(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        if isinstance(s.dtype, pl.Enum):
            existing = list(s.dtype.categories)
        else:
            existing = sorted({str(v) for v in s.drop_nulls().to_list()})
        promoted_set = set(promoted)
        rest = [lvl for lvl in existing if lvl not in promoted_set]
        ordered = promoted + rest
        return s.cast(pl.Utf8).cast(pl.Enum(ordered))

    return _label_callable(relevel, col)


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
