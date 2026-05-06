"""``position_stack()`` and ``position_fill()`` — vertical stacking by group.

Both compute ``ymin``/``ymax`` columns by walking each x-group and
cumulative-summing y. ``position_stack`` stacks raw y; ``position_fill``
normalises so stacks reach 1.0 per x.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .position import Position


def _sort_within_x(data: pl.DataFrame, reverse: bool) -> tuple[pl.DataFrame, list, list]:
    sort_cols = ["x"]
    sort_desc = [False]
    if "group" in data.columns:
        sort_cols.append("group")
        sort_desc.append(reverse)
    return data.sort(sort_cols, descending=sort_desc), sort_cols, sort_desc


@dataclass
class PositionStack(Position):
    vjust: float = 1.0  # placeholder; only used by some downstream geoms
    reverse: bool = False

    def compute_layer(self, data: pl.DataFrame) -> pl.DataFrame:
        if "x" not in data.columns or "y" not in data.columns:
            return data
        sorted_data, _, _ = _sort_within_x(data, self.reverse)
        return sorted_data.with_columns(
            ymax=pl.col("y").cum_sum().over("x"),
        ).with_columns(
            ymin=pl.col("ymax") - pl.col("y"),
        )


@dataclass
class PositionFill(Position):
    reverse: bool = False

    def compute_layer(self, data: pl.DataFrame) -> pl.DataFrame:
        if "x" not in data.columns or "y" not in data.columns:
            return data
        sorted_data, _, _ = _sort_within_x(data, self.reverse)
        return sorted_data.with_columns(
            _ymax_raw=pl.col("y").cum_sum().over("x"),
            _total=pl.col("y").sum().over("x"),
        ).with_columns(
            ymax=pl.col("_ymax_raw") / pl.col("_total"),
            ymin=(pl.col("_ymax_raw") - pl.col("y")) / pl.col("_total"),
            y=pl.col("y") / pl.col("_total"),
        ).drop("_ymax_raw", "_total")


def position_stack(*, vjust=1.0, reverse=False):
    return PositionStack(vjust=vjust, reverse=reverse)


def position_fill(*, reverse=False):
    return PositionFill(reverse=reverse)
