"""``position_nudge()`` — constant offset on x and/or y."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .position import Position


@dataclass
class PositionNudge(Position):
    x: float = 0.0
    y: float = 0.0

    def compute_layer(self, data: pl.DataFrame) -> pl.DataFrame:
        from .position import to_numeric_positions

        # Discrete x or y → integer positions first (ggplot2 does the
        # same: ``geom_text(position=position_nudge(x=0.2))`` works on a
        # categorical x by nudging in the post-transform space).
        for axis, delta in (("x", self.x), ("y", self.y)):
            if axis in data.columns and delta != 0:
                col = data[axis]
                if not col.dtype.is_numeric():
                    data = data.with_columns(to_numeric_positions(col))

        out_cols = []
        if "x" in data.columns and self.x != 0:
            out_cols.append((pl.col("x") + self.x).alias("x"))
        if "y" in data.columns and self.y != 0:
            out_cols.append((pl.col("y") + self.y).alias("y"))
        return data.with_columns(out_cols) if out_cols else data


def position_nudge(*, x=0.0, y=0.0):
    return PositionNudge(x=x, y=y)
