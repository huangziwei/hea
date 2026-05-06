"""``stat_count()`` — counts per discrete x value (used by ``geom_bar``)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .stat import Stat


@dataclass
class StatCount(Stat):
    width: float = 0.9
    default_y_label: str = "count"

    def compute_panel(self, data, params):
        if "group" in data.columns:
            out = []
            for _, sub in data.group_by("group", maintain_order=True):
                out.append(self.compute_group(sub, params))
            return pl.concat(out) if out else self.compute_group(data, params)
        return self.compute_group(data, params)

    def compute_group(self, data, params):
        out = data.group_by("x", maintain_order=True).agg(
            pl.len().alias("count"),
        ).sort("x")
        return out.with_columns(
            y=pl.col("count").cast(pl.Float64),
            width=pl.lit(self.width),
        )


def stat_count(*, width=0.9):
    return StatCount(width=width)
