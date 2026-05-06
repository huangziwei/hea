"""``stat_count()`` — counts per discrete x value (used by ``geom_bar``).

Preserves ``group``, ``fill``, and ``colour`` aesthetics so downstream
positions (``position_dodge`` / ``stack`` / ``fill``) can split bars per
group. Each output row corresponds to a unique combination of ``x`` and
those aesthetics.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .stat import Stat


@dataclass
class StatCount(Stat):
    width: float = 0.9
    default_y_label: str = "count"

    def compute_panel(self, data, params):
        groupby_cols = ["x"]
        for aes in ("group", "fill", "colour"):
            if aes in data.columns:
                groupby_cols.append(aes)

        out = data.group_by(groupby_cols, maintain_order=True).agg(
            pl.len().alias("count"),
        ).sort(groupby_cols)
        return out.with_columns(
            y=pl.col("count").cast(pl.Float64),
            width=pl.lit(self.width),
        )


def stat_count(*, width=0.9):
    return StatCount(width=width)
