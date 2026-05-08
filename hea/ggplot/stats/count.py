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
        # ``prop`` — fraction of the bar's count within its layer-level
        # group (ggplot2's ``StatCount$compute_group``: ``count /
        # sum(abs(count))``). With ``group=1`` all bars share one group, so
        # ``prop`` is the global proportion; with each bar in its own group
        # (the default when no discrete aes splits) ``prop = 1``.
        count_f = pl.col("count").cast(pl.Float64)
        if "group" in out.columns:
            prop_expr = count_f / count_f.abs().sum().over("group")
        else:
            prop_expr = pl.lit(1.0)
        return out.with_columns(
            y=count_f,
            width=pl.lit(self.width),
            prop=prop_expr,
        )


def stat_count(*, width=0.9):
    return StatCount(width=width)
