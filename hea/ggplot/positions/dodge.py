"""``position_dodge()`` / ``position_dodge2()`` — side-by-side groups.

When several groups share an x position (typical for ``geom_bar`` with
``aes(fill=group)``), dodging shifts each group horizontally so the bars
sit alongside instead of overlapping. ggplot2's algorithm reads the
``group`` aesthetic; if the user didn't supply one, dodge is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .position import Position


@dataclass
class PositionDodge(Position):
    width: float | None = None
    preserve: str = "total"  # ``"total"`` or ``"single"``; phase 1.3 ignores

    def compute_layer(self, data: pl.DataFrame) -> pl.DataFrame:
        """Mirrors ggplot2's ``collide()`` + ``pos_dodge()``: groups are
        dodged only when they OVERLAP at the same x. A boxplot with
        ``group=x`` (each x has exactly one group) gets no dodge —
        ggplot2 leaves x at 3, 4, 5 not 2.75, 4.0, 5.25."""
        if "group" not in data.columns or "x" not in data.columns:
            return data

        if self.width is not None:
            base_width = self.width
        elif "width" in data.columns:
            base_width = float(data["width"].mean())
        else:
            base_width = 0.9

        # Per-x group rank + count. If only one group at a given x,
        # offset stays 0 — no dodging.
        unique_xg = (
            data.select(["x", "group"])
            .unique(maintain_order=False)
            .sort(["x", "group"])
        )
        unique_xg = unique_xg.with_columns(
            _n_at_x=pl.col("group").count().over("x"),
            _rank_at_x=(pl.col("group").rank("dense").over("x") - 1).cast(pl.Float64),
        )
        # Offset = (rank - (n-1)/2) * slot_width; slot_width = base_width/n.
        unique_xg = unique_xg.with_columns(
            _offset=pl.when(pl.col("_n_at_x") > 1)
                      .then(
                          (pl.col("_rank_at_x") - (pl.col("_n_at_x") - 1) / 2)
                          * (base_width / pl.col("_n_at_x"))
                      )
                      .otherwise(pl.lit(0.0)),
            _slot_width=pl.when(pl.col("_n_at_x") > 1)
                          .then(base_width / pl.col("_n_at_x"))
                          .otherwise(pl.lit(None, dtype=pl.Float64)),
        ).select(["x", "group", "_offset", "_slot_width"])

        result = data.join(unique_xg, on=["x", "group"], how="left")
        result = result.with_columns(x=pl.col("x") + pl.col("_offset"))
        if "width" in result.columns:
            result = result.with_columns(
                width=pl.when(pl.col("_slot_width").is_not_null())
                        .then(pl.col("_slot_width"))
                        .otherwise(pl.col("width")),
            )
        return result.drop("_offset", "_slot_width")


# ggplot2's position_dodge2 differs from dodge only when boxplot-style geoms
# need uneven slot widths. For Phase 1.3 we alias to PositionDodge — refine
# in Phase 1.9c when geom_boxplot lands.
PositionDodge2 = PositionDodge


def position_dodge(*, width=None, preserve="total"):
    return PositionDodge(width=width, preserve=preserve)


def position_dodge2(*, width=None, preserve="total"):
    return PositionDodge2(width=width, preserve=preserve)
