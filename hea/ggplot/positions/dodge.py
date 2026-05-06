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
        if "group" not in data.columns or "x" not in data.columns:
            return data

        groups = sorted(data["group"].unique().to_list())
        n_groups = len(groups)
        if n_groups <= 1:
            return data

        if self.width is not None:
            base_width = self.width
        elif "width" in data.columns:
            base_width = float(data["width"].mean())
        else:
            base_width = 0.9

        slot_width = base_width / n_groups
        offsets = pl.DataFrame({
            "group": groups,
            "_offset": [(i - (n_groups - 1) / 2) * slot_width for i in range(n_groups)],
        })

        result = data.join(offsets, on="group", how="left")
        result = result.with_columns(
            x=pl.col("x") + pl.col("_offset"),
        ).drop("_offset")

        if "width" in result.columns:
            result = result.with_columns(width=pl.lit(slot_width))

        return result


# ggplot2's position_dodge2 differs from dodge only when boxplot-style geoms
# need uneven slot widths. For Phase 1.3 we alias to PositionDodge — refine
# in Phase 1.9c when geom_boxplot lands.
PositionDodge2 = PositionDodge


def position_dodge(*, width=None, preserve="total"):
    return PositionDodge(width=width, preserve=preserve)


def position_dodge2(*, width=None, preserve="total"):
    return PositionDodge2(width=width, preserve=preserve)
