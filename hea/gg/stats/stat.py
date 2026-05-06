"""Base :class:`Stat` — transforms layer data before geoms see it.

Default flow: ``compute_layer`` calls ``compute_panel`` per panel,
which calls ``compute_group`` per ``group`` value. Subclasses override
the smallest level they need (most need only ``compute_group``).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class Stat:
    def compute_layer(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        return self.compute_panel(data, params)

    def compute_panel(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        if "group" not in data.columns:
            return self.compute_group(data, params)
        out = []
        for _, sub in data.group_by("group", maintain_order=True):
            out.append(self.compute_group(sub, params))
        return pl.concat(out) if out else data

    def compute_group(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        return data
