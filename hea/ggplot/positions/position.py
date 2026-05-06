"""Base :class:`Position` — adjusts geom positions (jitter, dodge, stack, …)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class Position:
    def compute_layer(self, data: pl.DataFrame) -> pl.DataFrame:
        return data
