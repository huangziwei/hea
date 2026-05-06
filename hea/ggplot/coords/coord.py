"""Base :class:`Coord` — coordinate system. Cartesian is the no-op default."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class Coord:
    is_linear: bool = True

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return data
