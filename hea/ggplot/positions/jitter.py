"""``position_jitter()`` — random ±width offset on x and y.

Default width/height: ``0.4 * resolution(x)``, matching ggplot2.
``resolution(x)`` is the smallest non-zero gap between unique values
(treated as ``1`` for single-valued data). The factor of 0.4 keeps
neighbouring tick groups visually separated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .position import Position


@dataclass
class PositionJitter(Position):
    width: float | None = None
    height: float | None = None
    seed: int | None = None

    def compute_layer(self, data: pl.DataFrame) -> pl.DataFrame:
        if "x" not in data.columns and "y" not in data.columns:
            return data

        rng = np.random.default_rng(self.seed)
        n = len(data)
        out_cols = []

        if "x" in data.columns:
            x = data["x"].to_numpy()
            w = self.width if self.width is not None else 0.4 * _resolution(x)
            out_cols.append(pl.Series("x", x + (rng.random(n) - 0.5) * 2 * w))

        if "y" in data.columns:
            y = data["y"].to_numpy()
            h = self.height if self.height is not None else 0.4 * _resolution(y)
            out_cols.append(pl.Series("y", y + (rng.random(n) - 0.5) * 2 * h))

        return data.with_columns(out_cols) if out_cols else data


def _resolution(x) -> float:
    """ggplot2 ``resolution(x, zero=FALSE)`` — smallest gap between unique
    values. Falls back to 1 if everything's the same."""
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = np.unique(arr)
    if len(arr) < 2:
        return 1.0
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0]
    return float(diffs.min()) if len(diffs) > 0 else 1.0


def position_jitter(*, width=None, height=None, seed=None):
    return PositionJitter(width=width, height=height, seed=seed)
