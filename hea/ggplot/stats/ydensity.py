"""``stat_ydensity()`` — kernel density per group, with violinwidth normalisation.

For each (x, group) tuple, fits a Gaussian KDE of y and emits one row per
grid point with columns ``y`` (eval point), ``density`` (raw density),
``violinwidth`` (density divided by max within the group, in [0, 1]).
``geom_violin`` draws polygons of width ``violinwidth · 0.4`` on each side
of the group's x position.

Bandwidth uses R's ``bw.nrd0`` by default — same as :class:`StatDensity`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from .density import StatDensity
from .stat import Stat


@dataclass
class StatYdensity(Stat):
    bw: object = "nrd0"
    n: int = 512
    scale: str = "area"  # ggplot2 also has "count" and "width"; we ignore for now

    def compute_panel(self, data, params):
        groupby_cols = ["x"]
        for aes in ("group", "fill", "colour"):
            if aes in data.columns and aes not in groupby_cols:
                groupby_cols.append(aes)

        chunks = []
        for keys, sub in data.group_by(groupby_cols, maintain_order=True):
            chunk = self._chunk(sub, keys, groupby_cols)
            if chunk is not None:
                chunks.append(chunk)
        if not chunks:
            return pl.DataFrame()
        return pl.concat(chunks)

    def _chunk(self, sub, keys, groupby_cols):
        y = sub["y"].to_numpy().astype(float)
        y = y[~np.isnan(y)]
        if len(y) < 2:
            return None

        bw = self.bw if not isinstance(self.bw, str) else StatDensity._nrd0(y)
        sigma_y = y.std(ddof=1)
        kde = gaussian_kde(y, bw_method=(bw / sigma_y) if sigma_y > 0 else bw)
        y_min, y_max = float(y.min()), float(y.max())
        grid = np.linspace(y_min - 3 * bw, y_max + 3 * bw, self.n)
        density = kde(grid)
        max_d = density.max() if density.max() > 0 else 1.0
        violinwidth = density / max_d

        n = len(grid)
        cols: dict = {col: [keys[i]] * n for i, col in enumerate(groupby_cols)}
        cols.update({
            "y": grid,
            "density": density,
            "violinwidth": violinwidth,
        })
        return pl.DataFrame(cols)


def stat_ydensity(*, bw="nrd0", n=512, scale="area"):
    return StatYdensity(bw=bw, n=n, scale=scale)
