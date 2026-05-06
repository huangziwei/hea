"""``stat_bin()`` — histogram binning of a continuous x.

Phase 1.2 form: simple equal-width bins via ``numpy.histogram``. Real
Wilkinson break-finding (parity with ggplot2's ``bin_breaks``) is a
later polish; this version handles the canonical "30 equal-width bins"
default and explicit ``binwidth`` / ``bins`` overrides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .stat import Stat


@dataclass
class StatBin(Stat):
    bins: int | None = None
    binwidth: float | None = None
    boundary: float | None = None
    center: float | None = None
    closed: str = "right"

    default_y_label: str = "count"

    def compute_group(self, data, params):
        x = data["x"].to_numpy().astype(float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return pl.DataFrame({
                "x": [], "y": [], "width": [], "count": [], "density": [],
            })

        breaks = self._compute_breaks(x)
        counts, _ = np.histogram(x, bins=breaks)
        mids = (breaks[:-1] + breaks[1:]) / 2
        widths = np.diff(breaks)
        total = counts.sum()
        densities = counts / (total * widths) if total > 0 else counts.astype(float)

        return pl.DataFrame({
            "x": mids,
            "y": counts.astype(float),
            "width": widths,
            "count": counts.astype(float),
            "density": densities,
        })

    def _compute_breaks(self, x):
        x_min, x_max = float(x.min()), float(x.max())

        if self.binwidth is not None:
            binwidth = float(self.binwidth)
            boundary = self.boundary if self.boundary is not None else x_min
            start = boundary - np.ceil((boundary - x_min) / binwidth) * binwidth
            n_bins = int(np.ceil((x_max - start) / binwidth)) + 1
            return start + binwidth * np.arange(n_bins + 1)

        n_bins = self.bins if self.bins is not None else 30
        return np.linspace(x_min, x_max, n_bins + 1)


def stat_bin(*, bins=None, binwidth=None, boundary=None, center=None, closed="right"):
    return StatBin(bins=bins, binwidth=binwidth, boundary=boundary,
                   center=center, closed=closed)
