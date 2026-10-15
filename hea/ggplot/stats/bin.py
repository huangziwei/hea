"""``stat_bin()`` — histogram binning of a continuous x.

Simple equal-width bins via ``numpy.histogram``. Real Wilkinson
break-finding (parity with ggplot2's ``bin_breaks``) is a later polish;
this version handles the canonical "30 equal-width bins" default and
explicit ``binwidth`` / ``bins`` overrides.
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
    pad: bool = False

    default_y_label: str = "count"

    def compute_group(self, data, params):
        x = data["x"].to_numpy().astype(float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return pl.DataFrame(
                {
                    "x": [],
                    "y": [],
                    "width": [],
                    "count": [],
                    "density": [],
                }
            )

        breaks = self._compute_breaks(x)
        counts = _count_per_bin(x, breaks, self.closed)
        mids = (breaks[:-1] + breaks[1:]) / 2
        widths = np.diff(breaks)

        if self.pad:
            mids = np.concatenate(
                ([mids[0] - widths[0]], mids, [mids[-1] + widths[-1]])
            )
            widths = np.concatenate(([widths[0]], widths, [widths[-1]]))
            counts = np.concatenate(([0], counts, [0]))

        total = counts.sum()
        densities = counts / (total * widths) if total > 0 else counts.astype(float)

        max_count = float(np.abs(counts).max()) if len(counts) else 0.0
        max_density = float(np.abs(densities).max()) if len(densities) else 0.0
        ncount = counts / max_count if max_count > 0 else counts.astype(float)
        ndensity = densities / max_density if max_density > 0 else densities

        return pl.DataFrame(
            {
                "x": mids,
                "y": counts.astype(float),
                "width": widths,
                "count": counts.astype(float),
                "density": densities,
                "ncount": ncount.astype(float),
                "ndensity": ndensity.astype(float),
            }
        )

    def _compute_breaks(self, x):
        x_min, x_max = float(x.min()), float(x.max())

        if self.binwidth is not None:
            binwidth = float(self.binwidth)
            if self.boundary is not None:
                boundary = float(self.boundary)
            elif self.center is not None:
                boundary = float(self.center) - binwidth / 2
            else:
                boundary = binwidth / 2
            shift = np.floor((x_min - boundary) / binwidth)
            start = boundary + shift * binwidth
            n_bins = int(np.ceil((x_max - start) / binwidth))
            return start + binwidth * np.arange(n_bins + 1)

        n_bins = self.bins if self.bins is not None else 30
        return np.linspace(x_min, x_max, n_bins + 1)


def _count_per_bin(x, breaks, closed: str) -> np.ndarray:
    """Bin ``x`` into edges ``breaks`` with R/ggplot2 semantics."""
    n_bins = len(breaks) - 1
    if n_bins <= 0:
        return np.zeros(0, dtype=int)

    if closed == "right":
        if n_bins == 1:
            in_only = (x >= breaks[0]) & (x <= breaks[1])
            return np.array([int(in_only.sum())])
        idx = np.searchsorted(breaks[1:-1], x, side="left")
    else:  # "left"
        if n_bins == 1:
            in_only = (x >= breaks[0]) & (x <= breaks[1])
            return np.array([int(in_only.sum())])
        idx = np.searchsorted(breaks[1:-1], x, side="right")
    in_range = (x >= breaks[0]) & (x <= breaks[-1])
    idx = idx[in_range]
    counts = np.bincount(idx, minlength=n_bins)[:n_bins]
    return counts.astype(int)


def stat_bin(
    *, bins=None, binwidth=None, boundary=None, center=None, closed="right", pad=False
):
    return StatBin(
        bins=bins,
        binwidth=binwidth,
        boundary=boundary,
        center=center,
        closed=closed,
        pad=pad,
    )
