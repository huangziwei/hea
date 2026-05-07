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
        counts = _count_per_bin(x, breaks, self.closed)
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
            # ggplot2's default boundary when neither boundary nor center
            # is supplied: ``binwidth / 2``. Equivalent to centering bins
            # on multiples of binwidth (e.g. width=200 → bin centers at
            # …, 2800, 3000, 3200, …).
            if self.boundary is not None:
                boundary = float(self.boundary)
            elif self.center is not None:
                boundary = float(self.center) - binwidth / 2
            else:
                boundary = binwidth / 2
            shift = np.floor((x_min - boundary) / binwidth)
            start = boundary + shift * binwidth
            n_bins = int(np.ceil((x_max - start) / binwidth))
            # Last edge has to STRICTLY exceed x_max under right-closed
            # (where x == break is in the bin to the LEFT).
            return start + binwidth * np.arange(n_bins + 1)

        n_bins = self.bins if self.bins is not None else 30
        return np.linspace(x_min, x_max, n_bins + 1)


def _count_per_bin(x, breaks, closed: str) -> np.ndarray:
    """Bin ``x`` into edges ``breaks`` with R/ggplot2 semantics.

    ``closed='right'`` (ggplot2 default): each bin is ``(low, high]`` —
    EXCEPT the leftmost bin which is fully closed ``[low, high]`` so
    the data minimum lands in a bin (matches R's ``cut(..., right=TRUE,
    include.lowest=TRUE)``).

    ``closed='left'``: each bin is ``[low, high)`` except the rightmost
    which is fully closed (mirror image; matches numpy's default).
    """
    n_bins = len(breaks) - 1
    if n_bins <= 0:
        return np.zeros(0, dtype=int)

    if closed == "right":
        # searchsorted(breaks[1:-1], x, side='left'):
        #   x <= breaks[1] → 0 (bin 0)
        #   breaks[1] < x <= breaks[2] → 1 (bin 1)
        #   ...
        # The leftmost bin includes x == breaks[0] AND x == breaks[1]
        # (the latter via side='left' on breaks[1] giving 0).
        if n_bins == 1:
            in_only = (x >= breaks[0]) & (x <= breaks[1])
            return np.array([int(in_only.sum())])
        idx = np.searchsorted(breaks[1:-1], x, side="left")
    else:  # "left"
        # Mirror: each bin is [low, high) except rightmost.
        if n_bins == 1:
            in_only = (x >= breaks[0]) & (x <= breaks[1])
            return np.array([int(in_only.sum())])
        idx = np.searchsorted(breaks[1:-1], x, side="right")
    # Drop x outside [breaks[0], breaks[-1]] (defensive — _compute_breaks
    # guarantees the data fits, but stay robust).
    in_range = (x >= breaks[0]) & (x <= breaks[-1])
    idx = idx[in_range]
    counts = np.bincount(idx, minlength=n_bins)[:n_bins]
    return counts.astype(int)


def stat_bin(*, bins=None, binwidth=None, boundary=None, center=None, closed="right"):
    return StatBin(bins=bins, binwidth=binwidth, boundary=boundary,
                   center=center, closed=closed)
