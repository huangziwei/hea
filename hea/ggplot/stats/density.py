"""``stat_density()`` — Gaussian kernel density estimate over x.

Bandwidth defaults to R's ``bw.nrd0`` (Silverman's rule of thumb). The
``bw=`` argument accepts a scalar or the string ``"nrd0"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from .stat import Stat


_PANEL_X_RANGE_KEY = "_stat_density_panel_x_range"


@dataclass
class StatDensity(Stat):
    # Mirrors ggplot2's ``stat_density()`` parameter defaults
    # (R/stat-density.R). ``adjust`` multiplies ``bw`` and is the most
    # common knob users reach for. ``kernel`` is hardcoded to gaussian
    # (scipy's ``gaussian_kde``); ``bounds`` is not yet honoured.
    #
    # ``trim`` matches R: ``False`` (default) evaluates each group's
    # density on the panel-wide x range (so all curves share the same
    # grid and per-group tails extend across the panel); ``True`` clips
    # each group to its own ``[min(x), max(x)]``. ggplot2 reads this
    # range from ``scales$x$dimension()``; hea reads it from the panel
    # data (``compute_panel``) since stats run before scales are trained.
    # Identical to R when this is the sole layer driving the x scale and
    # no manual ``xlim()`` is set.
    bw: object = "nrd0"
    adjust: float = 1.0
    n: int = 512
    trim: bool = False

    default_y_label: str = "density"

    def compute_panel(self, data, params):
        # Capture the panel-wide x range before the base class splits
        # ``data`` per group. With ``trim = False`` every group's KDE
        # is evaluated on this shared grid, so a low-mass group's curve
        # extends across the panel — matching ggplot2's behaviour where
        # ``scales$x$dimension()`` is the panel x range.
        if "x" in data.columns and len(data):
            xs = data["x"].to_numpy().astype(float)
            xs = xs[~np.isnan(xs)]
            if xs.size:
                params = {
                    **params,
                    _PANEL_X_RANGE_KEY: (float(xs.min()), float(xs.max())),
                }
        return super().compute_panel(data, params)

    def compute_group(self, data, params):
        x = data["x"].to_numpy().astype(float)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return pl.DataFrame({"x": [], "y": [], "density": [], "count": []})

        bw = self._bandwidth(x) * float(self.adjust)
        if self.trim:
            x_min, x_max = float(x.min()), float(x.max())
        else:
            panel_range = params.get(_PANEL_X_RANGE_KEY)
            if panel_range is None:
                x_min, x_max = float(x.min()), float(x.max())
            else:
                x_min, x_max = panel_range
        grid = np.linspace(x_min, x_max, self.n)

        # scipy gaussian_kde takes bw_method as a multiplier on x.std() — pass our
        # absolute bandwidth as bw / x.std to recover the bandwidth we computed.
        sigma_x = x.std(ddof=1)
        kde = gaussian_kde(x, bw_method=(bw / sigma_x) if sigma_x > 0 else bw)
        density = kde(grid)

        max_d = float(density.max()) if density.size else 0.0
        ndensity = density / max_d if max_d > 0 else density

        return pl.DataFrame({
            "x": grid,
            "y": density,
            "density": density,
            "ndensity": ndensity,
            "count": density * len(x),
        })

    def _bandwidth(self, x: np.ndarray) -> float:
        if isinstance(self.bw, str):
            return self._nrd0(x)
        return float(self.bw)

    @staticmethod
    def _nrd0(x: np.ndarray) -> float:
        """R's ``bw.nrd0``: ``0.9 · min(σ, IQR/1.34) · n^(-1/5)``."""
        n = len(x)
        sigma = x.std(ddof=1)
        q1, q3 = np.percentile(x, [25, 75])
        iqr_scaled = (q3 - q1) / 1.34
        if iqr_scaled > 0 and sigma > 0:
            scale = min(sigma, iqr_scaled)
        else:
            scale = sigma if sigma > 0 else 1.0
        return max(0.9 * scale * n ** (-1 / 5), 1e-10)


def stat_density(*, bw="nrd0", adjust=1.0, n=512, trim=False):
    return StatDensity(bw=bw, adjust=adjust, n=n, trim=trim)
