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


@dataclass
class StatDensity(Stat):
    bw: object = "nrd0"
    n: int = 512

    default_y_label: str = "density"

    def compute_group(self, data, params):
        x = data["x"].to_numpy().astype(float)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return pl.DataFrame({"x": [], "y": [], "density": [], "count": []})

        bw = self._bandwidth(x)
        x_min, x_max = float(x.min()), float(x.max())
        grid = np.linspace(x_min - 3 * bw, x_max + 3 * bw, self.n)

        # scipy gaussian_kde takes bw_method as a multiplier on x.std() — pass our
        # absolute bandwidth as bw / x.std to recover the bandwidth we computed.
        sigma_x = x.std(ddof=1)
        kde = gaussian_kde(x, bw_method=(bw / sigma_x) if sigma_x > 0 else bw)
        density = kde(grid)

        return pl.DataFrame({
            "x": grid,
            "y": density,
            "density": density,
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


def stat_density(*, bw="nrd0", n=512):
    return StatDensity(bw=bw, n=n)
