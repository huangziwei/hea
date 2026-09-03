"""``stat_density_ridges()`` — joint-bandwidth KDE per group on a shared x-grid.

Mirrors ggridges' ``stat_density_ridges()`` (R/stats.R). One bandwidth
per panel (the mean of each group's ``bw.nrd0``); each group's KDE is
then evaluated on the same grid spanning ``[min(x) - 3·bw, max(x) +
3·bw]``. The y aesthetic carries the per-group baseline through to the
geom unchanged — :class:`GeomDensityRidges` later turns ``y`` and
``height`` into the vertically-offset polygon.

Output columns: ``x`` (grid), ``y`` (carry-through baseline), ``height``
(= density; the geom's required aes), ``density``, ``ndensity``
(per-group density / max), ``count``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from .density import StatDensity
from .stat import _GROUPING_AES, Stat


@dataclass
class StatDensityRidges(Stat):
    bandwidth: object | None = None
    n: int = 512
    from_: float | None = None
    to: float | None = None

    def compute_panel(self, data, params):
        if "x" not in data.columns or len(data) == 0:
            return pl.DataFrame()

        group_keys = self._group_keys(data)

        if self.bandwidth is None:
            bws = []
            if group_keys:
                for _, sub in data.group_by(group_keys, maintain_order=True):
                    xs = _to_clean_floats(sub["x"])
                    if len(xs) > 1:
                        bws.append(StatDensity._nrd0(xs))
            else:
                xs = _to_clean_floats(data["x"])
                if len(xs) > 1:
                    bws.append(StatDensity._nrd0(xs))
            bw = float(np.mean(bws)) if bws else 1.0
        else:
            bw = float(self.bandwidth)

        xs_all = _to_clean_floats(data["x"])
        if xs_all.size == 0:
            return pl.DataFrame()
        from_x = (
            float(self.from_)
            if self.from_ is not None
            else float(xs_all.min()) - 3 * bw
        )
        to_x = float(self.to) if self.to is not None else float(xs_all.max()) + 3 * bw
        grid = np.linspace(from_x, to_x, self.n)

        preserve = []
        if "y" in data.columns:
            preserve.append("y")
        for col in _GROUPING_AES:
            if col in data.columns and col not in preserve:
                preserve.append(col)

        chunks = []
        if group_keys:
            for _, sub in data.group_by(group_keys, maintain_order=True):
                chunk = self._compute_one(sub, grid, bw, preserve)
                if chunk is not None:
                    chunks.append(chunk)
        else:
            chunk = self._compute_one(data, grid, bw, preserve)
            if chunk is not None:
                chunks.append(chunk)
        if not chunks:
            return pl.DataFrame()
        return pl.concat(chunks, how="diagonal_relaxed")

    def _group_keys(self, data: pl.DataFrame) -> list[str]:
        """Decide which columns to group by inside :meth:`compute_panel`."""
        keys: list[str] = []
        if "group" in data.columns and data["group"].n_unique() > 1:
            keys.append("group")
        if "y" in data.columns:
            y = data["y"]
            if y.dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean) and (
                "group" not in keys or y.n_unique() > data["group"].n_unique()
            ):
                keys.append("y")
        return keys

    def _compute_one(self, sub, grid, bw, preserve):
        x = _to_clean_floats(sub["x"])
        if len(x) < 3:
            return None
        sigma = x.std(ddof=1)
        kde = gaussian_kde(x, bw_method=(bw / sigma) if sigma > 0 else bw)
        density = kde(grid)
        max_d = float(density.max()) if density.size else 0.0
        ndensity = density / max_d if max_d > 0 else density

        cols: dict = {
            "x": grid,
            "density": density,
            "ndensity": ndensity,
            "height": density,  # default mapping in ggridges: height = after_stat(density)
            "count": density * len(x),
        }
        for col in preserve:
            cols[col] = pl.Series([sub[col][0]] * len(grid), dtype=sub[col].dtype)
        return pl.DataFrame(cols)


def _to_clean_floats(s: pl.Series) -> np.ndarray:
    arr = s.cast(pl.Float64, strict=False).to_numpy()
    return arr[~np.isnan(arr)]


def stat_density_ridges(*, bandwidth=None, n=512, from_=None, to=None):
    return StatDensityRidges(bandwidth=bandwidth, n=n, from_=from_, to=to)
