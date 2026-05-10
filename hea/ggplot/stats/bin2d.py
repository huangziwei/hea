"""``stat_bin_2d()`` and ``stat_binhex()`` — 2D binning for ``geom_bin2d``
and ``geom_hex``.

Both bin ``(x, y)`` into rectangular or hexagonal cells and emit one row
per cell with a ``count`` column. The default fill aesthetic is set to
``count`` so the bin colour reflects bin density via the ``fill`` scale
(matches ggplot2's ``aes(fill = after_stat(count))`` default for these
geoms).

Hex binning delegates the cell geometry to ``matplotlib.axes.Axes.hexbin``
in a hidden figure — well-tested algorithm, returns the hex polygon shape
in data coordinates so the geom can re-draw with our scale-mapped fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .stat import Stat


def _broadcast_pair(value):
    """Accept a scalar, ``None``, or 2-element sequence; return ``(vx, vy)``.

    Lets ``bins=30`` mean ``(30, 30)`` and ``binwidth=(0.1, 0.05)`` route
    distinct widths to each axis."""
    if value is None:
        return None, None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"expected scalar or 2-element sequence; got length {len(value)}"
            )
        return float(value[0]), float(value[1])
    return float(value), float(value)


@dataclass
class StatBin2d(Stat):
    bins: int | tuple[int, int] | None = None
    binwidth: float | tuple[float, float] | None = None
    drop: bool = True
    # Title for the colorbar when fill is auto-mapped to count by this stat
    # (no user fill mapping). Mirrors ggplot2's ``after_stat(count)``
    # default — the colorbar reads "count", not "fill".
    default_fill_label: str = "count"

    def compute_group(self, data, params):
        if "x" not in data.columns or "y" not in data.columns:
            return pl.DataFrame()
        x = data["x"].to_numpy().astype(float)
        y = data["y"].to_numpy().astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return pl.DataFrame()

        x_breaks = self._breaks_axis(x, axis="x")
        y_breaks = self._breaks_axis(y, axis="y")
        # numpy histogram2d returns counts in shape (n_x, n_y).
        counts, _, _ = np.histogram2d(x, y, bins=[x_breaks, y_breaks])

        x_mids = (x_breaks[:-1] + x_breaks[1:]) / 2
        y_mids = (y_breaks[:-1] + y_breaks[1:]) / 2
        x_widths = np.diff(x_breaks)
        y_heights = np.diff(y_breaks)

        ix, iy = np.indices(counts.shape)
        ix, iy = ix.ravel(), iy.ravel()
        flat = counts.ravel()
        total = float(flat.sum())
        density = flat / total if total > 0 else flat

        # ``ncount`` / ``ndensity`` — per-group normalised forms (peak=1).
        max_count = float(np.abs(flat).max()) if len(flat) else 0.0
        max_density = float(np.abs(density).max()) if len(density) else 0.0
        ncount = flat / max_count if max_count > 0 else flat.astype(float)
        ndensity = density / max_density if max_density > 0 else density

        out = pl.DataFrame({
            "x": x_mids[ix],
            "y": y_mids[iy],
            "width": x_widths[ix],
            "height": y_heights[iy],
            "count": flat.astype(float),
            "density": density.astype(float),
            "ncount": ncount.astype(float),
            "ndensity": ndensity.astype(float),
            # Default fill = count so ``geom_bin2d()`` colours bins by
            # density without the user needing ``aes(fill=after_stat(count))``.
            "fill": flat.astype(float),
        })

        if self.drop:
            out = out.filter(pl.col("count") > 0)
        return out

    def _breaks_axis(self, arr, *, axis):
        a_min, a_max = float(arr.min()), float(arr.max())
        bw_x, bw_y = _broadcast_pair(self.binwidth)
        bw = bw_x if axis == "x" else bw_y
        if bw is not None:
            # Same boundary convention as ``stat_bin``: bins centered on
            # multiples of ``binwidth`` (boundary = binwidth/2).
            boundary = bw / 2
            shift = np.floor((a_min - boundary) / bw)
            start = boundary + shift * bw
            n = max(int(np.ceil((a_max - start) / bw)), 1)
            return start + bw * np.arange(n + 1)

        if isinstance(self.bins, (tuple, list)):
            n = int(self.bins[0]) if axis == "x" else int(self.bins[1])
        elif self.bins is None:
            n = 30
        else:
            n = int(self.bins)
        return np.linspace(a_min, a_max, n + 1)


@dataclass
class StatBinhex(Stat):
    bins: int = 30
    drop: bool = True
    default_fill_label: str = "count"

    def compute_group(self, data, params):
        if "x" not in data.columns or "y" not in data.columns:
            return pl.DataFrame()
        x = data["x"].to_numpy().astype(float)
        y = data["y"].to_numpy().astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return pl.DataFrame()

        offsets, counts, hex_w, hex_h = _matplotlib_hexbin(x, y, gridsize=self.bins)

        if self.drop:
            keep = counts > 0
            offsets = offsets[keep]
            counts = counts[keep]
        if len(counts) == 0:
            return pl.DataFrame()

        total = float(counts.sum())
        density = counts / total if total > 0 else counts

        max_count = float(np.abs(counts).max()) if len(counts) else 0.0
        max_density = float(np.abs(density).max()) if len(density) else 0.0
        ncount = counts / max_count if max_count > 0 else counts.astype(float)
        ndensity = density / max_density if max_density > 0 else density

        n = len(counts)
        return pl.DataFrame({
            "x": offsets[:, 0].astype(float),
            "y": offsets[:, 1].astype(float),
            "width": np.full(n, hex_w),
            "height": np.full(n, hex_h),
            "count": counts.astype(float),
            "density": density.astype(float),
            "ncount": ncount.astype(float),
            "ndensity": ndensity.astype(float),
            "fill": counts.astype(float),
        })


def _matplotlib_hexbin(x, y, *, gridsize):
    """Compute hex bins via matplotlib's ``hexbin`` in a hidden figure.

    Returns ``(offsets, counts, hex_width, hex_height)`` — offsets are
    hex centres in data coordinates, counts are per-hex sample counts,
    and the width/height describe the (data-units) bounding box of one
    hex polygon. matplotlib's hexbin runs the standard skewed-grid
    algorithm and yields polygon shapes in data coords (matching what
    we'd want to re-render via ``PolyCollection``).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        hb = ax.hexbin(x, y, gridsize=int(gridsize), mincnt=0)
        offsets = np.array(hb.get_offsets(), copy=True)
        counts = np.array(hb.get_array(), copy=True)
        verts = hb.get_paths()[0].vertices
        hex_w = float(verts[:, 0].max() - verts[:, 0].min())
        hex_h = float(verts[:, 1].max() - verts[:, 1].min())
    finally:
        plt.close(fig)
    return offsets, counts, hex_w, hex_h


def stat_bin_2d(*, bins=None, binwidth=None, drop=True):
    return StatBin2d(bins=bins, binwidth=binwidth, drop=drop)


def stat_binhex(*, bins=30, drop=True):
    return StatBinhex(bins=bins, drop=drop)
