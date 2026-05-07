"""``geom_rect``, ``geom_tile``, ``geom_raster`` — rectangular geoms.

* :class:`GeomRect` reads ``xmin``/``xmax``/``ymin``/``ymax`` directly.
* :class:`GeomTile` reads ``x``/``y`` plus optional ``width``/``height``
  (default 1 each), computes the four edges, and delegates to ``GeomRect``.
  Used for heatmap-style grids when cell sizes vary.
* :class:`GeomRaster` is the regular-grid fast path: when ``x``/``y`` form
  a uniformly-spaced grid (constant width and height across rows), one
  ``ax.imshow`` call replaces N rectangles. Falls back to tile rendering
  if irregular.

Edge / fill behaviour mirrors :class:`GeomBar`: missing ``colour`` →
no edge; ``fill`` is per-row when distinct, scalar otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomRect(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": None,
        "fill": "grey35",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("xmin", "xmax", "ymin", "ymax")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return

        xmin = data["xmin"].to_numpy()
        xmax = data["xmax"].to_numpy()
        ymin = data["ymin"].to_numpy()
        ymax = data["ymax"].to_numpy()
        widths = xmax - xmin
        heights = ymax - ymin

        fills = _per_row_color(data, "fill", "grey35")
        edges = _per_row_color(data, "colour", "none", missing_value="none")
        alpha = float(_scalar(data, "alpha", default=1.0))
        linewidth = float(_scalar(data, "size", default=0.5)) * _PT_PER_MM

        patches = [
            Rectangle((x0, y0), w, h)
            for x0, y0, w, h in zip(xmin, ymin, widths, heights)
        ]
        coll = PatchCollection(
            patches,
            facecolors=fills,
            edgecolors=edges,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(coll)
        # PatchCollection doesn't auto-update the axes data limits; do it
        # explicitly so autoscale picks up rect bounds.
        ax.update_datalim(
            [(float(xmin.min()), float(ymin.min())),
             (float(xmax.max()), float(ymax.max()))]
        )
        ax.autoscale_view()


@dataclass
class GeomTile(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": None,
        "fill": "grey20",
        "size": 0.1,
        "linetype": "solid",
        "alpha": 1.0,
        "width": 1.0,
        "height": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        rect_data = _tile_to_rect(data)
        GeomRect().draw_panel(rect_data, ax)


@dataclass
class GeomRaster(Geom):
    """Regular-grid heatmap via ``ax.imshow``.

    Requires the ``(x, y)`` cells to form a uniformly-spaced rectangular
    grid (constant cell width and height). When that holds, one ``imshow``
    call replaces what would be N ``Rectangle`` patches — substantially
    faster for big grids. If the grid isn't regular, falls back to
    :class:`GeomTile`.
    """

    default_aes: dict = field(default_factory=lambda: {
        "fill": "grey20",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return

        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        xs = np.unique(x)
        ys = np.unique(y)

        # Regular-grid check: uniform spacing on both axes and exactly one
        # row per (x, y) cell. If it fails, defer to GeomTile.
        if (
            len(x) != len(xs) * len(ys)
            or (len(xs) > 1 and not np.allclose(np.diff(xs), np.diff(xs)[0]))
            or (len(ys) > 1 and not np.allclose(np.diff(ys), np.diff(ys)[0]))
        ):
            GeomTile().draw_panel(data, ax)
            return

        # Build a 2D fill grid indexed (y, x) — matplotlib's imshow expects
        # row-major (y first, x second). Convert hex strings to RGBA.
        fills = data["fill"].to_list()
        rgba = np.array([_color_to_rgba(c, "grey20") for c in fills])
        # Map each row's (x_i, y_i) to its position in the regular grid.
        x_idx = np.searchsorted(xs, x)
        y_idx = np.searchsorted(ys, y)
        grid = np.zeros((len(ys), len(xs), 4))
        grid[y_idx, x_idx] = rgba

        dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
        dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
        # imshow's `extent` is (left, right, bottom, top). origin="lower"
        # aligns the (0, 0) cell with the bottom-left corner.
        extent = (xs[0] - dx / 2, xs[-1] + dx / 2,
                  ys[0] - dy / 2, ys[-1] + dy / 2)
        alpha = float(_scalar(data, "alpha", default=1.0))
        ax.imshow(
            grid, extent=extent, origin="lower",
            alpha=alpha, aspect="auto", interpolation="nearest",
        )


def _tile_to_rect(data):
    """Inflate (x, y, width, height) → (xmin, xmax, ymin, ymax) for GeomRect."""
    import polars as pl

    x = data["x"]
    y = data["y"]
    if "width" in data.columns:
        w = data["width"]
    else:
        w = pl.Series("width", [1.0] * len(data))
    if "height" in data.columns:
        h = data["height"]
    else:
        h = pl.Series("height", [1.0] * len(data))

    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2
    out_cols = {
        "xmin": xmin.alias("xmin"),
        "xmax": xmax.alias("xmax"),
        "ymin": ymin.alias("ymin"),
        "ymax": ymax.alias("ymax"),
    }
    # Carry over fill/colour/alpha/size/linetype if present.
    for aes in ("fill", "colour", "alpha", "size", "linetype"):
        if aes in data.columns:
            out_cols[aes] = data[aes].alias(aes)
    return pl.DataFrame(out_cols)


def _scalar(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _per_row_color(df, col, default, *, missing_value=None):
    """Return per-row colour values as a list (or scalar if uniform).

    ``missing_value`` lets callers pass through e.g. ``"none"`` when the
    aesthetic column has nulls (used for missing edge colour)."""
    from .._util import r_color

    if col not in df.columns or len(df) == 0:
        return r_color(default)
    vals = df[col].to_list()
    if missing_value is not None:
        vals = [missing_value if v is None else v for v in vals]
    else:
        vals = [default if v is None else v for v in vals]
    distinct = set(vals)
    if len(distinct) <= 1:
        only = next(iter(distinct))
        return only if only == "none" else r_color(only)
    return [v if v == "none" else r_color(v) for v in vals]


def _color_to_rgba(c, default):
    import matplotlib.colors as mcolors
    from .._util import r_color

    if c is None:
        c = default
    return mcolors.to_rgba(r_color(c))


def geom_rect(mapping=None, data=None, *, stat="identity", position="identity",
              **kwargs):
    """Filled rectangles from ``xmin``/``xmax``/``ymin``/``ymax`` aesthetics.

    Each row produces one rectangle. Use :func:`geom_tile` if your data
    has center coordinates and width/height instead.
    """
    return _make_layer(GeomRect(), mapping, data, stat, position, kwargs)


def geom_tile(mapping=None, data=None, *, stat="identity", position="identity",
              **kwargs):
    """Heatmap-style tiles centred at ``(x, y)`` with ``width``/``height``
    (default 1 each).
    """
    return _make_layer(GeomTile(), mapping, data, stat, position, kwargs)


def geom_raster(mapping=None, data=None, *, stat="identity", position="identity",
                **kwargs):
    """Fast heatmap via :func:`matplotlib.axes.Axes.imshow`. Requires a
    regular ``(x, y)`` grid (uniformly spaced cells); falls back to
    :func:`geom_tile` for irregular grids.
    """
    return _make_layer(GeomRaster(), mapping, data, stat, position, kwargs)


def _make_layer(geom, mapping, data, stat, position, kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype",
                           "alpha", "width", "height", "xmin", "xmax",
                           "ymin", "ymax"}}
    return Layer(
        geom=geom,
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
