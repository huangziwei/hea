"""``geom_bin2d()`` and ``geom_hex()`` — 2D bin counts as rectangular tiles
or hexagonal cells.

Both go through the standard build pipeline: a :class:`StatBin2d` /
:class:`StatBinhex` computes per-bin counts, the result flows through the
``fill`` scale (default: continuous gradient mapped from ``count``), and
the geom renders one polygon per non-empty bin.

Override the default count-driven fill via
``aes(fill=after_stat("density"))`` or any explicit fill mapping. A
constant ``geom_bin2d(fill="red")`` still wins, applied after the scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.collections import PolyCollection

from ..aes import split_layer_kwargs
from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


# ---------------------------------------------------------------------------
# Hexagon geom — draws a regular hex polygon per bin centre.
# ---------------------------------------------------------------------------


@dataclass
class GeomHex(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": None,
        "fill": "grey20",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")
    key_glyph: str = "polygon"

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return

        x = data["x"].to_numpy().astype(float)
        y = data["y"].to_numpy().astype(float)
        n = len(x)
        # Width / height of the hex polygon in data units. Stat output
        # carries these per-row (constant across rows for a given panel)
        # so a future ``coord_fixed`` could override on the geom side.
        w = data["width"].to_numpy().astype(float) if "width" in data.columns \
            else np.ones(n)
        h = data["height"].to_numpy().astype(float) if "height" in data.columns \
            else np.ones(n)

        # Pointy-top hex polygon, vertices in CCW order relative to centre:
        # top, upper-right, lower-right, bottom, lower-left, upper-left.
        # Width is the horizontal extent (between the two vertical edges);
        # height is the vertical extent (top vertex to bottom vertex).
        rel = np.array([
            [ 0.0,  0.5],
            [ 0.5,  0.25],
            [ 0.5, -0.25],
            [ 0.0, -0.5],
            [-0.5, -0.25],
            [-0.5,  0.25],
        ])
        polygons = np.zeros((n, 6, 2))
        polygons[:, :, 0] = x[:, None] + rel[None, :, 0] * w[:, None]
        polygons[:, :, 1] = y[:, None] + rel[None, :, 1] * h[:, None]

        fills = _per_row_color(data, "fill", "grey20")
        edges = _per_row_color(data, "colour", "none", missing_value="none")
        alpha = float(_scalar(data, "alpha", 1.0))
        linewidth = float(_scalar(data, "size", 0.5)) * _PT_PER_MM

        coll = PolyCollection(
            polygons,
            facecolors=fills,
            edgecolors=edges,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(coll)
        # PolyCollection doesn't auto-update data limits.
        ax.update_datalim([
            (float(polygons[:, :, 0].min()), float(polygons[:, :, 1].min())),
            (float(polygons[:, :, 0].max()), float(polygons[:, :, 1].max())),
        ])
        ax.autoscale_view()


# ---------------------------------------------------------------------------
# Helpers (mirrors rect.py's variants — kept local to avoid coupling).
# ---------------------------------------------------------------------------


def _scalar(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _per_row_color(df, col, default, *, missing_value=None):
    """Per-row colour list (or scalar when uniform). ``missing_value`` lets
    callers pass through e.g. ``"none"`` for a missing edge colour."""
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


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def geom_hex(mapping=None, data=None, *, stat="binhex", position="identity",
             bins=30, **kwargs):
    """Hexagonal 2D bin counts. Each non-empty hex is rendered as a
    polygon; ``fill`` defaults to ``count`` (mapped through the fill
    scale). Pass ``aes(fill=after_stat("density"))`` to switch to
    density. ``bins`` controls the x-direction grid resolution.
    """
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat
    from ..stats.bin2d import StatBinhex

    if stat == "binhex":
        stat_obj = StatBinhex(bins=bins)
    elif isinstance(stat, str):
        stat_obj = resolve_stat(stat)
    else:
        stat_obj = stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomHex(),
        stat=stat_obj,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )


def geom_bin2d(mapping=None, data=None, *, stat="bin_2d", position="identity",
               bins=30, binwidth=None, **kwargs):
    """Rectangular 2D bin counts. Each non-empty bin is rendered as a
    tile (via :class:`GeomTile`); ``fill`` defaults to ``count`` (mapped
    through the fill scale).

    ``bins`` (default 30) — number of x and y bins. Pass a ``(nx, ny)``
    tuple for distinct counts per axis. ``binwidth`` overrides ``bins``
    with literal cell sizes (scalar = same on both axes).
    """
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat
    from ..stats.bin2d import StatBin2d
    from .rect import GeomTile

    if stat == "bin_2d":
        stat_obj = StatBin2d(bins=bins, binwidth=binwidth)
    elif isinstance(stat, str):
        stat_obj = resolve_stat(stat)
    else:
        stat_obj = stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomTile(),
        stat=stat_obj,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )
