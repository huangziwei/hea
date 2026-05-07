"""``geom_polygon`` — filled polygons from ``(x, y)`` vertices.

Each row is one vertex; rows are split into polygons by ``group``. The
polygon for a group is the closed shape connecting its vertices in row
order. Open shapes are closed automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomPolygon(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": None,
        "fill": "grey20",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return

        # Auto-group fallback: when no group aes is set (`add_group` returns
        # group=-1), treat the whole frame as a single polygon. Otherwise
        # split per group.
        if "group" in data.columns:
            groups = data["group"].to_list()
        else:
            groups = [-1] * len(data)

        order = sorted(set(groups), key=lambda g: groups.index(g))
        patches = []
        per_patch_fills = []
        per_patch_edges = []
        for g in order:
            sub = data.filter(data["group"] == g) if "group" in data.columns else data
            if len(sub) < 3:
                # Need at least 3 vertices for a polygon.
                continue
            xy = np.column_stack([sub["x"].to_numpy(), sub["y"].to_numpy()])
            patches.append(Polygon(xy, closed=True))
            per_patch_fills.append(_first_color(sub, "fill", "grey20"))
            per_patch_edges.append(_first_color(sub, "colour", "none",
                                                missing_value="none"))

        if not patches:
            return

        alpha = float(_scalar(data, "alpha", default=1.0))
        linewidth = float(_scalar(data, "size", default=0.5)) * _PT_PER_MM

        coll = PatchCollection(
            patches,
            facecolors=per_patch_fills,
            edgecolors=per_patch_edges,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(coll)
        # PatchCollection doesn't auto-update axes data limits.
        all_x = data["x"].to_numpy()
        all_y = data["y"].to_numpy()
        ax.update_datalim(
            [(float(all_x.min()), float(all_y.min())),
             (float(all_x.max()), float(all_y.max()))]
        )
        ax.autoscale_view()


def _scalar(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _first_color(df, col, default, *, missing_value=None):
    """First non-null colour in column ``col``, or default. Polygons take a
    single fill/edge colour each (per-vertex colour wouldn't make visual
    sense for filled shapes)."""
    from .._util import r_color

    if col not in df.columns or len(df) == 0:
        return r_color(default)
    for v in df[col].to_list():
        if v is None:
            continue
        if missing_value is not None and v == missing_value:
            return missing_value
        return r_color(v)
    return missing_value if missing_value is not None else r_color(default)


def geom_polygon(mapping=None, data=None, *, stat="identity",
                 position="identity", **kwargs):
    """Filled polygons. Each ``group`` becomes one polygon, vertices in
    row order. Aesthetics: ``x``, ``y``, ``group``, ``fill``, ``colour``,
    ``alpha``, ``size``, ``linetype``."""
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype",
                           "alpha", "group"}}
    return Layer(
        geom=GeomPolygon(),
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
