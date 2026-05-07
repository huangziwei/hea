"""``geom_segment``, ``geom_curve`` — line segments / curves between
``(x, y)`` and ``(xend, yend)``.

* :class:`GeomSegment` draws a straight line per row via
  :class:`matplotlib.collections.LineCollection` (efficient for many).
* :class:`GeomCurve` draws an arc per row via
  :class:`matplotlib.patches.FancyArrowPatch` with ``arc3,rad=curvature``.

Arrow heads are not yet supported — :class:`GeomSegment` ignores any
``arrow=`` parameter and renders flat segments. Track as a polish item.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


def _scalar(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _per_row_color(df, col, default):
    """Per-row colour values; returns scalar when uniform, else list."""
    from .._util import r_color

    if col not in df.columns or len(df) == 0:
        return r_color(default)
    vals = df[col].to_list()
    vals = [default if v is None else v for v in vals]
    distinct = set(vals)
    if len(distinct) <= 1:
        return r_color(next(iter(distinct)))
    return [r_color(v) for v in vals]


def _per_row_widths(df, col, default):
    if col not in df.columns or len(df) == 0:
        return float(default) * _PT_PER_MM
    arr = df[col].to_numpy()
    if np.all(arr == arr[0]):
        return float(arr[0]) * _PT_PER_MM
    return arr.astype(float) * _PT_PER_MM


def _update_lims(ax, xs, ys):
    ax.update_datalim(
        [(float(np.min(xs)), float(np.min(ys))),
         (float(np.max(xs)), float(np.max(ys)))]
    )
    ax.autoscale_view()


@dataclass
class GeomSegment(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y", "xend", "yend")

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty

        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        xend = data["xend"].to_numpy()
        yend = data["yend"].to_numpy()

        # (n, 2, 2) — n segments, each two (x, y) endpoints.
        segments = np.stack(
            [np.column_stack([x, y]), np.column_stack([xend, yend])],
            axis=1,
        )

        coll = LineCollection(
            segments,
            colors=_per_row_color(data, "colour", "black"),
            linewidths=_per_row_widths(data, "size", 0.5),
            linestyles=r_lty(_scalar(data, "linetype", default="solid")),
            alpha=float(_scalar(data, "alpha", default=1.0)),
        )
        ax.add_collection(coll)
        _update_lims(ax, np.concatenate([x, xend]), np.concatenate([y, yend]))


@dataclass
class GeomCurve(Geom):
    """Curved segment from ``(x, y)`` to ``(xend, yend)``.

    ``curvature`` controls the bend: 0 = straight, ±0.5 (default) = a
    moderate arc, sign flips the side. Implementation: one
    :class:`matplotlib.patches.FancyArrowPatch` per row with
    ``connectionstyle='arc3,rad=<curvature>'``.
    """

    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
        "curvature": 0.5,
    })
    required_aes: tuple = ("x", "y", "xend", "yend")

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty
        from .._util import r_color

        if len(data) == 0:
            return

        n = len(data)
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        xend = data["xend"].to_numpy()
        yend = data["yend"].to_numpy()
        colours = (data["colour"].to_list()
                   if "colour" in data.columns else ["black"] * n)
        sizes = (data["size"].to_numpy()
                 if "size" in data.columns else np.full(n, 0.5))
        linetypes = (data["linetype"].to_list()
                     if "linetype" in data.columns else ["solid"] * n)
        alphas = (data["alpha"].to_numpy()
                  if "alpha" in data.columns else np.full(n, 1.0))
        curv = (data["curvature"].to_numpy()
                if "curvature" in data.columns else np.full(n, 0.5))

        for i in range(n):
            patch = FancyArrowPatch(
                (float(x[i]), float(y[i])),
                (float(xend[i]), float(yend[i])),
                connectionstyle=f"arc3,rad={float(curv[i])}",
                color=r_color(colours[i] if colours[i] is not None else "black"),
                linewidth=float(sizes[i]) * _PT_PER_MM,
                linestyle=r_lty(linetypes[i] if linetypes[i] is not None else "solid"),
                alpha=float(alphas[i]) if alphas[i] is not None else 1.0,
                arrowstyle="-",
            )
            ax.add_patch(patch)

        _update_lims(ax, np.concatenate([x, xend]), np.concatenate([y, yend]))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_AES_PARAM_KEYS = frozenset({
    "colour", "color", "size", "linetype", "alpha", "curvature",
})


def _make_layer(geom, mapping, data, stat, position, kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items() if k in _AES_PARAM_KEYS}
    return Layer(
        geom=geom,
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def geom_segment(mapping=None, data=None, *, stat="identity", position="identity",
                 **kwargs):
    """Straight line segment from ``(x, y)`` to ``(xend, yend)``."""
    return _make_layer(GeomSegment(), mapping, data, stat, position, kwargs)


def geom_curve(mapping=None, data=None, *, stat="identity", position="identity",
               **kwargs):
    """Curved segment. ``curvature`` (default 0.5) controls arc magnitude;
    sign flips the bending side."""
    return _make_layer(GeomCurve(), mapping, data, stat, position, kwargs)
