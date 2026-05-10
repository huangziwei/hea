"""``geom_segment``, ``geom_curve`` — line segments / curves between
``(x, y)`` and ``(xend, yend)``.

* :class:`GeomSegment` draws a straight line per row via
  :class:`matplotlib.collections.LineCollection` (efficient for many).
  When an ``arrow=`` spec is set it switches to per-row
  :class:`matplotlib.patches.FancyArrowPatch` so an arrowhead can render.
* :class:`GeomCurve` draws an arc per row via
  :class:`matplotlib.patches.FancyArrowPatch` with ``arc3,rad=curvature``;
  also accepts ``arrow=``.

The :func:`arrow` factory mirrors R's ``grid::arrow()`` — pass its
result as ``geom_segment(..., arrow=arrow(...))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from ..aes import split_layer_kwargs
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


def _to_numeric_axis(arr):
    """Coerce datetime64 / timedelta64 arrays to matplotlib's float
    representation so they can sit in the same numpy array as numeric
    coordinates — needed by ``LineCollection`` / ``column_stack``, which
    require a uniform dtype.

    Numeric dtypes pass through unchanged."""
    kind = getattr(arr.dtype, "kind", None)
    if kind == "M":  # datetime64
        from matplotlib.dates import date2num
        return date2num(arr)
    if kind == "m":  # timedelta64
        return arr.astype("float64")
    return arr


def _update_lims(ax, xs, ys):
    ax.update_datalim(
        [(float(np.min(xs)), float(np.min(ys))),
         (float(np.max(xs)), float(np.max(ys)))]
    )
    ax.autoscale_view()


# ---------------------------------------------------------------------------
# arrow() — grid::arrow() port for geom_segment / geom_curve / annotate
# ---------------------------------------------------------------------------


@dataclass
class Arrow:
    """Specification for an arrowhead on a line geom — port of R's
    ``grid::arrow()``. Construct via :func:`arrow`."""
    angle: float = 30.0      # half-angle at the tip, degrees
    length: float = 0.25     # head length, inches (matches grid default)
    ends: str = "last"       # "first" | "last" | "both"
    type: str = "open"       # "open" (outline) | "closed" (filled triangle)


def arrow(angle=30.0, length=0.25, ends="last", type="open"):
    """Arrow-head spec for line-drawing geoms (port of ``grid::arrow()``).

    Pass to ``geom_segment(arrow=...)``, ``geom_curve(arrow=...)``, or
    ``annotate("segment", arrow=...)``.

    Parameters
    ----------
    angle : float, default 30
        Half-angle of the arrowhead at the tip, in degrees.
    length : float, default 0.25
        Length of the arrowhead in inches (matches R's default
        ``unit(0.25, "inches")``).
    ends : {"last", "first", "both"}, default "last"
        Which end gets the arrowhead.
    type : {"open", "closed"}, default "open"
        ``"open"`` = outline-only ``>`` head, ``"closed"`` = filled
        triangle ``|>`` head.
    """
    return Arrow(angle=float(angle), length=float(length),
                 ends=str(ends), type=str(type))


def _arrow_to_style(arrow_spec):
    """Convert :class:`Arrow` (or matplotlib-style string, or ``None``)
    to ``(arrowstyle, mutation_scale)`` for :class:`FancyArrowPatch`."""
    if arrow_spec is None:
        return ("-", 1.0)
    if isinstance(arrow_spec, str):
        return (arrow_spec, 10.0)
    head = "|>" if arrow_spec.type == "closed" else ">"
    tail = "<|" if arrow_spec.type == "closed" else "<"
    if arrow_spec.ends == "first":
        style = f"{tail}-"
    elif arrow_spec.ends == "both":
        style = f"{tail}-{head}"
    else:
        style = f"-{head}"
    # ``mutation_scale`` sets the arrow-head size in points. Map R's
    # ``length`` (inches) → points so 0.25" → 18 pt — matches grid's
    # default arrow visually at typical figure sizes.
    mutation_scale = arrow_spec.length * 72.0
    return (style, mutation_scale)


@dataclass
class GeomSegment(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y", "xend", "yend")
    key_glyph: str = "path"
    # ``arrow=arrow(...)`` (or matplotlib arrowstyle string). When set,
    # we render via per-row FancyArrowPatch so the head can draw;
    # otherwise we use the fast LineCollection path.
    arrow: object = None

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty
        from .._util import r_color

        if len(data) == 0:
            return
        x = _to_numeric_axis(data["x"].to_numpy())
        y = _to_numeric_axis(data["y"].to_numpy())
        xend = _to_numeric_axis(data["xend"].to_numpy())
        yend = _to_numeric_axis(data["yend"].to_numpy())

        if self.arrow is None:
            # Fast path: no arrowhead → one LineCollection for all rows.
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
        else:
            # Arrow needed → per-row FancyArrowPatch (each can carry its
            # own colour / size / linestyle, like ggplot2's behaviour).
            n = len(data)
            colours = data["colour"].to_list() if "colour" in data.columns else ["black"] * n
            sizes = data["size"].to_numpy() if "size" in data.columns else np.full(n, 0.5)
            linetypes = data["linetype"].to_list() if "linetype" in data.columns else ["solid"] * n
            alphas = data["alpha"].to_numpy() if "alpha" in data.columns else np.full(n, 1.0)
            arrowstyle, mutation_scale = _arrow_to_style(self.arrow)

            for i in range(n):
                patch = FancyArrowPatch(
                    (float(x[i]), float(y[i])),
                    (float(xend[i]), float(yend[i])),
                    color=r_color(colours[i] if colours[i] is not None else "black"),
                    linewidth=float(sizes[i]) * _PT_PER_MM,
                    linestyle=r_lty(linetypes[i] if linetypes[i] is not None else "solid"),
                    alpha=float(alphas[i]) if alphas[i] is not None else 1.0,
                    arrowstyle=arrowstyle,
                    mutation_scale=mutation_scale,
                    shrinkA=0, shrinkB=0,
                )
                ax.add_patch(patch)

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
    arrow: object = None

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty
        from .._util import r_color

        if len(data) == 0:
            return

        n = len(data)
        x = _to_numeric_axis(data["x"].to_numpy())
        y = _to_numeric_axis(data["y"].to_numpy())
        xend = _to_numeric_axis(data["xend"].to_numpy())
        yend = _to_numeric_axis(data["yend"].to_numpy())
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
        arrowstyle, mutation_scale = _arrow_to_style(self.arrow)

        for i in range(n):
            patch = FancyArrowPatch(
                (float(x[i]), float(y[i])),
                (float(xend[i]), float(yend[i])),
                connectionstyle=f"arc3,rad={float(curv[i])}",
                color=r_color(colours[i] if colours[i] is not None else "black"),
                linewidth=float(sizes[i]) * _PT_PER_MM,
                linestyle=r_lty(linetypes[i] if linetypes[i] is not None else "solid"),
                alpha=float(alphas[i]) if alphas[i] is not None else 1.0,
                arrowstyle=arrowstyle,
                mutation_scale=mutation_scale,
                shrinkA=0, shrinkB=0,
            )
            ax.add_patch(patch)

        _update_lims(ax, np.concatenate([x, xend]), np.concatenate([y, yend]))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_layer(geom, mapping, data, stat, position, kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params, geom_params = split_layer_kwargs(kwargs)
    return Layer(
        geom=geom,
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )


def geom_segment(mapping=None, data=None, *, stat="identity", position="identity",
                 arrow=None, **kwargs):
    """Straight line segment from ``(x, y)`` to ``(xend, yend)``.

    Pass ``arrow=arrow(...)`` to add an arrowhead. Without ``arrow``
    the geom uses a single ``LineCollection`` for all rows (fast); with
    it, one ``FancyArrowPatch`` per row.
    """
    return _make_layer(GeomSegment(arrow=arrow), mapping, data, stat, position, kwargs)


def geom_curve(mapping=None, data=None, *, stat="identity", position="identity",
               arrow=None, **kwargs):
    """Curved segment. ``curvature`` (default 0.5) controls arc magnitude;
    sign flips the bending side. ``arrow=arrow(...)`` adds an arrowhead."""
    return _make_layer(GeomCurve(arrow=arrow), mapping, data, stat, position, kwargs)
