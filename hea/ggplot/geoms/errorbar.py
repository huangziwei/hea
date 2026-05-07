"""Error-range geoms — ``geom_errorbar``, ``geom_errorbarh``, ``geom_linerange``,
``geom_pointrange``, ``geom_crossbar``.

All five share a ``ymin``/``ymax`` (or ``xmin``/``xmax`` for ``errorbarh``)
range concept and differ in shape:

* :class:`GeomLinerange` — vertical line ``ymin → ymax``, no caps.
* :class:`GeomErrorbar` — vertical line plus horizontal end caps.
* :class:`GeomErrorbarh` — horizontal version of ``geom_errorbar``.
* :class:`GeomPointrange` — vertical line plus a point at ``(x, y)``.
* :class:`GeomCrossbar` — rectangular outline with a thicker bar at ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


def _vlines(ax, x, ymin, ymax, **kw):
    ax.vlines(x, ymin, ymax, **kw)


def _hlines(ax, y, xmin, xmax, **kw):
    ax.hlines(y, xmin, xmax, **kw)


def _scalar(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _scalar_color(df, col, default):
    from .._util import r_color
    val = _scalar(df, col, default=default)
    return r_color(val)


def _line_kwargs(data):
    """Common kwargs for ``vlines``/``hlines`` from a frame's aes columns."""
    from ...plot._util import r_lty
    return {
        "colors": _scalar_color(data, "colour", "black"),
        "linewidths": float(_scalar(data, "size", default=0.5)) * _PT_PER_MM,
        "linestyles": r_lty(_scalar(data, "linetype", default="solid")),
        "alpha": float(_scalar(data, "alpha", default=1.0)),
    }


@dataclass
class GeomLinerange(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "ymin", "ymax")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        _vlines(
            ax,
            data["x"].to_numpy(),
            data["ymin"].to_numpy(),
            data["ymax"].to_numpy(),
            **_line_kwargs(data),
        )


@dataclass
class GeomErrorbar(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
        "width": 0.5,
    })
    required_aes: tuple = ("x", "ymin", "ymax")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        ymin = data["ymin"].to_numpy()
        ymax = data["ymax"].to_numpy()
        w = data["width"].to_numpy() if "width" in data.columns else np.full(len(data), 0.5)
        kw = _line_kwargs(data)
        _vlines(ax, x, ymin, ymax, **kw)
        _hlines(ax, ymin, x - w / 2, x + w / 2, **kw)
        _hlines(ax, ymax, x - w / 2, x + w / 2, **kw)


@dataclass
class GeomErrorbarh(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
        "height": 0.5,
    })
    required_aes: tuple = ("y", "xmin", "xmax")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        y = data["y"].to_numpy()
        xmin = data["xmin"].to_numpy()
        xmax = data["xmax"].to_numpy()
        h = data["height"].to_numpy() if "height" in data.columns else np.full(len(data), 0.5)
        kw = _line_kwargs(data)
        _hlines(ax, y, xmin, xmax, **kw)
        _vlines(ax, xmin, y - h / 2, y + h / 2, **kw)
        _vlines(ax, xmax, y - h / 2, y + h / 2, **kw)


@dataclass
class GeomPointrange(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": None,
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
        "shape": "o",
    })
    required_aes: tuple = ("x", "y", "ymin", "ymax")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        ymin = data["ymin"].to_numpy()
        ymax = data["ymax"].to_numpy()

        kw = _line_kwargs(data)
        _vlines(ax, x, ymin, ymax, **kw)

        # Point at (x, y). ggplot2's pointrange uses size as line width AND
        # for the point: the point's diameter is ~4× the line width.
        line_pt = float(_scalar(data, "size", default=0.5)) * _PT_PER_MM
        point_size_pt2 = (line_pt * 4) ** 2
        ax.scatter(
            x, y,
            s=point_size_pt2,
            c=_scalar_color(data, "colour", "black"),
            marker=_scalar(data, "shape", default="o"),
            alpha=float(_scalar(data, "alpha", default=1.0)),
        )


@dataclass
class GeomCrossbar(Geom):
    """Rectangular outline ``[x ± w/2] × [ymin, ymax]`` with a thicker
    horizontal bar at ``y`` (the median line). ggplot2 default: no fill.
    """

    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": None,
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
        "width": 0.5,
    })
    required_aes: tuple = ("x", "y", "ymin", "ymax")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        ymin = data["ymin"].to_numpy()
        ymax = data["ymax"].to_numpy()
        w = data["width"].to_numpy() if "width" in data.columns else np.full(len(data), 0.5)

        kw = _line_kwargs(data)
        # Box outline: top, bottom, left, right.
        _hlines(ax, ymax, x - w / 2, x + w / 2, **kw)
        _hlines(ax, ymin, x - w / 2, x + w / 2, **kw)
        _vlines(ax, x - w / 2, ymin, ymax, **kw)
        _vlines(ax, x + w / 2, ymin, ymax, **kw)
        # Median line — thicker (ggplot2 doubles the width).
        median_kw = {**kw, "linewidths": kw["linewidths"] * 2}
        _hlines(ax, y, x - w / 2, x + w / 2, **median_kw)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_AES_PARAM_KEYS = frozenset({
    "colour", "color", "fill", "size", "linetype", "alpha", "width", "height",
    "shape",
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


def geom_errorbar(mapping=None, data=None, *, stat="identity", position="identity",
                  **kwargs):
    """Vertical error bars: line from ``ymin`` to ``ymax`` at each ``x``,
    with horizontal caps of ``width`` (default 0.5)."""
    return _make_layer(GeomErrorbar(), mapping, data, stat, position, kwargs)


def geom_errorbarh(mapping=None, data=None, *, stat="identity", position="identity",
                   **kwargs):
    """Horizontal error bars: line from ``xmin`` to ``xmax`` at each ``y``,
    with vertical caps of ``height`` (default 0.5)."""
    return _make_layer(GeomErrorbarh(), mapping, data, stat, position, kwargs)


def geom_linerange(mapping=None, data=None, *, stat="identity", position="identity",
                   **kwargs):
    """Vertical line from ``ymin`` to ``ymax`` at each ``x``, no caps."""
    return _make_layer(GeomLinerange(), mapping, data, stat, position, kwargs)


def geom_pointrange(mapping=None, data=None, *, stat="identity", position="identity",
                    **kwargs):
    """Vertical range line plus a point at ``(x, y)``. The range uses
    ``size`` as the line width; the point's diameter is ~4× that."""
    return _make_layer(GeomPointrange(), mapping, data, stat, position, kwargs)


def geom_crossbar(mapping=None, data=None, *, stat="identity", position="identity",
                  **kwargs):
    """Rectangular outline at ``[x ± width/2] × [ymin, ymax]`` with a
    thicker bar at ``y`` (the centre line)."""
    return _make_layer(GeomCrossbar(), mapping, data, stat, position, kwargs)
