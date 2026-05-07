"""``geom_contour`` and ``geom_contour_filled`` — 2D contour plots.

Both expect a regular ``(x, y)`` grid with a ``z`` column (one row per
cell). matplotlib's ``ax.contour`` / ``ax.contourf`` does the actual
contour computation given the reshaped grid.

Following ggplot2's split: ``geom_contour`` draws iso-lines as a
:class:`matplotlib.contour.QuadContourSet` of lines; ``geom_contour_filled``
draws filled bands between successive iso-levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


def _reshape_grid(data):
    """Turn a long-form ``(x, y, z)`` frame on a regular grid into the
    ``(X, Y, Z)`` 2D arrays matplotlib's contour expects."""
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    z = data["z"].to_numpy()
    xs = np.unique(x)
    ys = np.unique(y)
    if len(x) != len(xs) * len(ys):
        raise ValueError(
            "geom_contour: data must form a regular (x, y) grid; "
            f"got {len(x)} rows but {len(xs)} × {len(ys)} = "
            f"{len(xs) * len(ys)} cells"
        )
    # Sort by (y, x) so the reshaped grid is row-major in y.
    Z = np.zeros((len(ys), len(xs)))
    x_idx = np.searchsorted(xs, x)
    y_idx = np.searchsorted(ys, y)
    Z[y_idx, x_idx] = z
    X, Y = np.meshgrid(xs, ys)
    return X, Y, Z


@dataclass
class GeomContour(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y", "z")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color
        from ...plot._util import r_lty

        if len(data) == 0:
            return
        X, Y, Z = _reshape_grid(data)
        bins = data["bins"][0] if "bins" in data.columns else None
        n = bins if isinstance(bins, int) else 10
        ax.contour(
            X, Y, Z,
            levels=n,
            colors=r_color(_first(data, "colour", "black") or "black"),
            linewidths=float(_first(data, "size", 0.5) or 0.5) * _PT_PER_MM,
            linestyles=r_lty(_first(data, "linetype", "solid") or "solid"),
            alpha=float(_first(data, "alpha", 1.0) or 1.0),
        )


@dataclass
class GeomContourFilled(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "fill": None,  # auto via colormap when unset
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y", "z")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        X, Y, Z = _reshape_grid(data)
        bins = data["bins"][0] if "bins" in data.columns else None
        n = bins if isinstance(bins, int) else 10
        ax.contourf(
            X, Y, Z,
            levels=n,
            alpha=float(_first(data, "alpha", 1.0) or 1.0),
        )


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def geom_contour(mapping=None, data=None, *, stat="identity",
                 position="identity", bins=10, **kwargs):
    """Iso-lines on a regular ``(x, y)`` grid of ``z`` values.

    Pass ``bins=`` for the number of contour levels (default 10).
    """
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "linetype", "alpha"}}
    aes_params["bins"] = bins
    return Layer(
        geom=GeomContour(),
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def geom_contour_filled(mapping=None, data=None, *, stat="identity",
                        position="identity", bins=10, **kwargs):
    """Filled bands between successive iso-levels on a regular ``(x, y)``
    grid of ``z`` values.
    """
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"fill", "alpha"}}
    aes_params["bins"] = bins
    return Layer(
        geom=GeomContourFilled(),
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
