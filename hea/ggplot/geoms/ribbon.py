"""``geom_ribbon()`` — filled band between ``ymin`` and ``ymax``.

``geom_area()`` is a thin wrapper that injects ``ymin = 0`` and treats ``y``
as ``ymax`` — the canonical "area under a curve" form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .geom import Geom


@dataclass
class GeomRibbon(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": None,
        "fill": "grey60",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 0.4,
    })
    required_aes: tuple = ("x", "ymin", "ymax")

    def draw_panel(self, data, ax) -> None:
        if "group" in data.columns:
            for _, sub in data.group_by("group", maintain_order=True):
                self._draw_one(sub, ax)
        else:
            self._draw_one(data, ax)

    def _draw_one(self, sub, ax) -> None:
        from .._util import r_color

        x = sub["x"].to_numpy()
        ymin = sub["ymin"].to_numpy()
        ymax = sub["ymax"].to_numpy()
        order = np.argsort(x)
        x, ymin, ymax = x[order], ymin[order], ymax[order]

        fill = r_color(_first(sub, "fill", "grey60"))
        edge_raw = _first(sub, "colour", None)
        edge = r_color(edge_raw) if edge_raw is not None else "none"
        alpha = float(_first(sub, "alpha", 0.4))

        ax.fill_between(x, ymin, ymax, facecolor=fill, edgecolor=edge,
                        alpha=alpha, linewidth=0)


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def geom_ribbon(mapping=None, data=None, *, position="identity", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.identity import StatIdentity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}
    return Layer(
        geom=GeomRibbon(),
        stat=StatIdentity(),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


# ---------------------------------------------------------------------------
# geom_area = ribbon with ymin=0 injected
# ---------------------------------------------------------------------------

@dataclass
class GeomArea(GeomRibbon):
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        # Inject ymin/ymax from y if positions hasn't already.
        if "y" in data.columns and "ymin" not in data.columns:
            data = data.with_columns(
                ymin=pl.lit(0.0).cast(pl.Float64),
                ymax=pl.col("y").cast(pl.Float64),
            )
        super().draw_panel(data, ax)


def geom_area(mapping=None, data=None, *, position="stack", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.identity import StatIdentity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}
    return Layer(
        geom=GeomArea(),
        stat=StatIdentity(),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
