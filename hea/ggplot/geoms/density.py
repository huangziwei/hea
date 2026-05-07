"""``geom_density()`` — KDE curve, optionally filled.

ggplot2 default: ``colour = "black"``, ``fill = NA`` (line only). We
match that — no fill unless the user sets one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import Geom


@dataclass
class GeomDensity(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": None,
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty

        if len(data) == 0:
            return

        # Per-group: ``aes(colour = species)`` puts a different colour on
        # each group's rows, so plotting ``data`` whole would merge all
        # curves into one line painted in the first group's colour. Split
        # like geom_path / geom_line do.
        if "group" in data.columns:
            for _, sub in data.group_by("group", maintain_order=True):
                self._draw_one(sub, ax, r_lty)
        else:
            self._draw_one(data, ax, r_lty)

    def _draw_one(self, sub, ax, r_lty) -> None:
        x = sub["x"].to_numpy()
        y = sub["y"].to_numpy()
        order = np.argsort(x)
        xs, ys = x[order], y[order]

        colour = _first(sub, "colour", default="black")
        fill = _first(sub, "fill", default=None)
        size = float(_first(sub, "size", default=0.5))
        linetype = _first(sub, "linetype", default="solid")
        alpha = float(_first(sub, "alpha", default=1.0))

        if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
            ax.fill_between(xs, 0, ys, color=fill, alpha=alpha, linewidth=0)

        ax.plot(xs, ys, color=colour, linewidth=size * 2.83,
                linestyle=r_lty(linetype), alpha=alpha)


def _first(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def geom_density(mapping=None, data=None, *, bw="nrd0", n=512,
                 position="identity", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.density import StatDensity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}
    return Layer(
        geom=GeomDensity(),
        stat=StatDensity(bw=bw, n=n),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
