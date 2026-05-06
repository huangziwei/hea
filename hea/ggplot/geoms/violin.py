"""``geom_violin()`` — symmetric KDE polygon per group.

Reads ``y`` (KDE eval points) + ``violinwidth`` (per-group normalised
density) from :class:`StatYdensity` and draws a polygon at
``x ± width/2 · violinwidth``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import Geom


@dataclass
class GeomViolin(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": "white",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y", "violinwidth")

    half_width: float = 0.4

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return

        groupby_cols = ["x"]
        if "group" in data.columns:
            groupby_cols.append("group")

        for keys, sub in data.group_by(groupby_cols, maintain_order=True):
            self._draw_violin(sub, keys[0], ax, r_color)

    def _draw_violin(self, sub, x_center, ax, r_color):
        y = sub["y"].to_numpy()
        vw = sub["violinwidth"].to_numpy()
        order = np.argsort(y)
        y, vw = y[order], vw[order]

        half = vw * self.half_width
        # Build closed polygon: left side ascending, right side descending.
        poly_x = np.concatenate([x_center - half, (x_center + half)[::-1]])
        poly_y = np.concatenate([y, y[::-1]])

        fill = r_color(_first(sub, "fill", "white"))
        edge = r_color(_first(sub, "colour", "black"))
        alpha = float(_first(sub, "alpha", 1.0))

        ax.fill(poly_x, poly_y, facecolor=fill, edgecolor=edge,
                alpha=alpha, linewidth=0.5)


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def geom_violin(mapping=None, data=None, *, bw="nrd0", n=512,
                position="dodge", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.ydensity import StatYdensity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}

    return Layer(
        geom=GeomViolin(),
        stat=StatYdensity(bw=bw, n=n),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
