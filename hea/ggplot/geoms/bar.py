"""``geom_bar()``, ``geom_col()`` — bar charts.

* ``geom_bar()`` — counts per discrete x via :class:`StatCount`.
* ``geom_col()`` — uses y as supplied (``stat_identity``).

``geom_histogram`` lives in ``histogram.py`` and reuses :class:`GeomBar`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import Geom


@dataclass
class GeomBar(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": None,
        "fill": "grey35",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        if "width" in data.columns:
            width = data["width"].to_numpy()
        else:
            width = 0.9

        fill = r_color(_scalar(data, "fill", default="grey35"))
        edge = r_color(_edge_colour(data))
        alpha = float(_scalar(data, "alpha", default=1.0))

        ax.bar(x, y, width=width, color=fill, edgecolor=edge, alpha=alpha,
               linewidth=0.5, align="center")


def _scalar(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _edge_colour(df):
    """Map ggplot2's ``colour = NA`` (no edge) to matplotlib's ``edgecolor='none'``."""
    if "colour" not in df.columns or len(df) == 0:
        return "none"
    val = df["colour"][0]
    if val is None:
        return "none"
    if isinstance(val, float) and np.isnan(val):
        return "none"
    return val


def geom_bar(mapping=None, data=None, *, stat="count", position="stack", **kwargs):
    from ..layer import Layer
    from ..positions.identity import PositionIdentity
    from ..stats.count import StatCount

    if stat == "count":
        stat_obj = StatCount()
    elif hasattr(stat, "compute_layer"):
        stat_obj = stat
    else:
        raise ValueError(f"geom_bar: unknown stat {stat!r}")

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}

    return Layer(
        geom=GeomBar(),
        stat=stat_obj,
        position=PositionIdentity(),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def geom_col(mapping=None, data=None, *, position="identity", **kwargs):
    from ..layer import Layer
    from ..positions.identity import PositionIdentity
    from ..stats.identity import StatIdentity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}
    return Layer(
        geom=GeomBar(),
        stat=StatIdentity(),
        position=PositionIdentity(),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
