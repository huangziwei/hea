"""``geom_smooth()`` — smoothed conditional means, with optional CI ribbon.

A smooth layer is conceptually two layers stacked:

1. A :class:`GeomRibbon` for the CI band (drawn first, so the line sits on top);
2. A :class:`GeomPath` for the fitted curve.

Rather than building two ``Layer`` objects, we ship a small composite geom
that draws both itself. ggplot2 does the same internally
(``GeomSmooth$draw_group``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import Geom
from .path import GeomPath
from .ribbon import GeomRibbon


@dataclass
class GeomSmooth(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "#3366FF",  # ggplot2 default smooth colour
        "fill": "grey60",
        "size": 1.0,
        "linetype": "solid",
        "alpha": 0.4,
        "weight": 1.0,
    })
    required_aes: tuple = ("x", "y")

    se: bool = True

    def draw_panel(self, data, ax) -> None:
        if self.se and "ymin" in data.columns and "ymax" in data.columns:
            ribbon = GeomRibbon()
            ribbon.draw_panel(data, ax)
        path = GeomPath(sort_by_x=True)
        path.draw_panel(data, ax)


def geom_smooth(mapping=None, data=None, *, method="loess", formula=None,
                se=True, level=0.95, span=0.75, n=80,
                position="identity", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.smooth import StatSmooth

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}

    return Layer(
        geom=GeomSmooth(se=se),
        stat=StatSmooth(method=method, formula=formula, se=se,
                        level=level, span=span, n=n),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
