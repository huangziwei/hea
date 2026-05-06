"""``geom_point()`` — scatter plot."""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import Geom


@dataclass
class GeomPoint(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 1.5,
        "shape": "o",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        colour = data["colour"].to_list() if "colour" in data.columns else self.default_aes["colour"]
        size = data["size"].to_numpy() if "size" in data.columns else self.default_aes["size"]
        shape = data["shape"].to_list()[0] if "shape" in data.columns else self.default_aes["shape"]
        alpha = data["alpha"].to_numpy() if "alpha" in data.columns else self.default_aes["alpha"]

        # ggplot2 size is in mm; matplotlib `s` is in points² (1pt ≈ 0.353mm).
        # ggplot2's scaling is `s = (size_mm * pt_per_mm)²`. Approximate constant
        # of ~6 lines up visually with ggplot2 default at size=1.5.
        s = (size * 6) ** 2 if hasattr(size, "__len__") else (size * 6) ** 2

        ax.scatter(x, y, s=s, c=colour, marker=shape, alpha=alpha)


def geom_point(mapping=None, data=None, *, stat="identity", position="identity",
               na_rm=False, show_legend=True, inherit_aes=True, **kwargs):
    from ..layer import Layer
    from ..stats.identity import StatIdentity
    from ..positions.identity import PositionIdentity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "shape", "alpha", "fill", "stroke"}}
    geom_params = {k: v for k, v in kwargs.items() if k not in aes_params}

    return Layer(
        geom=GeomPoint(),
        stat=StatIdentity() if stat == "identity" else stat,
        position=PositionIdentity() if position == "identity" else position,
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        inherit_aes=inherit_aes,
        show_legend=show_legend,
    )
