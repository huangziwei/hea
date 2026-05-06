"""``geom_point()`` — scatter plot."""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import Geom


# ggplot2 size is in mm. R's grid graphics uses 72.27 pt/inch (TeX convention),
# so 1 mm = 72.27 / 25.4 ≈ 2.8454 pt. matplotlib's ``s`` is the marker area in
# pt² (i.e. diameter² for a circle), so ``s = (size_mm * _PT_PER_MM) ** 2``.
_PT_PER_MM = 72.27 / 25.4


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

        s = (size * _PT_PER_MM) ** 2

        ax.scatter(x, y, s=s, c=colour, marker=shape, alpha=alpha)


def geom_point(mapping=None, data=None, *, stat="identity", position="identity",
               na_rm=False, show_legend=True, inherit_aes=True, **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.identity import StatIdentity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "shape", "alpha", "fill", "stroke"}}
    geom_params = {k: v for k, v in kwargs.items() if k not in aes_params}

    return Layer(
        geom=GeomPoint(),
        stat=StatIdentity() if stat == "identity" else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        inherit_aes=inherit_aes,
        show_legend=show_legend,
        na_rm=na_rm,
    )


def geom_jitter(mapping=None, data=None, *, width=None, height=None, seed=None,
                **kwargs):
    """``geom_point(position=position_jitter(...))`` — shortcut matching ggplot2."""
    from ..positions.jitter import position_jitter

    return geom_point(mapping=mapping, data=data,
                      position=position_jitter(width=width, height=height, seed=seed),
                      **kwargs)
