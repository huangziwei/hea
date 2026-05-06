"""``geom_text()`` — text labels at (x, y).

Smaller surface than ggplot2's full geom_text: we ship the common
arguments (``hjust``, ``vjust``, ``angle``, ``size``, ``colour``,
``family``, ``fontface``). ``geom_label`` (background-boxed text)
arrives in 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomText(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 3.88,  # ggplot2 default — 11pt / 2.83 mm/pt
        "angle": 0.0,
        "hjust": 0.5,
        "vjust": 0.5,
        "alpha": 1.0,
        "family": "",
        "fontface": "plain",
    })
    required_aes: tuple = ("x", "y", "label")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        labels = data["label"].to_list()

        # Per-row attributes — broadcast-style fallback to default if absent.
        colour = data["colour"].to_list() if "colour" in data.columns else None
        size = data["size"].to_numpy() if "size" in data.columns else None
        angle = data["angle"].to_numpy() if "angle" in data.columns else None
        hjust = data["hjust"].to_numpy() if "hjust" in data.columns else None
        vjust = data["vjust"].to_numpy() if "vjust" in data.columns else None

        for i, label in enumerate(labels):
            if label is None:
                continue
            kwargs = {
                "color": r_color(colour[i]) if colour else self.default_aes["colour"],
                "fontsize": float(size[i]) * _PT_PER_MM if size is not None
                            else self.default_aes["size"] * _PT_PER_MM,
                "rotation": float(angle[i]) if angle is not None
                            else self.default_aes["angle"],
                "ha": _hjust_to_ha(hjust[i] if hjust is not None
                                    else self.default_aes["hjust"]),
                "va": _vjust_to_va(vjust[i] if vjust is not None
                                    else self.default_aes["vjust"]),
            }
            ax.text(x[i], y[i], str(label), **kwargs)


def _hjust_to_ha(h: float) -> str:
    if h <= 0.25:
        return "left"
    if h >= 0.75:
        return "right"
    return "center"


def _vjust_to_va(v: float) -> str:
    if v <= 0.25:
        return "bottom"
    if v >= 0.75:
        return "top"
    return "center"


def geom_text(mapping=None, data=None, *, stat="identity", position="identity",
              na_rm=False, **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "angle", "hjust", "vjust",
                           "alpha", "family", "fontface", "label"}}

    return Layer(
        geom=GeomText(),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        na_rm=na_rm,
    )
