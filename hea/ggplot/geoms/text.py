"""``geom_text()`` and ``geom_label()`` — text annotations at (x, y).

Smaller surface than ggplot2's full geom_text: we ship the common
arguments (``hjust``, ``vjust``, ``angle``, ``size``, ``colour``,
``family``, ``fontface``). ``geom_label`` adds a rounded background
box; surface mirrors ggplot2's: ``fill``, ``label_padding``, ``label_r``
(corner radius), ``label_size`` (border line width).
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


@dataclass
class GeomLabel(GeomText):
    """``geom_text`` plus a rounded background box (ggplot2 ``geom_label``)."""
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": "white",
        "size": 3.88,
        "angle": 0.0,
        "hjust": 0.5,
        "vjust": 0.5,
        "alpha": 1.0,
        "family": "",
        "fontface": "plain",
    })
    label_padding: float = 0.25  # ggplot2 default in lines; we treat as box pad
    label_r: float = 0.15        # corner radius
    label_size: float = 0.25     # border line width (mm)

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        labels = data["label"].to_list()
        n = len(labels)

        colour = data["colour"].to_list() if "colour" in data.columns else [self.default_aes["colour"]] * n
        fill = data["fill"].to_list() if "fill" in data.columns else [self.default_aes["fill"]] * n
        size = data["size"].to_numpy() if "size" in data.columns else [self.default_aes["size"]] * n
        angle = data["angle"].to_numpy() if "angle" in data.columns else [self.default_aes["angle"]] * n
        hjust = data["hjust"].to_numpy() if "hjust" in data.columns else [self.default_aes["hjust"]] * n
        vjust = data["vjust"].to_numpy() if "vjust" in data.columns else [self.default_aes["vjust"]] * n

        # Border width in pt — ggplot2 ``label_size`` is mm.
        border_pt = float(self.label_size) * _PT_PER_MM

        for i, label in enumerate(labels):
            if label is None:
                continue
            ax.text(
                x[i], y[i], str(label),
                color=r_color(colour[i]),
                fontsize=float(size[i]) * _PT_PER_MM,
                rotation=float(angle[i]),
                ha=_hjust_to_ha(float(hjust[i])),
                va=_vjust_to_va(float(vjust[i])),
                bbox=dict(
                    boxstyle=f"round,pad={float(self.label_padding)},"
                             f"rounding_size={float(self.label_r)}",
                    facecolor=r_color(fill[i]),
                    edgecolor=r_color(colour[i]),
                    linewidth=border_pt,
                ),
            )


def geom_label(mapping=None, data=None, *, stat="identity", position="identity",
               na_rm=False, label_padding=0.25, label_r=0.15, label_size=0.25,
               **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "angle", "hjust",
                           "vjust", "alpha", "family", "fontface", "label"}}

    return Layer(
        geom=GeomLabel(label_padding=label_padding, label_r=label_r,
                       label_size=label_size),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        na_rm=na_rm,
    )
