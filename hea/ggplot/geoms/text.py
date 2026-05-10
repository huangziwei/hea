"""``geom_text()`` and ``geom_label()`` — text annotations at (x, y).

Smaller surface than ggplot2's full geom_text: we ship the common
arguments (``hjust``, ``vjust``, ``angle``, ``size``, ``colour``,
``family``, ``fontface``). ``geom_label`` adds a rounded background
box; surface mirrors ggplot2's: ``fill``, ``label_padding``, ``label_r``
(corner radius), ``label_size`` (border line width).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..aes import split_layer_kwargs
from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomText(Geom):
    # Mirrors ggplot2's ``GeomText$default_aes`` (R/geom-text.R). ``size``
    # is in MM (ggplot2's text-size convention) — 11 pt × 25.4 / 72.27 ≈
    # 3.88 mm. ``lineheight`` is the ratio of line spacing to font size.
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 3.88,
        "angle": 0.0,
        "hjust": 0.5,
        "vjust": 0.5,
        "alpha": 1.0,
        "family": "",
        "fontface": "plain",
        "lineheight": 1.2,
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

        x_range, y_range = _axis_ranges_for_inward(ax, x, y, hjust, vjust)

        for i, label in enumerate(labels):
            if label is None:
                continue
            h_raw = hjust[i] if hjust is not None else self.default_aes["hjust"]
            v_raw = vjust[i] if vjust is not None else self.default_aes["vjust"]
            kwargs = {
                "color": r_color(colour[i]) if colour else self.default_aes["colour"],
                "fontsize": float(size[i]) * _PT_PER_MM if size is not None
                            else self.default_aes["size"] * _PT_PER_MM,
                "rotation": float(angle[i]) if angle is not None
                            else self.default_aes["angle"],
                "ha": _hjust_to_ha(_resolve_just(h_raw, x[i], x_range, axis="h")),
                "va": _vjust_to_va(_resolve_just(v_raw, y[i], y_range, axis="v")),
            }
            ax.text(x[i], y[i], str(label), **kwargs)


# ggplot2's compute_just (R/utilities.R) maps these names to numerics.
_HJUST_ALIASES = {
    "left": 0.0, "right": 1.0,
    "center": 0.5, "centre": 0.5, "middle": 0.5,
}
_VJUST_ALIASES = {
    "bottom": 0.0, "top": 1.0,
    "center": 0.5, "centre": 0.5, "middle": 0.5,
}


def _resolve_just(raw, coord, axis_range, *, axis: str) -> float:
    """Resolve a justify value to numeric in [0, 1].

    Accepts numerics (passed through), static strings (``"left"``,
    ``"right"``, ``"top"``, ``"bottom"``, ``"center"``/``"centre"``/
    ``"middle"``), and the position-aware ``"inward"`` / ``"outward"``
    (resolved per ggplot2's ``compute_just``: split on the panel midpoint).
    Unknown strings fall back to 0.5 (matches ggplot2's silent-NA behaviour).
    """
    aliases = _HJUST_ALIASES if axis == "h" else _VJUST_ALIASES
    if isinstance(raw, str):
        if raw in aliases:
            return aliases[raw]
        if raw in ("inward", "outward"):
            lo, hi = axis_range
            if hi <= lo:
                return 0.5
            mid = 0.5 * (lo + hi)
            # ggplot2's just_dir splits at the panel midpoint with a small tol.
            tol = 1e-6 * (hi - lo)
            if coord < mid - tol:
                return 0.0 if raw == "inward" else 1.0
            if coord > mid + tol:
                return 1.0 if raw == "inward" else 0.0
            return 0.5
        return 0.5  # unknown alias → ggplot2 silently uses NA → 0.5
    return float(raw)


def _axis_ranges_for_inward(ax, x, y, hjust, vjust) -> tuple:
    """Pick axis ranges for resolving inward/outward.

    ``draw_panel`` runs before scales call ``apply_to_axis``, so
    ``ax.get_xlim()`` may still be matplotlib's auto view from artists
    drawn earlier in the layer stack. That's the right reference for
    inward/outward — same panel the data lands in. If no other layer
    has drawn yet we fall back to the geom's own data range. If neither
    hjust nor vjust uses ``inward``/``outward`` we skip the work."""
    needs_h = _has_inward_outward(hjust)
    needs_v = _has_inward_outward(vjust)
    if not (needs_h or needs_v):
        return ((0.0, 1.0), (0.0, 1.0))

    def _range(axis_lim, coords):
        lo, hi = axis_lim
        if hi > lo and not (lo == 0.0 and hi == 1.0):
            return (float(lo), float(hi))
        # Fall back to layer data extent.
        if len(coords) == 0:
            return (0.0, 1.0)
        try:
            return (float(coords.min()), float(coords.max()))
        except (TypeError, ValueError):
            return (0.0, 1.0)

    return (_range(ax.get_xlim(), x), _range(ax.get_ylim(), y))


def _has_inward_outward(arr) -> bool:
    if arr is None:
        return False
    try:
        return any(v in ("inward", "outward") for v in arr if isinstance(v, str))
    except TypeError:
        return False


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

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomText(),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        na_rm=na_rm,
    )


@dataclass
class GeomLabel(GeomText):
    """``geom_text`` plus a rounded background box (ggplot2 ``geom_label``).

    Mirrors ggplot2's ``GeomLabel$default_aes`` (R/geom-label.R) — adds
    ``fill = "white"``, ``linewidth = 0.5 * borderwidth = 0.25``,
    ``linetype = "solid"`` to ``GeomText``'s defaults.
    """
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
        "lineheight": 1.2,
        "linewidth": 0.25,
        "linetype": "solid",
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

        x_range, y_range = _axis_ranges_for_inward(ax, x, y, hjust, vjust)

        for i, label in enumerate(labels):
            if label is None:
                continue
            ax.text(
                x[i], y[i], str(label),
                color=r_color(colour[i]),
                fontsize=float(size[i]) * _PT_PER_MM,
                rotation=float(angle[i]),
                ha=_hjust_to_ha(_resolve_just(hjust[i], x[i], x_range, axis="h")),
                va=_vjust_to_va(_resolve_just(vjust[i], y[i], y_range, axis="v")),
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

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomLabel(label_padding=label_padding, label_r=label_r,
                       label_size=label_size),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        na_rm=na_rm,
    )
