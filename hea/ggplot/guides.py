"""Guide system — legends (3.1), colorbars (3.2), and (later) axes (3.3).

For each non-positional discrete scale, build a legend entry per trained
level. ggplot2's auto-merge rule kicks in when two scales share the same
aes source column AND the same set of levels: those scales collapse into
one legend whose entries combine all the aesthetic glyphs (e.g.
``aes(colour=species, shape=species)`` yields one legend, with each entry
showing both a coloured swatch and a shape marker).

Continuous colour/fill scales render as a :func:`matplotlib.figure.Figure.colorbar`
instead of a legend; the colormap is built from the scale's palette.
Continuous size/alpha (sample-point legends) are deferred polish.

The renderer reads ``theme(legend.position=...)`` (one of ``"right"``,
``"left"``, ``"top"``, ``"bottom"``, ``"none"``) and
``theme(legend.direction=...)`` (``"vertical"`` / ``"horizontal"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from .scales.color_continuous import ScaleContinuousColor
from .scales.discrete import ScaleDiscreteColor, ScaleIdentity


# Aesthetics that contribute to ``guide_legend`` entries.
_LEGEND_AES = ("colour", "fill", "shape", "linetype", "size", "alpha")


@dataclass
class LegendGroup:
    """One legend block: title plus N entries, where each entry has values
    for each aesthetic in ``aes_values``."""

    title: str
    levels: list = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    # ``aes_values[aesthetic]`` is a list of mapped values, one per level
    # (e.g. ``{"colour": ["#FF0000", "#00FF00"], "shape": ["o", "^"]}``).
    aes_values: dict = field(default_factory=dict)


def build_legend_groups(plot, build_output) -> list[LegendGroup]:
    """Walk the trained scales, return one :class:`LegendGroup` per
    distinct (source-column, levels) pair. Each group merges all
    aesthetics that share that key — ggplot2's ``guides_merge`` semantics.

    Skips:
    * positional aesthetics (``x``/``y``)
    * :class:`ScaleIdentity` (data already holds drawable values)
    * scales without trained ``levels`` (untrained discrete or continuous)
    """
    aes_source = build_output.aes_source or {}
    plot_labels = getattr(plot, "labels", {}) or {}
    scales = build_output.scales

    # Group key → LegendGroup. Use insertion order so the visual order of
    # the legends matches the user's aes order.
    groups: dict[tuple, LegendGroup] = {}
    seen_scales: set[int] = set()

    for aes_name, scale in scales.items():
        if aes_name in ("x", "y") or aes_name not in _LEGEND_AES:
            continue
        if id(scale) in seen_scales:
            continue
        if isinstance(scale, ScaleIdentity):
            continue
        if not isinstance(scale, ScaleDiscreteColor):
            # Continuous scales for non-positional aes get a colourbar (3.2)
            # or sample-point legend; both deferred.
            continue
        if not getattr(scale, "levels", None):
            continue

        seen_scales.add(id(scale))
        source = aes_source.get(aes_name, aes_name)
        levels = list(scale.levels)
        key = (source, tuple(levels))

        # Title precedence: labs() override > scale.name > aes-source > aes name
        title = (plot_labels.get(aes_name)
                 or scale.name
                 or aes_source.get(aes_name)
                 or aes_name)

        if key not in groups:
            groups[key] = LegendGroup(
                title=title,
                levels=levels,
                labels=[str(level) for level in levels],
                aes_values={},
            )

        groups[key].aes_values[aes_name] = _scale_mapped_values(scale, levels)

    return list(groups.values())


def _scale_mapped_values(scale, levels):
    """Return per-level mapped values from the scale.

    Calls the palette directly when available — :meth:`ScaleDiscreteColor.map`
    hardcodes ``return_dtype=pl.Utf8`` which would coerce numeric outputs
    (size, alpha) to strings.
    """
    n = len(levels)
    if isinstance(getattr(scale, "values", None), dict):
        return [scale.values.get(level) for level in levels]
    palette = getattr(scale, "palette", None)
    if palette is not None:
        return list(palette(n))
    # Fallback (no palette set yet): defer to scale.map.
    mapped = scale.map(pl.Series(levels))
    return mapped.to_list() if hasattr(mapped, "to_list") else list(mapped)


@dataclass
class ColorbarSpec:
    """Continuous colour/fill scale → colorbar info: cmap, value range,
    title, and the aesthetic name (for theme overrides)."""

    title: str
    aesthetic: str
    vmin: float
    vmax: float
    palette: object  # callable: array → list of hex


def build_colorbar_specs(plot, build_output) -> list[ColorbarSpec]:
    """Walk continuous colour/fill scales, return one :class:`ColorbarSpec`
    per. Scales for size/alpha use the same ``ScaleContinuousColor`` class
    but should render as sample-point legends, not colourbars — those are
    skipped here (deferred polish)."""
    aes_source = build_output.aes_source or {}
    plot_labels = getattr(plot, "labels", {}) or {}
    scales = build_output.scales

    specs: list[ColorbarSpec] = []
    seen: set[int] = set()
    for aes_name, scale in scales.items():
        if aes_name not in ("colour", "fill"):
            continue
        if id(scale) in seen:
            continue
        if not isinstance(scale, ScaleContinuousColor):
            continue
        if scale.range_ is None or scale.palette is None:
            continue
        seen.add(id(scale))
        title = (plot_labels.get(aes_name)
                 or scale.name
                 or aes_source.get(aes_name)
                 or aes_name)
        lo, hi = scale.range_
        specs.append(ColorbarSpec(
            title=title, aesthetic=aes_name,
            vmin=float(lo), vmax=float(hi), palette=scale.palette,
        ))
    return specs


def _palette_to_cmap(palette, n: int = 256, name: str = "hea_pal"):
    """Build a matplotlib colormap from a hea palette callable. The palette
    receives values in [0, 1] and returns hex strings."""
    from matplotlib.colors import LinearSegmentedColormap

    samples = palette(np.linspace(0, 1, n))
    return LinearSegmentedColormap.from_list(name, list(samples), N=n)


def apply_legends(fig, axes_list, plot, build_output, *,
                   colorbar_caxes: list | None = None) -> None:
    """Render legend groups + colorbars onto the first axes using
    ``theme(legend.position=...)`` / ``theme(legend.direction=...)``.

    Multiple groups stack vertically (right/left) or horizontally
    (top/bottom). With a single group the offset math is a no-op.

    Colorbars are placed via matplotlib's ``fig.colorbar``. By default
    that *shrinks* the parent axes to make room — which is a panel-
    alignment hazard for patchwork composition. Pass ``colorbar_caxes``
    to provide pre-allocated dedicated axes (one per colorbar spec), in
    which case ``fig.colorbar(cax=...)`` is used instead and the host
    axes is left untouched.
    """
    pos = plot.theme.get("legend.position") or "right"
    if pos == "none":
        return

    groups = build_legend_groups(plot, build_output)
    cbar_specs = build_colorbar_specs(plot, build_output)
    if not groups and not cbar_specs:
        return

    direction = plot.theme.get("legend.direction") or (
        "vertical" if pos in ("right", "left") else "horizontal"
    )

    target = axes_list[0]

    # Colorbars first — they reserve space on the figure edge.
    for i, spec in enumerate(cbar_specs):
        cax = colorbar_caxes[i] if colorbar_caxes and i < len(colorbar_caxes) else None
        _render_colorbar(fig, axes_list, target, spec, pos, direction, cax=cax)

    # Then discrete legends (with stacking offsets if multiple).
    legends = []
    for i, group in enumerate(groups):
        handles = [_make_handle(group, j) for j in range(len(group.levels))]
        ncols = len(handles) if direction == "horizontal" else 1
        kw = _legend_position_kwargs(pos, i, len(groups))
        leg = target.legend(
            handles, group.labels, title=group.title, ncols=ncols, **kw,
        )
        legends.append(leg)
        # ax.legend replaces the previous legend artist on each call. To keep
        # earlier ones around when stacking, re-add them via add_artist.
        if i < len(groups) - 1:
            target.add_artist(leg)


def _render_colorbar(fig, axes_list, target, spec: ColorbarSpec,
                     pos: str, direction: str, *,
                     cax=None) -> None:
    """Render one colorbar with theme-aware location.

    ``cax``: a pre-allocated dedicated axes. When given, the colorbar
    fills it exactly and the host ``axes_list`` is left untouched
    (block-engine path); otherwise matplotlib's auto-shrink applies.
    """
    import matplotlib as mpl

    cmap = _palette_to_cmap(spec.palette, name=f"hea_{spec.aesthetic}")
    norm = mpl.colors.Normalize(vmin=spec.vmin, vmax=spec.vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    # matplotlib's location accepts: "left"/"right"/"top"/"bottom"
    # — same vocab as ggplot2's legend.position.
    location = pos if pos in ("right", "left", "top", "bottom") else "right"
    orientation = "horizontal" if location in ("top", "bottom") else "vertical"
    if cax is not None:
        cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    else:
        cb = fig.colorbar(
            mappable, ax=axes_list, location=location,
            orientation=orientation, shrink=0.6, pad=0.05,
        )
    cb.set_label(spec.title)


def _make_handle(group: LegendGroup, idx: int) -> Line2D:
    """Build one legend handle with all the group's aesthetics applied at
    the i-th level. Defaults to a small black dot; aesthetics override.
    """
    kwargs = {
        "marker": "o",
        "linestyle": "",
        "color": "black",
        "markersize": 6,
        "markerfacecolor": "black",
        "markeredgecolor": "black",
    }
    aes_values = group.aes_values

    # Linetype implies a line glyph (no marker).
    if "linetype" in aes_values:
        from ..plot._util import r_lty
        kwargs["linestyle"] = r_lty(aes_values["linetype"][idx]) or "-"
        kwargs["marker"] = ""

    if "shape" in aes_values:
        kwargs["marker"] = aes_values["shape"][idx]
        if "linetype" not in aes_values:
            kwargs["linestyle"] = ""

    if "colour" in aes_values:
        c = aes_values["colour"][idx]
        if c is not None:
            kwargs["color"] = c
            kwargs["markeredgecolor"] = c
            kwargs["markerfacecolor"] = c

    if "fill" in aes_values:
        f = aes_values["fill"][idx]
        if f is not None:
            kwargs["markerfacecolor"] = f

    if "size" in aes_values:
        s = aes_values["size"][idx]
        if s is not None:
            kwargs["markersize"] = float(s)

    if "alpha" in aes_values:
        a = aes_values["alpha"][idx]
        if a is not None:
            kwargs["alpha"] = float(a)

    return Line2D([0], [0], **kwargs)


def _legend_position_kwargs(pos: str, idx: int, total: int) -> dict:
    """Position kwargs for the i-th legend in a stack of ``total``.

    For right/left: stack vertically; legend i sits below legend i-1.
    For top/bottom: stack horizontally; legend i sits to the right of i-1.
    Spacing is approximate — ggplot2 measures legend extents and packs;
    we use a fixed offset that's good enough for the common 1-2 legend case.
    """
    spacing = 0.18
    centre = 0.5 + (total - 1) * spacing / 2
    y_for_idx = centre - idx * spacing
    x_for_idx = 0.5 - (total - 1) * spacing / 2 + idx * spacing

    if pos == "right":
        return {"loc": "center left", "bbox_to_anchor": (1.02, y_for_idx)}
    if pos == "left":
        return {"loc": "center right", "bbox_to_anchor": (-0.02, y_for_idx)}
    if pos == "top":
        return {"loc": "lower center", "bbox_to_anchor": (x_for_idx, 1.02)}
    if pos == "bottom":
        return {"loc": "upper center", "bbox_to_anchor": (x_for_idx, -0.02)}
    return {}


# ---------------------------------------------------------------------------
# guide_legend / guides factories
# ---------------------------------------------------------------------------

@dataclass
class GuideLegend:
    """ggplot2's ``guide_legend()``. Currently a metadata holder — auto-build
    from trained scales handles the rendering. Per-aesthetic overrides
    (``ncol``/``reverse``/``override.aes``) ride on this struct for the day
    we wire them in."""

    title: str | None = None
    ncol: int | None = None
    nrow: int | None = None
    reverse: bool = False
    override_aes: dict = field(default_factory=dict)


def guide_legend(*, title=None, ncol=None, nrow=None, reverse=False,
                 override_aes=None):
    return GuideLegend(
        title=title, ncol=ncol, nrow=nrow, reverse=reverse,
        override_aes=override_aes or {},
    )


@dataclass
class GuideAxis:
    """ggplot2's ``guide_axis()``. Controls positional axis appearance:

    * ``angle`` — rotate tick labels (degrees, counter-clockwise).
    * ``n_dodge`` — split labels across this many rows to avoid overlap.
      v1 only honours 1; values > 1 are accepted but unimplemented.
    * ``position`` — ``"bottom"``/``"top"`` for x, ``"left"``/``"right"`` for y.
      v1 accepts but doesn't reposition the axis (matplotlib default).
    """

    angle: float | None = None
    n_dodge: int = 1
    position: str | None = None


def guide_axis(*, angle=None, n_dodge=1, position=None):
    return GuideAxis(angle=angle, n_dodge=n_dodge, position=position)


@dataclass
class Guides:
    """Per-aesthetic guide overrides — added to a ggplot via ``+``."""

    overrides: dict = field(default_factory=dict)


def guides(**kwargs):
    """Override the default guide for one or more aesthetics. Pass
    ``guide_legend()``, ``guide_axis()``, etc. keyed by aes name
    (e.g. ``guides(x=guide_axis(angle=45))``)."""
    return Guides(overrides=dict(kwargs))


def apply_axis_guides(axes_list, plot) -> None:
    """Apply ``guide_axis`` rotation overrides to every axis. Called after
    the positional scales have set ticks. Reads from ``plot.guide_overrides``
    AND from ``theme(axis.text.x/y = element_text(angle=...))`` — either
    surface produces the same effect."""
    overrides = getattr(plot, "guide_overrides", {}) or {}
    theme = plot.theme

    for ax in axes_list:
        for axis_name in ("x", "y"):
            angle = _resolve_axis_angle(axis_name, overrides, theme)
            if angle is None:
                continue
            ax.tick_params(axis=axis_name, rotation=float(angle))


def _resolve_axis_angle(axis_name: str, overrides: dict, theme) -> float | None:
    """guide_axis(angle=) wins over theme(axis_text_x=element_text(angle=))."""
    g = overrides.get(axis_name)
    if isinstance(g, GuideAxis) and g.angle is not None:
        return g.angle
    elem = theme.get(f"axis.text.{axis_name}")
    if elem is None:
        elem = theme.get("axis.text")
    if elem is not None and getattr(elem, "angle", None) is not None:
        return elem.angle
    return None
