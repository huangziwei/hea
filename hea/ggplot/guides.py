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
from matplotlib.legend_handler import HandlerLine2D, HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch as _MplPatch
from matplotlib.patches import Rectangle as _MplRect

from ._util import r_color
from .scales.color_continuous import ScaleContinuousColor
from .scales.discrete import ScaleDiscreteColor, ScaleIdentity
from .theme import element_blank, element_rect, element_text


# ggplot2 sizes are in mm; matplotlib widths/lengths are in pt. R's TeX
# convention: 72.27 pt/inch, 25.4 mm/inch → ≈ 2.8454 pt/mm.
_PT_PER_MM = 72.27 / 25.4


def _add_key_bg(handlebox, fontsize, *, fc, ec, lw):
    """Paint ``legend.key`` (panel-colour rect) into ``handlebox`` first.

    Returns the (xdescent, ydescent, width, height, trans) tuple so the
    caller can position glyphs in the same coordinate space. Used by the
    Line2D and Patch key handlers to keep a ggplot2-style gray bg behind
    each legend key.
    """
    xdescent = handlebox.xdescent
    ydescent = handlebox.ydescent
    width = handlebox.width
    height = handlebox.height
    trans = handlebox.get_transform()
    bg = _MplRect(
        (-xdescent, -ydescent), width, height,
        facecolor=fc, edgecolor=ec, linewidth=lw, transform=trans,
    )
    handlebox.add_artist(bg)
    return xdescent, ydescent, width, height, trans


class _HandlerLine2DKeyBg(HandlerLine2D):
    """Wraps the standard ``Line2D`` legend handler with a panel-colour
    rectangle behind each glyph — matches ggplot2's ``legend.key`` element.
    Without this, matplotlib draws handles on a transparent background.

    matplotlib's ``DrawingArea`` paints children in *insertion order*
    (zorder is ignored within an OffsetBox), so we override
    ``legend_artist`` directly: add the bg first so it paints under the
    glyph, then the glyph on top, while still returning the glyph as the
    primary handle for ``Legend.legend_handles``.
    """

    def __init__(self, *, facecolor, edgecolor="none", linewidth=0.0, **kw):
        super().__init__(**kw)
        self._key_fc = facecolor
        self._key_ec = edgecolor
        self._key_lw = linewidth

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        xdescent, ydescent, width, height, trans = _add_key_bg(
            handlebox, fontsize,
            fc=self._key_fc, ec=self._key_ec, lw=self._key_lw,
        )
        glyphs = self.create_artists(
            legend, orig_handle, xdescent, ydescent, width, height,
            fontsize, trans,
        )
        for g in glyphs:
            handlebox.add_artist(g)
        return glyphs[0] if glyphs else None


class _HandlerPatchKeyBg(HandlerPatch):
    """``HandlerPatch`` with ``legend.key`` bg — same trick as
    :class:`_HandlerLine2DKeyBg`, applied to ``Patch`` legend handles
    (the polygon key glyph for ``geom_bar`` etc.).
    """

    def __init__(self, *, facecolor, edgecolor="none", linewidth=0.0, **kw):
        super().__init__(**kw)
        self._key_fc = facecolor
        self._key_ec = edgecolor
        self._key_lw = linewidth

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        xdescent, ydescent, width, height, trans = _add_key_bg(
            handlebox, fontsize,
            fc=self._key_fc, ec=self._key_ec, lw=self._key_lw,
        )
        glyphs = self.create_artists(
            legend, orig_handle, xdescent, ydescent, width, height,
            fontsize, trans,
        )
        for g in glyphs:
            handlebox.add_artist(g)
        return glyphs[0] if glyphs else None


def _legend_key_handler(theme):
    """Return a ``handler_map`` that paints ``legend.key`` behind each glyph,
    or ``None`` if the theme blanks/omits the key.

    Two handler entries: ``Line2D`` (point/path glyphs) and ``Patch``
    (polygon glyph for bar/ribbon/etc.). ggplot2's key element shows
    behind both kinds.
    """
    elem = theme.get("legend.key")
    if isinstance(elem, element_blank):
        return None
    if not isinstance(elem, element_rect) or not elem.fill:
        return None
    fc = r_color(elem.fill)
    ec = r_color(elem.colour) if elem.colour else "none"
    lw = (elem.size * _PT_PER_MM) if (elem.colour and elem.size) else 0.0
    return {
        Line2D: _HandlerLine2DKeyBg(facecolor=fc, edgecolor=ec, linewidth=lw),
        _MplPatch: _HandlerPatchKeyBg(facecolor=fc, edgecolor=ec, linewidth=lw),
    }


def _legend_title_alignment(theme) -> str:
    """Map ``legend.title``'s ``hjust`` to matplotlib's ``alignment`` arg.
    ggplot2's default is left-aligned; matplotlib's is centered."""
    elem = theme.get("legend.title")
    if not isinstance(elem, element_text) or elem.hjust is None:
        return "left"
    h = float(elem.hjust)
    if h <= 0:
        return "left"
    if h >= 1:
        return "right"
    return "center"


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
    # Aesthetic names this group represents — multi-entry when multiple
    # aes share the same source/levels and merge into one legend (e.g.
    # ``aes(colour=g, shape=g)``). Used to look up ``guides(colour=...)``
    # overrides; an override on any merged aes applies to the group.
    aes_names: list[str] = field(default_factory=list)
    # Geom that contributed to this legend — drives the legend key glyph
    # (rectangle for bar, line for path, circle for point). ``"point"``
    # is the safe default when no contributor is found.
    key_glyph: str = "point"
    # Layer-level constants the user passed via geom kwargs
    # (``geom_bar(alpha=1/5, fill=None)``). Applied on top of the
    # scale-mapped values when building the key, so the legend reflects
    # the actual layer style — matches ggplot2's ``draw_key_*`` reading
    # of layer aesthetics.
    layer_aes_params: dict = field(default_factory=dict)
    # Geom-level defaults (``geom_bar.default_aes`` etc.). Used as a
    # fallback when an aesthetic isn't mapped and isn't in
    # ``aes_params``, so e.g. ``aes(colour=drv)`` on ``geom_bar`` shows
    # legend keys with grey35 fill (the bar default), matching R.
    layer_default_aes: dict = field(default_factory=dict)


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

        # Title precedence: labs() override > scale.name > aes-source > aes name.
        # ``scale.name`` may be the ``_NAME_MISSING`` sentinel (factory wasn't
        # called with name=) — treat that as "no override". An explicit
        # ``name=None`` (suppress) yields ``""`` here, which is falsy and
        # falls through to the source name — for legend titles that's fine
        # since no axis-style suppression UI exists yet.
        title = (plot_labels.get(aes_name)
                 or _scale_name_or_none(scale)
                 or aes_source.get(aes_name)
                 or aes_name)

        contributor_idx, contributor = _find_layer_for_aes(
            plot, aes_name,
            layer_mappings=build_output.layer_mappings,
            visible_only=True,
        )
        if contributor is None:
            # Every contributing layer has ``show_legend=False`` — ggplot2's
            # rule: the scale is still trained, but no guide is produced.
            continue

        if key not in groups:
            geom = getattr(contributor, "geom", None)
            # Effective ``aes_params`` (post-promotion) drops column-shaped
            # entries that were promoted into the mapping; those aren't
            # constants for the key glyph. Falls back to the raw layer
            # field for legacy callers without a build_output.
            eff_params_list = getattr(build_output, "layer_aes_params", None)
            if (eff_params_list is not None
                    and contributor_idx is not None
                    and contributor_idx < len(eff_params_list)):
                layer_params = dict(eff_params_list[contributor_idx] or {})
            else:
                layer_params = dict(getattr(contributor, "aes_params", {}))
            groups[key] = LegendGroup(
                title=title,
                levels=levels,
                labels=_resolve_discrete_labels(scale, levels),
                aes_values={},
                key_glyph=getattr(geom, "key_glyph", "point") if geom else "point",
                layer_aes_params=layer_params,
                layer_default_aes=(dict(getattr(geom, "default_aes", {}))
                                    if geom is not None else {}),
            )

        groups[key].aes_values[aes_name] = _scale_mapped_values(scale, levels)
        if aes_name not in groups[key].aes_names:
            groups[key].aes_names.append(aes_name)

    return list(groups.values())


def _stat_default_label_for(plot, aes_name):
    """If any layer's stat declares a ``default_<aes>_label`` (e.g.
    ``StatBin2d.default_fill_label = "count"``), return it.

    Mirrors ggplot2's behaviour: ``geom_bin2d()`` auto-maps fill to
    ``after_stat(count)`` and the colorbar reads "count" by default,
    not the bare aesthetic name "fill".
    """
    attr = f"default_{aes_name}_label"
    for layer in getattr(plot, "layers", []):
        stat = getattr(layer, "stat", None)
        if stat is None:
            continue
        # If the user explicitly mapped this aesthetic on the layer,
        # don't override their choice with the stat default.
        mapping = getattr(layer, "mapping", None) or {}
        if aes_name in mapping:
            continue
        tag = getattr(stat, attr, None)
        if tag:
            return tag
    return None


def _scale_name_or_none(scale):
    """Return ``scale.name`` if it was explicitly set, else ``None``.

    Filters the ``_NAME_MISSING`` sentinel (factory wasn't called with
    name=) so legend / colorbar title fallbacks treat it as "no override".
    """
    from .scales.scale import _NAME_MISSING

    nm = getattr(scale, "name", None)
    if nm is _NAME_MISSING:
        return None
    return nm


def _resolve_discrete_labels(scale, levels):
    """Apply ``scale.labels`` to a discrete scale's levels for legend display.

    Mirrors ggplot2's ``labels`` semantics on discrete scales:

    * ``"default"`` (or attr missing) → ``str(level)`` for each level.
    * ``None`` → blank labels (``labels = NULL`` in R hides the legend
      entries' text but keeps the keys).
    * dict → per-level lookup; missing keys fall back to ``str(level)``,
      matching R's named-vector behaviour for partial overrides.
    * callable → invoked once with the levels list, return the labels.
    * list / tuple → used positionally; padded / truncated to match
      ``len(levels)``.
    """
    labels = getattr(scale, "labels", "default")
    if isinstance(labels, str) and labels == "default":
        return [str(level) for level in levels]
    if labels is None:
        return ["" for _ in levels]
    if callable(labels):
        return [str(x) for x in labels(levels)]
    if isinstance(labels, dict):
        return [str(labels.get(level, level)) for level in levels]
    out = [str(x) for x in labels]
    if len(out) < len(levels):
        out += [str(levels[i]) for i in range(len(out), len(levels))]
    return out[:len(levels)]


def _find_layer_for_aes(plot, aes_name, *, layer_mappings=None,
                         visible_only: bool = False):
    """Return ``(idx, layer)`` for the first layer whose mapping uses
    ``aes_name`` — or ``(None, None)`` if none does.

    Looks at the layer's own mapping first, then falls back to the plot-
    level mapping (when ``inherit_aes=True``) — matches ggplot2's lookup
    order. The contributing layer drives the legend key glyph + any
    layer-level aes constants the user supplied via geom kwargs.

    ``layer_mappings`` (preferred): per-layer effective mapping from
    ``BuildOutput.layer_mappings`` — already includes column-shaped
    ``aes_params`` promoted by ``_promote_string_aes_params``. Without
    this, ``geom_point(color="species")`` would have an empty
    ``layer.mapping`` and the legend would be silently dropped.

    ``visible_only=True`` skips layers with ``show_legend=False`` —
    ggplot2's rule for whether a scale should produce a guide.
    """
    plot_mapping = getattr(plot, "mapping", None) or {}
    for i, layer in enumerate(plot.layers):
        if visible_only and getattr(layer, "show_legend", True) is False:
            continue
        if layer_mappings is not None and i < len(layer_mappings):
            layer_mapping = layer_mappings[i] or {}
        else:
            layer_mapping = getattr(layer, "mapping", None) or {}
        if aes_name in layer_mapping:
            return i, layer
        if getattr(layer, "inherit_aes", True) and aes_name in plot_mapping:
            return i, layer
    return None, None


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
                 or _scale_name_or_none(scale)
                 or aes_source.get(aes_name)
                 or _stat_default_label_for(plot, aes_name)
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
                   colorbar_caxes: list | None = None,
                   legend_host_axes: list | None = None) -> None:
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
    handler_map = _legend_key_handler(plot.theme)
    alignment = _legend_title_alignment(plot.theme)

    target = axes_list[0]

    # Colorbars first — they reserve space on the figure edge.
    for i, spec in enumerate(cbar_specs):
        cax = colorbar_caxes[i] if colorbar_caxes and i < len(colorbar_caxes) else None
        _render_colorbar(fig, axes_list, target, spec, pos, direction, cax=cax)

    # Then discrete legends. When the block engine pre-allocates a host
    # axes per legend group (``legend_host_axes``), render INTO that
    # axes — the legend stays bounded by the host's bbox, so it can't
    # extend into a sibling plot's panel area in a patchwork
    # composition. Falls back to ``target.legend(bbox_to_anchor=...)``
    # for legacy callers without the block engine.
    legends = []
    for i, group in enumerate(groups):
        handles = [_make_handle(group, j) for j in range(len(group.levels))]
        labels = list(group.labels)
        title = group.title

        # Apply ``guides(<aes>=guide_legend(...))`` overrides.
        gl = _find_guide_legend(group, plot)
        if gl is not None:
            if gl.title is not None:
                title = gl.title
            if gl.reverse:
                handles = list(reversed(handles))
                labels = list(reversed(labels))
            for h in handles:
                for k, v in (gl.override_aes or {}).items():
                    _apply_handle_override(h, k, v, group.key_glyph)

        # Resolve column count: explicit ``ncol`` wins; ``nrow`` is
        # converted to ``ceil(n/nrow)``; otherwise default by direction.
        if gl is not None and gl.ncol is not None:
            ncols = max(1, int(gl.ncol))
        elif gl is not None and gl.nrow is not None:
            from math import ceil
            ncols = max(1, ceil(len(handles) / max(1, int(gl.nrow))))
        else:
            ncols = len(handles) if direction == "horizontal" else 1

        host = (legend_host_axes[i]
                if legend_host_axes and i < len(legend_host_axes)
                else None)
        # Per-glyph spacing: polygon keys are filled rectangles that
        # *cover* the panel-colour bg, so they need visible vertical gaps
        # between rows or adjacent colour blocks would butt together.
        # Point/path keys are small glyphs sitting on the bg, so
        # ``labelspacing=0`` (between rows) and ``columnspacing=0``
        # (between columns) let the per-key bgs abut into one
        # continuous gray block — matches ggplot2 regardless of whether
        # the user added ``guides(... = guide_legend(nrow=...))``. Without
        # the columnspacing override, multi-column / wrapped layouts
        # split the bg into separate cells and the legend "style" visibly
        # changes when guides() introduces wrapping.
        if group.key_glyph == "polygon":
            labelspacing = 0.4
            columnspacing = 1.0  # match matplotlib default-ish
        else:
            labelspacing = 0.0
            columnspacing = 0.0
        # ggplot2's ``legend.key.size = unit(1.2, "lines")`` produces
        # square keys. matplotlib's ``handlelength`` and ``handleheight``
        # are both in font-size units but use different reference
        # dimensions (length is per-em horizontal, height is per-em
        # vertical incl. line spacing), so the same numeric value yields
        # a non-square box — empirically ``(1.2, 1.5)`` gives a square
        # ~12×12 px bbox at the default 10 pt fontsize.
        sizing = {"handlelength": 1.2, "handleheight": 1.5,
                  "labelspacing": labelspacing,
                  "columnspacing": columnspacing}
        if host is not None:
            host.set_axis_off()
            leg = host.legend(
                handles, labels, title=title, ncols=ncols,
                loc="center left", bbox_to_anchor=(0.0, 0.5),
                frameon=False, alignment=alignment,
                handler_map=handler_map, **sizing,
            )
        else:
            kw = _legend_position_kwargs(pos, i, len(groups))
            leg = target.legend(
                handles, labels, title=title, ncols=ncols,
                frameon=False, alignment=alignment,
                handler_map=handler_map, **sizing, **kw,
            )
        legends.append(leg)
        if host is None and i < len(groups) - 1:
            # ax.legend replaces the previous legend artist on each call.
            # Re-add to keep earlier legends visible when stacking. Only
            # relevant for the legacy bbox_to_anchor path; the host-axes
            # path uses one host per group so no re-adding needed.
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
    # ggplot2 places the colorbar title ABOVE the bar (not to its side
    # rotated 90° — matplotlib's default ``cb.set_label``). We use
    # ``cb.ax.set_title`` so the title sits on top of the cax, matching
    # R/ggplot2's layout. For horizontal colorbars matplotlib's default
    # is already on top, but ``set_title`` works there too.
    cb.ax.set_title(spec.title, fontsize=10, pad=4)


def _find_guide_legend(group, plot):
    """Return the :class:`GuideLegend` from ``guides(...)`` that targets
    any aes in this group, or ``None``. Picks the *first* matching entry
    in user order — same as ggplot2's behaviour when an aes appears in
    multiple ``guides()`` calls."""
    overrides = getattr(plot, "guide_overrides", {}) or {}
    for aes_name in group.aes_names:
        if aes_name in overrides:
            spec = overrides[aes_name]
            if isinstance(spec, GuideLegend):
                return spec
            if spec is None or spec is False:
                # ``guides(colour = "none")`` / ``guides(colour = NULL)``
                # suppress — we already handle suppression at draw time
                # by returning a sentinel; treat None/False as "no extra
                # override settings" here.
                return None
    return None


def _apply_handle_override(handle, key, value, key_glyph):
    """Apply one ``override.aes`` entry to a legend handle in-place.

    Mirrors the per-glyph property mapping used by :func:`_make_point_handle`
    et al. ``size`` is interpreted in ggplot2's mm units and converted to
    matplotlib's pt; colour names go through :func:`r_color`."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Canonical names.
    if key == "color":
        key = "colour"
    if key == "linetype":
        key = "linestyle"

    if key == "alpha" and value is not None:
        try:
            handle.set_alpha(float(value))
        except (TypeError, ValueError):
            pass
        return
    if key == "colour":
        c = r_color(value) if value is not None else None
        if isinstance(handle, Line2D):
            if c is not None:
                handle.set_color(c)
                if key_glyph == "point":
                    handle.set_markeredgecolor(c)
                    handle.set_markerfacecolor(c)
        elif isinstance(handle, Patch):
            if c is not None:
                handle.set_edgecolor(c)
        return
    if key == "fill":
        c = "none" if value is None else r_color(value)
        if isinstance(handle, Line2D):
            handle.set_markerfacecolor(c)
        elif isinstance(handle, Patch):
            handle.set_facecolor(c)
        return
    if key == "size":
        sz = float(value) * _PT_PER_MM  # ggplot mm → matplotlib pt
        if isinstance(handle, Line2D):
            if key_glyph == "point":
                handle.set_markersize(sz)
            else:  # path / line
                handle.set_linewidth(sz)
        elif isinstance(handle, Patch):
            handle.set_linewidth(sz)
        return
    if key == "linewidth":
        lw = float(value) * _PT_PER_MM
        try:
            handle.set_linewidth(lw)
        except AttributeError:
            pass
        return
    if key == "linestyle":
        from ..plot._util import r_lty
        if isinstance(handle, Line2D):
            handle.set_linestyle(r_lty(value) or "-")
        return
    if key == "shape":
        if isinstance(handle, Line2D):
            handle.set_marker(value)
        return


def _make_handle(group: LegendGroup, idx: int):
    """Build one legend handle for the i-th level.

    Dispatches by the contributing geom's ``key_glyph`` (mirrors R's
    ``draw_key_*`` family). Defaults to a circle marker for unrecognised
    glyphs.
    """
    if group.key_glyph == "polygon":
        return _make_polygon_handle(group, idx)
    if group.key_glyph == "path":
        return _make_path_handle(group, idx)
    return _make_point_handle(group, idx)


def _resolve_aes(group: LegendGroup, idx: int, name: str, default):
    """Resolve aesthetic ``name`` at level ``idx``.

    Precedence (highest to lowest):
      1. layer ``aes_params`` — user-supplied constants
         (``geom_bar(alpha=0.2)``) win over the scale.
      2. scale-mapped value — when the aes is mapped (``aes(fill=drv)``).
      3. geom ``default_aes`` — fills in unmapped aes (e.g. ``fill`` is
         ``grey35`` when only ``colour=drv`` is mapped on ``geom_bar``).
         Without this fallback, the legend key would render with no
         fill while the bars show grey35 — the legend would lie.
      4. caller-supplied ``default``.
    """
    lap = group.layer_aes_params or {}
    # American spelling alias.
    for k in (name, "color" if name == "colour" else None):
        if k and k in lap:
            return lap[k]

    aes_values = group.aes_values
    if name in aes_values:
        v = aes_values[name][idx]
        if v is not None:
            return v

    lda = group.layer_default_aes or {}
    if name in lda and lda[name] is not None:
        return lda[name]

    return default


def _is_na(v):
    if v is None:
        return True
    if isinstance(v, float):
        try:
            return v != v  # NaN check
        except TypeError:
            return False
    return False


def _make_point_handle(group: LegendGroup, idx: int) -> Line2D:
    kwargs = {
        "marker": "o",
        "linestyle": "",
        "color": "black",
        "markersize": 6,
        "markerfacecolor": "black",
        "markeredgecolor": "black",
    }
    aes_values = group.aes_values

    if "linetype" in aes_values:
        from ..plot._util import r_lty
        kwargs["linestyle"] = r_lty(aes_values["linetype"][idx]) or "-"
        kwargs["marker"] = ""

    if "shape" in aes_values:
        kwargs["marker"] = aes_values["shape"][idx]
        if "linetype" not in aes_values:
            kwargs["linestyle"] = ""

    colour = _resolve_aes(group, idx, "colour", None)
    if colour is not None:
        c = r_color(colour)
        kwargs["color"] = c
        kwargs["markeredgecolor"] = c
        kwargs["markerfacecolor"] = c

    fill = _resolve_aes(group, idx, "fill", None)
    if fill is not None:
        kwargs["markerfacecolor"] = "none" if _is_na(fill) else r_color(fill)

    size = _resolve_aes(group, idx, "size", None)
    if size is not None:
        # ggplot2 ``size`` is in mm; matplotlib ``markersize`` is in pt.
        # Without the conversion, geom_point's default ``size=1.5`` renders
        # the legend marker at 1.5 pt, too small to tell shapes apart when
        # ``aes(colour=x, shape=x)`` merges into one legend.
        kwargs["markersize"] = float(size) * _PT_PER_MM

    alpha = _resolve_aes(group, idx, "alpha", None)
    if alpha is not None:
        kwargs["alpha"] = float(alpha)

    return Line2D([0], [0], **kwargs)


def _make_polygon_handle(group: LegendGroup, idx: int) -> _MplPatch:
    """Filled rectangle key — for ``geom_bar``/``rect``/``ribbon``/etc.

    Mirrors R's ``draw_key_polygon``: fills with the layer's ``fill``
    (or ``colour`` when no fill is mapped — for hollow-bar plots), edges
    with ``colour``, applies ``alpha`` to both. ``fill=NA`` (Python
    ``None``/NaN) becomes ``"none"`` (transparent), so the panel-colour
    key bg shows through, matching the ``geom_bar(fill=NA)`` look.

    Border thickness reads the layer's ``size`` aes in mm and converts
    to matplotlib points (``size_mm * 72.27/25.4``), so ``geom_bar``'s
    default ``size=0.5 mm`` renders as ~1.42 pt — the same border R
    paints on the legend key.

    Colour names go through ``r_color()`` so R-flavoured greys
    (``"grey35"``) — which appear as defaults from geom ``default_aes``
    — translate to matplotlib-friendly ``"0.35"`` form.
    """
    fill = _resolve_aes(group, idx, "fill", None)
    colour = _resolve_aes(group, idx, "colour", None)
    alpha = _resolve_aes(group, idx, "alpha", 1.0)
    size_mm = _resolve_aes(group, idx, "size", 0.5)

    facecolor = "none" if (fill is None or _is_na(fill)) else r_color(fill)
    edgecolor = "none" if (colour is None or _is_na(colour)) else r_color(colour)
    return _MplPatch(
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=float(alpha) if alpha is not None else None,
        linewidth=float(size_mm) * _PT_PER_MM,
    )


def _make_path_handle(group: LegendGroup, idx: int) -> Line2D:
    """Horizontal-line key — for ``geom_line``/``path``/``segment``/``smooth``.
    Mirrors R's ``draw_key_path``."""
    from ..plot._util import r_lty

    colour = _resolve_aes(group, idx, "colour", "black")
    alpha = _resolve_aes(group, idx, "alpha", 1.0)
    size = _resolve_aes(group, idx, "size", 0.5)
    linetype = _resolve_aes(group, idx, "linetype", "solid")

    return Line2D(
        [0], [0],
        marker="",
        color=r_color("black" if (colour is None or _is_na(colour)) else colour),
        alpha=float(alpha) if alpha is not None else None,
        linewidth=float(size) * (72.27 / 25.4),
        linestyle=r_lty(linetype) or "-",
    )


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
      Position ``i`` lands in row ``i % n_dodge`` (0-indexed).
    * ``check_overlap`` — drop labels that would overlap a previously
      kept one in spatial order along the axis. Mirrors ggplot2's
      ``guide_axis(check.overlap = TRUE)``: greedy first-fit, no
      reflow. Resolved AFTER ``angle`` / ``n_dodge`` so each strategy
      composes (e.g. dodge first, then drop any survivors that still
      collide).
    * ``position`` — ``"bottom"``/``"top"`` for x, ``"left"``/``"right"`` for y.
      v1 accepts but doesn't reposition the axis (matplotlib default).
    """

    angle: float | None = None
    n_dodge: int = 1
    check_overlap: bool = False
    position: str | None = None


def guide_axis(*, angle=None, n_dodge=1, check_overlap=False, position=None):
    return GuideAxis(
        angle=angle, n_dodge=n_dodge,
        check_overlap=check_overlap, position=position,
    )


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
    """Apply ``guide_axis`` overrides to every axis. Called after the
    positional scales have set ticks.

    Resolves three label-overflow strategies, in this order so they
    compose: angle → n_dodge → check_overlap. Angle reads from
    ``plot.guide_overrides`` AND from ``theme(axis.text.x/y =
    element_text(angle=...))`` — either surface produces the same
    rotation. n_dodge and check_overlap come from ``guide_axis(...)``
    only (matches ggplot2: theme can rotate, the guide picks the
    layout strategy).
    """
    overrides = getattr(plot, "guide_overrides", {}) or {}
    theme = plot.theme

    for ax in axes_list:
        for axis_name in ("x", "y"):
            angle = _resolve_axis_angle(axis_name, overrides, theme)
            if angle is not None:
                ax.tick_params(axis=axis_name, rotation=float(angle))

            g = overrides.get(axis_name)
            if not isinstance(g, GuideAxis):
                continue
            if g.n_dodge and g.n_dodge > 1:
                _apply_n_dodge(ax, axis_name, int(g.n_dodge))
            if g.check_overlap:
                _apply_check_overlap(ax, axis_name)


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


def _apply_n_dodge(ax, axis_name: str, n: int) -> None:
    """Stack tick labels across ``n`` rows: position ``i`` (in tick
    order) lands in row ``i % n``. Mirrors ggplot2's
    ``guide_axis(n.dodge = N)``.

    Implementation: shift each row's labels outward by an integer
    multiple of the line height via matplotlib's per-tick ``set_pad``.
    Same approach for x and y — pad always means "perpendicular
    distance from the axis spine to the label" so dodging y labels
    leftward / rightward just works.
    """
    target_axis = ax.xaxis if axis_name == "x" else ax.yaxis
    ticks = target_axis.get_major_ticks()
    if not ticks:
        return
    label = ticks[0].label1 if axis_name == "x" else ticks[0].label1
    font_size = float(label.get_fontsize())
    # 1.2× font size is matplotlib's default line spacing — gives the
    # alternate row enough clearance to sit fully below (or beside) the
    # primary row without colliding.
    line_step = font_size * 1.2
    base_pad = ticks[0].get_pad()
    for i, tick in enumerate(ticks):
        row = i % n
        tick.set_pad(base_pad + row * line_step)


def _apply_check_overlap(ax, axis_name: str) -> None:
    """Greedy first-fit overlap drop. Walks tick labels in spatial
    order along ``axis_name``; hides any whose bbox would intersect
    the previously kept one. Mirrors ggplot2's
    ``guide_axis(check.overlap = TRUE)``.

    Forces a draw so the bboxes reflect the post-rotation /
    post-n_dodge layout (otherwise this would test pre-layout
    extents and miss most collisions on rotated axes).
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    target_axis = ax.xaxis if axis_name == "x" else ax.yaxis
    labels = [lbl for lbl in target_axis.get_majorticklabels()
              if lbl.get_text() and lbl.get_visible()]
    if len(labels) <= 1:
        return

    # Spatial order along the axis (display coords).
    if axis_name == "x":
        labels.sort(key=lambda lbl: lbl.get_window_extent(renderer).x0)
    else:
        labels.sort(key=lambda lbl: lbl.get_window_extent(renderer).y0)

    last_bbox = None
    for lbl in labels:
        bbox = lbl.get_window_extent(renderer)
        if last_bbox is not None and bbox.overlaps(last_bbox):
            lbl.set_visible(False)
        else:
            last_bbox = bbox
