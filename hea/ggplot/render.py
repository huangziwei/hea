"""Render — walk per-layer drawable data into a matplotlib :class:`Figure`.

Single-panel and faceted modes share the per-axes drawing logic; faceted
mode adds a subplot grid plus per-panel data filtering. ``scales="free*"``
modes mean each panel autoscales independently (matplotlib's
``sharex``/``sharey``).
"""

from __future__ import annotations

import contextlib
import math

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from ._measure import STRIP_TEXT_SIZE_PT, strip_cell_height_in
from ._util import r_color
from .theme import element_blank, element_line, element_rect, element_text

_PT_PER_MM = 72.27 / 25.4


def render(plot, build_output, ax=None, subplotspec=None) -> plt.Figure:
    """Render into a user-supplied ``ax`` or ``subplotspec``.

    For standalone plotting and patchwork composition use
    :func:`hea.ggplot._block.render_block` /
    :func:`hea.ggplot._block.render_super_block` — those own figure
    sizing and the gridspec layout. This entry point exists for users
    integrating ggplot output into a custom matplotlib layout.
    """
    layout = build_output.layout
    n_panels = 1 if layout is None else len(layout)

    if n_panels <= 1:
        return _render_single(plot, build_output, ax=ax, subplotspec=subplotspec)
    if ax is not None:
        return _render_single(plot, build_output, ax=ax, subplotspec=None)
    return _render_facets(plot, build_output, layout, subplotspec=subplotspec)


def _is_coord_flip(coord) -> bool:
    """``coord_flip()`` swaps x↔y at render time. Detect it without
    importing the class at module load (avoids a circular import)."""
    return type(coord).__name__ == "CoordFlip"


def _is_coord_polar(coord) -> bool:
    """``coord_polar()`` switches to matplotlib's polar projection."""
    return type(coord).__name__ == "CoordPolar"


def _polar_x_range(x_scale):
    """Return ``(lo, hi)`` for the trained x-scale so the polar rescale
    can map it to ``[0, 2π]``.
    """
    from .scales.continuous import ScaleContinuous
    from .scales.ordinal import ScaleOrdinal

    if isinstance(x_scale, ScaleOrdinal):
        levels = x_scale.resolved_limits()
        n = len(levels)
        if n == 0:
            return None
        return (0.0, float(n))
    if isinstance(x_scale, ScaleContinuous):
        if x_scale.range_ is None:
            return None
        return (float(x_scale.range_[0]), float(x_scale.range_[1]))
    return None


def _polar_prep_layer_data(df, x_scale):
    """Convert ordinal x to numeric positions and rescale theta to [0, 2π]."""
    from .scales.ordinal import ScaleOrdinal

    if isinstance(x_scale, ScaleOrdinal) and "x" in df.columns:
        levels = x_scale.resolved_limits()
        if levels:
            level_to_pos = {str(lvl): float(i) + 0.5 for i, lvl in enumerate(levels)}
            x_dtype = df["x"].dtype
            if not x_dtype.is_numeric():
                df = df.with_columns(
                    pl.col("x")
                    .cast(pl.Utf8)
                    .replace_strict(
                        level_to_pos,
                        default=None,
                    )
                    .alias("x"),
                )
    return df


def _polar_apply_scales(ax, x_scale, y_scale, x_range):
    """Apply scale ticks/limits on a polar axes."""
    import numpy as _np

    from .scales.continuous import ScaleContinuous
    from .scales.ordinal import ScaleOrdinal

    if y_scale is not None:
        with contextlib.suppress(Exception):
            y_scale.apply_to_axis(ax, "y", view_limits=None)

    rmin, rmax = ax.get_ylim()
    if rmin > 0:
        ax.set_ylim(0.0, rmax)

    if x_scale is None or x_range is None:
        ax.set_xlim(0.0, 2 * math.pi)
        return
    lo, hi = x_range
    span = hi - lo
    if span <= 0:
        return
    factor = (2 * math.pi) / span

    def _rescale(v):
        return (v - lo) * factor

    if isinstance(x_scale, ScaleOrdinal):
        levels = x_scale.resolved_limits()
        if not levels:
            return
        ticks = [_rescale(i + 0.5) for i in range(len(levels))]
        if x_scale.breaks is None:
            tick_pos: list = []
            tick_labels: list = []
        elif isinstance(x_scale.breaks, str) and x_scale.breaks == "default":
            tick_pos = ticks
            tick_labels = list(levels)
        else:
            tick_pos = []
            tick_labels = []
            for i, lvl in enumerate(levels):
                if lvl in x_scale.breaks:
                    tick_pos.append(ticks[i])
                    tick_labels.append(lvl)
        if x_scale.labels != "default" and tick_pos:
            if callable(x_scale.labels):
                tick_labels = [str(s) for s in x_scale.labels(tick_labels)]
            else:
                tick_labels = [str(s) for s in x_scale.labels]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0.0, 2 * math.pi)
        return

    if isinstance(x_scale, ScaleContinuous):
        if x_scale.breaks is None:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlim(0.0, 2 * math.pi)
            return
        if x_scale.range_ is None:
            return
        break_range = x_scale._expanded_break_range()
        breaks = x_scale._compute_breaks(break_range)
        if breaks is None:
            return
        breaks_arr = _np.atleast_1d(_np.asarray(breaks, dtype=float))
        if breaks_arr.size == 0:
            return
        labels = x_scale._compute_labels(breaks_arr.tolist())
        mask = (breaks_arr >= break_range[0]) & (breaks_arr <= break_range[1])
        breaks_arr = breaks_arr[mask]
        labels = [labels[i] for i in range(len(labels)) if mask[i]]
        tick_pos = [_rescale(b) for b in breaks_arr]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels)
        ax.set_xlim(0.0, 2 * math.pi)


def _coord_view_limits(coord, axis: str):
    """Coord's ``xlim`` / ``ylim`` zoom for ``axis`` (visible axis name)."""
    if coord is None:
        return None
    if _is_coord_flip(coord):
        attr = "ylim" if axis == "x" else "xlim"
    else:
        attr = "xlim" if axis == "x" else "ylim"
    return getattr(coord, attr, None)


def _panel_scale(build_output, panel_id, axis: str):
    """Return the scale that governs ``axis`` on panel ``panel_id``."""
    panel = (
        build_output.panel_scales.get(panel_id) if build_output.panel_scales else None
    )
    if panel is not None:
        sc = panel.get(axis)
        if sc is not None:
            return sc
    if build_output.scales is None:
        return None
    return build_output.scales.get(axis)


def _render_single(plot, build_output, ax, subplotspec=None):
    is_polar = _is_coord_polar(plot.coordinates)
    subplot_kw = {"projection": "polar"} if is_polar else None

    if subplotspec is not None:
        fig = subplotspec.get_gridspec().figure
        ax = fig.add_subplot(
            subplotspec,
            projection="polar" if is_polar else None,
        )
        owns_fig = False
    elif ax is None:
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
        owns_fig = True
    else:
        if is_polar and getattr(ax, "name", None) != "polar":
            raise ValueError(
                "coord_polar() requires a polar axes; got a Cartesian ax. "
                "Pass subplot_kw={'projection': 'polar'} when creating the axes.",
            )
        fig = ax.figure
        owns_fig = False

    is_flipped = _is_coord_flip(plot.coordinates)
    ax._hea_coord_flipped = is_flipped

    if not is_polar:
        for axis in ("x", "y"):
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = _panel_scale(build_output, 1, scale_aes)
            if sc is not None:
                sc.setup_axis(ax, axis)

    if is_polar:
        x_scale = _panel_scale(build_output, 1, "x")
        x_range = _polar_x_range(x_scale)
    else:
        x_range = None

    for layer, df in zip(plot.layers, build_output.data):
        if is_flipped:
            from .coords.flip import flip_columns

            df = flip_columns(df)
        if is_polar:
            df = _polar_prep_layer_data(df, x_scale)
            if x_range is not None:
                df = plot.coordinates.rescale_theta(df, x_range)
        layer.geom.draw_panel(df, ax)

    if is_polar:
        _polar_apply_scales(
            ax,
            x_scale,
            _panel_scale(build_output, 1, "y"),
            x_range,
        )
    else:
        for axis in ("x", "y"):
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = _panel_scale(build_output, 1, scale_aes)
            if sc is not None:
                sc.apply_to_axis(
                    ax,
                    axis,
                    view_limits=_coord_view_limits(plot.coordinates, axis),
                )

    xlabel, ylabel = _default_labels(plot, build_output)
    if is_flipped:
        xlabel, ylabel = ylabel, xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if owns_fig:
        _apply_plot_titles(plot, fig, ax_list=[ax])

    _apply_theme(plot.theme, fig, [ax], owns_fig=owns_fig)

    apply = getattr(plot.coordinates, "apply_to_axes", None)
    if apply is not None:
        apply(ax)

    from .guides import apply_axis_guides, apply_legends

    apply_axis_guides([ax], plot)
    apply_legends(fig, [ax], plot, build_output)

    if owns_fig:
        fig.tight_layout()
    return fig


def _render_facets(plot, build_output, layout, subplotspec=None):
    facet = plot.facet
    n_panels = len(layout)
    nrow, ncol = facet.grid_dims(n_panels)

    sharex, sharey = facet.share_axes()
    is_flipped = _is_coord_flip(plot.coordinates)

    if subplotspec is not None:
        fig = subplotspec.get_gridspec().figure
        sub_gs = subplotspec.subgridspec(nrow, ncol)
        axes = sub_gs.subplots(sharex=sharex, sharey=sharey, squeeze=False)
        owns_fig = False
    else:
        fig, axes = plt.subplots(
            nrow,
            ncol,
            sharex=sharex,
            sharey=sharey,
            figsize=(3.0 * ncol, 2.5 * nrow),
            squeeze=False,
        )
        owns_fig = True
    flat_axes = axes.flatten()

    for panel_row in layout.iter_rows(named=True):
        idx = panel_row["PANEL"] - 1
        panel_ax = flat_axes[idx]
        panel_ax._hea_coord_flipped = is_flipped

        for axis in ("x", "y"):
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = _panel_scale(build_output, panel_row["PANEL"], scale_aes)
            if sc is not None:
                sc.setup_axis(panel_ax, axis)

        for layer, df in zip(plot.layers, build_output.data):
            if "PANEL" not in df.columns:
                panel_data = df
            else:
                panel_data = df.filter(pl.col("PANEL") == panel_row["PANEL"])
            if is_flipped:
                from .coords.flip import flip_columns

                panel_data = flip_columns(panel_data)
            if len(panel_data) > 0:
                layer.geom.draw_panel(panel_data, panel_ax)

        for axis in ("x", "y"):
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = _panel_scale(build_output, panel_row["PANEL"], scale_aes)
            if sc is not None:
                sc.apply_to_axis(
                    panel_ax,
                    axis,
                    view_limits=_coord_view_limits(plot.coordinates, axis),
                )

        labels = facet.panel_labels(panel_row, layout)
        if labels.get("top"):
            panel_ax.set_title(labels["top"], y=1.0, pad=0)
        if labels.get("right"):
            _draw_right_strip(plot.theme, panel_ax, labels["right"])

    for unused_ax in flat_axes[n_panels:]:
        unused_ax.set_visible(False)

    xlabel, ylabel = _default_labels(plot, build_output)
    if is_flipped:
        xlabel, ylabel = ylabel, xlabel
    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel is not None:
        fig.supylabel(ylabel)

    if owns_fig:
        _apply_plot_titles(plot, fig, ax_list=list(flat_axes[:n_panels]))
    _apply_theme(
        plot.theme, fig, list(flat_axes[:n_panels]), owns_fig=owns_fig, is_faceted=True
    )

    apply = getattr(plot.coordinates, "apply_to_axes", None)
    if apply is not None:
        for panel_ax in flat_axes[:n_panels]:
            apply(panel_ax)

    from .guides import apply_axis_guides, apply_legends

    apply_axis_guides(list(flat_axes[:n_panels]), plot)
    apply_legends(fig, list(flat_axes[:n_panels]), plot, build_output)

    if owns_fig:
        fig.tight_layout()
    return fig


def _apply_theme(
    theme, fig, axes_list, *, owns_fig: bool, is_faceted: bool = False
) -> None:
    if theme is None or not theme.elements:
        return

    if owns_fig:
        _apply_plot_background(theme, fig)

    for ax in axes_list:
        ax.set_axisbelow(True)
        _apply_panel_background(theme, ax)
        _apply_grid(theme, ax)
        _apply_spines(theme, ax)
        _apply_ticks_and_text(theme, ax)
        _apply_axis_titles(theme, ax)
        _apply_strip_text(theme, ax)
        if is_faceted:
            _apply_strip_background(theme, ax)


def _apply_plot_background(theme, fig) -> None:
    pb = theme.get("plot.background")
    if isinstance(pb, element_blank):
        fig.patch.set_facecolor("none")
    elif isinstance(pb, element_rect) and pb.fill:
        fig.patch.set_facecolor(r_color(pb.fill))


def _apply_panel_background(theme, ax) -> None:
    pnb = theme.get("panel.background")
    if isinstance(pnb, element_blank):
        ax.set_facecolor("none")
    elif isinstance(pnb, element_rect) and pnb.fill:
        ax.set_facecolor(r_color(pnb.fill))


def _apply_grid(theme, ax) -> None:
    """Draw major / minor gridlines from ``panel.grid.*`` theme elements."""
    from ..plot._util import r_lty

    elem = theme.get("panel.grid.major")
    if elem is None:
        elem = theme.get("panel.grid")
    if isinstance(elem, element_blank):
        ax.grid(False, which="major")
    elif isinstance(elem, element_line):
        ax.grid(
            True,
            which="major",
            color=r_color(elem.colour) or "white",
            linewidth=(elem.size or 0.5) * _PT_PER_MM,
            linestyle=r_lty(elem.linetype) if elem.linetype else "-",
            zorder=0,
        )

    minor = theme.get("panel.grid.minor")
    if minor is None:
        minor = theme.get("panel.grid")
    if isinstance(minor, element_blank):
        ax.grid(False, which="minor")
        return
    if isinstance(minor, element_line) and ax.get_xscale() != "linear":
        ax.grid(
            True,
            which="minor",
            color=r_color(minor.colour) or "white",
            linewidth=(minor.size or 0.25) * _PT_PER_MM,
            linestyle=r_lty(minor.linetype) if minor.linetype else "-",
            zorder=0,
        )


def _apply_spines(theme, ax) -> None:
    """Apply ``panel.border`` (all four sides) or ``axis.line`` (bottom/left
    only) to matplotlib spines. ``panel.border`` wins when set — it's a
    superset of ``axis.line`` semantics. With both blank, all four hide
    (ggplot2's ``theme_gray`` default — coloured panel background carries
    the visual weight).
    """
    if getattr(ax, "name", None) == "polar":
        panel_border = theme.get("panel.border")
        polar_spine = ax.spines["polar"]
        if isinstance(panel_border, element_blank):
            polar_spine.set_visible(False)
        elif isinstance(panel_border, element_rect):
            polar_spine.set_visible(True)
            if panel_border.colour:
                polar_spine.set_color(r_color(panel_border.colour))
            if panel_border.size:
                polar_spine.set_linewidth(panel_border.size * _PT_PER_MM)
        return

    axis_line = theme.get("axis.line")
    panel_border = theme.get("panel.border")

    all_sides = ("top", "right", "bottom", "left")

    if isinstance(panel_border, element_rect):
        for side in all_sides:
            sp = ax.spines[side]
            sp.set_visible(True)
            if panel_border.colour:
                sp.set_color(r_color(panel_border.colour))
            if panel_border.size:
                sp.set_linewidth(panel_border.size * _PT_PER_MM)
        return

    if isinstance(axis_line, element_line):
        for side in ("bottom", "left"):
            sp = ax.spines[side]
            sp.set_visible(True)
            if axis_line.colour:
                sp.set_color(r_color(axis_line.colour))
            if axis_line.size:
                sp.set_linewidth(axis_line.size * _PT_PER_MM)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    else:
        for side in all_sides:
            ax.spines[side].set_visible(False)


def _apply_ticks_and_text(theme, ax) -> None:
    """Apply ``axis.ticks`` (line styling) and ``axis.text`` (tick label
    styling) to ``ax``.
    """
    ticks = theme.get("axis.ticks")
    text = theme.get("axis.text")
    text_x = theme.get("axis.text.x")
    text_y = theme.get("axis.text.y")

    tick_kwargs = {"which": "both"}
    if isinstance(ticks, element_blank):
        tick_kwargs["length"] = 0
    elif isinstance(ticks, element_line):
        if ticks.colour:
            tick_kwargs["color"] = r_color(ticks.colour)
        if ticks.size:
            tick_kwargs["width"] = ticks.size * _PT_PER_MM
            tick_kwargs["length"] = ticks.size * _PT_PER_MM * 8
    if len(tick_kwargs) > 1:
        ax.tick_params(**tick_kwargs)

    def _resolve_text(override):
        if isinstance(override, element_blank):
            return override
        if override is None:
            return text
        if isinstance(text, element_text) and isinstance(override, element_text):
            return _merge_text(text, override)
        return override

    x_text = _resolve_text(text_x)
    y_text = _resolve_text(text_y)

    def _apply(side: str, elem) -> None:
        if not isinstance(elem, (element_blank, element_text)):
            return
        kw = {"axis": side, "which": "both"}
        if isinstance(elem, element_blank):
            if side == "x":
                kw["labelbottom"] = False
                kw["labeltop"] = False
            else:
                kw["labelleft"] = False
                kw["labelright"] = False
        else:  # element_text
            if elem.colour:
                kw["labelcolor"] = r_color(elem.colour)
            if elem.size:
                kw["labelsize"] = elem.size
        if len(kw) > 2:  # something beyond "axis" and "which"
            ax.tick_params(**kw)

    _apply("x", x_text)
    _apply("y", y_text)


def _apply_axis_titles(theme, ax) -> None:
    base = theme.get("axis.title")
    x_override = theme.get("axis.title.x")
    y_override = theme.get("axis.title.y")

    def _resolve(side_override):
        if isinstance(side_override, element_blank):
            return side_override
        if side_override is None:
            return base
        if isinstance(base, element_text) and isinstance(side_override, element_text):
            return _merge_text(base, side_override)
        return side_override

    x_elem = _resolve(x_override)
    y_elem = _resolve(y_override)

    _apply_label_element(ax.xaxis.label, x_elem, ax, axis="x")
    _apply_label_element(ax.yaxis.label, y_elem, ax, axis="y")


def _merge_text(base, override):
    """Merge two element_text objects (override wins on non-None)."""
    return element_text(
        family=override.family or base.family,
        face=override.face or base.face,
        colour=override.colour or base.colour,
        size=override.size or base.size,
        hjust=override.hjust if override.hjust is not None else base.hjust,
        vjust=override.vjust if override.vjust is not None else base.vjust,
        angle=override.angle if override.angle is not None else base.angle,
        lineheight=override.lineheight
        if override.lineheight is not None
        else base.lineheight,
    )


def _apply_text_element(text_artist, elem) -> None:
    """Apply :class:`element_text` styling to a matplotlib ``Text`` artist."""
    if not isinstance(elem, element_text):
        return
    if elem.colour:
        text_artist.set_color(r_color(elem.colour))
    if elem.size:
        text_artist.set_size(elem.size)
    if elem.angle is not None:
        text_artist.set_rotation(elem.angle)
    if elem.family:
        text_artist.set_family(elem.family)
    if elem.face:
        if "bold" in elem.face:
            text_artist.set_weight("bold")
        if "italic" in elem.face:
            text_artist.set_style("italic")


def _apply_label_element(text_artist, elem, ax, *, axis):
    if isinstance(elem, element_blank):
        if axis == "x":
            ax.set_xlabel("")
        else:
            ax.set_ylabel("")
        return
    _apply_text_element(text_artist, elem)


def _apply_strip_text(theme, ax) -> None:
    """Style the strip label (set as ``ax.set_title`` for facet panels)."""
    text = theme.get("strip.text")
    title_artist = ax.title
    if isinstance(text, element_blank):
        title_artist.set_text("")
        return
    if isinstance(text, element_text):
        if text.colour:
            title_artist.set_color(r_color(text.colour))
        if text.size:
            title_artist.set_size(text.size)
        if text.face and "bold" in text.face:
            title_artist.set_weight("bold")


def _draw_right_strip(theme, ax, label: str) -> None:
    """Paint a facet_grid right-side strip — vertical bar at the right
    edge of the panel with the rotated row label centred inside.
    """
    if not label:
        return
    fig = ax.figure
    ax_width_in = ax.get_position().width * fig.get_figwidth()
    if ax_width_in <= 0:
        return
    strip_w_in = strip_cell_height_in(label, fontsize=STRIP_TEXT_SIZE_PT)
    strip_w_axes = strip_w_in / ax_width_in

    bg = theme.get("strip.background") if theme is not None else None
    if isinstance(bg, element_rect) and not isinstance(bg, element_blank):
        facecolor = r_color(bg.fill) if bg.fill else "none"
        edgecolor = r_color(bg.colour) if bg.colour else "none"
        linewidth = (bg.size * _PT_PER_MM) if (bg.colour and bg.size) else 0.0
        rect = Rectangle(
            (1.0, 0.0),
            strip_w_axes,
            1.0,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            clip_on=False,
            zorder=-1,  # paints before axes content; matches top strip
        )
        fig.add_artist(rect)

    ax.text(
        1.0 + strip_w_axes / 2.0,
        0.5,
        label,
        transform=ax.transAxes,
        rotation=-90,
        ha="center",
        va="center",
    )


def _apply_strip_background(theme, ax) -> None:
    """Paint the panel-wide rectangle behind a facet panel's top strip."""
    title = ax.title
    label = title.get_text()
    if not label:
        return
    bg = theme.get("strip.background")
    if isinstance(bg, element_blank) or not isinstance(bg, element_rect):
        return

    fig = ax.figure
    ax_height_in = ax.get_position().height * fig.get_figheight()
    if ax_height_in <= 0:
        return
    fontsize = title.get_fontsize() or STRIP_TEXT_SIZE_PT
    strip_h_in = strip_cell_height_in(label, fontsize=fontsize)
    strip_h_axes = strip_h_in / ax_height_in

    facecolor = r_color(bg.fill) if bg.fill else "none"
    edgecolor = r_color(bg.colour) if bg.colour else "none"
    linewidth = (bg.size * _PT_PER_MM) if (bg.colour and bg.size) else 0.0

    rect = Rectangle(
        (0.0, 1.0),
        1.0,
        strip_h_axes,
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False,
        zorder=-1,
    )
    fig.add_artist(rect)

    title.set_y(1.0 + strip_h_axes / 2.0)
    title.set_va("center")


def _default_labels(plot, build_output=None):
    """Resolve x/y labels with explicit ``labs()`` overrides taking priority."""
    from .scales.scale import _NAME_MISSING

    explicit = plot.labels

    def _scale_name_for(axis_key):
        """Return ``""`` for explicit None (suppress), the string for an
        explicit name, or ``None`` to defer to mapping fallback."""
        if build_output is None:
            return None
        scales = getattr(build_output, "scales", None) or {}
        sc = scales.get(axis_key)
        if sc is None:
            return None
        nm = getattr(sc, "name", _NAME_MISSING)
        if nm is _NAME_MISSING:
            return None
        if nm is None:
            return ""
        return str(nm)

    def _from_mapping(mapping, key):
        from .aes import AfterStat

        m = mapping.get(key) if key in mapping else None
        if isinstance(m, str):
            return m
        hea_label = getattr(m, "__hea_label__", None)
        if hea_label is not None:
            return hea_label
        if isinstance(m, AfterStat):
            return str(m.expr) if isinstance(m.expr, str) else None
        if isinstance(m, pl.Expr):
            try:
                return m.meta.output_name()
            except Exception:  # noqa: BLE001
                return None
        return None

    def _from_layers(key):
        effective = (
            getattr(build_output, "layer_mappings", None)
            if build_output is not None
            else None
        )
        for i, layer in enumerate(plot.layers):
            m = None
            if effective is not None and i < len(effective):
                m = effective[i]
            if not m:
                m = getattr(layer, "mapping", None)
            if not m:
                continue
            label = _from_mapping(m, key)
            if label is not None:
                return label
        return None

    is_polar = type(getattr(plot, "coordinates", None)).__name__ == "CoordPolar"

    if "x" in explicit:
        xlabel = "" if explicit["x"] is None else str(explicit["x"])
    elif is_polar:
        xlabel = None
    else:
        scale_x = _scale_name_for("x")
        if scale_x is not None:
            xlabel = scale_x
        else:
            xlabel = _from_mapping(plot.mapping, "x") or _from_layers("x")

    if "y" in explicit:
        ylabel = "" if explicit["y"] is None else str(explicit["y"])
    elif is_polar:
        ylabel = None
    else:
        scale_y = _scale_name_for("y")
        if scale_y is not None:
            ylabel = scale_y
        else:
            ylabel = _from_mapping(plot.mapping, "y") or _from_layers("y")
        if ylabel is None:
            for layer in plot.layers:
                tag = getattr(layer.stat, "default_y_label", None)
                if tag:
                    ylabel = tag
                    break
    return xlabel, ylabel


def _apply_plot_titles(plot, fig, ax_list=None, *, skip_caption: bool = False) -> None:
    """Render ``title`` / ``subtitle`` / ``caption`` from ``plot.labels``."""
    title = plot.labels.get("title")
    subtitle = plot.labels.get("subtitle")
    caption = plot.labels.get("caption")

    if title is not None or subtitle is not None:
        is_faceted = ax_list is not None and len(ax_list) > 1
        target_ax = ax_list[0] if is_faceted else (ax_list or fig.axes)[0]
        title_elem = plot.theme.get("plot.title")
        sub_elem = plot.theme.get("plot.subtitle")

        if title is not None and subtitle is not None:
            title_loc = _title_loc(plot.theme, "plot.title", default_hjust=0.0)
            sub_loc = _title_loc(plot.theme, "plot.subtitle", default_hjust=0.0)
            sub_size = _text_size(sub_elem, default=11.0)
            extra_pad = sub_size * 1.2 + rcParams["axes.titlepad"]
            title_y = 1.15 if is_faceted else None  # facets: clear strip row
            kw = {"loc": title_loc, "pad": extra_pad}
            if title_y is not None:
                kw["y"] = title_y
            title_artist = target_ax.set_title(str(title), **kw)
            _apply_text_element(title_artist, title_elem)

            sub_anchor_y = 1.0
            sub_lift_pts = 2.0  # small breathing room above spine/strip
            sub_trans = offset_copy(
                target_ax.transAxes,
                fig=fig,
                x=0,
                y=sub_lift_pts,
                units="points",
            )
            sub_x, sub_ha = _hjust_to_axes_x_ha(sub_loc)
            sub_artist = target_ax.text(
                sub_x,
                sub_anchor_y,
                str(subtitle),
                transform=sub_trans,
                ha=sub_ha,
                va="bottom",
            )
            _apply_text_element(sub_artist, sub_elem)
        else:
            elem_key = "plot.title" if title is not None else "plot.subtitle"
            text_str = str(title if title is not None else subtitle)
            loc = _title_loc(plot.theme, elem_key, default_hjust=0.0)
            kw = {"loc": loc}
            if is_faceted:
                kw["y"] = 1.15
            title_artist = target_ax.set_title(text_str, **kw)
            _apply_text_element(title_artist, plot.theme.get(elem_key))

    if caption is not None and not skip_caption:
        x, ha = _caption_x_ha(plot.theme)
        cap_artist = fig.text(x, 0.01, str(caption), ha=ha, va="bottom")
        _apply_text_element(cap_artist, plot.theme.get("plot.caption"))


def _text_size(elem, *, default: float) -> float:
    if isinstance(elem, element_text) and elem.size:
        return float(elem.size)
    return default


def _hjust_to_axes_x_ha(loc: str) -> tuple:
    """Map ``set_title``-style loc to axes-coord (x, ha) for ``ax.text``."""
    if loc == "right":
        return (1.0, "right")
    if loc == "center":
        return (0.5, "center")
    return (0.0, "left")


def _title_loc(theme, element_key: str, *, default_hjust: float) -> str:
    """Map a theme element's ``hjust`` to ``ax.set_title``'s ``loc=``."""
    elem = theme.get(element_key)
    hjust = default_hjust
    if isinstance(elem, element_text) and elem.hjust is not None:
        hjust = float(elem.hjust)
    if hjust <= 0.0:
        return "left"
    if hjust >= 1.0:
        return "right"
    return "center"


def _caption_x_ha(theme) -> tuple:
    """``plot.caption`` is figure-level; ggplot2 default ``hjust=1`` →
    right-aligned at the figure edge (with a small inset)."""
    elem = theme.get("plot.caption")
    hjust = 1.0
    if isinstance(elem, element_text) and elem.hjust is not None:
        hjust = float(elem.hjust)
    if hjust <= 0.0:
        return (0.05, "left")
    if hjust >= 1.0:
        return (0.95, "right")
    return (0.5, "center")
