"""Render — walk per-layer drawable data into a matplotlib :class:`Figure`.

Single-panel and faceted modes share the per-axes drawing logic; faceted
mode adds a subplot grid plus per-panel data filtering. ``scales="free*"``
modes mean each panel autoscales independently (matplotlib's
``sharex``/``sharey``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from ._util import r_color
from .theme import element_blank, element_line, element_rect, element_text


# ggplot2 sizes are in mm; matplotlib widths/lengths are in pt. R's TeX
# convention: 72.27 pt/inch, 25.4 mm/inch → ≈ 2.8454 pt/mm.
_PT_PER_MM = 72.27 / 25.4


def render(plot, build_output, ax=None) -> "plt.Figure":
    layout = build_output.layout
    n_panels = 1 if layout is None else len(layout)

    if n_panels <= 1 or ax is not None:
        return _render_single(plot, build_output, ax)
    return _render_facets(plot, build_output, layout)


def _render_single(plot, build_output, ax):
    if ax is None:
        fig, ax = plt.subplots()
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    for layer, df in zip(plot.layers, build_output.data):
        layer.geom.draw_panel(df, ax)

    if build_output.scales is not None:
        for axis in ("x", "y"):
            sc = build_output.scales.get(axis)
            if sc is not None:
                sc.apply_to_axis(ax, axis)

    xlabel, ylabel = _default_labels(plot)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if owns_fig:
        _apply_plot_titles(plot, fig)

    _apply_theme(plot.theme, fig, [ax], owns_fig=owns_fig)

    # Coord-level overrides (e.g. ``coord_cartesian(xlim=...)``) must beat
    # scale-level limits, so apply after both scales and theme have run.
    apply = getattr(plot.coordinates, "apply_to_axes", None)
    if apply is not None:
        apply(ax)

    if owns_fig:
        fig.tight_layout()
    return fig


def _render_facets(plot, build_output, layout):
    facet = plot.facet
    n_panels = len(layout)
    nrow, ncol = facet.grid_dims(n_panels)

    sharex, sharey = facet.share_axes()

    fig, axes = plt.subplots(
        nrow, ncol,
        sharex=sharex,
        sharey=sharey,
        figsize=(3.0 * ncol, 2.5 * nrow),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for panel_row in layout.iter_rows(named=True):
        idx = panel_row["PANEL"] - 1
        panel_ax = flat_axes[idx]

        for layer, df in zip(plot.layers, build_output.data):
            if "PANEL" not in df.columns:
                panel_data = df
            else:
                panel_data = df.filter(pl.col("PANEL") == panel_row["PANEL"])
            if len(panel_data) > 0:
                layer.geom.draw_panel(panel_data, panel_ax)

        # Apply positional scales per axis. With sharex/sharey, matplotlib
        # propagates limits across the shared axes, so calling apply_to_axis
        # on each panel is consistent for "fixed" and gives independent
        # ticks for "free*".
        if build_output.scales is not None:
            for axis in ("x", "y"):
                sc = build_output.scales.get(axis)
                if sc is not None:
                    sc.apply_to_axis(panel_ax, axis)

        labels = facet.panel_labels(panel_row, layout)
        if labels.get("top"):
            panel_ax.set_title(labels["top"])
        if labels.get("right"):
            # Right-side strip: vertical text outside the right edge,
            # rotated to read top-to-bottom (ggplot2's strip.text.y default).
            panel_ax.text(
                1.02, 0.5, labels["right"],
                transform=panel_ax.transAxes,
                rotation=-90, ha="left", va="center",
            )

    # Hide unused panels (when the grid has more cells than panels).
    for unused_ax in flat_axes[n_panels:]:
        unused_ax.set_visible(False)

    # Common axis labels — set on the figure rather than per-panel so they
    # land in the canonical "outer edge only" position.
    xlabel, ylabel = _default_labels(plot)
    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel is not None:
        fig.supylabel(ylabel)

    _apply_plot_titles(plot, fig)
    _apply_theme(plot.theme, fig, list(flat_axes[:n_panels]), owns_fig=True)

    apply = getattr(plot.coordinates, "apply_to_axes", None)
    if apply is not None:
        for panel_ax in flat_axes[:n_panels]:
            apply(panel_ax)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Theme application — translates :class:`Theme` elements to matplotlib calls.
# ---------------------------------------------------------------------------

def _apply_theme(theme, fig, axes_list, *, owns_fig: bool) -> None:
    if theme is None or not theme.elements:
        return

    if owns_fig:
        _apply_plot_background(theme, fig)

    for ax in axes_list:
        # ggplot2 draws gridlines / ticks behind the data layers. Without
        # this, matplotlib gridlines paint on top of geoms regardless of
        # the ``zorder=`` we pass to ``ax.grid``.
        ax.set_axisbelow(True)
        _apply_panel_background(theme, ax)
        _apply_grid(theme, ax)
        _apply_spines(theme, ax)
        _apply_ticks_and_text(theme, ax)
        _apply_axis_titles(theme, ax)
        _apply_strip_text(theme, ax)


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
    from ..plot._util import r_lty

    for which, key in (("major", "panel.grid.major"), ("minor", "panel.grid.minor")):
        elem = theme.get(key)
        if elem is None:
            elem = theme.get("panel.grid")
        if isinstance(elem, element_blank):
            ax.grid(False, which=which)
            continue
        if not isinstance(elem, element_line):
            continue
        if which == "minor":
            # matplotlib only draws minor gridlines if minor ticks are on.
            ax.minorticks_on()
        ax.grid(
            True,
            which=which,
            color=r_color(elem.colour) or "white",
            linewidth=(elem.size or 0.5) * _PT_PER_MM,
            linestyle=r_lty(elem.linetype) if elem.linetype else "-",
            zorder=0,
        )


def _apply_spines(theme, ax) -> None:
    """Apply ``panel.border`` (all four sides) or ``axis.line`` (bottom/left
    only) to matplotlib spines. ``panel.border`` wins when set — it's a
    superset of ``axis.line`` semantics. With both blank, all four hide
    (ggplot2's ``theme_gray`` default — coloured panel background carries
    the visual weight)."""
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

    # panel.border is element_blank or None — fall back to axis.line.
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
        # Both blank → hide everything (theme_gray / theme_minimal style).
        for side in all_sides:
            ax.spines[side].set_visible(False)


def _apply_ticks_and_text(theme, ax) -> None:
    ticks = theme.get("axis.ticks")
    text = theme.get("axis.text")

    tick_kwargs = {"which": "both"}

    if isinstance(ticks, element_blank):
        tick_kwargs["length"] = 0
    elif isinstance(ticks, element_line):
        if ticks.colour:
            tick_kwargs["color"] = r_color(ticks.colour)
        # ggplot2 ``size`` for ticks is line width in mm; the *length* of
        # the tick mark itself doesn't have a direct theme element. Use a
        # length proportional to the line width so size scales sensibly.
        if ticks.size:
            tick_kwargs["width"] = ticks.size * _PT_PER_MM
            tick_kwargs["length"] = ticks.size * _PT_PER_MM * 8

    if isinstance(text, element_blank):
        tick_kwargs["labelleft"] = False
        tick_kwargs["labelbottom"] = False
        tick_kwargs["labeltop"] = False
        tick_kwargs["labelright"] = False
    elif isinstance(text, element_text):
        if text.colour:
            tick_kwargs["labelcolor"] = r_color(text.colour)
        if text.size:
            tick_kwargs["labelsize"] = text.size

    if len(tick_kwargs) > 1:  # something beyond just "which"
        ax.tick_params(**tick_kwargs)


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
        lineheight=override.lineheight if override.lineheight is not None else base.lineheight,
    )


def _apply_label_element(text_artist, elem, ax, *, axis):
    if isinstance(elem, element_blank):
        if axis == "x":
            ax.set_xlabel("")
        else:
            ax.set_ylabel("")
        return
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


def _default_labels(plot):
    """Resolve x/y labels with explicit ``labs()`` overrides taking priority.

    Precedence per axis: ``plot.labels[axis]`` (set by ``labs()``/``xlab()``/
    ``ylab()``) → mapping deparse → stat default (y only).
    """
    explicit = plot.labels
    mapping = plot.mapping

    if "x" in explicit:
        xlabel = str(explicit["x"])
    else:
        m = mapping.get("x") if "x" in mapping else None
        xlabel = m if isinstance(m, str) else None

    if "y" in explicit:
        ylabel = str(explicit["y"])
    else:
        m = mapping.get("y") if "y" in mapping else None
        ylabel = m if isinstance(m, str) else None
        if ylabel is None:
            # No user-mapped y → fall back to the first layer's stat default,
            # so histograms get "count" / density gets "density" without
            # needing labs() (matches ggplot2 deparsing of `after_stat(count)`).
            for layer in plot.layers:
                tag = getattr(layer.stat, "default_y_label", None)
                if tag:
                    ylabel = tag
                    break
    return xlabel, ylabel


def _apply_plot_titles(plot, fig) -> None:
    """Render ``title`` / ``subtitle`` / ``caption`` from ``plot.labels``.

    Uses ``fig.suptitle`` for the title (interacts well with ``tight_layout``);
    subtitle and caption land via ``fig.text`` at conventional positions.
    Theme styling for these elements (``plot.title``, ``plot.subtitle``,
    ``plot.caption``) is not yet wired — defaults from matplotlib apply.
    """
    title = plot.labels.get("title")
    subtitle = plot.labels.get("subtitle")
    caption = plot.labels.get("caption")
    if title is not None:
        fig.suptitle(str(title))
    if subtitle is not None:
        # Sit just below where suptitle lands by default. Slight overlap
        # with axes is possible on tight layouts; ggplot2 has the same risk.
        fig.text(0.5, 0.94, str(subtitle), ha="center", va="bottom",
                 fontsize="medium")
    if caption is not None:
        fig.text(0.99, 0.01, str(caption), ha="right", va="bottom",
                 fontsize="small", style="italic")
