"""Render — walk per-layer drawable data into a matplotlib :class:`Figure`.

Single-panel and faceted modes share the per-axes drawing logic; faceted
mode adds a subplot grid plus per-panel data filtering. ``scales="free*"``
modes mean each panel autoscales independently (matplotlib's
``sharex``/``sharey``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from ._measure import STRIP_TEXT_SIZE_PT, strip_cell_height_in
from ._util import r_color
from .theme import element_blank, element_line, element_rect, element_text


# ggplot2 sizes are in mm; matplotlib widths/lengths are in pt. R's TeX
# convention: 72.27 pt/inch, 25.4 mm/inch → ≈ 2.8454 pt/mm.
_PT_PER_MM = 72.27 / 25.4


def render(plot, build_output, ax=None, subplotspec=None) -> "plt.Figure":
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
        return _render_single(plot, build_output, ax=ax,
                              subplotspec=subplotspec)
    if ax is not None:
        # Single ax requested for a faceted plot — collapse to one panel.
        return _render_single(plot, build_output, ax=ax, subplotspec=None)
    return _render_facets(plot, build_output, layout,
                          subplotspec=subplotspec)


def _is_coord_flip(coord) -> bool:
    """``coord_flip()`` swaps x↔y at render time. Detect it without
    importing the class at module load (avoids a circular import)."""
    return type(coord).__name__ == "CoordFlip"


def _coord_view_limits(coord, axis: str):
    """Coord's ``xlim`` / ``ylim`` zoom for ``axis`` (visible axis name).

    Under :func:`coord_flip` the coord's ``xlim`` zooms the visible
    *y* axis (and vice versa) — coord limits live in data-space, which
    flips along with the geometry. Returns ``None`` when the coord
    doesn't constrain that axis, letting the scale fall back to its
    own (data-driven) range.
    """
    if coord is None:
        return None
    if _is_coord_flip(coord):
        attr = "ylim" if axis == "x" else "xlim"
    else:
        attr = "xlim" if axis == "x" else "ylim"
    return getattr(coord, attr, None)


def _panel_scale(build_output, panel_id, axis: str):
    """Return the scale that governs ``axis`` on panel ``panel_id``.

    Prefers ``BuildOutput.panel_scales`` (per-panel clones produced for
    ``scales="free*"``); falls back to the global scale for fixed mode
    or unfaceted plots.
    """
    panel = build_output.panel_scales.get(panel_id) if build_output.panel_scales else None
    if panel is not None:
        sc = panel.get(axis)
        if sc is not None:
            return sc
    if build_output.scales is None:
        return None
    return build_output.scales.get(axis)


def _render_single(plot, build_output, ax, subplotspec=None):
    if subplotspec is not None:
        fig = subplotspec.get_gridspec().figure
        ax = fig.add_subplot(subplotspec)
        owns_fig = False
    elif ax is None:
        fig, ax = plt.subplots()
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    is_flipped = _is_coord_flip(plot.coordinates)
    # Stash on ax so geoms that need to branch (e.g. GeomBar uses ax.barh
    # when flipped) can read without a signature change.
    ax._hea_coord_flipped = is_flipped

    # Pre-axis hook: discrete scales register their category order on
    # matplotlib's category unit BEFORE geoms draw, so the data lands at
    # the levels' positions (not row-encounter positions).
    for axis in ("x", "y"):
        scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
        sc = _panel_scale(build_output, 1, scale_aes)
        if sc is not None:
            sc.setup_axis(ax, axis)

    for layer, df in zip(plot.layers, build_output.data):
        if is_flipped:
            from .coords.flip import flip_columns
            df = flip_columns(df)
        layer.geom.draw_panel(df, ax)

    for axis in ("x", "y"):
        # Under coord_flip, the scale registered for the x aesthetic
        # applies to the visible y axis (and vice versa) — scales bind
        # to aesthetics, not axes.
        scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
        sc = _panel_scale(build_output, 1, scale_aes)
        if sc is not None:
            sc.apply_to_axis(
                ax, axis, view_limits=_coord_view_limits(plot.coordinates, axis),
            )

    xlabel, ylabel = _default_labels(plot)
    if is_flipped:
        xlabel, ylabel = ylabel, xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if owns_fig:
        _apply_plot_titles(plot, fig, ax_list=[ax])

    _apply_theme(plot.theme, fig, [ax], owns_fig=owns_fig)

    # Coord-level overrides (e.g. ``coord_cartesian(xlim=...)``) must beat
    # scale-level limits, so apply after both scales and theme have run.
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
            nrow, ncol,
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

        # Pre-axis hook: see _render_single for rationale.
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

        # Apply positional scales per axis. ``panel_scales`` carries the
        # per-panel scale (for ``free*``); fixed mode falls back to the
        # global. Under coord_flip the scales swap axes (the x
        # aesthetic's scale lands on the visible y axis).
        for axis in ("x", "y"):
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = _panel_scale(build_output, panel_row["PANEL"], scale_aes)
            if sc is not None:
                sc.apply_to_axis(
                    panel_ax, axis,
                    view_limits=_coord_view_limits(plot.coordinates, axis),
                )

        labels = facet.panel_labels(panel_row, layout)
        if labels.get("top"):
            # ``y=1.0`` disables matplotlib's auto-title-positioning
            # (``_autotitlepos = False``) so ``_apply_strip_background``
            # can re-center the title without matplotlib yanking it back.
            # ``pad=0`` removes matplotlib's default 6-pt title pad —
            # otherwise the title's transform is offset upward and our
            # ``set_y`` lands ~8 px too high relative to the strip.
            panel_ax.set_title(labels["top"], y=1.0, pad=0)
        if labels.get("right"):
            _draw_right_strip(plot.theme, panel_ax, labels["right"])

    # Hide unused panels (when the grid has more cells than panels).
    for unused_ax in flat_axes[n_panels:]:
        unused_ax.set_visible(False)

    # Common axis labels — set on the figure rather than per-panel so they
    # land in the canonical "outer edge only" position.
    xlabel, ylabel = _default_labels(plot)
    if is_flipped:
        xlabel, ylabel = ylabel, xlabel
    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel is not None:
        fig.supylabel(ylabel)

    if owns_fig:
        _apply_plot_titles(plot, fig, ax_list=list(flat_axes[:n_panels]))
    _apply_theme(plot.theme, fig, list(flat_axes[:n_panels]),
                 owns_fig=owns_fig, is_faceted=True)

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


# ---------------------------------------------------------------------------
# Theme application — translates :class:`Theme` elements to matplotlib calls.
# ---------------------------------------------------------------------------

def _apply_theme(theme, fig, axes_list, *, owns_fig: bool,
                 is_faceted: bool = False) -> None:
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
        # Only faceted plots have strip titles. Skipping for single-panel
        # plots avoids painting a strip-bg behind a centered ``labs(title=)``.
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
    """Draw major / minor gridlines from ``panel.grid.*`` theme elements.

    We do **not** enable matplotlib's minor ticks here. ggplot2's default
    minor gridlines are technically present but rendered invisibly small,
    so visually R only ever shows major gridlines. matplotlib renders
    minor gridlines crisply once minor ticks are on, which would then
    over-paint the panel. Skip the minor pass unless the theme explicitly
    sets a non-blank ``panel.grid.minor`` AND we're on a transformed
    scale (log/sqrt), where matplotlib's locator generates minor ticks
    on its own.
    """
    from ..plot._util import r_lty

    elem = theme.get("panel.grid.major")
    if elem is None:
        elem = theme.get("panel.grid")
    if isinstance(elem, element_blank):
        ax.grid(False, which="major")
    elif isinstance(elem, element_line):
        ax.grid(
            True, which="major",
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
    # On linear scales, matplotlib won't show minor gridlines without
    # minor ticks — and turning those on visually pollutes the axis.
    # Only render minor gridlines when the locator already supplies minor
    # ticks (log/symlog/function scales auto-supply them).
    if isinstance(minor, element_line) and ax.get_xscale() != "linear":
        ax.grid(
            True, which="minor",
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


def _apply_text_element(text_artist, elem) -> None:
    """Apply :class:`element_text` styling to a matplotlib ``Text`` artist.

    Skips silently for ``None`` or non-``element_text`` (e.g. ``element_blank``
    is handled at call sites because the response — hide vs draw nothing —
    depends on the artist)."""
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

    Mirrors :func:`_apply_strip_background` for the right-side case
    (``facet_grid(rows ~ cols)``). Called from the per-panel render
    paths whenever ``facet.panel_labels()`` returns a ``"right"`` entry.
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
            (1.0, 0.0), strip_w_axes, 1.0,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            clip_on=False,
            zorder=-1,  # paints before axes content; matches top strip
        )
        fig.add_artist(rect)

    ax.text(
        1.0 + strip_w_axes / 2.0, 0.5, label,
        transform=ax.transAxes,
        rotation=-90, ha="center", va="center",
    )


def _apply_strip_background(theme, ax) -> None:
    """Paint the panel-wide rectangle behind a facet panel's top strip.

    ggplot2's strip is a full-width bar above the panel that carries the
    facet label. We render it as an axes-relative ``Rectangle`` patch
    spanning ``x ∈ [0, 1]`` and ``y ∈ [1, 1 + h]`` (axes coords), with the
    title text re-centered vertically inside the bar.
    """
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
        (0.0, 1.0), 1.0, strip_h_axes,
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False,
        # Lower zorder than the default Axes zorder (0) so the rectangle
        # paints in the figure's pre-axes pass — axes content (including
        # ``ax.title``) then renders on top. The strip lives outside the
        # panel area (y > 1 in axes coords), so being "behind" the axes
        # doesn't matter visually.
        zorder=-1,
    )
    # Attach to the figure (not ``ax.patches``) so geom-level tests
    # iterating ``ax.patches`` (counting bars, histogram bins) don't trip
    # on the strip rectangle.
    fig.add_artist(rect)

    # Center the title text vertically within the strip bar. ``set_title``
    # at the call site passed ``y=1.0`` to disable matplotlib's auto-title
    # positioning, so this ``set_y`` value sticks across draws.
    title.set_y(1.0 + strip_h_axes / 2.0)
    title.set_va("center")


def _default_labels(plot):
    """Resolve x/y labels with explicit ``labs()`` overrides taking priority.

    Precedence per axis: ``plot.labels[axis]`` (set by ``labs()``/``xlab()``/
    ``ylab()``) → ``plot.mapping`` deparse → first layer mapping deparse →
    stat default (y only).

    The layer-mapping fallback matches the patchwork-doc idiom
    ``ggplot(df).geom_point(aes("mpg", "disp"))`` — aes on the layer, not
    the plot. ggplot2 picks up the labels from the first matching layer;
    we do the same.
    """
    explicit = plot.labels

    def _from_mapping(mapping, key):
        from .aes import AfterStat

        m = mapping.get(key) if key in mapping else None
        if isinstance(m, str):
            return m
        # ``fct_reorder("class", ...)`` and friends return a tagged
        # callable — pull the source column name back out so the axis
        # label says ``class``, not ``<function reorder>`` or nothing.
        hea_label = getattr(m, "__hea_label__", None)
        if hea_label is not None:
            return hea_label
        # ``after_stat("prop")`` → label ``"prop"``. ggplot2 deparses the
        # post-stat expression for the label (``y=after_stat(count*100)``
        # gives ``"count * 100"``); since users pass the expression as a
        # string, we forward it directly. Callables / polars-expr forms
        # have no useful deparse, so we leave the label unset and let the
        # stat-default fallback kick in.
        if isinstance(m, AfterStat):
            return str(m.expr) if isinstance(m.expr, str) else None
        # Polars expressions: best-effort deparse via the source column
        # name. ``col("carat").log()`` returns ``"carat"`` from
        # ``.meta.output_name()`` — not the full ``log10(carat)`` ggplot2
        # would deparse, but a useful default vs. a blank axis label.
        # Users wanting precise labels should use labs() / xlab() / ylab().
        if isinstance(m, pl.Expr):
            try:
                return m.meta.output_name()
            except Exception:
                return None
        return None

    def _from_layers(key):
        for layer in plot.layers:
            m = getattr(layer, "mapping", None)
            if m is None:
                continue
            label = _from_mapping(m, key)
            if label is not None:
                return label
        return None

    if "x" in explicit:
        xlabel = str(explicit["x"])
    else:
        xlabel = _from_mapping(plot.mapping, "x") or _from_layers("x")

    if "y" in explicit:
        ylabel = str(explicit["y"])
    else:
        ylabel = _from_mapping(plot.mapping, "y") or _from_layers("y")
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


def _apply_plot_titles(plot, fig, ax_list=None, *, skip_caption: bool = False) -> None:
    """Render ``title`` / ``subtitle`` / ``caption`` from ``plot.labels``.

    Single-panel plots: ``ax.set_title(loc='left')`` on the (sole) axes so
    the title aligns with the panel's left edge.

    Faceted plots: ``ax.set_title(loc='left', y=1.15)`` on the top-left
    panel — the strip labels (``ax.set_title``) occupy ``y=1.0``, so we
    push the plot title above them with ``y=1.15``.

    Caption is figure-level (footer). Pass ``skip_caption=True`` when
    composing — per-leaf captions would all stomp on the same
    ``fig.text`` location.
    """
    title = plot.labels.get("title")
    subtitle = plot.labels.get("subtitle")
    caption = plot.labels.get("caption")

    if title is not None or subtitle is not None:
        is_faceted = ax_list is not None and len(ax_list) > 1
        target_ax = ax_list[0] if is_faceted else (ax_list or fig.axes)[0]
        title_elem = plot.theme.get("plot.title")
        sub_elem = plot.theme.get("plot.subtitle")

        if title is not None and subtitle is not None:
            # Render as two SEPARATE text artists so each carries its own
            # ggplot2 size (title ~13.2pt, subtitle ~11pt). The title goes
            # via ``set_title`` with an enlarged ``pad=`` so tight_layout
            # reserves the gap; the subtitle drops into that gap as a
            # separate artist anchored at axes y=1.0.
            title_loc = _title_loc(plot.theme, "plot.title", default_hjust=0.0)
            sub_loc = _title_loc(plot.theme, "plot.subtitle", default_hjust=0.0)
            sub_size = _text_size(sub_elem, default=11.0)
            # Reserve subtitle line height + matplotlib's normal title pad,
            # so tight_layout pushes the axes down enough to avoid clipping.
            extra_pad = sub_size * 1.2 + rcParams["axes.titlepad"]
            title_y = 1.15 if is_faceted else None  # facets: clear strip row
            kw = {"loc": title_loc, "pad": extra_pad}
            if title_y is not None:
                kw["y"] = title_y
            title_artist = target_ax.set_title(str(title), **kw)
            _apply_text_element(title_artist, title_elem)

            # Subtitle sits a few points above the spine top (or strip top
            # for facets). va='bottom' anchors the baseline at axes y=1.0,
            # so the text grows upward into the title's pad.
            sub_anchor_y = 1.0 if not is_faceted else 1.0
            sub_lift_pts = 2.0  # small breathing room above spine/strip
            sub_trans = offset_copy(
                target_ax.transAxes, fig=fig, x=0, y=sub_lift_pts,
                units="points",
            )
            sub_x, sub_ha = _hjust_to_axes_x_ha(sub_loc)
            sub_artist = target_ax.text(
                sub_x, sub_anchor_y, str(subtitle), transform=sub_trans,
                ha=sub_ha, va="bottom",
            )
            _apply_text_element(sub_artist, sub_elem)
        else:
            # Only one of title/subtitle is set — single Text artist via
            # set_title, styled from whichever element is present.
            elem_key = "plot.title" if title is not None else "plot.subtitle"
            text_str = str(title if title is not None else subtitle)
            loc = _title_loc(plot.theme, elem_key, default_hjust=0.0)
            kw = {"loc": loc}
            if is_faceted:
                # Strip labels occupy y=1.0 — push the title above them.
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
