"""Render — walk per-layer drawable data into a matplotlib :class:`Figure`.

Single-panel and faceted modes share the per-axes drawing logic; faceted
mode adds a subplot grid plus per-panel data filtering. ``scales="free*"``
modes mean each panel autoscales independently (matplotlib's
``sharex``/``sharey``).
"""

from __future__ import annotations

import math

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


def _is_coord_polar(coord) -> bool:
    """``coord_polar()`` switches to matplotlib's polar projection."""
    return type(coord).__name__ == "CoordPolar"


def _polar_x_range(x_scale):
    """Return ``(lo, hi)`` for the trained x-scale so the polar rescale
    can map it to ``[0, 2π]``.

    Discrete scales: returns ``(0, n)``. Combined with categories
    placed at half-integer positions ``[0.5, 1.5, …, n-0.5]`` (see
    :func:`_polar_prep_layer_data`), the rescale puts bar centers at
    ``(i+0.5)·2π/n`` — matching ggplot2's bar placement. Discrete
    expansion padding (``add=0.6``) is deliberately ignored: on a
    closed circle it would leave a gap at the 0/2π seam.

    Continuous scales: the trained data range ``(min, max)``. For data
    already in ``[0, 2π]`` (pycircstat2's radians) this yields a
    factor of 1.0 in :meth:`CoordPolar.rescale_theta` — no-op.
    """
    from .scales.continuous import ScaleContinuous
    from .scales.ordinal import ScaleOrdinal

    if isinstance(x_scale, ScaleOrdinal):
        levels = x_scale.resolved_limits()
        n = len(levels)
        if n == 0:
            return None
        # NOTE: deliberately ignore the scale's discrete expansion padding
        # (``pad_lo, pad_hi = x_scale._padding()``). On Cartesian, the
        # +0.6/-0.6 pad keeps bars off the axis edges; on a closed circle
        # it just leaves a wedge of empty space at the 0/2π seam and the
        # bars no longer tile the full circle. Categories live at
        # half-integer positions ``[0.5, 1.5, … n-0.5]`` (see
        # :func:`_polar_prep_layer_data`); mapping range ``(0, n)`` into
        # ``[0, 2π]`` then puts bar centers at ``(i+0.5)·2π/n`` and the
        # first bar's left edge at the top — matching ggplot2.
        return (0.0, float(n))
    if isinstance(x_scale, ScaleContinuous):
        if x_scale.range_ is None:
            return None
        return (float(x_scale.range_[0]), float(x_scale.range_[1]))
    return None


def _polar_prep_layer_data(df, x_scale):
    """Convert ordinal x to numeric positions and rescale theta to [0, 2π].

    Cartesian rendering relies on matplotlib's ``StrCategoryConverter``
    to place ordinal strings at integer positions. On polar that
    converter doesn't run the same way, and we want the value
    interpreted as an angle in radians anyway — so we replicate the
    string→position step in polars-space, then let the coord's
    ``rescale_theta`` spread positions evenly around ``[0, 2π]``.

    No-op when x is already numeric in a continuous scale.
    """
    from .scales.ordinal import ScaleOrdinal

    if isinstance(x_scale, ScaleOrdinal) and "x" in df.columns:
        levels = x_scale.resolved_limits()
        if levels:
            # +0.5 puts each category at the *center* of its 1-wide slot.
            # Combined with the [0, n] training range and the [0, 2π]
            # rescale, this places bar centers at (i+0.5)·2π/n — matching
            # ggplot2, so the first bar's left edge sits at theta=0 (the
            # top) rather than its center sitting at the top.
            level_to_pos = {str(lvl): float(i) + 0.5 for i, lvl in enumerate(levels)}
            x_dtype = df["x"].dtype
            if not x_dtype.is_numeric():
                df = df.with_columns(
                    pl.col("x").cast(pl.Utf8).replace_strict(
                        level_to_pos, default=None,
                    ).alias("x"),
                )
    return df


def _polar_apply_scales(ax, x_scale, y_scale, x_range):
    """Apply scale ticks/limits on a polar axes.

    Y (radial) is mostly Cartesian-like: ``set_ylim`` + ``set_yticks``
    behave the same on polar, so we delegate to the scale's normal
    ``apply_to_axis``. X (angular) needs rescaled tick positions —
    the data was remapped from ``x_range`` to ``[0, 2π]``, so the
    scale's breaks (which live in original-data space) must follow
    the same affine transform before they hit ``set_xticks``.
    """
    import numpy as _np

    from .scales.continuous import ScaleContinuous
    from .scales.ordinal import ScaleOrdinal

    if y_scale is not None:
        try:
            y_scale.apply_to_axis(ax, "y", view_limits=None)
        except Exception:
            # Radial-axis polishes (set_rgrids quirks etc.) shouldn't
            # take down the plot; fall back to matplotlib auto-ticks.
            pass

    if x_scale is None or x_range is None:
        # No x-scale info, but still pin the angular axis below.
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
        # Pin angular range — see ScaleContinuous branch for rationale.
        ax.set_xlim(0.0, 2 * math.pi)
        return

    if isinstance(x_scale, ScaleContinuous):
        # Explicit ``breaks=None`` means "no ticks" — clear matplotlib's
        # default degree-spoke ticks too.
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
        # Pin the angular axis to a full circle. ``set_xticks`` on a
        # polar axes auto-expands xlim to enclose all tick positions —
        # for continuous data trained on ``[0, ~2π)``, that pushes xlim
        # past 2π and matplotlib then renders the polar projection as a
        # near-full circle visually collapsed to a near-vertical sliver.
        # Apply AFTER set_xticks so the auto-expand can't undo it.
        ax.set_xlim(0.0, 2 * math.pi)


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
    is_polar = _is_coord_polar(plot.coordinates)
    subplot_kw = {"projection": "polar"} if is_polar else None

    if subplotspec is not None:
        fig = subplotspec.get_gridspec().figure
        ax = fig.add_subplot(
            subplotspec, projection="polar" if is_polar else None,
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
    # Stash on ax so geoms that need to branch (e.g. GeomBar uses ax.barh
    # when flipped) can read without a signature change.
    ax._hea_coord_flipped = is_flipped

    # Pre-axis hook: discrete scales register their category order on
    # matplotlib's category unit BEFORE geoms draw, so the data lands at
    # the levels' positions (not row-encounter positions).
    # Skip on polar: matplotlib's string-category converter wouldn't
    # interpret strings as theta anyway; the polar pre-pass below
    # converts ordinal x to numeric positions before drawing.
    if not is_polar:
        for axis in ("x", "y"):
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = _panel_scale(build_output, 1, scale_aes)
            if sc is not None:
                sc.setup_axis(ax, axis)

    # Polar pre-pass: resolve ordinal strings to numeric positions and
    # rescale the theta-axis data to [0, 2π] so ordinal x fans evenly
    # around the circle (matches ggplot2). For continuous data already
    # in [0, 2π] (pycircstat2's radians) this is a no-op.
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
        # On polar, x is angular (was rescaled to [0, 2π] above) and y
        # is radial. The standard ``apply_to_axis`` would put ordinal
        # ticks at the wrong (unrescaled) positions; ``_polar_apply_scales``
        # adapts both axes.
        _polar_apply_scales(
            ax, x_scale, _panel_scale(build_output, 1, "y"), x_range,
        )
    else:
        for axis in ("x", "y"):
            # Under coord_flip, the scale registered for the x aesthetic
            # applies to the visible y axis (and vice versa) — scales bind
            # to aesthetics, not axes.
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = _panel_scale(build_output, 1, scale_aes)
            if sc is not None:
                sc.apply_to_axis(
                    ax, axis,
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
    xlabel, ylabel = _default_labels(plot, build_output)
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
    the visual weight).

    Polar axes branch: their ``spines`` keys are
    ``{"polar", "start", "end", "inner"}``, not the Cartesian quartet.
    Looking up ``"top"`` raises ``KeyError``, which would crash every
    polar plot. We map ``panel.border`` → ``ax.spines["polar"]`` (the
    outer ring) and otherwise leave matplotlib's polar defaults.
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
    """Apply ``axis.ticks`` (line styling) and ``axis.text`` (tick label
    styling) to ``ax``.

    Per-axis overrides: ``axis.text.x`` / ``axis.text.y`` merge over
    ``axis.text`` for each axis independently, mirroring the resolution
    pattern in :func:`_apply_axis_titles`. Without this, the only way
    to suppress one axis's tick labels was via ``scale_*(breaks=None)``
    — the discoverable theme form was silently ignored. Matters on
    polar (suppress the radial spoke numbers, keep the rim labels)
    and on Cartesian alike.
    """
    ticks = theme.get("axis.ticks")
    text = theme.get("axis.text")
    text_x = theme.get("axis.text.x")
    text_y = theme.get("axis.text.y")

    # Tick line styling stays global (no per-axis override yet).
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


def _default_labels(plot, build_output=None):
    """Resolve x/y labels with explicit ``labs()`` overrides taking priority.

    Precedence per axis: ``plot.labels[axis]`` (set by ``labs()``/``xlab()``/
    ``ylab()``) → ``scale.name`` (when explicitly set on the axis scale,
    e.g. ``scale_x_date(name=...)``) → ``plot.mapping`` deparse → first
    layer mapping deparse → stat default (y only).

    The scale.name fallback uses the ``_NAME_MISSING`` sentinel to tell
    "user passed name=None to *suppress*" apart from "user didn't pass
    name=" (which yields the auto label). With name=None, the resolved
    label is ``""`` — matplotlib renders no axis title.

    The layer-mapping fallback matches the patchwork-doc idiom
    ``ggplot(df).geom_point(aes("mpg", "disp"))`` — aes on the layer, not
    the plot. ggplot2 picks up the labels from the first matching layer;
    we do the same.
    """
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
        # Prefer the build-time *effective* mapping per layer: kwarg-style
        # aes (``geom_bar(x="clarity")``) lands in ``layer.aes_params`` and
        # only gets promoted into a mapping during ``build`` (via
        # ``_promote_string_aes_params``). The promoted view lives in
        # ``build_output.layer_mappings[i]``; ``layer.mapping`` retains
        # only what the user passed through ``aes(...)``. Reading
        # ``layer.mapping`` alone would drop the kwarg-style label.
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

    # Polar coord suppresses auto-derived axis titles by default. Tick
    # labels (categories around the rim, radial tick numbers) already
    # carry the per-axis context, and matplotlib drops the ylabel at the
    # 9 o'clock spoke where it collides with the 180° tick label. ggplot2
    # and pycircstat2 both follow this convention. Users opt back in with
    # ``labs(x="...", y="...")`` — that lands in ``explicit`` and bypasses
    # this branch.
    is_polar = (
        type(getattr(plot, "coordinates", None)).__name__ == "CoordPolar"
    )

    # ``labs(x=None)`` is the explicit-suppress form (mirrors ggplot2's
    # ``labs(x = NULL)``); resolve to ``""`` so matplotlib renders nothing.
    # Pre-fix, ``str(None)`` returned the literal "None" — visible bug.
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
