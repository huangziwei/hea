"""``PlotBlock`` — a single ggplot's margin-aware layout container.

Each leaf ``ggplot`` produces a block that knows the inch-size of its
four margins (left/right/top/bottom). The panel area is whatever's left
after the margins. Composition (``PlotGrid``) takes max margins per side
across siblings sharing a row or column so panels align.

This file owns the rendering pipeline for the block:

* ``measure_block`` — pure measurement; returns size info, no axes.
* ``render_block`` — given a measured block and a target gridspec cell,
  allocate axes, render the data, apply scales/theme/coords/guides.

The block uses a 3×3 inner gridspec — left margin, panel, right margin
across; top margin, panel row, bottom margin down. The panel cell
contains a single ``Axes`` (or a facet sub-gridspec). Margins are
absolute inches; ``width_ratios``/``height_ratios`` carry those numbers
so matplotlib allocates space proportionally even when the parent
``subplotspec`` allocation differs from the sum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from . import _measure as M

_TICK_MARK_LEN_IN = 0.05  # rcParams['xtick.major.size'] = 3.5pt ≈ 0.05"
_TICK_TO_LABEL_PAD_IN = 0.05  # rcParams['xtick.major.pad']  = 3.5pt ≈ 0.05"
_LABEL_PAD_IN = 0.06  # rcParams['axes.labelpad']    = 4pt   ≈ 0.06"

ELEMENT_GAP_IN = 0.10

BLOCK_GAP_IN = 0.20

_PANEL_MARGIN_PAD_IN = _TICK_TO_LABEL_PAD_IN
_AXIS_LABELPAD_IN = _LABEL_PAD_IN
_TICK_MARK_PAD_IN = _TICK_MARK_LEN_IN + _TICK_TO_LABEL_PAD_IN + 0.02

_DEFAULT_YTICK_RESERVE_IN = (
    M.text_size_in("00000", fontsize=M.AXIS_TEXT_SIZE_PT)[0] + _TICK_MARK_PAD_IN
)
_DEFAULT_XTICK_RESERVE_IN = (
    M.text_size_in("0", fontsize=M.AXIS_TEXT_SIZE_PT)[1] + _TICK_MARK_PAD_IN
)


@dataclass
class PlotBlock:
    """Measured + (after render) realized layout for one leaf plot.

    The four ``margin_*_in`` fields are the OUTER margins around the panel
    cell — anything outside the data area that the plot needs (tick text,
    axis labels, title, caption, colorbar, legend). Composition takes
    ``max(siblings.margin_X)`` per side so panels align. The panel itself
    is flexible: it consumes whatever's left of the cell allocation.
    """

    plot: object
    build_output: object

    margin_left_in: float = 0.0
    margin_right_in: float = 0.0
    margin_top_in: float = 0.0
    margin_bottom_in: float = 0.0

    legend_w_in: float = 0.0  # nonzero when legend.position in ("left", "right", None)
    legend_h_in: float = 0.0  # nonzero when legend.position in ("top", "bottom")
    colorbar_w_in: float = 0.0  # right/left side colorbar dims
    colorbar_h_in: float = 0.0  # top/bottom side colorbar dims

    panel_grid_rows: int = 1
    panel_grid_cols: int = 1

    panel_axes: list = field(default_factory=list)

    figure: object | None = None

    @property
    def n_panels(self) -> int:
        return self.panel_grid_rows * self.panel_grid_cols

    @property
    def outer_margin_top_in(self) -> float:
        return self.margin_top_in

    @property
    def outer_margin_bottom_in(self) -> float:
        return self.margin_bottom_in

    @property
    def outer_margin_left_in(self) -> float:
        return self.margin_left_in

    @property
    def outer_margin_right_in(self) -> float:
        return self.margin_right_in


def measure_block(plot, build_output) -> PlotBlock:
    """Compute per-side margin sizes (inches) for ``plot``.

    Reads ``plot.labels`` for title/subtitle/xlab/ylab/caption, queries the
    scales/aes_source for legend and colorbar presence, and consults the
    facet for panel grid dims. Doesn't touch matplotlib state — pure.
    """
    from .render import _default_labels  # avoid circular at module load

    labels = plot.labels or {}

    import matplotlib as mpl

    title = labels.get("title")
    subtitle = labels.get("subtitle")
    title_h = M.text_size_in(
        title,
        fontsize=mpl.rcParams["axes.titlesize"],
        weight=mpl.rcParams["axes.titleweight"],
    )[1]
    subtitle_h = M.text_size_in(
        subtitle,
        fontsize="medium",
    )[1]
    margin_top = 0.0
    if title_h > 0:
        margin_top += title_h + M.ROW_GAP_IN
    if subtitle_h > 0:
        margin_top += subtitle_h + M.ROW_GAP_IN

    xlabel, ylabel = _default_labels(plot, build_output)
    ylab_w = M.text_size_in(
        ylabel,
        fontsize=M.AXIS_TITLE_SIZE_PT,
        rotation=90.0,
    )[0]
    ytick_reserve = _predict_axis_tick_reserve_in(build_output, "y")
    margin_left = ylab_w + ytick_reserve + _PANEL_MARGIN_PAD_IN
    if ylab_w > 0:
        margin_left += _AXIS_LABELPAD_IN

    xlab_h = M.text_size_in(xlabel, fontsize=M.AXIS_TITLE_SIZE_PT)[1]
    caption = labels.get("caption")
    caption_h = M.text_size_in(caption, fontsize=M.CAPTION_SIZE_PT)[1]
    margin_bottom = _DEFAULT_XTICK_RESERVE_IN + _PANEL_MARGIN_PAD_IN
    if xlab_h > 0:
        margin_bottom += xlab_h + _AXIS_LABELPAD_IN
    if caption_h > 0:
        margin_bottom += caption_h + M.ROW_GAP_IN

    legend_pos = plot.theme.get("legend.position") if plot.theme else None
    cbar_w = _measure_colorbar_width(plot, build_output)
    legend_w_raw, legend_h_raw = _measure_legend_size(plot, build_output)

    legend_w_field = 0.0
    legend_h_field = 0.0
    colorbar_w_field = 0.0
    colorbar_h_field = 0.0

    margin_right = 0.0
    if legend_pos in (None, "right"):
        legend_w_field = legend_w_raw
        colorbar_w_field = cbar_w
        margin_right = cbar_w + legend_w_raw
        if margin_right > 0:
            margin_right += M.COL_GAP_IN
    elif legend_pos == "left":
        legend_w_field = legend_w_raw
        colorbar_w_field = cbar_w
        margin_left += legend_w_raw + (M.COL_GAP_IN if legend_w_raw > 0 else 0.0)
        margin_left += cbar_w + (M.COL_GAP_IN if cbar_w > 0 else 0.0)
    elif legend_pos == "top":
        legend_h_field = legend_h_raw
        colorbar_h_field = cbar_w  # bar width on right == bar height on top
        margin_top += legend_h_raw + (M.ROW_GAP_IN if legend_h_raw > 0 else 0.0)
        margin_top += cbar_w + (M.ROW_GAP_IN if cbar_w > 0 else 0.0)
    elif legend_pos == "bottom":
        legend_h_field = legend_h_raw
        colorbar_h_field = cbar_w
        margin_bottom += legend_h_raw + (M.ROW_GAP_IN if legend_h_raw > 0 else 0.0)
        margin_bottom += cbar_w + (M.ROW_GAP_IN if cbar_w > 0 else 0.0)

    layout = build_output.layout
    n_panels = 1 if layout is None else len(layout)
    if n_panels > 1:
        nrow, ncol = plot.facet.grid_dims(n_panels)
        margin_top += M.strip_cell_height_in("Sample")
        if getattr(plot.facet, "rows", None):
            margin_right += M.strip_cell_height_in("Sample")
    else:
        nrow, ncol = 1, 1

    return PlotBlock(
        plot=plot,
        build_output=build_output,
        margin_left_in=margin_left,
        margin_right_in=margin_right,
        margin_top_in=margin_top,
        margin_bottom_in=margin_bottom,
        legend_w_in=legend_w_field,
        legend_h_in=legend_h_field,
        colorbar_w_in=colorbar_w_field,
        colorbar_h_in=colorbar_h_field,
        panel_grid_rows=nrow,
        panel_grid_cols=ncol,
    )


def _measure_legend_size(plot, build_output) -> tuple[float, float]:
    """Approximate the legend's (w, h) in inches."""
    pos = plot.theme.get("legend.position") if plot.theme else None
    if pos == "none":
        return (0.0, 0.0)

    from .guides import GuideLegend, build_legend_groups

    groups = build_legend_groups(plot, build_output)
    if not groups:
        return (0.0, 0.0)

    overrides = getattr(plot, "guide_overrides", {}) or {}

    def _user_nrow(group) -> int:
        for aes_name in group.aes_names:
            spec = overrides.get(aes_name)
            if isinstance(spec, GuideLegend) and spec.nrow:
                return max(1, int(spec.nrow))
        return 1

    if pos in ("top", "bottom"):
        cell_sizes = [
            M.legend_cell_size_horizontal_in(
                g.title,
                g.labels,
                nrow=_user_nrow(g),
            )
            for g in groups
        ]
        widths = [w for w, _ in cell_sizes]
        heights = [h for _, h in cell_sizes]
        return (sum(widths) + (len(widths) - 1) * M.COL_GAP_IN, max(heights))

    cell_sizes = [M.legend_cell_size_in(g.title, g.labels) for g in groups]
    widths = [w for w, _ in cell_sizes]
    heights = [h for _, h in cell_sizes]
    return (max(widths), sum(heights))


def _predict_axis_tick_reserve_in(build_output, axis: str) -> float:
    """Inch reserve for tick LABEL text on ``axis`` plus tick-mark pad."""
    if axis == "x":
        return _DEFAULT_XTICK_RESERVE_IN

    labels = _predict_axis_tick_labels(build_output, axis)
    if not labels:
        return _DEFAULT_YTICK_RESERVE_IN
    return (
        M.max_label_width_in(labels, fontsize=M.AXIS_TEXT_SIZE_PT) + _TICK_MARK_PAD_IN
    )


def _predict_axis_tick_labels(build_output, axis: str) -> list[str] | None:
    """Predict the tick-label strings the ``axis`` will draw at render time."""
    from .scales.continuous import ScaleContinuous
    from .scales.ordinal import ScaleOrdinal
    from .scales.transformed import IdentityTrans

    scales = getattr(build_output, "scales", None)
    if scales is None:
        return None
    scale = scales.get(axis)
    if scale is None:
        return None

    if isinstance(scale, ScaleOrdinal):
        levels = scale.resolved_limits()
        if not levels:
            return None
        if scale.breaks is None:
            return []
        if scale.breaks == "default":
            tick_labels = list(levels)
        else:
            tick_labels = [str(b) for b in scale.breaks if str(b) in levels]
        if scale.labels != "default" and tick_labels:
            if callable(scale.labels):
                tick_labels = [str(s) for s in scale.labels(tick_labels)]
            else:
                tick_labels = [str(s) for s in scale.labels]
        return tick_labels

    if isinstance(scale, ScaleContinuous):
        if scale.range_ is None:
            return None
        if scale.breaks == "default" and not isinstance(scale.transform, IdentityTrans):
            return None
        if scale.breaks is None:
            return []
        try:
            break_range = scale._expanded_break_range()
            breaks = scale._compute_breaks(break_range)
            breaks = [b for b in breaks if break_range[0] <= b <= break_range[1]]
            if not breaks:
                return None
            return list(scale._compute_labels(breaks))
        except Exception:  # noqa: BLE001
            return None

    return None


def _measure_colorbar_width(plot, build_output) -> float:
    """Width (inches) of the colorbar cell for the right margin."""
    pos = plot.theme.get("legend.position") if plot.theme else None
    if pos == "none":
        return 0.0

    from .guides import build_colorbar_specs

    specs = build_colorbar_specs(plot, build_output)
    if not specs:
        return 0.0

    return max(
        M.colorbar_cell_width_in(_predict_colorbar_tick_labels(s.vmin, s.vmax))
        for s in specs
    )


def _predict_colorbar_tick_labels(vmin: float, vmax: float) -> list[str]:
    """Plausible tick-label strings for a vertical colorbar over
    ``[vmin, vmax]``.
    """
    import matplotlib.ticker as mticker

    if math.isnan(vmin) or math.isnan(vmax) or vmax <= vmin:  # NaN / degenerate
        return [_format_tick_g(0.0 if math.isnan(vmax) else vmax)]

    candidates: set[float] = set()
    for n in (3, 5, 7, 9):
        for t in mticker.MaxNLocator(nbins=n).tick_values(vmin, vmax):
            candidates.add(float(t))
    span = vmax - vmin
    candidates = {t for t in candidates if vmin - span <= t <= vmax + span}
    if not candidates:
        candidates = {vmin, vmax}
    return [_format_tick_g(t) for t in sorted(candidates)]


def _format_tick_g(t: float) -> str:
    """Approximate matplotlib's ``ScalarFormatter`` short form for tick
    text. Whole numbers render without a trailing ``.0``; otherwise
    ``"%g"`` strips trailing zeros without going scientific until the
    magnitude actually warrants it."""
    if t == 0:
        return "0"
    if t == int(t) and abs(t) < 1e16:
        return f"{int(t)}"
    return f"{t:g}"


def render_block(
    plot,
    build_output,
    block: PlotBlock,
    *,
    fig,
    subplotspec=None,
) -> None:
    """Render the block into ``fig``, either at ``subplotspec`` or filling
    the whole figure.

    The block's margin sizes (in inches) become ``width_ratios`` /
    ``height_ratios`` on a 3×3 inner gridspec. The middle cell hosts the
    panel ``Axes`` (or a facet sub-gridspec). matplotlib normalizes ratios
    to whatever the parent allocated, so the per-cell inch sizes are
    correct as long as the figure size matches the sum of margins +
    panel.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    panel_w = max(
        fig.get_figwidth() - block.margin_left_in - block.margin_right_in,
        0.5,
    )
    panel_h = max(
        fig.get_figheight() - block.margin_top_in - block.margin_bottom_in,
        0.5,
    )

    width_ratios = [block.margin_left_in, panel_w, block.margin_right_in]
    height_ratios = [block.margin_top_in, panel_h, block.margin_bottom_in]
    width_ratios = [max(r, 1e-6) for r in width_ratios]
    height_ratios = [max(r, 1e-6) for r in height_ratios]

    if subplotspec is None:
        gs = GridSpec(
            3,
            3,
            figure=fig,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            left=0.0,
            right=1.0,
            top=1.0,
            bottom=0.0,
            wspace=0.0,
            hspace=0.0,
        )
    else:
        gs = GridSpecFromSubplotSpec(
            3,
            3,
            subplot_spec=subplotspec,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=0.0,
            hspace=0.0,
        )

    panel_cell = gs[1, 1]
    block.figure = fig

    from .render import _is_coord_polar

    is_polar = _is_coord_polar(plot.coordinates)
    polar_kw = {"projection": "polar"} if is_polar else {}

    if block.n_panels == 1:
        ax = fig.add_subplot(panel_cell, **polar_kw)
        block.panel_axes = [ax]
        cb_caxes = _allocate_colorbar_caxes(fig, gs, 1, 2, plot, build_output)
        leg_hosts = _allocate_legend_host_axes(
            fig, gs, 1, 2, plot, build_output, block=block
        )
        _render_single_into(
            plot, build_output, ax, colorbar_caxes=cb_caxes, legend_host_axes=leg_hosts
        )
    else:
        nrow, ncol = block.panel_grid_rows, block.panel_grid_cols
        sharex, sharey = plot.facet.share_axes()
        sub_gs = GridSpecFromSubplotSpec(
            nrow,
            ncol,
            subplot_spec=panel_cell,
            wspace=0.05,
            hspace=0.20,
        )
        axes = []
        for r in range(nrow):
            row_axes = []
            for c in range(ncol):
                share_x_with = _share_anchor(sharex, r, c, axes, row_axes, axis="x")
                share_y_with = _share_anchor(sharey, r, c, axes, row_axes, axis="y")
                share_kw = (
                    {} if is_polar else {"sharex": share_x_with, "sharey": share_y_with}
                )
                ax = fig.add_subplot(
                    sub_gs[r, c],
                    **share_kw,
                    **polar_kw,
                )
                row_axes.append(ax)
            axes.append(row_axes)
        block.panel_axes = [ax for row in axes for ax in row]
        cb_caxes = _allocate_colorbar_caxes(fig, gs, 1, 2, plot, build_output)
        leg_hosts = _allocate_legend_host_axes(
            fig, gs, 1, 2, plot, build_output, block=block
        )
        _render_facets_into(
            plot,
            build_output,
            axes,
            colorbar_caxes=cb_caxes,
            legend_host_axes=leg_hosts,
        )

    from .render import _apply_plot_titles

    if subplotspec is None:
        _apply_plot_titles(plot, fig, ax_list=block.panel_axes)


def _render_single_into(
    plot,
    build_output,
    ax,
    *,
    colorbar_caxes: list | None = None,
    legend_host_axes: list | None = None,
) -> None:
    """Run the single-panel rendering pipeline against ``ax``."""
    from .render import (
        _apply_theme,
        _coord_view_limits,
        _default_labels,
        _is_coord_flip,
        _is_coord_polar,
        _polar_apply_scales,
        _polar_prep_layer_data,
        _polar_x_range,
    )

    is_flipped = _is_coord_flip(plot.coordinates)
    is_polar = _is_coord_polar(plot.coordinates)
    ax._hea_coord_flipped = is_flipped

    from .render import _panel_scale

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
        x_scale = None
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

    _apply_theme(plot.theme, ax.figure, [ax], owns_fig=False)

    apply = getattr(plot.coordinates, "apply_to_axes", None)
    if apply is not None:
        apply(ax)

    from .guides import apply_axis_guides, apply_legends

    apply_axis_guides([ax], plot)
    apply_legends(
        ax.figure,
        [ax],
        plot,
        build_output,
        colorbar_caxes=colorbar_caxes,
        legend_host_axes=legend_host_axes,
    )


def _render_facets_into(
    plot,
    build_output,
    axes_grid,
    *,
    composing: bool = False,
    colorbar_caxes: list | None = None,
    legend_host_axes: list | None = None,
) -> None:
    """Render each facet panel into its allocated axes."""
    from .render import (
        _apply_theme,
        _coord_view_limits,
        _default_labels,
        _is_coord_flip,
    )

    facet = plot.facet
    layout = build_output.layout
    is_flipped = _is_coord_flip(plot.coordinates)
    flat_axes = [ax for row in axes_grid for ax in row]
    n_panels = len(layout)

    from .render import _panel_scale

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
            from .render import _draw_right_strip

            _draw_right_strip(plot.theme, panel_ax, labels["right"])

    for unused_ax in flat_axes[n_panels:]:
        unused_ax.set_visible(False)

    sharex, sharey = facet.share_axes()
    _hide_redundant_facet_ticks(axes_grid, sharex, sharey, n_panels)

    fig = flat_axes[0].figure
    xlabel, ylabel = _default_labels(plot, build_output)
    if is_flipped:
        xlabel, ylabel = ylabel, xlabel
    _set_facet_axis_labels(fig, flat_axes[:n_panels], xlabel, ylabel)

    _apply_theme(
        plot.theme,
        fig,
        list(flat_axes[:n_panels]),
        owns_fig=False,
        is_faceted=True,
    )

    apply = getattr(plot.coordinates, "apply_to_axes", None)
    if apply is not None:
        for panel_ax in flat_axes[:n_panels]:
            apply(panel_ax)

    from .guides import apply_axis_guides, apply_legends

    apply_axis_guides(list(flat_axes[:n_panels]), plot)
    apply_legends(
        fig,
        list(flat_axes[:n_panels]),
        plot,
        build_output,
        colorbar_caxes=colorbar_caxes,
        legend_host_axes=legend_host_axes,
    )


def _set_facet_axis_labels(fig, panel_axes: list, xlabel, ylabel) -> None:
    """Place ``xlabel`` / ``ylabel`` via ``fig.text`` at the union bbox of
    ``panel_axes`` — so the label spans the whole panel area of one facet
    leaf, not just a single panel.
    """
    if not panel_axes:
        return
    if xlabel is None and ylabel is None:
        return

    bboxes = []
    for ax in panel_axes:
        try:
            bbox = ax.get_subplotspec().get_position(fig)
        except Exception:  # noqa: BLE001, S112
            continue
        bboxes.append(bbox)
    if not bboxes:
        return

    x0 = min(b.x0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    fig_w = fig.get_figwidth()
    fig_h = fig.get_figheight()

    if xlabel is not None:
        y_offset_in = _DEFAULT_XTICK_RESERVE_IN + _AXIS_LABELPAD_IN
        fig.text(
            cx,
            y0 - y_offset_in / fig_h,
            xlabel,
            ha="center",
            va="top",
            fontsize="medium",
        )
    if ylabel is not None:
        x_offset_in = _DEFAULT_YTICK_RESERVE_IN + _AXIS_LABELPAD_IN
        fig.text(
            x0 - x_offset_in / fig_w,
            cy,
            ylabel,
            ha="right",
            va="center",
            rotation=90,
            fontsize="medium",
        )


def _hide_redundant_facet_ticks(axes_grid, sharex, sharey, n_panels: int) -> None:
    """When facet panels share scales, only the bottom row's xtick labels
    and the leftmost column's ytick labels are informative — the rest are
    redundant and visually cluttering when panels pack tightly.
    """
    nrow = len(axes_grid)
    ncol = len(axes_grid[0]) if nrow else 0
    if nrow == 0 or ncol == 0:
        return

    visible = [[ax.get_visible() for ax in row] for row in axes_grid]

    if sharex in (True, "col"):
        for c in range(ncol):
            bottom_visible_r = None
            for r in range(nrow - 1, -1, -1):
                if visible[r][c]:
                    bottom_visible_r = r
                    break
            if bottom_visible_r is None:
                continue
            for r in range(bottom_visible_r):
                axes_grid[r][c].tick_params(
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labeltop=False,
                )

    if sharey in (True, "row"):
        for r in range(nrow):
            left_visible_c = None
            for c in range(ncol):
                if visible[r][c]:
                    left_visible_c = c
                    break
            if left_visible_c is None:
                continue
            for c in range(left_visible_c + 1, ncol):
                axes_grid[r][c].tick_params(
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )


def _share_anchor(spec, r: int, c: int, axes_grid, row_axes, *, axis: str):
    """Pick the anchor axes for matplotlib ``sharex=`` / ``sharey=``."""
    if not spec:
        return None
    if r == 0 and c == 0:
        return None
    if spec is True:
        return row_axes[0] if r == 0 else axes_grid[0][0]
    if spec == "col":
        if r == 0:
            return None
        return axes_grid[0][c]
    if spec == "row":
        if c == 0:
            return None
        return row_axes[0]
    return None


DEFAULT_PANEL_W_IN = 3.5
DEFAULT_PANEL_H_IN = 3.0


def _has_right_guide(blk) -> bool:
    """Whether the block (leaf or nested) hosts a colorbar/legend on its
    right side. Used to skip lift_right — collapsing a column containing
    a guide would zero out its cax / host axes and the guide would
    render at zero width."""
    if isinstance(blk, PlotBlock):
        plot = blk.plot
        bo = blk.build_output
        pos = plot.theme.get("legend.position") if plot.theme else None
        if pos not in (None, "right"):
            return False
        from .guides import build_colorbar_specs, build_legend_groups

        if build_colorbar_specs(plot, bo):
            return True
        return bool(build_legend_groups(plot, bo))
    if isinstance(blk, SuperBlock):
        for r in range(blk.nrow):
            cell = blk.cells[r][blk.ncol - 1]
            if cell is None:
                continue
            _, child_blk = cell
            if _has_right_guide(child_blk):
                return True
        return False
    return False


def _has_left_guide(blk) -> bool:
    """Whether the block hosts a colorbar/legend on its LEFT side."""
    if isinstance(blk, PlotBlock):
        plot = blk.plot
        bo = blk.build_output
        pos = plot.theme.get("legend.position") if plot.theme else None
        if pos != "left":
            return False
        from .guides import build_colorbar_specs, build_legend_groups

        if build_colorbar_specs(plot, bo):
            return True
        return bool(build_legend_groups(plot, bo))
    if isinstance(blk, SuperBlock):
        for r in range(blk.nrow):
            cell = blk.cells[r][0]
            if cell is None:
                continue
            _, child_blk = cell
            if _has_left_guide(child_blk):
                return True
        return False
    return False


def _has_top_guide(blk) -> bool:
    """Whether the block hosts a colorbar/legend on its TOP side
    (``theme(legend.position="top")``)."""
    if isinstance(blk, PlotBlock):
        plot = blk.plot
        bo = blk.build_output
        pos = plot.theme.get("legend.position") if plot.theme else None
        if pos != "top":
            return False
        from .guides import build_colorbar_specs, build_legend_groups

        if build_colorbar_specs(plot, bo):
            return True
        return bool(build_legend_groups(plot, bo))
    if isinstance(blk, SuperBlock):
        for c in range(blk.ncol):
            cell = blk.cells[0][c]
            if cell is None:
                continue
            _, child_blk = cell
            if _has_top_guide(child_blk):
                return True
        return False
    return False


def _has_bottom_guide(blk) -> bool:
    """Whether the block hosts a colorbar/legend on its BOTTOM side
    (``theme(legend.position="bottom")``)."""
    if isinstance(blk, PlotBlock):
        plot = blk.plot
        bo = blk.build_output
        pos = plot.theme.get("legend.position") if plot.theme else None
        if pos != "bottom":
            return False
        from .guides import build_colorbar_specs, build_legend_groups

        if build_colorbar_specs(plot, bo):
            return True
        return bool(build_legend_groups(plot, bo))
    if isinstance(blk, SuperBlock):
        for c in range(blk.ncol):
            cell = blk.cells[blk.nrow - 1][c]
            if cell is None:
                continue
            _, child_blk = cell
            if _has_bottom_guide(child_blk):
                return True
        return False
    return False


@dataclass
class GuideAreaBlock:
    """Placeholder block for a :func:`guide_area` cell.

    Sized to the legends collected by ``plot_layout(guides="collect")``.
    With zero outer margins the legend draws flush inside the cell,
    consuming the cell's full panel area. ``merged_groups`` is the
    deduplicated legend list to render; ``legend_theme`` carries the
    theme used for legend styling.
    """

    legend_w_in: float = 0.0
    legend_h_in: float = 0.0
    merged_groups: list = field(default_factory=list)
    legend_theme: object | None = None
    panel_axes: list = field(default_factory=list)
    figure: object | None = None

    @property
    def n_panels(self) -> int:
        return 0

    @property
    def outer_margin_top_in(self) -> float:
        return 0.0

    @property
    def outer_margin_bottom_in(self) -> float:
        return 0.0

    @property
    def outer_margin_left_in(self) -> float:
        return 0.0

    @property
    def outer_margin_right_in(self) -> float:
        return 0.0


@dataclass
class SuperBlock:
    """Recursively composed block representing a (possibly nested) ``PlotGrid``.

    Exposes the same ``outer_margin_*`` interface as :class:`PlotBlock` so a
    parent grid can compose nested grids and leaf plots uniformly. The
    internal layout is stored as a 2D table of child blocks (each itself a
    PlotBlock or SuperBlock), with super-margins computed per side.
    """

    grid: object  # PlotGrid
    nrow: int
    ncol: int
    cells: list  # list[list[tuple | None]]
    row_super_top_in: list
    row_super_bottom_in: list
    col_super_left_in: list
    col_super_right_in: list
    panel_h_in: list
    panel_w_in: list
    annot_title_h_in: float = 0.0
    annot_caption_h_in: float = 0.0

    @property
    def outer_margin_top_in(self) -> float:
        if self.nrow == 0:
            return 0.0
        return self.row_super_top_in[0] + self.annot_title_h_in

    @property
    def outer_margin_bottom_in(self) -> float:
        if self.nrow == 0:
            return 0.0
        return self.row_super_bottom_in[-1] + self.annot_caption_h_in

    @property
    def outer_margin_left_in(self) -> float:
        if self.ncol == 0:
            return 0.0
        return self.col_super_left_in[0]

    @property
    def outer_margin_right_in(self) -> float:
        if self.ncol == 0:
            return 0.0
        return self.col_super_right_in[-1]

    @property
    def total_inner_w_in(self) -> float:
        """Width of the ``panel area`` of this super-block (between outer_left
        and outer_right). Sums per-col panel + per-col inner left/right
        margins (the latter excluded for outer cols)."""
        w = 0.0
        for c in range(self.ncol):
            w += self.panel_w_in[c]
            if c > 0:
                w += self.col_super_left_in[c]
            if c < self.ncol - 1:
                w += self.col_super_right_in[c]
        return w

    @property
    def total_inner_h_in(self) -> float:
        h = 0.0
        for r in range(self.nrow):
            h += self.panel_h_in[r]
            if r > 0:
                h += self.row_super_top_in[r]
            if r < self.nrow - 1:
                h += self.row_super_bottom_in[r]
        return h

    @property
    def total_w_in(self) -> float:
        return (
            self.outer_margin_left_in
            + self.total_inner_w_in
            + self.outer_margin_right_in
        )

    @property
    def total_h_in(self) -> float:
        return (
            self.outer_margin_top_in
            + self.total_inner_h_in
            + self.outer_margin_bottom_in
        )


def compute_block(thing, *, collect_state=None):
    """Return a block for ``thing`` (a ``ggplot`` leaf, a ``PlotGrid``, or a
    :class:`GuideArea` placeholder).

    Recurses into nested grids. ``collect_state`` carries the merged-legend
    payload from :func:`_prepare_collect` when ``guides="collect"`` is in
    effect at the outermost grid; nested calls forward it so a
    ``guide_area()`` placeholder inside a sub-grid still gets sized.
    """
    from .build import build
    from .core import ggplot
    from .patchwork import GuideArea, PlotGrid

    if isinstance(thing, GuideArea):
        if collect_state is None:
            return GuideAreaBlock()
        merged_groups, legend_theme = collect_state
        return _measure_guide_area_block(merged_groups, legend_theme)
    if isinstance(thing, ggplot):
        bo = build(thing)
        blk = measure_block(thing, bo)
        return blk
    if isinstance(thing, PlotGrid):
        return compose_super_block(thing, collect_state=collect_state)
    raise TypeError(f"compute_block: unsupported child type {type(thing).__name__}")


def compose_super_block(grid, *, collect_state=None) -> SuperBlock:
    """Recursively measure a :class:`PlotGrid` → :class:`SuperBlock`.

    Each child is a leaf ``PlotBlock``, a nested ``SuperBlock``, or a
    :class:`GuideAreaBlock` placeholder. Super margins per row/col are
    taken as the max of children's ``outer_margin_*`` along that axis
    (except the outermost rows/cols which contribute to *this*
    SuperBlock's outer margins instead).

    When ``grid.guides == "collect"`` and we're at the outermost call
    (``collect_state`` is ``None``), the collection runs once: leaves are
    built, their legends de-duplicated, and a fresh tree with per-leaf
    legends suppressed is composed instead. ``collect_state`` then flows
    down so any ``guide_area()`` inside a nested sub-grid receives the
    merged groups too.
    """
    if grid.guides == "collect" and collect_state is None:
        grid, collect_state = _prepare_collect(grid)

    nrow, ncol = grid._dims()

    if grid.widths is not None and len(grid.widths) != ncol:
        raise ValueError(
            f"PlotGrid: widths has length {len(grid.widths)} "
            f"but the grid has {ncol} columns"
        )
    if grid.heights is not None and len(grid.heights) != nrow:
        raise ValueError(
            f"PlotGrid: heights has length {len(grid.heights)} "
            f"but the grid has {nrow} rows"
        )

    cells: list = [[None] * ncol for _ in range(nrow)]
    for i, child in enumerate(grid.children):
        r, c = grid._cell_for(i)
        cells[r][c] = (child, compute_block(child, collect_state=collect_state))

    row_super_top = [0.0] * nrow
    row_super_bottom = [0.0] * nrow
    col_super_left = [0.0] * ncol
    col_super_right = [0.0] * ncol
    for r in range(nrow):
        for c in range(ncol):
            cell = cells[r][c]
            if cell is None:
                continue
            _, blk = cell
            row_super_top[r] = max(row_super_top[r], blk.outer_margin_top_in)
            row_super_bottom[r] = max(row_super_bottom[r], blk.outer_margin_bottom_in)
            col_super_left[c] = max(col_super_left[c], blk.outer_margin_left_in)
            col_super_right[c] = max(col_super_right[c], blk.outer_margin_right_in)

    for c in range(ncol):
        if c < ncol - 1:
            col_super_right[c] += BLOCK_GAP_IN
    for r in range(nrow):
        if r < nrow - 1:
            row_super_bottom[r] += BLOCK_GAP_IN

    panel_h = [DEFAULT_PANEL_H_IN] * nrow
    panel_w = [DEFAULT_PANEL_W_IN] * ncol
    guide_area_natural_h = [0.0] * nrow
    for r in range(nrow):
        for c in range(ncol):
            cell = cells[r][c]
            if cell is None:
                continue
            _, blk = cell
            if isinstance(blk, SuperBlock):
                inner_h = blk.total_inner_h_in
                inner_w = blk.total_inner_w_in
                if _has_top_guide(blk):
                    inner_h += blk.outer_margin_top_in
                if _has_bottom_guide(blk):
                    inner_h += blk.outer_margin_bottom_in
                if _has_left_guide(blk):
                    inner_w += blk.outer_margin_left_in
                if _has_right_guide(blk):
                    inner_w += blk.outer_margin_right_in
                panel_h[r] = max(panel_h[r], inner_h)
                panel_w[c] = max(panel_w[c], inner_w)
            elif isinstance(blk, GuideAreaBlock):
                guide_area_natural_h[r] = max(
                    guide_area_natural_h[r],
                    blk.legend_h_in,
                )

    if grid.widths is not None:
        total = sum(grid.widths)
        avg = DEFAULT_PANEL_W_IN * ncol / total if total > 0 else DEFAULT_PANEL_W_IN
        panel_w = [w * avg for w in grid.widths]
    if grid.heights is not None:
        total = sum(grid.heights)
        avg = DEFAULT_PANEL_H_IN * nrow / total if total > 0 else DEFAULT_PANEL_H_IN
        panel_h = [h * avg for h in grid.heights]
    else:
        for r in range(nrow):
            row_has_only_guide_area = guide_area_natural_h[r] > 0 and all(
                cells[r][c] is None or isinstance(cells[r][c][1], GuideAreaBlock)
                for c in range(ncol)
            )
            if row_has_only_guide_area:
                panel_h[r] = guide_area_natural_h[r]

    annot_title_h, annot_caption_h = _annotation_extents(grid)

    return SuperBlock(
        grid=grid,
        nrow=nrow,
        ncol=ncol,
        cells=cells,
        row_super_top_in=row_super_top,
        row_super_bottom_in=row_super_bottom,
        col_super_left_in=col_super_left,
        col_super_right_in=col_super_right,
        panel_h_in=panel_h,
        panel_w_in=panel_w,
        annot_title_h_in=annot_title_h,
        annot_caption_h_in=annot_caption_h,
    )


def _annotation_extents(grid) -> tuple[float, float]:
    """Heights (inches) reserved for plot_annotation title and caption rows."""
    if grid.annotation is None:
        return (0.0, 0.0)
    a = grid.annotation
    title_lines = [s for s in (a.title, a.subtitle) if s]
    title_h = M.text_block_size_in(
        title_lines,
        fontsize=M.TITLE_SIZE_PT,
        weight="bold",
    )[1]
    if title_h > 0:
        title_h += 0.1
    caption_h = M.text_size_in(a.caption, fontsize=M.CAPTION_SIZE_PT)[1]
    if caption_h > 0:
        caption_h += 0.05
    return (title_h, caption_h)


def _prepare_collect(grid):
    """Implement ``plot_layout(guides="collect")``."""
    from .core import ggplot
    from .patchwork import GuideArea, PlotGrid
    from .theme import theme as theme_fn

    leaves = grid.leaves()
    if not leaves:
        return grid, ([], None)

    merged_groups = _collect_legend_groups(leaves)
    legend_theme = leaves[0].theme

    suppress = theme_fn(legend_position="none")

    def _broadcast(g):
        new_children = []
        for child in g.children:
            if isinstance(child, GuideArea):
                new_children.append(child)
            elif isinstance(child, PlotGrid):
                new_children.append(_broadcast(child))
            elif isinstance(child, ggplot):
                new_children.append(child + suppress)
            else:
                raise TypeError(
                    f"unexpected child type {type(child).__name__} "
                    "in PlotGrid under guides='collect'"
                )
        return PlotGrid(
            children=new_children,
            direction=g.direction,
            nrow=g.nrow,
            ncol=g.ncol,
            byrow=g.byrow,
            widths=g.widths,
            heights=g.heights,
            annotation=g.annotation,
            guides=g.guides,
        )

    new_grid = _broadcast(grid)

    if merged_groups and new_grid.find_guide_area() is None:
        pos = legend_theme.get("legend.position") if legend_theme else "right"
        new_grid = _wrap_with_guide_area(new_grid, pos or "right")

    return new_grid, (merged_groups, legend_theme)


def _wrap_with_guide_area(grid, pos: str):
    """Wrap ``grid`` in an outer 1×2 / 2×1 grid with a :func:`guide_area`
    on the side indicated by ``pos``. Annotation and ``guides`` flag stay
    on the new outer grid so :func:`_render_guide_area_cell` still has
    access to the merged legend payload."""
    from .patchwork import _DIRECTION_H, _DIRECTION_V, PlotGrid, guide_area

    ga = guide_area()
    inner_annotation = grid.annotation
    inner = PlotGrid(
        children=list(grid.children),
        direction=grid.direction,
        nrow=grid.nrow,
        ncol=grid.ncol,
        byrow=grid.byrow,
        widths=grid.widths,
        heights=grid.heights,
        annotation=None,
        guides=None,
    )
    if pos == "top":
        children = [ga, inner]
        direction = _DIRECTION_V
    elif pos == "bottom":
        children = [inner, ga]
        direction = _DIRECTION_V
    elif pos == "left":
        children = [ga, inner]
        direction = _DIRECTION_H
    else:  # "right" (default)
        children = [inner, ga]
        direction = _DIRECTION_H
    return PlotGrid(
        children=children,
        direction=direction,
        annotation=inner_annotation,
        guides=grid.guides,
    )


def _collect_legend_groups(leaves) -> list:
    """Return one :class:`LegendGroup` per distinct
    ``(title, levels, labels, key_glyph)`` across all leaves.
    """
    from .build import build
    from .guides import LegendGroup, build_legend_groups

    seen: dict[tuple, LegendGroup] = {}
    for leaf in leaves:
        bo = build(leaf)
        for g in build_legend_groups(leaf, bo):
            key = (
                g.title,
                tuple(str(v) for v in g.levels),
                tuple(g.labels),
                g.key_glyph,
            )
            if key in seen:
                merged = seen[key]
                for aes_name, vals in g.aes_values.items():
                    merged.aes_values.setdefault(aes_name, list(vals))
                for k, v in g.layer_aes_params.items():
                    merged.layer_aes_params.setdefault(k, v)
                for k, v in g.layer_default_aes.items():
                    merged.layer_default_aes.setdefault(k, v)
            else:
                seen[key] = LegendGroup(
                    title=g.title,
                    levels=list(g.levels),
                    labels=list(g.labels),
                    aes_values={k: list(v) for k, v in g.aes_values.items()},
                    key_glyph=g.key_glyph,
                    layer_aes_params=dict(g.layer_aes_params),
                    layer_default_aes=dict(g.layer_default_aes),
                )
    return list(seen.values())


def _measure_guide_area_block(merged_groups, legend_theme) -> GuideAreaBlock:
    """Compute the natural size of the merged legend."""
    if not merged_groups:
        return GuideAreaBlock()

    pos = legend_theme.get("legend.position") if legend_theme else "right"
    direction = (legend_theme.get("legend.direction") if legend_theme else None) or (
        "vertical" if pos in ("right", "left") else "horizontal"
    )

    widths, heights = [], []
    for g in merged_groups:
        w, h = M.legend_cell_size_in(g.title, g.labels)
        widths.append(w)
        heights.append(h)

    if direction == "horizontal":
        legend_w = sum(widths) + (len(widths) - 1) * M.COL_GAP_IN
        legend_h = max(heights)
    else:
        legend_w = max(widths)
        legend_h = sum(heights) + (len(heights) - 1) * M.ROW_GAP_IN
    return GuideAreaBlock(
        legend_w_in=legend_w,
        legend_h_in=legend_h,
        merged_groups=merged_groups,
        legend_theme=legend_theme,
    )


def _render_guide_area_cell(blk: GuideAreaBlock, fig, panel_cell) -> None:
    """Draw the merged legends collected by ``plot_layout(guides="collect")``
    into the :func:`guide_area` slot. Lays out one ``host`` axes per
    legend group inside the cell and renders into each via the same
    matplotlib ``Axes.legend`` path the per-plot host uses."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    from .guides import (
        _legend_key_handler,
        _legend_title_alignment,
        _make_handle,
    )

    if not blk.merged_groups:
        return

    theme = blk.legend_theme
    pos = (theme.get("legend.position") if theme else None) or "right"
    direction = (theme.get("legend.direction") if theme else None) or (
        "vertical" if pos in ("right", "left") else "horizontal"
    )

    n = len(blk.merged_groups)
    if direction == "horizontal":
        sub = GridSpecFromSubplotSpec(
            1, n, subplot_spec=panel_cell, wspace=0.1, hspace=0.0
        )
        sub_specs = [sub[0, i] for i in range(n)]
    else:
        sub = GridSpecFromSubplotSpec(
            n, 1, subplot_spec=panel_cell, wspace=0.0, hspace=0.1
        )
        sub_specs = [sub[i, 0] for i in range(n)]

    handler_map = _legend_key_handler(theme) if theme else None
    alignment = _legend_title_alignment(theme) if theme else "left"

    for group, sp in zip(blk.merged_groups, sub_specs):
        host = fig.add_subplot(sp)
        host.set_axis_off()
        host.set_label("<merged-legend>")
        handles = [_make_handle(group, j) for j in range(len(group.levels))]
        ncols = len(handles) if direction == "horizontal" else 1
        labelspacing = 0.4 if group.key_glyph == "polygon" else 0.0
        sizing = {
            "handlelength": 1.2,
            "handleheight": 1.5,
            "labelspacing": labelspacing,
        }
        host.legend(
            handles,
            group.labels,
            title=group.title,
            ncols=ncols,
            loc="center",
            frameon=False,
            alignment=alignment,
            handler_map=handler_map,
            **sizing,
        )
        blk.panel_axes.append(host)
    blk.figure = fig


def _redistribute_to_leftover(ratios, panel_idx, panel_weights, total_in):
    """Make absolute (decoration) entries keep their inch values when the
    figure dimension is smaller than ``sum(ratios)``.
    """
    if not panel_idx or not panel_weights:
        return
    weight_sum = sum(panel_weights)
    if weight_sum <= 0:
        return
    abs_sum = sum(ratios) - sum(panel_weights)
    leftover = total_in - abs_sum
    if leftover <= 0:
        return
    for idx, weight in zip(panel_idx, panel_weights):
        ratios[idx] = leftover * weight / weight_sum


def render_super_block(
    sb: SuperBlock,
    fig,
    parent_subspec=None,
    tag_iter=None,
    outer_top_y: float | None = None,
    lift_top: bool = False,
    lift_bottom: bool = False,
    lift_left: bool = False,
    lift_right: bool = False,
) -> None:
    """Render a :class:`SuperBlock` into ``fig`` at ``parent_subspec``
    (or the whole figure if ``None``).

    Each cell of the grid hosts either a leaf ``PlotBlock`` (rendered via
    the standard panel pipeline) or a nested ``SuperBlock`` (rendered
    recursively into a sub-gridspec).

    ``outer_top_y``: figure-relative y at which titles of *topmost-row*
    children should anchor. When set, leaves in row 0 lift their title
    from the inner top-margin cell up to the outer's top-margin row;
    nested ``SuperBlock`` children in row 0 forward this y to their
    own topmost-row children. Mirrors R/patchwork's ``simplify_gt``
    behaviour where every title row, regardless of nesting depth, lands
    in the super-gtable's row 3.

    ``lift_top`` / ``lift_bottom``: when True, this SuperBlock's first
    super-top row / last super-bottom row is collapsed to ~0 height
    because the parent has reserved that space in its OWN top/bottom
    margin. This makes the inner panel area extend flush to the cell
    edges so panels align with sibling leaves' panels at the parent
    level (e.g. ``p1 | (p2 / p3)`` — p2 and p3 panels share top/bottom
    bounds with p1 because the nested's inner margins are zeroed).
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    grid = sb.grid

    height_ratios: list[float] = []
    panel_h_idx: list[int] = []  # indices into height_ratios
    panel_h_weights: list[float] = []  # parallel; nominal inch (= weight)
    if sb.annot_title_h_in > 0:
        height_ratios.append(sb.annot_title_h_in)
    for r in range(sb.nrow):
        super_top = sb.row_super_top_in[r]
        super_bottom = sb.row_super_bottom_in[r]
        if r == 0 and lift_top:
            super_top = 0.0
        if r == sb.nrow - 1 and lift_bottom:
            super_bottom = 0.0
        height_ratios.append(super_top)
        panel_h_idx.append(len(height_ratios))
        panel_h_weights.append(sb.panel_h_in[r])
        height_ratios.append(sb.panel_h_in[r])
        height_ratios.append(super_bottom)
    if sb.annot_caption_h_in > 0:
        height_ratios.append(sb.annot_caption_h_in)

    width_ratios: list[float] = []
    panel_w_idx: list[int] = []
    panel_w_weights: list[float] = []
    for c in range(sb.ncol):
        super_left = sb.col_super_left_in[c]
        super_right = sb.col_super_right_in[c]
        if c == 0 and lift_left:
            super_left = 0.0
        if c == sb.ncol - 1 and lift_right:
            super_right = 0.0
        width_ratios.append(super_left)
        panel_w_idx.append(len(width_ratios))
        panel_w_weights.append(sb.panel_w_in[c])
        width_ratios.append(sb.panel_w_in[c])
        width_ratios.append(super_right)

    if parent_subspec is None:
        avail_h_in = fig.get_figheight()
        avail_w_in = fig.get_figwidth()
    else:
        bbox = parent_subspec.get_position(fig)
        avail_h_in = bbox.height * fig.get_figheight()
        avail_w_in = bbox.width * fig.get_figwidth()
    _redistribute_to_leftover(
        height_ratios,
        panel_h_idx,
        panel_h_weights,
        avail_h_in,
    )
    _redistribute_to_leftover(
        width_ratios,
        panel_w_idx,
        panel_w_weights,
        avail_w_in,
    )

    gs_h = [max(r, 1e-6) for r in height_ratios]
    gs_w = [max(r, 1e-6) for r in width_ratios]
    title_row_offset = 1 if sb.annot_title_h_in > 0 else 0

    if parent_subspec is None:
        gs = GridSpec(
            len(gs_h),
            len(gs_w),
            figure=fig,
            height_ratios=gs_h,
            width_ratios=gs_w,
            left=0.0,
            right=1.0,
            top=1.0,
            bottom=0.0,
            wspace=0.0,
            hspace=0.0,
        )
    else:
        gs = GridSpecFromSubplotSpec(
            len(gs_h),
            len(gs_w),
            subplot_spec=parent_subspec,
            height_ratios=gs_h,
            width_ratios=gs_w,
            wspace=0.0,
            hspace=0.0,
        )

    if grid.annotation is not None:
        _apply_block_annotation(
            grid, fig, gs, title_row_offset, sb.ncol, sb.annot_caption_h_in > 0
        )

    for r in range(sb.nrow):
        for c in range(sb.ncol):
            cell = sb.cells[r][c]
            if cell is None:
                continue
            child, blk = cell
            panel_cell = gs[title_row_offset + 3 * r + 1, 3 * c + 1]
            right_cell_row = title_row_offset + 3 * r + 1
            right_cell_col = 3 * c + 2

            top_cell_row = title_row_offset + 3 * r
            panel_col = 3 * c + 1
            if r == 0 and outer_top_y is not None:
                child_top_y = outer_top_y
            else:
                child_top_y = gs[top_cell_row, panel_col].get_position(fig).y1

            if isinstance(blk, GuideAreaBlock):
                _render_guide_area_cell(blk, fig, panel_cell)
                continue
            if isinstance(blk, SuperBlock):
                outermost = parent_subspec is None
                child_lift_top = not _has_top_guide(blk)
                child_lift_bottom = not _has_bottom_guide(blk)
                child_lift_left = (
                    (c == 0) and (lift_left or outermost) and not _has_left_guide(blk)
                )
                child_lift_right = (
                    (c == sb.ncol - 1)
                    and (lift_right or outermost)
                    and not _has_right_guide(blk)
                )
                render_super_block(
                    blk,
                    fig,
                    parent_subspec=panel_cell,
                    tag_iter=tag_iter,
                    outer_top_y=child_top_y,
                    lift_top=child_lift_top,
                    lift_bottom=child_lift_bottom,
                    lift_left=child_lift_left,
                    lift_right=child_lift_right,
                )
            else:
                _render_leaf_cell(
                    child,
                    blk,
                    fig,
                    gs,
                    panel_cell,
                    right_cell_row,
                    right_cell_col,
                    top_cell_row,
                    panel_col,
                    tag_iter=tag_iter,
                    title_y_override=outer_top_y
                    if r == 0 and outer_top_y is not None
                    else None,
                )


def _render_leaf_title_in_top_cell(
    leaf,
    fig,
    gs,
    top_cell_row,
    panel_col,
    *,
    fontsize_title=None,
    fontsize_subtitle=None,
    y_override: float | None = None,
) -> None:
    """Render the leaf's title and subtitle as ``fig.text`` artists
    anchored to the TOP of the top-margin cell.
    """
    import matplotlib as mpl

    labels = leaf.labels or {}
    title = labels.get("title")
    subtitle = labels.get("subtitle")
    if not title and not subtitle:
        return

    if fontsize_title is None:
        fontsize_title = mpl.rcParams["axes.titlesize"]
    if fontsize_subtitle is None:
        fontsize_subtitle = "medium"

    cell = gs[top_cell_row, panel_col]
    bbox = cell.get_position(fig)
    cell_top_y = y_override if y_override is not None else bbox.y1
    y_cursor = cell_top_y - 0.005
    if title:
        fig.text(
            bbox.x0,
            y_cursor,
            str(title),
            ha="left",
            va="top",
            fontsize=fontsize_title,
            fontweight=mpl.rcParams["axes.titleweight"],
        )
        line_h_in = M.text_size_in(
            title,
            fontsize=mpl.rcParams["font.size"] * 1.2,
        )[1]
        y_cursor -= (line_h_in + M.ROW_GAP_IN) / fig.get_figheight()
    if subtitle:
        fig.text(
            bbox.x0,
            y_cursor,
            str(subtitle),
            ha="left",
            va="top",
            fontsize=fontsize_subtitle,
        )


def _render_leaf_cell(
    leaf,
    blk: PlotBlock,
    fig,
    gs,
    panel_cell,
    right_cell_row,
    right_cell_col,
    top_cell_row,
    panel_col,
    *,
    tag_iter=None,
    title_y_override: float | None = None,
) -> None:
    """Render a single ggplot leaf into its assigned cell, with cax for
    colorbars allocated in the right-margin column.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    from .render import _is_coord_polar

    bo = blk.build_output
    is_polar = _is_coord_polar(leaf.coordinates)
    polar_kw = {"projection": "polar"} if is_polar else {}

    if blk.n_panels == 1:
        ax = fig.add_subplot(panel_cell, **polar_kw)
        blk.panel_axes = [ax]
        blk.figure = fig
        cb_caxes = _allocate_colorbar_caxes(
            fig,
            gs,
            right_cell_row,
            right_cell_col,
            leaf,
            bo,
        )
        leg_hosts = _allocate_legend_host_axes(
            fig,
            gs,
            right_cell_row,
            right_cell_col,
            leaf,
            bo,
            block=blk,
        )
        _render_single_into(
            leaf, bo, ax, colorbar_caxes=cb_caxes, legend_host_axes=leg_hosts
        )
    else:
        sub_nrow = blk.panel_grid_rows
        sub_ncol = blk.panel_grid_cols
        sharex, sharey = leaf.facet.share_axes()
        sub_gs = GridSpecFromSubplotSpec(
            sub_nrow,
            sub_ncol,
            subplot_spec=panel_cell,
            wspace=0.05,
            hspace=0.20,
        )
        axes = []
        for sr in range(sub_nrow):
            row_axes = []
            for sc in range(sub_ncol):
                share_x = _share_anchor(sharex, sr, sc, axes, row_axes, axis="x")
                share_y = _share_anchor(sharey, sr, sc, axes, row_axes, axis="y")
                share_kw = {} if is_polar else {"sharex": share_x, "sharey": share_y}
                ax = fig.add_subplot(
                    sub_gs[sr, sc],
                    **share_kw,
                    **polar_kw,
                )
                row_axes.append(ax)
            axes.append(row_axes)
        blk.panel_axes = [a for row in axes for a in row]
        blk.figure = fig
        cb_caxes = _allocate_colorbar_caxes(
            fig,
            gs,
            right_cell_row,
            right_cell_col,
            leaf,
            bo,
        )
        leg_hosts = _allocate_legend_host_axes(
            fig,
            gs,
            right_cell_row,
            right_cell_col,
            leaf,
            bo,
            block=blk,
        )
        _render_facets_into(
            leaf,
            bo,
            axes,
            composing=True,
            colorbar_caxes=cb_caxes,
            legend_host_axes=leg_hosts,
        )

    _render_leaf_title_in_top_cell(
        leaf, fig, gs, top_cell_row, panel_col, y_override=title_y_override
    )

    if tag_iter is not None:
        tag = next(tag_iter, None)
        if tag is not None:
            corner_cell = gs[top_cell_row, panel_col - 1]
            bbox = corner_cell.get_position(fig)
            fig.text(
                bbox.x0,
                bbox.y1,
                tag,
                ha="left",
                va="top",
                fontsize="large",
                fontweight="bold",
            )


def _allocate_legend_host_axes(
    fig, gs, panel_row_idx, right_col_idx, leaf, bo, block=None
) -> list:
    """Carve a host ``Axes`` per discrete legend group inside the
    margin cell on the side dictated by ``legend.position``.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    pos = leaf.theme.get("legend.position") if leaf.theme else None
    if pos == "none":
        return []
    if pos not in (None, "right", "left", "top", "bottom"):
        return []

    from .guides import build_legend_groups

    groups = build_legend_groups(leaf, bo)
    if not groups:
        return []

    panel_col_idx = right_col_idx - 1
    left_col_idx = right_col_idx - 2
    top_row_idx = panel_row_idx - 1
    bottom_row_idx = panel_row_idx + 1
    if pos in (None, "right"):
        cell = gs[panel_row_idx, right_col_idx]
    elif pos == "left":
        cell = gs[panel_row_idx, left_col_idx]
    elif pos == "top":
        cell = gs[top_row_idx, panel_col_idx]
    else:  # "bottom"
        cell = gs[bottom_row_idx, panel_col_idx]

    if block is not None:
        if pos in ("top", "bottom"):
            total = block.margin_top_in if pos == "top" else block.margin_bottom_in
            legend_size = block.legend_h_in
            cbar_size = block.colorbar_h_in
        else:
            total = (
                block.margin_right_in
                if pos in (None, "right")
                else block.margin_left_in
            )
            legend_size = block.legend_w_in
            cbar_size = block.colorbar_w_in
        decoration_size = max(total - legend_size - cbar_size, 0.0)
    else:
        decoration_size = 0.0
        legend_size = 1.0
        cbar_size = 0.0

    if pos in ("top", "bottom"):
        if pos == "bottom":
            outer_ratios = [
                max(decoration_size, 1e-6),
                max(legend_size, 1e-6),
            ]
            legend_row = 1
        else:  # top
            outer_ratios = [
                max(legend_size, 1e-6),
                max(decoration_size, 1e-6),
            ]
            legend_row = 0
        outer = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=cell,
            height_ratios=outer_ratios,
            hspace=0.0,
        )
        inner = GridSpecFromSubplotSpec(
            1,
            len(groups),
            subplot_spec=outer[legend_row, 0],
            wspace=0.0,
        )
        hosts = []
        for i in range(len(groups)):
            host = fig.add_subplot(inner[0, i])
            host.set_label("<legend>")
            hosts.append(host)
        return hosts

    if pos == "left":
        outer_ratios = [
            max(decoration_size, 1e-6),  # ylabel/yticks closer to panel = right
            max(legend_size, 1e-6),  # legend on outer = left
        ]
        outer = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=cell,
            width_ratios=[outer_ratios[1], outer_ratios[0]],  # legend, then decoration
            wspace=0.0,
        )
        legend_subspec = outer[0, 0]
    else:
        legend_subspec = cell

    sub = GridSpecFromSubplotSpec(
        len(groups),
        1,
        subplot_spec=legend_subspec,
        wspace=0.0,
        hspace=0.1 if len(groups) > 1 else 0.0,
    )
    hosts = []
    for i in range(len(groups)):
        host = fig.add_subplot(sub[i, 0])
        host.set_label("<legend>")
        hosts.append(host)
    return hosts


def _allocate_colorbar_caxes(fig, gs, panel_row_idx, right_col_idx, leaf, bo) -> list:
    """Carve a tight cax (or stack of caxes) inside the right-margin
    cell for each colorbar in ``leaf``.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    from .guides import build_colorbar_specs

    pos = leaf.theme.get("legend.position") if leaf.theme else None
    if pos not in (None, "right"):
        return []

    specs = build_colorbar_specs(leaf, bo)
    if not specs:
        return []

    right_cell = gs[panel_row_idx, right_col_idx]
    tick_reserve = max(
        0.0,
        right_cell_width_in_estimate(fig, right_cell)
        - M.COLORBAR_PANEL_PAD_IN
        - M.COLORBAR_BAR_WIDTH_IN
        - M.COLORBAR_BAR_PAD_IN,
    )
    sub = GridSpecFromSubplotSpec(
        len(specs) * 2 + 1,
        4,
        subplot_spec=right_cell,
        width_ratios=[
            M.COLORBAR_PANEL_PAD_IN,
            M.COLORBAR_BAR_WIDTH_IN,
            M.COLORBAR_BAR_PAD_IN,
            max(tick_reserve, 1e-6),
        ],
        wspace=0.0,
        hspace=0.2,
    )
    caxes = []
    for i in range(len(specs)):
        cax = fig.add_subplot(sub[i * 2 + 1, 1])
        cax.set_label("<colorbar>")
        caxes.append(cax)
    return caxes


def right_cell_width_in_estimate(fig, subplotspec) -> float:
    """Approximate cell width in inches via its subplotspec position."""
    try:
        bbox = subplotspec.get_position(fig)
    except Exception:  # noqa: BLE001
        return 0.0
    return bbox.width * fig.get_figwidth()


def _apply_block_annotation(
    grid, fig, gs, title_row_offset, ncol, has_caption_row
) -> None:
    """Render plot_annotation title/caption into the reserved gridspec
    rows. Uses ``fig.text`` (not Axes) so ``fig.axes`` stays clean."""
    a = grid.annotation
    if a.title is not None or a.subtitle is not None:
        title_lines = [s for s in (a.title, a.subtitle) if s]
        bbox = gs[0, 0 : 3 * ncol].get_position(fig)
        import matplotlib as mpl

        fig.text(
            bbox.x0 + 0.01,
            (bbox.y0 + bbox.y1) / 2,
            "\n".join(str(s) for s in title_lines),
            ha="left",
            va="center",
            fontsize="large",
            fontweight=mpl.rcParams["axes.titleweight"],
        )
    if a.caption is not None and has_caption_row:
        bbox = gs[gs.nrows - 1, 0 : 3 * ncol].get_position(fig)
        fig.text(
            bbox.x1 - 0.01,
            (bbox.y0 + bbox.y1) / 2,
            str(a.caption),
            ha="right",
            va="center",
            fontsize="small",
            style="italic",
        )


def default_figsize_for(block: PlotBlock) -> tuple[float, float]:
    """Default standalone figure size for a measured block.

    Per-panel default of 3.5×3.0 inches. R's effective default depends
    on the rendering context (``dev.new()`` is 7×7, Quarto chunks are
    7×5, Jupyter is 7×4.32, …), so there's no single "R default" to
    match — we pick a per-panel size that's reasonable in notebook
    layouts and let users pass ``figsize=`` for exact parity. Faceted
    plots scale the panel cell by the grid dims.
    """
    panel_w = 3.5 * block.panel_grid_cols
    panel_h = 3.0 * block.panel_grid_rows
    fig_w = block.margin_left_in + panel_w + block.margin_right_in
    fig_h = block.margin_top_in + panel_h + block.margin_bottom_in
    return (fig_w, fig_h)
