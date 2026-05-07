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

from dataclasses import dataclass, field

import polars as pl

from . import _measure as M


# ---- Spacing system (all inches) -----------------------------------------
#
# Three tiers: (a) intra-decoration pads that match matplotlib's actual
# rendering so reserved cell sizes equal what gets drawn; (b) the
# generic ``ELEMENT_GAP`` between any two decoration elements (e.g.
# colorbar to panel, plot to legend); (c) ``BLOCK_GAP`` between sibling
# plot blocks in a compose. Define once, reference everywhere — no
# more inline 0.04 / 0.05 / 0.10 magic numbers scattered through the
# render path.

# (a) Pads matplotlib actually uses internally (so our reservations
# match what gets rendered):
_TICK_MARK_LEN_IN = 0.05      # rcParams['xtick.major.size'] = 3.5pt ≈ 0.05"
_TICK_TO_LABEL_PAD_IN = 0.05  # rcParams['xtick.major.pad']  = 3.5pt ≈ 0.05"
_LABEL_PAD_IN = 0.06          # rcParams['axes.labelpad']    = 4pt   ≈ 0.06"

# (b) Generic separation between any two adjacent decoration elements
# (e.g. between an axis label and the figure edge, between a legend and
# a colorbar in stacked guides). Uniform 0.10" — ggplot2's default
# ``theme(plot.margin)`` would correspond to about this.
ELEMENT_GAP_IN = 0.10

# (c) Whitespace between sibling plot blocks in a patchwork compose
# (panel-to-panel gap, also between a plot's panel and an adjacent
# plot's decorations).
BLOCK_GAP_IN = 0.20

# (Existing names kept for backwards reference but redefined in terms
# of the system above.)
_PANEL_MARGIN_PAD_IN = _TICK_TO_LABEL_PAD_IN
_AXIS_LABELPAD_IN = _LABEL_PAD_IN
_TICK_MARK_PAD_IN = _TICK_MARK_LEN_IN + _TICK_TO_LABEL_PAD_IN + 0.02

# Tick-label reserve when we can't measure actual labels. ggplot2's
# font is 11pt; ~5 char labels (e.g. "0.500") give the width / height
# seed below; pad covers tick mark + tick-to-label gap + safety.
_DEFAULT_YTICK_RESERVE_IN = (
    M.text_size_in("00000", fontsize=M.AXIS_TEXT_SIZE_PT)[0]
    + _TICK_MARK_PAD_IN
)
_DEFAULT_XTICK_RESERVE_IN = (
    M.text_size_in("0", fontsize=M.AXIS_TEXT_SIZE_PT)[1]
    + _TICK_MARK_PAD_IN
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

    # For faceted plots, the panel cell hosts a nrow×ncol sub-grid of
    # facet panels (each has its own strip / data area). For non-faceted
    # plots, both are 1.
    panel_grid_rows: int = 1
    panel_grid_cols: int = 1

    # After ``render_block``, the realized matplotlib axes for each panel
    # (length == panel_grid_rows * panel_grid_cols, row-major).
    panel_axes: list = field(default_factory=list)

    # Filled in during render to support legend/colorbar placement.
    figure: object | None = None

    # ------------------------------------------------------------------

    @property
    def n_panels(self) -> int:
        return self.panel_grid_rows * self.panel_grid_cols

    # ---- Outer-margin protocol shared with SuperBlock ----------------
    # Composition takes max of these across siblings sharing a row/col.

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


# ---- Measurement ----------------------------------------------------------

def measure_block(plot, build_output) -> PlotBlock:
    """Compute per-side margin sizes (inches) for ``plot``.

    Reads ``plot.labels`` for title/subtitle/xlab/ylab/caption, queries the
    scales/aes_source for legend and colorbar presence, and consults the
    facet for panel grid dims. Doesn't touch matplotlib state — pure.
    """
    from .render import _default_labels  # avoid circular at module load

    labels = plot.labels or {}

    # --- TOP margin: title + subtitle (matplotlib default sizing).
    # Reserve room separately for each so sibling plots in compose mode
    # share a baseline. Faceted plots also need room for strip labels
    # (added after the facet check below).
    import matplotlib as mpl
    title = labels.get("title")
    subtitle = labels.get("subtitle")
    title_h = M.text_size_in(
        title, fontsize=mpl.rcParams["axes.titlesize"],
        weight=mpl.rcParams["axes.titleweight"],
    )[1]
    subtitle_h = M.text_size_in(
        subtitle, fontsize="medium",
    )[1]
    margin_top = 0.0
    if title_h > 0:
        margin_top += title_h + M.ROW_GAP_IN
    if subtitle_h > 0:
        margin_top += subtitle_h + M.ROW_GAP_IN

    # --- LEFT margin: ylab (rotated 90) + ytick reserve.
    # ylab gap uses ``axes.labelpad`` (matplotlib's actual rendering)
    # — same rationale as the xlab side.
    xlabel, ylabel = _default_labels(plot)
    ylab_w = M.text_size_in(
        ylabel, fontsize=M.AXIS_TITLE_SIZE_PT, rotation=90.0,
    )[0]
    margin_left = ylab_w + _DEFAULT_YTICK_RESERVE_IN + _PANEL_MARGIN_PAD_IN
    if ylab_w > 0:
        margin_left += _AXIS_LABELPAD_IN

    # --- BOTTOM margin: xtick reserve + xlab + caption.
    # The xlab pad uses matplotlib's ``axes.labelpad`` (4pt ≈ 0.06")
    # rather than our generic ROW_GAP_IN — otherwise the reserved
    # space falls short of what matplotlib actually renders, leaving
    # the xlab to encroach on the next plot's panel in vertical
    # compose (e.g. ``p1 / p2``).
    xlab_h = M.text_size_in(xlabel, fontsize=M.AXIS_TITLE_SIZE_PT)[1]
    caption = labels.get("caption")
    caption_h = M.text_size_in(caption, fontsize=M.CAPTION_SIZE_PT)[1]
    margin_bottom = _DEFAULT_XTICK_RESERVE_IN + _PANEL_MARGIN_PAD_IN
    if xlab_h > 0:
        margin_bottom += xlab_h + _AXIS_LABELPAD_IN
    if caption_h > 0:
        margin_bottom += caption_h + M.ROW_GAP_IN

    # --- RIGHT margin: colorbar + legend (separate, additive)
    cbar_w = _measure_colorbar_width(plot, build_output)
    legend_w, _ = _measure_legend_size(plot, build_output)
    margin_right = cbar_w + legend_w
    if margin_right > 0:
        margin_right += M.COL_GAP_IN

    # --- Facet grid dims
    layout = build_output.layout
    n_panels = 1 if layout is None else len(layout)
    if n_panels > 1:
        nrow, ncol = plot.facet.grid_dims(n_panels)
        # Strip labels paint ``ax.set_title`` above each facet panel —
        # they overflow into the leaf's top-margin cell otherwise.
        # Reserve their height in margin_top so the title stays above.
        margin_top += M.strip_cell_height_in("Sample")
    else:
        nrow, ncol = 1, 1

    return PlotBlock(
        plot=plot,
        build_output=build_output,
        margin_left_in=margin_left,
        margin_right_in=margin_right,
        margin_top_in=margin_top,
        margin_bottom_in=margin_bottom,
        panel_grid_rows=nrow,
        panel_grid_cols=ncol,
    )


def _measure_legend_size(plot, build_output) -> tuple[float, float]:
    """Approximate the legend's (w, h) in inches for the right margin."""
    pos = plot.theme.get("legend.position") if plot.theme else None
    if pos == "none":
        return (0.0, 0.0)

    from .guides import build_legend_groups
    groups = build_legend_groups(plot, build_output)
    if not groups:
        return (0.0, 0.0)

    # Stack groups vertically (ggplot2 default). Width = max group width;
    # height = sum.
    widths = []
    height = 0.0
    for g in groups:
        w, h = M.legend_cell_size_in(g.title, g.labels)
        widths.append(w)
        height += h
    return (max(widths), height)


def _measure_colorbar_width(plot, build_output) -> float:
    """Width (inches) of the colorbar cell for the right margin."""
    pos = plot.theme.get("legend.position") if plot.theme else None
    if pos == "none":
        return 0.0

    from .guides import build_colorbar_specs
    specs = build_colorbar_specs(plot, build_output)
    if not specs:
        return 0.0

    # One colorbar per spec, stacked. We size for the widest tick label
    # we'd plausibly print (4 sig figs — refine if a diff item complains).
    sample_labels = ["0.0", "0.5", "1.0"]
    return max(M.colorbar_cell_width_in(sample_labels) for _ in specs)


# ---- Render ---------------------------------------------------------------

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

    # We need a positive panel size (the residual after margins). Sum of
    # all three width ratios is whatever — matplotlib normalizes — but we
    # need the panel ratio to be nonzero. Use a baseline that keeps the
    # panel a sensible size relative to its margins.
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
    # GridSpec requires strictly positive ratios. A 0 margin (e.g. no
    # title, no caption) gets a tiny floor so matplotlib doesn't choke;
    # the visual effect is negligible.
    width_ratios = [max(r, 1e-6) for r in width_ratios]
    height_ratios = [max(r, 1e-6) for r in height_ratios]

    if subplotspec is None:
        gs = GridSpec(
            3, 3, figure=fig,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            left=0.0, right=1.0, top=1.0, bottom=0.0,
            wspace=0.0, hspace=0.0,
        )
    else:
        gs = GridSpecFromSubplotSpec(
            3, 3,
            subplot_spec=subplotspec,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=0.0, hspace=0.0,
        )

    panel_cell = gs[1, 1]
    block.figure = fig

    if block.n_panels == 1:
        ax = fig.add_subplot(panel_cell)
        block.panel_axes = [ax]
        # Pre-allocate cax/legend host in the right-margin cell so
        # fig.colorbar doesn't shrink the panel and the legend stays
        # bounded by its host (no overflow into adjacent figure space).
        cb_caxes = _allocate_colorbar_caxes(fig, gs, 1, 2, plot, build_output)
        leg_hosts = _allocate_legend_host_axes(fig, gs, 1, 2, plot, build_output)
        _render_single_into(plot, build_output, ax,
                              colorbar_caxes=cb_caxes,
                              legend_host_axes=leg_hosts)
    else:
        nrow, ncol = block.panel_grid_rows, block.panel_grid_cols
        sharex, sharey = plot.facet.share_axes()
        sub_gs = GridSpecFromSubplotSpec(
            nrow, ncol, subplot_spec=panel_cell,
            wspace=0.05, hspace=0.20,
        )
        axes = []
        for r in range(nrow):
            row_axes = []
            for c in range(ncol):
                share_x_with = _share_anchor(sharex, r, c, axes, row_axes, axis="x")
                share_y_with = _share_anchor(sharey, r, c, axes, row_axes, axis="y")
                ax = fig.add_subplot(
                    sub_gs[r, c],
                    sharex=share_x_with,
                    sharey=share_y_with,
                )
                row_axes.append(ax)
            axes.append(row_axes)
        block.panel_axes = [ax for row in axes for ax in row]
        cb_caxes = _allocate_colorbar_caxes(fig, gs, 1, 2, plot, build_output)
        leg_hosts = _allocate_legend_host_axes(fig, gs, 1, 2, plot, build_output)
        _render_facets_into(plot, build_output, axes,
                              colorbar_caxes=cb_caxes,
                              legend_host_axes=leg_hosts)

    # Title/subtitle/caption text rides on the panel ``Axes`` (title via
    # ``ax.set_title(loc='left')``, caption via ``fig.text``). Margin
    # sizes already reserve the room; the existing renderer's
    # ``_apply_plot_titles`` does the actual placement so we don't add
    # extra Axes that would break tests counting ``fig.axes``.
    from .render import _apply_plot_titles
    if subplotspec is None:
        # Standalone — owns the figure, can use fig.suptitle / fig.text.
        _apply_plot_titles(plot, fig, ax_list=block.panel_axes)


def _render_single_into(plot, build_output, ax, *,
                          colorbar_caxes: list | None = None,
                          legend_host_axes: list | None = None) -> None:
    """Run the single-panel rendering pipeline against ``ax``.

    ``colorbar_caxes``: pre-allocated dedicated axes for any colorbars
    in this plot, so ``fig.colorbar`` doesn't shrink the panel. Block-
    engine composition allocates these in the right-margin column.
    """
    from .render import (
        _apply_theme, _is_coord_flip, _default_labels,
    )

    is_flipped = _is_coord_flip(plot.coordinates)
    ax._hea_coord_flipped = is_flipped

    for layer, df in zip(plot.layers, build_output.data):
        if is_flipped:
            from .coords.flip import flip_columns
            df = flip_columns(df)
        layer.geom.draw_panel(df, ax)

    if build_output.scales is not None:
        for axis in ("x", "y"):
            scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
            sc = build_output.scales.get(scale_aes)
            if sc is not None:
                sc.apply_to_axis(ax, axis)

    xlabel, ylabel = _default_labels(plot)
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
    apply_legends(ax.figure, [ax], plot, build_output,
                   colorbar_caxes=colorbar_caxes,
                   legend_host_axes=legend_host_axes)


def _render_facets_into(plot, build_output, axes_grid, *,
                          composing: bool = False,
                          colorbar_caxes: list | None = None,
                          legend_host_axes: list | None = None) -> None:
    """Render each facet panel into its allocated axes.

    ``composing=True`` skips ``fig.supxlabel``/``supylabel`` — those paint
    across the whole figure, which is wrong when the figure also hosts
    sibling plots. In that case we set the axis label on the bottom-row
    centre panel and left-column middle panel so it lives inside this
    leaf's panel column area."""
    from .render import _apply_theme, _is_coord_flip, _default_labels

    facet = plot.facet
    layout = build_output.layout
    is_flipped = _is_coord_flip(plot.coordinates)
    flat_axes = [ax for row in axes_grid for ax in row]
    n_panels = len(layout)

    for panel_row in layout.iter_rows(named=True):
        idx = panel_row["PANEL"] - 1
        panel_ax = flat_axes[idx]
        panel_ax._hea_coord_flipped = is_flipped

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

        if build_output.scales is not None:
            for axis in ("x", "y"):
                scale_aes = ("y" if axis == "x" else "x") if is_flipped else axis
                sc = build_output.scales.get(scale_aes)
                if sc is not None:
                    sc.apply_to_axis(panel_ax, axis)

        labels = facet.panel_labels(panel_row, layout)
        if labels.get("top"):
            panel_ax.set_title(labels["top"])
        if labels.get("right"):
            panel_ax.text(
                1.02, 0.5, labels["right"],
                transform=panel_ax.transAxes,
                rotation=-90, ha="left", va="center",
            )

    for unused_ax in flat_axes[n_panels:]:
        unused_ax.set_visible(False)

    # Hide redundant tick labels on inner panels when scales are shared.
    # ``sharex`` shares limits across columns/all → tick labels on the
    # non-bottom rows are redundant; same for ``sharey`` and non-left
    # columns. matplotlib's ``add_subplot(sharex=other)`` links the axes
    # but doesn't auto-hide the labels (only ``plt.subplots()`` does).
    sharex, sharey = facet.share_axes()
    _hide_redundant_facet_ticks(axes_grid, sharex, sharey, n_panels)

    fig = flat_axes[0].figure
    xlabel, ylabel = _default_labels(plot)
    if is_flipped:
        xlabel, ylabel = ylabel, xlabel
    # Always use the bbox-aware placement — even for standalone faceted
    # plots. ``fig.supxlabel`` / ``supylabel`` use matplotlib's default
    # fig-rel x position which doesn't reserve room for our wider tick
    # labels, leading to the ylabel kissing or overlapping the leftmost
    # panel's yticks.
    _set_facet_axis_labels(fig, flat_axes[:n_panels], xlabel, ylabel)

    _apply_theme(
        plot.theme, fig, list(flat_axes[:n_panels]), owns_fig=False,
    )

    apply = getattr(plot.coordinates, "apply_to_axes", None)
    if apply is not None:
        for panel_ax in flat_axes[:n_panels]:
            apply(panel_ax)

    from .guides import apply_axis_guides, apply_legends
    apply_axis_guides(list(flat_axes[:n_panels]), plot)
    apply_legends(fig, list(flat_axes[:n_panels]), plot, build_output,
                   colorbar_caxes=colorbar_caxes,
                   legend_host_axes=legend_host_axes)


def _set_facet_axis_labels(fig, panel_axes: list, xlabel, ylabel) -> None:
    """Place ``xlabel`` / ``ylabel`` via ``fig.text`` at the union bbox of
    ``panel_axes`` — so the label spans the whole panel area of one facet
    leaf, not just a single panel.

    Offsets from the panel area use the SAME inch-based reserves that
    :func:`measure_block` allocated for tick labels and label-pad,
    converted to figure-relative units. This keeps the label cleanly
    outside the tick text — without it the ylabel can overlap with a
    wide leftmost ytick label like ``"6000"``.
    """
    if not panel_axes:
        return
    if xlabel is None and ylabel is None:
        return

    bboxes = []
    for ax in panel_axes:
        try:
            bbox = ax.get_subplotspec().get_position(fig)
        except Exception:
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
        # Reserve room for xtick labels first, then a labelpad gap, then
        # the xlabel itself. ``y_offset`` measures from panel.y0 down to
        # the xlabel's TOP edge.
        y_offset_in = _DEFAULT_XTICK_RESERVE_IN + _AXIS_LABELPAD_IN
        fig.text(cx, y0 - y_offset_in / fig_h, xlabel,
                 ha="center", va="top",
                 fontsize="medium")
    if ylabel is not None:
        # Same on the left: ytick reserve + labelpad before the ylabel.
        x_offset_in = _DEFAULT_YTICK_RESERVE_IN + _AXIS_LABELPAD_IN
        fig.text(x0 - x_offset_in / fig_w, cy, ylabel,
                 ha="right", va="center",
                 rotation=90, fontsize="medium")


def _hide_redundant_facet_ticks(axes_grid, sharex, sharey, n_panels: int) -> None:
    """When facet panels share scales, only the bottom row's xtick labels
    and the leftmost column's ytick labels are informative — the rest are
    redundant and visually cluttering when panels pack tightly.

    For ``sharex=True`` / ``'col'``: hide ``labelbottom`` on every row
    except the lowest row that contains a *visible* panel in that column.
    Empty trailing cells in :func:`facet_wrap` (e.g. n=5 in a 2×3 grid)
    expose the panel above them, so we walk each column from the bottom
    looking for the first visible panel.

    Same logic for ``sharey=True`` / ``'row'`` and the leftmost-visible
    column per row. ``sharex='row'`` and ``sharey='col'`` are unusual
    and left untouched."""
    nrow = len(axes_grid)
    ncol = len(axes_grid[0]) if nrow else 0
    if nrow == 0 or ncol == 0:
        return

    # Visibility map — facet_wrap may hide trailing cells; we treat
    # those as "not present" for tick-bookkeeping.
    visible = [
        [ax.get_visible() for ax in row]
        for row in axes_grid
    ]

    if sharex in (True, "col"):
        # For each column, find the lowest visible panel — that's the one
        # that keeps labelbottom; everything above it loses both the
        # tick marks and labels (R hides both on inner panels).
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
                    bottom=False, top=False,
                    labelbottom=False, labeltop=False,
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
                    left=False, right=False,
                    labelleft=False, labelright=False,
                )


def _share_anchor(spec, r: int, c: int, axes_grid, row_axes, *, axis: str):
    """Pick the anchor axes for matplotlib ``sharex=`` / ``sharey=``.

    ``spec`` mirrors :meth:`Facet.share_axes` return values:
    ``True`` (share with (0,0)), ``False`` (no share), ``'col'`` (share
    within column — pick column's first row), ``'row'`` (share within
    row — pick row's first column). ``axes_grid`` only contains
    completed rows; the current row in progress is in ``row_axes``.
    """
    if not spec:
        return None
    if r == 0 and c == 0:
        return None
    if spec is True:
        # Anchor on (0, 0). On row 0 it lives in ``row_axes`` (current
        # row hasn't been appended to ``axes_grid`` yet); on later rows
        # it's in ``axes_grid[0][0]``.
        return row_axes[0] if r == 0 else axes_grid[0][0]
    if spec == "col":
        # Share within a column — anchor is row 0 of this column.
        if r == 0:
            return None
        return axes_grid[0][c]
    if spec == "row":
        # Share within a row — anchor is column 0 of this row.
        if c == 0:
            return None
        return row_axes[0]
    return None


# =====================================================================
# Composition primitives — used by PlotGrid for nested rendering.
# =====================================================================

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
        if build_legend_groups(plot, bo):
            return True
        return False
    if isinstance(blk, SuperBlock):
        # Walk children in the rightmost col.
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
    """Mirror of :func:`_has_right_guide` for the left side. Today no
    decoration ever lives in the leftmost col except yticks/ylabel
    (which we WANT to lift), so this returns False for typical leaves;
    kept for symmetry / future left-side legend support."""
    return False


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
    # Cells row-major; each is either (child, child_block) or None.
    cells: list  # list[list[tuple | None]]
    # Per-row top/bottom and per-col left/right — max across siblings.
    row_super_top_in: list
    row_super_bottom_in: list
    col_super_left_in: list
    col_super_right_in: list
    # Per-row panel height and per-col panel width (defaults or user-overridden).
    panel_h_in: list
    panel_w_in: list
    # Annotation row heights (top/bottom) — 0 when absent.
    annot_title_h_in: float = 0.0
    annot_caption_h_in: float = 0.0

    # ---- Outer-margin protocol ----
    # Composition takes max of these across siblings sharing a row/col.

    @property
    def outer_margin_top_in(self) -> float:
        # Top decoration above ALL panels = first row's super_top + any
        # plot_annotation title row.
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

    # ---- Total inner extents (panel + INNER margins between cells). ----

    @property
    def total_inner_w_in(self) -> float:
        """Width of the ``panel area`` of this super-block (between outer_left
        and outer_right). Sums per-col panel + per-col inner left/right
        margins (the latter excluded for outer cols)."""
        w = 0.0
        for c in range(self.ncol):
            w += self.panel_w_in[c]
            # Add inner-margin contributions for non-edge columns.
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
        return self.outer_margin_left_in + self.total_inner_w_in + self.outer_margin_right_in

    @property
    def total_h_in(self) -> float:
        return self.outer_margin_top_in + self.total_inner_h_in + self.outer_margin_bottom_in


def compute_block(thing):
    """Return a block for ``thing`` (a ``ggplot`` leaf or a ``PlotGrid``).

    Recurses into nested grids. The returned object exposes the
    ``outer_margin_*`` properties so the parent's super-margin
    computation works uniformly.
    """
    from .core import ggplot
    from .patchwork import PlotGrid
    from .build import build

    if isinstance(thing, ggplot):
        bo = build(thing)
        blk = measure_block(thing, bo)
        # Cache the build output on the block so the renderer doesn't
        # re-run build later.
        return blk
    if isinstance(thing, PlotGrid):
        return compose_super_block(thing)
    raise TypeError(
        f"compute_block: unsupported child type {type(thing).__name__}"
    )


def compose_super_block(grid) -> SuperBlock:
    """Recursively measure a :class:`PlotGrid` → :class:`SuperBlock`.

    Each child is a leaf ``PlotBlock`` or a nested ``SuperBlock``. Super
    margins per row/col are taken as the max of children's
    ``outer_margin_*`` along that axis (except the outermost rows/cols
    which contribute to *this* SuperBlock's outer margins instead).
    """
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
        cells[r][c] = (child, compute_block(child))

    # Per-row top/bottom: max across this row's children.
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

    # Panel size per row/col. For nested SuperBlocks, the panel cell must
    # accommodate the nested's full inner extent — otherwise the nested's
    # panels would scale up to fill a too-large allocation.
    panel_h = [DEFAULT_PANEL_H_IN] * nrow
    panel_w = [DEFAULT_PANEL_W_IN] * ncol
    for r in range(nrow):
        for c in range(ncol):
            cell = cells[r][c]
            if cell is None:
                continue
            _, blk = cell
            if isinstance(blk, SuperBlock):
                # Nested grid — panel cell must be ≥ nested's inner extent.
                panel_h[r] = max(panel_h[r], blk.total_inner_h_in)
                panel_w[c] = max(panel_w[c], blk.total_inner_w_in)

    if grid.widths is not None:
        total = sum(grid.widths)
        avg = DEFAULT_PANEL_W_IN * ncol / total if total > 0 else DEFAULT_PANEL_W_IN
        panel_w = [w * avg for w in grid.widths]
    if grid.heights is not None:
        total = sum(grid.heights)
        avg = DEFAULT_PANEL_H_IN * nrow / total if total > 0 else DEFAULT_PANEL_H_IN
        panel_h = [h * avg for h in grid.heights]

    annot_title_h, annot_caption_h = _annotation_extents(grid)

    return SuperBlock(
        grid=grid,
        nrow=nrow, ncol=ncol,
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
        title_lines, fontsize=M.TITLE_SIZE_PT, weight="bold",
    )[1]
    if title_h > 0:
        title_h += 0.1
    caption_h = M.text_size_in(a.caption, fontsize=M.CAPTION_SIZE_PT)[1]
    if caption_h > 0:
        caption_h += 0.05
    return (title_h, caption_h)


def render_super_block(sb: SuperBlock, fig, parent_subspec=None,
                        tag_iter=None, outer_top_y: float | None = None,
                        lift_top: bool = False,
                        lift_bottom: bool = False,
                        lift_left: bool = False,
                        lift_right: bool = False) -> None:
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
    if sb.annot_title_h_in > 0:
        height_ratios.append(sb.annot_title_h_in)
    for r in range(sb.nrow):
        # When the parent has lifted our top decoration into its own
        # top-margin (lift_top), zero out our first super-top row so
        # the panel sits flush at the cell top — matching the parent
        # leaf siblings whose panels live at the cell top edge.
        # Same for lift_bottom on the last row.
        super_top = sb.row_super_top_in[r]
        super_bottom = sb.row_super_bottom_in[r]
        if r == 0 and lift_top:
            super_top = 0.0
        if r == sb.nrow - 1 and lift_bottom:
            super_bottom = 0.0
        height_ratios.extend([
            super_top,
            sb.panel_h_in[r],
            super_bottom,
        ])
    if sb.annot_caption_h_in > 0:
        height_ratios.append(sb.annot_caption_h_in)

    width_ratios: list[float] = []
    for c in range(sb.ncol):
        # Symmetric to the row lift: collapse our outermost super-left /
        # super-right when the parent has reserved that space — keeps
        # the leftmost / rightmost child's panel flush with the cell
        # edges so it aligns with the parent's leaf siblings (e.g.
        # ``p1 / (p2 | p3)`` — p2.panel.left == p1.panel.left).
        super_left = sb.col_super_left_in[c]
        super_right = sb.col_super_right_in[c]
        if c == 0 and lift_left:
            super_left = 0.0
        if c == sb.ncol - 1 and lift_right:
            super_right = 0.0
        width_ratios.extend([
            super_left,
            sb.panel_w_in[c],
            super_right,
        ])

    gs_h = [max(r, 1e-6) for r in height_ratios]
    gs_w = [max(r, 1e-6) for r in width_ratios]
    title_row_offset = 1 if sb.annot_title_h_in > 0 else 0

    if parent_subspec is None:
        gs = GridSpec(
            len(gs_h), len(gs_w), figure=fig,
            height_ratios=gs_h, width_ratios=gs_w,
            left=0.0, right=1.0, top=1.0, bottom=0.0,
            wspace=0.0, hspace=0.0,
        )
    else:
        gs = GridSpecFromSubplotSpec(
            len(gs_h), len(gs_w), subplot_spec=parent_subspec,
            height_ratios=gs_h, width_ratios=gs_w,
            wspace=0.0, hspace=0.0,
        )

    # plot_annotation title/caption (only at the outermost SuperBlock).
    if grid.annotation is not None:
        _apply_block_annotation(grid, fig, gs, title_row_offset, sb.ncol,
                                  sb.annot_caption_h_in > 0)

    # Render each cell.
    for r in range(sb.nrow):
        for c in range(sb.ncol):
            cell = sb.cells[r][c]
            if cell is None:
                continue
            child, blk = cell
            panel_cell = gs[title_row_offset + 3*r + 1, 3*c + 1]
            right_cell_row = title_row_offset + 3*r + 1
            right_cell_col = 3*c + 2

            top_cell_row = title_row_offset + 3*r
            panel_col = 3*c + 1
            # Topmost-row children get title-lifting: their titles land
            # at ``outer_top_y`` if the outer asked, else at this
            # SuperBlock's own top-margin cell (the row 0 of our grid).
            if r == 0:
                if outer_top_y is not None:
                    child_top_y = outer_top_y
                else:
                    child_top_y = gs[top_cell_row, panel_col].get_position(fig).y1
            else:
                child_top_y = None  # use inner cell's top as usual

            if isinstance(blk, SuperBlock):
                # Forward simplify_gt-style lifting to the nested grid.
                # Top/bottom: lift when this child is at row 0 / row
                # last AND we ourselves were lifted (or we're the
                # outermost SuperBlock — parent_subspec is None).
                # Left/right: same logic per col, BUT skip when the
                # nested's right-edge col hosts a colorbar/legend —
                # collapsing that col would zero out the guide's cax /
                # host axes and the guide would render at zero width.
                outermost = parent_subspec is None
                child_lift_top = (r == 0) and (lift_top or outer_top_y is not None or outermost)
                child_lift_bottom = (r == sb.nrow - 1) and (lift_bottom or outermost)
                child_lift_left = (
                    (c == 0) and (lift_left or outermost)
                    and not _has_left_guide(blk)
                )
                child_lift_right = (
                    (c == sb.ncol - 1) and (lift_right or outermost)
                    and not _has_right_guide(blk)
                )
                render_super_block(blk, fig,
                                    parent_subspec=panel_cell,
                                    tag_iter=tag_iter,
                                    outer_top_y=child_top_y,
                                    lift_top=child_lift_top,
                                    lift_bottom=child_lift_bottom,
                                    lift_left=child_lift_left,
                                    lift_right=child_lift_right)
            else:
                _render_leaf_cell(child, blk, fig, gs, panel_cell,
                                   right_cell_row, right_cell_col,
                                   top_cell_row, panel_col,
                                   tag_iter=tag_iter,
                                   title_y_override=child_top_y if r == 0 and outer_top_y is not None else None)


def _render_leaf_title_in_top_cell(leaf, fig, gs, top_cell_row, panel_col,
                                     *, fontsize_title=None,
                                     fontsize_subtitle=None,
                                     y_override: float | None = None) -> None:
    """Render the leaf's title and subtitle as ``fig.text`` artists
    anchored to the TOP of the top-margin cell.

    Matches matplotlib's default ``axes.titlesize`` ("large") /
    ``axes.titleweight`` ("normal") so the styling stays consistent
    with our pre-block-engine ``ax.set_title(loc='left')`` rendering.
    Subtitle uses a smaller font (matches ggplot2's relative sizing).

    Anchoring at the cell top (rather than above the panel) keeps
    sibling titles aligned even when one sibling has a subtitle and
    the other doesn't — the super-grid reserves the same top margin
    for both via :func:`measure_block`.
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
        # Subtitle smaller than title; matplotlib calls this "medium".
        fontsize_subtitle = "medium"

    cell = gs[top_cell_row, panel_col]
    bbox = cell.get_position(fig)
    # x always comes from this leaf's own panel column (so titles
    # of side-by-side leaves get distinct x positions). y can be
    # overridden by a parent compose to lift the title up to the
    # outer top-margin row.
    cell_top_y = y_override if y_override is not None else bbox.y1
    y_cursor = cell_top_y - 0.005
    if title:
        fig.text(
            bbox.x0, y_cursor, str(title),
            ha="left", va="top",
            fontsize=fontsize_title,
            fontweight=mpl.rcParams["axes.titleweight"],
        )
        line_h_in = M.text_size_in(
            title, fontsize=mpl.rcParams["font.size"] * 1.2,
        )[1]
        y_cursor -= (line_h_in + M.ROW_GAP_IN) / fig.get_figheight()
    if subtitle:
        fig.text(
            bbox.x0, y_cursor, str(subtitle),
            ha="left", va="top",
            fontsize=fontsize_subtitle,
        )


def _render_leaf_cell(leaf, blk: PlotBlock, fig, gs, panel_cell,
                       right_cell_row, right_cell_col,
                       top_cell_row, panel_col, *, tag_iter=None,
                       title_y_override: float | None = None) -> None:
    """Render a single ggplot leaf into its assigned cell, with cax for
    colorbars allocated in the right-margin column.

    ``top_cell_row`` and ``panel_col``: the gridspec row/col of the leaf's
    top-margin cell — used to place ``tag_levels`` text above the title
    instead of overlapping it.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from .render import _apply_plot_titles

    bo = blk.build_output
    if blk.n_panels == 1:
        ax = fig.add_subplot(panel_cell)
        blk.panel_axes = [ax]
        blk.figure = fig
        cb_caxes = _allocate_colorbar_caxes(
            fig, gs, right_cell_row, right_cell_col, leaf, bo,
        )
        leg_hosts = _allocate_legend_host_axes(
            fig, gs, right_cell_row, right_cell_col, leaf, bo,
        )
        _render_single_into(leaf, bo, ax,
                              colorbar_caxes=cb_caxes,
                              legend_host_axes=leg_hosts)
    else:
        sub_nrow = blk.panel_grid_rows
        sub_ncol = blk.panel_grid_cols
        sharex, sharey = leaf.facet.share_axes()
        sub_gs = GridSpecFromSubplotSpec(
            sub_nrow, sub_ncol, subplot_spec=panel_cell,
            wspace=0.05, hspace=0.20,
        )
        axes = []
        for sr in range(sub_nrow):
            row_axes = []
            for sc in range(sub_ncol):
                share_x = _share_anchor(sharex, sr, sc, axes, row_axes, axis="x")
                share_y = _share_anchor(sharey, sr, sc, axes, row_axes, axis="y")
                ax = fig.add_subplot(
                    sub_gs[sr, sc], sharex=share_x, sharey=share_y,
                )
                row_axes.append(ax)
            axes.append(row_axes)
        blk.panel_axes = [a for row in axes for a in row]
        blk.figure = fig
        cb_caxes = _allocate_colorbar_caxes(
            fig, gs, right_cell_row, right_cell_col, leaf, bo,
        )
        leg_hosts = _allocate_legend_host_axes(
            fig, gs, right_cell_row, right_cell_col, leaf, bo,
        )
        _render_facets_into(leaf, bo, axes, composing=True,
                              colorbar_caxes=cb_caxes,
                              legend_host_axes=leg_hosts)

    # Title/subtitle: render as fig.text. ``title_y_override`` lifts the
    # anchor up to a parent's top-margin (used by nested compositions
    # so that p1's and p2's titles align in ``p1 | (p2 / p3)``).
    _render_leaf_title_in_top_cell(leaf, fig, gs, top_cell_row, panel_col,
                                     y_override=title_y_override)

    if tag_iter is not None:
        tag = next(tag_iter, None)
        if tag is not None:
            # Place the tag at the upper-left CORNER of the leaf — in
            # the (top-margin, left-margin) cell, where the ylab column
            # meets the title row. The title lives in the panel column
            # so the two don't overlap.
            corner_cell = gs[top_cell_row, panel_col - 1]
            bbox = corner_cell.get_position(fig)
            fig.text(
                bbox.x0, bbox.y1, tag,
                ha="left", va="top",
                fontsize="large", fontweight="bold",
            )


def _allocate_legend_host_axes(fig, gs, panel_row_idx, right_col_idx,
                                  leaf, bo) -> list:
    """Carve a host ``Axes`` per discrete legend group inside the right-
    margin cell. The legend renders inside the host (via
    :func:`apply_legends` host path), so it stays bounded by the host's
    bbox — preventing the legend from extending into the next plot's
    panel area in a horizontal compose.

    Multiple groups stack vertically. Returns ``[]`` when the leaf has
    no discrete legend or when ``legend.position`` is ``"none"`` /
    ``"top"`` / ``"bottom"`` / ``"left"`` (those need a different cell;
    fall back to the legacy panel-relative path)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    pos = leaf.theme.get("legend.position") if leaf.theme else None
    if pos not in (None, "right"):
        return []

    from .guides import build_legend_groups
    groups = build_legend_groups(leaf, bo)
    if not groups:
        return []

    right_cell = gs[panel_row_idx, right_col_idx]
    sub = GridSpecFromSubplotSpec(
        len(groups), 1, subplot_spec=right_cell,
        wspace=0.0, hspace=0.1,
    )
    hosts = []
    for i in range(len(groups)):
        host = fig.add_subplot(sub[i, 0])
        host.set_label("<legend>")
        hosts.append(host)
    return hosts


def _allocate_colorbar_caxes(fig, gs, panel_row_idx, right_col_idx,
                               leaf, bo) -> list:
    """Carve a tight cax (or stack of caxes) inside the right-margin
    cell for each colorbar in ``leaf``.

    Only allocates for the default right-side colorbar placement. When
    the theme requests ``legend.position`` of ``"top"``/``"bottom"``/
    ``"left"``/``"none"``, we return ``[]`` and let the legacy
    auto-shrink path handle it — those placements need a cell on a
    different side which the block engine doesn't reserve yet.
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
    # Inch-absolute width ratios: panel-side pad, the bar, bar-to-tick pad,
    # tick text reserve. matplotlib normalizes these against the cell's
    # actual width — but since `_measure_colorbar_width` already sums to
    # exactly these inches, the cax lands at its measured size with the
    # panel-side pad acting as breathing room.
    tick_reserve = max(
        0.0,
        right_cell_width_in_estimate(fig, right_cell)
        - M.COLORBAR_PANEL_PAD_IN
        - M.COLORBAR_BAR_WIDTH_IN
        - M.COLORBAR_BAR_PAD_IN,
    )
    sub = GridSpecFromSubplotSpec(
        len(specs) * 2 + 1, 4,
        subplot_spec=right_cell,
        width_ratios=[
            M.COLORBAR_PANEL_PAD_IN,
            M.COLORBAR_BAR_WIDTH_IN,
            M.COLORBAR_BAR_PAD_IN,
            max(tick_reserve, 1e-6),
        ],
        wspace=0.0, hspace=0.2,
    )
    caxes = []
    for i in range(len(specs)):
        # cax = the bar column only (col 1). Tick labels render in col 3
        # because matplotlib renders tick text outside the cax.
        cax = fig.add_subplot(sub[i * 2 + 1, 1])
        cax.set_label("<colorbar>")
        caxes.append(cax)
    return caxes


def right_cell_width_in_estimate(fig, subplotspec) -> float:
    """Approximate cell width in inches via its subplotspec position."""
    try:
        bbox = subplotspec.get_position(fig)
    except Exception:
        return 0.0
    return bbox.width * fig.get_figwidth()


def _apply_block_annotation(grid, fig, gs, title_row_offset, ncol,
                              has_caption_row) -> None:
    """Render plot_annotation title/caption into the reserved gridspec
    rows. Uses ``fig.text`` (not Axes) so ``fig.axes`` stays clean."""
    a = grid.annotation
    if a.title is not None or a.subtitle is not None:
        title_lines = [s for s in (a.title, a.subtitle) if s]
        bbox = gs[0, 0:3*ncol].get_position(fig)
        fig.text(
            bbox.x0 + 0.01, (bbox.y0 + bbox.y1) / 2,
            "\n".join(str(s) for s in title_lines),
            ha="left", va="center",
            fontsize="large", fontweight="bold",
        )
    if a.caption is not None and has_caption_row:
        bbox = gs[gs.nrows - 1, 0:3*ncol].get_position(fig)
        fig.text(
            bbox.x1 - 0.01, (bbox.y0 + bbox.y1) / 2,
            str(a.caption),
            ha="right", va="center",
            fontsize="small", style="italic",
        )


def default_figsize_for(block: PlotBlock) -> tuple[float, float]:
    """Default standalone figure size for a measured block.

    Per-panel default of 3.5×3.0 inches (ggplot2's ``options(repr.plot.*=*)``
    default is 7×7 for one panel; we go closer to a typical Quarto/RMD
    output). Faceted plots scale the panel cell by the grid dims.
    """
    panel_w = 3.5 * block.panel_grid_cols
    panel_h = 3.0 * block.panel_grid_rows
    fig_w = block.margin_left_in + panel_w + block.margin_right_in
    fig_h = block.margin_top_in + panel_h + block.margin_bottom_in
    return (fig_w, fig_h)
