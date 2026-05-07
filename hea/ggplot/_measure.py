"""Off-screen measurement of decoration sizes (in inches).

The patchwork-gtable layout engine needs to know cell sizes BEFORE it
allocates the figure's gridspec. matplotlib gives text extents only via
a renderer, so we keep a lazy off-screen Agg figure as a measurement
surface — every leaf plot, every composition pass queries through it.

All public sizes are returned in inches (matplotlib uses pixels with a
``dpi`` divisor; we hide that here). Dpi-independence matters because
the final figure may render at a different dpi than our measurement
surface — physical inches are the invariant.
"""

from __future__ import annotations

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text


# Internal measurement figure. Lazily created on first call. Its dpi is
# fixed; we always convert pixel extents to inches before returning, so
# the chosen value doesn't leak.
_MEASURE_DPI = 100.0
_measure_fig: Figure | None = None


def _figure() -> Figure:
    global _measure_fig
    if _measure_fig is None:
        fig = Figure(figsize=(1.0, 1.0), dpi=_MEASURE_DPI)
        FigureCanvasAgg(fig)
        _measure_fig = fig
    return _measure_fig


# ---- ggplot2 theme_grey font sizes (in points) ----------------------------
# We mirror the ggplot2 defaults so measured sizes match the rendered
# ones. Callers that override the theme should pass an explicit fontsize.

BASE_SIZE_PT = 11.0
TITLE_SIZE_PT = BASE_SIZE_PT * 1.2          # plot.title, plot.subtitle (subtitle inherits)
AXIS_TITLE_SIZE_PT = BASE_SIZE_PT           # axis.title
AXIS_TEXT_SIZE_PT = BASE_SIZE_PT * 0.8      # axis.text
CAPTION_SIZE_PT = BASE_SIZE_PT * 0.8        # plot.caption
LEGEND_TITLE_SIZE_PT = BASE_SIZE_PT
LEGEND_TEXT_SIZE_PT = BASE_SIZE_PT * 0.8
STRIP_TEXT_SIZE_PT = BASE_SIZE_PT * 0.8     # facet strip


# ---- Primitive: measure a single text artist ------------------------------

def text_size_in(
    text: str | None,
    *,
    fontsize: float = BASE_SIZE_PT,
    family: str | None = None,
    weight: str | None = None,
    rotation: float = 0.0,
) -> tuple[float, float]:
    """Return ``(width_in, height_in)`` for ``text`` rendered with the given font.

    Empty/None text returns ``(0.0, 0.0)`` — callers can use this as a
    sentinel for "no decoration on this side". Rotation is in degrees;
    the returned bbox is the axis-aligned bounding box AFTER rotation.
    """
    if not text:
        return (0.0, 0.0)
    fig = _figure()
    renderer = fig.canvas.get_renderer()
    artist = Text(
        x=0,
        y=0,
        text=text,
        fontsize=fontsize,
        family=family,
        weight=weight,
        rotation=rotation,
        figure=fig,
    )
    bbox = artist.get_window_extent(renderer=renderer)
    return (bbox.width / fig.dpi, bbox.height / fig.dpi)


# ---- Aggregates -----------------------------------------------------------

def text_block_size_in(
    lines: list[str] | tuple[str, ...],
    *,
    fontsize: float = BASE_SIZE_PT,
    family: str | None = None,
    weight: str | None = None,
    line_spacing: float = 1.2,
) -> tuple[float, float]:
    """Multi-line text. Width = max line width; height = N * lh * line_spacing.

    Used for title+subtitle stacks and any wrapped axis label.
    """
    lines = [s for s in lines if s]
    if not lines:
        return (0.0, 0.0)
    sizes = [text_size_in(s, fontsize=fontsize, family=family, weight=weight)
             for s in lines]
    width = max(w for w, _ in sizes)
    line_h = max(h for _, h in sizes)
    height = line_h * line_spacing * len(lines)
    return (width, height)


def max_label_width_in(
    labels: list[str],
    *,
    fontsize: float = AXIS_TEXT_SIZE_PT,
    family: str | None = None,
    rotation: float = 0.0,
) -> float:
    """Widest tick/strip/legend label in the list (inches)."""
    if not labels:
        return 0.0
    return max(
        text_size_in(s, fontsize=fontsize, family=family, rotation=rotation)[0]
        for s in labels
    )


def max_label_height_in(
    labels: list[str],
    *,
    fontsize: float = AXIS_TEXT_SIZE_PT,
    family: str | None = None,
    rotation: float = 0.0,
) -> float:
    """Tallest tick/strip/legend label in the list (inches)."""
    if not labels:
        return 0.0
    return max(
        text_size_in(s, fontsize=fontsize, family=family, rotation=rotation)[1]
        for s in labels
    )


# ---- Compound decorations -------------------------------------------------
# These are sized empirically to match R/patchwork's spacing. Tweak if a
# diff catalog item points at a specific decoration.

# Vertical colorbar: bar itself + small pad + tick labels.
COLORBAR_BAR_WIDTH_IN = 0.20
COLORBAR_BAR_PAD_IN = 0.10
# Whitespace reserved between the panel's right edge and the colorbar.
# Prevents the bar from kissing the panel — matches R/ggplot2's
# generous gap (more than the typical legend ``COL_GAP_IN``).
COLORBAR_PANEL_PAD_IN = 0.25

# Legend (right of plot): key glyph + key->text pad + text + outer pad.
LEGEND_KEY_WIDTH_IN = 0.22
LEGEND_KEY_HEIGHT_IN = 0.18
LEGEND_KEY_PAD_IN = 0.08
LEGEND_BOX_PAD_IN = 0.10
LEGEND_LINE_SPACING = 1.15

# Strip (facet label) padding above/below text.
STRIP_PAD_IN = 0.06

# Cell padding between rows (e.g. between title and panel).
ROW_GAP_IN = 0.04
COL_GAP_IN = 0.04


def colorbar_cell_width_in(
    tick_labels: list[str],
    *,
    fontsize: float = AXIS_TEXT_SIZE_PT,
) -> float:
    """Horizontal extent of a vertical colorbar cell (panel pad + bar +
    bar-to-text pad + tick text)."""
    if not tick_labels:
        return 0.0
    text_w = max_label_width_in(tick_labels, fontsize=fontsize)
    return (
        COLORBAR_PANEL_PAD_IN
        + COLORBAR_BAR_WIDTH_IN
        + COLORBAR_BAR_PAD_IN
        + text_w
    )


def legend_cell_size_in(
    title: str | None,
    entries: list[str],
    *,
    title_fontsize: float = LEGEND_TITLE_SIZE_PT,
    text_fontsize: float = LEGEND_TEXT_SIZE_PT,
) -> tuple[float, float]:
    """Approximate (w, h) of a vertical right-side legend in inches."""
    if not entries:
        return (0.0, 0.0)
    text_w = max_label_width_in(entries, fontsize=text_fontsize)
    line_h = max(
        max_label_height_in(entries, fontsize=text_fontsize),
        LEGEND_KEY_HEIGHT_IN,
    )
    width = (
        LEGEND_KEY_WIDTH_IN
        + LEGEND_KEY_PAD_IN
        + text_w
        + 2 * LEGEND_BOX_PAD_IN
    )
    height = line_h * LEGEND_LINE_SPACING * len(entries) + 2 * LEGEND_BOX_PAD_IN
    if title:
        title_w, title_h = text_size_in(title, fontsize=title_fontsize, weight="bold")
        width = max(width, title_w + 2 * LEGEND_BOX_PAD_IN)
        height += title_h + LEGEND_KEY_PAD_IN
    return (width, height)


def strip_cell_height_in(
    label: str,
    *,
    fontsize: float = STRIP_TEXT_SIZE_PT,
) -> float:
    """Height of a facet strip (top, horizontal) including padding."""
    if not label:
        return 0.0
    _, h = text_size_in(label, fontsize=fontsize)
    return h + 2 * STRIP_PAD_IN


def strip_cell_width_in(
    label: str,
    *,
    fontsize: float = STRIP_TEXT_SIZE_PT,
) -> float:
    """Width of a facet strip rotated 90° (right strip in facet_grid)."""
    if not label:
        return 0.0
    _, h = text_size_in(label, fontsize=fontsize, rotation=90.0)
    # When rotated 90°, the rendered "height" of the artist is the
    # column width we need.
    return h + 2 * STRIP_PAD_IN
