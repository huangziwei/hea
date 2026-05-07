"""Geoms — visual primitives that draw layer data onto matplotlib axes."""

from .bar import geom_bar, geom_col
from .blank import geom_blank
from .boxplot import geom_boxplot
from .contour import geom_contour, geom_contour_filled
from .density import geom_density
from .dotplot import geom_dotplot
from .errorbar import (
    geom_crossbar,
    geom_errorbar,
    geom_errorbarh,
    geom_linerange,
    geom_pointrange,
)
from .hex import geom_hex
from .histogram import geom_histogram
from .path import geom_line, geom_path, geom_step
from .point import geom_jitter, geom_point
from .polygon import geom_polygon
from .rect import geom_raster, geom_rect, geom_tile
from .refline import geom_abline, geom_hline, geom_vline
from .ribbon import geom_area, geom_ribbon
from .segment import geom_curve, geom_segment
from .smooth import geom_smooth
from .text import geom_label, geom_text
from .violin import geom_violin

__all__ = [
    "geom_blank",
    "geom_point",
    "geom_jitter",
    "geom_bar",
    "geom_col",
    "geom_histogram",
    "geom_density",
    "geom_line",
    "geom_path",
    "geom_step",
    "geom_ribbon",
    "geom_area",
    "geom_smooth",
    "geom_boxplot",
    "geom_violin",
    "geom_text",
    "geom_label",
    "geom_hline",
    "geom_vline",
    "geom_abline",
    "geom_rect",
    "geom_tile",
    "geom_raster",
    "geom_polygon",
    "geom_errorbar",
    "geom_errorbarh",
    "geom_linerange",
    "geom_pointrange",
    "geom_crossbar",
    "geom_segment",
    "geom_curve",
    "geom_contour",
    "geom_contour_filled",
    "geom_hex",
    "geom_dotplot",
]
