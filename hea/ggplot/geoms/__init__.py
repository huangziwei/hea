"""Geoms — visual primitives that draw layer data onto matplotlib axes."""

from .bar import geom_bar, geom_col
from .bin2d import geom_bin2d, geom_hex
from .blank import geom_blank
from .boxplot import geom_boxplot
from .contour import geom_contour, geom_contour_filled
from .density import geom_density
from .density_ridges import geom_density_ridges, geom_density_ridges2
from .dotplot import geom_dotplot
from .errorbar import (
    geom_crossbar,
    geom_errorbar,
    geom_errorbarh,
    geom_linerange,
    geom_pointrange,
)
from .histogram import geom_freqpoly, geom_histogram
from .path import geom_line, geom_path, geom_step
from .point import geom_jitter, geom_point
from .polygon import geom_polygon
from .rect import geom_raster, geom_rect, geom_tile
from .refline import geom_abline, geom_hline, geom_vline
from .ribbon import geom_area, geom_ribbon
from .segment import geom_curve, geom_segment
from .smooth import geom_smooth
from .text import geom_label, geom_label_repel, geom_text, geom_text_repel
from .violin import geom_violin

__all__ = [
    "geom_abline",
    "geom_area",
    "geom_bar",
    "geom_bin2d",
    "geom_blank",
    "geom_boxplot",
    "geom_col",
    "geom_contour",
    "geom_contour_filled",
    "geom_crossbar",
    "geom_curve",
    "geom_density",
    "geom_density_ridges",
    "geom_density_ridges2",
    "geom_dotplot",
    "geom_errorbar",
    "geom_errorbarh",
    "geom_freqpoly",
    "geom_hex",
    "geom_histogram",
    "geom_hline",
    "geom_jitter",
    "geom_label",
    "geom_label_repel",
    "geom_line",
    "geom_linerange",
    "geom_path",
    "geom_point",
    "geom_pointrange",
    "geom_polygon",
    "geom_raster",
    "geom_rect",
    "geom_ribbon",
    "geom_segment",
    "geom_smooth",
    "geom_step",
    "geom_text",
    "geom_text_repel",
    "geom_tile",
    "geom_violin",
    "geom_vline",
]
