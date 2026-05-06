"""hea.ggplot — Grammar of Graphics in Python (a port of R's ggplot2).

This is the *package*; the :func:`ggplot` factory function lives inside
it. Use ``from hea.ggplot import ggplot, aes, geom_point`` rather than
``from hea import ggplot`` (the latter binds the package, not the function).

Phase 0+ surface: ``ggplot``, ``aes``, ``geom_point``, ``geom_blank``,
``geom_bar``/``geom_col``/``geom_histogram``, ``geom_density``, plus the
underlying ``stat_*`` family. Subsequent phases land continuous scales,
polar coords (the strategic piece — see
``.claude/plans/ggplot2-port.md``), facets, themes, guides, and
circular-statistics extensions in :mod:`hea.ggplot.circular`.
"""

from __future__ import annotations

from .aes import aes
from .core import ggplot
from .geoms import geom_bar, geom_blank, geom_col, geom_density, geom_histogram, geom_point
from .scales import (
    scale_x_continuous,
    scale_x_log10,
    scale_x_log2,
    scale_x_reverse,
    scale_x_sqrt,
    scale_y_continuous,
    scale_y_log10,
    scale_y_log2,
    scale_y_reverse,
    scale_y_sqrt,
)
from .stats import stat_bin, stat_count, stat_density, stat_identity

__all__ = [
    "ggplot",
    "aes",
    "geom_blank",
    "geom_point",
    "geom_bar",
    "geom_col",
    "geom_histogram",
    "geom_density",
    "stat_identity",
    "stat_bin",
    "stat_count",
    "stat_density",
    "scale_x_continuous",
    "scale_y_continuous",
    "scale_x_log10",
    "scale_y_log10",
    "scale_x_log2",
    "scale_y_log2",
    "scale_x_sqrt",
    "scale_y_sqrt",
    "scale_x_reverse",
    "scale_y_reverse",
]
