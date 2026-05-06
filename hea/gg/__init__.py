"""hea.gg — Grammar of Graphics in Python (a port of R's ggplot2).

Phase 0 surface: ``ggplot``, ``aes``, ``geom_point``, ``geom_blank``.
Subsequent phases land scales, more geoms, polar coords (the strategic
piece — see ``.claude/plans/ggplot2-port.md``), facets, themes, guides,
and circular-statistics extensions in :mod:`hea.gg.circular`.
"""

from __future__ import annotations

from .aes import aes
from .core import ggplot
from .geoms import geom_bar, geom_blank, geom_col, geom_density, geom_histogram, geom_point
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
]
