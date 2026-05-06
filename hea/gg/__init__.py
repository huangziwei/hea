"""hea.gg — Grammar of Graphics in Python (a port of R's ggplot2).

Phase 0 surface: ``ggplot``, ``aes``, ``geom_point``, ``geom_blank``.
Subsequent phases land scales, more geoms, polar coords (the strategic
piece — see ``.claude/plans/ggplot2-port.md``), facets, themes, guides,
and circular-statistics extensions in :mod:`hea.gg.circular`.
"""

from __future__ import annotations

from .aes import aes
from .core import ggplot
from .geoms import geom_blank, geom_point

__all__ = [
    "ggplot",
    "aes",
    "geom_point",
    "geom_blank",
]
