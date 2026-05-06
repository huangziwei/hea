"""Base :class:`Geom` — visual primitive that paints a layer's drawable data.

Geoms see a polars DataFrame whose columns are *canonical aesthetic names*
(``x``, ``y``, ``colour``, ``size``, …) after the build pipeline has run
mappings, stat, position, and coord transform. A geom only needs to know
how to take that frame and render it into a matplotlib ``Axes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class Geom:
    """Base geom. Subclasses fill in ``default_aes``, ``required_aes``,
    and override :meth:`draw_panel`."""

    default_aes: dict = field(default_factory=dict)
    required_aes: tuple = ()

    def draw_panel(self, data: pl.DataFrame, ax) -> None:
        """Render ``data`` onto ``ax``. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.draw_panel not implemented")
