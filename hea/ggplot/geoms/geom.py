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
    # Legend key glyph — controls how this geom's contribution to a legend
    # is drawn (R's ``draw_key_*`` family). Recognised values:
    #   * ``"point"``   — circle marker (default; ``geom_point`` etc.)
    #   * ``"polygon"`` — filled rectangle (``geom_bar``, ``geom_ribbon``,
    #                     ``geom_rect``/``tile``, ``geom_polygon``, ``geom_violin``)
    #   * ``"path"``    — horizontal line (``geom_line``, ``geom_path``,
    #                     ``geom_step``, ``geom_segment``)
    # Subclasses override the default to opt into the right glyph.
    key_glyph: str = "point"

    def setup_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Geom-specific data preparation, run after stat/position but
        before scale training. Default: pass-through.

        Bar/area geoms override this to expose their implicit y baseline
        (0) as ``ymin``/``ymax`` columns. Without it, the y scale trains
        only on raw y values (e.g. ``[70, 150]``) and the auto-computed
        ticks miss 0 — so a ``geom_col`` chart shows ticks at ``80, 100,
        120, 140`` instead of ``0, 50, 100, 150``.
        """
        return data

    def draw_panel(self, data: pl.DataFrame, ax) -> None:
        """Render ``data`` onto ``ax``. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.draw_panel not implemented")
