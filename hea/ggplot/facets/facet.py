"""Base :class:`Facet` — splits a plot into multiple panels.

Concrete subclasses override:

* :meth:`compute_layout` — return a DataFrame with one row per panel
  (``PANEL`` index, ``ROW``/``COL``, plus any facet variable values).
* :meth:`map_data` — add a ``PANEL`` column to a per-layer data frame so
  downstream code knows which panel each row belongs to.
* :meth:`grid_dims` — return ``(nrow, ncol)`` for the matplotlib subplot grid.
* :meth:`facet_vars` — list the facet variables (used by build to inject
  them into per-layer frames before stat).

:class:`FacetNull` (the default) collapses everything to a single panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class Facet:
    scales: str = "fixed"  # "fixed" | "free" | "free_x" | "free_y"

    def compute_layout(self, data: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame({"PANEL": [1], "ROW": [1], "COL": [1]})

    def map_data(self, data: pl.DataFrame, layout: pl.DataFrame) -> pl.DataFrame:
        if "PANEL" in data.columns:
            return data
        return data.with_columns(PANEL=pl.lit(1, dtype=pl.Int64))

    def grid_dims(self, n_panels: int) -> tuple[int, int]:
        return (1, 1)

    def facet_vars(self) -> list[str]:
        return []
