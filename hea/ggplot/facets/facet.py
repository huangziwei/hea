"""Base :class:`Facet` — splits a plot into multiple panels.

Concrete subclasses override:

* :meth:`compute_layout` — return a DataFrame with one row per panel
  (``PANEL`` index, ``ROW``/``COL``, plus any facet variable values).
* :meth:`map_data` — add a ``PANEL`` column to a per-layer data frame so
  downstream code knows which panel each row belongs to.
* :meth:`grid_dims` — return ``(nrow, ncol)`` for the matplotlib subplot grid.
* :meth:`facet_vars` — list the facet variables (used by build to inject
  them into per-layer frames before stat).
* :meth:`share_axes` — return ``(sharex, sharey)`` for ``plt.subplots``
  given the facet's ``scales`` mode. Default = facet_wrap convention.
* :meth:`panel_labels` — strip-text dict for one panel. Default = single
  ``"top"`` label (facet_wrap convention).

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

    def share_axes(self) -> tuple:
        """Return ``(sharex, sharey)`` for ``plt.subplots``.

        Default follows facet_wrap: ``free_x`` → no x sharing,
        ``free_y`` → no y sharing. Subclasses (e.g. :class:`FacetGrid`)
        may override with ``'col'`` / ``'row'`` for axis-sharing within
        a column or row.
        """
        sharex = self.scales in ("fixed", "free_y")
        sharey = self.scales in ("fixed", "free_x")
        return (sharex, sharey)

    def panel_labels(self, panel_row: dict, layout: pl.DataFrame) -> dict:
        """Strip-text labels for one panel. Returns ``{"top": str}``
        (and optionally ``"right"``).

        Default — comma-joined facet values as a single top label
        (matches facet_wrap)."""
        text = ", ".join(
            f"{panel_row[v]}" for v in self.facet_vars()
            if v in panel_row
        )
        return {"top": text} if text else {}
