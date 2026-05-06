"""Base :class:`Stat` — transforms layer data before geoms see it.

Default flow: ``compute_layer`` calls ``compute_panel`` per panel, which
splits on ``group`` and dispatches each chunk to ``compute_group``.
Subclasses typically override only ``compute_group``.

The base ``compute_panel`` re-attaches the per-group identifier columns
(``group``, ``colour``, ``fill``, ``linetype``) to each chunk's output.
Without this, stats that emit fewer rows than they consume (smooth,
density, …) would lose the grouping aesthetic and downstream geoms would
draw every group in the default colour.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


_GROUPING_AES = ("group", "colour", "fill", "linetype")


@dataclass
class Stat:
    def compute_layer(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        return self.compute_panel(data, params)

    def compute_panel(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        if "group" not in data.columns or len(data) == 0:
            return self.compute_group(data, params)

        # Columns to attach back to each chunk so per-group aesthetics
        # survive the row-count change the stat may introduce.
        preserve = [col for col in _GROUPING_AES if col in data.columns]

        chunks = []
        for _, sub in data.group_by("group", maintain_order=True):
            chunk = self.compute_group(sub, params)
            if chunk is None or len(chunk) == 0:
                continue
            for col in preserve:
                if col not in chunk.columns:
                    chunk = chunk.with_columns(pl.lit(sub[col][0]).alias(col))
            chunks.append(chunk)
        if not chunks:
            return pl.DataFrame()
        return pl.concat(chunks, how="diagonal_relaxed")

    def compute_group(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        return data
