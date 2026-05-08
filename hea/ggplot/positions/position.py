"""Base :class:`Position` — adjusts geom positions (jitter, dodge, stack, …)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class Position:
    def compute_layer(self, data: pl.DataFrame) -> pl.DataFrame:
        return data


def to_numeric_positions(series: pl.Series) -> pl.Series:
    """Map a discrete series to 0-based integer positions.

    ggplot2 maps discrete axis values to integers before any position
    adjustment that does arithmetic on x/y (dodge, nudge, jitter); we do
    the same here so ``geom_bar(aes(x=drv), position="dodge")`` doesn't
    blow up trying to add a float offset to a string column.

    Level order matches :class:`ScaleOrdinal`: ``pl.Enum`` /
    ``pl.Categorical`` use their declared categories; plain strings sort
    alphabetically (matches R's ``factor()`` default of
    ``levels = sort(unique(x))``).

    Numeric / floating-point input passes through unchanged.

    NOTE: when the user supplies an explicit
    ``scale_x_discrete(limits=[...])`` reordering, the scale may register
    a different level order at render time. The dodge/jitter math
    happens against the natural sort order computed here, so explicit
    ``limits=`` may shift bars under different ticks. Honouring user
    ``limits`` at the position step would require passing the scale into
    :meth:`Position.compute_layer`.
    """
    if series.dtype in (pl.Categorical, pl.Enum):
        levels = [str(v) for v in series.cat.get_categories().to_list()]
    elif series.dtype in (pl.Utf8, pl.Boolean):
        levels = sorted(str(v) for v in series.drop_nulls().unique().to_list())
    else:
        return series
    idx_map = {v: float(i) for i, v in enumerate(levels)}
    return (
        series.cast(pl.Utf8)
        .replace_strict(idx_map, return_dtype=pl.Float64)
        .alias(series.name)
    )
