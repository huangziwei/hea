"""``geom_bar()``, ``geom_col()`` — bar charts.

* ``geom_bar()`` — counts per discrete x via :class:`StatCount`.
* ``geom_col()`` — uses y as supplied (``stat_identity``).

``geom_histogram`` lives in ``histogram.py`` and reuses :class:`GeomBar`.

When the build pipeline produces ``ymin``/``ymax`` columns (which positions
like :class:`PositionStack` / :class:`PositionFill` add for stacking),
:meth:`GeomBar.draw_panel` reads them as ``bottom``/``top`` of each bar.
Otherwise bars draw from y=0 up to y.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..aes import split_layer_kwargs
from .geom import Geom

_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomBar(Geom):
    default_aes: dict = field(
        default_factory=lambda: {
            "colour": None,
            "fill": "grey35",
            "size": 0.5,
            "linetype": "solid",
            "alpha": 1.0,
            "width": 0.9,
        }
    )
    required_aes: tuple = ("x", "y")
    key_glyph: str = "polygon"

    def setup_data(self, data):
        import polars as pl

        if (
            "y" in data.columns
            and "ymin" not in data.columns
            and "ymax" not in data.columns
        ):
            zero = pl.lit(0.0)
            y = pl.col("y")
            data = data.with_columns(
                ymin=pl.min_horizontal(y, zero).cast(pl.Float64),
                ymax=pl.max_horizontal(y, zero).cast(pl.Float64),
            )
        return data

    def draw_panel(self, data, ax) -> None:
        x = data["x"].to_numpy()

        if "ymin" in data.columns and "ymax" in data.columns:
            ymin = data["ymin"].to_numpy()
            ymax = data["ymax"].to_numpy()
            height = ymax - ymin
            bottom = ymin
        else:
            height = data["y"].to_numpy()
            bottom = 0

        if "width" in data.columns:
            width = data["width"].to_numpy()
        else:
            width = 0.9

        fill = _row_colour(data, "fill", when_all_none="none", when_missing="grey35")
        edge = _row_colour(data, "colour", when_all_none="none", when_missing="none")
        alpha = float(_scalar(data, "alpha", default=1.0))
        linewidth = float(_scalar(data, "size", default=0.5)) * _PT_PER_MM

        if getattr(ax, "_hea_coord_flipped", False):
            ax.barh(
                height,
                width=x,
                height=width,
                left=bottom,
                color=fill,
                edgecolor=edge,
                alpha=alpha,
                linewidth=linewidth,
                align="center",
            )
        else:
            ax.bar(
                x,
                height,
                width=width,
                bottom=bottom,
                color=fill,
                edgecolor=edge,
                alpha=alpha,
                linewidth=linewidth,
                align="center",
            )


def _scalar(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _row_colour(df, col, *, when_all_none, when_missing):
    """Resolve a per-row colour-like aesthetic into a matplotlib value."""
    from .._util import r_color

    if col not in df.columns or len(df) == 0:
        return r_color(when_missing) if when_missing != "none" else "none"
    vals = df[col].to_list()

    def _is_na(v):
        if v is None:
            return True
        return bool(isinstance(v, float) and np.isnan(v))

    if all(_is_na(v) for v in vals):
        return when_all_none if when_all_none == "none" else r_color(when_all_none)
    distinct = {v for v in vals if not _is_na(v)}
    if len(distinct) <= 1:
        return r_color(next(iter(distinct)))
    return ["none" if _is_na(v) else r_color(v) for v in vals]


def geom_bar(mapping=None, data=None, *, stat="count", position="stack", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    stat_obj = resolve_stat(stat) if isinstance(stat, str) else stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomBar(),
        stat=stat_obj,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )


def geom_col(mapping=None, data=None, *, position="stack", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.identity import StatIdentity

    aes_params, geom_params = split_layer_kwargs(kwargs)
    return Layer(
        geom=GeomBar(),
        stat=StatIdentity(),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )
