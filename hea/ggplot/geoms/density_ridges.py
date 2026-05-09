"""``geom_density_ridges()`` / ``geom_density_ridges2()`` — joy-plot ridgelines.

Port of the ggridges (R) package's ``geom_density_ridges`` family
(R/geoms.R). Each discrete ``y`` level becomes a horizontal baseline;
the per-group KDE supplied by :class:`StatDensityRidges` is plotted as
a filled, vertically-offset ridgeline.

* ``geom_density_ridges`` — open ridgeline: filled area underneath plus
  a line along the top only (no edge along the baseline).
* ``geom_density_ridges2`` — closed polygon: full outline including the
  baseline.

Vertical offset (R/geoms.R::GeomDensityRidges$setup_data):

    iscale = yrange / ((n_y - 1) · hmax)
    ymin   = y
    ymax   = y + iscale · scale · height

with ``yrange = max(y) - min(y)``, ``n_y = unique(y)``, ``hmax =
max(height)``. The default ``scale = 1.8`` lets adjacent ridges overlap
by ~80%; ``scale = 1.0`` makes the tallest ridge just kiss the next
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..aes import split_layer_kwargs
from ..positions.position import to_numeric_positions
from .geom import Geom


@dataclass
class GeomDensityRidges(Geom):
    # Mirrors ggridges' ``GeomDensityRidges$default_aes``
    # (R/geoms.R): colour=black, fill=grey70, scale=1.8,
    # rel_min_height=0, alpha=NA.
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": "grey70",
        "size": 0.5,
        "linewidth": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y", "height")
    key_glyph: str = "polygon"

    scale: float = 1.8
    rel_min_height: float = 0.0
    panel_scaling: bool = True
    # geom_density_ridges (False) draws the top line only; geom_density_ridges2
    # (True) closes the polygon and outlines the whole boundary.
    closed: bool = False

    def setup_data(self, data: pl.DataFrame) -> pl.DataFrame:
        if (
            len(data) == 0
            or "y" not in data.columns
            or "height" not in data.columns
        ):
            return data

        # Discrete y → 0-based integer positions so iscale arithmetic
        # works and ridges land on the ordinal axis ticks the y scale
        # produces. Numeric y passes through.
        y = data["y"]
        if y.dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean):
            y_num = to_numeric_positions(y)
            data = data.with_columns(y_num.alias("y"))
            y = data["y"]
        y_num = y.cast(pl.Float64, strict=False)

        unique_y = y_num.drop_nulls().unique()
        n_y = len(unique_y)
        if n_y >= 2:
            yrange = float(unique_y.max() - unique_y.min())
        else:
            yrange = 1.0

        # ``panel_scaling = True`` (ggridges default) computes ``hmax``
        # per-PANEL. setup_data runs before facet.map_data assigns
        # PANEL, so for now we use a global ``hmax`` (correct for
        # non-faceted plots; for faceted plots this matches
        # ``panel_scaling = False``).
        h = data["height"].cast(pl.Float64, strict=False)
        h_finite = h.drop_nulls().drop_nans()
        hmax = float(h_finite.max()) if len(h_finite) else 1.0
        if hmax <= 0:
            hmax = 1.0

        iscale = yrange / ((n_y - 1) * hmax) if n_y >= 2 else 1.0
        scale_param = float(self.scale)
        rel_min = float(self.rel_min_height)

        return data.with_columns(
            ymin=y_num,
            ymax=(y_num + iscale * scale_param * h.fill_null(0.0)).cast(pl.Float64),
        ).with_columns(
            _ridge_min_height=pl.lit(hmax * rel_min, dtype=pl.Float64),
        )

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return

        # One ridge per (group, y) — ``y`` is the baseline so two
        # ridgelines at different y values are different polygons even
        # when they share the auto-assigned ``group = -1`` (hea's
        # ``_add_group`` only splits on colour/fill/shape/linetype, not
        # discrete y). ggridges draws highest-y first so lower ridges
        # overlap on top (R/geoms.R::GeomRidgeline$draw_panel orders by
        # ymin descending).
        groupby_cols = ["y"]
        if "group" in data.columns and data["group"].n_unique() > 1:
            groupby_cols = ["group", "y"]
        groups = list(data.group_by(groupby_cols, maintain_order=True))
        groups.sort(key=lambda kv: -float(kv[1]["y"][0]))
        for _, sub in groups:
            self._draw_one(sub, ax, r_color)

    def _draw_one(self, sub, ax, r_color):
        x = sub["x"].to_numpy().astype(float)
        ymin = sub["ymin"].to_numpy().astype(float)
        ymax = sub["ymax"].to_numpy().astype(float)
        h = sub["height"].to_numpy().astype(float)
        order = np.argsort(x)
        x, ymin, ymax, h = x[order], ymin[order], ymax[order], h[order]

        min_h = float(_first(sub, "_ridge_min_height", 0.0))
        if min_h > 0:
            mask = h < min_h
            ymax = ymax.copy()
            ymax[mask] = np.nan

        fill = r_color(_first(sub, "fill", "grey70"))
        edge_raw = _first(sub, "colour", "black")
        edge = r_color(edge_raw) if edge_raw is not None else "none"
        alpha = float(_first(sub, "alpha", 1.0))
        lw_raw = _first(sub, "linewidth", None)
        if lw_raw is None:
            lw_raw = _first(sub, "size", 0.5)
        lw = float(lw_raw)

        if self.closed:
            valid = ~np.isnan(ymax)
            if not valid.any():
                return
            poly_x = np.concatenate([x[valid], x[valid][::-1]])
            poly_y = np.concatenate([ymax[valid], ymin[valid][::-1]])
            ax.fill(
                poly_x, poly_y,
                facecolor=fill, edgecolor=edge,
                linewidth=lw * 2.83, alpha=alpha,
            )
        else:
            ax.fill_between(
                x, ymin, ymax, where=~np.isnan(ymax),
                facecolor=fill, edgecolor="none", alpha=alpha,
                linewidth=0,
            )
            # Top line only — drop NaN segments so rel_min_height cuts
            # don't connect across the gap.
            line_y = np.where(np.isnan(ymax), np.nan, ymax)
            ax.plot(x, line_y, color=edge, linewidth=lw * 2.83, alpha=alpha)


@dataclass
class GeomDensityRidges2(GeomDensityRidges):
    closed: bool = True


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    if val is None:
        return default
    return val


def _resolve_stat(stat, *, bandwidth, n):
    from ..stats import resolve_stat
    from ..stats.density_ridges import StatDensityRidges

    if stat == "density_ridges":
        return StatDensityRidges(bandwidth=bandwidth, n=n)
    if isinstance(stat, str):
        return resolve_stat(stat)
    return stat


def geom_density_ridges(mapping=None, data=None, *, stat="density_ridges",
                        position="identity", scale=1.8, rel_min_height=0.0,
                        panel_scaling=True, bandwidth=None, n=512, **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position

    aes_params, geom_params = split_layer_kwargs(kwargs)
    return Layer(
        geom=GeomDensityRidges(scale=scale, rel_min_height=rel_min_height,
                               panel_scaling=panel_scaling),
        stat=_resolve_stat(stat, bandwidth=bandwidth, n=n),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )


def geom_density_ridges2(mapping=None, data=None, *, stat="density_ridges",
                         position="identity", scale=1.8, rel_min_height=0.0,
                         panel_scaling=True, bandwidth=None, n=512, **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position

    aes_params, geom_params = split_layer_kwargs(kwargs)
    return Layer(
        geom=GeomDensityRidges2(scale=scale, rel_min_height=rel_min_height,
                                panel_scaling=panel_scaling),
        stat=_resolve_stat(stat, bandwidth=bandwidth, n=n),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )
