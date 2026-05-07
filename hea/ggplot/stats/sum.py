"""``stat_sum()`` — counts rows per ``(x, y)`` (and per discrete aes when
they're mapped). Default geom: ``geom_point``; the resulting points get a
``size`` aesthetic = the count, so frequent ``(x, y)`` pairs visually pop.

``geom_count()`` is the alias.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .stat import Stat


_GROUP_KEYS = ("x", "y", "group", "colour", "fill", "shape", "linetype")


@dataclass
class StatSum(Stat):
    def compute_panel(self, data, params):
        if "x" not in data.columns or "y" not in data.columns or len(data) == 0:
            return pl.DataFrame()
        keys = [c for c in _GROUP_KEYS if c in data.columns]
        out = (
            data.group_by(keys, maintain_order=True)
            .agg(pl.len().alias("n"))
            .sort(keys)
        )
        # ggplot2 also computes ``prop = n / sum(n)`` for after_stat use.
        total = out["n"].sum()
        out = out.with_columns(
            prop=(pl.col("n") / total) if total else pl.col("n") * 0.0,
            size=pl.col("n").cast(pl.Float64),
        )
        return out


def stat_sum(mapping=None, data=None, *, geom="point",
             position="identity", **kwargs):
    """Count rows per ``(x, y)``. Output: ``n`` (count), ``prop``
    (fraction of total), ``size = n`` for visual sizing."""
    from ..geoms.point import GeomPoint
    from ..layer import Layer
    from ..positions import resolve_position

    if isinstance(geom, str):
        if geom != "point":
            raise ValueError(f"stat_sum: unknown geom {geom!r}; expected 'point'")
        geom_obj = GeomPoint()
    else:
        geom_obj = geom

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "shape", "alpha"}}

    return Layer(
        geom=geom_obj,
        stat=StatSum(),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def geom_count(mapping=None, data=None, **kwargs):
    """Scatter where point size scales with row multiplicity at each
    ``(x, y)``. Alias for :func:`stat_sum` with the default point geom."""
    return stat_sum(mapping=mapping, data=data, **kwargs)
