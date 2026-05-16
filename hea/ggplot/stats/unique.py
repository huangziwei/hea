"""``stat_unique()`` — drop duplicate rows. Used to overlay an unduplicated
scatter on top of a coloured one, or to ensure each ``(x, y)`` pair shows
only once when the same combination repeats in the data.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..aes import split_layer_kwargs
from .stat import Stat


@dataclass
class StatUnique(Stat):
    def compute_panel(self, data, params):
        if len(data) == 0:
            return data
        return data.unique(maintain_order=True)


def stat_unique(mapping=None, data=None, *, geom="point",
                position="identity", **kwargs):
    from ..geoms.point import GeomPoint
    from ..layer import Layer
    from ..positions import resolve_position

    if isinstance(geom, str):
        if geom != "point":
            raise ValueError(f"stat_unique: unknown geom {geom!r}; expected 'point'")
        geom_obj = GeomPoint()
    else:
        geom_obj = geom

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=geom_obj,
        stat=StatUnique(),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )
