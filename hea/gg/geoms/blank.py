"""``geom_blank()`` — draws nothing. Used to extend scales without painting."""

from __future__ import annotations

from .geom import Geom


class GeomBlank(Geom):
    def draw_panel(self, data, ax) -> None:
        return  # intentional: blank layer extends scale ranges only


def geom_blank(mapping=None, data=None):
    from ..layer import Layer
    from ..positions.identity import PositionIdentity
    from ..stats.identity import StatIdentity
    return Layer(
        geom=GeomBlank(),
        stat=StatIdentity(),
        position=PositionIdentity(),
        mapping=mapping,
        data=data,
    )
