"""``geom_blank()`` — draws nothing. Used to extend scales without painting."""

from __future__ import annotations

from .geom import Geom


class GeomBlank(Geom):
    def draw_panel(self, data, ax) -> None:
        return  # intentional: blank layer extends scale ranges only


def geom_blank(mapping=None, data=None, *, position="identity"):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.identity import StatIdentity
    return Layer(
        geom=GeomBlank(),
        stat=StatIdentity(),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
    )
