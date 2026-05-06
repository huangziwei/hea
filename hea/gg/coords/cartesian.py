"""``coord_cartesian()`` — the default identity coord."""

from __future__ import annotations

from .coord import Coord


class CoordCartesian(Coord):
    pass


def coord_cartesian():
    return CoordCartesian()
