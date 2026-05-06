"""``position_identity()`` — pass-through."""

from __future__ import annotations

from .position import Position


class PositionIdentity(Position):
    pass


def position_identity():
    return PositionIdentity()
