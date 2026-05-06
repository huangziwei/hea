"""``stat_identity()`` — pass-through. Used by geoms whose data is already drawable."""

from __future__ import annotations

from .stat import Stat


class StatIdentity(Stat):
    pass


def stat_identity():
    return StatIdentity()
