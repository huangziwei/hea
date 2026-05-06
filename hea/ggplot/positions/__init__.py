"""Positions — adjust per-row x/y after stat, before coord transform."""

from .dodge import PositionDodge, PositionDodge2, position_dodge, position_dodge2
from .identity import PositionIdentity, position_identity
from .jitter import PositionJitter, position_jitter
from .nudge import PositionNudge, position_nudge
from .position import Position
from .stack import PositionFill, PositionStack, position_fill, position_stack


_NAME_TO_CLASS = {
    "identity": PositionIdentity,
    "jitter": PositionJitter,
    "dodge": PositionDodge,
    "dodge2": PositionDodge2,
    "stack": PositionStack,
    "fill": PositionFill,
    "nudge": PositionNudge,
}


def resolve_position(p) -> Position:
    """Coerce ``p`` to a :class:`Position` instance.

    Accepts an instance, or a string naming one of the built-ins. ggplot2
    geoms often default ``position="stack"`` etc.; this resolves the string
    once at layer-construction time so the build pipeline never sees one.
    """
    if isinstance(p, Position):
        return p
    if isinstance(p, str):
        cls = _NAME_TO_CLASS.get(p)
        if cls is None:
            raise ValueError(
                f"unknown position {p!r}; expected one of {sorted(_NAME_TO_CLASS)}"
            )
        return cls()
    raise TypeError(
        f"position must be a Position instance or a string, got {type(p).__name__}"
    )


__all__ = [
    "Position",
    "PositionIdentity",
    "PositionJitter",
    "PositionNudge",
    "PositionDodge",
    "PositionDodge2",
    "PositionStack",
    "PositionFill",
    "position_identity",
    "position_jitter",
    "position_nudge",
    "position_dodge",
    "position_dodge2",
    "position_stack",
    "position_fill",
    "resolve_position",
]
