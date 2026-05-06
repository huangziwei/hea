"""Theme stub. Real :class:`Theme` + ``theme_*()`` family lands in Phase 1.8."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Theme:
    pass


def theme_default():
    return Theme()
