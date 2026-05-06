"""``ScalesList`` — registry of scales by aesthetic.

A single :class:`Scale` may cover multiple aesthetics (e.g. one
``scale_color_*`` covers both ``"colour"`` and ``"fill"`` when the same
mapping drives both). The dict is keyed by aesthetic for fast lookup;
identity is preserved so a shared scale isn't double-counted.
"""

from __future__ import annotations

import copy as _copy

from .scale import Scale


class ScalesList:
    def __init__(self):
        self._by_aes: dict[str, Scale] = {}

    def add(self, scale: Scale) -> None:
        for aes in scale.aesthetics:
            self._by_aes[aes] = scale

    def get(self, aesthetic: str) -> Scale | None:
        return self._by_aes.get(aesthetic)

    def has(self, aesthetic: str) -> bool:
        return aesthetic in self._by_aes

    def get_or_default(self, aesthetic: str) -> Scale | None:
        """Return the registered scale for ``aesthetic``, auto-creating a
        :class:`ScaleContinuous` for ``"x"`` / ``"y"`` if missing."""
        sc = self._by_aes.get(aesthetic)
        if sc is not None:
            return sc
        if aesthetic in ("x", "y"):
            from .continuous import ScaleContinuous

            sc = ScaleContinuous(aesthetics=(aesthetic,))
            self._by_aes[aesthetic] = sc
            return sc
        return None

    def copy(self) -> "ScalesList":
        """Independent copy — each ``draw()`` builds fresh scales so repeated
        builds don't accumulate state."""
        new = ScalesList()
        # Preserve sharing: same Scale instance bound to multiple aesthetics
        # ends up shared in the copy too.
        seen: dict[int, Scale] = {}
        for aes, sc in self._by_aes.items():
            if id(sc) in seen:
                new._by_aes[aes] = seen[id(sc)]
            else:
                clone = _copy.copy(sc)
                seen[id(sc)] = clone
                new._by_aes[aes] = clone
        return new

    def items(self):
        return self._by_aes.items()
