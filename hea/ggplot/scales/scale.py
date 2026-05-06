"""Base :class:`Scale` — abstract enough for both continuous and discrete scales.

A scale is the contract between *data values* and *axis appearance*. For
positional scales (x, y) on a Cartesian coord, the contract is mostly about
ticks: matplotlib autoscales the limits from artist extents, and the scale
contributes break positions + tick labels. Phase 1.1 ships only
:class:`ScaleContinuous`; discrete and non-positional (colour, fill, size,
shape) scales arrive in 1.5 / 1.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Scale:
    aesthetics: tuple[str, ...] = ()
    name: str | None = None
    # ``breaks``: "default" | None (no ticks) | array-like | callable
    breaks: Any = "default"
    # ``labels``: "default" | array-like | callable(breaks) -> list[str]
    labels: Any = "default"
    limits: tuple | None = None
    expand: tuple = (0.05, 0.0)

    def apply_to_axis(self, ax, axis: str) -> None:
        """Push limits/breaks/labels onto ``ax``. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__}.apply_to_axis not implemented"
        )


def fmt_number(x: float) -> str:
    """Format a numeric break for display.

    Phase 1.1 form: integer if exact, otherwise ``%g``. Full
    ``scales::label_number`` parity (thousands separators, accuracy=,
    big.mark, etc.) lands in Phase 5 alongside guides.
    """
    if x == int(x):
        return str(int(x))
    return f"{x:g}"
