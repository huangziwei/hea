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

    def train(self, data) -> None:
        """Update internal state from a column of values. Default: no-op.

        Subclasses override to track ranges (continuous), levels (discrete),
        or whatever else they need at map-time."""

    def map(self, data):
        """Translate raw data values into drawable values. Default: identity.

        Override in non-positional scales (colour, fill, shape, size) where
        the value the geom draws differs from the value in the data."""
        return data

    def apply_to_axis(self, ax, axis: str) -> None:
        """Push limits/breaks/labels onto ``ax``. Default: no-op.

        Only positional scales (``ScaleContinuous`` for now) override this."""


def fmt_number(x: float) -> str:
    """Format a numeric break for display.

    Phase 1.1 form: integer if exact, otherwise ``%g``. Full
    ``scales::label_number`` parity (thousands separators, accuracy=,
    big.mark, etc.) lands in Phase 3 alongside guides.
    """
    if x == int(x):
        return str(int(x))
    return f"{x:g}"
