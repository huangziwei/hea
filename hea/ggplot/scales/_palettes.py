"""Built-in palette functions for discrete scales.

A palette is a callable ``palette(n) -> list[str]`` returning ``n`` colour
strings (hex codes preferred). ggplot2's default qualitative palette is
``scales::hue_pal``, equally-spaced hues in HCL space at lightness 65
and chroma 100.

We approximate via HSV (in :mod:`colorsys`); the colours are
qualitatively similar but not byte-identical to ggplot2. A faithful HCL
port (matching `colorspace` package output) is a polish item — file as
follow-up if oracle parity matters.
"""

from __future__ import annotations

import colorsys


def hue_pal(*, h_start: float = 15, c: float = 100, l: float = 65):
    """Default qualitative palette — equally-spaced hues."""

    def palette(n: int) -> list[str]:
        if n <= 0:
            return []
        # Match ggplot2's behaviour: span 360° among n levels, starting at h_start.
        hues = [(h_start + i * 360.0 / n) % 360 for i in range(n)]
        # HSV approximation (real ggplot2 uses HCLuv).
        s = min(c / 100.0, 1.0)
        v = l / 100.0 + 0.20  # bump value a bit so colours aren't muddy
        v = min(v, 1.0)
        return [_hsv_to_hex(h / 360.0, s, v) for h in hues]

    return palette


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def manual_pal(values):
    """Palette from an explicit list (or dict mapping levels → colours)."""

    if isinstance(values, dict):
        # dict palette — used when user passes scale_color_manual(values={...}).
        # The dict-based map happens in ScaleDiscreteColor.map directly; the
        # callable form here just returns the values in order.
        ordered = list(values.values())
    else:
        ordered = list(values)

    def palette(n: int) -> list[str]:
        if n > len(ordered):
            raise ValueError(
                f"manual palette has {len(ordered)} colours but {n} levels are needed"
            )
        return ordered[:n]

    return palette
