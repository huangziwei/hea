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


# ---------------------------------------------------------------------------
# Continuous palettes — return a function that takes values in [0, 1] and
# yields a list of hex codes.
# ---------------------------------------------------------------------------

def gradient_pal(low: str = "#132B43", high: str = "#56B1F7"):
    """Two-colour gradient. ggplot2 defaults: dark blue → light blue."""
    import numpy as np
    from matplotlib.colors import to_hex, to_rgb

    lo = np.asarray(to_rgb(low))
    hi = np.asarray(to_rgb(high))

    def palette(values) -> list[str]:
        v = np.asarray(values, dtype=float).reshape(-1, 1)
        rgb = lo + v * (hi - lo)
        return [to_hex(c) for c in rgb]

    return palette


def gradient2_pal(low: str = "#3B4CC0", mid: str = "#DDDDDD",
                  high: str = "#B40426", midpoint: float = 0.5):
    """Diverging three-colour gradient with a midpoint anchor (in [0, 1])."""
    import numpy as np
    from matplotlib.colors import to_hex, to_rgb

    lo = np.asarray(to_rgb(low))
    mi = np.asarray(to_rgb(mid))
    hi = np.asarray(to_rgb(high))

    def palette(values) -> list[str]:
        v = np.asarray(values, dtype=float)
        out = np.empty((len(v), 3))
        below = v <= midpoint
        # piecewise lerp: [0, midpoint] mixes lo↔mid, [midpoint, 1] mixes mid↔hi
        denom_lo = midpoint if midpoint > 0 else 1.0
        denom_hi = (1 - midpoint) if midpoint < 1 else 1.0
        t_lo = np.clip(v / denom_lo, 0, 1)
        t_hi = np.clip((v - midpoint) / denom_hi, 0, 1)
        out[below] = lo + t_lo[below, None] * (mi - lo)
        out[~below] = mi + t_hi[~below, None] * (hi - mi)
        return [to_hex(c) for c in out]

    return palette


def gradientn_pal(colors, *, n_interp: int = 256):
    """N-stop gradient from an explicit colour list, equally spaced."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    cmap = LinearSegmentedColormap.from_list("hea_gradientn", list(colors),
                                              N=n_interp)

    def palette(values) -> list[str]:
        return [to_hex(c) for c in cmap(values)]

    return palette


def viridis_pal(*, option: str = "viridis", direction: int = 1):
    """Continuous viridis-family palette. Reads matplotlib's colormap of the
    same name. ``direction=-1`` reverses."""
    import matplotlib
    import numpy as np
    from matplotlib.colors import to_hex

    cmap = matplotlib.colormaps[option]

    def palette(values) -> list[str]:
        v = np.asarray(values, dtype=float)
        if direction == -1:
            v = 1.0 - v
        return [to_hex(c) for c in cmap(v)]

    return palette


def viridis_pal_discrete(*, option: str = "viridis", direction: int = 1,
                         begin: float = 0.0, end: float = 1.0):
    """Discrete viridis palette: equally-spaced samples for n levels."""
    import matplotlib
    import numpy as np
    from matplotlib.colors import to_hex

    cmap = matplotlib.colormaps[option]

    def palette(n: int) -> list[str]:
        if n <= 0:
            return []
        if n == 1:
            return [to_hex(cmap((begin + end) / 2))]
        positions = np.linspace(begin, end, n)
        if direction == -1:
            positions = positions[::-1]
        return [to_hex(c) for c in cmap(positions)]

    return palette


def brewer_pal_discrete(*, palette: str = "Set1", direction: int = 1):
    """Discrete ColorBrewer palette via matplotlib's bundled colormaps.

    matplotlib ships the same hex codes as ``RColorBrewer::brewer.pal``
    for the qualitative palettes (Set1, Set2, Pastel1, Dark2, Accent,
    Paired). Sequential / diverging palettes are sampled evenly across
    the corresponding ``LinearSegmentedColormap``.
    """
    import matplotlib
    import numpy as np
    from matplotlib.colors import ListedColormap, to_hex

    cmap = matplotlib.colormaps[palette]

    if isinstance(cmap, ListedColormap):
        base = list(cmap.colors)
        if direction == -1:
            base = base[::-1]

        def pal(n: int) -> list[str]:
            if n > len(base):
                raise ValueError(
                    f"ColorBrewer palette {palette!r} has {len(base)} colours "
                    f"but {n} levels are needed"
                )
            return [to_hex(c) for c in base[:n]]

        return pal

    def pal(n: int) -> list[str]:
        if n <= 0:
            return []
        if n == 1:
            return [to_hex(cmap(0.5))]
        positions = np.linspace(0, 1, n)
        if direction == -1:
            positions = positions[::-1]
        return [to_hex(c) for c in cmap(positions)]

    return pal


def brewer_pal_continuous(*, palette: str = "Blues", direction: int = 1):
    """Continuous ColorBrewer-derived palette — for ``scale_*_distiller``
    style smooth-color use."""
    import matplotlib
    import numpy as np
    from matplotlib.colors import to_hex

    cmap = matplotlib.colormaps[palette]

    def pal(values) -> list[str]:
        v = np.asarray(values, dtype=float)
        if direction == -1:
            v = 1.0 - v
        return [to_hex(c) for c in cmap(v)]

    return pal
