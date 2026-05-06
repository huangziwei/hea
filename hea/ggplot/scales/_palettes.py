"""Built-in palette functions for discrete scales.

A palette is a callable ``palette(n) -> list[str]`` returning ``n`` colour
strings (hex codes preferred). ggplot2's default qualitative palette is
``scales::hue_pal``, equally-spaced hues in CIE LUV (HCLuv) space at
lightness 65 and chroma 100. We port that conversion below — the
resulting colours match ``grDevices::hcl(...)`` byte-for-byte at n=3
(``#F8766D``, ``#00BA38``, ``#619CFF`` — the canonical Adelie-Gentoo-
Chinstrap colours).
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# HCL → sRGB hex (port of grDevices::hcl)
# ---------------------------------------------------------------------------

# CIE D65 white point. Values are derived from Xn=95.047, Yn=100, Zn=108.883
# via u_n = 4*Xn / (Xn + 15*Yn + 3*Zn), v_n = 9*Yn / (...). Same constants
# `colorspace`/`grDevices` use, hard-coded for speed.
_U_N = 0.1978300664283
_V_N = 0.4683199493879


def _hcl_to_rgb(h: float, c: float, l: float) -> tuple[float, float, float]:
    """HCL polar in CIE LUV → linear sRGB triple in [0, 1].

    Reference: ``grDevices/src/colors.c::hcl_to_rgb`` (R sources). The path
    is HCL → Luv → XYZ (D65) → linear RGB → sRGB-gamma.
    """
    if l <= 0:
        return (0.0, 0.0, 0.0)

    # HCL → Luv (Cartesian)
    h_rad = math.radians(h)
    u = c * math.cos(h_rad)
    v = c * math.sin(h_rad)

    # Luv → XYZ. CIE 1976 inverse lightness function.
    if l > 8:
        y = ((l + 16) / 116) ** 3
    else:
        # CIE 1976: y = L / kappa where kappa = 903.3 = 24389/27.
        y = l * 27 / 24389

    # When L is non-positive the polar formulae blow up; we already returned.
    u_prime = u / (13 * l) + _U_N
    v_prime = v / (13 * l) + _V_N

    x = (9 * y * u_prime) / (4 * v_prime)
    z = (12 - 3 * u_prime - 20 * v_prime) * y / (4 * v_prime)

    # XYZ → linear sRGB (D65 primaries; standard matrix).
    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    return (r_lin, g_lin, b_lin)


def _to_srgb(c_lin: float) -> float:
    """Linear-RGB → sRGB gamma curve, clamped to [0, 1]. Matches the IEC
    sRGB transfer function used by ``grDevices::convertColor``."""
    if c_lin <= 0:
        return 0.0
    if c_lin >= 1:
        return 1.0
    if c_lin > 0.0031308:
        return 1.055 * c_lin ** (1 / 2.4) - 0.055
    return 12.92 * c_lin


def hcl_to_hex(h: float, c: float, l: float) -> str:
    """``hcl(h, c, l)`` colour as ``#RRGGBB``. Out-of-gamut colours clip
    per channel (matches ``grDevices::hcl(fixup=TRUE)``)."""
    r_lin, g_lin, b_lin = _hcl_to_rgb(h, c, l)
    r = _to_srgb(r_lin)
    g = _to_srgb(g_lin)
    b = _to_srgb(b_lin)
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


# ---------------------------------------------------------------------------
# Discrete palettes
# ---------------------------------------------------------------------------

def hue_pal(*, h: tuple = (15, 375), c: float = 100, l: float = 65,
            h_start: float = 0, direction: int = 1):
    """ggplot2's default qualitative palette — equally-spaced hues in HCL.

    Defaults match ``scales::hue_pal``: chroma 100, lightness 65, hues
    spanning ``[15, 375)`` (i.e. starting at red and going round once).
    The full-circle range is detected and the last point dropped so
    consecutive levels don't share a colour.
    """
    h_lo, h_hi = float(h[0]), float(h[1])

    def palette(n: int) -> list[str]:
        if n <= 0:
            return []
        # Drop one slot when the range is a full circle, otherwise the
        # first and last hues coincide (R's `hue_pal` does the same).
        end = h_hi
        if (h_hi - h_lo) % 360 < 1 and n > 1:
            end = h_hi - 360 / n
        if n == 1:
            hues = [h_lo]
        else:
            step = (end - h_lo) / (n - 1)
            hues = [h_lo + i * step for i in range(n)]
        # `h_start` rotates the whole palette; `direction=-1` reverses.
        rotated = [(hh + h_start) % 360 for hh in hues]
        if direction == -1:
            rotated = rotated[::-1]
        return [hcl_to_hex(hh, c, l) for hh in rotated]

    return palette


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


# ---------------------------------------------------------------------------
# Non-colour palettes — size, alpha, shape, linetype
# ---------------------------------------------------------------------------

def rescale_pal(range_: tuple = (1.0, 6.0)):
    """Linear rescale: ``[0, 1] -> [range_[0], range_[1]]``. Default matches
    ggplot2's ``scale_size_continuous`` (1 mm to 6 mm)."""
    import numpy as np

    lo, hi = float(range_[0]), float(range_[1])

    def palette(values) -> list[float]:
        v = np.asarray(values, dtype=float)
        return (lo + v * (hi - lo)).tolist()

    return palette


def area_pal(range_: tuple = (1.0, 6.0)):
    """Area-proportional rescale: ``size ∝ √value``. Used by
    ``scale_size_area`` so visual area (not radius) tracks the underlying
    quantity — which matches Tufte/Cleveland's recommendation."""
    import numpy as np

    lo, hi = float(range_[0]), float(range_[1])

    def palette(values) -> list[float]:
        v = np.sqrt(np.clip(np.asarray(values, dtype=float), 0.0, 1.0))
        return (lo + v * (hi - lo)).tolist()

    return palette


def alpha_pal(range_: tuple = (0.1, 1.0)):
    """Linear rescale into alpha-friendly range. ggplot2 default is (0.1, 1)."""
    return rescale_pal(range_)


# ggplot2's default shape sequence — `scales::shape_pal`. R uses pch codes
# 16, 17, 15, 3, 7, 8, … which we map to the closest matplotlib markers.
# ``"o"`` (filled circle), ``"^"`` (triangle up), ``"s"`` (square), etc.
_DEFAULT_SHAPES = ["o", "^", "s", "+", "x", "*", "D", "v", "<", ">"]


def shape_pal():
    """Discrete shape palette — cycles through ggplot2's default pch sequence."""

    def palette(n: int) -> list[str]:
        if n <= 0:
            return []
        if n > len(_DEFAULT_SHAPES):
            # ggplot2 errors at n>6; we cycle to keep things usable.
            import warnings

            warnings.warn(
                f"shape palette only has {len(_DEFAULT_SHAPES)} distinct shapes "
                f"({n} requested); cycling.",
                UserWarning,
                stacklevel=2,
            )
        return [_DEFAULT_SHAPES[i % len(_DEFAULT_SHAPES)] for i in range(n)]

    return palette


# ggplot2's default linetype sequence: solid, dashed, dotted, dotdash,
# longdash, twodash. Matplotlib has built-in names for the first four;
# longdash/twodash become explicit dash tuples.
_DEFAULT_LINETYPES = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    (0, (10, 3)),     # longdash
    (0, (5, 1, 1, 1)),  # twodash
]


def linetype_pal():
    def palette(n: int):
        if n <= 0:
            return []
        if n > len(_DEFAULT_LINETYPES):
            import warnings

            warnings.warn(
                f"linetype palette only has {len(_DEFAULT_LINETYPES)} distinct "
                f"styles ({n} requested); cycling.",
                UserWarning,
                stacklevel=2,
            )
        return [_DEFAULT_LINETYPES[i % len(_DEFAULT_LINETYPES)] for i in range(n)]

    return palette
