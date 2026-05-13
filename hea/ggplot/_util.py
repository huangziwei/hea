"""Small helpers for hea.ggplot — color/shape parsing, group computation."""

from __future__ import annotations

import re

import numpy as np
import polars as pl


def polar_arc_interp(ax, *artists, steps: int = 100) -> None:
    """Enable arc-edge tessellation on polar axes.

    matplotlib's ``PolarTransform.transform_path_non_affine`` substitutes
    ``Path.arc()`` (CURVE4) segments for constant-r ``LINETO`` edges and
    radial lines for constant-θ ``LINETO`` edges — but **only** when the
    path's ``_interpolation_steps`` is not 1. ``ax.bar`` Rectangle paths
    set this to 100 by default; ``ax.plot`` Line2D and
    ``ax.fill_between`` PolyCollection paths set it to 1, so paths and
    ribbons render as chord polylines between consecutive data points.

    Call this on the artists returned by ``ax.plot`` / ``ax.fill_between``
    on a polar axes to lift them to the same arc-edged behaviour
    matplotlib gives bars. No-op when ``ax`` is not polar.

    Accepts ``Line2D`` instances (single path via ``.get_path()``) and
    ``PolyCollection`` instances (multiple paths via ``.get_paths()``).
    ``ax.plot`` returns a list of ``Line2D`` — unpack with ``*``.
    """
    if getattr(ax, "name", None) != "polar":
        return
    for a in artists:
        if hasattr(a, "get_paths"):
            for p in a.get_paths():
                p._interpolation_steps = steps
        else:
            a.get_path()._interpolation_steps = steps


def to_series(x, length: int, name: str = "value") -> pl.Series:
    """Coerce a scalar / array / Series into a polars Series of given length.

    Length-1 Series broadcast to ``length`` — covers the "constant
    computed from data" idiom, e.g. ``geom_tile(fill=pl.len())``
    where the Expr evaluates to a single number that we want every
    row to share. Mirrors :func:`numpy.asarray` 0-dim broadcasting
    one level up at the Series boundary.
    """
    if isinstance(x, pl.Series):
        if len(x) == 1 and length != 1:
            return pl.Series(name=name, values=[x[0]] * length, dtype=x.dtype)
        if len(x) != length:
            raise ValueError(f"length mismatch: aes value has {len(x)} rows, data has {length}")
        return x.alias(name)
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = np.repeat(arr, length)
    if len(arr) != length:
        raise ValueError(f"length mismatch: aes value has {len(arr)} rows, data has {length}")
    return pl.Series(name=name, values=arr)


_GREY_RX = re.compile(r"^gr(a|e)y(\d{1,3})$", re.IGNORECASE)


# ggplot2 shape vocabulary — names + R pch integer codes. Each entry is
# ``(matplotlib_marker, fill_mode)`` where ``fill_mode`` is:
#
# * ``"solid"``    — paint with ``colour`` (matplotlib's default).
# * ``"open"``     — hollow: face=none, edge=colour. pch 0/1/2/5/6 etc.
# * ``"fillable"`` — pch 21–25: face=``fill`` aes, edge=``colour`` aes.
# * ``"stroke"``   — line-only glyphs (``+``/``x``); face never paints.
#
# Names are sourced from ggplot2's ``translate_shape_string()``. pch codes
# 14 has two synonymous names ("square triangle" and "triangle square");
# both map to the same row.
_GGPLOT_SHAPE_NAMES: dict[str, tuple[str, str]] = {
    "square open":         ("s", "open"),
    "circle open":         ("o", "open"),
    "triangle open":       ("^", "open"),
    "plus":                ("+", "stroke"),
    "cross":               ("x", "stroke"),
    "diamond open":        ("D", "open"),
    "triangle down open":  ("v", "open"),
    "square cross":        ("$⊠$", "solid"),  # ⊠
    "asterisk":            ("*", "solid"),
    "diamond plus":        ("$⟐$", "solid"),  # ⟐
    "circle plus":         ("$⊕$", "solid"),  # ⊕
    "star":                ("*", "solid"),
    "square plus":         ("$⊞$", "solid"),  # ⊞
    "circle cross":        ("$⊗$", "solid"),  # ⊗
    "square triangle":     ("$⧈$", "solid"),  # ⧈
    "triangle square":     ("$⧈$", "solid"),
    "square":              ("s", "solid"),
    "circle small":        (".", "solid"),
    "triangle":            ("^", "solid"),
    "diamond":             ("D", "solid"),
    "circle":              ("o", "solid"),
    "bullet":              (".", "solid"),
    "circle filled":       ("o", "fillable"),
    "square filled":       ("s", "fillable"),
    "diamond filled":      ("D", "fillable"),
    "triangle filled":     ("^", "fillable"),
    "triangle down filled":("v", "fillable"),
}

# R pch integer code → ggplot2 shape name. Mirrors
# ggplot2:::translate_shape_string's reverse table.
_R_PCH_TO_NAME: dict[int, str] = {
    0:  "square open",          1:  "circle open",
    2:  "triangle open",        3:  "plus",
    4:  "cross",                5:  "diamond open",
    6:  "triangle down open",   7:  "square cross",
    8:  "asterisk",             9:  "diamond plus",
    10: "circle plus",          11: "star",
    12: "square plus",          13: "circle cross",
    14: "square triangle",
    15: "square",               16: "circle small",
    17: "triangle",             18: "diamond",
    19: "circle",               20: "bullet",
    21: "circle filled",        22: "square filled",
    23: "diamond filled",       24: "triangle filled",
    25: "triangle down filled",
}


def r_shape(shape):
    """Translate a ggplot2 / R shape value to ``(matplotlib_marker, fill_mode)``.

    Accepts:
      * ggplot2 shape NAME (``"circle open"``, ``"square filled"``, …)
      * R pch INTEGER (0–25)
      * matplotlib marker (passed through; e.g. ``"o"``, ``"^"``)

    Returns a ``(marker, fill_mode)`` tuple. ``fill_mode`` controls how the
    geom should colour the marker — see ``_GGPLOT_SHAPE_NAMES`` above.
    Unknown values fall through as ``(value, "solid")`` so existing
    matplotlib-flavoured callers keep working.
    """
    if shape is None:
        return ("o", "solid")
    if isinstance(shape, str):
        if shape in _GGPLOT_SHAPE_NAMES:
            return _GGPLOT_SHAPE_NAMES[shape]
        # Pass-through: matplotlib markers like 'o', 's', '^', '$x$', etc.
        return (shape, "solid")
    if isinstance(shape, (bool, np.bool_)):
        # Bool subclasses int — treat as pass-through to avoid pch=1 mapping.
        return (shape, "solid")
    if isinstance(shape, (int, np.integer)):
        name = _R_PCH_TO_NAME.get(int(shape))
        if name is not None:
            return _GGPLOT_SHAPE_NAMES[name]
    return (shape, "solid")


def r_color(c):
    """Translate an R-flavoured colour name into something matplotlib accepts.

    Handles the ``grey0``..``grey100`` / ``gray0``..``gray100`` family that R
    exposes programmatically (matplotlib only takes ``"gray"`` ≡ 50%). Other
    names pass through unchanged — matplotlib already knows ``"red"``,
    ``"black"``, hex codes, RGBA tuples, etc.
    """
    if c is None:
        return None
    if not isinstance(c, str):
        return c
    m = _GREY_RX.match(c)
    if m:
        n = int(m.group(2))
        n = max(0, min(100, n))
        return f"{n / 100:.4f}"
    return c
