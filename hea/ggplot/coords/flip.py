"""``coord_flip()`` — swap x and y axes.

Visual effect: every geom that would render at ``(x, y)`` instead renders
at ``(y, x)``. ggplot2's classic use is the rotated bar chart:
``geom_bar(aes(x=cat)) + coord_flip()``.

Implementation: swap paired aesthetic columns (``x``↔``y``, ``xmin``↔``ymin``,
…) in each layer's render frame, then apply the scale registered for the
``x`` aesthetic to the visible y axis and vice versa, and swap default
axis labels. Stats and positions still run in the original ``aes(x, y)``
space — only the visual mapping is flipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .cartesian import CoordCartesian


# Aesthetic column pairs swapped by coord_flip.
_FLIP_PAIRS = [
    ("x", "y"),
    ("xmin", "ymin"),
    ("xmax", "ymax"),
    ("xend", "yend"),
    ("xintercept", "yintercept"),
]


@dataclass
class CoordFlip(CoordCartesian):
    """Cartesian coords with x and y swapped at render time."""


def coord_flip(*, xlim=None, ylim=None, expand=True):
    return CoordFlip(xlim=xlim, ylim=ylim, expand=expand)


def flip_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Swap the paired ``x``/``y`` aesthetic columns. Single-sided pairs
    (e.g. only ``x`` is present) get renamed in place."""
    for x_col, y_col in _FLIP_PAIRS:
        has_x = x_col in df.columns
        has_y = y_col in df.columns
        if has_x and has_y:
            tmp = f"__hea_flip_tmp_{x_col}"
            df = df.rename({x_col: tmp, y_col: x_col}).rename({tmp: y_col})
        elif has_x:
            df = df.rename({x_col: y_col})
        elif has_y:
            df = df.rename({y_col: x_col})
    return df
