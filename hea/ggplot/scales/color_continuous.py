"""Continuous colour scales — viridis_c, gradient/2/n, distiller (brewer-cont).

A continuous-colour scale takes numeric data, normalises it to ``[0, 1]``
across the trained range, and runs the values through a palette function
to produce hex codes. The auto-default for a numeric ``colour``/``fill``
mapping is :func:`scale_color_gradient` (matching ggplot2's
``scale_color_continuous``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ._palettes import (
    brewer_pal_continuous,
    gradient2_pal,
    gradient_pal,
    gradientn_pal,
    viridis_pal,
)
from .scale import Scale


@dataclass
class ScaleContinuousColor(Scale):
    """Maps continuous numeric values to hex colours via a palette."""

    palette: Any = None  # callable: array of values in [0, 1] -> list[hex]
    range_: list | None = field(default=None, init=False, repr=False)

    def train(self, data) -> None:
        if isinstance(data, pl.Series):
            arr = data.drop_nulls().cast(pl.Float64).to_numpy()
        else:
            arr = np.asarray(data, dtype=float)
            arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return
        lo, hi = float(arr.min()), float(arr.max())
        if self.range_ is None:
            self.range_ = [lo, hi]
        else:
            self.range_[0] = min(self.range_[0], lo)
            self.range_[1] = max(self.range_[1], hi)

    def map(self, data):
        if self.range_ is None or self.palette is None:
            return data
        lo, hi = self.range_

        if isinstance(data, pl.Series):
            arr = data.cast(pl.Float64).to_numpy()
        else:
            arr = np.asarray(data, dtype=float)

        if hi == lo:
            normalised = np.full_like(arr, 0.5, dtype=float)
        else:
            normalised = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        # Honour user-supplied limits if set: values outside become NaN
        # in palette space (palette receives clipped values, not NaN; for
        # NaN-handling we leave the colour as null).
        colours = self.palette(normalised)

        if isinstance(data, pl.Series):
            return pl.Series(name=data.name, values=colours)
        return colours


# ---------------------------------------------------------------------------
# Factories — gradient family
# ---------------------------------------------------------------------------

def scale_color_gradient(*, low="#132B43", high="#56B1F7", name=None,
                        breaks="default", labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=gradient_pal(low=low, high=high),
    )


def scale_fill_gradient(*, low="#132B43", high="#56B1F7", name=None,
                       breaks="default", labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=gradient_pal(low=low, high=high),
    )


def scale_color_gradient2(*, low="#3B4CC0", mid="#DDDDDD", high="#B40426",
                         midpoint=0.5, name=None, breaks="default",
                         labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=gradient2_pal(low=low, mid=mid, high=high, midpoint=midpoint),
    )


def scale_fill_gradient2(*, low="#3B4CC0", mid="#DDDDDD", high="#B40426",
                        midpoint=0.5, name=None, breaks="default",
                        labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=gradient2_pal(low=low, mid=mid, high=high, midpoint=midpoint),
    )


def scale_color_gradientn(*, colours, name=None, breaks="default",
                         labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=gradientn_pal(colours),
    )


def scale_fill_gradientn(*, colours, name=None, breaks="default",
                        labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=gradientn_pal(colours),
    )


# ggplot2 has both spellings; expose them.
scale_colour_gradient = scale_color_gradient
scale_colour_gradient2 = scale_color_gradient2
scale_colour_gradientn = scale_color_gradientn

# ggplot2's `scale_color_continuous()` defaults to `gradient`. Match.
scale_color_continuous = scale_color_gradient
scale_colour_continuous = scale_color_gradient
scale_fill_continuous = scale_fill_gradient


# ---------------------------------------------------------------------------
# Factories — viridis family
# ---------------------------------------------------------------------------

def scale_color_viridis_c(*, option="viridis", direction=1, name=None,
                         breaks="default", labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=viridis_pal(option=option, direction=direction),
    )


def scale_fill_viridis_c(*, option="viridis", direction=1, name=None,
                        breaks="default", labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=viridis_pal(option=option, direction=direction),
    )


scale_colour_viridis_c = scale_color_viridis_c


# Discrete viridis lives with the discrete colour scale below — see
# scales.discrete for ScaleDiscreteColor.


# ---------------------------------------------------------------------------
# Factories — brewer continuous (a.k.a. distiller in ggplot2 nomenclature)
# ---------------------------------------------------------------------------

def scale_color_distiller(*, palette="Blues", direction=1, name=None,
                         breaks="default", labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=brewer_pal_continuous(palette=palette, direction=direction),
    )


def scale_fill_distiller(*, palette="Blues", direction=1, name=None,
                        breaks="default", labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=brewer_pal_continuous(palette=palette, direction=direction),
    )


scale_colour_distiller = scale_color_distiller
