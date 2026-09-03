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
from .scale import _NAME_MISSING, Scale


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

        colours = self.palette(normalised)

        if isinstance(data, pl.Series):
            return pl.Series(name=data.name, values=colours)
        return colours


def scale_color_gradient(
    *,
    low="#132B43",
    high="#56B1F7",
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=gradient_pal(low=low, high=high),
    )


def scale_fill_gradient(
    *,
    low="#132B43",
    high="#56B1F7",
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=gradient_pal(low=low, high=high),
    )


def scale_color_gradient2(
    *,
    low="#832424",
    mid="white",
    high="#3A3A98",
    midpoint=0,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=gradient2_pal(low=low, mid=mid, high=high, midpoint=midpoint),
    )


def scale_fill_gradient2(
    *,
    low="#832424",
    mid="white",
    high="#3A3A98",
    midpoint=0,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=gradient2_pal(low=low, mid=mid, high=high, midpoint=midpoint),
    )


def scale_color_gradientn(
    *, colours, name=_NAME_MISSING, breaks="default", labels="default", limits=None
):
    return ScaleContinuousColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=gradientn_pal(colours),
    )


def scale_fill_gradientn(
    *, colours, name=_NAME_MISSING, breaks="default", labels="default", limits=None
):
    return ScaleContinuousColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=gradientn_pal(colours),
    )


scale_colour_gradient = scale_color_gradient
scale_colour_gradient2 = scale_color_gradient2
scale_colour_gradientn = scale_color_gradientn

scale_color_continuous = scale_color_gradient
scale_colour_continuous = scale_color_gradient
scale_fill_continuous = scale_fill_gradient


@dataclass
class ScaleBinnedColor(ScaleContinuousColor):
    """Discretise a continuous range into ``n_breaks`` bins, then colour
    each bin with the palette evaluated at the bin's normalised centre.

    Matches ggplot2's ``scale_*_binned`` / ``_b`` family (viridis_b,
    gradient_b, …). Same training as :class:`ScaleContinuousColor`;
    the override is in ``map()``.
    """

    n_breaks: int = 10

    def map(self, data):  # type: ignore[override]
        if self.range_ is None or self.palette is None:
            return data
        lo, hi = self.range_

        if isinstance(data, pl.Series):
            arr = data.cast(pl.Float64).to_numpy()
        else:
            arr = np.asarray(data, dtype=float)

        edges = np.linspace(lo, hi, self.n_breaks + 1)
        idx = np.clip(np.digitize(arr, edges[1:-1], right=False), 0, self.n_breaks - 1)
        bin_centres = (np.arange(self.n_breaks) + 0.5) / self.n_breaks
        normalised = bin_centres[idx]
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            normalised = normalised.copy()
            normalised[nan_mask] = np.nan
        colours = self.palette(normalised)
        if isinstance(data, pl.Series):
            return pl.Series(name=data.name, values=colours)
        return colours


def scale_color_viridis_c(
    *,
    option="viridis",
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=viridis_pal(option=option, direction=direction),
    )


def scale_color_viridis_b(
    *,
    option="viridis",
    direction=1,
    n_breaks=10,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    """Binned viridis colour scale — discretises into ``n_breaks`` bins."""
    return ScaleBinnedColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=viridis_pal(option=option, direction=direction),
        n_breaks=n_breaks,
    )


def scale_fill_viridis_b(
    *,
    option="viridis",
    direction=1,
    n_breaks=10,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    """Binned viridis fill scale — discretises into ``n_breaks`` bins."""
    return ScaleBinnedColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=viridis_pal(option=option, direction=direction),
        n_breaks=n_breaks,
    )


scale_colour_viridis_b = scale_color_viridis_b


def scale_fill_viridis_c(
    *,
    option="viridis",
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=viridis_pal(option=option, direction=direction),
    )


scale_colour_viridis_c = scale_color_viridis_c


def scale_color_distiller(
    *,
    palette="Blues",
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=brewer_pal_continuous(palette=palette, direction=direction),
    )


def scale_fill_distiller(
    *,
    palette="Blues",
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleContinuousColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=brewer_pal_continuous(palette=palette, direction=direction),
    )


scale_colour_distiller = scale_color_distiller
