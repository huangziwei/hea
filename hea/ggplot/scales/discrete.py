"""``ScaleDiscreteColor`` and friends — non-positional discrete scales.

A discrete scale tracks the *unique levels* in its trained data and maps
each level to a value drawn from a palette function. For colour/fill
that's a hex code; for shape, a marker glyph; for linetype, a dash spec.
For now we only ship ``ScaleDiscreteColor``; the rest land in 1.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..._polars_compat import cat_pool
from ._palettes import (
    brewer_pal_discrete,
    colorblind_pal,
    hue_pal,
    manual_pal,
    viridis_pal_discrete,
)
from .scale import _NAME_MISSING, Scale


def _polars_dtype_for(values) -> pl.DataType:
    """Best-fit polars dtype for ``values`` — Float64 if any element is
    numeric (size/alpha palettes), else Utf8 (colour/shape/linetype
    palettes return hex strings or marker codes)."""
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return pl.Float64
        if v is not None:
            return pl.Utf8
    return pl.Utf8


@dataclass
class ScaleDiscreteColor(Scale):
    """Maps discrete levels (e.g. species names) to colours via a palette."""

    palette: Any = None  # None = default hue_pal; else a callable n -> list[str]
    values: Any = None  # explicit dict {level: color} — wins over palette
    levels: list | None = field(default=None, init=False, repr=False)

    def train(self, data) -> None:
        if isinstance(data, pl.Series):
            if data.dtype in (pl.Categorical, pl.Enum):
                new_levels = cat_pool(data).to_list()
            else:
                new_levels = sorted(data.drop_nulls().unique().to_list())
        else:
            new_levels = sorted({v for v in data if v is not None})
        if self.levels is None:
            self.levels = list(new_levels)
        else:
            for v in new_levels:
                if v not in self.levels:
                    self.levels.append(v)

    def map(self, data):
        if self.levels is None or len(self.levels) == 0:
            return data

        if isinstance(self.values, dict):
            mapping = dict(self.values)
        else:
            pal = self.palette if self.palette is not None else hue_pal()
            colours = pal(len(self.levels))
            mapping = dict(zip(self.levels, colours))

        return_dtype = _polars_dtype_for(mapping.values())

        if isinstance(data, pl.Series):
            return data.map_elements(
                lambda v: mapping.get(v),
                return_dtype=return_dtype,
            ).alias(data.name)
        return [mapping.get(v) for v in data]


def scale_color_manual(
    *, values, name=_NAME_MISSING, breaks="default", labels="default", limits=None
):
    """Manual qualitative palette. ``values`` may be a list (ordered) or a
    dict ``{level: hex}`` (explicit per-level)."""
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("colour",),
            name=name,
            breaks=breaks,
            labels=labels,
            limits=limits,
            values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=manual_pal(values),
    )


def scale_fill_manual(
    *, values, name=_NAME_MISSING, breaks="default", labels="default", limits=None
):
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("fill",),
            name=name,
            breaks=breaks,
            labels=labels,
            limits=limits,
            values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=manual_pal(values),
    )


scale_colour_manual = scale_color_manual


@dataclass
class ScaleIdentity(Scale):
    """Pass-through scale: the column already holds drawable values."""

    def train(self, data) -> None:
        pass

    def map(self, data):
        return data


def scale_color_identity(*, name=_NAME_MISSING):
    return ScaleIdentity(aesthetics=("colour",), name=name)


def scale_fill_identity(*, name=_NAME_MISSING):
    return ScaleIdentity(aesthetics=("fill",), name=name)


scale_colour_identity = scale_color_identity


def scale_color_viridis_d(
    *,
    option="viridis",
    direction=1,
    begin=0.0,
    end=1.0,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleDiscreteColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=viridis_pal_discrete(
            option=option, direction=direction, begin=begin, end=end
        ),
    )


def scale_fill_viridis_d(
    *,
    option="viridis",
    direction=1,
    begin=0.0,
    end=1.0,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleDiscreteColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=viridis_pal_discrete(
            option=option, direction=direction, begin=begin, end=end
        ),
    )


scale_colour_viridis_d = scale_color_viridis_d


def scale_color_brewer(
    *,
    palette="Set1",
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    """Discrete ColorBrewer palette. Common picks: ``Set1``/``Set2`` (qualitative),
    ``RdBu``/``Spectral`` (diverging), ``Blues``/``YlOrRd`` (sequential)."""
    return ScaleDiscreteColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=brewer_pal_discrete(palette=palette, direction=direction),
    )


def scale_fill_brewer(
    *,
    palette="Set1",
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleDiscreteColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=brewer_pal_discrete(palette=palette, direction=direction),
    )


scale_colour_brewer = scale_color_brewer


def scale_color_hue(
    *,
    h=(15, 375),
    c=100,
    lightness=65,
    h_start=0,
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    """Equally-spaced HCL hues. ggplot2's default discrete-colour palette;
    explicit form lets you tune chroma / lightness / hue range."""
    return ScaleDiscreteColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=hue_pal(
            h=h, c=c, lightness=lightness, h_start=h_start, direction=direction
        ),
    )


def scale_fill_hue(
    *,
    h=(15, 375),
    c=100,
    lightness=65,
    h_start=0,
    direction=1,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
):
    return ScaleDiscreteColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=hue_pal(
            h=h, c=c, lightness=lightness, h_start=h_start, direction=direction
        ),
    )


scale_colour_hue = scale_color_hue


scale_color_discrete = scale_color_hue
scale_colour_discrete = scale_color_hue
scale_fill_discrete = scale_fill_hue


def scale_color_colorblind(
    *, name=_NAME_MISSING, breaks="default", labels="default", limits=None
):
    """ggthemes' ``scale_colour_colorblind`` — Okabe-Ito 8-colour qualitative
    palette designed to remain distinguishable under common colour-vision
    deficiencies. 8 levels max."""
    return ScaleDiscreteColor(
        aesthetics=("colour",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=colorblind_pal(),
    )


def scale_fill_colorblind(
    *, name=_NAME_MISSING, breaks="default", labels="default", limits=None
):
    return ScaleDiscreteColor(
        aesthetics=("fill",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        palette=colorblind_pal(),
    )


scale_colour_colorblind = scale_color_colorblind
