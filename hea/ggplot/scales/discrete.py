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

from ._palettes import brewer_pal_discrete, hue_pal, manual_pal, viridis_pal_discrete
from .scale import Scale


@dataclass
class ScaleDiscreteColor(Scale):
    """Maps discrete levels (e.g. species names) to colours via a palette."""

    palette: Any = None  # None = default hue_pal; else a callable n -> list[str]
    values: Any = None    # explicit dict {level: color} — wins over palette
    levels: list | None = field(default=None, init=False, repr=False)

    def train(self, data) -> None:
        if isinstance(data, pl.Series):
            if data.dtype in (pl.Categorical, pl.Enum):
                # Categorical / Enum carry an explicit level order — honour it
                # (matches ggplot2's behaviour for factor columns: factor levels
                # drive the scale order regardless of which row appears first).
                new_levels = data.cat.get_categories().to_list()
            else:
                # Plain string / boolean: sort alphabetically — same as R's
                # ``factor(...)`` default. ggplot2 silently runs character
                # columns through ``factor()`` before mapping, so the level
                # order ends up sorted regardless of CSV row order.
                new_levels = sorted(data.drop_nulls().unique().to_list())
        else:
            new_levels = sorted(set(v for v in data if v is not None))
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

        if isinstance(data, pl.Series):
            return data.map_elements(
                lambda v: mapping.get(v),
                return_dtype=pl.Utf8,
            ).alias(data.name)
        return [mapping.get(v) for v in data]


def scale_color_manual(*, values, name=None, breaks="default", labels="default",
                      limits=None):
    """Manual qualitative palette. ``values`` may be a list (ordered) or a
    dict ``{level: hex}`` (explicit per-level)."""
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
            limits=limits, values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=manual_pal(values),
    )


def scale_fill_manual(*, values, name=None, breaks="default", labels="default",
                     limits=None):
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
            limits=limits, values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=manual_pal(values),
    )


# British/American aliases.
scale_colour_manual = scale_color_manual


@dataclass
class ScaleIdentity(Scale):
    """Pass-through scale: the column already holds drawable values."""

    def train(self, data) -> None:
        pass

    def map(self, data):
        return data


def scale_color_identity(*, name=None):
    return ScaleIdentity(aesthetics=("colour",), name=name)


def scale_fill_identity(*, name=None):
    return ScaleIdentity(aesthetics=("fill",), name=name)


scale_colour_identity = scale_color_identity


# ---------------------------------------------------------------------------
# Discrete palette factories — viridis_d, brewer
# ---------------------------------------------------------------------------

def scale_color_viridis_d(*, option="viridis", direction=1, begin=0.0, end=1.0,
                         name=None, breaks="default", labels="default",
                         limits=None):
    return ScaleDiscreteColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=viridis_pal_discrete(option=option, direction=direction,
                                      begin=begin, end=end),
    )


def scale_fill_viridis_d(*, option="viridis", direction=1, begin=0.0, end=1.0,
                        name=None, breaks="default", labels="default",
                        limits=None):
    return ScaleDiscreteColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=viridis_pal_discrete(option=option, direction=direction,
                                      begin=begin, end=end),
    )


scale_colour_viridis_d = scale_color_viridis_d


def scale_color_brewer(*, palette="Set1", direction=1, name=None,
                      breaks="default", labels="default", limits=None):
    """Discrete ColorBrewer palette. Common picks: ``Set1``/``Set2`` (qualitative),
    ``RdBu``/``Spectral`` (diverging), ``Blues``/``YlOrRd`` (sequential)."""
    return ScaleDiscreteColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=brewer_pal_discrete(palette=palette, direction=direction),
    )


def scale_fill_brewer(*, palette="Set1", direction=1, name=None,
                     breaks="default", labels="default", limits=None):
    return ScaleDiscreteColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=brewer_pal_discrete(palette=palette, direction=direction),
    )


scale_colour_brewer = scale_color_brewer


# ---------------------------------------------------------------------------
# Default qualitative palette — equally-spaced HCL hues (ggplot2's default).
# ---------------------------------------------------------------------------

def scale_color_hue(*, h=(15, 375), c=100, l=65, h_start=0, direction=1,
                    name=None, breaks="default", labels="default", limits=None):
    """Equally-spaced HCL hues. ggplot2's default discrete-colour palette;
    explicit form lets you tune chroma / lightness / hue range."""
    return ScaleDiscreteColor(
        aesthetics=("colour",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=hue_pal(h=h, c=c, l=l, h_start=h_start, direction=direction),
    )


def scale_fill_hue(*, h=(15, 375), c=100, l=65, h_start=0, direction=1,
                   name=None, breaks="default", labels="default", limits=None):
    return ScaleDiscreteColor(
        aesthetics=("fill",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        palette=hue_pal(h=h, c=c, l=l, h_start=h_start, direction=direction),
    )


scale_colour_hue = scale_color_hue
