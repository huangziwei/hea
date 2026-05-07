"""Non-colour aesthetic scales — size, alpha, shape, linetype.

These reuse the same continuous/discrete shape as the colour scales:

* :class:`ScaleContinuousColor` (renamed-as-alias) holds train/map for any
  continuous mapping where ``palette([0,1]_values) -> drawable_values``;
* :class:`ScaleDiscreteColor` does the same for discrete levels where
  ``palette(n) -> n_drawable_values``.

The output type is whatever the palette returns — colours for the colour
scales, mm sizes for size, marker codes for shape, linestyle specs for
linetype. The class names hint at colour for historical reasons but the
underlying logic is type-agnostic.
"""

from __future__ import annotations

from ._palettes import (
    alpha_pal,
    area_pal,
    linetype_pal,
    manual_pal,
    rescale_pal,
    shape_pal,
)
from .color_continuous import ScaleContinuousColor
from .discrete import ScaleDiscreteColor


# ---------------------------------------------------------------------------
# Size — continuous (default), area-proportional, manual, discrete
# ---------------------------------------------------------------------------

def scale_size_continuous(*, range=(1.0, 6.0), name=None, breaks="default",
                         labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("size",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=rescale_pal(range_=range),
    )


def scale_size_area(*, max_size=6.0, name=None, breaks="default",
                   labels="default", limits=None):
    """Area-proportional size: marker visual *area* tracks the data (not
    radius). ggplot2's recommended default for quantitative size mapping
    when zero is a meaningful baseline."""
    return ScaleContinuousColor(
        aesthetics=("size",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=area_pal(range_=(0.0, max_size)),
    )


def scale_size_manual(*, values, name=None, breaks="default",
                     labels="default", limits=None):
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("size",), name=name, breaks=breaks, labels=labels,
            limits=limits, values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("size",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=manual_pal(values),
    )


# alias matches ggplot2 — the ``_continuous`` suffix is the canonical default
scale_size = scale_size_continuous


def scale_radius(*, range=(1.0, 6.0), name=None, breaks="default",
                 labels="default", limits=None):
    """Linear-radius size scale. Unlike :func:`scale_size_area` (area ∝
    value), the *radius* tracks the value linearly. Most useful inside
    ``coord_polar`` for radial geoms, but applies anywhere ``size`` is
    mapped."""
    return ScaleContinuousColor(
        aesthetics=("size",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=rescale_pal(range_=range),
    )


# ---------------------------------------------------------------------------
# Alpha — continuous (default), manual
# ---------------------------------------------------------------------------

def scale_alpha_continuous(*, range=(0.1, 1.0), name=None, breaks="default",
                          labels="default", limits=None):
    return ScaleContinuousColor(
        aesthetics=("alpha",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=alpha_pal(range_=range),
    )


def scale_alpha_manual(*, values, name=None, breaks="default",
                      labels="default", limits=None):
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("alpha",), name=name, breaks=breaks, labels=labels,
            limits=limits, values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("alpha",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=manual_pal(values),
    )


scale_alpha = scale_alpha_continuous


# ---------------------------------------------------------------------------
# Shape — discrete only (continuous shape doesn't make sense)
# ---------------------------------------------------------------------------

def scale_shape(*, name=None, breaks="default", labels="default", limits=None):
    return ScaleDiscreteColor(
        aesthetics=("shape",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=shape_pal(),
    )


def scale_shape_manual(*, values, name=None, breaks="default",
                      labels="default", limits=None):
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("shape",), name=name, breaks=breaks, labels=labels,
            limits=limits, values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("shape",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=manual_pal(values),
    )


# ---------------------------------------------------------------------------
# Linetype — discrete only
# ---------------------------------------------------------------------------

def scale_linetype(*, name=None, breaks="default", labels="default", limits=None):
    return ScaleDiscreteColor(
        aesthetics=("linetype",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=linetype_pal(),
    )


def scale_linetype_manual(*, values, name=None, breaks="default",
                         labels="default", limits=None):
    if isinstance(values, dict):
        return ScaleDiscreteColor(
            aesthetics=("linetype",), name=name, breaks=breaks, labels=labels,
            limits=limits, values=dict(values),
        )
    return ScaleDiscreteColor(
        aesthetics=("linetype",), name=name, breaks=breaks, labels=labels,
        limits=limits, palette=manual_pal(values),
    )
