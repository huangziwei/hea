"""Scales — the data-to-axis-appearance contract."""

from .continuous import ScaleContinuous, scale_x_continuous, scale_y_continuous
from .discrete import (
    ScaleDiscreteColor,
    ScaleIdentity,
    scale_color_identity,
    scale_color_manual,
    scale_colour_identity,
    scale_colour_manual,
    scale_fill_identity,
    scale_fill_manual,
)
from .list import ScalesList
from .scale import Scale
from .transformed import (
    IdentityTrans,
    Log10Trans,
    Log2Trans,
    ReverseTrans,
    SqrtTrans,
    Trans,
    scale_x_log10,
    scale_x_log2,
    scale_x_reverse,
    scale_x_sqrt,
    scale_y_log10,
    scale_y_log2,
    scale_y_reverse,
    scale_y_sqrt,
)

__all__ = [
    "Scale",
    "ScaleContinuous",
    "ScaleDiscreteColor",
    "ScaleIdentity",
    "ScalesList",
    "Trans",
    "IdentityTrans",
    "Log10Trans",
    "Log2Trans",
    "SqrtTrans",
    "ReverseTrans",
    "scale_x_continuous",
    "scale_y_continuous",
    "scale_x_log10",
    "scale_y_log10",
    "scale_x_log2",
    "scale_y_log2",
    "scale_x_sqrt",
    "scale_y_sqrt",
    "scale_x_reverse",
    "scale_y_reverse",
    "scale_color_manual",
    "scale_colour_manual",
    "scale_fill_manual",
    "scale_color_identity",
    "scale_colour_identity",
    "scale_fill_identity",
]
