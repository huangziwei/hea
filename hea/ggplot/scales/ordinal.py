"""``scale_x/y_ordinal`` — discrete positional scales.

ggplot2's ``scale_*_discrete``/``scale_*_ordinal``. Each unique level on
the axis sits at integer position ``1..n`` (R uses 1-based; matplotlib
uses 0-based but the visual result is identical).

Implementation: matplotlib already maps strings → integer positions when
you call ``ax.plot(string_values, ...)`` — we just need to keep the axis
out of ScaleContinuous's MaxNLocator path and let matplotlib's category
unit handle the labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scale import Scale


@dataclass
class ScaleOrdinal(Scale):
    """Discrete positional scale. ``apply_to_axis`` is mostly a no-op —
    matplotlib already places categorical strings sensibly."""

    def apply_to_axis(self, ax, axis: str) -> None:
        if self.limits is not None:
            if axis == "x":
                ax.set_xlim(self.limits)
            else:
                ax.set_ylim(self.limits)


def scale_x_ordinal(*, name=None, breaks="default", labels="default",
                    limits=None):
    return ScaleOrdinal(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits,
    )


def scale_y_ordinal(*, name=None, breaks="default", labels="default",
                    limits=None):
    return ScaleOrdinal(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits,
    )


# ggplot2's other spelling.
scale_x_discrete = scale_x_ordinal
scale_y_discrete = scale_y_ordinal
