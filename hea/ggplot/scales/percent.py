"""``scale_x/y_percent`` — format axis ticks as percentages.

A thin :class:`ScaleContinuous` subclass that installs
:class:`matplotlib.ticker.PercentFormatter`. Default assumes data is in
``[0, 1]`` and multiplies by 100 for display (e.g. ``0.5`` → ``"50%"``).
For data already in ``[0, 100]`` pass ``xmax=100``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .continuous import ScaleContinuous


@dataclass
class ScalePercent(ScaleContinuous):
    """Continuous scale that formats ticks as percentages."""

    xmax: float = 1.0  # data unit corresponding to 100%
    decimals: int | None = None

    def apply_to_axis(self, ax, axis: str) -> None:
        from matplotlib.ticker import PercentFormatter

        if self.limits is not None:
            if axis == "x":
                ax.set_xlim(self.limits)
            else:
                ax.set_ylim(self.limits)

        target_axis = ax.xaxis if axis == "x" else ax.yaxis
        fmt = PercentFormatter(xmax=self.xmax, decimals=self.decimals)
        target_axis.set_major_formatter(fmt)


def scale_x_percent(*, name=None, breaks="default", labels="default",
                    limits=None, xmax=1.0, decimals=None):
    return ScalePercent(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits, xmax=xmax, decimals=decimals,
    )


def scale_y_percent(*, name=None, breaks="default", labels="default",
                    limits=None, xmax=1.0, decimals=None):
    return ScalePercent(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits, xmax=xmax, decimals=decimals,
    )
