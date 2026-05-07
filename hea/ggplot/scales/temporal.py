"""Temporal positional scales — ``scale_x/y_date``, ``_datetime``, ``_time``.

Each is a thin :class:`ScaleContinuous` subclass that installs matplotlib's
date locator/formatter on ``apply_to_axis``. Polars ``Date``/``Datetime``
columns convert to numpy ``datetime64`` automatically when the geom calls
``.to_numpy()``, so matplotlib already handles the data; the scale's
job is to pick sensible ticks.

* ``scale_*_date`` — date-only ticks (no time-of-day).
* ``scale_*_datetime`` — datetime ticks with time portion.
* ``scale_*_time`` — time-of-day ticks (treats values as same-day).

``date_breaks=`` / ``date_labels=`` are accepted but currently fall through
to matplotlib's auto-locator/formatter; explicit string forms like
``"1 month"`` and ``"%Y-%m"`` would land in a polish pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from .continuous import ScaleContinuous


@dataclass
class ScaleDate(ScaleContinuous):
    """Date axis. Default formatter strips time."""

    date_format: str | None = None  # passed to mdates.DateFormatter

    def apply_to_axis(self, ax, axis: str) -> None:
        import matplotlib.dates as mdates

        target_axis = ax.xaxis if axis == "x" else ax.yaxis
        target_axis.set_major_locator(mdates.AutoDateLocator())
        target_axis.set_major_formatter(
            mdates.DateFormatter(self.date_format or "%Y-%m-%d")
        )

        # Limits, if user set them, still apply.
        if self.limits is not None:
            if axis == "x":
                ax.set_xlim(self.limits)
            else:
                ax.set_ylim(self.limits)


@dataclass
class ScaleDatetime(ScaleContinuous):
    """Datetime axis. Default formatter shows year-month-day plus time."""

    date_format: str | None = None

    def apply_to_axis(self, ax, axis: str) -> None:
        import matplotlib.dates as mdates

        target_axis = ax.xaxis if axis == "x" else ax.yaxis
        target_axis.set_major_locator(mdates.AutoDateLocator())
        target_axis.set_major_formatter(
            mdates.DateFormatter(self.date_format or "%Y-%m-%d %H:%M")
        )

        if self.limits is not None:
            if axis == "x":
                ax.set_xlim(self.limits)
            else:
                ax.set_ylim(self.limits)


@dataclass
class ScaleTime(ScaleContinuous):
    """Time-of-day axis."""

    date_format: str | None = None

    def apply_to_axis(self, ax, axis: str) -> None:
        import matplotlib.dates as mdates

        target_axis = ax.xaxis if axis == "x" else ax.yaxis
        target_axis.set_major_locator(mdates.AutoDateLocator())
        target_axis.set_major_formatter(
            mdates.DateFormatter(self.date_format or "%H:%M:%S")
        )

        if self.limits is not None:
            if axis == "x":
                ax.set_xlim(self.limits)
            else:
                ax.set_ylim(self.limits)


def scale_x_date(*, name=None, breaks="default", labels="default",
                 limits=None, date_format=None):
    return ScaleDate(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits, date_format=date_format,
    )


def scale_y_date(*, name=None, breaks="default", labels="default",
                 limits=None, date_format=None):
    return ScaleDate(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits, date_format=date_format,
    )


def scale_x_datetime(*, name=None, breaks="default", labels="default",
                     limits=None, date_format=None):
    return ScaleDatetime(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits, date_format=date_format,
    )


def scale_y_datetime(*, name=None, breaks="default", labels="default",
                     limits=None, date_format=None):
    return ScaleDatetime(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits, date_format=date_format,
    )


def scale_x_time(*, name=None, breaks="default", labels="default",
                 limits=None, date_format=None):
    return ScaleTime(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits, date_format=date_format,
    )


def scale_y_time(*, name=None, breaks="default", labels="default",
                 limits=None, date_format=None):
    return ScaleTime(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits, date_format=date_format,
    )
