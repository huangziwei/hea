"""Temporal positional scales — ``scale_x/y_date``, ``_datetime``, ``_time``.

Each is a thin :class:`ScaleContinuous` subclass that installs matplotlib's
date locator/formatter on ``apply_to_axis``. Polars ``Date``/``Datetime``
columns convert to numpy ``datetime64`` automatically when the geom calls
``.to_numpy()``, so matplotlib already handles the data; the scale's
job is to pick sensible ticks.

* ``scale_*_date`` — date-only ticks (no time-of-day).
* ``scale_*_datetime`` — datetime ticks with time portion.
* ``scale_*_time`` — time-of-day ticks (treats values as same-day).

``date_labels=`` is accepted as the R-style alias for ``date_format=`` —
both feed matplotlib's :class:`DateFormatter`. Explicit ``breaks=`` (a
list / array / polars Series of dates) installs a fixed locator at
those dates; ``"default"`` keeps matplotlib's :class:`AutoDateLocator`.
``date_breaks=`` (string shorthand like ``"1 month"``) is a polish item.
"""

from __future__ import annotations

from dataclasses import dataclass

from .continuous import ScaleContinuous
from .scale import _NAME_MISSING


def _to_mpl_dates(values):
    """Coerce a polars Series / numpy array / iterable of date-like
    things into a list matplotlib's ``date2num`` can ingest.

    Polars Date / Datetime columns yield Python ``date`` / ``datetime``
    via ``.to_list()``, which ``date2num`` accepts directly."""
    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None and isinstance(values, pl.Series):
        return values.to_list()
    return list(values)


def _apply_date_axis(scale, ax, axis: str, *, default_format: str,
                     view_limits) -> None:
    """Shared ``apply_to_axis`` body for date / datetime / time scales.

    Honours explicit ``breaks=`` (list / Series of dates) by installing a
    :class:`FixedLocator`; falls back to matplotlib's
    :class:`AutoDateLocator` for ``"default"``. The formatter pattern
    comes from ``date_format`` (== R's ``date_labels``)."""
    import matplotlib.dates as mdates
    from matplotlib.ticker import FixedLocator

    target_axis = ax.xaxis if axis == "x" else ax.yaxis

    if isinstance(scale.breaks, str) and scale.breaks == "default":
        target_axis.set_major_locator(mdates.AutoDateLocator())
    else:
        breaks = scale.breaks
        if callable(breaks):
            lim = view_limits or scale.limits or (None, None)
            breaks = breaks(lim)
        target_axis.set_major_locator(
            FixedLocator(mdates.date2num(_to_mpl_dates(breaks)))
        )

    target_axis.set_major_formatter(
        mdates.DateFormatter(scale.date_format or default_format)
    )

    # Optional explicit ``labels=[...]`` overrides the formatter (used
    # when the user wants completely custom strings, not strftime).
    if not (isinstance(scale.labels, str) and scale.labels == "default") \
            and scale.labels is not None and not callable(scale.labels):
        try:
            ticks = target_axis.get_major_locator()()
            target_axis.set_major_formatter(
                mdates.DateFormatter("")  # blank to start
            )
            target_axis.set_ticklabels(list(scale.labels))
            del ticks  # silence unused
        except Exception:
            pass

    if view_limits is not None:
        if axis == "x":
            ax.set_xlim(view_limits)
        else:
            ax.set_ylim(view_limits)
    elif scale.limits is not None:
        if axis == "x":
            ax.set_xlim(scale.limits)
        else:
            ax.set_ylim(scale.limits)


@dataclass
class ScaleDate(ScaleContinuous):
    """Date axis. Default formatter strips time."""

    date_format: str | None = None  # passed to mdates.DateFormatter

    def apply_to_axis(self, ax, axis: str, view_limits=None) -> None:
        _apply_date_axis(self, ax, axis,
                         default_format="%Y-%m-%d",
                         view_limits=view_limits)


@dataclass
class ScaleDatetime(ScaleContinuous):
    """Datetime axis. Default formatter shows year-month-day plus time."""

    date_format: str | None = None

    def apply_to_axis(self, ax, axis: str, view_limits=None) -> None:
        _apply_date_axis(self, ax, axis,
                         default_format="%Y-%m-%d %H:%M",
                         view_limits=view_limits)


@dataclass
class ScaleTime(ScaleContinuous):
    """Time-of-day axis."""

    date_format: str | None = None

    def apply_to_axis(self, ax, axis: str, view_limits=None) -> None:
        _apply_date_axis(self, ax, axis,
                         default_format="%H:%M:%S",
                         view_limits=view_limits)


def _resolve_date_format(date_format, date_labels):
    """``date_labels`` is R's spelling; both feed ``DateFormatter``."""
    if date_format is None:
        return date_labels
    return date_format


def scale_x_date(*, name=_NAME_MISSING, breaks="default", labels="default",
                 limits=None, date_format=None, date_labels=None):
    return ScaleDate(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        date_format=_resolve_date_format(date_format, date_labels),
    )


def scale_y_date(*, name=_NAME_MISSING, breaks="default", labels="default",
                 limits=None, date_format=None, date_labels=None):
    return ScaleDate(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        date_format=_resolve_date_format(date_format, date_labels),
    )


def scale_x_datetime(*, name=_NAME_MISSING, breaks="default", labels="default",
                     limits=None, date_format=None, date_labels=None):
    return ScaleDatetime(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        date_format=_resolve_date_format(date_format, date_labels),
    )


def scale_y_datetime(*, name=_NAME_MISSING, breaks="default", labels="default",
                     limits=None, date_format=None, date_labels=None):
    return ScaleDatetime(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        date_format=_resolve_date_format(date_format, date_labels),
    )


def scale_x_time(*, name=_NAME_MISSING, breaks="default", labels="default",
                 limits=None, date_format=None, date_labels=None):
    return ScaleTime(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        date_format=_resolve_date_format(date_format, date_labels),
    )


def scale_y_time(*, name=_NAME_MISSING, breaks="default", labels="default",
                 limits=None, date_format=None, date_labels=None):
    return ScaleTime(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits,
        date_format=_resolve_date_format(date_format, date_labels),
    )
