"""``scale_x/y_ordinal`` — discrete positional scales.

ggplot2's ``scale_*_discrete``/``scale_*_ordinal``. Each unique level on
the axis sits at integer position ``0..n-1`` (matplotlib uses 0-based;
visually identical to R's 1-based).

``limits=`` controls which levels appear and their order — accepts:
  * ``None``         — use trained levels (sorted strings or Enum order).
  * ``list``         — explicit ordering; rows whose value is not in the
                       list are dropped (matches R's "removed rows
                       containing non-finite outside the scale range").
  * ``callable(x)``  — applied to trained levels; e.g. ``limits=reversed``
                       for ggplot2's ``limits=rev``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .scale import Scale


@dataclass
class ScaleOrdinal(Scale):
    # Trained category levels in the order they should appear on the axis
    # absent an explicit ``limits=`` override.
    levels: list | None = field(default=None, init=False, repr=False)
    # Default discrete expansion: matches ggplot2's
    # ``expansion(add=0.6, mult=0)`` — half a category-position of padding
    # on each side so bars/points don't kiss the axis bounds.
    expand: tuple = (0.0, 0.6)

    def train(self, data) -> None:
        """Accumulate observed category levels across calls.

        Order rule: ``pl.Enum`` / ``pl.Categorical`` use their declared
        category order (matches R's factor levels); plain strings sort
        alphabetically (matches R's ``factor()`` default which calls
        ``sort(unique(x))``).
        """
        if data is None or len(data) == 0:
            return
        if isinstance(data, pl.Series):
            if data.dtype in (pl.Categorical, pl.Enum):
                new_levels = [str(v)
                              for v in data.cat.get_categories().to_list()]
            else:
                new_levels = sorted(
                    str(v) for v in data.drop_nulls().unique().to_list()
                )
        else:
            new_levels = sorted(str(v) for v in data if v is not None)
        if self.levels is None:
            self.levels = []
        for v in new_levels:
            if v not in self.levels:
                self.levels.append(v)

    def resolved_limits(self) -> list[str]:
        """Return the final axis order after applying ``limits=``."""
        trained = list(self.levels) if self.levels else []
        if self.limits is None:
            return trained
        if callable(self.limits):
            return [str(v) for v in self.limits(trained)]
        return [str(v) for v in self.limits]

    def setup_axis(self, ax, axis: str) -> None:
        """Lock in the category order on matplotlib's axis converter
        before any geom draws.

        matplotlib's ``StrCategoryConverter`` registers strings on first
        encounter; calling ``update_units`` here pre-registers the levels
        in our resolved order, so subsequent ``ax.bar([...])`` /
        ``ax.scatter([...])`` calls put each value at the integer
        position dictated by our order, not the data's row order.
        """
        levels = self.resolved_limits()
        if not levels:
            return
        if axis == "x":
            ax.xaxis.update_units(levels)
        else:
            ax.yaxis.update_units(levels)

    def apply_to_axis(self, ax, axis: str) -> None:
        levels = self.resolved_limits()
        n = len(levels)
        if n == 0:
            return

        # ggplot2's discrete expansion (``expansion(add=0.6, mult=0)``)
        # on a 0-based n-category axis → ``[-0.6, n - 1 + 0.6]``. We
        # only set this when ``limits`` is explicit (otherwise let
        # matplotlib autoscale from the artists, which gives a tighter
        # fit that matches existing behaviour for the default-order case).
        if self.limits is not None:
            pad_lo, pad_hi = self._padding()
            lo = -pad_lo
            hi = (n - 1) + pad_hi
            if axis == "x":
                ax.set_xlim(lo, hi)
            else:
                ax.set_ylim(lo, hi)

        # Tick labels: by default the level strings themselves (matches
        # R: ``breaks`` and ``labels`` both default to the levels). A
        # user-supplied ``breaks`` / ``labels`` can override.
        if self.breaks is None:
            ticks = []
            tick_labels = []
        elif self.breaks == "default":
            ticks = list(range(n))
            tick_labels = list(levels)
        else:
            # Explicit breaks: list of level names. Position each at its
            # index in ``levels``; drop ones not in levels.
            ticks = []
            tick_labels = []
            for b in self.breaks:
                s = str(b)
                if s in levels:
                    ticks.append(levels.index(s))
                    tick_labels.append(s)

        if self.labels != "default" and ticks:
            if callable(self.labels):
                tick_labels = [str(s) for s in self.labels(tick_labels)]
            else:
                tick_labels = [str(s) for s in self.labels]

        if axis == "x":
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)

    def _padding(self) -> tuple[float, float]:
        """Return ``(pad_lo, pad_hi)`` from the ``expand`` field.

        Accepts ``Expansion``, legacy ``(mult, add)`` tuple, or anything
        else (falls back to ggplot2's discrete default ``add=0.6``).
        """
        from ..expansion import Expansion

        exp = self.expand
        if isinstance(exp, Expansion):
            _, _, a_lo, a_hi = exp.split()
            return (float(a_lo), float(a_hi))
        if isinstance(exp, (list, tuple)) and len(exp) >= 2:
            # Order is ``(mult, add)`` to match ScaleContinuous's legacy
            # form; for discrete we use ``add`` only.
            return (float(exp[1]), float(exp[1]))
        return (0.6, 0.6)


def scale_x_ordinal(*, name=None, breaks="default", labels="default",
                    limits=None, expand=None):
    kwargs = dict(aesthetics=("x",), name=name, breaks=breaks,
                  labels=labels, limits=limits)
    if expand is not None:
        kwargs["expand"] = expand
    return ScaleOrdinal(**kwargs)


def scale_y_ordinal(*, name=None, breaks="default", labels="default",
                    limits=None, expand=None):
    kwargs = dict(aesthetics=("y",), name=name, breaks=breaks,
                  labels=labels, limits=limits)
    if expand is not None:
        kwargs["expand"] = expand
    return ScaleOrdinal(**kwargs)


# ggplot2's other spelling.
scale_x_discrete = scale_x_ordinal
scale_y_discrete = scale_y_ordinal
