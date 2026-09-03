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

import polars as pl

from ..._polars_compat import cat_pool
from .scale import _NAME_MISSING, Scale


@dataclass
class ScaleOrdinal(Scale):
    levels: list | None = field(default=None, init=False, repr=False)
    expand: tuple = (0.0, 0.6)

    def train(self, data) -> None:
        """Accumulate observed category levels across calls.

        Order rule: ``pl.Enum`` / ``pl.Categorical`` use their declared
        category order (matches R's factor levels); plain strings sort
        alphabetically (matches R's ``factor()`` default which calls
        ``sort(unique(x))``).

        Numeric data is ignored once levels have been captured — the
        build pipeline calls ``train`` again post-position-adjustment
        with integer positions in place of strings, and we don't want
        those float positions to graft onto the level list.
        """
        if data is None or len(data) == 0:
            return
        if isinstance(data, pl.Series):
            if data.dtype.is_numeric() and self.levels:
                return
            if data.dtype in (pl.Categorical, pl.Enum):
                new_levels = [str(v) for v in cat_pool(data).to_list()]
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

    def apply_to_axis(self, ax, axis: str, view_limits=None) -> None:
        levels = self.resolved_limits()
        n = len(levels)
        if n == 0:
            return

        if view_limits is not None:
            if axis == "x":
                ax.set_xlim(view_limits)
            else:
                ax.set_ylim(view_limits)
        elif self.limits is not None:
            pad_lo, pad_hi = self._padding()
            lo = -pad_lo
            hi = (n - 1) + pad_hi
            if axis == "x":
                ax.set_xlim(lo, hi)
            else:
                ax.set_ylim(lo, hi)

        if self.breaks is None:
            ticks = []
            tick_labels = []
        elif isinstance(self.breaks, str) and self.breaks == "default":
            ticks = list(range(n))
            tick_labels = list(levels)
        else:
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
        """Return ``(pad_lo, pad_hi)`` from the ``expand`` field."""
        from ..expansion import Expansion

        exp = self.expand
        if isinstance(exp, Expansion):
            _, _, a_lo, a_hi = exp.split()
            return (float(a_lo), float(a_hi))
        if isinstance(exp, (list, tuple)) and len(exp) >= 2:
            return (float(exp[1]), float(exp[1]))
        return (0.6, 0.6)


def scale_x_ordinal(
    *, name=_NAME_MISSING, breaks="default", labels="default", limits=None, expand=None
):
    kwargs = {
        "aesthetics": ("x",),
        "name": name,
        "breaks": breaks,
        "labels": labels,
        "limits": limits,
    }
    if expand is not None:
        kwargs["expand"] = expand
    return ScaleOrdinal(**kwargs)


def scale_y_ordinal(
    *, name=_NAME_MISSING, breaks="default", labels="default", limits=None, expand=None
):
    kwargs = {
        "aesthetics": ("y",),
        "name": name,
        "breaks": breaks,
        "labels": labels,
        "limits": limits,
    }
    if expand is not None:
        kwargs["expand"] = expand
    return ScaleOrdinal(**kwargs)


scale_x_discrete = scale_x_ordinal
scale_y_discrete = scale_y_ordinal
