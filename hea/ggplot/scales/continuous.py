"""``ScaleContinuous`` + ``scale_x_continuous`` / ``scale_y_continuous``.

Phase 1.1 form: matplotlib autoscale handles axis limits (so geoms with
non-trivial extents like bar widths still fit); the scale contributes
breaks + labels. User-supplied ``limits=`` overrides autoscale. Wilkinson
``extended_breaks`` parity is checklist 1.1c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scale import Scale, fmt_number
from .transformed import IdentityTrans, Trans


@dataclass
class ScaleContinuous(Scale):
    transform: Trans = field(default_factory=IdentityTrans)

    def apply_to_axis(self, ax, axis: str) -> None:
        # Set matplotlib axis scale FIRST. Done before limits because some
        # scales (log) reject non-positive limits and would error on the
        # default linear-scale autoscaled values otherwise.
        ms = self.transform.matplotlib_scale()
        if ms is not None:
            scale_name, scale_kwargs = ms
            if axis == "x":
                ax.set_xscale(scale_name, **scale_kwargs)
            else:
                ax.set_yscale(scale_name, **scale_kwargs)

        if self.limits is not None:
            if axis == "x":
                ax.set_xlim(self.limits)
            else:
                ax.set_ylim(self.limits)

        # Reverse: flip after any other limits are settled. matplotlib
        # treats lo>hi as an inverted axis automatically.
        if self.transform.reversed():
            if axis == "x":
                lo, hi = ax.get_xlim()
                ax.set_xlim(hi, lo)
            else:
                lo, hi = ax.get_ylim()
                ax.set_ylim(hi, lo)

        if self.breaks is None:
            if axis == "x":
                ax.set_xticks([])
            else:
                ax.set_yticks([])
            return

        # When a non-linear transform is in play and the user didn't ask for
        # explicit breaks, defer to matplotlib's native locator (LogLocator
        # for log; default for FuncScale). MaxNLocator-on-linear-coords would
        # produce e.g. evenly spaced linear ticks on a log axis — wrong.
        if self.breaks == "default" and ms is not None:
            return

        cur_lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
        breaks = self._compute_breaks(cur_lim)
        labels = self._compute_labels(breaks)
        if axis == "x":
            ax.set_xticks(breaks)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(breaks)
            ax.set_yticklabels(labels)

    def _compute_breaks(self, lim):
        from matplotlib.ticker import MaxNLocator

        if self.breaks == "default":
            # Phase 1.1: matplotlib MaxNLocator ≈ ggplot2's default. Real
            # Wilkinson `extended_breaks` port is 1.1c.
            loc = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
            return loc.tick_values(*lim)
        if callable(self.breaks):
            return np.asarray(self.breaks(lim))
        return np.asarray(self.breaks)

    def _compute_labels(self, breaks):
        if self.labels == "default":
            return [fmt_number(b) for b in breaks]
        if callable(self.labels):
            return list(self.labels(breaks))
        return [str(x) for x in self.labels]


def scale_x_continuous(*, name=None, breaks="default", labels="default",
                       limits=None, expand=(0.05, 0.0)):
    return ScaleContinuous(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits, expand=expand,
    )


def scale_y_continuous(*, name=None, breaks="default", labels="default",
                       limits=None, expand=(0.05, 0.0)):
    return ScaleContinuous(
        aesthetics=("y",), name=name, breaks=breaks, labels=labels,
        limits=limits, expand=expand,
    )
