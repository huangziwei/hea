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
        else:
            # Honour an Expansion's symmetric multiplicative padding via
            # matplotlib's margins (asymmetric / additive is polish).
            self._apply_expansion(ax, axis)

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

    def _apply_expansion(self, ax, axis: str) -> None:
        from ..expansion import Expansion

        exp = self.expand
        if isinstance(exp, Expansion):
            m_lo, m_hi, _a_lo, _a_hi = exp.split()
            # Symmetric in matplotlib's margins API.
            mult = max(m_lo, m_hi)
        elif isinstance(exp, (list, tuple)) and len(exp) >= 1:
            mult = float(exp[0])
        else:
            return
        if mult <= 0:
            return
        if axis == "x":
            ax.margins(x=mult)
        else:
            ax.margins(y=mult)

    def _compute_breaks(self, lim):
        if self.breaks == "default":
            from ._breaks import extended_breaks

            return extended_breaks(lim[0], lim[1], m=5)
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


def _coerce_limits(args, kwarg_lo, kwarg_hi):
    """Accept ``xlim(lo, hi)``, ``xlim((lo, hi))``, ``xlim([lo, hi])``."""
    if args and (kwarg_lo is not None or kwarg_hi is not None):
        raise TypeError("pass limits either positionally or as a tuple, not both")
    if not args:
        return (kwarg_lo, kwarg_hi)
    if len(args) == 1:
        first = args[0]
        if isinstance(first, (list, tuple)):
            if len(first) != 2:
                raise ValueError(f"limits must have length 2; got {len(first)}")
            return tuple(first)
        raise TypeError(
            "single-arg form must be a (lo, hi) tuple/list; "
            "use xlim(lo, hi) for two-arg form"
        )
    if len(args) == 2:
        return (args[0], args[1])
    raise TypeError(f"expected 1 tuple or 2 scalars; got {len(args)} args")


def xlim(*args, lo=None, hi=None):
    """Shortcut for ``scale_x_continuous(limits=(lo, hi))``.

    ``xlim(0, 10)``, ``xlim((0, 10))``, and ``xlim(lo=0, hi=10)`` all work.
    A bound of ``None`` leaves that side to matplotlib's autoscale.
    """
    return scale_x_continuous(limits=_coerce_limits(args, lo, hi))


def ylim(*args, lo=None, hi=None):
    """Shortcut for ``scale_y_continuous(limits=(lo, hi))``."""
    return scale_y_continuous(limits=_coerce_limits(args, lo, hi))


def lims(*, x=None, y=None, **rest):
    """Set limits on multiple aesthetics in one call.

    Currently supports ``x`` and ``y`` only — non-positional limits
    (``colour=``, ``fill=``, ...) need guide infrastructure that hasn't
    landed yet. Returns a list of scales, which ``ggplot.__add__`` already
    accepts (see ``core.py``'s ``list`` dispatch).
    """
    if rest:
        unknown = ", ".join(sorted(rest))
        raise NotImplementedError(
            f"lims() supports x= and y= only for now (got {unknown}). "
            f"Non-positional limits land with guide infrastructure."
        )
    out = []
    if x is not None:
        out.append(xlim(x))
    if y is not None:
        out.append(ylim(y))
    return out
