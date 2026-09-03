"""``ScaleContinuous`` + ``scale_x_continuous`` / ``scale_y_continuous``.

matplotlib autoscale handles axis limits (so geoms with non-trivial
extents like bar widths still fit); the scale contributes breaks +
labels. User-supplied ``limits=`` overrides autoscale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .scale import _NAME_MISSING, Scale, format_breaks
from .transformed import IdentityTrans, Trans


@dataclass
class ScaleContinuous(Scale):
    transform: Trans = field(default_factory=IdentityTrans)
    range_: list | None = field(default=None, init=False, repr=False)

    def train(self, data) -> None:
        if data is None or len(data) == 0:
            return
        try:
            lo = float(data.min())
            hi = float(data.max())
        except (TypeError, ValueError):
            return
        if math.isnan(lo) or math.isnan(hi):
            return
        if self.range_ is None:
            self.range_ = [lo, hi]
        else:
            self.range_[0] = min(self.range_[0], lo)
            self.range_[1] = max(self.range_[1], hi)

    def apply_to_axis(self, ax, axis: str, view_limits=None) -> None:

        if view_limits is not None:
            if axis == "x":
                ax.set_xlim(view_limits)
            else:
                ax.set_ylim(view_limits)
        elif self.limits is not None:
            if axis == "x":
                ax.set_xlim(self.limits)
            else:
                ax.set_ylim(self.limits)
        else:
            self._apply_expansion(ax, axis)

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

        if isinstance(self.breaks, str) and self.breaks == "default":
            tick_spec = self.transform.tick_positions_and_labels(
                *(
                    view_limits
                    if view_limits is not None
                    else self._expanded_break_range()
                    if self.range_ is not None
                    else (ax.get_xlim() if axis == "x" else ax.get_ylim())
                )
            )
            if tick_spec is not None:
                positions, labels = tick_spec
                if axis == "x":
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                else:
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels)
                return

        if view_limits is not None:
            break_range = tuple(view_limits)
        elif self.range_ is not None:
            break_range = self._expanded_break_range()
        else:
            break_range = ax.get_xlim() if axis == "x" else ax.get_ylim()
        breaks = self._compute_breaks(break_range)
        labels = self._compute_labels(breaks)
        if self.transform.name != "identity":
            try:
                tick_positions = np.asarray(self.transform.transform(breaks))
            except Exception:  # noqa: BLE001
                tick_positions = np.asarray(breaks)
            mask = (tick_positions >= break_range[0]) & (
                tick_positions <= break_range[1]
            )
            tick_positions = tick_positions[mask]
            labels = [labels[i] for i in range(len(labels)) if mask[i]]
        else:
            mask = (np.asarray(breaks) >= break_range[0]) & (
                np.asarray(breaks) <= break_range[1]
            )
            tick_positions = np.asarray(breaks)[mask]
            labels = [labels[i] for i in range(len(labels)) if mask[i]]
        if axis == "x":
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(labels)

    def _apply_expansion(self, ax, axis: str) -> None:
        from ..expansion import Expansion

        exp = self.expand
        if isinstance(exp, Expansion):
            m_lo, m_hi, _a_lo, _a_hi = exp.split()
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

    def _expanded_break_range(self) -> tuple[float, float]:
        """Trained data range padded by this scale's ``expand`` factor."""
        from ..expansion import Expansion

        lo, hi = self.range_
        span = hi - lo
        exp = self.expand
        if isinstance(exp, Expansion):
            m_lo, m_hi, a_lo, a_hi = exp.split()
        elif isinstance(exp, (list, tuple)):
            mult = float(exp[0]) if len(exp) >= 1 else 0.0
            add = float(exp[1]) if len(exp) >= 2 else 0.0
            m_lo = m_hi = mult
            a_lo = a_hi = add
        else:
            return (lo, hi)
        return (lo - m_lo * span - a_lo, hi + m_hi * span + a_hi)

    def _compute_breaks(self, lim):
        if isinstance(self.breaks, str) and self.breaks == "default":
            from ._breaks import extended_breaks

            return extended_breaks(lim[0], lim[1], m=5)
        if callable(self.breaks):
            return np.asarray(self.breaks(lim))
        return np.asarray(self.breaks)

    def _compute_labels(self, breaks):
        if isinstance(self.labels, str) and self.labels == "default":
            return format_breaks(breaks)
        if self.labels is None:
            return ["" for _ in breaks]
        if callable(self.labels):
            return list(self.labels(breaks))
        if isinstance(self.labels, dict):
            return [str(self.labels.get(b, b)) for b in breaks]
        return [str(x) for x in self.labels]


def scale_x_continuous(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return ScaleContinuous(
        aesthetics=("x",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        expand=expand,
    )


def scale_y_continuous(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return ScaleContinuous(
        aesthetics=("y",),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        expand=expand,
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
