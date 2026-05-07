"""``ScaleContinuous`` + ``scale_x_continuous`` / ``scale_y_continuous``.

Phase 1.1 form: matplotlib autoscale handles axis limits (so geoms with
non-trivial extents like bar widths still fit); the scale contributes
breaks + labels. User-supplied ``limits=`` overrides autoscale. Wilkinson
``extended_breaks`` parity is checklist 1.1c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scale import Scale, fmt_number, format_breaks
from .transformed import IdentityTrans, Trans


@dataclass
class ScaleContinuous(Scale):
    transform: Trans = field(default_factory=IdentityTrans)
    # Trained data range — used for break computation so ticks reflect
    # *data* extent, not the (expanded) axis view limit. Without this,
    # bars at gear ∈ {3, 4, 5} get axis xlim ≈ [2, 6] (bar widths +
    # margins), and matplotlib's auto-locator yields ticks at 2..6
    # rather than R's preferred 3..5.
    range_: list | None = field(default=None, init=False, repr=False)

    def train(self, data) -> None:
        if data is None or len(data) == 0:
            return
        try:
            lo = float(data.min())
            hi = float(data.max())
        except (TypeError, ValueError):
            return
        if not (lo == lo and hi == hi):  # NaN check
            return
        if self.range_ is None:
            self.range_ = [lo, hi]
        else:
            self.range_[0] = min(self.range_[0], lo)
            self.range_[1] = max(self.range_[1], hi)

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

        # Compute breaks against the EXPANDED data range — matches
        # ggplot2's ``scales::breaks_extended``, which works on the
        # post-expansion view limits, not the raw data range. Without
        # this, a density y in ``[4e-5, 1.1e-3]`` gets breaks at
        # ``2.5e-4`` increments instead of ``3e-4``, and the labels
        # don't switch to scientific the way R does. We then trim
        # breaks back inside the expanded range so out-of-view ticks
        # don't get drawn as labels.
        #
        # We DON'T fall back to ``ax.get_xlim()`` here even when
        # untrained — matplotlib's autoscaled view bakes in artist
        # extents (bar widths, ribbon padding) that would push the
        # break range past the data, e.g. bars at ``gear ∈ {3, 4, 5}``
        # would yield ``[2, 3, 4, 5, 6]`` instead of the expected
        # ``[3, 4, 5]``.
        if self.range_ is not None:
            break_range = self._expanded_break_range()
        else:
            break_range = ax.get_xlim() if axis == "x" else ax.get_ylim()
        breaks = self._compute_breaks(break_range)
        breaks = np.asarray(
            [b for b in breaks if break_range[0] <= b <= break_range[1]]
        )
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

    def _expanded_break_range(self) -> tuple[float, float]:
        """Trained data range padded by this scale's ``expand`` factor.

        Mirrors ggplot2's call to ``breaks_extended`` on the expanded
        view limits. Reads ``expand`` in either ``Expansion`` or legacy
        ``(mult, add)`` form.
        """
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
        if self.breaks == "default":
            from ._breaks import extended_breaks

            return extended_breaks(lim[0], lim[1], m=5)
        if callable(self.breaks):
            return np.asarray(self.breaks(lim))
        return np.asarray(self.breaks)

    def _compute_labels(self, breaks):
        if self.labels == "default":
            # Per-axis (vector) format choice: scientific only when its
            # max width strictly beats fixed, matching R's ``format()``.
            return format_breaks(breaks)
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
