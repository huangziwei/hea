"""``ScaleContinuous`` + ``scale_x_continuous`` / ``scale_y_continuous``.

Phase 1.1 form: matplotlib autoscale handles axis limits (so geoms with
non-trivial extents like bar widths still fit); the scale contributes
breaks + labels. User-supplied ``limits=`` overrides autoscale. Wilkinson
``extended_breaks`` parity is checklist 1.1c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scale import Scale, _NAME_MISSING, format_breaks
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

    def apply_to_axis(self, ax, axis: str, view_limits=None) -> None:
        # The matplotlib axis stays LINEAR for ``ScaleContinuous``.
        # ``scale_x_log10()`` etc. pre-transform the data in build.py
        # (matches ggplot2 — stat sees transformed values), so calling
        # ``set_xscale("log")`` here would log a second time and break
        # the display. ``Trans.matplotlib_scale()`` is reserved for
        # ``coord_trans()`` (display-only transform, data untouched).

        if view_limits is not None:
            # ``coord_cartesian(xlim=/ylim=)`` zoom — set limits to match
            # the coord view so break filtering further down sees the
            # right window. Coord's ``apply_to_axes`` will run again
            # later but is now a no-op (idempotent set_xlim with the
            # same value).
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

        # When a non-linear transform is in play and the user didn't ask
        # for explicit breaks, ask the transform for nice ticks at
        # original-units values (e.g. 10/100/1000 for log10) mapped to
        # the transformed positions of the data. The matplotlib axis is
        # linear (data is pre-transformed in build); we just place ticks
        # at the right positions and label them with the inverse-mapped
        # values. Without this, a log10-transformed axis would get
        # MaxNLocator's linear ticks (e.g. 1, 2, 3 on a log axis labeled
        # as 10, 100, 1000) — wrong.
        if isinstance(self.breaks, str) and self.breaks == "default":
            tick_spec = self.transform.tick_positions_and_labels(
                *(view_limits if view_limits is not None
                  else self._expanded_break_range()
                  if self.range_ is not None
                  else (ax.get_xlim() if axis == "x" else ax.get_ylim()))
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
        if view_limits is not None:
            # Compute breaks against the coord-zoomed view so e.g.
            # ``coord_cartesian(ylim=(0, 50))`` on a histogram whose
            # raw counts reach ~10k yields ticks at 0/10/.../50, not
            # 0/2500/.../10000 (almost all of which would land outside
            # the visible window and disappear).
            break_range = tuple(view_limits)
        elif self.range_ is not None:
            break_range = self._expanded_break_range()
        else:
            break_range = ax.get_xlim() if axis == "x" else ax.get_ylim()
        breaks = self._compute_breaks(break_range)
        # Labels reflect the user-supplied (raw-units) break values, not
        # the transformed positions — so ``breaks=[100,200,400]`` on a
        # ``scale_x_log10()`` axis still labels as 100/200/400 even
        # though the underlying tick positions are 2/2.30/2.60.
        labels = self._compute_labels(breaks)
        # When the data was pre-transformed (scale_x_log10 etc.), the
        # axis lives in transformed space. User-supplied breaks are in
        # raw units, so map them through ``transform`` before placing
        # on the axis. The labels stay in raw units (above).
        if self.transform.name != "identity":
            try:
                tick_positions = np.asarray(self.transform.transform(breaks))
            except Exception:
                tick_positions = np.asarray(breaks)
            mask = (tick_positions >= break_range[0]) & (tick_positions <= break_range[1])
            tick_positions = tick_positions[mask]
            labels = [labels[i] for i in range(len(labels)) if mask[i]]
        else:
            mask = (np.asarray(breaks) >= break_range[0]) & (np.asarray(breaks) <= break_range[1])
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
        if isinstance(self.breaks, str) and self.breaks == "default":
            from ._breaks import extended_breaks

            return extended_breaks(lim[0], lim[1], m=5)
        if callable(self.breaks):
            return np.asarray(self.breaks(lim))
        return np.asarray(self.breaks)

    def _compute_labels(self, breaks):
        if isinstance(self.labels, str) and self.labels == "default":
            # Per-axis (vector) format choice: scientific only when its
            # max width strictly beats fixed, matching R's ``format()``.
            return format_breaks(breaks)
        if self.labels is None:
            # ggplot2 / R: ``labels = NULL`` suppresses tick labels.
            return ["" for _ in breaks]
        if callable(self.labels):
            return list(self.labels(breaks))
        if isinstance(self.labels, dict):
            # Dict lookup keyed by the break value — falls back to the
            # break itself when missing (matches ggplot2's named-vector
            # treatment in ``scale_*_continuous(labels = c(...))``).
            return [str(self.labels.get(b, b)) for b in breaks]
        return [str(x) for x in self.labels]


def scale_x_continuous(*, name=_NAME_MISSING, breaks="default", labels="default",
                       limits=None, expand=(0.05, 0.0)):
    return ScaleContinuous(
        aesthetics=("x",), name=name, breaks=breaks, labels=labels,
        limits=limits, expand=expand,
    )


def scale_y_continuous(*, name=_NAME_MISSING, breaks="default", labels="default",
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
