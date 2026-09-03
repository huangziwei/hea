"""Axis transformations — ``Trans`` objects + ``scale_x/y_*`` factories.

A ``Trans`` describes the data ↔ display mapping for a non-linear axis.
ggplot2's pipeline applies the transform to *data* before any stat runs
— so binning, smoothing, etc. operate on transformed values and produce
visibly different output than running them on raw data. We mirror that:
:meth:`Trans.transform` is invoked in the build pipeline (build.py)
right after aesthetics resolve and before ``stat.compute_layer``.

The axis itself stays linear at the matplotlib level (data is already
transformed); :meth:`tick_positions_and_labels` provides nice ticks at
original-units values mapped through ``transform`` so the user still
sees ``"10"``/``"100"``/``"1000"`` etc. on a log axis.

For backwards compatibility with older callers, :meth:`matplotlib_scale`
still returns the matplotlib scale name — but since the data is
pre-transformed, the scale stays ``None`` (linear). Subclasses that
need a non-trivial display behaviour (e.g. :class:`ReverseTrans`)
implement it via :meth:`reversed` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scale import _NAME_MISSING


@dataclass
class Trans:
    """Base — identity by default. Subclasses override :meth:`transform`
    and :meth:`inverse` and (optionally) :meth:`tick_positions_and_labels`."""

    name: str = "identity"

    def transform(self, x):
        """Map raw values to transformed (display) space. Identity by default."""
        return np.asarray(x, dtype=float)

    def inverse(self, x):
        """Map transformed values back to raw space."""
        return np.asarray(x, dtype=float)

    def matplotlib_scale(self):
        """No matplotlib scale name: the data is pre-transformed, so the
        axis stays linear. Present for callers that inspect it."""
        return

    def reversed(self) -> bool:
        """True if the axis should display high-to-low (e.g. ``ReverseTrans``)."""
        return False

    def tick_positions_and_labels(self, lo: float, hi: float):
        """Return ``(positions, labels)`` for nice ticks across the
        *transformed* range ``[lo, hi]``. Default: defer (returns ``None``)
        so the scale's linear tick logic runs."""
        return


class IdentityTrans(Trans):
    def __init__(self):
        super().__init__(name="identity")


class Log10Trans(Trans):
    def __init__(self):
        super().__init__(name="log-10")

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(arr > 0, np.log10(arr), np.nan)

    def inverse(self, x):
        return 10.0 ** np.asarray(x, dtype=float)

    def matplotlib_scale(self):
        return ("log", {"base": 10})

    def tick_positions_and_labels(self, lo: float, hi: float):
        return _log_breaks(lo, hi, base=10.0)


class Log2Trans(Trans):
    def __init__(self):
        super().__init__(name="log-2")

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(arr > 0, np.log2(arr), np.nan)

    def inverse(self, x):
        return 2.0 ** np.asarray(x, dtype=float)

    def matplotlib_scale(self):
        return ("log", {"base": 2})

    def tick_positions_and_labels(self, lo: float, hi: float):
        return _log_breaks(lo, hi, base=2.0)


class SqrtTrans(Trans):
    def __init__(self):
        super().__init__(name="sqrt")

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        return np.sqrt(np.where(arr >= 0, arr, np.nan))

    def inverse(self, x):
        return np.asarray(x, dtype=float) ** 2

    def matplotlib_scale(self):
        forward = lambda x: np.sqrt(np.maximum(x, 0))
        inverse = lambda x: x**2
        return ("function", {"functions": (forward, inverse)})

    def tick_positions_and_labels(self, lo: float, hi: float):
        """Nice raw-unit breaks at sqrt-spaced positions.

        ``lo`` and ``hi`` arrive in **transformed (sqrt)** space (the
        scale's trained range is in transformed space because data is
        pre-transformed in build). We inverse-map to raw, ask
        matplotlib's locator for nice breaks there, then forward-map
        back to display positions and format the labels in raw units.

        Without this, the fall-through path in
        ``ScaleContinuous.apply_to_axis`` computes nice breaks across
        the transformed range and labels them with their *transformed*
        values — e.g. a count axis would show ``"30", "60", "90", "120"``
        (sqrt-units) instead of ``"1000", "5000", "10000"``.
        """
        import matplotlib.ticker as _mt

        from .scale import format_breaks

        raw_lo = max(0.0, float(lo)) ** 2
        raw_hi = max(0.0, float(hi)) ** 2
        if raw_hi <= raw_lo:
            return None
        locator = _mt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
        raw_breaks = [
            float(b)
            for b in locator.tick_values(raw_lo, raw_hi)
            if raw_lo <= b <= raw_hi and b >= 0
        ]
        if not raw_breaks:
            return None
        positions = [b**0.5 for b in raw_breaks]
        labels = format_breaks(raw_breaks)
        return positions, labels


class ReverseTrans(Trans):
    """Visual reverse: data values aren't transformed (so stats run on
    raw values just fine). The :meth:`reversed` flag tells the scale's
    ``apply_to_axis`` to flip ``xlim``/``ylim`` after limits are set."""

    def __init__(self):
        super().__init__(name="reverse")

    def reversed(self) -> bool:
        return True


def _log_breaks(lo: float, hi: float, *, base: float) -> tuple:
    """Generate ``(positions, labels)`` for a log-base axis spanning
    transformed range ``[lo, hi]``.
    """
    raw_lo = base**lo
    raw_hi = base**hi
    log_b = np.log(base)
    k_lo = int(np.floor(np.log(raw_lo) / log_b))
    k_hi = int(np.ceil(np.log(raw_hi) / log_b))
    decades = k_hi - k_lo
    if decades <= 3:
        candidates = []
        for k in range(k_lo, k_hi + 1):
            for m in (1, 2, 5):
                v = m * (base**k)
                if raw_lo - 1e-12 <= v <= raw_hi * (1 + 1e-12):
                    candidates.append(v)
    else:
        candidates = [
            base**k
            for k in range(k_lo, k_hi + 1)
            if raw_lo - 1e-12 <= base**k <= raw_hi * (1 + 1e-12)
        ]
    positions = [np.log(v) / log_b for v in candidates]
    labels = [_format_log_tick(v) for v in candidates]
    return positions, labels


def _format_log_tick(value: float) -> str:
    """Render a log-scale tick value like ggplot2: integers as plain
    digits, fractions like ``0.1`` for log10, scientific for very
    large/small."""
    if value <= 0:
        return ""
    abs_v = abs(value)
    if abs_v >= 10000 or (0 < abs_v < 0.01):
        exp = round(np.log10(abs_v))
        return f"$10^{{{exp}}}$"
    if abs_v >= 1:
        return f"{round(value)}"
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return s


def _scale_factory(aes_name: str, trans: Trans, name, breaks, labels, limits, expand):
    from .continuous import ScaleContinuous

    return ScaleContinuous(
        aesthetics=(aes_name,),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        expand=expand,
        transform=trans,
    )


def scale_x_log10(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("x", Log10Trans(), name, breaks, labels, limits, expand)


def scale_y_log10(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("y", Log10Trans(), name, breaks, labels, limits, expand)


def scale_x_log2(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("x", Log2Trans(), name, breaks, labels, limits, expand)


def scale_y_log2(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("y", Log2Trans(), name, breaks, labels, limits, expand)


def scale_x_sqrt(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("x", SqrtTrans(), name, breaks, labels, limits, expand)


def scale_y_sqrt(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("y", SqrtTrans(), name, breaks, labels, limits, expand)


def scale_x_reverse(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("x", ReverseTrans(), name, breaks, labels, limits, expand)


def scale_y_reverse(
    *,
    name=_NAME_MISSING,
    breaks="default",
    labels="default",
    limits=None,
    expand=(0.05, 0.0),
):
    return _scale_factory("y", ReverseTrans(), name, breaks, labels, limits, expand)
