"""Axis transformations — ``Trans`` objects + ``scale_x/y_*`` factories.

A ``Trans`` describes how to map raw data values onto a non-linear axis.
We delegate the actual rendering to matplotlib's axis-scale system
(``ax.set_xscale("log", ...)`` / ``"function"``) — geoms keep drawing in
raw data coordinates, matplotlib transforms display coords at render time.
This is closer to matplotlib's idiom and avoids re-implementing
log/sqrt rendering ourselves.

ggplot2 differs: it transforms data before plotting because R's grid has
no axis-side transform. The visible result is the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trans:
    """Base — identity by default. Override :meth:`matplotlib_scale` and/or
    :meth:`reversed`."""

    name: str = "identity"

    def matplotlib_scale(self):
        """Return ``(scale_name, kwargs)`` for ``ax.set_xscale(scale_name, **kwargs)``,
        or ``None`` for linear (the matplotlib default)."""
        return None

    def reversed(self) -> bool:
        """True if the axis should display high-to-low (e.g. ``ReverseTrans``)."""
        return False


class IdentityTrans(Trans):
    def __init__(self):
        super().__init__(name="identity")


class Log10Trans(Trans):
    def __init__(self):
        super().__init__(name="log-10")

    def matplotlib_scale(self):
        return ("log", {"base": 10})


class Log2Trans(Trans):
    def __init__(self):
        super().__init__(name="log-2")

    def matplotlib_scale(self):
        return ("log", {"base": 2})


class SqrtTrans(Trans):
    def __init__(self):
        super().__init__(name="sqrt")

    def matplotlib_scale(self):
        # matplotlib's FuncScale (since 3.1). Negative values become NaN at
        # render time, matching ggplot2's behaviour ("removed N rows").
        forward = lambda x: np.sqrt(np.maximum(x, 0))  # noqa: E731
        inverse = lambda x: x ** 2  # noqa: E731
        return ("function", {"functions": (forward, inverse)})


class ReverseTrans(Trans):
    def __init__(self):
        super().__init__(name="reverse")

    def reversed(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _scale_factory(aes_name: str, trans: Trans, name, breaks, labels, limits, expand):
    from .continuous import ScaleContinuous
    return ScaleContinuous(
        aesthetics=(aes_name,), name=name, breaks=breaks, labels=labels,
        limits=limits, expand=expand, transform=trans,
    )


def scale_x_log10(*, name=None, breaks="default", labels="default",
                  limits=None, expand=(0.05, 0.0)):
    return _scale_factory("x", Log10Trans(), name, breaks, labels, limits, expand)


def scale_y_log10(*, name=None, breaks="default", labels="default",
                  limits=None, expand=(0.05, 0.0)):
    return _scale_factory("y", Log10Trans(), name, breaks, labels, limits, expand)


def scale_x_log2(*, name=None, breaks="default", labels="default",
                 limits=None, expand=(0.05, 0.0)):
    return _scale_factory("x", Log2Trans(), name, breaks, labels, limits, expand)


def scale_y_log2(*, name=None, breaks="default", labels="default",
                 limits=None, expand=(0.05, 0.0)):
    return _scale_factory("y", Log2Trans(), name, breaks, labels, limits, expand)


def scale_x_sqrt(*, name=None, breaks="default", labels="default",
                 limits=None, expand=(0.05, 0.0)):
    return _scale_factory("x", SqrtTrans(), name, breaks, labels, limits, expand)


def scale_y_sqrt(*, name=None, breaks="default", labels="default",
                 limits=None, expand=(0.05, 0.0)):
    return _scale_factory("y", SqrtTrans(), name, breaks, labels, limits, expand)


def scale_x_reverse(*, name=None, breaks="default", labels="default",
                    limits=None, expand=(0.05, 0.0)):
    return _scale_factory("x", ReverseTrans(), name, breaks, labels, limits, expand)


def scale_y_reverse(*, name=None, breaks="default", labels="default",
                    limits=None, expand=(0.05, 0.0)):
    return _scale_factory("y", ReverseTrans(), name, breaks, labels, limits, expand)
