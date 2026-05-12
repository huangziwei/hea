"""``labs`` / ``xlab`` / ``ylab`` / ``ggtitle`` — set axis, title, and guide labels.

ggplot2-faithful API. ``labs(...)`` returns a :class:`Labels` whose ``+`` into
a ggplot merges into ``plot.labels`` (handled in ``core.py:ggplot_add``).

The renderer reads ``x``, ``y``, ``title``, ``subtitle``, ``caption`` for
output. Aesthetic-guide titles (``colour``/``fill``/``size``/...) are accepted
and stored, ready for guide infrastructure to pick up — currently a no-op.

Passing ``None`` explicitly **suppresses** the auto-derived label (matches
ggplot2's ``labs(x = NULL)``). Omitting the kwarg leaves the auto-derived
default in place. The distinction lives in the ``_LABEL_MISSING`` sentinel
below: without it, ``None`` would be indistinguishable from "user didn't
pass this kwarg" and the suppression form would be silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Sentinel for "user didn't pass this kwarg." ``None`` itself is a valid
# value meaning "explicitly suppress this label."
_LABEL_MISSING: Any = object()


@dataclass
class Labels:
    labels: dict = field(default_factory=dict)


def labs(
    *,
    x=_LABEL_MISSING,
    y=_LABEL_MISSING,
    title=_LABEL_MISSING,
    subtitle=_LABEL_MISSING,
    caption=_LABEL_MISSING,
    tag=_LABEL_MISSING,
    color=_LABEL_MISSING,
    colour=_LABEL_MISSING,
    fill=_LABEL_MISSING,
    alpha=_LABEL_MISSING,
    size=_LABEL_MISSING,
    shape=_LABEL_MISSING,
    linetype=_LABEL_MISSING,
    **kwargs,
):
    """Set labels for axes, title/subtitle/caption/tag, and aesthetic guides.

    Each keyword corresponds to a ggplot2 label slot. ``color`` and ``colour``
    are aliased to the canonical ``"colour"`` key (matching ggplot2's US/UK
    spelling treatment). Any extra kwargs are passed through unchanged so
    custom-named aesthetics can be labelled too.

    Pass ``None`` to explicitly suppress an auto-derived label (e.g.
    ``labs(x=None, y=None)`` on a polar plot to drop the axis titles).
    """
    out = {}
    if x is not _LABEL_MISSING:
        out["x"] = x
    if y is not _LABEL_MISSING:
        out["y"] = y
    if title is not _LABEL_MISSING:
        out["title"] = title
    if subtitle is not _LABEL_MISSING:
        out["subtitle"] = subtitle
    if caption is not _LABEL_MISSING:
        out["caption"] = caption
    if tag is not _LABEL_MISSING:
        out["tag"] = tag
    # ``color`` and ``colour`` both map to canonical ``"colour"``. When
    # both are supplied, ``color`` (assigned second) wins — matches the
    # existing "last assignment wins" semantics asserted in tests.
    if colour is not _LABEL_MISSING:
        out["colour"] = colour
    if color is not _LABEL_MISSING:
        out["colour"] = color
    if fill is not _LABEL_MISSING:
        out["fill"] = fill
    if alpha is not _LABEL_MISSING:
        out["alpha"] = alpha
    if size is not _LABEL_MISSING:
        out["size"] = size
    if shape is not _LABEL_MISSING:
        out["shape"] = shape
    if linetype is not _LABEL_MISSING:
        out["linetype"] = linetype
    for k, v in kwargs.items():
        out[k] = v
    return Labels(labels=out)


def xlab(label):
    """Sugar for ``labs(x=label)``. Pass ``None`` to suppress."""
    return labs(x=label)


def ylab(label):
    """Sugar for ``labs(y=label)``. Pass ``None`` to suppress."""
    return labs(y=label)


def ggtitle(label, subtitle=_LABEL_MISSING):
    """Sugar for ``labs(title=label, subtitle=subtitle)``."""
    return labs(title=label, subtitle=subtitle)
