"""``labs`` / ``xlab`` / ``ylab`` / ``ggtitle`` — set axis, title, and guide labels.

ggplot2-faithful API. ``labs(...)`` returns a :class:`Labels` whose ``+`` into
a ggplot merges into ``plot.labels`` (handled in ``core.py:ggplot_add``).

The renderer reads ``x``, ``y``, ``title``, ``subtitle``, ``caption`` for
output. Aesthetic-guide titles (``colour``/``fill``/``size``/...) are accepted
and stored, ready for guide infrastructure to pick up — currently a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Labels:
    labels: dict = field(default_factory=dict)


def labs(
    *,
    x=None,
    y=None,
    title=None,
    subtitle=None,
    caption=None,
    tag=None,
    color=None,
    colour=None,
    fill=None,
    alpha=None,
    size=None,
    shape=None,
    linetype=None,
    **kwargs,
):
    """Set labels for axes, title/subtitle/caption/tag, and aesthetic guides.

    Each keyword corresponds to a ggplot2 label slot. ``color`` and ``colour``
    are aliased to the canonical ``"colour"`` key (matching ggplot2's US/UK
    spelling treatment). Any extra kwargs are passed through unchanged so
    custom-named aesthetics can be labelled too.
    """
    out = {}
    if x is not None:
        out["x"] = x
    if y is not None:
        out["y"] = y
    if title is not None:
        out["title"] = title
    if subtitle is not None:
        out["subtitle"] = subtitle
    if caption is not None:
        out["caption"] = caption
    if tag is not None:
        out["tag"] = tag
    if colour is not None:
        out["colour"] = colour
    if color is not None:
        out["colour"] = color
    if fill is not None:
        out["fill"] = fill
    if alpha is not None:
        out["alpha"] = alpha
    if size is not None:
        out["size"] = size
    if shape is not None:
        out["shape"] = shape
    if linetype is not None:
        out["linetype"] = linetype
    for k, v in kwargs.items():
        if v is not None:
            out[k] = v
    return Labels(labels=out)


def xlab(label):
    """Sugar for ``labs(x=label)``."""
    return labs(x=label)


def ylab(label):
    """Sugar for ``labs(y=label)``."""
    return labs(y=label)


def ggtitle(label, subtitle=None):
    """Sugar for ``labs(title=label, subtitle=subtitle)``."""
    return labs(title=label, subtitle=subtitle)
