"""Render — walk per-layer drawable data into a matplotlib :class:`Figure`.

Phase 0 form: single panel, no theme, no facet, no real scales.
Each layer's geom is asked to draw its data onto one shared axes object.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def render(plot, build_output) -> "plt.Figure":
    fig, ax = plt.subplots()

    for layer, df in zip(plot.layers, build_output.data):
        layer.geom.draw_panel(df, ax)

    # Phase 0 axis labels: derive from the first layer that has x/y mappings.
    # Real label resolution (with `labs(...)`, deparse, etc.) is Phase 6.1.
    xlabel, ylabel = _default_labels(plot)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig


def _default_labels(plot):
    mapping = plot.mapping
    xlabel = mapping.get("x") if "x" in mapping else None
    ylabel = mapping.get("y") if "y" in mapping else None
    xlabel = xlabel if isinstance(xlabel, str) else None
    ylabel = ylabel if isinstance(ylabel, str) else None
    return xlabel, ylabel
