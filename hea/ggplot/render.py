"""Render — walk per-layer drawable data into a matplotlib :class:`Figure`.

Phase 0 form: single panel, no theme, no facet, no real scales.
Each layer's geom is asked to draw its data onto one shared axes object.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def render(plot, build_output, ax=None) -> "plt.Figure":
    if ax is None:
        fig, ax = plt.subplots()
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    for layer, df in zip(plot.layers, build_output.data):
        layer.geom.draw_panel(df, ax)

    # Phase 0 axis labels: derive from the first layer that has x/y mappings.
    # Real label resolution (with `labs(...)`, deparse, etc.) is Phase 6.1.
    xlabel, ylabel = _default_labels(plot)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Only manage layout for the figure we created. Caller-supplied axes
    # belong to a parent figure that's responsible for its own layout.
    if owns_fig:
        fig.tight_layout()
    return fig


def _default_labels(plot):
    mapping = plot.mapping
    xlabel = mapping.get("x") if "x" in mapping else None
    ylabel = mapping.get("y") if "y" in mapping else None
    xlabel = xlabel if isinstance(xlabel, str) else None
    ylabel = ylabel if isinstance(ylabel, str) else None
    if ylabel is None:
        # No user-mapped y → fall back to the first layer's stat default,
        # so histograms get "count" / density gets "density" without
        # needing labs() (matches ggplot2 deparsing of `after_stat(count)`).
        for layer in plot.layers:
            tag = getattr(layer.stat, "default_y_label", None)
            if tag:
                ylabel = tag
                break
    return xlabel, ylabel
