"""Render — walk per-layer drawable data into a matplotlib :class:`Figure`.

Single-panel and faceted modes share the per-axes drawing logic; faceted
mode adds a subplot grid plus per-panel data filtering. ``scales="free*"``
modes mean each panel autoscales independently (matplotlib's
``sharex``/``sharey``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl


def render(plot, build_output, ax=None) -> "plt.Figure":
    layout = build_output.layout
    n_panels = 1 if layout is None else len(layout)

    if n_panels <= 1 or ax is not None:
        return _render_single(plot, build_output, ax)
    return _render_facets(plot, build_output, layout)


def _render_single(plot, build_output, ax):
    if ax is None:
        fig, ax = plt.subplots()
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    for layer, df in zip(plot.layers, build_output.data):
        layer.geom.draw_panel(df, ax)

    if build_output.scales is not None:
        for axis in ("x", "y"):
            sc = build_output.scales.get(axis)
            if sc is not None:
                sc.apply_to_axis(ax, axis)

    xlabel, ylabel = _default_labels(plot)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if owns_fig:
        fig.tight_layout()
    return fig


def _render_facets(plot, build_output, layout):
    facet = plot.facet
    n_panels = len(layout)
    nrow, ncol = facet.grid_dims(n_panels)

    sharex = facet.scales in ("fixed", "free_y")
    sharey = facet.scales in ("fixed", "free_x")

    fig, axes = plt.subplots(
        nrow, ncol,
        sharex=sharex,
        sharey=sharey,
        figsize=(3.0 * ncol, 2.5 * nrow),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for panel_row in layout.iter_rows(named=True):
        idx = panel_row["PANEL"] - 1
        panel_ax = flat_axes[idx]

        for layer, df in zip(plot.layers, build_output.data):
            if "PANEL" not in df.columns:
                panel_data = df
            else:
                panel_data = df.filter(pl.col("PANEL") == panel_row["PANEL"])
            if len(panel_data) > 0:
                layer.geom.draw_panel(panel_data, panel_ax)

        # Apply positional scales per axis. With sharex/sharey, matplotlib
        # propagates limits across the shared axes, so calling apply_to_axis
        # on each panel is consistent for "fixed" and gives independent
        # ticks for "free*".
        if build_output.scales is not None:
            for axis in ("x", "y"):
                sc = build_output.scales.get(axis)
                if sc is not None:
                    sc.apply_to_axis(panel_ax, axis)

        # Strip label = facet variable values (joined with ", " for multi-facet).
        strip_text = ", ".join(
            f"{panel_row[v]}" for v in facet.facet_vars()
            if v in panel_row
        )
        if strip_text:
            panel_ax.set_title(strip_text)

    # Hide unused panels (when the grid has more cells than panels).
    for unused_ax in flat_axes[n_panels:]:
        unused_ax.set_visible(False)

    # Common axis labels — set on the figure rather than per-panel so they
    # land in the canonical "outer edge only" position.
    xlabel, ylabel = _default_labels(plot)
    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel is not None:
        fig.supylabel(ylabel)

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
