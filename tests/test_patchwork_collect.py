"""Tests for ``plot_layout(guides="collect")`` + ``guide_area()`` + ``&``.

These exercise the patchwork-style cross-leaf legend collection: dedup
across plots, ``guide_area()`` placement, ``show_legend=False`` honoured
during the merge, and ``& theme(...)`` broadcast to every leaf.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import hea
from hea.ggplot import (
    aes, geom_density, geom_boxplot, geom_point,
    guide_area, plot_annotation, plot_layout, theme,
)


@pytest.fixture
def mpg():
    return hea.tidy.DataFrame({
        "drv": ["f"]*5 + ["r"]*5 + ["4"]*5,
        "cty": [20.0, 22, 19, 21, 23, 14, 13, 15, 12, 16, 17, 18, 16, 19, 18],
        "hwy": [28.0, 30, 27, 29, 31, 22, 21, 23, 20, 24, 25, 26, 24, 27, 26],
    })


def _legend_axes(fig) -> list:
    """Axes that hold a collected merged legend (labelled by the renderer)."""
    return [ax for ax in fig.axes if ax.get_label() == "<merged-legend>"]


def test_amp_operator_broadcasts_theme_to_all_leaves(mpg):
    """``(p1 + p2) & theme(...)`` should propagate to every leaf, unlike
    ``+`` which only hits the rightmost leaf."""
    p1 = mpg.ggplot(aes(x="cty", y="hwy", color="drv")).geom_point()
    p2 = mpg.ggplot(aes(x="cty", color="drv", fill="drv")).geom_density(alpha=0.5)
    composition = (p1 + p2) & theme(legend_position="bottom")
    leaves = composition.leaves()
    assert len(leaves) == 2
    for leaf in leaves:
        assert leaf.theme.get("legend.position") == "bottom"


def test_amp_operator_skips_guide_area_placeholders(mpg):
    """``&`` should pass through ``guide_area()`` without trying to add
    the operand to it (it isn't a ggplot)."""
    p1 = mpg.ggplot(aes(x="cty", color="drv", fill="drv")).geom_density(alpha=0.5)
    composition = (guide_area() / p1) & theme(legend_position="top")
    # The guide_area still exists; the leaf has the theme applied.
    assert composition.find_guide_area() is not None
    leaves = composition.leaves()
    assert len(leaves) == 1
    assert leaves[0].theme.get("legend.position") == "top"


def test_plot_layout_guides_collect_dedups_legends(mpg):
    """Two plots mapping the same colour scale produce ONE merged legend
    when ``guides="collect"`` is in effect — not one per plot."""
    p1 = mpg.ggplot(aes(x="cty", color="drv", fill="drv")).geom_density(alpha=0.5)
    p2 = mpg.ggplot(aes(x="hwy", color="drv", fill="drv")).geom_density(alpha=0.5)
    composition = ((p1 + p2)
                   + plot_layout(guides="collect")) & theme(legend_position="top")
    fig = composition.draw()
    try:
        legends = _legend_axes(fig)
        assert len(legends) == 1, (
            f"expected 1 merged legend, got {len(legends)}"
        )
    finally:
        plt.close(fig)


def test_guide_area_receives_collected_legend(mpg):
    """An explicit ``guide_area()`` slot in the composition tree must host
    the merged legend (not auto-placed at the figure edge)."""
    p1 = mpg.ggplot(aes(x="cty", color="drv", fill="drv")).geom_density(alpha=0.5)
    p2 = mpg.ggplot(aes(x="hwy", color="drv", fill="drv")).geom_density(alpha=0.5)
    composition = ((guide_area() / (p1 + p2))
                   + plot_layout(guides="collect", heights=[1, 4]))
    fig = composition.draw()
    try:
        legends = _legend_axes(fig)
        assert len(legends) == 1
        # The guide_area is in row 0; the legend host axes y-position should
        # be ABOVE the panel axes (which are in row 1).
        legend_y = legends[0].get_position().y0
        panel_axes = [ax for ax in fig.axes
                      if ax.get_label() == "" and ax.get_position().width > 0.05]
        panel_top = max(ax.get_position().y1 for ax in panel_axes)
        assert legend_y >= panel_top - 0.01, (
            f"legend (y0={legend_y:.3f}) should sit above panels "
            f"(top={panel_top:.3f}) when guide_area is row 0"
        )
    finally:
        plt.close(fig)


def test_show_legend_false_excludes_layer_from_merge(mpg):
    """A layer's ``show_legend=False`` must keep it from contributing to
    the merged guide — same rule ggplot2 applies per-plot."""
    # p1 has show_legend=False on the boxplot (key_glyph="polygon");
    # p2 has show_legend=False on the geom_point (key_glyph="point").
    # p3 contributes via geom_density (key_glyph="polygon"). Without
    # show_legend handling, the merged legend would have two groups (one
    # per glyph). With it, only p3's polygon contributes.
    p1 = mpg.ggplot(aes(x="drv", y="cty", color="drv")).geom_boxplot(show_legend=False)
    p2 = mpg.ggplot(aes(x="cty", y="hwy", color="drv")).geom_point(show_legend=False)
    p3 = mpg.ggplot(aes(x="cty", color="drv", fill="drv")).geom_density(alpha=0.5)
    composition = ((guide_area() / (p1 + p2) / p3)
                   + plot_layout(guides="collect", heights=[1, 3, 4]))
    fig = composition.draw()
    try:
        legends = _legend_axes(fig)
        assert len(legends) == 1, (
            f"all show_legend=False layers should be excluded; "
            f"expected 1 merged legend (from p3), got {len(legends)}"
        )
    finally:
        plt.close(fig)


def test_auto_placement_when_no_guide_area(mpg):
    """``guides="collect"`` without an explicit ``guide_area()`` should
    auto-place the merged legend at the position dictated by the broadcast
    ``theme(legend.position)`` — not silently drop it."""
    p1 = mpg.ggplot(aes(x="cty", color="drv", fill="drv")).geom_density(alpha=0.5)
    p2 = mpg.ggplot(aes(x="hwy", color="drv", fill="drv")).geom_density(alpha=0.5)
    composition = ((p1 + p2)
                   + plot_layout(guides="collect")) & theme(legend_position="top")
    fig = composition.draw()
    try:
        legends = _legend_axes(fig)
        assert len(legends) == 1
        # Top auto-placement: legend should be ABOVE the panels.
        legend_y = legends[0].get_position().y0
        panel_axes = [ax for ax in fig.axes
                      if ax.get_label() == "" and ax.get_position().width > 0.05]
        panel_top = max(ax.get_position().y1 for ax in panel_axes)
        assert legend_y >= panel_top - 0.01
    finally:
        plt.close(fig)


def test_plot_layout_guides_validation():
    """``plot_layout(guides=)`` rejects unknown values."""
    with pytest.raises(ValueError, match="guides must be"):
        plot_layout(guides="bogus")
