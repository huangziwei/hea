"""Pin behaviour of `hea.ggplot._block` — the per-plot block engine.

Two kinds of tests here:

* **Measurement** — ``measure_block`` returns sensible margin sizes given
  the plot's labels/scales. We don't pin exact inches (font-dependent);
  we pin invariants like "title set → margin_top > 0".

* **Layout** — after ``render_block``, the panel ``Axes``' figure-relative
  bbox matches the block's margin allocation, so composition (Phase C)
  can rely on it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import pytest

from hea.ggplot import (
    aes, facet_wrap, geom_bar, geom_point, ggplot, labs,
)
from hea.ggplot._block import (
    PlotBlock, default_figsize_for, measure_block, render_block,
)
from hea.ggplot.build import build


# ----- Measurement --------------------------------------------------------

def _simple_plot(**labs_kw) -> ggplot:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    p = ggplot(df, aes("x", "y")) + geom_point()
    if labs_kw:
        p = p + labs(**labs_kw)
    return p


def test_measure_no_labels_has_minimal_top_margin():
    """A plot with no title should have margin_top = 0."""
    p = _simple_plot()
    block = measure_block(p, build(p))
    assert block.margin_top_in == 0.0


def test_measure_title_grows_top_margin():
    p_no = _simple_plot()
    p_yes = _simple_plot(title="A title")
    a = measure_block(p_no, build(p_no))
    b = measure_block(p_yes, build(p_yes))
    assert b.margin_top_in > a.margin_top_in


def test_measure_subtitle_grows_top_margin_more():
    p_t = _simple_plot(title="A title")
    p_ts = _simple_plot(title="A title", subtitle="A subtitle")
    a = measure_block(p_t, build(p_t))
    b = measure_block(p_ts, build(p_ts))
    assert b.margin_top_in > a.margin_top_in


def test_measure_caption_grows_bottom_margin():
    p_no = _simple_plot()
    p_yes = _simple_plot(caption="Source: nowhere")
    a = measure_block(p_no, build(p_no))
    b = measure_block(p_yes, build(p_yes))
    assert b.margin_bottom_in > a.margin_bottom_in


def test_measure_ylab_grows_left_margin():
    p_no = _simple_plot()
    p_yes = _simple_plot(y="The y-axis values")
    a = measure_block(p_no, build(p_no))
    b = measure_block(p_yes, build(p_yes))
    # y axis title takes some left margin (rotated 90°).
    assert b.margin_left_in > a.margin_left_in


def test_measure_blank_xlab_shrinks_bottom_margin():
    """Empty xlab string skips the xlab band of the bottom margin."""
    p_default = _simple_plot()  # default xlabel = column name "x"
    p_blank = _simple_plot(x="")
    a = measure_block(p_default, build(p_default))
    b = measure_block(p_blank, build(p_blank))
    # Default plot's xlabel "x" has nonzero height; blank string drops it.
    assert a.margin_bottom_in > b.margin_bottom_in


def test_measure_facet_wrap_panel_grid_dims():
    """A facet_wrap plot's block reports the facet grid dims."""
    df = pl.DataFrame({"g": ["a", "b", "c"] * 4, "x": list(range(12)), "y": list(range(12))})
    p = ggplot(df, aes("x", "y")) + geom_point() + facet_wrap("g")
    block = measure_block(p, build(p))
    # facet_wrap on 3 unique groups → wrap_dims gives (1, 3).
    assert (block.panel_grid_rows, block.panel_grid_cols) == (1, 3)


def test_measure_no_facet_panel_grid_is_one_one():
    p = _simple_plot()
    block = measure_block(p, build(p))
    assert (block.panel_grid_rows, block.panel_grid_cols) == (1, 1)


# ----- Layout / render ----------------------------------------------------

def test_render_block_panel_bbox_matches_margins():
    """After render, panel's figure-relative bbox respects block margins."""
    p = _simple_plot(title="t", x="xlab", y="ylab", caption="caption")
    bo = build(p)
    block = measure_block(p, bo)
    fig_w, fig_h = default_figsize_for(block)
    fig = plt.figure(figsize=(fig_w, fig_h))
    try:
        render_block(p, bo, block, fig=fig)
        ax = block.panel_axes[0]
        # Force a draw so set_position is final (matplotlib's auto-layout
        # may adjust before then).
        fig.canvas.draw()
        bbox = ax.get_position()  # in figure-relative coords

        expected_left = block.margin_left_in / fig_w
        expected_bottom = block.margin_bottom_in / fig_h
        expected_right = 1 - block.margin_right_in / fig_w
        expected_top = 1 - block.margin_top_in / fig_h

        assert bbox.x0 == pytest.approx(expected_left, abs=1e-3)
        assert bbox.y0 == pytest.approx(expected_bottom, abs=1e-3)
        assert bbox.x1 == pytest.approx(expected_right, abs=1e-3)
        assert bbox.y1 == pytest.approx(expected_top, abs=1e-3)
    finally:
        plt.close(fig)


def test_render_block_facet_creates_n_panel_axes():
    df = pl.DataFrame({"g": ["a", "b", "c"] * 4, "x": list(range(12)), "y": list(range(12))})
    p = ggplot(df, aes("x", "y")) + geom_point() + facet_wrap("g")
    bo = build(p)
    block = measure_block(p, bo)
    fig_w, fig_h = default_figsize_for(block)
    fig = plt.figure(figsize=(fig_w, fig_h))
    try:
        render_block(p, bo, block, fig=fig)
        assert len(block.panel_axes) == 3
        # All panel axes should sit inside the panel cell (between left
        # and right margins).
        fig.canvas.draw()
        for ax in block.panel_axes:
            bbox = ax.get_position()
            assert bbox.x0 >= block.margin_left_in / fig_w - 1e-3
            assert bbox.x1 <= 1 - block.margin_right_in / fig_w + 1e-3


    finally:
        plt.close(fig)


def test_draw_uses_block_engine_no_extra_axes():
    """Standalone draw() returns a fig with only panel + (optional) colorbar
    axes — no decoration axes from the block infrastructure."""
    p = _simple_plot(title="title", caption="caption")
    fig = p.draw()
    try:
        # Title/caption are rendered as fig.text / ax.set_title — they
        # don't add to fig.axes. So a single-panel plot with title+caption
        # has exactly 1 axes.
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_draw_respects_user_figsize():
    p = _simple_plot()
    fig = p.draw(figsize=(8, 5))
    try:
        assert fig.get_figwidth() == pytest.approx(8.0)
        assert fig.get_figheight() == pytest.approx(5.0)
    finally:
        plt.close(fig)


def test_draw_respects_user_width_height_in_cm():
    p = _simple_plot()
    fig = p.draw(width=20, height=15, units="cm")
    try:
        # 20 cm = 7.874 in, 15 cm = 5.906 in
        assert fig.get_figwidth() == pytest.approx(20 / 2.54, abs=1e-3)
        assert fig.get_figheight() == pytest.approx(15 / 2.54, abs=1e-3)
    finally:
        plt.close(fig)
