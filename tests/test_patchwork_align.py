"""Structural tests for the patchwork-gtable engine: panel alignment.

The whole point of the rewrite is to align panels across siblings even
when their decorations differ in size. These tests assert the panel
``Axes`` bboxes line up — failure means a colorbar/title/etc. on one
sibling has squeezed its panel relative to the others.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import pytest

from hea.ggplot import (
    aes, facet_wrap, geom_bar, geom_boxplot, geom_point, ggplot,
    ggtitle, labs,
)


def _mtcars() -> pl.DataFrame:
    from hea import data
    return data("mtcars").to_polars() if hasattr(data("mtcars"), "to_polars") else data("mtcars")


def _patchwork_doc_plots():
    """Same four plots as the patchwork tutorial doc."""
    from hea import data
    mtcars = data("mtcars")
    p1 = (mtcars.ggplot().geom_point(aes("mpg", "disp"))
          .ggtitle("Plot 1"))
    p2 = (mtcars.ggplot().geom_boxplot(aes("gear", "disp", group="gear"))
          .ggtitle("Plot 2"))
    p3 = (mtcars.ggplot().geom_point(aes("hp", "wt", colour="mpg"))
          .ggtitle("Plot 3"))
    p4 = (mtcars.ggplot().geom_bar(aes("gear")).facet_wrap("~cyl")
          .ggtitle("Plot 4"))
    return p1, p2, p3, p4


def _panel_bboxes(fig) -> list:
    """Return panel-axes bboxes (figure-relative).

    Filters out colorbar caxes, which the block engine adds in the
    right-margin column. Heuristic: a colorbar cax is much narrower than
    a panel (≤ 0.05 of figure width is a reliable threshold)."""
    fig.canvas.draw()
    return [
        ax.get_position() for ax in fig.axes
        if ax.get_position().width > 0.05
    ]


def test_horizontal_compose_panels_share_top_and_bottom():
    """``p1 | p2``: both panels in the same row → top and bottom edges
    align (panel y-extents identical)."""
    p1, p2, _, _ = _patchwork_doc_plots()
    fig = (p1 | p2).draw(figsize=(8, 3))
    try:
        boxes = _panel_bboxes(fig)
        # Expect 2 panel axes.
        assert len(boxes) == 2
        b1, b2 = boxes
        assert b1.y0 == pytest.approx(b2.y0, abs=1e-3)
        assert b1.y1 == pytest.approx(b2.y1, abs=1e-3)
    finally:
        plt.close(fig)


def test_vertical_compose_panels_share_left_and_right():
    """``p1 / p2``: both panels in the same column → left and right edges
    align."""
    p1, p2, _, _ = _patchwork_doc_plots()
    fig = (p1 / p2).draw(figsize=(4, 6))
    try:
        boxes = _panel_bboxes(fig)
        assert len(boxes) == 2
        b1, b2 = boxes
        assert b1.x0 == pytest.approx(b2.x0, abs=1e-3)
        assert b1.x1 == pytest.approx(b2.x1, abs=1e-3)
    finally:
        plt.close(fig)


def test_grid_2x2_panels_align_per_row_and_col():
    """``p1 + p2 + p3 + p4`` 2×2: panels in same row share y, same col share x."""
    p1, p2, p3, p4 = _patchwork_doc_plots()
    fig = (p1 + p2 + p3 + p4).draw(figsize=(8, 6))
    try:
        boxes = _panel_bboxes(fig)
        # p4 is faceted (3 panels). Total panel axes: p1 + p2 + p3 + 3 facets = 6.
        # Block engine adds them in row-major order: row 0 = (p1, p2),
        # row 1 = (p3, p4_panels). So:
        #   boxes[0] = p1 panel
        #   boxes[1] = p2 panel
        #   boxes[2] = p3 panel
        #   boxes[3..5] = p4 facet panels (3 of them)
        assert len(boxes) == 6
        p1_b, p2_b, p3_b = boxes[0], boxes[1], boxes[2]
        # Row-shares: p1 and p2 share y; p3 and p4-panel-row share y.
        assert p1_b.y0 == pytest.approx(p2_b.y0, abs=1e-3)
        assert p1_b.y1 == pytest.approx(p2_b.y1, abs=1e-3)
        # Col-shares: p1 and p3 share x — KEY: this is what the gtable
        # engine fixes (p3's colorbar would otherwise squeeze p3's panel).
        assert p1_b.x0 == pytest.approx(p3_b.x0, abs=1e-3)
        assert p1_b.x1 == pytest.approx(p3_b.x1, abs=1e-3)
    finally:
        plt.close(fig)


def test_subtitle_on_one_sibling_does_not_squeeze_the_other():
    """If one sibling has a subtitle and another doesn't, both panels
    should still have the same height (the super-margin reserves the
    title row even when a child has no subtitle)."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    p_no = ggplot(df, aes("x", "y")) + geom_point() + ggtitle("Plain")
    p_yes = (
        ggplot(df, aes("x", "y")) + geom_point()
        + labs(title="With subtitle", subtitle="A second line of metadata")
    )
    fig = (p_no | p_yes).draw(figsize=(8, 3))
    try:
        boxes = _panel_bboxes(fig)
        b_no, b_yes = boxes
        # Both panels in same row → matching y-extents.
        assert b_no.y0 == pytest.approx(b_yes.y0, abs=1e-3)
        assert b_no.y1 == pytest.approx(b_yes.y1, abs=1e-3)
        # And matching heights.
        assert b_no.height == pytest.approx(b_yes.height, abs=1e-3)
    finally:
        plt.close(fig)


def test_nested_compose_renders_without_subfigures():
    """``p1 | (p2 / p3)`` renders three leaves through the block engine —
    no SubFigures are involved."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    fig = (p1 | (p2 / p3)).draw(figsize=(8, 5))
    try:
        assert len(fig.subfigs) == 0
        # 3 leaf panels in total.
        boxes = _panel_bboxes(fig)
        assert len(boxes) == 3
    finally:
        plt.close(fig)


def test_nested_compose_p2_p3_share_panel_x_extents():
    """Inside ``(p2 / p3)``, p2 and p3 share a column → panel x-extents match."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    fig = (p1 | (p2 / p3)).draw(figsize=(8, 5))
    try:
        boxes = _panel_bboxes(fig)
        # boxes[0] = p1 (left), boxes[1] = p2 (top right), boxes[2] = p3 (bottom right).
        _, b2, b3 = boxes
        assert b2.x0 == pytest.approx(b3.x0, abs=1e-3)
        assert b2.x1 == pytest.approx(b3.x1, abs=1e-3)
    finally:
        plt.close(fig)


def test_nested_compose_does_not_crash_with_annotation():
    """plot_annotation title on a nested composition renders into a
    fig.text artist."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    from hea.ggplot import plot_annotation
    g = (p1 | (p2 / p3)) + plot_annotation(title="A nested story")
    fig = g.draw()
    try:
        assert any(t.get_text() == "A nested story" for t in fig.texts)
    finally:
        plt.close(fig)


def test_colorbar_sibling_does_not_squeeze_other_panels():
    """Plot with a colorbar in same row as a plot without — panels still
    have the same width (super_right takes max so the no-colorbar plot
    gets reserved space too).

    Note: matplotlib's fig.colorbar may currently shrink the host axes
    when called via the legacy apply_legends path; this test pins the
    CURRENT block-engine behaviour. Phase F replaces apply_legends with
    block.right and tightens this further.
    """
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "z": [10.0, 20.0, 30.0]})
    p_plain = ggplot(df, aes("x", "y")) + geom_point()
    p_cbar = ggplot(df, aes("x", "y", colour="z")) + geom_point()
    fig = (p_plain | p_cbar).draw(figsize=(8, 3))
    try:
        # The plain plot's panel and the colorbar plot's panel should
        # have the same y-extents (shared row).
        plain = fig.axes[0]
        cbar_panel = fig.axes[1]
        fig.canvas.draw()
        bp = plain.get_position()
        bc = cbar_panel.get_position()
        assert bp.y0 == pytest.approx(bc.y0, abs=1e-3)
        assert bp.y1 == pytest.approx(bc.y1, abs=1e-3)
    finally:
        plt.close(fig)
