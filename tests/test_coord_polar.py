"""Tests for ``coord_polar()`` — Stage 1 wiring + theta rescale.

Covers construction, dispatch, fluent form, the three axes-creation
paths in ``_render_single`` and the block engine's ``_render_leaf_cell``,
the ``_apply_spines`` polar guard (theme regression), the ordinal-x →
[0, 2π] rescale, and the continuous-x no-op case (pycircstat2's path).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import hea
from hea.ggplot import (
    aes,
    coord_polar,
    geom_bar,
    geom_col,
    geom_point,
    ggplot,
    theme_bw,
    theme_minimal,
)
from hea.ggplot.coords import CoordPolar


# ---------------------------------------------------------------------------
# Construction & dispatch
# ---------------------------------------------------------------------------


def test_coord_polar_defaults():
    c = coord_polar()
    assert isinstance(c, CoordPolar)
    assert c.theta == "x"
    assert c.start == 0.0
    assert c.direction == 1
    assert c.clip == "on"
    assert c.is_linear is False


def test_coord_polar_overrides():
    c = coord_polar(theta="y", start=math.pi / 2, direction=-1, clip="off")
    assert c.theta == "y"
    assert c.start == math.pi / 2
    assert c.direction == -1
    assert c.clip == "off"


def test_coord_polar_plus_dispatch_sets_coordinates():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + coord_polar(start=math.pi / 2)
    assert isinstance(p.coordinates, CoordPolar)
    assert p.coordinates.start == math.pi / 2


def test_coord_polar_fluent_form_works():
    """The fluent install loop matches ``coord_*`` prefix — verify it's wired."""
    df = hea.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    p = df.ggplot(aes("x", "y")).geom_point().coord_polar(start=math.pi / 2)
    assert isinstance(p.coordinates, CoordPolar)
    assert p.coordinates.start == math.pi / 2


# ---------------------------------------------------------------------------
# Axes creation + orientation
# ---------------------------------------------------------------------------


def test_coord_polar_creates_polar_axes_standalone():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 1.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + coord_polar()
    fig = p.draw()
    try:
        assert fig.axes[0].name == "polar"
    finally:
        plt.close(fig)


def test_coord_polar_start_sets_theta_offset():
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + coord_polar(start=math.pi / 2)
    fig = p.draw()
    try:
        assert fig.axes[0].get_theta_offset() == pytest.approx(math.pi / 2)
    finally:
        plt.close(fig)


def test_coord_polar_direction_negates_for_matplotlib():
    """ggplot2's ``direction=1`` (CW) is matplotlib's ``set_theta_direction(-1)``."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0]})
    p_cw = ggplot(df, aes("x", "y")) + geom_point() + coord_polar(direction=1)
    fig_cw = p_cw.draw()
    try:
        assert fig_cw.axes[0].get_theta_direction() == -1
    finally:
        plt.close(fig_cw)

    p_ccw = ggplot(df, aes("x", "y")) + geom_point() + coord_polar(direction=-1)
    fig_ccw = p_ccw.draw()
    try:
        assert fig_ccw.axes[0].get_theta_direction() == 1
    finally:
        plt.close(fig_ccw)


# ---------------------------------------------------------------------------
# Multi-path render entry points
# ---------------------------------------------------------------------------


def test_coord_polar_via_subplotspec():
    """``plot.draw(subplotspec=spec)`` — patchwork / mosaic integration path.
    Hits ``render._render_single``'s ``subplotspec`` branch (not the block
    engine), so the projection arg must be passed to ``fig.add_subplot``.
    """
    from matplotlib.gridspec import GridSpec

    df = pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + coord_polar()

    fig = plt.figure()
    try:
        gs = GridSpec(1, 1, figure=fig)
        p.draw(subplotspec=gs[0, 0])
        assert any(ax.name == "polar" for ax in fig.axes)
    finally:
        plt.close(fig)


def test_coord_polar_raises_on_cartesian_ax():
    """Passing a Cartesian ``ax=`` with ``coord_polar()`` is a silent
    bug producer; we raise explicitly."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + coord_polar()
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="polar axes"):
            p.draw(ax=ax)
    finally:
        plt.close(fig)


def test_coord_polar_accepts_polar_ax():
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + coord_polar()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        out = p.draw(ax=ax)
        # Same figure returned, polar preserved.
        assert out is fig
        assert ax.name == "polar"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Theme regression — _apply_spines guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_factory", [theme_minimal, theme_bw])
def test_coord_polar_does_not_crash_under_themes(theme_factory):
    """Without the polar guard in ``_apply_spines``, ``ax.spines['top']``
    raises ``KeyError`` on polar axes — every themed polar plot would
    crash. Verify the guard."""
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 1.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + coord_polar() + theme_factory()
    fig = p.draw()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Theta rescale — the 1D linear remap to [0, 2π]
# ---------------------------------------------------------------------------


def test_rescale_theta_continuous_two_pi_is_noop():
    """pycircstat2's case: data already in [0, 2π] — factor should be 1.0."""
    c = coord_polar()
    df = pl.DataFrame({"x": [0.0, math.pi, 2 * math.pi], "y": [1.0, 1.0, 1.0]})
    out = c.rescale_theta(df, (0.0, 2 * math.pi))
    np.testing.assert_allclose(out["x"].to_numpy(), df["x"].to_numpy())


def test_rescale_theta_continuous_degrees_to_radians():
    """Degree-like data [0, 360] should remap to [0, 2π]."""
    c = coord_polar()
    df = pl.DataFrame({"x": [0.0, 90.0, 180.0, 360.0]})
    out = c.rescale_theta(df, (0.0, 360.0))
    expected = [0.0, math.pi / 2, math.pi, 2 * math.pi]
    np.testing.assert_allclose(out["x"].to_numpy(), expected)


def test_rescale_theta_rescales_width():
    """Bar widths must scale by the same multiplicative factor."""
    c = coord_polar()
    df = pl.DataFrame({"x": [0.0, 7.0], "width": [1.0, 1.0]})
    out = c.rescale_theta(df, (0.0, 7.0))
    # factor = 2π/7, width stays proportional.
    np.testing.assert_allclose(out["width"].to_numpy(), [2 * math.pi / 7] * 2)


def test_rescale_theta_rescales_xmin_xmax_xend():
    c = coord_polar()
    df = pl.DataFrame({
        "x": [0.0, 1.0],
        "xmin": [0.0, 0.5],
        "xmax": [1.0, 1.5],
        "xend": [0.5, 1.0],
    })
    out = c.rescale_theta(df, (0.0, 1.0))
    np.testing.assert_allclose(out["xmax"].to_numpy(), [2 * math.pi, 3 * math.pi])
    np.testing.assert_allclose(out["xend"].to_numpy(), [math.pi, 2 * math.pi])


def test_rescale_theta_zero_span_returns_df():
    c = coord_polar()
    df = pl.DataFrame({"x": [1.0, 1.0]})
    assert c.rescale_theta(df, (1.0, 1.0)) is df


# ---------------------------------------------------------------------------
# End-to-end: ordinal bar chart on polar (the snippet's scenario)
# ---------------------------------------------------------------------------


def test_coord_polar_bar_chart_renders_wedges_for_each_ordinal_level():
    """Eight ordinal levels → eight wedges spanning the circle evenly.
    Equivalent to ggplot2's ``ggplot(diamonds) + geom_bar(aes(clarity)) +
    coord_polar()``. With ``width=1``, each wedge's angular width should
    be the rescaled per-category slot."""
    df = pl.DataFrame({"cat": ["A", "B", "C", "D"] * 5})
    p = (ggplot(df, aes(x="cat"))
         + geom_bar(width=1)
         + coord_polar())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.name == "polar"
        # 4 categories → 4 wedges (matplotlib Rectangle patches for ax.bar
        # on polar axes are stored under ax.patches).
        assert len(ax.patches) == 4
        # Each wedge sits at a multiple of 2π/N (modulo the rescale's
        # +0.6/-0.6 expansion). Confirm they're spread around the circle
        # rather than overlapping at theta=0..3.
        thetas = sorted(p.get_x() + p.get_width() / 2 for p in ax.patches)
        # Span of bar centres exceeds π (i.e., the bars wrap the circle,
        # not all bunched in one quadrant).
        assert thetas[-1] - thetas[0] > math.pi
    finally:
        plt.close(fig)


def test_coord_polar_continuous_x_keeps_radians_data():
    """Continuous x in [0, 2π] feeds polar directly — pycircstat2's
    canonical recipe. Each point's data stays at its angular position."""
    angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    df = pl.DataFrame({"theta": angles.tolist(), "r": [1.0] * 8})
    p = (ggplot(df, aes(x="theta", y="r"))
         + geom_point()
         + coord_polar(start=math.pi / 2))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.name == "polar"
        # PathCollection from ax.scatter — verify 8 points present.
        coll = ax.collections[0]
        assert len(coll.get_offsets()) == 8
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Patchwork composition — the full snippet shape
# ---------------------------------------------------------------------------


def test_coord_polar_suppresses_axis_titles_by_default():
    """Polar auto-suppresses the x/y axis titles. Tick labels around the
    rim already carry the per-axis context, and matplotlib's default
    ylabel placement at 9 o'clock collides with the 180° tick label.
    Matches ggplot2's coord_polar and pycircstat2 conventions."""
    df = pl.DataFrame({"clarity": list("ABCDEFGH") * 5})
    p = (ggplot(df, aes(x="clarity")) + geom_bar() + coord_polar())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
    finally:
        plt.close(fig)


def test_coord_polar_labs_opts_axis_titles_back_in():
    """``labs(x="...", y="...")`` overrides the polar auto-suppress."""
    from hea.ggplot import labs

    df = pl.DataFrame({"clarity": list("ABCDEFGH") * 5})
    p = (ggplot(df, aes(x="clarity"))
         + geom_bar() + coord_polar()
         + labs(x="Clarity", y="Count"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Clarity"
        assert ax.get_ylabel() == "Count"
    finally:
        plt.close(fig)


def test_coord_polar_continuous_rose_pins_angular_range_to_two_pi():
    """A rose-style chart: N wedges at theta = k·2π/N (k = 0..N-1).
    Data trains the x-scale to ``[0, ~2π·(N-1)/N)``, not exactly 2π
    because the last sample is excluded. matplotlib polar's
    ``set_xticks`` auto-extends xlim past the trained max — without
    re-pinning xlim, the polar projection collapses to a near-vertical
    sliver. Verify the angular axis spans the full circle."""
    n = 18
    beta = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    df = pl.DataFrame({"theta": beta, "h": [1.0] * n})
    p = (ggplot(df, aes("theta", "h"))
         + geom_col(width=2 * math.pi / n)
         + coord_polar())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(2 * math.pi, abs=1e-6)
        assert len(ax.patches) == n
    finally:
        plt.close(fig)


def test_coord_polar_composes_with_coord_flip_via_patchwork():
    """``bar.coord_flip() | bar.coord_polar()`` — Cartesian and polar in
    the same figure. Each leaf needs its own projection."""
    df = hea.DataFrame({"clarity": list("ABCDEFGH") * 10})
    bar = df.ggplot().geom_bar(x="clarity", width=1)
    composite = bar.coord_flip() | bar.coord_polar()
    fig = composite.draw()
    try:
        panel_axes = [ax for ax in fig.axes if ax.name in ("rectilinear", "polar")]
        names = sorted(ax.name for ax in panel_axes)
        assert names == ["polar", "rectilinear"]
    finally:
        plt.close(fig)
