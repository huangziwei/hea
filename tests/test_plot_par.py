"""Tests for ``hea.plot.par`` — the scoped multi-panel layout context manager.

``par(mfrow=(r, c))`` ports R's ``par(mfrow=c(r, c))`` idiom, but the
state is bounded by the ``with`` block: when the block exits the next
plot call once again opens its own fresh figure (no leakage).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from hea.plot import barplot, boxplot, curve, density, hist, par, plot, qqnorm
from hea.plot.par import _PAR_STACK


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def diastolic():
    rng = np.random.default_rng(0)
    return pl.Series("diastolic", rng.normal(72, 12, 80))


# ---------------------------------------------------------------------------
# Basic scoping: plots claim grid cells in row-major order.
# ---------------------------------------------------------------------------


def test_mfrow_pulls_cells_in_row_major_order(diastolic):
    with par(mfrow=(2, 3)) as p:
        a0 = hist(diastolic, main="a")
        a1 = hist(diastolic, main="b")
        a2 = hist(diastolic, main="c")
        a3 = hist(diastolic, main="d")
    fig = p.figure
    assert len(fig.axes) == 6
    # All four claimed cells came from the same figure, in row-major order.
    used = [a0, a1, a2, a3]
    fig_axes = list(fig.axes)
    assert [fig_axes.index(a) for a in used] == [0, 1, 2, 3]


def test_mfcol_pulls_cells_in_column_major_order(diastolic):
    with par(mfcol=(2, 3)) as p:
        a0 = hist(diastolic, main="a")
        a1 = hist(diastolic, main="b")
    fig_axes = list(p.figure.axes)
    # Column-major: a0 → (0,0), a1 → (1,0). Matplotlib's ``axes`` list is
    # still in C order, so a1 should be the 4th entry (row 1, col 0 of a
    # 2×3 grid).
    assert fig_axes.index(a0) == 0
    assert fig_axes.index(a1) == 3


# ---------------------------------------------------------------------------
# Unused cells get hidden so the figure doesn't show blank panels.
# ---------------------------------------------------------------------------


def test_unused_cells_are_hidden(diastolic):
    with par(mfrow=(2, 2)) as p:
        hist(diastolic)
        hist(diastolic)
    fig = p.figure
    visibles = [a.get_visible() for a in fig.axes]
    assert visibles == [True, True, False, False]


# ---------------------------------------------------------------------------
# Exhaustion raises a clear error rather than silently overflowing.
# ---------------------------------------------------------------------------


def test_too_many_plots_raises(diastolic):
    with pytest.raises(RuntimeError, match="all 2 cells used"):
        with par(mfrow=(1, 2)):
            hist(diastolic)
            hist(diastolic)
            hist(diastolic)


# ---------------------------------------------------------------------------
# Explicit ax= bypasses the grid — caller fully in control.
# ---------------------------------------------------------------------------


def test_explicit_ax_bypasses_grid(diastolic):
    fig_ext, ax_ext = plt.subplots()
    with par(mfrow=(1, 2)) as p:
        # Drop a hist onto an externally-owned axes; the par grid
        # shouldn't advance.
        hist(diastolic, ax=ax_ext)
        # Next call should still get cell 0.
        a0 = hist(diastolic)
        a1 = hist(diastolic)
    fig_par = p.figure
    assert ax_ext.figure is fig_ext
    assert a0.figure is fig_par
    assert a1.figure is fig_par
    assert list(fig_par.axes).index(a0) == 0
    assert list(fig_par.axes).index(a1) == 1


# ---------------------------------------------------------------------------
# State scoping: stack empty before/after the with-block; no leakage.
# ---------------------------------------------------------------------------


def test_stack_is_clean_before_and_after(diastolic):
    assert _PAR_STACK == []
    with par(mfrow=(1, 2)):
        assert len(_PAR_STACK) == 1
        hist(diastolic)
    assert _PAR_STACK == []


def test_outside_par_each_call_makes_own_figure(diastolic):
    a0 = hist(diastolic)
    a1 = hist(diastolic)
    assert a0.figure is not a1.figure


# ---------------------------------------------------------------------------
# Nested par(): innermost wins, outer resumes after the inner exits.
# ---------------------------------------------------------------------------


def test_nested_par_innermost_wins(diastolic):
    with par(mfrow=(1, 3)) as outer:
        outer_a0 = hist(diastolic)
        with par(mfrow=(1, 2)) as inner:
            inner_a0 = hist(diastolic)
            inner_a1 = hist(diastolic)
        outer_a1 = hist(diastolic)
    # outer_a0 and outer_a1 come from the outer figure...
    assert outer_a0.figure is outer.figure
    assert outer_a1.figure is outer.figure
    # ...and the two inner ones from the inner figure.
    assert inner_a0.figure is inner.figure
    assert inner_a1.figure is inner.figure


# ---------------------------------------------------------------------------
# Every single-panel plotter we ported routes through resolve_ax.
# ---------------------------------------------------------------------------


def test_par_works_for_every_single_panel_plotter(diastolic):
    """The exact scenario the user reported: hist + plot(density(.)) +
    plot(sort(.)) inside par(mfrow=(1,3))."""
    from hea.R import sort

    with par(mfrow=(2, 3)) as p:
        hist(diastolic)
        plot(density(diastolic))
        plot(sort(diastolic))
        boxplot(diastolic)
        barplot([1, 2, 3], names=["a", "b", "c"])
        qqnorm(diastolic)
    fig_axes = [a for a in p.figure.axes if a.get_visible()]
    assert len(fig_axes) == 6


def test_curve_inside_par(diastolic):
    """curve() with explicit from_/to should also pull from par."""
    with par(mfrow=(1, 2)) as p:
        curve(lambda x: x**2, -1, 1)
        curve(lambda x: x**3, -1, 1)
    assert len([a for a in p.figure.axes if a.get_visible()]) == 2


def test_profile_plot_single_param_inside_par():
    """``Profile.plot(which="...", ...)`` is a single-panel call —
    inside ``par(mfrow=...)`` it should pull a cell rather than open
    its own figure (Bates Fig. 1.7-style ``par`` ergonomics)."""
    from hea import data, lme

    dye = data("Dyestuff")
    fm = lme("Yield ~ 1 + (1 | Batch)", dye, REML=False)
    pr = fm.profile()

    with par(mfrow=(1, 3)) as p:
        pr.plot(which=".sigma", transform="log")
        pr.plot(which=".sigma")
        pr.plot(which=".sigma", transform="square")

    titles = [a.get_title() for a in p.figure.axes if a.get_visible()]
    assert titles == ["log(.sigma)", ".sigma", ".sigma²"]


# ---------------------------------------------------------------------------
# Argument validation.
# ---------------------------------------------------------------------------


def test_par_rejects_both_mfrow_and_mfcol():
    with pytest.raises(ValueError, match="not both"):
        par(mfrow=(1, 2), mfcol=(2, 1))


def test_par_requires_one_of_mfrow_mfcol():
    with pytest.raises(ValueError, match="mfrow|mfcol"):
        par()


def test_par_rejects_bad_shape():
    with pytest.raises(TypeError, match="positive ints"):
        par(mfrow=(0, 3))
    with pytest.raises(TypeError, match="positive ints"):
        par(mfrow=("a", 3))


# ---------------------------------------------------------------------------
# Cleanup: an exception inside the block still pops the stack.
# ---------------------------------------------------------------------------


def test_exception_inside_par_unwinds_stack():
    assert _PAR_STACK == []
    with pytest.raises(RuntimeError, match="boom"):
        with par(mfrow=(1, 2)):
            raise RuntimeError("boom")
    assert _PAR_STACK == []
