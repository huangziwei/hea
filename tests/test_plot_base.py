"""Tests for the R base-graphics ports: ``hist``, ``boxplot``, ``barplot``,
``density``, ``rug``, ``curve``.

Structural checks only (Axes returned, labels set, basic shape of binned
output) — pixel comparison belongs in a visual regression tool, not unit
tests.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from hea.plot import barplot, boxplot, curve, density, hist, rug
from hea.plot.density import _Density
from hea.R import table


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def diastolic():
    rng = np.random.default_rng(0)
    return pl.Series("diastolic", rng.normal(72, 12, 200))


# ---------------------------------------------------------------------------
# hist
# ---------------------------------------------------------------------------


def test_hist_returns_axes_with_explicit_labels(diastolic):
    """The chapter form: ``hist(pima$diastolic, xlab='Diastolic', main='')``."""
    ax = hist(diastolic, xlab="Diastolic", main="")
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_xlabel() == "Diastolic"
    assert ax.get_title() == ""


def test_hist_defaults_to_series_name(diastolic):
    """``xlab`` defaults to the polars Series name; ``main`` defaults to
    ``"Histogram of <name>"`` — matches R's deparsed call labels."""
    ax = hist(diastolic)
    assert ax.get_xlabel() == "diastolic"
    assert ax.get_title() == "Histogram of diastolic"


def test_hist_freq_is_counts_by_default(diastolic):
    ax = hist(diastolic)
    bars = [p for p in ax.patches]
    total_height = sum(b.get_height() for b in bars)
    assert total_height == diastolic.len()
    assert ax.get_ylabel() == "Frequency"


def test_hist_probability_switches_to_density(diastolic):
    """``probability=True`` is the negation of ``freq=True``."""
    ax = hist(diastolic, probability=True)
    bars = ax.patches
    # density-mode bar heights integrate to 1.0 (up to numerical error).
    area = sum(b.get_height() * b.get_width() for b in bars)
    assert abs(area - 1.0) < 1e-9
    assert ax.get_ylabel() == "Density"


def test_hist_breaks_int_gives_that_many_bins(diastolic):
    ax = hist(diastolic, breaks=20)
    assert len(ax.patches) == 20


def test_hist_breaks_explicit_edges(diastolic):
    edges = [40, 50, 60, 70, 80, 90, 100, 110]
    ax = hist(diastolic, breaks=edges)
    assert len(ax.patches) == len(edges) - 1


def test_hist_rejects_both_freq_and_probability(diastolic):
    with pytest.raises(ValueError, match="freq= or probability="):
        hist(diastolic, freq=True, probability=False)


# ---------------------------------------------------------------------------
# boxplot — standalone, single + multi vector.
# ---------------------------------------------------------------------------


def test_boxplot_single_vector():
    ax = boxplot(pl.Series("x", [1.0, 2.0, 3.0, 4.0]))
    assert isinstance(ax, matplotlib.axes.Axes)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["x"]


def test_boxplot_variadic_vectors():
    ax = boxplot([1, 2, 3, 4], [10, 20, 30, 40], names=("a", "b"))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["a", "b"]


def test_boxplot_list_of_vectors():
    ax = boxplot([[1, 2, 3], [4, 5, 6]], names=["x", "y"])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["x", "y"]


def test_boxplot_horizontal_flag():
    ax = boxplot([1.0, 2.0, 3.0], horizontal=True)
    # In horizontal mode matplotlib uses y-tick labels for groups.
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["1"]


def test_boxplot_requires_at_least_one_arg():
    with pytest.raises(TypeError, match="at least one"):
        boxplot()


# ---------------------------------------------------------------------------
# barplot — vector heights & table() input.
# ---------------------------------------------------------------------------


def test_barplot_vector_heights():
    ax = barplot([10, 20, 30], names=["a", "b", "c"], main="t", ylab="count")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["a", "b", "c"]
    assert ax.get_title() == "t"
    assert ax.get_ylabel() == "count"


def test_barplot_accepts_hea_table_frame():
    """``barplot(table(x))`` — the chapter form for a category-count plot."""
    s = pl.Series("type", ["a", "a", "b", "c", "c", "c"])
    ax = barplot(table(s))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["a", "b", "c"]
    heights = [b.get_height() for b in ax.patches]
    assert heights == [2.0, 1.0, 3.0]


def test_barplot_horiz_uses_y_axis():
    ax = barplot([1, 2, 3], names=["a", "b", "c"], horiz=True)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["a", "b", "c"]


def test_barplot_rejects_length_mismatch():
    with pytest.raises(ValueError, match="names has"):
        barplot([1, 2, 3], names=["only", "two"])


# ---------------------------------------------------------------------------
# density — KDE object + .plot()
# ---------------------------------------------------------------------------


def test_density_returns_density_object(diastolic):
    d = density(diastolic)
    assert isinstance(d, _Density)
    assert d.n == diastolic.len()
    assert d.x.shape == (512,)
    assert d.y.shape == (512,)
    assert d.bw > 0


def test_density_plot_draws_line(diastolic):
    d = density(diastolic)
    ax = d.plot()
    assert len(ax.lines) == 1
    assert ax.get_ylabel() == "Density"


def test_density_grid_size_n(diastolic):
    d = density(diastolic, n=128)
    assert d.x.shape == (128,)


def test_density_rejects_tiny_input():
    with pytest.raises(ValueError, match="at least 2"):
        density([1.0])


def test_plot_dispatches_density_object(diastolic):
    """``plot(density(x))`` should route through ``_Density.plot``, not
    fall through to the single-vector scatter path that tries to take
    ``len()`` of the density object."""
    from hea.plot import plot

    ax = plot(density(diastolic), main="")
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_title() == ""
    # The dispatch should have drawn a density curve, not scatter points.
    assert len(ax.lines) == 1


# ---------------------------------------------------------------------------
# rug — overlay on existing axes.
# ---------------------------------------------------------------------------


def test_rug_overlays_ticks(diastolic):
    fig, ax = plt.subplots()
    hist(diastolic, ax=ax)
    n_lines_before = len(ax.lines)
    rug(diastolic, ax=ax)
    # One tick per data point added.
    assert len(ax.lines) - n_lines_before == diastolic.len()


def test_rug_side_accepts_r_integer_codes(diastolic):
    fig, ax = plt.subplots()
    hist(diastolic, ax=ax)
    rug(diastolic, side=2, ax=ax)
    assert len(ax.lines) == diastolic.len()


def test_rug_falls_back_to_last_hea_axes(diastolic):
    """Like other overlays, ``rug`` targets the most recent hea axes
    when ``ax=`` is omitted (so ``hist(x); rug(x)`` works directly)."""
    ax = hist(diastolic)
    rug(diastolic)
    assert len(ax.lines) == diastolic.len()


def test_rug_with_no_prior_plot_errors():
    """No prior hea plot → ``rug`` still raises (no sensible default)."""
    from hea.plot import _util

    saved, _util._LAST_AX = _util._LAST_AX, None
    try:
        with pytest.raises(ValueError, match="no previous hea plot"):
            rug([1.0, 2.0])
    finally:
        _util._LAST_AX = saved


def test_rug_rejects_bad_side(diastolic):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="side="):
        rug(diastolic, side="nowhere", ax=ax)


# ---------------------------------------------------------------------------
# curve — function evaluation + overlay.
# ---------------------------------------------------------------------------


def test_curve_plots_function():
    ax = curve(lambda x: x**2, -1, 1, n=21)
    assert len(ax.lines) == 1
    xs, ys = ax.lines[0].get_data()
    assert len(xs) == 21
    assert np.allclose(ys, xs**2)


def test_curve_overlay_uses_existing_xlim():
    """When ``add=True`` and no ``from_/to`` given, the existing axes
    x-limits are used (standard 'overlay a theoretical curve on a
    histogram' pattern)."""
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 5.0)
    curve(lambda x: x, add=True, ax=ax)
    xs, _ = ax.lines[0].get_data()
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(5.0)


def test_curve_overlay_does_not_overwrite_labels():
    fig, ax = plt.subplots()
    ax.set_xlabel("kept")
    curve(lambda x: x, 0, 1, add=True, ax=ax, xlab="ignored")
    assert ax.get_xlabel() == "kept"
