"""hea.ggplot tests — Phase 0 onward.

Each test ID matches the inventory in `.claude/plans/ggplot2-port.md` §9.1.
PNG diff parity (the visual-snapshot tests) come later; Phase 0 just
asserts the build pipeline runs and produces a matplotlib Figure with
expected primitives.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import polars as pl
import pytest

from conftest import load_dataset

from hea.ggplot import (
    aes, geom_blank, geom_density, geom_histogram, geom_point, ggplot,
    scale_x_continuous, scale_y_continuous,
)


def test_gg_c1_minimal_scatter_renders():
    """GG-C1: ``ggplot(mtcars, aes(wt, mpg)) + geom_point()`` renders identifiably.

    Asserts (Phase 0 form):
    - ``draw()`` returns a matplotlib Figure;
    - the figure has exactly one axes with one PathCollection (scatter);
    - that scatter holds N=nrow(mtcars) points;
    - the points span roughly the data range on each axis.
    """
    mtcars = load_dataset("datasets", "mtcars")

    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point()
    fig = p.draw()

    try:
        assert isinstance(fig, plt.Figure)
        axes = fig.axes
        assert len(axes) == 1, f"expected one axes, got {len(axes)}"
        ax = axes[0]

        scatters = [c for c in ax.collections if c.__class__.__name__ == "PathCollection"]
        assert len(scatters) == 1, f"expected one scatter, got {len(scatters)}"

        offsets = scatters[0].get_offsets()
        assert offsets.shape == (len(mtcars), 2), \
            f"expected {len(mtcars)} points, got {offsets.shape[0]}"

        x_vals = mtcars["wt"].to_numpy()
        y_vals = mtcars["mpg"].to_numpy()
        # Some matplotlib slack on autoscaling; just check the offsets cover the
        # data extents within float precision.
        assert offsets[:, 0].min() == pytest.approx(x_vals.min())
        assert offsets[:, 0].max() == pytest.approx(x_vals.max())
        assert offsets[:, 1].min() == pytest.approx(y_vals.min())
        assert offsets[:, 1].max() == pytest.approx(y_vals.max())

        assert ax.get_xlabel() == "wt"
        assert ax.get_ylabel() == "mpg"
    finally:
        plt.close(fig)


def test_geom_blank_layer_adds_no_artists():
    """`geom_blank()` extends scales but draws nothing (Phase 0: no scales yet,
    so this just asserts the layer doesn't put anything on the axes)."""
    mtcars = load_dataset("datasets", "mtcars")

    p = ggplot(mtcars, aes("wt", "mpg")) + geom_blank()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.collections) == 0, "geom_blank should not produce artists"
    finally:
        plt.close(fig)


def test_aes_canonicalises_color_to_colour():
    """``aes(color=...)`` should canonicalise to ``colour`` (matches ggplot2)."""
    a = aes(color="red")
    assert "colour" in a
    assert "color" not in a
    assert a["colour"] == "red"


def test_aes_positional_args_bind_x_then_y():
    a = aes("wt", "mpg")
    assert a["x"] == "wt"
    assert a["y"] == "mpg"


def test_plus_returns_new_plot_not_mutated():
    """``+`` is non-mutating: original plot keeps its layer count."""
    mtcars = load_dataset("datasets", "mtcars")
    base = ggplot(mtcars, aes("wt", "mpg"))
    extended = base + geom_point()
    assert len(base.layers) == 0
    assert len(extended.layers) == 1


def test_plus_unknown_type_errors():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars)
    with pytest.raises(TypeError, match="can't add"):
        p + 42  # noqa: B015


def test_plus_list_adds_each_in_order():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + [geom_blank(), geom_point()]
    assert len(p.layers) == 2


def test_aes_expression_evaluated_via_formula_parser():
    """Per plan §13 Q3: aes value `"log(wt)"` parses as a Call node and is
    evaluated, not treated as a (missing) column literally named `log(wt)`."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("log(wt)", "mpg")) + geom_point()
    fig = p.draw()
    try:
        offsets = fig.axes[0].collections[0].get_offsets()
        np.testing.assert_allclose(offsets[:, 0], np.log(mtcars["wt"].to_numpy()))
    finally:
        plt.close(fig)


def test_aes_callable_value():
    """Callable aes values get the layer's data and produce the column."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes(x=lambda d: d["wt"] * 2, y="mpg"))
         + geom_point())
    fig = p.draw()
    try:
        offsets = fig.axes[0].collections[0].get_offsets()
        np.testing.assert_allclose(offsets[:, 0], mtcars["wt"].to_numpy() * 2)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 1.1 — Continuous scales: scale_x/y_continuous knobs (limits, breaks,
# labels) plus auto-default registration in build.
# ---------------------------------------------------------------------------


def test_scale_x_continuous_limits_override_autoscale():
    """``scale_x_continuous(limits=(0, 10))`` should clamp xlim regardless of data."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + scale_x_continuous(limits=(0, 10)))
    fig = p.draw()
    try:
        assert fig.axes[0].get_xlim() == (0, 10)
    finally:
        plt.close(fig)


def test_scale_x_continuous_explicit_breaks_and_labels():
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + scale_x_continuous(breaks=[2, 3, 4], labels=["light", "mid", "heavy"]))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        ticks = ax.get_xticks()
        assert list(ticks) == [2.0, 3.0, 4.0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["light", "mid", "heavy"]
    finally:
        plt.close(fig)


def test_scale_y_continuous_breaks_none_hides_ticks():
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + scale_y_continuous(breaks=None))
    fig = p.draw()
    try:
        assert len(fig.axes[0].get_yticks()) == 0
    finally:
        plt.close(fig)


def test_scale_default_auto_registered_when_user_omits():
    """Without `+ scale_x_continuous()`, a default ScaleContinuous is created
    inside build() so render() always has something to apply."""
    from hea.ggplot.scales import ScaleContinuous

    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point()

    from hea.ggplot.build import build
    bo = build(p)
    assert isinstance(bo.scales.get("x"), ScaleContinuous)
    assert isinstance(bo.scales.get("y"), ScaleContinuous)


def test_repeated_draw_produces_identical_ticks():
    """build() copies the ScalesList so repeated draws are deterministic — no
    accumulating range_ state, no break drift."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + scale_x_continuous())

    fig1 = p.draw()
    ticks1 = list(fig1.axes[0].get_xticks())
    plt.close(fig1)

    fig2 = p.draw()
    ticks2 = list(fig2.axes[0].get_xticks())
    plt.close(fig2)

    assert ticks1 == ticks2


def test_scale_plus_is_non_mutating():
    mtcars = load_dataset("datasets", "mtcars")
    base = ggplot(mtcars, aes("wt", "mpg")) + geom_point()
    extended = base + scale_x_continuous(limits=(0, 10))
    # original plot's scales unchanged
    assert base.scales.get("x") is None
    assert extended.scales.get("x") is not None


# ---------------------------------------------------------------------------
# Faraway "Linear Models with R" page 5 — the three exploratory plots.
# These transliterate the R one-liners directly to hea.ggplot. They lock the
# minimum viable surface (geom_point + geom_histogram + geom_density) for
# the use cases the book opens with.
# ---------------------------------------------------------------------------


def test_gg_c8_histogram_pima_diastolic():
    """`ggplot(pima, aes(x=diastolic)) + geom_histogram()` (Faraway p.5)."""
    from hea.data import data as _hea_data

    pima = _hea_data("pima", package="faraway")
    p = ggplot(pima, aes(x="diastolic")) + geom_histogram()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Default bins=30; bar heights sum to N (no missing in pima.diastolic).
        assert len(ax.patches) == 30
        total = sum(bar.get_height() for bar in ax.patches)
        assert total == len(pima), f"expected {len(pima)} obs, got {total}"
        assert ax.get_xlabel() == "diastolic"
        assert ax.get_ylabel() == "count"
    finally:
        plt.close(fig)


def test_gg_c10_density_pima_diastolic():
    """`ggplot(pima, aes(x=diastolic)) + geom_density()` (Faraway p.5)."""
    from hea.data import data as _hea_data

    pima = _hea_data("pima", package="faraway")
    p = ggplot(pima, aes(x="diastolic")) + geom_density()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 1, f"expected one density curve, got {len(ax.lines)}"
        xy = ax.lines[0].get_xydata()
        # 512 points by default, monotone in x, non-negative density.
        assert xy.shape == (512, 2)
        assert (xy[:-1, 0] <= xy[1:, 0]).all(), "x grid must be sorted ascending"
        assert (xy[:, 1] >= 0).all(), "density must be non-negative"
        # Density integrates to ≈ 1 over its support (3·bw padding on each side).
        from numpy import trapezoid
        area = trapezoid(xy[:, 1], xy[:, 0])
        assert 0.95 < area < 1.05, f"density should integrate to ≈1, got {area:.4f}"
        assert ax.get_xlabel() == "diastolic"
        assert ax.get_ylabel() == "density"
    finally:
        plt.close(fig)


def test_draw_into_existing_axes_via_subplot_mosaic():
    """`draw(ax=ax)` paints into a caller-supplied axes (e.g. ``plt.subplot_mosaic``)."""
    from hea.data import data as _hea_data

    pima = _hea_data("pima", package="faraway")
    fig, axes = plt.subplot_mosaic([["hist", "scatter"]], figsize=(8, 3))
    try:
        p1 = ggplot(pima, aes(x="diastolic")) + geom_histogram()
        p2 = ggplot(pima, aes(x="diastolic", y="diabetes")) + geom_point()

        returned = p1.draw(ax=axes["hist"])
        assert returned is fig, "draw(ax=) should return the parent Figure"

        p2.draw(ax=axes["scatter"])

        assert len(axes["hist"].patches) == 30
        assert len(axes["scatter"].collections) == 1
        assert axes["scatter"].get_xlabel() == "diastolic"
        assert axes["scatter"].get_ylabel() == "diabetes"
    finally:
        plt.close(fig)


def test_faraway_p5_scatter_pima_diastolic_diabetes():
    """`ggplot(pima, aes(x=diastolic, y=diabetes)) + geom_point()` (Faraway p.5)."""
    from hea.data import data as _hea_data

    pima = _hea_data("pima", package="faraway")
    p = ggplot(pima, aes(x="diastolic", y="diabetes")) + geom_point()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        assert offsets.shape == (len(pima), 2)
        assert ax.get_xlabel() == "diastolic"
        assert ax.get_ylabel() == "diabetes"
    finally:
        plt.close(fig)
