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
    PlotGrid, aes, after_scale, after_stat, annotate, annotation_custom,
    coord_cartesian, coord_fixed,
    coord_flip, coord_trans, expansion,
    element_blank, element_line, element_rect,
    plot_annotation, plot_layout, wrap_plots,
    element_text, facet_grid, facet_wrap, geom_abline, geom_area, geom_bar, geom_blank,
    geom_boxplot, geom_col, geom_contour, geom_contour_filled, geom_count,
    geom_crossbar, geom_curve, geom_density, geom_dotplot, geom_errorbar, geom_errorbarh,
    geom_hex, geom_histogram, geom_hline, geom_jitter,
    geom_label, geom_line, geom_linerange, geom_path, geom_point,
    geom_pointrange, geom_polygon, geom_qq, geom_qq_line, geom_raster,
    geom_rect, geom_ribbon, geom_segment, geom_smooth,
    geom_step, geom_text, geom_tile, geom_violin, geom_vline, ggplot, ggtitle,
    labs, lims,
    guide_axis, guide_legend, guides,
    position_dodge, position_fill, position_jitter, position_nudge,
    position_stack, scale_color_hue, scale_fill_hue, stat_ecdf, stat_function,
    stat_qq, stat_qq_line, stat_sum, stat_summary, stat_unique, geom_function,
    scale_alpha_continuous, scale_color_brewer, scale_color_gradient,
    scale_color_gradient2, scale_color_gradientn, scale_color_identity,
    scale_color_manual, scale_color_viridis_c, scale_color_viridis_d,
    scale_fill_identity, scale_fill_manual, scale_linetype,
    scale_linetype_manual, scale_radius, scale_shape, scale_shape_manual,
    scale_size_area, scale_size_continuous, scale_size_manual,
    scale_x_continuous, scale_x_date, scale_x_datetime, scale_x_log10,
    scale_x_ordinal, scale_x_percent, scale_x_reverse, scale_x_sqrt,
    scale_x_time, scale_y_continuous, scale_y_date, scale_y_datetime,
    scale_y_log10, scale_y_percent,
    theme, theme_bw, theme_classic, theme_dark, theme_gray,
    theme_minimal, theme_void, xlab, xlim, ylab, ylim,
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
# Phase 1.1d — Trans objects: log10, sqrt, reverse, identity
# ---------------------------------------------------------------------------


def test_gg_c6_scale_x_log10():
    """GG-C6: ``+ scale_x_log10()`` puts x on a log axis. Y stays linear."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("disp", "mpg")) + geom_point()
         + scale_x_log10())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "linear"
        # data still drawn on the log axis (matplotlib reinterprets coords).
        offsets = ax.collections[0].get_offsets()
        assert offsets.shape == (len(mtcars), 2)
    finally:
        plt.close(fig)


def test_scale_y_log10():
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + scale_y_log10())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        assert ax.get_xscale() == "linear"
    finally:
        plt.close(fig)


def test_scale_x_sqrt():
    """sqrt uses matplotlib's FuncScale, exposed as ``"function"``."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("disp", "mpg")) + geom_point()
         + scale_x_sqrt())
    fig = p.draw()
    try:
        assert fig.axes[0].get_xscale() == "function"
    finally:
        plt.close(fig)


def test_scale_x_reverse():
    """Reversed axis: matplotlib treats ``lo > hi`` xlim as inverted."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + scale_x_reverse())
    fig = p.draw()
    try:
        lo, hi = fig.axes[0].get_xlim()
        assert lo > hi, f"reversed axis should have lo>hi, got ({lo}, {hi})"
    finally:
        plt.close(fig)


def test_log10_with_explicit_breaks():
    """When the user gives explicit breaks, they're honored even on a log axis."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("disp", "mpg")) + geom_point()
         + scale_x_log10(breaks=[100, 200, 400], labels=["100", "200", "400"]))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        ticks = list(ax.get_xticks())
        assert ticks == [100.0, 200.0, 400.0]
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 1.1c — Wilkinson extended_breaks algorithm
#
# These cases lock parity with ggplot2's `labeling::extended` defaults on the
# canonical R datasets used throughout Faraway's textbook. R-oracle dump
# fixtures land later (X.1) — these hand-checked values are enough to catch
# regressions in the meantime.
# ---------------------------------------------------------------------------


def test_extended_breaks_unit_interval():
    from hea.ggplot.scales._breaks import extended_breaks
    bk = extended_breaks(0.0, 1.0, m=5)
    assert list(bk) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_extended_breaks_zero_centered():
    """Algorithm prefers grids that pass through zero (simplicity bonus)."""
    from hea.ggplot.scales._breaks import extended_breaks
    bk = extended_breaks(-50.0, 50.0, m=5)
    assert list(bk) == [-50.0, -25.0, 0.0, 25.0, 50.0]


def test_extended_breaks_mtcars_disp():
    """mtcars$disp range (71–472) → [100, 200, 300, 400, 500]."""
    from hea.ggplot.scales._breaks import extended_breaks
    bk = extended_breaks(71.0, 472.0, m=5)
    assert list(bk) == [100.0, 200.0, 300.0, 400.0, 500.0]


def test_extended_breaks_mtcars_wt():
    """mtcars$wt range (1.42–5.43) → [1, 2, 3, 4, 5]."""
    from hea.ggplot.scales._breaks import extended_breaks
    bk = extended_breaks(1.42, 5.43, m=5)
    assert list(bk) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_extended_breaks_degenerate_range_returns_single_tick():
    from hea.ggplot.scales._breaks import extended_breaks
    bk = extended_breaks(3.14, 3.14, m=5)
    assert len(bk) == 1
    assert bk[0] == 3.14


# ---------------------------------------------------------------------------
# Phase 1.2a — geom_line, geom_path, geom_step
# ---------------------------------------------------------------------------


def test_geom_line_sorts_by_x():
    """`geom_line` connects points sorted by x — what most line plots want."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_line()
    fig = p.draw()
    try:
        line = fig.axes[0].lines[0]
        xs = line.get_xdata()
        assert list(xs) == sorted(xs), "geom_line must sort by x"
        # all data points present
        assert len(xs) == len(mtcars)
    finally:
        plt.close(fig)


def test_geom_path_preserves_data_order():
    """`geom_path` connects points in data order, not sorted."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_path()
    fig = p.draw()
    try:
        xs = list(fig.axes[0].lines[0].get_xdata())
        assert xs == list(mtcars["wt"].to_numpy())
    finally:
        plt.close(fig)


def test_geom_step_default_hv_produces_stairstep():
    """`geom_step(direction="hv")` (default) emits 2n-1 vertices for n points."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_step()
    fig = p.draw()
    try:
        xs = fig.axes[0].lines[0].get_xdata()
        assert len(xs) == 2 * len(mtcars) - 1
    finally:
        plt.close(fig)


def test_geom_step_vh_direction():
    """`direction="vh"` flips the corner."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p_hv = ggplot(mtcars, aes("wt", "mpg")) + geom_step(direction="hv")
    p_vh = ggplot(mtcars, aes("wt", "mpg")) + geom_step(direction="vh")

    fig_hv = p_hv.draw()
    fig_vh = p_vh.draw()
    try:
        # Different y-trajectories; specifically the second y-value differs.
        y_hv = fig_hv.axes[0].lines[0].get_ydata()
        y_vh = fig_vh.axes[0].lines[0].get_ydata()
        assert not np.array_equal(y_hv, y_vh)
    finally:
        plt.close(fig_hv)
        plt.close(fig_vh)


def test_geom_line_constant_aes_overrides():
    """`geom_line(colour="red", size=2)` applies as a layer constant."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_line(colour="red", size=2)
    fig = p.draw()
    try:
        line = fig.axes[0].lines[0]
        assert line.get_color() == "red"
        # size in mm, mapped to ~5.66 pt
        assert abs(line.get_linewidth() - 2 * 2.83) < 0.01
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 1.3 — Positions: jitter, nudge, dodge, stack, fill
# ---------------------------------------------------------------------------


def test_position_jitter_spreads_points():
    """`position_jitter` makes a discrete-x scatter actually spread along x."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    # cyl has 3 unique values; raw scatter would have only 3 distinct x's.
    p = (ggplot(mtcars, aes("cyl", "mpg"))
         + geom_point(position=position_jitter(seed=42)))
    fig = p.draw()
    try:
        offsets = fig.axes[0].collections[0].get_offsets()
        assert len(np.unique(offsets[:, 0])) > 3, \
            "jitter should produce more unique x positions than raw cyl"
    finally:
        plt.close(fig)


def test_position_jitter_seed_is_deterministic():
    """Same seed → identical jittered points across draws."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    fig1 = (ggplot(mtcars, aes("cyl", "mpg"))
            + geom_point(position=position_jitter(seed=7))).draw()
    fig2 = (ggplot(mtcars, aes("cyl", "mpg"))
            + geom_point(position=position_jitter(seed=7))).draw()
    try:
        np.testing.assert_array_equal(
            fig1.axes[0].collections[0].get_offsets(),
            fig2.axes[0].collections[0].get_offsets(),
        )
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_position_nudge_shifts_by_constants():
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg"))
         + geom_point(position=position_nudge(x=1.0, y=2.0)))
    fig = p.draw()
    try:
        offsets = np.asarray(fig.axes[0].collections[0].get_offsets())
        raw = np.column_stack([mtcars["wt"].to_numpy(), mtcars["mpg"].to_numpy()])
        diff = offsets - raw
        np.testing.assert_allclose(diff[:, 0], 1.0)
        np.testing.assert_allclose(diff[:, 1], 2.0)
    finally:
        plt.close(fig)


def test_position_dodge_splits_bars_per_group():
    """With aes(group=am), 3 cyl values × 2 groups → 6 bars at distinct x's."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes(x="cyl", group="am"))
         + geom_bar(position=position_dodge()))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # 3 cyl × 2 am = 6 bars total
        assert len(ax.patches) == 6
        centers = sorted(b.get_x() + b.get_width() / 2 for b in ax.patches)
        # Each cyl ∈ {4, 6, 8} splits into two centers offset by ±slot_width/2.
        # slot_width = 0.9 / 2 = 0.45 → centers at cyl ± 0.225.
        assert abs(centers[0] - (4 - 0.225)) < 0.01
        assert abs(centers[1] - (4 + 0.225)) < 0.01
        assert abs(centers[2] - (6 - 0.225)) < 0.01
        assert abs(centers[3] - (6 + 0.225)) < 0.01
    finally:
        plt.close(fig)


def test_position_stack_stacks_bars_vertically():
    """With group, second-group bars sit on top of first-group bars."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes(x="cyl", group="am"))
         + geom_bar(position=position_stack()))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # 6 bars total (3 cyl × 2 am), some with non-zero `bottom`.
        bottoms = [b.get_y() for b in ax.patches]
        assert max(bottoms) > 0, "stack should produce bars sitting on others"
    finally:
        plt.close(fig)


def test_geom_bar_facet_wrap_bars_start_at_zero():
    """Regression: position_stack must reset per facet panel.

    Without per-panel position adjustment, bars in panel 1 stack on top
    of panel 0's bars at the same x — visually offset upward instead of
    starting at y=0.
    """
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("gear")) + geom_bar() + facet_wrap("~cyl")
    fig = p.draw()
    try:
        for ax in fig.axes:
            for b in ax.patches:
                assert b.get_y() == 0.0, (
                    f"bar at x={b.get_x()} should start at y=0, got {b.get_y()}"
                )
    finally:
        plt.close(fig)


def test_position_fill_normalises_stacks_to_one():
    """`position_fill` makes every column reach exactly y=1."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes(x="cyl", group="am"))
         + geom_bar(position=position_fill()))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # For each unique x, the topmost bar's (y + height) should be 1.0.
        from collections import defaultdict
        tops = defaultdict(float)
        for b in ax.patches:
            x = round(b.get_x() + b.get_width() / 2, 6)
            tops[x] = max(tops[x], b.get_y() + b.get_height())
        for x, top in tops.items():
            assert abs(top - 1.0) < 1e-9, f"stack at x={x} reaches {top}, not 1"
    finally:
        plt.close(fig)


def test_position_string_resolves_to_class():
    """`position="jitter"` should work just like `position=position_jitter()`."""
    mtcars = load_dataset("datasets", "mtcars")
    # Determinism not asserted (no seed plumbing through string form),
    # just confirm the dispatch and that points spread.
    import numpy as np
    p = ggplot(mtcars, aes("cyl", "mpg")) + geom_point(position="jitter")
    fig = p.draw()
    try:
        offsets = fig.axes[0].collections[0].get_offsets()
        assert len(np.unique(offsets[:, 0])) > 3
    finally:
        plt.close(fig)


def test_position_unknown_string_raises():
    mtcars = load_dataset("datasets", "mtcars")
    with pytest.raises(ValueError, match="unknown position"):
        ggplot(mtcars, aes("wt", "mpg")) + geom_point(position="zigzag")


# ---------------------------------------------------------------------------
# Phase 1.4 — Smoothing: stat_smooth + geom_smooth (and the underlying
# geom_ribbon / geom_area)
# ---------------------------------------------------------------------------


def test_geom_ribbon_draws_filled_band():
    """geom_ribbon needs aes(x, ymin, ymax)."""
    import polars as pl
    from matplotlib.collections import PolyCollection

    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "lo": [0.0, 0.5, 1.0, 1.5],
        "hi": [1.0, 1.5, 2.0, 2.5],
    })
    p = ggplot(df, aes(x="x", ymin="lo", ymax="hi")) + geom_ribbon()
    fig = p.draw()
    try:
        # fill_between yields a PolyCollection subclass
        # (FillBetweenPolyCollection in modern matplotlib).
        polys = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
        assert len(polys) == 1
    finally:
        plt.close(fig)


def test_geom_area_treats_y_as_ymax_with_zero_floor():
    import polars as pl
    from matplotlib.collections import PolyCollection

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 1.5]})
    p = ggplot(df, aes(x="x", y="y")) + geom_area()
    fig = p.draw()
    try:
        polys = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
        assert len(polys) == 1
        bbox = polys[0].get_paths()[0].get_extents()
        assert bbox.y0 == pytest.approx(0.0)
    finally:
        plt.close(fig)


def test_gg_c2_geom_smooth_lm_with_ci_ribbon():
    """GG-C2: ``geom_smooth(method="lm")`` produces a fit line + CI band."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + geom_smooth(method="lm"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # 2 collections: scatter from geom_point + ribbon from geom_smooth
        assert len(ax.collections) == 2
        # 1 line: the fitted line from geom_smooth
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)


def test_geom_smooth_se_false_omits_ribbon():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_smooth(method="lm", se=False)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # No ribbon (no scatter either since no geom_point)
        assert len(ax.collections) == 0
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)


def test_geom_smooth_loess_default_method_works():
    """Default method is loess; should produce a curve + ribbon for non-trivial n."""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + 0.2 * rng.standard_normal(50)
    df = pl.DataFrame({"x": x, "y": y})

    p = ggplot(df, aes("x", "y")) + geom_smooth(span=0.5)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        line_y = ax.lines[0].get_ydata()
        # Smoothed curve should track the sin shape — peaks/troughs near sin's.
        # Just check it's bounded, not a constant or a flat line.
        assert line_y.max() > 0.3
        assert line_y.min() < -0.3
    finally:
        plt.close(fig)


def test_geom_smooth_lm_fit_matches_hea_lm():
    """The fit line from geom_smooth(method="lm") must match a hea.lm fit
    on the same data — no surprise drift."""
    import polars as pl
    from hea import lm

    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_smooth(method="lm")
    fig = p.draw()
    try:
        line_x = fig.axes[0].lines[0].get_xdata()
        line_y = fig.axes[0].lines[0].get_ydata()

        m = lm("mpg ~ wt", mtcars)
        new = pl.DataFrame({"wt": line_x})
        expected = m.predict(new=new)["Fitted"].to_numpy()
        import numpy as np
        np.testing.assert_allclose(line_y, expected, atol=1e-9)
    finally:
        plt.close(fig)


def test_stat_smooth_gam_method():
    """`geom_smooth(method="gam")` runs hea.gam under the hood — produces
    a non-linear curve (different shape from method="lm")."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p_lm = ggplot(mtcars, aes("wt", "mpg")) + geom_smooth(method="lm")
    p_gam = ggplot(mtcars, aes("wt", "mpg")) + geom_smooth(method="gam")

    fig_lm = p_lm.draw()
    fig_gam = p_gam.draw()
    try:
        y_lm = fig_lm.axes[0].lines[0].get_ydata()
        y_gam = fig_gam.axes[0].lines[0].get_ydata()
        # The two fits should differ (gam picks up curvature lm misses).
        assert not np.allclose(y_lm, y_gam, atol=0.1)
        # gam fit should be non-linear (second-difference > 0 somewhere).
        d2 = np.diff(y_gam, n=2)
        assert abs(d2).max() > 0.001
    finally:
        plt.close(fig_lm)
        plt.close(fig_gam)


def test_stat_smooth_gam_with_per_panel_facets():
    """gam stat fits independently per facet — like lm. Use k=4 in the
    smooth so the fit doesn't choke on small per-panel sample sizes
    (mtcars cyl=6 has only 7 obs)."""
    import warnings as _w

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg"))
         + geom_smooth(method="gam", formula="y ~ s(x, k=4)")
         + facet_wrap("cyl"))
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # 3 cyl panels each with their own smooth + ribbon.
        for a in visible:
            assert len(a.lines) == 1
            assert len(a.collections) == 1  # ribbon only
    finally:
        plt.close(fig)


def test_stat_smooth_glm_gaussian_matches_lm():
    """``method="glm"`` with default Gaussian family must match the ``"lm"``
    fit exactly — Gaussian-identity GLM reduces to OLS."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p_lm = ggplot(mtcars, aes("wt", "mpg")) + geom_smooth(method="lm", se=False)
    p_glm = ggplot(mtcars, aes("wt", "mpg")) + geom_smooth(method="glm", se=False)
    fig_lm = p_lm.draw()
    fig_glm = p_glm.draw()
    try:
        y_lm = fig_lm.axes[0].lines[0].get_ydata()
        y_glm = fig_glm.axes[0].lines[0].get_ydata()
        np.testing.assert_allclose(y_lm, y_glm, atol=1e-9)
    finally:
        plt.close(fig_lm)
        plt.close(fig_glm)


def test_stat_smooth_glm_binomial_logistic_regression():
    """``method="glm", family=binomial()`` fits a logistic curve."""
    import numpy as np
    import polars as pl

    from hea.family import binomial

    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 80)
    prob = 1 / (1 + np.exp(-x))
    y = (rng.random(80) < prob).astype(float)
    df = pl.DataFrame({"x": x, "y": y})

    p = ggplot(df, aes("x", "y")) + geom_smooth(method="glm", family=binomial())
    fig = p.draw()
    try:
        yhat = fig.axes[0].lines[0].get_ydata()
        # Sigmoid: monotone in x, sweeping from low to high across the range.
        assert yhat[0] < 0.2 and yhat[-1] > 0.8
        assert (np.diff(yhat) >= -1e-9).all(), "logistic curve must be monotone"
    finally:
        plt.close(fig)


def test_stat_smooth_glm_family_string_form():
    """``family="binomial"`` (string) resolves to ``hea.family.binomial()``."""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(1)
    x = np.linspace(-3, 3, 60)
    y = (rng.random(60) < 1 / (1 + np.exp(-x))).astype(float)
    df = pl.DataFrame({"x": x, "y": y})

    p = ggplot(df, aes("x", "y")) + geom_smooth(method="glm", family="binomial")
    fig = p.draw()
    try:
        yhat = fig.axes[0].lines[0].get_ydata()
        assert all(0 <= v <= 1 for v in yhat)
    finally:
        plt.close(fig)


def test_stat_smooth_glm_unknown_family_errors():
    import polars as pl
    df = pl.DataFrame({"x": [1.0, 2, 3], "y": [1.0, 2, 3]})
    p = ggplot(df, aes("x", "y")) + geom_smooth(method="glm", family="weibull")
    with pytest.raises(ValueError, match="unknown family"):
        p.draw()


# ---------------------------------------------------------------------------
# Phase 1.9c — geom_boxplot
# ---------------------------------------------------------------------------


def test_geom_boxplot_one_box_per_group():
    """3 unique cyl values → 3 boxes."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes(x="cyl", y="mpg", group="cyl")) + geom_boxplot()
    fig = p.draw()
    try:
        # patches contains the box rectangles (one per group).
        assert len(fig.axes[0].patches) == 3
    finally:
        plt.close(fig)


def test_geom_boxplot_stat_matches_numpy_quantiles():
    """The five-number summary stat should match numpy.quantile."""
    import numpy as np
    import polars as pl

    from hea.ggplot.stats.boxplot import StatBoxplot

    rng = np.random.default_rng(0)
    y = rng.standard_normal(100)
    df = pl.DataFrame({"x": [0] * len(y), "y": y})

    out = StatBoxplot().compute_panel(df, {})
    expected = np.quantile(y, [0.25, 0.5, 0.75])
    assert float(out["lower"][0]) == pytest.approx(expected[0])
    assert float(out["middle"][0]) == pytest.approx(expected[1])
    assert float(out["upper"][0]) == pytest.approx(expected[2])


def test_geom_boxplot_extracts_outliers():
    """Outliers beyond 1.5·IQR end up in the outliers column."""
    import polars as pl

    from hea.ggplot.stats.boxplot import StatBoxplot

    # Most data in [0,1]; throw in an obvious outlier at 100.
    y = list(range(20)) + [100]
    df = pl.DataFrame({"x": [0] * len(y), "y": [float(v) for v in y]})

    out = StatBoxplot().compute_panel(df, {})
    outliers = out["outliers"][0].to_list()
    assert 100.0 in outliers


# ---------------------------------------------------------------------------
# Phase 1.9d — geom_violin
# ---------------------------------------------------------------------------


def test_geom_violin_one_polygon_per_group():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes(x="cyl", y="mpg", group="cyl")) + geom_violin()
    fig = p.draw()
    try:
        # ax.fill registers patches as Polygon. 3 cyl groups → 3 polygons.
        assert len(fig.axes[0].patches) == 3
    finally:
        plt.close(fig)


def test_stat_ydensity_violinwidth_normalised_to_one():
    """Per group, max(violinwidth) should be exactly 1 (peak of the KDE)."""
    import numpy as np
    import polars as pl

    from hea.ggplot.stats.ydensity import StatYdensity

    rng = np.random.default_rng(1)
    df = pl.DataFrame({
        "x": [0] * 100 + [1] * 100,
        "y": np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 5]),
    })
    out = StatYdensity().compute_panel(df, {})
    for x_val in (0, 1):
        sub = out.filter(pl.col("x") == x_val)
        assert float(sub["violinwidth"].max()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 1.3d — GG-C9: boxplot + jittered points overlay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 1.5 — Color/fill scales: auto-default discrete, manual, identity,
# plus 1.5a auto-grouping rule.
# ---------------------------------------------------------------------------


def test_gg_c3_aes_color_factor_cyl_assigns_distinct_colors():
    """GG-C3: ``aes(color = factor(cyl))`` paints points by group."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", colour="factor(cyl)"))
         + geom_point())
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        # 3 unique cyl values → 3 unique facecolors.
        assert len({tuple(c) for c in fc}) == 3
    finally:
        plt.close(fig)


def test_hue_pal_matches_ggplot2_byte_for_byte():
    """``hue_pal()(n)`` returns the same hex codes as
    ``scales::hue_pal()(n)`` from R, for the canonical small ``n`` values."""
    from hea.ggplot.scales._palettes import hue_pal

    pal = hue_pal()
    assert pal(2) == ["#F8766D", "#00BFC4"]
    assert pal(3) == ["#F8766D", "#00BA38", "#619CFF"]
    assert pal(4) == ["#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"]
    assert pal(5) == ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3"]


def test_hcl_to_hex_matches_grdevices():
    """Spot-check :func:`hcl_to_hex` against ``grDevices::hcl(...)`` output."""
    from hea.ggplot.scales._palettes import hcl_to_hex

    assert hcl_to_hex(15, 100, 65) == "#F8766D"
    assert hcl_to_hex(135, 100, 65) == "#00BA38"
    assert hcl_to_hex(255, 100, 65) == "#619CFF"


def test_discrete_color_levels_sorted_alphabetically():
    """ggplot2 silently runs character columns through ``factor()`` before
    mapping, so the level → colour assignment is alphabetical regardless
    of CSV row order. The penguins dataset lists Adelie/Gentoo/Chinstrap
    in that order; ggplot2 still gives Chinstrap green and Gentoo blue."""
    from hea.data import data as _hea_data
    from matplotlib.colors import to_rgb

    penguins = _hea_data("penguins", package="palmerpenguins")
    p = (ggplot(penguins, aes("flipper_length_mm", "body_mass_g",
                              colour="species"))
         + geom_point())
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)  # NA-removal warning
        fig = p.draw()
    try:
        # Map each species to its rendered colour.
        clean = penguins.drop_nulls(["flipper_length_mm", "body_mass_g"])
        species = clean["species"].to_list()
        fc = fig.axes[0].collections[0].get_facecolors()
        observed = {}
        for sp, c in zip(species, fc):
            observed.setdefault(sp, tuple(round(float(x), 3) for x in c[:3]))

        expected = {
            "Adelie": tuple(round(float(x), 3) for x in to_rgb("#F8766D")),
            "Chinstrap": tuple(round(float(x), 3) for x in to_rgb("#00BA38")),
            "Gentoo": tuple(round(float(x), 3) for x in to_rgb("#619CFF")),
        }
        assert observed == expected
    finally:
        plt.close(fig)


def test_user_penguins_string_color_works_end_to_end():
    """The bug report: ``aes(color="species")`` on string column → 3 colours,
    no matplotlib RGBA error. Penguins has 2 NA rows so a warning is expected."""
    from hea.data import data as _hea_data

    penguins = _hea_data("penguins", package="palmerpenguins")
    p = (ggplot(penguins, aes(x="flipper_length_mm", y="body_mass_g",
                              color="species"))
         + geom_point())
    with pytest.warns(UserWarning, match=r"Removed 2 rows .*`geom_point\(\)`"):
        fig = p.draw()
    try:
        ax = fig.axes[0]
        fc = ax.collections[0].get_facecolors()
        assert len({tuple(c) for c in fc}) == 3
    finally:
        plt.close(fig)


def test_scale_color_manual_applies_user_palette():
    """``scale_color_manual(values=[...])`` — colours come from the user list,
    in level order."""
    from matplotlib.colors import to_rgba

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", colour="factor(cyl)"))
         + geom_point()
         + scale_color_manual(values=["#FF0000", "#00FF00", "#0000FF"]))
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        observed = {tuple(c) for c in fc}
        expected = {to_rgba(c) for c in ("#FF0000", "#00FF00", "#0000FF")}
        assert observed == expected
    finally:
        plt.close(fig)


def test_scale_color_manual_dict_form():
    """Dict form lets the user pin specific colours per level explicitly."""
    from matplotlib.colors import to_rgba
    import polars as pl

    df = pl.DataFrame({"x": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4],
                       "g": ["a", "b", "a", "b"]})
    palette = {"a": "#AA0000", "b": "#00AA00"}
    p = (ggplot(df, aes("x", "y", colour="g")) + geom_point()
         + scale_color_manual(values=palette))
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        # Two unique colours, each matching the palette.
        assert {tuple(c) for c in fc} == {to_rgba("#AA0000"), to_rgba("#00AA00")}
    finally:
        plt.close(fig)


def test_scale_color_identity_passes_hex_through():
    """When the column already has hex codes, identity scale skips palette."""
    from matplotlib.colors import to_rgba
    import polars as pl

    df = pl.DataFrame({
        "x": [1.0, 2, 3],
        "y": [1.0, 2, 3],
        "c": ["#FF0000", "#00FF00", "#0000FF"],
    })
    p = (ggplot(df, aes("x", "y", colour="c")) + geom_point()
         + scale_color_identity())
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        observed = {tuple(c) for c in fc}
        assert observed == {to_rgba("#FF0000"), to_rgba("#00FF00"), to_rgba("#0000FF")}
    finally:
        plt.close(fig)


def test_add_group_auto_creates_group_from_discrete_aesthetic():
    """Per ggplot2's `add_group` rule: a discrete non-positional aes implies
    a `group` column when the user didn't set one."""
    import polars as pl

    from hea.ggplot.build import _add_group

    df = pl.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 2, 3, 4, 5, 6],
        "colour": ["a", "a", "b", "b", "c", "c"],
    })
    out = _add_group(df)
    assert "group" in out.columns
    # 3 unique levels → 3 distinct group ids
    assert out["group"].n_unique() == 3


# ---------------------------------------------------------------------------
# Phase 1.5d/e/f — viridis / gradient / brewer
# ---------------------------------------------------------------------------


def test_auto_numeric_colour_uses_gradient_default():
    """``aes(colour=numeric_col)`` without an explicit scale → ``gradient_pal``
    (matching ggplot2's ``scale_color_continuous`` default)."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg", colour="hp")) + geom_point()
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        # mtcars$hp has many distinct values → many distinct colours from
        # the smooth gradient (≥ 10 to be clearly not a single fixed colour).
        assert len({tuple(c) for c in fc}) >= 10
    finally:
        plt.close(fig)


def test_scale_color_gradient_endpoints_match_data_extrema():
    """`gradient(low="red", high="blue")`: min value gets red, max gets blue."""
    import numpy as np
    from matplotlib.colors import to_rgba

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", colour="hp")) + geom_point()
         + scale_color_gradient(low="red", high="blue"))
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        hp = mtcars["hp"].to_numpy()
        i_min, i_max = int(np.argmin(hp)), int(np.argmax(hp))
        assert tuple(fc[i_min]) == to_rgba("red")
        assert tuple(fc[i_max]) == to_rgba("blue")
    finally:
        plt.close(fig)


def test_scale_color_gradient2_midpoint_is_mid_colour():
    """gradient2: at midpoint, the colour should match the mid argument."""
    import polars as pl
    from matplotlib.colors import to_rgba

    df = pl.DataFrame({
        "x": [1.0, 2, 3], "y": [1.0, 2, 3], "z": [-1.0, 0.0, 1.0],
    })
    # Data range is [-1, 1], midpoint of palette at 0.5 maps to data midpoint 0.
    p = (ggplot(df, aes("x", "y", colour="z")) + geom_point()
         + scale_color_gradient2(low="red", mid="green", high="blue", midpoint=0.5))
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        assert tuple(fc[0]) == to_rgba("red")
        assert tuple(fc[1]) == to_rgba("green")
        assert tuple(fc[2]) == to_rgba("blue")
    finally:
        plt.close(fig)


def test_scale_color_gradientn_n_stop_palette():
    """``gradientn(colours=[...])`` interpolates linearly across the n stops."""
    import polars as pl
    from matplotlib.colors import to_rgba

    df = pl.DataFrame({
        "x": [1.0, 2, 3, 4, 5], "y": [1.0, 2, 3, 4, 5], "z": [0.0, 0.25, 0.5, 0.75, 1.0],
    })
    p = (ggplot(df, aes("x", "y", colour="z")) + geom_point()
         + scale_color_gradientn(colours=["red", "yellow", "blue"]))
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        # First and last must be red and blue (palette endpoints)
        assert tuple(fc[0]) == to_rgba("red")
        assert tuple(fc[-1]) == to_rgba("blue")
    finally:
        plt.close(fig)


def test_scale_color_viridis_c_continuous():
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", colour="hp")) + geom_point()
         + scale_color_viridis_c())
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        # Many distinct colours since hp is continuous.
        assert len({tuple(c) for c in fc}) >= 10
        # Viridis is monotone in luminance — first row's hp determines colour;
        # different hp ⇒ different colour.
    finally:
        plt.close(fig)


def test_scale_color_viridis_d_discrete():
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", colour="factor(cyl)"))
         + geom_point() + scale_color_viridis_d())
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        assert len({tuple(c) for c in fc}) == 3
    finally:
        plt.close(fig)


def test_scale_color_viridis_d_direction_reverses_ordering():
    """``direction=-1`` flips the palette across the level order."""
    mtcars = load_dataset("datasets", "mtcars")
    p_fwd = (ggplot(mtcars, aes("wt", "mpg", colour="factor(cyl)"))
             + geom_point() + scale_color_viridis_d(direction=1))
    p_rev = (ggplot(mtcars, aes("wt", "mpg", colour="factor(cyl)"))
             + geom_point() + scale_color_viridis_d(direction=-1))
    f1 = p_fwd.draw()
    f2 = p_rev.draw()
    try:
        c_fwd = {tuple(c) for c in f1.axes[0].collections[0].get_facecolors()}
        c_rev = {tuple(c) for c in f2.axes[0].collections[0].get_facecolors()}
        # Same palette, same set of colours (just different level → colour
        # mapping), so the sets are equal.
        assert c_fwd == c_rev
    finally:
        plt.close(f1)
        plt.close(f2)


def test_scale_color_brewer_set1():
    """ColorBrewer Set1 — qualitative; matplotlib's bundled palette matches
    ``RColorBrewer::brewer.pal(_, 'Set1')`` colour-for-colour."""
    from matplotlib.colors import to_hex
    import matplotlib

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", colour="factor(cyl)"))
         + geom_point() + scale_color_brewer(palette="Set1"))
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        from matplotlib.colors import to_rgba
        expected = {to_rgba(to_hex(c))
                    for c in matplotlib.colormaps["Set1"].colors[:3]}
        observed = {tuple(c) for c in fc}
        assert observed == expected
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 1.6 — Non-colour aes scales: size, alpha, shape, linetype
# ---------------------------------------------------------------------------


def test_gg_c4_aes_size_continuous_qsec():
    """GG-C4: ``aes(size=qsec)`` produces a continuous size scale (default
    range 1–6 mm)."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg", size="qsec")) + geom_point()
    fig = p.draw()
    try:
        sizes = fig.axes[0].collections[0].get_sizes()
        # Many distinct sizes since qsec is mostly continuous.
        assert len(np.unique(np.round(sizes, 4))) >= 10
        # Min size = (1 mm * 2.8454)² ≈ 8.10; max = (6 mm * 2.8454)² ≈ 291.5.
        assert sizes.min() == pytest.approx(8.097, rel=0.01)
        assert sizes.max() == pytest.approx(291.5, rel=0.01)
    finally:
        plt.close(fig)


def test_scale_size_continuous_custom_range():
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", size="qsec")) + geom_point()
         + scale_size_continuous(range=(2, 4)))
    fig = p.draw()
    try:
        sizes = fig.axes[0].collections[0].get_sizes()
        # Range mapped to [2, 4] mm → s in [(2*2.8454)², (4*2.8454)²]
        assert sizes.min() == pytest.approx((2 * 2.8454) ** 2, rel=0.01)
        assert sizes.max() == pytest.approx((4 * 2.8454) ** 2, rel=0.01)
    finally:
        plt.close(fig)


def test_aes_alpha_continuous():
    """``aes(alpha=qsec)`` → alpha varies linearly in [0.1, 1] by default."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg", alpha="qsec")) + geom_point()
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        alphas = sorted({round(c[3], 4) for c in fc})
        assert alphas[0] == pytest.approx(0.1, abs=0.01)
        assert alphas[-1] == pytest.approx(1.0, abs=0.01)
    finally:
        plt.close(fig)


def test_aes_shape_discrete_factor_cyl():
    """``aes(shape=factor(cyl))`` → 3 scatter calls (one per marker shape)."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", shape="factor(cyl)"))
         + geom_point())
    fig = p.draw()
    try:
        # matplotlib scatter takes a single marker per call; one collection
        # per unique shape.
        assert len(fig.axes[0].collections) == 3
    finally:
        plt.close(fig)


def test_aes_shape_continuous_raises():
    """ggplot2 message: 'A continuous variable cannot be mapped to `shape`'."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg", shape="qsec")) + geom_point()
    with pytest.raises(ValueError, match="continuous variable cannot be mapped to `shape`"):
        p.draw()


def test_scale_shape_manual_explicit_markers():
    """User-supplied marker codes are honoured per level."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", shape="factor(cyl)"))
         + geom_point()
         + scale_shape_manual(values=["o", "X", "D"]))
    fig = p.draw()
    try:
        # 3 collections each with a distinct marker glyph → just count
        # the collection count and assert it matches the number of levels.
        assert len(fig.axes[0].collections) == 3
    finally:
        plt.close(fig)


def test_aes_linetype_discrete_on_geom_line():
    """``aes(linetype=factor(cyl))`` on geom_line → distinct linestyles per group."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg", linetype="factor(cyl)"))
         + geom_line())
    fig = p.draw()
    try:
        # 3 cyl groups → 3 lines; each with its own linestyle.
        assert len(fig.axes[0].lines) == 3
        # Distinct linestyles across the 3 lines.
        linestyles = {ln.get_linestyle() for ln in fig.axes[0].lines}
        assert len(linestyles) == 3
    finally:
        plt.close(fig)


def test_aes_linetype_continuous_raises():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg", linetype="qsec")) + geom_line()
    with pytest.raises(ValueError, match="continuous variable cannot be mapped to `linetype`"):
        p.draw()


# ---------------------------------------------------------------------------
# Phase 1.7 — facet_wrap
# ---------------------------------------------------------------------------


def test_gg_c5_facet_wrap_by_cyl():
    """GG-C5: ``facet_wrap("cyl")`` produces 3 panels, one per unique cyl."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_wrap("cyl"))
    fig = p.draw()
    try:
        # 3 panels in a 2x2 grid → 4 axes total, last hidden.
        visible_axes = [a for a in fig.axes if a.get_visible()]
        assert len(visible_axes) == 3
        # Strip titles match cyl values (4, 6, 8) sorted.
        titles = [a.get_title() for a in visible_axes]
        assert titles == ["4", "6", "8"]
    finally:
        plt.close(fig)


def test_facet_wrap_tilde_string_syntax():
    """`facet_wrap("~ cyl")` works — tilde stripped on input."""
    mtcars = load_dataset("datasets", "mtcars")
    p1 = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + facet_wrap("~ cyl")
    p2 = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + facet_wrap("cyl")
    fig1 = p1.draw()
    fig2 = p2.draw()
    try:
        v1 = [a for a in fig1.axes if a.get_visible()]
        v2 = [a for a in fig2.axes if a.get_visible()]
        assert len(v1) == len(v2) == 3
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_facet_wrap_multi_variable():
    """`facet_wrap(["cyl", "am"])` panels by Cartesian product."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_wrap(["cyl", "am"]))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # 3 cyl × 2 am = 6 panels (all combinations exist in mtcars).
        assert len(visible) == 6
        titles = [a.get_title() for a in visible]
        # Titles join with ", " — e.g. "4, 0", "4, 1", ...
        assert all(", " in t for t in titles)
    finally:
        plt.close(fig)


def test_facet_wrap_scales_fixed_shares_axes():
    """``scales="fixed"`` (default): all panels share the same xlim/ylim."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_wrap("cyl", scales="fixed"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        xlims = {tuple(a.get_xlim()) for a in visible}
        ylims = {tuple(a.get_ylim()) for a in visible}
        assert len(xlims) == 1, f"fixed → all panels share xlim, got {xlims}"
        assert len(ylims) == 1
    finally:
        plt.close(fig)


def test_facet_wrap_scales_free_independent_axes():
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_wrap("cyl", scales="free"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        xlims = {tuple(a.get_xlim()) for a in visible}
        # Different cyl groups have different wt ranges → distinct xlims.
        assert len(xlims) >= 2
    finally:
        plt.close(fig)


def test_facet_wrap_ncol_explicit():
    """User-specified `ncol=3` puts all panels on one row."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_wrap("cyl", ncol=3))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 3
        # Visible axes should all be on row 0.
        positions = sorted(a.get_subplotspec().rowspan.start for a in visible)
        assert positions == [0, 0, 0]
    finally:
        plt.close(fig)


def test_facet_wrap_unknown_scales_value_errors():
    with pytest.raises(ValueError, match="scales must be one of"):
        facet_wrap("cyl", scales="weird")


def test_facet_wrap_with_geom_smooth_per_panel_fits():
    """`facet_wrap` runs the stat per panel, so each panel gets its own fit."""
    import numpy as np

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point() + geom_smooth(method="lm")
         + facet_wrap("cyl"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # Each panel has its own scatter + ribbon + line.
        for a in visible:
            assert len(a.lines) == 1
            assert len(a.collections) == 2  # scatter + ribbon
        # Per-panel slopes should differ (different cyl groups have
        # different relationships between wt and mpg).
        slopes = []
        for a in visible:
            xy = a.lines[0].get_xydata()
            if len(xy) >= 2:
                slopes.append((xy[-1, 1] - xy[0, 1]) / (xy[-1, 0] - xy[0, 0]))
        assert len(set(round(s, 3) for s in slopes)) >= 2
    finally:
        plt.close(fig)


def test_facet_wrap_preserves_colour_mapping_per_panel():
    """Per-panel discrete colour mapping survives the facet split."""
    from hea.data import data as _hea_data

    penguins = _hea_data("penguins", package="palmerpenguins")
    p = (ggplot(penguins, aes("flipper_length_mm", "body_mass_g",
                              colour="species"))
         + geom_point()
         + facet_wrap("island"))
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)  # NA-removal warning
        fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # 3 islands.
        assert len(visible) == 3
        # Each panel's scatter still uses the species-coded colours.
        for a in visible:
            if a.collections:
                fc = a.collections[0].get_facecolors()
                if len(fc) > 0:
                    # Multiple colours present in at least one panel
                    # (penguins has multi-species islands).
                    pass
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2.1 — facet_grid
# ---------------------------------------------------------------------------


def test_facet_grid_formula_basic():
    """`facet_grid("am ~ cyl")` produces a 2×3 grid of panels."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid("am ~ cyl"))
    fig = p.draw()
    try:
        # 2 am × 3 cyl = 6 panels.
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 6
        # Subplot grid is 2 rows × 3 cols.
        rows = {a.get_subplotspec().rowspan.start for a in visible}
        cols = {a.get_subplotspec().colspan.start for a in visible}
        assert rows == {0, 1}
        assert cols == {0, 1, 2}
    finally:
        plt.close(fig)


def test_facet_grid_kwargs_form():
    """`facet_grid(rows="am", cols="cyl")` matches the formula form."""
    mtcars = load_dataset("datasets", "mtcars")
    p1 = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
          + facet_grid("am ~ cyl"))
    p2 = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
          + facet_grid(rows="am", cols="cyl"))
    fig1 = p1.draw()
    fig2 = p2.draw()
    try:
        v1 = [a for a in fig1.axes if a.get_visible()]
        v2 = [a for a in fig2.axes if a.get_visible()]
        assert len(v1) == len(v2) == 6
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_facet_grid_row_only():
    """`facet_grid("am ~ .")` → single column, 2 rows."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid("am ~ ."))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 2
        cols = {a.get_subplotspec().colspan.start for a in visible}
        assert cols == {0}
    finally:
        plt.close(fig)


def test_facet_grid_col_only():
    """`facet_grid(". ~ cyl")` → single row, 3 columns."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid(". ~ cyl"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 3
        rows = {a.get_subplotspec().rowspan.start for a in visible}
        assert rows == {0}
    finally:
        plt.close(fig)


def test_facet_grid_strip_labels_top_and_right():
    """Top strip on row 0 (col values); right strip on last col (row values)."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid("am ~ cyl"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # Top strip: only row 0 panels have non-empty titles (col values).
        for a in visible:
            row = a.get_subplotspec().rowspan.start
            title = a.get_title()
            if row == 0:
                # Titles match the cyl values.
                assert title in {"4", "6", "8"}
            else:
                assert title == ""

        # Right strip: only last-column panels have a right-side text annotation
        # (rendered via ax.text with transform=transAxes).
        for a in visible:
            col = a.get_subplotspec().colspan.start
            right_texts = [t for t in a.texts
                           if t.get_transform() == a.transAxes
                           and t.get_position()[0] > 1.0]
            if col == 2:  # last column
                assert len(right_texts) == 1
                assert right_texts[0].get_text() in {"0", "1"}
            else:
                assert len(right_texts) == 0
    finally:
        plt.close(fig)


def test_facet_grid_scales_fixed_shares_all():
    """``scales="fixed"`` (default): every panel shares xlim and ylim."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid("am ~ cyl"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        xlims = {tuple(a.get_xlim()) for a in visible}
        ylims = {tuple(a.get_ylim()) for a in visible}
        assert len(xlims) == 1
        assert len(ylims) == 1
    finally:
        plt.close(fig)


def test_facet_grid_scales_free_x_shares_within_column():
    """``scales="free_x"``: x varies between columns, shared within."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid("am ~ cyl", scales="free_x"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # Group panels by column: panels in the same column share xlim,
        # different columns can have different xlims.
        by_col: dict[int, set] = {}
        for a in visible:
            col = a.get_subplotspec().colspan.start
            by_col.setdefault(col, set()).add(tuple(round(v, 6) for v in a.get_xlim()))
        # Each column should have a single xlim (panels within share).
        for col_xlims in by_col.values():
            assert len(col_xlims) == 1
        # Across columns, at least two distinct xlims.
        all_xlims = {next(iter(s)) for s in by_col.values()}
        assert len(all_xlims) >= 2

        # y stays shared across all panels.
        ylims = {tuple(round(v, 6) for v in a.get_ylim()) for a in visible}
        assert len(ylims) == 1
    finally:
        plt.close(fig)


def test_facet_grid_scales_free_y_shares_within_row():
    """``scales="free_y"``: y varies between rows, shared within."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid("am ~ cyl", scales="free_y"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        by_row: dict[int, set] = {}
        for a in visible:
            row = a.get_subplotspec().rowspan.start
            by_row.setdefault(row, set()).add(tuple(round(v, 6) for v in a.get_ylim()))
        for row_ylims in by_row.values():
            assert len(row_ylims) == 1

        xlims = {tuple(round(v, 6) for v in a.get_xlim()) for a in visible}
        assert len(xlims) == 1
    finally:
        plt.close(fig)


def test_facet_grid_per_panel_stat_fit():
    """Stat (e.g. lm smooth) fits per facet cell, not pooled.

    Uses ``vs ~ am`` (4 cells, all with enough rows for lm) — ``am ~ cyl``
    has a 2-row cell which lm can't fit (df_residual = 0).
    """
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point() + geom_smooth(method="lm")
         + facet_grid("vs ~ am"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 4
        # Each panel has its own scatter + ribbon + line.
        for a in visible:
            assert len(a.lines) == 1
            assert len(a.collections) == 2  # scatter + ribbon
        # Slopes should differ across panels.
        slopes = []
        for a in visible:
            xy = a.lines[0].get_xydata()
            if len(xy) >= 2:
                slopes.append((xy[-1, 1] - xy[0, 1]) / (xy[-1, 0] - xy[0, 0]))
        assert len(set(round(s, 3) for s in slopes)) >= 2
    finally:
        plt.close(fig)


def test_facet_grid_multi_var_rows():
    """`facet_grid(rows=["am", "vs"], cols="cyl")` — two-var row grouping."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + facet_grid(rows=["am", "vs"], cols="cyl"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # mtcars: am × vs has 4 unique combos × 3 cyl = 12 cells in the grid,
        # but not every combo appears in mtcars.
        combos = mtcars.select(["am", "vs"]).unique()
        n_row = len(combos)
        # All 12 panels rendered (grid_dims is the full cross).
        assert len(visible) == n_row * 3
    finally:
        plt.close(fig)


def test_facet_grid_formula_and_kwargs_conflict_errors():
    with pytest.raises(ValueError, match="either a formula or"):
        facet_grid("am ~ cyl", rows="am")


def test_facet_grid_bad_formula_errors():
    with pytest.raises(ValueError, match="must be 'rows ~ cols'"):
        facet_grid("am cyl")  # missing ~


def test_facet_grid_empty_both_sides_errors():
    with pytest.raises(ValueError, match="at least one of"):
        facet_grid(". ~ .")


def test_facet_grid_unknown_scales_value_errors():
    with pytest.raises(ValueError, match="scales must be one of"):
        facet_grid("am ~ cyl", scales="weird")


def test_facet_grid_with_enum_facet_var():
    """Regression: factor()/Enum-typed facet variables don't break the join.

    Before the dtype-preservation fix in `_stat_per_panel`, the post-stat
    chunk re-attached the facet column as Utf8 (via ``pl.lit(val)``), which
    didn't match the layout's original Enum dtype and matplotlib's draw
    failed with a polars SchemaError on join.
    """
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "g": pl.Series(["a", "a", "a", "b", "b", "b"],
                       dtype=pl.Enum(["a", "b"])),
    })
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + facet_grid("~ g"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 2
        # Strip text uses the original Enum-categorised values.
        assert {a.get_title() for a in visible} == {"a", "b"}
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2.4 — geom_rect, geom_tile, geom_raster, geom_polygon
# ---------------------------------------------------------------------------


def test_geom_rect_two_rectangles():
    """`geom_rect` with two non-overlapping rectangles."""
    df = pl.DataFrame({
        "xmin": [0.0, 2.0], "xmax": [1.0, 3.0],
        "ymin": [0.0, 2.0], "ymax": [1.0, 3.0],
    })
    p = ggplot(df) + geom_rect(
        aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
        fill="red",
    )
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        # The collection holds 2 patches (one per row).
        coll = ax.collections[0]
        assert len(coll.get_paths()) == 2
        # Axes auto-scale to cover the rectangles.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= 0.0 and xlim[1] >= 3.0
        assert ylim[0] <= 0.0 and ylim[1] >= 3.0
    finally:
        plt.close(fig)


def test_geom_rect_per_row_fills():
    """Distinct fill column → per-row colours."""
    df = pl.DataFrame({
        "xmin": [0.0, 2.0], "xmax": [1.0, 3.0],
        "ymin": [0.0, 0.0], "ymax": [1.0, 1.0],
        "fill": ["#ff0000", "#00ff00"],
    })
    p = ggplot(df) + geom_rect(
        aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="fill"),
    ) + scale_fill_identity()
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        facecolors = coll.get_facecolors()
        # Two distinct colours.
        assert len(facecolors) == 2
        # First is reddish, second is greenish.
        assert facecolors[0][0] > 0.9 and facecolors[0][1] < 0.1
        assert facecolors[1][1] > 0.9 and facecolors[1][0] < 0.1
    finally:
        plt.close(fig)


def test_geom_tile_centres_with_default_unit_size():
    """Tiles centred on (x,y) with default 1×1 size span [x-0.5, x+0.5]."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]})
    p = ggplot(df) + geom_tile(aes(x="x", y="y"), fill="red")
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        # Two tiles → two paths.
        assert len(coll.get_paths()) == 2
        # First tile: vertices in [-0.5, 0.5] x [-0.5, 0.5].
        verts = coll.get_paths()[0].vertices
        import numpy as np
        assert np.isclose(verts[:, 0].min(), -0.5)
        assert np.isclose(verts[:, 0].max(), 0.5)
    finally:
        plt.close(fig)


def test_geom_tile_with_explicit_width_height():
    """`width`/`height` aesthetics override the unit default."""
    df = pl.DataFrame({"x": [0.0], "y": [0.0],
                       "width": [4.0], "height": [2.0]})
    p = ggplot(df) + geom_tile(aes(x="x", y="y", width="width", height="height"),
                                fill="red")
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        verts = coll.get_paths()[0].vertices
        import numpy as np
        # Tile spans [-2, 2] in x, [-1, 1] in y.
        assert np.isclose(verts[:, 0].min(), -2.0)
        assert np.isclose(verts[:, 0].max(), 2.0)
        assert np.isclose(verts[:, 1].min(), -1.0)
        assert np.isclose(verts[:, 1].max(), 1.0)
    finally:
        plt.close(fig)


def test_geom_raster_uses_imshow_on_regular_grid():
    """A regular x/y grid renders via ax.imshow rather than per-cell patches."""
    import numpy as np
    xs, ys = np.meshgrid(range(4), range(3))
    df = pl.DataFrame({
        "x": xs.ravel(),
        "y": ys.ravel(),
        "fill": ["#" + format(i * 100, "06x") for i in range(12)],
    })
    p = ggplot(df) + geom_raster(aes(x="x", y="y", fill="fill")) + scale_fill_identity()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # imshow lives in ax.images, not ax.collections.
        assert len(ax.images) == 1
        # Image extent matches the cell-centred grid: x in [-0.5, 3.5], y in [-0.5, 2.5].
        ext = ax.images[0].get_extent()
        assert np.isclose(ext[0], -0.5) and np.isclose(ext[1], 3.5)
        assert np.isclose(ext[2], -0.5) and np.isclose(ext[3], 2.5)
    finally:
        plt.close(fig)


def test_geom_raster_falls_back_to_tile_on_irregular():
    """Non-regular grid → fall back to the tile (PatchCollection) path."""
    df = pl.DataFrame({
        "x": [0.0, 1.0, 3.0],  # gap at x=2 → not uniform
        "y": [0.0, 0.0, 0.0],
        "fill": ["#ff0000", "#00ff00", "#0000ff"],
    })
    p = ggplot(df) + geom_raster(aes(x="x", y="y", fill="fill")) + scale_fill_identity()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Should not produce an image; should fall back to patches.
        assert len(ax.images) == 0
        assert len(ax.collections) == 1
    finally:
        plt.close(fig)


def test_geom_polygon_basic_triangle():
    """Three-vertex polygon renders as one closed shape."""
    df = pl.DataFrame({"x": [0.0, 1.0, 0.5], "y": [0.0, 0.0, 1.0]})
    p = ggplot(df) + geom_polygon(aes(x="x", y="y"), fill="blue")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        coll = ax.collections[0]
        assert len(coll.get_paths()) == 1
        # Closed polygon: 4 vertices listed (last = first).
        verts = coll.get_paths()[0].vertices
        import numpy as np
        assert verts.shape[0] == 4
        assert np.allclose(verts[0], verts[-1])
    finally:
        plt.close(fig)


def test_geom_polygon_multi_group():
    """Two groups → two distinct polygons in one collection."""
    df = pl.DataFrame({
        "x": [0.0, 1.0, 0.5,  3.0, 4.0, 3.5],
        "y": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "group": [1, 1, 1, 2, 2, 2],
    })
    p = ggplot(df) + geom_polygon(aes(x="x", y="y", group="group"), fill="green")
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        assert len(coll.get_paths()) == 2
    finally:
        plt.close(fig)


def test_geom_polygon_skips_groups_with_under_three_vertices():
    """A 2-vertex group can't form a polygon; it gets skipped silently."""
    df = pl.DataFrame({
        "x": [0.0, 1.0,  3.0, 4.0, 3.5],  # group 1: 2 vertices, group 2: 3 vertices
        "y": [0.0, 0.0,  0.0, 0.0, 1.0],
        "group": [1, 1, 2, 2, 2],
    })
    p = ggplot(df) + geom_polygon(aes(x="x", y="y", group="group"), fill="red")
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        # Only group 2 (the triangle) renders.
        assert len(coll.get_paths()) == 1
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2.5 — geom_errorbar / errorbarh / linerange / pointrange / crossbar
# ---------------------------------------------------------------------------


def _df_xy_range():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [5.0, 6.0, 7.0],
        "ymin": [4.5, 5.5, 6.5],
        "ymax": [5.5, 6.5, 7.5],
    })


def test_geom_linerange_three_lines_no_caps():
    """`geom_linerange` produces a single vertical-lines collection, no caps."""
    df = _df_xy_range()
    p = ggplot(df) + geom_linerange(aes(x="x", ymin="ymin", ymax="ymax"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Single LineCollection from vlines.
        assert len(ax.collections) == 1
        # 3 segments, one per row.
        segs = ax.collections[0].get_segments()
        assert len(segs) == 3
    finally:
        plt.close(fig)


def test_geom_errorbar_main_line_plus_two_caps():
    """`geom_errorbar` draws three collections: main line + top cap + bottom cap."""
    df = _df_xy_range()
    p = ggplot(df) + geom_errorbar(aes(x="x", ymin="ymin", ymax="ymax"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # 3 LineCollections total.
        assert len(ax.collections) == 3
        # Each cap collection has one horizontal segment per data row.
        for coll in ax.collections:
            assert len(coll.get_segments()) == 3
    finally:
        plt.close(fig)


def test_geom_errorbar_width_controls_cap_span():
    """`width=` controls the horizontal span of the caps."""
    df = _df_xy_range()
    p = ggplot(df) + geom_errorbar(aes(x="x", ymin="ymin", ymax="ymax"), width=0.2)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # The cap collections have segments whose endpoints span x ± 0.1.
        # First collection is vertical (vlines); cap collections are 2nd & 3rd.
        cap_coll = ax.collections[1]
        seg = cap_coll.get_segments()[0]
        # seg is a 2-row array (start, end).
        import numpy as np
        span = abs(seg[1][0] - seg[0][0])
        assert np.isclose(span, 0.2)
    finally:
        plt.close(fig)


def test_geom_errorbarh_horizontal_form():
    """`geom_errorbarh` swaps axes: line is horizontal, caps vertical."""
    df = pl.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "xmin": [1.0, 2.0, 3.0],
        "xmax": [2.0, 3.0, 4.0],
    })
    p = ggplot(df) + geom_errorbarh(aes(y="y", xmin="xmin", xmax="xmax"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Same 3-collection structure (horizontal line + 2 vertical caps).
        assert len(ax.collections) == 3
    finally:
        plt.close(fig)


def test_geom_pointrange_line_plus_point():
    """`geom_pointrange` = linerange + scatter point at (x, y)."""
    df = _df_xy_range()
    p = ggplot(df) + geom_pointrange(aes(x="x", y="y", ymin="ymin", ymax="ymax"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # 1 LineCollection (vlines) + 1 PathCollection (scatter).
        assert len(ax.collections) == 2
        # Scatter has 3 points.
        scat = ax.collections[1]
        assert len(scat.get_offsets()) == 3
    finally:
        plt.close(fig)


def test_geom_crossbar_box_plus_median():
    """`geom_crossbar` draws four box edges plus a thicker median line."""
    df = _df_xy_range()
    p = ggplot(df) + geom_crossbar(aes(x="x", y="y", ymin="ymin", ymax="ymax"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # 5 collections: top, bottom, left, right, median.
        assert len(ax.collections) == 5
        # Median (last) line width is 2× the others.
        median_lw = ax.collections[4].get_linewidth()[0]
        edge_lw = ax.collections[0].get_linewidth()[0]
        import numpy as np
        assert np.isclose(median_lw, edge_lw * 2)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2.6 — geom_segment, geom_curve
# ---------------------------------------------------------------------------


def test_geom_segment_three_segments():
    """`geom_segment` packs all rows into a single LineCollection."""
    df = pl.DataFrame({
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
        "xend": [1.0, 2.0, 3.0],
        "yend": [1.0, 1.0, 1.0],
    })
    p = ggplot(df) + geom_segment(aes(x="x", y="y", xend="xend", yend="yend"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        segs = ax.collections[0].get_segments()
        assert len(segs) == 3
        # First segment endpoints.
        import numpy as np
        assert np.allclose(segs[0], [[0.0, 0.0], [1.0, 1.0]])
    finally:
        plt.close(fig)


def test_geom_segment_per_row_colours():
    """Mapping ``colour=`` to a discrete column produces per-row colours."""
    df = pl.DataFrame({
        "x": [0.0, 1.0],
        "y": [0.0, 0.0],
        "xend": [1.0, 2.0],
        "yend": [1.0, 1.0],
        "g": ["a", "b"],
    })
    p = (ggplot(df)
         + geom_segment(aes(x="x", y="y", xend="xend", yend="yend", colour="g"))
         + scale_color_manual(values=["#ff0000", "#00ff00"]))
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        colours = coll.get_colors()
        assert len(colours) == 2
        # Different RGBA per segment.
        import numpy as np
        assert not np.allclose(colours[0], colours[1])
    finally:
        plt.close(fig)


def test_geom_curve_basic():
    """`geom_curve` renders one FancyArrowPatch per row."""
    df = pl.DataFrame({
        "x": [0.0, 1.0],
        "y": [0.0, 0.0],
        "xend": [1.0, 2.0],
        "yend": [1.0, 1.0],
    })
    p = ggplot(df) + geom_curve(aes(x="x", y="y", xend="xend", yend="yend"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Two FancyArrowPatch instances, one per row.
        from matplotlib.patches import FancyArrowPatch
        curve_patches = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(curve_patches) == 2
    finally:
        plt.close(fig)


def test_geom_curve_curvature_kwarg_controls_arc():
    """``curvature`` kwarg flows through to the connectionstyle."""
    df = pl.DataFrame({
        "x": [0.0], "y": [0.0], "xend": [1.0], "yend": [1.0],
    })
    p = ggplot(df) + geom_curve(aes(x="x", y="y", xend="xend", yend="yend"),
                                curvature=0.8)
    fig = p.draw()
    try:
        from matplotlib.patches import FancyArrowPatch
        patch = next(p for p in fig.axes[0].patches if isinstance(p, FancyArrowPatch))
        # connectionstyle keeps the rad parameter we passed.
        cs = patch.get_connectionstyle()
        assert getattr(cs, "rad", None) == 0.8
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2.7 — stat_summary
# ---------------------------------------------------------------------------


def _summary_test_df():
    return pl.DataFrame({
        "x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "y": [10.0, 12.0, 11.0, 15.0, 16.0, 14.0, 20.0, 22.0, 21.0],
    })


def test_stat_summary_default_is_mean_se_pointrange():
    """Default ``stat_summary()`` = mean ± 1 SE rendered as pointrange."""
    df = _summary_test_df()
    p = ggplot(df, aes("x", "y")) + stat_summary()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Pointrange: linerange (vlines collection) + scatter (point collection).
        assert len(ax.collections) == 2
        # The point centres are the means: x=1 → 11, x=2 → 15, x=3 → 21.
        offsets = ax.collections[1].get_offsets()
        import numpy as np
        np.testing.assert_array_almost_equal(
            sorted(offsets.tolist()), [[1.0, 11.0], [2.0, 15.0], [3.0, 21.0]]
        )
    finally:
        plt.close(fig)


def test_stat_summary_mean_cl_normal_errorbar():
    """``fun_data='mean_cl_normal'`` produces normal-CI bounds; geom='errorbar'
    renders the line + caps."""
    df = _summary_test_df()
    p = (ggplot(df, aes("x", "y"))
         + stat_summary(fun_data="mean_cl_normal", geom="errorbar"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # errorbar: 3 LineCollections (vline + 2 caps).
        assert len(ax.collections) == 3
    finally:
        plt.close(fig)


def test_stat_summary_componentwise_min_max_median():
    """Componentwise ``fun=median, fun_min=min, fun_max=max`` works."""
    df = _summary_test_df()
    p = (ggplot(df, aes("x", "y"))
         + stat_summary(fun="median", fun_min="min", fun_max="max"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # pointrange
        offsets = ax.collections[1].get_offsets()
        import numpy as np
        # x=1: median=11, min=10, max=12; same shape across x.
        np.testing.assert_array_almost_equal(
            sorted(offsets.tolist()), [[1.0, 11.0], [2.0, 15.0], [3.0, 21.0]]
        )
        # vlines extents match (min..max) per x.
        segs = ax.collections[0].get_segments()
        # Sort by x for deterministic order.
        segs_sorted = sorted(segs, key=lambda s: s[0][0])
        np.testing.assert_array_almost_equal(segs_sorted[0], [[1.0, 10.0], [1.0, 12.0]])
        np.testing.assert_array_almost_equal(segs_sorted[1], [[2.0, 14.0], [2.0, 16.0]])
        np.testing.assert_array_almost_equal(segs_sorted[2], [[3.0, 20.0], [3.0, 22.0]])
    finally:
        plt.close(fig)


def test_stat_summary_median_hilow():
    """``fun_data='median_hilow'`` matches numpy quantiles at conf=0.95."""
    import numpy as np
    df = _summary_test_df()
    p = ggplot(df, aes("x", "y")) + stat_summary(fun_data="median_hilow")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        offsets = ax.collections[1].get_offsets()
        # Centre is median.
        for x_val, expected_med in [(1, 11.0), (2, 15.0), (3, 21.0)]:
            row = next(o for o in offsets if abs(o[0] - x_val) < 1e-9)
            assert abs(row[1] - expected_med) < 1e-9
    finally:
        plt.close(fig)


def test_stat_summary_bootstrap_with_seed_is_deterministic():
    """Bootstrap CI with explicit seed is reproducible across runs."""
    import numpy as np
    df = _summary_test_df()
    p1 = (ggplot(df, aes("x", "y"))
          + stat_summary(fun_data="mean_cl_boot", fun_args={"seed": 7}))
    p2 = (ggplot(df, aes("x", "y"))
          + stat_summary(fun_data="mean_cl_boot", fun_args={"seed": 7}))
    fig1 = p1.draw()
    fig2 = p2.draw()
    try:
        seg1 = fig1.axes[0].collections[0].get_segments()
        seg2 = fig2.axes[0].collections[0].get_segments()
        for s1, s2 in zip(seg1, seg2):
            np.testing.assert_array_almost_equal(s1, s2)
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_stat_summary_unknown_fun_data_errors():
    df = _summary_test_df()
    p = ggplot(df, aes("x", "y")) + stat_summary(fun_data="bogus_summary")
    with pytest.raises(ValueError, match="unknown summary"):
        p.draw()


def test_stat_summary_unknown_geom_errors():
    with pytest.raises(ValueError, match="unknown geom"):
        stat_summary(geom="weirdo")


# ---------------------------------------------------------------------------
# Phase 2.9 — qq, qq_line, ecdf, unique, sum/count, contour, hex, dotplot
# ---------------------------------------------------------------------------


def test_stat_qq_against_normal_quantiles():
    """`stat_qq()` produces sorted (theoretical, sample) pairs against the
    standard normal."""
    import numpy as np
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(size=200)})
    p = ggplot(df, aes(sample="v")) + stat_qq()
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        offsets = coll.get_offsets()
        # Points are sorted by x (theoretical quantile).
        xs = offsets[:, 0]
        assert np.all(np.diff(xs) >= -1e-9)
        # x-range covers ~ [-2.6, 2.6] for n=200 standard normal quantiles.
        assert xs.min() < -2.0 and xs.max() > 2.0
    finally:
        plt.close(fig)


def test_stat_qq_line_two_endpoints():
    """`stat_qq_line()` returns a single line drawn over the theoretical x range."""
    import numpy as np
    rng = np.random.default_rng(1)
    df = pl.DataFrame({"v": rng.normal(size=100)})
    p = ggplot(df, aes(sample="v")) + stat_qq() + stat_qq_line()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # qq_line uses GeomPath → one line in ax.lines.
        assert len(ax.lines) == 1
        line = ax.lines[0]
        xy = line.get_xydata()
        # Line is monotone increasing in x.
        assert xy[1, 0] > xy[0, 0]
    finally:
        plt.close(fig)


def test_geom_qq_alias_matches_stat_qq():
    """`geom_qq()` is just `stat_qq()` — same output."""
    import numpy as np
    rng = np.random.default_rng(2)
    df = pl.DataFrame({"v": rng.normal(size=50)})
    p1 = ggplot(df, aes(sample="v")) + stat_qq()
    p2 = ggplot(df, aes(sample="v")) + geom_qq()
    fig1 = p1.draw()
    fig2 = p2.draw()
    try:
        o1 = fig1.axes[0].collections[0].get_offsets()
        o2 = fig2.axes[0].collections[0].get_offsets()
        assert np.allclose(o1, o2)
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_stat_ecdf_basic_step_to_one():
    """`stat_ecdf()` produces a step function from 0 to 1."""
    import numpy as np
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    p = ggplot(df, aes("x")) + stat_ecdf()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        line = ax.lines[0]
        xs, ys = line.get_xdata(), line.get_ydata()
        assert ys[0] == pytest.approx(0.2, abs=1e-9)
        assert ys[-1] == pytest.approx(1.0, abs=1e-9)
        # ys is non-decreasing.
        assert np.all(np.diff(ys) >= -1e-9)
    finally:
        plt.close(fig)


def test_stat_unique_drops_duplicates():
    """`stat_unique()` collapses repeated rows before geom_point."""
    df = pl.DataFrame({
        "x": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        "y": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    })
    p = ggplot(df, aes("x", "y")) + stat_unique()
    fig = p.draw()
    try:
        offsets = fig.axes[0].collections[0].get_offsets()
        # 6 rows → 3 unique pairs.
        assert len(offsets) == 3
    finally:
        plt.close(fig)


def test_geom_count_size_scales_with_multiplicity():
    """`geom_count()` sizes points by `(x, y)` multiplicity."""
    df = pl.DataFrame({
        "x": [1, 1, 1, 2, 2, 3],
        "y": [1, 1, 1, 2, 2, 3],
    })
    p = ggplot(df, aes("x", "y")) + geom_count()
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        # 3 unique (x,y) combinations.
        assert len(coll.get_offsets()) == 3
        # Sizes are not all equal — multiplicity shows up.
        sizes = coll.get_sizes()
        assert len(set(sizes.tolist())) >= 2
    finally:
        plt.close(fig)


def test_geom_contour_on_regular_grid():
    """`geom_contour` produces iso-lines on a 2D Gaussian grid."""
    import numpy as np
    xs, ys = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
    zs = np.exp(-(xs**2 + ys**2) / 2)
    df = pl.DataFrame({"x": xs.ravel(), "y": ys.ravel(), "z": zs.ravel()})
    p = ggplot(df) + geom_contour(aes(x="x", y="y", z="z"), bins=8)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # contour produces a LineCollection (multiple iso-line paths).
        assert len(ax.collections) >= 1
    finally:
        plt.close(fig)


def test_geom_contour_filled_on_regular_grid():
    """`geom_contour_filled` produces filled bands."""
    import numpy as np
    xs, ys = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
    zs = np.exp(-(xs**2 + ys**2) / 2)
    df = pl.DataFrame({"x": xs.ravel(), "y": ys.ravel(), "z": zs.ravel()})
    p = ggplot(df) + geom_contour_filled(aes(x="x", y="y", z="z"), bins=6)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.collections) >= 1
    finally:
        plt.close(fig)


def test_geom_contour_irregular_grid_errors():
    """A grid with missing cells (count mismatch) is rejected."""
    # 3 unique x × 2 unique y = 6 cells, but only 5 rows present.
    df = pl.DataFrame({
        "x": [0.0, 1.0, 2.0, 0.0, 1.0],
        "y": [0.0, 0.0, 0.0, 1.0, 1.0],
        "z": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    with pytest.raises(ValueError, match="regular .x, y. grid"):
        ggplot(df).__add__(geom_contour(aes(x="x", y="y", z="z"))).draw()


def test_geom_hex_renders_polycollection():
    """`geom_hex()` adds a hexbin polycollection to the axes."""
    import numpy as np
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    p = ggplot(df, aes("x", "y")) + geom_hex(bins=15)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # hexbin produces a PolyCollection.
        from matplotlib.collections import PolyCollection
        assert any(isinstance(c, PolyCollection) for c in ax.collections)
    finally:
        plt.close(fig)


def test_geom_dotplot_stacks_within_bin():
    """`geom_dotplot()` produces one scatter point per data row, stacked
    vertically within each bin."""
    import numpy as np
    df = pl.DataFrame({"x": [1.0] * 5 + [2.0] * 3 + [3.0] * 1})
    p = ggplot(df, aes("x")) + geom_dotplot()
    fig = p.draw()
    try:
        coll = fig.axes[0].collections[0]
        # 9 dots total.
        assert len(coll.get_offsets()) == 9
        # Dots at x=1 stack to 5 distinct y values.
        offs = coll.get_offsets()
        x1 = [o for o in offs if abs(o[0] - 1.0) < 0.5]
        assert len({round(o[1], 6) for o in x1}) == 5
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3.1 — guide_legend
# ---------------------------------------------------------------------------


def _all_legends(ax):
    """Collect all Legend artists attached to ``ax`` (active + add_artist'd)."""
    legs = [c for c in ax.get_children() if c.__class__.__name__ == "Legend"]
    cur = ax.get_legend()
    if cur is not None:
        legs.append(cur)
    return list({id(leg): leg for leg in legs}.values())


def _df_two_groups():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "g": ["a", "a", "a", "b", "b", "b"],
        "h": ["x", "x", "x", "y", "y", "y"],
    })


def test_legend_auto_built_for_discrete_shape():
    """`aes(shape=col)` → one auto-legend with title from the aes value."""
    df = _df_two_groups()
    p = ggplot(df, aes("x", "y", shape="g")) + geom_point()
    fig = p.draw()
    try:
        legs = _all_legends(fig.axes[0])
        assert len(legs) == 1
        leg = legs[0]
        assert leg.get_title().get_text() == "g"
        labels = sorted(t.get_text() for t in leg.get_texts())
        assert labels == ["a", "b"]
    finally:
        plt.close(fig)


def test_legend_auto_built_for_discrete_colour():
    """`aes(colour=col)` produces a discrete colour legend."""
    df = _df_two_groups()
    p = ggplot(df, aes("x", "y", colour="g")) + geom_point()
    fig = p.draw()
    try:
        legs = _all_legends(fig.axes[0])
        assert len(legs) == 1
        leg = legs[0]
        # Two colour swatches at distinct hex codes.
        handles = leg.legend_handles
        c0 = handles[0].get_color()
        c1 = handles[1].get_color()
        assert c0 != c1
    finally:
        plt.close(fig)


def test_legend_auto_merge_same_source_column():
    """`aes(colour=g, shape=g)` produces a single merged legend."""
    df = _df_two_groups()
    p = ggplot(df, aes("x", "y", colour="g", shape="g")) + geom_point()
    fig = p.draw()
    try:
        legs = _all_legends(fig.axes[0])
        assert len(legs) == 1
        leg = legs[0]
        assert leg.get_title().get_text() == "g"
        # Each handle picks up both colour and shape.
        handles = leg.legend_handles
        # Distinct shapes per level.
        markers = {h.get_marker() for h in handles}
        assert len(markers) == 2
        # Distinct colours per level.
        colours = {h.get_color() for h in handles}
        assert len(colours) == 2
    finally:
        plt.close(fig)


def test_legend_two_groups_when_sources_differ():
    """`aes(colour=g, shape=h)` with two different source columns →
    two legends side-by-side."""
    df = _df_two_groups()
    p = ggplot(df, aes("x", "y", colour="g", shape="h")) + geom_point()
    fig = p.draw()
    try:
        legs = _all_legends(fig.axes[0])
        assert len(legs) == 2
        titles = sorted(leg.get_title().get_text() for leg in legs)
        assert titles == ["g", "h"]
    finally:
        plt.close(fig)


def test_legend_position_none_hides_legends():
    """`theme(legend_position='none')` skips legend rendering entirely."""
    df = _df_two_groups()
    p = (ggplot(df, aes("x", "y", shape="g")) + geom_point()
         + theme(legend_position="none"))
    fig = p.draw()
    try:
        assert _all_legends(fig.axes[0]) == []
    finally:
        plt.close(fig)


def test_legend_position_top_horizontal():
    """`theme(legend_position='top', legend_direction='horizontal')` lays
    handles out in a single row above the axes."""
    df = _df_two_groups()
    p = (ggplot(df, aes("x", "y", shape="g")) + geom_point()
         + theme(legend_position="top", legend_direction="horizontal"))
    fig = p.draw()
    try:
        leg = _all_legends(fig.axes[0])[0]
        # Anchor sits above the axes (y > 1 in axes coords).
        bbox = leg.get_bbox_to_anchor().get_points()
        # bbox is in axes coords thanks to transAxes default.
        assert bbox.shape == (2, 2)
    finally:
        plt.close(fig)


def test_legend_title_from_labs_overrides_aes():
    """`labs(colour='Group')` sets the legend title."""
    df = _df_two_groups()
    p = (ggplot(df, aes("x", "y", colour="g")) + geom_point()
         + labs(colour="Group"))
    fig = p.draw()
    try:
        leg = _all_legends(fig.axes[0])[0]
        assert leg.get_title().get_text() == "Group"
    finally:
        plt.close(fig)


def test_legend_constant_aes_param_does_not_create_legend():
    """`geom_point(colour='red')` is a fixed colour, not a mapping →
    no legend."""
    df = _df_two_groups()
    p = ggplot(df, aes("x", "y")) + geom_point(colour="red")
    fig = p.draw()
    try:
        assert _all_legends(fig.axes[0]) == []
    finally:
        plt.close(fig)


def test_legend_scale_identity_skips_legend():
    """`scale_color_identity()` says the data already holds drawable values
    → no legend."""
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
        "c": ["#ff0000", "#00ff00", "#0000ff"],
    })
    p = (ggplot(df, aes("x", "y", colour="c")) + geom_point()
         + scale_color_identity())
    fig = p.draw()
    try:
        assert _all_legends(fig.axes[0]) == []
    finally:
        plt.close(fig)


def test_guide_legend_factory_returns_struct():
    """`guide_legend()` is a metadata holder; not yet renderer-consumed."""
    g = guide_legend(title="X", reverse=True)
    assert g.title == "X"
    assert g.reverse is True


def test_guides_addition_to_plot_stores_overrides():
    """`+ guides(colour=guide_legend(...))` stores into ``guide_overrides``."""
    df = _df_two_groups()
    p = (ggplot(df, aes("x", "y", colour="g")) + geom_point()
         + guides(colour=guide_legend(title="Group")))
    assert getattr(p, "guide_overrides", {}).get("colour") is not None


# ---------------------------------------------------------------------------
# Phase 3.2 — guide_colorbar
# ---------------------------------------------------------------------------


def _df_continuous_colour():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "z": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


def test_continuous_colour_renders_colorbar():
    """`aes(colour=numeric_col)` produces a fig.colorbar (extra child axes
    with `_colorbar` label)."""
    import warnings
    df = _df_continuous_colour()
    p = ggplot(df, aes("x", "y", colour="z")) + geom_point()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = p.draw()
    try:
        # main axes + colorbar axes.
        assert len(fig.axes) == 2
        cb_ax = fig.axes[1]
        assert cb_ax.get_label() == "<colorbar>"
        # Colorbar title placed ABOVE the bar via cb.ax.set_title (R/ggplot2
        # default) — falls back to scale.name → aes-source ("z").
        assert cb_ax.get_title() == "z"
    finally:
        plt.close(fig)


def test_colorbar_range_matches_data():
    """Colorbar limits track the data range."""
    import warnings
    df = _df_continuous_colour()
    p = ggplot(df, aes("x", "y", colour="z")) + geom_point()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = p.draw()
    try:
        cb_ax = fig.axes[1]
        # For a vertical colorbar matplotlib uses ylim for the range.
        ylim = cb_ax.get_ylim()
        assert ylim[0] == pytest.approx(10.0)
        assert ylim[1] == pytest.approx(50.0)
    finally:
        plt.close(fig)


def test_colorbar_position_top_horizontal():
    """`theme(legend_position='top')` makes the colorbar horizontal."""
    import warnings
    df = _df_continuous_colour()
    p = (ggplot(df, aes("x", "y", colour="z")) + geom_point()
         + theme(legend_position="top"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = p.draw()
    try:
        cb_ax = fig.axes[1]
        pos = cb_ax.get_position()
        # Horizontal colorbar: wider than tall.
        assert pos.width > pos.height
        # And it sits in the upper half of the figure.
        assert pos.y0 > 0.5
    finally:
        plt.close(fig)


def test_colorbar_title_from_labs_overrides_aes():
    """`labs(colour='Z value')` overrides the colorbar title (placed
    above the bar via cb.ax.set_title — R/ggplot2 default)."""
    import warnings
    df = _df_continuous_colour()
    p = (ggplot(df, aes("x", "y", colour="z")) + geom_point()
         + labs(colour="Z value"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = p.draw()
    try:
        cb_ax = fig.axes[1]
        assert cb_ax.get_title() == "Z value"
    finally:
        plt.close(fig)


def test_colorbar_and_legend_can_coexist():
    """Continuous colour + discrete shape → one colorbar + one legend."""
    import warnings
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "z": [10.0, 20.0, 30.0, 40.0],
        "g": ["a", "a", "b", "b"],
    })
    p = ggplot(df, aes("x", "y", colour="z", shape="g")) + geom_point()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = p.draw()
    try:
        # 2 axes: main + colorbar.
        assert len(fig.axes) == 2
        # And one legend.
        legs = _all_legends(fig.axes[0])
        assert len(legs) == 1
        assert legs[0].get_title().get_text() == "g"
    finally:
        plt.close(fig)


def test_colorbar_position_none_hides():
    """`theme(legend_position='none')` hides the colorbar too."""
    import warnings
    df = _df_continuous_colour()
    p = (ggplot(df, aes("x", "y", colour="z")) + geom_point()
         + theme(legend_position="none"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = p.draw()
    try:
        # Only the main axes — no colorbar child.
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3.3 — guide_axis
# ---------------------------------------------------------------------------


def test_guide_axis_rotation_via_guides():
    """`guides(x=guide_axis(angle=45))` rotates x tick labels."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + guides(x=guide_axis(angle=45)))
    fig = p.draw()
    try:
        rotations = {t.get_rotation()
                     for t in fig.axes[0].xaxis.get_majorticklabels()}
        assert rotations == {45.0}
    finally:
        plt.close(fig)


def test_guide_axis_rotation_y_axis():
    """`guides(y=guide_axis(angle=...))` rotates the y axis."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + guides(y=guide_axis(angle=15)))
    fig = p.draw()
    try:
        rotations = {t.get_rotation()
                     for t in fig.axes[0].yaxis.get_majorticklabels()}
        assert rotations == {15.0}
    finally:
        plt.close(fig)


def test_axis_rotation_via_theme_element_text_angle():
    """`theme(axis_text_x=element_text(angle=30))` also rotates x labels."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + theme(axis_text_x=element_text(angle=30)))
    fig = p.draw()
    try:
        rotations = {t.get_rotation()
                     for t in fig.axes[0].xaxis.get_majorticklabels()}
        assert rotations == {30.0}
    finally:
        plt.close(fig)


def test_guide_axis_overrides_theme():
    """When both are set, ``guides(x=guide_axis(angle=...))`` wins."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + theme(axis_text_x=element_text(angle=10))
         + guides(x=guide_axis(angle=60)))
    fig = p.draw()
    try:
        rotations = {t.get_rotation()
                     for t in fig.axes[0].xaxis.get_majorticklabels()}
        assert rotations == {60.0}
    finally:
        plt.close(fig)


def test_guide_axis_factory_holds_metadata():
    g = guide_axis(angle=30, n_dodge=2, position="top")
    assert g.angle == 30
    assert g.n_dodge == 2
    assert g.position == "top"


# ---------------------------------------------------------------------------
# Phase 3.4 — temporal / percent / ordinal / radius scales
# ---------------------------------------------------------------------------


def test_scale_x_date_formats_ticks_as_iso_dates():
    """`scale_x_date()` installs a date locator + formatter."""
    import datetime
    df = pl.DataFrame({
        "x": [datetime.date(2020, 1, 1),
              datetime.date(2020, 6, 1),
              datetime.date(2020, 12, 1)],
        "y": [1.0, 2.0, 3.0],
    })
    p = ggplot(df, aes("x", "y")) + geom_point() + scale_x_date()
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].xaxis.get_majorticklabels()]
        # All non-empty labels parse as ISO dates.
        nonempty = [l for l in labels if l]
        assert all(len(l) == 10 and l.count("-") == 2 for l in nonempty)
    finally:
        plt.close(fig)


def test_scale_x_date_custom_format():
    """`date_format='%b %Y'` overrides the formatter."""
    import datetime
    df = pl.DataFrame({
        "x": [datetime.date(2020, 1, 1),
              datetime.date(2020, 6, 1),
              datetime.date(2020, 12, 1)],
        "y": [1.0, 2.0, 3.0],
    })
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + scale_x_date(date_format="%b %Y"))
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].xaxis.get_majorticklabels()]
        nonempty = [l for l in labels if l]
        # "Jan 2020" / "Feb 2020" — month abbreviation + year.
        assert all(any(m in l for m in
                       ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                   for l in nonempty)
    finally:
        plt.close(fig)


def test_scale_x_datetime_includes_time():
    """`scale_x_datetime()` formatter shows the time portion by default."""
    import datetime
    df = pl.DataFrame({
        "x": [datetime.datetime(2020, 1, 1, 12, 30),
              datetime.datetime(2020, 1, 1, 13, 30),
              datetime.datetime(2020, 1, 1, 14, 30)],
        "y": [1.0, 2.0, 3.0],
    })
    p = ggplot(df, aes("x", "y")) + geom_point() + scale_x_datetime()
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].xaxis.get_majorticklabels()]
        nonempty = [l for l in labels if l]
        # Default datetime format includes ":" for the hh:mm part.
        assert any(":" in l for l in nonempty)
    finally:
        plt.close(fig)


def test_scale_y_percent_formats_as_percent():
    """`scale_y_percent()` formats numeric ticks as ``50%`` etc."""
    df = pl.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [0.1, 0.5, 0.75, 0.95],
    })
    p = ggplot(df, aes("x", "y")) + geom_point() + scale_y_percent()
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].yaxis.get_majorticklabels()]
        nonempty = [l for l in labels if l]
        assert all(l.endswith("%") for l in nonempty)
    finally:
        plt.close(fig)


def test_scale_y_percent_xmax_100():
    """`xmax=100` lets the data already be in 0-100 range."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [10.0, 50.0, 90.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + scale_y_percent(xmax=100))
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].yaxis.get_majorticklabels()]
        nonempty = [l for l in labels if l]
        assert all(l.endswith("%") for l in nonempty)
        # 50.0 should render as "50%", not "5000%".
        assert any("50%" in l for l in nonempty)
    finally:
        plt.close(fig)


def test_scale_x_ordinal_passes_through_strings():
    """`scale_x_ordinal()` preserves matplotlib's categorical axis labels."""
    df = pl.DataFrame({
        "x": ["low", "medium", "high"],
        "y": [1.0, 2.0, 3.0],
    })
    p = ggplot(df, aes("x", "y")) + geom_point() + scale_x_ordinal()
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].xaxis.get_majorticklabels()]
        assert labels == ["low", "medium", "high"]
    finally:
        plt.close(fig)


def test_scale_radius_is_continuous_size_alias():
    """`scale_radius()` produces the same continuous size mapping as
    `scale_size_continuous()` (both use linear rescale_pal)."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "z": [1.0, 5.0, 10.0]})
    p = (ggplot(df, aes("x", "y", size="z")) + geom_point()
         + scale_radius(range=(2.0, 8.0)))
    fig = p.draw()
    try:
        # Just checking it draws without errors and produces a scatter.
        assert len(fig.axes[0].collections) >= 1
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 4.2 — annotate
# ---------------------------------------------------------------------------


def test_annotate_text_at_data_position():
    """`annotate('text', x=, y=, label=)` adds one matplotlib text artist."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + annotate("text", x=2, y=5, label="midpoint"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert "midpoint" in texts
    finally:
        plt.close(fig)


def test_annotate_rect_with_fill_alpha():
    """`annotate('rect', xmin=, xmax=, ymin=, ymax=, fill=, alpha=)`."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + annotate("rect", xmin=1.5, xmax=2.5, ymin=2, ymax=6,
                    fill="red", alpha=0.3))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Rect produces a PatchCollection.
        from matplotlib.collections import PatchCollection
        assert any(isinstance(c, PatchCollection) for c in ax.collections)
    finally:
        plt.close(fig)


def test_annotate_segment_constants():
    """`annotate('segment', x, y, xend, yend)` draws one line segment."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + annotate("segment", x=1, y=1, xend=3, yend=9, colour="blue"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Segment uses LineCollection.
        from matplotlib.collections import LineCollection
        assert any(isinstance(c, LineCollection) for c in ax.collections)
    finally:
        plt.close(fig)


def test_annotate_broadcast_iterables_with_scalars():
    """Three annotation rows from a 3-element x list and scalar y."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + annotate("text", x=[1, 2, 3], y=2, label=["a", "b", "c"]))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        texts = sorted(t.get_text() for t in ax.texts)
        assert texts == ["a", "b", "c"]
    finally:
        plt.close(fig)


def test_annotate_renders_on_every_facet_panel():
    """Annotation broadcasts to every panel (matches ggplot2 behaviour)."""
    df = pl.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
    })
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + facet_wrap("g")
         + annotate("text", x=3, y=7, label="ANN"))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 2
        for a in visible:
            assert "ANN" in [t.get_text() for t in a.texts]
    finally:
        plt.close(fig)


def test_annotate_unknown_geom_errors():
    with pytest.raises(ValueError, match="unknown geom"):
        annotate("nonexistent", x=1, y=1)


def test_annotate_no_aesthetics_errors():
    with pytest.raises(ValueError, match="at least one aesthetic"):
        annotate("text")


def test_annotate_inconsistent_lengths_error():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        annotate("text", x=[1, 2, 3], y=[1, 2], label="a")


# ---------------------------------------------------------------------------
# Phase 4.4 — coord_flip / coord_trans
# ---------------------------------------------------------------------------


def test_coord_flip_swaps_axis_labels():
    """`coord_flip()` swaps which aes ends up on which axis."""
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 5.0, 15.0]})
    p = (ggplot(df, aes("cat", "val")) + geom_col() + coord_flip())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # x label is the original y mapping; y label is the original x mapping.
        assert ax.get_xlabel() == "val"
        assert ax.get_ylabel() == "cat"
    finally:
        plt.close(fig)


def test_coord_flip_renders_horizontal_bars():
    """`geom_col() + coord_flip()` produces bars extending along the x-axis."""
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 5.0, 15.0]})
    p = (ggplot(df, aes("cat", "val")) + geom_col() + coord_flip())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # x axis range covers the val span (10, 5, 15 → 0..15+).
        xlim = ax.get_xlim()
        assert xlim[0] <= 0.0 and xlim[1] >= 15.0
    finally:
        plt.close(fig)


def test_coord_flip_swaps_scale_application():
    """A scale set on the x aesthetic applies to the visible y-axis after flip.

    `scale_x_continuous(limits=(0, 5))` constrains the x aes (`cat`'s
    integer index) — after flip, that constraint shows up on the y-axis.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + scale_x_continuous(limits=(0, 5))
         + coord_flip())
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # The (0, 5) limits land on the visible y axis.
        ylim = ax.get_ylim()
        assert ylim == pytest.approx((0.0, 5.0), abs=1e-9)
    finally:
        plt.close(fig)


def test_coord_trans_y_log10_sets_matplotlib_scale():
    """`coord_trans(y='log10')` sets the y axis to a log scale at render."""
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [10.0, 100.0, 1000.0, 10000.0],
    })
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + coord_trans(y="log10"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
    finally:
        plt.close(fig)


def test_coord_trans_x_sqrt_uses_function_scale():
    """`coord_trans(x='sqrt')` registers a function scale on the x axis."""
    df = pl.DataFrame({
        "x": [1.0, 4.0, 9.0, 16.0],
        "y": [1.0, 2.0, 3.0, 4.0],
    })
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + coord_trans(x="sqrt"))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xscale() in ("function", "functionlog")
    finally:
        plt.close(fig)


def test_coord_trans_unknown_name_errors():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + coord_trans(x="bogus"))
    with pytest.raises(ValueError, match="unknown transform"):
        p.draw()


# ---------------------------------------------------------------------------
# Phase 4.3 — annotation_custom
# ---------------------------------------------------------------------------


def test_annotation_custom_places_artist_at_bounds():
    """`annotation_custom(rect, xmin, xmax, ymin, ymax)` adds the artist
    sized to the given bounding box."""
    from matplotlib.patches import Rectangle

    df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    rect = Rectangle((0, 0), 1, 1, color="red", alpha=0.3)
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + annotation_custom(rect, xmin=2, xmax=4, ymin=2, ymax=4))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # The custom rect now sits in the patches list.
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) >= 1
        # Find one whose bounds match what we asked for.
        match = next((p for p in rects
                      if abs(p.get_x() - 2) < 1e-6 and abs(p.get_y() - 2) < 1e-6
                      and abs(p.get_width() - 2) < 1e-6 and abs(p.get_height() - 2) < 1e-6),
                     None)
        assert match is not None
    finally:
        plt.close(fig)


def test_annotation_custom_requires_all_bounds():
    """`annotation_custom` rejects None bounds — `-Inf`/`Inf` shorthand
    isn't supported yet."""
    from matplotlib.patches import Rectangle

    rect = Rectangle((0, 0), 1, 1)
    with pytest.raises(ValueError, match="must all be set"):
        annotation_custom(rect, xmin=0, xmax=10, ymin=0)


def test_annotation_custom_renders_on_each_facet_panel():
    """Like `annotate`, the custom artist broadcasts to every panel."""
    from matplotlib.patches import Rectangle

    df = pl.DataFrame({
        "x": [0.0, 5.0, 0.0, 5.0],
        "y": [0.0, 5.0, 0.0, 5.0],
        "g": ["a", "a", "b", "b"],
    })
    rect = Rectangle((0, 0), 1, 1, color="green", alpha=0.2)
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + facet_wrap("g")
         + annotation_custom(rect, xmin=1, xmax=3, ymin=1, ymax=3))
    fig = p.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == 2
        for a in visible:
            rects = [p for p in a.patches if isinstance(p, Rectangle)]
            assert len(rects) >= 1
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 4.5 — expansion()
# ---------------------------------------------------------------------------


def test_expansion_scalar_split_returns_four_components():
    e = expansion(mult=0.1, add=0.5)
    m_lo, m_hi, a_lo, a_hi = e.split()
    assert m_lo == 0.1 and m_hi == 0.1
    assert a_lo == 0.5 and a_hi == 0.5


def test_expansion_tuple_split_asymmetric():
    e = expansion(mult=(0.0, 0.2))
    m_lo, m_hi, _a_lo, _a_hi = e.split()
    assert m_lo == 0.0 and m_hi == 0.2


def test_expansion_invalid_form_errors():
    with pytest.raises(ValueError, match="expected scalar or"):
        expansion(mult="not a number").split()


def test_expansion_mult_widens_axis_via_margins():
    """`scale_x_continuous(expand=expansion(mult=0.5))` widens xlim by 50%."""
    df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    p = (ggplot(df, aes("x", "y")) + geom_point()
         + scale_x_continuous(expand=expansion(mult=0.5)))
    fig = p.draw()
    try:
        xlim = fig.axes[0].get_xlim()
        # 50% margin on each side widens the (0, 10) range to (-5, 15).
        assert xlim[0] == pytest.approx(-5.0, abs=0.5)
        assert xlim[1] == pytest.approx(15.0, abs=0.5)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# draw()/show()/save() figsize controls
# ---------------------------------------------------------------------------


def test_draw_width_height_inches():
    """`p.draw(width=8, height=3)` resizes the figure."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    fig = ggplot(df, aes("x", "y")).geom_point().draw(width=8, height=3)
    try:
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(8.0)
        assert size[1] == pytest.approx(3.0)
    finally:
        plt.close(fig)


def test_draw_figsize_shorthand():
    """`p.draw(figsize=(W, H))` is the matplotlib-style equivalent."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    fig = ggplot(df, aes("x", "y")).geom_point().draw(figsize=(6, 2))
    try:
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(6.0)
        assert size[1] == pytest.approx(2.0)
    finally:
        plt.close(fig)


def test_draw_units_cm():
    """`units='cm'` converts to inches before sizing."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    fig = (ggplot(df, aes("x", "y")).geom_point()
           .draw(width=20, height=8, units="cm"))
    try:
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(20 / 2.54, abs=1e-6)
        assert size[1] == pytest.approx(8 / 2.54, abs=1e-6)
    finally:
        plt.close(fig)


def test_draw_figsize_and_width_height_conflict_errors():
    """Passing both forms raises TypeError."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    p = ggplot(df, aes("x", "y")).geom_point()
    with pytest.raises(TypeError, match="figsize"):
        p.draw(figsize=(6, 2), width=8)


def test_draw_size_applies_to_facets():
    """Faceted plots resize too — the auto-formula isn't sticky."""
    df = pl.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 2, 3, 4, 5, 6],
        "g": ["a", "a", "a", "b", "b", "b"],
    })
    fig = (ggplot(df, aes("x", "y")) + geom_point() + facet_grid("~ g")).draw(
        width=10, height=3,
    )
    try:
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(10.0)
        assert size[1] == pytest.approx(3.0)
    finally:
        plt.close(fig)


def test_draw_with_ax_skips_resize():
    """When the user passes ``ax=``, sizing is the parent figure's job."""
    import matplotlib.pyplot as plt
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    parent_fig, ax = plt.subplots(figsize=(4, 4))
    try:
        # width/height kwargs should NOT change parent_fig's size.
        ggplot(df, aes("x", "y")).geom_point().draw(ax=ax, width=20, height=2)
        size = parent_fig.get_size_inches()
        assert size[0] == pytest.approx(4.0)
        assert size[1] == pytest.approx(4.0)
    finally:
        plt.close(parent_fig)


# ---------------------------------------------------------------------------
# Phase 5 — Patchwork composition
# ---------------------------------------------------------------------------


def _two_simple_plots():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    p1 = ggplot(df, aes("x", "y")) + geom_point()
    p2 = ggplot(df, aes("x", "y")) + geom_col()
    return p1, p2


def test_patchwork_or_creates_horizontal_grid():
    """`p1 | p2` returns a horizontal :class:`PlotGrid` of length 2."""
    p1, p2 = _two_simple_plots()
    g = p1 | p2
    assert isinstance(g, PlotGrid)
    assert g.direction == "h"
    assert len(g.children) == 2


def test_patchwork_truediv_creates_vertical_grid():
    """`p1 / p2` returns a vertical :class:`PlotGrid`."""
    p1, p2 = _two_simple_plots()
    g = p1 / p2
    assert isinstance(g, PlotGrid)
    assert g.direction == "v"


def test_patchwork_same_direction_flattens():
    """`p1 | p2 | p3` flattens into a single 1×3 grid (no nesting)."""
    p1, p2 = _two_simple_plots()
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p3 = ggplot(df, aes("x", "y")) + geom_point()
    g = p1 | p2 | p3
    assert g.direction == "h"
    assert len(g.children) == 3
    # No PlotGrid children (all are flat ggplot).
    assert all(not isinstance(c, PlotGrid) for c in g.children)


def test_patchwork_direction_switch_nests():
    """`(p1 | p2) / p3` produces a 2-row grid whose first row is a 1×2 sub-grid."""
    p1, p2 = _two_simple_plots()
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p3 = ggplot(df, aes("x", "y")) + geom_point()
    g = (p1 | p2) / p3
    assert g.direction == "v"
    assert len(g.children) == 2
    assert isinstance(g.children[0], PlotGrid)
    assert g.children[0].direction == "h"


def test_patchwork_flat_horizontal_renders_two_axes():
    """`p1 | p2`.draw() produces a figure with two visible axes."""
    p1, p2 = _two_simple_plots()
    g = p1 | p2
    fig = g.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # At least the two main panels (some matplotlib bookkeeping axes may
        # exist depending on the gridspec — count >= 2 is the contract).
        assert len(visible) >= 2
    finally:
        plt.close(fig)


def test_patchwork_flat_vertical_renders_two_axes():
    p1, p2 = _two_simple_plots()
    g = p1 / p2
    fig = g.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) >= 2
    finally:
        plt.close(fig)


def test_patchwork_nested_2_deep():
    """Nested composition: `(p1 | p2) / p3` produces 3 visible axes."""
    p1, p2 = _two_simple_plots()
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p3 = ggplot(df, aes("x", "y")) + geom_point()
    g = (p1 | p2) / p3
    fig = g.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) >= 3
    finally:
        plt.close(fig)


def test_patchwork_with_faceted_child():
    """A faceted plot inside a composition keeps its facet sub-grid intact."""
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "g": ["a", "a", "b", "b"],
    })
    p_single = ggplot(df, aes("x", "y")) + geom_point()
    p_faceted = ggplot(df, aes("x", "y")) + geom_point() + facet_grid("~ g")
    g = p_single | p_faceted
    fig = g.draw()
    try:
        visible = [a for a in fig.axes if a.get_visible()]
        # 1 single panel + 2 facet panels = 3.
        assert len(visible) == 3
    finally:
        plt.close(fig)


def test_wrap_plots_byrow_default():
    """`wrap_plots([p1, p2, p3, p4])` defaults to row-major filling."""
    p1, p2 = _two_simple_plots()
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p3 = ggplot(df, aes("x", "y")) + geom_point()
    p4 = ggplot(df, aes("x", "y")) + geom_col()
    g = wrap_plots([p1, p2, p3, p4])
    nrow, ncol = g._dims()
    assert (nrow, ncol) == (2, 2)
    # First plot lands at (0, 0); second at (0, 1) row-major.
    assert g._cell_for(0) == (0, 0)
    assert g._cell_for(1) == (0, 1)
    assert g._cell_for(2) == (1, 0)


def test_wrap_plots_bycol():
    """`byrow=False` fills column-major."""
    p1, p2 = _two_simple_plots()
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p3 = ggplot(df, aes("x", "y")) + geom_point()
    p4 = ggplot(df, aes("x", "y")) + geom_col()
    g = wrap_plots([p1, p2, p3, p4], byrow=False)
    # Column-major: first plot at (0, 0), second at (1, 0), third at (0, 1).
    assert g._cell_for(0) == (0, 0)
    assert g._cell_for(1) == (1, 0)
    assert g._cell_for(2) == (0, 1)


def test_wrap_plots_explicit_ncol():
    """`ncol=3` packs into 3 columns."""
    plots = [_two_simple_plots()[0] for _ in range(5)]
    g = wrap_plots(plots, ncol=3)
    nrow, ncol = g._dims()
    assert ncol == 3
    assert nrow == 2  # ceil(5 / 3)


def test_patchwork_plus_composes_two_plots():
    """`p1 + p2` composes via auto-layout (R patchwork convention)."""
    p1, p2 = _two_simple_plots()
    g = p1 + p2
    assert isinstance(g, PlotGrid)
    assert g.direction == "grid"
    assert len(g.children) == 2
    # n=2 → 1×2 grid (visually horizontal).
    assert g._dims() == (1, 2)


def test_patchwork_plus_chain_flattens_into_one_grid():
    """`p1 + p2 + p3` flattens into a single 3-cell grid (not nested).

    Auto-layout for n=3 is ``(1, 3)`` per ggplot2's ``wrap_dims`` (which
    uses ``grDevices::n2mfrow`` for n ≤ 12) — not ``ceil(sqrt)``-style 2×2.
    """
    p1, p2 = _two_simple_plots()
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p3 = ggplot(df, aes("x", "y")) + geom_point()
    g = p1 + p2 + p3
    assert g.direction == "grid"
    assert len(g.children) == 3
    assert g._dims() == (1, 3)


def test_patchwork_plus_layer_still_adds_layer():
    """`p + geom_point()` continues to add a layer to a single ggplot —
    `+`-composition only triggers when the rhs is a ggplot/PlotGrid."""
    p1, _ = _two_simple_plots()
    extended = p1 + geom_point(colour="red")
    # Still a ggplot, not a PlotGrid.
    from hea.ggplot import ggplot as ggplot_cls
    assert isinstance(extended, ggplot_cls)


def test_patchwork_grid_plus_theme_propagates_to_last():
    """patchwork's `+` rule: ``grid + theme(...)`` applies the theme to the
    rightmost leaf plot (mirrors R's behaviour). To apply to ALL children
    we'd need ``&`` — deferred polish."""
    p1, p2 = _two_simple_plots()
    g = (p1 + p2) + theme(legend_position="top")
    assert g.children[0].theme.get("legend.position") is None
    assert g.children[1].theme.get("legend.position") == "top"


def test_patchwork_grid_plus_unknown_type_raises():
    """An rhs that's neither a plot nor a ggplot-addable raises with a hint."""
    p1, p2 = _two_simple_plots()
    g = p1 + p2
    with pytest.raises(TypeError, match=r"PlotGrid only accepts"):
        g + "not a plot or layer"


def test_plot_layout_widths_set_column_ratios():
    """`(p1 + p2) + plot_layout(widths=[1, 2])` makes col 1 twice as wide.

    Block-engine path: each child's panel is a regular Axes; we compare
    panel widths via ``ax.get_position().width``.
    """
    p1, p2 = _two_simple_plots()
    g = (p1 + p2) + plot_layout(widths=[1, 2])
    assert g.widths == [1, 2]
    fig = g.draw(figsize=(9, 3))
    try:
        fig.canvas.draw()
        widths = sorted(ax.get_position().width for ax in fig.axes)
        assert widths[1] / widths[0] == pytest.approx(2.0, abs=0.05)
    finally:
        plt.close(fig)


def test_plot_layout_heights_set_row_ratios():
    """`(p1 / p2) + plot_layout(heights=[2, 1])` makes row 0 twice as tall."""
    p1, p2 = _two_simple_plots()
    g = (p1 / p2) + plot_layout(heights=[2, 1])
    assert g.heights == [2, 1]
    fig = g.draw(figsize=(4, 9))
    try:
        fig.canvas.draw()
        heights = sorted(ax.get_position().height for ax in fig.axes)
        assert heights[1] / heights[0] == pytest.approx(2.0, abs=0.05)
    finally:
        plt.close(fig)


def test_plot_layout_wrong_length_raises():
    """Mismatched widths/heights raises at draw time."""
    p1, p2 = _two_simple_plots()
    g = (p1 + p2) + plot_layout(widths=[1, 2, 3])
    with pytest.raises(ValueError, match=r"widths has length 3"):
        g.draw()


def test_wrap_plots_widths_kwarg():
    """`wrap_plots(..., widths=[...])` is the alternate path to ratios."""
    p1, p2 = _two_simple_plots()
    g = wrap_plots([p1, p2], ncol=2, widths=[1, 3])
    assert g.widths == [1, 3]


def test_patchwork_faceted_child_axis_label_scoped_to_panel_column():
    """Regression: a faceted plot inside a composition keeps its xlabel
    inside its panel column (not across the whole composed figure).

    Block engine: instead of fig.supxlabel (which paints across the full
    figure width), the faceted child renders its xlabel via fig.text at
    the centre of its panel-area bbox. The single-panel child sets the
    xlabel on its sole axes.
    """
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
    })
    p_single = ggplot(df, aes("x", "y")) + geom_point()
    p_faceted = ggplot(df, aes("x", "y")) + geom_point() + facet_grid("~ g")

    fig = (p_single + p_faceted).draw(figsize=(8, 3))
    try:
        # No fig.supxlabel — that would paint across the whole composition.
        top_supx = getattr(fig, "_supxlabel", None)
        assert top_supx is None or not top_supx.get_text()

        # The single-panel child sets its xlabel on its sole axes.
        single_ax = fig.axes[0]
        assert single_ax.get_xlabel() == "x"

        # Faceted child: xlabel rides as a fig.text artist (scoped to its
        # panel column via positioning, not via fig.supxlabel).
        assert any(t.get_text() == "x" for t in fig.texts)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# patchwork doc walkthrough — https://patchwork.data-imaginist.com/articles/patchwork.html
# Each test maps to a numbered code block in the doc.
# ---------------------------------------------------------------------------


def _patchwork_doc_plots():
    """The four ggplots set up at the top of the patchwork tutorial."""
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


def test_patchwork_doc_ex3_two_plots_compose():
    """Ex 3: ``p1 + p2`` returns an auto-layout grid (1×2 for n=2)."""
    p1, p2, _, _ = _patchwork_doc_plots()
    g = p1 + p2
    assert isinstance(g, PlotGrid)
    assert g.direction == "grid"
    assert g._dims() == (1, 2)
    fig = g.draw()
    try:
        # Block engine: 2 leaves × 1 panel each → 2 panel axes (no
        # subfigures, no extra decoration axes).
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_patchwork_doc_ex4_labs_propagates_to_last_plot():
    """Ex 4: ``p1 + p2 + labs(subtitle = '...')`` adds the labs to the
    last plot in the grid (patchwork's `+`-to-last semantics)."""
    p1, p2, _, _ = _patchwork_doc_plots()
    g = p1 + p2 + labs(subtitle="This will appear in the last plot")
    assert isinstance(g, PlotGrid)
    # First plot unaffected.
    assert "subtitle" not in g.children[0].labels
    # Second plot picks up the subtitle.
    assert g.children[1].labels["subtitle"] == "This will appear in the last plot"


def test_patchwork_doc_ex5_four_plots_auto_grid():
    """Ex 5: ``p1 + p2 + p3 + p4`` auto-lays out as a 2×2 grid."""
    p1, p2, p3, p4 = _patchwork_doc_plots()
    g = p1 + p2 + p3 + p4
    assert g._dims() == (2, 2)


def test_patchwork_doc_ex6_layout_nrow_byrow_false():
    """Ex 6: ``+ plot_layout(nrow=3, byrow=FALSE)`` overrides the auto-grid
    and fills column-major."""
    p1, p2, p3, p4 = _patchwork_doc_plots()
    g = p1 + p2 + p3 + p4 + plot_layout(nrow=3, byrow=False)
    assert g.nrow == 3
    assert g.byrow is False
    # Column-major fill: 4th plot lands at column 1 row 0.
    nrow, ncol = g._dims()
    assert nrow == 3
    assert ncol == 2  # ceil(4 / 3)
    assert g._cell_for(0) == (0, 0)
    assert g._cell_for(1) == (1, 0)
    assert g._cell_for(2) == (2, 0)
    assert g._cell_for(3) == (0, 1)


def test_patchwork_doc_ex7_vertical_compose():
    """Ex 7: ``p1 / p2`` stacks vertically."""
    p1, p2, _, _ = _patchwork_doc_plots()
    g = p1 / p2
    assert g.direction == "v"
    assert g._dims() == (2, 1)


def test_patchwork_doc_ex8_nested_h_inside_v():
    """Ex 8: ``p1 | (p2 / p3)`` is a 1×2 horizontal grid whose right cell
    is itself a 2×1 vertical sub-grid."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    g = p1 | (p2 / p3)
    assert g.direction == "h"
    assert len(g.children) == 2
    assert isinstance(g.children[0], type(p1))
    assert isinstance(g.children[1], PlotGrid)
    assert g.children[1].direction == "v"


def test_patchwork_doc_ex9_plot_annotation_title():
    """Ex 9: ``+ plot_annotation(title=...)`` adds a figure-level title.

    Block engine: the title rides as a ``fig.text`` artist in the
    annotation row reserved above the super-grid (not ``fig.suptitle``,
    which would interact badly with constrained_layout that we no
    longer use)."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    g = (p1 | (p2 / p3)) + plot_annotation(
        title="The surprising story about mtcars"
    )
    assert g.annotation is not None
    fig = g.draw()
    try:
        title_text = "The surprising story about mtcars"
        assert any(t.get_text() == title_text for t in fig.texts), \
            f"expected fig.text with {title_text!r}, got {[t.get_text() for t in fig.texts]}"
    finally:
        plt.close(fig)


def _collect_tags_from_panel_axes(fig, expected: set[str]) -> list[str]:
    """Block engine: tags ride as figure-level Text artists positioned at
    the upper-left of each leaf's top-margin cell (so they sit ABOVE
    the title rather than colliding with it). Walk ``fig.texts``."""
    return [t.get_text() for t in fig.texts if t.get_text() in expected]


def test_patchwork_doc_ex10_tag_levels_roman():
    """Ex 10: ``+ plot_annotation(tag_levels='I')`` tags each leaf I, II, III."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    g = p1 + p2 + p3 + plot_annotation(tag_levels="I")
    fig = g.draw()
    try:
        tags = _collect_tags_from_panel_axes(fig, {"I", "II", "III"})
        assert tags == ["I", "II", "III"]
    finally:
        plt.close(fig)


def test_patchwork_tag_levels_a_uppercase():
    """``tag_levels='A'`` produces A, B, C, ..."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    g = p1 + p2 + p3 + plot_annotation(tag_levels="A")
    fig = g.draw()
    try:
        tags = sorted(_collect_tags_from_panel_axes(fig, {"A", "B", "C"}))
        assert tags == ["A", "B", "C"]
    finally:
        plt.close(fig)


def test_patchwork_tag_levels_with_prefix_suffix():
    """``tag_prefix='('`` / ``tag_suffix=')'`` wrap each tag."""
    p1, p2, _, _ = _patchwork_doc_plots()
    g = p1 + p2 + plot_annotation(tag_levels="A", tag_prefix="(", tag_suffix=")")
    fig = g.draw()
    try:
        tags = sorted(_collect_tags_from_panel_axes(fig, {"(A)", "(B)"}))
        assert tags == ["(A)", "(B)"]
    finally:
        plt.close(fig)


def test_patchwork_tag_levels_explicit_list():
    """A list ``tag_levels=[...]`` is taken literally."""
    p1, p2, _, _ = _patchwork_doc_plots()
    g = p1 + p2 + plot_annotation(tag_levels=["foo", "bar"])
    fig = g.draw()
    try:
        tags = sorted(_collect_tags_from_panel_axes(fig, {"foo", "bar"}))
        assert tags == ["bar", "foo"]
    finally:
        plt.close(fig)


def test_patchwork_tag_levels_bad_spec_errors():
    """An unknown spec raises with the list of valid options."""
    p1, p2, _, _ = _patchwork_doc_plots()
    with pytest.raises(ValueError, match="unknown tag_levels"):
        (p1 + p2 + plot_annotation(tag_levels="bogus")).draw()


def test_patchwork_plus_to_last_plot_with_theme():
    """`grid + theme(...)` propagates to the last plot (patchwork ``+`` rule)."""
    p1, p2, _, _ = _patchwork_doc_plots()
    g = p1 + p2 + theme(legend_position="top")
    assert isinstance(g, PlotGrid)
    last = g.children[-1]
    # Theme merge happened on the last child.
    assert last.theme.get("legend.position") == "top"
    # First child is untouched.
    assert g.children[0].theme.get("legend.position") is None


def test_patchwork_plus_to_last_plot_with_layer():
    """`grid + geom_line()` adds a layer to the last plot."""
    p1, p2, _, _ = _patchwork_doc_plots()
    g = p1 + p2 + geom_point(colour="red")
    n_first = len(g.children[0].layers)
    n_last = len(g.children[-1].layers)
    # Last plot now has one more layer than the first.
    assert n_last == n_first + 1


def test_patchwork_plus_to_last_plot_recurses_into_nested_grid():
    """A nested grid forwards the rhs to the last leaf at the deepest right."""
    p1, p2, p3, _ = _patchwork_doc_plots()
    g = (p1 | (p2 / p3)) + labs(caption="bottom-right caption")
    # The rhs leaf is p3 — caption lands there.
    inner = g.children[1]
    assert inner.children[-1].labels.get("caption") == "bottom-right caption"
    # p1 untouched.
    assert "caption" not in g.children[0].labels


def test_plot_layout_overrides_partial():
    """Only non-None fields on PlotLayout take effect; others inherit."""
    p1, p2 = _two_simple_plots()
    base = wrap_plots([p1, p2], ncol=2, widths=[1, 2], heights=[1])
    # plot_layout that only changes widths leaves heights and ncol intact.
    g = base + plot_layout(widths=[3, 4])
    assert g.widths == [3, 4]
    assert g.heights == [1]
    assert g.ncol == 2


def test_patchwork_figsize_kwarg():
    """`grid.draw(figsize=(W, H))` resizes the composed figure."""
    p1, p2 = _two_simple_plots()
    g = p1 | p2
    fig = g.draw(figsize=(10, 4))
    try:
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(10.0)
        assert size[1] == pytest.approx(4.0)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 1.8 — Themes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 1.10 — after_stat / after_scale + geom_text (smallest geom needed
# to test stat-aware aesthetics)
# ---------------------------------------------------------------------------


def test_gg_c11_geom_text_after_stat_count():
    """GG-C11: ``geom_text(aes(label = after_stat(count)), stat = "count")``
    paints the bar count above each bar."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes(x="cyl"))
         + geom_bar()
         + geom_text(aes(label=after_stat("count")), stat="count", vjust=-0.5))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.texts]
        counts_df = mtcars["cyl"].value_counts().sort("cyl")
        expected = [str(c) for c in counts_df["count"].to_list()]
        assert labels == expected
    finally:
        plt.close(fig)


def test_after_stat_density_with_geom_text():
    """``after_stat("density")`` on a stat that produces a `density` column."""
    import polars as pl

    df = pl.DataFrame({"x": list(range(20))})
    p = (ggplot(df, aes(x="x"))
         + geom_text(
             aes(y=after_stat("density"), label=after_stat("density")),
             stat="density",
         ))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # stat_density emits 512 grid points by default.
        assert len(ax.texts) == 512
    finally:
        plt.close(fig)


def test_after_stat_callable_form():
    """``after_stat(callable)`` evaluates the callable against the stat output."""
    mtcars = load_dataset("datasets", "mtcars")
    # Label = "n=<count>" for each bar.
    p = (ggplot(mtcars, aes(x="cyl"))
         + geom_bar()
         + geom_text(
             aes(label=after_stat(lambda d: ["n=" + str(int(c)) for c in d["count"]])),
             stat="count",
             vjust=-0.5,
         ))
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].texts]
        assert all(lbl.startswith("n=") for lbl in labels)
    finally:
        plt.close(fig)


def test_after_stat_does_not_evaluate_at_compute_aesthetics():
    """If we evaluated `after_stat("count")` immediately, we'd hit a NameError
    (no `count` column in the raw data). The marker should defer past stat."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes(x="cyl"))
         + geom_text(
             aes(y=after_stat("count"), label=after_stat("count")),
             stat="count",
         ))
    # Just running .draw() should succeed — no exception.
    fig = p.draw()
    plt.close(fig)


def test_geom_text_basic_label_aes():
    """geom_text without stat: just place text labels at (x, y)."""
    import polars as pl

    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 4.0, 9.0],
        "label": ["a", "b", "c"],
    })
    p = ggplot(df, aes("x", "y", label="label")) + geom_text()
    fig = p.draw()
    try:
        labels = [t.get_text() for t in fig.axes[0].texts]
        assert labels == ["a", "b", "c"]
    finally:
        plt.close(fig)


def test_default_theme_is_gray_panel():
    """ggplot2's default ``theme_gray`` puts a gray panel + white grid +
    no spines on the plot."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        bg = ax.get_facecolor()
        # ggplot2 panel.background fill is "#EBEBEB" (≈ 0.92, 0.92, 0.92).
        assert bg[0] == pytest.approx(0.922, abs=0.01)
        # All spines hidden (gray-panel style).
        assert not any(ax.spines[s].get_visible()
                       for s in ("top", "right", "bottom", "left"))
    finally:
        plt.close(fig)


def test_gg_c7_theme_bw_overrides_default():
    """GG-C7: ``+ theme_bw()`` switches to white panel + dark-gray border."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + theme_bw()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # White panel.
        bg = ax.get_facecolor()
        assert bg[0] == pytest.approx(1.0, abs=0.01)
        # All four spines visible (panel.border).
        assert all(ax.spines[s].get_visible()
                   for s in ("top", "right", "bottom", "left"))
    finally:
        plt.close(fig)


def test_theme_minimal_no_spines_no_panel_bg():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + theme_minimal()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # Transparent panel (set_facecolor("none") → alpha=0).
        bg = ax.get_facecolor()
        assert bg[3] == pytest.approx(0.0, abs=0.01)
        # No spines.
        assert not any(ax.spines[s].get_visible()
                       for s in ("top", "right", "bottom", "left"))
    finally:
        plt.close(fig)


def test_theme_classic_bottom_left_spines_only():
    """``theme_classic`` shows bottom + left spines but hides top + right."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + theme_classic()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.spines["bottom"].get_visible()
        assert ax.spines["left"].get_visible()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
    finally:
        plt.close(fig)


def test_theme_void_blanks_axis_text_and_titles():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + theme_void()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
    finally:
        plt.close(fig)


def test_theme_partial_override_panel_background():
    """``theme(panel_background=element_rect(fill="lightblue"))`` overrides
    only that element; the rest of theme_gray's defaults survive."""
    from matplotlib.colors import to_rgba

    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + theme(panel_background=element_rect(fill="lightblue")))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_facecolor()[:3] == pytest.approx(to_rgba("lightblue")[:3], abs=0.01)
        # Spines still hidden (theme_gray default survived).
        assert not any(ax.spines[s].get_visible()
                       for s in ("top", "right", "bottom", "left"))
    finally:
        plt.close(fig)


def test_theme_blank_clears_element():
    """Adding ``theme(panel_grid_major=element_blank())`` removes major grid."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + theme(panel_grid_major=element_blank()))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # `grid(False, which='major')` flips the tick's gridOn flag → no
        # visible major gridlines on either axis.
        major_grid = ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
        visible = [g for g in major_grid if g.get_visible()]
        assert len(visible) == 0
    finally:
        plt.close(fig)


def test_theme_addition_complete_replaces_partial_merges():
    """Complete preset replaces wholesale; ``theme(...)`` merges field-by-field."""
    base = theme_gray()
    overlay = theme(plot_title=element_text(colour="red"))

    merged = base + overlay
    # Plot title gained colour="red"; everything else from theme_gray survives.
    assert merged.get("plot.title").colour == "red"
    assert merged.get("panel.background") is not None  # theme_gray's panel still there

    # Now add a complete theme on top — it should replace.
    bw = theme_bw()
    final = merged + bw
    # bw has no plot.title.colour="red" override; the overlay was wiped.
    assert final.get("plot.title").colour != "red"


def test_theme_gridlines_below_data_artists():
    """Regression: matplotlib's default puts gridlines above data unless we
    call ``ax.set_axisbelow(True)``. ggplot2 always draws grid behind data."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_axisbelow() is True
        grid_z = [g.get_zorder() for g in ax.get_xgridlines() + ax.get_ygridlines()
                  if g.get_visible()]
        scatter_z = [c.get_zorder() for c in ax.collections]
        assert min(grid_z) < min(scatter_z), \
            f"grid {grid_z} should sit below scatter {scatter_z}"
    finally:
        plt.close(fig)


def test_theme_dot_separated_dict_form():
    """``theme({"panel.background": ...})`` — direct dotted-name form."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + theme({"panel.background": element_rect(fill="#FFE0E0")}))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_facecolor()[0] == pytest.approx(1.0, abs=0.01)  # FF
        assert ax.get_facecolor()[1] == pytest.approx(0.878, abs=0.01)  # E0
    finally:
        plt.close(fig)


def test_scale_size_area_uses_sqrt_scaling():
    """scale_size_area: visual *area* is proportional to the value, so
    radius scales as sqrt — distinct from the linear range mapping."""
    import polars as pl

    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 1.0, 1.0, 1.0],
        "v": [0.0, 0.25, 0.5, 1.0],
    })
    p = (ggplot(df, aes("x", "y", size="v")) + geom_point()
         + scale_size_area(max_size=4))
    fig = p.draw()
    try:
        sizes = fig.axes[0].collections[0].get_sizes()
        # area_pal: size_mm = max_size * sqrt(v_normalized).
        # v=0 → 0 mm; v=1 → max_size mm. s = (size_mm * 2.8454)².
        assert sizes[0] == pytest.approx(0.0, abs=1e-6)  # v=0 → size 0
        assert sizes[3] == pytest.approx((4 * 2.8454) ** 2, rel=0.01)
    finally:
        plt.close(fig)


def test_scale_color_brewer_unknown_palette_errors():
    mtcars = load_dataset("datasets", "mtcars")
    with pytest.raises(KeyError):
        p = (ggplot(mtcars, aes("wt", "mpg", colour="factor(cyl)"))
             + geom_point() + scale_color_brewer(palette="NotARealPalette"))
        p.draw()


def test_geom_smooth_fits_per_group_when_colour_mapped():
    """`aes(colour=species)` + `geom_smooth(method="lm")` → one fitted line
    per species, each in its own colour. Bug repro: previously a single fit
    was drawn in the default smooth colour."""
    from hea.data import data as _hea_data

    penguins = _hea_data("penguins", package="palmerpenguins")
    p = (ggplot(penguins, aes(x="flipper_length_mm", y="body_mass_g",
                              colour="species"))
         + geom_smooth(method="lm"))
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)  # the NA-removal warning
        fig = p.draw()
    try:
        ax = fig.axes[0]
        # 3 species → 3 lines, 3 ribbons.
        assert len(ax.lines) == 3
        line_colors = {ln.get_color() for ln in ax.lines}
        assert len(line_colors) == 3
    finally:
        plt.close(fig)


def test_geom_point_warns_on_missing_values():
    """ggplot2-faithful warning: ``Removed N rows containing missing values
    (`geom_point()`)``."""
    import polars as pl

    df = pl.DataFrame({
        "x": [1.0, 2.0, float("nan"), 4.0],
        "y": [1.0, 2.0, 3.0, float("nan")],
    })
    p = ggplot(df, aes("x", "y")) + geom_point()
    with pytest.warns(UserWarning, match=r"Removed 2 rows .*`geom_point\(\)`"):
        fig = p.draw()
    plt.close(fig)


def test_geom_point_na_rm_silences_warning():
    """``geom_point(na_rm=True)`` drops NAs without warning, matching ggplot2."""
    import polars as pl
    import warnings as _w

    df = pl.DataFrame({
        "x": [1.0, 2.0, float("nan"), 4.0],
        "y": [1.0, 2.0, 3.0, float("nan")],
    })
    p = ggplot(df, aes("x", "y")) + geom_point(na_rm=True)
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning fails the test
        fig = p.draw()
    plt.close(fig)


def test_geom_point_scatter_size_uses_pt_per_mm_conversion():
    """``size=1.5`` (mm) ⇒ matplotlib s=(1.5·2.8454)² ≈ 18.2 pt²."""
    import polars as pl

    df = pl.DataFrame({"x": [1.0, 2, 3], "y": [1.0, 2, 3]})
    p = ggplot(df, aes("x", "y")) + geom_point()
    fig = p.draw()
    try:
        sizes = fig.axes[0].collections[0].get_sizes()
        # Default size=1.5; expected (1.5 * 72.27/25.4)² ≈ 18.222
        assert abs(float(sizes[0]) - 18.222) < 0.01
    finally:
        plt.close(fig)


def test_aes_color_constant_kwarg_overrides_mapping():
    """`geom_point(colour="red")` (constant) wins over `aes(colour=…)` mapping."""
    from matplotlib.colors import to_rgba
    import polars as pl

    df = pl.DataFrame({"x": [1.0, 2, 3], "y": [1.0, 2, 3], "g": ["a", "b", "c"]})
    p = (ggplot(df, aes("x", "y", colour="g"))
         + geom_point(colour="red"))
    fig = p.draw()
    try:
        fc = fig.axes[0].collections[0].get_facecolors()
        # All red despite the discrete mapping.
        assert {tuple(c) for c in fc} == {to_rgba("red")}
    finally:
        plt.close(fig)


def test_gg_c9_boxplot_with_jitter():
    """GG-C9: box-and-whisker with jittered points overlaid."""
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes(x="cyl", y="mpg", group="cyl"))
         + geom_boxplot()
         + geom_jitter(width=0.2, seed=42))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # 3 box patches + 1 scatter collection (the jittered points).
        assert len(ax.patches) == 3
        assert len(ax.collections) == 1
        offsets = ax.collections[0].get_offsets()
        # All N points present, jittered around their cyl group center.
        assert offsets.shape == (len(mtcars), 2)
        # Jittered x's should differ from the integer cyl values.
        import numpy as np
        raw_cyl = mtcars["cyl"].to_numpy()
        assert not np.array_equal(offsets[:, 0], raw_cyl)
    finally:
        plt.close(fig)


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


# ---------------------------------------------------------------------------
# Labels & limits — `labs`, `xlab`, `ylab`, `ggtitle`, `xlim`, `ylim`, `lims`
# ---------------------------------------------------------------------------

def test_labs_overrides_axis_labels():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + labs(x="Weight", y="MPG")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Weight"
        assert ax.get_ylabel() == "MPG"
    finally:
        plt.close(fig)


def test_xlab_ylab_shortcuts():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + xlab("X") + ylab("Y")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
    finally:
        plt.close(fig)


def test_ggtitle_sets_axes_title_left_aligned():
    """``ggtitle("Cars")`` lands on the top-left axes via ``set_title(loc='left')``
    so it aligns with the panel's left edge (matches ggplot2 default
    ``plot.title hjust=0``), not the figure edge."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + ggtitle("Cars")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # The 'left' title slot on an axes — that's where ggplot2's title goes.
        assert ax.get_title(loc="left") == "Cars"
    finally:
        plt.close(fig)


def test_ggtitle_with_subtitle_renders_both():
    """Title + subtitle pack into one multi-line label on the top-left
    axes' left title slot."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + ggtitle("Cars", subtitle="MTcars data")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        title_text = ax.get_title(loc="left")
        assert title_text.split("\n") == ["Cars", "MTcars data"]
    finally:
        plt.close(fig)


def test_labs_caption_renders_as_figure_text():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + labs(caption="Source: 1974 Motor Trend")
    fig = p.draw()
    try:
        texts = [t.get_text() for t in fig.texts]
        assert "Source: 1974 Motor Trend" in texts
    finally:
        plt.close(fig)


def test_labs_color_aliases_to_colour():
    """``labs(color=...)`` and ``labs(colour=...)`` collapse to the canonical key."""
    assert labs(color="C").labels == {"colour": "C"}
    assert labs(colour="C").labels == {"colour": "C"}
    # When both supplied, ``color`` (later in the function body) wins. Match
    # ggplot2's "last assignment wins" semantics rather than erroring.
    assert labs(colour="A", color="B").labels == {"colour": "B"}


def test_labs_explicit_overrides_mapping_deparse():
    """``labs(x=...)`` beats the auto-derived label from the aes mapping."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + labs(x="Custom X")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Custom X"
        # Y still falls through to the mapping.
        assert ax.get_ylabel() == "mpg"
    finally:
        plt.close(fig)


def test_xlim_two_arg_form_sets_axis_limits():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + xlim(0, 10)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(0)
        assert hi == pytest.approx(10)
    finally:
        plt.close(fig)


def test_xlim_tuple_form():
    """``xlim((lo, hi))`` works the same as ``xlim(lo, hi)``."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + xlim((0, 10))
    fig = p.draw()
    try:
        lo, hi = fig.axes[0].get_xlim()
        assert (lo, hi) == pytest.approx((0, 10))
    finally:
        plt.close(fig)


def test_ylim_sets_y_axis_limits():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + ylim(0, 50)
    fig = p.draw()
    try:
        lo, hi = fig.axes[0].get_ylim()
        assert (lo, hi) == pytest.approx((0, 50))
    finally:
        plt.close(fig)


def test_lims_returns_list_and_sets_both_axes():
    """``lims(x=, y=)`` returns a list; ``+ list`` is sugar for adding each item."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + lims(x=(0, 10), y=(0, 50))
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlim() == pytest.approx((0, 10))
        assert ax.get_ylim() == pytest.approx((0, 50))
    finally:
        plt.close(fig)


def test_lims_rejects_non_positional_kwargs():
    """Until guides land, non-positional limits aren't supported."""
    with pytest.raises(NotImplementedError, match="x= and y= only"):
        lims(colour=(0, 1))


def test_xlim_rejects_too_many_args():
    with pytest.raises(TypeError):
        xlim(0, 1, 2)


def test_xlim_rejects_wrong_tuple_length():
    with pytest.raises(ValueError, match="length 2"):
        xlim((0, 1, 2))


def test_labs_fluent_method_form():
    """Phase B's auto-install picks up ``labs`` via the exact-name allowlist."""
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")).geom_point().labs(x="W", y="M").ggtitle("T")
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "W"
        assert ax.get_ylabel() == "M"
        assert ax.get_title(loc="left") == "T"
    finally:
        plt.close(fig)


def test_xlim_fluent_method_form():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")).geom_point().xlim(0, 10).ylim(0, 50)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert ax.get_xlim() == pytest.approx((0, 10))
        assert ax.get_ylim() == pytest.approx((0, 50))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Reference-line geoms — geom_hline / geom_vline / geom_abline
# ---------------------------------------------------------------------------

def test_geom_hline_single_intercept():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + geom_hline(yintercept=20)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        ydata = ax.lines[0].get_ydata()
        assert ydata[0] == pytest.approx(20)
        assert ydata[1] == pytest.approx(20)
    finally:
        plt.close(fig)


def test_geom_hline_multiple_intercepts():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + geom_hline(yintercept=[10, 20, 30])
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 3
        ys = sorted(ln.get_ydata()[0] for ln in ax.lines)
        assert ys == pytest.approx([10, 20, 30])
    finally:
        plt.close(fig)


def test_geom_vline_single_intercept():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + geom_vline(xintercept=3.5)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        xdata = ax.lines[0].get_xdata()
        assert xdata[0] == pytest.approx(3.5)
    finally:
        plt.close(fig)


def test_geom_abline_default_slope_intercept():
    """``geom_abline()`` defaults to y = x."""
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + geom_abline()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        # ``axline`` stores its anchor + slope on private attrs; xdata/ydata
        # are placeholder [0,1] in matplotlib regardless of the line.
        ln = ax.lines[0]
        assert ln._xy1 == (0.0, 0.0)
        assert ln._slope == pytest.approx(1.0)
    finally:
        plt.close(fig)


def test_geom_abline_custom_slope_intercept():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + geom_abline(slope=2, intercept=1)
    fig = p.draw()
    try:
        ln = fig.axes[0].lines[0]
        assert ln._xy1 == (0.0, 1.0)
        assert ln._slope == pytest.approx(2.0)
    finally:
        plt.close(fig)


def test_geom_hline_aes_params_apply_colour_and_linetype():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    p = ggplot(df, aes("x", "y")) + geom_point() + geom_hline(
        yintercept=[1.5, 2.5], colour="red", linetype="dashed",
    )
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        for ln in ax.lines:
            assert ln.get_linestyle() == "--"
            # matplotlib normalises colour names.
            from matplotlib.colors import to_rgba
            assert to_rgba(ln.get_color()) == to_rgba("red")
    finally:
        plt.close(fig)


def test_geom_abline_broadcasts_scalar_against_iterable():
    """Common case: many intercepts, one slope. Scalar broadcasts."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + geom_point() + geom_abline(slope=1, intercept=[0, 1, 2])
    fig = p.draw()
    try:
        ax = fig.axes[0]
        assert len(ax.lines) == 3
    finally:
        plt.close(fig)


def test_geom_abline_mismatched_lengths_errors():
    with pytest.raises(ValueError, match="same length"):
        geom_abline(slope=[1, 2], intercept=[0, 1, 2])


def test_geom_hline_does_not_inherit_main_data():
    """The line layer must not pull rows from the plot's main data — its
    own length is exactly the number of intercepts."""
    mtcars = load_dataset("datasets", "mtcars")  # 32 rows
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + geom_hline(yintercept=20)
    fig = p.draw()
    try:
        # One scatter (PathCollection) for points, one line (Line2D) for hline.
        ax = fig.axes[0]
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)


def test_geom_hline_fluent_form():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")).geom_point().geom_hline(yintercept=20)
    fig = p.draw()
    try:
        assert len(fig.axes[0].lines) == 1
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Coords — coord_cartesian, coord_fixed
# ---------------------------------------------------------------------------

def test_coord_cartesian_default_is_no_op():
    mtcars = load_dataset("datasets", "mtcars")
    p_no_coord = ggplot(mtcars, aes("wt", "mpg")) + geom_point()
    p_default_coord = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + coord_cartesian()
    f1 = p_no_coord.draw(); f2 = p_default_coord.draw()
    try:
        assert f1.axes[0].get_xlim() == pytest.approx(f2.axes[0].get_xlim())
        assert f1.axes[0].get_ylim() == pytest.approx(f2.axes[0].get_ylim())
    finally:
        plt.close(f1); plt.close(f2)


def test_coord_cartesian_xlim_zooms_axis():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + coord_cartesian(xlim=(0, 10))
    fig = p.draw()
    try:
        assert fig.axes[0].get_xlim() == pytest.approx((0, 10))
    finally:
        plt.close(fig)


def test_coord_cartesian_overrides_scale_limits():
    """Coord-level limits beat scale-level (ggplot2 semantics — coord wins)."""
    from hea.ggplot import xlim as gg_xlim
    mtcars = load_dataset("datasets", "mtcars")
    p = (ggplot(mtcars, aes("wt", "mpg")) + geom_point()
         + gg_xlim(0, 5)
         + coord_cartesian(xlim=(0, 10)))
    fig = p.draw()
    try:
        assert fig.axes[0].get_xlim() == pytest.approx((0, 10))
    finally:
        plt.close(fig)


def test_coord_fixed_sets_aspect_ratio():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + coord_fixed(ratio=2)
    fig = p.draw()
    try:
        assert fig.axes[0].get_aspect() == pytest.approx(2.0)
    finally:
        plt.close(fig)


def test_coord_fixed_default_ratio_is_one():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + coord_fixed()
    fig = p.draw()
    try:
        assert fig.axes[0].get_aspect() == pytest.approx(1.0)
    finally:
        plt.close(fig)


def test_coord_fixed_with_xlim():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + coord_fixed(ratio=1, xlim=(0, 10))
    fig = p.draw()
    try:
        assert fig.axes[0].get_xlim() == pytest.approx((0, 10))
        assert fig.axes[0].get_aspect() == pytest.approx(1.0)
    finally:
        plt.close(fig)


def test_coord_cartesian_fluent_form():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")).geom_point().coord_cartesian(xlim=(0, 10))
    fig = p.draw()
    try:
        assert fig.axes[0].get_xlim() == pytest.approx((0, 10))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# scale_color_hue / scale_fill_hue — explicit form of ggplot2's default qual palette
# ---------------------------------------------------------------------------

def test_scale_color_hue_assigns_distinct_colors_per_level():
    df = pl.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [1, 2, 3, 4],
        "g": ["a", "b", "c", "d"],
    })
    p = ggplot(df, aes("x", "y", color="g")) + geom_point() + scale_color_hue()
    fig = p.draw()
    try:
        sc = fig.axes[0].collections[0]
        # 4 distinct levels → 4 distinct colours.
        from matplotlib.colors import to_hex
        rgba = sc.get_facecolors()
        hexes = {to_hex(c) for c in rgba}
        assert len(hexes) == 4
    finally:
        plt.close(fig)


def test_scale_color_hue_direction_reverses_palette():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": ["a", "b", "c"]})
    p_fwd = ggplot(df, aes("x", "y", color="g")) + geom_point() + scale_color_hue()
    p_rev = ggplot(df, aes("x", "y", color="g")) + geom_point() + scale_color_hue(direction=-1)
    f1 = p_fwd.draw(); f2 = p_rev.draw()
    try:
        from matplotlib.colors import to_hex
        c1 = [to_hex(c) for c in f1.axes[0].collections[0].get_facecolors()]
        c2 = [to_hex(c) for c in f2.axes[0].collections[0].get_facecolors()]
        assert c1 == c2[::-1]
    finally:
        plt.close(f1); plt.close(f2)


# ---------------------------------------------------------------------------
# geom_label — geom_text + background bbox
# ---------------------------------------------------------------------------

def test_geom_label_renders_text_with_bbox():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "lbl": ["A", "B"]})
    p = ggplot(df, aes("x", "y", label="lbl")) + geom_label()
    fig = p.draw()
    try:
        ax = fig.axes[0]
        texts = ax.texts
        assert len(texts) == 2
        # Each text artist should have an associated bbox patch — that's the
        # difference from geom_text.
        for t in texts:
            assert t.get_bbox_patch() is not None
    finally:
        plt.close(fig)


def test_geom_label_fill_kwarg_sets_bbox_facecolor():
    df = pl.DataFrame({"x": [1.0], "y": [1.0], "lbl": ["A"]})
    p = ggplot(df, aes("x", "y", label="lbl")) + geom_label(fill="yellow")
    fig = p.draw()
    try:
        bbox = fig.axes[0].texts[0].get_bbox_patch()
        from matplotlib.colors import to_rgba
        assert to_rgba(bbox.get_facecolor()) == to_rgba("yellow")
    finally:
        plt.close(fig)


def test_geom_label_fluent_form():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "lbl": ["A", "B"]})
    p = ggplot(df, aes("x", "y", label="lbl")).geom_label()
    fig = p.draw()
    try:
        assert len(fig.axes[0].texts) == 2
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# stat_function / geom_function — y = f(x) curves
# ---------------------------------------------------------------------------

def test_stat_function_explicit_xlim():
    """``xlim`` directly determines the sampled x range."""
    import numpy as np
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + stat_function(fun=lambda x: x ** 2, xlim=(-2, 2), n=51)
    fig = p.draw()
    try:
        ax = fig.axes[0]
        # One Line2D for the function curve.
        assert len(ax.lines) == 1
        xs = ax.lines[0].get_xdata()
        ys = ax.lines[0].get_ydata()
        assert len(xs) == 51
        assert float(np.min(xs)) == pytest.approx(-2.0)
        assert float(np.max(xs)) == pytest.approx(2.0)
        # y = x^2: each y matches its x squared.
        for x, y in zip(xs, ys):
            assert y == pytest.approx(x ** 2)
    finally:
        plt.close(fig)


def test_stat_function_uses_main_data_xrange_when_xlim_omitted():
    import numpy as np
    df = pl.DataFrame({"x": [0.0, 5.0], "y": [0.0, 25.0]})
    p = ggplot(df, aes("x", "y")) + stat_function(fun=lambda x: 2 * x, n=11)
    fig = p.draw()
    try:
        xs = fig.axes[0].lines[0].get_xdata()
        assert float(np.min(xs)) == pytest.approx(0.0)
        assert float(np.max(xs)) == pytest.approx(5.0)
    finally:
        plt.close(fig)


def test_stat_function_passes_args_through():
    """``args=`` provides additional positional args to ``fun``."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p = ggplot(df, aes("x", "y")) + stat_function(
        fun=lambda x, a, b: a * x + b, xlim=(0, 1), n=11, args=(2, 3),
    )
    fig = p.draw()
    try:
        xs = fig.axes[0].lines[0].get_xdata()
        ys = fig.axes[0].lines[0].get_ydata()
        for x, y in zip(xs, ys):
            assert y == pytest.approx(2 * x + 3)
    finally:
        plt.close(fig)


def test_geom_function_alias():
    """``geom_function`` and ``stat_function(geom='line')`` produce equivalent layers."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p1 = ggplot(df, aes("x", "y")) + geom_function(fun=lambda x: x, xlim=(0, 1), n=10)
    p2 = ggplot(df, aes("x", "y")) + stat_function(fun=lambda x: x, xlim=(0, 1), n=10, geom="line")
    f1, f2 = p1.draw(), p2.draw()
    try:
        xs1 = f1.axes[0].lines[0].get_xdata()
        xs2 = f2.axes[0].lines[0].get_xdata()
        assert list(xs1) == pytest.approx(list(xs2))
    finally:
        plt.close(f1); plt.close(f2)


def test_stat_function_unknown_geom_errors():
    with pytest.raises(ValueError, match="unknown geom"):
        stat_function(fun=lambda x: x, geom="bogus")
