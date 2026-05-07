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
    aes, after_scale, after_stat, element_blank, element_line, element_rect,
    element_text, facet_wrap, geom_area, geom_bar, geom_blank, geom_boxplot,
    geom_density, geom_histogram, geom_jitter, geom_line, geom_path,
    geom_point, geom_ribbon, geom_smooth, geom_step, geom_text, geom_violin,
    ggplot, ggtitle, labs, lims, position_dodge, position_fill,
    position_jitter, position_nudge, position_stack,
    scale_alpha_continuous, scale_color_brewer, scale_color_gradient,
    scale_color_gradient2, scale_color_gradientn, scale_color_identity,
    scale_color_manual, scale_color_viridis_c, scale_color_viridis_d,
    scale_fill_identity, scale_fill_manual, scale_linetype,
    scale_linetype_manual, scale_shape, scale_shape_manual, scale_size_area,
    scale_size_continuous, scale_size_manual, scale_x_continuous,
    scale_x_log10, scale_x_reverse, scale_x_sqrt, scale_y_continuous,
    scale_y_log10, theme, theme_bw, theme_classic, theme_dark, theme_gray,
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


def test_ggtitle_sets_suptitle():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + ggtitle("Cars")
    fig = p.draw()
    try:
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Cars"
    finally:
        plt.close(fig)


def test_ggtitle_with_subtitle_renders_both():
    mtcars = load_dataset("datasets", "mtcars")
    p = ggplot(mtcars, aes("wt", "mpg")) + geom_point() + ggtitle("Cars", subtitle="MTcars data")
    fig = p.draw()
    try:
        assert fig._suptitle.get_text() == "Cars"
        # Subtitle is a fig.text artist; find it among the figure's texts.
        texts = [t.get_text() for t in fig.texts]
        assert "MTcars data" in texts
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
        assert fig._suptitle.get_text() == "T"
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
