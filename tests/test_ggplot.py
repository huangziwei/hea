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
    aes, geom_area, geom_bar, geom_blank, geom_boxplot, geom_density,
    geom_histogram, geom_jitter, geom_line, geom_path, geom_point,
    geom_ribbon, geom_smooth, geom_step, geom_violin, ggplot,
    position_dodge, position_fill, position_jitter, position_nudge,
    position_stack,
    scale_color_identity, scale_color_manual, scale_fill_identity,
    scale_fill_manual, scale_x_continuous, scale_x_log10, scale_x_reverse,
    scale_x_sqrt, scale_y_continuous, scale_y_log10,
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


def test_stat_smooth_gam_glm_unimplemented():
    import polars as pl
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    p = ggplot(df, aes("x", "y")) + geom_smooth(method="gam")
    with pytest.raises(NotImplementedError, match="method='gam'"):
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
