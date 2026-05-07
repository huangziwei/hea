"""Phase B (fluent ggplot API) — entry point + auto-install tests.

Plan: ``.claude/plans/method-based-ggplot-api.md`` Phase B.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import pytest

import hea
import hea.ggplot as hg
from hea.ggplot import aes, geom_point, geom_smooth, scale_x_log10, theme_minimal
from hea.ggplot.core import (
    _FLUENT_INSTALL_EXACT,
    _FLUENT_INSTALL_PREFIXES,
    _FLUENT_SKIP_EXACT,
    _FLUENT_SKIP_PREFIXES,
    _should_install_fluent,
    ggplot as ggplot_class,
)


@pytest.fixture
def df():
    return hea.DataFrame({"x": [1, 2, 3, 4], "y": [4.0, 5.0, 6.0, 7.0]})


def test_df_ggplot_returns_ggplot(df):
    """``df.ggplot(aes())`` produces a ``hea.ggplot.core.ggplot``."""
    p = df.ggplot(aes("x", "y"))
    assert isinstance(p, ggplot_class)
    assert p.data is df
    assert p.mapping is not None


def test_df_ggplot_chains_with_plus(df):
    """The ``+ geom_*()`` form keeps working off a fluent entry."""
    p = df.ggplot(aes("x", "y")) + geom_point()
    assert isinstance(p, ggplot_class)
    assert len(p.layers) == 1


def test_df_ggplot_equivalent_to_function_form(df):
    """``df.ggplot(aes())`` and ``ggplot(df, aes())`` produce the same plot shape."""
    p_fluent = df.ggplot(aes("x", "y")) + geom_point()
    p_func = ggplot_class(df, aes("x", "y")) + geom_point()
    assert p_fluent.data is p_func.data
    assert len(p_fluent.layers) == len(p_func.layers)
    assert type(p_fluent.layers[0]) is type(p_func.layers[0])


def test_df_ggplot_chains_through_verbs(df):
    """``df.filter(...).ggplot(aes())`` — the natural tidyverse chain."""
    p = df.filter(hea.col("x") > 1).ggplot(aes("x", "y")) + geom_point()
    assert isinstance(p, ggplot_class)
    # filter cut to 3 rows; ggplot snapshots the filtered frame.
    assert p.data.height == 3


def test_df_ggplot_captures_caller_frame_env(df):
    """``aes`` expressions reference user-defined helpers — must be captured.

    Critical regression test for the ``_env=`` wrinkle on ``ggplot.__init__``:
    without it, ``f_back`` from inside ``__init__`` would point at the
    ``DataFrame.ggplot`` wrapper, not the test function, and ``my_helper``
    would silently fail to resolve at build time.
    """
    def my_helper(s):
        return s * 2

    p = df.ggplot(aes(x="my_helper(x)", y="y"))
    assert "my_helper" in p.plot_env
    assert p.plot_env["my_helper"] is my_helper


def test_df_ggplot_renders_end_to_end(df):
    """End-to-end smoke test: build → render produces a matplotlib Figure."""
    fig = (df.ggplot(aes("x", "y")) + geom_point()).draw()
    assert fig is not None
    plt.close(fig)


def test_function_form_still_works_after_env_kwarg(df):
    """The ``_env=None`` path must keep the original frame-walk semantics."""
    def helper(s):
        return s + 1

    p = ggplot_class(df, aes(x="helper(x)", y="y"))
    assert "helper" in p.plot_env
    assert p.plot_env["helper"] is helper


# ---------------------------------------------------------------------------
# Auto-install: every layer-addable name in hea.ggplot.__all__ has a method
# ---------------------------------------------------------------------------


def _expected_fluent_names() -> set[str]:
    """Names from ``hea.ggplot.__all__`` that should be installed as methods."""
    return {n for n in hg.__all__ if _should_install_fluent(n)}


def _expected_skipped_names() -> set[str]:
    """Names from ``hea.ggplot.__all__`` that should NOT be installed."""
    return {n for n in hg.__all__ if not _should_install_fluent(n)}


def test_fluent_methods_installed_for_every_addable_name():
    """For every name in ``hea.ggplot.__all__`` that pattern-matches an
    install rule, ``ggplot.<name>`` must exist."""
    expected = _expected_fluent_names()
    missing = {name for name in expected if not hasattr(ggplot_class, name)}
    assert not missing, (
        f"Auto-install missed: {sorted(missing)}\n"
        "Check the install loop in hea/ggplot/core.py:_install_fluent_methods."
    )
    # Sanity: we matched a non-trivial number of names.
    assert len(expected) >= 30


def test_fluent_skip_list_not_installed():
    """Names matching the skip rules must NOT have methods on ``ggplot``."""
    skipped = _expected_skipped_names()
    leaked = {name for name in skipped if hasattr(ggplot_class, name)}
    # ``ggplot`` is in __all__ and would otherwise leak (the class itself
    # showing up as an attribute on instances). _FLUENT_SKIP_EXACT prevents it.
    # Note: some bound methods that pre-exist on the class (e.g. ``draw``,
    # ``show``) aren't in __all__ — they'd never enter this set.
    assert not leaked, f"These skipped names got installed: {sorted(leaked)}"


def test_should_install_fluent_predicate():
    """Direct unit test of the install/skip predicate."""
    # Install
    assert _should_install_fluent("geom_point")
    assert _should_install_fluent("stat_smooth")
    assert _should_install_fluent("scale_x_log10")
    assert _should_install_fluent("facet_wrap")
    assert _should_install_fluent("theme_minimal")
    assert _should_install_fluent("theme")  # exact

    # Skip
    assert not _should_install_fluent("position_dodge")
    assert not _should_install_fluent("element_text")
    assert not _should_install_fluent("after_stat")
    assert not _should_install_fluent("after_scale")
    assert not _should_install_fluent("aes")
    assert not _should_install_fluent("ggplot")


# ---------------------------------------------------------------------------
# Equivalence: method form produces the same plot as `+` form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,call_args,call_kwargs", [
    ("geom_point", (), {}),
    ("geom_smooth", (), {"method": "lm"}),
    ("theme_minimal", (), {}),
    ("scale_x_log10", (), {}),
    ("facet_wrap", ("g",), {}),
])
def test_fluent_method_matches_plus_form(df, name, call_args, call_kwargs):
    """For each representative name, the fluent and ``+`` forms produce
    structurally equivalent plots."""
    df_with_g = df.with_columns(g=hea.lit("a"))
    base_args = ("g",) if name == "facet_wrap" else ()
    fn = getattr(hg, name)

    p_fluent = getattr(df_with_g.ggplot(aes("x", "y")), name)(*call_args, **call_kwargs)
    p_plus = df_with_g.ggplot(aes("x", "y")) + fn(*call_args, **call_kwargs)

    # Both must be hea ggplot instances with the same number of layers.
    assert isinstance(p_fluent, ggplot_class)
    assert isinstance(p_plus, ggplot_class)
    assert len(p_fluent.layers) == len(p_plus.layers)
    if p_fluent.layers:
        assert type(p_fluent.layers[0]) is type(p_plus.layers[0])


def test_fluent_chain_full(df):
    """A realistic chain: tidyverse → ggplot → fluent layers + theme."""
    p = (
        df.filter(hea.col("x") > 1)
        .ggplot(aes("x", "y"))
        .geom_point()
        .geom_smooth(method="lm")
        .theme_minimal()
    )
    assert isinstance(p, ggplot_class)
    assert len(p.layers) == 2
    assert p.data.height == 3


def test_fluent_chain_renders(df):
    """End-to-end: a fluent-only chain renders to a Figure."""
    fig = (
        df.ggplot(aes("x", "y"))
        .geom_point()
        .theme_minimal()
    ).draw()
    assert fig is not None
    plt.close(fig)


def test_fluent_and_plus_interleaved(df):
    """Users can mix the two forms freely."""
    p = (
        df.ggplot(aes("x", "y"))
        .geom_point()
        + geom_smooth(method="lm")
        + theme_minimal()
    )
    assert isinstance(p, ggplot_class)
    assert len(p.layers) == 2


# ---------------------------------------------------------------------------
# Plot-level kwarg mappings — sugar for ``aes(...)`` at the ggplot entry
# point. Three forms are equivalent; mixing merges with kwargs winning.
# ---------------------------------------------------------------------------


def test_kwargs_form_equals_aes_form(df):
    """``df.ggplot(x="x", y="y")`` ≡ ``df.ggplot(aes(x="x", y="y"))``."""
    p_aes = df.ggplot(aes(x="x", y="y"))
    p_kw = df.ggplot(x="x", y="y")
    assert p_aes.mapping == p_kw.mapping


def test_kwargs_canonicalize_color_to_colour(df):
    """American ``color=`` kwarg canonicalizes to British ``colour``."""
    p = df.ggplot(x="x", y="y", color="g")
    assert p.mapping == aes(x="x", y="y", colour="g")


def test_kwargs_merge_with_aes_kwargs_win(df):
    """When both forms are given, kwargs override matching keys in aes()."""
    p = df.ggplot(aes(x="a", y="b"), x="z")
    assert p.mapping["x"] == "z"
    assert p.mapping["y"] == "b"


def test_top_level_ggplot_accepts_kwargs(df):
    """``ggplot(df, x="x")`` (function form) also accepts kwarg sugar."""
    from hea.ggplot import ggplot
    p1 = ggplot(df, aes(x="x", y="y"))
    p2 = ggplot(df, x="x", y="y")
    assert p1.mapping == p2.mapping


def test_layer_level_x_y_kwargs_work(df):
    """Regression: ``geom_point(x="x", y="y")`` should NOT silently lose
    data. The layer factory's narrow aes filter doesn't list ``x``/``y``
    so they previously landed in ``geom_params`` (= dropped). Layer now
    sweeps aes-named keys out of geom_params into aes_params.
    """
    from hea.ggplot import geom_point
    p = df.ggplot().geom_point(x="x", y="y")
    fig = p.draw()
    try:
        assert len(fig.axes[0].collections[0].get_offsets()) == len(df)
    finally:
        plt.close(fig)


def test_layer_level_color_constant_still_means_set(df):
    """``geom_point(color="red")`` — "red" is not a column → still SET."""
    from hea.ggplot import geom_point
    p = df.ggplot(x="x", y="y").geom_point(color="red")
    fig = p.draw()
    try:
        # Single facecolor matching matplotlib's "red".
        fc = fig.axes[0].collections[0].get_facecolors()
        assert tuple(round(c, 3) for c in fc[0]) == (1.0, 0.0, 0.0, 1.0)
    finally:
        plt.close(fig)


def test_layer_level_color_column_means_map(df):
    """``geom_point(color="g")`` where "g" is a column → MAP via promotion."""
    from hea.ggplot import geom_point
    df_g = df.with_columns(g=hea.lit("a"))
    p = df_g.ggplot(x="x", y="y").geom_point(color="g")
    # The promoted mapping should now contain colour: "g".
    # Inspect via build (no need to draw).
    from hea.ggplot.build import build
    bo = build(p)
    assert bo.aes_source.get("colour") == "g"
