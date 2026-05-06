"""Phase B (fluent ggplot API) — entry point tests.

The auto-install loop and full method coverage land in a separate session;
this file currently covers just the ``df.ggplot(...)`` entry-point work
(plan: ``.claude/plans/method-based-ggplot-api.md`` Phase B, first checkbox).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import pytest

import hea
from hea.ggplot import aes, geom_point
from hea.ggplot.core import ggplot as ggplot_class


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
