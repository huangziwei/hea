"""Tests for ``hea.dataframe`` — the chapter-3 tidyverse verbs.

Examples mirror ``dev/r4ds/data-transform.qmd`` so the test names tie
back to the source they implement.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import hea
from hea import DataFrame, GroupBy, desc, factor, tbl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    """Small frame with two groups, used in most tests."""
    return DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.fixture
def tied():
    """Frame with ties in ``x`` for slice_min/max with_ties tests."""
    return DataFrame({"g": ["a", "a", "a", "b"], "x": [1, 1, 2, 5]})


# ---------------------------------------------------------------------------
# Class identity / IS-A
# ---------------------------------------------------------------------------


def test_is_pl_dataframe_subclass(df):
    """The whole point of subclassing: hea functions accepting ``pl.DataFrame``
    must accept our subclass without conversion."""
    assert isinstance(df, pl.DataFrame)
    assert isinstance(df, DataFrame)


def test_hea_is_polars_superset():
    """``hea.*`` is a strict superset of ``polars.__all__`` so users can
    drop ``import polars as pl`` entirely. Regression-tested in detail in
    ``tests/test_polars_superset.py``; this is just the smoke check.
    """
    hea_public = {n for n in dir(hea) if not n.startswith("_")}
    assert hea_public >= set(pl.__all__)
    # And the hea-specific helpers desc / tbl / factor live under hea.*.
    assert hea.desc is desc
    assert hea.tbl is tbl
    assert hea.factor is factor


def test_data_returns_subclass():
    gala = hea.data("gala", package="faraway")
    assert isinstance(gala, DataFrame)
    assert isinstance(gala, pl.DataFrame)


def test_tbl_rewraps_plain_dataframe(df):
    plain = pl.DataFrame.with_columns(df, q=pl.col("x") + 100)
    # Native polars dropped the subclass; tbl() restores it.
    rewrapped = tbl(plain)
    assert isinstance(rewrapped, DataFrame)
    # tbl() on an already-wrapped frame is a no-op.
    assert tbl(rewrapped) is rewrapped


def test_methods_preserve_subclass(df):
    """Every tidyverse method must return our subclass so chains stay typed."""
    out = (
        df.filter(pl.col("x") > 1)
        .arrange("x")
        .distinct()
        .mutate(z=pl.col("x") * 2)
        .select("g", "z")
        .rename(group="g")
        .relocate("z")
    )
    assert isinstance(out, DataFrame)


# ---------------------------------------------------------------------------
# Row verbs
# ---------------------------------------------------------------------------


def test_filter_passthrough(df):
    out = df.filter(pl.col("x") > 3)
    assert out["x"].to_list() == [4, 5, 6]


def test_arrange_ascending(df):
    out = df.arrange("y")
    assert out["y"].to_list() == [10, 20, 30, 40, 50, 60]


def test_arrange_desc(df):
    out = df.arrange(desc("x"))
    assert out["x"].to_list() == [6, 5, 4, 3, 2, 1]


def test_desc_negates_list():
    """``desc()`` on values mirrors dplyr's ``-xtfrm(x)`` — used in
    composable forms like ``min_rank(desc(x))``."""
    import numpy as np
    from hea import min_rank
    out = desc([1, 5, 5, 17, 22, None])
    assert isinstance(out, np.ndarray)
    # NaN propagates through negation
    assert np.array_equal(out[:5], np.array([-1.0, -5.0, -5.0, -17.0, -22.0]))
    assert np.isnan(out[5])
    # The motivating use case — descending min_rank matches R reference
    ranks = min_rank(desc([1, 5, 5, 17, 22, None]))
    assert ranks[:5].tolist() == [5.0, 3.0, 3.0, 2.0, 1.0]
    assert np.isnan(ranks[5])


def test_desc_negates_series_and_expr():
    s = pl.Series([1, 5, 17, None])
    assert desc(s).to_list() == [-1, -5, -17, None]
    df = pl.DataFrame({"x": [1, 5, 17, None]})
    assert df.select(desc(pl.col("x")))["x"].to_list() == [-1, -5, -17, None]


def test_arrange_multi_with_desc(df):
    """Mix ascending and descending within one call."""
    out = df.arrange("g", desc("x"))
    assert out["g"].to_list() == ["a", "a", "a", "b", "b", "b"]
    assert out["x"].to_list() == [3, 2, 1, 6, 5, 4]


def test_arrange_puts_nulls_last():
    """dplyr semantics: NAs sort to the end regardless of direction.

    Polars' default puts nulls first; ``arrange`` overrides to match
    dplyr so the rows you actually want to look at land at the head.
    """
    df = DataFrame({"x": [3, None, 1, 2, None]})
    asc = df.arrange("x")
    assert asc["x"].to_list() == [1, 2, 3, None, None]
    dsc = df.arrange(desc("x"))
    assert dsc["x"].to_list() == [3, 2, 1, None, None]


def test_distinct_no_args_dedupes_full_row():
    df = DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    out = df.distinct()
    assert out.height == 2


def test_distinct_subset_drops_other_columns_by_default(df):
    """dplyr default: ``distinct(cols)`` returns just those columns."""
    out = df.distinct("g")
    assert out.columns == ["g"]
    assert out.height == 2


def test_distinct_subset_keep_all_true(df):
    """``keep_all=True`` mirrors dplyr's ``.keep_all = TRUE``."""
    out = df.distinct("g", keep_all=True)
    assert out.height == 2
    assert set(out.columns) == {"g", "x", "y"}


# ---------------------------------------------------------------------------
# Column verbs
# ---------------------------------------------------------------------------


def test_mutate_kwargs_auto_alias(df):
    """The motivating ergonomics fix: kwarg name becomes the column name."""
    out = df.mutate(z=pl.col("x") + pl.col("y"))
    assert "z" in out.columns
    assert out["z"].to_list() == [11, 22, 33, 44, 55, 66]


def test_mutate_positional_passthrough(df):
    out = df.mutate(pl.col("x").alias("xx"))
    assert "xx" in out.columns


def test_mutate_before_after_mutually_exclusive(df):
    with pytest.raises(ValueError, match="_before= or _after="):
        df.mutate(z=pl.col("x"), _before="g", _after="x")


def test_mutate_before_places_new_column(df):
    out = df.mutate(z=pl.col("x") + 1, _before="g")
    assert out.columns == ["z", "g", "x", "y"]


def test_mutate_after_places_new_column(df):
    out = df.mutate(z=pl.col("x") + 1, _after="g")
    assert out.columns == ["g", "z", "x", "y"]


def test_mutate_before_position_int(df):
    """``_before=1`` matches dplyr's ``.before = 1`` (before the first column)."""
    out = df.mutate(z=pl.col("x") + 1, _before=1)
    assert out.columns == ["z", "g", "x", "y"]


def test_mutate_after_position_int(df):
    out = df.mutate(z=pl.col("x") + 1, _after=2)
    # _after=2 → after the 2nd column (x), position 2.
    assert out.columns == ["g", "x", "z", "y"]


def test_mutate_position_out_of_range(df):
    with pytest.raises(ValueError, match="out of range"):
        df.mutate(z=pl.col("x"), _before=99)
    with pytest.raises(ValueError, match="out of range"):
        df.mutate(z=pl.col("x"), _before=0)  # dplyr is 1-indexed; 0 is invalid


def test_mutate_keep_none_drops_existing(df):
    out = df.mutate(z=pl.col("x") + 1, _keep="none")
    assert out.columns == ["z"]


def test_mutate_keep_used_keeps_referenced_only():
    """The r4ds example: keep originals referenced by new expressions
    plus the new columns. Self-references (gain_per_hour → gain) don't
    cause `gain` to be doubled — it's already a new column."""
    df = DataFrame({
        "dep_delay": [1.0], "arr_delay": [2.0], "air_time": [60.0],
        "extra": [99],  # NOT referenced — should be dropped
    })
    out = df.mutate(
        gain=pl.col("dep_delay") - pl.col("arr_delay"),
        hours=pl.col("air_time") / 60,
        gain_per_hour=pl.col("gain") / pl.col("hours"),
        _keep="used",
    )
    assert out.columns == [
        "dep_delay", "arr_delay", "air_time",  # referenced originals
        "gain", "hours", "gain_per_hour",      # new
    ]


def test_mutate_is_sequential():
    """Later expressions can refer to columns created earlier in the same call.
    Polars' ``with_columns`` is parallel; we chain so dplyr semantics hold."""
    df = DataFrame({"x": [1, 2, 3]})
    out = df.mutate(
        y=pl.col("x") * 2,
        z=pl.col("y") + 1,
    )
    assert out["z"].to_list() == [3, 5, 7]


def test_mutate_keep_unused_drops_referenced():
    """Inverse of `used`: drop the originals referenced; keep the rest."""
    df = DataFrame({
        "dep_delay": [1.0], "arr_delay": [2.0], "air_time": [60.0],
        "extra": [99],
    })
    out = df.mutate(
        gain=pl.col("dep_delay") - pl.col("arr_delay"),
        _keep="unused",
    )
    # dep_delay, arr_delay are dropped (referenced); air_time and extra survive.
    assert out.columns == ["air_time", "extra", "gain"]


def test_mutate_by_is_windowed(df):
    """``_by`` makes each expression compute within groups; row count preserved."""
    out = df.mutate(x_mean=pl.col("x").mean(), _by="g")
    assert out.height == df.height
    assert out.filter(pl.col("g") == "a")["x_mean"].unique().to_list() == [2.0]
    assert out.filter(pl.col("g") == "b")["x_mean"].unique().to_list() == [5.0]


def test_cols_between():
    """dplyr's ``year:day`` slice syntax via list helper."""
    d = DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    assert d.cols_between("a", "c") == ["a", "b", "c"]
    # Order-insensitive: end before start still yields the same range.
    assert d.cols_between("c", "a") == ["a", "b", "c"]
    # Splat into select.
    assert d.select(d.cols_between("a", "c")).columns == ["a", "b", "c"]
    # Negate via pl.exclude.
    assert d.select(pl.exclude(d.cols_between("a", "c"))).columns == ["d"]


def test_cols_between_missing_column():
    d = DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="not in frame"):
        d.cols_between("a", "z")


def test_select_with_kwargs_renames(df):
    out = df.select("g", x_plus_one=pl.col("x") + 1)
    assert out.columns == ["g", "x_plus_one"]
    assert out["x_plus_one"].to_list() == [2, 3, 4, 5, 6, 7]


def test_select_kwarg_string_is_column_ref(df):
    """dplyr's ``select(tail_num = tailnum)``: bare-string RHS is a column ref."""
    out = df.select(group="g", val="x")
    assert out.columns == ["group", "val"]
    assert out["group"].to_list() == ["a", "a", "a", "b", "b", "b"]
    assert out["val"].to_list() == [1, 2, 3, 4, 5, 6]


def test_rename_kwargs_new_equals_old(df):
    out = df.rename(group="g")
    assert "group" in out.columns and "g" not in out.columns


def test_rename_dict_polars_style(df):
    out = df.rename({"g": "group"})
    assert "group" in out.columns and "g" not in out.columns


def test_rename_rejects_both_forms(df):
    with pytest.raises(ValueError, match="dict or kwargs"):
        df.rename({"g": "group"}, x="xx")


def test_relocate_default_moves_to_front(df):
    out = df.relocate("y")
    assert out.columns == ["y", "g", "x"]


def test_relocate_before(df):
    out = df.relocate("y", _before="x")
    assert out.columns == ["g", "y", "x"]


def test_relocate_after(df):
    out = df.relocate("y", _after="g")
    assert out.columns == ["g", "y", "x"]


def test_relocate_position_int(df):
    """Positional anchors apply to the columns *after* the moves are removed.

    ``_before=1`` of the remaining columns (g, x) → at the very front.
    """
    out = df.relocate("y", _before=1)
    assert out.columns == ["y", "g", "x"]
    out2 = df.relocate("y", _after=2)
    assert out2.columns == ["g", "x", "y"]


def test_relocate_accepts_list_from_cols_between():
    """``cols_between`` output is a list — relocate flattens it."""
    d = DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    out = d.relocate(d.cols_between("a", "b"), _after="d")
    assert out.columns == ["c", "d", "a", "b"]


def test_relocate_accepts_selector():
    """Polars selectors expand against the frame schema."""
    import polars.selectors as cs
    d = DataFrame({"arr_a": [1], "arr_b": [2], "dep_a": [3], "x": [4]})
    out = d.relocate(cs.starts_with("arr"), _before="x")
    # arr columns moved to right before x; dep_a stays where it was.
    assert out.columns == ["dep_a", "arr_a", "arr_b", "x"]


def test_relocate_preserves_frame_order_of_movers():
    """dplyr behavior: movers retain their original relative order."""
    d = DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    # Specified b,a but original order is a,b — result should be a,b,c,d.
    out = d.relocate("b", "a")
    assert out.columns == ["a", "b", "c", "d"]


def test_relocate_anchor_must_exist(df):
    with pytest.raises(ValueError):
        df.relocate("y", _before="nope")


def test_relocate_anchor_cannot_be_moving(df):
    with pytest.raises(ValueError):
        df.relocate("y", _before="y")


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def test_group_by_returns_groupby(df):
    g = df.group_by("g")
    assert isinstance(g, GroupBy)
    assert g.groups == ["g"]


def test_group_by_summarize_named(df):
    """Reproduces the user's motivating example."""
    out = df.group_by("g").summarize(travel=pl.col("x").mean())
    assert out["g"].to_list() == ["a", "b"]
    assert out["travel"].to_list() == [2.0, 5.0]


def test_group_by_maintain_order_default():
    """tibble convention: group_by preserves first-seen order, not lex sort."""
    df = DataFrame({"g": ["b", "a", "b", "a"], "x": [1, 2, 3, 4]})
    out = df.group_by("g").summarize(n=pl.len())
    assert out["g"].to_list() == ["b", "a"]


def test_group_by_derived_kwarg_column():
    """dplyr's ``group_by(name = expr)`` — kwarg defines a new column to group on."""
    df = DataFrame({"sched_dep_time": [515, 540, 600, 700, 715, 800]})
    out = (
        df.group_by(hour=pl.col("sched_dep_time") // 100)
        .summarize(n=pl.len())
        .arrange("hour")
    )
    assert out["hour"].to_list() == [5, 6, 7, 8]
    assert out["n"].to_list() == [2, 1, 2, 1]


def test_group_by_mixed_positional_and_derived():
    """Positional col plus kwarg-derived col are both grouping keys."""
    df = DataFrame({
        "g": ["a", "a", "a", "b"],
        "x": [10, 12, 25, 11],
    })
    out = (
        df.group_by("g", bucket=pl.col("x") // 10)
        .summarize(n=pl.len())
        .arrange("g", "bucket")
    )
    assert out["g"].to_list() == ["a", "a", "b"]
    assert out["bucket"].to_list() == [1, 2, 1]
    assert out["n"].to_list() == [2, 1, 1]


def test_group_by_maintain_order_option_still_works():
    """Backwards compat: bare ``maintain_order=`` is treated as a polars option,
    not a derived column."""
    df = DataFrame({"g": ["b", "a", "b", "a"], "x": [1, 2, 3, 4]})
    # maintain_order=False relaxes the order guarantee — just check the
    # grouping still works and "maintain_order" did not become a column.
    out = df.group_by("g", maintain_order=False).summarize(n=pl.len())
    assert set(out.columns) == {"g", "n"}
    assert sorted(out["g"].to_list()) == ["a", "b"]


def test_group_by_underscore_maintain_order_escape_hatch():
    """``_maintain_order=`` frees the ``maintain_order`` name for a derived col."""
    df = DataFrame({"x": [1, 2, 3, 4]})
    out = (
        df.group_by(maintain_order=pl.col("x") % 2, _maintain_order=True)
        .summarize(n=pl.len())
        .arrange("maintain_order")
    )
    assert out.columns == ["maintain_order", "n"]
    assert out["maintain_order"].to_list() == [0, 1]
    assert out["n"].to_list() == [2, 2]


def test_group_by_no_args_raises():
    df = DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="at least one column"):
        df.group_by()


def test_summarize_no_group_collapses_to_one_row(df):
    out = df.summarize(total=pl.col("x").sum())
    assert out.height == 1
    assert out["total"].item() == 21


def test_summarize_by_kwarg(df):
    """The dplyr 1.1 ``.by =`` per-call grouping form."""
    out = df.summarize(mean_x=pl.col("x").mean(), _by="g")
    assert sorted(out["g"].to_list()) == ["a", "b"]


def test_summarise_british_alias(df):
    out = df.summarise(s=pl.col("x").sum())
    assert out["s"].item() == 21


def test_count_with_columns(df):
    out = df.count("g")
    assert out["g"].to_list() == ["a", "b"]
    assert out["n"].to_list() == [3, 3]


def test_count_no_columns_returns_total(df):
    out = df.count()
    assert out.height == 1 and out["n"].item() == 6


def test_count_sort_descending():
    df = DataFrame({"g": ["a", "b", "b", "b", "c", "c"]})
    out = df.count("g", sort=True)
    assert out["g"].to_list() == ["b", "c", "a"]


def test_count_custom_name(df):
    out = df.count("g", name="freq")
    assert "freq" in out.columns


def test_groupby_count(df):
    out = df.group_by("g").count()
    assert out["n"].to_list() == [3, 3]


def test_groupby_mutate_is_windowed(df):
    """Exercise 6f: ``group_by(g) |> mutate(...)`` is windowed, not collapsing."""
    out = df.group_by("g").mutate(mean_x=pl.col("x").mean())
    assert out.height == df.height
    assert out.filter(pl.col("g") == "a")["mean_x"].unique().to_list() == [2.0]


def test_ungroup_returns_underlying(df):
    g = df.group_by("g")
    assert g.ungroup() is df


def test_ungroup_on_dataframe_is_noop(df):
    """Symmetric API: ungroup on a flat frame is a no-op."""
    assert df.ungroup() is df


# ---------------------------------------------------------------------------
# Slice family
# ---------------------------------------------------------------------------


def test_slice_head(df):
    out = df.slice_head(2)
    assert out.height == 2


def test_slice_tail(df):
    out = df.slice_tail(1)
    assert out["x"].to_list() == [6]


def test_slice_min_with_ties_keeps_all(tied):
    out = tied.slice_min("x", n=1)
    # x=1 appears twice; with_ties=True keeps both.
    assert out.height == 2


def test_slice_min_no_ties(tied):
    out = tied.slice_min("x", n=1, with_ties=False)
    assert out.height == 1


def test_slice_max_with_ties(df):
    out = df.slice_max("y", n=1)
    assert out["y"].to_list() == [60]


def test_groupby_slice_min(df):
    out = df.group_by("g").slice_min("x", n=1)
    # Smallest x per group: a→1, b→4.
    assert sorted(zip(out["g"].to_list(), out["x"].to_list())) == [("a", 1), ("b", 4)]


def test_groupby_slice_max(df):
    out = df.group_by("g").slice_max("y", n=1)
    assert sorted(zip(out["g"].to_list(), out["y"].to_list())) == [("a", 30), ("b", 60)]


def test_groupby_slice_max_keeps_all_null_group():
    """dplyr parity: a group with only null values still yields ``n`` rows.

    Critical for r4ds chapter-3 example: ``flights |> group_by(dest)
    |> slice_max(arr_delay, n=1)`` → 108 rows because LGA has a single
    null-arr_delay row that must survive.
    """
    df = DataFrame({
        "g": ["a", "a", "a", "b"],
        "x": [3, 1, 2, None],  # b has only a null
    })
    out = df.group_by("g").slice_max("x", n=1)
    # a's max=3 (1 row); b's only row is null → kept.
    assert out.height == 2
    b_row = out.filter(pl.col("g") == "b")
    assert b_row.height == 1
    assert b_row["x"].item() is None


def test_slice_max_keeps_all_null_when_only_rows():
    """Same fix applies to the ungrouped slice_max."""
    df = DataFrame({"x": [None, None]})
    assert df.slice_max("x", n=1).height == 2  # both NAs tied at the cutoff
    assert df.slice_max("x", n=1, with_ties=False).height == 1


def test_groupby_slice_max_n_gt_1(df):
    out = df.group_by("g").slice_max("y", n=2)
    assert out.height == 4


def test_groupby_slice_head_per_group(df):
    out = df.group_by("g").slice_head(1)
    assert out["g"].to_list() == ["a", "b"]


def test_slice_sample_n(df):
    out = df.slice_sample(n=3, seed=0)
    assert out.height == 3


def test_slice_sample_prop(df):
    out = df.slice_sample(prop=0.5, seed=0)
    assert out.height == 3


def test_slice_sample_requires_one_of_n_prop(df):
    with pytest.raises(ValueError):
        df.slice_sample()
    with pytest.raises(ValueError):
        df.slice_sample(n=1, prop=0.1)


def test_groupby_slice_sample(df):
    out = df.group_by("g").slice_sample(n=2, seed=0)
    assert out.height == 4
    assert sorted(out["g"].unique().to_list()) == ["a", "b"]


# ---------------------------------------------------------------------------
# End-to-end integration with hea.lm
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Chapter 5 — pivots and pull
# ---------------------------------------------------------------------------


def test_pivot_longer_basic():
    """Smallest example from the chapter: bp1/bp2 columns → long.

    Row-major output matches dplyr: each input row's pivoted values
    appear contiguously, in the original column order.
    """
    d = DataFrame({"id": ["A", "B", "C"], "bp1": [100, 140, 120], "bp2": [120, 115, 125]})
    out = d.pivot_longer(
        ["bp1", "bp2"],
        names_to="measurement",
        values_to="value",
    )
    assert out.shape == (6, 3)
    assert out.columns == ["id", "measurement", "value"]
    assert out["id"].to_list() == ["A", "A", "B", "B", "C", "C"]
    assert out["measurement"].to_list() == ["bp1", "bp2", "bp1", "bp2", "bp1", "bp2"]


def test_pivot_longer_billboard_row_order():
    """dplyr orders pivoted rows so all weeks for one song come first.

    Polars' raw ``unpivot`` is column-major (all-of-wk1 first); we
    reorder by tagging the input row index and sorting at the end.
    """
    billboard = hea.data("billboard", package="tidyr")
    out = (
        billboard.pivot_longer(
            pl.selectors.starts_with("wk"),
            names_to="week",
            values_to="rank",
        )
        .slice_head(76)  # all weeks of the first song
    )
    # All 76 rows belong to the first artist (2 Pac).
    assert out["artist"].n_unique() == 1
    assert out["week"].to_list()[:3] == ["wk1", "wk2", "wk3"]


def test_pivot_longer_billboard():
    """The chapter's billboard example end-to-end."""
    billboard = hea.data("billboard", package="tidyr")
    long = billboard.pivot_longer(
        pl.selectors.starts_with("wk"),
        names_to="week",
        values_to="rank",
        values_drop_na=True,
    )
    # 76 wk-cols, 317 songs, but with values_drop_na the result is much smaller.
    assert "week" in long.columns and "rank" in long.columns
    assert long["rank"].null_count() == 0
    # The drop_na collapse must remove rows; raw billboard has 317*76 = 24092
    # cells but only ~5300 have non-null rank.
    assert long.height < 24092
    assert long.height > 1000


def test_pivot_longer_names_prefix():
    """``names_prefix`` strips a regex prefix from each name before assignment."""
    d = DataFrame({"id": [1], "wk1": [10], "wk2": [20], "wk3": [30]})
    out = d.pivot_longer(
        pl.selectors.starts_with("wk"),
        names_to="week",
        values_to="rank",
        names_prefix="wk",
    )
    assert out["week"].to_list() == ["1", "2", "3"]


def test_pivot_longer_names_sep_multi():
    """The who2 case: multi-piece name split into multiple new columns."""
    d = DataFrame({
        "country": ["X", "X"],
        "year": [2000, 2001],
        "sp_m_014": [1, 5],
        "sp_f_014": [2, 6],
        "ep_m_014": [3, 7],
        "ep_f_014": [4, 8],
    })
    out = d.pivot_longer(
        pl.exclude(["country", "year"]),
        names_to=["diagnosis", "gender", "age"],
        names_sep="_",
        values_to="count",
    )
    # 4 pivoted cols × 2 rows = 8 long rows
    assert out.height == 8
    assert {"diagnosis", "gender", "age", "count"}.issubset(out.columns)
    assert sorted(out["diagnosis"].unique().to_list()) == ["ep", "sp"]
    assert sorted(out["gender"].unique().to_list()) == ["f", "m"]


def test_pivot_longer_names_pattern():
    """``names_pattern`` regex extracts groups into the listed names."""
    d = DataFrame({"id": [1, 2], "a_2020": [10, 30], "a_2021": [20, 40]})
    out = d.pivot_longer(
        pl.exclude("id"),
        names_to=["letter", "year"],
        names_pattern=r"([a-z]+)_(\d+)",
        values_to="value",
    )
    assert out.height == 4
    assert sorted(out["year"].unique().to_list()) == ["2020", "2021"]
    assert out["letter"].unique().to_list() == ["a"]


def test_pivot_longer_dot_value_sentinel():
    """The household example: ``.value`` makes name-pieces into output columns."""
    d = DataFrame({
        "family": [1, 2],
        "name_child1": ["A", "C"],
        "name_child2": ["B", None],
        "dob_child1": ["2000", "2001"],
        "dob_child2": ["2010", None],
    })
    out = d.pivot_longer(
        pl.exclude("family"),
        names_to=[".value", "child"],
        names_sep="_",
        values_drop_na=True,
    )
    # Two original "values" columns (name, dob) survive as output columns.
    # child column gets the second piece of each name.
    assert set(out.columns) == {"family", "child", "name", "dob"}
    # Family 2 had child2 = null name+dob → drop_na keeps only child1.
    fam2 = out.filter(pl.col("family") == 2)
    assert fam2.height == 1
    assert fam2["child"].item() == "child1"


def test_pivot_longer_rejects_both_sep_and_pattern():
    d = DataFrame({"id": [1], "a_1": [10]})
    with pytest.raises(ValueError, match="names_sep or names_pattern"):
        d.pivot_longer(
            ["a_1"],
            names_to=["x", "y"],
            names_sep="_",
            names_pattern=r"(\w+)_(\d+)",
        )


def test_pivot_longer_requires_split_when_multi():
    d = DataFrame({"id": [1], "a_1": [10]})
    with pytest.raises(ValueError, match="names_sep= or names_pattern="):
        d.pivot_longer(["a_1"], names_to=["x", "y"])


def test_pivot_wider_basic():
    """Inverse of pivot_longer for the same bp dataset."""
    long = DataFrame({
        "id": ["A", "B", "B", "A", "A"],
        "measurement": ["bp1", "bp1", "bp2", "bp2", "bp3"],
        "value": [100, 140, 115, 120, 105],
    })
    wide = long.pivot_wider(names_from="measurement", values_from="value")
    assert wide.height == 2
    assert {"id", "bp1", "bp2", "bp3"}.issubset(wide.columns)


def test_pivot_wider_id_cols_selector():
    """``id_cols`` accepts a selector — the cms example pattern."""
    d = DataFrame({
        "org_id": [1, 1, 2, 2],
        "org_name": ["a", "a", "b", "b"],
        "metric": ["x", "y", "x", "y"],
        "score": [10, 20, 30, 40],
    })
    out = d.pivot_wider(
        id_cols=pl.selectors.starts_with("org"),
        names_from="metric",
        values_from="score",
    )
    assert out.height == 2
    assert set(out.columns) == {"org_id", "org_name", "x", "y"}


def test_pivot_wider_values_fill():
    long = DataFrame({"id": [1, 1, 2], "k": ["a", "b", "a"], "v": [10, 20, 30]})
    out = long.pivot_wider(names_from="k", values_from="v", values_fill=0)
    # Row 2 has no "b" → filled with 0.
    assert out.filter(pl.col("id") == 2)["b"].item() == 0


def test_pivot_wider_names_prefix():
    long = DataFrame({"id": [1, 2], "k": ["a", "b"], "v": [10, 20]})
    out = long.pivot_wider(names_from="k", values_from="v", names_prefix="m_")
    assert "m_a" in out.columns and "m_b" in out.columns


def test_pivot_round_trip():
    """longer → wider returns the original frame (modulo column order)."""
    wide = DataFrame({"id": [1, 2, 3], "x": [10, 20, 30], "y": [100, 200, 300]})
    long = wide.pivot_longer(["x", "y"], names_to="k", values_to="v")
    back = long.pivot_wider(names_from="k", values_from="v")
    assert back.sort("id").select(["id", "x", "y"]).equals(
        wide.sort("id").select(["id", "x", "y"])
    )


def test_pull_by_name(df):
    s = df.pull("x")
    assert isinstance(s, pl.Series)
    assert s.to_list() == [1, 2, 3, 4, 5, 6]


def test_pull_default_last_column(df):
    """No arg: dplyr default returns the last column."""
    s = df.pull()
    assert s.name == "y"


def test_pull_int_position(df):
    """1-indexed position; negative counts from the right."""
    assert df.pull(1).name == "g"
    assert df.pull(-1).name == "y"


# ---------------------------------------------------------------------------
# End-to-end integration with hea.lm
# ---------------------------------------------------------------------------


def test_chain_then_lm():
    """A full tidyverse chain still produces something hea.lm accepts."""
    gala = hea.data("gala", package="faraway")
    sub = (
        gala.filter(pl.col("Area") > 0)
        .mutate(log_area=pl.col("Area").log())
        .select("Species", "log_area", "Elevation")
    )
    m = hea.lm("Species ~ log_area + Elevation", sub)
    assert m is not None


# ---------------------------------------------------------------------------
# summary() — R's per-column summary
# ---------------------------------------------------------------------------


def _entries(summary, col):
    for b in summary.blocks:
        if b.name == col:
            return b.entries
    raise KeyError(col)


def test_summary_numeric_six_stats():
    """Numeric columns get Min / 1st Qu / Median / Mean / 3rd Qu / Max
    using R's ``quantile`` type 7 (linear interpolation)."""
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    e = _entries(df.summary(width=80), "x")
    assert [lbl for lbl, _ in e] == [
        "Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.",
    ]
    by_label = dict(e)
    assert by_label["Min."] == "1"
    assert by_label["Max."] == "5"
    assert by_label["Median"] == "3"
    # quantile(0.25, linear) on 1..5 = 2 — not the polars default ("nearest")
    assert by_label["1st Qu."] == "2"
    assert by_label["3rd Qu."] == "4"


def test_summary_format_keeps_integer_means_verbatim():
    """Integer columns whose Mean has a small fractional part still
    display as integers (matches R's ``format.default``: signif at
    digits=4 yields integer-valued stats, so D=0 and the original value
    rounds to integer)."""
    g = hea.data("gavote", package="faraway")
    e = dict(_entries(g.summary(width=120), "votes"))
    # Mean = 16331.025 — R prints "16331", not signif-rounded "16330".
    assert e["Mean"] == "16331"
    # Max stays as the literal integer.
    assert e["Max."] == "263211"


def test_summary_format_decimals_when_signif_is_non_integer():
    """When any signif'd value has a fractional part, the whole block
    aligns at common decimals (gavote 'other' Mean=381.7 forces 1
    decimal across the column)."""
    g = hea.data("gavote", package="faraway")
    e = dict(_entries(g.summary(width=120), "other"))
    assert e["Mean"] == "381.7"
    assert e["Min."] == "5.0"
    assert e["Max."] == "7920.0"


def test_summary_factor_levels_in_category_order():
    """Enum columns show all levels with counts, in the dtype's category
    order — matches R's ``summary.factor`` on a factor with ``levels=``."""
    g = hea.data("gavote", package="faraway")
    e = _entries(g.summary(), "equip")
    # gavote.equip levels are LEVER, OS-CC, OS-PC, PAPER, PUNCH.
    assert e == [
        ("LEVER", "74"),
        ("OS-CC", "44"),
        ("OS-PC", "22"),
        ("PAPER", "2"),
        ("PUNCH", "17"),
    ]


def test_summary_factor_other_collapse():
    """When more levels than ``maxsum`` exist, the (maxsum-1) most
    populous are kept and the rest pool into ``(Other)``."""
    levels = list("abcdefghij")  # 10 levels
    counts = [50, 30, 20, 10, 8, 6, 4, 3, 2, 1]
    data = [lvl for lvl, n in zip(levels, counts) for _ in range(n)]
    df = DataFrame({"f": pl.Series(data, dtype=pl.Enum(levels))})
    e = _entries(df.summary(maxsum=7), "f")
    # Top 6 by count + (Other) = 7 entries.
    labels = [lbl for lbl, _ in e]
    assert labels == ["a", "b", "c", "d", "e", "f", "(Other)"]
    counts_out = dict(e)
    assert counts_out["a"] == "50"
    # (Other) sums g+h+i+j = 4+3+2+1 = 10.
    assert counts_out["(Other)"] == "10"


def test_summary_string_length_class_mode():
    """Polars ``String`` columns get R's character-summary shape:
    Length / Class / Mode rather than per-value counts."""
    df = DataFrame({"s": ["foo", "bar", "baz", "foo"]})
    e = _entries(df.summary(), "s")
    assert e == [
        ("Length", "4"),
        ("Class", "character"),
        ("Mode", "character"),
    ]


def test_summary_boolean():
    """Boolean columns: Mode / FALSE / TRUE counts."""
    df = DataFrame({"b": [True, False, True, True, False]})
    e = _entries(df.summary(), "b")
    assert e == [
        ("Mode", "logical"),
        ("FALSE", "2"),
        ("TRUE", "3"),
    ]


def test_summary_appends_nas_when_present():
    """An NA's row appears for any dtype with nulls; absent otherwise."""
    df_with = DataFrame({"x": [1.0, 2.0, None, 4.0]})
    e_with = _entries(df_with.summary(), "x")
    assert ("NA's", "1") in e_with

    df_clean = DataFrame({"x": [1.0, 2.0, 3.0]})
    e_clean = _entries(df_clean.summary(), "x")
    assert all(lbl != "NA's" for lbl, _ in e_clean)


def test_summary_factor_reserves_slot_for_nas():
    """When a factor has nulls, the maxsum budget is reduced by 1 so
    the NA's row fits — matches R's summary.factor."""
    levels = list("abcdef")
    data = [lvl for lvl, n in zip(levels, [10, 8, 6, 4, 3, 2]) for _ in range(n)]
    data += [None, None]  # 2 nulls
    df = DataFrame({"f": pl.Series(data, dtype=pl.Enum(levels))})
    e = _entries(df.summary(maxsum=5), "f")
    # 6 levels > slots(=4 because nulls reserved one) → keep top 3, then (Other), then NA's.
    labels = [lbl for lbl, _ in e]
    assert labels == ["a", "b", "c", "(Other)", "NA's"]


def test_summary_repr_packs_blocks_horizontally():
    """The repr lays multiple columns side-by-side within ``width``;
    columns that would overflow wrap to a new row group separated by
    a blank line."""
    df = DataFrame({"x": [1, 2, 3], "y": [10, 20, 30], "z": [100, 200, 300]})
    s = repr(df.summary(width=200))
    # All three column headers on the first line — wide enough to fit.
    first_line = s.split("\n")[0]
    assert "x" in first_line and "y" in first_line and "z" in first_line

    narrow = repr(df.summary(width=30))
    # Narrow forces wrapping; expect a blank line separating row groups.
    assert "" in narrow.split("\n")


def test_summary_repr_preserves_block_alignment():
    """The R-style 'label:value' alignment within each block: labels
    left-aligned, values right-aligned, both padded to the block max."""
    df = DataFrame({"x": [1, 2, 3, 4, 5]})
    s = repr(df.summary(width=80))
    lines = s.split("\n")
    # Find the body lines (skip the centered header).
    body = [l for l in lines if ":" in l]
    # Every line has the colon at the same column.
    colon_positions = {l.index(":") for l in body}
    assert len(colon_positions) == 1


def test_summary_empty_dataframe():
    """No columns → empty string repr; doesn't blow up."""
    df = DataFrame({})
    s = repr(df.summary())
    assert s == ""


def test_summary_all_null_numeric():
    """All-null numeric column: stats are NA, plus an NA's count row."""
    df = DataFrame({"x": pl.Series([None, None, None], dtype=pl.Float64)})
    e = _entries(df.summary(), "x")
    by_label = dict(e)
    assert by_label["Min."] == "NA"
    assert by_label["Mean"] == "NA"
    assert by_label["NA's"] == "3"


def test_summary_dates_stay_as_dates():
    """``Date`` input shouldn't bleed datetime '00:00:00' suffixes
    into Median / Quartile / Mean rows just because polars promotes
    those stats to datetime internally."""
    import datetime as dt
    df = DataFrame({
        "d": [dt.date(2020, 1, 1), dt.date(2020, 6, 1), dt.date(2020, 12, 31)],
    })
    e = dict(_entries(df.summary(), "d"))
    for lbl in ("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."):
        assert " " not in e[lbl], f"{lbl} = {e[lbl]!r} has time component"


# ===========================================================================
# Tidyverse helpers (dplyr / readr / stringr / tibble)
#
# These live in hea/dataframe.py with the rest of the tidyverse port (R.py
# is base R only). Each section compares behavior to a documented R / dplyr
# reference value so the port is self-checking against the source language.
# ===========================================================================


# ---- if_else / case_when (dplyr) -----------------------------------------


def test_if_else_chains_with_parse_number():
    """``parse_number(if_else(age == "five", "5", age))`` — the canonical
    tidyverse one-liner, end-to-end."""
    df = DataFrame({"age": ["25", "five", "30 yo", "$1,234.56", "12.5", None]})
    out = df.mutate(
        age_num=hea.parse_number(
            hea.if_else(pl.col("age") == "five", "5", pl.col("age"))
        )
    )
    assert out["age_num"].to_list() == [25.0, 5.0, 30.0, 1234.56, 12.5, None]


def test_if_else_null_cond_yields_null_by_default():
    """dplyr semantics: ``NA in condition → NA out`` (polars' raw when/then
    instead routes null through ``otherwise``)."""
    df = DataFrame({"cond": [True, None, False], "x": [1, 2, 3], "y": [10, 20, 30]})
    out = df.mutate(z=hea.if_else(pl.col("cond"), pl.col("x"), pl.col("y")))
    assert out["z"].to_list() == [1, None, 30]


def test_if_else_missing_override():
    """``missing=`` overrides the null-out value."""
    df = DataFrame({"cond": [True, None, False], "x": [1, 2, 3], "y": [10, 20, 30]})
    out = df.mutate(
        z=hea.if_else(pl.col("cond"), pl.col("x"), pl.col("y"), missing=-1)
    )
    assert out["z"].to_list() == [1, -1, 30]


def test_if_else_bare_strings_are_lifted_to_lit():
    """``then("YES")`` must produce the literal "YES", not a column ref —
    polars' raw ``when/then("x")`` would try to resolve "x" as a column."""
    df = DataFrame({"g": ["a", "b", "c"]})
    out = df.mutate(label=hea.if_else(pl.col("g") == "a", "YES", "NO"))
    assert out["label"].to_list() == ["YES", "NO", "NO"]


def test_case_when_drive_type():
    """Canonical R-for-Data-Science example: drv code → full label."""
    df = DataFrame({"drv": ["f", "r", "4", "f", "r"]})
    out = df.mutate(
        drive_type=hea.case_when(
            (pl.col("drv") == "f", "front-wheel drive"),
            (pl.col("drv") == "r", "rear-wheel drive"),
            (pl.col("drv") == "4", "4-wheel drive"),
        )
    )
    assert out["drive_type"].to_list() == [
        "front-wheel drive", "rear-wheel drive", "4-wheel drive",
        "front-wheel drive", "rear-wheel drive",
    ]


def test_case_when_unmatched_default_null():
    """No matching branch + no ``default`` → null (dplyr's ``.default = NA``)."""
    df = DataFrame({"x": [1, 2, 3, 4]})
    out = df.mutate(g=hea.case_when(
        (pl.col("x") < 2, "low"), (pl.col("x") > 3, "high"),
    ))
    assert out["g"].to_list() == ["low", None, None, "high"]


def test_case_when_default_fills_unmatched():
    df = DataFrame({"x": [1, 2, 3, 4]})
    out = df.mutate(g=hea.case_when(
        (pl.col("x") < 2, "low"), (pl.col("x") > 3, "high"), default="mid",
    ))
    assert out["g"].to_list() == ["low", "mid", "mid", "high"]


def test_case_when_null_condition_falls_through_to_default():
    """Null condition (NA == anything → NA) falls through to ``default`` —
    matches dplyr 1.1+ semantics. Use ``if_else`` for null-in → null-out."""
    df = DataFrame({"x": [1, None, 4]})
    out = df.mutate(g=hea.case_when(
        (pl.col("x") < 2, "low"), (pl.col("x") > 3, "high"), default="mid",
    ))
    assert out["g"].to_list() == ["low", "mid", "high"]


def test_case_when_bare_strings_lifted():
    df = DataFrame({"label": ["x", "y", "z"]})
    out = df.mutate(g=hea.case_when(
        (pl.col("label") == "x", "first"), default="other",
    ))
    assert out["g"].to_list() == ["first", "other", "other"]


def test_case_when_usage_errors():
    with pytest.raises(TypeError, match="at least one"):
        hea.case_when(default="x")
    with pytest.raises(TypeError, match="must be a .* tuple"):
        hea.case_when(pl.col("x") < 2)


# ---- parse_number / parse_double (readr) ---------------------------------


def test_parse_double_strict_list():
    """Whole string must be a valid double; otherwise null."""
    assert hea.parse_double(["1.2", "5.6", "1e3", "-0.5"]) == [
        1.2, 5.6, 1000.0, -0.5
    ]
    assert hea.parse_double(["$1.99", "1,234", "abc", ""]) == [
        None, None, None, None
    ]


def test_parse_double_tuple_returns_list():
    assert hea.parse_double(("1.2", "5.6")) == [1.2, 5.6]


def test_parse_double_series_in_series_out():
    out = hea.parse_double(pl.Series(["1.2", "abc"]))
    assert isinstance(out, pl.Series)
    assert out.dtype == pl.Float64
    assert out.to_list() == [1.2, None]


def test_parse_double_expr_in_expr_out():
    df = pl.DataFrame({"x": ["1.2", "5.6", "abc"]})
    out = df.select(hea.parse_double(pl.col("x")).alias("v"))["v"]
    assert out.to_list() == [1.2, 5.6, None]


def test_parse_number_list_returns_list():
    """Regression: list input must not hit ``.cast`` AttributeError."""
    assert hea.parse_number(["$1,234.56", "30 yo", "five", "5"]) == [
        1234.56, 30.0, None, 5.0
    ]


def test_parse_number_series_in_series_out():
    out = hea.parse_number(pl.Series(["$1.99", "abc"]))
    assert isinstance(out, pl.Series)
    assert out.to_list() == [1.99, None]


def test_parse_number_inside_mutate():
    """parse_number alone: handles currency, units, unparseable → null."""
    df = DataFrame({"age": ["25", "five", "30 yo", "$1,234.56", "12.5", None]})
    out = df.mutate(age_num=hea.parse_number(pl.col("age")))
    assert out["age_num"].to_list() == [25.0, None, 30.0, 1234.56, 12.5, None]


# ---- str_wrap (stringr) --------------------------------------------------


def test_str_wrap_single_string():
    out = hea.str_wrap("hello world this is a long string", width=15)
    assert out == "hello world\nthis is a long\nstring"


def test_str_wrap_list_passthrough():
    out = hea.str_wrap(["short", "much longer line here"], width=10)
    assert out == ["short", "much\nlonger\nline here"]


# ---- glimpse (tibble) ----------------------------------------------------


def test_glimpse_dispatches_on_dataframe():
    """polars / hea DataFrame has .glimpse() — just confirm we don't error."""
    df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    hea.glimpse(df)


# ---- rank family (dplyr) -------------------------------------------------
#
# Reference values from R 4.6 / dplyr 1.x:
#   x <- c(1, 5, 5, 17, 22, NA)
#   min_rank(x)     -> c(1, 2, 2, 4, 5, NA)
#   dense_rank(x)   -> c(1, 2, 2, 3, 4, NA)
#   percent_rank(x) -> c(0, 0.25, 0.25, 0.75, 1.0, NA)
#   cume_dist(x)    -> c(0.2, 0.6, 0.6, 0.8, 1.0, NA)
#   ntile(x, 3)     -> c(1, 1, 2, 2, 3, NA)

_RANK_X = [1, 5, 5, 17, 22, None]


def test_min_rank_eager_list():
    """List input returns a polars Series with null preserved (not NaN), so
    the result composes cleanly with ``mutate`` (no NaN→null mismatch)."""
    out = hea.min_rank(_RANK_X)
    assert isinstance(out, pl.Series)
    assert out.to_list() == [1.0, 2.0, 2.0, 4.0, 5.0, None]


def test_dense_rank_eager_list():
    assert hea.dense_rank(_RANK_X).to_list() == [1.0, 2.0, 2.0, 3.0, 4.0, None]


def test_percent_rank_eager_list():
    assert hea.percent_rank(_RANK_X).to_list() == [
        0.0, 0.25, 0.25, 0.75, 1.0, None
    ]


def test_cume_dist_eager_list():
    assert hea.cume_dist(_RANK_X).to_list() == [0.2, 0.6, 0.6, 0.8, 1.0, None]


def test_ntile_eager_list():
    assert hea.ntile(_RANK_X, 3).to_list() == [1.0, 1.0, 2.0, 2.0, 3.0, None]


def test_ntile_eager_size_imbalance():
    """``ntile(1:10, 3)`` puts the extras in the earlier buckets."""
    assert hea.ntile(list(range(1, 11)), 3).to_list() == [
        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0
    ]
    assert hea.ntile(list(range(1, 11)), 4).to_list() == [
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0
    ]


def test_eager_rank_into_mutate_produces_null_not_nan():
    """Regression: a list-input rank result must surface as polars null
    (matching R's uniform NA), not as a literal NaN value in the column."""
    df = pl.DataFrame({"x": _RANK_X})
    out = df.with_columns(
        rn=hea.row_number(_RANK_X),
        mr=hea.min_rank(_RANK_X),
        dr=hea.dense_rank(_RANK_X),
        pr=hea.percent_rank(_RANK_X),
        cd=hea.cume_dist(_RANK_X),
        nt=hea.ntile(_RANK_X, 3),
    )
    for col in ("rn", "mr", "dr", "pr", "cd", "nt"):
        assert out[col].null_count() == 1, f"{col} should have null, not NaN"
        assert out[col][-1] is None


def test_rank_verbs_series_in_series_out():
    s = pl.Series(_RANK_X)
    for fn in (hea.min_rank, hea.dense_rank, hea.percent_rank, hea.cume_dist):
        out = fn(s)
        assert isinstance(out, pl.Series), f"{fn.__name__} did not return Series"
    assert isinstance(hea.ntile(s, 3), pl.Series)


def test_rank_verbs_expr_in_mutate():
    """Composes inside ``mutate()`` — the canonical dplyr use case."""
    df = pl.DataFrame({"x": _RANK_X})
    out = df.select(
        mr=hea.min_rank(pl.col("x")),
        dr=hea.dense_rank(pl.col("x")),
        pr=hea.percent_rank(pl.col("x")),
        cd=hea.cume_dist(pl.col("x")),
        nt=hea.ntile(pl.col("x"), 3),
    )
    assert out["mr"].to_list() == [1, 2, 2, 4, 5, None]
    assert out["dr"].to_list() == [1, 2, 2, 3, 4, None]
    assert out["pr"].to_list() == pytest.approx(
        [0.0, 0.25, 0.25, 0.75, 1.0, None], nan_ok=True
    )
    assert out["cd"].to_list() == pytest.approx(
        [0.2, 0.6, 0.6, 0.8, 1.0, None], nan_ok=True
    )
    assert out["nt"].to_list() == [1, 1, 2, 2, 3, None]


def test_row_number_no_arg_is_position_expr():
    """``row_number()`` (no args) returns the 1-based position expression."""
    df = pl.DataFrame({"v": [10, 20, 30, None]})
    out = df.select(rn=hea.row_number())["rn"].to_list()
    assert out == [1, 2, 3, 4]


def test_row_number_eager_list_is_ordinal_rank():
    """``row_number(x)`` is ``rank(x, "ordinal")`` — ties by first appearance."""
    # R reference: row_number(c(3, 1, 1, 2)) -> c(4, 1, 2, 3)
    assert hea.row_number([3, 1, 1, 2]).to_list() == [4.0, 1.0, 2.0, 3.0]
    assert hea.row_number(_RANK_X).to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, None]


def test_row_number_expr_form():
    """Inside ``select`` / ``mutate`` with a column ref."""
    df = pl.DataFrame({"x": _RANK_X})
    out = df.select(rn=hea.row_number(pl.col("x")))["rn"].to_list()
    assert out == [1, 2, 3, 4, 5, None]


# ---- lag / lead (dplyr) --------------------------------------------------
#
# R reference (dplyr):
#   x <- c(2, 5, 11, 11, 19, 35)
#   lag(x)                    -> c(NA, 2, 5, 11, 11, 19)
#   lag(x, n=2, default=0)    -> c(0, 0, 2, 5, 11, 11)
#   lead(x)                   -> c(5, 11, 11, 19, 35, NA)
#   lead(x, n=2, default=-1)  -> c(11, 11, 19, 35, -1, -1)


_LAG_X = [2, 5, 11, 11, 19, 35]


def test_lag_default_n1_fills_null():
    assert hea.lag(_LAG_X).to_list() == [None, 2, 5, 11, 11, 19]


def test_lag_with_n_and_default():
    assert hea.lag(_LAG_X, n=2, default=0).to_list() == [0, 0, 2, 5, 11, 11]


def test_lead_default_n1_fills_null():
    assert hea.lead(_LAG_X).to_list() == [5, 11, 11, 19, 35, None]


def test_lead_with_n_and_default():
    assert hea.lead(_LAG_X, n=2, default=-1).to_list() == [
        11, 11, 19, 35, -1, -1
    ]


def test_lag_inside_mutate_per_group():
    """``mutate`` handles the grouping; lag operates per-group automatically."""
    df = DataFrame({"g": ["a", "a", "b", "b", "b"], "x": [1, 2, 3, 4, 5]})
    out = df.group_by("g").mutate(lx=hea.lag(pl.col("x"))).sort("g", "x")
    # R: group_by(g) %>% mutate(lx = lag(x)) yields [NA, 1, NA, 3, 4]
    assert out["lx"].to_list() == [None, 1, None, 3, 4]


def test_lag_with_order_by_string_column():
    """``order_by`` reorders, computes lag, restores. R reference (verified):
        tibble(t=c(3,1,2), x=c("c","a","b")) %>% mutate(prev=lag(x, order_by=t))
        -> c("b", NA, "a")  # row at t=3 gets the row at t=2; t=1 has no predecessor
    """
    df = DataFrame({"t": [3, 1, 2], "x": ["c", "a", "b"]})
    out = df.mutate(prev=hea.lag(pl.col("x"), order_by="t"))
    assert out["prev"].to_list() == ["b", None, "a"]


def test_lag_lead_dispatch_ndarray():
    """ndarray input returns ndarray (matches min_rank etc.)."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = hea.lag(arr)
    assert isinstance(out, np.ndarray)
    # n=1 default=None → leading NaN
    assert np.isnan(out[0]) and out[1:].tolist() == [1.0, 2.0, 3.0]


# ---- between / na_if / near (dplyr) --------------------------------------


def test_between_eager_list():
    """``between([1,5,10,15,20], 5, 15)`` -> [F, T, T, T, F] (R dplyr)."""
    out = hea.between([1, 5, 10, 15, 20], 5, 15)
    assert out.to_list() == [False, True, True, True, False]


def test_between_propagates_null():
    """NA in input propagates (matches R)."""
    out = hea.between([None, 5, 10], 5, 10)
    assert out.to_list() == [None, True, True]


def test_between_expr_inside_mutate():
    df = DataFrame({"x": [1, 5, 10, 15, 20]})
    out = df.mutate(in_range=hea.between(pl.col("x"), 5, 15))
    assert out["in_range"].to_list() == [False, True, True, True, False]


def test_na_if_replaces_value_with_null():
    assert hea.na_if([1, 0, 3, 0], 0).to_list() == [1, None, 3, None]
    assert hea.na_if(["a", "", "b"], "").to_list() == ["a", None, "b"]


def test_na_if_expr_inside_mutate():
    df = DataFrame({"y": [1, 0, 3, 0]})
    out = df.mutate(y2=hea.na_if(pl.col("y"), 0))
    assert out["y2"].to_list() == [1, None, 3, None]


def test_near_scalar_within_default_tol():
    """Default ``tol`` ≈ 1.49e-8 (sqrt(.Machine$double.eps))."""
    assert hea.near(1.0, 1 + 1e-10) is True
    assert hea.near(1.0, 1 + 1e-6) is False


def test_near_vector_with_custom_tol():
    out = hea.near([1, 2, 3], [1, 2.001, 3], tol=0.01)
    assert out.to_list() == [True, True, True]


def test_near_expr_inside_mutate():
    df = DataFrame({"a": [1.0, 1.00000001, 2.0], "b": [1.0, 1.0, 2.0]})
    out = df.mutate(close=hea.near(pl.col("a"), pl.col("b")))
    assert out["close"].to_list() == [True, True, True]


# ---- cummean / cumall / cumany (dplyr) -----------------------------------


def test_cummean_eager():
    """``cummean([1..5])`` -> [1, 1.5, 2, 2.5, 3] (R)."""
    out = hea.cummean([1, 2, 3, 4, 5])
    assert out.to_list() == [1.0, 1.5, 2.0, 2.5, 3.0]


def test_cummean_propagates_null():
    """NA propagates from the first missing value (matches cumsum/seq_along)."""
    out = hea.cummean([1, None, 3])
    assert out.to_list() == [1.0, None, None]


def test_cummean_expr_inside_mutate():
    df = DataFrame({"x": [1, 2, 3, 4, 5]})
    out = df.mutate(cm=hea.cummean(pl.col("x")))
    assert out["cm"].to_list() == [1.0, 1.5, 2.0, 2.5, 3.0]


def test_cumall_absorbs_false_after_seen():
    """``cumall([T,T,F,T])`` -> [T,T,F,F] (R)."""
    out = hea.cumall([True, True, False, True])
    assert out.to_list() == [True, True, False, False]


def test_cumall_na_propagates_until_false_seen():
    """``cumall([T,NA,T])`` -> [T,NA,NA]; ``cumall([F,NA])`` -> [F,F]."""
    assert hea.cumall([True, None, True]).to_list() == [True, None, None]
    assert hea.cumall([False, None]).to_list() == [False, False]


def test_cumall_expr_inside_mutate():
    df = DataFrame({"b": [True, True, False, None, True]})
    out = df.mutate(c=hea.cumall(pl.col("b")))
    assert out["c"].to_list() == [True, True, False, False, False]


def test_cumany_absorbs_true_after_seen():
    """``cumany([F,F,T,F])`` -> [F,F,T,T] (R)."""
    out = hea.cumany([False, False, True, False])
    assert out.to_list() == [False, False, True, True]


def test_cumany_na_propagates_until_true_seen():
    """``cumany([F,NA,F])`` -> [F,NA,NA]; ``cumany([T,NA])`` -> [T,T]."""
    assert hea.cumany([False, None, False]).to_list() == [False, None, None]
    assert hea.cumany([True, None]).to_list() == [True, True]


def test_cumany_expr_inside_mutate():
    df = DataFrame({"b": [False, False, True, None, False]})
    out = df.mutate(c=hea.cumany(pl.col("b")))
    assert out["c"].to_list() == [False, False, True, True, True]


# ---- consecutive_id (dplyr) ----------------------------------------------


def test_consecutive_id_single_input():
    """``consecutive_id([1,1,2,2,2,1,1])`` -> [1,1,2,2,2,3,3] (R)."""
    out = hea.consecutive_id([1, 1, 2, 2, 2, 1, 1])
    assert out.to_list() == [1, 1, 2, 2, 2, 3, 3]


def test_consecutive_id_multi_input():
    """Increments when *any* input changes."""
    df = DataFrame({"a": ["a", "a", "b", "a"], "b": [1, 1, 1, 1]})
    out = df.mutate(g=hea.consecutive_id("a", "b"))
    assert out["g"].to_list() == [1, 1, 2, 3]


def test_consecutive_id_expr_inside_mutate():
    df = DataFrame({"x": [1, 1, 2, 2, 2, 1, 1]})
    out = df.mutate(g=hea.consecutive_id(pl.col("x")))
    assert out["g"].to_list() == [1, 1, 2, 2, 2, 3, 3]


def test_consecutive_id_empty_args_raises():
    with pytest.raises(TypeError, match="at least one"):
        hea.consecutive_id()


# ---- first / last / nth (dplyr) ------------------------------------------
#
# R reference (dplyr):
#   x <- c(2, 5, NA, 11, 19, 35)
#   first(x)                  -> 2
#   first(integer(0), default=-1L) -> -1
#   first(c(NA, 5), na_rm=TRUE)    -> 5
#   last(x)                   -> 35
#   nth(x, 2)                 -> 5
#   nth(x, -1)                -> 35
#   nth(x, -2)                -> 19
#   nth(x, 100, default=-1)   -> -1   (OOB)
#   nth(x, 3, na_rm=TRUE)     -> 11   (skip NA before counting)
#   nth(x, 0, default=-1)     -> -1   (n=0 is OOB, default applies)
#   nth(c(1,NA,3), 2)         -> NA   (null at index returned as-is)
#   first(c("a","b","c"), order_by=c(3,1,2)) -> "b"

_NTH_X = [2, 5, None, 11, 19, 35]


def test_first_eager_returns_scalar():
    """dplyr returns a length-1 vector; in Python we return a scalar."""
    assert hea.first(_NTH_X) == 2
    assert hea.first(pl.Series(_NTH_X)) == 2
    assert hea.first([10, 20, 30]) == 10


def test_first_default_fires_on_empty_input():
    assert hea.first([], default=-1) == -1
    assert hea.first([]) is None


def test_first_na_rm_skips_leading_nulls():
    assert hea.first([None, 5, 10], na_rm=True) == 5
    # All-null + na_rm → empty arr → default
    assert hea.first([None, None], na_rm=True, default=-1) == -1


def test_last_eager():
    assert hea.last(_NTH_X) == 35
    assert hea.last([5, None], na_rm=True) == 5


def test_nth_eager_positive_and_negative():
    """Negative ``n`` counts from the end (-1 = last, -2 = second-to-last)."""
    assert hea.nth(_NTH_X, 2) == 5
    assert hea.nth(_NTH_X, -1) == 35
    assert hea.nth(_NTH_X, -2) == 19


def test_nth_default_fires_on_oob():
    """``default`` returns when index is out of range."""
    assert hea.nth(_NTH_X, 100, default=-1) == -1
    assert hea.nth(_NTH_X, -100, default=-1) == -1


def test_nth_n_zero_is_oob():
    """``nth(x, 0)`` is degenerate in dplyr: returns ``default``."""
    assert hea.nth(_NTH_X, 0) is None
    assert hea.nth(_NTH_X, 0, default=-1) == -1


def test_nth_null_at_index_returned_as_is():
    """A ``None`` *value* at index ``n`` is returned, not replaced by default
    (matches dplyr — default only fires on OOB)."""
    assert hea.nth([1, None, 3], 2) is None
    assert hea.nth([1, None, 3], 2, default=-1) is None


def test_nth_na_rm_skips_nulls_before_counting():
    """With ``na_rm=True``, null entries don't consume an index slot."""
    assert hea.nth(_NTH_X, 3, na_rm=True) == 11  # x[1,2,4] = 2,5,11


def test_first_order_by_reorders_then_picks():
    """``first(x, order_by=t)`` returns the value of x when sorted by t."""
    # R: first(c("a","b","c"), order_by=c(3,1,2)) -> "b"
    assert hea.first(["a", "b", "c"], order_by=[3, 1, 2]) == "b"
    assert hea.last(["a", "b", "c"], order_by=[3, 1, 2]) == "a"
    assert hea.nth(["a", "b", "c"], 1, order_by=[3, 1, 2]) == "b"


def test_first_expr_inside_mutate_broadcasts():
    """In an Expr context, ``first(col)`` is a scalar that broadcasts."""
    df = DataFrame({"x": [10, 20, 30, 40]})
    out = df.mutate(
        fst=hea.first(pl.col("x")),
        lst=hea.last(pl.col("x")),
        sec=hea.nth(pl.col("x"), 2),
    )
    assert out["fst"].to_list() == [10, 10, 10, 10]
    assert out["lst"].to_list() == [40, 40, 40, 40]
    assert out["sec"].to_list() == [20, 20, 20, 20]


def test_nth_expr_oob_uses_default():
    """The OOB-default path in Expr context (when/otherwise machinery)."""
    df = DataFrame({"x": [10, 20, 30, 40]})
    out = df.mutate(oob=hea.nth(pl.col("x"), 10, default=-1))
    assert out["oob"].to_list() == [-1, -1, -1, -1]


def test_lag_with_first_default_canonical_pattern():
    """The motivating use case from R:

        mutate(diff = time - lag(time, default = first(time)))

    First row gets diff=0 (lag uses first(time)); subsequent rows are
    real diffs. Verified against R's output.
    """
    events = DataFrame({"time": [1, 3, 8, 15, 20]})
    out = events.mutate(
        diff=pl.col("time") - hea.lag(
            pl.col("time"), default=hea.first(pl.col("time"))
        ),
        has_gap=pl.col("diff") >= 5,
    )
    assert out["diff"].to_list() == [0, 2, 5, 7, 5]
    assert out["has_gap"].to_list() == [False, False, True, True, True]


def test_first_last_nth_shadow_polars_top_level():
    """``hea.first`` / ``hea.last`` / ``hea.nth`` deliberately replace polars'
    top-level versions (which are *column* selectors, not element pickers).
    Polars' versions remain accessible as ``pl.first`` / ``pl.last`` / ``pl.nth``.
    """
    assert hea.first is not pl.first
    assert hea.last is not pl.last
    assert hea.nth is not pl.nth
