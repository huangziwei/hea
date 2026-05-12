"""Tests for ``hea.dataframe`` two-table verbs (r4ds chapter 19).

Mirrors the joins chapter in *R for Data Science* (2e): mutating joins,
filtering joins, cross joins, and non-equi / rolling / overlap joins.
"""

from __future__ import annotations

import datetime as _dt

import polars as pl
import pytest

from hea import (
    DataFrame,
    between,
    closest,
    col,
    join_by,
    overlaps,
    within,
)


# ---------------------------------------------------------------------------
# Fixtures — small frames that mirror the chapter's shapes (carrier/airline,
# tailnum/plane, day/weather, dest/airport, plus the parties/employees pair
# used for non-equi / rolling / overlap joins).
# ---------------------------------------------------------------------------


@pytest.fixture
def flights2():
    """Minimal stand-in for ``flights2`` from r4ds ch19."""
    return DataFrame(
        {
            "year": [2013, 2013, 2013, 2013, 2013],
            "origin": ["EWR", "LGA", "JFK", "JFK", "LGA"],
            "dest": ["IAH", "IAH", "MIA", "BQN", "ATL"],
            "tailnum": ["N14228", "N24211", "N619AA", "N804JB", "N668DN"],
            "carrier": ["UA", "UA", "AA", "B6", "DL"],
        }
    )


@pytest.fixture
def airlines():
    return DataFrame(
        {
            "carrier": ["UA", "AA", "B6", "DL", "WN"],
            "name": [
                "United Air Lines Inc.",
                "American Airlines Inc.",
                "JetBlue Airways",
                "Delta Air Lines Inc.",
                "Southwest Airlines Co.",
            ],
        }
    )


@pytest.fixture
def airports():
    return DataFrame(
        {
            "faa": ["EWR", "JFK", "LGA", "IAH", "ATL"],
            "name_airport": [
                "Newark Liberty Intl",
                "John F Kennedy Intl",
                "La Guardia",
                "George Bush Intl",
                "Hartsfield Jackson",
            ],
        }
    )


@pytest.fixture
def planes():
    """Has a ``year`` column that collides with flights2."""
    return DataFrame(
        {
            "tailnum": ["N14228", "N24211", "N619AA", "N804JB", "N668DN"],
            "year": [1999, 1998, 1990, 2012, 1991],
            "type": ["Fixed wing"] * 5,
        }
    )


@pytest.fixture
def parties():
    """4 quarterly parties with dates + the surrounding date window."""
    return DataFrame(
        {
            "q": [1, 2, 3, 4],
            "party": [
                _dt.date(2022, 1, 10),
                _dt.date(2022, 4, 4),
                _dt.date(2022, 7, 11),
                _dt.date(2022, 10, 3),
            ],
            "start": [
                _dt.date(2022, 1, 1),
                _dt.date(2022, 4, 4),
                _dt.date(2022, 7, 11),
                _dt.date(2022, 10, 3),
            ],
            "end": [
                _dt.date(2022, 4, 3),
                _dt.date(2022, 7, 10),
                _dt.date(2022, 10, 2),
                _dt.date(2022, 12, 31),
            ],
        }
    )


@pytest.fixture
def employees():
    return DataFrame(
        {
            "name": ["Alice", "Bob", "Carl", "Dora"],
            "birthday": [
                _dt.date(2022, 2, 15),
                _dt.date(2022, 4, 3),
                _dt.date(2022, 8, 1),
                _dt.date(2022, 11, 5),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Natural joins — by= defaults to shared column names (dplyr behaviour).
# ---------------------------------------------------------------------------


def test_left_join_natural_uses_shared_columns(flights2, airlines):
    """``left_join(other)`` joins on every shared column name (here: ``carrier``)."""
    out = flights2.left_join(airlines)
    assert isinstance(out, DataFrame)
    assert "name" in out.columns
    # all 5 left rows preserved + airline name attached
    assert out.height == flights2.height
    assert out["name"].to_list() == [
        "United Air Lines Inc.",
        "United Air Lines Inc.",
        "American Airlines Inc.",
        "JetBlue Airways",
        "Delta Air Lines Inc.",
    ]


def test_natural_join_with_no_shared_cols_raises(flights2):
    other = DataFrame({"zzz": [1, 2, 3]})
    with pytest.raises(ValueError, match="No shared column names"):
        flights2.left_join(other)


def test_natural_join_emits_dplyr_info_message(flights2, airlines, capsys):
    """Natural join prints ``Joining with `by = join_by(...)``` to stderr,
    mirroring dplyr's diagnostic — so over-joining on extra shared
    columns (e.g. ``year`` in flights/planes) is visible."""
    flights2.left_join(airlines)
    msg = capsys.readouterr().err
    assert "Joining with `by = join_by(" in msg
    assert "'carrier'" in msg


def test_explicit_by_does_not_emit_message(flights2, airlines, capsys):
    """When the user passes ``by=`` (or ``join_by(...)``) explicitly, no
    info message — they already know what they're joining on."""
    flights2.left_join(airlines, "carrier")
    assert capsys.readouterr().err == ""


def test_natural_join_auto_casts_numeric_keys(capsys):
    """i64 + f64 on the same key name → both cast to f64 (matches
    dplyr's implicit numeric coercion; polars alone would SchemaError)."""
    left = DataFrame({"k": pl.Series([1, 2, 3], dtype=pl.Int64), "x": ["a", "b", "c"]})
    right = DataFrame({"k": pl.Series([1.0, 2.0, 4.0], dtype=pl.Float64), "y": ["p", "q", "r"]})
    out = left.left_join(right)
    assert out["x"].to_list() == ["a", "b", "c"]
    assert out["y"].to_list() == ["p", "q", None]
    assert out.schema["k"] == pl.Float64


def test_explicit_by_also_auto_casts_numeric_keys():
    """Auto-cast applies to explicit ``by=`` too — the user named the
    keys, they expect the join to work even if dtypes differ."""
    left = DataFrame({"k": pl.Series([1, 2, 3], dtype=pl.Int32)})
    right = DataFrame({"k": pl.Series([1, 2], dtype=pl.Int64), "y": ["p", "q"]})
    out = left.left_join(right, "k")
    assert out["y"].to_list() == ["p", "q", None]


def test_non_numeric_mismatch_still_errors():
    """Non-numeric type mismatches (e.g. string vs int) aren't coerced
    — let polars surface the SchemaError."""
    left = DataFrame({"k": ["1", "2", "3"]})
    right = DataFrame({"k": [1, 2], "y": ["p", "q"]})
    with pytest.raises(Exception, match="(?i)schema|type|cast"):
        left.left_join(right, "k")


# ---------------------------------------------------------------------------
# Explicit by= forms: str, list[str], dict, join_by().
# ---------------------------------------------------------------------------


def test_left_join_by_string(flights2, planes):
    """``by="tailnum"`` — explicit single-column equi join."""
    out = flights2.left_join(planes, "tailnum")
    assert "tailnum" in out.columns
    # ``year`` collides → dplyr suffix .x/.y on both sides
    assert "year.x" in out.columns
    assert "year.y" in out.columns
    assert "year" not in out.columns


def test_left_join_by_list(flights2):
    """``by=[...]`` — explicit multi-column equi join."""
    other = DataFrame(
        {
            "origin": ["EWR", "JFK"],
            "dest": ["IAH", "MIA"],
            "miles": [1400, 1090],
        }
    )
    out = flights2.left_join(other, ["origin", "dest"])
    assert out["miles"].to_list() == [1400, None, 1090, None, None]


def test_left_join_by_dict_renames(flights2, airports):
    """``by={left: right}`` — dict form maps unequal column names."""
    out = flights2.left_join(airports, {"dest": "faa"})
    assert "name_airport" in out.columns
    # right key dropped under keep=False default
    assert "faa" not in out.columns


def test_left_join_join_by_string_arg(flights2, planes):
    """``join_by("tailnum")`` — same as by= but routed via join_by()."""
    out = flights2.left_join(planes, join_by("tailnum"))
    assert "year.x" in out.columns and "year.y" in out.columns


def test_left_join_join_by_eq_rename(flights2, airports):
    """``join_by(col("dest") == col("faa"))`` — equality with rename."""
    out = flights2.left_join(airports, join_by(col("dest") == col("faa")))
    assert "name_airport" in out.columns
    assert "faa" not in out.columns
    assert out.height == flights2.height


# ---------------------------------------------------------------------------
# keep= toggle. dplyr default is False (right-side key dropped); True keeps it.
# ---------------------------------------------------------------------------


def test_keep_false_drops_right_key(flights2, airports):
    out = flights2.left_join(airports, {"dest": "faa"}, keep=False)
    assert "faa" not in out.columns
    assert "dest" in out.columns


def test_keep_true_keeps_both_keys(flights2, airports):
    out = flights2.left_join(airports, {"dest": "faa"}, keep=True)
    assert "faa" in out.columns
    assert "dest" in out.columns


# ---------------------------------------------------------------------------
# Suffix on non-key collisions defaults to dplyr's (".x", ".y").
# ---------------------------------------------------------------------------


def test_suffix_default_is_dplyr_two_sided(flights2, planes):
    out = flights2.left_join(planes, "tailnum")
    assert "year.x" in out.columns
    assert "year.y" in out.columns


def test_suffix_custom(flights2, planes):
    out = flights2.left_join(planes, "tailnum", suffix=("_l", "_r"))
    assert "year_l" in out.columns
    assert "year_r" in out.columns


# ---------------------------------------------------------------------------
# inner / right / full mutating joins.
# ---------------------------------------------------------------------------


def test_inner_join_drops_unmatched():
    a = DataFrame({"k": [1, 2, 3], "x": ["a", "b", "c"]})
    b = DataFrame({"k": [1, 2, 4], "y": ["p", "q", "r"]})
    out = a.inner_join(b, "k")
    assert out["k"].to_list() == [1, 2]


def test_right_join_keeps_right_unmatched():
    a = DataFrame({"k": [1, 2, 3], "x": ["a", "b", "c"]})
    b = DataFrame({"k": [1, 2, 4], "y": ["p", "q", "r"]})
    out = a.right_join(b, "k")
    assert sorted(out["k"].to_list()) == [1, 2, 4]
    assert out.height == 3


def test_full_join_keeps_all_unmatched():
    a = DataFrame({"k": [1, 2, 3], "x": ["a", "b", "c"]})
    b = DataFrame({"k": [1, 2, 4], "y": ["p", "q", "r"]})
    out = a.full_join(b, "k")
    assert sorted(out["k"].to_list()) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Filtering joins: semi_join keeps left rows with a match; anti_join keeps
# rows with NO match.
# ---------------------------------------------------------------------------


def test_semi_join_keeps_left_columns_only(airports, flights2):
    """``semi_join`` filters left to rows matching right, no right cols added."""
    out = airports.semi_join(flights2, join_by(col("faa") == col("origin")))
    assert set(out.columns) == set(airports.columns)
    assert sorted(out["faa"].to_list()) == ["EWR", "JFK", "LGA"]


def test_anti_join_returns_left_with_no_match(flights2, airports):
    out = flights2.anti_join(airports, join_by(col("dest") == col("faa")))
    # MIA and BQN are in flights2.dest but not in airports.faa
    assert set(out["dest"].to_list()) == {"MIA", "BQN"}


# ---------------------------------------------------------------------------
# Cross join — Cartesian product. Both sides' columns get dplyr suffix if
# they collide (here ``name`` is on both sides).
# ---------------------------------------------------------------------------


def test_cross_join_yields_cartesian_product():
    df = DataFrame({"name": ["John", "Simon", "Tracy", "Max"]})
    out = df.cross_join(df)
    assert out.height == 16
    assert "name.x" in out.columns and "name.y" in out.columns


# ---------------------------------------------------------------------------
# Non-equi self-join via ``join_by(col("id") < col("id"))``.
# ---------------------------------------------------------------------------


def test_non_equi_self_join_inequality():
    df = DataFrame({"id": [1, 2, 3, 4], "name": ["John", "Simon", "Tracy", "Max"]})
    out = df.inner_join(df, join_by(col("id") < col("id")))
    # 4 choose 2 = 6 ordered pairs (i, j) with i < j
    assert out.height == 6
    assert "id.x" in out.columns and "id.y" in out.columns
    pairs = list(zip(out["id.x"].to_list(), out["id.y"].to_list()))
    assert all(a < b for a, b in pairs)


def test_non_equi_left_join_not_implemented():
    df = DataFrame({"id": [1, 2, 3]})
    with pytest.raises(NotImplementedError, match="non-equi"):
        df.left_join(df, join_by(col("id") < col("id")))


# ---------------------------------------------------------------------------
# Rolling join via ``closest()``. dplyr ch19's parties/birthday example.
# ---------------------------------------------------------------------------


def test_closest_backward_rolling_join(employees, parties):
    """``closest(birthday >= party)`` → largest party date ≤ birthday."""
    out = employees.left_join(
        parties.select("q", "party"),
        join_by(closest(col("birthday") >= col("party"))),
    )
    # Each employee should get the most recent past quarterly party.
    result = {name: q for name, q in zip(out["name"].to_list(), out["q"].to_list())}
    assert result == {"Alice": 1, "Bob": 1, "Carl": 3, "Dora": 4}


def test_closest_forward_rolling_join():
    """``closest(x <= y)`` → smallest right value ≥ left."""
    left = DataFrame({"t": [1, 3, 5, 7]})
    right = DataFrame({"u": [2, 4, 8], "label": ["a", "b", "c"]})
    out = left.left_join(right, join_by(closest(col("t") <= col("u"))))
    assert out["u"].to_list() == [2, 4, 8, 8]
    assert out["label"].to_list() == ["a", "b", "c", "c"]


def test_closest_with_equi_grouping():
    """closest() + equi key partitions the asof per group."""
    left = DataFrame({"g": ["a", "a", "b"], "t": [1, 4, 1]})
    right = DataFrame(
        {"g": ["a", "a", "b"], "u": [0, 3, 2], "lab": ["x", "y", "z"]}
    )
    out = left.left_join(right, join_by("g", closest(col("t") >= col("u"))))
    assert out["lab"].to_list() == ["x", "y", None]


def test_closest_rejects_non_inequality():
    with pytest.raises(ValueError, match="binary inequality"):
        closest(pl.lit(True))


# ---------------------------------------------------------------------------
# Overlap join via ``between()``. dplyr ch19's parties/birthday range example.
# ---------------------------------------------------------------------------


def test_between_overlap_join(employees, parties):
    out = employees.inner_join(
        parties,
        join_by(between(col("birthday"), col("start"), col("end"))),
    )
    # Each birthday falls in exactly one quarter window → 4 rows.
    assert out.height == 4
    result = {name: q for name, q in zip(out["name"].to_list(), out["q"].to_list())}
    assert result == {"Alice": 1, "Bob": 1, "Carl": 3, "Dora": 4}


def test_overlaps_helper_finds_overlapping_intervals():
    left = DataFrame(
        {
            "a": ["A", "B"],
            "a_lo": [_dt.date(2022, 1, 1), _dt.date(2022, 5, 1)],
            "a_hi": [_dt.date(2022, 3, 15), _dt.date(2022, 8, 15)],
        }
    )
    right = DataFrame(
        {
            "b": ["x", "y"],
            "b_lo": [_dt.date(2022, 3, 1), _dt.date(2022, 7, 1)],
            "b_hi": [_dt.date(2022, 4, 30), _dt.date(2022, 12, 31)],
        }
    )
    out = left.inner_join(
        right, join_by(overlaps("a_lo", "a_hi", "b_lo", "b_hi"))
    )
    assert sorted(zip(out["a"].to_list(), out["b"].to_list())) == [
        ("A", "x"),
        ("B", "y"),
    ]


def test_within_helper_finds_contained_intervals():
    left = DataFrame(
        {"a": ["A", "B"], "a_lo": [10, 20], "a_hi": [12, 50]}
    )
    right = DataFrame(
        {"b": ["x", "y"], "b_lo": [0, 0], "b_hi": [15, 100]}
    )
    # A's [10, 12] is within both [0, 15] and [0, 100]; B's [20, 50] is
    # within [0, 100] but NOT [0, 15].
    out = left.inner_join(
        right, join_by(within("a_lo", "a_hi", "b_lo", "b_hi"))
    )
    assert sorted(zip(out["a"].to_list(), out["b"].to_list())) == [
        ("A", "x"),
        ("A", "y"),
        ("B", "y"),
    ]


# ---------------------------------------------------------------------------
# Sanity: result type is hea DataFrame, not pl.DataFrame.
# ---------------------------------------------------------------------------


def test_join_preserves_subclass(flights2, airlines):
    out = flights2.left_join(airlines)
    assert type(out) is DataFrame
    assert isinstance(out, pl.DataFrame)


# ---------------------------------------------------------------------------
# Argument validation.
# ---------------------------------------------------------------------------


def test_join_by_rejects_unsupported_arg():
    with pytest.raises(TypeError, match="unsupported arg type"):
        join_by(123)


def test_join_by_two_closests_disallowed():
    with pytest.raises(ValueError, match="only one closest"):
        join_by(
            closest(col("x") >= col("y")),
            closest(col("a") >= col("b")),
        )


def test_na_matches_invalid_value(flights2, airlines):
    with pytest.raises(ValueError, match="na_matches"):
        flights2.left_join(airlines, na_matches="bogus")


def test_suffix_must_be_two_tuple(flights2, airlines):
    with pytest.raises(TypeError, match="suffix"):
        flights2.left_join(airlines, suffix=".x")  # type: ignore[arg-type]
