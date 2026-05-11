"""Phase 5 regression test: hea is a strict superset of polars.

Two layers:

1. **Coverage check** — every name in ``polars.__all__`` is reachable via
   ``hea.*``. New polars releases that add a public name will fail this
   test until categorized (either re-exported as-is or wrapped).
2. **Targeted tests** — wrapped factories (constructors / I/O readers /
   scanners) return ``hea.DataFrame`` / ``hea.LazyFrame``; expression
   builders pass through unchanged.

Pinned polars version (``pyproject.toml``); bump cadence runs this test.
"""

from __future__ import annotations

import os
import tempfile

import polars as pl
import pytest

import hea


# ---------------------------------------------------------------------------
# 1. Coverage — strict superset
# ---------------------------------------------------------------------------


def test_hea_dir_is_superset_of_polars_all():
    """``dir(hea) ⊇ pl.__all__`` — the foundation of the 100%-superset goal."""
    hea_public = {n for n in dir(hea) if not n.startswith("_")}
    pl_all = set(pl.__all__)
    missing = pl_all - hea_public
    assert not missing, (
        f"polars added public names that hea doesn't re-export: {sorted(missing)}. "
        "Star import in hea/__init__.py should cover them automatically — "
        "check whether polars's __all__ has changed or whether the import order "
        "is dropping these names."
    )


def test_hea_overrides_match_actual_shadows():
    """``_HEA_OVERRIDES`` must equal the set of polars names hea re-binds.

    Already enforced at import time (raises in ``hea/__init__.py``); this
    test gives a clear failure message if a developer changes the override
    list and forgets to update _HEA_OVERRIDES.
    """
    pl_all = set(pl.__all__)
    actual_shadows = {
        n for n in pl_all
        if hasattr(hea, n) and getattr(hea, n) is not getattr(pl, n)
    }
    assert actual_shadows == set(hea._HEA_OVERRIDES), (
        f"hea shadows {sorted(actual_shadows)} but _HEA_OVERRIDES "
        f"declares {sorted(hea._HEA_OVERRIDES)}"
    )


# ---------------------------------------------------------------------------
# 2. Wrapped factories return hea subclasses
# ---------------------------------------------------------------------------


def test_constructors_return_hea_dataframe():
    cases = [
        ("from_dict",     lambda: hea.from_dict({"x": [1, 2, 3]})),
        ("from_dicts",    lambda: hea.from_dicts([{"x": 1}, {"x": 2}])),
        ("from_records",  lambda: hea.from_records([(1, 2), (3, 4)], schema=["a", "b"])),
        ("concat",        lambda: hea.concat([
                              hea.from_dict({"x": [1]}),
                              hea.from_dict({"x": [2]}),
                          ])),
    ]
    for name, fn in cases:
        result = fn()
        assert isinstance(result, hea.DataFrame), (
            f"hea.{name}(...) returned {type(result).__module__}.{type(result).__name__}"
        )


def test_concat_polymorphic_lazyframe():
    """``concat`` of LazyFrames should return ``hea.LazyFrame``."""
    lf = hea.from_dict({"x": [1, 2]}).lazy()
    out = hea.concat([lf, lf])
    assert isinstance(out, hea.LazyFrame)


def test_read_csv_returns_hea_dataframe(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    df = hea.read_csv(str(p))
    assert isinstance(df, hea.DataFrame)


def test_scan_csv_returns_hea_lazyframe(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    lf = hea.scan_csv(str(p))
    assert isinstance(lf, hea.LazyFrame)
    df = lf.collect()
    assert isinstance(df, hea.DataFrame)


def test_read_parquet_returns_hea_dataframe(tmp_path):
    p = str(tmp_path / "x.parquet")
    hea.from_dict({"x": [1, 2, 3]}).write_parquet(p)
    df = hea.read_parquet(p)
    assert isinstance(df, hea.DataFrame)


def test_scan_parquet_returns_hea_lazyframe(tmp_path):
    p = str(tmp_path / "x.parquet")
    hea.from_dict({"x": [1, 2, 3]}).write_parquet(p)
    lf = hea.scan_parquet(p)
    assert isinstance(lf, hea.LazyFrame)


def test_merge_sorted_polymorphic():
    """``merge_sorted`` is polymorphic — DataFrames in, DataFrame out."""
    df = hea.from_dict({"x": [1, 3]}).sort("x")
    out = hea.merge_sorted([df, df], key="x")
    assert isinstance(out, hea.DataFrame)


# ---------------------------------------------------------------------------
# 3. Expression builders, dtypes, sub-namespaces pass through unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "col", "lit", "when", "sum", "mean", "count", "len", "min", "max",
    "first", "last", "nth", "concat_str", "concat_list", "all", "any",
    "fold", "format", "select", "struct", "duration",
    "date_range", "datetime_range",
])
def test_expr_builder_is_polars_function(name):
    # ``exclude`` is intentionally shadowed (see _HEA_OVERRIDES); covered
    # separately below.
    assert getattr(hea, name) is getattr(pl, name)


def test_hea_exclude_extends_polars_exclude():
    """``hea.exclude`` accepts everything ``pl.exclude`` does plus
    DataFrame/Series/list. Names-and-strings path must produce the same
    Expr as ``pl.exclude`` for a representative input."""
    df = hea.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert df.select(hea.exclude("a")).columns == df.select(pl.exclude("a")).columns
    # Slice-of-DataFrame form (the new affordance).
    assert df.select(hea.exclude(df["a":"b"])).columns == ["c"]


def test_dplyr_named_aliases_route_to_polars():
    """``hea.n`` / ``hea.n_distinct`` are dplyr-named aliases for the
    polars row-count and unique-count expressions."""
    assert hea.n is pl.len
    assert hea.n_distinct is pl.n_unique
    # End-to-end inside a summarize pipeline (the use case that motivates
    # exposing these names at all).
    df = hea.DataFrame({
        "dest": ["A", "A", "B", "B", "B"],
        "carrier": ["UA", "DL", "UA", "UA", "AA"],
    })
    out = (
        df.group_by("dest")
        .summarize(rows=hea.n(), carriers=hea.n_distinct("carrier"))
        .arrange("dest")
    )
    assert out["rows"].to_list() == [2, 3]
    assert out["carriers"].to_list() == [2, 2]


@pytest.mark.parametrize("name", [
    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64", "Boolean", "String", "Utf8", "Date", "Datetime",
    "Duration", "Time", "Categorical", "Enum", "List", "Array", "Struct",
    "Object", "Null", "Decimal", "DataType",
])
def test_dtype_is_polars_dtype(name):
    assert getattr(hea, name) is getattr(pl, name)


def test_sub_namespaces_re_exported():
    assert hea.selectors is pl.selectors
    assert hea.api is pl.api
    assert hea.exceptions is pl.exceptions
    assert hea.plugins is pl.plugins


# ---------------------------------------------------------------------------
# 4. Class overrides — hea.DataFrame / hea.LazyFrame / hea.Series are subclasses
# ---------------------------------------------------------------------------


def test_dataframe_is_hea_subclass():
    assert hea.DataFrame is hea.dataframe.DataFrame
    assert hea.DataFrame is not pl.DataFrame
    assert issubclass(hea.DataFrame, pl.DataFrame)


def test_lazyframe_is_hea_subclass():
    assert hea.LazyFrame is hea.dataframe.LazyFrame
    assert hea.LazyFrame is not pl.LazyFrame
    assert issubclass(hea.LazyFrame, pl.LazyFrame)


def test_series_is_hea_subclass():
    assert hea.Series is hea.dataframe.Series
    assert hea.Series is not pl.Series
    assert issubclass(hea.Series, pl.Series)


def test_end_to_end_chain_stays_in_hea():
    """Realistic chain: read → filter → groupby → collect — must stay hea."""
    df = hea.from_dict({"g": ["a", "b", "a", "b"], "x": [1, 2, 3, 4]})
    out = (
        df
        .lazy()
        .filter(hea.col("x") > 1)
        .with_columns(z=hea.col("x") * 2)
        .group_by("g")
        .agg(hea.col("z").sum())
        .collect()
    )
    assert isinstance(out, hea.DataFrame)
