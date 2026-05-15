"""Namespace contract: polars passthrough lives in sub-modules.

The top-level ``hea`` namespace is intentionally empty of polars names
— no ``hea.col``, ``hea.Int64``, ``hea.read_csv``. The polars passthrough
splits across three hea sub-modules:

* ``hea.tidy``   — expression builders (``col``, ``lit``, ``when``,
                   ``coalesce``, the horizontal reducers) plus the
                   ``DataFrame`` / ``LazyFrame`` / ``Series`` subclasses
* ``hea.dtypes`` — every polars datatype (``Int64``, ``String``, …)
* ``hea.io``     — readers / scanners / DataFrame factories
                   (``read_csv``, ``concat``, ``from_dict``, …) — wrapped
                   so the result is the hea subclass

Polars's own sub-namespaces (``api``, ``exceptions``, ``plugins``,
``selectors``) ARE re-exported at the top level as a stable convention.
"""

from __future__ import annotations

import polars as pl
import pytest

import hea


# ---------------------------------------------------------------------------
# 1. Top-level shape — sub-modules only, no polars name leaks
# ---------------------------------------------------------------------------


def test_top_level_carries_no_polars_names():
    """``hea.col``, ``hea.Int64``, ``hea.read_csv`` must all fail.
    Anything in ``pl.__all__`` is inaccessible at the top-level — except
    the three data type classes (which appear at top-level *as hea
    subclasses*, never the raw polars versions) and the four polars
    sub-namespaces."""
    type_allowlist = {"DataFrame", "LazyFrame", "Series"}
    sub_namespace_allowlist = {"api", "exceptions", "plugins", "selectors"}
    for name in pl.__all__:
        if name in sub_namespace_allowlist | type_allowlist:
            continue
        assert not hasattr(hea, name), (
            f"hea.{name} leaked at top level — polars names should live "
            "in hea.tidy / hea.dtypes / hea.io instead"
        )


def test_top_level_data_classes_are_hea_subclasses():
    """``hea.DataFrame`` / ``hea.LazyFrame`` / ``hea.Series`` are the hea
    subclasses, not raw polars. Users get the tidyverse verbs for free."""
    assert hea.DataFrame is hea.tidy.DataFrame
    assert hea.LazyFrame is hea.tidy.LazyFrame
    assert hea.Series is hea.tidy.Series
    assert hea.DataFrame is not pl.DataFrame
    assert hea.LazyFrame is not pl.LazyFrame
    assert hea.Series is not pl.Series
    assert issubclass(hea.DataFrame, pl.DataFrame)
    assert issubclass(hea.LazyFrame, pl.LazyFrame)
    assert issubclass(hea.Series, pl.Series)


def test_polars_sub_namespaces_are_re_exported():
    assert hea.selectors is pl.selectors
    assert hea.api is pl.api
    assert hea.exceptions is pl.exceptions
    assert hea.plugins is pl.plugins


# ---------------------------------------------------------------------------
# 2. hea.tidy — expression builders + DataFrame subclasses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "col", "lit", "when", "coalesce",
    "concat_str", "concat_list",
    "all_horizontal", "any_horizontal",
    "min_horizontal", "max_horizontal", "mean_horizontal", "sum_horizontal",
    "Expr", "Schema",
])
def test_tidy_re_exports_polars_expr(name):
    """``hea.tidy.col`` etc. are the polars originals — no wrapping."""
    assert getattr(hea.tidy, name) is getattr(pl, name)


def test_tidy_dataframe_is_hea_subclass():
    assert hea.tidy.DataFrame is not pl.DataFrame
    assert issubclass(hea.tidy.DataFrame, pl.DataFrame)


def test_tidy_lazyframe_is_hea_subclass():
    assert hea.tidy.LazyFrame is not pl.LazyFrame
    assert issubclass(hea.tidy.LazyFrame, pl.LazyFrame)


def test_tidy_series_is_hea_subclass():
    assert hea.tidy.Series is not pl.Series
    assert issubclass(hea.tidy.Series, pl.Series)


def test_tidy_exclude_extends_polars_exclude():
    """``hea.tidy.exclude`` accepts everything ``pl.exclude`` does plus
    DataFrame/Series/list."""
    df = hea.tidy.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert df.select(hea.tidy.exclude("a")).columns == df.select(pl.exclude("a")).columns
    # Slice-of-DataFrame form (the new affordance).
    assert df.select(hea.tidy.exclude(df["a":"b"])).columns == ["c"]


def test_tidy_n_and_n_distinct_alias_polars():
    """``hea.tidy.n`` / ``hea.tidy.n_distinct`` are dplyr-named aliases."""
    assert hea.tidy.n is pl.len
    assert hea.tidy.n_distinct is pl.n_unique
    df = hea.tidy.DataFrame({
        "dest": ["A", "A", "B", "B", "B"],
        "carrier": ["UA", "DL", "UA", "UA", "AA"],
    })
    out = (
        df.group_by("dest")
        .summarize(rows=hea.tidy.n(), carriers=hea.tidy.n_distinct("carrier"))
        .arrange("dest")
    )
    assert out["rows"].to_list() == [2, 3]
    assert out["carriers"].to_list() == [2, 2]


# ---------------------------------------------------------------------------
# 3. hea.dtypes — every polars dtype is reachable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64", "Boolean", "String", "Utf8", "Date", "Datetime",
    "Duration", "Time", "Categorical", "Enum", "List", "Array", "Struct",
    "Object", "Null", "Decimal", "DataType",
])
def test_dtype_is_polars_dtype(name):
    assert getattr(hea.dtypes, name) is getattr(pl, name)


# ---------------------------------------------------------------------------
# 4. hea.io — wrapped factories return hea subclasses
# ---------------------------------------------------------------------------


def test_constructors_return_hea_dataframe():
    cases = [
        ("from_dict",    lambda: hea.tidy.from_dict({"x": [1, 2, 3]})),
        ("from_dicts",   lambda: hea.tidy.from_dicts([{"x": 1}, {"x": 2}])),
        ("from_records", lambda: hea.tidy.from_records([(1, 2), (3, 4)], schema=["a", "b"])),
        ("concat",       lambda: hea.tidy.concat([
                             hea.tidy.from_dict({"x": [1]}),
                             hea.tidy.from_dict({"x": [2]}),
                         ])),
    ]
    for name, fn in cases:
        result = fn()
        assert isinstance(result, hea.tidy.DataFrame), (
            f"hea.io.{name}(...) returned {type(result).__module__}.{type(result).__name__}"
        )


def test_concat_polymorphic_lazyframe():
    lf = hea.tidy.from_dict({"x": [1, 2]}).lazy()
    out = hea.tidy.concat([lf, lf])
    assert isinstance(out, hea.tidy.LazyFrame)


def test_read_csv_returns_hea_dataframe(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    df = hea.io.read_csv(str(p))
    assert isinstance(df, hea.tidy.DataFrame)


def test_scan_csv_returns_hea_lazyframe(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    lf = hea.io.scan_csv(str(p))
    assert isinstance(lf, hea.tidy.LazyFrame)
    assert isinstance(lf.collect(), hea.tidy.DataFrame)


def test_read_parquet_returns_hea_dataframe(tmp_path):
    p = str(tmp_path / "x.parquet")
    hea.tidy.from_dict({"x": [1, 2, 3]}).write_parquet(p)
    df = hea.io.read_parquet(p)
    assert isinstance(df, hea.tidy.DataFrame)


def test_scan_parquet_returns_hea_lazyframe(tmp_path):
    p = str(tmp_path / "x.parquet")
    hea.tidy.from_dict({"x": [1, 2, 3]}).write_parquet(p)
    lf = hea.io.scan_parquet(p)
    assert isinstance(lf, hea.tidy.LazyFrame)


def test_merge_sorted_polymorphic():
    df = hea.tidy.from_dict({"x": [1, 3]}).sort("x")
    out = hea.tidy.merge_sorted([df, df], key="x")
    assert isinstance(out, hea.tidy.DataFrame)


def test_end_to_end_chain_stays_in_hea():
    """Realistic chain: read → filter → groupby → collect — must stay hea."""
    df = hea.tidy.from_dict({"g": ["a", "b", "a", "b"], "x": [1, 2, 3, 4]})
    out = (
        df.lazy()
        .filter(hea.tidy.col("x") > 1)
        .with_columns(z=hea.tidy.col("x") * 2)
        .group_by("g")
        .agg(hea.tidy.col("z").sum())
        .collect()
    )
    assert isinstance(out, hea.tidy.DataFrame)
