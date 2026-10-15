"""``hea`` carries ``DataFrame`` / ``LazyFrame`` / ``Series`` and four ``pl``
sub-namespaces; ``hea.tidy`` carries the expression builders and constructors,
``hea.dtypes`` the datatypes, ``hea.io`` the readers and scanners."""

from __future__ import annotations

import polars as pl
import pytest

import hea


def test_top_level_carries_no_polars_names():
    """No ``pl.__all__`` name outside the two allowlists resolves on ``hea``."""
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
    """``hea.DataFrame`` / ``hea.LazyFrame`` / ``hea.Series`` are the ``hea.tidy``
    subclasses of the ``pl`` classes."""
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


@pytest.mark.parametrize(
    "name",
    [
        "col",
        "lit",
        "when",
        "coalesce",
        "concat_str",
        "concat_list",
        "all_horizontal",
        "any_horizontal",
        "min_horizontal",
        "max_horizontal",
        "mean_horizontal",
        "sum_horizontal",
        "Expr",
        "Schema",
    ],
)
def test_tidy_re_exports_polars_expr(name):
    """``hea.tidy`` re-exports each ``name`` unwrapped."""
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
    """``hea.tidy.exclude`` matches ``pl.exclude`` on a name and takes a
    ``hea.tidy.DataFrame`` column slice."""
    df = hea.tidy.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert (
        df.select(hea.tidy.exclude("a")).columns == df.select(pl.exclude("a")).columns
    )
    assert df.select(hea.tidy.exclude(df["a":"b"])).columns == ["c"]


def test_tidy_n_and_n_distinct_alias_polars():
    """``hea.tidy.n`` is ``pl.len`` and ``hea.tidy.n_distinct`` is ``pl.n_unique``."""
    assert hea.tidy.n is pl.len
    assert hea.tidy.n_distinct is pl.n_unique
    df = hea.tidy.DataFrame(
        {
            "dest": ["A", "A", "B", "B", "B"],
            "carrier": ["UA", "DL", "UA", "UA", "AA"],
        }
    )
    out = (
        df.group_by("dest")
        .summarize(rows=hea.tidy.n(), carriers=hea.tidy.n_distinct("carrier"))
        .arrange("dest")
    )
    assert out["rows"].to_list() == [2, 3]
    assert out["carriers"].to_list() == [2, 2]


@pytest.mark.parametrize(
    "name",
    [
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "Boolean",
        "String",
        "Utf8",
        "Date",
        "Datetime",
        "Duration",
        "Time",
        "Categorical",
        "Enum",
        "List",
        "Array",
        "Struct",
        "Object",
        "Null",
        "Decimal",
        "DataType",
    ],
)
def test_dtype_is_polars_dtype(name):
    assert getattr(hea.dtypes, name) is getattr(pl, name)


def test_dtypes_cover_every_polars_dtype():
    """``hea.dtypes`` re-exports every ``pl.DataType`` subclass in ``pl.__all__``."""
    names = [
        n
        for n in pl.__all__
        if isinstance(getattr(pl, n), type) and issubclass(getattr(pl, n), pl.DataType)
    ]
    missing = [n for n in names if getattr(hea.dtypes, n, None) is not getattr(pl, n)]
    assert not missing, f"hea.dtypes lacks polars dtypes: {missing}"


def test_constructors_return_hea_dataframe():
    cases = [
        ("from_dict", lambda: hea.tidy.from_dict({"x": [1, 2, 3]})),
        ("from_dicts", lambda: hea.tidy.from_dicts([{"x": 1}, {"x": 2}])),
        (
            "from_records",
            lambda: hea.tidy.from_records([(1, 2), (3, 4)], schema=["a", "b"]),
        ),
        (
            "concat",
            lambda: hea.tidy.concat(
                [
                    hea.tidy.from_dict({"x": [1]}),
                    hea.tidy.from_dict({"x": [2]}),
                ]
            ),
        ),
    ]
    for name, fn in cases:
        result = fn()
        assert isinstance(result, hea.tidy.DataFrame), (
            f"hea.tidy.{name}(...) returned {type(result).__module__}.{type(result).__name__}"
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


def test_io_covers_every_polars_reader_and_scanner():
    """``hea.io.__all__`` holds each ``read_*`` / ``scan_*`` name of ``pl.__all__``."""
    polars_io = {n for n in pl.__all__ if n.startswith(("read_", "scan_"))}
    missing = sorted(polars_io - set(hea.io.__all__))
    assert not missing, f"hea.io lacks polars readers/scanners: {missing}"


def test_merge_sorted_polymorphic():
    df = hea.tidy.from_dict({"x": [1, 3]}).sort("x")
    out = hea.tidy.merge_sorted([df, df], key="x")
    assert isinstance(out, hea.tidy.DataFrame)


def test_end_to_end_chain_stays_in_hea():
    """``collect()`` on a ``lazy()`` chain returns a ``hea.tidy.DataFrame``."""
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


def test_headerless_read_csv_uses_readr_names(tmp_path):
    """``hea.io.read_csv`` with ``col_names=False`` names the columns ``X1``…``Xn``."""
    p = tmp_path / "x.csv"
    p.write_text("a,b,c\n1,2,3\n")
    assert hea.io.read_csv(str(p), col_names=False).columns == ["X1", "X2", "X3"]
    assert hea.io.read_csv(str(p)).columns == ["a", "b", "c"]
    assert hea.io.read_csv(str(p), col_names=["p", "q", "r"]).columns == ["p", "q", "r"]


def test_lazy_group_by_order_matches_eager():
    """``lazy().group_by`` defaults ``maintain_order=True``."""
    df = hea.tidy.DataFrame({"g": ["b", "a", "c", "a"], "x": [1, 2, 3, 4]})
    eager = df.group_by("g").summarize(n=hea.tidy.n())["g"].to_list()
    runs = {
        tuple(df.lazy().group_by("g").agg(pl.len()).collect()["g"].to_list())
        for _ in range(8)
    }
    assert runs == {tuple(eager)}
    out = df.lazy().group_by("g", maintain_order=False).agg(pl.len()).collect()
    assert sorted(out["g"].to_list()) == ["a", "b", "c"]


def test_collect_batches_yields_hea_dataframes(tmp_path):
    """``collect_batches`` yields ``hea.tidy.DataFrame`` chunks."""
    p = tmp_path / "b.csv"
    p.write_text("a,b\n1,2\n3,4\n5,6\n")
    batches = list(hea.io.scan_csv(str(p)).collect_batches(chunk_size=1))
    assert batches, "collect_batches yielded nothing"
    for b in batches:
        assert isinstance(b, hea.tidy.DataFrame)
    assert batches[0].mutate(c=hea.tidy.col("a") * 2)["c"].to_list() == [
        batches[0]["a"][0] * 2
    ]
