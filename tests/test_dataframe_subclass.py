"""Regression test: hea.DataFrame / hea.LazyFrame subclass coverage.

Polars operations should preserve the hea subclass identity so users don't
need ``tbl(...)`` to re-wrap. Two layers:

1. **Curated map** (``DF_METHODS`` / ``LF_METHODS``) — methods we know return
   a DataFrame/LazyFrame, paired with a callable that exercises them. We
   assert ``isinstance(result, hea.DataFrame)`` (or ``hea.LazyFrame``).
2. **Coverage check** — every public method on ``pl.DataFrame`` / ``pl.LazyFrame``
   must be in exactly one of: the curated map, ``DF_NON_DF`` (returns
   something else), or ``DF_ALLOWLIST`` (known leak; Phase 3 will fix). When
   polars adds a new public method, this fails until categorized.

Polars is pinned (see ``pyproject.toml``), so the categorization is stable
within a pinned release. The version-bump cadence runs this test.
"""

from __future__ import annotations

import polars as pl
import pytest

from hea.dataframe import DataFrame, LazyFrame, tbl

# A few exercised methods are deprecated upstream (e.g. ``approx_n_unique``,
# ``with_context``). They still return DataFrame/LazyFrame correctly today, so
# they belong in the curated set. The coverage test below will flag them when
# polars removes them.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df() -> DataFrame:
    return tbl(pl.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [4.0, 5.0, 6.0, 7.0],
        "g": ["a", "a", "b", "b"],
    }))


@pytest.fixture
def df_num() -> DataFrame:
    """Numeric-only frame for methods that reject mixed dtypes (e.g. _horizontal)."""
    return tbl(pl.DataFrame({"x": [1, 2, 3, 4], "y": [4.0, 5.0, 6.0, 7.0]}))


@pytest.fixture
def lf(df: DataFrame) -> LazyFrame:
    return df.lazy()


# ---------------------------------------------------------------------------
# Categorization — known leaks (Phase 3 will fix)
# ---------------------------------------------------------------------------

# Methods that drop subclass today. Adding to this set should be a deliberate
# decision; new entries here mark a regression unless paired with a plan.
DF_ALLOWLIST: set[str] = set()

LF_ALLOWLIST: set[str] = {
    # Streaming pivot needs materialized `on_columns` data; signature
    # differs from DataFrame.pivot. Revisit when first caller appears.
    "pivot",
}


# ---------------------------------------------------------------------------
# Categorization — methods that don't return DataFrame/LazyFrame
# ---------------------------------------------------------------------------

# Public pl.DataFrame methods whose return is not a DataFrame.
DF_NON_DF: set[str] = {
    # Returns LazyFrame — tested separately via the LazyFrame suite below.
    "lazy",
    # Returns Series.
    "drop_in_place", "fold", "get_column", "hash_rows", "is_duplicated",
    "is_unique", "max_horizontal", "mean_horizontal", "min_horizontal",
    "sum_horizontal", "to_series", "to_struct",
    # Returns scalar / collection / IO side-effect (not a frame).
    "equals", "estimated_size", "get_column_index", "glimpse", "is_empty",
    "item", "iter_columns", "iter_rows", "iter_slices", "n_chunks",
    "n_unique", "row", "rows", "rows_by_key", "to_arrow", "to_dict",
    "to_dicts", "to_init_repr", "to_jax", "to_numpy", "to_pandas", "to_torch",
    # Returns a different polars container.
    "collect_schema", "get_columns", "group_by", "group_by_dynamic",
    "partition_by", "rolling",
    # Class methods / serialization / IO writes.
    "deserialize", "serialize", "show",
    "write_avro", "write_clipboard", "write_csv", "write_database",
    "write_delta", "write_excel", "write_iceberg", "write_ipc",
    "write_ipc_stream", "write_json", "write_ndjson", "write_parquet",
    # Deprecated alias.
    "melt", "with_row_count",
}

# Public pl.LazyFrame methods whose return is not a LazyFrame.
LF_NON_DF: set[str] = {
    # Materializing terminal ops (return DataFrame — tested in
    # ``test_lf_materializing_methods_return_hea_dataframe``).
    "collect", "describe",
    # Materializing ops that return non-DataFrame containers.
    "collect_async", "collect_batches", "fetch",
    # Sinks — write to disk, return None or async result.
    "sink_batches", "sink_csv", "sink_delta", "sink_iceberg", "sink_ipc",
    "sink_ndjson", "sink_parquet",
    # Returns scalar / different container / IO.
    "collect_schema", "explain", "group_by", "group_by_dynamic",
    "profile", "remote", "rolling", "show", "show_graph",
    "deserialize", "serialize",
    # Deprecated.
    "melt", "with_row_count",
}


# ---------------------------------------------------------------------------
# Curated method map — every entry must produce a hea.DataFrame.
# ---------------------------------------------------------------------------

DF_METHODS = {
    # row verbs
    "filter":            lambda d: d.filter(pl.col("x") > 1),
    "remove":            lambda d: d.remove(pl.col("x") > 100),
    "sort":              lambda d: d.sort("x"),
    "head":              lambda d: d.head(2),
    "tail":              lambda d: d.tail(2),
    "limit":             lambda d: d.limit(2),
    "slice":             lambda d: d.slice(0, 2),
    "sample":            lambda d: d.sample(n=2, seed=0),
    "gather_every":      lambda d: d.gather_every(2),
    "reverse":           lambda d: d.reverse(),
    "unique":            lambda d: d.unique(),
    "shift":             lambda d: d.shift(1),
    "set_sorted":        lambda d: d.set_sorted("x"),
    "top_k":             lambda d: d.top_k(2, by="x"),
    "bottom_k":          lambda d: d.bottom_k(2, by="x"),

    # column verbs
    "with_columns":      lambda d: d.with_columns(z=pl.col("x") * 2),
    "with_columns_seq":  lambda d: d.with_columns_seq(z=pl.col("x") * 2),
    "with_row_index":    lambda d: d.with_row_index(),
    "select":            lambda d: d.select("x"),
    "select_seq":        lambda d: d.select_seq("x"),
    "drop":              lambda d: d.drop("y"),
    "rename":            lambda d: d.rename({"x": "X"}),
    "cast":              lambda d: d.cast({pl.Int64: pl.Float64}),
    "to_dummies":        lambda d: d.to_dummies(),
    "transpose":         lambda d: d.transpose(),

    # null handling
    "drop_nulls":        lambda d: d.drop_nulls(),
    "drop_nans":         lambda d: d.drop_nans(),
    "fill_null":         lambda d: d.fill_null(0),
    "fill_nan":          lambda d: d.fill_nan(0),
    "interpolate":       lambda d: d.interpolate(),

    # joins / set ops
    "join":              lambda d: d.join(d, on="x"),
    "join_asof":         lambda d: d.sort("x").join_asof(d.sort("x"), on="x"),
    "join_where":        lambda d: d.join_where(d, pl.col("x") < pl.col("x_right")),
    "merge_sorted":      lambda d: d.sort("x").merge_sorted(d.sort("x"), key="x"),
    "vstack":            lambda d: d.vstack(d),
    "hstack":            lambda d: d.hstack([pl.Series("z", [10, 20, 30, 40])]),
    "extend":            lambda d: d.clone().extend(d),
    "update":            lambda d: d.update(d, on="x"),

    # column-mutating in place but returning self
    "insert_column":     lambda d: d.clone().insert_column(0, pl.Series("z", [10, 20, 30, 40])),
    "replace_column":    lambda d: d.clone().replace_column(0, pl.Series("x", [10, 20, 30, 40])),

    # aggregations that return one-row DataFrame
    "approx_n_unique":   lambda d: d.approx_n_unique(),
    "count":             lambda d: d.count(),
    "max":               lambda d: d.max(),
    "mean":              lambda d: d.mean(),
    "median":            lambda d: d.median(),
    "min":               lambda d: d.min(),
    "null_count":        lambda d: d.null_count(),
    "product":           lambda d: d.product(),
    "quantile":          lambda d: d.quantile(0.5),
    "std":               lambda d: d.std(),
    "sum":               lambda d: d.sum(),
    "var":               lambda d: d.var(),

    # storage / identity
    "clone":             lambda d: d.clone(),
    "clear":             lambda d: d.clear(),
    "rechunk":           lambda d: d.rechunk(),
    "shrink_to_fit":     lambda d: d.shrink_to_fit(),

    # reshape
    "explode":           lambda d: tbl(pl.DataFrame({"a": [[1, 2]]})).explode("a"),
    "pivot":             lambda d: d.pivot(on="g", values="y", aggregate_function="first"),
    "unpivot":           lambda d: d.unpivot(on=["x", "y"], index="g"),
    "unstack":           lambda d: d.unstack(step=2),

    # bypass-_from_pydf overrides (Phase 3)
    "describe":          lambda d: d.describe(),
    "corr":              lambda d: d.select("x", "y").corr(),
    "sql":               lambda d: d.sql("SELECT * FROM self"),
    "match_to_schema":   lambda d: d.match_to_schema(d.collect_schema()),

    # user functions
    "pipe":              lambda d: d.pipe(lambda x: x.head(1)),
    "map_rows":          lambda d: d.map_rows(lambda r: (r[0] * 2,)),
    "map_columns":       lambda d: d.map_columns("x", lambda s: s * 2),
}


LF_METHODS = {
    # row verbs
    "filter":            lambda lf: lf.filter(pl.col("x") > 1),
    "remove":            lambda lf: lf.remove(pl.col("x") > 100),
    "sort":              lambda lf: lf.sort("x"),
    "head":              lambda lf: lf.head(2),
    "tail":              lambda lf: lf.tail(2),
    "first":             lambda lf: lf.first(),
    "last":              lambda lf: lf.last(),
    "limit":             lambda lf: lf.limit(2),
    "slice":             lambda lf: lf.slice(0, 2),
    "gather_every":      lambda lf: lf.gather_every(2),
    "reverse":           lambda lf: lf.reverse(),
    "unique":            lambda lf: lf.unique(),
    "shift":             lambda lf: lf.shift(1),
    "set_sorted":        lambda lf: lf.set_sorted("x"),
    "top_k":             lambda lf: lf.top_k(2, by="x"),
    "bottom_k":          lambda lf: lf.bottom_k(2, by="x"),

    # column verbs
    "with_columns":      lambda lf: lf.with_columns(z=pl.col("x") * 2),
    "with_columns_seq":  lambda lf: lf.with_columns_seq(z=pl.col("x") * 2),
    "with_row_index":    lambda lf: lf.with_row_index(),
    "with_context":      lambda lf: lf.with_context(lf),
    "select":            lambda lf: lf.select("x"),
    "select_seq":        lambda lf: lf.select_seq("x"),
    "drop":              lambda lf: lf.drop("y"),
    "rename":            lambda lf: lf.rename({"x": "X"}),
    "cast":              lambda lf: lf.cast({pl.Int64: pl.Float64}),

    # null handling
    "drop_nulls":        lambda lf: lf.drop_nulls(),
    "drop_nans":         lambda lf: lf.drop_nans(),
    "fill_null":         lambda lf: lf.fill_null(0),
    "fill_nan":          lambda lf: lf.fill_nan(0),
    "interpolate":       lambda lf: lf.interpolate(),

    # joins / set ops
    "join":              lambda lf: lf.join(lf, on="x"),
    "join_asof":         lambda lf: lf.sort("x").join_asof(lf.sort("x"), on="x"),
    "join_where":        lambda lf: lf.join_where(lf, pl.col("x") < pl.col("x_right")),
    "merge_sorted":      lambda lf: lf.sort("x").merge_sorted(lf.sort("x"), key="x"),
    "update":            lambda lf: lf.update(lf, on="x"),

    # aggregations
    "approx_n_unique":   lambda lf: lf.approx_n_unique(),
    "count":             lambda lf: lf.count(),
    "max":               lambda lf: lf.max(),
    "mean":              lambda lf: lf.mean(),
    "median":            lambda lf: lf.median(),
    "min":               lambda lf: lf.min(),
    "null_count":        lambda lf: lf.null_count(),
    "quantile":          lambda lf: lf.quantile(0.5),
    "std":               lambda lf: lf.std(),
    "sum":               lambda lf: lf.sum(),
    "var":               lambda lf: lf.var(),

    # planning / introspection
    "cache":             lambda lf: lf.cache(),
    "lazy":              lambda lf: lf.lazy(),
    "inspect":           lambda lf: lf.inspect(),

    # storage / identity
    "clone":             lambda lf: lf.clone(),
    "clear":             lambda lf: lf.clear(),

    # reshape
    "explode":           lambda lf: tbl(pl.DataFrame({"a": [[1, 2]]})).lazy().explode("a"),
    "unpivot":           lambda lf: lf.unpivot(on=["x", "y"], index="g"),
    "unnest":            lambda lf: lf.with_columns(s=pl.struct("x", "y")).select("s").unnest("s"),

    # bypass-_from_pyldf overrides (Phase 3)
    "match_to_schema":   lambda lf: lf.match_to_schema(lf.collect_schema()),
    "sql":               lambda lf: lf.sql("SELECT * FROM self"),

    # user functions
    "pipe":              lambda lf: lf.pipe(lambda x: x.head(1)),
    "pipe_with_schema":  lambda lf: lf.pipe_with_schema(lambda x, schema: x.head(1)),
    "map_batches":       lambda lf: lf.map_batches(lambda x: x.head(1)),
}


# DataFrame.unnest needs a struct column; not in the main fixture. Skip in
# DF_METHODS but acknowledge in coverage by adding to DF_METHODS via a
# specialised lambda.
DF_METHODS["unnest"] = lambda d: d.with_columns(
    s=pl.struct("x", "y")
).select("s").unnest("s")

# upsample needs a temporal index column; build dedicated frame inline.
DF_METHODS["upsample"] = lambda d: tbl(
    pl.DataFrame({
        "t": pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 4), interval="1d", eager=True
        ),
        "v": [1, 2, 3, 4],
    })
).upsample("t", every="1d")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_curated_df_methods_preserve_subclass(df: DataFrame):
    failures: list[str] = []
    for name, fn in sorted(DF_METHODS.items()):
        try:
            result = fn(df)
        except Exception as e:
            failures.append(f"  {name}: call failed ({type(e).__name__}: {e})")
            continue
        if not isinstance(result, DataFrame):
            failures.append(
                f"  {name}: returned {type(result).__module__}.{type(result).__name__}"
            )
    assert not failures, "DataFrame methods leaked subclass:\n" + "\n".join(failures)


def test_curated_lf_methods_preserve_subclass(lf: LazyFrame):
    failures: list[str] = []
    for name, fn in sorted(LF_METHODS.items()):
        try:
            result = fn(lf)
        except Exception as e:
            failures.append(f"  {name}: call failed ({type(e).__name__}: {e})")
            continue
        if not isinstance(result, LazyFrame):
            failures.append(
                f"  {name}: returned {type(result).__module__}.{type(result).__name__}"
            )
    assert not failures, "LazyFrame methods leaked subclass:\n" + "\n".join(failures)


def test_lf_materializing_methods_return_hea_dataframe(lf: LazyFrame):
    """LazyFrame methods that materialize must return ``hea.DataFrame``.

    ``collect`` is the structural fix for the lazy round-trip
    (`polars/lazyframe/frame.py:2510`). ``describe`` materializes too —
    despite living on LazyFrame it returns a DataFrame.
    """
    out = lf.collect()
    assert isinstance(out, DataFrame), (
        f"lf.collect() returned {type(out).__module__}.{type(out).__name__}"
    )

    # Multi-step lazy chains should also stay in hea-land.
    out2 = lf.filter(pl.col("x") > 0).with_columns(z=pl.col("x") + pl.col("y")).collect()
    assert isinstance(out2, DataFrame)

    # describe() is on LazyFrame but materializes — must return hea.DataFrame.
    out3 = lf.describe()
    assert isinstance(out3, DataFrame)


def _public_callables(cls) -> set[str]:
    return {
        name for name in dir(cls)
        if not name.startswith("_") and callable(getattr(cls, name, None))
    }


def test_df_method_coverage():
    """Every public ``pl.DataFrame`` method must be categorized.

    Categories: ``DF_METHODS`` (returns DataFrame, exercised), ``DF_NON_DF``
    (returns something else, OK to ignore), or ``DF_ALLOWLIST`` (known leak,
    Phase 3 will fix). New polars releases that add public methods break
    this test until categorized.
    """
    public = _public_callables(pl.DataFrame)
    categorized = set(DF_METHODS) | DF_NON_DF | DF_ALLOWLIST
    uncategorized = public - categorized
    stale = categorized - public  # entries we list but polars no longer has
    msgs = []
    if uncategorized:
        msgs.append(
            "polars added new pl.DataFrame methods that aren't categorized:\n"
            f"  {sorted(uncategorized)}\n"
            "Add to DF_METHODS (returns DataFrame), DF_NON_DF (doesn't), or DF_ALLOWLIST."
        )
    if stale:
        msgs.append(f"Categorized methods that no longer exist on pl.DataFrame: {sorted(stale)}")
    assert not msgs, "\n".join(msgs)


def test_lf_method_coverage():
    public = _public_callables(pl.LazyFrame)
    categorized = set(LF_METHODS) | LF_NON_DF | LF_ALLOWLIST
    uncategorized = public - categorized
    stale = categorized - public
    msgs = []
    if uncategorized:
        msgs.append(
            "polars added new pl.LazyFrame methods that aren't categorized:\n"
            f"  {sorted(uncategorized)}\n"
            "Add to LF_METHODS (returns LazyFrame), LF_NON_DF (doesn't), or LF_ALLOWLIST."
        )
    if stale:
        msgs.append(f"Categorized methods that no longer exist on pl.LazyFrame: {sorted(stale)}")
    assert not msgs, "\n".join(msgs)
