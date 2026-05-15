"""I/O readers, scanners, and DataFrame factories.

Every polars top-level constructor / reader / scanner / multi-frame
operator lives here, re-exported and wrapped so the result is a hea
subclass (``hea.tidy.DataFrame`` / ``LazyFrame`` / ``Series``) instead of
a bare ``polars`` class. Plus a readr-friendly :func:`read_csv` shim that
accepts the R kwargs (``na=``, ``skip=``, ``comment=``, ``col_names=``)
common in translated R scripts.

Usage:

>>> from hea.io import read_csv, scan_parquet, concat, from_dict
>>> df = read_csv("flights.csv", na=["NA", "N/A"])
>>> wide = from_dict({"x": [1, 2, 3], "y": [4, 5, 6]})
"""
from __future__ import annotations

import functools as _functools

import polars as _pl

from .tidy import DataFrame, LazyFrame, Series

# These polars I/O helpers return plain dict[str, DataType] (schema
# introspection only) — no wrapping needed.
from polars import (  # noqa: F401  re-exported as-is
    read_ipc_schema,
    read_parquet_metadata,
    read_parquet_schema,
)


def _rewrap(obj):
    if isinstance(obj, _pl.DataFrame) and not isinstance(obj, DataFrame):
        return DataFrame._from_pydf(obj._df)
    if isinstance(obj, _pl.LazyFrame) and not isinstance(obj, LazyFrame):
        return LazyFrame._from_pyldf(obj._ldf)
    if isinstance(obj, _pl.Series) and not isinstance(obj, Series):
        return Series._from_pyseries(obj._s)
    if isinstance(obj, list):
        return [_rewrap(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_rewrap(x) for x in obj)
    return obj


def _wrap_factory(name: str):
    pl_func = getattr(_pl, name)

    @_functools.wraps(pl_func)
    def wrapper(*args, **kwargs):
        return _rewrap(pl_func(*args, **kwargs))

    return wrapper


# DataFrame-returning constructors and eager I/O readers.
_DF_FACTORIES = (
    "concat",
    "from_arrow",
    "from_dataframe",
    "from_dict",
    "from_dicts",
    "from_numpy",
    "from_pandas",
    "from_records",
    "from_repr",
    "from_torch",
    "json_normalize",
    "read_avro",
    "read_clipboard",
    "read_csv",
    "read_csv_batched",
    "read_database",
    "read_database_uri",
    "read_delta",
    "read_excel",
    "read_ipc",
    "read_ipc_stream",
    "read_json",
    "read_lines",
    "read_ndjson",
    "read_ods",
    "read_parquet",
)

# LazyFrame-returning scanners.
_LF_FACTORIES = (
    "scan_csv",
    "scan_delta",
    "scan_iceberg",
    "scan_ipc",
    "scan_lines",
    "scan_ndjson",
    "scan_parquet",
    "scan_pyarrow_dataset",
)

# Polymorphic — could return DataFrame, LazyFrame, list, etc. ``_rewrap``
# handles each case.
_POLY_FACTORIES = (
    "align_frames",
    "collect_all",
    "from_epoch",
    "merge_sorted",
    "union",
)

for _name in (*_DF_FACTORIES, *_LF_FACTORIES, *_POLY_FACTORIES):
    if hasattr(_pl, _name):
        globals()[_name] = _wrap_factory(_name)


# Override ``read_csv`` with a thin readr-kwarg shim. R-translated scripts
# use names like ``na=``, ``skip=``, ``comment=``, ``col_names=`` (readr);
# polars uses ``null_values=``, ``skip_rows=``, ``comment_prefix=``,
# ``has_header=`` / ``new_columns=``. The shim translates and dispatches.
_polars_read_csv = globals()["read_csv"]


def read_csv(source, *args, **kwargs):
    """readr-kwarg-friendly wrapper around polars ``read_csv``.

    Accepted readr aliases (translated to the polars equivalent):

    * ``na=`` → ``null_values=``
    * ``skip=`` → ``skip_rows=``
    * ``comment=`` → ``comment_prefix=``
    * ``col_names=False`` → ``has_header=False``
    * ``col_names=["a", "b", ...]`` → ``has_header=False`` + ``new_columns=...``

    Translator-stripped readr kwargs (so the .py never carries them):
    ``col_types=`` (use polars ``schema_overrides=`` for column-type
    hints), ``id=`` (multi-file id-column — port if needed).
    """
    if "na" in kwargs:
        kwargs["null_values"] = kwargs.pop("na")
    if "skip" in kwargs:
        kwargs["skip_rows"] = kwargs.pop("skip")
    if "comment" in kwargs:
        kwargs["comment_prefix"] = kwargs.pop("comment")
    if "col_names" in kwargs:
        col_names = kwargs.pop("col_names")
        if col_names is False:
            kwargs["has_header"] = False
        elif isinstance(col_names, (list, tuple)):
            kwargs["has_header"] = False
            kwargs["new_columns"] = list(col_names)
        # ``col_names=True`` is polars default — no-op.
    # readr accepts inline CSV content as the first arg (R detects this
    # heuristically — embedded newlines = literal). Polars's reader
    # treats every string as a path; wrap inline-string content in
    # StringIO so it gets parsed instead of being looked up on disk.
    if isinstance(source, str) and "\n" in source:
        import io as _io
        source = _io.StringIO(source)
    # readr also accepts a list of paths and concatenates the results
    # row-wise. Polars's reader takes a single path; emulate by reading
    # each and concatenating.
    if isinstance(source, (list, tuple)):
        frames = [_polars_read_csv(p, *args, **kwargs) for p in source]
        return _pl.concat(frames, how="vertical_relaxed")
    return _polars_read_csv(source, *args, **kwargs)


__all__ = [
    *_DF_FACTORIES, *_LF_FACTORIES, *_POLY_FACTORIES,
    "read_ipc_schema", "read_parquet_metadata", "read_parquet_schema",
]

del _name
