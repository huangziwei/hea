"""Polars datatype re-exports.

User-facing path for declaring column types in schemas, dtype-overrides
on I/O readers, and cast targets in expressions. Re-exported here so the
top-level ``hea`` namespace stays slim — every polars dtype that lives at
``polars.Int64`` lives here as ``hea.dtypes.Int64``.

Usage:

>>> from hea.dtypes import Int64, Float64, String, Date
>>> from hea.io import read_csv
>>> read_csv("foo.csv", schema_overrides={"id": Int64, "name": String})
"""
from polars import (
    Array,
    BaseExtension,
    Binary,
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Extension,
    Field,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    List,
    Null,
    Object,
    String,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    Utf8,
)

__all__ = [
    "Array", "BaseExtension", "Binary", "Boolean", "Categorical",
    "DataType", "Date", "Datetime", "Decimal", "Duration", "Enum",
    "Extension", "Field", "Float16", "Float32", "Float64",
    "Int8", "Int16", "Int32", "Int64", "Int128",
    "List", "Null", "Object", "String", "Struct", "Time",
    "UInt8", "UInt16", "UInt32", "UInt64", "UInt128",
    "Unknown", "Utf8",
]
