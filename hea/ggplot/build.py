"""Build pipeline — turns a ``ggplot`` object into per-layer drawable data.

Phase 0 form: identity scale + identity coord + identity stat + identity
position. Just evaluates each layer's aes mapping against its data,
stitches in default-aes constants, and returns the per-layer
DataFrame ready for :func:`render.render`.

Real scale training, position adjustment, coord transformation, and
faceting come in Phase 1 (checklist 1.1, 1.3, 1.7, plus polar in
Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .aes import Aes
from ._util import to_series
from .scales.list import ScalesList


@dataclass
class BuildOutput:
    data: list[pl.DataFrame]  # one per layer; columns = canonical aes names
    scales: ScalesList = None


def build(plot) -> BuildOutput:
    """Run the build pipeline. Returns per-layer drawable frames.

    Order: mapping → stat → position → aes_params → default_aes. Stat runs
    *before* aes_params/default_aes because stats like ``stat_bin`` change
    the row count (counts per bin), and broadcasting a length-N constant
    aes onto a length-K stat output would mismatch.

    A copy of the plot's :class:`ScalesList` is returned with auto-registered
    defaults for any positional aesthetic the user mapped but didn't add an
    explicit scale for.
    """
    layers_data: list[pl.DataFrame] = []
    for layer in plot.layers:
        ld = _layer_data(layer, plot)
        mapping = _resolve_mapping(layer, plot)
        df = _compute_aesthetics(mapping, ld, plot.plot_env)
        df = layer.stat.compute_layer(df, layer.stat_params)
        df = layer.position.compute_layer(df)
        df = _apply_aes_params(df, layer)
        df = _apply_default_aes(df, layer.geom)
        layers_data.append(df)

    # Independent copy so multiple draw() calls don't accumulate state.
    scales = plot.scales.copy()
    # Auto-register defaults for any positional aesthetic that has data.
    for df in layers_data:
        for axis in ("x", "y"):
            if axis in df.columns:
                scales.get_or_default(axis)

    return BuildOutput(data=layers_data, scales=scales)


def _layer_data(layer, plot) -> pl.DataFrame:
    if layer.data is None:
        return plot.data
    if callable(layer.data):
        return layer.data(plot.data)
    return layer.data


def _resolve_mapping(layer, plot) -> Aes:
    if layer.inherit_aes:
        merged = Aes(plot.mapping)
        if layer.mapping is not None:
            merged = merged + layer.mapping
        return merged
    return layer.mapping if layer.mapping is not None else Aes()


def _compute_aesthetics(mapping: Aes, data: pl.DataFrame, env: dict) -> pl.DataFrame:
    n = len(data)
    out_cols: dict[str, pl.Series] = {}
    for aes_name, expr in mapping.items():
        value = _eval_aes_value(expr, data, env)
        out_cols[aes_name] = to_series(value, n, name=aes_name)
    return pl.DataFrame(out_cols) if out_cols else pl.DataFrame({"_dummy": [None] * n})


def _eval_aes_value(expr, data: pl.DataFrame, env: dict):
    """Resolve one aes value. See plan §13 Q3 for disambiguation rules."""
    if callable(expr) and not isinstance(expr, str):
        return expr(data)

    if isinstance(expr, str):
        # Bare-column-name fast path: avoids parser overhead and side-steps
        # cases where a column name (e.g. "Sepal.Length") looks like an
        # expression to the formula parser.
        if expr in data.columns:
            return data[expr]

        # Otherwise: parse as an expression and eval against data + env.
        from ..formula import parse
        from ..plot.formula_eval import DEFAULT_ENV, eval_node

        # `parse` wants a formula; wrap as `~ <expr>` and take the rhs.
        ast = parse(f"~ {expr}").rhs
        full_env = {**DEFAULT_ENV, **env}
        return eval_node(ast, data, full_env)

    # Anything else (number, list, etc.) treated as a constant — broadcast in to_series.
    return expr


def _apply_aes_params(df: pl.DataFrame, layer) -> pl.DataFrame:
    """``geom_point(colour="red")`` — constants set as kwargs override mapping."""
    n = len(df) if len(df.columns) else 0
    for k, v in layer.aes_params.items():
        from .aes import _canon
        col = _canon(k)
        if n == 0:
            continue
        df = df.with_columns(to_series(v, n, name=col).alias(col))
    return df


def _apply_default_aes(df: pl.DataFrame, geom) -> pl.DataFrame:
    """Fill in any aes the geom needs but the layer didn't set."""
    n = len(df) if len(df.columns) else 0
    if n == 0:
        return df
    for aes_name, default in geom.default_aes.items():
        if aes_name not in df.columns:
            df = df.with_columns(to_series(default, n, name=aes_name).alias(aes_name))
    return df
