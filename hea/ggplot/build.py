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
    layout: pl.DataFrame | None = None  # one row per panel from facet.compute_layout


_NON_POSITIONAL_AES = ("colour", "fill", "size", "alpha", "shape", "linetype")


def build(plot) -> BuildOutput:
    """Run the build pipeline. Returns per-layer drawable frames.

    Order:

    1. Per layer: ``compute_aesthetics`` → ``add_group`` → ``stat`` → ``position``.
    2. Auto-register default scales (``ScaleContinuous`` for x/y;
       ``ScaleDiscreteColor`` for string-typed colour/fill).
    3. Train all scales across all layer data.
    4. Map non-positional scales — replaces the column with the scale's
       output (e.g. species names → hex codes for colour).
    5. Per layer: ``aes_params`` (layer constants override mapped values)
       → ``default_aes`` (geom defaults fill anything still missing).

    Stat runs before aes_params/default_aes because stats like ``stat_bin``
    change the row count; broadcasting a length-N constant onto a length-K
    stat output would mismatch.

    Mapping happens after stat/position because counts per fill *value*
    (not per fill *colour*) is what stats like ``stat_count`` need.
    """
    facet = plot.facet
    facet_vars = facet.facet_vars()
    layout = facet.compute_layout(plot.data)

    layers_data: list[pl.DataFrame] = []
    for layer in plot.layers:
        ld = _layer_data(layer, plot)
        mapping = _resolve_mapping(layer, plot)
        df = _compute_aesthetics(mapping, ld, plot.plot_env)
        # Carry facet variables alongside aes columns so map_data can
        # assign each row a panel after stat/position run.
        df = _attach_facet_columns(df, ld, facet_vars)
        df = _drop_na(df, layer)
        df = _add_group(df)
        # Stat per panel: each panel's data is computed separately so
        # group-aware stats (count, bin, density) honour facet boundaries.
        df = _stat_per_panel(df, layer, facet_vars)
        df = layer.position.compute_layer(df)
        df = facet.map_data(df, layout)
        layers_data.append(df)

    # Independent copy so multiple draw() calls don't accumulate state.
    scales = plot.scales.copy()

    # Auto-register defaults: positional always; non-positional only when
    # the data column suggests a discrete scale would help.
    for df in layers_data:
        for axis in ("x", "y"):
            if axis in df.columns:
                scales.get_or_default(axis)
        for aes in _NON_POSITIONAL_AES:
            if aes in df.columns:
                scales.get_or_default_for_data(aes, df[aes])

    # Train every registered scale on every layer's data.
    for df in layers_data:
        for aes_name, scale in list(scales.items()):
            if aes_name in df.columns:
                scale.train(df[aes_name])

    # Map non-positional scales — replace the column with mapped values
    # (e.g. ``["Adelie", "Adelie", …]`` → ``["#FF6B6B", "#FF6B6B", …]``).
    # Positional scales contribute ticks via ``apply_to_axis`` later, not
    # via column rewriting.
    for i, df in enumerate(layers_data):
        for aes_name, scale in list(scales.items()):
            if aes_name in ("x", "y"):
                continue
            if aes_name in df.columns:
                mapped = scale.map(df[aes_name])
                if isinstance(mapped, pl.Series):
                    df = df.with_columns(mapped.alias(aes_name))
                elif mapped is not df[aes_name]:
                    df = df.with_columns(pl.Series(aes_name, mapped))
        layers_data[i] = df

    # Layer-level constants and geom defaults run last so they override
    # anything the scale produced.
    for i, layer in enumerate(plot.layers):
        df = layers_data[i]
        df = _apply_aes_params(df, layer)
        df = _apply_default_aes(df, layer.geom)
        layers_data[i] = df

    return BuildOutput(data=layers_data, scales=scales, layout=layout)


def _attach_facet_columns(df: pl.DataFrame, source: pl.DataFrame,
                          facet_vars: list[str]) -> pl.DataFrame:
    """Inject facet variable columns from ``source`` into ``df`` so map_data
    can assign panels later. Only valid before stat (when row counts match)."""
    if not facet_vars or len(df) == 0 or len(df) != len(source):
        return df
    cols = [source[v].alias(v) for v in facet_vars
            if v in source.columns and v not in df.columns]
    return df.with_columns(cols) if cols else df


def _stat_per_panel(df: pl.DataFrame, layer, facet_vars: list[str]) -> pl.DataFrame:
    """Run the layer's stat once per facet-variable combination so
    group-aware stats (count, bin, density) honour panel boundaries.

    Without this, a histogram faceted by `cyl` would compute bin counts
    pooled across cyl values and split them later — producing the wrong
    bar heights per panel.
    """
    facet_in_df = [v for v in facet_vars if v in df.columns]
    if not facet_in_df:
        return layer.stat.compute_layer(df, layer.stat_params)

    chunks = []
    for keys, sub in df.group_by(facet_in_df, maintain_order=True):
        chunk = layer.stat.compute_layer(sub.drop(facet_in_df), layer.stat_params)
        if chunk is None or len(chunk) == 0:
            continue
        # Re-attach facet column values so map_data can assign panels.
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        for col, val in zip(facet_in_df, keys_tuple):
            chunk = chunk.with_columns(pl.lit(val).alias(col))
        chunks.append(chunk)
    if not chunks:
        return pl.DataFrame()
    return pl.concat(chunks, how="diagonal_relaxed")


def _drop_na(df: pl.DataFrame, layer) -> pl.DataFrame:
    """Drop rows with missing values in any mapped aesthetic, with a
    ggplot2-style warning unless ``layer.na_rm`` is True.

    Mirrors R: ``Warning: Removed N rows containing missing values
    (`geom_*()`).``  Catches both polars-null and float NaN.
    """
    if len(df) == 0:
        return df

    na_mask = None
    for col in df.columns:
        s = df[col]
        col_na = s.is_null()
        if s.dtype.is_float():
            col_na = col_na | s.is_nan()
        if col_na.any():
            na_mask = col_na if na_mask is None else (na_mask | col_na)

    if na_mask is None:
        return df
    n_dropped = int(na_mask.sum())
    if n_dropped == 0:
        return df

    out = df.filter(~na_mask)
    if not getattr(layer, "na_rm", False):
        import warnings

        geom_name = _geom_factory_name(layer.geom)
        warnings.warn(
            f"Removed {n_dropped} rows containing missing values "
            f"(`{geom_name}()`).",
            UserWarning,
            stacklevel=4,
        )
    return out


def _geom_factory_name(geom) -> str:
    """``GeomPoint`` → ``geom_point``. Used in NA-removal warnings."""
    cls = type(geom).__name__
    if cls.startswith("Geom"):
        return "geom_" + cls[4:].lower()
    return cls.lower()


def _add_group(df: pl.DataFrame) -> pl.DataFrame:
    """ggplot2's auto-grouping rule (``utilities-aes.R::add_group``).

    If the user didn't supply ``group``, derive one from any *discrete*
    non-positional aesthetic mapping. The resulting integer column splits
    line/path/polygon geoms into per-group lines automatically.
    """
    if "group" in df.columns or len(df) == 0:
        return df

    # Only colour/fill drive auto-grouping in ggplot2 — adding size/alpha/etc.
    # would over-fragment groups (e.g. ``aes(size=mpg)`` would split each row).
    discrete_cols = [
        col for col in df.columns
        if col in ("colour", "fill", "shape", "linetype")
        and df[col].dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean)
    ]
    if not discrete_cols:
        return df.with_columns(group=pl.lit(-1, dtype=pl.Int64))
    # Hash the tuple of discrete-aesthetic values so identical level
    # combinations share a group id; downstream group_by gets cheap keys.
    # ``.hash()`` returns u64 — leave it that way (no cast).
    return df.with_columns(
        group=pl.struct(discrete_cols).hash()
    )


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
