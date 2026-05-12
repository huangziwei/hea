"""Build pipeline — turns a ``ggplot`` object into per-layer drawable data.

Phase 0 form: identity scale + identity coord + identity stat + identity
position. Just evaluates each layer's aes mapping against its data,
stitches in default-aes constants, and returns the per-layer
DataFrame ready for :func:`render.render`.

Real scale training, position adjustment, coord transformation, and
faceting come in Phase 1 (checklist 1.1, 1.3, 1.7, plus polar in
Phase 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    # Original aes mapping value (typically a column name) per aesthetic,
    # used by the guide system for legend titles and auto-merge keys.
    aes_source: dict = None
    # Effective per-layer mapping after ``_promote_string_aes_params``.
    # Same length as ``data``; ``layer_mappings[i]`` is what
    # ``plot.layers[i]`` actually maps after promoting column-shaped
    # ``aes_params`` (``geom_point(color="species")``) into the mapping.
    # Used by the guide system: ``_find_layer_for_aes`` would otherwise
    # only see ``layer.mapping`` and miss kwargs-style mappings.
    layer_mappings: list[dict] | None = None
    # Effective per-layer ``aes_params`` after promotion — column-shaped
    # entries (``geom_point(color="species")``) are stripped because
    # they're actual mappings, not constants. Without this the legend
    # would treat ``"species"`` as a literal colour and matplotlib would
    # choke. Same length as ``data``.
    layer_aes_params: list[dict] | None = None
    # Per-panel positional scales — populated by ``_train_panel_scales``
    # when ``scales != "fixed"``. For fixed mode every panel maps to the
    # same global scale instance (kept here for uniform render-time lookup).
    # Shape: ``{panel_id: {"x": Scale, "y": Scale}}``.
    panel_scales: dict = field(default_factory=dict)
    # Coord instance for the plot. Populated from ``plot.coordinates``;
    # render branches on its type (e.g. polar needs ``projection="polar"``
    # at axes-creation time, before any geom draws).
    coord: object = None


_NON_POSITIONAL_AES = ("colour", "fill", "size", "alpha", "shape", "linetype")

# Aesthetics that share the x or y scale's coordinate space. Training the
# positional scale on every present sibling lets the axis range cover the
# full data extent — without ``ymin`` the y scale for ``geom_bar`` would
# train on counts only (e.g. [68, 152]) and miss the bar baseline at 0;
# without ``lower``/``upper`` ``geom_boxplot`` would have no ``y`` column
# to train on at all; without ``outliers`` (a list column) boxplot ticks
# would max out at the upper whisker and miss outlier points drawn beyond.
_X_POSITIONAL_AES = ("x", "xmin", "xmax", "xend", "xintercept",
                     "xlower", "xmiddle", "xupper")
_Y_POSITIONAL_AES = ("y", "ymin", "ymax", "yend", "yintercept",
                     "lower", "middle", "upper")
# ``outliers`` is a list column produced by stat_boxplot. It belongs on
# the *distribution* axis, which is normally y but flips to x when the
# layer's data carries ``flipped_aes=True``. Train it dynamically.
_OUTLIERS_COLUMN = "outliers"


def _positional_aes_for(axis: str) -> tuple[str, ...]:
    return _X_POSITIONAL_AES if axis == "x" else _Y_POSITIONAL_AES


def _share_key(panel_row: dict, mode):
    """Return the scale-group key for ``panel_row`` under matplotlib-style
    share mode. Two panels with the same key share an axis scale instance.

    * ``True``        → all panels share (``"_global_"``).
    * ``False``       → each panel is independent (the panel id).
    * ``"col"``       → grouped by column (facet_grid free_x).
    * ``"row"``       → grouped by row (facet_grid free_y).
    """
    if mode is True:
        return "_global_"
    if mode is False:
        return ("panel", panel_row.get("PANEL"))
    if mode == "col":
        return ("col", panel_row.get("COL"))
    if mode == "row":
        return ("row", panel_row.get("ROW"))
    return "_global_"


def _train_panel_scales(plot, scales, layers_data, layout) -> dict:
    """Build per-panel positional scales by cloning the global x/y scales
    and re-training each clone on the data inside that panel's group.

    For ``scales="fixed"`` every panel keys into the same group, so the
    returned dict effectively points at a single trained clone per axis —
    same range as the global scale, but a fresh instance so per-panel
    apply_to_axis side-effects don't leak across panels.

    For ``free*`` modes each scale-group (per-col, per-row, or per-panel)
    sees only its own data, which is what makes the per-column / per-row
    axis limits in facet_grid correct.
    """
    import copy

    panel_scales: dict = {}
    if layout is None or len(layout) == 0:
        return panel_scales

    facet = getattr(plot, "facet", None)
    if facet is None or not hasattr(facet, "share_axes"):
        return panel_scales
    sharex, sharey = facet.share_axes()

    global_x = scales.get("x") if scales is not None else None
    global_y = scales.get("y") if scales is not None else None

    # Group panels by share key per axis. ``key_to_scale_*`` stores one
    # cloned scale per group, which all panels in the group will share.
    key_to_scale_x: dict = {}
    key_to_scale_y: dict = {}
    for panel_row in layout.iter_rows(named=True):
        kx = _share_key(panel_row, sharex)
        ky = _share_key(panel_row, sharey)
        if global_x is not None and kx not in key_to_scale_x:
            cloned = copy.deepcopy(global_x)
            cloned.range_ = None
            key_to_scale_x[kx] = cloned
        if global_y is not None and ky not in key_to_scale_y:
            cloned = copy.deepcopy(global_y)
            cloned.range_ = None
            key_to_scale_y[ky] = cloned
        pid = panel_row["PANEL"]
        panel_scales[pid] = {
            "x": key_to_scale_x.get(kx),
            "y": key_to_scale_y.get(ky),
        }

    # Train each scale-group's clone on data only from panels in that group.
    panel_to_keys: dict = {
        r["PANEL"]: (_share_key(r, sharex), _share_key(r, sharey))
        for r in layout.iter_rows(named=True)
    }
    for df in layers_data:
        if "PANEL" not in df.columns:
            continue
        for pid, (kx, ky) in panel_to_keys.items():
            sub = df.filter(pl.col("PANEL") == pid)
            if len(sub) == 0:
                continue
            scale_x = key_to_scale_x.get(kx)
            scale_y = key_to_scale_y.get(ky)
            if scale_x is not None:
                for col in _X_POSITIONAL_AES:
                    if col in sub.columns:
                        _train_series(scale_x, sub[col])
            if scale_y is not None:
                for col in _Y_POSITIONAL_AES:
                    if col in sub.columns:
                        _train_series(scale_y, sub[col])
                # outliers: same flipped/non-flipped routing as the global
                # training step above. Only consider rows in this panel.
                if _OUTLIERS_COLUMN in sub.columns:
                    flipped = (
                        "flipped_aes" in sub.columns
                        and bool(sub["flipped_aes"].any())
                    )
                    target = scale_x if flipped else scale_y
                    if target is not None:
                        _train_series(target, sub[_OUTLIERS_COLUMN])
    return panel_scales


def _train_series(scale, series) -> None:
    """Train ``scale`` on ``series``, flattening list-typed columns first.

    ``stat_boxplot`` stores per-box outlier values in a ``List(Float64)``
    column; calling ``scale.train`` on the list directly would feed a
    length-N series of lists to ``min/max`` and miss the outlier extents.
    Explode the list, drop nulls (empty lists explode to a single null),
    and then train on the flattened scalars.
    """
    import polars as pl

    if isinstance(series.dtype, pl.List):
        series = series.explode().drop_nulls()
    scale.train(series)


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

    # Pre-resolve each layer's effective mapping (after kwarg promotion)
    # so aes_source tracking + the per-layer build loop see the same
    # picture. Without this, ``geom_point(color="species",
    # shape="species")`` would get TWO separate legends (colour and
    # shape) instead of one merged legend — guide auto-merge keys off
    # aes_source.
    resolved: list[tuple] = []  # (layer, ld, mapping, effective_aes_params)
    for layer in plot.layers:
        ld = _layer_data(layer, plot)
        mapping = _resolve_mapping(layer, plot)
        mapping, layer_eff_params = _promote_string_aes_params(
            mapping, layer.aes_params, ld,
        )
        resolved.append((layer, ld, mapping, layer_eff_params))

    # Track the original aes value (column name / expression) per aesthetic
    # so the guide system can name legend titles and detect auto-merge.
    # First layer wins for each aesthetic (matches ggplot2's "first scale"
    # semantics — later layers don't rename the legend).
    aes_source: dict = {}
    for _, _, mapping, _ in resolved:
        for aes_name, value in mapping.items():
            if aes_name in aes_source:
                continue
            if isinstance(value, str):
                aes_source[aes_name] = value
                continue
            # Tagged callables (``fct_reorder("class", ...)`` etc.) carry
            # the source column name — same role as a bare string mapping
            # for legend titles + auto-merge keys.
            tagged = getattr(value, "__hea_aes_source__", None)
            if isinstance(tagged, str):
                aes_source[aes_name] = tagged

    # Pass 1 (per layer): aesthetics → stat → after_stat. Stop BEFORE
    # position so the scale system can see the original dtype of x / y
    # — discrete column → ``ScaleOrdinal``, continuous → ``ScaleContinuous``.
    # Position adjustments (dodge / nudge / jitter) convert discrete
    # axes to integer positions for arithmetic; if scale registration
    # ran after that, ``ScaleContinuous`` would always win and the
    # discrete tick labels would be lost.
    pre_position: list[pl.DataFrame] = []
    deferred_after_stat: list[dict] = []  # one per layer
    deferred_after_scale: list[dict] = []
    effective_aes_params: list[dict] = []  # per-layer; reduced after promotion.
    for layer, ld, mapping, layer_eff_params in resolved:
        effective_aes_params.append(layer_eff_params)
        immediate, after_stat_map, after_scale_map = _split_deferred(mapping)
        df = _compute_aesthetics(immediate, ld, plot.plot_env)
        df = _attach_facet_columns(df, ld, facet_vars)
        df = _drop_na(df, layer)
        df = _add_group(df)
        # Drop rows outside ``scale_x/y_*(limits=...)`` BEFORE the stat
        # so smooths / bins / counts only see the in-range subset.
        # Distinct from ``coord_cartesian(xlim=...)`` which keeps data
        # and only zooms at render time.
        df = _drop_out_of_scale_limits(df, plot.scales)
        # Apply user-added positional scale transforms (scale_x_log10 etc.)
        # to the data BEFORE the stat runs — matches ggplot2's pipeline so
        # binning, smoothing, etc. operate on transformed values. Without
        # this, ``geom_bin2d() + scale_x_log10()`` would bin in linear
        # space then visually log-warp the bins (= wrong; chunky bins).
        df = _apply_scale_transforms_pre_stat(df, plot.scales)
        df = _stat_per_panel(df, layer, facet_vars)
        df = _apply_deferred(df, after_stat_map, plot.plot_env)
        pre_position.append(df)
        deferred_after_stat.append(after_stat_map)
        deferred_after_scale.append(after_scale_map)

    # Independent copy so multiple draw() calls don't accumulate state.
    scales = plot.scales.copy()

    # Auto-register defaults from the pre-position data so dtype-based
    # choice (discrete vs continuous) sees the user's original column
    # type, not whatever position adjustment hands back.
    import polars as _pl_for_dtype

    for df in pre_position:
        for axis in ("x", "y"):
            sibling_for_dtype = next(
                (df[c] for c in _positional_aes_for(axis)
                 if c in df.columns
                 and not isinstance(df[c].dtype, _pl_for_dtype.List)),
                None,
            )
            if sibling_for_dtype is not None:
                scales.get_or_default(axis, data=sibling_for_dtype)
        for aes in _NON_POSITIONAL_AES:
            if aes in df.columns:
                scales.get_or_default_for_data(aes, df[aes])

    # Train ordinal scales on pre-position data so their level catalogue
    # is locked in before position rewrites x / y to numeric. Continuous
    # scales train post-position (need the actual numeric extent).
    from .scales.ordinal import ScaleOrdinal

    for df in pre_position:
        for axis in ("x", "y"):
            sc = scales.get(axis)
            if isinstance(sc, ScaleOrdinal) and axis in df.columns:
                _train_series(sc, df[axis])

    # Drop rows outside an explicit ``scale_x_discrete(limits=...)`` /
    # ``scale_y_discrete(limits=...)`` BEFORE position runs, so the
    # dodge / jitter rank counts only see the kept categories.
    for axis in ("x", "y"):
        sc = scales.get(axis)
        if not isinstance(sc, ScaleOrdinal) or sc.limits is None:
            continue
        keep = [str(v) for v in sc.resolved_limits()]
        for i, df in enumerate(pre_position):
            if axis not in df.columns or len(df) == 0:
                continue
            mask = df[axis].cast(pl.Utf8).is_in(keep)
            pre_position[i] = df.filter(mask)

    # Pass 2: position → setup_data → facet map. Ordinal scales already
    # know their levels at this point, so position can convert discrete
    # x / y to integer positions without losing axis identity.
    layers_data: list[pl.DataFrame] = []
    for layer, df in zip(plot.layers, pre_position):
        df = _position_per_panel(df, layer, facet_vars)
        # Geom-specific data prep — bar/area geoms expose their implicit
        # y=0 baseline as ymin/ymax here so the next phase (scale
        # training) sees 0 in the y data range.
        df = layer.geom.setup_data(df)
        if not getattr(layer, "broadcast_panels", False):
            df = facet.map_data(df, layout)
        layers_data.append(df)

    # Train every registered scale on every layer's data. Positional scales
    # train on every present sibling (y, ymin, ymax, lower, …) so the axis
    # range spans the full data extent, not just the primary aesthetic.
    # ``ScaleOrdinal`` already captured its levels in the pre-position
    # pass; ``train()`` skips numeric data so post-position numeric
    # positions don't pollute the level list.
    for df in layers_data:
        # ``flipped_aes`` is the per-row flag stat_boxplot tags onto its
        # output when the distribution lives on x; outliers (and any
        # other distribution-axis-only column) follow.
        flipped = (
            "flipped_aes" in df.columns
            and bool(df["flipped_aes"].any())
        )
        outliers_axis = "x" if flipped else "y"
        for aes_name, scale in list(scales.items()):
            if aes_name in ("x", "y"):
                for col in _positional_aes_for(aes_name):
                    if col in df.columns:
                        _train_series(scale, df[col])
                if (
                    aes_name == outliers_axis
                    and _OUTLIERS_COLUMN in df.columns
                ):
                    _train_series(scale, df[_OUTLIERS_COLUMN])
            elif aes_name in df.columns:
                scale.train(df[aes_name])

    # Per-panel positional scales — for ``scales="free*"`` each panel (or
    # column / row, in facet_grid) needs its own trained range so axis
    # limits and ticks reflect that panel's data, not the global pool.
    # ``_train_panel_scales`` returns ``{panel_id: {"x": scale, "y": scale}}``
    # — every panel maps to the right scale instance regardless of the
    # share mode (fixed → one shared object; col/row → grouped clones;
    # per-panel → fresh clone each).
    panel_scales = _train_panel_scales(plot, scales, layers_data, layout)

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
        # Resolve after_scale() now that scale-mapped columns are in place.
        df = _apply_deferred(df, deferred_after_scale[i], plot.plot_env)
        layers_data[i] = df

    # Layer-level constants and geom defaults run last so they override
    # anything the scale produced.
    for i, layer in enumerate(plot.layers):
        df = layers_data[i]
        df = _apply_aes_params(df, layer, effective_aes_params[i])
        df = _apply_default_aes(df, layer.geom)
        layers_data[i] = df

    layer_mappings = [dict(m) if m is not None else {}
                      for _, _, m, _ in resolved]
    layer_aes_params_out = [dict(p) if p is not None else {}
                            for _, _, _, p in resolved]
    return BuildOutput(
        data=layers_data, scales=scales, layout=layout, aes_source=aes_source,
        layer_mappings=layer_mappings, layer_aes_params=layer_aes_params_out,
        panel_scales=panel_scales, coord=plot.coordinates,
    )


def _split_deferred(mapping):
    """Return ``(immediate, after_stat_map, after_scale_map)`` from a mapping.

    Markers ``after_stat()`` / ``after_scale()`` push the aes value to a
    later phase; everything else evaluates against the raw layer data."""
    from .aes import AfterScale, AfterStat

    immediate = type(mapping)()
    after_stat_map: dict = {}
    after_scale_map: dict = {}
    for name, value in mapping.items():
        if isinstance(value, AfterStat):
            after_stat_map[name] = value.expr
        elif isinstance(value, AfterScale):
            after_scale_map[name] = value.expr
        else:
            immediate[name] = value
    return immediate, after_stat_map, after_scale_map


def _apply_deferred(df: pl.DataFrame, deferred: dict, env: dict) -> pl.DataFrame:
    """Resolve a deferred-aes dict against the current ``df``. Each value
    can be a column name, a callable, or an expression string parseable by
    :func:`hea.formula.parse`."""
    if not deferred or len(df) == 0:
        return df
    for aes_name, expr in deferred.items():
        value = _eval_aes_value(expr, df, env)
        # ``_eval_aes_value`` returns a polars Series for column lookups, a
        # numpy array / list for callables, or a scalar literal otherwise.
        df = df.with_columns(to_series(value, len(df), name=aes_name).alias(aes_name))
    return df


def _attach_facet_columns(df: pl.DataFrame, source: pl.DataFrame,
                          facet_vars: list[str]) -> pl.DataFrame:
    """Inject facet variable columns from ``source`` into ``df`` so map_data
    can assign panels later. Only valid before stat (when row counts match)."""
    if not facet_vars or len(df) == 0 or len(df) != len(source):
        return df
    cols = [source[v].alias(v) for v in facet_vars
            if v in source.columns and v not in df.columns]
    return df.with_columns(cols) if cols else df


def _drop_out_of_scale_limits(df: pl.DataFrame, scales) -> pl.DataFrame:
    """Drop rows whose positional value lies outside an explicit
    ``scale_x/y_*(limits=...)`` range.

    Mirrors ggplot2's pipeline: ``scale_*_continuous(limits=c(lo, hi))``
    REMOVES out-of-range data before the stat (so e.g. ``geom_smooth``
    fits only on the in-range subset). This is distinct from
    ``coord_cartesian(xlim=...)``, which keeps the data and only zooms
    the axis at render time.

    Limits with ``None`` or ``NaN`` for one bound mean "no constraint
    on that side" (matches R's ``c(NA, 6)``).
    """
    if df is None or len(df) == 0 or len(df.columns) == 0:
        return df
    import math as _math

    for axis in ("x", "y"):
        sc = scales.get(axis)
        if sc is None or getattr(sc, "limits", None) is None:
            continue
        lim = sc.limits
        if not isinstance(lim, (list, tuple)) or len(lim) != 2:
            continue
        # ScaleOrdinal's `limits=` is a list of CATEGORY NAMES, not a
        # numeric range — drop-by-range doesn't apply there. The
        # discrete-mapping pipeline filters by `resolved_limits()` later.
        from .scales.ordinal import ScaleOrdinal

        if isinstance(sc, ScaleOrdinal):
            continue
        lo, hi = lim
        # Treat None/NaN as open bound on that side.
        def _is_open(b):
            return b is None or (isinstance(b, float) and _math.isnan(b))
        if _is_open(lo) and _is_open(hi):
            continue
        for sibling in _positional_aes_for(axis):
            if sibling not in df.columns:
                continue
            col = pl.col(sibling)
            mask = pl.lit(True)
            if not _is_open(lo):
                mask = mask & (col >= lo)
            if not _is_open(hi):
                mask = mask & (col <= hi)
            # Keep nulls (matches `_drop_na`'s contract — that step
            # already removed any aesthetic-required nulls upstream).
            df = df.filter(mask | col.is_null())
    return df


def _apply_scale_transforms_pre_stat(df: pl.DataFrame, scales) -> pl.DataFrame:
    """Apply user-added positional scale transforms to the data BEFORE
    ``stat.compute_layer`` runs. Mirrors ggplot2's pipeline.

    Looks at the user-explicitly-added scales in ``plot.scales`` (NOT
    auto-registered ones — those don't have a transform yet). For any
    explicit scale with a non-identity ``transform``, every positional
    sibling column of that aesthetic (``x`` plus ``xmin``/``xmax``/
    ``xend``/etc.) is transformed in-place.
    """
    if df is None or len(df) == 0 or len(df.columns) == 0:
        return df
    for axis in ("x", "y"):
        sc = scales.get(axis)
        if sc is None:
            continue
        trans = getattr(sc, "transform", None)
        if trans is None or trans.name == "identity":
            continue
        for sibling in _positional_aes_for(axis):
            if sibling not in df.columns:
                continue
            arr = df[sibling].to_numpy()
            try:
                transformed = trans.transform(arr)
            except Exception:
                continue
            df = df.with_columns(pl.Series(sibling, transformed))
    return df


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

    # Capture original dtypes so re-attached scalar values keep them
    # (otherwise an Enum-typed facet variable becomes Utf8 after pl.lit and
    # the join in facet.map_data fails with a schema mismatch).
    facet_dtypes = {v: df[v].dtype for v in facet_in_df}

    chunks = []
    for keys, sub in df.group_by(facet_in_df, maintain_order=True):
        chunk = layer.stat.compute_layer(sub.drop(facet_in_df), layer.stat_params)
        if chunk is None or len(chunk) == 0:
            continue
        # Re-attach facet column values so map_data can assign panels.
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        for col, val in zip(facet_in_df, keys_tuple):
            chunk = chunk.with_columns(
                pl.lit(val).cast(facet_dtypes[col], strict=False).alias(col)
            )
        chunks.append(chunk)
    if not chunks:
        return pl.DataFrame()
    return pl.concat(chunks, how="diagonal_relaxed")


def _position_per_panel(df: pl.DataFrame, layer, facet_vars: list[str]) -> pl.DataFrame:
    """Run the layer's position adjustment once per facet-variable combination.

    Without this, ``position_stack`` (and ``position_fill``) would
    cumulative-sum across panels — bars in panel 1 would stack on top
    of panel 0's bars at the same x. Mirrors :func:`_stat_per_panel`.
    """
    facet_in_df = [v for v in facet_vars if v in df.columns]
    if not facet_in_df:
        return layer.position.compute_layer(df)

    facet_dtypes = {v: df[v].dtype for v in facet_in_df}

    chunks = []
    for keys, sub in df.group_by(facet_in_df, maintain_order=True):
        chunk = layer.position.compute_layer(sub.drop(facet_in_df))
        if chunk is None or len(chunk) == 0:
            continue
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        for col, val in zip(facet_in_df, keys_tuple):
            chunk = chunk.with_columns(
                pl.lit(val).cast(facet_dtypes[col], strict=False).alias(col)
            )
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

    Group codes are assigned in the SORT ORDER of the discrete columns
    (matches R's ``interaction()``), not via hashing — so that
    ``position_stack`` produces stacks in alphabetical / factor-level
    order (e.g. Adelie at the bottom, Gentoo at the top), matching
    ggplot2's output.
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

    # Build a sorted-unique table of the discrete combinations and join
    # it back to assign 1-based integer codes in sort order.
    levels = (
        df.select(discrete_cols)
        .unique(maintain_order=False)
        .sort(discrete_cols)
        .with_row_index(name="group", offset=1)
        .with_columns(pl.col("group").cast(pl.Int64))
    )
    return df.join(levels, on=discrete_cols, how="left")


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
    # Polars expression — evaluate against the layer's data. Lets users
    # write ``aes(x=col("carat").log())`` and the like; without this the
    # Expr would fall through as a "constant" and ``to_series`` would
    # try to numpy-broadcast it (which fails because Expr can't convert).
    if isinstance(expr, pl.Expr):
        return data.select(expr).to_series()

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


def _apply_aes_params(df: pl.DataFrame, layer, aes_params=None) -> pl.DataFrame:
    """``geom_point(colour="red")`` — constants set as kwargs override mapping.

    ``aes_params`` overrides ``layer.aes_params`` — used by the build
    pipeline after string-valued kwargs that match a data column have
    been *promoted* into the mapping (so the user-friendly
    ``geom_point(color="species")`` does what they expect)."""
    n = len(df) if len(df.columns) else 0
    if aes_params is None:
        aes_params = layer.aes_params
    for k, v in aes_params.items():
        from .aes import _canon
        col = _canon(k)
        if n == 0:
            continue
        df = df.with_columns(to_series(v, n, name=col).alias(col))
    return df


def _promote_string_aes_params(mapping, aes_params, data):
    """Move ``aes_params`` whose value is a column-reference-shaped
    thing INTO the mapping (= map), leaving plain constants in
    ``aes_params`` (= set).

    Three shapes promote:

    * Strings matching a data column — ``geom_point(color="species")``
      behaves like ``geom_point(aes(color="species"))`` when ``species``
      exists. ``color="red"`` (no such column) stays SET.
    * Polars ``Expr`` — ``geom_tile(fill=pl.col("n"))`` or
      ``geom_tile(fill=pl.len())``. Without this, the Expr would flow
      through ``to_series`` unevaluated and crash on
      ``np.asarray(<Expr>)``. Promoting routes it through the build
      pipeline's ``_eval_aes_value`` (which calls ``data.select(expr)``)
      so the Expr is computed against the layer data, just like a
      mapped column.
    * Callables — ``geom_boxplot(group=cut_width(col("carat"), 0.1))``
      or ``geom_point(colour=fct_reorder("class", "hwy"))``. The
      ``cut_*`` / ``fct_*`` helpers return closures that consume a
      DataFrame; without promotion the closure would land in
      ``to_series`` and broadcast as a 1-row Object (silent wrong
      result — one box for the whole panel instead of one per bin).

    Explicit ``aes(color=...)`` still wins (the promoted entry has
    lower priority in the merge)."""
    if not aes_params:
        return mapping, aes_params
    from .aes import Aes, AfterScale, AfterStat, _canon

    promoted = Aes()
    keep = {}
    for k, v in aes_params.items():
        canon = _canon(k)
        if isinstance(v, str) and v in data.columns:
            promoted[canon] = v
        elif isinstance(v, pl.Expr):
            promoted[canon] = v
        elif isinstance(v, (AfterStat, AfterScale)):
            # ``geom_bar(y=after_stat("sqrt(count)"))`` — without this,
            # the marker stays in aes_params and gets broadcast as a
            # scalar object (=raw counts on the bar). Fluent / kwarg
            # users shouldn't have to wrap in ``aes(...)`` to get the
            # deferred-stat pipeline.
            promoted[canon] = v
        elif callable(v) and not isinstance(v, type):
            promoted[canon] = v
        else:
            keep[k] = v
    if not promoted:
        return mapping, aes_params
    # promoted has LOWER priority than the existing mapping — explicit
    # ``aes(color=...)`` should beat a kwarg fallback.
    new_mapping = promoted + (mapping if mapping is not None else Aes())
    return new_mapping, keep


def _apply_default_aes(df: pl.DataFrame, geom) -> pl.DataFrame:
    """Fill in any aes the geom needs but the layer didn't set."""
    n = len(df) if len(df.columns) else 0
    if n == 0:
        return df
    for aes_name, default in geom.default_aes.items():
        if aes_name not in df.columns:
            df = df.with_columns(to_series(default, n, name=aes_name).alias(aes_name))
    return df
