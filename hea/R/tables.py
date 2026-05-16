"""R's contingency-table family: ``table`` (1- and 2-way frequency),
``xtabs`` (formula form), ``prop_table`` (counts → proportions),
``addmargins`` (append Sum rows / columns).

Returns polars DataFrames rather than R's "table" S3 object — the first
column carries the row labels, with one count column per ``y`` level.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def table(x, y=None, *, dnn=None):
    """R: ``table()`` — 1- or 2-way frequency table.

    1-way: returns a 2-column DataFrame ``(value, n)`` sorted by value.
    2-way: returns a wide DataFrame whose first column is the row label
    (named ``""`` by default; pass ``dnn=("row", "col")`` for R-style
    dimension names) and remaining columns are the levels of ``y``.

    Nulls are dropped (R's ``useNA="no"`` default).
    """
    if y is None:
        s = pl.Series(x).drop_nulls().cast(pl.Utf8)
        out = (
            pl.DataFrame({"value": s})
            .group_by("value").len(name="n")
            .sort("value")
        )
        if dnn is not None:
            out = out.rename({"value": str(dnn[0])})
        return out

    df = pl.DataFrame({
        "__x__": pl.Series(x).cast(pl.Utf8),
        "__y__": pl.Series(y).cast(pl.Utf8),
    }).drop_nulls()
    counts = df.group_by(["__x__", "__y__"]).len(name="n")
    pivot = (
        counts.pivot(values="n", index="__x__", on="__y__")
        .fill_null(0)
        .sort("__x__")
    )
    # sort columns alphabetically so output is reproducible (polars'
    # pivot uses encounter order, which depends on group_by ordering)
    label_col = "__x__"
    other_cols = sorted(c for c in pivot.columns if c != label_col)
    pivot = pivot.select([label_col, *other_cols])
    new_label = str(dnn[0]) if dnn is not None else ""
    return pivot.rename({label_col: new_label})


def xtabs(formula: str, data: pl.DataFrame):
    """R: ``xtabs()`` — formula-based contingency table.

    Count form (``~ a`` / ``~ a + b``) returns level counts. Weighted
    form (``w ~ a`` / ``w ~ a + b``) sums ``w`` per cell — the same
    layout, just aggregating a numeric column instead of counting rows.
    """
    if "~" not in formula:
        raise ValueError("xtabs(): formula must contain '~'")
    lhs, rhs = formula.split("~", 1)
    lhs = lhs.strip()
    cols = [p.strip() for p in rhs.split("+")]
    if lhs == "":
        # Count form.
        if len(cols) == 1:
            return table(data[cols[0]])
        if len(cols) == 2:
            return table(data[cols[0]], data[cols[1]], dnn=(cols[0], cols[1]))
        raise NotImplementedError(
            "xtabs(): 3+ way tables not supported in v1"
        )
    # Weighted form: ``w ~ a (+ b)`` — sum ``w`` per group.
    # Use polars directly (hea's GroupBy is dplyr-shaped).
    base = pl.DataFrame._from_pydf(data._df) if hasattr(data, "_df") else data
    if len(cols) == 1:
        return (
            base.group_by(cols[0]).agg(pl.col(lhs).sum())
            .sort(cols[0])
            .rename({lhs: "n"})
        )
    if len(cols) == 2:
        wide = (
            base.group_by(cols).agg(pl.col(lhs).sum().alias("__w__"))
            .pivot(values="__w__", index=cols[0], on=cols[1])
            .fill_null(0)
            .sort(cols[0])
        )
        label_col = cols[0]
        other_cols = sorted(c for c in wide.columns if c != label_col)
        return wide.select([label_col, *other_cols])
    raise NotImplementedError(
        "xtabs(): 3+ way weighted tables not supported in v1"
    )


def prop_table(tbl, margin=None):
    """R: ``prop.table()`` — convert counts to proportions.

    ``margin=None`` divides by the grand total (default).
    ``margin=1`` divides each row by its sum (row-conditional).
    ``margin=2`` divides each column by its sum (column-conditional).

    Accepts the polars DataFrame produced by :func:`table` /
    :func:`xtabs`, or a plain numpy 2-D array.
    """
    if isinstance(tbl, pl.DataFrame):
        # 1-way table from this module: cols are exactly ['value', 'n'].
        if tbl.columns == ["value", "n"]:
            n = tbl["n"].cast(pl.Float64).to_numpy()
            return tbl.with_columns(pl.Series("n", n / n.sum()))
        # 2-way: first col carries row labels; rest are counts.
        label_col = tbl.columns[0]
        count_cols = tbl.columns[1:]
        mat = tbl.select(count_cols).to_numpy().astype(float)
        mat = _apply_margin_proportion(mat, margin)
        return pl.DataFrame(
            {label_col: tbl[label_col], **{c: mat[:, i] for i, c in enumerate(count_cols)}}
        )
    arr = np.asarray(tbl, dtype=float)
    return _apply_margin_proportion(arr, margin)


def _apply_margin_proportion(mat: np.ndarray, margin) -> np.ndarray:
    if margin is None:
        total = mat.sum()
        if total == 0:
            return mat.copy()
        return mat / total
    if margin == 1:
        rs = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rs > 0, mat / rs, 0.0)
    if margin == 2:
        cs = mat.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cs > 0, mat / cs, 0.0)
    raise ValueError("prop_table(): margin must be None, 1, or 2")


def addmargins(tbl, margin=None):
    """R: ``addmargins()`` — append ``Sum`` rows / columns to a table.

    ``margin=None``: add both a column-of-row-sums and a row-of-column-sums
    (plus the grand total at the corner). ``margin=1``: only the column-sum
    row. ``margin=2``: only the row-sum column.
    """
    if not isinstance(tbl, pl.DataFrame):
        raise NotImplementedError(
            "addmargins(): only polars DataFrame inputs supported in v1"
        )
    if tbl.columns == ["value", "n"]:
        # 1-way: append a "Sum" row, cast `n` to keep types consistent
        n_total = float(tbl["n"].sum())
        n_cast = tbl["n"].cast(pl.Float64)
        return pl.concat([
            tbl.with_columns(n_cast),
            pl.DataFrame({"value": ["Sum"], "n": [n_total]}),
        ])
    label_col = tbl.columns[0]
    count_cols = tbl.columns[1:]
    mat = tbl.select(count_cols).to_numpy().astype(float)
    margins = (1, 2) if margin is None else (margin,)
    # Cast count columns to Float64 so the appended Sum row (float) lines up.
    new_tbl = tbl.with_columns(*(pl.col(c).cast(pl.Float64) for c in count_cols))
    if 2 in margins:
        new_tbl = new_tbl.with_columns(
            pl.Series("Sum", mat.sum(axis=1))
        )
    if 1 in margins:
        sum_row: dict = {label_col: "Sum"}
        col_sums = mat.sum(axis=0)
        for i, c in enumerate(count_cols):
            sum_row[c] = float(col_sums[i])
        if 2 in margins:
            sum_row["Sum"] = float(mat.sum())
        new_tbl = pl.concat([new_tbl, pl.DataFrame([sum_row])])
    return new_tbl
