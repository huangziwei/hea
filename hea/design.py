"""Formula → fitting-ready design bundle.

The orchestration layer that sits between ``hea.formula`` (parse trees,
term algebra, basis construction) and the model classes. Given a formula
string and a polars DataFrame, ``prepare_design``:

1. Parses the formula, expands the RHS into terms, materializes the
   fixed-effect design matrix.
2. Evaluates the LHS — which may be a bare name or a small expression
   (``log(y)``, ``y^0.25``, ``I(y/100)``, etc.).
3. NA-omits rows referenced by either side, mirroring R's
   ``na.action = na.omit``.
4. Returns a ``Design`` bundle that downstream models specialize as
   they see fit.

User-facing data prep — ``data()`` and ``factor()`` — lives in
``hea.data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl

from .formula import (
    BinOp,
    Call,
    ExpandedFormula,
    Literal,
    Name,
    Paren,
    UnaryOp,
    deparse,
    expand,
    materialize,
    parse,
    referenced_columns,
)

__all__ = [
    "Design",
    "prepare_design",
    "normalize_data",
    "is_matrix_col",
    "matrix_to_2d",
    "long_form_view",
]


# ---------------------------------------------------------------------------
# Matrix-argument carrier (mgcv summation convention support)
# ---------------------------------------------------------------------------
#
# mgcv lets you pass an n×m matrix as a `s()` / `te()` argument; the smooth's
# linear-predictor contribution at row i becomes ``Σ_k f(X[i, k], …)`` —
# integrating the smooth over the second dimension. Wood §7.4.1's chicago
# distributed-lag model (`te(pm10, lag, k=c(10,5))` with both as n×6) is the
# canonical example.
#
# Carrier: a polars ``Array(Float64, m)`` column. The user can either build
# the DataFrame directly with such columns or pass a dict of mixed 1-D / 2-D
# arrays to ``normalize_data`` which packs them. ``Array`` (fixed-shape) is
# preferred over ``List`` so ``.to_numpy()`` returns a contiguous (n, m)
# float matrix without iterating row-by-row.


def is_matrix_col(s: pl.Series) -> bool:
    """True if ``s`` is a fixed-shape Array(Float64, m) column."""
    dt = s.dtype
    return isinstance(dt, pl.Array) and dt.inner == pl.Float64


def matrix_to_2d(s: pl.Series) -> np.ndarray:
    """Materialize an ``Array(Float64, m)`` series as an (n, m) ndarray."""
    if not is_matrix_col(s):
        raise TypeError(f"column {s.name!r} is not a matrix column (got dtype {s.dtype})")
    return s.to_numpy().astype(float)


def normalize_data(
    data: pl.DataFrame | Mapping[str, np.ndarray | list | pl.Series],
) -> pl.DataFrame:
    """Convert a polars DataFrame or a dict of arrays into a polars
    DataFrame where 2-D ndarray entries become ``Array(Float64, m)``
    columns (matrix-arg carriers) and 1-D entries become ``Float64``.

    Mirrors mgcv's ``data=list(...)`` convention where matrix entries
    encode the summation-convention dimension. Pass-through if ``data``
    is already a polars DataFrame — caller is responsible for any matrix
    columns being typed as ``Array(Float64, m)``.
    """
    if isinstance(data, pl.DataFrame):
        return data
    if not isinstance(data, Mapping):
        raise TypeError(
            f"data must be a polars DataFrame or a mapping of arrays, "
            f"got {type(data).__name__}"
        )

    n_ref: int | None = None
    series: dict[str, pl.Series] = {}
    for name, val in data.items():
        if isinstance(val, pl.Series):
            arr = val
            n_here = arr.len()
            series[name] = arr
        else:
            arr_np = np.asarray(val)
            if arr_np.ndim == 1:
                n_here = arr_np.shape[0]
                series[name] = pl.Series(name, arr_np)
            elif arr_np.ndim == 2:
                n_here, m = arr_np.shape
                series[name] = pl.Series(
                    name,
                    arr_np.astype(float),
                    dtype=pl.Array(pl.Float64, m),
                )
            else:
                raise ValueError(
                    f"data column {name!r}: ndim must be 1 or 2, "
                    f"got ndim={arr_np.ndim}"
                )
        if n_ref is None:
            n_ref = n_here
        elif n_here != n_ref:
            raise ValueError(
                f"data column {name!r} has length {n_here}, "
                f"expected {n_ref} (mismatched row counts across columns)"
            )
    return pl.DataFrame(series)


def long_form_view(
    data: pl.DataFrame, matrix_vars: list[str],
) -> tuple[pl.DataFrame, int, int]:
    """Expand a DataFrame to long form for matrix-arg evaluation.

    For each row i of the original (n rows) and each k in 0..m-1:

      * matrix variables: the long-form column at position ``i*m + k``
        carries ``data[var][i, k]``.
      * scalar variables: repeated m times — long-form position ``i*m + k``
        carries ``data[var][i]``.

    Returns ``(long_df, n, m)``. All ``matrix_vars`` must be matrix
    columns of identical width m; the function raises otherwise.
    """
    if not matrix_vars:
        raise ValueError("long_form_view requires at least one matrix variable")
    arrs: list[np.ndarray] = []
    widths: set[int] = set()
    for v in matrix_vars:
        a = matrix_to_2d(data[v])
        arrs.append(a)
        widths.add(a.shape[1])
    if len(widths) != 1:
        raise ValueError(
            f"matrix variables {matrix_vars} have inconsistent widths "
            f"{sorted(widths)} — summation convention requires identical m"
        )
    n = arrs[0].shape[0]
    m = widths.pop()
    long_cols: dict[str, np.ndarray] = {}
    for v, a in zip(matrix_vars, arrs):
        long_cols[v] = a.reshape(n * m).astype(float)
    matrix_set = set(matrix_vars)
    for col in data.columns:
        if col in matrix_set:
            continue
        s = data[col]
        if is_matrix_col(s):
            # Other matrix cols not involved in this smooth — flatten too,
            # so the long DataFrame has no ragged columns. Most smooths
            # only see their own variables, so this rarely fires.
            a = matrix_to_2d(s)
            if a.shape != (n, m):
                # Different m → can't co-exist in one long view; just
                # repeat the first column. Will only matter if some
                # later smooth tries to use this var simultaneously
                # under the same long view, which we don't do.
                long_cols[col] = np.repeat(a[:, 0], m)
            else:
                long_cols[col] = a.reshape(n * m)
        else:
            long_cols[col] = np.repeat(s.to_numpy(), m)
    return pl.DataFrame(long_cols), n, m


@dataclass(slots=True)
class Design:
    """Bundle returned by ``prepare_design``.

    Attributes
    ----------
    expanded : ExpandedFormula
        Output of ``formula.expand`` for the parsed formula. Pass this
        to downstream materializers (``materialize_bars`` for lme,
        ``materialize_smooths`` for gam) so they share the same parse.
    data : polars.DataFrame
        Input data with rows dropped where the response or any
        RHS-referenced column is NA. Row positions align with ``X``
        and ``y``.
    X : polars.DataFrame
        Materialized fixed-effect design with R-canonical column names.
    y : polars.Series
        Response column, with NA rows dropped. For non-trivial LHS
        expressions, holds the *evaluated* response (e.g. log(y));
        the Series name is the deparsed LHS source.
    response : str
        Response label — bare column name for ``y ~ ...`` formulas,
        deparsed LHS source (e.g. ``"medFPQ^0.25"``) otherwise.
    """
    expanded: ExpandedFormula
    data: pl.DataFrame
    X: pl.DataFrame
    y: pl.Series
    response: str


# LHS function table — maps R-side function names to a polars-expr builder.
# Mirrors what mgcv/base R accept on a formula LHS: arithmetic via
# `_eval_lhs_expr` (UnaryOp/BinOp), plus these elementary transforms.
_LHS_FUNCS: dict[str, "callable"] = {
    "log":   lambda e: e.log(),
    "log2":  lambda e: e.log(2.0),
    "log10": lambda e: e.log10(),
    "exp":   lambda e: e.exp(),
    "sqrt":  lambda e: e.sqrt(),
    "abs":   lambda e: e.abs(),
}


def _lhs_referenced_cols(node, columns: set[str]) -> set[str]:
    """Walk an LHS AST and collect ``Name`` idents that exist in ``data``.

    Used by ``prepare_design`` to decide which columns to NA-drop on
    before evaluating the response. Names that don't match a column name
    are silently skipped — they'll error later in ``_eval_lhs_expr``.
    """
    out: set[str] = set()
    def visit(n):
        if isinstance(n, Name):
            if n.ident in columns:
                out.add(n.ident)
            return
        if isinstance(n, Literal):
            return
        if isinstance(n, Paren):
            visit(n.expr); return
        if isinstance(n, UnaryOp):
            visit(n.operand); return
        if isinstance(n, BinOp):
            visit(n.left); visit(n.right); return
        if isinstance(n, Call):
            for a in n.args:
                visit(a)
            for v in n.kwargs.values():
                visit(v)
            return
        # Anything else (Dot, Empty, Subscript, …) shouldn't appear on a
        # response LHS — let _eval_lhs_expr raise the clearer error.
    visit(node)
    return out


def _eval_lhs_expr(node, columns: set[str]) -> pl.Expr:
    """Recursively evaluate an LHS AST as a polars expression.

    Supported:
      * ``Name``    → ``pl.col(name)``
      * numeric ``Literal``
      * ``+``/``-`` (unary), ``+``/``-``/``*``/``/``/``^``
      * ``I(expr)`` (R's "as is" — just unwraps)
      * one-arg numeric calls listed in ``_LHS_FUNCS``
      * parens

    Multi-column responses (``cbind(succ, fail)``) and arbitrary user
    functions are not yet supported.
    """
    if isinstance(node, Name):
        if node.ident not in columns:
            raise KeyError(f"LHS references unknown column {node.ident!r}")
        return pl.col(node.ident)
    if isinstance(node, Literal):
        if node.kind != "num":
            raise NotImplementedError(
                f"LHS literal kind {node.kind!r} not supported"
            )
        return pl.lit(float(node.value))
    if isinstance(node, Paren):
        return _eval_lhs_expr(node.expr, columns)
    if isinstance(node, UnaryOp):
        e = _eval_lhs_expr(node.operand, columns)
        if node.op == "-":
            return -e
        if node.op == "+":
            return e
        raise NotImplementedError(f"LHS unary op {node.op!r} not supported")
    if isinstance(node, BinOp):
        L = _eval_lhs_expr(node.left, columns)
        R = _eval_lhs_expr(node.right, columns)
        if node.op == "+":
            return L + R
        if node.op == "-":
            return L - R
        if node.op == "*":
            return L * R
        if node.op == "/":
            return L / R
        if node.op == "^":
            return L ** R
        raise NotImplementedError(f"LHS binary op {node.op!r} not supported")
    if isinstance(node, Call):
        if node.kwargs:
            raise NotImplementedError(
                f"LHS call {node.fn}() with kwargs is not supported"
            )
        if node.fn == "I":
            if len(node.args) != 1:
                raise ValueError("I() takes exactly one argument")
            return _eval_lhs_expr(node.args[0], columns)
        if node.fn == "cbind":
            raise NotImplementedError(
                "multi-column response cbind() (e.g. for binomial trials) "
                "is not yet supported on the LHS"
            )
        if node.fn in _LHS_FUNCS:
            if len(node.args) != 1:
                raise ValueError(f"{node.fn}() takes exactly one argument on LHS")
            return _LHS_FUNCS[node.fn](_eval_lhs_expr(node.args[0], columns))
        raise NotImplementedError(
            f"LHS function {node.fn}() not supported "
            f"(allowed: I, {', '.join(sorted(_LHS_FUNCS))})"
        )
    raise NotImplementedError(
        f"LHS contains unsupported node {type(node).__name__}"
    )


def _na_mask_with_matrix_cols(
    data: pl.DataFrame, na_cols: set[str],
) -> np.ndarray:
    """Boolean keep-mask for rows with no NA across ``na_cols``.

    Polars' ``drop_nulls`` only drops rows where the *cell* is null; for
    a matrix-typed (``Array(Float64, m)``) column, NaN *inside* the
    array is kept. mgcv's ``na.omit`` equivalent drops a row if any
    matrix entry is NaN, so we walk matrix columns row-wise.
    """
    n = data.height
    keep = np.ones(n, dtype=bool)
    for col in na_cols:
        s = data[col]
        if is_matrix_col(s):
            a = matrix_to_2d(s)
            keep &= ~np.any(np.isnan(a), axis=1)
        else:
            arr = s.to_numpy()
            if np.issubdtype(arr.dtype, np.floating):
                keep &= ~np.isnan(arr)
            # null check for non-float columns (categoricals etc.)
            null_mask = s.is_null().to_numpy()
            keep &= ~null_mask
    return keep


def prepare_design(
    formula: str,
    data: pl.DataFrame | Mapping[str, np.ndarray | list | pl.Series],
) -> Design:
    """Parse a formula, expand, and materialize the fixed-effect design.

    NA-omit policy matches R's ``na.action = na.omit``: rows with NA in
    the response or in any RHS-referenced column are dropped before the
    design matrix is built. All three outputs (``Design.data``,
    ``Design.X``, ``Design.y``) share the same row ordering.

    The LHS may be either a bare column name or a small expression
    R/mgcv accepts: arithmetic (``y/100``, ``y^0.25``), unary minus,
    one-arg transforms (``log``, ``log2``, ``log10``, ``exp``,
    ``sqrt``, ``abs``), and ``I(expr)``. The deparsed LHS becomes
    the response label (``Design.response``) so downstream printers
    can show e.g. ``medFPQ^0.25`` rather than a placeholder name.

    ``data`` may be a polars DataFrame or a mapping of name → 1-D / 2-D
    ndarray; 2-D entries become matrix columns (``Array(Float64, m)``),
    enabling mgcv's matrix-argument summation convention for ``s()`` and
    ``te()`` smooths (Wood §7.4.1's distributed-lag models).
    """
    data = normalize_data(data)
    f_parsed = parse(formula)
    if f_parsed.lhs is None:
        raise NotImplementedError("formula must have a response (LHS)")

    columns = set(data.columns)
    if isinstance(f_parsed.lhs, Name):
        response_label = f_parsed.lhs.ident
        lhs_cols = {f_parsed.lhs.ident} & columns
    else:
        response_label = deparse(f_parsed.lhs)
        lhs_cols = _lhs_referenced_cols(f_parsed.lhs, columns)

    expanded = expand(f_parsed, data_columns=list(data.columns))

    na_cols = (referenced_columns(expanded) | lhs_cols) & columns
    if na_cols:
        # Custom NA mask so that NaN inside ``Array(Float64, m)`` matrix
        # columns triggers a row drop (polars' drop_nulls keeps these).
        keep = _na_mask_with_matrix_cols(data, na_cols)
        if not keep.all():
            data_clean = data.filter(pl.Series(keep))
        else:
            data_clean = data
    else:
        data_clean = data

    if isinstance(f_parsed.lhs, Name):
        y = data_clean[f_parsed.lhs.ident]
    else:
        # Evaluate the LHS expression over the cleaned frame and tag
        # the resulting Series with the deparsed label so consumers
        # (stats printers, residual formatters) see the original text.
        y = data_clean.select(
            _eval_lhs_expr(f_parsed.lhs, columns).alias(response_label)
        )[response_label]

    X = materialize(expanded, data_clean)
    return Design(expanded=expanded, data=data_clean, X=X, y=y, response=response_label)
