"""R's base matrix / frame utilities: row/col sums and means, ``apply``,
``rbind`` / ``cbind`` / ``sweep`` / ``expand.grid`` / ``matrix`` / ``rep``,
plus the renamed ``R_range`` / ``R_round`` (Python-builtin clashes).
"""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from ._shared import NamedVector


def _to_2d(x) -> np.ndarray:
    """Promote DataFrame/list/array to a 2-D numpy array."""
    if isinstance(x, pl.DataFrame):
        return x.to_numpy()
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def rowSums(x, na_rm=False):
    """R: per-row sum. Returns a 1-D numpy array."""
    arr = _to_2d(x)
    if na_rm:
        return np.nansum(arr, axis=1)
    return arr.sum(axis=1)


def colSums(x, na_rm=False):
    """R: per-column sum. Returns a 1-D numpy array."""
    arr = _to_2d(x)
    if na_rm:
        return np.nansum(arr, axis=0)
    return arr.sum(axis=0)


def rowMeans(x, na_rm=False):
    """R: per-row mean. Returns a 1-D numpy array."""
    arr = _to_2d(x)
    if na_rm:
        return np.nanmean(arr, axis=1)
    return arr.mean(axis=1)


def colMeans(x, na_rm=False):
    """R: per-column mean. Returns a 1-D numpy array."""
    arr = _to_2d(x)
    if na_rm:
        return np.nanmean(arr, axis=0)
    return arr.mean(axis=0)


def apply(X, MARGIN, FUN, *args, **kwargs):
    """R: ``apply(X, MARGIN, FUN, ...)``.

    MARGIN=1 → over rows, MARGIN=2 → over columns. Returns numpy array
    of FUN-results stacked; loses R's name attribute (Python has no
    named-vector primitive).
    """
    arr = _to_2d(X)
    axis = 1 if MARGIN == 1 else 0  # iterate along the *other* axis
    results = []
    if MARGIN == 2:
        for j in range(arr.shape[1]):
            results.append(FUN(arr[:, j], *args, **kwargs))
    else:
        for i in range(arr.shape[0]):
            results.append(FUN(arr[i, :], *args, **kwargs))
    return np.asarray(results)


def rbind(*args):
    """R: row-bind. Concatenates 1-D vectors or 2-D arrays vertically."""
    if all(isinstance(a, pl.DataFrame) for a in args):
        return pl.concat(args, how="vertical")
    arrs = [np.atleast_2d(np.asarray(a)) for a in args]
    return np.vstack(arrs)


def cbind(*args):
    """R: column-bind. Concatenates vectors as columns of a matrix."""
    if all(isinstance(a, pl.DataFrame) for a in args):
        return pl.concat(args, how="horizontal")
    arrs = []
    for a in args:
        arr = np.asarray(a)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrs.append(arr)
    # Broadcast scalars to longest column
    max_rows = max(a.shape[0] for a in arrs)
    arrs = [np.broadcast_to(a, (max_rows, a.shape[1])) if a.shape[0] == 1 else a for a in arrs]
    return np.hstack(arrs)


def sweep(x, MARGIN, STATS, FUN="-"):
    """R: sweep out a summary statistic along a margin.

    ``MARGIN=1`` sweeps along rows, ``MARGIN=2`` along columns. ``FUN``
    is the operator name (``"-"``, ``"+"``, ``"*"``, ``"/"``).
    """
    arr = _to_2d(x).astype(float)
    stats = np.asarray(STATS, dtype=float).ravel()
    if MARGIN == 2:
        # broadcast along axis 0 (rows broadcast, columns aligned)
        op_arr = stats[np.newaxis, :]
    else:
        op_arr = stats[:, np.newaxis]
    ops = {"-": np.subtract, "+": np.add, "*": np.multiply, "/": np.divide}
    op = ops.get(FUN, FUN if callable(FUN) else ops["-"])
    return op(arr, op_arr)


def expand_grid(**kwargs):
    """R: ``expand.grid(...)``. Cartesian product of named inputs.

    Returns a polars DataFrame whose columns are the named args. R's
    column ordering: first arg varies fastest; we match that.
    """
    if not kwargs:
        return pl.DataFrame()
    keys = list(kwargs.keys())
    values = []
    for v in kwargs.values():
        if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
            values.append([v])
        else:
            values.append(list(v))
    # R's expand.grid: first variable varies fastest. itertools.product
    # varies the *last* fastest, so reverse, product, reverse again.
    rev_values = list(reversed(values))
    rows = list(itertools.product(*rev_values))
    rows = [tuple(reversed(r)) for r in rows]
    cols = {k: [r[i] for r in rows] for i, k in enumerate(keys)}
    return pl.DataFrame(cols)


def R_range(x, na_rm=False):
    """R: ``range(x)`` — ``[min(x), max(x)]``. Named ``R_range`` to avoid
    colliding with Python's builtin ``range``; the translator routes R's
    ``range()`` here via the FUNCTION_TABLE registry.
    """
    if isinstance(x, (pl.Expr, pl.Series)):
        return [x.min(), x.max()]
    arr = np.asarray(x, dtype=float)
    if na_rm:
        arr = arr[~np.isnan(arr)]
    return [float(arr.min()), float(arr.max())]


def _flatten_to_values(x) -> np.ndarray:
    """Flatten an R-shaped argument to a 1-D numpy array.

    Accepts a scalar, list/tuple (possibly with nested vectors), numpy
    array, ``pl.Series``, or :class:`hea.NamedVector`. The list case is
    what the translator emits for R's ``c(scalar, vec, vec2)`` — a
    Python list with mixed scalar + vector entries that R would have
    flattened via ``c()``.
    """

    if isinstance(x, NamedVector):
        return x.values
    if isinstance(x, pl.Series):
        return x.to_numpy().ravel()
    if isinstance(x, np.ndarray):
        return x.ravel()
    if isinstance(x, (list, tuple)):
        parts: list[np.ndarray] = []
        for item in x:
            if isinstance(item, NamedVector):
                parts.append(item.values)
            elif isinstance(item, pl.Series):
                parts.append(item.to_numpy().ravel())
            elif isinstance(item, np.ndarray):
                parts.append(item.ravel())
            elif isinstance(item, (list, tuple)):
                parts.append(_flatten_to_values(item))
            else:
                parts.append(np.asarray([item]))
        return np.concatenate(parts) if parts else np.asarray([])
    return np.asarray([x])


def rep(x, times=1, each=1, length_out=None):
    """R: ``rep(x, times, each, length.out)`` — repeat values.

    Semantics:

    - ``each=N`` repeats each element ``N`` times in place
      (``rep(c(1,2), each=3) → c(1,1,1,2,2,2)``).
    - ``times=N`` repeats the (already-each-expanded) vector ``N`` times
      (``rep(c(1,2), 3) → c(1,2,1,2,1,2)``).
    - ``length.out`` truncates / cycles to that length, overriding
      ``times`` when set.

    Flattens any list-with-vector input (the translator emits R's
    ``c(scalar, vec)`` as a Python list, so ``rep([0, named_vec], …)``
    needs to be equivalent to R's ``rep(c(0, named_vec), …)``).
    """
    arr = _flatten_to_values(x)
    arr = np.repeat(arr, int(each))
    out = np.tile(arr, int(times))
    if length_out is not None:
        n = int(length_out)
        if n <= len(out):
            return out[:n]
        reps = -(-n // max(len(out), 1))
        return np.tile(out, reps)[:n]
    return out


def R_round(x, digits=0):
    """R: ``round(x, digits)`` — vectorized over containers.

    Named ``R_round`` to avoid colliding with Python's builtin ``round``;
    the translator routes R's ``round()`` here via FUNCTION_TABLE.
    Handles dict (named-vector), ndarray, Series/Expr, scalar.
    """
    d = int(digits)
    if isinstance(x, dict):
        return {k: round(v, d) for k, v in x.items()}
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.round(d)
    if isinstance(x, pl.DataFrame):
        return x.with_columns(pl.col(pl.Float64, pl.Float32).round(d))
    arr = np.asarray(x)
    if arr.shape == ():
        return round(float(arr), d)
    return np.round(arr, d)


def matrix(data, nrow=None, ncol=None, byrow=False):
    """R: ``matrix(data, nrow, ncol, byrow=FALSE)``.

    Reshapes ``data`` to (nrow × ncol). Either ``nrow`` or ``ncol`` may
    be inferred from the other. R fills column-major (``byrow=FALSE``);
    we match that default.

    ``None`` / ``NA`` data fills with ``float64`` NaN — R's ``matrix(NA,
    r, c)`` creates a numeric matrix that downstream assignments mutate
    in-place. Object dtype (what ``np.asarray(None)`` gives) would
    break later integer indexing.
    """
    if data is None:
        arr = np.array([np.nan], dtype=float)
    else:
        arr = np.asarray(data).ravel()
        if arr.dtype == object:
            # Mixed / None entries — coerce to float with NaN for None.
            try:
                arr = np.asarray(
                    [np.nan if v is None else v for v in arr.tolist()],
                    dtype=float,
                )
            except (TypeError, ValueError):
                pass  # leave as object; downstream may want strings/factors
    if nrow is None and ncol is None:
        return arr.reshape(-1, 1)
    if nrow is None:
        nrow = -(-arr.size // ncol)
    if ncol is None:
        ncol = -(-arr.size // nrow)
    if arr.size < nrow * ncol:
        # R recycles values; numpy reshape doesn't — tile to fill.
        reps = -(-(nrow * ncol) // arr.size)
        arr = np.tile(arr, reps)[: nrow * ncol]
    elif arr.size > nrow * ncol:
        arr = arr[: nrow * ncol]
    order = "C" if byrow else "F"
    return arr.reshape(nrow, ncol, order=order)
