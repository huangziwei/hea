"""R's base vector helpers — sequence construction, sorting, set ops,
cumulative ops, binning.

* ``seq`` / ``seq_len`` / ``seq_along`` — sequence constructors (0-based;
  diverges from R for ``1:n`` muscle memory — use ``seq(1, n)`` for that).
* ``rev`` / ``sort`` / ``order`` — reverse, sort, permutation indices.
* ``which`` / ``which_max`` / ``which_min`` — boolean → indices (0-based).
* ``cumsum`` / ``cumprod`` / ``cummax`` / ``cummin`` / ``diff``.
* ``unique`` / ``duplicated`` / ``tabulate``.
* ``cut`` / ``findInterval`` — bin a numeric vector to a factor / index.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ._shared import _fmt


def seq(*args, from_=None, to=None, by=None, length_out=None, along_with=None):
    """R: ``seq()`` — flexible sequence constructor.

    The one-argument and ``along_with`` forms are 0-based to match Python
    indexing — different from R's 1-based defaults. The two-argument
    ``seq(from, to)`` form keeps R's inclusive-on-both-ends semantics, so
    explicit ``seq(1, n)`` gives ``[1, 2, …, n]``.

    Call shapes:

    * ``seq(n)`` → ``0, 1, …, n-1`` (Python; for R's ``1:n`` use ``seq(1, n)``)
    * ``seq(from, to)`` → ``from, from+1, …, to`` (inclusive, R-faithful)
    * ``seq(from, to, by=step)`` → step-spaced, inclusive
    * ``seq(from, to, length_out=n)`` → ``n`` evenly spaced
    * ``seq(along_with=x)`` → ``0, 1, …, len(x)-1``

    ``from_=`` / ``to=`` accept R's named forms after translation — R's
    ``seq(from = 1, to = 10)`` becomes ``seq(from_=1, to=10)`` (Python
    keyword conflict on ``from``).
    """
    if from_ is not None or to is not None:
        # R's named form: fold into positional slots.
        if args:
            raise TypeError(
                "seq(): pass both endpoints positionally OR via from_= / to=, not both"
            )
        if from_ is None or to is None:
            raise TypeError("seq(): from_= and to= must both be supplied together")
        args = (from_, to)
    if along_with is not None:
        return np.arange(len(along_with))
    if length_out is not None:
        if len(args) == 0:
            return np.arange(int(length_out))
        if len(args) == 1:
            return np.linspace(0, args[0], int(length_out))
        return np.linspace(args[0], args[1], int(length_out))
    if len(args) == 0:
        raise ValueError("seq(): need at least one positional argument")
    if len(args) == 1:
        return np.arange(int(args[0]))
    start, stop = args[0], args[1]
    step = by if by is not None else (1 if stop >= start else -1)
    n_steps = int(np.floor((stop - start) / step + 1e-10)) + 1
    return start + np.arange(n_steps) * step


def seq_len(n):
    """R: ``seq_len(n)`` → ``0, 1, …, n-1`` (0-based; differs from R's ``1:n``)."""
    return np.arange(int(n))


def seq_along(x):
    """R: ``seq_along(x)`` → ``0, 1, …, len(x)-1`` (0-based; differs from R)."""
    return np.arange(len(x))


def rev(x):
    """R: reverse element order."""
    if isinstance(x, pl.Expr):
        return x.reverse()
    if isinstance(x, (pl.Series, pl.DataFrame)):
        return x.reverse()
    if isinstance(x, list):
        return x[::-1]
    return np.asarray(x)[::-1]


def sort(x, decreasing=False):
    """R: sort. ``decreasing=True`` matches R's keyword."""
    if isinstance(x, pl.Expr):
        return x.sort(descending=decreasing)
    if isinstance(x, pl.Series):
        return x.sort(descending=decreasing)
    if isinstance(x, pl.DataFrame):
        raise TypeError(
            "sort(DataFrame): pass a column name to .sort() instead"
        )
    arr = np.sort(np.asarray(x))
    return arr[::-1] if decreasing else arr


def order(*args, decreasing=False):
    """R: ``order()`` — permutation that sorts. Multi-key supported.

    Returns 0-based indices (Python convention; R returns 1-based).
    """
    if len(args) == 1:
        idx = np.argsort(np.asarray(args[0]), kind="stable")
    else:
        keys = [np.asarray(a) for a in args]
        idx = np.lexsort(list(reversed(keys)))
    return idx[::-1].copy() if decreasing else idx


def which(cond):
    """R: ``which()`` — indices where cond is True. 0-based."""
    return np.flatnonzero(np.asarray(cond))


def which_max(x):
    """R: ``which.max()`` — index of first max. 0-based.

    Dispatches: ``pl.Expr`` → scalar ``pl.Expr`` (broadcasts inside
    ``mutate``); ``pl.Series`` → Python int; list / ndarray → int.
    """
    if isinstance(x, pl.Expr):
        return x.arg_max()
    if isinstance(x, pl.Series):
        return x.arg_max()
    return int(np.argmax(np.asarray(x)))


def which_min(x):
    """R: ``which.min()`` — index of first min. 0-based. Dispatches like
    :func:`which_max`.
    """
    if isinstance(x, pl.Expr):
        return x.arg_min()
    if isinstance(x, pl.Series):
        return x.arg_min()
    return int(np.argmin(np.asarray(x)))


def cumsum(x):
    """R: ``cumsum()`` — cumulative sum.

    Dispatches like :func:`rank`: ``pl.Expr`` → ``pl.Expr`` (polars'
    ``cum_sum`` method, so it composes inside ``mutate``); ``pl.Series``
    → ``pl.Series``; list / tuple / ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        return x.cum_sum()
    if isinstance(x, pl.Series):
        return x.cum_sum()
    return np.cumsum(np.asarray(x))


def cumprod(x):
    """R: ``cumprod()`` — cumulative product. Dispatches like :func:`cumsum`."""
    if isinstance(x, pl.Expr):
        return x.cum_prod()
    if isinstance(x, pl.Series):
        return x.cum_prod()
    return np.cumprod(np.asarray(x))


def cummax(x):
    """R: ``cummax()`` — cumulative max. Dispatches like :func:`cumsum`."""
    if isinstance(x, pl.Expr):
        return x.cum_max()
    if isinstance(x, pl.Series):
        return x.cum_max()
    return np.maximum.accumulate(np.asarray(x))


def cummin(x):
    """R: ``cummin()`` — cumulative min. Dispatches like :func:`cumsum`."""
    if isinstance(x, pl.Expr):
        return x.cum_min()
    if isinstance(x, pl.Series):
        return x.cum_min()
    return np.minimum.accumulate(np.asarray(x))


def diff(x, lag=1, differences=1):
    """R: ``diff()`` — lagged and iterated differences."""
    arr = np.asarray(x)
    for _ in range(int(differences)):
        arr = arr[lag:] - arr[:-lag]
    return arr


def unique(x):
    """R: unique values, preserving order of first occurrence.

    Note: in an Expr context, the result has *shorter* length than the
    input — so it can't be used as a column inside ``mutate``. Use it
    inside ``summarize`` or assign to a separate frame instead.
    """
    if isinstance(x, pl.Expr):
        return x.unique(maintain_order=True)
    if isinstance(x, (pl.Series, pl.DataFrame)):
        return x.unique(maintain_order=True)
    arr = np.asarray(x)
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def duplicated(x):
    """R: ``duplicated()`` — True for the 2nd+ occurrence of each value."""
    if isinstance(x, pl.Expr):
        return ~x.is_first_distinct()
    if isinstance(x, pl.Series):
        return ~x.is_first_distinct()
    arr = np.asarray(x)
    seen: set = set()
    out = np.zeros(arr.shape[0] if arr.ndim else 1, dtype=bool)
    for i, v in enumerate(arr.tolist() if arr.ndim else [arr.item()]):
        if v in seen:
            out[i] = True
        else:
            seen.add(v)
    return out


def tabulate(x, nbins=None):
    """R: ``tabulate()`` — counts of integer values in ``0..nbins-1``.

    0-based (Python convention; R / dplyr's ``tabulate()`` is 1-based):
    ``tabulate([0, 1, 1, 2])`` → ``[1, 2, 1]``.
    """
    arr = np.asarray(x, dtype=int)
    if nbins is None:
        nbins = int(arr.max()) + 1 if arr.size else 0
    if nbins == 0:
        return np.zeros(0, dtype=int)
    return np.bincount(arr, minlength=int(nbins))[:int(nbins)]


def cut(x, breaks, *, labels=None, right=True, include_lowest=False):
    """R: ``cut()`` — bin a numeric vector into intervals (a factor).

    Parameters
    ----------
    x : array-like
        Numeric vector to bin.
    breaks : int or array-like
        Number of equal-width bins (R's scalar form), or an explicit
        strictly-increasing sequence of cut points.
    labels : list, False, or None
        ``None`` (default): auto-generate ``"(a,b]"`` / ``"[a,b)"``-style
        labels. A list: custom labels (length must equal ``len(breaks)-1``).
        ``False``: return integer codes instead of a factor (0-based,
        Python convention; ``NaN`` for out-of-range. R / dplyr's ``cut()``
        emits 1-based codes).
    right : bool, default True
        If True, bins are right-closed ``(a, b]`` (R's default). If False,
        left-closed ``[a, b)``.
    include_lowest : bool, default False
        If True, include the boundary value in the lowest bin (right=True)
        or in the highest bin (right=False), matching R's ``include.lowest``.

    Returns
    -------
    pl.Series
        ``pl.Enum`` factor with the bin labels, or a numpy ``float64`` array
        of 0-based integer codes when ``labels=False``.
    """
    arr = np.asarray(x, dtype=float)
    if isinstance(breaks, (int, np.integer)) and not isinstance(breaks, bool):
        n = int(breaks)
        if n < 1:
            raise ValueError("cut(): need at least 1 bin")
        if arr.size == 0:
            raise ValueError("cut(): empty input with scalar breaks")
        rng = float(arr.max()) - float(arr.min())
        eps = 0.001 * rng if rng > 0 else 0.001
        breaks = np.linspace(arr.min() - eps, arr.max() + eps, n + 1)
    breaks_arr = np.asarray(breaks, dtype=float)
    if not (np.diff(breaks_arr) > 0).all():
        raise ValueError("cut(): breaks must be strictly increasing")
    n_bins = len(breaks_arr) - 1
    if n_bins < 1:
        raise ValueError("cut(): need at least 2 break points")

    idx = np.digitize(arr, breaks_arr, right=right) - 1
    if include_lowest:
        if right:
            idx = np.where(arr == breaks_arr[0], 0, idx)
        else:
            idx = np.where(arr == breaks_arr[-1], n_bins - 1, idx)
    out_of_range = (idx < 0) | (idx >= n_bins)

    if labels is False:
        codes = idx.astype(float)
        codes[out_of_range] = np.nan
        return codes

    if labels is None:
        if right:
            lab_list = [
                f"({_fmt(breaks_arr[i])},{_fmt(breaks_arr[i + 1])}]"
                for i in range(n_bins)
            ]
            if include_lowest:
                lab_list[0] = (
                    f"[{_fmt(breaks_arr[0])},{_fmt(breaks_arr[1])}]"
                )
        else:
            lab_list = [
                f"[{_fmt(breaks_arr[i])},{_fmt(breaks_arr[i + 1])})"
                for i in range(n_bins)
            ]
            if include_lowest:
                lab_list[-1] = (
                    f"[{_fmt(breaks_arr[-2])},{_fmt(breaks_arr[-1])}]"
                )
    else:
        lab_list = [str(la) for la in labels]
        if len(lab_list) != n_bins:
            raise ValueError(
                f"cut(): {len(lab_list)} labels but {n_bins} bins"
            )

    result = [
        None if out_of_range[i] else lab_list[int(idx[i])]
        for i in range(len(arr))
    ]
    return pl.Series(result, dtype=pl.Utf8).cast(pl.Enum(lab_list))


def findInterval(
    x,
    vec,
    *,
    rightmost_closed: bool = False,
    all_inside: bool = False,
    left_open: bool = False,
):
    """R: ``findInterval()`` — for each x, the index of its enclosing
    interval in a sorted ``vec``.

    Returns ``i`` in ``0..len(vec)`` (same convention as R, treating the
    return as an index into ``vec``):

    * ``i = 0``        → x is below ``vec[0]``
    * ``vec[i-1] ≤ x < vec[i]`` (default, ``left_open=False``)
    * ``i = len(vec)`` → x is at or above ``vec[-1]``
    """
    arr = np.asarray(x, dtype=float)
    vec_arr = np.asarray(vec, dtype=float)
    if not (np.diff(vec_arr) >= 0).all():
        raise ValueError("findInterval(): vec must be non-decreasing")
    side = "left" if left_open else "right"
    idx = np.searchsorted(vec_arr, arr, side=side)
    if rightmost_closed:
        idx = np.where(arr == vec_arr[-1], len(vec_arr) - 1, idx)
    if all_inside:
        idx = np.clip(idx, 1, len(vec_arr) - 1)
    return idx
