"""Row-context tidyverse helpers — anything that looks at neighbors,
position, or a running summary.

* dplyr rank family — ``row_number`` / ``min_rank`` / ``dense_rank`` /
  ``percent_rank`` / ``cume_dist`` / ``ntile``.
* dplyr window helpers — ``lag`` / ``lead`` / ``between`` / ``na_if`` /
  ``near``.
* dplyr cumulative ops — ``cummean`` / ``cumall`` / ``cumany``.
* dplyr positional pickers — ``first`` / ``last`` / ``nth`` (shadow
  polars' column-selector versions; this surface is dplyr's element
  picker), plus ``consecutive_id`` (run-length identifier).

Every function dispatches on input type — ``pl.Expr`` in / ``pl.Expr``
out for use inside ``mutate`` / ``summarize``; ``pl.Series`` /
list / ndarray in eager Python code.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as _sps


# ---- shared helpers (rank + sort dispatch) -------------------------

def _as_array(x) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(float)
    return np.asarray(x, dtype=float)


def _rankdata_with_nan(arr: np.ndarray, method: str) -> np.ndarray:
    """``scipy.stats.rankdata`` but propagating NaN → NaN (R / dplyr
    convention)."""
    out = np.full(arr.shape, np.nan)
    finite = ~np.isnan(arr)
    if finite.any():
        out[finite] = _sps.rankdata(arr[finite], method=method)
    return out


def _eager_rank_out(x, arr: np.ndarray):
    """Wrap an ndarray rank result based on the original input type.

    For Python list / tuple input, return a :class:`pl.Series` with NaN
    converted to null — so ``mutate(rn=min_rank(x))`` ends up with a
    proper polars null column instead of a literal NaN value. For
    ndarray input, return the ndarray unchanged (preserves the
    lm/Wilcoxon contract used by :func:`hea.R.rank` / :func:`hea.R.signed_rank`).
    """
    if isinstance(x, (list, tuple)):
        return pl.Series(arr, nan_to_null=True)
    return arr


# ---- dplyr rank family ----------------------------------------------

def row_number(x=None):
    """dplyr's ``row_number()`` — 0-based row position, or ordinal rank.

    Two call shapes:

    * ``row_number()`` (no args) returns the 0-based row position as a
      polars expression, suitable for use inside ``mutate()`` /
      ``select()``. (R / dplyr's ``row_number()`` is 1-based; hea
      follows Python indexing.)
    * ``row_number(x)`` returns the 0-based ordinal rank of ``x`` (ties
      broken by first appearance). Dispatches on input like
      :func:`min_rank`.

    Examples
    --------
    >>> import hea
    >>> from hea.tidy import row_number
    >>> hea.DataFrame({"x": [10, 20, 30]}).mutate(id=row_number())  # doctest: +SKIP
    """
    if x is None:
        return pl.int_range(0, pl.len())
    if isinstance(x, pl.Expr):
        return x.rank("ordinal") - 1
    if isinstance(x, pl.Series):
        return x.rank("ordinal") - 1
    return _eager_rank_out(x, _rankdata_with_nan(_as_array(x), method="ordinal") - 1)


def min_rank(x):
    """dplyr's ``min_rank()`` — 0-based ranks; ties get the smallest rank,
    next rank skipped.

    ``min_rank([1, 5, 5, 17, 22, None])`` → ``Series[0, 1, 1, 3, 4, null]``.
    Dispatches on input like :func:`row_number`; NA / null propagates.
    R / dplyr's ``min_rank()`` starts at 1; hea follows Python indexing.
    """
    if isinstance(x, pl.Expr):
        return x.rank("min") - 1
    if isinstance(x, pl.Series):
        return x.rank("min") - 1
    return _eager_rank_out(x, _rankdata_with_nan(_as_array(x), method="min") - 1)


def dense_rank(x):
    """dplyr's ``dense_rank()`` — 0-based; like :func:`min_rank` but no gaps
    after ties.

    ``dense_rank([1, 5, 5, 17, 22, None])`` → ``Series[0, 1, 1, 2, 3, null]``.
    R / dplyr's ``dense_rank()`` starts at 1; hea follows Python indexing.
    """
    if isinstance(x, pl.Expr):
        return x.rank("dense") - 1
    if isinstance(x, pl.Series):
        return x.rank("dense") - 1
    return _eager_rank_out(x, _rankdata_with_nan(_as_array(x), method="dense") - 1)


def percent_rank(x):
    """dplyr's ``percent_rank()`` — ``(min_rank(x) - 1) / (n - 1)``.

    ``n`` is the non-null count. Returns 0 for the minimum and 1 for the
    maximum; NA / null propagates. ``NaN`` if there's only one non-null value
    (division by zero, matches R).
    """
    if isinstance(x, pl.Expr):
        return (x.rank("min") - 1) / (x.count() - 1)
    if isinstance(x, pl.Series):
        return (x.rank("min") - 1) / (x.count() - 1)
    arr = _as_array(x)
    n = int((~np.isnan(arr)).sum())
    return _eager_rank_out(x, (_rankdata_with_nan(arr, method="min") - 1) / (n - 1))


def cume_dist(x):
    """dplyr's ``cume_dist()`` — cumulative distribution: ``rank("max") / n``.

    ``n`` is the non-null count. Returns the proportion of values ≤ each
    entry; NA / null propagates.
    """
    if isinstance(x, pl.Expr):
        return x.rank("max") / x.count()
    if isinstance(x, pl.Series):
        return x.rank("max") / x.count()
    arr = _as_array(x)
    n = int((~np.isnan(arr)).sum())
    return _eager_rank_out(x, _rankdata_with_nan(arr, method="max") / n)


def ntile(x, n):
    """dplyr's ``ntile(x, n)`` — bucket ``x`` into ``n`` roughly-equal groups.

    Uses ordinal rank, so ties may end up in different buckets. Where ``n``
    doesn't divide the non-null count evenly, the first ``count % n`` buckets
    get one extra element. Bucket labels are 0-based: ``ntile(range(10), 4)``
    → ``[0,0,0,1,1,1,2,2,3,3]`` (sizes 3, 3, 2, 2). NA / null propagates.
    R / dplyr's ``ntile()`` is 1-based; hea follows Python indexing.
    """
    if isinstance(x, pl.Expr):
        r = x.rank("ordinal")
        count = x.count()
        n_larger = count % n
        larger_size = (count + n - 1) // n
        smaller_size = count // n
        threshold = larger_size * n_larger
        return (
            pl.when(r <= threshold)
            .then((r + larger_size - 1) // larger_size)
            .otherwise(
                (r - threshold + smaller_size - 1) // smaller_size + n_larger
            )
        ) - 1
    if isinstance(x, pl.Series):
        r = x.rank("ordinal")
        count = x.count()
        if count == 0:
            return pl.Series([None] * len(x), dtype=pl.UInt32)
        n_larger = count % n
        larger_size = -(-count // n)
        smaller_size = count // n
        threshold = larger_size * n_larger
        upper = (r + larger_size - 1) // larger_size
        lower = (r - threshold + smaller_size - 1) // smaller_size + n_larger
        cond = (r <= threshold).to_numpy()
        return pl.Series(np.where(cond, upper.to_numpy(), lower.to_numpy()) - 1)
    arr = _as_array(x)
    mask = ~np.isnan(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if mask.any():
        ordinal = _sps.rankdata(arr[mask], method="ordinal").astype(np.int64)
        count = int(mask.sum())
        n_larger = count % n
        larger_size = -(-count // n)
        smaller_size = count // n
        threshold = larger_size * n_larger
        upper = (ordinal + larger_size - 1) // larger_size
        lower = (ordinal - threshold + smaller_size - 1) // smaller_size + n_larger
        out[mask] = np.where(ordinal <= threshold, upper, lower) - 1
    return _eager_rank_out(x, out)


# ---- dplyr window / mutate helpers --------------------------------

def lag(x, n=1, default=None, order_by=None):
    """dplyr's ``lag()`` — value ``n`` positions before each entry.

    ``lag([2, 5, 11, 11, 19, 35])`` → ``[NA, 2, 5, 11, 11, 19]``. Entries
    with no predecessor (the first ``n``) get ``default`` (``None`` →
    null/NA, matching dplyr's default).

    ``order_by`` reorders the input by another vector before computing
    the lag, then restores the original positions. Use when rows aren't
    already in chronological order:

    >>> df.mutate(prev=hea.lag(pl.col("x"), order_by="t"))  # doctest: +SKIP

    Inside ``group_by() %>% mutate()`` the lag is per-group automatically
    (polars / dplyr's ``mutate`` handles the grouping).

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    return _lag_lead(x, int(n), default, order_by)


def lead(x, n=1, default=None, order_by=None):
    """dplyr's ``lead()`` — value ``n`` positions after each entry.

    ``lead([2, 5, 11, 11, 19, 35])`` → ``[5, 11, 11, 19, 35, NA]``. Mirror
    of :func:`lag` — see that docstring for full arguments.
    """
    return _lag_lead(x, -int(n), default, order_by)


def _lag_lead(x, k, default, order_by):
    """Signed shift: ``k > 0`` lags, ``k < 0`` leads. Shared by lag/lead."""
    if isinstance(x, pl.Expr):
        if order_by is None:
            return x.shift(k, fill_value=default)
        ob = order_by if isinstance(order_by, pl.Expr) else pl.col(order_by)
        inv = ob.arg_sort().arg_sort()
        return x.sort_by(ob).shift(k, fill_value=default).gather(inv)

    is_series = isinstance(x, pl.Series)
    is_ndarray = isinstance(x, np.ndarray)

    if order_by is not None:
        ob_arr = (
            order_by.to_numpy() if isinstance(order_by, pl.Series)
            else np.asarray(order_by)
        )
        order = np.argsort(ob_arr, kind="stable")
        inv = np.argsort(order, kind="stable")
        x_arr = (x.to_numpy() if is_series else np.asarray(x))[order]
        s = pl.Series(x_arr)
    else:
        s = x if is_series else pl.Series(x)

    out = s.shift(k, fill_value=default)

    if order_by is not None:
        out = out.gather(pl.Series(inv))

    if is_series:
        return out
    if is_ndarray:
        return out.to_numpy()
    return out  # list/tuple → pl.Series, matching min_rank/dense_rank pattern


def between(x, left, right):
    """dplyr's ``between(x, left, right)`` — ``left <= x <= right`` (both inclusive).

    NA / null in ``x`` propagates. Wraps polars' ``Expr.between`` /
    ``Series.is_between`` with the dplyr name and a top-level dispatch.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        return x.is_between(left, right, closed="both")
    if isinstance(x, pl.Series):
        return x.is_between(left, right, closed="both")
    is_ndarray = isinstance(x, np.ndarray)
    s = pl.Series(x)
    out = s.is_between(left, right, closed="both")
    return out.to_numpy() if is_ndarray else out


def na_if(x, y):
    """dplyr's ``na_if(x, y)`` — replace value ``y`` in ``x`` with NA / null.

    ``na_if([1, 0, 3, 0], 0)`` → ``[1, NA, 3, NA]``. Useful for cleaning
    sentinel codes (empty strings, ``-99``, …) into proper missing values.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        return pl.when(x == y).then(None).otherwise(x)
    if isinstance(x, pl.Series):
        return x.set(x == y, None)
    is_ndarray = isinstance(x, np.ndarray)
    s = pl.Series(x)
    out = s.set(s == y, None)
    return out.to_numpy() if is_ndarray else out


def near(x, y, tol=1.5e-8):
    """dplyr's ``near(x, y, tol)`` — approximate equality, ``|x - y| < tol``.

    Default tolerance matches dplyr: ``sqrt(.Machine$double.eps)`` ≈ ``1.49e-8``.
    Element-wise; ``y`` may be a scalar or same-length vector.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        return (x - y).abs() < tol
    if isinstance(x, pl.Series):
        return (x - y).abs() < tol
    is_ndarray = isinstance(x, np.ndarray)
    # Scalar shortcut — dplyr's near(1, 1+1e-10) returns a length-1 logical;
    # in Python, returning a bool is the natural shape for scalar inputs.
    if np.isscalar(x) and np.isscalar(y):
        return bool(abs(float(x) - float(y)) < tol)
    arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float) if not np.isscalar(y) else y
    out = np.abs(arr - y_arr) < tol
    return out if is_ndarray else pl.Series(out)


# ---- dplyr cumulative helpers --------------------------------------

def cummean(x):
    """dplyr's ``cummean()`` — cumulative mean.

    ``cummean([1, 2, 3, 4, 5])`` → ``[1, 1.5, 2, 2.5, 3]``. NA / null
    propagates from the first missing value onward (matches dplyr's
    ``cumsum(x) / seq_along(x)`` definition).

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        csum = x.cum_sum()
        denom = pl.int_range(1, pl.len() + 1)
        has_na = x.is_null().cum_max()
        return pl.when(has_na).then(None).otherwise(csum / denom)
    is_series = isinstance(x, pl.Series)
    is_ndarray = isinstance(x, np.ndarray)
    arr = np.asarray(x.to_list() if is_series else x, dtype=float)
    csum = np.cumsum(arr)
    out = csum / np.arange(1, len(arr) + 1, dtype=float)
    if is_ndarray:
        return out
    return pl.Series(out, nan_to_null=True)


def cumall(x):
    """dplyr's ``cumall()`` — cumulative all (logical AND).

    TRUE while every value so far has been TRUE. ``FALSE`` absorbs
    everything after it; ``NA`` propagates only until a ``FALSE`` is
    seen (``FALSE`` takes precedence over ``NA``).

    ``cumall([T, T, F, T])`` → ``[T, T, F, F]``;
    ``cumall([T, NA, T])`` → ``[T, NA, NA]``;
    ``cumall([F, NA])`` → ``[F, F]``.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        # has_false = any FALSE so far; has_na = any NA so far
        has_false = (x == False).fill_null(False).cum_max()  # noqa: E712
        has_na = x.is_null().cum_max()
        return (
            pl.when(has_false).then(False)
            .when(has_na).then(None)
            .otherwise(True)
        )
    return _cumall_cumany_eager(x, all_=True)


def cumany(x):
    """dplyr's ``cumany()`` — cumulative any (logical OR).

    FALSE until the first TRUE, then TRUE forever (TRUE absorbs).
    ``NA`` propagates only until a ``TRUE`` is seen.

    ``cumany([F, F, T, F])`` → ``[F, F, T, T]``;
    ``cumany([F, NA, F])`` → ``[F, NA, NA]``;
    ``cumany([T, NA])`` → ``[T, T]``.

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple → ``pl.Series``; ndarray → ``ndarray``.
    """
    if isinstance(x, pl.Expr):
        has_true = x.fill_null(False).cum_max()
        has_na = x.is_null().cum_max()
        return (
            pl.when(has_true).then(True)
            .when(has_na).then(None)
            .otherwise(False)
        )
    return _cumall_cumany_eager(x, all_=False)


def _cumall_cumany_eager(x, all_):
    """Shared eager loop for ``cumall`` (all_=True) and ``cumany`` (all_=False).

    Returns a ``pl.Series`` of Boolean (with null) for Series / list /
    tuple input, or an object ndarray for ndarray input.
    """
    is_series = isinstance(x, pl.Series)
    is_ndarray = isinstance(x, np.ndarray)
    src = x.to_list() if is_series else list(x)
    if all_:
        absorb, default_state = False, True   # FALSE absorbs; start TRUE
    else:
        absorb, default_state = True, False   # TRUE absorbs; start FALSE
    state = default_state
    out = []
    for v in src:
        if state is absorb:
            out.append(absorb)
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            state = None
            out.append(None)
            continue
        v_bool = bool(v)
        if v_bool is absorb:
            state = absorb
            out.append(absorb)
        else:
            # non-absorbing: state is True (cumall) or False (cumany), or None
            out.append(state)
    if is_ndarray:
        return np.asarray(out, dtype=object)
    return pl.Series(out, dtype=pl.Boolean)


# ---- dplyr positional pickers (first / last / nth, consecutive_id) -

def first(x, default=None, order_by=None, na_rm=True):
    """dplyr's ``first()`` — first non-null element of ``x``.

    ``na_rm=True`` is hea's default (matches the rest of the R-shaped
    API; diverges from dplyr's ``na_rm=FALSE``). Pass ``na_rm=False``
    to get the literal first row, even if it's null. Returns ``default``
    if ``x`` is empty (or, with the default ``na_rm=True``, has no
    non-null entries). ``order_by`` reorders ``x`` before picking.

    Shadows polars' top-level ``pl.first`` (which is a *column* selector,
    not an *element* picker) — the dplyr shape is what you want inside
    ``mutate``:

    >>> df.mutate(diff=pl.col("time") - hea.lag(  # doctest: +SKIP
    ...     pl.col("time"), default=hea.first(pl.col("time"))
    ... ))

    polars' first-column selector remains accessible as ``pl.first``;
    inside a polars Expr, ``pl.col("x").first()`` is the equivalent
    method shape.

    Type-in / type-out: ``pl.Expr`` → scalar ``pl.Expr`` (broadcasts
    inside ``mutate``); ``pl.Series`` / list / tuple / ndarray → Python
    scalar.
    """
    return _first_last_nth(x, 0, default, order_by, na_rm)


def last(x, default=None, order_by=None, na_rm=True):
    """dplyr's ``last()`` — last non-null element of ``x``. Mirror of
    :func:`first`.

    ``na_rm=True`` is hea's default — see :func:`first` for the rationale.
    Shadows polars' top-level ``pl.last``; use ``pl.col("x").last()`` for
    the polars method shape (which returns the literal last row,
    equivalent to ``hea.last(x, na_rm=False)``).
    """
    return _first_last_nth(x, -1, default, order_by, na_rm)


def nth(x, n, order_by=None, default=None, na_rm=True):
    """dplyr's ``nth(x, n)`` — n-th element, 0-based.

    ``nth(x, 0)`` is the first, ``nth(x, 1)`` the second. Negative ``n``
    counts from the end: ``nth(x, -1)`` is the last, ``nth(x, -2)`` the
    second-to-last. Out-of-bounds returns ``default``. R / dplyr's
    ``nth()`` is 1-based; hea follows Python indexing.

    ``na_rm=True`` is hea's default — null entries don't consume an
    index slot (so ``nth([1, None, 3, 4], 1)`` returns ``3``). Pass
    ``na_rm=False`` to count literal row positions. A ``None`` *value*
    at index ``n`` (when ``na_rm=False``) is returned as-is — ``default``
    only fires on OOB.

    Shadows polars' top-level ``pl.nth``. Mirror of :func:`first` for
    the dispatch matrix.
    """
    return _first_last_nth(x, int(n), default, order_by, na_rm)


def _first_last_nth(x, k, default, order_by, na_rm):
    """Shared logic. ``k`` is 0-based: 0 = first, -1 = last, 1 = second…
    """
    if isinstance(x, pl.Expr):
        return _first_last_nth_expr(x, k, default, order_by, na_rm)
    return _first_last_nth_eager(x, k, default, order_by, na_rm)


def _first_last_nth_expr(x_expr, k, default, order_by, na_rm):
    src = x_expr
    if order_by is not None:
        ob = order_by if isinstance(order_by, pl.Expr) else pl.col(order_by)
        src = src.sort_by(ob)
    if na_rm:
        src = src.drop_nulls()
    # polars' ``slice`` handles negative offsets (from end); slice(-1, 1)
    # is the last element, slice(0, 1) the first. ``.first()`` on a
    # 0-length slice (OOB) yields null — no ComputeError, unlike
    # ``.gather()``.
    val = src.slice(k, 1).first()
    if default is None:
        return val
    # OOB if ``|k| > len`` for negative k, or ``k >= len`` for non-negative.
    need_len = -k if k < 0 else k + 1
    in_bounds = src.len() >= need_len
    return pl.when(in_bounds).then(val).otherwise(pl.lit(default))


def _first_last_nth_eager(x, k, default, order_by, na_rm):
    if isinstance(x, pl.Series):
        arr = x.to_list()
    elif isinstance(x, np.ndarray):
        arr = x.tolist()
    else:
        arr = list(x)
    if order_by is not None:
        ob_list = (
            order_by.to_list() if isinstance(order_by, pl.Series)
            else list(order_by)
        )
        order = sorted(range(len(arr)), key=lambda i: ob_list[i])
        arr = [arr[i] for i in order]
    if na_rm:
        arr = [
            v for v in arr
            if not (v is None or (isinstance(v, float) and np.isnan(v)))
        ]
    idx = k if k >= 0 else len(arr) + k
    if 0 <= idx < len(arr):
        return arr[idx]
    return default


# ---- runs / consecutive identity (dplyr) ----------------------------

def consecutive_id(*args):
    """dplyr's ``consecutive_id()`` — 0-based id for each run of consecutive
    equal values.

    Returns 0 for the first row, then increments each time *any* of the
    inputs changes from the previous row. With multiple inputs, treats
    them as a tuple — the id increments when the tuple changes.

    ``consecutive_id([1, 1, 2, 2, 2, 1, 1])`` → ``[0, 0, 1, 1, 1, 2, 2]``.
    ``consecutive_id(["a","a","b","a"], [1,1,1,1])`` → ``[0, 0, 1, 2]``.

    Thin wrapper around polars' ``Expr.rle_id`` (also 0-based). R /
    dplyr's ``consecutive_id()`` is 1-based; hea follows Python indexing.

    Type-in / type-out: all-Expr / string → ``pl.Expr``; eager inputs
    follow the first arg's type (Series / list → Series; ndarray →
    ndarray).
    """
    if not args:
        raise TypeError("consecutive_id() requires at least one argument")

    # Pure-Expr / column-name path → return an Expr suitable for mutate().
    if all(isinstance(a, (pl.Expr, str)) for a in args):
        exprs = [a if isinstance(a, pl.Expr) else pl.col(a) for a in args]
        if len(exprs) == 1:
            return exprs[0].rle_id()
        return pl.struct(exprs).rle_id()

    # Eager path.
    first = args[0]
    if len(args) == 1:
        if isinstance(first, pl.Series):
            return first.rle_id()
        is_ndarray = isinstance(first, np.ndarray)
        out = pl.Series(first).rle_id()
        return out.to_numpy() if is_ndarray else out

    # Multiple eager args — combine into a tiny frame, struct-then-rle.
    cols = {}
    for i, a in enumerate(args):
        name = f"__c{i}"
        cols[name] = a.to_list() if isinstance(a, pl.Series) else list(a)
    df = pl.DataFrame(cols)
    out = df.select(pl.struct(pl.all()).rle_id()).to_series().rename("")
    is_ndarray = isinstance(first, np.ndarray)
    return out.to_numpy() if is_ndarray else out

