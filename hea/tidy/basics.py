"""Small standalone tidyverse-flavored utilities.

* ``tbl`` — rewrap a plain polars object as its ``hea`` subclass.
* ``desc`` — descending-sort marker / value negation (dplyr ``desc()``).
* ``exclude`` — :func:`polars.exclude` lifted to also accept frames /
  series / lists.
* ``n`` / ``n_distinct`` — :data:`pl.len` / :data:`pl.n_unique` aliases
  (dplyr's ``n()`` / ``n_distinct()``).
* ``if_else`` / ``case_when`` — vectorized conditionals with dplyr's
  null-handling defaults.
* ``glimpse`` — ``tibble::glimpse`` (one-line-per-column transpose-view).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def tbl(obj):
    """Re-wrap a plain polars container as the corresponding hea subclass.

    Rarely needed in normal code: Phase 1-5 of the subclass-coverage work
    means every operation hea exposes already returns the right subclass.
    ``tbl`` is the documented escape hatch for the remaining cases — e.g.
    when an external library hands you a ``pl.DataFrame`` and you want
    to chain hea methods on it without copying data:

    >>> import polars as pl, hea
    >>> raw = pl.DataFrame({"x": [1, 2, 3]})  # plain polars
    >>> hea.tbl(raw).filter(pl.col("x") > 1)  # hea subclass
    """
    # Lazy imports — the class hierarchy lives in tidy.dataframe / tidy.series,
    # both of which import from this module via the package __init__. Resolving
    # at call time dodges the import cycle.
    from .dataframe import DataFrame
    from .series import LazyFrame, Series

    if isinstance(obj, DataFrame):
        return obj
    if isinstance(obj, pl.DataFrame):
        return DataFrame._from_pydf(obj._df)
    if isinstance(obj, LazyFrame):
        return obj
    if isinstance(obj, pl.LazyFrame):
        return LazyFrame._from_pyldf(obj._ldf)
    if isinstance(obj, Series):
        return obj
    if isinstance(obj, pl.Series):
        return Series._from_pyseries(obj._s)
    raise TypeError(
        f"tbl(): expected pl.DataFrame / pl.LazyFrame / pl.Series, got {type(obj).__name__}"
    )


class _Desc:
    """Marker for descending sort, produced by ``desc("colname")``."""

    __slots__ = ("col",)

    def __init__(self, col: str):
        self.col = col


def desc(col: Any) -> Any:
    """Reverse the sort order of a column or vector — mirrors dplyr's ``desc()``.

    Two call shapes:

    * ``desc("name")`` returns a ``_Desc`` marker that ``arrange()``
      recognizes: ``df.arrange("a", desc("b"))`` sorts by ``a``
      ascending then ``b`` descending.
    * ``desc(values)`` negates the values, matching R's
      ``-xtfrm(x)`` definition. Useful with verbs that take a vector
      directly, e.g. ``min_rank(desc(x))`` gives the descending
      min-rank.

    Type-in / type-out for the value form:
        * ``pl.Expr`` → negated ``pl.Expr``
        * ``pl.Series`` → negated ``pl.Series``
        * list / tuple / ndarray → negated ``np.ndarray`` (float)
    """
    if isinstance(col, str):
        return _Desc(col)
    if isinstance(col, (pl.Expr, pl.Series)):
        return -col
    return -np.asarray(col, dtype=float)


def exclude(*columns: Any) -> pl.Expr:
    """Like :func:`polars.exclude`, but also accepts a :class:`DataFrame`
    (uses ``.columns``), :class:`Series` (uses ``.name``), or list/tuple
    thereof. Lets ``df.select(hea.exclude(df["year":"day"]))`` mirror the
    positive form ``df.select(df["year":"day"])``.
    """
    flat: list[Any] = []
    for c in columns:
        if isinstance(c, (list, tuple)):
            flat.extend(c)
        elif isinstance(c, pl.DataFrame):
            flat.extend(c.columns)
        elif isinstance(c, pl.Series):
            flat.append(c.name)
        else:
            flat.append(c)
    return pl.exclude(flat)


# dplyr's ``n()`` — row-count expression for ``mutate`` / ``summarize``.
# Aliased to ``pl.len`` so ``from hea import n`` doesn't shadow the builtin
# ``len`` (which ``from hea import len`` would, since ``hea.len`` is
# ``polars.len`` via the star-import in __init__.py).
n = pl.len

# dplyr's ``n_distinct()`` — polars exposes the same operation as
# ``n_unique``. Both names route to the same Expr; ``n_unique`` is also
# reachable as ``hea.n_unique`` via the polars star-import.
n_distinct = pl.n_unique


# ---- conditionals (dplyr) -------------------------------------------

def if_else(condition, true_value, false_value, missing=None) -> pl.Expr:
    """dplyr's ``if_else()`` — vectorized conditional.

    Wraps ``pl.when(condition).then(true_value).otherwise(false_value)``
    with one dplyr-shaped twist: a null in ``condition`` produces
    ``missing`` (default ``None`` → null), matching dplyr's ``NA in →
    NA out``. Polars' raw ``when/then/otherwise`` instead routes nulls
    through the otherwise branch — use ``pl.when(...)`` directly if
    that's what you want.

    Returns a ``pl.Expr``. Use inside ``mutate`` / ``select`` / any
    polars verb. For Series-on-Series eager evaluation, materialize
    via ``df.select(if_else(...))`` or use ``Series.zip_with``.

    Parameters
    ----------
    condition : pl.Expr | pl.Series | bool
        Boolean predicate.
    true_value, false_value : pl.Expr | pl.Series | scalar
        Values for True / False entries. Bare scalars are auto-lifted
        via ``pl.lit`` by polars' ``when/then`` machinery.
    missing : scalar, optional
        Value emitted when ``condition`` is null. Defaults to ``None``
        (null), matching dplyr's ``NA`` default.
    """
    # Polars' .then("x") interprets a bare string as a column name; dplyr's
    # if_else treats strings as literals. Lift any non-Expr non-Series value
    # to pl.lit so "5" stays "5".
    def _lit(v):
        return v if isinstance(v, (pl.Expr, pl.Series)) else pl.lit(v)

    t, f = _lit(true_value), _lit(false_value)
    if isinstance(condition, (pl.Expr, pl.Series)):
        return (
            pl.when(condition.is_null()).then(pl.lit(missing))
            .when(condition).then(t)
            .otherwise(f)
        )
    return pl.when(condition).then(t).otherwise(f)


def case_when(*pairs, default=None) -> pl.Expr:
    """dplyr's ``case_when()`` — multi-branch vectorized conditional.

    Each pair is ``(condition, value)``. The result for each row is the
    ``value`` of the first pair whose ``condition`` is True; rows matching
    no condition take ``default``. Mirrors dplyr's ``case_when()`` —
    Python has no ``cond ~ value`` formula syntax, so pass tuples instead.

    Bare-string ``value``s are lifted to ``pl.lit`` (matching dplyr's
    "strings are values" convention). Polars' raw ``pl.when(...).then("x")``
    would interpret ``"x"`` as a column reference.

    Null conditions fall through to the next branch (and ultimately to
    ``default``), matching dplyr 1.1+. Use :func:`if_else` if you want
    null-in → null-out instead.

    Parameters
    ----------
    *pairs : tuple[condition, value]
        Each ``condition`` is a boolean ``pl.Expr`` / ``pl.Series``;
        each ``value`` is a scalar / ``pl.Expr`` / ``pl.Series``.
    default : scalar | pl.Expr, optional
        Value for rows matching no condition. Defaults to ``None`` (null),
        matching dplyr's ``.default = NA``.

    Examples
    --------
    >>> import hea
    >>> from hea import case_when, col
    >>> df = hea.DataFrame({"drv": ["f", "r", "4", "f"]})
    >>> df.mutate(label=case_when(
    ...     (col("drv") == "f", "front-wheel drive"),
    ...     (col("drv") == "r", "rear-wheel drive"),
    ...     (col("drv") == "4", "4-wheel drive"),
    ... ))  # doctest: +SKIP
    """
    if not pairs:
        raise TypeError(
            "case_when() requires at least one (condition, value) pair"
        )

    def _lit(v):
        return v if isinstance(v, (pl.Expr, pl.Series)) else pl.lit(v)

    expr = None
    for i, pair in enumerate(pairs):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise TypeError(
                f"case_when() pair {i} must be a (condition, value) "
                f"tuple, got {pair!r}"
            )
        cond, val = pair
        val = _lit(val)
        expr = pl.when(cond).then(val) if expr is None else expr.when(cond).then(val)
    return expr.otherwise(_lit(default))


def glimpse(df, **kwargs):
    """tibble::glimpse — print a one-line-per-column transpose-view."""
    return df.glimpse(**kwargs)
