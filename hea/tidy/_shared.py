"""Private cross-file helpers for the ``hea.tidy`` package.

Lives below the verb files and the class hierarchy in the import graph,
so anything here must avoid importing :class:`hea.tidy.dataframe.DataFrame`
/ :class:`~hea.tidy.groupby.GroupBy` etc. at module load — use lazy
imports inside the functions that need them.

* Verb-shared helpers — ``_split_arrange``, ``_resolve_anchor``,
  ``_check_groups``, ``_apply_groups``, ``_kwargs_to_exprs``,
  ``_resolve_lazy_factors``.
* Name cleaning — ``_clean_one_name`` / ``_disambiguate_clean_names``
  (janitor::make_clean_names port, used by ``clean_names``).
* Column-range placeholder — :class:`_TidyRange` and the public
  :func:`cols_between` constructor.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import polars as pl

from .basics import _Desc


# ---- janitor-style name cleaning ------------------------------------

_CLEAN_NAMES_REPLACE = (
    ("'", ""),
    ('"', ""),
    ("%", "_percent_"),
    ("#", "_number_"),
)
_CLEAN_NAMES_CAMEL_HEAD = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CLEAN_NAMES_CAMEL_TAIL = re.compile(r"([a-z])([A-Z])")
_CLEAN_NAMES_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def _clean_one_name(name: str) -> str:
    """Snake-case a single column name. Mirrors ``janitor::make_clean_names``
    with default options (case='snake', transliterations='Latin-ASCII').
    Disambiguation of duplicates is handled by the caller.
    """
    if not isinstance(name, str):
        name = str(name)
    for find, repl in _CLEAN_NAMES_REPLACE:
        name = name.replace(find, repl)
    # Latin-ASCII transliteration: decompose to base + combining marks,
    # then drop the combining marks (é → e, ñ → n, ü → u, …).
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # camelCase boundaries: ABCDef → ABC_Def, then aB → a_B. Digit↔upper
    # is intentionally not split (janitor: x1Test → x1test, not x1_test).
    name = _CLEAN_NAMES_CAMEL_HEAD.sub(r"\1_\2", name)
    name = _CLEAN_NAMES_CAMEL_TAIL.sub(r"\1_\2", name)
    name = _CLEAN_NAMES_NON_ALNUM.sub("_", name)
    name = name.strip("_").lower()
    if not name:
        name = "x"
    if name[0].isdigit():
        name = "x" + name
    return name


def _disambiguate_clean_names(names: list[str]) -> list[str]:
    """Resolve collisions in a cleaned-name list by appending ``_2``,
    ``_3``, … (janitor's behavior — first dup is ``_2``, not ``_1``).
    """
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
            continue
        k = 2
        while f"{n}_{k}" in seen:
            k += 1
        cand = f"{n}_{k}"
        out.append(cand)
        seen.add(cand)
    return out


# ---- verb-shared helpers --------------------------------------------

def _split_arrange(cols: tuple) -> tuple[list[str], list[bool]]:
    """Split ``arrange`` args into (column names, descending flags)."""
    names: list[str] = []
    desc_flags: list[bool] = []
    for c in cols:
        if isinstance(c, _Desc):
            names.append(c.col)
            desc_flags.append(True)
        else:
            names.append(c)
            desc_flags.append(False)
    return names, desc_flags


def _resolve_anchor(
    anchor: str | int,
    ref_columns: list[str],
    *,
    after: bool = False,
    verb: str = "mutate",
) -> int:
    """Convert a ``_before`` / ``_after`` anchor to an insertion index.

    Accepts either a column name or a **1-indexed** position (matching
    dplyr's ``.before = 1`` semantics — "before the first column").
    """
    if isinstance(anchor, bool):  # bool is an int subclass; reject explicitly
        raise TypeError(f"{verb}(): _before/_after must be a column name or position, not bool.")
    if isinstance(anchor, int):
        n = len(ref_columns)
        if not (1 <= anchor <= n):
            raise ValueError(
                f"{verb}(): position {anchor} out of range for {n} column(s)."
            )
        idx = anchor - 1
    elif anchor in ref_columns:
        idx = ref_columns.index(anchor)
    else:
        raise ValueError(f"{verb}(): column {anchor!r} not in frame.")
    if after:
        idx += 1
    return idx


_VALID_GROUPS = ("drop", "drop_last", "keep", "rowwise")


def _check_groups(_groups: str) -> None:
    """Validate dplyr-style ``_groups`` arg for ``summarize``."""
    if _groups not in _VALID_GROUPS:
        raise ValueError(
            f"summarize(_groups={_groups!r}): expected one of "
            f"{list(_VALID_GROUPS)!r} (matches dplyr's .groups argument)."
        )


def _apply_groups(out_df, by_cols: list, _groups: str):
    """Resolve dplyr-style ``_groups`` on a summarize result.

    Return type depends on ``_groups`` — analogous to dplyr where the
    result is either an ungrouped tibble or a grouped one (tracked via
    metadata; hea tracks it via the GroupBy wrapper type instead).

    - ``"drop"`` → plain :class:`DataFrame` (polars' natural behavior).
    - ``"drop_last"`` → :class:`GroupBy` on ``by_cols[:-1]``; if only one
      group var existed, collapses to ungrouped (matches dplyr).
    - ``"keep"`` → :class:`GroupBy` on all ``by_cols``.
    - ``"rowwise"`` → :class:`GroupBy` on all ``by_cols``. After a
      ``summarize``, each output row is unique by those columns, so
      "each row is its own group" — operationally equivalent to dplyr's
      rowwise. polars expressions are already row-vectorized, so there's
      no further behavioral distinction inside ``mutate`` downstream.
    """
    from .groupby import GroupBy

    if _groups == "drop":
        return out_df
    if _groups == "drop_last":
        remaining = list(by_cols[:-1])
        if not remaining:
            return out_df
        return GroupBy(out_df, remaining, {"maintain_order": True})
    # "keep" or "rowwise" — group on all original by cols
    return GroupBy(out_df, list(by_cols), {"maintain_order": True})


def _kwargs_to_exprs(args: tuple, kwargs: dict) -> list[pl.Expr]:
    """Translate ``(*args, **kwargs)`` of a verb into a list of polars exprs.

    Positional args pass through. Keyword args ``name=expr`` get
    ``.alias(name)`` so the kwarg name becomes the output column.
    """
    exprs: list[Any] = list(args)
    for name, expr in kwargs.items():
        if isinstance(expr, (pl.Expr, pl.Series)):
            exprs.append(expr.alias(name))
        else:
            # bare scalar / list — broadcast as a literal column
            exprs.append(pl.lit(expr).alias(name))
    return exprs


def _resolve_lazy_factors(
    df: pl.DataFrame, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    """Replace deferred factor placeholders with concrete expressions / series.

    Two placeholder shapes are resolved here:

    * ``_LazyFactor`` — returned by ``factor("col")`` / ``factor(pl.col(...))``
      for str/Expr inputs because the Enum's level set has to be detected
      from the actual data (polars expressions can't ``.to_list()``
      mid-pipeline). Resolved to a ``pl.Expr``.
    * Tagged callables — ``fct_reorder("col", "by")`` and friends (and
      the ``cut_*`` binning helpers) carry ``__hea_aes_source__`` so the
      ggplot build pipeline can invoke them. The same shape works in
      ``mutate`` / ``select``: call with the frame, get back a Series.

    Verbs that own a materialized frame (``mutate``, ``select``,
    ``GroupBy.mutate``) call this pre-pass so downstream code only
    sees real expressions / series. For kwargs the kwarg name is
    offered as a fallback column name when a ``_LazyFactor`` was
    built from a ``pl.Expr`` without an output_name.
    """
    from ..R import _LazyFactor

    def _resolve(v, fallback_name=None):
        if isinstance(v, _LazyFactor):
            return v._resolve(df, fallback_name=fallback_name)
        if callable(v) and hasattr(v, "__hea_aes_source__"):
            return v(df)
        return v

    new_args = tuple(_resolve(a) for a in args)
    new_kwargs = {k: _resolve(v, fallback_name=k) for k, v in kwargs.items()}
    return new_args, new_kwargs


# ---- column-range placeholder ---------------------------------------

class _TidyRange:
    """Frame-less placeholder for dplyr's ``a:b`` column range.

    The translator emits ``cols_between('a', 'b')`` for R's ``select(a:b)``
    style ranges. ``~`` flips the placeholder to its complement so
    ``select(!(a:b))`` → ``~cols_between('a', 'b')``. The receiver verb's
    tidy-select resolver expands against the frame at call time.

    The polars-native form ``~df["a":"b"]`` continues to work via
    :meth:`DataFrame.__invert__` — this class only covers the path where
    the receiver isn't available syntactically (i.e., inside arg lists
    of a method call, which is exactly where the translator emits).
    """

    __slots__ = ("start", "stop", "exclude")

    def __init__(self, start: str, stop: str, *, exclude: bool = False):
        self.start = start
        self.stop = stop
        self.exclude = exclude

    def __invert__(self) -> "_TidyRange":
        return _TidyRange(self.start, self.stop, exclude=not self.exclude)

    def __repr__(self) -> str:
        prefix = "!" if self.exclude else ""
        return f"{prefix}cols_between({self.start!r}, {self.stop!r})"

    def resolve(self, frame) -> list[str]:
        """Expand against ``frame``: list of column names (or their
        complement when ``exclude=True``)."""
        cols = frame.cols_between(self.start, self.stop)
        if not self.exclude:
            return cols
        skip = set(cols)
        return [c for c in frame.columns if c not in skip]


def cols_between(start: str, stop: str) -> _TidyRange:
    """tidy-select column range for use without a frame in hand.

    Pass to ``select`` / ``relocate`` / ``pivot_longer`` / ``fill`` /
    anywhere that accepts tidy-select cols. ``~cols_between('a', 'b')``
    is the negated form.

    The :class:`DataFrame` method of the same name (``df.cols_between(
    'a', 'b')``) is the eager equivalent — both routes expand to the
    same list of column names.
    """
    return _TidyRange(start, stop)
