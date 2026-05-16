"""forcats-style factor manipulation: ``fct_reorder``, ``fct_reorder2``,
``fct_rev``, ``fct_relevel``, ``fct_infreq``, ``fct_recode``,
``fct_collapse``, ``fct_lump_n``, ``fct_lump_lowfreq``.

Each ``fct_*`` returns a callable ``data -> Series`` so the operation can
fold into ``mutate`` / ``select`` (resolved by
:func:`~hea.tidy._shared._resolve_lazy_factors`) or be supplied inline
to ggplot ``aes(...)`` (resolved by the build pipeline). The callable
carries ``__hea_label__`` / ``__hea_aes_source__`` so axis labels
resolve to the source column name, not ``"<function>"``.

Composability — ``fct_rev`` and ``fct_relevel`` accept either a column
name OR another ``fct_*`` callable, so ``fct_rev(fct_infreq("g"))``
translates R's ``g |> fct_infreq() |> fct_rev()``. Aggregators
(``fct_reorder`` / ``fct_reorder2`` / ``fct_infreq``) need the original
column for grouping and only accept a column name.
"""
from __future__ import annotations

from typing import Callable, Union

import polars as pl


ColInput = Union[str, Callable[[pl.DataFrame], pl.Series]]


def _label_callable(fn: Callable, label: str) -> Callable:
    """Tag ``fn`` so the renderer / aes_source can pull a label from it."""
    fn.__hea_label__ = label
    fn.__hea_aes_source__ = label
    return fn


def _chain_label(col: ColInput) -> str:
    """Resolve the source-column label for a string OR a chained fct_* callable.

    Used so ``fct_rev(fct_infreq("marital"))`` still labels its axis
    ``"marital"`` — the outer tag propagates from the inner.
    """
    if callable(col):
        return getattr(col, "__hea_label__", "_chain")
    return col


def _resolve_col_input(col: ColInput, data: pl.DataFrame) -> pl.Series:
    """Get the input Series for a string col-name OR a chained callable.

    Lets ``fct_rev`` / ``fct_relevel`` accept either a column name
    (the existing contract) or the output of another ``fct_*`` call
    (so R's ``marital |> fct_infreq() |> fct_rev()`` translates to
    ``fct_rev(fct_infreq("marital"))``).
    """
    if callable(col):
        return col(data)
    return data[col]


def _append_unseen_enum_levels(s: pl.Series, levels: list[str]) -> list[str]:
    """Append Enum categories absent from the data so reordering keeps R parity.

    ``fct_reorder`` / ``fct_infreq`` derive their level order from a
    group_by, which only iterates values actually present in ``s``. R's
    ``fct_reorder`` keeps unobserved factor levels — their NA aggregate
    sorts to the end — so we append them in original Enum order to match.
    """
    if not isinstance(s.dtype, pl.Enum):
        return levels
    seen = set(levels)
    return levels + [lvl for lvl in s.dtype.categories.to_list() if lvl not in seen]


def fct_reorder(col: str, by: str, fn="median", *, desc: bool = False) -> Callable:
    """Reorder ``col``'s factor levels by aggregating ``by`` per level.

    Mirrors R's ``forcats::fct_reorder``. ``fn`` is the name of any
    :class:`polars.Series` aggregation method (``"median"``, ``"mean"``,
    ``"sum"``, ``"min"``, ``"max"``, ``"std"``, ``"count"``, …) or a
    callable ``Series -> scalar``. ``desc=True`` reverses the order so
    the largest aggregate appears first.
    """
    def reorder(data: pl.DataFrame) -> pl.Series:
        if isinstance(fn, str):
            agg_expr = getattr(pl.col(by), fn)()
            ordered = (
                data.lazy()
                .group_by(col, maintain_order=False)
                .agg(agg_expr.alias("_agg"))
                .sort("_agg", descending=desc)
                .collect()
            )
            levels = [str(v) for v in ordered[col].to_list() if v is not None]
        else:
            # Python callable: aggregate per-group in Python so users can
            # pass arbitrary scalar reducers (e.g. ``lambda s: s.quantile(0.9)``).
            # ``.lazy()`` routes through polars' native group_by — hea's
            # subclassed DataFrame exposes a different group_by API.
            grouped = (
                data.lazy()
                .group_by(col, maintain_order=False)
                .agg(pl.col(by).alias("_vals"))
                .collect()
            )
            rows = []
            for row in grouped.iter_rows(named=True):
                level = row[col]
                if level is None:
                    continue
                rows.append((str(level), fn(pl.Series(row["_vals"]))))
            rows.sort(key=lambda r: r[1], reverse=desc)
            levels = [r[0] for r in rows]
        levels = _append_unseen_enum_levels(data[col], levels)
        return data[col].cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(reorder, col)


def fct_reorder2(col: str, x: str, y: str, *, desc: bool = True) -> Callable:
    """Reorder ``col``'s levels by ``y`` at each level's largest ``x``.

    Mirrors R's ``forcats::fct_reorder2``. Designed for the line-plot
    legend-ordering case: a level's rank is the ``y`` value at its
    largest ``x``, so the legend order matches where lines end up at
    the right of the plot. ``desc=True`` (default, like forcats) puts
    the highest end-value first.
    """
    def reorder(data: pl.DataFrame) -> pl.Series:
        ordered = (
            data.lazy()
            .filter(pl.col(x).is_not_null())
            .sort(x)
            .group_by(col, maintain_order=False)
            .agg(pl.col(y).last().alias("_y_at_max_x"))
            .sort("_y_at_max_x", descending=desc, nulls_last=True)
            .collect()
        )
        levels = [str(v) for v in ordered[col].to_list() if v is not None]
        levels = _append_unseen_enum_levels(data[col], levels)
        return data[col].cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(reorder, col)


def fct_rev(col: ColInput) -> Callable:
    """Reverse the level order of ``col``.

    ``col`` is either a column name OR the output of another ``fct_*``
    call — so ``fct_rev(fct_infreq("g"))`` translates R's
    ``g |> fct_infreq() |> fct_rev()``. For Enum inputs, reverses the
    existing level order; for plain string columns, sorts alphabetically
    descending — symmetric with the ascending order ScaleOrdinal would
    otherwise pick.
    """
    def rev(data: pl.DataFrame) -> pl.Series:
        s = _resolve_col_input(col, data)
        if isinstance(s.dtype, pl.Enum):
            levels = list(s.dtype.categories)[::-1]
        else:
            levels = sorted(
                {str(v) for v in s.drop_nulls().to_list()},
                reverse=True,
            )
        return s.cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(rev, _chain_label(col))


def fct_relevel(col: ColInput, *levels: str) -> Callable:
    """Move ``levels`` to the front of ``col``'s factor levels.

    ``col`` is either a column name OR the output of another ``fct_*``
    call (composable with ``fct_infreq`` / ``fct_reorder`` / ``fct_rev``).
    Levels not listed keep their existing relative order behind the
    promoted ones (Enum order if the input is an Enum, alphabetical
    otherwise). Mirrors R's ``forcats::fct_relevel``.
    """
    promoted = [str(lvl) for lvl in levels]

    def relevel(data: pl.DataFrame) -> pl.Series:
        s = _resolve_col_input(col, data)
        if isinstance(s.dtype, pl.Enum):
            existing = list(s.dtype.categories)
        else:
            existing = sorted({str(v) for v in s.drop_nulls().to_list()})
        promoted_set = set(promoted)
        rest = [lvl for lvl in existing if lvl not in promoted_set]
        ordered = promoted + rest
        return s.cast(pl.Utf8).cast(pl.Enum(ordered))

    return _label_callable(relevel, _chain_label(col))


def fct_infreq(col: str) -> Callable:
    """Order ``col``'s factor levels by frequency, descending.

    Mirrors R's ``forcats::fct_infreq``. Most-common level first;
    ties broken by first-encountered order in the data.
    """
    def infreq(data: pl.DataFrame) -> pl.Series:
        # ``.lazy()`` routes through polars' native group_by API; hea's
        # subclassed DataFrame has its own group_by signature.
        counts = (
            data.lazy()
            .group_by(col, maintain_order=True)
            .agg(pl.len().alias("_n"))
            .sort("_n", descending=True)
            .collect()
        )
        levels = [str(v) for v in counts[col].to_list() if v is not None]
        levels = _append_unseen_enum_levels(data[col], levels)
        return data[col].cast(pl.Utf8).cast(pl.Enum(levels))

    return _label_callable(infreq, col)


def _input_levels(s: pl.Series) -> list[str]:
    """Return the canonical level list for ``s`` — Enum categories if
    available, otherwise sorted unique non-null string values. Shared by
    the rename / collapse / lump family so they all use the same source.
    """
    if isinstance(s.dtype, pl.Enum):
        return list(s.dtype.categories)
    return sorted({str(v) for v in s.drop_nulls().to_list()})


def fct_recode(col: str, **renames) -> Callable:
    """Rename (and optionally merge) factor levels via ``new=old`` kwargs.

    Mirrors R's ``forcats::fct_recode``. Kwarg direction matches R
    (``new = old``). Values can be either a single string (1:1 rename)
    or a list/tuple of strings (many:1 merge — Python's equivalent of
    R's repeated-keyword trick, since dict keys can't repeat). Use
    ``**{}`` for keys that aren't valid Python identifiers::

        fct_recode("partyid", **{
            "Republican, strong":    "Strong republican",   # 1:1
            "Republican, weak":      "Not str republican",
            "Other":                 ["No answer", "Don't know", "Other party"],  # many:1
        })

    Unmentioned levels keep their original name and position. For
    sweeping all unmentioned levels into a single bucket, use
    ``fct_collapse(..., other_level=)``.
    """
    if not renames:
        raise ValueError("fct_recode(): pass at least one new=old rename.")
    # Build a flat {old: new} mapping; list/tuple values map every entry.
    old_to_new: dict[str, str] = {}
    for new, old in renames.items():
        if isinstance(old, str):
            old_to_new[old] = new
        elif isinstance(old, (list, tuple)):
            for lvl in old:
                if not isinstance(lvl, str):
                    raise TypeError(
                        f"fct_recode(): {new!r} list contains non-string "
                        f"{lvl!r}."
                    )
                old_to_new[lvl] = new
        else:
            raise TypeError(
                f"fct_recode(): {new!r} maps to {type(old).__name__}; "
                "expected str or list/tuple of str."
            )

    def recode(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        # Preserve original order, replace renamed in place, dedupe
        # so multi-old → one-new doesn't create duplicate level entries.
        new_levels: list[str] = []
        seen: set[str] = set()
        for lvl in old_levels:
            mapped = old_to_new.get(lvl, lvl)
            if mapped not in seen:
                new_levels.append(mapped)
                seen.add(mapped)
        return s.cast(pl.Utf8).replace(old_to_new).cast(pl.Enum(new_levels))

    return _label_callable(recode, col)


def fct_collapse(
    col: str, *, other_level: str | None = None, **groups: list
) -> Callable:
    """Merge many old levels into one new level via ``new=[old, ...]`` kwargs.

    Mirrors R's ``forcats::fct_collapse``. Each kwarg maps a new level
    to a list/tuple of old levels that should map to it. Levels not in
    any group keep their original name (with ``other_level=None``) or
    are lumped into ``other_level`` if set. Result level order: declared
    new levels in kwarg order, then either ``other_level`` (if set) or
    any remaining original levels in their original order. Use ``**{}``
    for keys that aren't valid Python identifiers.
    """
    if not groups:
        raise ValueError("fct_collapse(): pass at least one new=[old,...] group.")
    for new, olds in groups.items():
        if not isinstance(olds, (list, tuple)):
            raise TypeError(
                f"fct_collapse(): {new!r} must map to a list/tuple of "
                f"old level names; got {type(olds).__name__}. For 1:1 "
                "rename use fct_recode."
            )
    # Flat {old: new} for the polars replace step.
    old_to_new = {str(old): new for new, olds in groups.items() for old in olds}

    def collapse(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        # Sweep originals into other_level (if set).
        if other_level is not None:
            for lvl in old_levels:
                if lvl not in old_to_new:
                    old_to_new[lvl] = other_level
        # Build the new level list.
        new_levels: list[str] = list(groups.keys())
        if other_level is not None:
            if other_level not in new_levels:
                new_levels.append(other_level)
        else:
            for lvl in old_levels:
                mapped = old_to_new.get(lvl, lvl)
                if mapped not in new_levels:
                    new_levels.append(mapped)
        return s.cast(pl.Utf8).replace(old_to_new).cast(pl.Enum(new_levels))

    return _label_callable(collapse, col)


def _lump_apply(
    s: pl.Series, lumped: set, kept_in_order: list[str], other_level: str
) -> pl.Series:
    """Shared cast: remap ``lumped`` levels to ``other_level``, return Enum.

    Level order = ``kept_in_order`` (original-factor-order of the kept set)
    with ``other_level`` appended unless it's already a kept name (in
    which case the lumped levels merge into the existing one and the
    Enum doesn't grow).
    """
    if not lumped:
        return s.cast(pl.Utf8).cast(pl.Enum(kept_in_order))
    new_levels = list(kept_in_order)
    if other_level not in new_levels:
        new_levels.append(other_level)
    old_to_new = {lvl: other_level for lvl in lumped}
    return s.cast(pl.Utf8).replace(old_to_new).cast(pl.Enum(new_levels))


def fct_lump_n(col: str, n: int, *, other_level: str = "Other") -> Callable:
    """Keep the top ``n`` levels by count; lump the rest into ``other_level``.

    Mirrors R's ``forcats::fct_lump_n``. When a tie spans the cutoff,
    all tied levels are kept (matches R's default ``ties.method="min"``).
    Kept levels are returned in their original factor-level order, then
    ``other_level`` appended. If ``other_level`` is already a level in
    the data, the lumped values merge into it.
    """
    if n < 0:
        raise ValueError(f"fct_lump_n(): n must be >= 0, got {n}.")

    def lump(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        # Per-level counts. value_counts() honors null; we drop nulls so
        # null doesn't compete for a top-n slot.
        vc = (
            s.cast(pl.Utf8)
            .drop_nulls()
            .value_counts()
            .sort("count", descending=True)
        )
        present = vc.height
        if present <= n:
            return s.cast(pl.Utf8).cast(pl.Enum(old_levels))
        cutoff = int(vc["count"][n - 1]) if n > 0 else (int(vc["count"].max()) + 1)
        kept: set[str] = set()
        for r in vc.iter_rows(named=True):
            if r["count"] >= cutoff:
                kept.add(str(r[s.name]))
        kept_in_order = [lvl for lvl in old_levels if lvl in kept]
        lumped = {lvl for lvl in old_levels if lvl not in kept}
        return _lump_apply(s, lumped, kept_in_order, other_level)

    return _label_callable(lump, col)


def fct_lump_lowfreq(col: str, *, other_level: str = "Other") -> Callable:
    """Lump levels into ``other_level`` so it stays the smallest level.

    Mirrors R's ``forcats::fct_lump_lowfreq``. Walks the level counts in
    descending order; a level is KEPT while its count exceeds the sum of
    all smaller-count levels (the "in_smallest" rule from the forcats
    source). Once that fails, every smaller level gets lumped. Kept
    levels preserve their original factor-level order.
    """
    def lump(data: pl.DataFrame) -> pl.Series:
        s = data[col]
        old_levels = _input_levels(s)
        vc = (
            s.cast(pl.Utf8)
            .drop_nulls()
            .value_counts()
            .sort("count", descending=True)
        )
        names = [str(v) for v in vc[s.name].to_list()]
        counts = vc["count"].to_list()
        # forcats::lump_cutoff — index where lumping starts (Python 0-based).
        # Levels at indices >= cutoff_idx get lumped.
        left = sum(counts)
        cutoff_idx = len(counts)
        for i, c in enumerate(counts):
            left -= c
            if c > left:
                cutoff_idx = i + 1
                break
        lumped = set(names[cutoff_idx:])
        kept_in_order = [lvl for lvl in old_levels if lvl not in lumped]
        return _lump_apply(s, lumped, kept_in_order, other_level)

    return _label_callable(lump, col)


