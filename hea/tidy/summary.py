"""R-style ``summary(df)`` — per-column stat blocks packed into a
terminal-width-aware print object.

* :class:`_SummaryBlock` — one column's ``(label, value)`` rows plus a
  centered header.
* :class:`Summary` — list of blocks; ``__repr__`` lays them out
  side-by-side, wrapping to the terminal width.
* ``_summary_block`` dispatches per dtype (numeric / boolean / factor
  / string / temporal) to produce the right entries.
"""
from __future__ import annotations

import datetime as _dt
import math
import shutil
from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass(slots=True)
class _SummaryBlock:
    """Per-column block of an R-style summary: a header (column name)
    plus a list of ``(label, value)`` rows."""

    name: str
    entries: list[tuple[str, str]] = field(default_factory=list)

    def render(self) -> list[str]:
        """Lay the block out as a list of equal-width strings.

        First element is the centered column name; subsequent elements
        are ``label:value`` rows with labels left-aligned and values
        right-aligned to the widest within the block.
        """
        if self.entries:
            lw = max(len(lbl) for lbl, _ in self.entries)
            vw = max(len(val) for _, val in self.entries)
            body = [f"{lbl:<{lw}}:{val:>{vw}}" for lbl, val in self.entries]
        else:
            body = []
        body_w = max((len(s) for s in body), default=0)
        block_w = max(body_w, len(self.name))
        head = f"{self.name:^{block_w}}"
        body = [f"{s:<{block_w}}" for s in body]
        return [head, *body]


class Summary:
    """R-style ``summary()`` of a DataFrame.

    Returned by :meth:`DataFrame.summary`. Holds one :class:`_SummaryBlock`
    per column; ``__repr__`` packs blocks side-by-side, wrapping to fit
    the terminal width. Display-only: for programmatic per-column stats
    use :meth:`DataFrame.describe` or polars expressions directly.
    """

    __slots__ = ("blocks", "width")

    def __init__(self, blocks: list[_SummaryBlock], width: int | None = None):
        self.blocks = blocks
        self.width = width

    def __repr__(self) -> str:
        w = self.width
        if w is None:
            try:
                w = shutil.get_terminal_size((80, 20)).columns
            except Exception:
                w = 80
        return _render_summary(self.blocks, w)


def _render_summary(blocks: list[_SummaryBlock], width: int) -> str:
    """Pack rendered blocks into rows that fit ``width`` columns."""
    rendered = [b.render() for b in blocks]
    sep = "  "
    rows: list[list[list[str]]] = []
    cur: list[list[str]] = []
    cur_w = 0
    for r in rendered:
        bw = len(r[0])
        added = bw if not cur else len(sep) + bw
        if cur and cur_w + added > width:
            rows.append(cur)
            cur = [r]
            cur_w = bw
        else:
            cur.append(r)
            cur_w += added
    if cur:
        rows.append(cur)

    out: list[str] = []
    for ri, row in enumerate(rows):
        if ri > 0:
            out.append("")
        height = max(len(r) for r in row)
        for i in range(height):
            cells = [
                r[i] if i < len(r) else " " * len(r[0])
                for r in row
            ]
            out.append(sep.join(cells).rstrip())
    return "\n".join(out)


def _summary_block(
    name: str,
    s: pl.Series,
    *,
    maxsum: int,
    digits: int,
) -> _SummaryBlock:
    """Build a single column's summary block, dispatching on dtype."""
    dtype = s.dtype
    n_null = s.null_count()

    if dtype.is_numeric():
        entries = _numeric_entries(s, digits)
        nas_inline = False
    elif dtype == pl.Boolean:
        entries = _boolean_entries(s)
        nas_inline = False
    elif isinstance(dtype, (pl.Enum, pl.Categorical)):
        # _factor_entries appends NA's itself so it can reserve a slot
        # in the maxsum budget (matches R's summary.factor).
        entries = _factor_entries(s, maxsum=maxsum, n_null=n_null)
        nas_inline = True
    elif dtype == pl.String:
        entries = _string_entries(s)
        nas_inline = False
    elif dtype in (pl.Date, pl.Datetime, pl.Time):
        entries = _temporal_entries(s)
        nas_inline = False
    else:
        # Lists, structs, objects — fall back to length/class.
        entries = [
            ("Length", str(s.len())),
            ("Class", str(dtype)),
            ("Mode", str(dtype)),
        ]
        nas_inline = False

    if not nas_inline and n_null > 0:
        entries.append(("NA's", str(n_null)))
    return _SummaryBlock(name, entries)


_NUMERIC_LABELS = ("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")


def _numeric_entries(s: pl.Series, digits: int) -> list[tuple[str, str]]:
    """Six-number summary using R's ``quantile`` type 7 (linear)."""
    s_clean = s.drop_nulls()
    if s_clean.is_empty():
        return list(zip(_NUMERIC_LABELS, ["NA"] * 6))
    stats = [
        s_clean.min(),
        s_clean.quantile(0.25, interpolation="linear"),
        s_clean.median(),
        s_clean.mean(),
        s_clean.quantile(0.75, interpolation="linear"),
        s_clean.max(),
    ]
    formatted = _format_numeric_stats([float(v) for v in stats], digits)
    return list(zip(_NUMERIC_LABELS, formatted))


def _signif_round(x: float, digits: int) -> float:
    """Round ``x`` to ``digits`` significant figures."""
    if x == 0:
        return 0.0
    return round(x, digits - int(math.floor(math.log10(abs(x)))) - 1)


def _format_numeric_stats(values: list[float], digits: int) -> list[str]:
    """Format six-number stats matching R's ``format.default`` semantics.

    Two-stage: ``signif`` each value at ``digits`` to determine whether
    decimals are needed at all, then format each **original** value
    (not the signif version) at the resulting common decimal count via
    fixed-point rounding. This is why ``Mean = 16331.025`` in an
    integer-magnitude column prints as ``16331`` rather than the signif
    rounded ``16330``: the column's signif values are all
    integer-valued, so D = 0, and the original is rounded to integer.
    """
    signifs = [0.0 if v == 0 else _signif_round(v, digits) for v in values]

    if all(v == int(v) for v in signifs):
        return [str(int(round(v))) for v in values]

    max_dec = 0
    for v in signifs:
        if v == 0 or v == int(v):
            continue
        int_digits = int(math.floor(math.log10(abs(v)))) + 1
        if int_digits >= digits:
            continue
        max_dec = max(max_dec, digits - int_digits)
    if max_dec == 0:
        return [str(int(round(v))) for v in values]
    return [f"{v:.{max_dec}f}" for v in values]


def _boolean_entries(s: pl.Series) -> list[tuple[str, str]]:
    """``Mode: logical`` plus FALSE / TRUE counts (matches R)."""
    s_clean = s.drop_nulls()
    n_true = int(s_clean.sum())
    n_false = int(s_clean.len() - n_true)
    return [("Mode", "logical"), ("FALSE", str(n_false)), ("TRUE", str(n_true))]


def _factor_entries(
    s: pl.Series,
    *,
    maxsum: int,
    n_null: int,
) -> list[tuple[str, str]]:
    """Factor counts in level order, with ``(Other)`` collapse and NA's row.

    Matches R's ``summary.factor``: when more levels exist than slots,
    the (maxsum-1) most populous levels are kept and the rest pooled as
    ``(Other)``; an ``NA's`` slot is reserved when the column has nulls.
    """
    slots = maxsum - 1 if n_null > 0 else maxsum

    counts = s.value_counts(sort=True)
    levels = counts[:, 0].to_list()
    nums = counts[:, 1].to_list()
    counted: dict[Any, int] = {
        lvl: int(n) for lvl, n in zip(levels, nums) if lvl is not None
    }

    if isinstance(s.dtype, (pl.Enum, pl.Categorical)):
        all_levels = s.cat.get_categories().to_list()
    else:
        all_levels = sorted(counted.keys())

    pairs = [(lvl, counted.get(lvl, 0)) for lvl in all_levels]

    if len(pairs) > slots:
        # Keep the (slots-1) most populous; pool the rest as "(Other)".
        ranked = sorted(pairs, key=lambda x: -x[1])
        keep_set = {lvl for lvl, _ in ranked[: slots - 1]}
        keep = [(lvl, n) for lvl, n in pairs if lvl in keep_set]
        other = sum(n for lvl, n in pairs if lvl not in keep_set)
        pairs = keep + [("(Other)", other)]

    entries = [(str(lvl), str(n)) for lvl, n in pairs]
    if n_null > 0:
        entries.append(("NA's", str(n_null)))
    return entries


def _string_entries(s: pl.Series) -> list[tuple[str, str]]:
    """R's character summary: just shape, no per-value counts."""
    return [
        ("Length", str(s.len())),
        ("Class", "character"),
        ("Mode", "character"),
    ]


def _temporal_entries(s: pl.Series) -> list[tuple[str, str]]:
    """Six-stat summary on Date / Datetime / Time, formatted as strings.

    For ``Date`` input, polars promotes quantile / mean / median to
    ``Datetime``; we drop the time component to keep the block visually
    consistent (and matches R's ``summary`` on dates returning dates).
    """
    s_clean = s.drop_nulls()
    if s_clean.is_empty():
        return list(zip(_NUMERIC_LABELS, ["NA"] * 6))
    stats = [
        s_clean.min(),
        s_clean.quantile(0.25, interpolation="linear"),
        s_clean.median(),
        s_clean.mean(),
        s_clean.quantile(0.75, interpolation="linear"),
        s_clean.max(),
    ]
    if s.dtype == pl.Date:
        stats = [
            v.date() if isinstance(v, _dt.datetime) else v
            for v in stats
        ]
    formatted = ["NA" if v is None else str(v) for v in stats]
    return list(zip(_NUMERIC_LABELS, formatted))


