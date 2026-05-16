"""dplyr two-table verb helpers (chapter 19 of R4DS).

``closest`` / ``overlaps`` / ``within`` build inequality-join expressions;
``join_by`` is the bundle constructor that ``DataFrame.*_join`` accepts in
place of a plain column list.

Each predicate returns a ``pl.Expr`` (or a wrapper of one) that the join
verbs unpack into the appropriate ``polars.join_where`` / ``join`` call.
The natural-join message helper here mirrors dplyr's ``Joining with by``
auto-detect signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl


# Map polars BinaryExpr ops (from Expr.meta.serialize) to a readable name.
_BIN_OPS = {"Eq", "Lt", "LtEq", "Gt", "GtEq"}

# Map a "left <op> right" inequality op to the polars join_asof strategy that
# delivers dplyr's closest() semantics.
_CLOSEST_STRATEGY = {
    "GtEq": "backward",  # left >= right: pick largest right ≤ left
    "Gt":   "backward",
    "LtEq": "forward",   # left <= right: pick smallest right ≥ left
    "Lt":   "forward",
    "Eq":   "nearest",
}


def _extract_col_name(arg: Any) -> str:
    """Pull a single column name from a string or ``col(name)`` expression."""
    if isinstance(arg, str):
        return arg
    if isinstance(arg, pl.Expr):
        names = arg.meta.root_names()
        if len(names) != 1:
            raise ValueError(
                f"join_by helper: expected a single column reference, got {names!r}."
            )
        return names[0]
    raise TypeError(
        f"join_by helper: expected str or col(name), got {type(arg).__name__}."
    )


def _parse_join_binary(expr: pl.Expr) -> tuple[str, str, str] | None:
    """Parse ``col(L) <op> col(R)`` → ``(op, L, R)``; return ``None`` otherwise.

    Reads polars' JSON serialization so we can read off the operator (which
    polars doesn't otherwise expose on the public Expr API).
    """
    import json
    if not isinstance(expr, pl.Expr):
        return None
    try:
        tree = json.loads(expr.meta.serialize(format="json"))
    except Exception:
        return None
    if not isinstance(tree, dict) or "BinaryExpr" not in tree:
        return None
    bx = tree["BinaryExpr"]
    left = bx.get("left"); right = bx.get("right"); op = bx.get("op")
    if op not in _BIN_OPS:
        return None
    if not (isinstance(left, dict) and isinstance(right, dict)):
        return None
    if "Column" not in left or "Column" not in right:
        return None
    return op, left["Column"], right["Column"]


@dataclass(frozen=True)
class _Closest:
    """Marker that asks ``join_by`` to route through ``join_asof``."""
    op: str    # one of _CLOSEST_STRATEGY's keys
    left: str
    right: str


def closest(expr: pl.Expr) -> _Closest:
    """Inside :func:`join_by`: take only the closest matching right row.

    Maps to polars' :meth:`pl.DataFrame.join_asof`:

    - ``closest(col('x') >= col('y'))`` — backward asof (largest ``y`` ≤ ``x``).
    - ``closest(col('x') <= col('y'))`` — forward asof (smallest ``y`` ≥ ``x``).
    - ``closest(col('x') == col('y'))`` — nearest asof.

    Mirrors dplyr's ``join_by(closest(...))`` rolling join.
    """
    parsed = _parse_join_binary(expr)
    if parsed is None:
        raise ValueError(
            "closest(): expected a binary inequality between two column "
            "references, e.g. closest(col('x') >= col('y'))."
        )
    return _Closest(*parsed)


def overlaps(x_lower: Any, x_upper: Any, y_lower: Any, y_upper: Any) -> pl.Expr:
    """Inside :func:`join_by`: rows where ``[x_lower, x_upper]`` overlaps ``[y_lower, y_upper]``.

    Equivalent to ``(x_lower <= y_upper) & (y_lower <= x_upper)``. By
    convention the first two arguments are left-side columns and the last
    two are right-side. Accepts column-name strings or ``col(name)``
    expressions.

    Mirrors dplyr's ``join_by(overlaps(...))`` overlap join.
    """
    xl = pl.col(_extract_col_name(x_lower))
    xu = pl.col(_extract_col_name(x_upper))
    yl = pl.col(_extract_col_name(y_lower))
    yu = pl.col(_extract_col_name(y_upper))
    return (xl <= yu) & (yl <= xu)


def within(x_lower: Any, x_upper: Any, y_lower: Any, y_upper: Any) -> pl.Expr:
    """Inside :func:`join_by`: rows where ``[x_lower, x_upper]`` is contained in ``[y_lower, y_upper]``.

    Equivalent to ``(x_lower >= y_lower) & (x_upper <= y_upper)``. By
    convention the first two arguments are left-side columns and the last
    two are right-side.

    Mirrors dplyr's ``join_by(within(...))``.
    """
    xl = pl.col(_extract_col_name(x_lower))
    xu = pl.col(_extract_col_name(x_upper))
    yl = pl.col(_extract_col_name(y_lower))
    yu = pl.col(_extract_col_name(y_upper))
    return (xl >= yl) & (xu <= yu)


@dataclass
class _JoinBy:
    """Normalized join specification produced by :func:`join_by`."""
    # Parallel lists: equi-key columns on the left and right.
    equi_left: list[str] = field(default_factory=list)
    equi_right: list[str] = field(default_factory=list)
    # Non-equi inequalities as ``(polars_op, left_col, right_col)``. Kept
    # separately so we can rewrite right-side columns under suffix
    # disambiguation when needed.
    ineqs: list[tuple[str, str, str]] = field(default_factory=list)
    # Free-form predicate exprs (e.g. from overlaps / within / between).
    exprs: list[pl.Expr] = field(default_factory=list)
    # Rolling spec — at most one closest() per call.
    asof: _Closest | None = None


def join_by(*args: Any) -> _JoinBy:
    """Specify how two tables match for a join — dplyr's ``join_by()``.

    Accepts any combination of:

    - Strings — ``join_by("tailnum")``: same column on both sides.
    - Polars equality — ``join_by(col("dest") == col("faa"))``: rename on
      the right side.
    - Polars inequality — ``join_by(col("id") < col("id"))``: non-equi
      (routes through :meth:`pl.DataFrame.join_where`).
    - :func:`closest` — ``join_by(closest(col("x") >= col("y")))``: rolling
      join (routes through :meth:`pl.DataFrame.join_asof`).
    - :func:`between` (already exported) — pass its return value directly
      for ``y_lower <= x <= y_upper`` predicates inside a join.
    - :func:`overlaps`, :func:`within` — interval-relationship predicates.

    Multiple arguments are conjoined. ``closest()`` may appear at most
    once and combines only with equi-key arguments.
    """
    spec = _JoinBy()
    for a in args:
        _consume_join_by_arg(spec, a)
    return spec


def _consume_join_by_arg(spec: _JoinBy, a: Any) -> None:
    if isinstance(a, str):
        spec.equi_left.append(a); spec.equi_right.append(a); return
    if isinstance(a, _Closest):
        if spec.asof is not None:
            raise ValueError(
                "join_by(): only one closest() condition is supported per call."
            )
        spec.asof = a; return
    if isinstance(a, pl.Expr):
        parsed = _parse_join_binary(a)
        if parsed is not None:
            op, L, R = parsed
            if op == "Eq":
                spec.equi_left.append(L); spec.equi_right.append(R); return
            spec.ineqs.append((op, L, R)); return
        spec.exprs.append(a); return
    raise TypeError(
        f"join_by(): unsupported arg type {type(a).__name__}. "
        "Pass strings, col() comparisons, closest(), between(), "
        "overlaps(), or within()."
    )


# Map "Lt"/"LtEq"/"Gt"/"GtEq" → corresponding Expr builder for non-equi
# predicates (used to reconstitute inequalities with suffixed right refs).
_INEQ_BUILDERS: dict[str, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    "Lt":   lambda l, r: l < r,
    "LtEq": lambda l, r: l <= r,
    "Gt":   lambda l, r: l > r,
    "GtEq": lambda l, r: l >= r,
}


def _numeric_supertype(a: Any, b: Any) -> Any | None:
    """Common numeric dtype that ``a`` and ``b`` both cast to, or ``None``.

    Mirrors dplyr's implicit coercion of numeric join keys: integer +
    float promotes to float; integer + integer promotes to the wider
    signed integer. Non-numeric mismatches (string vs date, etc.)
    return ``None`` and let polars raise its native ``SchemaError``.
    """
    if a == b:
        return a
    if not (a.is_numeric() and b.is_numeric()):
        return None
    if a.is_float() or b.is_float():
        return pl.Float64
    # Both integer kinds: promote to Int64 (widest signed). Hea doesn't
    # try to preserve unsigned-ness — dplyr/R doesn't have it either.
    return pl.Int64


def _align_equi_key_types(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast equi-join key columns to a common supertype where they mismatch.

    Polars rejects equi joins on mismatched dtypes; dplyr/R coerces
    numeric keys (e.g. ``flights.year: integer`` ↔ ``planes.year: integer``
    with ``NA`` works in R because integer is nullable). We auto-cast
    numeric pairs to a common supertype; non-numeric mismatches fall
    through and surface polars' native error.
    """
    left_cast: list[pl.Expr] = []
    right_cast: list[pl.Expr] = []
    for lk, rk in zip(left_keys, right_keys):
        lt = left.schema[lk]
        rt = right.schema[rk]
        if lt == rt:
            continue
        super_t = _numeric_supertype(lt, rt)
        if super_t is None:
            continue
        if lt != super_t:
            left_cast.append(pl.col(lk).cast(super_t))
        if rt != super_t:
            right_cast.append(pl.col(rk).cast(super_t))
    if left_cast:
        left = pl.DataFrame.with_columns(left, left_cast)
    if right_cast:
        right = pl.DataFrame.with_columns(right, right_cast)
    return left, right


def _emit_natural_join_message(shared: list[str]) -> None:
    """Print dplyr's ``Joining with `by = join_by(...)``` info message.

    Goes to stderr to match R's ``message()`` channel — visible in
    Jupyter and REPL output but doesn't pollute stdout-piped scripts.
    """
    import sys
    quoted = ", ".join(repr(c) for c in shared)
    print(f"Joining with `by = join_by({quoted})`", file=sys.stderr)


