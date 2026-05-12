"""R AST node types.

Frozen dataclasses, one per construct in the documented sublanguage. Each
node carries a ``span`` (start_byte, end_byte) into the original source so
error messages, gap entries, and round-trip preservation can quote the
original text.

The AST is intentionally close to R's surface syntax — we do not normalize
``x %>% f(., y)`` to ``f(x, y)`` here. That's the translator's job
(``r_to_py``); the AST stays faithful so we can also pretty-print it back
to R unchanged for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# (start_byte, end_byte) into the source string. Half-open: [start, end).
Span = tuple[int, int]


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NumLit:
    value: float
    span: Span


@dataclass(frozen=True, slots=True)
class IntLit:
    """Integer literal (``42L``)."""
    value: int
    span: Span


@dataclass(frozen=True, slots=True)
class ComplexLit:
    """Complex literal (``1i``, ``3.5i``)."""
    value: float  # the imaginary part
    span: Span


@dataclass(frozen=True, slots=True)
class StrLit:
    value: str
    span: Span
    # Whether this was a raw string (``r"(...)"``) — preserve for round-trip.
    raw: bool = False


@dataclass(frozen=True, slots=True)
class BoolLit:
    """``TRUE`` or ``FALSE``."""
    value: bool
    span: Span


@dataclass(frozen=True, slots=True)
class NullLit:
    """``NULL``."""
    span: Span


@dataclass(frozen=True, slots=True)
class NaLit:
    """``NA`` and its typed variants (``NA_integer_`` etc.)."""
    kind: str  # one of: "NA", "NA_integer_", "NA_real_", "NA_character_", "NA_complex_"
    span: Span


@dataclass(frozen=True, slots=True)
class InfLit:
    """``Inf``. Negative is represented as ``UnaryOp("-", InfLit(...))``."""
    span: Span


@dataclass(frozen=True, slots=True)
class NanLit:
    """``NaN``."""
    span: Span


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Identifier:
    """Bare identifier. ``backticked`` is True for `` `weird name` `` form."""
    name: str
    span: Span
    backticked: bool = False


# ---------------------------------------------------------------------------
# Compound expressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnaryOp:
    op: str  # "-", "+", "!", "~"
    operand: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class BinOp:
    """Binary expression. ``op`` is the literal operator text, e.g. ``+``,
    ``==``, ``&&``, ``%in%``, ``%any%``. Pipes have their own node type."""
    op: str
    left: "Node"
    right: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class Pipe:
    """Pipe expression: ``lhs |> rhs`` or ``lhs %>% rhs``.

    Distinguished from generic BinOp because the rhs has special semantics
    (it's a call into which lhs is threaded as the first argument; magrittr
    additionally honors ``.`` placeholder).
    """
    op: str  # "|>" or "%>%"
    lhs: "Node"
    rhs: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class Assign:
    """Assignment. ``op`` is the literal operator text used (``<-``, ``=``,
    ``<<-``, ``->``, ``->>``). For ``->``/``->>`` the parser flips so
    ``target`` is always on the LHS of the node regardless of source side."""
    op: str
    target: "Node"
    value: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class Tilde:
    """Formula: ``lhs ~ rhs`` or unary ``~ rhs``."""
    lhs: Optional["Node"]
    rhs: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class NamedArg:
    """Named argument inside a Call. ``name`` is the identifier text or
    string-literal text (R allows both ``f(x = 1)`` and ``f("x" = 1)``)."""
    name: str
    value: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class MissingArg:
    """An empty argument slot — ``f(, 2)`` (occurs in subscript expressions
    like ``df[, "a"]`` where the row position is intentionally empty)."""
    span: Span


@dataclass(frozen=True, slots=True)
class Call:
    """Function call: ``func(args...)``."""
    func: "Node"
    args: tuple["Node", ...]
    span: Span


@dataclass(frozen=True, slots=True)
class Subscript:
    """Single-bracket subscript: ``x[i]``, ``x[i, j]``. Multi-arg supported."""
    target: "Node"
    args: tuple["Node", ...]
    span: Span


@dataclass(frozen=True, slots=True)
class DoubleSubscript:
    """Double-bracket subscript: ``x[[i]]``."""
    target: "Node"
    args: tuple["Node", ...]
    span: Span


@dataclass(frozen=True, slots=True)
class Dollar:
    """Component access: ``x$name``. ``name`` is the literal identifier text."""
    target: "Node"
    name: str
    span: Span


@dataclass(frozen=True, slots=True)
class At:
    """Slot access: ``x@name``."""
    target: "Node"
    name: str
    span: Span


@dataclass(frozen=True, slots=True)
class FunctionDef:
    """``function(params) body`` and R 4.1+ ``\\(params) body`` lambda."""
    params: tuple["Param", ...]
    body: "Node"
    span: Span
    shorthand: bool = False  # True for ``\(x) ...``


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    default: Optional["Node"]
    span: Span


@dataclass(frozen=True, slots=True)
class If:
    cond: "Node"
    then: "Node"
    otherwise: Optional["Node"]
    span: Span


@dataclass(frozen=True, slots=True)
class For:
    var: str
    iterable: "Node"
    body: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class While:
    cond: "Node"
    body: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class Repeat:
    body: "Node"
    span: Span


@dataclass(frozen=True, slots=True)
class Break:
    span: Span


@dataclass(frozen=True, slots=True)
class Next:
    span: Span


@dataclass(frozen=True, slots=True)
class Block:
    """Brace block: ``{ stmt1; stmt2; ... }``. The value of the block is the
    value of its last statement."""
    statements: tuple["Node", ...]
    span: Span


@dataclass(frozen=True, slots=True)
class Program:
    """Top-level: sequence of statements (the whole parsed script)."""
    statements: tuple["Node", ...] = field(default_factory=tuple)
    span: Span = (0, 0)


# Union type alias for static checkers / docs.
Node = (
    NumLit | IntLit | ComplexLit | StrLit | BoolLit | NullLit | NaLit
    | InfLit | NanLit | Identifier | UnaryOp | BinOp | Pipe | Assign
    | Tilde | NamedArg | MissingArg | Call | Subscript | DoubleSubscript
    | Dollar | At | FunctionDef | If | For | While | Repeat | Break | Next
    | Block | Program
)
