"""NSE slot context — tracks which kind of expression-slot we're inside
while emitting Python AST.

R's tidyverse uses Non-Standard Evaluation: ``mutate(y = x * 2)`` looks up
``x`` against the data frame. In hea/polars, that has to become an explicit
``col("x")`` (for filter/mutate/summarize) or a string ``"x"`` (for
select/group_by/count/distinct/arrange). The translator decides which by
asking "what verb-slot am I currently inside?".

Slots stack — a ``summarize(arr_delay = mean(arr_delay))`` puts the
arr_delay-inside-mean still in :data:`Slot.EXPR`, but inside a hypothetical
``mutate(x = c(a, b))`` the ``c(...)`` body would inherit the parent's
EXPR slot until something pushes a different one.

The context isn't a global — :class:`NSEContext` lives on the translator
instance.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum


class Slot(Enum):
    """What an identifier or call means at the current point.

    - ``NONE`` — outside any verb's NSE-bearing argument. Bare names are
      Python locals; calls are regular function calls.
    - ``EXPR`` — inside filter / mutate / summarize / arrange-via-desc /
      case_when conditions etc. Bare names become ``col("name")``; calls
      get method-form translation when their first arg is a column ref.
    - ``COLUMN_NAME`` — inside select / group_by / count / distinct /
      arrange (top-level) / slice_* col args. Bare names become the
      string ``"name"``; calls to tidy-select helpers pass through.
    """

    NONE = "none"
    EXPR = "expr"
    COLUMN_NAME = "column_name"


class NSEContext:
    """Stack of :class:`Slot` values. Top of stack is the current slot."""

    __slots__ = ("_stack",)

    def __init__(self):
        self._stack: list[Slot] = [Slot.NONE]

    @property
    def current(self) -> Slot:
        return self._stack[-1]

    def is_expr(self) -> bool:
        return self._stack[-1] is Slot.EXPR

    def is_column_name(self) -> bool:
        return self._stack[-1] is Slot.COLUMN_NAME

    def is_none(self) -> bool:
        return self._stack[-1] is Slot.NONE

    @contextmanager
    def enter(self, slot: Slot):
        """Push ``slot`` for the body, pop on exit. Use as
        ``with self.nse.enter(Slot.EXPR): ...``."""
        self._stack.append(slot)
        try:
            yield
        finally:
            self._stack.pop()
