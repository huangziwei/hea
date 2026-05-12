"""dplyr / tidyr verb table — R name → (hea method name, arg slot).

Verbs are recognized at the position they appear in a call: ``filter(df, ...)``
or after a pipe ``df |> filter(...)``. The translator rewrites both forms
to ``df.filter(...)``, and walks the remaining args with ``arg_slot`` so
NSE behaves the way each verb expects.

``slot`` is one of:
- :attr:`hea.translate.nse.Slot.EXPR` — bare identifiers become ``col("x")``.
- :attr:`hea.translate.nse.Slot.COLUMN_NAME` — bare identifiers become the
  literal string ``"x"``.

For verbs whose args are heterogenous (e.g. ``count(col, sort = TRUE)``),
the ``slot`` here is the **default** for bare/positional args; the
translator overrides per-kwarg via the keyword-name table in
:mod:`hea.translate.registry.kwargs` (which the function table also
consults).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nse import Slot


@dataclass(frozen=True, slots=True)
class Verb:
    hea_method: str  # method name on hea.DataFrame
    slot: Slot       # NSE behavior for the verb's expression args


# fmt: off
VERB_TABLE: dict[str, Verb] = {
    # EXPR-slot verbs — args are expressions over column refs.
    "filter":     Verb("filter",    Slot.EXPR),
    "mutate":     Verb("mutate",    Slot.EXPR),
    "transmute":  Verb("mutate",    Slot.EXPR),   # transmute = mutate(_keep="none")
    "summarize":  Verb("summarize", Slot.EXPR),
    "summarise":  Verb("summarize", Slot.EXPR),   # British spelling

    # COLUMN_NAME-slot verbs — args are column names.
    "select":     Verb("select",    Slot.COLUMN_NAME),
    "group_by":   Verb("group_by",  Slot.COLUMN_NAME),
    "count":      Verb("count",     Slot.COLUMN_NAME),
    "add_count":  Verb("count",     Slot.COLUMN_NAME),
    "distinct":   Verb("distinct",  Slot.COLUMN_NAME),
    "arrange":    Verb("arrange",   Slot.COLUMN_NAME),
    "rename":     Verb("rename",    Slot.COLUMN_NAME),
    "relocate":   Verb("relocate",  Slot.COLUMN_NAME),
    "pull":       Verb("pull",      Slot.COLUMN_NAME),

    # Stateless verbs — no NSE.
    "ungroup":    Verb("ungroup",   Slot.NONE),
    "glimpse":    Verb("glimpse",   Slot.NONE),
}
# fmt: on


# Verbs whose first positional arg is the data frame (i.e. eligible for
# pipe rewriting). All entries in VERB_TABLE currently fit this — kept as
# a separate frozenset for explicitness and to make future divergence easy.
DATAFRAME_VERBS: frozenset[str] = frozenset(VERB_TABLE.keys())
