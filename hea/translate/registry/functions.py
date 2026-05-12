"""R function → hea function/method mapping.

Each entry tells the translator three things:

- ``hea_name``  — the name to emit in Python.
- ``form``      — ``"method"``, ``"function"``, or ``"operator"``.
  - ``method``    → ``f(x, ...)`` becomes ``col("x").f(...)`` when in EXPR
    slot, else ``hea.f(x, ...)`` (function fallback).
  - ``function``  → always emit as ``hea.f(...)``; args may still be
    rewritten by the surrounding NSE slot.
  - ``operator``  → translated to a Python operator (not a call). Used
    for the handful of R functions that map to Python syntax (`!`, `&&`,
    etc — these usually arrive as BinOp / UnaryOp from the parser, but
    we register a few aliases like ``isTRUE`` / ``identity`` here).
- ``arg_slot``  — the slot type to use when visiting the function's args.
  ``None`` means "inherit from surrounding context", ``EXPR`` /
  ``COLUMN_NAME`` force a specific slot for nested rewriting.

This is also where kwarg-name aliasing lives — the R kwarg ``na.rm``
becomes the Python kwarg ``na_rm`` (and many similar dot-name kwargs
follow the same rule).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nse import Slot


@dataclass(frozen=True, slots=True)
class Func:
    hea_name: str
    form: str            # "method" | "function"
    arg_slot: Slot | None = None  # None = inherit parent slot


# fmt: off
FUNCTION_TABLE: dict[str, Func] = {
    # ---- aggregators / reductions (method form, .name() on a column) ----
    "mean":     Func("mean",     "method"),
    "median":   Func("median",   "method"),
    "sum":      Func("sum",      "method"),
    "prod":     Func("product",  "method"),
    "min":      Func("min",      "method"),
    "max":      Func("max",      "method"),
    "var":      Func("var",      "method"),
    "sd":       Func("std",      "method"),   # polars exposes .std(), not .sd()
    "IQR":      Func("IQR",      "function"), # IQR is a hea function, not a method
    "quantile": Func("quantile", "method"),
    "n":        Func("n",        "function", Slot.NONE),
    "n_distinct": Func("n_distinct", "function"),
    "length":   Func("len",      "function"),

    # ---- elementwise math (method form on the column) ----
    "log":      Func("log",      "method"),
    "log2":     Func("log",      "method"),   # arg=2 handled in emitter
    "log10":    Func("log10",    "method"),
    "exp":      Func("exp",      "method"),
    "sqrt":     Func("sqrt",     "method"),
    "abs":      Func("abs",      "method"),
    "round":    Func("round",    "method"),
    "floor":    Func("floor",    "method"),
    "ceiling":  Func("ceil",     "method"),
    "sign":     Func("sign",     "method"),

    # ---- dplyr helpers ----
    "desc":     Func("desc",     "function", Slot.COLUMN_NAME),
    "across":   Func("across",   "function", Slot.COLUMN_NAME),
    "if_else":  Func("if_else",  "function"),
    "ifelse":   Func("if_else",  "function"),
    "case_when": Func("case_when", "function"),  # bespoke handler — args are condition-value pairs via ~
    "coalesce": Func("coalesce", "function"),
    "na_if":    Func("na_if",    "function"),
    "between":  Func("between",  "function"),
    "near":     Func("near",     "function"),

    # ---- ranking / cumulative ----
    "row_number":  Func("row_number",  "function", Slot.NONE),
    "min_rank":    Func("min_rank",    "method"),
    "dense_rank":  Func("dense_rank",  "method"),
    "percent_rank": Func("percent_rank", "method"),
    "cume_dist":   Func("cume_dist",   "method"),
    "ntile":       Func("ntile",       "method"),
    "lag":         Func("lag",         "method"),
    "lead":        Func("lead",        "method"),
    "first":       Func("first",       "function"),
    "last":        Func("last",        "function"),
    "nth":         Func("nth",         "function"),
    "cumsum":      Func("cumsum",      "method"),
    "cumprod":     Func("cumprod",     "method"),
    "cummax":      Func("cummax",      "method"),
    "cummin":      Func("cummin",      "method"),
    "cummean":     Func("cummean",     "method"),

    # ---- tidy-select helpers ----
    "starts_with":  Func("selectors.starts_with",  "function"),
    "ends_with":    Func("selectors.ends_with",    "function"),
    "contains":     Func("selectors.contains",     "function"),
    "matches":      Func("selectors.matches",      "function"),
    "everything":   Func("selectors.all",          "function"),
    "all_of":       Func("all_of",                 "function"),
    "any_of":       Func("any_of",                 "function"),

    # ---- base R that maps directly to hea ----
    "c":        Func("__list__",   "function"),  # bespoke — emit as a list/Series
    "list":     Func("__list__",   "function"),
    "is.na":    Func("is_null",    "method"),    # R's is.na ~= polars .is_null()
    "is.null":  Func("is_null",    "method"),
    "is.finite": Func("is_finite", "method"),

    # ---- forcats ----
    "fct_infreq":      Func("fct_infreq",      "function"),
    "fct_relevel":     Func("fct_relevel",     "function"),
    "fct_recode":      Func("fct_recode",      "function"),
    "fct_collapse":    Func("fct_collapse",    "function"),
    "fct_lump_n":      Func("fct_lump_n",      "function"),
    "fct_lump_lowfreq": Func("fct_lump_lowfreq", "function"),
    "fct_reorder":     Func("fct_reorder",     "function"),
    "fct_reorder2":    Func("fct_reorder2",    "function"),
    "fct_rev":         Func("fct_rev",         "function"),
}
# fmt: on


# Kwarg name aliases — R-side dotted form → Python-side underscore form.
# Looked up per call: any kwarg whose R name matches a key here gets its
# Python name from the value. Unknown kwargs pass through with their dots
# stripped (``na.rm`` → ``na_rm`` is the universal rule, but specific
# overrides go here when the mapping isn't dot→underscore).
KWARG_ALIASES: dict[str, str] = {
    # dplyr's dot-prefixed kwargs map to underscore-prefixed in hea, by
    # convention (so we don't collide with positional/expression args).
    ".by":       "_by",
    ".keep":     "_keep",
    ".before":   "_before",
    ".after":    "_after",
    ".keep_all": "keep_all",
    ".default":  "default",
}


def normalize_kwarg_name(r_name: str) -> str:
    """R kwarg name → Python kwarg name.

    Rules, in order:
    1. Explicit alias in :data:`KWARG_ALIASES` wins.
    2. Otherwise, replace ``.`` with ``_`` (``na.rm`` → ``na_rm``).
    """
    if r_name in KWARG_ALIASES:
        return KWARG_ALIASES[r_name]
    return r_name.replace(".", "_")
