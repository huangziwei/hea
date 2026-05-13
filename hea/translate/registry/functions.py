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
    "sd":       Func("sd",       "method"),   # hea.R.sd uses R's N-1 SD; polars .std() also has ddof=1 default — both fine
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
    # ``where(is.X)`` — tidyselect predicate selector. Bespoke handler
    # (``_emit_where_call``) maps known R-predicate identifiers to the
    # equivalent ``polars.selectors`` constructor.
    "where":        Func("__where__",              "function"),

    # ---- base R that maps directly to hea ----
    "c":        Func("__list__",   "function"),  # bespoke — emit as a list/Series
    "list":     Func("__list__",   "function"),
    "is.na":    Func("is_na",      "method"),    # R's is.na — element-wise NA
    "is.null":  Func("is_null",    "method"),    # R's is.null — scalar None check
    "range":    Func("R_range",    "function"),  # R range() collides with builtin; uses hea.R.R_range
    "round":    Func("R_round",    "function"),  # R round() needs to vectorize over dict/array; builtin only handles scalars
    "sessionInfo": Func("session_info", "function"),  # R's sessionInfo() -> hea.session_info()
    "is.finite": Func("is_finite", "method"),

    # ---- tibble / data.frame literal — bespoke handler ----
    # Translates to ``hea.DataFrame({"col": [values], ...})``. See
    # ``r_to_py._emit_data_frame_call`` for the implementation. The
    # ``__data_frame__`` marker steers the dispatch.
    "data.frame":     Func("__data_frame__", "function"),
    "tibble":         Func("__data_frame__", "function"),
    "data_frame":     Func("__data_frame__", "function"),
    "as_tibble":      Func("__data_frame__", "function"),
    "as.data.frame":  Func("__data_frame__", "function"),

    # tribble — row-form literal. ``tribble(~col, ~col, val, val, val, val)``
    # → reshape to column-major then dispatch to ``hea.DataFrame``. The
    # ``__tribble__`` marker steers the dispatch in ``_emit_call``.
    "tribble":        Func("__tribble__", "function"),

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


@dataclass(frozen=True, slots=True)
class KwargAlias:
    """How to translate a known R kwarg.

    - ``py_name`` is the Python kwarg name to emit (``.by`` → ``_by``).
    - ``value_slot`` is the NSE slot to push when translating the kwarg's
      VALUE. ``None`` means "inherit the surrounding context".

    Why this matters: ``mutate(x = ..., .by = origin)`` puts ``.by`` inside
    a verb whose default slot is EXPR — but ``.by`` semantically takes a
    column NAME (string), not a column expression. The override forces
    ``origin`` to render as ``"origin"`` rather than ``col("origin")``.
    """

    py_name: str
    value_slot: "Slot | None" = None


# Kwarg name aliases — R-side dotted form → (Python-side name, value slot).
# Looked up per call: any kwarg whose R name matches a key here gets the
# entry's settings. Unknown kwargs default to dot→underscore name with the
# surrounding slot inherited.
KWARG_ALIASES: dict[str, KwargAlias] = {
    # dplyr's dot-prefixed kwargs map to underscore-prefixed in hea, by
    # convention (so we don't collide with positional/expression args).
    ".by":       KwargAlias("_by",       Slot.COLUMN_NAME),
    ".keep":     KwargAlias("_keep",     Slot.NONE),         # string literal: "all"|"used"|"unused"|"none"
    ".before":   KwargAlias("_before",   Slot.COLUMN_NAME),  # col name or int position
    ".after":    KwargAlias("_after",    Slot.COLUMN_NAME),
    ".keep_all": KwargAlias("keep_all",  Slot.NONE),         # logical
    ".default":  KwargAlias("default",   None),              # case_when default; inherits context
    ".cols":     KwargAlias("cols",      Slot.COLUMN_NAME),  # across() / pivot_*
    ".fns":      KwargAlias("fns",       None),              # across() function list
    ".names":    KwargAlias("names",     Slot.NONE),         # across() name pattern
    ".names_sep": KwargAlias("names_sep", Slot.NONE),

    # pivot_* kwargs (no dot prefix in R). cols / id_cols / names_from /
    # values_from are column lists → COLUMN_NAME. Everything else is a
    # literal string / bool → Slot.NONE so we don't strip the quotes on
    # reverse-direction emission.
    "cols":          KwargAlias("cols",          Slot.COLUMN_NAME),
    "id_cols":       KwargAlias("id_cols",       Slot.COLUMN_NAME),
    "names_from":    KwargAlias("names_from",    Slot.COLUMN_NAME),
    "values_from":   KwargAlias("values_from",   Slot.COLUMN_NAME),
    "names_to":      KwargAlias("names_to",      Slot.NONE),
    "values_to":     KwargAlias("values_to",     Slot.NONE),
    "names_prefix":  KwargAlias("names_prefix",  Slot.NONE),
    "names_pattern": KwargAlias("names_pattern", Slot.NONE),
    "values_drop_na": KwargAlias("values_drop_na", Slot.NONE),
    "values_fill":   KwargAlias("values_fill",   Slot.NONE),
}


def resolve_kwarg(r_name: str) -> KwargAlias:
    """R kwarg name → :class:`KwargAlias`.

    Rules:
    1. Explicit entry in :data:`KWARG_ALIASES` wins.
    2. Otherwise, replace ``.`` with ``_`` in the name, append a trailing
       ``_`` if the result is a hard Python keyword (``lambda`` →
       ``lambda_``), and inherit the surrounding slot.

    Soft keywords (``match``, ``case``, ``type``) are left alone — they
    are valid kwarg names in any function.
    """
    hit = KWARG_ALIASES.get(r_name)
    if hit is not None:
        return hit
    import keyword

    py_name = r_name.replace(".", "_")
    if keyword.iskeyword(py_name):
        py_name = py_name + "_"
    return KwargAlias(py_name, None)


# Backwards-compat shim for existing import sites — returns just the name.
def normalize_kwarg_name(r_name: str) -> str:
    return resolve_kwarg(r_name).py_name
