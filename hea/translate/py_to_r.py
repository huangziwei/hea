"""Python AST → R source string.

The inverse of :mod:`hea.translate.r_to_py`. Emitted as a string directly
(no R-AST builder available from Python). Pretty-printing is minimal: one
statement per line, two-space indentation inside pipes / brace blocks.

Symmetric scope to the forward translator — covers tidyverse verbs, the
function-helper registry, joins, pivot, case_when (with tuple-pair
inversion to ``~`` syntax), patchwork composition, and ggplot fluent
chains.

The forward direction is necessarily lossy in a few places — see
``.claude/plans/r-translator.md`` §3.x for what's expected to round-trip
exactly. Notable losses:

- ``mutate(across(c(a, b), mean))`` expands forward to two NamedArgs and
  cannot reliably re-collapse into ``across()``. Reverse emits the
  expanded form.
- ``transmute`` forward becomes ``mutate(..., _keep="none")``. Reverse
  emits ``mutate(..., .keep = "none")``, not ``transmute(...)``.
- ``1.0`` written explicitly in R forward-translates to int ``1`` (per
  the integer-rendering rule) and reverse-emits as ``1`` — losing the
  explicit-decimal mark.
"""

from __future__ import annotations

import ast as P
import functools
import re
from typing import Optional

from .nse import NSEContext, Slot
from .registry.functions import FUNCTION_TABLE, KWARG_ALIASES, Func
from .registry.ggplot import is_chain_extension
from .registry.verbs import VERB_TABLE, Verb


class PyToRError(Exception):
    """Raised when a Python AST node can't be reversed within the
    documented sublanguage."""


# ---------------------------------------------------------------------------
# Inverse-direction registries (built at module load).
# ---------------------------------------------------------------------------


def _build_inverse_verbs() -> dict[str, str]:
    """``hea_method`` → canonical R name. Canonical = the American /
    primary form when multiple R names map to the same hea method.
    """
    # Default: first-seen R name wins.
    inverse: dict[str, str] = {}
    for r_name, verb in VERB_TABLE.items():
        inverse.setdefault(verb.hea_method, r_name)

    # Explicit canonicals — supersede whatever first-seen happened to pick.
    canonical = {
        "mutate":    "mutate",     # NOT transmute (auto-kwargs encode that)
        "summarize": "summarize",  # NOT summarise
    }
    for hea_method, r_name in canonical.items():
        if r_name in VERB_TABLE:
            inverse[hea_method] = r_name
    return inverse


def _build_inverse_functions() -> dict[str, str]:
    """``hea_name`` → canonical R name."""
    inverse: dict[str, str] = {}
    for r_name, func in FUNCTION_TABLE.items():
        # Bespoke markers like "__list__" map back to c() explicitly.
        if func.hea_name == "__list__":
            continue
        inverse.setdefault(func.hea_name, r_name)
    # Canonical overrides.
    canonical = {
        "if_else":  "if_else",   # NOT ifelse
        "is_null":  "is.na",     # forward maps both is.na and is.null to is_null
    }
    for hea_name, r_name in canonical.items():
        if r_name in FUNCTION_TABLE:
            inverse[hea_name] = r_name
    # Special: __list__ is the marker for c() / list() — reverse always c().
    inverse["__list__"] = "c"
    return inverse


def _build_function_arg_slots() -> dict[str, Slot]:
    """``hea_name`` → arg_slot to push when emitting this helper's args
    in reverse. Mirrors :attr:`Func.arg_slot` from the forward table so
    ``desc("x")`` round-trips to ``desc(x)`` (COLUMN_NAME unwrap).
    """
    out: dict[str, Slot] = {}
    for func in FUNCTION_TABLE.values():
        if func.arg_slot is not None and func.hea_name not in out:
            out[func.hea_name] = func.arg_slot
    return out


def _build_inverse_kwargs() -> dict[str, str]:
    """Python kwarg name → R kwarg name.

    Built from KWARG_ALIASES (the user-defined map), plus the universal
    rule ``na_rm`` → ``na.rm`` (which the forward direction handles via
    its dot→underscore default).
    """
    inverse: dict[str, str] = {}
    for r_name, alias in KWARG_ALIASES.items():
        inverse[alias.py_name] = r_name
    # Universal additions — Python names that always reverse to dotted R.
    universal = {
        "na_rm":     "na.rm",
        "keep_all":  ".keep_all",  # already in KWARG_ALIASES but double-check
    }
    inverse.update(universal)
    return inverse


_HEA_METHOD_TO_R: dict[str, str] = _build_inverse_verbs()
_HEA_FN_TO_R: dict[str, str] = _build_inverse_functions()
_HEA_FN_ARG_SLOTS: dict[str, Slot] = _build_function_arg_slots()
_PY_KWARG_TO_R: dict[str, str] = _build_inverse_kwargs()


# Aesthetic kwarg names — reverse direction uses this set to decide
# whether a geom kwarg gets wrapped in ``aes(...)``. Sourced from
# ``hea/ggplot/aes.py`` (kept in sync; if hea adds aesthetics we may
# under-detect in reverse, which is a benign gap not a crash).
_AESTHETIC_NAMES: frozenset[str] = frozenset({
    "x", "y", "z",
    "xmin", "xmax", "ymin", "ymax",
    "xend", "yend",
    "xintercept", "yintercept", "slope", "intercept",
    "colour", "color", "fill", "alpha", "size", "shape",
    "linetype", "linewidth", "stroke",
    "label", "family", "fontface", "hjust", "vjust", "angle", "lineheight",
    "group", "weight",
    "lower", "middle", "upper",
})


# ---------------------------------------------------------------------------
# Operator emission tables
# ---------------------------------------------------------------------------


_PY_BIN_TO_R = {
    P.Add:      "+",
    P.Sub:      "-",
    P.Mult:     "*",
    P.Div:      "/",
    P.FloorDiv: "%/%",
    P.Mod:      "%%",
    P.Pow:      "^",
    P.BitAnd:   "&",
    P.BitOr:    "|",
}

_PY_CMP_TO_R = {
    P.Eq:    "==",
    P.NotEq: "!=",
    P.Lt:    "<",
    P.LtE:   "<=",
    P.Gt:    ">",
    P.GtE:   ">=",
    P.In:    "%in%",
    P.NotIn: "%!in%",  # not idiomatic R but the closest direct translation
}

_PY_UNARY_TO_R = {
    P.UAdd:   "+",
    P.USub:   "-",
    P.Not:    "!",
    P.Invert: "!",
}

_PY_BOOL_TO_R = {
    P.And: "&&",
    P.Or:  "||",
}


# Operator precedence — lower number = tighter binding. We track this so
# we can decide when to parenthesize a sub-expression. R's precedence
# table mirrors :data:`hea.translate.r_parser._LBP` but with our own
# tightness encoding (higher = looser).
_PREC = {
    "::": 1, ":::": 1,
    "$":  2, "@": 2,
    "[":  3, "[[": 3,
    "^":  4,
    "u-": 5, "u+": 5,
    ":":  6,
    "%in%": 7, "%%": 7, "%/%": 7, "|>": 7,
    "*":  8, "/": 8,
    "+":  9, "-": 9,
    "<": 10, "<=": 10, ">": 10, ">=": 10, "==": 10, "!=": 10,
    "!": 11,
    "&": 12, "&&": 12,
    "|": 13, "||": 13,
    "~": 14,
    "<-": 15, "=": 15,
}


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def translate(src: str) -> str:
    """Translate Python source to R source."""
    module = P.parse(src)
    return Translator().translate(module)


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class Translator:
    """Stateful walker. One instance per translation."""

    def __init__(self):
        self.nse = NSEContext()
        # Packages discovered during translation that need a top-level
        # ``library()`` call in the preamble. Populated by
        # :meth:`_maybe_smart_data_assign`; merged with auto-detected
        # tidyverse / patchwork usage in :meth:`translate`.
        self._extra_libs: set[str] = set()

    # -- top-level --------------------------------------------------------

    def translate(self, module: P.Module) -> str:
        # Detect autoloaded-dataset references before emitting so any
        # uniquely-resolved package gets a library() in the preamble. The
        # heuristic: bare names that are referenced but never bound in
        # this script — those are the hea autoload candidates.
        for name, pkg in _collect_dataset_refs(module):
            self._extra_libs.add(pkg)

        out: list[str] = []
        for s in module.body:
            text = self._emit_stmt(s)
            if text:  # imports / no-ops return ""
                out.append(text)
        body = "\n".join(out)

        preamble = _build_preamble(body, self._extra_libs)
        if preamble:
            return "\n".join(preamble) + "\n" + body
        return body

    # -- statements -------------------------------------------------------

    def _emit_stmt(self, stmt: P.stmt) -> str:
        # Drop ``import`` / ``from X import Y`` — Python's import machinery
        # has no clean R analog (R uses ``library()``, which is a runtime
        # call). The hea side's translated output is run in a namespace
        # that already has the relevant names available.
        if isinstance(stmt, (P.Import, P.ImportFrom)):
            return ""
        if isinstance(stmt, P.Assign):
            # Smart rewrite: ``X = data("X", package="pkg")`` →
            # ``library(pkg)``. The literal translation would assign the
            # string ``"X"`` to the variable (R's data() returns a name
            # string, not the dataset), breaking downstream uses.
            smart = self._maybe_smart_data_assign(stmt)
            if smart is not None:
                return smart
            target = self._emit_expr(stmt.targets[0], prec=20)
            value = self._emit_expr(stmt.value, prec=20)
            return f"{target} <- {value}"
        if isinstance(stmt, P.Expr):
            return self._emit_expr(stmt.value, prec=20)
        if isinstance(stmt, P.If):
            return self._emit_if_stmt(stmt)
        if isinstance(stmt, P.For):
            return self._emit_for_stmt(stmt)
        if isinstance(stmt, P.While):
            cond = self._emit_expr(stmt.test, prec=20)
            body = self._emit_block(stmt.body)
            return f"while ({cond}) {body}"
        if isinstance(stmt, P.Break):
            return "break"
        if isinstance(stmt, P.Continue):
            return "next"
        if isinstance(stmt, P.Return):
            value = self._emit_expr(stmt.value, prec=20) if stmt.value else "NULL"
            return f"return({value})"
        if isinstance(stmt, P.Pass):
            return "invisible(NULL)"
        if isinstance(stmt, P.FunctionDef):
            return self._emit_function_def(stmt)
        raise PyToRError(f"unsupported statement: {type(stmt).__name__}")

    def _maybe_smart_data_assign(self, stmt: P.Assign) -> Optional[str]:
        """Detect ``<X> = data("<X>", package="<pkg>")`` and reverse to
        ``library(<pkg>)``.

        R's ``data()`` is side-effectful — it loads a dataset by name
        into the calling environment and returns the name string. Naive
        translation of the Python assignment would bind the string to
        the variable, breaking the rest of the script. ``library(pkg)``
        loads the dataset package and makes the name available, which is
        what the user actually wants.

        Returns ``None`` if the pattern doesn't match (any deviation —
        multi-target, non-Name target, mismatched names, missing
        ``package`` kwarg — falls back to the generic Assign emitter).
        """
        if len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
        if not isinstance(target, P.Name):
            return None
        value = stmt.value
        if not isinstance(value, P.Call):
            return None
        # Accept both bare ``data(...)`` and ``hea.data(...)``.
        func = value.func
        if isinstance(func, P.Name) and func.id == "data":
            pass
        elif (
            isinstance(func, P.Attribute)
            and isinstance(func.value, P.Name)
            and func.value.id == "hea"
            and func.attr == "data"
        ):
            pass
        else:
            return None
        # First positional arg must be a string matching the LHS name.
        if not value.args or not isinstance(value.args[0], P.Constant):
            return None
        if value.args[0].value != target.id:
            return None
        # ``package`` kwarg — required for this rewrite. Without it, R
        # would need to know where to find the dataset, and we'd be
        # guessing.
        pkg = None
        for kw in value.keywords:
            if kw.arg == "package" and isinstance(kw.value, P.Constant) and isinstance(kw.value.value, str):
                pkg = kw.value.value
        if pkg is None:
            return None
        # Stage in the preamble (along with any auto-detected libs)
        # instead of emitting inline — keeps every ``library()`` call at
        # the top of the script, matching the standard R idiom.
        self._extra_libs.add(pkg)
        return ""

    def _emit_if_stmt(self, stmt: P.If) -> str:
        cond = self._emit_expr(stmt.test, prec=20)
        body = self._emit_block(stmt.body)
        out = f"if ({cond}) {body}"
        if stmt.orelse:
            if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], P.If):
                # ``elif`` chain
                else_text = self._emit_if_stmt(stmt.orelse[0])
                out += f" else {else_text}"
            else:
                else_body = self._emit_block(stmt.orelse)
                out += f" else {else_body}"
        return out

    def _emit_for_stmt(self, stmt: P.For) -> str:
        target = self._emit_expr(stmt.target, prec=20)
        iterable = self._emit_expr(stmt.iter, prec=20)
        body = self._emit_block(stmt.body)
        return f"for ({target} in {iterable}) {body}"

    def _emit_block(self, body: list[P.stmt]) -> str:
        if len(body) == 1:
            return self._emit_stmt(body[0])
        lines = [self._emit_stmt(s) for s in body]
        inner = "\n  ".join(lines)
        return "{\n  " + inner + "\n}"

    def _emit_function_def(self, stmt: P.FunctionDef) -> str:
        params = self._emit_params(stmt.args)
        body = self._emit_block(stmt.body)
        return f"{stmt.name} <- function({params}) {body}"

    def _emit_params(self, args: P.arguments) -> str:
        # Combine positional args + defaults.
        n_args = len(args.args)
        n_defaults = len(args.defaults)
        defaults_offset = n_args - n_defaults
        parts = []
        for i, arg in enumerate(args.args):
            if i >= defaults_offset:
                default = args.defaults[i - defaults_offset]
                parts.append(f"{arg.arg} = {self._emit_expr(default, prec=20)}")
            else:
                parts.append(arg.arg)
        return ", ".join(parts)

    # -- expressions ------------------------------------------------------

    def _emit_expr(self, expr: P.expr, *, prec: int) -> str:
        """Emit an expression with its outer-context precedence ``prec``.

        We wrap the result in parens iff the expression's own precedence
        is looser than ``prec``. Smaller numbers bind tighter.
        """
        if isinstance(expr, P.Constant):
            return self._emit_constant(expr.value)
        if isinstance(expr, P.Name):
            return self._emit_name(expr)
        if isinstance(expr, P.List):
            return self._emit_c([self._emit_expr(e, prec=20) for e in expr.elts])
        if isinstance(expr, P.Tuple):
            # In Python AST, Tuples appear inside case_when args. We
            # handle that bespoke in _emit_case_when; bare tuples don't
            # show up here for translatable shapes.
            return self._emit_c([self._emit_expr(e, prec=20) for e in expr.elts])
        if isinstance(expr, P.Dict):
            return self._emit_c_dict(expr.keys, expr.values)
        if isinstance(expr, P.Set):
            return self._emit_c([self._emit_expr(e, prec=20) for e in expr.elts])
        if isinstance(expr, P.BinOp):
            return self._emit_binop(expr, prec)
        if isinstance(expr, P.UnaryOp):
            return self._emit_unaryop(expr, prec)
        if isinstance(expr, P.BoolOp):
            return self._emit_boolop(expr, prec)
        if isinstance(expr, P.Compare):
            return self._emit_compare(expr, prec)
        if isinstance(expr, P.Call):
            return self._emit_call(expr)
        if isinstance(expr, P.Attribute):
            return self._emit_attribute(expr)
        if isinstance(expr, P.Subscript):
            return self._emit_subscript(expr)
        if isinstance(expr, P.IfExp):
            return self._emit_ifexp(expr, prec)
        if isinstance(expr, P.Lambda):
            return self._emit_lambda(expr)
        if isinstance(expr, P.Starred):
            # ``*args`` rare in our scope. Emit as-is for visibility.
            return f"...{self._emit_expr(expr.value, prec=20)}"
        raise PyToRError(f"unsupported expression: {type(expr).__name__}")

    # -- atoms ------------------------------------------------------------

    def _emit_constant(self, value) -> str:
        if value is None:
            return "NULL"
        if value is True:
            return "TRUE"
        if value is False:
            return "FALSE"
        if isinstance(value, bool):  # redundant but explicit (True/False matched above)
            return "TRUE" if value else "FALSE"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value != value:  # NaN
                return "NaN"
            if value == float("inf"):
                return "Inf"
            if value == float("-inf"):
                return "-Inf"
            # Avoid scientific notation for clean output; use repr otherwise.
            if value.is_integer() and abs(value) < 1e16:
                return f"{value:.1f}"
            return repr(value)
        if isinstance(value, complex):
            return f"{value.imag}i"  # assumes pure imag
        if isinstance(value, str):
            return _quote_string(value)
        raise PyToRError(f"unsupported constant type: {type(value).__name__}")

    def _emit_name(self, name: P.Name) -> str:
        # NSE: in EXPR slot, a bare Python name is unusual (col-wrapping
        # is the convention) but possible. Emit as-is.
        if self.nse.is_column_name():
            # In COLUMN_NAME slot a bare name reverses to itself — but we
            # shouldn't get here in normal usage (col names should be
            # strings).
            return name.id
        return name.id

    # -- collection literals ----------------------------------------------

    def _emit_c(self, parts: list[str]) -> str:
        if not parts:
            return "c()"
        return f"c({', '.join(parts)})"

    def _emit_c_dict(self, keys: list, values: list) -> str:
        parts = []
        for k, v in zip(keys, values):
            k_str = self._emit_expr(k, prec=20) if k is not None else "NULL"
            v_str = self._emit_expr(v, prec=20)
            parts.append(f"{k_str} = {v_str}")
        return f"c({', '.join(parts)})"

    # -- operators --------------------------------------------------------

    def _emit_binop(self, expr: P.BinOp, outer_prec: int) -> str:
        op = _PY_BIN_TO_R.get(type(expr.op))
        if op is None:
            raise PyToRError(f"unsupported binop: {type(expr.op).__name__}")
        my_prec = _PREC.get(op, 20)
        # Right side of certain right-associative ops needs paren if same prec.
        left = self._emit_expr(expr.left, prec=my_prec)
        right = self._emit_expr(expr.right, prec=my_prec - 1 if op == "^" else my_prec)
        text = f"{left} {op} {right}"
        return _maybe_paren(text, my_prec, outer_prec)

    def _emit_unaryop(self, expr: P.UnaryOp, outer_prec: int) -> str:
        op = _PY_UNARY_TO_R.get(type(expr.op))
        if op is None:
            raise PyToRError(f"unsupported unary: {type(expr.op).__name__}")
        my_prec = 5
        operand = self._emit_expr(expr.operand, prec=my_prec)
        text = f"{op}{operand}"
        return _maybe_paren(text, my_prec, outer_prec)

    def _emit_boolop(self, expr: P.BoolOp, outer_prec: int) -> str:
        op = _PY_BOOL_TO_R[type(expr.op)]
        my_prec = _PREC[op]
        parts = [self._emit_expr(v, prec=my_prec) for v in expr.values]
        text = f" {op} ".join(parts)
        return _maybe_paren(text, my_prec, outer_prec)

    def _emit_compare(self, expr: P.Compare, outer_prec: int) -> str:
        # R doesn't support chained comparison; we serialize as left-assoc &&.
        my_prec = 10
        left = self._emit_expr(expr.left, prec=my_prec)
        pieces = []
        for op, right in zip(expr.ops, expr.comparators):
            r_op = _PY_CMP_TO_R.get(type(op))
            if r_op is None:
                raise PyToRError(f"unsupported comparison: {type(op).__name__}")
            right_str = self._emit_expr(right, prec=my_prec)
            if r_op == "%in%":
                # ``a %in% b`` — not a comparison in Python terms but
                # we represent it as one for translation symmetry.
                pieces.append((r_op, right_str))
            else:
                pieces.append((r_op, right_str))
        if len(pieces) == 1:
            r_op, right_str = pieces[0]
            text = f"{left} {r_op} {right_str}"
        else:
            # Chained: a < b < c → (a < b) & (b < c)
            text_parts: list[str] = []
            prev_left = left
            for r_op, right_str in pieces:
                text_parts.append(f"{prev_left} {r_op} {right_str}")
                prev_left = right_str
            text = " & ".join(f"({p})" for p in text_parts)
        return _maybe_paren(text, my_prec, outer_prec)

    def _emit_ifexp(self, expr: P.IfExp, outer_prec: int) -> str:
        # Python ``a if cond else b`` → R ``if (cond) a else b``.
        cond = self._emit_expr(expr.test, prec=20)
        body = self._emit_expr(expr.body, prec=20)
        orelse = self._emit_expr(expr.orelse, prec=20)
        text = f"if ({cond}) {body} else {orelse}"
        return _maybe_paren(text, 16, outer_prec)

    def _emit_lambda(self, expr: P.Lambda) -> str:
        params = self._emit_params(expr.args)
        body = self._emit_expr(expr.body, prec=20)
        return f"function({params}) {body}"

    # -- subscript / attribute -------------------------------------------

    def _emit_subscript(self, expr: P.Subscript) -> str:
        target = self._emit_expr(expr.value, prec=3)
        slice_ = expr.slice
        # df[i] / df[i, j]
        if isinstance(slice_, P.Tuple):
            parts = [self._emit_expr(e, prec=20) for e in slice_.elts]
            return f"{target}[{', '.join(parts)}]"
        return f"{target}[{self._emit_expr(slice_, prec=20)}]"

    def _emit_attribute(self, expr: P.Attribute) -> str:
        # ``df.col`` standalone — translate to ``df$col``. Inside Call
        # this gets intercepted by _emit_call's chain detection.
        target = self._emit_expr(expr.value, prec=2)
        return f"{target}${expr.attr}"

    # -- calls ------------------------------------------------------------

    def _emit_call(self, call: P.Call) -> str:
        """Four dispatch layers:

        1. **Method-form helper on an expression** — ``col("x").mean()`` /
           ``(...).sd()`` etc. Reverse to ``mean(x, ...)`` / ``sd(...)``.
           Must beat chain detection because ``mean`` isn't a verb.
        2. **Method chain on something** — walk the attribute chain.
           If any method is a known verb or ggplot extension, emit as
           ``|>`` pipe / ``+`` chain.
        3. **Known helper function** — ``col("x")``, ``case_when(...)``,
           ``if_else(...)``. Translate by registry inverse.
        4. **Plain function call** — emit ``func(args)``.
        """
        # 1) Method-form helper inversion.
        if isinstance(call.func, P.Attribute):
            method_form = self._maybe_emit_method_form_helper(call)
            if method_form is not None:
                return method_form

        # 2) Method-chain detection.
        if isinstance(call.func, P.Attribute):
            base, chain = _flatten_method_chain(call)
            if _is_translation_chain(chain):
                return self._emit_chain(base, chain)

        # 3) Helper-function calls.
        if isinstance(call.func, P.Name):
            name = call.func.id
            r = self._maybe_emit_helper(name, call)
            if r is not None:
                return r

        # ``hea.DataFrame({...})`` → ``data.frame(...)`` — the only hea-
        # qualified call that needs a name change (everything else just
        # has the prefix stripped).
        if (
            isinstance(call.func, P.Attribute)
            and isinstance(call.func.value, P.Name)
            and call.func.value.id == "hea"
            and call.func.attr in ("DataFrame", "from_dict")
        ):
            return self._emit_data_frame_reverse(call.args, call.keywords)

        # ``hea.X(...)`` / ``selectors.X(...)`` → ``X(...)``. Both
        # namespaces are stripped — R uses bare names for the same
        # functions (via the dplyr / tidyselect imports).
        if (
            isinstance(call.func, P.Attribute)
            and isinstance(call.func.value, P.Name)
            and call.func.value.id in ("hea", "selectors")
        ):
            return self._emit_plain_call(call.func.attr, call.args, call.keywords)

        # 4) Fallback: plain call.
        if isinstance(call.func, P.Name):
            return self._emit_plain_call(call.func.id, call.args, call.keywords)
        # Callee is some expression — emit as ``(expr)(args)``.
        callee = self._emit_expr(call.func, prec=2)
        args_text = self._emit_args(call.args, call.keywords)
        return f"({callee})({args_text})"

    def _maybe_emit_method_form_helper(self, call: P.Call) -> Optional[str]:
        """If ``call`` is ``<expr>.method(args)`` where ``method`` is a
        registered method-form helper (e.g. ``col("x").mean()``), reverse
        to ``r_func(<expr>, args)``. Returns ``None`` if this pattern
        doesn't apply.

        Explicitly **not** matched: the namespace-qualified forms like
        ``hea.mean(x)`` and ``selectors.starts_with("wk")``. Those are
        handled by separate paths in :meth:`_emit_call`.
        """
        attr = call.func
        if not isinstance(attr, P.Attribute):
            return None
        # Skip namespace-qualified calls — ``hea.X`` / ``selectors.X``
        # are NOT method-form helpers on a column expression.
        if isinstance(attr.value, P.Name) and attr.value.id in ("hea", "selectors"):
            return None
        method_name = attr.attr

        # %in% — ``<expr>.is_in(rhs)`` → ``<expr> %in% rhs``. The forward
        # direction turns ``a %in% b`` into ``col("a").is_in(b)``; this is
        # the symmetric inverse.
        if method_name == "is_in" and len(call.args) == 1 and not call.keywords:
            left = self._emit_expr(attr.value, prec=7)
            right = self._emit_expr(call.args[0], prec=7)
            return f"{left} %in% {right}"

        # Only reverse the FUNCTION_TABLE method-form helpers — verb
        # methods (filter, mutate, etc.) are handled by the chain path.
        if method_name in _HEA_METHOD_TO_R:
            return None
        r_name = _HEA_FN_TO_R.get(method_name)
        if r_name is None:
            return None
        # Build the function-form call: r_name(<receiver>, *args, **kwargs).
        # The receiver and remaining args inherit whatever NSE slot the
        # caller pushed (which is what makes ``col("x").mean()`` inside
        # a summarize EXPR slot reverse to ``mean(x)`` cleanly).
        receiver = self._emit_expr(attr.value, prec=20)
        rest = self._emit_args(call.args, call.keywords)
        if rest:
            return f"{r_name}({receiver}, {rest})"
        return f"{r_name}({receiver})"

    def _emit_plain_call(self, name: str, args: list, kwargs: list) -> str:
        """Plain function call ``name(args)`` with no NSE rewriting."""
        return f"{name}({self._emit_args(args, kwargs)})"

    def _emit_data_frame_reverse(self, args: list, kwargs: list) -> str:
        """``hea.DataFrame({"a": [1, 2], "b": [3, 4]})`` → ``data.frame(a = c(1, 2), b = c(3, 4))``.

        Also accepts ``hea.DataFrame(a=[1, 2], b=[3, 4])`` (Python kwarg form)
        and ``hea.from_dict({...})``. Other shapes (e.g. list-of-dicts,
        polars Series args) fall back to a bare ``data.frame(...)`` call
        with the original args; the user gets readable but possibly-wrong R.
        """
        parts: list[str] = []
        # Dict literal positional: unpack keys/values as R named args.
        if len(args) == 1 and not kwargs and isinstance(args[0], P.Dict):
            d = args[0]
            for k, v in zip(d.keys, d.values):
                if isinstance(k, P.Constant) and isinstance(k.value, str):
                    key_text = k.value
                else:
                    key_text = self._emit_expr(k, prec=20) if k is not None else "NULL"
                parts.append(f"{key_text} = {self._emit_expr(v, prec=20)}")
            return f"data.frame({', '.join(parts)})"
        # Python kwarg form.
        if not args and kwargs:
            for kw in kwargs:
                parts.append(f"{kw.arg} = {self._emit_expr(kw.value, prec=20)}")
            return f"data.frame({', '.join(parts)})"
        # Fallback — let the user figure out the shape mismatch.
        return self._emit_plain_call("data.frame", args, kwargs)

    def _emit_args(self, args: list, kwargs: list) -> str:
        parts: list[str] = []
        for a in args:
            parts.append(self._emit_expr(a, prec=20))
        for kw in kwargs:
            value = self._emit_expr(kw.value, prec=20)
            r_name = _PY_KWARG_TO_R.get(kw.arg, kw.arg)
            parts.append(f"{r_name} = {value}")
        return ", ".join(parts)

    # ---- helper dispatch ------------------------------------------------

    def _maybe_emit_helper(self, name: str, call: P.Call) -> Optional[str]:
        """If ``name`` is a known hea helper, emit the R-side version.
        Returns ``None`` if ``name`` isn't a registered helper.

        Helpers with a registered :attr:`Func.arg_slot` (e.g. ``desc``
        pushes COLUMN_NAME) have that slot pushed for their args so the
        forward direction's NSE rewrites reverse cleanly: forward
        ``desc(x)`` → ``desc("x")`` → reverse ``desc(x)``.
        """
        if name == "col":
            return self._emit_col_unwrap(call)
        if name == "case_when":
            return self._emit_case_when(call)
        if name == "if_else":
            return self._emit_helper_call("if_else", call.args, call.keywords, name)
        # Generic hea-helper → R name lookup.
        r_name = _HEA_FN_TO_R.get(name)
        if r_name is None:
            return None
        if r_name == "c":
            # __list__ marker → c()
            return self._emit_helper_call("c", call.args, call.keywords, name)
        return self._emit_helper_call(r_name, call.args, call.keywords, name)

    def _emit_helper_call(self, r_name: str, args: list, kwargs: list, hea_name: str) -> str:
        """Emit a helper call, pushing the registered arg_slot (if any)
        for the duration of arg translation."""
        slot = _HEA_FN_ARG_SLOTS.get(hea_name)
        if slot is not None:
            with self.nse.enter(slot):
                args_text = self._emit_args_with_slot(args, kwargs, slot)
        else:
            args_text = self._emit_args(args, kwargs)
        return f"{r_name}({args_text})"

    def _emit_args_with_slot(self, args: list, kwargs: list, slot: Slot) -> str:
        """Emit args under a specific NSE slot — used for helper calls
        whose registry entry forces a non-inherited slot."""
        parts: list[str] = []
        for a in args:
            parts.append(self._emit_verb_value(a, slot))
        for kw in kwargs:
            kw_slot = self._kwarg_slot(kw.arg, slot)
            value_str = self._emit_verb_value(kw.value, kw_slot)
            r_name = _PY_KWARG_TO_R.get(kw.arg, kw.arg)
            parts.append(f"{r_name} = {value_str}")
        return ", ".join(parts)

    def _emit_col_unwrap(self, call: P.Call) -> str:
        """``col("x")`` → bare ``x``. Single string arg, no kwargs."""
        if len(call.args) == 1 and not call.keywords and isinstance(call.args[0], P.Constant) and isinstance(call.args[0].value, str):
            return call.args[0].value
        # Unusual shapes — keep as a function call.
        return self._emit_plain_call("col", call.args, call.keywords)

    def _emit_case_when(self, call: P.Call) -> str:
        """``case_when((c, v), (c, v), default=d)`` →
        ``case_when(c ~ v, c ~ v, .default = d)``."""
        parts: list[str] = []
        for arg in call.args:
            if isinstance(arg, P.Tuple) and len(arg.elts) == 2:
                cond = self._emit_expr(arg.elts[0], prec=20)
                value = self._emit_expr(arg.elts[1], prec=20)
                parts.append(f"{cond} ~ {value}")
            else:
                # Non-tuple positional — keep as-is.
                parts.append(self._emit_expr(arg, prec=20))
        for kw in call.keywords:
            r_name = _PY_KWARG_TO_R.get(kw.arg, kw.arg)
            value = self._emit_expr(kw.value, prec=20)
            parts.append(f"{r_name} = {value}")
        return f"case_when({', '.join(parts)})"

    # ---- chain emission ------------------------------------------------

    def _emit_chain(self, base: P.expr, chain: list[tuple[str, list, list]]) -> str:
        """Emit a method chain as a mix of ``|>`` pipes and ``+`` ggplot
        composition. The chain walks left-to-right; once a ggplot
        construction starts (any ``.ggplot()`` call or chain extension),
        the rest of the chain composes with ``+``.

        ``chain`` is a list of ``(method_name, args, kwargs)`` tuples in
        application order (first applied first).
        """
        # State machine: pipe-mode until ggplot starts, then plus-mode.
        in_ggplot = False
        # Start with the base. We emit it via ``_emit_expr`` so nested
        # method chains (rare) get translated.
        current = self._emit_expr(base, prec=20)

        for method_name, args, kwargs in chain:
            if method_name == "ggplot":
                # Switch to ggplot mode. ``base.ggplot(x="a", y="b")``
                # → ``ggplot(base, aes(x = a, y = b))``.
                aes_text = self._emit_aes_kwargs(kwargs)
                if aes_text:
                    current = f"ggplot({current}, aes({aes_text}))"
                else:
                    current = f"ggplot({current})"
                in_ggplot = True
                continue

            if in_ggplot or is_chain_extension(method_name) or method_name == "theme":
                # ggplot chain extension — compose with ``+``.
                in_ggplot = True
                ext = self._emit_ggplot_extension_call(method_name, args, kwargs)
                current = f"{current} + {ext}"
                continue

            # Default: dplyr pipe.
            r_method, slot, auto = self._lookup_verb_method(method_name)
            verb_args = self._emit_verb_args(args, kwargs, slot)
            current = f"{current} |>\n  {r_method}({verb_args})"

        return current

    def _lookup_verb_method(self, py_method: str) -> tuple[str, Slot, tuple]:
        """Resolve a Python method name to (R verb name, slot, auto_kwargs).
        If not a known verb, treat as opaque method call (pipe through
        unchanged, slot=NONE)."""
        r_name = _HEA_METHOD_TO_R.get(py_method)
        if r_name is None:
            return py_method, Slot.NONE, ()
        verb = VERB_TABLE[r_name]
        return r_name, verb.slot, verb.auto_kwargs

    def _emit_verb_args(self, args: list, kwargs: list, slot: Slot) -> str:
        """Emit args for a dplyr verb call, applying inverse NSE rules.

        - EXPR slot: ``col("x")`` unwraps to bare ``x``; comparisons /
          method-form helpers reverse too (handled implicitly because
          ``_emit_expr`` returns ``col("x").mean()`` → ``mean(x)`` via
          the helper-call path).
        - COLUMN_NAME slot: bare ``"x"`` string literals reverse to
          bare ``x``.
        - NONE slot: pass-through.
        """
        parts: list[str] = []
        with self.nse.enter(slot):
            for a in args:
                parts.append(self._emit_verb_value(a, slot))
            for kw in kwargs:
                r_name = _PY_KWARG_TO_R.get(kw.arg, kw.arg)
                # Per-kwarg slot override mirroring forward direction.
                kw_slot = self._kwarg_slot(kw.arg, slot)
                value_str = self._emit_verb_value(kw.value, kw_slot)
                parts.append(f"{r_name} = {value_str}")
        return ", ".join(parts)

    def _kwarg_slot(self, py_name: str, parent: Slot) -> Slot:
        """Inverse of KWARG_ALIASES value_slot lookup."""
        for r_name, alias in KWARG_ALIASES.items():
            if alias.py_name == py_name and alias.value_slot is not None:
                return alias.value_slot
        return parent

    def _emit_verb_value(self, value: P.expr, slot: Slot) -> str:
        """Emit a single verb-arg value, applying COLUMN_NAME unwrap."""
        if slot is Slot.COLUMN_NAME:
            # Strings unwrap to bare names; lists of strings unwrap to
            # c(name, name); other values pass through.
            if isinstance(value, P.Constant) and isinstance(value.value, str):
                return value.value
            if isinstance(value, P.List):
                if all(isinstance(e, P.Constant) and isinstance(e.value, str) for e in value.elts):
                    return "c(" + ", ".join(e.value for e in value.elts) + ")"
            if isinstance(value, P.Dict):
                # ``by={"a": "b"}`` → ``c("a" = "b")``.
                parts = []
                for k, v in zip(value.keys, value.values):
                    if isinstance(k, P.Constant) and isinstance(v, P.Constant):
                        parts.append(f'"{k.value}" = "{v.value}"')
                    else:
                        parts.append(f"{self._emit_expr(k, prec=20)} = {self._emit_expr(v, prec=20)}")
                return f"c({', '.join(parts)})"
        with self.nse.enter(slot):
            return self._emit_expr(value, prec=20)

    # ---- ggplot extension reverse --------------------------------------

    def _emit_ggplot_extension_call(self, method: str, args: list, kwargs: list) -> str:
        """``geom_x(args)`` reverse — wrap any aesthetic-name string-valued
        kwargs in ``aes(...)``; others stay as named args."""
        aes_pairs: list[tuple[str, str]] = []
        other_pairs: list[tuple[str, str]] = []
        positional: list[str] = []
        for a in args:
            # facet_wrap('~island') / facet_grid('y~x') — the string
            # carries the formula. Try to round-trip it.
            if isinstance(a, P.Constant) and isinstance(a.value, str) and "~" in a.value:
                positional.append(a.value.strip())
            else:
                positional.append(self._emit_expr(a, prec=20))
        for kw in kwargs:
            r_name = _PY_KWARG_TO_R.get(kw.arg, kw.arg)
            value = kw.value
            if kw.arg in _AESTHETIC_NAMES and isinstance(value, P.Constant) and isinstance(value.value, str):
                # Aesthetic-name + string value = column mapping → goes in aes().
                aes_pairs.append((r_name, value.value))
            else:
                other_pairs.append((r_name, self._emit_expr(value, prec=20)))
        body_parts: list[str] = []
        body_parts.extend(positional)
        if aes_pairs:
            body_parts.append("aes(" + ", ".join(f"{k} = {v}" for k, v in aes_pairs) + ")")
        body_parts.extend(f"{k} = {v}" for k, v in other_pairs)
        return f"{method}({', '.join(body_parts)})"

    def _emit_aes_kwargs(self, kwargs: list) -> str:
        """Emit ``ggplot()`` root aesthetics — every kwarg goes into ``aes(...)``."""
        parts: list[str] = []
        for kw in kwargs:
            value = kw.value
            if isinstance(value, P.Constant) and isinstance(value.value, str):
                parts.append(f"{kw.arg} = {value.value}")
            else:
                parts.append(f"{kw.arg} = {self._emit_expr(value, prec=20)}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_method_chain(call: P.Call) -> tuple[P.expr, list]:
    """Walk ``call.func`` left as long as it's an Attribute on a Call.
    Returns ``(base, chain)`` where chain is ordered first-applied first.
    """
    chain: list[tuple[str, list, list]] = []
    cur: P.AST = call
    while (
        isinstance(cur, P.Call)
        and isinstance(cur.func, P.Attribute)
    ):
        chain.append((cur.func.attr, list(cur.args), list(cur.keywords)))
        cur = cur.func.value
    chain.reverse()
    return cur, chain  # type: ignore[return-value]


def _is_translation_chain(chain: list) -> bool:
    """``True`` iff any method in the chain is a known verb or ggplot
    extension — i.e. this is something we have a translation strategy
    for. Otherwise fall back to a plain Python-style call (which the
    user can clean up manually)."""
    for method_name, _, _ in chain:
        if method_name in _HEA_METHOD_TO_R:
            return True
        if method_name == "ggplot":
            return True
        if is_chain_extension(method_name) or method_name == "theme":
            return True
    return False


def _maybe_paren(text: str, my_prec: int, outer_prec: int) -> str:
    """Wrap ``text`` in parens iff ``my_prec`` is looser (higher) than
    ``outer_prec``."""
    if my_prec > outer_prec:
        return f"({text})"
    return text


# ---------------------------------------------------------------------------
# Library preamble inference
# ---------------------------------------------------------------------------


# Function names that, if they appear in the emitted R source, signal the
# script needs the tidyverse meta-package. Members are gathered from the
# forward translator's surface: dplyr/tidyr verbs, tidy-select helpers,
# expression helpers, lubridate/forcats/stringr fns. ``ggplot``/``aes``
# and the open-set prefixes ``geom_*``/``scale_*``/``coord_*``/``facet_*``/
# ``theme_*`` are handled by :data:`_GGPLOT_PATTERN` below.
_TIDYVERSE_FN_NAMES: frozenset[str] = frozenset({
    # dplyr verbs (mirror VERB_TABLE.keys minus joins which only need dplyr)
    "filter", "mutate", "transmute", "summarize", "summarise",
    "select", "group_by", "ungroup", "count", "add_count", "distinct",
    "arrange", "rename", "relocate", "pull", "glimpse",
    "inner_join", "left_join", "right_join", "full_join",
    "semi_join", "anti_join", "cross_join", "nest_join",
    # tidyr
    "pivot_longer", "pivot_wider", "separate", "unite",
    "drop_na", "replace_na", "fill", "complete", "expand",
    "nest", "unnest",
    # dplyr expression helpers
    "case_when", "if_else", "coalesce", "na_if", "between", "near",
    "desc", "n_distinct", "row_number", "lag", "lead",
    "min_rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
    "cummean", "cumall", "cumany", "consecutive_id", "nth",
    # tidy-select helpers
    "starts_with", "ends_with", "contains", "matches", "everything",
    "all_of", "any_of", "last_col", "num_range",
    # forcats
    "fct_infreq", "fct_relevel", "fct_recode", "fct_collapse",
    "fct_lump_n", "fct_lump_lowfreq", "fct_reorder", "fct_reorder2", "fct_rev",
    # stringr
    "str_detect", "str_replace", "str_replace_all",
    "str_to_lower", "str_to_upper", "str_to_title",
    "str_length", "str_sub", "str_trim", "str_squish",
    "str_pad", "str_wrap", "str_split", "str_extract", "str_extract_all",
    "str_count", "str_starts", "str_ends",
    # lubridate
    "ymd", "mdy", "dmy", "ymd_hms", "as_date",
    "wday", "yday",
    # ggplot top-level entry points
    "ggplot", "aes", "labs", "xlab", "ylab", "ggtitle",
    "xlim", "ylim", "lims", "guides", "annotate",
})

_PATCHWORK_FN_NAMES: frozenset[str] = frozenset({
    "plot_annotation", "plot_layout", "wrap_plots",
})

# Open-set ggplot prefixes — any ``geom_x(``, ``scale_y(``, etc.
_GGPLOT_PATTERN = re.compile(
    r"\b(?:geom_|scale_|coord_|facet_|theme_|stat_|position_|element_)\w+\s*\(|\btheme\s*\("
)


def _name_call_pattern(names: frozenset[str]) -> re.Pattern:
    """Compile a regex matching any of ``names`` immediately followed by
    ``(``. Word-boundary anchored so ``filter`` doesn't match ``filterX``."""
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\s*\(")


_TIDYVERSE_PATTERN = _name_call_pattern(_TIDYVERSE_FN_NAMES)
_PATCHWORK_PATTERN = _name_call_pattern(_PATCHWORK_FN_NAMES)


@functools.lru_cache(maxsize=1)
def _dataset_registry() -> dict[str, tuple[str, ...]]:
    """Build a ``{dataset_name: (pkg1, pkg2, ...)}`` map from rdatasets.

    Cached for the life of the process — the first call scans ~75
    packages × ~30 items each (≈600ms once); subsequent calls are
    instant. If rdatasets isn't installed, returns an empty map and
    autoload detection becomes a no-op (translator still works).
    """
    try:
        import rdatasets
    except ImportError:
        return {}
    registry: dict[str, list[str]] = {}
    for pkg in rdatasets.packages():
        for item in rdatasets.items(pkg):
            name = item.removesuffix(".pkl")
            registry.setdefault(name, []).append(pkg)
    # Freeze the lists into tuples so the cached dict is hashable-safe.
    return {n: tuple(pkgs) for n, pkgs in registry.items()}


# Names we should NEVER consider as dataset references even if they
# happen to collide with one in rdatasets. ``c`` / ``data`` / etc.
# These are functions, builtins, or other names with deterministic
# Python meanings. Without this filter, autoload would over-trigger.
_DATASET_REF_EXCLUSIONS: frozenset[str] = frozenset({
    # Python builtins users commonly write
    "True", "False", "None",
    "print", "len", "range", "list", "dict", "set", "tuple", "int",
    "float", "str", "bool", "type", "object",
    # hea / polars surface that's a name, not a dataset
    "hea", "pl", "col", "n", "desc", "selectors",
    "DataFrame", "LazyFrame", "Series", "Expr",
    # Common R-side names that map back via the FUNCTION_TABLE
    "case_when", "if_else", "coalesce", "data", "first", "last",
    # Single-letter / very-short names that would false-positive
    "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
    "p", "q", "p1", "p2", "p3", "p4",
    "df", "dat", "obj", "out", "tmp", "res", "ans",
})


def _collect_dataset_refs(module: P.Module) -> list[tuple[str, str]]:
    """Walk ``module`` looking for bare names that look like autoload
    references to rdatasets-known datasets.

    Returns a list of ``(dataset_name, package_name)`` for each unique
    match — names defined in the script (assignment targets, function
    args, for-loop vars) are excluded, as are short / common names
    listed in :data:`_DATASET_REF_EXCLUSIONS`. Ambiguous names (matched
    by more than one package) are dropped — picking a package would be
    a guess.
    """
    registry = _dataset_registry()
    if not registry:
        return []

    defined: set[str] = set()
    for stmt in P.walk(module):
        if isinstance(stmt, P.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, P.Name):
                    defined.add(tgt.id)
                elif isinstance(tgt, P.Tuple):
                    for e in tgt.elts:
                        if isinstance(e, P.Name):
                            defined.add(e.id)
        elif isinstance(stmt, P.For):
            if isinstance(stmt.target, P.Name):
                defined.add(stmt.target.id)
        elif isinstance(stmt, (P.FunctionDef, P.AsyncFunctionDef, P.Lambda)):
            if hasattr(stmt, "name"):
                defined.add(stmt.name)  # type: ignore[attr-defined]
            for a in stmt.args.args:
                defined.add(a.arg)
        elif isinstance(stmt, P.ImportFrom):
            for alias in stmt.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(stmt, P.Import):
            for alias in stmt.names:
                defined.add((alias.asname or alias.name).split(".")[0])

    referenced: set[str] = set()
    for node in P.walk(module):
        if isinstance(node, P.Name) and isinstance(node.ctx, P.Load):
            referenced.add(node.id)

    candidates = referenced - defined - _DATASET_REF_EXCLUSIONS
    out: list[tuple[str, str]] = []
    for name in sorted(candidates):
        pkgs = registry.get(name)
        if pkgs is None:
            continue
        if len(pkgs) == 1:
            pkg = pkgs[0]
            # Skip R's default-loaded packages — emitting ``library(datasets)``
            # is redundant since base R always has them in scope.
            if pkg in _R_DEFAULT_PACKAGES:
                continue
            out.append((name, pkg))
        # Ambiguous (len > 1) — skip. The user can disambiguate by
        # writing the explicit ``X = data("X", package="pkg")`` form,
        # which the smart-data-assign rewrite still picks up.
    return out


# R's default-loaded packages (always in scope at the start of every R
# session). Emitting ``library()`` for these is redundant — skip them
# even when an autoload candidate resolves there.
_R_DEFAULT_PACKAGES: frozenset[str] = frozenset({
    "base", "datasets", "graphics", "grDevices",
    "methods", "stats", "utils",
})


def _build_preamble(r_source: str, extra_libs: set[str]) -> list[str]:
    """Infer the ``library()`` calls the translated R script needs.

    Detection is regex-based on the emitted source — simpler than
    tracking flags through every emit path, and accepts the rare
    false-positive (a function name appearing inside a string literal)
    in exchange for not missing real usages. False positives just emit
    a redundant ``library()`` call, which loads but does no harm.

    Order: tidyverse first, then patchwork, then any data packages
    (sorted) from :attr:`Translator._extra_libs` — that's the
    convention r4ds-style scripts use.
    """
    libs: list[str] = []
    if _TIDYVERSE_PATTERN.search(r_source) or _GGPLOT_PATTERN.search(r_source):
        libs.append("library(tidyverse)")
    if _PATCHWORK_PATTERN.search(r_source):
        libs.append("library(patchwork)")
    for pkg in sorted(extra_libs):
        if pkg in ("tidyverse", "patchwork"):
            continue
        libs.append(f"library({pkg})")
    return libs


def _quote_string(s: str) -> str:
    """R-style string literal — prefer double quotes; escape minimally."""
    if '"' in s and "'" not in s:
        return f"'{s}'"
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'
