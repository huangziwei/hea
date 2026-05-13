"""R AST → Python AST → Python source.

This is Phase 1's emitter (scalars / operators / control flow) merged with
Phase 2's verb dispatch and NSE handling. The split is logical, not
file-level: one ``Translator`` walks the R AST, building stdlib ``ast``
nodes that ``ast.unparse`` finally renders.

Public entry: :func:`translate`. Returns a Python source string. Callers
needing the AST instead can use :class:`Translator` directly.

NSE handling lives in :mod:`hea.translate.nse`; the verb and function
dispatch tables live in :mod:`hea.translate.registry`. This module is the
glue that walks the R AST and asks those tables what to emit.
"""

from __future__ import annotations

import ast as P
from typing import Optional

from . import _datasets, r_ast as R
from .nse import NSEContext, Slot
from .r_parser import parse as parse_r
from .registry.functions import FUNCTION_TABLE, Func, resolve_kwarg
from .registry.ggplot import is_chain_extension
from .registry.verbs import VERB_TABLE, Verb


class RTranslateError(Exception):
    """Raised when a node can't be translated within the documented sublanguage."""

    def __init__(self, message: str, node: R.Node):
        self.node = node
        super().__init__(f"{message} at span {node.span}")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def translate(src: str) -> str:
    """Translate an R source string to a Python source string.

    Raises :class:`RTranslateError` on out-of-grammar inputs; the parser
    may raise :class:`hea.translate.r_parser.RParseError` first.
    """
    prog = parse_r(src)
    return Translator().translate(prog)


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


# R BinOp → Python ast operator class (for arithmetic / shift / bitwise).
_BINOP_PY = {
    "+":   P.Add,
    "-":   P.Sub,
    "*":   P.Mult,
    "/":   P.Div,
    "^":   P.Pow,
    "%%":  P.Mod,
    "%/%": P.FloorDiv,
}

# R comparison ops → Python ast comparison op.
_CMP_PY = {
    "==": P.Eq,
    "!=": P.NotEq,
    "<":  P.Lt,
    "<=": P.LtE,
    ">":  P.Gt,
    ">=": P.GtE,
}


class Translator:
    """Stateful walker. One instance per translation."""

    def __init__(self):
        self.nse = NSEContext()
        # Packages the source declared via ``library(pkg)`` /
        # ``require(pkg)``. Used to disambiguate autoload candidates
        # below — if ``penguins`` is in two packages and one of them
        # was loaded, prefer that one.
        self._loaded_packages: set[str] = set()

    # -- public ------------------------------------------------------------

    def translate(self, prog: R.Program) -> str:
        module = self._visit_program(prog)
        P.fix_missing_locations(module)
        return P.unparse(module)

    # -- top-level ---------------------------------------------------------

    def _visit_program(self, prog: R.Program) -> P.Module:
        body: list[P.stmt] = []
        for stmt in prog.statements:
            # ``library(pkg)`` / ``require(pkg)`` — drop, but record the
            # package so autoload inference can prefer it for ambiguous
            # dataset names below.
            if _is_library_call(stmt):
                _record_library_pkg(stmt, self._loaded_packages)
                continue
            if _is_noop_call(stmt):
                continue
            # Standalone ``data("X", package="Y")`` — rewrite as a
            # Python assignment ``X = hea.data("X", package="Y")``. The
            # R side is side-effectful (data() loads into the env); the
            # Python equivalent needs an explicit binding.
            smart = self._maybe_smart_data_call(stmt)
            if smart is not None:
                body.append(smart)
                continue
            body.append(self._as_stmt(self._visit(stmt)))

        # Autoload preamble: bare names referenced but not defined that
        # match a known rdatasets entry get a ``hea.data(...)`` load
        # prepended to the module body.
        preamble = self._build_autoload_preamble(body)
        return P.Module(body=preamble + body, type_ignores=[])

    def _maybe_smart_data_call(self, stmt: R.Node) -> Optional[P.AST]:
        """If ``stmt`` is a standalone ``data("X", package="Y")`` call,
        emit it as ``X = hea.data("X", package="Y")``.

        R's ``data()`` loads the named dataset into the calling
        environment as a side-effect; Python needs an explicit binding
        for the script to work after translation.
        """
        if not (
            isinstance(stmt, R.Call)
            and isinstance(stmt.func, R.Identifier)
            and stmt.func.name == "data"
            and stmt.args
        ):
            return None
        first = stmt.args[0]
        if isinstance(first, R.StrLit):
            name = first.value
        elif isinstance(first, R.Identifier):
            name = first.name
        else:
            return None
        pkg: Optional[str] = None
        for arg in stmt.args[1:]:
            if isinstance(arg, R.NamedArg) and arg.name == "package":
                if isinstance(arg.value, R.StrLit):
                    pkg = arg.value.value
                elif isinstance(arg.value, R.Identifier):
                    pkg = arg.value.name
        keywords = []
        if pkg:
            keywords.append(P.keyword(arg="package", value=P.Constant(value=pkg)))
        return P.Assign(
            targets=[P.Name(id=name, ctx=P.Store())],
            value=P.Call(
                func=P.Attribute(value=P.Name("hea", ctx=P.Load()), attr="data", ctx=P.Load()),
                args=[P.Constant(value=name)],
                keywords=keywords,
            ),
        )

    def _build_autoload_preamble(self, body: list[P.stmt]) -> list[P.stmt]:
        """Scan ``body`` for bare ``Name`` references that aren't bound
        anywhere and match a known dataset; emit a ``hea.data(...)``
        assignment for each.

        The user's ``library(...)`` declarations bias the resolution: an
        ambiguous name like ``penguins`` (modeldata or palmerpenguins)
        gets the loaded package if one was declared.
        """
        defined: set[str] = set()
        referenced: set[str] = set()
        for stmt in body:
            for node in P.walk(stmt):
                if isinstance(node, P.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, P.Name):
                            defined.add(tgt.id)
                elif isinstance(node, P.For) and isinstance(node.target, P.Name):
                    defined.add(node.target.id)
                elif isinstance(node, (P.FunctionDef, P.AsyncFunctionDef)):
                    defined.add(node.name)
                    for a in node.args.args:
                        defined.add(a.arg)
                elif isinstance(node, P.Lambda):
                    for a in node.args.args:
                        defined.add(a.arg)
                elif isinstance(node, P.Name) and isinstance(node.ctx, P.Load):
                    referenced.add(node.id)

        loaded = frozenset(self._loaded_packages)
        candidates = sorted(referenced - defined - _datasets.DATASET_REF_EXCLUSIONS)
        out: list[P.stmt] = []
        for name in candidates:
            pkg = _datasets.resolve_dataset(name, loaded_packages=loaded)
            if pkg is None:
                continue
            out.append(_make_data_load_stmt(name, pkg))
        return out

    def _as_stmt(self, node) -> P.stmt:
        """Promote an expression node to a statement, wrapping in
        ``ast.Expr`` if needed. Pass-through for nodes already-statement
        (Assign / If / For / While / etc)."""
        if isinstance(node, P.stmt):
            return node
        return P.Expr(value=node)

    # -- dispatch ----------------------------------------------------------

    def _visit(self, node: R.Node) -> P.AST:
        """Dispatch on R AST node type."""
        method = getattr(self, "_visit_" + type(node).__name__, None)
        if method is None:
            raise RTranslateError(
                f"no translator for {type(node).__name__}", node
            )
        return method(node)

    # -- literals ----------------------------------------------------------

    def _visit_NumLit(self, n: R.NumLit) -> P.AST:
        # R's bare ``1`` is technically a double, but downstream hea/polars
        # code reads cleaner with Python ints when the value is whole — and
        # polars coerces literals on comparison either way. Users wanting
        # explicit double semantics can write ``1L`` (which becomes IntLit
        # — also emitted as int) or just any non-integer literal.
        if n.value.is_integer() and -2**53 < n.value < 2**53:
            return P.Constant(value=int(n.value))
        return P.Constant(value=n.value)

    def _visit_IntLit(self, n: R.IntLit) -> P.AST:
        return P.Constant(value=n.value)

    def _visit_ComplexLit(self, n: R.ComplexLit) -> P.AST:
        return P.Constant(value=complex(0, n.value))

    def _visit_StrLit(self, n: R.StrLit) -> P.AST:
        return P.Constant(value=n.value)

    def _visit_BoolLit(self, n: R.BoolLit) -> P.AST:
        return P.Constant(value=n.value)

    def _visit_NullLit(self, n: R.NullLit) -> P.AST:
        return P.Constant(value=None)

    def _visit_NaLit(self, n: R.NaLit) -> P.AST:
        # In EXPR slot, R's NA maps to polars' null literal. Outside, to None.
        if self.nse.is_expr():
            # ``pl.lit(None)`` — but we want everything to flow through hea,
            # so emit ``hea.lit(None)``.
            return _call(_attr(_name("hea"), "lit"), [P.Constant(None)])
        return P.Constant(value=None)

    def _visit_InfLit(self, n: R.InfLit) -> P.AST:
        return _call(_name("float"), [P.Constant("inf")])

    def _visit_NanLit(self, n: R.NanLit) -> P.AST:
        return _call(_name("float"), [P.Constant("nan")])

    # -- identifiers -------------------------------------------------------

    def _visit_Identifier(self, n: R.Identifier) -> P.AST:
        slot = self.nse.current
        if slot is Slot.EXPR:
            return _call(_name("col"), [P.Constant(n.name)])
        if slot is Slot.COLUMN_NAME:
            return P.Constant(value=n.name)
        # NONE — emit as a Python name. Dot identifiers (``data.frame``,
        # ``na.omit``) become underscores so the result is a valid Python
        # identifier; this matches how the rest of hea names things.
        py_name = n.name.replace(".", "_")
        return _name(py_name)

    # -- operators ---------------------------------------------------------

    def _visit_UnaryOp(self, n: R.UnaryOp) -> P.AST:
        operand = self._visit(n.operand)
        if n.op == "-":
            return P.UnaryOp(P.USub(), operand)
        if n.op == "+":
            return P.UnaryOp(P.UAdd(), operand)
        if n.op == "!":
            # In EXPR slot, polars Expr negation uses ``~``; elsewhere ``not``.
            if self.nse.is_expr():
                return P.UnaryOp(P.Invert(), operand)
            return P.UnaryOp(P.Not(), operand)
        if n.op == "?":
            # Help operator — rare. Translate to a comment placeholder.
            raise RTranslateError("R help operator `?` not supported", n)
        raise RTranslateError(f"unknown unary operator {n.op!r}", n)

    def _visit_BinOp(self, n: R.BinOp) -> P.AST:
        op = n.op

        # ggplot chain extension: ``<plot expr> + geom_x(args)``. We
        # detect by the RHS shape — it's a chain step iff it's a Call to
        # a name like ``geom_*`` / ``scale_*`` / ``labs`` / etc. The LHS
        # is recursively translated, so a long chain unfolds left-to-right.
        if op == "+" and _is_ggplot_chain_call(n.right):
            return self._emit_ggplot_chain_step(n.left, n.right)

        left = self._visit(n.left)
        right = self._visit(n.right)

        # Arithmetic / shift / bitwise.
        py_op = _BINOP_PY.get(op)
        if py_op is not None:
            return P.BinOp(left=left, op=py_op(), right=right)

        # Comparison.
        cmp = _CMP_PY.get(op)
        if cmp is not None:
            return P.Compare(left=left, ops=[cmp()], comparators=[right])

        # Logical and/or.
        #
        # R distinguishes single (``&`` / ``|`` — elementwise) from double
        # (``&&`` / ``||`` — short-circuit / scalar). We mirror:
        # - ``&``/``|`` ALWAYS emit Python bitwise. That's correct for
        #   polars Expr (elementwise) AND patchwork plot composition
        #   (``p1 | p2``), and the scalar-bool case (``True | False``)
        #   evaluates the same way. Short-circuit was never R's semantics
        #   for these operators anyway.
        # - ``&&``/``||`` are short-circuit in R; outside EXPR slot we
        #   emit Python ``and``/``or``. Inside EXPR slot they still mean
        #   elementwise (polars Expr), so we emit bitwise.
        if op == "&":
            return P.BinOp(left=left, op=P.BitAnd(), right=right)
        if op == "|":
            return P.BinOp(left=left, op=P.BitOr(), right=right)
        if op == "&&":
            if self.nse.is_expr():
                return P.BinOp(left=left, op=P.BitAnd(), right=right)
            return P.BoolOp(op=P.And(), values=[left, right])
        if op == "||":
            if self.nse.is_expr():
                return P.BinOp(left=left, op=P.BitOr(), right=right)
            return P.BoolOp(op=P.Or(), values=[left, right])

        # Sequence ``a:b`` → ``hea.seq(a, b)``.
        if op == ":":
            return _call(_attr(_name("hea"), "seq"), [left, right])

        # Namespace access ``pkg::name`` / ``pkg:::name``. Drop the package
        # qualifier — the function registry handles renaming.
        if op == "::" or op == ":::":
            return right

        # ``%in%`` → ``.is_in(...)``.
        if op == "%in%":
            return _call(_attr(left, "is_in"), [right])

        # Other ``%infix%`` operators — emit as a function call so the user
        # sees something sensible. The registry can resolve specific names
        # in a later phase.
        if op.startswith("%") and op.endswith("%"):
            fname = op.strip("%")
            return _call(_name(fname), [left, right])

        raise RTranslateError(f"unknown binary operator {op!r}", n)

    # -- pipes -------------------------------------------------------------

    def _visit_Pipe(self, n: R.Pipe) -> P.AST:
        """``lhs |> rhs`` or ``lhs %>% rhs``.

        For the native pipe ``|>``, the lhs is inserted as the first
        positional arg of rhs (which must be a call).
        For magrittr ``%>%``, the ``.`` placeholder in rhs's args becomes
        the lhs; if no placeholder, lhs is inserted as the first positional
        arg (same as native pipe).
        """
        if not isinstance(n.rhs, R.Call):
            # ``x |> f`` — rhs is a bare name, equivalent to ``x |> f()``.
            # Synthesize a zero-arg call.
            synth_call = R.Call(n.rhs, (), n.rhs.span)  # type: ignore[arg-type]
            return self._emit_call_with_first(n.lhs, synth_call)

        if n.op == "%>%":
            return self._emit_magrittr(n.lhs, n.rhs)
        return self._emit_call_with_first(n.lhs, n.rhs)

    def _emit_magrittr(self, lhs: R.Node, rhs: R.Call) -> P.AST:
        """Replace ``.`` placeholders in rhs args with lhs; if none, thread
        lhs as the first positional arg (native-pipe semantics)."""
        new_args: list[R.Node] = []
        placeholder_found = False
        for arg in rhs.args:
            if isinstance(arg, R.Identifier) and arg.name == ".":
                new_args.append(lhs)
                placeholder_found = True
            elif isinstance(arg, R.NamedArg) and isinstance(arg.value, R.Identifier) and arg.value.name == ".":
                new_args.append(R.NamedArg(arg.name, lhs, arg.span))
                placeholder_found = True
            else:
                new_args.append(arg)
        if not placeholder_found:
            return self._emit_call_with_first(lhs, rhs)
        synth = R.Call(rhs.func, tuple(new_args), rhs.span)
        return self._visit_Call(synth)

    def _emit_call_with_first(self, lhs: R.Node, rhs: R.Call) -> P.AST:
        """Build a synthetic ``rhs.func(lhs, *rhs.args)`` and translate it.
        Centralizes the pipe-rewrite so verb dispatch sees a normal call."""
        synth = R.Call(rhs.func, (lhs, *rhs.args), rhs.span)
        return self._visit_Call(synth)

    # -- calls -------------------------------------------------------------

    def _visit_Call(self, n: R.Call) -> P.AST:
        """Three dispatch layers, tried in order:

        1. If ``func`` is a known **verb**: rewrite to ``first_arg.verb(rest)``
           with the verb's NSE slot active for the rest.
        2. If ``func`` is a known **function/helper** in
           :data:`FUNCTION_TABLE`: dispatch on form.
        3. Otherwise: emit a regular function call, walking args with the
           current slot.
        """
        # Strip namespace qualifier: ``pkg::name(...)`` — keep just ``name``.
        func = n.func
        if isinstance(func, R.BinOp) and func.op in ("::", ":::"):
            func = func.right  # type: ignore[assignment]

        if isinstance(func, R.Identifier):
            name = func.name

            # 0) ggplot(df, aes(...)) — the only ggplot entry point that
            # takes a data frame. Rewritten to ``df.ggplot(...)`` so it
            # composes with the chain rewriter via the ``+`` operator.
            if name == "ggplot":
                return self._emit_ggplot_root(n.args)

            # 1) Verb dispatch.
            verb = VERB_TABLE.get(name)
            if verb is not None and n.args:
                return self._emit_verb_call(verb, n.args)

            # 2) Function-helper dispatch.
            helper = FUNCTION_TABLE.get(name)
            if helper is not None:
                return self._emit_helper_call(helper, name, n.args)

            # 3) Default: regular call. Args walked under current slot,
            # which is what's wanted for nested user calls.
            return self._emit_regular_call(name, n.args)

        # Non-identifier callable (e.g. ``f()(g)`` — Call of Call).
        callee = self._visit(func)
        args, kwargs = self._translate_args(n.args)
        return _call(callee, args, kwargs)

    def _emit_verb_call(self, verb: Verb, args: tuple[R.Node, ...]) -> P.AST:
        """First arg becomes the receiver; rest walked under ``verb.slot``.

        Any ``across()`` call inside the verb's args is expanded BEFORE
        translation — it's a translate-time macro that fans one entry
        into N kwargs (one per matched column).

        After translating the user's kwargs, any :attr:`Verb.auto_kwargs`
        are appended — that's how ``transmute`` becomes ``mutate(..., _keep="none")``.
        Auto kwargs only land if the user didn't already pass that name,
        so the user can override the default (``transmute(..., .keep="all")``
        would emit ``.mutate(..., _keep="all")``, not both).
        """
        receiver = self._visit(args[0])
        rest = self._expand_across_in_args(args[1:])
        with self.nse.enter(verb.slot):
            py_args, py_kwargs = self._translate_args(rest)
        if verb.auto_kwargs:
            existing = {kw.arg for kw in py_kwargs}
            for name, value in verb.auto_kwargs:
                if name in existing:
                    continue
                py_kwargs.append(P.keyword(arg=name, value=P.Constant(value=value)))
        return _call(_attr(receiver, verb.hea_method), py_args, py_kwargs)

    # ----- ggplot ---------------------------------------------------------

    def _emit_ggplot_root(self, args: tuple[R.Node, ...]) -> P.AST:
        """``ggplot(df, aes(x = a, y = b))`` → ``df.ggplot(x="a", y="b")``.

        The first positional arg is the data frame (receiver). Subsequent
        args may be:
        - ``aes(...)`` positional (or as ``mapping = aes(...)``) — unwrap
          its kwargs into ``ggplot()``'s kwargs.
        - Other named kwargs (``environment = ...``) — pass through.
        """
        if not args:
            return _call(_attr(_name("hea"), "ggplot"))
        receiver = self._visit(args[0])
        kwargs = self._collect_ggplot_kwargs(args[1:])
        return _call(_attr(receiver, "ggplot"), [], kwargs)

    def _emit_ggplot_chain_step(self, left: R.Node, right: R.Call) -> P.AST:
        """``<plot> + <ext>(args)`` → ``<plot translated>.<ext>(args)``."""
        receiver = self._visit(left)
        func_name = right.func.name  # type: ignore[attr-defined]
        kwargs = self._collect_ggplot_kwargs(right.args)
        # Positional args (besides aes) pass through under NSE.NONE so
        # ``annotate("text", x=, y=, label=)`` keeps its string literal.
        positional: list[P.AST] = []
        with self.nse.enter(Slot.NONE):
            for arg in right.args:
                if isinstance(arg, R.NamedArg):
                    continue  # collected by _collect_ggplot_kwargs
                if _is_named_call(arg, "aes"):
                    continue
                if isinstance(arg, R.Tilde):
                    # facet_wrap(~island) / facet_grid(rows ~ cols)
                    positional.append(P.Constant(value=_format_formula(arg)))
                    continue
                positional.append(self._visit(arg))
        return _call(_attr(receiver, func_name), positional, kwargs)

    def _collect_ggplot_kwargs(self, args: tuple[R.Node, ...]) -> list[P.keyword]:
        """Pull aes() unwrap + regular kwargs out of a ggplot-extension
        call's argument list. Positional non-aes args are ignored here —
        the caller handles them."""
        kwargs: list[P.keyword] = []
        for arg in args:
            if _is_named_call(arg, "aes"):
                kwargs.extend(self._translate_aes_args(arg.args))
                continue
            if isinstance(arg, R.NamedArg) and _is_named_call(arg.value, "aes"):
                # ``mapping = aes(...)`` form.
                kwargs.extend(self._translate_aes_args(arg.value.args))
                continue
            if isinstance(arg, R.NamedArg):
                alias = resolve_kwarg(arg.name)
                if alias.value_slot is not None:
                    with self.nse.enter(alias.value_slot):
                        value = self._visit(arg.value)
                else:
                    with self.nse.enter(Slot.NONE):
                        value = self._visit(arg.value)
                kwargs.append(P.keyword(arg=alias.py_name, value=value))
        return kwargs

    # ggplot's documented positional-aesthetic order for ``aes()``. R's
    # convention (and ``?aes``): the first positional is ``x``, the
    # second is ``y``. Everything else is named. We map by position so
    # ``aes(x, y, color = z)`` translates as the user intends.
    _AES_POS_AESTHETICS = ("x", "y")

    def _translate_aes_args(self, args: tuple[R.Node, ...]) -> list[P.keyword]:
        """Translate ``aes()`` args to ggplot kwargs.

        Each arg becomes a kwarg by either its explicit name (NamedArg)
        or its position (mapping to ``x``, then ``y`` — R's convention).
        The value translation rules:

        - Bare ``Identifier`` → string ``"name"`` (the column reference).
        - ``StrLit`` → string (pass-through).
        - Anything else → translate under EXPR slot for column-aware
          expressions like ``log(weight)``.
        """
        kwargs: list[P.keyword] = []
        pos_idx = 0
        for arg in args:
            if isinstance(arg, R.NamedArg):
                name = arg.name
                value = self._translate_aes_value(arg.value)
                kwargs.append(P.keyword(arg=name, value=value))
                continue
            # Positional — map to x, y in order.
            if pos_idx < len(self._AES_POS_AESTHETICS):
                name = self._AES_POS_AESTHETICS[pos_idx]
                value = self._translate_aes_value(arg)
                kwargs.append(P.keyword(arg=name, value=value))
                pos_idx += 1
            # Extra positional args (3rd+) in aes are non-standard. Drop
            # silently in v1 rather than guess at color/fill/etc.
        return kwargs

    def _translate_aes_value(self, node: R.Node) -> P.AST:
        """Translate a single aes() value: identifier → string, otherwise
        an EXPR-slot expression."""
        if isinstance(node, R.Identifier):
            return P.Constant(value=node.name)
        if isinstance(node, R.StrLit):
            return P.Constant(value=node.value)
        with self.nse.enter(Slot.EXPR):
            return self._visit(node)

    # ----- across() expansion ---------------------------------------------

    def _expand_across_in_args(self, args: tuple[R.Node, ...]) -> tuple[R.Node, ...]:
        """Walk verb args; replace each ``across(...)`` call with the
        list of synthetic ``NamedArg``\\s it expands to."""
        out: list[R.Node] = []
        for arg in args:
            if isinstance(arg, R.Call) and _is_named_call(arg, "across"):
                out.extend(self._expand_across(arg))
            else:
                out.append(arg)
        return tuple(out)

    def _expand_across(self, call: R.Call) -> list[R.Node]:
        """Translate-time expansion of ``across(cols, fn[, .names = ...])``.

        Supported in v1:
        - ``across(col, fn)``               — single col, single fn
        - ``across(c(a, b, ...), fn)``      — multiple cols, single fn
        - ``fn`` may be an :class:`R.Identifier` (named function) or a
          :class:`R.FunctionDef` (anonymous lambda) of one parameter.

        Out of scope (raise :class:`RTranslateError` until Phase 5+):
        - list-of-functions form ``list(mean = mean, sd = sd)``
        - ``.names`` glue templates

        Returns a list of synthetic :class:`R.NamedArg` nodes — one per
        target column. Each NamedArg's value is the function applied to
        an :class:`R.Identifier` for that column.
        """
        if len(call.args) < 2:
            raise RTranslateError("across() requires (cols, fns) — got fewer", call)

        cols_arg = call.args[0]
        fn_arg = call.args[1]

        # Reject the unsupported `.names` / list-of-fns forms cleanly so the
        # user sees what's missing instead of a wrong-looking translation.
        for extra in call.args[2:]:
            if isinstance(extra, R.NamedArg) and extra.name == ".names":
                raise RTranslateError(
                    "across(.names = ...) not supported in v1 — defer to Phase 5+",
                    extra,
                )
        if _is_named_call(fn_arg, "list"):
            raise RTranslateError(
                "across() with list(...) of functions not supported in v1",
                fn_arg,
            )

        col_names = _extract_col_names(cols_arg)
        results: list[R.Node] = []
        for col_name in col_names:
            synthetic_col = R.Identifier(col_name, call.span)
            applied = self._apply_across_fn(fn_arg, synthetic_col, call.span)
            results.append(R.NamedArg(col_name, applied, call.span))
        return results

    def _apply_across_fn(self, fn_arg: R.Node, col: R.Identifier, span) -> R.Node:
        """Apply ``fn_arg`` to ``col``, returning an R AST node.

        - ``fn_arg`` is an Identifier: emit ``fn_arg(col)``.
        - ``fn_arg`` is a FunctionDef with one param ``p``: substitute
          ``p`` with ``col`` in the body.
        """
        if isinstance(fn_arg, R.Identifier):
            return R.Call(fn_arg, (col,), span)
        if isinstance(fn_arg, R.FunctionDef):
            if len(fn_arg.params) != 1:
                raise RTranslateError(
                    "across() lambda must take exactly one parameter",
                    fn_arg,
                )
            return _substitute_identifier(fn_arg.body, fn_arg.params[0].name, col)
        raise RTranslateError(
            f"across() fn must be an identifier or lambda, got {type(fn_arg).__name__}",
            fn_arg,
        )

    def _emit_helper_call(self, helper: Func, r_name: str, args: tuple[R.Node, ...]) -> P.AST:
        """Translate one of the registered helpers (mean, desc, n, …).

        ``form="method"``  — only used in EXPR slot; emits
        ``col("x").func(rest_kwargs)`` when the first arg is a single
        identifier, else ``(<expr>).func(...)``.

        ``form="function"`` — emits ``hea.func(args)`` (with the registry's
        ``hea_name`` resolving qualified names like ``selectors.starts_with``).
        """
        # Special-case: c(...) → Python list / dict literal.
        if helper.hea_name == "__list__":
            return self._emit_c_call(args)

        # Special-case: case_when — Tilde args become (cond, value) tuples.
        if helper.hea_name == "case_when":
            return self._emit_case_when(args)

        # Special-case: data.frame / tibble — emit ``hea.DataFrame({...})``.
        if helper.hea_name == "__data_frame__":
            return self._emit_data_frame_call(args)

        # Override arg slot if registry specifies one.
        arg_slot_ctx = self.nse.enter(helper.arg_slot) if helper.arg_slot is not None else _null_ctx()

        if helper.form == "method" and self.nse.is_expr() and args:
            # ``mean(x, na.rm = TRUE)`` → ``col("x").mean()``.
            # The first arg becomes the receiver; remaining args become
            # the method's positional / kw args.
            #
            # ``na_rm`` is dropped: polars ``Expr.mean()`` / ``.sum()`` etc.
            # don't take that kwarg, and their behavior matches R's
            # ``na.rm = TRUE`` (skip nulls). If the user wrote
            # ``na.rm = FALSE``, we still drop the kwarg — the resulting
            # behavior diverges from R, but the parity runner will catch
            # any value diff and report it as a real hea/R gap rather
            # than a translator bug.
            first = args[0]
            if isinstance(first, R.Identifier):
                receiver = _call(_name("col"), [P.Constant(first.name)])
            else:
                # Complex first arg — visit with EXPR slot still active so
                # nested column refs get col()-wrapped.
                receiver = self._visit(first)
            with arg_slot_ctx:
                rest_args, rest_kwargs = self._translate_args(args[1:])
            rest_kwargs = [kw for kw in rest_kwargs if kw.arg != "na_rm"]
            return _call(_attr(receiver, helper.hea_name), rest_args, rest_kwargs)

        # Function form (or method-form fallback outside EXPR slot).
        with arg_slot_ctx:
            py_args, py_kwargs = self._translate_args(args)

        # ``selectors.starts_with`` → ast.Attribute chain.
        callee = _dotted_name(helper.hea_name)
        return _call(callee, py_args, py_kwargs)

    def _emit_regular_call(self, name: str, args: tuple[R.Node, ...]) -> P.AST:
        """Unknown function — emit as-is with dot→underscore name fix."""
        py_args, py_kwargs = self._translate_args(args)
        py_name = name.replace(".", "_")
        return _call(_name(py_name), py_args, py_kwargs)

    def _emit_case_when(self, args: tuple[R.Node, ...]) -> P.AST:
        """``case_when(cond1 ~ val1, cond2 ~ val2, .default = d)`` →
        ``case_when((c1, v1), (c2, v2), default=d)``.

        Every Tilde-typed positional arg becomes a 2-tuple. The named
        ``.default`` becomes the Python ``default=`` kwarg (handled by
        the generic kwarg path).
        """
        tuples: list[P.AST] = []
        kwargs: list[P.keyword] = []
        # case_when's cond/value pairs are NSE expressions — push EXPR slot.
        with self.nse.enter(Slot.EXPR):
            for arg in args:
                if isinstance(arg, R.Tilde):
                    if arg.lhs is None:
                        # ``~ value`` form — degenerate; treat as the default branch.
                        kwargs.append(P.keyword(
                            arg="default", value=self._visit(arg.rhs)
                        ))
                        continue
                    cond = self._visit(arg.lhs)
                    value = self._visit(arg.rhs)
                    tuples.append(P.Tuple(elts=[cond, value], ctx=P.Load()))
                elif isinstance(arg, R.NamedArg):
                    alias = resolve_kwarg(arg.name)
                    slot_ctx = self.nse.enter(alias.value_slot) if alias.value_slot is not None else _null_ctx()
                    with slot_ctx:
                        value = self._visit(arg.value)
                    kwargs.append(P.keyword(arg=alias.py_name, value=value))
                else:
                    # Non-tilde positional — surprising. Keep as-is so the
                    # user can see what happened.
                    tuples.append(self._visit(arg))
        return _call(_name("case_when"), tuples, kwargs)

    def _emit_data_frame_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``data.frame(a = c(1, 2), b = c("x", "y"))`` →
        ``hea.DataFrame({"a": [1, 2], "b": ["x", "y"]})``.

        Unnamed positional args become ``V1``, ``V2``, …  by position
        (R's default — though uncommon in idiomatic code). Cross-column
        references inside tibble (e.g. ``tibble(x = 1:3, y = x * 2)``)
        are not expanded — that would need build-time evaluation. Emit
        the literal as-is and let polars fail loudly at runtime.
        """
        keys: list[P.AST] = []
        values: list[P.AST] = []
        for i, arg in enumerate(args):
            if isinstance(arg, R.NamedArg):
                keys.append(P.Constant(value=arg.name))
                values.append(self._visit(arg.value))
            elif isinstance(arg, R.MissingArg):
                continue
            else:
                # Skip kwargs like ``stringsAsFactors = FALSE`` that have
                # no Python equivalent — they're noise post-translation.
                if isinstance(arg, R.NamedArg) and arg.name == "stringsAsFactors":
                    continue
                keys.append(P.Constant(value=f"V{i + 1}"))
                values.append(self._visit(arg))
        return _call(
            _attr(_name("hea"), "DataFrame"),
            [P.Dict(keys=keys, values=values)],
        )

    def _emit_c_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``c(a, b, c)`` → Python list. ``c("a" = "b", "x" = "y")`` →
        Python dict (idiomatic for join ``by`` mappings). The split is
        decided by whether any arg is named.
        """
        if any(isinstance(a, R.NamedArg) for a in args):
            keys: list[P.AST] = []
            values: list[P.AST] = []
            for a in args:
                if isinstance(a, R.NamedArg):
                    keys.append(P.Constant(value=a.name))
                    values.append(self._visit(a.value))
                else:
                    # An unnamed entry in an otherwise-named c() — R uses
                    # an empty key. Emit a None-keyed entry so the user
                    # sees the mismatch (and we don't silently drop it).
                    keys.append(P.Constant(value=None))
                    values.append(self._visit(a))
            return P.Dict(keys=keys, values=values)
        elems = [self._visit(a) for a in args]
        return P.List(elts=list(elems), ctx=P.Load())

    # -- args --------------------------------------------------------------

    def _translate_args(self, args: tuple[R.Node, ...]) -> tuple[list[P.AST], list[P.keyword]]:
        """Walk an R argument list, splitting positional from named.

        Named-arg name is resolved via :func:`resolve_kwarg` — known
        kwargs (``.by``, ``.keep``, ``.before`` etc.) may force a specific
        NSE slot for their value, overriding the surrounding verb's slot.
        Unknown kwargs inherit the current slot.
        """
        py_args: list[P.AST] = []
        py_kwargs: list[P.keyword] = []
        for arg in args:
            if isinstance(arg, R.NamedArg):
                alias = resolve_kwarg(arg.name)
                if alias.value_slot is not None:
                    with self.nse.enter(alias.value_slot):
                        value = self._visit(arg.value)
                else:
                    value = self._visit(arg.value)
                py_kwargs.append(P.keyword(arg=alias.py_name, value=value))
            elif isinstance(arg, R.MissingArg):
                # Empty arg in subscript context — represent as None.
                py_args.append(P.Constant(value=None))
            else:
                py_args.append(self._visit(arg))
        return py_args, py_kwargs

    # -- assignment & top-level control flow -------------------------------

    def _visit_Assign(self, n: R.Assign) -> P.AST:
        """``x <- expr`` / ``x = expr`` → ``x = expr``.

        ``<<-`` global-scope assignment is emitted the same as ``<-`` in
        Phase 2; nested scope semantics aren't translatable to Python's
        rules without ``global`` declarations, which we punt to a later
        phase.
        """
        # The LHS of an Assign is the variable name — NOT a column ref.
        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
        # Ensure target is a Name (or Attribute / Subscript) in Store context.
        if isinstance(target, P.Name):
            target.ctx = P.Store()
        elif isinstance(target, P.Attribute):
            target.ctx = P.Store()
        elif isinstance(target, P.Subscript):
            target.ctx = P.Store()
        value = self._visit(n.value)
        return P.Assign(targets=[target], value=value)

    def _visit_NamedArg(self, n: R.NamedArg) -> P.AST:
        """A NamedArg shouldn't reach _visit at the top level — it should
        be consumed inside _translate_args. If we got here it's a misuse,
        so emit the value alone."""
        return self._visit(n.value)

    def _visit_MissingArg(self, n: R.MissingArg) -> P.AST:
        return P.Constant(value=None)

    # -- subscript / dollar / at ------------------------------------------

    def _visit_Subscript(self, n: R.Subscript) -> P.AST:
        """``df[i]`` / ``df[i, j]`` — Python ``df[i]`` / ``df[i, j]``."""
        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
            if len(n.args) == 1:
                slice_ = self._visit(n.args[0])
            else:
                slice_ = P.Tuple(elts=[self._visit(a) for a in n.args], ctx=P.Load())
        return P.Subscript(value=target, slice=slice_, ctx=P.Load())

    def _visit_DoubleSubscript(self, n: R.DoubleSubscript) -> P.AST:
        """``x[[i]]`` — translate to ``x[i]`` (polars has no double-bracket
        distinction; both flatten to single-element selection)."""
        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
            slice_ = self._visit(n.args[0]) if len(n.args) == 1 else \
                P.Tuple(elts=[self._visit(a) for a in n.args], ctx=P.Load())
        return P.Subscript(value=target, slice=slice_, ctx=P.Load())

    def _visit_Dollar(self, n: R.Dollar) -> P.AST:
        """``df$col`` — polars accepts ``df["col"]`` as the equivalent
        Series-getter, which is the closest hea idiom."""
        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
        return P.Subscript(value=target, slice=P.Constant(value=n.name), ctx=P.Load())

    def _visit_At(self, n: R.At) -> P.AST:
        """``obj@slot`` — Python attribute access ``obj.slot``."""
        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
        return P.Attribute(value=target, attr=n.name, ctx=P.Load())

    # -- formulas / blocks / control flow ----------------------------------

    def _visit_Tilde(self, n: R.Tilde) -> P.AST:
        """Formula. Emit as a string literal so consumers like
        ``hea.lm(formula="y ~ x")`` and ``facet_wrap("~island")`` work.
        The fluent ``y ~ x`` syntax in R has no Python operator
        equivalent without monkey-patching.
        """
        return P.Constant(value=_format_formula(n))

    def _visit_Block(self, n: R.Block) -> P.AST:
        """Brace block. As a statement, becomes the sequence of inner
        statements. As an expression, the value is the last statement —
        the surrounding context decides. v1 only handles the statement
        case; an expression-form block falls through to its last value,
        which is what R does at runtime."""
        if not n.statements:
            return P.Constant(value=None)
        # In Phase 2 scope, only the statement case is required (function
        # bodies, control-flow bodies). The expression case can come later.
        # For safety here, emit the last statement's value.
        return self._visit(n.statements[-1])

    def _visit_If(self, n: R.If) -> P.AST:
        """R's ``if`` is an expression. We translate to a Python ternary
        when the branches are simple expressions, else to a statement-form
        if/else (caller wraps with ``_as_stmt`` as needed)."""
        with self.nse.enter(self.nse.current):
            cond = self._visit(n.cond)
            then = self._visit(n.then)
            otherwise = self._visit(n.otherwise) if n.otherwise is not None else P.Constant(None)
        return P.IfExp(test=cond, body=then, orelse=otherwise)

    def _visit_For(self, n: R.For) -> P.stmt:
        with self.nse.enter(Slot.NONE):
            iterable = self._visit(n.iterable)
            body = self._as_stmt(self._visit(n.body))
        return P.For(
            target=_name(n.var, ctx=P.Store()),
            iter=iterable,
            body=[body],
            orelse=[],
        )

    def _visit_While(self, n: R.While) -> P.stmt:
        with self.nse.enter(Slot.NONE):
            cond = self._visit(n.cond)
            body = self._as_stmt(self._visit(n.body))
        return P.While(test=cond, body=[body], orelse=[])

    def _visit_Repeat(self, n: R.Repeat) -> P.stmt:
        with self.nse.enter(Slot.NONE):
            body = self._as_stmt(self._visit(n.body))
        return P.While(test=P.Constant(True), body=[body], orelse=[])

    def _visit_Break(self, n: R.Break) -> P.stmt:
        return P.Break()

    def _visit_Next(self, n: R.Next) -> P.stmt:
        return P.Continue()

    def _visit_FunctionDef(self, n: R.FunctionDef) -> P.AST:
        """``function(x) body`` → Python ``lambda`` for simple expression
        bodies, ``def`` otherwise. Phase 2 keeps it simple — always lambda.
        Top-level ``f <- function(...) ...`` becomes ``f = lambda ...``."""
        py_args = P.arguments(
            posonlyargs=[],
            args=[P.arg(arg=p.name) for p in n.params],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[self._visit(p.default) for p in n.params if p.default is not None],
            vararg=None,
            kwarg=None,
        )
        with self.nse.enter(Slot.NONE):
            body = self._visit(n.body)
        return P.Lambda(args=py_args, body=body)


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------


def _name(name: str, *, ctx: Optional[P.expr_context] = None) -> P.Name:
    return P.Name(id=name, ctx=ctx or P.Load())


def _attr(value: P.AST, attr: str) -> P.Attribute:
    return P.Attribute(value=value, attr=attr, ctx=P.Load())


def _call(func: P.AST, args: list[P.AST] | None = None, kwargs: list[P.keyword] | None = None) -> P.Call:
    return P.Call(func=func, args=args or [], keywords=kwargs or [])


def _dotted_name(qualified: str) -> P.AST:
    """``"selectors.starts_with"`` → ``ast.Attribute(ast.Name("selectors"), "starts_with")``."""
    parts = qualified.split(".")
    node: P.AST = _name(parts[0])
    for part in parts[1:]:
        node = _attr(node, part)
    return node


_LIBRARY_CALL_NAMES: frozenset[str] = frozenset({"library", "require"})

_NOOP_CALL_NAMES: frozenset[str] = frozenset({
    "suppressMessages", "suppressWarnings", "suppressPackageStartupMessages",
})


def _is_library_call(node) -> bool:
    """``True`` if ``node`` is ``library(pkg)`` / ``require(pkg)``.

    These are tracked separately from other no-ops because the package
    name they declare is used to disambiguate autoload candidates.
    """
    return (
        isinstance(node, R.Call)
        and isinstance(node.func, R.Identifier)
        and node.func.name in _LIBRARY_CALL_NAMES
    )


def _is_noop_call(node) -> bool:
    """``True`` if ``node`` is an R top-level call whose Python equivalent
    is empty (message suppression, etc). Library calls are handled
    separately by :func:`_is_library_call`."""
    return (
        isinstance(node, R.Call)
        and isinstance(node.func, R.Identifier)
        and node.func.name in _NOOP_CALL_NAMES
    )


def _record_library_pkg(node: R.Call, packages: set[str]) -> None:
    """Extract the package name from a library()/require() call. Accepts
    both ``library(dplyr)`` (bare name, R's NSE) and ``library("dplyr")``
    (string form)."""
    if not node.args:
        return
    arg = node.args[0]
    if isinstance(arg, R.Identifier):
        packages.add(arg.name)
    elif isinstance(arg, R.StrLit):
        packages.add(arg.value)


def _make_data_load_stmt(name: str, pkg: str) -> P.stmt:
    """Build a Python AST node for ``<name> = hea.data("<name>", package="<pkg>")``."""
    return P.Assign(
        targets=[P.Name(id=name, ctx=P.Store())],
        value=P.Call(
            func=P.Attribute(value=P.Name("hea", ctx=P.Load()), attr="data", ctx=P.Load()),
            args=[P.Constant(value=name)],
            keywords=[P.keyword(arg="package", value=P.Constant(value=pkg))],
        ),
    )


def _is_named_call(node, name: str) -> bool:
    """``True`` if ``node`` is ``Call(Identifier(name), ...)``."""
    return (
        isinstance(node, R.Call)
        and isinstance(node.func, R.Identifier)
        and node.func.name == name
    )


def _is_ggplot_chain_call(node) -> bool:
    """``True`` iff ``node`` is a Call whose head identifier marks it as a
    ggplot chain extension (geom_*, scale_*, labs, theme, …).

    Plus ``theme(...)`` itself, which the prefix rule already catches
    via ``theme_*``-startswith — but we also want bare ``theme``.
    """
    if not isinstance(node, R.Call) or not isinstance(node.func, R.Identifier):
        return False
    name = node.func.name
    if name == "theme":
        return True
    return is_chain_extension(name)


def _format_formula(t: R.Tilde) -> str:
    """Render a Tilde back to R-style formula text. Matches hea's idiom
    of compact ``"~island"`` / ``"rows ~ cols"`` style — no padding before
    the ``~`` in the unary case."""
    if t.lhs is None:
        return f"~{_unparse_for_formula(t.rhs)}"
    return f"{_unparse_for_formula(t.lhs)} ~ {_unparse_for_formula(t.rhs)}"


def _extract_col_names(cols_arg) -> list[str]:
    """Best-effort extraction of column names from across()'s first arg.

    Accepts: ``a`` (bare ident), ``"a"`` (string), or ``c(a, b, "c")``.
    Rejects: tidy-select helpers (``starts_with("x")``), slices, etc. —
    those need a runtime resolver (Phase 5+). Raise so the user sees the
    gap instead of a silently-wrong expansion.
    """
    if isinstance(cols_arg, R.Identifier):
        return [cols_arg.name]
    if isinstance(cols_arg, R.StrLit):
        return [cols_arg.value]
    if _is_named_call(cols_arg, "c"):
        names: list[str] = []
        for a in cols_arg.args:
            if isinstance(a, R.Identifier):
                names.append(a.name)
            elif isinstance(a, R.StrLit):
                names.append(a.value)
            else:
                raise RTranslateError(
                    f"across() column list contains {type(a).__name__} "
                    "— only bare names and strings are supported in v1",
                    a,
                )
        return names
    raise RTranslateError(
        f"across() col form not supported: {type(cols_arg).__name__} "
        "— pass a bare name, a string, or c(a, b, ...).",
        cols_arg,
    )


def _substitute_identifier(node: R.Node, param_name: str, replacement: R.Identifier) -> R.Node:
    """Recursively replace ``Identifier(name=param_name)`` with ``replacement``.

    Walks every field of every dataclass node, transforming tuples and
    nested dataclasses. Non-dataclass values (strings, ints, tuples of
    primitives like ``Span``) pass through untouched.
    """
    from dataclasses import fields, replace as _dc_replace, is_dataclass

    if isinstance(node, R.Identifier) and node.name == param_name:
        return replacement
    if not is_dataclass(node):
        return node
    new_kwargs = {}
    for f in fields(node):
        v = getattr(node, f.name)
        if isinstance(v, tuple) and v and is_dataclass(v[0]):
            new_kwargs[f.name] = tuple(_substitute_identifier(x, param_name, replacement) for x in v)
        elif is_dataclass(v):
            new_kwargs[f.name] = _substitute_identifier(v, param_name, replacement)
        else:
            new_kwargs[f.name] = v
    return _dc_replace(node, **new_kwargs)


def _unparse_for_formula(node: R.Node) -> str:
    """Render an R AST node back to source text for embedding in a formula
    string. Used only by Tilde — handles the small subset that appears in
    typical formulas (identifiers, calls, arithmetic, ``:``, ``*``)."""
    if isinstance(node, R.Identifier):
        return node.name
    if isinstance(node, R.NumLit):
        return str(node.value)
    if isinstance(node, R.IntLit):
        return str(node.value)
    if isinstance(node, R.UnaryOp):
        return f"{node.op}{_unparse_for_formula(node.operand)}"
    if isinstance(node, R.BinOp):
        return f"{_unparse_for_formula(node.left)} {node.op} {_unparse_for_formula(node.right)}"
    if isinstance(node, R.Call):
        func_text = _unparse_for_formula(node.func)
        args = ", ".join(_unparse_for_formula(a) for a in node.args)
        return f"{func_text}({args})"
    if isinstance(node, R.NamedArg):
        return f"{node.name} = {_unparse_for_formula(node.value)}"
    if isinstance(node, R.StrLit):
        return repr(node.value)
    # Last-resort fallback: opaque marker.
    return f"<{type(node).__name__}>"


class _NullCtx:
    """Drop-in for the case where no NSE slot needs to be pushed — used as
    the ``with`` branch when ``Func.arg_slot`` is None."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _null_ctx() -> _NullCtx:
    return _NullCtx()
