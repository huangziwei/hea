"""R AST → Python AST → Python source.

The scalar / operator / control-flow emitter is merged here with verb
dispatch and NSE handling. The split is logical, not file-level: one
``Translator`` walks the R AST, building stdlib ``ast`` nodes that
``ast.unparse`` finally renders.

Public entry: :func:`translate`. Returns a Python source string. Callers
needing the AST instead can use :class:`Translator` directly.

NSE handling lives in :mod:`hea.translate.nse`; the verb and function
dispatch tables live in :mod:`hea.translate.registry`. This module is the
glue that walks the R AST and asks those tables what to emit.
"""

from __future__ import annotations

import ast as P
import functools
import importlib
import re
from typing import ClassVar

from . import _datasets
from . import gaps as _gaps
from . import r_ast as R
from .nse import NSEContext, Slot
from .r_parser import parse as parse_r
from .registry.functions import FUNCTION_TABLE, Func, resolve_kwarg
from .registry.ggplot import is_chain_extension
from .registry.verbs import VERB_TABLE, Verb


def _callable_exports(module_name: str) -> frozenset[str]:
    """Public names a module exports — callables and module-level
    constants (e.g. ``hea.R.pi``, ``hea.R.LETTERS``).
    """
    import types

    try:
        mod = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return frozenset()
    out: set[str] = set()
    for n in dir(mod):
        if n.startswith("_"):
            continue
        v = getattr(mod, n, None)
        if isinstance(v, types.ModuleType):
            continue
        out.add(n)
    return frozenset(out)


@functools.cache
def _hea_exports() -> frozenset[str]:
    return _callable_exports("hea") - {"R", "plot", "ggplot"}


@functools.cache
def _hea_r_exports() -> frozenset[str]:
    return _callable_exports("hea.R")


@functools.cache
def _hea_tidy_exports() -> frozenset[str]:
    return _callable_exports("hea.tidy")


@functools.cache
def _hea_models_exports() -> frozenset[str]:
    return _callable_exports("hea.models")


@functools.cache
def _hea_family_exports() -> frozenset[str]:
    return _callable_exports("hea.family")


@functools.cache
def _hea_io_exports() -> frozenset[str]:
    return _callable_exports("hea.io")


@functools.cache
def _hea_plot_exports() -> frozenset[str]:
    return _callable_exports("hea.plot")


@functools.cache
def _hea_ggplot_exports() -> frozenset[str]:
    return _callable_exports("hea.ggplot")


def _module_exports(module_name: str) -> frozenset[str]:
    """Public submodule attributes of ``module_name``. Counterpart to
    :func:`_callable_exports` for names like ``hea.selectors`` that
    the translator emits as Attribute roots (``selectors.starts_with``).
    """
    import types

    try:
        mod = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return frozenset()
    out: set[str] = set()
    for n in dir(mod):
        if n.startswith("_"):
            continue
        v = getattr(mod, n, None)
        if isinstance(v, types.ModuleType):
            out.add(n)
    return frozenset(out)


@functools.cache
def _hea_submodules() -> frozenset[str]:
    return _module_exports("hea")


_PY_BUILTINS: frozenset[str] = frozenset(
    __builtins__.keys() if isinstance(__builtins__, dict) else dir(__builtins__)
) | {  # type: ignore[union-attr]
    "True",
    "False",
    "None",
}


_UNPORTED_TAG = "__HEA_UNPORTED__"
_UNPORTED_LINE_RE = re.compile(rf"^\s*['\"]({_UNPORTED_TAG}):([0-9]+)['\"]\s*$")


class RTranslateError(Exception):
    """Raised when a node can't be translated within the documented sublanguage."""

    def __init__(self, message: str, node: R.Node):
        self.node = node
        super().__init__(f"{message} at span {node.span}")  # type: ignore[union-attr]


def translate(
    src: str, *, log_gaps: bool = False, source_label: str = "<inline>"
) -> str:
    """Translate an R source string to a Python source string.

    Raises :class:`RTranslateError` on out-of-grammar inputs; the parser
    may raise :class:`hea.translate.r_parser.RParseError` first.

    ``log_gaps`` controls whether unportable-construct gaps (replacement
    functions, ``with(df, expr)``, Python-keyword kwargs, ...) are
    appended to :mod:`hea.translate.gaps`'s persistent registry. The
    inline UX (``hea.from_R``) usually leaves this off; the parity
    runner sets it.
    """
    prog = parse_r(src)
    return Translator(src=src, log_gaps=log_gaps, source_label=source_label).translate(
        prog
    )


_BINOP_PY = {
    "+": P.Add,
    "-": P.Sub,
    "*": P.Mult,
    "/": P.Div,
    "^": P.Pow,
    "%%": P.Mod,
    "%/%": P.FloorDiv,
}

_CMP_PY = {
    "==": P.Eq,
    "!=": P.NotEq,
    "<": P.Lt,
    "<=": P.LtE,
    ">": P.Gt,
    ">=": P.GtE,
}


_R_GENERIC_METHOD_FORM: frozenset[str] = frozenset(
    {
        "summary",
        "anova",
        "coef",
        "coefficients",
        "residuals",
        "resid",
        "fitted",
        "fitted_values",
        "predict",
        "confint",
        "vcov",
        "logLik",
        "deviance",
        "formula",
        "nobs",
        "AIC",
        "BIC",
        "update",
        "print",
        "plot",
    }
)


class Translator:
    """Stateful walker. One instance per translation."""

    def __init__(
        self, src: str = "", *, log_gaps: bool = False, source_label: str = "<inline>"
    ):
        self.nse = NSEContext()
        self._loaded_packages: set[str] = set()
        self._namespaced_refs: dict[str, str] = {}
        self._shifted_loop_vars: set[str] = set()
        self._index_context: int = 0
        self._with_stack: list[str] = []
        self._src = src
        self._unported: list[tuple[str, str, str, str]] = []
        self._log_gaps = log_gaps
        self._source_label = source_label

    def translate(self, prog: R.Program) -> str:
        module = self._visit_program(prog)
        P.fix_missing_locations(module)
        src = P.unparse(module)
        src = self._rewrite_unported(src)
        return src

    def _emit_unported(
        self,
        node: R.Node,
        kind: str,
        subject: str,
        notes: str = "",
    ) -> P.stmt:
        """Record an unportable construct and return a sentinel statement
        that :meth:`_rewrite_unported` will replace with a comment block."""
        r_text = self._slice_source(node)
        idx = len(self._unported)
        self._unported.append((kind, subject, r_text, notes))
        if self._log_gaps:
            _gaps.log_gap(
                kind=kind,
                subject=subject,
                source=self._source_label,
                snippet=r_text[:200],
                notes=notes,
            )
        return P.Expr(value=P.Constant(value=f"{_UNPORTED_TAG}:{idx}"))

    def _slice_source(self, node: R.Node) -> str:
        """Slice the R source for ``node``'s span. Returns ``"<unknown>"``
        when the translator was constructed without source (older API)."""
        if not self._src:
            return "<unknown>"
        span = getattr(node, "span", None)
        if span is None:
            return "<unknown>"
        try:
            start, end = span
        except (TypeError, ValueError):
            return "<unknown>"
        return self._src[start:end]

    def _rewrite_unported(self, py_source: str) -> str:
        """Replace each ``"__HEA_UNPORTED__:N"`` line in the unparse output
        with the recorded R source as a commented-out block. Indentation
        is preserved so nested unported statements stay attached to their
        enclosing scope."""
        if not self._unported:
            return py_source
        out_lines: list[str] = []
        for line in py_source.splitlines():
            m = _UNPORTED_LINE_RE.match(line)
            if not m:
                out_lines.append(line)
                continue
            indent = line[: len(line) - len(line.lstrip())]
            idx = int(m.group(2))
            kind, subject, r_text, _notes = self._unported[idx]
            out_lines.append(
                f"{indent}# UNPORTED [{kind}: {subject}] — translator declined; original R was:"
            )
            for r_line in r_text.splitlines() or [""]:
                out_lines.append(f"{indent}#   {r_line}")
        return "\n".join(out_lines)

    def _visit_program(self, prog: R.Program) -> P.Module:
        body: list[P.stmt] = []
        for stmt in prog.statements:
            if _is_library_call(stmt):
                _record_library_pkg(stmt, self._loaded_packages)
                continue
            if _is_noop_call(stmt):
                continue
            smart = self._maybe_smart_data_call(stmt)
            if smart is not None:
                body.append(smart)
                continue
            unported = self._maybe_unported(stmt)
            if unported is not None:
                body.append(unported)
                continue
            body.append(self._as_stmt(self._visit(stmt)))

        autoload = self._build_autoload_preamble(body)
        imports = self._build_import_preamble(autoload + body)
        return P.Module(body=imports + autoload + body, type_ignores=[])

    def _maybe_unported(self, stmt: R.Node) -> P.stmt | None:
        """Return a sentinel statement for top-level constructs the
        translator cannot emit as valid Python; otherwise ``None``."""
        if (
            isinstance(stmt, R.Assign)
            and isinstance(stmt.target, R.Call)
            and isinstance(stmt.target.func, R.Identifier)
        ):
            fn = stmt.target.func.name
            return self._emit_unported(
                stmt,
                kind="replacement_function",
                subject=f"{fn}<-",
                notes=f"R's `{fn}(x) <- v` setter has no direct hea analog yet.",
            )
        kw = _first_python_keyword_call(stmt)
        if kw is not None:
            return self._emit_unported(
                stmt,
                kind="python_keyword_call",
                subject=kw,
                notes=f"`{kw}(...)` collides with Python's `{kw}` keyword; needs renamed-helper translation or different surface.",
            )
        return None

    def _build_import_preamble(self, body: list[P.stmt]) -> list[P.stmt]:
        """Scan ``body`` for Load Name + root-of-Attribute references that
        aren't locally bound; emit minimal imports from ``hea``, ``hea.R``,
        and ``hea.plot`` (in that priority).
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
                elif (
                    isinstance(node, P.Attribute)
                    and isinstance(node.value, P.Name)
                    and isinstance(node.value.ctx, P.Load)
                ):
                    referenced.add(node.value.id)

        candidates = referenced - defined - _PY_BUILTINS

        r_names = sorted(n for n in candidates if n in _hea_r_exports())
        used = set(r_names)
        tidy_names = sorted(n for n in (candidates - used) if n in _hea_tidy_exports())
        used |= set(tidy_names)
        models_names = sorted(
            n for n in (candidates - used) if n in _hea_models_exports()
        )
        used |= set(models_names)
        family_names = sorted(
            n for n in (candidates - used) if n in _hea_family_exports()
        )
        used |= set(family_names)
        hea_names = sorted(n for n in (candidates - used) if n in _hea_exports())
        used |= set(hea_names)
        io_names = sorted(n for n in (candidates - used) if n in _hea_io_exports())
        used |= set(io_names)
        plot_names = sorted(n for n in (candidates - used) if n in _hea_plot_exports())
        used |= set(plot_names)
        ggplot_names = sorted(
            n for n in (candidates - used) if n in _hea_ggplot_exports()
        )
        used |= set(ggplot_names)
        submod_names = sorted(n for n in (candidates - used) if n in _hea_submodules())

        out: list[P.stmt] = []
        if "hea" in referenced:
            out.append(P.Import(names=[P.alias(name="hea", asname=None)]))
        if "np" in referenced:
            out.append(P.Import(names=[P.alias(name="numpy", asname="np")]))
        if hea_names or submod_names:
            out.append(
                P.ImportFrom(
                    module="hea",
                    names=[
                        P.alias(name=n, asname=None) for n in (hea_names + submod_names)
                    ],
                    level=0,
                )
            )
        if r_names:
            out.append(
                P.ImportFrom(
                    module="hea.R",
                    names=[P.alias(name=n, asname=None) for n in r_names],
                    level=0,
                )
            )
        if tidy_names:
            out.append(
                P.ImportFrom(
                    module="hea.tidy",
                    names=[P.alias(name=n, asname=None) for n in tidy_names],
                    level=0,
                )
            )
        if models_names:
            out.append(
                P.ImportFrom(
                    module="hea.models",
                    names=[P.alias(name=n, asname=None) for n in models_names],
                    level=0,
                )
            )
        if family_names:
            out.append(
                P.ImportFrom(
                    module="hea.family",
                    names=[P.alias(name=n, asname=None) for n in family_names],
                    level=0,
                )
            )
        if io_names:
            out.append(
                P.ImportFrom(
                    module="hea.io",
                    names=[P.alias(name=n, asname=None) for n in io_names],
                    level=0,
                )
            )
        if plot_names:
            out.append(
                P.ImportFrom(
                    module="hea.plot",
                    names=[P.alias(name=n, asname=None) for n in plot_names],
                    level=0,
                )
            )
        if ggplot_names:
            out.append(
                P.ImportFrom(
                    module="hea.ggplot",
                    names=[P.alias(name=n, asname=None) for n in ggplot_names],
                    level=0,
                )
            )
        return out

    def _maybe_smart_data_call(self, stmt: R.Node) -> P.AST | None:
        """If ``stmt`` is a standalone ``data("X", package="Y")`` call,
        emit it as ``X = data("X", package="Y")`` (bare call — import
        preamble will pull in ``from hea import data``).
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
        pkg: str | None = None
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
                func=P.Name(id="data", ctx=P.Load()),
                args=[P.Constant(value=name)],
                keywords=keywords,
            ),
        )

    def _build_autoload_preamble(self, body: list[P.stmt]) -> list[P.stmt]:
        """Scan ``body`` for bare ``Name`` references that aren't bound
        anywhere and match a known dataset; emit a ``hea.data(...)``
        assignment for each.
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
        emitted_helpers = frozenset(
            f.hea_name.split(".", 1)[0] for f in FUNCTION_TABLE.values()
        )
        candidates = sorted(
            referenced - defined - _datasets.DATASET_REF_EXCLUSIONS - emitted_helpers
        )
        out: list[P.stmt] = []
        emitted: set[str] = set()
        for name, pkg in sorted(self._namespaced_refs.items()):
            if name in defined or name not in referenced:
                continue
            out.append(_make_data_load_stmt(name, pkg))
            emitted.add(name)
        for name in candidates:
            if name in emitted:
                continue
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

    def _visit(self, node: R.Node) -> P.AST:
        """Dispatch on R AST node type."""
        method = getattr(self, "_visit_" + type(node).__name__, None)
        if method is None:
            raise RTranslateError(f"no translator for {type(node).__name__}", node)
        return method(node)

    def _visit_NumLit(self, n: R.NumLit) -> P.AST:
        if n.value.is_integer() and -(2**53) < n.value < 2**53:
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
        if self.nse.is_expr():
            return _call(
                _attr(_attr(_name("hea"), "tidy"), "lit"),
                [P.Constant(None)],
            )
        return P.Constant(value=None)

    def _visit_InfLit(self, n: R.InfLit) -> P.AST:
        return _call(_name("float"), [P.Constant("inf")])

    def _visit_NanLit(self, n: R.NanLit) -> P.AST:
        return _call(_name("float"), [P.Constant("nan")])

    def _visit_Identifier(self, n: R.Identifier) -> P.AST:
        slot = self.nse.current
        if slot is Slot.EXPR:
            return _call(_name("col"), [P.Constant(n.name)])
        if slot is Slot.COLUMN_NAME:
            return P.Constant(value=n.name)
        if (
            self._with_stack
            and n.name not in FUNCTION_TABLE
            and n.name not in VERB_TABLE
        ):
            df_name = self._with_stack[-1]
            return P.Subscript(
                value=_name(df_name),
                slice=P.Constant(value=n.name),
                ctx=P.Load(),
            )
        return _name(_to_py_identifier(n.name))

    def _visit_UnaryOp(self, n: R.UnaryOp) -> P.AST:
        operand = self._visit(n.operand)
        if n.op == "-":
            return P.UnaryOp(P.USub(), operand)
        if n.op == "+":
            return P.UnaryOp(P.UAdd(), operand)
        if n.op == "!":
            if self.nse.is_expr():
                return P.UnaryOp(P.Invert(), operand)
            if self.nse.current is Slot.COLUMN_NAME:
                if (
                    isinstance(operand, P.Call)
                    and isinstance(operand.func, P.Name)
                    and operand.func.id == "cols_between"
                ):
                    return P.UnaryOp(P.Invert(), operand)
                return P.UnaryOp(
                    P.Invert(),
                    _call(
                        _attr(_name("selectors"), "by_name"),
                        [operand],
                    ),
                )
            return P.UnaryOp(P.Not(), operand)
        if n.op == "?":
            raise RTranslateError("R help operator `?` not supported", n)
        raise RTranslateError(f"unknown unary operator {n.op!r}", n)

    def _visit_BinOp(self, n: R.BinOp) -> P.AST:
        op = n.op

        if op == "+" and _is_ggplot_chain_call(n.right):
            return self._emit_ggplot_chain_step(n.left, n.right)

        if (
            op == "-"
            and isinstance(n.left, R.Identifier)
            and n.left.name in self._shifted_loop_vars
            and isinstance(n.right, (R.IntLit, R.NumLit))
            and n.right.value == 1
        ):
            return self._visit(n.left)

        left = self._visit(n.left)
        right = self._visit(n.right)

        py_op = _BINOP_PY.get(op)
        if py_op is not None:
            return P.BinOp(left=left, op=py_op(), right=right)

        cmp = _CMP_PY.get(op)
        if cmp is not None:
            return P.Compare(left=left, ops=[cmp()], comparators=[right])

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

        if op == ":":
            if self.nse.current is Slot.COLUMN_NAME:
                return _call(_name("cols_between"), [left, right])
            if self._index_context and isinstance(left, P.Constant) and left.value == 1:
                return _call(_name("range"), [right])
            return _call(_name("seq"), [left, right])

        if op == "::" or op == ":::":
            if isinstance(n.left, R.Identifier):
                pkg_name = n.left.name
                self._loaded_packages.add(pkg_name)
                if isinstance(n.right, R.Identifier):
                    self._namespaced_refs.setdefault(n.right.name, pkg_name)
            return right

        if op == "%in%":
            return _call(_attr(left, "is_in"), [right])

        if op == "%*%":
            return P.BinOp(left=left, op=P.MatMult(), right=right)

        if op.startswith("%") and op.endswith("%"):
            fname = "".join(c for c in op.strip("%") if c.isalnum() or c == "_")
            if not fname:
                fname = "_infix_" + "".join(f"{ord(c):x}" for c in op.strip("%"))
            return _call(_name(fname), [left, right])

        raise RTranslateError(f"unknown binary operator {op!r}", n)

    def _visit_Pipe(self, n: R.Pipe) -> P.AST:
        """``lhs |> rhs`` or ``lhs %>% rhs``."""
        if not isinstance(n.rhs, R.Call):
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
            elif (
                isinstance(arg, R.NamedArg)
                and isinstance(arg.value, R.Identifier)
                and arg.value.name == "."
            ):
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

    def _visit_Call(self, n: R.Call) -> P.AST:
        """Three dispatch layers, tried in order:"""
        func = n.func
        if isinstance(func, R.BinOp) and func.op in ("::", ":::"):
            func = func.right  # type: ignore[assignment]

        if isinstance(func, R.Identifier):
            name = func.name

            if name == "ggplot":
                return self._emit_ggplot_root(n.args)

            if name == "with":
                return self._emit_with_call(n.args)

            if name == "slice" and n.args:
                emitted = self._maybe_emit_slice(n.args)
                if emitted is not None:
                    return emitted

            verb = VERB_TABLE.get(name)
            if verb is not None and n.args:
                return self._emit_verb_call(verb, n.args)

            helper = FUNCTION_TABLE.get(name)
            if helper is not None:
                return self._emit_helper_call(helper, name, n.args)

            name = self._R_MODEL_ALIASES.get(name, name)

            synthesized = self._maybe_lm_no_data(name, n.args)
            if synthesized is not None:
                return synthesized

            if name in _R_GENERIC_METHOD_FORM:
                positional = [a for a in n.args if not isinstance(a, R.NamedArg)]
                if len(positional) == 1 and isinstance(positional[0], R.Identifier):
                    receiver = _name(_to_py_identifier(positional[0].name))
                    rest = tuple(a for a in n.args if isinstance(a, R.NamedArg))
                    py_args, py_kwargs = self._translate_args(rest)
                    return _call(_attr(receiver, name), py_args, py_kwargs)

            return self._emit_regular_call(name, n.args)

        if isinstance(func, R.Dollar):
            target = self._visit(func.target)
            args, kwargs = self._translate_args(n.args)
            callee = P.Attribute(value=target, attr=func.name, ctx=P.Load())
            return _call(callee, args, kwargs)

        callee = self._visit(func)
        args, kwargs = self._translate_args(n.args)
        return _call(callee, args, kwargs)

    def _emit_verb_call(self, verb: Verb, args: tuple[R.Node, ...]) -> P.AST:
        """First arg becomes the receiver; rest walked under ``verb.slot``."""
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

    def _maybe_emit_slice(self, args: tuple[R.Node, ...]) -> P.AST | None:
        """dplyr ``slice(df, <positions>)`` → ``df.slice([...])`` / ``df.slice(drop([...]))``."""
        if len(args) != 2 or any(isinstance(a, R.NamedArg) for a in args):
            return None
        idx = args[1]
        negated = isinstance(idx, R.UnaryOp) and idx.op == "-"
        inner = idx.operand if negated else idx
        positions = self._shift_slice_positions(inner)
        if positions is None:
            return None
        receiver = self._visit(args[0])
        arg = _call(_name("drop"), [positions]) if negated else positions
        return _call(_attr(receiver, "slice"), [arg])

    def _shift_slice_positions(self, node: R.Node) -> P.AST | None:
        """Shift an R 1-based ``slice`` index to a 0-based Python positions
        expression, or ``None`` if it isn't a statically shiftable literal."""
        if (
            isinstance(node, R.Call)
            and isinstance(node.func, R.Identifier)
            and node.func.name == "c"
            and node.args
            and all(_is_pos_int_lit(x) for x in node.args)
        ):
            return P.List(
                elts=[P.Constant(value=int(x.value) - 1) for x in node.args],
                ctx=P.Load(),
            )
        if (
            isinstance(node, R.BinOp)
            and node.op == ":"
            and _is_pos_int_lit(node.left)
            and _is_pos_int_lit(node.right)
        ):
            a, b = int(node.left.value), int(node.right.value)
            if a == 1:
                return _call(_name("range"), [P.Constant(value=b)])
            return _call(_name("range"), [P.Constant(value=a - 1), P.Constant(value=b)])
        if (
            isinstance(node, R.Call)
            and isinstance(node.func, R.Identifier)
            and node.func.name == "n"
            and not node.args
        ):
            return P.List(elts=[P.Constant(value=-1)], ctx=P.Load())
        if _is_pos_int_lit(node):
            return P.List(elts=[P.Constant(value=int(node.value) - 1)], ctx=P.Load())
        return None

    def _emit_ggplot_root(self, args: tuple[R.Node, ...]) -> P.AST:
        """``ggplot(df, aes(x = a, y = b))`` → ``df.ggplot(x="a", y="b")``."""
        if not args:
            return _call(_attr(_attr(_name("hea"), "ggplot"), "ggplot"))
        receiver = self._visit(args[0])
        kwargs = self._collect_ggplot_kwargs(args[1:])
        return _call(_attr(receiver, "ggplot"), [], kwargs)

    def _emit_ggplot_chain_step(self, left: R.Node, right: R.Call) -> P.AST:
        """``<plot> + <ext>(args)`` → ``<plot translated>.<ext>(args)``."""
        receiver = self._visit(left)
        func_name = right.func.name  # type: ignore[attr-defined]
        kwargs = self._collect_ggplot_kwargs(right.args)
        positional: list[P.AST] = []
        with self.nse.enter(Slot.NONE):
            for arg in right.args:
                if isinstance(arg, R.NamedArg):
                    continue  # collected by _collect_ggplot_kwargs
                if _is_named_call(arg, "aes"):
                    continue
                if isinstance(arg, R.Tilde):
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

    _AES_POS_AESTHETICS = ("x", "y")

    def _translate_aes_args(self, args: tuple[R.Node, ...]) -> list[P.keyword]:
        """Translate ``aes()`` args to ggplot kwargs."""
        kwargs: list[P.keyword] = []
        pos_idx = 0
        for arg in args:
            if isinstance(arg, R.NamedArg):
                name = arg.name
                value = self._translate_aes_value(arg.value)
                kwargs.append(P.keyword(arg=name, value=value))
                continue
            if pos_idx < len(self._AES_POS_AESTHETICS):
                name = self._AES_POS_AESTHETICS[pos_idx]
                value = self._translate_aes_value(arg)
                kwargs.append(P.keyword(arg=name, value=value))
                pos_idx += 1
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
        """Translate-time expansion of ``across(cols, fn[, .names = ...])``."""
        if len(call.args) < 2:
            raise RTranslateError("across() requires (cols, fns) — got fewer", call)

        cols_arg = call.args[0]
        fn_arg = call.args[1]

        for extra in call.args[2:]:
            if isinstance(extra, R.NamedArg) and extra.name == ".names":
                raise RTranslateError(
                    "across(.names = ...) not supported",
                    extra,
                )
        if _is_named_call(fn_arg, "list"):
            raise RTranslateError(
                "across() with list(...) of functions not supported",
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
        """Apply ``fn_arg`` to ``col``, returning an R AST node."""
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

    def _emit_helper_call(
        self, helper: Func, r_name: str, args: tuple[R.Node, ...]
    ) -> P.AST:
        """Translate one of the registered helpers (mean, desc, n, …)."""
        if helper.hea_name == "__list__":
            return self._emit_c_call(args)

        if helper.hea_name == "case_when":
            return self._emit_case_when(args)

        if helper.hea_name == "__data_frame__":
            return self._emit_data_frame_call(args)

        if helper.hea_name == "__tribble__":
            return self._emit_tribble_call(args)

        if helper.hea_name == "__where__":
            return self._emit_where_call(args)

        if helper.hea_name == "__join_by__":
            return self._emit_join_by_call(args)

        if helper.hea_name == "__quote__":
            return self._emit_quote_call(args)

        arg_slot_ctx = (
            self.nse.enter(helper.arg_slot)
            if helper.arg_slot is not None
            else _null_ctx()
        )

        if helper.form == "method" and self.nse.is_expr() and args:
            first = args[0]
            if isinstance(first, R.Identifier):
                receiver = _call(_name("col"), [P.Constant(first.name)])
            else:
                receiver = self._visit(first)
            with arg_slot_ctx:
                rest_args, rest_kwargs = self._translate_args(args[1:])
            rest_kwargs = [kw for kw in rest_kwargs if kw.arg != "na_rm"]
            return _call(_attr(receiver, helper.hea_name), rest_args, rest_kwargs)

        if helper.drop_kwargs:
            args = tuple(
                a
                for a in args
                if not (isinstance(a, R.NamedArg) and a.name in helper.drop_kwargs)
            )

        with arg_slot_ctx:
            py_args, py_kwargs = self._translate_args(args)

        callee = _dotted_name(helper.hea_name)
        return _call(callee, py_args, py_kwargs)

    def _emit_regular_call(self, name: str, args: tuple[R.Node, ...]) -> P.AST:
        """Unknown function — emit as-is, normalizing the name into a
        valid Python identifier (dot→underscore, keyword→trailing-``_``)."""
        py_args, py_kwargs = self._translate_args(args)
        return _call(_name(_to_py_identifier(name)), py_args, py_kwargs)

    def _emit_with_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``with(df, expr)`` — R's NSE binding."""
        if len(args) < 2 or not isinstance(args[0], R.Identifier):
            return self._emit_regular_call("with", args)
        df_name = _to_py_identifier(args[0].name)
        body = args[1]
        self._with_stack.append(df_name)
        try:
            return self._visit(body)
        finally:
            self._with_stack.pop()

    _LM_LIKE: frozenset[str] = frozenset(
        {"lm", "glm", "gam", "bam", "gmm", "lmer", "glmer"}
    )
    _R_MODEL_ALIASES: ClassVar[dict[str, str]] = {"lmer": "gmm", "glmer": "gmm"}
    _FORMULA_OPS: frozenset[str] = frozenset({"+", "-", "*", "/", ":", "^", "|"})

    def _maybe_lm_no_data(self, name: str, args: tuple[R.Node, ...]) -> P.AST | None:
        """Catch ``lm(y ~ x)`` / ``glm(...)`` / etc. with no ``data`` arg."""
        if name not in self._LM_LIKE:
            return None
        if not args or not isinstance(args[0], R.Tilde):
            return None

        positional = [a for a in args if not isinstance(a, R.NamedArg)]
        has_data_kwarg = any(
            isinstance(a, R.NamedArg) and a.name in ("data", ".data") for a in args
        )
        if has_data_kwarg or len(positional) >= 2:
            return None

        formula = args[0]
        pairs: list[tuple[str, R.Node]] = []
        seen: set[str] = set()
        counter = [0]

        def take(node: R.Node) -> R.Node:
            if isinstance(node, R.Identifier):
                if node.name not in seen:
                    pairs.append((node.name, node))
                    seen.add(node.name)
                return node
            new_name = f"term_{counter[0]}"
            counter[0] += 1
            pairs.append((new_name, node))
            seen.add(new_name)
            return R.Identifier(name=new_name, span=node.span)

        def walk(side: R.Node) -> R.Node:
            if isinstance(side, R.BinOp) and side.op in self._FORMULA_OPS:
                return R.BinOp(
                    op=side.op,
                    left=walk(side.left),
                    right=walk(side.right),
                    span=side.span,
                )
            if isinstance(side, R.UnaryOp) and side.op in ("-", "+"):
                return R.UnaryOp(op=side.op, operand=walk(side.operand), span=side.span)
            if isinstance(side, (R.NumLit, R.IntLit, R.BoolLit)):
                return side
            return take(side)

        new_lhs = walk(formula.lhs) if formula.lhs is not None else None
        new_rhs = walk(formula.rhs)
        new_tilde = R.Tilde(lhs=new_lhs, rhs=new_rhs, span=formula.span)
        formula_str = _format_formula(new_tilde)

        keys: list[P.AST] = []
        values: list[P.AST] = []
        for col_name, ast_node in pairs:
            keys.append(P.Constant(value=col_name))
            values.append(self._visit(ast_node))
        df_arg = _call(
            _attr(_attr(_name("hea"), "tidy"), "DataFrame"),
            [P.Dict(keys=keys, values=values)],
            [],
        )

        py_kwargs: list[P.keyword] = []
        for a in args[1:]:
            if isinstance(a, R.NamedArg):
                alias = resolve_kwarg(a.name)
                value = self._visit(a.value)
                py_kwargs.append(P.keyword(arg=alias.py_name, value=value))
        py_kwargs.append(P.keyword(arg="data", value=df_arg))

        return _call(
            _name(_to_py_identifier(name)),
            [P.Constant(value=formula_str)],
            py_kwargs,
        )

    def _emit_case_when(self, args: tuple[R.Node, ...]) -> P.AST:
        """``case_when(cond1 ~ val1, cond2 ~ val2, .default = d)`` →
        ``case_when((c1, v1), (c2, v2), default=d)``.
        """
        tuples: list[P.AST] = []
        kwargs: list[P.keyword] = []
        with self.nse.enter(Slot.EXPR):
            for arg in args:
                if isinstance(arg, R.Tilde):
                    if arg.lhs is None:
                        kwargs.append(
                            P.keyword(arg="default", value=self._visit(arg.rhs))
                        )
                        continue
                    cond = self._visit(arg.lhs)
                    value = self._visit(arg.rhs)
                    tuples.append(P.Tuple(elts=[cond, value], ctx=P.Load()))
                elif isinstance(arg, R.NamedArg):
                    alias = resolve_kwarg(arg.name)
                    slot_ctx = (
                        self.nse.enter(alias.value_slot)
                        if alias.value_slot is not None
                        else _null_ctx()
                    )
                    with slot_ctx:
                        value = self._visit(arg.value)
                    kwargs.append(P.keyword(arg=alias.py_name, value=value))
                else:
                    tuples.append(self._visit(arg))
        return _call(_name("case_when"), tuples, kwargs)

    _WHERE_PREDICATE_MAP: ClassVar[dict[str, str]] = {
        "is.character": "string",
        "is.string": "string",
        "is.numeric": "numeric",
        "is.double": "float",
        "is.integer": "integer",
        "is.logical": "boolean",
        "is.boolean": "boolean",
        "is.factor": "categorical",
        "is.Date": "date",
        "is.POSIXct": "datetime",
        "is.POSIXlt": "datetime",
    }

    def _emit_quote_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``quote(x[i]^2 == 0)`` → ``quote('x[i]^2 == 0')``."""
        positional = [a for a in args if not isinstance(a, R.NamedArg)]
        if not positional:
            return _call(_name("quote"), [P.Constant(value="")])
        src = _unparse_for_plotmath(positional[0])
        return _call(_name("quote"), [P.Constant(value=src)])

    _JOIN_BY_BINOPS: frozenset[str] = frozenset({"==", "<", "<=", ">", ">=", "!="})

    def _emit_join_by_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``join_by(x, dest == faa, closest(t >= u))`` — dplyr's NSE
        join spec. Each arg gets its own treatment:
        """
        py_args: list[P.AST] = []
        for a in args:
            if isinstance(a, R.Identifier):
                py_args.append(P.Constant(value=a.name))
                continue
            if isinstance(a, R.BinOp) and a.op in self._JOIN_BY_BINOPS:
                with self.nse.enter(Slot.EXPR):
                    py_args.append(self._visit(a))
                continue
            with self.nse.enter(Slot.EXPR):
                py_args.append(self._visit(a))
        return _call(_name("join_by"), py_args)

    def _emit_where_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``where(is.character)`` → ``selectors.string()``. Known
        R-predicate identifiers map to polars selector constructors;
        unknown ones fall through to ``where(<id>)`` and will surface
        as a runtime ``NameError`` so the gap is visible.
        """
        if len(args) == 1 and isinstance(args[0], R.Identifier):
            sel = self._WHERE_PREDICATE_MAP.get(args[0].name)
            if sel is not None:
                return _call(_attr(_name("selectors"), sel), [])
        py_args, py_kwargs = self._translate_args(args)
        return _call(_name("where"), py_args, py_kwargs)

    def _emit_tribble_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``tribble(~a, ~b, 1, "x", 2, "y")`` →
        ``hea.DataFrame({"a": [1, 2], "b": ["x", "y"]})``.
        """
        col_names: list[str] = []
        data_args: list[R.Node] = []
        seen_data = False
        for a in args:
            if isinstance(a, R.MissingArg):
                continue
            if (
                not seen_data
                and isinstance(a, R.Tilde)
                and a.lhs is None
                and isinstance(a.rhs, R.Identifier)
            ):
                col_names.append(a.rhs.name)
                continue
            seen_data = True
            data_args.append(a)
        n = len(col_names)
        if n == 0:
            return self._emit_data_frame_call(args)
        keys: list[P.AST] = [P.Constant(value=name) for name in col_names]
        columns: list[list[P.AST]] = [[] for _ in range(n)]
        for i, a in enumerate(data_args):
            columns[i % n].append(self._visit(a))
        values: list[P.AST] = [P.List(elts=col, ctx=P.Load()) for col in columns]
        return _call(
            _attr(_attr(_name("hea"), "tidy"), "DataFrame"),
            [P.Dict(keys=keys, values=values)],
        )

    def _emit_data_frame_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``data.frame(a = c(1, 2), b = c("x", "y"))`` →
        ``hea.DataFrame({"a": [1, 2], "b": ["x", "y"]})``.
        """
        literal_keys: list[P.AST] = []
        literal_values: list[P.AST] = []
        expr_columns: list[tuple[str, P.AST]] = []
        for i, arg in enumerate(args):
            if isinstance(arg, R.MissingArg):
                continue
            if isinstance(arg, R.NamedArg):
                if arg.name == "stringsAsFactors":
                    continue
                with self.nse.enter(Slot.EXPR):
                    value = self._visit(arg.value)
                if _contains_col_call(value):
                    expr_columns.append((arg.name, value))
                else:
                    literal_keys.append(P.Constant(value=arg.name))
                    literal_values.append(value)
            else:
                with self.nse.enter(Slot.EXPR):
                    value = self._visit(arg)
                key_name = f"V{i + 1}"
                if _contains_col_call(value):
                    expr_columns.append((key_name, value))
                else:
                    literal_keys.append(P.Constant(value=key_name))
                    literal_values.append(value)
        df_call: P.AST = _call(
            _attr(_attr(_name("hea"), "tidy"), "DataFrame"),
            [P.Dict(keys=literal_keys, values=literal_values)],
        )
        if expr_columns:
            aliased = [
                _call(_attr(expr, "alias"), [P.Constant(value=name)])
                for name, expr in expr_columns
            ]
            df_call = _call(_attr(df_call, "with_columns"), aliased)
        return df_call

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
                    keys.append(P.Constant(value=None))
                    values.append(self._visit(a))
            return P.Dict(keys=keys, values=values)
        elems = [self._visit(a) for a in args]
        if args and all(_is_numeric_literal(a) for a in args):
            return _call(
                _attr(_name("np"), "array"),
                [P.List(elts=list(elems), ctx=P.Load())],
            )
        return P.List(elts=list(elems), ctx=P.Load())

    def _translate_args(
        self, args: tuple[R.Node, ...]
    ) -> tuple[list[P.AST], list[P.keyword]]:
        """Walk an R argument list, splitting positional from named."""
        py_args: list[P.AST] = []
        py_kwargs: list[P.keyword] = []
        name_counts: dict[str, int] = {}
        for arg in args:
            if isinstance(arg, R.NamedArg):
                py_name = resolve_kwarg(arg.name).py_name
                name_counts[py_name] = name_counts.get(py_name, 0) + 1
        merged_keys: list[str] = []  # insertion order
        merged_values: dict[str, list[P.AST]] = {}
        for arg in args:
            if isinstance(arg, R.NamedArg):
                alias = resolve_kwarg(arg.name)
                if alias.value_slot is not None:
                    with self.nse.enter(alias.value_slot):
                        value = self._visit(arg.value)
                else:
                    value = self._visit(arg.value)
                is_id = alias.py_name.isidentifier()
                is_dup = name_counts[alias.py_name] > 1
                if is_id and not is_dup:
                    py_kwargs.append(P.keyword(arg=alias.py_name, value=value))
                else:
                    if alias.py_name not in merged_values:
                        merged_keys.append(alias.py_name)
                        merged_values[alias.py_name] = []
                    merged_values[alias.py_name].append(value)
            elif isinstance(arg, R.MissingArg):
                py_args.append(P.Constant(value=None))
            else:
                py_args.append(self._visit(arg))
        if merged_keys:
            dict_keys: list[P.AST] = []
            dict_values: list[P.AST] = []
            for name in merged_keys:
                vals = merged_values[name]
                dict_keys.append(P.Constant(value=name))
                if len(vals) == 1:
                    dict_values.append(vals[0])
                else:
                    dict_values.append(P.List(elts=vals, ctx=P.Load()))
            py_kwargs.append(
                P.keyword(
                    arg=None,
                    value=P.Dict(keys=dict_keys, values=dict_values),
                )
            )
        return py_args, py_kwargs

    def _visit_Assign(self, n: R.Assign) -> P.AST:
        """``x <- expr`` / ``x = expr`` → ``x = expr``."""
        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
        if isinstance(target, (P.Name, P.Attribute, P.Subscript)):
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

    def _visit_Subscript(self, n: R.Subscript) -> P.AST:
        """``df[i]`` / ``df[i, j]`` — Python ``df[i]`` / ``df[i, j]``."""

        def _arg(a):
            if isinstance(a, R.MissingArg):
                return P.Slice(lower=None, upper=None, step=None)
            if isinstance(a, R.BinOp) and a.op == ":":
                return self._range_subscript(a)
            if isinstance(a, R.IntLit):
                return P.Constant(value=a.value - 1)
            if isinstance(a, R.NumLit) and a.value == int(a.value) and a.value > 0:
                return P.Constant(value=int(a.value) - 1)
            if (
                isinstance(a, R.Call)
                and isinstance(a.func, R.Identifier)
                and a.func.name == "c"
                and a.args
                and all(
                    (isinstance(x, R.IntLit) and x.value > 0)
                    or (
                        isinstance(x, R.NumLit)
                        and x.value == int(x.value)
                        and x.value > 0
                    )
                    for x in a.args
                )
            ):
                elts = [P.Constant(value=int(x.value) - 1) for x in a.args]
                return P.List(elts=elts, ctx=P.Load())
            if (
                isinstance(a, R.BinOp)
                and a.op == "+"
                and isinstance(a.right, (R.IntLit, R.NumLit))
                and a.right.value == 1
            ):
                self._index_context += 1
                try:
                    return self._visit(a.left)
                finally:
                    self._index_context -= 1
            self._index_context += 1
            try:
                return self._visit(a)
            finally:
                self._index_context -= 1

        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
            if len(n.args) == 1:
                slice_ = _arg(n.args[0])
            else:
                slice_ = P.Tuple(elts=[_arg(a) for a in n.args], ctx=P.Load())
        return P.Subscript(value=target, slice=slice_, ctx=P.Load())

    def _range_subscript(self, bin_op: R.BinOp) -> P.AST:
        """Emit a Python slice for an R ``a:b`` index expression."""
        with self.nse.enter(Slot.NONE):
            left = self._visit(bin_op.left)
            right = self._visit(bin_op.right)
        if isinstance(left, P.Constant) and left.value == 1:
            return P.Slice(lower=None, upper=right, step=None)
        shifted = P.BinOp(left=left, op=P.Sub(), right=P.Constant(value=1))
        return P.Slice(lower=shifted, upper=right, step=None)

    def _visit_DoubleSubscript(self, n: R.DoubleSubscript) -> P.AST:
        """``x[[i]]`` — translate to ``x[i]`` (polars has no double-bracket
        distinction; both flatten to single-element selection)."""

        def _arg(a):
            if isinstance(a, R.MissingArg):
                return P.Slice(lower=None, upper=None, step=None)
            return self._visit(a)

        with self.nse.enter(Slot.NONE):
            target = self._visit(n.target)
            slice_ = (
                _arg(n.args[0])
                if len(n.args) == 1
                else P.Tuple(elts=[_arg(a) for a in n.args], ctx=P.Load())
            )
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
        the surrounding context decides. Only the statement case is
        handled; an expression-form block falls through to its last value,
        which is what R does at runtime."""
        if not n.statements:
            return P.Constant(value=None)
        return self._visit(n.statements[-1])

    def _visit_If(self, n: R.If) -> P.AST:
        """R's ``if`` is an expression. We translate to a Python ternary
        when the branches are simple expressions, else to a statement-form
        if/else (caller wraps with ``_as_stmt`` as needed)."""
        with self.nse.enter(self.nse.current):
            cond = self._visit(n.cond)
            then = self._visit(n.then)
            otherwise = (
                self._visit(n.otherwise)
                if n.otherwise is not None
                else P.Constant(None)
            )
        return P.IfExp(test=cond, body=then, orelse=otherwise)

    def _visit_For(self, n: R.For) -> P.stmt:
        iterable, shifted = self._visit_for_iter(n.iterable)
        if shifted:
            self._shifted_loop_vars.add(n.var)
        try:
            with self.nse.enter(Slot.NONE):
                body = self._visit_block_as_stmts(n.body)
        finally:
            if shifted:
                self._shifted_loop_vars.discard(n.var)
        return P.For(
            target=_name(n.var, ctx=P.Store()),
            iter=iterable,
            body=body,
            orelse=[],
        )

    def _visit_block_as_stmts(self, body: R.Node) -> list[P.stmt]:
        """Translate a control-flow body to a list of Python statements."""
        if isinstance(body, R.Block):
            if not body.statements:
                return [P.Pass()]
            return [self._as_stmt(self._visit(s)) for s in body.statements]
        return [self._as_stmt(self._visit(body))]

    def _visit_for_iter(self, iter_node: R.Node) -> tuple[P.AST, bool]:
        """Translate the iter of an R ``for(i in <iter>)``."""
        if isinstance(iter_node, R.BinOp) and iter_node.op == ":":
            with self.nse.enter(Slot.NONE):
                left = self._visit(iter_node.left)
                right = self._visit(iter_node.right)
            if isinstance(left, P.Constant) and left.value == 1:
                return _call(_name("range"), [right]), True
            shifted = P.BinOp(left=left, op=P.Sub(), right=P.Constant(value=1))
            return _call(_name("range"), [shifted, right]), True
        with self.nse.enter(Slot.NONE):
            return self._visit(iter_node), False

    def _visit_While(self, n: R.While) -> P.stmt:
        with self.nse.enter(Slot.NONE):
            cond = self._visit(n.cond)
            body = self._visit_block_as_stmts(n.body)
        return P.While(test=cond, body=body, orelse=[])

    def _visit_Repeat(self, n: R.Repeat) -> P.stmt:
        with self.nse.enter(Slot.NONE):
            body = self._visit_block_as_stmts(n.body)
        return P.While(test=P.Constant(True), body=body, orelse=[])

    def _visit_Break(self, n: R.Break) -> P.stmt:
        return P.Break()

    def _visit_Next(self, n: R.Next) -> P.stmt:
        return P.Continue()

    def _visit_FunctionDef(self, n: R.FunctionDef) -> P.AST:
        """``function(x) body`` → Python ``lambda`` for simple expression
        bodies, ``def`` otherwise. Currently always emits a lambda.
        Top-level ``f <- function(...) ...`` becomes ``f = lambda ...``."""
        py_args = P.arguments(
            posonlyargs=[],
            args=[P.arg(arg=p.name) for p in n.params],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[
                self._visit(p.default) for p in n.params if p.default is not None
            ],
            vararg=None,
            kwarg=None,
        )
        with self.nse.enter(Slot.NONE):
            body = self._visit(n.body)
        return P.Lambda(args=py_args, body=body)


def _contains_col_call(node: P.AST) -> bool:
    """``True`` if the AST contains a ``col(...)`` Call."""
    for sub in P.walk(node):
        if (
            isinstance(sub, P.Call)
            and isinstance(sub.func, P.Name)
            and sub.func.id == "col"
        ):
            return True
    return False


def _is_numeric_literal(node: R.Node) -> bool:
    """Numeric literal — int/float, plain or with leading unary sign.
    Used by ``c(...)`` to decide list vs ``np.array`` emission.
    """
    if isinstance(node, (R.NumLit, R.IntLit)):
        return True
    if isinstance(node, R.UnaryOp) and node.op in ("-", "+"):
        return _is_numeric_literal(node.operand)
    return False


def _to_py_identifier(name: str) -> str:
    """Normalize an R identifier into a valid Python identifier:"""
    import keyword

    py = name.replace(".", "_")
    if keyword.iskeyword(py):
        py = py + "_"
    return py


def _is_pos_int_lit(x: R.Node) -> bool:
    """True for an R positive integer literal — ``3L`` (IntLit) or a whole
    double ``3`` (NumLit). Used to statically shift ``slice`` positions."""
    return (isinstance(x, R.IntLit) and x.value > 0) or (
        isinstance(x, R.NumLit) and x.value == int(x.value) and x.value > 0
    )


def _name(name: str, *, ctx: P.expr_context | None = None) -> P.Name:
    return P.Name(id=name, ctx=ctx or P.Load())


def _attr(value: P.AST, attr: str) -> P.Attribute:
    return P.Attribute(value=value, attr=attr, ctx=P.Load())


def _call(
    func: P.AST, args: list[P.AST] | None = None, kwargs: list[P.keyword] | None = None
) -> P.Call:
    return P.Call(func=func, args=args or [], keywords=kwargs or [])


def _dotted_name(qualified: str) -> P.AST:
    """``"selectors.starts_with"`` → ``ast.Attribute(ast.Name("selectors"), "starts_with")``."""
    parts = qualified.split(".")
    node: P.AST = _name(parts[0])
    for part in parts[1:]:
        node = _attr(node, part)
    return node


_LIBRARY_CALL_NAMES: frozenset[str] = frozenset({"library", "require"})

_NOOP_CALL_NAMES: frozenset[str] = frozenset(
    {
        "suppressMessages",
        "suppressWarnings",
        "suppressPackageStartupMessages",
    }
)


def _is_library_call(node) -> bool:
    """``True`` if ``node`` is ``library(pkg)`` / ``require(pkg)``."""
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
    """Build a Python AST node for ``<name> = data("<name>", package="<pkg>")``."""
    return P.Assign(
        targets=[P.Name(id=name, ctx=P.Store())],
        value=P.Call(
            func=P.Name(id="data", ctx=P.Load()),
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


def _contains_call_to(node: R.Node, fn_name: str) -> bool:
    """Recursively walk ``node``'s subtree looking for any
    ``Call(Identifier(fn_name), ...)``. Used to detect ``with(...)`` /
    ``class(...)`` etc. anywhere inside a top-level R statement."""
    return _first_matching_call(node, lambda n: n == fn_name) == fn_name


def _first_python_keyword_call(node: R.Node) -> str | None:
    """Return the first hard-Python-keyword function name called anywhere
    inside ``node``'s subtree, or ``None`` if there is none.
    """
    import keyword

    return _first_matching_call(
        node,
        lambda name: keyword.iskeyword(name) and name != "with",
    )


def _first_matching_call(node: R.Node, predicate) -> str | None:
    """Walk ``node``'s subtree; return the name of the first ``R.Call``
    whose function-identifier name matches ``predicate``, else ``None``.
    """
    from dataclasses import fields, is_dataclass

    if (
        isinstance(node, R.Call)
        and isinstance(node.func, R.Identifier)
        and predicate(node.func.name)
    ):
        return node.func.name
    if not is_dataclass(node):
        return None
    for f in fields(node):
        v = getattr(node, f.name, None)
        if isinstance(v, R.Node):
            hit = _first_matching_call(v, predicate)
            if hit is not None:
                return hit
        elif isinstance(v, (tuple, list)):
            for item in v:
                if isinstance(item, R.Node):
                    hit = _first_matching_call(item, predicate)
                    if hit is not None:
                        return hit
    return None


def _is_ggplot_chain_call(node) -> bool:
    """``True`` iff ``node`` is a Call whose head identifier marks it as a
    ggplot chain extension (geom_*, scale_*, labs, theme, …).
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
    """Best-effort extraction of column names from across()'s first arg."""
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
                    "— only bare names and strings are supported",
                    a,
                )
        return names
    raise RTranslateError(
        f"across() col form not supported: {type(cols_arg).__name__} "
        "— pass a bare name, a string, or c(a, b, ...).",
        cols_arg,
    )


def _substitute_identifier(
    node: R.Node, param_name: str, replacement: R.Identifier
) -> R.Node:
    """Recursively replace ``Identifier(name=param_name)`` with ``replacement``."""
    from dataclasses import fields, is_dataclass
    from dataclasses import replace as _dc_replace

    if isinstance(node, R.Identifier) and node.name == param_name:
        return replacement
    if not is_dataclass(node):
        return node
    new_kwargs = {}
    for f in fields(node):
        v = getattr(node, f.name)
        if isinstance(v, tuple) and v and is_dataclass(v[0]):
            new_kwargs[f.name] = tuple(
                _substitute_identifier(x, param_name, replacement) for x in v
            )
        elif is_dataclass(v):
            new_kwargs[f.name] = _substitute_identifier(v, param_name, replacement)
        else:
            new_kwargs[f.name] = v
    return _dc_replace(node, **new_kwargs)


def _unparse_for_plotmath(node: R.Node) -> str:
    """Render an R AST node back to source text suitable for plotmath."""
    if isinstance(node, R.Identifier):
        return node.name
    if isinstance(node, R.NumLit):
        if node.value.is_integer():
            return str(int(node.value))
        return str(node.value)
    if isinstance(node, R.IntLit):
        return str(node.value)
    if isinstance(node, R.StrLit):
        return repr(node.value)
    if isinstance(node, R.UnaryOp):
        return f"{node.op}{_unparse_for_plotmath(node.operand)}"
    if isinstance(node, R.BinOp):
        return f"{_unparse_for_plotmath(node.left)} {node.op} {_unparse_for_plotmath(node.right)}"
    if isinstance(node, R.Subscript):
        base = _unparse_for_plotmath(node.target)
        idx = ", ".join(
            _unparse_for_plotmath(a)
            for a in node.args
            if not isinstance(a, R.MissingArg)
        )
        return f"{base}[{idx}]"
    if isinstance(node, R.Call):
        func_text = _unparse_for_plotmath(node.func)
        args = ", ".join(_unparse_for_plotmath(a) for a in node.args)
        return f"{func_text}({args})"
    if isinstance(node, R.NamedArg):
        return f"{node.name} = {_unparse_for_plotmath(node.value)}"
    return f"<{type(node).__name__}>"


def _unparse_for_formula(node: R.Node) -> str:
    """Render an R AST node back to source text for embedding in a formula
    string. Used only by Tilde — handles the small subset that appears in
    typical formulas (identifiers, calls, arithmetic, ``:``, ``*``)."""
    if isinstance(node, R.Identifier):
        return node.name
    if isinstance(node, R.NumLit):
        if node.value.is_integer():
            return str(int(node.value))
        return str(node.value)
    if isinstance(node, R.IntLit):
        return str(node.value)
    if isinstance(node, R.UnaryOp):
        return f"{node.op}{_unparse_for_formula(node.operand)}"
    if isinstance(node, R.BinOp):
        inner = f"{_unparse_for_formula(node.left)} {node.op} {_unparse_for_formula(node.right)}"
        if node.op in ("|", "||"):
            return f"({inner})"
        return inner
    if isinstance(node, R.Call):
        func_text = _unparse_for_formula(node.func)
        args = ", ".join(_unparse_for_formula(a) for a in node.args)
        return f"{func_text}({args})"
    if isinstance(node, R.NamedArg):
        return f"{node.name} = {_unparse_for_formula(node.value)}"
    if isinstance(node, R.StrLit):
        return repr(node.value)
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
