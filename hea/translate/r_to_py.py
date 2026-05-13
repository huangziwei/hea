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
import functools
import importlib
import re
from typing import Optional

from . import _datasets, gaps as _gaps, r_ast as R
from .nse import NSEContext, Slot
from .r_parser import parse as parse_r
from .registry.functions import FUNCTION_TABLE, Func, resolve_kwarg
from .registry.ggplot import is_chain_extension
from .registry.verbs import VERB_TABLE, Verb


# ---------------------------------------------------------------------------
# Standalone runnable preamble — discover what's importable from hea, hea.R,
# and hea.plot once at import time. The translator uses these to emit a
# needs-based import block at the top of every translated module so the
# output can be ``python script.py``'d without a wrapper.
# ---------------------------------------------------------------------------


def _callable_exports(module_name: str) -> frozenset[str]:
    """Public names a module exports — callables and module-level
    constants (e.g. ``hea.R.pi``, ``hea.R.LETTERS``).

    Submodule attributes and dunders are excluded; everything else with
    a public name is importable as a bare identifier in translated R
    scripts.
    """
    import types

    try:
        mod = importlib.import_module(module_name)
    except Exception:
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


# Top-level ``hea`` callables. Computed lazily so additions to
# ``hea.__init__`` that land AFTER ``translate.r_to_py`` is imported
# (e.g. the ``read_csv`` shim and ``cols`` stub) still register. Strip
# the submodule names so ``plot``/``R`` resolve to the function-bearing
# surfaces below, not the submodule object.
@functools.cache
def _hea_exports() -> frozenset[str]:
    return _callable_exports("hea") - {"R", "plot", "ggplot"}


@functools.cache
def _hea_r_exports() -> frozenset[str]:
    return _callable_exports("hea.R")


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
    except Exception:
        return frozenset()
    out: set[str] = set()
    for n in dir(mod):
        if n.startswith("_"):
            continue
        v = getattr(mod, n, None)
        if isinstance(v, types.ModuleType):
            out.add(n)
    return frozenset(out)


# Submodules of ``hea`` we want to import on demand — e.g. ``selectors``
# is ``polars.selectors`` re-exported via ``hea.__init__``. Translator
# emits ``selectors.starts_with(...)`` so the preamble must contain
# ``from hea import selectors``. Lazy to dodge import-order issues like
# ``_hea_exports``.
@functools.cache
def _hea_submodules() -> frozenset[str]:
    return _module_exports("hea")

# Python builtins — names we never need to import.
_PY_BUILTINS: frozenset[str] = frozenset(__builtins__.keys() if isinstance(__builtins__, dict) else dir(__builtins__)) | {  # type: ignore[union-attr]
    "True", "False", "None",
}


# ---------------------------------------------------------------------------
# Unported-construct sentinel. R idioms outside the v1 sublanguage
# (replacement-function assigns, ``with(df, expr)``, ...) emit a uniquely
# tagged statement that ``Translator.translate`` rewrites to a Python
# comment block after ``ast.unparse``. The sentinel survives a regular
# AST walk so downstream passes (autoload, import inference) don't see
# the original R text as a phantom name reference.
# ---------------------------------------------------------------------------

_UNPORTED_TAG = "__HEA_UNPORTED__"
_UNPORTED_LINE_RE = re.compile(rf"^\s*['\"]({_UNPORTED_TAG}):([0-9]+)['\"]\s*$")


class RTranslateError(Exception):
    """Raised when a node can't be translated within the documented sublanguage."""

    def __init__(self, message: str, node: R.Node):
        self.node = node
        super().__init__(f"{message} at span {node.span}")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def translate(src: str, *, log_gaps: bool = False, source_label: str = "<inline>") -> str:
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
    return Translator(src=src, log_gaps=log_gaps, source_label=source_label).translate(prog)


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

    def __init__(self, src: str = "", *, log_gaps: bool = False, source_label: str = "<inline>"):
        self.nse = NSEContext()
        # Packages the source declared via ``library(pkg)`` /
        # ``require(pkg)``. Used to disambiguate autoload candidates
        # below — if ``penguins`` is in two packages and one of them
        # was loaded, prefer that one.
        self._loaded_packages: set[str] = set()
        # Explicit ``pkg::name`` references — emit an autoload for
        # ``name`` from ``pkg`` regardless of whether rdatasets carries
        # it (the dataset may live in a bundled CSV or be downloaded).
        # Keyed by the bare name so the autoload preamble dedupes.
        self._namespaced_refs: dict[str, str] = {}
        # Loop variables whose iter we shifted from R's 1-based ``a:b``
        # to Python's 0-based ``range(a-1, b)``. While translating the
        # body of such a loop, ``(var - 1)`` collapses to ``var`` —
        # R-source ``(i - 1)`` is the manual "shift to 0-based for
        # arithmetic" pattern; after we've already shifted the loop
        # counter the subtraction is redundant.
        self._shifted_loop_vars: set[str] = set()
        # Depth of "index context" — inside a subscript's arg list,
        # ``1:N`` shifts to ``range(N)`` (positions are 0-based in hea).
        # Tracked as a counter so nested subscripts compose.
        self._index_context: int = 0
        # Stack of data-frame names introduced by ``with(df, expr)``. R's
        # ``with()`` evaluates ``expr`` in an env where ``df``'s columns
        # are bound locally; while a frame is active, bare identifiers
        # in value position rewrite to ``df["name"]`` so the same
        # resolution happens at runtime.
        self._with_stack: list[str] = []
        # Original R source — used to slice back the text of an
        # unportable statement when emitting its comment block.
        self._src = src
        # ``[(kind, subject, r_text, notes)]`` — one row per unported
        # statement. Resolved post-unparse into Python comments.
        self._unported: list[tuple[str, str, str, str]] = []
        self._log_gaps = log_gaps
        self._source_label = source_label

    # -- public ------------------------------------------------------------

    def translate(self, prog: R.Program) -> str:
        module = self._visit_program(prog)
        P.fix_missing_locations(module)
        src = P.unparse(module)
        src = self._rewrite_unported(src)
        return src

    # -- unported sentinel ------------------------------------------------

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
            out_lines.append(f"{indent}# UNPORTED [{kind}: {subject}] — translator declined; original R was:")
            for r_line in r_text.splitlines() or [""]:
                out_lines.append(f"{indent}#   {r_line}")
        return "\n".join(out_lines)

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
            # Top-level checks for constructs we can't translate cleanly.
            # Each emits a sentinel that ``_rewrite_unported`` later turns
            # into a comment block — and logs a gap row.
            unported = self._maybe_unported(stmt)
            if unported is not None:
                body.append(unported)
                continue
            body.append(self._as_stmt(self._visit(stmt)))

        # Autoload preamble: bare names referenced but not defined that
        # match a known rdatasets entry get a ``hea.data(...)`` load
        # prepended to the module body.
        autoload = self._build_autoload_preamble(body)
        # Import preamble: scan body+autoload for bare Load Name refs and
        # emit minimal ``from hea[.R|.plot] import ...`` so the translated
        # source is runnable standalone (``python script.py``).
        imports = self._build_import_preamble(autoload + body)
        return P.Module(body=imports + autoload + body, type_ignores=[])

    # -- unportable-construct detection ------------------------------------

    def _maybe_unported(self, stmt: R.Node) -> Optional[P.stmt]:
        """Return a sentinel statement for top-level constructs the
        translator cannot emit as valid Python; otherwise ``None``."""
        # ``f(x) <- v`` replacement-function assignment. R has dozens of
        # these (``levels``, ``names``, ``colnames``, ``contrasts``,
        # ``diag``, ``dim``, ``attr``, ...) — far cleaner to detect by
        # structure (any Assign whose target is a Call) than to maintain
        # a whitelist.
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
        # Any statement containing a call to a Python-keyword name
        # (``class(x)``, ``try(expr)``, ``except(...)`` etc.). Emitting
        # ``class(x)`` would be a Python syntax error.
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

        Names not present in any of the three are left alone — they'll
        surface as ``NameError`` at runtime, which is the right signal
        that a gap remains rather than silently masking with a wildcard.
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
                elif isinstance(node, P.Attribute) and isinstance(node.value, P.Name):
                    # ``hea.data(...)`` / ``selectors.starts_with(...)``
                    # — the root name (``hea`` / ``selectors``) is what
                    # must be importable.
                    if isinstance(node.value.ctx, P.Load):
                        referenced.add(node.value.id)

        candidates = referenced - defined - _PY_BUILTINS

        # hea.R first: R-script translations want R semantics (e.g. ``mean``
        # is a scalar reducer with na_rm=True, not the polars expression
        # helper). Fall back to ``hea`` for names hea.R doesn't carry.
        # hea.ggplot picks up geom_/scale_/coord_/facet_/theme_/position_*
        # helpers, which the translator emits as bare names.
        r_names = sorted(n for n in candidates if n in _hea_r_exports())
        used = set(r_names)
        hea_names = sorted(n for n in (candidates - used) if n in _hea_exports())
        used |= set(hea_names)
        plot_names = sorted(n for n in (candidates - used) if n in _hea_plot_exports())
        used |= set(plot_names)
        ggplot_names = sorted(n for n in (candidates - used) if n in _hea_ggplot_exports())
        used |= set(ggplot_names)
        # Submodules used as Attribute roots: ``selectors.starts_with``,
        # ``pl.col``, etc. ``from hea import selectors`` resolves the root.
        submod_names = sorted(n for n in (candidates - used) if n in _hea_submodules())

        out: list[P.stmt] = []
        if "hea" in referenced:
            out.append(P.Import(names=[P.alias(name="hea", asname=None)]))
        if "np" in referenced:
            # Translator emits ``np.array([...])`` for all-numeric ``c(...)``.
            out.append(P.Import(names=[P.alias(name="numpy", asname="np")]))
        if hea_names or submod_names:
            out.append(P.ImportFrom(
                module="hea",
                names=[P.alias(name=n, asname=None) for n in (hea_names + submod_names)],
                level=0,
            ))
        if r_names:
            out.append(P.ImportFrom(
                module="hea.R",
                names=[P.alias(name=n, asname=None) for n in r_names],
                level=0,
            ))
        if plot_names:
            out.append(P.ImportFrom(
                module="hea.plot",
                names=[P.alias(name=n, asname=None) for n in plot_names],
                level=0,
            ))
        if ggplot_names:
            out.append(P.ImportFrom(
                module="hea.ggplot",
                names=[P.alias(name=n, asname=None) for n in ggplot_names],
                level=0,
            ))
        return out

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
        # Function-table emitted names (``sd → std``, ``cumsum``, ``ntile``,
        # …) are translator output, not dataset references. Without this,
        # ``sd(x)`` translating to ``std(x)`` would trigger a phantom
        # ``std = hea.data("std", package="KMsurv")`` autoload.
        emitted_helpers = frozenset(
            f.hea_name.split(".", 1)[0] for f in FUNCTION_TABLE.values()
        )
        candidates = sorted(
            referenced - defined - _datasets.DATASET_REF_EXCLUSIONS - emitted_helpers
        )
        out: list[P.stmt] = []
        emitted: set[str] = set()
        # First: explicit ``pkg::name`` references take precedence over
        # registry-based resolution. ``hea.data`` will fall back to a
        # bundled CSV or GitHub download if rdatasets doesn't carry it.
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
        # Inside ``with(df, expr)``: bare identifier resolves to a
        # column lookup on ``df``. Matches R's NSE binding — ``with()``
        # checks the data frame's columns before falling back to the
        # parent environment. Known R functions (``mean``, ``sum``,
        # tidyverse verbs, …) bypass the rewrite so e.g.
        # ``with(df, tapply(x, g, mean))`` keeps ``mean`` as a function
        # rather than mis-resolving to ``df["mean"]``.
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
        # NONE — emit as a Python name. Dot identifiers (``data.frame``,
        # ``na.omit``) become underscores; identifiers that collide with
        # Python keywords (``lambda``, ``class``, ``True``, ...) get a
        # trailing ``_`` so the result is always a valid Python name.
        return _name(_to_py_identifier(n.name))

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
            # In COLUMN_NAME (tidy-select) slot, ``!cols`` means "exclude
            # these cols". Strategy depends on operand shape:
            # - ``!(a:b)`` — operand is ``cols_between(...)`` which
            #   supports ``__invert__`` natively → emit ``~operand``.
            # - ``!single_col`` / ``!c(...)`` — wrap in
            #   ``~selectors.by_name(operand)`` so polars' selector tree
            #   handles the inversion at expansion time.
            if self.nse.current is Slot.COLUMN_NAME:
                if (
                    isinstance(operand, P.Call)
                    and isinstance(operand.func, P.Name)
                    and operand.func.id == "cols_between"
                ):
                    return P.UnaryOp(P.Invert(), operand)
                return P.UnaryOp(P.Invert(), _call(
                    _attr(_name("selectors"), "by_name"),
                    [operand],
                ))
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

        # ``(var - 1)`` where ``var`` is a 0-based-shifted loop counter
        # → just ``var``. R's ``for(i in 1:N) ... i - 1 ...`` is the
        # manual "convert 1-based loop counter to 0-based for math"
        # pattern; after the translator has already shifted the loop,
        # the subtraction is redundant and (worse) makes i=0 become -1.
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

        # Sequence ``a:b`` → ``seq(a, b)`` by default; the for-loop iter
        # and subscript visitors special-case to ``range(...)`` shifts so
        # those positions match hea's 0-based indexing without breaking
        # value uses elsewhere (e.g. ``plot(m, which=1:2)`` panel IDs).
        if op == ":":
            # In COLUMN_NAME slot (``select(year:day)``, ``relocate(...)``)
            # ``a:b`` is dplyr's column-range selector, not a numeric
            # range. Emit ``cols_between('a', 'b')`` — hea's tidy-select
            # placeholder that supports ``~`` for ``select(!(a:b))``.
            if self.nse.current is Slot.COLUMN_NAME:
                return _call(_name("cols_between"), [left, right])
            # Inside an index context (subscript arg, recursively into
            # function calls there), ``1:N`` shifts to ``range(N)`` so
            # downstream consumers like ``sample(1:N)`` produce 0-based
            # values that match hea's container indexing.
            if self._index_context and isinstance(left, P.Constant) and left.value == 1:
                return _call(_name("range"), [right])
            return _call(_name("seq"), [left, right])

        # Namespace access ``pkg::name`` / ``pkg:::name``. Drop the package
        # qualifier — the function registry handles renaming — but
        # record the LHS as a loaded package AND the (name, pkg) pair as
        # an explicit autoload hint. ``Lahman::Batting`` is often the
        # only hint we get that ``Batting`` is a dataset (the user
        # didn't write ``library(Lahman)``); the rdatasets registry
        # doesn't carry Lahman, so without this hint the autoload pass
        # would silently skip it.
        if op == "::" or op == ":::":
            if isinstance(n.left, R.Identifier):
                pkg_name = n.left.name
                self._loaded_packages.add(pkg_name)
                if isinstance(n.right, R.Identifier):
                    self._namespaced_refs.setdefault(n.right.name, pkg_name)
            return right

        # ``%in%`` → ``.is_in(...)``.
        if op == "%in%":
            return _call(_attr(left, "is_in"), [right])

        # ``%*%`` — R matrix multiplication. Python's ``@`` matmul
        # operator is the right target (works on numpy arrays + hea
        # DataFrames-as-matrices).
        if op == "%*%":
            return P.BinOp(left=left, op=P.MatMult(), right=right)

        # Other ``%infix%`` operators — emit as a function call so the user
        # sees something sensible. The registry can resolve specific names
        # in a later phase. The function-name strips ``%`` and any other
        # operator-only character so we never produce ``*(...)``-style
        # invalid Python identifiers.
        if op.startswith("%") and op.endswith("%"):
            fname = "".join(c for c in op.strip("%") if c.isalnum() or c == "_")
            if not fname:
                # All-symbol infix (``%~%``, ``%^%``) with no usable
                # Python identifier. Emit as ``_infix_<op>(left, right)``
                # to surface the original operator in the gap log.
                fname = "_infix_" + "".join(f"{ord(c):x}" for c in op.strip("%"))
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

            # ``with(df, expr)`` — push the df name onto the stack so
            # bare identifiers in ``expr``'s subtree rewrite to
            # ``df["name"]``; the call itself "returns" the translated
            # expression value (R's ``with()`` is value-producing).
            if name == "with":
                return self._emit_with_call(n.args)

            # 1) Verb dispatch.
            verb = VERB_TABLE.get(name)
            if verb is not None and n.args:
                return self._emit_verb_call(verb, n.args)

            # 2) Function-helper dispatch.
            helper = FUNCTION_TABLE.get(name)
            if helper is not None:
                return self._emit_helper_call(helper, name, n.args)

            # 2.5) Model fits with no ``data`` argument — R resolves
            # ``lm(y ~ x)``'s ``y``/``x`` from the caller's environment.
            # Translate-time we can do the same by building a frame from
            # the formula's terms.
            synthesized = self._maybe_lm_no_data(name, n.args)
            if synthesized is not None:
                return synthesized

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

        # Special-case: tribble — row-form literal, reshape to column-major.
        if helper.hea_name == "__tribble__":
            return self._emit_tribble_call(args)

        # Special-case: where(is.X) — tidyselect predicate.
        if helper.hea_name == "__where__":
            return self._emit_where_call(args)

        # Special-case: join_by — bare ID → string; comparison → Expr.
        if helper.hea_name == "__join_by__":
            return self._emit_join_by_call(args)

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

        # Drop R-side kwargs that have no hea counterpart (e.g. readr's
        # ``col_types=`` / ``id=``) so the emitted .py doesn't carry
        # them. Done pre-translate so the dropped arg's value isn't
        # walked at all (the value may itself reference R-only names).
        if helper.drop_kwargs:
            args = tuple(
                a for a in args
                if not (isinstance(a, R.NamedArg) and a.name in helper.drop_kwargs)
            )

        # Function form (or method-form fallback outside EXPR slot).
        with arg_slot_ctx:
            py_args, py_kwargs = self._translate_args(args)

        # ``selectors.starts_with`` → ast.Attribute chain.
        callee = _dotted_name(helper.hea_name)
        return _call(callee, py_args, py_kwargs)

    def _emit_regular_call(self, name: str, args: tuple[R.Node, ...]) -> P.AST:
        """Unknown function — emit as-is, normalizing the name into a
        valid Python identifier (dot→underscore, keyword→trailing-``_``)."""
        py_args, py_kwargs = self._translate_args(args)
        return _call(_name(_to_py_identifier(name)), py_args, py_kwargs)

    def _emit_with_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``with(df, expr)`` — R's NSE binding.

        ``expr`` is translated with bare identifiers rewritten to
        ``df["name"]`` lookups. Currently requires the ``df`` argument
        to be a bare :class:`R.Identifier` (the common case across
        idiomatic R scripts); anything more elaborate falls through to
        the regular-call emission so the user can see the gap.
        """
        if len(args) < 2 or not isinstance(args[0], R.Identifier):
            return self._emit_regular_call("with", args)
        df_name = _to_py_identifier(args[0].name)
        body = args[1]
        self._with_stack.append(df_name)
        try:
            return self._visit(body)
        finally:
            self._with_stack.pop()

    _LM_LIKE: frozenset[str] = frozenset({"lm", "glm", "gam", "bam", "lme", "lmer", "glmer"})
    _FORMULA_OPS: frozenset[str] = frozenset({"+", "-", "*", "/", ":", "^", "|"})

    def _maybe_lm_no_data(self, name: str, args: tuple[R.Node, ...]) -> Optional[P.AST]:
        """Catch ``lm(y ~ x)`` / ``glm(...)`` / etc. with no ``data`` arg.

        R resolves the formula's variables in the caller's environment;
        hea requires an explicit ``data=`` frame. We build that frame
        at translate time from the formula's *additive terms*: bare
        identifiers become eponymous columns, compound expressions get
        synthesized names (``term_0``…) with the expression itself as
        the column value. The formula text is rewritten to match.
        """
        if name not in self._LM_LIKE:
            return None
        if not args or not isinstance(args[0], R.Tilde):
            return None

        positional = [a for a in args if not isinstance(a, R.NamedArg)]
        has_data_kwarg = any(
            isinstance(a, R.NamedArg) and a.name in ("data", ".data")
            for a in args
        )
        # Second positional arg fills the ``data`` slot in R.
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
            # Compound term — synthesize a fresh name.
            new_name = f"term_{counter[0]}"
            counter[0] += 1
            pairs.append((new_name, node))
            seen.add(new_name)
            return R.Identifier(name=new_name, span=node.span)

        def walk(side: R.Node) -> R.Node:
            # Recurse through top-level formula operators; everything
            # else is a terminal "term."
            if isinstance(side, R.BinOp) and side.op in self._FORMULA_OPS:
                return R.BinOp(
                    op=side.op,
                    left=walk(side.left),
                    right=walk(side.right),
                    span=side.span,
                )
            if isinstance(side, R.UnaryOp) and side.op in ("-", "+"):
                return R.UnaryOp(op=side.op, operand=walk(side.operand), span=side.span)
            # Literals (``y ~ 1`` intercept, ``y ~ 0`` no-intercept) pass through.
            if isinstance(side, (R.NumLit, R.IntLit, R.BoolLit)):
                return side
            return take(side)

        new_lhs = walk(formula.lhs) if formula.lhs is not None else None
        new_rhs = walk(formula.rhs)
        new_tilde = R.Tilde(lhs=new_lhs, rhs=new_rhs, span=formula.span)
        formula_str = _format_formula(new_tilde)

        # hea.DataFrame({col: <visited value>, ...}).
        keys: list[P.AST] = []
        values: list[P.AST] = []
        for col_name, ast_node in pairs:
            keys.append(P.Constant(value=col_name))
            values.append(self._visit(ast_node))
        df_arg = _call(
            _attr(_name("hea"), "DataFrame"),
            [P.Dict(keys=keys, values=values)],
            [],
        )

        # Propagate any non-formula kwargs (weights=, method=, …).
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

    # R's tidyselect predicates → equivalent ``polars.selectors`` calls.
    # All return a Selector (no extra args needed); the translator emits
    # ``selectors.<name>()``.
    _WHERE_PREDICATE_MAP: dict[str, str] = {
        "is.character": "string",
        "is.string":    "string",
        "is.numeric":   "numeric",
        "is.double":    "float",
        "is.integer":   "integer",
        "is.logical":   "boolean",
        "is.boolean":   "boolean",
        "is.factor":    "categorical",
        "is.Date":      "date",
        "is.POSIXct":   "datetime",
        "is.POSIXlt":   "datetime",
    }

    _JOIN_BY_BINOPS: frozenset[str] = frozenset({"==", "<", "<=", ">", ">=", "!="})

    def _emit_join_by_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``join_by(x, dest == faa, closest(t >= u))`` — dplyr's NSE
        join spec. Each arg gets its own treatment:

        - ``Identifier`` (bare name) → emit ``'name'`` (string; same key
          on both sides; hea's ``join_by`` recognises this shape).
        - ``BinOp`` with a comparison op → emit
          ``col('lhs') op col('rhs')`` so polars' ``Expr.__eq__`` /
          ``__lt__`` / etc. fires and produces a join-binary expression.
        - ``Call`` to one of ``closest`` / ``between`` / ``overlaps`` /
          ``within`` (or anything else) → translate normally with
          ``Slot.EXPR`` active so its args (column names) become col()s.
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
        # Fallback — let ``where`` resolve at runtime (will likely error).
        py_args, py_kwargs = self._translate_args(args)
        return _call(_name("where"), py_args, py_kwargs)

    def _emit_tribble_call(self, args: tuple[R.Node, ...]) -> P.AST:
        """``tribble(~a, ~b, 1, "x", 2, "y")`` →
        ``hea.DataFrame({"a": [1, 2], "b": ["x", "y"]})``.

        Leading args of the form ``~name`` (unary tilde over a bare
        identifier) are column headers; the remaining args fill those
        columns in row-major order. A trailing partial row falls
        through to ``hea.DataFrame`` which will raise (matches R's own
        behavior — tribble requires complete rows).
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
            # No header rows — degenerate; fall back to data.frame
            # emission so the user sees a useful error / shape.
            return self._emit_data_frame_call(args)
        keys: list[P.AST] = [P.Constant(value=name) for name in col_names]
        # Reshape row-major data into n parallel column lists.
        columns: list[list[P.AST]] = [[] for _ in range(n)]
        for i, a in enumerate(data_args):
            columns[i % n].append(self._visit(a))
        values: list[P.AST] = [P.List(elts=col, ctx=P.Load()) for col in columns]
        return _call(
            _attr(_name("hea"), "DataFrame"),
            [P.Dict(keys=keys, values=values)],
        )

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

        All-numeric-literal vectors (``c(2, 3, 5)`` / ``c(-1.5, 0, 1.5)``)
        emit as ``np.array([...])`` so R's elementwise arithmetic
        (``primes * 2``, ``primes - 1``) carries over — Python's bare
        ``list`` rejects ``-``  and repeats on ``*``.
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
        # R lets named args use any string literal name (``fct_recode(
        # "Republican, strong" = ...)``) and even repeat the same name
        # (``fct_recode("Other" = "x", "Other" = "y")`` — many-to-one
        # merge). Python accepts neither shape as plain ``name=v`` kwargs,
        # so anything non-identifier OR repeated lands in a trailing
        # ``**{...}`` dict; repeats become value lists.
        name_counts: dict[str, int] = {}
        for arg in args:
            if isinstance(arg, R.NamedArg):
                py_name = resolve_kwarg(arg.name).py_name
                name_counts[py_name] = name_counts.get(py_name, 0) + 1
        # Dict of merged kwargs we'll emit as a single ``**{}`` at the end.
        merged_keys: list[str] = []      # insertion order
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
                # Empty arg in subscript context — represent as None.
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
                    # Many → list. Matches fct_recode's many-to-one merge
                    # shape (and is generally less lossy than picking the
                    # last value as Python would do for plain dup kwargs).
                    dict_values.append(P.List(elts=vals, ctx=P.Load()))
            py_kwargs.append(P.keyword(
                arg=None,
                value=P.Dict(keys=dict_keys, values=dict_values),
            ))
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
        """``df[i]`` / ``df[i, j]`` — Python ``df[i]`` / ``df[i, j]``.

        R's blank-axis form ``df[, j]`` / ``df[i, ]`` maps to a Python
        slice (``df[:, j]`` / ``df[i, :]``) — polars and numpy both accept
        that, whereas ``None`` is rejected.

        R-range subscripts ``vec[a:b]`` translate to Python slices
        ``vec[a-1:b]`` so 0-based positional indexing on the hea side
        selects the same elements R's 1-based ``a:b`` would.
        """
        def _arg(a):
            if isinstance(a, R.MissingArg):
                return P.Slice(lower=None, upper=None, step=None)
            if isinstance(a, R.BinOp) and a.op == ":":
                return self._range_subscript(a)
            # R 1-based literal index → Python 0-based.
            if isinstance(a, R.IntLit):
                return P.Constant(value=a.value - 1)
            if isinstance(a, R.NumLit) and a.value == int(a.value) and a.value > 0:
                return P.Constant(value=int(a.value) - 1)
            # ``c(1, 3, 5)`` of positive int literals → shifted Python list.
            if (
                isinstance(a, R.Call)
                and isinstance(a.func, R.Identifier)
                and a.func.name == "c"
                and a.args
                and all(
                    (isinstance(x, R.IntLit) and x.value > 0)
                    or (isinstance(x, R.NumLit) and x.value == int(x.value) and x.value > 0)
                    for x in a.args
                )
            ):
                elts = [
                    P.Constant(value=int(x.value) - 1)
                    for x in a.args
                ]
                return P.List(elts=elts, ctx=P.Load())
            # ``vec[expr + 1]`` — R's idiom for "shift 0-based-arithmetic
            # result to 1-based index" (e.g. ``letters[dm + 1]`` where
            # ``dm`` was built from ``%% s``). hea is 0-based, so the
            # ``+ 1`` is redundant; drop it.
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
            # Recurse with index context active: ``sample(1:N)`` and any
            # other nested ``1:N`` literal inside the subscript arg should
            # shift to ``range(N)`` so the produced indices are 0-based.
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
        """Emit a Python slice for an R ``a:b`` index expression.

        R ``vec[1:5]`` selects 1-based positions 1..5 = first 5 elements;
        Python ``vec[:5]`` does the same. R ``vec[a:b]`` (positions a..b
        1-based, length b-a+1) becomes Python ``vec[a-1:b]`` (positions
        a-1..b-1 0-based, same length).
        """
        with self.nse.enter(Slot.NONE):
            left = self._visit(bin_op.left)
            right = self._visit(bin_op.right)
        # ``1:N`` → ``:N``
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
            slice_ = _arg(n.args[0]) if len(n.args) == 1 else \
                P.Tuple(elts=[_arg(a) for a in n.args], ctx=P.Load())
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
        iterable, shifted = self._visit_for_iter(n.iterable)
        # When the R loop ``for(i in a:b)`` is shifted to a 0-based
        # ``range(a-1, b)``, body references to ``(i - 1)`` (R's manual
        # "shift to 0-based for arithmetic") are no longer needed — the
        # loop counter is already 0-based. Rewrite them so the math
        # matches the original R script.
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
        """Translate a control-flow body to a list of Python statements.

        R's ``{stmt1; stmt2; ...}`` blocks contain multiple statements —
        the visitor that's emitting an `if` / `for` / `while` / function
        body needs every one, not just the last. Single-statement bodies
        (no braces) collapse to a 1-element list.
        """
        if isinstance(body, R.Block):
            if not body.statements:
                return [P.Pass()]
            return [self._as_stmt(self._visit(s)) for s in body.statements]
        return [self._as_stmt(self._visit(body))]

    def _visit_for_iter(self, iter_node: R.Node) -> tuple[P.AST, bool]:
        """Translate the iter of an R ``for(i in <iter>)``.

        R is 1-based; hea is 0-based throughout. Emitting ``range(a-1, b)``
        for R's ``a:b`` makes the loop counter a 0-based index, which is
        how the body normally uses it. ``range`` matches Python idiom
        and is what numpy / polars indexing expects.

        Returns the iter AST plus a flag indicating whether the loop's
        counter was shifted (caller propagates this into the body so
        ``(i - 1)`` references collapse to ``i``).
        """
        if isinstance(iter_node, R.BinOp) and iter_node.op == ":":
            with self.nse.enter(Slot.NONE):
                left = self._visit(iter_node.left)
                right = self._visit(iter_node.right)
            # ``1:N`` → ``range(N)``; ``a:b`` → ``range(a-1, b)``.
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
    """Normalize an R identifier into a valid Python identifier:

    1. ``.`` → ``_`` (``data.frame`` → ``data_frame``).
    2. Trailing ``_`` if the result is a hard Python keyword (``lambda``
       → ``lambda_``, ``class`` → ``class_``).

    Soft keywords (``match``, ``case``, ``type``, ``_``) are left alone
    — they're valid identifiers outside their reserved-statement
    contexts.
    """
    import keyword

    py = name.replace(".", "_")
    if keyword.iskeyword(py):
        py = py + "_"
    return py


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


def _contains_call_to(node: R.Node, fn_name: str) -> bool:
    """Recursively walk ``node``'s subtree looking for any
    ``Call(Identifier(fn_name), ...)``. Used to detect ``with(...)`` /
    ``class(...)`` etc. anywhere inside a top-level R statement."""
    return _first_matching_call(node, lambda n: n == fn_name) == fn_name


def _first_python_keyword_call(node: R.Node) -> Optional[str]:
    """Return the first hard-Python-keyword function name called anywhere
    inside ``node``'s subtree, or ``None`` if there is none.

    R has plenty of functions whose names collide with Python keywords
    — ``class``, ``try``, ``while``, ``return``, ``except``, ``finally``,
    ``raise``, ``yield``, ``del``, ``assert``, ``global``, ``nonlocal``,
    ``lambda``, ``pass``, ``elif``, ``async``, ``await``. Emitting them
    as Python identifiers is a syntax error.

    Soft keywords (``match``, ``case``, ``type``, ``_``) are valid as
    identifiers outside their reserved-statement contexts, so we do
    **not** unport them. ``with`` is also a Python keyword but the
    translator handles ``with(df, expr)`` natively via NSE rebinding —
    excluded here so it isn't flagged as a gap.
    """
    import keyword
    return _first_matching_call(
        node,
        lambda name: keyword.iskeyword(name) and name != "with",
    )


def _first_matching_call(node: R.Node, predicate) -> Optional[str]:
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
