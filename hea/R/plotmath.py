"""R's plotmath subset (``quote`` → matplotlib mathtext) plus ``cat``.

Living here because ``cat`` is R's stdout printer and ``quote`` produces
the math-rendered label strings that ``hea.plot`` / ``hea.ggplot`` consume,
so the two are conceptually a "base R text/output" pair.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def quote(r_source: str) -> str:
    """R: ``quote(expr)`` — render an R plotmath expression as a
    matplotlib mathtext string (``r"$...$"`` form).

    The translator passes the inner expression through as R-source
    text; this function re-parses with hea's R parser and walks the AST
    to emit matplotlib's TeX-flavored syntax.

    Supported plotmath subset (covers r4ds chapter 16 usage):

    * Subscript: ``x[i]`` → ``x_i``; ``x[i, j]`` → ``x_{i, j}``
    * Superscript: ``x^2`` → ``x^2``
    * Sums / products: ``sum(expr, i==1, n)`` → ``\\sum_{i=1}^{n} expr``
    * Greek letters: ``pi`` → ``\\pi``, ``mu`` / ``sigma`` / etc.
    * Misc functions: ``sqrt(x)``, ``frac(a, b)``, ``bar(x)``, ``hat(x)``,
      ``tilde(x)``, ``dot(x)``.
    """
    if not r_source:
        return ""
    # Import lazily — translate is a heavier surface than this file should pull in.
    from ..translate.r_parser import parse as _parse

    try:
        ast = _parse(r_source)
    except Exception:
        # Couldn't parse — return the raw text inside math delimiters so
        # the user sees something rather than a crash.
        return r"$" + r_source + r"$"
    # ``parse`` returns a Program of statements; render its single expression.
    expr = ast.statements[0] if ast.statements else None
    body = _plotmath_render(expr) if expr is not None else ""
    return r"$" + body + r"$"


_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
    "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
    "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta",
    "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron",
    "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
}


def _plotmath_render(node) -> str:
    """Walk an R AST and emit matplotlib mathtext (no surrounding ``$``)."""
    from ..translate import r_ast as _R

    if isinstance(node, _R.Identifier):
        if node.name in _GREEK:
            return "\\" + node.name
        if node.name == "infinity":
            return r"\infty"
        return node.name
    if isinstance(node, (_R.NumLit, _R.IntLit)):
        v = node.value
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
    if isinstance(node, _R.StrLit):
        return node.value
    if isinstance(node, _R.UnaryOp):
        op = node.op
        return f"{op}{_plotmath_render(node.operand)}"
    if isinstance(node, _R.BinOp):
        l = _plotmath_render(node.left)
        r = _plotmath_render(node.right)
        op = node.op
        if op == "^":
            return f"{l}^{{{r}}}"
        if op == "==":
            return f"{l} = {r}"
        return f"{l} {op} {r}"
    if isinstance(node, _R.Subscript):
        base = _plotmath_render(node.target)
        idx = ", ".join(
            _plotmath_render(a) for a in node.args
            if not isinstance(a, _R.MissingArg)
        )
        return f"{base}_{{{idx}}}"
    if isinstance(node, _R.Call):
        return _plotmath_render_call(node)
    if isinstance(node, _R.NamedArg):
        # Inside an arg list — render value only; the name carries no
        # plotmath meaning for the functions we support.
        return _plotmath_render(node.value)
    return ""


def _plotmath_render_call(node) -> str:
    """Render a function-call AST as plotmath. Covers R's plotmath family
    (``sum``, ``prod``, ``integral``, ``frac``, ``sqrt``, ``hat``, …)."""
    from ..translate import r_ast as _R

    fname = node.func.name if isinstance(node.func, _R.Identifier) else None
    args = [_plotmath_render(a) for a in node.args]

    if fname in ("sum", "prod", "integral"):
        big = {"sum": r"\sum", "prod": r"\prod", "integral": r"\int"}[fname]
        # ``sum(body, lower, upper)`` — R plotmath order.
        body = args[0] if args else ""
        lower = args[1] if len(args) > 1 else ""
        upper = args[2] if len(args) > 2 else ""
        if lower and upper:
            return f"{big}_{{{lower}}}^{{{upper}}} {body}"
        if lower:
            return f"{big}_{{{lower}}} {body}"
        return f"{big} {body}"
    if fname == "sqrt":
        return r"\sqrt{" + (args[0] if args else "") + "}"
    if fname == "frac":
        num = args[0] if args else ""
        denom = args[1] if len(args) > 1 else ""
        return r"\frac{" + num + "}{" + denom + "}"
    if fname in ("hat", "bar", "tilde", "dot"):
        return "\\" + fname + "{" + (args[0] if args else "") + "}"
    if fname == "paste":
        return " ".join(args)
    # Fallback: render as ``fname(args)`` — the user sees what was emitted.
    if fname is None:
        return ""
    return fname + "(" + ", ".join(args) + ")"


def cat(*args, sep=" ", end="", file=None, fill=False, labels=None, append=False):
    """R: ``cat(...)`` — flatten args into a single string, ``sep``-joined,
    and write to stdout (or ``file=`` if given). No trailing newline by
    default — matches R; pass ``sep="\\n"`` (idiomatic R) or ``end="\\n"``
    if you want one.

    Vector args are flattened (R's ``cat`` is recursive); polars Series
    and numpy arrays are iterated. NA / None become an empty string.
    """
    def _flatten(xs):
        for x in xs:
            if x is None:
                yield ""
            elif isinstance(x, str):
                yield x
            elif isinstance(x, (pl.Series, np.ndarray, list, tuple)):
                yield from _flatten(x)
            else:
                yield str(x)
    text = sep.join(_flatten(args))
    if end:
        text = text + end
    if file is None:
        import sys
        sys.stdout.write(text)
    else:
        # ``file=`` accepts a path (R semantics) or an open file-like object.
        if isinstance(file, str):
            mode = "a" if append else "w"
            with open(file, mode, encoding="utf-8") as fh:
                fh.write(text)
        else:
            file.write(text)
