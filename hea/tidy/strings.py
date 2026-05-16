"""String operations: stringr's ``str_*`` family plus readr's
``parse_number`` / ``parse_double`` (string → numeric).

Every function dispatches on input type — polars ``Expr`` / ``Series``
stay lazy and return the same shape; lists / numpy arrays go through
Python regex; scalars return scalars. ``str_wrap`` is the wrap-to-width
helper from stringr (textwrap-based); the rest mirror stringr's regex
and case-folding surface.
"""
from __future__ import annotations

import textwrap

import numpy as np
import polars as pl


# ---- readr parsers --------------------------------------------------

def parse_number(x):
    """readr's ``parse_number()`` — pull the first number out of a string column.

    Strips comma thousand-separators, then extracts the first signed
    integer or decimal via ``(-?\\d+(?:\\.\\d+)?)`` and casts to
    ``Float64`` with ``strict=False`` (unparseable → null). Handles
    currency symbols, trailing units, and mixed text the same way
    readr does (``"$1,234.56"`` → ``1234.56``, ``"30 yo"`` → ``30``,
    ``"five"`` → null). Locale-specific thousand/decimal separators
    aren't supported — US-style only.

    Type-in / type-out: ``pl.Series`` → ``pl.Series``; ``pl.Expr`` →
    ``pl.Expr``; list / tuple / ndarray → ``list`` (with ``None`` for
    unparseable entries).
    """
    array_like = not isinstance(x, (pl.Series, pl.Expr))
    if array_like:
        x = pl.Series(x, dtype=pl.Utf8)
    out = (
        x.cast(pl.Utf8)
        .str.replace_all(",", "")
        .str.extract(r"(-?\d+(?:\.\d+)?)")
        .cast(pl.Float64, strict=False)
    )
    return out.to_list() if array_like else out


def parse_double(x):
    """readr's ``parse_double()`` — strict floating-point parser.

    Unlike :func:`parse_number`, this does *not* strip currency symbols
    or extract numbers from mixed text — the whole string must be a
    valid double, otherwise the value becomes null. ``"1.234"`` →
    ``1.234``; ``"$1.99"``, ``"1,234"``, ``"abc"`` → null.

    Type-in / type-out: ``pl.Series`` → ``pl.Series``; ``pl.Expr`` →
    ``pl.Expr``; list / tuple / ndarray → ``list`` (with ``None`` for
    unparseable entries).
    """
    array_like = not isinstance(x, (pl.Series, pl.Expr))
    if array_like:
        x = pl.Series(x, dtype=pl.Utf8)
    out = x.cast(pl.Utf8).cast(pl.Float64, strict=False)
    return out.to_list() if array_like else out


# ---- stringr --------------------------------------------------------

def str_wrap(string, width=80, indent=0, exdent=0, whitespace_only=True):
    """stringr's ``str_wrap()`` — wrap text to a fixed line width.

    Wraps each input string to lines no longer than ``width`` characters,
    breaking on whitespace by default. Mirrors stringr's defaults
    (``width=80``, ``whitespace_only=TRUE``); ``indent`` / ``exdent`` add
    spaces to the first / subsequent lines.

    Accepts a single string or an iterable of strings; returns the same
    shape. Built on Python's :mod:`textwrap` — no R-style pipe, but
    ``hea.str_wrap("...", width=30)`` does what you want.

    Parameters
    ----------
    string : str | Iterable[str]
        Text to wrap. ``None`` entries pass through unchanged.
    width : int, default 80
        Maximum line length (characters).
    indent : int, default 0
        Spaces prepended to the first line of each string.
    exdent : int, default 0
        Spaces prepended to subsequent lines.
    whitespace_only : bool, default True
        Only break at whitespace; never split a word or hyphenated token.
    """
    def _wrap_one(s):
        if s is None:
            return None
        return textwrap.fill(
            str(s),
            width=int(width),
            initial_indent=" " * int(indent),
            subsequent_indent=" " * int(exdent),
            break_long_words=not whitespace_only,
            break_on_hyphens=not whitespace_only,
        )

    if isinstance(string, str):
        return _wrap_one(string)
    return [_wrap_one(s) for s in string]


def str_c(*args, sep="", collapse=None):
    """stringr: ``str_c(..., sep="", collapse=NULL)`` — concatenate.

    Vectorized over its args (R rules: a length-1 arg recycles; mixed
    polars Expr args produce a polars Expr; otherwise eager). When
    ``collapse`` is given, vector results get reduced to a single string
    joined by it (R's tidyverse semantics).
    """
    # Any Expr arg → return an Expr that polars evaluates per row.
    if any(isinstance(a, pl.Expr) for a in args):
        as_exprs = [a if isinstance(a, pl.Expr) else pl.lit(a) for a in args]
        e = pl.concat_str(as_exprs, separator=sep)
        if collapse is not None:
            # ``collapse=`` reduces to a single scalar — polars can do
            # this with ``.str.concat()`` at evaluation time.
            return e.str.concat(collapse)
        return e
    # Eager path: broadcast each arg to the longest length, then join.
    def _as_list(v):
        if isinstance(v, (pl.Series, np.ndarray)):
            return list(v)
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    cols = [_as_list(a) for a in args]
    if not cols:
        return ""
    n = max(len(c) for c in cols)
    # Recycle (R semantics) — single-element vectors broadcast.
    cols = [(c * n if len(c) == 1 else c) for c in cols]
    out: list = []
    for i in range(n):
        parts = [cols[j][i] for j in range(len(cols))]
        if any(p is None for p in parts):
            out.append(None)
        else:
            out.append(sep.join(str(p) for p in parts))
    if collapse is not None:
        joined = collapse.join("" if v is None else v for v in out)
        return joined
    if len(out) == 1:
        return out[0]
    return out


def str_glue(template, *, sep="", _envir=None):
    """stringr: ``str_glue(template)`` — interpolate ``{name}`` from the
    calling frame's locals. Use ``{{`` / ``}}`` to escape braces.

    Inside ``mutate()`` you typically want a polars Expr that references
    columns; pass the column names in ``{...}`` and the implementation
    builds the equivalent ``pl.concat_str`` chain.
    """
    import inspect
    import re as _re

    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        env = {}
    else:
        env = {**frame.f_back.f_globals, **frame.f_back.f_locals}

    # Walk the template, splitting on ``{...}`` placeholders.
    parts = _re.split(r"(\{\{|\}\}|\{[^}]*\}|[^{}]+)", template)
    parts = [p for p in parts if p]
    pieces: list = []
    needs_expr = False
    for p in parts:
        if p == "{{":
            pieces.append("{")
        elif p == "}}":
            pieces.append("}")
        elif p.startswith("{") and p.endswith("}"):
            name = p[1:-1].strip()
            if name in env:
                v = env[name]
                if isinstance(v, pl.Expr):
                    needs_expr = True
                pieces.append(v)
            else:
                # Treat as column reference if name looks like an identifier.
                if _re.match(r"^[A-Za-z_]\w*$", name):
                    pieces.append(pl.col(name))
                    needs_expr = True
                else:
                    raise NameError(f"str_glue(): {name!r} not found")
        else:
            pieces.append(p)

    if needs_expr:
        as_exprs = [
            p if isinstance(p, pl.Expr) else pl.lit(p)
            for p in pieces
        ]
        return pl.concat_str(as_exprs, separator="")
    return "".join(str(p) for p in pieces)


def str_flatten(x, collapse="", last=None):
    """stringr: ``str_flatten(x, collapse, last=NULL)`` — paste-and-collapse.

    On a polars Expr: returns an Expr that ``.str.concat(collapse)``.
    On a Series / list: returns a single string. ``last=`` overrides
    the final separator (R/Oxford-comma idiom).
    """
    if isinstance(x, pl.Expr):
        return x.str.join(collapse)
    if isinstance(x, (pl.Series, np.ndarray)):
        items = [str(v) for v in x if v is not None]
    else:
        items = [str(v) for v in x if v is not None]
    if last is None or len(items) < 2:
        return collapse.join(items)
    return collapse.join(items[:-1]) + last + items[-1]


def str_length(x):
    """stringr: ``str_length(x)`` — UTF-8 character count.

    Polars Expr/Series → ``.str.len_chars()``; scalar / list → ints.
    """
    if isinstance(x, pl.Expr):
        return x.str.len_chars()
    if isinstance(x, pl.Series):
        return x.str.len_chars()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [None if v is None else len(v) for v in x]
    return len(x)


def str_sub(x, start=0, end=None):
    """stringr: ``str_sub(x, start, end)`` — 0-based half-open substring.

    Matches Python slicing: ``str_sub(s, 0, 5)`` is ``s[0:5]``. Negative
    positions count from the end (Python convention; ``end=None`` reads
    to the end of the string). R / stringr's ``str_sub()`` is 1-based
    inclusive; hea follows Python.
    """
    # Inputs already use Python's 0-based half-open convention — pass
    # straight through to polars / native slicing.
    def _norm(s, e, length):
        if e is None:
            e = length
        if s < 0:
            s = max(0, length + s)
        if e < 0:
            e = max(0, length + e)
        s = max(0, min(s, length))
        e = max(s, min(e, length))
        return s, e - s

    if isinstance(x, pl.Expr):
        # polars ``.str.slice(offset, length)`` accepts negative offsets.
        if end is None:
            return x.str.slice(start)
        if end >= 0 and start >= 0:
            return x.str.slice(start, max(0, end - start))
        # Mixed signs: defer to a length-aware expression.
        len_expr = x.str.len_chars()
        norm_start = pl.when(pl.lit(start) < 0).then(
            (len_expr + pl.lit(start)).clip(lower_bound=0)
        ).otherwise(pl.lit(start))
        norm_end = pl.when(pl.lit(end) < 0).then(
            (len_expr + pl.lit(end)).clip(lower_bound=0)
        ).otherwise(pl.lit(end))
        return x.str.slice(norm_start, (norm_end - norm_start).clip(lower_bound=0))
    if isinstance(x, (pl.Series, list, tuple, np.ndarray)):
        out = []
        for v in x:
            if v is None:
                out.append(None)
                continue
            s, n = _norm(start, end, len(v))
            out.append(v[s:s + n])
        if isinstance(x, pl.Series):
            return pl.Series(x.name, out)
        return out
    # Scalar
    if x is None:
        return None
    s, n = _norm(start, end, len(x))
    return x[s:s + n]


def str_to_upper(x, locale=None):
    """stringr: ``str_to_upper(x, locale=None)`` — uppercase. ``locale=``
    falls back to ASCII upper (polars doesn't carry per-locale tables)."""
    if isinstance(x, pl.Expr):
        return x.str.to_uppercase()
    if isinstance(x, pl.Series):
        return x.str.to_uppercase()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [None if v is None else v.upper() for v in x]
    return None if x is None else x.upper()


def str_to_lower(x, locale=None):
    """stringr: ``str_to_lower(x, locale=None)``."""
    if isinstance(x, pl.Expr):
        return x.str.to_lowercase()
    if isinstance(x, pl.Series):
        return x.str.to_lowercase()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [None if v is None else v.lower() for v in x]
    return None if x is None else x.lower()


def str_to_title(x, locale=None):
    """stringr: ``str_to_title(x, locale=None)`` — title case (each word
    capitalized). Uses Python's ``str.title``."""
    if isinstance(x, pl.Expr):
        return x.str.to_titlecase()
    if isinstance(x, pl.Series):
        return x.str.to_titlecase()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [None if v is None else v.title() for v in x]
    return None if x is None else x.title()


def str_sort(x, decreasing=False, na_last=True, locale=None, numeric=False):
    """stringr: ``str_sort(x, ...)`` — locale-naive sort (locale=
    accepted but ignored)."""
    if isinstance(x, pl.Series):
        return x.sort(descending=decreasing, nulls_last=na_last)
    items = list(x)
    nulls = [v for v in items if v is None]
    rest = sorted((v for v in items if v is not None), reverse=decreasing)
    if na_last:
        return rest + nulls
    return nulls + rest


def str_equal(x, y, locale=None, ignore_case=False):
    """stringr: ``str_equal(x, y)`` — element-wise equality with optional
    case-folding. NFC normalization is NOT applied (Python doesn't have
    it built-in); pass pre-normalized inputs if you need it."""
    if ignore_case:
        if isinstance(x, str) and isinstance(y, str):
            return x.casefold() == y.casefold()
    return x == y


def str_detect(string, pattern, negate=False):
    """stringr: ``str_detect(string, pattern, negate=False)`` — regex
    membership check. Polars Expr in, Expr out."""
    if isinstance(string, pl.Expr):
        out = string.str.contains(pattern)
        return ~out if negate else out
    if isinstance(string, pl.Series):
        out = string.str.contains(pattern)
        return ~out if negate else out
    import re as _re

    rx = _re.compile(pattern)
    res = [None if v is None else bool(rx.search(v)) for v in string]
    if negate:
        res = [None if v is None else (not v) for v in res]
    return res


def str_count(string, pattern=r"\s+"):
    """stringr: ``str_count(string, pattern)`` — count regex matches per
    element."""
    if isinstance(string, pl.Expr):
        return string.str.count_matches(pattern)
    if isinstance(string, pl.Series):
        return string.str.count_matches(pattern)
    import re as _re

    rx = _re.compile(pattern)
    return [None if v is None else len(rx.findall(v)) for v in string]


def str_view(string, pattern=None, *, match=None, html=False) -> None:
    """stringr: ``str_view(string, pattern)`` — print each element with
    matches of ``pattern`` highlighted (ANSI brackets in this port).

    ``pattern=None`` just prints each value indexed (matches R's
    "view-the-string" behavior with no pattern). Pure side-effect — no
    return value.
    """
    import re as _re

    def _iter(s):
        if isinstance(s, (pl.Series, np.ndarray)):
            return list(s)
        if isinstance(s, (list, tuple)):
            return list(s)
        return [s]

    values = _iter(string)
    width = len(str(len(values)))
    rx = _re.compile(pattern) if pattern else None
    for i, v in enumerate(values, 1):
        if v is None:
            print(f"[{i:{width}d}] <NA>")
            continue
        text = str(v)
        if rx is not None:
            text = rx.sub(lambda m: "<" + m.group(0) + ">", text)
        print(f"[{i:{width}d}] {text}")


# stringr deprecated ``str_view_all`` in favor of ``str_view`` (which now
# highlights all matches by default). Keep an alias for older scripts.
str_view_all = str_view
