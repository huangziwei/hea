"""R's elementwise math (vectorized scalar functions) plus the base
character-vector constants ``LETTERS`` / ``letters`` / ``pi``.

Every function dispatches on input type via :func:`_elementwise`: polars
``Expr`` / ``Series`` get the matching polars method, scalars / lists /
ndarrays go through numpy — preserving R's "math functions vectorize over
containers" semantics.

``abs`` and ``round`` are defined here as module attributes but **not**
exported through ``from hea.R import *`` (they collide with Python
builtins; the translator routes R's ``abs()`` / ``round()`` to the
builtin via its ``FUNCTION_TABLE`` for surface compatibility). Use the
fully-qualified ``hea.R.abs`` / ``hea.R.round`` when you need them.
"""
from __future__ import annotations

import numpy as np
import polars as pl


pi = float(np.pi)

# R's character-vector constants of the Latin alphabet, always
# in scope in base R (``LETTERS`` uppercase, ``letters`` lowercase).
LETTERS: list[str] = [chr(ord("A") + i) for i in range(26)]
letters: list[str] = [chr(ord("a") + i) for i in range(26)]


def _elementwise(x, fn_expr, fn_np):
    """Dispatch math by input type.

    polars Expr/Series → call its method; scalars/lists/arrays → numpy.
    Keeps R's "math functions vectorize over containers" semantics.
    """
    if isinstance(x, (pl.Expr, pl.Series)):
        return fn_expr(x)
    return fn_np(x)


def sqrt(x):
    """R: elementwise square root."""
    return _elementwise(x, lambda v: v.sqrt(), np.sqrt)


def exp(x):
    """R: elementwise e^x."""
    return _elementwise(x, lambda v: v.exp(), np.exp)


def log(x, base=None):
    """R: elementwise log. Default is natural log; ``base`` for arbitrary base."""
    if base is None:
        return _elementwise(x, lambda v: v.log(), np.log)
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.log(base)
    return np.log(x) / np.log(base)


def log2(x):
    """R: elementwise base-2 log."""
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.log(2.0)
    return np.log2(x)


def log10(x):
    """R: elementwise base-10 log."""
    return _elementwise(x, lambda v: v.log10(), np.log10)


def log1p(x):
    """R: ``log(1 + x)``, accurate for small ``x``."""
    return _elementwise(x, lambda v: (v + 1).log(), np.log1p)


def expm1(x):
    """R: ``exp(x) - 1``, accurate for small ``x``."""
    return _elementwise(x, lambda v: v.exp() - 1, np.expm1)


def abs(x):
    """R: elementwise absolute value. Shadows the builtin; intentional."""
    return _elementwise(x, lambda v: v.abs(), np.abs)


def sign(x):
    """R: elementwise sign (-1, 0, 1)."""
    return _elementwise(x, lambda v: v.sign(), np.sign)


def sin(x):
    return _elementwise(x, lambda v: v.sin(), np.sin)


def cos(x):
    return _elementwise(x, lambda v: v.cos(), np.cos)


def tan(x):
    return _elementwise(x, lambda v: v.tan(), np.tan)


def asin(x):
    return _elementwise(x, lambda v: v.arcsin(), np.arcsin)


def acos(x):
    return _elementwise(x, lambda v: v.arccos(), np.arccos)


def atan(x):
    return _elementwise(x, lambda v: v.arctan(), np.arctan)


def atan2(y, x):
    """R: two-argument arctangent."""
    if isinstance(y, (pl.Expr, pl.Series)) or isinstance(x, (pl.Expr, pl.Series)):
        return pl.arctan2(y, x)
    return np.arctan2(y, x)


def floor(x):
    return _elementwise(x, lambda v: v.floor(), np.floor)


def ceiling(x):
    """R: ``ceiling`` (note: R uses ``ceiling`` not ``ceil``)."""
    return _elementwise(x, lambda v: v.ceil(), np.ceil)


def round(x, digits=0):
    """R: round half-to-even, ``digits`` decimal places. Shadows builtin."""
    if isinstance(x, (pl.Expr, pl.Series)):
        return x.round(int(digits))
    return np.round(x, int(digits))


def trunc(x):
    """R: truncate toward zero."""
    if isinstance(x, pl.Expr):
        return pl.when(x >= 0).then(x.floor()).otherwise(x.ceil())
    return np.trunc(x)
