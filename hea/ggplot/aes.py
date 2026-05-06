"""``aes()`` — aesthetic mapping.

An ``Aes`` is a dict from canonical aesthetic name (``"x"``, ``"y"``,
``"colour"``, …) to a *mapping value*. The mapping value can be:

* a column name (bare string that exists in the layer's data frame);
* an expression string parseable by :func:`hea.formula.parse` (e.g.
  ``"log(price)"``); the expression is evaluated against the data
  frame plus the caller's frame at ggplot construction;
* a Python callable ``data -> Series-like`` for the rare cases where
  the expression form is awkward (e.g. column names that contain
  parentheses).

Disambiguation between "bare column name" and "expression" is delegated
to the formula parser: it produces a typed AST. A single :class:`Name`
node whose identifier matches a column means column lookup; any other
AST shape is evaluated as an expression. See plan §13 Q3.
"""

from __future__ import annotations


# Canonical names use British spellings. American aliases canonicalise on
# input so internal code can assume one spelling. Matches ggplot2.
_AES_ALIASES = {
    "color": "colour",
    "gray": "grey",
    "outlier_color": "outlier_colour",
}

# Positional aes args bind to these names in order: aes("wt", "mpg") ⇒ aes(x="wt", y="mpg").
_POSITIONAL_AES = ("x", "y")


def _canon(name: str) -> str:
    return _AES_ALIASES.get(name, name)


class Aes(dict):
    """Aesthetic mapping. ``dict`` subclass so ``aes(...) + aes(...)`` merges
    by right-side override (matching ggplot2's ``%+%`` semantics)."""

    def __add__(self, other: "Aes") -> "Aes":
        return Aes({**self, **other})

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"aes({body})"


class AfterStat:
    """Marker: resolve this aes value *after* the stat has run.

    ggplot2's ``after_stat()`` lets aes mappings reference computed stat
    columns (``count``, ``density``, ``prop``, …). In R the bare-name form
    works via NSE; in Python pass a string (``after_stat("count")``) or a
    callable (``after_stat(lambda d: d["count"] / d["count"].sum())``).
    """

    __slots__ = ("expr",)

    def __init__(self, expr):
        self.expr = expr

    def __repr__(self) -> str:
        return f"after_stat({self.expr!r})"


def after_stat(expr) -> "AfterStat":
    return AfterStat(expr)


class AfterScale:
    """Marker: resolve this aes value *after* the non-positional scale has
    mapped (e.g. when you want to tweak a scale-output colour value)."""

    __slots__ = ("expr",)

    def __init__(self, expr):
        self.expr = expr

    def __repr__(self) -> str:
        return f"after_scale({self.expr!r})"


def after_scale(expr) -> "AfterScale":
    return AfterScale(expr)


def aes(*args, **kwargs) -> Aes:
    """Build an :class:`Aes` mapping.

    Positional args fill ``x`` then ``y``: ``aes("wt", "mpg")`` is
    equivalent to ``aes(x="wt", y="mpg")``. Keyword args set any aes by
    name; American spellings (``color``, ``gray``) canonicalise to British
    (``colour``, ``grey``).
    """
    out = Aes()
    for i, v in enumerate(args):
        if i >= len(_POSITIONAL_AES):
            raise TypeError(
                f"aes() takes at most {len(_POSITIONAL_AES)} positional args "
                f"({_POSITIONAL_AES}); got {len(args)}"
            )
        out[_POSITIONAL_AES[i]] = v
    for k, v in kwargs.items():
        out[_canon(k)] = v
    return out
