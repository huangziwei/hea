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

# Every kwarg name that the layer factories should treat as an aesthetic
# (route to ``Layer.aes_params``) rather than a geom param. Build-time
# promotion (``_promote_string_aes_params``) then resolves string values
# against the data: column match → MAP, otherwise SET. Includes American
# aliases so ``geom_point(color=...)`` is recognised as an aesthetic.
_ALL_AES_NAMES = frozenset({
    # Positional
    "x", "y", "z",
    "xmin", "xmax", "ymin", "ymax",
    "xend", "yend",
    "xintercept", "yintercept", "slope", "intercept",
    # Style
    "colour", "color", "fill", "alpha", "size", "shape",
    "linetype", "linewidth", "stroke",
    # Text
    "label", "family", "fontface", "hjust", "vjust", "angle", "lineheight",
    # Structural
    "group", "weight",
    # Boxplot/violin extras
    "lower", "middle", "upper", "ymin_final", "ymax_final",
    "outlier_colour", "outlier_color", "outlier_fill",
    "outlier_shape", "outlier_size", "outlier_stroke", "outlier_alpha",
})


def _canon(name: str) -> str:
    return _AES_ALIASES.get(name, name)


def split_layer_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Split a geom factory's ``**kwargs`` into ``(aes_params, geom_params)``.

    Keys named after any aesthetic — see :data:`_ALL_AES_NAMES` —
    become ``aes_params`` (which build-time promotion may further
    route to the mapping when the value is a string matching a data
    column). Everything else stays in ``geom_params``."""
    aes_params: dict = {}
    geom_params: dict = {}
    for k, v in kwargs.items():
        if k in _ALL_AES_NAMES:
            aes_params[k] = v
        else:
            geom_params[k] = v
    return aes_params, geom_params


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
    """Build an :class:`Aes` mapping (variable → aesthetic).

    Positional args fill ``x`` then ``y``: ``aes("wt", "mpg")`` ≡
    ``aes(x="wt", y="mpg")``. Keyword args set any aes by name; American
    spellings (``color``, ``gray``) canonicalise to British (``colour``,
    ``grey``).

    **When you don't need ``aes()``** — direct kwargs are syntactic sugar:

    * Plot-level: ``df.ggplot(x="wt", y="mpg", color="cyl")`` is
      equivalent to ``df.ggplot(aes(x="wt", y="mpg", color="cyl"))``.
    * Layer-level: ``geom_point(color="species", x="bill_length_mm")``
      works too — string values that match a data column become
      mappings, others (``color="red"``, ``color="#FF0000"``) stay as
      constants. The "matches a column" rule is the disambiguation.

    **When you DO need ``aes()``** — three cases that kwargs can't
    cleanly express:

    1. **``after_stat()`` / ``after_scale()`` markers**::

           geom_histogram(aes(y=after_stat("density")))

       The marker wraps the aesthetic value (= reference a stat-output
       column, not a data column).

    2. **Composing or sharing a mapping** across plots / layers::

           common = aes(x="mpg", y="disp", color="species")
           p1 = ggplot(df1) + geom_point(common)
           p2 = ggplot(df2) + geom_point(common)

       A reusable, named mapping object is cleaner than spreading
       a dict.

    3. **Forcing MAP semantics when the column name collides with a
       constant**, e.g. a column literally named ``"red"``::

           geom_point(aes(color="red"))   # MAP — column "red"
           geom_point(color="red")         # SET — the constant

       Inside ``aes()``, every value is unconditionally a column
       reference / expression — no SET ambiguity.

    For everything else, kwargs are usually clearer.
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
