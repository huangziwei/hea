"""R's ``factor`` / ``ordered`` / ``interaction`` family.

Covers the categorical primitives: ``factor`` (and ``forcats::fct``),
``ordered``, ``levels`` / ``nlevels``, ``is.factor``, and ``interaction``.
All eager forms route through :class:`pl.Enum`; the deferred ``factor("col")``
form is resolved by ``hea.DataFrame.mutate`` / ``select`` via the private
:class:`_LazyFactor` placeholder.
"""
from __future__ import annotations

import polars as pl

from ..formula import set_ordered_cols


def _label_key_to_str(k):
    # polars casts bool to lowercase "true"/"false", but str(True) is "True".
    # Match polars so factor(col != x, labels={False: ..., True: ...}) works.
    if isinstance(k, bool):
        return "true" if k else "false"
    return str(k)


class _LazyFactor:
    """Deferred ``factor()`` placeholder, resolved by ``mutate``/``select``.

    Returned by ``factor("colname")`` or ``factor(pl.col(...))`` so the
    tidyverse-style ``df.mutate(species=hea.factor("species"))`` works.
    Polars expressions can't ``.to_list()`` mid-pipeline, so auto level
    detection has to peek at the source frame — the verb that owns the
    frame calls ``_resolve(self)`` to materialize a real ``pl.Expr``.
    """

    __slots__ = ("col_ref", "levels", "labels", "ordered", "strict")

    def __init__(self, col_ref, levels, labels, ordered, strict):
        self.col_ref = col_ref  # str column name or pl.Expr
        self.levels = levels
        self.labels = labels
        self.ordered = ordered
        self.strict = strict

    def _column_name(self) -> str | None:
        if isinstance(self.col_ref, str):
            return self.col_ref
        try:
            return self.col_ref.meta.output_name()
        except Exception:
            return None

    def _resolve(self, df: pl.DataFrame, fallback_name: str | None = None) -> pl.Expr:
        col_name = self._column_name() or fallback_name
        base = pl.col(self.col_ref) if isinstance(self.col_ref, str) else self.col_ref
        s_utf8 = base.cast(pl.Utf8)

        if self.labels is not None:
            old = [_label_key_to_str(k) for k in self.labels.keys()]
            new = [str(v) for v in self.labels.values()]
            out_expr = s_utf8.replace_strict(old, new, return_dtype=pl.Enum(new))
        else:
            if self.levels is None:
                if col_name is None or col_name not in df.columns:
                    raise ValueError(
                        "factor(): can't auto-detect levels — column "
                        f"{col_name!r} not in the frame. Pass levels= "
                        "explicitly, or use the eager Series form "
                        "factor(df['col'])."
                    )
                src = df[col_name]
                # R's factor() sorts numerically when the input is numeric
                # (then string-casts); only character/factor inputs sort lex.
                if src.dtype.is_numeric():
                    levels_list = [str(v) for v in src.drop_nulls().unique().sort().to_list()]
                else:
                    levels_list = src.cast(pl.Utf8).drop_nulls().unique().sort().to_list()
            else:
                levels_list = [str(v) for v in self.levels]
            out_expr = s_utf8.cast(pl.Enum(levels_list), strict=self.strict)

        if self.ordered and col_name:
            from ..formula import _ORDERED_COLS_CV
            set_ordered_cols(_ORDERED_COLS_CV.get() | frozenset({col_name}))
        return out_expr


def factor(
    series,
    levels=None,
    labels: dict | None = None,
    ordered: bool = False,
    strict: bool = False,
):
    """Polars equivalent of R's ``factor()`` — cast a column to ``pl.Enum``.

    Two call forms:

    * **Eager** — ``factor(df["col"])`` takes a ``pl.Series`` and returns
      a ``pl.Series``. Use with ``df.with_columns(factor(df["col"]))``;
      the returned Series keeps its input name, so ``with_columns``
      replaces the original column.
    * **Deferred** — ``factor("col")`` or ``factor(pl.col("col"))``
      returns a placeholder that ``DataFrame.mutate`` / ``select``
      resolves against the receiver, enabling tidyverse-style
      ``df.mutate(species=hea.factor("species"))``. The placeholder
      isn't a ``pl.Expr`` and won't work inside polars-native verbs
      (``with_columns``, ``filter``, etc.); pass a Series there.

    Parameters
    ----------
    series : pl.Series | list | np.ndarray | str | pl.Expr
        Column to convert. ``pl.Series`` (and bare lists / numpy
        arrays, which get wrapped) trigger the eager path; ``str``
        (column name) or ``pl.Expr`` trigger the deferred path.
        Int64 inputs route through Utf8 (``pl.Enum`` can't accept
        integers directly). Values not in ``levels=`` become null,
        matching R's ``factor()`` (which produces ``NA``).
    levels : list | None, optional
        Level order, no relabel. If None, auto-detected via
        ``unique().sort()`` on the string-cast values — that's Unicode
        collation, which can diverge from R's locale-aware ``factor()``
        default for non-ASCII or punctuation-heavy levels. For poly
        contrasts on ordered factors, pass levels explicitly to control
        the order. Mutually exclusive with ``labels``.
    labels : dict | None, optional
        ``{level: label}`` mapping that combines R's ``factor(x, levels=,
        labels=)`` into one argument: keys are the expected raw values
        (insertion order = level order), values are the displayed labels.
        Errors if the column contains a value not in ``labels.keys()``
        (via ``replace_strict``). Use this for coded integer columns —
        e.g. ``factor(s, labels={0: "no", 1: "yes"})`` collapses cast +
        rename into one pass. Mutually exclusive with ``levels``.
    ordered : bool, optional
        If True, also register the series's name in hea's ordered-cols
        contextvar so subsequent ``gam``/``lm``/``lme`` calls in this
        session apply poly contrasts. Process-global; pair with
        ``hea.formula.with_ordered_cols`` if you need scoped use.
        ``ordered=False`` does NOT remove an already-registered name —
        call ``set_ordered_cols(frozenset())`` to clear.
    strict : bool, optional
        If False (default), values not in ``levels=`` become null —
        R's ``factor()`` semantics. If True, raise on unknown values
        — forcats's ``fct()`` semantics, useful for catching typos in
        coded data. Only affects the ``levels=`` / auto-detect path;
        the ``labels=`` path always errors on unknown values.
    """
    if levels is not None and labels is not None:
        raise ValueError(
            "factor(): pass either `levels=` (list, reorder only) or "
            "`labels=` (dict {level: label}, reorder + rename), not both."
        )
    if isinstance(levels, dict):
        raise TypeError(
            "factor(): `levels=` expects a list/sequence, not a dict. "
            "For {level: label} mapping, pass it as `labels=` instead."
        )

    if isinstance(series, (str, pl.Expr)):
        return _LazyFactor(
            series, levels=levels, labels=labels, ordered=ordered, strict=strict
        )

    if not isinstance(series, pl.Series):
        series = pl.Series(series)

    s = series.cast(pl.Utf8)

    if labels is not None:
        old = [_label_key_to_str(k) for k in labels.keys()]
        new = [str(v) for v in labels.values()]
        out = s.replace_strict(old, new, return_dtype=pl.Enum(new))
    else:
        if levels is None:
            # R's factor() sorts numerically when input is numeric
            # (then string-casts); only character/factor inputs sort lex.
            if series.dtype.is_numeric():
                levels_list = [str(v) for v in series.drop_nulls().unique().sort().to_list()]
            else:
                levels_list = s.drop_nulls().unique().sort().to_list()
        else:
            levels_list = [str(v) for v in levels]
        out = s.cast(pl.Enum(levels_list), strict=strict)

    if ordered and series.name:
        from ..formula import _ORDERED_COLS_CV
        set_ordered_cols(_ORDERED_COLS_CV.get() | frozenset({series.name}))

    from ..tidy import Series as _HeaSeries
    result = _HeaSeries._from_pyseries(out._s)
    if ordered:
        # Local marker so unnamed Series (factor(bare_list, ordered=True)
        # has empty name → can't go in _ORDERED_COLS_CV) still print with
        # ``Levels: a < b < c``. Lost on derived ops, which is fine for
        # the print-after-construction use case.
        result._hea_ordered = True
    return result


def fct(x, levels=None, na=None):
    """forcats: ``fct(x, levels=NULL, na=NA)`` — thin alias around
    :func:`factor`.

    Differs from ``factor()`` only in spelling — forcats' ``fct``
    documents itself as "factor with stricter handling" but the polars
    Enum already errors on unknown levels (we cast through Utf8 and
    nulls land where R's NA would). ``na=`` is accepted for R-side
    surface compatibility but ignored.
    """
    return factor(x, levels=levels)


def ordered(series, levels=None, labels: dict | None = None):
    """R's ``ordered(x, ...)`` — shortcut for ``factor(x, ordered=True)``.

    Returns an Enum series with the ordered-factor flag set: the
    series's name (if any) is registered in hea's ordered-cols
    contextvar so subsequent model fitting (``gam`` / ``lm`` / ``lme``)
    applies polynomial contrasts, and ``print(s)`` shows
    ``Levels: a < b < c`` (with ``<`` separators) to match R's display.
    Polars ``Enum`` already provides ``<`` / ``>`` / sort semantics from
    the declared level order, so comparison and ordering "just work".
    """
    return factor(series, levels=levels, labels=labels, ordered=True)


def is_factor(x):
    """R: ``is.factor()`` — True for ``pl.Enum`` / ``pl.Categorical`` columns."""
    if isinstance(x, pl.Series):
        return isinstance(x.dtype, (pl.Enum, pl.Categorical))
    return False


def levels(x):
    """R: ``levels()`` — categories of a factor / Enum, in storage order."""
    if isinstance(x, pl.Series):
        if isinstance(x.dtype, pl.Enum):
            return x.dtype.categories.to_list()
        if isinstance(x.dtype, pl.Categorical):
            return x.cat.get_categories().to_list()
    return None


def nlevels(x):
    """R: ``nlevels()`` — number of factor categories."""
    lv = levels(x)
    return len(lv) if lv is not None else 0


def interaction(*args, drop=False, sep=".", lex_order=False):
    """R: ``interaction()`` — combine vectors into a single factor.

    Each argument is coerced to a string and joined with ``sep``; the
    result is a categorical (R's "factor"). Strings are interpreted as
    column names (polars convention), so the typical dplyr/ggplot use::

        df.ggplot(group=interaction("day", "month"))

    works identically to ``interaction(col("day"), col("month"))``.

    Parameters
    ----------
    *args : str | pl.Expr | pl.Series | list-like
        Vectors to interact. Same length required in eager mode.
    drop : bool, default False
        R default. The eager result's factor levels include the full
        Cartesian product of input unique values, even unobserved
        combinations. With ``drop=True``, only observed combinations.

        In Expr context this argument is accepted but the result always
        carries only observed levels — polars can't enumerate the
        Cartesian product without materialization. The actual grouping
        behavior (which is what matters inside ``group_by`` / ``ggplot
        group=``) is identical either way; only the level metadata
        differs.
    sep : str, default ``"."``
        Separator joining the component strings.
    lex_order : bool, default False
        If True, factor levels are sorted lexicographically by label.
        With the default ``lex_order=False``, levels follow R's "first
        factor varies fastest" Cartesian-product ordering (or
        first-appearance order when ``drop=True``).

        In Expr context, level ordering is always first-appearance —
        ``lex_order=True`` only takes effect in eager mode.

    Returns
    -------
    ``pl.Expr`` casting to ``pl.Categorical`` (Expr / string-name input);
    ``pl.Series`` with ``pl.Enum`` dtype carrying the computed levels
    (eager input).
    """
    if not args:
        raise TypeError("interaction(): need at least one argument")

    has_string_or_expr = any(isinstance(a, (str, pl.Expr)) for a in args)

    if has_string_or_expr:
        col_exprs = []
        for a in args:
            if isinstance(a, pl.Expr):
                col_exprs.append(a.cast(pl.Utf8))
            elif isinstance(a, str):
                col_exprs.append(pl.col(a).cast(pl.Utf8))
            elif isinstance(a, pl.Series):
                col_exprs.append(pl.lit(a).cast(pl.Utf8))
            else:
                col_exprs.append(pl.lit(list(a)).cast(pl.Utf8))
        combined = pl.concat_str(col_exprs, separator=sep)
        return combined.cast(pl.Categorical)

    # Eager path — compute Cartesian-product / observed levels explicitly.
    from itertools import product as _product
    str_cols: list[list] = []
    n: int | None = None
    levels_per_col: list[list[str]] = []
    for a in args:
        vals = a.to_list() if isinstance(a, pl.Series) else list(a)
        if n is None:
            n = len(vals)
        elif len(vals) != n:
            raise ValueError(
                "interaction(): all inputs must have the same length"
            )
        str_vals = [str(v) if v is not None else None for v in vals]
        str_cols.append(str_vals)
        seen: dict[str, None] = {}
        for v in str_vals:
            if v is not None and v not in seen:
                seen[v] = None
        levels_per_col.append(list(seen.keys()))

    combined: list[str | None] = [
        sep.join(str_cols[c][i] for c in range(len(str_cols)))
        if all(str_cols[c][i] is not None for c in range(len(str_cols)))
        else None
        for i in range(n or 0)
    ]

    if drop:
        seen_lvl: dict[str, None] = {}
        for v in combined:
            if v is not None and v not in seen_lvl:
                seen_lvl[v] = None
        levels = list(seen_lvl.keys())
    else:
        # R's lex.order=FALSE has the FIRST factor varying fastest;
        # itertools.product varies the LAST iterable fastest, so we
        # reverse both the input list and each output tuple.
        levels = [
            sep.join(reversed(combo))
            for combo in _product(*reversed(levels_per_col))
        ]

    if lex_order:
        levels = sorted(levels)

    return pl.Series(combined, dtype=pl.Enum(levels))
