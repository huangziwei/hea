"""``hea.Series`` / ``hea.LazyFrame`` subclasses plus the polars-class
install hooks that lift bare ``pl.Series`` / ``pl.Expr`` / ``pl.LazyGroupBy``
into hea's tidyverse-shaped surface.

* :class:`Series` — preserves the subclass through chains (polars'
  ``call_expr`` decorator and a couple of explicit ``wrap_s`` sites
  otherwise leak back to plain ``pl.Series``).
* :class:`LazyFrame` — overrides ``collect()`` and ``group_by()`` so the
  eager round-trip preserves :class:`hea.DataFrame` / :class:`GroupBy`.
* :class:`_HeaLazyGroupBy` — wraps polars' lazy GroupBy so
  ``LazyFrame.group_by(...).agg(...).collect()`` produces a hea frame.
* ``_install_*`` hooks patch :class:`polars.Expr`, :class:`polars.Series`
  shared base, and the lazy GroupBy class at module import so the wrapping
  is invisible to user code.
"""
from __future__ import annotations

from typing import Any

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy as _PlLazyGroupBy

from .dataframe import DataFrame


class Series(pl.Series):
    """``pl.Series`` that preserves the hea subclass through chains.

    Two leak shapes need handling:

    * **Direct methods** (`head`, `slice`, `cast`, `clone`, …) already
      route through ``self._from_pyseries(...)`` and propagate the
      subclass automatically.
    * **Expression-dispatched methods** (``unique``, ``drop_nulls``,
      ``shift``, ``top_k``, ``sample``, ``gather_every``, ``not_``, the
      ``rolling_*`` family, the trig family, etc. — 116 total) have empty
      function bodies on ``pl.Series`` and are auto-wrapped by polars's
      ``call_expr`` decorator. Internally they go through
      ``wrap_s(self._s)`` (`polars/series/utils.py:99`) which hardcodes
      ``pl.Series._from_pyseries`` — losing subclass identity. Plus two
      explicit ``wrap_s`` sites: ``set`` and ``shrink_dtype``.

    We install thin wrappers for every leaky method below at class-def
    time. New polars releases that add expr-dispatched methods are picked
    up automatically by ``_install_series_subclass_overrides``.
    """

    def __invert__(self):
        """``~s`` → ``pl.exclude(s.name)``, mirroring ``~df``'s
        column-exclusion semantics for the single-column case (so
        ``df.select(~df["x"])`` matches ``df.select(~df["x":"y"])``).
        Boolean Series fall through to polars's logical-NOT so filter
        masks like ``df.filter(~mask)`` keep working.
        """
        if self.dtype == pl.Boolean:
            return super().__invert__()
        return pl.exclude(self.name)

    def to_frame(self, name: str | None = None) -> "DataFrame":
        out = super().to_frame(name) if name is not None else super().to_frame()
        return DataFrame._from_pydf(out._df)

    def to_dummies(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return DataFrame._from_pydf(super().to_dummies(*args, **kwargs)._df)

    def value_counts(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return DataFrame._from_pydf(super().value_counts(*args, **kwargs)._df)

    def hist(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return DataFrame._from_pydf(super().hist(*args, **kwargs)._df)

    def is_close(self, *args: Any, **kwargs: Any) -> "Series":
        # Bypasses self._from_pyseries; rewrap.
        out = super().is_close(*args, **kwargs)
        return type(self)._from_pyseries(out._s)

    def _is_ordered_factor(self) -> bool:
        """True when this Series should print levels with ``<`` separators
        (R's ordered-factor display). Two sources, in order:

        1. Local marker set by ``factor(..., ordered=True)`` /
           ``ordered()`` — covers unnamed Series (bare-list inputs).
           Lost on derived ops; that's fine since this is a print-time
           cosmetic.
        2. The ``_ORDERED_COLS_CV`` contextvar — covers named columns
           registered for poly contrasts in model fitting, so the same
           ordered-ness flows to ``df["col"]`` views.
        """
        if getattr(self, "_hea_ordered", False):
            return True
        if self.name:
            from ..formula import _ORDERED_COLS_CV
            return self.name in _ORDERED_COLS_CV.get()
        return False

    def __str__(self) -> str:
        base = super().__str__()
        if isinstance(self.dtype, pl.Enum):
            sep = " < " if self._is_ordered_factor() else " "
            return base + "\nLevels: " + sep.join(self.dtype.categories.to_list())
        return base

    def _repr_html_(self) -> str:
        base = super()._repr_html_()
        if isinstance(self.dtype, pl.Enum):
            sep = " &lt; " if self._is_ordered_factor() else " "
            levels_html = (
                f"<small>Levels: {sep.join(self.dtype.categories.to_list())}</small>"
            )
            stripped = base.rstrip()
            if stripped.endswith("</div>"):
                return stripped[: -len("</div>")] + levels_html + "</div>"
            return base + levels_html
        return base

    # ---- ggplot entry point -------------------------------------------

    def ggplot(self, mapping=None, **aes_kwargs):
        """Start a ggplot from this Series.

        Builds a one-column DataFrame ``{self.name: self}`` and chains
        through :meth:`DataFrame.ggplot`. ``x`` defaults to the Series
        name so single-vector geoms (``geom_histogram``, ``geom_density``,
        ``geom_qq``, ``geom_boxplot``, ``geom_rug``) chain without an
        explicit aesthetic:

        ::

            s.ggplot().geom_histogram()
            s.ggplot().geom_density()
            s.ggplot(color="red").geom_rug()

        Unnamed Series get the column name ``"value"`` (same convention
        as :func:`hea.R.ts`). User-passed ``x=`` / ``y=`` override the
        default.

        ggplot2 in R strictly requires a data.frame — hea adds this
        entry point because a Series is unambiguously one numeric vector,
        with zero false-positive risk.
        """
        import inspect

        from hea.ggplot.core import ggplot as _ggplot
        from hea.plot.dispatch import _frame_env

        name = self.name or "value"
        df = DataFrame({name: self.rename(name)})
        mapped_x = ("x" in aes_kwargs) or (
            mapping is not None and "x" in mapping)
        if not mapped_x:
            aes_kwargs["x"] = name

        env = _frame_env(inspect.currentframe().f_back)
        return _ggplot(df, mapping, _env=env, **aes_kwargs)


def _install_series_subclass_overrides() -> None:
    """Install hea.Series-aware wrappers for every method on pl.Series that
    bypasses ``self._from_pyseries`` (i.e. routes through ``wrap_s``).

    Runs once at module-import time. Picks up future polars expr-dispatched
    additions automatically — no maintenance treadmill on version bumps.
    """
    from polars.series.utils import _is_empty_method, _undecorated

    def _make_wrapper(meth_name: str):
        pl_method = getattr(pl.Series, meth_name)

        def wrapper(self, *args: Any, **kwargs: Any):
            out = pl_method(self, *args, **kwargs)
            if isinstance(out, pl.Series) and not isinstance(out, Series):
                return type(self)._from_pyseries(out._s)
            return out

        wrapper.__name__ = meth_name
        wrapper.__qualname__ = f"Series.{meth_name}"
        wrapper.__doc__ = pl_method.__doc__
        return wrapper

    # All expr-dispatched methods (auto-discovered).
    leaky_names: list[str] = []
    for name in dir(pl.Series):
        if name.startswith("_"):
            continue
        attr = pl.Series.__dict__.get(name)
        if attr is None or not hasattr(attr, "__wrapped__"):
            continue
        if _is_empty_method(_undecorated(attr)):
            leaky_names.append(name)

    # Plus the two explicit wrap_s sites in polars/series/series.py.
    leaky_names += ["set", "shrink_dtype"]

    for name in leaky_names:
        setattr(Series, name, _make_wrapper(name))


_install_series_subclass_overrides()


# DataFrame methods that return ``pl.Series``. Re-wrap as ``hea.Series`` so
# chains like ``df.get_column("x").to_frame()`` stay in hea-land.
_DF_SERIES_RETURNING = (
    "drop_in_place",
    "fold",
    "get_column",
    "hash_rows",
    "is_duplicated",
    "is_unique",
    "max_horizontal",
    "mean_horizontal",
    "min_horizontal",
    "sum_horizontal",
    "to_series",
    "to_struct",
)


def _install_df_series_overrides() -> None:
    def _make(meth_name: str):
        pl_method = getattr(pl.DataFrame, meth_name)

        def wrapper(self, *args: Any, **kwargs: Any):
            out = pl_method(self, *args, **kwargs)
            if isinstance(out, pl.Series) and not isinstance(out, Series):
                return Series._from_pyseries(out._s)
            return out

        wrapper.__name__ = meth_name
        wrapper.__qualname__ = f"DataFrame.{meth_name}"
        wrapper.__doc__ = pl_method.__doc__
        return wrapper

    for name in _DF_SERIES_RETURNING:
        setattr(DataFrame, name, _make(name))

    # ``__getitem__`` is polymorphic (Series for str key, DataFrame for slice,
    # row tuple for int) — handle each branch.
    pl_getitem = pl.DataFrame.__getitem__

    def __getitem__(self, item):
        out = pl_getitem(self, item)
        if isinstance(out, pl.Series) and not isinstance(out, Series):
            return Series._from_pyseries(out._s)
        if isinstance(out, pl.DataFrame) and not isinstance(out, DataFrame):
            return DataFrame._from_pydf(out._df)
        return out

    __getitem__.__doc__ = pl_getitem.__doc__
    DataFrame.__getitem__ = __getitem__


_install_df_series_overrides()


def _install_is_in_mixed_list_support() -> None:
    """Teach ``pl.Expr.is_in`` to accept Python lists that mix literals
    and ``Expr`` values.

    Polars' built-in ``is_in`` tries to coerce the ``other`` list into a
    homogeneous ``Series``; a list like ``[1, col("r").max()]`` errors
    with ``failed to determine supertype of i64 and object``. The
    dplyr-faithful translation of ``r %in% c(1, max(r))`` is
    ``col("r").is_in([1, col("r").max()])``, so we patch ``is_in``:
    when ``other`` contains any ``pl.Expr``, we expand into an OR-chain
    (``(self == v0) | (self == v1) | …``), which polars evaluates row-
    wise without dtype headaches. All-literal lists pass through to the
    original ``is_in`` unchanged.

    Series-side eager ``is_in`` is left alone — mixing an Expr into an
    eager membership test has no column to bind against, so polars'
    original error is the right answer there.
    """
    _orig_expr_is_in = pl.Expr.is_in

    def wrapper(self, other, *args, **kwargs):
        if isinstance(other, (list, tuple)) and any(
            isinstance(v, pl.Expr) for v in other
        ):
            nulls_equal = kwargs.get("nulls_equal", False)
            result = None
            for v in other:
                rhs = v if isinstance(v, pl.Expr) else pl.lit(v)
                cmp = self.eq_missing(rhs) if nulls_equal else self == rhs
                result = cmp if result is None else (result | cmp)
            return result
        return _orig_expr_is_in(self, other, *args, **kwargs)

    wrapper.__name__ = "is_in"
    wrapper.__qualname__ = "Expr.is_in (hea-patched)"
    wrapper.__doc__ = (_orig_expr_is_in.__doc__ or "") + (
        "\n\nhea extension: accepts a Python list mixing literals and "
        "``pl.Expr`` values. Mixed lists are expanded to an OR-chain of "
        "``self == v`` comparisons, matching R's ``x %in% c(1, max(x))``."
    )
    pl.Expr.is_in = wrapper


_install_is_in_mixed_list_support()


def _install_expr_is_na_alias() -> None:
    """Alias ``pl.Expr.is_na`` to ``is_null`` so R-translated code that
    emits ``col("x").is_na()`` works.

    Polars named its null-check ``is_null`` (``is_nan`` is the float-NaN
    one); the R-to-Python translator emits the R spelling. Without this
    alias, ``Expr`` raises ``AttributeError: 'Expr' object has no
    attribute 'is_na'``.
    """
    if not hasattr(pl.Expr, "is_na"):
        pl.Expr.is_na = pl.Expr.is_null


_install_expr_is_na_alias()


def _install_expr_r_aliases() -> None:
    """Alias R/dplyr spellings of cumulative ops on ``pl.Expr``.

    Polars renamed ``cumsum`` → ``cum_sum``, ``cummax`` → ``cum_max``,
    ``cummin`` → ``cum_min``, ``cumprod`` → ``cum_prod`` somewhere
    around v1.0. R / dplyr keep the un-underscored spellings; the
    R-to-Python translator emits R names. Without these aliases,
    ``col('x').cumsum()`` raises AttributeError on current polars.
    """
    aliases = {
        "cumsum":  "cum_sum",
        "cummax":  "cum_max",
        "cummin":  "cum_min",
        "cumprod": "cum_prod",
    }
    for r_name, polars_name in aliases.items():
        if not hasattr(pl.Expr, r_name) and hasattr(pl.Expr, polars_name):
            setattr(pl.Expr, r_name, getattr(pl.Expr, polars_name))


_install_expr_r_aliases()


class LazyFrame(pl.LazyFrame):
    """``pl.LazyFrame`` that re-wraps materialized results as ``hea.DataFrame``.

    Mostly empty — polars LazyFrame methods route through
    ``self._from_pyldf(...)``, which respects the subclass, so chains
    (``.filter(...).with_columns(...).join(...)``) propagate
    ``hea.LazyFrame`` automatically. The overrides below cover the
    handful of methods that bypass ``self._from_pyldf`` (calling
    ``wrap_ldf`` / ``wrap_df`` instead) — including the eager-via-lazy
    leak point at `polars/lazyframe/frame.py:2510` (collect).
    """

    def _wrap(self, lf: pl.LazyFrame) -> "LazyFrame":
        return type(self)._from_pyldf(lf._ldf)

    def collect(self, *args: Any, **kwargs: Any):
        out = super().collect(*args, **kwargs)
        if isinstance(out, pl.DataFrame):
            return DataFrame._from_pydf(out._df)
        # background=True path — polars returns InProcessQuery whose
        # .fetch() / .fetch_blocking() still uses pl.DataFrame. Rare
        # enough to leave un-wrapped for now (allowlisted).
        return out

    def describe(self, *args: Any, **kwargs: Any) -> "DataFrame":
        # Despite living on LazyFrame, describe() materializes — returns DataFrame.
        return DataFrame._from_pydf(super().describe(*args, **kwargs)._df)

    def match_to_schema(self, *args: Any, **kwargs: Any) -> "LazyFrame":
        return self._wrap(super().match_to_schema(*args, **kwargs))

    def sql(self, *args: Any, **kwargs: Any) -> "LazyFrame":
        return self._wrap(super().sql(*args, **kwargs))

    def group_by(self, *args: Any, **kwargs: Any) -> "_HeaLazyGroupBy":
        return _HeaLazyGroupBy(super().group_by(*args, **kwargs).lgb)

    def group_by_dynamic(self, *args: Any, **kwargs: Any) -> "_HeaLazyGroupBy":
        return _HeaLazyGroupBy(super().group_by_dynamic(*args, **kwargs).lgb)

    def rolling(self, *args: Any, **kwargs: Any) -> "_HeaLazyGroupBy":
        return _HeaLazyGroupBy(super().rolling(*args, **kwargs).lgb)


class _HeaLazyGroupBy(_PlLazyGroupBy):
    """Subclass of polars's ``LazyGroupBy`` that re-wraps every LazyFrame
    return as ``hea.LazyFrame``.

    polars's ``LazyGroupBy.agg`` (and ``head``/``tail``/``sum``/etc.) all
    use ``wrap_ldf(...)`` (`polars/lazyframe/group_by.py:194,263,…`) which
    hardcodes ``pl.LazyFrame``. We auto-wrap every LazyFrame-returning
    method via ``_install_lazy_groupby_overrides`` below so that
    ``df.lazy().group_by('g').agg(...)`` chains stay in hea-land.

    Private (leading underscore) — only reachable via ``LazyFrame.group_by``,
    not part of the public API surface.
    """


def _install_lazy_groupby_overrides() -> None:
    def _make(meth_name: str):
        pl_method = getattr(_PlLazyGroupBy, meth_name)

        def wrapper(self, *args: Any, **kwargs: Any):
            out = pl_method(self, *args, **kwargs)
            if isinstance(out, pl.LazyFrame) and not isinstance(out, LazyFrame):
                return LazyFrame._from_pyldf(out._ldf)
            return out

        wrapper.__name__ = meth_name
        wrapper.__qualname__ = f"_HeaLazyGroupBy.{meth_name}"
        wrapper.__doc__ = pl_method.__doc__
        return wrapper

    for name in dir(_PlLazyGroupBy):
        if name.startswith("_"):
            continue
        attr = getattr(_PlLazyGroupBy, name, None)
        if not callable(attr):
            continue
        setattr(_HeaLazyGroupBy, name, _make(name))


_install_lazy_groupby_overrides()


