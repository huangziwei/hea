"""Tests for ``hea.R`` — the R-like free-function namespace.

The module is designed for ``from hea.R import *``, so we exercise it
that way to confirm the public surface is what we advertise and that
none of it shadows a Python builtin.
"""

from __future__ import annotations

import builtins
import math
import sys

import numpy as np
import polars as pl
import pytest

from conftest import have_rscript, r_scalar_values

import hea
from hea import R as R_mod
from hea.R import nmath as _nm
from hea.R import (
    as_character, as_integer, as_logical, as_numeric,
    colnames, complete_cases, cor, cov, cumsum, cummax, cummin, cumprod,
    dim, diff, duplicated,
    factor, head,
    is_factor, is_finite, is_na, is_null, is_numeric,
    length, levels, mean, median,
    na_omit, names, ncol, nlevels, nrow,
    order,
    quantile,
    rank as R_rank, rev, sd, seq, seq_along, seq_len, signed_rank,
    sort, summary, tail,
    tabulate, unique, var, which, which_max, which_min,
    # distributions (a representative subset; full grid checked elsewhere)
    dnorm, pnorm, qnorm, rnorm,
    dt, pt, qt, dchisq, qchisq, pchisq, qf, pf,
    ptukey, qtukey,
    dhyper, phyper,
    dsignrank, psignrank, qsignrank, dwilcox, pwilcox, qwilcox,
    dbinom, pbinom, dpois, ppois, punif, qexp, pgamma, pbeta,
    dcauchy, pcauchy, qcauchy, rcauchy,
    dlogis, plogis, qlogis,
    dlnorm, plnorm, qlnorm,
    dweibull, pweibull, qweibull, rweibull,
    dgeom, pgeom, qgeom, rgeom,
    qhyper, dnbinom, pnbinom, qnbinom,
    rnbinom, rhyper, rsignrank, rwilcox, rmultinom,
    r2dtable, rWishart, dmultinom, pbirthday, qbirthday,
    psmirnov, qsmirnov, rsmirnov,
    set_seed,
)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_no_builtin_collisions():
    """``from hea.R import *`` must not redefine a Python builtin."""
    py_builtins = set(dir(builtins))
    exported = set(R_mod.__all__)
    overlap = exported & py_builtins
    assert overlap == set(), f"R module shadows builtins: {overlap}"


def test_all_exports_are_defined():
    """Every name in ``__all__`` must resolve to an attribute."""
    for name in R_mod.__all__:
        assert hasattr(R_mod, name), f"R.{name} declared but not defined"


# ---------------------------------------------------------------------------
# Shape / preview
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    return pl.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6, 7], "b": list("abcdefg")}
    )


def test_head_tail_dispatch_on_dataframe(df):
    assert head(df, 3).height == 3
    assert tail(df, 2).height == 2


def test_head_tail_on_list():
    assert head([1, 2, 3, 4, 5, 6, 7, 8, 9], 4) == [1, 2, 3, 4]
    assert tail([1, 2, 3, 4, 5, 6, 7, 8, 9], 4) == [6, 7, 8, 9]


def test_nrow_ncol_dim(df):
    assert nrow(df) == 7
    assert ncol(df) == 2
    assert dim(df) == (7, 2)


def test_length_on_dataframe_is_ncol(df):
    """R: ``length(data.frame)`` returns ``ncol``, not ``nrow``."""
    assert length(df) == 2


def test_colnames_and_names(df):
    assert colnames(df) == ["a", "b"]
    assert names(df) == ["a", "b"]
    assert names(df["a"]) == "a"
    assert names({"x": 1, "y": 2}) == ["x", "y"]


def test_summary_dispatches_on_hea_dataframe():
    """``summary(hea.tidy.DataFrame)`` must reach the existing ``.summary()``."""
    d = hea.tidy.tbl(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    out = summary(d)
    # Existing .summary() returns a Summary object with __repr__
    assert "Min" in repr(out)


def test_summary_raises_on_unsupported():
    with pytest.raises(TypeError, match="no .summary"):
        summary(np.array([1, 2, 3]))


def test_complete_cases_and_na_omit_on_dataframe():
    d = pl.DataFrame({"a": [1, None, 3, 4], "b": [10, 20, None, 40]})
    assert complete_cases(d).to_list() == [True, False, False, True]
    assert na_omit(d).height == 2


def test_complete_cases_on_array():
    arr = np.array([1.0, np.nan, 3.0])
    assert complete_cases(arr).tolist() == [True, False, True]
    assert na_omit(arr).tolist() == [1.0, 3.0]


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def test_seq_one_arg_is_zero_based():
    """One-arg ``seq(n)`` matches ``np.arange(n)``, not R's ``1:n``."""
    assert seq(5).tolist() == [0, 1, 2, 3, 4]


def test_seq_from_to_is_inclusive():
    """Two-arg ``seq(from, to)`` keeps R's inclusive endpoints."""
    assert seq(2, 6).tolist() == [2, 3, 4, 5, 6]
    # The R-1:n bridge: explicit start makes 1-based available again.
    assert seq(1, 5).tolist() == [1, 2, 3, 4, 5]


def test_seq_with_by():
    assert seq(2, 10, by=2).tolist() == [2, 4, 6, 8, 10]
    assert seq(10, 2, by=-2).tolist() == [10, 8, 6, 4, 2]


def test_seq_length_out():
    assert seq(0, 1, length_out=5).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_seq_along_with_is_zero_based():
    assert seq(along_with=["a", "b", "c"]).tolist() == [0, 1, 2]


def test_seq_len_seq_along_zero_based():
    """Both return Python-style indices, safe for ``x[i]`` iteration."""
    assert seq_len(4).tolist() == [0, 1, 2, 3]
    assert seq_along(["x", "y"]).tolist() == [0, 1]


def test_rev():
    assert rev([1, 2, 3]) == [3, 2, 1]
    assert rev(np.array([1, 2, 3])).tolist() == [3, 2, 1]
    assert rev(pl.Series([1, 2, 3])).to_list() == [3, 2, 1]


def test_sort_decreasing():
    assert sort([3, 1, 2]).tolist() == [1, 2, 3]
    assert sort([3, 1, 2], decreasing=True).tolist() == [3, 2, 1]


def test_order_zero_based_python_convention():
    # x = [3, 1, 2]; sorted is [1, 2, 3] from indices [1, 2, 0]
    assert order([3, 1, 2]).tolist() == [1, 2, 0]


def test_order_decreasing():
    assert order([3, 1, 2], decreasing=True).tolist() == [0, 2, 1]


def test_which_zero_based():
    assert which([True, False, True, True]).tolist() == [0, 2, 3]


def test_which_max_min():
    assert which_max([1, 5, 3]) == 1
    assert which_min([4, 2, 8, 2]) == 1  # first occurrence


def test_cumulative():
    assert cumsum([1, 2, 3, 4]).tolist() == [1, 3, 6, 10]
    assert cumprod([1, 2, 3, 4]).tolist() == [1, 2, 6, 24]
    assert cummax([1, 3, 2, 4, 1]).tolist() == [1, 3, 3, 4, 4]
    assert cummin([4, 3, 5, 1, 2]).tolist() == [4, 3, 3, 1, 1]


def test_diff():
    assert diff([1, 3, 6, 10]).tolist() == [2, 3, 4]
    assert diff([1, 2, 4, 7, 11], differences=2).tolist() == [1, 1, 1]


def test_unique_preserves_order():
    """R's ``unique`` preserves first-occurrence order, unlike np.unique."""
    assert unique([3, 1, 2, 1, 3, 2]).tolist() == [3, 1, 2]


def test_duplicated():
    assert duplicated([1, 2, 2, 3, 1]).tolist() == [
        False, False, True, False, True
    ]
    s = pl.Series([1, 2, 2, 3, 1])
    assert duplicated(s).to_list() == [False, False, True, False, True]


def test_tabulate_zero_based():
    """hea's ``tabulate`` uses 0-based bins (Python convention).
    R / dplyr's ``tabulate(c(1,2,2,3,3,3))`` -> c(1, 2, 3); hea's
    equivalent is ``tabulate([0,1,1,2,2,2])`` -> [1, 2, 3]."""
    assert tabulate([0, 1, 1, 2, 2, 2]).tolist() == [1, 2, 3]
    assert tabulate([1, 1, 3], nbins=5).tolist() == [0, 2, 0, 1, 0]


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_mean_median():
    assert mean([1, 2, 3, 4, 5]) == 3.0
    assert median([1, 2, 3, 4, 5]) == 3.0


def test_var_sd_use_n_minus_1():
    # var of 1..5 with N-1: sum((x-3)^2) / 4 = 10/4 = 2.5
    assert var([1, 2, 3, 4, 5]) == pytest.approx(2.5)
    assert sd([1, 2, 3, 4, 5]) == pytest.approx(np.sqrt(2.5))


def test_var_two_arg_is_covariance():
    """R: ``var(x, y)`` returns the sample covariance."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert var(x, y) == pytest.approx(cov(x, y))


def test_cor_scalar_and_matrix():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert cor(x, y) == pytest.approx(1.0)
    m = np.column_stack([x, y])
    assert cor(m).shape == (2, 2)


# ---------------------------------------------------------------------------
# IQR — agreement with R stats::IQR
# ---------------------------------------------------------------------------
#
# R reference (computed via R --vanilla):
#   IQR(c(2,5,11,11,19,35))            -> 10.5   (type 7 default, linear)
#   IQR(c(2,5,11,11,19,35), type=1)    -> 14
#   IQR(c(2,5,11,11,19,35), type=4)    -> 11.5
#   IQR(c(1, NA, 3), na.rm=TRUE)       -> 1


def test_IQR_eager_list_matches_R_default():
    from hea.R import IQR
    assert IQR([2, 5, 11, 11, 19, 35]) == 10.5


def test_IQR_eager_quantile_types():
    from hea.R import IQR
    x = [2, 5, 11, 11, 19, 35]
    assert IQR(x, type=1) == 14
    assert IQR(x, type=4) == 11.5
    assert IQR(x, type=7) == 10.5  # default


def test_IQR_na_rm_drops_nulls():
    from hea.R import IQR
    assert IQR([1, None, 3], na_rm=True) == 1.0


def test_IQR_invalid_type_raises():
    from hea.R import IQR
    with pytest.raises(ValueError, match="1..9"):
        IQR([1, 2, 3], type=10)
    with pytest.raises(ValueError, match="1..9"):
        IQR([1, 2, 3], type=0)


def test_IQR_series_returns_scalar():
    from hea.R import IQR
    s = pl.Series([2, 5, 11, 11, 19, 35])
    assert IQR(s) == 10.5
    # hea's default ``na_rm=True`` skips nulls — IQR over the non-null
    # values [1.0, 3.0] gives 1.0.
    assert IQR(pl.Series([1.0, None, 3.0])) == 1.0
    # Opt-out: na_rm=False yields null (hea is graceful; R errors here).
    assert IQR(pl.Series([1.0, None, 3.0]), na_rm=False) is None


# ---------------------------------------------------------------------------
# interaction — R reference values
# ---------------------------------------------------------------------------
#
#   interaction(c("a","b","a"), c(1,2,1))
#     -> values: a.1 b.2 a.1
#     -> Levels: a.1 b.1 a.2 b.2   (Cartesian product; first factor fastest)
#   interaction(..., drop=TRUE)
#     -> Levels: a.1 b.2           (observed only)
#   interaction(..., lex.order=TRUE)
#     -> Levels: a.1 a.2 b.1 b.2   (alphabetical)


def test_interaction_default_matches_R_cartesian_levels():
    from hea.R import interaction
    out = interaction(["a", "b", "a"], [1, 2, 1])
    assert isinstance(out, pl.Series)
    assert out.to_list() == ["a.1", "b.2", "a.1"]
    assert out.dtype.categories.to_list() == ["a.1", "b.1", "a.2", "b.2"]


def test_interaction_drop_true_keeps_observed_only():
    from hea.R import interaction
    out = interaction(["a", "b", "a"], [1, 2, 1], drop=True)
    assert out.dtype.categories.to_list() == ["a.1", "b.2"]


def test_interaction_lex_order_sorts_levels():
    from hea.R import interaction
    out = interaction(["a", "b", "a"], [1, 2, 1], lex_order=True)
    assert out.dtype.categories.to_list() == ["a.1", "a.2", "b.1", "b.2"]


def test_interaction_custom_sep():
    from hea.R import interaction
    out = interaction(["a", "b"], [1, 2], sep="_")
    assert out.to_list() == ["a_1", "b_2"]
    assert out.dtype.categories.to_list() == ["a_1", "b_1", "a_2", "b_2"]


def test_interaction_na_propagates():
    """If any input is null at row i, the result at row i is null."""
    from hea.R import interaction
    out = interaction(["a", None, "b"], [1, 2, None])
    assert out.to_list() == ["a.1", None, None]


def test_interaction_no_args_raises():
    from hea.R import interaction
    with pytest.raises(TypeError, match="at least one"):
        interaction()


def test_interaction_unequal_lengths_raises():
    from hea.R import interaction
    with pytest.raises(ValueError, match="same length"):
        interaction(["a", "b"], [1, 2, 3])


def test_interaction_expr_returns_categorical_expr():
    """In Expr context, returns a Categorical-typed Expr that composes
    with mutate / group_by / ggplot's group= aesthetic."""
    from hea.R import interaction
    df = pl.DataFrame({"day": [1, 2, 1, 2], "month": [1, 1, 2, 2]})
    out = df.with_columns(g=interaction("day", "month"))
    assert out["g"].dtype == pl.Categorical
    assert out["g"].to_list() == ["1.1", "2.1", "1.2", "2.2"]


def test_IQR_non_type_7_on_expr_raises():
    """Polars only has linear interpolation; other R types only work eager."""
    from hea.R import IQR
    with pytest.raises(NotImplementedError, match="type=4"):
        IQR(pl.col("x"), type=4)
    with pytest.raises(NotImplementedError, match="type=4"):
        IQR(pl.Series([1.0, 2.0]), type=4)


def test_quantile_default_probs():
    # Linear interpolation matches R's type 7 (default).
    out = quantile([1, 2, 3, 4, 5])
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_na_rm_drops_nans():
    arr = [1.0, 2.0, float("nan"), 4.0]
    assert mean(arr, na_rm=True) == pytest.approx(7.0 / 3)
    assert sd(arr, na_rm=True) > 0


# ---------------------------------------------------------------------------
# Coercion / predicates
# ---------------------------------------------------------------------------


def test_as_numeric_array_and_series():
    assert as_numeric(["1", "2", "3"]).dtype.kind == "f"
    s = pl.Series([1, 2, 3])
    assert as_numeric(s).dtype == pl.Float64


def test_as_integer_character_logical():
    assert as_integer([1.5, 2.7]).dtype == np.int64
    assert as_character([1, 2]).dtype.kind == "U"
    assert as_logical([0, 1, 2]).tolist() == [False, True, True]


def test_is_na_array_and_series():
    assert is_na([1.0, float("nan"), 3.0]).tolist() == [False, True, False]
    s = pl.Series([1, None, 3])
    assert is_na(s).to_list() == [False, True, False]


def test_is_null_is_finite_is_numeric():
    assert is_null(None) is True
    assert is_null(0) is False
    assert is_finite([1.0, float("inf"), float("nan")]).tolist() == [
        True, False, False
    ]
    assert is_numeric(pl.Series([1, 2, 3]))
    assert not is_numeric(pl.Series(["a", "b"]))


def test_factor_levels_nlevels_is_factor():
    s = factor(pl.Series("g", ["b", "a", "c", "a"]))
    assert is_factor(s)
    assert levels(s) == ["a", "b", "c"]
    assert nlevels(s) == 3
    assert not is_factor(pl.Series([1, 2, 3]))
    assert levels(pl.Series([1, 2, 3])) is None
    assert nlevels(pl.Series([1, 2, 3])) == 0


def test_factor_accepts_list_and_unknown_value_handling():
    """R's ``factor()`` accepts a character vector — bare Python lists
    and numpy arrays should work the same. Unknown values default to
    null (R parity); ``strict=True`` raises (forcats ``fct()`` parity).
    """
    month_levels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # list input, all values known
    y1 = factor(["Dec", "Apr", "Jan", "Mar"], levels=month_levels)
    assert y1.to_list() == ["Dec", "Apr", "Jan", "Mar"]
    assert levels(y1) == month_levels

    # numpy array input
    y_np = factor(np.array(["Dec", "Apr"]), levels=month_levels)
    assert y_np.to_list() == ["Dec", "Apr"]

    # default strict=False — unknown "Jam" → null (R factor() semantics)
    y2 = factor(["Dec", "Apr", "Jam", "Mar"], levels=month_levels)
    assert y2.to_list() == ["Dec", "Apr", None, "Mar"]

    # strict=True — unknown raises (forcats fct() semantics)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        factor(["Dec", "Apr", "Jam", "Mar"], levels=month_levels, strict=True)

    # strict=True with clean input still works
    y3 = factor(["Dec", "Apr"], levels=month_levels, strict=True)
    assert y3.to_list() == ["Dec", "Apr"]


def test_factor_repr_appends_levels_line():
    """R prints ``Levels: ...`` after the values — useful for inspecting
    factor objects. The hea.tidy.Series repr override appends the line when
    dtype is ``pl.Enum``. Non-enum series are unchanged.
    """
    y1 = factor(["Dec", "Apr", "Jan", "Mar"])  # auto: alphabetical
    text = str(y1)
    assert "Levels: Apr Dec Jan Mar" in text

    # explicit levels keep the user-specified order in the Levels line
    month_levels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    y2 = factor(["Dec", "Apr"], levels=month_levels)
    assert "Levels: " + " ".join(month_levels) in str(y2)

    # HTML repr (Jupyter) appends Levels inside the rendered div
    html = y2._repr_html_()
    assert "Levels: " + " ".join(month_levels) in html
    assert html.rstrip().endswith("</div>")

    # non-enum series: no Levels line
    assert "Levels:" not in str(pl.Series([1, 2, 3]))


def test_ordered_factor_renders_with_lt_separators():
    """R's ``ordered()`` displays ``Levels: a < b < c`` to distinguish
    ordered factors from regular ones. hea matches via two detection
    paths: a local ``_hea_ordered`` marker on the Series (covers
    unnamed inputs like ``ordered([a,b,c])``) and the global
    ``_ORDERED_COLS_CV`` contextvar (covers named columns).
    """
    from hea.R import ordered
    from hea.formula import _ORDERED_COLS_CV, set_ordered_cols

    # 1. Bare-list input (unnamed): local marker path
    y = ordered(["a", "b", "c"])
    assert "Levels: a < b < c" in str(y)
    assert "&lt;" in y._repr_html_()

    # 2. factor(..., ordered=True) is the same alias underneath
    from hea.R import factor
    y2 = factor(["c", "a", "b"], levels=["a", "b", "c"], ordered=True)
    assert "Levels: a < b < c" in str(y2)

    # 3. Unordered factor still uses spaces (no <)
    y3 = factor(["a", "b", "c"])
    assert "Levels: a b c" in str(y3)
    assert "<" not in str(y3).split("Levels:")[1].split("\n")[0]

    # 4. Contextvar path: named column registered for poly contrasts
    prev = _ORDERED_COLS_CV.get()
    try:
        s = pl.Series("g", ["a", "b", "c"]).cast(pl.Enum(["a", "b", "c"]))
        hea_s = hea.tidy.Series._from_pyseries(s._s)
        # No local marker → check before registration: should be space-separated
        assert "Levels: a b c" in str(hea_s)
        # Now register the name in the ordered-cols contextvar
        set_ordered_cols(prev | frozenset({"g"}))
        # Same Series — detection now flips to "<" via the contextvar
        assert "Levels: a < b < c" in str(hea_s)
    finally:
        set_ordered_cols(prev)


# ---------------------------------------------------------------------------
# Readr-style parsing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rank — base R / Lindeløv constructions. The dplyr rank family (min_rank,
# dense_rank, percent_rank, cume_dist, ntile, row_number) is tested in
# test_dataframe.py — it lives in hea.dataframe with the tidyverse port.
# ---------------------------------------------------------------------------


def test_rank_expr_in_expr_out_preserves_average_method():
    """``rank()`` keeps ``ties.method = "average"`` — the lm/Wilcoxon contract."""
    df = pl.DataFrame({"x": [1.0, 5.0, 5.0, 17.0]})
    out = df.select(r=R_rank(pl.col("x")))["r"].to_list()
    assert out == [1.0, 2.5, 2.5, 4.0]


def test_signed_rank_expr_matches_eager_numpy():
    """Expr path must produce the same values as the existing numpy path."""
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    eager = signed_rank(arr)
    df = pl.DataFrame({"x": arr})
    expr_out = df.select(s=signed_rank(pl.col("x")))["s"].to_numpy()
    np.testing.assert_array_equal(eager, expr_out)


def test_rank_signed_rank_ndarray_backwards_compat():
    """The existing numpy-out contract (used by the tests-as-lm notebook)
    must keep working for ndarray input."""
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    out = R_rank(arr)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [3.0, 1.5, 4.0, 1.5, 5.0]
    out2 = signed_rank(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert isinstance(out2, np.ndarray)
    assert out2.tolist() == [-4.5, -2.5, 0.0, 2.5, 4.5]


# ---------------------------------------------------------------------------
# Expr-dispatch contract — enforcement test
#
# Vector-shape R.py functions (length-preserving transforms and scalar
# reductions) must dispatch on pl.Expr: given a pl.Expr, return a pl.Expr.
# This is what makes ``mutate(m = mean(col("x")))`` work for the R-shaped
# API the same way it does for the polars-shaped one. Exemptions below
# are categorical — anything not in a SKIP category must pass.
# ---------------------------------------------------------------------------


# Names that legitimately don't fit the "Expr in, Expr out" contract.
# Grouped by reason so future additions to R.__all__ trip the test until
# the author explicitly classifies them.
_R_EXPR_SKIP = {
    # Hypothesis tests — return HTest, not Expr.
    "t_test", "wilcox_test", "cor_test", "kruskal_test", "chisq_test",
    "fisher_test", "prop_test", "binom_test", "var_test", "bartlett_test",
    "shapiro_test", "ks_test", "mcnemar_test", "friedman_test", "aov",
    # Result classes — not callable in the vector-shape sense.
    "HTest", "AnovaTable", "Terms",
    # Model generics — operate on fitted models, not columns.
    "coef", "coefficients", "fixef", "ranef", "refit", "refitML",
    "resid", "residuals", "fitted", "fitted_values",
    "predict", "confint", "vcov", "logLik", "deviance", "profile", "bootMer",
    "nobs", "weights", "df_residual", "formula", "model_matrix", "model_frame",
    "terms", "update", "AIC", "BIC", "effects", "simulate",
    "variable_names", "case_names", "labels",
    "anova", "add1", "drop1", "step",
    # lme4 merMod accessors (operate on a fitted gmm, not on columns).
    "VarCorr", "getME", "getData", "extractAIC", "rePCA",
    "isREML", "isLMM", "isGLMM", "isNLMM", "isSingular",
    "hatvalues", "rstandard", "rstudent",
    "cooks_distance", "dffits", "dfbeta", "dfbetas", "influence",
    # emmeans — model-shaped (operate on fitted models / EmmGrid tables).
    "emmeans", "EmmGrid", "summary_emmgrid_contrasts",
    # Distribution PDFs/CDFs/quantiles/random — scalar in, scalar out.
    "dnorm", "pnorm", "qnorm", "rnorm",
    "dt", "pt", "qt", "rt",
    "dchisq", "pchisq", "qchisq", "rchisq",
    "pf", "qf", "rf",
    "dbinom", "pbinom", "qbinom", "rbinom",
    "dpois", "ppois", "qpois", "rpois",
    "dunif", "punif", "qunif", "runif",
    "ptukey", "qtukey",
    "dsignrank", "psignrank", "qsignrank", "rsignrank",
    "dwilcox", "pwilcox", "qwilcox", "rwilcox",
    "dhyper", "phyper", "qhyper", "rhyper",
    "dnbinom", "pnbinom", "qnbinom", "rnbinom",
    # combinatorial / multivariate + Smirnov distribution surface.
    "dmultinom", "rmultinom", "pbirthday", "qbirthday", "r2dtable", "rWishart",
    "psmirnov", "qsmirnov", "rsmirnov",
    "dexp", "pexp", "qexp", "rexp",
    "dgamma", "pgamma", "qgamma", "rgamma",
    "dbeta", "pbeta", "qbeta", "rbeta",
    "dcauchy", "pcauchy", "qcauchy", "rcauchy",
    "dlogis", "plogis", "qlogis", "rlogis",
    "dlnorm", "plnorm", "qlnorm", "rlnorm",
    "dweibull", "pweibull", "qweibull", "rweibull",
    "dgeom", "pgeom", "qgeom", "rgeom",
    "set_seed",
    # Frame-meta — operate on the DataFrame, not a column.
    "nrow", "ncol", "dim", "length", "colnames", "names",
    "head", "tail", "summary", "complete_cases", "na_omit",
    # Matrix / frame utilities — operate on 2D shapes, not single columns.
    "rowSums", "colSums", "rowMeans", "colMeans",
    "apply", "rbind", "cbind", "sweep", "expand_grid", "matrix",
    "R_range", "R_round",
    # Distance / clustering — operate on data matrices, Dist objects, or hclust
    # trees, not single columns (base-R stats clustering surface).
    "Dist", "dist", "as_dist", "as_matrix_dist", "format_dist", "labels_dist",
    "print_dist", "mahalanobis", "cmdscale",
    "Hclust", "hclust", "as_hclust", "cutree", "cophenetic", "print_hclust",
    "Kmeans", "kmeans", "fitted_kmeans", "print_kmeans",
    # Dendrogram subsystem — operate on Dendrogram trees, not single columns.
    "Dendrogram", "as_dendrogram", "cophenetic_dendrogram", "cut_dendrogram",
    "dendrapply", "is_leaf", "labels_dendrogram", "merge_dendrogram",
    "midcache_dendrogram", "nleaves", "nobs_dendrogram", "order_dendrogram",
    "print_dendrogram", "reorder", "reorder_dendrogram", "rev_dendrogram",
    "str_dendrogram",
    # Vector primitives — variadic / multi-arg; not column ops.
    "rep", "sample", "sapply", "tapply",
    # Length-changing transforms — would shorten/lengthen the column.
    "diff", "which", "tabulate",
    # Container / contingency tables — return tables, not Exprs.
    "table", "xtabs", "prop_table", "addmargins",
    # Sequence generators — take ints, not columns.
    "seq", "seq_len", "seq_along",
    # Time series construction — returns a DataFrame (R's ``ts``), not an Expr.
    "ts",
    # Variadic / index-based — multi-input shape.
    "order",
    # Bucketing — eager-only (custom labels machinery).
    "cut", "findInterval",
    # Categorical / dtype introspection — Expr has no eval-time dtype info.
    "factor", "fct", "ordered", "levels", "nlevels", "is_factor", "is_numeric", "is_null",
    # cov: no clean polars top-level for 2-vector covariance (compute
    # manually via (x - x.mean()) * (y - y.mean()) / (n - 1) if needed).
    "cov",
    # I/O & clock — side-effect functions; not column ops.
    "cat", "today", "now",
    # plotmath — takes R-source string, not a column.
    "quote",
    # stringr regex-debug pretty-printers — print to stdout, return None.
    "str_view", "str_view_all",
    # lubridate parsers — operate on strings / scalars, not column Exprs.
    "ymd", "mdy", "dmy",
    "ymd_hms", "ymd_hm", "mdy_hms", "mdy_hm", "dmy_hms", "dmy_hm",
    # stringr helpers that don't take an Expr as the first arg or take
    # multiple required args (covered separately by _R_EXPR_EXTRA below).
    "str_glue", "str_sort", "str_equal",
}


# Functions that need an extra positional / keyword arg beyond ``x`` to
# produce a meaningful Expr. Keyed by name; value is a callable that
# returns the *additional* args + kwargs given the test's ``pl.col("x")``.
_R_EXPR_EXTRA: dict[str, callable] = {
    "cor":      lambda c: ((c,), {}),       # cor needs (x, y)
    "quantile": lambda c: ((0.5,), {}),     # Expr needs scalar prob
    "atan2":    lambda c: ((c,), {}),       # atan2 needs (y, x)
    "str_detect": lambda c: (("[aeiou]",), {}),  # needs pattern
    "str_count":  lambda c: (("[aeiou]",), {}),  # needs pattern
    "str_sub":    lambda c: ((1, 3), {}),        # start, end
}


def test_R_vector_functions_dispatch_on_expr():
    """Every R.py function not in ``_R_EXPR_SKIP`` must dispatch on Expr.

    This is the load-bearing rule for the R-shaped API: an R function
    applied to ``pl.col("x")`` inside ``mutate`` must produce an Expr that
    polars can evaluate, not a numpy array (which triggers the
    ``pl.lit(ndarray-of-Expr)`` failure path).
    """
    import hea.R as R_mod
    c = pl.col("x")
    failures = []
    for name in R_mod.__all__:
        if name in _R_EXPR_SKIP:
            continue
        fn = getattr(R_mod, name)
        if not callable(fn):
            continue
        extra_args, extra_kwargs = _R_EXPR_EXTRA.get(name, lambda _: ((), {}))(c)
        try:
            result = fn(c, *extra_args, **extra_kwargs)
        except Exception as e:
            failures.append(
                f"  hea.R.{name}(pl.col('x'), *{extra_args}) raised "
                f"{type(e).__name__}: {e}"
            )
            continue
        if not isinstance(result, pl.Expr):
            failures.append(
                f"  hea.R.{name}: expected pl.Expr from pl.Expr input, "
                f"got {type(result).__name__}"
            )
    if failures:
        msg = "Expr dispatch missing or broken for:\n" + "\n".join(failures)
        msg += (
            "\n\nFix by adding ``isinstance(x, pl.Expr): return x.<method>()``"
            " dispatch at the top of the function — or add the name to"
            " _R_EXPR_SKIP with the category that applies."
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Distributions — agreement with known R values
# ---------------------------------------------------------------------------


# --- Live-R bit-exact distribution checks -----------------------------------
# Bit-exactness to R is achievable only on macOS, where hea and R share Apple's
# scalar libm (on BOTH Intel and arm64); on Linux/glibc they drift a few ulp
# (the libm floor — see tests/test_rs_parity.py header, and note R itself is not
# bit-identical across arches because clang fuses nmath's Horners to FMA on arm64
# but not on x86-64). So: bit-for-bit vs the *live* R on this machine when on
# macOS+Rscript (no frozen, arch-locked literals); otherwise a tolerance check vs
# the committed R 4.6.0 value (a ~portable reference, exact only where libm matches).
_R_BITEXACT = sys.platform == "darwin" and have_rscript()


def _assert_r(checks, *, rel_tol=1e-12):
    """``checks``: list of ``(got, r_expr, fallback)`` — bit-exact vs live R on
    macOS, else ``math.isclose`` vs the committed fallback."""
    if _R_BITEXACT:
        ref = r_scalar_values([e for _, e, _ in checks])
        for got, expr, _ in checks:
            assert got == ref[expr], f"{expr}: {got!r} != live R {ref[expr]!r}"
    else:
        for got, expr, fb in checks:
            assert math.isclose(got, fb, rel_tol=rel_tol), f"{expr}: {got!r} !~ R {fb!r}"


def test_dnorm_pnorm_qnorm():
    _assert_r([
        (dnorm(0), "dnorm(0)", 0.3989422804014327),
        (pnorm(1.96), "pnorm(1.96)", 0.97500210485177963),
        (qnorm(0.975), "qnorm(0.975)", 1.9599639845400536),
    ])


def test_pnorm_lower_tail_false():
    # R's pnorm(.., lower.tail=FALSE) uses the upper-tail kernel directly.
    _assert_r([(pnorm(1.96, lower_tail=False),
                "pnorm(1.96, lower.tail=FALSE)", 0.024997895148220428)])


def test_qnorm_lower_tail_false():
    # P(Z > q) = 0.025  →  q = qnorm(0.975); R's lower.tail=FALSE path differs
    # from 1-p by 1 ulp (0.5 - p + 0.5 idiom) — we replicate it exactly.
    _assert_r([(qnorm(0.025, lower_tail=False),
                "qnorm(0.025, lower.tail=FALSE)", 1.9599639845400538)])


def test_t_distribution():
    _assert_r([
        (qt(0.975, df=10), "qt(0.975, df=10)", 2.2281388519862739),
        (pt(2, df=10), "pt(2, df=10)", 0.96330598261462974),
    ])


def test_chisq_distribution():
    _assert_r([
        (qchisq(0.95, df=1), "qchisq(0.95, df=1)", 3.841458820694124),
        (pchisq(3.841458821, df=1), "pchisq(3.841458821, df=1)", 0.9500000000091211),
    ])


def test_f_distribution():
    _assert_r([
        (qf(0.95, 2, 10), "qf(0.95, 2, 10)", 4.1028210151304005),
        (pf(4.102821, 2, 10), "pf(4.102821, 2, 10)", 0.94999999958445847),
    ])


def test_tukey_studentized_range():
    # ptukey/qtukey bit-exact to R via the Python nmath reference (ptukey.c/
    # qtukey.c). The nmath scalar takes (q, rr=nranges, cc=nmeans, df); R's
    # user signature is ptukey(q, nmeans, df, nranges=1). The public ptukey/
    # qtukey add a Rust f64 fast path (≤ few ulp), covered in test_rs_parity.
    _assert_r([
        (float(_nm.ptukey(3.5, 1, 4, 20, True, False)),
         "ptukey(3.5, 4, 20)", 0.90504154945144333),
        (float(_nm.ptukey(3.5, 1, 4, 20, False, False)),
         "ptukey(3.5, 4, 20, lower.tail=FALSE)", 0.094958450548556672),
        (float(_nm.ptukey(3.5, 2, 4, 20, True, False)),
         "ptukey(3.5, 4, 20, nranges=2)", 0.82597563005997021),
        (float(_nm.qtukey(0.95, 1, 4, 20, True, False)),
         "qtukey(0.95, 4, 20)", 3.9582934614503928),
        (float(_nm.qtukey(0.99, 1, 6, 30, True, False)),
         "qtukey(0.99, 6, 30)", 5.2418260490366073),
    ])


def test_noncentral_t_f_chisq():
    # Noncentral t / F / chi-square d/p/q bit-exact to R via the Python nmath
    # reference (pnt/pnf/pnchisq etc.). The public pt/pf/pchisq(ncp!=0) add a
    # Rust f64 fast path (≤ few ulp), covered in test_rs_parity.
    _assert_r([
        (float(_nm.pnt(2, 10, 1, True, False)), "pt(2, 10, ncp=1)", 0.80761156253031108),
        (float(_nm.qnt(0.9, 10, 1, True, False)), "qt(0.9, 10, ncp=1)", 2.5260798970603702),
        (float(_nm.dnt(2, 10, 1, False)), "dt(2, 10, ncp=1)", 0.22542404659006754),
        (float(_nm.pnf(3, 4, 20, 5, True, False)), "pf(3, 4, 20, ncp=5)", 0.70573017245554626),
        (float(_nm.qnf(0.9, 4, 20, 5, True, False)), "qf(0.9, 4, 20, ncp=5)", 4.7522167215545368),
        (float(_nm.pnchisq(10, 5, 3, True, False)), "pchisq(10, 5, ncp=3)", 0.71723684643114338),
        (float(_nm.dnchisq(10, 5, 3, False)), "dchisq(10, 5, ncp=3)", 0.061806315927121186),
        (float(_nm.qnchisq(0.9, 5, 3, True, False)), "qchisq(0.9, 5, ncp=3)", 14.322122198979427),
    ])


def test_noncentral_tukey_hyper_rs_fast_path():
    # The public noncentral / tukey / hyper surface routes through the Rust f64
    # `_rs` fast path when built. It is ≤ a few ulp from R (f64 accumulators vs
    # R's 80-bit LDOUBLE) — verified here to a tight tolerance; the 0-ulp Python
    # reference is asserted above, and _rs-vs-R at scale in test_rs_parity.
    cases = [
        (float(pt(2, 10, ncp=1)), 0.80761156253031108),
        (float(dt(2, 10, ncp=1)), 0.22542404659006754),
        (float(pf(3, 4, 20, ncp=5)), 0.70573017245554626),
        (float(pchisq(10, 5, ncp=3)), 0.71723684643114338),
        (float(dchisq(10, 5, ncp=3)), 0.061806315927121186),
        (float(qchisq(0.9, 5, ncp=3)), 14.322122198979427),
        (float(ptukey(3.5, 4, 20)), 0.90504154945144333),
        (float(qtukey(0.95, 4, 20)), 3.9582934614503928),
        (float(phyper(5, 20, 25, 15)), 0.22989638312676378),
        (float(dhyper(5, 20, 25, 15)), 0.14695170166964935),
    ]
    for got, ref in cases:
        assert math.isclose(got, ref, rel_tol=1e-9), f"{got!r} !~ R {ref!r}"


def test_signrank_wilcox_distributions():
    # Exact Wilcoxon signed-rank / rank-sum d/p/q, bit-exact to R
    # (ported nmath signrank.c / wilcox.c).
    _assert_r([
        (float(psignrank(10, 8)), "psignrank(10, 8)", 0.15625000000000003),
        (float(dsignrank(10, 8)), "dsignrank(10, 8)", 0.03125),
        (float(qsignrank(0.975, 12)), "qsignrank(0.975, 12)", 64.0),
        (float(pwilcox(20, 6, 8)), "pwilcox(20, 6, 8)", 0.33100233100233101),
        (float(dwilcox(20, 6, 8)), "dwilcox(20, 6, 8)", 0.044622044622044624),
        (float(qwilcox(0.1, 10, 10)), "qwilcox(0.1, 10, 10)", 33.0),
    ])


def test_cauchy_logis_lnorm_weibull_geom():
    # Second-half continuous + geometric families. Closed-form (no LDOUBLE
    # series) → the public API (Rust f64 _rs fast path) is itself 0-ulp to R,
    # so assert it directly (unlike the noncentral f64 kernels).
    _assert_r([
        (float(dcauchy(1.5, 0, 2)), "dcauchy(1.5, 0, 2)", 0.10185916357881301),
        (float(pcauchy(1.5, 0, 2)), "pcauchy(1.5, 0, 2)", 0.70483276469913347),
        (float(qcauchy(0.9, 0, 2)), "qcauchy(0.9, 0, 2)", 6.1553670743505089),
        (float(dlogis(1.0)), "dlogis(1)", 0.19661193324148188),
        (float(plogis(1.0)), "plogis(1)", 0.7310585786300049),
        (float(qlogis(0.8)), "qlogis(0.8)", 1.3862943611198908),
        (float(dlnorm(2.0)), "dlnorm(2)", 0.15687401927898109),
        (float(plnorm(2.0)), "plnorm(2)", 0.75589140421441725),
        (float(qlnorm(0.9)), "qlnorm(0.9)", 3.6022244792791591),
        (float(dweibull(1.5, 2, 1.3)), "dweibull(1.5, 2, 1.3)", 0.46884775144958446),
        (float(pweibull(1.5, 2, 1.3)), "pweibull(1.5, 2, 1.3)", 0.73588243335006742),
        (float(qweibull(0.9, 2, 1.3)), "qweibull(0.9, 2, 1.3)", 1.9726552682006904),
        (float(dgeom(3, 0.4)), "dgeom(3, 0.4)", 0.086399999999999991),
        (float(pgeom(3, 0.4)), "pgeom(3, 0.4)", 0.87040000000000006),
        (float(qgeom(0.9, 0.4)), "qgeom(0.9, 0.4)", 4.0),
    ])


def test_nbinom_qhyper():
    # Negative binomial (prob + mu parameterizations) and hypergeometric
    # quantile. f64 discrete kernels → the public _rs path is 0-ulp to R.
    _assert_r([
        (float(dnbinom(3, 5, 0.4)), "dnbinom(3, 5, 0.4)", 0.077414399999999994),
        (float(dnbinom(3, 5, mu=8)), "dnbinom(3, 5, mu=8)", 0.068650105431054348),
        (float(pnbinom(3, 5, 0.4)), "pnbinom(3, 5, 0.4)", 0.17367040000000014),
        (float(pnbinom(3, 5, mu=8)), "pnbinom(3, 5, mu=8)", 0.15077356023716559),
        (float(qnbinom(0.9, 5, 0.4)), "qnbinom(0.9, 5, 0.4)", 13.0),
        (float(qnbinom(0.9, 5, mu=8)), "qnbinom(0.9, 5, mu=8)", 14.0),
        (float(dnbinom(3, 5, 0.4, log=True)),
         "dnbinom(3, 5, 0.4, log=TRUE)", -2.558582469179334),
        (float(pnbinom(10, 5, 0.4, lower_tail=False)),
         "pnbinom(10, 5, 0.4, lower.tail=FALSE)", 0.2172777056501759),
        (float(qhyper(0.9, 20, 25, 15)), "qhyper(0.9, 20, 25, 15)", 9.0),
        (float(qhyper(0.5, 8, 14, 9)), "qhyper(0.5, 8, 14, 9)", 3.0),
    ])


def test_r_generators_cauchy_weibull_geom():
    # r* draws bit-exact to R's set.seed MT stream (rcauchy/rweibull consume one
    # uniform per variate; rgeom interleaves exp_rand + rpois per rgeom.c).
    set_seed(42)
    assert float(rcauchy(3, 1, 2)[0]) == pytest.approx(0.45155182574632846, rel=1e-12)
    set_seed(13)
    assert float(rweibull(3, 1.7, 2.3)[0]) == pytest.approx(1.2236370854093777, rel=1e-12)
    set_seed(101)
    assert rgeom(4, 0.08).astype(int).tolist() == [7, 23, 9, 9]


def test_pbirthday_qbirthday_dmultinom():
    # Closed-form combinatorial densities — pure R ports (birthday.R / distn.R),
    # bit-exact (long-double sum/prod + nmath lgammafn).
    _assert_r([
        (float(pbirthday(23)), "pbirthday(23)", 0.5072972343239854),
        (float(pbirthday(10, 365, 3)),
         "pbirthday(10, 365, 3)", 0.0012248510714326258),
        (float(qbirthday(0.5)), "qbirthday(0.5)", 23.0),
        (float(qbirthday(0.9, 365, 3)), "qbirthday(0.9, 365, 3)", 135.0),
        (float(dmultinom([1, 2, 3], prob=[0.2, 0.3, 0.5])),
         "dmultinom(c(1,2,3), prob=c(0.2,0.3,0.5))", 0.13499999999999993),
        (float(dmultinom([1, 2, 3], prob=[0.2, 0.3, 0.5], log=True)),
         "dmultinom(c(1,2,3), prob=c(0.2,0.3,0.5), log=TRUE)", -2.002480500543708),
    ])


def test_psmirnov_qsmirnov():
    # Two-sample Smirnov CDF (exact recursion + asymptotic) and quantile,
    # bit-exact to R (reuses the ks.test exact kernels; ties via z=).
    _assert_r([
        (float(psmirnov(0.5, (5, 8))), "psmirnov(0.5, c(5,8))", 0.6837606837606837),
        (float(psmirnov(0.4, (5, 8), alternative="greater")),
         "psmirnov(0.4, c(5,8), alternative='greater')", 0.7016317016317015),
        (float(psmirnov(0.5, (5, 8), lower_tail=False)),
         "psmirnov(0.5, c(5,8), lower.tail=FALSE)", 0.3162393162393162),
        (float(psmirnov(0.5, (12, 15), exact=False)),
         "psmirnov(0.5, c(12,15), exact=FALSE)", 0.9286552524988926),
        (float(psmirnov(0.5, (5, 5), z=[1, 1, 2, 3, 3, 4, 5, 6, 7, 7])),
         "psmirnov(0.5, c(5,5), z=c(1,1,2,3,3,4,5,6,7,7))", 0.6428571428571428),
        (float(qsmirnov(0.95, (5, 8))), "qsmirnov(0.95, c(5,8))", 0.75),
    ])


def test_r_generators_nbinom_hyper_rank():
    # r* draws bit-exact to R's set.seed MT stream: rnbinom (prob + mu via
    # rpois∘rgamma), rhyper (H2PE), rsignrank/rwilcox (rank statistics).
    set_seed(42)
    assert rnbinom(6, 5, prob=0.4).astype(int).tolist() == [9, 4, 8, 7, 4, 4]
    set_seed(7)
    assert rnbinom(6, 5, mu=3).astype(int).tolist() == [3, 1, 5, 0, 0, 7]
    set_seed(42)
    assert rhyper(8, 20, 25, 15).astype(int).tolist() == [9, 9, 6, 8, 7, 7, 8, 5]
    set_seed(42)
    assert rsignrank(5, 8).astype(int).tolist() == [25, 20, 25, 27, 21]
    set_seed(42)
    assert rwilcox(5, 6, 8).astype(int).tolist() == [17, 15, 15, 11, 39]


def test_r_multinom_2dtable_wishart_smirnov():
    # Multivariate r* generators bit-exact to R's set.seed stream (rmultinom.c
    # sequential rbinom, rcont.c AS 159, ks.c Smirnov_sim). rWishart's RNG
    # stream is exact; its chol/crossprod carry ≤ few ulp of platform-BLAS.
    set_seed(42)
    assert rmultinom(4, 10, [0.2, 0.3, 0.5]).tolist() == [
        [4, 1, 2, 3], [4, 5, 3, 1], [2, 4, 5, 6]]
    set_seed(7)
    tabs = r2dtable(2, [3, 2], [2, 3])
    assert [t.tolist() for t in tabs] == [[[0, 3], [2, 0]], [[1, 2], [1, 1]]]
    set_seed(101)
    assert rsmirnov(6, (5, 8)).tolist() == pytest.approx(
        [0.25, 0.4, 0.4, 0.2, 0.3, 0.675])
    set_seed(7)
    w = rWishart(1, 5, [[2.0, 0.5], [0.5, 1.0]])[:, :, 0]
    assert w.ravel().tolist() == pytest.approx(
        [26.17012340132881, 10.434953805570906,
         10.434953805570906, 4.8473550148401925])


def test_binom():
    # dbinom(3, 10, 0.5) = C(10,3) / 2^10 = 120/1024 = 0.1171875
    assert float(dbinom(3, 10, 0.5)) == pytest.approx(0.1171875)
    # pbinom(3, 10, 0.5) = sum_{k=0..3} C(10,k)/1024 = (1+10+45+120)/1024
    assert float(pbinom(3, 10, 0.5)) == pytest.approx(176 / 1024)


def test_poisson_with_lambda_keyword():
    # dpois / ppois bit-exact to R (ported dpois saddlepoint, ppois->pgamma).
    _assert_r([
        (float(dpois(2, lambda_=3)), "dpois(2, 3)", 0.22404180765538773),
        (float(ppois(2, lambda_=3)), "ppois(2, 3)", 0.42319008112684348),
    ])


def test_uniform_exp_gamma_beta():
    assert float(punif(0.3)) == pytest.approx(0.3)
    assert float(qexp(0.5)) == pytest.approx(np.log(2), rel=1e-6)
    # pgamma / pbeta bit-exact to R (ported nmath pgamma / toms708 pbeta).
    _assert_r([
        (float(pgamma(1, shape=2, rate=1)), "pgamma(1, shape=2, rate=1)", 0.26424111765711528),
        (float(pbeta(0.5, 2, 5)), "pbeta(0.5, 2, 5)", 0.890625),
    ])


def test_set_seed_reproducible():
    set_seed(42)
    a = rnorm(5)
    set_seed(42)
    b = rnorm(5)
    np.testing.assert_array_equal(a, b)


def test_rnorm_size_and_params():
    set_seed(0)
    out = rnorm(1000, mean=10, sd=2)
    assert len(out) == 1000
    # very loose sanity
    assert abs(np.mean(out) - 10) < 0.5
    assert abs(np.std(out, ddof=1) - 2) < 0.5


def test_rmersenne_family_samplers_match_r():
    """``RMersenneTwister``'s nmath samplers (exp_rand / rgamma / rpois /
    rbinom / rnbinom) are bit-exact vs R's ``set.seed(); r*()`` — the basis for
    byte-exact ``simulate.merMod`` / ``bootMer``. References from R 4.x."""
    from hea.R.rng import RMersenneTwister as MT

    def f(seed, fn, k):
        r = MT(seed)
        return [fn(r) for _ in range(k)]

    np.testing.assert_allclose(
        f(1, lambda r: r.exp_rand(), 5),
        [0.755181833128345, 1.181642779107106, 0.145706726703793,
         0.139795261868498, 0.436068625779175], rtol=1e-12)
    np.testing.assert_allclose(
        f(2, lambda r: r.rgamma(3, 2), 5),
        [2.56593501810152, 2.42084845343624, 2.06431284019708,
         5.40513641980691, 3.00717049677133], rtol=1e-10)
    np.testing.assert_allclose(
        f(3, lambda r: r.rgamma(0.4), 5),
        [0.01631525060736624, 0.12958103428420428, 0.00772927389486502,
         0.00584445661994953, 1.22420510697687734], rtol=1e-9)
    assert [int(x) for x in f(4, lambda r: r.rpois(3.0), 8)] == \
        [3, 0, 2, 2, 4, 2, 4, 5]
    assert [int(x) for x in f(5, lambda r: r.rpois(25.0), 8)] == \
        [20, 22, 25, 33, 21, 25, 20, 25]
    assert [int(x) for x in f(6, lambda r: r.rbinom(10, 0.3), 8)] == \
        [3, 5, 2, 2, 4, 6, 6, 4]
    assert [int(x) for x in f(7, lambda r: r.rbinom(200, 0.4), 8)] == \
        [82, 76, 76, 78, 81, 85, 90, 70]
    assert [int(x) for x in f(8, lambda r: r.rnbinom(2, 5), 8)] == \
        [4, 5, 6, 0, 8, 2, 5, 13]


def test_rmersenne_composed_families_match_r():
    """``RMersenneTwister``'s composed continuous families (rchisq/rt/rf central,
    Cheng's rbeta BB+BC) and weighted ``sample_prob`` (ProbSample[No]Replace +
    Walker alias) are bit-exact vs R's ``set.seed(); r*()`` / ``sample(prob=)``.
    References from R 4.6.0 (Rejection sample.kind)."""
    from hea.R.rng import RMersenneTwister as MT

    def f(seed, fn, k):
        r = MT(seed)
        return [fn(r) for _ in range(k)]

    np.testing.assert_allclose(f(1, lambda r: r.rchisq(3), 8),
        [0.94331456701213001, 5.5437815656796143, 5.3543968318754569,
         2.9152466284968503, 5.0531907965078293, 3.7492110953605073,
         3.3173247857629455, 1.4358542591922681], rtol=1e-9)
    np.testing.assert_allclose(f(1, lambda r: r.rchisq(3, 2.5), 8),
        [1.4007473814073526, 5.3543968318754569, 4.5570610095050101,
         5.6340352822938744, 3.092611107839025, 1.4629708539445847,
         5.7511096054004343, 8.7129370516950448], rtol=1e-9)
    np.testing.assert_allclose(f(1, lambda r: r.rt(4), 8),
        [-0.67291659996486219, -0.5843383715604018, 0.57211571515607695,
         -0.34111699754345642, -0.21805345233165202, 0.60311428389395705,
         -0.41526835800270151, -0.013495013427469081], rtol=1e-9)
    np.testing.assert_allclose(f(1, lambda r: r.rf(4, 7), 8),
        [0.25307567624745586, 1.6113499786337722, 1.5400491076567895,
         1.6052631863861921, 0.58985814242435342, 0.69212578066138286,
         0.23118802715891759, 0.84371345838504663], rtol=1e-9)
    # rbeta BB (min > 1), with the aa/bb swap (rbeta(a,b)+rbeta(b,a)==1).
    np.testing.assert_allclose(f(2, lambda r: r.rbeta(2, 3), 6),
        [0.20153661209123486, 0.44718349735790824, 0.16045907961851755,
         0.3800521727916239, 0.4336392373583321, 0.58685628551895686], rtol=1e-9)
    np.testing.assert_allclose(f(2, lambda r: r.rbeta(3, 2), 6),
        [0.79846338790876514, 0.5528165026420917, 0.83954092038148243,
         0.61994782720837605, 0.56636076264166801, 0.4131437144810432], rtol=1e-9)
    # rbeta BC (min <= 1), incl. the a == 1 edge.
    np.testing.assert_allclose(f(3, lambda r: r.rbeta(0.5, 0.8), 6),
        [0.61473057792667174, 0.21442516685532181, 0.96858430275314145,
         0.25050089115849417, 0.36212582276500171, 0.32241222305695116], rtol=1e-9)
    np.testing.assert_allclose(f(4, lambda r: r.rbeta(1, 3), 6),
        [0.19073474794599599, 0.44489425584764741, 0.070961268480035325,
         0.11254197198550225, 0.017583976693441462, 0.097764348433268394], rtol=1e-9)
    # weighted sample: ProbSampleReplace, ProbSampleNoReplace, and the Walker
    # alias path (n=250, >200 sizeable weights → R_unif_index + unif_rand).
    p = np.array([0.30, 0.11, 0.24, 0.05, 0.19, 0.07])
    assert [int(x) + 1 for x in MT(5).sample_prob(p, 10, replace=True)] == \
        [1, 5, 6, 1, 1, 5, 3, 2, 4, 1]
    assert [int(x) + 1 for x in MT(6).sample_prob(p, 4, replace=False)] == \
        [5, 4, 1, 3]
    assert [int(x) + 1 for x in
            MT(9).sample_prob(np.arange(1, 251, dtype=float), 8, replace=True)] == \
        [187, 248, 152, 249, 227, 30, 232, 232]


def test_public_r_surface_routes_through_mersenne_twister():
    """``set_seed`` + ``runif``/``rnorm``/``sample``/``rpois``/``rgamma``/
    ``rbinom``/``rexp`` now draw from R's bit-exact MT stream (subsystem A),
    not numpy. Proven two ways: (1) ``runif``/``rnorm`` match R's canonical
    ``set.seed(1)`` reference values; (2) every routed function reproduces a
    fresh ``RMersenneTwister(seed)`` draw, and all share ONE advancing stream
    like R's global RNG."""
    from hea.R import (runif, sample, rpois, rgamma, rbinom, rexp,
                       rchisq, rt, rf, rbeta)
    from hea.R.rng import RMersenneTwister as MT

    # (1) R parity — set.seed(1); runif(5) / rnorm(5) (R 4.x reference values).
    set_seed(1)
    np.testing.assert_allclose(
        runif(5),
        [0.2655087, 0.3721239, 0.5728534, 0.9082078, 0.2016819], rtol=1e-6)
    set_seed(1)
    np.testing.assert_allclose(
        rnorm(5),
        [-0.6264538, 0.1836433, -0.8356286, 1.5952808, 0.3295078], rtol=1e-6)

    # (2) routing — each public fn == the same draw off RMersenneTwister(seed).
    def draws(seed, fn, k):
        r = MT(seed)
        return np.array([fn(r) for _ in range(k)])

    set_seed(11)
    np.testing.assert_array_equal(runif(4), MT(11).unif_rand(4))
    set_seed(11)
    np.testing.assert_array_equal(rnorm(4), MT(11).rnorm(4))
    set_seed(11)
    np.testing.assert_array_equal(rpois(6, 3.0),
                                  draws(11, lambda r: r.rpois(3.0), 6))
    set_seed(12)
    np.testing.assert_array_equal(rgamma(5, 2.0, scale=1.5),
                                  draws(12, lambda r: r.rgamma(2.0, scale=1.5), 5))
    set_seed(13)
    np.testing.assert_array_equal(rbinom(7, 20, 0.3),
                                  draws(13, lambda r: r.rbinom(20, 0.3), 7))
    set_seed(14)
    np.testing.assert_array_equal(rexp(5, 2.0),
                                  draws(14, lambda r: r.exp_rand() / 2.0, 5))

    # unweighted sample is R's shrinking-pool walk on the same stream.
    vals = np.arange(1, 11)
    set_seed(2)
    np.testing.assert_array_equal(sample(vals),
                                  vals[MT(2).sample_int(10, 10)])

    # (3) one advancing global stream: interleaved runif then rpois ==
    # sequential draws off a single MT (R's global-RNG semantics).
    set_seed(99)
    u, p = runif(2), rpois(3, 4.0)
    r = MT(99)
    np.testing.assert_array_equal(u, r.unif_rand(2))
    np.testing.assert_array_equal(p, np.array([r.rpois(4.0) for _ in range(3)]))

    # (4) the composed/weighted families also route through the stream — R parity
    # for rt/rbeta/walker-sample, and routing-equivalence for rchisq/rf.
    def mt_draws(seed, fn, k):
        r = MT(seed)
        return np.array([fn(r) for _ in range(k)])

    set_seed(1)
    np.testing.assert_allclose(rt(8, 4),
        [-0.67291659996486219, -0.5843383715604018, 0.57211571515607695,
         -0.34111699754345642, -0.21805345233165202, 0.60311428389395705,
         -0.41526835800270151, -0.013495013427469081], rtol=1e-9)
    set_seed(2)
    np.testing.assert_allclose(rbeta(6, 2, 3),
        [0.20153661209123486, 0.44718349735790824, 0.16045907961851755,
         0.3800521727916239, 0.4336392373583321, 0.58685628551895686], rtol=1e-9)
    set_seed(9)
    np.testing.assert_array_equal(
        sample(250, 8, replace=True, prob=list(range(1, 251))),
        [187, 248, 152, 249, 227, 30, 232, 232])
    set_seed(7)
    np.testing.assert_array_equal(rchisq(5, 3),
                                  mt_draws(7, lambda r: r.rchisq(3), 5))
    set_seed(8)
    np.testing.assert_array_equal(rf(5, 4, 7),
                                  mt_draws(8, lambda r: r.rf(4, 7), 5))

    # (5) set_seed no longer touches numpy's global RNG — fully decoupled.
    set_seed(123)
    a = np.random.random()
    set_seed(123)
    b = np.random.random()
    assert a != b


def test_rgenerator_family_rd_matches_r():
    """``RGenerator`` (numpy-Generator facade over ``RMersenneTwister``) drives
    ``family.py``'s ``rd`` hooks bit-exactly vs R's ``set.seed(k); family$rd(...)``
    — the basis for R-exact ``qq.gam(rep>0)``. References from R 4.6.0 / mgcv.
    Covers all ten built-in families: gaussian, Gamma, poisson, binomial,
    gaulss, shash, negbin, scat, inverse.gaussian (mgcv ``rig``) and tweedie
    (per-jump ``rTweedie``)."""
    from hea.R.rng import RGenerator
    from hea import family as F

    mu = np.array([0.5, 1.5, 3.0, 6.0, 10.0])
    wt = np.array([1, 1, 2, 1, 3.0])
    mu2 = np.column_stack([[0.5, 1.5, 3, 6, 10], [0.8, 1.2, 1.5, 0.9, 2.0]])
    mu4 = np.column_stack([[0.5, 1.5, 3, 6, 10], [-0.2, 0.1, 0.3, 0, 0.2],
                           [0.1, -0.1, 0.2, 0, 0.3], [-0.3, 0, 0.1, 0.2, -0.1]])

    def chk(got, ref):
        np.testing.assert_allclose(np.asarray(got, float), ref,
                                   rtol=1e-9, atol=1e-12)

    chk(F.Poisson().rd(RGenerator(1), mu, wt, 1.0), [0, 1, 3, 9, 7])
    chk(F.Gamma().rd(RGenerator(1), mu, wt, 0.7),
        [0.148055784081653, 2.78469406100319, 5.37491660494231,
         5.75866952765452, 16.3564500353662])
    chk(F.Gaussian().rd(RGenerator(1), mu, wt, 2.0),
        [-0.385939475352115, 1.75971087975415, 2.16437138758995,
         8.2560677461767, 10.2690419690764])
    chk(F.Binomial().rd(RGenerator(1), np.array([.2, .5, .8, .3, .6]),
                        np.array([1, 2, 3, 1, 4.]), 1.0),
        [0, 0.5, 0.666666666666667, 1, 0.75])
    chk(F.gaulss().rd(RGenerator(1), mu2, wt, 1.0),
        [-0.283067263427916, 1.6530361035184, 2.60608089440757,
         7.77253422459755, 10.0951207003788])
    chk(F.shash().rd(RGenerator(1), mu4, wt, 1.0),
        [0.0675404284664644, 1.02120881215142, 3.52687189173171,
         7.24982044353249, 9.4132951645412])
    chk(F.nb(theta=2.0).rd(RGenerator(1), mu, wt, 1.0), [1, 1, 1, 3, 6])
    chk(F.Scat(theta=(4.0, 1.5)).rd(RGenerator(1), mu, wt, 1.0),
        [-0.509374899947293, 0.623492442659397, 3.85817357273412,
         5.48832450368482, 9.67291982150252])

    chk(F.InverseGaussian().rd(RGenerator(1), mu, wt, 0.5),
        [0.366005266522837, 1.2796574965637, 1.12218739016413,
         0.62960207413193, 20.5664902471926])
    chk(F.Tweedie(p=1.5).rd(RGenerator(1), mu, wt, 2.0),
        [0, 2.21006742517074, 3.56580988134542, 6.63216281011408,
         2.46511332727586])


def test_rmvn_matches_r():
    """``RMersenneTwister.rmvn`` (port of ``mgcv::rmvn``) reproduces R's
    ``set.seed(k); mgcv::rmvn(n, mu, V)`` draws. The pivoted-Cholesky root
    (``mroot``, via LAPACK ``dpstrf``) and the column-major ``rnorm(p*n)`` draw
    order are bit-exact; only the trailing ``R %*% Z`` GEMM is BLAS-bound, so
    the vector-``mu`` ``n==1`` branch is bit-identical and ``n>1`` matches to
    machine precision. Reference: ``set.seed(101)`` in R 4.x / mgcv 1.9-4.

    This is the bit-exact guarantee behind itsadug's simultaneous-CI path
    (:meth:`hea.models.gam.gam.get_difference`): the MVN *draws* now come off
    R's MT stream rather than numpy. The downstream ``crit`` is only
    Monte-Carlo-close to R because hea's smooth basis differs from mgcv's, which
    makes the (basis-dependent) realized draw differ — see the itsadug
    get_difference test."""
    from hea.R.rng import RGenerator, RMersenneTwister

    V = np.array([[2.0, 0.3, -0.4],
                  [0.3, 1.5, 0.2],
                  [-0.4, 0.2, 1.0]])
    mu = np.array([10.0, -5.0, 2.5])

    # vector-mu, n=2 -> (2, 3); zero mean
    A_R = np.array([[-0.46108522671538527, 0.59723538372992691, -0.4195265282782813],
                    [0.30315005420217234, 0.42033284459609094, 1.1035833980527343]])
    A = RMersenneTwister(101).rmvn(2, np.zeros(3), V)
    assert A.shape == (2, 3)
    np.testing.assert_allclose(A, A_R, rtol=0, atol=1e-13)

    # vector-mu, n=1 -> length-3 vector (R's as.numeric); bit-identical
    b_R = np.array([9.5389147732846151, -4.4027646162700727, 2.0804734717217186])
    b = RMersenneTwister(101).rmvn(1, mu, V)
    assert b.shape == (3,)
    assert np.array_equal(b, b_R), "n=1 vector-mu rmvn must be bit-identical to R"

    # RGenerator.multivariate_normal facade maps to the same draws.
    sim = RGenerator(101).multivariate_normal(np.zeros(3), V, size=2)
    np.testing.assert_allclose(sim, A_R, rtol=0, atol=1e-13)
    v = RGenerator(101).multivariate_normal(mu, V)  # size=None -> (p,)
    assert v.shape == (3,) and np.array_equal(v, b_R)


# ---------------------------------------------------------------------------
# Model generics
# ---------------------------------------------------------------------------


from hea.R import (  # noqa: E402  — grouped with the model-generic tests
    AIC as R_AIC, BIC as R_BIC,
    coef, coefficients, confint, deviance,
    df_residual, fitted, fitted_values, fixef,
    formula as R_formula,
    logLik, model_frame, model_matrix, nobs,
    predict as R_predict, ranef, resid,
    residuals as R_residuals, vcov,
)


@pytest.fixture(scope="module")
def gala():
    return hea.data("gala", package="faraway")


@pytest.fixture(scope="module")
def m_lm(gala):
    return hea.models.lm("Species ~ Area + Elevation", gala)


@pytest.fixture(scope="module")
def m_glm(gala):
    return hea.models.glm("Species ~ Area + Elevation", gala, family=hea.family.poisson())


@pytest.fixture(scope="module")
def m_gam():
    mt = hea.data("mtcars", package="R")
    return hea.models.gam("mpg ~ s(wt) + s(hp)", mt)


@pytest.fixture(scope="module")
def m_lme():
    sleep = hea.data("sleepstudy", package="lme4")
    return hea.models.gmm("Reaction ~ Days + (Days|Subject)", sleep)


# ---- coef / coefficients / fixef ------------------------------------


def test_coef_returns_named_vector(m_lm):
    c = coef(m_lm)
    from hea.R import NamedVector
    assert isinstance(c, NamedVector)
    assert set(c.names) == {"(Intercept)", "Area", "Elevation"}
    # Name and 0-based positional indexing both work.
    assert c["Area"] == c[1]["Area"]
    assert all(isinstance(v, float) for v in c.values.tolist())


def test_coefficients_alias(m_lm):
    a, b = coefficients(m_lm), coef(m_lm)
    assert a.names == b.names
    assert (a.values == b.values).all()


def test_coef_works_on_glm_gam_lme(m_glm, m_gam, m_lme):
    assert "(Intercept)" in coef(m_glm)
    # gam: intercept + 9 wt basis + 9 hp basis
    assert "(Intercept)" in coef(m_gam)
    # gmm: lme4 coef.merMod — a per-group dict (fixef + matching ranef BLUP),
    # keyed by grouping factor; use fixef() for the fixed effects alone.
    c = coef(m_lme)
    assert isinstance(c, dict) and set(c) == {"Subject"}
    sub = c["Subject"]
    assert "(Intercept)" in sub.columns and "Days" in sub.columns


def test_fixef_equals_coef_for_non_mixed(m_lm):
    """For a non-mixed model fixef and coef coincide (both the fixed-effect
    NamedVector). For a gmm they differ — coef is the per-group
    coef.merMod dict; fixef is the fixed effects only."""
    a, b = fixef(m_lm), coef(m_lm)
    assert a.names == b.names
    assert (a.values == b.values).all()


def test_fixef_is_fixed_effects_only(m_lme):
    """gmm fixef() = fixed effects β̂ as a NamedVector — not the per-group
    coef.merMod dict."""
    from hea.R import NamedVector
    f = fixef(m_lme)
    assert isinstance(f, NamedVector)
    assert set(f.names) == {"(Intercept)", "Days"}


def test_ranef_returns_random_effects(m_lme):
    re = ranef(m_lme)
    assert re is not None  # actual structure left to the model


def test_ranef_raises_for_non_mixed(m_lm):
    with pytest.raises(TypeError, match="random effects"):
        ranef(m_lm)


# ---- residuals / fitted / predict -----------------------------------


def test_resid_shape_and_alias(m_lm):
    r = resid(m_lm)
    assert isinstance(r, np.ndarray)
    assert r.shape == (30,)
    np.testing.assert_array_equal(R_residuals(m_lm), r)


def test_resid_type_dispatch_glm(m_glm):
    """For glm/gam, ``type=`` dispatches via ``residuals_of()``."""
    dev = resid(m_glm)  # default = deviance
    pearson = resid(m_glm, type="pearson")
    response = resid(m_glm, type="response")
    assert dev.shape == pearson.shape == response.shape == (30,)
    # response residuals are y - mu, easy to verify magnitude differs
    assert not np.allclose(dev, response)


def test_resid_type_invalid_for_lm(m_lm):
    # lm supports R's residuals.lm types (response/working/pearson/deviance);
    # an unknown type still raises.
    with pytest.raises(ValueError, match="not supported"):
        resid(m_lm, type="garbage")


def test_fitted_shape_matches_resid(m_lm, m_glm, m_gam, m_lme):
    for m in (m_lm, m_glm, m_gam, m_lme):
        f = fitted(m)
        r = resid(m)
        assert isinstance(f, np.ndarray)
        assert f.shape == r.shape


def test_fitted_values_alias(m_glm):
    np.testing.assert_array_equal(fitted_values(m_glm), fitted(m_glm))


def test_predict_dispatches_to_method(m_lm):
    out = R_predict(m_lm)
    # lm.predict() returns a polars DataFrame with "fit" column
    assert isinstance(out, pl.DataFrame)
    np.testing.assert_array_almost_equal(
        out["fit"].to_numpy(), fitted(m_lm)
    )


# ---- confint --------------------------------------------------------


def test_confint_default_level_returns_cached(m_lm):
    out = confint(m_lm)
    assert isinstance(out, pl.DataFrame)
    assert out.shape[0] == 3


def test_confint_custom_level_lm_recomputes(m_lm):
    """``level=0.99`` is wider than ``level=0.95`` for lm."""
    ci_95 = confint(m_lm, level=0.95)
    ci_99 = confint(m_lm, level=0.99)
    # both have shape (3, 3): coef, low, high (column names differ)
    assert ci_95.shape == ci_99.shape
    # 99% CI is strictly wider than 95% CI
    lo95 = ci_95[ci_95.columns[1]].to_numpy()
    hi95 = ci_95[ci_95.columns[2]].to_numpy()
    lo99 = ci_99[ci_99.columns[1]].to_numpy()
    hi99 = ci_99[ci_99.columns[2]].to_numpy()
    assert np.all(lo99 < lo95)
    assert np.all(hi99 > hi95)


def test_confint_dispatches_to_profile_object():
    """``confint(profile(fm))`` defers to the Profile's own ``.confint``,
    mirroring R's S3 ``confint.profile`` dispatch — the
    ``lme4::profile`` workflow Bates uses in the gmm book.
    """
    from hea.models import gmm
    from hea import data
    dye = data("Dyestuff")
    fm = gmm("Yield ~ 1 + (1 | Batch)", dye, REML=False)
    pr = fm.profile()
    out = confint(pr)
    assert isinstance(out, pl.DataFrame)
    # One row per profiled parameter — for the random-intercept Dyestuff
    # fit that's ``.sig01``, ``.sigma``, ``(Intercept)``.
    assert set(out["parameter"].to_list()) == {".sig01", ".sigma", "(Intercept)"}
    # 99% CI strictly wider than 95% on the (Intercept) row.
    ci99 = confint(pr, level=0.99)
    icpt95 = out.filter(pl.col("parameter") == "(Intercept)")
    icpt99 = ci99.filter(pl.col("parameter") == "(Intercept)")
    assert icpt99[icpt99.columns[1]][0] < icpt95[icpt95.columns[1]][0]
    assert icpt99[icpt99.columns[2]][0] > icpt95[icpt95.columns[2]][0]


# ---- vcov -----------------------------------------------------------


def test_vcov_shape_lm_glm(m_lm, m_glm):
    assert vcov(m_lm).shape == (3, 3)
    assert vcov(m_glm).shape == (3, 3)


def test_vcov_gam_uses_Vp(m_gam):
    """gam's vcov is the Bayesian posterior ``Vp``."""
    V = vcov(m_gam)
    assert V.shape == m_gam.Vp.shape
    np.testing.assert_array_equal(V, m_gam.Vp)


def test_vcov_lme_returns_dataframe(m_lme):
    """gmm stores ``vcov_beta`` as a DataFrame with named cols."""
    V = vcov(m_lme)
    assert isinstance(V, pl.DataFrame)
    assert V.shape == (2, 2)


# ---- scalars: logLik / deviance / nobs / df_residual ----------------


def test_logLik_matches_loglike(m_lm, m_glm, m_gam):
    for m in (m_lm, m_glm, m_gam):
        assert logLik(m) == pytest.approx(m.loglike)


def test_logLik_lme_REML_uses_minus_half_criterion(m_lme):
    """R's ``logLik.lmerMod(REML=TRUE) = -REML_criterion / 2``."""
    expected = -m_lme.REML_criterion / 2.0
    assert logLik(m_lme) == pytest.approx(expected)


def test_deviance_glm_gam(m_glm, m_gam):
    assert deviance(m_glm) == pytest.approx(m_glm.deviance)
    assert deviance(m_gam) == pytest.approx(m_gam.deviance)


def test_deviance_lm_falls_back_to_rss(m_lm):
    """``deviance.lm = sum(resid^2) = rss``."""
    assert deviance(m_lm) == pytest.approx(m_lm.rss)
    np.testing.assert_allclose(
        deviance(m_lm), float((resid(m_lm) ** 2).sum())
    )


def test_nobs(m_lm, m_glm, m_gam, m_lme):
    assert nobs(m_lm) == 30
    assert nobs(m_glm) == 30
    assert nobs(m_gam) == 32
    assert nobs(m_lme) == 180


def test_df_residual_lm(m_lm):
    # n=30, p=3 (intercept + Area + Elevation); df_residuals = 30 - 3 = 27
    assert df_residual(m_lm) == 27


def test_df_residual_raises_for_reml_lme(m_lme):
    """REML gmm fit has no defined residual df; we raise."""
    with pytest.raises(TypeError, match="residual df"):
        df_residual(m_lme)


# ---- formula / model_matrix / model_frame ---------------------------


def test_formula_returns_string(m_lm):
    assert R_formula(m_lm) == "Species ~ Area + Elevation"


def test_model_matrix_returns_design(m_lm):
    X = model_matrix(m_lm)
    assert isinstance(X, pl.DataFrame)
    assert X.columns == ["(Intercept)", "Area", "Elevation"]
    assert X.height == 30


def test_model_frame_returns_data(m_lm, gala):
    """``model.frame()`` returns the original data passed at fit time."""
    assert model_frame(m_lm) is gala


# ---- AIC / BIC: scalar vs comparison table --------------------------


def test_AIC_single_model_returns_scalar(m_lm):
    assert R_AIC(m_lm) == pytest.approx(m_lm.AIC)
    assert isinstance(R_AIC(m_lm), float)


def test_BIC_single_model_returns_scalar(m_lm):
    assert R_BIC(m_lm) == pytest.approx(m_lm.BIC)
    assert isinstance(R_BIC(m_lm), float)


def test_AIC_multiple_models_returns_table(gala):
    m1 = hea.models.lm("Species ~ Area", gala)
    m2 = hea.models.lm("Species ~ Area + Elevation", gala)
    out = R_AIC(m1, m2)
    assert isinstance(out, pl.DataFrame)
    assert out.height == 2
    assert "df" in out.columns
    assert "AIC" in out.columns
    # row labels should recover the caller's variable names
    label_col = out[""]
    assert label_col.to_list() == ["m1", "m2"]


def test_AIC_no_args_raises():
    with pytest.raises(TypeError, match="at least one"):
        R_AIC()


def test_R_AIC_does_not_print_or_return_none(m_lm, capsys):
    """``hea.R.AIC`` must always return; never call ``print()``."""
    out = R_AIC(m_lm)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert out is not None


# ---------------------------------------------------------------------------
# Regression diagnostics
# ---------------------------------------------------------------------------


from hea.R import (  # noqa: E402  — grouped with the diagnostic tests
    cooks_distance, dfbetas, dffits, hatvalues, influence,
    rstandard, rstudent,
)


def test_hatvalues_sum_to_p_full_rank(m_lm):
    """For an unweighted full-rank lm, ``sum(h_ii) == p``."""
    assert hatvalues(m_lm).sum() == pytest.approx(m_lm.p)


def test_hatvalues_in_unit_interval(m_lm, m_glm, m_gam):
    for m in (m_lm, m_glm, m_gam):
        h = hatvalues(m)
        assert (h >= -1e-9).all()
        assert (h <= 1 + 1e-9).all()


def test_rstandard_matches_cached_attribute(m_lm, m_glm):
    np.testing.assert_array_equal(
        rstandard(m_lm), m_lm.std_residuals
    )
    np.testing.assert_array_equal(
        rstandard(m_glm), m_glm.std_dev_residuals
    )


def test_rstandard_pearson_dispatch(m_glm):
    np.testing.assert_array_equal(
        rstandard(m_glm, type="pearson"), m_glm.std_pearson_residuals
    )


def test_rstandard_invalid_type_raises(m_glm):
    with pytest.raises(ValueError, match="not recognized"):
        rstandard(m_glm, type="bogus")


def test_rstandard_algebraic_identity(m_lm):
    """``rstandard_i = e_i / (σ · √(1 − h_i))`` (unweighted lm)."""
    e = m_lm.residuals.to_series().to_numpy()
    h = hatvalues(m_lm)
    expected = e / (m_lm.sigma * np.sqrt(1 - h))
    np.testing.assert_allclose(rstandard(m_lm), expected, rtol=1e-10)


def test_rstudent_closed_form_matches_loo_refit(m_lm, gala):
    """Spot-check ``rstudent`` by refitting lm without observation 0.

    R's identity: ``rstudent_i = e_i / (σ_(-i) · √(1 − h_i))``.
    Refit ``lm`` with row 0 dropped, recompute σ from the new fit, and
    verify the rstudent value at i=0 lines up with that identity.
    """
    rs_full = rstudent(m_lm)
    h = hatvalues(m_lm)
    e0 = m_lm.residuals.to_series().to_numpy()[0]
    h0 = h[0]

    # Refit without row 0
    gala_drop0 = gala.slice(1)  # drop first row
    m_drop0 = hea.models.lm("Species ~ Area + Elevation", gala_drop0)
    sigma_loo_0 = m_drop0.sigma  # σ from the leave-one-out fit

    expected_rs0 = e0 / (sigma_loo_0 * np.sqrt(1 - h0))
    assert rs_full[0] == pytest.approx(expected_rs0, rel=1e-8)


def test_rstudent_glm_returns_array(m_glm):
    """Used to raise NotImplementedError; now glm uses Williams' likelihood
    residual formula (matches ``rstudent.glm`` in R)."""
    out = rstudent(m_glm)
    assert isinstance(out, np.ndarray)
    assert out.shape == (30,)


def test_cooks_distance_matches_unified_formula_lm(m_lm):
    """``D_i = r_std_i^2 · h_i / ((1 − h_i) · p)`` for lm."""
    h = hatvalues(m_lm)
    r = rstandard(m_lm)
    expected = r ** 2 * h / ((1 - h) * m_lm.p)
    np.testing.assert_allclose(cooks_distance(m_lm), expected, rtol=1e-12)


def test_cooks_distance_glm_uses_pearson_and_sum_hat(m_glm):
    """``cooks.distance.glm = std_pearson^2 · h / ((1−h) · sum(h))``."""
    h = hatvalues(m_glm)
    rp = rstandard(m_glm, type="pearson")
    expected = rp ** 2 * h / ((1 - h) * h.sum())
    np.testing.assert_allclose(cooks_distance(m_glm), expected, rtol=1e-12)


def test_dffits_matches_rstudent_x_leverage_term(m_lm):
    """``DFFITS_i = r_i^* · √(h_i / (1 − h_i))``."""
    expected = rstudent(m_lm) * np.sqrt(hatvalues(m_lm) / (1 - hatvalues(m_lm)))
    np.testing.assert_allclose(dffits(m_lm), expected, rtol=1e-12)


def test_dfbetas_shape_and_columns(m_lm):
    out = dfbetas(m_lm)
    assert isinstance(out, pl.DataFrame)
    assert out.shape == (30, 3)
    assert out.columns == ["(Intercept)", "Area", "Elevation"]


def test_dfbetas_matches_loo_refit_first_obs(m_lm, gala):
    """Check ``dfbetas[0, j]`` against an actual leave-one-out refit.

    R-faithful formula: ``dfbetas_ij = (β̂_j − β̂_(-i)_j) / (σ_(-i) · √diag(XtXinv)_j)``.
    Refit dropping row 0 and verify each coefficient agrees.
    """
    out = dfbetas(m_lm).row(0)  # dfbetas for observation 0 across all coefs

    gala_drop0 = gala.slice(1)
    m_drop0 = hea.models.lm("Species ~ Area + Elevation", gala_drop0)
    bhat_full = coef(m_lm).values
    bhat_drop = coef(m_drop0).values
    delta = bhat_full - bhat_drop
    sigma_loo = m_drop0.sigma
    sd_j = np.sqrt(np.diag(m_lm.XtXinv))
    expected = delta / (sigma_loo * sd_j)

    np.testing.assert_allclose(np.array(out), expected, rtol=1e-8)


def test_influence_returns_dict_with_four_keys(m_lm):
    infl = influence(m_lm)
    assert set(infl.keys()) == {"hat", "sigma", "coefficients", "residuals"}
    assert len(infl["hat"]) == 30
    assert len(infl["sigma"]) == 30
    assert len(infl["residuals"]) == 30
    assert isinstance(infl["coefficients"], pl.DataFrame)
    assert infl["coefficients"].shape == (30, 3)


def test_influence_hat_matches_hatvalues(m_lm):
    np.testing.assert_array_equal(influence(m_lm)["hat"], hatvalues(m_lm))


def test_influence_sigma_at_obs0_matches_loo_refit(m_lm, gala):
    """``influence(m)['sigma'][0]`` should equal ``σ`` from the row-0-dropped fit."""
    sigma_full = influence(m_lm)["sigma"]
    m_drop0 = hea.models.lm("Species ~ Area + Elevation", gala.slice(1))
    assert sigma_full[0] == pytest.approx(m_drop0.sigma, rel=1e-8)


def test_dffits_glm_returns_array(m_glm):
    """glm uses ``p_i · √h_i / (σ_(-i) · (1−h_i))`` (R's ``dffits``)."""
    out = dffits(m_glm)
    assert out.shape == (30,)


def test_influence_glm_returns_dict(m_glm):
    infl = influence(m_glm)
    assert set(infl.keys()) == {"hat", "sigma", "coefficients", "residuals"}


def test_weighted_lm_diagnostics_match_loo_refit(gala):
    """Closed-form weighted-lm diagnostics should match an actual
    leave-one-out refit to numerical precision."""
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, gala.height)
    m_w = hea.models.lm("Species ~ Area + Elevation", gala, weights=w)

    # Refit dropping observation 0
    m_drop0 = hea.models.lm(
        "Species ~ Area + Elevation", gala.slice(1), weights=w[1:]
    )
    b_full = coef(m_w).values
    b_drop = coef(m_drop0).values
    delta = b_full - b_drop

    # σ_(-0) from the refit's own weighted RSS
    e_drop = m_drop0.residuals.to_series().to_numpy()
    weighted_rss_drop = float(np.sum(w[1:] * e_drop * e_drop))
    sigma_loo_0 = np.sqrt(weighted_rss_drop / (m_w.n - m_w.p - 1))

    XtXinv = np.asarray(m_w.XtXinv)
    sd_j = np.sqrt(np.diag(XtXinv))
    expected_dfbetas_0 = delta / (sigma_loo_0 * sd_j)

    np.testing.assert_allclose(
        np.array(dfbetas(m_w).row(0)), expected_dfbetas_0, atol=1e-10
    )
    # influence(m)['sigma'][0] should equal that LOO σ exactly.
    assert influence(m_w)["sigma"][0] == pytest.approx(sigma_loo_0, rel=1e-12)


def test_weighted_lm_rstudent_dffits_consistent(gala):
    """``DFFITS_i = rstudent_i · √(h_i / (1 - h_i))`` must hold for weighted lm."""
    rng = np.random.default_rng(1)
    w = rng.uniform(0.5, 2.0, gala.height)
    m_w = hea.models.lm("Species ~ Area + Elevation", gala, weights=w)
    h = hatvalues(m_w)
    expected = rstudent(m_w) * np.sqrt(h / (1 - h))
    np.testing.assert_allclose(dffits(m_w), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Hypothesis tests (the new batch + verifying the consolidation kept the rest)
# ---------------------------------------------------------------------------


from hea.R import (  # noqa: E402  — grouped with the test-batch tests
    HTest,
    bartlett_test, binom_test, chisq_test, cor_test, fisher_test,
    friedman_test, ks_test,
    mcnemar_test, prop_test, shapiro_test,
    t_test, var_test, wilcox_test,
)


def test_consolidation_preserved_existing_tests():
    """The existing stats.py functions still work after the move."""
    out = t_test([1.0, 2.0, 3.0, 4.0, 5.0], mu=3.0)
    assert isinstance(out, HTest)
    assert out.method == "One Sample t-test"
    assert out.p_value == pytest.approx(1.0)


def test_chisq_test_still_works():
    out = chisq_test([10, 10, 10, 10])
    assert isinstance(out, HTest)
    assert out.statistic["X-squared"] == pytest.approx(0.0)


# ---- fisher_test ----------------------------------------------------


def test_fisher_test_2x2_bit_exact_vs_r():
    """Fisher exact 2×2: p-value, conditional-MLE odds ratio, and CI all
    bit-exact to R (ported dhyper / phyper / uniroot). Note R's odds ratio is
    the conditional MLE, not scipy's sample ad/bc."""
    tbl = np.array([[8, 2], [1, 5]])
    res = fisher_test(tbl)
    assert res.method == "Fisher's Exact Test for Count Data"
    assert res.null_value == 1.0
    _assert_r([
        (res.p_value, "fisher.test(matrix(c(8,1,2,5),2,2))$p.value",
         0.034965034965034968),
        (res.estimate["odds ratio"],
         "fisher.test(matrix(c(8,1,2,5),2,2))$estimate", 15.469687462886908),
        (res.conf_int[0], "fisher.test(matrix(c(8,1,2,5),2,2))$conf.int[1]",
         1.008849380396617),
        (res.conf_int[1], "fisher.test(matrix(c(8,1,2,5),2,2))$conf.int[2]",
         1049.7914461317548),
    ])
    res_g = fisher_test(tbl, alternative="greater")
    _assert_r([(res_g.p_value,
                'fisher.test(matrix(c(8,1,2,5),2,2), alternative="greater")$p.value',
                0.024475524475524479)])


def test_fisher_test_from_two_vectors():
    """Passing parallel vectors should produce the same call as a 2x2 table."""
    x = ["a", "a", "a", "a", "b", "b"]
    y = ["x", "x", "x", "y", "x", "y"]
    res = fisher_test(x, y)
    assert isinstance(res, HTest)
    assert "odds ratio" in res.estimate


def test_fisher_test_exact_rxc_bit_exact_vs_r():
    """r×c Fisher **exact** p-value via the ported FEXACT network algorithm
    (src/fexact.c) — bit-exact to R's fisher.test(x)$p.value across shapes,
    sparsity, transposes, and zero cells."""
    r = fisher_test(np.array([[3, 5, 2], [7, 2, 8], [4, 6, 1]]))
    assert r.method == "Fisher's Exact Test for Count Data"
    assert not r.statistic and not r.parameter        # p-value-only, df = NA
    _assert_r([
        (r.p_value,
         "fisher.test(matrix(c(3,5,2,7,2,8,4,6,1),3,3,byrow=TRUE))$p.value",
         0.07481072655669109),
        (fisher_test(np.array([[1, 9, 3], [8, 2, 4]])).p_value,
         "fisher.test(matrix(c(1,9,3,8,2,4),2,3,byrow=TRUE))$p.value",
         0.0068231106325061432),
        (fisher_test(np.array([[1, 8], [5, 3], [10, 1]])).p_value,
         "fisher.test(matrix(c(1,8,5,3,10,1),3,2,byrow=TRUE))$p.value",
         0.0010927377463923423),
        (fisher_test(np.array([[10, 2, 3], [1, 8, 4],
                               [2, 3, 9], [5, 1, 2]])).p_value,
         "fisher.test(matrix(c(10,2,3,1,8,4,2,3,9,5,1,2),4,3,byrow=TRUE))$p.value",
         0.0016923070283238997),
        (fisher_test(np.array([[1, 8, 5, 4, 4, 2, 2], [5, 3, 3, 4, 3, 1, 0],
                               [10, 1, 4, 0, 0, 0, 0]])).p_value,
         "fisher.test(matrix(c(1,8,5,4,4,2,2,5,3,3,4,3,1,0,"
         "10,1,4,0,0,0,0),3,7,byrow=TRUE))$p.value",
         0.0035599802033163706),
        (fisher_test(np.array([[0, 3, 2], [5, 1, 4], [2, 2, 6]])).p_value,
         "fisher.test(matrix(c(0,3,2,5,1,4,2,2,6),3,3,byrow=TRUE))$p.value",
         0.17749332374450155),
        (fisher_test(np.array([[1, 2, 1, 0], [3, 3, 6, 1], [10, 10, 14, 9],
                               [6, 7, 12, 11]])).p_value,
         "fisher.test(matrix(c(1,2,1,0,3,3,6,1,10,10,14,9,6,7,12,11),"
         "4,4,byrow=TRUE))$p.value",
         0.78268493896653246),
    ])
    # transpose invariance (the algorithm sorts/swaps margins internally)
    a = fisher_test(np.array([[1, 9, 3], [8, 2, 4]])).p_value
    b = fisher_test(np.array([[1, 8], [9, 2], [3, 4]])).p_value
    assert a == b


def test_fisher_test_exact_hybrid_vs_r():
    """r×c Fisher hybrid (asymptotic-χ²) p-value — bit-exact to R's
    fisher.test(x, hybrid=TRUE)."""
    m = np.array([[3, 5, 2], [7, 2, 8], [4, 6, 1]])
    r = fisher_test(m, hybrid=True)
    assert "hybrid using asym.chisq." in r.method
    _assert_r([
        (r.p_value,
         "fisher.test(matrix(c(3,5,2,7,2,8,4,6,1),3,3,byrow=TRUE),"
         "hybrid=TRUE)$p.value",
         0.07481072655669109),
    ])


def test_fisher_test_exact_rejects_bad_input():
    # < 2 rows/cols and negative entries are rejected as in R.
    with pytest.raises(ValueError, match="at least 2 rows"):
        fisher_test(np.array([[1, 2, 3]]))
    with pytest.raises(ValueError, match="nonnegative"):
        fisher_test(np.array([[1, -2, 3], [4, 5, 6]]))


def test_chisq_test_simulate_p_value():
    # Monte-Carlo chisq.test bit-exact to R's set.seed stream: table path via
    # rcont2 (chisq_sim), goodness-of-fit via weighted sample.int.
    m = np.array([[10, 20, 30], [15, 25, 10], [5, 12, 18]])
    set_seed(42)
    r = chisq_test(m, simulate_p_value=True, B=2000)
    assert r.statistic["X-squared"] == pytest.approx(13.124269005847953)
    assert r.p_value == pytest.approx(0.00849575212393803)
    assert "df" not in r.parameter  # df = NA for the MC test
    set_seed(7)
    r2 = chisq_test([12, 20, 8, 15], p=[0.25, 0.25, 0.25, 0.25],
                    simulate_p_value=True, B=1000)
    assert r2.statistic["X-squared"] == pytest.approx(5.581818181818182)
    assert r2.p_value == pytest.approx(0.13886113886113885)


def test_fisher_test_simulate_p_value():
    # r×c fisher.test via Monte-Carlo (Fisher_sim / rcont2), bit-exact to R.
    m = np.array([[3, 5, 2], [7, 2, 8], [4, 6, 1]])
    set_seed(42)
    r = fisher_test(m, simulate_p_value=True, B=2000)
    assert r.p_value == pytest.approx(0.06896551724137931)
    m2 = np.array([[10, 2, 3], [1, 8, 4], [2, 3, 9], [5, 1, 2]])
    set_seed(99)
    r2 = fisher_test(m2, simulate_p_value=True, B=5000)
    assert r2.p_value == pytest.approx(0.0023995200959808036)


# ---- prop_test ------------------------------------------------------


def test_prop_test_one_sample_known_value():
    """1-sample prop.test of x=5, n=10, p=0.5 → X²=0 (or 0.1 with correction)."""
    res = prop_test(5, 10, p=0.5, correct=False)
    assert isinstance(res, HTest)
    assert res.statistic["X-squared"] == pytest.approx(0.0)
    assert res.estimate == {"p": 0.5}


def test_prop_test_one_sample_continuity_correction():
    """Yates correction subtracts ``0.5/n`` from |p̂ - p₀|."""
    # x=4, n=10, p=0.5 → diff=0.1, after correction: 0.1 - 0.05 = 0.05
    # X² = 0.05² / (0.25/10) = 0.0025 / 0.025 = 0.1
    res = prop_test(4, 10, p=0.5, correct=True)
    assert res.statistic["X-squared"] == pytest.approx(0.1)


def test_prop_test_two_sample_returns_chisq():
    """2-sample prop.test on (5/10, 8/10) — verify against direct chi-sq."""
    from scipy import stats as ss
    tbl = np.array([[5, 5], [8, 2]])
    res = prop_test([5, 8], [10, 10])
    expected = ss.chi2_contingency(tbl, correction=True)
    assert res.statistic["X-squared"] == pytest.approx(expected.statistic)
    assert res.p_value == pytest.approx(expected.pvalue)
    assert res.estimate == {"prop 1": 0.5, "prop 2": 0.8}


def test_prop_test_p_vector_not_supported():
    """``p`` as a vector hypothesis (one prob per group) is still deferred."""
    with pytest.raises(NotImplementedError, match="vector hypothesis"):
        prop_test([1, 2, 3], [10, 10, 10], p=[0.1, 0.2, 0.3])


# ---- binom_test -----------------------------------------------------


def test_binom_test_exact_p_value():
    """``P(X ≥ 8 | n=10, p=0.5) + P(X ≤ 2 | n=10, p=0.5)``."""
    res = binom_test(8, 10, p=0.5)
    # Two-sided exact p for 8/10 at p=0.5 is 0.1093750 (from R)
    assert res.p_value == pytest.approx(0.109375, rel=1e-5)
    assert res.estimate == {"probability of success": 0.8}
    assert res.null_value == 0.5


def test_binom_test_with_succ_fail_pair():
    """Pass ``(succ, fail)`` as ``x``, omit ``n``."""
    res = binom_test([8, 2])
    assert res.statistic["number of successes"] == 8
    assert res.parameter["number of trials"] == 10


def test_binom_test_ci_brackets_estimate():
    res = binom_test(8, 10, p=0.5)
    lo, hi = res.conf_int
    assert lo < 0.8 < hi


# ---- var_test -------------------------------------------------------


def test_var_test_f_statistic_known():
    """F = var(x)/var(y) when ratio=1."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, 50)
    y = rng.normal(0, 1, 60)
    res = var_test(x, y)
    F_expected = float(np.var(x, ddof=1) / np.var(y, ddof=1))
    assert res.statistic["F"] == pytest.approx(F_expected)
    assert res.parameter == {"num df": 49, "denom df": 59}


def test_var_test_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, 100)
    y = rng.normal(0, 1, 100)
    res = var_test(x, y)
    lo, hi = res.conf_int
    assert lo < res.estimate["ratio of variances"] < hi


def test_var_test_one_sided():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 3, 50)
    y = rng.normal(0, 1, 50)
    res = var_test(x, y, alternative="greater")
    assert res.alternative == "greater"
    assert res.p_value < 1e-3  # variances obviously differ


# ---- bartlett_test --------------------------------------------------


def test_bartlett_test_matches_scipy():
    from scipy import stats as ss
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0, 2, 30)
    c = rng.normal(0, 1.5, 30)
    x = np.concatenate([a, b, c])
    g = ["A"] * 30 + ["B"] * 30 + ["C"] * 30
    res = bartlett_test(x, g)
    expected = ss.bartlett(a, b, c)
    assert res.statistic["Bartlett's K-squared"] == pytest.approx(
        expected.statistic
    )
    assert res.p_value == pytest.approx(expected.pvalue)
    assert res.parameter == {"df": 2}


def test_bartlett_test_requires_2_groups():
    with pytest.raises(ValueError, match="at least 2"):
        bartlett_test([1.0, 2.0, 3.0], ["A", "A", "A"])


# ---- shapiro_test ---------------------------------------------------


def test_shapiro_test_high_p_for_normal_sample():
    rng = np.random.default_rng(4)
    x = rng.normal(size=50)
    res = shapiro_test(x)
    assert isinstance(res, HTest)
    assert 0 < res.statistic["W"] < 1
    # plenty of power-but-not-rejection on a clean normal sample
    assert res.p_value > 0.05


def test_shapiro_test_rejects_obvious_nonnormal():
    """A heavy outlier should drive the W-statistic and p-value down."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(size=49), [50.0]])
    res = shapiro_test(x)
    assert res.p_value < 0.001


def test_cor_test_spearman_exact():
    """Spearman exact p-value (AS 89, src/prho.c) — bit-exact to R.

    (The reported ``S`` may differ from R by <=1 ulp: R's ``cor`` centers via
    the system ``sqrtl`` which numpy's long-double sqrt does not reproduce; the
    p-value uses ``round(S)`` so it is unaffected.)
    """
    x = [3.1, 1.5, 4.2, 2.8, 5.9, 0.7, 3.3, 4.8, 1.1, 2.2]
    y = [2.9, 1.8, 5.1, 2.2, 6.3, 1.2, 2.7, 4.4, 0.9, 3.0]
    xr = "c(" + ",".join(repr(v) for v in x) + ")"
    yr = "c(" + ",".join(repr(v) for v in y) + ")"
    res = cor_test(x, y, method="spearman")
    _assert_r([(res.p_value,
                f'cor.test({xr}, {yr}, method="spearman")$p.value',
                0.0013802671414576686)])
    res_g = cor_test(x, y, method="spearman", alternative="greater")
    _assert_r([(res_g.p_value,
                f'cor.test({xr}, {yr}, method="spearman", '
                f'alternative="greater")$p.value', 0.00069013357072883431)])


def test_cor_test_kendall_exact():
    """Kendall exact p-value + integer T statistic (src/kendall.c), bit-exact."""
    x = [3.1, 1.5, 4.2, 2.8, 5.9, 0.7, 3.3, 4.8, 1.1, 2.2]
    y = [2.9, 1.8, 5.1, 2.2, 6.3, 1.2, 2.7, 4.4, 0.9, 3.0]
    xr = "c(" + ",".join(repr(v) for v in x) + ")"
    yr = "c(" + ",".join(repr(v) for v in y) + ")"
    res = cor_test(x, y, method="kendall")
    assert res.statistic["T"] == 39.0
    _assert_r([(res.p_value, f'cor.test({xr}, {yr}, method="kendall")$p.value',
                0.0022128527336859882)])
    res_l = cor_test(x, y, method="kendall", alternative="less")
    _assert_r([(res_l.p_value,
                f'cor.test({xr}, {yr}, method="kendall", '
                f'alternative="less")$p.value', 0.99952684082892418)])


def test_wilcox_test_one_sample_exact():
    """One-sample signed-rank, exact, no ties — bit-exact to R."""
    x = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]
    xr = "c(" + ",".join(repr(v) for v in x) + ")"
    res = wilcox_test(x, mu=1)
    assert res.statistic["V"] == 43.0
    assert res.method == "Wilcoxon signed rank exact test"
    _assert_r([(res.p_value, f"wilcox.test({xr}, mu=1)$p.value", 0.01171875)])


def test_wilcox_test_ties_permutation_exact():
    """Ties trigger R's exact permutation distribution (src/permdist.c)."""
    x = [1.0, 2, 2, 3, 3, 3, 4, 5, 5, 1]
    xr = "c(" + ",".join(repr(v) for v in x) + ")"
    res = wilcox_test(x, mu=3)
    assert res.statistic["V"] == 22.0
    _assert_r([(res.p_value,
                f"suppressWarnings(wilcox.test({xr}, mu=3))$p.value", 0.9375)])


def test_wilcox_test_two_sample_exact():
    """Two-sample rank-sum, exact, no ties — bit-exact to R."""
    x = [0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46]
    y = [1.15, 0.88, 0.90, 0.74, 1.21]
    xr = "c(" + ",".join(repr(v) for v in x) + ")"
    yr = "c(" + ",".join(repr(v) for v in y) + ")"
    res = wilcox_test(x, y)
    assert res.statistic["W"] == 35.0
    _assert_r([(res.p_value, f"wilcox.test({xr}, {yr})$p.value",
                0.2544122544122544)])


def test_wilcox_test_two_sample_asymptotic():
    """Large n uses R's continuity-corrected normal (tie-corrected sd)."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    y = rng.normal(size=65) + 0.3
    xr = "c(" + ",".join(repr(float(v)) for v in x) + ")"
    yr = "c(" + ",".join(repr(float(v)) for v in y) + ")"
    res = wilcox_test(x, y)
    assert "with continuity correction" in res.method
    _assert_r([(res.p_value, f"wilcox.test({xr}, {yr})$p.value", res.p_value)])


def test_shapiro_test_bit_exact_vs_r():
    """W and p are 0-ulp to R's ``.Call(C_SWilk)`` (ported ``src/swilk.c``)."""
    x = [2.1, 3.4, 1.9, 5.2, 4.8, 2.7, 3.3, 6.1, 0.8, 4.4, 3.9, 2.2, 5.5, 1.1, 3.7]
    res = shapiro_test(x)
    xr = "c(" + ",".join(repr(v) for v in x) + ")"
    _assert_r([
        (res.statistic["W"], f"shapiro.test({xr})$statistic", 0.97403652031051269),
        (res.p_value, f"shapiro.test({xr})$p.value", 0.91267650064222361),
    ])


def test_shapiro_test_n3_exact_p_branch():
    """n == 3 hits swilk's exact closed-form P value; bit-exact to R."""
    x = [1.0, 2.0, 10.0]
    res = shapiro_test(x)
    _assert_r([
        (res.statistic["W"], "shapiro.test(c(1,2,10))$statistic", 0.8321917808219178),
        (res.p_value, "shapiro.test(c(1,2,10))$p.value", 0.19391752148144781),
    ])


# ---- ks_test --------------------------------------------------------


def test_ks_test_two_sample_exact_bit_exact_vs_r():
    """Two-sample exact Smirnov (nx*ny < 10000) — D and p bit-exact to R."""
    a = [0.80, 1.83, 0.50, 1.62, 2.48, 1.68, 0.55, 1.30]
    b = [1.15, 0.88, 0.90, 0.74, 1.21, 2.05, 1.53]
    ar = "c(" + ",".join(repr(v) for v in a) + ")"
    br = "c(" + ",".join(repr(v) for v in b) + ")"
    res = ks_test(a, b)
    assert "Exact" in res.method
    _assert_r([
        (res.statistic["D"], f"ks.test({ar}, {br})$statistic", 0.3571428571428571),
        (res.p_value, f"ks.test({ar}, {br})$p.value", 0.58461538461538454),
    ])


def test_ks_test_two_sample_ties_bit_exact_vs_r():
    """Ties trigger R's exact ties recursion (src/ks.c); bit-exact to R."""
    a = [1.0, 2, 2, 3, 3, 4, 5, 1]
    b = [2.0, 3, 3, 3, 4, 5, 6]
    ar = "c(" + ",".join(repr(v) for v in a) + ")"
    br = "c(" + ",".join(repr(v) for v in b) + ")"
    res = ks_test(a, b)
    _assert_r([
        (res.statistic["D"],
         f"suppressWarnings(ks.test({ar}, {br}))$statistic", 0.35714285714285715),
        (res.p_value,
         f"suppressWarnings(ks.test({ar}, {br}))$p.value", 0.51048951048951052),
    ])


def test_ks_test_one_sample_exact_bit_exact_vs_r():
    """One-sample exact (n < 100, no ties) vs pnorm — D and p bit-exact to R."""
    x = [-0.62, 0.41, 1.30, -1.05, 0.72, -0.31, 2.10, 0.05, -1.44, 0.88,
         0.19, -0.77, 1.61, -0.23, 0.50]
    xr = "c(" + ",".join(repr(v) for v in x) + ")"
    res = ks_test(x, "pnorm")
    assert "Exact" in res.method
    _assert_r([
        (res.statistic["D"], f'ks.test({xr}, "pnorm")$statistic',
         0.12576369289434408),
        (res.p_value, f'ks.test({xr}, "pnorm")$p.value', 0.94765241386502763),
    ])
    res_g = ks_test(x, "pnorm", alternative="greater")
    _assert_r([
        (res_g.p_value, f'ks.test({xr}, "pnorm", alternative="greater")$p.value',
         0.97710988321079739),
    ])


# ---- mcnemar_test ---------------------------------------------------


def test_mcnemar_test_known_table():
    """Standard textbook example: ``[[101, 121], [59, 33]]`` → χ² ≈ 21.36."""
    tbl = np.array([[101, 121], [59, 33]])
    res = mcnemar_test(tbl, correct=False)
    # (b - c)^2 / (b + c) = (121 - 59)^2 / (121 + 59) = 3844 / 180
    expected_stat = (121 - 59) ** 2 / (121 + 59)
    assert res.statistic["McNemar's chi-squared"] == pytest.approx(
        expected_stat
    )


def test_mcnemar_test_continuity_correction():
    """Yates: ``(|b - c| - 1)² / (b + c)``; verify offset is applied."""
    tbl = np.array([[101, 121], [59, 33]])
    res_raw = mcnemar_test(tbl, correct=False)
    res_corr = mcnemar_test(tbl, correct=True)
    # (62 - 1)^2 / 180 < 62^2 / 180
    assert res_corr.statistic["McNemar's chi-squared"] < res_raw.statistic[
        "McNemar's chi-squared"
    ]


def test_mcnemar_test_rejects_non_2x2():
    with pytest.raises(ValueError, match="2x2"):
        mcnemar_test(np.array([[1, 2, 3], [4, 5, 6]]))


# ---- friedman_test --------------------------------------------------


def test_friedman_test_matches_scipy_long_to_wide():
    """Long-form (y, groups, blocks) reshaped → ``friedmanchisquare(*samples)``."""
    from scipy import stats as ss
    # 3 groups × 5 blocks
    rng = np.random.default_rng(8)
    samples = [rng.normal(loc=mu, size=5) for mu in (0, 0.5, 1.0)]
    y, groups, blocks = [], [], []
    for gi, sample in enumerate(samples):
        for bi, val in enumerate(sample):
            y.append(val)
            groups.append(f"g{gi}")
            blocks.append(f"b{bi}")
    res = friedman_test(y, groups, blocks)
    expected = ss.friedmanchisquare(*samples)
    assert res.statistic["Friedman chi-squared"] == pytest.approx(
        expected.statistic
    )
    assert res.parameter == {"df": 2}


def test_friedman_test_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        friedman_test([1.0, 2.0], ["a", "b", "c"], ["1", "2", "3"])


# ---- HTest repr is human-readable -----------------------------------


def test_htest_repr_contains_method_and_p():
    out = t_test([1.0, 2.0, 3.0, 4.0, 5.0], mu=3.0)
    s = repr(out)
    assert "One Sample t-test" in s
    assert "p-value" in s


# ---------------------------------------------------------------------------
# DataFrame helpers: cut / findInterval / table / xtabs / prop_table /
# addmargins
# ---------------------------------------------------------------------------


from hea.R import (  # noqa: E402  — grouped with the helper tests
    addmargins, cut, findInterval, prop_table, table, xtabs,
)


# ---- cut ------------------------------------------------------------


def test_cut_default_is_right_closed():
    """``cut(x, breaks)`` defaults to ``right=True``: ``(a, b]`` semantics."""
    out = cut([1, 2, 5, 10, 0, 11], breaks=[0, 2, 5, 10])
    # boundary value 2 → (0,2]; 5 → (2,5]; 10 → (5,10]; 0 and 11 are out-of-range
    assert out.to_list() == ["(0,2]", "(0,2]", "(2,5]", "(5,10]", None, None]


def test_cut_left_closed_with_right_false():
    out = cut([0, 2, 5, 10], breaks=[0, 2, 5, 10], right=False)
    # 0 → [0,2); 2 → [2,5); 5 → [5,10); 10 not in any (right edge open)
    assert out.to_list() == ["[0,2)", "[2,5)", "[5,10)", None]


def test_cut_include_lowest_brings_in_boundary():
    out = cut(
        [1, 2, 5, 10, 0, 11], breaks=[0, 2, 5, 10], include_lowest=True
    )
    # x=0 now in [0,2]; lowest label changes from "(0,2]" to "[0,2]"
    assert out.to_list() == [
        "[0,2]", "[0,2]", "(2,5]", "(5,10]", "[0,2]", None,
    ]


def test_cut_returns_pl_enum_factor():
    out = cut([1, 3, 7], breaks=[0, 2, 5, 10])
    assert isinstance(out, pl.Series)
    assert isinstance(out.dtype, pl.Enum)
    assert out.dtype.categories.to_list() == ["(0,2]", "(2,5]", "(5,10]"]


def test_cut_labels_false_returns_codes():
    out = cut([1, 3, 7, 100], breaks=[0, 2, 5, 10], labels=False)
    # 0-based codes (hea convention; R / dplyr emits 1-based);
    # out-of-range → NaN.
    assert isinstance(out, np.ndarray)
    assert out[0] == 0
    assert out[1] == 1
    assert out[2] == 2
    assert np.isnan(out[3])


def test_cut_custom_labels():
    out = cut([0.5, 2.5, 6.0], breaks=[0, 2, 5, 10],
              labels=["lo", "med", "hi"])
    assert out.to_list() == ["lo", "med", "hi"]
    assert out.dtype.categories.to_list() == ["lo", "med", "hi"]


def test_cut_scalar_breaks_makes_n_equal_width_bins():
    """``breaks=4`` should produce 4 bins covering ``[min - eps, max + eps]``."""
    out = cut(np.linspace(0, 10, 11), breaks=4)
    cats = out.dtype.categories.to_list()
    assert len(cats) == 4


def test_cut_breaks_must_be_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        cut([1, 2, 3], breaks=[2, 1, 5])


def test_cut_label_count_must_match_bins():
    with pytest.raises(ValueError, match="2 labels but 3 bins"):
        cut([1, 2], breaks=[0, 2, 5, 10], labels=["a", "b"])


# ---- findInterval ---------------------------------------------------


def test_findInterval_basic():
    """``findInterval(x, vec)`` returns 0..N where vec[i-1] ≤ x < vec[i]."""
    out = findInterval([0.5, 2.0, 3.5, 7.0, 10.0, 11.0], [1, 5, 10])
    # 0.5 < 1 → 0; 2.0 in [1,5) → 1; 3.5 → 1; 7.0 in [5,10) → 2;
    # 10.0 ≥ 10 → 3 (above all); 11.0 → 3
    assert out.tolist() == [0, 1, 1, 2, 3, 3]


def test_findInterval_rightmost_closed_pulls_back_endpoint():
    out = findInterval([10.0, 11.0], [1, 5, 10], rightmost_closed=True)
    # 10.0 now in [5, 10] (last interval) → 2; 11.0 still 3
    assert out.tolist() == [2, 3]


def test_findInterval_all_inside_clips():
    out = findInterval([0, 11.0], [1, 5, 10], all_inside=True)
    # 0 normally → 0 but all_inside clamps to [1, len(vec)-1] = [1, 2]
    assert out.tolist() == [1, 2]


def test_findInterval_left_open():
    out = findInterval([1.0, 5.0, 10.0], [1, 5, 10], left_open=True)
    # left_open: (vec[i-1], vec[i]]; x=1 not > 1 → 0; x=5 → 1; x=10 → 2
    assert out.tolist() == [0, 1, 2]


def test_findInterval_rejects_unsorted_vec():
    with pytest.raises(ValueError, match="non-decreasing"):
        findInterval([1.0], [3, 1, 2])


# ---- table ----------------------------------------------------------


def test_table_one_way_returns_value_n():
    out = table(["a", "b", "a", "c", "b", "a"])
    assert out.columns == ["value", "n"]
    assert out["value"].to_list() == ["a", "b", "c"]
    assert out["n"].to_list() == [3, 2, 1]


def test_table_two_way_pivots():
    out = table(["a", "a", "b", "b", "b"], ["x", "y", "x", "y", "y"])
    # First col is row label; remaining cols are y-levels (sorted).
    assert out.columns == ["", "x", "y"]
    assert out[""].to_list() == ["a", "b"]
    assert out["x"].to_list() == [1, 1]
    assert out["y"].to_list() == [1, 2]


def test_table_dnn_renames_label_column():
    out = table(["a", "a", "b"], ["x", "y", "y"], dnn=("group", "outcome"))
    assert out.columns[0] == "group"


def test_table_drops_nulls_by_default():
    out = table([1, 2, None, 2, None])
    assert out["value"].to_list() == ["1", "2"]
    assert out["n"].to_list() == [1, 2]


# ---- xtabs ----------------------------------------------------------


def test_xtabs_one_way():
    df = pl.DataFrame({"g": ["a", "b", "a", "b", "a"]})
    out = xtabs("~ g", df)
    assert out["value"].to_list() == ["a", "b"]
    assert out["n"].to_list() == [3, 2]


def test_xtabs_two_way_uses_dnn():
    df = pl.DataFrame({
        "g": ["a", "b", "a", "b", "a"],
        "h": ["x", "x", "y", "y", "y"],
    })
    out = xtabs("~ g + h", df)
    # The first column carries the row variable's name (left side of +)
    assert out.columns[0] == "g"
    assert sorted(out.columns[1:]) == ["x", "y"]


def test_xtabs_lhs_weighted_one_way():
    """R: ``xtabs(w ~ g, df)`` sums ``w`` per level of ``g``."""
    df = pl.DataFrame({"w": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]})
    out = xtabs("w ~ g", df)
    assert out.columns == ["g", "n"]
    rows = {r["g"]: r["n"] for r in out.iter_rows(named=True)}
    assert rows == {"a": 4.0, "b": 2.0}


def test_xtabs_lhs_weighted_two_way():
    """R: ``xtabs(w ~ a + b, df)`` sums ``w`` per (a, b) cell, wide."""
    df = pl.DataFrame({
        "w": [10, 20, 30, 40],
        "a": ["x", "x", "y", "y"],
        "b": ["p", "q", "p", "q"],
    })
    out = xtabs("w ~ a + b", df)
    rows = {r["a"]: (r["p"], r["q"]) for r in out.iter_rows(named=True)}
    assert rows == {"x": (10, 20), "y": (30, 40)}


# ---- prop_table -----------------------------------------------------


def test_prop_table_grand_total_sums_to_one():
    tbl = table(["a", "a", "b", "b", "b"], ["x", "y", "x", "y", "y"])
    out = prop_table(tbl)
    counts = out.select(["x", "y"]).to_numpy().astype(float)
    assert counts.sum() == pytest.approx(1.0)


def test_prop_table_row_proportions_sum_to_one():
    tbl = table(["a", "a", "b", "b", "b"], ["x", "y", "x", "y", "y"])
    out = prop_table(tbl, margin=1)
    rows = out.select(["x", "y"]).to_numpy().astype(float)
    np.testing.assert_allclose(rows.sum(axis=1), 1.0)


def test_prop_table_column_proportions_sum_to_one():
    tbl = table(["a", "a", "b", "b", "b"], ["x", "y", "x", "y", "y"])
    out = prop_table(tbl, margin=2)
    cols = out.select(["x", "y"]).to_numpy().astype(float)
    np.testing.assert_allclose(cols.sum(axis=0), 1.0)


def test_prop_table_works_on_ndarray():
    arr = np.array([[1, 2], [3, 4]], dtype=float)
    out = prop_table(arr)
    assert out.sum() == pytest.approx(1.0)


def test_prop_table_invalid_margin():
    tbl = table(["a"], ["x"])
    with pytest.raises(ValueError, match="must be None, 1, or 2"):
        prop_table(tbl, margin=3)


# ---- addmargins -----------------------------------------------------


def test_addmargins_2way_default_adds_both():
    tbl = table(["a", "a", "b", "b", "b"], ["x", "y", "x", "y", "y"])
    out = addmargins(tbl)
    # Adds a "Sum" column AND a "Sum" row (with the grand total at the corner)
    assert "Sum" in out.columns
    assert out[""].to_list()[-1] == "Sum"
    grand_total = float(out.row(-1, named=True)["Sum"])
    assert grand_total == 5.0


def test_addmargins_margin_2_adds_only_row_sums():
    tbl = table(["a", "a", "b"], ["x", "y", "y"])
    out = addmargins(tbl, margin=2)
    assert "Sum" in out.columns
    assert "Sum" not in out[""].to_list()


def test_addmargins_margin_1_adds_only_column_sums():
    tbl = table(["a", "a", "b"], ["x", "y", "y"])
    out = addmargins(tbl, margin=1)
    assert "Sum" in out[""].to_list()
    assert "Sum" not in out.columns


def test_addmargins_oneway_appends_sum_row():
    tbl = table(["a", "b", "a", "c", "b", "a"])
    out = addmargins(tbl)
    assert out["value"].to_list() == ["a", "b", "c", "Sum"]
    assert out["n"].to_list() == [3.0, 2.0, 1.0, 6.0]


# ---------------------------------------------------------------------------
# Newly enabled deferred functions: glm/gam jackknife, update, terms,
# prop_test for k > 2.
# ---------------------------------------------------------------------------


from hea.R import (  # noqa: E402  — grouped with the deferred-fn tests
    Terms, terms, update,
)


# ---- glm/gam jackknife: rstudent / dffits / dfbetas / influence -----


@pytest.fixture(scope="module")
def m_glm_gauss(gala):
    """Gaussian glm — used to verify jackknife formulas reduce to lm's."""
    return hea.models.glm("Species ~ Area + Elevation", gala, family=hea.family.gaussian())


def test_glm_rstudent_returns_array(m_glm):
    out = rstudent(m_glm)
    assert isinstance(out, np.ndarray)
    assert out.shape == (30,)


def test_glm_dffits_returns_array(m_glm):
    out = dffits(m_glm)
    assert out.shape == (30,)


def test_glm_dfbetas_shape(m_glm):
    out = dfbetas(m_glm)
    assert isinstance(out, pl.DataFrame)
    assert out.shape == (30, 3)
    assert out.columns == ["(Intercept)", "Area", "Elevation"]


def test_glm_dfbetas_closed_form_vs_loo_refit(gala, m_glm):
    """For a Poisson glm (scale_known=True), the closed-form ``dfbetas[0]``
    should be a tight first-order approximation to the actual change in
    coefficients when we drop observation 0 and refit.
    """
    predicted = np.array(dfbetas(m_glm).row(0))
    m_drop0 = hea.models.glm(
        "Species ~ Area + Elevation",
        gala.slice(1),
        family=hea.family.poisson(),
    )
    bhat_full = coef(m_glm).values
    bhat_drop = coef(m_drop0).values
    delta = bhat_full - bhat_drop
    # Poisson is scale-known, so sigma_(-i) = 1; closed form scales delta
    # by sqrt(diag(XtWXinv)).
    XtWXinv = np.asarray(m_glm.V_bhat) / m_glm.dispersion
    sd_j = np.sqrt(np.diag(XtWXinv))
    expected = delta / sd_j
    # First-order Taylor approximation; tight on this dataset (~3e-4).
    np.testing.assert_allclose(predicted, expected, atol=1e-3)


def test_glm_influence_dict_shape(m_glm):
    infl = influence(m_glm)
    assert set(infl.keys()) == {"hat", "sigma", "coefficients", "residuals"}
    assert len(infl["hat"]) == 30
    assert isinstance(infl["coefficients"], pl.DataFrame)
    assert infl["coefficients"].shape == (30, 3)


def test_glm_known_scale_keeps_sigma_at_one(m_glm):
    """Poisson is scale-known → ``influence(m)['sigma']`` is all 1s."""
    np.testing.assert_array_equal(influence(m_glm)["sigma"], np.ones(30))


def test_glm_unknown_scale_uses_loo_deviance(m_glm_gauss):
    """Gaussian glm is unknown-scale → leave-one-out σ varies by row."""
    sigma = influence(m_glm_gauss)["sigma"]
    assert sigma.shape == (30,)
    assert sigma.std() > 0  # not constant


def test_gam_diagnostics_run_end_to_end():
    """gam uses the penalized full design ``_X_full``; check shapes only."""
    mt = hea.data("mtcars", package="R")
    g = hea.models.gam("mpg ~ s(wt) + s(hp)", mt)
    assert rstudent(g).shape == (32,)
    assert dffits(g).shape == (32,)
    db = dfbetas(g)
    assert db.shape == (32, 19)
    assert "(Intercept)" in db.columns
    infl = influence(g)
    assert infl["hat"].shape == (32,)
    assert infl["sigma"].shape == (32,)


# ---- update ---------------------------------------------------------


def test_update_full_formula_returns_new_fit(gala, m_lm):
    new = update(m_lm, "Species ~ Area")
    assert new is not m_lm
    assert new.p == 2
    assert "Elevation" not in coef(new)


def test_update_formula_optional_refits_same_model(gala, m_lm):
    """R parity: ``update(fm)`` (no formula) refits with the original
    formula — equivalent to R's default ``formula. = .``. Lets calls
    like ``update(fm, REML=FALSE)`` work without retyping the formula.
    """
    new = update(m_lm)
    assert new is not m_lm
    assert new.formula == m_lm.formula
    assert coef(new).names == coef(m_lm).names


def test_update_delta_add_term(gala, m_lm):
    """``. ~ . + x`` keeps existing RHS and appends a term."""
    new = update(m_lm, ". ~ . + Adjacent")
    assert "Adjacent" in coef(new)
    # Check that all original terms are still there
    assert "Area" in coef(new)
    assert "Elevation" in coef(new)


def test_update_delta_drop_term(gala, m_lm):
    new = update(m_lm, ". ~ . - Area")
    assert "Area" not in coef(new)
    assert "Elevation" in coef(new)


def test_update_glm_carries_family(gala, m_glm):
    """``update(glm, …)`` should keep the original family without re-specifying."""
    new = update(m_glm, ". ~ . + Adjacent")
    # Same family class as original
    assert type(new.family).__name__ == type(m_glm.family).__name__


def test_update_glm_can_override_family(gala, m_glm):
    """Explicit ``family=`` in kwargs wins over the auto-forward."""
    new = update(m_glm, "Species ~ Area + Elevation", family=hea.family.gaussian())
    assert type(new.family).__name__ == "Gaussian"


def test_update_requires_tilde():
    with pytest.raises(ValueError, match="must contain '~'"):
        update(m_lm, "Area + Elevation")


def test_update_lhs_dot_substitution(gala, m_lm):
    """``log(y) ~ . - x`` keeps the original RHS structure even when
    transforming the LHS."""
    new = update(m_lm, "Species ~ . - Area")
    assert "Area" not in coef(new)
    assert "Elevation" in coef(new)


def test_update_carries_weights_for_lm(gala):
    """``weights`` is auto-forwarded when the model was fit with weights."""
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, gala.height)
    m_w = hea.models.lm("Species ~ Area", gala, weights=w)
    new = update(m_w, ". ~ . + Elevation")
    assert new.weights is not None
    np.testing.assert_array_equal(new.weights, w)


def test_update_carries_method_for_gam():
    """``method`` (REML/ML) is auto-forwarded for gam."""
    mt = hea.data("mtcars", package="R")
    m_gam_reml = hea.models.gam("mpg ~ s(wt)", mt, method="REML")
    new = update(m_gam_reml, ". ~ . + s(hp)")
    assert new.method == "REML"


def test_update_kwargs_override_auto_forward(gala):
    """Explicit kwargs win over the auto-forward."""
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, gala.height)
    m_w = hea.models.lm("Species ~ Area", gala, weights=w)
    # Override: refit unweighted
    new = update(m_w, ". ~ . + Elevation", weights=None)
    assert new.weights is None


# ---- terms ----------------------------------------------------------


def test_terms_returns_dataclass(m_lm):
    t = terms(m_lm)
    assert isinstance(t, Terms)
    assert t.formula == "Species ~ Area + Elevation"
    assert t.response == "Species"
    assert t.term_labels == ["Area", "Elevation"]


def test_terms_repr_is_human_readable(m_lm):
    s = repr(terms(m_lm))
    assert "Species" in s
    assert "Area" in s


def test_terms_for_glm(m_glm):
    t = terms(m_glm)
    assert t.response == "Species"
    assert "Area" in t.term_labels


# ---- prop_test extension to k > 2 -----------------------------------


def test_prop_test_k_3_returns_chi_squared():
    """3-sample equality test produces a chi-squared with df=2."""
    res = prop_test([5, 8, 9], [10, 10, 10])
    assert res.parameter == {"df": 2}
    assert "3-sample" in res.method
    # Continuity correction is silently dropped for k > 2 (matches R).
    assert "continuity correction" not in res.method


def test_prop_test_k_2_still_supports_continuity_correction():
    """k=2 keeps R's default Yates continuity correction."""
    res = prop_test([5, 8], [10, 10])
    assert "continuity correction" in res.method


def test_prop_test_estimates_for_k_3():
    res = prop_test([3, 7, 8], [10, 10, 10])
    assert res.estimate == {"prop 1": 0.3, "prop 2": 0.7, "prop 3": 0.8}
