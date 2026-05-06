"""Tests for ``hea.R`` — the R-like free-function namespace.

The module is designed for ``from hea.R import *``, so we exercise it
that way to confirm the public surface is what we advertise and that
none of it shadows a Python builtin.
"""

from __future__ import annotations

import builtins

import numpy as np
import polars as pl
import pytest

import hea
from hea import R as R_mod
from hea.R import (
    as_character, as_integer, as_logical, as_numeric,
    colnames, complete_cases, cor, cov, cumsum, cummax, cummin, cumprod,
    dim, diff, duplicated,
    factor, glimpse, head,
    is_factor, is_finite, is_na, is_null, is_numeric,
    length, levels, mean, median,
    na_omit, names, ncol, nlevels, nrow,
    order, quantile, rev, sd, seq, seq_along, seq_len, sort, summary, tail,
    tabulate, unique, var, which, which_max, which_min,
    # distributions (a representative subset; full grid checked elsewhere)
    dnorm, pnorm, qnorm, rnorm,
    pt, qt, qchisq, pchisq, qf, pf,
    dbinom, pbinom, dpois, ppois, punif, qexp, pgamma, pbeta,
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
    """``summary(hea.DataFrame)`` must reach the existing ``.summary()``."""
    d = hea.tbl(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    out = summary(d)
    # Existing .summary() returns a Summary object with __repr__
    assert "Min" in repr(out)


def test_summary_raises_on_unsupported():
    with pytest.raises(TypeError, match="no .summary"):
        summary(np.array([1, 2, 3]))


def test_glimpse_dispatches(df):
    # polars DataFrame has .glimpse() — just confirm we don't error
    glimpse(df)


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


def test_tabulate_one_based():
    """R's ``tabulate(c(1,2,2,3,3,3))`` is c(1, 2, 3) — 1-based bins."""
    assert tabulate([1, 2, 2, 3, 3, 3]).tolist() == [1, 2, 3]
    assert tabulate([2, 2, 4], nbins=5).tolist() == [0, 2, 0, 1, 0]


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


# ---------------------------------------------------------------------------
# Distributions — agreement with known R values
# ---------------------------------------------------------------------------


def test_dnorm_pnorm_qnorm():
    assert dnorm(0) == pytest.approx(0.3989422804, rel=1e-6)
    assert pnorm(1.96) == pytest.approx(0.9750021048, rel=1e-6)
    assert qnorm(0.975) == pytest.approx(1.959963985, rel=1e-6)


def test_pnorm_lower_tail_false():
    assert pnorm(1.96, lower_tail=False) == pytest.approx(
        1 - 0.9750021048, rel=1e-6
    )


def test_qnorm_lower_tail_false():
    # P(Z > q) = 0.025  →  q = qnorm(0.975) ≈ 1.959964
    assert qnorm(0.025, lower_tail=False) == pytest.approx(
        1.959963985, rel=1e-6
    )


def test_t_distribution():
    assert qt(0.975, df=10) == pytest.approx(2.228138852, rel=1e-6)
    assert pt(2, df=10) == pytest.approx(0.963306167, rel=1e-6)


def test_chisq_distribution():
    assert qchisq(0.95, df=1) == pytest.approx(3.841458821, rel=1e-6)
    assert pchisq(3.841458821, df=1) == pytest.approx(0.95, abs=1e-6)


def test_f_distribution():
    assert qf(0.95, 2, 10) == pytest.approx(4.102821, rel=1e-5)
    assert pf(4.102821, 2, 10) == pytest.approx(0.95, abs=1e-5)


def test_binom():
    # dbinom(3, 10, 0.5) = C(10,3) / 2^10 = 120/1024 = 0.1171875
    assert float(dbinom(3, 10, 0.5)) == pytest.approx(0.1171875)
    # pbinom(3, 10, 0.5) = sum_{k=0..3} C(10,k)/1024 = (1+10+45+120)/1024
    assert float(pbinom(3, 10, 0.5)) == pytest.approx(176 / 1024)


def test_poisson_with_lambda_keyword():
    # dpois(2, lambda=3) = 3^2 * exp(-3) / 2 = 9 * 0.04979 / 2
    expected = 9 * np.exp(-3) / 2
    assert float(dpois(2, lambda_=3)) == pytest.approx(expected, rel=1e-6)
    assert float(ppois(2, lambda_=3)) == pytest.approx(0.4231900811, rel=1e-6)


def test_uniform_exp_gamma_beta():
    assert float(punif(0.3)) == pytest.approx(0.3)
    assert float(qexp(0.5)) == pytest.approx(np.log(2), rel=1e-6)
    # pgamma(1, shape=2, rate=1) = 1 - 2*exp(-1) = 0.2642411
    assert float(pgamma(1, shape=2, rate=1)) == pytest.approx(
        1 - 2 * np.exp(-1), rel=1e-6
    )
    # pbeta(0.5, 2, 5) = 57/64 = 0.890625
    assert float(pbeta(0.5, 2, 5)) == pytest.approx(57 / 64)


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
    return hea.lm("Species ~ Area + Elevation", gala)


@pytest.fixture(scope="module")
def m_glm(gala):
    return hea.glm("Species ~ Area + Elevation", gala, family=hea.poisson())


@pytest.fixture(scope="module")
def m_gam():
    mt = hea.data("mtcars", package="R")
    return hea.gam("mpg ~ s(wt) + s(hp)", mt)


@pytest.fixture(scope="module")
def m_lme():
    sleep = hea.data("sleepstudy", package="lme4")
    return hea.lme("Reaction ~ Days + (Days|Subject)", sleep)


# ---- coef / coefficients / fixef ------------------------------------


def test_coef_returns_named_dict(m_lm):
    c = coef(m_lm)
    assert isinstance(c, dict)
    assert set(c.keys()) == {"(Intercept)", "Area", "Elevation"}
    assert all(isinstance(v, float) for v in c.values())


def test_coefficients_alias(m_lm):
    assert coefficients(m_lm) == coef(m_lm)


def test_coef_works_on_glm_gam_lme(m_glm, m_gam, m_lme):
    assert "(Intercept)" in coef(m_glm)
    # gam: intercept + 9 wt basis + 9 hp basis
    assert "(Intercept)" in coef(m_gam)
    # lme: fixed effects only (= fixef)
    c = coef(m_lme)
    assert set(c.keys()) == {"(Intercept)", "Days"}


def test_fixef_equals_coef(m_lm, m_lme):
    assert fixef(m_lm) == coef(m_lm)
    assert fixef(m_lme) == coef(m_lme)


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
    with pytest.raises(ValueError, match="not supported"):
        resid(m_lm, type="pearson")


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
    # lm.predict() returns a polars DataFrame with "Fitted" column
    assert isinstance(out, pl.DataFrame)
    np.testing.assert_array_almost_equal(
        out["Fitted"].to_numpy(), fitted(m_lm)
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
    """lme stores ``vcov_beta`` as a DataFrame with named cols."""
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
    """REML lme fit has no defined residual df; we raise."""
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
    m1 = hea.lm("Species ~ Area", gala)
    m2 = hea.lm("Species ~ Area + Elevation", gala)
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
    m_drop0 = hea.lm("Species ~ Area + Elevation", gala_drop0)
    sigma_loo_0 = m_drop0.sigma  # σ from the leave-one-out fit

    expected_rs0 = e0 / (sigma_loo_0 * np.sqrt(1 - h0))
    assert rs_full[0] == pytest.approx(expected_rs0, rel=1e-8)


def test_rstudent_raises_for_glm(m_glm):
    with pytest.raises(NotImplementedError, match="lm only"):
        rstudent(m_glm)


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
    m_drop0 = hea.lm("Species ~ Area + Elevation", gala_drop0)
    bhat_full = np.array(list(coef(m_lm).values()))
    bhat_drop = np.array(list(coef(m_drop0).values()))
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
    m_drop0 = hea.lm("Species ~ Area + Elevation", gala.slice(1))
    assert sigma_full[0] == pytest.approx(m_drop0.sigma, rel=1e-8)


def test_dfbetas_raises_for_glm(m_glm):
    with pytest.raises(NotImplementedError, match="lm only"):
        dfbetas(m_glm)


def test_dffits_raises_for_glm(m_glm):
    with pytest.raises(NotImplementedError, match="lm only"):
        dffits(m_glm)


def test_influence_raises_for_glm(m_glm):
    with pytest.raises(NotImplementedError, match="lm only"):
        influence(m_glm)


def test_diagnostics_raise_for_weighted_lm(gala):
    """Diagnostics that need leave-one-out σ refuse weighted lm."""
    w = np.ones(gala.height)
    w[0] = 2.0
    m_wt = hea.lm("Species ~ Area + Elevation", gala, weights=w)
    for fn in (rstudent, dffits, dfbetas, influence):
        with pytest.raises(NotImplementedError, match="weighted"):
            fn(m_wt)


# ---------------------------------------------------------------------------
# Hypothesis tests (the new batch + verifying the consolidation kept the rest)
# ---------------------------------------------------------------------------


from hea.R import (  # noqa: E402  — grouped with the test-batch tests
    HTest,
    bartlett_test, binom_test, chisq_test, cor_test,
    fisher_test, friedman_test, kruskal_test, ks_test,
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


def test_fisher_test_2x2_matches_scipy():
    """Fisher exact on a 2×2 should match ``scipy.stats.fisher_exact``."""
    from scipy import stats as ss
    tbl = np.array([[8, 2], [1, 5]])
    res = fisher_test(tbl)
    expected = ss.fisher_exact(tbl)
    assert isinstance(res, HTest)
    assert res.method == "Fisher's Exact Test for Count Data"
    assert res.estimate["odds ratio"] == pytest.approx(expected.statistic)
    assert res.p_value == pytest.approx(expected.pvalue)
    assert res.null_value == 1.0


def test_fisher_test_from_two_vectors():
    """Passing parallel vectors should produce the same call as a 2x2 table."""
    x = ["a", "a", "a", "a", "b", "b"]
    y = ["x", "x", "x", "y", "x", "y"]
    res = fisher_test(x, y)
    assert isinstance(res, HTest)
    assert "odds ratio" in res.estimate


def test_fisher_test_rejects_non_2x2():
    with pytest.raises(NotImplementedError, match="2x2"):
        fisher_test(np.array([[1, 2, 3], [4, 5, 6]]))


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


def test_prop_test_k_greater_than_2_raises():
    with pytest.raises(NotImplementedError, match="k>2"):
        prop_test([1, 2, 3], [10, 10, 10])


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


# ---- ks_test --------------------------------------------------------


def test_ks_test_two_sample_matches_scipy():
    from scipy import stats as ss
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 100)
    y = rng.normal(1, 1, 100)  # shifted
    res = ks_test(x, y)
    expected = ss.ks_2samp(x, y)
    assert res.statistic["D"] == pytest.approx(expected.statistic)
    assert res.p_value == pytest.approx(expected.pvalue)


def test_ks_test_one_sample_with_pnorm_string():
    """R-style 'pnorm' string should map onto scipy's 'norm'."""
    from scipy import stats as ss
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    res = ks_test(x, "pnorm")
    expected = ss.kstest(x, "norm")
    assert res.statistic["D"] == pytest.approx(expected.statistic)


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
