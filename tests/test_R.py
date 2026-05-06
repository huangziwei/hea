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


def test_seq_one_arg_is_one_based():
    assert seq(5).tolist() == [1, 2, 3, 4, 5]


def test_seq_from_to():
    assert seq(2, 6).tolist() == [2, 3, 4, 5, 6]


def test_seq_with_by():
    assert seq(2, 10, by=2).tolist() == [2, 4, 6, 8, 10]
    assert seq(10, 2, by=-2).tolist() == [10, 8, 6, 4, 2]


def test_seq_length_out():
    assert seq(0, 1, length_out=5).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_seq_along_with():
    assert seq(along_with=["a", "b", "c"]).tolist() == [1, 2, 3]


def test_seq_len_seq_along_one_based():
    assert seq_len(4).tolist() == [1, 2, 3, 4]
    assert seq_along(["x", "y"]).tolist() == [1, 2]


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
