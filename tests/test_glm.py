"""End-to-end tests for ``hea.glm``.

Sections (top → bottom):

1. **Oracle presence** — every (family, link, dataset) triple in the
   glm-port plan has a JSON oracle dumped by
   ``tests/scripts/make_glm_oracles.R``, parses to the expected shape,
   and carries the right ``test_kind`` (z for scale-known, t for
   estimated-scale families). Catches a missing fixture before parity
   tests fail with a confusing FileNotFoundError.

2. **Per-oracle parity** — ``hea.glm(...)`` outputs pinned against the
   corresponding ``stats::glm(...)`` oracle. Phase 1 (engine + core
   fit), Phase 2 (Wald inference), Phase 3 (null deviance / AIC / BIC
   / logLik), Phase 4 (predict link/response, se.fit). Tolerances
   tight (5e-5 on coef / dev / dispersion / AIC) — Fisher-IRLS is
   essentially deterministic.

3. **Edge cases** — Phase 6 API surface that's commonly used in R but
   easy to overlook in a port: cbind LHS, formula-side offset, prior
   weights, intercept-only, rank-deficient X, factor-response
   binomial, Quasi family.

4. **compare integration** — ``anova(glm1, glm2[, ...])`` against
   ``anova.glm()`` oracles; ``AIC()`` / ``BIC()`` smoke-tests; cross-
   model-type rejections.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import polars as pl
import pytest

from conftest import load_dataset, load_glm_oracle
from hea import (
    AIC,
    BIC,
    Binomial,
    Gamma,
    Gaussian,
    InverseGaussian,
    Poisson,
    Quasi,
    anova,
    glm,
)
from hea.compare import _anova_glm_table


# =============================================================================
# 1. Oracle presence (every (family, link, dataset) triple in glm-port plan)
# =============================================================================


EXPECTED = [
    ("gaussian_identity_iris",        "gaussian", "identity"),
    ("gaussian_log_insurance",        "gaussian", "log"),
    ("gamma_inverse_trees",           "Gamma",    "inverse"),
    ("gamma_log_trees",               "Gamma",    "log"),
    ("poisson_log_quine",             "poisson",  "log"),
    ("poisson_sqrt_quine",            "poisson",  "sqrt"),
    ("binomial_logit_menarche",       "binomial", "logit"),
    ("binomial_probit_menarche",      "binomial", "probit"),
    ("binomial_cauchit_menarche",     "binomial", "cauchit"),
    ("binomial_cloglog_menarche",     "binomial", "cloglog"),
    ("ig_canonical_insurance",        "inverse.gaussian", "1/mu^2"),
]

REQUIRED_KEYS = {
    "id", "formula", "family_name", "link_name", "n", "coef_names",
    "coefficients", "std_error", "test_stat", "p_value", "test_kind",
    "ci_lower", "ci_upper", "vcov",
    "deviance", "null_deviance", "df_residual", "df_null",
    "aic", "bic", "loglik", "loglik_df", "dispersion", "iter", "converged",
    "fitted_values", "linear_pred",
    "res_deviance", "res_pearson", "res_working", "res_response",
    "pred_link_fit", "pred_link_se", "pred_resp_fit", "pred_resp_se",
}


@pytest.mark.parametrize("oid,family,link", EXPECTED)
def test_oracle_present(oid: str, family: str, link: str):
    o = load_glm_oracle(oid)
    assert o["family_name"] == family, f"{oid}: family mismatch"
    assert o["link_name"] == link, f"{oid}: link mismatch"
    missing = REQUIRED_KEYS - set(o.keys())
    assert not missing, f"{oid}: missing keys {sorted(missing)}"
    p = len(o["coefficients"])
    assert len(o["std_error"]) == p
    assert len(o["test_stat"]) == p
    assert len(o["p_value"]) == p
    assert len(o["coef_names"]) == p
    n = o["n"]
    assert len(o["fitted_values"]) == n
    assert len(o["linear_pred"]) == n
    assert len(o["res_deviance"]) == n
    assert o["test_kind"] in ("z", "t")
    # binomial / poisson are scale-known → z-test; the rest → t-test.
    if family in ("poisson", "binomial"):
        assert o["test_kind"] == "z"
    else:
        assert o["test_kind"] == "t"


# =============================================================================
# 2. Per-oracle parity (Phase 1-4 of glm-port plan)
# =============================================================================
#
# Per-oracle case construction. Each entry is a dict that knows how to
# rebuild the hea fit corresponding to its R-side oracle. Centralizing
# this keeps the parametrized tests below from each having a custom
# fitting branch.


def _build_iris():
    d = load_dataset("R", "iris")
    return glm("Sepal.Length ~ Petal.Length + Species", d, family=Gaussian())


def _build_insurance_gaussian_log():
    d = load_dataset("MASS", "Insurance").filter(load_dataset("MASS", "Insurance")["Claims"] > 0)
    off = np.log(d["Holders"].to_numpy().astype(float))
    return glm("Claims ~ District + Group", d, family=Gaussian(link="log"),
               offset=off)


def _build_trees(link: str):
    d = load_dataset("R", "trees")
    return glm("Volume ~ log(Height) + log(Girth)", d,
               family=Gamma(link=link))


def _build_quine(link: str):
    d = load_dataset("MASS", "quine")
    return glm("Days ~ Sex + Age + Eth + Lrn", d, family=Poisson(link=link))


def _build_menarche(link: str):
    # cbind(Menarche, Total - Menarche) ~ Age. hea's parser doesn't yet
    # accept cbind() on the LHS (Phase 6.1), so we pre-convert to
    # (proportion, weights=Total) which is the algebraically equivalent
    # binomial-with-size form.
    d = load_dataset("MASS", "menarche")
    d2 = d.with_columns([(d["Menarche"] / d["Total"]).alias("p")])
    return glm("p ~ Age", d2, family=Binomial(link=link),
               weights=d["Total"].to_numpy().astype(float))


def _build_ig_insurance():
    d = load_dataset("MASS", "Insurance").filter(load_dataset("MASS", "Insurance")["Claims"] > 0)
    return glm("Claims ~ Group", d, family=InverseGaussian())


CASES = {
    "gaussian_identity_iris":     _build_iris,
    "gaussian_log_insurance":     _build_insurance_gaussian_log,
    "gamma_inverse_trees":        lambda: _build_trees("inverse"),
    "gamma_log_trees":            lambda: _build_trees("log"),
    "poisson_log_quine":          lambda: _build_quine("log"),
    "poisson_sqrt_quine":         lambda: _build_quine("sqrt"),
    "binomial_logit_menarche":    lambda: _build_menarche("logit"),
    "binomial_probit_menarche":   lambda: _build_menarche("probit"),
    "binomial_cauchit_menarche":  lambda: _build_menarche("cauchit"),
    "binomial_cloglog_menarche":  lambda: _build_menarche("cloglog"),
    "ig_canonical_insurance":     _build_ig_insurance,
}

ALL_ORACLES = list(CASES.keys())


def _allclose(actual, expected, *, atol, rtol=0.0, name=""):
    np.testing.assert_allclose(
        actual, expected, atol=atol, rtol=rtol,
        err_msg=f"{name}: hea={actual} R={expected}",
    )


# Phase 1 — engine + core fit (coef, deviance, fitted/η, residuals, iter).
# Tight tolerances: Fisher IRLS is essentially deterministic.

@pytest.mark.parametrize("oid", ALL_ORACLES)
def test_glm_core_fit(oid: str):
    o = load_glm_oracle(oid)
    m = CASES[oid]()

    assert m.column_names == o["coef_names"]
    assert m.n == o["n"]
    assert m.df_residual == o["df_residual"]
    assert m.converged == o["converged"]
    # IRLS iter count: hea must not blow up vs R, but converging faster is
    # fine — gaussian_log_insurance is the canonical example, where R was
    # given a deliberately poor `start = log(y+1e-3)` to satisfy μ>0 and so
    # needs 11 iters, while hea's default init lands near the optimum and
    # converges in 4. Allow hea ≤ R+2.
    assert m.iter <= o["iter"] + 2, f"iter: hea={m.iter} R={o['iter']}"

    _allclose(m._bhat_arr, np.asarray(o["coefficients"]),
              atol=5e-5, name="coef")
    _allclose(m.deviance, o["deviance"], atol=5e-4, name="deviance")
    _allclose(m.fitted_values, np.asarray(o["fitted_values"]),
              atol=5e-4, name="fitted")
    _allclose(m.linear_predictors, np.asarray(o["linear_pred"]),
              atol=5e-4, name="linear_pred")
    _allclose(m.residuals_of("response"), np.asarray(o["res_response"]),
              atol=5e-4, name="res_response")
    _allclose(m.residuals_of("working"), np.asarray(o["res_working"]),
              atol=5e-4, name="res_working")
    _allclose(m.residuals_of("pearson"), np.asarray(o["res_pearson"]),
              atol=5e-4, name="res_pearson")
    _allclose(m.residuals_of("deviance"), np.asarray(o["res_deviance"]),
              atol=5e-4, name="res_deviance")


# Phase 2 — Wald inference (vcov, SE, t/z, p, CI). The t-vs-z dispatch is
# scale_known-driven; the column header for the printed summary follows
# the same rule.

@pytest.mark.parametrize("oid", ALL_ORACLES)
def test_glm_inference(oid: str):
    o = load_glm_oracle(oid)
    m = CASES[oid]()

    # vcov
    _allclose(m.vcov, np.asarray(o["vcov"]), atol=5e-6, name="vcov")
    _allclose(m._se_bhat_arr, np.asarray(o["std_error"]),
              atol=5e-5, name="se")

    # t / z statistic — column 3 of summary(m)$coefficients
    expected_kind = o["test_kind"]
    assert m._test_kind == expected_kind, \
        f"test kind: hea={m._test_kind} R={expected_kind}"
    _allclose(m._stat_arr, np.asarray(o["test_stat"]),
              atol=5e-3, name="test_stat")

    # p-values — broaden tol on extreme-tail entries (R uses exact tail
    # math; we use scipy's). Anything > 1e-10 should match to 5e-3 abs.
    p_hea = np.asarray(m.p_values.row(0), dtype=float)
    p_R = np.asarray(o["p_value"])
    mask_meaningful = p_R > 1e-10
    if mask_meaningful.any():
        _allclose(p_hea[mask_meaningful], p_R[mask_meaningful],
                  atol=5e-3, name="p_value(meaningful)")
    # Tiny p-values: just assert hea is also tiny.
    if (~mask_meaningful).any():
        assert np.all(p_hea[~mask_meaningful] < 1e-6), \
            f"p-value(tiny): hea={p_hea} R={p_R}"

    # 95% Wald CI — confint.default convention.
    ci_low = m.ci_bhat[m.ci_bhat.columns[1]].to_numpy()
    ci_hi = m.ci_bhat[m.ci_bhat.columns[2]].to_numpy()
    _allclose(ci_low, np.asarray(o["ci_lower"]), atol=5e-4, name="CI lower")
    _allclose(ci_hi, np.asarray(o["ci_upper"]), atol=5e-4, name="CI upper")

    # dispersion — Pearson estimator; 1.0 for scale-known.
    _allclose(m.dispersion, o["dispersion"], atol=5e-5, name="dispersion")


# Phase 3 — null deviance, AIC, BIC, logLik. Family/link-aware AIC routes
# through family.aic + 2·rank (R glm convention; the dispersion df is
# folded into family.aic for unknown-scale families).

@pytest.mark.parametrize("oid", ALL_ORACLES)
def test_glm_aic_bic_loglik(oid: str):
    o = load_glm_oracle(oid)
    m = CASES[oid]()

    _allclose(m.null_deviance, o["null_deviance"], atol=5e-4,
              name="null_deviance")
    assert m.df_null == o["df_null"]
    _allclose(m.aic, o["aic"], atol=5e-3, name="aic")
    _allclose(m.loglike, o["loglik"], atol=5e-3, name="loglik")
    assert m.npar == o["loglik_df"], \
        f"npar (= logLik$df): hea={m.npar} R={o['loglik_df']}"
    _allclose(m.bic, o["bic"], atol=5e-3, name="bic")


# Phase 4 — predict(type=link/response, se.fit=TRUE). The link-scale SE
# is √diag(X·vcov·Xᵀ); the response-scale SE is |dμ/dη(η̂)|·se_link
# (delta method). predict.glm with no newdata reuses the fit-time
# offset so η̂ matches `m$linear.predictors`.

@pytest.mark.parametrize("oid", ALL_ORACLES)
def test_glm_predict(oid: str):
    o = load_glm_oracle(oid)
    m = CASES[oid]()

    # Link scale.
    fit_link, se_link = m.predict(type="link", se_fit=True)
    _allclose(fit_link, np.asarray(o["pred_link_fit"]),
              atol=5e-4, name="pred_link_fit")
    _allclose(se_link, np.asarray(o["pred_link_se"]),
              atol=5e-5, name="pred_link_se")

    # Response scale (delta method on SE). Use rtol on the SE because the
    # delta-method multiplier |dμ/dη| is huge for some links (IG canonical:
    # ~μ³/2 ≈ 1.9e4 on Insurance), which amplifies tiny vcov differences
    # (~1e-7) to ~5e-3 absolute even though relative error stays ~1e-4.
    fit_resp, se_resp = m.predict(type="response", se_fit=True)
    _allclose(fit_resp, np.asarray(o["pred_resp_fit"]),
              atol=5e-4, name="pred_resp_fit")
    _allclose(se_resp, np.asarray(o["pred_resp_se"]),
              atol=5e-5, rtol=5e-4, name="pred_resp_se")

    # Without se_fit, predict returns the bare fit only — same numbers.
    _allclose(m.predict(type="link"), fit_link, atol=0.0, name="link no-se")
    _allclose(m.predict(type="response"), fit_resp, atol=0.0,
              name="response no-se")


# =============================================================================
# 3. Edge cases (Phase 6)
# =============================================================================
#
# Each block here covers one ``glm()`` API surface that's commonly used
# in R but easy to overlook in a port.
#
# Where the value is small and stable we inline an R-computed reference
# (``Rscript -e ...`` produced the literals in the comment beside each
# ``np.testing.assert_allclose`` call). For larger pinned cases we reuse
# the existing JSON oracles via :func:`load_glm_oracle`.


# 6.1 — cbind(success, failure) on LHS for binomial.

def test_cbind_lhs_matches_proportion_weights_form():
    """``cbind(s, f) ~ x`` and ``p ~ x, weights=s+f`` must produce the
    same fit, since hea rewrites the former into the latter internally.
    """
    d = pl.DataFrame({
        "s": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "f": [9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0.1],
        "x": np.arange(10, dtype=float),
    })
    m_cb = glm("cbind(s, f) ~ x", d, family=Binomial())

    # Equivalent rewrite the user could do by hand.
    p = d["s"] / (d["s"] + d["f"])
    w = (d["s"] + d["f"]).to_numpy()
    d2 = d.with_columns(p.alias("p"))
    m_pw = glm("p ~ x", d2, family=Binomial(), weights=w)

    np.testing.assert_allclose(m_cb._bhat_arr, m_pw._bhat_arr, atol=1e-10)
    np.testing.assert_allclose(m_cb.deviance, m_pw.deviance, atol=1e-10)
    np.testing.assert_allclose(m_cb.fitted_values, m_pw.fitted_values, atol=1e-10)


def test_cbind_lhs_matches_menarche_oracle():
    """The R menarche oracle was generated from ``cbind(Menarche, Total -
    Menarche) ~ Age`` with logit. hea's cbind path must reproduce the
    same coefficients and deviance as R."""
    o = load_glm_oracle("binomial_logit_menarche")
    d = load_dataset("MASS", "menarche")
    # cbind(a, b) accepts any expression, including subtraction.
    m = glm("cbind(Menarche, Total - Menarche) ~ Age", d,
            family=Binomial(link="logit"))
    np.testing.assert_allclose(m._bhat_arr, np.asarray(o["coefficients"]),
                               atol=5e-5)
    np.testing.assert_allclose(m.deviance, o["deviance"], atol=5e-4)


def test_cbind_lhs_rejects_non_binomial():
    d = pl.DataFrame({"s": [1.0, 2], "f": [3.0, 4], "x": [1.0, 2]})
    with pytest.raises(ValueError, match="cbind.*Binomial"):
        glm("cbind(s, f) ~ x", d, family=Gaussian())


# 6.2 — offset(...) inside the formula.

def test_formula_offset_matches_kwarg_offset():
    """``y ~ x + offset(log(t))`` must produce the same fit as
    ``y ~ x, offset=log(t)``. Both are valid R syntax and ``glm.fit`` adds
    them together when both are present."""
    d = pl.DataFrame({
        "y": [1, 5, 12, 30, 50, 80],
        "x": [1.0, 2, 3, 4, 5, 6],
        "t": [1.0, 2, 5, 10, 20, 30],
    })
    log_t = np.log(d["t"].to_numpy().astype(float))

    m_form = glm("y ~ x + offset(log(t))", d, family=Poisson())
    m_kw   = glm("y ~ x", d, family=Poisson(), offset=log_t)
    np.testing.assert_allclose(m_form._bhat_arr, m_kw._bhat_arr, atol=1e-12)
    np.testing.assert_allclose(m_form.deviance, m_kw.deviance, atol=1e-12)


def test_formula_offset_sums_with_kwarg_offset():
    """When both are present, R sums them (η = Xβ + Σ formula_offsets +
    kwarg_offset). Verify hea does the same by checking that splitting an
    offset between formula and kwarg gives the same fit as putting it all
    on one side."""
    d = pl.DataFrame({
        "y": [1, 5, 12, 30, 50, 80],
        "x": [1.0, 2, 3, 4, 5, 6],
        "t": [1.0, 2, 5, 10, 20, 30],
    })
    log_t = np.log(d["t"].to_numpy().astype(float))
    half = log_t / 2

    m_split = glm("y ~ x + offset(log(t)/2)", d, family=Poisson(), offset=half)
    m_all   = glm("y ~ x", d, family=Poisson(), offset=log_t)
    np.testing.assert_allclose(m_split._bhat_arr, m_all._bhat_arr, atol=1e-10)


# 6.3 — frequency weights (Phase 1 already plumbed; just pin one oracle).

def test_weighted_poisson_matches_r():
    """``glm(y ~ x, weights=...)`` for Poisson. R-pinned literals from:
        d <- data.frame(y=c(1,2,3,4,5), x=c(1,2,3,4,5))
        coef(glm(y~x, data=d, family=poisson(), weights=c(1,2,1,3,2)))
    """
    d = pl.DataFrame({"y": [1.0, 2, 3, 4, 5], "x": [1.0, 2, 3, 4, 5]})
    w = np.array([1.0, 2, 1, 3, 2])
    m = glm("y ~ x", d, family=Poisson(), weights=w)
    np.testing.assert_allclose(
        m._bhat_arr, [-0.009707952, 0.3360234], atol=5e-7,
    )


# 6.4 — intercept-only.

def test_intercept_only_gaussian():
    """``y ~ 1`` Gaussian: β̂_0 = mean(y); deviance = Σ (y-ȳ)²; df_residual = n-1."""
    y = np.array([1.0, 2, 3, 4, 5])
    d = pl.DataFrame({"y": y})
    m = glm("y ~ 1", d, family=Gaussian())
    np.testing.assert_allclose(m._bhat_arr, [y.mean()], atol=1e-12)
    np.testing.assert_allclose(m.deviance, ((y - y.mean()) ** 2).sum(), atol=1e-12)
    assert m.df_residual == len(y) - 1
    assert m.df_null == len(y) - 1
    # null deviance == residual deviance for the intercept-only model.
    np.testing.assert_allclose(m.null_deviance, m.deviance, atol=1e-12)


def test_intercept_only_poisson():
    """``y ~ 1`` Poisson/log: β̂_0 = log(mean(y))."""
    y = np.array([1, 2, 3, 4, 5, 4, 6, 8])
    d = pl.DataFrame({"y": y})
    m = glm("y ~ 1", d, family=Poisson())
    np.testing.assert_allclose(m._bhat_arr, [np.log(y.mean())], atol=1e-7)


# 6.5 — rank-deficient X.

def test_rank_deficient_x_drops_to_NA_slot():
    """``y ~ x + z`` with z = 2x is rank-deficient. R sets the dropped
    coef to NA, lowers the model rank, and bumps df_residual accordingly.
    hea does the same via the QR pivot."""
    n = 8
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    z = 2 * x          # perfectly collinear
    y = 1 + 0.5 * x + rng.normal(scale=0.1, size=n)
    d = pl.DataFrame({"y": y, "x": x, "z": z})
    m = glm("y ~ x + z", d, family=Gaussian())

    coefs = m._bhat_arr
    # One slot — the second-encountered collinear column — must be NaN.
    assert np.sum(np.isnan(coefs)) == 1, f"expected exactly 1 NaN coef, got {coefs}"
    # The non-NaN coefs are the intercept and (x or z, whichever R kept).
    assert m.rank == 2
    assert m.df_residual == n - m.rank
    # Fit still hits the data: deviance ≈ Σ (y - ŷ)² with x's true slope.
    np.testing.assert_allclose(m.deviance,
                               ((y - m.fitted_values) ** 2).sum(),
                               atol=1e-12)


# 6.6 — factor-response binomial.

def test_factor_response_binomial_string():
    """``y ~ x`` with y a 2-level string column. R uses level-1 = failure
    (=0), level-2 = success (=1), with factor levels in alphabetical order
    when not explicitly declared. R-pinned literals from:
        d <- data.frame(y=factor(c("a","b","a","b","a","b","a","b")),
                        x=c(1.0,2,3,4,5,6,7,8))
        coef(glm(y ~ x, data=d, family=binomial()))
    """
    d = pl.DataFrame({
        "y": ["a", "b", "a", "b", "a", "b", "a", "b"],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })
    m = glm("y ~ x", d, family=Binomial())
    np.testing.assert_allclose(m._bhat_arr, [-0.8822461, 0.1960547], atol=5e-7)
    np.testing.assert_allclose(m.fitted_values[:3],
                               [0.3348809, 0.3798614, 0.4270048], atol=5e-6)


def test_factor_response_binomial_enum_respects_declared_order():
    """``pl.Enum`` declares level order explicitly; R's ``factor(..., levels=)``
    is equivalent. Reverse the levels and the encoding flips sign on β̂_x.
    """
    base = pl.DataFrame({
        "y": ["a", "b", "a", "b", "a", "b", "a", "b"],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })
    # default alphabetical ⇒ "a"=0, "b"=1
    d_ab = base.with_columns(pl.col("y").cast(pl.Enum(["a", "b"])))
    # reversed ⇒ "b"=0, "a"=1
    d_ba = base.with_columns(pl.col("y").cast(pl.Enum(["b", "a"])))
    m_ab = glm("y ~ x", d_ab, family=Binomial())
    m_ba = glm("y ~ x", d_ba, family=Binomial())
    # Reversing the success/failure flips the sign of every coefficient.
    np.testing.assert_allclose(m_ab._bhat_arr, -m_ba._bhat_arr, atol=1e-10)


def test_factor_response_binomial_rejects_three_level():
    d = pl.DataFrame({"y": ["a", "b", "c", "a", "b", "c"],
                      "x": [1.0, 2, 3, 4, 5, 6]})
    with pytest.raises(ValueError, match="2 levels"):
        glm("y ~ x", d, family=Binomial())


def test_factor_response_binomial_boolean():
    d = pl.DataFrame({
        "y": [False, True, False, True, False, True, False, True],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })
    m = glm("y ~ x", d, family=Binomial())
    # FALSE=0, TRUE=1 — same as factor("a","b") above (alphabetical).
    np.testing.assert_allclose(m._bhat_arr, [-0.8822461, 0.1960547], atol=5e-7)


# 6.7 — Quasi family: variance="mu" + log link == quasi-Poisson. Same
# point estimates as Poisson; dispersion estimated; t-tests for Wald.

def test_quasi_poisson_matches_poisson_betas_with_estimated_dispersion():
    d = load_dataset("MASS", "quine")  # over-dispersed counts — Wood §3.3.5 territory
    m_pois = glm("Days ~ Sex + Age + Eth + Lrn", d, family=Poisson(link="log"))
    m_quasi = glm("Days ~ Sex + Age + Eth + Lrn", d, family=Quasi(link="log", variance="mu"))

    # Point estimates must be identical (same IRLS path; scale only enters SE).
    np.testing.assert_allclose(m_quasi._bhat_arr, m_pois._bhat_arr, atol=1e-10)
    # Same deviance and df.
    np.testing.assert_allclose(m_quasi.deviance, m_pois.deviance, atol=1e-10)
    assert m_quasi.df_residual == m_pois.df_residual

    # Dispersion: Poisson is fixed at 1; Quasi is the Pearson chi^2 / df_resid.
    assert m_pois.dispersion == 1.0
    assert m_quasi.dispersion != 1.0
    assert m_quasi.dispersion > 1.0  # quine is overdispersed

    # SE(quasi) == sqrt(disp) * SE(poisson).
    np.testing.assert_allclose(
        m_quasi._se_bhat_arr,
        np.sqrt(m_quasi.dispersion) * m_pois._se_bhat_arr,
        atol=1e-10,
    )

    # Quasi has no proper likelihood — AIC/BIC/logLik are NaN.
    assert np.isnan(m_quasi.aic) and np.isnan(m_quasi.bic) and np.isnan(m_quasi.loglike)
    # Wald tests use t-distribution because scale is unknown.
    assert m_quasi._test_kind == "t"
    assert m_pois._test_kind == "z"


def test_quasi_rejects_unknown_variance():
    with pytest.raises(ValueError, match="variance must be"):
        Quasi(variance="bogus")


# =============================================================================
# 4. compare integration (Phase 5 — anova / AIC / BIC)
# =============================================================================
#
# Pins ``anova(glm1, glm2[, ...])`` against the ``anova.glm()`` oracles
# generated by ``tests/scripts/make_glm_oracles.R``. Test selection
# mirrors R's defaults: scale-known families (Poisson/Binomial) → Chisq
# LRT; unknown-scale families (Gaussian/Gamma/IG) → F-test using the
# largest model's Pearson dispersion as denominator.
#
# ``AIC()`` / ``BIC()`` already accept anything with ``.AIC``/``.BIC``/
# ``.npar``; we just smoke-test that they don't blow up on a glm fit
# and that the printed ``df`` and ``AIC`` values match the per-model
# oracles.


def _fits_anova_poisson_quine():
    d = load_dataset("MASS", "quine")
    fam = Poisson(link="log")
    return [
        glm("Days ~ Sex + Age", d, family=fam),
        glm("Days ~ Sex + Age + Eth + Lrn", d, family=fam),
    ]


def _fits_anova_gamma_trees():
    d = load_dataset("R", "trees")
    fam = Gamma(link="inverse")
    return [
        glm("Volume ~ log(Girth)", d, family=fam),
        glm("Volume ~ log(Height) + log(Girth)", d, family=fam),
    ]


def _fits_anova_gaussian_iris():
    d = load_dataset("R", "iris")
    fam = Gaussian()
    return [
        glm("Sepal.Length ~ 1", d, family=fam),
        glm("Sepal.Length ~ Petal.Length", d, family=fam),
        glm("Sepal.Length ~ Petal.Length + Species", d, family=fam),
    ]


def _fits_anova_binomial_menarche():
    # Same proportion+weights rewrite the per-oracle parity tests use
    # for cbind.
    d = load_dataset("MASS", "menarche")
    p = (d["Menarche"] / d["Total"]).rename("p")
    d2 = d.with_columns(p)
    w = d["Total"].to_numpy().astype(float)
    fam = Binomial(link="logit")
    return [
        glm("p ~ 1", d2, family=fam, weights=w),
        glm("p ~ Age", d2, family=fam, weights=w),
    ]


ANOVA_CASES = {
    "anova_poisson_quine":     _fits_anova_poisson_quine,
    "anova_gamma_trees":       _fits_anova_gamma_trees,
    "anova_gaussian_iris":     _fits_anova_gaussian_iris,
    "anova_binomial_menarche": _fits_anova_binomial_menarche,
}


@pytest.mark.parametrize("oid", list(ANOVA_CASES.keys()))
def test_anova_glm(oid: str):
    o = load_glm_oracle(oid)
    fits = ANOVA_CASES[oid]()

    # Public anova() prints and returns None; call the builder directly so
    # we can assert on column values.
    labels = [f"m{i}" for i in range(len(fits))]
    df, _ = _anova_glm_table(*fits, labels=labels, test=None)
    assert isinstance(df, pl.DataFrame)

    # Per-row residual df / dev — order should match the oracle (sorted by
    # df_residuals descending, which equals the input order for nested fits).
    np.testing.assert_array_equal(
        np.asarray(df["Resid. Df"].to_numpy()),
        np.asarray(o["resid_df"], dtype=int),
    )
    np.testing.assert_allclose(
        df["Resid. Dev"].to_numpy(),
        np.asarray(o["resid_dev"]),
        atol=5e-3,
    )

    # Incremental columns: Df / Deviance / stat / p — first row is None.
    test = o["test"]
    stat_col = "F" if test == "F" else "Deviance"
    p_col = "Pr(>F)" if test == "F" else "Pr(>Chi)"

    df_hea = df["Df"].to_list()
    dev_hea = df["Deviance"].to_list()
    stat_hea = df[stat_col].to_list()
    p_hea = df[p_col].to_list()
    for i in range(len(o["df"])):
        df_R, dev_R = o["df"][i], o["deviance"][i]
        stat_R, p_R = o["stat"][i], o["pvalue"][i]
        if df_R is None:
            assert df_hea[i] is None and dev_hea[i] is None, \
                f"row {i}: expected None for first row"
            assert stat_hea[i] is None and p_hea[i] is None
            continue
        assert df_hea[i] == df_R, \
            f"row {i}: Df hea={df_hea[i]} R={df_R}"
        np.testing.assert_allclose(dev_hea[i], dev_R, atol=5e-3,
            err_msg=f"row {i}: Deviance hea={dev_hea[i]} R={dev_R}")
        np.testing.assert_allclose(stat_hea[i], stat_R, atol=5e-3,
            err_msg=f"row {i}: {stat_col} hea={stat_hea[i]} R={stat_R}")
        # p-value: meaningful values to 5e-3; tiny values just need to be tiny.
        if p_R > 1e-10:
            np.testing.assert_allclose(p_hea[i], p_R, atol=5e-3,
                err_msg=f"row {i}: {p_col} hea={p_hea[i]} R={p_R}")
        else:
            assert p_hea[i] < 1e-6, \
                f"row {i}: {p_col} hea={p_hea[i]} should be tiny like R={p_R}"


# AIC()/BIC() smoke — the helpers print and return None; we just
# confirm they don't raise on glm fits and that the per-model values
# shown agree with the per-model oracles already pinned in section 2.

def test_aic_bic_dispatch_on_glm():
    fits = _fits_anova_gaussian_iris()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        AIC(*fits)
        BIC(*fits)
    out = buf.getvalue()
    # Each model's AIC and BIC should appear in the printed table to 2dp,
    # which is the rounding the helpers apply.
    for m in fits:
        assert f"{m.AIC:.2f}" in out, f"AIC table missing {m.AIC:.2f}"
        assert f"{m.BIC:.2f}" in out, f"BIC table missing {m.BIC:.2f}"


def test_anova_rejects_mixed_types():
    from hea import lm
    d = load_dataset("R", "iris")
    m_lm = lm("Sepal.Length ~ Petal.Length", d)
    m_glm = glm("Sepal.Length ~ Petal.Length", d, family=Gaussian())
    with pytest.raises(TypeError, match="same type"):
        anova(m_lm, m_glm)


def test_anova_rejects_mixed_families():
    d = load_dataset("R", "trees")
    m1 = glm("Volume ~ log(Girth)", d, family=Gamma(link="inverse"))
    m2 = glm("Volume ~ log(Girth)", d, family=Gamma(link="log"))
    with pytest.raises(ValueError, match="family and link"):
        anova(m1, m2)


def test_anova_glm_test_argument():
    """`test=` switches the statistic — pinned to R's anova.glm output on
    Wood's heart data (cbind(ha, ok) ~ ck via the proportion + weights form)."""
    heart = pl.DataFrame({
        "ck": [20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460],
        "ha": [2, 13, 30, 30, 21, 19, 18, 13, 19, 15, 7, 8],
        "ok": [88, 26, 8, 5, 0, 1, 1, 1, 1, 0, 0, 0],
    }).with_columns(
        n=pl.col("ha") + pl.col("ok"),
        p=pl.col("ha") / (pl.col("ha") + pl.col("ok")),
    )
    n = heart["n"].to_numpy()
    m0 = glm("p ~ 1",  data=heart, family=Binomial(link="logit"), weights=n)
    m1 = glm("p ~ ck", data=heart, family=Binomial(link="logit"), weights=n)

    # Default + explicit Chisq + LRT alias all give the same table
    for kw in [{}, {"test": "Chisq"}, {"test": "LRT"}, {"test": "lrt"}]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            anova(m0, m1, **kw)
        out = buf.getvalue()
        assert "Pr(>Chi)" in out
        assert "234.78" in out  # R's deviance, matches at 2dp

    # test='F' produces an F column
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        anova(m0, m1, test="F")
    out = buf.getvalue()
    assert "Pr(>F)" in out
    assert "234.78" in out

    # test='Rao' is the score test, not implemented yet
    with pytest.raises(NotImplementedError, match="Rao"):
        anova(m0, m1, test="Rao")

    # Bogus value
    with pytest.raises(ValueError, match="must be"):
        anova(m0, m1, test="bogus")


def test_anova_glm_F_on_scale_known_pins_to_R():
    """`test='F'` on a scale-known family (Poisson) overrides the default
    Chisq. Exercises the F branch with ``dispersion_full = 1``, so
    ``F = Δdev / Δdf``. Pinned to R's ``anova.glm(..., test='F')`` on
    MASS::quine — a path the auto-detect oracle doesn't cover.
    """
    d = load_dataset("MASS", "quine")
    fam = Poisson(link="log")
    m0 = glm("Days ~ Sex + Age", d, family=fam)
    m1 = glm("Days ~ Sex + Age + Eth + Lrn", d, family=fam)
    df, _ = _anova_glm_table(m0, m1, labels=["m0", "m1"], test="F")
    # Schema: F column appears under test='F', not under default Chisq.
    assert "F" in df.columns and "Pr(>F)" in df.columns
    assert "Pr(>Chi)" not in df.columns
    # Numerics from `anova(m0, m1, test='F')`:
    #   Df=2  Deviance=211.5687  F=105.78436  Pr(>F) < 2.22e-16
    assert df["Df"][1] == 2
    np.testing.assert_allclose(df["Deviance"][1], 211.5687, atol=5e-3)
    np.testing.assert_allclose(df["F"][1], 105.78436, atol=5e-3)
    # R reports as "< 2.22e-16"; SciPy computes the upper tail more precisely.
    assert df["Pr(>F)"][1] < 1e-20


def test_anova_glm_F_three_model_uses_full_dispersion():
    """3+ models with ``test='F'``: the F denominator is locked to the
    largest (full) model's dispersion across all rows — not the
    immediately-preceding row's dispersion. Pinned to R's
    ``anova(m0, m1, m2, test='F')`` on Gamma trees.
    """
    d = load_dataset("R", "trees")
    fam = Gamma(link="inverse")
    m0 = glm("Volume ~ 1", d, family=fam)
    m1 = glm("Volume ~ log(Girth)", d, family=fam)
    m2 = glm("Volume ~ log(Height) + log(Girth)", d, family=fam)
    df, _ = _anova_glm_table(m0, m1, m2, labels=["m0", "m1", "m2"], test="F")
    # Resid Df / Dev across the three models.
    assert df["Resid. Df"].to_list() == [30, 29, 28]
    np.testing.assert_allclose(
        df["Resid. Dev"].to_numpy(),
        [8.3172, 0.8592, 0.8002],
        atol=5e-4,
    )
    # Row 1 (m1 vs m0): F uses m2's dispersion (0.02660), so
    #   F = (7.4580 / 1) / 0.02660 ≈ 280.36, not (7.4580/1) / m1.dispersion.
    assert df["Df"][1] == 1
    np.testing.assert_allclose(df["Deviance"][1], 7.4580, atol=5e-4)
    np.testing.assert_allclose(df["F"][1], 280.35781, atol=5e-3)
    np.testing.assert_allclose(df["Pr(>F)"][1], 4.0473e-16, rtol=5e-2)
    # Row 2 (m2 vs m1): F = (0.05902 / 1) / 0.02660 ≈ 2.219.
    assert df["Df"][2] == 1
    np.testing.assert_allclose(df["Deviance"][2], 0.05902, atol=5e-4)
    np.testing.assert_allclose(df["F"][2], 2.21882, atol=5e-4)
    np.testing.assert_allclose(df["Pr(>F)"][2], 0.14752, atol=5e-4)


def test_anova_glm_F_explicit_matches_auto_on_unknown_scale():
    """Sanity: ``test='F'`` and ``test=None`` produce identical numerics
    on unknown-scale families (where ``None`` auto-resolves to F).
    Locks the equivalence so future refactors of the test-selection
    branch can't drift one path away from the other.
    """
    d = load_dataset("R", "trees")
    fam = Gamma(link="inverse")
    m0 = glm("Volume ~ log(Girth)", d, family=fam)
    m1 = glm("Volume ~ log(Height) + log(Girth)", d, family=fam)
    df_auto, _ = _anova_glm_table(m0, m1, labels=["m0", "m1"], test=None)
    df_F, _    = _anova_glm_table(m0, m1, labels=["m0", "m1"], test="F")
    assert df_auto.columns == df_F.columns
    for col in ["Resid. Df", "Resid. Dev", "Df", "Deviance", "F", "Pr(>F)"]:
        a, b = df_auto[col].to_list(), df_F[col].to_list()
        for x, y in zip(a, b):
            if x is None:
                assert y is None
            else:
                np.testing.assert_allclose(x, y, rtol=0, atol=0)


def test_anova_glm_F_printed_table_has_F_and_Pr_columns():
    """End-to-end: the public ``anova(..., test='F')`` printed table
    has the F-test header columns and locked numerics — guards against
    refactors of ``_anova_glm`` (the printer) drifting from
    ``_anova_glm_table`` (the builder)."""
    d = load_dataset("R", "trees")
    fam = Gamma(link="inverse")
    m0 = glm("Volume ~ log(Girth)", d, family=fam)
    m1 = glm("Volume ~ log(Height) + log(Girth)", d, family=fam)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        anova(m0, m1, test="F")
    out = buf.getvalue()
    # Header must include the F-test columns, not Chisq.
    assert "F" in out and "Pr(>F)" in out
    assert "Pr(>Chi)" not in out
    # F-stat from R: 2.219 (rounded to 4 places by the formatter).
    assert "2.2188" in out


def test_anova_gam_single_pins_to_mgcv_on_trees():
    """``anova(gam_single)`` should produce mgcv's anova.gam single-model
    output: parametric Terms F-table + smooth significance table."""
    from hea import gam
    trees = load_dataset("mgcv", "trees").with_columns(
        Hclass=((pl.col("Height") / 10).floor() - 5)
            .cast(pl.Int64)
            .replace_strict([1, 2, 3], ["small", "medium", "large"],
                            return_dtype=pl.Enum(["small", "medium", "large"])),
    )
    ct7 = gam("Volume ~ Hclass + s(Girth)",
              family=Gamma(link="log"), data=trees)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        anova(ct7)
    out = buf.getvalue()
    # Header
    assert "Family: Gamma" in out
    assert "Volume ~ Hclass + s(Girth)" in out
    # Parametric table — pinned to mgcv (df=2, F=6.802, p≈0.00428)
    assert "Parametric Terms:" in out
    assert "Hclass" in out
    assert "6.802" in out  # F-stat
    assert "0.00428" in out  # p-value
    # Smooth table — pinned to mgcv (edf=2.444, Ref.df=3.076, F=152.7)
    assert "Approximate significance of smooth terms:" in out
    assert "s(Girth)" in out
    assert "2.444" in out
    assert "3.076" in out
    assert "152.7" in out


def test_anova_lm_rejects_test_argument():
    """`test=` is glm-only; lm/lme always use F/Chisq respectively."""
    from hea import lm
    d = load_dataset("R", "iris")
    m1 = lm("Sepal.Length ~ Petal.Length", d)
    m2 = lm("Sepal.Length ~ Petal.Length + Petal.Width", d)
    with pytest.raises(TypeError, match="test="):
        anova(m1, m2, test="Chisq")
