"""
Notebook examples → regression tests for hea.lm.

Pins the printed numerical outputs of example/lm.ipynb so the
formulaic→formula.py migration can be validated against book-standard
results (Faraway 2014, Wood 2017, Gelman & Hill 2007, Breheny & Burchett 2017).

One textbook formula (Wood §1.5 sc.mod2) is rewritten from formulaic's
`count ~ time.ipc + {prop.partner*time.ipc}` to the R form
`count ~ time.ipc + I(prop.partner*time.ipc)` so it matches Wood verbatim
and uses the syntax formula.py is designed for.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import f as f_dist

from conftest import load_dataset
from hea.models import lm


def _assert_coef(m, col, est, se=None, tval=None, pval=None):
    if col not in m.bhat.columns:
        raise KeyError(f"{col!r} not in {list(m.bhat.columns)!r}")
    np.testing.assert_allclose(m.bhat[col][0], est, atol=5e-3)
    if se is not None:
        np.testing.assert_allclose(m.se_bhat[col][0], se, atol=5e-3)
    if tval is not None:
        np.testing.assert_allclose(m.t_values[col][0], tval, atol=5e-3)
    if pval is not None:
        if pval == 0.0:
            assert m.p_values[col][0] < 1e-3
        else:
            np.testing.assert_allclose(m.p_values[col][0], pval, atol=5e-3)


def _assert_summary(m, *, n, p, df_residuals, sigma, r2, r2adj,
                    fstats, f_pvalue, loglike, AIC, BIC):
    assert m.n == n
    assert m.p == p
    assert m.df_residuals == df_residuals
    np.testing.assert_allclose(m.sigma, sigma, atol=5e-3)
    np.testing.assert_allclose(m.r_squared, r2, atol=5e-4)
    np.testing.assert_allclose(m.r_squared_adjusted, r2adj, atol=5e-4)
    if fstats is None:
        assert m.fstats is None
        assert m.f_p_value is None
    else:
        np.testing.assert_allclose(m.fstats, fstats, atol=5e-3)
        # Notebook prints small p-values in scientific notation (use rtol)
        # and larger ones rounded to 2 decimals (use atol).
        if abs(f_pvalue) < 1e-3:
            np.testing.assert_allclose(m.f_p_value, f_pvalue, rtol=1e-2)
        else:
            np.testing.assert_allclose(m.f_p_value, f_pvalue, atol=1e-2)
    np.testing.assert_allclose(m.loglike, loglike, atol=5e-3)
    np.testing.assert_allclose(m.AIC, AIC, atol=5e-3)
    np.testing.assert_allclose(m.BIC, BIC, atol=5e-3)


def _f_test(m_reduced, m_full):
    rss0, rss1 = m_reduced.rss, m_full.rss
    df0, df1 = m_reduced.df_residuals, m_full.df_residuals
    F = ((rss0 - rss1) / (df0 - df1)) / (rss1 / df1)
    p = f_dist.sf(F, df0 - df1, df1)
    return F, p


# ---------------------------------------------------------------------------
# Faraway (2014)
# ---------------------------------------------------------------------------


def test_faraway_2_6_gala():
    gala = load_dataset("faraway", "gala")
    m = lm("Species ~ Area + Adjacent + Elevation + Nearest + Scruz", gala)

    _assert_summary(
        m, n=30, p=6, df_residuals=24, sigma=60.975,
        r2=0.7658, r2adj=0.7171,
        fstats=15.6994, f_pvalue=6.837893e-07,
        loglike=-162.5350, AIC=339.0700, BIC=348.8784,
    )
    _assert_coef(m, "(Intercept)",  7.068, 19.154,  0.369, 0.715)
    _assert_coef(m, "Area",      -0.024,  0.022, -1.068, 0.296)
    _assert_coef(m, "Adjacent",  -0.075,  0.018, -4.226, 0.000)
    _assert_coef(m, "Elevation",  0.319,  0.054,  5.953, 0.000)
    _assert_coef(m, "Nearest",    0.009,  1.054,  0.009, 0.993)
    _assert_coef(m, "Scruz",     -0.241,  0.215, -1.117, 0.275)


def test_faraway_2_10_odor():
    odor = load_dataset("faraway", "odor")
    m = lm("odor ~ temp + gas + pack", odor)

    _assert_summary(
        m, n=15, p=4, df_residuals=11, sigma=36.012,
        r2=0.3337, r2adj=0.1519,
        fstats=1.8361, f_pvalue=0.20,
        loglike=-72.7155, AIC=155.4310, BIC=158.9713,
    )
    _assert_coef(m, "(Intercept)",  15.200,  9.298,  1.635, 0.130)
    _assert_coef(m, "temp",      -12.125, 12.732, -0.952, 0.361)
    _assert_coef(m, "gas",       -17.000, 12.732, -1.335, 0.209)
    _assert_coef(m, "pack",      -21.375, 12.732, -1.679, 0.121)

    # cor=True path: odor is an orthogonal design — feature corr ≈ I.
    feats = [c for c in m.X.columns if c != "(Intercept)"]
    corr = m.X[feats].corr().to_numpy()
    np.testing.assert_allclose(corr, np.eye(len(feats)), atol=1e-2)


def test_faraway_3_2_anova_gala():
    gala = load_dataset("faraway", "gala")
    m_full = lm("Species ~ Area + Adjacent + Elevation + Nearest + Scruz", gala)
    m_null = lm("Species ~ 1", gala)
    m2 = lm("Species ~ Adjacent + Elevation + Nearest + Scruz", gala)
    m3 = lm("Species ~ Elevation + Nearest + Scruz", gala)
    m4 = lm("Species ~ I(Area+Adjacent) + Elevation + Nearest + Scruz", gala)

    assert m_full.df_residuals == 24
    np.testing.assert_allclose(m_full.rss, 89231.366, atol=1e-2)

    # null vs full: F = 15.699, p ≈ 0
    assert m_null.df_residuals == 29
    np.testing.assert_allclose(m_null.rss, 381081.367, atol=1e-2)
    F, p = _f_test(m_null, m_full)
    np.testing.assert_allclose(F, 15.699, atol=5e-3)
    assert p < 1e-3

    # m2 vs full: F = 1.140, p = 0.296
    assert m2.df_residuals == 25
    np.testing.assert_allclose(m2.rss, 93469.084, atol=1e-2)
    F, p = _f_test(m2, m_full)
    np.testing.assert_allclose(F, 1.140, atol=5e-3)
    np.testing.assert_allclose(p, 0.296, atol=5e-3)

    # m3 vs full: F = 9.287, p = 0.001
    assert m3.df_residuals == 26
    np.testing.assert_allclose(m3.rss, 158291.629, atol=1e-2)
    F, p = _f_test(m3, m_full)
    np.testing.assert_allclose(F, 9.287, atol=5e-3)
    np.testing.assert_allclose(p, 0.001, atol=5e-4)

    # m4 vs full: F = 5.476, p = 0.028  (uses I(Area+Adjacent))
    assert m4.df_residuals == 25
    np.testing.assert_allclose(m4.rss, 109591.121, atol=1e-2)
    F, p = _f_test(m4, m_full)
    np.testing.assert_allclose(F, 5.476, atol=5e-3)
    np.testing.assert_allclose(p, 0.028, atol=5e-3)


def test_faraway_4_2_fat_predict():
    fat = load_dataset("faraway", "fat")
    formula = (
        "brozek ~ age + weight + height + neck + chest + abdom + hip "
        "+ thigh + knee + ankle + biceps + forearm + wrist"
    )
    m = lm(formula, data=fat)

    _assert_summary(
        m, n=252, p=14, df_residuals=238, sigma=3.988,
        r2=0.7490, r2adj=0.7353,
        fstats=54.6255, f_pvalue=7.980828e-64,
        loglike=-698.9579, AIC=1427.9158, BIC=1480.8573,
    )

    # Prediction at the per-feature median (notebook 4.2 cells 1cae/ee3a).
    x0 = m.X.median()
    pred_pi = m.predict(newdata=x0, interval="prediction")
    np.testing.assert_allclose(pred_pi["fit"][0], 17.49322, atol=5e-4)
    np.testing.assert_allclose(pred_pi["lwr"][0],  9.61783, atol=5e-4)
    np.testing.assert_allclose(pred_pi["upr"][0], 25.36861, atol=5e-4)

    pred_ci = m.predict(newdata=x0, interval="confidence")
    np.testing.assert_allclose(pred_ci["fit"][0], 17.49322, atol=5e-4)
    np.testing.assert_allclose(pred_ci["lwr"][0], 16.94426, atol=5e-4)
    np.testing.assert_allclose(pred_ci["upr"][0], 18.04219, atol=5e-4)


# ---------------------------------------------------------------------------
# Wood (2017)
# ---------------------------------------------------------------------------


def test_wood_1_5_sperm_main_effects():
    df = load_dataset("gamair", "sperm.comp1")
    m = lm("count ~ time.ipc + prop.partner", df)

    _assert_summary(
        m, n=15, p=3, df_residuals=12, sigma=136.609,
        r2=0.4573, r2adj=0.3669,
        fstats=5.0562, f_pvalue=0.03,
        loglike=-93.3673, AIC=194.7346, BIC=197.5668,
    )
    _assert_coef(m, "(Intercept)",     357.418,  88.082,  4.058, 0.002)
    _assert_coef(m, "time.ipc",        1.942,   0.907,  2.141, 0.053)
    _assert_coef(m, "prop.partner", -339.560, 126.253, -2.690, 0.020)


def test_wood_1_5_sperm_interaction_with_I():
    """Wood §1.5.2 sc.mod2 — uses I() exactly as in the textbook."""
    df = load_dataset("gamair", "sperm.comp1")
    m = lm("count ~ time.ipc + I(prop.partner*time.ipc)", df)

    _assert_summary(
        m, n=15, p=3, df_residuals=12, sigma=128.023,
        r2=0.5234, r2adj=0.4440,
        fstats=6.5888, f_pvalue=0.01,
        loglike=-92.3937, AIC=192.7874, BIC=195.6196,
    )
    _assert_coef(m, "I(prop.partner * time.ipc)", -5.478, 1.741, -3.146, 0.008)
    _assert_coef(m, "(Intercept)", 140.470, 64.063, 2.193, 0.049)
    _assert_coef(m, "time.ipc",    5.618,  1.549, 3.626, 0.003)


def test_wood_1_5_AIC_table():
    df = load_dataset("gamair", "sperm.comp1")
    scmod1 = lm("count ~ time.ipc + prop.partner", df)
    scmod3 = lm("count ~ prop.partner", df)
    scmod4 = lm("count ~ 1", df)

    np.testing.assert_allclose(scmod1.AIC, 194.73, atol=5e-3)
    np.testing.assert_allclose(scmod3.AIC, 197.59, atol=5e-3)
    np.testing.assert_allclose(scmod4.AIC, 199.90, atol=5e-3)
    # AIC() table reports df = p + 1 (R convention; see lm.AIC()).
    assert scmod1.p + 1 == 4
    assert scmod3.p + 1 == 3
    assert scmod4.p + 1 == 2


def test_wood_1_6_4_plantgrowth():
    df = load_dataset("R", "PlantGrowth")
    pgm1 = lm("weight ~ group", data=df)
    pgm0 = lm("weight ~ 1", df)

    _assert_summary(
        pgm1, n=30, p=3, df_residuals=27, sigma=0.623,
        r2=0.2641, r2adj=0.2096,
        fstats=4.8461, f_pvalue=0.02,
        loglike=-26.8095, AIC=61.6190, BIC=67.2238,
    )
    _assert_coef(pgm1, "(Intercept)", 5.032, 0.197, 25.527, 0.000)
    _assert_coef(pgm1, "grouptrt1", -0.371, 0.279, -1.331, 0.194)
    _assert_coef(pgm1, "grouptrt2",  0.494, 0.279,  1.772, 0.088)

    _assert_summary(
        pgm0, n=30, p=1, df_residuals=29, sigma=0.701,
        r2=0.0, r2adj=0.0,
        fstats=None, f_pvalue=None,
        loglike=-31.4104, AIC=66.8208, BIC=69.6232,
    )
    _assert_coef(pgm0, "(Intercept)", 5.073, 0.128, 39.627, 0.0)

    # anova(pgm0, pgm1)
    np.testing.assert_allclose(pgm0.rss, 14.258, atol=5e-3)
    np.testing.assert_allclose(pgm1.rss, 10.492, atol=5e-3)
    F, p = _f_test(pgm0, pgm1)
    np.testing.assert_allclose(F, 4.846, atol=5e-3)
    np.testing.assert_allclose(p, 0.016, atol=5e-3)


# ---------------------------------------------------------------------------
# Gelman & Hill (2007)
# ---------------------------------------------------------------------------


def test_gelman_3_4_kidiq_two_predictors():
    df = load_dataset("rstanarm", "kidiq")
    m = lm("kid.score ~ mom.hs + mom.iq", df)

    _assert_summary(
        m, n=434, p=3, df_residuals=431, sigma=18.136,
        r2=0.2141, r2adj=0.2105,
        fstats=58.7241, f_pvalue=2.793258e-23,
        loglike=-1871.9945, AIC=3751.9890, BIC=3768.2812,
    )
    _assert_coef(m, "(Intercept)", 25.732, 5.875, 4.380, 0.000)
    _assert_coef(m, "mom.hs",     5.950, 2.212, 2.690, 0.007)
    _assert_coef(m, "mom.iq",     0.564, 0.061, 9.309, 0.000)


def test_gelman_3_5_kidiq_iq_only():
    df = load_dataset("rstanarm", "kidiq")
    m = lm("kid.score ~ mom.iq", df)

    _assert_summary(
        m, n=434, p=2, df_residuals=432, sigma=18.266,
        r2=0.2010, r2adj=0.1991,
        fstats=108.6428, f_pvalue=7.661950e-23,
        loglike=-1875.6079, AIC=3757.2158, BIC=3769.4349,
    )
    _assert_coef(m, "(Intercept)", 25.80, 5.917,  4.360, 0.0)
    _assert_coef(m, "mom.iq",     0.61, 0.059, 10.423, 0.0)


def test_gelman_3_5_kidiq_interaction():
    df = load_dataset("rstanarm", "kidiq")
    m = lm("kid.score ~ mom.hs + mom.iq + mom.hs:mom.iq", df)

    _assert_summary(
        m, n=434, p=4, df_residuals=430, sigma=17.971,
        r2=0.2301, r2adj=0.2247,
        fstats=42.8389, f_pvalue=3.066596e-24,
        loglike=-1867.5429, AIC=3745.0857, BIC=3765.4510,
    )
    _assert_coef(m, "(Intercept)",   -11.482, 13.758, -0.835, 0.404)
    _assert_coef(m, "mom.hs",       51.268, 15.338,  3.343, 0.001)
    _assert_coef(m, "mom.iq",        0.969,  0.148,  6.531, 0.000)
    _assert_coef(m, "mom.hs:mom.iq", -0.484, 0.162, -2.985, 0.003)


# ---------------------------------------------------------------------------
# Visreg (Breheny & Burchett, 2017)
# ---------------------------------------------------------------------------


def test_breheny_airquality():
    # NA-omit: Ozone has 37 NAs, Solar.R has 7. R-style na.omit on referenced
    # columns leaves 111 rows.
    df = load_dataset("R", "airquality")
    m = lm("Ozone ~ Solar.R + Wind + Temp", data=df)

    _assert_summary(
        m, n=111, p=4, df_residuals=107, sigma=21.181,
        r2=0.6059, r2adj=0.5948,
        fstats=54.8337, f_pvalue=1.508994e-21,
        loglike=-494.3586, AIC=998.7171, BIC=1012.2648,
    )
    _assert_coef(m, "(Intercept)", -64.342, 23.055, -2.791, 0.006)
    _assert_coef(m, "Solar.R",     0.060,  0.023,  2.580, 0.011)
    _assert_coef(m, "Wind",       -3.334,  0.654, -5.094, 0.000)
    _assert_coef(m, "Temp",        1.652,  0.254,  6.516, 0.000)


def test_wood_2_1_1_stomata_rank_deficient_anova():
    """Wood 2017 §2.1.1 stomata: `tree` is nested in `CO2` (3 trees per
    level), so `area ~ CO2 + tree` has rank 6 (not 7) — R's lm() drops
    one aliased tree dummy via dqrdc2 pivoting. Verify the F-test in
    anova(m0, m1) matches the book: Df=4 (not 5), F=6.665 (not 5.025),
    Res.Df_full=18 (not 17)."""
    from hea.R import anova  # noqa: F401 — keeps import close to use
    df = load_dataset("gamair", "stomata")
    m1 = lm("area ~ CO2 + tree", data=df)
    m0 = lm("area ~ CO2", data=df)
    assert m1.df_residuals == 18
    assert m0.df_residuals == 22
    assert len(m1._aliased_cols) == 1
    # R surfaces the alias only when the model is printed — check the
    # header line appears in repr() (hea's summary() prints to stdout).
    assert "(1 not defined because of singularities)" in repr(m1)
    np.testing.assert_allclose(m1.rss, 0.8604, atol=5e-3)
    np.testing.assert_allclose(m0.rss, 2.1348, atol=5e-3)
    f_stat = ((m0.rss - m1.rss) / 4) / (m1.rss / m1.df_residuals)
    np.testing.assert_allclose(f_stat, 6.6654, atol=5e-3)


def test_wood_2_1_1_stomata_anova_single_model_sequential(capsys):
    """anova(lm) for a single fit — sequential (Type I) SS, R parity.
    Wood §2.1.1: anova(lm(area ~ CO2 + tree)) decomposes into CO2 + tree
    incremental F-tests; pinned to R's anova.lm output."""
    from hea.R import anova
    df = load_dataset("gamair", "stomata")
    m = lm("area ~ CO2 + tree", data=df)
    anova(m)
    out = capsys.readouterr().out
    # CO2 row
    assert "CO2" in out and "184.5452" in out and "6.686e-11" in out
    # tree row
    assert "tree" in out and "6.6654" in out and "0.001788" in out
    # Residuals row uses df=18 / SS=0.8604
    assert "Residuals" in out and "0.8604" in out


# ---------------------------------------------------------------------------
# Weighted least squares — parity with R's lm(..., weights=) inference
#
# R's summary.lm uses the *weighted* residual sum of squares (Σ wᵢrᵢ²) for
# the residual variance, a weighted TSS for R², and adds Σ log wᵢ to the
# Gaussian log-likelihood; zero-weight rows drop out of the effective
# sample size. Oracle values below are from R 4.6.0 (hardcoded; CI never
# calls R):
#
#   m1 <- lm(mpg ~ wt + hp, data = mtcars, weights = 1/wt)
#   m2 <- lm(Ozone ~ Solar.R + Wind + Temp, data = airquality, weights = 1/Wind)
# ---------------------------------------------------------------------------


def test_weighted_lm_mtcars_summary_matches_R():
    mt = load_dataset("R", "mtcars")
    w = 1.0 / mt["wt"].to_numpy()
    m = lm("mpg ~ wt + hp", mt, weights=w)

    _assert_summary(
        m, n=32, p=3, df_residuals=29, sigma=1.5539921,
        r2=0.83887884, r2adj=0.82776704,
        fstats=75.494389, f_pvalue=3.189423998e-12,
        loglike=-75.885285, AIC=159.77057, BIC=165.63351,
    )
    _assert_coef(m, "(Intercept)", 39.002317,   1.5414618,    25.302163,  0.0)
    _assert_coef(m, "wt",          -4.4438234,   0.68829958,   -6.4562344, 0.0)
    _assert_coef(m, "hp",          -0.031460081, 0.0097760391, -3.2180805, 0.0031685207)

    # deviance.lm == Σ wᵢ rᵢ² (weighted), not the raw Σ rᵢ²
    np.testing.assert_allclose(m.rss, 70.031852, rtol=1e-5)

    # confint(m)
    ci = m.ci_bhat
    np.testing.assert_allclose(
        ci[ci.columns[1]].to_numpy(), [35.849673, -5.8515541, -0.051454326], rtol=1e-5
    )
    np.testing.assert_allclose(
        ci[ci.columns[2]].to_numpy(), [42.15496, -3.0360927, -0.011465836], rtol=1e-5
    )


def test_weighted_lm_airquality_na_weight_alignment_matches_R():
    """Weights are carried through na.omit: ``1/Wind`` is length-153, the
    fit drops the 42 rows with NA Ozone/Solar.R, and the surviving 111
    weights must line up with X's rows (R keeps weights in the model
    frame). Before the fix this raised a length-mismatch error."""
    aq = load_dataset("R", "airquality")
    w = 1.0 / aq["Wind"].to_numpy()
    m = lm("Ozone ~ Solar.R + Wind + Temp", aq, weights=w)

    assert m.n == 111
    assert len(m.weights) == 111
    _assert_summary(
        m, n=111, p=4, df_residuals=107, sigma=8.271088,
        r2=0.63202682, r2adj=0.62170982,
        fstats=61.260689, f_pvalue=3.921851128e-23,
        loglike=-513.5424, AIC=1037.0848, BIC=1050.6325,
    )
    _assert_coef(m, "(Intercept)", -41.6861,     28.199857,   -1.478238,  0.1422818)
    _assert_coef(m, "Solar.R",     0.088902323,  0.028574423,  3.1112553, 0.0023887572)
    _assert_coef(m, "Wind",        -4.9880352,   0.80229441,  -6.217213,  0.0)
    _assert_coef(m, "Temp",        1.5031619,    0.30449779,   4.9365281, 0.0)


def test_weighted_lm_predict_intervals_match_R():
    mt = load_dataset("R", "mtcars")
    w = 1.0 / mt["wt"].to_numpy()
    m = lm("mpg ~ wt + hp", mt, weights=w)
    nd = pl.DataFrame({"wt": [3.0], "hp": [150.0]})

    ci = m.predict(newdata=nd, interval="confidence")
    np.testing.assert_allclose(ci["fit"][0], 20.951834, rtol=1e-5)
    np.testing.assert_allclose(ci["lwr"][0], 19.949818, rtol=1e-5)
    np.testing.assert_allclose(ci["upr"][0], 21.95385,  rtol=1e-5)

    # R's predict.lm uses a constant prediction variance (res.var) for
    # weighted fits by default (its "Assuming constant prediction
    # variance" path); hea matches that.
    pi = m.predict(newdata=nd, interval="prediction")
    np.testing.assert_allclose(pi["lwr"][0], 17.619351, rtol=1e-5)
    np.testing.assert_allclose(pi["upr"][0], 24.284317, rtol=1e-5)


def test_weighted_lm_anova_sequential_matches_R(capsys):
    """anova(m) sequential SS must use the *weighted* RSS chain."""
    from hea.R import anova
    mt = load_dataset("R", "mtcars")
    w = 1.0 / mt["wt"].to_numpy()
    m = lm("mpg ~ wt + hp", mt, weights=w)
    anova(m)
    out = capsys.readouterr().out
    assert "339.6128" in out   # wt Sum Sq (weighted)
    assert "140.6327" in out   # wt F value
    assert "25.0087" in out    # hp Sum Sq
    assert "10.356" in out     # hp F value
    assert "70.0319" in out    # Residuals SS == deviance(m)


def test_weighted_lm_zero_weights_match_R():
    """R drops zero-weight rows from the fit: df.residual and the
    log-likelihood sample size count only the nonzero-weight rows, while
    the residual vector keeps every row."""
    mt = load_dataset("R", "mtcars")
    w = np.ones(mt.height)
    w[:2] = 0.0
    m = lm("mpg ~ wt + hp", mt, weights=w)

    assert m.df_residuals == 27       # 30 nonzero rows − 3 params
    assert m._n_eff == 30
    assert len(m._residuals_arr) == 32  # rows retained, as in R
    _assert_coef(m, "(Intercept)", 37.612212,  1.6458459)
    _assert_coef(m, "wt",          -3.9191052, 0.64013433)
    np.testing.assert_allclose(m.sigma,     2.6185529,   rtol=1e-5)
    np.testing.assert_allclose(m.r_squared, 0.83533139,  rtol=1e-5)
    np.testing.assert_allclose(m.loglike,   -69.866403,  rtol=1e-5)
    np.testing.assert_allclose(m.AIC,       147.73281,   rtol=1e-5)
    np.testing.assert_allclose(m.BIC,       153.3376,    rtol=1e-5)


def test_weighted_lm_input_validation():
    gala = load_dataset("faraway", "gala")
    n = gala.height

    # negative weights → hard error (R: lm.wfit "missing or negative ...")
    with pytest.raises(ValueError, match="negative weights"):
        lm("Species ~ Area", gala, weights=-np.ones(n))

    # NaN weights → hard error
    bad = np.ones(n)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="negative weights"):
        lm("Species ~ Area", gala, weights=bad)

    # length mismatch against the input frame → error
    with pytest.raises(ValueError, match="Length of weights"):
        lm("Species ~ Area", gala, weights=np.ones(n + 1))

    # weights + subset: filtered in lockstep, no crash
    w = np.linspace(0.5, 2.0, n)
    m = lm("Species ~ Area", gala, weights=w, subset=np.arange(10))
    assert m.n == 10
    assert len(m.weights) == 10
    np.testing.assert_array_equal(m.weights, w[:10])


# ---------------------------------------------------------------------------
# predict.lm surface — se.fit, type="terms", level= (R 4.6.0 oracles)
#
#   m  <- lm(mpg ~ wt + hp, data = mtcars, weights = 1/wt)
#   nd <- data.frame(wt = c(2.5, 3.5), hp = c(100, 200))
# ---------------------------------------------------------------------------


def test_lm_predict_se_fit_matches_R():
    mt = load_dataset("R", "mtcars")
    w = 1.0 / mt["wt"].to_numpy()
    m = lm("mpg ~ wt + hp", mt, weights=w)
    nd = pl.DataFrame({"wt": [2.5, 3.5], "hp": [100.0, 200.0]})

    p = m.predict(newdata=nd, se_fit=True)
    assert "se.fit" in p.columns
    np.testing.assert_allclose(p["fit"].to_numpy(),    [24.74675, 17.156918],     rtol=1e-5)
    np.testing.assert_allclose(p["se.fit"].to_numpy(), [0.52776628, 0.67550803], rtol=1e-5)

    # se.fit travels with intervals too — it's the mean SE, not the PI width
    pi = m.predict(newdata=nd, interval="prediction", se_fit=True)
    assert {"fit", "se.fit", "lwr", "upr"}.issubset(pi.columns)
    np.testing.assert_allclose(pi["se.fit"].to_numpy(), p["se.fit"].to_numpy())


def test_lm_predict_level_is_alpha_alias():
    mt = load_dataset("R", "mtcars")
    m = lm("mpg ~ wt + hp", mt)
    nd = pl.DataFrame({"wt": [3.0], "hp": [150.0]})
    a = m.predict(newdata=nd, interval="confidence", level=0.90)
    b = m.predict(newdata=nd, interval="confidence", alpha=0.10)
    np.testing.assert_allclose(a["lwr"].to_numpy(), b["lwr"].to_numpy())
    np.testing.assert_allclose(a["upr"].to_numpy(), b["upr"].to_numpy())


def test_lm_predict_terms_numeric_matches_R():
    mt = load_dataset("R", "mtcars")
    w = 1.0 / mt["wt"].to_numpy()
    m = lm("mpg ~ wt + hp", mt, weights=w)
    nd = pl.DataFrame({"wt": [2.5, 3.5], "hp": [100.0, 200.0]})

    pt = m.predict(newdata=nd, type="terms", se_fit=True)
    assert pt.columns == ["wt", "hp", "se.wt", "se.hp"]
    np.testing.assert_allclose(pt.constant, 20.090625, rtol=1e-6)
    np.testing.assert_allclose(pt["wt"].to_numpy(),    [3.1873323, -1.2564911],  rtol=1e-5)
    np.testing.assert_allclose(pt["hp"].to_numpy(),    [1.4687925, -1.6772156],  rtol=1e-5)
    np.testing.assert_allclose(pt["se.wt"].to_numpy(), [0.49368287, 0.19461671], rtol=1e-5)
    np.testing.assert_allclose(pt["se.hp"].to_numpy(), [0.45641882, 0.52118508], rtol=1e-5)

    # R identity: fit == constant + rowSums(terms)
    fit = m.predict(newdata=nd)["fit"].to_numpy()
    np.testing.assert_allclose(
        pt.constant + pt["wt"].to_numpy() + pt["hp"].to_numpy(), fit, rtol=1e-6
    )

    # terms= selects a subset of term labels
    only_wt = m.predict(newdata=nd, type="terms", terms="wt")
    assert only_wt.columns == ["wt"]
    np.testing.assert_allclose(only_wt["wt"].to_numpy(), [3.1873323, -1.2564911], rtol=1e-5)


def test_lm_predict_terms_factor_matches_R():
    """A factor term's dummy columns collapse into one term column."""
    pg = load_dataset("R", "PlantGrowth")
    m = lm("weight ~ group", pg)
    nd = pl.DataFrame({"group": ["ctrl", "trt1", "trt2"]})
    pt = m.predict(newdata=nd, type="terms", se_fit=True)
    assert pt.columns == ["group", "se.group"]
    np.testing.assert_allclose(pt.constant, 5.073, rtol=1e-6)
    np.testing.assert_allclose(pt["group"].to_numpy(), [-0.041, -0.412, 0.453], rtol=1e-5)
    np.testing.assert_allclose(
        pt["se.group"].to_numpy(), [0.16095464, 0.16095464, 0.16095464], rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Rank-deficient parity — aliased coefficients kept as NA rows (R 4.6.0)
#
#   m <- lm(area ~ CO2 + tree, data = stomata)   # tree nested in CO2
#   coef(m) keeps all 7 columns with tree6 = NA; summary prints the NA row.
# ---------------------------------------------------------------------------


def test_rank_deficient_keeps_aliased_as_NA():
    st = load_dataset("gamair", "stomata")
    m = lm("area ~ CO2 + tree", st)

    # coef(m) carries every original column, NA for the aliased one (R parity)
    full = ["(Intercept)", "CO22", "tree2", "tree3", "tree4", "tree5", "tree6"]
    assert list(m.bhat.columns) == full
    assert m._aliased_cols == ["tree6"]
    # the aliased coefficient resolves to NaN (not a KeyError) — the foot-gun fix
    from hea.R import coef
    assert np.isnan(coef(m)["tree6"])
    assert np.isnan(m.se_bhat["tree6"][0])
    assert np.isnan(m.p_values["tree6"][0])

    # estimable coefficients + df unchanged (fit runs on the kept columns)
    assert m.df_residuals == 18
    np.testing.assert_allclose(
        [m.bhat.row(0)[i] for i in range(6)],
        [1.6233739, 0.7063876, -0.02473095, -0.46041298, 0.45947681, 0.5737821],
        rtol=1e-6,
    )
    np.testing.assert_allclose(m.sigma, 0.21863203, rtol=1e-5)

    # summary prints the singularities header + the aliased "NA" row
    s = repr(m.summary())
    assert "(1 not defined because of singularities)" in s
    assert "tree6" in s
    tree6_line = next(ln for ln in s.splitlines() if ln.startswith("tree6"))
    assert tree6_line.split() == ["tree6", "NA", "NA", "NA", "NA", "NA", "NA"]

    # summary.lm$aliased is the full-width logical; $coefficients stays estimable
    su = m.summary()
    assert su.aliased.tolist() == [False, False, False, False, False, False, True]
    assert su.coefficients.shape == (6, 4)


# ---------------------------------------------------------------------------
# Track D — contrasts= / na.action= / subset-as-expression (R 4.6.0 oracles)
# ---------------------------------------------------------------------------


def test_lm_contrasts_argument_matches_R():
    # lm(weight ~ group, contrasts=list(group="contr.sum"))
    pg = load_dataset("R", "PlantGrowth")
    m = lm("weight ~ group", pg, contrasts={"group": "contr.sum"})
    assert list(m.bhat.columns) == ["(Intercept)", "group1", "group2"]
    np.testing.assert_allclose(m._bhat_arr, [5.073, -0.041, -0.412], rtol=1e-5)
    np.testing.assert_allclose(m.sigma, 0.62337463, rtol=1e-6)


def test_lm_na_action_exclude_pads_accessors():
    """na.action='exclude' pads resid/fitted/predict back to the model-frame
    length with NA (R's naresid/napredict); the raw m.residuals stays
    complete-case, like R's m$residuals."""
    from hea.R import fitted, predict, resid
    aq = load_dataset("R", "airquality")  # 153 rows, 42 with NA Ozone/Solar.R

    me = lm("Ozone ~ Solar.R + Wind + Temp", aq, na_action="exclude")
    r = resid(me)
    assert len(r) == 153
    assert int(np.isnan(r).sum()) == 42
    np.testing.assert_allclose(
        r[:6], [7.95452, 1.00129, -12.8228, -0.475226, np.nan, np.nan],
        rtol=1e-4, equal_nan=True,
    )
    assert list(np.where(np.isnan(r))[0][:6]) == [4, 5, 9, 10, 24, 25]
    assert len(fitted(me)) == 153
    assert predict(me).height == 153
    # raw component stays complete-case (R's m$residuals)
    assert me.residuals.height == 111

    # default na.action='omit' is unchanged (drops to complete cases)
    mo = lm("Ozone ~ Solar.R + Wind + Temp", aq)
    assert len(resid(mo)) == 111
    # coefficients identical between omit and exclude (same fit)
    np.testing.assert_allclose(me._bhat_arr, mo._bhat_arr, rtol=1e-10)


def test_lm_na_action_fail_raises():
    aq = load_dataset("R", "airquality")
    with pytest.raises(ValueError, match="missing values"):
        lm("Ozone ~ Solar.R + Wind + Temp", aq, na_action="fail")
    # a complete-case frame fits fine under na.fail
    pg = load_dataset("R", "PlantGrowth")
    m = lm("weight ~ group", pg, na_action="fail")
    assert m.n == 30


def test_lm_subset_as_expression():
    g = load_dataset("faraway", "gala")

    # polars expression (native Python form)
    m_e = lm("Species ~ Area", g, subset=pl.col("Area") > 10)
    assert m_e.n == 13
    np.testing.assert_allclose(m_e._bhat_arr, [134.33592, 0.058669392], rtol=1e-5)

    # R-style string, single comparison
    m_s = lm("Species ~ Area", g, subset="Area > 10")
    assert m_s.n == 13
    np.testing.assert_allclose(m_s._bhat_arr, m_e._bhat_arr, rtol=1e-12)

    # R-style compound: R's `&` binds looser than `<`/`>` (opposite of
    # Python) — routed through hea's translator so precedence matches R.
    m_c = lm("Species ~ 1", g, subset="Area > 10 & Elevation < 200")
    assert m_c.n == 2  # R: subset = Area > 10 & Elevation < 200  -> 2 rows
    m_c2 = lm("Species ~ 1", g,
              subset=(pl.col("Area") > 10) & (pl.col("Elevation") < 200))
    assert m_c2.n == 2


# ---------------------------------------------------------------------------
# Track E — saturated guard / singular.ok / residual types / weights()
# ---------------------------------------------------------------------------


def test_lm_saturated_fit_returns_nan_not_crash():
    """df_residual == 0 (n == rank): R returns NaN sigma/SE/adj-R² and no
    F-statistic rather than erroring (was a ZeroDivisionError in hea)."""
    g = load_dataset("faraway", "gala")
    m = lm("Species ~ Area", g, subset="Area > 10 & Elevation < 200")  # 2 rows
    assert m.df_residuals == 0
    np.testing.assert_allclose(m._bhat_arr, [15.8321, 1.39296], rtol=1e-4)
    assert np.isnan(m.sigma)
    assert np.all(np.isnan(m._se_bhat_arr))
    assert np.isclose(m.r_squared, 1.0)          # perfect fit
    assert np.isnan(m.r_squared_adjusted)
    assert m.fstats is None
    assert isinstance(repr(m.summary()), str)    # summary must not crash


def test_lm_singular_ok_false_raises():
    st = load_dataset("gamair", "stomata")
    with pytest.raises(ValueError, match="singular fit"):
        lm("area ~ CO2 + tree", st, singular_ok=False)
    # default singular_ok=True still fits (drops the aliased dummy)
    assert lm("area ~ CO2 + tree", st, singular_ok=True).df_residuals == 18


def test_lm_residuals_types_match_R():
    from hea.R import resid
    mt = load_dataset("R", "mtcars")
    w = 1.0 / mt["wt"].to_numpy()
    m = lm("mpg ~ wt + hp", mt, weights=w)
    # residuals.lm: response/working = raw r; pearson/deviance = √w·r
    np.testing.assert_allclose(
        resid(m, "response")[:3], [-2.8988903, -1.7657153, -2.9668587], rtol=1e-5
    )
    np.testing.assert_allclose(resid(m, "working"), resid(m, "response"))
    np.testing.assert_allclose(
        resid(m, "pearson")[:3], [-1.7909404, -1.0413621, -1.9478381], rtol=1e-5
    )
    np.testing.assert_allclose(resid(m, "deviance"), resid(m, "pearson"))
    # unweighted: weighted residuals equal the raw residuals
    mu = lm("mpg ~ wt", mt)
    np.testing.assert_allclose(resid(mu, "pearson"), resid(mu, "response"))


def test_lm_weights_generic():
    from hea.R import weights
    mt = load_dataset("R", "mtcars")
    w = 1.0 / mt["wt"].to_numpy()
    np.testing.assert_allclose(
        weights(lm("mpg ~ wt + hp", mt, weights=w))[:3],
        [0.38167939, 0.34782609, 0.43103448], rtol=1e-5,
    )
    assert weights(lm("mpg ~ wt", mt)) is None   # R: NULL when unweighted
