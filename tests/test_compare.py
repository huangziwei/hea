"""Tests for ``hea.compare`` — model-comparison helpers.

Covers everything exposed by ``hea/compare.py``:

* ``anova(...)`` — multi-model F / Chisq / LRT tables across lm, glm,
  gam, and gmm dispatch branches; plus the single-model forms
  (``anova(lm)`` Type-I, ``anova(gam)`` parametric + smooth tables).
* ``AIC()`` / ``BIC()`` — printed comparison tables.
* ``drop1(...)`` — single-term-deletion tables for lm and glm.

Numerical pins go against R / mgcv oracles. The comments next to each
test flag the R quirk being locked in (denominator choice, label flip,
recalibrated AIC, marginality scope, …) so future refactors don't
quietly drift toward the natural-looking-but-wrong alternative.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import polars as pl
import pytest
from conftest import load_dataset, load_glm_oracle

from hea.family import Binomial, Gamma, Gaussian, Poisson
from hea.models import gam, glm, lm
from hea.R import (
    AIC,
    BIC,
    _anova_gam_rdf,
    _anova_gam_table,
    _anova_glm_table,
    _extract_aic_lm,
    add1,
    anova,
    drop1,
    step,
)


def _capture(fn, *args, **kwargs) -> str:
    """Run ``fn(*args, **kwargs)`` and return what it printed to stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


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
    "anova_poisson_quine": _fits_anova_poisson_quine,
    "anova_gamma_trees": _fits_anova_gamma_trees,
    "anova_gaussian_iris": _fits_anova_gaussian_iris,
    "anova_binomial_menarche": _fits_anova_binomial_menarche,
}


@pytest.mark.parametrize("oid", list(ANOVA_CASES.keys()))
def test_anova_glm(oid: str):
    o = load_glm_oracle(oid)
    fits = ANOVA_CASES[oid]()

    labels = [f"m{i}" for i in range(len(fits))]
    df, _ = _anova_glm_table(*fits, labels=labels, test=None)
    assert isinstance(df, pl.DataFrame)

    np.testing.assert_array_equal(
        np.asarray(df["Resid. Df"].to_numpy()),
        np.asarray(o["resid_df"], dtype=int),
    )
    np.testing.assert_allclose(
        df["Resid. Dev"].to_numpy(),
        np.asarray(o["resid_dev"]),
        atol=5e-3,
    )

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
            assert df_hea[i] is None and dev_hea[i] is None, (
                f"row {i}: expected None for first row"
            )
            assert stat_hea[i] is None and p_hea[i] is None
            continue
        assert df_hea[i] == df_R, f"row {i}: Df hea={df_hea[i]} R={df_R}"
        np.testing.assert_allclose(
            dev_hea[i],
            dev_R,
            atol=5e-3,
            err_msg=f"row {i}: Deviance hea={dev_hea[i]} R={dev_R}",
        )
        np.testing.assert_allclose(
            stat_hea[i],
            stat_R,
            atol=5e-3,
            err_msg=f"row {i}: {stat_col} hea={stat_hea[i]} R={stat_R}",
        )
        if p_R > 1e-10:
            np.testing.assert_allclose(
                p_hea[i],
                p_R,
                atol=5e-3,
                err_msg=f"row {i}: {p_col} hea={p_hea[i]} R={p_R}",
            )
        else:
            assert p_hea[i] < 1e-6, (
                f"row {i}: {p_col} hea={p_hea[i]} should be tiny like R={p_R}"
            )


def test_anova_rejects_mixed_types():
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
    heart = pl.DataFrame(
        {
            "ck": [20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460],
            "ha": [2, 13, 30, 30, 21, 19, 18, 13, 19, 15, 7, 8],
            "ok": [88, 26, 8, 5, 0, 1, 1, 1, 1, 0, 0, 0],
        }
    ).with_columns(
        n=pl.col("ha") + pl.col("ok"),
        p=pl.col("ha") / (pl.col("ha") + pl.col("ok")),
    )
    n = heart["n"].to_numpy()
    m0 = glm("p ~ 1", data=heart, family=Binomial(link="logit"), weights=n)
    m1 = glm("p ~ ck", data=heart, family=Binomial(link="logit"), weights=n)

    for kw in [{}, {"test": "Chisq"}, {"test": "LRT"}, {"test": "lrt"}]:
        out = _capture(anova, m0, m1, **kw)
        assert "Pr(>Chi)" in out
        assert "234.78" in out  # R's deviance, matches at 2dp

    out = _capture(anova, m0, m1, test="F")
    assert "Pr(>F)" in out
    assert "234.78" in out

    with pytest.raises(NotImplementedError, match="Rao"):
        anova(m0, m1, test="Rao")

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
    assert "F" in df.columns and "Pr(>F)" in df.columns
    assert "Pr(>Chi)" not in df.columns
    assert df["Df"][1] == 2
    np.testing.assert_allclose(df["Deviance"][1], 211.5687, atol=5e-3)
    np.testing.assert_allclose(df["F"][1], 105.78436, atol=5e-3)
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
    assert df["Resid. Df"].to_list() == [30, 29, 28]
    np.testing.assert_allclose(
        df["Resid. Dev"].to_numpy(),
        [8.3172, 0.8592, 0.8002],
        atol=5e-4,
    )
    assert df["Df"][1] == 1
    np.testing.assert_allclose(df["Deviance"][1], 7.4580, atol=5e-4)
    np.testing.assert_allclose(df["F"][1], 280.35781, atol=5e-3)
    np.testing.assert_allclose(df["Pr(>F)"][1], 4.0473e-16, rtol=5e-2)
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
    df_F, _ = _anova_glm_table(m0, m1, labels=["m0", "m1"], test="F")
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
    out = _capture(anova, m0, m1, test="F")
    assert "F" in out and "Pr(>F)" in out
    assert "Pr(>Chi)" not in out
    assert "2.2188" in out


def test_anova_lm_rejects_test_argument():
    """`test=` is glm-only; lm/gmm always use F/Chisq respectively."""
    d = load_dataset("R", "iris")
    m1 = lm("Sepal.Length ~ Petal.Length", d)
    m2 = lm("Sepal.Length ~ Petal.Length + Petal.Width", d)
    with pytest.raises(TypeError, match="test="):
        anova(m1, m2, test="Chisq")


def test_anova_gam_single_pins_to_mgcv_on_trees():
    """``anova(gam_single)`` should produce mgcv's anova.gam single-model
    output: parametric Terms F-table + smooth significance table."""
    trees = load_dataset("mgcv", "trees").with_columns(
        Hclass=((pl.col("Height") / 10).floor() - 5)
        .cast(pl.Int64)
        .replace_strict(
            [1, 2, 3],
            ["small", "medium", "large"],
            return_dtype=pl.Enum(["small", "medium", "large"]),
        ),
    )
    ct7 = gam("Volume ~ Hclass + s(Girth)", family=Gamma(link="log"), data=trees)
    out = _capture(anova, ct7)
    assert "Family: Gamma" in out
    assert "Volume ~ Hclass + s(Girth)" in out
    assert "Parametric Terms:" in out
    assert "Hclass" in out
    assert "6.802" in out  # F-stat
    assert "0.00428" in out  # p-value
    assert "Approximate significance of smooth terms:" in out
    assert "s(Girth)" in out
    assert "2.444" in out
    assert "3.076" in out
    assert "152.7" in out


def test_anova_gam_two_model_F_pins_to_mgcv_on_trees():
    """`anova(gam1, gam2)` on Gamma family auto-picks F (unknown-scale)
    and matches mgcv's ``anova.gam`` numerics. Locks the residual-df
    convention (``n - sum(edf1)``, not ``n - sum(edf)``), the F denom
    (the *full* model's ``scale``), and the F-test against
    ``F(Δdf, df_residual_full)``.
    """
    trees = load_dataset("R", "trees")
    fam = Gamma(link="log")
    g1 = gam("Volume ~ s(Girth)", data=trees, family=fam)
    g2 = gam("Volume ~ s(Girth) + s(Height)", data=trees, family=fam)
    df, _ = _anova_gam_table(g1, g2, labels=["g1", "g2"], test=None)
    assert "F" in df.columns and "Pr(>F)" in df.columns
    np.testing.assert_allclose(
        df["Resid. Df"].to_numpy(), [26.6424, 25.9560], atol=5e-4
    )
    np.testing.assert_allclose(df["Resid. Dev"].to_numpy(), [0.3787, 0.1842], atol=5e-4)
    np.testing.assert_allclose(df["Df"][1], 0.6864, atol=5e-4)
    np.testing.assert_allclose(df["Deviance"][1], 0.1945, atol=5e-4)
    np.testing.assert_allclose(df["F"][1], 41.0742, atol=5e-3)
    np.testing.assert_allclose(df["Pr(>F)"][1], 7.487e-06, rtol=1e-3)


def test_anova_gam_three_model_F_walk_pins_to_mgcv_on_trees():
    """3-model F walk: each row's F denom is the *full* model's scale,
    not the immediately-preceding row's. Pinned to mgcv.
    """
    trees = load_dataset("R", "trees")
    fam = Gamma(link="log")
    g0 = gam("Volume ~ 1", data=trees, family=fam)
    g1 = gam("Volume ~ s(Girth)", data=trees, family=fam)
    g2 = gam("Volume ~ s(Girth) + s(Height)", data=trees, family=fam)
    df, _ = _anova_gam_table(g0, g1, g2, labels=["g0", "g1", "g2"], test="F")
    np.testing.assert_allclose(
        df["Resid. Df"].to_numpy(), [30.0, 26.6424, 25.9560], atol=5e-4
    )
    np.testing.assert_allclose(df["Df"][1], 3.3576, atol=5e-4)
    np.testing.assert_allclose(df["Deviance"][1], 7.9385, atol=5e-4)
    np.testing.assert_allclose(df["F"][1], 342.7106, atol=5e-2)
    assert df["Pr(>F)"][1] < 1e-15
    np.testing.assert_allclose(df["F"][2], 41.0742, atol=5e-3)
    np.testing.assert_allclose(df["Pr(>F)"][2], 7.487e-06, rtol=1e-3)


def test_anova_gam_explicit_chisq_on_unknown_scale():
    """`test='Chisq'` overrides the default F for unknown-scale families.
    Stat = Δdev / dispersion_full, p from chi-square. Pinned to mgcv.
    """
    trees = load_dataset("R", "trees")
    fam = Gamma(link="log")
    g1 = gam("Volume ~ s(Girth)", data=trees, family=fam)
    g2 = gam("Volume ~ s(Girth) + s(Height)", data=trees, family=fam)
    df, _ = _anova_gam_table(g1, g2, labels=["g1", "g2"], test="Chisq")
    assert "F" not in df.columns
    assert "Pr(>Chi)" in df.columns
    np.testing.assert_allclose(df["Deviance"][1], 0.1945, atol=5e-4)
    np.testing.assert_allclose(df["Pr(>Chi)"][1], 4.8926e-08, rtol=1e-3)


def test_anova_gam_residual_df_uses_edf1_not_edf():
    """Locks the mgcv-specific residual-df convention. ``n - sum(edf1)``
    differs from ``n - sum(edf)`` (which is ``g.df_residuals``) because
    edf1 is the 1-step effective df designed for hypothesis testing.
    """
    trees = load_dataset("R", "trees")
    g = gam("Volume ~ s(Girth)", data=trees, family=Gamma(link="log"))
    rdf_anova = _anova_gam_rdf(g)
    rdf_naive = g.n - float(np.sum(g.edf))
    rdf_edf1 = g.n - float(np.sum(g.edf1))
    np.testing.assert_allclose(rdf_anova, rdf_edf1, rtol=1e-12)
    assert abs(rdf_anova - rdf_naive) > 0.5  # they really differ
    np.testing.assert_allclose(rdf_anova, 26.6424, atol=5e-4)


def test_anova_gam_rejects_mixed_with_glm_or_lm():
    """`anova(gam, glm)` and `anova(gam, lm)` should fail the same-type
    guard, not silently fall into the glm or lm branches.
    """
    trees = load_dataset("R", "trees")
    g = gam("Volume ~ s(Girth)", data=trees, family=Gamma(link="log"))
    g_glm = glm("Volume ~ log(Girth)", trees, family=Gamma(link="log"))
    g_lm = lm("Volume ~ Girth", trees)
    with pytest.raises(TypeError, match="same type"):
        anova(g, g_glm)
    with pytest.raises(TypeError, match="same type"):
        anova(g, g_lm)


def test_anova_gam_rejects_mixed_families():
    """All gam fits in a multi-model anova must share family/link."""
    trees = load_dataset("R", "trees")
    g_log = gam("Volume ~ s(Girth)", data=trees, family=Gamma(link="log"))
    g_inv = gam("Volume ~ s(Girth)", data=trees, family=Gamma(link="inverse"))
    with pytest.raises(ValueError, match="family and link"):
        anova(g_log, g_inv)


def test_anova_gam_test_argument_validation():
    """LRT alias, Rao not-implemented, bogus value — same surface as
    `anova(glm,...)` so users get a consistent error contract."""
    trees = load_dataset("R", "trees")
    fam = Gamma(link="log")
    g1 = gam("Volume ~ s(Girth)", data=trees, family=fam)
    g2 = gam("Volume ~ s(Girth) + s(Height)", data=trees, family=fam)

    df_chi, _ = _anova_gam_table(g1, g2, labels=["g1", "g2"], test="Chisq")
    df_lrt, _ = _anova_gam_table(g1, g2, labels=["g1", "g2"], test="LRT")
    assert df_chi.columns == df_lrt.columns
    np.testing.assert_allclose(
        df_chi["Pr(>Chi)"].to_list()[1:],
        df_lrt["Pr(>Chi)"].to_list()[1:],
        rtol=1e-12,
    )

    with pytest.raises(NotImplementedError, match="Rao"):
        anova(g1, g2, test="Rao")
    with pytest.raises(ValueError, match="must be"):
        anova(g1, g2, test="bogus")


def test_aic_bic_dispatch_on_glm():
    fits = _fits_anova_gaussian_iris()
    aic_table = AIC(*fits)
    bic_table = BIC(*fits)
    assert aic_table["AIC"].to_list() == pytest.approx([m.AIC for m in fits])
    assert bic_table["BIC"].to_list() == pytest.approx([m.BIC for m in fits])


def test_drop1_lm_F_test_pins_to_R_on_iris():
    """`drop1(lm, test="F")` against R's drop1.lm on iris.
    Pins the Mallows-style AIC column (extractAIC) and the F + p values.
    """
    from hea.R import _drop1_lm

    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length + Species", iris)
    out = _capture(_drop1_lm, m, test="F", k=2.0)
    assert "Single term deletions" in out
    assert "Sepal.Length ~ Petal.Length + Species" in out
    assert "16.6817" in out
    assert "-321.4488" in out
    assert "22.2745" in out
    assert "38.9562" in out
    assert "-196.2296" in out
    assert "194.9496" in out
    assert "7.8434" in out
    assert "24.525" in out
    assert "-267.6411" in out
    assert "34.3231" in out


def test_drop1_lm_no_test_omits_F_columns():
    """`drop1(lm)` (no test) shows just Df/Sum of Sq/RSS/AIC."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length + Species", iris)
    out = _capture(drop1, m)
    assert "F value" not in out
    assert "Pr(>F)" not in out
    for col in ("Df", "Sum of Sq", "RSS", "AIC"):
        assert col in out


def test_drop1_lm_AIC_uses_extractAIC_not_AIC_lm():
    """The AIC column must be ``n*log(RSS/n) + 2p`` (R's extractAIC),
    not the standard ``AIC.lm`` formula. They differ by a constant —
    pinning to R's printed value catches the wrong choice."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length + Species", iris)
    aic_extract = _extract_aic_lm(m.rss, m.df_residuals, m.n, k=2.0)
    np.testing.assert_allclose(aic_extract, -321.4488, atol=5e-4)
    assert abs(m.AIC - aic_extract) > 100  # they really are different


def test_drop1_glm_F_pins_to_R_on_iris_gaussian():
    """`drop1(glm, test="F")` on Gaussian iris matches R. Same F values
    as the lm path (Gaussian Pearson dispersion = dev/df), but the
    Deviance/AIC columns use the glm conventions."""
    iris = load_dataset("R", "iris")
    g = glm("Sepal.Length ~ Petal.Length + Species", iris, family=Gaussian())
    out = _capture(drop1, g, test="F")
    assert "106.2327" in out
    assert "38.9562" in out
    assert "231.452" in out
    assert "194.9496" in out
    assert "160.0404" in out
    assert "34.3231" in out


def test_drop1_glm_Chisq_gaussian_uses_n_log_dev_ratio():
    """Gaussian Chisq uses the proper σ-unknown LRT
    ``n*log(dev_drop/dev_full)``, not ``Δdev/dispersion``. The two
    diverge enough that they're easy to distinguish numerically.
    """
    iris = load_dataset("R", "iris")
    g = glm("Sepal.Length ~ Petal.Length + Species", iris, family=Gaussian())
    out = _capture(drop1, g, test="Chisq")
    assert "scaled dev." in out
    assert "Pr(>Chi)" in out
    assert "127.2192" in out
    assert "57.8077" in out


def test_drop1_glm_Chisq_poisson_uses_LRT_label():
    """Scale-known families (Poisson, Binomial) get column header
    ``"LRT"`` (= raw Δdev) — matches R."""
    quine = load_dataset("MASS", "quine")
    g = glm("Days ~ Sex + Age + Eth + Lrn", quine, family=Poisson(link="log"))
    out = _capture(drop1, g, test="Chisq")
    assert "LRT" in out
    assert "scaled dev." not in out
    assert "14.4041" in out
    assert "0.000148" in out
    assert "168.3239" in out
    assert "166.8448" in out
    assert "45.798" in out


def test_drop1_glm_Chisq_gamma_pins_to_R_on_trees():
    """Gamma (unknown-scale) Chisq: stat = Δdev/dispersion_full,
    column labeled "scaled dev." Pinned to R."""
    trees = load_dataset("R", "trees")
    g = glm("Volume ~ log(Girth) + log(Height)", trees, family=Gamma(link="inverse"))
    out = _capture(drop1, g, test="Chisq")
    assert "scaled dev." in out
    assert "152.6788" in out
    assert "2.2188" in out
    assert "336.331" in out


def test_drop1_glm_F_gamma_uses_residual_mean_deviance_denom():
    """Gamma F-test denom is dev_full/df_full (residual mean deviance),
    not Pearson dispersion. The two values would give F=152.68 vs the
    correct R-matching F=142.12 — easy to distinguish."""
    trees = load_dataset("R", "trees")
    g = glm("Volume ~ log(Girth) + log(Height)", trees, family=Gamma(link="inverse"))
    out = _capture(drop1, g, test="F")
    assert "142.1225" in out
    assert "2.0654" in out


def test_drop1_glm_AIC_for_Gamma_holds_dispersion_fixed():
    """For Gamma, drop1.glm AIC ≠ standard glm.AIC of the dropped fit.
    R freezes dispersion at the full model's value across all rows so
    AICs are comparable across drops. Pin: drop log(Girth) AIC = 336.33
    (R's drop1) vs 240.26 (standalone refit AIC).
    """
    trees = load_dataset("R", "trees")
    g = glm("Volume ~ log(Girth) + log(Height)", trees, family=Gamma(link="inverse"))
    g_dropped = glm("Volume ~ log(Height)", trees, family=Gamma(link="inverse"))
    np.testing.assert_allclose(g_dropped.AIC, 240.2625, atol=5e-3)
    out = _capture(drop1, g, test="Chisq")
    assert "336.331" in out
    assert "240.26" not in out


def test_drop1_none_row_AIC_matches_full_model():
    """The ``<none>`` row's AIC must equal the full model's AIC.
    For lm this is extractAIC; for glm this is m.AIC + (k-2)*edf
    (= m.AIC for default k=2)."""
    iris = load_dataset("R", "iris")
    g = glm("Sepal.Length ~ Petal.Length + Species", iris, family=Gaussian())
    out = _capture(drop1, g, test="F")
    np.testing.assert_allclose(g.AIC, 106.2327, atol=5e-4)
    assert "106.2327" in out


def test_drop1_lm_k_log_n_gives_BIC_style_AIC():
    """`k=log(n)` swaps AIC for BIC in the Mallows-style formula
    (matches R's ``drop1.lm(..., k=log(nobs(m)))``)."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length + Species", iris)
    bic_full = _extract_aic_lm(m.rss, m.df_residuals, m.n, k=float(np.log(m.n)))
    out = _capture(drop1, m, test="F", k=float(np.log(m.n)))
    bic_str = f"{round(bic_full, 4)}"
    assert bic_str in out


def test_drop1_glm_LRT_alias_matches_Chisq():
    """test='LRT' must give the same numerics as test='Chisq'."""
    iris = load_dataset("R", "iris")
    g = glm("Sepal.Length ~ Petal.Length + Species", iris, family=Gaussian())
    out_chi = _capture(drop1, g, test="Chisq")
    out_lrt = _capture(drop1, g, test="LRT")
    assert out_chi == out_lrt


def test_drop1_glm_rejects_Rao_and_invalid_test():
    """test='Rao' is not implemented; bogus values raise ValueError."""
    iris = load_dataset("R", "iris")
    g = glm("Sepal.Length ~ Petal.Length + Species", iris, family=Gaussian())
    with pytest.raises(NotImplementedError, match="Rao"):
        drop1(g, test="Rao")
    with pytest.raises(ValueError, match="must be"):
        drop1(g, test="bogus")


def test_drop1_lm_rejects_Chisq_test():
    """drop1(lm, test='Chisq') is not supported (same surface as anova(lm))."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length + Species", iris)
    with pytest.raises(ValueError, match="must be 'F' or None"):
        drop1(m, test="Chisq")


def test_drop1_intercept_only_model_raises():
    """drop1 needs at least one term to drop; intercept-only fits fail."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ 1", iris)
    with pytest.raises(TypeError, match="at least one RHS term"):
        drop1(m)


def test_drop1_respects_marginality_on_gavote():
    """When an interaction is in the model, R's ``drop1`` (via
    ``drop.scope``) refuses to drop a main effect that participates in
    that interaction. This is the Faraway (2016) Ch.1 ``gavote`` model:
    ``cperAA + cpergore*usage + equip`` should drop only ``cperAA``,
    ``equip``, and ``cpergore:usage`` — never ``cpergore`` or ``usage``
    alone, because they're "marginal to" the interaction.
    """
    from hea import data

    g = data("gavote", package="faraway")
    g = g.mutate(usage=pl.col("rural")).select(pl.exclude("rural"))
    g = g.mutate(
        undercount=(pl.col("ballots") - pl.col("votes")) / pl.col("ballots"),
        pergore=pl.col("gore") / pl.col("votes"),
    )
    g = g.mutate(
        cpergore=pl.col("pergore") - pl.col("pergore").mean(),
        cperAA=pl.col("perAA") - pl.col("perAA").mean(),
    )
    m = lm("undercount ~ cperAA + cpergore*usage + equip", data=g)
    out = _capture(drop1, m, test="F")
    assert "cperAA" in out
    assert "equip" in out
    assert "cpergore:usage" in out
    lines = out.splitlines()
    drop_rows = [
        ln for ln in lines if ln.startswith(("cperAA", "equip", "cpergore", "usage"))
    ]
    starts = {ln.split()[0] for ln in drop_rows}
    assert starts == {"cperAA", "equip", "cpergore:usage"}
    assert "0.8264" in out
    assert "2.4964" in out
    assert "0.0517" in out


def test_drop1_gam_not_implemented():
    """drop1(gam) raises NotImplementedError: mgcv's drop1.gam has
    smoothing-parameter semantics that are not ported."""
    trees = load_dataset("R", "trees")
    g = gam("Volume ~ s(Girth)", data=trees, family=Gamma(link="log"))
    with pytest.raises(NotImplementedError, match="gam"):
        drop1(g)


def test_add1_lm_F_pins_to_R_on_iris():
    """`add1(lm, test='F')` on iris matches R numerics exactly. Pins
    the augmented-MSE F-denominator and the extractAIC formula.
    """
    iris = load_dataset("R", "iris")
    m0 = lm("Sepal.Length ~ Petal.Length", iris)
    out = _capture(add1, m0, "Petal.Length + Petal.Width + Species", test="F")
    assert "-267.6411" in out
    assert "0.6443" in out
    assert "23.8807" in out
    assert "-269.6347" in out
    assert "3.9663" in out
    assert "7.8434" in out
    assert "16.6817" in out
    assert "-321.4488" in out
    assert "34.3231" in out


def test_add1_glm_Chisq_gaussian_uses_n_log_dev_ratio():
    """Add1's Gaussian Chisq mirrors drop1's: ``n*log(dev_cur/dev_aug)``.
    For Petal.Width on iris: 150*log(24.525/23.881) ≈ 3.994. The
    naive Δdev/dispersion alternative would give a different number.
    """
    iris = load_dataset("R", "iris")
    g0 = glm("Sepal.Length ~ Petal.Length", iris, family=Gaussian())
    out = _capture(add1, g0, "Petal.Length + Petal.Width + Species", test="Chisq")
    assert "scaled dev." in out
    assert "3.9936" in out
    assert "57.8077" in out


def test_add1_glm_Chisq_poisson_uses_LRT_label():
    """Scale-known glm add1: column header is ``LRT`` (= raw Δdev) since
    dispersion=1. Pinned to R's ``add1.glm(..., test='Chisq')`` on quine.
    """
    quine = load_dataset("MASS", "quine")
    g0 = glm("Days ~ Sex", quine, family=Poisson(link="log"))
    out = _capture(add1, g0, "Sex + Age + Eth + Lrn", test="Chisq")
    assert "LRT" in out
    assert "scaled dev." not in out
    assert "148.9548" in out
    assert "182.139" in out
    assert "8.0321" in out


def test_add1_glm_Chisq_gamma_uses_current_dispersion():
    """For non-Gaussian unknown-scale, R uses the *current* model's
    Pearson dispersion as the Chisq normalizer (different from drop1
    which also uses the current/full model's dispersion — both are
    consistent: 'the smaller-residual side's dispersion'). Pin to R.
    """
    trees = load_dataset("R", "trees")
    g0 = glm("Volume ~ log(Girth)", trees, family=Gamma(link="inverse"))
    out = _capture(add1, g0, "log(Girth) + log(Height)", test="Chisq")
    assert "2.0921" in out
    assert "2.2188" not in out


def test_add1_glm_F_gamma_uses_aug_residual_mean_deviance():
    """add1 F-denom is the *augmented* model's dev/df_resid (mirror of
    drop1's "current model's dev/df_resid"). On Gamma trees adding
    log(Height): F = 2.0654, not 2.2188 (Pearson dispersion alt).
    """
    trees = load_dataset("R", "trees")
    g0 = glm("Volume ~ log(Girth)", trees, family=Gamma(link="inverse"))
    out = _capture(add1, g0, "log(Girth) + log(Height)", test="F")
    assert "2.0654" in out


def test_add1_respects_marginality():
    """add.scope blocks adding ``a:b`` unless both ``a`` and ``b`` are
    already present. With current=``y~a`` and scope=``y~a*b``, only
    ``b`` is addable; ``a:b`` is held until ``b`` is in.
    """
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length", iris)
    out = _capture(add1, m, "Petal.Length * Petal.Width")
    assert "Petal.Width" in out
    rows = [ln for ln in out.splitlines() if ":" in ln and ln.startswith("Petal")]
    assert not rows, f"unexpected interaction rows: {rows}"


def test_add1_raises_on_empty_scope():
    """If scope adds nothing (already in current), R errors —
    we match with a ValueError."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length", iris)
    with pytest.raises(ValueError, match="no terms in scope"):
        add1(m, "Petal.Length")


def test_add1_gam_not_implemented():
    """add1(gam) raises — same boundary as drop1(gam)."""
    trees = load_dataset("R", "trees")
    g = gam("Volume ~ s(Girth)", data=trees, family=Gamma(link="log"))
    with pytest.raises(NotImplementedError, match="gam"):
        add1(g, "s(Girth) + s(Height)")


def test_step_backward_on_gavote_matches_R():
    """Faraway (2016) Ch.1: starting from the saturated two-way
    interaction model on gavote, ``step(biglm, trace=FALSE)`` reduces
    to a smaller model. Pinned against R's actual output (the book
    text has a typo describing the chosen interactions; R's real
    output has ``equip:perAA`` not ``econ:perAA``).
    """
    from hea import data

    g = data("gavote", package="faraway")
    g = g.mutate(usage=pl.col("rural")).select(pl.exclude("rural"))
    g = g.mutate(
        undercount=(pl.col("ballots") - pl.col("votes")) / pl.col("ballots"),
        pergore=pl.col("gore") / pl.col("votes"),
    )
    biglm = lm(
        "undercount ~ (equip+econ+usage+atlanta)^2 + "
        "(equip+econ+usage+atlanta)*(perAA+pergore)",
        g,
    )
    smallm = step(biglm, trace=False)
    final_terms = {t.label for t in smallm._expanded.terms}
    assert final_terms == {
        "equip",
        "econ",
        "usage",
        "perAA",
        "equip:econ",
        "equip:perAA",
        "usage:perAA",
    }
    assert "atlanta" not in final_terms
    assert "pergore" not in final_terms


def test_step_forward_from_intercept_on_iris():
    """Forward selection from intercept-only adds Petal.Length and
    Species (the two strong predictors); Petal.Width's marginal benefit
    over Petal.Length isn't enough to enter."""
    iris = load_dataset("R", "iris")
    m0 = lm("Sepal.Length ~ 1", iris)
    m_final = step(
        m0,
        scope="Petal.Length + Petal.Width + Species",
        direction="forward",
        trace=False,
    )
    final_terms = {t.label for t in m_final._expanded.terms}
    assert final_terms == {"Petal.Length", "Species"}


def test_step_glm_poisson_keeps_full_when_no_drop_helps():
    """On quine + Poisson, every term is significant; backward step
    should keep the full model. Returned model should be ``isinstance``
    of glm and have the same formula (after step's intercept-explicit
    rewrite)."""
    quine = load_dataset("MASS", "quine")
    g_full = glm("Days ~ Sex + Age + Eth + Lrn", quine, family=Poisson(link="log"))
    g_step = step(g_full, trace=False)
    final_terms = {t.label for t in g_step._expanded.terms}
    assert final_terms == {"Sex", "Age", "Eth", "Lrn"}
    assert isinstance(g_step, glm)


def test_step_trace_prints_expected_output():
    """trace=True prints a header (Start: AIC=...), the formula, the
    candidate table, and "Step: AIC=..." for each move. Pinned to R's
    backward step on iris."""
    iris = load_dataset("R", "iris")
    m_full = lm("Sepal.Length ~ Petal.Length + Petal.Width + Species", iris)
    out = _capture(step, m_full, direction="backward", trace=True)
    assert "Start:  AIC=" in out
    assert "-319.45" in out
    assert "- Petal.Width" in out
    assert "Step:  AIC=-321.45" in out
    assert "Petal.Length + Species" in out


def test_step_trace_false_is_silent():
    """trace=False prints nothing; only the final model is returned."""
    iris = load_dataset("R", "iris")
    m_full = lm("Sepal.Length ~ Petal.Length + Petal.Width + Species", iris)
    out = _capture(step, m_full, trace=False)
    assert out == ""


def test_step_with_BIC_penalty():
    """Custom k swaps AIC for BIC (k=log(n)). BIC penalizes models
    more, so step should retain at least as few terms as default AIC.
    """
    iris = load_dataset("R", "iris")
    m_full = lm("Sepal.Length ~ Petal.Length + Petal.Width + Species", iris)
    n_log = float(np.log(m_full.n))
    m_aic = step(m_full, trace=False)
    m_bic = step(m_full, k=n_log, trace=False)
    aic_terms = {t.label for t in m_aic._expanded.terms}
    bic_terms = {t.label for t in m_bic._expanded.terms}
    assert "Petal.Width" not in aic_terms
    assert "Petal.Width" not in bic_terms
    assert len(bic_terms) <= len(aic_terms)


def test_step_dict_scope_with_lower_bound():
    """A dict scope with both ``lower`` and ``upper`` constrains the
    walk on both sides. A term in ``lower`` can never be dropped, even
    if dropping would improve AIC."""
    iris = load_dataset("R", "iris")
    m_full = lm("Sepal.Length ~ Petal.Length + Petal.Width + Species", iris)
    m_kept = step(
        m_full,
        scope={"lower": "Petal.Width", "upper": "Petal.Length + Petal.Width + Species"},
        trace=False,
    )
    final_terms = {t.label for t in m_kept._expanded.terms}
    assert "Petal.Width" in final_terms


def test_step_rejects_invalid_direction():
    """Bogus direction raises ValueError."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length", iris)
    with pytest.raises(ValueError, match="direction must be"):
        step(m, direction="sideways")


def test_step_unsupported_model_type():
    """step(gam) and step(gmm) intentionally raise NotImplementedError."""
    trees = load_dataset("R", "trees")
    g = gam("Volume ~ s(Girth)", data=trees, family=Gamma(link="log"))
    with pytest.raises(NotImplementedError, match="gam"):
        step(g)


def test_step_handles_NAs_in_predictors():
    """When some predictors carry NAs and others don't, the original
    fit drops those rows but a sub-formula that excludes the NA columns
    would re-include them. step() must pin every refit to a common
    row set (R does this via na.action) — without that pinning,
    weights/AIC mismatches crash the refit. Regression test for the
    Faraway WCGS book example.
    """
    np.random.seed(42)
    n = 500
    df = pl.DataFrame(
        {
            "y": np.random.binomial(1, 0.3, n).astype(float),
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
            "d": np.random.randn(n),
            "e": [None if i % 30 == 0 else float(np.random.randn()) for i in range(n)],
        }
    )
    m = glm("y ~ a + b + c + d + e", data=df, family=Binomial())
    assert m.n == 483
    m_step = step(m, trace=False)
    assert m_step.n == 483
    out_drop = _capture(drop1, m, test="Chisq")
    assert "Single term deletions" in out_drop
    m_small = glm("y ~ a + b", data=df, family=Binomial())
    out_add = _capture(add1, m_small, "a + b + c + d + e", test="Chisq")
    assert "Single term additions" in out_add


def test_step_already_minimal_returns_input():
    """If no move improves AIC (the input is already the minimum),
    step returns the input (or an equivalent fit). The returned
    model should at least share the same term set."""
    iris = load_dataset("R", "iris")
    m = lm("Sepal.Length ~ Petal.Length", iris)
    m_out = step(m, trace=False)
    assert {t.label for t in m_out._expanded.terms} == {
        t.label for t in m._expanded.terms
    }
