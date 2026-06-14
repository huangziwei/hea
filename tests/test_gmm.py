"""
Notebook examples → regression tests for hea.gmm.

Pins printed numerical outputs from Bates, "lme4: Mixed-effects Modeling
with R" (lMMwR.pdf, in example/data/) so the formulae→formula.py
migration can be validated against book-standard lme4 results. Models
covered: fm01, fm02 (Ch 1, Dyestuff/Dyestuff2); fm03, fm04, fm04a (Ch 2,
Penicillin/Pastes); fm06, fm07 (Ch 3, sleepstudy); fm10, fm16, fm17
(Ch 4, Machines/ergoStool).

The post-migration gmm is expected to expose, at minimum:
    m.n, m.n_groups       — sample size, dict of group → #levels
    m.sigma                — residual SD
    m.sd_re[group]         — np.ndarray of component SDs (length 1 for
                             scalar bars; length 2+ for vector bars)
    m.corr_re[group]       — np.ndarray correlation matrix (only present
                             for vector bars; missing/None for scalar)
    m.bhat / m.se_bhat / m.t_values   — DataFrames keyed by fixed-effect
                                        column name (R-canonical, e.g.
                                        '(Intercept)', 'MachineB')
    m.REML_criterion       — only set when REML=True
    m.deviance, m.loglike, m.df_resid   — only set when REML=False (ML)
    m.AIC, m.BIC           — set for both; REML uses the REML criterion
                             as ``-2 log L`` (matches lme4's AIC()/BIC())
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from conftest import assert_fp_equiv, load_dataset
from hea.family import Gaussian, Poisson
from hea.models.gmm import gmm


# ---------------------------------------------------------------------------
# Shared fits / profiles. Each model and (where applicable) its profile is
# computed once per module — both are immutable post-construction in lmpy.gmm,
# and `profile()` here always uses the default n_grid=41, so the results are
# bit-identical across the tests that consume them. Cuts ~14s off the file.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fm01ML():
    data = load_dataset("lme4", "Dyestuff")
    return gmm("Yield ~ 1 + (1|Batch)", data, REML=False)


@pytest.fixture(scope="module")
def fm01ML_profile(fm01ML):
    return fm01ML.profile()


@pytest.fixture(scope="module")
def fm03ML():
    data = load_dataset("lme4", "Penicillin")
    return gmm("diameter ~ 1 + (1|plate) + (1|sample)", data, REML=False)


@pytest.fixture(scope="module")
def fm03ML_profile(fm03ML):
    return fm03ML.profile(n_grid=41)


@pytest.fixture(scope="module")
def fm04ML():
    data = load_dataset("lme4", "Pastes")
    return gmm("strength ~ 1 + (1|sample) + (1|batch)", data, REML=False)


@pytest.fixture(scope="module")
def fm04aML():
    data = load_dataset("lme4", "Pastes")
    return gmm("strength ~ 1 + (1|sample)", data, REML=False)


@pytest.fixture(scope="module")
def fm06ML():
    data = load_dataset("lme4", "sleepstudy")
    return gmm("Reaction ~ 1 + Days + (1+Days|Subject)", data, REML=False)


@pytest.fixture(scope="module")
def fm07ML():
    data = load_dataset("lme4", "sleepstudy")
    return gmm(
        "Reaction ~ 1 + Days + (1|Subject) + (0+Days|Subject)",
        data, REML=False,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_fixed(m, col, est, se=None, tval=None, *, atol=5e-3):
    if col not in m.bhat.columns:
        raise KeyError(f"{col!r} not in {list(m.bhat.columns)!r}")
    np.testing.assert_allclose(m.bhat[col][0], est, atol=atol)
    if se is not None:
        np.testing.assert_allclose(m.se_bhat[col][0], se, atol=atol)
    if tval is not None:
        # rtol covers large |t| where the pinned value is R's print()-rounded
        # display (e.g. 104.2 vs full-precision 104.1623); atol covers small
        # |t| where relative tol would be too lax.
        np.testing.assert_allclose(m.t_values[col][0], tval, atol=5e-2, rtol=1e-3)


def _assert_re_scalar(m, group, sd, *, atol=5e-3):
    sds = np.asarray(m.sd_re[group]).ravel()
    assert sds.shape == (1,), f"expected scalar bar at {group!r}, got shape {sds.shape}"
    np.testing.assert_allclose(sds[0], sd, atol=atol)


def _assert_re_vector(m, group, sds, corr=None, *, atol=5e-3, corr_atol=5e-2):
    got = np.asarray(m.sd_re[group]).ravel()
    np.testing.assert_allclose(got, np.asarray(sds), atol=atol)
    if corr is not None:
        C = np.asarray(m.corr_re[group])
        # Off-diagonal only — diagonal is trivially 1.
        i, j = np.triu_indices(C.shape[0], k=1)
        np.testing.assert_allclose(C[i, j], np.asarray(corr), atol=corr_atol)


def _assert_ml_summary(m, *, AIC, BIC, loglike, deviance, df_resid,
                       atol=5e-2):
    np.testing.assert_allclose(m.AIC, AIC, atol=atol)
    np.testing.assert_allclose(m.BIC, BIC, atol=atol)
    np.testing.assert_allclose(m.loglike, loglike, atol=atol)
    np.testing.assert_allclose(m.deviance, deviance, atol=atol)
    assert m.df_resid == df_resid


def _lrt(m_reduced, m_full):
    chisq = m_reduced.deviance - m_full.deviance
    df = m_full.npar - m_reduced.npar
    p = chi2.sf(chisq, df)
    return chisq, df, p


# ---------------------------------------------------------------------------
# Ch 1: A Simple, Linear, Mixed-effects Model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 1 of lme-family-port.md: the public ``family=`` argument was added.
# Default (``None``) and explicit ``Gaussian()`` must produce the same fit
# (FP-equal — see conftest.assert_fp_equiv); non-Gaussian families must raise
# ``NotImplementedError`` with a message pointing at the port plan until
# Phase 2-5 land the Laplace path.
# ---------------------------------------------------------------------------


def test_family_default_equals_explicit_gaussian():
    """``family=None`` (default) and ``family=Gaussian()`` produce the same fit."""
    data = load_dataset("lme4", "Dyestuff")
    m_default = gmm("Yield ~ 1 + (1|Batch)", data, REML=True)
    m_explicit = gmm("Yield ~ 1 + (1|Batch)", data, family=Gaussian(), REML=True)

    assert_fp_equiv(m_default.theta, m_explicit.theta)
    assert_fp_equiv(m_default._beta, m_explicit._beta)
    assert_fp_equiv(m_default.sigma, m_explicit.sigma)
    assert_fp_equiv(m_default.REML_criterion, m_explicit.REML_criterion)


def test_family_non_gaussian_runs_glmm_path():
    """Poisson family dispatches to the GLMM Laplace path (Phase 5).

    Just smoke-checks that ``hea.models.gmm(..., family=poisson())`` fits without
    raising. Numerical parity with ``lme4::glmer`` is pinned in
    ``test_gmm_glmm.py``'s Phase 5 acceptance tests.
    """
    data = load_dataset("lme4", "Dyestuff")
    # Dyestuff's Yield is continuous, but for a smoke test we can fit a
    # Poisson model — the optimizer should still converge.
    m = gmm("Yield ~ 1 + (1|Batch)", data, family=Poisson())
    assert m.theta.shape == (1,)
    assert m._beta.shape == (1,)
    assert np.isfinite(m.deviance)


def test_bates_1_4_dyestuff_fm01_REML():
    """fm01 <- lmer(Yield ~ 1 + (1|Batch), Dyestuff)  -- REML (default)"""
    data = load_dataset("lme4", "Dyestuff")
    m = gmm("Yield ~ 1 + (1|Batch)", data, REML=True)

    assert m.n == 30
    assert m.n_groups == {"Batch": 6}
    np.testing.assert_allclose(m.REML_criterion, 319.7, atol=0.1)
    np.testing.assert_allclose(m.sigma, 49.5101, atol=5e-3)
    _assert_re_scalar(m, "Batch", 42.0010)
    _assert_fixed(m, "(Intercept)", 1527.5, se=19.38, tval=78.80)


def test_bates_1_4_dyestuff_fm01_ML(fm01ML):
    """fm01ML <- lmer(Yield ~ 1 + (1|Batch), Dyestuff, REML=FALSE)"""
    m = fm01ML

    _assert_ml_summary(
        m, AIC=333.3271, BIC=337.5307, loglike=-163.6635,
        deviance=327.3271, df_resid=27,
    )
    np.testing.assert_allclose(m.sigma, 49.5101, atol=5e-3)
    _assert_re_scalar(m, "Batch", 37.2602, atol=5e-3)
    _assert_fixed(m, "(Intercept)", 1527.5, se=17.6938, tval=86.33)


def test_bates_1_4_dyestuff_fm01_ML_profile_confint(fm01ML_profile):
    """confint(profile(fm01ML), level=...) — pinned to lme4 4.5/R 4.5.

    The 99% lower bound for .sig01 is the regression: lme4 reports 0
    (the natural σ ≥ 0 boundary) when the profile flattens to an
    asymptote above the −2.576 threshold. Lmpy used to return NaN.
    """
    pr = fm01ML_profile

    # Tolerance reflects the spline-inversion residual (R uses
    # interpSpline+backSpline, hea uses CubicSpline+brentq); the
    # profile grid itself now matches R to ~1e-9.
    ci99 = pr.confint(level=0.99).to_dict(as_series=False)
    assert ci99["parameter"] == [".sig01", ".sigma", "(Intercept)"]
    np.testing.assert_allclose(ci99["0.5%"], [0.0, 35.5632, 1465.874], atol=2e-3)
    np.testing.assert_allclose(ci99["99.5%"], [113.6877, 75.6680, 1589.126], atol=2e-3)

    ci95 = pr.confint(level=0.95).to_dict(as_series=False)
    np.testing.assert_allclose(ci95["2.5%"], [12.1985, 38.2300, 1486.452], atol=2e-3)
    np.testing.assert_allclose(ci95["97.5%"], [84.0631, 67.6577, 1568.548], atol=2e-3)


def test_bates_1_4_dyestuff_fm01_ML_plot_fig17(fm01ML_profile):
    """plot(which=, transform=, ax=) building blocks for Bates Fig. 1.7."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr = fm01ML_profile

    fig, axes = plt.subplots(1, 3, sharey=True)
    pr.plot(which=".sigma", transform="log",    ax=axes[0])
    pr.plot(which=".sigma",                     ax=axes[1])
    pr.plot(which=".sigma", transform="square", ax=axes[2])

    x_log = axes[0].get_lines()[0].get_xdata()
    x_id  = axes[1].get_lines()[0].get_xdata()
    x_sq  = axes[2].get_lines()[0].get_xdata()
    np.testing.assert_allclose(x_log, np.log(x_id))
    np.testing.assert_allclose(x_sq, x_id ** 2)
    assert axes[0].get_title() == "log(.sigma)"
    assert axes[1].get_title() == ".sigma"
    assert axes[2].get_title() == ".sigma²"
    assert all(ax.get_xlabel() == ".sigma" for ax in axes)

    # Single-parameter via which= without ax= still builds its own figure.
    fig2 = pr.plot(which=".sigma")
    assert [a.get_title() for a in fig2.axes] == [".sigma"]

    # ax= with multiple parameters is rejected.
    fig3, ax3 = plt.subplots()
    try:
        pr.plot(ax=ax3)
    except ValueError:
        pass
    else:
        raise AssertionError("ax= with all-params should raise")


def test_bates_1_4_dyestuff_fm01_ML_plot_ranef_qqranef(fm01ML):
    """Caterpillar (Fig 1.11) and qqmath (Fig 1.12) of ranef(., condVar=TRUE).

    BLUPs and condSDs pinned to R lme4 4.5; bars use level=0.95 default
    (±qnorm(0.975)·SE ≈ ±1.96·SE).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    m = fm01ML

    # Numerical ranef + condSD pinned to R.
    [(_, _, _, b_mat, se_mat)] = m._ranef()
    b_ref  = [-16.628222, 0.369516, 26.974671, -21.801446, 53.579825, -42.494344]
    sd_ref = [19.03445] * 6
    np.testing.assert_allclose(b_mat.ravel(),  b_ref,  atol=5e-3)
    np.testing.assert_allclose(se_mat.ravel(), sd_ref, atol=5e-3)

    # Caterpillar (Fig 1.11): BLUP on x, sorted by BLUP, level index on y.
    m.plot_ranef(strip=False)
    ax = plt.gcf().axes[0]
    assert ax.get_title() == ""
    ec = ax.containers[0]  # ErrorbarContainer
    x_dots = ec[0].get_xdata()
    np.testing.assert_allclose(np.sort(x_dots), np.sort(b_ref), atol=5e-3)
    plt.close("all")

    # qqmath (Fig 1.12): BLUP on x, normal quantiles (Hazen) on y.
    m.plot_qq_ranef(strip=False)
    ax = plt.gcf().axes[0]
    assert ax.get_title() == ""
    assert ax.get_ylabel() == "Standard normal quantiles"
    ec = ax.containers[0]
    x_dots = ec[0].get_xdata()
    y_dots = ec[0].get_ydata()
    n = 6
    q_expect = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    np.testing.assert_allclose(x_dots, np.sort(b_ref), atol=5e-3)
    np.testing.assert_allclose(y_dots, q_expect, atol=1e-10)
    plt.close("all")


def test_plot_ranef_layout_vertical_stacks_panels():
    """``plot_ranef(layout="vertical")`` flips a 1×n panel row into n×1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.models import gmm
    from hea import data
    pen = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", pen)

    fig_h = fm.plot_ranef()
    fig_v = fm.plot_ranef(layout="vertical")
    try:
        # Horizontal: two panels in one row.
        rows_h = {ax.get_subplotspec().rowspan.start for ax in fig_h.axes}
        cols_h = {ax.get_subplotspec().colspan.start for ax in fig_h.axes}
        assert rows_h == {0} and cols_h == {0, 1}

        # Vertical: two panels in one column.
        rows_v = {ax.get_subplotspec().rowspan.start for ax in fig_v.axes}
        cols_v = {ax.get_subplotspec().colspan.start for ax in fig_v.axes}
        assert rows_v == {0, 1} and cols_v == {0}

        # And the vertical figure is taller than wide vs. the horizontal
        # one (rough shape check; exact sizes depend on level counts).
        w_h, h_h = fig_h.get_size_inches()
        w_v, h_v = fig_v.get_size_inches()
        assert h_v > h_h and w_v < w_h
    finally:
        plt.close(fig_h)
        plt.close(fig_v)


def test_plot_ranef_aspect_controls_subplot_width():
    """``aspect=`` sets each subplot's width:height ratio in inches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.models import gmm
    from hea import data
    pen = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", pen)

    fig_wide = fm.plot_ranef(layout="vertical", aspect=2.0)
    fig_narrow = fm.plot_ranef(layout="vertical", aspect=0.5)
    try:
        w_wide, _ = fig_wide.get_size_inches()
        w_narrow, _ = fig_narrow.get_size_inches()
        # aspect=2.0 → 4× wider subplots than aspect=0.5
        assert w_wide == pytest.approx(4.0 * w_narrow, rel=1e-9)
    finally:
        plt.close(fig_wide)
        plt.close(fig_narrow)


def test_plot_ranef_explicit_layout_tuple():
    """An explicit ``(nrow, ncol)`` tuple is respected; over-allocating
    leaves the trailing cells hidden."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.models import gmm
    from hea import data
    pen = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", pen)

    fig = fm.plot_ranef(layout=(2, 2))
    try:
        # Four cells allocated; two visible (one per ranef panel).
        visibles = [ax for ax in fig.axes if ax.get_visible()]
        hidden = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(visibles) == 2
        assert len(hidden) == 2
    finally:
        plt.close(fig)


def test_plot_ranef_layout_rejects_too_few_cells():
    """A (nrow, ncol) tuple with fewer cells than panels raises."""
    from hea.models import gmm
    from hea import data
    pen = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", pen)

    with pytest.raises(ValueError, match="holds 1 cells"):
        fm.plot_ranef(layout=(1, 1))


def test_plot_ranef_layout_rejects_bad_value():
    from hea.models import gmm
    from hea import data
    pen = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", pen)

    with pytest.raises(ValueError, match="layout:"):
        fm.plot_ranef(layout="diagonal")


def test_plot_ranef_which_filters_to_one_term():
    """``which="<term>"`` picks every panel whose grouping factor matches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.models import gmm
    from hea import data
    pen = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", pen)
    fig = fm.plot_ranef(which="plate")
    try:
        titles = [a.get_title() for a in fig.axes if a.get_visible()]
        assert titles == ["plate: (Intercept)"]
    finally:
        plt.close(fig)


def test_plot_ranef_which_filters_to_one_panel_title():
    """``which="<term>: <col>"`` picks a single column of a vector bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.models import gmm
    from hea import data
    sleep = data("sleepstudy", "lme4")
    fm = gmm("Reaction ~ 1 + Days + (1 + Days | Subject)", sleep)

    # Term key pulls both columns of the vector bar.
    fig_all = fm.plot_ranef(which="Subject")
    # Full title pulls just one.
    fig_one = fm.plot_ranef(which="Subject: Days")
    try:
        assert sorted(a.get_title() for a in fig_all.axes) == [
            "Subject: (Intercept)", "Subject: Days",
        ]
        assert [a.get_title() for a in fig_one.axes] == ["Subject: Days"]
    finally:
        plt.close(fig_all)
        plt.close(fig_one)


def test_plot_ranef_which_accepts_list():
    """A list of keys / titles works (mix-and-match allowed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.models import gmm
    from hea import data
    sleep = data("sleepstudy", "lme4")
    fm = gmm("Reaction ~ 1 + Days + (1 + Days | Subject)", sleep)
    fig = fm.plot_ranef(which=["Subject: (Intercept)"])
    try:
        titles = [a.get_title() for a in fig.axes]
        assert titles == ["Subject: (Intercept)"]
    finally:
        plt.close(fig)


def test_plot_ranef_which_unknown_raises():
    from hea.models import gmm
    from hea import data
    pen = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", pen)
    with pytest.raises(KeyError, match="no matching panel"):
        fm.plot_ranef(which="nonexistent")


def test_bates_2_plot_design_layout_matches_fig_2_3_2_4():
    """plot_design() — 4-panel mosaic A=Z' / B=Λ / C=Z'Z / D=L.

    Layout: AAA over BCD; top panel is the wide Z transpose, bottom row
    is three q×q sparsity panels. Matches Bates lme4-book Figs 2.3+2.4
    (the Penicillin crossed-RE example).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.models import gmm
    from hea import data
    penicillin = data("Penicillin", "lme4")
    fm = gmm("diameter ~ 1 + (1 | plate) + (1 | sample)", penicillin)
    fig = fm.plot_design()
    try:
        # Four labelled axes from subplot_mosaic: A, B, C, D.
        axd = {ax.get_label(): ax for ax in fig.axes if ax.get_label() in "ABCD"}
        assert set(axd) == {"A", "B", "C", "D"}
        # Bottom-row labels match Bates' panel captions.
        assert axd["B"].get_xlabel() == "Λ"
        assert axd["C"].get_xlabel() == "Z'Z"
        assert axd["D"].get_xlabel() == "L"
        # Top panel renders the q × n shape; bottom panels render q × q.
        assert axd["A"].images[0].get_array().shape == (fm.q, fm.n)
        assert axd["B"].images[0].get_array().shape == (fm.q, fm.q)
        assert axd["C"].images[0].get_array().shape == (fm.q, fm.q)
        assert axd["D"].images[0].get_array().shape == (fm.q, fm.q)
    finally:
        plt.close(fig)


def test_bates_1_4_dyestuff_fm01_ML_plot_density(fm01ML_profile):
    """plot_density() — profile-implied density peaks pinned to lme4:::dens."""
    import matplotlib
    matplotlib.use("Agg")
    pr = fm01ML_profile
    fig = pr.plot_density()
    peaks = {}
    for ax in fig.axes:
        x, y = ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata()
        peaks[ax.get_title()] = (float(y.max()), float(x[np.argmax(y)]))
    # lme4:::dens reference, npts=201, upper=0.999. Peak heights agree to
    # ~1e-3; peak x can differ a bit (lme4 uses cubic spline, hea uses
    # monotone PCHIP) so widen the location tolerance.
    for name, (h_ref, x_ref), x_atol in [
        (".sig01",      (0.0287, 33.095),  1.0),
        (".sigma",      (0.0574, 47.817),  1.0),
        ("(Intercept)", (0.0225, 1527.5),  2.0),
    ]:
        h, x = peaks[name]
        np.testing.assert_allclose(h, h_ref, atol=2e-3)
        np.testing.assert_allclose(x, x_ref, atol=x_atol)


def test_bates_1_4_dyestuff2_fm02_REML():
    """fm02 <- lmer(Yield ~ 1 + (1|Batch), Dyestuff2)  -- singular fit, σ₁=0"""
    data = load_dataset("lme4", "Dyestuff2")
    m = gmm("Yield ~ 1 + (1|Batch)", data, REML=True)

    np.testing.assert_allclose(m.REML_criterion, 161.8, atol=0.1)
    np.testing.assert_allclose(m.sigma, 3.7165, atol=5e-3)
    _assert_re_scalar(m, "Batch", 0.0, atol=1e-4)
    _assert_fixed(m, "(Intercept)", 5.6656, se=0.6784, tval=8.352)


# ---------------------------------------------------------------------------
# Ch 2: Models With Multiple Random-effects Terms
# ---------------------------------------------------------------------------


def test_bates_2_1_penicillin_fm03_REML():
    """fm03 <- lmer(diameter ~ 1 + (1|plate) + (1|sample), Penicillin)"""
    data = load_dataset("lme4", "Penicillin")
    m = gmm("diameter ~ 1 + (1|plate) + (1|sample)", data, REML=True)

    assert m.n == 144
    assert m.n_groups == {"plate": 24, "sample": 6}
    np.testing.assert_allclose(m.REML_criterion, 330.9, atol=0.1)
    np.testing.assert_allclose(m.sigma, 0.5499, atol=5e-3)
    _assert_re_scalar(m, "plate", 0.8467)
    _assert_re_scalar(m, "sample", 1.9316)
    _assert_fixed(m, "(Intercept)", 22.9722, se=0.8086, tval=28.41)


def test_bates_2_6_penicillin_fm03_ML_profile_pairs(fm03ML_profile):
    """plot_pairs (Bates Fig 2.6): each profile row carries the full
    optimum, traces are pinned to lme4 ``profile(fm03ML)`` output.

    Profile of σ₁ (.sig01): as σ₁ varies, the optimal (σ₂, σ, β₀) at
    each grid point should match what lme4 records. Same for profile of
    σ₂. The intercept stays orthogonal to the variance components in
    this model, so its row is essentially constant.
    """
    import matplotlib
    matplotlib.use("Agg")
    from scipy.interpolate import PchipInterpolator

    pr = fm03ML_profile

    # Per-row schema: every parameter has a column, plus zeta.
    assert list(pr.data[".sig01"].columns) == [
        ".sig01", ".sig02", ".sigma", "(Intercept)", "zeta",
    ]

    def _interps(name):
        df = pr.data[name]
        v = df[name].to_numpy()
        o = np.argsort(v)
        return {
            col: PchipInterpolator(v[o], df[col].to_numpy()[o])
            for col in df.columns if col not in (name, "zeta")
        }

    # Profile of .sig01 — pinned to lme4 rows ζ ≈ -3.0 / 0 / +2.5.
    sp = _interps(".sig01")
    for sig01, refs in [
        (0.5501273, {".sig02": 1.766020, ".sigma": 0.5595737, "(Intercept)": 22.97222}),
        (1.3197227, {".sig02": 1.780696, ".sigma": 0.5490436, "(Intercept)": 22.97222}),
    ]:
        for col, ref in refs.items():
            np.testing.assert_allclose(float(sp[col](sig01)), ref, atol=1e-2)

    # Profile of .sig02 — pinned to lme4 rows ζ ≈ -2.6 / +2.5.
    sp = _interps(".sig02")
    for sig02, refs in [
        (0.9584949, {".sig01": 0.8435989, ".sigma": 0.5503961, "(Intercept)": 22.97222}),
        (4.6831540, {".sig01": 0.8463784, ".sigma": 0.5499141, "(Intercept)": 22.97222}),
    ]:
        for col, ref in refs.items():
            np.testing.assert_allclose(float(sp[col](sig02)), ref, atol=1e-2)

    # Render and check the splom layout (Bates Fig 2.6 / lme4 splom.thpr):
    # origin at lower-left so the diagonal runs from the bottom-left cell
    # (.sig01) to the top-right cell ((Intercept)). Cells *above* the
    # display diagonal (display_row + display_col < n-1) are v-space; cells
    # *below* (display_row + display_col > n-1) are ζ-space, axis-clamped
    # to ±1.05·√χ²₂(0.99).
    from scipy.stats import chi2 as _chi2
    fig = pr.plot_pairs()
    assert len(fig.axes) == 16
    # Diagonal cells (r + c == n-1) carry parameter labels.
    diag_axes_in_order = [fig.axes[r * 4 + (3 - r)] for r in range(4)]
    diag_labels = [ax.texts[0].get_text() for ax in diag_axes_in_order]
    assert diag_labels == ["(Intercept)", ".sigma", ".sig02", ".sig01"]
    # ζ-space cell (e.g., bottom-row .sig02 column at r=3, c=1, both
    # vid_row=0 and vid_col=1, vid_row<vid_col so ζ-space).
    mlev = float(np.sqrt(_chi2.ppf(0.99, 2)))
    ax_zeta = fig.axes[3 * 4 + 1]
    np.testing.assert_allclose(ax_zeta.get_xlim(), (-1.05 * mlev, 1.05 * mlev))
    np.testing.assert_allclose(ax_zeta.get_ylim(), (-1.05 * mlev, 1.05 * mlev))


def test_bates_2_7_penicillin_fm03_ML_profile_pairs_log(fm03ML_profile):
    """plot_pairs(transform="log") (Bates Fig 2.7): the log-scale variant
    of the splom, R's ``splom(log(profile(fm03)))``.

    ζ is invariant under monotone v-reparameterization, so the
    zeta-space lower triangle is bit-identical to Fig 2.6; only the
    upper-triangle v-space axis limits change (log applied to .sig*,
    .sigma) and diagonal labels become ``log(.sigXX)``. Reference
    bwd-spline values come from R's
    ``predict(attr(log(profile(fm03)),"backward")[[nm]], ±mlev)$y``.
    """
    import matplotlib
    matplotlib.use("Agg")
    from scipy.stats import chi2 as _chi2

    pr = fm03ML_profile

    fig = pr.plot_pairs(transform="log")
    n = 4
    assert len(fig.axes) == n * n

    # Diagonal labels: log() wraps variance components only; (Intercept)
    # stays on natural scale (matches R's logProf with signames=FALSE).
    diag_axes = [fig.axes[r * n + (n - 1 - r)] for r in range(n)]
    diag_labels = [ax.texts[0].get_text() for ax in diag_axes]
    assert diag_labels == ["(Intercept)", "log(.sigma)", "log(.sig02)", "log(.sig01)"]

    # Zeta-space lower triangle: still ±1.05·mlev — log on v doesn't move ζ.
    mlev = float(np.sqrt(_chi2.ppf(0.99, 2)))
    ax_zeta = fig.axes[3 * n + 1]
    np.testing.assert_allclose(ax_zeta.get_xlim(), (-1.05 * mlev, 1.05 * mlev))
    np.testing.assert_allclose(ax_zeta.get_ylim(), (-1.05 * mlev, 1.05 * mlev))

    # v-space upper triangle: each parameter's axis runs from
    # bwd[name](-mlev) to bwd[name](+mlev), in log space for .sig*.
    # Top row of the splom (r=0) is the (Intercept) row across all cols.
    # axis layout: at r=0,c=k the cell is (vid_row=3=(Intercept), vid_col=k).
    # x-axis = column parameter, y-axis = (Intercept).
    # R reference (predict(bwd[[name]], ±mlev)$y from log(profile)):
    r_ref = {
        ".sig01":     (-0.601424,  0.377905),
        ".sig02":     (-0.114746,  1.797648),
        ".sigma":     (-0.785579, -0.383550),
        "(Intercept)": (19.565139, 26.379308),
    }
    col_names = [".sig01", ".sig02", ".sigma"]  # cols 0..2 in display
    for c, name in enumerate(col_names):
        ax = fig.axes[0 * n + c]
        np.testing.assert_allclose(ax.get_xlim(), r_ref[name], atol=1e-3)
        np.testing.assert_allclose(ax.get_ylim(), r_ref["(Intercept)"], atol=1e-3)


def test_bates_2_2_pastes_fm04_ML(fm04ML):
    """fm04 <- lmer(strength ~ 1 + (1|sample) + (1|batch), Pastes, REML=FALSE)"""
    m = fm04ML

    assert m.n == 60
    assert m.n_groups == {"sample": 30, "batch": 10}
    _assert_ml_summary(
        m, AIC=255.9945, BIC=264.3724, loglike=-123.9972,
        deviance=247.9945, df_resid=56,
    )
    np.testing.assert_allclose(m.sigma, 0.8234, atol=5e-3)
    _assert_re_scalar(m, "sample", 2.9041)
    _assert_re_scalar(m, "batch", 1.0951)
    _assert_fixed(m, "(Intercept)", 60.0533, se=0.6421, tval=93.52)


def test_bates_2_2_pastes_fm04a_ML_and_LRT(fm04ML, fm04aML):
    """fm04a <- lmer(strength ~ 1 + (1|sample), Pastes, REML=FALSE);
       anova(fm04a, fm04) — LRT for σ_batch = 0."""
    full = fm04ML
    red = fm04aML

    _assert_ml_summary(
        red, AIC=254.4020, BIC=260.6855, loglike=-124.2010,
        deviance=248.4020, df_resid=57,
    )
    np.testing.assert_allclose(red.sigma, 0.8234, atol=5e-3)
    _assert_re_scalar(red, "sample", 3.1037)
    _assert_fixed(red, "(Intercept)", 60.0533, se=0.5765, tval=104.2)

    chisq, df, p = _lrt(red, full)
    np.testing.assert_allclose(chisq, 0.4072, atol=5e-3)
    assert df == 1
    np.testing.assert_allclose(p, 0.5234, atol=5e-3)


# ---------------------------------------------------------------------------
# Ch 3: Models for Longitudinal Data (sleepstudy)
# ---------------------------------------------------------------------------


def test_bates_3_2_sleepstudy_fm07_uncorrelated_ML(fm07ML):
    """fm07 <- lmer(Reaction ~ 1+Days + (1|Subject) + (0+Days|Subject),
                    sleepstudy, REML=FALSE)"""
    m = fm07ML

    assert m.n == 180
    assert m.n_groups == {"Subject": 18}
    _assert_ml_summary(
        m, AIC=1762.0, BIC=1778.0, loglike=-876.00,
        deviance=1752.0, df_resid=175, atol=0.1,
    )
    np.testing.assert_allclose(m.sigma, 25.5556, atol=5e-3)
    # lme4 lists the two scalar bars on Subject as two rows; we expose the
    # second one under the disambiguated key "Subject.1".
    _assert_re_scalar(m, "Subject", 24.1717)        # (Intercept)
    _assert_re_scalar(m, "Subject.1", 5.7986)        # Days
    _assert_fixed(m, "(Intercept)", 251.405, se=6.708, tval=37.48)
    _assert_fixed(m, "Days",         10.467, se=1.519, tval=6.89)


def test_bates_3_2_sleepstudy_fm06_correlated_ML(fm06ML):
    """fm06 <- lmer(Reaction ~ 1+Days + (1+Days|Subject),
                    sleepstudy, REML=FALSE)  -- correlated REs"""
    m = fm06ML

    _assert_ml_summary(
        m, AIC=1763.9393, BIC=1783.0971, loglike=-875.9697,
        deviance=1751.9393, df_resid=174, atol=0.1,
    )
    np.testing.assert_allclose(m.sigma, 25.5918, atol=5e-3)
    _assert_re_vector(m, "Subject", sds=[23.7803, 5.7168], corr=[0.0813])
    _assert_fixed(m, "(Intercept)", 251.405, se=6.632, tval=37.907)
    _assert_fixed(m, "Days",         10.467, se=1.502, tval=6.968)


def test_sleepstudy_reml_theta_matches_lme4_nloptwrap():
    """hea's lmer uses NLopt ``LN_BOBYQA`` — lme4's DEFAULT ``nloptwrap`` —
    so θ̂/σ̂ land on lme4's fit to the CHOLMOD floor (~1e-9), not the ~1e-5
    scatter the old scipy L-BFGS-B left.

    Reference: ``lmer(Reaction ~ Days + (Days|Subject), sleepstudy)`` (REML,
    default control) → ``getME(m,"theta")`` / ``sigma(m)`` at 16 digits.
    """
    data = load_dataset("lme4", "sleepstudy")
    m = gmm("Reaction ~ Days + (Days|Subject)", data, REML=True)
    # lme4 2.0-2 nloptwrap reference (16 sig figs).
    np.testing.assert_allclose(
        m.theta,
        [0.9667417739793641, 0.01516905889466504, 0.2309099532076919],
        rtol=0, atol=1e-7,
    )
    np.testing.assert_allclose(m.sigma, 25.591795721655899, rtol=0, atol=1e-6)


def test_bates_3_2_sleepstudy_LRT_fm07_vs_fm06(fm06ML, fm07ML):
    """anova(fm07, fm06): test whether the (Intercept,Days) correlation
       is non-zero. Book: χ²=0.0639 on 1 df, p=0.8004."""
    chisq, df, p = _lrt(fm07ML, fm06ML)
    np.testing.assert_allclose(chisq, 0.0639, atol=5e-3)
    assert df == 1
    np.testing.assert_allclose(p, 0.8004, atol=5e-3)


# ---------------------------------------------------------------------------
# lmer control parity (gmm-lmer-parity #1-#3, #5): optimizer / optCtrl / start
# are honored on the LMM path. Each was previously read only by the GLMM
# branch and silently dropped for a Gaussian-identity (lmer) fit.
# ---------------------------------------------------------------------------

_SLEEP_F = "Reaction ~ Days + (Days|Subject)"
# lme4 2.0-2 nloptwrap REML reference (16 sig figs; same pin as
# test_sleepstudy_reml_theta_matches_lme4_nloptwrap above).
_SLEEP_THETA = (0.9667417739793641, 0.01516905889466504, 0.2309099532076919)
_SLEEP_SIGMA = 25.591795721655899


@pytest.fixture(scope="module")
def sleepstudy_data():
    return load_dataset("lme4", "sleepstudy")


def test_lmm_optimizer_nloptwrap_explicit_accepted(sleepstudy_data):
    """#5: control(optimizer="nloptwrap") — lme4's lmer default — used to raise
    NotImplementedError by name, even though the LMM path runs exactly it. Now
    accepted and identical to the default fit."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True,
            control={"optimizer": "nloptwrap"})
    np.testing.assert_allclose(m.theta, _SLEEP_THETA, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.sigma, _SLEEP_SIGMA, rtol=0, atol=1e-6)


@pytest.mark.parametrize("opt", ["bobyqa", "Nelder_Mead"])
def test_lmm_optimizer_alternatives_converge_to_same_optimum(
        sleepstudy_data, opt):
    """#1: control(optimizer="bobyqa"/"Nelder_Mead") was silently ignored —
    NLopt BOBYQA ran regardless. Now honored: the other two ported optimizers
    minimize the SAME profiled deviance and land on the same θ̂/σ̂."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True, control={"optimizer": opt})
    np.testing.assert_allclose(m.theta, _SLEEP_THETA, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma, _SLEEP_SIGMA, rtol=0, atol=1e-3)


def test_lmm_optimizer_unsupported_raises(sleepstudy_data):
    """#1: an un-ported optimizer must raise, not silently fall back to NLopt
    BOBYQA."""
    with pytest.raises(NotImplementedError, match="separate port"):
        gmm(_SLEEP_F, sleepstudy_data, REML=True,
            control={"optimizer": "nlminbwrap"})


def test_lmm_start_theta_array_gives_valid_fit(sleepstudy_data):
    """#3: start= (θ vector) was silently ignored (θ₀=identity always). A warm
    start near the optimum now produces a correct fit."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True, start=np.array(_SLEEP_THETA))
    np.testing.assert_allclose(m.theta, _SLEEP_THETA, rtol=0, atol=1e-4)


def test_lmm_start_theta_dict_form(sleepstudy_data):
    """#3: start={'theta': ...} — lme4's named-list form."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True,
            start={"theta": list(_SLEEP_THETA)})
    np.testing.assert_allclose(m.theta, _SLEEP_THETA, rtol=0, atol=1e-4)


def test_lmm_start_is_actually_used(sleepstudy_data):
    """#2+#3: a far start + a 2-evaluation cap leaves θ̂ pinned at that start —
    proving start= is the optimizer's θ₀ (not the identity [1,0,1]) AND that
    optCtrl's maxeval bites. If start were dropped, θ̂ would sit at the
    identity start instead."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True,
            start=np.array([5.0, 0.0, 5.0]),
            control={"optCtrl": {"maxeval": 2}})
    np.testing.assert_allclose(m.theta, [5.0, 0.0, 5.0], rtol=0, atol=1e-9)


def test_lmm_start_bad_shape_raises(sleepstudy_data):
    with pytest.raises(ValueError, match="shape"):
        gmm(_SLEEP_F, sleepstudy_data, REML=True, start=np.array([1.0, 2.0]))


def test_lmm_start_fixef_rejected(sleepstudy_data):
    """#3: lmer profiles β out of the deviance, so a beta/fixef start has no
    meaning and must raise rather than be silently dropped."""
    with pytest.raises(ValueError, match="profiles"):
        gmm(_SLEEP_F, sleepstudy_data, REML=True,
            start={"theta": list(_SLEEP_THETA), "fixef": [0.0, 0.0]})


def test_lmm_optctrl_maxeval_starves_optimizer(sleepstudy_data):
    """#2: optCtrl was silently dropped on the LMM path. A tiny maxeval now
    starves the optimizer so it stops far short of the optimum (much worse
    REML criterion) — proof the knob is threaded through."""
    full = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    capped = gmm(_SLEEP_F, sleepstudy_data, REML=True,
                 control={"optCtrl": {"maxeval": 2}})
    assert capped.REML_criterion > full.REML_criterion + 1.0


def test_lmm_optctrl_unknown_key_raises(sleepstudy_data):
    with pytest.raises(ValueError, match="unknown optCtrl key"):
        gmm(_SLEEP_F, sleepstudy_data, REML=True,
            control={"optCtrl": {"bogus": 1}})


# ---------------------------------------------------------------------------
# Prior weights on the LMM path (gmm-lmer-parity #4). weights= used to raise
# NotImplementedError; now lmer(y~x+(1|g), weights=w) is reproduced via √w
# row-scaling of the profiled-deviance design. Reference: lme4 4.x,
# lmer(Reaction~Days+(Days|Subject), sleepstudy, weights=rep(c(1,2,3),len=180)).
# ---------------------------------------------------------------------------

_W = np.tile([1.0, 2.0, 3.0], 60)                 # == R rep(c(1,2,3), len=180)
_W_THETA = (0.5720719237602743, 0.02526250546376536, 0.1520012796905056)
_W_SIGMA = 38.62892535113247
_W_BETA = (251.804690405274, 10.43587074687652)
_W_SE = (6.446985455645814, 1.573630563126574)
_W_REMLCRIT = 1778.291462756913
_W_ML_THETA = (0.5468139587813955, 0.02800468576341488, 0.1458069717176397)
_W_ML_DEV = 1786.531026746311


def test_lmm_weights_deviance_fn_bit_exact(sleepstudy_data):
    """#4 headline: hea's weighted profiled REML deviance, evaluated at lme4's
    exact θ̂, equals lme4's REMLcrit to the CHOLMOD floor. The √w row-scaling
    reproduces lme4's objective bit-for-bit; θ̂ only scatters because the
    surface is flat."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True, weights=_W)
    dev_at_lme4 = m._reml_deviance(np.array(_W_THETA)) - m._log_det_weights
    np.testing.assert_allclose(dev_at_lme4, _W_REMLCRIT, rtol=0, atol=1e-6)


def test_lmm_weights_reml_matches_lme4(sleepstudy_data):
    """#4: weighted REML fit reproduces lme4 (θ̂/σ̂/β̂/se at flat-surface
    optimizer scatter; criterion ~1e-8)."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True, weights=_W)
    np.testing.assert_allclose(m.theta, _W_THETA, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma, _W_SIGMA, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m._beta, _W_BETA, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m._se_beta, _W_SE, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.REML_criterion, _W_REMLCRIT, rtol=0, atol=1e-5)


def test_lmm_weights_ml_matches_lme4(sleepstudy_data):
    """#4: weighted ML fit + deviance (bit-exact at lme4's θ̂)."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=False, weights=_W)
    np.testing.assert_allclose(m.theta, _W_ML_THETA, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma, 38.63274937880161, rtol=0, atol=1e-3)
    dev_at_lme4 = m._ml_deviance(np.array(_W_ML_THETA)) - m._log_det_weights
    np.testing.assert_allclose(dev_at_lme4, _W_ML_DEV, rtol=0, atol=1e-6)


def test_lmm_weights_residuals_response_and_pearson(sleepstudy_data):
    """#4: residuals = y−μ on the original scale; the 'scaled' (Pearson)
    residuals fold in √w — matches lme4 residuals(type='pearson')."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True, weights=_W)
    np.testing.assert_allclose(
        m.residuals[:3],
        [-5.319909703504237, -15.53001223271121, -42.78891476191828],
        rtol=0, atol=5e-3)
    np.testing.assert_allclose(
        (m.scaled_residuals * m.sigma)[:3],
        [-5.319909703504237, -21.96275392332027, -74.1125743683764],
        rtol=0, atol=5e-3)


def test_lmm_unit_weights_reproduce_unweighted(sleepstudy_data):
    """#4: all-ones weights must reproduce the unweighted fit (√w≡1)."""
    m_w = gmm(_SLEEP_F, sleepstudy_data, REML=True, weights=np.ones(180))
    m_0 = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    np.testing.assert_allclose(m_w.theta, m_0.theta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(m_w.REML_criterion, m_0.REML_criterion,
                               rtol=0, atol=1e-9)


def test_lmm_weights_bad_length_raises(sleepstudy_data):
    with pytest.raises(ValueError, match="length"):
        gmm(_SLEEP_F, sleepstudy_data, REML=True, weights=np.ones(5))


# ---------------------------------------------------------------------------
# Fit → accessor contract (gmm._FIT_CONTRACT). Both fit paths must populate the
# post-fit state the shared accessor layer relies on; _assert_fit_contract()
# enforces it at fit time. Prompted by the _ranef/_Z_sp_solve coupling bug (#4).
# ---------------------------------------------------------------------------


def test_fit_contract_satisfied_by_lmm_fits(sleepstudy_data):
    """Every _FIT_CONTRACT attribute is present on a fitted LMM — REML, ML, and
    weighted. (The GLMM path is exercised by the glmer suite, which now also
    runs _assert_fit_contract on every fit.)"""
    for m in (
        gmm(_SLEEP_F, sleepstudy_data, REML=True),
        gmm(_SLEEP_F, sleepstudy_data, REML=False),
        gmm(_SLEEP_F, sleepstudy_data, REML=True, weights=_W),
    ):
        missing = [a for a in gmm._FIT_CONTRACT if not hasattr(m, a)]
        assert not missing, f"fit missing contract attrs: {missing}"


def test_fit_contract_is_enforced(sleepstudy_data):
    """_assert_fit_contract raises (naming the gap) when a contract attribute is
    absent — so the _ranef/_Z_sp_solve class of bug fails loudly at fit time."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    del m._Z_sp_solve
    with pytest.raises(RuntimeError, match=r"_Z_sp_solve"):
        m._assert_fit_contract()


# ---------------------------------------------------------------------------
# vcov() method (gmm-lmer-parity #14) — over the existing _vcov_beta_arr.
# ---------------------------------------------------------------------------


def test_vcov_method_matches_attr_and_se(fm06ML):
    """#14: vcov() == the vcov_beta attr, its diagonal is se², and
    correlation=True is unit-diagonal + symmetric."""
    m = fm06ML
    V = m.vcov().to_numpy()
    np.testing.assert_allclose(V, m.vcov_beta.to_numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.sqrt(np.diag(V)), m.se_bhat.to_numpy().ravel(), rtol=0, atol=1e-12)
    C = m.vcov(correlation=True).to_numpy()
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)
    np.testing.assert_allclose(C, C.T, atol=1e-12)


def test_vcov_full_joint_matches_lme4(sleepstudy_data):
    """#14: vcov(full=True) is the joint [b̂; β̂] conditional covariance
    (lme4 vcov(full=TRUE)). The bottom-right p×p block == the default vcov; the
    RE blocks inflate the postVar by the fixed-effect uncertainty. Labels are
    '<grp>.<level>.<comp>' then the fixed-effect names."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    Vf = m.vcov(full=True)
    A = Vf.to_numpy()
    assert A.shape == (38, 38)
    assert Vf.columns[:2] == ["Subject.308.(Intercept)", "Subject.308.Days"]
    assert Vf.columns[-2:] == ["(Intercept)", "Days"]
    np.testing.assert_allclose(A[-2:, -2:], m.vcov().to_numpy(), atol=1e-9)
    np.testing.assert_allclose(
        A[:2, :2],
        [[171.616396993957, -19.7195640120661],
         [-19.7195640120661, 6.96558466442668]], atol=1e-4)
    np.testing.assert_allclose(
        A[:2, 2:4],
        [[25.9108094732363, 1.72493958602702],
         [1.72493958602702, 1.65330175214122]], atol=1e-4)
    np.testing.assert_allclose(A, A.T, atol=1e-9)               # symmetric


# ---------------------------------------------------------------------------
# logLik(REML=) toggle (gmm-lmer-parity #19) — one fit yields both criteria,
# recomputing the other at the fitted θ̂ (lme4 devCrit). lme4 4.x reference.
# ---------------------------------------------------------------------------


def test_loglik_reml_toggle_matches_lme4(sleepstudy_data):
    """#19: logLik(REML=) recomputes the other criterion at the fitted θ̂ (no
    refit) — so a single REML fit reports both REML and ML log-likelihoods."""
    mR = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    np.testing.assert_allclose(mR.logLik(), -871.8141359800, atol=1e-5)
    np.testing.assert_allclose(mR.logLik(REML=True), -871.8141359800, atol=1e-5)
    np.testing.assert_allclose(mR.logLik(REML=False), -875.9929332160, atol=1e-5)
    mM = gmm(_SLEEP_F, sleepstudy_data, REML=False)
    np.testing.assert_allclose(mM.logLik(), -875.9696722445, atol=1e-5)
    np.testing.assert_allclose(mM.logLik(REML=False), -875.9696722445, atol=1e-5)
    np.testing.assert_allclose(mM.logLik(REML=True), -871.8368952841, atol=1e-5)


# ---------------------------------------------------------------------------
# ranef(condVar=) (gmm-lmer-parity #21) — now a method (was a property), over
# the posterior SDs _ranef() already computes. lme4 4.x postVar reference.
# ---------------------------------------------------------------------------


def test_ranef_condvar_matches_lme4_postvar(sleepstudy_data):
    """#21: ranef(condVar=True) appends per-level conditional-SD columns
    (√diag(postVar)) matching lme4's ranef(m, condVar=TRUE); the BLUP columns
    and the default ranef() are unchanged."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    plain = m.ranef()["Subject"]
    assert "(Intercept) condsd" not in plain.columns        # default: BLUPs only
    cv = m.ranef(condVar=True)["Subject"]
    np.testing.assert_allclose(
        cv["(Intercept) condsd"].to_numpy()[0], 12.0708569506, atol=1e-5)
    np.testing.assert_allclose(
        cv["Days condsd"].to_numpy()[0], 2.3048390209, atol=1e-5)
    np.testing.assert_allclose(
        cv["(Intercept)"].to_numpy(), plain["(Intercept)"].to_numpy(), atol=0)


def test_ranef_postvar_drop_whichel(sleepstudy_data):
    """#21: ranef(postVar=True) attaches the full (c×c×n_levels) conditional
    covariance arrays (lme4's postVar); drop= reduces scalar bars to level-named
    vectors; whichel= selects grouping factors. postVar diag == condsd²."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    rf = m.ranef(postVar=True)
    assert isinstance(rf, dict)                       # back-compat dict access
    pv = rf.postVar["Subject"]
    assert pv.shape == (2, 2, 18)
    np.testing.assert_allclose(
        pv[:, :, 0],
        [[145.705587520721, -21.4445035980931],
         [-21.4445035980931, 5.31228291228546]], atol=1e-4)
    cv = m.ranef(condVar=True)["Subject"]
    np.testing.assert_allclose(
        cv["(Intercept) condsd"].to_numpy()[0], np.sqrt(pv[0, 0, 0]), atol=1e-7)
    # drop on a scalar bar → level-named vector
    dy = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"), REML=True)
    dv = dy.ranef(drop=True)["Batch"]
    assert len(dv) == 6 and "A" in list(dv.names)
    # whichel selects the grouping factor (both its bars)
    m2 = gmm("Reaction ~ Days + (1|Subject) + (0+Days|Subject)",
             sleepstudy_data, REML=True)
    assert set(m2.ranef(whichel="Subject")) == {"Subject", "Subject.1"}


# ---------------------------------------------------------------------------
# lme4 predicate / extractor surface (gmm-lmer-parity #10–#12, #17, #18, #20):
# isREML/isLMM/isGLMM/isNLMM/isSingular, getME, VarCorr, coef (per-group),
# getData, extractAIC, rePCA — additive shims over the fit→accessor contract,
# each also reachable through the hea.R generic of the same name.
# ---------------------------------------------------------------------------


def test_predicates_match_lme4(sleepstudy_data, fm06ML):
    """#17: isREML/isLMM/isGLMM/isNLMM/isSingular on an LMM. A REML LMM is
    isREML; an ML LMM is not; neither is a GLMM. isSingular flags a boundary
    fit (Dyestuff2, variance → 0) and is False for sleepstudy."""
    import hea.R as R
    mR = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    assert (mR.isREML(), mR.isLMM(), mR.isGLMM(), mR.isNLMM()) == \
        (True, True, False, False)
    assert fm06ML.isREML() is False and fm06ML.isLMM() is True
    assert mR.isSingular() is False
    # classic singular fit: lmer(Yield ~ 1 + (1|Batch), Dyestuff2) → θ ≈ 0
    sing = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff2"), REML=True)
    assert sing.isSingular() is True
    # hea.R generics route to the same answers (and are merMod-only)
    assert R.isREML(mR) is True and R.isLMM(mR) is True
    assert R.isGLMM(mR) is False and R.isSingular(sing) is True
    with pytest.raises(TypeError):
        R.isREML(object())


def test_getME_matches_contract_and_lme4(sleepstudy_data):
    """#11: getME(name) extracts named pieces off the fit→accessor contract.
    θ/β match the pinned lme4 values; b = Λu equals the concatenated ranef
    BLUPs; lower = [0, −inf, 0]; dims are right; an unknown name raises."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    np.testing.assert_allclose(m.getME("theta"), _SLEEP_THETA, atol=1e-7)
    np.testing.assert_allclose(m.getME("beta"), m._beta, atol=0)
    np.testing.assert_array_equal(m.getME("fixef"), m.getME("beta"))
    assert m.getME("n") == 180 and m.getME("p") == 2 and m.getME("q") == 36
    assert m.getME("n_rtrms") == 1 and m.getME("n_rfacs") == 1
    np.testing.assert_array_equal(m.getME("lower"), [0.0, -np.inf, 0.0])
    assert m.getME("X").shape == (180, 2) and m.getME("Z").shape == (180, 36)
    np.testing.assert_allclose(m.getME("Zt"), m.getME("Z").T, atol=0)
    np.testing.assert_allclose(m.getME("mu"), m.fitted, atol=0)
    np.testing.assert_allclose(m.getME("y"), m.y, atol=0)
    assert m.getME("sigma") == pytest.approx(m.sigma)
    assert m.getME("is_REML") is True and m.getME("REML") == 2   # p when REML
    # b = Λu == the BLUPs ranef() reports (level-major, 2 components/level)
    b = m.getME("b").reshape(18, 2)
    rf = m.ranef()["Subject"]
    np.testing.assert_allclose(b[:, 0], rf["(Intercept)"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(b[:, 1], rf["Days"].to_numpy(), atol=1e-9)
    with pytest.raises(ValueError, match="not supported"):
        m.getME("Cm")               # a real lme4 name hea doesn't expose


def test_getME_advanced_names_match_lme4(sleepstudy_data):
    """#11: the advanced getME pieces — RX/A/Tlist/ST/mmList/Tp/devcomp and the
    per-term dims — bit-exact to lme4 2.0.1. RX (fixed-effect Cholesky) and A
    (=Λᵀ Zᵀ) are basis-invariant; RZX is in hea's Cholesky basis but satisfies
    the invariant RXᵀRX + RZXᵀRZX = XᵀX."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    np.testing.assert_allclose(
        m.getME("RX"),
        [[3.7859220865198, 2.299136523593964], [0.0, 16.555807462279105]],
        atol=1e-8)
    RX, RZX, X = m.getME("RX"), m.getME("RZX"), m.getME("X")
    np.testing.assert_allclose(RX.T @ RX + RZX.T @ RZX, X.T @ X, atol=1e-6)
    A = m.getME("A")
    assert A.shape == (36, 180)
    np.testing.assert_allclose(
        A[:3, :3],
        [[0.96674177397936, 0.98191083287403, 0.99707989176869],
         [0.0, 0.23090995320769, 0.46181990641538], [0.0, 0.0, 0.0]], atol=1e-8)
    np.testing.assert_allclose(
        m.getME("Tlist")[0],
        [[0.96674177397936, 0.0], [0.01516905889467, 0.23090995320769]], atol=1e-8)
    np.testing.assert_allclose(
        m.getME("ST")["Subject"],
        [[0.96674177397936, 0.0], [0.01569091075089, 0.23090995320769]], atol=1e-8)
    np.testing.assert_allclose(m.getME("mmList")[0][:3], [[1, 0], [1, 1], [1, 2]])
    np.testing.assert_array_equal(m.getME("Tp"), [0, 3])
    assert m.getME("p_i").tolist() == [2] and m.getME("l_i").tolist() == [18]
    assert m.getME("q_i").tolist() == [36] and m.getME("m_i").tolist() == [3]
    assert m.getME("k") == 1 and m.getME("m") == 3
    cmp = m.getME("devcomp")["cmp"]
    np.testing.assert_allclose(cmp["ldL2"], 75.9613332066632, atol=1e-6)
    np.testing.assert_allclose(cmp["ldRX2"], 8.27605283509015, atol=1e-6)
    np.testing.assert_allclose(cmp["wrss"], 98881.5684446342, atol=1e-3)
    np.testing.assert_allclose(cmp["ussq"], 17697.7530254616, atol=1e-3)
    np.testing.assert_allclose(cmp["sigmaREML"], 25.5917957216559, atol=1e-6)
    np.testing.assert_allclose(cmp["sigmaML"], 25.4492219341985, atol=1e-6)
    assert m.getME("devcomp")["dims"]["nmp"] == 178


def test_VarCorr_matches_lme4(sleepstudy_data):
    """#10: VarCorr() repackages sd_re/corr_re/sigma into lme4's object — a
    per-bar covariance σ²ΛΛᵀ with stddev/correlation views and residual sc.
    Std.Dev. = [24.740, 5.922], corr = 0.0655, sc = 25.592 (lme4 4.x)."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    vc = m.VarCorr()
    assert vc.sc == pytest.approx(25.5918, abs=1e-3)
    np.testing.assert_allclose(vc.stddev("Subject"), [24.7404, 5.9221], atol=1e-2)
    np.testing.assert_allclose(vc.stddev("Subject"), m.sd_re["Subject"], atol=0)
    assert vc.correlation("Subject")[0, 1] == pytest.approx(0.0655, abs=1e-3)
    sd, corr = m.sd_re["Subject"], m.corr_re["Subject"]
    np.testing.assert_allclose(vc["Subject"], np.outer(sd, sd) * corr, atol=1e-9)
    # print layout (lme4 print.VarCorr default): Std.Dev. + Corr, Residual row,
    # no Variance column.
    txt = repr(vc)
    assert "Std.Dev." in txt and "Residual" in txt and "Corr" in txt
    assert "Variance" not in txt
    assert set(vc.as_dict()) == {"Subject", "sc"}


def test_coef_is_fixef_plus_ranef(sleepstudy_data):
    """#12: coef() = fixef broadcast to each level + the matching ranef BLUP
    (lme4 coef.merMod). fixef() stays fixed-effects-only after the change."""
    import hea.R as R
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    cf = m.coef()["Subject"]
    fx = dict(zip(m.column_names, np.asarray(m._beta).ravel()))
    rf = m.ranef()["Subject"]
    np.testing.assert_allclose(
        cf["(Intercept)"].to_numpy(),
        fx["(Intercept)"] + rf["(Intercept)"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(
        cf["Days"].to_numpy(), fx["Days"] + rf["Days"].to_numpy(), atol=1e-12)
    row0 = cf.row(0, named=True)                       # Subject 308 (first level)
    assert row0["Subject"] == "308"
    np.testing.assert_allclose(
        [row0["(Intercept)"], row0["Days"]], [253.6637, 19.6663], atol=1e-2)
    # R.coef(gmm) → per-group dict; R.fixef → fixed effects only
    assert isinstance(R.coef(m), dict)
    np.testing.assert_allclose(list(R.fixef(m)), [251.4051, 10.4673], atol=1e-3)


def test_extractAIC_and_rePCA_match_lme4(sleepstudy_data, fm06ML):
    """#20: extractAIC()=(edf,AIC) on the fit's own criterion (REML AIC for a
    REML fit, ML for ML); rePCA() = PC SDs of the relative RE covariance
    (basis-invariant → bit-exact to lme4 despite hea's distinct Λ basis)."""
    mR = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    edf, aic = mR.extractAIC()
    assert edf == 6
    np.testing.assert_allclose(aic, 1755.6283, atol=1e-2)        # REML AIC
    np.testing.assert_allclose(mR.extractAIC(k=0)[1], -2 * mR.logLik(), atol=1e-9)
    assert fm06ML.extractAIC() == (6, pytest.approx(fm06ML.AIC))  # ML AIC
    # rePCA: PC SDs of the relative covariance, largest first
    pcs = mR.rePCA()["Subject"]
    np.testing.assert_allclose(pcs, [0.96687, 0.23088], atol=1e-4)
    sd_rel = mR.sd_re["Subject"] / mR.sigma
    Srel = np.outer(sd_rel, sd_rel) * mR.corr_re["Subject"]
    np.testing.assert_allclose(np.prod(pcs), np.sqrt(np.linalg.det(Srel)), atol=1e-9)


def test_getData_and_R_generic_routing(sleepstudy_data):
    """#18 + routing: getData() returns the fit's data frame; the new hea.R
    generics delegate to the methods and thread the new method kwargs
    (vcov correlation=, logLik REML=)."""
    import hea.R as R
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    assert m.getData() is m.data
    assert R.getData(m) is m.data
    assert R.VarCorr(m).sc == pytest.approx(m.sigma)
    np.testing.assert_array_equal(R.getME(m, "theta"), m.getME("theta"))
    assert R.extractAIC(m) == m.extractAIC()
    np.testing.assert_allclose(R.rePCA(m)["Subject"], m.rePCA()["Subject"], atol=0)
    np.testing.assert_allclose(
        R.vcov(m, correlation=True).to_numpy(),
        m.vcov(correlation=True).to_numpy(), atol=0)
    assert R.logLik(m, REML=False) == pytest.approx(m.logLik(REML=False))
    with pytest.raises(TypeError):       # REML= meaningless off a mixed model
        R.logLik(object(), REML=True)


# ---------------------------------------------------------------------------
# Method-signature fills (gmm-lmer-parity #22, #23, #25, #26, #27): the
# remaining lme4 keyword surface. Real-behavior args (re.form=NA sentinel,
# boot_scale="vcov", residuals scaled=) are honored; deep ports (newparams,
# newdata, partial-bars, custom zeta) raise NotImplementedError rather than
# silently no-op; the bit-exact simulate/bootMer RNG path is untouched.
# ---------------------------------------------------------------------------


def test_residuals_type_default_and_scaled(sleepstudy_data):
    """#27: LMM residual types all collapse to y−μ; the R generic defaults to
    'response' (lme4's LMM default, not 'deviance'); scaled= divides by σ̂."""
    import hea.R as R
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    r = m.residuals_of("response")
    for t in ("deviance", "working", "pearson"):
        np.testing.assert_allclose(m.residuals_of(t), r, atol=1e-12)
    np.testing.assert_allclose(
        m.residuals_of("response", scaled=True), r / m.sigma, atol=1e-12)
    np.testing.assert_allclose(R.residuals(m), r, atol=1e-12)        # default = response
    np.testing.assert_allclose(R.residuals(m, scaled=True), r / m.sigma, atol=1e-12)
    with pytest.raises(TypeError):              # scaled= is mixed-model only
        R.residuals(object(), scaled=True)


def test_predict_population_sentinels(sleepstudy_data):
    """#22: re_form accepts lme4's no-RE sentinels (False / 'NA' / NaN / '~0')
    for population-level (Xβ); a partial-bars formula and newparams= raise."""
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    pop = m.predict(re_form=False)["fit"].to_numpy()
    for s in ("NA", "~0", "~ 0", float("nan")):
        np.testing.assert_allclose(m.predict(re_form=s)["fit"].to_numpy(), pop, atol=0)
    assert not np.allclose(pop, m.predict()["fit"].to_numpy())   # pop != full-RE
    # a re_form naming the full bar == including all RE
    np.testing.assert_allclose(
        m.predict(re_form="~(1+Days|Subject)")["fit"].to_numpy(),
        m.predict()["fit"].to_numpy(), atol=1e-9)


def test_predict_newparams_partial_bars_na_action(sleepstudy_data):
    """#22: predict newparams (β/θ substitution with the fitted modes kept),
    partial-bars re_form (a subset of RE terms), and na.omit / na.exclude — all
    bit-exact / correct vs lme4 2.0.1."""
    import polars as pl
    fm7 = gmm("Reaction ~ Days + (1|Subject) + (0+Days|Subject)",
              sleepstudy_data, REML=True)
    np.testing.assert_allclose(
        fm7.predict(re_form="~(1|Subject)")["fit"].to_numpy()[:3],
        [252.917769618, 263.385055577, 273.852341537], atol=1e-4)
    np.testing.assert_allclose(
        fm7.predict(re_form="~(0+Days|Subject)")["fit"].to_numpy()[:3],
        [251.405104848, 271.195887848, 290.986670847], atol=1e-4)
    fm = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    np.testing.assert_allclose(                       # substitute β, keep modes
        fm.predict(newparams={"beta": [240, 11]})["fit"].to_numpy()[:3],
        [242.25855095, 262.457526715, 282.656502481], atol=1e-4)
    np.testing.assert_allclose(                       # population at new β
        fm.predict(re_form="NA", newparams={"beta": [250, 10]})["fit"].to_numpy()[:3],
        [250.0, 260.0, 270.0], atol=1e-9)
    with pytest.raises(ValueError, match="length"):
        fm.predict(newparams={"beta": [1, 2, 3]})
    # na.omit drops NA rows; na.exclude pads them back with NaN
    nd = sleepstudy_data.head(4).with_columns(pl.Series("Days", [0.0, None, 2.0, 3.0]))
    base = fm.predict(sleepstudy_data.head(4))["fit"].to_numpy()
    om = fm.predict(nd, na_action="na.omit")
    assert om.height == 3
    np.testing.assert_allclose(om["fit"].to_numpy(), base[[0, 2, 3]], atol=1e-9)
    ex = fm.predict(nd, na_action="na.exclude")["fit"].to_numpy()
    assert ex.shape == (4,) and np.isnan(ex[1])
    np.testing.assert_allclose(ex[[0, 2, 3]], base[[0, 2, 3]], atol=1e-9)


def test_simulate_newparams_newdata_bit_exact(sleepstudy_data):
    """#25: simulate newparams (β/θ/σ substitution, draw order preserved) and
    newdata (fresh design — newdata's grouping levels set the RE-draw dimension
    q) reproduce R's simulate(seed=42) bit-for-bit (lme4 2.0.1)."""
    fm = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    np.testing.assert_allclose(                       # default preserved
        fm.simulate(nsim=1, seed=42)["sim_1"].to_numpy()[:3],
        [265.247804221, 271.209734781, 238.864548297], atol=1e-2)
    np.testing.assert_allclose(                       # newparams β=(250,10)
        fm.simulate(nsim=1, seed=42,
                    newparams={"beta": [250, 10], "theta": fm.theta,
                               "sigma": fm.sigma})["sim_1"].to_numpy()[:3],
        [263.842699372, 269.337343973, 236.524871529], atol=1e-2)
    np.testing.assert_allclose(                       # newdata (subject 308 ×6)
        fm.simulate(nsim=1, seed=42,
                    newdata=sleepstudy_data.head(6))["sim_1"].to_numpy()[:3],
        [294.616627039, 309.182078615, 310.994409646], atol=1e-2)


def test_profile_vector_bar_matches_lme4(sleepstudy_data):
    """#24: profile() supports vector bars (1+x|g) — the component SDs and the
    correlation are profiled on the sd/cor scale (a constrained re-optimization
    of the relative-Cholesky θ); confint matches lme4 2.0.1 across every
    parameter. Plus the maxpts / which / signames / prof_scale tuning args."""
    fm = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    ci = fm.confint(method="profile")
    d = {r["parameter"]: (list(r.values())[1], list(r.values())[2])
         for r in ci.iter_rows(named=True)}
    np.testing.assert_allclose(d[".sig01"], [14.3815, 37.7160], atol=2e-2)  # sd_int
    np.testing.assert_allclose(d[".sig02"], [3.8012, 8.7534], atol=2e-2)    # sd_Days
    np.testing.assert_allclose(d[".sig03"], [-0.4815, 0.6850], atol=2e-2)   # cor
    np.testing.assert_allclose(d[".sigma"], [22.8983, 28.8580], atol=2e-2)
    np.testing.assert_allclose(d["(Intercept)"], [237.6807, 265.1295], atol=2e-2)
    np.testing.assert_allclose(d["Days"], [7.3587, 13.5759], atol=2e-2)
    # which= profiles a subset (maxpts is R's name for n_grid)
    assert [r["parameter"] for r in
            fm.profile(which=["Days"], maxpts=60).confint().iter_rows(named=True)] == ["Days"]
    # signames= controls the variance-component labelling
    assert [s[0] for s in fm._variance_component_specs(signames=False)] == \
        ["sd_(Intercept)|Subject", "sd_Days|Subject", "cor_Days.(Intercept)|Subject"]
    # prof_scale='varcov' — the diagonal (variances, σ²) match lme4 (the
    # off-diagonal covariance is numerically fragile in lme4 too — it warns NAs)
    # which= the two asserted components: profile CIs are per-parameter, so this
    # is identical to profiling all 6 on the varcov scale but ~3× faster.
    vc = {r["parameter"]: (list(r.values())[1], list(r.values())[2])
          for r in fm.profile(prof_scale="varcov", which=[".sigma", ".sig01"])
          .confint().iter_rows(named=True)}
    np.testing.assert_allclose(vc[".sigma"], [524.331, 832.784], atol=3)     # σ²
    np.testing.assert_allclose(vc[".sig01"], [207.007, 1422.500], atol=3)    # var_int


def test_confint_oldnames_signames_zeta():
    """#23: confint oldNames/signames relabel the variance components to the
    descriptive sd_…/sigma names (same CI values as .sig0i); zeta overrides the
    ±Φ⁻¹((1+level)/2) cutoff. Pinned to lme4 2.0.1 (Dyestuff)."""
    import polars as pl
    dy = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"), REML=True)
    default = dy.confint(method="profile")
    assert default["parameter"].to_list() == [".sig01", ".sigma", "(Intercept)"]
    new = dy.confint(method="profile", oldNames=False)
    assert new["parameter"].to_list() == [
        "sd_(Intercept)|Batch", "sigma", "(Intercept)"]
    np.testing.assert_allclose(             # same CI, only relabeled
        new.select(new.columns[1:]).to_numpy(),
        default.select(default.columns[1:]).to_numpy(), atol=1e-6)
    d = {r["parameter"]: tuple(list(r.values())[1:])
         for r in default.iter_rows(named=True)}
    np.testing.assert_allclose(d[".sig01"], [12.19853, 84.06305], atol=2e-2)
    np.testing.assert_allclose(d[".sigma"], [38.22998, 67.65770], atol=2e-2)
    np.testing.assert_allclose(d["(Intercept)"], [1486.4515, 1568.5485], atol=2e-2)
    # zeta overrides the cutoff (zeta=1 → ~68% interval, strictly narrower)
    z1 = dy.confint(method="profile", zeta=1.0)
    zc = z1.filter(pl.col("parameter") == "(Intercept)").row(0)[1:]
    assert zc[0] > d["(Intercept)"][0] and zc[1] < d["(Intercept)"][1]


def test_na_exclude_padding_and_varcorr_alignment(sleepstudy_data):
    """#18 polish: na.action='na.exclude' pads fitted()/residuals() back to the
    full model-frame length with NaN at the dropped rows (R's napredict/naresid);
    the VarCorr / RE-table Std.Dev. column right-aligns so decimals line up
    (lme4's print layout)."""
    import re
    import polars as pl
    import hea.R as R
    d_na = sleepstudy_data.with_columns(
        pl.when(pl.int_range(pl.len()) == 5).then(None)
        .otherwise(pl.col("Reaction")).alias("Reaction"))
    m = gmm(_SLEEP_F, d_na, na_action="na.exclude")
    assert m.n == 179
    fv, rs = R.fitted(m), R.resid(m)
    assert len(fv) == 180 and np.isnan(fv[5]) and np.isfinite(fv[[0, 1, 2, 6]]).all()
    assert len(rs) == 180 and np.isnan(rs[5]) and np.isfinite(rs[[0, 1, 2, 6]]).all()
    assert len(R.fitted(gmm(_SLEEP_F, d_na))) == 179      # na.omit: unpadded
    # VarCorr Std.Dev. decimals align (right-justified numeric column)
    fm = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    lines = repr(fm.VarCorr()).split("\n")
    dot_cols = []
    for ln in lines[1:]:
        mt = re.search(r"\d+\.\d+", ln)
        if mt:
            dot_cols.append(ln.index(mt.group()) + mt.group().index("."))
    assert len(dot_cols) == 3 and len(set(dot_cols)) == 1   # all . at one column


def test_confint_boot_scale_vcov_and_cosmetic_args(fm01ML):
    """#23: boot_scale='vcov' reports variance components squared (= sdcor²);
    zeta= raises; quiet/oldNames/signames are accepted (no-op labels)."""
    m = fm01ML                                          # Dyestuff, scalar bar
    sd = m._boot_profile_stat("sdcor")
    vc = m._boot_profile_stat("vcov")
    assert np.isclose(vc[".sig01"], sd[".sig01"] ** 2)
    assert np.isclose(vc[".sigma"], sd[".sigma"] ** 2)
    assert np.isclose(vc["(Intercept)"], sd["(Intercept)"])      # fixef unscaled
    # Wald with the descriptive (oldNames=False) labels for the var components
    w = m.confint(method="Wald", quiet=True, oldNames=False, signames=False)
    assert w.height > 0 and "sd_(Intercept)|Batch" in w["parameter"].to_list()
    ci = m.confint(method="boot", boot_scale="vcov", nsim=25, seed=1)
    bounds = ci.select(ci.columns[1:]).to_numpy().astype(float)
    assert ci.height == 3 and np.isfinite(bounds).all()


def test_simulate_bootMer_re_form_preserve_bit_exact(fm01ML):
    """#25/#26: the re_form=NA sentinel reproduces the default (use_u=False)
    draw bit-for-bit; a no-op newdata/newparams also reproduce it; bootMer
    threads re_form through simulate without perturbing the bit-exact RNG."""
    m = fm01ML
    s0 = m.simulate(nsim=3, seed=7)
    s1 = m.simulate(nsim=3, seed=7, re_form="NA")
    np.testing.assert_array_equal(s0.to_numpy(), s1.to_numpy())
    # newdata=self.data and an empty newparams are no-ops → same draw
    np.testing.assert_array_equal(
        m.simulate(nsim=3, seed=7, newdata=m.data).to_numpy(), s0.to_numpy())
    np.testing.assert_array_equal(
        m.simulate(nsim=3, seed=7, newparams={}).to_numpy(), s0.to_numpy())
    b0 = m.bootMer(lambda x: np.asarray(x._beta), nsim=4, seed=3)
    b1 = m.bootMer(lambda x: np.asarray(x._beta), nsim=4, seed=3, re_form="NA")
    np.testing.assert_array_equal(b0.t, b1.t)


def test_anova_single_model_sequential_F(sleepstudy_data):
    """#13: anova(m) for one mixed model — the Type-I sequential fixed-effect
    F-table (lme4 anova.merMod). effects = RX·β̂; per-term SS = Σ effects²;
    F = MeanSq/σ̂²; no p-value (no exact denominator df)."""
    import hea.R as R
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    a = R.anova(m)
    assert a[""].to_list() == ["Days"] and a["npar"].to_list() == [1]
    np.testing.assert_allclose(a["Sum Sq"][0], 30030.9390201738, atol=1e-3)
    np.testing.assert_allclose(a["Mean Sq"][0], 30030.9390201738, atol=1e-3)
    np.testing.assert_allclose(a["F value"][0], 45.85296, atol=1e-4)
    # a multi-column factor term (Machine: 3 levels → 2 cols, one term)
    mm = gmm("score ~ Machine + (1|Worker)",
             load_dataset("nlme", "Machines"), REML=True)
    am = R.anova(mm)
    assert am["npar"].to_list() == [2]
    np.testing.assert_allclose(am["Sum Sq"][0], 1755.263333, atol=1e-3)
    np.testing.assert_allclose(am["F value"][0], 87.79816, atol=1e-4)


def test_influence_family_matches_lme4(sleepstudy_data):
    """#20: the merMod influence diagnostics — hatvalues / cooks.distance /
    rstudent (closed form) and influence() (case/group-deletion refits) —
    bit-exact to lme4 2.0.1."""
    import hea.R as R
    fm = gmm(_SLEEP_F, sleepstudy_data, REML=True)
    h = fm.hatvalues()
    np.testing.assert_allclose(
        h[:3], [0.229304037742, 0.1697299938, 0.126823715021], atol=1e-6)
    np.testing.assert_allclose(h.sum(), 29.0221239507, atol=1e-4)
    np.testing.assert_allclose(
        fm.rstudent()[:3],
        [-0.182653763073, -0.627179564317, -1.76447385118], atol=1e-6)
    np.testing.assert_allclose(
        fm.cooks_distance()[:3],
        [0.00496313251478, 0.0402062018559, 0.22609918476], atol=1e-6)
    np.testing.assert_allclose(R.hatvalues(fm), h)            # generic routes
    # Dyestuff (scalar bar) — closed-form cooks + case/group-deletion influence
    dy = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"), REML=True)
    np.testing.assert_allclose(
        dy.cooks_distance()[:4],
        [0.117739352431, 0.466667001741, 0.466667001741, 0.00975819830532],
        atol=1e-6)
    infg = dy.influence(groups="Batch")          # 6 group-deletion refits
    np.testing.assert_allclose(
        infg.dfbeta().ravel(), [4.5, -0.1, -7.3, 5.9, -14.5, 11.5], atol=1e-3)
    np.testing.assert_allclose(
        infg.cooks_distance(),
        [0.0449141639196, 2.21798340342e-05, 0.118196335569,
         0.0772080022736, 0.466331010572, 0.293328305104], atol=1e-5)
    info = dy.influence()                         # 30 obs-deletion refits
    np.testing.assert_allclose(
        info.dfbeta().ravel()[:4],
        [-1.40934940504, 2.78490516842, 2.78490516842, -0.399076317567], atol=1e-4)
    np.testing.assert_allclose(
        info.cooks_distance()[:4],
        [0.00511038517164, 0.0199543278625, 0.0199543278625, 0.000409758709795],
        atol=1e-6)
    np.testing.assert_allclose(R.cooks_distance(infg), infg.cooks_distance())


def test_lmer_rejects_glmer_keys_and_wires_optinfo(sleepstudy_data):
    """#6 remainder: lmerControl() rejects glmer-only inner-loop keys; the LMM
    fit now populates m.optinfo via calc.derivs (the post-fit gradient/Hessian
    checkConv that was previously inert). A clean fit yields no message; a
    boundary (singular) fit surfaces lme4's message in optinfo + the summary."""
    import io
    import contextlib
    for key in ("tolPwrss", "compDev", "nAGQ0initStep", "check.response.not.const"):
        with pytest.raises(ValueError, match="glmer-only"):
            gmm(_SLEEP_F, sleepstudy_data, control={key: 1})
    m = gmm(_SLEEP_F, sleepstudy_data, REML=True, control={"restart_edge": False})
    assert m.optinfo["derivs"] is not None
    np.testing.assert_allclose(m.optinfo["derivs"]["gradient"], 0.0, atol=1e-2)
    assert m.optinfo["conv"]["lme4"]["messages"] == []          # clean → no warning
    sing = gmm("Yield ~ 1 + (1|Batch)",
               load_dataset("lme4", "Dyestuff2"), REML=True)
    assert any("singular" in msg for msg in sing.optinfo["conv"]["lme4"]["messages"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sing.summary()
    assert "isSingular" in buf.getvalue()


# ---------------------------------------------------------------------------
# Ch 4: Building Linear Mixed Models
# ---------------------------------------------------------------------------


def test_bates_4_1_machines_fm10_ML():
    """fm10 <- lmer(score ~ Machine + (1|Worker) + (1|Machine:Worker),
                    Machines, REML=FALSE)"""
    data = load_dataset("nlme", "Machines")
    m = gmm(
        "score ~ Machine + (1|Worker) + (1|Machine:Worker)",
        data, REML=False,
    )

    assert m.n == 54
    assert m.n_groups == {"Worker": 6, "Machine:Worker": 18}
    _assert_ml_summary(
        m, AIC=237.2694, BIC=249.2034, loglike=-112.6347,
        deviance=225.2694, df_resid=48,
    )
    np.testing.assert_allclose(m.sigma, 0.9616, atol=5e-3)
    _assert_re_scalar(m, "Machine:Worker", 3.3970)
    _assert_re_scalar(m, "Worker",         4.3645)
    _assert_fixed(m, "(Intercept)", 52.3556, se=2.2692, tval=23.07)
    _assert_fixed(m, "MachineB",      7.9667, se=1.9873, tval=4.009)
    _assert_fixed(m, "MachineC",     13.9167, se=1.9873, tval=7.003)


def test_bates_4_2_ergostool_fm16_ML():
    """fm16 <- lmer(effort ~ 1 + (1|Subject) + (1|Type),
                    ergoStool, REML=FALSE)"""
    data = load_dataset("nlme", "ergoStool")
    m = gmm("effort ~ 1 + (1|Subject) + (1|Type)", data, REML=False)

    assert m.n == 36
    assert m.n_groups == {"Subject": 9, "Type": 4}
    _assert_ml_summary(
        m, AIC=144.0224, BIC=150.3564, loglike=-68.0112,
        deviance=136.0224, df_resid=32,
    )
    np.testing.assert_allclose(m.sigma, 1.101, atol=5e-3)
    _assert_re_scalar(m, "Subject", 1.305)
    _assert_re_scalar(m, "Type",    1.505)
    _assert_fixed(m, "(Intercept)", 10.25)


def test_bates_4_2_ergostool_fm17_ML():
    """fm17 <- lmer(effort ~ 1 + Type + (1|Subject), ergoStool, REML=FALSE)"""
    data = load_dataset("nlme", "ergoStool")
    m = gmm("effort ~ 1 + Type + (1|Subject)", data, REML=False)

    _assert_ml_summary(
        m, AIC=134.1444, BIC=143.6456, loglike=-61.0722,
        deviance=122.1444, df_resid=30,
    )
    np.testing.assert_allclose(m.sigma, 1.037, atol=5e-3)
    _assert_re_scalar(m, "Subject", 1.256)
    _assert_fixed(m, "(Intercept)", 8.5556)
    _assert_fixed(m, "TypeT2",      3.8889)
    _assert_fixed(m, "TypeT3",      2.2222)
    _assert_fixed(m, "TypeT4",      0.6667)


# ---------------------------------------------------------------------------
# offset(...) — algebraic identity: fitting (y-off) ~ X + (1|g) gives the
# same β̂, û, and σ̂ as fitting y ~ X + offset(off) + (1|g). Fitted values
# shift by the offset; residuals are unchanged.
# ---------------------------------------------------------------------------


def test_predict_no_args_equals_fitted():
    """predict() with no args returns a 1-col DataFrame matching self.fitted —
    R's ``predict(fm)`` → ``na.omit(fitted(fm))`` short-circuit."""
    import polars as pl
    gpa = pl.read_csv("datasets/m-clark/gpa.csv")
    fm = gmm("gpa ~ occasion + (1 | student)", data=gpa)
    out = fm.predict()
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["fit"]
    np.testing.assert_array_equal(out["fit"].to_numpy(), fm.fitted)


def test_predict_newdata_eq_orig_matches_fitted():
    """predict(newdata=fit_data) matches fitted (round-trips X, Z, BLUP)."""
    import polars as pl
    gpa = pl.read_csv("datasets/m-clark/gpa.csv")
    fm = gmm("gpa ~ occasion + (1 | student)", data=gpa)
    p = fm.predict(newdata=gpa)["fit"].to_numpy()
    np.testing.assert_allclose(p, fm.fitted, atol=1e-10)


def test_predict_pinned_to_R_lmer():
    """predict.merMod cross-check: head values pinned to R 4.5 / lme4 4.5."""
    import polars as pl
    gpa = pl.read_csv("datasets/m-clark/gpa.csv")
    fm = gmm("gpa ~ occasion + (1 | student)", data=gpa)

    # Conditional (re.form=NULL) — includes BLUPs.
    r_conditional = [2.528319363, 2.634633649, 2.740947934, 2.847262220,
                     2.953576506]
    # R reference rounded to 9 sig figs; combined with BLAS-reduction-order
    # drift in the X·β + Z·b path (~1e-6 abs on Linux/OpenBLAS), atol=1e-5
    # is the honest floor. Tighter on platforms with matched BLAS to the
    # R machine but kept permissive to keep CI green across Python builds.
    np.testing.assert_allclose(
        fm.predict(newdata=gpa.head(5))["fit"].to_numpy(),
        r_conditional, atol=1e-5,
    )
    # Population (re.form=NA) — Xβ only.
    r_population = [2.599214286, 2.705528571, 2.811842857, 2.918157143,
                    3.024471429]
    np.testing.assert_allclose(
        fm.predict(newdata=gpa.head(5), re_form=False)["fit"].to_numpy(),
        r_population, atol=1e-9,
    )


def test_predict_allow_new_levels():
    """A new student id falls back to the population mean (Zb = 0)."""
    import polars as pl
    gpa = pl.read_csv("datasets/m-clark/gpa.csv")
    fm = gmm("gpa ~ occasion + (1 | student)", data=gpa)
    nd = pl.DataFrame({"occasion": [0, 1, 2], "student": [99999, 99999, 99999]})

    with pytest.raises(ValueError, match="new level"):
        fm.predict(newdata=nd)

    p = fm.predict(newdata=nd, allow_new_levels=True)["fit"].to_numpy()
    r_population = [2.599214286, 2.705528571, 2.811842857]
    np.testing.assert_allclose(p, r_population, atol=1e-9)


def test_predict_random_only():
    """random_only=True returns just ZΛu — sum equals fitted minus Xβ-offset."""
    import polars as pl
    gpa = pl.read_csv("datasets/m-clark/gpa.csv")
    fm = gmm("gpa ~ occasion + (1 | student)", data=gpa)
    pred_re = fm.predict(newdata=gpa.head(20), random_only=True)["fit"].to_numpy()
    X_head = fm._build_X_for_newdata(gpa.head(20))
    expected = fm.fitted[:20] - X_head @ fm._beta - fm._offset[:20]
    np.testing.assert_allclose(pred_re, expected, atol=1e-10)


def test_predict_se_fit_matches_R():
    """se.fit at the first 5 rows of gpa, pinned to R lme4 4.5."""
    import polars as pl
    gpa = pl.read_csv("datasets/m-clark/gpa.csv")
    fm = gmm("gpa ~ occasion + (1 | student)", data=gpa)
    ans = fm.predict(newdata=gpa.head(5), se_fit=True)
    assert isinstance(ans, pl.DataFrame)
    assert ans.columns == ["fit", "se.fit"]
    r_se = [0.09227442221, 0.09191399288, 0.09173324716, 0.09173324716,
            0.09191399288]
    np.testing.assert_allclose(ans["se.fit"].to_numpy(), r_se, atol=1e-6)


def test_predict_via_R_dispatcher():
    """hea.R.predict() routes to model.predict() — required for
    ``from hea.R import predict; predict(fm)`` ergonomics."""
    import polars as pl
    from hea.R import predict
    gpa = pl.read_csv("datasets/m-clark/gpa.csv")
    fm = gmm("gpa ~ occasion + (1 | student)", data=gpa)
    out = predict(fm)
    np.testing.assert_array_equal(out["fit"].to_numpy(), fm.fitted)


def test_gmm_offset_matches_y_minus_offset():
    import polars as pl

    rng = np.random.default_rng(0)
    n = 80
    g = np.repeat(np.arange(8), 10).astype(str)
    x = rng.standard_normal(n)
    o = rng.uniform(1.0, 3.0, n)
    u_re = rng.standard_normal(8)[np.repeat(np.arange(8), 10)]
    y = 0.5 + 0.7 * x + 1.5 * o + u_re + 0.3 * rng.standard_normal(n)
    d = pl.DataFrame({"y": y, "x": x, "o": o, "g": g})
    d_minus = d.with_columns((pl.col("y") - 1.5 * pl.col("o")).alias("y_minus"))

    m_off  = gmm("y ~ x + offset(1.5*o) + (1|g)", data=d, REML=True)
    m_pre  = gmm("y_minus ~ x + (1|g)", data=d_minus, REML=True)

    np.testing.assert_allclose(m_off._beta, m_pre._beta, atol=1e-10)
    np.testing.assert_allclose(m_off.sigma, m_pre.sigma, atol=1e-10)
    np.testing.assert_allclose(m_off.residuals, m_pre.residuals, atol=1e-10)
    # Fitted values shift by exactly the offset.
    np.testing.assert_allclose(
        m_off.fitted - m_pre.fitted, 1.5 * o, atol=1e-10,
    )
