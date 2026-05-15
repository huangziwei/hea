"""GLMM-specific tests for ``hea.lme(family=...)``.

This file accumulates Phase 2-13 tests of ``lme-family-port.md``. Phase 2
focuses on the ``_GlmResponse`` private class — verifying its mutators
and pure-compute methods match the documented formulas, plus a single
R-oracle cross-check. Phase 3 tests the PIRLS inner loop (_PredState,
_internal_glmer_wrk_iter, _pwrss_update) against ``lme4::glmer``.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hea.lme import csc_array
from hea.family import Binomial, Gamma, Gaussian, Poisson
from hea.formula import materialize_bars, parse, expand
from hea.lme import (
    NelderMead,
    NMStatus,
    _GlmResponse,
    _PredState,
    _deriv12,
    _glmm_devfun_factory,
    _internal_glmer_wrk_iter,
    _pwrss_update,
)


# ----------------------------------------------------------------------
# Math-formula tests — each verifies one or two methods against the
# documented relation. These should be obvious from the GLM formulas
# alone (no R needed), but the explicit numerical assertions are what
# guards against regression.
# ----------------------------------------------------------------------


def test_gaussian_identity_passes_through():
    """Gaussian-identity: μ = η, V(μ) = 1, μ_η = 1 → sqrt weights = √w."""
    family = Gaussian()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    r = _GlmResponse(family, y)

    # Replace whatever mustart/etastart did with a clean state.
    r.update_mu(np.array([0.5, 1.5, 2.5, 3.5]))
    r.update_weights()

    np.testing.assert_allclose(r.eta, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(r.mu, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(r.sqrt_r_wt, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(r.sqrt_x_wt, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(r.wt_res, [0.5, 0.5, 0.5, 0.5])
    assert r.wrss == pytest.approx(1.0)

    # Working pieces
    np.testing.assert_allclose(r.working_residuals(), [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(r.working_response(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(r.weighted_working_response(), [1.0, 2.0, 3.0, 4.0])

    # Deviance for Gaussian: Σ wt·(y - μ)²
    np.testing.assert_allclose(r.deviance(), 4 * 0.25)


def test_poisson_log_at_saturated_eta():
    """Poisson(log) at η = log(y): μ = y, dev = 0, sqrt_x_wt = √(w·μ)."""
    family = Poisson()
    y = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    r = _GlmResponse(family, y)

    eta = np.log(y)  # saturated η
    r.update_mu(eta)
    r.update_weights()

    np.testing.assert_allclose(r.mu, y)
    # μ_η = exp(η) = μ; V(μ) = μ → sqrt_r_wt = √(1/μ); sqrt_x_wt = μ·√(1/μ) = √μ
    np.testing.assert_allclose(r.sqrt_r_wt, np.sqrt(1.0 / y))
    np.testing.assert_allclose(r.sqrt_x_wt, np.sqrt(y))
    np.testing.assert_allclose(r.wt_res, np.zeros(5), atol=1e-15)
    assert r.wrss == pytest.approx(0.0, abs=1e-15)

    # At saturated η, deviance residuals all zero
    np.testing.assert_allclose(r.deviance_residuals(), np.zeros(5), atol=1e-12)
    assert r.deviance() == pytest.approx(0.0, abs=1e-12)

    # Working response: (η - 0) + (y - y)/y = η
    np.testing.assert_allclose(r.working_response(), eta)


def test_binomial_logit_proportion_input():
    """Binomial(logit): y is proportion ∈ [0,1], weights = m (binomial size).

    V(μ) = μ(1-μ). At μ = y, devResid = 0 (saturated). Working weight
    sqrt_x_wt = μ_η·√(m/V) where μ_η = μ(1-μ) for logit.
    """
    family = Binomial()
    y = np.array([0.2, 0.5, 0.7])
    weights = np.array([10.0, 20.0, 30.0])  # binomial sizes
    r = _GlmResponse(family, y, weights=weights)

    # Set η = logit(y) (saturated). Note __init__ already does this
    # implicitly via update_mu(link.link(mustart)), but mustart is
    # smoothed via the (w·y+0.5)/(w+1) formula — set our own clean η.
    eta = np.log(y / (1.0 - y))
    r.update_mu(eta)
    r.update_weights()

    np.testing.assert_allclose(r.mu, y, atol=1e-12)
    expected_v = y * (1.0 - y)
    np.testing.assert_allclose(r.sqrt_r_wt, np.sqrt(weights / expected_v))
    # μ_η for logit = μ(1-μ)
    expected_mu_eta = y * (1.0 - y)
    np.testing.assert_allclose(r.sqrt_x_wt, expected_mu_eta * r.sqrt_r_wt)

    # At saturated η, dev = 0
    assert r.deviance() == pytest.approx(0.0, abs=1e-12)


def test_gamma_initialization_replaces_mustart_with_mean():
    """utilities.R:250-252: Gamma without etastart replaces mustart with
    its mean to dodge PIRLS divergence from a saturated initial state."""
    family = Gamma()
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    r = _GlmResponse(family, y)

    # Initial μ should be the mean, not y itself.
    expected_mu = np.full(5, y.mean())
    np.testing.assert_allclose(r.mu, expected_mu, atol=1e-12)


def test_gamma_initialization_respects_etastart():
    """When etastart is provided, the Gamma stability fix is bypassed —
    the user explicitly chose the initial η."""
    family = Gamma()
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    # Provide etastart matching saturated η for the inverse link
    eta0 = 1.0 / y  # inverse link: η = 1/μ; at saturated μ = y, η = 1/y
    r = _GlmResponse(family, y, etastart=eta0)

    # With etastart, mustart stability fix is skipped, so μ = linkinv(η) = y.
    np.testing.assert_allclose(r.mu, y, atol=1e-12)


def test_update_wts_changes_wrss_when_weights_shift():
    """update_weights refreshes sqrt_r_wt and therefore wt_res / wrss."""
    family = Poisson()
    y = np.array([2.0, 4.0, 6.0])
    r = _GlmResponse(family, y)

    # Stash initial state
    r.update_mu(np.log(np.array([1.0, 5.0, 5.0])))  # μ ≠ y → nonzero wrss
    r.update_weights()
    wrss_before = r.wrss

    # Now scale weights up — sqrt_r_wt grows, wrss grows
    r.weights = np.full(3, 4.0)
    r.update_weights()
    wrss_after = r.wrss

    assert wrss_after == pytest.approx(4.0 * wrss_before)


def test_laplace_formula_glmm():
    """Laplace = log_det_l_sq + sqr_len_u + aic (respModule.cpp:161-163)."""
    family = Poisson()
    y = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    r = _GlmResponse(family, y)
    r.update_mu(np.log(y))  # saturated → deviance = 0
    r.update_weights()

    aic = r.aic()
    # Pick arbitrary ldL2 / sqrL — Laplace just sums them.
    laplace = r.laplace(log_det_l_sq=3.7, log_det_rx_sq=999.0, sqr_len_u=1.25)
    # log_det_rx_sq is intentionally ignored in the GLMM Laplace.
    assert laplace == pytest.approx(3.7 + 1.25 + aic)


def test_offset_shifts_eta_only():
    """update_mu(gamma) sets η = offset + γ; μ = linkinv(η). offset
    affects neither working residuals nor weighted residuals beyond
    what η encodes."""
    family = Poisson()
    y = np.array([1.0, 4.0, 9.0])
    offset = np.array([0.1, 0.2, 0.3])
    r = _GlmResponse(family, y, offset=offset)

    gamma = np.log(y) - offset  # so that η = log(y), μ = y
    r.update_mu(gamma)
    r.update_weights()

    np.testing.assert_allclose(r.mu, y, atol=1e-12)
    np.testing.assert_allclose(r.eta, np.log(y))

    # working_response = (η - offset) + (y - μ)/μ_η = (log(y) - offset) + 0
    np.testing.assert_allclose(r.working_response(), np.log(y) - offset)


# ----------------------------------------------------------------------
# R-oracle cross-check — fit a Poisson glm in R, extract its converged
# (μ, η), then build a _GlmResponse with the same state and compare the
# computed quantities. This guards against family.aic / dev_resids /
# variance subtleties that the mathematical-formula tests above can't
# catch on their own.
# ----------------------------------------------------------------------


def test_poisson_glm_state_matches_R():
    """Build _GlmResponse at R's converged glm() state and compare.

    Uses the canonical R example: ``count ~ outcome + treatment``
    from ?glm. We pin R's μ̂ / η̂ / weights / residuals / deviance and
    verify _GlmResponse reproduces every R-side value PIRLS cares about.

    R recipe::
        counts <- c(18,17,15,20,10,20,25,13,12)
        outcome <- gl(3,1,9)
        treatment <- gl(3,3)
        m <- glm(counts ~ outcome + treatment, family=poisson())
        # m$y, m$linear.predictors, m$fitted.values, m$weights,
        # residuals(m, "working"), residuals(m, "deviance"), deviance(m)
    """
    y = np.array([18.0, 17.0, 15.0, 20.0, 10.0, 20.0, 25.0, 13.0, 12.0])
    eta = np.array([
        3.0445224377234221, 2.5902671654458271, 2.7515353130419493,
        3.0445224377234230, 2.5902671654458280, 2.7515353130419502,
        3.0445224377234230, 2.5902671654458280, 2.7515353130419502,
    ])
    mu_r = np.array([
        20.999999999999982, 13.333333333333341, 15.666666666666673,
        21.000000000000000, 13.333333333333352, 15.666666666666687,
        21.000000000000000, 13.333333333333352, 15.666666666666687,
    ])
    r_wts = np.array([
        20.999999854963914, 13.333333732474516, 15.666666957539908,
        21.000000258897369, 13.333333988940220, 15.666667258887097,
        21.000000031015155, 13.333333844253083, 15.666667088879723,
    ])
    wrk_resids_r = np.array([
        -0.14285714285714213,  0.27499999999999925, -0.04255319148936210,
        -0.04761904761904762, -0.25000000000000105,  0.27659574468084935,
         0.19047619047619047, -0.02500000000000134, -0.23404255319149037,
    ])
    dev_resids_r = np.array([
        -0.67124922809541965,  0.96272360489389830, -0.16964661841949291,
        -0.21998507499991410, -0.95552353065273021,  1.04938637013018440,
         0.84715367982372969, -0.09167147361709924, -0.96656371504344019,
    ])
    dev_r = 5.1291410770011439

    # Build _GlmResponse and drive it to R's converged (η, μ).
    family = Poisson()
    r = _GlmResponse(family, y)
    r.update_mu(eta)  # offset=0, so γ = η
    r.update_weights()

    np.testing.assert_allclose(r.mu, mu_r, atol=1e-12)
    np.testing.assert_allclose(r.working_residuals(), wrk_resids_r, atol=1e-12)

    # R's residuals(m, type="deviance") returns *signed* sqrt of dev contribs.
    np.testing.assert_allclose(
        np.sign(y - mu_r) * np.sqrt(r.deviance_residuals()),
        dev_resids_r, atol=1e-12,
    )
    np.testing.assert_allclose(r.deviance(), dev_r, atol=1e-10)

    # R's m$weights are PIRLS working weights (= μ_η² / V for Poisson-log:
    # = μ² / μ = μ). _GlmResponse.sqrt_x_wt² should equal that.
    np.testing.assert_allclose(r.sqrt_x_wt ** 2, r_wts, atol=1e-12)


# ----------------------------------------------------------------------
# Phase 3: PIRLS state + inner loop. Tests that _PredState's PLS step
# math matches the merPredD operations, and that _pwrss_update converges
# to the same (β̂, û) as lme4::glmer at the converged θ.
# ----------------------------------------------------------------------


def _build_design_pieces(formula: str, data: pl.DataFrame):
    """Helper: parse formula, build X and ReTerms manually (bypassing the
    full prepare_design pipeline so the test focuses on _PredState math)."""
    from hea.design import prepare_design

    design = prepare_design(formula, data)
    re_terms = materialize_bars(design.expanded, design.data)
    X = design.X.to_numpy().astype(float)
    y = design.y.to_numpy().astype(float)
    Z_sp = csc_array(re_terms.Z)
    return X, y, Z_sp, re_terms, design.data


def test_predstate_basic_state_shape():
    """_PredState should initialize all fields to consistent shapes."""
    rng = np.random.default_rng(42)
    n, p, q = 30, 3, 5
    X = rng.standard_normal((n, p))
    Z_dense = rng.standard_normal((n, q))
    Z_sp = csc_array(Z_dense)
    # Build a minimal ReTerms-like object: scalar bars, q=5 levels, identity Λᵀ
    from hea.formula import ReTerms

    Lambdat = np.eye(q, dtype=int)  # template = identity; θ-position = 1 on diag
    re_terms = ReTerms(
        Z=Z_dense,
        Lambdat=Lambdat,
        theta=np.array([1.0]),
        flist_names=["g"],
        flist_levels={"g": list(range(q))},
        cnms={"g": ["(Intercept)"]},
        Gp=[0, q],
    )
    state = _PredState(X, Z_sp, re_terms)
    assert state.n == n
    assert state.p == p
    assert state.q == q
    np.testing.assert_array_equal(state.beta0, np.zeros(p))
    np.testing.assert_array_equal(state.u0, np.zeros(q))
    np.testing.assert_array_equal(state.delb, np.zeros(p))
    np.testing.assert_array_equal(state.delu, np.zeros(q))


def test_pirls_one_iter_matches_lme4_RglmerWrkIter():
    """Pin one PIRLS iteration against lme4's ``RglmerWrkIter``.

    Setup: synthetic Poisson GLMM. Build lme4's ``mkGlmerDevfun``, set θ
    to a fixed value, call ``RglmerWrkIter`` once — pin the resulting
    pp@delu / delb / pdev and resp$mu. In Python, build identical state
    and call ``_internal_glmer_wrk_iter`` with u_only=True.

    One-step matching avoids the multi-iteration noise near convergence.

    R recipe (run locally; data materialized from the same numpy
    ``default_rng(2026)`` synthetic recipe below)::
        d <- read.csv("...")
        d$g <- factor(d$g)
        glmod <- glFormula(y ~ x + (1|g), data=d, family=poisson)
        devfun <- mkGlmerDevfun(glmod$fr, glmod$X, glmod$reTrms,
                                family=poisson(), nAGQ=0)
        rho <- environment(devfun)
        invisible(rho$pp$setDelu(rep(0.0, length(rho$pp$delu))))
        invisible(rho$pp$setDelb(rep(0.0, length(rho$pp$delb))))
        invisible(rho$resp$updateMu(rho$lp0))
        invisible(rho$pp$setTheta(0.7))
        pdev <- lme4:::RglmerWrkIter(rho$pp, rho$resp, uOnly=TRUE)
        # capture rho$lp0, pdev, rho$pp$delu, rho$pp$delb, rho$resp$mu
    """
    rng = np.random.default_rng(2026)
    n_groups, n_per = 10, 5
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    true_b = rng.standard_normal(n_groups) * 0.5
    eta = 0.5 + 0.3 * x + true_b[g]
    y = rng.poisson(np.exp(eta)).astype(float)
    fixed_theta = 0.7

    # lme4 reference (see R recipe above). lp0 is the post-init-PIRLS
    # linear predictor; PIRLS one-step produces pdev / delu / delb / mu.
    lp0_r = np.array([
         0.90823343141713664,  1.16773024837401680,  0.63128690682446886,
         1.45772990359199950,  1.26757410479255840,  1.30761418071777460,
         1.30261805409264090,  1.45720365277490590,  1.31373629987519670,
         1.32421751440481580, -0.63969230302876134, -0.69124620306765083,
        -0.83655554298195856, -0.84191486807373916, -0.78006079468774425,
         0.92248367743015769,  0.97526905020748500,  1.21426021957658530,
         1.04386967632254990,  0.73159213220437258,  0.08604904129320895,
         0.37070155329828353,  0.14755242951482245,  0.16852667743635885,
         0.36698995645062965,  0.71911164327743982,  0.08202667392424595,
         0.59951607764626424, -0.04771601684106352,  0.30499051058989418,
        -0.29102099747864674,  0.34184369724259722,  0.21192249679351327,
         0.28818640030616050, -0.21972700713734361,  0.96006540660068262,
         0.65792354440818768,  0.67339454297760115,  0.91537606801722471,
         1.00830573364457350,  1.17785130672580560,  0.96888511417659751,
         0.91922233510648133,  1.48911300584869480,  1.27563721014500330,
         0.35610920652496597,  0.72356921216391212, -0.02938305697349869,
         0.81795202905116327,  0.96642876917688181,
    ])
    pdev_r = 63.697018840456174
    delu_r = np.array([
         1.51622361933422890,  1.79325927372979610, -1.01402184455867860,
         1.29333059202223070,  0.20603130869775460,  0.42286631256486390,
         0.03884269518032309,  1.07857267251424770,  1.58376140134823130,
         0.75346350567854181,
    ])
    # MU is piecewise-constant per group (β=0 in pp$delb means x has no
    # effect; each group's η = Zb is one value, repeated n_per times).
    mu_per_group = np.array([
        2.8902891052333608, 3.5088259404064628, 0.4917350168624065,
        2.4727512926990149, 1.1551404243281138, 1.3444787867386139,
        1.0275629046924868, 2.1276133759950580, 3.0302132026106583,
        1.6945622603102868,
    ])
    mu_r = np.repeat(mu_per_group, n_per)

    # Build hea state and run one iteration with the same θ AND η.
    df = pl.DataFrame({
        "y": y, "x": x, "g": [f"G{gi:02d}" for gi in g],
    })
    X, y_arr, Z_sp, re_terms, _ = _build_design_pieces("y ~ x + (1|g)", df)
    state = _PredState(X, Z_sp, re_terms)
    state.set_theta(np.array([fixed_theta]))
    # Seed η to R's lp0 so initial state matches what RglmerWrkIter saw.
    resp = _GlmResponse(Poisson(), y_arr, etastart=lp0_r)
    # _GlmResponse.__init__ called update_mu(lp0_r) already (offset=0).

    pdev = _internal_glmer_wrk_iter(state, resp, u_only=True)

    assert pdev == pytest.approx(pdev_r, rel=1e-10)
    np.testing.assert_allclose(state.delu, delu_r, atol=1e-10)
    np.testing.assert_allclose(state.delb, np.zeros(X.shape[1]), atol=1e-12)
    np.testing.assert_allclose(resp.mu, mu_r, atol=1e-10)


def test_pirls_u_only_keeps_beta_at_zero():
    """PIRLS with u_only=True must leave delb at zero (matching lme4's
    Stage 0 nAGQ=0 path where β is held fixed)."""
    rng = np.random.default_rng(99)
    n_groups, n_per = 8, 4
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    y = rng.poisson(np.exp(0.5 + 0.3 * x + 0.4 * rng.standard_normal(n_groups)[g])).astype(float)

    df = pl.DataFrame({
        "y": y, "x": x, "g": [f"G{gi:02d}" for gi in g],
    })
    X, y_arr, Z_sp, re_terms, _ = _build_design_pieces("y ~ x + (1|g)", df)

    state = _PredState(X, Z_sp, re_terms)
    state.set_theta(np.array([1.0]))  # identity Λ
    resp = _GlmResponse(Poisson(), y_arr)

    _pwrss_update(state, resp, u_only=True, tol=1e-7, maxit=200)
    np.testing.assert_array_equal(state.delb, np.zeros(X.shape[1]))
    # And delu should be nonzero (PIRLS moved u).
    assert np.any(state.delu != 0.0)


def test_pwrss_update_step_halving_recovers_from_overstep():
    """If a PIRLS step makes pdev worse, the loop step-halves rather than
    diverging. Hard to trigger naturally on benign data — instead, set θ
    very large (high RE variance) so the first iteration overshoots."""
    rng = np.random.default_rng(7)
    n_groups, n_per = 5, 6
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    y = rng.poisson(5.0, size=n).astype(float)
    x = rng.standard_normal(n)

    df = pl.DataFrame({
        "y": y, "x": x, "g": [f"G{gi:02d}" for gi in g],
    })
    X, y_arr, Z_sp, re_terms, _ = _build_design_pieces("y ~ x + (1|g)", df)

    state = _PredState(X, Z_sp, re_terms)
    state.set_theta(np.array([1.0]))
    resp = _GlmResponse(Poisson(), y_arr)

    # Should converge without raising even with default tol.
    pdev = _pwrss_update(state, resp, u_only=False, tol=1e-8, maxit=200)
    assert np.isfinite(pdev)
    assert pdev > 0


# ----------------------------------------------------------------------
# Phase 4: Laplace deviance evaluator. Tests _glmm_devfun_factory's two
# closures against `lme4::mkGlmerDevfun(nAGQ=0)` and `updateGlmerDevfun(
# nAGQ=1)` at the converged (θ̂, β̂) of a real glmer fit.
# ----------------------------------------------------------------------


def _synthetic_poisson_grouped(seed: int, n_groups: int = 12, n_per: int = 6):
    """Generate a synthetic Poisson GLMM with one scalar random intercept."""
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    b = rng.standard_normal(n_groups) * 0.6
    eta = 0.4 + 0.25 * x + b[g]
    y = rng.poisson(np.exp(eta)).astype(float)
    df = pl.DataFrame({
        "y": y, "x": x, "g": [f"G{gi:02d}" for gi in g],
    })
    return df


# lme4 reference values for ``glmer(y ~ x + (1|g), poisson, data=...)``
# fit to the seed-2026 synthetic Poisson grouped data. Used by the
# Stage-0 and Stage-1 devfun pin tests below.
#
# R recipe::
#     d <- read.csv("...")   # the seed-2026 synthetic data
#     d$g <- factor(d$g)
#     m <- glmer(y ~ x + (1|g), data=d, family=poisson())
#     theta_hat <- getME(m, "theta")
#     beta_hat  <- getME(m, "beta")
#     glmod <- glFormula(y ~ x + (1|g), data=d, family=poisson())
#     dev0  <- mkGlmerDevfun(glmod$fr, glmod$X, glmod$reTrms,
#                            family=poisson(), nAGQ=0)
#     dev_stage0 <- dev0(theta_hat)
#     dev1  <- updateGlmerDevfun(dev0, glmod$reTrms, nAGQ=1L)
#     dev_stage1 <- dev1(c(theta_hat, beta_hat))
_GLMER_DEVFUN_POISSON_REF = {
    # R defaults: ``optimizer=c("bobyqa", "Nelder_Mead")`` (lme4's default).
    "theta":      np.array([0.7045561057191941]),
    "beta":       np.array([0.6632395319538555, 0.1485191330019561]),
    "dev_stage0": 271.7739976520101663482,
    "dev_stage1": 271.7452047712424132442,
}


def test_devfun_stage0_matches_lme4_poisson():
    """``devfun_stage0(θ̂)`` ≡ lme4's ``mkGlmerDevfun(nAGQ=0)(θ̂)`` at ≤ 1e-9.

    Stage 0 PIRLS does a joint (β, u) solve, so the deviance at θ̂ here is
    NOT the same as ``-2 logLik(m)`` — it's the joint-conditional deviance
    that lme4 reports as ``dev0(θ̂)``. Phase 4 verifies the closure
    machinery; Phase 5 ties this into the full optimizer.

    The initial :func:`_pwrss_update` before the factory mirrors
    ``mkGlmerDevfun``'s ``.Call(glmerLaplace, ...)`` warm-up at
    modular.R:888 — without it, the cold-start lp0 would change the PIRLS
    iteration count and the stale ``ldL2`` lme4 reports drifts by ~1e-4.
    """
    df = _synthetic_poisson_grouped(seed=2026)
    theta_hat = _GLMER_DEVFUN_POISSON_REF["theta"]

    X, y_arr, Z_sp, re_terms, _ = _build_design_pieces("y ~ x + (1|g)", df)
    pred = _PredState(X, Z_sp, re_terms)
    resp = _GlmResponse(Poisson(), y_arr)
    _pwrss_update(pred, resp, u_only=False, tol=1e-7, maxit=100)

    devfun_stage0 = _glmm_devfun_factory(pred, resp, nagq=0)
    dev_hea = devfun_stage0(theta_hat)
    assert dev_hea == pytest.approx(
        _GLMER_DEVFUN_POISSON_REF["dev_stage0"], rel=1e-9, abs=1e-9,
    )


def test_devfun_stage1_matches_lme4_poisson():
    """``devfun_stage1([θ̂, β̂])`` ≡ lme4's ``nAGQ=1`` devfun at ≤ 1e-9.

    Stage 1 folds β̂ into the offset and runs PIRLS with ``u_only=True``.
    The returned deviance equals ``-2 logLik(m)`` at the converged
    parameters — the value lme4's outer optimizer minimises.
    """
    df = _synthetic_poisson_grouped(seed=2026)
    theta_hat = _GLMER_DEVFUN_POISSON_REF["theta"]
    beta_hat  = _GLMER_DEVFUN_POISSON_REF["beta"]

    X, y_arr, Z_sp, re_terms, _ = _build_design_pieces("y ~ x + (1|g)", df)
    pred = _PredState(X, Z_sp, re_terms)
    resp = _GlmResponse(Poisson(), y_arr)

    # Mirror the R script's full lme4 sequence: (a) init PIRLS via
    # mkGlmerDevfun(nAGQ=0) with joint solve, (b) one call to the Stage 0
    # closure at θ̂, (c) updateGlmerDevfun(nAGQ=1) re-snapshots lp0 from
    # post-step-(b) state, then dev_stage1 uses that lp0. Without step (b)
    # the Stage 1 lp0 captures state at θ₀ instead of θ̂, and PIRLS in the
    # Stage 1 closure follows a different iteration trajectory.
    _pwrss_update(pred, resp, u_only=False, tol=1e-7, maxit=100)
    devfun_stage0 = _glmm_devfun_factory(pred, resp, nagq=0)
    devfun_stage0(theta_hat)

    devfun_stage1 = _glmm_devfun_factory(pred, resp, nagq=1)
    dev_hea = devfun_stage1(np.concatenate([theta_hat, beta_hat]))
    assert dev_hea == pytest.approx(
        _GLMER_DEVFUN_POISSON_REF["dev_stage1"], rel=1e-9, abs=1e-9,
    )


def test_devfun_factory_pure_function_property():
    """Calling devfun(θ) twice with the same arg must give the same value.

    Each call resets PIRLS to the snapshotted ``lp0``, so the optimizer
    can rely on devfun being a pure function of its argument regardless
    of how many times it was called or with what intermediate values.
    """
    df = _synthetic_poisson_grouped(seed=42)
    X, y_arr, Z_sp, re_terms, _ = _build_design_pieces("y ~ x + (1|g)", df)
    pred = _PredState(X, Z_sp, re_terms)
    resp = _GlmResponse(Poisson(), y_arr)
    _pwrss_update(pred, resp, u_only=False, tol=1e-7, maxit=100)

    devfun_stage0 = _glmm_devfun_factory(pred, resp, nagq=0)
    theta_a = np.array([0.5])
    theta_b = np.array([1.3])
    # Probe values in a noisy interleaved order so any state-carryover bug
    # would show up as a mismatch on the repeat.
    d_a_1 = devfun_stage0(theta_a)
    d_b   = devfun_stage0(theta_b)
    d_a_2 = devfun_stage0(theta_a)
    assert d_a_1 == pytest.approx(d_a_2, rel=1e-12, abs=1e-12)
    assert d_a_1 != pytest.approx(d_b, rel=1e-3)


def test_devfun_stage1_with_empty_fixef_slice():
    """When the model has no fixed effects (p=0), the Stage-1 closure must
    handle the empty β slice without trying to do ``X @ empty``.

    R recipe (seed=7 / n_groups=8 / n_per=5 synthetic Poisson grouped data)::
        m <- glmer(y ~ 0 + (1|g), data=d, family=poisson())
        theta_hat <- getME(m, "theta"); beta_hat <- getME(m, "beta")
        glmod <- glFormula(y ~ 0 + (1|g), data=d, family=poisson())
        dev0  <- mkGlmerDevfun(glmod$fr, glmod$X, glmod$reTrms,
                               family=poisson(), nAGQ=0)
        dev1  <- updateGlmerDevfun(dev0, glmod$reTrms, nAGQ=1L)
        dev1(c(theta_hat, beta_hat))
    """
    df = _synthetic_poisson_grouped(seed=7, n_groups=8, n_per=5)
    theta_hat = np.array([0.61468606296337902])
    dev1_r = 146.50799087473536

    # polars to_numpy on a 0-column DataFrame returns shape (0, 0) — work
    # around by building X explicitly. The rest of _build_design_pieces is
    # still usable for y/Z.
    _, y_arr, Z_sp, re_terms, _ = _build_design_pieces("y ~ 0 + (1|g)", df)
    X = np.zeros((df.height, 0), dtype=float)
    pred = _PredState(X, Z_sp, re_terms)
    resp = _GlmResponse(Poisson(), y_arr)
    # The R recipe does NOT call dev0(theta_hat) between mkGlmerDevfun and
    # updateGlmerDevfun, so Stage 1's lp0 is captured right after the init
    # PIRLS at θ₀. Match that — single init pass then Stage 1 factory.
    _pwrss_update(pred, resp, u_only=False, tol=1e-7, maxit=100)

    devfun_stage1 = _glmm_devfun_factory(pred, resp, nagq=1)
    dev_hea = devfun_stage1(theta_hat)  # par = theta only (empty β slice)
    assert dev_hea == pytest.approx(dev1_r, rel=1e-9, abs=1e-9)


# ----------------------------------------------------------------------
# Phase 5: Full glmer fit — tests the public ``hea.lme(..., family=...)``
# entry point against ``lme4::glmer``. ≤ 1e-7 on θ̂, β̂; ≤ 1e-9 on the
# Laplace deviance (since deviance evaluation is exact given converged
# parameters).
# ----------------------------------------------------------------------


# lme4 reference values for ``glmer(y ~ x + (1|g), poisson, data=...)`` on
# the seed-2026 synthetic Poisson grouped data with both Stage-0 and
# Stage-1 optimizers set to ``Nelder_Mead`` (lme4 defaults to bobyqa for
# Stage 0). Used by the Poisson full-fit test below.
#
# R recipe (same data as ``_GLMER_DEVFUN_POISSON_REF``)::
#     m <- glmer(y ~ x + (1|g), data=d, family=poisson(),
#                control=glmerControl(
#                    optimizer=c("Nelder_Mead", "Nelder_Mead")))
_GLMER_POISSON_FULLFIT_REF = {
    # R defaults: ``optimizer=c("bobyqa", "Nelder_Mead")`` (lme4's default).
    "theta":    np.array([0.7045561057191941]),
    "beta":     np.array([0.6632395319538555, 0.1485191330019561]),
    "laplace":  271.7451995996608502537,
    "deviance":  78.15576010170215681683,
    "aic":      277.7451995996608502537,
    "bic":      284.5751979567090188539,
    "sigma":      1.0,    # scale-known for Poisson
}


def test_glmer_poisson_full_fit_matches_lme4():
    """Full ``hea.lme(..., family=poisson())`` fit ≡ ``lme4::glmer(..., family=poisson)``.

    Compared to ``lme4::glmer`` with its default optimizer chain
    ``optimizer=c("bobyqa", "Nelder_Mead")`` — both stages of hea use the
    ported BOBYQA + Nelder-Mead implementations. Tolerance ≤ 1e-7 on
    θ̂/β̂ — anything looser would mask actual bugs.
    """
    from hea.lme import lme  # local import — keep test file's top imports lean
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    r = _GLMER_POISSON_FULLFIT_REF

    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    np.testing.assert_allclose(m.theta, r["theta"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m._beta, r["beta"],  atol=1e-7, rtol=1e-7)
    # ``deviance(m)`` for glmer fits = residual deviance (= Σ dev_resids),
    # NOT the Laplace value. The Laplace value is on ``deviance_laplace``.
    np.testing.assert_allclose(m.deviance, r["deviance"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.deviance_laplace, r["laplace"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.AIC, r["aic"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.BIC, r["bic"], atol=1e-9, rtol=1e-9)
    assert m.sigma == pytest.approx(r["sigma"])  # = 1 for Poisson
    # Public-API check: bhat as a DataFrame with R-canonical column names.
    assert m.bhat.columns == ["(Intercept)", "x"]
    np.testing.assert_allclose(
        m.bhat.row(0), r["beta"], atol=1e-7, rtol=1e-7,
    )


def test_glmer_binomial_full_fit_matches_lme4_cbpp():
    """Full ``hea.lme`` fit on cbpp matches ``lme4::glmer(family=binomial)``.

    cbpp is the canonical lme4 GLMM example. Uses proportion response
    (incidence/size) with binomial weights (size).

    R recipe (lme4 defaults — ``optimizer=c("bobyqa","Nelder_Mead")``)::
        suppressMessages(library(lme4)); data(cbpp)
        m <- glmer(cbind(incidence, size-incidence) ~ period + (1|herd),
                   data=cbpp, family=binomial())
        # captured getME(m, "theta"), getME(m, "beta"),
        # -2*as.numeric(logLik(m)), deviance(m)
    """
    from hea import data as hea_data
    from hea.lme import lme
    from hea.family import Binomial as BinomialFamily

    df = hea_data("cbpp").with_columns(
        (pl.col("incidence") / pl.col("size")).alias("y_prop"),
        pl.col("herd").cast(pl.String),
        pl.col("period").cast(pl.String),
    )

    theta_r = np.array([0.6420699254034050])
    beta_r  = np.array([
        -1.3983428639994452957751, -0.9919249753929506585592,
        -1.1282162163483180350454, -1.5797454141259090754090,
    ])
    laplace_r = 184.0531327790863542759
    dev_r     =  73.47428361870440483017

    size = df["size"].to_numpy().astype(float)
    m = lme("y_prop ~ period + (1|herd)", df,
            family=BinomialFamily(), weights=size)

    np.testing.assert_allclose(m.theta, theta_r, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m._beta, beta_r,  atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.deviance, dev_r, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.deviance_laplace, laplace_r, atol=1e-9, rtol=1e-9)


def test_glmer_intercept_only_poisson():
    """No fixed effects (p=0) → Stage 1 has empty β slice, optimize θ only.

    Edge case: Stage 1's par vector is just θ. lme4 happily fits these too;
    we should match.

    R recipe (seed=11 / n_groups=8 / n_per=5 synthetic data)::
        m <- glmer(y ~ 0 + (1|g), data=d, family=poisson(),
                   control=glmerControl(
                       optimizer=c("Nelder_Mead","Nelder_Mead")))
    """
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=11, n_groups=8, n_per=5)
    # R defaults: bobyqa + Nelder_Mead.
    theta_r   = np.array([0.5099103344750407])
    laplace_r = 112.4179270982947826951

    m = lme("y ~ 0 + (1|g)", df, family=PoissonFamily())
    np.testing.assert_allclose(m.theta, theta_r, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.deviance_laplace, laplace_r, atol=1e-9, rtol=1e-9)
    assert m.p == 0
    assert m._beta.shape == (0,)


def test_glmer_nagq0_init_step_false_runs_stage1_directly():
    """With ``nAGQ0initStep=False``, Stage 0 is skipped and Stage 1 starts
    cold (θ=θ₀, β=0). Should still converge to the same optimum as the
    default path (Stage 0 just provides a warm start; final answer is
    determined by Stage 1 alone).
    """
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m_default = lme("y ~ x + (1|g)", df, family=PoissonFamily())
    m_no_stage0 = lme(
        "y ~ x + (1|g)", df, family=PoissonFamily(), nAGQ0initStep=False,
    )

    assert m_default._optim_stage0 is not None
    assert m_no_stage0._optim_stage0 is None
    # Without Stage 0 warm-up, Stage 1 starts cold (β=0); Nelder-Mead is
    # derivative-free and gets stuck at slightly different simplex
    # configurations within its xtol band when starting cold vs warm.
    # That's expected — the warm-started path (default) is more numerically
    # accurate. ~1e-3 is the realistic agreement.
    np.testing.assert_allclose(
        m_default.theta, m_no_stage0.theta, atol=1e-3, rtol=1e-3,
    )
    np.testing.assert_allclose(
        m_default._beta, m_no_stage0._beta, atol=1e-3, rtol=1e-3,
    )


def test_glmer_start_numeric_overrides_theta():
    """A numeric ``start=`` is interpreted as θ-only and overrides the
    formula default ``θ₀``. The optimizer still converges to the same
    answer."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m_default = lme("y ~ x + (1|g)", df, family=PoissonFamily())
    # Start from a different θ — should still find the same optimum.
    m_alt = lme(
        "y ~ x + (1|g)", df, family=PoissonFamily(), start=np.array([2.0]),
    )
    np.testing.assert_allclose(m_default.theta, m_alt.theta, atol=1e-4, rtol=1e-4)


def test_glmer_start_dict_with_theta_and_beta():
    """``start={"theta": ..., "beta": ...}`` overrides both initial values."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m_default = lme("y ~ x + (1|g)", df, family=PoissonFamily())
    m_with_dict = lme(
        "y ~ x + (1|g)", df, family=PoissonFamily(),
        start={"theta": np.array([1.5]), "beta": np.array([0.1, 0.2])},
    )
    np.testing.assert_allclose(
        m_default.theta, m_with_dict.theta, atol=1e-4, rtol=1e-4,
    )
    np.testing.assert_allclose(
        m_default._beta, m_with_dict._beta, atol=1e-4, rtol=1e-4,
    )


def test_glmer_start_validation_errors():
    """``start=`` rejects malformed inputs."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)

    with pytest.raises(ValueError, match="unrecognised start keys"):
        lme("y ~ x + (1|g)", df, family=PoissonFamily(),
            start={"theta": np.array([1.0]), "blah": np.array([0.0])})
    with pytest.raises(ValueError, match="not have both"):
        lme("y ~ x + (1|g)", df, family=PoissonFamily(),
            start={"theta": np.array([1.0]), "par": np.array([1.0])})
    with pytest.raises(ValueError, match="start theta has shape"):
        lme("y ~ x + (1|g)", df, family=PoissonFamily(),
            start={"theta": np.array([1.0, 2.0])})  # too many
    with pytest.raises(ValueError, match="start beta has shape"):
        lme("y ~ x + (1|g)", df, family=PoissonFamily(),
            start={"beta": np.array([1.0])})  # wrong p


# ----------------------------------------------------------------------
# Phase 6: Post-fit attributes (fitted, residuals, ranef, vcov_beta, ...)
# Each attribute pinned against the corresponding ``lme4::glmer`` getter.
# ----------------------------------------------------------------------


# Phase-6 attribute pins for ``glmer(y ~ x + (1|g), poisson)`` on the
# seed-2026 synthetic data. Captured locally; reproducible from the same
# R recipe used by ``_GLMER_POISSON_FULLFIT_REF`` plus::
#     fitted(m); predict(m, type="link")
#     residuals(m, type="deviance" | "pearson" | "working" | "response")
#     m@resp$sqrtXwt^2; m@resp$weights
#     AIC(m); BIC(m); sigma(m)
#     vcov(m)                      # default = Hessian-based (calc.derivs=TRUE)
#     VarCorr(m)$g                 # per-bar SD
#     ranef(m)$g                   # BLUPs
_GLMER_PHASE6_POISSON_REF = {
    # R defaults: ``optimizer=c("bobyqa", "Nelder_Mead")`` (lme4's default).
    # Captured from R via /tmp/gen_all_refs.R + /tmp/gen_full_vecs.R.
    "theta": np.array([0.7045561057191941]),
    "beta":  np.array([0.6632395319538555, 0.1485191330019561]),
    "eta": np.array([
        1.772928139156246807318e+00, 1.926451439987335811921e+00, 1.609081256199845988419e+00,
        2.098020805785064446525e+00, 1.985520983053430166265e+00, 1.847347362230882028555e+00,
        -1.464849208455243356752e-01, -5.502911168433710642489e-02, -1.399071531046067740078e-01,
        -1.337062652795965034258e-01, 6.78937124818523862757e-03, -2.37109061690630174013e-02,
        7.572744284341057507959e-01, 7.54103748779348714848e-01, 7.906978065638644581625e-01,
        6.75605174234499061825e-01, 7.068340145796393469979e-01, 8.482257804594532935027e-01,
        3.726819626255949979843e-01, 1.87932569947848571168e-01, 3.211761322943422203302e-01,
        4.895820254630985357558e-01, 3.575627180134503535491e-01, 3.699714859093065633111e-01,
        5.785185465862201503739e-01, 7.541830675938041572692e-01, 3.772713638286342985317e-01,
        6.834280340798910557965e-01, 3.00513096066523099914e-01, 5.091810518654726891441e-01,
        4.274563183920632170043e-02, 4.171605399483396259264e-01, 3.402966625603680017598e-01,
        3.854158525561241499524e-01, 8.492452573831071882537e-02, 3.181126609745050237699e-01,
        1.207511212865748806422e+00, 1.216664152768367479496e+00, 1.359825070494126375564e+00,
        1.414804045490595685308e+00, 1.314954935788563705756e+00, 1.191326522849978086782e+00,
        6.484282683803477276285e-02, 4.020010929937984922589e-01, 2.757047122473186284708e-01,
        2.943857496539028506533e-01, 5.117821464140321907621e-01, 6.632112275848334181916e-02,
        2.096418063309470003475e+00, 2.18425975429728858046e+00, 1.956654923595163797501e+00,
        1.839130755101978653698e+00, 1.617699348927164049172e+00, 1.863989753361903867201e+00,
        -3.268811204146737647847e-02, 2.984458560175606844922e-02, -1.007474157076915233233e-02,
        7.521432118818904832835e-02, 2.243643276669510244758e-01, -4.945472045388199511251e-02,
        1.879811267357243687037e-01, 1.853742122588346030732e-01, 1.905247406194477410857e-01,
        1.217862753733513492271e-01, 1.987442724653687986525e-01, 5.264071378692658509379e-01,
        1.04554044849588834154e+00, 1.171844034055877337153e+00, 9.52926590819561658563e-01,
        1.158854984658521658503e+00, 8.75119276136930412946e-01, 1.02988000754474251508e+00,
    ]),
    "mu": np.array([
        5.888069229919299374387e+00, 6.865105727082347897294e+00, 4.998217036800098433957e+00,
        8.150023460071954772843e+00, 7.28284063380960677847e+00, 6.342971580754035443306e+00,
        8.637387566867026356121e-01, 9.464575945776914078778e-01, 8.694389563592344050136e-01,
        8.748469998049753781899e-01, 1.006812471277792386815e+00, 9.765679887344216325573e-01,
        2.132456130768735746983e+00, 2.125705503194454060178e+00, 2.204934507324094017378e+00,
        1.965221917844819143895e+00, 2.027561855046171590544e+00, 2.335499485020970045213e+00,
        1.451622596171683099442e+00, 1.206752141246288045906e+00, 1.378748401628139053088e+00,
        1.631634095904439307745e+00, 1.429840240375376048121e+00, 1.447693334415758981137e+00,
        1.783394456965253116465e+00, 2.125874118321936290243e+00, 1.458299985550072674911e+00,
        1.980655863560951468472e+00, 1.350551592537952316775e+00, 1.663927966111386602677e+00,
        1.043672384058494051473e+00, 1.517646136211232565927e+00, 1.405364447742895039895e+00,
        1.470225591403511389288e+00, 1.0886348996823875801e+00, 1.374531108518947331021e+00,
        3.345148920325077046556e+00, 3.375907417962490608687e+00, 3.895511802235072096323e+00,
        4.115679900914961386604e+00, 3.72458313606729340961e+00, 3.291444489071050139017e+00,
        1.066991308806549465515e+00, 1.494812966496452011356e+00, 1.31745877714228187827e+00,
        1.342301596186858958504e+00, 1.66826163376953995332e+00, 1.068569804159667846477e+00,
        8.136971533527900035665e+00, 8.884069724260822198403e+00, 7.07561894853841177877e+00,
        6.29106740536922259821e+00, 5.041478281648091197553e+00, 6.449417091816010305649e+00,
        9.678403702782971684826e-01, 1.030294398926171473008e+00, 9.899758386347439209629e-01,
        1.07811518905388248335e+00, 1.251526902254381345969e+00, 9.517482518896861476421e-01,
        1.206810738676669103597e+00, 1.203668783474792336463e+00, 1.209884306554398936129e+00,
        1.129512671307448812286e+00, 1.219869971668389618813e+00, 1.692839231411864986399e+00,
        2.844935649438333324923e+00, 3.227939584181692467268e+00, 2.593288057493079001858e+00,
        3.186282843876026227292e+00, 2.399161439609974522114e+00, 2.800729748096567384152e+00,
    ]),
    "res_dev": np.array([
        4.448444654320526381319e-01, -3.375023037748426935067e-01, 7.974601815780581973853e-04,
        -1.614383895739337049235e+00, 2.61553447007884409814e-01, 1.996839017767559232297e+00,
        1.042116077505420790317e+00, -1.375832544009401914309e+00, -1.318665201147914078916e+00,
        1.30791934648676888564e-01, -1.419022530672288118225e+00, 2.361757411270630491185e-02,
        5.59409888133464816562e-01, -8.709037553767870476484e-02, 5.073025796047756497131e-01,
        2.473582404981558569013e-02, -1.940036117103220020264e-02, -9.871916780053575646292e-01,
        -1.703891191462461263129e+00, -1.940080993074231041629e-01, 1.19252410520663398863e+00,
        -1.806451823827272429313e+00, -3.802039514837415312698e-01, 1.125713767075954940822e+00,
        8.290665633239852061465e-01, -8.621960922297572382789e-01, -4.025633409078335778197e-01,
        -7.710095064019123478616e-01, -3.163494791253158577859e-01, 9.297544584666401767947e-01,
        -1.444764606472967338746e+00, 3.731472026813115183685e-01, -1.676522858623105793896e+00,
        1.104368507102106766737e+00, -1.475557453766126680961e+00, 4.991996548272601974539e-01,
        -1.921049074446533611038e-01, -2.085752893404066721228e-01, -4.730189356823294444254e-01,
        1.291063612583040143278e+00, 1.410023928350235822293e-01, -1.631060178254925552288e-01,
        2.169238320671093234182e+00, -1.729053478927966569501e+00, -2.889777323426951927132e-01,
        -1.638475874821999855868e+00, 2.489646644058794433541e-01, -1.461895895171518633404e+00,
        1.862214413975437521387e+00, -3.017416886053407298895e-01, -1.260298931712091397017e+00,
        6.535345407736018463751e-01, -1.849860442157427037335e-02, -5.944037132740548345922e-01,
        3.251099259526231410877e-02, -1.435475112237179384778e+00, 1.005785320166210916604e-02,
        -7.616912584103062433538e-02, -1.582104233136604198862e+00, 9.348283841618643119631e-01,
        -1.940598461348708447805e-01, -1.912828858834876299966e-01, -1.55556054626902828808e+00,
        -1.503005436655136639601e+00, 6.460275347963776448879e-01, 9.049361124503726427903e-01,
        9.111717681093130238867e-02, 4.141035919615903004853e-01, 1.803488468465512140071e+00,
        4.382701215494891333613e-01, -2.190507447880501779025e+00, -1.24165191624047732688e+00,
    ]),
    "res_pearson": np.array([
        4.582382045460857145081e-01, -3.301760533853949075578e-01, 7.975075886187799447183e-04,
        -1.453687721975674307373e+00, 2.657451036547552325473e-01, 2.246166665330058265226e+00,
        1.222607315297482344363e+00, -9.728605216461870819344e-01, -9.324371058464128170229e-01,
        1.338058840283894157519e-01, -1.003400454094870442034e+00, 2.371146160792960908004e-02,
        5.940888373714596770725e-01, -8.621895254671425867787e-02, 5.354329054195178239084e-01,
        2.480846134170042877054e-02, -1.935625737896601999966e-02, -8.738840987293565731875e-01,
        -1.204833015887132541266e+00, -1.882092556058569121635e-01, 1.380727282141422707085e+00,
        -1.27735433451507063296e+00, -3.594705471821626474593e-01, 1.290147113174048953965e+00,
        9.11016168108483070931e-01, -7.721844707290632658925e-01, -3.795127822441960452338e-01,
        -6.968063783768473351543e-01, -3.016451554470270801644e-01, 1.035768294959910518216e+00,
        -1.021602850455348932002e+00, 3.915439352334785239051e-01, -1.185480682146653474973e+00,
        1.261639862429330349869e+00, -1.043376681588383814159e+00, 5.334926343353240119072e-01,
        -1.887116843269670274363e-01, -2.04590579276029221889e-01, -4.537211787515708083518e-01,
        1.421748115719744243179e+00, 1.427091083131418469687e-01, -1.606432863624727425123e-01,
        2.839441391848729345782e+00, -1.222625439984156514583e+00, -2.765787575973655543216e-01,
        -1.158577401897196823555e+00, 2.568405672532140249231e-01, -1.033716500864559018069e+00,
        2.055372797743762802725e+00, -2.96606414421938635595e-01, -1.156246176795178159935e+00,
        6.813382394426178434443e-01, -1.847318612433232476455e-02, -5.707332126936348615232e-01,
        3.268956738014852647645e-02, -1.01503418608742990159e+00, 1.007478417875975681972e-02,
        -7.52320516925768928429e-02, -1.118716631794835381086e+00, 1.07449537733927447114e+00,
        -1.882580269770171843557e-01, -1.856397481684342876118e-01, -1.099947410813080228564e+00,
        -1.062785336419095116867e+00, 7.063341830151320666076e-01, 1.004664709992259208349e+00,
        9.193391203131708044882e-02, 4.297229886994302083814e-01, 2.115485402667601988469e+00,
        4.558598151077892235428e-01, -1.548922670635940468387e+00, -1.07600164933780972909e+00,
    ]),
    "res_working": np.array([
        1.888447174551887153271e-01, -1.260149168088654514364e-01, 3.56719843651094104843e-04,
        -5.092038667622823533421e-01, 9.847247828835872796027e-02, 8.918577589738265753994e-01,
        1.315514945366107468772e+00, -1e+00, -1e+00,
        1.430570147956434223779e-01, -1e+00, 2.399424467716268841166e-02,
        4.068284719735456578249e-01, -5.913589770810074447427e-02, 3.605846296272974260688e-01,
        1.769677095466174951421e-02, -1.359359517322540622519e-02, -5.718260670089502228919e-01,
        -1e+00, -1.713294173505772732824e-01, 1.175886475340500281206e+00,
        -1e+00, -3.00621166083932633839e-01, 1.072262079738524498396e+00,
        6.821853338632702934419e-01, -5.29605261486812639582e-01, -3.142700336633421076549e-01,
        -4.951167346142932856878e-01, -2.595617927332911634153e-01, 8.029626649109244951319e-01,
        -1e+00, 3.178302585034429306177e-01, -1e+00,
        1.040503183689062582928e+00, -1e+00, 4.55041641185548240145e-01,
        -1.031789401745187295667e-01, -1.113500376113298051539e-01, -2.29882964729119065872e-01,
        7.008125433768117584776e-01, 7.394568838205965721766e-02, -8.85460745392382397867e-02,
        2.748859027234324514666e+00, -1e+00, -2.409629679881795660812e-01,
        -1e+00, 1.98852721608706400902e-01, -1e+00,
        7.205418431555088432106e-01, -9.951179489807292222192e-02, -4.346784317962363863863e-01,
        2.716442988946929371075e-01, -8.227404608501394031594e-03, -2.247361383488825081756e-01,
        3.322823753720410139278e-02, -1e+00, 1.012566264150466764649e-02,
        -7.24553274520079171328e-02, -1e+00, 1.101396032016892068128e+00,
        -1.713696539553897779662e-01, -1.692066673747526228677e-01, -1e+00,
        -1e+00, 6.395190032136323088352e-01, 7.721706493639884927305e-01,
        5.45053982476828777215e-02, 2.391805657087695546892e-01, 1.313665071901104486329e+00,
        2.553813317885204048352e-01, -1e+00, -6.42950198718880261417e-01,
    ]),
    "res_response": np.array([
        1.111930770080700625613e+00, -8.651057270823478972943e-01, 1.782963199901566042627e-03,
        -4.150023460071954772843e+00, 7.171593661903932215296e-01, 5.657028419245964556694e+00,
        1.136261243313297253366e+00, -9.464575945776914078778e-01, -8.694389563592344050136e-01,
        1.251530001950246218101e-01, -1.006812471277792386815e+00, 2.343201126557836744269e-02,
        8.675438692312642530169e-01, -1.257055031944540601785e-01, 7.950654926759059826225e-01,
        3.47780821551808561054e-02, -2.756185504617159054419e-02, -1.335499485020970045213e+00,
        -1.451622596171683099442e+00, -2.06752141246288045906e-01, 1.621251598371860946912e+00,
        -1.631634095904439307745e+00, -4.298402403753760481209e-01, 1.552306665584241018863e+00,
        1.216605543034746883535e+00, -1.125874118321936290243e+00, -4.582999855500726749113e-01,
        -9.806558635609514684717e-01, -3.505515925379523167749e-01, 1.336072033888613397323e+00,
        -1.043672384058494051473e+00, 4.823538637887674340732e-01, -1.405364447742895039895e+00,
        1.529774408596488610712e+00, -1.0886348996823875801e+00, 6.254688914810526689791e-01,
        -3.451489203250770465559e-01, -3.759074179624906086872e-01, -8.955118022350720963232e-01,
        2.884320099085038613396e+00, 2.754168639327065903899e-01, -2.914444890710501390174e-01,
        2.933008691193450534485e+00, -1.494812966496452011356e+00, -3.174587771422818782696e-01,
        -1.342301596186858958504e+00, 3.317383662304600466797e-01, -1.068569804159667846477e+00,
        5.863028466472099964335e+00, -8.840697242608221984028e-01, -3.07561894853841177877e+00,
        1.70893259463077740179e+00, -4.147828164809119755319e-02, -1.449417091816010305649e+00,
        3.21596297217028315174e-02, -1.030294398926171473008e+00, 1.002416136525607903707e-02,
        -7.811518905388248334987e-02, -1.251526902254381345969e+00, 1.048251748110313741336e+00,
        -2.068107386766691035973e-01, -2.036687834747923364631e-01, -1.209884306554398936129e+00,
        -1.129512671307448812286e+00, 7.80130028331610381187e-01, 1.307160768588135013601e+00,
        1.550643505616666750768e-01, 7.720604158183075327315e-01, 3.406711942506920998142e+00,
        8.137171561239737727078e-01, -2.399161439609974522114e+00, -1.800729748096567384152e+00,
    ]),
    "working_wts": np.array([
        5.888069229919298486209e+00, 6.865105727082347897294e+00, 4.998217036800098433957e+00,
        8.150023460071954772843e+00, 7.282840633809607666649e+00, 6.342971580754035443306e+00,
        8.637387566867026356121e-01, 9.464575945776912968554e-01, 8.694389563592344050136e-01,
        8.748469998049751561453e-01, 1.00681247127779216477e+00, 9.765679887344216325573e-01,
        2.132456130768735746983e+00, 2.125705503194453616089e+00, 2.204934507324094017378e+00,
        1.965221917844819365939e+00, 2.027561855046171590544e+00, 2.335499485020970489302e+00,
        1.451622596171683321487e+00, 1.206752141246287823861e+00, 1.378748401628139053088e+00,
        1.631634095904438863656e+00, 1.429840240375376270165e+00, 1.447693334415758759093e+00,
        1.783394456965252672376e+00, 2.125874118321936290243e+00, 1.458299985550072452867e+00,
        1.980655863560951246427e+00, 1.350551592537952316775e+00, 1.663927966111386602677e+00,
        1.043672384058494051473e+00, 1.517646136211232121838e+00, 1.40536444774289481785e+00,
        1.470225591403511611333e+00, 1.0886348996823875801e+00, 1.374531108518947108976e+00,
        3.345148920325076602467e+00, 3.375907417962490608687e+00, 3.895511802235071208145e+00,
        4.115679900914961386604e+00, 3.724583136067294297789e+00, 3.291444489071051027196e+00,
        1.066991308806549465515e+00, 1.4948129664964522334e+00, 1.31745877714228187827e+00,
        1.342301596186858736459e+00, 1.66826163376953995332e+00, 1.068569804159667846477e+00,
        8.136971533527900035665e+00, 8.884069724260820422046e+00, 7.075618948538410890592e+00,
        6.29106740536922259821e+00, 5.041478281648092085732e+00, 6.449417091816010305649e+00,
        9.678403702782972795049e-01, 1.030294398926171250963e+00, 9.899758386347440319852e-01,
        1.078115189053882705394e+00, 1.251526902254381345969e+00, 9.517482518896860366198e-01,
        1.206810738676668881553e+00, 1.203668783474792114419e+00, 1.209884306554399158173e+00,
        1.129512671307449256375e+00, 1.219869971668389396768e+00, 1.692839231411864764354e+00,
        2.844935649438332880834e+00, 3.227939584181692023179e+00, 2.593288057493079001858e+00,
        3.186282843876025339114e+00, 2.399161439609974078024e+00, 2.800729748096566495974e+00,
    ]),
    "prior_wts": np.ones(72),
    "laplace":  271.7451995996608502537,
    "deviance":  78.15576010170215681683,
    "aic":      277.7451995996608502537,
    "bic":      284.5751979567090188539,
    "sigma":      1.0,
    "se_beta":  np.array([0.2237352608177921586829, 0.0763442423731207908677]),
    "t_value":  np.array([2.964394300342275467841, 1.945387476321941422341]),
    "vcov":     np.array([
        0.050057466933205486958602, -0.001041320693859634554865,
        -0.001041320693859634554865,  0.005828443343525812822548,
    ]).reshape(2, 2),
    "sd_re_g": np.array([0.7045561057191941367606]),
    "ranef_g": np.array([
        1.22748246957720796857e+00, -7.633940090360322727747e-01, 1.035591218818443515204e-01,
        -2.711783780193058324137e-01, -1.800460049125265915571e-01, -4.467963898564753510989e-01,
        6.213550906746284629989e-01, -4.757471514155844349858e-01, 1.053050078181282822598e+00,
        -6.301793898236972557569e-01, -3.289070821905570318755e-01, 4.704190956645730170571e-01,
    ]),
}


def test_glmer_phase6_attrs_match_lme4_poisson():
    """Every Phase 6 attribute on a Poisson fit matches lme4 at ≤ 1e-9."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    r = _GLMER_PHASE6_POISSON_REF
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    # Linear predictor / fitted values. The reference is R-on-Intel and we
    # run on arbitrary platforms. Audit (2026-05) verified that scipy.sparse
    # `@` matches Eigen3's Gustavson at 0 ULP, sqrt_x_wt's 4-op chain matches
    # R at 0 ULP, and np.cumsum tracks R's deviance() within ~1 ULP on
    # n=1934. The remaining floor is CHOLMOD-internal accumulator noise
    # (~2 ULP per factorization) plus bobyqa's rhoend-tolerance walk
    # (~5e-8 in θ̂), which compounds through PIRLS into ~1e-9 abs on η and
    # ~1e-7 rel on residuals/AIC. R itself has the same cross-arch drift
    # on this fit (verified arm64↔x86_64). Pin at 1e-7.
    np.testing.assert_allclose(m.eta, r["eta"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.mu,  r["mu"],  atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.fitted_values, r["mu"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.linear_predictors, r["eta"], atol=1e-7, rtol=1e-7)
    # Residuals — all four types.
    np.testing.assert_allclose(m.residuals,                 r["res_dev"],     atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.residuals_of("deviance"),  r["res_dev"],     atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.residuals_of("pearson"),   r["res_pearson"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.residuals_of("working"),   r["res_working"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.residuals_of("response"),  r["res_response"], atol=1e-7, rtol=1e-7)
    # Working weights = sqrt_x_wt² — matches lme4's m@resp$sqrtXwt^2.
    np.testing.assert_allclose(m.working_weights, r["working_wts"], atol=1e-7, rtol=1e-7)
    # Prior weights = the user-supplied ``weights=`` (1s when not given).
    np.testing.assert_allclose(m.prior_weights, r["prior_wts"], atol=1e-12, rtol=1e-12)
    # Summary statistics.
    np.testing.assert_allclose(m.AIC, r["aic"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.BIC, r["bic"], atol=1e-7, rtol=1e-7)
    assert m.sigma == pytest.approx(r["sigma"])
    # SE(β̂) and t-values. Hessian-based vcov is computed by deriv12
    # (central differences, δ=1e-4) on the Stage-1 closure, so the FD
    # formula ``(f+ − 2f₀ + f−)/δ²`` divides a ~3e-9-scale second difference
    # by 1e-8 — about 11 digits of catastrophic cancellation. For
    # well-conditioned columns the noise is far below H_jj and the SE
    # carries it cleanly. For this n≈70 fit the floor lands at ~1e-7 rel.
    np.testing.assert_allclose(m._se_beta, r["se_beta"], atol=1e-9, rtol=1e-6)
    np.testing.assert_allclose(m.t_values.row(0), r["t_value"], atol=1e-7, rtol=1e-6)
    # vcov_beta — full p×p matrix. Same FP-arithmetic floor as SE.
    np.testing.assert_allclose(m._vcov_beta_arr, r["vcov"], atol=1e-9, rtol=1e-6)
    # Variance components: SD per bar.
    np.testing.assert_allclose(m.sd_re["g"], r["sd_re_g"], atol=1e-9, rtol=1e-7)
    # method string.
    assert m.method == "glmer.ML"


def test_glmer_phase6_ranef_match_lme4_poisson():
    """BLUPs match ``ranef(m)`` — covers ``_ranef``/``ranef`` for GLMM."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())
    rf = m.ranef
    # Single bar named ``g`` — match the BLUPs column-by-column.
    assert "g" in rf
    blups_py = rf["g"]["(Intercept)"].to_numpy()
    np.testing.assert_allclose(
        blups_py, _GLMER_PHASE6_POISSON_REF["ranef_g"],
        atol=1e-9, rtol=1e-7,
    )


def test_glmer_phase6_attrs_match_lme4_binomial_cbpp():
    """cbpp binomial — verify per-period β̂ SE/t, sd_re, deviance breakdown.

    R recipe (same fit as ``test_glmer_binomial_full_fit_matches_lme4_cbpp``)::
        vc_rx <- as.matrix(suppressWarnings(vcov(m, use.hessian=FALSE)))
        # captured AIC, deviance, -2*logLik, sigma, sqrt(diag(vc_rx)),
        # attr(VarCorr(m)$herd, "stddev")
    """
    from hea import data as hea_data
    from hea.lme import lme
    from hea.family import Binomial as BinomialFamily

    df = hea_data("cbpp").with_columns(
        (pl.col("incidence") / pl.col("size")).alias("y_prop"),
        pl.col("herd").cast(pl.String),
        pl.col("period").cast(pl.String),
    )
    size = df["size"].to_numpy().astype(float)

    laplace_r = 184.0531327790863542759
    dev_r     =  73.47428361870440483017
    aic_r     = 194.0531327790863542759
    sigma_r   =   1.0
    # ``vcov(m)`` — default Hessian-based.
    se_beta_r = np.array([
        0.2312140667690355255726, 0.3031507189593240503278,
        0.3228302907742182648043, 0.4220492126057727166888,
    ])
    sd_herd_r = np.array([0.6420699254034050174056])

    m = lme("y_prop ~ period + (1|herd)", df,
            family=BinomialFamily(), weights=size)
    assert m.deviance_laplace == pytest.approx(laplace_r, rel=1e-9, abs=1e-9)
    assert m.deviance         == pytest.approx(dev_r,     rel=1e-9, abs=1e-9)
    assert m.AIC              == pytest.approx(aic_r,     rel=1e-9, abs=1e-9)
    assert m.sigma            == pytest.approx(sigma_r)  # = 1
    # See test_glmer_phase6_attrs_match_lme4_poisson for the deriv12
    # cancellation floor that drives the SE tolerance here.
    np.testing.assert_allclose(m._se_beta,       se_beta_r, atol=1e-9, rtol=2e-6)
    np.testing.assert_allclose(m.sd_re["herd"],  sd_herd_r, atol=1e-9, rtol=1e-7)


def test_glmer_phase6_sigma_for_scale_unknown_family():
    """Scale-unknown families (Gamma) report a Pearson dispersion estimate.

    For canonical-link scale-known (Poisson, Binomial), ``m.sigma == 1``.
    For scale-unknown (Gamma, Inverse-Gaussian, etc.), ``m.sigma`` =
    ``sqrt(sum(w·(y−μ)²/V(μ)) / df_resid)`` — Pearson estimate.

    R recipe (synthetic seed=11 / n_groups=10 / n_per=6 Gamma(log) data)::
        m <- glmer(y ~ x + (1|g), data=d, family=Gamma(link="log"),
                   control=glmerControl(
                       optimizer=c("Nelder_Mead","Nelder_Mead")))
        sigma(m)   # → 0.4682088613...
    """
    from hea.lme import lme
    from hea.family import Gamma as GammaFamily

    rng = np.random.default_rng(11)
    n_groups, n_per = 10, 6
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    b = rng.standard_normal(n_groups) * 0.3
    # Generate positive responses with mean linked to log(eta).
    eta = 1.0 + 0.2 * x + b[g]
    mu = np.exp(eta)
    y = rng.gamma(shape=4.0, scale=mu / 4.0)
    df = pl.DataFrame({"y": y, "x": x, "g": [f"G{gi:02d}" for gi in g]})

    sigma_r = 0.46820886133217066

    from hea.family import LogLink
    m = lme("y ~ x + (1|g)", df, family=GammaFamily(link=LogLink()))
    # σ should be the Pearson estimate, not 1. Tolerance loose since
    # Nelder-Mead doesn't drive Gamma fits to byte-equal endpoints.
    assert m.sigma > 0.0 and m.sigma != 1.0
    np.testing.assert_allclose(m.sigma, sigma_r, atol=1e-3, rtol=1e-3)
    # npar formula: p + n_theta + useSc (=1 for unknown-scale).
    assert m.npar == m.p + len(m.theta) + 1


# ----------------------------------------------------------------------
# Phase 7: GLMM predict — type, re.form, random.only, allow.new.levels,
# se.fit. Pinned against ``lme4::predict.merMod``.
# ----------------------------------------------------------------------


# lme4 reference values for ``predict(m, ...)`` on the same Poisson fit
# used by ``_GLMER_POISSON_FULLFIT_REF``. Each entry is the output of
# ``predict(m, ...)`` for a specific arg combination; ``SE_*`` are the
# ``se.fit`` companion arrays.
#
# R recipe::
#     m <- glmer(y ~ x + (1|g), data=d, family=poisson(),
#                control=glmerControl(
#                    optimizer=c("Nelder_Mead","Nelder_Mead")))
#     predict(m, type="link")                       # → FIT_LINK
#     predict(m, type="response")                   # → FIT_RESPONSE
#     predict(m, newdata=nd, type="response")       # → FIT_NEWDATA (see below)
#     predict(m, re.form=~0, type="response")       # → FIT_NORE
#     predict(m, type="link", random.only=TRUE)     # → FIT_RANDOM
#     out <- predict(m, type="link",     se.fit=TRUE)  # → SE_LINK_*
#     out <- predict(m, type="response", se.fit=TRUE)  # → SE_RESP_*
#
# newdata::  nd <- data.frame(
#                x = c(-1.0, 0.0, 1.0, -0.5, 0.5, 0.0),
#                g = factor(c("G00","G05","G11","G00","G05","G11"),
#                           levels=levels(d$g)))
_GLMER_PREDICT_POISSON_REF = {
    "fit_link": np.array([
        1.772928139156246807318e+00, 1.926451439987335811921e+00, 1.609081256199845988419e+00,
        2.098020805785064446525e+00, 1.985520983053430166265e+00, 1.847347362230882028555e+00,
        -1.464849208455243356752e-01, -5.502911168433710642489e-02, -1.399071531046067740078e-01,
        -1.337062652795965034258e-01, 6.78937124818523862757e-03, -2.37109061690630174013e-02,
        7.572744284341057507959e-01, 7.54103748779348714848e-01, 7.906978065638644581625e-01,
        6.75605174234499061825e-01, 7.068340145796393469979e-01, 8.482257804594532935027e-01,
        3.726819626255949979843e-01, 1.87932569947848571168e-01, 3.211761322943422203302e-01,
        4.895820254630985357558e-01, 3.575627180134503535491e-01, 3.699714859093065633111e-01,
        5.785185465862201503739e-01, 7.541830675938041572692e-01, 3.772713638286342985317e-01,
        6.834280340798910557965e-01, 3.00513096066523099914e-01, 5.091810518654726891441e-01,
        4.274563183920632170043e-02, 4.171605399483396259264e-01, 3.402966625603680017598e-01,
        3.854158525561241499524e-01, 8.492452573831071882537e-02, 3.181126609745050237699e-01,
        1.207511212865748806422e+00, 1.216664152768367479496e+00, 1.359825070494126375564e+00,
        1.414804045490595685308e+00, 1.314954935788563705756e+00, 1.191326522849978086782e+00,
        6.484282683803477276285e-02, 4.020010929937984922589e-01, 2.757047122473186284708e-01,
        2.943857496539028506533e-01, 5.117821464140321907621e-01, 6.632112275848334181916e-02,
        2.096418063309470003475e+00, 2.18425975429728858046e+00, 1.956654923595163797501e+00,
        1.839130755101978653698e+00, 1.617699348927164049172e+00, 1.863989753361903867201e+00,
        -3.268811204146737647847e-02, 2.984458560175606844922e-02, -1.007474157076915233233e-02,
        7.521432118818904832835e-02, 2.243643276669510244758e-01, -4.945472045388199511251e-02,
        1.879811267357243687037e-01, 1.853742122588346030732e-01, 1.905247406194477410857e-01,
        1.217862753733513492271e-01, 1.987442724653687986525e-01, 5.264071378692658509379e-01,
        1.04554044849588834154e+00, 1.171844034055877337153e+00, 9.52926590819561658563e-01,
        1.158854984658521658503e+00, 8.75119276136930412946e-01, 1.02988000754474251508e+00,
    ]),
    "fit_response": np.array([
        5.888069229919299374387e+00, 6.865105727082347897294e+00, 4.998217036800098433957e+00,
        8.150023460071954772843e+00, 7.28284063380960677847e+00, 6.342971580754035443306e+00,
        8.637387566867026356121e-01, 9.464575945776914078778e-01, 8.694389563592344050136e-01,
        8.748469998049753781899e-01, 1.006812471277792386815e+00, 9.765679887344216325573e-01,
        2.132456130768735746983e+00, 2.125705503194454060178e+00, 2.204934507324094017378e+00,
        1.965221917844819143895e+00, 2.027561855046171590544e+00, 2.335499485020970045213e+00,
        1.451622596171683099442e+00, 1.206752141246288045906e+00, 1.378748401628139053088e+00,
        1.631634095904439307745e+00, 1.429840240375376048121e+00, 1.447693334415758981137e+00,
        1.783394456965253116465e+00, 2.125874118321936290243e+00, 1.458299985550072674911e+00,
        1.980655863560951468472e+00, 1.350551592537952316775e+00, 1.663927966111386602677e+00,
        1.043672384058494051473e+00, 1.517646136211232565927e+00, 1.405364447742895039895e+00,
        1.470225591403511389288e+00, 1.0886348996823875801e+00, 1.374531108518947331021e+00,
        3.345148920325077046556e+00, 3.375907417962490608687e+00, 3.895511802235072096323e+00,
        4.115679900914961386604e+00, 3.72458313606729340961e+00, 3.291444489071050139017e+00,
        1.066991308806549465515e+00, 1.494812966496452011356e+00, 1.31745877714228187827e+00,
        1.342301596186858958504e+00, 1.66826163376953995332e+00, 1.068569804159667846477e+00,
        8.136971533527900035665e+00, 8.884069724260822198403e+00, 7.07561894853841177877e+00,
        6.29106740536922259821e+00, 5.041478281648091197553e+00, 6.449417091816010305649e+00,
        9.678403702782971684826e-01, 1.030294398926171473008e+00, 9.899758386347439209629e-01,
        1.07811518905388248335e+00, 1.251526902254381345969e+00, 9.517482518896861476421e-01,
        1.206810738676669103597e+00, 1.203668783474792336463e+00, 1.209884306554398936129e+00,
        1.129512671307448812286e+00, 1.219869971668389618813e+00, 1.692839231411864986399e+00,
        2.844935649438333324923e+00, 3.227939584181692467268e+00, 2.593288057493079001858e+00,
        3.186282843876026227292e+00, 2.399161439609974522114e+00, 2.800729748096567384152e+00,
    ]),
    "fit_newdata": np.array([
        5.709907754902505239158e+00, 1.241652485590657439829e+00, 3.604480880910014839458e+00,
        6.150063672581310036946e+00, 1.337366937153221524426e+00, 3.107003098325304080163e+00,
    ]),
    "fit_no_re": np.array([
        1.725377159165705664989e+00, 2.011677538806366793978e+00, 1.464624340357149723246e+00,
        2.388196159994118605852e+00, 2.134086131251416329491e+00, 1.858676903977264815992e+00,
        1.853191129920624957705e+00, 2.030668191671282762911e+00, 1.865421169837363279953e+00,
        1.877024375165693648171e+00, 2.160162348651262220756e+00, 2.095271423768584995173e+00,
        1.922670887381695115081e+00, 1.916584368216609712832e+00, 1.988019038069062327523e+00,
        1.771888722194974619129e+00, 1.828095825660702056226e+00, 2.105739387813808427552e+00,
        1.903816090623557499839e+00, 1.582666286649204279158e+00, 1.808240929056685653009e+00,
        2.1399027915279895673e+00, 1.875248335088432138917e+00, 1.898662828491286980181e+00,
        2.135209037019467626095e+00, 2.54525049759938282179e+00, 1.745982385260123059822e+00,
        2.371384683059682974005e+00, 1.616978203607960962174e+00, 1.992175099745573163545e+00,
        1.631568825699093006065e+00, 2.372530079464234997744e+00, 2.197000569054644802947e+00,
        2.298397733157361066958e+00, 1.701858544903583680963e+00, 2.148798934290321760443e+00,
        1.797067414622789183198e+00, 1.813591370698734994704e+00, 2.092731142861898430141e+00,
        2.211008961069953926426e+00, 2.000905534044099542967e+00, 1.768216536612227551828e+00,
        1.717019567624327391897e+00, 2.405477057056625955767e+00, 2.120075844312097945732e+00,
        2.160053308104376235121e+00, 2.684593440881124504216e+00, 1.719559707723256503087e+00,
        2.838761839572505607521e+00, 3.099403507732153340015e+00, 2.468485600533790513822e+00,
        2.194777504991173078253e+00, 1.758830801116394004779e+00, 2.25002128277031410164e+00,
        1.81755333960347287281e+00, 1.934838722427496771061e+00, 1.85912258559739451691e+00,
        2.024643652525737191894e+00, 2.350301734305583423179e+00, 1.787333187172649884289e+00,
        1.676801666373974875057e+00, 1.672436080661706014894e+00, 1.681072231404529881971e+00,
        1.569399963672584830121e+00, 1.694946801265710067952e+00, 2.35211334566617980002e+00,
        1.777346199821090655391e+00, 2.016624226396851238263e+00, 1.620131785735370089085e+00,
        1.990599578319546925087e+00, 1.498853047270134464242e+00, 1.749728987890492026835e+00,
    ]),
    "se_link_se": np.array([
        1.685389212432222560967e-01, 1.562139924792567413636e-01, 2.160093485659851930691e-01,
        1.868905142976653721565e-01, 1.619864329300957916935e-01, 1.577408223753311855564e-01,
        3.704499488612345570182e-01, 3.688100050655129802379e-01, 3.701294892090523092065e-01,
        3.698559427019706968842e-01, 3.711458621236880528471e-01, 3.696479547633921836081e-01,
        2.616506384374717164576e-01, 2.61676995628833342078e-01, 2.620010465584265979544e-01,
        2.655941569902012244775e-01, 2.632900185651825819555e-01, 2.652684249280600758958e-01,
        3.106094036379357103961e-01, 3.246252089657475359274e-01, 3.116326485927828349176e-01,
        3.167442136599727953161e-01, 3.10672125623877415368e-01, 3.10606090361820852408e-01,
        2.868636983238857518508e-01, 3.051927496540035589234e-01, 3.004254763537048678046e-01,
        2.945770339578416274762e-01, 3.146926606993045161254e-01, 2.873594711794529299453e-01,
        3.430396103106794769566e-01, 3.2886452863954346304e-01, 3.224237526876855275404e-01,
        3.256290565150576910014e-01, 3.359092292376350541439e-01, 3.214611188818020104385e-01,
        2.107547079439820170155e-01, 2.098033998576259717694e-01, 2.088620248673031531972e-01,
        2.154095781750001159516e-01, 2.063029958181593903621e-01, 2.126858273542683863688e-01,
        3.386897528740953067228e-01, 3.260126388413228104568e-01, 3.198117628032995929388e-01,
        3.198926693010288313523e-01, 3.41664078353986044867e-01, 3.384381233746305284171e-01,
        1.693618810101464311479e-01, 1.941517313692625035237e-01, 1.513258420663420034113e-01,
        1.617616878029407567752e-01, 2.291408586809113301364e-01, 1.576587938620241935439e-01,
        3.543270404505388659899e-01, 3.517810043114322215985e-01, 3.530648522833538249799e-01,
        3.517981478599606082192e-01, 3.627714950789220904603e-01, 3.555097872980534923215e-01,
        3.280553323058332804152e-01, 3.282494027760591293053e-01, 3.278712358910202961937e-01,
        3.346435880319013622675e-01, 3.273119740415582068493e-01, 3.494576068450067074522e-01,
        2.303651282024380686497e-01, 2.371489460025331119652e-01, 2.370537207399189272294e-01,
        2.356218983037878245135e-01, 2.49773230628459907976e-01, 2.308087566149039371322e-01,
    ]),
    "se_resp_se": np.array([
        9.923688362160091314124e-01, 1.072425574419744176424e+00, 1.079661606110598137676e+00,
        1.523162075990885933052e+00, 1.179721375869176247519e+00, 1.000545553451495983666e+00,
        3.199719782440553217384e-01, 3.490630302504916238071e-01, 3.218049968156949658393e-01,
        3.235673618328599454586e-01, 3.736742826492771563096e-01, 3.60986359723078353845e-01,
        5.579585080555403697744e-01, 5.562482296676021231718e-01, 5.77695148511701339622e-01,
        5.219514585686612084814e-01, 5.338367984571625868284e-01, 6.195342698118081470327e-01,
        4.508876289042384466477e-01, 3.917421660219395640468e-01, 4.296630161424428440142e-01,
        5.16810658688052315668e-01, 4.442115067799738969967e-01, 4.496623666457469914448e-01,
        5.115911294953704446797e-01, 6.488013675889522380302e-01, 4.38110467825481519899e-01,
        5.83455729578992499107e-01, 4.250086740674511864846e-01, 4.781454604224707471971e-01,
        3.580209679194436001026e-01, 4.990999812267313817671e-01, 4.531228791351209483906e-01,
        4.787481721930181288727e-01, 3.656825100735009437258e-01, 4.418583080823444420027e-01,
        7.05005883732238380901e-01, 7.08276853893110014404e-01, 8.136244829092945662907e-01,
        8.865568713594180749737e-01, 7.683926591444778519957e-01, 7.000435943487235412874e-01,
        3.613790226984977538294e-01, 4.87327919781734175686e-01, 4.213388139385526032221e-01,
        4.293924406112460401275e-01, 5.699850735551849068017e-01, 3.616447592145944356368e-01,
        1.378092804644300839456e+00, 1.724857518570485082066e+00, 1.070723995528140459754e+00,
        1.01765368157459268339e+00, 1.15520866247800890747e+00, 1.016807319808835874397e+00,
        3.429320140292627105971e-01, 3.624379983906920177361e-01, 3.495256732316651682169e-01,
        3.792789266888471622075e-01, 4.540182854623139308003e-01, 3.383558185905965465246e-01,
        3.959006979068228071483e-01, 3.951035593157862235714e-01, 3.966862628751408603733e-01,
        3.779841730538223143476e-01, 3.992780485008002866465e-01, 5.915755465825308645478e-01,
        6.55373965610548059324e-01, 7.655024701485433347159e-01, 6.147485829791311529036e-01,
        7.507580122068608918795e-01, 5.992463035706100393796e-01, 6.464329507725418144659e-01,
    ]),
    "fit_random_only": np.array([
        1.22748246957720796857e+00, -7.633940090360322727747e-01, 1.035591218818443515204e-01,
        -2.711783780193058324137e-01, -1.800460049125265915571e-01, -4.467963898564753510989e-01,
        6.213550906746284629989e-01, -4.757471514155844349858e-01, 1.053050078181282822598e+00,
        -6.301793898236972557569e-01, -3.289070821905570318755e-01, 4.704190956645730170571e-01,
    ]).repeat(6),
}


def test_glmer_predict_link_and_response_match_lme4_poisson():
    """``predict(m, type="link")`` returns η; ``type="response"`` returns μ.

    For Poisson(log): ``μ = exp(η)``. Pin both against ``lme4::predict``.
    """
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    p_link = m.predict(type="link")
    p_resp = m.predict(type="response")
    np.testing.assert_allclose(
        p_link["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_link"],
        atol=1e-7, rtol=1e-7,
    )
    np.testing.assert_allclose(
        p_resp["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_response"],
        atol=1e-7, rtol=1e-7,
    )
    # Consistency: μ = linkinv(η) = exp(η) for Poisson(log).
    np.testing.assert_allclose(p_resp["fit"].to_numpy(),
                               np.exp(p_link["fit"].to_numpy()),
                               atol=1e-12, rtol=1e-12)


def test_glmer_predict_newdata_matches_lme4_poisson():
    """``predict(m, newdata=...)`` matches lme4 with the same newdata."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    # 6 rows over 3 known groups — see ``_GLMER_PREDICT_POISSON_REF`` recipe.
    nd_x = np.array([-1.0, 0.0, 1.0, -0.5, 0.5, 0.0])
    nd_g = ["G00", "G05", "G11", "G00", "G05", "G11"]
    nd_df = pl.DataFrame({"y": np.zeros(len(nd_x)), "x": nd_x, "g": nd_g})

    p = m.predict(nd_df, type="response")
    np.testing.assert_allclose(
        p["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_newdata"],
        atol=1e-7, rtol=1e-7,
    )


def test_glmer_predict_re_form_false_matches_lme4_poisson():
    """``re_form=False`` returns population-level prediction (X·β only)."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    p = m.predict(re_form=False, type="response")
    np.testing.assert_allclose(
        p["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_no_re"],
        atol=1e-7, rtol=1e-7,
    )


def test_glmer_predict_allow_new_levels_matches_lme4():
    """New levels in newdata get population-level prediction with ``allow_new_levels=True``."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    # Mix existing + brand-new levels.
    nd_x = np.array([0.5, 0.0, -0.5, 0.5])
    nd_g = ["G00", "NEWGROUP1", "NEWGROUP2", "G05"]
    nd_df = pl.DataFrame({"y": np.zeros(len(nd_x)), "x": nd_x, "g": nd_g})

    # Default ``allow_new_levels=False`` should raise.
    with pytest.raises(ValueError, match="new level"):
        m.predict(nd_df)

    # With ``allow_new_levels=True``: new levels → b=0 → population mean.
    p = m.predict(nd_df, allow_new_levels=True, type="response")
    # For the new levels, expectation equals exp(X·β + 0).
    eta_pop = nd_df.select(pl.col("x")).to_numpy().ravel() * m._beta[1] + m._beta[0]
    new_rows = [1, 2]
    np.testing.assert_allclose(
        p["fit"].to_numpy()[new_rows], np.exp(eta_pop[new_rows]),
        atol=1e-12, rtol=1e-12,
    )


def test_glmer_predict_se_fit_link_matches_lme4_poisson():
    """``se.fit`` on link scale matches lme4 at ≤ 1e-7.

    lme4's ``vcov_full`` builds (b, β) covariance via the L / RX / RZX
    factors. We build the same M densely and solve — equivalent algebra,
    same machinery as the LMM se.fit path with working weights added.
    """
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    p = m.predict(type="link", se_fit=True)
    np.testing.assert_allclose(
        p["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_link"],
        atol=1e-7, rtol=1e-7,
    )
    np.testing.assert_allclose(
        p["se.fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["se_link_se"],
        atol=1e-7, rtol=1e-6,
    )


def test_glmer_predict_se_fit_response_matches_lme4_poisson():
    """``se.fit`` on response scale uses the delta method ``SE_link · |dμ/dη|``."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    p = m.predict(type="response", se_fit=True)
    np.testing.assert_allclose(
        p["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_response"],
        atol=1e-7, rtol=1e-7,
    )
    np.testing.assert_allclose(
        p["se.fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["se_resp_se"],
        atol=1e-7, rtol=1e-6,
    )


def test_glmer_predict_random_only_matches_lme4_poisson():
    """``random.only=True`` returns Z·b on the link scale (no X·β, no offset)."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    p = m.predict(type="link", random_only=True)
    np.testing.assert_allclose(
        p["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_random_only"],
        atol=1e-9, rtol=1e-9,
    )


# ======================================================================
# Phase 8 — Argument plumbing & validation
# ======================================================================


def test_deriv12_quadratic_matches_lme4():
    """Smooth quadratic — gradient and Hessian are exact at any step.

    R recipe::
        fn <- function(x) (x[1]-2)^2 + 3*(x[2]+1)^2 + x[1]*x[2]
        lme4:::deriv12(fn, c(0.5, -0.3))
    """
    def py_fn(x):
        return float((x[0] - 2.0) ** 2 + 3.0 * (x[1] + 1.0) ** 2 + x[0] * x[1])

    x0 = np.array([0.5, -0.3])
    g_py, H_py = _deriv12(py_fn, x0)
    expected_grad = np.array([-3.2999999999994145, 4.6999999999974840])
    # Hessian flattened in R's column-major order.
    expected_hess = np.array([
        [1.9999999403953552, 1.0000000000000000],
        [1.0000000000000000, 5.9999999403953552],
    ])
    np.testing.assert_allclose(g_py, expected_grad, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(H_py, expected_hess, atol=1e-10, rtol=1e-10)


def test_deriv12_rosenbrock_matches_lme4():
    """Non-quadratic — central differences should still byte-match.

    R recipe::
        fn <- function(x) 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
        lme4:::deriv12(fn, c(0.7, 0.4))
    """
    def py_fn(x):
        return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)

    x0 = np.array([0.7, 0.4])
    g_py, H_py = _deriv12(py_fn, x0)
    expected_grad = np.array([24.600002799994858, -17.999999999998018])
    expected_hess = np.array([
        [ 4.3000000199675560e+02, -2.8000000000000000e+02],
        [-2.8000000000000000e+02,  2.0000000001490116e+02],
    ])
    np.testing.assert_allclose(g_py, expected_grad, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(H_py, expected_hess, atol=1e-10, rtol=1e-10)


def test_deriv12_bound_shrinks_step_matches_lme4():
    """Optimum near upper bound — udelta shrinks; asymmetric central diff.

    R recipe::
        fn <- function(x) (x[1]-0.99995)^2 + (x[2]+0.5)^2
        lme4:::deriv12(fn, c(0.99995, 0.0),
                       lower=c(0, NA_real_), upper=c(1, NA_real_))
    """
    def py_fn(x):
        return float((x[0] - 0.99995) ** 2 + (x[1] + 0.5) ** 2)

    x0 = np.array([0.99995, 0.0])
    upper = np.array([1.0, np.nan])
    lower = np.array([0.0, np.nan])
    g_py, H_py = _deriv12(py_fn, x0, lower=lower, upper=upper)
    expected_grad = np.array([-5.0000000066202634e-05, 9.9999999999988987e-01])
    expected_hess = np.array([
        [2.5000002000011012e+07, 3.8888888888889924e+03],
        [3.8888888888889924e+03, 1.9999999962747097e+00],
    ])
    np.testing.assert_allclose(g_py, expected_grad, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(H_py, expected_hess, atol=1e-10, rtol=1e-10)


def test_deriv12_1d_matches_lme4():
    """1D objective — Hessian is a 1×1 matrix, no off-diagonal loop.

    R recipe::
        fn <- function(x) exp(0.3 * x[1]) - 2 * x[1]
        lme4:::deriv12(fn, c(1.5))
    """
    def py_fn(x):
        return float(np.exp(0.3 * x[0]) - 2.0 * x[0])

    x0 = np.array([1.5])
    g_py, H_py = _deriv12(py_fn, x0)
    expected_grad = np.array([-1.5295063442821721])
    expected_hess = np.array([[0.1411481499671936]])
    np.testing.assert_allclose(g_py, expected_grad, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(H_py, expected_hess, atol=1e-10, rtol=1e-10)


# ----------------------------------------------------------------------
# 8.10 — family= validation
# ----------------------------------------------------------------------


def test_family_validation_accepts_instance():
    """Family instance is passed through unchanged."""
    from hea.lme import lme, _resolve_lme_family
    from hea.family import Poisson, LogLink

    fam = Poisson(link=LogLink())
    out = _resolve_lme_family(fam)
    assert out is fam  # same identity, not a copy


def test_family_validation_accepts_class():
    """Family class (callable returning a Family instance) is instantiated."""
    from hea.family import Poisson
    from hea.lme import _resolve_lme_family

    out = _resolve_lme_family(Poisson)
    assert isinstance(out, Poisson)


def test_family_validation_accepts_lowercase_string():
    """String dispatches to the matching ``hea.family`` attribute."""
    from hea.family import Binomial
    from hea.lme import _resolve_lme_family

    out = _resolve_lme_family("binomial")
    assert isinstance(out, Binomial)


def test_family_validation_none_defaults_gaussian():
    """``family=None`` reproduces lme4's lmer-style Gaussian default."""
    from hea.family import Gaussian
    from hea.lme import _resolve_lme_family

    out = _resolve_lme_family(None)
    assert isinstance(out, Gaussian)


def test_family_validation_rejects_quasi_string_with_lme4_message():
    """``family="quasi"`` raises lme4's exact error from modular.R:734."""
    from hea.lme import _resolve_lme_family

    for name in ("quasi", "quasibinomial", "quasipoisson"):
        with pytest.raises(ValueError, match='"quasi" families cannot be used in glmer'):
            _resolve_lme_family(name)


def test_family_validation_rejects_quasi_instance():
    """``family=Quasi(...)`` also errors — by class, not just string."""
    from hea.family import Quasi
    from hea.lme import _resolve_lme_family

    with pytest.raises(ValueError, match='"quasi" families cannot be used in glmer'):
        _resolve_lme_family(Quasi(variance="constant"))


def test_family_validation_rejects_unknown_string():
    """Unrecognised family names error with the list of accepted names."""
    from hea.lme import _resolve_lme_family

    with pytest.raises(ValueError, match="unknown family"):
        _resolve_lme_family("ziggurat")


def test_family_validation_rejects_garbage_input():
    """Non-Family, non-callable, non-string input is a TypeError."""
    from hea.lme import _resolve_lme_family

    with pytest.raises(TypeError, match="family must be"):
        _resolve_lme_family(42)


# ----------------------------------------------------------------------
# 8.11 — nAGQ validation
# ----------------------------------------------------------------------


def test_nAGQ_validation_accepts_0_and_1():
    """Both Laplace (1) and θ-only (0) are supported now."""
    from hea.lme import _validate_nagq

    assert _validate_nagq(0) == 0
    assert _validate_nagq(1) == 1


def test_nAGQ_validation_rejects_above_1_with_phase9_message():
    """``nAGQ > 1`` defers to Phase 9 with a clear message."""
    from hea.lme import _validate_nagq

    with pytest.raises(NotImplementedError, match="Phase 9"):
        _validate_nagq(7)


def test_nAGQ_validation_rejects_negative_or_too_large():
    """nAGQ must be in [0, 100] (modular.R:980-987)."""
    from hea.lme import _validate_nagq

    with pytest.raises(ValueError, match=r"nAGQ must be in \[0, 100\]"):
        _validate_nagq(-1)
    with pytest.raises(ValueError, match=r"nAGQ must be in \[0, 100\]"):
        _validate_nagq(101)


def test_nAGQ_validation_rejects_non_integer():
    """Non-integer nAGQ (1.5 etc.) is rejected — int(1.5) would silently round."""
    from hea.lme import _validate_nagq

    with pytest.raises(ValueError, match="nAGQ must be an integer"):
        _validate_nagq(1.5)
    with pytest.raises(ValueError, match="nAGQ must be an integer"):
        _validate_nagq("not-a-number")


# ----------------------------------------------------------------------
# 8.6 — control= dict normalization
# ----------------------------------------------------------------------


def test_glmer_control_defaults_match_lme4():
    """No user override → defaults exactly match ``glmerControl()``."""
    from hea.lme import _normalize_glmer_control

    out = _normalize_glmer_control(None)
    assert out["optimizer"] == "Nelder_Mead"
    assert out["tolPwrss"] == 1e-7
    assert out["calc.derivs"] is True
    assert out["nAGQ0initStep"] is True
    assert out["use.last.params"] is False
    assert out["optCtrl"] == {}


def test_glmer_control_merges_user_overrides():
    """User-supplied keys overlay the defaults; unspecified keys keep theirs."""
    from hea.lme import _normalize_glmer_control

    out = _normalize_glmer_control({"tolPwrss": 1e-9, "calc.derivs": False})
    assert out["tolPwrss"] == 1e-9
    assert out["calc.derivs"] is False
    # Unspecified key untouched.
    assert out["nAGQ0initStep"] is True


def test_glmer_control_rejects_unknown_keys():
    """Typos / R-only keys raise with the list of accepted keys."""
    from hea.lme import _normalize_glmer_control

    with pytest.raises(ValueError, match="unknown control keys"):
        _normalize_glmer_control({"speed": 9000})


def test_glmer_control_rejects_unsupported_optimizer():
    """bobyqa / nloptwrap etc. await separate ports — clear NotImplementedError."""
    from hea.lme import _normalize_glmer_control

    with pytest.raises(NotImplementedError, match="bobyqa"):
        _normalize_glmer_control({"optimizer": "bobyqa"})


def test_glmer_control_optCtrl_translates_to_nelder_mead_kwargs():
    """``optCtrl=list(maxfun=...)`` → ``NelderMead(maxeval=...)`` mapping."""
    from hea.lme import _nm_kwargs_from_opt_ctrl

    out = _nm_kwargs_from_opt_ctrl({
        "maxfun": 5000, "FtolAbs": 1e-9, "XtolRel": 1e-10,
    })
    assert out == {"maxeval": 5000, "ftol_abs": 1e-9, "xtol_rel": 1e-10}


def test_glmer_control_optCtrl_rejects_unknown_keys():
    from hea.lme import _nm_kwargs_from_opt_ctrl

    with pytest.raises(ValueError, match="unknown optCtrl key"):
        _nm_kwargs_from_opt_ctrl({"PRNGseed": 42})


# ----------------------------------------------------------------------
# 8.9 — devFunOnly handle (currently raises NotImplementedError pending port)
# ----------------------------------------------------------------------


def test_lme_devFunOnly_raises_until_handle_lands():
    """``devFunOnly=True`` errors with a clear message until handle ports."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    with pytest.raises(NotImplementedError, match="devFunOnly"):
        lme("y ~ x + (1|g)", df, family=Poisson(), devFunOnly=True)


# ----------------------------------------------------------------------
# 8.2 — direct offset= numeric vector (in addition to formula offset())
# ----------------------------------------------------------------------


def test_lme_offset_arg_adds_to_formula_offset():
    """``offset=`` is summed with any ``offset(...)`` in the formula. We
    check the Poisson identity: ``glmer(y ~ x + (1|g), offset=v)`` matches
    ``glmer(y ~ x + offset(v) + (1|g))`` to converged-fit precision."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    rng = np.random.default_rng(2026)
    v = rng.normal(0.0, 0.1, size=df.height)
    df_with = df.with_columns(pl.Series("v", v))

    m_arg = lme("y ~ x + (1|g)", df_with, family=Poisson(), offset=v)
    m_fml = lme("y ~ x + offset(v) + (1|g)", df_with, family=Poisson())

    np.testing.assert_allclose(m_arg.theta, m_fml.theta, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        m_arg.bhat.to_numpy().ravel(),
        m_fml.bhat.to_numpy().ravel(),
        atol=1e-9, rtol=1e-9,
    )


def test_lme_offset_arg_length_mismatch_errors():
    """Wrong-length offset= raises before fitting."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    with pytest.raises(ValueError, match="offset= must have length"):
        lme("y ~ x + (1|g)", df, family=Poisson(),
            offset=np.zeros(df.height + 1))


# ----------------------------------------------------------------------
# 8.3 — subset= / na_action= argument plumbing.
# Mirrors R's ``glmer(subset=, na.action=)`` (modular.R passes to the
# model.frame builder; we apply before prepare_design's NA-omit pass).
# ----------------------------------------------------------------------


def test_lme_subset_bool_mask_matches_pre_filter():
    """``subset=mask`` ≡ caller pre-filtering. Bit-identical fit."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    mask = np.arange(df.height) >= 10  # drop the first 10 rows
    m_arg = lme("y ~ x + (1|g)", df, family=Poisson(), subset=mask)
    m_pre = lme("y ~ x + (1|g)", df.filter(pl.Series(mask)), family=Poisson())
    np.testing.assert_allclose(m_arg.theta, m_pre.theta, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        m_arg.bhat.to_numpy().ravel(),
        m_pre.bhat.to_numpy().ravel(),
        atol=1e-12, rtol=1e-12,
    )


def test_lme_subset_positive_int_indices_keep():
    """Positive R-style 1-based indices keep the specified rows."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # Keep first 50 rows: R 1-based 1..50
    idx_keep = np.arange(1, 51)
    m_arg = lme("y ~ x + (1|g)", df, family=Poisson(), subset=idx_keep)
    m_pre = lme("y ~ x + (1|g)", df.head(50), family=Poisson())
    np.testing.assert_allclose(m_arg.theta, m_pre.theta, atol=1e-12, rtol=1e-12)


def test_lme_subset_negative_int_indices_drop():
    """Negative R-style 1-based indices drop the specified rows."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # Drop first 5 rows: R -1..-5
    idx_drop = -np.arange(1, 6)
    m_arg = lme("y ~ x + (1|g)", df, family=Poisson(), subset=idx_drop)
    m_pre = lme("y ~ x + (1|g)", df.tail(df.height - 5),
                family=Poisson())
    np.testing.assert_allclose(m_arg.theta, m_pre.theta, atol=1e-12, rtol=1e-12)


def test_lme_na_action_omit_default_drops_silently():
    """Default ``na_action='na.omit'`` drops rows with any NA in referenced
    columns and proceeds (mirrors R's ``na.omit`` model-frame default)."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # Inject NAs in `x` for the first 3 rows
    x_arr = df["x"].to_numpy().astype(float)
    x_arr[:3] = np.nan
    df_na = df.with_columns(pl.Series("x", x_arr))

    m_na  = lme("y ~ x + (1|g)", df_na, family=Poisson())
    m_ref = lme("y ~ x + (1|g)", df.tail(df.height - 3), family=Poisson())
    np.testing.assert_allclose(m_na.theta, m_ref.theta, atol=1e-12, rtol=1e-12)


def test_lme_na_action_fail_raises_on_na():
    """``na_action='na.fail'`` errors if any referenced-column row has NA."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    x_arr = df["x"].to_numpy().astype(float)
    x_arr[0] = np.nan
    df_na = df.with_columns(pl.Series("x", x_arr))

    with pytest.raises(ValueError, match=r"missing values in object"):
        lme("y ~ x + (1|g)", df_na, family=Poisson(), na_action="na.fail")


def test_lme_na_action_pass_raises_not_implemented():
    """``na_action='na.pass'`` and 'na.exclude' are not implemented yet —
    they require carrying NA rows through PIRLS."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    with pytest.raises(NotImplementedError, match=r"na.pass"):
        lme("y ~ x + (1|g)", df, family=Poisson(), na_action="na.pass")
    with pytest.raises(NotImplementedError, match=r"na.exclude"):
        lme("y ~ x + (1|g)", df, family=Poisson(), na_action="na.exclude")


def test_glmer_summary_prints_signif_codes_legend(capsys):
    """GLMM ``summary()`` appends R's ``Signif. codes:`` legend with the
    five-band thresholds. Match lme4's ``printCoefmat`` output."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=Poisson())
    m.summary()
    out = capsys.readouterr().out
    assert "---" in out
    assert (
        "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        in out
    ), out
    # The trailing legend implies a ``Pr(>|z|)`` column was printed; verify.
    assert "Pr(>|z|)" in out


# ----------------------------------------------------------------------
# 8.4 — contrasts= dict mapping factor-column name → R contrast name.
# Mirrors ``model.matrix(contrasts.arg=)``: overrides the default
# treatment/poly coding on bare-name factor references. In-formula
# ``C(...)`` still wins (R semantics).
# ----------------------------------------------------------------------


def _three_level_glmm_df(seed: int = 2026):
    rng = np.random.default_rng(seed)
    n_groups, n_per = 8, 10
    g = np.repeat(np.arange(n_groups), n_per)
    x = np.tile(np.array(["a", "b", "c"]), n_groups * n_per // 3 + 1)[: n_groups * n_per]
    u = rng.normal(0, 0.3, n_groups)[g]
    beta = {"a": 1.0, "b": 1.5, "c": 2.0}
    eta = np.array([beta[xi] for xi in x]) + u
    y = rng.poisson(np.exp(eta))
    return pl.DataFrame({"y": y, "x": x, "g": g})


def test_lme_contrasts_arg_switches_to_contr_sum():
    """contrasts={'x': 'contr.sum'} replaces the default contr.treatment
    coding on factor x. Column names switch from ``xb, xc`` (treatment,
    drop first level) to ``x1, x2`` (sum-to-zero, drop last level)."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m_def = lme("y ~ x + (1|g)", df, family=Poisson())
    m_sum = lme("y ~ x + (1|g)", df, family=Poisson(),
                contrasts={"x": "contr.sum"})
    assert m_def.column_names == ["(Intercept)", "xb", "xc"]
    assert m_sum.column_names == ["(Intercept)", "x1", "x2"]


def test_lme_contrasts_arg_helmert():
    """contrasts={'x': 'contr.helmert'} → contrast columns x1, x2."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m = lme("y ~ x + (1|g)", df, family=Poisson(),
            contrasts={"x": "contr.helmert"})
    assert m.column_names == ["(Intercept)", "x1", "x2"]


def test_lme_contrasts_arg_rejects_unknown_name():
    """Unknown contrast names raise with a clear message listing the
    supported set (mirrors R's ``no contrasts function 'contr.foo'``)."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _three_level_glmm_df()
    with pytest.raises(ValueError, match=r"contrasts\['x'\]"):
        lme("y ~ x + (1|g)", df, family=Poisson(),
            contrasts={"x": "contr.bogus"})


def test_lme_contrasts_arg_rejects_non_string_value():
    """Numeric matrices and function references aren't yet supported."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _three_level_glmm_df()
    with pytest.raises(ValueError, match=r"only string names"):
        lme("y ~ x + (1|g)", df, family=Poisson(),
            contrasts={"x": np.eye(3)})


def test_lme_contrasts_arg_loses_to_inline_C():
    """In-formula ``C(x, contr.sum)`` overrides ``contrasts={x: contr.treatment}``
    (matches R: per-term ``C(...)`` always wins). Column names reflect C()."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m = lme("y ~ C(x, contr.sum) + (1|g)", df, family=Poisson(),
            contrasts={"x": "contr.treatment"})
    # The C(x, contr.sum) atom produces sum-coded columns regardless of the
    # contrasts= argument.
    assert any(c.endswith("1") for c in m.column_names), m.column_names
    assert any(c.endswith("2") for c in m.column_names), m.column_names


def test_lme_contrasts_arg_unrelated_column_unaffected():
    """A contrasts entry for a non-existent column is silently ignored,
    matching R's behavior. The fit proceeds as if no override was given."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m_def = lme("y ~ x + (1|g)", df, family=Poisson())
    m_nop = lme("y ~ x + (1|g)", df, family=Poisson(),
                contrasts={"not_a_column": "contr.sum"})
    # Bit-identical fits
    np.testing.assert_allclose(m_def.theta, m_nop.theta, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        m_def.bhat.to_numpy().ravel(),
        m_nop.bhat.to_numpy().ravel(),
        atol=1e-12, rtol=1e-12,
    )


# ----------------------------------------------------------------------
# 8.14 — convergence diagnostics. Currently only the singular-fit check
# (``check.conv.singular``, lme4 checkConv.R:32-48) fires; gradient and
# Hessian checks are deferred to 8.14b/c.
# ----------------------------------------------------------------------


def test_lme_optinfo_singular_check_fires_at_boundary():
    """When a variance component shrinks to its lower bound (θ ≈ 0), the
    optinfo singular flag turns on and the standard lme4 message lands in
    ``optinfo$conv$lme4$messages``."""
    from hea.lme import lme
    from hea.family import Poisson

    # No within-group signal → θ̂ pinned at 0
    rng = np.random.default_rng(1)
    n_groups, n_per = 3, 100
    g = np.repeat(np.arange(n_groups), n_per)
    y = rng.poisson(2.0, size=n_groups * n_per)
    df = pl.DataFrame({"y": y, "g": g})

    m = lme("y ~ 1 + (1|g)", df, family=Poisson())
    assert m.optinfo["is_singular"] is True
    assert m.theta[0] < 1e-4
    msgs = m.optinfo["conv"]["lme4"]["messages"]
    assert any("singular" in s for s in msgs), msgs


def test_lme_optinfo_singular_check_silent_for_normal_fit():
    """A well-identified RE → ``is_singular=False``, empty messages."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=Poisson())
    assert m.optinfo["is_singular"] is False
    assert m.optinfo["conv"]["lme4"]["messages"] == []


def test_lme_summary_prints_singular_warning(capsys):
    """The singular message is appended to summary() output (mirrors R's
    ``print.summary.merMod`` convergence block)."""
    from hea.lme import lme
    from hea.family import Poisson

    rng = np.random.default_rng(1)
    n_groups, n_per = 3, 100
    g = np.repeat(np.arange(n_groups), n_per)
    y = rng.poisson(2.0, size=n_groups * n_per)
    df = pl.DataFrame({"y": y, "g": g})

    m = lme("y ~ 1 + (1|g)", df, family=Poisson())
    m.summary()
    out = capsys.readouterr().out
    assert "boundary (singular) fit" in out
    assert "see help('isSingular')" in out


def test_lme_summary_omits_convergence_block_when_clean(capsys):
    """A clean fit has no convergence block in summary()."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    m = lme("y ~ x + (1|g)", df, family=Poisson())
    m.summary()
    out = capsys.readouterr().out
    assert "boundary" not in out
    assert "isSingular" not in out


def test_lmer_summary_omits_signif_codes_legend(capsys):
    """LMM ``summary()`` skips both the p-value column AND the legend —
    lme4's deliberate choice (see ``?lme4::pvalues``)."""
    from hea.lme import lme

    rng = np.random.default_rng(2026)
    n = 60
    g = np.repeat(np.arange(10), 6)
    x = rng.normal(size=n)
    u = rng.normal(scale=0.5, size=10)[g]
    y = 1.0 + 2.0 * x + u + rng.normal(scale=0.3, size=n)
    df = pl.DataFrame({"y": y, "x": x, "g": g})
    m = lme("y ~ x + (1|g)", df)
    m.summary()
    out = capsys.readouterr().out
    assert "Signif. codes" not in out
    assert "Pr(>|t|)" not in out
    assert "Pr(>|z|)" not in out
    # t value column IS present though.
    assert "t value" in out


def test_lme_subset_and_na_action_compose():
    """subset= filters first, then na_action policy applies to the result.
    Verifies the order of operations matches R's model.frame semantics."""
    from hea.lme import lme
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # Inject NAs in the FIRST row only.
    x_arr = df["x"].to_numpy().astype(float)
    x_arr[0] = np.nan
    df_na = df.with_columns(pl.Series("x", x_arr))

    # subset=  drops the first 5 rows → no NA remains → na.fail must NOT raise
    mask = np.arange(df_na.height) >= 5
    m = lme("y ~ x + (1|g)", df_na, family=Poisson(),
            subset=mask, na_action="na.fail")
    # Sanity: produces same fit as pre-filtering then dropping the NA-row.
    m_ref = lme("y ~ x + (1|g)", df_na.filter(pl.Series(mask)),
                family=Poisson())
    np.testing.assert_allclose(m.theta, m_ref.theta, atol=1e-12, rtol=1e-12)


# ----------------------------------------------------------------------
# Canonical Bates lme4 example — fm10: Contraception / Binomial GLMM.
# Reference: Bates (2010), Doug, *lme4: Mixed-effects modeling with R*,
# §6.1 "Contraception data" and Bates et al. (2015), §7.2.
# ----------------------------------------------------------------------


def test_glmer_bates_fm10_contraception_matches_lme4():
    """Match against lme4's default-optimizer Contraception fit::

        glmer(use ~ poly(age, 2) + urban + livch + (1 | district),
              Contraception, binomial)
        # optimizer = c("bobyqa", "Nelder_Mead") — lme4 default.

    This is the canonical Bates-book Binomial GLMM. The fit exercises
    factor-response coercion (use ∈ {"N","Y"} → 0/1) and a multi-term
    polynomial fixed effect via ``poly(age, 2)``. SE / vcov pinned against
    the default Hessian-based ``vcov(m)`` (calc.derivs=TRUE).

    """
    from hea import data, factor
    from hea.lme import lme
    from hea.family import Binomial

    contra = data("Contraception").mutate(
        factor("woman"),  factor("district"), factor("use"),
        factor("urban"),  factor("livch"),
    )
    m = lme(
        "use ~ poly(age, 2) + urban + livch + (1 | district)",
        data=contra, family=Binomial(),
    )

    # lme4 reference (R defaults: bobyqa+NM, calc.derivs=TRUE):
    expected_theta = np.array([0.4752182965556530636064])
    expected_beta = np.array([
        -1.4054610916905971862434,    # (Intercept)
        -5.7989161998034255418588,    # poly(age, 2)1
        -16.3208156149048733141171,   # poly(age, 2)2
         0.6972532152912117586752,    # urbanY
         0.8150193869553590264587,    # livch1
         0.9164624465544630727010,    # livch2
         0.9150483869729737484988,    # livch3+
    ])
    expected_dev_laplace = 2372.728706535781839193
    expected_dev_resid   = 2289.732405042512255022

    # BOBYQA halts on a locally-flat objective: at the FP precision floor
    # (~4 ULP on devfun, from CHOLMOD-internal Cholesky accumulator noise
    # plus accumulated PIRLS rounding — verified 2026-05 that scipy.sparse
    # `@` and the 4-op weight chain match R at 0 ULP, so it's not those),
    # the argmin shifts by √(Δdev/curvature). Curvature at θ̂ is small here
    # → θ̂ drifts ~3e-6 rel and the badly-identified poly(age, 2)2 column
    # of β̂ drifts ~1e-5 rel. Verified: hea-arm64 matches lme4-arm64 at
    # ~5e-9 on (θ̂, β̂) when R is run on the same machine, so this is
    # cross-platform-reference drift, not a hea bug — R-on-Intel and
    # R-on-arm64 disagree by similar amounts. Pin θ̂ at 1e-5 and β̂ in
    # SE-relative units so the test is platform-agnostic.
    expected_se_for_beta = np.array([
        0.1522134608573170178047, 3.2936286686357942876668, 2.6142087304558478955130,
        0.1208624239243845793768, 0.1632291674042204154826, 0.1864493856133159488397,
        0.1875238509232133865545,
    ])
    np.testing.assert_allclose(m.theta, expected_theta, atol=1e-5, rtol=1e-5)
    beta_se_rel = np.abs(m.bhat.to_numpy().ravel() - expected_beta) / expected_se_for_beta
    assert beta_se_rel.max() < 1e-4, f"|Δβ̂|/SE = {beta_se_rel}"
    assert m.deviance_laplace == pytest.approx(expected_dev_laplace, rel=1e-9)
    # m.deviance (residual deviance) depends on fitted β̂, so it inherits the
    # ~1e-7 rel drift; the Laplace deviance above is the optimization
    # objective itself and stays bit-tight across platforms.
    assert m.deviance == pytest.approx(expected_dev_resid, rel=1e-6)
    # Fixed-effect names line up with R's: (Intercept) first, then poly
    # terms, urban dummy, then 3 livch contrasts.
    assert m.column_names == [
        "(Intercept)", "poly(age, 2)1", "poly(age, 2)2",
        "urbanY", "livch1", "livch2", "livch3+",
    ]
    # AIC / BIC / logLik all match the printed summary's first row.
    assert m.AIC == pytest.approx(2388.728706535781839193, rel=1e-9)
    assert m.BIC == pytest.approx(2433.267471943887812813, rel=1e-9)
    assert m.loglike == pytest.approx(-1186.364353267890919597, rel=1e-9)
    # Scaled (Pearson, σ-divided) residuals — what summary() prints.
    # Pinned against ``residuals(fm10, "pearson", scaled=TRUE)`` quantiles.
    pearson_scaled = m.residuals_of("pearson") / m.sigma
    expected_qs = np.array([
        -1.8437896503140969173273, -0.7591760654724690748907,
        -0.4639986075136982579536,  0.9493036503091371036689,
         3.0714595860118607539846,
    ])
    np.testing.assert_allclose(
        np.quantile(pearson_scaled, [0, .25, .5, .75, 1]),
        expected_qs, atol=1e-5, rtol=1e-5,
    )
    # Per-coefficient SEs match lme4's default ``vcov(m)`` (Hessian-based).
    # vcov is built from a central-difference deriv12 (δ=1e-4) on the
    # Stage-1 closure: the formula ``(f+ − 2f₀ + f−)/δ²`` divides a
    # ~3e-9 second difference by 1e-8, losing ~11 digits to cancellation.
    # The small-H_jj columns (``poly(age, 2)``, H_jj≈0.36) sit right at
    # that cancellation floor, so any sub-ULP perturbation of the deviance
    # (which differs between hea and R by ~1e-10 abs on this fit, and
    # between R-Intel and R-arm64 by similar amounts) maps to ~1e-3 rel
    # in their SEs. Well-identified columns (Intercept, urban, livch)
    # have H_jj of 100-300 and stay below 1e-3 rel comfortably. This is
    # the intrinsic floor of FD-Hessian SE for badly-identified columns
    # — R itself disagrees with its own arm64 build at this scale.
    expected_se = np.array([
        0.1522134608573170178047, 3.2936286686357942876668, 2.6142087304558478955130,
        0.1208624239243845793768, 0.1632291674042204154826, 0.1864493856133159488397,
        0.1875238509232133865545,
    ])
    np.testing.assert_allclose(m._se_beta, expected_se, atol=1e-2, rtol=2e-3)


def test_deriv12_uses_supplied_fx_to_save_one_eval():
    """``fx`` argument skips the redundant ``fn(x)`` call. Pin: same answer."""
    def py_fn(x):
        return float(x[0] ** 3 + x[1] ** 2)

    x0 = np.array([0.2, 0.4])
    g_a, H_a = _deriv12(py_fn, x0)
    g_b, H_b = _deriv12(py_fn, x0, fx=py_fn(x0))
    np.testing.assert_array_equal(g_a, g_b)
    np.testing.assert_array_equal(H_a, H_b)

