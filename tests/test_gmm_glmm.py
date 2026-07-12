"""GLMM-specific tests for ``hea.models.gmm(family=...)``.

This file accumulates Phase 2-13 tests of ``lme-family-port.md``. Phase 2
focuses on the ``_GlmResponse`` private class — verifying its mutators
and pure-compute methods match the documented formulas, plus a single
R-oracle cross-check. Phase 3 tests the PIRLS inner loop (_PredState,
_internal_glmer_wrk_iter, _pwrss_update) against ``lme4::glmer``.
"""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from hea.models.gmm import csc_array
from hea.family import Binomial, Gamma, Gaussian, Poisson
from hea.formula import materialize_bars
from conftest import load_dataset
from hea.models.gmm import (
    _GlmResponse,
    _PredState,
    _deriv12,
    _glmm_devfun_factory,
    _internal_glmer_wrk_iter,
    _pwrss_update,
    gmm,
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
    from hea.formula import prepare_design

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
    from hea.R.rng import RGenerator
    gen = RGenerator(2026)
    n_groups, n_per = 10, 5
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = gen.normal(0, 1, n)
    true_b = gen.normal(0, 1, n_groups) * 0.5
    eta = 0.5 + 0.3 * x + true_b[g]
    y = gen.poisson(np.exp(eta)).astype(float)
    fixed_theta = 0.7

    # lme4 reference (see R recipe above). lp0 is the post-init-PIRLS
    # linear predictor; PIRLS one-step produces pdev / delu / delb / mu.
    lp0_r = np.array([
        -0.38559841412085882, -0.45261563898109003, -0.40156879775808924,
        -0.41094901996050881, -0.43531770083821075, -0.49084607909465039,
        -0.41626299458311622, -0.42819732284376549, -0.38072070972572591,
        -0.40531782044690545, -0.66288691403297217, -0.6763809256233777,
        -0.65506493816269507, -0.65524836259488062, -0.7524509432587787,
        0.31998678144544357, 0.28939067070893187, 0.27268771379787682,
        0.22987608830704492, 0.29246197828173032, 0.26728448437417651,
        0.2741521482415491, 0.22275275648299855, 0.34243521192197945,
        0.28306758194468062, 0.17568734650690293, 0.1682690284513515,
        0.098214827004023647, 0.12280510740918449, 0.16804642189181557,
        -0.040337595793121225, -0.011217047237106759, -0.028852712113060897,
        -0.0042488794387191831, -0.010545284074168146, 0.18333872776234761,
        0.1594219123976012, 0.097221012108133109, 0.15874182576234713,
        0.10002159718441234, -0.053086615484341029, 0.027958747905701034,
        -0.055053559106356578, 0.0082220805235372896, -0.039534948054757005,
        -0.37455441934327727, -0.40413546753393786, -0.45079993775051996,
        -0.40046791695021278, -0.4160834572379063,
    ])
    pdev_r = 46.135228359878639
    delu_r = np.array([
        -0.44663565367342584, -0.44567088030310376, -0.70400770270651725,
        0.36963450032648654, 0.36990671024423449, 0.19375187648539827,
        0.000272881029242308, 0.19383628180949175, 0.00082717010020249497,
        -0.4473243967846941,
    ])
    # MU is piecewise-constant per group (β=0 in pp$delb means x has no
    # effect; each group's η = Zb is one value, repeated n_per times).
    mu_per_group = np.array([
        0.73150958340811034, 0.73200376893763663, 0.61091014556436507,
        1.2953023596170867, 1.2955491990377275, 1.1452538477659513,
        1.0001910349653251, 1.1453215156304992, 1.0005791867340421,
        0.73115699287996483,
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
    from hea.R.rng import RGenerator
    gen = RGenerator(seed)
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = gen.normal(0, 1, n)
    b = gen.normal(0, 1, n_groups) * 0.6
    eta = 0.4 + 0.25 * x + b[g]
    y = gen.poisson(np.exp(eta)).astype(float)
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
    "theta": np.array([
        1.016115903317668323,
    ]),
    "beta": np.array([
        0.2903914913216376625, 0.2509776419271669834,
    ]),
    "dev_stage0": 243.34952145315117,
    "dev_stage1": 243.3102979511286037,
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
    theta_hat = np.array([0.7368656664375749])
    dev1_r = 136.7418329949486804

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
# Phase 5: Full glmer fit — tests the public ``hea.models.gmm(..., family=...)``
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
    "theta": np.array([
        1.016115903317668323,
    ]),
    "beta": np.array([
        0.2903914913216376625, 0.2509776419271669834,
    ]),
    "laplace": 243.3102979511286037,
    "deviance": 68.83701292601592,
    "aic": 249.3102915250964,
    "bic": 256.14028988214454,
    "sigma": 1.0,
}


def test_glmer_poisson_full_fit_matches_lme4():
    """Full ``hea.models.gmm(..., family=poisson())`` fit ≡ ``lme4::glmer(..., family=poisson)``.

    Compared to ``lme4::glmer`` with its default optimizer chain
    ``optimizer=c("bobyqa", "Nelder_Mead")`` — both stages of hea use the
    ported BOBYQA + Nelder-Mead implementations. Tolerance ≤ 1e-7 on
    θ̂/β̂ — anything looser would mask actual bugs.
    """
    from hea.models.gmm import gmm  # local import — keep test file's top imports lean
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    r = _GLMER_POISSON_FULLFIT_REF

    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

    np.testing.assert_allclose(m.theta, r["theta"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m._beta, r["beta"],  atol=1e-7, rtol=1e-7)
    # ``deviance(m)`` for glmer fits = residual deviance (= Σ dev_resids),
    # NOT the Laplace value. The Laplace value is on ``deviance_laplace``.
    # deviance/laplace/AIC/BIC: rtol 1e-7 — the glmer θ̂/β̂ match lme4 to
    # ~1e-8 but these criterion-scale quantities amplify that to ~rel 2.6e-8
    # (optimizer stop-band on this data); θ̂/β̂ themselves stay tight above.
    np.testing.assert_allclose(m.deviance, r["deviance"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.deviance_laplace, r["laplace"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.AIC, r["aic"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m.BIC, r["bic"], atol=1e-7, rtol=1e-7)
    assert m.sigma == pytest.approx(r["sigma"])  # = 1 for Poisson
    # Public-API check: bhat as a DataFrame with R-canonical column names.
    assert m.bhat.columns == ["(Intercept)", "x"]
    np.testing.assert_allclose(
        m.bhat.row(0), r["beta"], atol=1e-7, rtol=1e-7,
    )


def test_glmer_binomial_full_fit_matches_lme4_cbpp():
    """Full ``hea.gmm`` fit on cbpp matches ``lme4::glmer(family=binomial)``.

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
    from hea.models.gmm import gmm
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
    m = gmm("y_prop ~ period + (1|herd)", df,
            family=BinomialFamily(), weights=size)

    np.testing.assert_allclose(m.theta, theta_r, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(m._beta, beta_r,  atol=1e-7, rtol=1e-7)
    # Residual deviance is Σ_i dev_resid_i over 56 binomial contributions
    # — a BLAS-touched reduction. The references were dumped on R-Intel-MKL;
    # OpenBLAS-on-Linux-CI drifts by ~1e-7 abs (well below the 1e-5 floor
    # the user policy guards). Same fit; identical algorithm; FP-rounding-
    # order intrinsic.
    np.testing.assert_allclose(m.deviance, dev_r, atol=1e-6, rtol=1e-7)
    np.testing.assert_allclose(m.deviance_laplace, laplace_r, atol=1e-6, rtol=1e-7)


def test_glmer_intercept_only_poisson():
    """No fixed effects (p=0) → Stage 1 has empty β slice, optimize θ only.

    Edge case: Stage 1's par vector is just θ. lme4 happily fits these too;
    we should match.

    R recipe (seed=11 / n_groups=8 / n_per=5 synthetic data)::
        m <- glmer(y ~ 0 + (1|g), data=d, family=poisson(),
                   control=glmerControl(
                       optimizer=c("Nelder_Mead","Nelder_Mead")))
    """
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=11, n_groups=8, n_per=5)
    # R defaults: bobyqa + Nelder_Mead.
    theta_r   = np.array([0.53024471740897927])
    laplace_r = 119.792872854065422

    m = gmm("y ~ 0 + (1|g)", df, family=PoissonFamily())
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
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m_default = gmm("y ~ x + (1|g)", df, family=PoissonFamily())
    m_no_stage0 = gmm(
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
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m_default = gmm("y ~ x + (1|g)", df, family=PoissonFamily())
    # Start from a different θ — should still find the same optimum.
    m_alt = gmm(
        "y ~ x + (1|g)", df, family=PoissonFamily(), start=np.array([2.0]),
    )
    np.testing.assert_allclose(m_default.theta, m_alt.theta, atol=1e-4, rtol=1e-4)


def test_glmer_start_dict_with_theta_and_beta():
    """``start={"theta": ..., "beta": ...}`` overrides both initial values."""
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m_default = gmm("y ~ x + (1|g)", df, family=PoissonFamily())
    m_with_dict = gmm(
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
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)

    with pytest.raises(ValueError, match="unrecognised start keys"):
        gmm("y ~ x + (1|g)", df, family=PoissonFamily(),
            start={"theta": np.array([1.0]), "blah": np.array([0.0])})
    with pytest.raises(ValueError, match="not have both"):
        gmm("y ~ x + (1|g)", df, family=PoissonFamily(),
            start={"theta": np.array([1.0]), "par": np.array([1.0])})
    with pytest.raises(ValueError, match="start theta has shape"):
        gmm("y ~ x + (1|g)", df, family=PoissonFamily(),
            start={"theta": np.array([1.0, 2.0])})  # too many
    with pytest.raises(ValueError, match="start beta has shape"):
        gmm("y ~ x + (1|g)", df, family=PoissonFamily(),
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
    "theta": np.array([
        1.016115903317668323,
    ]),
    "beta": np.array([
        0.2903914913216376625, 0.2509776419271669834,
    ]),
    "eta": np.array([
        1.410618575763747495, 1.008984116283886179, 1.314908011603527438,
        1.258692307627020224, 1.112650719170245051, 0.6484802656970650725,
        0.4268356621102136206, 0.3553131917271484252, 0.6398406977133539453,
        0.4924301284176147986, 0.5088883079114701058, 0.4280186499596442262,
        0.3320296517719327767, 0.3309303885178981885, -0.2516050122900676533,
        0.7256725472931924159, 0.5423099502373873726, 0.4422090719232229516,
        -1.19285035831270303, -0.817772904069066886, -1.073373250908278065,
        -1.032215317690151846, -1.340252049583974925, -0.6229946817996792063,
        1.576265365936122986, 2.043052916300658772, 1.998594910635623556,
        1.578759840006084936, 1.726129475194391638, 1.997260828757384754,
        0.8168964107991325552, 0.9914159586235062882, 0.8857253582622698351,
        1.033176212246468939, 0.9954418375734035429, 1.241942273456929424,
        0.6565879475072170512, 0.2838177377734819595, 0.6525121859590568008,
        0.3006016537843976821, 0.2171896670334557133, 0.7028948090532111115,
        0.3365015307267892331, 0.7157126570304641611, 0.4295046202248447575,
        0.994656000063650847, 0.8173766763422971593, 0.5377163455625784128,
        -0.7792054084830456473, -0.8727893952269596411, -1.273445862655852912,
        -1.031579083177026845, -1.213389834148754565, -0.8670387118314621944,
        1.09771368937355307, 1.201463771861822849, 0.9492449647722798201,
        1.018367398093589449, 1.293165724508679393, 1.008805796267273402,
        -1.315437230669789237, -1.237935186649913089, -1.3298227475379929,
        -1.353017743667277006, -1.715226128109039738, -1.34837996387266057,
        -0.2705179773180875769, 0.4705362917442145743, -0.07058667896009035436,
        0.2184214870598313318, 0.3116061315567875689, 0.1825884680017172412,
    ]),
    "mu": np.array([
        4.098489846605073872, 2.742813219932385138, 3.72440836713951029,
        3.520814333601276935, 3.042412296001069105, 1.912631926846440011,
        1.532400809514812723, 1.426627392217481471, 1.896178789626724592,
        1.636287781736086355, 1.663440932550954932, 1.534214693742889102,
        1.393794176366830451, 1.392262871452873974, 0.777551800827108619,
        2.066120196343083748, 1.719975334365544395, 1.55614105182869511,
        0.303355359113717582, 0.441413631123788508, 0.3418534129496869634,
        0.3562169521168302988, 0.2617796788051917067, 0.536335874946961666,
        4.836858143005048127, 7.714123860948184586, 7.37868110539672184,
        4.848938621178506381, 5.618863814308841853, 7.368843903923817606,
        2.263464062481340022, 2.695047848162313908, 2.424742560807499281,
        2.809976758307406008, 2.705919654166755617, 3.462331733318023463,
        1.92820197112095415, 1.328190829672666995, 1.920359073417487572,
        1.350671199600845584, 1.24257975620534511, 2.01959058272932257,
        1.400041012348513858, 2.045644005493395312, 1.53649618585312786,
        2.703794076292931781, 2.264551387360178136, 1.712092566649681125,
        0.458770401584667753, 0.4177845561549925479, 0.2798655799599198968,
        0.3564436617485257641, 0.2971881514635524857, 0.4201940242689446992,
        2.99730541297464903, 3.324980375089983564, 2.583758094277295303,
        2.76867093471674286, 3.644305179962852304, 2.742324165038875439,
        0.2683569636895323796, 0.2899823596485366917, 0.2645241447005106106,
        0.25845911922611714, 0.1799230304580301809, 0.2596605796078225725,
        0.7629841834635650022, 1.600852487021343418, 0.93184696449925708,
        1.244111333953813681, 1.365616713760832379, 1.2003203361980741,
    ]),
    "res_dev": np.array([
        -0.04884645390704908663, -1.211469441338492103, 0.6277863829239614946,
        0.2498901661640894056, -0.02437229963147290873, 0.7252971746250961305,
        -1.750657481927754278, -0.3776617547232786198, 0.07472289234106493927,
        0.2746702969771119829, 1.531708291669557687, -0.4608601605329862294,
        -1.669607245053057687, -0.3502355862415348353, 1.912173477329732307,
        0.6081766527380599863, -0.5960962743184309298, -0.4773509817848333725,
        0.9961984916937192125, -0.9395888793762817714, -0.8268656637564374412,
        -0.8440579981456609593, -0.7235740166772044457, -1.035698677171079218,
        -0.8987757353211180966, 1.731344908468326338, -0.1406255619334571272,
        -0.3976850654317526934, -0.7204489683674165645, 0.2293015800356683065,
        -0.1786930440893856886, -1.18628149234387692, 0.9241413011417493051,
        0.1121159004987996449, -1.192038436119701617, 1.233884036645889637,
        -0.7370400580887541686, 0.5419404307589417158, -1.95977502454617869,
        1.220412310862178495, 0.6236902809220684629, -0.01380766039494967824,
        0.4760930833667426088, -0.8123193318676235064, -1.752995257183046718,
        -1.190913998766729431, 2.054676687303735427, 0.2142626269063969413,
        -0.9578835018776216126, -0.9140946954829051174, -0.7481518294569892147,
        -0.8443265502736790884, -0.7709580422611239658, 0.7579350052615420053,
        -1.341336440719550493, 0.8542260566751301853, -0.3783187170961173074,
        1.203348250856165924, 0.6718126518456715157, -1.211212919986904835,
        -0.7326076217041866956, -0.7615541473178866205, -0.7273570577103251056,
        -0.7189702625646169354, -0.5998717037134360108, -0.7206394099795300212,
        0.2588519298040967698, -0.5105216847052216345, 1.696728538914729967,
        -1.577410114050124657, -1.652644374183890985, 0.6654889350162482842,
    ]),
    "res_pearson": np.array([
        -0.04864962861702307184, -1.052331882838583477, 0.6609719279825245764,
        0.2553772799677168637, -0.02431547455514571424, 0.7862503952464425794,
        -1.237901777006080728, -0.3571852013864462716, 0.07539567705309770107,
        0.2843335103999931057, 1.811644885654701609, -0.4312933484281847307,
        -1.180590604895206841, -0.3324425327178806566, 2.520385552440207455,
        0.6497004863264164154, -0.5489799821611754949, -0.4458209510873717418,
        1.264839630724914787, -0.6643896681344378941, -0.5846823179724925934,
        -0.5968391342035391789, -0.5116440938828392682, -0.7323495578936070061,
        -0.8352066482505369205, 1.903152888513097274, -0.1394068986154040346,
        -0.3855253571965830406, -0.682945523228154272, 0.2325075729800148427,
        -0.1751195297676840512, -1.032520592667734771, 1.011623062144364704,
        0.1133588310231557622, -1.037053410912243789, 1.363799449474119241,
        -0.6684458403937537652, 0.5829290370297512958, -1.385770209456635538,
        1.419163467494346298, 0.6794770169891328759, -0.01378528299208498127,
        0.507050464226960651, -0.731086599034586504, -1.23955483374198816,
        -1.036168291136472686, 2.482285190483610737, 0.2200337280044147303,
        -0.6773259197643832774, -0.6463625578226144253, -0.5290232319661584137,
        -0.5970290292343629623, -0.5451496596931455496, 0.8944534656887541679,
        -1.153663041804929845, 0.91859770601258528, -0.3631675116780098134,
        1.340997449410668452, 0.7101573192836547754, -1.05213038930029823,
        -0.5180318172559793544, -0.5385001018092165914, -0.5143191078508658753,
        -0.5083887481309132816, -0.4241733495376980856, -0.5095690135867982917,
        0.2713435863111542945, -0.4748891045385976728, 2.142448416915235132,
        -1.11539738835708846, -1.168596043875227419, 0.7299069010709221539,
    ]),
    "res_working": np.array([
        -0.02403076506012490066, -0.6354108282937867003, 0.3424951044882850737,
        0.1361008053806081752, -0.01394035123274238799, 0.568519252392915786,
        -1, -0.2990461241280052973, 0.05475285924578520563,
        0.2222788817001483119, 1.404654064791970525, -0.3482007413444935251,
        -1, -0.2817448338929948171, 2.858263844040740409,
        0.4519968418632327456, -0.4185963135518598333, -0.3573847314002464026,
        2.296463932338621117, -1, -1,
        -1, -1, -1,
        -0.379762665907717345, 0.6852205427775047708, -0.0513209745735937542,
        -0.1750772050342383479, -0.2881123066528660792, 0.08565198344615491954,
        -0.116398606387642628, -0.6289490738793839242, 0.6496596647637103983,
        0.06762448875451011754, -0.630439877082036304, 0.7329362008446478338,
        -0.4813821295812423795, 0.5058077162698830787, -1,
        1.221117915956577038, 0.6095546302055526722, -0.009700274351075351131,
        0.4285295804621312432, -0.5111563902054371278, -1,
        -0.6301493487362537227, 1.649531396589013044, 0.1681611374049198082,
        -1, -1, -1,
        -1, -1, 1.379852978013678566,
        -0.6663669989480455191, 0.5037682740803200998, -0.225933726369448945,
        0.8059206449218350166, 0.3720036476338589027, -0.6353458089496782923,
        -1, -1, -1,
        -1, -1, -1,
        0.3106431583686338205, -0.3753328254118722951, 2.219412751547780172,
        -1, -1, 0.6662218740164413955,
    ]),
    "res_response": np.array([
        -0.0984898466050738719, -1.742813219932385138, 1.27559163286048971,
        0.4791856663987230647, -0.04241229600106910524, 1.087368073153559989,
        -1.532400809514812723, -0.4266273922174814714, 0.1038212103732754077,
        0.3637122182639136447, 2.336559067449044846, -0.5342146937428891018,
        -1.393794176366830451, -0.3922628714528739735, 2.222448199172891492,
        0.933879803656916252, -0.7199753343655443949, -0.5561410518286951099,
        0.696644640886282418, -0.441413631123788508, -0.3418534129496869634,
        -0.3562169521168302988, -0.2617796788051917067, -0.536335874946961666,
        -1.836858143005048127, 5.285876139051815414, -0.3786811053967218399,
        -0.8489386211785063807, -1.618863814308841853, 0.6311560960761823935,
        -0.2634640624813400223, -1.695047848162313908, 1.575257439192500719,
        0.1900232416925939916, -1.705919654166755617, 2.537668266681976537,
        -0.9282019711209541502, 0.6718091703273330051, -1.920359073417487572,
        1.649328800399154416, 0.7574202437946548905, -0.01959058272932256983,
        0.599958987651486142, -1.045644005493395312, -1.53649618585312786,
        -1.703794076292931781, 3.735448612639821864, 0.2879074333503188754,
        -0.458770401584667753, -0.4177845561549925479, -0.2798655799599198968,
        -0.3564436617485257641, -0.2971881514635524857, 0.5798059757310553008,
        -1.99730541297464903, 1.675019624910016436, -0.583758094277295303,
        2.23132906528325714, 1.355694820037147696, -1.742324165038875439,
        -0.2683569636895323796, -0.2899823596485366917, -0.2645241447005106106,
        -0.25845911922611714, -0.1799230304580301809, -0.2596605796078225725,
        0.2370158165364349978, -0.6008524870213434177, 2.06815303550074292,
        -1.244111333953813681, -1.365616713760832379, 0.7996796638019258996,
    ]),
    "working_wts": np.array([
        4.098489846605073872, 2.742813219932385138, 3.72440836713951029,
        3.520814333601276935, 3.042412296001069105, 1.912631926846440011,
        1.532400809514812723, 1.426627392217481471, 1.896178789626724592,
        1.636287781736086355, 1.663440932550954932, 1.534214693742889102,
        1.393794176366830451, 1.392262871452873974, 0.777551800827108619,
        2.066120196343083748, 1.719975334365544395, 1.55614105182869511,
        0.303355359113717582, 0.441413631123788508, 0.3418534129496869634,
        0.3562169521168302988, 0.2617796788051917067, 0.536335874946961666,
        4.836858143005048127, 7.714123860948184586, 7.37868110539672184,
        4.848938621178506381, 5.618863814308841853, 7.368843903923817606,
        2.263464062481340022, 2.695047848162313908, 2.424742560807499281,
        2.809976758307406008, 2.705919654166755617, 3.462331733318023463,
        1.92820197112095415, 1.328190829672666995, 1.920359073417487572,
        1.350671199600845584, 1.24257975620534511, 2.01959058272932257,
        1.400041012348513858, 2.045644005493395312, 1.53649618585312786,
        2.703794076292931781, 2.264551387360178136, 1.712092566649681125,
        0.458770401584667753, 0.4177845561549925479, 0.2798655799599198968,
        0.3564436617485257641, 0.2971881514635524857, 0.4201940242689446992,
        2.99730541297464903, 3.324980375089983564, 2.583758094277295303,
        2.76867093471674286, 3.644305179962852304, 2.742324165038875439,
        0.2683569636895323796, 0.2899823596485366917, 0.2645241447005106106,
        0.25845911922611714, 0.1799230304580301809, 0.2596605796078225725,
        0.7629841834635650022, 1.600852487021343418, 0.93184696449925708,
        1.244111333953813681, 1.365616713760832379, 1.2003203361980741,
    ]),
    "prior_wts": np.ones(72),
    "laplace": 243.3102979511286037,
    "deviance": 68.83701292601592,
    "aic": 249.3102915250964,
    "bic": 256.14028988214454,
    "sigma": 1.0,
    "se_beta": np.array([
        0.3240398280839200673, 0.1067890110072432786,
    ]),
    "t_value": np.array([
        0.8961598734289905055, 2.350219742274265222,
    ]),
    "vcov": np.array([
        0.1050018101846564716, 3.378233554156873537e-05,
        3.378233554156873537e-05, 0.01140389287190512503,
    ]).reshape(2, 2),
    "sd_re_g": np.array([
        1.016115903317668323,
    ]),
    "ranef_g": np.array([
        0.9895708665079689936, 0.3209495804845613476, 0.09721379600562723988,
        -1.281275415166094778, 1.273775043561228282, 0.6592637890781458676,
        0.217243020009267096, 0.3483427836059935778, -1.270218944503318959,
        0.9691542015173817415, -1.570322741609319284, -0.1091674116882686563,
    ]),
}


def test_glmer_phase6_attrs_match_lme4_poisson():
    """Every Phase 6 attribute on a Poisson fit matches lme4 — well-determined
    quantities at ≤1e-7; Hessian-derived se/t/vcov at ≤1e-5 (flat-optimum drift)."""
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    r = _GLMER_PHASE6_POISSON_REF
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

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
    # SE(β̂), t-values and vcov are all derived from the Hessian-based vcov,
    # computed by deriv12 (central differences, δ=1e-4) on the Stage-1 closure:
    # the FD formula ``(f+ − 2f₀ + f−)/δ²`` divides a ~3e-9-scale second
    # difference by 1e-8 — ~11 digits of catastrophic cancellation. That FD
    # floor sits ON TOP OF the glmer flat-optimum θ̂-wander, which differs across
    # BLAS/LAPACK builds (CI Linux OpenBLAS vs the reference); the vcov, being an
    # inverse Hessian, amplifies it. A measured CI flip on vcov[0,0] (≈0.105) was
    # 1.7e-6 rel — so this Hessian-derived TRIO (se/t/vcov) is pinned at
    # rtol=1e-5 (≈6× margin), distinct from the well-determined quantities above
    # which stay at 1e-7. (T4 — bit-exact reference-BLAS linalg — would remove
    # this cross-platform drift; held for now.)
    np.testing.assert_allclose(m._se_beta, r["se_beta"], atol=1e-9, rtol=1e-5)
    np.testing.assert_allclose(m.t_values.row(0), r["t_value"], atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(m._vcov_beta_arr, r["vcov"], atol=5e-9, rtol=1e-5)
    # Variance components: SD per bar.
    np.testing.assert_allclose(m.sd_re["g"], r["sd_re_g"], atol=1e-9, rtol=1e-7)
    # method string.
    assert m.method == "glmer.ML"


def test_glmer_phase6_ranef_match_lme4_poisson():
    """BLUPs match ``ranef(m)`` — covers ``_ranef``/``ranef`` for GLMM."""
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())
    rf = m.ranef()
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
    from hea.models.gmm import gmm
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

    m = gmm("y_prop ~ period + (1|herd)", df,
            family=BinomialFamily(), weights=size)
    # Reference dumped on R-Intel-MKL; OpenBLAS-on-Linux-CI drifts ~1e-7 abs
    # on these BLAS-touched reductions. Same algorithm; FP-rounding-order
    # intrinsic. Pin at the documented 1e-7-rel floor.
    assert m.deviance_laplace == pytest.approx(laplace_r, rel=1e-7, abs=1e-6)
    assert m.deviance         == pytest.approx(dev_r,     rel=1e-7, abs=1e-6)
    assert m.AIC              == pytest.approx(aic_r,     rel=1e-7, abs=1e-6)
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
        sqrt(sum(residuals(m,"pearson")^2)/(nrow(d)-2))  # → 0.4140884...
        # (lme4's own sigma(m) = 0.4477 uses a different scale convention)
    """
    from hea.models.gmm import gmm
    from hea.family import Gamma as GammaFamily

    from hea.R.rng import RGenerator
    gen = RGenerator(11)
    n_groups, n_per = 10, 6
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = gen.normal(0, 1, n)
    b = gen.normal(0, 1, n_groups) * 0.3
    # Generate positive responses with mean linked to log(eta).
    eta = 1.0 + 0.2 * x + b[g]
    mu = np.exp(eta)
    y = gen.gamma(4.0, scale=mu / 4.0, size=n)
    df = pl.DataFrame({"y": y, "x": x, "g": [f"G{gi:02d}" for gi in g]})

    # Pearson dispersion sqrt(Σ pearson²/(n−p)) from the lme4 fit (R:
    # sqrt(sum(residuals(m,"pearson")^2)/(nrow(d)-2))). NOTE: lme4's own
    # sigma(m) uses a different (profiled-scale) convention — 0.4477 here —
    # which only coincided with Pearson on the previous data; hea reports
    # the Pearson estimate, so pin to that.
    sigma_r = 0.41408845

    from hea.family import LogLink
    m = gmm("y ~ x + (1|g)", df, family=GammaFamily(link=LogLink()))
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
        1.410618575763747495, 1.008984116283886179, 1.314908011603527438,
        1.258692307627020224, 1.112650719170245051, 0.6484802656970650725,
        0.4268356621102136206, 0.3553131917271484252, 0.6398406977133539453,
        0.4924301284176147986, 0.5088883079114701058, 0.4280186499596442262,
        0.3320296517719327767, 0.3309303885178981885, -0.2516050122900676533,
        0.7256725472931924159, 0.5423099502373873726, 0.4422090719232229516,
        -1.19285035831270303, -0.817772904069066886, -1.073373250908278065,
        -1.032215317690151846, -1.340252049583974925, -0.6229946817996792063,
        1.576265365936122986, 2.043052916300658772, 1.998594910635623556,
        1.578759840006084936, 1.726129475194391638, 1.997260828757384754,
        0.8168964107991325552, 0.9914159586235062882, 0.8857253582622698351,
        1.033176212246468939, 0.9954418375734035429, 1.241942273456929424,
        0.6565879475072170512, 0.2838177377734819595, 0.6525121859590568008,
        0.3006016537843976821, 0.2171896670334557133, 0.7028948090532111115,
        0.3365015307267892331, 0.7157126570304641611, 0.4295046202248447575,
        0.994656000063650847, 0.8173766763422971593, 0.5377163455625784128,
        -0.7792054084830456473, -0.8727893952269596411, -1.273445862655852912,
        -1.031579083177026845, -1.213389834148754565, -0.8670387118314621944,
        1.09771368937355307, 1.201463771861822849, 0.9492449647722798201,
        1.018367398093589449, 1.293165724508679393, 1.008805796267273402,
        -1.315437230669789237, -1.237935186649913089, -1.3298227475379929,
        -1.353017743667277006, -1.715226128109039738, -1.34837996387266057,
        -0.2705179773180875769, 0.4705362917442145743, -0.07058667896009035436,
        0.2184214870598313318, 0.3116061315567875689, 0.1825884680017172412,
    ]),
    "fit_response": np.array([
        4.098489846605073872, 2.742813219932385138, 3.72440836713951029,
        3.520814333601276935, 3.042412296001069105, 1.912631926846440011,
        1.532400809514812723, 1.426627392217481471, 1.896178789626724592,
        1.636287781736086355, 1.663440932550954932, 1.534214693742889102,
        1.393794176366830451, 1.392262871452873974, 0.777551800827108619,
        2.066120196343083748, 1.719975334365544395, 1.55614105182869511,
        0.303355359113717582, 0.441413631123788508, 0.3418534129496869634,
        0.3562169521168302988, 0.2617796788051917067, 0.536335874946961666,
        4.836858143005048127, 7.714123860948184586, 7.37868110539672184,
        4.848938621178506381, 5.618863814308841853, 7.368843903923817606,
        2.263464062481340022, 2.695047848162313908, 2.424742560807499281,
        2.809976758307406008, 2.705919654166755617, 3.462331733318023463,
        1.92820197112095415, 1.328190829672666995, 1.920359073417487572,
        1.350671199600845584, 1.24257975620534511, 2.01959058272932257,
        1.400041012348513858, 2.045644005493395312, 1.53649618585312786,
        2.703794076292931781, 2.264551387360178136, 1.712092566649681125,
        0.458770401584667753, 0.4177845561549925479, 0.2798655799599198968,
        0.3564436617485257641, 0.2971881514635524857, 0.4201940242689446992,
        2.99730541297464903, 3.324980375089983564, 2.583758094277295303,
        2.76867093471674286, 3.644305179962852304, 2.742324165038875439,
        0.2683569636895323796, 0.2899823596485366917, 0.2645241447005106106,
        0.25845911922611714, 0.1799230304580301809, 0.2596605796078225725,
        0.7629841834635650022, 1.600852487021343418, 0.93184696449925708,
        1.244111333953813681, 1.365616713760832379, 1.2003203361980741,
    ]),
    "fit_newdata": np.array([
        2.798223400284284956, 2.58481846813017091, 1.540645865305753981,
        3.172352850996323248, 2.930415146927733527, 1.198683749816668032,
    ]),
    "fit_no_re": np.array([
        1.52355696444375166, 1.01960291224287225, 1.384497343793165136,
        1.308814075241625519, 1.130974728684837061, 0.7109944886110036721,
        1.111695226779797485, 1.03496086172365942, 1.375601537444076428,
        1.187061051716656879, 1.206759571820205323, 1.113011126918823201,
        1.26467586452162184, 1.26328641663990382, 0.7055209532333136524,
        1.874718943314101161, 1.560640250782709737, 1.411983249326680179,
        1.092452377723830637, 1.589631949439612724, 1.231092718786930051,
        1.282819124944775124, 0.9427287963068671228, 1.931468768365345978,
        1.353224580357631046, 2.158207190685331511, 2.064359207408915431,
        1.356604377639971881, 1.572009019574181155, 2.061607019430379051,
        1.170736012711218299, 1.393964951387209439, 1.254154410730505154,
        1.4534098598524261, 1.399588197200379414, 1.790828719279337289,
        1.551690415784868549, 1.068840822488273279, 1.545378966372037866,
        1.086931549021746379, 0.9999466484474333061, 1.625233973393697351,
        0.9882285788521697478, 1.443931892391569383, 1.08454640168743599,
        1.908491646999511682, 1.598449173652904731, 1.208492314925415645,
        1.633971469926817299, 1.487994959952338991, 0.99677828275204583,
        1.269521250546169711, 1.058474912536648826, 1.496576599356923776,
        1.137189068218839916, 1.261510194532347251, 0.9802876433663364475,
        1.050444277992483189, 1.382663239444161141, 1.040448213416996559,
        1.29031388559154192, 1.394293109024702382, 1.271884926288606055,
        1.242723072321849642, 0.8651058699795604046, 1.248499933828213004,
        0.8509936896702776643, 1.785509312085482136, 1.039334633291437271,
        1.387618403353868102, 1.523139314164073665, 1.338776155293978531,
    ]),
    "se_link_se": np.array([
        0.2440871320377657194, 0.2369274897127532087, 0.230815906270152027,
        0.2262148594128431944, 0.2263634653712263689, 0.3232384351747241702,
        0.3090922126342824927, 0.3135896863330352446, 0.3139177562008805267,
        0.3076580898523741947, 0.3077083087509580039, 0.30904332752036906,
        0.3224451789717902028, 0.3225070651917196218, 0.4345287048112644879,
        0.3446298218205750552, 0.3235081207617416243, 0.3197902169938039441,
        0.5748131590613042574, 0.5697843497286311232, 0.5682070386075938062,
        0.5670060487018034889, 0.589144204959556439, 0.5851886893889733932,
        0.2002056716332980824, 0.1809257250980097764, 0.1729703189109912942,
        0.1995650513227913148, 0.1697969391033485598, 0.1727590384954106051,
        0.2548529980277992046, 0.2409873930278719767, 0.2467318580580190879,
        0.2410824669811160725, 0.2409375449262892155, 0.2610398319781462306,
        0.3126732415119491959, 0.3215075733318513151, 0.3123211070079600082,
        0.3193685165509938528, 0.3314357212546850118, 0.3173355031360147183,
        0.3206236431395189102, 0.2826761101780483898, 0.3036465826993356987,
        0.3121488321244780817, 0.2879599327600677605, 0.2897349958884014032,
        0.5733942864542097562, 0.5688791954792619121, 0.5819078380983475629,
        0.5677810320302154601, 0.5766625674403542678, 0.5690745970662752784,
        0.2299552656334751533, 0.2326609305218474699, 0.2410373902369071231,
        0.2337565109380463357, 0.2420600388704609862, 0.2345499409027388404,
        0.6455831253414562321, 0.6467239232289027084, 0.6455628592275504118,
        0.6456565739052432251, 0.6670062786414453493, 0.6456253584789747313,
        0.4074044706759856749, 0.3733950263743070486, 0.3722354344021468475,
        0.3540896553167470762, 0.3574880892470048699, 0.3539987501635701306,
    ]),
    "se_resp_se": np.array([
        1.000388633225062662, 0.6498478517521677489, 0.8596526934113299934,
        0.7964605203026959845, 0.6886909911984917532, 0.6182361520591636017,
        0.4736531570845207151, 0.4473756366841243337, 0.5952441911713443057,
        0.5034171735920006663, 0.5118545962728325849, 0.4741388143137083011,
        0.4494222126318014876, 0.4490146126314731312, 0.3378685770984262704,
        0.7120466348521080624, 0.5564259880533786173, 0.497638684570495704,
        0.1743726503480499124, 0.2515105758865468499, 0.1942435132259601738,
        0.2019771642219961028, 0.1542259790464217684, 0.3138576840788409039,
        0.9683664340712575536, 1.395683453841265687, 1.276292824727641939,
        0.9676786857496911809, 0.9540658777239680033, 1.273034388448805787,
        0.5768506027869364505, 0.6494725555168608944, 0.5982612378593433755,
        0.6774361295519044646, 0.6519576387451900823, 0.9038064944180618232,
        0.6028971606265641769, 0.4270234107287606551, 0.5997686716909937932,
        0.4313618575189892557, 0.4118353178920082791, 0.6408877937011118187,
        0.4488862502477867222, 0.5782546905057768249, 0.4665518164632190268,
        0.8439861633390796092, 0.6521000654299515809, 0.4960531330286978013,
        0.2630563240480818243, 0.2376689394848166825, 0.1628559727971052862,
        0.2023819478537558503, 0.1713772805372896502, 0.2391217423284673904,
        0.6892461631997580662, 0.773593028833852614, 0.6227823088294430764,
        0.6471948584075091215, 0.8821406543568881542, 0.6432119716192330472,
        0.1732467242775596328, 0.187538525986524679, 0.1707669601851088681,
        0.166875826483454448, 0.1200097889184357086, 0.1676434518473451929,
        0.3108431672766777676, 0.5977503559886035989, 0.3468664594300374659,
        0.4405269530533396649, 0.4881917092048277396, 0.4249118984742671912,
    ]),
    "fit_random_only": np.array([
        0.9895708665079689936, 0.9895708665079689936, 0.9895708665079689936,
        0.9895708665079689936, 0.9895708665079689936, 0.9895708665079689936,
        0.3209495804845613476, 0.3209495804845613476, 0.3209495804845613476,
        0.3209495804845613476, 0.3209495804845613476, 0.3209495804845613476,
        0.09721379600562723988, 0.09721379600562723988, 0.09721379600562723988,
        0.09721379600562723988, 0.09721379600562723988, 0.09721379600562723988,
        -1.281275415166094778, -1.281275415166094778, -1.281275415166094778,
        -1.281275415166094778, -1.281275415166094778, -1.281275415166094778,
        1.273775043561228282, 1.273775043561228282, 1.273775043561228282,
        1.273775043561228282, 1.273775043561228282, 1.273775043561228282,
        0.6592637890781458676, 0.6592637890781458676, 0.6592637890781458676,
        0.6592637890781458676, 0.6592637890781458676, 0.6592637890781458676,
        0.217243020009267096, 0.217243020009267096, 0.217243020009267096,
        0.217243020009267096, 0.217243020009267096, 0.217243020009267096,
        0.3483427836059935778, 0.3483427836059935778, 0.3483427836059935778,
        0.3483427836059935778, 0.3483427836059935778, 0.3483427836059935778,
        -1.270218944503318959, -1.270218944503318959, -1.270218944503318959,
        -1.270218944503318959, -1.270218944503318959, -1.270218944503318959,
        0.9691542015173817415, 0.9691542015173817415, 0.9691542015173817415,
        0.9691542015173817415, 0.9691542015173817415, 0.9691542015173817415,
        -1.570322741609319284, -1.570322741609319284, -1.570322741609319284,
        -1.570322741609319284, -1.570322741609319284, -1.570322741609319284,
        -0.1091674116882686563, -0.1091674116882686563, -0.1091674116882686563,
        -0.1091674116882686563, -0.1091674116882686563, -0.1091674116882686563,
    ]),
}


def test_glmer_predict_link_and_response_match_lme4_poisson():
    """``predict(m, type="link")`` returns η; ``type="response"`` returns μ.

    For Poisson(log): ``μ = exp(η)``. Pin both against ``lme4::predict``.
    """
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

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
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

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
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

    p = m.predict(re_form=False, type="response")
    np.testing.assert_allclose(
        p["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_no_re"],
        atol=1e-7, rtol=1e-7,
    )


def test_glmer_predict_allow_new_levels_matches_lme4():
    """New levels in newdata get population-level prediction with ``allow_new_levels=True``."""
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

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
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

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
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

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
    """``random.only=True`` returns Z·b on the link scale (no X·β, no offset).

    Tolerance covers cross-BLAS drift in the Z·b dense multiplication path
    (~3e-9 abs Linux-OpenBLAS vs reference); see top of test_gmm_glmm.py
    "FP precision floor" note in the plan.
    """
    from hea.models.gmm import gmm
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=PoissonFamily())

    p = m.predict(type="link", random_only=True)
    np.testing.assert_allclose(
        p["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_random_only"],
        atol=1e-7, rtol=1e-7,
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
    from hea.models.gmm import _resolve_lme_family
    from hea.family import Poisson, LogLink

    fam = Poisson(link=LogLink())
    out = _resolve_lme_family(fam)
    assert out is fam  # same identity, not a copy


def test_family_validation_accepts_class():
    """Family class (callable returning a Family instance) is instantiated."""
    from hea.family import Poisson
    from hea.models.gmm import _resolve_lme_family

    out = _resolve_lme_family(Poisson)
    assert isinstance(out, Poisson)


def test_family_validation_accepts_lowercase_string():
    """String dispatches to the matching ``hea.family`` attribute."""
    from hea.family import Binomial
    from hea.models.gmm import _resolve_lme_family

    out = _resolve_lme_family("binomial")
    assert isinstance(out, Binomial)


def test_family_validation_none_defaults_gaussian():
    """``family=None`` reproduces lme4's lmer-style Gaussian default."""
    from hea.family import Gaussian
    from hea.models.gmm import _resolve_lme_family

    out = _resolve_lme_family(None)
    assert isinstance(out, Gaussian)


def test_family_validation_rejects_quasi_string_with_lme4_message():
    """``family="quasi"`` raises lme4's exact error from modular.R:734."""
    from hea.models.gmm import _resolve_lme_family

    for name in ("quasi", "quasibinomial", "quasipoisson"):
        with pytest.raises(ValueError, match='"quasi" families cannot be used in glmer'):
            _resolve_lme_family(name)


def test_family_validation_rejects_quasi_instance():
    """``family=Quasi(...)`` also errors — by class, not just string."""
    from hea.family import Quasi
    from hea.models.gmm import _resolve_lme_family

    with pytest.raises(ValueError, match='"quasi" families cannot be used in glmer'):
        _resolve_lme_family(Quasi(variance="constant"))


def test_family_validation_rejects_unknown_string():
    """Unrecognised family names error with the list of accepted names."""
    from hea.models.gmm import _resolve_lme_family

    with pytest.raises(ValueError, match="unknown family"):
        _resolve_lme_family("ziggurat")


def test_family_validation_rejects_garbage_input():
    """Non-Family, non-callable, non-string input is a TypeError."""
    from hea.models.gmm import _resolve_lme_family

    with pytest.raises(TypeError, match="family must be"):
        _resolve_lme_family(42)


# ----------------------------------------------------------------------
# 8.11 — nAGQ validation
# ----------------------------------------------------------------------


def test_nAGQ_validation_accepts_0_and_1():
    """Both Laplace (1) and θ-only (0) are supported now."""
    from hea.models.gmm import _validate_nagq

    assert _validate_nagq(0) == 0
    assert _validate_nagq(1) == 1


def test_nAGQ_validation_accepts_above_1_for_agq():
    """``nAGQ > 1`` is accepted now that AGQ (Phase 9) lands; the
    single-scalar-RE constraint is enforced at fit time, not here."""
    from hea.models.gmm import _validate_nagq

    assert _validate_nagq(7) == 7
    assert _validate_nagq(25) == 25


def test_nAGQ_validation_rejects_negative_or_too_large():
    """nAGQ must be in [0, 100] (modular.R:980-987)."""
    from hea.models.gmm import _validate_nagq

    with pytest.raises(ValueError, match=r"nAGQ must be in \[0, 100\]"):
        _validate_nagq(-1)
    with pytest.raises(ValueError, match=r"nAGQ must be in \[0, 100\]"):
        _validate_nagq(101)


def test_nAGQ_validation_rejects_non_integer():
    """Non-integer nAGQ (1.5 etc.) is rejected — int(1.5) would silently round."""
    from hea.models.gmm import _validate_nagq

    with pytest.raises(ValueError, match="nAGQ must be an integer"):
        _validate_nagq(1.5)
    with pytest.raises(ValueError, match="nAGQ must be an integer"):
        _validate_nagq("not-a-number")


# ----------------------------------------------------------------------
# 9.1 — GHrule: Gauss-Hermite quadrature nodes/weights (nAGQ>1)
# Reference values from ``lme4:::GHrule(n)`` (lme4 2.0-2), columns (z, w, ldnorm).
# ----------------------------------------------------------------------

_GHRULE_REF = {
    1: [(0.0, 1.0, -0.918938533204673)],
    2: [(-1.0, 0.5, -1.41893853320467),
        (1.0, 0.5, -1.41893853320467)],
    3: [(-1.73205080756888, 0.166666666666667, -2.41893853320467),
        (-1.34584108632233e-16, 0.666666666666667, -0.918938533204673),
        (1.73205080756888, 0.166666666666667, -2.41893853320467)],
    5: [(-2.85697001387281, 0.0112574113277207, -5.00007736328886),
        (-1.35562617997427, 0.222075922005613, -1.83779970312048),
        (3.86509861497841e-17, 0.533333333333333, -0.918938533204673),
        (1.35562617997427, 0.222075922005613, -1.83779970312048),
        (2.85697001387281, 0.0112574113277207, -5.00007736328886)],
}


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_gh_rule_matches_lme4_table(n):
    """``_gh_rule(n)`` reproduces ``lme4:::GHrule(n)`` (z, w, ldnorm) columns.

    atol carries the near-zero middle node of odd rules (scipy vs lme4 differ
    at ~1e-16 there); rtol pins the O(1) nodes/weights.
    """
    from hea.models.gmm import _gh_rule

    got = _gh_rule(n)
    ref = np.array(_GHRULE_REF[n])
    assert got.shape == (n, 3)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_gh_rule_spot_checks_high_order():
    """n ∈ {10, 25}: pin the extreme node + a central weight vs lme4."""
    from hea.models.gmm import _gh_rule

    r10 = _gh_rule(10)
    assert r10.shape == (10, 3)
    np.testing.assert_allclose(
        r10[0], (-4.85946282833231, 4.31065263071828e-06, -12.7261280231764),
        rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(r10[4, :2], (-0.484935707515498, 0.344642334932019),
                               rtol=1e-9)

    r25 = _gh_rule(25)
    assert r25.shape == (25, 3)
    np.testing.assert_allclose(
        r25[0], (-8.71759767839959, 1.5300389979987e-17, -38.9171931744236),
        rtol=1e-9, atol=1e-20)
    # central weight (row 13, 1-based) of the order-25 rule
    np.testing.assert_allclose(r25[12, 1], 0.248169351176485, rtol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 4, 7, 10, 25, 100])
def test_gh_rule_structural_properties(n):
    """Weights sum to 1; nodes/weights symmetric; ldnorm = log φ(z)."""
    from hea.models.gmm import _gh_rule

    r = _gh_rule(n)
    z, w, ldnorm = r[:, 0], r[:, 1], r[:, 2]
    assert r.shape == (n, 3)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=0, atol=1e-12)
    # forward/reverse symmetry (the lme4 #968 symmetrization guarantees it)
    np.testing.assert_allclose(z, -z[::-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(w, w[::-1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        ldnorm, -0.5 * np.log(2 * np.pi) - 0.5 * z**2, rtol=1e-12, atol=0)


def test_gh_rule_order_zero_and_out_of_range():
    """ord=0 → empty (0,3) matrix (lme4 asMatrix); ord∉[0,100] raises."""
    from hea.models.gmm import _gh_rule

    assert _gh_rule(0).shape == (0, 3)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        _gh_rule(101)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        _gh_rule(-1)


# ----------------------------------------------------------------------
# 9.2–9.5 — nAGQ>1 adaptive Gauss-Hermite end-to-end (cbpp binomial).
# References from ``lme4::glmer(cbind(incidence, size-incidence) ~ period +
# (1|herd), cbpp, binomial(), nAGQ=k)`` (lme4 2.0-2, default optimizer chain).
# ``lap`` = -2*logLik(m): on the aic scale at nAGQ=1 (glmerLaplace), on the
# deviance scale at nAGQ>1 (glmerAGQ) — the ~84 jump is lme4's own behaviour.
# ----------------------------------------------------------------------

_CBPP_AGQ_REF = {
    1: dict(
        theta=0.642069925403,
        beta=[-1.398342863999, -0.991924975393,
              -1.128216216348, -1.579745414126],
        dev=73.474283618704, lap=184.053132779086),
    5: dict(
        theta=0.647369199521,
        beta=[-1.399201882121, -0.991438021208,
              -1.127859471008, -1.579506387556],
        dev=73.375824032273, lap=100.011367851843),
    25: dict(
        theta=0.647519912197,
        beta=[-1.399223727356, -0.991408884230,
              -1.127809594563, -1.579480951182],
        dev=73.373002519661, lap=100.010030540190),
}


@pytest.fixture(scope="module")
def cbpp_frame():
    from hea import data as hea_data

    return hea_data("cbpp").with_columns(
        (pl.col("incidence") / pl.col("size")).alias("y_prop"),
        pl.col("herd").cast(pl.String),
        pl.col("period").cast(pl.String),
    )


@pytest.mark.parametrize("k", [1, 5, 25])
def test_glmer_cbpp_agq_matches_lme4(cbpp_frame, k):
    """nAGQ ∈ {1, 5, 25} on cbpp matches ``lme4::glmer(..., nAGQ=k)``.

    θ̂/β̂ and the AGQ-corrected -2logL pin to lme4 at ~1e-9 (cbpp is a
    well-conditioned surface). Residual deviance is a BLAS-touched reduction —
    1e-6 abs covers OpenBLAS-vs-MKL FP-order drift.
    """
    from hea.models.gmm import gmm
    from hea.family import Binomial

    size = cbpp_frame["size"].to_numpy().astype(float)
    m = gmm("y_prop ~ period + (1|herd)", cbpp_frame,
            family=Binomial(), weights=size, nAGQ=k)
    ref = _CBPP_AGQ_REF[k]
    np.testing.assert_allclose(m.theta, [ref["theta"]], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(m._beta, ref["beta"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(m.deviance, ref["dev"], atol=1e-6, rtol=1e-7)
    np.testing.assert_allclose(m.deviance_laplace, ref["lap"],
                               atol=1e-6, rtol=1e-7)


def test_glmer_agq_deviance_decreases_with_nagq(cbpp_frame):
    """The AGQ -2logL (nAGQ>1, all on the deviance scale) refines downward as
    node count rises — Laplace (k=1) is the coarsest approximation."""
    from hea.models.gmm import gmm
    from hea.family import Binomial

    size = cbpp_frame["size"].to_numpy().astype(float)
    lap = {
        k: gmm("y_prop ~ period + (1|herd)", cbpp_frame,
               family=Binomial(), weights=size, nAGQ=k).deviance_laplace
        for k in (5, 10)
    }
    # k=5 → k=10 is a clean ~1.3e-3 decrease, well above optimizer noise.
    assert lap[5] > lap[10]


def test_glmer_agq_rejects_non_scalar_re():
    """nAGQ>1 requires a single scalar RE (modular.R:918-920): crossed factors
    and vector (random-slope) terms raise lme4's exact message."""
    from hea.models.gmm import gmm
    from hea.family import Binomial

    rng = np.random.default_rng(3)
    n = 80
    df = pl.DataFrame({
        "y": rng.integers(0, 2, n).astype(float),
        "x": rng.normal(size=n),
        "a": (np.arange(n) % 8).astype(str),
        "b": (np.arange(n) % 5).astype(str),
    })
    with pytest.raises(ValueError, match="single, scalar random-effects term"):
        gmm("y ~ x + (1|a) + (1|b)", df, family=Binomial(), nAGQ=3)
    with pytest.raises(ValueError, match="single, scalar random-effects term"):
        gmm("y ~ x + (1 + x|a)", df, family=Binomial(), nAGQ=3)


# ----------------------------------------------------------------------
# 8.15 — pre-fit identifiability / response validation (checkNlevels /
# checkZdims / checkZrank / checkResponse — modular.R lFormula/glFormula).
# ----------------------------------------------------------------------


def _pois_df(n, glevels, *, const_y=False, seed=0):
    rng = np.random.default_rng(seed)
    y = np.ones(n) if const_y else (rng.integers(0, 5, n)).astype(float)
    return pl.DataFrame({
        "y": y,
        "x": rng.normal(size=n),
        "g": np.array([str(i % glevels) for i in range(n)]),
    })


def test_gmm_prefit_single_level_grouping_factor_raises():
    """check.nlev.gtr.1 (default stop): grouping factor with one level."""
    from hea.models.gmm import gmm

    df = pl.DataFrame({"y": np.arange(20.0) % 4, "x": np.linspace(0, 1, 20),
                       "g": ["A"] * 20})
    with pytest.raises(ValueError, match="> 1 sampled level"):
        gmm("y ~ x + (1|g)", df, family=Poisson())


def test_gmm_prefit_constant_response_raises():
    """check.response.not.const (default stop): constant response."""
    from hea.models.gmm import gmm

    df = _pois_df(28, 4, const_y=True)
    with pytest.raises(ValueError, match="Response is constant"):
        gmm("y ~ x + (1|g)", df, family=Poisson())


def test_gmm_prefit_nlevels_ge_nobs_raises():
    """check.nobs.vs.nlev (default stop): as many groups as observations."""
    from hea.models.gmm import gmm

    n = 24
    df = pl.DataFrame({"y": np.arange(n) % 5.0, "x": np.linspace(0, 1, n),
                       "g": [str(i) for i in range(n)]})
    with pytest.raises(ValueError, match="must be < number of observations"):
        gmm("y ~ x + (1|g)", df, family=Poisson())


def test_gmm_prefit_few_levels_warns_only_when_enabled():
    """check.nlev.gtreq.5 defaults to ignore (silent); opt-in → warning."""
    from hea.models.gmm import gmm

    df = _pois_df(30, 3)  # 3 < 5 sampled levels
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gmm("y ~ x + (1|g)", df, family=Poisson())
    assert not any("5 sampled levels" in str(w.message) for w in rec)
    with pytest.warns(UserWarning, match="< 5 sampled levels"):
        gmm("y ~ x + (1|g)", df, family=Poisson(),
            control={"check.nlev.gtreq.5": "warning"})


def test_gmm_prefit_level_check_downgrade_to_ignore():
    """The stop-by-default level checks are downgradable: a single-level
    factor with check.nlev.gtr.1='ignore' no longer raises *that* error."""
    from hea.models.gmm import gmm

    df = pl.DataFrame({"y": np.arange(20.0) % 4, "x": np.linspace(0, 1, 20),
                       "g": ["A"] * 20})
    try:
        gmm("y ~ x + (1|g)", df, family=Poisson(),
            control={"check.nlev.gtr.1": "ignore",
                     "check.nobs.vs.nlev": "ignore"})
    except ValueError as e:
        assert "sampled level" not in str(e)


def test_gmm_prefit_checks_pass_for_well_posed_model():
    """≥5 well-sampled levels + varying response → no spurious pre-fit error."""
    from hea.models.gmm import gmm

    m = gmm("y ~ x + (1|g)", _pois_df(40, 8), family=Poisson())
    assert m.theta.shape == (1,)


# ----------------------------------------------------------------------
# 8.16 — autoscale + checkScaleX (modular.R:128-158 / 442-453).
# ----------------------------------------------------------------------


def test_gmm_autoscale_matches_unscaled_fit():
    """autoscale=True is a numerical reparameterization — fitted θ/deviance and
    the *un-scaled* β̂/SE match the unscaled fit to the optimizer floor."""
    from hea.models.gmm import gmm

    rng = np.random.default_rng(1)
    ng, npg = 10, 12
    g = np.repeat(np.arange(ng), npg)
    b = rng.normal(0, 0.5, ng)
    x = rng.normal(500, 200, ng * npg)          # large-scale predictor
    xc = (x - x.mean()) / x.std()
    y = rng.poisson(np.exp(0.4 + 0.8 * xc + b[g])).astype(float)
    df = pl.DataFrame({"y": y, "x": x, "g": g.astype(str)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0 = gmm("y ~ x + (1|g)", df, family=Poisson())
        m1 = gmm("y ~ x + (1|g)", df, family=Poisson(),
                 control={"autoscale": True})
    np.testing.assert_allclose(m1.theta, m0.theta, atol=1e-3)
    np.testing.assert_allclose(m1.deviance, m0.deviance, atol=1e-2)
    np.testing.assert_allclose(m1._beta, m0._beta, atol=1e-3)
    np.testing.assert_allclose(m1._se_beta, m0._se_beta, atol=1e-3)


def test_gmm_checkscalex_warns_on_disparate_scales():
    """check.scaleX (default 'warning') flags very differently-scaled X cols."""
    from hea.models.gmm import gmm

    rng = np.random.default_rng(2)
    n = 60
    g = (np.arange(n) % 6).astype(str)
    df = pl.DataFrame({
        "y": rng.integers(0, 4, n).astype(float),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1e4, n),       # SD ratio ~ 1e4 > tol=1e3
        "g": g,
    })
    with pytest.warns(UserWarning, match="very different scales"):
        gmm("y ~ x1 + x2 + (1|g)", df, family=Poisson())
    # 'ignore' suppresses it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gmm("y ~ x1 + x2 + (1|g)", df, family=Poisson(),
            control={"check.scaleX": "ignore"})


# ----------------------------------------------------------------------
# Phase 12 — glmer.nb (negative binomial + θ estimation, nbinom.R:96).
# Reference: lme4::glmer.nb(y ~ x + (1|g)) on the committed synthetic NB data
# (R seed 42, rnbinom size=2). getME(m,"glmer.nb.theta") / theta / fixef /
# -2logLik / AIC.
# ----------------------------------------------------------------------

_NB_REF = dict(nb_theta=1.8720849154, cov_theta=0.8048198320,
               beta=[0.7458609323, 0.8719223520],
               m2logL=1248.88099643, AIC=1256.88099643, npar=4)


def _nb_frame():
    return pl.read_csv("datasets/synthetic/seed_synth_nb_count.csv").with_columns(
        pl.col("g").cast(pl.String))


def test_glmer_nb_matches_lme4():
    """glmer_nb estimates the NB dispersion θ + (cov θ, β) matching lme4's
    glmer.nb; -2logL/AIC carry the extra θ parameter (npar=4)."""
    from hea.models.gmm import glmer_nb

    m = glmer_nb("y ~ x + (1|g)", _nb_frame())
    np.testing.assert_allclose(m._nb_theta, _NB_REF["nb_theta"], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(m.theta, [_NB_REF["cov_theta"]], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(m._beta, _NB_REF["beta"], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(m.deviance_laplace, _NB_REF["m2logL"], atol=2e-2)
    np.testing.assert_allclose(m.AIC, _NB_REF["AIC"], atol=2e-2)
    assert m.npar == _NB_REF["npar"]


def test_gmm_free_theta_nb_delegates_to_glmer_nb():
    """gmm(family=nb()) with free θ runs the θ-estimation loop (== glmer_nb)."""
    from hea.models.gmm import gmm, glmer_nb
    from hea.family import nb

    df = _nb_frame()
    m_gmm = gmm("y ~ x + (1|g)", df, family=nb())
    m_fn = glmer_nb("y ~ x + (1|g)", df)
    assert hasattr(m_gmm, "_nb_theta")
    np.testing.assert_allclose(m_gmm._nb_theta, m_fn._nb_theta, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(m_gmm._beta, m_fn._beta, atol=1e-9, rtol=1e-9)


def test_gmm_fixed_theta_nb_skips_loop():
    """nb(theta=Θ) (fixed dispersion) fits directly — no θ-estimation loop."""
    from hea.models.gmm import gmm
    from hea.family import nb

    m = gmm("y ~ x + (1|g)", _nb_frame(), family=nb(theta=2.0))
    assert not hasattr(m, "_nb_theta")          # loop not triggered
    assert m.theta.shape == (1,)


# ----------------------------------------------------------------------
# Phase 10 — simulate.merMod (predict.R:673-938). numpy RNG; distributions
# (not R's byte-exact stream) — what the parametric bootstrap needs.
# ----------------------------------------------------------------------


def _sim_poisson_model(seed=0):
    from hea.models.gmm import gmm

    rng = np.random.default_rng(seed)
    ng, npg = 12, 12
    g = np.repeat(np.arange(ng), npg)
    b = rng.normal(0, 0.6, ng)
    x = rng.normal(size=ng * npg)
    y = rng.poisson(np.exp(0.4 + 0.7 * x + b[g])).astype(float)
    df = pl.DataFrame({"y": y, "x": x, "g": g.astype(str)})
    return gmm("y ~ x + (1|g)", df, family=Poisson()), df


def test_gmm_simulate_shape_seed_and_refit():
    """simulate → (n × nsim), seed-reproducible, and the draws refit cleanly."""
    from hea.models.gmm import gmm

    m, df = _sim_poisson_model()
    s = m.simulate(nsim=5, seed=1)
    assert s.shape == (m.n, 5)
    assert s.equals(m.simulate(nsim=5, seed=1))          # reproducible
    df2 = df.with_columns(pl.Series("y", s["sim_1"]))
    m2 = gmm("y ~ x + (1|g)", df2, family=Poisson())     # bootMer building block
    assert np.isfinite(m2.theta[0]) and m2._beta.shape == (2,)


def test_gmm_simulate_shares_one_r_stream_with_set_seed():
    """gmm.simulate draws from the ONE process-global R stream that
    :func:`hea.R.set_seed` controls — R keeps a single ``.Random.seed``, so
    ``set_seed(k); simulate()`` must equal ``simulate(seed=k)``, and
    ``simulate(seed=k)`` leaves that shared stream advanced for the public R
    surface (``runif``) to continue. (Regression for the gmm
    ``_GLOBAL_SIM_RNG`` → shared-stream unification.)"""
    import hea.R as R

    m, _ = _sim_poisson_model()

    s_seeded = m.simulate(nsim=4, seed=314)
    R.set_seed(314)
    s_global = m.simulate(nsim=4)                 # seed=None → continue the stream
    assert s_global.equals(s_seeded), \
        "set_seed(k); simulate() must equal simulate(seed=k) — one R stream"

    # simulate(seed=k) and set_seed(k) leave the stream in the identical state,
    # and runif() reads that very same stream (cross-module): same prefix → same
    # continuation, deterministically.
    R.set_seed(314)
    m.simulate(nsim=4)
    a = np.asarray(R.runif(3))
    m.simulate(nsim=4, seed=314)
    b = np.asarray(R.runif(3))
    assert np.array_equal(a, b)


def test_gmm_simulate_conditional_mean_tracks_fitted():
    """use_u=True draws scatter around the fitted μ (conditional simulation)."""
    m, _ = _sim_poisson_model()
    sims = m.simulate(nsim=400, seed=2, use_u=True).to_numpy()
    assert np.corrcoef(sims.mean(axis=1), m.fitted_values)[0, 1] > 0.99


def test_gmm_simulate_binomial_bernoulli_and_gaussian_sd():
    from hea.models.gmm import gmm

    rng = np.random.default_rng(3)
    ng, npg = 12, 12
    g = np.repeat(np.arange(ng), npg)
    b = rng.normal(0, 0.6, ng)
    x = rng.normal(size=ng * npg)
    p = 1 / (1 + np.exp(-(0.2 + 0.6 * x + b[g])))
    yb = (rng.uniform(size=ng * npg) < p).astype(float)
    mb = gmm("y ~ x + (1|g)",
             pl.DataFrame({"y": yb, "x": x, "g": g.astype(str)}),
             family=Binomial())
    sb = mb.simulate(nsim=3, seed=4).to_numpy()
    assert set(np.unique(sb)).issubset({0.0, 1.0})       # Bernoulli draws

    yg = 0.4 + 0.7 * x + b[g] + rng.normal(0, 0.5, ng * npg)
    mg = gmm("y ~ x + (1|g)", pl.DataFrame({"y": yg, "x": x, "g": g.astype(str)}))
    sg = mg.simulate(nsim=150, seed=5, use_u=True).to_numpy()
    # residual SD of conditional draws ≈ σ̂
    np.testing.assert_allclose((sg - mg.fitted[:, None]).std(), mg.sigma,
                               rtol=0.1)


def test_gmm_simulate_negative_binomial_counts():
    """NB simulate yields non-negative integer counts."""
    from hea.models.gmm import glmer_nb

    m = glmer_nb("y ~ x + (1|g)", _nb_frame())
    s = m.simulate(nsim=3, seed=6).to_numpy()
    assert np.all(s >= 0) and np.all(s == np.round(s))


# ----------------------------------------------------------------------
# Phase 11 — bootMer / confint(method=) (bootMer.R, profile.R:807).
# ----------------------------------------------------------------------


def test_boot_ci_conversions_match_boot_package():
    """``_norm_inter`` + the perc/basic/norm conversions byte-match R's
    ``boot::boot.ci`` (which ``confint.bootMer`` calls).

    R recipe (reproduce the replicate vector with the bit-exact RNG)::
        set.seed(7); t <- rnorm(500, 3.4, 0.8); t0 <- 3.5
        b <- structure(list(t0=t0,t=matrix(t,ncol=1),R=500,sim="parametric"),
                       class="boot")
        sapply(c("perc","basic","norm"),
               function(ty) boot.ci(b, type=ty)[[c(perc="percent",
                   basic="basic",norm="normal")[[ty]]]])
    """
    from hea.R.rng import RMersenneTwister
    from hea.models.gmm import _boot_ci_one

    t = RMersenneTwister(7).rnorm(500, mean=3.4, sd=0.8)
    t0 = 3.5
    np.testing.assert_allclose(_boot_ci_one(t0, t, 0.95, "perc"),
                               [1.91562019410927, 5.15293511924754], atol=1e-10)
    np.testing.assert_allclose(_boot_ci_one(t0, t, 0.95, "basic"),
                               [1.84706488075246, 5.08437980589073], atol=1e-10)
    np.testing.assert_allclose(_boot_ci_one(t0, t, 0.95, "norm"),
                               [1.99893156024296, 5.12905716082019], atol=1e-10)


def test_bootMer_sleepstudy_fixef_ci_matches_lme4():
    """``bootMer(m, fixef, nsim=20, seed=101)`` then percentile CI matches
    lme4 — the NLopt-BOBYQA fit makes simulate byte-exact, so the whole
    simulate→refit→boot.ci chain tracks lme4 to the CHOLMOD floor.

    R recipe::
        m <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy)
        b <- bootMer(m, function(x) fixef(x), nsim=20, seed=101, use.u=FALSE)
        confint(b, type="perc")
    """
    m = gmm("Reaction ~ Days + (Days|Subject)",
            load_dataset("lme4", "sleepstudy"), REML=True)

    def fixef_fun(x):
        return {n: float(v) for n, v in zip(x.column_names, x._beta)}

    b = m.bootMer(fixef_fun, nsim=20, seed=101, use_u=False)
    assert b.t.shape == (20, 2) and b.nfail == 0
    ci = b.confint(type="perc")
    np.testing.assert_allclose(
        ci.filter(pl.col("parameter") == "(Intercept)").row(0)[1:],
        [243.552942479216, 260.474381762565], atol=1e-4)
    np.testing.assert_allclose(
        ci.filter(pl.col("parameter") == "Days").row(0)[1:],
        [7.41510477042856, 11.92888251008135], atol=1e-4)


def test_bootMer_is_seed_reproducible():
    """Same seed ⇒ identical bootstrap draws. The simulated *responses* are
    bit-identical (the RNG is the bit-exact MT); the refit statistics agree to
    the BLAS cross-fit floor (~1e-12 — refits go through CHOLMOD/BLAS, whose
    reductions aren't bit-stable across separate fits, see
    [[cross-fit-deviance-residual-blas-flake]])."""
    m = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"))
    # the RNG-driven part is exactly reproducible
    s1 = m.simulate(nsim=15, seed=99).to_numpy()
    s2 = m.simulate(nsim=15, seed=99).to_numpy()
    np.testing.assert_array_equal(s1, s2)

    def f(x):
        return [float(x._beta[0])]

    a = m.bootMer(f, nsim=15, seed=99)
    b = m.bootMer(f, nsim=15, seed=99)
    np.testing.assert_allclose(a.t, b.t, atol=1e-9)


def test_confint_wald_matches_lme4_dyestuff():
    """``confint(method="Wald")`` — β̂ ± z·SE; NaN rows for variance comps.

    R: ``confint(lmer(Yield ~ 1 + (1|Batch), Dyestuff), method="Wald")``.
    """
    m = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"))
    ci = m.confint(method="Wald")
    row = ci.filter(pl.col("parameter") == "(Intercept)").row(0)
    np.testing.assert_allclose(row[1:], [1489.50921023, 1565.49078977], atol=1e-4)
    # variance components are NaN under Wald
    sig = ci.filter(pl.col("parameter") == ".sig01").row(0)
    assert np.isnan(sig[1]) and np.isnan(sig[2])


def test_confint_boot_matches_lme4_dyestuff():
    """``confint(method="boot", nsim=100, seed=42)`` matches lme4's
    bootstrap CIs (NLopt fit ⇒ simulate byte-exact). ``.sigma`` and the
    intercept pin tightly; ``.sig01``'s lower bound is a near-zero-variance
    boundary case (one replicate fits θ≈0) so only its upper bound is pinned.

    R: ``confint(m, method="boot", nsim=100, seed=42, boot.type="perc")``.
    """
    m = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"))
    ci = m.confint(method="boot", nsim=100, seed=42, boot_type="perc")
    np.testing.assert_allclose(
        ci.filter(pl.col("parameter") == ".sigma").row(0)[1:],
        [31.0015422409, 66.2635695910], atol=1e-3)
    np.testing.assert_allclose(
        ci.filter(pl.col("parameter") == "(Intercept)").row(0)[1:],
        [1486.1377486083, 1565.6216518725], atol=1e-3)
    # .sig01 upper bound (lower is a θ≈0 boundary flip vs R's exact 0)
    assert ci.filter(pl.col("parameter") == ".sig01").row(0)[2] == \
        pytest.approx(69.1179477023, abs=1e-2)


def test_confint_profile_matches_lme4_dyestuff():
    """``confint(method="profile")`` (the default) inverts the profile-ζ
    curve — the canonical Dyestuff CIs (Bates §1.5).

    R: ``confint(m)`` (default method="profile").
    """
    m = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"))
    ci = m.confint(method="profile")
    np.testing.assert_allclose(
        ci.filter(pl.col("parameter") == ".sigma").row(0)[1:],
        [38.2299848826, 67.6576962448], atol=1e-2)
    np.testing.assert_allclose(
        ci.filter(pl.col("parameter") == "(Intercept)").row(0)[1:],
        [1486.4514999688, 1568.5484943681], atol=1e-2)


def test_confint_parm_filter_and_unknown_method():
    """``parm=`` subsets rows (names or indices); a bad method raises."""
    m = gmm("Yield ~ 1 + (1|Batch)", load_dataset("lme4", "Dyestuff"))
    one = m.confint(method="Wald", parm=["(Intercept)"])
    assert one.height == 1 and one["parameter"][0] == "(Intercept)"
    with pytest.raises(ValueError, match="method must be"):
        m.confint(method="nope")


def test_confint_profile_glmm_unknown_scale_raises():
    """11.2 — profiling a non-fixed-scale GLMM (Gamma) raises lme4's exact
    message (profile.R:74-75)."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(8), 6)
    y = rng.gamma(2.0, 2.0, size=48) + 0.1
    df = pl.DataFrame({"y": y, "x": rng.normal(size=48),
                       "g": [f"g{i}" for i in g]})
    m = gmm("y ~ x + (1|g)", df, family=Gamma(link="log"))
    with pytest.raises(NotImplementedError,
                       match="non-fixed scale parameters"):
        m.profile()
    with pytest.raises(NotImplementedError,
                       match="non-fixed scale parameters"):
        m.confint(method="profile")


def test_confint_profile_glmm_scale_known_matches_lme4_cbpp():
    """11.1 — scale-known GLMM profile CIs. cbpp binomial ``(1|herd)``: the
    constrained-Laplace profile (pin one param, re-optimise the rest over the
    Stage-1 ``[θ,β]`` devfun) matches lme4's ``confint(method="profile")`` to
    ~1e-3 (the profile-spline-inversion floor).

    R recipe::
        m <- glmer(cbind(incidence, size-incidence) ~ period + (1|herd),
                   cbpp, binomial)
        confint(m, method="profile")
    """
    d = load_dataset("lme4", "cbpp")
    df = d.with_columns((pl.col("incidence") / pl.col("size")).alias("yp"))
    size = df["size"].to_numpy().astype(float)
    m = gmm("yp ~ period + (1|herd)", df, family=Binomial(), weights=size)
    ci = m.confint(method="profile")

    def row(p):
        return ci.filter(pl.col("parameter") == p).row(0)[1:]

    np.testing.assert_allclose(row(".sig01"),
                               [0.3460704968, 1.0998887415], atol=2e-3)
    np.testing.assert_allclose(row("(Intercept)"),
                               [-1.9011888852, -0.9477681542], atol=2e-3)
    np.testing.assert_allclose(row("period2"),
                               [-1.6168537016, -0.4077095870], atol=2e-3)
    np.testing.assert_allclose(row("period4"),
                               [-2.5008377491, -0.8006870282], atol=2e-3)


# ----------------------------------------------------------------------
# Phase 13 — full-parity fixture matrix (named scenarios on vendored data).
# Inline R recipes (the test-file convention); references from lme4 2.0-2.
# ----------------------------------------------------------------------


def test_glmer_salamander_crossed_re_binomial_matches_lme4():
    """Crossed-RE binomial — the canonical salamander mating model
    (Phase 13.4 'crossed REs' edge case + a binomial scenario).

    R recipe::
        m <- glmer(Mate ~ Cross + (1|Male) + (1|Female), salamander,
                   family=binomial,
                   control=glmerControl(optimizer=c("bobyqa","Nelder_Mead")))
        getME(m,"theta"); fixef(m); AIC(m); logLik(m); deviance(m)
        sqrt(diag(vcov(m)))

    The deviance/AIC/logLik objective matches lme4 tightly (~1e-4); θ̂/β̂/SE
    sit on the GLMM flat-surface eval-noise floor (~1e-4, the documented
    Laplace optimiser floor — see the cm1–cm4 note in Phase 8).
    """
    m = gmm("Mate ~ Cross + (1|Male) + (1|Female)",
            load_dataset("lme4", "salamander"), family=Binomial())
    assert m.n_groups == {"Male": 60, "Female": 60}
    assert m.npar == 6
    # objective — tight
    assert m.AIC == pytest.approx(430.55321272338, abs=1e-3)
    assert m.loglike == pytest.approx(-209.27660636169, abs=1e-3)
    assert m.deviance == pytest.approx(280.05266923684, abs=5e-2)  # residual dev
    # variance components (= θ since σ≡1 for binomial) + β̂ — flat-surface floor
    np.testing.assert_allclose(
        [m.sd_re["Male"][0], m.sd_re["Female"][0]],
        [1.02031290360, 1.08368233431], atol=2e-3)
    np.testing.assert_allclose(
        m._beta, [1.0082303061708, -0.7020656417882,
                  -2.9043009109890, -0.0178654949691], atol=2e-3)
    np.testing.assert_allclose(
        m._se_beta, [0.3937554830, 0.4614657936, 0.5607604607, 0.5431333131],
        atol=2e-3)


def test_glmer_random_slope_poisson_singular_matches_lme4():
    """Vector-bar (random-slope) Poisson GLMM ``(1+x|g)`` — exercises the
    correlated-RE Laplace path AND a singular fit (Phase 13.4: θ on the
    boundary, corr → −1). hea lands on lme4's singular fit to ~1e-7.

    Data: ``datasets/synthetic/seed_synth_vbar_poisson.csv`` (R seed 2024:
    x=rnorm·0.6, b0=rnorm·0.5, b1=rnorm·0.3, y=rpois(exp(0.4+0.5x+b0+b1·x))).
    R recipe::
        m <- glmer(y ~ x + (1+x|g), d, family=poisson,
                   control=glmerControl(optimizer=c("bobyqa","Nelder_Mead")))
        getME(m,"theta"); fixef(m); AIC(m); logLik(m); isSingular(m)
    """
    import polars as pl

    d = pl.read_csv("datasets/synthetic/seed_synth_vbar_poisson.csv")
    m = gmm("y ~ x + (1+x|g)", d, family=Poisson())
    # singular vector-bar fit is sharply determined → tight (~1e-7) parity
    np.testing.assert_allclose(
        m.theta, [0.13383940, -0.11718276, 3.4184318e-05], atol=1e-6)
    np.testing.assert_allclose(m._beta, [0.3396320812, 0.2940939975], atol=1e-6)
    assert m.AIC == pytest.approx(363.072516146, abs=1e-5)
    assert m.loglike == pytest.approx(-176.536258073, abs=1e-5)
    # correlation driven to the −1 boundary (singular), matching lme4.
    assert m.corr_re["g"][0, 1] == pytest.approx(-0.99999996, abs=1e-6)


def test_glmm_predicates_and_logLik_method():
    """GLMM predicate surface + the logLik() method (gmm-lmer-parity #17/#19).

    isGLMM=True / isLMM=isREML=isNLMM=False; a boundary GLMM is isSingular. The
    logLik() METHOD must return the Laplace log-likelihood (= ``m.loglike`` =
    −deviance_Laplace/2), NOT −residual_deviance/2 — this regression-locks the
    GLMM branch fix (``self.deviance`` holds Σ deviance-residuals, a different
    quantity). The hea.R generics route to the same answers.
    """
    import polars as pl
    import hea.R as R

    d = pl.read_csv("datasets/synthetic/seed_synth_vbar_poisson.csv")
    m = gmm("y ~ x + (1+x|g)", d, family=Poisson())
    assert m.isGLMM() is True and m.isLMM() is False
    assert m.isREML() is False and m.isNLMM() is False
    assert m.isSingular() is True                       # corr → −1 boundary
    # logLik() == Laplace logLik (= loglike == −deviance_Laplace/2)
    assert m.logLik() == pytest.approx(m.loglike, abs=0)
    assert m.logLik() == pytest.approx(-0.5 * m.deviance_laplace, abs=0)
    # the two deviances genuinely differ, so the old −residual_dev/2 was wrong
    assert m.deviance != pytest.approx(m.deviance_laplace, abs=1e-6)
    # getME on a GLMM + generic routing
    np.testing.assert_array_equal(R.getME(m, "theta"), m.theta)
    assert R.isGLMM(m) is True and R.isSingular(m) is True
    assert R.logLik(m) == pytest.approx(m.loglike, abs=0)


def test_lmer_sleepstudy_uncorrelated_bars_matches_lme4():
    """gaussian_no_corr — the ``||`` uncorrelated-slopes syntax expands to two
    independent scalar bars. With the NLopt-BOBYQA optimizer θ̂ matches lme4
    to the CHOLMOD floor (~1e-7).

    R: ``lmer(Reaction ~ Days + (Days || Subject), sleepstudy)`` →
    ``getME(m,"theta")`` / ``fixef`` / ``sigma`` / ``REMLcrit``.
    """
    m = gmm("Reaction ~ Days + (Days || Subject)",
            load_dataset("lme4", "sleepstudy"), REML=True)
    assert m.corr_re["Subject"] is None  # bars are independent (no correlation)
    np.testing.assert_allclose(
        m.theta, [0.97989652794932, 0.23423122533467], atol=1e-7)
    np.testing.assert_allclose(m._beta, [251.4051048485, 10.4672859596], atol=1e-6)
    assert m.sigma == pytest.approx(25.5652792016, abs=1e-6)
    assert m.REML_criterion == pytest.approx(1743.6692935815, abs=1e-6)


# ----------------------------------------------------------------------
# 8.12 / 8.13 — restart_edge + check.boundary (modular.R:688-740 / 879-907).
# ----------------------------------------------------------------------


def test_check_boundary_pins_near_zero_when_improving():
    """check.boundary pins a near-bound param to the bound iff it lowers dev."""
    from hea.models.gmm import _check_boundary

    lower, upper = np.array([0.0]), np.array([np.inf])
    def devfun(p):  # minimised at 0
        return float(p[0] ** 2 + 1.0)
    out = _check_boundary(devfun, np.array([1e-7]), devfun(np.array([1e-7])),
                          lower, upper, 1e-5)
    assert out[0] == 0.0                                  # pinned to the bound

    # not pinned when the interior point is strictly better
    def devfun2(p):
        return float((p[0] - 1e-7) ** 2)
    out2 = _check_boundary(devfun2, np.array([1e-7]), 0.0, lower, upper, 1e-5)
    assert out2[0] == 1e-7


def test_restart_edge_restarts_only_on_negative_inward_gradient():
    """restart_edge restarts the optimizer when the inward gradient at a bound
    is negative; otherwise it leaves θ alone."""
    from hea.models.gmm import _restart_edge

    lower, upper = np.array([0.0]), np.array([np.inf])
    # decreasing in θ at the lower bound → inward gradient < 0 → restart
    called = []

    def refit(p0):
        called.append(np.asarray(p0))
        return np.array([2.0])
    out = _restart_edge(lambda p: float(-p[0] + 1), np.array([0.0]),
                        lower, upper, refit)
    assert called and out[0] == 2.0
    # interior point → no restart
    out2 = _restart_edge(lambda p: float((p[0] - 1) ** 2), np.array([1.0]),
                         lower, upper, lambda p: np.array([99.0]))
    assert out2[0] == 1.0


def test_gmm_restart_edge_rejected_for_glmer():
    """restart_edge=True is unsupported for glmer (matches lme4 modular.R:869)."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _poisson_re_df()
    with pytest.raises(NotImplementedError, match="restart_edge"):
        gmm("y ~ x + (1|g)", df, family=Poisson(),
            control={"restart_edge": True})
    # default (off) fits fine
    m = gmm("y ~ x + (1|g)", df, family=Poisson())
    assert m.theta.shape == (1,)


# ----------------------------------------------------------------------
# 8.9 — devFunOnly=True returns a callable _DevFunHandle (lmer.R:46/151/175).
# ----------------------------------------------------------------------


def _poisson_re_df(seed=0):
    rng = np.random.default_rng(seed)
    ng, npg = 8, 10
    g = np.repeat(np.arange(ng), npg)
    b = rng.normal(0, 0.6, ng)
    x = rng.normal(size=ng * npg)
    y = rng.poisson(np.exp(0.3 + 0.5 * x + b[g])).astype(float)
    return pl.DataFrame({"y": y, "x": x, "g": g.astype(str)})


def test_gmm_devfunonly_glmer_stage1_handle():
    """glmer(devFunOnly=True): a callable [θ, β] handle reproducing the Laplace
    deviance at the fitted optimum."""
    from hea.models.gmm import gmm, _DevFunHandle
    from hea.family import Poisson

    df = _poisson_re_df()
    m = gmm("y ~ x + (1|g)", df, family=Poisson(), nAGQ=1)
    h = gmm("y ~ x + (1|g)", df, family=Poisson(), nAGQ=1, devFunOnly=True)
    assert isinstance(h.devfun, _DevFunHandle)
    assert h.devfun.par_names == ["theta1", "(Intercept)", "x"]
    assert h.devfun.lower.shape == (3,) and h.devfun.upper.shape == (3,)
    assert h.devfun.lower[0] == 0.0                 # θ variance bound ≥ 0
    assert np.isinf(h.devfun.lower[1])              # β unbounded
    par = np.concatenate([m.theta, m._beta])
    np.testing.assert_allclose(h.devfun(par), m.deviance_laplace,
                               atol=1e-9, rtol=1e-9)


def test_gmm_devfunonly_glmer_nagq0_theta_only():
    """nAGQ=0 → θ-only Stage-0 closure (lmer.R:151)."""
    from hea.models.gmm import gmm

    df = _poisson_re_df()
    m = gmm("y ~ x + (1|g)", df, family=Poisson(), nAGQ=0)
    h = gmm("y ~ x + (1|g)", df, family=Poisson(), nAGQ=0, devFunOnly=True)
    assert h.devfun.par_names == ["theta1"]
    np.testing.assert_allclose(h.devfun(m.theta), m.deviance_laplace,
                               atol=1e-9, rtol=1e-9)


def test_gmm_devfunonly_lmer_profiled_deviance():
    """lmer(devFunOnly=True) → the profiled REML deviance closure over θ."""
    from hea.models.gmm import gmm

    rng = np.random.default_rng(1)
    ng, npg = 8, 10
    g = np.repeat(np.arange(ng), npg)
    b = rng.normal(0, 0.6, ng)
    x = rng.normal(size=ng * npg)
    y = 0.3 + 0.5 * x + b[g] + rng.normal(0, 0.4, ng * npg)
    df = pl.DataFrame({"y": y, "x": x, "g": g.astype(str)})
    m = gmm("y ~ x + (1|g)", df)
    h = gmm("y ~ x + (1|g)", df, devFunOnly=True)
    assert h.devfun.par_names == ["theta1"]
    assert h.devfun.lower[0] == 0.0
    np.testing.assert_allclose(h.devfun(m.theta), m._optim.fun,
                               atol=1e-9, rtol=1e-9)


# ----------------------------------------------------------------------
# 8.6 — control= dict normalization
# ----------------------------------------------------------------------


def test_glmer_control_defaults_match_lme4():
    """No user override → defaults exactly match ``glmerControl()``."""
    from hea.models.gmm import _normalize_glmer_control

    out = _normalize_glmer_control(None)
    # lme4's glmer default optimizer chain (lmerControl.R:177): bobyqa for
    # Stage 0, Nelder_Mead for Stage 1 — both ported.
    assert out["optimizer"] == ["bobyqa", "Nelder_Mead"]
    assert out["tolPwrss"] == 1e-7
    # lme4's glmerControl()$calc.derivs is literally NULL (resolved to the
    # smart rule at fit time using nobsmax/nparmax — see __init__).
    assert out["calc.derivs"] is None
    assert out["nAGQ0initStep"] is True
    assert out["use.last.params"] is False
    assert out["optCtrl"] == {}
    assert out["restart_edge"] is False
    assert out["boundary.tol"] == 1e-5
    assert out["compDev"] is True
    # Keys added for full glmerControl() surface parity (merControl
    # signature, lmerControl.R:65-185).
    assert out["sparseX"] is False
    assert out["standardize.X"] is False
    assert out["autoscale"] is None
    assert out["check.nobs.vs.rankZ"] == "ignore"
    assert out["check.conv.nobsmax"] == 1e4
    assert out["check.conv.nparmax"] == 20


def test_glmer_control_merges_user_overrides():
    """User-supplied keys overlay the defaults; unspecified keys keep theirs."""
    from hea.models.gmm import _normalize_glmer_control

    out = _normalize_glmer_control({"tolPwrss": 1e-9, "calc.derivs": False})
    assert out["tolPwrss"] == 1e-9
    assert out["calc.derivs"] is False
    # Unspecified key untouched.
    assert out["nAGQ0initStep"] is True


def test_glmer_control_rejects_unknown_keys():
    """Typos / R-only keys raise with the list of accepted keys."""
    from hea.models.gmm import _normalize_glmer_control

    with pytest.raises(ValueError, match="unknown control keys"):
        _normalize_glmer_control({"speed": 9000})


def test_glmer_control_optimizer_chain():
    """lme4's glmer optimizer is ``c(stage0, stage1)`` with each entry in
    {bobyqa, Nelder_Mead} (both ported). A scalar replicates to both stages
    (lmerControl.R:109-112). Genuinely unported optimizers (nloptwrap /
    optimx / L-BFGS-B) raise ``NotImplementedError``."""
    from hea.models.gmm import _normalize_glmer_control

    # bobyqa is ported now → accepted, replicated to a 2-stage chain.
    assert _normalize_glmer_control(
        {"optimizer": "bobyqa"})["optimizer"] == ["bobyqa", "bobyqa"]
    assert _normalize_glmer_control(
        {"optimizer": "Nelder_Mead"})["optimizer"] == ["Nelder_Mead", "Nelder_Mead"]
    assert _normalize_glmer_control(
        {"optimizer": ["bobyqa", "Nelder_Mead"]})["optimizer"] == \
        ["bobyqa", "Nelder_Mead"]
    # Unported optimizers still raise with a clear message.
    for opt in ("nloptwrap", "optimx", "L-BFGS-B"):
        with pytest.raises(NotImplementedError, match="optimizer"):
            _normalize_glmer_control({"optimizer": opt})


def test_glmer_optimizer_dispatch_per_stage():
    """Full per-stage dispatch — ``optimizer=c(stage0, stage1)`` actually
    routes each stage to the named ported optimizer. The default chain
    reproduces the no-control fit byte-for-byte; the other ported combos
    minimise the same Laplace objective and label ``optinfo`` correctly."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    base = gmm("y ~ x + (1|g)", df, family=Poisson())
    explicit = gmm("y ~ x + (1|g)", df, family=Poisson(),
                   control={"optimizer": ["bobyqa", "Nelder_Mead"]})
    # Default and the explicit default chain are the same path → identical.
    np.testing.assert_array_equal(base.theta, explicit.theta)
    assert base.optinfo["optimizer"] == "bobyqa+Nelder_Mead"

    # Every ported (stage0, stage1) combo fits and minimises the same devfun.
    for chain in (["Nelder_Mead", "Nelder_Mead"], ["bobyqa", "bobyqa"],
                  ["Nelder_Mead", "bobyqa"]):
        m = gmm("y ~ x + (1|g)", df, family=Poisson(),
                control={"optimizer": chain})
        assert m.optinfo["optimizer"] == "+".join(chain)
        assert m.deviance_laplace == pytest.approx(base.deviance_laplace, abs=1e-4)


def test_glmer_control_optCtrl_translates_to_nelder_mead_kwargs():
    """``optCtrl=list(maxfun=...)`` → ``NelderMead(maxeval=...)`` mapping."""
    from hea.models.gmm import _nm_kwargs_from_opt_ctrl

    out = _nm_kwargs_from_opt_ctrl({
        "maxfun": 5000, "FtolAbs": 1e-9, "XtolRel": 1e-10,
    })
    assert out == {"maxeval": 5000, "ftol_abs": 1e-9, "xtol_rel": 1e-10}


def test_glmer_control_optCtrl_rejects_unknown_keys():
    from hea.models.gmm import _nm_kwargs_from_opt_ctrl

    with pytest.raises(ValueError, match="unknown optCtrl key"):
        _nm_kwargs_from_opt_ctrl({"PRNGseed": 42})


def test_glmer_control_optCtrl_routes_per_optimizer():
    """A single ``optCtrl`` list is split per stage: bobyqa picks up
    ``rhobeg``/``rhoend``/``npt``/``maxfun``; Nelder_Mead picks up
    ``XtolRel``/``FtolAbs``/… Each ignores the other's keys (lme4's per-stage
    behaviour); a key in neither vocabulary still raises."""
    from hea.models.gmm import (
        _bobyqa_kwargs_from_opt_ctrl, _nm_kwargs_from_opt_ctrl,
    )

    mixed = {"rhoend": 1e-9, "XtolRel": 1e-10, "maxfun": 9}
    assert _bobyqa_kwargs_from_opt_ctrl(mixed) == {"rhoend": 1e-9, "maxfun": 9}
    assert _nm_kwargs_from_opt_ctrl(mixed) == {"xtol_rel": 1e-10, "maxeval": 9}
    # bobyqa-only keys no longer crash the NM translator (skipped, not raised).
    assert _nm_kwargs_from_opt_ctrl({"rhoend": 1e-9}) == {}
    # NM-only keys are skipped by the bobyqa translator.
    assert _bobyqa_kwargs_from_opt_ctrl({"XtolRel": 1e-9}) == {}
    # Genuine typos still raise from both.
    for fn in (_bobyqa_kwargs_from_opt_ctrl, _nm_kwargs_from_opt_ctrl):
        with pytest.raises(ValueError, match="unknown optCtrl key"):
            fn({"bogus": 1})


def test_glmer_optCtrl_bobyqa_tuning_runs_end_to_end():
    """Selecting bobyqa AND tuning its ``rhoend`` via ``optCtrl`` fits
    (the knob is wired, not just accepted)."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=Poisson(),
            control={"optimizer": ["bobyqa", "bobyqa"],
                     "optCtrl": {"rhoend": 1e-9}})
    assert m.optinfo["optimizer"] == "bobyqa+bobyqa"
    assert np.isfinite(m.deviance_laplace)


def test_glmer_calc_derivs_null_resolves_to_smart_rule():
    """calc.derivs=None (lme4's default) resolves via the nobsmax/nparmax
    smart rule (lmer.R:51-53). A small fit → True → identical to explicit
    True; squeezing nparmax below the (θ, β) count flips it off → identical
    to explicit False (RX-based vcov). This makes those keys functional."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # None (default) → True for this small problem → same path as explicit True.
    m_null = gmm("y ~ x + (1|g)", df, family=Poisson())
    m_true = gmm("y ~ x + (1|g)", df, family=Poisson(),
                 control={"calc.derivs": True})
    np.testing.assert_array_equal(m_null._se_beta, m_true._se_beta)
    # Squeezing nparmax below npar flips the smart rule OFF → RX fallback,
    # i.e. the same path as an explicit calc.derivs=False.
    m_off = gmm("y ~ x + (1|g)", df, family=Poisson(),
                control={"check.conv.nparmax": 1})
    m_false = gmm("y ~ x + (1|g)", df, family=Poisson(),
                  control={"calc.derivs": False})
    np.testing.assert_array_equal(m_off._se_beta, m_false._se_beta)
    assert np.all(np.isfinite(m_off._se_beta))


# ----------------------------------------------------------------------
# 8.9 — devFunOnly handle (currently raises NotImplementedError pending port)
# ----------------------------------------------------------------------


def test_gmm_devFunOnly_returns_callable_handle():
    """``devFunOnly=True`` returns an unfitted instance carrying a callable
    ``_DevFunHandle`` (8.9) — no longer raises."""
    from hea.models.gmm import gmm, _DevFunHandle
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    h = gmm("y ~ x + (1|g)", df, family=Poisson(), devFunOnly=True)
    assert isinstance(h.devfun, _DevFunHandle)
    val = h.devfun(np.concatenate([np.array([0.5]), np.zeros(2)]))
    assert np.isfinite(val)


# ----------------------------------------------------------------------
# 8.2 — direct offset= numeric vector (in addition to formula offset())
# ----------------------------------------------------------------------


def test_gmm_offset_arg_adds_to_formula_offset():
    """``offset=`` is summed with any ``offset(...)`` in the formula. We
    check the Poisson identity: ``glmer(y ~ x + (1|g), offset=v)`` matches
    ``glmer(y ~ x + offset(v) + (1|g))`` to converged-fit precision."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    rng = np.random.default_rng(2026)
    v = rng.normal(0.0, 0.1, size=df.height)
    df_with = df.with_columns(pl.Series("v", v))

    m_arg = gmm("y ~ x + (1|g)", df_with, family=Poisson(), offset=v)
    m_fml = gmm("y ~ x + offset(v) + (1|g)", df_with, family=Poisson())

    np.testing.assert_allclose(m_arg.theta, m_fml.theta, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        m_arg.bhat.to_numpy().ravel(),
        m_fml.bhat.to_numpy().ravel(),
        atol=1e-9, rtol=1e-9,
    )


def test_gmm_offset_arg_length_mismatch_errors():
    """Wrong-length offset= raises before fitting."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    with pytest.raises(ValueError, match="offset= must have length"):
        gmm("y ~ x + (1|g)", df, family=Poisson(),
            offset=np.zeros(df.height + 1))


# ----------------------------------------------------------------------
# 8.3 — subset= / na_action= argument plumbing.
# Mirrors R's ``glmer(subset=, na.action=)`` (modular.R passes to the
# model.frame builder; we apply before prepare_design's NA-omit pass).
# ----------------------------------------------------------------------


def test_gmm_subset_bool_mask_matches_pre_filter():
    """``subset=mask`` ≡ caller pre-filtering. Bit-identical fit."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    mask = np.arange(df.height) >= 10  # drop the first 10 rows
    m_arg = gmm("y ~ x + (1|g)", df, family=Poisson(), subset=mask)
    m_pre = gmm("y ~ x + (1|g)", df.filter(pl.Series(mask)), family=Poisson())
    np.testing.assert_allclose(m_arg.theta, m_pre.theta, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        m_arg.bhat.to_numpy().ravel(),
        m_pre.bhat.to_numpy().ravel(),
        atol=1e-12, rtol=1e-12,
    )


def test_gmm_subset_positive_int_indices_keep():
    """Non-negative 0-based indices keep the specified rows.
    R's ``subset = 1:50`` (1-based) becomes ``range(50)`` here."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    idx_keep = np.arange(50)
    m_arg = gmm("y ~ x + (1|g)", df, family=Poisson(), subset=idx_keep)
    m_pre = gmm("y ~ x + (1|g)", df.head(50), family=Poisson())
    np.testing.assert_allclose(m_arg.theta, m_pre.theta, atol=1e-12, rtol=1e-12)


def test_gmm_subset_negative_int_indices_drop():
    """Negative indices drop the rows they reference (Python convention:
    -1 is the last row). R's ``subset = -(1:5)`` (drop rows 1..5 1-based)
    is expressed here as ``np.arange(5)`` of POSITIVE values then negated
    via the rule ``-(n - k)``; simplest is to enumerate the rows to drop
    in Python 0-based form."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # Drop first 5 rows: 0-based positions [0..4] → reference as -n, -(n-1),
    # ..., -(n-4) under Python's slice convention.
    n = df.height
    idx_drop = -np.arange(n, n - 5, -1)
    m_arg = gmm("y ~ x + (1|g)", df, family=Poisson(), subset=idx_drop)
    m_pre = gmm("y ~ x + (1|g)", df.tail(df.height - 5),
                family=Poisson())
    np.testing.assert_allclose(m_arg.theta, m_pre.theta, atol=1e-12, rtol=1e-12)


def test_gmm_na_action_omit_default_drops_silently():
    """Default ``na_action='na.omit'`` drops rows with any NA in referenced
    columns and proceeds (mirrors R's ``na.omit`` model-frame default)."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # Inject NAs in `x` for the first 3 rows
    x_arr = df["x"].to_numpy().astype(float)
    x_arr[:3] = np.nan
    df_na = df.with_columns(pl.Series("x", x_arr))

    m_na  = gmm("y ~ x + (1|g)", df_na, family=Poisson())
    m_ref = gmm("y ~ x + (1|g)", df.tail(df.height - 3), family=Poisson())
    np.testing.assert_allclose(m_na.theta, m_ref.theta, atol=1e-12, rtol=1e-12)


def test_gmm_na_action_fail_raises_on_na():
    """``na_action='na.fail'`` errors if any referenced-column row has NA."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    x_arr = df["x"].to_numpy().astype(float)
    x_arr[0] = np.nan
    df_na = df.with_columns(pl.Series("x", x_arr))

    with pytest.raises(ValueError, match=r"missing values in object"):
        gmm("y ~ x + (1|g)", df_na, family=Poisson(), na_action="na.fail")


def test_gmm_na_action_pass_raises_not_implemented():
    """``na_action='na.pass'`` is not implemented (it would carry NA rows
    through PIRLS); ``'na.exclude'`` IS implemented now (pads fitted/residuals
    back) and fits cleanly when there are no missing rows."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    with pytest.raises(NotImplementedError, match=r"na.pass"):
        gmm("y ~ x + (1|g)", df, family=Poisson(), na_action="na.pass")
    m = gmm("y ~ x + (1|g)", df, family=Poisson(), na_action="na.exclude")
    assert m.n == df.height                     # no NA rows → fits normally


def test_glmer_summary_prints_signif_codes_legend(capsys):
    """GLMM ``summary()`` appends R's ``Signif. codes:`` legend with the
    five-band thresholds. Match lme4's ``printCoefmat`` output."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=Poisson())
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


def test_gmm_contrasts_arg_switches_to_contr_sum():
    """contrasts={'x': 'contr.sum'} replaces the default contr.treatment
    coding on factor x. Column names switch from ``xb, xc`` (treatment,
    drop first level) to ``x1, x2`` (sum-to-zero, drop last level)."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m_def = gmm("y ~ x + (1|g)", df, family=Poisson())
    m_sum = gmm("y ~ x + (1|g)", df, family=Poisson(),
                contrasts={"x": "contr.sum"})
    assert m_def.column_names == ["(Intercept)", "xb", "xc"]
    assert m_sum.column_names == ["(Intercept)", "x1", "x2"]


def test_gmm_contrasts_arg_helmert():
    """contrasts={'x': 'contr.helmert'} → contrast columns x1, x2."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m = gmm("y ~ x + (1|g)", df, family=Poisson(),
            contrasts={"x": "contr.helmert"})
    assert m.column_names == ["(Intercept)", "x1", "x2"]


def test_gmm_contrasts_arg_rejects_unknown_name():
    """Unknown contrast names raise with a clear message listing the
    supported set (mirrors R's ``no contrasts function 'contr.foo'``)."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _three_level_glmm_df()
    with pytest.raises(ValueError, match=r"contrasts\['x'\]"):
        gmm("y ~ x + (1|g)", df, family=Poisson(),
            contrasts={"x": "contr.bogus"})


def test_gmm_contrasts_arg_rejects_non_string_value():
    """Numeric matrices and function references aren't yet supported."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _three_level_glmm_df()
    with pytest.raises(ValueError, match=r"only string names"):
        gmm("y ~ x + (1|g)", df, family=Poisson(),
            contrasts={"x": np.eye(3)})


def test_gmm_contrasts_arg_loses_to_inline_C():
    """In-formula ``C(x, contr.sum)`` overrides ``contrasts={x: contr.treatment}``
    (matches R: per-term ``C(...)`` always wins). Column names reflect C()."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m = gmm("y ~ C(x, contr.sum) + (1|g)", df, family=Poisson(),
            contrasts={"x": "contr.treatment"})
    # The C(x, contr.sum) atom produces sum-coded columns regardless of the
    # contrasts= argument.
    assert any(c.endswith("1") for c in m.column_names), m.column_names
    assert any(c.endswith("2") for c in m.column_names), m.column_names


def test_gmm_contrasts_arg_unrelated_column_unaffected():
    """A contrasts entry for a non-existent column is silently ignored,
    matching R's behavior. The fit proceeds as if no override was given."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _three_level_glmm_df()
    m_def = gmm("y ~ x + (1|g)", df, family=Poisson())
    m_nop = gmm("y ~ x + (1|g)", df, family=Poisson(),
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


def test_gmm_optinfo_singular_check_fires_at_boundary():
    """When a variance component shrinks to its lower bound (θ ≈ 0), the
    optinfo singular flag turns on and the standard lme4 message lands in
    ``optinfo$conv$lme4$messages``."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    # No within-group signal → θ̂ pinned at 0
    rng = np.random.default_rng(1)
    n_groups, n_per = 3, 100
    g = np.repeat(np.arange(n_groups), n_per)
    y = rng.poisson(2.0, size=n_groups * n_per)
    df = pl.DataFrame({"y": y, "g": g})

    m = gmm("y ~ 1 + (1|g)", df, family=Poisson())
    assert m.optinfo["is_singular"] is True
    assert m.theta[0] < 1e-4
    msgs = m.optinfo["conv"]["lme4"]["messages"]
    assert any("singular" in s for s in msgs), msgs


def test_gmm_optinfo_singular_check_silent_for_normal_fit():
    """A well-identified RE → ``is_singular=False``, empty messages."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=Poisson())
    assert m.optinfo["is_singular"] is False
    assert m.optinfo["conv"]["lme4"]["messages"] == []


def test_gmm_summary_prints_singular_warning(capsys):
    """The singular message is appended to summary() output (mirrors R's
    ``print.summary.merMod`` convergence block)."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    rng = np.random.default_rng(1)
    n_groups, n_per = 3, 100
    g = np.repeat(np.arange(n_groups), n_per)
    y = rng.poisson(2.0, size=n_groups * n_per)
    df = pl.DataFrame({"y": y, "g": g})

    m = gmm("y ~ 1 + (1|g)", df, family=Poisson())
    m.summary()
    out = capsys.readouterr().out
    assert "boundary (singular) fit" in out
    assert "see help('isSingular')" in out


def test_gmm_summary_omits_convergence_block_when_clean(capsys):
    """A clean fit has no convergence block in summary()."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=Poisson())
    m.summary()
    out = capsys.readouterr().out
    assert "boundary" not in out
    assert "isSingular" not in out
    assert "failed to converge" not in out


def test_gmm_checkconv_gradient_diagnostic(capsys):
    """8.14 — the scaled-gradient convergence check (lme4 checkConv): a clean
    fit carries no message and attaches its (θ, β) derivatives; a deliberately
    under-converged fit (loose Nelder-Mead, tiny maxfun) trips "Model failed
    to converge with max|grad|" and summary() prints it; action="ignore"
    suppresses the check."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)

    clean = gmm("y ~ x + (1|g)", df, family=Poisson())
    assert clean.optinfo["conv"]["lme4"]["code"] == 0
    assert clean.optinfo["conv"]["lme4"]["messages"] == []
    # derivatives are attached (calc.derivs resolves on for this small fit).
    assert clean.optinfo["derivs"] is not None
    assert clean.optinfo["derivs"]["gradient"].shape == (3,)   # θ + (intercept, x)

    under = gmm("y ~ x + (1|g)", df, family=Poisson(),
                control={"optimizer": ["Nelder_Mead", "Nelder_Mead"],
                         "optCtrl": {"XtolRel": 5e-2, "FtolAbs": 5e-2,
                                     "maxfun": 40}})
    msgs = under.optinfo["conv"]["lme4"]["messages"]
    assert any("failed to converge with max|grad|" in m for m in msgs), msgs
    assert under.optinfo["conv"]["lme4"]["code"] == -1
    under.summary()
    assert "failed to converge with max|grad|" in capsys.readouterr().out

    # action="ignore" → the gradient check is skipped even when under-converged.
    ignored = gmm("y ~ x + (1|g)", df, family=Poisson(),
                  control={"optimizer": ["Nelder_Mead", "Nelder_Mead"],
                           "optCtrl": {"XtolRel": 5e-2, "FtolAbs": 5e-2,
                                       "maxfun": 40},
                           "check.conv.grad": {"action": "ignore",
                                               "tol": 2e-3, "relTol": None}})
    assert not any("max|grad|" in m
                   for m in ignored.optinfo["conv"]["lme4"]["messages"])


def test_lmer_summary_omits_signif_codes_legend(capsys):
    """LMM ``summary()`` skips both the p-value column AND the legend —
    lme4's deliberate choice (see ``?lme4::pvalues``)."""
    from hea.models.gmm import gmm

    rng = np.random.default_rng(2026)
    n = 60
    g = np.repeat(np.arange(10), 6)
    x = rng.normal(size=n)
    u = rng.normal(scale=0.5, size=10)[g]
    y = 1.0 + 2.0 * x + u + rng.normal(scale=0.3, size=n)
    df = pl.DataFrame({"y": y, "x": x, "g": g})
    m = gmm("y ~ x + (1|g)", df)
    m.summary()
    out = capsys.readouterr().out
    assert "Signif. codes" not in out
    assert "Pr(>|t|)" not in out
    assert "Pr(>|z|)" not in out
    # t value column IS present though.
    assert "t value" in out


def test_gmm_subset_and_na_action_compose():
    """subset= filters first, then na_action policy applies to the result.
    Verifies the order of operations matches R's model.frame semantics."""
    from hea.models.gmm import gmm
    from hea.family import Poisson

    df = _synthetic_poisson_grouped(seed=2026)
    # Inject NAs in the FIRST row only.
    x_arr = df["x"].to_numpy().astype(float)
    x_arr[0] = np.nan
    df_na = df.with_columns(pl.Series("x", x_arr))

    # subset=  drops the first 5 rows → no NA remains → na.fail must NOT raise
    mask = np.arange(df_na.height) >= 5
    m = gmm("y ~ x + (1|g)", df_na, family=Poisson(),
            subset=mask, na_action="na.fail")
    # Sanity: produces same fit as pre-filtering then dropping the NA-row.
    m_ref = gmm("y ~ x + (1|g)", df_na.filter(pl.Series(mask)),
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
    from hea.R import factor
    from hea import data
    from hea.models.gmm import gmm
    from hea.family import Binomial

    contra = data("Contraception").mutate(
        factor("woman"),  factor("district"), factor("use"),
        factor("urban"),  factor("livch"),
    )
    m = gmm(
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
    # maps to a visible SE shift. The reference values below are from
    # R-Intel-MKL; observed cross-platform drift against that reference:
    #
    #   - hea-arm64-Accelerate (M4 dev):    poly1 SE diff ≈ 1.2e-2 (0.4% rel)
    #   - hea-x86_64-OpenBLAS (Linux CI):   poly1 SE diff ≈ 2.4e-2 (0.7% rel)
    #
    # R itself drifts ~1.0e-2 between Intel-MKL and arm64-Accelerate builds
    # on the same fit (verified). Well-identified columns (Intercept, urban,
    # livch) have H_jj of 100-300 and stay below 1e-3 rel comfortably. The
    # tolerance below is set to cover the BLAS-noise envelope without
    # masking real algorithmic bugs (which would shift SEs by ≫ 1e-1).
    expected_se = np.array([
        0.1522134608573170178047, 3.2936286686357942876668, 2.6142087304558478955130,
        0.1208624239243845793768, 0.1632291674042204154826, 0.1864493856133159488397,
        0.1875238509232133865545,
    ])
    np.testing.assert_allclose(m._se_beta, expected_se, atol=3e-2, rtol=1e-2)


# ----------------------------------------------------------------------
# Canonical Bates lme4 vignette sweep — Contraception cm1–cm4.
# Reference: lme4 ``vignettes/glmer.Rnw`` §"Contraception" (cm1..cm6 built by
# ``update()``). Transforms: ``age_s = age / (2*sd(age))``; ``ch = factor(
# livch != 0, labels=c("N","Y"))``. Fits use lme4's DEFAULT optimizer chain
# ``c("bobyqa","Nelder_Mead")``. Reference values were generated against
# byte-identical data — R is never run in CI.
#
# Tolerances follow the fm10 precedent: the Laplace objective (deviance /
# AIC / BIC / logLik) is curvature-independent and pins tight (rel 1e-7);
# θ̂/β̂ sit on a flat, ill-conditioned surface (age_s / I(age_s^2) / livch
# correlated 0.5–0.76) where lme4's gradient-free optimiser only resolves
# ~1e-5, so they pin loose (θ̂ abs 1e-4; β̂ in SE units < 3e-3). The parity
# investigation confirms this is an eval-noise floor, not a
# tolerance bug: tightening both sides closes θ̂ to ~5e-7 but leaves the β̂
# floor on the correlated columns.
# ----------------------------------------------------------------------

_CM_REF = {
    "cm1": {
        "formula": "use ~ age_s + I(age_s^2) + urban + livch + (1|district)",
        "colnames": ["(Intercept)", "age_s", "I(age_s^2)", "urbanY",
                     "livch1", "livch2", "livch3+"],
        "theta": [0.47523606313475225],
        "beta": [-1.0350274405489304, 0.063726453536789948,
                 -1.4824808685473088, 0.69728511186961906,
                 0.81497673072220556, 0.91645948135201727,
                 0.91502720195197496],
        "se": [0.17575150734060407, 0.16725908267340714, 0.23701672177850333,
               0.12086082595584657, 0.16319404552565814, 0.1863459657207191,
               0.18731529204621494],
        "AIC": 2388.7287068547207, "BIC": 2433.2674722628267,
        "logLik": -1186.3643534273604, "dev_laplace": 2372.7287068547207,
        "dev_resid": 2289.7306592727905, "npar": 8, "df_resid": 1926,
        "pearson_q": [-1.8438164551717313, -0.75918383875135065,
                      -0.46400094058664731, 0.94930520663395612,
                      3.0714958908257421],
    },
    "cm2": {
        "formula": "use ~ age_s + I(age_s^2) + urban + ch + (1|district)",
        "colnames": ["(Intercept)", "age_s", "I(age_s^2)", "urbanY", "chY"],
        "theta": [0.47399234899802645],
        "beta": [-1.0063736786246373, 0.11277569344137259,
                 -1.5062536618932687, 0.69292195041328508,
                 0.86038209211396821],
        "se": [0.16911401578167221, 0.14213752282172851, 0.23418953619202565,
               0.1206581477445645, 0.14830391129817388],
        "AIC": 2385.1858190738521, "BIC": 2418.5898931299316,
        "logLik": -1186.5929095369261, "dev_laplace": 2373.1858190738521,
        "dev_resid": 2290.3956618293355, "npar": 6, "df_resid": 1928,
        "pearson_q": [-1.8150497388800013, -0.76198051917562681,
                      -0.46193204188532933, 0.9518062700522788,
                      3.1033486907613974],
    },
    "cm3": {
        "formula": ("use ~ age_s + I(age_s^2) + urban + ch + age_s:ch "
                    "+ (1|district)"),
        "colnames": ["(Intercept)", "age_s", "I(age_s^2)", "urbanY", "chY",
                     "age_s:chY"],
        "theta": [0.47226931870180711],
        "beta": [-1.32329838287304, -0.85255290282845397,
                 -1.8707276887929793, 0.7140073373117648,
                 1.2107566086549038, 1.2321877016246101],
        "se": [0.21505379700145275, 0.39318304454754804, 0.27312648602724332,
               0.12125989955644181, 0.20733477577072387, 0.45856006976258906],
        "AIC": 2379.1813079332615, "BIC": 2418.1527276653542,
        "logLik": -1182.5906539666307, "dev_laplace": 2365.1813079332615,
        "dev_resid": 2282.9246322145832, "npar": 7, "df_resid": 1927,
        "pearson_q": [-1.8720410335459285, -0.75602285608215181,
                      -0.46679093243452663, 0.9485588267712094,
                      2.9973591301444071],
    },
    "cm4": {
        "formula": ("use ~ age_s + I(age_s^2) + urban + ch + age_s:ch "
                    "+ (1+urban|district)"),
        "colnames": ["(Intercept)", "age_s", "I(age_s^2)", "urbanY", "chY",
                     "age_s:chY"],
        "theta": [0.61496553579326108, -0.57509470937327523,
                  0.44190385654424946],
        "beta": [-1.3441349102161679, -0.8324500403806161,
                 -1.8362230871367438, 0.79013260776466188,
                 1.2115315825963644, 1.1981322762954401],
        "se": [0.223614513550251, 0.39491862178620152, 0.27608314721759164,
               0.16332396584891001, 0.2087001594039371, 0.46105123099560541],
        "AIC": 2371.5304045633152, "BIC": 2421.6365156474344,
        "logLik": -1176.7652022816576, "dev_laplace": 2353.5304045633152,
        "dev_resid": 2235.2914451677857, "npar": 9, "df_resid": 1925,
        "pearson_q": [-1.9336610694581922, -0.73354163593473076,
                      -0.44574628672232086, 0.89726774208216042,
                      3.0327222434958871],
    },
}


@pytest.fixture(scope="module")
def cm_frame():
    """Contraception with the vignette transforms (age_s, ch)."""
    from hea import data
    from hea.R import factor
    c = data("Contraception")
    sd_age = c["age"].std()
    return c.with_columns(
        (pl.col("age") / (2 * sd_age)).alias("age_s"),
        pl.when(pl.col("livch") == "0").then(pl.lit("N"))
            .otherwise(pl.lit("Y")).alias("ch"),
    ).mutate(
        factor("district"), factor("use"), factor("urban"),
        factor("livch"), factor("ch"),
    )


@pytest.mark.parametrize("tag", ["cm1", "cm2", "cm3", "cm4"])
def test_glmer_contraception_cm_components_match_lme4(cm_frame, tag):
    from hea.models.gmm import gmm
    ref = _CM_REF[tag]
    m = gmm(ref["formula"], data=cm_frame, family=Binomial())

    # Fixed-effect names line up with R's model.matrix expansion.
    assert m.column_names == ref["colnames"]
    assert m.npar == ref["npar"]
    assert m.df_resid == ref["df_resid"]

    # Laplace objective — curvature-independent, pins tight across platforms.
    assert m.deviance_laplace == pytest.approx(ref["dev_laplace"], rel=1e-7)
    assert m.AIC == pytest.approx(ref["AIC"], rel=1e-7)
    assert m.BIC == pytest.approx(ref["BIC"], rel=1e-7)
    assert m.loglike == pytest.approx(ref["logLik"], rel=1e-7)
    # Residual deviance depends on β̂ → inherits the flat-surface drift.
    assert m.deviance == pytest.approx(ref["dev_resid"], rel=1e-5)

    # θ̂ (variance components) — loose, the gradient-free optimiser floor.
    np.testing.assert_allclose(m.theta, ref["theta"], atol=1e-4, rtol=1e-4)

    # β̂ in SE-relative units (fm10 convention): |Δβ̂|/SE stays small even
    # where the raw β̂ is poorly identified.
    beta_se_rel = np.abs(m.bhat.to_numpy().ravel()
                         - np.array(ref["beta"])) / np.array(ref["se"])
    assert beta_se_rel.max() < 3e-3, f"{tag}: |Δβ̂|/SE = {beta_se_rel}"

    # SE magnitudes — loose (FD-Hessian cancellation + cross-platform BLAS
    # noise; see fm10). Catches gross errors, not sub-ULP drift.
    np.testing.assert_allclose(m._se_beta, ref["se"], atol=5e-2, rtol=3e-2)

    # Scaled (Pearson, σ-divided) residual 5-number summary — what summary()
    # prints. Drifts with the fit, hence the loose absolute tolerance.
    pearson_scaled = m.residuals_of("pearson") / m.sigma
    np.testing.assert_allclose(
        np.quantile(pearson_scaled, [0, .25, .5, .75, 1]),
        ref["pearson_q"], atol=5e-3, rtol=5e-3,
    )


def test_glmer_contraception_cm4_summary_layout_matches_lme4(cm_frame, capsys):
    """cm4 (random slope + correlation) exercises the full summary layout:
    the glmerMod header tag, the 1-decimal criterion table, the
    Variance/Std.Dev./Corr random-effects block, the z+Pr fixed-effects
    table, and the fixed-effect correlation matrix — each matching a section
    of lme4's print.summary.merMod."""
    from hea.models.gmm import gmm
    m = gmm(_CM_REF["cm4"]["formula"], data=cm_frame, family=Binomial())
    m.summary()
    out = capsys.readouterr().out

    assert ("Generalized linear mixed model fit by maximum likelihood "
            "(Laplace Approximation) ['glmerMod']") in out
    assert "Family: binomial  ( logit )" in out
    # 1-decimal criterion row (lme4's .prt.aictab digits=1), not 4-decimal.
    assert "2371.5" in out and "2353.5" in out
    assert "2371.5304" not in out
    # Random-effects block with the correlated random slope.
    assert "Groups   Name        Variance Std.Dev. Corr" in out
    assert "district (Intercept)" in out
    assert "urbanY" in out
    assert "-0.79" in out
    assert "Number of obs: 1934, groups:  district, 60" in out
    # GLMM fixed-effects table uses z + Pr(>|z|), with the signif legend.
    assert "z value" in out and "Pr(>|z|)" in out
    assert "Signif. codes:" in out
    assert "Correlation of Fixed Effects:" in out


def test_glmer_contraception_anova_cm1_to_cm4(cm_frame, capsys):
    """anova(cm1..cm4): rows sort ascending by npar (cm2<cm3<cm1<cm4). The
    cm1 row has MORE params than cm3 yet a worse deviance, so its LRT is
    negative — lme4 clamps it to 0 (p=1). Pin the printed table against
    lme4's anova.merMod."""
    from hea.models.gmm import gmm
    from hea.R.model_selection import anova
    cm1 = gmm(_CM_REF["cm1"]["formula"], data=cm_frame, family=Binomial())
    cm2 = gmm(_CM_REF["cm2"]["formula"], data=cm_frame, family=Binomial())
    cm3 = gmm(_CM_REF["cm3"]["formula"], data=cm_frame, family=Binomial())
    cm4 = gmm(_CM_REF["cm4"]["formula"], data=cm_frame, family=Binomial())
    anova(cm1, cm2, cm3, cm4)
    out = capsys.readouterr().out

    # Caller names recovered (R-style), rows printed in npar order.
    for name in ("cm1", "cm2", "cm3", "cm4"):
        assert name in out
    # The cm1 row's negative LRT is clamped to 0 (was -7.5474), p=1.
    assert "0.0000" in out
    assert "1.000000" in out
    assert "-7.5" not in out                # no spurious negative χ²
    # The two genuine LRTs match lme4 (cm2→cm3, cm1→cm4).
    assert "8.0045" in out
    assert "19.1983" in out
    # Underlying LRT is exactly the Laplace-deviance gap (format-independent).
    assert (cm2.deviance_laplace - cm3.deviance_laplace) == pytest.approx(
        8.0045111405906937, rel=1e-5)
    assert (cm1.deviance_laplace - cm4.deviance_laplace) == pytest.approx(
        19.198302291405525, rel=1e-5)


def _parse_drop1(out):
    """Parse a printed drop1 table into ``{label: {npar, AIC, LRT}}``."""
    lines = out.splitlines()
    hi = next(i for i, ln in enumerate(lines)
              if "npar" in ln and "AIC" in ln)
    rows = {}
    for ln in lines[hi + 1:]:
        if not ln.strip() or ln.lstrip().startswith("---"):
            break
        p = ln.split()
        if p[0] == "<none>":
            rows[p[0]] = {"AIC": float(p[1])}
        else:
            row = {"npar": int(p[1]), "AIC": float(p[2])}
            if len(p) > 3:                      # test=Chisq adds LRT/Pr
                row["LRT"] = float(p[3])
            rows[p[0]] = row
    return rows


# Reference: drop1(cm{1,3}, test="Chisq") in lme4.
# LRT = Δ(-2logL); npar = #coefficients removed (livch = 3 dummies).
_DROP1_REF = {
    "cm1": {
        "age_s":      {"npar": 1, "AIC": 2386.9, "LRT": 0.14469280800858542},
        "I(age_s^2)": {"npar": 1, "AIC": 2427.6, "LRT": 40.887073899940333},
        "urban":      {"npar": 1, "AIC": 2419.7, "LRT": 33.020369708456656},
        "livch":      {"npar": 3, "AIC": 2417.9, "LRT": 35.214906898611389},
    },
    "cm3": {  # marginality: bare age_s / ch are NOT droppable (inside age_s:ch)
        "I(age_s^2)": {"npar": 1, "AIC": 2428.3, "LRT": 51.136395895061469},
        "urban":      {"npar": 1, "AIC": 2411.6, "LRT": 34.41381832595971},
        "age_s:ch":   {"npar": 1, "AIC": 2385.2, "LRT": 8.0045111405906937},
    },
}


@pytest.mark.parametrize("tag", ["cm1", "cm3"])
def test_glmer_contraception_drop1_matches_lme4(cm_frame, capsys, tag):
    """drop1(gmm, test="Chisq") matches lme4's drop1.merMod: single fixed-term
    deletions (bars preserved), marginality-respecting term selection, Δnpar
    column (3 for the 4-level livch), and the Laplace LRT / AIC per row."""
    from hea.models.gmm import gmm
    from hea.R.model_selection import drop1
    m = gmm(_CM_REF[tag]["formula"], data=cm_frame, family=Binomial())
    drop1(m, test="Chisq")
    rows = _parse_drop1(capsys.readouterr().out)

    ref = _DROP1_REF[tag]
    # Exact term selection (incl. marginality): the dropped rows are exactly
    # the droppable terms — cm3 excludes the bare age_s / ch main effects.
    assert set(rows) == {"<none>"} | set(ref)
    for term, exp in ref.items():
        assert rows[term]["npar"] == exp["npar"], term
        # AIC printed at 1 decimal; LRT pinned loosely (each row is a separate
        # default-optimizer refit → the flat-surface drift, see cm-sweep note).
        assert rows[term]["AIC"] == pytest.approx(exp["AIC"], abs=0.05), term
        assert rows[term]["LRT"] == pytest.approx(exp["LRT"], abs=5e-3), term
    # The <none> row carries the full-model AIC.
    assert rows["<none>"]["AIC"] == pytest.approx(
        _CM_REF[tag]["AIC"], abs=0.05)


def test_glmer_drop1_no_test_emits_npar_aic_only(cm_frame, capsys):
    """drop1(gmm) without a test prints just npar + AIC (lme4 default
    test="none") — no LRT / Pr(Chi) columns, no signif legend."""
    from hea.models.gmm import gmm
    from hea.R.model_selection import drop1
    m = gmm(_CM_REF["cm1"]["formula"], data=cm_frame, family=Binomial())
    drop1(m)
    out = capsys.readouterr().out
    assert "npar" in out and "AIC" in out
    assert "LRT" not in out and "Pr(Chi)" not in out
    assert "Signif. codes" not in out
    rows = _parse_drop1(out)
    assert rows["livch"]["npar"] == 3       # multi-level factor Δnpar


def test_gmm_refitML_lmm_refits_by_ml():
    """refitML(REML LMM) → an ML fit (lme4 refitML.merMod); an already-ML LMM
    and any GLMM (ML by construction) are returned unchanged."""
    from hea.models.gmm import gmm
    from hea.family import Poisson
    from hea.R import refitML

    rng = np.random.default_rng(2026)
    n = 120
    g = np.repeat(np.arange(12), 10)
    x = rng.normal(size=n)
    u = rng.normal(scale=0.5, size=12)[g]
    y = 1.0 + 2.0 * x + u + rng.normal(scale=0.3, size=n)
    df = pl.DataFrame({"y": y, "x": x, "g": g})

    m_reml = gmm("y ~ x + (1|g)", df)                 # REML=True (default)
    m_ml = refitML(m_reml)
    assert m_reml.REML is True and m_ml.REML is False
    ref = gmm("y ~ x + (1|g)", df, REML=False)
    np.testing.assert_allclose(m_ml.theta, ref.theta, atol=1e-9)
    assert m_ml.AIC == pytest.approx(ref.AIC, rel=1e-9)
    # already-ML LMM and any GLMM are no-ops (same object back).
    assert refitML(ref) is ref
    pois = gmm("y ~ x + (1|g)", _synthetic_poisson_grouped(seed=7),
               family=Poisson())
    assert pois.REML is False and refitML(pois) is pois


def test_gmm_refit_newresp_and_validation():
    """refit() re-fits; refit(newresp=) swaps the response (idempotent on the
    original y, different on a perturbed one); bad type / length / non-gmm
    raise."""
    from hea.models.gmm import gmm
    from hea.family import Poisson
    from hea.models.lm import lm
    from hea.R import refit

    df = _synthetic_poisson_grouped(seed=2026)
    m = gmm("y ~ x + (1|g)", df, family=Poisson())
    y0 = df["y"].to_numpy().astype(float)
    # refit with the original response reproduces the fit; no-newresp too.
    np.testing.assert_allclose(refit(m, y0).theta, m.theta, atol=1e-8)
    np.testing.assert_allclose(refit(m).theta, m.theta, atol=1e-8)
    # a different response gives a different fit.
    assert not np.allclose(refit(m, y0 + 2.0).theta, m.theta)
    # guards
    with pytest.raises(TypeError):
        refit(lm("y ~ x", df))
    with pytest.raises(ValueError, match="length"):
        refit(m, y0[:-1])


def test_deriv12_uses_supplied_fx_to_save_one_eval():
    """``fx`` argument skips the redundant ``fn(x)`` call. Pin: same answer."""
    def py_fn(x):
        return float(x[0] ** 3 + x[1] ** 2)

    x0 = np.array([0.2, 0.4])
    g_a, H_a = _deriv12(py_fn, x0)
    g_b, H_b = _deriv12(py_fn, x0, fx=py_fn(x0))
    np.testing.assert_array_equal(g_a, g_b)
    np.testing.assert_array_equal(H_a, H_b)

