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
    "theta":      np.array([0.70455609788924856]),
    "beta":       np.array([0.66323953341179220, 0.14851913323415467]),
    "dev_stage0": 271.77399765135942,
    "dev_stage1": 271.74520477038624,
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
    "theta":    np.array([0.70455662201657199]),
    "beta":     np.array([0.66324568584127941, 0.14851170395298091]),
    "laplace":  271.74519961363023,
    "deviance":  78.155739442738806,
    "aic":      277.74519961363023,
    "bic":      284.5751979706784,
    "sigma":      1.0,    # scale-known for Poisson
}


def test_glmer_poisson_full_fit_matches_lme4():
    """Full ``hea.lme(..., family=poisson())`` fit ≡ ``lme4::glmer(..., family=poisson)``.

    Compared to ``lme4::glmer`` configured with
    ``optimizer=c("Nelder_Mead", "Nelder_Mead")`` so both stages use the
    same algorithm hea has ported from lme4 (see ``hea._nelder_mead``).
    Tolerance ≤ 1e-7 on θ̂/β̂ — anything looser would mask actual bugs.
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

    R recipe::
        suppressMessages(library(lme4)); data(cbpp)
        m <- glmer(cbind(incidence, size-incidence) ~ period + (1|herd),
                   data=cbpp, family=binomial(),
                   control=glmerControl(
                       optimizer=c("Nelder_Mead","Nelder_Mead")))
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

    theta_r = np.array([0.64206385678459621])
    beta_r  = np.array([
        -1.3983383581638527, -0.99192429647839764,
        -1.1281979544471477, -1.5797336556545503,
    ])
    laplace_r = 184.0531328091428
    dev_r     =  73.474376562348297

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
    theta_r   = np.array([0.50991210937499964])
    laplace_r = 112.41792709835545

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
#     vcov(m, use.hessian=FALSE)   # RX-based; Phase 8 will pin Hessian-based
#     VarCorr(m)$g                 # per-bar SD
#     ranef(m)$g                   # BLUPs
_GLMER_PHASE6_POISSON_REF = {
    "theta": np.array([0.70455662201657199]),
    "beta":  np.array([0.66324568584127941, 0.14851170395298091]),
    "eta": np.array([
        1.7729347165001661e+00, 1.9264503379702644e+00, 1.6090960292460870e+00,
        2.0980111217169899e+00, 1.9855169263308092e+00, 1.8473502170732887e+00,
        -1.4648045824321387e-01, -5.5029223772085523e-02, -1.3990301952553597e-01,
        -1.3370244188013369e-01, 6.7861669446641626e-03, -2.3712584818416538e-02,
        7.5727540730254739e-01, 7.5410488624655425e-01, 7.9069711357197059e-01,
        6.7561023826827715e-01, 7.0683751651188176e-01, 8.4822220986501617e-01,
        3.7268263570884946e-01, 1.8794248429333510e-01, 3.2117938174038180e-01,
        4.8957685110411947e-01, 3.5756414736518177e-01, 3.6997229457215747e-01,
        5.7851790841970629e-01, 7.5417364255200647e-01, 3.7728079220245586e-01,
        6.8342214818593616e-01, 3.0052636400710381e-01, 5.0918388200710707e-01,
        4.2758174037888841e-02, 4.1715435375195364e-01, 3.4029432108189372e-01,
        3.8541125423265798e-01, 8.4934958179896269e-02, 3.1811142916298196e-01,
        1.2075159788084968e+00, 1.2166684608620288e+00, 1.3598222175684658e+00,
        1.4147984424759237e+00, 1.3149543273028947e+00, 1.1913320983614026e+00,
        6.4854382573565850e-02, 4.0199578382450923e-01, 2.7570572049890935e-01,
        2.9438582346227399e-01, 5.1177134592300733e-01, 6.6332604556258057e-02,
        2.0964110029231371e+00, 2.1842483001260602e+00, 1.9566548542974731e+00,
        1.8391365645049689e+00, 1.6177162345028149e+00, 1.8639943192877406e+00,
        -3.2682967536433627e-02, 2.9846602169373737e-02, -1.0070728207137947e-02,
        7.5214068325729855e-02, 2.2435661413152652e-01, -4.9448737274826127e-02,
        1.8798647378427574e-01, 1.8537968966568491e-01, 1.9052996043250131e-01,
        1.2179493350797582e-01, 1.9874908113387407e-01, 5.2639555663848459e-01,
        1.0455418362129891e+00, 1.1718391039601570e+00, 9.5293261116530703e-01,
        1.1588507042836282e+00, 8.7512918838533871e-01, 1.0298821786011483e+00,
    ]),
    "mu": np.array([
        5.8881079579030091e+00, 6.8650981616228082e+00, 4.9982908762369211e+00,
        8.1499445350721160e+00, 7.2828110894051896e+00, 6.3429896889641366e+00,
        8.6374261121787443e-01, 9.4645748849139666e-01, 8.6944255026133566e-01,
        8.7485034470091894e-01, 1.0068092451502142e+00, 9.7656634942057463e-01,
        2.1324582181637668e+00, 2.1257079211161276e+00, 2.2049329793228836e+00,
        1.9652318698201909e+00, 2.0275689554428378e+00, 2.3354911459143888e+00,
        1.4516235732348732e+00, 1.2067641054632416e+00, 1.3787528818039514e+00,
        1.6316256532657472e+00, 1.4298422841214598e+00, 1.4476945051120516e+00,
        1.7833933188629929e+00, 2.1258540819639364e+00, 1.4583137350122979e+00,
        1.9806442056648859e+00, 1.3505695116951082e+00, 1.6639326752698638e+00,
        1.0436854740869832e+00, 1.5176367477832289e+00, 1.4053611571161446e+00,
        1.4702188308462174e+00, 1.0886462568616280e+00, 1.3745294153567318e+00,
        3.3451648631513060e+00, 3.3759219617191669e+00, 3.8955006886453432e+00,
        4.1156568407646947e+00, 3.7245808697125216e+00, 3.2914628406085620e+00,
        1.0670036387471686e+00, 1.4948050303024245e+00, 1.3174601054728592e+00,
        1.3423016952599571e+00, 1.6682436158220388e+00, 1.0685820733325033e+00,
        8.1369140835681026e+00, 8.8839679651877805e+00, 7.0756184582143753e+00,
        6.2911039528211781e+00, 5.0415634106297320e+00, 6.4494465394433691e+00,
        9.6784534935076139e-01, 1.0302964765865879e+00, 9.8997981177574323e-01,
        1.0781149164390589e+00, 1.2515172485945181e+00, 9.5175394638692890e-01,
        1.2068171915695332e+00, 1.2036753764764887e+00, 1.2098906219407781e+00,
        1.1295224508225532e+00, 1.2198758376328065e+00, 1.6928196263635760e+00,
        2.8449395974069240e+00, 3.2279236701697918e+00, 2.5933036700307990e+00,
        3.1862692054201269e+00, 2.3991852208119977e+00, 2.8007358286454282e+00,
    ]),
    "res_dev": np.array([
        4.4482802472200211e-01, -3.3749947902049326e-01, 7.6443049126338942e-04,
        -1.6143590013895415e+00, 2.6156457022082796e-01, 1.9968309300192022e+00,
        1.0421112117468010e+00, -1.3758324669024180e+00, -1.3186679265541690e+00,
        1.3078827608813151e-01, -1.4190202571846635e+00, 2.3619239572480594e-02,
        5.5940837008477839e-01, -8.7092017348430389e-02, 5.0730366569008900e-01,
        2.4728704111923423e-02, -1.9405336326110117e-02, -9.8718684761231756e-01,
        -1.7038917648928720e+00, -1.9401866492637515e-01, 1.1925196875415813e+00,
        -1.8064471502182107e+00, -3.8020556744024142e-01, 1.1257126519675509e+00,
        8.2906749979504402e-01, -8.6218378483004421e-01, -4.0257407469891043e-01,
        -7.7100202007381236e-01, -3.1636418156297164e-01, 9.2975039150609851e-01,
        -1.4447736667637483e+00, 3.7315519934529162e-01, -1.6765208958531621e+00,
        1.1043748767086048e+00, -1.4755651506196723e+00, 4.9920119821702841e-01,
        -1.9211347027288223e-01, -2.0858305366165089e-01, -4.7301353457226819e-01,
        1.2910761300716711e+00, 1.4100358137654076e-01, -1.6311598038624422e-01,
        2.1692226962358974e+00, -1.7290488890152438e+00, -2.8897883996566237e-01,
        -1.6384759352886189e+00, 2.4897905572925086e-01, -1.4619042877921271e+00,
        1.8622366429956230e+00, -3.0170812921757889e-01, -1.2602987625988098e+00,
        6.5351934970605963e-01, -1.8536466014823852e-02, -5.9441484701371250e-01,
        3.2505903685173067e-02, -1.4354765596042227e+00, 1.0053853279340471e-02,
        -7.6168866518139006e-02, -1.5820981313398472e+00, 9.3482167503377023e-01,
        -1.9406554452172822e-01, -1.9128871796738992e-01, -1.5555646061419488e+00,
        -1.5030119432809264e+00, 6.4602172793946955e-01, 9.0495284126019693e-01,
        9.1114815176077862e-02, 4.1411278369238103e-01, 1.8034770962822269e+00,
        4.3827806873074698e-01, -2.1905183043343865e+00, -1.2416550648584170e+00,
    ]),
    "res_pearson": np.array([
        4.5822073740210040e-01, -3.3017334788212488e-01, 7.6447405248529986e-04,
        -1.4536671144824396e+00, 2.6575659045370648e-01, 2.2461562691125203e+00,
        1.2226004398623096e+00, -9.7286046712331598e-01, -9.3243903299965714e-01,
        1.3380205208194978e-01, -1.0033988464963541e+00, 2.3713140375261392e-02,
        5.9408711716922802e-01, -8.6220561915479832e-02, 5.3543411997006074e-01,
        2.4801299441055320e-02, -1.9361209975036008e-02, -8.7388020216939988e-01,
        -1.2048334213636644e+00, -1.8821921375603518e-01, 1.3807212233404849e+00,
        -1.2773510297744106e+00, -3.5947199943791791e-01, 1.2901456185429510e+00,
        9.1101731103030181e-01, -7.7217436760623481e-01, -3.7952237886713752e-01,
        -6.9680014548556013e-01, -3.0165857345159569e-01, 1.0357631785814698e+00,
        -1.0216092570483997e+00, 3.9155276726657412e-01, -1.1854792942587165e+00,
        1.2616483387454451e+00, -1.0433821240857197e+00, 5.3349440709735330e-01,
        -1.8871995142227740e-01, -2.0459805411613002e-01, -4.5371619513451700e-01,
        1.4217634656765881e+00, 1.4271032605876219e-01, -1.6065295381045619e-01,
        2.8394130494508585e+00, -1.2226221944257452e+00, -2.7657977544421164e-01,
        -1.1585774446535533e+00, 2.5685590430320804e-01, -1.0337224353435033e+00,
        2.0554001936380186e+00, -2.9657397263497420e-01, -1.1562460325256696e+00,
        6.8132168921651792e-01, -1.8510943742109359e-02, -5.7074350522181316e-01,
        3.2684422188086489e-02, -1.0150352095304811e+00, 1.0070770764229024e-02,
        -7.5231798651285639e-02, -1.1187123171729709e+00, 1.0744863258499318e+00,
        -1.8826339766215183e-01, -1.8564524912075575e-01, -1.0999502815767530e+00,
        -1.0627899372983136e+00, 7.0632717368878950e-01, 1.0046855958775875e+00,
        9.1931507588512137e-02, 4.2973290562897748e-01, 2.1154693397227402e+00,
        4.5586843127282528e-01, -1.5489303473081020e+00, -1.0760041146539039e+00,
    ]),
    "res_working": np.array([
        1.8883689804033080e-01, -1.2601395366185292e-01, 3.4194163673115604e-04,
        -5.0919911383610350e-01, 9.8476934495548674e-02, 8.9185235802577834e-01,
        1.3155046121668192e+00, -1.0000000000000000e+00, -1.0000000000000000e+00,
        1.4305264444042187e-01, -1.0000000000000000e+00, 2.3995963605882221e-02,
        4.0682709487422580e-01, -5.9136967909553258e-02, 3.6058557250174328e-01,
        1.7691617316886978e-02, -1.3597049495570170e-02, -5.7182453817075762e-01,
        -1.0000000000000000e+00, -1.7133763303630159e-01, 1.1758794049262977e+00,
        -1.0000000000000000e+00, -3.0062216574156531e-01, 1.0722604039778405e+00,
        6.8218640737796299e-01, -5.2960082797584773e-01, -3.1427649895132681e-01,
        -4.9511376291618803e-01, -2.5957161675825646e-01, 8.0295756227844195e-01,
        -1.0000000000000000e+00, 3.1783841088544157e-01, -1.0000000000000000e+00,
        1.0405125666043080e+00, -1.0000000000000000e+00, 4.5504343352371235e-01,
        -1.0318321436215976e-01, -1.1135386599035335e-01, -2.2988076763933335e-01,
        7.0082207308114408e-01, 7.3946341862819162e-02, -8.8551156346845808e-02,
        2.7488157066611638e+00, -1.0000000000000000e+00, -2.4096373328808865e-01,
        -1.0000000000000000e+00, 1.9886566987668997e-01, -1.0000000000000000e+00,
        7.2055399088850725e-01, -9.9501480492911271e-02, -4.3467839262075586e-01,
        2.7163691142195889e-01, -8.2441511183016947e-03, -2.2473967813809745e-01,
        3.3222922103007695e-02, -1.0000000000000000e+00, 1.0121608648042416e-02,
        -7.2455092910751293e-02, -1.0000000000000000e+00, 1.1013834590257785e+00,
        -1.7137408467023563e-01, -1.6921121795538124e-01, -1.0000000000000000e+00,
        -1.0000000000000000e+00, 6.3951111932919347e-01, 7.7219117340010912e-01,
        5.4503934893524210e-02, 2.3918667500263299e-01, 1.3136511428793614e+00,
        2.5538670530275492e-01, -1.0000000000000000e+00, -6.4295097389329703e-01,
    ]),
    "res_response": np.array([
        1.1118920420969909e+00, -8.6509816162280817e-01, 1.7091237630788569e-03,
        -4.1499445350721160e+00, 7.1718891059481038e-01, 5.6570103110358634e+00,
        1.1362573887821257e+00, -9.4645748849139666e-01, -8.6944255026133566e-01,
        1.2514965529908106e-01, -1.0068092451502142e+00, 2.3433650579425369e-02,
        8.6754178183623321e-01, -1.2570792111612761e-01, 7.9506702067711643e-01,
        3.4768130179809065e-02, -2.7568955442837773e-02, -1.3354911459143888e+00,
        -1.4516235732348732e+00, -2.0676410546324164e-01, 1.6212471181960486e+00,
        -1.6316256532657472e+00, -4.2984228412145975e-01, 1.5523054948879484e+00,
        1.2166066811370071e+00, -1.1258540819639364e+00, -4.5831373501229788e-01,
        -9.8064420566488586e-01, -3.5056951169510819e-01, 1.3360673247301362e+00,
        -1.0436854740869832e+00, 4.8236325221677112e-01, -1.4053611571161446e+00,
        1.5297811691537826e+00, -1.0886462568616280e+00, 6.2547058464326821e-01,
        -3.4516486315130601e-01, -3.7592196171916692e-01, -8.9550068864534316e-01,
        2.8843431592353053e+00, 2.7541913028747844e-01, -2.9146284060856198e-01,
        2.9329963612528314e+00, -1.4948050303024245e+00, -3.1746010547285919e-01,
        -1.3423016952599571e+00, 3.3175638417796116e-01, -1.0685820733325033e+00,
        5.8630859164318974e+00, -8.8396796518778054e-01, -3.0756184582143753e+00,
        1.7088960471788219e+00, -4.1563410629732012e-02, -1.4494465394433691e+00,
        3.2154650649238614e-02, -1.0302964765865879e+00, 1.0020188224256765e-02,
        -7.8114916439058879e-02, -1.2515172485945181e+00, 1.0482460536130711e+00,
        -2.0681719156953315e-01, -2.0367537647648870e-01, -1.2098906219407781e+00,
        -1.1295224508225532e+00, 7.8012416236719351e-01, 1.3071803736364240e+00,
        1.5506040259307596e-01, 7.7207632983020824e-01, 3.4066963299692010e+00,
        8.1373079457987307e-01, -2.3991852208119977e+00, -1.8007358286454282e+00,
    ]),
    "working_wts": np.array([
        5.8881079579030091e+00, 6.8650981616228064e+00, 4.9982908762369203e+00,
        8.1499445350721160e+00, 7.2828110894051905e+00, 6.3429896889641375e+00,
        8.6374261121787443e-01, 9.4645748849139655e-01, 8.6944255026133566e-01,
        8.7485034470091894e-01, 1.0068092451502140e+00, 9.7656634942057452e-01,
        2.1324582181637664e+00, 2.1257079211161281e+00, 2.2049329793228836e+00,
        1.9652318698201909e+00, 2.0275689554428378e+00, 2.3354911459143892e+00,
        1.4516235732348735e+00, 1.2067641054632416e+00, 1.3787528818039514e+00,
        1.6316256532657472e+00, 1.4298422841214595e+00, 1.4476945051120513e+00,
        1.7833933188629927e+00, 2.1258540819639360e+00, 1.4583137350122979e+00,
        1.9806442056648856e+00, 1.3505695116951082e+00, 1.6639326752698635e+00,
        1.0436854740869832e+00, 1.5176367477832287e+00, 1.4053611571161444e+00,
        1.4702188308462174e+00, 1.0886462568616282e+00, 1.3745294153567318e+00,
        3.3451648631513056e+00, 3.3759219617191674e+00, 3.8955006886453436e+00,
        4.1156568407646947e+00, 3.7245808697125224e+00, 3.2914628406085624e+00,
        1.0670036387471686e+00, 1.4948050303024245e+00, 1.3174601054728592e+00,
        1.3423016952599574e+00, 1.6682436158220391e+00, 1.0685820733325033e+00,
        8.1369140835681009e+00, 8.8839679651877823e+00, 7.0756184582143753e+00,
        6.2911039528211790e+00, 5.0415634106297311e+00, 6.4494465394433691e+00,
        9.6784534935076127e-01, 1.0302964765865876e+00, 9.8997981177574301e-01,
        1.0781149164390589e+00, 1.2515172485945179e+00, 9.5175394638692878e-01,
        1.2068171915695334e+00, 1.2036753764764887e+00, 1.2098906219407781e+00,
        1.1295224508225534e+00, 1.2198758376328063e+00, 1.6928196263635760e+00,
        2.8449395974069240e+00, 3.2279236701697909e+00, 2.5933036700307981e+00,
        3.1862692054201265e+00, 2.3991852208119973e+00, 2.8007358286454287e+00,
    ]),
    "prior_wts": np.ones(72),
    "laplace":  271.74519961363023,
    "deviance":  78.155739442738806,
    "aic":      277.74519961363023,
    "bic":      284.5751979706784,
    "sigma":      1.0,
    "se_beta":  np.array([0.22221510757525684, 0.07701572680278147]),
    "t_value":  np.array([2.9847011442129796,  1.9283295778443179]),
    "vcov":     np.array([
        4.9379554034682971e-02, -1.1994196115711706e-03,
        -1.1994196115711706e-03,  5.9314221749606722e-03,
    ]).reshape(2, 2),
    "sd_re_g": np.array([0.70455662201657199]),
    "ranef_g": np.array([
         1.2274770008942444e+00, -7.6339801780652361e-01,  1.0355347045266639e-01,
        -2.7118482819260464e-01, -1.8004802873499109e-01, -4.4679868996487870e-01,
         6.2134984695012585e-01, -4.7574788459633205e-01,  1.0530558783051334e+00,
        -6.3018368798826496e-01, -3.2891520964726800e-01,  4.7040992174478846e-01,
    ]),
}


def test_glmer_phase6_attrs_match_lme4_poisson():
    """Every Phase 6 attribute on a Poisson fit matches lme4 at ≤ 1e-9."""
    from hea.lme import lme
    from hea.family import Poisson as PoissonFamily

    df = _synthetic_poisson_grouped(seed=2026)
    r = _GLMER_PHASE6_POISSON_REF

    m = lme("y ~ x + (1|g)", df, family=PoissonFamily())

    # Linear predictor / fitted values.
    np.testing.assert_allclose(m.eta, r["eta"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.mu,  r["mu"],  atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.fitted_values, r["mu"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.linear_predictors, r["eta"], atol=1e-9, rtol=1e-9)
    # Residuals — all four types.
    np.testing.assert_allclose(m.residuals,                 r["res_dev"],     atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.residuals_of("deviance"),  r["res_dev"],     atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.residuals_of("pearson"),   r["res_pearson"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.residuals_of("working"),   r["res_working"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.residuals_of("response"),  r["res_response"], atol=1e-9, rtol=1e-9)
    # Working weights = sqrt_x_wt² — matches lme4's m@resp$sqrtXwt^2.
    np.testing.assert_allclose(m.working_weights, r["working_wts"], atol=1e-9, rtol=1e-9)
    # Prior weights = the user-supplied ``weights=`` (1s when not given).
    np.testing.assert_allclose(m.prior_weights, r["prior_wts"], atol=1e-12, rtol=1e-12)
    # Summary statistics.
    np.testing.assert_allclose(m.AIC, r["aic"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(m.BIC, r["bic"], atol=1e-9, rtol=1e-9)
    assert m.sigma == pytest.approx(r["sigma"])
    # SE(β̂) and t-values.
    np.testing.assert_allclose(m._se_beta, r["se_beta"], atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(m.t_values.row(0), r["t_value"], atol=1e-7, rtol=1e-7)
    # vcov_beta — full p×p matrix.
    np.testing.assert_allclose(m._vcov_beta_arr, r["vcov"], atol=1e-9, rtol=1e-7)
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

    laplace_r = 184.0531328091428
    dev_r     =  73.474376562348297
    aic_r     = 194.0531328091428
    sigma_r   =   1.0
    se_beta_r = np.array([
        0.22785120243880524, 0.30534753437042639,
        0.32599944470550102, 0.42870015465286354,
    ])
    sd_herd_r = np.array([0.64206385678459621])

    m = lme("y_prop ~ period + (1|herd)", df,
            family=BinomialFamily(), weights=size)
    assert m.deviance_laplace == pytest.approx(laplace_r, rel=1e-9, abs=1e-9)
    assert m.deviance         == pytest.approx(dev_r,     rel=1e-9, abs=1e-9)
    assert m.AIC              == pytest.approx(aic_r,     rel=1e-9, abs=1e-9)
    assert m.sigma            == pytest.approx(sigma_r)  # = 1
    np.testing.assert_allclose(m._se_beta,       se_beta_r, atol=1e-9, rtol=1e-7)
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
         1.7729347165001661e+00,  1.9264503379702644e+00,  1.6090960292460870e+00,
         2.0980111217169899e+00,  1.9855169263308092e+00,  1.8473502170732887e+00,
        -1.4648045824321387e-01, -5.5029223772085523e-02, -1.3990301952553597e-01,
        -1.3370244188013369e-01,  6.7861669446641626e-03, -2.3712584818416538e-02,
         7.5727540730254739e-01,  7.5410488624655425e-01,  7.9069711357197059e-01,
         6.7561023826827715e-01,  7.0683751651188176e-01,  8.4822220986501617e-01,
         3.7268263570884946e-01,  1.8794248429333510e-01,  3.2117938174038180e-01,
         4.8957685110411947e-01,  3.5756414736518177e-01,  3.6997229457215747e-01,
         5.7851790841970629e-01,  7.5417364255200647e-01,  3.7728079220245586e-01,
         6.8342214818593616e-01,  3.0052636400710381e-01,  5.0918388200710707e-01,
         4.2758174037888841e-02,  4.1715435375195364e-01,  3.4029432108189372e-01,
         3.8541125423265798e-01,  8.4934958179896269e-02,  3.1811142916298196e-01,
         1.2075159788084968e+00,  1.2166684608620288e+00,  1.3598222175684658e+00,
         1.4147984424759237e+00,  1.3149543273028947e+00,  1.1913320983614026e+00,
         6.4854382573565850e-02,  4.0199578382450923e-01,  2.7570572049890935e-01,
         2.9438582346227399e-01,  5.1177134592300733e-01,  6.6332604556258057e-02,
         2.0964110029231371e+00,  2.1842483001260602e+00,  1.9566548542974731e+00,
         1.8391365645049689e+00,  1.6177162345028149e+00,  1.8639943192877406e+00,
        -3.2682967536433627e-02,  2.9846602169373737e-02, -1.0070728207137947e-02,
         7.5214068325729855e-02,  2.2435661413152652e-01, -4.9448737274826127e-02,
         1.8798647378427574e-01,  1.8537968966568491e-01,  1.9052996043250131e-01,
         1.2179493350797582e-01,  1.9874908113387407e-01,  5.2639555663848459e-01,
         1.0455418362129891e+00,  1.1718391039601570e+00,  9.5293261116530703e-01,
         1.1588507042836282e+00,  8.7512918838533871e-01,  1.0298821786011483e+00,
    ]),
    "fit_response": np.array([
        5.88810795790300912e+00, 6.86509816162280817e+00, 4.99829087623692114e+00,
        8.14994453507211603e+00, 7.28281108940518962e+00, 6.34298968896413662e+00,
        8.63742611217874434e-01, 9.46457488491396659e-01, 8.69442550261335656e-01,
        8.74850344700918936e-01, 1.00680924515021419e+00, 9.76566349420574631e-01,
        2.13245821816376679e+00, 2.12570792111612761e+00, 2.20493297932288357e+00,
        1.96523186982019094e+00, 2.02756895544283777e+00, 2.33549114591438878e+00,
        1.45162357323487323e+00, 1.20676410546324164e+00, 1.37875288180395139e+00,
        1.63162565326574716e+00, 1.42984228412145975e+00, 1.44769450511205156e+00,
        1.78339331886299290e+00, 2.12585408196393644e+00, 1.45831373501229788e+00,
        1.98064420566488586e+00, 1.35056951169510819e+00, 1.66393267526986377e+00,
        1.04368547408698320e+00, 1.51763674778322888e+00, 1.40536115711614462e+00,
        1.47021883084621741e+00, 1.08864625686162797e+00, 1.37452941535673179e+00,
        3.34516486315130601e+00, 3.37592196171916692e+00, 3.89550068864534316e+00,
        4.11565684076469473e+00, 3.72458086971252156e+00, 3.29146284060856198e+00,
        1.06700363874716864e+00, 1.49480503030242451e+00, 1.31746010547285919e+00,
        1.34230169525995713e+00, 1.66824361582203884e+00, 1.06858207333250332e+00,
        8.13691408356810264e+00, 8.88396796518778054e+00, 7.07561845821437529e+00,
        6.29110395282117807e+00, 5.04156341062973201e+00, 6.44944653944336910e+00,
        9.67845349350761386e-01, 1.03029647658658785e+00, 9.89979811775743235e-01,
        1.07811491643905888e+00, 1.25151724859451807e+00, 9.51753946386928895e-01,
        1.20681719156953315e+00, 1.20367537647648870e+00, 1.20989062194077812e+00,
        1.12952245082255320e+00, 1.21987583763280649e+00, 1.69281962636357597e+00,
        2.84493959740692404e+00, 3.22792367016979176e+00, 2.59330367003079898e+00,
        3.18626920542012693e+00, 2.39918522081199770e+00, 2.80073582864542825e+00,
    ]),
    "fit_newdata": np.array([
        5.70995408672909566, 1.24165727065417753, 3.60444321759289021,
        6.15009073125400718, 1.33736712338764252, 3.10699371508957078,
    ]),
    "fit_no_re": np.array([
        1.72539794323052997, 2.01168632314924078, 1.46465398718648387,
        2.38818609284880212, 2.13408914449852860, 1.85869237478601623,
        1.85320682906005585, 2.03067610455639880, 1.86543635881044567,
        1.87703907639717915, 2.16016408643122082, 2.09527630601050729,
        1.92268363530416408, 1.91659737975346256, 1.98802889556122109,
        1.77190770890466509, 1.82811255895908831, 2.10574376951404041,
        1.90382965204230192, 1.58269218643330123, 1.80825846839043192,
        2.13990552164835535, 1.87526311121284062, 1.89867661062368343,
        2.13521199568662556, 2.54523165971190446, 1.74600238070760683,
        2.37137552462019086, 1.61700293024455632, 1.99218476971545488,
        1.63159304212434919, 2.37252085962151904, 2.19700047816457733,
        2.29839245095114020, 1.70188021404053202, 2.14880122984674538,
        1.79708540275966411, 1.81360869387646506, 2.09273614616707171,
        2.21100816667631284, 2.00091480874062855, 1.76823566746759764,
        1.71704066806369338, 2.40546604964636046, 2.12007953628394752,
        2.16005505124459507, 2.68456641438202936, 1.71958071223643882,
        2.83872533188171428, 3.09935003017082256, 2.46847111199375213,
        2.19477752535682002, 1.75885029865853393, 2.25001850580022955,
        1.81757050224017225, 1.93485094045442740, 1.85913803781140574,
        2.02465184283753974, 2.35029370716724317, 1.78735156345378332,
        1.67682426059907508, 1.67245883408117657, 1.68109466927883799,
        1.56942630720011200, 1.69496872745158567, 2.35210522201212502,
        1.77736497160540252, 2.01663278461343154, 1.62015640263493332,
        1.99060931943177799, 1.49888165490449521, 1.74974883863680208,
    ]),
    "se_link_se": np.array([
        1.68538645954805333e-01, 1.56214024232805726e-01, 2.16008810962955766e-01,
        1.86890762283355355e-01, 1.61986563289629654e-01, 1.57740696774181149e-01,
        3.70449847096034124e-01, 3.68809930155671484e-01, 3.70129389745610882e-01,
        3.69855845358439961e-01, 3.71145798700533824e-01, 3.69647886368265954e-01,
        2.61650555790805150e-01, 2.61676912100076486e-01, 2.62000972057361559e-01,
        2.65594046045867671e-01, 2.63289919756053037e-01, 2.65268359342863802e-01,
        3.10609262140044973e-01, 3.24624968801857317e-01, 3.11632483449639286e-01,
        3.16744110902230314e-01, 3.10671977563652058e-01, 3.10605947711657782e-01,
        2.86863716051871276e-01, 3.05192906421416332e-01, 3.00425257767325149e-01,
        2.94577145131915297e-01, 3.14692346794342659e-01, 2.87359413178223444e-01,
        3.43039384634079403e-01, 3.28864605558450429e-01, 3.22423786093495868e-01,
        3.25629117037944393e-01, 3.35909045177652243e-01, 3.21461137387788587e-01,
        2.10754611872980252e-01, 2.09803310144817207e-01, 2.08862011508721468e-01,
        2.15409579291644560e-01, 2.06302964072583001e-01, 2.12685719754191954e-01,
        3.38689549598418516e-01, 3.26012786196890114e-01, 3.19811800004582869e-01,
        3.19892725007859813e-01, 3.41664289142716515e-01, 3.38437921881600545e-01,
        1.69362223960737857e-01, 1.94152169951972320e-01, 1.51325881038022914e-01,
        1.61761399326298239e-01, 2.29140128830820466e-01, 1.57658572592517060e-01,
        3.54326908990102551e-01, 3.51780904037306885e-01, 3.53064732666388104e-01,
        3.51798066912931251e-01, 3.62771455218385552e-01, 3.55509646675473967e-01,
        3.28055030534884717e-01, 3.28249098459170885e-01, 3.27870936603638530e-01,
        3.34643220662943774e-01, 3.27311682752264188e-01, 3.49457561374648173e-01,
        2.30365011686698762e-01, 2.37148904210649880e-01, 2.37053530594688172e-01,
        2.35621850644197051e-01, 2.49772973617565425e-01, 2.30808628463998050e-01,
    ]),
    "se_resp_se": np.array([
        9.92373742460687103e-01, 1.07242461058033545e+00, 1.07967486902292764e+00,
        1.52314934672669389e+00, 1.17971753946035052e+00, 1.00054761316864949e+00,
        3.19973318255990846e-01, 3.49062920225824247e-01, 3.21806240547095779e-01,
        3.23568513801480995e-01, 3.73673021430357799e-01, 3.60985686961688867e-01,
        5.57958877983219637e-01, 5.56248684824341200e-01, 5.77694583903929804e-01,
        5.21953883723830447e-01, 5.33838467578409026e-01, 6.19531904536494871e-01,
        4.50887726987579507e-01, 3.91745760087206074e-01, 4.29664184619912348e-01,
        5.16807816868929848e-01, 4.44211930012143152e-01, 4.49662523757288157e-01,
        5.11590834631117874e-01, 6.48795585902405603e-01, 4.38114279746700308e-01,
        5.83452515626832202e-01, 4.25013889144223000e-01, 4.78146717133619503e-01,
        3.58025222782326125e-01, 4.99097010440741062e-01, 4.53121865106123678e-01,
        4.78746059741012697e-01, 3.65686124678614610e-01, 4.41857789233547071e-01,
        7.05008922384584569e-01, 7.08279602359266125e-01, 8.13622109664076110e-01,
        8.86551908577901893e-01, 7.68392073349732252e-01, 7.00047143299009189e-01,
        3.61382981827152228e-01, 4.87325552750020152e-01, 4.21339287765502668e-01,
        4.29392547079377518e-01, 5.69979269116711929e-01, 3.61648696258584479e-01,
        1.37808586537054301e+00, 1.72484165822501567e+00, 1.07072419707818756e+00,
        1.01765777871555985e+00, 1.15522448942044731e+00, 1.01681053542039046e+00,
        3.42933651015901231e-01, 3.62438625960081862e-01, 3.49526957589724019e-01,
        3.79278743513257299e-01, 4.54014733503543322e-01, 3.38357709202005086e-01,
        3.95902450630366998e-01, 3.95105357165910498e-01, 3.96687971403681672e-01,
        3.77987030754360742e-01, 3.99279613164421676e-01, 5.91568618476158381e-01,
        6.55374543604598125e-01, 7.65498561256385379e-01, 6.14751790884963123e-01,
        7.50754646831705563e-01, 5.99251626861527997e-01, 6.46433995299630304e-01,
    ]),
    "fit_random_only": np.array([
         1.22747700089424439, -0.76339801780652361,  0.10355347045266639,
        -0.27118482819260464, -0.18004802873499109, -0.44679868996487870,
         0.62134984695012585, -0.47574788459633205,  1.05305587830513336,
        -0.63018368798826496, -0.32891520964726800,  0.47040992174478846,
    ]).repeat(6),    # n_per = 6 obs per group; b is piecewise-constant
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
        atol=1e-9, rtol=1e-9,
    )
    np.testing.assert_allclose(
        p_resp["fit"].to_numpy(), _GLMER_PREDICT_POISSON_REF["fit_response"],
        atol=1e-9, rtol=1e-9,
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
        atol=1e-9, rtol=1e-9,
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
        atol=1e-9, rtol=1e-9,
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
        atol=1e-9, rtol=1e-9,
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
        atol=1e-9, rtol=1e-9,
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
# Canonical Bates lme4 example — fm10: Contraception / Binomial GLMM.
# Reference: Bates (2010), Doug, *lme4: Mixed-effects modeling with R*,
# §6.1 "Contraception data" and Bates et al. (2015), §7.2.
# ----------------------------------------------------------------------


def test_glmer_bates_fm10_contraception_matches_lme4():
    """Bit-by-bit match against lme4's

        glmer(use ~ poly(age, 2) + urban + livch + (1 | district),
              Contraception, binomial,
              control = glmerControl(optimizer = "Nelder_Mead",
                                     calc.derivs = FALSE))

    This is the canonical Bates-book Binomial GLMM. The fit also exercises
    factor-response coercion (use ∈ {"N","Y"} → 0/1) and a multi-term
    polynomial fixed effect via ``poly(age, 2)``.
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

    # lme4 reference (R-script: glmer + Nelder_Mead, calc.derivs=FALSE):
    expected_theta = np.array([0.47521470485270112])
    expected_beta = np.array([
        -1.40544920712743404,    # (Intercept)
        -5.79868870265507130,    # poly(age, 2)1
        -16.32092873981419956,   # poly(age, 2)2
         0.69725598076645778,    # urbanY
         0.81500802452171706,    # livch1
         0.91645635197069453,    # livch2
         0.91503735175035861,    # livch3+
    ])
    expected_dev_laplace = 2372.728706542644
    expected_dev_resid = 2289.7328633517777

    np.testing.assert_allclose(m.theta, expected_theta, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        m.bhat.to_numpy().ravel(), expected_beta, atol=1e-8, rtol=1e-8,
    )
    assert m.deviance_laplace == pytest.approx(expected_dev_laplace, rel=1e-9)
    assert m.deviance == pytest.approx(expected_dev_resid, rel=1e-9)
    # Fixed-effect names line up with R's: (Intercept) first, then poly
    # terms, urban dummy, then 3 livch contrasts.
    assert m.column_names == [
        "(Intercept)", "poly(age, 2)1", "poly(age, 2)2",
        "urbanY", "livch1", "livch2", "livch3+",
    ]
    # AIC / BIC / logLik all match the printed summary's first row.
    assert m.AIC == pytest.approx(2388.728706542644, rel=1e-9)
    assert m.BIC == pytest.approx(2433.26747195075, rel=1e-9)
    assert m.loglike == pytest.approx(-1186.364353271322, rel=1e-9)
    # Scaled (Pearson, σ-divided) residuals — what summary() prints.
    # Pinned against ``residuals(fm10, "pearson", scaled=TRUE)`` quantiles.
    pearson_scaled = m.residuals_of("pearson") / m.sigma
    expected_qs = np.array([
        -1.8437935222300932, -0.7591759664616604, -0.4639993223870970,
         0.9493032396680572,  3.0714631499755760,
    ])
    np.testing.assert_allclose(
        np.quantile(pearson_scaled, [0, .25, .5, .75, 1]),
        expected_qs, atol=1e-8, rtol=1e-8,
    )
    # Per-coefficient SEs match lme4's RX-based ``vcov(m, use.hessian=FALSE)``.
    expected_se = np.array([
        0.15045559012118487, 3.2691846924049117, 2.5943129964813090,
        0.11987785020750072, 0.16218877245296692, 0.18509835776861358,
        0.18576758945818120,
    ])
    np.testing.assert_allclose(m._se_beta, expected_se, atol=1e-7, rtol=1e-7)


def test_deriv12_uses_supplied_fx_to_save_one_eval():
    """``fx`` argument skips the redundant ``fn(x)`` call. Pin: same answer."""
    def py_fn(x):
        return float(x[0] ** 3 + x[1] ** 2)

    x0 = np.array([0.2, 0.4])
    g_a, H_a = _deriv12(py_fn, x0)
    g_b, H_b = _deriv12(py_fn, x0, fx=py_fn(x0))
    np.testing.assert_array_equal(g_a, g_b)
    np.testing.assert_array_equal(H_a, H_b)

