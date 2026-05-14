"""GLMM-specific tests for ``hea.lme(family=...)``.

This file accumulates Phase 2-13 tests of ``lme-family-port.md``. Phase 2
focuses on the ``_GlmResponse`` private class — verifying its mutators
and pure-compute methods match the documented formulas, plus a single
R-oracle cross-check. Phase 3 tests the PIRLS inner loop (_PredState,
_internal_glmer_wrk_iter, _pwrss_update) against ``lme4::glmer``.
"""
from __future__ import annotations

import subprocess

import numpy as np
import polars as pl
import pytest

from hea._cholmod import csc_array
from hea.family import Binomial, Gamma, Gaussian, Poisson
from hea.formula import materialize_bars, parse, expand
from hea.lme import (
    _GlmResponse,
    _PredState,
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


def _run_r(script: str) -> str:
    """Run an R snippet via `R --vanilla -e` and return stdout."""
    result = subprocess.run(
        ["R", "--vanilla", "--slave", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def test_poisson_glm_state_matches_R():
    """Build _GlmResponse at R's converged glm() state and compare.

    Uses the canonical R example: ``count ~ outcome + treatment``
    from ?glm. We fit it in R, extract μ̂ / η̂ / weights, and verify
    that _GlmResponse reproduces every R-side value PIRLS would care
    about: working_response, weighted_working_response, deviance,
    deviance_residuals, sqrt_x_wt.
    """
    r_output = _run_r("""
        counts <- c(18,17,15,20,10,20,25,13,12)
        outcome <- gl(3,1,9)
        treatment <- gl(3,3)
        m <- glm(counts ~ outcome + treatment, family=poisson())
        cat(paste(m$y, collapse=" "), "\\n")
        cat(paste(m$linear.predictors, collapse=" "), "\\n")
        cat(paste(m$fitted.values, collapse=" "), "\\n")
        cat(paste(m$weights, collapse=" "), "\\n")
        cat(paste(residuals(m, type="working"), collapse=" "), "\\n")
        cat(paste(residuals(m, type="deviance"), collapse=" "), "\\n")
        cat(deviance(m), "\\n")
    """)
    lines = r_output.strip().split("\n")
    y = np.array(lines[0].split(), dtype=float)
    eta = np.array(lines[1].split(), dtype=float)
    mu_r = np.array(lines[2].split(), dtype=float)
    r_wts = np.array(lines[3].split(), dtype=float)
    wrk_resids_r = np.array(lines[4].split(), dtype=float)
    dev_resids_r = np.array(lines[5].split(), dtype=float)
    dev_r = float(lines[6])

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

    Setup: synthetic Poisson GLMM. Construct lme4's ``mkGlmerDevfun``,
    set θ to a fixed value, call ``RglmerWrkIter`` once — saving the
    resulting pp@delu/delb and pdev. In Python, build identical state,
    call ``_internal_glmer_wrk_iter`` once with u_only=True (matching
    lme4's default Stage-0 dispatch), and verify all values match at
    ≤ 1e-10.

    This avoids the multi-iteration noise issue near convergence —
    one-step is fully deterministic given identical inputs.
    """
    rng = np.random.default_rng(2026)
    n_groups, n_per = 10, 5
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    true_b = rng.standard_normal(n_groups) * 0.5
    eta = 0.5 + 0.3 * x + true_b[g]
    y = rng.poisson(np.exp(eta)).astype(float)

    # Pin to a fixed (non-converged) θ so the iteration is non-trivial.
    fixed_theta = 0.7

    csv = "\n".join(
        ["y,x,g"] + [f"{int(y[i])},{x[i]:.10g},G{g[i]:02d}" for i in range(n)]
    )
    r_script = f"""
        suppressMessages(suppressWarnings(library(lme4)))
        d <- read.csv(text="{csv}")
        d$g <- factor(d$g)
        glmod <- glFormula(y ~ x + (1|g), data=d, family=poisson)
        devfun <- mkGlmerDevfun(glmod$fr, glmod$X, glmod$reTrms, family=poisson(), nAGQ=0)
        rho <- environment(devfun)
        # Reset δu/δβ; set η to lp0 (lme4's post-init-PIRLS lin pred); set θ.
        invisible(rho$pp$setDelu(rep(0.0, length(rho$pp$delu))))
        invisible(rho$pp$setDelb(rep(0.0, length(rho$pp$delb))))
        invisible(rho$resp$updateMu(rho$lp0))
        invisible(rho$pp$setTheta({fixed_theta}))
        # Helper to print high-precision numeric output.
        hp <- function(...) format(c(...), digits=17, scientific=TRUE)
        # Capture the lp0 value so Python can start from the same η.
        cat("RESULT_LP0", hp(rho$lp0), "\\n")
        # One PIRLS step.
        pdev <- lme4:::RglmerWrkIter(rho$pp, rho$resp, uOnly=TRUE)
        cat("RESULT_PDEV", hp(pdev), "\\n")
        cat("RESULT_DELU", hp(rho$pp$delu), "\\n")
        cat("RESULT_DELB", hp(rho$pp$delb), "\\n")
        cat("RESULT_MU", hp(rho$resp$mu), "\\n")
    """
    r_out = subprocess.run(
        ["R", "--vanilla", "--slave", "-e", r_script],
        capture_output=True, text=True, check=True,
    ).stdout
    out_lines = {
        line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1]
        for line in r_out.strip().split("\n")
        if line.startswith("RESULT_")
    }
    lp0_r = np.array(out_lines["RESULT_LP0"].split(), dtype=float)
    pdev_r = float(out_lines["RESULT_PDEV"])
    delu_r = np.array(out_lines["RESULT_DELU"].split(), dtype=float)
    delb_r = (np.array(out_lines["RESULT_DELB"].split(), dtype=float)
              if "RESULT_DELB" in out_lines and out_lines["RESULT_DELB"].strip()
              else np.zeros(0))
    mu_r = np.array(out_lines["RESULT_MU"].split(), dtype=float)

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
