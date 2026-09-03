"""Pin :class:`hea.gmm.NelderMead` against lme4's ``Nelder_Mead()``.

Both run the same algorithm (NLopt-derived bounded simplex). With identical
``x0``, ``xstep``, ``xtol``, and the same objective, every reflection /
expansion / contraction / shrink step must match — and the final
``(xpos, value, evals)`` tuple should agree at floating-point precision.

Reference values were captured by running lme4's ``Nelder_Mead`` locally;
the R recipe is preserved in a block comment above each test so future-me
can re-derive after a math change. Tests must NEVER call R at runtime — CI
has no R installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from hea.models.gmm import (
    NelderMead,
    _nlopt_compute_rescaling,
    _nlopt_default_step,
    _nlopt_ln_bobyqa,
)


def test_nelder_mead_quadratic_matches_lme4():
    """Minimize ``f(x) = (x[0]-2)^2 + (x[1]-3)^2``, bounded ``[-5, 5]``."""

    def py_fn(x):
        return (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2

    x0 = np.array([0.0, 0.0])
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.5, 0.5])
    xt = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    expected_par = np.array([1.9999241652667235, 2.9999784858794447])
    expected_fval = 6.2137641543925668e-09
    expected_feval = 75

    np.testing.assert_allclose(nm.xpos(), expected_par, atol=1e-12)
    assert nm.value() == pytest.approx(expected_fval, rel=1e-12, abs=1e-12)
    assert nm.nevals == expected_feval


def test_nelder_mead_rosenbrock_matches_lme4():
    """A harder objective — Rosenbrock function, narrow curved valley."""

    def py_fn(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    x0 = np.array([-1.2, 1.0])
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.1, 0.1])
    xt = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    expected_par = np.array([1.0000275944324057, 1.0000553049566485])
    expected_fval = 7.6278280955058216e-10
    expected_feval = 199

    np.testing.assert_allclose(nm.xpos(), expected_par, atol=1e-10)
    assert nm.value() == pytest.approx(expected_fval, rel=1e-10, abs=1e-10)
    assert nm.nevals == expected_feval


def test_nelder_mead_bounded_at_optimum_matches_lme4():
    """Optimum on the lower bound — verifies the bound-pinning logic."""

    def py_fn(x):
        return (x[0] + 3) ** 2 + (x[1] - 2) ** 2

    x0 = np.array([0.0, 0.0])
    lb = np.array([0.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.5, 0.5])
    xt = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    expected_par = np.array([0.0, 2.00003950484097])
    expected_fval = 9.0000000015606325
    expected_feval = 51

    np.testing.assert_allclose(nm.xpos(), expected_par, atol=1e-12)
    assert nm.value() == pytest.approx(expected_fval, rel=1e-12, abs=1e-12)
    assert nm.nevals == expected_feval


def test_nelder_mead_1d_matches_lme4():
    """1D case — smallest possible simplex (2 vertices)."""

    def py_fn(x):
        return (x[0] - 1.7) ** 2 + 0.1 * np.sin(x[0])

    x0 = np.array([0.0])
    lb = np.array([-5.0])
    ub = np.array([5.0])
    xst = np.array([0.5])
    xt = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    expected_par = np.array([1.706787109375])
    expected_fval = 9.9122814366828446e-02
    expected_feval = 29

    np.testing.assert_allclose(nm.xpos(), expected_par, atol=1e-12)
    assert nm.value() == pytest.approx(expected_fval, rel=1e-12, abs=1e-12)
    assert nm.nevals == expected_feval


def test_nelder_mead_infeasible_x0_raises():
    """x0 outside [lb, ub] is rejected by the constructor."""
    with pytest.raises(ValueError, match="not a feasible point"):
        NelderMead(
            lb=np.array([0.0]),
            ub=np.array([1.0]),
            xstep=np.array([0.1]),
            x0=np.array([-0.5]),
        )


def test_nelder_mead_zero_xstep_raises():
    """xstep must be nonzero in every coordinate."""
    with pytest.raises(ValueError, match="must be nonzero"):
        NelderMead(
            lb=np.array([0.0]),
            ub=np.array([1.0]),
            xstep=np.array([0.0]),
            x0=np.array([0.5]),
        )


# ----------------------------------------------------------------------
# NLopt LN_BOBYQA — lme4's DEFAULT lmer optimizer (``nloptwrap``).
#
# References captured from ``nloptr::nloptr(algorithm="NLOPT_LN_BOBYQA",
# xtol_abs=1e-8, ftol_abs=1e-8, maxeval=1e5)`` — exactly lme4's nloptwrap
# settings (utilities.R:836-839). Tests never call R at runtime.
# ----------------------------------------------------------------------


def test_nlopt_bobyqa_quadratic_matches_nloptr():
    """Anisotropic quadratic with mixed bounds ``lb=(0,-Inf,0)`` — exercises
    the variable rescaling (unequal default steps ⇒ non-trivial ``s``).
    """

    def fq(x):
        return (
            3 * (x[0] - 0.7) ** 2
            + (x[1] + 0.3) ** 2
            + 5 * (x[2] - 0.9) ** 2
            + 0.2 * x[0] * x[1]
        )

    r = _nlopt_ln_bobyqa(
        fq,
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, -np.inf, 0.0]),
        np.array([np.inf, np.inf, np.inf]),
    )
    expected_sol = np.array([0.712374762653482, -0.371237567717368, 0.899999814113911])
    expected_fval = -0.0473578595314938
    np.testing.assert_allclose(r.x, expected_sol, rtol=0, atol=1e-12)
    assert r.fun == pytest.approx(expected_fval, rel=0, abs=1e-12)


def test_nlopt_default_step_matches_nlopt_heuristic():
    """``nlopt_set_default_initial_step`` (options.c): for ``x=(1,0,1)`` with
    ``lb=(0,-Inf,0)``, ``ub=(Inf,Inf,Inf)`` the per-axis steps are
    ``(x-lb)*0.75`` where finite, else 1 (the both-infinite middle axis)."""
    dx = _nlopt_default_step(
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, -np.inf, 0.0]),
        np.array([np.inf, np.inf, np.inf]),
    )
    np.testing.assert_allclose(dx, [0.75, 1.0, 0.75], rtol=0, atol=0)


def test_nlopt_compute_rescaling_makes_steps_equal():
    """``nlopt_compute_rescaling`` (rescale.c): ``s[i]=dx[i]/dx[0]`` when the
    steps differ, so ``dx[i]/s[i]`` is constant; all-ones when equal."""
    s = _nlopt_compute_rescaling(np.array([0.75, 1.0, 0.75]))
    np.testing.assert_allclose(s, [1.0, 1.0 / 0.75, 1.0], rtol=0, atol=0)
    np.testing.assert_array_equal(
        _nlopt_compute_rescaling(np.array([0.2, 0.2, 0.2])), [1.0, 1.0, 1.0]
    )


def test_nlopt_bobyqa_respects_maxeval():
    """``maxeval`` caps the evaluation count (NLopt ``nlopt_stop_evals``)."""

    def fq(x):
        return (x[0] - 0.123) ** 2 + (x[1] + 0.4) ** 2

    r = _nlopt_ln_bobyqa(
        fq,
        np.array([1.0, 1.0]),
        np.array([-np.inf, -np.inf]),
        np.array([np.inf, np.inf]),
        maxeval=8,
    )
    assert r.nfev <= 8
