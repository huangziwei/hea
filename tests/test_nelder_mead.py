"""Pin :class:`hea.lme.NelderMead` against lme4's ``Nelder_Mead()``.

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

from hea.lme import NelderMead, NMStatus


def test_nelder_mead_quadratic_matches_lme4():
    """Minimize ``f(x) = (x[0]-2)^2 + (x[1]-3)^2``, bounded ``[-5, 5]``.

    R recipe::
        Nelder_Mead(function(x) (x[1]-2)^2 + (x[2]-3)^2,
                    par=c(0,0), lower=c(-5,-5), upper=c(5,5),
                    control=list(xst=c(0.5,0.5),
                                 xt =c(0.5,0.5)*5e-4, maxfun=10000))
    """
    def py_fn(x):
        return (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2

    x0 = np.array([0.0, 0.0])
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.5, 0.5])
    xt  = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    # lme4 reference (Nelder_Mead) — see R recipe in the docstring.
    expected_par = np.array([1.9999241652667235, 2.9999784858794447])
    expected_fval = 6.2137641543925668e-09
    expected_feval = 75

    np.testing.assert_allclose(nm.xpos(), expected_par, atol=1e-12)
    assert nm.value() == pytest.approx(expected_fval, rel=1e-12, abs=1e-12)
    assert nm.nevals == expected_feval


def test_nelder_mead_rosenbrock_matches_lme4():
    """A harder objective — Rosenbrock function, narrow curved valley.

    R recipe::
        Nelder_Mead(function(x) 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2,
                    par=c(-1.2,1.0), lower=c(-5,-5), upper=c(5,5),
                    control=list(xst=c(0.1,0.1),
                                 xt =c(0.1,0.1)*5e-4, maxfun=10000))
    """
    def py_fn(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    x0 = np.array([-1.2, 1.0])
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.1, 0.1])
    xt  = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    expected_par = np.array([1.0000275944324057, 1.0000553049566485])
    expected_fval = 7.6278280955058216e-10
    expected_feval = 199

    np.testing.assert_allclose(nm.xpos(), expected_par, atol=1e-10)
    assert nm.value() == pytest.approx(expected_fval, rel=1e-10, abs=1e-10)
    assert nm.nevals == expected_feval


def test_nelder_mead_bounded_at_optimum_matches_lme4():
    """Optimum on the lower bound — verifies the bound-pinning logic.

    R recipe::
        Nelder_Mead(function(x) (x[1]+3)^2 + (x[2]-2)^2,
                    par=c(0,0), lower=c(0,-5), upper=c(5,5),
                    control=list(xst=c(0.5,0.5),
                                 xt =c(0.5,0.5)*5e-4, maxfun=10000))
    """
    def py_fn(x):
        return (x[0] + 3) ** 2 + (x[1] - 2) ** 2

    x0 = np.array([0.0, 0.0])
    lb = np.array([0.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.5, 0.5])
    xt  = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    expected_par = np.array([0.0, 2.00003950484097])
    expected_fval = 9.0000000015606325
    expected_feval = 51

    np.testing.assert_allclose(nm.xpos(), expected_par, atol=1e-12)
    assert nm.value() == pytest.approx(expected_fval, rel=1e-12, abs=1e-12)
    assert nm.nevals == expected_feval


def test_nelder_mead_1d_matches_lme4():
    """1D case — smallest possible simplex (2 vertices).

    R recipe::
        Nelder_Mead(function(x) (x[1]-1.7)^2 + 0.1 * sin(x[1]),
                    par=c(0), lower=c(-5), upper=c(5),
                    control=list(xst=c(0.5), xt=c(0.5)*5e-4, maxfun=10000))
    """
    def py_fn(x):
        return (x[0] - 1.7) ** 2 + 0.1 * np.sin(x[0])

    x0 = np.array([0.0])
    lb = np.array([-5.0])
    ub = np.array([5.0])
    xst = np.array([0.5])
    xt  = xst * 5e-4

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
            lb=np.array([0.0]), ub=np.array([1.0]),
            xstep=np.array([0.1]), x0=np.array([-0.5]),
        )


def test_nelder_mead_zero_xstep_raises():
    """xstep must be nonzero in every coordinate."""
    with pytest.raises(ValueError, match="must be nonzero"):
        NelderMead(
            lb=np.array([0.0]), ub=np.array([1.0]),
            xstep=np.array([0.0]), x0=np.array([0.5]),
        )
