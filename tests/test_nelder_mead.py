"""Pin :class:`hea.lme.NelderMead` against lme4's ``Nelder_Mead()``.

Both run the same algorithm (NLopt-derived bounded simplex). With identical
``x0``, ``xstep``, ``xtol``, and the same objective, every reflection /
expansion / contraction / shrink step must match — and the final
``(xpos, value, evals)`` tuple should agree at floating-point precision.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from hea.lme import NelderMead, NMStatus


def _r_nelder_mead(fn_body_r: str, x0, lower, upper, xst, xt, maxfun=10000):
    """Call ``lme4::Nelder_Mead`` from R on the given R-side objective body.

    Returns dict with ``par``, ``fval``, ``feval``, ``convergence``.
    The objective body must be R code that uses ``x`` and returns a scalar.
    """
    x0_str  = ",".join(f"{v:.17g}" for v in x0)
    lo_str  = ",".join(("-Inf" if np.isneginf(v) else f"{v:.17g}") for v in lower)
    up_str  = ",".join(("Inf"  if np.isposinf(v) else f"{v:.17g}") for v in upper)
    xst_str = ",".join(f"{v:.17g}" for v in xst)
    xt_str  = ",".join(f"{v:.17g}" for v in xt)
    r_script = f"""
        suppressMessages(suppressWarnings(library(lme4)))
        fn <- function(x) {{ {fn_body_r} }}
        opt <- Nelder_Mead(fn, par=c({x0_str}),
                           lower=c({lo_str}), upper=c({up_str}),
                           control=list(xst=c({xst_str}), xt=c({xt_str}),
                                        maxfun={maxfun}))
        hp <- function(...) format(c(...), digits=17, scientific=TRUE)
        cat("PAR",  hp(opt$par),  "\\n")
        cat("FVAL", hp(opt$fval), "\\n")
        cat("FEVAL", opt$feval, "\\n")
        cat("CONV",  opt$convergence, "\\n")
    """
    out = subprocess.run(
        ["R", "--vanilla", "--slave", "-e", r_script],
        capture_output=True, text=True, check=True,
    ).stdout
    lines = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
             for l in out.strip().split("\n")}
    return {
        "par":  np.array(lines["PAR"].split(),  dtype=float),
        "fval": float(lines["FVAL"]),
        "feval": int(lines["FEVAL"]),
        "convergence": int(lines["CONV"]),
    }


def test_nelder_mead_quadratic_matches_lme4():
    """Minimize ``f(x) = (x[0]-2)^2 + (x[1]-3)^2``, bounded ``[-5, 5]``."""
    def py_fn(x):
        return (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2

    x0 = np.array([0.0, 0.0])
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.5, 0.5])
    xt  = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)

    r = _r_nelder_mead("(x[1] - 2)^2 + (x[2] - 3)^2",
                       x0, lb, ub, xst, xt)

    np.testing.assert_allclose(nm.xpos(), r["par"], atol=1e-12)
    assert nm.value() == pytest.approx(r["fval"], rel=1e-12, abs=1e-12)
    # Evaluations should match exactly — same algorithm, same trajectory.
    assert nm.nevals == r["feval"]


def test_nelder_mead_rosenbrock_matches_lme4():
    """A harder objective. Rosenbrock function — narrow curved valley."""
    def py_fn(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    x0 = np.array([-1.2, 1.0])
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.1, 0.1])
    xt  = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)
    r = _r_nelder_mead(
        "100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2",
        x0, lb, ub, xst, xt,
    )

    np.testing.assert_allclose(nm.xpos(), r["par"], atol=1e-10)
    assert nm.value() == pytest.approx(r["fval"], rel=1e-10, abs=1e-10)
    assert nm.nevals == r["feval"]


def test_nelder_mead_bounded_at_optimum_matches_lme4():
    """Optimum on the lower bound — verifies the bound-pinning logic."""
    def py_fn(x):
        return (x[0] + 3) ** 2 + (x[1] - 2) ** 2

    x0 = np.array([0.0, 0.0])
    lb = np.array([0.0, -5.0])
    ub = np.array([5.0, 5.0])
    xst = np.array([0.5, 0.5])
    xt  = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)
    r = _r_nelder_mead(
        "(x[1] + 3)^2 + (x[2] - 2)^2",
        x0, lb, ub, xst, xt,
    )

    np.testing.assert_allclose(nm.xpos(), r["par"], atol=1e-12)
    assert nm.value() == pytest.approx(r["fval"], rel=1e-12, abs=1e-12)
    assert nm.nevals == r["feval"]


def test_nelder_mead_1d_matches_lme4():
    """1D case — smallest possible simplex (2 vertices)."""
    def py_fn(x):
        return (x[0] - 1.7) ** 2 + 0.1 * np.sin(x[0])

    x0 = np.array([0.0])
    lb = np.array([-5.0])
    ub = np.array([5.0])
    xst = np.array([0.5])
    xt  = xst * 5e-4

    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xt)
    nm.minimize(py_fn)
    r = _r_nelder_mead(
        "(x[1] - 1.7)^2 + 0.1 * sin(x[1])",
        x0, lb, ub, xst, xt,
    )

    np.testing.assert_allclose(nm.xpos(), r["par"], atol=1e-12)
    assert nm.value() == pytest.approx(r["fval"], rel=1e-12, abs=1e-12)
    assert nm.nevals == r["feval"]


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
