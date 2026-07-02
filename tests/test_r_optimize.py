"""Bit-exact tests for hea.R.optimize (R's ``nlm`` / ``optim`` ports).

Pinned values were produced by R 4.6.0 (arm64, CRAN build with the
Accelerate ``libRblas.vecLib`` symlink) running the identical Rosenbrock
problems::

    fr <- function(x) {
        v <- 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
        attr(v, "gradient") <- c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
                                 200*(x[2]-x[1]^2))
        v
    }
    nlm(fr, c(-1.2, 1))                          # default config
    nlm(fr, p0, typsize=p0, fscale=3.5, stepmax=2, ndigit=7,
        gradtol=1e-6, steptol=1e-4, iterlim=200,
        check.analyticals=FALSE)                 # mgcv gam.control style
    nlm(frv, c(-1.2, 1))                         # no gradient → FD path
    optim(c(-1.2, 1), frv, grv, method="L-BFGS-B",
          control=list(fnscale=3.7, factr=1e7, lmm=2))
    optim(c(-1.2, 1), frv, grv, method="L-BFGS-B",
          lower=c(-2, .3), upper=c(.5, .6), control=list(factr=1e7))
    nlm(fr, c(-1.2, 1), hessian=TRUE)

Equality is asserted EXACTLY (``==``, not allclose): the ports were
validated instruction-order-faithful against R's own compiled ``optif9``
and ``lbfgsb`` entry points (driven via ctypes) over randomized
batteries — 72/72 identical full trajectories for uncmin (n ≤ 4) and
48/48 for L-BFGS-B with 2 parameters. Bit-parity holds wherever every
BLAS dot has length ≤ 4 (see hea/R/_linpack.py for the Accelerate
emulation boundary); at n=2 as here, everything is inside it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hea.R.optimize import nlm, optim, optimHess


def _fr(x):
    v = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                  200 * (x[1] - x[0] ** 2)])
    return float(v), g


def _frv(x):
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def _grv(x):
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                     200 * (x[1] - x[0] ** 2)])


def test_nlm_rosenbrock_default_bitexact_vs_r():
    r = nlm(_fr, [-1.2, 1])
    assert r["minimum"] == 1.1820961652299623e-20
    assert r["estimate"][0] == 1.0000000000906322
    assert r["estimate"][1] == 1.0000000001752587
    assert r["gradient"][0] == 2.5835205088543604e-09
    assert r["gradient"][1] == -1.2011280858814644e-09
    assert r["code"] == 1 and r["iterations"] == 24


def test_nlm_rosenbrock_mgcv_config_bitexact_vs_r():
    # the exact knob set mgcv's gam.outer uses (mgcv.r:1697-1703 with
    # gam.control defaults): typsize = initial lsp (negative entries —
    # optchk takes |.|), small stepmax, loose steptol, no analytic
    # checks (msg 15)
    p0 = np.array([-1.2, 1.0])
    r = nlm(_fr, p0, typsize=p0, fscale=3.5, stepmax=2, ndigit=7,
            gradtol=1e-6, steptol=1e-4, iterlim=200,
            check_analyticals=False)
    assert r["minimum"] == 5.7433811131008786e-16
    assert r["estimate"][0] == 1.0000000207487953
    assert r["estimate"][1] == 1.000000040298318
    assert r["gradient"][0] == 5.2120685075120562e-07
    assert r["gradient"][1] == -2.3985462505038413e-07
    assert r["code"] == 1 and r["iterations"] == 42


def test_nlm_rosenbrock_fd_gradient_bitexact_vs_r():
    # no gradient supplied → optdrv's forward-difference path (fstofd)
    r = nlm(_frv, [-1.2, 1])
    assert r["minimum"] == 3.9737658773242689e-12
    assert r["estimate"][0] == 0.9999980066391061
    assert r["estimate"][1] == 0.99999601495019041
    assert r["code"] == 1 and r["iterations"] == 23


def test_nlm_hessian_true_bitexact_vs_r():
    # want.hessian → fdhess at the optimum + symmetrization
    r = nlm(_fr, [-1.2, 1], hessian=True)
    h = r["hessian"]
    assert h[0, 0] == 802.24001414660472
    assert h[0, 1] == -400.02000003629195
    assert h[1, 0] == -400.02000003629195
    assert h[1, 1] == 200.00000000000011


def test_optim_lbfgsb_unbounded_bitexact_vs_r():
    r = optim([-1.2, 1], _frv, _grv, method="L-BFGS-B",
              control={"fnscale": 3.7, "factr": 1e7, "lmm": 2})
    assert r["par"][0] == 1.0000001654856896
    assert r["par"][1] == 1.0000003405957025
    assert r["value"] == 3.6648220862233968e-14
    assert r["counts"] == {"function": 51, "gradient": 51}
    assert r["convergence"] == 0
    assert r["message"].startswith(
        "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH")


def test_optim_lbfgsb_bounded_bitexact_vs_r():
    r = optim([-1.2, 1], _frv, _grv, method="L-BFGS-B",
              lower=[-2, 0.3], upper=[0.5, 0.6],
              control={"factr": 1e7})
    assert r["par"][0] == 0.5
    assert r["par"][1] == 0.29999999999999999
    assert r["value"] == 0.49999999999999989
    assert r["counts"] == {"function": 4, "gradient": 4}
    assert r["convergence"] == 0
    assert r["message"].startswith(
        "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL")


def test_optim_maxit_and_method_validation():
    # convergence=1 when the iteration cap trips (R optim semantics)
    r = optim([-1.2, 1], _frv, _grv, method="L-BFGS-B",
              control={"maxit": 2})
    assert r["convergence"] == 1
    with pytest.raises(NotImplementedError, match="L-BFGS-B"):
        optim([0.0], _frv, _grv, method="BFGS")


def test_optimHess_matches_r():
    # optimHess(c(-1.2, 1), frv, grv) in R 4.6.0 (bit-exact: FD of the
    # analytic gradient with ndeps=1e-3 + symmetrization)
    h = optimHess([-1.2, 1], _frv, _grv)
    assert h[0, 0] == 1330.0004000000313
    assert h[0, 1] == 480.0000000000075
    assert h[1, 0] == 480.0000000000075
    assert h[1, 1] == 200.00000000000284


def test_nlm_nonfinite_value_mapping():
    # optimize.c fcn: NaN/Inf → DBL_MAX with a warning (nlm recovers by
    # backtracking); the optimum is still found
    def f(x):
        v = _frv(x)
        if x[0] > 1.5:
            return math.inf, _grv(x)
        return v, _grv(x)
    with pytest.warns(UserWarning, match="replaced by maximum positive"):
        r = nlm(f, [1.6, 1.0])
    assert r["code"] in (1, 2, 3)
    assert np.isfinite(r["minimum"])
