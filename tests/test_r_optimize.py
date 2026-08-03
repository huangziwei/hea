"""Bit-exact tests for hea.R.optimize (R's ``nlm`` / ``optim`` ports).

Every float is asserted EXACTLY (``==``, not allclose): the ports were
validated instruction-order-faithful against R's own compiled ``optif9``
and ``lbfgsb`` entry points (driven via ctypes) over randomized
batteries — 72/72 identical full trajectories for uncmin (n ≤ 4) and
48/48 for L-BFGS-B with 2 parameters.

The expected values are platform-dependent, because they are *R's*
values and R's arithmetic differs by platform: clang fuses ``a ± b*c``
on arm64, Accelerate vs reference BLAS ordering (see
``hea/R/_linpack.py`` / ``hea/R/_shared.py``), and libm — uncmin's
``qraux2``/``qrupdt`` call ``hypot`` (uncmin.c:364) and ``optdrv``
calls ``pow`` for ``rnf``, which hea mirrors by delegating to the same
platform libm (``np.hypot`` / ``**``), so Apple-vs-glibc 1-ulp
differences legitimately fork trajectories across platforms while
hea == R holds on each. Two sources:

* darwin/arm64 — the ``_PINS`` below, produced by R 4.6.0 (CRAN arm64
  build with the Accelerate ``libRblas.vecLib`` symlink). No R needed
  at test time.
* everywhere else — a live-R oracle: ``_R_ORACLE`` runs the identical
  problems in Rscript and ships every result as ``%a`` hex floats
  (decimal round-trips would hide 1-ulp gaps). Skipped when Rscript is
  absent. On reference-BLAS platforms (Linux CI) sequential dot order
  makes the emulation exact at every length, so bit-parity is expected
  there too.

The Rosenbrock problems, verbatim (the Python objectives below must
stay literal-for-literal identical — R parses these decimal literals
to the same doubles Python does)::

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
    optimHess(c(-1.2, 1), frv, grv)
"""

from __future__ import annotations

import math
import platform
import shutil
import subprocess
import sys

import numpy as np
import pytest

from hea.R.optimize import nlm, optim, optimHess

_DARWIN_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"

# R 4.6.0 on darwin/arm64 (Accelerate BLAS). Layouts: nlm entries are
# [minimum, estimate..., gradient..., code, iterations] (nlm_fd omits
# the gradient); optim entries are [par..., value, fncount, grcount,
# convergence]; hessians are column-major.
_PINS = {
    "nlm_default": [
        1.1820961652299623e-20,
        1.0000000000906322,
        1.0000000001752587,
        2.5835205088543604e-09,
        -1.2011280858814644e-09,
        1,
        24,
    ],
    "nlm_mgcv": [
        5.7433811131008786e-16,
        1.0000000207487953,
        1.000000040298318,
        5.2120685075120562e-07,
        -2.3985462505038413e-07,
        1,
        42,
    ],
    "nlm_fd": [3.9737658773242689e-12, 0.9999980066391061, 0.99999601495019041, 1, 23],
    "nlm_hess": [
        802.24001414660472,
        -400.02000003629195,
        -400.02000003629195,
        200.00000000000011,
    ],
    "optim_unb": [
        1.0000001654856896,
        1.0000003405957025,
        3.6648220862233968e-14,
        51,
        51,
        0,
    ],
    "optim_unb_msg": "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH",
    "optim_bnd": [0.5, 0.29999999999999999, 0.49999999999999989, 4, 4, 0],
    "optim_bnd_msg": "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
    "opthess": [
        1330.0004000000313,
        480.0000000000075,
        480.0000000000075,
        200.00000000000284,
    ],
}

_R_ORACLE = r"""
fr <- function(x) {
    v <- 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
    attr(v, "gradient") <- c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
                             200*(x[2]-x[1]^2))
    v
}
frv <- function(x) 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
grv <- function(x) c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
                     200*(x[2]-x[1]^2))
emit <- function(key, ...) cat(key, sprintf("%a", as.double(c(...))), "\n")

r <- nlm(fr, c(-1.2, 1))
emit("nlm_default", r$minimum, r$estimate, r$gradient, r$code, r$iterations)

p0 <- c(-1.2, 1)
r <- nlm(fr, p0, typsize=p0, fscale=3.5, stepmax=2, ndigit=7,
         gradtol=1e-6, steptol=1e-4, iterlim=200, check.analyticals=FALSE)
emit("nlm_mgcv", r$minimum, r$estimate, r$gradient, r$code, r$iterations)

r <- nlm(frv, c(-1.2, 1))
emit("nlm_fd", r$minimum, r$estimate, r$code, r$iterations)

r <- nlm(fr, c(-1.2, 1), hessian=TRUE)
emit("nlm_hess", r$hessian)

r <- optim(c(-1.2, 1), frv, grv, method="L-BFGS-B",
           control=list(fnscale=3.7, factr=1e7, lmm=2))
emit("optim_unb", r$par, r$value, r$counts, r$convergence)
cat("optim_unb_msg", r$message, "\n")

r <- optim(c(-1.2, 1), frv, grv, method="L-BFGS-B",
           lower=c(-2, .3), upper=c(.5, .6), control=list(factr=1e7))
emit("optim_bnd", r$par, r$value, r$counts, r$convergence)
cat("optim_bnd_msg", r$message, "\n")

h <- optimHess(c(-1.2, 1), frv, grv)
emit("opthess", h)
"""


@pytest.fixture(scope="module")
def rv(tmp_path_factory):
    """Expected R values: the darwin/arm64 pins, or the live-R oracle."""
    if _DARWIN_ARM64:
        return _PINS
    if shutil.which("Rscript") is None:
        pytest.skip(
            "pins are darwin/arm64 receipts; elsewhere the "
            "live-R oracle needs Rscript on PATH"
        )
    rf = tmp_path_factory.mktemp("r_optimize") / "oracle.R"
    rf.write_text(_R_ORACLE)
    out = subprocess.run(
        ["Rscript", "--vanilla", str(rf)],
        check=True,
        text=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    ).stdout
    vals = {}
    for line in out.splitlines():
        key, _, rest = line.partition(" ")
        if not key:
            continue
        if key.endswith("_msg"):
            vals[key] = rest.strip()
        else:
            vals[key] = [float.fromhex(tok) for tok in rest.split()]
    return vals


def _fr(x):
    v = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array(
        [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
    )
    return float(v), g


def _frv(x):
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def _grv(x):
    return np.array(
        [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
    )


def test_nlm_rosenbrock_default_bitexact_vs_r(rv):
    e = rv["nlm_default"]
    r = nlm(_fr, [-1.2, 1])
    assert r["minimum"] == e[0]
    assert r["estimate"][0] == e[1]
    assert r["estimate"][1] == e[2]
    assert r["gradient"][0] == e[3]
    assert r["gradient"][1] == e[4]
    assert r["code"] == int(e[5]) and r["iterations"] == int(e[6])


def test_nlm_rosenbrock_mgcv_config_bitexact_vs_r(rv):
    # the exact knob set mgcv's gam.outer uses (mgcv.r:1697-1703 with
    # gam.control defaults): typsize = initial lsp (negative entries —
    # optchk takes |.|), small stepmax, loose steptol, no analytic
    # checks (msg 15)
    e = rv["nlm_mgcv"]
    p0 = np.array([-1.2, 1.0])
    r = nlm(
        _fr,
        p0,
        typsize=p0,
        fscale=3.5,
        stepmax=2,
        ndigit=7,
        gradtol=1e-6,
        steptol=1e-4,
        iterlim=200,
        check_analyticals=False,
    )
    assert r["minimum"] == e[0]
    assert r["estimate"][0] == e[1]
    assert r["estimate"][1] == e[2]
    assert r["gradient"][0] == e[3]
    assert r["gradient"][1] == e[4]
    assert r["code"] == int(e[5]) and r["iterations"] == int(e[6])


def test_nlm_rosenbrock_fd_gradient_bitexact_vs_r(rv):
    # no gradient supplied → optdrv's forward-difference path (fstofd)
    e = rv["nlm_fd"]
    r = nlm(_frv, [-1.2, 1])
    assert r["minimum"] == e[0]
    assert r["estimate"][0] == e[1]
    assert r["estimate"][1] == e[2]
    assert r["code"] == int(e[3]) and r["iterations"] == int(e[4])


def test_nlm_hessian_true_bitexact_vs_r(rv):
    # want.hessian → fdhess at the optimum + symmetrization
    e = rv["nlm_hess"]  # column-major
    r = nlm(_fr, [-1.2, 1], hessian=True)
    h = r["hessian"]
    assert h[0, 0] == e[0]
    assert h[1, 0] == e[1]
    assert h[0, 1] == e[2]
    assert h[1, 1] == e[3]


def test_optim_lbfgsb_unbounded_bitexact_vs_r(rv):
    e = rv["optim_unb"]
    r = optim(
        [-1.2, 1],
        _frv,
        _grv,
        method="L-BFGS-B",
        control={"fnscale": 3.7, "factr": 1e7, "lmm": 2},
    )
    assert r["par"][0] == e[0]
    assert r["par"][1] == e[1]
    assert r["value"] == e[2]
    assert r["counts"] == {"function": int(e[3]), "gradient": int(e[4])}
    assert r["convergence"] == int(e[5])
    assert r["message"] == rv["optim_unb_msg"]


def test_optim_lbfgsb_bounded_bitexact_vs_r(rv):
    e = rv["optim_bnd"]
    r = optim(
        [-1.2, 1],
        _frv,
        _grv,
        method="L-BFGS-B",
        lower=[-2, 0.3],
        upper=[0.5, 0.6],
        control={"factr": 1e7},
    )
    assert r["par"][0] == e[0]
    assert r["par"][1] == e[1]
    assert r["value"] == e[2]
    assert r["counts"] == {"function": int(e[3]), "gradient": int(e[4])}
    assert r["convergence"] == int(e[5])
    assert r["message"] == rv["optim_bnd_msg"]


def test_optim_maxit_and_method_validation():
    # convergence=1 when the iteration cap trips (R optim semantics)
    r = optim([-1.2, 1], _frv, _grv, method="L-BFGS-B", control={"maxit": 2})
    assert r["convergence"] == 1
    with pytest.raises(NotImplementedError, match="L-BFGS-B"):
        optim([0.0], _frv, _grv, method="BFGS")


def test_optimHess_matches_r(rv):
    # optimHess(c(-1.2, 1), frv, grv): bit-exact FD of the analytic
    # gradient with ndeps=1e-3 + symmetrization
    e = rv["opthess"]  # column-major
    h = optimHess([-1.2, 1], _frv, _grv)
    assert h[0, 0] == e[0]
    assert h[1, 0] == e[1]
    assert h[0, 1] == e[2]
    assert h[1, 1] == e[3]


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
