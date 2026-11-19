"""R's ``optim(method = "L-BFGS-B")`` on problems of 3 and 4 parameters, from
``tests/fixtures/lbfgsb`` (written by ``tests/scripts/make_lbfgsb_fixtures.R``).
With three or more stored corrections the ``2 col``-term dot products reach
five terms, which R hands to its BLAS. Under the arithmetic the fixtures were
written with (``PINNED_ARITHMETIC``) hea evaluates the objective at R's
points, to the bit; elsewhere it reaches R's optimum.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURE_ROOT, PINNED_ARITHMETIC, load_hex_fixture

from hea.R.optimize import optim

_DIR = FIXTURE_ROOT / "lbfgsb"
# off the pinned arithmetic: the optimum, relative to its scale
_OFF_PAR = 1e-9


def _quartic3():
    def fn(x):
        a = x[0] - 1
        b = x[1] + 2
        c3 = x[2] - 3
        e = x[0] * x[1] - 1
        return float(
            ((((a * a) * (a * a) + (b * b) * (b * b)) + (c3 * c3) * (c3 * c3)) + e * e)
            + 0.5 * x[2]
        )

    def gr(x):
        a = x[0] - 1
        b = x[1] + 2
        c3 = x[2] - 3
        e = x[0] * x[1] - 1
        return np.array(
            [
                4 * ((a * a) * a) + 2 * (e * x[1]),
                4 * ((b * b) * b) + 2 * (e * x[0]),
                4 * ((c3 * c3) * c3) + 0.5,
            ]
        )

    return fn, gr


def _rosen4():
    def fn(x):
        s = 0.0
        for i in range(3):
            t = x[i + 1] - x[i] * x[i]
            u = 1 - x[i]
            s = s + (100 * (t * t) + u * u)
        return float(s)

    def gr(x):
        g = np.zeros(4)
        for i in range(3):
            t = x[i + 1] - x[i] * x[i]
            g[i] = g[i] + (-400 * x[i] * t - 2 * (1 - x[i]))
            g[i + 1] = g[i + 1] + 200 * t
        return g

    return fn, gr


_CASES = {
    "quartic3_m3": (_quartic3, [0.3, 0.2, -0.5], 3, None),
    "quartic3_m5": (_quartic3, [0.3, 0.2, -0.5], 5, None),
    "rosen4_m5": (_rosen4, [-1.2, 1.0, -0.5, 0.8], 5, None),
    "rosen4_box": (
        _rosen4,
        [-1.2, 1.0, -0.5, 0.8],
        4,
        ([-1.5, -1.0, -2.0, 0.0], [2.0, 2.0, 1.5, 3.0]),
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_lbfgsb_follows_r(case):
    """The points L-BFGS-B evaluates the objective at, and the optimum it
    reports, are R's."""
    make, x0, lmm, box = _CASES[case]
    fn0, gr = make()
    xs: list[np.ndarray] = []

    def fn(x):
        xs.append(np.array(x, dtype=float, copy=True))
        return fn0(x)

    kw = {}
    if box is not None:
        kw = {"lower": np.array(box[0]), "upper": np.array(box[1])}
    res = optim(
        np.array(x0),
        fn,
        gr,
        method="L-BFGS-B",
        control={"lmm": lmm, "fnscale": 2.5, "factr": 1e7},
        **kw,
    )
    want = load_hex_fixture(_DIR / f"{case}.txt.gz")
    par = np.ravel(want["par"])
    if PINNED_ARITHMETIC:
        np.testing.assert_array_equal(np.array(xs), np.asarray(want["xs"], dtype=float))
        np.testing.assert_array_equal(res["par"], par)
        assert res["value"] == float(np.ravel(want["value"])[0])
        return
    scale = max(1.0, float(np.max(np.abs(par))))
    np.testing.assert_allclose(res["par"], par, rtol=0, atol=_OFF_PAR * scale)
