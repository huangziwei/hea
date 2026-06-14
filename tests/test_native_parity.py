"""Differential parity gate: the compiled ``hea._native`` kernels must equal the
pure-Python ``hea.R.nmath`` kernels **bit-for-bit**.

``nmath.py`` is pinned 0-ulp to R (tests/test_R.py), so ``native == python`` here
transitively guarantees ``native == R`` without needing R in CI. If the native
extension is not built, these tests skip (the Python fallback still runs).

T0 covers ``pnorm`` only; extend per kernel as Tier 1 lands.
"""
import numpy as np
import pytest

from hea.R import nmath

# Skip the whole module when the extension isn't compiled in (sdist / no toolchain).
native = pytest.importorskip("hea._native")


def _bits(v: float) -> int:
    return np.float64(v).view(np.int64).item()


def _assert_bit_exact(got, exp):
    """Bit-for-bit equality, NaN- and signed-zero-aware."""
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    assert got.shape == exp.shape
    for g, e in zip(got.ravel(), exp.ravel()):
        if np.isnan(e):
            assert np.isnan(g), f"expected NaN, got {g!r}"
        else:
            assert _bits(g) == _bits(e), f"bit mismatch: native={g!r} python={e!r}"


def _grid() -> np.ndarray:
    # Stress every branch of pnorm_both: central (|x|<=0.6745), mid
    # (<=sqrt(32)), far tail (>sqrt(32)), the cutoffs that gate log_p/tail,
    # tiny (|x|<=eps), zero, and the non-finite lanes.
    pts = [
        -50.0, -40.0, -38.4674, -8.2924, -5.657, -5.0, -1.0, -0.6744,
        -1e-8, -1e-300, 0.0, 1e-300, 1e-8, 0.5, 0.6744, 0.67448975, 1.0,
        5.0, 5.657, 8.2924, 38.0, 40.0, 50.0, 1e170, 1e171,
        np.inf, -np.inf, np.nan,
    ]
    return np.array(pts, dtype=float)


@pytest.mark.parametrize("lower_tail", [True, False])
@pytest.mark.parametrize("log_p", [True, False])
@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.5, 2.0), (-3.0, 0.5)])
def test_pnorm_bit_exact(lower_tail, log_p, mu, sigma):
    x = _grid()
    got = native.pnorm(x, mu, sigma, lower_tail, log_p)
    exp = np.array(
        [nmath.pnorm5(float(xi), mu, sigma, lower_tail, log_p) for xi in x]
    )
    _assert_bit_exact(got, exp)


def test_pnorm_degenerate_sigma():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    for sig in (0.0, -1.0):
        for lt in (True, False):
            for lp in (True, False):
                got = native.pnorm(x, 0.0, sig, lt, lp)
                exp = np.array([nmath.pnorm5(float(xi), 0.0, sig, lt, lp) for xi in x])
                _assert_bit_exact(got, exp)
