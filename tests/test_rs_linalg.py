"""T4 spike artifact — deterministic dense Cholesky (`hea._rs.chol_lower`).

A mechanical port of reference LAPACK `dpotf2` (lower) with strict in-order
accumulation. This is NOT wired into any fit path: the T4 spike measured that a
portable deterministic kernel matches R's `chol()` only at n=2 and diverges
1e-16..1e-11 for n≥5 (R routes through OpenBLAS `ddot`/`dgemv`, whose SIMD
reductions match neither this kernel nor Accelerate) — so it cannot make fits
0-ulp to an OpenBLAS-linked R (plan §7.3, go/no-go = NO-GO for bit-exactness).

What the kernel *does* guarantee, and what these tests pin: it is a faithful
Cholesky (L Lᵀ == A) and **bit-identical across runs** (the property optimized
BLAS does not promise) — the building block if the BLAS-flake-elimination path is
ever pursued.
"""

import numpy as np
import pytest

rs = pytest.importorskip("hea._rs")


def _spd(n, seed):
    g = np.random.default_rng(seed)
    m = g.standard_normal((n, n))
    a = m @ m.T + n * np.eye(n)
    return (a + a.T) / 2.0


@pytest.mark.parametrize("n", [1, 2, 5, 20, 100, 300])
def test_chol_lower_reconstructs(n):
    a = _spd(n, n)
    lo = np.asarray(rs.chol_lower(a))
    # lower-triangular
    assert np.allclose(np.triu(lo, 1), 0.0)
    # faithful factor
    assert np.abs(lo @ lo.T - a).max() / np.abs(a).max() < 1e-13


@pytest.mark.parametrize("n", [2, 50, 200])
def test_chol_lower_deterministic(n):
    """The whole point vs optimized BLAS: identical bits every run."""
    a = _spd(n, 2 * n + 1)
    first = np.asarray(rs.chol_lower(a))
    for _ in range(4):
        again = np.asarray(rs.chol_lower(a.copy()))
        assert (first.view(np.int64) == again.view(np.int64)).all()


def test_chol_lower_matches_numpy_within_tol():
    a = _spd(60, 99)
    lo = np.asarray(rs.chol_lower(a))
    np.testing.assert_allclose(lo, np.linalg.cholesky(a), rtol=0, atol=1e-10)


def test_chol_lower_rejects_non_pd():
    a = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    with pytest.raises(ValueError, match="not positive definite"):
        rs.chol_lower(a)
