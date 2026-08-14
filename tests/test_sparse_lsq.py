"""``hea.sparse.lsqr`` / ``lsmr`` against the scipy implementations they port.

The port targets scipy 1.18.0's ``scipy.sparse.linalg.lsqr`` and ``lsmr``, which
are translations of the Paige-Saunders and Fong-Saunders originals. The
signatures, defaults and return tuples are scipy's, so the check is
element-for-element equality of the whole return, not a tolerance on ``x``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr as sp_lsmr
from scipy.sparse.linalg import lsqr as sp_lsqr

from hea.sparse import lsmr, lsqr

# Both algorithms are a sequence of scalar recurrences around two sparse
# products, so a faithful port reproduces every value exactly. Anything less
# than `==` here would hide a transposed rotation.
EXACT = {"rtol": 0.0, "atol": 0.0}


def _cases():
    """Least-squares problems spanning the shapes and ranks that branch."""
    rng = np.random.default_rng(11)
    out = {}

    A = sp.csc_array(sp.random_array((60, 20), density=0.3, rng=rng))
    out["overdetermined"] = (A, rng.standard_normal(60))

    A = sp.csc_array(sp.random_array((20, 60), density=0.3, rng=rng))
    out["underdetermined"] = (A, rng.standard_normal(20))

    A = sp.csc_array(sp.random_array((40, 40), density=0.2, rng=rng))
    out["square"] = (A, rng.standard_normal(40))

    # Consistent: b is exactly in the range of A, so it terminates on istop 1
    # rather than 2.
    A = sp.csc_array(sp.random_array((50, 15), density=0.4, rng=rng))
    out["consistent"] = (A, np.asarray(A @ rng.standard_normal(15)).ravel())

    # Rank deficient: two columns repeated.
    D = np.asarray(sp.random_array((30, 8), density=0.5, rng=rng).todense())
    D[:, 6] = D[:, 0]
    D[:, 7] = D[:, 1]
    out["rank_deficient"] = (sp.csc_array(D), rng.standard_normal(30))

    # Badly scaled, to exercise the condition-number tests.
    D = np.asarray(sp.random_array((40, 12), density=0.4, rng=rng).todense())
    D[:, 0] *= 1e-8
    out["ill_conditioned"] = (sp.csc_array(D), rng.standard_normal(40))

    # A dense ndarray rather than a sparse one.
    out["dense"] = (rng.standard_normal((25, 10)), rng.standard_normal(25))

    # b = 0, which returns before the loop.
    A = sp.csc_array(sp.random_array((20, 10), density=0.4, rng=rng))
    out["zero_rhs"] = (A, np.zeros(20))
    return out


CASES = _cases()
NAMES = sorted(CASES)


def _same(got, want, name):
    assert len(got) == len(want), name
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        np.testing.assert_allclose(g, w, err_msg=f"{name}[{i}]", **EXACT)


@pytest.mark.parametrize("name", NAMES)
def test_lsqr_matches_scipy(name):
    A, b = CASES[name]
    _same(lsqr(A, b), sp_lsqr(A, b), name)


@pytest.mark.parametrize("name", NAMES)
def test_lsmr_matches_scipy(name):
    A, b = CASES[name]
    _same(lsmr(A, b), sp_lsmr(A, b), name)


@pytest.mark.parametrize("damp", [0.0, 1e-3, 0.5, 10.0])
def test_lsqr_damped_matches_scipy(damp):
    A, b = CASES["overdetermined"]
    _same(lsqr(A, b, damp=damp), sp_lsqr(A, b, damp=damp), f"damp={damp}")


@pytest.mark.parametrize("damp", [0.0, 1e-3, 0.5, 10.0])
def test_lsmr_damped_matches_scipy(damp):
    A, b = CASES["overdetermined"]
    _same(lsmr(A, b, damp=damp), sp_lsmr(A, b, damp=damp), f"damp={damp}")


@pytest.mark.parametrize("tol", [1e-2, 1e-6, 1e-12, 0.0])
def test_tolerances_match_scipy(tol):
    A, b = CASES["overdetermined"]
    _same(lsqr(A, b, atol=tol, btol=tol), sp_lsqr(A, b, atol=tol, btol=tol), "lsqr")
    _same(lsmr(A, b, atol=tol, btol=tol), sp_lsmr(A, b, atol=tol, btol=tol), "lsmr")


@pytest.mark.parametrize("conlim", [1e2, 1e8, 0.0])
def test_conlim_matches_scipy(conlim):
    A, b = CASES["ill_conditioned"]
    _same(lsqr(A, b, conlim=conlim), sp_lsqr(A, b, conlim=conlim), "lsqr")
    _same(lsmr(A, b, conlim=conlim), sp_lsmr(A, b, conlim=conlim), "lsmr")


def test_x0_matches_scipy():
    A, b = CASES["overdetermined"]
    x0 = np.linspace(-1.0, 1.0, A.shape[1])
    _same(lsqr(A, b, x0=x0.copy()), sp_lsqr(A, b, x0=x0.copy()), "lsqr")
    _same(lsmr(A, b, x0=x0.copy()), sp_lsmr(A, b, x0=x0.copy()), "lsmr")


def test_calc_var_matches_scipy():
    A, b = CASES["overdetermined"]
    _same(lsqr(A, b, calc_var=True), sp_lsqr(A, b, calc_var=True), "lsqr")


def test_a_2d_b_is_squeezed_like_scipy():
    A, b = CASES["overdetermined"]
    _same(lsqr(A, b.reshape(-1, 1)), sp_lsqr(A, b.reshape(-1, 1)), "lsqr")
    _same(lsmr(A, b.reshape(-1, 1)), sp_lsmr(A, b.reshape(-1, 1)), "lsmr")


def test_csr_and_coo_inputs_agree_with_csc():
    A, b = CASES["overdetermined"]
    ref = lsqr(A, b)
    _same(lsqr(sp.csr_array(A), b), ref, "csr")
    _same(lsqr(sp.coo_array(A).tocsr(), b), ref, "coo")


# --- the contract that is hea's, not scipy's ----------------------------------


def test_the_iteration_limit_is_reported_not_raised():
    # Non-convergence is a normal outcome for an iterative solver. It must
    # arrive as istop, and it must not be an exception of any kind.
    A, b = CASES["ill_conditioned"]
    x, istop, itn = lsqr(A, b, atol=0.0, btol=0.0, conlim=0.0, iter_lim=3)[:3]
    assert istop == 7
    assert itn == 3
    assert np.all(np.isfinite(x))

    x, istop, itn = lsmr(A, b, atol=0.0, btol=0.0, conlim=0.0, maxiter=3)[:3]
    assert istop == 7
    assert itn == 3
    assert np.all(np.isfinite(x))


def test_no_cholmod_error_escapes_a_singular_system():
    # A direct solve raises on a rank-deficient system; these must not, since
    # the least-squares answer is well defined and is what was asked for.
    A, b = CASES["rank_deficient"]
    for solver in (lsqr, lsmr):
        x, istop = solver(A, b)[:2]
        assert istop in (1, 2, 3, 4, 5, 6, 7)
        assert np.all(np.isfinite(x))


def test_the_first_two_returns_are_x_and_istop():
    # `lsqr(...)[:2]` is the shape callers unpack; pinning it keeps the swap
    # with scipy a one-line import change.
    A, b = CASES["overdetermined"]
    x, istop = lsqr(A, b)[:2]
    assert x.shape == (A.shape[1],)
    assert isinstance(istop, int)


def test_normal_equations_and_lsqr_reach_the_same_least_squares_answer():
    A, b = CASES["overdetermined"]
    dense = np.asarray(A.todense())
    exact = np.linalg.lstsq(dense, b, rcond=None)[0]
    for solver, kw in ((lsqr, {"iter_lim": 500}), (lsmr, {"maxiter": 500})):
        x = solver(A, b, atol=1e-14, btol=1e-14, conlim=1e15, **kw)[0]
        np.testing.assert_allclose(x, exact, rtol=1e-6, atol=1e-8)


def test_a_matvec_only_operator_is_accepted():
    # No scipy LinearOperator involved: anything with shape, matvec and rmatvec
    # works, which is what keeps this module free of scipy.sparse.linalg.
    A, b = CASES["overdetermined"]

    class Op:
        shape = A.shape
        dtype = np.dtype(np.float64)

        def matvec(self, x):
            return A @ x

        def rmatvec(self, x):
            return A.T @ x

    _same(lsqr(Op(), b), lsqr(A, b), "operator")
    _same(lsmr(Op(), b), lsmr(A, b), "operator")


def test_a_3d_input_is_rejected():
    for solver in (lsqr, lsmr):
        with pytest.raises(ValueError, match="unsupported"):
            solver(np.zeros((2, 2, 2)), np.zeros(2))
