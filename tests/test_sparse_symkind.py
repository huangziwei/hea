"""``sym_kind`` — factorizing a Gram matrix without forming it.

``cho_factor(A, sym_kind="col")`` factorizes ``AᵀA`` from ``A`` and ``Aᵀ``
directly, which is CHOLMOD's ``A->stype == 0``. The checks here are that it is
the *same* factorization as the explicitly-formed product, to the floor a
Cholesky of that matrix has, and that the shape plumbing follows the Gram
matrix rather than the input.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from hea.sparse import CholmodError, Factor, cho_factor, cho_solve


def _rect(m, n, seed=0, density=0.3):
    rng = np.random.default_rng(seed)
    A = sp.csc_array(sp.random_array((m, n), density=density, rng=rng))
    A = sp.csc_array(A + sp.eye_array(m, n, format="csc"))
    A.sort_indices()
    return A


def _gram(A, kind, ridge):
    G = (A.T @ A) if kind == "col" else (A @ A.T)
    n = G.shape[0]
    return sp.csc_array(G + ridge * sp.eye_array(n, format="csc"))


@pytest.mark.parametrize("kind", ["col", "row"])
@pytest.mark.parametrize("shape", [(60, 20), (20, 60), (40, 40), (1, 5), (5, 1)])
def test_the_factorization_matches_the_explicit_product(kind, shape):
    m, n = shape
    A = _rect(m, n, seed=m * 100 + n)
    ridge = 1.0
    G = _gram(A, kind, ridge)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(G.shape[0])

    got = cho_factor(A, beta=ridge, sym_kind=kind).solve(b)
    want = cho_factor(G).solve(b)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("kind", ["col", "row"])
def test_the_residual_is_the_gram_matrix_s(kind):
    A = _rect(50, 18, seed=7)
    ridge = 0.5
    G = _gram(A, kind, ridge)
    rng = np.random.default_rng(2)
    b = rng.standard_normal(G.shape[0])
    x = cho_factor(A, beta=ridge, sym_kind=kind).solve(b)
    np.testing.assert_allclose(np.asarray(G @ x).ravel(), b, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("order", ["amd", "metis", "natural", "best"])
def test_every_ordering_agrees(order):
    A = _rect(45, 22, seed=3)
    G = _gram(A, "col", 1.0)
    rng = np.random.default_rng(4)
    b = rng.standard_normal(22)
    got = cho_factor(A, beta=1.0, sym_kind="col", order=order).solve(b)
    want = cho_factor(G, order=order).solve(b)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("supernodal", ["simplicial", "supernodal", "auto"])
def test_both_factorization_paths_agree(supernodal):
    A = _rect(120, 60, seed=5, density=0.15)
    G = _gram(A, "col", 1.0)
    rng = np.random.default_rng(6)
    b = rng.standard_normal(60)
    got = cho_factor(A, beta=1.0, sym_kind="col", supernodal=supernodal).solve(b)
    want = cho_factor(G, supernodal=supernodal).solve(b)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)


def test_the_ldl_factorization_works_too():
    A = _rect(40, 15, seed=8)
    G = _gram(A, "col", 1.0)
    rng = np.random.default_rng(9)
    b = rng.standard_normal(15)
    F = cho_factor(A, beta=1.0, sym_kind="col", use_ll=False)
    assert F.is_ll is False
    np.testing.assert_allclose(
        F.solve(b), cho_factor(G, use_ll=False).solve(b), rtol=1e-9, atol=1e-11
    )


def test_n_is_the_gram_matrix_s_dimension():
    A = _rect(60, 20, seed=10)
    assert cho_factor(A, beta=1.0, sym_kind="col").n == 20
    assert cho_factor(A, beta=1.0, sym_kind="row").n == 60


def test_the_right_hand_side_is_checked_against_the_factored_dimension():
    A = _rect(60, 20, seed=11)
    F = cho_factor(A, beta=1.0, sym_kind="col")
    with pytest.raises(ValueError, match="expected n = 20"):
        F.solve(np.ones(60))
    assert F.solve(np.ones(20)).shape == (20,)


def test_L_is_the_gram_matrix_s_factor():
    A = _rect(50, 16, seed=12)
    F = cho_factor(A, beta=1.0, sym_kind="col")
    L = F.L
    assert L.shape == (16, 16)
    G = _gram(A, "col", 1.0)
    p = F.P
    np.testing.assert_allclose(
        np.asarray((L @ L.T).todense()),
        np.asarray(G.todense())[np.ix_(p, p)],
        rtol=1e-9,
        atol=1e-11,
    )


@pytest.mark.parametrize("supernodal", ["simplicial", "supernodal"])
def test_L_survives_both_factorization_paths(supernodal):
    A = _rect(200, 90, seed=24, density=0.12)
    F = cho_factor(A, beta=1.0, sym_kind="col", supernodal=supernodal)
    assert F.is_super is (supernodal == "supernodal")
    L = F.L
    assert L.shape == (90, 90)
    G = _gram(A, "col", 1.0)
    p = F.P
    np.testing.assert_allclose(
        np.asarray((L @ L.T).todense()),
        np.asarray(G.todense())[np.ix_(p, p)],
        rtol=1e-9,
        atol=1e-10,
    )


def test_a_rectangular_input_is_rejected_under_sym():
    A = _rect(60, 20, seed=13)
    with pytest.raises(ValueError, match="expected a square matrix"):
        cho_factor(A)
    with pytest.raises(ValueError, match="expected a square matrix"):
        cho_factor(A, sym_kind="sym")


def test_none_normalises_to_sym():
    G = _gram(_rect(30, 30, seed=14), "col", 1.0)
    rng = np.random.default_rng(15)
    b = rng.standard_normal(30)
    np.testing.assert_allclose(
        cho_factor(G, sym_kind=None).solve(b), cho_factor(G).solve(b), rtol=0, atol=0
    )


def test_an_unknown_sym_kind_is_rejected():
    with pytest.raises(ValueError, match="sym_kind must be one of"):
        cho_factor(_rect(10, 10, seed=16), sym_kind="colwise")


def test_refactorize_keeps_the_analysis():
    A = _rect(60, 20, seed=17)
    F = cho_factor(A, beta=1.0, sym_kind="col")
    rng = np.random.default_rng(18)
    b = rng.standard_normal(20)
    for scale in (1.0, 2.0, 0.5):
        A2 = sp.csc_array(A * scale)
        F.factorize(A2, beta=1.0)
        want = cho_factor(_gram(A2, "col", 1.0)).solve(b)
        np.testing.assert_allclose(F.solve(b), want, rtol=1e-9, atol=1e-11)


def test_cho_solve_takes_sym_kind():
    A = _rect(50, 18, seed=19)
    G = _gram(A, "col", 1.0)
    rng = np.random.default_rng(20)
    b = rng.standard_normal(18)
    np.testing.assert_allclose(
        cho_solve(A, b, beta=1.0, sym_kind="col"),
        cho_solve(G, b),
        rtol=1e-9,
        atol=1e-11,
    )


def test_a_rank_deficient_gram_matrix_still_raises():
    A = _rect(6, 20, seed=21)
    with pytest.raises(CholmodError, match="not positive definite"):
        cho_factor(A, sym_kind="col")


def test_the_factor_class_takes_it_directly():
    A = _rect(40, 12, seed=22)
    rng = np.random.default_rng(23)
    b = rng.standard_normal(12)
    F = Factor(A, 1.0, sym_kind="col")
    np.testing.assert_allclose(
        F.solve(b), cho_factor(_gram(A, "col", 1.0)).solve(b), rtol=1e-9, atol=1e-11
    )
