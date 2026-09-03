"""``Factor.inv_diagonal`` / ``Factor.selected_inverse`` — the selected inverse.

Ported from ``SuiteSparse/MATLAB_Tools/sparseinv`` at 7.6.0. The oracle here is
a dense inverse and an ``n``-solve reconstruction, both of which catch the
failure this module is most exposed to: the sweep produces the selected inverse
of ``P A P'``, so an unpermute that is wrong agrees with upstream — upstream is
handed the already-permuted matrix — and only disagrees with ``A``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from hea.sparse import cho_factor


def _spd(n, seed=0, density=0.2):
    rng = np.random.default_rng(seed)
    B = sp.random_array((n, n), density=density, rng=rng, format="csc")
    return sp.csc_array(B @ B.T + sp.eye_array(n, format="csc") * (n * 0.05 + 1.0))


def _dense_inv(A):
    return np.linalg.inv(np.asarray(A.todense()))


@pytest.mark.parametrize("n", [1, 2, 5, 30, 120])
def test_inv_diagonal_matches_a_dense_inverse(n):
    A = _spd(n)
    got = cho_factor(A).inv_diagonal()
    want = np.diag(_dense_inv(A))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("order", ["amd", "metis", "natural", "best"])
def test_inv_diagonal_is_in_a_s_ordering_not_the_factors(order):
    A = _spd(60, seed=3)
    F = cho_factor(A, order=order)
    if order != "natural":
        assert not np.array_equal(F.P, np.arange(60)), "ordering is the identity"
    np.testing.assert_allclose(
        F.inv_diagonal(), np.diag(_dense_inv(A)), rtol=1e-9, atol=1e-12
    )


def test_a_permuted_answer_would_be_visibly_wrong():
    A = _spd(60, seed=3)
    d = np.diag(_dense_inv(A))
    assert d.max() / d.min() > 1.5


@pytest.mark.parametrize("n", [8, 40])
def test_inv_diagonal_matches_n_solves(n):
    A = _spd(n, seed=1)
    F = cho_factor(A)
    columns = F.solve(np.eye(n))
    np.testing.assert_allclose(
        F.inv_diagonal(), np.diag(columns), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("n", [5, 30, 120])
def test_selected_inverse_entries_match_a_dense_inverse(n):
    A = _spd(n)
    Z = cho_factor(A).selected_inverse()
    want = _dense_inv(A)
    Zc = Z.tocoo()
    np.testing.assert_allclose(Zc.data, want[Zc.row, Zc.col], rtol=1e-9, atol=1e-12)


def test_selected_inverse_is_symmetric():
    A = _spd(40, seed=2)
    Z = cho_factor(A).selected_inverse()
    np.testing.assert_allclose(
        np.asarray(Z.todense()), np.asarray(Z.T.todense()), rtol=1e-9, atol=1e-12
    )


def test_selected_inverse_covers_the_diagonal_and_agrees_with_inv_diagonal():
    A = _spd(50, seed=4)
    F = cho_factor(A)
    Z = F.selected_inverse()
    np.testing.assert_allclose(Z.diagonal(), F.inv_diagonal(), rtol=0, atol=0)


def test_selected_inverse_has_sorted_indices():
    Z = cho_factor(_spd(40, seed=5)).selected_inverse()
    assert Z.has_sorted_indices
    for j in range(Z.shape[1]):
        rows = Z.indices[Z.indptr[j] : Z.indptr[j + 1]]
        assert np.all(np.diff(rows) > 0)


def test_the_pattern_contains_a_s_own_pattern():
    A = _spd(60, seed=6)
    Z = cho_factor(A).selected_inverse()
    inside = set(zip(*sp.coo_array(Z).nonzero(), strict=True))
    for i, j in zip(*sp.coo_array(A).nonzero(), strict=True):
        assert (i, j) in inside


def test_the_trace_of_inv_a_times_b_is_exact_on_the_pattern():
    A = _spd(60, seed=7)
    Z = cho_factor(A).selected_inverse()
    B = sp.csc_array(A)  # any B whose pattern fits inside Z's
    exact = float(np.trace(_dense_inv(A) @ np.asarray(B.todense())))
    Zc, Bc = Z.tocsr(), B.tocsr()
    selected = float(sum((Zc.multiply(Bc.T)).data))
    assert abs(selected - exact) <= 1e-9 * abs(exact)


def test_the_supernodal_and_simplicial_paths_agree():
    A = _spd(150, seed=8, density=0.08)
    want = np.diag(_dense_inv(A))
    simp = cho_factor(A, supernodal="simplicial")
    sup = cho_factor(A, supernodal="supernodal")
    assert simp.is_super is False
    assert sup.is_super is True
    np.testing.assert_allclose(simp.inv_diagonal(), want, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sup.inv_diagonal(), want, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(simp.selected_inverse().todense()),
        np.asarray(sup.selected_inverse().todense()),
        rtol=1e-9,
        atol=1e-12,
    )


def test_an_ll_factor_is_converted_rather_than_refused():
    A = _spd(40, seed=9)
    F = cho_factor(A)
    assert F.is_ll is True
    np.testing.assert_allclose(
        F.inv_diagonal(), np.diag(_dense_inv(A)), rtol=1e-9, atol=1e-12
    )


def test_the_factor_still_solves_after_a_selected_inverse():
    A = _spd(40, seed=10)
    F = cho_factor(A)
    b = np.arange(1.0, 41.0)
    before = F.solve(b)
    F.inv_diagonal()
    assert F.is_ll is True
    np.testing.assert_allclose(F.solve(b), before, rtol=0, atol=0)


def test_it_survives_a_refactorization():
    A = _spd(40, seed=11)
    F = cho_factor(A)
    F.inv_diagonal()
    A2 = sp.csc_array(A * 2.0)
    F.factorize(A2)
    np.testing.assert_allclose(
        F.inv_diagonal(), np.diag(_dense_inv(A2)), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_systems(n):
    A = sp.csc_array(np.eye(n) * 3.0)
    F = cho_factor(A)
    np.testing.assert_allclose(F.inv_diagonal(), np.full(n, 1 / 3), rtol=1e-12)
    assert F.selected_inverse().nnz == n


def test_a_diagonal_matrix_has_no_off_diagonal_entries():
    A = sp.csc_array(sp.diags_array(np.arange(1.0, 11.0), format="csc"))
    Z = cho_factor(A).selected_inverse()
    assert Z.nnz == 10
    np.testing.assert_allclose(Z.diagonal(), 1.0 / np.arange(1.0, 11.0), rtol=1e-12)
