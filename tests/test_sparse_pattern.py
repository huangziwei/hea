"""``hea.sparse.PatternPlan`` — a pattern held open across changing values.

The behaviour under test is mostly scipy's, negated: scipy prunes entries that
come out exactly zero, so a pattern derived from an expression is a property of
that expression's arithmetic. These check that a plan's pattern is not.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from hea.sparse import CholmodError, PatternPlan, cho_factor
from hea.sparse.pattern import _linear_index


def _dense(M):
    return np.asarray(M.todense() if sp.issparse(M) else M, dtype=float)


# --- what scipy prunes, and the plan does not --------------------------------


def test_union_keeps_what_a_scipy_add_cancels():
    A = sp.csc_array(np.array([[1.0, 0.0], [2.0, 3.0]]))
    B = sp.csc_array(np.array([[0.0, 4.0], [-2.0, 0.0]]))
    assert (A + B).nnz == 3
    assert PatternPlan.union(A, B).nnz == 4


def test_of_product_keeps_what_a_scipy_product_cancels():
    # The asymmetry blamed for this is not real: `A.T @ A` drops exact
    # cancellations exactly as `A + B` does. `of_product` works because its
    # operands are non-negative, not because multiplication is exempt.
    A = sp.csc_array(np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]))
    assert (A.T @ A).nnz == 2
    assert PatternPlan.of_product(A).nnz == 4


def test_of_product_survives_an_explicitly_stored_zero():
    # `abs(A).T @ abs(A)` is the near-miss: |0| is still 0, so a stored zero
    # produces a zero in the product and is pruned back out.
    A = sp.csc_array(
        (np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 1, 2])), shape=(2, 2)
    )
    assert A.nnz == 2
    assert (abs(A).T @ abs(A)).nnz == 1
    assert PatternPlan.of_product(A).nnz == 2


def test_of_product_is_a_superset_of_the_real_product():
    rng = np.random.default_rng(0)
    A = sp.random_array((40, 12), density=0.25, rng=rng, format="csc")
    plan = PatternPlan.of_product(A)
    G = sp.csc_array(A.T @ A)
    # containment: every entry of the real product is in the plan
    assert plan.scatter(G).shape == (plan.nnz,)
    assert plan.nnz >= G.nnz


# --- scatter ------------------------------------------------------------------


def test_scatter_then_materialize_round_trips():
    rng = np.random.default_rng(1)
    A = sp.csc_array(sp.random_array((20, 20), density=0.2, rng=rng))
    plan = PatternPlan.union(A)
    assert np.allclose(_dense(plan.materialize(plan.scatter(A))), _dense(A))


def test_scatter_is_zero_where_the_matrix_has_no_entry():
    A = sp.csc_array(np.array([[1.0, 0.0], [2.0, 3.0]]))
    B = sp.csc_array(np.array([[0.0, 4.0], [-2.0, 0.0]]))
    plan = PatternPlan.union(A, B)
    assert plan.scatter(A).tolist() == [1.0, 2.0, 0.0, 3.0]
    assert plan.scatter(B).tolist() == [0.0, -2.0, 4.0, 0.0]


def test_scatter_accepts_unsorted_and_non_csc_input():
    A = sp.csc_array(np.array([[1.0, 0.0], [2.0, 3.0]]))
    plan = PatternPlan.union(A)
    coo = sp.coo_array(A)
    csr = sp.csr_array(A)
    assert plan.scatter(coo).tolist() == plan.scatter(A).tolist()
    assert plan.scatter(csr).tolist() == plan.scatter(A).tolist()


def _unsorted_csc():
    """A CSC matrix whose column 0 lists row 1 before row 0."""
    M = sp.csc_array(
        (np.array([5.0, 4.0]), np.array([1, 0]), np.array([0, 2, 2])), shape=(2, 2)
    )
    assert not M.has_sorted_indices
    return M


def test_scatter_handles_unsorted_indices_without_touching_the_caller():
    plan = PatternPlan.union(sp.csc_array(np.array([[4.0, 0.0], [5.0, 0.0]])))
    M = _unsorted_csc()
    assert plan.scatter(M).tolist() == [4.0, 5.0]
    assert M.indices.tolist() == [1, 0]  # still the caller's own ordering


def test_union_handles_unsorted_indices_without_touching_the_caller():
    M = _unsorted_csc()
    plan = PatternPlan.union(M)
    assert plan.indices.tolist() == [0, 1]
    assert M.indices.tolist() == [1, 0]


def test_scatter_rejects_an_entry_outside_the_pattern():
    A = sp.csc_array(np.array([[1.0, 0.0], [0.0, 3.0]]))
    plan = PatternPlan.union(A)
    outside = sp.csc_array(np.array([[1.0, 7.0], [0.0, 3.0]]))
    with pytest.raises(ValueError, match="outside the pattern"):
        plan.scatter(outside)


def test_scatter_rejects_an_entry_past_the_last_slot():
    # searchsorted returns nnz for this one rather than a wrong slot, so it is
    # a separate arm of the guard from the case above.
    A = sp.csc_array(np.array([[1.0, 0.0], [0.0, 0.0]]))
    plan = PatternPlan.union(A)
    outside = sp.csc_array(np.array([[1.0, 0.0], [0.0, 5.0]]))
    with pytest.raises(ValueError, match="outside the pattern"):
        plan.scatter(outside)


def test_union_of_empty_matrices_is_empty():
    # The union deduplicates by comparing neighbours, which has no first
    # neighbour to compare when nothing is stored.
    empty = sp.csc_array((3, 3))
    plan = PatternPlan.union(empty, empty)
    assert plan.nnz == 0
    assert plan.indptr.tolist() == [0, 0, 0, 0]
    assert plan.materialize(np.zeros(0)).nnz == 0


def test_union_of_one_empty_and_one_full_matrix():
    A = sp.csc_array(np.array([[1.0, 0.0], [0.0, 3.0]]))
    plan = PatternPlan.union(A, sp.csc_array((2, 2)))
    assert plan.nnz == 2
    assert plan.scatter(A).tolist() == [1.0, 3.0]


def test_union_of_three_matrices():
    A = sp.csc_array(np.array([[1.0, 0.0], [0.0, 0.0]]))
    B = sp.csc_array(np.array([[0.0, 2.0], [0.0, 0.0]]))
    C = sp.csc_array(np.array([[0.0, 0.0], [0.0, 3.0]]))
    plan = PatternPlan.union(A, B, C)
    assert plan.nnz == 3
    assert np.allclose(
        _dense(plan.materialize(plan.scatter(A) + plan.scatter(B) + plan.scatter(C))),
        _dense(A) + _dense(B) + _dense(C),
    )


def test_union_matches_a_general_set_union():
    # The fast path assumes each operand's linear index is already ascending,
    # which `_linear_index` guarantees. If that assumption ever breaks the
    # answer diverges from the general routine, so pin them together.
    rng = np.random.default_rng(5)
    A = sp.csc_array(sp.random_array((50, 50), density=0.1, rng=rng))
    B = sp.csc_array(sp.random_array((50, 50), density=0.1, rng=rng))
    plan = PatternPlan.union(A, B)
    want = np.union1d(_linear_index(A), _linear_index(B))
    got = _linear_index(plan.materialize(np.ones(plan.nnz)))
    assert np.array_equal(got, want)


def test_scatter_rejects_a_shape_mismatch():
    plan = PatternPlan.union(sp.csc_array(np.eye(3)))
    with pytest.raises(ValueError, match="expected"):
        plan.scatter(sp.csc_array(np.eye(2)))


def test_scatter_of_an_empty_matrix_is_all_zeros():
    A = sp.csc_array(np.array([[1.0, 0.0], [0.0, 3.0]]))
    plan = PatternPlan.union(A)
    empty = sp.csc_array((2, 2))
    assert plan.scatter(empty).tolist() == [0.0, 0.0]


# --- materialize ---------------------------------------------------------------


def test_materialize_rejects_the_wrong_number_of_values():
    plan = PatternPlan.union(sp.csc_array(np.eye(3)))
    with pytest.raises(ValueError, match="expected"):
        plan.materialize(np.ones(2))


def test_materialize_keeps_explicit_zeros():
    A = sp.csc_array(np.array([[1.0, 0.0], [2.0, 3.0]]))
    B = sp.csc_array(np.array([[0.0, 4.0], [-2.0, 0.0]]))
    plan = PatternPlan.union(A, B)
    M = plan.materialize(plan.scatter(A) + plan.scatter(B))
    assert M.nnz == 4  # the (1, 0) entry is 2 + (-2) and is stored as zero
    assert np.allclose(_dense(M), _dense(A) + _dense(B))


def test_materialize_does_not_alias_the_plan():
    plan = PatternPlan.union(sp.csc_array(np.eye(3)))
    first = plan.materialize(np.ones(3))
    second = plan.materialize(np.full(3, 2.0))
    first.data[:] = 99.0
    assert second.data.tolist() == [2.0, 2.0, 2.0]
    assert plan.nnz == 3


def test_an_all_zero_pattern_is_not_factorizable():
    # Why `materialize` takes values rather than defaulting to zeros: CHOLMOD
    # factorizes numerically while it analyzes.
    plan = PatternPlan.union(sp.csc_array(np.eye(3)))
    with pytest.raises(CholmodError, match="not positive definite"):
        cho_factor(plan.materialize(np.zeros(3)))


def test_the_pattern_is_immutable_through_the_views():
    plan = PatternPlan.union(sp.csc_array(np.eye(3)))
    with pytest.raises(ValueError):
        plan.indices[0] = 7
    with pytest.raises(ValueError):
        plan.indptr[0] = 7


# --- the idiom the class exists for -------------------------------------------


def _penalized_system(n=24, seed=3):
    """A data term and a second-difference penalty on the same unknowns.

    Each row of ``A`` touches two columns, one of them far from the other, so
    ``AtA`` reaches outside the penalty's band and the union is strictly wider
    than either operand. Every column is touched at least once, so ``AtA`` has
    a zero-free diagonal.
    """
    rng = np.random.default_rng(seed)
    m = 3 * n
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=np.int64)
    cols[0::2] = np.arange(m) % n
    cols[1::2] = rng.integers(0, n, size=m)
    A = sp.csc_array((rng.standard_normal(2 * m), (rows, cols)), shape=(m, n))
    d = sp.diags_array(
        [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    return A, sp.csc_array(d)


def test_one_analysis_many_refactorizations_matches_factorizing_each():
    A, R = _penalized_system()
    AtA = sp.csc_array(A.T @ A)
    RtR = sp.csc_array(R.T @ R)
    plan = PatternPlan.union(AtA, RtR)
    a, r = plan.scatter(AtA), plan.scatter(RtR)
    rhs = np.arange(1.0, A.shape[1] + 1.0)

    lambdas = [1e-3, 1.0, 7.5, 250.0]
    M = plan.materialize(a + lambdas[0] * lambdas[0] * r)
    factor = cho_factor(M)
    for lam in lambdas:
        M.data = a + lam * lam * r
        factor.factorize(M)
        shared = factor.solve(rhs)
        fresh = cho_factor(sp.csc_array(AtA + lam * lam * RtR)).solve(rhs)
        assert np.allclose(shared, fresh, rtol=1e-10, atol=1e-12)


def test_the_shared_pattern_is_wider_than_any_one_lambda():
    # The point of the union: at no single lambda does the assembled matrix
    # carry every slot, so a factor analyzed on one lambda's matrix is not
    # guaranteed to take another's.
    A, R = _penalized_system()
    AtA = sp.csc_array(A.T @ A)
    RtR = sp.csc_array(R.T @ R)
    plan = PatternPlan.union(AtA, RtR)
    assert plan.nnz >= sp.csc_array(AtA + RtR).nnz
    assert plan.nnz > AtA.nnz
    assert plan.nnz > RtR.nnz


def test_a_factor_on_the_plan_takes_a_narrower_matrix():
    # `Factor.factorize` documents containment, not equality; the plan is how a
    # caller lands on the safe side of that on purpose.
    A, R = _penalized_system()
    AtA = sp.csc_array(A.T @ A)
    RtR = sp.csc_array(R.T @ R)
    plan = PatternPlan.union(AtA, RtR)
    factor = cho_factor(plan.materialize(plan.scatter(AtA) + plan.scatter(RtR)))
    narrower = sp.csc_array(AtA + 1e-6 * sp.eye_array(AtA.shape[0], format="csc"))
    assert narrower.nnz < plan.nnz
    factor.factorize(narrower)
    rhs = np.ones(AtA.shape[0])
    assert np.allclose(factor.solve(rhs), cho_factor(narrower).solve(rhs))


def test_of_product_supports_the_same_loop():
    A, _ = _penalized_system()
    plan = PatternPlan.of_product(A)
    G = sp.csc_array(A.T @ A)
    n = G.shape[0]
    ridge = sp.eye_array(n, format="csc")
    values = plan.scatter(G) + plan.scatter(sp.csc_array(ridge))
    M = plan.materialize(values)
    rhs = np.ones(n)
    assert np.allclose(
        cho_factor(M).solve(rhs), cho_factor(sp.csc_array(G + ridge)).solve(rhs)
    )
