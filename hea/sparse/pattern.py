"""Shared sparse patterns — hold one open across matrices whose values move.

:meth:`Factor.factorize` pays for the symbolic analysis once and reuses it for
every subsequent set of values, which is the whole reason :class:`Factor` is a
separate object from a solve. Reaching that idiom from scipy takes one step
scipy will not take for you: **every sparse operation drops entries that come
out exactly zero**, products as much as sums, so a matrix assembled from an
expression has whatever pattern that particular arithmetic happened to produce.
Step a coefficient and the pattern moves with it.

    >>> import numpy as np, scipy.sparse as sp
    >>> A = sp.csc_array(np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]))
    >>> (A.T @ A).nnz, (abs(A).T @ abs(A)).nnz
    (2, 4)

The two off-diagonal entries of ``A.T @ A`` cancel, so scipy does not store
them. Nothing is wrong with either matrix; they simply are not the same
pattern, and a factor analyzed on one cannot always refactorize the other.

Padding a pattern with explicit zeros does not fix it, which is the trap worth
naming: the padding is pruned on the way in, the analysis narrows back to one
matrix's own pattern, and the symptom is bit-identical results that are
slightly slower — a null result rather than a visible error.

A :class:`PatternPlan` is the pattern computed once, deliberately, from
operands that cannot cancel, plus the scatter that lays any matrix's values out
on it. **The values are always the caller's**: a plan carries a pattern and
nothing else, and :meth:`PatternPlan.materialize` therefore requires an array
to fill it with. That is not a stylistic choice — a matrix of structural zeros
is not a usable input to :func:`cho_factor`, which factorizes numerically as it
analyzes and rejects a zero leading minor, and the resulting error names the
minor rather than the empty values::

    plan = PatternPlan.union(AtA, RtR)
    a, r = plan.scatter(AtA), plan.scatter(RtR)
    M = plan.materialize(a + lam * lam * r)
    factor = cho_factor(M)
    for lam in candidates:
        M.data = a + lam * lam * r
        factor.factorize(M)             # one analysis, many refactorizations

**When this pays.** When the pattern is fixed by construction and only the
values move — a penalty weight stepped through a search, a parameter loop over
one fixed design. It does *not* pay when the pattern varies per problem,
however similar the problems look: two matrices built the same way from
different data have different cancellations, so the analysis is not shared and
holding it open only costs the wider pattern's arithmetic.

Reusing an analysis across a *wider* pattern than the values need is cheap —
refactorizing against one measured 0.527 s where an exact match measured
0.564 — so the conservative choice here is the cheap one.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_array, issparse

from hea import _rs

__all__ = ["PatternPlan"]


def _as_sorted_csc(M):
    """``M`` as a CSC matrix with sorted indices."""
    if not issparse(M):
        M = csc_array(np.asarray(M, dtype=np.float64))
    M = csc_array(M)
    if M.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {M.ndim}-D")
    if not M.has_sorted_indices:
        M = M.copy()
        M.sort_indices()
    return M


def _ones_like_pattern(M):
    """``M``'s pattern carrying 1.0 in every stored slot."""
    M = _as_sorted_csc(M)
    return csc_array((np.ones(M.nnz), M.indices, M.indptr), shape=M.shape)


class PatternPlan:
    """A sparsity pattern held open, with a scatter onto it.

    Build one with :meth:`union` or :meth:`of_product`, lay values out with
    :meth:`scatter`, and turn a value array into a matrix with
    :meth:`materialize`. The pattern is immutable; the values are not.
    """

    __slots__ = ("_indices", "_indptr", "_shape")

    def __init__(self, shape, indptr, indices):
        self._shape = shape
        self._indptr = indptr
        self._indices = indices

    @classmethod
    def _from_csc(cls, M) -> PatternPlan:
        """A plan holding a sorted CSC matrix's pattern, values discarded."""
        return cls(
            M.shape,
            np.ascontiguousarray(M.indptr, dtype=np.int64),
            np.ascontiguousarray(M.indices, dtype=np.int64),
        )

    @classmethod
    def union(cls, *mats) -> PatternPlan:
        """The union of several matrices' patterns.

        Computed by adding the operands' patterns **carrying ones**, never the
        operands themselves. ``A + B`` through scipy drops any entry where the
        two cancel exactly, which is the one case a caller building a union
        does not want; ones cannot cancel, so the same linear-time merge gives
        the structural answer instead of a data-dependent one.

            >>> import numpy as np, scipy.sparse as sp
            >>> A = sp.csc_array(np.array([[1.0, 0.0], [2.0, 3.0]]))
            >>> B = sp.csc_array(np.array([[0.0, 4.0], [-2.0, 0.0]]))
            >>> (A + B).nnz, PatternPlan.union(A, B).nnz
            (3, 4)
        """
        if not mats:
            raise ValueError("union needs at least one matrix")
        pats = [_ones_like_pattern(M) for M in mats]
        shape = pats[0].shape
        for P in pats[1:]:
            if P.shape != shape:
                raise ValueError(f"shapes differ: {shape} and {P.shape}")
        merged = pats[0]
        for P in pats[1:]:
            merged = merged + P
        return cls._from_csc(_as_sorted_csc(merged))

    @classmethod
    def of_product(cls, A) -> PatternPlan:
        """The pattern of ``A.T @ A``, with nothing cancelled.

        Computed as ``P.T @ P`` for ``P`` the pattern of ``A`` carrying ones,
        so every entry of the product is a positive count and none of them can
        prune. The result is a superset of any particular ``A.T @ A``'s
        pattern, which is the side of :meth:`Factor.factorize`'s containment
        contract a caller wants to be on.

            >>> import numpy as np, scipy.sparse as sp
            >>> A = sp.csc_array(np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]))
            >>> (A.T @ A).nnz, PatternPlan.of_product(A).nnz
            (2, 4)
        """
        P = _ones_like_pattern(A)
        return cls._from_csc(_as_sorted_csc(P.T @ P))

    @property
    def shape(self) -> tuple[int, int]:
        """The shape every matrix on this pattern has."""
        return self._shape

    @property
    def nnz(self) -> int:
        """Stored entries in the pattern, and the length :meth:`scatter`
        returns and :meth:`materialize` takes."""
        return int(self._indices.size)

    @property
    def indptr(self) -> np.ndarray:
        """The CSC column pointers, as a read-only view."""
        out = self._indptr.view()
        out.flags.writeable = False
        return out

    @property
    def indices(self) -> np.ndarray:
        """The CSC row indices, as a read-only view."""
        out = self._indices.view()
        out.flags.writeable = False
        return out

    def scatter(self, B) -> np.ndarray:
        """``B``'s values laid out on this pattern, zero where ``B`` has no
        entry.

        ``B``'s pattern must be contained in this one. It is checked rather
        than assumed, because the failure is silent otherwise: an entry with
        nowhere to go lands in a neighbouring slot and the result is a
        plausible matrix with two values in the wrong places. Containment holds
        by construction for the matrices a plan was built from, and stops
        holding the moment a caller passes a fourth one.

        Both patterns are CSC with sorted row indices, so this is a merge of
        the two — linear in the entries, with no per-entry key to build or
        hold.

            >>> import numpy as np, scipy.sparse as sp
            >>> A = sp.csc_array(np.array([[1.0, 0.0], [2.0, 3.0]]))
            >>> B = sp.csc_array(np.array([[0.0, 4.0], [-2.0, 0.0]]))
            >>> plan = PatternPlan.union(A, B)
            >>> plan.scatter(A)
            array([1., 2., 0., 3.])
        """
        B = _as_sorted_csc(B)
        if B.shape != self._shape:
            raise ValueError(f"B is {B.shape}, expected {self._shape}")
        out, missing = _rs.pattern_scatter(
            self._shape[1],
            self._indptr,
            self._indices,
            np.ascontiguousarray(B.indptr, dtype=np.int64),
            np.ascontiguousarray(B.indices, dtype=np.int64),
            np.ascontiguousarray(B.data, dtype=np.float64),
        )
        if missing:
            raise ValueError(
                f"B has {missing} entr{'y' if missing == 1 else 'ies'} "
                "outside the pattern"
            )
        return out

    def materialize(self, values) -> csc_array:
        """A CSC matrix on this pattern carrying ``values``.

        ``values`` is required, not defaulted: a matrix of structural zeros is
        not a usable input to :func:`cho_factor`, which factorizes numerically
        as it analyzes and rejects a zero leading minor. The matrix returned
        owns its values, so a caller stepping a parameter can assign to
        ``M.data`` in place and refactorize.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.nnz,):
            raise ValueError(f"values has shape {values.shape}, expected ({self.nnz},)")
        M = csc_array(
            (values.copy(), self._indices.copy(), self._indptr.copy()),
            shape=self._shape,
        )
        M.has_sorted_indices = True
        return M

    def __repr__(self) -> str:
        nrow, ncol = self._shape
        return f"<PatternPlan {nrow}x{ncol}, {self.nnz} stored>"
