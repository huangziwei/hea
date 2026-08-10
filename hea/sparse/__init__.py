"""Sparse Cholesky factorization — a self-contained replacement for
``scikit-sparse``/CHOLMOD.

Every routine behind this module is a mechanical port of SuiteSparse 7.6.0, the
version R's ``Matrix`` ships and therefore the one ``lme4`` factorizes with.
Given the same fill-reducing ordering, the factor, its permutation and its
numeric values are bit-identical to what ``cholmod_l_analyze`` +
``cholmod_l_factorize_p`` + ``cholmod_l_solve`` produce — not equal to a
tolerance, equal in every bit.

The ordering is the same too. ``order="best"`` is CHOLMOD's default strategy —
try AMD, and try METIS only if AMD's own fill estimate says it is worth looking
further, then keep the smaller ``nnz(L)`` — and it selects the same method, with
the same permutation, as a CHOLMOD built with its Partition module.
``order="amd"`` and ``order="metis"`` pin one method; ``order="natural"``
applies no fill-reducing permutation at all.

METIS is not free: on a 3.4M-row system it costs seconds, where AMD costs
tenths. It earns that back in ``nnz(L)`` (−27% on that system), in solve time,
and in memory, so it pays for a factor that is solved against repeatedly or
refactorized through :meth:`Factor.factorize`; for a single factorize-and-solve
at that size, ``order="amd"`` finishes sooner.

**The numeric factorization is parallel, and it needs the cores.** It runs over
the supernodal elimination tree and again inside a supernode's panel, on a
private pool sized to the machine's performance cores (``RAYON_NUM_THREADS``
overrides it). That is where it gets its speed: CHOLMOD hands one supernode at a
time to a vendor BLAS and relies on the BLAS to thread, which on a matrix made
of many small supernodes it barely can. The other side of that is a real
single-thread cost — on a 3.4M-row system the same factorization is 16 s at one
thread and 3 s at eight — so pinning ``RAYON_NUM_THREADS=1`` for reproducibility,
or running in a one-core container, gives up most of the difference.

**On macOS, set ``MallocSpaceEfficient=1`` in the environment for a large
factorization.** A parallel factorization makes many large short-lived
allocations from every worker, and libmalloc's default per-thread magazines hold
the freed blocks rather than returning them: on the same 3.4M-row system that is
**2.7 GB of a 9.2 GB peak**, recovered at no measurable cost in time. It has to
be an environment variable — libmalloc reads it before ``main`` — so a library
cannot set it for you::

    MallocSpaceEfficient=1 python fit.py

The factor itself is sized by the matrix, not by the thread count: ``L`` is
4.7 GB there, and what hea holds beyond it is one workspace, one ``Map`` per
worker, and ``A``'s pattern.

The API mirrors the slice of ``sksparse.cholmod`` hea and pywarper use, with two
additions CHOLMOD has and scikit-sparse 0.5.0 does not expose:
``system="P"`` / ``"Pt"``, which is what any caller doing its own triangular
solve against :attr:`Factor.L` needs, since ``L @ L.T == A[p][:, p]`` rather
than ``A``.

    >>> F = cho_factor(A)          # analyze + numeric factorization
    >>> F.factorize(A2)            # new values, same pattern: reuses the analysis
    >>> x = F.solve(b)             # A \\ b
    >>> F.half_log_det()           # ½ log|det A|

This module imports numpy, ``scipy.sparse`` and the compiled extension, and
nothing else from hea.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_array, issparse

from hea._rs import CholFactor as _CholFactor

__all__ = ["CholmodError", "Factor", "cho_factor", "cho_solve"]

#: The systems :meth:`Factor.solve` accepts, as ``cholmod_solve`` names them.
#: ``D`` is the identity for an ``LL'`` factor, so ``LD``/``L`` and ``DLt``/``Lt``
#: name the same solve there.
SYSTEMS = ("A", "LDLt", "LD", "DLt", "L", "Lt", "D", "P", "Pt")


class CholmodError(Exception):
    """The matrix could not be factorized — it is not positive definite.

    Named for ``sksparse.cholmod.CholmodError`` so that callers catching that
    keep working after the switch.
    """


def _as_csc(A):
    """The input as a sorted CSC matrix with the index and value types the
    extension takes.

    The full symmetric matrix may be passed: ``stype`` tells the factorization
    which half is the stored one, and entries in the other half are ignored
    rather than folded in — the same contract as ``A->stype``. So there is no
    triangle to extract, which matters when refactorizing 742 times per fit.

    The three arrays are handed to the extension as **views**, not copies —
    ``cholmod_sparse`` is a view onto the caller's buffers and so is this. The
    exception is the index type: the port is ``int64`` throughout, one ``itype``
    the way each CHOLMOD build has one, while scipy uses ``int32`` below 2³¹
    nonzeros. That upcast is a real copy of ``indices`` per call — 197 MB on a
    3.4M-row system — and it is the price of the single-itype scope.
    """
    if not issparse(A):
        A = csc_array(np.asarray(A, dtype=np.float64))
    A = csc_array(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {A.shape}")
    A.sort_indices()
    return (
        A.shape[0],
        np.ascontiguousarray(A.indptr, dtype=np.int64),
        np.ascontiguousarray(A.indices, dtype=np.int64),
        np.ascontiguousarray(A.data, dtype=np.float64),
    )


class Factor:
    """A numeric Cholesky factorization, reusable for new values and repeated
    solves.

    Holds its own workspace, so a sequence of ``factorize``/``solve`` calls
    against one symbolic analysis pays for it once. That is the property
    ``gmm`` depends on and the one scikit-sparse 0.5.0 gives up. It is also what
    scikit-sparse 0.4.15 does — a ``Common`` per ``analyze`` call, kept on the
    returned factor — so the workspace is not an extra cost against that bar.

    It does **not** keep ``A``: the values are read straight out of the caller's
    arrays and the factorization keeps only ``A``'s pattern, which
    :attr:`L` needs to prune the supernodal factor.
    """

    __slots__ = ("_F", "_lower", "_n")

    def __init__(
        self, A, beta=0.0, *, lower=False, order="best", use_ll=True, supernodal=None
    ):
        n, indptr, indices, data = _as_csc(A)
        self._n = n
        self._lower = bool(lower)
        if supernodal is None:
            # The supernodal factorization is ``LL'`` and only ``LL'``, so a
            # caller who asked for ``LDL'`` has to get the simplicial one --
            # the two disagree about which matrices are factorizable at all.
            supernodal = "auto" if use_ll else "simplicial"
        self._F = _CholFactor(
            n,
            indptr,
            indices,
            data,
            -1 if lower else 1,
            float(beta),
            order,
            bool(use_ll),
            supernodal,
        )
        self._check()

    def _check(self):
        if self._F.minor < self._n:
            raise CholmodError(
                "matrix is not positive definite: the leading minor of order "
                f"{self._F.minor + 1} is not"
            )

    def factorize(self, A, beta=0.0) -> None:
        """Refactorize ``A`` against the same symbolic analysis.

        ``A`` must have the same shape and its pattern must be contained in the
        one this factor was analyzed on — which is what lets the analysis, by
        far the expensive half, be paid for once across a sequence of
        refactorizations.

        Containment rather than equality, because a caller that builds ``A`` as
        a product does not control its pattern: an entry that comes out
        numerically zero is simply not emitted, so the pattern can shrink from
        one call to the next. Those are filled back in as explicit zeros, which
        changes no arithmetic. A pattern that has *grown* raises instead of
        silently dropping the new entries.
        """
        n, indptr, indices, data = _as_csc(A)
        if n != self._n:
            raise ValueError(f"A is {n}-by-{n}, expected n = {self._n}")
        self._F.refactorize(indptr, indices, data, float(beta))
        self._check()

    def solve(self, b, system="A"):
        """``A \\ b`` for ``b`` of shape ``(n,)`` or ``(n, k)``.

        ``system`` selects which factor to solve against; see :data:`SYSTEMS`.
        """
        b = np.asarray(b, dtype=np.float64)
        if b.ndim > 2:
            raise ValueError(f"b must be 1- or 2-D, got {b.ndim}-D")
        if b.shape[0] != self._n:
            raise ValueError(f"b has {b.shape[0]} rows, expected n = {self._n}")
        flat = b.ndim == 1
        b2 = b.reshape(self._n, 1) if flat else b
        nrhs = b2.shape[1]
        # cholmod_dense is column-major; ravel(order="F") is that layout flat
        x = self._F.solve(np.ravel(b2, order="F"), nrhs, system)
        x = x.reshape((self._n, nrhs), order="F")
        return x[:, 0] if flat else x

    def half_log_det(self) -> float:
        """``½ log|det A|``."""
        return float(self._F.half_log_det())

    @property
    def L(self):
        """The factor as a sparse lower-triangular matrix, in the permuted
        ordering: ``L @ L.T == A[p][:, p]`` for ``p = self.P``.

        For an ``LDL'`` factorization the diagonal holds ``D``, not ones — the
        same convention CHOLMOD's ``L`` uses.

        The supernodal factor is **pruned** on the way out, so this is the same
        matrix whichever path ran. ``scikit-sparse`` does not prune — its
        ``L()`` is ``cholmod_factor_to_sparse`` alone — and so returns the extra
        entries relaxed supernode amalgamation put in the dense blocks. Pruning
        is why the factorization keeps ``A``'s pattern.
        """
        indptr, indices, data = self._F.factor_csc()
        out = csc_array((data, indices, indptr), shape=(self._n, self._n))
        out.sort_indices()
        return out

    @property
    def P(self):
        """The fill-reducing permutation ``p``, with ``L @ L.T == A[p][:, p]``."""
        return np.asarray(self._F.perm, dtype=np.intp)

    #: ``scikit-sparse`` spells the permutation ``perm``; both names work.
    perm = P

    @property
    def n(self) -> int:
        return self._n

    @property
    def order(self) -> str:
        """Which fill-reducing ordering was used — never ``"best"``.

        With ``order="best"`` the analysis tries AMD and, if AMD's own fill
        estimate says it is worth looking further, METIS — keeping the one with
        the smaller ``nnz(L)`` — so this reports what it settled on. The
        difference is not cosmetic: on a crossed random-effects matrix METIS
        gives 41% less fill than AMD, and on a 3.4M-row conformal system 27%.
        """
        return self._F.ordering

    @property
    def is_ll(self) -> bool:
        """``LL'`` if set, ``LDL'`` if not."""
        return bool(self._F.is_ll)

    @property
    def is_super(self) -> bool:
        """Whether the supernodal factorization ran.

        Chosen the way CHOLMOD chooses it — by the flops-per-nonzero of the
        analysis — and worth 4x on the matrices that trip it. It changes no
        answer: :attr:`L`, :meth:`solve` and :meth:`half_log_det` are the same
        either way.
        """
        return bool(self._F.is_super)

    @property
    def nnz(self) -> int:
        """Entries in ``L``."""
        return int(self._F.nnz)

    def __repr__(self) -> str:
        return (
            f"<hea.sparse.Factor n={self._n} nnz={self.nnz} "
            f"{'LL' if self.is_ll else 'LDL'}' "
            f"{'lower' if self._lower else 'upper'}>"
        )


def cho_factor(
    A, beta=0.0, *, lower=False, order="best", use_ll=True, supernodal=None
) -> Factor:
    """Factorize ``beta*I + A`` and return a reusable :class:`Factor`.

    ``lower`` selects which triangle of ``A`` is the stored half, matching
    ``sksparse.cholmod.cho_factor``'s argument of the same name.

    ``order`` is the fill-reducing ordering: ``"best"`` (the default) is
    CHOLMOD's ``Common->nmethods == 0`` strategy, AMD then METIS, keeping the
    smaller ``nnz(L)``; ``"amd"``, ``"metis"`` and ``"natural"`` pin one.
    Pin only when you know what you are pinning — AMD is not reliably the
    better of the two, and picking wrong costs 2x on a `gmm` fit — except that
    ``"amd"`` is the right choice for one large factorize-and-solve, where
    METIS's own cost outruns the fill it saves. :attr:`Factor.order` reports
    what ``"best"`` chose.

    ``use_ll`` defaults to ``True`` here, where CHOLMOD's own default is
    ``LDL'``, because that is what ``sksparse.cholmod.cho_factor`` does and this
    is its replacement: it makes :attr:`Factor.L` the Cholesky factor with unit
    ``D``, and it makes a merely *indefinite* matrix an error rather than a
    successful factorization. ``rowfac`` fails an ``LL'`` on any non-positive
    pivot but an ``LDL'`` only on a zero one
    (``t_cholmod_rowfac_worker.c:424``), so the two forms disagree about what
    "not positive definite" means. Pass ``use_ll=False`` for the ``LDL'``.
    """
    return Factor(
        A, beta, lower=lower, order=order, use_ll=use_ll, supernodal=supernodal
    )


def cho_solve(A, b, beta=0.0, *, lower=False, order="best"):
    """One-shot ``A \\ b`` — analyze, factorize and solve.

    For repeated solves against one matrix, or repeated factorizations of one
    pattern, build a :class:`Factor` instead: this throws the analysis away.
    """
    return cho_factor(A, beta, lower=lower, order=order).solve(b)
