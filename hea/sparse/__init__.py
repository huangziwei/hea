"""Sparse Cholesky factorization — a self-contained replacement for
``scikit-sparse``/CHOLMOD.

Every routine behind this module is a mechanical port of SuiteSparse 7.6.0, the
version R's ``Matrix`` ships and therefore the one ``lme4`` factorizes with.
Given the same fill-reducing ordering, the factor and its permutation are
bit-identical to what ``cholmod_l_analyze`` + ``cholmod_l_factorize_p`` +
``cholmod_l_solve`` produce — not equal to a tolerance, equal in every bit.

The *values* carry one qualifier, and it is the same one CHOLMOD carries: a
supernodal factorization is mostly dense BLAS calls, so ``L``'s bits are the
BLAS's as much as the algorithm's, and two CHOLMOD builds on two BLAS libraries
do not agree either. hea links the same libraries CHOLMOD does — see below — and
against a CHOLMOD on the same one, every entry of ``L`` and of ``X`` is equal in
every bit.

**Which BLAS, and how to see it.** The dense kernels behind the factorization
are the platform's own — Accelerate on macOS, OpenBLAS on Linux and Windows, the
same ``scipy-openblas32`` binary numpy and scipy already carry, vendored into
the wheel so nothing is needed from the system. Calls too small to be worth the
vendor's dispatch stay on hea's own portable kernels, which is worth 8–22% of
the CPU on a sparse factorization made of many small supernodes.
:func:`build_info` reports which backend a given build has::

    >>> from hea.sparse import build_info
    >>> sorted(build_info())
    ['backend', 'blas', 'min_flops']
    >>> build_info()["backend"] in {"accelerate", "openblas", None}
    True

``backend`` is ``'openblas'`` on a Linux or Windows wheel, and ``None`` on a
build with no vendor BLAS: the Alpine (musllinux) wheel, which has no OpenBLAS
to vendor, and a source build on a machine where none was found. Those are
correct and somewhat slower, never wrong.

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
at that size, ``order="amd"`` finishes sooner — 5.2 s against 12.4.

**The break-even is somewhere between five and ten refactorizations**, and the
range is honest rather than vague: METIS's ordering cost is stable but the
saving per refactorize swings about 2× with how rested the machine's memory is,
so the durable quantity is the flop ratio (2.15× fewer, matching a 2.23×
measured refactorize) and not the seconds. Below that count pin ``order="amd"``;
above it take the default. The strategy cannot make this call for you because it
turns on how many numeric factorizations you will do, which is something you
know and the library does not.

**The numeric factorization is parallel, and it needs the cores.** It runs over
the supernodal elimination tree and again inside a supernode's panel, on a
private pool sized to the machine's performance cores (``RAYON_NUM_THREADS``
overrides it). That is where the wall-clock difference comes from: CHOLMOD hands
one supernode at a time to the BLAS and relies on the BLAS to thread, which on a
matrix made of many small supernodes it barely can, so it runs at a little over
one core however many are free. hea is 2.7–3.0× faster on wall clock than
CHOLMOD called from C, on everything from a 12k-row system to a 3.4M-row one.

It is not only the threads. Pinned to ``RAYON_NUM_THREADS=1`` — genuinely one
thread, hea's CPU there equals its wall clock — it is still 1.2–1.3× faster than
CHOLMOD called from C, which is using 1.2 cores for that comparison, and
1.3–1.6× faster than ``scikit-sparse``. So a one-core container or a pinned
deterministic run is slower than this machine, not slower than the library this
replaces.

The trade is CPU for wall clock, and past a point it stops being free. On the
small and medium systems hea also uses *less* total CPU (1.05–1.17×). On a
3.4M-row system it is ahead on CPU through two threads and behind past four —
0.7–0.9× at the thread counts a busy machine actually hands out — because the
memory bandwidth saturates and the same traffic then costs more CPU. That is a
band rather than a number on purpose: it is a point on a curve, and which point
depends on how many cores are free. Pinning ``RAYON_NUM_THREADS`` is how a
caller chooses; ``=1`` gives up most of the wall-clock difference and is the
cheapest per core.

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

The API mirrors the slice of ``sksparse.cholmod`` hea uses, with two additions
CHOLMOD has and scikit-sparse 0.5.0 does not expose:
``system="P"`` / ``"Pt"``, which is what any caller doing its own triangular
solve against :attr:`Factor.L` needs, since ``L @ L.T == A[p][:, p]`` rather
than ``A``.

    >>> F = cho_factor(A)          # analyze + numeric factorization  # doctest: +SKIP
    >>> F.factorize(A2)            # new values, same pattern: reuses the analysis  # doctest: +SKIP
    >>> x = F.solve(b)             # A \\ b  # doctest: +SKIP
    >>> F.half_log_det()           # ½ log|det A|  # doctest: +SKIP

:class:`PatternPlan` is what makes that second line reachable from scipy. Every
scipy sparse operation drops entries that come out exactly zero, so a matrix
assembled from an expression has whatever pattern that arithmetic produced, and
stepping a coefficient moves it. A plan computes the pattern once from operands
that cannot cancel and lays each matrix's values out on it, which is what turns
a sweep over a penalty weight into one analysis and many refactorizations.

This module imports numpy, ``scipy.sparse`` and the compiled extension, and
nothing else from hea.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_array, issparse

from hea._rs import CholFactor as _CholFactor
from hea._rs import build_info as _build_info
from hea.sparse.lsq import lsmr, lsqr
from hea.sparse.pattern import PatternPlan

__all__ = [
    "CholmodError",
    "Factor",
    "PatternPlan",
    "build_info",
    "cho_factor",
    "cho_solve",
    "lsmr",
    "lsqr",
]


def build_info() -> dict:
    """Which dense kernels this build's factorization uses.

    Three build-time constants, read once:

    ``backend``
        ``"accelerate"``, ``"openblas"``, or ``None`` for hea's own portable
        kernels. Which one a wheel has depends on the platform it was built
        for — see the module docstring.
    ``min_flops``
        The per-call flop count above which a dense kernel is handed to the
        vendor. Calls below it stay on hea's kernels, which is faster: a sparse
        factorization is mostly tiny calls and the vendor's dispatch costs more
        than they do. ``None`` when there is no vendor backend.
    ``blas``
        Whether the build *asked* for a vendor backend. This can be ``True``
        while ``backend`` is ``None``: the feature is on by default and a source
        build simply may not find an OpenBLAS, in which case the portable
        kernels are used rather than the build failing.

    This is public because the answer varies per wheel and changes performance
    by a factor of two on a large factorization, so "which one did I get" is a
    question a caller can reasonably need answered — in a bug report, or when a
    timing does not match a published one.

        >>> from hea.sparse import build_info
        >>> build_info()["backend"] in {"accelerate", "openblas", None}
        True
    """
    return _build_info()


SYSTEMS = ("A", "LDLt", "LD", "DLt", "L", "Lt", "D", "P", "Pt")


class CholmodError(Exception):
    """The matrix could not be factorized — it is not positive definite.

    Named for ``sksparse.cholmod.CholmodError`` so that callers catching that
    keep working after the switch.
    """


SYM_KINDS = ("sym", "row", "col")


def _parse_sym_kind(sym_kind):
    """``None`` normalises to ``"sym"``, as in ``scikit-sparse``'s
    ``cholmod.pyx``, so the two stay swappable."""
    if sym_kind is None:
        return "sym"
    if sym_kind not in SYM_KINDS:
        raise ValueError(f"sym_kind must be one of {SYM_KINDS}, not {sym_kind!r}")
    return sym_kind


def _as_csc(A, sym_kind="sym"):
    """The input as a sorted CSC matrix with the index and value types the
    extension takes.
    """
    if not issparse(A):
        A = csc_array(np.asarray(A, dtype=np.float64))
    A = csc_array(A)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {A.ndim}-D")
    if sym_kind == "sym" and A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {A.shape}")
    if sym_kind == "col":
        A = csc_array(A.T)
    A.sort_indices()
    return (
        A.shape[0],
        A.shape[1],
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

    __slots__ = ("_F", "_lower", "_n", "_sym_kind")

    def __init__(
        self,
        A,
        beta=0.0,
        *,
        lower=False,
        order="best",
        use_ll=True,
        supernodal=None,
        sym_kind=None,
    ):
        sym_kind = _parse_sym_kind(sym_kind)
        nrow, ncol, indptr, indices, data = _as_csc(A, sym_kind)
        self._n = nrow
        self._sym_kind = sym_kind
        self._lower = bool(lower)
        if supernodal is None:
            supernodal = "auto" if use_ll else "simplicial"
        self._F = _CholFactor(
            nrow,
            ncol,
            indptr,
            indices,
            data,
            0 if sym_kind != "sym" else (-1 if lower else 1),
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
        nrow, _ncol, indptr, indices, data = _as_csc(A, self._sym_kind)
        if nrow != self._n:
            raise ValueError(
                f"A has {nrow} rows after sym_kind={self._sym_kind!r}, "
                f"expected n = {self._n}"
            )
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
        x = self._F.solve(np.ravel(b2, order="F"), nrhs, system)
        x = x.reshape((self._n, nrhs), order="F")
        return x[:, 0] if flat else x

    def half_log_det(self) -> float:
        """``½ log|det A|``."""
        return float(self._F.half_log_det())

    def inv_diagonal(self) -> np.ndarray:
        """``diag(inv(A))``, without forming ``inv(A)``.

        Every standard error, hat-matrix diagonal and effective-degrees-of-
        freedom count in a penalized linear model is this quantity. Takahashi's
        recursion gets it from the factor alone, which is the difference
        between one factorization's worth of work and ``n`` triangular solves.

        **Budget two to three orders of magnitude more than a refactorize of
        the same matrix.** Two things compound. The recursion dot-products each
        column of ``L`` against itself, so the *work* is ``Σ_j |L_j|²`` — the
        factorization's flop count, not its nonzero count, and on a large
        system those differ by two orders of magnitude. And the sweep is
        **scalar** where the numeric factorization is blocked and threaded, so
        it does that work at a small fraction of the factorization's rate.

        It also holds roughly two ``L``s while it runs, since the entries it
        computes live on the pattern of ``L + L'``.

        So this is worth reaching for when the alternative is ``n`` triangular
        solves, or when an *exact* answer replaces a stochastic estimate whose
        noise is changing a decision. It is not worth reaching for to make an
        existing estimate cheaper.

        Returned in ``A``'s own ordering, not the factor's.
        """
        return self._F.inv_diagonal()

    def selected_inverse(self):
        """The entries of ``inv(A)`` on the pattern of ``L + L'``, as CSC.

        The off-diagonal companion to :meth:`inv_diagonal`, in ``A``'s
        ordering. What it is *for*: ``tr(inv(A) @ B)`` touches ``inv(A)`` only
        where ``B`` has entries, so any ``B`` whose pattern fits inside this one
        gives an **exact** trace from one sweep. That covers the
        ``tr(M⁻¹ AᵀA)`` an effective-degrees-of-freedom count needs, which is
        otherwise estimated with stochastic probes.

        **Where that exactness earns its cost**, given the price above: not
        wherever a probe estimate is noisy, but wherever what you compute
        *from* it is a difference of nearly equal numbers. A stochastic trace
        carries a few percent of relative error; catastrophic cancellation
        downstream turns that into unbounded relative error in the result,
        while a well-conditioned expression absorbs it and never notices.

        The case worth knowing, because it is easy to walk into: a GCV score
        ``n·RSS / (n − γ·edf)²`` near the interpolating end, where ``edf``
        approaches ``n``. At ``γ = 1`` the denominator is a cancellation of two
        numbers that agree to seven digits, so its condition number with
        respect to ``edf`` runs to ``10⁷`` and a few percent of trace error is
        *five to six orders of magnitude* larger than the quantity itself — the
        denominator is then not noisy, it is entirely noise, and even its sign
        is arbitrary. At ``γ = 1.2`` the same denominator is ``−0.2n``, its
        condition number is single digits, and the identical estimator with the
        identical error is perfectly adequate.

        So the test is not "is my trace noisy" but "how well conditioned is the
        arithmetic I feed it to". Where the answer is "badly", an exact trace
        replaces a coin flip with a determinate answer — which is not the same
        as a correct one. Inside the region described next it returns one wrong
        number rather than a spread of them, and a caller who reads
        determinism as correctness is worse off than with visible noise. Where
        the answer is "fine", this buys nothing probes do not already give.

        Exactness moves that cancellation rather than removing it. The trace is
        exact on the pattern but still computed in floating point, so a
        difference taken against it runs out of digits once it falls to the
        level of its own rounding error — around ``sqrt(eps)*n``, equivalently
        ``cond(A)*eps``. For the GCV case that is roughly six orders below
        where a stochastic trace fails, which is the entire gain.

        That floor grows with ``n``: it is near 3e-6 at ``n = 200`` and 6e-4 at
        ``n = 40,000``, so a search bracket that was safe on a small system can
        lie wholly inside it on a large one, and the same bracket then returns
        a repeatable answer with no digits behind it. Test ``n - edf`` at the
        end of the bracket against that floor rather than reusing a bracket
        because it worked before.

        Entries outside this pattern are not computed and are not zero; they
        are simply absent. Same cost and memory as :meth:`inv_diagonal`, which
        is the same sweep with only the diagonal kept.
        """
        indptr, indices, data = self._F.selected_inverse()
        out = csc_array((data, indices, indptr), shape=(self._n, self._n))
        out.sort_indices()
        return out

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
        gives 41% less fill than AMD, and 27% on a 3.4M-row mesh.
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
    A,
    beta=0.0,
    *,
    lower=False,
    order="best",
    use_ll=True,
    supernodal=None,
    sym_kind=None,
) -> Factor:
    """Factorize ``beta*I + A`` and return a reusable :class:`Factor`.

    ``lower`` selects which triangle of ``A`` is the stored half, matching
    ``sksparse.cholmod.cho_factor``'s argument of the same name.

    ``sym_kind`` selects **what** is factorized, and is the reason a
    rectangular ``A`` is accepted at all:

    ``"sym"`` (the default, and what ``None`` normalises to)
        ``A`` itself, which must be square.
    ``"row"``
        ``A @ A.T``, for an ``A`` of shape ``(m, n)``; the factor is ``m``-by-``m``.
    ``"col"``
        ``A.T @ A``; the factor is ``n``-by-``n``.

    **The product is never formed.** CHOLMOD factorizes ``A A'`` from ``A`` and
    ``A'`` directly — the explicit product exists only as a pattern for the
    fill-reducing ordering — so the normal equations cost no memory and the
    caller writes no ``A.T @ A``. The values of that product are never
    materialized in any array.

    What it does *not* save is a transpose: ``"col"`` needs ``A'`` in column
    form and takes it once per factorization, which trades a full-values
    transpose of ``A`` for the product it removes. Reach for this because it is
    the correct API and one fewer intermediate, not because it is faster.

    What it does not *preserve* is the ordering. The pattern this path hands the
    ordering is identical, entry for entry, to a formed product's — but the
    adjacency lists arrive in a different order, since ``cholmod_aat`` returns
    its columns unsorted, and both AMD and METIS break ties by that order. So
    the permutation differs from factorizing a product built outside, and with
    it ``nnz(L)`` in either direction and the last couple of digits of the
    solution. Neither permutation is the better one. A caller pinned to another
    implementation's output to the last digit should keep forming the product.

    ``order`` is the fill-reducing ordering: ``"best"`` (the default) is
    CHOLMOD's ``Common->nmethods == 0`` strategy, AMD then METIS, keeping the
    smaller ``nnz(L)``; ``"amd"``, ``"metis"`` and ``"natural"`` pin one.
    Pin only when you know what you are pinning — AMD is not reliably the
    better of the two, and picking wrong costs 2x on a `gmm` fit — except that
    ``"amd"`` is the right choice for one large factorize-and-solve, where
    METIS's own cost outruns the fill it saves. :attr:`Factor.order` reports
    what ``"best"`` chose.

    What that trade is worth, on a 3.4M-row mesh where the strategy selects
    METIS: the ordering costs 10.4 s against AMD's 0.64, and buys 380.5M
    nonzeros in ``L`` against 523.0M, so every subsequent
    :meth:`Factor.refactorize` does 2.2x less work. The up-front cost pays for
    itself somewhere between five and ten factorizations — the spread is real,
    since the numeric side of a system this size is memory-bound and depends on
    how much of a 3-4 GB factor stays resident. Below that ``"amd"`` wins the
    whole job, 2.4x faster for a single factorize-and-solve. The strategy cannot
    see how many times the factor will be reused, so on a large one-shot solve
    it is the caller who has to say. None of this is visible at small sizes,
    where the whole analysis is milliseconds either way.

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
        A,
        beta,
        lower=lower,
        order=order,
        use_ll=use_ll,
        supernodal=supernodal,
        sym_kind=sym_kind,
    )


def cho_solve(A, b, beta=0.0, *, lower=False, order="best", sym_kind=None):
    """One-shot ``A \\ b`` — analyze, factorize and solve.

    For repeated solves against one matrix, or repeated factorizations of one
    pattern, build a :class:`Factor` instead: this throws the analysis away.

    ``sym_kind`` is :func:`cho_factor`'s; under ``"row"`` or ``"col"`` the
    system solved is the Gram matrix's, so ``b`` has that many rows.
    ``scikit-sparse`` 0.5.0 puts ``sym_kind`` on its factories but not on
    ``cho_solve``; having it here is a deliberate superset, since a one-shot
    normal-equations solve is exactly the case this argument exists for.
    """
    return cho_factor(A, beta, lower=lower, order=order, sym_kind=sym_kind).solve(b)
