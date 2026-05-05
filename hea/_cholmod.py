"""CHOLMOD compatibility shim.

Routes through ``sksparse.cholmod`` when scikit-sparse is installed (the fast
path used by ``hea.lme`` for the inner Cholesky of ``M = Λ Zᵀ Z Λᵀ + I``).
Falls back to ``scipy.sparse.linalg.splu`` otherwise — slower than CHOLMOD
because SuperLU re-runs symbolic analysis on every refactor and doesn't
exploit symmetry, but it preserves sparsity. A one-time ``UserWarning``
points users at ``hea[fast]``.

Both backends expose the slice of the API that ``hea.lme`` uses:

* ``factorize(M)`` — refactor with new numeric values
* ``solve(b)`` — solve ``M⁻¹ b``
* ``half_log_det()`` — ``½·log|det M|`` (CHOLMOD: ``Σ log diag L``;
  splu: ``½·Σ log|diag U|``, valid because ``M`` is SPD)
* ``L`` — sparse Cholesky factor for inspection. The sksparse path returns
  it directly; the splu fallback computes it lazily via dense Cholesky on
  first access (used once at end of fit, not in the hot loop).

``csc_array`` / ``eye_array`` are scipy.sparse symbols that ``sksparse.cholmod``
re-exports — we always pull them from scipy.sparse so they're available on
either path.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csc_array, eye_array

try:
    from sksparse.cholmod import (
        CholmodError as _SksparseCholmodError,
        cho_factor as _sks_cho_factor,
    )

    _HAS_SKSPARSE = True
except ImportError:
    _HAS_SKSPARSE = False


class CholmodError(Exception):
    """Raised when the Cholesky factor cannot be built (e.g. non-SPD matrix).

    Unifies ``sksparse.cholmod.CholmodError`` (fast path) and the SuperLU
    ``RuntimeError`` / non-positive-diagonal check (fallback) so callers can
    catch a single exception type regardless of which backend is in use.
    """


class _SksparseFactor:
    """Wraps an ``sksparse.cholmod`` factor with our unified API."""

    __slots__ = ("_F",)

    def __init__(self, M):
        try:
            self._F = _sks_cho_factor(M)
        except _SksparseCholmodError as e:
            raise CholmodError(str(e)) from e

    def factorize(self, M) -> None:
        try:
            self._F.factorize(M)
        except _SksparseCholmodError as e:
            raise CholmodError(str(e)) from e

    def solve(self, b):
        return self._F.solve(b)

    def half_log_det(self) -> float:
        return float(np.log(self._F.L.diagonal()).sum())

    @property
    def L(self):
        return self._F.L


class _SpluFactor:
    """``scipy.sparse.linalg.splu`` fallback — sparse LU on SPD matrices."""

    __slots__ = ("_M", "_lu", "_L_cache")

    def __init__(self, M):
        self._M = None
        self._lu = None
        self._L_cache = None
        self.factorize(M)

    def factorize(self, M) -> None:
        from scipy.sparse.linalg import splu

        M = M.tocsc() if hasattr(M, "tocsc") else M
        self._M = M
        try:
            self._lu = splu(M)
        except RuntimeError as e:
            raise CholmodError(str(e)) from e
        self._L_cache = None

    def solve(self, b):
        return self._lu.solve(b)

    def half_log_det(self) -> float:
        # |det M| = |det U| since L is unit-diagonal and the permutation
        # signs cancel for SPD M (det M > 0).
        return 0.5 * float(np.log(np.abs(self._lu.U.diagonal())).sum())

    @property
    def L(self):
        # Cholesky's L isn't directly available from SuperLU. We compute it
        # via dense Cholesky on first access — lme.py only touches .L once
        # per fit (storing the snapshot on the result), so this is cold-path.
        if self._L_cache is None:
            from scipy.linalg import cholesky as _scipy_cholesky

            M_dense = self._M.toarray()
            L_dense = _scipy_cholesky(M_dense, lower=True)
            self._L_cache = csc_array(L_dense)
        return self._L_cache


_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if _WARNED:
        return
    warnings.warn(
        "scikit-sparse is not installed; hea.lme is using a "
        "scipy.sparse.linalg.splu fallback. This is functional but slower "
        "than CHOLMOD for large mixed-effect models (no symbolic-analysis "
        "reuse across deviance evaluations). Install SuiteSparse "
        "(e.g. `apt install libsuitesparse-dev` or `brew install suite-sparse`) "
        "and `pip install scikit-sparse` (or `pip install hea[fast]`) for "
        "the fast path.",
        UserWarning,
        stacklevel=3,
    )
    _WARNED = True


def cho_factor(M):
    if _HAS_SKSPARSE:
        return _SksparseFactor(M)
    _warn_once()
    return _SpluFactor(M)


__all__ = ["CholmodError", "cho_factor", "csc_array", "eye_array"]
