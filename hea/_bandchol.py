"""mgcv's ``bandchol`` (misc.r): the Cholesky factor of a symmetric
positive definite banded matrix through the C ``band_chol`` (mat.c), which
calls R's bundled LAPACK ``dpbtrf`` on lower band storage. With fewer than
32 off-diagonals ``dpbtrf`` runs ``DPBTF2``: per column a square root, then
BLAS ``DSCAL`` by its reciprocal and the rank-one ``DSYR`` update, which R's
BLAS (Accelerate on arm64 macOS) forms as one fused multiply-add per entry.
"""

from __future__ import annotations

import math

import numpy as np

from ._rfma import _rfma


def _dpbtf2_lower(ab: np.ndarray) -> tuple[np.ndarray, int]:
    """LAPACK ``DPBTF2`` with ``UPLO = 'L'``: ``ab`` is the ``(kd+1) x n`` lower
    band storage (row ``r`` holds the ``r``-th subdiagonal, ``ab[r, j] =
    A[j+r, j]``); returns the factor ``L`` in the same storage and ``INFO``."""
    ab = np.array(ab, dtype=float)
    kd = ab.shape[0] - 1
    n = ab.shape[1]
    for j in range(n):
        ajj = ab[0, j]
        if ajj <= 0.0:
            return ab, j + 1
        ajj = math.sqrt(ajj)
        ab[0, j] = ajj
        kn = min(kd, n - j - 1)
        if kn > 0:
            ab[1 : 1 + kn, j] = ab[1 : 1 + kn, j] * (1.0 / ajj)
            x = ab[1 : 1 + kn, j]
            for col in range(kn):
                if x[col] != 0.0:
                    temp = -1.0 * x[col]
                    for row in range(col, kn):
                        ab[row - col, j + 1 + col] = _rfma(
                            x[row], temp, ab[row - col, j + 1 + col]
                        )
    return ab, 0


def bandchol(B: np.ndarray) -> np.ndarray:
    """mgcv's ``bandchol(B)``: ``B`` holds the non-zero diagonals of a banded
    matrix by row (leading diagonal first, later ones zero padded at the
    end); the factor's diagonals come back the same way. A square ``B`` is
    read as the full matrix: its superdiagonals are packed into band rows
    first and the factor is returned as a square matrix, as mgcv does."""
    B = np.asarray(B, dtype=float)
    n = B.shape[1]
    k = 0
    if n == B.shape[0]:
        A = np.zeros_like(B)
        for i in range(1, n + 1):
            b = np.diag(B, i - 1)
            if np.any(b != 0):
                k = i
                A[i - 1, : b.size] = b
        B = A[:k, :]
    if B.shape[0] - 1 >= 32:
        raise NotImplementedError("bandchol: the blocked DPBTRF (>= 32 bands)")
    out, info = _dpbtf2_lower(B)
    if info > 0:
        raise ValueError("not positive definite")
    if k > 0:
        A = np.zeros((n, n))
        for i in range(1, k + 1):
            idx = np.arange(n - i + 1)
            A[idx, idx + i - 1] = out[i - 1, : n - i + 1]
        out = A
    return out
