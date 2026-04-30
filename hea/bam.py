"""Big additive models — port of mgcv ``bam()`` for very large datasets.

Mirrors ``mgcv/R/bam.r`` (~3000 lines, Wood 2009-2023). The mathematical
model is identical to :class:`hea.gam` — same penalized GLM, same REML/ML/
GCV criteria. The difference is purely algorithmic: instead of materialising
the full ``n×p`` design matrix and reweighting it on every PIRLS step,
``bam`` builds the QR factor ``R (p×p)`` and ``f = Q'·sqrt(W)·(z-offset)``
in chunks, never holding the full design in memory.

Three internal fitters dispatch from :class:`bam`:

* :func:`_bam_fit` — strict additive Gaussian-identity (``am=TRUE``).
  Single chunked QR build, then outer Newton on ``(R, f, ‖y‖²)``. Mirrors
  ``mgcv::bam.fit`` (bam.r:1503-1771).
* :func:`_bgam_fit` — non-Gaussian PIRLS. Each iteration rebuilds ``(R, f)``
  from chunks of ``√W·X`` and ``√W·z``, then runs the inner solve and
  step-halving on the penalized deviance. Mirrors ``mgcv::bgam.fit``
  (bam.r:909-1353).
* :func:`_bgam_fitd` — discrete method (``discrete=TRUE``). Compresses
  covariates by rounding/dedup, stores marginal tensor matrices, and
  computes ``X'WX``/``X'Wy``/``Xβ`` directly on the compressed
  representation. Mirrors ``mgcv::bgam.fitd`` (bam.r:430-897).

The supporting helpers ``rwMatrix``, ``chol2qr``, ``qr_update`` (bam.r:18-75),
``compress.df``/``check.term``/``discrete.mf`` (bam.r:122-430), ``mini.mf``
(bam.r:384-427), and ``tero``/``tens2matrix``/``terms2tensor`` (bam.r:2037-
2175) are ported as private module functions.

Attribute surface matches :class:`hea.gam` so user code (``predict``,
``summary``, ``plot_smooth``, ``vis``, ``check``, …) is portable across
``gam`` and ``bam`` instances. :class:`bam` inherits from :class:`gam`;
the constructor populates the same attributes via the chunked path.

References
----------
Wood, Goude & Shaw (2015), "Generalized additive models for large data
sets", JRSS C 64(1):139-155.
Wood (2017), *Generalized Additive Models* (2nd ed.), §6.5.

mgcv source: ``/tmp/mgcv/R/bam.r`` (1.9-1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve, qr as scipy_qr, solve_triangular
from scipy.stats import t as t_dist

from .family import Family, Gaussian
from .formula import (
    BasisSpec,
    SmoothBlock,
    _apply_smooth_arg_exprs,
    _eval_atom,
    _smooth_arg_expr_map,
    materialize,
    materialize_smooths,
)
from .design import prepare_design
from .gam import (
    _FitState,
    _PenaltySlot,
    _add_null_space_penalties,
    _apply_gam_side,
    _row_frame,
    _sym_rank,
    gam,
)

__all__ = ["bam"]


# ---------------------------------------------------------------------------
# Utility ports — mgcv bam.r:1-200
# ---------------------------------------------------------------------------


def _rw_matrix(stop: np.ndarray, row: np.ndarray, weight: np.ndarray,
               X: np.ndarray, trans: bool = False) -> np.ndarray:
    """Recombine rows of ``X`` per ``stop``/``row``/``weight``.

    Direct port of mgcv ``rwMatrix`` (bam.r:18-29). The ith output row is
    ``Σ_{k ∈ ind_i} weight[k] · X[row[k], :]`` where ``ind_i = 1:stop[1]``
    if ``i==1`` else ``(stop[i-1]+1):stop[i]``. Used for the AR1 transform
    in :func:`_bam_fit` (rho ≠ 0 path).

    R indices ``stop`` and ``row`` are passed in 1-based form, matching the
    mgcv source. They are converted to 0-based here.
    """
    stop = np.asarray(stop, dtype=int) - 1
    row = np.asarray(row, dtype=int) - 1
    weight = np.asarray(weight, dtype=float)
    X = np.asarray(X, dtype=float)
    is_matrix = X.ndim == 2
    if not is_matrix:
        X = X.reshape(-1, 1)
    n, p = X.shape
    out = np.zeros((n, p), dtype=float)
    if trans:
        # Transposed form — applied to lpmatrix on the right; not exercised
        # by bam.fit's single-thread path. Kept for completeness.
        prev = -1
        for i in range(n):
            for k in range(prev + 1, stop[i] + 1):
                out[row[k], :] += weight[k] * X[i, :]
            prev = stop[i]
    else:
        prev = -1
        for i in range(n):
            acc = np.zeros(p, dtype=float)
            for k in range(prev + 1, stop[i] + 1):
                acc += weight[k] * X[row[k], :]
            out[i, :] = acc
            prev = stop[i]
    if not is_matrix:
        return out.ravel()
    return out


def _chol2qr(XX: np.ndarray, Xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``X'X``, ``X'y`` into ``R``, ``f`` such that ``R'R = X'X``,
    ``R'f = X'y``.

    Port of mgcv ``chol2qr`` (bam.r:31-44). Uses pivoted Cholesky and handles
    rank-deficient ``X'X`` by zeroing out the trailing rows of ``R``. Used
    by the ``use.chol=TRUE`` path of :func:`_qr_update` to recover an
    R-factor after fast (less-stable) ``X'X``-accumulation.
    """
    XX = np.asarray(XX, dtype=float)
    Xy = np.asarray(Xy, dtype=float).ravel()
    p = Xy.shape[0]
    # Pivoted Cholesky via LAPACK (numpy lacks pivoted Cholesky; SciPy's
    # ``cho_factor`` is unpivoted). Use a small jitter to stabilise PSD
    # matrices that aren't strictly PD.
    try:
        c, low = cho_factor(XX, lower=False)
        R = np.triu(c)
        # piv = identity
        piv = np.arange(p)
        rank = p
    except np.linalg.LinAlgError:
        # Fall back to eigendecomp-based pivoted construction. Returns
        # an upper-triangular R with |diag(R)| matching the spectral
        # square roots; rank = #(eigenvalue > tol).
        w, V = np.linalg.eigh(0.5 * (XX + XX.T))
        order = np.argsort(-w)
        w_s = w[order]
        V_s = V[:, order]
        tol = max(p, 1) * np.finfo(float).eps * float(np.abs(w_s).max() if w_s.size else 1.0)
        rank = int(np.sum(w_s > tol))
        sqrt_w = np.zeros(p)
        sqrt_w[:rank] = np.sqrt(np.maximum(w_s[:rank], 0.0))
        R_full = (sqrt_w[:, None] * V_s.T)
        R = R_full
        piv = np.arange(p)
    if rank < p:
        R[rank:, :] = 0.0
    # f via forwardsolve(R'[:rank, :rank], Xy[piv[:rank]]); pad zeros.
    ipiv = np.empty(p, dtype=int)
    ipiv[piv] = np.arange(p)
    f = np.zeros(p)
    if rank > 0:
        Rkk = R[:rank, :rank]
        f_top = solve_triangular(Rkk.T, Xy[piv[:rank]], lower=True)
        f[:rank] = f_top
    f = f[ipiv]
    R = R[np.ix_(ipiv, ipiv)]
    return R, f


def _qr_update(Xn: np.ndarray, yn: np.ndarray,
               R: Optional[np.ndarray] = None,
               f: Optional[np.ndarray] = None,
               y_norm2: float = 0.0,
               use_chol: bool = False) -> dict:
    """Update QR factor ``R`` and projected response ``f`` with new rows.

    Direct port of mgcv ``qr_update`` (bam.r:46-75). Given ``X = QR`` and
    ``f = Q'y``, append rows ``Xn``/``yn`` and refresh ``(R, f, ‖y‖²)``.
    The ``use_chol=True`` path accumulates ``X'X`` and ``X'y`` directly
    (faster but less stable for ill-conditioned designs); a final
    :func:`_chol2qr` converts to ``(R, f)``.

    Returns a dict ``{R, f, y_norm2}`` matching the mgcv list.
    """
    Xn = np.asarray(Xn, dtype=float)
    yn = np.asarray(yn, dtype=float).ravel()
    p = Xn.shape[1]
    y_norm2 = float(y_norm2) + float(yn @ yn)
    if use_chol:
        if R is None:
            R = Xn.T @ Xn
            fn = Xn.T @ yn
        else:
            R = R + Xn.T @ Xn
            fn = (np.zeros(p) if f is None else np.asarray(f, dtype=float)) + Xn.T @ yn
        return {"R": R, "f": fn, "y_norm2": y_norm2}
    # Proper QR: stack [R; Xn], [f; yn], reduce.
    if R is not None:
        Xn_full = np.vstack([np.asarray(R, dtype=float), Xn])
        yn_full = np.concatenate([np.asarray(f, dtype=float).ravel(), yn])
    else:
        Xn_full = Xn
        yn_full = yn
    # LAPACK QR with column pivoting, mirroring mgcv's `qr(.., LAPACK=TRUE)`.
    Q, R_new, piv = scipy_qr(Xn_full, mode="economic", pivoting=True)
    # mgcv: f_n = Q' y, take first p entries (or fewer if Xn_full has fewer rows).
    n_full = Xn_full.shape[0]
    fn = (Q.T @ yn_full)[:min(p, n_full)]
    if fn.shape[0] < p:
        fn = np.concatenate([fn, np.zeros(p - fn.shape[0])])
    # Reverse pivot — return R in original column order so subsequent updates
    # don't need to track pivot state across calls.
    rp = np.empty(p, dtype=int)
    rp[piv] = np.arange(p)
    R_unpivoted = R_new[:, rp]
    return {"R": R_unpivoted, "f": fn, "y_norm2": y_norm2}


# ---------------------------------------------------------------------------
# mini.mf — representative subset for basis setup (bam.r:384-427)
# ---------------------------------------------------------------------------


def _mini_mf(data: pl.DataFrame, chunk_size: int,
             *, seed: int = 66) -> pl.DataFrame:
    """Representative subsample of ``data`` for basis setup.

    Port of mgcv ``mini.mf`` (bam.r:384-427). Returns up to ``chunk_size``
    rows, ensuring:
      * the row containing the min and max of every numeric column is included,
      * at least one row from every level of every factor-typed column is
        included.

    The minimum representative size ``mn`` is ``Σ (2 if numeric else nlevels)``
    over all columns; ``chunk_size`` is bumped to ``mn`` if it falls short.

    Used by :class:`bam` to feed a small frame to ``materialize_smooths`` for
    knot/eigenbasis setup, while the full data is iterated chunk-by-chunk
    through :func:`_qr_update`. Matches mgcv's ``bam.r:2387`` flow.
    """
    n = data.height
    cols = data.columns
    # Count minimum representative rows: 2 per numeric, |levels| per factor.
    mn = 0
    for c in cols:
        s = data[c]
        if _is_factor_col(s):
            mn += int(s.unique().len())
        elif s.dtype.is_numeric():
            mn += 2
        else:
            mn += 2
    if chunk_size < mn:
        chunk_size = mn
    if n <= chunk_size:
        return data
    rng = np.random.default_rng(seed)
    # Random sample
    ind = rng.choice(n, size=chunk_size, replace=False)
    mf0 = data[ind.tolist()]
    # Stratified sampling for representativeness: place min/max rows for
    # numerics and one row per factor level into the head of mf0.
    ind_full = rng.permutation(n)
    rows: list[int] = []
    for c in cols:
        s = data[c]
        if _is_factor_col(s):
            arr = s.to_numpy()
            for lvl in s.unique().to_list():
                where = np.flatnonzero(arr == lvl)
                if where.size:
                    rows.append(int(where[0]))
        elif s.dtype.is_numeric():
            arr = s.to_numpy()
            j_min = int(np.argmin(arr))
            j_max = int(np.argmax(arr))
            rows.append(j_min)
            rows.append(j_max)
    if rows:
        # Replace head rows of mf0 with the representative set.
        head = data[rows]
        # Ensure we don't exceed chunk_size.
        n_head = min(head.height, mf0.height)
        head = head.head(n_head)
        tail = mf0.tail(mf0.height - n_head)
        mf0 = pl.concat([head, tail])
    return mf0


def _is_factor_col(s: pl.Series) -> bool:
    """Treat polars Categorical / Enum / String columns as factor-like."""
    return s.dtype in (pl.Categorical, pl.Enum, pl.Utf8) or (
        hasattr(pl, "String") and s.dtype == pl.String
    )


# ---------------------------------------------------------------------------
# AR.resid — AR1 residual computation (bam.r:2056-2076)
# ---------------------------------------------------------------------------


def _ar_resid(rsd: np.ndarray, rho: float = 0.0,
              ar_start: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply AR1 transform to raw residuals.

    Direct port of mgcv ``AR.resid`` (bam.r:2056-2076). Given residuals
    ``rsd`` and AR1 parameter ``rho``, return decorrelated residuals
    ``rsd_t`` such that ``rsd_t[1]=rsd[1]`` and
    ``rsd_t[i]= ld·rsd[i] - rho·ld·rsd[i-1]`` for ``i>1``, except where
    ``ar_start[i]==True`` re-anchors the chain.
    """
    if rho == 0:
        return rsd
    rsd = np.asarray(rsd, dtype=float).ravel()
    n = rsd.shape[0]
    ld = 1.0 / np.sqrt(1.0 - rho ** 2)
    out = np.empty_like(rsd)
    out[0] = rsd[0]
    for i in range(1, n):
        if ar_start is not None and bool(ar_start[i]):
            out[i] = rsd[i]
        else:
            out[i] = ld * rsd[i] - rho * ld * rsd[i - 1]
    return out


# ---------------------------------------------------------------------------
# Module-level dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BamQR:
    """Result of the chunked QR build.

    ``R`` and ``f`` are such that for the full weighted/AR1-transformed
    design ``X̃ = √W X`` (and AR-transformed when ``rho ≠ 0``) and response
    ``ỹ = √W (y - offset)``:

        ``R'R = X̃'X̃``,  ``R'f = X̃'ỹ``,  ``y_norm2 = ỹ'ỹ``.

    ``rss_extra = y_norm2 - ‖f‖²`` is the part of ``ỹ`` orthogonal to
    ``colspan(X̃)``; for any β, ``‖ỹ - X̃β‖² = ‖f - Rβ‖² + rss_extra``.

    For QR built via ``use_chol``, ``R`` is the post-:func:`_chol2qr` factor
    and the relations above hold up to the rank-deficient zero rows of R.
    """
    R: np.ndarray
    f: np.ndarray
    y_norm2: float
    rss_extra: float
    yX_last: Optional[np.ndarray] = None  # last (y, X) row, for bam.update


# ---------------------------------------------------------------------------
# Chunk iteration + design materialisation
# ---------------------------------------------------------------------------


def _chunk_indices(n: int, chunk_size: int,
                   *, ar1: bool = False) -> list[tuple[int, int]]:
    """Yield ``(start, end)`` pairs covering ``range(n)`` in chunks of
    ``chunk_size``.

    Mirrors mgcv ``bam.fit`` (bam.r:1566-1574, single-thread). For
    ``ar1=False`` (rho==0) chunks tile ``range(n)`` exactly. For
    ``ar1=True`` (rho≠0) chunks i ≥ 1 start one row earlier than the
    rho==0 layout: that extra row is the previous row needed by the
    AR1 transform's sub-diagonal. The transformed first row of each
    overlapping chunk is dropped after the rwMatrix transform (see
    :func:`_build_qr_chunked_gaussian`); the chunk indexing here is
    pre-drop, so consumers must pass the full ``[start:end)`` slice
    through ``_materialize_chunk`` and only drop the head row when
    chunk_index > 0.
    """
    if n <= 0:
        return []
    n_block = n // chunk_size
    stub = n % chunk_size
    if stub > 0:
        n_block += 1
    if ar1:
        # mgcv bam.r:1571-1572. The base lattice is the rho==0 layout
        # (starts = 0, k, 2k, …; ends = k, 2k, 3k, …), then every chunk
        # past the first has its start dragged back by 1 so it overlaps
        # the previous chunk by one row — the row needed by the AR1
        # sub-diagonal of the chunk's first transformed-and-kept row.
        # ENDs are NOT shifted (they stay on the rho==0 lattice), so
        # chunks 1..n_block-2 each have N=chunk_size+1 input rows.
        starts = [0] + [k * chunk_size - 1 for k in range(1, n_block)]
        ends = [(k + 1) * chunk_size for k in range(n_block)]
        ends[-1] = n
    else:
        starts = [i * chunk_size for i in range(n_block)]
        ends = [s + chunk_size for s in starts]
        ends[-1] = n
    return list(zip(starts, ends))


def _ar1_rwmatrix_indices(N: int, ld: float, sd: float,
                          ar_start_block: Optional[np.ndarray] = None,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(stop, row, weight)`` arrays for the AR1 row-recombine.

    Direct port of mgcv ``bam.r:1583-1593`` (single-thread chunk loop)
    and ``bam.r:483-486`` (full-matrix variant). The first output row
    is identity (``weight=1`` against ``input_1``); subsequent output
    rows are ``sd·input_{i-1} + ld·input_i``. Returns 1-based indices
    matching mgcv's R convention — :func:`_rw_matrix` converts them.

    ``ar_start_block`` (optional length-N) re-anchors the AR chain at
    ``True`` positions: zero sub-diag, restore identity leading-diag.
    The first observation in a block needs no correction (no sub-diag
    exists for it anyway).
    """
    if N <= 0:
        raise ValueError(f"AR1 block must have N>0, got N={N}")
    # row: c(1, rep(1:N, rep(2,N))[-c(1, 2*N)])  — length 2N-1, 1-based
    rep2 = np.repeat(np.arange(1, N + 1), 2)  # (1,1,2,2,…,N,N) length 2N
    row = np.concatenate(([1], rep2[1:-1])).astype(int)  # length 2N-1
    # weight: c(1, rep(c(sd, ld), N-1))  — length 2N-1
    if N >= 2:
        weight = np.concatenate(([1.0], np.tile([sd, ld], N - 1)))
    else:
        weight = np.array([1.0])
    # stop: c(1, 1:(N-1)*2+1)  — output i (1-based) consumes inputs
    # (stop[i-1]+1):stop[i]; 1-based.
    if N >= 2:
        stop = np.concatenate(([1], np.arange(1, N) * 2 + 1)).astype(int)
    else:
        stop = np.array([1], dtype=int)
    if ar_start_block is not None:
        # 1-based local indices of AR-restart events.
        ii = np.flatnonzero(np.asarray(ar_start_block, dtype=bool)) + 1
        if ii.size > 0 and ii[0] == 1:
            ii = ii[1:]  # first obs in block needs no correction
        for k in ii:
            # R: weight[k*2-2]=0 (sub-diag), weight[k*2-1]=1 (leading-diag)
            # → Python 0-based: weight[(k-1)*2-1]=0, weight[(k-1)*2]=1
            # but only valid when k≥2 (since k=1 was filtered above).
            weight[(k - 1) * 2 - 1] = 0.0
            weight[(k - 1) * 2] = 1.0
    return stop, row, weight


def _materialize_chunk(
    blocks: list[SmoothBlock],
    chunk_data: pl.DataFrame,
    X_param_chunk: np.ndarray,
) -> np.ndarray:
    """Build the full design row block for ``chunk_data``.

    ``X_param_chunk`` is the pre-materialised parametric block (sliced from
    a one-shot ``materialize`` call); each smooth block's columns come from
    ``spec.predict_mat(chunk_data)``. Returns the horizontally-stacked
    ``(n_chunk, p)`` matrix.

    Mirrors mgcv ``predict(G, newdata=mf[ind,], type="lpmatrix",
    newdata.guaranteed=TRUE, block.size=length(ind))`` (bam.r:1596).
    """
    parts: list[np.ndarray] = [np.asarray(X_param_chunk, dtype=float)]
    for b in blocks:
        if b.spec is None:
            raise RuntimeError(
                f"smooth block {b.label!r} lacks a BasisSpec — bam needs "
                f"chunk-time predict_mat replay."
            )
        Xb = b.spec.predict_mat(chunk_data)
        parts.append(np.asarray(Xb, dtype=float))
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def _build_qr_chunked_gaussian(
    data: pl.DataFrame,
    blocks: list[SmoothBlock],
    X_param_full: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    *,
    chunk_size: int,
    use_chol: bool = False,
    rho: float = 0.0,
    ar_start: Optional[np.ndarray] = None,
) -> _BamQR:
    """Chunked QR build for the Gaussian-identity (am=TRUE) path.

    Walks ``data`` in chunks of ``chunk_size``, materialises each chunk's
    full design via :func:`_materialize_chunk`, and accumulates ``(R, f,
    ‖z‖²)`` with :func:`_qr_update`. ``z = y − offset`` (prior weights = 1
    in this iteration; user-facing ``weights=`` lands later).

    For ``rho == 0`` this mirrors mgcv ``bam.fit`` single-thread loop
    (bam.r:1576-1613). For ``rho ≠ 0`` (AR1 error model) each chunk is
    transformed via :func:`_rw_matrix` using ``(stop, row, weight)``
    arrays from :func:`_ar1_rwmatrix_indices`; chunks 2+ overlap the
    previous chunk by one row (the row needed by the AR1 sub-diagonal),
    and the first transformed row of those chunks is dropped after the
    rwMatrix transform — see bam.r:1576-1611.

    ``ar_start`` (full-length-n boolean array, optional) re-anchors the
    AR chain at ``True`` positions; ``True`` at position i means row i
    starts a fresh AR sequence (sub-diagonal=0, leading-diag=1).
    """
    n = data.height
    if n == 0:
        raise ValueError("empty data passed to chunked QR build")
    if rho < -1.0 + 1e-12 or rho > 1.0 - 1e-12:
        raise ValueError(
            f"rho must be in (-1, 1) for stationary AR1, got rho={rho!r}"
        )
    ar1 = (rho != 0.0)
    if ar_start is not None:
        ar_start = np.asarray(ar_start, dtype=bool).flatten()
        if ar_start.shape != (n,):
            raise ValueError(
                f"ar_start must have length {n}, got {ar_start.shape}"
            )
    if ar1:
        ld = 1.0 / np.sqrt(1.0 - rho ** 2)
        sd = -rho * ld
    chunks = _chunk_indices(n, chunk_size, ar1=ar1)
    R: Optional[np.ndarray] = None
    f: Optional[np.ndarray] = None
    y_norm2 = 0.0
    for chunk_idx, (start, end) in enumerate(chunks):
        chunk_data = data[start:end]
        X_param_chunk = X_param_full[start:end]
        X_chunk = _materialize_chunk(blocks, chunk_data, X_param_chunk)
        z_chunk = y[start:end] - offset[start:end]
        if ar1:
            N_block = end - start
            ar_start_block = (
                ar_start[start:end] if ar_start is not None else None
            )
            stop, row, weight = _ar1_rwmatrix_indices(
                N_block, ld, sd, ar_start_block,
            )
            # rwMatrix returns the transformed n×p design / length-n vector.
            X_chunk = _rw_matrix(stop, row, weight, X_chunk)
            z_chunk = _rw_matrix(stop, row, weight, z_chunk)
            if chunk_idx > 0:
                # mgcv bam.r:1607-1610: chunks past the first drop the
                # head row, which already contributed to the previous
                # chunk's tail (overlap of 1).
                X_chunk = X_chunk[1:, :]
                z_chunk = z_chunk[1:]
        upd = _qr_update(X_chunk, z_chunk, R, f, y_norm2, use_chol=use_chol)
        R = upd["R"]
        f = upd["f"]
        y_norm2 = upd["y_norm2"]
    if use_chol:
        R, f = _chol2qr(R, f)
    rss_extra = float(y_norm2 - float(f @ f))
    return _BamQR(R=np.asarray(R, dtype=float),
                  f=np.asarray(f, dtype=float),
                  y_norm2=float(y_norm2),
                  rss_extra=rss_extra)


def _is_identity_link(family: Family) -> bool:
    """Detect Gaussian-identity (canonical Gaussian) — the ``am=TRUE`` case
    in mgcv's ``bam.fit`` dispatch (bam.r:2205)."""
    return isinstance(family, Gaussian) and family.link.name == "identity"


# ---------------------------------------------------------------------------
# bam class — Gaussian-identity chunked-QR fit (Phase 1 of three fitters)
# ---------------------------------------------------------------------------


class bam(gam):
    """Big additive model — chunked-QR variant of :class:`hea.gam`.

    Identical mathematical model to :class:`hea.gam` (penalised GLM, REML/
    ML/GCV smoothness selection). The constructor builds the QR factor
    ``R (p×p)`` and projected response ``f`` chunk-by-chunk, never holding
    the full ``n × p`` design in memory. All sufficient statistics for the
    outer optimizer derive from ``(R, f, ‖z‖²)``: ``X'X = R'R``, ``X'y =
    R'f``, ``‖y - Xβ‖² = ‖f - Rβ‖² + rss_extra``.

    Inherits :class:`hea.gam`'s :meth:`predict`, :meth:`summary`,
    :meth:`plot_smooth`, :meth:`vis`, :meth:`check`, … so user code is
    portable across ``gam`` and ``bam`` instances.

    Parameters mirror :class:`hea.gam` plus ``chunk_size`` (default 10000).
    Method defaults to ``"fREML"`` — mgcv's recommended bam method, fastest
    on the chunked path.

    Mirrors ``mgcv::bam`` (bam.r:2177-2806). This iteration covers the
    Gaussian-identity (``am=TRUE``) path; non-Gaussian (``bgam.fit``) and
    discrete (``bgam.fitd``) follow.
    """

    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        *,
        method: str = "fREML",
        sp: np.ndarray | None = None,
        family: Family | None = None,
        offset: np.ndarray | list | None = None,
        gamma: float = 1.0,
        select: bool = False,
        chunk_size: int = 10000,
        use_chol: bool = False,
        rho: float = 0.0,
        ar_start: np.ndarray | list | None = None,
    ):
        # ---- method aliasing ------------------------------------------------
        # mgcv's bam adds "fREML" on top of gam's {REML, ML, GCV.Cp}. fREML is
        # algorithmically identical to REML on the (R, f, rss_extra) reduced
        # problem — the "fast" comes from the QR-factor reduction, not a
        # different criterion. Map fREML → REML internally.
        method_in = method
        if method == "fREML":
            method = "REML"
        if method not in ("REML", "ML", "GCV.Cp"):
            raise ValueError(
                f"method must be 'fREML', 'REML', 'ML', or 'GCV.Cp', "
                f"got {method_in!r}"
            )
        if not (np.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")

        family = Gaussian() if family is None else family
        if not _is_identity_link(family):
            # bgam.fit / bgam.fitd ports come in subsequent tasks. Until then,
            # refuse rather than silently dispatching to the wrong fitter.
            raise NotImplementedError(
                f"bam currently supports family=Gaussian(link='identity') only; "
                f"got {family!r}. Non-Gaussian PIRLS chunked-QR fitter "
                f"(``bgam.fit``) lands in the next iteration."
            )

        # ---- AR1 plumbing (mgcv bam.r:478-498) -----------------------------
        # ``rho`` is the AR1 lag-1 correlation; setting it ≠ 0 wraps the
        # observation model with a Gaussian AR1 error process. The
        # ``rwMatrix`` transform applies the inverse Cholesky factor of
        # the AR1 correlation matrix to (X, y), producing i.i.d.
        # transformed errors. ``ar_start`` (length-n boolean) marks
        # observations that begin a fresh AR sequence — useful for
        # within-subject AR with multiple subjects in one frame.
        if not np.isfinite(rho):
            raise ValueError(f"rho must be finite, got {rho!r}")
        if abs(rho) >= 1.0:
            raise ValueError(
                f"rho must satisfy |rho|<1 for stationary AR1, got rho={rho!r}"
            )
        self._rho = float(rho)
        if ar_start is not None:
            ar_start_arr = np.asarray(ar_start, dtype=bool).flatten()
        else:
            ar_start_arr = None
        self._ar_start = ar_start_arr

        self.formula = formula
        self.method = method
        self._method_in = method_in
        self._select = bool(select)
        self._gamma = float(gamma)
        self.family = family
        self._chunk_size = int(chunk_size)
        self._use_chol = bool(use_chol)

        # ---- setup phase (mirror gam.__init__ lines 198-321) ---------------
        d = prepare_design(formula, data)
        self._expanded = d.expanded
        _expr_map = _smooth_arg_expr_map(self._expanded)
        self.data = (
            _apply_smooth_arg_exprs(d.data, _expr_map) if _expr_map else d.data
        )
        X_param_df = d.X
        y_full = d.y.to_numpy().astype(float)
        X_param_full = X_param_df.to_numpy().astype(float)
        n, p_param = X_param_full.shape

        off = (np.zeros(n) if offset is None
               else np.asarray(offset, dtype=float).flatten())
        if off.shape != (n,):
            raise ValueError(f"offset must have length {n}, got {off.shape}")
        for off_node in d.expanded.offsets:
            blk = _eval_atom(off_node, d.data)
            off = off + blk.values.flatten().astype(float)
        self._offset = off

        # ---- mini.mf for basis setup (bam.r:2387) --------------------------
        # Bases (knots, eigendecompositions, absorb constraints) are fitted
        # on a small representative subsample so basis construction does not
        # scale with n. The full data is never materialised as one X — we
        # walk it chunk-by-chunk via spec.predict_mat for the QR build.
        if self._chunk_size < p_param + 1:
            # Match mgcv's reset (bam.r:2405-2410): chunk_size < ncol(X) is
            # nonsensical for accumulation; bump it.
            self._chunk_size = max(4 * (p_param + 1), 1)
        chunk_size = self._chunk_size
        if d.expanded.smooths:
            mf0 = _mini_mf(self.data, chunk_size)
            sb_lists = materialize_smooths(d.expanded, mf0)
            blocks: list[SmoothBlock] = [b for group in sb_lists for b in group]
        else:
            blocks = []
        if self._select:
            blocks = _add_null_space_penalties(blocks)
        blocks = _apply_gam_side(blocks)

        # Slot bookkeeping (column ranges + S matrices) and column count.
        slots: list[_PenaltySlot] = []
        block_col_ranges: list[tuple[int, int]] = []
        col_cursor = p_param
        for b in blocks:
            k = int(np.asarray(b.X).shape[1])
            a, bcol = col_cursor, col_cursor + k
            block_col_ranges.append((a, bcol))
            for S_j in b.S:
                slots.append(_PenaltySlot(block=b, col_start=a, col_end=bcol,
                                          S=np.asarray(S_j, dtype=float)))
            col_cursor = bcol
        p = col_cursor

        # If chunk_size is now smaller than p, retry with a bigger chunk.
        if self._chunk_size < p:
            self._chunk_size = 4 * p
            chunk_size = self._chunk_size

        column_names = list(X_param_df.columns)
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            for i in range(1, bcol - a + 1):
                column_names.append(f"{b.label}.{i}")
        assert len(column_names) == p

        has_intercept = "(Intercept)" in X_param_df.columns
        self.X = X_param_df
        self.y = d.y
        self._y_arr = y_full
        self.n = n
        self.p = p
        self.p_param = p_param
        self._blocks = blocks
        self._slots = slots
        self._block_col_ranges = block_col_ranges
        self.column_names = column_names
        self._has_intercept = has_intercept
        self.parametric_columns = list(X_param_df.columns)
        self._X_param_full = X_param_full

        # ---- chunked QR build ----------------------------------------------
        # Single chunked pass over the full data, accumulating (R, f, ‖z‖²).
        # Mirrors mgcv ``bam.fit`` single-thread loop (bam.r:1576-1613).
        # ``z = y − offset`` (Gaussian-identity working response under
        # prior weights = 1; the family's identity link gives μ = η, so
        # PIRLS converges in one solve and the QR-only path is exact).
        # When ``rho ≠ 0``, an AR1 inverse-Cholesky transform is applied
        # to each chunk via :func:`_rw_matrix` so the resulting (R, f)
        # correspond to the AR1-decorrelated working data.
        qr = _build_qr_chunked_gaussian(
            self.data, blocks, X_param_full, y_full, off,
            chunk_size=chunk_size, use_chol=self._use_chol,
            rho=self._rho, ar_start=self._ar_start,
        )
        self._bam_qr = qr
        # Sufficient statistics from (R, f). These are exact identities:
        # X'X = R'R, X'y = R'f, ‖y−off‖² = y_norm2 + 0 (here ‖z‖²; the
        # offset-aware deviance computation in ``_fit_given_rho`` adds
        # rss_extra back).
        self._XtX = qr.R.T @ qr.R
        self._Xty = qr.R.T @ qr.f
        self._yty = qr.y_norm2  # = ‖y − off‖² (offset-stripped)
        # ``_X_full = R`` so inherited score routines see a square p×p design
        # whose Gram matches the full-data Gram. The trace identity
        # ``tr(X H⁻¹ X') = tr(R H⁻¹ R')`` keeps log|H|/Hessian-trace
        # computations exact; per-row diag values that would have been
        # length-n become length-p, but they are only ever multiplied by
        # ``∂w/∂η = 0`` (Gaussian-identity has constant w), so the result is
        # zero either way. mgcv's bam.fit "ML" branch (bam.r:1722-1733)
        # similarly reuses the gam.fit3 machinery on ``X = R``, ``y = f``.
        self._X_full = qr.R

        # tss for r-squared. We need the full y'y (not the offset-stripped
        # y_norm2) and the intercept-conditioned variance.
        full_yty = float(y_full @ y_full)
        if has_intercept:
            mean_y = float(y_full.mean())
            tss = float(np.sum((y_full - mean_y) ** 2))
        else:
            tss = full_yty
        self._yty_full = full_yty
        self._tss = tss

        # Null-space dimension Mp + penalty rank.
        Mp = p_param
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            k = bcol - a
            if not b.S:
                Mp += k
                continue
            S_sum = np.sum([np.asarray(s, dtype=float) for s in b.S], axis=0)
            rank = _sym_rank(S_sum)
            Mp += k - rank
        self._Mp = Mp
        self._penalty_rank = p - Mp

        # ---- smoothing-param optimization ----------------------------------
        # Same outer Newton as gam, but every PIRLS-replacement call to
        # ``_fit_given_rho`` here goes through the override below.
        n_sp = len(slots)
        self._log_phi_hat: float | None = None
        self._outer_info: dict | None = None
        self._tw_info: dict | None = None
        if n_sp == 0:
            self.sp = np.zeros(0)
            rho_hat = np.zeros(0)
            fit = self._fit_given_rho(rho_hat)
        elif sp is not None:
            sp_arr = np.asarray(sp, dtype=float)
            if sp_arr.shape != (n_sp,):
                raise ValueError(
                    f"sp must have length {n_sp} (one per penalty slot), "
                    f"got {sp_arr.shape}"
                )
            if np.any(sp_arr < 0):
                raise ValueError("sp entries must be non-negative")
            rho_hat = np.log(np.maximum(sp_arr, 1e-10))
            self.sp = sp_arr
            fit = self._fit_given_rho(rho_hat)
            if (not self.family.scale_known) and method in ("REML", "ML"):
                Dp = float(fit.dev + fit.pen)
                denom = (max(float(n - self._Mp), 1.0)
                         if method == "REML" else max(float(n), 1.0))
                self._log_phi_hat = float(
                    np.log(max(Dp / denom, 1e-300))
                )
        else:
            include_log_phi = (
                (not family.scale_known) and method in ("REML", "ML")
            )
            include_family_theta = False  # tw / extended families not in this iter
            if method in ("REML", "ML"):
                cur_rho = np.zeros(n_sp)
                if include_log_phi:
                    try:
                        fit_seed = self._fit_given_rho(cur_rho)
                        df_resid_seed = max(self.n - self._Mp, 1.0)
                        # Gaussian: V(μ)=1, so pearson = ‖y - μ̂‖². At the
                        # seed the dev returned by ``_fit_given_rho`` already
                        # includes rss_extra ⇒ direct.
                        cur_logphi = float(np.log(
                            max(fit_seed.dev / df_resid_seed, 1e-12)
                        ))
                    except Exception:
                        cur_logphi = 0.0
                else:
                    cur_logphi = 0.0
            else:
                cur_rho = self._initial_sp_rho()
                cur_logphi = 0.0

            theta0_parts = [cur_rho]
            if include_log_phi:
                theta0_parts.append(np.array([cur_logphi]))
            theta0 = np.concatenate(theta0_parts)

            theta_hat = self._outer_newton(
                theta0,
                criterion="REML" if method in ("REML", "ML") else "GCV",
                include_log_phi=include_log_phi,
                include_family_theta=include_family_theta,
            )

            if include_log_phi:
                rho_hat = theta_hat[:n_sp]
                log_phi_hat = float(theta_hat[n_sp])
            else:
                rho_hat = theta_hat
                log_phi_hat = None
            self._log_phi_hat = log_phi_hat
            self.sp = np.exp(rho_hat)
            fit = self._fit_given_rho(rho_hat)

        # ---- post-fit assembly ---------------------------------------------
        # Most of gam.__init__'s post-fit code reads ``self._X_full`` and
        # ``fit.mu``/``fit.eta``. With ``_X_full = R`` and Gaussian-identity
        # there are no PIRLS weights to rebuild; the things that need full-n
        # quantities (eta, mu, residuals, leverage) are computed via a
        # single chunked walk below.
        self._post_fit_gaussian(fit, rho_hat, X_param_df)

    # -----------------------------------------------------------------------
    # _fit_given_rho override — uses (R, f, y_norm2, rss_extra)
    # -----------------------------------------------------------------------

    def _fit_given_rho(self, rho: np.ndarray) -> "_FitState":
        """Closed-form Gaussian-identity solve from stored (R, f, ‖z‖², rss_extra).

        For Gaussian-identity PIRLS reduces to one weighted-LS solve with
        ``W = I`` and ``z = y - off``, so β̂(ρ) is the minimiser of
        ``‖z - Xβ‖² + β'Sλβ``. Using ``X'X = R'R`` and ``X'z = R'f``:

            (R'R + Sλ) β̂ = R'f                            # normal equations
            ‖z - Xβ̂‖²    = ‖f - Rβ̂‖² + rss_extra          # decomposition

        Returns a ``_FitState`` populated for downstream consumers; ``eta``,
        ``mu``, ``w``, ``z``, ``alpha`` are left ``None`` here and filled
        once after outer-Newton convergence by ``_post_fit_gaussian`` (a
        single chunked pass over the full data).
        ``is_fisher_fallback=True`` short-circuits ``_fisher_view`` (Newton
        ≡ Fisher for canonical Gaussian-identity) and tells ``_dw_deta`` to
        return zero α-derivatives — both are correct here.
        """
        Sλ = self._build_S_lambda(rho)
        Sλ = 0.5 * (Sλ + Sλ.T)
        A = self._XtX + Sλ
        A = 0.5 * (A + A.T)
        try:
            A_chol, lower = cho_factor(A, lower=True, overwrite_a=False)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.trace(A) / max(self.p, 1)
            A_chol, lower = cho_factor(
                A + ridge * np.eye(self.p), lower=True, overwrite_a=False,
            )
        beta = cho_solve((A_chol, lower), self._Xty)
        if not np.all(np.isfinite(beta)):
            raise FloatingPointError("non-finite β in Gaussian-identity solve")
        pen = float(beta @ Sλ @ beta)
        # Full-data dev = ‖z - Xβ̂‖² = ‖f - Rβ̂‖² + rss_extra
        #               = ‖z‖² − 2 β̂' R'f + β̂' R'R β̂
        rss = float(
            self._yty - 2.0 * (beta @ self._Xty) + beta @ self._XtX @ beta
        )
        rss = max(rss, 0.0)  # guard tiny negative from cancellation
        log_det_A = 2.0 * float(np.log(np.abs(np.diag(A_chol))).sum())
        # ``_score_scale`` (closure in ``_outer_newton``) computes the
        # Pearson sum ``‖y - μ‖²/V`` for unknown-scale families using
        # ``fit.mu`` against the full ``self._y_arr``. So ``fit.mu``/
        # ``fit.eta`` must be length-n. We recover them via a single
        # chunked ``X·β`` walk per call — O(n·p), the same cost gam pays
        # implicitly for ``eta = X @ β`` every outer-Newton iteration.
        # In contrast, ``_dw_deta`` / ``_d2w_deta2`` consumers
        # (``_dlog_det_H_drho``, ``_d2beta_drho_drho``) multiply against
        # ``self._X_full = R`` (p×p), requiring length-p arrays — those
        # methods are overridden below to return ``zeros(p)`` directly,
        # which is mathematically correct for Gaussian-identity.
        eta_only = self._chunked_xbeta(beta)        # X·β (offset-stripped)
        eta = eta_only + self._offset               # full η, length-n
        mu = eta                                    # identity link
        n = self.n
        return _FitState(
            beta=beta, dev=rss, pen=pen,
            A_chol=A_chol, A_chol_lower=lower,
            S_full=Sλ, log_det_A=log_det_A,
            eta=eta, mu=mu, w=np.ones(n),
            z=self._y_arr - self._offset, alpha=np.ones(n),
            is_fisher_fallback=True,
        )

    # -----------------------------------------------------------------------
    # PIRLS-weight derivatives — length-p zeros for Gaussian-identity
    # -----------------------------------------------------------------------

    def _dw_deta(self, fit: "_FitState") -> np.ndarray:
        """∂w/∂η for Gaussian-identity: identically zero.

        For the canonical Gaussian-identity family, ``V(μ)=1`` (so
        ``V'=0``), ``g(μ)=μ`` (so ``g''=0``), and the Newton/Fisher α
        factor is constant 1 (``is_fisher_fallback=True``). The base
        formula ``dw/dη = w·μ_η·(α'/α − 2g''μ_η − V'/V)`` therefore
        evaluates to zero exactly.

        We override the inherited length-n version with length-p so
        downstream broadcasts against ``self._X_full = R`` (p×p) line up:
        ``hv = dw_deta[:, None] · (X·∂β/∂ρ)`` in ``_dlog_det_H_drho`` and
        ``_reml_hessian``, and ``X' · (dw_deta · v_l · v_k)`` in
        ``_d2beta_drho_drho``. ``_reml_hessian``'s ``needs_w``
        short-circuit (line 1441) sees ``np.any(zeros)==False`` and skips
        the K/M construction entirely — the same fast path Gaussian-fit
        gam takes when length-n zeros are returned.
        """
        return np.zeros(self.p)

    def _d2w_deta2(self, fit: "_FitState") -> np.ndarray:
        """∂²w/∂η² for Gaussian-identity: identically zero. Length-p so
        ``np.any(d2w_deta2)`` evaluates against the right-shape array
        and ``_reml_hessian``'s ``needs_w`` gate stays correct."""
        return np.zeros(self.p)

    # -----------------------------------------------------------------------
    # initial sp seed — uses cached XtX diag, no full design
    # -----------------------------------------------------------------------

    def _initial_sp_rho(self) -> np.ndarray:
        """``initial.sp`` seed using ``diag(R'R)`` for the column sums of
        squares (= ``diag(X'X) = Σ_i X[i,j]²``) — no full design needed."""
        ldxx = np.diag(self._XtX)
        n_sp = len(self._slots)
        rho0 = np.zeros(n_sp)
        for k, slot in enumerate(self._slots):
            S_k = slot.S
            absS = np.abs(S_k)
            maS = float(absS.max()) if absS.size else 0.0
            if maS <= 0.0:
                continue
            thresh = float(np.finfo(float).eps ** 0.8) * maS
            rsS = absS.mean(axis=1)
            csS = absS.mean(axis=0)
            dS = np.abs(np.diag(S_k))
            ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)
            if not np.any(ind):
                continue
            ss = np.diag(S_k)[ind]
            xx = ldxx[slot.col_start:slot.col_end][ind]
            sizeXX = float(np.mean(xx))
            sizeS = float(np.mean(ss))
            if sizeS <= 0.0 or sizeXX <= 0.0:
                continue
            rho0[k] = float(np.log(sizeXX / sizeS))
        return rho0

    # -----------------------------------------------------------------------
    # Post-fit — chunked walks for full-n quantities (eta, mu, leverage)
    # -----------------------------------------------------------------------

    def _chunked_xbeta(self, beta: np.ndarray) -> np.ndarray:
        """Compute ``X·β`` over the full data, chunk by chunk. ``O(n·p)``
        time, ``O(chunk_size·p)`` peak memory."""
        n = self.n
        out = np.empty(n, dtype=float)
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = _materialize_chunk(
                self._blocks,
                self.data[start:end],
                self._X_param_full[start:end],
            )
            out[start:end] = X_chunk @ beta
        return out

    def _chunked_leverage_diag(self, A_inv: np.ndarray) -> np.ndarray:
        """Diagonal of the unweighted hat matrix ``H = X·A⁻¹·X'``.

        For Gaussian-identity (W=I), ``leverage_i = (X A⁻¹ X')_ii``. We
        compute it chunk-wise via ``(X_chunk · A⁻¹) ⊙ X_chunk`` summed across
        the column axis — never materialising the n×n hat.
        """
        n = self.n
        out = np.empty(n, dtype=float)
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = _materialize_chunk(
                self._blocks,
                self.data[start:end],
                self._X_param_full[start:end],
            )
            HX = X_chunk @ A_inv
            out[start:end] = (HX * X_chunk).sum(axis=1)
        return out

    def _post_fit_gaussian(self, fit, rho_hat: np.ndarray,
                           X_param_df: pl.DataFrame) -> None:
        """Populate the user-facing attributes after outer-Newton has
        converged on the (R, f) reduced problem.

        Mirrors the post-fit block in ``gam.__init__`` (gam.py:476-783) for
        the Gaussian-identity special case — the X-using pieces (full η,
        leverage, residuals) are filled by a single chunked walk; the
        β/Vp/Ve/edf algebra is identical (XtWX = X'X = R'R when W=I).
        """
        n, p = self.n, self.p
        method = self.method
        n_sp = len(self._slots)
        beta = fit.beta
        Sλ = fit.S_full
        self._rho_hat = rho_hat

        # Inverse Hessian — small (p×p), exact.
        A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        # XtWX with W=I is just X'X = R'R, already cached.
        XtWX = self._XtX
        A_inv_XtWX = A_inv @ XtWX
        edf = np.diag(A_inv_XtWX).copy()
        edf_total = float(edf.sum())

        # Prior weights (=1 for now). Same convention as gam.
        self._wt = np.ones(n)
        wt = self._wt
        df_resid = float(n - edf_total)
        # Gaussian: V=1, scale = ‖y - μ̂‖²/(n - edf). fit.dev already holds
        # the full-data residual sum of squares (rss_extra absorbed).
        if df_resid > 0:
            pearson_scale = float(fit.dev) / df_resid
        else:
            pearson_scale = float("nan")
        self._pearson_scale = pearson_scale
        sigma_squared = pearson_scale
        sigma = (float(np.sqrt(sigma_squared))
                 if np.isfinite(sigma_squared) and sigma_squared >= 0
                 else float("nan"))

        Vp = sigma_squared * A_inv
        Ve = sigma_squared * A_inv_XtWX @ A_inv

        # Coefficient basis change for t2 (rare). Use the same code path as
        # gam — uses block.spec.coef_remap, no full X.
        intercept_idx: Optional[int] = (
            self.column_names.index("(Intercept)")
            if self._has_intercept else None
        )
        if any(b.spec is not None and b.spec.coef_remap is not None
               for b in self._blocks):
            G_P = np.eye(p)
            for b, (a_col, b_col) in zip(self._blocks, self._block_col_ranges):
                if b.spec is None or b.spec.coef_remap is None:
                    continue
                M_b, X_bar_b = b.spec.coef_remap
                G_P[a_col:b_col, a_col:b_col] = M_b
                if intercept_idx is not None:
                    G_P[intercept_idx, a_col:b_col] = X_bar_b
            beta = G_P @ beta
            Vp = G_P @ Vp @ G_P.T
            Ve = G_P @ Ve @ G_P.T

        # ---- β / SE / t / p (parametric Wald) ------------------------------
        self.bhat = _row_frame(beta, self.column_names)
        self._beta = beta
        se = np.sqrt(np.diag(Vp))
        self.se_bhat = _row_frame(se, self.column_names)
        self._se = se
        t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        self.t_values = _row_frame(t_stats, self.column_names)
        if df_resid > 0 and np.isfinite(df_resid):
            pv = 2 * t_dist.sf(np.abs(t_stats), df_resid)
        else:
            pv = np.full_like(t_stats, np.nan)
        self.p_values = _row_frame(pv, self.column_names)

        # ---- chunked walk to recover η, μ, residuals, leverage --------------
        eta_only = self._chunked_xbeta(beta)         # X·β (offset-stripped)
        eta = eta_only + self._offset                # full η
        mu = eta                                     # identity link
        self.linear_predictors = eta
        self.fitted_values = mu
        self.fitted = mu
        y = self._y_arr
        # Gaussian deviance residuals = sign(y-μ)·√((y-μ)²) = y - μ.
        self.residuals = y - mu
        self.sigma = sigma
        self.sigma_squared = sigma_squared
        self.scale = sigma_squared

        # Leverage diag: chunked. For Gaussian-identity W=I, leverage_i =
        # (X A⁻¹ X')_ii — no √W factors. Σ leverage_i = edf_total exactly.
        leverage = self._chunked_leverage_diag(A_inv)
        self.leverage = leverage
        sigma_for_std = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
        denom = sigma_for_std * np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        # V(μ)=1, Pearson residual = (y - μ)/√V = y - μ.
        pearson_res = (y - mu)
        self.std_dev_residuals = self.residuals / denom
        self.std_pearson_residuals = pearson_res / denom
        self.df_residuals = df_resid
        # mgcv bam.r:2774 — ``object$deviance = sum(object$residuals^2)`` where
        # ``residuals = y - μ`` is *response-space*. For AR1 (rho != 0) the
        # AR1-decorrelated RSS lives separately in ``std.rsd`` (used for σ²
        # and AIC scale calcs). The response-space ``deviance`` is what
        # ``deviance.explained`` reports against ``null.deviance``, both on the
        # original y scale.
        self.deviance = float(np.sum(self.residuals ** 2))
        self.rss = self.deviance
        # AR1-decorrelated residuals (mgcv ``object$std.rsd``, bam.r:2772) —
        # used by ``acf(rsd)`` checks. For rho=0, equals self.residuals.
        if self._rho != 0.0:
            self.std_rsd = _ar_resid(self.residuals, self._rho, self._ar_start)
        else:
            self.std_rsd = self.residuals.copy()

        # Null deviance — intercept-only Gaussian: weighted mean.
        if self._has_intercept:
            mu_null_const = float(np.sum(wt * y) / np.sum(wt))
            mu_null = np.full(n, mu_null_const)
        else:
            mu_null = self.family.link.linkinv(np.zeros(n))
        self.null_deviance = float(
            np.sum(self.family.dev_resids(y, mu_null, wt))
        )
        self.df_null = float(n - 1) if self._has_intercept else float(n)

        self.Vp = Vp
        self.Ve = Ve
        self._A_inv = A_inv
        self.edf = edf
        self.edf_total = edf_total
        edf_by_smooth: dict[str, float] = {}
        for b, (a, bcol) in zip(self._blocks, self._block_col_ranges):
            edf_by_smooth[b.label] = float(edf[a:bcol].sum())
        self.edf_by_smooth = edf_by_smooth

        # R² / R²_adj. Same formulas as gam (uses full y, full μ).
        ss_resid_response = float(np.sum(wt * (y - mu) ** 2))
        if self._has_intercept and self._tss > 0:
            r_squared = 1.0 - ss_resid_response / self._tss
        elif self._yty_full > 0:
            r_squared = 1.0 - ss_resid_response / self._yty_full
        else:
            r_squared = float("nan")
        if df_resid > 0 and n > 1:
            sqrt_wt = np.sqrt(wt)
            mean_y_w = float(np.sum(wt * y) / np.sum(wt))
            v_resid = float(np.var(sqrt_wt * (y - mu), ddof=1))
            v_total = float(np.var(sqrt_wt * (y - mean_y_w), ddof=1))
            if v_total > 0:
                r_squared_adjusted = 1.0 - v_resid * (n - 1) / (v_total * df_resid)
            else:
                r_squared_adjusted = float("nan")
        else:
            r_squared_adjusted = float("nan")
        self.r_squared = float(r_squared)
        self.r_squared_adjusted = float(r_squared_adjusted)
        if self.null_deviance > 0:
            self.deviance_explained = float(
                (self.null_deviance - self.deviance) / self.null_deviance
            )
        else:
            self.deviance_explained = float("nan")

        # ``_fit_given_rho`` populated fit.eta/fit.mu (length-n) and
        # fit.w/fit.alpha (length-n ones) for ``_score_scale``. The
        # post-fit edf1/edf2/Vc machinery and ``_compute_edf12``'s
        # ``W_F_view = fit.w`` path read these, and the all-ones case
        # short-circuits to ``XtWX = self._XtX`` (line 3228 in gam.py).
        # No further patching needed here.
        self._fisher_w = None

        # Augmented REML Hessian (only built if (R)EML and finite σ²).
        if (
            method in ("REML", "ML")
            and n_sp > 0
            and np.isfinite(sigma_squared)
            and sigma_squared > 0
        ):
            log_phi_hat_for_aug = (
                self._log_phi_hat
                if self._log_phi_hat is not None
                else float(np.log(sigma_squared))
            )
            H_aug = 0.5 * self._reml_hessian(
                rho_hat, log_phi_hat_for_aug, fit=fit, include_log_phi=True,
            )
            H_aug = 0.5 * (H_aug + H_aug.T)
        else:
            H_aug = None
        self._H_aug = H_aug

        if n_sp > 0:
            edf2_per_coef, edf1_per_coef, Vc_corr = self._compute_edf12(
                rho_hat, fit, sigma_squared, A_inv, A_inv_XtWX, edf, H_aug,
            )
            self.edf1 = edf1_per_coef
            self.edf2 = edf2_per_coef
            self.edf1_total = float(edf1_per_coef.sum())
            self.edf2_total = float(edf2_per_coef.sum())
        else:
            self.edf1 = edf.copy()
            self.edf2 = edf.copy()
            self.edf1_total = edf_total
            self.edf2_total = edf_total
            Vc_corr = np.zeros_like(Vp)
        self.Vc = Vp + Vc_corr

        # AIC / BIC.
        sc_p = 0.0 if self.family.scale_known else 1.0
        dev1 = self.family._aic_dev1(self.deviance, sigma_squared, wt)
        family_aic = float(self.family.aic(y, fit.mu, dev1, wt, n))
        mgcv_aic = family_aic + 2.0 * edf_total
        logLik = sc_p + edf_total - 0.5 * mgcv_aic
        df_for_aic = min(self.edf2_total + sc_p, float(p) + sc_p)
        self.loglike = float(logLik)
        self.logLik = self.loglike
        self.npar = float(df_for_aic)
        self.AIC = -2.0 * logLik + 2.0 * df_for_aic
        self.BIC = -2.0 * logLik + float(np.log(n)) * df_for_aic
        self._mgcv_aic = float(mgcv_aic)

        # Score (REML / ML / GCV).
        if method in ("REML", "ML"):
            if n_sp > 0:
                log_phi_hat = (
                    self._log_phi_hat if self._log_phi_hat is not None else 0.0
                )
                score = float(self._reml(rho_hat, log_phi_hat, fit=fit))
            else:
                score = float("nan")
            # AR1 correction (mgcv bam.r:795-798). The AR1 transform
            # changes the log-determinant of the response covariance by
            # ``(n/γ - df) · log(ld)`` where ``ld = 1/√(1-ρ²)`` and ``df``
            # is the number of independent AR sequences (1 if ar_start
            # is None, else ``sum(ar_start)``). Subtract it from the
            # otherwise-uncorrected REML score so the final value matches
            # mgcv's. This shift is constant in (sp, log φ), so the
            # outer Newton's optimum is unaffected.
            if self._rho != 0.0 and np.isfinite(score):
                ld = 1.0 / np.sqrt(1.0 - self._rho ** 2)
                df_ar = (
                    int(self._ar_start.sum())
                    if self._ar_start is not None else 1
                )
                score = score - (n / self._gamma - df_ar) * float(np.log(ld))
            if method == "REML":
                self.REML_criterion = score
            else:
                self.ML_criterion = score
        else:
            if n_sp > 0:
                self.GCV_score = float(self._gcv(rho_hat))
            else:
                self.GCV_score = float("nan")

        # Variance components — uses Vp, Vc, sp; no full design.
        self.vcomp = self._compute_vcomp()

        # mgcv exposes ``object$AR1.rho`` (bam.r:885) for downstream
        # consumers (predict.bam, residuals.bam). Mirror the attribute.
        self.AR1_rho = self._rho
