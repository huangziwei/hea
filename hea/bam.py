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

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve, qr as scipy_qr, solve_triangular
from scipy.linalg.lapack import dpstrf
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
from .discrete import (
    DiscreteDesign,
    DiscretizedFrame,
    Xbd,
    XWXd,
    XWyd,
    build_discrete_design,
    discrete_full_X,
    discrete_mf,
)
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


@dataclass
class _BlockRepara:
    """Per-block reparameterization data — mgcv ``Sl.setup`` + ``Sl.
    initial.repara`` (fast-REML.r:68-402, 490-735) for multi-S blocks.

    For a block at columns ``[col_start, col_end)`` with multiple S
    matrices, ``D = U`` (eigenvectors of ``S_total = ΣS_j``); the
    penalised subspace is the top ``rank`` directions and the rest is
    null. Each ``S_j`` projects to ``U[:, :rank]'·S_j·U[:, :rank]``,
    shape ``(rank, rank)``.

    For a singleton block (one S), no repara is applied here — hea's
    ``_absorb_sumzero`` already lives upstream. Keeping the singleton
    path empty avoids touching designs that already match mgcv at
    machine precision.
    """
    col_start: int
    col_end: int
    U: np.ndarray              # (m, rank) basis (m = col_end - col_start)
    rank: int                  # numerical rank of S_total
    slot_indices: list         # global slot indices in this block
    S_proj: list               # projected S matrices, shape (rank, rank) each


def _build_init_repara(slots: list, p: int) -> list:
    """Build per-block reparameterization data for multi-S blocks.

    Mirrors mgcv ``Sl.setup`` (fast-REML.r:335-369) for the
    ``cholesky=FALSE`` path: eigendecompose ``S_total = ΣS_j`` for each
    block with multiple penalty matrices, take the top-rank
    eigenvectors as ``U``, project each ``S_j``.

    Returns: list of ``_BlockRepara`` objects, one per multi-S block.
    Singleton blocks are not in the list (no repara needed).
    """
    # Group slots by col range. Multi-S blocks have multiple slots
    # sharing the same range.
    by_range: dict = {}
    for k, slot in enumerate(slots):
        key = (int(slot.col_start), int(slot.col_end))
        by_range.setdefault(key, []).append(k)

    repara_blocks: list = []
    for (cs, ce), slot_idxs in by_range.items():
        if len(slot_idxs) <= 1:
            continue   # singleton, skip
        # Multi-S block — eigendecompose total penalty.
        S_total = np.zeros_like(slots[slot_idxs[0]].S)
        for k in slot_idxs:
            S_total = S_total + slots[k].S
        S_total = 0.5 * (S_total + S_total.T)
        # eigh returns ascending; we want descending (largest first).
        eigval, U_full = np.linalg.eigh(S_total)
        order = np.argsort(eigval)[::-1]
        U_full = U_full[:, order]
        eigval = eigval[order]
        m = ce - cs
        # mgcv rank determination (fast-REML.r:357-358):
        # ``rank <- sum(D > .Machine$double.eps^.8 * max(D))``.
        thresh = float(np.finfo(float).eps) ** 0.8 * float(eigval[0])
        rank = int(np.sum(eigval > thresh))
        if rank == 0:
            continue
        U = U_full[:, :rank]
        # Project each S_j onto the rank-r range of S_total.
        S_proj = []
        for k in slot_idxs:
            P = U.T @ slots[k].S @ U
            P = 0.5 * (P + P.T)
            S_proj.append(P)
        repara_blocks.append(_BlockRepara(
            col_start=cs, col_end=ce, U=U, rank=rank,
            slot_indices=list(slot_idxs), S_proj=S_proj,
        ))
    return repara_blocks


def _apply_init_repara(
    XX: np.ndarray, Xy: np.ndarray, repara_blocks: list,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Apply mgcv-style ``Sl.initial.repara`` (forward) to ``XX``,
    ``Xy``. Returns the repara'd matrices plus a "padded" S list
    (one (m, m) matrix per slot, with the projected ``(rank, rank)``
    block placed at the top-left of the slot's column range and the
    null tail zeroed).

    For each multi-S block at ``[cs, ce)`` with basis ``U`` of size
    ``(m, rank)``:

      * ``XX_new`` = ``D' XX D`` where ``D`` is identity outside the
        block and ``[U, U_null]`` inside (``U_null`` = orthogonal
        complement of ``U``). For our purposes only the ``U`` columns
        are used since the null tail is unpenalised.
      * ``Xy_new[block]`` = ``[U; U_null]' Xy[block]``.

    Implementation simplification: when ``rank == m`` (the common
    case for te smooths whose S_total is full-rank within the post-
    absorb block), ``U`` is square orthogonal and the repara is just
    a basis change. When ``rank < m``, we still need a square
    transform — extend ``U`` with an orthonormal completion ``U_null``
    so ``D = [U, U_null]`` is m×m. ``np.linalg.qr`` on ``U`` produces
    that completion as a side effect.
    """
    XX_new = XX.copy()
    Xy_new = Xy.copy()
    for blk in repara_blocks:
        cs, ce = blk.col_start, blk.col_end
        m = ce - cs
        if blk.rank == m:
            D = blk.U
        else:
            # Extend U to an orthogonal basis [U, U_null] of m×m.
            Q, _ = np.linalg.qr(blk.U, mode="complete")
            # ``np.linalg.qr(U, mode='complete')`` returns Q of shape
            # (m, m) whose first ``rank`` columns span the column space
            # of U (up to sign/orthogonal rotation within that space).
            # mgcv's convention is ``D = [U, U_null]`` literally — we
            # keep ``U`` exactly and grab the trailing columns of Q
            # for the null completion.
            U_null = Q[:, blk.rank:]
            # Re-orthogonalise U_null against U (numerical safety).
            U_null = U_null - blk.U @ (blk.U.T @ U_null)
            U_null, _ = np.linalg.qr(U_null)
            D = np.concatenate([blk.U, U_null], axis=1)
        # XX[block, block] <- D' XX[block, block] D, etc.
        # Two-sided sandwich on the relevant block range.
        # Step 1: rows.
        XX_new[cs:ce, :] = D.T @ XX_new[cs:ce, :]
        # Step 2: columns.
        XX_new[:, cs:ce] = XX_new[:, cs:ce] @ D
        # Xy: rows only.
        Xy_new[cs:ce] = D.T @ Xy_new[cs:ce]

    return XX_new, Xy_new


def _undo_init_repara_beta(
    beta: np.ndarray, repara_blocks: list,
) -> np.ndarray:
    """Inverse of ``_apply_init_repara`` for β: ``β[block] = D ·
    β_new[block]`` (mgcv ``Sl.initial.repara(..., inverse=TRUE,
    both.sides=FALSE)``).
    """
    out = beta.copy()
    for blk in repara_blocks:
        cs, ce = blk.col_start, blk.col_end
        m = ce - cs
        if blk.rank == m:
            D = blk.U
        else:
            Q, _ = np.linalg.qr(blk.U, mode="complete")
            U_null = Q[:, blk.rank:]
            U_null = U_null - blk.U @ (blk.U.T @ U_null)
            U_null, _ = np.linalg.qr(U_null)
            D = np.concatenate([blk.U, U_null], axis=1)
        out[cs:ce] = D @ out[cs:ce]
    return out


def _build_repara_slots(
    slots: list, repara_blocks: list,
) -> tuple[list, list]:
    """Build a "repara'd slot view" for ``_pi_fit_chol``.

    For each multi-S block, the original slots' S matrices are
    replaced with the rank×rank projected S, and the slot's effective
    column range is ``[col_start, col_start + rank)`` (the penalised
    sub-block). Slots in singleton blocks pass through unchanged.

    Returns ``(slots_pre, slot_idx_map)`` where ``slot_idx_map[k]`` is
    the global slot index that ``slots_pre[k]`` corresponds to, so
    callers can recover original ordering for the gradient.
    """
    # Map (col_start, col_end) -> _BlockRepara
    blk_by_range = {(b.col_start, b.col_end): b for b in repara_blocks}
    slots_pre: list = []
    for k, slot in enumerate(slots):
        key = (int(slot.col_start), int(slot.col_end))
        if key not in blk_by_range:
            # Singleton block — keep as-is.
            slots_pre.append(slot)
            continue
        # Multi-S — find which entry in this block the slot is.
        blk = blk_by_range[key]
        try:
            local_idx = blk.slot_indices.index(k)
        except ValueError:
            # Shouldn't happen if repara was built consistently.
            slots_pre.append(slot)
            continue
        S_proj = blk.S_proj[local_idx]
        # Wrap a lightweight slot-like object with the projected S
        # placed at columns [col_start, col_start + rank).
        from types import SimpleNamespace
        slots_pre.append(SimpleNamespace(
            col_start=int(slot.col_start),
            col_end=int(slot.col_start + blk.rank),
            S=S_proj,
        ))
    return slots_pre


def _pi_fit_chol(
    XX: np.ndarray, Xy: np.ndarray, rho: np.ndarray,
    slots: list, p: int, *, yy: float = 0.0,
    log_phi: float = 0.0, n: int = 0, Mp: int = 0,
    gamma: float = 1.0, phi_fixed: bool = True,
    ldet_S: float = 0.0, ldet_S_grad: Optional[np.ndarray] = None,
    ldet_S_hess: Optional[np.ndarray] = None,
) -> dict:
    """mgcv ``Sl.fitChol`` (fast-REML.r:1348-1444) port — given ``XX =
    X'WX`` and ``Xy = X'Wy``, solve the penalised LS problem at fixed
    ``rho`` and return β plus the REML Newton step + grad + Hessian
    via the Implicit Function Theorem.

    The "POI" (Performance-Oriented Iteration) optimizer mgcv uses for
    ``discrete=TRUE`` calls this routine *once* per PIRLS iter to
    propose a single (rho, log φ) Newton step, with step-halving on
    the outside if the step is "uphill". By contrast hea's existing
    ``_outer_newton`` runs Newton to convergence at each fixed (W, z),
    which over-shoots when the basin is flat. Routing the
    ``discrete=TRUE`` PIRLS through ``_pi_fit_chol`` is what closes
    the residual auto-sp gap.

    The β solve uses diagonal preconditioning (``D = sqrt(diag(A))``)
    + pivoted Cholesky with mgcv's ``rank.tol = ε·100``. The gradient
    of REML w.r.t. ``rho`` is

        REML' = (∂log|A|/∂rho - ∂log|S|/∂rho
                 + (rss' + bSb')/(φ·γ)) / 2

    where ``rss' = 2 d_β/d_rho · A · d_β/d_rho ≈ 0`` at converged β
    (drops out by IFT, but kept for completeness) and ``bSb' = β'S_kβ
    + 2·β'S_k·d_β/d_rho_k``. Hessian similarly via second-order IFT.

    Args:
        XX: (p, p) X'WX.
        Xy: (p,) X'Wy.
        rho: (n_sp,) log smoothing params.
        slots: list of penalty slots, each with .col_start, .col_end, .S.
        p: total parameter count.
        yy: ‖√W·z‖² (only used when phi_fixed=False).
        log_phi: log φ.
        n: nobs.
        Mp: null-space dimension.
        gamma: γ inflation factor.
        phi_fixed: True for canonical-link families (Poisson, Binomial).
        ldet_S, ldet_S_grad, ldet_S_hess: log|S|_+ and its derivatives,
            computed externally and passed in (they don't depend on XX).

    Returns dict with:
        beta:        (p,) coefficients.
        grad:        (n_sp[+1 if !phi_fixed],) REML gradient.
        hess:        (n_sp[+1], n_sp[+1]) REML Hessian.
        step:        (n_sp[+1],) regularised Newton step (-H⁻¹g, capped).
        ldetXXS:     log|X'WX + Sλ| (rank-revealing pseudo-det).
        rank:        numerical rank of A.
        PP:          (p, p) ≈ A⁻¹ in original (un-pivoted) basis.
    """
    n_sp = len(slots)

    # 1. Build A = XX + Σ exp(rho_k) S_k_padded.
    A = XX.copy()
    sp = np.exp(rho)
    for k, slot in enumerate(slots):
        cs, ce = int(slot.col_start), int(slot.col_end)
        A[cs:ce, cs:ce] += sp[k] * slot.S
    A = 0.5 * (A + A.T)

    # 2. Diagonal preconditioning: D = sqrt(diag(A)).
    diag_A = np.diag(A).copy()
    d = np.where(diag_A > 0.0, np.sqrt(np.maximum(diag_A, 0.0)), 1.0)
    A_pre = (A / d) / d[:, None]
    A_pre = 0.5 * (A_pre + A_pre.T)

    # 3. Pivoted Cholesky on the preconditioned matrix.
    rank_tol = float(np.finfo(float).eps * 100.0)
    A_pre_f = np.asfortranarray(A_pre.copy())
    R_pre, piv_1based, rank_A, _info = dpstrf(
        A_pre_f, lower=0, tol=rank_tol,
    )
    R_pre = np.triu(R_pre)
    rank_A = int(rank_A)
    piv = np.asarray(piv_1based, dtype=int) - 1
    ipiv = np.empty(p, dtype=int)
    ipiv[piv] = np.arange(p)

    # 4. β solve in mgcv's gauge (zeros at rank-deficient pivoted
    #    positions, top-rank back-sub in preconditioned coords, then
    #    un-precondition).
    Xy_over_d = Xy / d
    beta_piv = np.zeros(p, dtype=float)
    if rank_A > 0:
        b_piv = Xy_over_d[piv]
        z = solve_triangular(
            R_pre[:rank_A, :rank_A].T, b_piv[:rank_A], lower=True,
        )
        beta_piv[:rank_A] = solve_triangular(
            R_pre[:rank_A, :rank_A], z, lower=False,
        )
    beta = beta_piv[ipiv] / d

    # 5. log|A| (rank-revealing).
    if rank_A > 0:
        ldetXXS = 2.0 * float(
            np.log(np.abs(np.diag(R_pre)[:rank_A])).sum()
        ) + 2.0 * float(np.log(d[piv[:rank_A]]).sum())
    else:
        ldetXXS = 0.0

    # 6. PP = A⁻¹ (rank-r pseudo-inverse) in preconditioned, pivoted
    #    coords, then un-pivot and un-precondition.
    if rank_A > 0:
        I_r = np.eye(rank_A)
        z_r = solve_triangular(
            R_pre[:rank_A, :rank_A].T, I_r, lower=True,
        )
        PP_pre_top = solve_triangular(
            R_pre[:rank_A, :rank_A], z_r, lower=False,
        )
    else:
        PP_pre_top = np.zeros((0, 0))
    PP_pre = np.zeros((p, p))
    PP_pre[:rank_A, :rank_A] = PP_pre_top
    PP = np.zeros((p, p))
    PP[np.ix_(piv, piv)] = PP_pre
    PP = (PP / d) / d[:, None]
    PP = 0.5 * (PP + PP.T)

    # 7. d_β/d_rho_k via IFT (mgcv ``Sl.iftChol``):
    #     d_β/d_rho_k = -A⁻¹ · (sp_k · S_k_padded · β)
    #    Using the pivoted/preconditioned chol structure:
    #     v = sp_k · S_k_padded · β              (length p)
    #     v_pp[piv] = (v / d)[piv]
    #     w = -backsolve(R, forwardsolve(R', v_pp[:rank]))
    #     d_β[piv][:rank] = w; d_β[piv][rank:] = 0
    #     d_β = (d_β[ipiv]) / d
    Skb = np.zeros((p, n_sp))
    for k, slot in enumerate(slots):
        cs, ce = int(slot.col_start), int(slot.col_end)
        Skb[cs:ce, k] = sp[k] * (slot.S @ beta[cs:ce])

    db = np.zeros((p, n_sp))
    if rank_A > 0:
        Skb_over_d = Skb / d[:, None]
        Skb_pre = Skb_over_d[piv, :]  # (p, n_sp), pivoted
        w_top = -solve_triangular(
            R_pre[:rank_A, :rank_A], solve_triangular(
                R_pre[:rank_A, :rank_A].T, Skb_pre[:rank_A, :], lower=True,
            ),
            lower=False,
        )
        db_piv = np.zeros((p, n_sp))
        db_piv[:rank_A, :] = w_top
        db = db_piv[ipiv, :] / d[:, None]

    # 8. b'Sb derivatives (Sl.iftChol):
    #     bSb1[k] = β' · sp_k · S_k_padded · β = β' Skb[:, k]
    bSb1 = np.einsum("i,ik->k", beta, Skb)
    # bSb2[k, j] = δ_kj · β' Skb[:,k]
    #            + 2·(db[:,k]' · (Skb[:,j] + S_db[:,j])
    #                 + db[:,j]' · Skb[:,k])
    # where S_db[:, k] = Σ_j sp_j S_j db[:, k] padded — but mgcv's Sl.mult
    # uses the *current-lambda* S so this is equivalent to
    # (A - XX) · db[:, k] (since A = XX + Σ sp_j S_j_padded).
    if n_sp > 0:
        S_db = np.zeros((p, n_sp))
        for k_inner in range(n_sp):
            for j, slot in enumerate(slots):
                cs, ce = int(slot.col_start), int(slot.col_end)
                S_db[cs:ce, k_inner] += (
                    sp[j] * (slot.S @ db[cs:ce, k_inner])
                )
        bSb2 = np.diag(bSb1) + 2.0 * (
            db.T @ (Skb + S_db) + Skb.T @ db
        )
        bSb2 = 0.5 * (bSb2 + bSb2.T)
    else:
        bSb2 = np.zeros((0, 0))

    # 9. rss' is 0 to first order at converged β (IFT). At PIRLS-
    #    converged β̂, the gradient w.r.t. rho_k of (½ rss) is just
    #    ½·d_β/d_rho_k · X'(Xβ−y) = -½·d_β/d_rho_k · S_λ β = ½·bSb1[k]
    #    via the score equation, so rss1 is rolled into bSb1.
    #    Following mgcv's convention exactly, rss1[k] = 0.
    rss1 = np.zeros(n_sp)
    # rss2[k, j] = 2 · d_β[:,k]' · XX · d_β[:,j]
    if n_sp > 0:
        XX_db = XX @ db
        rss2 = 2.0 * (db.T @ XX_db)
        rss2 = 0.5 * (rss2 + rss2.T)
    else:
        rss2 = np.zeros((0, 0))

    # 10. log|XX+S| derivatives via d.detXXS (fast-REML.r:1219-1237).
    #     d1[k] = sp_k · tr(S_k · PP[block, block])  (= sp_k * tr(S_k_padded · PP))
    #     d2[k, j] = -tr((sp_j · S_j · PP)[block_k, block_j]
    #                  · (sp_k · S_k · PP)[block_j, block_k]) + δ_kj·d1[k]
    SPP = np.zeros((p, p, n_sp))
    for k, slot in enumerate(slots):
        cs, ce = int(slot.col_start), int(slot.col_end)
        SPP[cs:ce, :, k] = sp[k] * (slot.S @ PP[cs:ce, :])
    dXXS_d1 = np.zeros(n_sp)
    for k, slot in enumerate(slots):
        cs, ce = int(slot.col_start), int(slot.col_end)
        dXXS_d1[k] = float(np.trace(SPP[cs:ce, cs:ce, k]))
    dXXS_d2 = np.zeros((n_sp, n_sp))
    for i in range(n_sp):
        cs_i, ce_i = int(slots[i].col_start), int(slots[i].col_end)
        for j in range(i, n_sp):
            cs_j, ce_j = int(slots[j].col_start), int(slots[j].col_end)
            # sum over col_start_i:col_end_i (rows) and col_start_j:col_end_j (cols)
            v = -float(
                np.sum(
                    SPP[cs_i:ce_i, cs_j:ce_j, i].T *
                    SPP[cs_j:ce_j, cs_i:ce_i, j]
                )
            )
            dXXS_d2[i, j] = dXXS_d2[j, i] = v
        dXXS_d2[i, i] += dXXS_d1[i]

    # 11. REML gradient and Hessian (rho-only; log φ added below if free).
    phi = float(np.exp(log_phi))
    if ldet_S_grad is None:
        ldet_S_grad = np.zeros(n_sp)
    if ldet_S_hess is None:
        ldet_S_hess = np.zeros((n_sp, n_sp))
    grad = (
        dXXS_d1 - ldet_S_grad
        + (rss1 + bSb1) / (phi * gamma)
    ) / 2.0
    hess = (
        dXXS_d2 - ldet_S_hess
        + (rss2 + bSb2) / (phi * gamma)
    ) / 2.0

    # 12. log φ slot for non-fixed scale (Gaussian etc.).
    if not phi_fixed:
        rss_bSb = float(yy - beta @ Xy)
        grad_phi = (-rss_bSb / (phi * gamma) + n / gamma - Mp) / 2.0
        grad = np.concatenate([grad, [grad_phi]])
        # cross derivatives w.r.t. log φ
        d_phi = np.concatenate([
            -(rss1 + bSb1), [rss_bSb],
        ]) / (2.0 * phi * gamma)
        n_old = hess.shape[0]
        hess_new = np.zeros((n_old + 1, n_old + 1))
        hess_new[:n_old, :n_old] = hess
        hess_new[:n_old, n_old] = d_phi[:n_old]
        hess_new[n_old, :n_old] = d_phi[:n_old]
        hess_new[n_old, n_old] = d_phi[n_old]
        hess = hess_new

    # 13. Newton step from eigen-regularised Hessian (Sl.fitChol:1430-1440).
    if hess.shape[0] > 0:
        eig_w, eig_v = np.linalg.eigh(hess)
        eig_w_abs = np.abs(eig_w)
        if eig_w_abs.size > 0:
            me = float(eig_w_abs.max() * float(np.finfo(float).eps) ** 0.5)
            eig_w_clamped = np.where(eig_w_abs < me, me, eig_w_abs)
        else:
            eig_w_clamped = eig_w_abs
        step = -eig_v @ ((eig_v.T @ grad) / eig_w_clamped)
        # Cap |step| <= 4 (Sl.fitChol:1438-1439).
        ms = float(np.max(np.abs(step))) if step.size else 0.0
        if ms > 4.0:
            step = step * (4.0 / ms)
    else:
        step = np.zeros(0)

    return {
        "beta": beta,
        "grad": grad,
        "hess": hess,
        "step": step,
        "ldetXXS": ldetXXS,
        "rank": rank_A,
        "PP": PP,
        "R_pre": R_pre,
        "d": d,
        "piv": piv,
        "ipiv": ipiv,
    }


def _chol2qr(XX: np.ndarray, Xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``X'X``, ``X'y`` into ``R``, ``f`` such that ``R'R = X'X``,
    ``R'f = X'y``.

    Port of mgcv ``chol2qr`` (bam.r:31-44). Uses LAPACK pivoted Cholesky
    (``dpstrf``) so rank-deficient PSD inputs are handled correctly. For
    a rank-deficient ``XX`` the bottom rows of ``R_piv`` (after pivoting,
    the rows corresponding to dropped pivots) are zeroed out, and ``f``
    is forward-solved on the top-``rank`` subsystem only with the
    pivoted-bottom positions padded by zero. This is mgcv's exact
    convention (``R[(rank+1):p,] <- 0`` then ``f <- c(forwardsolve(...),
    rep(0, p-rank))[ipiv]``).

    Note: an earlier version of this routine replaced the bottom-right
    ``(p-r) × (p-r)`` block with the identity matrix so a single full
    forward-solve was non-singular. That broke the gram identity for
    rank-deficient inputs — ``R'R`` then equalled ``XX + I_at_pivoted_
    bottom_positions``, biasing every downstream PIRLS solve by exactly
    1.0 on the dropped diagonals (verified on the small_data Poisson
    te(pm10, lag) fit, where two diagonals of ``R'R - XX`` came out at
    exactly 1.0). mgcv leaves those rows zero instead and only solves
    the top-rank subsystem; we now do the same.

    Output ``R`` is in original (un-pivoted) column ordering: column ``j``
    of ``R`` corresponds to column ``j`` of ``X``. ``R`` is *not*
    upper-triangular after un-pivoting (just like the chunked path's
    ``_qr_update`` output), but the gram identities ``R'R = XX`` and
    ``R'f = Xy`` hold exactly (the second by consistency: when ``Xy``
    lies in ``range(XX)``, the rank-``r`` forward-solve makes the bottom
    rows of ``R_piv'·f_piv`` automatically equal to ``Xy[piv][r:]``).
    """
    XX = np.asarray(XX, dtype=float)
    Xy = np.asarray(Xy, dtype=float).ravel()
    p = Xy.shape[0]
    if p == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

    XX_sym = 0.5 * (XX + XX.T)
    # ``dpstrf`` overwrites the input — pass a contiguous Fortran copy.
    A = np.asfortranarray(XX_sym.copy())
    c, piv_1based, rank, info = dpstrf(A, lower=0)
    R_piv = np.triu(c)
    # ``R_piv' R_piv = XX[piv, :][:, piv]`` (DPSTRF spec). For rank<p,
    # the trailing block has ~zero diag and DPSTRF leaves garbage in its
    # rows; mgcv (bam.r:40) zeros those rows so ``R_piv' R_piv`` equals
    # ``XX[piv, piv]`` exactly (modulo float noise).
    if rank < p:
        R_piv[rank:, :] = 0.0

    piv = np.asarray(piv_1based, dtype=int) - 1   # 0-based
    ipiv = np.empty(p, dtype=int)
    ipiv[piv] = np.arange(p)

    # mgcv bam.r:41: ``f <- c(forwardsolve(t(R[ind,ind]), Xy[piv[ind]]),
    #                        rep(0, p-rank))[ipiv]``. We compute the
    # pivoted ``f_piv`` here (top-rank from forwardsolve, bottom zeros)
    # and downstream ``R'f`` over our column-only-unpivoted ``R`` lands
    # the same Xy as mgcv's row+col-unpivoted ``R'f`` would (verified
    # by carrying through the index permutation: see docstring).
    f = np.zeros(p, dtype=float)
    if rank > 0:
        f[:rank] = solve_triangular(
            R_piv[:rank, :rank].T, Xy[piv][:rank], lower=True,
        )
    R_out = R_piv[:, ipiv]
    return R_out, f


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


def _smooth_specs_from_expanded(expanded, data: pl.DataFrame) -> list[dict]:
    """Build the ``discrete_mf`` ``smooth_specs`` list from an expanded
    formula, mirroring how mgcv ``bam.r:2206-2215`` derives ``dk`` directly
    from the formula (not from already-built smooth blocks).

    For each smooth call:
      * ``term`` — full list of arg variables (``_smooth_term_vars``).
      * ``by``  — column name from ``by=`` (None if unset / NA / non-name).
      * ``margins`` — for ``te``/``ti``/``t2`` parsed via
        ``_te_parse_margins`` (honors the ``d=c(...)`` kwarg); for
        ``s(...)`` a single margin spanning all vars.

    Used by :class:`bam` for the ``discrete=True`` setup path: the
    discretised model frame is built before ``materialize_smooths`` so the
    smooth basis construction runs on the padded scalar mf0, not on the
    matrix-arg long form.
    """
    from .formula import (
        _smooth_term_vars, _smooth_by_expr, _te_parse_margins, _apply_tero,
    )
    out: list[dict] = []
    for call in expanded.smooths:
        term_vars = _smooth_term_vars(call)
        by_expr = _smooth_by_expr(call)
        # discrete_mf only handles plain-column by= — drop complex exprs.
        if by_expr is not None and by_expr not in data.columns:
            by_expr = None
        if call.fn in ("te", "ti", "t2"):
            te_specs = _te_parse_margins(call, data)
            # tero (bam.r:1900-1917, called at bam.r:2109) — discrete=True
            # only. Putting the largest-k margin last makes ``compress_df``
            # process small-pool margins first, matching mgcv's MT-state
            # consumption sequence so the per-margin shuffle order agrees.
            te_specs = _apply_tero(te_specs)
            margins = [{"term": list(s["term"])} for s in te_specs]
        else:
            margins = [{"term": term_vars}]
        out.append({"term": term_vars, "by": by_expr, "margins": margins})
    return out


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


@dataclass
class _PirlsQR:
    """Output of one PIRLS-step chunked accumulation. Carries the reduced
    sufficient statistics ``(R, f, y_norm2, rss_extra)`` plus the full-length
    quantities (``eta``, ``mu``, ``wt``, ``dev``) needed by the outer PIRLS
    loop's divergence test and post-fit step."""
    R: np.ndarray
    f: np.ndarray
    y_norm2: float
    rss_extra: float
    eta: np.ndarray         # full η (length n) — *with* offset
    mu: np.ndarray          # length-n
    wt: np.ndarray          # length-n PIRLS weights (Fisher form)
    z: np.ndarray           # length-n working response (offset-stripped)
    dev: float              # Σ family.dev_resids(y, μ, w_prior)


def _build_qr_chunked_pirls(
    data: pl.DataFrame,
    blocks: list[SmoothBlock],
    X_param_full: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    family: Family,
    *,
    coef: Optional[np.ndarray],
    eta_init: Optional[np.ndarray],
    chunk_size: int,
    use_chol: bool = False,
    prior_w: Optional[np.ndarray] = None,
) -> _PirlsQR:
    """One PIRLS-step chunked QR build for non-Gaussian families.

    Mirrors mgcv ``bgam.fit`` inner accumulation (bam.r:1059-1099). For each
    chunk:

    * Materialise ``X_chunk`` (parametric + smooth columns).
    * Compute the chunk's η. If ``coef is not None`` use ``η = X·β + offset``
      (mgcv bam.r:1066); otherwise fall back to the supplied ``eta_init``
      (the family-initialised η used on iter 1).
    * Form Fisher working response and weights (mgcv bam.r:1078-1083):

          z = (η − offset) + (y − μ) / μ_η
          w = w_prior · μ_η² / V(μ)

    * Drop rows where ``w_prior > 0 & μ_η != 0`` is false (mgcv's ``good``
      mask, bam.r:1080).
    * Accumulate ``√w · X_good`` and ``√w · z_good`` into ``(R, f, ‖z‖²)``
      via :func:`_qr_update`.
    * Sum chunkwise deviance via ``family.dev_resids(y, μ, w_prior)``.

    Returns a :class:`_PirlsQR` carrying both the reduced sufficient
    statistics and the full-length (η, μ, w, z, dev) needed by the outer
    PIRLS step-halving and convergence checks.

    The Newton-form α is **not** applied here — mgcv uses Fisher weights
    inside the PIRLS loop (gam.fit3.r:270). Newton α enters only at the
    post-fit score-derivative stage to make the converged Hessian match
    the observed-info form (Wood 2011 §3.3).
    """
    n = data.height
    if n == 0:
        raise ValueError("empty data passed to chunked PIRLS build")
    if (coef is None) and (eta_init is None):
        raise ValueError("either coef or eta_init must be provided")
    if prior_w is None:
        prior_w = np.ones(n, dtype=float)

    link = family.link
    R: Optional[np.ndarray] = None
    f: Optional[np.ndarray] = None
    y_norm2 = 0.0
    eta_full = np.empty(n, dtype=float)
    mu_full = np.empty(n, dtype=float)
    wt_full = np.zeros(n, dtype=float)   # mgcv ``wt`` carries 0 for !good rows
    z_full = np.zeros(n, dtype=float)
    dev_total = 0.0

    for start, end in _chunk_indices(n, chunk_size):
        chunk_data = data[start:end]
        X_param_chunk = X_param_full[start:end]
        X_chunk = _materialize_chunk(blocks, chunk_data, X_param_chunk)
        off_chunk = offset[start:end]
        y_chunk = y[start:end]
        wp_chunk = prior_w[start:end]

        # mgcv bam.r:1066: ``if (is.null(coef)) eta1 <- eta[ind] else
        # eta[ind] <- eta1 <- drop(X %*% coef) + offset[ind]``.
        if coef is None:
            eta_chunk = eta_init[start:end]
        else:
            eta_chunk = X_chunk @ coef + off_chunk

        mu_chunk = link.linkinv(eta_chunk)
        mu_eta_chunk = link.mu_eta(eta_chunk)
        V_chunk = family.variance(mu_chunk)

        # ``good`` mask (mgcv bam.r:1080).
        good = (wp_chunk > 0) & (mu_eta_chunk != 0)
        # Avoid div-by-zero in the score computation; ``!good`` rows are
        # dropped before the QR update so the placeholder values don't
        # leak into (R, f).
        safe_mu_eta = np.where(mu_eta_chunk != 0, mu_eta_chunk, 1.0)
        safe_V = np.where(V_chunk != 0, V_chunk, 1.0)

        z_chunk = (eta_chunk - off_chunk) + (y_chunk - mu_chunk) / safe_mu_eta
        w_chunk = wp_chunk * mu_eta_chunk * mu_eta_chunk / safe_V
        # mgcv bam.r:1085: ``w[!good] <- 0``.
        w_chunk = np.where(good, w_chunk, 0.0)

        dev_total += float(np.sum(family.dev_resids(y_chunk, mu_chunk, wp_chunk)))

        eta_full[start:end] = eta_chunk
        mu_full[start:end] = mu_chunk
        wt_full[start:end] = w_chunk
        z_full[start:end] = z_chunk

        if not np.any(good):
            # All rows dropped — skip the QR update for this chunk.
            continue
        sqrt_w = np.sqrt(w_chunk[good])
        Xg = sqrt_w[:, None] * X_chunk[good]
        zg = sqrt_w * z_chunk[good]
        upd = _qr_update(Xg, zg, R, f, y_norm2, use_chol=use_chol)
        R = upd["R"]
        f = upd["f"]
        y_norm2 = upd["y_norm2"]

    if R is None:
        raise FloatingPointError(
            "chunked PIRLS build accumulated zero rows — every observation "
            "was dropped by the (w_prior > 0 & μ_η != 0) good mask"
        )
    if use_chol:
        R, f = _chol2qr(R, f)
    rss_extra = float(y_norm2 - float(f @ f))
    return _PirlsQR(
        R=np.asarray(R, dtype=float),
        f=np.asarray(f, dtype=float),
        y_norm2=float(y_norm2),
        rss_extra=rss_extra,
        eta=eta_full, mu=mu_full, wt=wt_full, z=z_full,
        dev=dev_total,
    )


def _build_qr_discrete_pirls(
    design: DiscreteDesign,
    y: np.ndarray,
    offset: np.ndarray,
    family: Family,
    *,
    coef: Optional[np.ndarray],
    eta_init: Optional[np.ndarray],
    use_chol: bool = False,
    prior_w: Optional[np.ndarray] = None,
) -> _PirlsQR:
    """One PIRLS-step build for ``bam(..., discrete=True)``.

    Direct port of the inner accumulation in mgcv ``bgam.fitd``
    (bam.r:530-620). Mirrors :func:`_build_qr_chunked_pirls` in shape
    (returns the same :class:`_PirlsQR`) but runs the PIRLS-step
    sufficient-statistics build via the discrete kernels:

    * ``η = Xβ + offset`` via :func:`Xbd` (or use ``eta_init`` on iter 1
      when β is still unknown — same convention as the chunked path).
    * Form Fisher working response ``z`` and weights ``W`` per row.
    * Drop ``!(w_prior > 0 & μ_η ≠ 0)`` rows by zeroing their weight.
    * ``X'WX`` via :func:`XWXd`, ``X'Wz`` via :func:`XWyd`, then convert to
      ``(R, f)`` via :func:`_chol2qr`.
    * ``y_norm2 = Σ wᵢ·zᵢ²`` (the working-response sum-of-squares — for
      Gaussian-identity this collapses to ``Σ (yᵢ-offᵢ)²``).

    Relies on the design-level ``X_full`` cache built by
    :func:`discrete_full_X` on first access. ``Xd_list`` is invariant
    across PIRLS iters, so the cached X persists across every outer
    Newton step at no extra cost. The optimised scatter-add kernels in
    ``Xbd`` / ``XWXd`` / ``XWyd`` (``use_kernel=True``) remain
    available for very-large-n cases where the full ``n × p`` matrix
    no longer fits — current default is BLAS-on-cached-X, which beats
    the kernels at chicago-scale sizes.
    """
    n = design.n
    if n == 0:
        raise ValueError("empty data passed to discrete PIRLS build")
    if (coef is None) and (eta_init is None):
        raise ValueError("either coef or eta_init must be provided")
    if prior_w is None:
        prior_w = np.ones(n, dtype=float)

    link = family.link

    # mgcv bam.r:537-539: ``if (is.null(coef)) eta1 <- eta else
    # eta1 <- Xbd(...) + offset``.
    if coef is None:
        eta_full = np.asarray(eta_init, dtype=float)
    else:
        eta_full = Xbd(design, np.asarray(coef, dtype=float)) + offset

    mu_full = link.linkinv(eta_full)
    mu_eta = link.mu_eta(eta_full)
    V_full = family.variance(mu_full)

    good = (prior_w > 0) & (mu_eta != 0)
    safe_mu_eta = np.where(mu_eta != 0, mu_eta, 1.0)
    safe_V = np.where(V_full != 0, V_full, 1.0)

    z_full = (eta_full - offset) + (y - mu_full) / safe_mu_eta
    w_full = prior_w * mu_eta * mu_eta / safe_V
    w_full = np.where(good, w_full, 0.0)
    if not np.any(good):
        raise FloatingPointError(
            "discrete PIRLS build saw zero good rows — every observation "
            "was dropped by the (w_prior > 0 & μ_η != 0) good mask"
        )

    dev_total = float(np.sum(family.dev_resids(y, mu_full, prior_w)))

    XWX = XWXd(design, w_full)
    Xy = XWyd(design, w_full, z_full)
    y_norm2 = float(np.sum(w_full * z_full * z_full))

    R, f = _chol2qr(XWX, Xy)
    rss_extra = float(y_norm2 - float(f @ f))

    return _PirlsQR(
        R=np.asarray(R, dtype=float),
        f=np.asarray(f, dtype=float),
        y_norm2=y_norm2,
        rss_extra=rss_extra,
        eta=eta_full, mu=mu_full, wt=w_full, z=z_full,
        dev=dev_total,
    )


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
        data,
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
        discrete: bool = False,
        discrete_m: int | None = None,
    ):
        # ``data`` may be a polars DataFrame OR a mapping of name → 1-D /
        # 2-D ndarray. 2-D entries become matrix columns for mgcv's
        # summation convention (Wood §7.4.1 distributed-lag models).
        # ``prepare_design`` normalizes via ``normalize_data``.
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
        # mgcv bam.r:2360-2361 — AR1 only valid with Gaussian-identity errors.
        if rho != 0.0 and not _is_identity_link(family):
            raise NotImplementedError(
                "AR coefficients (rho != 0) require family=Gaussian(link='identity')"
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
        self._discrete = bool(discrete)
        self._discrete_m = discrete_m
        if self._discrete and self._rho != 0.0:
            raise NotImplementedError(
                "discrete=True with AR1 (rho != 0) is not yet supported"
            )
        self._discrete_design: Optional[DiscreteDesign] = None
        self._discrete_frame: Optional[DiscretizedFrame] = None

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
            if self._discrete:
                # mgcv-T flow (bam.r:2206-2232): basis setup runs on the
                # discretised scalar ``mf0 = dk$mf`` (padded to ``maxr``),
                # not on the matrix-arg long form. Build smooth_specs from
                # the formula directly, run ``discrete_mf`` to get the
                # padded scalar columns, then ``materialize_smooths`` on
                # that scalar frame. ``sparse.cons=0`` ⇒ Householder QR
                # absorb on the padded ``colMeans``.
                smooth_specs_pre = _smooth_specs_from_expanded(
                    d.expanded, self.data,
                )
                # mgcv's ``pmf.names = names(model.frame(parametric_formula,
                # data))``, which *includes the response label* (because R's
                # ``model.frame(y ~ x)`` evaluates the LHS into a column).
                # ``discrete.mf`` then loops over those names and runs
                # ``compress.df`` on each — including y. Skipping the
                # response leaves the RNG state desynced by the unique-value
                # count of y at the pad loop, breaking bit-exact parity.
                # Build mgcv's ``pmf.names`` order: response first, then
                # parametric data covariates (skipping the synthetic
                # ``(Intercept)`` column).
                names_pmf: list[str] = []
                if d.response and d.response not in names_pmf:
                    names_pmf.append(d.response)
                for col in X_param_df.columns:
                    if col == "(Intercept)" or col in names_pmf:
                        continue
                    if col in self.data.columns:
                        names_pmf.append(col)
                # Ensure the evaluated response is available as a column on
                # the data frame passed to ``discrete_mf``. For a bare
                # ``y ~ ...`` formula this is a no-op; for ``log(y) ~ ...``
                # we attach the deparsed name with the evaluated values
                # (matching ``model.frame(log(y) ~ ...)`` in R).
                data_for_discrete = self.data
                if (d.response
                        and d.response not in self.data.columns):
                    data_for_discrete = self.data.with_columns(
                        pl.Series(name=d.response, values=y_full)
                    )
                self._discrete_frame = discrete_mf(
                    smooth_specs_pre, data_for_discrete,
                    names_pmf=names_pmf,
                    m=self._discrete_m,
                )
                mf_dict = {
                    nm: arr for nm, arr in self._discrete_frame.mf.items()
                    if nm != "(Intercept)"
                }
                mf0 = pl.DataFrame(mf_dict) if mf_dict else pl.DataFrame()
                sb_lists = materialize_smooths(
                    d.expanded, mf0, sparse_cons=0, tero=True,
                )
            else:
                # discrete=FALSE: basis setup on a representative subsample
                # of the original (possibly matrix-arg) data; sparse.cons=-1
                # (sweep-drop absorb on row-summed colMeans).
                mf0 = _mini_mf(self.data, chunk_size)
                sb_lists = materialize_smooths(
                    d.expanded, mf0, sparse_cons=-1,
                )
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

        # ---- discrete (compressed) design ----------------------------------
        # ``self._discrete_frame`` was populated upstream (before
        # materialize_smooths) so the basis specs were fitted on the same
        # padded scalar mf0 mgcv-T uses for ``smoothCon`` (bam.r:2206-2232).
        # Here we just hand the frozen blocks + frame to
        # ``build_discrete_design`` to assemble the per-marginal Xd table.
        if self._discrete:
            assert self._discrete_frame is not None
            self._discrete_design = build_discrete_design(
                blocks, X_param_full, self._discrete_frame,
                data=self.data,
            )

        # ---- family-independent post-setup --------------------------------
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

        self._log_phi_hat: float | None = None
        self._outer_info: dict | None = None
        self._tw_info: dict | None = None

        # ---- family dispatch ----------------------------------------------
        # Gaussian-identity (am=TRUE in mgcv) takes the closed-form chunked
        # QR path: build (R, f, rss_extra) once, run outer Newton on the
        # reduced data, then a single chunked walk for full-n quantities.
        # Mirrors mgcv ``bam.fit`` (bam.r:1503-1771).
        #
        # All other families take the PIRLS chunked path: outer loop alternates
        # rebuilding (R, f) from chunks of √W·X / √W·z (PIRLS Fisher weights
        # at the current β̂) with sp optimisation on the fixed reduced data.
        # Mirrors mgcv ``bgam.fit`` (bam.r:909-1353).
        #
        # ``discrete=True`` short-circuits both: bgam.fitd unifies all
        # families on the same PIRLS scaffold, but rebuilds (X'WX, X'Wz)
        # via the discrete kernels instead of a chunked QR pass. For
        # Gaussian-identity this still converges in one PIRLS iter
        # because z = y - offset and W = I are constant.
        if _is_identity_link(family) and not self._discrete:
            # ---- chunked QR build (Gaussian-identity) -----------------------
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

            # ---- smoothing-param optimization ---------------------------------
            # Same outer Newton as gam, but every PIRLS-replacement call to
            # ``_fit_given_rho`` here goes through the override below.
            n_sp = len(slots)
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

            # ---- post-fit assembly (Gaussian-identity) ----------------------
            # Most of gam.__init__'s post-fit code reads ``self._X_full`` and
            # ``fit.mu``/``fit.eta``. With ``_X_full = R`` and Gaussian-identity
            # there are no PIRLS weights to rebuild; the things that need full-n
            # quantities (eta, mu, residuals, leverage) are computed via a
            # single chunked walk below.
            self._post_fit_gaussian(fit, rho_hat, X_param_df)
        else:
            # ---- non-Gaussian PIRLS chunked path (mgcv bgam.fit) ------------
            fit, rho_hat = self._bgam_fit_loop(sp_user=sp)
            self._post_fit_pirls(fit, rho_hat, X_param_df)

    # -----------------------------------------------------------------------
    # _fit_given_rho override — uses (R, f, y_norm2, rss_extra)
    # -----------------------------------------------------------------------

    def _fit_given_rho(self, rho: np.ndarray) -> "_FitState":
        """Closed-form Gaussian-on-(R, f) solve at fixed ρ.

        For Gaussian-identity (``am=TRUE``) the chunked QR build stores
        ``(R, f, ‖y−off‖², rss_extra)`` and PIRLS reduces to one solve:

            (R'R + Sλ) β̂ = R'f                            # normal equations
            ‖z − Xβ̂‖²   = ‖f − Rβ̂‖² + rss_extra           # working-RSS

        For non-Gaussian (PIRLS path) the same identity holds with
        ``z = (η − off) + (y − μ)/μ_η`` and weights ``W = w_prior μ_η²/V`` —
        the chunked PIRLS build stores ``(R, f, ‖√W·z‖², rss_extra)`` for
        the *current* working data, so this solve produces β̂ at the next
        Newton step on the IRLS-linearised problem. The non-Gaussianness
        is in the *outer* PIRLS loop (rebuilding R/f), not in the inner
        score evaluation, which is faithful to mgcv ``fast.REML.fit`` /
        ``magic`` running on the reduced data.

        ``fit.mu`` is the *response-scale* μ = linkinv(η) (not the working
        response). ``_score_scale`` reads it against ``self._y_arr`` to
        compute the Pearson sum used by the outer-Newton convergence
        check; for non-Gaussian families the link inverse is required.
        ``is_fisher_fallback=True`` keeps Newton≡Fisher for the
        Gaussian-on-(R,f) inner score, and bam's overridden ``_dw_deta``
        returns ``zeros(p)`` (length-p so the broadcast against
        ``self._X_full = R`` lines up).

        Rank handling (mgcv gam.fit3 / gdi1 style): we run pivoted
        Cholesky on ``A = R'R + Sλ``. When ``A`` is rank-deficient (the
        smoothing penalty doesn't fully regularise the unpenalised null
        space — e.g. te-only Poisson on small_data has rank(A) = 14 of
        15), the rank-deficient pivoted positions get β = 0 in mgcv's
        gauge. ``log|A|`` is the rank-revealing pseudo-determinant
        (sum of log of positive pivots), which mgcv's REML score reads.
        For full-rank ``A`` this collapses to the regular Cholesky path;
        no extra cost in the common case.
        """
        Sλ = self._build_S_lambda(rho)
        Sλ = 0.5 * (Sλ + Sλ.T)
        A = self._XtX + Sλ
        A = 0.5 * (A + A.T)

        # mgcv ``Sl.fitChol`` (fast-REML.r:1367-1370) preconditions
        # ``A = XX + Sλ`` by ``D = sqrt(diag(A))`` *before* pivoted
        # Cholesky:
        #     A_pre = D⁻¹ A D⁻¹           (unit-diagonal up to noise)
        #     R = chol(A_pre, pivot=TRUE)
        #     β[piv] = backsolve(R, forwardsolve(R', (Xy/D)[piv])) / D[piv]
        # Without preconditioning, ``dpstrf``'s rank determination uses
        # the relative ``A[i,i] / max(A[k,k])`` ratio, which can drop or
        # keep the small-eigenvalue direction depending on column scaling
        # (and that scaling drifts with ``rho``). With preconditioning all
        # diagonals become 1, so the rank tolerance acts on the relative
        # eigenvalue spread — that's mgcv's gauge.
        diag_A = np.diag(A).copy()
        d = np.where(diag_A > 0.0, np.sqrt(np.maximum(diag_A, 0.0)), 1.0)
        # A_pre[i, j] = A[i, j] / (d[i] * d[j])
        A_pre = (A / d) / d[:, None]
        A_pre = 0.5 * (A_pre + A_pre.T)

        # Pivoted Cholesky with rank revealing (mgcv ``chol(A_pre,
        # pivot=TRUE)``). mgcv uses ``rank.tol = .Machine$double.eps *
        # 100 ≈ 2.22e-14`` (gam.fit3.r:131); we mirror that so dpstrf's
        # rank determination matches mgcv's.
        rank_tol = float(np.finfo(float).eps * 100.0)
        A_pre_f = np.asfortranarray(A_pre.copy())
        R_pre, piv_1based, rank_A, _info = dpstrf(
            A_pre_f, lower=0, tol=rank_tol,
        )
        R_pre = np.triu(R_pre)
        rank_A = int(rank_A)
        piv = np.asarray(piv_1based, dtype=int) - 1
        ipiv = np.empty(self.p, dtype=int)
        ipiv[piv] = np.arange(self.p)

        # Solve in mgcv's pseudo-inverse gauge with the preconditioning
        # un-applied at the end:
        #     β[piv][:rank] = backsolve(R, forwardsolve(R', (Xy/D)[piv][:rank]))
        #     β[piv][rank:] = 0
        #     β = β / D                       (un-precondition)
        Xy_over_d = self._Xty / d
        if rank_A > 0:
            b_piv = Xy_over_d[piv]
            z = solve_triangular(
                R_pre[:rank_A, :rank_A].T, b_piv[:rank_A], lower=True,
            )
            beta_piv_top = solve_triangular(
                R_pre[:rank_A, :rank_A], z, lower=False,
            )
        else:
            beta_piv_top = np.zeros(0, dtype=float)
        beta_piv = np.zeros(self.p, dtype=float)
        beta_piv[:rank_A] = beta_piv_top
        # Un-pivot, then un-precondition.
        beta = beta_piv[ipiv] / d

        if not np.all(np.isfinite(beta)):
            raise FloatingPointError("non-finite β in bam (R, f) solve")

        # ``A_chol``/``A_chol_lower`` are consumed by every downstream
        # variance / Newton-step / hat-matrix routine via
        # ``cho_solve((A_chol, lower), …)`` — they expect a *triangular*
        # factor in the *original* column basis. The pivoted Chol
        # ``R_piv`` is triangular in *pivoted* basis only; once we
        # un-pivot, triangularity is lost, breaking the
        # ``solve_triangular`` callsites in ``_make_K``.
        #
        # Strategy: rebuild a non-pivoted Cholesky of ``A`` for storage,
        # falling back to a tiny ridge when the standard Cholesky fails
        # on the rank-deficient direction. The β / log_det that drive
        # the *outer* optimiser were already computed above via the
        # rank-revealing pivoted path, so the ridge here is only seen by
        # the variance-estimator code (which mgcv computes via a
        # different gdi1-internal routine anyway). The bias lives along
        # the dropped null direction and decays with sp magnitude.
        try:
            A_chol, lower = cho_factor(A, lower=True, overwrite_a=False)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.trace(A) / max(self.p, 1)
            A_chol, lower = cho_factor(
                A + ridge * np.eye(self.p),
                lower=True, overwrite_a=False,
            )

        pen = float(beta @ Sλ @ beta)
        # Full-data working RSS = ‖z̃ − X̃β̂‖² = ‖f − Rβ̂‖² + rss_extra
        #                       = ‖z̃‖² − 2 β̂' R'f + β̂' R'R β̂
        # (z̃ = √W·z, X̃ = √W·X for non-Gaussian; W=I, z = y−off for Gaussian).
        rss = float(
            self._yty - 2.0 * (beta @ self._Xty) + beta @ self._XtX @ beta
        )
        rss = max(rss, 0.0)  # guard tiny negative from cancellation
        # Rank-revealing log|A|. With the preconditioning, log|A| =
        # log|D R_pre' R_pre D| = 2·Σ log|diag(R_pre)[:rank]|
        #                       + 2·Σ log d[piv][:rank]
        # mirroring mgcv ``Sl.fitChol``'s
        # ``ldetXXS = 2*sum(log(diag(R)) + log(d[piv]))`` (fast-REML.r:1391).
        if rank_A > 0:
            log_det_A = 2.0 * float(
                np.log(np.abs(np.diag(R_pre)[:rank_A])).sum()
            ) + 2.0 * float(np.log(d[piv[:rank_A]]).sum())
        else:
            log_det_A = 0.0
        # ``_score_scale`` reads ``fit.mu`` vs ``self._y_arr`` for the
        # Pearson sum — must be length-n response-scale μ. Recovered via
        # a chunked ``X·β`` walk per call (O(n·p), same cost gam pays for
        # ``eta = X @ β`` every outer-Newton iteration). For non-Gaussian
        # bam this also gives the response-scale μ at the current β,
        # which the downstream score-scale calc needs.
        eta_only = self._chunked_xbeta(beta)        # X·β (offset-stripped)
        eta = eta_only + self._offset               # full η, length-n
        if isinstance(self.family, Gaussian) and self.family.link.name == "identity":
            mu = eta                                # identity link short-circuit
            z_full = self._y_arr - self._offset
        else:
            mu = self.family.link.linkinv(eta)
            # Working response on the response-scale; the score-derivative
            # consumers don't read fit.z (bam's _dw_deta/_d2w_deta2 are
            # already overridden to zeros), so the value here is informational.
            mu_eta = self.family.link.mu_eta(eta)
            safe_mu_eta = np.where(mu_eta != 0, mu_eta, 1.0)
            z_full = (eta - self._offset) + (self._y_arr - mu) / safe_mu_eta
        n = self.n
        return _FitState(
            beta=beta, dev=rss, pen=pen,
            A_chol=A_chol, A_chol_lower=lower,
            S_full=Sλ, log_det_A=log_det_A,
            eta=eta, mu=mu, w=np.ones(n),
            z=z_full, alpha=np.ones(n),
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
        time, ``O(chunk_size·p)`` peak memory.

        For ``discrete=True`` this delegates to :func:`Xbd` against the
        compressed design — same answer, but goes through the
        per-marginal Xd gather instead of materialising chunks.
        """
        if self._discrete_design is not None:
            return Xbd(self._discrete_design, beta)
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
            # AR1 correction (mgcv bam.r:1715, 1737). The AR1 transform
            # changes the log-determinant of the response covariance by
            # ``(n - df) · log(ld)`` where ``ld = 1/√(1-ρ²)`` and ``df``
            # is the number of independent AR sequences (1 if ar_start
            # is None, else ``sum(ar_start)``). mgcv subtracts that from
            # gcv.ubre (which holds the score V, not 2V); ``self._reml``
            # returns 2V, so we double the correction here. The shift is
            # constant in (sp, log φ), so the outer Newton optimum is
            # unaffected.
            if self._rho != 0.0 and np.isfinite(score):
                ld = 1.0 / np.sqrt(1.0 - self._rho ** 2)
                df_ar = (
                    int(self._ar_start.sum())
                    if self._ar_start is not None else 1
                )
                score = score - 2.0 * (n - df_ar) * float(np.log(ld))
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

    # -----------------------------------------------------------------------
    # Non-Gaussian PIRLS chunked — outer loop driver (mgcv bgam.fit)
    # -----------------------------------------------------------------------

    def _chunked_leverage_diag_weighted(self, A_inv: np.ndarray,
                                        w_full: np.ndarray) -> np.ndarray:
        """Weighted hat-matrix diagonal ``hᵢ = wᵢ·(X·A⁻¹·X')ᵢᵢ``.

        ``Σ hᵢ = tr(W X A⁻¹ X') = tr(A⁻¹ X'WX) = edf_total`` at the PIRLS-
        converged β̂. Walks the data chunk-by-chunk so the n×p design is
        never materialised. ``w_full`` carries the PIRLS Fisher weights at β̂
        (zero on rows the ``good`` mask dropped, by construction in
        :func:`_build_qr_chunked_pirls`).

        For ``discrete=True`` we materialise the cached full-X once via
        :func:`discrete_full_X` and compute the diag in a single matmul —
        avoids the per-chunk re-gather, which the discrete kernels make
        cheap anyway since basis values are stored per unique row.
        """
        if self._discrete_design is not None:
            X = discrete_full_X(self._discrete_design)
            HX = X @ A_inv
            return w_full * (HX * X).sum(axis=1)
        n = self.n
        out = np.empty(n, dtype=float)
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = _materialize_chunk(
                self._blocks,
                self.data[start:end],
                self._X_param_full[start:end],
            )
            HX = X_chunk @ A_inv
            out[start:end] = w_full[start:end] * (HX * X_chunk).sum(axis=1)
        return out

    def _bgam_fit_loop(self, *, sp_user) -> tuple["_FitState", np.ndarray]:
        """Outer PIRLS loop with chunked QR rebuild per iter.

        Direct port of mgcv ``bgam.fit`` (bam.r:909-1353). Each iteration:

        1. Build (R, f, ‖z̃‖², rss_extra) over chunks of √W·X / √W·z, where
           (W, z) are the Fisher PIRLS weights/working response computed from
           the chunk's η = X·β + offset (or the family-initialised η on
           iter 1, when β is still ``None``).
        2. Update reduced sufficient stats ``self._XtX = R'R``, ``self._Xty
           = R'f``, ``self._yty = ‖z̃‖²``, ``self._X_full = R``.
        3. Run ``_outer_newton`` over (ρ, log φ) on the reduced (R, f) data,
           then recover β̂ at converged ρ̂ via ``_fit_given_rho``.
        4. Step-halving (mgcv "kk" inner loop): if ``it > 1`` and the new
           penalised deviance increases, halve β toward ``β₀`` and rebuild.

        Convergence (mgcv:1154): ``|dev - devold| / (0.1+|dev|) < ε`` after
        ``it > 1`` (= mgcv's ``iter > 2``; the ``dev*2`` seed in step 0 makes
        the first iter's check meaningless, and the second iter's compares
        against that synthetic seed).

        Note: at the converged β̂, the (R, f) reduced problem looks Gaussian-
        on-(R, f) — so the inner Newton sees ``W = I`` after reduction. The
        non-Gaussianness lives only in the *outer* loop's ``W`` construction.
        Mirrors how mgcv's ``fast.REML.fit`` runs on (qrx$R, qrx$f) without
        knowing the original family.
        """
        family = self.family
        link = family.link
        n = self.n
        p = self.p
        n_sp = len(self._slots)
        method = self.method   # already mapped fREML → REML

        blocks = self._blocks
        chunk_size = self._chunk_size
        y = self._y_arr
        off = self._offset
        prior_w = np.ones(n)   # user-facing weights= lands later

        include_log_phi = (not family.scale_known) and method in ("REML", "ML")

        # ---- Initialize μ̂, η̂, dev for iter 0 (mgcv bam.r:950-969) -----
        mu = family.initialize(y, prior_w)
        eta = link.link(mu)
        if not (link.valideta(eta) and family.validmu(mu)):
            raise FloatingPointError(
                "PIRLS init: cannot find valid starting μ̂ from family.initialize"
            )
        coef: Optional[np.ndarray] = None
        coef0: Optional[np.ndarray] = None
        eta0: Optional[np.ndarray] = None
        dev0: Optional[float] = None
        # mgcv:969 — dev = sum(dev_resids) * 2 to avoid spurious convergence at iter 1.
        dev = 2.0 * float(np.sum(family.dev_resids(y, mu, prior_w)))

        eps = 1e-7
        maxit = 200          # mgcv default control$maxit
        conv = False

        rho_hat: Optional[np.ndarray] = None
        log_phi_hat: Optional[float] = None
        fit: Optional[_FitState] = None

        for it in range(maxit):
            devold = dev
            kk = 0
            while True:   # mgcv "repeat" — re-enters on step halving
                if self._discrete_design is not None:
                    qr = _build_qr_discrete_pirls(
                        self._discrete_design, y, off, family,
                        coef=coef,
                        eta_init=eta if coef is None else None,
                        use_chol=self._use_chol,
                        prior_w=prior_w,
                    )
                else:
                    qr = _build_qr_chunked_pirls(
                        self.data, blocks, self._X_param_full, y, off,
                        family,
                        coef=coef,
                        eta_init=eta if coef is None else None,
                        chunk_size=chunk_size, use_chol=self._use_chol,
                        prior_w=prior_w,
                    )
                self._bam_qr = qr
                # Reduced-data sufficient stats consumed by ``_outer_newton``
                # via the inherited ``_fit_given_rho`` machinery. ``_X_full =
                # R`` keeps the inner-score routines on the (R, f) reduced
                # design just like the Gaussian-identity path. The bam-class
                # ``_dw_deta`` / ``_d2w_deta2`` overrides return ``zeros(p)``,
                # which matches "Gaussian-on-(R, f)" exactly: at the PIRLS-
                # converged β̂ the inner score sees a constant-W problem.
                self._XtX = qr.R.T @ qr.R
                self._Xty = qr.R.T @ qr.f
                self._yty = float(qr.y_norm2)
                self._X_full = qr.R
                self._wt_full = qr.wt

                new_eta = qr.eta
                new_mu = qr.mu
                new_dev = qr.dev
                if not np.isfinite(new_dev):
                    raise FloatingPointError(
                        f"non-finite deviance at PIRLS iter {it}"
                    )

                dev = new_dev

                # Convergence (mgcv:1154). it>1 == mgcv iter>2 (1-based).
                if it > 1 and abs(dev - devold) / (0.1 + abs(dev)) < eps:
                    conv = True
                    eta = new_eta
                    mu = new_mu
                    break

                if kk > 0:
                    # mgcv:1159 — already shrunk this iter's step, accept.
                    eta = new_eta
                    mu = new_mu
                    break

                # Divergence test + step halving (mgcv:1163-1190).
                if (it > 1 and coef is not None and coef0 is not None
                        and rho_hat is not None and dev0 is not None):
                    Sλ_h = self._build_S_lambda(rho_hat)
                    Sλ_h = 0.5 * (Sλ_h + Sλ_h.T)
                    Sb0 = Sλ_h @ coef0
                    Sb = Sλ_h @ coef
                    old_pdev = float(dev0) + float(coef0 @ Sb0)
                    new_pdev = float(new_dev) + float(coef @ Sb)
                    while old_pdev < new_pdev and kk < 6:
                        coef = (coef0 + coef) / 2
                        new_eta = (eta0 + new_eta) / 2
                        new_mu = link.linkinv(new_eta)
                        Sb = Sλ_h @ coef
                        new_dev = float(np.sum(
                            family.dev_resids(y, new_mu, prior_w)
                        ))
                        new_pdev = float(new_dev) + float(coef @ Sb)
                        kk += 1
                    if kk > 0:
                        eta = new_eta
                        mu = new_mu
                        dev = new_dev
                        continue   # rebuild (R, f) with halved coef

                eta = new_eta
                mu = new_mu
                break
            # end while

            if conv:
                break

            # Snapshot for next iter's divergence check (mgcv:1196-1202).
            if it > 0 and coef is not None:
                coef0 = coef.copy()
                eta0 = eta.copy()
                dev0 = dev

            # ---- sp optimisation on the current (R, f, rss_extra) -----------
            if n_sp == 0:
                rho_hat = np.zeros(0)
                log_phi_hat = None
                self.sp = np.zeros(0)
                fit = self._fit_given_rho(rho_hat)
            elif sp_user is not None:
                sp_arr = np.asarray(sp_user, dtype=float)
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
                if include_log_phi:
                    Dp = float(fit.dev + fit.pen)
                    denom = (max(float(n - self._Mp), 1.0)
                             if method == "REML" else max(float(n), 1.0))
                    log_phi_hat = float(np.log(max(Dp / denom, 1e-300)))
            else:
                # mgcv POI ("perf" optimizer, bgam.fitd at bam.r:430-780):
                # take a *single* Newton step on (ρ, log φ) per PIRLS
                # iter, with step-halving when the prior step was uphill.
                # The Newton step / β / REML grad+Hess come from
                # ``_pi_fit_chol`` (mgcv's ``Sl.fitChol``) which uses
                # diagonal preconditioning + pivoted Chol + IFT
                # derivatives. For ``discrete=TRUE`` non-Gaussian fits
                # this matches mgcv's ``c("perf", "chol")`` optimizer
                # cadence — joint sp-and-coef updates, not nested
                # outer-Newton-on-frozen-(R,f).
                #
                # ``method == "GCV.Cp"`` falls back to ``_outer_newton``
                # because POI's REML formulas don't apply to GCV/UBRE.
                if (method in ("REML", "ML")
                        and self._discrete_design is not None):
                    # Lazily build ``Sl.initial.repara`` data on first
                    # PIRLS iter — depends only on the slot S matrices,
                    # not on rho/W.
                    if not hasattr(self, "_repara_blocks"):
                        self._repara_blocks = _build_init_repara(
                            self._slots, self.p,
                        )
                        self._repara_slots = _build_repara_slots(
                            self._slots, self._repara_blocks,
                        )
                    if rho_hat is None:
                        rho_cur = self._initial_sp_rho()
                        Nstep = np.zeros(n_sp + (1 if include_log_phi else 0))
                    else:
                        rho_cur = rho_hat.copy()
                    if include_log_phi:
                        if log_phi_hat is None:
                            try:
                                fit_seed = self._fit_given_rho(rho_cur)
                                df_resid_seed = max(n - self._Mp, 1.0)
                                log_phi_cur = float(np.log(
                                    max(fit_seed.dev / df_resid_seed, 1e-12)
                                ))
                            except Exception:
                                log_phi_cur = 0.0
                        else:
                            log_phi_cur = log_phi_hat
                        theta_cur = np.concatenate(
                            [rho_cur, [log_phi_cur]],
                        )
                    else:
                        theta_cur = rho_cur

                    # Newton step + halving (mgcv bam.r:669-682).
                    halve_max = 30
                    halves = 0
                    while True:
                        theta_try = theta_cur + Nstep
                        if include_log_phi:
                            rho_try = theta_try[:n_sp]
                            log_phi_try = float(theta_try[n_sp])
                        else:
                            rho_try = theta_try
                            log_phi_try = 0.0
                        S_full_try = self._build_S_lambda(rho_try)
                        S_full_try = 0.5 * (S_full_try + S_full_try.T)
                        S_pinv_try = self._S_pinv(S_full_try)
                        ldS_grad = self._dlog_det_S_drho(
                            rho_try, S_pinv=S_pinv_try, S_full=S_full_try,
                        )
                        ldS_hess = self._d2log_det_S_drho_drho(
                            rho_try, S_pinv=S_pinv_try, S_full=S_full_try,
                        )
                        # mgcv ``Sl.initial.repara`` (fast-REML.r:490) —
                        # rotate XX, Xy into the multi-S blocks' eigen
                        # basis so the pivoted Cholesky in
                        # ``_pi_fit_chol`` runs in mgcv's gauge. β
                        # comes back in the repara'd basis and gets
                        # un-rotated below. Without this step the pivot
                        # tie-breaking on rank-deficient ``H`` differs
                        # from mgcv's, leaving a residual coef-gauge gap.
                        XX_pre, Xy_pre = _apply_init_repara(
                            self._XtX, self._Xty, self._repara_blocks,
                        )
                        out = _pi_fit_chol(
                            XX_pre, Xy_pre, rho_try,
                            self._repara_slots, self.p,
                            yy=self._yty, log_phi=log_phi_try, n=n,
                            Mp=self._Mp, gamma=self._gamma,
                            phi_fixed=not include_log_phi,
                            ldet_S_grad=ldS_grad, ldet_S_hess=ldS_hess,
                        )
                        # Undo the initial-repara on β — the rest of
                        # the PIRLS / post-fit machinery (chunked X·β,
                        # variance, edf) operates in the original basis.
                        out["beta"] = _undo_init_repara_beta(
                            out["beta"], self._repara_blocks,
                        )
                        if float(np.max(np.abs(Nstep))) == 0.0:
                            # First call or zero step — accept and
                            # snapshot the new step for next iter.
                            Nstep = out["step"]
                            theta_cur = theta_try
                            break
                        # mgcv: ``sum(prop$grad * Nstep) > dev * 1e-7`` =
                        # uphill. Halve and retry.
                        if (float(np.dot(out["grad"], Nstep))
                                > abs(dev) * 1e-7
                                and halves < halve_max):
                            Nstep = Nstep / 2.0
                            halves += 1
                            continue
                        Nstep = out["step"]
                        theta_cur = theta_try
                        break

                    if include_log_phi:
                        rho_hat = theta_cur[:n_sp]
                        log_phi_hat = float(theta_cur[n_sp])
                    else:
                        rho_hat = theta_cur
                        log_phi_hat = None
                    self.sp = np.exp(rho_hat)
                    fit = self._fit_given_rho(rho_hat)
                else:
                    # Non-discrete or GCV path: fall back to the
                    # converge-fully outer-Newton.
                    if rho_hat is None:
                        rho0 = self._initial_sp_rho()
                    else:
                        rho0 = rho_hat.copy()
                    if include_log_phi:
                        if log_phi_hat is None:
                            try:
                                fit_seed = self._fit_given_rho(rho0)
                                df_resid_seed = max(n - self._Mp, 1.0)
                                log_phi0 = float(np.log(
                                    max(fit_seed.dev / df_resid_seed, 1e-12)
                                ))
                            except Exception:
                                log_phi0 = 0.0
                        else:
                            log_phi0 = log_phi_hat
                        theta0 = np.concatenate([rho0, [log_phi0]])
                    else:
                        theta0 = rho0

                    theta_hat = self._outer_newton(
                        theta0,
                        criterion=(
                            "REML" if method in ("REML", "ML") else "GCV"
                        ),
                        include_log_phi=include_log_phi,
                        include_family_theta=False,
                    )
                    if include_log_phi:
                        rho_hat = theta_hat[:n_sp]
                        log_phi_hat = float(theta_hat[n_sp])
                    else:
                        rho_hat = theta_hat
                        log_phi_hat = None
                    self.sp = np.exp(rho_hat)
                    fit = self._fit_given_rho(rho_hat)

            self._log_phi_hat = log_phi_hat

            new_coef = fit.beta
            if not np.all(np.isfinite(new_coef)):
                warnings.warn(
                    f"non-finite coefficients at PIRLS iteration {it+1}",
                    stacklevel=2,
                )
                break
            coef = new_coef.copy()
        # end outer iter loop

        if not conv:
            warnings.warn("PIRLS algorithm did not converge", stacklevel=2)

        if fit is None:
            raise FloatingPointError("bgam.fit produced no usable fit")

        self._rho_hat = rho_hat if rho_hat is not None else np.zeros(0)
        self._log_phi_hat = log_phi_hat
        self._iter = it + 1
        return fit, self._rho_hat

    # -----------------------------------------------------------------------
    # Post-fit assembly (non-Gaussian PIRLS path)
    # -----------------------------------------------------------------------

    def _post_fit_pirls(self, fit, rho_hat: np.ndarray,
                        X_param_df: pl.DataFrame) -> None:
        """Populate user-facing attributes after PIRLS converges.

        Mirrors gam.__init__'s post-fit (gam.py:476-783) on the (R, f) reduced
        problem. The PIRLS chunked build returns full-length (η, μ, w, z)
        at the converged β̂; ``self._wt_full`` holds the Fisher weights at β̂
        and ``self._XtX = R'R = X'WX`` is the Gram of √W·X. So
        ``Vp = σ²·A⁻¹`` and ``Ve = σ²·A⁻¹·X'WX·A⁻¹`` work directly with
        ``A⁻¹ = (X'WX + Sλ)⁻¹`` from ``fit.A_chol``.
        """
        n, p = self.n, self.p
        method = self.method
        n_sp = len(self._slots)
        family = self.family
        y = self._y_arr
        beta = fit.beta
        Sλ = fit.S_full
        self._rho_hat = rho_hat

        A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        XtWX = self._XtX                # = R'R = X'WX at converged β̂
        A_inv_XtWX = A_inv @ XtWX
        edf = np.diag(A_inv_XtWX).copy()
        edf_total = float(edf.sum())

        # Prior weights (=1 for now). Same convention as gam.
        self._wt = np.ones(n)
        wt = self._wt
        df_resid = float(n - edf_total)

        # Pearson scale = Σ wᵢ·(yᵢ - μᵢ)²/V(μᵢ) / df_resid (mgcv gam.fit3.r:606).
        if df_resid > 0 and not family.scale_known:
            V = family.variance(fit.mu)
            pearson_scale = float(
                np.sum(wt * (y - fit.mu) ** 2 / V)
            ) / df_resid
        else:
            pearson_scale = 1.0 if family.scale_known else float("nan")
        self._pearson_scale = pearson_scale
        scale = 1.0 if family.scale_known else pearson_scale
        sigma_squared = scale
        sigma = (float(np.sqrt(sigma_squared))
                 if np.isfinite(sigma_squared) and sigma_squared >= 0
                 else float("nan"))

        Vp = sigma_squared * A_inv
        Ve = sigma_squared * A_inv_XtWX @ A_inv

        # Coefficient basis change for t2 smooths (rare).
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

        # ---- linear predictors / fitted / residuals -------------------------
        eta = fit.eta
        mu = fit.mu
        self.linear_predictors = eta
        self.fitted_values = mu
        self.fitted = mu
        # Deviance residuals: sign(y-μ)·√d_i (default residual type, mgcv).
        di = family.dev_resids(y, mu, wt)
        self.residuals = np.sign(y - mu) * np.sqrt(np.maximum(di, 0.0))
        self.sigma = sigma
        self.sigma_squared = sigma_squared
        self.scale = sigma_squared

        # Leverage h_i = w_i·(X A⁻¹ X')_ii via chunked walk.
        leverage = self._chunked_leverage_diag_weighted(A_inv, self._wt_full)
        self.leverage = leverage
        sigma_for_std = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
        denom = sigma_for_std * np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        V_mu = family.variance(mu)
        pearson_res = (y - mu) * np.sqrt(self._wt / np.maximum(V_mu, 0.0))
        self.std_dev_residuals = self.residuals / denom
        self.std_pearson_residuals = pearson_res / denom
        self.df_residuals = df_resid
        self.deviance = float(np.sum(di))
        self.rss = self.deviance     # Gaussian-era alias

        # Null deviance: intercept-only μ̂ = weighted mean of y; without
        # intercept, η ≡ 0 ⇒ μ ≡ linkinv(0).
        if self._has_intercept:
            mu_null_const = float(np.sum(wt * y) / np.sum(wt))
            mu_null = np.full(n, mu_null_const)
        else:
            mu_null = family.link.linkinv(np.zeros(n))
        self.null_deviance = float(
            np.sum(family.dev_resids(y, mu_null, wt))
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

        # R² / R²_adj / dev_explained.
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
                r_squared_adjusted = (
                    1.0 - v_resid * (n - 1) / (v_total * df_resid)
                )
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

        # The (R, f) reduced problem is Gaussian-on-(R, f), so
        # ``_compute_edf12`` and ``_reml_hessian`` see W=I just like the
        # Gaussian-identity path. ``self._fisher_w = None`` keeps the inherited
        # XtWX-rebuild short-circuit on (line 3228 in gam.py).
        self._fisher_w = None

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
        sc_p = 0.0 if family.scale_known else 1.0
        dev1 = family._aic_dev1(self.deviance, sigma_squared, wt)
        family_aic = float(family.aic(y, fit.mu, dev1, wt, n))
        mgcv_aic = family_aic + 2.0 * edf_total
        logLik = sc_p + edf_total - 0.5 * mgcv_aic
        df_for_aic = min(self.edf2_total + sc_p, float(p) + sc_p)
        self.loglike = float(logLik)
        self.logLik = self.loglike
        self.npar = float(df_for_aic)
        self.AIC = -2.0 * logLik + 2.0 * df_for_aic
        self.BIC = -2.0 * logLik + float(np.log(n)) * df_for_aic
        self._mgcv_aic = float(mgcv_aic)

        if method in ("REML", "ML"):
            if n_sp > 0:
                log_phi_hat = (
                    self._log_phi_hat
                    if self._log_phi_hat is not None else 0.0
                )
                score = float(self._reml(rho_hat, log_phi_hat, fit=fit))
            else:
                score = float("nan")
            if method == "REML":
                self.REML_criterion = score
            else:
                self.ML_criterion = score
        else:
            if n_sp > 0:
                self.GCV_score = float(self._gcv(rho_hat))
            else:
                self.GCV_score = float("nan")

        self.vcomp = self._compute_vcomp()
        self.AR1_rho = self._rho   # always 0 for the non-Gaussian path
