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
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve, qr as scipy_qr, solve_triangular
from scipy.linalg.lapack import dpstrf
from scipy.stats import t as t_dist

from ..family import Family, Gaussian
from ..formula import (
    BasisSpec,
    SmoothBlock,
    _apply_smooth_arg_exprs,
    _eval_atom,
    _eval_by_col,
    _LinearTransformRawBasis,
    _RawBasis,
    _smooth_arg_expr_map,
    _T2PredictRawBasis,
    _T2RawBasis,
    _TensorRawBasis,
    materialize,
    materialize_smooths,
)
from ..design import is_matrix_col, matrix_to_2d, prepare_design
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
    if trans:
        # Transposed form — applied to lpmatrix on the right; not exercised
        # by bam.fit's single-thread path. Kept for completeness.
        out = np.zeros((n, p), dtype=float)
        prev = -1
        for i in range(n):
            for k in range(prev + 1, stop[i] + 1):
                out[row[k], :] += weight[k] * X[i, :]
            prev = stop[i]
        return out.ravel() if not is_matrix else out
    if n == 0:
        out = np.zeros((0, p), dtype=float)
        return out.ravel() if not is_matrix else out
    # Vectorized segmented reduction: output row i sums
    # ``weight[k] · X[row[k], :]`` over k in ``(stop[i-1]+1):stop[i]``
    # (with ``stop[-1] := -1`` for the first row). ``np.add.reduceat``
    # sums each segment left-to-right, matching the scalar-loop
    # arithmetic bit-exactly.
    K = int(stop[-1]) + 1
    weighted = weight[:K, None] * X[row[:K], :]
    starts = np.empty(n, dtype=np.intp)
    starts[0] = 0
    starts[1:] = stop[:-1] + 1
    out = np.add.reduceat(weighted, starts, axis=0)
    # ``reduceat`` returns ``weighted[starts[i]]`` (not zero) for any
    # empty segment ``starts[i] == starts[i+1]``. The scalar loop leaves
    # that output row at zero, so re-zero those positions when present.
    nonempty = np.empty(n, dtype=bool)
    nonempty[0] = stop[0] >= 0
    nonempty[1:] = stop[1:] >= starts[1:]
    if not nonempty.all():
        out[~nonempty] = 0.0
    return out.ravel() if not is_matrix else out


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


def _estimate_theta(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    *,
    scale: float = 1.0,
    wt: Optional[np.ndarray] = None,
    tol: float = 1e-7,
) -> np.ndarray:
    """Inner Newton on the family's extra parameters θ at fixed (y, μ).

    Direct port of mgcv ``estimate.theta`` (R/efam.r:5-96). Used inside
    the bgam.fitd PIRLS loop after iter 1: each PIRLS step updates β at
    fixed θ, then this routine updates θ at fixed β; the two alternate
    until both converge.

    Negative log-likelihood objective per mgcv:

        nll(θ) = dev(y, μ, w, θ) / (2·scale) − ls(y, w, θ, scale)

    where ``dev`` is the family's deviance and ``ls`` is the saturated
    log-likelihood (mgcv ``family$ls`` returning ``{ls, lsth1, lsth2}``).
    Gradient and Hessian come from ``family.Dd(level=2)`` (μ-side
    derivatives summed over observations) plus the ``ls`` derivatives.

    For ``scale < 0`` (scale-unknown extended families), an extra
    log-φ slot is appended to θ and updated jointly. Scat is
    ``scale_known = True`` (always called with ``scale = 1``) so the
    scale<0 branches are dead for the user's model — we keep them so
    future families like ``betar`` plug in unchanged.

    Newton specifics:

    * Eigen-decompose H; if any eigenvalue ≤ 0 use ``|λ_i|`` floored at
      ``max(λ)·1e-5`` (mgcv R/efam.r:60-64) to make H positive-def.
    * Step ``= -H⁻¹·g`` capped to ``|step|_∞ ≤ 4`` (R/efam.r:69-70).
    * Step halving (≤ 25 iters) while uphill (R/efam.r:75-82).
    * Outer iters capped at 100 (R/efam.r:57).
    * Componentwise convergence ``|g_i| ≤ tol·(|nll| + 1)`` — only
      update components flagged by the ``uconv`` mask.

    Returns the converged θ (same shape as the family's internal θ when
    ``scale ≥ 0``; appended with ``log φ̂`` when ``scale < 0``).
    """
    if not family.is_extended:
        raise ValueError(
            f"_estimate_theta called with non-extended family "
            f"{type(family).__name__}"
        )
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if wt is None:
        wt = np.ones_like(y)
    else:
        wt = np.asarray(wt, dtype=float)
    n_theta = int(family.n_theta)
    if scale >= 0 and n_theta == 0:
        raise ValueError(
            "estimate_theta called with no free parameters: family is "
            "scale-known and n_theta=0"
        )
    theta = family.get_theta().copy()
    # mgcv: when scale<0 (scale-unknown extended family), append a
    # starting log φ slot to θ — using either ``log(var(y)*0.1)`` if
    # μ ≈ y (all data already explained ⇒ score scale init) or
    # ``log(mean((y-μ)²/V(μ)))`` otherwise.
    if scale < 0:
        if np.allclose(y, mu):
            log_phi0 = float(np.log(np.var(y) * 0.1))
        else:
            V = family.variance(mu)
            log_phi0 = float(np.log(np.mean((y - mu) ** 2 / V)))
        theta = np.concatenate([theta, [log_phi0]])
    n_total = theta.shape[0]
    # ``del.ind`` deletes the first n_theta slots when scale<0 AND
    # n_theta=0 (the original n_theta-slot was a no-op then). When
    # n_theta>0 we keep all slots and del.ind is unused. We follow
    # mgcv's index discipline so future ``betar``-style families
    # (n_theta=0, scale<0) get the right gradient/Hessian sub-block.
    del_ind = slice(0, n_theta)

    def _nlogl(theta_eval: np.ndarray, deriv: int):
        # mgcv R/efam.r:14-45 verbatim. ``theta_eval`` may include a
        # trailing log φ slot when scale<0; strip it for the family
        # calls and re-add the φ-direction gradient / Hessian rows.
        nth = n_theta
        if scale < 0:
            scale_eval = float(np.exp(theta_eval[nth]))
            theta_for_family = theta_eval[:nth]
            get_scale = True
        else:
            scale_eval = float(scale)
            theta_for_family = theta_eval
            get_scale = False
        dev = float(np.sum(
            family.dev_resids(y, mu, wt, theta=theta_for_family)
        )) / scale_eval
        if deriv > 0:
            Dd = family.Dd(y, mu, theta_for_family, wt, level=deriv)
        ls = family.ls_extended(y, wt, theta=theta_for_family,
                                scale=scale_eval)
        nll = dev / 2.0 - float(ls["ls"])

        if deriv > 0:
            Dth = np.atleast_2d(Dd["Dth"])
            g1 = Dth.sum(axis=0) / (2.0 * scale_eval)
            if get_scale:
                g = np.concatenate([g1, [-dev / 2.0]])
            else:
                g = g1.copy()
            ind = slice(0, g.shape[0])
            g = g - np.atleast_1d(ls["lsth1"])[ind]
        else:
            g = None

        if deriv > 1:
            Dth2_packed = np.atleast_2d(Dd["Dth2"])
            xs = Dth2_packed.sum(axis=0) / (2.0 * scale_eval)
            Dth2 = np.zeros((nth, nth), dtype=float)
            iu, ju = np.triu_indices(nth)
            Dth2[iu, ju] = xs[:iu.size]
            Dth2[ju, iu] = xs[:iu.size]
            if get_scale:
                # mgcv R/efam.r:41: rbind(cbind(Dth2,-g1), c(-g1,dev/2))
                top = np.column_stack([Dth2, -g1.reshape(-1, 1)])
                bot = np.append(-g1, dev / 2.0).reshape(1, -1)
                Dth2 = np.vstack([top, bot])
            ls_h2 = np.atleast_2d(ls["lsth2"])
            H = Dth2 - ls_h2[ind, ind]
        else:
            H = None
        return nll, g, H

    # Initial probe
    nll, g, H = _nlogl(theta, 2)
    if n_theta == 0:
        # Drop the first n_theta=0 slots — no-op slice; this keeps the
        # mgcv index discipline so betar-style families work.
        g = g[n_theta:]
        H = H[n_theta:, n_theta:]
    eps_thresh = float(np.finfo(float).eps ** 0.75)
    step_failed = False
    uconv = np.abs(g) > tol * (abs(nll) + 1.0)

    if np.any(uconv):
        for _ in range(100):
            H_act = H[np.ix_(uconv, uconv)]
            evals, evecs = np.linalg.eigh(0.5 * (H_act + H_act.T))
            pdef = bool(np.all(evals > 0.0))
            if not pdef:
                # mgcv R/efam.r:60-64: |λ| floored at max(|λ|)*1e-5
                evals = np.abs(evals)
                thresh = float(evals.max()) * 1e-5 if evals.size > 0 else 0.0
                evals = np.where(evals < thresh, thresh, evals)
            # Newton step via eigen: step = −V·diag(1/λ)·Vᵀ·g
            step0 = -evecs @ ((evecs.T @ g[uconv]) / evals)
            if n_theta == 0:
                step0 = np.concatenate([np.zeros(n_theta), step0])
            ms = float(np.max(np.abs(step0)))
            if ms > 4.0:
                step0 = step0 * 4.0 / ms
            step = np.zeros_like(theta)
            step[uconv] = step0

            # mgcv R/efam.r:73: deriv-2 probe at the proposed θ+step.
            # Reused as the next iteration's (g, H) when no halving fires.
            nll1, g1, H1 = _nlogl(theta + step, 2)
            it_halv = 0
            while nll1 - nll > eps_thresh * abs(nll):
                step = step / 2.0
                it_halv += 1
                if np.all(theta == theta + step) or it_halv > 25:
                    step_failed = True
                    break
                # mgcv R/efam.r:81: deriv=0 probe inside halving — only
                # the nll value matters for the uphill check.
                nll1, _, _ = _nlogl(theta + step, 0)
            if step_failed:
                break
            theta = theta + step
            # mgcv R/efam.r:86: if iter>0 (halving fired) re-probe at
            # the halved θ for fresh (g, H); otherwise reuse the
            # deriv-2 evaluation at the un-halved θ.
            if it_halv > 0:
                nll, g, H = _nlogl(theta, 2)
            else:
                nll, g, H = nll1, g1, H1
            if n_theta == 0:
                g = g[n_theta:]
                H = H[n_theta:, n_theta:]
            uconv = np.abs(g) > tol * (abs(nll) + 1.0)
            if not np.any(uconv):
                break

    if step_failed:
        import warnings
        warnings.warn("step failure in theta estimation", stacklevel=2)
    return theta


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
    from ..formula import (
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
    if n == 0:
        return np.empty(0, dtype=float)
    ld = 1.0 / np.sqrt(1.0 - rho ** 2)
    out = ld * rsd
    out[1:] -= rho * ld * rsd[:-1]
    out[0] = rsd[0]
    if ar_start is not None:
        np.copyto(out, rsd, where=np.asarray(ar_start, dtype=bool))
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

        if family.is_extended:
            # Extended-family Newton branch (mgcv bgam.fitd, bam.r:577-591).
            # Per-chunk: w = Deta2/2, z = (η-off) - Deta.Deta2; ``good`` is
            # finiteness of (w, z). See ``_build_qr_discrete_pirls`` for
            # the full derivation.
            theta = family.get_theta()
            deta = family.dDeta(y_chunk, mu_chunk, wp_chunk, theta, level=0)
            Deta2 = deta["Deta2"]
            w_chunk = Deta2 * 0.5
            z_chunk = (eta_chunk - off_chunk) - deta["Deta.Deta2"]
            good = np.isfinite(z_chunk) & np.isfinite(w_chunk)
            w_chunk = np.where(good, w_chunk, 0.0)
            z_chunk = np.where(good, z_chunk, 0.0)
            dev_total += float(np.sum(
                family.dev_resids(y_chunk, mu_chunk, wp_chunk, theta=theta)
            ))
        else:
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

            dev_total += float(np.sum(
                family.dev_resids(y_chunk, mu_chunk, wp_chunk)
            ))

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

    if family.is_extended:
        # Extended-family Newton branch (mgcv bgam.fitd, bam.r:577-591).
        # ``dDeta`` returns η-space derivatives of ``-logL`` at fixed θ;
        # the IRLS-equivalent Newton step is
        #
        #     w = Deta2 * 0.5     (observed Hessian, rho==0 branch)
        #     z = (η − offset) − Deta.Deta2
        #
        # ``Deta.Deta2 = Dmu / (Dmu2·μ_η − Dmu·g2g)`` for non-identity
        # link, ``Dmu/Dmu2`` for identity — already computed inside
        # ``dDeta``. The good-row mask is just finiteness of (w, z);
        # extended families have no μ_η==0 boundary the way the
        # standard Fisher branch does. mgcv's ``rho != 0`` AR1 branch
        # (using ``EDeta2`` / ``Deta.EDeta2``) is not yet wired —
        # ``rho`` is unsupported on the discrete path today.
        theta = family.get_theta()
        deta = family.dDeta(y, mu_full, prior_w, theta, level=0)
        Deta2 = deta["Deta2"]
        w_full = Deta2 * 0.5
        z_full = (eta_full - offset) - deta["Deta.Deta2"]
        good = np.isfinite(z_full) & np.isfinite(w_full)
        w_full = np.where(good, w_full, 0.0)
        z_full = np.where(good, z_full, 0.0)
        if not np.any(good):
            raise FloatingPointError(
                "discrete PIRLS build (extended family) saw zero good "
                "rows — every observation has non-finite Deta2 or "
                "Deta.Deta2"
            )
        dev_total = float(np.sum(
            family.dev_resids(y, mu_full, prior_w, theta=theta)
        ))
    else:
        # Standard exponential-family Fisher branch.
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
                "discrete PIRLS build saw zero good rows — every "
                "observation was dropped by the (w_prior > 0 & μ_η != 0) "
                "good mask"
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
    # predict — override of gam.predict (newdata=None case)
    # -----------------------------------------------------------------------

    def predict(
        self,
        newdata: pl.DataFrame | None = None,
        type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | list | None = None,
    ):
        """Predict from the fitted bam — :func:`predict.bam` parity.

        Override of :meth:`hea.gam.predict` because ``self._X_full =
        qr.R`` (p × p) on every bam path: Gaussian-identity stores R for
        the chunked-QR closed-form solve (bam.py:1961), PIRLS / discrete
        store R for the inner-Newton's reduced-data scoring
        (bam.py:2791). The inherited routine assumes ``_X_full`` is the
        full ``n × p`` design and shape-clashes against ``_offset``
        (length n) when ``newdata=None``.

        Behaviour for each (newdata, type, se_fit) combination:

        * ``newdata`` not None → delegate to ``super().predict(...)`` —
          parent rebuilds the design via per-block ``spec.predict_mat``
          the same way it does for gam fits.
        * ``newdata=None``, ``type='link' | 'response'``, no
          ``se_fit``, no extra ``offset`` → cached
          :attr:`linear_predictors` / :attr:`fitted_values`.
        * ``newdata=None`` with ``type='lpmatrix'`` → route through
          ``super().predict(newdata=self.data, type='lpmatrix')``,
          which re-evaluates each smooth's basis on training rows.
          For non-discrete bam this is bit-equal to the design used
          during the fit; for discrete bam it evaluates basis exactly
          (vs. mgcv's discretize-then-gather via ``Xbd``) — *Phase 5.2
          will replace this with the discrete-aware path*.
        * ``newdata=None`` with ``se_fit=True`` → cached eta (+ extra
          offset, if any) for the link-scale prediction; chunked
          ``diag(X·Vp·X')`` (via :meth:`_chunked_var_eta_diag`) for
          per-row link-scale variance; delta-method
          ``|μ_η|`` multiplier for response-scale SE.
        """
        if type not in ("link", "response", "lpmatrix"):
            raise ValueError(
                f"type must be 'link', 'response', or 'lpmatrix'; got {type!r}"
            )
        if type == "lpmatrix" and se_fit:
            raise ValueError(
                "se_fit=True is not allowed with type='lpmatrix'"
            )

        if newdata is not None:
            return super().predict(
                newdata=newdata, type=type, se_fit=se_fit, offset=offset,
            )

        # ---- newdata=None branch -------------------------------------------
        extra: Optional[np.ndarray] = None
        if offset is not None:
            extra = np.asarray(offset, dtype=float).flatten()
            if extra.shape != (self.n,):
                raise ValueError(
                    f"offset must have length {self.n}, got {extra.shape}"
                )

        # Fast path: cached arrays cover the most common ask.
        if type in ("link", "response") and not se_fit and extra is None:
            if type == "link":
                return pl.DataFrame({"fit": self.linear_predictors.copy()})
            return pl.DataFrame({"fit": self.fitted_values.copy()})

        if type == "lpmatrix":
            # lpmatrix returns a raw ndarray (design matrix, not prediction).
            return super().predict(
                newdata=self.data, type="lpmatrix", se_fit=False, offset=None,
            )

        # link / response with se_fit=True or with an extra offset.
        eta = self.linear_predictors.copy()
        if extra is not None:
            eta = eta + extra

        if not se_fit:
            if type == "link":
                return pl.DataFrame({"fit": eta})
            return pl.DataFrame({"fit": self.family.link.linkinv(eta)})

        # se_fit=True
        var_eta = self._chunked_var_eta_diag(self.Vp)
        se_link = np.sqrt(np.maximum(var_eta, 0.0))
        if type == "link":
            return pl.DataFrame({"fit": eta, "se.fit": se_link})
        mu = self.family.link.linkinv(eta)
        mu_eta = self.family.link.mu_eta(eta)
        return pl.DataFrame({"fit": mu, "se.fit": se_link * np.abs(mu_eta)})

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

    def _chunked_var_eta_diag(self, Vp: np.ndarray) -> np.ndarray:
        """``diag(X·Vp·X')`` over the full data, chunk by chunk.

        Per-row link-scale variance ``Var(η_i) = X_i·Vp·X_iᵀ``. Same chunk
        walk as :meth:`_chunked_leverage_diag`; passing ``Vp`` instead of
        ``A_inv`` returns the predict-time link-scale variance used by
        :meth:`predict` ``se_fit=True``. Discrete bam dispatches on
        :attr:`_discrete_design`: each chunk's design rows are gathered
        from ``Xbd``-style per-marginal-Xd lookups, identical to what the
        fit would use; non-discrete bam re-evaluates basis on the
        ``self.data`` chunk via :func:`_materialize_chunk`.
        """
        n = self.n
        out = np.empty(n, dtype=float)
        if self._discrete_design is not None:
            # Discrete: row gather via predict_mat on training data is
            # bit-equal to the design used at fit time when the
            # discretization didn't round (the common case for small or
            # already-unique covariates). Phase 5.2 will replace this
            # with a true Xbd-gather.
            from ..formula import materialize
            X_param_full = self._X_param_full
            for start, end in _chunk_indices(n, self._chunk_size):
                cols = [X_param_full[start:end]]
                for b in self._blocks:
                    if b.spec is None:
                        raise RuntimeError(
                            f"smooth block {b.label!r} (cls={b.cls!r}) "
                            f"has no BasisSpec; predict requires every "
                            f"smooth to carry one."
                        )
                    cols.append(np.asarray(
                        b.spec.predict_mat(self.data[start:end]),
                        dtype=float,
                    ))
                X_chunk = np.concatenate(cols, axis=1)
                HX = X_chunk @ Vp
                out[start:end] = (HX * X_chunk).sum(axis=1)
            return out
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = _materialize_chunk(
                self._blocks,
                self.data[start:end],
                self._X_param_full[start:end],
            )
            HX = X_chunk @ Vp
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
        from ..R import NamedVector
        self.bhat = _row_frame(beta, self.column_names)
        self.coef = NamedVector(list(self.column_names), np.asarray(beta).reshape(-1))
        self.coefficients = self.coef
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

        # ---- Extended-family preinit (mgcv bgam.fitd, bam.r:534-541) ----
        # ``family.preinitialize(y)`` may return ``{"Theta": ...}`` to
        # override the family's internal θ from data (Scat: c(1.5,
        # log(0.8·sd(y)))). Standard families return None. Fires once,
        # before the first PIRLS iter.
        if family.is_extended:
            pini = family.preinitialize(y)
            if pini is not None and "Theta" in pini:
                family.set_theta(pini["Theta"])

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

            # ---- Recompute (η, μ) at the current β before θ-update ------
            # mgcv bgam.fitd:497-500: at iter > 1 set
            # ``eta <- Xbd(coef) + offset; mu <- linkinv(eta)``. This makes
            # the subsequent ``estimate.theta`` see the *post-β* μ, not
            # the stale initialise'd μ from the previous (R,f) build (which
            # was ≈ y on iter 0). We do the same for iter ≥ 1 here.
            if it >= 1 and coef is not None:
                if self._discrete_design is not None:
                    eta = (Xbd(self._discrete_design,
                                np.asarray(coef, dtype=float))
                            + off)
                else:
                    eta = self._chunked_xbeta(
                        np.asarray(coef, dtype=float)) + off
                mu = link.linkinv(eta)

            # ---- Extended-family θ update (mgcv bgam.fitd:557-571) ------
            # mgcv's ``iter > 1`` (1-based) ⇒ our ``it >= 1`` (0-based).
            # Update θ at the current (μ, β) snapshot via inner Newton on
            # ``-logL = dev/2 - ls``. Only fires for families whose θ is
            # actually free (``estimate_theta_callback = True``) — Scat
            # with both θ user-locked has ``n_theta = 0`` and stays put.
            if (it >= 1
                    and family.is_extended
                    and family.estimate_theta_callback):
                theta_new = _estimate_theta(
                    family, y, mu, scale=1.0,
                    wt=prior_w, tol=1e-7,
                )
                family.set_theta(theta_new)
                # mgcv recomputes the deviance under the new θ before the
                # convergence check (the alternating cadence: β-step at
                # old θ → θ-step at new μ → β-step at new θ). Without
                # this, ``devold`` and the iter-0-style seed compare
                # against stale θ.
                dev = 2.0 * float(np.sum(
                    family.dev_resids(y, mu, prior_w, theta=theta_new)
                ))
                # mgcv bgam.fitd:567-569: re-evaluate the previous iter's
                # ``dev0`` at the saved μ₀ but under the NEW θ. Without
                # this the step-halving check at iter ≥ 2 compares
                # ``dev0`` (under old θ) against ``new_dev`` (under new
                # θ) — apples to oranges, and divergent β iterates slip
                # through unhalved. eta0 is saved at the end of every
                # iter > 0; mu0 = linkinv(eta0) is the corresponding μ.
                if eta0 is not None:
                    mu0_at_eta0 = link.linkinv(eta0)
                    dev0 = float(np.sum(family.dev_resids(
                        y, mu0_at_eta0, prior_w, theta=theta_new
                    )))

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
        from ..R import NamedVector
        self.bhat = _row_frame(beta, self.column_names)
        self.coef = NamedVector(list(self.column_names), np.asarray(beta).reshape(-1))
        self.coefficients = self.coef
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


# ===========================================================================
# Inlined ``hea.discrete`` — discrete-covariate machinery for ``bam(discrete=TRUE)``.
#
# Lives here per the single-consumer rule (only ``bam`` uses any of this).
# Public symbols (``RMersenneTwister``, ``compress_df``, ``discrete_mf``,
# ``DiscretizedFrame``, ``DiscreteDesign``, ``build_discrete_design``,
# ``discrete_full_X``, ``Xbd``, ``XWXd``, ``XWyd``) are reachable as
# ``from hea.models.bam import <name>`` per the inline convention.
# ===========================================================================


# ---------------------------------------------------------------------------
# R's Mersenne-Twister RNG and ``sample()`` — bit-exact port.
#
# Reproduces R 3.6+ default ``RNGkind("Mersenne-Twister", "Inversion",
# "Rejection")`` so mgcv's ``temp.seed(8547)`` + ``sample()`` calls inside
# ``compress.df`` / ``discrete.mf`` are matched bit-exactly from Python.
#
# Direct port of R's ``src/main/RNG.c`` and ``src/main/random.c``:
#   * ``RNG_Init`` — 50× LCG warm-up then 625 LCG iterations to fill
#     ``i_seed[0..624]``. ``FixupSeeds(initial=1)`` overwrites
#     ``i_seed[0]`` (= mti) with N=624 so the first ``MT_genrand``
#     regenerates the state.
#   * ``MT_genrand`` — standard MT19937 step + R's tempering masks,
#     scaled via ``y * 2.3283064365386963e-10``.
#   * ``unif_rand`` — wraps MT output through R's ``fixup`` (avoid 0.0 / 1.0).
#   * ``R_unif_index`` (Sample_kind = REJECTION) — bits = ceil(log2(dn));
#     draw ``rbits(bits)`` until < dn.
#   * ``rbits`` — pack low ``bits`` bits from successive
#     ``floor(unif_rand() * 65536)`` chunks.
#   * ``sample(n, k, replace=TRUE)`` — independent ``R_unif_index(n)`` per draw.
#   * ``sample(n, k, replace=FALSE)`` — partial Fisher-Yates with swap-and-pop.
#
# All return 0-based indices (R returns 1-based; caller adjusts).
# ---------------------------------------------------------------------------

# Period parameters (RNG.c:646-650).
_MT_N = 624
_MT_M = 397
_MT_MATRIX_A = 0x9908B0DF
_MT_UPPER_MASK = 0x80000000
_MT_LOWER_MASK = 0x7FFFFFFF

# MT_genrand scale factor (RNG.c:722) — IEEE-754 nearest-double to 2^-32.
_INV_2P32 = 2.3283064365386963e-10
# fixup() boundary epsilon (RNG.c:86, step away from 0/1).
_I2_32M1 = 2.328306437080797e-10


class RMersenneTwister:
    """R's default RNG, reproducible bit-exactly across platforms."""

    __slots__ = ("_mt", "_mti")

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        # ``RNG_Init``: warm 50 LCG steps, then 625 more to fill
        # ``i_seed[0..624]``. ``FixupSeeds(initial=1)`` overwrites
        # ``i_seed[0]`` with ``N=624`` so the first ``MT_genrand`` call
        # regenerates the state.
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        state = [0] * _MT_N
        # ``state[0]`` is i_seed[0] which gets thrown away; we still need
        # to advance the LCG so the 624 state words match R.
        s = (69069 * s + 1) & 0xFFFFFFFF
        for j in range(_MT_N):
            s = (69069 * s + 1) & 0xFFFFFFFF
            state[j] = s
        self._mt = state
        self._mti = _MT_N  # force regen on first genrand

    def _genrand_int32(self) -> int:
        if self._mti >= _MT_N:
            mt = self._mt
            for kk in range(_MT_N - _MT_M):
                y = (mt[kk] & _MT_UPPER_MASK) | (mt[kk + 1] & _MT_LOWER_MASK)
                mt[kk] = mt[kk + _MT_M] ^ (y >> 1) ^ ((y & 1) * _MT_MATRIX_A)
            for kk in range(_MT_N - _MT_M, _MT_N - 1):
                y = (mt[kk] & _MT_UPPER_MASK) | (mt[kk + 1] & _MT_LOWER_MASK)
                mt[kk] = mt[kk + (_MT_M - _MT_N)] ^ (y >> 1) ^ ((y & 1) * _MT_MATRIX_A)
            y = (mt[_MT_N - 1] & _MT_UPPER_MASK) | (mt[0] & _MT_LOWER_MASK)
            mt[_MT_N - 1] = mt[_MT_M - 1] ^ (y >> 1) ^ ((y & 1) * _MT_MATRIX_A)
            self._mti = 0
        y = self._mt[self._mti]
        self._mti += 1
        # Tempering (RNG.c:716-719).
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def unif_rand(self) -> float:
        u = self._genrand_int32() * _INV_2P32
        # ``fixup`` (RNG.c:100-105). MT output is in (0, 1) for non-zero
        # state; these branches almost never fire, but kept for parity.
        if u <= 0.0:
            return 0.5 * _I2_32M1
        if (1.0 - u) <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return u

    def _rbits(self, bits: int) -> int:
        """``rbits(bits)`` — RNG.c:875-885. Returns int in [0, 2^bits)."""
        v = 0
        n = 0
        while n <= bits:
            v1 = int(self.unif_rand() * 65536)
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """``R_unif_index(dn)`` for ``Sample_kind=REJECTION`` (R 3.6+
        default). Returns int in [0, dn)."""
        if dn <= 0:
            return 0
        # ``ceil(log2(dn))``: for dn=1 use 0; otherwise ``(dn-1).bit_length()``.
        bits = (dn - 1).bit_length() if dn > 1 else 0
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    def sample_replace(self, n: int, k: int) -> np.ndarray:
        """``sample(n, k, replace=TRUE)`` — 0-based indices in [0, n)."""
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            out[i] = self.unif_index(n)
        return out

    def sample_no_replace(self, n: int, k: int) -> np.ndarray:
        """``sample(n, k, replace=FALSE)`` — 0-based indices in [0, n).

        R uses the ``replace`` branch for ``k < 2`` (no allocation), but
        the resulting draw is identical to a single ``unif_index(n)`` so
        we route k=1 through the FY path uniformly. For k >= 2 this is
        partial Fisher-Yates with swap-and-pop:

            x[0..n-1] = 0..n-1
            for i in 0..k-1:
                j = unif_index(n_remaining)
                out[i] = x[j]
                x[j] = x[--n_remaining]
        """
        if k < 0 or k > n:
            raise ValueError(f"k={k} not in [0, n={n}]")
        x = list(range(n))
        m = n
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            j = self.unif_index(m)
            out[i] = x[j]
            m -= 1
            x[j] = x[m]
        return out


# ---------------------------------------------------------------------------
# compress.df  — bam.r:122-184
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CompressResult:
    """Return value of :func:`compress_df`.

    ``xu`` is a dict matching the input ``dat`` layout: keys are the
    column names; values are the discretised unique values (1-D arrays).
    For matrix arguments the unique-pool is shared across all matrix
    columns of the input — values appear once each.

    ``k`` is the index mapping. Its shape is ``(n,)`` when *every*
    column of the input is 1-D, and ``(n, m_cols)`` when any column is a
    matrix (the matrix-argument summation case). Indices are 0-based;
    ``xu[name][k[i, q]]`` reproduces row ``i``, matrix-column ``q`` of
    the discretised input.
    """
    xu: dict[str, np.ndarray]
    k: np.ndarray


def compress_df(dat: dict[str, np.ndarray], m: Optional[int] = None,
                *, rng: Optional[RMersenneTwister] = None) -> _CompressResult:
    """Discretise a small dataframe by rounding (numeric) / dedup (factor).

    Direct port of mgcv ``compress.df`` (bam.r:122-184). The input
    ``dat`` is a dict of named columns sharing the same number of rows
    ``n``. Each column is one of:

      * 1-D numeric ``np.ndarray`` — rounded to a length-``m`` regular
        grid spanning ``[min, max]`` of the column.
      * 1-D factor-like (string / object dtype, or any ``np.ndarray`` of
        kind ``"O"``/``"U"``) — kept verbatim, dedup'd to its unique
        levels.
      * 2-D numeric ``np.ndarray`` of shape ``(n, m_cols)`` — under
        mgcv's summation convention, the entries are pooled into one
        unique table; the returned ``k`` is shaped ``(n, m_cols)`` so
        ``xu[name][k[i, q]]`` reproduces matrix entry ``[i, q]``.

    The default ``m`` follows mgcv: 1000 if a single variable, 100 in
    2-D, 25 in 3+. If supplied for a multi-variable input, mgcv reduces
    it to ``round(m**(1/d)) + 1`` so the joint grid stays bounded.

    A random shuffle of the unique-row order (mgcv bam.r:170-175) breaks
    spurious dependencies between jointly-discretised covariates, which
    would otherwise confuse the ``gam.side`` identifiability check. The
    caller is responsible for fixing the RNG state outside this routine
    (mgcv uses ``temp.seed(8547)`` in :func:`discrete_mf`); supply
    ``rng=`` to override. The default :class:`RMersenneTwister` is seeded
    to match ``discrete.mf``'s ``temp.seed(8547)``.
    """
    if rng is None:
        rng = RMersenneTwister(8547)
    names = list(dat.keys())
    d = len(names)
    n = next(iter(dat.values())).shape[0]
    if m is None:
        m = 1000 if d == 1 else (100 if d == 2 else 25)
    elif d > 1:
        # mgcv: m <- round(m^{1/d}) + 1
        m = int(round(m ** (1.0 / d))) + 1

    # Detect factor / matrix columns. mgcv treats string / object arrays
    # as factors; numeric (any float / int dtype) gets the rounding path.
    is_factor = {nm: _is_factor_arr(dat[nm]) for nm in names}
    is_matrix = {nm: dat[nm].ndim == 2 for nm in names}

    # mgcv ``mm`` (metric grid points) and ``mf`` (factor grid points)
    # — used as the cap above which rounding kicks in. Factor cols
    # contribute their level count; numeric cols contribute ``m``.
    mf_total = 1
    mm_total = 1
    for nm in names:
        if is_factor[nm]:
            mf_total *= int(np.unique(np.asarray(dat[nm]).ravel()).size)
        else:
            mm_total *= m

    # mgcv: if the first column is a matrix, all columns are vectorised
    # (matrix-arg case). Build a working dict of 1-D arrays of length n*m_cols.
    matrix_input = is_matrix[names[0]]
    if matrix_input:
        ncols_mat = dat[names[0]].shape[1]
        flat: dict[str, np.ndarray] = {}
        for nm in names:
            arr = dat[nm]
            if arr.ndim == 1:
                # broadcast scalar columns across the matrix-column axis
                arr = np.broadcast_to(arr[:, None], (n, ncols_mat))
            flat[nm] = arr.reshape(-1)
        work = flat
        n_eff = n * ncols_mat
    else:
        work = {nm: np.asarray(dat[nm]).ravel() for nm in names}
        n_eff = n

    # Initial uniquecombs on raw (or vectorised) input.
    xu_table, k_idx = _uniquecombs(work, names)

    if xu_table[names[0]].size > mm_total * mf_total:
        # Too many unique combinations — round metric variables to an
        # m-point grid before re-deduplicating (mgcv bam.r:155-163).
        rounded = {}
        for nm in names:
            if is_factor[nm]:
                rounded[nm] = work[nm]
            else:
                col = work[nm].astype(float)
                xl_lo = float(np.min(col))
                xl_hi = float(np.max(col))
                if xl_hi == xl_lo:
                    rounded[nm] = col.copy()
                else:
                    grid = np.linspace(xl_lo, xl_hi, m)
                    dx = grid[1] - grid[0]
                    kx = np.round((col - xl_lo) / dx).astype(int)
                    rounded[nm] = grid[kx]
        work = rounded
        xu_table, k_idx = _uniquecombs(work, names)

    nu = xu_table[names[0]].size

    if nu == n_eff:
        # No compression possible — return original ordering with identity index.
        k_out = np.arange(n_eff, dtype=np.int64)
        if matrix_input:
            k_out = k_out.reshape(n, ncols_mat)
        return _CompressResult(xu={nm: work[nm].copy() for nm in names}, k=k_out)

    # Shuffle xu rows to break induced dependencies (bam.r:171).
    perm = rng.sample_no_replace(nu, nu)
    xu_table = {nm: xu_table[nm][np.argsort(perm)] for nm in names}
    # ``perm[old_pos] = new_pos``; old k pointed to old_pos, after the
    # shuffle the same data should point to new_pos.
    k_idx = perm[k_idx]

    if matrix_input:
        k_idx = k_idx.reshape(n, ncols_mat)

    return _CompressResult(xu=xu_table, k=k_idx.astype(np.int64))


def _is_factor_arr(a: np.ndarray) -> bool:
    """Treat string / unicode / object arrays as factors."""
    return a.dtype.kind in ("U", "O", "S")


def _uniquecombs(work: dict[str, np.ndarray],
                 names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Numpy port of R's ``uniquecombs`` (single-thread).

    Returns ``(xu, idx)`` where ``xu`` is a dict of unique columns (in
    canonical sort order) and ``idx[i]`` is the unique-row index for
    input row ``i``.
    """
    n = next(iter(work.values())).size
    if len(names) == 1:
        col = work[names[0]]
        u, inv = np.unique(col, return_inverse=True)
        return {names[0]: u}, inv.astype(np.int64)
    # Multi-column: stack into a structured key. Numeric columns are kept
    # numeric; factor columns are converted to integer codes with the
    # same lex order as ``np.unique``.
    keys: list[np.ndarray] = []
    for nm in names:
        col = work[nm]
        if _is_factor_arr(col):
            _, codes = np.unique(col, return_inverse=True)
            keys.append(codes)
        else:
            keys.append(col)
    # Use a structured array to do row-wise uniquecombs.
    dtype = np.dtype([(nm, k.dtype) for nm, k in zip(names, keys)])
    arr = np.empty(n, dtype=dtype)
    for nm, k in zip(names, keys):
        arr[nm] = k
    u, inv = np.unique(arr, return_inverse=True)
    xu = {nm: u[nm] for nm in names}
    # For factor cols we stored codes; the unique table here returns
    # codes in lex order — but the original column had string values.
    # We need to map codes back to the original strings.
    for nm in names:
        col = work[nm]
        if _is_factor_arr(col):
            levels, _ = np.unique(col, return_inverse=True)
            xu[nm] = levels[xu[nm]]
    return xu, inv.astype(np.int64)


# ---------------------------------------------------------------------------
# discrete.mf  — bam.r:201-380
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiscretizedFrame:
    """Output of :func:`discrete_mf`. Compact representation of an
    arbitrary mgcv model frame after per-marginal discretisation.

    Attributes
    ----------
    mf : dict[str, np.ndarray]
        Discretised model frame. Each named entry is a 1-D array; values
        are padded to the longest unique-table length (``maxr`` in mgcv)
        with random repeats so all entries share length, matching
        ``mgcv::discrete.mf`` (bam.r:355-365). Padding rows are *never*
        referenced via ``k`` — they exist only so downstream code can
        treat ``mf`` as a regular frame.
    k : np.ndarray
        Integer index matrix of shape ``(n, n_k_cols)``. ``k[i, q]`` is
        the row-index into the discretised table for variable
        ``names_for_col(q)`` of original observation ``i``. ``n_k_cols``
        sums over scalar variables (one column each) plus matrix
        variables (one column per matrix column under the summation
        convention). The final column is the all-ones intercept index.
    ks : np.ndarray
        ``(n_vars + 1, 2)`` int matrix. ``ks[j, 0]:ks[j, 1]`` is the
        slice of ``k`` columns associated with the ``j``th variable in
        ``names``. Final row corresponds to the intercept.
    nr : np.ndarray
        ``(n_vars + 1,)`` int vector. ``nr[j]`` is the number of unique
        rows for variable ``j`` (length of the un-padded ``mf`` entry).
        Final entry is 1 for the intercept.
    names : list[str]
        Variable names matching ``ks``/``nr``, in the order they were
        first discretised. ``"(Intercept)"`` is appended last.
    n : int
        Original (un-discretised) row count.
    """
    mf: dict[str, np.ndarray]
    k: np.ndarray
    ks: np.ndarray
    nr: np.ndarray
    names: list[str]
    n: int


def check_term(term: Sequence[str], rec: dict) -> int:
    """Has any variable in ``term`` already been discretised?

    Direct port of mgcv ``check.term`` (bam.r:185-200). Returns the
    1-based index of the prior discretisation if every variable in
    ``term`` matches an existing discretisation of the same dimension;
    raises :class:`ValueError` for partial overlap (mgcv: "bam can not
    discretize with this nesting structure"); returns 0 if no overlap.
    """
    vnames = rec["vnames"]
    ki = rec["ki"]
    d = rec["d"]
    ii = [j for j, nm in enumerate(vnames) if nm in term]
    if ii:
        i_min = min(ii)
        if len(term) == d[i_min]:
            if any(t not in [vnames[j] for j in ii] for t in term):
                raise ValueError(
                    "bam can not discretize with this nesting structure"
                )
            return ki[i_min]
        raise ValueError("bam can not discretize with this nesting structure")
    return 0


def discrete_mf(smooth_specs: list[dict], mf: pl.DataFrame,
                names_pmf: Sequence[str], m: Optional[int] = None,
                *, rng: Optional[RMersenneTwister] = None,
                full: bool = True) -> DiscretizedFrame:
    """Discretise the model frame ``mf`` per marginal of every smooth.

    Direct port of mgcv ``discrete.mf`` (bam.r:201-380). Walks the
    ``smooth_specs`` list (one dict per smooth, with keys ``term`` —
    list[str] of variables, ``by`` — str/None, ``margins`` — list of
    margin specs each with its own ``term``), discretises each
    marginal's variables jointly, and assembles:

      * ``mf``  — discretised, padded to common length ``maxr``
      * ``k``   — index matrix
      * ``ks``  — per-variable ``k``-column ranges
      * ``nr``  — per-variable un-padded length

    Followed by the parametric covariates (``names_pmf``), each
    discretised individually, and finally an intercept index column.

    ``rng`` should be supplied with a fixed seed for reproducibility.
    mgcv uses ``temp.seed(8547)`` (bam.r:233) — the default
    ``RMersenneTwister(8547)`` matches that exactly.
    """
    if rng is None:
        rng = RMersenneTwister(8547)

    n = mf.height
    # Pre-count how many index columns ``k`` will need: each smooth term
    # contributes ``len(margins) + (by != None)`` marginals; each parametric
    # variable contributes 1.
    nk = 0
    for spec in smooth_specs:
        n_marg = len(spec.get("margins", [{"term": spec["term"]}]))
        nk += n_marg + (1 if spec.get("by") not in (None, "NA") else 0)
    pmf_in_mf = [nm for nm in names_pmf if nm in mf.columns]
    nk += len(pmf_in_mf)

    # Bookkeeping. ``k`` will grow with extra columns when matrix-arg
    # smooths are encountered (each matrix column becomes one ``k`` column).
    k = np.zeros((n, nk), dtype=np.int64)
    ks = np.full((nk, 2), -1, dtype=np.int64)
    nr = np.zeros(nk, dtype=np.int64)
    var_order: list[str] = []
    mf0: dict[str, np.ndarray] = {}
    rec = {"vnames": [], "ki": [], "d": []}
    ik = -1  # 0-based marginal index counter (mgcv ``ik`` is 1-based)

    # Walk smooths, discretising each marginal once.
    def _discretise_marginal(termi: list[str], mi: Optional[int]):
        nonlocal ik
        prev = check_term(termi, rec)
        if prev != 0:
            return  # already discretised — re-use the entry
        ik += 1
        # Pull out the columns referenced in ``termi`` from the model frame.
        dat: dict[str, np.ndarray] = {}
        for nm in termi:
            s = mf[nm]
            if is_matrix_col(s):
                dat[nm] = matrix_to_2d(s)
            else:
                dat[nm] = s.to_numpy()
        cr = compress_df(dat, m=mi, rng=rng)
        ki = cr.k                                # (n,) or (n, m_cols)
        if ki.ndim == 1:
            ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
            ks[ik, 1] = ks[ik, 0] + 1
            # A scalar margin following a matrix-arg margin in the same
            # smooth-spec list can land past hea's pre-counted ``nk``
            # (which counts one slot per margin, not per matrix-column).
            # Extend ``k`` here too (mgcv's ``cbind`` happens whenever the
            # incoming index is a matrix; scalar writes assume room exists,
            # which is only guaranteed if no prior margin was matrix-arg).
            if ks[ik, 1] > k.shape[1]:
                k_ext = np.zeros(
                    (n, ks[ik, 1] - k.shape[1]), dtype=np.int64,
                )
                k_full = np.concatenate([k, k_ext], axis=1)
                _set_k(k_full)
                k_full[:, ks[ik, 0]] = ki
            else:
                k[:, ks[ik, 0]] = ki
        else:
            ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
            ks[ik, 1] = ks[ik, 0] + ki.shape[1]
            # Extend k if needed.
            need_cols = ks[ik, 1] - k.shape[1]
            if need_cols > 0:
                k_ext = np.zeros((n, need_cols), dtype=np.int64)
                # ``k`` is captured in closure; rebind via mutation
                k_full = np.concatenate([k, k_ext], axis=1)
                # Replace the whole column block.
                _set_k(k_full)
                k_local = k_full
            else:
                k_local = k
            k_local[:, ks[ik, 0]:ks[ik, 1]] = ki
        nr[ik] = cr.xu[termi[0]].size
        # Take the first variable's column as the canonical mf entry name.
        # Each variable in termi maps to its own discretised column in mf0
        # but they share one set of (k, ks, nr) — mgcv does this via
        # duplicated rows in nr/ks, so we replicate that here.
        nr_first = nr[ik]
        ks_first = ks[ik].copy()
        var_order.append(termi[0])
        mf0[termi[0]] = cr.xu[termi[0]]
        # Duplicate index info for every additional variable in this
        # joint discretisation (bam.r:255-262).
        for extra in termi[1:]:
            ik += 1
            ks[ik] = ks_first
            nr[ik] = nr_first
            var_order.append(extra)
            mf0[extra] = cr.xu[extra]
        # Update the dedup record.
        rec["vnames"].extend(termi)
        rec["ki"].extend([ik - len(termi) + 1] * len(termi))
        rec["d"].extend([len(termi)] * len(termi))

    # Helper for the rare matrix-arg ``k`` extension above.
    def _set_k(k_new: np.ndarray):
        nonlocal k
        k = k_new

    # --- smooths ---
    for spec in smooth_specs:
        margins = spec.get("margins", [{"term": spec["term"]}])
        by = spec.get("by")
        # ``by`` is processed first (matches mgcv jj==1 path).
        if by not in (None, "NA"):
            _discretise_marginal([by], m)
        for marg in margins:
            _discretise_marginal(list(marg["term"]), m)

    # --- parametric ---
    # mgcv passes ``pmf.names = names(model.frame(parametric_formula, data))``
    # which always *includes the response* (since ``model.frame(y ~ ...)``
    # evaluates the LHS into a column). The response usually has < n unique
    # values, so its ``compress.df`` shuffle consumes RNG calls that
    # otherwise wouldn't fire. Skipping it leaves the RNG state desynced
    # from mgcv at the pad loop. Any column listed in ``names_pmf`` that is
    # actually present in ``mf`` gets discretised here, response included.
    for nm in pmf_in_mf:
        # Skip if already discretised (a parametric covariate shared with a
        # smooth — mgcv guards via ``rec``).
        if check_term([nm], rec) != 0:
            continue
        ik += 1
        s = mf[nm]
        if is_matrix_col(s):
            arr = matrix_to_2d(s)
            cr = compress_df({nm: arr}, m=m, rng=rng)
            mf_entry = cr.xu[nm]
            ki = cr.k.ravel()  # parametric matrix is dropped to vector
            nr[ik] = mf_entry.size
        else:
            arr = s.to_numpy()
            cr = compress_df({nm: arr}, m=m, rng=rng)
            mf_entry = cr.xu[nm]
            ki = cr.k
            nr[ik] = mf_entry.size
        ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
        ks[ik, 1] = ks[ik, 0] + 1
        # Matrix-arg smooth margins may have grown ``k`` past the
        # pre-counted ``nk``; extend by one column if our write would
        # otherwise be out-of-bounds (mgcv side: cbind in the smooth loop
        # already reserved enough room because its pre-count uses
        # ``length(term)`` with one slot per *variable*; hea's pre-count
        # uses one slot per margin, which can be smaller for matrix args).
        if ks[ik, 1] > k.shape[1]:
            k = np.concatenate(
                [k, np.zeros((n, ks[ik, 1] - k.shape[1]), dtype=np.int64)],
                axis=1,
            )
        k[:, ks[ik, 0]] = ki
        var_order.append(nm)
        mf0[nm] = mf_entry

    # --- pad mf0 to common length ---
    if full and mf0:
        maxr = max(arr.size for arr in mf0.values())
        for nm, arr in list(mf0.items()):
            if arr.size < maxr:
                # mgcv: ``mf0[[i]][(me+1):maxr] <- sample(mf0[[i]], maxr-me, replace=TRUE)``
                pad = arr[rng.sample_replace(arr.size, maxr - arr.size)]
                mf0[nm] = np.concatenate([arr, pad])
    else:
        maxr = max((arr.size for arr in mf0.values()), default=0)

    # --- intercept ---
    ik += 1
    # Trim k to the columns actually used.
    used_cols = int(np.max(ks[:ik]) if ik > 0 else 0)
    if used_cols < k.shape[1]:
        k = k[:, :used_cols]
    elif used_cols > k.shape[1]:
        # k was extended for matrix args; ensure size matches.
        pass
    # Append the intercept column (all 0 → unique row 0 = the constant 1).
    k = np.concatenate([k, np.zeros((n, 1), dtype=np.int64)], axis=1)
    ks_final = np.concatenate(
        [ks[:ik], np.array([[k.shape[1] - 1, k.shape[1]]], dtype=np.int64)],
        axis=0,
    )
    nr_final = np.concatenate([nr[:ik], np.array([1], dtype=np.int64)])
    var_order.append("(Intercept)")
    mf0["(Intercept)"] = np.ones(1, dtype=float)

    return DiscretizedFrame(
        mf=mf0, k=k, ks=ks_final, nr=nr_final, names=var_order, n=n,
    )


# ---------------------------------------------------------------------------
# Discrete design — per-marginal Xd blocks plus term packing.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _DiscreteTerm:
    """Per-term entry in :class:`DiscreteDesign`.

    A "term" here is one row in mgcv's ``ts``/``dt`` structure: one
    block of contiguous columns in the global coefficient vector.
    Parametric blocks have ``kind="param"`` and a single ``Xd``
    referenced by a single ``k`` column. Smooth blocks have one
    ``Xd`` per marginal with column-index entries ``k_cols`` aligning
    to the global ``k[:, ks[var, 0]:ks[var, 1]]`` slice for each
    marginal.
    """
    kind: str                        # "param" | "single" | "tensor"
    Xd_list: list[np.ndarray]        # one (m_j, p_j) block per marginal
    k_cols: list[tuple[int, int]]    # (start, stop) into global k for each marginal
    coef_slice: slice                # where this term lives in the full coef
    qc: int = 0                      # tensor-constraint indicator (1 if Householder)
    v: Optional[np.ndarray] = None   # Householder vec, length = Π p_j (qc=1 case)
    # absorb / by / keep_cols replays for the *whole* term — applied as
    # column transforms on the (Khatri-Rao) tensor block. None for params
    # and unconstrained smooths.
    absorb: Optional[object] = None
    by: Optional[object] = None
    keep_cols: Optional[np.ndarray] = None
    # Pre-computed length-n by-mask (factor: indicator; numeric: scalar
    # multiplier). Cached at design-build time because the by-column lives
    # in the original input data, not in ``dframe.mf`` (which only carries
    # discretised marginals). Invariant under PIRLS / outer-Newton.
    by_mask: Optional[np.ndarray] = None
    # Predict-time replay (used for predict.bamd, not the fitter).
    spec: Optional[BasisSpec] = None
    label: str = ""


@dataclass(slots=True)
class DiscreteDesign:
    """Compressed design store consumed by the discrete kernels.

    Mirrors mgcv's ``G$Xd``/``G$k``/``G$ks``/``G$ts``/``G$dt``/``G$v``/
    ``G$qc``/``G$drop`` package (bam.r:2300+ in ``bam`` setup), but
    represented as a Python dataclass with one :class:`_DiscreteTerm`
    per term. Total coef count is ``p`` (sum over terms of their post-
    constraint column count).
    """
    terms: list[_DiscreteTerm]
    k: np.ndarray                    # (n, n_k_cols) global index matrix
    ks: np.ndarray                   # (n_vars+1, 2) per-variable k-column slice
    nr: np.ndarray                   # (n_vars+1,) per-variable unique-row count
    n: int                           # original observation count
    p: int                           # total coef count (post-constraint)
    var_index: dict[str, int]        # variable name → row in ks/nr
    # Lazily populated caches — invariant under PIRLS (only depend on
    # the design's Xd_list / k / constraint structure, not on weights
    # or coefs). Set to ``False`` to disable caching for very-large-n
    # cases where ``X_full`` doesn't fit in memory.
    _full_X_cache: object = None
    _T_cache: object = None


def build_discrete_design(blocks: list[SmoothBlock],
                          X_param_full: np.ndarray,
                          dframe: DiscretizedFrame,
                          *,
                          param_terms: Sequence[str] = ("(Intercept)",),
                          data: Optional[pl.DataFrame] = None,
                          ) -> DiscreteDesign:
    """Build :class:`DiscreteDesign` from a fitted set of
    :class:`SmoothBlock` plus a discretised model frame.

    The basis machinery for each smooth (knots, eigenbasis, by-mask,
    absorb constraint) has *already* been frozen in ``block.spec`` —
    typically by running ``materialize_smooths`` on a representative
    subsample (mgcv's ``mini.mf`` flow). What this routine does:

      * For each smooth, evaluate the per-marginal raw bases at the
        unique discretised values (one tiny ``predict_mat``-style call
        per margin), giving a list of ``Xd_j`` of shape ``(nr_j, p_j_raw)``.
      * Capture the term's ``by`` / ``absorb`` / ``keep_cols`` so the
        kernels can apply them on the row-tensor product at compute time.
      * For parametric columns, store the un-discretised columns of
        ``X_param_full`` directly (no compression) — equivalent to a
        single-marginal "term" with ``Xd = X_param_full`` and ``k =
        identity``.

    Constraint Householder vectors (mgcv ``v``/``qc``) are *not* yet
    extracted from ``absorb`` — for now we apply absorb post-hoc as a
    column transform on the materialised row-tensor. A future
    optimisation can switch to Householder for ``te`` smooths.
    """
    var_index = {nm: j for j, nm in enumerate(dframe.names)}
    terms: list[_DiscreteTerm] = []
    p_total = 0

    # Parametric columns: treat each parametric column-group as a single
    # "term". For now we keep them as one big block — index into
    # X_param_full via a synthetic identity ``k`` column. The simplest
    # representation just stores the full param matrix as a single Xd
    # with k-col = -1 (signal: identity gather). The kernels handle this
    # via a special-case branch.
    if X_param_full is not None and X_param_full.shape[1] > 0:
        p_par = X_param_full.shape[1]
        terms.append(_DiscreteTerm(
            kind="param",
            Xd_list=[np.asarray(X_param_full, dtype=float)],
            k_cols=[(-1, -1)],     # sentinel: gather is identity
            coef_slice=slice(p_total, p_total + p_par),
            label="(parametric)",
        ))
        p_total += p_par

    # Discretised model frame as a polars DataFrame with each column
    # length = nr[var] (no padding here — we feed only the unique values
    # so basis evaluators see the right domain).
    for block in blocks:
        spec = block.spec
        if spec is None:
            raise ValueError(
                f"SmoothBlock {block.label!r} has no spec — predict_mat "
                "replay is required for the discrete fitter"
            )
        term_vars = list(block.term)
        # Identify margins: tensor smooths have ``raw`` of type
        # ``_TensorRawBasis`` / ``_T2RawBasis`` / ``_T2PredictRawBasis``;
        # everything else is single-margin. Each margin variable list comes
        # from the raw basis itself (``_raw_basis_vars``) — that's the only
        # source that survives mgcv's ``tero`` reorder, where the block's
        # declaration order (``block.term``) no longer matches the post-
        # tero margin order.
        raw = spec.predict_raw if spec.predict_raw is not None else spec.raw
        if isinstance(raw, _TensorRawBasis):
            margin_raws = list(raw.margins)
            margin_vars = [_raw_basis_vars(m) or term_vars for m in margin_raws]
        elif isinstance(raw, (_T2RawBasis, _T2PredictRawBasis)):
            margin_raws = list(raw.margins)
            margin_vars = [_raw_basis_vars(m) or term_vars for m in margin_raws]
        else:
            margin_raws = [raw]
            margin_vars = [_raw_basis_vars(raw) or term_vars]

        # Evaluate each marginal raw basis on the discretised unique
        # values for its variables. The frame for marginal j is built
        # from ``dframe.mf`` taking exactly the variables in
        # ``margin_vars[j]``, length ``nr[var0]``.
        Xd_list: list[np.ndarray] = []
        k_cols: list[tuple[int, int]] = []
        for mvars, mraw in zip(margin_vars, margin_raws):
            v0 = mvars[0]
            j_var = var_index[v0]
            length_j = int(dframe.nr[j_var])
            sub = {nm: dframe.mf[nm][:length_j] for nm in mvars}
            sub_df = pl.DataFrame(sub)
            Xd = mraw.eval(sub_df)
            Xd_list.append(np.asarray(Xd, dtype=float))
            k_cols.append((int(dframe.ks[j_var, 0]), int(dframe.ks[j_var, 1])))

        # Term column count after by/absorb/keep_cols. Use block.X.shape[1]
        # as the authoritative post-transform width.
        p_term = block.X.shape[1]
        kind = "single" if len(margin_raws) == 1 else "tensor"

        # Pre-compute the by-mask on the original n rows so the kernels
        # (which can't see ``data``) can apply it row-wise. Factor:
        # indicator (col == level), float; numeric: col values, float.
        # Both apply identically as ``X *= by_mask[:, None]``.
        by_mask: Optional[np.ndarray] = None
        if spec.by is not None:
            if data is None:
                raise ValueError(
                    f"build_discrete_design: smooth {block.label!r} has "
                    "by= but no data was passed; cannot evaluate the "
                    "by-column on the original n rows. Pass ``data=`` "
                    "to build_discrete_design."
                )
            col = _eval_by_col(spec.by.expr, data)
            arr = col.to_numpy() if isinstance(col, pl.Series) else col
            if spec.by.kind == "factor":
                by_mask = (arr == spec.by.level).astype(float)
            elif spec.by.kind == "numeric":
                by_mask = np.asarray(arr, dtype=float)
            else:
                raise ValueError(
                    f"build_discrete_design: unsupported by.kind="
                    f"{spec.by.kind!r} on smooth {block.label!r}"
                )

        terms.append(_DiscreteTerm(
            kind=kind,
            Xd_list=Xd_list,
            k_cols=k_cols,
            coef_slice=slice(p_total, p_total + p_term),
            absorb=spec.absorb,
            by=spec.by,
            by_mask=by_mask,
            keep_cols=spec.keep_cols,
            spec=spec,
            label=block.label,
        ))
        p_total += p_term

    return DiscreteDesign(
        terms=terms, k=dframe.k, ks=dframe.ks, nr=dframe.nr,
        n=dframe.n, p=p_total, var_index=var_index,
    )


def _raw_basis_vars(raw: _RawBasis) -> list[str]:
    """Return the variable names a raw basis evaluates on.

    Walks past ``_LinearTransformRawBasis`` wrappers (which do not declare
    their own ``term`` — they inherit from the inner basis). For mgcv-style
    leaf classes (``_CRRawBasis``, ``_TPRawBasis``, etc.) ``term`` is either
    a single string (1-D) or a list of strings (multi-D). Returns a list
    in either case so callers can iterate uniformly.
    """
    inner = raw
    while isinstance(inner, _LinearTransformRawBasis):
        inner = inner.inner
    term_attr = getattr(inner, "term", None)
    if term_attr is None:
        return []
    if isinstance(term_attr, str):
        return [term_attr]
    return list(term_attr)


def _split_term_vars_by_margins(term_vars: list[str],
                                margin_raws: list[_RawBasis]) -> list[list[str]]:
    """Best-effort decomposition of ``term_vars`` across margin bases.

    For multi-d margins the raw basis carries no explicit variable list,
    so we fall back to splitting ``term_vars`` evenly: each margin gets
    the next chunk in declaration order. This matches mgcv's
    convention: ``te(x, y, z, d=c(2,1))`` gives margins
    ``[("x","y"), ("z",)]``.
    """
    n_marg = len(margin_raws)
    if n_marg == 1:
        return [list(term_vars)]
    # Heuristic: split evenly. For most te smooths each margin is 1-D.
    if len(term_vars) == n_marg:
        return [[v] for v in term_vars]
    # Multi-d margins: walk margin raws looking for ``ranks``/``margins``
    # attrs that hint at per-margin dimensionality. For ``_TensorRawBasis``
    # we could check if each margin's basis itself is a tensor — but in
    # practice te()'s margins are univariate. Fall back to evenly
    # distributing extras to the first margin (matches mgcv's d=
    # ordering).
    chunks: list[list[str]] = []
    extra = len(term_vars) - n_marg
    cursor = 0
    for j in range(n_marg):
        size = 1 + (extra if j == 0 else 0)
        chunks.append(term_vars[cursor:cursor + size])
        cursor += size
        extra = 0
    return chunks


# ---------------------------------------------------------------------------
# Kernels — Xbd / XWXd / XWyd
# ---------------------------------------------------------------------------


def _term_full_design(term: _DiscreteTerm, k: np.ndarray, n: int) -> np.ndarray:
    """Materialise ``X_term`` (n × p_term_post) for one term.

    Used by the kernels as a generic fallback: gather per-marginal Xd,
    Khatri-Rao multiply, sum across summation columns, then apply
    ``by`` / ``absorb`` / ``keep_cols``. This is the *correct*
    semantics; the optimised path (Householder constraint, tensor-
    aware scatter-add for X'WX) replicates this row-by-row but avoids
    forming the n × p_raw block.

    For now the fallback is the only path — it still avoids the
    chunked-QR overhead because ``Xd_list`` is small and reusable
    across PIRLS iterations.
    """
    if term.kind == "param":
        return term.Xd_list[0]

    # Determine summation width (same across margins for a given term —
    # mgcv enforces this by jointly discretising matrix-arg margins).
    q_lo, q_hi = term.k_cols[0]
    n_sum = q_hi - q_lo

    # Khatri-Rao across margins, then sum over summation columns.
    X_full = None
    for q_off in range(n_sum):
        # For each margin, gather rows: Xd_j[k[:, ks_j_lo + q_off], :]
        Xq_blocks = []
        for Xd_j, (ks_lo, ks_hi) in zip(term.Xd_list, term.k_cols):
            kcol = k[:, ks_lo + q_off]
            Xq_blocks.append(Xd_j[kcol])
        # Khatri-Rao over the margin blocks.
        if len(Xq_blocks) == 1:
            Xq = Xq_blocks[0]
        else:
            Xq = Xq_blocks[0]
            for nxt in Xq_blocks[1:]:
                # Row tensor: (n, p1) ⊗_row (n, p2) → (n, p1*p2)
                pa = Xq.shape[1]
                pb = nxt.shape[1]
                Xq = (Xq[:, :, None] * nxt[:, None, :]).reshape(-1, pa * pb)
        if X_full is None:
            X_full = Xq
        else:
            X_full = X_full + Xq

    # Apply by-mask, then absorb, then keep_cols. Order matches
    # ``BasisSpec.predict_mat`` (formula.py:2415-2435): raw → by → absorb.
    # ``term.by_mask`` is pre-computed length-n at build_discrete_design
    # time (factor: indicator; numeric: scalar multiplier per row).
    if term.by_mask is not None:
        X_full = X_full * term.by_mask[:, None]
    if term.absorb is not None:
        X_full = term.absorb.apply(X_full)
    if term.keep_cols is not None:
        X_full = X_full[:, term.keep_cols]
    return X_full


def discrete_full_X(design: DiscreteDesign) -> np.ndarray:
    """Materialise the full ``n × p`` design matrix from the compressed
    representation.

    Result is cached on the design object — basis values on the
    discretised unique grid (``Xd_list``) are fixed for the life of the
    design, so the materialisation is invariant across PIRLS iters and
    across outer-Newton steps. The cache is populated lazily on first
    call and reused thereafter; this turns every downstream
    ``Xbd``/``XWXd``/``XWyd`` invocation into a single BLAS matmul on
    cached ``X_full``, which dominates scatter-add for the
    moderate-``n`` × moderate-``p`` regime where chicago lives.

    For very large ``n`` where ``X_full`` doesn't fit in memory, callers
    can opt out of caching by setting ``design._full_X_cache = False``
    before invoking the kernels — they will then route through the
    scatter-add paths in ``Xbd``/``XWXd``/``XWyd`` directly.
    """
    cache = design._full_X_cache
    if isinstance(cache, np.ndarray):
        return cache
    n = design.n
    blocks = [_term_full_design(t, design.k, n) for t in design.terms]
    full = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    if cache is not False:                 # ``False`` disables caching
        design._full_X_cache = full
    return full


# ---------------------------------------------------------------------------
# Scatter-add kernel helpers
# ---------------------------------------------------------------------------


def _grouped_sum_axis0(idx: np.ndarray, values: np.ndarray,
                       n_groups: int) -> np.ndarray:
    """Sum rows of ``values`` grouped by integer keys ``idx``.

    Pure-numpy equivalent of ``np.add.at(out, idx, values)`` but uses
    argsort + ``np.add.reduceat`` so the inner accumulation runs at
    BLAS-equivalent speed. ``idx`` is a length-n int array; ``values``
    has shape ``(n, ...)``; the output has shape ``(n_groups, ...)``.

    For 1-D ``values`` this collapses to ``np.bincount``.
    """
    if values.ndim == 1:
        return np.bincount(idx, weights=values, minlength=n_groups)
    if values.shape[0] == 0:
        return np.zeros((n_groups,) + values.shape[1:], dtype=values.dtype)
    order = np.argsort(idx, kind='stable')
    sorted_idx = idx[order]
    sorted_vals = values[order]
    unique_groups, starts = np.unique(sorted_idx, return_index=True)
    sums = np.add.reduceat(sorted_vals, starts, axis=0)
    out = np.zeros((n_groups,) + values.shape[1:], dtype=sorted_vals.dtype)
    out[unique_groups] = sums
    return out


def _term_constraint_T(term: _DiscreteTerm) -> Optional[np.ndarray]:
    """Materialise the constraint matrix ``T`` (p_raw × p_post) such that
    ``X_term_post = X_term_raw @ T``.

    Returns ``None`` for the identity case (no absorb / keep_cols), so
    callers can short-circuit the multiplication. For tensor smooths
    the absorb is the rank-1 sum-to-zero Householder; for singletons
    it's the per-margin absorb chain. Both are realised here by
    feeding ``np.eye(p_raw)`` through ``term.absorb.apply`` — the same
    path ``_term_full_design`` would use, but applied once at term
    setup rather than per row.
    """
    if term.kind == "param":
        return None
    p_raw = int(np.prod([Xd.shape[1] for Xd in term.Xd_list]))
    if term.absorb is None and term.keep_cols is None:
        return None
    T = np.eye(p_raw, dtype=float)
    if term.absorb is not None:
        T = term.absorb.apply(T)
    if term.keep_cols is not None:
        T = T[:, term.keep_cols]
    return np.ascontiguousarray(T)


def _design_constraint_Ts(design: DiscreteDesign) -> list[Optional[np.ndarray]]:
    """Per-term constraint matrices, cached on the design object.

    The ``T`` matrices depend only on the design (not on weights or
    coefs), so we compute them once and reuse across every PIRLS
    iteration.
    """
    cache = design._T_cache
    if cache is not None:
        return cache
    Ts = [_term_constraint_T(t) for t in design.terms]
    design._T_cache = Ts
    return Ts


# ---------------------------------------------------------------------------
# Per-term kernels — operate on the *unconstrained* raw column space
# (``p_raw``). Constraint application via ``T`` is layered on top in
# the public Xbd / XWyd / XWXd.
# ---------------------------------------------------------------------------


def _term_Xb_raw(term: _DiscreteTerm, b_raw: np.ndarray,
                 k: np.ndarray, n: int) -> np.ndarray:
    """Compute ``X_term_raw @ b_raw`` (length ``n``).

    Direct port of mgcv ``singleXb`` / ``tensorXb`` (discrete.c:375-444),
    but with the inner C loops replaced by numpy gathers + einsum.
    Tensor terms keep the C kernel's structure: contract the *final*
    marginal first against ``b`` to form ``C`` (the m_d-rowed working
    matrix), then per-q gather pre-final marginals at row indices and
    contract with ``C`` row-wise.
    """
    if term.kind == "param":
        return term.Xd_list[0] @ b_raw

    if term.kind == "single":
        Xd = term.Xd_list[0]
        ks_lo, ks_hi = term.k_cols[0]
        tmp = Xd @ b_raw                       # (m,)
        result = tmp[k[:, ks_lo]].copy()
        for q in range(ks_lo + 1, ks_hi):
            result += tmp[k[:, q]]
        return result

    return _tensor_Xb_raw(term, b_raw, k, n)


def _tensor_Xb_raw(term: _DiscreteTerm, b_raw: np.ndarray,
                   k: np.ndarray, n: int) -> np.ndarray:
    Xd_list = term.Xd_list
    d = len(Xd_list)
    ps = tuple(Xd.shape[1] for Xd in Xd_list)
    B = np.asarray(b_raw, dtype=float).reshape(ps)

    # C[..., g] = Σ_{l_d} B[..., l_d] · Xd_d[g, l_d]  — final marginal applied
    Xd_d = Xd_list[-1]
    if d == 2:
        C = B @ Xd_d.T                          # (p1, m_d)
    elif d == 3:
        C = np.einsum('ijd,gd->ijg', B, Xd_d)   # (p1, p2, m_d)
    else:
        in_letters = "abcdefghij"[:d]
        out_letters = in_letters[:-1] + "G"
        C = np.einsum(f"{in_letters},G{in_letters[-1]}->{out_letters}",
                      B, Xd_d)

    ks_lo_list = [term.k_cols[j][0] for j in range(d)]
    n_sum = term.k_cols[0][1] - term.k_cols[0][0]
    result = np.zeros(n, dtype=float)
    for q in range(n_sum):
        k_per_marg = [k[:, ks_lo_list[j] + q] for j in range(d)]
        if d == 2:
            X1_at_row = Xd_list[0][k_per_marg[0]]    # (n, p1)
            C_gathered = C[:, k_per_marg[1]]          # (p1, n)
            result += np.einsum('rp,pr->r', X1_at_row, C_gathered)
        elif d == 3:
            X1_at_row = Xd_list[0][k_per_marg[0]]    # (n, p1)
            X2_at_row = Xd_list[1][k_per_marg[1]]    # (n, p2)
            C_gathered = C[:, :, k_per_marg[2]]       # (p1, p2, n)
            result += np.einsum('rp,rq,pqr->r',
                                X1_at_row, X2_at_row, C_gathered)
        else:
            in_letters = "abcdefghij"[:d-1]
            X_at_rows = [Xd_list[j][k_per_marg[j]] for j in range(d-1)]
            C_gathered = C[(slice(None),) * (d-1) + (k_per_marg[d-1],)]
            expr = (",".join(f"r{l}" for l in in_letters)
                    + "," + in_letters + "r" + "->r")
            result += np.einsum(expr, *X_at_rows, C_gathered)
    return result


def _term_Xty_raw(term: _DiscreteTerm, wy: np.ndarray,
                  k: np.ndarray, n: int) -> np.ndarray:
    """Compute ``X_term_raw.T @ wy`` (length ``p_raw``).

    Direct port of ``singleXty`` / ``tensorXty`` (discrete.c:329-373).
    Singleton: ``temp = bincount(k_q, wy)``; ``Xty += Xd.T @ temp``,
    accumulated per q. Tensor: form the d-D scatter-summed weight
    tensor ``W̄`` from ``(k_1_q, …, k_d_q)`` then einsum it against
    every marginal ``Xd_j`` to land in the (p_1×…×p_d) coefficient
    space — equivalent to mgcv's per-pre-final-col extraction +
    ``singleXty(M_d, work, …)`` but vectorised over all pre-final
    columns at once.
    """
    if term.kind == "param":
        return term.Xd_list[0].T @ wy

    if term.kind == "single":
        Xd = term.Xd_list[0]
        m = Xd.shape[0]
        ks_lo, ks_hi = term.k_cols[0]
        temp = np.bincount(k[:, ks_lo], weights=wy, minlength=m)
        for q in range(ks_lo + 1, ks_hi):
            temp += np.bincount(k[:, q], weights=wy, minlength=m)
        return Xd.T @ temp

    return _tensor_Xty_raw(term, wy, k, n)


def _tensor_Xty_raw(term: _DiscreteTerm, wy: np.ndarray,
                    k: np.ndarray, n: int) -> np.ndarray:
    Xd_list = term.Xd_list
    d = len(Xd_list)
    ms = tuple(Xd.shape[0] for Xd in Xd_list)
    M = int(np.prod(ms))

    ks_lo_list = [term.k_cols[j][0] for j in range(d)]
    n_sum = term.k_cols[0][1] - term.k_cols[0][0]

    # W̄[g_1,…,g_d] = Σ_{rows with all k_j_q[row]=g_j} wy[row]
    W_flat = np.zeros(M, dtype=float)
    for q in range(n_sum):
        flat_idx = np.zeros(n, dtype=np.int64)
        stride = 1
        for j in range(d - 1, -1, -1):
            flat_idx = flat_idx + k[:, ks_lo_list[j] + q] * stride
            stride *= ms[j]
        W_flat += np.bincount(flat_idx, weights=wy, minlength=M)
    W_total = W_flat.reshape(ms)

    if d == 2:
        result = np.einsum('ab,ai,bj->ij',
                           W_total, Xd_list[0], Xd_list[1])
    elif d == 3:
        result = np.einsum('abc,ai,bj,ck->ijk',
                           W_total, Xd_list[0], Xd_list[1], Xd_list[2])
    else:
        in_letters = "abcdefghij"[:d]
        out_letters = "ABCDEFGHIJ"[:d]
        operand_subs = [in_letters] + [in_letters[j] + out_letters[j]
                                        for j in range(d)]
        expr = ",".join(operand_subs) + "->" + out_letters
        result = np.einsum(expr, W_total, *Xd_list)
    return result.reshape(-1)


def _term_pair_XWX_raw(term_a: _DiscreteTerm, term_b: _DiscreteTerm,
                       w: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    """Compute the unconstrained ``X_a_raw.T @ diag(w) @ X_b_raw`` block
    of shape ``(p_a_raw, p_b_raw)``.

    Direct port of mgcv ``XWXijs`` (discrete.c:1073-1430) for the
    ``acc_w == 1`` branch (n > m_im·m_jm — by far the common case for
    our matrix-arg models). The ``r``/``c`` sub-block decomposition is
    fused: we build the (n × p_a_pre × p_b_pre) outer-product weight
    tensor, scatter into a (m_a_final · m_b_final, p_a_pre · p_b_pre)
    grouped table, then contract with both final-marginal bases via a
    single einsum — equivalent to the per-(r,c) loop but vectorised.
    """
    if term_a.kind == "param" and term_b.kind == "param":
        Xa = term_a.Xd_list[0]
        Xb = term_b.Xd_list[0]
        return Xa.T @ (w[:, None] * Xb)

    if term_a.kind == "param":
        Xa = term_a.Xd_list[0]
        p_a = Xa.shape[1]
        p_b_raw = int(np.prod([Xd.shape[1] for Xd in term_b.Xd_list]))
        result = np.empty((p_a, p_b_raw), dtype=float)
        for i in range(p_a):
            result[i, :] = _term_Xty_raw(term_b, w * Xa[:, i], k, n)
        return result

    if term_b.kind == "param":
        return _term_pair_XWX_raw(term_b, term_a, w, k, n).T

    if term_a.kind == "single" and term_b.kind == "single":
        return _single_single_XWX(term_a, term_b, w, k, n)

    return _general_XWX(term_a, term_b, w, k, n)


def _single_single_XWX(term_a: _DiscreteTerm, term_b: _DiscreteTerm,
                       w: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    Xd_a = term_a.Xd_list[0]
    Xd_b = term_b.Xd_list[0]
    m_a = Xd_a.shape[0]
    m_b = Xd_b.shape[0]
    ks_a_lo, ks_a_hi = term_a.k_cols[0]
    ks_b_lo, ks_b_hi = term_b.k_cols[0]

    W_total = np.zeros((m_a, m_b), dtype=float)
    for q_a in range(ks_a_hi - ks_a_lo):
        k_a = k[:, ks_a_lo + q_a]
        for q_b in range(ks_b_hi - ks_b_lo):
            k_b = k[:, ks_b_lo + q_b]
            flat_idx = k_a * m_b + k_b
            W_flat = np.bincount(flat_idx, weights=w, minlength=m_a * m_b)
            W_total += W_flat.reshape(m_a, m_b)
    return Xd_a.T @ W_total @ Xd_b


def _row_tensor_pre(Xd_list: list[np.ndarray],
                    k_per_marg: list[np.ndarray],
                    da: int, n: int) -> np.ndarray:
    """Row-tensor of the *pre-final* marginals at this q's indices.

    Returns shape ``(n, p_pre)`` where ``p_pre = ∏_{j<da-1} p_j``.
    For singleton terms (da == 1) returns ``(n, 1)`` ones — caller
    treats it as the empty pre-tensor.
    """
    if da == 1:
        return np.ones((n, 1), dtype=float)
    if da == 2:
        return Xd_list[0][k_per_marg[0]]
    if da == 3:
        X1 = Xd_list[0][k_per_marg[0]]                          # (n, p1)
        X2 = Xd_list[1][k_per_marg[1]]                          # (n, p2)
        return (X1[:, :, None] * X2[:, None, :]).reshape(n, -1)
    blocks = [Xd_list[j][k_per_marg[j]] for j in range(da - 1)]
    out = blocks[0]
    for jj in range(1, da - 1):
        out = (out[:, :, None] * blocks[jj][:, None, :]).reshape(n, -1)
    return out


def _general_XWX(term_a: _DiscreteTerm, term_b: _DiscreteTerm,
                 w: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    Xd_a_list = term_a.Xd_list
    Xd_b_list = term_b.Xd_list
    da = len(Xd_a_list)
    db = len(Xd_b_list)
    pa = [Xd.shape[1] for Xd in Xd_a_list]
    pb = [Xd.shape[1] for Xd in Xd_b_list]
    p_a_raw = int(np.prod(pa))
    p_b_raw = int(np.prod(pb))
    p_a_pre = int(np.prod(pa[:-1])) if da > 1 else 1
    p_b_pre = int(np.prod(pb[:-1])) if db > 1 else 1
    p_a_final = pa[-1]
    p_b_final = pb[-1]
    Xd_a_final = Xd_a_list[-1]
    Xd_b_final = Xd_b_list[-1]
    m_a_final = Xd_a_final.shape[0]
    m_b_final = Xd_b_final.shape[0]

    ks_a_lo = [term_a.k_cols[j][0] for j in range(da)]
    ks_b_lo = [term_b.k_cols[j][0] for j in range(db)]
    n_sum_a = term_a.k_cols[0][1] - term_a.k_cols[0][0]
    n_sum_b = term_b.k_cols[0][1] - term_b.k_cols[0][0]

    XWX_block = np.zeros((p_a_pre, p_a_final, p_b_pre, p_b_final),
                          dtype=float)

    for q_a in range(n_sum_a):
        k_a_per = [k[:, ks_a_lo[j] + q_a] for j in range(da)]
        Xa_pre = _row_tensor_pre(Xd_a_list, k_a_per, da, n)    # (n, p_a_pre)
        k_a_final = k_a_per[-1]

        for q_b in range(n_sum_b):
            k_b_per = [k[:, ks_b_lo[j] + q_b] for j in range(db)]
            Xb_pre = _row_tensor_pre(Xd_b_list, k_b_per, db, n)
            k_b_final = k_b_per[-1]

            # Outer-product weight: F[row, r, c] = w[row] · Xa_pre[row,r] · Xb_pre[row,c]
            F = (w[:, None, None]
                 * Xa_pre[:, :, None] * Xb_pre[:, None, :]
                 ).reshape(n, p_a_pre * p_b_pre)

            joint_idx = (k_a_final.astype(np.int64) * m_b_final
                          + k_b_final.astype(np.int64))
            W_flat = _grouped_sum_axis0(joint_idx, F,
                                         m_a_final * m_b_final)
            W_grouped = W_flat.reshape(m_a_final, m_b_final,
                                        p_a_pre, p_b_pre)

            # Out[r, la, c, lb] = Σ_{ga, gb} Xd_af[ga,la] · W̄[ga,gb,r,c] · Xd_bf[gb,lb]
            XWX_block += np.einsum('Aa,ABrc,Bb->racb',
                                    Xd_a_final, W_grouped, Xd_b_final)

    return XWX_block.reshape(p_a_raw, p_b_raw)


# ---------------------------------------------------------------------------
# Public kernels
# ---------------------------------------------------------------------------


def Xbd(design: DiscreteDesign, beta: np.ndarray,
        *, X: Optional[np.ndarray] = None,
        use_kernel: bool = False) -> np.ndarray:
    """Compute ``X β`` on the compressed design.

    Direct port of mgcv ``Xbd`` (misc.r:385 + src/discrete.c:474-557).
    Default path uses the cached full-X matmul (set up by
    :func:`discrete_full_X`); set ``use_kernel=True`` to instead route
    through the per-term scatter-add kernels (lift
    ``β_post → β_raw = T·β_post``, then ``_term_Xb_raw`` accumulates
    into the n-vector η). The ``X=`` argument lets callers pass an
    explicit X matrix — used mainly as a verification oracle.
    """
    beta = np.asarray(beta, dtype=float)
    if X is not None:
        return X @ beta
    if not use_kernel:
        return discrete_full_X(design) @ beta
    n = design.n
    eta = np.zeros(n, dtype=float)
    Ts = _design_constraint_Ts(design)
    for term, T in zip(design.terms, Ts):
        b_post = beta[term.coef_slice]
        b_raw = b_post if T is None else (T @ b_post)
        eta += _term_Xb_raw(term, b_raw, design.k, n)
    return eta


def XWXd(design: DiscreteDesign, w: np.ndarray,
         *, X: Optional[np.ndarray] = None,
         use_kernel: bool = False) -> np.ndarray:
    """Compute ``X' diag(w) X`` on the compressed design.

    Direct port of mgcv ``XWXd0`` (discrete.c:1457 driver around
    ``XWXijs``). Default path uses the cached full-X (single
    ``X' diag(w) X`` BLAS gemm); set ``use_kernel=True`` to route
    through per-term-pair scatter-add blocks (each block computed via
    :func:`_term_pair_XWX_raw` in the unconstrained raw space, then
    sandwiched ``T_a' (·) T_b`` to land in the post-constraint Gram).
    The full Gram is symmetric so kernel mode only computes
    upper-triangular blocks and mirrors.
    """
    w_arr = np.asarray(w, dtype=float)
    if X is None and not use_kernel:
        X = discrete_full_X(design)
    if X is not None:
        # Signed-weight-safe form. ``sqrt(w)·X then Xw'·Xw`` is faster
        # for non-negative ``w`` but NaN's on negative entries — and
        # extended families (Scat etc) routinely produce per-row
        # negative Newton-Hessian weights ``Deta2/2``. The direct
        # ``X' (w·X)`` is the same FLOP count and handles any sign.
        return X.T @ (w_arr[:, None] * X)

    n = design.n
    p = design.p
    XWX = np.zeros((p, p), dtype=float)
    Ts = _design_constraint_Ts(design)
    terms = design.terms

    for i, term_i in enumerate(terms):
        T_i = Ts[i]
        s_i = term_i.coef_slice
        block_raw = _term_pair_XWX_raw(term_i, term_i, w_arr,
                                        design.k, n)
        if T_i is None:
            block_post = block_raw
        else:
            block_post = T_i.T @ block_raw @ T_i
        XWX[s_i, s_i] = block_post

        for j in range(i + 1, len(terms)):
            term_j = terms[j]
            T_j = Ts[j]
            s_j = term_j.coef_slice
            block_raw = _term_pair_XWX_raw(term_i, term_j, w_arr,
                                            design.k, n)
            if T_i is None and T_j is None:
                block_post = block_raw
            elif T_i is None:
                block_post = block_raw @ T_j
            elif T_j is None:
                block_post = T_i.T @ block_raw
            else:
                block_post = T_i.T @ block_raw @ T_j
            XWX[s_i, s_j] = block_post
            XWX[s_j, s_i] = block_post.T
    return XWX


def XWyd(design: DiscreteDesign, w: np.ndarray, y: np.ndarray,
         *, X: Optional[np.ndarray] = None,
         use_kernel: bool = False) -> np.ndarray:
    """Compute ``X' (w · y)`` on the compressed design.

    Direct port of mgcv ``XWyd`` (misc.r:333 + src/discrete.c:717-802).
    Default path uses the cached full-X (single ``X' · (w·y)`` BLAS
    gemv); set ``use_kernel=True`` to scatter-add ``w·y`` into the
    m-d-grouped weight tensor per term, then contract against every
    marginal ``Xd`` to land in raw coefficient space, then apply ``T'``
    to land in the post-constraint slot.
    """
    w_arr = np.asarray(w, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if X is None and not use_kernel:
        X = discrete_full_X(design)
    if X is not None:
        return X.T @ (w_arr * y_arr)

    n = design.n
    p = design.p
    wy = w_arr * y_arr
    Xy = np.zeros(p, dtype=float)
    Ts = _design_constraint_Ts(design)
    for term, T in zip(design.terms, Ts):
        Xty_raw = _term_Xty_raw(term, wy, design.k, n)
        if T is None:
            Xy[term.coef_slice] = Xty_raw
        else:
            Xy[term.coef_slice] = T.T @ Xty_raw
    return Xy
