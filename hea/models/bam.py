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

mgcv source: ``ref/mgcv/R/bam.r`` (1.9-4).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve, qr as scipy_qr, solve_triangular
from scipy.linalg.lapack import dpstrf
from ..R import distributions as _dist
from ..R._shared import _rfma_vec

from ..family import Family, Gaussian, _coerce_response
from ..formula import (
    BasisSpec,
    SmoothBlock,
    _apply_smooth_arg_exprs,
    _eval_atom,
    _factor_levels,
    _LinearTransformRawBasis,
    _RawBasis,
    _smooth_arg_expr_map,
    _smooth_id_value,
    _smooth_sp_value,
    _T2PredictRawBasis,
    _T2RawBasis,
    _TensorRawBasis,
    is_matrix_col,
    materialize,
    materialize_smooths,
    matrix_to_2d,
    normalize_data,
    prepare_design,
)
from .gam import (
    _FitState,
    _PenaltySlot,
    _R_rank,
    _Sl,
    _add_factor_stub_rows,
    _add_null_space_penalties,
    _apply_gam_side,
    _block_s_scale,
    _row_frame,
    _sl_initial_repara,
    _sl_mult,
    _sl_setup,
    _sl_term_mult,
    _sym_rank,
    gam,
)
# Safe at module level: nothing on the hea.R.__init__ chain
# (model_selection → models.gam → formula) imports this module.
from ..R.rng import RMersenneTwister
from .._dispatch import rs_fn

__all__ = ["bam"]

# Rust accelerator for the discrete X'WX smooth×smooth raw block (mgcv XWXijs).
# ``None`` when the extension is unavailable — the numpy oracle then runs.
_rs_xwx_smooth_block = rs_fn("xwx_smooth_block")

# Rust accelerator for ``rwMatrix`` (mgcv misc.c:710-748) — the AR1 row-recombine.
# ``None`` when unavailable — :func:`_rw_matrix` then runs the numpy fallback.
_rs_rw_matrix = rs_fn("rw_matrix")

# Rust fixed-order dense matmul for the REML-Hessian cross products (mgcv
# ``mgcv_pmmult2``→``dgemm``). numpy's ``@`` runs those through a threaded BLAS
# whose reduction order is not pinned, so the fREML Hessian (hence the converged
# ``rho``/fit) wobbles ~1 ULP per run — hea's ``discrete=FALSE`` run-to-run
# nondeterminism (measured: mgcv on a single-thread BLAS is bit-stable, numpy is
# not). The kernel sums each dot product strictly in index order → deterministic
# and cross-platform. ``None`` when unavailable — the einsum fallback below is
# also fixed-order (numpy's in-order C loop, no BLAS reorder).
_rs_reml_pmmult = rs_fn("reml_pmmult")


def _pmmult(a: np.ndarray, b: np.ndarray,
            at: bool = False, bt: bool = False) -> np.ndarray:
    """``op(a) @ op(b)`` (``op`` = transpose if the flag is set) with a FIXED
    reduction order — mgcv ``mgcv_mmult`` (mat.c:431) semantics, but pinned so
    the REML Hessian is deterministic. Routes to the Rust kernel
    (:data:`_rs_reml_pmmult`); the fallback is ``einsum(optimize=False)`` —
    numpy's own in-order loop, NOT the threaded-BLAS ``@`` — so the Python path
    is deterministic too."""
    if _rs_reml_pmmult is not None:
        return np.asarray(_rs_reml_pmmult(
            np.ascontiguousarray(a, dtype=float),
            np.ascontiguousarray(b, dtype=float), at, bt))
    # Fallback: accumulate the SAME left-fold over k as the Rust kernel, via
    # in-order rank-1 updates (C += a[:,k] ⊗ b[k,:]). Separate multiply/add, no
    # BLAS reorder, k strictly ascending → bit-identical to the Rust path.
    aa = np.ascontiguousarray(a.T if at else a, dtype=float)
    bb = np.ascontiguousarray(b.T if bt else b, dtype=float)
    m, k = aa.shape
    n = bb.shape[1]
    out = np.zeros((m, n), dtype=float)
    for kk in range(k):
        out += aa[:, kk, None] * bb[None, kk, :]
    return out


# ---------------------------------------------------------------------------
# Utility ports — mgcv bam.r:1-200
# ---------------------------------------------------------------------------


def _rw_matrix(stop: np.ndarray, row: np.ndarray, weight: np.ndarray,
               X: np.ndarray, trans: bool = False) -> np.ndarray:
    """Recombine rows of ``X`` per ``stop``/``row``/``weight``.

    Direct port of mgcv ``rwMatrix`` (C kernel src/misc.c:710-748; R wrapper
    bam.r:18-29). Forward (``trans=False``): the ith output row is
    ``Σ_{k ∈ ind_i} weight[k] · X[row[k], :]`` where ``ind_i = 1:stop[1]`` if
    ``i==1`` else ``(stop[i-1]+1):stop[i]``. Transposed (``trans=True``):
    ``out[row[k], :] += weight[k] · X[i, :]`` over the same (i, k) pairs — the
    scatter adjoint. BOTH are exercised by the AR1 X'Wy / y-norm² path
    (:func:`XWyd`, discrete.c:1152-1156 calls rwMatrix forward then transpose).

    FMA: mgcv's C accumulates ``*X1p += weight * *Xp`` in one expression, which
    fuses to ``fma`` on arm64. The rust kernel (:data:`_rs_rw_matrix`, the
    primary path) mirrors that with ``rfma`` and is 0-ULP to live arm64 R
    (verified vs ``mgcv:::rwMatrix``). The numpy fallback below (used only when
    the extension is unavailable) pre-rounds the ``weight·X`` products before
    summing (reduceat / scatter-add), so it diverges from the fused C by ≤1 ULP
    per accumulation — but that is far BELOW the downstream Accelerate ``dgemm``
    reduction floor (~1e-13) the discrete contraction sits at, so it does not
    move any end-to-end result.

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
    if _rs_rw_matrix is not None:
        # Native single-pass port (misc.c:710-748) with arm64 ``fma`` — 0-ULP to
        # live R and avoids the fallback's ``np.add.at`` scatter on the X'Wy path.
        out = _rs_rw_matrix(
            np.ascontiguousarray(stop, dtype=np.int64),
            np.ascontiguousarray(row, dtype=np.int64),
            np.ascontiguousarray(weight, dtype=np.float64),
            np.ascontiguousarray(X, dtype=np.float64),
            bool(trans),
        )
        return out.ravel() if not is_matrix else out
    if trans:
        # Vectorised scatter-add (mgcv rwMatrix transpose, misc.c:734-742).
        # ``i_of_k`` maps each input index k to the output segment i it falls
        # in; ``np.add.at`` then scatters ``weight[k]·X[i_of_k]`` into
        # ``out[row[k]]`` IN k-ASCENDING ORDER — exactly mgcv's i-outer/k-inner
        # accumulation order — so it is bit-identical to the scalar reference
        # (the AR1 bidiagonal Tᵀ hits each target with ≤2 sources, and FP
        # addition commutes for two terms). ~260× faster than the prior Python
        # double-loop on big-n (the X'Wy hot path: 970ms→3.7ms at n=5e5).
        if n == 0:
            out = np.zeros((0, p), dtype=float)
            return out.ravel() if not is_matrix else out
        K = int(stop[-1]) + 1
        starts = np.empty(n, dtype=np.intp)
        starts[0] = 0
        starts[1:] = stop[:-1] + 1
        lengths = np.maximum((stop + 1) - starts, 0)
        i_of_k = np.repeat(np.arange(n), lengths)
        out = np.zeros((n, p), dtype=float)
        np.add.at(out, row[:K], weight[:K, None] * X[i_of_k, :])
        return out.ravel() if not is_matrix else out
    if n == 0:
        out = np.zeros((0, p), dtype=float)
        return out.ravel() if not is_matrix else out
    # Vectorized segmented reduction: output row i sums
    # ``weight[k] · X[row[k], :]`` over k in ``(stop[i-1]+1):stop[i]``
    # (with ``stop[-1] := -1`` for the first row). ``np.add.reduceat`` sums
    # each segment left-to-right; bit-identical to a NON-fused scalar loop
    # (it pre-rounds ``weight·X`` like the scatter path above — the ≤1-ULP
    # FMA gap vs mgcv's fused C is sub-floor, see the docstring).
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


def _sl_rsb(sl: _Sl, rho_full: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """mgcv ``Sl.rSb`` (fast-REML.r:453-482): stack every penalty's ``rS·β``
    end to end. ``sum(a²)`` is the penalty ``βᵀSλβ`` — but as the per-block
    root reduction the discrete/REML convergence test reads (``sum(rSb²)``,
    bgam.fitd:591/611), NOT a full-matrix quadratic form ``coef·Sλ·coef`` (a
    different FP reduction order). After ``Sl.setup``'s initial repara each
    singleton penalty is a multiple of identity on its ``ind`` columns, so its
    root is ``exp(ρ_k/2)·β[ind]``; a multi-S block contributes
    ``exp(ρ_k/2)·(β[ind] · rS[j])`` per term.

    ``beta`` is in the INITIAL-REPARA gauge (mgcv ``prop$beta``); ``rho_full``
    is the full per-penalty log-sp (``lsp.full``), ordered to match
    ``sl.blocks`` (so ``rho_full[k]`` is the k-th penalty's log-sp, exactly the
    order :func:`_sl_sb` / ``_build_S_lambda`` consume).
    """
    parts: list[np.ndarray] = []
    k = 0
    for blk in sl.blocks:
        ind = blk.pen_cols()                       # (start:stop)[ind], absolute
        if blk.n_sp == 1:                          # singleton — multiple of I
            parts.append(beta[ind] * np.exp(rho_full[k] / 2.0))
            k += 1
        else:                                      # multi-S block
            for j in range(blk.n_sp):
                parts.append(np.exp(rho_full[k] / 2.0)
                             * (beta[ind] @ blk.rS[j]))
                k += 1
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)


def _sl_sb(sl: _Sl, rho_full: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """mgcv ``Sl.Sb`` (fast-REML.r:431-451): ``S·β`` where ``S`` is the total
    penalty in the INITIAL-REPARA gauge. ``sum(β·Sl.Sb(β))`` is ``βᵀSλβ`` — the
    quadratic form mgcv's NON-discrete ``bgam.fit`` step-halving reads
    (bam.r:1171-1179), as opposed to ``sum(rSb²)`` (the root reduction
    ``bgam.fitd`` reads). Mathematically equal, different FP reduction.

    After ``Sl.setup``'s initial repara a singleton penalty is a multiple of
    identity on its ``ind`` columns ⇒ ``β[ind]·exp(ρ_k)``; a multi-S block
    contributes ``Σ_j exp(ρ_k)·(S[j]·β[ind])``. ``beta`` is in the initial-repara
    gauge (``prop$beta``); ``rho_full`` is the full per-penalty log-sp.
    """
    a = np.zeros_like(beta, dtype=float)
    k = 0
    for blk in sl.blocks:
        ind = blk.pen_cols()                       # (start:stop)[ind], absolute
        if blk.n_sp == 1:                          # singleton — multiple of I
            a[ind] = a[ind] + beta[ind] * np.exp(rho_full[k])
            k += 1
        else:                                      # multi-S block
            for j in range(blk.n_sp):
                a[ind] = a[ind] + np.exp(rho_full[k]) * (blk.S[j] @ beta[ind])
                k += 1
    return a


def _sl_initial_repara_ldet_const(sl: _Sl) -> float:
    """``Σ_pen log λ`` — the ρ-independent gauge shift the non-orthogonal
    ``Sl.setup`` transforms fold into ``log|Sλ|_+``.

    bam evaluates ``log|Sλ|_+`` in the ORIGINAL gauge (``_log_det_S_pos``)
    but :func:`_pi_fit_chol` returns ``ldetXXS`` in the reparameterised gauge
    (its gram is ``D'(X'WX)D`` and its penalties are the repara'd
    identities/projections). The REML score is ``ldetXXS − ldetS``; under the
    block transform ``X→XD`` both terms pick up the SAME ρ-independent
    ``2·log|D|_pen``, so subtracting this constant from the original-gauge
    ``ldetS`` realigns the score VALUE while leaving the
    congruence-invariant ρ-grad/Hessian untouched.

    Per repara block the shift is ``−2·Σ_j log‖D[:,j]‖``: a diagonal/eigen
    singleton has ``‖D[:,j]‖ = 1/√λ_j`` on penalised columns (and 1
    elsewhere, contributing 0), so it sums to ``Σ log λ_j``; an orthogonal
    multi-S ``D`` has unit columns → 0. This recovers exactly the value the
    old ``_repara_ldet_const`` computed, read off gam's ``Sl`` blocks (so a
    block ``Sl.setup`` split into singletons is accounted for correctly —
    its now-nonzero shift cancels the matching shift in ``_pi_fit_chol``'s
    ``ldetXXS``).
    """
    total = 0.0
    for blk in sl.blocks:
        if not blk.repara:
            continue
        D = blk.D
        if D.ndim == 1:
            colnorm = np.abs(D)
        else:
            colnorm = np.sqrt(np.einsum("ij,ij->j", D, D))
        total += -2.0 * float(np.sum(np.log(colnorm)))
    return total


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
    # mgcv efam.r:48 ``n.theta <- length(theta)`` — the count of *passed*
    # (fixed) θ entries, BEFORE the scale slot is appended. This is mgcv's
    # ``del.ind = 1:n.theta``, dropped from g/H when ``family$n.theta==0``.
    n_passed = int(theta.shape[0])
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

    def _nlogl(theta_eval: np.ndarray, deriv: int):
        # mgcv R/efam.r:14-45 verbatim. ``theta_eval`` may include a
        # trailing log φ slot when scale<0; strip it for the family
        # calls and re-add the φ-direction gradient / Hessian rows.
        # mgcv efam.r:16 ``nth <- length(theta) - if (scale<0) 1 else 0`` —
        # the count of FAMILY θ entries actually passed to ``Dd``/``ls``
        # (= len(theta_eval) minus the appended scale slot). NOT
        # ``family$n.theta``: those coincide for tw/nb/scat (the families
        # bam fits, all free-θ) but DIVERGE when the family has fixed θ it
        # reports as ``n.theta==0`` (then nth = n_passed, not 0).
        nth = int(theta_eval.shape[0]) - (1 if scale < 0 else 0)
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
        # mgcv efam.r:53-54: when family$n.theta==0 the optimization is over
        # the appended scale param ONLY — drop the passed (fixed) θ via
        # ``g[-del.ind]`` / ``H[-del.ind,-del.ind]`` (del.ind = 1:length(theta)).
        g = g[n_passed:]
        H = H[n_passed:, n_passed:]
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
                # mgcv efam.r:67 ``step0 <- c(rep(0,n.theta),step0)`` where R's
                # ``n.theta`` is the PASSED count (``n_passed``), NOT the
                # family's ``n.theta==0``. Pads the dropped fixed-θ slots back
                # with zeros so the optimisation moves only the appended scale.
                step0 = np.concatenate([np.zeros(n_passed), step0])
            ms = float(np.max(np.abs(step0)))
            if ms > 4.0:
                step0 = step0 * 4.0 / ms
            step = np.zeros_like(theta)
            if n_theta == 0:
                # mgcv efam.r:71 ``step[uconv] <- step0``: after the del.ind
                # reduction g/H/uconv cover only the appended scale slot
                # (length len(theta)-n_passed == 1 for scale<0), and R recycles
                # that length-1 logical mask across the full-length ``step``.
                # numpy has no recycling — but with the n_passed zero-prepend
                # ``step0`` is already full length and the recycled mask is
                # all-TRUE (we only enter the loop when any(uconv)), so the
                # assignment is just ``step <- step0``.
                step[:] = step0
            else:
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
                g = g[n_passed:]
                H = H[n_passed:, n_passed:]
            uconv = np.abs(g) > tol * (abs(nll) + 1.0)
            if not np.any(uconv):
                break

    if step_failed:
        import warnings
        warnings.warn("step failure in theta estimation", stacklevel=2)
    return theta


def _sl_add_s(sl: _Sl, A: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """mgcv ``Sl.addS`` (fast-REML.r:1016-1039): add the total penalty
    ``Sλ`` to ``A``, returning a copy (mgcv forces ``A <- A*1``).

    Operates in the INITIAL-REPARA gauge, so each block contributes its
    reparameterised penalty, exactly as mgcv:

      * singleton (``n_sp==1``) → a multiple of identity on its penalised
        columns: ``diag(A)[ind] += exp(rho_k)`` (mgcv ``mgcv_madi`` with
        ``diag=-1``), ``ind = (start:stop)[blk.ind]``;
      * multi-S block → ``A[ind,ind] += Σ_j exp(rho_k) S_j`` added one
        penalty at a time (mgcv loops ``mgcv_madi`` per ``S[[j]]``), over
        the leading rank columns ``ind``.

    ``rho`` is the per-penalty log-sp in ``sl.blocks`` order (the order
    :func:`_sl_sb` / ``_build_S_lambda`` consume).
    """
    A = A * 1.0                                    # force copy (Sl.addS:1021)
    sp = np.exp(np.asarray(rho, dtype=float))
    k = 0
    for blk in sl.blocks:
        ind = blk.pen_cols()                       # (start:stop)[ind]
        if blk.n_sp == 1:                          # singleton — multiple of I
            A[ind, ind] += sp[k]
            k += 1
        else:                                      # multi-S block
            for j in range(blk.n_sp):
                A[np.ix_(ind, ind)] += sp[k] * blk.S[j]
                k += 1
    return A


def _sl_chol_lambda(sl: _Sl, rho: np.ndarray) -> None:
    """mgcv ``ldetS(repara=FALSE)`` λ/St update used by ``Sl.fitChol``
    (fast-REML.r:1598): set each block's current smoothing parameters and,
    for multi-S blocks, the block total ``St = Σ_i exp(ρ_i) S_i`` — in the
    INITIAL-REPARA (Sl.setup-projected) gauge, WITHOUT the gam.reparam
    stability transform that the QR-path ``_ldet_s`` applies. ``Sl.fitChol``
    operates directly on the initial-repara'd gram, so the penalty must be
    applied in that gauge.

    Mutates ``sl`` in place (mgcv updates ``ldS$Sl``); the downstream
    structured products :func:`_sl_mult` / :func:`_sl_term_mult` then read
    ``blk.lam`` / ``blk.St``. ``blk.Srp`` is reset to ``None`` so
    ``_sl_term_mult`` uses the un-transformed per-penalty form
    ``lam_i·(S_i·A)``. ``rho`` is the per-penalty log-sp in ``sl.blocks``
    order.
    """
    sp = np.exp(np.asarray(rho, dtype=float))
    k = 0
    for blk in sl.blocks:
        if blk.n_sp == 1:                          # singleton — multiple of I
            blk.lam = np.array([sp[k]])
            k += 1
        else:                                      # multi-S — pre-summed St
            m = blk.n_sp
            blk.lam = sp[k:k + m].copy()
            St = sp[k] * blk.S[0]
            for j in range(1, m):
                St = St + sp[k + j] * blk.S[j]
            blk.St = St
            blk.Srp = None
            k += m


def _d_det_xxs(sl: _Sl, PP: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mgcv ``d.detXXS`` (fast-REML.r:1329-1367): first/second ρ-derivatives
    of ``log|X'X+Sλ|`` given the (unpivoted) inverse-Hessian ``PP = A⁻¹``.

    With ``SPP[[k]] = Sl.termMult(PP)`` (stripped to ``ind`` rows):

        d1[k]   =  Σ diag(SPP[[k]][, ind_k])                 (mgcv :1347)
        d2[i,j] = -Σ (SPP[[i]][, ind_j])ᵀ · SPP[[j]][, ind_i] (mgcv :1354)
        d2[i,i] += d1[i]     (linear-term correction, mgcv :1362)

    All blocks bam builds are linear, so the non-linear ``AdS`` branch and
    the sparse-``Matrix`` guard are omitted. Reads the current ``blk.lam`` /
    ``blk.St`` set by :func:`_sl_chol_lambda`.
    """
    SA, inds = _sl_term_mult(sl, PP, full=False)
    nd = len(SA)
    d1 = np.zeros(nd)
    d2 = np.zeros((nd, nd))
    for i in range(nd):
        indi = inds[i]
        d1[i] = float(np.trace(SA[i][:, indi]))
        for j in range(i, nd):
            indj = inds[j]
            v = -float(np.sum(SA[i][:, indj].T * SA[j][:, indi]))
            d2[i, j] = d2[j, i] = v
        d2[i, i] += d1[i]                          # nli[2,i]==0 (linear)
    return d1, d2


def _sl_ift_chol(sl: _Sl, XX: np.ndarray, R_pre: np.ndarray, d: np.ndarray,
                 beta: np.ndarray, piv: np.ndarray, ipiv: np.ndarray,
                 rank_A: int, p: int) -> dict:
    """mgcv ``Sl.iftChol`` (fast-REML.r:1405-1488): derivatives of β̂ by
    implicit differentiation, and from them ``b'Sb`` and RSS derivatives.

    ``R_pre``/``d``/``piv``/``ipiv``/``rank_A`` are the diagonally
    preconditioned pivoted Cholesky factor of the penalised Hessian and its
    pivots (mgcv's ``R``, ``d``, ``piv``). The "all-in-one" mgcv path
    (:1453-1487) is followed term for term:

        D       = unlist(Sl.termMult(β, full=TRUE))   cols are dSλ/dρ_j · β
        bSb1    = colSums(β·D)
        db[piv] = -backsolve(R, forwardsolve(Rᵀ, (D/d)[piv])) / d
        S.db    = Sl.mult(db)
        bSb2    = diag(bSb1) + 2(dbᵀ(D+S.db) + Dᵀdb)
        XX.db   = XX·db ;  rss2 = 2·dbᵀ·XX.db

    The two triangular solves are mgcv ``mgcv_Rpforwardsolve`` /
    ``mgcv_Rpbacksolve`` (``dtrsm`` wrappers); the cross products are mgcv
    ``mgcv_pmmult2`` (``dgemm``) — so ``solve_triangular`` / ``@`` are the
    faithful ports. Reads the current ``blk.lam`` / ``blk.St`` set by
    :func:`_sl_chol_lambda`.
    """
    SA, _inds = _sl_term_mult(sl, beta, full=True)
    nd = len(SA)
    D = np.zeros((p, nd))
    for kk in range(nd):
        D[:, kk] = SA[kk]
    bSb1 = np.einsum("i,ik->k", beta, D)           # colSums(beta*D)

    db = np.zeros((p, nd))
    if rank_A > 0 and nd > 0:
        D_pre = (D / d[:, None])[piv, :]           # (D/d)[piv,]
        w_top = -solve_triangular(
            R_pre[:rank_A, :rank_A],
            solve_triangular(
                R_pre[:rank_A, :rank_A].T, D_pre[:rank_A, :], lower=True,
            ),
            lower=False,
        )
        db_piv = np.zeros((p, nd))
        db_piv[:rank_A, :] = w_top
        db = db_piv[ipiv, :] / d[:, None]

    S_db = _sl_mult(sl, db)                         # Sl.mult(db, k=0)

    if nd > 0:
        # mgcv pmmult2 cross products, via the fixed-order kernel so the REML
        # Hessian is run/platform deterministic (numpy ``@`` is not — §D1).
        bSb2 = np.diag(bSb1) + 2.0 * (
            _pmmult(db, D + S_db, at=True) + _pmmult(D, db, at=True)
        )
        bSb2 = 0.5 * (bSb2 + bSb2.T)
        XX_db = _pmmult(XX, db)                     # pmmult2(XX, db)
        rss2 = 2.0 * _pmmult(db, XX_db, at=True)    # 2 pmmult2(db, XX.db)
        rss2 = 0.5 * (rss2 + rss2.T)
    else:
        bSb2 = np.zeros((0, 0))
        rss2 = np.zeros((0, 0))
    rss1 = np.zeros(nd)
    return {"db": db, "bSb1": bSb1, "bSb2": bSb2,
            "rss1": rss1, "rss2": rss2}


def _pi_fit_chol(
    XX: np.ndarray, Xy: np.ndarray, rho: np.ndarray,
    sl: _Sl, p: int, *, yy: float = 0.0,
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
    + pivoted Cholesky with ``chol(pivot=TRUE)``'s default rank tolerance
    (``N·eps·max(diag)``, matching mgcv Sl.fitChol). The gradient
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
        sl: the ``_Sl`` block-diagonal penalty (mgcv ``Sl.setup`` output),
            consumed via the structured ``_sl_add_s`` / ``_sl_ift_chol`` /
            ``_d_det_xxs`` ports — one ρ entry per penalty in ``sl.blocks``
            order.
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
    n_sp = sum(blk.n_sp for blk in sl.blocks)

    # 0. mgcv Sl.fitChol:1598 ``ldS <- ldetS(Sl, rho, repara=FALSE)`` — set
    #    each block's current λ and total St in the initial-repara gauge, so
    #    the structured penalty products below read the right λ.
    _sl_chol_lambda(sl, rho)

    # 1. Build A = XX + Sλ via mgcv Sl.addS (identity/block form).
    A = _sl_add_s(sl, XX, rho)
    A = 0.5 * (A + A.T)

    # 2. Diagonal preconditioning: D = sqrt(diag(A)).
    diag_A = np.diag(A).copy()
    d = np.where(diag_A > 0.0, np.sqrt(np.maximum(diag_A, 0.0)), 1.0)
    A_pre = (A / d) / d[:, None]
    A_pre = 0.5 * (A_pre + A_pre.T)

    # 3. Pivoted Cholesky on the preconditioned matrix. mgcv's Sl.fitChol
    #    factorizer is ``chol(A_pre, pivot=TRUE)`` (fast-REML.r:1606) = LAPACK
    #    DPSTRF with tol=-1 → its default ``N·eps·max(diag)`` tolerance. Use
    #    dpstrf's default (NOT gam.fit3's QR-path ``eps·100``, which is a
    #    different routine) so the rank determination matches mgcv's chol.
    A_pre_f = np.asfortranarray(A_pre.copy())
    R_pre, piv_1based, rank_A, _info = dpstrf(A_pre_f, lower=0)
    R_pre = np.triu(R_pre)
    # mgcv Sl.fitChol:1607 ``r <- min(attr(R,"rank"), Rrank(R))`` — take the
    # SMALLER of dpstrf's pivot rank and mgcv's own Cline-condition rank
    # estimate ``Rrank`` (mgcv.r:4, tol=eps^0.9) on the FULL pivoted factor.
    # Full-rank, well-conditioned A: both = p (no change). Rank-deficient or
    # near-singular: ``Rrank`` can drop a leading direction that dpstrf's
    # tolerance kept, so the gauge + ``ldetXXS`` match mgcv. ``_R_rank``
    # starts at p and reduces; the pseudo-det / β / IFT below all key off
    # this ``rank_A`` exactly like mgcv truncates ``R <- R[1:r,1:r]``.
    rank_A = min(int(rank_A),
                 _R_rank(R_pre, tol=float(np.finfo(float).eps) ** 0.9))
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

    # 7-9. β̂ derivatives by IFT + b'Sb / RSS derivatives (mgcv Sl.iftChol,
    #      fast-REML.r:1405). The penalty products (Skb=Sl.termMult,
    #      S.db=Sl.mult) use the block identity/St form; the chol solves and
    #      X'X cross products are dtrsm/dgemm (solve_triangular / @).
    dift = _sl_ift_chol(sl, XX, R_pre, d, beta, piv, ipiv, rank_A, p)
    bSb1 = dift["bSb1"]
    bSb2 = dift["bSb2"]
    rss1 = dift["rss1"]
    rss2 = dift["rss2"]

    # 10. log|X'X+Sλ| derivatives (mgcv d.detXXS, fast-REML.r:1329):
    #     d1[k] = Σ diag(S_k·PP[ind]) ; d2[i,j] = -Σ (S_i·PP)ᵀ·(S_j·PP) over
    #     the cross blocks ; d2[i,i] += d1[i] for linear terms.
    dXXS_d1, dXXS_d2 = _d_det_xxs(sl, PP)

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
    #     rss_bSb = ‖y-Xβ‖²+β'Sβ = yy - β'X'Wz (Sl.fitChol:1646 identity).
    rss_bSb = float(yy - beta @ Xy)
    if not phi_fixed:
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

    # 12b. REML VALUE + unpenalised working RSS (Sl.fit:1714, 1736) for
    #      fast.REML.fit's step-halving + reml.scale. The discrete POI never
    #      needs the value (it step-halves on the gradient), so it's only
    #      meaningful when the caller passes the log|S|_+ VALUE in ``ldet_S``.
    #      This is the Gaussian working-model REML on the reduced (R, f) —
    #      ``(nobs/γ-Mp)·log(2πφ)`` normalisation, NOT any non-Gaussian ls:
    #      ``fast.REML.fit``/``Sl.fit`` treat the linearised (R, f) as Gaussian
    #      (the family lives only in the OUTER PIRLS loop's W, z build).
    reml_value = (
        rss_bSb / (phi * gamma)
        + (n / gamma - Mp) * float(np.log(2.0 * np.pi * phi))
        + Mp * float(np.log(gamma))
        + ldetXXS - ldet_S
    ) / 2.0
    rss_unpen = float(yy - 2.0 * (beta @ Xy) + beta @ XX @ beta)

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
        "reml": reml_value,
        "rss": rss_unpen,
        "ldetXXS": ldetXXS,
        "rank": rank_A,
        "PP": PP,
        "R_pre": R_pre,
        "d": d,
        "piv": piv,
        "ipiv": ipiv,
    }


def _reg_newton_step(grad: np.ndarray, hess: np.ndarray,
                     max_step: float = 4.0) -> np.ndarray:
    """Eigen-regularised Newton step ``-H⁻¹g`` with a ``|step| ≤ max_step``
    cap — the exact formula ``_pi_fit_chol`` uses internally
    (Sl.fitChol:1430-1440), factored out so the discrete-POI optimiser can
    recompute the step in *working* (id-linked) space from the L-contracted
    ``(T'g, T'HT)`` instead of the full per-penalty grad/Hessian
    ``_pi_fit_chol`` returns.

    ``max_step`` is mgcv's per-routine cap: ``4`` in ``Sl.fitChol``
    (fast-REML.r:1438) for the discrete POI, ``maxNstep = 5`` in
    ``fast.REML.fit`` (fast-REML.r:1749, 1818) for the non-discrete
    converge-fully Newton. The eigen-flip-and-floor regularisation
    (negative λ → |λ|, then floor at ``max|λ|·√eps``) is identical in both."""
    if hess.shape[0] == 0:
        return np.zeros(0)
    eig_w, eig_v = np.linalg.eigh(hess)
    eig_w_abs = np.abs(eig_w)
    if eig_w_abs.size > 0:
        me = float(eig_w_abs.max() * float(np.finfo(float).eps) ** 0.5)
        eig_w_clamped = np.where(eig_w_abs < me, me, eig_w_abs)
    else:
        eig_w_clamped = eig_w_abs
    step = -eig_v @ ((eig_v.T @ grad) / eig_w_clamped)
    ms = float(np.max(np.abs(step))) if step.size else 0.0
    if ms > max_step:
        step = step * (max_step / ms)
    return step


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

    The dropped-pivot rows must be left at ZERO (chol2qr:40 ``R[(rank+1):
    p,] <- 0``), NOT padded with an identity block: padding would make
    ``R'R = XX + I`` on the dropped diagonals, breaking the gram identity
    ``R'R = XX`` by exactly 1.0 there and biasing every downstream PIRLS
    solve. mgcv solves only the top-rank subsystem (chol2qr:41); so do we.

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
    # mgcv: rngs <- temp.seed(66), then sample(1:n, chunk.size) and
    # sample(1:n, n) from the continuing stream — bit-exact via the
    # hea.R.rng port so the representative frame (hence knot/eigen
    # setup at n > chunk_size) matches mgcv's.
    rng = RMersenneTwister(seed)
    ind = rng.sample_int(n, chunk_size)
    mf0 = data[ind.tolist()]
    # Stratified sampling for representativeness: place min/max rows for
    # numerics and one row per factor level into the head of mf0. The
    # factor pick is mgcv's ind[fac[ind]==X][1] — the first match in the
    # RANDOMIZED row order, one random row per level, levels in R's
    # order.
    ind_full = rng.sample_int(n, n)
    rows: list[int] = []
    for c in cols:
        s = data[c]
        if _is_factor_col(s):
            arr_perm = s.to_numpy()[ind_full]
            for lvl in _factor_levels(s):
                where = np.flatnonzero(arr_perm == lvl)
                if where.size:
                    rows.append(int(ind_full[where[0]]))
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


def _ar1_tri_weight(w: np.ndarray, ar_weights: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Build the symmetric tridiagonal effective weight ``W_eff = D·Tᵀ·T·D``.

    Direct port of mgcv ``XWXd0`` (src/discrete.c:2143-2156). ``D =
    diag(√w)`` is the √IRLS-weight diagonal and ``T`` the bidiagonal AR1
    whitening transform encoded by ``ar_weights`` (the length-``2n-1``
    ``ar.weight`` array from :func:`_ar1_rwmatrix_indices`: even indices
    ``0,2,…,2n-2`` are the leading diagonals ``t_ii`` — ``ar_weights[0]=1``
    for the un-transformed first row — odd indices ``1,3,…,2n-3`` are the
    sub-diagonals ``t_{i+1,i}``). The product ``(T D)ᵀ(T D)`` is tridiagonal:

    * diagonal   ``w_diag[i] = (t_{i+1,i}² + t_ii²)·d_i²`` (``i<n-1``),
                 ``w_diag[n-1] = t_{n-1,n-1}²·d_{n-1}²``;
    * off-diag   ``w_off[i]  = t_{i+1,i}·t_{i+1,i+1}·d_{i+1}·d_i`` (``i<n-1``),
                 super == sub since ``W_eff`` is symmetric.

    Mirrors the mgcv multiply order — ``ws = ((odd·even)·d₊)·d`` and the
    ``odd²+even²`` sum-of-squares fuses on arm64 (clang single expression,
    :func:`_rfma_vec`) — so the table matches arm64 R up to the downstream
    ``dgemm`` reduction floor.
    """
    w = np.asarray(w, dtype=float)
    aw = np.asarray(ar_weights, dtype=float)
    n = w.shape[0]
    d = np.sqrt(w)                       # mgcv discrete.c:2148  w[i] = sqrt(w[i])
    even = aw[0::2]                      # leading diagonals t_ii   (len n)
    odd = aw[1::2]                       # sub diagonals t_{i+1,i}  (len n-1)
    if n == 1:
        # single observation: only discrete.c:2152 runs (i=0), with its
        # ``w[0] *= w[0]·even·even`` association ``d·((d·even)·even)`` — NOT
        # ``(even·d)²`` (which reassociates the rounding, ≤1 ULP off).
        e0 = float(even[0])
        d0 = float(d[0])
        return np.array([d0 * ((d0 * e0) * e0)]), np.zeros(0, dtype=float)
    # off-diagonal: ws[i] = ((odd[i]·even[i+1])·d[i+1])·d[i]   (discrete.c:2150)
    w_off = odd * even[1:] * d[1:] * d[:-1]
    # diagonal i<n-1: ``w[i] *= (odd²+even²)·w[i]`` ⇒ d·((odd²+even²)·d)
    # (discrete.c:2151); the odd²+even² sum-of-squares fuses fma(odd,odd,even²)
    # on arm64 (verified vs clang -O2 codegen: ``fmul even,even`` then
    # ``fmadd odd,odd,even²``).
    sumsq = (even * even)                # even[i]²  (== last-row diagonal base)
    sumsq[:-1] = _rfma_vec(odd, odd, sumsq[:-1])   # arm64: fma(odd,odd,even²)
    w_diag = d * (sumsq * d)
    # last row i=n-1: discrete.c:2152 ``w[i] *= w[i]·even·even`` associates as
    # d·((d·even)·even) — a DIFFERENT rounding from the general d·(even²·d)
    # (differs ≤1 ULP in ~31% of inputs). Match C's association exactly.
    dl = float(d[-1])
    el = float(even[-1])
    w_diag[-1] = dl * ((dl * el) * el)
    return w_diag, w_off


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
    prior_w: Optional[np.ndarray] = None,
) -> _BamQR:
    """Chunked QR build for the Gaussian-identity (am=TRUE) path.

    Walks ``data`` in chunks of ``chunk_size``, materialises each chunk's
    full design via :func:`_materialize_chunk`, and accumulates ``(R, f,
    ‖z‖²)`` with :func:`_qr_update`. ``z = y − offset``. ``prior_w`` (mgcv
    ``G$w``) scales each row by ``√w`` BEFORE the AR1 transform — mgcv weights
    then ``rwMatrix``-decorrelates (bam.r:654: ``rwMatrix(..., sqrt(w)*z)``);
    ``None`` ≡ all-ones. The resulting ``R'R = X'WX`` and ``y_norm2 =
    Σ wᵢ(yᵢ−offᵢ)²`` are the prior-weighted Gram / working-RSS.

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
        if prior_w is not None:
            # mgcv weights the working data (√w·X, √w·z) before the AR1
            # rwMatrix transform — see bam.r:654.
            sw = np.sqrt(prior_w[start:end])
            X_chunk = sw[:, None] * X_chunk
            z_chunk = sw * z_chunk
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
            # Extended-family Fisher-scoring branch — the NON-discrete
            # serial chunked path mirrors mgcv ``bgam.fit`` (bam.r:1070-1076),
            # which uses the EXPECTED (Fisher) Hessian unconditionally:
            #     w <- dd$EDeta2 * .5
            #     z <- (eta1-offset) - dd$Deta.EDeta2
            #     good <- is.finite(z) & is.finite(w)
            # (Contrast the DISCRETE path ``_build_qr_discrete_pirls``, which
            # ports ``bgam.fitd``'s rho==0 OBSERVED-Hessian ``Deta2`` branch.)
            theta = family.get_theta()
            deta = family.dDeta(y_chunk, mu_chunk, wp_chunk, theta, level=0)
            EDeta2 = deta["EDeta2"]
            w_chunk = EDeta2 * 0.5
            z_chunk = (eta_chunk - off_chunk) - deta["Deta.EDeta2"]
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
    rho: float = 0.0,
    ar_start: Optional[np.ndarray] = None,
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

    The ``n × p`` design is never materialised: ``Xbd``/``XWXd``/``XWyd``
    scatter-add directly on the per-marginal ``Xd``/``k`` store, exactly as
    mgcv ``discrete=TRUE`` (src/discrete.c). ``Xd_list`` is invariant across
    PIRLS iters, so the weight-table contractions are the only per-iter work.
    """
    n = design.n
    if n == 0:
        raise ValueError("empty data passed to discrete PIRLS build")
    if (coef is None) and (eta_init is None):
        raise ValueError("either coef or eta_init must be provided")
    if prior_w is None:
        prior_w = np.ones(n, dtype=float)

    link = family.link

    # AR1 error model (mgcv bgam.fitd, bam.r:478-497). ``rho != 0`` builds the
    # rwMatrix ``(stop, row, weight)`` once: ``weight`` (the length-2n-1
    # ``ar.weight``) is the bidiagonal whitening transform, fed to XWXd as the
    # tridiagonal ``tri`` weight and to XWyd / y_norm2 via :func:`_rw_matrix`.
    ar1 = (rho != 0.0)
    ar_stop = ar_row = ar_weight = None
    if ar1:
        ld = 1.0 / np.sqrt(1.0 - rho ** 2)
        sd = -rho * ld
        asb = None if ar_start is None else np.asarray(ar_start, dtype=bool)
        ar_stop, ar_row, ar_weight = _ar1_rwmatrix_indices(n, ld, sd, asb)

    # mgcv bam.r:572 forms ``eta <- Xbd(coef) + offset`` ONCE per build. hea's
    # outer loop (``_bgam_fit_loop``) already formed exactly that η before calling
    # us (it needs it for step-halving), so it hands it back as ``eta_init`` —
    # reusing it here drops a second, redundant ``Xbd`` pass. Xbd is linear in
    # coef, so after step-halving ``eta_init == Xbd(halved coef) + offset`` to the
    # bit; the recompute-from-coef branch is the fallback for callers that supply
    # only coef. The not-both-None guard above makes the ``else`` safe.
    if eta_init is not None:
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
        # uses the EXPECTED (Fisher) Hessian ``EDeta2`` / ``Deta.EDeta2``
        # instead of the observed ``Deta2`` (bam.r:638-641).
        theta = family.get_theta()
        deta = family.dDeta(y, mu_full, prior_w, theta, level=0)
        if ar1:
            w_full = deta["EDeta2"] * 0.5
            z_full = (eta_full - offset) - deta["Deta.EDeta2"]
        else:
            w_full = deta["Deta2"] * 0.5
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

    if ar1:
        XWX = XWXd(design, w_full, ar_weights=ar_weight)
        Xy = XWyd(design, w_full, z_full, ar=(ar_stop, ar_row, ar_weight))
        # mgcv bam.r:654 — y_norm2 = ‖T·(√w·z)‖² (one rwMatrix, forward).
        tz = _rw_matrix(ar_stop, ar_row, ar_weight,
                        np.sqrt(w_full) * z_full, trans=False)
        y_norm2 = float(np.sum(tz * tz))
    else:
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
# bam class — Gaussian-identity chunked-QR fit (one of three fitters)
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

    # Lazily-filled cache for the :attr:`_cmX` property (mgcv object$cmX).
    _cmX_cache: np.ndarray | None = None

    def __init__(
        self,
        formula: str,
        data,
        *,
        method: str = "fREML",
        sp: np.ndarray | None = None,
        family: Family | None = None,
        offset: np.ndarray | list | None = None,
        weights: np.ndarray | list | None = None,
        scale: float = 0.0,
        gamma: float = 1.0,
        select: bool = False,
        chunk_size: int = 10000,
        use_chol: bool = False,
        rho: float = 0.0,
        ar_start: np.ndarray | list | None = None,
        discrete: bool = False,
        discrete_m: int | None = None,
        knots: dict | None = None,
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
        if knots is not None and not isinstance(knots, dict):
            raise TypeError(
                "knots must be a dict mapping covariate name -> knot sequence "
                "(mgcv's knots=list(...)), or None"
            )
        # mgcv's per-covariate knot override; threaded into both
        # materialize_smooths call sites (discrete and non-discrete) below.
        self.knots = knots

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
        # mgcv bam() dispatch (bam.r:2668-2698) is a three-way split on AR1
        # (rho != 0): bam.fit (Gaussian-identity, non-discrete) honours it,
        # bgam.fitd (discrete, ANY family) honours it, but bgam.fit (generalized,
        # non-discrete) does NOT — there mgcv warns and silently drops rho
        # (bam.r:2679 ``warning("AR1 parameter rho unused with generalized
        # model")``; rho is never even passed to bgam.fit). Mirror that exactly:
        # discrete non-Gaussian AR1 runs through the Fisher EDeta2 branch in
        # ``_build_qr_discrete_pirls``; generalized non-discrete warns + rho=0.
        if rho != 0.0 and not _is_identity_link(family) and not discrete:
            warnings.warn(
                "AR1 parameter rho unused with generalized model",
                stacklevel=2,
            )
            rho = 0.0
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

        # bam(scale=) resolution — identical to gam (estimate.gam,
        # mgcv.r:1936-1971). scale=0 → family default (the historical bam
        # path); scale>0 → φ KNOWN at that value (REML/ML drop the log-φ slot,
        # GCV.Cp → UBRE at φ=scale); scale<0 → force φ estimation. The
        # inherited _scale_known_fit / _scale_fixed_value properties read
        # _scale_resolved; every fit/post-fit branch below dispatches on
        # _scale_known_fit (== family.scale_known on the default scale=0 path,
        # so existing behaviour is byte-identical).
        if not (np.isscalar(scale) and np.isfinite(scale)):
            raise ValueError(f"scale must be a finite number, got {scale!r}")
        scale = float(scale)
        if self._family_mgcv_extended:
            if scale != 0.0:
                raise NotImplementedError(
                    "scale= for mgcv-extended families (tw, scat, nb) is not "
                    "ported — their scale handling is family-driven "
                    "(mgcv.r:1948-1949)."
                )
            self._scale_resolved = 1.0 if self.family.scale_known else -1.0
        elif self.family.scale_known:
            # poisson/binomial carry φ ≡ 1 through hea's PIRLS. mgcv-bam would
            # fix (scale>0) or estimate a quasi-dispersion (scale<0) here
            # (bam.r:472, 617-624), but the inner φ-slot diverges for a
            # scale-known family in hea, so only the φ=1 default is supported.
            if scale != 0.0:
                raise NotImplementedError(
                    "scale= for scale-known families (poisson/binomial) is "
                    "not ported — bam's quasi-likelihood dispersion handling "
                    "needs a PIRLS φ-slot hea doesn't carry for these "
                    "families. Only scale=0 (φ=1) is supported."
                )
            self._scale_resolved = 1.0
        elif method in ("REML", "ML"):
            # scale-unknown family (gaussian/Gamma/inverse.gaussian): scale>0
            # fixes φ (REML/ML drop the log-φ slot); scale≤0 estimates it.
            self._scale_resolved = scale if scale > 0 else -1.0
        else:  # GCV.Cp
            self._scale_resolved = scale if scale != 0.0 else -1.0
        self.scale_estimated = not self._scale_known_fit

        self._chunk_size = int(chunk_size)
        self._use_chol = bool(use_chol)
        self._discrete = bool(discrete)
        self._discrete_m = discrete_m
        self._discrete_design: Optional[DiscreteDesign] = None
        self._discrete_frame: Optional[DiscretizedFrame] = None

        # ---- setup phase (mirror gam.__init__ lines 198-321) ---------------
        d = prepare_design(formula, data)
        self._expanded = d.expanded
        # R model.matrix's ``assign`` for the parametric block (0 = intercept,
        # i = expanded.terms[i-1]). The inherited predict()/summary() term
        # machinery (gam.py:_term_column_groups, _pterms_rows) groups parametric
        # columns by this; without it, type='terms'/terms=/exclude= silently
        # drop every parametric term (P4).
        self._param_assign = list(d.param_assign or [])
        _expr_map = _smooth_arg_expr_map(self._expanded)
        self.data = (
            _apply_smooth_arg_exprs(d.data, _expr_map) if _expr_map else d.data
        )
        X_param_df = d.X
        # Same factor/boolean binomial-response coercion as glm/gam.
        y_full = _coerce_response(d.y, self.family)
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
                    d.expanded, mf0, sparse_cons=0, tero=True, knots=self.knots,
                )
            else:
                # discrete=FALSE: basis setup on a representative subsample
                # of the original (possibly matrix-arg) data; sparse.cons=-1
                # (sweep-drop absorb on row-summed colMeans).
                mf0 = _mini_mf(self.data, chunk_size)
                sb_lists = materialize_smooths(
                    d.expanded, mf0, sparse_cons=-1, knots=self.knots,
                )
            blocks: list[SmoothBlock] = [b for group in sb_lists for b in group]
            if discrete:
                # ``discrete_mf``'s binned model frame is NON-matrix, so the
                # ``materialize_smooths`` above can't detect the matrix-argument
                # summation convention — the blocks' specs get
                # ``summation_dim = matrix_vars = None``. That makes
                # ``BasisSpec.predict_mat`` take the non-summation branch and
                # shape-crash on a matrix-arg te/s at PREDICT (whereas
                # ``discrete=FALSE`` sets them and predicts fine). The metadata
                # is predict-only (fit-method-independent — the discrete fit
                # itself never calls ``predict_mat``), and the te covariates that
                # are matrix-typed in the original data ARE the summation
                # variables, so record them here. ``blk.term`` is the covariate
                # list (the by lives in ``spec.by``, not here), matching the
                # ``matrix_vars`` ``discrete=FALSE`` produces. ``predict_mat``
                # recomputes ``m`` from the prediction data, so the stored
                # ``summation_dim`` value only flags the route.
                for blk in blocks:
                    if blk.spec is None or blk.spec.summation_dim is not None:
                        continue
                    mvars = tuple(
                        t for t in blk.term
                        if t in self.data.columns and is_matrix_col(self.data[t])
                    )
                    if mvars:
                        blk.spec.matrix_vars = mvars
                        blk.spec.summation_dim = matrix_to_2d(
                            self.data[mvars[0]]).shape[1]
            # Per-block ``id`` (mgcv's sp-linkage key) and ``sp=`` from the
            # smooth spec, parallel to ``blocks`` — every block born from a
            # smooth call inherits that call's id/sp (a by=factor smooth's
            # level blocks all share it: the ``s(x, by=fac, id=1)`` single-λ
            # idiom). Both block-list transforms below are length/order-
            # preserving, so this stays aligned. Mirrors gam.__init__
            # (gam.py:1184-1191); the L-matrix is built from these after the
            # slots, exactly like gam.
            block_ids: list[str | None] = []
            block_sps: list[tuple[float, ...] | None] = []
            for call_node, group_blocks in zip(d.expanded.smooths, sb_lists):
                block_ids.extend([_smooth_id_value(call_node)]
                                 * len(group_blocks))
                block_sps.extend([_smooth_sp_value(call_node)]
                                 * len(group_blocks))
        else:
            blocks = []
            block_ids = []
            block_sps = []
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
            for j, S_j in enumerate(b.S):
                slots.append(_PenaltySlot(block=b, col_start=a, col_end=bcol,
                                          S=np.asarray(S_j, dtype=float),
                                          S_scale=_block_s_scale(b, j)))
            col_cursor = bcol
        p = col_cursor

        # ------------- L matrix: working → per-penalty log-sp ---------------
        # mgcv's gam.setup (mgcv.r:1280-1320): ``ρ_full = L·θ`` maps the
        # *working* (estimated) log smoothing parameters θ to the log-sp
        # multiplying each S_k. A block whose ``id`` was seen before reuses
        # the first such block's working columns (its j-th penalty shares
        # the j-th column); everything else extends L block-diagonally with
        # an identity. ``self._L is None`` ⇔ no linkage — the mapping is the
        # identity and every code path below stays byte-identical to the
        # pre-L behaviour. Direct port of gam.__init__ (gam.py:1242-1343);
        # the inherited ``_work_dim``/``_rho_full``/``_T_working`` then drive
        # the optimizers/post-fit in working space exactly as for gam.
        slot_work_col: list[int] = []
        n_work = 0
        id_first_cols: dict[str, tuple[int, int]] = {}
        # Per-block working-column range + whether this block *defines* its
        # id group (mgcv's idx[[id]]$sp.done: only the defining term's sp= is
        # consumed, mgcv.r:1430-1438).
        block_work_info: list[tuple[int, int, bool]] = []
        for b, bid in zip(blocks, block_ids):
            nS = len(b.S)
            if nS == 0:
                block_work_info.append((0, 0, False))
                continue
            if bid is None or bid not in id_first_cols:
                defining = True
                wstart = n_work
                n_work += nS
                if bid is not None:
                    id_first_cols[bid] = (wstart, nS)
            else:
                defining = False
                wstart, nc = id_first_cols[bid]
                if nS > nc:
                    # mgcv's exact refusal (mgcv.r:1312-1314).
                    raise ValueError(
                        "Later terms sharing an `id' can not have more "
                        "smoothing parameters than the first such term"
                    )
            block_work_info.append((wstart, nS, defining))
            slot_work_col.extend(range(wstart, wstart + nS))
        if n_work == len(slots):
            self._L = None                       # identity — no id linkage
        else:
            L = np.zeros((len(slots), n_work))
            L[np.arange(len(slots)), slot_work_col] = 1.0
            self._L = L
        self._n_work = n_work

        # ------------- sp=: gam-level + per-smooth merge, fixed fold -------
        # mgcv gam.setup (mgcv.r:1400-1459): the working sp vector starts from
        # bam(sp=) — or all -1 ("estimate") — then any s(..., sp=) values
        # overwrite their term's working entries (id groups: defining term
        # only). Entries >= 0 are folded out of the optimisation
        # (mgcv.r:1513-1538): lsp0 = L[, fixed] @ log(sp_fixed); L <- L[,
        # free]. Mirrors gam.__init__ (gam.py:1281-1343).
        if n_work > 0:
            sp_work = np.full(n_work, -1.0)
            if sp is not None:
                sp_arr = np.asarray(sp, dtype=float).flatten()
                if sp_arr.shape != (n_work,):
                    raise ValueError(
                        f"sp must have length {n_work} (one per estimated "
                        f"smoothing parameter; id-linked penalties share "
                        f"one), got {sp_arr.shape}"
                    )
                sp_work = sp_arr.copy()
            for (wstart, nS, defining), bsp in zip(block_work_info, block_sps):
                if bsp is None or not defining or nS == 0:
                    continue
                if len(bsp) != nS:
                    # mgcv's exact message (mgcv.r:1426).
                    raise ValueError(
                        "incorrect number of smoothing parameters "
                        "supplied for a smooth term"
                    )
                sp_work[wstart:wstart + nS] = bsp
            fixed_mask = sp_work >= 0.0
            if np.any(fixed_mask) and not np.all(fixed_mask):
                # Mixed: fold the fixed working columns into (L, lsp0).
                L_cur = (self._L if self._L is not None
                         else np.eye(len(slots)))
                fixed_vals = sp_work[fixed_mask]
                log_fixed = np.empty(fixed_vals.shape[0])
                zero = fixed_vals == 0.0
                log_fixed[~zero] = np.log(fixed_vals[~zero])
                if np.any(zero):
                    # mgcv's "effective zero" for a fixed sp of 0
                    # (mgcv.r:1519-1527), ported bug-for-bug: the i-th zero
                    # reads the i-th *penalty's* block X and S (G$off[i]/
                    # G$S[[i]] with i the literal loop counter). bam never
                    # materialises the full X; the slot's stored block basis
                    # IS X[:, col_start:col_end], and the result is scaled by
                    # eps·0.1 (≈2e-18) so the mini-mf vs full-data row count
                    # is immaterial — it is an "effectively zero penalty".
                    eps = np.finfo(float).eps
                    for i, dst in enumerate(np.flatnonzero(zero)):
                        sl = slots[i]
                        Xblk = np.asarray(sl.block.X, dtype=float)
                        ef0 = (np.linalg.norm(Xblk) ** 2
                               / np.linalg.norm(sl.S) * eps * 0.1)
                        log_fixed[dst] = np.log(ef0)
                self._lsp0 = L_cur[:, fixed_mask] @ log_fixed
                self._L = L_cur[:, ~fixed_mask]
                n_work = int(np.count_nonzero(~fixed_mask))
                self._n_work = n_work
                sp = None       # the outer machinery estimates the rest
            elif np.all(fixed_mask):
                sp = sp_work    # all fixed (possibly via s(..., sp=))
            else:
                sp = None       # nothing fixed — free optimisation

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
        # Prior weights (mgcv bam(weights=) → G$w). Threaded into the chunked
        # QR build (√w row scaling), the PIRLS Fisher weights (w·μ_η²/V), and
        # every post-fit consumer that reads ``self._wt`` (scale, leverage,
        # Pearson/deviance residuals, null deviance, R²). The common
        # binomial-trials case is the cbind(succ, fail) response, so this is
        # analytic/prior weights.
        if weights is None:
            self._wt = np.ones(n)
            self._has_prior_weights = False
        else:
            w_arr = np.asarray(weights, dtype=float).flatten()
            if w_arr.shape != (n,):
                raise ValueError(
                    f"weights must have length {n}, got {w_arr.shape}"
                )
            if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0):
                raise ValueError(
                    "weights must be finite and non-negative"
                )
            self._wt = w_arr
            self._has_prior_weights = True
        self.prior_weights = self._wt
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
        # tss for r-squared, prior-weighted (mgcv summary.gam uses the weighted
        # mean / weighted TSS; reduces to the unweighted form when weights=1).
        w = self._wt
        full_yty = float(np.sum(w * y_full * y_full))
        if has_intercept:
            mean_y = float(np.sum(w * y_full) / np.sum(w))
            tss = float(np.sum(w * (y_full - mean_y) ** 2))
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
                prior_w=(None if not self._has_prior_weights else self._wt),
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
            # ``n_work`` = number of *estimated* (working) sp's = ncol(L);
            # equals ``len(slots)`` when no smooths share an id. The optimiser
            # and user sp= live in working space; ``rho_hat`` (full per-penalty
            # log-sp) = ``_rho_full(working)`` feeds ``_fit_given_rho``/post-fit.
            n_sp = len(slots)
            n_work = self._work_dim
            if n_sp == 0:
                self.sp = np.zeros(0)
                rho_hat = np.zeros(0)
                fit = self._fit_given_rho(rho_hat)
            elif sp is not None:
                sp_arr = np.asarray(sp, dtype=float)
                if sp_arr.shape != (n_work,):
                    raise ValueError(
                        f"sp must have length {n_work} (one per estimated "
                        f"smoothing parameter; id-linked penalties share one), "
                        f"got {sp_arr.shape}"
                    )
                if np.any(sp_arr < 0):
                    raise ValueError("sp entries must be non-negative")
                rho_hat = self._rho_full(np.log(np.maximum(sp_arr, 1e-10)))
                self.sp = sp_arr
                fit = self._fit_given_rho(rho_hat)
                if (not self._scale_known_fit) and method in ("REML", "ML"):
                    Dp = float(fit.dev + fit.pen)
                    denom = (max(float(n - self._Mp), 1.0)
                             if method == "REML" else max(float(n), 1.0))
                    self._log_phi_hat = float(
                        np.log(max(Dp / denom, 1e-300))
                    )
            else:
                include_log_phi = (
                    (not self._scale_known_fit) and method in ("REML", "ML")
                )
                # initial.spg seed → working space by least squares (mgcv
                # ``coef(lm(lsp ~ L - 1 + offset(lsp0)))``, mgcv.r:4617-4618 /
                # fast-REML.r:1768; identity when no smooths share an id).
                # mgcv's Gaussian bam seeds fast.REML.fit from
                # ``initial.sp(qrx$R)`` exactly like the non-Gaussian path
                # (bam.r:1229) — there is no PIRLS outer loop for Gaussian
                # identity (W=I, z=y), so the seed feeds the one and only
                # converge-fully REML solve.
                rho0_full = self._initial_sp_rho()
                if self._lsp0 is not None:
                    rho0_full = rho0_full - self._lsp0
                if self._L is None:
                    cur_rho = rho0_full
                else:
                    cur_rho, *_ = np.linalg.lstsq(self._L, rho0_full,
                                                  rcond=None)

                if method in ("REML", "ML"):
                    # Non-discrete Gaussian REML/ML → ``fast.REML.fit``
                    # (bam.r:1240), converge-fully on the fixed (R, f).
                    if include_log_phi:
                        # mgcv bam.fit:1689 — the Gaussian path has NO PIRLS
                        # loop, so the log φ seed is unconditionally
                        # ``log(var(y)·0.05)`` (scale<=0, in.out NULL). ``var``
                        # uses R's n−1 denominator. fast.REML.fit converges fully
                        # so the seed only sets the Newton start; mirror it for
                        # source fidelity (and to share the discrete + non-
                        # discrete-generalized seed, bam.py:4134/4302).
                        cur_logphi = float(np.log(
                            max(float(np.var(y_full, ddof=1)) * 0.05, 1e-300)
                        ))
                        theta0 = np.concatenate([cur_rho, [cur_logphi]])
                    else:
                        theta0 = cur_rho
                    theta_hat = self._fast_reml_fit(
                        theta0, include_log_phi=include_log_phi,
                    )
                else:
                    # GCV.Cp → outer-Newton on V_g/V_u (no log φ in outer θ).
                    # mgcv uses ``magic`` here, which has no Sl.fitChol β/PP to
                    # reuse — null the F9 slots so the fit re-solves.
                    self._reml_beta = self._reml_A_inv = None
                    theta_hat = self._outer_newton(
                        cur_rho,
                        criterion="GCV",
                        include_log_phi=False,
                        include_family_theta=False,
                    )

                theta_sp = theta_hat[:n_work]
                if include_log_phi:
                    log_phi_hat = float(theta_hat[n_work])
                else:
                    log_phi_hat = None
                self._log_phi_hat = log_phi_hat
                self.sp = np.exp(theta_sp)            # mgcv m$sp (working)
                rho_hat = self._rho_full(theta_sp)    # full per-penalty log-sp
                fit = self._fit_given_rho(rho_hat)
                # F9: reuse fast.REML.fit's Sl.fitChol β̂ / A⁻¹ instead of the
                # re-solve (mgcv bgam.fit:1310 Sl.postproc). None on the GCV
                # path ⇒ keep the _fit_given_rho solve.
                if self._reml_beta is not None:
                    fit.beta = self._reml_beta
                    fit.A_inv = self._reml_A_inv

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
        unconditional: bool = False,
        terms: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        iterms_type: int | None = None,
    ):
        """Predict from the fitted bam — :func:`predict.bam` parity.

        Override of :meth:`hea.gam.predict` because ``self._X_full =
        qr.R`` (p × p) on every bam path: Gaussian-identity stores R for
        the chunked-QR closed-form solve (bam.py:1961), PIRLS / discrete
        store R for the inner-Newton's reduced-data scoring
        (bam.py:2791). The inherited routine assumes ``_X_full`` is the
        full ``n × p`` design and shape-clashes against ``_offset``
        (length n) when ``newdata=None``.

        ``predict.bam`` routes a DISCRETE fit to ``predict.bamd``
        (bam.r:1421); hea mirrors that for the value / SE / lpmatrix surface
        via :meth:`_predict_bamd` (bin the covariates, gather via the
        discrete kernels). ``terms``/``iterms`` and term selection on a
        discrete fit fall through to the exact-eval parent path below
        (the per-term discrete decomposition is not ported).

        Behaviour for each (newdata, type, se_fit) combination (non-discrete
        fit, or the discrete decomposition surface):

        * ``newdata`` not None → delegate to ``super().predict(...)`` —
          parent rebuilds the design via per-block ``spec.predict_mat``
          the same way it does for gam fits.
        * ``newdata=None``, ``type='link' | 'response'``, no
          ``se_fit``, no extra ``offset`` → cached
          :attr:`linear_predictors` / :attr:`fitted_values`.
        * ``newdata=None`` with ``type='lpmatrix'`` → route through
          ``super().predict(newdata=self.data, type='lpmatrix')``,
          which re-evaluates each smooth's basis on training rows
          (bit-equal to the fit-time design for non-discrete bam).
        * ``newdata=None`` with ``se_fit=True`` → cached eta (+ extra
          offset, if any) for the link-scale prediction; chunked
          ``diag(X·Vp·X')`` (via :meth:`_chunked_var_eta_diag`) for
          per-row link-scale variance; delta-method
          ``|μ_η|`` multiplier for response-scale SE.
        """
        if type not in ("link", "response", "terms", "iterms", "lpmatrix"):
            raise ValueError(
                "type must be 'link', 'response', 'terms', 'iterms', or "
                f"'lpmatrix'; got {type!r}"
            )
        if type == "lpmatrix" and se_fit:
            raise ValueError(
                "se_fit=True is not allowed with type='lpmatrix'"
            )

        # predict.bam routes a DISCRETE fit (``object$dinfo`` set) to
        # predict.bamd (bam.r:1421) — bin newdata's covariates to the
        # compress.df grid and gather via the discrete kernels, NOT exact
        # basis evaluation. hea mirrors this for the value / SE / lpmatrix
        # surface; ``terms``/``iterms`` and term selection stay on the
        # exact-eval parent (the per-term discrete decomposition with
        # mgcv's parametric-term grouping is not ported — see
        # :meth:`_predict_bamd`).
        if (self._discrete_design is not None
                and type in ("link", "response", "lpmatrix")
                and terms is None and exclude is None):
            return self._predict_bamd(
                newdata, type=type, se_fit=se_fit, offset=offset,
                unconditional=unconditional,
            )

        # predict.bam delegates the non-discrete case to predict.gam, so the
        # whole gam.predict surface (terms/iterms decomposition, terms=/
        # exclude= selection, unconditional=Vc) is available. bam only
        # *overrides* the cached/chunked fast paths for whole-model
        # link/response/lpmatrix; everything else routes to the parent.
        if newdata is not None:
            return super().predict(
                newdata=newdata, type=type, se_fit=se_fit, offset=offset,
                unconditional=unconditional, terms=terms, exclude=exclude,
                iterms_type=iterms_type,
            )

        # ---- newdata=None branch -------------------------------------------
        # Per-term decomposition, term selection, or the sp-uncertainty
        # covariance all need the full n×p design, which bam's fast paths
        # don't carry (self._X_full is the p×p R factor). Route those through
        # the parent on the training frame — same mechanism as the lpmatrix
        # branch below, bit-equal to the fit-time design for non-discrete bam.
        if (type in ("terms", "iterms")
                or terms is not None or exclude is not None or unconditional):
            return super().predict(
                newdata=self.data, type=type, se_fit=se_fit, offset=offset,
                unconditional=unconditional, terms=terms, exclude=exclude,
                iterms_type=iterms_type,
            )

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
    # predict.bamd — discrete (binned) prediction path
    # -----------------------------------------------------------------------

    def _build_predict_discrete_design(self, newdata):
        """Re-discretise ``newdata`` and assemble its binned
        :class:`DiscreteDesign`, mirroring mgcv ``predict.bamd``'s discrete
        setup (bam.r:1843-1921).

        Like ``predict.bamd`` (bam.r:1847 calls ``discrete.mf`` with no
        ``m=``) this always re-grids to ``compress.df``'s DEFAULT resolution
        (1-D: 1000 levels), independent of the fit's ``discrete=`` — so for a
        coarse fit the prediction grid is FINER than the fit grid. The fitted
        per-margin bases (frozen in ``block.spec``) are evaluated on the new
        grid (mgcv ``PredictMat``) and a numeric by-variable is binned
        (bam.r:1889), exactly as the fitter does.

        Returns ``(design, n_user, n_stubs, off_new)`` where ``off_new`` is
        the (stub-trimmed) formula offset evaluated on ``newdata``.
        """
        newdata = normalize_data(newdata)
        expr_map = _smooth_arg_expr_map(self._expanded)
        if expr_map:
            newdata = _apply_smooth_arg_exprs(newdata, expr_map)
        # Carry fit-time factor levels through (predict.gam's xlevels); the
        # stubs keep the parametric contrasts at the fit's column count.
        n_user = newdata.height
        newdata, n_stubs = _add_factor_stub_rows(newdata, self.data)
        # Parametric design on newdata — response-free, the same call
        # gam.predict uses. ``build_discrete_design`` stores these columns
        # directly (identity gather), so the parametric part is exact
        # (matching the fitter's ``X_param_full`` handling), not re-binned.
        X_param = materialize(self._expanded, newdata).to_numpy().astype(float)
        # Discretise the smooth covariates at the default resolution.
        specs = _smooth_specs_from_expanded(self._expanded, newdata)
        names_pmf = [c for c in self.parametric_columns
                     if c != "(Intercept)" and c in newdata.columns]
        frame = discrete_mf(specs, newdata, names_pmf=names_pmf, m=None)
        design = build_discrete_design(
            self._blocks, X_param, frame, data=newdata)
        off_new = np.zeros(newdata.height)
        for off_node in self._expanded.offsets:
            blk = _eval_atom(off_node, newdata)
            off_new = off_new + blk.values.flatten().astype(float)
        if n_stubs > 0:
            off_new = off_new[:n_user]
        return design, n_user, n_stubs, off_new

    def _predict_bamd(self, newdata, *, type, se_fit, offset, unconditional):
        """Predict from a discrete bam fit via the discrete kernels — port
        of mgcv ``predict.bamd`` (bam.r:1773-2033), value/SE/lpmatrix
        surface.

        The covariates of ``newdata`` (or the training frame when
        ``newdata=None``) are binned to ``compress.df``'s grid and the
        prediction is gathered from the per-marginal compressed design
        (mgcv ``Xbd``); link-scale SE is ``sqrt(diag(Xd·V·Xd'))`` via
        :func:`diagXVXd` (mgcv ``diagXVXd``), one column at a time — the
        ``n × p`` design is never formed. For a continuous covariate this
        differs from exact basis evaluation by the binning — exactly the
        divergence F1 was about, and it makes ``lpmatrix @ coef == fitted``
        hold for a continuous discrete fit. ``type='response'`` applies the
        delta-method ``|dμ/dη|`` SE multiplier and ``linkinv``
        (bam.r:1989-1990).

        ``newdata=None`` + the DEFAULT discretisation (``discrete_m is
        None``) ⇒ the prediction grid equals the fit grid, so the cached
        fit values / design are reused (bit-identical to ``fitted``, which
        is what mgcv returns there too); a fit with a custom ``discrete=m``
        is re-gridded at the (finer) default resolution, matching mgcv.

        ``terms``/``iterms`` and term selection are handled by the caller
        on the exact-eval parent path, not here.
        """
        beta = np.asarray(self.coefficients).reshape(-1)
        extra = None
        if offset is not None:
            extra = np.asarray(offset, dtype=float).flatten()

        # Reuse the fit grid only when it provably equals the predict grid
        # (newdata omitted AND the fit used the default resolution).
        reuse = newdata is None and self._discrete_m is None
        if reuse:
            design = self._discrete_design
            n_pred = self.n
        else:
            src = self.data if newdata is None else newdata
            design, n_pred, n_stubs, off_new = (
                self._build_predict_discrete_design(src))

        if extra is not None and extra.shape != (n_pred,):
            raise ValueError(
                f"offset must have length {n_pred}, got {extra.shape}")

        # lpmatrix: the design matrix is the requested output — form it column
        # by column via Xbd(e_kk) (mgcv predict.bamd's ``Xbd(Xd, I)``); no
        # other branch needs the dense design.
        if type == "lpmatrix":
            p = design.p
            Xf = np.empty((n_pred, p), dtype=float)
            ek = np.zeros(p, dtype=float)
            for kk in range(p):
                ek[kk] = 1.0
                Xf[:, kk] = Xbd(design, ek)[:n_pred]
                ek[kk] = 0.0
            return Xf

        # value (link scale) — η = Xβ via the scatter (mgcv Xbd), no
        # materialise. Reuse the cached η for bit-identical newdata=None
        # default-resolution predictions.
        if reuse:
            eta = self.linear_predictors.copy()
        else:
            eta = Xbd(design, beta)[:n_pred] + off_new
        if extra is not None:
            eta = eta + extra

        if not se_fit:
            if type == "link":
                return pl.DataFrame({"fit": eta})
            if reuse and extra is None:
                return pl.DataFrame({"fit": self.fitted_values.copy()})
            return pl.DataFrame({"fit": self.family.link.linkinv(eta)})

        # SE = sqrt(diag(Xd·V·Xd')). unconditional=True uses the
        # sp-uncertainty-corrected Vc (predict.gam's ``unconditional``),
        # falling back to Vp when Vc is unavailable (GCV fits).
        V = self.Vp
        if unconditional:
            Vc = getattr(self, "Vc", None)
            if Vc is not None:
                V = Vc
        var_eta = diagXVXd(design, V)[:n_pred]   # mgcv diagXVXd, no materialise
        se_link = np.sqrt(np.maximum(var_eta, 0.0))
        if type == "link":
            return pl.DataFrame({"fit": eta, "se.fit": se_link})
        if reuse and extra is None:
            mu = self.fitted_values.copy()
        else:
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

        # Pivoted Cholesky with rank revealing. mgcv's bam coef solve runs
        # through Sl.fitChol's ``chol(A_pre, pivot=TRUE)`` (fast-REML.r:1606)
        # = LAPACK DPSTRF with tol=-1 → its default ``N·eps·max(diag)``. Use
        # dpstrf's default (NOT gam.fit3's QR-path ``eps·100``, a different
        # routine) so the rank determination matches mgcv's chol.
        A_pre_f = np.asfortranarray(A_pre.copy())
        R_pre, piv_1based, rank_A, _info = dpstrf(A_pre_f, lower=0)
        R_pre = np.triu(R_pre)
        # mgcv Sl.fitChol:1607 ``r <- min(attr(R,"rank"), Rrank(R))`` — same
        # min as ``_pi_fit_chol``: dpstrf's pivot rank ∧ mgcv's Cline-condition
        # ``Rrank`` (mgcv.r:4, tol=eps^0.9). Full-rank → p (no change); only
        # bites when a near-singular leading direction survives dpstrf's tol.
        rank_A = min(int(rank_A),
                     _R_rank(R_pre, tol=float(np.finfo(float).eps) ** 0.9))
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

    @property
    def _cmX(self) -> np.ndarray:
        """mgcv ``object$cmX`` — true design column means, computed lazily.

        bam's ``_X_full`` is the p×p QR factor, so the inherited iterms SE
        widening (gam.py ``_terms_frame``) can't recover ``colMeans(X)`` from
        it — its ``_X_full.mean(axis=0)`` fallback would average the R factor's
        rows, not the design. Only ``type='iterms'`` + ``se_fit`` on a
        constrained smooth reads this, so compute on first access and cache —
        ordinary fits (and the matrix-argument/distributed-lag smooths whose
        bases don't re-materialise from a ``self.data`` chunk) never pay for
        it.
        """
        if self._cmX_cache is None:
            self._cmX_cache = self._chunked_colmeans()
        return self._cmX_cache

    def _chunked_colmeans(self) -> np.ndarray:
        """Column means of the full n×p design (backs :attr:`_cmX`).

        Same chunk-walk dispatch as :meth:`_chunked_var_eta_diag`: discrete bam
        gathers each block via ``spec.predict_mat``, non-discrete re-evaluates
        the basis with :func:`_materialize_chunk`.
        """
        n = self.n
        acc = np.zeros(self.p, dtype=float)
        if self._discrete_design is not None:
            X_param_full = self._X_param_full
            for start, end in _chunk_indices(n, self._chunk_size):
                cols = [X_param_full[start:end]]
                for b in self._blocks:
                    if b.spec is None:
                        raise RuntimeError(
                            f"smooth block {b.label!r} (cls={b.cls!r}) "
                            f"has no BasisSpec; cmX requires every smooth "
                            f"to carry one."
                        )
                    cols.append(np.asarray(
                        b.spec.predict_mat(self.data[start:end]),
                        dtype=float,
                    ))
                acc += np.concatenate(cols, axis=1).sum(axis=0)
            return acc / n
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = _materialize_chunk(
                self._blocks,
                self.data[start:end],
                self._X_param_full[start:end],
            )
            acc += X_chunk.sum(axis=0)
        return acc / n

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
        ``A_inv`` returns the link-scale variance. For a discrete fit the
        predict-time SE goes through :meth:`_predict_bamd` (mgcv
        ``diagXVXd``) instead; the discrete branch here uses the same
        compressed-grid :func:`diagXVXd` (the fit/edf gauge), one column at a
        time — no materialise.
        """
        n = self.n
        out = np.empty(n, dtype=float)
        if self._discrete_design is not None:
            # diag(Xd·Vp·Xd') on the compressed grid (mgcv diagXVXd) — same
            # gauge as the fit/edf and _predict_bamd's SE; no materialise.
            return diagXVXd(self._discrete_design, Vp)
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
        self._rho_hat = rho_hat
        # mgcv m$full.sp — per-penalty sp expansion exp(L·log(sp)+lsp0).
        # Equals m$sp when nothing shares an id and nothing is fixed.
        self.full_sp = np.exp(np.asarray(rho_hat, dtype=float))

        # Inverse Hessian — small (p×p), exact. ``fit.A_inv`` is set (= mgcv's
        # un-repara'd Sl.fitChol PP, bgam.fitd:823) when fast.REML.fit supplied
        # the reuse (F9); otherwise cho_solve the A_chol. Identical full-rank.
        if fit.A_inv is not None:
            A_inv = fit.A_inv
        else:
            A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        # XtWX with W=I is just X'X = R'R, already cached.
        XtWX = self._XtX
        A_inv_XtWX = A_inv @ XtWX
        edf = np.diag(A_inv_XtWX).copy()
        edf_total = float(edf.sum())

        # Prior weights (mgcv G$w) — resolved in __init__; the chunked QR was
        # built from √w·(X, z) so fit.dev is the weighted RSS Σ wᵢ(yᵢ−μ̂ᵢ)².
        wt = self._wt
        df_resid = float(n - edf_total)
        # Gaussian: V=1, scale = Σwᵢ(yᵢ - μ̂ᵢ)²/(n - edf). fit.dev already holds
        # the full-data residual sum of squares (rss_extra absorbed).
        if df_resid > 0:
            pearson_scale = float(fit.dev) / df_resid
        else:
            pearson_scale = float("nan")
        self._pearson_scale = pearson_scale
        # A user scale=φ fixes the Gaussian scale; otherwise the REML/Pearson
        # estimate (mgcv G$sig2 <- scale when known, mgcv.r:1942).
        sigma_squared = (self._scale_fixed_value if self._scale_known_fit
                         else pearson_scale)
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
        # User-facing coefficient reporting (mirrors gam.py:1840-1854). The
        # inherited summary()/_se_report_for read _beta_report/_se_report; bam
        # never set them, so summary() raised AttributeError. bam doesn't drop
        # columns today (_keep_cols is None) but keep gam's full branch for
        # forward-compat with drop.intercept (P7).
        if self._keep_cols is not None:
            beta_rep = np.zeros(self._keep_cols.size)
            beta_rep[self._keep_cols] = np.asarray(beta).reshape(-1)
            se_rep = np.zeros(self._keep_cols.size)
            se_rep[self._keep_cols] = se
        else:
            beta_rep = np.asarray(beta).reshape(-1)
            se_rep = se
        self._beta_report = beta_rep
        self._se_report = se_rep
        t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        self.t_values = _row_frame(t_stats, self.column_names)
        if df_resid > 0 and np.isfinite(df_resid):
            pv = 2 * _dist.pt(np.abs(t_stats), df_resid, lower_tail=False)
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
        # Gaussian deviance residuals = sign(y-μ)·√(w(y-μ)²) = √w·(y-μ)
        # (mirrors gam.py:1881; reduces to y-μ when weights=1).
        self.residuals = self._deviance_residuals(y, mu, self._wt)
        self.sigma = sigma
        self.sigma_squared = sigma_squared
        self.scale = sigma_squared

        # Leverage diag: chunked. WLS hat h_i = w_i·(X A⁻¹ X')_ii (A = X'WX);
        # the chunk walk returns the unweighted (X A⁻¹ X')_ii, so scale by the
        # prior weight. Σ h_i = tr(A⁻¹ X'WX) = edf_total exactly.
        leverage = self._chunked_leverage_diag(A_inv) * self._wt
        self.leverage = leverage
        sigma_for_std = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
        denom = sigma_for_std * np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        # V(μ)=1, Pearson residual = √w·(y - μ)/√V = √w·(y - μ).
        pearson_res = np.sqrt(self._wt) * (y - mu)
        self.std_dev_residuals = self.residuals / denom
        self.std_pearson_residuals = pearson_res / denom
        self.df_residuals = df_resid
        # mgcv bam.r:2774 — ``object$deviance = sum(object$residuals^2)``. With
        # the weighted deviance residuals √w·(y−μ) this is Σ wᵢ(yᵢ−μᵢ)² (=
        # fit.dev), the weighted RSS. For AR1 (rho != 0) the AR1-decorrelated
        # RSS lives separately in ``std.rsd`` (used for σ² and AIC scale
        # calcs). The ``deviance`` is what ``deviance.explained`` reports
        # against ``null.deviance``, both on the original y scale.
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

        # mgcv oo$rank.est (P5). bam's _X_full = R already encodes √W·X
        # (R'R = X'WX); with _fisher_w=None the inherited _estimate_rank runs
        # the pivoted-QR rank reveal on R directly — rank-equivalent to gam's
        # √W·X path (same Gram, same column space, same Frobenius scaling).
        self.rank = self._estimate_rank()

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
            # Working-space view (id linkage): the criterion is optimised over
            # θ (ρ = L·θ), so H_aug — and every CI built on it (vcomp, edf2,
            # Vc) — lives in working coordinates H_θ = T'·H_ρ·T,
            # T = blockdiag(L, I_logφ). ``None`` ⇔ identity (no id linkage)
            # → byte-identical to the pre-L path (gam.py:2032-2035).
            T_aug = self._T_working(1)
            if T_aug is not None:
                H_aug = T_aug.T @ H_aug @ T_aug
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
        sc_p = 0.0 if self._scale_known_fit else 1.0
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

        For ``discrete=True`` the diagonal comes from :func:`diagXVXd` on the
        compressed grid (mgcv ``diagXVXd``), never forming the n×p design.
        """
        if self._discrete_design is not None:
            return w_full * diagXVXd(self._discrete_design, A_inv)
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

    # -----------------------------------------------------------------------
    # On-demand leverage / standardised residuals (mgcv bgam.fitd stores no
    # n-length hat — bam.r:806-894). The Gaussian-identity *non-discrete*
    # post-fit (``_post_fit_gaussian``) still sets these eagerly as instance
    # attributes, which shadow these (non-data-descriptor) cached properties;
    # the PIRLS/discrete post-fit leaves them lazy, so the O(n·p²) ``diagXVXd``
    # leverage runs only when a diagnostic (``hatvalues``/``rstandard``/Cook's
    # D) actually asks for it.
    # -----------------------------------------------------------------------

    @cached_property
    def leverage(self) -> np.ndarray:
        """Hat-matrix diagonal ``hᵢ = wᵢ·(X A⁻¹ X')ᵢᵢ`` (mgcv ``influence``);
        ``Σ hᵢ = edf_total``. Computed lazily via :func:`diagXVXd` on the
        compressed grid (no n×p materialise)."""
        return self._chunked_leverage_diag_weighted(self._A_inv, self._lev_w)

    def _std_resid_denom(self) -> np.ndarray:
        sigma = self.sigma
        sigma_for_std = sigma if (np.isfinite(sigma) and sigma > 0) else 1.0
        return sigma_for_std * np.sqrt(np.clip(1.0 - self.leverage, 1e-12, None))

    @cached_property
    def std_dev_residuals(self) -> np.ndarray:
        """Standardised deviance residuals ``rᵢ / (σ·√(1−hᵢ))`` (mgcv
        ``rstandard``)."""
        return self.residuals / self._std_resid_denom()

    @cached_property
    def std_pearson_residuals(self) -> np.ndarray:
        """Standardised Pearson residuals (mgcv ``rstandard(type="pearson")``).
        ``V(μ)=1`` for Gaussian-identity, recovering ``√w·(y−μ)``."""
        mu = self.fitted_values
        V_mu = self.family.variance(mu)
        pearson = (self._y_arr - mu) * np.sqrt(self._wt / np.maximum(V_mu, 0.0))
        return pearson / self._std_resid_denom()

    def _fast_reml_fit(
        self, theta0: np.ndarray, *, include_log_phi: bool,
        max_iter: int = 200,
    ) -> np.ndarray:
        """mgcv ``fast.REML.fit`` (fast-REML.r:1740-1875) — Newton optimiser
        for the working log-sp (and log φ when the scale is free) on the
        reduced ``(R, f)`` data, run to FULL convergence.

        This is the optimiser mgcv's *non-discrete* ``bgam.fit`` (bam.r:1240)
        calls once per PIRLS iter, via ``Sl.fit``. ``Sl.fit`` and ``Sl.fitChol``
        compute the SAME REML score/grad/Hessian (QR vs Cholesky numerics), so
        this drives :func:`_pi_fit_chol` (hea's ``Sl.fitChol`` port) as the
        per-evaluation oracle — the very same routine the *discrete*
        ``bgam.fitd`` (bam.r:733) one-step POI uses. The two paths differ only
        in CADENCE: ``bgam.fitd`` takes ONE step per PIRLS iter with
        gradient-based halving; this runs the step to FULL convergence with
        reml-VALUE halving.

        CRITICAL: the REML here is the **Gaussian working-model** REML on the
        reduced ``(R, f)`` — ``(nobs/γ−Mp)·log(2πφ)`` normalisation, NO
        non-Gaussian ``ls`` term (``Sl.fit``/``Sl.fitChol`` treat the linearised
        ``(R, f)`` as Gaussian; the family lives only in the OUTER PIRLS loop's
        W, z build). This is why it must come from ``_pi_fit_chol`` and NOT from
        ``_reml`` (the full non-Gaussian Tweedie/Gamma REML, which gam uses):
        on a scale-unknown Tweedie bam, mgcv-bam reaches sp 0.259 (Gaussian
        working REML); ``_reml`` on the same reduced data minimises a different
        objective (0.207) — verified against mgcv-bam.

        ``fast.REML.fit``'s loop (vs gam.fit3's ``newton`` in
        :meth:`_outer_newton`):

        * eigen-flip Hessian regularisation (negative λ → |λ|, floor at
          ``max|λ|·√eps``), step capped at ``maxNstep = 5`` — :func:`_reg_newton_step`
          with ``max_step=5`` (fast-REML.r:1805-1818);
        * step-halving on the reml VALUE (``trial$reml > best$reml``), with the
          ``not.moved`` early-stall detector (fast-REML.r:1827-1839);
        * the TIGHT convergence threshold ``max|grad| ≤ reml.scale·√eps`` AND
          ``|Δreml| ≤ reml.scale·√eps`` (fast-REML.r:1855-1857), where
          ``reml.scale = |reml| + rss/nobs`` (fast-REML.r:1776).

        ``theta`` layout matches :meth:`_outer_newton`: working ρ (``n_work``)
        then a single log φ slot when ``include_log_phi``. The working↔full
        map is ``T = blockdiag(L, I_φ)`` — ``_T_working`` — exactly mgcv's
        augmented ``L`` (fast-REML.r:1782) and its ``t(L)`` grad/Hess
        contraction (fast-REML.r:1784-1785, 1848-1849). Returns ``theta_hat``;
        the caller recovers β̂ via ``_fit_given_rho(_rho_full(θ_hat[:n_work]))``.
        """
        n_work = self._work_dim
        # T = blockdiag(L, 1_φ): the working→full Jacobian (mgcv's augmented
        # L). None ⇔ identity (no id linkage, no φ slot) → zero-cost contract.
        T_work = self._T_working(1 if include_log_phi else 0)
        conv_tol = float(np.finfo(float).eps) ** 0.5     # mgcv conv.tol
        max_step = 5.0                                   # mgcv maxNstep
        nobs = float(self.n)
        n_int = int(self.n)

        # ``Sl.setup`` + ``Sl.initial.repara`` (fast-REML.r:68-429, 517-588):
        # relay to gam's shared Sl machinery (the SAME ``Sl.setup`` mgcv calls
        # from both gam and bam) to reparameterize every penalty block into
        # mgcv's well-scaled gauge, so _pi_fit_chol's pivoted Cholesky
        # factorizes the same conditioned matrix mgcv does (bam.r:541/664).
        # ``both_sides=True`` realises the two-sided gram transform
        # ``D'(X'WX)D`` / ``D'(X'Wz)``. Singleton transforms are non-orthogonal
        # (eigen ``U·diag(1/√λ)``), so the reml VALUE's ``ldetXXS`` shifts by
        # ``ldet_const``; the grad/Hess are congruence-invariant. β is
        # recovered by the caller (not used here). Lazily built — depends only
        # on the slot S matrices.
        if not hasattr(self, "_sl"):
            self._sl = _sl_setup(self._slots, self.p)
        XX_pre = _sl_initial_repara(self._sl, self._XtX, both_sides=True)
        Xy_pre = _sl_initial_repara(self._sl, self._Xty, both_sides=True)
        # ``log|Sλ|_+`` correction to the repara'd gauge: subtract the
        # rho-independent ``Σ_pen log λ`` so ``ldetXXS − ldet_S`` (computed in
        # _pi_fit_chol's repara'd gauge) matches mgcv's invariant difference.
        ldS_const = _sl_initial_repara_ldet_const(self._sl)

        def _eval(t):
            # One Sl.fit / Sl.fitChol evaluation at working θ → dict with the
            # Gaussian working-model REML VALUE, the t(L)-contracted working
            # grad/Hess, and the unpenalised working RSS (for reml.scale).
            theta_sp = t[:n_work]
            rho = self._rho_full(theta_sp)
            log_phi = (float(t[n_work]) if include_log_phi
                       else float(np.log(self._scale_fixed_value)))
            # log|S|_+ value + ρ-derivatives — the XX-independent pieces.
            S_full = self._build_S_lambda(rho)
            S_full = 0.5 * (S_full + S_full.T)
            S_pinv = self._S_pinv(S_full)
            ldS_val = float(self._log_det_S_pos(rho)) - ldS_const
            ldS_grad = self._dlog_det_S_drho(
                rho, S_pinv=S_pinv, S_full=S_full)
            ldS_hess = self._d2log_det_S_drho_drho(
                rho, S_pinv=S_pinv, S_full=S_full)
            try:
                out = _pi_fit_chol(
                    XX_pre, Xy_pre, rho, self._sl, self.p,
                    yy=self._yty, log_phi=log_phi, n=n_int,
                    Mp=self._Mp, gamma=self._gamma,
                    phi_fixed=not include_log_phi,
                    ldet_S=ldS_val, ldet_S_grad=ldS_grad, ldet_S_hess=ldS_hess,
                )
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                return None
            reml_val = float(out["reml"])
            if not np.isfinite(reml_val):
                return None
            g = out["grad"]
            H = out["hess"]
            if T_work is not None:
                g = T_work.T @ g
                H = T_work.T @ H @ T_work
            H = 0.5 * (H + H.T)
            # Keep ``out`` (β/PP in the repara'd gauge) so the converged fit
            # reuses Sl.fitChol's solution instead of re-solving — F5/F9, mgcv
            # bgam.fit:1310 Sl.postproc. Carrying the reference is free.
            return {"reml": reml_val, "grad": g, "hess": H,
                    "rss": float(out["rss"]), "out": out}

        # F5/F9 reuse slots: the converged β̂ / A⁻¹ from Sl.fitChol, un-repara'd
        # to the original basis (bgam.fitd:759/823). ``None`` until the fit
        # succeeds, so a failed initial eval falls back to ``_fit_given_rho``.
        self._reml_beta = None
        self._reml_A_inv = None

        theta = np.asarray(theta0, dtype=float).copy()

        # ---- initial fit + typical reml scale (fast-REML.r:1774-1776) ----
        best = _eval(theta)
        if best is None:
            self._outer_info = {
                "conv": "initial fit failed", "iter": 0,
                "grad": np.zeros_like(theta),
                "hess": np.zeros((theta.size, theta.size)),
                "score": float("inf"), "score_scale": float("nan"),
            }
            return theta
        reml_scale = abs(best["reml"]) + best["rss"] / nobs
        grad = best["grad"]
        hess = best["hess"]
        grad2 = np.diag(hess)
        # active set: drop dims with ~0 grad AND ~0 curvature (fast-REML.r:1791).
        uconv_ind = ((np.abs(grad) > reml_scale * conv_tol * 0.1)
                     | (np.abs(grad2) > reml_scale * conv_tol * 0.1))
        if not np.any(uconv_ind):
            uconv_ind = np.ones_like(uconv_ind, dtype=bool)

        step_failed = False
        conv_text = "no convergence in 200 iterations"
        it_done = 0
        for it in range(max_iter):
            it_done = it + 1
            # Newton step on the active subblock (fast-REML.r:1802-1821).
            if hess.size > 0:
                H1 = hess[np.ix_(uconv_ind, uconv_ind)]
                g1 = grad[uconv_ind]
                uc_step = _reg_newton_step(g1, H1, max_step=max_step)
                step = np.zeros_like(grad)
                step[uconv_ind] = uc_step
            else:
                step = np.zeros_like(grad)

            # Try the step; step-halve on the reml VALUE until improvement
            # or failure (fast-REML.r:1822-1839).
            theta_try = theta + step
            trial = _eval(theta_try)
            k = 0
            not_moved = 0
            while trial is None or trial["reml"] > best["reml"]:
                # ``not.moved``: count consecutive halvings with a
                # numerically-insignificant reml change from best — an
                # early step-failure indicator (fast-REML.r:1828-1831).
                if (trial is not None
                        and (trial["reml"] - best["reml"])
                        < conv_tol * reml_scale):
                    not_moved += 1
                else:
                    not_moved = 0
                if (k == 25 or not np.any(step != 0.0) or not_moved > 3):
                    step_failed = True
                    break
                step = step / 2.0
                k += 1
                theta_try = theta + step
                trial = _eval(theta_try)
            if step_failed:
                conv_text = "step failed"
                break

            # Step accepted. Convergence test (fast-REML.r:1847-1864).
            grad = trial["grad"]
            hess = trial["hess"]
            grad2 = np.diag(hess)
            uconv_ind = ((np.abs(grad) > reml_scale * conv_tol * 0.1)
                         | (np.abs(grad2) > reml_scale * conv_tol * 0.1))
            converged = True
            if np.any(np.abs(grad) > reml_scale * conv_tol):
                converged = False
            if abs(best["reml"] - trial["reml"]) > reml_scale * conv_tol:
                if converged:           # otherwise can't progress
                    uconv_ind = np.ones_like(uconv_ind, dtype=bool)
                converged = False
            best = trial
            theta = theta_try
            if converged:
                conv_text = "full convergence"
                break
            reml_scale = abs(best["reml"]) + best["rss"] / nobs

        self._outer_info = {
            "conv": conv_text, "iter": it_done,
            "grad": grad, "hess": hess,
            "score": float(best["reml"]), "score_scale": float(reml_scale),
        }
        # F5/F9: reuse Sl.fitChol's converged β̂ / A⁻¹ (mgcv bgam.fit:1310's
        # Sl.postproc does NOT re-solve). ``best["out"]`` is the _pi_fit_chol
        # result at the accepted θ; un-repara β (both_sides=False) per
        # bgam.fitd:759 and PP (both_sides=True, cov=True) per bgam.fitd:823.
        # Full-rank: identical to _fit_given_rho's solve (~1e-12); rank-
        # deficient: mgcv's pivoted-Cholesky null-space gauge.
        best_out = best.get("out")
        if best_out is not None:
            self._reml_beta = _sl_initial_repara(
                self._sl, best_out["beta"], inverse=True,
                both_sides=False, cov=False)
            self._reml_A_inv = _sl_initial_repara(
                self._sl, best_out["PP"], inverse=True,
                both_sides=True, cov=True)
        return theta

    def _bgam_rsb_penalty(self, rho_full: np.ndarray,
                          coef: np.ndarray) -> float:
        """mgcv ``sum(rSb²)`` (bgam.fitd:591/611) — the penalty ``βᵀSλβ`` as the
        per-block penalty-root reduction the PIRLS step-halving + convergence
        test read, NOT a full-matrix quadratic form ``coef·Sλ·coef``.

        ``coef`` is original-gauge; reparam'd to mgcv's ``prop$beta`` (the
        initial-repara gauge ``Sl.rSb`` operates in) by the FORWARD initial
        repara before stacking the roots. ``rho_full`` is the full per-penalty
        log-sp. Builds ``self._sl`` lazily if a path reached here before any
        ``Sl.fitChol`` step did (it depends only on the slot S matrices).
        """
        if not hasattr(self, "_sl"):
            self._sl = _sl_setup(self._slots, self.p)
        b = _sl_initial_repara(self._sl, np.asarray(coef, dtype=float),
                               inverse=False, both_sides=False, cov=False)
        a = _sl_rsb(self._sl, rho_full, b)
        return float(np.dot(a, a))

    def _bgam_fit_loop(self, *, sp_user) -> tuple["_FitState", np.ndarray]:
        """Outer PIRLS loop with chunked QR rebuild per iter.

        Direct port of mgcv ``bgam.fit`` (bam.r:909-1353). Each iteration:

        1. Build (R, f, ‖z̃‖², rss_extra) over chunks of √W·X / √W·z, where
           (W, z) are the Fisher PIRLS weights/working response computed from
           the chunk's η = X·β + offset (or the family-initialised η on
           iter 1, when β is still ``None``).
        2. Update reduced sufficient stats ``self._XtX = R'R``, ``self._Xty
           = R'f``, ``self._yty = ‖z̃‖²``, ``self._X_full = R``.
        3. Update (ρ, log φ) on the reduced (R, f), then recover β̂ at the new
           ρ̂ via ``_fit_given_rho``. The sp-update optimiser mirrors mgcv's
           per-``discrete`` split EXACTLY:
             * discrete=TRUE  → one ``_pi_fit_chol`` (Sl.fitChol) POI step,
               gradient-halving, warm-started (mgcv ``bgam.fitd``);
             * discrete=FALSE → ``_fast_reml_fit`` (mgcv ``fast.REML.fit``) run
               to full convergence, reml-value halving, re-seeded from
               ``initial.sp`` each iter (mgcv ``bgam.fit``);
             * GCV.Cp → ``_outer_newton`` on V_g/V_u.
           Both REML optimisers see the SAME Gaussian working-model REML on
           (R, f) and reach mgcv's true optimum (tight √eps convergence).
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
        n_sp = len(self._slots)
        method = self.method   # already mapped fREML → REML

        blocks = self._blocks
        chunk_size = self._chunk_size
        y = self._y_arr
        off = self._offset
        prior_w = self._wt     # mgcv bam(weights=) → G$w (ones if unset)

        include_log_phi = (not self._scale_known_fit) and method in ("REML", "ML")

        # mgcv bgam.fitd:500 — Gaussian-identity ⇒ the PIRLS (W, z) are CONSTANT
        # across iters, so the working model (R, f, X'X, X'y, ‖z‖²) is built ONCE
        # and reused; later iters only refresh the penalised deviance and take
        # another sp Newton step. ``additive`` gates that reuse (bam.r:567).
        additive = _is_identity_link(family)

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
        mu = family.gam_initialize(y, prior_w)
        eta = link.link(mu)
        if not (link.valideta(eta) and family.validmu(mu)):
            raise FloatingPointError(
                "PIRLS init: cannot find valid starting μ̂ from family.initialize"
            )
        coef: Optional[np.ndarray] = None
        coef0: Optional[np.ndarray] = None
        eta0: Optional[np.ndarray] = None
        dev0: Optional[float] = None
        # NON-DISCRETE only (bgam.fit): the θ that built the PREVIOUS working
        # model (mgcv ``theta0``, snapshotted at iter-end before estimate.theta,
        # bam.r:1198) — the step-halving evaluates dev0/dev1 at this θ0.
        theta0_snap: Optional[np.ndarray] = None
        # mgcv:969 — dev = sum(dev_resids) * 2 to avoid spurious convergence at iter 1.
        dev = 2.0 * float(np.sum(family.dev_resids(y, mu, prior_w)))

        eps = 1e-7
        maxit = 200          # mgcv default control$maxit
        conv = False

        rho_hat: Optional[np.ndarray] = None      # full per-penalty log-sp
        log_phi_hat: Optional[float] = None
        fit: Optional[_FitState] = None
        # Last accepted discrete-POI step (β, PP). For additive the per-iter
        # ``_fit_given_rho`` is deferred (Item 2b) and the converged fit is built
        # from this once, after the loop — skipping 6 of 7 O(n) η recomputes.
        last_out: Optional[dict] = None
        # Persistent *working* (id-linked) sp warm-start across PIRLS iters.
        # ``None`` until the first sp step; equals ``rho_hat`` slot-for-slot
        # when no smooths share an id (``_work_dim == len(slots)``).
        theta_sp_warm: Optional[np.ndarray] = None
        # Discrete-POI Newton step, carried across PIRLS iters (its last
        # element is the log-φ step the bgam.fitd:678 convergence test reads).
        Nstep: Optional[np.ndarray] = None
        n_work = self._work_dim

        for it in range(maxit):
            devold = dev

            # ---- Recompute (η, μ) at the current β (mgcv bgam.fitd:571-572):
            # ``eta <- Xbd(coef) + offset; mu <- linkinv(eta)`` so the
            # subsequent step-halving + ``estimate.theta`` see the *post-β* μ,
            # not the stale initialise'd μ (≈ y on iter 0).
            if it >= 1 and coef is not None and not additive:
                if self._discrete_design is not None:
                    eta = (Xbd(self._discrete_design,
                                np.asarray(coef, dtype=float))
                            + off)
                else:
                    eta = self._chunked_xbeta(
                        np.asarray(coef, dtype=float)) + off
                mu = link.linkinv(eta)

            # ---- Coef step-halving BEFORE the working-model build ----------
            # mgcv bgam.fitd:585-604 halves the β step (toward the previous
            # accepted β₀) while the PENALISED deviance fails to improve,
            # using the cheap deviance at the OLD θ — then builds (R,f)/W,z
            # ONCE, after halving (632-665). hea formerly built first, then
            # halved + REBUILT; reordering to mgcv's cadence (F7) means
            # ``estimate.theta`` sees the halved μ (614-630 runs after 585-604)
            # and the working model is built once. Identical on the monotone
            # path: when halving never fires μ is unchanged, so θ and the build
            # see exactly the same μ as before. ``it > 1`` == mgcv ``iter>2``
            # (1-based; c.iter=2 since hea never warm-starts ``coef``).
            #
            # PATH SPLIT: mgcv's discrete (``bgam.fitd``) and non-discrete
            # (``bgam.fit``) fitters genuinely differ in halving AND θ cadence,
            # so branch on ``discrete``:
            #   * DISCRETE — halve here (before the build), sum(rSb²), current θ,
            #     kk<30 (bgam.fitd:585-604); estimate θ HERE too (mid-iter,
            #     bgam.fitd:615), so this build sees this-iter θ.
            #   * NON-DISCRETE — halve with β'Sβ (Sl.Sb), θ0 (the PREVIOUS
            #     build's θ), kk<6 (bgam.fit:1163-1190); estimate θ at the END of
            #     the iter (below, after the conv-check + snapshot, bgam.fit:1204)
            #     so the NEXT build uses the PREVIOUS iter's θ. (hea used to
            #     impose bgam.fitd's mid-iter θ on this path → ~3e-6 fitted drift
            #     on scat, iter 10 vs mgcv 12; see test_scat_bam_simple_
            #     nondiscrete_matches_mgcv.)
            kk = 0
            discrete = self._discrete_design is not None
            if (it > 1 and coef is not None and coef0 is not None
                    and rho_hat is not None and eta0 is not None
                    and not additive):
                if discrete:
                    # mgcv bgam.fitd:578/591 — bSb0/bSb = sum(rSb²), the per-block
                    # penalty-root reduction (Sl.rSb), NOT coef·Sλ·coef; dev0 at
                    # μ₀ under the CURRENT θ (estimate.theta runs after halving).
                    bSb0 = self._bgam_rsb_penalty(rho_hat, coef0)
                    mu0 = link.linkinv(eta0)
                    dev0 = float(np.sum(family.dev_resids(y, mu0, prior_w)))
                    dev_cur = float(np.sum(family.dev_resids(y, mu, prior_w)))
                    # bgam.fitd:596 — halve while pen-dev not improving / nonfin.
                    while ((not np.isfinite(dev_cur)
                            or dev0 + bSb0
                            < dev_cur + self._bgam_rsb_penalty(rho_hat, coef))
                           and kk < 30):
                        coef = (coef0 + coef) / 2
                        eta = (eta0 + eta) / 2
                        mu = link.linkinv(eta)
                        dev_cur = float(np.sum(
                            family.dev_resids(y, mu, prior_w)))
                        kk += 1
                else:
                    # mgcv bgam.fit:1163-1190 — β'Sβ via Sl.Sb on the repara'd β
                    # at the full sp, θ0 (=``theta0_snap``, the θ that built the
                    # PREVIOUS working model) for an extended family, kk<6. Halve
                    # the coef AND the Sb vector linearly (bam.r:1181-1184).
                    if not hasattr(self, "_sl"):
                        self._sl = _sl_setup(self._slots, self.p)
                    efam = family.is_extended
                    theta_now = family.get_theta() if efam else None
                    use_t0 = efam and theta0_snap is not None
                    if use_t0:
                        family.set_theta(theta0_snap)
                    dev0 = float(np.sum(
                        family.dev_resids(y, link.linkinv(eta0), prior_w)))
                    dev1 = float(np.sum(
                        family.dev_resids(y, link.linkinv(eta), prior_w)))
                    if use_t0:
                        family.set_theta(theta_now)
                    pcoef0 = _sl_initial_repara(
                        self._sl, coef0, inverse=False,
                        both_sides=False, cov=False)
                    pcoef = _sl_initial_repara(
                        self._sl, coef, inverse=False,
                        both_sides=False, cov=False)
                    Sb0 = _sl_sb(self._sl, rho_hat, pcoef0)
                    Sb = _sl_sb(self._sl, rho_hat, pcoef)
                    while (dev0 + float(pcoef0 @ Sb0)
                           < dev1 + float(pcoef @ Sb)) and kk < 6:
                        coef = (coef0 + coef) / 2
                        pcoef = (pcoef0 + pcoef) / 2
                        eta = (eta0 + eta) / 2
                        Sb = (Sb0 + Sb) / 2
                        mu = link.linkinv(eta)
                        if use_t0:
                            family.set_theta(theta0_snap)
                        dev1 = float(np.sum(
                            family.dev_resids(y, mu, prior_w)))
                        if use_t0:
                            family.set_theta(theta_now)
                        kk += 1

            # ---- DISCRETE extended-family θ update at the (halved) μ --------
            # mgcv bgam.fitd:614-630 — estimate θ MID-iter (before the build), so
            # this iter's build sees this-iter θ. The NON-discrete path estimates
            # θ at the END of the iter instead (bam.r:1204; see below). Only fires
            # for families whose θ is free (Scat with both θ locked has
            # ``n_theta = 0`` and stays put).
            if (it >= 1 and discrete
                    and family.is_extended
                    and family.estimate_theta_callback):
                theta_new = _estimate_theta(
                    family, y, mu, scale=1.0,
                    wt=prior_w, tol=1e-7,
                )
                family.set_theta(theta_new)

            # ---- Build the working model (mgcv bgam.fitd:567 ``if (iter==1 ||
            # !additive)``) ---------------------------------------------------
            # Additive (Gaussian-identity) rebuilds only at iter 0; later iters
            # reuse the cached (R, f, X'X, X'y, ‖z‖²) and refresh dev cheaply.
            # Non-additive (PIRLS) rebuilds every iter, as W, z change.
            if it == 0 or not additive:
                # Build at the (halved) coef and the freshly-updated θ.
                if self._discrete_design is not None:
                    qr = _build_qr_discrete_pirls(
                        self._discrete_design, y, off, family,
                        coef=coef,
                        # Pass the η we already formed above (iter 0: family
                        # init; iter>0: Xbd(coef)+off, possibly step-halved) so
                        # the build reuses it instead of a second Xbd pass (mgcv
                        # forms η once, bam.r:572). Bit-identical; saves ~1
                        # Xbd/iter.
                        eta_init=eta,
                        use_chol=self._use_chol,
                        prior_w=prior_w,
                        rho=self._rho, ar_start=self._ar_start,
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

                eta = qr.eta
                mu = qr.mu
                dev = qr.dev
                # Convergence-test deviance — mgcv's two non-Gaussian fitters
                # DIFFER here, so gate on the path:
                #   * DISCRETE (``bgam.fitd:606-611``) reads the PENALISED family
                #     deviance ``dev + βᵀSλβ`` (``dev <- dev + sum(rSb^2)``),
                #     added for iter>1 (it>=1; iter==1 keeps raw ``crit<-dev``).
                #   * NON-DISCRETE (``bgam.fit:1058-1154``) converges on the RAW
                #     (unpenalised) family deviance — the penalty appears ONLY in
                #     its step-halving divergence check (bam.r:1179), never in the
                #     1154 convergence test.
                # The penalty is the per-block penalty-root reduction
                # ``sum(rSb²)`` (:meth:`_bgam_rsb_penalty`), NOT a full-matrix
                # quadratic form — matching mgcv's FP reduction so the relative-
                # change test crosses ε at mgcv's iteration. (An earlier
                # ``coef·Sλ·coef`` form had to be gated on rho!=0 because its
                # ~1e-13 reduction-order drift shifted sensitive extended-family
                # convergence; the faithful ``sum(rSb²)`` removes that drift.)
                # For discrete AR1 the penalty makes the model-B overshoot
                # DETECTABLE (it raises pen-dev while un-pen plateaus early),
                # stopping the outer loop at mgcv's sp.
                if (it >= 1 and coef is not None and rho_hat is not None
                        and self._discrete_design is not None):
                    dev = float(dev) + self._bgam_rsb_penalty(rho_hat, coef)
                if not np.isfinite(dev):
                    raise FloatingPointError(
                        f"non-finite deviance at PIRLS iter {it}"
                    )
            else:
                # Additive, iter>1: cheap penalised-deviance refresh from the
                # cached (R, f) and current coef — mgcv bgam.fitd:669
                # ``dev <- qrx$y.norm2 - sum(coef*qrx$f)``. This equals ‖z‖² −
                # βᵀX'Wz, which at the sp-update solution is ‖z−Xβ‖²_W + βᵀSβ
                # (the penalised deviance the bam.r:678 convergence test reads).
                # self._XtX/_Xty/_yty/_X_full/_wt_full/_bam_qr persist from iter
                # 0; eta/mu stay stale (unused — halving and θ are gated off).
                dev = self._yty - float(coef @ self._Xty)
                if not np.isfinite(dev):
                    raise FloatingPointError(
                        f"non-finite penalised deviance at PIRLS iter {it}"
                    )

            # Convergence (mgcv bgam.fitd:678). it>1 == mgcv iter>2 (1-based).
            # The DISCRETE path ANDs a scale-unknown clause: the log-φ Newton
            # step (last element of the previous iter's ``Nstep``) must also
            # have shrunk below ``ε·(|log φ|+1)`` — so we don't declare
            # convergence with φ̂ unsettled. The non-discrete path
            # (bgam.fit:1154) has no such clause (φ converges fully inside
            # ``_fast_reml_fit`` each iter).
            phi_conv = True
            if (self._discrete_design is not None and include_log_phi
                    and Nstep is not None and Nstep.size):
                phi_step = float(Nstep[-1])
                log_phi_now = (log_phi_hat
                               if log_phi_hat is not None else 0.0)
                phi_conv = abs(phi_step) < eps * (abs(log_phi_now) + 1.0)
            if (it > 1 and abs(dev - devold) / (0.1 + abs(dev)) < eps
                    and phi_conv):
                conv = True
                break

            # Snapshot the start-of-next-β-step reference (mgcv bgam.fitd:606-
            # 609 / bgam.fit:1196-1201 ``coef0 <- coef; eta0 <- eta``). dev0/μ0
            # are recomputed fresh from eta0 in the next iter's halving block
            # (mgcv:580/1166), so they are NOT snapshotted here. ``coef`` is this
            # iter's (post-halving) β that built the working model; the sp step
            # below overwrites it. For the non-discrete path also snapshot
            # ``theta0`` (bam.r:1198) — the θ that built THIS model, read by the
            # next iter's step-halving.
            if it > 0 and coef is not None:
                coef0 = coef.copy()
                eta0 = eta.copy()
                if (not discrete) and family.is_extended:
                    theta0_snap = np.asarray(family.get_theta(),
                                             dtype=float).copy()

            # ---- NON-DISCRETE extended-family θ update at iter END ----------
            # mgcv bgam.fit:1204-1217 estimates θ at the END of the iter (after
            # the conv-check + snapshot), so the NEXT iter's working-model build
            # uses THIS θ — i.e. each build sees the PREVIOUS iter's θ. (The
            # discrete path estimated θ mid-iter, above.) θ is fit at the current
            # μ = linkinv(eta), the post-build/post-halving μ (== bam.r:1209's
            # ``linkinv(eta)``).
            if (it >= 1 and (not discrete)
                    and family.is_extended
                    and family.estimate_theta_callback):
                theta_new = _estimate_theta(
                    family, y, mu, scale=1.0,
                    wt=prior_w, tol=1e-7,
                )
                family.set_theta(theta_new)

            # ---- sp optimisation on the current (R, f, rss_extra) -----------
            if n_sp == 0:
                rho_hat = np.zeros(0)
                log_phi_hat = None
                self.sp = np.zeros(0)
                fit = self._fit_given_rho(rho_hat)
            elif sp_user is not None:
                sp_arr = np.asarray(sp_user, dtype=float)
                if sp_arr.shape != (n_work,):
                    raise ValueError(
                        f"sp must have length {n_work} (one per estimated "
                        f"smoothing parameter; id-linked penalties share one), "
                        f"got {sp_arr.shape}"
                    )
                if np.any(sp_arr < 0):
                    raise ValueError("sp entries must be non-negative")
                rho_hat = self._rho_full(np.log(np.maximum(sp_arr, 1e-10)))
                self.sp = sp_arr
                fit = self._fit_given_rho(rho_hat)
                if include_log_phi:
                    Dp = float(fit.dev + fit.pen)
                    denom = (max(float(n - self._Mp), 1.0)
                             if method == "REML" else max(float(n), 1.0))
                    log_phi_hat = float(np.log(max(Dp / denom, 1e-300)))
            else:
                # mgcv bam's REML sp-update splits by *discrete* — port both
                # cadences exactly. CRUCIALLY both optimise the **Gaussian
                # working-model REML** on the reduced (R, f): ``Sl.fit`` /
                # ``Sl.fitChol`` treat the IRLS-linearised (R, f) as Gaussian
                # (``(nobs/γ−Mp)·log(2πφ)`` normalisation, NO family ``ls``
                # term) — the non-Gaussianness lives only in the OUTER PIRLS
                # loop that rebuilds W, z. ``_pi_fit_chol`` computes exactly
                # this.
                #
                # (1) discrete=TRUE → ``bgam.fitd`` (bam.r:706-757): a *single*
                #     ``Sl.fitChol`` Newton step on (ρ, log φ) per PIRLS iter,
                #     GRADIENT-based halving (``sum(grad·Nstep) > dev·1e-7``),
                #     warm-started ``Nstep``/sp. The branch below.
                #
                # (2) discrete=FALSE → ``bgam.fit`` (bam.r:1226-1261):
                #     ``fast.REML.fit`` run to FULL convergence on (R, f) each
                #     PIRLS iter, sp RE-SEEDED from ``initial.sp(R)`` every
                #     iter, reml-VALUE halving. ``_fast_reml_fit`` ports that
                #     loop (fast-REML.r:1740-1875), driving the SAME
                #     ``_pi_fit_chol`` evaluator. The ``elif`` after this.
                #
                # Both reach the SAME optimum (verified: mgcv bgam.fit and
                # bgam.fitd BOTH give Tweedie sp 0.258993) — cadence does not
                # change the result. The FIX for plan item P19 (scale-UNKNOWN
                # Gamma / inverse-Gaussian / fixed-p Tweedie / extended φ) is
                # using *these* (Gaussian-working-REML) optimisers instead of
                # the old ``_outer_newton``, which minimised ``_reml`` — the
                # FULL non-Gaussian REML carrying the family's ``ls0(φ)``
                # saturated-likelihood term (what mgcv-**gam** uses). On the
                # reduced (R, f) that is a DIFFERENT objective with a different
                # φ̂ optimum (Tweedie sp 0.207, not 0.259) — the divergence was
                # the WRONG OBJECTIVE, not loose convergence (the tight
                # ``√eps`` ``_fast_reml_fit`` reproduces 0.207 too when fed
                # ``_reml``; only ``_pi_fit_chol``'s Gaussian working REML
                # hits 0.259). ``_outer_newton`` now serves only
                # ``method == "GCV.Cp"`` (no REML/φ formulas) — the final
                # ``else``.
                if (method in ("REML", "ML")
                        and self._discrete_design is not None):
                    # Lazily build the shared ``Sl`` (gam's ``Sl.setup``) on
                    # first PIRLS iter — depends only on the slot S matrices,
                    # not on rho/W.
                    if not hasattr(self, "_sl"):
                        self._sl = _sl_setup(self._slots, self.p)
                    if theta_sp_warm is None:
                        # Full-space initial.spg seed → working space by least
                        # squares (mgcv mgcv.r:4617-4618); identity when no id.
                        rho0_full = self._initial_sp_rho()
                        if self._lsp0 is not None:
                            rho0_full = rho0_full - self._lsp0
                        if self._L is None:
                            rho_cur = rho0_full
                        else:
                            rho_cur, *_ = np.linalg.lstsq(
                                self._L, rho0_full, rcond=None)
                        Nstep = np.zeros(n_work + (1 if include_log_phi else 0))
                    else:
                        rho_cur = theta_sp_warm.copy()
                    if include_log_phi:
                        if log_phi_hat is None:
                            # mgcv bgam.fitd:697-702 — the iter-1 log φ SEED is
                            # ``log(var(y)·0.05)`` when ``coef`` is NULL (the
                            # standard free fit; hea never warm-starts ``coef``
                            # into this loop, so this branch always applies),
                            # NOT a working-RSS estimate. The seed steers the
                            # JOINT (sp, log φ, β) Newton trajectory, and because
                            # the discrete PIRLS stops on a step-size test
                            # (bgam.fitd:678) rather than a true fixed point, the
                            # seed changes WHERE it stops — a wrong seed lands in
                            # a different basin (sp/log φ off ~0.7%/2%). ``var``
                            # uses R's n−1 denominator. The ``coef``-supplied /
                            # ``y.norm2==0`` branches don't arise here.
                            log_phi_cur = float(np.log(
                                max(float(np.var(y, ddof=1)) * 0.05, 1e-300)
                            ))
                        else:
                            log_phi_cur = log_phi_hat
                        theta_cur = np.concatenate(
                            [rho_cur, [log_phi_cur]],
                        )
                    else:
                        theta_cur = rho_cur

                    # Newton step + halving (mgcv bam.r:669-682). ``theta_cur``/
                    # ``Nstep`` are *working* (id-linked); ``rho_try`` =
                    # ``_rho_full(working)`` feeds the full-space S build and
                    # ``_pi_fit_chol``, whose full per-penalty grad/Hessian are
                    # contracted back to working space via T = blockdiag(L, I).
                    T_pi = self._T_working(1 if include_log_phi else 0)
                    halve_max = 30
                    halves = 0
                    while True:
                        theta_try = theta_cur + Nstep
                        theta_sp_try = theta_try[:n_work]
                        rho_try = self._rho_full(theta_sp_try)
                        if include_log_phi:
                            log_phi_try = float(theta_try[n_work])
                        else:
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
                        # ``Sl.initial.repara`` (fast-REML.r:517-588,
                        # bam.r:664-665) — reparameterize XX, Xy into mgcv's
                        # well-scaled gauge (every penalty block, two-sided)
                        # so the pivoted Cholesky in ``_pi_fit_chol``
                        # factorizes the same conditioned matrix mgcv does. β
                        # comes back in the repara'd basis and gets un-rotated
                        # below. The POI step-halves on the
                        # (congruence-invariant) gradient and passes no
                        # ``ldet_S`` value, so no value correction is needed
                        # here.
                        XX_pre = _sl_initial_repara(
                            self._sl, self._XtX, both_sides=True)
                        Xy_pre = _sl_initial_repara(
                            self._sl, self._Xty, both_sides=True)
                        out = _pi_fit_chol(
                            XX_pre, Xy_pre, rho_try,
                            self._sl, self.p,
                            yy=self._yty, log_phi=log_phi_try, n=n,
                            Mp=self._Mp, gamma=self._gamma,
                            phi_fixed=not include_log_phi,
                            ldet_S_grad=ldS_grad, ldet_S_hess=ldS_hess,
                        )
                        # Undo the initial-repara on β (bam.r:759,
                        # inverse=TRUE) — the rest of the PIRLS / post-fit
                        # machinery (chunked X·β, variance, edf) operates in
                        # the original basis.
                        out["beta"] = _sl_initial_repara(
                            self._sl, out["beta"], inverse=True,
                            both_sides=False, cov=False,
                        )
                        # Contract the full per-penalty grad/Hessian to working
                        # space (g_θ = T'g, H_θ = T'HT) and recompute the step
                        # there. ``T_pi is None`` ⇔ no id linkage → reuse
                        # ``_pi_fit_chol``'s own full-space step/grad (the helper
                        # reproduces that exact value, but keep the original
                        # object for byte-identical no-id behaviour).
                        if T_pi is None:
                            grad_w = out["grad"]
                            step_w = out["step"]
                        else:
                            grad_w = T_pi.T @ out["grad"]
                            hess_w = T_pi.T @ out["hess"] @ T_pi
                            step_w = _reg_newton_step(grad_w, hess_w)
                        if float(np.max(np.abs(Nstep))) == 0.0:
                            # First call or zero step — accept and
                            # snapshot the new step for next iter.
                            Nstep = step_w
                            theta_cur = theta_try
                            break
                        # mgcv: ``sum(prop$grad * Nstep) > dev * 1e-7`` =
                        # uphill. Halve and retry.
                        if (float(np.dot(grad_w, Nstep))
                                > abs(dev) * 1e-7
                                and halves < halve_max):
                            Nstep = Nstep / 2.0
                            halves += 1
                            continue
                        Nstep = step_w
                        theta_cur = theta_try
                        break

                    theta_sp_warm = theta_cur[:n_work]
                    log_phi_hat = (float(theta_cur[n_work])
                                   if include_log_phi else None)
                    self.sp = np.exp(theta_sp_warm)          # mgcv m$sp
                    rho_hat = self._rho_full(theta_sp_warm)  # full per-penalty
                    # mgcv bgam.fitd REUSES Sl.fitChol's β / PP — the discrete
                    # POI never re-solves: ``coef <- Sl.initial.repara(prop$beta,
                    # inverse=TRUE)`` (bam.r:759) and ``PP <- Sl.initial.repara(
                    # prop$PP, inverse=TRUE, both.sides=TRUE, cov=TRUE)``
                    # (bam.r:823). ``out`` is the final accepted POI step (its
                    # ``rho_try`` == ``rho_hat``) and ``out["beta"]`` is already
                    # un-repara'd above. ``_fit_given_rho`` still supplies the
                    # gauge-invariant η/μ/dev/pen/rss + the A_chol other code
                    # paths need; we override only the gauge-DEPENDENT β and A⁻¹.
                    # For full-rank A this is identical to the re-solve (~1e-12);
                    # for rank-deficient A it adopts mgcv's pivoted-Cholesky
                    # null-space gauge instead of _fit_given_rho's ridge
                    # fallback (so coef + SE match mgcv, not just the fit).
                    #
                    # Item 2b: _fit_given_rho's only unique outputs here are the
                    # gauge-invariant η/μ/A_chol for post-fit; its β is discarded
                    # (overridden by out["beta"]). For additive (Gaussian-
                    # identity) those η/μ don't feed the loop (dev is refreshed
                    # cheaply, coef = out["beta"]), so we DEFER the full solve to
                    # ONE post-loop call — dropping its per-iter O(n)
                    # _chunked_xbeta η recompute (bam.py:2760). Non-additive
                    # keeps solving each iter (its η/μ build the next W, z).
                    last_out = out
                    if additive:
                        fit = None
                    else:
                        fit = self._fit_given_rho(rho_hat)
                        fit.beta = out["beta"]
                        fit.A_inv = _sl_initial_repara(
                            self._sl, out["PP"], inverse=True,
                            both_sides=True, cov=True,
                        )
                elif method in ("REML", "ML"):
                    # Non-discrete REML/ML → mgcv ``bgam.fit`` (bam.r:1226-1261):
                    # ``fast.REML.fit`` to FULL convergence on the current
                    # reduced (R, f). The sp is RE-SEEDED from ``initial.sp(R)``
                    # every PIRLS iter (bam.r:1229 — NOT warm-started; the
                    # converge-fully Newton makes the seed immaterial to the
                    # result), while log φ is carried forward from the previous
                    # iter's scale estimate (bam.r:1233). ``_initial_sp_rho``
                    # gives the full-space seed; ``coef(lm(lsp ~ L-1+offset))``
                    # (the lstsq) projects it to working space, identity when
                    # no smooths share an id (fast-REML.r:1768).
                    rho0_full = self._initial_sp_rho()
                    if self._lsp0 is not None:
                        rho0_full = rho0_full - self._lsp0
                    if self._L is None:
                        rho0 = rho0_full
                    else:
                        rho0, *_ = np.linalg.lstsq(
                            self._L, rho0_full, rcond=None)
                    if include_log_phi:
                        if log_phi_hat is None:
                            # mgcv bgam.fit:1232-1238 — at iter 1 ``coef`` is NULL
                            # (hea never warm-starts it into this loop), so the
                            # ``is.null(coef)`` branch fires and the seed is
                            # ``log(var(y)·0.05)`` — NOT a working-RSS estimate.
                            # ``var`` uses R's n−1 denominator. fast.REML.fit
                            # converges fully each iter so the seed only sets the
                            # Newton start, but mirror it for source fidelity (and
                            # to share the discrete path's seed, bam.py:4134).
                            log_phi0 = float(np.log(
                                max(float(np.var(y, ddof=1)) * 0.05, 1e-300)
                            ))
                        else:                       # iter > 1 — carry forward
                            log_phi0 = log_phi_hat
                        theta0 = np.concatenate([rho0, [log_phi0]])
                    else:
                        theta0 = rho0

                    theta_hat = self._fast_reml_fit(
                        theta0, include_log_phi=include_log_phi,
                    )
                    theta_sp_warm = theta_hat[:n_work]   # log φ carry-forward
                    log_phi_hat = (float(theta_hat[n_work])
                                   if include_log_phi else None)
                    self.sp = np.exp(theta_hat[:n_work])
                    rho_hat = self._rho_full(theta_hat[:n_work])
                    fit = self._fit_given_rho(rho_hat)
                    # F9: reuse fast.REML.fit's Sl.fitChol β̂ / A⁻¹ (mgcv
                    # bgam.fit:1310 Sl.postproc — no re-solve). Full-rank:
                    # identical; rank-deficient: mgcv's null-space gauge.
                    if self._reml_beta is not None:
                        fit.beta = self._reml_beta
                        fit.A_inv = self._reml_A_inv
                else:
                    # GCV.Cp only (no log φ in the outer vector): the
                    # converge-fully outer-Newton on V_g/V_u. L-aware — it
                    # optimises in working space and maps via ``_rho_full``.
                    if theta_sp_warm is None:
                        rho0_full = self._initial_sp_rho()
                        if self._lsp0 is not None:
                            rho0_full = rho0_full - self._lsp0
                        if self._L is None:
                            rho0 = rho0_full
                        else:
                            rho0, *_ = np.linalg.lstsq(
                                self._L, rho0_full, rcond=None)
                    else:
                        rho0 = theta_sp_warm.copy()

                    theta_hat = self._outer_newton(
                        rho0,
                        criterion="GCV",
                        include_log_phi=False,
                        include_family_theta=False,
                    )
                    theta_sp_warm = theta_hat[:n_work]
                    log_phi_hat = None
                    self.sp = np.exp(theta_sp_warm)
                    rho_hat = self._rho_full(theta_sp_warm)
                    fit = self._fit_given_rho(rho_hat)

            self._log_phi_hat = log_phi_hat

            # Additive deferred the per-iter solve: coef is the POI β directly
            # (== the fit.beta the non-deferred path would have copied).
            new_coef = (last_out["beta"] if (additive and fit is None)
                        else fit.beta)
            if not np.all(np.isfinite(new_coef)):
                warnings.warn(
                    f"non-finite coefficients at PIRLS iteration {it+1}",
                    stacklevel=2,
                )
                break
            coef = new_coef.copy()
        # end outer iter loop

        # Item 2b: additive deferred its per-iter _fit_given_rho; build the
        # converged fit ONCE now from the last accepted (rho_hat, POI β/PP).
        # Bit-identical to the per-iter path's *final* fit (same rho_hat, same
        # out), minus the 6 discarded intermediate solves. ``rho_hat`` /
        # ``last_out`` are from the last sp step (iter before the conv break).
        if additive and fit is None and last_out is not None:
            fit = self._fit_given_rho(rho_hat)
            fit.beta = last_out["beta"]
            fit.A_inv = _sl_initial_repara(
                self._sl, last_out["PP"], inverse=True,
                both_sides=True, cov=True,
            )

        if not conv:
            warnings.warn("PIRLS algorithm did not converge", stacklevel=2)

        if fit is None:
            raise FloatingPointError("bgam.fit produced no usable fit")

        # mgcv bgam.fitd reports the *build-point* β of the converged iter
        # (``object$coefficients <- coef``, bam.r:806): the (step-halved) β that
        # built the final working model — NOT a further model solve from it.
        # For a non-Gaussian AR1 fit the two differ. Near the optimum the
        # per-iter model solve overshoots — it raises the penalised family
        # deviance — so the step-control loop (bam.r:585-604) halves the next β
        # back toward the previous accepted iterate and convergence (678) fires
        # at that halving-stabilised *build point*, which is NOT a fixed point
        # of the solve (one solve step moves it, e.g. dev 656.94 → 657.60).
        # ``self._XtX`` and ``fit.A_inv`` are the converged working model built
        # at this ``coef``; the loop deferred ``fit.beta`` is the prior iter's
        # overshooting sp-step solve. Report ``coef`` so β / μ / the variance
        # are mutually consistent and match mgcv. For rho==0 / Gaussian the
        # iteration reaches a genuine fixed point (coef == fit.beta), so this is
        # a no-op there — only the AR1 step-halving regime exposes the gap.
        if (conv and not additive and coef is not None
                and fit.beta is not None
                and np.asarray(coef).shape == np.asarray(fit.beta).shape):
            # Report at the build-point β, not the deferred overshoot solve.
            # The loop's η = qr.eta (built from this β; Xbd is linear so the
            # step-halved η equals Xbd(β) exactly) and μ = linkinv(η) are
            # already at β, so override fit's η/μ/β together; recompute the
            # working RSS ‖f − Rβ‖² + rss_extra = yty − 2β·Xty + β·XtX·β at β
            # (the AR1-whitened working deviance the scale reads). fit.A_inv /
            # self._XtX are the converged model (built at this β), so the
            # variance / edf stay consistent. No-op for rho==0 / Gaussian.
            cf = np.asarray(coef, dtype=float)
            fit.beta = cf.copy()
            fit.eta = np.asarray(eta, dtype=float)
            fit.mu = np.asarray(mu, dtype=float)
            fit.dev = float(self._yty - 2.0 * cf @ self._Xty
                            + cf @ self._XtX @ cf)

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
        ``A⁻¹ = (X'WX + Sλ)⁻¹`` from ``fit.A_chol`` (or ``fit.A_inv`` when the
        discrete POI reuses Sl.fitChol's PP — mgcv bgam.fitd:823).
        """
        n, p = self.n, self.p
        method = self.method
        n_sp = len(self._slots)
        family = self.family
        y = self._y_arr
        beta = fit.beta
        self._rho_hat = rho_hat
        # mgcv m$full.sp — per-penalty expansion exp(L·log(sp)+lsp0).
        self.full_sp = np.exp(np.asarray(rho_hat, dtype=float))

        # A⁻¹ = (X'WX + Sλ)⁻¹. mgcv's discrete POI hands back Sl.fitChol's PP
        # (the rank-revealing pseudo-inverse, un-repara'd — bgam.fitd:823) on
        # ``fit.A_inv``; every other path leaves it None and we cho_solve the
        # (ridge-stabilised) A_chol. Identical for full-rank A.
        if fit.A_inv is not None:
            A_inv = fit.A_inv
        else:
            A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        XtWX = self._XtX                # = R'R = X'WX at converged β̂
        A_inv_XtWX = A_inv @ XtWX
        edf = np.diag(A_inv_XtWX).copy()
        edf_total = float(edf.sum())

        # Prior weights (mgcv G$w) — resolved in __init__. The PIRLS Fisher
        # weight self._wt_full already folds these in (w = w_prior·μ_η²/V), so
        # leverage uses _wt_full; scale/Pearson/null-deviance read self._wt.
        wt = self._wt
        df_resid = float(n - edf_total)

        # Scale (φ) reporting. When φ is KNOWN (scale-known family, or user
        # scale=φ), report the fixed value (mgcv G$sig2 <- scale, mgcv.r:1942).
        # When φ is ESTIMATED:
        #   * REML/ML/fREML → ``exp(log φ̂)``, the jointly REML-estimated log
        #     scale (mgcv bgam.fitd:787 ``scale <- exp(log.phi)`` for discrete,
        #     bam.r:1253 ``object$scale <- exp(fit$rho[nsp])`` for non-discrete).
        #     At the REML optimum the grad-w.r.t.-log φ stationary condition
        #     (Sl.fitChol:1647) gives ``exp(log φ̂) = rss_bSb/(n−Mp) =
        #     (dev+pen)/(n−Mp)`` with Mp the NULL-SPACE dim — NOT a raw-Pearson
        #     Σwᵢ(yᵢ−μᵢ)²/V/(n−edf) nor a working-RSS/(n−edf) statistic. Holds
        #     for discrete + non-discrete, rho==0 + AR1 alike; ``self._log_phi_hat``
        #     carries it from the fit loop (None ⇔ GCV.Cp or no joint log φ).
        #   * GCV.Cp → magic's Pearson-type ``fit$scale`` (bam.r:1291): the raw
        #     Pearson statistic over (n−edf).
        if df_resid > 0 and not self._scale_known_fit:
            if self._log_phi_hat is not None:
                pearson_scale = float(np.exp(self._log_phi_hat))
            else:
                V = family.variance(fit.mu)
                pearson_scale = float(
                    np.sum(wt * (y - fit.mu) ** 2 / V)
                ) / df_resid
        else:
            pearson_scale = (self._scale_fixed_value
                             if self._scale_known_fit else float("nan"))
        self._pearson_scale = pearson_scale
        scale = self._scale_fixed_value if self._scale_known_fit else pearson_scale
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
        # User-facing coefficient reporting (mirrors gam.py:1840-1854). The
        # inherited summary()/_se_report_for read _beta_report/_se_report; bam
        # never set them, so summary() raised AttributeError. bam doesn't drop
        # columns today (_keep_cols is None) but keep gam's full branch for
        # forward-compat with drop.intercept (P7).
        if self._keep_cols is not None:
            beta_rep = np.zeros(self._keep_cols.size)
            beta_rep[self._keep_cols] = np.asarray(beta).reshape(-1)
            se_rep = np.zeros(self._keep_cols.size)
            se_rep[self._keep_cols] = se
        else:
            beta_rep = np.asarray(beta).reshape(-1)
            se_rep = se
        self._beta_report = beta_rep
        self._se_report = se_rep
        t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        self.t_values = _row_frame(t_stats, self.column_names)
        if df_resid > 0 and np.isfinite(df_resid):
            pv = 2 * _dist.pt(np.abs(t_stats), df_resid, lower_tail=False)
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

        # Leverage / standardised residuals are NOT computed here. mgcv's
        # ``bgam.fitd`` postproc (bam.r:806-894) stores no n-length hat —
        # edf/edf1/edf2 are p-space (``diag(F)`` etc.); the per-observation
        # ``hᵢ = wᵢ·(X A⁻¹ X')ᵢᵢ`` (O(n·p²) via ``diagXVXd``) and the
        # standardised residuals built from it are deferred to the
        # ``leverage`` / ``std_*_residuals`` cached properties, computed only
        # if a diagnostic asks for them. ``_lev_w`` carries the Fisher weights.
        self._lev_w = self._wt_full
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
        # Extended-family postproc (bam.r:1322-1331): find.null.dev
        # null deviance + θ-embedding family relabel.
        self._postproc = {}
        if family.is_extended:
            pp = family.postproc(
                y, prior_weights=wt, fitted=mu,
                linear_predictors=eta, offset=self._offset,
                intercept=self._has_intercept,
            )
            self._postproc = pp
            if pp.get("null_deviance") is not None:
                self.null_deviance = float(pp["null_deviance"])

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

        # mgcv oo$rank.est (P5) — see _post_fit_gaussian. _X_full = R
        # (R'R = X'WX) with _fisher_w=None → rank reveal on R is
        # rank-equivalent to the √W·X path.
        self.rank = self._estimate_rank()

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
            # Working-space view under id linkage (see _post_fit_gaussian).
            T_aug = self._T_working(1)
            if T_aug is not None:
                H_aug = T_aug.T @ H_aug @ T_aug
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
        sc_p = 0.0 if self._scale_known_fit else 1.0
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
                # bam's _fit_given_rho returns fit.dev = the working-data RSS
                # (the reduced Gaussian-on-(R,f) problem). mgcv's fREML reports
                # the criterion on the RESPONSE deviance (bgam.fit recomputes
                # dev each PIRLS iter, bam.r:1084). _reml enters dev ONLY as
                # Dp/φ, so swap the working RSS for the response deviance
                # (self.deviance) — matches mgcv-bam's sp.criterion to ~5e-9
                # (P16). Non-Gaussian only; for Gaussian-identity the two
                # coincide so the correction is 0. The argmin is unchanged
                # (the fit is already pinned to mgcv).
                phi = float(np.exp(log_phi_hat))
                if np.isfinite(phi) and phi > 0 and np.isfinite(score):
                    score = score + (self.deviance - float(fit.dev)) / phi
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
# ``Xbd``, ``XWXd``, ``XWyd``, ``diagXVXd``) are reachable as
# ``from hea.models.bam import <name>`` per the inline convention.
# ===========================================================================


# ---------------------------------------------------------------------------
# R's Mersenne-Twister RNG and ``sample()`` — bit-exact port; lives in
# ``hea.R.rng`` (imported at the top of this module), re-exported here so
# ``from hea.models.bam import RMersenneTwister`` keeps working. mgcv's
# ``temp.seed(8547)`` + ``sample()`` calls inside ``compress.df`` /
# ``discrete.mf`` are matched bit-exactly through it.
# ---------------------------------------------------------------------------


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

    # Initial uniquecombs on raw (or vectorised) input. R5: pass the round
    # threshold so a high-cardinality (continuous) column early-exits to the
    # ``(None, None)`` sentinel instead of argsorting a unique table we are
    # about to discard by rounding (saves the ~47s continuous-by sort in 2D
    # RF; byte-identical because the round decision is the same ``>`` test).
    threshold = mm_total * mf_total
    xu_table, k_idx = _uniquecombs(work, names, max_unique=threshold)

    if xu_table is None or xu_table[names[0]].size > threshold:
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


# Span cap for the integer-lattice fast path in :func:`_unique_inverse`: above
# this the offset/bincount table would cost more memory than the sort it
# replaces, so we fall back to ``np.unique``. Pixel-coordinate lattices and
# low-level by-stimuli sit far below it (span ≤ a few hundred).
_UNIQUE_FAST_SPAN_CAP = 1 << 20


def _distinct_exceeds_1d(a: np.ndarray, threshold: int) -> bool:
    """Exact predicate: does ``a`` have **more than** ``threshold`` distinct
    values? Early-exits as soon as ``threshold + 1`` distinct are seen.

    For a high-cardinality (continuous) column the first chunk already blows
    past ``threshold`` ⇒ O(threshold), avoiding the full ``np.unique`` argsort
    over all n·m flattened entries. For a low-cardinality column it scans all
    of ``a`` in a few cheap chunked passes and returns ``False``. The result is
    exactly ``np.unique(a).size > threshold`` — same ``>`` mgcv's ``compress.df``
    uses to decide whether to round (bam.r:152), so the round/keep decision is
    byte-identical; only the *route* to that boolean is faster.
    """
    n = a.size
    if n <= threshold:
        return False
    step = max(int(threshold) + 1, 1 << 16)
    seen: Optional[np.ndarray] = None
    for s in range(0, n, step):
        u = np.unique(a[s:s + step])
        seen = u if seen is None else np.union1d(seen, u)
        if seen.size > threshold:
            return True
    return False


def _unique_inverse(col: np.ndarray, max_unique: Optional[int] = None):
    """``np.unique(col, return_inverse=True)`` with an O(n) fast path for
    low-cardinality integer-valued columns — **byte-identical** output.

    For a signal-regression / RF smooth the broadcast coordinate margins are a
    tiny integer pixel grid repeated n times (and a binary/low-level by= can be
    too); the generic ``np.unique`` argsorts all n·m flattened entries — the
    S3 hotspot (67s of the 2D fit). When the values span a small contiguous
    integer range we factorise by ``offset + bincount`` in one pass instead.

    Safety is structural: we only return the fast result if it **exactly
    reconstructs** the input (``u[inv] == col``). Because ``u`` is built from
    the sorted distinct offsets (ascending, distinct), an exact reconstruction
    is sufficient for ``(u, inv)`` to equal ``np.unique``'s output bit-for-bit
    — so any non-integer / NaN / rounding mismatch silently falls back. The
    discretisation downstream (the seeded ``compress_df`` shuffle, ``k``, the
    RNG state) is therefore provably unchanged.

    ``max_unique`` (R5): when set, the caller only needs to know whether the
    distinct count exceeds it (``compress.df`` rounds a continuous variable
    whose unique table it then **discards**). If the slow path is reached and
    ``a`` has > ``max_unique`` distinct values, return ``(None, None)`` instead
    of paying the full argsort for a table that will be thrown away. The lattice
    fast path is unaffected (it is already O(n) and returns the real table). The
    round/keep decision is identical (``_distinct_exceeds_1d`` is exact), so the
    bamT fit is byte-identical.

    Parity note — pure speedup, **not an mgcv divergence.** The non-sentinel
    output is identical to ``np.unique`` (hence to mgcv's ``uniquecombs``); only
    the *route* to it is faster, and the sentinel only fires when the result is
    about to be discarded by rounding. Safe under a parity audit.
    """
    a = np.asarray(col)
    if a.size and a.dtype.kind in "fiu":
        mn = a.min()
        mx = a.max()
        if np.isfinite(mn) and np.isfinite(mx):
            span = int(round(float(mx - mn))) + 1
            if 1 <= span <= _UNIQUE_FAST_SPAN_CAP:
                # a ∈ [mn, mx] ⇒ codes ∈ [0, span-1] (no out-of-range gather).
                codes = np.rint(a - mn).astype(np.intp)
                seen = np.zeros(span, dtype=bool)
                seen[codes] = True
                present = np.nonzero(seen)[0]              # sorted distinct
                u = (present + mn).astype(a.dtype)         # sorted unique
                remap = np.empty(span, dtype=np.int64)
                remap[present] = np.arange(present.size, dtype=np.int64)
                inv = remap[codes]
                if np.array_equal(u[inv], a):
                    return u, inv
    # Slow path (continuous / non-lattice). If the caller only needs the
    # >max_unique decision and we exceed it, skip the discarded argsort.
    if max_unique is not None and _distinct_exceeds_1d(a, max_unique):
        return None, None
    u, inv = np.unique(a, return_inverse=True)
    return u, np.asarray(inv).reshape(-1).astype(np.int64)


def _uniquecombs(work: dict[str, np.ndarray],
                 names: list[str],
                 max_unique: Optional[int] = None):
    """Numpy port of R's ``uniquecombs`` (single-thread).

    Returns ``(xu, idx)`` where ``xu`` is a dict of unique columns (in
    canonical sort order) and ``idx[i]`` is the unique-row index for
    input row ``i``.

    ``max_unique`` (R5) is honoured only for the single-column case (the path
    that hits the expensive continuous-by argsort): when the column has more
    than ``max_unique`` distinct values, return ``(None, None)`` so the caller
    can round without building a unique table it would discard. Multi-column
    keys (joint margins — always low-cardinality lattices here) ignore it.
    """
    n = next(iter(work.values())).size
    if len(names) == 1:
        col = work[names[0]]
        u, inv = _unique_inverse(col, max_unique=max_unique)
        if u is None:
            return None, None        # EXCEEDED — caller rounds, see compress_df
        return {names[0]: u}, inv
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

    Parity note — the ``by=`` is discretised here like any other marginal,
    and the fit weights the smooth by the **binned** by, NOT the raw by. This
    matches mgcv ``bam(discrete=TRUE)`` (bamT), which bins the by-variable at
    bam.r:2469-2483: ``by.var <- dk$mf[[termk]][1:dk$nr[termk]]`` (the
    discretised uniques) and applies ``by.var[dk$k[, ks_by]]`` — the bin
    representative the raw ``by[i,q]`` was rounded to (a *lossy* step for a
    continuous by). :func:`build_discrete_design` reconstructs that exact
    weight as ``by_unique[k_by]`` from this frame's ``mf``/``k``. The *exact*
    (un-binned) by lives at ``bam(discrete=FALSE)`` (bamF); a continuous by
    makes bamF ≠ bamT by ~1e-3. Do **not** substitute the raw by here or in
    ``build_discrete_design`` to "improve accuracy" — that reproduces bamF
    under the ``discrete=TRUE`` flag, a parity bug (RF1a).
    """
    if rng is None:
        rng = RMersenneTwister(8547)

    n = mf.height
    # Pre-count how many index columns ``k`` will need: each smooth term
    # contributes one slot per marginal VARIABLE plus ``(by != None)``; each
    # parametric variable contributes 1. A multi-D margin (e.g. a 2-D ad/tp
    # space margin, ``d=c(1,2)``) has >1 variable and is jointly discretised one
    # ``ik`` per variable, so counting per-margin (the old behaviour) undersizes
    # ``nr``/``ks`` and crashes; count per-variable to match.
    nk = 0
    for spec in smooth_specs:
        margins = spec.get("margins", [{"term": spec["term"]}])
        n_marg_vars = sum(len(marg["term"]) for marg in margins)
        nk += n_marg_vars + (1 if spec.get("by") not in (None, "NA") else 0)
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
        # ``by`` is processed first (matches mgcv jj==1 path). It is discretised
        # like any other marginal — including a matrix-arg by= (RF summation
        # convention), which mgcv ``bam(discrete=TRUE)`` also bins (bam.r:2469-
        # 2483). ``build_discrete_design`` then weights the smooth by the
        # *binned* by (``by_unique[k_by]``), so hea ``discrete=True`` == bamT.
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
    # absorb / by / keep_cols for the term. The constraint ``T`` (absorb /
    # keep_cols) is applied by the kernels via ``_design_constraint_Ts``; None
    # for params and unconstrained smooths. ``by`` records the by-spec for
    # reference only — the by= weighting is carried by the by-marginal that
    # :func:`build_discrete_design` prepends to ``Xd_list`` (mgcv
    # discrete.mf:261-294 / bam.r:2469-2483), not by any post-hoc column mask.
    absorb: Optional[object] = None
    by: Optional[object] = None
    keep_cols: Optional[np.ndarray] = None
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

        # by= is the FIRST marginal of the term (mgcv discrete.mf:261-269 +
        # fit-side bam.r:2469-2483): an ``m_by × 1`` basis — the discretised
        # unique by-values (numeric) or a factor-level indicator
        # (``as.numeric(by.var==by.level)``). Gathering it at the by index
        # columns reproduces the per-row by weight (summed over the matrix
        # columns for a matrix-argument by=, the signal-regression convention);
        # the smooth's own marginals follow, so the term becomes a tensor. The
        # constraint (``absorb``) still acts on the smooth's raw column space:
        # the by-marginal has ``p=1`` so it changes neither the term dimension
        # nor the column order, and ``by·(X_raw @ T) == absorb.apply(by·X_raw)``
        # for the linear ``T = absorb.apply(I)`` (mgcv ``apply.by=FALSE``).
        if spec.by is not None:
            by_name = spec.by.expr
            j_by = var_index[by_name]
            ks_by = (int(dframe.ks[j_by, 0]), int(dframe.ks[j_by, 1]))
            nr_by = int(dframe.nr[j_by])
            by_vals = np.asarray(dframe.mf[by_name][:nr_by])
            if spec.by.kind == "factor":
                by_col = (by_vals == spec.by.level).astype(float)
            else:
                by_col = by_vals.astype(float)
            Xd_list = [by_col.reshape(nr_by, 1)] + Xd_list
            k_cols = [ks_by] + k_cols

        kind = "single" if len(Xd_list) == 1 else "tensor"

        terms.append(_DiscreteTerm(
            kind=kind,
            Xd_list=Xd_list,
            k_cols=k_cols,
            coef_slice=slice(p_total, p_total + p_term),
            absorb=spec.absorb,
            by=spec.by,
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
# Per-term kernels — operate on the *unconstrained* raw column space
# (``p_raw``). Constraint application via ``T`` is layered on top in
# the public Xbd / XWyd / XWXd.
# ---------------------------------------------------------------------------


def _term_constraint_T(term: _DiscreteTerm) -> Optional[np.ndarray]:
    """Materialise the constraint matrix ``T`` (p_raw × p_post) such that
    ``X_term_post = X_term_raw @ T``.

    Returns ``None`` for the identity case (no absorb / keep_cols), so
    callers can short-circuit the multiplication. For tensor smooths
    the absorb is the rank-1 sum-to-zero Householder; for singletons
    it's the per-margin absorb chain. Both are realised here by feeding
    ``np.eye(p_raw)`` through ``term.absorb.apply`` once at term setup;
    the kernels then sandwich each raw cross-product block with ``T``
    (= mgcv's constraint ``Z``, applied post-hoc in ``XWXd0``,
    discrete.c:2230-2266).
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
            expr = (",".join(f"r{letter}" for letter in in_letters)
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


def _khatri_rao_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise tensor (Khatri-Rao) product: ``out[:, i*b+j] = A[:, i]·B[:, j]``.

    C-order over ``(i, j)`` so the column index matches mgcv's tensor column
    convention (first marginal slowest-varying — ``tensorXj``, discrete.c:301).
    """
    n, a = A.shape
    b = B.shape[1]
    return (A[:, :, None] * B[:, None, :]).reshape(n, a * b)


def _truncated_tensor(term: _DiscreteTerm, s: int,
                      k: np.ndarray, n: int) -> np.ndarray:
    """Row-tensor product of a term's *non-final* marginals at summation index
    ``s`` — the ``dXi`` working columns of mgcv ``XWXijs`` (the matrix whose
    column ``r`` is extracted by ``tensorXj``, discrete.c:1754/1828).

    Shape ``(n, Π_{l<d-1} p_l)``; a single column of ones for singletons.
    """
    d = len(term.Xd_list)
    if d == 1:
        return np.ones((n, 1), dtype=float)
    out = None
    for ell in range(d - 1):
        ks_l = term.k_cols[ell][0]
        G = term.Xd_list[ell][k[:, ks_l + s], :]      # (n, p_l)
        out = G if out is None else _khatri_rao_rows(out, G)
    return out


# Largest dense ``W̄`` (m_im·m_jm entries) we will materialise on the !acc_w
# ``indReduce`` branch before falling back to the per-column factor path. Bounds
# the table to ~32 MB (f64); signal-regression marginals sit well under it. MUST
# match ``XWX_DENSE_MSIZE_CAP`` in rust/src/discrete.rs (the rust kernel and this
# numpy spec take the same branch so ``rs == python`` holds).
_XWX_DENSE_MSIZE_CAP = 4_000_000

# Empty ``woff`` sentinel for the rust ``xwx_smooth_block`` kernel: non-AR1 blocks
# pass it to signal the plain ``diag(w)`` weight (no tridiagonal super/sub).
_XWX_EMPTY_F64 = np.empty(0, dtype=float)


def _wbar_contract(Ki_list: list[np.ndarray], Kj_list: list[np.ndarray],
                   vals_list: list[np.ndarray],
                   Xim: np.ndarray, Xjm: np.ndarray) -> np.ndarray:
    """Contract ``Xd_im' W̄ Xd_jm`` where ``W̄[a,b] = Σ vals·[K_i=a][K_j=b]``
    (summed over the summation-convention sets ``(K_i, K_j, vals)``).

    Three faithful paths from mgcv ``XWXijs`` (discrete.c:1793-2025), none
    forming the ``n×p`` design:

    * ``n > m_im·m_jm`` (mgcv ``acc_w``, 1801 — STRICT ``>``): accumulate the
      dense ``m_im×m_jm`` table by flat bincount, then ``Xd_im' W̄ Xd_jm``.
    * ``acc_w=0`` but ``min(p_im,p_jm) > 15`` and ``m_im·m_jm`` within the dense
      cap: mgcv's ``indReduce`` sparse branch (1884, 1922). indReduce collapses
      the ``(K_i,K_j)`` duplicates (hash table) then forms the product; the flat
      bincount into ``W̄`` IS that dedup, and one BLAS ``Xd_im' W̄ Xd_jm``
      replaces the per-column scatter loop below (measured 14-63× on
      signal-regression blocks, within the contraction's dgemm floor).
    * otherwise (``acc_w=0``, small ``p`` or marginals past the cap): mgcv's
      DIRECT accumulation (1924-2006) — form the smaller factor (``rfac`` cost
      choice, 1810) ``C = W̄ Xd_jm`` (m_im × p_jm) or ``D = W̄' Xd_im`` by one
      bincount per column, then ``Xd_im' C`` / ``D' Xd_jm``. Never builds m×m.
    """
    mim, pim = Xim.shape
    mjm, pjm = Xjm.shape
    msize = mim * mjm
    nst = len(Ki_list)              # summation-convention sets s_i·s_j (×3 for AR1)
    n = Ki_list[0].shape[0]
    # mgcv acc_w (1801, STRICT >) OR the !acc_w large-p indReduce branch (1884):
    # both collapse (K_i,K_j) into the dense W̄ then contract once. Two guards on
    # the !acc_w case so the dense table never costs more than the factor path it
    # replaces: ``msize ≤ cap`` (absolute memory) and ``msize ≤ 16·nst·n`` (the W̄
    # scan stays under the per-column factor work — without it a few rows on a
    # huge grid would scan an msize-sized table; mgcv's hash is O(n_u) there).
    if n > msize or (min(pim, pjm) > 15 and msize <= _XWX_DENSE_MSIZE_CAP
                     and msize <= 16 * nst * n):
        Wflat = np.zeros(msize, dtype=float)
        for Ki, Kj, vals in zip(Ki_list, Kj_list, vals_list):
            Wflat += np.bincount(Ki * mjm + Kj, weights=vals, minlength=msize)
        return Xim.T @ Wflat.reshape(mim, mjm) @ Xjm
    # Form the smaller factor (mgcv ``rfac`` cost choice, discrete.c:1810):
    # ``C = W̄ Xd_jm`` (m_im × p_jm) or ``D = W̄' Xd_im`` (m_jm × p_im) — one
    # bincount per column of the chosen factor, never an m_im×m_jm table.
    if pjm <= pim:
        C = np.zeros((mim, pjm), dtype=float)
        for Ki, Kj, vals in zip(Ki_list, Kj_list, vals_list):
            WX = vals[:, None] * Xjm[Kj]
            for c in range(pjm):
                C[:, c] += np.bincount(Ki, weights=WX[:, c], minlength=mim)
        return Xim.T @ C
    D = np.zeros((mjm, pim), dtype=float)
    for Ki, Kj, vals in zip(Ki_list, Kj_list, vals_list):
        WX = vals[:, None] * Xim[Ki]
        for c in range(pim):
            D[:, c] += np.bincount(Kj, weights=WX[:, c], minlength=mjm)
    return D.T @ Xjm


def _param_smooth_block(pterm: _DiscreteTerm, sterm: _DiscreteTerm,
                        w: np.ndarray, k: np.ndarray, n: int,
                        w_off: Optional[np.ndarray] = None) -> np.ndarray:
    """Raw block ``X_param' W X_smooth`` (shape ``p_par × pt_smooth``).

    mgcv ``XWXijs`` forms the left factor ``D = W̄' X_param`` by direct
    accumulation when one term has ``m==n`` (discrete.c:1965-2006; the
    ``mim==n ⇒ rfac=0`` guard at 1811 guarantees no ``n×p`` product). Each
    parametric column ``a`` is fed as the "y" of ``X_smooth'(W·X_param[:,a])``
    — the same scatter as ``XWyd``. For the AR1 ``tri`` weight (``w_off`` given)
    ``W·X_param[:,a]`` is the tridiagonal matvec :func:`_tri_matvec`.
    """
    Xp = pterm.Xd_list[0]
    p_par = Xp.shape[1]
    WXp = (w[:, None] * Xp) if w_off is None else _tri_matvec(w, w_off, Xp)
    rows = [_term_Xty_raw(sterm, WXp[:, a], k, n) for a in range(p_par)]
    return np.vstack(rows)


def _smooth_smooth_block(ti: _DiscreteTerm, tj: _DiscreteTerm,
                         w: np.ndarray, k: np.ndarray, n: int,
                         w_off: Optional[np.ndarray] = None) -> np.ndarray:
    """Raw block ``X_i' W X_j`` for two smooth terms (shape ``pt_i × pt_j``).

    Faithful port of mgcv ``XWXijs`` general case (discrete.c:1793-2027):
    decompose each tensor term into (non-final row-tensor) ⊗ (final marginal);
    for each sub-block ``(r,c)`` accumulate the final-marginal weight table
    ``W̄[a,b] = Σ_{s,t,rows} w·dXi_r·dXj_c·[K_i=a][K_j=b]`` then contract
    ``Xd_im' W̄ Xd_jm``. The ``n×pt`` term design is never materialised.

    For the AR1 ``tri`` weight (``w_off`` given, length ``n-1``) each ``(s,t)``
    contributes THREE scatters into ``W̄`` (mgcv XWXijs ``tri`` branches,
    discrete.c:1843-1881) — the diagonal plus the super/sub couplings::

        diag : (K_i[l],   K_j[l])   += w[l]·dXi[l]·dXj[l]          l=0..n-1
        super: (K_i[l],   K_j[l+1]) += w_off[l]·dXi[l]·dXj[l+1]    l=0..n-2
        sub  : (K_i[l+1], K_j[l])   += w_off[l]·dXi[l+1]·dXj[l]    l=0..n-2

    :func:`_wbar_contract` already contracts an arbitrary list of scatter
    triples, so the only ``tri`` change here is emitting the extra two per
    ``(s,t)`` (and routing every block through it — the ``si==1`` diagonal
    shortcut and the rust kernel assume a diagonal ``W̄`` and so are skipped).
    """
    Xim = ti.Xd_list[-1]
    Xjm = tj.Xd_list[-1]
    mim, pim = Xim.shape
    mjm, pjm = Xjm.shape
    ks_im = ti.k_cols[-1][0]
    si = ti.k_cols[-1][1] - ks_im
    ks_jm = tj.k_cols[-1][0]
    sj = tj.k_cols[-1][1] - ks_jm
    ndi = int(np.prod([Xd.shape[1] for Xd in ti.Xd_list[:-1]])) if len(ti.Xd_list) > 1 else 1
    ndj = int(np.prod([Xd.shape[1] for Xd in tj.Xd_list[:-1]])) if len(tj.Xd_list) > 1 else 1

    diag_term = ti is tj           # same term ⇒ K_i and K_j are the same columns
    TTi = [_truncated_tensor(ti, s, k, n) for s in range(si)]
    TTj = TTi if diag_term else [_truncated_tensor(tj, t, k, n) for t in range(sj)]

    # Rust runs the (r,c)×(s,t)×rows accumulation in one tight pass — the
    # signal-regression / tensor case where numpy's per-(s,t) bincount loop is
    # call-overhead bound. The plain single×single off-diagonal (one bincount /
    # small factor) and the non-AR1 si==1 diagonal shortcut already beat mgcv in
    # numpy, so they stay; rust handles everything with a tensor or summation
    # axis, including the AR1 ``tri`` weight (it does the diag + super/sub
    # scatters internally — the ``si==1`` shortcut never applies under AR1, where
    # ``W̄`` is tridiagonal not diagonal, so that exclusion is gated on non-AR1).
    is_general = si > 1 or sj > 1 or ndi > 1 or ndj > 1
    ar1 = w_off is not None
    if (_rs_xwx_smooth_block is not None and is_general
            and not (diag_term and si == 1 and not ar1)):
        TTi3 = np.ascontiguousarray(np.stack([t.T for t in TTi]))   # (si, ndi, n)
        TTj3 = (TTi3 if diag_term
                else np.ascontiguousarray(np.stack([t.T for t in TTj])))
        Ki = np.ascontiguousarray(
            np.stack([k[:, ks_im + s] for s in range(si)]).astype(np.int64))
        Kj = (Ki if diag_term else np.ascontiguousarray(
            np.stack([k[:, ks_jm + t] for t in range(sj)]).astype(np.int64)))
        woff_arg = (np.ascontiguousarray(w_off, dtype=float) if ar1
                    else _XWX_EMPTY_F64)
        return _rs_xwx_smooth_block(
            np.ascontiguousarray(Xim), np.ascontiguousarray(Xjm),
            Ki, Kj, TTi3, TTj3, np.ascontiguousarray(w), woff_arg, diag_term)

    Ki_all = [k[:, ks_im + s].astype(np.int64) for s in range(si)]
    Kj_all = [k[:, ks_jm + t] for t in range(sj)]
    block = np.zeros((ndi * pim, ndj * pjm), dtype=float)
    for r in range(ndi):
        c0 = r if diag_term else 0          # symmetric term ⇒ only upper sub-blocks
        for c in range(c0, ndj):
            if diag_term and si == 1 and w_off is None:
                # Same final marginal ⇒ K_i ≡ K_j ⇒ W̄ is diagonal; skip the
                # m_im×m_jm table (mgcv simple branch, discrete.c:1742-1792).
                # (The AR1 ``tri`` super/sub couplings break this — w_off path
                # routes through the general three-scatter contraction.)
                wb = np.bincount(k[:, ks_im],
                                 weights=w * TTi[0][:, r] * TTj[0][:, c],
                                 minlength=mim)
                sub = (Xim * wb[:, None]).T @ Xjm
            else:
                Ki_list = []
                Kj_list = []
                vals_list = []
                for s in range(si):
                    Ki = Ki_all[s]
                    dXi = TTi[s][:, r]
                    for t in range(sj):
                        Kj = Kj_all[t]
                        dXj = TTj[t][:, c]
                        Ki_list.append(Ki)
                        Kj_list.append(Kj)
                        vals_list.append(w * dXi * dXj)
                        if w_off is not None:
                            # super: (K_i[l], K_j[l+1]) += w_off·dXi[l]·dXj[l+1]
                            Ki_list.append(Ki[:-1])
                            Kj_list.append(Kj[1:])
                            vals_list.append(w_off * dXi[:-1] * dXj[1:])
                            # sub: (K_i[l+1], K_j[l]) += w_off·dXi[l+1]·dXj[l]
                            Ki_list.append(Ki[1:])
                            Kj_list.append(Kj[:-1])
                            vals_list.append(w_off * dXi[1:] * dXj[:-1])
                sub = _wbar_contract(Ki_list, Kj_list, vals_list, Xim, Xjm)
            block[r * pim:(r + 1) * pim, c * pjm:(c + 1) * pjm] = sub
            if diag_term and c > r:
                block[c * pim:(c + 1) * pim, r * pjm:(r + 1) * pjm] = sub.T
    return block


def _tri_matvec(w: np.ndarray, w_off: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``W_eff · V`` for the symmetric tridiagonal ``W_eff`` (diagonal ``w``,
    off-diagonal ``w_off``), ``V`` an ``n`` vector or ``n×c`` matrix.

    ``(W_eff V)[i] = w[i]·V[i] + w_off[i]·V[i+1] + w_off[i-1]·V[i-1]`` — the
    dense form of the ``tri`` weight applied to a full-length column (mgcv
    forms the same product implicitly in XWXijs' dense/direct branches).
    """
    col = V.ndim == 2
    wb = w[:, None] if col else w
    wo = w_off[:, None] if col else w_off
    out = wb * V
    out[:-1] += wo * V[1:]        # super: w_off[i]·V[i+1]
    out[1:] += wo * V[:-1]        # sub:   w_off[i-1]·V[i-1]
    return out


def _term_pair_XWX_raw(ti: _DiscreteTerm, tj: _DiscreteTerm,
                       w: np.ndarray, k: np.ndarray, n: int,
                       w_off: Optional[np.ndarray] = None) -> np.ndarray:
    """Raw (pre-constraint) cross-product block ``X_i' W X_j`` for one term
    pair, dispatching on parametric (``m==n``) vs smooth (compressed) terms —
    mgcv ``XWXijs`` (discrete.c:1672).

    ``w_off`` (length ``n-1``) is the AR1 tridiagonal off-diagonal; ``None`` is
    the plain ``diag(w)`` weight.
    """
    pi = ti.kind == "param"
    pj = tj.kind == "param"
    if pi and pj:
        Xi = ti.Xd_list[0]
        Xj = tj.Xd_list[0]
        WXj = (w[:, None] * Xj) if w_off is None else _tri_matvec(w, w_off, Xj)
        return Xi.T @ WXj
    if pi:
        return _param_smooth_block(ti, tj, w, k, n, w_off=w_off)
    if pj:
        return _param_smooth_block(tj, ti, w, k, n, w_off=w_off).T
    return _smooth_smooth_block(ti, tj, w, k, n, w_off=w_off)


# ---------------------------------------------------------------------------
# Public kernels
# ---------------------------------------------------------------------------


def Xbd(design: DiscreteDesign, beta: np.ndarray) -> np.ndarray:
    """Compute ``X β`` on the compressed design — per-term scatter-add only.

    Direct port of mgcv ``Xbd`` (src/discrete.c:502-572): lift each term's
    post-constraint β to raw space (``β_raw = T·β_post``) and accumulate
    ``_term_Xb_raw`` into the n-vector η. The ``n×p`` design is never formed
    (mgcv ``discrete=TRUE`` scatter-adds on the compressed Xd/k grid).
    """
    beta = np.asarray(beta, dtype=float)
    n = design.n
    eta = np.zeros(n, dtype=float)
    Ts = _design_constraint_Ts(design)
    for term, T in zip(design.terms, Ts):
        b_post = beta[term.coef_slice]
        b_raw = b_post if T is None else (T @ b_post)
        eta += _term_Xb_raw(term, b_raw, design.k, n)
    return eta


def XWXd(design: DiscreteDesign, w: np.ndarray,
        ar_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute ``X' W X`` (``p × p``, post-constraint) on the compressed design.

    Direct port of mgcv ``XWXd0``/``XWXijs`` (src/discrete.c:1672-2273): for
    each term pair ``(i ≤ j)`` form the raw cross-product block via
    :func:`_term_pair_XWX_raw` (final-marginal weight-table scatter — never the
    ``n×p`` design), then apply the term constraints. mgcv applies the
    constraint post-hoc to the raw block column-by-column then row-by-row with
    ``Ztb`` (discrete.c:2230-2266); hea applies the equivalent dense constraint
    matrix ``T`` (= ``Z``) as ``T_i' B_raw T_j``. The upper triangle is mirrored
    to the lower (mgcv ``up2lo``, discrete.c:2269).

    ``ar_weights`` (the length-``2n-1`` ``ar.weight`` array) selects the AR1
    error model: ``W`` becomes the symmetric tridiagonal ``D·Tᵀ·T·D`` built by
    :func:`_ar1_tri_weight` (discrete.c:2143-2156), and the ``tri`` super/sub
    couplings are scattered alongside the diagonal in the block kernels
    (XWXijs ``tri`` branches). ``ar_weights=None`` is the plain ``diag(w)``.
    """
    w = np.asarray(w, dtype=float)
    n = design.n
    p = design.p
    if ar_weights is None:
        w_off = None
    else:
        w, w_off = _ar1_tri_weight(w, ar_weights)
    XWX = np.zeros((p, p), dtype=float)
    Ts = _design_constraint_Ts(design)
    terms = design.terms
    nt = len(terms)
    for i in range(nt):
        sl_i = terms[i].coef_slice
        Ti = Ts[i]
        for j in range(i, nt):
            raw = _term_pair_XWX_raw(terms[i], terms[j], w, design.k, n,
                                     w_off=w_off)
            Tj = Ts[j]
            blk = raw if Ti is None else (Ti.T @ raw)
            if Tj is not None:
                blk = blk @ Tj
            sl_j = terms[j].coef_slice
            XWX[sl_i, sl_j] = blk
            if i != j:
                XWX[sl_j, sl_i] = blk.T
    return XWX


def XWyd(design: DiscreteDesign, w: np.ndarray, y: np.ndarray,
        ar: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        ) -> np.ndarray:
    """Compute ``X' (W · y)`` on the compressed design — scatter-add only.

    Direct port of mgcv ``XWyd``/``singleXty``/``tensorXty`` (src/discrete.c:
    329-1186): scatter-add ``W·y`` into the per-term m-grouped weight tensor,
    contract against every marginal ``Xd`` to land in raw coefficient space,
    then apply ``T'`` to the post-constraint slot. The ``n×p`` design is never
    materialised.

    For ``ar = (stop, row, weight)`` (the AR1 error model, discrete.c:
    1109-1157) the effective weight is the tridiagonal ``W = D·Tᵀ·T·D``
    (``D = diag(√w)``, ``T`` the rwMatrix whitening transform): mgcv forms
    ``Wy = D·Tᵀ·T·D·y`` as a dense n-vector via two :func:`_rw_matrix`
    passes (forward then transpose) bracketed by ``√w``, then scatters
    ``X'·Wy``. ``ar=None`` is the plain ``X'(w·y)`` diagonal path.
    """
    w_arr = np.asarray(w, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = design.n
    p = design.p
    if ar is None:
        wy = w_arr * y_arr
    else:
        # mgcv discrete.c:1110,1152-1156 — Wy = D·Tᵀ·T·D·y, D = diag(√w).
        stop, row, weight = ar
        sw = np.sqrt(w_arr)
        wy = sw * y_arr
        wy = _rw_matrix(stop, row, weight, wy, trans=False)
        wy = _rw_matrix(stop, row, weight, wy, trans=True)
        wy = sw * wy
    Xy = np.zeros(p, dtype=float)
    Ts = _design_constraint_Ts(design)
    for term, T in zip(design.terms, Ts):
        Xty_raw = _term_Xty_raw(term, wy, design.k, n)
        if T is None:
            Xy[term.coef_slice] = Xty_raw
        else:
            Xy[term.coef_slice] = T.T @ Xty_raw
    return Xy


def diagXVXd(design: DiscreteDesign, V: np.ndarray) -> np.ndarray:
    """Compute ``diag(X V X')`` (length ``n``) on the compressed design.

    Direct port of mgcv ``diagXVXt`` (src/discrete.c:629-756), the kernel
    behind R's ``diagXVXd``. For each column ``kk`` of the ``p×p`` matrix
    ``V``::

        (XV)[:, kk] = X · V[:, kk]      # one column of X·V, via Xbd
        X[:, kk]    = X · e_kk          # the kk-th column of X, via Xbd(e_kk)
        diag       += (XV)[:, kk] · X[:, kk]

    summed over ``kk`` gives ``diag(X V X')``. mgcv forms each X column on the
    fly through ``Xbd`` (line 743-749) so the ``n×p`` design is NEVER
    materialised — this is the post-fit / predict de-materialisation kernel
    (used for the hat diagonal ``w·diag(X A⁻¹ X')`` and the prediction SE
    ``diag(X Vp X')``). The kernels apply the term constraint (``T``), so the
    columns are post-constraint, exactly as mgcv's ``Xbd`` with ``v``/``qc``.

    Term-subset selection (mgcv ``rs``/``cs`` for ``predict`` terms/iterms) is
    not yet wired — callers needing it stay on the full design for now.
    """
    n, p = design.n, design.p
    V = np.asarray(V, dtype=float)
    diag = np.zeros(n, dtype=float)
    e = np.zeros(p, dtype=float)
    for kk in range(p):
        xv = Xbd(design, V[:, kk])   # (X·V)[:, kk]
        e[kk] = 1.0
        xi = Xbd(design, e)          # X[:, kk]
        e[kk] = 0.0
        diag += xv * xi
    return diag
