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

mgcv source: ``mgcv/R/bam.r`` (1.9-4).
"""

from __future__ import annotations

import copy as _copy
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .._dispatch import rs_fn
from .._rfma import _rfma, _rfma_vec
from .._rlinalg import (
    r_backsolve,
    r_chol,
    r_chol2inv,
    r_crossprod,
    r_matprod,
    r_matprod_safe,
    r_norm,
    r_qr,
    r_qr_coef,
    r_qr_lapack,
    r_qr_qy_lapack,
    r_qr_R,
    r_tcrossprod,
)
from .._rsum import r_all_equal, r_colsums, r_mean, r_rowsums, r_sum, r_var
from ..family import (
    DiscreteX,
    Family,
    Gaussian,
    GeneralFamily,
    _coerce_response,
    betar,
    ocat,
    ziP,
)
from ..family import (
    gfam as _gfam_family,
)
from ..family import (
    negbin as _negbin_family,
)
from ..family import (
    tw as _tw_family,
)
from ..formula import (
    BasisSpec,
    Name,
    SmoothBlock,
    Term,
    _AbsorbTransform,
    _apply_smooth_arg_exprs,
    _collect_name_idents,
    _encode_term_marginals,
    _eval_atom,
    _factor_levels,
    _FSRawBasis,
    _is_factor_like,
    _LinearTransformRawBasis,
    _RawBasis,
    _smooth_arg_expr_map,
    _smooth_by_expr,
    _smooth_id_value,
    _smooth_sp_value,
    _smooth_term_vars,
    _SZRawBasis,
    _TensorRawBasis,
    is_matrix_col,
    materialize,
    materialize_smooths,
    matrix_to_2d,
    normalize_data,
    prepare_design,
)
from ..R import distributions as _dist
from ..R.rng import RMersenneTwister
from . import _xwxd as _xwxd_py
from ._magic import magic, magic_post_proc
from .gam import (
    _add_null_space_penalties,
    _apply_gam_side,
    _block_s_scale,
    _cbind_family_stash,
    _cbind_response_intake,
    _FitState,
    _gam_fit3_score,
    _initial_sp,
    _ldet_s,
    _lm_on_L,
    _mgcv_mmult,
    _mgcv_pcrossprod,
    _MultiDesign,
    _na_pass_expand,
    _parametric_penalty,
    _ParaPenBlock,
    _penalty_rank,
    _penalty_root_of,
    _PenaltySlot,
    _pls_qr,
    _preml_hessian,
    _r_eigen_sym,
    _R_rank,
    _row_frame,
    _Sl,
    _sl_initial_repara,
    _sl_mult,
    _sl_repara,
    _sl_setup,
    _sl_term_mult,
    _SlBlock,
    _suffix_smooth_label,
    _sym_rank,
    _total_penalty_space,
    gam,
    gam_control,
)

__all__ = ["bam"]

_rs_xwx_smooth_block = rs_fn("xwx_smooth_block")

# Rust accelerator for the weighted bin accumulation (mgcv discrete.c:1786/327).
# ``None`` when unavailable — :func:`_bin_accum` then falls back to np.bincount.
_rs_bin_accum = rs_fn("bin_accum")

# Rust accelerator for ``rwMatrix`` (mgcv misc.c:710-748) — the AR1 row-recombine.
# ``None`` when unavailable — :func:`_rw_matrix` then runs the numpy fallback.
_rs_rw_matrix = rs_fn("rw_matrix")

_rs_reml_pmmult = rs_fn("reml_pmmult")


def _pmmult(
    a: np.ndarray, b: np.ndarray, at: bool = False, bt: bool = False
) -> np.ndarray:
    """``op(a) @ op(b)`` (``op`` = transpose if the flag is set) with a FIXED
    reduction order — mgcv ``mgcv_mmult`` (mat.c:431) semantics, but pinned so
    the REML Hessian is deterministic. Routes to the Rust kernel
    (:data:`_rs_reml_pmmult`); the fallback is ``einsum(optimize=False)`` —
    numpy's own in-order loop, NOT the threaded-BLAS ``@`` — so the Python path
    is deterministic too."""
    if _rs_reml_pmmult is not None:
        return np.asarray(
            _rs_reml_pmmult(
                np.ascontiguousarray(a, dtype=float),
                np.ascontiguousarray(b, dtype=float),
                at,
                bt,
            )
        )
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


def _rw_matrix(
    stop: np.ndarray,
    row: np.ndarray,
    weight: np.ndarray,
    X: np.ndarray,
    trans: bool = False,
) -> np.ndarray:
    """Recombine rows of ``X`` per ``stop``/``row``/``weight``.

    Direct port of mgcv ``rwMatrix`` (C kernel src/misc.c:710-748; R wrapper
    bam.r:18-29). Forward (``trans=False``): the ith output row is
    ``Σ_{k ∈ ind_i} weight[k] · X[row[k], :]`` where ``ind_i = 1:stop[1]`` if
    ``i==1`` else ``(stop[i-1]+1):stop[i]``. Transposed (``trans=True``):
    ``out[row[k], :] += weight[k] · X[i, :]`` over the same (i, k) pairs — the
    scatter adjoint. BOTH are exercised by the AR1 X'Wy / y-norm² path
    (:func:`XWyd`, discrete.c:1152-1156 calls rwMatrix forward then transpose).

    mgcv's C accumulates ``*X1p += weight * *Xp`` in one expression, which
    fuses to ``fma`` on arm64; the Rust kernel (:data:`_rs_rw_matrix`) and
    the Python loop below both fuse it.

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
    # misc.c:731-745: i outer, its indices k inner, each column updated
    # ``*X1p += weight * *Xp`` (one fused multiply-add on arm64)
    out = np.zeros((n, p), dtype=float)
    start = 0
    for i in range(n):
        end = int(stop[i]) + 1
        for k in range(start, end):
            rk, w = int(row[k]), float(weight[k])
            src, dst = (i, rk) if trans else (rk, i)
            out[dst] = _rfma_vec(w, X[src], out[dst])
        start = end
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
    """
    parts: list[np.ndarray] = []
    k = 0
    for blk in sl.blocks:
        ind = blk.pen_cols()  # (start:stop)[ind], absolute
        if blk.n_sp == 1:  # singleton — multiple of I
            parts.append(beta[ind] * np.exp(rho_full[k] / 2.0))
            k += 1
        else:  # multi-S block
            for j in range(blk.n_sp):
                bS = r_matprod(np.reshape(beta[ind], (1, -1)), np.asarray(blk.rS[j]))[0]
                parts.append(np.exp(rho_full[k] / 2.0) * bS)
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
    """
    a = np.zeros_like(beta, dtype=float)
    k = 0
    for blk in sl.blocks:
        ind = blk.pen_cols()  # (start:stop)[ind], absolute
        if blk.n_sp == 1:  # singleton — multiple of I
            a[ind] = a[ind] + beta[ind] * np.exp(rho_full[k])
            k += 1
        else:  # multi-S block
            for j in range(blk.n_sp):
                Sb_j = r_matprod(
                    np.asarray(blk.S[j], dtype=float), beta[ind].reshape(-1, 1)
                )
                a[ind] = a[ind] + np.exp(rho_full[k]) * Sb_j[:, 0]
                k += 1
    return a


def _estimate_theta(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    *,
    scale: float = 1.0,
    wt: np.ndarray | None = None,
    tol: float = 1e-7,
) -> np.ndarray:
    """Inner Newton on the family's extra parameters θ at fixed (y, μ).
    Direct port of mgcv ``estimate.theta`` (R/efam.r:5-96). Used inside
      ``max(λ)·1e-5`` (mgcv R/efam.r:60-64) to make H positive-def.
    * Step ``= -H⁻¹·g`` capped to ``|step|_∞ ≤ 4`` (R/efam.r:69-70).
    * Step halving (≤ 25 iters) while uphill (R/efam.r:75-82).
    * Outer iters capped at 100 (R/efam.r:57).
    """
    # mgcv efam.r:12 ``if (!inherits(family,"extended.family")) stop`` —
    # mgcv's extended.family set is hea's ``is_extended`` families PLUS
    # tw (hea's flag gates bam's bgam.fitd Newton branch only; tw is
    # mgcv-extended but takes the fitd θ-in-outer route there).
    if not (family.is_extended or isinstance(family, _tw_family)):
        raise ValueError(
            f"_estimate_theta called with non-extended family {type(family).__name__}"
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
    if scale < 0:
        if r_all_equal(y, mu):
            log_phi0 = float(np.log(r_var(y) * 0.1))
        else:
            V = family.variance(mu)
            log_phi0 = float(np.log(r_mean((y - mu) ** 2 / V)))
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
        dev = (
            float(r_sum(family.dev_resids(y, mu, wt, theta=theta_for_family)))
            / scale_eval
        )
        if deriv > 0:
            Dd = family.Dd(y, mu, theta_for_family, wt, level=deriv)
        ls = family.ls_extended(y, wt, theta=theta_for_family, scale=scale_eval)
        nll = dev / 2.0 - float(ls["ls"])

        if deriv > 0:
            # mgcv efam.r:28 ``colSums(as.matrix(Dd$Dth))`` — R's
            # as.matrix maps a length-n vector (1-θ families: nb, tw)
            # to an n×1 COLUMN; np.atleast_2d would give the 1×n
            # transpose and a length-n "gradient".
            Dth = np.asarray(Dd["Dth"], dtype=float)
            if Dth.ndim == 1:
                Dth = Dth.reshape(-1, 1)
            g1 = r_colsums(Dth) / (2.0 * scale_eval)
            if get_scale:
                g = np.concatenate([g1, [-dev / 2.0]])
            else:
                g = g1.copy()
            ind = slice(0, g.shape[0])
            g = g - np.atleast_1d(ls["lsth1"])[ind]
        else:
            g = None

        if deriv > 1:
            # mgcv efam.r:34 ``colSums(as.matrix(Dd$Dth2))`` — same
            # as.matrix column semantics as Dth above.
            Dth2_packed = np.asarray(Dd["Dth2"], dtype=float)
            if Dth2_packed.ndim == 1:
                Dth2_packed = Dth2_packed.reshape(-1, 1)
            xs = r_colsums(Dth2_packed) / (2.0 * scale_eval)
            Dth2 = np.zeros((nth, nth), dtype=float)
            iu, ju = np.triu_indices(nth)
            Dth2[iu, ju] = xs[: iu.size]
            Dth2[ju, iu] = xs[: iu.size]
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
            evals, evecs = _r_eigen_sym(H_act)
            pdef = not bool(np.any(evals <= 0.0))
            if not pdef:
                # mgcv R/efam.r:60-64: |λ| floored at max(|λ|)*1e-5
                evals = np.abs(evals)
                thresh = float(evals.max()) * 1e-5 if evals.size > 0 else 0.0
                evals = np.where(evals < thresh, thresh, evals)
            step0 = _eigen_step(evecs, evals, g[uconv])
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
    """
    A = A * 1.0  # force copy (Sl.addS:1021)
    sp = np.exp(np.asarray(rho, dtype=float))
    k = 0
    for blk in sl.blocks:
        ind = blk.pen_cols()  # (start:stop)[ind]
        if blk.n_sp == 1:  # singleton — multiple of I
            A[ind, ind] += sp[k]
            k += 1
        else:  # multi-S block
            for j in range(blk.n_sp):
                A[np.ix_(ind, ind)] += sp[k] * blk.S[j]
                k += 1
    return A


def _d_det_xxs(sl: _Sl, PP: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mgcv ``d.detXXS`` (fast-REML.r:1329-1367): first/second ρ-derivatives
    of ``log|X'X+Sλ|`` given the (unpivoted) inverse-Hessian ``PP = A⁻¹``.
    """
    SA, inds = _sl_term_mult(sl, PP, full=False)
    nd = len(SA)
    d1 = np.zeros(nd)
    d2 = np.zeros((nd, nd))
    for i in range(nd):
        indi = inds[i]
        d1[i] = r_sum(np.diag(SA[i][:, indi]))
        for j in range(i, nd):
            indj = inds[j]
            v = -r_sum(SA[i][:, indj].T * SA[j][:, indi])
            d2[i, j] = d2[j, i] = v
        d2[i, i] += d1[i]  # nli[2,i]==0 (linear)
    return d1, d2


def _sl_ift_chol(
    sl: _Sl,
    XX: np.ndarray,
    R: np.ndarray,
    d: np.ndarray,
    beta: np.ndarray,
    piv: np.ndarray,
) -> dict:
    """mgcv ``Sl.iftChol`` (fast-REML.r:1405-1488), dense ``R``, one
    thread: derivatives of β̂ by implicit differentiation on the
    (preconditioned, possibly truncated) pivoted Cholesky factor ``R`` with
    0-based pivots ``piv``, and from them the ``b'Sb`` and RSS derivatives.
    The triangular solves are mgcv's ``pforwardsolve``/``pbacksolve`` (one
    ``dtrsm`` each), the products ``pmmult2`` (one ``dgemm`` each)."""
    Sb = _sl_mult(sl, beta)
    Skb, _inds = _sl_term_mult(sl, beta, full=True)
    nd = len(Skb)
    p = beta.size
    db = np.zeros((p, nd))
    rss1 = np.zeros(nd)
    if nd == 0:
        return {
            "bSb": 0.0,
            "bSb1": np.zeros(0),
            "bSb2": np.zeros(0),
            "db": db,
            "rss1": rss1,
            "rss2": rss1,
        }
    D = np.column_stack(Skb)
    bSb1 = r_colsums(beta[:, None] * D)
    dp = d[piv][:, None]
    D1 = r_backsolve(R, D[piv, :] / dp, upper=True, transpose=True)
    db[piv, :] = -r_backsolve(R, D1) / dp
    S_db = _sl_mult(sl, db)
    bSb2 = np.diag(r_colsums(beta[:, None] * D))
    bSb2 = bSb2 + 2.0 * (_mgcv_mmult(db, D + S_db, 1, 0) + _mgcv_mmult(D, db, 1, 0))
    XX_db = _mgcv_mmult(XX, db, 0, 0)
    rss2 = 2.0 * _mgcv_mmult(db, XX_db, 1, 0)
    return {
        "bSb": r_sum(beta * Sb),
        "bSb1": bSb1,
        "bSb2": bSb2,
        "db": db,
        "rss1": rss1,
        "rss2": rss2,
    }


def _pi_fit_chol(
    XX: np.ndarray,
    Xy: np.ndarray,
    rho: np.ndarray,
    sl: _Sl,
    p: int,
    *,
    yy: float = 0.0,
    log_phi: float = 0.0,
    n: int = 0,
    Mp: int = 0,
    gamma: float = 1.0,
    phi_fixed: bool = True,
    L: np.ndarray | None = None,
    rho0=0.0,
    tol: float = 0.0,
) -> dict:
    """mgcv ``Sl.fitChol`` (fast-REML.r:1585-1680), one thread: given ``XX =
    X'WX`` and ``Xy = X'Wy`` (both in the Sl initial reparameterization)
    solve the penalized least squares problem at working log smoothing
    parameters ``rho`` (full ones ``L rho + rho0``) and return β̂, the REML
    gradient and Hessian in the working space and the Newton step over the
    components whose gradient or Hessian diagonal exceeds ``tol``, capped
    at 4. The factor is R's ``chol(t(XXp/d)/d, pivot = TRUE)`` truncated
    to ``min(rank, Rrank(R))``; ``PP`` is ``chol2inv`` of it."""
    rho = np.asarray(rho, dtype=float)
    rho0 = np.atleast_1d(np.asarray(rho0, dtype=float))
    if L is None:
        rho = rho + rho0
    else:
        rho = r_matprod(np.asarray(L, dtype=float), rho.reshape(-1, 1))[:, 0] + rho0
    if rho.size < rho0.size:
        rho = rho0
    fixed = np.zeros(rho.size, dtype=bool)
    ldS = _ldet_s(sl, rho, fixed=fixed, root=False, repara=False, deriv=2)
    XXp = _sl_add_s(sl, XX, rho)
    d = np.diag(XXp).copy()
    ind = d <= 0
    d[ind] = 1.0
    d[~ind] = np.sqrt(d[~ind])
    R_full, piv1, rank = r_chol((XXp / d[:, None]).T / d[:, None], pivot=True)
    r = min(int(rank), _R_rank(R_full, tol=float(np.finfo(float).eps) ** 0.9))
    piv = np.asarray(piv1, dtype=int) - 1
    R = R_full
    if r < p:
        R = R_full[:r, :r]
        piv = piv[:r]
    beta = np.zeros(p)
    dp = d[piv]
    if r > 0:
        z = r_backsolve(np.ascontiguousarray(R.T), Xy[piv] / dp, upper=False)
        beta[piv] = r_backsolve(R, z) / dp
    dift = _sl_ift_chol(sl, XX, R, d, beta, piv)
    PP = np.zeros((p, p))
    if r > 0:
        PP[np.ix_(piv, piv)] = r_chol2inv(R)
    PP = (PP / d[:, None]).T / d[:, None]
    ldetXXS = 2.0 * r_sum(np.log(np.diag(R)) + np.log(dp)) if r > 0 else 0.0
    dXXS_d1, dXXS_d2 = _d_det_xxs(sl, PP)
    phi = float(np.exp(log_phi))
    nrho = rho.size
    if nrho == 0:
        reml1 = np.zeros(0)
        reml2 = np.zeros((0, 0))
    else:
        reml1 = (
            dXXS_d1[~fixed]
            - ldS["ldet1"]
            + (dift["rss1"][~fixed] + dift["bSb1"][~fixed]) / (phi * gamma)
        ) / 2.0
        reml2 = (
            dXXS_d2[np.ix_(~fixed, ~fixed)]
            - ldS["ldet2"]
            + (
                dift["rss2"][np.ix_(~fixed, ~fixed)]
                + dift["bSb2"][np.ix_(~fixed, ~fixed)]
            )
            / (phi * gamma)
        ) / 2.0
    out = {
        "beta": beta,
        "db": dift["db"],
        "PP": PP,
        "R": R,
        "piv": piv,
        "rank": r,
        "ldetS": float(ldS["ldetS"]),
        "ldetXXS": ldetXXS,
        "d": d,
    }
    if nrho == 0 and phi_fixed:
        out.update(grad=0.0, step=0.0, hess=reml2)
        return out
    if not phi_fixed:
        nn = reml1.size
        rss_bSb = yy - r_sum(beta * Xy)
        reml1 = np.append(reml1, (-rss_bSb / (phi * gamma) + n / gamma - Mp) / 2.0)
        dd = np.concatenate(
            [-(dift["rss1"][~fixed] + dift["bSb1"][~fixed]), [rss_bSb]]
        ) / (2.0 * phi * gamma)
        new2 = np.empty((nn + 1, nn + 1))
        new2[:nn, :nn] = reml2
        new2[:nn, nn] = dd[:nn]
        new2[nn, :] = dd
        reml2 = new2
        if L is not None:
            Lr = np.asarray(L, dtype=float)
            La = np.zeros((Lr.shape[0] + 1, Lr.shape[1] + 1))
            La[: Lr.shape[0], : Lr.shape[1]] = Lr
            La[-1, -1] = 1.0
            L = La
        out["rss_bSb"] = rss_bSb
    if L is not None:
        Lt = np.ascontiguousarray(np.asarray(L, dtype=float).T)
        reml1 = r_matprod(Lt, reml1.reshape(-1, 1))[:, 0]
        reml2 = r_matprod(r_matprod(Lt, reml2), L)
    uconv = (np.abs(reml1) > tol) | (np.abs(np.diag(reml2)) > tol)
    hess = reml2
    grad = reml1
    if grad.size > 0 and np.any(uconv):
        if np.count_nonzero(uconv) != reml2.shape[1]:
            reml1 = reml1[uconv]
            reml2 = reml2[np.ix_(uconv, uconv)]
        values, vectors = _r_eigen_sym(reml2)
        values = np.abs(values)
        me = float(np.max(values)) * float(np.finfo(float).eps) ** 0.5
        values[values < me] = me
        step = np.zeros(uconv.size)
        step[uconv] = _eigen_step(vectors, values, reml1)
        ms = float(np.max(np.abs(step)))
        if ms > 4:
            step = 4 * step / ms
    else:
        step = 0.0
    out.update(grad=grad, step=step, hess=hess)
    return out


def _sl_ift(
    sl: _Sl,
    R: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    beta_piv: np.ndarray,
    piv: np.ndarray,
    rp: np.ndarray,
) -> dict:
    """mgcv ``Sl.ift`` (fast-REML.r:1369-1403): derivatives of β̂ by
    implicit differentiation on the QR factor, used directly for the
    ``b'Sb`` and RSS derivatives. ``R``/``piv``/``rp`` are the pivoted-QR
    factor of the augmented ``rbind(X, E)`` and its pivots / reverse
    pivots; ``X`` is the ldetS-repara'd model matrix; ``beta_piv`` the
    pivoted β̂.
    """
    beta = beta_piv[rp]  # unpivot (:1374)
    Sb = _sl_mult(sl, beta)  # Sl.mult(β, k=0)
    Skb, _inds = _sl_term_mult(sl, beta, full=True)  # dSλ/dρ_i · β
    rsd = r_matprod(X, beta.reshape(-1, 1))[:, 0] - y
    nd = len(Skb)
    p = beta.size
    db = np.zeros((p, nd))
    rss1 = np.zeros(nd)
    bSb1 = np.zeros(nd)
    Rt = np.ascontiguousarray(R.T)
    for i in range(nd):
        w = r_backsolve(Rt, Skb[i][piv], upper=False)  # forwardsolve(t(R), .)
        db[:, i] = -r_backsolve(R, w)[rp]
        bSb1[i] = r_sum(beta * Skb[i])  # β'S_iβ (:1387)
    XX_db = r_matprod(np.ascontiguousarray(X.T), r_matprod(X, db))
    S_db = _sl_mult(sl, db)  # Sl.mult(db, k=0)
    rss2 = np.zeros((nd, nd))
    bSb2 = np.zeros((nd, nd))
    for k in range(nd):
        for j in range(k, nd):
            rss2[j, k] = rss2[k, j] = 2.0 * r_sum(db[:, j] * XX_db[:, k])
            v = (r_sum(beta * Skb[k]) if k == j else 0.0) + 2.0 * (
                r_sum(db[:, k] * (Skb[j] + S_db[:, j])) + r_sum(db[:, j] * Skb[k])
            )
            bSb2[j, k] = bSb2[k, j] = v
    return {
        "bSb": r_sum(beta * Sb),
        "bSb1": bSb1,
        "bSb2": bSb2,
        "d1b": db,
        "rss": r_sum(rsd * rsd),
        "rss1": rss1,
        "rss2": rss2,
    }


def _mgcv_pbsi(R: np.ndarray) -> np.ndarray:
    """mgcv's ``pbsi(R)`` (misc.r:601), the C ``mgcv_pbsi1`` with one
    thread: the inverse of the upper triangular ``R`` built in 50-column
    blocks. Each block's rows above it are multiplied by the inverse so far
    (``pdtrmm``, which with one thread is BLAS ``dtrmm`` on a copy added
    back into zeros) and then by minus the block's inverse (BLAS ``dtrsm``
    from the right); LAPACK ``dtrti2`` inverts the diagonal block a column
    at a time (reciprocal of the diagonal, BLAS ``dtrmv``, then the scale
    by minus that reciprocal)."""
    from scipy.linalg.blas import dtrmm, dtrmv, dtrsm

    A = np.array(R, dtype=float, order="F")
    n = A.shape[0]
    nb = 50
    for j in range(0, n, nb):
        jb = min(nb, n - j)
        if j:
            blk = dtrmm(
                1.0,
                np.asfortranarray(A[:j, :j]),
                np.asfortranarray(A[:j, j : j + jb]),
                side=0,
                lower=0,
                trans_a=0,
                diag=0,
            )
            A[:j, j : j + jb] = 0.0 + np.asarray(blk)
            A[:j, j : j + jb] = dtrsm(
                -1.0,
                np.asfortranarray(A[j : j + jb, j : j + jb]),
                np.asfortranarray(A[:j, j : j + jb]),
                side=1,
                lower=0,
                trans_a=0,
                diag=0,
            )
        for jj in range(j, j + jb):
            A[jj, jj] = 1.0 / A[jj, jj]
            ajj = -A[jj, jj]
            if jj > j:
                x = dtrmv(
                    np.asfortranarray(A[j:jj, j:jj]),
                    np.ascontiguousarray(A[j:jj, jj]),
                    lower=0,
                    trans=0,
                    diag=0,
                )
                A[j:jj, jj] = np.asarray(x) * ajj
    return np.ascontiguousarray(A)


def _sl_fit(
    sl: _Sl,
    X: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    *,
    fixed: np.ndarray | None = None,
    log_phi: float = 0.0,
    phi_fixed: bool = True,
    rss_extra: float = 0.0,
    nobs: float | None = None,
    Mp: int = 0,
    gamma: float = 1.0,
) -> dict:
    """mgcv ``Sl.fit`` (fast-REML.r:1682-1738): penalised regression via
    pivoted QR of the AUGMENTED model matrix ``rbind(X, E)`` at fixed ρ,
    returning β̂ plus the Gaussian working-model REML value, gradient and
    Hessian — the per-evaluation oracle of ``fast.REML.fit`` (the
    non-discrete bam cadence). The discrete POI keeps
    :func:`_pi_fit_chol` (= ``Sl.fitChol``, mgcv's discrete oracle).
    before ``fast.REML.fit`` (bam.r:1228/1686). β̂ (unpivoted) and PP come
    (``Sl.postproc``, fast-REML.r:1983/1993).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    n_sp = sum(blk.n_sp for blk in sl.blocks)
    if fixed is None:
        fixed = np.zeros(n_sp, dtype=bool)  # fast.REML.fit:1771
    free = ~fixed
    phi = float(np.exp(log_phi))
    if nobs is None:
        nobs = float(X.shape[0])

    ldS = _ldet_s(sl, rho, fixed=fixed, root=True, deriv=2)
    X = _sl_repara(ldS["rp"], X)
    E = ldS["E"]
    qrx = r_qr_lapack(np.concatenate([X, E], axis=0))  # qr(rbind(X, E), LAPACK=TRUE)
    piv = np.asarray(qrx["pivot"], dtype=int) - 1
    rp = np.empty(p, dtype=int)  # reverse pivot (:1697)
    rp[piv] = np.arange(p)
    R = r_qr_R(qrx)
    Qty0 = r_qr_qy_lapack(
        qrx, np.concatenate([y, np.zeros(E.shape[0])]), transpose=True
    )[:, 0]
    beta_piv = r_backsolve(R, Qty0)
    rss_bSb = r_sum(Qty0[p:] * Qty0[p:]) + rss_extra

    dift = _sl_ift(sl, R, X, y, beta_piv, piv, rp)

    P = _mgcv_pbsi(R)  # invert R
    PP = r_tcrossprod(P)[np.ix_(rp, rp)]  # unpivot
    ldetXXS = 2.0 * r_sum(np.log(np.abs(np.diag(R))))
    dXXS_d1, dXXS_d2 = _d_det_xxs(sl, PP)

    reml = (
        rss_bSb / (phi * gamma)
        + (nobs / gamma - Mp) * float(np.log(2.0 * np.pi * phi))
        + Mp * float(np.log(gamma))
        + ldetXXS
        - ldS["ldetS"]
    ) / 2.0
    reml_pen = (
        dift["bSb"] / (phi * gamma)
        - Mp * float(np.log(2.0 * np.pi * phi))
        + Mp * float(np.log(gamma))
        + ldetXXS
        - ldS["ldetS"]
    ) / 2.0
    grad = (
        dXXS_d1[free]
        - ldS["ldet1"]
        + (dift["rss1"][free] + dift["bSb1"][free]) / (phi * gamma)
    ) / 2.0
    hess = (
        dXXS_d2[np.ix_(free, free)]
        - ldS["ldet2"]
        + (dift["rss2"][np.ix_(free, free)] + dift["bSb2"][np.ix_(free, free)])
        / (phi * gamma)
    ) / 2.0

    if not phi_fixed:  # (:1724-1730)
        grad_phi = (-rss_bSb / (phi * gamma) + nobs / gamma - Mp) / 2.0
        grad = np.concatenate([grad, [grad_phi]])
        d_phi = np.concatenate(
            [
                -(dift["rss1"][free] + dift["bSb1"][free]),
                [rss_bSb],
            ]
        ) / (2.0 * phi * gamma)
        n_old = hess.shape[0]
        hess_new = np.zeros((n_old + 1, n_old + 1))
        hess_new[:n_old, :n_old] = hess
        hess_new[:n_old, n_old] = d_phi[:n_old]
        hess_new[n_old, :n_old] = d_phi[:n_old]
        hess_new[n_old, n_old] = d_phi[n_old]
        hess = hess_new

    return {
        "beta": beta_piv[rp],  # unpivoted (:1736)
        "grad": grad,
        "hess": hess,
        "reml": float(reml),
        "reml_pen": float(reml_pen),
        "rss": float(dift["rss"] + rss_extra),  # (:1737)
        "nobs": float(nobs),
        "d1b": dift["d1b"],
        "PP": PP,
        "rp": ldS["rp"],
        "ldetS": float(ldS["ldetS"]),
        "ldetXXS": ldetXXS,
    }


def _ident_test(X: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mgcv ``ident.test`` (fast-REML.r:1877-1895): structurally
    un-identifiable coefficients of the model with (initial-repara'd)
    model matrix ``X`` and balanced penalty root ``E`` (``attr(Sl,"E")``).
    Pivoted QR of ``rbind(X/‖X‖_F, E)``; rank by mgcv's Cline-condition
    ``Rrank(qr.R, tol=eps^.75)``; the trailing pivots are dropped.
    Returns 0-based ``(drop, undrop)``.
    """
    p = X.shape[1]
    Xnorm = r_norm(X, "F")
    qrx = r_qr_lapack(np.concatenate([X / Xnorm, E], axis=0))
    rank = _R_rank(r_qr_R(qrx), tol=float(np.finfo(float).eps) ** 0.75)
    drop = np.asarray(qrx["pivot"][rank:], dtype=int) - 1
    undrop = np.arange(p)
    if drop.size:
        undrop = np.setdiff1d(undrop, drop)  # (1:np)[-drop]
    return drop, undrop


def _sl_drop(sl: _Sl, drop: np.ndarray, np_total: int) -> _Sl:
    """mgcv ``Sl.drop`` (fast-REML.r:1899-1955): remove the coefficients
    in ``drop`` from the block-diagonal penalty. Returns a NEW ``_Sl``
    (mgcv is copy-on-modify; the caller's original must stay intact for
    the post-fit unwind, which uses the pre-drop blocks).
    """
    if drop.size == 0:
        return sl
    keep_full = np.ones(np_total, dtype=bool)
    keep_full[drop] = False
    ncum = np.cumsum(keep_full)  # new.loc (1-based count)
    drop_set = {int(i) for i in drop}
    new_blocks: list[_SlBlock] = []
    for blk in sl.blocks:
        cols = np.arange(blk.start, blk.stop)
        bdrop = np.array([int(c) in drop_set for c in cols], dtype=bool)
        new_start = int(ncum[blk.start]) - 1  # new.loc[start] (:1926)
        new_stop = int(ncum[blk.stop - 1])  # new.loc[stop], excl.
        if blk.n_sp == 1:  # singleton (:1918-1927)
            npd = int(np.sum(bdrop[blk.ind]))
            new_ind = blk.ind[~bdrop]
            new_rank = blk.rank - npd
            nb = _SlBlock(
                start=new_start,
                stop=new_stop,
                S=blk.S,
                rank=new_rank,
                repara=blk.repara,
                lam=np.array(blk.lam, copy=True),
                D=blk.D,
                Di=blk.Di,
                ind=new_ind,
                ldet=blk.ldet,
                rS=blk.rS,
            )
        else:  # multi-S (:1928-1942)
            keep_pen = ~bdrop[blk.ind]
            npd = int(np.sum(~keep_pen))
            new_ind = blk.ind[~bdrop]
            new_rank = blk.rank - npd
            new_S = [Sj[np.ix_(keep_pen, keep_pen)] for Sj in blk.S]
            new_rS = (
                [rSj[keep_pen, :] for rSj in blk.rS] if blk.rS is not None else None
            )
            nb = _SlBlock(
                start=new_start,
                stop=new_stop,
                S=new_S,
                rank=new_rank,
                repara=blk.repara,
                lam=np.array(blk.lam, copy=True),
                D=blk.D,
                Di=blk.Di,
                ind=new_ind,
                ldet=blk.ldet,
                rS=new_rS,
            )
        if nb.rank > 0:  # (:1944-1952)
            new_blocks.append(nb)
    keep_ix = np.where(keep_full)[0]
    p_new = int(keep_ix.size)
    return _Sl(
        new_blocks,
        sl.E[np.ix_(keep_ix, keep_ix)],
        sl.S[np.ix_(keep_ix, keep_ix)],
        sl.lam0,
        p_new,
    )


def _sl_xprep(sl: _Sl, X: np.ndarray) -> dict:
    """mgcv ``Sl.Xprep`` (fast-REML.r:1957-1973): apply the Sl.setup
    initial reparameterization to ``X`` (columns only), test structural
    identifiability against the balanced penalty root, drop
    un-identifiable coefficients from both ``X`` and (a copy of) ``Sl``,
    and return the post-drop penalty rank and null-space dimension
    ``Mp = ncol(X) − rank``.
    """
    X = _sl_initial_repara(sl, X, both_sides=False, cov=False)  # (:1962)
    drop, undrop = _ident_test(X, sl.E)  # (:1963)
    if drop.size:
        sl = _sl_drop(sl, drop, X.shape[1])  # (:1966)
        X = np.delete(X, drop, axis=1)  # (:1967)
    rank = int(sum(blk.rank for blk in sl.blocks))  # (:1970)
    return {
        "X": X,
        "sl": sl,
        "undrop": undrop,
        "rank": rank,
        "Mp": int(X.shape[1] - rank),
    }  # (:1972)


def _eigen_step(U: np.ndarray, d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``-U %*% ((t(U) %*% g)/d)``, the Newton step mgcv forms from an
    eigendecomposition (efam.r:66, fast-REML.r:1671/1815), with R's
    ``%*%``."""
    Ut_g = r_matprod(
        np.ascontiguousarray(U.T), np.asarray(g, dtype=float).reshape(-1, 1)
    )[:, 0]
    return -r_matprod(U, (Ut_g / d).reshape(-1, 1))[:, 0]


def _reg_newton_step(
    grad: np.ndarray, hess: np.ndarray, max_step: float = 4.0
) -> np.ndarray:
    """Eigen-regularised Newton step ``-H⁻¹g`` with a ``|step| ≤ max_step``
    cap — the exact formula ``_pi_fit_chol`` uses internally
    (Sl.fitChol:1430-1440), factored out so the discrete-POI optimiser can
    recompute the step in *working* (id-linked) space from the L-contracted
    ``(T'g, T'HT)`` instead of the full per-penalty grad/Hessian
    ``_pi_fit_chol`` returns.
    (fast-REML.r:1438) for the discrete POI, ``maxNstep = 5`` in
    ``fast.REML.fit`` (fast-REML.r:1749, 1818) for the non-discrete
    """
    if hess.shape[0] == 0:
        return np.zeros(0)
    eig_w, eig_v = _r_eigen_sym(hess)
    eig_w_abs = np.abs(eig_w)
    if eig_w_abs.size > 0:
        me = float(eig_w_abs.max() * float(np.finfo(float).eps) ** 0.5)
        eig_w_clamped = np.where(eig_w_abs < me, me, eig_w_abs)
    else:
        eig_w_clamped = eig_w_abs
    step = _eigen_step(eig_v, eig_w_clamped, grad)
    ms = float(np.max(np.abs(step))) if step.size else 0.0
    if ms > max_step:
        step = max_step * step / ms
    return step


def _mgcv_pchol(A: np.ndarray, nb: int = 40) -> tuple[np.ndarray, np.ndarray, int]:
    """mgcv's ``pchol(A)`` (misc.r), the C ``mgcv_bchol`` with one thread:
    Lucas's block pivoted Cholesky (2004, algorithm 5.1) in blocks of ``nb``
    columns. Per column the running dot products ``dots += a*a`` (fused),
    the pivot is the first largest ``A[l,l] - dots[l]``, rows and columns
    are swapped as BLAS ``dswap`` does, the diagonal is ``sqrt(A[j,j] -
    dots[j])``, the row is updated by ``dgemv('T')`` and divided by it; the
    trailing block takes one ``dsyrk('U','T')`` (``pdsyrk``). It stops at
    the first pivot below ``n * max diag * eps``. Returns the factor (zeroed
    below the diagonal and past the rank), the 0-based pivots and the
    rank."""
    from scipy.linalg.blas import dgemv, dsyrk

    A = np.array(A, dtype=float, order="F")
    n = A.shape[0]
    piv = np.arange(n)
    dots = np.zeros(n)
    r = -1
    tol = 0.0
    jb = nb
    k = 0
    eps = float(np.finfo(float).eps)
    while k < n:
        jb = min(jb, n - k)
        dots[k:] = 0.0
        j = k
        while j < k + jb:
            if j > k:
                dots[j:] = _rfma_vec(A[j - 1, j:], A[j - 1, j:], dots[j:])
            xmax = -1.0
            q = j
            for ll in range(j, n):
                x = A[ll, ll] - dots[ll]
                if x > xmax:
                    xmax = x
                    q = ll
            if j == 0:
                tol = n * xmax * eps
            if A[q, q] - dots[q] < tol:
                r = j
                break
            dots[j], dots[q] = dots[q], dots[j]
            piv[j], piv[q] = piv[q], piv[j]
            A[j, j], A[q, q] = A[q, q], A[j, j]
            if q - j - 1 > 0:
                tmp = A[j, j + 1 : q].copy()
                A[j, j + 1 : q] = A[j + 1 : q, q]
                A[j + 1 : q, q] = tmp
            if n - q - 1 > 0:
                tmp = A[j, q + 1 :].copy()
                A[j, q + 1 :] = A[q, q + 1 :]
                A[q, q + 1 :] = tmp
            if j > 0:
                tmp = A[:j, j].copy()
                A[:j, j] = A[:j, q]
                A[:j, q] = tmp
            A[j, j] = math.sqrt(A[j, j] - dots[j])
            if j > k and j < n - 1:
                A[j, j + 1 :] = dgemv(
                    -1.0,
                    np.asfortranarray(A[k:j, j + 1 :]),
                    np.ascontiguousarray(A[k:j, j]),
                    beta=1.0,
                    y=np.ascontiguousarray(A[j, j + 1 :]),
                    trans=1,
                )
            A[j, j + 1 :] = A[j, j + 1 :] / A[j, j]
            j += 1
        if r > 0:
            break
        if k + jb < n:
            A[j:, j:] = dsyrk(
                -1.0,
                np.asfortranarray(A[k:j, j:]),
                beta=1.0,
                c=np.asfortranarray(A[j:, j:]),
                trans=1,
                lower=0,
            )
        k += nb
    if r < 0:
        r = n
    for jj in range(n):
        A[(jj + 1 if jj < r else r) :, jj] = 0.0
    return A, piv, r


def _chol2qr(XX: np.ndarray, Xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mgcv ``chol2qr(XX, Xy)`` (bam.r:31-44) with one thread: ``R`` from
    mgcv's ``pchol`` (rows past the rank zeroed), ``f = forwardsolve(t(R),
    Xy[piv])`` on the leading rank block, then both un-pivoted -
    ``R[ipiv, ipiv]`` and ``f[ipiv]`` - so that ``R'R = X'X`` and ``R'f =
    X'y``."""
    XX = np.asarray(XX, dtype=float)
    Xy = np.asarray(Xy, dtype=float).ravel()
    p = Xy.shape[0]
    if p == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    R, piv, rank = _mgcv_pchol(XX)
    ipiv = np.empty(p, dtype=int)
    ipiv[piv] = np.arange(p)
    if rank < p:
        R[rank:, :] = 0.0
    f = np.zeros(p)
    if rank > 0:
        f[:rank] = r_backsolve(
            np.ascontiguousarray(R[:rank, :rank].T), Xy[piv[:rank]], upper=False
        )
    return R[np.ix_(ipiv, ipiv)], f[ipiv]


def _qr_update(
    Xn: np.ndarray,
    yn: np.ndarray,
    R: np.ndarray | None = None,
    f: np.ndarray | None = None,
    y_norm2: float = 0.0,
    use_chol: bool = False,
) -> dict:
    """Update QR factor ``R`` and projected response ``f`` with new rows.
    Direct port of mgcv ``qr_update`` (bam.r:46-75). Given ``X = QR`` and
    ``f = Q'y``, append rows ``Xn``/``yn`` and refresh ``(R, f, ‖y‖²)``:
    R's ``qr(rbind(R, Xn), tol = 0, LAPACK = TRUE)`` (``dgeqp3``), ``qr.qty``
    (``dormqr``) and ``qr.R`` with the columns un-pivoted, ``‖y‖²`` by R's
    ``sum``. The ``use_chol=True`` path accumulates ``crossprod(X)`` and
    ``t(X) %*% y`` instead, which :func:`_chol2qr` converts to ``(R, f)``.
    Returns a dict ``{R, f, y_norm2}`` matching mgcv's list.
    """
    Xn = np.asarray(Xn, dtype=float)
    yn = np.asarray(yn, dtype=float).ravel()
    p = Xn.shape[1]
    y_norm2 = float(y_norm2) + r_sum(yn * yn)
    if use_chol:
        XtX = r_crossprod(Xn)
        Xty = r_matprod(np.ascontiguousarray(Xn.T), yn.reshape(-1, 1)).ravel()
        if R is None:
            return {"R": XtX, "f": Xty, "y_norm2": y_norm2}
        return {
            "R": np.asarray(R, dtype=float) + XtX,
            "f": np.asarray(f, dtype=float).ravel() + Xty,
            "y_norm2": y_norm2,
        }
    if R is not None:
        Xn = np.vstack([np.asarray(R, dtype=float), Xn])
        yn = np.concatenate([np.asarray(f, dtype=float).ravel(), yn])
    qrx = r_qr_lapack(Xn)
    fn = r_qr_qy_lapack(qrx, yn, transpose=True)[: min(p, Xn.shape[0]), 0]
    rp = np.empty(p, dtype=int)
    rp[np.asarray(qrx["pivot"]) - 1] = np.arange(p)
    return {"R": r_qr_R(qrx)[:, rp], "f": fn, "y_norm2": y_norm2}


# ---------------------------------------------------------------------------
# mini.mf — representative subset for basis setup (bam.r:384-427)
# ---------------------------------------------------------------------------


def _mini_mf(data: pl.DataFrame, chunk_size: int, *, seed: int = 66) -> pl.DataFrame:
    """Representative subsample of ``data`` for basis setup.
    Port of mgcv ``mini.mf`` (bam.r:384-427). Returns up to ``chunk_size``
    through :func:`_qr_update`. Matches mgcv's ``bam.r:2387`` flow.
    """
    n = data.height
    cols = data.columns
    # Count minimum representative rows: |levels| per factor, 2 otherwise
    # (numeric scalar OR matrix column) — bam.r:389 ``if is.factor()
    # nlevels else 2``.
    mn = 0
    for c in cols:
        s = data[c]
        if _is_factor_col(s):
            mn += int(s.unique().len())
        else:
            mn += 2
    chunk_size = max(chunk_size, mn)
    if n <= chunk_size:
        return data
    rng = RMersenneTwister(seed)
    ind = rng.sample_int(n, chunk_size)
    mf0 = data[ind.tolist()]
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
        elif is_matrix_col(s):
            # Matrix column (bam.r:407-409): global min/max over ALL
            # entries, first ROW containing it (the ``rowSums(==min)``
            # trick). A censored/gfam matrix response reaches this path.
            arr2 = matrix_to_2d(s)
            gmin = arr2.min()
            gmax = arr2.max()
            rows.append(int(np.flatnonzero((arr2 == gmin).any(axis=1))[0]))
            rows.append(int(np.flatnonzero((arr2 == gmax).any(axis=1))[0]))
        elif s.dtype.is_numeric():
            arr = s.to_numpy()
            j_min = int(np.argmin(arr))
            j_max = int(np.argmax(arr))
            rows.append(j_min)
            rows.append(j_max)
    if rows:
        head = data[rows]
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


def _mini_mf_vars(expanded) -> list[str]:
    """Covariate column names the formula references, for building the
    frame fed to :func:`_mini_mf`.
    (mgcv.r:392-419): parametric-term variables, then each smooth's
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    for t in expanded.terms:
        names: set[str] = set()
        for a in t.atoms:
            _collect_name_idents(a, names)
        for nm in sorted(names):
            add(nm)
    for call in expanded.smooths:
        for v in _smooth_term_vars(call):
            add(v)
        by = _smooth_by_expr(call)
        if by is not None:
            add(by)
            by_names: set[str] = set()
            _collect_name_idents(call.kwargs.get("by"), by_names)
            for nm in sorted(by_names):
                add(nm)
    for off in expanded.offsets:
        names = set()
        _collect_name_idents(off, names)
        for nm in sorted(names):
            add(nm)
    return out


def _smooth_specs_from_expanded(expanded, data: pl.DataFrame) -> list[dict]:
    """Build the ``discrete_mf`` ``smooth_specs`` list from an expanded
    formula, mirroring how mgcv ``bam.r:2206-2215`` derives ``dk`` directly
    from the formula (not from already-built smooth blocks).
    """
    from ..formula import (
        _apply_tero,
        _fs_find_factor,
        _smooth_bs,
        _smooth_by_expr,
        _smooth_term_vars,
        _te_parse_margins,
    )

    out: list[dict] = []
    for call in expanded.smooths:
        term_vars = _smooth_term_vars(call)
        by_expr = _smooth_by_expr(call)
        if by_expr is not None and by_expr not in data.columns:
            by_expr = None
        fs_fac = None
        if call.fn == "s" and _smooth_bs(call) == "fs" and len(term_vars) > 1:
            fs_fac, _others = _fs_find_factor(term_vars, data)
        bs_s = _smooth_bs(call) if call.fn == "s" else None
        re_multi = bs_s == "re" and len(term_vars) > 1
        sz_fac = None
        if bs_s == "sz" and len(term_vars) > 1:
            sz_fac, _others = _fs_find_factor_any(term_vars, data)
        if fs_fac is not None or re_multi or (sz_fac and _others):
            # fs and sz with a factor and re of more than one variable are
            # tensor.possible: bam gives them one margin per variable, in
            # the term's order (bam.r:2261-2267)
            margins = [{"term": [t]} for t in term_vars]
        elif call.fn in ("te", "ti"):
            # discrete.mf discretizes a tensor.smooth.spec's margins one by
            # one (bam.r:2156-2158); any other smooth, t2 included, has its
            # variables discretized jointly. tero (bam.r:2037-2054, applied
            # at bam.r:2259) puts the largest-k margin last, so
            # ``compress_df`` processes small-pool margins first, matching
            # mgcv's MT-state consumption sequence.
            te_specs = _apply_tero(_te_parse_margins(call, data))
            margins = [{"term": list(s["term"])} for s in te_specs]
        else:
            margins = [{"term": term_vars}]
        out.append({"term": term_vars, "by": by_expr, "margins": margins})
    return out


# ---------------------------------------------------------------------------
# AR.resid — AR1 residual computation (bam.r:2056-2076)
# ---------------------------------------------------------------------------


def _ar_resid(
    rsd: np.ndarray, rho: float = 0.0, ar_start: np.ndarray | None = None
) -> np.ndarray:
    """Apply AR1 transform to raw residuals.
    Direct port of mgcv ``AR.resid`` (bam.r:2056-2076). Given residuals
    """
    if rho == 0:
        return rsd
    rsd = np.asarray(rsd, dtype=float).ravel()
    n = rsd.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    ld = 1.0 / np.sqrt(1.0 - rho**2)
    out = ld * rsd
    out[1:] -= rho * ld * rsd[:-1]
    out[0] = rsd[0]
    if ar_start is not None:
        np.copyto(out, rsd, where=np.asarray(ar_start, dtype=bool))
    return out


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
    yX_last: np.ndarray | None = None  # last (y, X) row, for bam.update


def _chunk_indices(
    n: int, chunk_size: int, *, ar1: bool = False
) -> list[tuple[int, int]]:
    """Yield ``(start, end)`` pairs covering ``range(n)`` in chunks of
    ``chunk_size``.
    Mirrors mgcv ``bam.fit`` (bam.r:1566-1574, single-thread). For
    ``ar1=False`` (rho == 0) the chunks tile ``range(n)`` exactly. For
    ``ar1=True`` chunks i >= 1 start one row earlier: that row is the
    previous observation the AR1 transform's sub-diagonal needs, and each
    overlapping chunk's transformed first row is dropped after the transform
    (see :func:`_build_qr_chunked_gaussian`). The indices are pre-drop, so
    consumers pass the full ``[start:end)`` slice.
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


def _ar1_rwmatrix_indices(
    N: int,
    ld: float,
    sd: float,
    ar_start_block: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(stop, row, weight)`` arrays for the AR1 row-recombine.
    Direct port of mgcv ``bam.r:1583-1593`` (single-thread chunk loop)
    and ``bam.r:483-486`` (full-matrix variant). The first output row
    """
    if N <= 0:
        raise ValueError(f"AR1 block must have N>0, got N={N}")
    rep2 = np.repeat(np.arange(1, N + 1), 2)  # (1,1,2,2,…,N,N) length 2N
    row = np.concatenate(([1], rep2[1:-1])).astype(int)  # length 2N-1
    if N >= 2:
        weight = np.concatenate(([1.0], np.tile([sd, ld], N - 1)))
    else:
        weight = np.array([1.0])
    if N >= 2:
        stop = np.concatenate(([1], np.arange(1, N) * 2 + 1)).astype(int)
    else:
        stop = np.array([1], dtype=int)
    if ar_start_block is not None:
        ii = np.flatnonzero(np.asarray(ar_start_block, dtype=bool)) + 1
        if ii.size > 0 and ii[0] == 1:
            ii = ii[1:]  # first obs in block needs no correction
        for k in ii:
            weight[(k - 1) * 2 - 1] = 0.0
            weight[(k - 1) * 2] = 1.0
    return stop, row, weight


def _ar1_tri_weight(
    w: np.ndarray,
    ar_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the symmetric tridiagonal effective weight ``W_eff = D·Tᵀ·T·D``.
    Direct port of mgcv ``XWXd0`` (src/discrete.c:2143-2156). ``D =
    """
    w = np.asarray(w, dtype=float)
    aw = np.asarray(ar_weights, dtype=float)
    n = w.shape[0]
    d = np.sqrt(w)  # mgcv discrete.c:2148  w[i] = sqrt(w[i])
    even = aw[0::2]  # leading diagonals t_ii   (len n)
    odd = aw[1::2]  # sub diagonals t_{i+1,i}  (len n-1)
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
    sumsq = even * even  # even[i]²  (== last-row diagonal base)
    sumsq[:-1] = _rfma_vec(odd, odd, sumsq[:-1])  # arm64: fma(odd,odd,even²)
    w_diag = d * (sumsq * d)
    # last row i=n-1: discrete.c:2152 ``w[i] *= w[i]·even·even`` associates as
    # d·((d·even)·even) — a DIFFERENT rounding from the general d·(even²·d)
    # (differs ≤1 ULP in ~31% of inputs). Match C's association exactly.
    dl = float(d[-1])
    el = float(even[-1])
    w_diag[-1] = dl * ((dl * el) * el)
    return w_diag, w_off


# a bam keeps its model-matrix chunks once built when they fit in this many
# bytes, rather than re-evaluating the bases on every chunked pass
_X_CACHE_BYTES = 1 << 28


def _materialize_chunk(
    blocks: list[SmoothBlock],
    chunk_data: pl.DataFrame,
    X_param_chunk: np.ndarray,
) -> np.ndarray:
    """Build the full design row block for ``chunk_data``, as bgam.fit's
    ``predict(G, newdata=mf[ind,], type="lpmatrix", newdata.guaranteed=TRUE,
    block.size=length(ind))`` does (bam.r:1064).
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
    ar_start: np.ndarray | None = None,
    prior_w: np.ndarray | None = None,
    x_rows=None,
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
        raise ValueError(f"rho must be in (-1, 1) for stationary AR1, got rho={rho!r}")
    ar1 = rho != 0.0
    if ar_start is not None:
        ar_start = np.asarray(ar_start, dtype=bool).flatten()
        if ar_start.shape != (n,):
            raise ValueError(f"ar_start must have length {n}, got {ar_start.shape}")
    if ar1:
        ld = 1.0 / np.sqrt(1.0 - rho**2)
        sd = -rho * ld
    chunks = _chunk_indices(n, chunk_size, ar1=ar1)
    # A frame that fits in one chunk is fitted from G$X, the design the
    # smooth constructors built (bam.r:1649); larger frames rebuild each
    # chunk's rows by prediction (bam.r:1596).
    X_whole = (
        np.concatenate(
            [np.asarray(X_param_full, dtype=float)]
            + [np.asarray(b.X, dtype=float) for b in blocks],
            axis=1,
        )
        if n <= chunk_size
        else None
    )
    R: np.ndarray | None = None
    f: np.ndarray | None = None
    y_norm2 = 0.0
    for chunk_idx, (start, end) in enumerate(chunks):
        chunk_data = data[start:end]
        X_param_chunk = X_param_full[start:end]
        if X_whole is not None:
            X_chunk = X_whole
        elif x_rows is not None:
            X_chunk = x_rows(start, end)
        else:
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
            ar_start_block = ar_start[start:end] if ar_start is not None else None
            stop, row, weight = _ar1_rwmatrix_indices(
                N_block,
                ld,
                sd,
                ar_start_block,
            )
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
    rss_extra = float(y_norm2 - r_sum(f * f))
    return _BamQR(
        R=np.asarray(R, dtype=float),
        f=np.asarray(f, dtype=float),
        y_norm2=float(y_norm2),
        rss_extra=rss_extra,
    )


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
    eta: np.ndarray  # full η (length n) — *with* offset
    mu: np.ndarray  # length-n
    wt: np.ndarray  # length-n PIRLS weights (Fisher form)
    z: np.ndarray  # length-n working response (offset-stripped)
    dev: float  # Σ family.dev_resids(y, μ, w_prior)
    # X'WX / X'Wz as accumulated, when the build formed them directly rather
    # than by accumulating (R, f). mgcv ``bgam.fitd`` hands exactly these to
    # ``Sl.initial.repara``/``Sl.fitChol`` (bam.r:657-665); recovering them as
    # ``R'R``/``R'f`` from a Cholesky factor is a lossy detour. ``None`` on the
    # chunked rails, where (R, f) IS what mgcv accumulates.
    XX: np.ndarray | None = None
    Xy: np.ndarray | None = None


def _build_qr_chunked_pirls(
    data: pl.DataFrame,
    blocks: list[SmoothBlock],
    X_param_full: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    family: Family,
    *,
    coef: np.ndarray | None,
    eta_init: np.ndarray | None,
    chunk_size: int,
    use_chol: bool = False,
    prior_w: np.ndarray | None = None,
    x_rows=None,
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
    R: np.ndarray | None = None
    f: np.ndarray | None = None
    y_norm2 = 0.0
    eta_full = np.empty(n, dtype=float)
    mu_full = np.empty(n, dtype=float)
    wt_full = np.zeros(n, dtype=float)  # mgcv ``wt`` carries 0 for !good rows
    z_full = np.zeros(n, dtype=float)
    dev_total = 0.0

    try:
        for start, end in _chunk_indices(n, chunk_size):
            # mgcv bam.r:1063/1068: window per-row family state to the
            # chunk BEFORE the X build — ``family$setInd(ind)`` (gfam's
            # fi index, read by its per-row linkinv/dev.resids/dDeta)
            # and ``subsety(y, ind)`` (the censored families' censor
            # bound, stashed on the family in hea). No-op default.
            family.set_ind(np.arange(start, end))
            chunk_data = data[start:end]
            X_param_chunk = X_param_full[start:end]
            if x_rows is not None:
                X_chunk = x_rows(start, end)
            else:
                X_chunk = _materialize_chunk(blocks, chunk_data, X_param_chunk)
            off_chunk = offset[start:end]
            y_chunk = y[start:end]
            wp_chunk = prior_w[start:end]

            # mgcv bam.r:1066: ``if (is.null(coef)) eta1 <- eta[ind] else
            # eta[ind] <- eta1 <- drop(X %*% coef) + offset[ind]``.
            if coef is None:
                eta_chunk = eta_init[start:end]
            else:
                eta_chunk = (
                    r_matprod(X_chunk, np.reshape(coef, (-1, 1)))[:, 0] + off_chunk
                )

            mu_chunk = link.linkinv(eta_chunk)

            if family.is_extended or isinstance(family, _tw_family):
                # Extended-family Fisher-scoring branch — the NON-discrete
                # serial chunked path mirrors mgcv ``bgam.fit`` (bam.r:1070-1076),
                # which uses the EXPECTED (Fisher) Hessian unconditionally:
                #     w <- dd$EDeta2 * .5
                #     z <- (eta1-offset) - dd$Deta.EDeta2
                #     good <- is.finite(z) & is.finite(w)
                # (Contrast the DISCRETE path ``_build_qr_discrete_pirls``, which
                # ports ``bgam.fitd``'s rho==0 OBSERVED-Hessian ``Deta2`` branch.)
                # mgcv's efam set includes tw() (free p); its EDeta2/Deta.EDeta2
                # reduce analytically to the standard IRLS (W, z), so this is a
                # branch relabel, not a numeric change, for tw.
                theta = family.get_theta()
                deta = family.dDeta(y_chunk, mu_chunk, wp_chunk, theta, level=0)
                EDeta2 = deta["EDeta2"]
                w_chunk = EDeta2 * 0.5
                # R computes z with non-finite dDeta entries SILENTLY and
                # masks them via `good` right after (bam.r:1075-1076);
                # numpy's invalid-op warning is noise mgcv doesn't emit.
                with np.errstate(invalid="ignore"):
                    z_chunk = (eta_chunk - off_chunk) - deta["Deta.EDeta2"]
                good = np.isfinite(z_chunk) & np.isfinite(w_chunk)
                w_chunk = np.where(good, w_chunk, 0.0)
                z_chunk = np.where(good, z_chunk, 0.0)
                dev_total += float(
                    r_sum(family.dev_resids(y_chunk, mu_chunk, wp_chunk, theta=theta))
                )
            else:
                mu_eta_chunk = link.mu_eta(eta_chunk)
                V_chunk = family.variance(mu_chunk)

                # ``good`` mask (mgcv bam.r:1080).
                good = (wp_chunk > 0) & (mu_eta_chunk != 0)
                safe_mu_eta = np.where(mu_eta_chunk != 0, mu_eta_chunk, 1.0)
                safe_V = np.where(V_chunk != 0, V_chunk, 1.0)

                z_chunk = (eta_chunk - off_chunk) + (y_chunk - mu_chunk) / safe_mu_eta
                w_chunk = wp_chunk * (mu_eta_chunk * mu_eta_chunk) / safe_V
                # mgcv bam.r:1085: ``w[!good] <- 0``.
                w_chunk = np.where(good, w_chunk, 0.0)

                dev_total += float(
                    r_sum(family.dev_resids(y_chunk, mu_chunk, wp_chunk))
                )

            eta_full[start:end] = eta_chunk
            mu_full[start:end] = mu_chunk
            wt_full[start:end] = w_chunk
            z_full[start:end] = z_chunk

            if not np.any(good):
                continue
            sqrt_w = np.sqrt(w_chunk[good])
            Xg = sqrt_w[:, None] * X_chunk[good]
            zg = sqrt_w * z_chunk[good]
            upd = _qr_update(Xg, zg, R, f, y_norm2, use_chol=use_chol)
            R = upd["R"]
            f = upd["f"]
            y_norm2 = upd["y_norm2"]
    finally:
        # Restore the full per-row view. mgcv resets only at fit end
        # (``family$setInd(NULL)``, bam.r:1307), leaving the LAST chunk's
        # window active for the full-length dev / estimate.theta calls at
        # bam.r:1166/1186/1204 — live mgcv 1.9-4 receipt: bam(gfam(...),
        # chunk.size=299) on n=300 returns garbage (dev 255.6 vs 388.7
        # unchunked, θ collapses to 0, poisson "non-integer x" warnings
        # from gaussian rows leaking through the 1-row fi window). hea
        # restores here so every full-data evaluation sees the full
        # state — a deliberate correctness divergence, identical whenever
        # chunk_size ≥ n (the window is then the identity; that is the
        # regime the mgcv pins run in).
        family.set_ind(None)

    if R is None:
        raise FloatingPointError(
            "chunked PIRLS build accumulated zero rows — every observation "
            "was dropped by the (w_prior > 0 & μ_η != 0) good mask"
        )
    if use_chol:
        R, f = _chol2qr(R, f)
    rss_extra = float(y_norm2 - r_sum(f * f))
    return _PirlsQR(
        R=np.asarray(R, dtype=float),
        f=np.asarray(f, dtype=float),
        y_norm2=float(y_norm2),
        rss_extra=rss_extra,
        eta=eta_full,
        mu=mu_full,
        wt=wt_full,
        z=z_full,
        dev=dev_total,
    )


def _build_qr_discrete_pirls(
    design: DiscreteDesign,
    y: np.ndarray,
    offset: np.ndarray,
    family: Family,
    *,
    coef: np.ndarray | None,
    eta_init: np.ndarray | None,
    use_chol: bool = False,
    prior_w: np.ndarray | None = None,
    rho: float = 0.0,
    ar_start: np.ndarray | None = None,
) -> _PirlsQR:
    """One PIRLS-step build for ``bam(..., discrete=True)``.
    (bam.r:530-620). Mirrors :func:`_build_qr_chunked_pirls` in shape
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
    ar1 = rho != 0.0
    ar_stop = ar_row = ar_weight = None
    if ar1:
        ld = 1.0 / np.sqrt(1.0 - rho**2)
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

    if family.is_extended or isinstance(family, _tw_family):
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
        # instead of the observed ``Deta2`` (bam.r:638-641). tw() rides
        # this branch too (mgcv efam) — for it, observed ≠ Fisher, and
        # the choice SELECTS THE BASIN: mgcv's two bam rails genuinely
        # converge to different (p, φ, sp) fixed points (live receipt:
        # disc sp 0.2327/φ 1.104 vs non-disc sp 0.2713/φ 1.251 on the
        # same data), so the discrete rail must use Deta2 to land on
        # mgcv-discrete's answer.
        theta = family.get_theta()
        deta = family.dDeta(y, mu_full, prior_w, theta, level=0)
        # R computes z with non-finite dDeta entries SILENTLY and masks
        # them via `good` (bam.r:640-642); numpy's invalid-op warning
        # is noise mgcv doesn't emit.
        with np.errstate(invalid="ignore"):
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
        dev_total = float(r_sum(family.dev_resids(y, mu_full, prior_w, theta=theta)))
    else:
        # bam.r:645-649: rows with mu.eta == 0 take mu.eta = .1 and keep the
        # weight that gives
        mu_eta = np.asarray(link.mu_eta(eta_full), dtype=float).copy()
        V_full = family.variance(mu_full)
        good = mu_eta != 0
        mu_eta[~good] = 0.1
        safe_V = np.where(V_full != 0, V_full, 1.0)

        z_full = (eta_full - offset) + (y - mu_full) / mu_eta
        w_full = prior_w * (mu_eta * mu_eta) / safe_V
        if not np.any(good):
            raise FloatingPointError(
                "discrete PIRLS build saw zero good rows — every "
                "observation was dropped by the (w_prior > 0 & μ_η != 0) "
                "good mask"
            )

        dev_total = float(r_sum(family.dev_resids(y, mu_full, prior_w)))

    if ar1:
        XWX = XWXd(design, w_full, ar_weights=ar_weight)
        Xy = XWyd(design, w_full, z_full, ar=(ar_stop, ar_row, ar_weight))
        # mgcv bam.r:654 — y_norm2 = ‖T·(√w·z)‖² (one rwMatrix, forward).
        tz = _rw_matrix(
            ar_stop, ar_row, ar_weight, np.sqrt(w_full) * z_full, trans=False
        )
        y_norm2 = r_sum(tz * tz)
    else:
        XWX = XWXd(design, w_full)
        Xy = XWyd(design, w_full, z_full)
        y_norm2 = r_sum(w_full * (z_full * z_full))

    R, f = _chol2qr(XWX, Xy)
    rss_extra = float(y_norm2 - r_sum(f * f))

    return _PirlsQR(
        R=np.asarray(R, dtype=float),
        f=np.asarray(f, dtype=float),
        y_norm2=y_norm2,
        rss_extra=rss_extra,
        eta=eta_full,
        mu=mu_full,
        wt=w_full,
        z=z_full,
        dev=dev_total,
        XX=XWX,
        Xy=Xy,
    )


def _is_identity_link(family: Family) -> bool:
    """Detect Gaussian-identity (canonical Gaussian) — the ``am=TRUE`` case
    in mgcv's ``bam.fit`` dispatch (bam.r:2205)."""
    return isinstance(family, Gaussian) and family.link.name == "identity"


# ---------------------------------------------------------------------------
# fast.REML.fit — the reduced-space REML Newton engine (fast-REML.r:1740-1875).
# mgcv keeps this a FREE fn shared by the fREML bam paths; hea mirrors that. The
# per-eval oracle is _pi_fit_chol (hea's Sl.fitChol port); the working↔full map
# L (mgcv's augmented L, fast-REML.r:1782) and the log|Sλ|_+ pieces are threaded
# as plain data, so the fn is self-contained like mgcv's.
# ---------------------------------------------------------------------------


def _fast_reml_fit(
    theta0: np.ndarray,
    *,
    include_log_phi: bool,
    max_iter: int = 200,
    work_dim: int,
    L,
    lsp0,
    p: int,
    sl: _Sl,
    X: np.ndarray,
    f: np.ndarray,
    rss_extra: float,
    gamma: float,
    n: int,
    scale_fixed_value: float,
) -> dict:
    """mgcv ``fast.REML.fit`` (fast-REML.r:1740-1875) — Newton optimiser
    for the working log-sp (and log φ when the scale is free) on the
    reduced ``(R, f)`` data, run to FULL convergence.
    (bam.r:1240/1690) call once per PIRLS iter. The per-evaluation oracle
    bam.r:1228/1686) is applied here, and ``Mp`` is the POST-drop
    ``ncol(X) − rank`` (``um$Mp``, bam.r:1242). ``f`` is the projected
    response ``qrx$f``; ``rss_extra = ‖z̃‖² − ‖f‖²`` (bam.r:1146/1675).
    ``Sl.postproc``'s ``beta[undrop] <-``, fast-REML.r:1982-1983).
      with ``max_step=5`` (fast-REML.r:1805-1818);
      ``not.moved`` early-stall detector (fast-REML.r:1827-1839);
      ``|Δreml| ≤ reml.scale·√eps`` (fast-REML.r:1855-1857), where
      ``reml.scale = |reml| + rss/nobs`` (fast-REML.r:1776).
    exactly mgcv's augmented ``L`` (fast-REML.r:1782) and its ``t(L)``
    grad/Hess contraction (fast-REML.r:1784-1785, 1848-1849). Returns a dict
    """
    n_work = work_dim
    n_extra = 1 if include_log_phi else 0
    if L is None:
        T_work = None
    else:
        n_slots_L, n_work_L = L.shape
        T_work = np.zeros((n_slots_L + n_extra, n_work_L + n_extra))
        T_work[:n_slots_L, :n_work_L] = L
        if n_extra > 0:
            T_work[n_slots_L:, n_work_L:] = np.eye(n_extra)
    conv_tol = float(np.finfo(float).eps) ** 0.5  # mgcv conv.tol
    max_step = 5.0  # mgcv maxNstep
    nobs = float(n)

    def _rho_full(theta_sp):
        rho = theta_sp if L is None else L @ theta_sp
        if lsp0 is not None:
            rho = rho + lsp0
        return rho

    # mgcv ``Sl.Xprep`` (fast-REML.r:1957-1973, called at bam.r:1228/1686):
    # initial repara of qrx$R (columns only — X rows are obs-space),
    # ident.test structural-identifiability drop, post-drop Mp.
    um = _sl_xprep(sl, np.asarray(X, dtype=float))
    X_pre = um["X"]
    sl_fit = um["sl"]
    undrop = um["undrop"]
    Mp = um["Mp"]
    f_arr = np.asarray(f, dtype=float)

    def _eval(t):
        # One Sl.fit evaluation at working θ → dict with the Gaussian
        # working-model REML VALUE, the t(L)-contracted working grad/Hess
        # (fast-REML.r:1784-1785), and the unpenalised working RSS (for
        # reml.scale).
        theta_sp = t[:n_work]
        rho = _rho_full(theta_sp)
        log_phi = (
            float(t[n_work]) if include_log_phi else float(np.log(scale_fixed_value))
        )
        try:
            out = _sl_fit(
                sl_fit,
                X_pre,
                f_arr,
                rho,
                log_phi=log_phi,
                phi_fixed=not include_log_phi,
                rss_extra=rss_extra,
                nobs=nobs,
                Mp=Mp,
                gamma=gamma,
            )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return None
        reml_val = float(out["reml"])
        if not np.isfinite(reml_val):
            return None
        g = out["grad"]
        H = out["hess"]
        if T_work is not None:
            Tt = np.ascontiguousarray(T_work.T)
            g = r_matprod(Tt, np.reshape(g, (-1, 1)))[:, 0]
            H = r_matprod(r_matprod(Tt, H), T_work)
        return {
            "reml": reml_val,
            "grad": g,
            "hess": H,
            "rss": float(out["rss"]),
            "out": out,
        }

    # F5/F9 reuse slots: the converged β̂ / A⁻¹ from Sl.fit, un-repara'd
    # to the original basis (Sl.postproc, fast-REML.r:1983/1993). ``None``
    # until the fit succeeds, so a failed initial eval falls back to
    # ``_fit_given_rho``.
    reml_beta = None
    reml_A_inv = None

    theta = np.asarray(theta0, dtype=float).copy()

    # ---- initial fit + typical reml scale (fast-REML.r:1774-1776) ----
    best = _eval(theta)
    if best is None:
        outer_info = {
            "conv": "initial fit failed",
            "iter": 0,
            "grad": np.zeros_like(theta),
            "hess": np.zeros((theta.size, theta.size)),
            "score": float("inf"),
            "score_scale": float("nan"),
        }
        return {
            "theta": theta,
            "reml_beta": reml_beta,
            "reml_A_inv": reml_A_inv,
            "outer_info": outer_info,
        }
    reml_scale = abs(best["reml"]) + best["rss"] / nobs
    grad = best["grad"]
    hess = best["hess"]
    grad2 = np.diag(hess)
    # active set: drop dims with ~0 grad AND ~0 curvature (fast-REML.r:1791).
    uconv_ind = (np.abs(grad) > reml_scale * conv_tol * 0.1) | (
        np.abs(grad2) > reml_scale * conv_tol * 0.1
    )
    if not np.any(uconv_ind):
        warnings.warn("Possible divergence detected in fast.REML.fit", stacklevel=2)
        uconv_ind = np.ones_like(uconv_ind, dtype=bool)

    step_failed = False
    conv_text = "no convergence in 200 iterations"
    it_done = 0
    for it in range(max_iter):
        it_done = it + 1
        # Newton step on the active subblock (fast-REML.r:1802-1821); the
        # reduced gradient and Hessian are what outer.info reports when the
        # loop ends on a failed step.
        hess = best["hess"][np.ix_(uconv_ind, uconv_ind)]
        grad = best["grad"][uconv_ind]
        step = np.zeros(uconv_ind.size)
        if hess.size > 0:
            step[uconv_ind] = _reg_newton_step(grad, hess, max_step=max_step)

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
            if (
                trial is not None
                and (trial["reml"] - best["reml"]) < conv_tol * reml_scale
            ):
                not_moved += 1
            else:
                not_moved = 0
            if k == 25 or not np.any(theta != theta + step) or not_moved > 3:
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
        uconv_ind = (np.abs(grad) > reml_scale * conv_tol * 0.1) | (
            np.abs(grad2) > reml_scale * conv_tol * 0.1
        )
        converged = True
        if np.any(np.abs(grad) > reml_scale * conv_tol):
            converged = False
        if abs(best["reml"] - trial["reml"]) > reml_scale * conv_tol:
            if converged:  # otherwise can't progress
                uconv_ind = np.ones_like(uconv_ind, dtype=bool)
            converged = False
        best = trial
        theta = theta_try
        if converged:
            conv_text = "full convergence"
            break
        reml_scale = abs(best["reml"]) + best["rss"] / nobs

    outer_info = {
        "conv": conv_text,
        "iter": it_done,
        "grad": grad,
        "hess": hess,
        "score": float(best["reml"]),
        "score_scale": float(reml_scale),
    }
    # F5/F9: reuse Sl.fit's converged β̂ / A⁻¹ (mgcv bgam.fit:1243/1310's
    # Sl.postproc does NOT re-solve). ``best["out"]`` is the _sl_fit result
    # at the accepted θ. The unwind is Sl.postproc's (fast-REML.r:1982-1994):
    # the ldetS multi-S reparameterization (``Sl.repara(fit$rp, .,
    # inverse=TRUE)``, dropped-space), zero-padded scatter into the
    # ``undrop`` positions, then the Sl.setup initial repara with the
    # ORIGINAL (pre-drop) Sl (β: both.sides irrelevant for vectors; PP:
    # both_sides=True, cov=True). Full-rank: identical to
    # _fit_given_rho's solve (~1e-12).
    best_out = best.get("out")
    if best_out is not None:
        beta_full = np.zeros(p)
        beta_full[undrop] = _sl_repara(best_out["rp"], best_out["beta"], inverse=True)
        reml_beta = _sl_initial_repara(
            sl, beta_full, inverse=True, both_sides=False, cov=False
        )
        PP_full = np.zeros((p, p))
        PP_full[np.ix_(undrop, undrop)] = _sl_repara(
            best_out["rp"], best_out["PP"], inverse=True
        )
        reml_A_inv = _sl_initial_repara(
            sl, PP_full, inverse=True, both_sides=True, cov=True
        )
    return {
        "theta": theta,
        "reml_beta": reml_beta,
        "reml_A_inv": reml_A_inv,
        "outer_info": outer_info,
        "fit": best_out,
        "undrop": undrop,
        "um_sl": sl_fit,
    }


def _sl_postproc(
    sl: _Sl,
    fit: dict,
    undrop: np.ndarray,
    X0: np.ndarray,
    hess: np.ndarray,
    *,
    scale: float,
    L: np.ndarray | None,
) -> dict:
    """mgcv ``Sl.postproc(Sl, fit, undrop, X0, cov = TRUE, scale, L)``
    (fast-REML.r:1976-2018): undo ``Sl.fit``'s ldetS and the initial
    reparameterizations of β̂, dβ̂/dρ and ``PP`` (``Sl`` is the set-up
    before any coefficient was dropped), then with ``X0`` (``qrx$R``) form
    ``F = crossprod(crossprod(t(X0), PP), X0)``, ``edf = diag(F)``,
    ``edf1``, ``Vp = PP * scale`` (``scale`` at most 0 estimates it as
    ``rss / (nobs - sum(edf))``), the smoothing-parameter correction from
    the outer Hessian ``hess`` (``fit$outer.info$hess``), ``Vc``, ``edf2``
    and ``Ve = F %*% Vp``."""
    p = X0.shape[1]
    nd = np.asarray(fit["d1b"]).shape[1]
    beta = np.zeros(p)
    beta[undrop] = _sl_repara(fit["rp"], fit["beta"], inverse=True)
    beta = _sl_initial_repara(sl, beta, inverse=True, both_sides=True, cov=True)
    d1b = np.zeros((p, nd))
    d1b[undrop, :] = _sl_repara(
        fit["rp"],
        np.ascontiguousarray(np.asarray(fit["d1b"]).T),
        inverse=True,
        both_sides=False,
    ).T
    for i in range(nd):
        d1b[:, i] = _sl_initial_repara(
            sl, np.ascontiguousarray(d1b[:, i]), inverse=True, both_sides=True, cov=True
        )
    PP = np.zeros((p, p))
    PP[np.ix_(undrop, undrop)] = _sl_repara(fit["rp"], fit["PP"], inverse=True)
    PP = _sl_initial_repara(sl, PP, inverse=True, both_sides=True, cov=True)
    XPP = r_crossprod(np.ascontiguousarray(X0.T), PP)
    hat = r_rowsums(XPP * X0)
    F = r_crossprod(XPP, X0)
    edf = np.diag(F).copy()
    edf1 = 2.0 * edf - r_rowsums(F.T * F)
    if scale <= 0:
        scale = fit["rss"] / (fit["nobs"] - r_sum(edf))
    Vp = PP * scale
    if L is not None:
        d1b = r_matprod(d1b, L)
    M = d1b.shape[1]
    ev_values, vectors = _r_eigen_sym(hess)
    pos = ev_values > 0
    values = np.zeros_like(ev_values)
    values[pos] = 1.0 / np.sqrt(ev_values[pos])
    rV = (values[:, None] * np.ascontiguousarray(vectors.T))[:, :M]
    V_sp = r_crossprod(rV)
    Vc = Vp + r_crossprod(r_matprod(rV, np.ascontiguousarray(d1b.T)))
    edf2 = r_rowsums(Vc * r_crossprod(X0)) / scale
    return {
        "beta": beta,
        "Vp": Vp,
        "Vc": Vc,
        "Ve": r_matprod(F, Vp),
        "V_sp": V_sp,
        "edf": edf,
        "edf1": edf1,
        "edf2": edf2,
        "hat": hat,
        "F": F,
        "d1b": d1b,
        "scale": scale,
    }


def _bgam_fitd_postproc(
    sl: _Sl, prop: dict, XWX: np.ndarray, scale: float, L: np.ndarray | None
) -> dict:
    """bgam.fitd's covariance post-processing (bam.r:800-881, REML): undo
    the initial reparameterization of the last proposal's ``db`` and ``PP``,
    ``F = PP %*% X'WX`` (``pmmult``), ``edf = diag(F)``, ``edf1``,
    ``Vp = PP * scale``, ``Ve = F %*% Vp``, and the smoothing-parameter
    correction from the proposal's Hessian into ``Vc`` and ``edf2``."""
    db = np.array(prop["db"], dtype=float, copy=True)
    for i in range(db.shape[1]):
        db[:, i] = _sl_initial_repara(
            sl, db[:, i], inverse=True, both_sides=True, cov=True
        )
    PP = _sl_initial_repara(sl, prop["PP"], inverse=True, both_sides=True, cov=True)
    F = _mgcv_mmult(PP, XWX, 0, 0)
    edf = np.diag(F).copy()
    edf1 = 2.0 * edf - r_rowsums(F.T * F)
    Vp = PP * scale
    Ve = _mgcv_mmult(F, Vp, 0, 0)
    if L is not None:
        db = r_matprod(db, np.asarray(L, dtype=float))
    M = db.shape[1]
    if M > 0:
        values, vectors = _r_eigen_sym(np.asarray(prop["hess"], dtype=float))
        pos = values > 0
        vals = np.zeros_like(values)
        vals[pos] = 1.0 / np.sqrt(values[pos])
        rV = (vals[:, None] * np.ascontiguousarray(vectors.T))[:, :M]
        Vc = Vp + _mgcv_pcrossprod(r_matprod(rV, np.ascontiguousarray(db.T)))
    else:
        Vc = Vp + 0.0
    edf2 = r_rowsums(Vc * XWX) / scale
    return {
        "edf": edf,
        "edf1": edf1,
        "edf2": edf2,
        "Vp": Vp,
        "Ve": Ve,
        "Vc": Vc,
        "db": db,
    }


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

    _cmX_cache: np.ndarray | None = None

    def __init__(
        self,
        formula: str | list[str],
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
        optimizer: tuple | str | None = None,
        start: np.ndarray | list | None = None,
        in_out: dict | None = None,
        coef: np.ndarray | list | None = None,
        samfrac: float = 1.0,
        min_sp: np.ndarray | list | None = None,
        paraPen: dict | None = None,
    ):
        # ``data`` may be a polars DataFrame OR a mapping of name → 1-D /
        # 2-D ndarray. 2-D entries become matrix columns for mgcv's
        # summation convention (Wood §7.4.1 distributed-lag models).
        #
        # ``in_out``: mgcv bam's warm start — ``{"sp": <FULL per-penalty
        # λ>, "scale": <scale>}``. Unlike gam's (working-length, checked
        # up front), bam reads it lazily where the seeds live: λ
        # replaces ``initial.sp`` (bam.r:687/1229/1687, re-read EVERY
        # PIRLS iter on the non-discrete rail) and scale replaces the
        # ``log(var(y)·0.05)`` log φ seed (bam.r:701/1237).
        # ``prepare_design`` normalizes via ``normalize_data``.
        # ---- general (multi-LP) families: the discrete general rail --------
        # mgcv::bam stops here ("general families not supported by bam",
        # bam.r:2653); hea routes formula LISTS to the discrete general
        # driver it completed from mgcv's dormant gamlss.gH branch.
        self._in_out = in_out
        # mgcv bam(coef=): warm-start coefficients for the PIRLS
        # (bgam.fitd:548 / bgam.fit:1066). bam(samfrac=): loose
        # pre-fit on a random subsample whose coefficients warm-start
        # the full fit (bam.r:2680-2691, non-discrete generalized rail
        # only — mgcv's other rails never read it). The Gaussian-
        # identity rail solves directly, so coef= is a no-op there
        # (mgcv's direct solve likewise takes no warm start).
        self._coef_start = (
            None if coef is None else np.asarray(coef, dtype=float).reshape(-1)
        )
        self._samfrac = float(samfrac)
        # bam(min.sp=): the fixed lower-bound penalty is NOT supported by
        # fast REML / NCV (fast-REML.r:2041), which is every rail hea's bam
        # uses (REML≡fREML alias). mgcv drops it with this exact warning
        # (bam.r:2238-2240); gam(min.sp=) is the supported route.
        if min_sp is not None:
            import warnings as _warnings

            _warnings.warn(
                "min.sp not supported with fast REML and NCV computation, and ignored.",
                stacklevel=2,
            )
        # bam(paraPen=): estimated penalties on parametric terms (mgcv's
        # parametricPenalty, passed to gam.setup at bam.r:2403). Single-LP
        # only — mgcv.r:930 stops multi-formula models.
        self._paraPen = paraPen
        if paraPen is not None and isinstance(formula, (list, tuple)):
            raise ValueError("paraPen not supported for multi-formula models")
        if isinstance(formula, (list, tuple)):
            self._init_general_discrete(
                [str(f) for f in formula],
                data,
                method=method,
                sp=sp,
                family=family,
                offset=offset,
                weights=weights,
                gamma=gamma,
                select=select,
                rho=rho,
                discrete=discrete,
                discrete_m=discrete_m,
                knots=knots,
                optimizer=optimizer,
                start=start,
                in_out=in_out,
            )
            return
        if optimizer is not None:
            raise ValueError(
                "optimizer= applies to the general-family (formula-list) "
                "rail only; single-formula bam chooses its optimizer from "
                "method= like mgcv (bam.r has no optimizer argument)."
            )
        if start is not None:
            raise ValueError(
                "start= applies to the general-family (formula-list) rail "
                "only (gam.fit5's start, gam.fit4.r:941); mgcv's bam warm-"
                "starts single-formula fits through coef= — use that."
            )
        # ---- method aliasing ------------------------------------------------
        # mgcv's bam accepts {fREML, GACV.Cp, GCV.Cp, REML, ML, P-REML,
        # P-ML, NCV} (bam.r:2207). fREML is algorithmically identical to
        # REML on the (R, f, rss_extra) reduced problem — the "fast" comes
        # from the QR-factor reduction, not a different criterion. Map
        # fREML → REML internally. The non-(f)REML/ML/GCV methods run mgcv's
        # compressed-refit route: gam(G=G, method=method) on the Gaussian
        # working (R, f) (bam.r:1263-1276/1722-1733) — hea's inherited
        # criterion machinery on the same reduced statistics.
        method_in = method
        if method == "fREML":
            method = "REML"
        # Keep the pre-alias string: mgcv's REPORTED criterion differs
        # between fREML (raw fit$reml, bam.r:1696 — no ml.pen) and explicit
        # REML (gam(G=G)'s aic-replaced gcv.ubre, bam.r:2784-2787) on the
        # gaussian-identity rail, even though the optimum is shared.
        self._method_in = method_in
        if method == "NCV":
            raise NotImplementedError(
                "method='NCV' (neighbourhood cross validation, bam.r:448 + "
                "ncv.c) is not yet ported. Use 'fREML', 'REML', 'ML', "
                "'GCV.Cp', 'GACV.Cp', 'P-REML' or 'P-ML'."
            )
        if method not in ("REML", "ML", "GCV.Cp", "GACV.Cp", "P-REML", "P-ML"):
            raise ValueError(
                f"method must be one of 'fREML', 'REML', 'ML', 'GCV.Cp', "
                f"'GACV.Cp', 'P-REML', 'P-ML', got {method_in!r}"
            )
        # mgcv bam.r:2226-2228: discretization is only wired for the fREML
        # criterion (hea has no NCV) — any other method warns and falls
        # back to the non-discrete path. hea aliases method="REML" to
        # fREML (same criterion on the reduced data), so explicit REML
        # keeps the discrete rail; the genuinely different criteria
        # (ML, GCV.Cp, GACV.Cp, P-REML, P-ML) fall back like mgcv.
        if discrete and method != "REML":
            warnings.warn("discretization only available with fREML")
            discrete = False
        if not (np.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")
        if knots is not None and not isinstance(knots, dict):
            raise TypeError(
                "knots must be a dict mapping covariate name -> knot sequence "
                "(mgcv's knots=list(...)), or None"
            )
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
        if isinstance(family, _negbin_family) and family.get_theta().size > 1:
            # mgcv bam dies on a θ-vector negbin too, just unhelpfully —
            # bam.r:2206's famname test becomes a length-2 if() condition
            # ("'length = 2' in coercion to 'logical(1)'", R ≥ 4.2).
            # Raise gam.outer's meaningful message (mgcv.r:1650) instead.
            raise ValueError(
                "Please provide a single value for theta or use nb to estimate it"
            )

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
            # bam.r:472/925 (both fitters): `if (is.null(G$family$scale))
            # scale <- 1 else scale <- if (G$family$scale<0) scale else
            # G$family$scale`. NULL-slot families (hea scale_known=True:
            # nb/scat/ocat/ziP/betar/censored/gfam) pin φ ≡ 1 and SILENTLY
            # IGNORE a user scale= — unlike gam, whose mgcv.r:1948 rewrite
            # only fires at scale ≤ 0 (live receipt: bam(nb(2), scale=2.5)
            # ≡ scale unset, sig2 1). tw (slot −1, hea scale_known=False):
            # the user scale is kept — scale>0 fixes φ (live receipt:
            # bam(tw(), scale=2) → sig2 2), scale≤0 estimates it jointly
            # with the Tweedie p in the PIRLS θ-updates (bam.r:616-630).
            if self.family.scale_known:
                self._scale_resolved = 1.0
            else:
                self._scale_resolved = scale if scale > 0.0 else -1.0
        elif self.family.scale_known and not isinstance(self.family, _negbin_family):
            # poisson/binomial under bam — bam.r:2206 rewrites ONLY
            # scale==0 (→ 1); everything else passes through: scale>0
            # fixes φ (verified scale=2.5 → φ=2.5; NOT reset to 1 like
            # estimate.gam), scale<0 ESTIMATES φ — a quasi-style fREML
            # fit (live receipt: bam(poisson, scale=-1) →
            # scale.estimated TRUE, φ̂ 0.9508, sp 0.080715 vs scale=0's
            # 0.085148, both rails). Fixed-θ negbin is excluded:
            # bam.r:2206 keys its known-scale list on famname ∈
            # {"poisson","binomial"} only, so under bam the negbin
            # scale is ESTIMATED like gaussian/Gamma (verified live:
            # bam(negbin(2)) → scale.estimated=TRUE) — unlike gam's
            # forced φ=1 (mgcv.r:1963-1966).
            if scale < 0.0:
                self._scale_resolved = -1.0
            else:
                self._scale_resolved = scale if scale > 0 else 1.0
        elif method in ("REML", "ML", "P-REML", "P-ML"):
            self._scale_resolved = scale if scale > 0 else -1.0
        else:  # GCV.Cp / GACV.Cp
            self._scale_resolved = scale if scale != 0.0 else -1.0
        self.scale_estimated = not self._scale_known_fit
        # Known scale collapses the Pearson-Laplace criteria onto their
        # plain Laplace siblings (mgcv.r:1968-1970, reached by the
        # compressed gam(G=G, scale=scale) refit): with φ fixed there is
        # no Pearson plug-in to make, so P-REML ≡ REML and P-ML ≡ ML.
        # Verified against mgcv 1.9-4: bam(poisson, "P-REML") returns
        # sp/edf/criterion identical to bam(poisson, "REML").
        if method in ("P-REML", "P-ML") and self._scale_known_fit:
            method = method[2:]
        self.method = method

        self._chunk_size = int(chunk_size)
        self._use_chol = bool(use_chol)
        self._discrete = bool(discrete)
        self._discrete_m = discrete_m
        self._discrete_design: DiscreteDesign | None = None
        self._param_tensor: list[_ParamBlock] | None = None
        # mgcv ``dinfo$pmf.names`` (bam.r:2568) — the parametric variables
        # ``discrete.mf`` compresses, replayed on newdata at predict.
        self._pmf_names: list[str] = []
        self._discrete_frame: DiscretizedFrame | None = None
        self._pls_lwork: dict = {"g": None, "o": None}

        # ---- setup phase (mirror gam.__init__ lines 198-321) ---------------
        # cbind(a, b) ~ ... two-column response intake (binomial counts /
        # censored bound / gfam index) — the shared gam helper; mgcv's
        # bgam.fit reaches the same states through family$initialize
        # (binomial, bam.r:951), the censored families' initialize
        # matrix→vector+attr conversion (efam.r:1117 and twins) and
        # gfam's preinitialize (bam.r:922). ``self.formula`` (set above)
        # keeps the original text.
        formula, data, _cbind_kind, _gfam_expr = _cbind_response_intake(
            formula, data, self.family
        )
        if _gfam_expr is not None:
            # predict-on-newdata re-derives the family index from this
            # expression (mgcv.r:2819/3174) — same hook as gam.
            self._gfam_fi_expr = _gfam_expr
        self._basis_state: dict = {}
        d = prepare_design(formula, data, basis_state=self._basis_state)
        self._expanded = d.expanded
        self._param_assign = list(d.param_assign or [])
        _expr_map = _smooth_arg_expr_map(self._expanded)
        self.data = _apply_smooth_arg_exprs(d.data, _expr_map) if _expr_map else d.data
        X_param_df = d.X
        y_full = _coerce_response(d.y, self.family)
        _binom_trials = _cbind_family_stash(_cbind_kind, d, self.family)
        X_param_full = X_param_df.to_numpy().astype(float)
        if X_param_full.shape[1] == 0:
            # 0-column polars frame → to_numpy() collapses to (0, 0); keep
            # the row count (pure-smooth no-intercept models, `y ~ s(x)-1`).
            # Same guard gam.__init__ carries (gam.py:8533-8536); without it
            # `n` came out 0 and every length-n array downstream mis-shaped.
            X_param_full = np.zeros((y_full.shape[0], 0))
        n, p_param = X_param_full.shape

        off = (
            np.zeros(n) if offset is None else np.asarray(offset, dtype=float).flatten()
        )
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
                    d.expanded,
                    self.data,
                )
                # mgcv keeps pmf.names on the fitted object (bam.r:2568) and
                # replays it at predict, where `discrete.mf` drops the names
                # newdata lacks (bam.r:244) — usually just the response.
                names_pmf = _pmf_names(d.expanded, d.response, self.data.columns)
                self._pmf_names = list(names_pmf)
                data_for_discrete = self.data
                if d.response and d.response not in self.data.columns:
                    data_for_discrete = self.data.with_columns(
                        pl.Series(name=d.response, values=y_full)
                    )
                if _cbind_kind in ("censor", "gfam"):
                    # Matrix response: mgcv's pmf compress loop sees the
                    # cbind(y, yat|index) response as ONE two-column
                    # entry — compress.df on the bare matrix, i.e. unique
                    # PAIRS (bam.r:329-333 workaround; d=2 → m=100
                    # default). Rebuild that shape under the response
                    # label so the compress/pad RNG stream matches mgcv's
                    # exactly; the scalar stash column stays for the rest
                    # of the pipeline.
                    second = (
                        "_hea_cnorm_yat" if _cbind_kind == "censor" else "_hea_gfam_fi"
                    )
                    pair = np.column_stack(
                        [
                            y_full,
                            d.data[second].to_numpy().astype(float),
                        ]
                    )
                    data_for_discrete = data_for_discrete.with_columns(
                        pl.Series(d.response, pair, dtype=pl.Array(pl.Float64, 2))
                    )
                self._discrete_frame = discrete_mf(
                    smooth_specs_pre,
                    data_for_discrete,
                    names_pmf=names_pmf,
                    m=self._discrete_m,
                )
                mf_dict = {
                    nm: arr
                    for nm, arr in self._discrete_frame.mf.items()
                    if nm != "(Intercept)" and np.asarray(arr).ndim == 1
                }
                mf0 = _discrete_frame_df(mf_dict, data_for_discrete)
                sb_lists = materialize_smooths(
                    d.expanded,
                    mf0,
                    sparse_cons=0,
                    tero=True,
                    knots=self.knots,
                    mod_con=2,
                )
            else:
                # discrete=FALSE: basis setup on a representative subsample
                # of the MODEL FRAME (mgcv's mini.mf(mf, chunk.size),
                # bam.r:2387); sparse.cons=-1 (sweep-drop absorb on
                # row-summed colMeans). mgcv's ``mf`` is
                # model.frame(gp$fake.formula): the response FIRST, then
                # ONLY the covariates the formula references — not raw
                # ``self.data``, which may carry unused columns and, for
                # matrix-y families, the scalar response stash. A wrong
                # column set changes ``mn`` and the number of
                # representative (min/max/level) rows spliced into the
                # subsample head — hence the retained tail and the fitted
                # basis when n > chunk_size. Build mgcv's model frame: the
                # response as ONE column (a 2-col matrix for censored/gfam,
                # so it counts as a single numeric entry, not
                # scalar-plus-stash), then the referenced covariates.
                mf_cols: dict[str, pl.Series] = {}
                resp_name = d.response or "_hea_y"
                if _cbind_kind in ("censor", "gfam"):
                    second = (
                        "_hea_cnorm_yat" if _cbind_kind == "censor" else "_hea_gfam_fi"
                    )
                    pair = np.column_stack(
                        [
                            np.asarray(y_full, dtype=float),
                            d.data[second].to_numpy().astype(float),
                        ]
                    )
                    mf_cols[resp_name] = pl.Series(
                        resp_name, pair, dtype=pl.Array(pl.Float64, 2)
                    )
                else:
                    mf_cols[resp_name] = pl.Series(
                        resp_name, np.asarray(y_full, dtype=float)
                    )
                for nm in _mini_mf_vars(d.expanded):
                    if nm in mf_cols or nm not in self.data.columns:
                        continue
                    mf_cols[nm] = self.data[nm]
                mf0 = _mini_mf(pl.DataFrame(mf_cols), chunk_size)
                basis_cols = [c for c in mf0.columns if c != resp_name]
                sb_lists = materialize_smooths(
                    d.expanded,
                    mf0.select(basis_cols),
                    sparse_cons=-1,
                    knots=self.knots,
                    mod_con=2,
                )
            blocks: list[SmoothBlock] = [b for group in sb_lists for b in group]
            if discrete:
                for blk in blocks:
                    if blk.spec is None or blk.spec.summation_dim is not None:
                        continue
                    mvars = tuple(
                        t
                        for t in blk.term
                        if t in self.data.columns and is_matrix_col(self.data[t])
                    )
                    if mvars:
                        blk.spec.matrix_vars = mvars
                        blk.spec.summation_dim = matrix_to_2d(
                            self.data[mvars[0]]
                        ).shape[1]
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
                block_ids.extend([_smooth_id_value(call_node)] * len(group_blocks))
                block_sps.extend([_smooth_sp_value(call_node)] * len(group_blocks))
        else:
            blocks = []
            block_ids = []
            block_sps = []
        if self._select:
            blocks = _add_null_space_penalties(blocks)
        blocks = _apply_gam_side(blocks, X_param_full)

        slots: list[_PenaltySlot] = []
        block_col_ranges: list[tuple[int, int]] = []
        col_cursor = p_param
        for b in blocks:
            k = int(np.asarray(b.X).shape[1])
            a, bcol = col_cursor, col_cursor + k
            block_col_ranges.append((a, bcol))
            for j, S_j in enumerate(b.S):
                slots.append(
                    _PenaltySlot(
                        block=b,
                        col_start=a,
                        col_end=bcol,
                        S=np.asarray(S_j, dtype=float),
                        S_scale=_block_s_scale(b, j),
                    )
                )
            col_cursor = bcol
        p = col_cursor

        # ------------- paraPen: estimated penalties on parametric terms -----
        # Direct port of gam.__init__: parametricPenalty (mgcv.r:767-836)
        # merged at mgcv.r:1180-1509, passed to gam.setup by bam (bam.r:2403).
        # Penalties are PREPENDED to the smooth slots (paraPen-first), enter
        # Mp (null space shrinks by their rank), and — since hea's bam is the
        # fast-REML rail throughout — thread through _sl_setup as their own
        # front block on the parametric columns.
        self._pp = None
        pp = None
        pp_slots: list[_PenaltySlot] = []
        if self._paraPen is not None:
            pp = _parametric_penalty(
                self._expanded.terms, d.param_assign, self._paraPen, sp
            )
            if pp is not None:
                self._pp = pp
                for Si, offi, nm in zip(pp["S"], pp["off"], pp["full_sp_names"]):
                    ncol = Si.shape[0]
                    pp_slots.append(
                        _PenaltySlot(
                            block=_ParaPenBlock(nm),
                            col_start=offi,
                            col_end=offi + ncol,
                            S=Si,
                            S_scale=1.0,
                        )
                    )

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
        # the optimizers/post-fit in working space exactly as for gam. paraPen
        # working sps come FIRST (matching G$S = c(PP$S, G$S)); seeding n_work
        # with the paraPen count shifts the smooth 1-hot columns past them.
        n_work_pp = pp["L"].shape[1] if pp is not None else 0
        n_pp = len(pp_slots)
        slot_work_col: list[int] = []
        n_work = n_work_pp
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
        if n_pp:
            slots = pp_slots + slots
        pp_is_identity = pp is None or (
            n_work_pp == n_pp and np.array_equal(pp["L"], np.eye(n_pp))
        )
        if pp_is_identity and n_work == len(slots):
            self._L = None  # identity — no id linkage
        else:
            L = np.zeros((len(slots), n_work))
            if n_pp:
                L[:n_pp, :n_work_pp] = pp["L"]
            if slot_work_col:
                L[n_pp + np.arange(len(slot_work_col)), np.asarray(slot_work_col)] = 1.0
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
            if pp is not None:
                # paraPen working sps sit first; _parametric_penalty already
                # merged any explicit P$sp with the bam(sp=) auto-fill (sp0),
                # mgcv.r:827-830 + the strip at mgcv.r:1181-1183.
                sp_work[:n_work_pp] = pp["sp"]
            if sp is not None:
                sp_arr = np.asarray(sp, dtype=float).flatten()
                if sp_arr.shape != (n_work,):
                    raise ValueError(
                        f"sp must have length {n_work} (one per estimated "
                        f"smoothing parameter; id-linked penalties share "
                        f"one), got {sp_arr.shape}"
                    )
                sp_work[n_work_pp:] = sp_arr[n_work_pp:]
            for (wstart, nS, defining), bsp in zip(block_work_info, block_sps):
                if bsp is None or not defining or nS == 0:
                    continue
                if len(bsp) != nS:
                    # mgcv's exact message (mgcv.r:1426).
                    raise ValueError(
                        "incorrect number of smoothing parameters "
                        "supplied for a smooth term"
                    )
                sp_work[wstart : wstart + nS] = bsp
            fixed_mask = sp_work >= 0.0
            if np.any(fixed_mask):
                L_cur = self._L if self._L is not None else np.eye(len(slots))
                fixed_vals = sp_work[fixed_mask]
                log_fixed = np.empty(fixed_vals.shape[0])
                zero = fixed_vals == 0.0
                log_fixed[~zero] = np.log(fixed_vals[~zero])
                if np.any(zero):
                    # mgcv's "effective zero" for a fixed sp of 0
                    # (mgcv.r:1519-1527), ported bug-for-bug: the i-th zero
                    # reads the i-th *penalty's* block X and S (G$off[i]/
                    # G$S[[i]] with i the literal loop counter), `norm(.,
                    # "F")^2 / norm(S, "F") * eps * .1`. The slot's block
                    # basis is G$X[, ii], built on the same mini frame.
                    eps = float(np.finfo(float).eps)
                    for i, dst in enumerate(np.flatnonzero(zero)):
                        sl = slots[i]
                        nx = r_norm(np.asarray(sl.block.X, dtype=float), "F")
                        ef0 = (
                            nx
                            * nx
                            / r_norm(np.asarray(sl.S, dtype=float), "F")
                            * eps
                            * 0.1
                        )
                        log_fixed[dst] = np.log(ef0)
                self._lsp0 = L_cur[:, fixed_mask] @ log_fixed
                self._L = L_cur[:, ~fixed_mask]
                n_work = int(np.count_nonzero(~fixed_mask))
                self._n_work = n_work
            sp = None
        # bam.r:2576: with no smoothing parameter left free (every one fixed,
        # or no smooths), fREML becomes REML - the compressed gam(G=G) refit
        # in bam.fit/bgam.fit, which reports aic/(2*gamma) + ml.pen.
        if self._method_in == "fREML" and n_work == 0:
            self._method_in = "REML"

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
        if weights is None:
            self._wt = np.ones(n)
            self._has_prior_weights = False
        else:
            w_arr = np.asarray(weights, dtype=float).flatten()
            if w_arr.shape != (n,):
                raise ValueError(f"weights must have length {n}, got {w_arr.shape}")
            if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0):
                raise ValueError("weights must be finite and non-negative")
            self._wt = w_arr
            self._has_prior_weights = True
        if _binom_trials is not None:
            # weights ← weights·n (R binomial initialize, run by bgam.fit
            # at bam.r:951 — "need to reset response and weights to post
            # initialization values ... to deal with binomial properly",
            # bam.r:975-978). Trials re-read from the design frame, which
            # prepare_design may have NA-filtered — keeps rows aligned.
            self._binom_n = _binom_trials
            self._wt = self._wt * self._binom_n
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
            self._param_tensor = param_tensor_blocks(
                d.expanded,
                self._param_assign,
                X_param_full,
                self._discrete_frame,
                {nm: j for j, nm in enumerate(self._discrete_frame.names)},
            )
            self._discrete_design = build_discrete_design(
                blocks,
                X_param_full,
                self._discrete_frame,
                data=self.data,
                param_tensor=self._param_tensor,
                fit_frame=True,
            )

        w = self._wt
        full_yty = float(np.sum(w * y_full * y_full))
        if has_intercept:
            mean_y = float(np.sum(w * y_full) / np.sum(w))
            tss = float(np.sum(w * (y_full - mean_y) ** 2))
        else:
            tss = full_yty
        self._yty_full = full_yty
        self._tss = tss

        Mp = p_param
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            k = bcol - a
            if not b.S:
                Mp += k
                continue
            S_sum = np.sum([np.asarray(s, dtype=float) for s in b.S], axis=0)
            rank = _sym_rank(S_sum)
            Mp += k - rank
        # paraPen shrinks the null space by the rank of each penalized
        # column-group's summed penalty (mgcv Mp = ncol(totalPenaltySpace$Z),
        # mgcv.r:1921-1924); the smooth loop above ranks summed penalties the
        # same way.
        if pp is not None:
            pp_groups: dict[tuple[int, int], np.ndarray] = {}
            for s in pp_slots:
                key = (s.col_start, s.col_end)
                pp_groups[key] = (
                    pp_groups[key] + s.S if key in pp_groups else s.S.copy()
                )
            for S_sum in pp_groups.values():
                Mp -= _sym_rank(S_sum)
        self._Mp = Mp
        self._penalty_rank = p - Mp

        # Working-Gaussian view for the criterion tail (`_score_all`
        # override): mgcv's compressed refits run with G$family <-
        # gaussian() and G$w <- 1 (bam.r:932/1266-1267) — the prior
        # weights are already absorbed into the √W·X chunks the QR sees.
        self._crit_family = Gaussian()
        self._crit_wt = np.ones(n)
        # gam.reparam basis (totalPenaltySpace + mini.roots) for the
        # stable log|Sλ|+ evaluation inside the (P-RE)ML criterion tail —
        # mgcv's compressed gam(G=G) refit builds the same basis from the
        # same penalties (gam.fit3.r:2661).
        self._setup_reparam(slots, p)

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
                self.data,
                blocks,
                X_param_full,
                y_full,
                off,
                chunk_size=chunk_size,
                use_chol=self._use_chol,
                rho=self._rho,
                ar_start=self._ar_start,
                prior_w=(None if not self._has_prior_weights else self._wt),
                x_rows=self._x_rows,
            )
            self._bam_qr = qr
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

            n_work = self._work_dim
            include_log_phi = (not self._scale_known_fit) and method in (
                "REML",
                "ML",
            )
            # initial.spg seed → working space by least squares (mgcv
            # ``coef(lm(lsp ~ L - 1 + offset(lsp0)))``, mgcv.r:4617-4618 /
            # fast-REML.r:1768; identity when no smooths share an id).
            # mgcv's Gaussian bam seeds fast.REML.fit from
            # ``initial.sp(qrx$R)`` exactly like the non-Gaussian path
            # (bam.r:1229) — there is no PIRLS outer loop for Gaussian
            # identity (W=I, z=y), so the seed feeds the one and only
            # converge-fully REML solve.
            # in.out replaces the initial.sp λ seed (bam.r:1687)
            io_rho0 = self._in_out_rho0()
            rho0_full = io_rho0 if io_rho0 is not None else self._initial_sp_rho()
            if self._L is None:
                cur_rho = rho0_full
            else:
                cur_rho = _lm_on_L(rho0_full, self._L, self._lsp0)

            if include_log_phi:
                # mgcv bam.fit:1689 — the Gaussian path has NO PIRLS
                # loop, so the log φ seed is unconditionally
                # ``log(var(y)·0.05)`` (scale<=0, in.out NULL). ``var``
                # uses R's n−1 denominator. Both optimizers converge fully
                # so the seed only sets the Newton start; mirror it for
                # source fidelity (and to share the discrete + non-
                # discrete-generalized seed, bam.py:4134/4302).
                # in.out$scale replaces it (bam.r:1701).
                if io_rho0 is not None:
                    cur_logphi = self._in_out_log_scale()
                else:
                    cur_logphi = float(np.log(r_var(y_full) * 0.05))
                theta0 = np.concatenate([cur_rho, [cur_logphi]])
            else:
                theta0 = cur_rho
            if method == "REML" and self._method_in == "fREML":
                # Non-discrete Gaussian fREML → ``fast.REML.fit``
                # (bam.r:1240), converge-fully on the fixed (R, f).
                theta_hat = self._fast_reml_fit(
                    theta0,
                    include_log_phi=include_log_phi,
                )
            elif method == "GCV.Cp":
                # magic on (qrx$R, qrx$f), bam.r:1678: its coefficients and
                # V = rV rV' are the fit.
                mfit = self._bam_magic()
                self._reml_beta = np.asarray(mfit["b"], dtype=float)
                self._reml_A_inv = r_tcrossprod(mfit["rV"])
                theta_hat = np.log(mfit["sp"])
            else:
                # ML / GACV.Cp / P-REML / P-ML → outer-Newton on the
                # criterion mgcv's compressed refit runs via
                # gam(G=G, method=method) (bam.r:1722-1733).
                # ML canNOT ride fast.REML.fit — its criterion drops the
                # Mp prior terms (remlInd=0) and range-projects log|H|
                # (MLpenalty1), which the ``self.method`` predicates
                # supply through the REML scoreType closures; it keeps
                # log φ in θ when the scale is estimated. None of these
                # produce Sl.fitChol β/PP — null the F9 slots so the fit
                # re-solves.
                self._reml_beta = self._reml_A_inv = None
                crit = (
                    "REML" if method in ("ML", "REML") else self._bam_outer_criterion()
                )
                self._g3 = self._fit3_args_compressed(method)
                if self._g3 is not None:
                    crit = "REML3"
                if include_log_phi:
                    theta0 = np.concatenate([theta0[:-1], [self._gam_g_log_scale0()]])
                theta_hat = self._outer_newton(
                    theta0,
                    criterion=crit,
                    include_log_phi=include_log_phi,
                    include_family_theta=False,
                )

            theta_sp = theta_hat[:n_work]
            if include_log_phi:
                log_phi_hat = float(theta_hat[n_work])
            else:
                log_phi_hat = None
            self._log_phi_hat = log_phi_hat
            if method == "GCV.Cp":
                self.sp = np.asarray(self._bam_magic_res["fit"]["sp"], dtype=float)
            else:
                self.sp = np.exp(theta_sp)  # mgcv m$sp (working)
            rho_hat = self._rho_full(theta_sp)  # full per-penalty log-sp
            self._fit3_final = None
            if getattr(self, "_g3", None) is not None and isinstance(
                self._outer_fit, dict
            ):
                # gam(G=G)'s last gam.fit3 fit on (R, f); its deviance there
                # plus rss.extra is the data's
                fit3 = self._outer_fit
                self._fit3_final = fit3
                fit = _FitState(
                    beta=np.asarray(fit3["coefficients"], dtype=float),
                    dev=float(fit3["deviance"]) + float(self._bam_qr.rss_extra),
                    pen=float("nan"),
                    A_chol=None,
                    A_chol_lower=False,
                    S_full=None,
                    log_det_A=float("nan"),
                    # (R'R + S)^{-1} = rV rV', Vp without its scale
                    A_inv=r_tcrossprod(np.asarray(fit3["rV"], dtype=float)),
                )
            else:
                # the post-fit forms the linear predictor from the final β
                fit = self._fit_given_rho(rho_hat, with_eta=False)
            if self._reml_beta is not None:
                fit.beta = self._reml_beta
                fit.A_inv = self._reml_A_inv

            self._post_fit_gaussian(fit, rho_hat, X_param_df)
        else:
            coef_ws = self._coef_start
            if 0.0 < self._samfrac < 1.0 and self._discrete_design is None:
                # bam(samfrac=): loose pre-fit on a subsample, then the
                # full fit warm-starts at its coefficients
                # (bam.r:2680-2691; the discrete branch never reads
                # samfrac).
                coef_ws = self._samfrac_prefit(coef_ws, sp)
            fit, rho_hat = self._bgam_fit_loop(sp_user=sp, coef_start=coef_ws)
            self._post_fit_pirls(fit, rho_hat, X_param_df)

    def _samfrac_prefit(self, coef_ws, sp):
        """bam(samfrac=)'s subsample pre-fit (bam.r:2680-2691): draw
        ceiling(n·samfrac) rows, fit with control$epsilon loosened to
        1e-2 sharing the FULL-data bases/penalties (mgcv subsets mf,
        never rebuilds G), and return the coefficients as the full
        fit's ``coef=`` warm start. mgcv draws from the ambient R RNG
        (irreproducible from the call alone); hea uses a fixed
        ``hea.R.rng`` stream — the subsample only picks the warm
        start, and the full fit converges to the same optimum either
        way. mgcv's too-small guard warns and skips."""
        from ..R.rng import RGenerator

        n = self.n
        k = int(np.ceil(n * float(self._samfrac)))
        if k < 2 * self.p:
            import warnings as _w

            _w.warn("samfrac too small - ignored", stacklevel=2)
            return coef_ws
        ind = RGenerator(1).mt.sample_int(n, k)
        saved = (
            self._y_arr,
            self._offset,
            self._wt,
            self.data,
            self._X_param_full,
            self.n,
            self._binom_n,
            getattr(self, "_x_cache", None),
        )
        self._x_cache = None
        try:
            self._y_arr = self._y_arr[ind]
            self._offset = self._offset[ind]
            self._wt = self._wt[ind]
            self.data = self.data[ind]
            self._X_param_full = self._X_param_full[ind]
            self.n = int(k)
            if self._binom_n is not None:
                self._binom_n = self._binom_n[ind]
            fit_s, _rho_s = self._bgam_fit_loop(
                sp_user=sp, coef_start=coef_ws, eps=1e-2
            )
            return np.asarray(fit_s.beta, dtype=float).copy()
        finally:
            (
                self._y_arr,
                self._offset,
                self._wt,
                self.data,
                self._X_param_full,
                self.n,
                self._binom_n,
                self._x_cache,
            ) = saved

    def _init_general_discrete(
        self,
        formulas,
        data,
        *,
        method,
        sp,
        family,
        offset,
        weights,
        gamma,
        select,
        rho,
        discrete,
        discrete_m,
        knots,
        optimizer=None,
        start=None,
        in_out=None,
    ):
        """bam for GENERAL (multi-LP) families on the discrete rail — the
        driver mgcv never wired: ``bam.r:2653`` stops with "general
        families not supported by bam" while ``gamlss.gH``'s complete
        discrete branch (gamlss.r:604-711) sat dormant. hea completes
        it: the multi-formula design is discretized once (discrete.mf →
        per-margin Xd tables + ``lpid``, bam.r:2382/2310-2553), the
        penalties/Sl/REML machinery runs in p-space, and every
        n-dependent assembly flows through the XWXd/XWyd/Xbd kernels via
        the family's DiscreteX dispatch inside gam.fit5.

        Smoothness selection defaults to BFGS over the deriv-1 REML
        trace: the discrete ``gamlss.gH`` tops out at that trace
        (gamlss.r:777), so full-Newton REML (needing REML2/trHid2H) is
        structurally unavailable and estimate_gam caps the effective
        ll order at 1 — mgcv.r:1907's coercion to the bfgs outer
        optimizer (gam.fit3.r:1722). ``optimizer=("efs",)`` selects
        extended Fellner-Schall (deriv-0 fits) instead, mgcv's other
        short-derivative route. AR1 (``rho``) and NCV stay off the
        discrete general path, mirroring mgcv's discrete limits.

        ``start=`` seeds the coefficient vector exactly like
        ``gam(start=)`` on the dense general path: gam.fit5 maps it into
        the fitting basis and PROTECTS it from the family initializer
        (gam.fit4.r:995-998 — ``start0 <- start; eval(family$initialize);
        start <- start0``), so a user start replaces the pilot solve
        entirely. That makes it both the warm-start lever (each outer
        BFGS/EFS trial already re-seeds from the previous inner fit) and
        the basin selector for multimodal likelihoods — e.g. seeding a
        discrete fit from a dense pilot fit's coefficients.
        """
        family = Gaussian() if family is None else family
        if not isinstance(family, GeneralFamily):
            raise ValueError(  # noqa: TRY004 - ValueError is the documented API contract
                f"a formula list requires a general (multi-LP) family; got {family!r}"
            )
        if not discrete:
            raise NotImplementedError(
                "bam with a general family runs on the discrete rail "
                "only — pass discrete=True. (mgcv's bam stops outright, "
                "bam.r:2653; hea's dense chunked path is not wired for "
                "multi-LP likelihoods.)"
            )
        if not getattr(family, "discrete_ok", False):
            raise NotImplementedError(
                f"family {getattr(family, 'name', family)!r} is not "
                "flagged discrete-capable (discrete_ok — mgcv's "
                "family$discrete.ok, set on gaulss/multinom/gevlss/"
                "gammals/gumbls): its ll/initialize_coef carry no "
                "DiscreteX branches."
            )
        if rho != 0.0:
            raise NotImplementedError(
                "AR1 (rho) is not available with general families on "
                "the discrete rail (mirrors mgcv's discrete limits)."
            )
        if offset is not None:
            raise NotImplementedError(
                "constructor offset= is not supported for formula-list "
                "bam; put offset(...) atoms in the per-LP formulas."
            )
        if method not in ("fREML", "REML"):
            raise ValueError(
                "general families use (f)REML smoothness selection "
                f"(mgcv.r:1894-1898 coerces the method); got {method!r}"
            )
        if not (np.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")
        if knots is not None and not isinstance(knots, dict):
            raise TypeError(
                "knots must be a dict mapping covariate name -> knot "
                "sequence (mgcv's knots=list(...)), or None"
            )
        if len(formulas) != family.n_lp:
            raise ValueError(
                f"family {family!r} expects {family.n_lp} linear "
                f"predictors; got {len(formulas)} formulas."
            )
        if optimizer is None:
            optimizer = ("outer", "newton")  # → bfgs via the derivs cap
        elif isinstance(optimizer, str):
            optimizer = (optimizer,)
        else:
            optimizer = tuple(optimizer)
        if optimizer not in (("outer", "newton"), ("outer", "bfgs"), ("efs",)):
            raise ValueError(
                "optimizer must be ('outer','newton'), ('outer','bfgs') "
                "or ('efs',) — newton is coerced to bfgs on the discrete "
                f"rail (mgcv.r:1907); got {optimizer!r}"
            )
        self._method_in = method
        self.method = "REML"
        self.family = family
        self.formula = list(formulas)
        self.knots = knots
        self._select = bool(select)
        self._gamma = float(gamma)
        self._edge_correct = False
        self._edge_theta1 = None
        self._control = gam_control()
        self._n_extra_coef = 0
        self._discrete = True
        self._discrete_m = discrete_m
        self._general_discrete = True
        self._pls_lwork = {"g": None, "o": None}

        md = self._build_discrete_general_md(formulas, data, knots, select, discrete_m)
        self._init_general_from_md(
            md,
            self.data,
            family=family,
            sp=sp,
            method="REML",
            weights=weights,
            start=start,
            optimizer=optimizer,
            in_out=in_out,
        )

    def _build_discrete_general_md(
        self, formulas, data, knots, select, discrete_m
    ) -> _MultiDesign:
        """Discretized multi-LP design bundle: the discrete analog of
        ``_prepare_multi_design`` (gam.setup.list). One shared
        ``discrete.mf`` pass over ALL formulas' smooth terms + parametric
        covariates (mgcv accumulates ``pmf.names`` across formulas,
        bam.r:2296-2307, and dk covers every term); per LP the smooths
        are materialized on the padded scalar ``mf0`` (bam.r:2206-2232
        flow, ``sparse_cons=0``/tero like the single-formula rail),
        labels of later formulas suffixed ``.j`` (mgcv's textra); terms
        assemble LP-contiguously into one :class:`DiscreteDesign` with
        ``lpid`` (bam.r:2310-2553). ``md.X`` is the
        :class:`hea.family.DiscreteX` bundle the general fit consumes.
        """
        import types

        from ..formula import prepare_design

        first = formulas[0]
        if "~" not in first:
            raise ValueError(f"first formula must contain '~': {first!r}")
        resp = first.split("~", 1)[0].strip()
        if not resp:
            raise ValueError("first formula must have a response on the lhs")
        full_formulas = [first]
        for j, f in enumerate(formulas[1:], start=1):
            if "~" not in f:
                raise ValueError(f"formula {j} must contain '~': {f!r}")
            lhs = f.split("~", 1)[0].strip()
            if lhs:
                raise NotImplementedError(
                    "formulas after the first must be response-less "
                    f"('~ ...'); got lhs {lhs!r}."
                )
            full_formulas.append(f"{resp} ~ {f.split('~', 1)[1].strip()}")

        ds = [prepare_design(f, data) for f in full_formulas]
        y = ds[0].y.to_numpy().astype(float)
        n = int(y.shape[0])

        _expr_map: dict = {}
        for d in ds:
            _expr_map.update(_smooth_arg_expr_map(d.expanded))
        data_m = (
            _apply_smooth_arg_exprs(ds[0].data, _expr_map) if _expr_map else ds[0].data
        )

        specs: list[dict] = []
        for d in ds:
            specs.extend(_smooth_specs_from_expanded(d.expanded, data_m))
        # mgcv unions each formula's pmf.names, response first (bam.r:2296-2307).
        names_pmf: list[str] = []
        for j_d, d in enumerate(ds):
            for nm in _pmf_names(
                d.expanded, ds[0].response if j_d == 0 else None, data_m.columns
            ):
                if nm not in names_pmf:
                    names_pmf.append(nm)
        data_for_discrete = data_m
        if ds[0].response and ds[0].response not in data_m.columns:
            data_for_discrete = data_m.with_columns(
                pl.Series(name=ds[0].response, values=y)
            )
        dframe = discrete_mf(
            specs, data_for_discrete, names_pmf=names_pmf, m=discrete_m
        )
        mf_dict = {
            nm: arr
            for nm, arr in dframe.mf.items()
            if nm != "(Intercept)" and np.asarray(arr).ndim == 1
        }
        mf0 = _discrete_frame_df(mf_dict, data_for_discrete)

        lp_parts: list[tuple[np.ndarray, list[SmoothBlock]]] = []
        lpi: list[np.ndarray] = []
        blocks_all: list[SmoothBlock] = []
        ranges_all: list[tuple[int, int]] = []
        slots: list[_PenaltySlot] = []
        names: list[str] = []
        nsdf: list[int] = []
        pstart: list[int] = []
        offsets: list[np.ndarray | None] = []
        L_parts: list[tuple[int, int, np.ndarray | None]] = []
        pof = 0
        for j, d in enumerate(ds):
            sb_lists = (
                materialize_smooths(
                    d.expanded, mf0, sparse_cons=0, tero=True, knots=knots, mod_con=2
                )
                if d.expanded.smooths
                else []
            )
            blocks = [b for grp in sb_lists for b in grp]
            block_ids: list[str | None] = []
            for call_node, grp in zip(d.expanded.smooths, sb_lists):
                block_ids.extend([_smooth_id_value(call_node)] * len(grp))
            Xp = d.X.to_numpy().astype(float)
            if Xp.shape[1] == 0:
                Xp = np.zeros((n, 0))
            if select:
                blocks = _add_null_space_penalties(blocks)
            blocks = _apply_gam_side(blocks, Xp)
            if j > 0:
                for b in blocks:
                    b.label = _suffix_smooth_label(b.label, f".{j}")
            lp_parts.append((Xp, blocks))
            p_par = Xp.shape[1]
            pstart.append(pof)
            nsdf.append(p_par)
            pnames = list(d.X.columns)
            if j > 0:
                pnames = [f"{nm}.{j}" for nm in pnames]
            names.extend(pnames)

            off = np.zeros(n)
            has_off = False
            for off_node in d.expanded.offsets:
                blk = _eval_atom(off_node, d.data)
                off = off + blk.values.flatten().astype(float)
                has_off = True
            offsets.append(off if has_off else None)

            col_cursor = pof + p_par
            n_slots_lp = 0
            slot_work_col: list[int] = []
            n_work_lp = 0
            id_first: dict[str, tuple[int, int]] = {}
            for b, bid in zip(blocks, block_ids):
                k = int(np.asarray(b.X).shape[1])
                a, bcol = col_cursor, col_cursor + k
                ranges_all.append((a, bcol))
                blocks_all.append(b)
                for i in range(1, k + 1):
                    names.append(f"{b.label}.{i}")
                for jS, S_j in enumerate(b.S):
                    slots.append(
                        _PenaltySlot(
                            block=b,
                            col_start=a,
                            col_end=bcol,
                            S=np.asarray(S_j, dtype=float),
                            S_scale=_block_s_scale(b, jS),
                        )
                    )
                col_cursor = bcol
                nS = len(b.S)
                n_slots_lp += nS
                if nS == 0:
                    continue
                if bid is None or bid not in id_first:
                    w0 = n_work_lp
                    n_work_lp += nS
                    if bid is not None:
                        id_first[bid] = (w0, nS)
                else:
                    w0, nc = id_first[bid]
                    if nS > nc:
                        raise ValueError(
                            "Later terms sharing an `id' can not have "
                            "more smoothing parameters than the first "
                            "such term"
                        )
                slot_work_col.extend(range(w0, w0 + nS))
            if n_work_lp == n_slots_lp:
                L_parts.append((n_slots_lp, n_work_lp, None))
            else:
                L_j = np.zeros((n_slots_lp, n_work_lp))
                L_j[np.arange(n_slots_lp), slot_work_col] = 1.0
                L_parts.append((n_slots_lp, n_work_lp, L_j))
            lpi.append(np.arange(pof, col_cursor))
            pof = col_cursor
        p = pof

        if all(Lj is None for _, _, Lj in L_parts):
            L = None
            n_work = len(slots)
        else:
            n_work = sum(w for _, w, _ in L_parts)
            L = np.zeros((len(slots), n_work))
            r0 = c0 = 0
            for nr_j, nc_j, Lj in L_parts:
                if nr_j:
                    L[r0 : r0 + nr_j, c0 : c0 + nc_j] = (
                        Lj if Lj is not None else np.eye(nr_j)
                    )
                r0 += nr_j
                c0 += nc_j

        var_index_lp = {nm: jv for jv, nm in enumerate(dframe.names)}
        self._param_tensor = [
            param_tensor_blocks(
                d.expanded, d.param_assign or [], Xp, dframe, var_index_lp
            )
            for d, (Xp, _blocks) in zip(ds, lp_parts)
        ]
        design, lpid = build_discrete_design_lps(
            lp_parts, dframe, param_tensors=self._param_tensor, fit_frame=True
        )
        assert design.p == p, (design.p, p)
        self._discrete_frame = dframe
        self._discrete_design = design
        self.data = data_m  # normalized frame for the shared tail

        lps = [
            types.SimpleNamespace(
                expanded=d.expanded,
                data=d.data,
                blocks=parts[1],
                param_assign=d.param_assign,
            )
            for d, parts in zip(ds, lp_parts)
        ]
        return _MultiDesign(
            lps=lps,
            X=DiscreteX(design, lpid),
            lpi=lpi,
            y=y,
            blocks=blocks_all,
            block_col_ranges=ranges_all,
            slots=slots,
            column_names=names,
            nsdf=nsdf,
            pstart=pstart,
            offsets=offsets,
            n_lp=len(ds),
            L=L,
            n_work=n_work,
            p=p,
            n=n,
        )

    def _object_R(self) -> np.ndarray:
        """mgcv's ``object$R`` for bam: a general family's gam.fit5 root,
        otherwise the R factor of the accumulated QR (``qrx$R``,
        bam.r:1346/1766), which ``_X_full`` holds."""
        if getattr(self, "_md", None) is not None:
            return self._R_fit5
        return self._X_full

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
        block_size: int | None = None,
        newdata_guaranteed: bool = False,
        na_action: str | None = "na.pass",
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

        A general-family discrete fit (formula list) routes whole-model
        link/response/lpmatrix to :meth:`_predict_bamd_general` (the
        per-LP binned kernel surface) and everything else — terms/iterms,
        term selection, family-``predict``-hooked response — to the
        exact-eval parent (:meth:`hea.gam._predict_general`).

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
            raise ValueError("se_fit=True is not allowed with type='lpmatrix'")
        # predict.bam's newdata stage: run predict.gam's shared
        # processing ONCE here (mgcv's predict.bamd does the same via
        # predict.gam(type="newdata"), bam.r:1806), then hand the
        # processed frame down with newdata_guaranteed so the parent
        # doesn't reprocess; NA-expand every output at this level.
        # predict.bam's block.size default is 50000 (bam.r:1416), not
        # predict.gam's 1000 — resolve before forwarding.
        na_mask = None
        if newdata is not None:
            newdata, na_mask = self._process_predict_newdata(
                newdata, newdata_guaranteed, na_action
            )
            newdata_guaranteed = True
        if block_size is None:
            block_size = 50000
        # Discrete GENERAL fit (formula list): whole-model link/response/
        # lpmatrix ride the binned per-LP kernel surface; terms/iterms,
        # terms=/exclude= selection and family-`predict`-hooked response
        # (gammals/gumbls/multinom) fall through to the exact-eval parent
        # (gam._predict_general via each block's spec.predict_mat) — the
        # same split as the single-LP discrete fit below, with the gfam
        # precedent for hooked families (mgcv's own hooked discrete
        # predict errors, bam.r:1995-2004 live receipt). newdata=None on
        # the parent path routes the training frame explicitly: md.X is
        # a DiscreteX the parent cannot np.asarray.
        if getattr(self, "_general_discrete", False):
            if offset is not None:
                raise NotImplementedError(
                    "offset= is not supported for multi-formula bam "
                    "predict; put offset(...) atoms in the per-LP "
                    "formulas."
                )
            fam_predict = getattr(self.family, "predict", None)
            if (
                type in ("link", "response", "lpmatrix")
                and terms is None
                and exclude is None
                and not (type == "response" and fam_predict is not None)
            ):
                return _na_pass_expand(
                    self._predict_bamd_general(
                        newdata, type=type, se_fit=se_fit, unconditional=unconditional
                    ),
                    na_mask,
                )
            return _na_pass_expand(
                super().predict(
                    newdata=self.data if newdata is None else newdata,
                    type=type,
                    se_fit=se_fit,
                    offset=offset,
                    unconditional=unconditional,
                    terms=terms,
                    exclude=exclude,
                    iterms_type=iterms_type,
                    block_size=block_size,
                    newdata_guaranteed=newdata_guaranteed,
                    na_action=na_action,
                ),
                na_mask,
            )

        # predict.bam routes a DISCRETE fit (``object$dinfo`` set) to
        # predict.bamd (bam.r:1421) — bin newdata's covariates to the
        # compress.df grid and gather via the discrete kernels, NOT exact
        # basis evaluation. hea mirrors this for the value / SE / lpmatrix
        # surface; ``terms``/``iterms`` and term selection stay on the
        # exact-eval parent (the per-term discrete decomposition with
        # mgcv's parametric-term grouping is not ported — see
        # :meth:`_predict_bamd`).
        # gfam stays OFF the discrete surface: mgcv's predict.bamd routes
        # hooked families to ``family$predict`` with the DISCRETE-kernel
        # X list and never calls ``setInd`` (bam.r:1995-2004, contrast
        # predict.gam's setInd at mgcv.r:3133) — a discrete gfam predict
        # errors in mgcv itself ("NA/NaN argument", live 1.9-4 receipt,
        # for type="link" too). hea routes it through the exact-eval
        # parent instead, where the family predict hook re-derives the
        # index from newdata.
        if (
            self._discrete_design is not None
            and type in ("link", "response", "lpmatrix")
            and terms is None
            and exclude is None
            and not isinstance(self.family, _gfam_family)
        ):
            return _na_pass_expand(
                self._predict_bamd(
                    newdata,
                    type=type,
                    se_fit=se_fit,
                    offset=offset,
                    unconditional=unconditional,
                ),
                na_mask,
            )

        if newdata is not None:
            return _na_pass_expand(
                super().predict(
                    newdata=newdata,
                    type=type,
                    se_fit=se_fit,
                    offset=offset,
                    unconditional=unconditional,
                    terms=terms,
                    exclude=exclude,
                    iterms_type=iterms_type,
                    block_size=block_size,
                    newdata_guaranteed=newdata_guaranteed,
                    na_action=na_action,
                ),
                na_mask,
            )

        if (
            type in ("terms", "iterms")
            or terms is not None
            or exclude is not None
            or unconditional
        ):
            return super().predict(
                newdata=self.data,
                type=type,
                se_fit=se_fit,
                offset=offset,
                unconditional=unconditional,
                terms=terms,
                exclude=exclude,
                iterms_type=iterms_type,
                block_size=block_size,
            )

        extra: np.ndarray | None = None
        if offset is not None:
            extra = np.asarray(offset, dtype=float).flatten()
            if extra.shape != (self.n,):
                raise ValueError(f"offset must have length {self.n}, got {extra.shape}")

        if type in ("link", "response") and not se_fit and extra is None:
            if type == "link":
                return pl.DataFrame({"fit": self.linear_predictors.copy()})
            return pl.DataFrame({"fit": self.fitted_values.copy()})

        if type == "lpmatrix":
            return super().predict(
                newdata=self.data,
                type="lpmatrix",
                se_fit=False,
                offset=None,
            )

        eta = self.linear_predictors.copy()
        if extra is not None:
            eta = eta + extra

        if not se_fit:
            if type == "link":
                return pl.DataFrame({"fit": eta})
            return pl.DataFrame({"fit": self.family.link.linkinv(eta)})

        var_eta = self._chunked_var_eta_diag(self.Vp)
        se_link = np.sqrt(np.maximum(var_eta, 0.0))
        if type == "link":
            return pl.DataFrame({"fit": eta, "se.fit": se_link})
        mu = self.family.link.linkinv(eta)
        mu_eta = self.family.link.mu_eta(eta)
        return pl.DataFrame({"fit": mu, "se.fit": se_link * np.abs(mu_eta)})

    def _build_predict_discrete_design(self, newdata):
        """Re-discretise ``newdata`` and assemble its binned
        :class:`DiscreteDesign`, mirroring mgcv ``predict.bamd``'s discrete
        setup (bam.r:1843-1921).
        Like ``predict.bamd`` (bam.r:1847 calls ``discrete.mf`` with no
        (bam.r:1889), exactly as the fitter does.
        """
        newdata = normalize_data(newdata)
        expr_map = _smooth_arg_expr_map(self._expanded)
        if expr_map:
            newdata = _apply_smooth_arg_exprs(newdata, expr_map)
        X_param = (
            materialize(
                self._expanded, newdata, basis_state=getattr(self, "_basis_state", None)
            )
            .to_numpy()
            .astype(float)
        )
        # Discretise the smooth covariates at the default resolution, plus the
        # fit's parametric variables (mgcv replays ``dinfo$pmf.names``,
        # bam.r:1847, dropping the ones newdata lacks).
        specs = _smooth_specs_from_expanded(self._expanded, newdata)
        names_pmf = [c for c in self._pmf_names if c in newdata.columns]
        frame = discrete_mf(specs, newdata, names_pmf=names_pmf, m=None)
        # mgcv re-runs terms2tensor on the prediction frame (bam.r:1854).
        param_tensor = param_tensor_blocks(
            self._expanded,
            self._param_assign,
            X_param,
            frame,
            {nm: j for j, nm in enumerate(frame.names)},
        )
        design = build_discrete_design(
            self._blocks, X_param, frame, data=newdata, param_tensor=param_tensor
        )
        off_new = np.zeros(newdata.height)
        for off_node in self._expanded.offsets:
            blk = _eval_atom(off_node, newdata)
            off_new = off_new + blk.values.flatten().astype(float)
        return design, newdata.height, off_new

    def _predict_bamd(self, newdata, *, type, se_fit, offset, unconditional):
        """Predict from a discrete bam fit via the discrete kernels — port
        of mgcv ``predict.bamd`` (bam.r:1773-2033), value/SE/lpmatrix
        surface.
        (bam.r:1989-1990).
        """
        beta = np.asarray(self.coefficients).reshape(-1)
        extra = None
        if offset is not None:
            extra = np.asarray(offset, dtype=float).flatten()

        reuse = newdata is None and self._discrete_m is None
        if reuse:
            design = self._discrete_design
            n_pred = self.n
        else:
            src = self.data if newdata is None else newdata
            design, n_pred, off_new = self._build_predict_discrete_design(src)

        if extra is not None and extra.shape != (n_pred,):
            raise ValueError(f"offset must have length {n_pred}, got {extra.shape}")

        if type == "lpmatrix":
            p = design.p
            Xf = np.empty((n_pred, p), dtype=float)
            ek = np.zeros(p, dtype=float)
            for kk in range(p):
                ek[kk] = 1.0
                Xf[:, kk] = Xbd(design, ek)[:n_pred]
                ek[kk] = 0.0
            return Xf

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

        V = self.Vp
        if unconditional:
            Vc = getattr(self, "Vc", None)
            if Vc is not None:
                V = Vc
        var_eta = diagXVXd(design, V)[:n_pred]  # mgcv diagXVXd, no materialise
        se_link = np.sqrt(np.maximum(var_eta, 0.0))
        if type == "link":
            return pl.DataFrame({"fit": eta, "se.fit": se_link})
        if reuse and extra is None:
            mu = self.fitted_values.copy()
        else:
            mu = self.family.link.linkinv(eta)
        mu_eta = self.family.link.mu_eta(eta)
        return pl.DataFrame({"fit": mu, "se.fit": se_link * np.abs(mu_eta)})

    def _build_predict_discrete_design_general(self, newdata):
        """Re-discretise ``newdata`` for a multi-LP (general-family) fit —
        the formula-list twin of :meth:`_build_predict_discrete_design`
        (mgcv ``predict.bamd``'s setup, bam.r:1843-1921, which mgcv can
        never reach for a general family): one shared ``discrete.mf``
        pass at the DEFAULT resolution over every LP's smooth terms and
        parametric covariates (no response column — prediction is
        response-free), each LP's FROZEN smooth blocks re-evaluated on
        the new grid, terms assembled LP-contiguously exactly like the
        fit build. Returns ``(design, n_user, offs)`` where ``offs`` are
        the per-LP formula offsets on ``newdata`` (stub-trimmed).
        """
        md = self._md
        newdata = normalize_data(newdata)
        expr_map: dict = {}
        for lp in md.lps:
            expr_map.update(_smooth_arg_expr_map(lp.expanded))
        if expr_map:
            newdata = _apply_smooth_arg_exprs(newdata, expr_map)
        specs: list[dict] = []
        names_pmf: list[str] = []
        lp_parts: list[tuple[np.ndarray, list]] = []
        for lp in md.lps:
            Xp_df = materialize(
                lp.expanded, newdata, basis_state=getattr(lp, "basis_state", None)
            )
            X_param = Xp_df.to_numpy().astype(float)
            if X_param.shape[1] == 0:
                X_param = np.zeros((newdata.height, 0))
            lp_parts.append((X_param, lp.blocks))
            specs.extend(_smooth_specs_from_expanded(lp.expanded, newdata))
            for nm in _pmf_names(lp.expanded, None, newdata.columns):
                if nm not in names_pmf:
                    names_pmf.append(nm)
        dframe = discrete_mf(specs, newdata, names_pmf=names_pmf, m=None)
        var_index_new = {nm: j for j, nm in enumerate(dframe.names)}
        param_tensors = [
            param_tensor_blocks(
                lp.expanded,
                getattr(lp, "param_assign", None) or [],
                Xp,
                dframe,
                var_index_new,
            )
            for lp, (Xp, _blk) in zip(md.lps, lp_parts)
        ]
        design, lpid_new = build_discrete_design_lps(
            lp_parts, dframe, param_tensors=param_tensors
        )
        assert lpid_new == md.X.lpid, (lpid_new, md.X.lpid)

        offs: list[np.ndarray | None] = []
        for lp in md.lps:
            if lp.expanded.offsets:
                off_j = np.zeros(newdata.height)
                for off_node in lp.expanded.offsets:
                    blk = _eval_atom(off_node, newdata)
                    off_j = off_j + blk.values.flatten().astype(float)
                offs.append(off_j)
            else:
                offs.append(None)
        return design, newdata.height, offs

    def _predict_bamd_general(self, newdata, *, type, se_fit, unconditional):
        """``predict.bamd`` for the discrete GENERAL fit — the per-LP
        value / SE / lpmatrix surface via the discrete kernels. mgcv has
        no comparator (``bam.r:2653`` stops before any general family is
        fitted); the protocol composes hea's two existing ports:
        predict.bamd's binned re-discretisation (:meth:`_predict_bamd` —
        newdata re-gridded at ``compress.df``'s default resolution,
        values gathered by ``Xbd``, link-scale variance by ``diagXVXd``)
        applied per linear predictor with ``lt=lpid[j]`` term selection
        (the ``gamlss.gH`` convention, gamlss.r:625-665), and
        predict.gam's multi-LP output semantics (one ``fit``/``fit.{j}``
        column per LP, per-LP linkinv + delta-method response SE —
        :meth:`hea.gam._predict_general`, the same-design dense referee).
        discrete predict is broken, bam.r:1995-2004 live receipt).
        """
        md = self._md
        lpid = md.X.lpid
        beta = np.asarray(self.coefficients, dtype=float).reshape(-1)

        reuse = newdata is None and self._discrete_m is None
        if reuse:
            design = self._discrete_design
            n_pred = int(md.n)
            offs = list(md.offsets)
        else:
            design, n_pred, offs = self._build_predict_discrete_design_general(
                self.data if newdata is None else newdata
            )

        if type == "lpmatrix":
            p = design.p
            Xf = np.empty((n_pred, p), dtype=float)
            ek = np.zeros(p, dtype=float)
            for kk in range(p):
                ek[kk] = 1.0
                Xf[:, kk] = Xbd(design, ek)[:n_pred]
                ek[kk] = 0.0
            return Xf

        V = None
        if se_fit:
            V = self.Vp
            if unconditional:
                Vc = getattr(self, "Vc", None)
                if Vc is not None:
                    V = Vc
        fits: list[np.ndarray] = []
        ses: list[np.ndarray] = []
        for j in range(len(md.lpi)):
            if reuse:
                eta_j = np.asarray(self.linear_predictors, dtype=float)[:, j].copy()
            else:
                eta_j = Xbd(design, beta, lt=lpid[j])[:n_pred]
                if offs[j] is not None:
                    eta_j = eta_j + offs[j]
            if se_fit:
                cols = np.asarray(md.lpi[j], dtype=int)
                var_j = diagXVXd(design, V[np.ix_(cols, cols)], lt=lpid[j])[:n_pred]
                se_j = np.sqrt(np.maximum(var_j, 0.0))
            if type == "response":
                link_j = self.family.links[j]
                fits.append(link_j.linkinv(eta_j))
                if se_fit:
                    ses.append(np.abs(link_j.mu_eta(eta_j)) * se_j)
            else:
                fits.append(eta_j)
                if se_fit:
                    ses.append(se_j)
        return self._general_response_frame(
            np.column_stack(fits), np.column_stack(ses) if se_fit else None
        )

    def _fit_given_rho(self, rho: np.ndarray, *, with_eta: bool = True) -> _FitState:
        """Closed-form Gaussian-on-(R, f) solve at fixed ρ. ``with_eta=False``
        skips the pass over the data for the linear predictor (``eta``,
        ``mu`` and ``z`` come back None), for a caller that forms its own."""
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
        # Sl.fitChol's chol(t(XXp/d)/d, pivot=TRUE) (fast-REML.r:1606)
        A_pre = (A / d[:, None]).T / d[:, None]
        R_pre, piv_1based, rank_A = r_chol(A_pre, pivot=True)
        # mgcv Sl.fitChol:1607 ``r <- min(attr(R,"rank"), Rrank(R))`` — same
        # min as ``_pi_fit_chol``: dpstrf's pivot rank ∧ mgcv's Cline-condition
        # ``Rrank`` (mgcv.r:4, tol=eps^0.9). Full-rank → p (no change); only
        # bites when a near-singular leading direction survives dpstrf's tol.
        rank_A = min(int(rank_A), _R_rank(R_pre, tol=float(np.finfo(float).eps) ** 0.9))
        piv = np.asarray(piv_1based, dtype=int) - 1
        ipiv = np.empty(self.p, dtype=int)
        ipiv[piv] = np.arange(self.p)

        Xy_over_d = self._Xty / d
        if rank_A > 0:
            b_piv = Xy_over_d[piv]
            z = solve_triangular(
                R_pre[:rank_A, :rank_A].T,
                b_piv[:rank_A],
                lower=True,
            )
            beta_piv_top = solve_triangular(
                R_pre[:rank_A, :rank_A],
                z,
                lower=False,
            )
        else:
            beta_piv_top = np.zeros(0, dtype=float)
        beta_piv = np.zeros(self.p, dtype=float)
        beta_piv[:rank_A] = beta_piv_top
        beta = beta_piv[ipiv] / d

        if not np.all(np.isfinite(beta)):
            raise FloatingPointError("non-finite β in bam (R, f) solve")

        try:
            A_chol, lower = cho_factor(A, lower=True, overwrite_a=False)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.trace(A) / max(self.p, 1)
            A_chol, lower = cho_factor(
                A + ridge * np.eye(self.p),
                lower=True,
                overwrite_a=False,
            )

        pen = float(beta @ Sλ @ beta)
        rss = float(self._yty - 2.0 * (beta @ self._Xty) + beta @ self._XtX @ beta)
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
        if not with_eta:
            eta = mu = z_full = None
        elif isinstance(self.family, Gaussian) and self.family.link.name == "identity":
            eta = self._chunked_xbeta(beta) + self._offset  # full η, length-n
            mu = eta  # identity link short-circuit
            z_full = self._y_arr - self._offset
        else:
            eta = self._chunked_xbeta(beta) + self._offset
            mu = self.family.link.linkinv(eta)
            mu_eta = self.family.link.mu_eta(eta)
            safe_mu_eta = np.where(mu_eta != 0, mu_eta, 1.0)
            z_full = (eta - self._offset) + (self._y_arr - mu) / safe_mu_eta
        n = self.n
        return _FitState(
            beta=beta,
            dev=rss,
            pen=pen,
            A_chol=A_chol,
            A_chol_lower=lower,
            S_full=Sλ,
            log_det_A=log_det_A,
            eta=eta,
            mu=mu,
            w=np.ones(n),
            z=z_full,
            alpha=np.ones(n),
            is_fisher_fallback=True,
            # Reduced Gaussian working fit (mgcv swaps G$family <-
            # gaussian() / G$w <- 1 before its compressed refits,
            # bam.r:932/1266-1267): criterion-tail readers take the
            # working-gaussian view (P ≡ dev, dW/dη ≡ 0, p-space).
            is_working_gaussian=True,
        )

    def _dw_deta(self, fit: _FitState) -> np.ndarray:
        """∂w/∂η for Gaussian-identity: identically zero."""
        return np.zeros(self.p)

    def _d2w_deta2(self, fit: _FitState) -> np.ndarray:
        """∂²w/∂η² for Gaussian-identity: identically zero. Length-p so
        ``np.any(d2w_deta2)`` evaluates against the right-shape array
        and ``_reml_hessian``'s ``needs_w`` gate stays correct."""
        return np.zeros(self.p)

    def _gam_g_log_scale0(self) -> float:
        """The initial log φ of mgcv's compressed ``gam(G=G)`` refit
        (estimate.gam, mgcv.r:2027-2033): ``log(in.out$scale)`` when in.out
        is given, else ``log(null.scale/10)`` from ``get.null.coef`` on the
        gaussian working problem, ``null.scale = Σ(f − mean(f))²/nrow(R)``."""
        if getattr(self, "_in_out", None) is not None:
            return self._in_out_log_scale()
        f_red = np.asarray(self._bam_qr.f, dtype=float)
        dev0 = f_red - r_mean(f_red)
        null_scale = float(r_colsums(dev0 * dev0)[0]) / f_red.size
        return float(np.log(null_scale / 10.0))

    def _bam_outer_criterion(self) -> str:
        """Outer-Newton criterion for the non-(f)REML/ML methods — the same
        scoreType mapping mgcv's compressed ``gam(G=G, method=method)``
        refit resolves to (mgcv.r:1945-1959): P-REML/P-ML → the
        Pearson-Laplace score (known-scale cases already collapsed to
        REML/ML in ``__init__``); GACV.Cp → GACV when φ is estimated,
        UBRE (via the GCV closures) when φ is known; GCV.Cp → GCV/UBRE."""
        if self.method in ("P-REML", "P-ML"):
            return "PREML"
        if self.method == "GACV.Cp" and not self._scale_known_fit:
            return "GACV"
        return "GCV"

    def _score_all(
        self,
        rho,
        log_phi,
        fit,
        deriv,
        scoreType,
        *,
        include_log_phi=False,
        include_family_theta=False,
        T_work=None,
    ):
        """gam's `_score_all` on the working-Gaussian view of the reduced
        problem. mgcv swaps ``G$family <- gaussian()`` and ``G$w <- 1``
        before every compressed refit (bam.r:932; bam.r:1265-1267 sets
        ``G$y <- qrx$f``, ``G$w <- 1``), so the criterion tail there reads
        gaussian ``ls``/variance/link quantities and unit prior weights —
        the response family lives only in the outer PIRLS loop that
        rebuilds (R, f). Bind the same view here.

        For GACV.Cp's GACV/GCV tail the refit is gam.fit3 on ``(R, f)``
        with ``nobs = ncol(R)`` (gam.fit3.r:744-752): its deviance is the
        reduced rows' residual sum ``‖f − Rβ‖²``, summed from residuals as
        gaussian ``dev.resids`` are, while ``dev.extra``/``n.true`` reach
        only the scale and Pearson-φ estimates (gam.fit3.r:641/648).
        Forming that sum from the residuals, rather than as the full-data
        RSS minus ``rss.extra``, keeps it accurate where the fit nearly
        interpolates the reduced rows and the score divides it by
        ``nobs − tr(A)``.
        """
        need_w = scoreType in ("REML", "PREML")
        dw_deta = self._dw_deta(fit) if (need_w and deriv >= 1) else None
        d2w_deta2 = self._d2w_deta2(fit) if (need_w and deriv >= 2) else None
        n_score = self.n
        fit_view = fit
        if self.method == "GACV.Cp" and scoreType in ("GACV", "GCV"):
            qrx = getattr(self, "_bam_qr", None)
            if qrx is not None:
                R_red = np.asarray(qrx.R, dtype=float)
                f_red = np.asarray(qrx.f, dtype=float)
                E_aug = _penalty_root_of(
                    self._slots,
                    self.p,
                    self._UrS,
                    self._reparam_Y,
                    self._keep_cols,
                    self._reparam_cache,
                    np.asarray(rho, dtype=float),
                    H_fixed=self._h_fixed_fit(),
                )
                b_red, _r, _ld, ok = _pls_qr(
                    R_red, self._pls_lwork, np.ones(R_red.shape[0]), f_red, E_aug
                )
                if not ok:
                    b_red = np.asarray(fit.beta, dtype=float)
                r_red = f_red - r_matprod(R_red, b_red[:, None])[:, 0]
                fit_view = _copy.copy(fit)
                fit_view.dev = float(r_colsums(r_red * r_red)[0])
                fit_view.rss = fit_view.dev
                n_score = self.p
        return _gam_fit3_score(
            rho,
            log_phi,
            fit_view,
            deriv,
            scoreType,
            T_work=T_work,
            include_log_phi=include_log_phi,
            include_family_theta=include_family_theta,
            dw_deta=dw_deta,
            d2w_deta2=d2w_deta2,
            Mp=self._Mp,
            wt=self._crit_wt,
            y=self._y_arr,
            binom_n=None,
            gamma=self._gamma,
            family=self._crit_family,
            family_mgcv_extended=False,
            use_ml_proj=self._use_ml_proj,
            pearson_scale_criterion=self._pearson_scale_criterion,
            reml_ind=self._reml_ind,
            penalty_rank=self._penalty_rank,
            slots=self._slots,
            p=self.p,
            UrS=self._UrS,
            reparam_cache=self._reparam_cache,
            X=self._X_full,
            XtX=self._XtX,
            n=n_score,
            scale_fixed_value=self._scale_fixed_value,
            scale_known=self._scale_known_fit,
            pls_lwork=self._pls_lwork,
        )

    def _ml_flavor_swap(
        self, family_aic: float, log_phi_hat: float, dev: float
    ) -> float:
        """The log-lik flavor difference in mgcv's final bam ML report.
        (bam.r:2784-2787) — while hea's ``self._reml`` carries the
        ``dev/φ̂ − 2·ls(φ̂)`` form. The two coincide for poisson/binomial/Tweedie
        (deviance–saturated-loglik identity) but differ for gaussian (``$aic`` at
        the MLE scale, ``+2``, ``−Σlog w``) and Gamma (``+2``, dispersion
        plug-in). Returns the additive correction in hea's 2V units.
        """
        phi = float(np.exp(log_phi_hat))
        if self._family_mgcv_extended:
            ls0 = float(
                self.family.ls_extended(
                    self._y_arr, self._wt, theta=self.family.get_theta(), scale=phi
                )["ls"]
            )
        elif self._binom_n is not None:
            ls0 = float(self.family.ls(self._y_arr, self._wt, phi, n=self._binom_n)[0])
        else:
            ls0 = float(self.family.ls(self._y_arr, self._wt, phi)[0])
        return (family_aic - (dev / phi - 2.0 * ls0)) / self._gamma

    def _compute_edf12(self, rho, fit, sigma_squared, A_inv, A_inv_XtWX, edf, H_aug):
        """mgcv-bam's fREML edf1/edf2/Vc — ``Sl.postproc`` (fast-REML.r:
        1997-2018) and ``bgam.fitd`` (bam.r:824-881) share the recipe::

            edf1 = 2·edf − rowSums(Fᵀ∘F)               (F = A⁻¹·X'WX)
            rV   = (Λ₊^{-1/2}·Vᵀ)[, 1:M]   from eigen(outer hess)
            Vc   = Vp + d1b·(rVᵀ·rV)·d1bᵀ
            edf2 = rowSums(Vc ∘ X'WX)/φ,  Σedf2 capped at Σedf1
                                                        (bam.r:2789)

        The column truncation keeps the ρ coordinates of every
        eigenvector — the ρρ block of the positive-eigenspace
        pseudo-inverse with the log-φ coupling marginalised. There is NO
        Vc2 Cholesky-derivative term and NO 0.1-prior regularisation —
        those belong to gam.fit3.post.proc, which mgcv-bam reaches only
        through the explicit REML/ML/P-REML/P-ML ``gam(G=G)`` route
        (bam.r:1263-1273): those methods dispatch to the inherited gam
        machinery. GCV.Cp/GACV.Cp also use super(): mgcv leaves edf2 NULL
        there and the AIC df below never reads it.

        The outer Hessian: the discrete POI's last accepted
        ``Sl.fitChol`` proposal (mgcv ``prop$hess``, bam.r:869) or
        ``fast.REML.fit``'s final Newton Hessian (mgcv
        ``fit$outer.info$hess``, fast-REML.r:2010) — both stashed by the
        fit loops; the ``_reml_hessian``-based ``H_aug`` argument is the
        fallback. d1b: ``_db_drho`` at the converged fit ≡ mgcv's
        unwound ``fit$d1b``/``prop$db`` (same ∂β̂/∂ρ in original
        coordinates).
        """
        if self._method_in != "fREML":
            return super()._compute_edf12(
                rho, fit, sigma_squared, A_inv, A_inv_XtWX, edf, H_aug
            )
        F = A_inv_XtWX
        edf1 = 2.0 * np.diag(F) - np.einsum("ij,ji->i", F, F)
        p = F.shape[0]
        n_sp = len(self._slots)
        pi_out = getattr(self, "_pi_last_out", None)
        oi = getattr(self, "_outer_info", None)
        if pi_out is not None:
            H = np.asarray(pi_out["hess"], dtype=float)
        elif (
            oi is not None
            and oi.get("hess") is not None
            and np.asarray(oi["hess"]).size > 0
        ):
            H = np.asarray(oi["hess"], dtype=float)
        else:
            H = H_aug
        if (
            n_sp == 0
            or H is None
            or H.size == 0
            or not (np.isfinite(sigma_squared) and sigma_squared > 0)
        ):
            return edf.copy(), edf1, np.zeros((p, p))
        db = self._db_drho(rho, fit.beta, fit.A_chol, fit.A_chol_lower)
        if self._L is not None:
            db = db @ self._L  # d1b <- d1b %*% L (fast-REML.r:2008)
        M = db.shape[1]
        # Sl.postproc (fast-REML.r:2010-2015): ev <- eigen(hess); 1/sqrt of
        # the positive values; rV <- (values*t(vectors))[,1:M];
        # Vc <- crossprod(rV %*% t(d1b)).
        vals, vecs = _r_eigen_sym(H)
        pos = vals > 0.0
        d = np.zeros_like(vals)
        d[pos] = 1.0 / np.sqrt(vals[pos])
        rV = (d[:, None] * np.ascontiguousarray(vecs.T))[:, :M]
        Vc_corr = r_crossprod(r_matprod(rV, np.ascontiguousarray(db.T)))
        edf2 = edf + np.einsum("ij,ij->i", Vc_corr, self._XtX) / sigma_squared
        if edf2.sum() > edf1.sum():  # bam.r:2789 (sum cap, not element-wise)
            edf2 = edf1.copy()
        return edf2, edf1, Vc_corr

    def _preml_hessian(self, rho, fit=None):
        """Working-Gaussian view of gam's `_preml_hessian` shim (the
        P-REML/P-ML H_aug for Vc/edf2) — same family/weight swap as
        `_score_all`, so the φ-row pieces inside the chain rule read the
        gaussian ``ls`` mgcv's compressed refit uses."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        dw_deta = self._dw_deta(fit)
        d2w_deta2 = self._d2w_deta2(fit)
        return _preml_hessian(
            rho,
            fit,
            slots=self._slots,
            Mp=self._Mp,
            n=self.n,
            family=self._crit_family,
            wt=self._crit_wt,
            y=self._y_arr,
            X=self._X_full,
            p=self.p,
            gamma=self._gamma,
            pearson_scale_criterion=self._pearson_scale_criterion,
            use_ml_proj=self._use_ml_proj,
            penalty_rank=self._penalty_rank,
            family_mgcv_extended=False,
            UrS=self._UrS,
            reparam_cache=self._reparam_cache,
            reml_ind=self._reml_ind,
            dw_deta=dw_deta,
            d2w_deta2=d2w_deta2,
        )

    def _in_out_rho0(self) -> np.ndarray | None:
        """bam's ``in.out$sp`` λ seed: ``log(in.out$sp)`` in FULL
        per-penalty space, replacing ``initial.sp`` (bam.r:687/1229/
        1687 — the non-discrete rail re-reads it every PIRLS iter).
        None when ``in_out`` is unset."""
        if getattr(self, "_in_out", None) is None:
            return None
        sp = self._in_out.get("sp") if isinstance(self._in_out, dict) else None
        if sp is None:
            raise ValueError("in.out incorrect: see documentation")
        sp = np.asarray(sp, dtype=float).reshape(-1)
        if sp.size != len(self._slots):
            raise ValueError("in.out incorrect: see documentation")
        return np.log(sp)

    def _in_out_log_scale(self) -> float:
        """``log(in.out$scale)`` — the log φ seed replacement
        (bam.r:701/1237); mgcv errors on a missing scale entry."""
        scale = self._in_out.get("scale") if isinstance(self._in_out, dict) else None
        if scale is None:
            raise ValueError("in.out incorrect: see documentation")
        return float(np.log(float(scale)))

    def _initial_sp_rho(self) -> np.ndarray:
        """``log(initial.sp(qrx$R, G$S, G$off))``: from the accumulated QR
        factor (bam.r:1229, 1687), or from the raw ``X'WX`` with ``XX =
        TRUE`` on the discrete rail (bam.r:687)."""
        if self._discrete_design is not None:
            return np.log(_initial_sp(self._XtX, self._slots, XX=True))
        return np.log(_initial_sp(self._bam_qr.R, self._slots))

    def _bam_magic(self) -> dict:
        """bam's GCV.Cp step (bam.r:1678 for bam.fit, bam.r:1220 per PIRLS
        iteration in bgam.fit): ``magic(qrx$f, qrx$R, G$sp, G$S, G$off, L,
        lsp0, rank, H, C = 0 rows, gamma, scale, gcv = scale <= 0,
        extra.rss = y.norm2 - sum(f^2), n.score = n)`` on the accumulated
        QR, then ``magic.post.proc(qrx$R, fit, 1)``. Keeps both as
        ``self._bam_magic_res`` and returns magic's list."""
        qrx = self._bam_qr
        R = qrx.R
        f = qrx.f
        n_p = len(self._slots)
        scale = float(self._scale_fixed_value) if self._scale_known_fit else -1.0
        fit = magic(
            f,
            R,
            np.full(self._work_dim, -1.0),
            self._slots,
            L=self._L,
            lsp0=np.zeros(n_p) if self._lsp0 is None else self._lsp0,
            rank=[_penalty_rank(sl.S) for sl in self._slots],
            H=self._h_fixed_fit(),
            C=np.zeros((0, R.shape[1])),
            gamma=self._gamma,
            scale=scale,
            gcv=scale <= 0,
            extra_rss=float(qrx.y_norm2 - r_sum(f * f)),
            n_score=int(self.n),
        )
        post = magic_post_proc(R, fit, w=np.ones(f.size))
        self._bam_magic_res = {"fit": fit, "post": post}
        self._mgcv_conv = fit["gcv_info"]
        return fit

    def _x_rows(self, start: int, end: int) -> np.ndarray:
        """Rows ``start:end`` of the full model matrix, as bgam.fit's
        ``predict(G, newdata = mf[ind, ], type = "lpmatrix")`` builds them
        (bam.r:1064). Kept, read-only, in ``self._x_cache`` when the whole
        matrix fits ``_X_CACHE_BYTES``, so later passes reuse them."""
        cache = getattr(self, "_x_cache", None)
        if cache is None and self.n * self.p * 8 <= _X_CACHE_BYTES:
            cache = self._x_cache = {}
        X = None if cache is None else cache.get((start, end))
        if X is None:
            X = _materialize_chunk(
                self._blocks, self.data[start:end], self._X_param_full[start:end]
            )
            if cache is not None:
                X.flags.writeable = False
                cache[(start, end)] = X
        return X

    def _chunked_xbeta(self, beta: np.ndarray) -> np.ndarray:
        """Compute ``X·β`` over the full data, chunk by chunk. ``O(n·p)``
        time, ``O(chunk_size·p)`` peak memory.
        """
        if self._discrete_design is not None:
            return Xbd(self._discrete_design, beta)
        n = self.n
        out = np.empty(n, dtype=float)
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = self._x_rows(start, end)
            out[start:end] = r_matprod(X_chunk, np.reshape(beta, (-1, 1)))[:, 0]
        return out

    @property
    def _cmX(self) -> np.ndarray:
        """mgcv ``object$cmX`` — true design column means, computed lazily."""
        if self._cmX_cache is None:
            self._cmX_cache = self._chunked_colmeans()
        return self._cmX_cache

    def _chunked_colmeans(self) -> np.ndarray:
        """Column means of the full n×p design (backs :attr:`_cmX`)."""
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
                    cols.append(
                        np.asarray(
                            b.spec.predict_mat(self.data[start:end]),
                            dtype=float,
                        )
                    )
                acc += np.concatenate(cols, axis=1).sum(axis=0)
            return acc / n
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = self._x_rows(start, end)
            acc += X_chunk.sum(axis=0)
        return acc / n

    def _chunked_leverage_diag(self, A_inv: np.ndarray) -> np.ndarray:
        """Diagonal of the unweighted hat matrix ``H = X·A⁻¹·X'``."""
        n = self.n
        out = np.empty(n, dtype=float)
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = self._x_rows(start, end)
            HX = X_chunk @ A_inv
            out[start:end] = (HX * X_chunk).sum(axis=1)
        return out

    def _chunked_var_eta_diag(self, Vp: np.ndarray) -> np.ndarray:
        """``diag(X·Vp·X')`` over the full data, chunk by chunk."""
        n = self.n
        out = np.empty(n, dtype=float)
        if self._discrete_design is not None:
            return diagXVXd(self._discrete_design, Vp)
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = self._x_rows(start, end)
            HX = X_chunk @ Vp
            out[start:end] = (HX * X_chunk).sum(axis=1)
        return out

    def _fit3_args_compressed(self, method) -> dict | None:
        """gam.outer's fixed arguments to newton's gam.fit3 calls in mgcv's
        compressed refit ``gam(G=G, method=method)`` (bam.r:1722-1733): the
        model matrix and response are ``qrx$R`` and ``qrx$f``, the weights
        1, the offset 0, ``dev.extra`` and ``pearson.extra`` the residual
        sum of squares outside ``R``'s span and ``n.true`` the data's n;
        the null coefficients are get.null.coef's on that problem. ``None``
        leaves the route to bam's own outer Newton."""
        if (
            not self._slots
            or getattr(self, "_UrS", None) is None
            or self._keep_cols is not None
            or getattr(self, "_bam_qr", None) is None
        ):
            return None
        qr = self._bam_qr
        R = np.asfortranarray(np.asarray(qr.R, dtype=float))
        f = np.asarray(qr.f, dtype=float)
        nq = f.size
        # bam's scale, -1 when estimated (bam.r:2577), is gam(G=G)'s
        scale = self._scale_fixed_value if self._scale_known_fit else -1.0
        if method == "GACV.Cp":
            score_type = "UBRE" if scale > 0 else "GACV"
        else:
            score_type = method
        Y, Z, Eb = _total_penalty_space(self._slots, self.p, self._H_fixed)
        nc = r_qr_coef(r_qr(R), np.full(nq, r_mean(f)))
        nc[np.isnan(nc)] = 0.0
        return {
            "X": R,
            "x_finite": r_matprod_safe(R),
            "cache": {},
            "family": self._crit_family,
            "y": f,
            "Eb": Eb,
            "UrS": self._UrS,
            "U1": np.hstack([Y, Z]),
            "Mp": Z.shape[1],
            "weights": np.ones(nq),
            "offset": np.zeros(nq),
            "null_coef": nc,
            "binom_n": None,
            "n_true": float(self.n),
            "pearson_extra": float(qr.rss_extra),
            "dev_extra": float(qr.rss_extra),
            "scale": scale,
            "score_type": score_type,
            "sname": "REML"
            if score_type in ("REML", "ML", "P-REML", "P-ML")
            else score_type,
            "gamma": self._gamma,
            "epsilon": 1e-7,
            "maxit": 200,
            "scale_est": "fletcher",
            "start": None,
            "mustart": None,
            "etastart": None,
            "fit4": False,
            "nth": 0,
            "use_wz": False,
        }

    def _post_fit_gaussian(
        self, fit, rho_hat: np.ndarray, X_param_df: pl.DataFrame
    ) -> None:
        """Populate the user-facing attributes once the smoothing parameters
        are chosen on the (R, f) reduced problem. Mirrors the post-fit block
        in ``gam.__init__`` for the Gaussian-identity case: the linear
        predictor and residuals take one chunked pass over the data (the
        leverage is left to the lazy property), and the β/Vp/Ve/edf algebra
        is the same (XtWX = X'X = R'R when W = I).
        """
        n, p = self.n, self.p
        method = self.method
        n_sp = len(self._slots)
        beta = fit.beta
        self._rho_hat = rho_hat
        # GCV.Cp: magic's fit and magic.post.proc's (bam.r:1740-1755).
        mg = getattr(self, "_bam_magic_res", None) if method == "GCV.Cp" else None
        if mg is not None:
            self.full_sp = np.asarray(mg["fit"]["sp_full"], dtype=float)
        else:
            self.full_sp = np.exp(np.asarray(rho_hat, dtype=float))

        if fit.A_inv is not None:
            A_inv = fit.A_inv
        else:
            A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        XtWX = self._XtX
        A_inv_XtWX = A_inv @ XtWX
        # bam.fit's fREML rail reports Sl.postproc's edf, covariances and
        # edf1/edf2 (bam.r:1693-1699) and the REML estimate of the scale
        # (bam.r:1704).
        post = None
        fr = getattr(self, "_freml", None)
        if fr is not None and self._method_in == "fREML" and n_sp > 0:
            post = _sl_postproc(
                self._sl,
                fr["fit"],
                fr["undrop"],
                self._bam_qr.R,
                fr["hess"],
                scale=(self._scale_fixed_value if self._scale_known_fit else -1.0),
                L=self._L,
            )
        self._sl_post = post
        # the compressed gam(G=G) refit on the gam.fit3 port: its
        # gam.fit3.post.proc (mgcv.r:1735)
        fit3 = getattr(self, "_fit3_final", None)
        post3 = self._fit3_post_proc(fit3, method) if fit3 is not None else None
        if post3 is not None:
            edf = post3["edf"]
            edf_total = r_sum(edf)
        elif post is not None:
            edf = post["edf"]
            edf_total = r_sum(edf)
        elif mg is not None:
            edf = np.asarray(mg["post"]["edf"], dtype=float)
            edf_total = r_sum(edf)
        else:
            edf = np.diag(A_inv_XtWX).copy()
            edf_total = float(edf.sum())

        wt = self._wt
        df_resid = float(n - edf_total)
        if df_resid > 0:
            pearson_scale = float(fit.dev) / df_resid
        else:
            pearson_scale = float("nan")
        self._pearson_scale = pearson_scale
        # A user scale=φ fixes the Gaussian scale; otherwise the REML/Pearson
        # estimate (mgcv G$sig2 <- scale when known, mgcv.r:1942). GCV.Cp
        # reports magic's final-fit scale (bam.r:1294 ``object$sig2 <-
        # object$scale <- fit$scale``).
        if self._scale_known_fit:
            sigma_squared = self._scale_fixed_value
        elif post3 is not None:
            sigma_squared = float(fit3["scale.est"])
        elif mg is not None:
            sigma_squared = float(mg["fit"]["scale"])
        elif post is not None and self._log_phi_hat is not None:
            sigma_squared = float(np.exp(self._log_phi_hat))
        else:
            sigma_squared = pearson_scale
        sigma = (
            float(np.sqrt(sigma_squared))
            if np.isfinite(sigma_squared) and sigma_squared >= 0
            else float("nan")
        )

        if post3 is not None:
            Vp = post3["Vp"]
            Ve = post3["Ve"]
        elif post is not None:
            Vp = post["Vp"]
            Ve = post["Ve"]
        elif mg is not None:
            Vp = np.asarray(mg["post"]["Vb"], dtype=float)
            Ve = np.asarray(mg["post"]["Ve"], dtype=float)
        else:
            Vp = sigma_squared * A_inv
            Ve = sigma_squared * A_inv_XtWX @ A_inv

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

        eta_only = self._chunked_xbeta(beta)  # X·β (offset-stripped)
        eta = eta_only + self._offset  # full η
        mu = eta  # identity link
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

        # leverage and the standardised residuals: the lazy properties
        self._lev_w = self._wt
        self.df_residuals = df_resid
        # mgcv bam.r:2774 — ``object$deviance = sum(object$residuals^2)``. With
        # the weighted deviance residuals √w·(y−μ) this is Σ wᵢ(yᵢ−μᵢ)² (=
        # fit.dev), the weighted RSS. For AR1 (rho != 0) the AR1-decorrelated
        # RSS lives separately in ``std.rsd`` (used for σ² and AIC scale
        # calcs). The ``deviance`` is what ``deviance.explained`` reports
        # against ``null.deviance``, both on the original y scale.
        self.deviance = r_sum(self.residuals * self.residuals)
        self.rss = self.deviance
        # AR1-decorrelated residuals (mgcv ``object$std.rsd``, bam.r:2772) —
        # used by ``acf(rsd)`` checks. For rho=0, equals self.residuals.
        if self._rho != 0.0:
            self.std_rsd = _ar_resid(self.residuals, self._rho, self._ar_start)
        else:
            self.std_rsd = self.residuals.copy()

        # bam.r:2790: no bam.fit branch keeps a null deviance, so bam takes
        # it about weighted.mean(y, w), with or without an intercept
        mu_null = np.full(n, r_sum((y * wt)[wt != 0]) / r_sum(wt))
        self.null_deviance = r_sum(self.family.dev_resids(y, mu_null, wt))
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

        ss_resid_response = float(np.sum(wt * (y - mu) ** 2))
        if self._has_intercept and self._tss > 0:
            r_squared = 1.0 - ss_resid_response / self._tss
        elif self._yty_full > 0:
            r_squared = 1.0 - ss_resid_response / self._yty_full
        else:
            r_squared = float("nan")
        if df_resid > 0 and n > 1:
            sqrt_wt = np.sqrt(wt)
            mean_y_w = r_sum(wt * y) / r_sum(wt)
            v_resid = r_var(sqrt_wt * (y - mu))
            v_total = r_var(sqrt_wt * (y - mean_y_w))
            if v_total > 0:
                r_squared_adjusted = 1.0 - v_resid * (n - 1) / (v_total * df_resid)
            else:
                r_squared_adjusted = float("nan")
        else:
            r_squared_adjusted = float("nan")
        # mgcv summary.gam: r.sq NULL for no.r.sq families (mgcv.r:4055).
        if getattr(self.family, "no_r_sq", False):
            r_squared = r_squared_adjusted = float("nan")
        self.r_squared = float(r_squared)
        self.r_squared_adjusted = float(r_squared_adjusted)
        if self.null_deviance > 0:
            self.deviance_explained = float(
                (self.null_deviance - self.deviance) / self.null_deviance
            )
        else:
            self.deviance_explained = float("nan")

        self._fisher_w = None

        if post3 is not None:
            self.rank = int(fit3["rank"])
        else:
            self.rank = (
                int(mg["fit"]["gcv_info"]["rank"])
                if mg is not None
                else self._estimate_rank()
            )

        if post3 is not None:
            hess3 = (self._outer_info or {}).get("hess")
            H_aug = (
                np.asarray(hess3, dtype=float)
                if method in ("REML", "ML", "P-REML", "P-ML") and hess3 is not None
                else None
            )
        elif (
            method in ("REML", "ML", "P-REML", "P-ML")
            and n_sp > 0
            and np.isfinite(sigma_squared)
            and sigma_squared > 0
        ):
            if self._pearson_scale_criterion:
                # P-REML / P-ML: the Vc sp-uncertainty correction uses the
                # criterion's OWN ρ-only Hessian at the Pearson plug-in
                # scale (gam.fit3.r:979) — no free log φ row. Mirrors
                # gam's post-fit gate.
                H_aug = self._preml_hessian(rho_hat, fit=fit)
                T_aug = self._T_working(0)
            elif self._method_in in ("REML", "ML") and np.size(
                (self._outer_info or {}).get("hess", [])
            ):
                # gam(G=G)'s outer Newton Hessian at its optimum, in working
                # coordinates: what gam.fit3.post.proc reads (mgcv.r:1735)
                H_aug = np.asarray(self._outer_info["hess"], dtype=float)
                T_aug = None
            else:
                log_phi_hat_for_aug = (
                    self._log_phi_hat
                    if self._log_phi_hat is not None
                    else float(np.log(sigma_squared))
                )
                # a known scale is no outer parameter: the Hessian is over
                # the log smoothing parameters alone, as gam's post-fit has it
                include_phi_aug = not self._scale_known_fit
                H_aug = 0.5 * self._reml_hessian(
                    rho_hat,
                    log_phi_hat_for_aug,
                    fit=fit,
                    include_log_phi=include_phi_aug,
                )
                # with id linkage the criterion is optimised over θ (ρ = Lθ),
                # so the Hessian behind vcomp, edf2 and Vc is the working
                # one, T'HT with T = blockdiag(L, I); T is None without
                # linkage
                T_aug = self._T_working(1 if include_phi_aug else 0)
            if T_aug is not None:
                H_aug = T_aug.T @ H_aug @ T_aug
            H_aug = 0.5 * (H_aug + H_aug.T)
        else:
            H_aug = None
        self._H_aug = H_aug

        if post3 is not None:
            self.edf1 = post3["edf1"]
            self.edf2 = post3["edf2"] if post3["edf2"] is not None else edf.copy()
            self.edf1_total = r_sum(self.edf1)
            self.edf2_total = r_sum(self.edf2)
            Vc_corr = np.zeros_like(Vp)
        elif mg is not None:
            # magic.post.proc's edf1; edf2 and Vc are NULL (bam.r:1740-1755),
            # so edf2 := edf for logLik.gam/AIC.
            self.edf1 = np.asarray(mg["post"]["edf1"], dtype=float)
            self.edf2 = edf.copy()
            self.edf1_total = r_sum(self.edf1)
            self.edf2_total = edf_total
            Vc_corr = np.zeros_like(Vp)
        elif n_sp > 0:
            edf2_per_coef, edf1_per_coef, Vc_corr = self._compute_edf12(
                rho_hat,
                fit,
                sigma_squared,
                A_inv,
                A_inv_XtWX,
                edf,
                H_aug,
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
        if post is not None:
            # bam.r:2789 caps edf2 by its sum.
            edf2 = post["edf2"]
            if r_sum(edf2) > r_sum(post["edf1"]):
                edf2 = post["edf1"].copy()
            self.edf1 = post["edf1"]
            self.edf2 = edf2
            self.edf1_total = r_sum(post["edf1"])
            self.edf2_total = r_sum(edf2)
            self.Vc = post["Vc"]
        elif post3 is not None and post3["Vc"] is not None:
            self.Vc = post3["Vc"]
        else:
            self.Vc = Vp + Vc_corr

        sc_p = 0.0 if self._scale_known_fit else 1.0
        # bam.r:2778-2783: the AR1 model's AIC takes its scale from the
        # decorrelated residuals and adds the transform's log-determinant
        if self._rho != 0.0:
            dev1 = r_sum(self.std_rsd * self.std_rsd)
        else:
            dev1 = self.family._aic_dev1(self.deviance, sigma_squared, wt)
        # cbind responses: family$aic gets the trials vector as ``n``
        # (same as gam, gam.fit3.r:850).
        n_aic = self._binom_n if self._binom_n is not None else n
        family_aic = float(self.family.aic(y, mu, dev1, wt, n_aic))
        if self._rho != 0.0:
            n_ar = int(self._ar_start.sum()) if self._ar_start is not None else 0
            family_aic = family_aic - 2 * (n - n_ar) * math.log(
                1 / math.sqrt(1 - self._rho * self._rho)
            )
        mgcv_aic = family_aic + 2.0 * edf_total
        logLik = sc_p + edf_total - 0.5 * mgcv_aic
        # mgcv's magic (GCV.Cp/GACV.Cp) rail leaves object$edf2 NULL
        # (bam.r:1278-1295), so logLik.gam's df falls back to edf there
        # (mgcv.r:4431); hea's GCV edf2 attribute stays a best-effort
        # extra but must not leak into AIC/BIC (same rule as gam).
        df_base = (
            self.edf2_total if method in ("REML", "ML", "P-REML", "P-ML") else edf_total
        )
        df_for_aic = min(df_base + sc_p, float(p) + sc_p)
        # logLik.gam (mgcv.r:4434): extended families add n.theta to the
        # df AFTER the np cap (clog/cnorm: +1, bcg: +2, gfam: composite).
        if self._family_mgcv_extended:
            df_for_aic += float(getattr(self.family, "n_theta", 0) or 0)
        self.loglike = float(logLik)
        self.logLik = self.loglike
        self.npar = float(df_for_aic)
        self.AIC = -2.0 * logLik + 2.0 * df_for_aic
        self.BIC = -2.0 * logLik + float(np.log(n)) * df_for_aic
        self._mgcv_aic = float(mgcv_aic)

        if method in ("REML", "ML"):
            if self._log_phi_hat is not None:
                log_phi_hat = self._log_phi_hat
            elif self._scale_known_fit:
                log_phi_hat = math.log(self._scale_fixed_value)
            else:
                log_phi_hat = 0.0
            # ``self._reml`` with the real family's ls; the ML flavor
            # (remlInd=0, range-projected log|H|) comes from the
            # ``self.method`` predicates. mgcv's final bam report swaps
            # the log-lik to the family $aic flavor (bam.r:2784-2787)
            # whenever the compressed refit set ml.pen — gam(G=G) does
            # for BOTH explicit REML and ML (gam.fit3.r:620); bam.fit's
            # own fREML rail sets none (bam.r:1696), so the fREML alias
            # keeps the raw fit$reml flavor (= mgcv fREML bit-for-bit).
            if post3 is not None:
                # bam.r:2784-2786: gam(G=G)'s working log-likelihood replaced
                # by the family's, aic/(2 gamma) + ml.pen
                score = 2.0 * (family_aic / (2 * self._gamma) + float(fit3["ml_pen"]))
            elif post is not None:
                score = 2.0 * float(fr["fit"]["reml"])  # gcv.ubre = fit$reml
                if self._rho != 0.0 and np.isfinite(score):
                    # bam.fit's fREML score for the AR1 transform
                    # (bam.r:1710-1713), which no ml.pen replaces
                    ld = 1 / math.sqrt(1 - self._rho * self._rho)
                    df_ar = (
                        int(self._ar_start.sum()) if self._ar_start is not None else 1
                    )
                    score = 2.0 * (score / 2.0 - (n - df_ar) * math.log(ld))
            else:
                score = float(self._reml(rho_hat, log_phi_hat, fit=fit))
            if (
                post3 is None
                and (method == "ML" or self._method_in == "REML")
                and np.isfinite(score)
            ):
                log_phi_swap = (
                    float(np.log(self._scale_fixed_value))
                    if self._scale_known_fit
                    else log_phi_hat
                )
                # the criterion's own deviance (the AR1-transformed one when
                # rho != 0); the AR1 log-determinant enters through the AIC
                score = score + self._ml_flavor_swap(
                    family_aic, log_phi_swap, float(fit.dev)
                )
            if method == "REML":
                self.REML_criterion = score
            else:
                self.ML_criterion = score
        elif method in ("P-REML", "P-ML"):
            if post3 is not None:
                score = 2.0 * float(self._outer_info["score"])
            elif n_sp > 0:
                score = 2.0 * float(
                    self._score_all(rho_hat, 0.0, fit, 0, "PREML")["score"]
                )
            else:
                score = float("nan")
            # AR1 correction — mgcv applies it to the compressed-refit
            # gcv.ubre too (bam.r:1734-1738); doubled for the 2V units.
            if self._rho != 0.0 and np.isfinite(score):
                ld = 1.0 / np.sqrt(1.0 - self._rho**2)
                df_ar = int(self._ar_start.sum()) if self._ar_start is not None else 1
                score = score - 2.0 * (n - df_ar) * float(np.log(ld))
            if method == "P-REML":
                self.REML_criterion = score
            else:
                self.ML_criterion = score
        elif method == "GACV.Cp":
            if post3 is not None:
                score = float(self._outer_info["score"])
                if self._rho != 0.0 and np.isfinite(score):
                    ld = 1.0 / np.sqrt(1.0 - self._rho**2)
                    df_ar = (
                        int(self._ar_start.sum()) if self._ar_start is not None else 1
                    )
                    score = score - (n - df_ar) * float(np.log(ld))
                self.GCV_score = score
            elif n_sp > 0:
                score = float(
                    self._score_all(
                        rho_hat,
                        0.0,
                        fit,
                        0,
                        self._bam_outer_criterion(),
                    )["score"]
                )
                # AR1 correction — the gam(G=G) branch corrects gcv.ubre
                # (bam.r:1734-1738; V units, single). mgcv's GCV.Cp rides
                # magic, which is NOT corrected (bam.r:1740-1760).
                if self._rho != 0.0 and np.isfinite(score):
                    ld = 1.0 / np.sqrt(1.0 - self._rho**2)
                    df_ar = (
                        int(self._ar_start.sum()) if self._ar_start is not None else 1
                    )
                    score = score - (n - df_ar) * float(np.log(ld))
                self.GCV_score = score
            else:
                self.GCV_score = float("nan")
        elif mg is not None:
            # magic's score (bam.r:1746 ``object$gcv.ubre <- fit$score``).
            self.GCV_score = float(mg["fit"]["score"])
        else:
            self.GCV_score = float(self._gcv(rho_hat)) if n_sp > 0 else float("nan")

        self.vcomp = self._compute_vcomp()

        # mgcv exposes ``object$AR1.rho`` (bam.r:885) for downstream
        # consumers (predict.bam, residuals.bam). Mirror the attribute.
        self.AR1_rho = self._rho

    def _chunked_leverage_diag_weighted(
        self, A_inv: np.ndarray, w_full: np.ndarray
    ) -> np.ndarray:
        """Weighted hat-matrix diagonal ``hᵢ = wᵢ·(X·A⁻¹·X')ᵢᵢ``."""
        if self._discrete_design is not None:
            return w_full * diagXVXd(self._discrete_design, A_inv)
        n = self.n
        out = np.empty(n, dtype=float)
        for start, end in _chunk_indices(n, self._chunk_size):
            X_chunk = self._x_rows(start, end)
            HX = X_chunk @ A_inv
            out[start:end] = w_full[start:end] * (HX * X_chunk).sum(axis=1)
        return out

    # -----------------------------------------------------------------------
    # On-demand leverage / standardised residuals (mgcv's bam stores no
    # n-length hat — bam.r:806-894): every post-fit leaves them to these
    # cached properties, so the O(n·p²) leverage pass over the data runs only
    # when a diagnostic (``hatvalues``/``rstandard``/Cook's D) asks for it.
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
        self,
        theta0: np.ndarray,
        *,
        include_log_phi: bool,
        max_iter: int = 200,
    ) -> np.ndarray:
        """Shim → free `_fast_reml_fit` (mgcv fast.REML.fit). Builds/caches the
        Sl setup, then persists the F5/F9 β̂/A⁻¹ reuse slots + outer_info.
        Feeds the reduced ``(R, f)`` from the current chunked QR build plus
        ``rss.extra = ‖z̃‖² − ‖f‖²`` (bam.r:1146/1675)."""
        if not hasattr(self, "_sl"):
            self._sl = _sl_setup(self._slots, self.p)
        qrx = self._bam_qr
        rss_extra = float(qrx.y_norm2 - r_sum(qrx.f * qrx.f))
        res = _fast_reml_fit(
            theta0,
            include_log_phi=include_log_phi,
            max_iter=max_iter,
            work_dim=self._work_dim,
            L=self._L,
            lsp0=self._lsp0,
            p=self.p,
            sl=self._sl,
            X=qrx.R,
            f=qrx.f,
            rss_extra=rss_extra,
            gamma=self._gamma,
            n=self.n,
            scale_fixed_value=self._scale_fixed_value,
        )
        self._reml_beta = res["reml_beta"]
        self._reml_A_inv = res["reml_A_inv"]
        self._outer_info = res["outer_info"]
        self._freml = (
            None
            if res["fit"] is None
            else {
                "fit": res["fit"],
                "undrop": res["undrop"],
                "hess": res["outer_info"]["hess"],
                "um_sl": res["um_sl"],
            }
        )
        # The non-discrete fREML rail reports c("perf","newton") (bam.r:1249/
        # 1700). A discrete fit overwrites this with ("perf","chol") after its
        # PIRLS loop, matching bgam.fitd (bam.r:784).
        if self._method_in == "fREML":
            self._rail_optimizer = ("perf", "newton")
        return res["theta"]

    def _bgam_rsb_penalty(self, rho_full: np.ndarray, coef: np.ndarray) -> float:
        """mgcv ``sum(rSb²)`` (bgam.fitd:591/611) — the penalty ``βᵀSλβ`` as the
        per-block penalty-root reduction the PIRLS step-halving + convergence
        test read, NOT a full-matrix quadratic form ``coef·Sλ·coef``.
        """
        if not hasattr(self, "_sl"):
            self._sl = _sl_setup(self._slots, self.p)
        b = _sl_initial_repara(
            self._sl,
            np.asarray(coef, dtype=float),
            inverse=False,
            both_sides=False,
            cov=False,
        )
        a = _sl_rsb(self._sl, rho_full, b)
        return float(np.dot(a, a))

    def _bgam_fit_loop(
        self, *, sp_user, coef_start=None, eps: float = 1e-7
    ) -> tuple[_FitState, np.ndarray]:
        """Outer PIRLS loop with a chunked QR rebuild per iteration.

        ``coef_start`` is mgcv's ``coef=`` warm start (bgam.fitd:548-560 /
        bgam.fit:1066): iteration 0 builds the working model at
        η = X·coef + offset instead of the family-initialize η, with dev seeded
        at 2·dev(coef) and the step-halving reference snapshots armed from the
        start. ``eps`` is control$epsilon; the samfrac pre-fit loosens it to
        1e-2 (bam.r:2685).

        Direct port of mgcv ``bgam.fit`` (bam.r:909-1353). Each iteration:

        1. Build (R, f, ‖z̃‖², rss_extra) over chunks of √W·X / √W·z, where
           (W, z) are the Fisher PIRLS weights/working response from the
           chunk's η = X·β + offset (the family-initialised η on iteration 1).
        2. Update the reduced sufficient statistics ``self._XtX = R'R``,
           ``self._Xty = R'f``, ``self._yty = ‖z̃‖²``, ``self._X_full = R``.
        3. Update (ρ, log φ) on the reduced (R, f), then recover β̂ at the new
           ρ̂ via ``_fit_given_rho``. The smoothing-parameter update follows
           mgcv's split on ``discrete``:

           * discrete=TRUE  → one ``_pi_fit_chol`` (Sl.fitChol) step,
             gradient-halving, warm-started (mgcv ``bgam.fitd``);
           * discrete=FALSE → ``_fast_reml_fit`` (mgcv ``fast.REML.fit``) run to
             convergence, reml-value halving, re-seeded from ``initial.sp`` each
             iteration (mgcv ``bgam.fit``);
           * GCV.Cp → ``_outer_newton`` on V_g/V_u.
        4. Step-halving (mgcv's "kk" inner loop): if ``it > 1`` and the new
           penalised deviance increases, halve β toward ``β₀`` and rebuild.

        Convergence (mgcv:1154): ``|dev - devold| / (0.1+|dev|) < ε`` after
        ``it > 1`` (mgcv's ``iter > 2``: the ``dev*2`` seed in step 0 makes the
        first iteration's check meaningless). At the converged β̂ the reduced
        (R, f) problem is Gaussian, so the inner Newton sees ``W = I``; the
        non-Gaussian weights live only in the outer loop's ``W``.
        """
        family = self.family
        link = family.link
        n = self.n
        method = self.method  # already mapped fREML → REML

        blocks = self._blocks
        chunk_size = self._chunk_size
        y = self._y_arr
        off = self._offset
        prior_w = self._wt  # mgcv bam(weights=) → G$w (ones if unset)

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
        # before the first PIRLS iter. tw is mgcv-extended (its
        # preinitialize is a no-op, like mgcv's NULL slot).
        if self._family_mgcv_extended:
            pini = family.preinitialize(y)
            if pini is not None and "Theta" in pini:
                family.set_theta(pini["Theta"])

        # ---- Initialize μ̂, η̂, dev for iter 0 (mgcv bam.r:950-969) -----
        # mgcv order (bam.r:961-966): η = linkfun(mustart), then μ =
        # linkinv(η), then validmu(μ) — the validity check sees the
        # ROUND-TRIPPED μ, so e.g. a log link's eps floor rescues the
        # zero-count mustart rows of the censored count families
        # (mustart 0 → η = −Inf → μ = eps > 0, valid in mgcv and here).
        mu = family.gam_initialize(y, prior_w)
        eta = link.link(mu)
        mu = link.linkinv(eta)
        if not (link.valideta(eta) and family.validmu(mu)):
            raise FloatingPointError(
                "PIRLS init: cannot find valid starting μ̂ from family.initialize"
            )
        coef: np.ndarray | None = None
        coef0: np.ndarray | None = None
        eta0: np.ndarray | None = None
        dev0: float | None = None
        # NON-DISCRETE only (bgam.fit): the θ that built the PREVIOUS working
        # model (mgcv ``theta0``, snapshotted at iter-end before estimate.theta,
        # bam.r:1198) — the step-halving evaluates dev0/dev1 at this θ0.
        theta0_snap: np.ndarray | None = None
        pcoef0_snap: np.ndarray | None = None
        dev0_snap = 0.0
        dev = 2.0 * float(r_sum(family.dev_resids(y, mu, prior_w)))
        # bgam.fitd's previous proposal beta (``b0``) and the first iteration
        # with step control (``c.iter``, bam.r:548-560): a warm start sets b0
        # from the start coefficients and checks from the second iteration.
        b0_prop: np.ndarray | None = None
        c_iter = 2

        if coef_start is not None:
            coef = np.asarray(coef_start, dtype=float).reshape(-1)
            if coef.shape != (self.p,):
                raise ValueError(f"coef must have length {self.p}, got {coef.shape}")
            if self._discrete_design is not None:
                eta = Xbd(self._discrete_design, coef) + off
                mu = link.linkinv(eta)
                dev0 = float(r_sum(family.dev_resids(y, mu, prior_w)))
                dev = 2.0 * dev0
                coef0 = coef.copy()
                eta0 = eta.copy()
                if not hasattr(self, "_sl"):
                    self._sl = _sl_setup(self._slots, self.p)
                b0_prop = _sl_initial_repara(
                    self._sl, coef0, inverse=False, both_sides=False, cov=False
                )
                c_iter = 1

        maxit = 200  # mgcv default control$maxit
        conv = False

        rho_hat: np.ndarray | None = None  # full per-penalty log-sp
        log_phi_hat: float | None = None
        fit: _FitState | None = None
        fit_build: tuple | None = None
        last_out: dict | None = None
        theta_sp_warm: np.ndarray | None = None
        Nstep: np.ndarray | None = None
        n_work = self._work_dim
        theta_sp_warm: np.ndarray | None = None
        crit_fitd: float | None = None

        def _bgamfit_halve(
            coef,
            eta,
            mu,
            dev,
            coef0,
            eta0,
            dev0_snap,
            rho_hat,
            theta0_snap,
            pcoef0_snap,
        ):
            """bgam.fit's divergence check (bam.r:1163-1190): while the
            penalized deviance rose, halve the step toward the previous
            iterate, at most 6 times; returns ``(coef, eta, mu, dev, kk)``.
            An extended family's deviances are at the previous working
            model's θ (``theta0``); otherwise ``dev0`` is the previous
            build's deviance and ``dev1`` this one's. For fREML the
            penalty is ``sum(fit$beta * Sl.Sb(um$Sl, log(full.sp),
            fit$beta))`` with the last two fast.REML.fit coefficient vectors,
            in that fit's own parameterization."""
            kk = 0
            if not hasattr(self, "_sl"):
                self._sl = _sl_setup(self._slots, self.p)
            efam = self._family_mgcv_extended
            theta_now = family.get_theta() if efam else None
            use_t0 = efam and theta0_snap is not None
            if efam:
                if use_t0:
                    family.set_theta(theta0_snap)
                dev0 = r_sum(family.dev_resids(y, link.linkinv(eta0), prior_w))
                dev1 = r_sum(family.dev_resids(y, link.linkinv(eta), prior_w))
                if use_t0:
                    family.set_theta(theta_now)
            else:
                dev0 = dev0_snap
                dev1 = dev
            fr = getattr(self, "_freml", None)
            if pcoef0_snap is not None and fr is not None:
                pcoef0 = pcoef0_snap
                pcoef = np.asarray(fr["fit"]["beta"], dtype=float)
                Sb0 = _sl_sb(fr["um_sl"], rho_hat, pcoef0)
                Sb = _sl_sb(fr["um_sl"], rho_hat, pcoef)
            else:
                pcoef0 = _sl_initial_repara(
                    self._sl, coef0, inverse=False, both_sides=False, cov=False
                )
                pcoef = _sl_initial_repara(
                    self._sl, coef, inverse=False, both_sides=False, cov=False
                )
                Sb0 = _sl_sb(self._sl, rho_hat, pcoef0)
                Sb = _sl_sb(self._sl, rho_hat, pcoef)
            while (dev0 + r_sum(pcoef0 * Sb0) < dev1 + r_sum(pcoef * Sb)) and kk < 6:
                coef = (coef0 + coef) / 2
                pcoef = (pcoef0 + pcoef) / 2
                eta = (eta0 + eta) / 2
                Sb = (Sb0 + Sb) / 2
                mu = link.linkinv(eta)
                dev = r_sum(family.dev_resids(y, mu, prior_w))
                if efam:
                    if use_t0:
                        family.set_theta(theta0_snap)
                    dev1 = r_sum(family.dev_resids(y, mu, prior_w))
                    if use_t0:
                        family.set_theta(theta_now)
                else:
                    dev1 = dev
                kk += 1
            return coef, eta, mu, dev, kk

        for it in range(maxit):
            devold = dev

            if it >= 1 and coef is not None and not additive:
                if self._discrete_design is not None:
                    eta = (
                        Xbd(self._discrete_design, np.asarray(coef, dtype=float)) + off
                    )
                else:
                    eta = self._chunked_xbeta(np.asarray(coef, dtype=float)) + off
                mu = link.linkinv(eta)

            kk = 0
            discrete = self._discrete_design is not None
            # bgam.fitd's penalty for the step control and the penalized
            # deviance (bam.r:571-611): rSb from the last proposal's
            # reparameterized beta, halved with coef, eta and prop$beta.
            rsb = None
            if (
                discrete
                and it >= 1
                and not additive
                and rho_hat is not None
                and coef is not None
            ):
                if not hasattr(self, "_sl"):
                    self._sl = _sl_setup(self._slots, self.p)
                if last_out is not None and "prop_beta" in last_out:
                    prop_beta = np.array(last_out["prop_beta"], dtype=float)
                else:
                    # the fixed-sp rail solves without Sl.fitChol
                    prop_beta = _sl_initial_repara(
                        self._sl, coef, inverse=False, both_sides=False, cov=False
                    )
                rsb = _sl_rsb(self._sl, rho_hat, prop_beta)
            if (
                it + 1 > c_iter
                and coef is not None
                and coef0 is not None
                and rho_hat is not None
                and eta0 is not None
                and not additive
                and discrete
                and rsb is not None
                and b0_prop is not None
            ):
                rsb0 = _sl_rsb(self._sl, rho_hat, b0_prop)
                bSb0 = r_sum(rsb0 * rsb0)
                mu0 = link.linkinv(eta0)
                dev0 = float(r_sum(family.dev_resids(y, mu0, prior_w)))
                dev_cur = float(r_sum(family.dev_resids(y, mu, prior_w)))
                kk_d = 1
                while (
                    not np.isfinite(dev_cur) or dev0 + bSb0 < dev_cur + r_sum(rsb * rsb)
                ) and kk_d < 30:
                    coef = coef0 * 0.5 + coef / 2.0
                    rsb = rsb0 * 0.5 + rsb / 2.0
                    eta = eta0 * 0.5 + eta / 2.0
                    prop_beta = b0_prop * 0.5 + prop_beta / 2.0
                    mu = link.linkinv(eta)
                    dev_cur = float(r_sum(family.dev_resids(y, mu, prior_w)))
                    kk_d += 1
                kk = kk_d - 1
            if rsb is not None:
                b0_prop = prop_beta

            # ---- DISCRETE extended-family θ update at the (halved) μ --------
            # mgcv bgam.fitd:614-630 — estimate θ MID-iter (before the build), so
            # this iter's build sees this-iter θ. The NON-discrete path estimates
            # θ at the END of the iter instead (bam.r:1204; see below). Gate =
            # mgcv's `family$n.theta>0 || (scale<0 && iter<5)` (bgam.fitd:617;
            # mgcv iter is 1-based, hea it = iter−1): families whose θ is free
            # (Scat with both θ locked has ``n_theta = 0`` and stays put) — plus
            # tw's early joint (θ, log φ) passes when the scale is estimated.
            scale_joint = (
                self._scale_resolved is not None and self._scale_resolved < 0 and it < 4
            )
            if (
                it >= 1
                and discrete
                and self._family_mgcv_extended
                and (family.n_theta > 0 or scale_joint)
            ):
                if scale_joint:
                    scale1 = float(self._scale_resolved)
                elif include_log_phi and log_phi_hat is not None:
                    scale1 = float(np.exp(log_phi_hat))  # exp(log.phi)
                else:
                    scale1 = self._scale_fixed_value  # log.phi=log(scale)
                theta_new = _estimate_theta(
                    family,
                    y,
                    mu,
                    scale=scale1,
                    wt=prior_w,
                    tol=1e-7,
                )
                if scale_joint:
                    log_phi_hat = float(theta_new[-1])
                    theta_new = theta_new[:-1]
                    if include_log_phi and Nstep is not None and Nstep.size:
                        Nstep[-1] = 0.0
                family.set_theta(theta_new)

            if it == 0 or not additive:
                if self._discrete_design is not None:
                    qr = _build_qr_discrete_pirls(
                        self._discrete_design,
                        y,
                        off,
                        family,
                        coef=coef,
                        # Pass the η we already formed above (iter 0: family
                        # init; iter>0: Xbd(coef)+off, possibly step-halved) so
                        # the build reuses it instead of a second Xbd pass (mgcv
                        # forms η once, bam.r:572). Bit-identical; saves ~1
                        # Xbd/iter.
                        eta_init=eta,
                        use_chol=self._use_chol,
                        prior_w=prior_w,
                        rho=self._rho,
                        ar_start=self._ar_start,
                    )
                else:
                    qr = _build_qr_chunked_pirls(
                        self.data,
                        blocks,
                        self._X_param_full,
                        y,
                        off,
                        family,
                        coef=coef,
                        eta_init=eta if coef is None else None,
                        chunk_size=chunk_size,
                        use_chol=self._use_chol,
                        prior_w=prior_w,
                        x_rows=self._x_rows,
                    )
                self._bam_qr = qr
                # Reduced-data sufficient stats consumed by ``_outer_newton``
                # via the inherited ``_fit_given_rho`` machinery. ``_X_full =
                # R`` keeps the inner-score routines on the (R, f) reduced
                # design just like the Gaussian-identity path. The bam-class
                # ``_dw_deta`` / ``_d2w_deta2`` overrides return ``zeros(p)``,
                # which matches "Gaussian-on-(R, f)" exactly: at the PIRLS-
                # converged β̂ the inner score sees a constant-W problem.
                #
                # X'WX and X'Wz come from the accumulation itself when the
                # build formed them (the discrete rail): mgcv ``bgam.fitd``
                # assigns ``qrx$R <- XWXd(...)``/``qrx$f <- XWyd(...)`` and
                # reparameterises THOSE into ``qrx$XX``/``qrx$Xy`` for
                # ``Sl.fitChol`` (bam.r:657-665) without factorising them
                # into an R first; rebuilding them as ``R'R``/``R'f`` would
                # lose a pivoted Cholesky's worth of accuracy in the
                # ill-conditioned directions. The chunked rails accumulate
                # (R, f) directly, as mgcv's ``bgam.fit`` does.
                self._XtX = qr.R.T @ qr.R if qr.XX is None else qr.XX
                self._Xty = qr.R.T @ qr.f if qr.Xy is None else qr.Xy
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
                # ``sum(rSb²)`` (:meth:`_bgam_rsb_penalty`), as mgcv forms it,
                # not a full-matrix quadratic form.
                if rsb is not None:
                    dev = float(dev) + r_sum(rsb * rsb)
                if not np.isfinite(dev):
                    raise FloatingPointError(f"non-finite deviance at PIRLS iter {it}")
            else:
                # Additive, iter>1: cheap penalised-deviance refresh from the
                # cached (R, f) and current coef — mgcv bgam.fitd:669
                # ``dev <- qrx$y.norm2 - sum(coef*qrx$f)``. This equals ‖z‖² −
                # βᵀX'Wz, which at the sp-update solution is ‖z−Xβ‖²_W + βᵀSβ
                # (the penalised deviance the bam.r:678 convergence test reads).
                # self._XtX/_Xty/_yty/_X_full/_wt_full/_bam_qr persist from iter
                # 0; eta/mu stay stale (unused — halving and θ are gated off).
                dev = self._yty - r_sum(coef * self._Xty)
                if not np.isfinite(dev):
                    raise FloatingPointError(
                        f"non-finite penalised deviance at PIRLS iter {it}"
                    )

            phi_conv = True
            if (
                self._discrete_design is not None
                and include_log_phi
                and Nstep is not None
                and Nstep.size
            ):
                phi_step = float(Nstep[-1])
                log_phi_now = log_phi_hat if log_phi_hat is not None else 0.0
                phi_conv = abs(phi_step) < eps * (abs(log_phi_now) + 1.0)
            if it > 1 and abs(dev - devold) / (0.1 + abs(dev)) < eps and phi_conv:
                conv = True
                break
            # bgam.fit tests divergence only after the build and the
            # convergence test; a halved step is rebuilt and tested again
            # (bam.r:1149-1192).
            if (
                not discrete
                and it > 1
                and coef is not None
                and coef0 is not None
                and rho_hat is not None
                and eta0 is not None
                and not additive
            ):
                coef, eta, mu, dev, kk = _bgamfit_halve(
                    coef,
                    eta,
                    mu,
                    dev,
                    coef0,
                    eta0,
                    dev0_snap,
                    rho_hat,
                    theta0_snap,
                    pcoef0_snap,
                )
                if kk > 0:
                    qr = _build_qr_chunked_pirls(
                        self.data,
                        blocks,
                        self._X_param_full,
                        y,
                        off,
                        family,
                        coef=coef,
                        eta_init=None,
                        chunk_size=chunk_size,
                        use_chol=self._use_chol,
                        prior_w=prior_w,
                        x_rows=self._x_rows,
                    )
                    self._bam_qr = qr
                    self._XtX = qr.R.T @ qr.R
                    self._Xty = qr.R.T @ qr.f
                    self._yty = float(qr.y_norm2)
                    self._X_full = qr.R
                    self._wt_full = qr.wt
                    eta = qr.eta
                    mu = qr.mu
                    dev = qr.dev
                    if abs(dev - devold) / (0.1 + abs(dev)) < eps:
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
                dev0_snap = dev  # bam.r:1201 ``dev0 <- dev``
                if (not discrete) and self._family_mgcv_extended:
                    theta0_snap = np.asarray(family.get_theta(), dtype=float).copy()
                # bam.r:1199 ``pcoef0 <- fit$beta`` (fREML): the last
                # fast.REML.fit's β in its own parameterization
                fr_prev = getattr(self, "_freml", None)
                pcoef0_snap = (
                    np.array(fr_prev["fit"]["beta"], copy=True)
                    if fr_prev is not None and self._method_in == "fREML"
                    else None
                )

            # ---- NON-DISCRETE extended-family θ update at iter END ----------
            # mgcv bgam.fit:1204-1217 estimates θ at the END of the iter (after
            # the conv-check + snapshot), so the NEXT iter's working-model build
            # uses THIS θ — i.e. each build sees the PREVIOUS iter's θ. (The
            # discrete path estimated θ mid-iter, above.) θ is fit at the current
            # μ = linkinv(eta), the post-build/post-halving μ (== bam.r:1209's
            # ``linkinv(eta)``). Gate = mgcv's `family$n.theta>0 || (scale<0 &&
            # iter<5)` (bam.r:1206) — free-θ families plus tw's early joint
            # (θ, log φ) passes when the scale is estimated.
            scale_joint_nd = (
                (not discrete)
                and self._scale_resolved is not None
                and self._scale_resolved < 0
                and it < 4
            )
            if (
                it >= 1
                and (not discrete)
                and self._family_mgcv_extended
                and (family.n_theta > 0 or scale_joint_nd)
            ):
                if scale_joint_nd:
                    # bam.r:1208 — iters 2-4 pass scale<0 through:
                    # estimate.theta optimizes (θ, log φ) jointly.
                    scale1 = float(self._scale_resolved)
                elif include_log_phi and log_phi_hat is not None:
                    scale1 = float(np.exp(log_phi_hat))
                else:
                    scale1 = self._scale_fixed_value
                theta_new = _estimate_theta(
                    family,
                    y,
                    mu,
                    scale=scale1,
                    wt=prior_w,
                    tol=1e-7,
                )
                if scale_joint_nd:
                    # bam.r:1212-1215: object$scale <- exp(θ_φ); the next
                    # iter's fast.REML log φ seed reads it (bam.r:1233).
                    log_phi_hat = float(theta_new[-1])
                    theta_new = theta_new[:-1]
                family.set_theta(theta_new)

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
            # Both cadences reach the same optimum. Scale-unknown families
            # (Gamma, inverse-Gaussian, fixed-p Tweedie, extended φ) must
            # use these Gaussian-working-REML optimisers rather than
            # ``_outer_newton``, which minimises ``_reml`` - the full
            # non-Gaussian REML carrying the family's ``ls0(φ)``
            # saturated-likelihood term (what mgcv's gam uses), a
            # different objective on the reduced (R, f) with a different
            # φ̂ optimum. ``_outer_newton`` serves only ``method ==
            # "GCV.Cp"`` (no REML/φ formulas) - the final ``else``.
            if method in ("REML", "ML") and self._discrete_design is not None:
                if not hasattr(self, "_sl"):
                    self._sl = _sl_setup(self._slots, self.p)
                if theta_sp_warm is None:
                    # Full-space initial.spg seed → working space by least
                    # squares (mgcv mgcv.r:4617-4618); identity when no id.
                    # in.out replaces the λ seed (bgam.fitd:687).
                    io_rho0 = self._in_out_rho0()
                    rho0_full = (
                        io_rho0 if io_rho0 is not None else self._initial_sp_rho()
                    )
                    if self._L is None:
                        rho_cur = rho0_full
                    else:
                        rho_cur = _lm_on_L(rho0_full, self._L, self._lsp0)
                    Nstep = np.zeros(n_work + (1 if include_log_phi else 0))
                else:
                    rho_cur = theta_sp_warm.copy()
                if include_log_phi:
                    if log_phi_hat is None:
                        if getattr(self, "_in_out", None) is not None:
                            log_phi_cur = self._in_out_log_scale()
                        else:
                            log_phi_cur = float(np.log(r_var(y) * 0.05))
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
                halve_max = 30
                halves = 0
                if crit_fitd is None:
                    crit_fitd = float(dev)  # bam.r:609 ``crit <- dev``
                while True:
                    theta_try = theta_cur + Nstep
                    theta_sp_try = theta_try[:n_work]
                    if include_log_phi:
                        log_phi_try = float(theta_try[n_work])
                    else:
                        log_phi_try = float(np.log(self._scale_fixed_value))
                    # ``Sl.initial.repara`` (fast-REML.r:517-588,
                    # bam.r:664-665) — reparameterize XX, Xy into mgcv's
                    # well-scaled gauge (every penalty block, two-sided)
                    # so the pivoted Cholesky in ``_pi_fit_chol``
                    # factorizes the same conditioned matrix mgcv does. β
                    # comes back in the repara'd basis and gets un-rotated
                    # below. log|Sλ|_+ + its ρ-derivatives come from
                    # ``_pi_fit_chol``'s internal ``ldetS(repara=FALSE)``
                    # (Sl.fitChol:1598), in the same gauge.
                    XX_pre = _sl_initial_repara(self._sl, self._XtX, both_sides=True)
                    Xy_pre = _sl_initial_repara(self._sl, self._Xty, both_sides=True)
                    out = _pi_fit_chol(
                        XX_pre,
                        Xy_pre,
                        theta_sp_try,
                        self._sl,
                        self.p,
                        yy=self._yty,
                        log_phi=log_phi_try,
                        n=n,
                        Mp=self._Mp,
                        gamma=self._gamma,
                        phi_fixed=not include_log_phi,
                        L=self._L,
                        rho0=(0.0 if self._lsp0 is None else self._lsp0),
                        tol=abs(crit_fitd) * float(np.finfo(float).eps) ** 0.5,
                    )
                    # Undo the initial-repara on β (bam.r:759,
                    # inverse=TRUE) — the rest of the PIRLS / post-fit
                    # machinery (chunked X·β, variance, edf) operates in
                    # the original basis; ``prop$beta`` itself feeds the
                    # next iteration's penalty (bam.r:574).
                    out["prop_beta"] = np.array(out["beta"], dtype=float)
                    out["beta"] = _sl_initial_repara(
                        self._sl,
                        out["beta"],
                        inverse=True,
                        both_sides=False,
                        cov=False,
                    )
                    grad_w = np.atleast_1d(np.asarray(out["grad"], dtype=float))
                    step_w = np.broadcast_to(
                        np.asarray(out["step"], dtype=float), Nstep.shape
                    ).copy()
                    # mgcv's Nstep starts as the scalar 0 and Sl.fitChol
                    # returns step = 0 with nothing to optimise, so an
                    # empty step is a zero one.
                    if Nstep.size == 0 or float(np.max(Nstep)) == 0.0:
                        Nstep = step_w
                        theta_cur = theta_try
                        break
                    if r_sum(grad_w * Nstep) > dev * 1e-7 and halves < halve_max:
                        Nstep = Nstep / 2.0
                        halves += 1
                        continue
                    Nstep = step_w
                    theta_cur = theta_try
                    break

                theta_sp_warm = theta_cur[:n_work]
                log_phi_hat = float(theta_cur[n_work]) if include_log_phi else None
                log_phi_c = (
                    log_phi_hat
                    if include_log_phi
                    else float(np.log(self._scale_fixed_value))
                )
                # bam.r:765 - the criterion whose size sets the next
                # Sl.fitChol tolerance
                crit_fitd = (
                    dev / (np.exp(log_phi_c) * self._gamma)
                    - out["ldetS"]
                    + out["ldetXXS"]
                ) / 2.0
                self.sp = np.exp(theta_sp_warm)  # mgcv m$sp
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
                        self._sl,
                        out["PP"],
                        inverse=True,
                        both_sides=True,
                        cov=True,
                    )
            elif method == "REML" and self._method_in == "fREML":
                # Non-discrete fREML → mgcv ``bgam.fit`` (bam.r:1226-1261):
                # ``fast.REML.fit`` to FULL convergence on the current
                # reduced (R, f). fast.REML.fit implements ONLY the REML
                # criterion (mgcv's explicit method="REML" runs the
                # gam(G=G) route, bam.r:1273 — same objective on the
                # Gaussian working data, so hea unifies them here; ML
                # canNOT ride this rail and takes the else below).
                # The sp is RE-SEEDED from ``initial.sp(R)``
                # every PIRLS iter (bam.r:1229 — NOT warm-started; the
                # converge-fully Newton makes the seed immaterial to the
                # result), while log φ is carried forward from the previous
                # iter's scale estimate (bam.r:1233). ``_initial_sp_rho``
                # gives the full-space seed; ``coef(lm(lsp ~ L-1+offset))``
                # (the lstsq) projects it to working space, identity when
                # no smooths share an id (fast-REML.r:1768). in.out's λ
                # replaces the seed — every iter, like mgcv (bam.r:1229).
                io_rho0 = self._in_out_rho0()
                rho0_full = io_rho0 if io_rho0 is not None else self._initial_sp_rho()
                if self._L is None:
                    rho0 = rho0_full
                else:
                    rho0 = _lm_on_L(rho0_full, self._L, self._lsp0)
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
                        # in.out$scale replaces it (bam.r:1237).
                        if getattr(self, "_in_out", None) is not None:
                            log_phi0 = self._in_out_log_scale()
                        else:
                            log_phi0 = float(np.log(r_var(y) * 0.05))
                    else:  # iter > 1 — carry forward
                        # bam.r:1233 ``log.phi <- log(object$scale)``, the
                        # scale stored as ``exp(.)`` of the last log scale
                        log_phi0 = float(np.log(np.exp(log_phi_hat)))
                    theta0 = np.concatenate([rho0, [log_phi0]])
                else:
                    theta0 = rho0

                theta_hat = self._fast_reml_fit(
                    theta0,
                    include_log_phi=include_log_phi,
                )
                theta_sp_warm = theta_hat[:n_work]  # log φ carry-forward
                log_phi_hat = float(theta_hat[n_work]) if include_log_phi else None
                self.sp = np.exp(theta_hat[:n_work])
                rho_hat = self._rho_full(theta_hat[:n_work])
                fit = self._fit_given_rho(rho_hat)
                if self._reml_beta is not None:
                    fit.beta = self._reml_beta
                    fit.A_inv = self._reml_A_inv
            elif method == "GCV.Cp":
                # magic on the current working (qrx$R, qrx$f) — bam.r:1220
                # — per PIRLS iteration, reseeded from initial.sp(qrx$R)
                # each time (G$sp stays -1); its coefficients are the fit.
                mfit = self._bam_magic()
                theta_sp_warm = np.log(mfit["sp"])
                log_phi_hat = None
                self.sp = np.asarray(mfit["sp"], dtype=float)
                rho_hat = self._rho_full(theta_sp_warm)
                fit = self._fit_given_rho(rho_hat)
                fit.beta = np.asarray(mfit["b"], dtype=float)
                fit.A_inv = r_tcrossprod(mfit["rV"])
            else:
                # ML / GACV.Cp / P-REML / P-ML: the converge-fully
                # outer-Newton on the mapped criterion — mgcv's
                # compressed refit for these methods is
                # gam(G=G, method=method) (bam.r:1273) — per PIRLS
                # iteration. ML keeps
                # log φ in θ when the scale is estimated (its remlInd=0
                # + range-projected log|H| come from the ``self.method``
                # predicates through the REML scoreType); the others are
                # ρ-only. L-aware — it optimises in working space and
                # maps via ``_rho_full``. Re-seeded from initial.sp on
                # the CURRENT working (R, f) every PIRLS iteration —
                # gam(G=G) derives its own seed per call (bam.r:1273);
                # nothing is warm-started across iterations.
                rho0_full = self._initial_sp_rho()
                if self._lsp0 is not None:
                    rho0_full = rho0_full - self._lsp0
                if self._L is None:
                    rho0 = rho0_full
                else:
                    rho0, *_ = np.linalg.lstsq(self._L, rho0_full, rcond=None)
                if include_log_phi:  # REML/ML with estimated scale
                    theta0 = np.concatenate([rho0, [self._gam_g_log_scale0()]])
                else:
                    theta0 = rho0
                crit = (
                    "REML" if method in ("ML", "REML") else self._bam_outer_criterion()
                )
                # bgam.fit's gam(G=G) on this iteration's working (R, f)
                # (bam.r:1261-1272) through the gam.fit3 port
                self._g3 = self._fit3_args_compressed(method)
                if self._g3 is not None:
                    crit = "REML3"
                theta_hat = self._outer_newton(
                    theta0,
                    criterion=crit,
                    include_log_phi=include_log_phi,
                    include_family_theta=False,
                )
                theta_sp_warm = theta_hat[:n_work]
                log_phi_hat = float(theta_hat[n_work]) if include_log_phi else None
                self.sp = np.exp(theta_sp_warm)
                rho_hat = self._rho_full(theta_sp_warm)
                fit = self._fit_given_rho(rho_hat)
                self._fit3_final = None
                if self._g3 is not None and isinstance(self._outer_fit, dict):
                    # the refit's coefficients start the next iteration
                    # (bam.r:1297 coef <- object$coefficients)
                    fit3 = self._outer_fit
                    self._fit3_final = fit3
                    fit.beta = np.asarray(fit3["coefficients"], dtype=float)
                    fit.A_inv = r_tcrossprod(np.asarray(fit3["rV"], dtype=float))

            self._log_phi_hat = log_phi_hat
            fit_build = (
                self._XtX,
                self._Xty,
                self._yty,
                self._X_full,
                self._wt_full,
                self._bam_qr,
            )

            new_coef = last_out["beta"] if (additive and fit is None) else fit.beta
            if not np.all(np.isfinite(new_coef)):
                warnings.warn(
                    f"non-finite coefficients at PIRLS iteration {it + 1}",
                    stacklevel=2,
                )
                break
            coef = new_coef.copy()

        if additive and fit is None and last_out is not None:
            fit = self._fit_given_rho(rho_hat)
            fit.beta = last_out["beta"]
            fit.A_inv = _sl_initial_repara(
                self._sl,
                last_out["PP"],
                inverse=True,
                both_sides=True,
                cov=True,
            )
        # Retain bgam.fitd's final accepted proposal (mgcv ``prop``): its
        # ldetS/ldetXXS feed the reported discrete criterion (bam.r:792)
        # and its hess is Sl.postproc-flavor edf2's Vr input (bam.r:869).
        # None on the non-discrete paths. ``dev`` is the loop's PENALISED
        # deviance (dev + Σ(rSb)², bgam.fitd:611/669) — the value mgcv's
        # final crit statement reads, NOT the response deviance.
        self._pi_last_out = last_out
        self._pi_last_dev = float(dev) if last_out is not None else None

        # Reported rail identity (gam.check reads both). bgam.fitd sets
        # ``optimizer=c("perf","chol")`` (bam.r:784) and, at object assembly,
        # ``outer.info <- list(grad=prop$grad, hess=prop$hess)`` (bam.r:884) —
        # ``prop`` is the last accepted Sl.fitChol proposal, i.e. ``last_out``.
        # Only the fREML rail reaches bgam.fitd; REML/ML run the full gam
        # fitter and keep its ("outer","newton"), GCV.Cp reports "magic".
        if last_out is not None and self._method_in == "fREML":
            self._rail_optimizer = ("perf", "chol")
            self._outer_info = {
                "grad": np.asarray(last_out["grad"], dtype=float),
                "hess": np.asarray(last_out["hess"], dtype=float),
            }

        if not conv:
            warnings.warn("PIRLS algorithm did not converge", stacklevel=2)

        if fit is None:
            raise FloatingPointError("bgam.fit produced no usable fit")

        # bgam.fit breaks on convergence before fitting the new build. Its
        # fREML route post-processes the last fit with that build's R
        # (Sl.postproc, bam.r:1309), but every other method reports the
        # gam(G=G) or magic.post.proc object of the last fit, formed on the
        # build that fit used (bam.r:1220, 1273).
        if not discrete and self._method_in != "fREML" and fit_build is not None:
            (
                self._XtX,
                self._Xty,
                self._yty,
                self._X_full,
                self._wt_full,
                self._bam_qr,
            ) = fit_build

        # bgam.fitd reports the β that built the converged working model
        # (``object$coefficients <- coef``, bam.r:806): near the optimum a
        # step-halved build point, not a further solve from it, and forms
        # the linear predictor as ``Xbd(coef) + offset`` (bam.r:809).
        # bgam.fit instead keeps the last working-model fit's coefficients
        # even when convergence is detected on a halved rebuild, and bam's
        # linear predictor is ``predict.bam`` at those (bam.r:2759).
        if (
            discrete
            and conv
            and not additive
            and coef is not None
            and fit.beta is not None
            and np.asarray(coef).shape == np.asarray(fit.beta).shape
        ):
            cf = np.asarray(coef, dtype=float)
            fit.beta = cf.copy()
            fit.dev = float(self._yty - 2.0 * cf @ self._Xty + cf @ self._XtX @ cf)
        if fit.beta is not None and (discrete or not additive):
            fit.eta = self._chunked_xbeta(np.asarray(fit.beta, dtype=float)) + off
            fit.mu = link.linkinv(fit.eta)

        self._rho_hat = rho_hat if rho_hat is not None else np.zeros(0)
        self._log_phi_hat = log_phi_hat
        self._iter = it + 1
        return fit, self._rho_hat

    def _post_fit_pirls(
        self, fit, rho_hat: np.ndarray, X_param_df: pl.DataFrame
    ) -> None:
        """Populate user-facing attributes after PIRLS converges.
        Mirrors gam.__init__'s post-fit (gam.py:476-783) on the (R, f) reduced
        """
        n, p = self.n, self.p
        method = self.method
        n_sp = len(self._slots)
        family = self.family
        y = self._y_arr
        beta = fit.beta
        self._rho_hat = rho_hat
        # GCV.Cp: the last PIRLS iteration's magic fit and magic.post.proc
        # (bam.r:1278-1295).
        mg = getattr(self, "_bam_magic_res", None) if method == "GCV.Cp" else None
        if mg is not None:
            self.full_sp = np.asarray(mg["fit"]["sp_full"], dtype=float)
        else:
            self.full_sp = np.exp(np.asarray(rho_hat, dtype=float))

        if fit.A_inv is not None:
            A_inv = fit.A_inv
        else:
            A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        XtWX = self._XtX  # = R'R = X'WX at converged β̂
        A_inv_XtWX = A_inv @ XtWX
        # bgam.fit's fREML rail reports Sl.postproc's edf, covariances and
        # edf1/edf2 from the last fast.REML.fit and the last accumulated R
        # (bam.r:1309-1319).
        post = None
        fr = getattr(self, "_freml", None)
        if (
            fr is not None
            and self._method_in == "fREML"
            and n_sp > 0
            and self._discrete_design is None
        ):
            post = _sl_postproc(
                self._sl,
                fr["fit"],
                fr["undrop"],
                self._bam_qr.R,
                fr["hess"],
                scale=(self._scale_fixed_value if self._scale_known_fit else -1.0),
                L=self._L,
            )
        pi_last = getattr(self, "_pi_last_out", None)
        if (
            post is None
            and pi_last is not None
            and self._discrete_design is not None
            and method == "REML"
        ):
            scale_d = (
                float(self._scale_fixed_value)
                if self._scale_known_fit
                else float(np.exp(self._log_phi_hat))
            )
            post = _bgam_fitd_postproc(self._sl, pi_last, self._XtX, scale_d, self._L)
        self._sl_post = post
        # the last PIRLS iteration's gam(G=G) refit on the gam.fit3 port:
        # its gam.fit3.post.proc is what bgam.fit returns (bam.r:1261-1272)
        fit3 = getattr(self, "_fit3_final", None)
        post3 = self._fit3_post_proc(fit3, method) if fit3 is not None else None
        if post3 is not None:
            edf = post3["edf"]
            edf_total = r_sum(edf)
        elif post is not None:
            edf = post["edf"]
            edf_total = r_sum(edf)
        elif mg is not None:
            edf = np.asarray(mg["post"]["edf"], dtype=float)
            edf_total = r_sum(edf)
        else:
            edf = np.diag(A_inv_XtWX).copy()
            edf_total = float(edf.sum())

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
            if post3 is not None:
                pearson_scale = float(fit3["scale.est"])
            elif self._log_phi_hat is not None and (
                self._method_in == "fREML" or self._discrete_design is not None
            ):
                pearson_scale = float(np.exp(self._log_phi_hat))
            elif mg is not None:
                # GCV.Cp — magic's final-fit ``fit$scale`` (bam.r:1294),
                # (‖f−Rβ̂‖² + extra.rss)/(n − trA) on the last working set.
                pearson_scale = float(mg["fit"]["scale"])
            else:
                # the last refit's scale.est on its own working problem
                # (gam.fit3.r:596-606 with the extras; a Gaussian has no
                # Fletcher correction): (|f - R beta|^2 + rss.extra) /
                # (n - edf). mgcv's non-fREML bam methods report this, not
                # the optimizer's exp(log phi).
                qr_ = self._bam_qr
                res_ = (
                    np.asarray(qr_.f, dtype=float)
                    - r_matprod(
                        np.asarray(qr_.R, dtype=float),
                        np.asarray(fit.beta, dtype=float).reshape(-1, 1),
                    )[:, 0]
                )
                pearson_scale = (r_sum(res_ * res_) + float(qr_.rss_extra)) / df_resid
        else:
            pearson_scale = (
                self._scale_fixed_value if self._scale_known_fit else float("nan")
            )
        self._pearson_scale = pearson_scale
        scale = self._scale_fixed_value if self._scale_known_fit else pearson_scale
        sigma_squared = scale
        sigma = (
            float(np.sqrt(sigma_squared))
            if np.isfinite(sigma_squared) and sigma_squared >= 0
            else float("nan")
        )

        if post3 is not None:
            Vp = post3["Vp"]
            Ve = post3["Ve"]
        elif post is not None:
            Vp = post["Vp"]
            Ve = post["Ve"]
        elif mg is not None:
            Vp = np.asarray(mg["post"]["Vb"], dtype=float)
            Ve = np.asarray(mg["post"]["Ve"], dtype=float)
        else:
            Vp = sigma_squared * A_inv
            Ve = sigma_squared * A_inv_XtWX @ A_inv

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

        eta = fit.eta
        mu = fit.mu
        self.linear_predictors = eta
        self.fitted_values = mu
        self.fitted = mu
        # bam.r:2770-2774: deviance residuals sqrt(dev.resids)*sign(y - mu)
        # for a family without its own ``residuals`` (betar, ocat and ziP
        # have one, efam.r; ziP's reads the linear predictor), and the
        # deviance as their sum of squares.
        di = family.dev_resids(y, mu, wt)
        if isinstance(family, (betar, ocat, ziP)):
            mu_r = eta if isinstance(family, ziP) else mu
            self.residuals = np.asarray(
                family.residuals_extended(y, mu_r, wt, "deviance"), dtype=float
            )
        else:
            with np.errstate(invalid="ignore"):
                self.residuals = np.sqrt(di) * np.sign(y - mu)
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
        self.deviance = float(r_sum(self.residuals * self.residuals))
        self.rss = self.deviance  # Gaussian-era alias

        # bam.r:2790 (bgam.fit:1331 for extended families): the null
        # deviance about weighted.mean(y, w), with or without an intercept
        mu_null = np.full(n, r_sum((y * wt)[wt != 0]) / r_sum(wt))
        self.null_deviance = r_sum(family.dev_resids(y, mu_null, wt))
        self.df_null = float(n - 1) if self._has_intercept else float(n)
        # Extended-family postproc (bam.r:1322-1331): find.null.dev
        # null deviance + θ-embedding family relabel (tw included —
        # mgcv's efam set).
        self._postproc = {}
        if self._family_mgcv_extended:
            pp = family.postproc(
                y,
                prior_weights=wt,
                fitted=mu,
                linear_predictors=eta,
                offset=self._offset,
                intercept=self._has_intercept,
            )
            self._postproc = pp
            if pp.get("null_deviance") is not None:
                self.null_deviance = float(pp["null_deviance"])
            if pp.get("deviance") is not None:
                # bam.r:1328 - the family's own deviance replaces
                # sum(residuals^2) (bam.r:2774 keeps it)
                self.deviance = float(pp["deviance"])
                self.rss = self.deviance

        self.Vp = Vp
        self.Ve = Ve
        self._A_inv = A_inv
        self.edf = edf
        self.edf_total = edf_total
        edf_by_smooth: dict[str, float] = {}
        for b, (a, bcol) in zip(self._blocks, self._block_col_ranges):
            edf_by_smooth[b.label] = float(edf[a:bcol].sum())
        self.edf_by_smooth = edf_by_smooth

        ss_resid_response = float(np.sum(wt * (y - mu) ** 2))
        if self._has_intercept and self._tss > 0:
            r_squared = 1.0 - ss_resid_response / self._tss
        elif self._yty_full > 0:
            r_squared = 1.0 - ss_resid_response / self._yty_full
        else:
            r_squared = float("nan")
        if df_resid > 0 and n > 1:
            sqrt_wt = np.sqrt(wt)
            mean_y_w = r_sum(wt * y) / r_sum(wt)
            v_resid = r_var(sqrt_wt * (y - mu))
            v_total = r_var(sqrt_wt * (y - mean_y_w))
            if v_total > 0:
                r_squared_adjusted = 1.0 - v_resid * (n - 1) / (v_total * df_resid)
            else:
                r_squared_adjusted = float("nan")
        else:
            r_squared_adjusted = float("nan")
        # mgcv summary.gam: r.sq NULL for no.r.sq families (mgcv.r:4055).
        if getattr(self.family, "no_r_sq", False):
            r_squared = r_squared_adjusted = float("nan")
        self.r_squared = float(r_squared)
        self.r_squared_adjusted = float(r_squared_adjusted)
        if self.null_deviance > 0:
            self.deviance_explained = float(
                (self.null_deviance - self.deviance) / self.null_deviance
            )
        else:
            self.deviance_explained = float("nan")

        self._fisher_w = None

        if post3 is not None:
            self.rank = int(fit3["rank"])
        else:
            self.rank = (
                int(mg["fit"]["gcv_info"]["rank"])
                if mg is not None
                else self._estimate_rank()
            )

        if post3 is not None:
            hess3 = (self._outer_info or {}).get("hess")
            H_aug = (
                np.asarray(hess3, dtype=float)
                if method in ("REML", "ML", "P-REML", "P-ML") and hess3 is not None
                else None
            )
        elif (
            method in ("REML", "ML", "P-REML", "P-ML")
            and n_sp > 0
            and np.isfinite(sigma_squared)
            and sigma_squared > 0
        ):
            if self._pearson_scale_criterion:
                # P-REML / P-ML: the criterion's own ρ-only Hessian at the
                # Pearson plug-in scale (gam.fit3.r:979) — see
                # _post_fit_gaussian.
                H_aug = self._preml_hessian(rho_hat, fit=fit)
                T_aug = self._T_working(0)
            elif self._method_in in ("REML", "ML") and np.size(
                (self._outer_info or {}).get("hess", [])
            ):
                # gam(G=G)'s outer Newton Hessian at its optimum, in working
                # coordinates: what gam.fit3.post.proc reads (mgcv.r:1735)
                H_aug = np.asarray(self._outer_info["hess"], dtype=float)
                T_aug = None
            else:
                log_phi_hat_for_aug = (
                    self._log_phi_hat
                    if self._log_phi_hat is not None
                    else float(np.log(sigma_squared))
                )
                # a known scale is no outer parameter (as in gam's post-fit)
                include_phi_aug = not self._scale_known_fit
                H_aug = 0.5 * self._reml_hessian(
                    rho_hat,
                    log_phi_hat_for_aug,
                    fit=fit,
                    include_log_phi=include_phi_aug,
                )
                T_aug = self._T_working(1 if include_phi_aug else 0)
            if T_aug is not None:
                H_aug = T_aug.T @ H_aug @ T_aug
            H_aug = 0.5 * (H_aug + H_aug.T)
        else:
            H_aug = None
        self._H_aug = H_aug

        if post3 is not None:
            self.edf1 = post3["edf1"]
            self.edf2 = post3["edf2"] if post3["edf2"] is not None else edf.copy()
            self.edf1_total = r_sum(self.edf1)
            self.edf2_total = r_sum(self.edf2)
            Vc_corr = np.zeros_like(Vp)
        elif mg is not None:
            # magic.post.proc's edf1; edf2 and Vc are NULL (bam.r:1278-1295),
            # so edf2 := edf for logLik.gam/AIC.
            self.edf1 = np.asarray(mg["post"]["edf1"], dtype=float)
            self.edf2 = edf.copy()
            self.edf1_total = r_sum(self.edf1)
            self.edf2_total = edf_total
            Vc_corr = np.zeros_like(Vp)
        elif n_sp > 0:
            edf2_per_coef, edf1_per_coef, Vc_corr = self._compute_edf12(
                rho_hat,
                fit,
                sigma_squared,
                A_inv,
                A_inv_XtWX,
                edf,
                H_aug,
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
        if post is not None:
            # bam.r:2789 caps edf2 by its sum.
            edf2 = post["edf2"]
            if r_sum(edf2) > r_sum(post["edf1"]):
                edf2 = post["edf1"].copy()
            self.edf1 = post["edf1"]
            self.edf2 = edf2
            self.edf1_total = r_sum(post["edf1"])
            self.edf2_total = r_sum(edf2)
            self.Vc = post["Vc"]
        elif post3 is not None and post3["Vc"] is not None:
            self.Vc = post3["Vc"]
        else:
            self.Vc = Vp + Vc_corr

        sc_p = 0.0 if self._scale_known_fit else 1.0
        dev1 = family._aic_dev1(self.deviance, sigma_squared, wt)
        n_aic = self._binom_n if self._binom_n is not None else n
        family_aic = float(family.aic(y, fit.mu, dev1, wt, n_aic))
        mgcv_aic = family_aic + 2.0 * edf_total
        logLik = sc_p + edf_total - 0.5 * mgcv_aic
        # GCV.Cp/GACV.Cp: mgcv-bam's magic rail sets no edf2 (bam.r:
        # 1278-1295) → logLik.gam df falls back to edf (mgcv.r:4431).
        df_base = (
            self.edf2_total if method in ("REML", "ML", "P-REML", "P-ML") else edf_total
        )
        df_for_aic = min(df_base + sc_p, float(p) + sc_p)
        # logLik.gam (mgcv.r:4434): extended families add n.theta AFTER
        # the np cap.
        if self._family_mgcv_extended:
            df_for_aic += float(getattr(self.family, "n_theta", 0) or 0)
        self.loglike = float(logLik)
        self.logLik = self.loglike
        self.npar = float(df_for_aic)
        self.AIC = -2.0 * logLik + 2.0 * df_for_aic
        self.BIC = -2.0 * logLik + float(np.log(n)) * df_for_aic
        self._mgcv_aic = float(mgcv_aic)

        if method in ("REML", "ML"):
            pi_out = getattr(self, "_pi_last_out", None)
            if pi_out is not None:
                # DISCRETE rail: bgam.fitd's own reported criterion — the
                # `crit <-` statement ends at `)/2` (bam.r:792); the
                # trailing `(length(y)/gamma-Mp)*log(2*pi*scale)` line
                # (bam.r:794) is a bare DISCARDED expression, confirmed
                # by R's own parse of the body. `dev` there is the loop's
                # PENALISED deviance (bgam.fitd:611/669), `lsat` the
                # family saturated log-lik at φ̂, ldetS/ldetXXS the final
                # proposal's. bgam.fitd attaches no "ml.pen" attr to
                # gcv.ubre, so the bam.r:2784 aic-replacement never fires
                # here — no _ml_flavor_swap, no dev-swap.
                phi = (
                    float(self._scale_fixed_value)
                    if self._scale_known_fit
                    else float(np.exp(self._log_phi_hat))
                )
                if self._family_mgcv_extended:
                    lsat = float(
                        self.family.ls_extended(
                            y, wt, theta=self.family.get_theta(), scale=phi
                        )["ls"]
                    )
                elif self._binom_n is not None:
                    lsat = float(self.family.ls(y, wt, phi, n=self._binom_n)[0])
                else:
                    lsat = float(self.family.ls(y, wt, phi)[0])
                gam_ = self._gamma
                dev_pen = float(self._pi_last_dev)
                crit = (
                    (dev_pen - 2.0 * lsat * phi) / (phi * gam_)
                    - float(pi_out["ldetS"])
                    + float(pi_out["ldetXXS"])
                ) / 2.0
                if self._rho != 0.0:
                    # AR1 correction (bam.r:795-798).
                    ld = 1.0 / np.sqrt(1.0 - self._rho**2)
                    df_ar = (
                        int(self._ar_start.sum()) if self._ar_start is not None else 1
                    )
                    crit -= (n / gam_ - df_ar) * float(np.log(ld))
                score = 2.0 * float(crit)  # hea stores 2V
            elif post is not None or post3 is not None:
                # fast.REML.fit's reml and gam.fit3's REML carry ml.pen, so
                # bam reports the family's own aic in place of the working
                # log-likelihood: gcv.ubre <- aic/(2*gamma) + ml.pen
                # (bam.r:2784-2786).
                ml_pen = (
                    float(fit3["ml_pen"])
                    if post3 is not None
                    else float(fr["fit"]["reml_pen"])
                )
                score = 2.0 * (family_aic / (2 * self._gamma) + ml_pen)
            else:
                if self._log_phi_hat is not None:
                    log_phi_hat = self._log_phi_hat
                elif self._scale_known_fit:
                    log_phi_hat = math.log(self._scale_fixed_value)
                else:
                    log_phi_hat = 0.0
                score = float(self._reml(rho_hat, log_phi_hat, fit=fit))
                # mgcv's report puts the family's own aic in place of the
                # working log-likelihood, aic/(2 gamma) + ml.pen (bam.r:
                # 2784-2786); ml.pen is the criterion less its
                # (dev/phi - 2 ls)/gamma part, taken with the deviance and
                # saturated log-likelihood the criterion used
                if np.isfinite(score):
                    score = score + self._ml_flavor_swap(
                        family_aic, log_phi_hat, float(fit.dev)
                    )
            if method == "REML":
                self.REML_criterion = score
            else:
                self.ML_criterion = score
        elif method in ("P-REML", "P-ML"):
            # mgcv's non-fREML methods report the COMPRESSED gam(G=G)
            # refit's criterion — evaluated on the working statistics, no
            # response-deviance swap (that swap mirrors fREML's bam.r:1084
            # dev recompute, which this route never does). 2V units.
            if post3 is not None:
                score = 2.0 * float(self._outer_info["score"])
            elif n_sp > 0:
                score = 2.0 * float(
                    self._score_all(rho_hat, 0.0, fit, 0, "PREML")["score"]
                )
            else:
                score = float("nan")
            if method == "P-REML":
                self.REML_criterion = score
            else:
                self.ML_criterion = score
        elif method == "GACV.Cp":
            if post3 is not None:
                self.GCV_score = float(self._outer_info["score"])
            elif n_sp > 0:
                self.GCV_score = float(
                    self._score_all(
                        rho_hat,
                        0.0,
                        fit,
                        0,
                        self._bam_outer_criterion(),
                    )["score"]
                )
            else:
                self.GCV_score = float("nan")
        elif mg is not None:
            # the last PIRLS iteration's magic score (bam.r:1287
            # ``object$gcv.ubre <- fit$score``).
            self.GCV_score = float(mg["fit"]["score"])
        else:
            self.GCV_score = float(self._gcv(rho_hat)) if n_sp > 0 else float("nan")

        self.vcomp = self._compute_vcomp()
        self.AR1_rho = self._rho  # always 0 for the non-Gaussian path


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


def _discrete_frame_df(mf_dict: dict, source: pl.DataFrame) -> pl.DataFrame:
    """The discretized frame ``mf0`` as a DataFrame. A factor keeps its
    levels and their order (compress.df returns the factor itself, so the
    smooths built on mf0 see the original levels): each factor column is
    cast to an Enum of the source column's levels."""
    if not mf_dict:
        return pl.DataFrame()
    out = pl.DataFrame(mf_dict)
    casts = []
    for nm in out.columns:
        if nm in source.columns and _is_factor_like(source[nm]):
            levels = [str(v) for v in _factor_levels(source[nm])]
            casts.append(pl.col(nm).cast(pl.String).cast(pl.Enum(levels)))
    return out.with_columns(casts) if casts else out


def compress_df(
    dat: dict[str, np.ndarray],
    m: int | None = None,
    *,
    rng: RMersenneTwister | None = None,
) -> _CompressResult:
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
        m = round(m ** (1.0 / d)) + 1

    is_factor = {nm: _is_factor_arr(dat[nm]) for nm in names}
    is_matrix = {nm: dat[nm].ndim == 2 for nm in names}

    mf_total = 1
    mm_total = 1
    for nm in names:
        if is_factor[nm]:
            mf_total *= int(np.unique(np.asarray(dat[nm]).ravel()).size)
        else:
            mm_total *= m

    matrix_input = is_matrix[names[0]]
    if matrix_input:
        ncols_mat = dat[names[0]].shape[1]
        flat: dict[str, np.ndarray] = {}
        for nm in names:
            arr = dat[nm]
            if arr.ndim == 1:
                arr = np.broadcast_to(arr[:, None], (n, ncols_mat))
            flat[nm] = arr.reshape(-1)
        work = flat
        n_eff = n * ncols_mat
    else:
        work = {nm: np.asarray(dat[nm]).ravel() for nm in names}
        n_eff = n

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
        k_out = np.arange(n_eff, dtype=np.int64)
        if matrix_input:
            k_out = k_out.reshape(n, ncols_mat)
        return _CompressResult(xu={nm: work[nm].copy() for nm in names}, k=k_out)

    # Shuffle xu rows to break induced dependencies (bam.r:171).
    perm = rng.sample_no_replace(nu, nu)
    xu_table = {nm: xu_table[nm][np.argsort(perm)] for nm in names}
    k_idx = perm[k_idx]

    if matrix_input:
        k_idx = k_idx.reshape(n, ncols_mat)

    return _CompressResult(xu=xu_table, k=k_idx.astype(np.int64))


def _is_factor_arr(a: np.ndarray) -> bool:
    """Treat string / unicode / object arrays as factors."""
    return a.dtype.kind in ("U", "O", "S")


_UNIQUE_FAST_SPAN_CAP = 1 << 20


def _distinct_exceeds_1d(a: np.ndarray, threshold: int) -> bool:
    """Exact predicate: does ``a`` have **more than** ``threshold`` distinct
    values? Early-exits as soon as ``threshold + 1`` distinct are seen.
    uses to decide whether to round (bam.r:152), so the round/keep decision is
    identical to mgcv's; only the way the boolean is computed differs.
    """
    n = a.size
    if n <= threshold:
        return False
    step = max(int(threshold) + 1, 1 << 16)
    seen: np.ndarray | None = None
    for s in range(0, n, step):
        u = np.unique(a[s : s + step])
        seen = u if seen is None else np.union1d(seen, u)
        if seen.size > threshold:
            return True
    return False


def _unique_inverse(col: np.ndarray, max_unique: int | None = None):
    """``np.unique(col, return_inverse=True)`` with an O(n) fast path for
    low-cardinality integer-valued columns — **byte-identical** output.
    """
    a = np.asarray(col)
    if a.size and a.dtype.kind in "fiu":
        mn = a.min()
        mx = a.max()
        if np.isfinite(mn) and np.isfinite(mx):
            span = round(float(mx - mn)) + 1
            if 1 <= span <= _UNIQUE_FAST_SPAN_CAP:
                codes = np.rint(a - mn).astype(np.intp)
                seen = np.zeros(span, dtype=bool)
                seen[codes] = True
                present = np.nonzero(seen)[0]  # sorted distinct
                u = (present + mn).astype(a.dtype)  # sorted unique
                remap = np.empty(span, dtype=np.int64)
                remap[present] = np.arange(present.size, dtype=np.int64)
                inv = remap[codes]
                if np.array_equal(u[inv], a):
                    return u, inv
    if max_unique is not None and _distinct_exceeds_1d(a, max_unique):
        return None, None
    u, inv = np.unique(a, return_inverse=True)
    return u, np.asarray(inv).reshape(-1).astype(np.int64)


def _uniquecombs(
    work: dict[str, np.ndarray], names: list[str], max_unique: int | None = None
):
    """Numpy port of R's ``uniquecombs`` (single-thread)."""
    n = next(iter(work.values())).size
    if len(names) == 1:
        col = work[names[0]]
        u, inv = _unique_inverse(col, max_unique=max_unique)
        if u is None:
            return None, None  # EXCEEDED — caller rounds, see compress_df
        return {names[0]: u}, inv
    keys: list[np.ndarray] = []
    for nm in names:
        col = work[nm]
        if _is_factor_arr(col):
            _, codes = np.unique(col, return_inverse=True)
            keys.append(codes)
        else:
            keys.append(col)
    dtype = np.dtype([(nm, k.dtype) for nm, k in zip(names, keys)])
    arr = np.empty(n, dtype=dtype)
    for nm, k in zip(names, keys):
        arr[nm] = k
    u, inv = np.unique(arr, return_inverse=True)
    xu = {nm: u[nm] for nm in names}
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
                raise ValueError("bam can not discretize with this nesting structure")
            return ki[i_min]
        raise ValueError("bam can not discretize with this nesting structure")
    return 0


def _pmf_names(expanded, response: str | None, data_columns) -> list[str]:
    """mgcv's ``pmf.names``: ``names(model.frame(parametric_formula, data))``
    (bam.r:2314-2318), the list :func:`discrete_mf` compresses one variable at
    a time.
    discretises it, since there ``names.pmf %in% names(mf)`` holds (bam.r:244).
    """
    cols = set(data_columns)
    names: list[str] = []
    if response:
        names.append(response)
    for t in expanded.terms:
        for a in t.atoms:
            if isinstance(a, Name) and a.ident in cols and a.ident not in names:
                names.append(a.ident)
    return names


def discrete_mf(
    smooth_specs: list[dict],
    mf: pl.DataFrame,
    names_pmf: Sequence[str],
    m: int | None = None,
    *,
    rng: RMersenneTwister | None = None,
    full: bool = True,
) -> DiscretizedFrame:
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
    nk = 0
    for spec in smooth_specs:
        margins = spec.get("margins", [{"term": spec["term"]}])
        n_marg_vars = sum(len(marg["term"]) for marg in margins)
        nk += n_marg_vars + (1 if spec.get("by") not in (None, "NA") else 0)
    pmf_in_mf = [nm for nm in names_pmf if nm in mf.columns]
    nk += len(pmf_in_mf)

    # Bookkeeping. ``k`` will grow with extra columns when matrix-arg
    # smooths are encountered (each matrix column becomes one ``k`` column).
    #
    # Column-major, as mgcv's ``kd`` is: every kernel reads whole index
    # COLUMNS (``Ki = k + (ks[im]+s)*n``, discrete.c:1838) and never a row, so
    # F order makes each ``k[:, q]`` a contiguous view. In C order it is a
    # strided slice that every consumer has to copy — ~0.9 MB per access at
    # n = 110k, several times per block.
    k = np.zeros((n, nk), dtype=np.int64, order="F")
    ks = np.full((nk, 2), -1, dtype=np.int64)
    nr = np.zeros(nk, dtype=np.int64)
    var_order: list[str] = []
    mf0: dict[str, np.ndarray] = {}
    rec = {"vnames": [], "ki": [], "d": []}
    ik = -1  # 0-based marginal index counter (mgcv ``ik`` is 1-based)

    def _discretise_marginal(termi: list[str], mi: int | None):
        nonlocal ik
        prev = check_term(termi, rec)
        if prev != 0:
            return  # already discretised — re-use the entry
        ik += 1
        dat: dict[str, np.ndarray] = {}
        for nm in termi:
            s = mf[nm]
            if is_matrix_col(s):
                dat[nm] = matrix_to_2d(s)
            else:
                dat[nm] = s.to_numpy()
        cr = compress_df(dat, m=mi, rng=rng)
        ki = cr.k  # (n,) or (n, m_cols)
        if ki.ndim == 1:
            ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
            ks[ik, 1] = ks[ik, 0] + 1
            if ks[ik, 1] > k.shape[1]:
                k_ext = np.zeros(
                    (n, ks[ik, 1] - k.shape[1]),
                    dtype=np.int64,
                )
                k_full = np.concatenate([k, k_ext], axis=1)
                _set_k(k_full)
                k_full[:, ks[ik, 0]] = ki
            else:
                k[:, ks[ik, 0]] = ki
        else:
            ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
            ks[ik, 1] = ks[ik, 0] + ki.shape[1]
            need_cols = ks[ik, 1] - k.shape[1]
            if need_cols > 0:
                k_ext = np.zeros((n, need_cols), dtype=np.int64)
                k_full = np.concatenate([k, k_ext], axis=1)
                _set_k(k_full)
                k_local = k_full
            else:
                k_local = k
            k_local[:, ks[ik, 0] : ks[ik, 1]] = ki
        nr[ik] = cr.xu[termi[0]].size
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
        rec["vnames"].extend(termi)
        rec["ki"].extend([ik - len(termi) + 1] * len(termi))
        rec["d"].extend([len(termi)] * len(termi))

    def _set_k(k_new: np.ndarray):
        nonlocal k
        k = k_new

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

    for nm in pmf_in_mf:
        if check_term([nm], rec) != 0:
            continue
        ik += 1
        s = mf[nm]
        if is_matrix_col(s):
            # mgcv's parametric-matrix workaround (bam.r:329-333):
            # ``compress.df(mfp[[i]], m=mi)`` on the BARE matrix — each
            # matrix column is a separate variable (d = ncol, unique
            # ROWS), NOT the smooth summation-convention vectorisation
            # (compress.df's ``is.matrix(dat[[1]])`` branch never fires
            # on a bare matrix). Reached by the censored/gfam families'
            # cbind(y, yat|index) response and by matrix parametric
            # covariates. One shared k column, like mgcv.
            arr = matrix_to_2d(s)
            dcols = {f"{nm}[,{j}]": arr[:, j] for j in range(arr.shape[1])}
            cr = compress_df(dcols, m=m, rng=rng)
            mf_entry = np.column_stack([cr.xu[c] for c in dcols])
            if mf_entry.shape[1] == 1:
                mf_entry = mf_entry[:, 0]
            ki = cr.k
            nr[ik] = mf_entry.shape[0]
        else:
            arr = s.to_numpy()
            cr = compress_df({nm: arr}, m=m, rng=rng)
            mf_entry = cr.xu[nm]
            ki = cr.k
            nr[ik] = mf_entry.size
        ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
        ks[ik, 1] = ks[ik, 0] + 1
        if ks[ik, 1] > k.shape[1]:
            k = np.concatenate(
                [k, np.zeros((n, ks[ik, 1] - k.shape[1]), dtype=np.int64)],
                axis=1,
            )
        k[:, ks[ik, 0]] = ki
        var_order.append(nm)
        mf0[nm] = mf_entry

    if full and mf0:
        maxr = max(arr.shape[0] for arr in mf0.values())
        for nm, arr in list(mf0.items()):
            me = arr.shape[0]
            if me >= maxr:
                continue
            if arr.ndim == 2:
                # mgcv (bam.r:354-356): rbind(mf0[[i]], matrix(sample(
                # mf0[[i]], (maxr-me)*ncol, TRUE), maxr-me, ncol)) —
                # sample() flattens the matrix COLUMN-MAJOR and matrix()
                # refills column-major.
                pool = arr.ravel(order="F")
                draw = pool[rng.sample_replace(pool.size, (maxr - me) * arr.shape[1])]
                pad = np.reshape(draw, (maxr - me, arr.shape[1]), order="F")
                mf0[nm] = np.concatenate([arr, pad], axis=0)
            else:
                pad = arr[rng.sample_replace(me, maxr - me)]
                mf0[nm] = np.concatenate([arr, pad])
    else:
        maxr = max((arr.shape[0] for arr in mf0.values()), default=0)

    ik += 1
    used_cols = int(np.max(ks[:ik]) if ik > 0 else 0)
    if used_cols < k.shape[1]:
        k = k[:, :used_cols]
    elif used_cols > k.shape[1]:
        pass
    k = np.asfortranarray(np.concatenate([k, np.zeros((n, 1), dtype=np.int64)], axis=1))
    ks_final = np.concatenate(
        [ks[:ik], np.array([[k.shape[1] - 1, k.shape[1]]], dtype=np.int64)],
        axis=0,
    )
    nr_final = np.concatenate([nr[:ik], np.array([1], dtype=np.int64)])
    var_order.append("(Intercept)")
    mf0["(Intercept)"] = np.ones(1, dtype=float)

    return DiscretizedFrame(
        mf=mf0,
        k=k,
        ks=ks_final,
        nr=nr_final,
        names=var_order,
        n=n,
    )


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

    kind: str  # "param" | "single" | "tensor"
    Xd_list: list[np.ndarray]  # one (m_j, p_j) block per marginal
    k_cols: list[tuple[int, int]]  # (start, stop) into global k for each marginal
    coef_slice: slice  # where this term lives in the full coef
    qc: int = 0  # tensor-constraint indicator (1 if Householder)
    v: np.ndarray | None = None  # Householder vec, length = Π p_j (qc=1 case)
    # absorb / by / keep_cols for the term. The constraint ``T`` (absorb /
    # keep_cols) is applied by the kernels via ``_design_constraint_Ts``; None
    # for params and unconstrained smooths. ``by`` records the by-spec for
    # reference only — the by= weighting is carried by the by-marginal that
    # :func:`build_discrete_design` prepends to ``Xd_list`` (mgcv
    # discrete.mf:261-294 / bam.r:2469-2483), not by any post-hoc column mask.
    absorb: object | None = None
    by: object | None = None
    keep_cols: np.ndarray | None = None
    spec: BasisSpec | None = None
    label: str = ""
    _k_rows_cache: tuple | None = None


@dataclass(slots=True)
class DiscreteDesign:
    """Compressed design store consumed by the discrete kernels.

    Mirrors mgcv's ``G$Xd``/``G$k``/``G$ks``/``G$ts``/``G$dt``/``G$v``/
    ``G$qc``/``G$drop`` package (bam.r:2300+ in ``bam`` setup), but
    represented as a Python dataclass with one :class:`_DiscreteTerm`
    per term. ``p`` is the coefficient count the fit sees: the terms'
    post-constraint widths less the ``drop`` columns. ``drop`` is mgcv's
    ``G$drop`` (bam.r:2498-2502), 0-based positions in the undropped
    coefficient vector that the terms' ``coef_slice`` index: the columns
    gam.side deleted from a tensor term, which bam keeps whole for the
    kernels. X'WX and X'Wy are formed over every column and these removed
    (misc.r:272, 331), and Xbd reads beta with zeros inserted there
    (misc.r:395-399).
    """

    terms: list[_DiscreteTerm]
    k: np.ndarray  # (n, n_k_cols) global index matrix
    ks: np.ndarray  # (n_vars+1, 2) per-variable k-column slice
    nr: np.ndarray  # (n_vars+1,) per-variable unique-row count
    n: int  # original observation count
    p: int  # coefficient count after the drop
    var_index: dict[str, int]  # variable name → row in ks/nr
    drop: np.ndarray | None = None
    _full_X_cache: object = None
    _T_cache: object = None
    _mgcv_cache: object = None


@dataclass(slots=True)
class _ParamBlock:
    """One block of mgcv's parametric ``terms2tensor`` decomposition.

    ``Xd_list``/``k_cols`` present ⇒ the block is stored discretised, exactly
    like a smooth: a row tensor of per-variable marginal model matrices indexed
    through ``k`` (bam.r:2436-2451). ``Xd_list is None`` ⇒ the block keeps its
    un-discretised ``n``-row columns ``X_param_full[:, col_start:col_stop]``.
    """

    label: str
    col_start: int
    col_stop: int
    Xd_list: list[np.ndarray] | None = None
    k_cols: list[tuple[int, int]] | None = None


def param_tensor_blocks(
    expanded,
    param_assign: Sequence[int],
    X_param_full: np.ndarray | None,
    dframe: DiscretizedFrame,
    var_index: dict[str, int],
) -> list[_ParamBlock]:
    """Split the parametric model matrix into mgcv's sequence of marginal
    model matrices — ``terms2tensor`` (bam.r:2091-2174) evaluated on the
    discretised model frame, then truncated to ``nr`` rows (bam.r:2442).

    mgcv stores the parametric part as ordinary discrete terms: the intercept
    becomes a 1×1 marginal read through the all-ones ``k`` column
    (terms2tensor:2115-2122), a main effect becomes that variable's
    ``model.matrix`` on its ``nr`` unique values, and an interaction becomes
    one marginal per variable in REVERSE order (:2150, "to conform with
    model.matrix called directly"), which is hea's ``_khatri_rao`` convention
    read backwards — the row tensor takes its FIRST marginal as slowest-varying
    (``tensorXj``, discrete.c:301) where ``_khatri_rao`` takes the first atom as
    fastest.

    A block is only discretised when every atom is a bare data column that
    :func:`discrete_mf` compressed, and when the resulting marginal widths
    multiply out to the width the dense design actually emitted. The first
    block that fails leaves it — and every later block — dense: the factor
    coding of a term depends on which margins earlier terms already covered
    (``_encode_term_marginals`` mutates ``covered``), so once a term is skipped
    that bookkeeping can no longer be trusted. The intercept is atom-free and
    always decomposes, which is the case that matters: it turns the ``n×p_par``
    weighted product in every parametric×smooth block into mgcv's single
    scatter pass over one ``k`` column.
    """
    assign = np.asarray(param_assign, dtype=int) if len(param_assign) else None
    if assign is None or X_param_full is None or X_param_full.shape[1] == 0:
        return []
    groups: list[tuple[int, Term]] = []
    if getattr(expanded, "intercept", False):
        groups.append((0, Term(())))
    groups += [(i + 1, t) for i, t in enumerate(expanded.terms)]

    mf_dict = {
        nm: arr
        for nm, arr in dframe.mf.items()
        if nm != "(Intercept)" and np.asarray(arr).ndim == 1
    }
    mf0 = pl.DataFrame(mf_dict) if mf_dict else pl.DataFrame()

    covered: set = {Term(())} if getattr(expanded, "intercept", False) else set()
    out: list[_ParamBlock] = []
    dense_from_here = False
    for gid, term in groups:
        cols = np.flatnonzero(assign == gid)
        if cols.size == 0:
            continue
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        if c1 - c0 != cols.size:
            dense_from_here = True
        label = term.label
        if dense_from_here:
            out.append(_ParamBlock(label, c0, c1))
            continue
        if not term.atoms:
            # Intercept: the literal 1×1 matrix (terms2tensor:2117), on the
            # all-ones index column discrete.mf appends (bam.r:376-380).
            xnames = ["(Intercept)"]
            values = [np.ones((1, 1))]
        else:
            names = [a.ident for a in term.atoms if isinstance(a, Name)]
            if len(names) != len(term.atoms) or any(
                nm not in var_index or nm not in mf0.columns for nm in names
            ):
                dense_from_here = True
                out.append(_ParamBlock(label, c0, c1))
                continue
            margs = _encode_term_marginals(term, mf0, covered)
            covered.add(term)
            xnames = names[::-1]
            values = [np.asarray(b.values, dtype=float) for b in margs[::-1]]
        Xd_list = [
            np.ascontiguousarray(v[: int(dframe.nr[var_index[nm]])])
            for v, nm in zip(values, xnames)
        ]
        if int(np.prod([X.shape[1] for X in Xd_list])) != c1 - c0:
            dense_from_here = True
            out.append(_ParamBlock(label, c0, c1))
            continue
        k_cols = [
            (int(dframe.ks[var_index[nm], 0]), int(dframe.ks[var_index[nm], 1]))
            for nm in xnames
        ]
        out.append(_ParamBlock(label, c0, c1, Xd_list, k_cols))
    return out


def build_discrete_design(
    blocks: list[SmoothBlock],
    X_param_full: np.ndarray,
    dframe: DiscretizedFrame,
    *,
    param_terms: Sequence[str] = ("(Intercept)",),
    data: pl.DataFrame | None = None,
    param_tensor: list[_ParamBlock] | None = None,
    fit_frame: bool = False,
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
      * For parametric columns, store one term per block of
        ``param_tensor`` (:func:`param_tensor_blocks`, mgcv's
        ``terms2tensor``); blocks it could not decompose keep the
        un-discretised columns of ``X_param_full`` as a single-marginal
        "term" with ``Xd = X_param_full`` and ``k = identity``. With no
        ``param_tensor`` the whole parametric part stays un-discretised.

    A tensor's single sum-to-zero constraint goes to the kernels as mgcv's
    Householder vector (``qc = 1``) and sz's Kronecker contrasts as
    ``qc = -1``; any other tensor constraint stays a dense column map
    (``absorb``/``keep_cols``) that the kernels apply after the product.
    """
    design, _lpid = _build_discrete_design_parts(
        [(X_param_full, blocks)],
        dframe,
        param_tensors=[param_tensor],
        fit_frame=fit_frame,
    )
    return design


def build_discrete_design_lps(
    lp_parts: list[tuple[np.ndarray | None, list[SmoothBlock]]],
    dframe: DiscretizedFrame,
    param_tensors: list[list[_ParamBlock] | None] | None = None,
    fit_frame: bool = False,
) -> tuple[DiscreteDesign, list[list[int]]]:
    """Multi-linear-predictor :class:`DiscreteDesign` + mgcv's ``lpid``.

    ``lp_parts[j] = (X_param_full_j, blocks_j)`` is LP j's parametric
    matrix and smooth blocks; terms append LP by LP (each LP: param term
    first, then its smooths), so coefficient slices are LP-contiguous —
    hea's stacked multi-formula layout (the dense multi-design puts LP
    j's columns at ``lpi[j] = pof + arange(p_j)``; mgcv instead orders
    discrete terms all-parametric-first and compensates with ``lpip``
    coefficient reordering inside the kernels, bam.r:2422-2453 +
    misc.r:403-409 — hea's per-term ``coef_slice`` makes the two
    equivalent). Returns ``(design, lpid)`` where ``lpid[j]`` lists LP
    j's term indices in ascending coefficient order (mgcv sorts lpid the
    same way, bam.r:2550-2553).

    ``param_tensors[j]`` is LP j's :func:`param_tensor_blocks` decomposition
    (``None`` ⇒ keep that LP's parametric part un-discretised).
    """
    return _build_discrete_design_parts(
        lp_parts, dframe, param_tensors=param_tensors, fit_frame=fit_frame
    )


def _build_discrete_design_parts(
    lp_parts: list[tuple[np.ndarray | None, list[SmoothBlock]]],
    dframe: DiscretizedFrame,
    param_tensors: list[list[_ParamBlock] | None] | None = None,
    fit_frame: bool = False,
) -> tuple[DiscreteDesign, list[list[int]]]:
    var_index = {nm: j for j, nm in enumerate(dframe.names)}
    terms: list[_DiscreteTerm] = []
    lpid: list[list[int]] = []
    drop: list[int] = []
    p_total = 0

    for j_lp, (X_param_full, blocks) in enumerate(lp_parts):
        lp_terms: list[int] = []
        # Parametric part: one discrete term per mgcv ``terms2tensor`` block
        # (bam.r:2436-2451). A block the decomposition could not reach keeps
        # its un-discretised ``n``-row columns as a single Xd with k-col = -1
        # (sentinel: the gather is the identity); the kernels branch on
        # ``kind == "param"`` for those.
        if X_param_full is not None and X_param_full.shape[1] > 0:
            p_par = X_param_full.shape[1]
            # Multi-LP labels get mgcv's ".j" suffix (bam.r:2304/2429
            # appends ".", j-1 to later formulae's term labels).
            suffix = "" if j_lp == 0 else f".{j_lp}"
            pblocks = (
                param_tensors[j_lp]
                if param_tensors is not None and j_lp < len(param_tensors)
                else None
            )
            if not pblocks:
                pblocks = [_ParamBlock("(parametric)", 0, p_par)]
            for pb in pblocks:
                width = pb.col_stop - pb.col_start
                lp_terms.append(len(terms))
                if pb.Xd_list is None:
                    terms.append(
                        _DiscreteTerm(
                            kind="param",
                            Xd_list=[
                                np.ascontiguousarray(
                                    X_param_full[:, pb.col_start : pb.col_stop],
                                    dtype=float,
                                )
                            ],
                            k_cols=[(-1, -1)],
                            coef_slice=slice(p_total, p_total + width),
                            label=pb.label + suffix,
                        )
                    )
                else:
                    terms.append(
                        _DiscreteTerm(
                            kind="single" if len(pb.Xd_list) == 1 else "tensor",
                            Xd_list=pb.Xd_list,
                            k_cols=pb.k_cols,
                            coef_slice=slice(p_total, p_total + width),
                            label=pb.label + suffix,
                        )
                    )
                p_total += width

        for block in blocks:
            lp_terms.append(len(terms))
            p_total = _append_smooth_discrete_term(
                terms,
                block,
                dframe,
                var_index,
                p_total,
                fit_frame=fit_frame,
                drop=drop,
            )
        lpid.append(lp_terms)

    design = DiscreteDesign(
        terms=terms,
        k=dframe.k,
        ks=dframe.ks,
        nr=dframe.nr,
        n=dframe.n,
        p=p_total - len(drop),
        var_index=var_index,
        drop=np.asarray(drop, dtype=np.intp) if drop else None,
    )
    return design, lpid


def _append_smooth_discrete_term(
    terms: list[_DiscreteTerm],
    block: SmoothBlock,
    dframe: DiscretizedFrame,
    var_index: dict[str, int],
    p_total: int,
    *,
    fit_frame: bool = False,
    drop: list[int] | None = None,
) -> int:
    """Append ``block``'s :class:`_DiscreteTerm` to ``terms``; return the
    updated coefficient counter. The per-smooth body of the design build
    (see :func:`build_discrete_design` for the conventions). ``fit_frame``
    says ``dframe`` is the discretised frame the smooths were constructed
    on, so a non-tensor smooth without ``by`` takes its marginal from the
    constructed design, as bam does (bam.r:2527); otherwise every marginal
    is evaluated on ``dframe`` as predict.bamd does (``PredictMat``).
    ``drop`` collects the design's ``G$drop``: a tensor whose columns
    gam.side deleted keeps every column, and the deleted ones are recorded
    at their place in the undropped coefficient vector (bam.r:2498-2502)."""
    spec = block.spec
    if spec is None:
        raise ValueError(
            f"SmoothBlock {block.label!r} has no spec — predict_mat "
            "replay is required for the discrete fitter"
        )
    term_vars = list(block.term)
    raw = spec.predict_raw if spec.predict_raw is not None else spec.raw
    # A t2 term is not a tensor.smooth to bam: one marginal, its
    # constructed design (bam.r:2520-2527). An fs term is one, as the
    # factor's indicators times the base basis (smooth.r:2128-2149).
    if isinstance(raw, (_TensorRawBasis, _FSRawBasis, _SZRawBasis)):
        margin_raws = list(raw.margins)
        margin_vars = [_raw_basis_vars(m) or term_vars for m in margin_raws]
    else:
        margin_raws = [raw]
        margin_vars = [_raw_basis_vars(raw) or term_vars]

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

    p_term = block.X.shape[1]

    # Fitting: bam.r:2527 takes `G$X[1:nr, first.para:last.para]`, the
    # constrained design the constructor built on the discretised frame -
    # bit-for-bit, not a re-evaluation of the basis at those rows.
    constructed = (
        fit_frame
        and len(margin_raws) == 1
        and spec.by is None
        and block.X is not None
        and np.asarray(block.X).shape[0] >= Xd_list[0].shape[0]
    )
    if constructed:
        Xd_list[0] = np.ascontiguousarray(
            np.asarray(block.X, dtype=float)[: Xd_list[0].shape[0]]
        )
    # A tensor's marginals are `margin[[j]]$X[1:nr, ]` (bam.r:2486), the
    # margins as constructed (np-reparameterized) on that frame.
    if (
        fit_frame
        and len(margin_raws) > 1
        and block.margin_X is not None
        and len(block.margin_X) == len(margin_raws)
        and all(
            np.asarray(Xc).shape[0] >= Xd.shape[0]
            and np.asarray(Xc).shape[1] == Xd.shape[1]
            for Xc, Xd in zip(block.margin_X, Xd_list)
        )
    ):
        Xd_list = [
            np.ascontiguousarray(np.asarray(Xc, dtype=float)[: Xd.shape[0]])
            for Xc, Xd in zip(block.margin_X, Xd_list)
        ]

    # mgcv stores the CONSTRAINED basis for a non-tensor smooth: bam.r:2527
    # takes `G$X[1:nr, first.para:last.para]` and leaves `v[[kb]] <-
    # rep(0,0)`, `qc[kb] <- 0` (:2524). Only tensor margins go in raw, with
    # the constraint deferred to `v`/`qc` (:2505-2515), because a tensor's
    # constraint acts on the product space, not on any one margin. Unless
    # the constructed design is used above, that marginal is PredictMat's:
    # for a `by` smooth `G$X` holds `X0 = PredictMat(sm)` with the by unset
    # (smooth.r:4266-4273, gam.setup's restore at mgcv.r:1267-1271), and
    # predict.bamd evaluates PredictMat - the raw basis with the absorbed
    # constraint applied row by row (`t(qr.qty(qrc, t(X)))`), then the
    # columns gam.side kept.
    applied = False
    if (
        not constructed
        and len(margin_raws) == 1
        and (spec.absorb is not None or spec.keep_cols is not None)
    ):
        Xc = Xd_list[-1]
        if spec.absorb is not None:
            Xc = spec.absorb.apply(Xc)
        if spec.keep_cols is not None:
            Xc = Xc[:, spec.keep_cols]
        if Xc.shape[1] == p_term:
            Xd_list[-1] = np.ascontiguousarray(Xc, dtype=float)
            applied = True

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

    # bam.r:2502-2509: a tensor's sum-to-zero constraint goes to the kernels
    # as the Householder vector of the smooth's ``qrc``, ``v = qr /
    # sqrt(qraux)`` with ``v[1] = sqrt(qraux)`` (``qc = 1``).
    qc_t, v_t = 0, None
    ab = spec.absorb
    tensor = len(margin_raws) > 1
    householder = (
        isinstance(ab, _AbsorbTransform)
        and ab.qr is not None
        and ab.nc == 1
        and ab.indi is None
    )
    term_absorb = None if (constructed or applied) else spec.absorb
    term_keep = None if (constructed or applied) else spec.keep_cols
    if (
        tensor
        and spec.keep_cols is not None
        and drop is not None
        and (householder or ab is None)
    ):
        p_term = _term_p_raw_list(Xd_list) - (1 if householder else 0)
        kept = {int(c) for c in spec.keep_cols}
        drop.extend(p_total + c for c in range(p_term) if c not in kept)
        term_keep = None
    if tensor and term_keep is None and householder:
        qr_col = np.asarray(ab.qr, dtype=float).reshape(-1)
        sq = math.sqrt(float(np.asarray(ab.qraux, dtype=float).reshape(-1)[0]))
        v_t = qr_col / sq
        v_t[0] = sq
        qc_t = 1
    elif isinstance(raw, _SZRawBasis):
        # the Kronecker sum-to-zero constraint (bam.r:2508-2511): v holds
        # the number of contrasts, their dimensions and the constraint count
        p_raw = int(np.prod([Xd.shape[1] for Xd in Xd_list]))
        v_t = np.array([len(raw.nf), *raw.nf, p_raw - p_term], dtype=float)
        qc_t = -1

    terms.append(
        _DiscreteTerm(
            kind=kind,
            qc=qc_t,
            v=v_t,
            Xd_list=Xd_list,
            k_cols=k_cols,
            coef_slice=slice(p_total, p_total + p_term),
            # Applied to the marginal above ⇒ no post-hoc T (mgcv's
            # `v=rep(0,0)`, `qc=0` for a non-tensor smooth, bam.r:2524).
            absorb=term_absorb,
            by=spec.by,
            keep_cols=term_keep,
            spec=spec,
            label=block.label,
        )
    )
    p_total += p_term
    return p_total


def _fs_find_factor_any(term_vars, data):
    """The factor variables of a term and the others (an sz term may have
    several factors)."""
    facs = [t for t in term_vars if _is_factor_like(data[t])]
    return facs, [t for t in term_vars if t not in facs]


def _raw_basis_vars(raw: _RawBasis) -> list[str]:
    """Return the variable names a raw basis evaluates on."""
    inner = raw
    while isinstance(inner, _LinearTransformRawBasis):
        inner = inner.inner
    term_attr = getattr(inner, "term", None)
    if term_attr is None:
        return []
    if isinstance(term_attr, str):
        return [term_attr]
    return list(term_attr)


def _split_term_vars_by_margins(
    term_vars: list[str], margin_raws: list[_RawBasis]
) -> list[list[str]]:
    """Best-effort decomposition of ``term_vars`` across margin bases."""
    n_marg = len(margin_raws)
    if n_marg == 1:
        return [list(term_vars)]
    if len(term_vars) == n_marg:
        return [[v] for v in term_vars]
    chunks: list[list[str]] = []
    extra = len(term_vars) - n_marg
    cursor = 0
    for j in range(n_marg):
        size = 1 + (extra if j == 0 else 0)
        chunks.append(term_vars[cursor : cursor + size])
        cursor += size
        extra = 0
    return chunks


def _term_constraint_T(term: _DiscreteTerm) -> np.ndarray | None:
    """The constraint matrix ``T`` (p_raw × p_post) with ``X_term_post =
    X_term_raw @ T``: the absorbed constraint applied to the identity, then
    the kept columns. The Python X'WX path applies it as ``T_i' B_raw T_j``
    where mgcv's XWXd0 applies a Householder constraint by ``Ztb``
    (discrete.c:2230-2266)."""
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


def _design_constraint_Ts(design: DiscreteDesign) -> list[np.ndarray | None]:
    """Per-term constraint matrices, cached on the design object."""
    cache = design._T_cache
    if cache is not None:
        return cache
    Ts = [_term_constraint_T(t) for t in design.terms]
    design._T_cache = Ts
    return Ts


def _term_Xb_raw(
    term: _DiscreteTerm, b_raw: np.ndarray, k: np.ndarray, n: int
) -> np.ndarray:
    """Compute ``X_term_raw @ b_raw`` (length ``n``) as mgcv's ``singleXb``
    / ``tensorXb`` (discrete.c:375-444) do: a singleton's marginal times
    ``b`` by BLAS ``dgemv('N')``, gathered at the first index column and the
    further index columns added in order."""
    from scipy.linalg.blas import dgemv

    b_raw = np.asarray(b_raw, dtype=float)
    if term.kind == "param":
        return np.asarray(
            dgemv(1.0, np.asfortranarray(term.Xd_list[0], dtype=float), b_raw)
        )

    if term.kind == "single":
        Xd = term.Xd_list[0]
        ks_lo, ks_hi = term.k_cols[0]
        tmp = np.asarray(dgemv(1.0, np.asfortranarray(Xd, dtype=float), b_raw))
        result = tmp[k[:, ks_lo]].copy()
        for q in range(ks_lo + 1, ks_hi):
            result += tmp[k[:, q]]
        return result

    return _tensor_Xb_raw(term, b_raw, k, n)


def _tensor_Xb_raw(
    term: _DiscreteTerm, b_raw: np.ndarray, k: np.ndarray, n: int
) -> np.ndarray:
    """``tensorXb`` (discrete.c:399-444) for an unconstrained raw ``b``:
    ``C = M B`` by BLAS ``dgemm`` with ``M`` the final marginal and
    ``vec(B) = b``, then per index set and column ``i`` of the row tensor
    of the other marginals (``tensorXj``'s products, in marginal order) the
    fused update ``f += C[k, i] * work``."""
    from scipy.linalg.blas import dgemm

    Xd_list = term.Xd_list
    d = len(Xd_list)
    ps = [Xd.shape[1] for Xd in Xd_list]
    pb = int(np.prod(ps[:-1]))
    M = np.asfortranarray(Xd_list[-1], dtype=float)
    pd = ps[-1]
    B = np.asfortranarray(np.asarray(b_raw, dtype=float).reshape((pd, pb), order="F"))
    C = np.asarray(dgemm(1.0, M, B))
    kd_lo = term.k_cols[d - 1][0]
    nq = term.k_cols[0][1] - term.k_cols[0][0]
    f = np.zeros(n)
    for q in range(nq):
        for i in range(pb):
            work = np.ones(n)
            jp, qq = i, pb
            for j in range(d - 1):
                qq //= ps[j]
                col, jp = divmod(jp, qq)
                work = work * Xd_list[j][k[:, term.k_cols[j][0] + q], col]
            f = _rfma_vec(C[k[:, kd_lo + q], i], work, f)
    return f


def _term_Xty_raw(
    term: _DiscreteTerm, wy: np.ndarray, k: np.ndarray, n: int
) -> np.ndarray:
    """Compute ``X_term_raw.T @ wy`` (length ``p_raw``).
    Direct port of ``singleXty`` / ``tensorXty`` (discrete.c:329-373): per
    index column, ``W y`` scattered into the marginal's rows in row order,
    then BLAS ``dgemv('T')`` against the marginal, the first overwriting
    and the rest accumulating (``beta = 1``), as XWyd's ``add`` does.
    """
    if term.kind == "param":
        # an un-discretised n-row block: the scatter is the identity
        return _single_xty(term.Xd_list[0], np.asarray(wy, dtype=float), None)

    if term.kind == "single":
        Xd = term.Xd_list[0]
        m = Xd.shape[0]
        ks_lo, ks_hi = term.k_cols[0]
        Xy = None
        for q in range(ks_lo, ks_hi):
            Xy = _single_xty(Xd, _bin_accum(k[:, q], wy, m), Xy)
        return Xy

    return _tensor_Xty_raw(term, wy, k, n)


def _single_xty(Xd: np.ndarray, temp: np.ndarray, Xy: np.ndarray | None) -> np.ndarray:
    """``singleXty``'s BLAS step: ``dgemv('T', m, p, 1, X, m, temp, 1, add,
    Xy, 1)`` - ``Xy`` overwritten when ``None``, else accumulated."""
    from scipy.linalg.blas import dgemv

    A = np.asfortranarray(Xd, dtype=float)
    if Xy is None:
        return np.asarray(dgemv(1.0, A, temp, trans=1))
    return np.asarray(dgemv(1.0, A, temp, beta=1.0, y=Xy, trans=1))


def _tensor_Xty_raw(
    term: _DiscreteTerm, wy: np.ndarray, k: np.ndarray, n: int
) -> np.ndarray:
    """``tensorXty`` (discrete.c:346-373) + its summation loop (XWyd :1164-1167)."""
    Xd_list = term.Xd_list
    d = len(Xd_list)
    M_final = Xd_list[-1]
    m_final, pd = M_final.shape
    pb = int(np.prod([Xd.shape[1] for Xd in Xd_list[:-1]]))
    n_sum = term.k_cols[0][1] - term.k_cols[0][0]
    ks_final = term.k_cols[d - 1][0]

    Xy = np.zeros(pb * pd, dtype=float)
    for q in range(n_sum):
        for i in range(pb):
            work = wy
            jp, div = i, pb
            for j in range(d - 1):
                div //= Xd_list[j].shape[1]
                col, jp = divmod(jp, div)
                work = work * Xd_list[j][k[:, term.k_cols[j][0] + q], col]
            temp = _bin_accum(k[:, ks_final + q], work, m_final)
            Xy[i * pd : (i + 1) * pd] = _single_xty(
                M_final, temp, None if q == 0 else Xy[i * pd : (i + 1) * pd]
            )
    return Xy


def _khatri_rao_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise tensor (Khatri-Rao) product: ``out[:, i*b+j] = A[:, i]·B[:, j]``.
    convention (first marginal slowest-varying — ``tensorXj``, discrete.c:301).
    """
    n, a = A.shape
    b = B.shape[1]
    return (A[:, :, None] * B[:, None, :]).reshape(n, a * b)


def _truncated_tensor(term: _DiscreteTerm, s: int, k: np.ndarray, n: int) -> np.ndarray:
    """Row-tensor product of a term's *non-final* marginals at summation index
    ``s`` — the ``dXi`` working columns of mgcv ``XWXijs`` (the matrix whose
    column ``r`` is extracted by ``tensorXj``, discrete.c:1754/1828).
    """
    d = len(term.Xd_list)
    if d == 1:
        return np.ones((n, 1), dtype=float)
    out = None
    for ell in range(d - 1):
        ks_l = term.k_cols[ell][0]
        G = term.Xd_list[ell][k[:, ks_l + s], :]  # (n, p_l)
        out = G if out is None else _khatri_rao_rows(out, G)
    return out


_XWX_DENSE_MSIZE_CAP = 4_000_000

_XWX_EMPTY_F64 = np.empty(0, dtype=float)


def _bin_accum(
    kcol: np.ndarray, v: np.ndarray, m: int, u: np.ndarray | None = None
) -> np.ndarray:
    """``wb[K[row]] += v[row]`` (times ``u[row]`` when given), the weighted bin
    accumulation mgcv writes as a plain loop (``wb[K[kk]] += w[kk]``,
    discrete.c:1786; ``singleXty``, :327).
    """
    if _rs_bin_accum is not None:
        return _rs_bin_accum(
            np.ascontiguousarray(kcol, dtype=np.int64),
            np.ascontiguousarray(v, dtype=float),
            _XWX_EMPTY_F64 if u is None else np.ascontiguousarray(u, dtype=float),
            m,
        )
    return np.bincount(kcol, weights=v if u is None else v * u, minlength=m)


def _wbar_contract(
    Ki_list: list[np.ndarray],
    Kj_list: list[np.ndarray],
    vals_list: list[np.ndarray],
    Xim: np.ndarray,
    Xjm: np.ndarray,
) -> np.ndarray:
    """Contract ``Xd_im' W̄ Xd_jm`` where ``W̄[a,b] = Σ vals·[K_i=a][K_j=b]``
    (summed over the summation-convention sets ``(K_i, K_j, vals)``).
    Three faithful paths from mgcv ``XWXijs`` (discrete.c:1793-2025), none
    """
    mim, pim = Xim.shape
    mjm, pjm = Xjm.shape
    msize = mim * mjm
    nst = len(Ki_list)  # summation-convention sets s_i·s_j (×3 for AR1)
    n = Ki_list[0].shape[0]
    if n > msize or (
        min(pim, pjm) > 15 and msize <= _XWX_DENSE_MSIZE_CAP and msize <= 16 * nst * n
    ):
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


def _marginal_support_bins(term: _DiscreteTerm, ell: int) -> np.ndarray:
    """Bins of ``term``'s marginal ``ell`` whose basis row is not all-zero."""
    Xd = term.Xd_list[ell]
    return np.flatnonzero(np.any(Xd != 0.0, axis=1))


def _terms_row_disjoint(ti: _DiscreteTerm, tj: _DiscreteTerm) -> bool:
    """``X_i' W X_j`` is structurally zero — for every weight — because no data
    row is in both terms' supports.
    lands on exact zeros the long way round (discrete.c:1924-2006 has no such
    """
    for ell_i in range(len(ti.Xd_list) - 1):
        a0, a1 = ti.k_cols[ell_i]
        if a1 - a0 != 1:
            continue
        for ell_j in range(len(tj.Xd_list) - 1):
            b0, b1 = tj.k_cols[ell_j]
            if b1 - b0 != 1 or a0 != b0:
                continue
            si = _marginal_support_bins(ti, ell_i)
            sj = _marginal_support_bins(tj, ell_j)
            if not np.intersect1d(si, sj, assume_unique=True).size:
                return True
    return False


def _param_smooth_block(
    pterm: _DiscreteTerm,
    sterm: _DiscreteTerm,
    w: np.ndarray,
    k: np.ndarray,
    n: int,
    w_off: np.ndarray | None = None,
) -> np.ndarray:
    """Raw block ``X_param' W X_smooth`` (shape ``p_par × pt_smooth``).
    accumulation when one term has ``m==n`` (discrete.c:1965-2006; the
    ``mim==n ⇒ rfac=0`` guard at 1811 guarantees no ``n×p`` product). Each
    parametric column ``a`` is fed as the "y" of ``X_smooth'(W·X_param[:,a])``,
    the same scatter as ``XWyd``; for the AR1 ``tri`` weight (``w_off`` given)
    ``W·X_param[:,a]`` is the tridiagonal matvec :func:`_tri_matvec`.
    """
    Xp = pterm.Xd_list[0]
    p_par = Xp.shape[1]
    WXp = (w[:, None] * Xp) if w_off is None else _tri_matvec(w, w_off, Xp)
    rows = [_term_Xty_raw(sterm, WXp[:, a], k, n) for a in range(p_par)]
    return np.vstack(rows)


def _smooth_smooth_block(
    ti: _DiscreteTerm,
    tj: _DiscreteTerm,
    w: np.ndarray,
    k: np.ndarray,
    n: int,
    w_off: np.ndarray | None = None,
) -> np.ndarray:
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
    ndi = (
        int(np.prod([Xd.shape[1] for Xd in ti.Xd_list[:-1]]))
        if len(ti.Xd_list) > 1
        else 1
    )
    ndj = (
        int(np.prod([Xd.shape[1] for Xd in tj.Xd_list[:-1]]))
        if len(tj.Xd_list) > 1
        else 1
    )

    diag_term = ti is tj  # same term ⇒ K_i and K_j are the same columns
    # mgcv's ``tensi``/``tensj`` (discrete.c:1797-1798): is the term a tensor at
    # all? A singleton's truncated row-tensor is identically 1, and mgcv keeps
    # four separate accumulation loops so the singleton case never touches it
    # (``Cq[*Kik] += *p0 * Xj[*Kjk]``, :1959).
    tens_i = len(ti.Xd_list) > 1
    tens_j = len(tj.Xd_list) > 1
    ar1 = w_off is not None
    use_rust = _rs_xwx_smooth_block is not None and not (
        diag_term and si == 1 and not ar1
    )
    # A singleton's truncated row-tensor is identically 1, so it is never
    # materialised — the rust kernel is told "no tensor" and drops the factor,
    # and the numpy paths below branch on ``tens_i``/``tens_j`` exactly as mgcv
    # branches on ``dt[i]>1`` (the diagonal shortcut, discrete.c:1783-1787) and
    # on ``tensi``/``tensj`` (the general case, :1927-1962). Building an
    # ``(n, 1)`` ones array and multiplying by it is pure loss.
    TTi = [_truncated_tensor(ti, s, k, n) for s in range(si)] if tens_i else []
    if diag_term:
        TTj = TTi
    elif tens_j:
        TTj = [_truncated_tensor(tj, t, k, n) for t in range(sj)]
    else:
        TTj = []

    # Rust runs the (r,c)×(s,t)×rows accumulation in one tight pass, the way
    # mgcv's C does (discrete.c:1924-2006). numpy cannot: `_wbar_contract`'s
    # factor path has to materialise an (n, p) gather `vals[:,None]*Xjm[Kj]`
    # and make p passes over n, where mgcv keeps one fused scalar loop with the
    # p-column accumulator L1-resident. So every block goes to rust when it is
    # built — including the plain single×single case, which an earlier revision
    # excluded on the belief that numpy already beat mgcv there (measured: it is
    # ~2× slower on the `s(x, by=fac)` blocks that dominate a by-factor fit).
    # The one exception stays in numpy: the non-AR1 `si==1` diagonal shortcut
    # below is a single bincount on a diagonal W̄ (mgcv's own simple branch,
    # :1742-1792) and has no per-row scatter to beat. It never applies under
    # AR1, where W̄ is tridiagonal, hence the `not ar1` guard.
    if use_rust:
        TTi3 = (
            np.ascontiguousarray(np.stack([t.T for t in TTi]))
            if tens_i
            else np.zeros((si, ndi, 0))
        )
        if diag_term:
            TTj3 = TTi3
        elif tens_j:
            TTj3 = np.ascontiguousarray(np.stack([t.T for t in TTj]))
        else:
            TTj3 = np.zeros((sj, ndj, 0))
        Ki, lo_i, hi_i = _term_k_rows_bounds(ti, k, ks_im, si)
        if diag_term:
            Kj, lo_j, hi_j = Ki, lo_i, hi_i
        else:
            Kj, lo_j, hi_j = _term_k_rows_bounds(tj, k, ks_jm, sj)
        bounds_ok = lo_i >= 0 and hi_i < mim and lo_j >= 0 and hi_j < mjm
        woff_arg = np.ascontiguousarray(w_off, dtype=float) if ar1 else _XWX_EMPTY_F64
        return _rs_xwx_smooth_block(
            np.ascontiguousarray(Xim),
            np.ascontiguousarray(Xjm),
            Ki,
            Kj,
            TTi3,
            TTj3,
            np.ascontiguousarray(w),
            woff_arg,
            diag_term,
            bounds_ok,
        )

    Ki_all = [k[:, ks_im + s].astype(np.int64) for s in range(si)]
    Kj_all = [k[:, ks_jm + t] for t in range(sj)]
    block = np.zeros((ndi * pim, ndj * pjm), dtype=float)
    for r in range(ndi):
        c0 = r if diag_term else 0  # symmetric term ⇒ only upper sub-blocks
        for c in range(c0, ndj):
            if diag_term and si == 1 and w_off is None:
                # Same final marginal ⇒ K_i ≡ K_j ⇒ W̄ is diagonal; skip the
                # m_im×m_jm table (mgcv simple branch, discrete.c:1742-1792).
                # (The AR1 ``tri`` super/sub couplings break this — w_off path
                # routes through the general three-scatter contraction.)
                # mgcv splits tensor from singleton here (discrete.c:1783-1787):
                # a singleton's dXi·dXj is identically 1, so the weight is `w`.
                if tens_i:
                    wb = _bin_accum(k[:, ks_im], w * TTi[0][:, r], mim, u=TTj[0][:, c])
                else:
                    wb = _bin_accum(k[:, ks_im], w, mim)
                sub = (Xim * wb[:, None]).T @ Xjm
            else:
                Ki_list = []
                Kj_list = []
                vals_list = []
                for s in range(si):
                    Ki = Ki_all[s]
                    dXi = TTi[s][:, r] if tens_i else None
                    for t in range(sj):
                        Kj = Kj_all[t]
                        dXj = TTj[t][:, c] if tens_j else None
                        Ki_list.append(Ki)
                        Kj_list.append(Kj)
                        if dXi is None and dXj is None:
                            vals_list.append(w)
                        elif dXj is None:
                            vals_list.append(w * dXi)
                        elif dXi is None:
                            vals_list.append(w * dXj)
                        else:
                            vals_list.append(w * dXi * dXj)
                        if w_off is not None:
                            Ki_list.append(Ki[:-1])
                            Kj_list.append(Kj[1:])
                            v = w_off if dXi is None else w_off * dXi[:-1]
                            vals_list.append(v if dXj is None else v * dXj[1:])
                            Ki_list.append(Ki[1:])
                            Kj_list.append(Kj[:-1])
                            v = w_off if dXi is None else w_off * dXi[1:]
                            vals_list.append(v if dXj is None else v * dXj[:-1])
                sub = _wbar_contract(Ki_list, Kj_list, vals_list, Xim, Xjm)
            block[r * pim : (r + 1) * pim, c * pjm : (c + 1) * pjm] = sub
            if diag_term and c > r:
                block[c * pim : (c + 1) * pim, r * pjm : (r + 1) * pjm] = sub.T
    return block


def _term_k_rows_bounds(
    term, k: np.ndarray, ks: int, s_count: int
) -> tuple[np.ndarray, int, int]:
    """The term's final-marginal index columns as a contiguous ``(s_count, n)``
    int64 block plus that block's ``(min, max)``, cached on the term.
    """
    cache = getattr(term, "_k_rows_cache", None)
    key = (id(k), ks, s_count)
    if cache is not None and cache[0] == key:
        return cache[1], cache[2], cache[3]
    rows = np.ascontiguousarray(k[:, ks : ks + s_count].T, dtype=np.int64)
    lo = int(rows.min()) if rows.size else 0
    hi = int(rows.max()) if rows.size else -1
    term._k_rows_cache = (key, rows, lo, hi)
    return rows, lo, hi


def _tri_matvec(w: np.ndarray, w_off: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``W_eff · V`` for the symmetric tridiagonal ``W_eff`` (diagonal ``w``,
    off-diagonal ``w_off``), ``V`` an ``n`` vector or ``n×c`` matrix.
    """
    col = V.ndim == 2
    wb = w[:, None] if col else w
    wo = w_off[:, None] if col else w_off
    out = wb * V
    out[:-1] += wo * V[1:]  # super: w_off[i]·V[i+1]
    out[1:] += wo * V[:-1]  # sub:   w_off[i-1]·V[i-1]
    return out


def _term_pair_XWX_raw(
    ti: _DiscreteTerm,
    tj: _DiscreteTerm,
    w: np.ndarray,
    k: np.ndarray,
    n: int,
    w_off: np.ndarray | None = None,
) -> np.ndarray:
    """Raw (pre-constraint) cross-product block ``X_i' W X_j`` for one term
    pair, dispatching on parametric (``m==n``) vs smooth (compressed) terms —
    mgcv ``XWXijs`` (discrete.c:1672).
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
    if ti is not tj and _terms_row_disjoint(ti, tj):
        return np.zeros((_term_ncol(ti), _term_ncol(tj)), dtype=float)
    return _smooth_smooth_block(ti, tj, w, k, n, w_off=w_off)


def _term_ncol(term: _DiscreteTerm) -> int:
    """Raw (pre-constraint) column count of a term: ``Π_ell p_ell``."""
    return int(np.prod([Xd.shape[1] for Xd in term.Xd_list]))


def Xbd(
    design: DiscreteDesign, beta: np.ndarray, lt: Sequence[int] | None = None
) -> np.ndarray:
    """Compute ``X β`` on the compressed design — per-term scatter-add only.

    Direct port of mgcv ``Xbd`` (src/discrete.c:502-572 + the R wrapper
    misc.r:385-436): lift each term's post-constraint β to raw space
    (``β_raw = T·β_post``) and accumulate ``_term_Xb_raw`` into the
    n-vector η. The ``n×p`` design is never formed (mgcv ``discrete=TRUE``
    scatter-adds on the compressed Xd/k grid).

    ``lt`` selects a term subset (mgcv's ``lt``, misc.r:388): only the
    listed terms contribute — the per-linear-predictor η of a general
    family (``lt = lpid[j]``, gamlss.r:936-938). ``beta`` stays FULL-p:
    mgcv reorders/truncates it to the selected terms via ``lpip``
    (misc.r:403-409); hea's terms carry their own ``coef_slice``, so the
    same rows are read in place. A matrix ``beta`` (p × bc) maps to an
    ``(n, bc)`` result, one column at a time (mgcv's ``bc`` loop,
    misc.r:402/430). With a ``drop`` the coefficients are first spread
    over the undropped vector, zeros at the dropped columns (misc.r:395).
    """
    beta = np.asarray(beta, dtype=float)
    if design.drop is not None:
        beta = _undrop(beta, design)
    return _xbd_undropped(design, beta, lt)


def _xbd_undropped(
    design: DiscreteDesign, beta: np.ndarray, lt: Sequence[int] | None
) -> np.ndarray:
    """:func:`Xbd` on a coefficient vector (or matrix) laid out as the
    terms' ``coef_slice``, i.e. before any ``drop``."""
    if beta.ndim == 2:
        return np.column_stack(
            [_xbd_undropped(design, beta[:, j], lt) for j in range(beta.shape[1])]
        )
    n = design.n
    eta = np.zeros(n, dtype=float)
    Ts = _design_constraint_Ts(design)
    sel = range(len(design.terms)) if lt is None else lt
    for it in sel:
        term = design.terms[it]
        T = Ts[it]
        b_post = beta[term.coef_slice]
        if term.qc == 1 and term.v is not None:
            b_raw = _zb_hh(term.v, b_post)  # tensorXb's Zb
        elif term.qc == -1 and term.v is not None:
            b_raw = _zb_kr(term.v, b_post, _term_p_raw(term))
        else:
            b_raw = b_post if T is None else (T @ b_post)
        eta += _term_Xb_raw(term, b_raw, design.k, n)
    return eta


_xwxd0_rs = rs_fn("xwxd0")


def _zb_hh(v: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """mgcv's ``Zb`` for a Householder constraint (discrete.c:188, ``qc >
    0``): ``(I - vv') (0, b0')'`` as mgcv.so computes it. The inner product
    ``v'(0, b0')'`` runs over ``p - 1`` terms in order; for ``p >= 9`` clang
    vectorizes the loop (it also copies ``b0``), so the first ``8 *
    floor((p - 1) / 8)`` products are rounded before each in-order add and
    only the remaining ones are fused. Every update is fused."""
    p = v.size
    b1 = np.empty(p)
    b1[0] = 0.0
    b1[1:] = b0
    x = 0.0
    n_vec = (p - 1) // 8 * 8 if p >= 9 else 0
    for i in range(1, 1 + n_vec):
        x = x + b1[i] * v[i]
    for i in range(1 + n_vec, p):
        x = _rfma(b1[i], v[i], x)
    for i in range(p):
        b1[i] = _rfma(-v[i], x, b1[i])
    return b1


def _term_p_raw(term) -> int:
    """A term's unconstrained coefficient count: the product of its
    marginals' column counts."""
    return _term_p_raw_list(term.Xd_list)


def _term_p_raw_list(Xd_list) -> int:
    """The product of the marginals' column counts."""
    return int(np.prod([np.asarray(Xd).shape[1] for Xd in Xd_list]))


def _p_undropped(design: DiscreteDesign) -> int:
    """The coefficient count before ``drop``: the length the terms'
    ``coef_slice`` index."""
    return design.p + (0 if design.drop is None else int(design.drop.size))


def _undrop(beta: np.ndarray, design: DiscreteDesign) -> np.ndarray:
    """``beta`` (rows = the fit's coefficients) with zeros inserted at the
    dropped positions: misc.r:395-399, ``b[-drop] <- beta``."""
    keep = np.ones(_p_undropped(design), dtype=bool)
    keep[design.drop] = False
    out = np.zeros((keep.size, *beta.shape[1:]), dtype=float)
    out[keep] = beta
    return out


def _zb_kr(v: np.ndarray, b0: np.ndarray, p: int) -> np.ndarray:
    """mgcv's ``Zb`` for a Kronecker product of sum-to-zero contrasts and an
    identity (discrete.c:188, ``qc < 0``): ``v = (M, d_1..d_M, nc)``; each
    pass appends minus the sum of each ``d_k - 1`` block."""
    M = round(v[0])
    pp, k0 = p, 1
    for i in range(M):
        mk = round(v[i + 1])
        k0 *= mk - 1
        pp //= mk
    k0 *= pp
    src = np.asarray(b0, dtype=float)
    for k in range(M + 1):
        mk = round(v[k + 1]) - 1 if k < M else pp
        p0m = k0 // mk
        out = []
        for i in range(p0m):
            x = 0.0
            for j in range(mk):
                z = float(src[j * p0m + i])
                x += z
                out.append(z)
            if k < M:
                out.append(-x)
        if k < M:
            k0 += p0m
        src = np.array(out)
    return src


def _ztb_kr(v: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """mgcv's ``Ztb`` for the Kronecker sum-to-zero constraint
    (discrete.c:241, ``qc < 0``): each pass subtracts the last level's
    block from the others and drops it."""
    p = np.asarray(b0).size
    w0 = np.asarray(b0, dtype=float).copy()
    M = round(v[0])
    pp = p
    for i in range(M):
        pp //= round(v[i + 1])
    k1 = p
    for k in range(M + 1):
        if k < M:
            mk = round(v[k + 1])
            mk1 = mk - 1
        else:
            mk1 = mk = pp
        p1m = k1 // mk
        out = []
        for i in range(p1m):
            z = float(w0[i + (mk - 1) * p1m]) if k < M else 0.0
            for j in range(mk1):
                out.append(float(w0[i + j * p1m]) - z)
        if k < M:
            k1 -= p1m
        w0 = np.array(out)
    return w0


def _ztb_hh(v: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """mgcv's ``Ztb`` for a Householder constraint (discrete.c:241, ``qc >
    0``): ``(I - vv') b0`` with the first element dropped, fused as compiled."""
    p = v.size
    x = 0.0
    for i in range(p):
        x = _rfma(b0[i], v[i], x)
    return np.array([_rfma(-v[i], x, b0[i]) for i in range(1, p)])


def _xwxd0_mgcv(
    design: DiscreteDesign, w: np.ndarray, ar_weights: np.ndarray | None
) -> np.ndarray | None:
    """``X'WX`` by mgcv's ``XWXd0`` as compiled (the extension's port, or
    its Python twin :func:`hea.models._xwxd.xwxd0` without it), on the
    design laid out as mgcv lays ``G$Xd``/``G$kd``/``G$ks`` out (cached on
    the design, which the PIRLS iterations share). ``None`` when the design
    carries something that kernel does not take: a constraint still held as
    a dense post-hoc map (``absorb``/``keep_cols``)."""
    if design._mgcv_cache is None:
        design._mgcv_cache = _mgcv_layout(design)
    lay = design._mgcv_cache
    if lay is False:
        return None
    arw = (
        np.zeros(0)
        if ar_weights is None
        else np.ascontiguousarray(ar_weights, dtype=float)
    )
    xwxd0 = _xwxd0_rs if _xwxd0_rs is not None else _xwxd_py.xwxd0
    out, ptot = xwxd0(
        lay["x"],
        np.ascontiguousarray(w, dtype=float),
        lay["k"],
        lay["ks"],
        lay["m"],
        lay["p"],
        lay["ts"],
        lay["dt"],
        lay["v"],
        lay["qc"],
        arw,
    )
    return np.asarray(out).reshape((ptot, ptot), order="F")


_xwxd1_rs = rs_fn("xwxd1")


def _xwxd1_mgcv(
    design: DiscreteDesign,
    w: np.ndarray,
    ar_weights: np.ndarray | None,
    lt: Sequence[int] | None,
    rt: Sequence[int] | None,
) -> np.ndarray | None:
    """The ``X'WX`` blocks of the left terms ``lt`` against the right terms
    ``rt`` by mgcv's ``XWXd1`` as compiled (the extension's port, or its
    Python twin :func:`hea.models._xwxd.xwxd1`), as ``XWXd(..., lt, rt)``
    calls it: one side given selects it for both and the result is
    symmetric. ``None`` as for :func:`_xwxd0_mgcv`."""
    if design._mgcv_cache is None:
        design._mgcv_cache = _mgcv_layout(design)
    lay = design._mgcv_cache
    if lay is False:
        return None
    arw = (
        np.zeros(0)
        if ar_weights is None
        else np.ascontiguousarray(ar_weights, dtype=float)
    )
    xwxd1 = _xwxd1_rs if _xwxd1_rs is not None else _xwxd_py.xwxd1
    out, ptot, ctot = xwxd1(
        lay["x"],
        np.ascontiguousarray(w, dtype=float),
        lay["k"],
        lay["ks"],
        lay["m"],
        lay["p"],
        lay["ts"],
        lay["dt"],
        lay["v"],
        lay["qc"],
        arw,
        [] if lt is None else [int(i) for i in lt],
        [] if rt is None else [int(i) for i in rt],
    )
    return np.asarray(out).reshape((ptot, ctot), order="F")


def _lt_rt_drop(out: np.ndarray, design: DiscreteDesign, lt) -> np.ndarray:
    """mgcv's ``drop`` on an ``XWXd(..., lt, rt)`` block (misc.r:301-306):
    both drop sets are taken from the left terms' coefficients and, since
    ``length(ldrop==0)`` is the length of ``ldrop``, only the columns are
    removed."""
    if design.drop is None:
        return out
    terms = design.terms
    lpi = np.concatenate(
        [np.arange(terms[i].coef_slice.start, terms[i].coef_slice.stop) for i in lt]
    )
    rdrop = np.flatnonzero(np.isin(lpi, design.drop))
    if rdrop.size:
        out = np.delete(out, rdrop, axis=1)
    return out


def _mgcv_layout(design: DiscreteDesign) -> dict | bool:
    """The design as mgcv lays ``G$Xd``/``G$kd``/``G$ks`` out for the C
    kernels: the marginals concatenated column-major, the index matrix (an
    identity column for an un-discretised n-row block) as int32 column-major,
    each marginal's index columns, sizes, and each term's first marginal and
    count. ``False`` when a term still carries its constraint as a dense
    post-hoc map."""
    n = design.n
    xs: list[np.ndarray] = []
    m: list[int] = []
    pp: list[int] = []
    ts: list[int] = []
    dt: list[int] = []
    ks: list[tuple[int, int]] = []
    kcols = [np.asarray(design.k, dtype=np.int32)]
    nk = kcols[0].shape[1] if kcols[0].ndim == 2 else 0
    vs: list[np.ndarray] = []
    qcs: list[int] = []
    for term in design.terms:
        hh = term.qc in (1, -1) and term.v is not None
        if not hh and (
            term.absorb is not None or term.keep_cols is not None or term.qc != 0
        ):
            return False
        if hh:
            vs.append(np.asarray(term.v, dtype=float))
        qcs.append(term.qc if hh else 0)
        ts.append(len(m))
        dt.append(len(term.Xd_list))
        for Xd, (a0, b0) in zip(term.Xd_list, term.k_cols):
            Xd = np.asarray(Xd, dtype=float)
            xs.append(np.ravel(Xd, order="F"))
            m.append(Xd.shape[0])
            pp.append(Xd.shape[1])
            if a0 < 0:
                kcols.append(np.arange(n, dtype=np.int32).reshape(n, 1))
                ks.append((nk, nk + 1))
                nk += 1
            else:
                ks.append((int(a0), int(b0)))
    k = np.concatenate(kcols, axis=1) if len(kcols) > 1 else kcols[0]
    return {
        "x": np.concatenate(xs),
        "k": np.ravel(k, order="F").astype(np.int32),
        "ks": np.ravel(np.asarray(ks, dtype=np.int32), order="F"),
        "m": np.asarray(m, dtype=np.int32),
        "p": np.asarray(pp, dtype=np.int32),
        "ts": np.asarray(ts, dtype=np.int32),
        "dt": np.asarray(dt, dtype=np.int32),
        "v": np.concatenate(vs) if vs else np.zeros(0),
        "qc": np.asarray(qcs, dtype=np.int32),
    }


def XWXd(
    design: DiscreteDesign,
    w: np.ndarray,
    ar_weights: np.ndarray | None = None,
    lt: Sequence[int] | None = None,
    rt: Sequence[int] | None = None,
) -> np.ndarray:
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

    ``lt``/``rt`` select left/right term subsets (mgcv's ``lt``/``rt``,
    misc.r:260-327, the C ``XWXd1``): the result is then ONLY the selected
    block — shape (Σ selected-left coef widths) × (Σ right widths), rows/cols
    in selection order — the cross-LP Hessian block ``X_i' diag(l2) X_j`` of
    a general family (``lt = lpid[i]``, ``rt = lpid[j]``, gamlss.r:656).
    ``lt`` alone selects its terms against themselves; without ``lt`` the
    whole ``X'WX`` is returned whatever ``rt`` is, as mgcv's dense branch
    tests ``is.null(lt)`` twice.
    """
    w = np.asarray(w, dtype=float)
    if lt is None:
        # misc.r:260 tests `is.null(lt)` twice: without lt the whole X'WX
        rt = None
        full = _xwxd0_mgcv(design, w, ar_weights)
        if full is not None:
            return _drop_rows_cols(full, design.drop)
    else:
        blk = _xwxd1_mgcv(design, w, ar_weights, lt, rt)
        if blk is not None:
            return _lt_rt_drop(blk, design, lt)
    n = design.n
    p = _p_undropped(design)
    if ar_weights is None:
        w_off = None
    else:
        w, w_off = _ar1_tri_weight(w, ar_weights)
    Ts = _design_constraint_Ts(design)
    terms = design.terms
    if lt is not None:
        # lt alone selects its terms against themselves
        if rt is None:
            rt = lt
        lt = list(lt)
        rt = list(rt)

        def _width(idx: int) -> int:
            csl = terms[idx].coef_slice
            return csl.stop - csl.start

        r_off = np.cumsum([0] + [_width(i) for i in lt])
        c_off = np.cumsum([0] + [_width(j) for j in rt])
        out = np.zeros((int(r_off[-1]), int(c_off[-1])), dtype=float)
        same = lt == rt
        for a, ia in enumerate(lt):
            Ti = Ts[ia]
            for b, ib in enumerate(rt):
                if same and b < a:
                    out[r_off[a] : r_off[a + 1], c_off[b] : c_off[b + 1]] = out[
                        r_off[b] : r_off[b + 1], c_off[a] : c_off[a + 1]
                    ].T
                    continue
                raw = _term_pair_XWX_raw(
                    terms[ia], terms[ib], w, design.k, n, w_off=w_off
                )
                blk = raw if Ti is None else (Ti.T @ raw)
                Tj = Ts[ib]
                if Tj is not None:
                    blk = blk @ Tj
                out[r_off[a] : r_off[a + 1], c_off[b] : c_off[b + 1]] = blk
        return _lt_rt_drop(out, design, lt)
    nt = len(terms)
    XWX = np.zeros((p, p), dtype=float)
    for i in range(nt):
        sl_i = terms[i].coef_slice
        Ti = Ts[i]
        for j in range(i, nt):
            raw = _term_pair_XWX_raw(terms[i], terms[j], w, design.k, n, w_off=w_off)
            Tj = Ts[j]
            blk = raw if Ti is None else (Ti.T @ raw)
            if Tj is not None:
                blk = blk @ Tj
            sl_j = terms[j].coef_slice
            XWX[sl_i, sl_j] = blk
            if i != j:
                XWX[sl_j, sl_i] = blk.T
    return _drop_rows_cols(XWX, design.drop)


def _drop_rows_cols(A: np.ndarray, drop: np.ndarray | None) -> np.ndarray:
    """``A[-drop, -drop]``."""
    if drop is None:
        return A
    return np.delete(np.delete(A, drop, axis=0), drop, axis=1)


def XWyd(
    design: DiscreteDesign,
    w: np.ndarray,
    y: np.ndarray,
    ar: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    lt: Sequence[int] | None = None,
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

    ``lt`` selects a term subset (mgcv's ``lt``, misc.r:335-349): the
    returned vector holds ONLY the selected terms' rows, in selection
    order — the per-LP gradient block of a general family
    (``lt = lpid[i]``, gamlss.r:625).
    """
    w_arr = np.asarray(w, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = design.n
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
    Ts = _design_constraint_Ts(design)
    if lt is not None:
        parts = []
        for it in lt:
            term = design.terms[it]
            T = Ts[it]
            Xty_raw = _term_Xty_raw(term, wy, design.k, n)
            if term.qc == 1 and term.v is not None:
                parts.append(_ztb_hh(term.v, Xty_raw))
            elif term.qc == -1 and term.v is not None:
                parts.append(_ztb_kr(term.v, Xty_raw))
            else:
                parts.append(Xty_raw if T is None else T.T @ Xty_raw)
        Xy = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
        if design.drop is not None:
            # misc.r:347: drop rebased onto the selected coefficients
            lpi = np.concatenate(
                [
                    np.arange(
                        design.terms[i].coef_slice.start,
                        design.terms[i].coef_slice.stop,
                    )
                    for i in lt
                ]
            )
            Xy = np.delete(Xy, np.flatnonzero(np.isin(lpi, design.drop)))
        return Xy
    Xy = np.zeros(_p_undropped(design), dtype=float)
    for term, T in zip(design.terms, Ts):
        Xty_raw = _term_Xty_raw(term, wy, design.k, n)
        if term.qc == 1 and term.v is not None:
            Xy[term.coef_slice] = _ztb_hh(term.v, Xty_raw)  # XWyd's Ztb
        elif term.qc == -1 and term.v is not None:
            Xy[term.coef_slice] = _ztb_kr(term.v, Xty_raw)
        elif T is None:
            Xy[term.coef_slice] = Xty_raw
        else:
            Xy[term.coef_slice] = T.T @ Xty_raw
    return Xy if design.drop is None else np.delete(Xy, design.drop)


def design_full_X(design: DiscreteDesign) -> np.ndarray:
    """Materialise the dense ``n × p`` design — column k is
    ``Xbd(design, e_k)``, the de-materialisation identity mgcv's
    ``diagXVXt`` uses per column (discrete.c:743-749). Reference /
    test comparator only (the fit path never forms it); cached on
    the design (weight- and coef-independent)."""
    if design._full_X_cache is not None:
        return design._full_X_cache
    p = design.p
    e = np.zeros(p)
    cols = []
    for k_ in range(p):
        e[k_] = 1.0
        cols.append(Xbd(design, e))
        e[k_] = 0.0
    X = np.column_stack(cols)
    design._full_X_cache = X
    return X


def diagXVXd(
    design: DiscreteDesign, V: np.ndarray, lt: Sequence[int] | None = None
) -> np.ndarray:
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

    ``lt`` selects a term subset (mgcv's ``lt``, misc.r:438 →
    ``workXVXd``): the result is ``diag(X_lt V X_lt')`` where, per mgcv's
    contract ("V is re-ordered and truncated to relate only to the
    selected terms, in the order they are selected", misc.r:481-486),
    ``V`` must arrive as the (Σ selected widths)² block in SELECTION
    order — the same convention as :func:`XWXd`'s ``lt``/``rt``. The
    general-family prediction SE uses ``lt = lpid[j]`` with
    ``V = Vp[lpi[j], lpi[j]]``. mgcv's asymmetric ``rt`` (scattered
    ``ijXVXd`` elements) has no hea caller and is not ported.

    ``type='terms'``/``iterms`` per-term selection (mgcv ``rs``/``cs``)
    is not wired — those callers stay on the exact-eval parent path.
    """
    n, p = design.n, design.p
    V = np.asarray(V, dtype=float)
    diag = np.zeros(n, dtype=float)
    if lt is not None:
        cols = np.concatenate(
            [
                np.arange(
                    design.terms[it].coef_slice.start, design.terms[it].coef_slice.stop
                )
                for it in lt
            ]
        )
        if V.shape != (cols.size, cols.size):
            raise ValueError(
                f"V must be ({cols.size}, {cols.size}) — the selected "
                f"block in selection order; got {V.shape}"
            )
        e = np.zeros(_p_undropped(design), dtype=float)
        vfull = np.zeros(e.size, dtype=float)
        for kk in range(cols.size):
            vfull[cols] = V[:, kk]
            xv = _xbd_undropped(design, vfull, lt)  # (X_lt·V)[:, kk]
            e[cols[kk]] = 1.0
            xi = _xbd_undropped(design, e, lt)  # X_lt[:, kk]
            e[cols[kk]] = 0.0
            diag += xv * xi
        return diag
    e = np.zeros(p, dtype=float)
    for kk in range(p):
        xv = Xbd(design, V[:, kk])  # (X·V)[:, kk]
        e[kk] = 1.0
        xi = Xbd(design, e)  # X[:, kk]
        e[kk] = 0.0
        diag += xv * xi
    return diag
