"""Generalized mixed models (``gmm``) — lme4's ``lmer`` + ``glmer`` under one class.

Gaussian-identity fits take the LMM path (``lmer``: profiled deviance, REML/ML);
any other family takes the GLMM path (``glmer``: Laplace approximation). The
public entry dispatches on ``family`` internally — the ``gmm`` name reflects
that it spans both, and is the long-term home for general families too.

Built on hea.formula's ``parse → expand → materialize / materialize_bars``
pipeline. The fixed-effect side comes from ``materialize`` (R-canonical
column names). The random-effect side comes from ``materialize_bars``,
which returns ``Z``, an integer ``Λᵀ`` template, and an initial ``θ``.

We optimize the ML or REML profiled deviance over ``θ`` using L-BFGS-B
(diagonal entries of ``Λ`` constrained to be ≥ 0 for identifiability),
then recover ``β̂``, ``σ̂``, ``SE(β̂)``, and the per-bar variance components
at the optimum.

References
----------
Bates, Mächler, Bolker, Walker (2015), "Fitting Linear Mixed-Effects
Models Using lme4", J. Stat. Software 67(1), §5 ("Profiled Deviance").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import warnings
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.linalg import qr as _scipy_qr, solve_triangular
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csc_array, eye_array
from scipy.special import digamma, polygamma, roots_hermite

from .. import family as _family_mod
from ..family import Family, Gaussian, _coerce_response
from ..formula import (
    BinOp,
    CONTRAST_FN_NAMES,
    ExpandedFormula,
    ReTerms,
    _bar_lhs_to_ef,
    _eval_atom,
    _eval_group,
    _flatten_nested_group,
    _materialize_re_lhs,
    materialize,
    materialize_bars,
    prepare_design,
)
from .lm import _apply_subset, _label_top_n, _lowess, _qq_plot
from ..utils import (
    format_df, format_pval, format_signif, format_signif_jointly,
    significance_code,
)

__all__ = ["gmm", "Profile"]


# ---------------------------------------------------------------------------
# CHOLMOD compatibility shim.
#
# Routes through ``sksparse.cholmod`` when scikit-sparse is installed (the
# fast path used here for the inner Cholesky of ``M = Λ Zᵀ Z Λᵀ + I``).
# Falls back to ``scipy.sparse.linalg.splu`` otherwise — slower than
# CHOLMOD because SuperLU re-runs symbolic analysis on every refactor and
# doesn't exploit symmetry, but it preserves sparsity. A one-time
# ``UserWarning`` points users at ``hea[fast]``.
#
# Both backends expose the slice of the API the rest of this module uses:
#   * ``factorize(M)`` — refactor with new numeric values
#   * ``solve(b)`` — solve ``M⁻¹ b``
#   * ``half_log_det()`` — ``½·log|det M|``
#   * ``L`` — sparse Cholesky factor. sksparse returns it directly; the
#     splu fallback computes via dense Cholesky on first access.
# ---------------------------------------------------------------------------

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
    ``RuntimeError`` / non-positive-diagonal check (fallback).
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
        # Cholesky's L isn't directly available from SuperLU. Compute via
        # dense Cholesky on first access — only touched once per fit
        # (snapshot stored on the result), so this is cold-path.
        if self._L_cache is None:
            from scipy.linalg import cholesky as _scipy_cholesky

            M_dense = self._M.toarray()
            L_dense = _scipy_cholesky(M_dense, lower=True)
            self._L_cache = csc_array(L_dense)
        return self._L_cache


_SKSPARSE_WARNED = False


def _warn_no_sksparse_once() -> None:
    global _SKSPARSE_WARNED
    if _SKSPARSE_WARNED:
        return
    warnings.warn(
        "scikit-sparse is not installed; hea.gmm is using a "
        "scipy.sparse.linalg.splu fallback. This is functional but slower "
        "than CHOLMOD for large mixed-effect models (no symbolic-analysis "
        "reuse across deviance evaluations). Install SuiteSparse "
        "(e.g. `apt install libsuitesparse-dev` or `brew install suite-sparse`) "
        "and `pip install scikit-sparse` (or `pip install hea[fast]`) for "
        "the fast path.",
        UserWarning,
        stacklevel=3,
    )
    _SKSPARSE_WARNED = True


def cho_factor(M):
    if _HAS_SKSPARSE:
        return _SksparseFactor(M)
    _warn_no_sksparse_once()
    return _SpluFactor(M)


@dataclass(slots=True)
class _FitInputs:
    """Pre-assembled inputs for :meth:`gmm._fit_from_components`.

    Built by the public formula-based ``gmm()`` constructor, or assembled
    directly by callers that bypass formula parsing (e.g. ``hea.gamm``,
    which composes ``smooth2random`` outputs into a unified design).

    Field naming follows hea conventions: matrix/vector symbols (``X``,
    ``y``, ``Z``, ``theta``) stay as their math names; longer-lived state
    uses snake_case. ``re_terms`` holds the full :class:`ReTerms` from
    :func:`materialize_bars` (carries ``Z``, ``Lambdat`` template, initial
    ``theta``, ``cnms``, ``flist_levels``, ``Gp``).
    """

    # Design pieces -----------------------------------------------------
    X_df: pl.DataFrame
    """Fixed-effects design matrix, columns named by formula expansion."""

    y: np.ndarray
    """Response on the response scale (i.e. before offset subtraction)."""

    re_terms: ReTerms
    """Random-effects structure from :func:`materialize_bars`."""

    offset: np.ndarray
    """Per-row offset; zeros if none specified."""

    # Inference mode ----------------------------------------------------
    family: Family
    """GLM family. Gaussian-identity is the current implemented path;
    other families raise :class:`NotImplementedError` until the GLMM
    Laplace path lands."""

    reml: bool
    """``True`` for REML, ``False`` for ML."""

    # Optional inputs ---------------------------------------------------
    weights: Optional[np.ndarray] = None
    """Prior weights (``None`` ≡ unit weights)."""

    mustart: Optional[np.ndarray] = None
    """Starting μ for GLMM PIRLS."""

    etastart: Optional[np.ndarray] = None
    """Starting η for GLMM PIRLS."""

    start: Optional[dict] = None
    """User-supplied starting values for the GLMM outer optimizer. Accepts
    ``None`` (use defaults: θ from ``re_terms.theta``, β from Stage 0's
    converged ``pp.delb``), a numpy array (interpreted as ``θ`` only), or a
    dict with keys ``"theta"``/``"par"`` and ``"beta"``/``"fixef"``. Mirrors
    lme4's ``getStart`` (modular.R:472-533)."""

    nagq0_init_step: bool = True
    """When True (default), run a Stage 0 (θ-only) optimization before the
    full Stage 1 (θ+β) one to warm-start the latter. When False, skip
    Stage 0 and run Stage 1 directly from ``θ₀`` and ``β=0``. Mirrors
    ``glmerControl(nAGQ0initStep=...)``."""

    # Argument plumbing -------------------------------------------------
    nAGQ: int = 1
    """Number of adaptive Gauss-Hermite quadrature points per group. Default
    1 ≡ Laplace approximation. ``0`` skips Stage 1 (LMM-style θ-only fit);
    ``>1`` is reserved for AGQ and currently raises."""

    tol_pwrss: float = 1e-7
    """PIRLS convergence tolerance — ``glmerControl(tolPwrss=)``."""

    maxit_pwrss: int = 100
    """PIRLS iteration cap. Mirrors lme4's ``mkGlmerDevfun(maxit=)`` /
    ``mkdevfun(maxit=)`` default of ``100L`` (lmer.R:308, modular.R:798).
    This is **not** a ``glmerControl()`` argument in lme4 (the control
    constructor has no ``maxit``); it lives on the modular devfun interface,
    so ``gmm`` likewise does not expose it through ``control=``."""

    calc_derivs: bool = True
    """When True (default), compute the numerical gradient + Hessian of the
    Stage 1 deviance at the optimum and store on ``m.optinfo$derivs``.
    Mirrors ``glmerControl(calc.derivs=)``."""

    use_last_params: bool = False
    """When True, do NOT restore (β, u) to ``opt$par`` after the Hessian
    pass — leaves the model at whatever state ``deriv12`` happened to
    finish at. Mirrors ``glmerControl(use.last.params=)``."""

    verbose: int = 0
    """Integer verbosity level. ``>0`` enables Nelder-Mead progress prints;
    ``>2`` enables PIRLS iteration prints."""

    opt_ctrl: Optional[dict] = None
    """Optimizer-specific control options. Nelder_Mead keys are recognised
    with lme4's R-flavoured names (``maxfun``, ``XtolRel``, ``FtolAbs``,
    ``FtolRel``, ``MinfMax``, ``verbose``). Mirrors ``glmerControl(optCtrl=)``."""

    optimizer: object = ("bobyqa", "Nelder_Mead")
    """Outer optimizer. **glmer** (Laplace path): a two-stage chain —
    ``glmerControl(optimizer=)`` — where ``optimizer[0]`` runs Stage 0 (θ-only)
    and ``optimizer[1]`` runs Stage 1 (θ+β); each is ``"bobyqa"`` or
    ``"Nelder_Mead"`` (both ported line-for-line from minqa / lme4's
    ``optimizer.cpp``); lme4's glmer default is ``c("bobyqa", "Nelder_Mead")``.
    **lmer** (Gaussian-identity LMM path): a single optimizer *name* string —
    ``lmerControl(optimizer=)`` — one of ``"nloptwrap"`` (lme4's default; NLopt
    LN_BOBYQA), ``"bobyqa"``, or ``"Nelder_Mead"``. A non-string here (the tuple
    default, e.g. a direct construction) is read as the lmer default."""

    check_conv_grad: Optional[dict] = None
    """``glmerControl(check.conv.grad=)`` — ``{action, tol, relTol}`` for the
    post-fit scaled-gradient convergence diagnostic (8.14)."""

    check_conv_hess: Optional[dict] = None
    """``glmerControl(check.conv.hess=)`` — ``{action, tol}`` for the post-fit
    Hessian definiteness / conditioning diagnostic (8.14)."""

    # Diagnostic carries ------------------------------------------------
    # These follow the data through the fit so the resulting ``gmm`` instance
    # can produce diagnostics, predict on new data, and round-trip formulas.
    expanded: Optional[ExpandedFormula] = None
    """The parsed/expanded formula, used by ``predict`` and ``profile``."""

    data: Optional[pl.DataFrame] = None
    """Post-NA-omit row set, in row-aligned order with X/Z/y/offset."""

    dev_fun_only: bool = False
    """When True, build the stage deviance closure and stop before optimizing —
    ``gmm()`` returns the unfitted instance carrying ``m.devfun`` (a
    :class:`_DevFunHandle`) instead of a fitted model. Mirrors
    ``(g)lmer(devFunOnly=TRUE)`` (lmer.R:46 / 151 / 175)."""

    restart_edge: bool = False
    """Post-optimization boundary-gradient restart — ``glmerControl(restart_edge=)``
    (8.12). lme4 default TRUE for lmer, FALSE/unsupported for glmer."""

    boundary_tol: float = 0.0
    """If > 0, after optimizing try pinning near-boundary θ to the bound when it
    lowers the deviance — ``check.boundary`` / ``glmerControl(boundary.tol=)``
    (8.13)."""


def _sparse_Lt_spec(
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the CSC structure of Λᵀ from the integer template.

    Returns ``(theta_pos, indices, indptr)`` such that for any θ vector,
    ``csc_array((theta[theta_pos], indices, indptr), shape=template.shape)``
    reconstructs Λᵀ. Because the structure is fixed, CHOLMOD can reuse the
    symbolic analysis across every deviance evaluation.
    """
    q = template.shape[0]
    indptr = np.empty(q + 1, dtype=np.int32)
    indptr[0] = 0
    indices_parts: list[np.ndarray] = []
    theta_pos_parts: list[np.ndarray] = []
    for j in range(q):
        col = template[:, j]
        nz_rows = np.nonzero(col)[0]
        indices_parts.append(nz_rows.astype(np.int32))
        theta_pos_parts.append((col[nz_rows] - 1).astype(np.int64))
        indptr[j + 1] = indptr[j] + nz_rows.size
    indices = (
        np.concatenate(indices_parts) if indices_parts
        else np.zeros(0, dtype=np.int32)
    )
    theta_pos = (
        np.concatenate(theta_pos_parts) if theta_pos_parts
        else np.zeros(0, dtype=np.int64)
    )
    return theta_pos, indices, indptr


def _bar_sizes(cnms: dict) -> list[int]:
    """Component count ``c`` per bar (1 for scalar bars, ≥ 2 for vector)."""
    return [
        len(names) if isinstance(names, list) else 1
        for names in cnms.values()
    ]


def _theta_diag_idx(bar_sizes: list[int]) -> list[int]:
    """0-indexed θ positions on the diagonal of any per-level Λᵀ block.

    ``materialize_bars`` packs each c×c upper-triangular Λᵀ block row by
    row: ``θ[off+0] = (0,0)``, ``θ[off+1] = (0,1)``, … . The diagonal
    positions therefore start each row, at cumulative offsets ``c, c-1,
    c-2, …``.
    """
    diag: list[int] = []
    off = 0
    for c in bar_sizes:
        cum = 0
        for i in range(c):
            diag.append(off + cum)
            cum += c - i
        off += c * (c + 1) // 2
    return diag


def _beta_sd_from_RX(RX: np.ndarray) -> np.ndarray:
    """Per-coefficient SD ``√diag((RX·RX')⁻¹)`` — port of lme4's ``pp$unsc()``.

    lme4's ``merPredD::unsc()`` (predModule.cpp:371) returns ``RXi·RXi'``
    where ``RXi = RX⁻¹`` from the upper-triangular factor of the
    Schur-complement Hessian. We store ``RX`` as the lower-triangular
    factor (``RX·RX' = VtV_schur``), so ``unsc = RX⁻ᵀ·RX⁻¹`` and
    ``diag(unsc)[j] = Σᵢ (RX⁻¹)[i,j]²`` — i.e. the column-norms² of
    ``A = RX⁻¹``. Used by Stage 1 Nelder-Mead step-size scaling
    (lmer.R:2535).
    """
    p = RX.shape[0]
    A = solve_triangular(RX, np.eye(p), lower=True)
    return np.sqrt(np.sum(A * A, axis=0))


def _deriv12(
    fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    delta: float = 1e-4,
    fx: Optional[float] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient + Hessian — port of lme4's ``deriv12``.

    lme4 ships its own central-difference scheme in ``R/deriv.R`` instead
    of using ``numDeriv::hessian`` (Richardson extrapolation); the comment
    at modular.R:664 explicitly notes the choice: "don't use numDeriv —
    cruder but fewer dependencies, no worries". The post-fit
    ``m@optinfo$derivs`` and the Hessian-based ``vcov()`` rely on this
    specific scheme, so for byte-match we port it directly rather than
    swapping in a more sophisticated estimator.

    Bound handling (lower/upper, ``NaN`` ≡ R's ``NA``): when
    ``x[j] + delta`` exceeds ``upper[j]``, the right step shrinks to
    ``upper[j] - x[j]`` and the central-difference formula uses the
    asymmetric step. Symmetric on the lower side. This is the same
    "udelta / ldelta" trick R/deriv.R:38-53 uses for optima at the bound.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 1:
        raise ValueError("x must be nonempty")
    if fx is None:
        fx = float(fn(x))
    fx = float(fx)

    xadd = x + delta
    udelta = np.full(n, delta)
    if upper is not None:
        upper = np.asarray(upper, dtype=float)
        active = ~np.isnan(upper) & (xadd > upper)
        if active.any():
            udelta = np.where(active, upper - x, delta)
            xadd = np.where(active, upper, xadd)

    xsub = x - delta
    ldelta = np.full(n, delta)
    if lower is not None:
        lower = np.asarray(lower, dtype=float)
        active = ~np.isnan(lower) & (xsub < lower)
        if active.any():
            ldelta = np.where(active, x - lower, delta)
            xsub = np.where(active, lower, xsub)

    g = np.empty(n)
    H = np.empty((n, n))
    for j in range(n):
        xj = x.copy()
        xj[j] = xadd[j]
        fadd = float(fn(xj))
        xj = x.copy()
        xj[j] = xsub[j]
        fsub = float(fn(xj))
        udj, ldj = udelta[j], ldelta[j]
        H[j, j] = fadd / udj**2 - 2.0 * fx / (udj * ldj) + fsub / ldj**2
        g[j] = (fadd - fsub) / (udj + ldj)
        for i in range(j):
            udi, ldi = udelta[i], ldelta[i]
            x_aa = x.copy()
            x_aa[i] = xadd[i]
            x_aa[j] = xadd[j]
            x_as = x.copy()
            x_as[i] = xadd[i]
            x_as[j] = xsub[j]
            x_sa = x.copy()
            x_sa[i] = xsub[i]
            x_sa[j] = xadd[j]
            x_ss = x.copy()
            x_ss[i] = xsub[i]
            x_ss[j] = xsub[j]
            val = (
                float(fn(x_aa)) / (udi + udj) ** 2
                - float(fn(x_as)) / (udi + ldj) ** 2
                - float(fn(x_sa)) / (ldi + udj) ** 2
                + float(fn(x_ss)) / (ldi + ldj) ** 2
            )
            H[i, j] = H[j, i] = val
    return g, H


def _per_bar_relative_cov(theta: np.ndarray, bar_sizes: list[int]) -> list[np.ndarray]:
    """Recover the c×c relative-covariance ``Σ_b = Λ_b Λ_bᵀ`` per bar."""
    blocks: list[np.ndarray] = []
    off = 0
    for c in bar_sizes:
        Lt = np.zeros((c, c))
        iu, ju = np.triu_indices(c)
        Lt[iu, ju] = theta[off:off + iu.size]
        L = Lt.T
        blocks.append(L @ L.T)
        off += c * (c + 1) // 2
    return blocks


class _GlmResponse:
    """GLMM response state — port of lme4's ``glmResp`` / ``lmerResp``.

    Mirrors the C++ class hierarchy in ``lme4/src/respModule.cpp`` and
    ``respModule.h``: holds the response ``y``, prior weights, offset, and
    the current ``(η, μ)`` plus the working weights / residuals that PIRLS
    reads each iteration.

    The Gaussian-identity case (``lmerResp``) is a degenerate path through
    the same state — no link inverse, no working weights — handled by
    skipping :meth:`update_weights` and reading ``μ`` directly as ``η``
    minus offset. For now the class is used only by the non-Gaussian
    Laplace path; the LMM profiled-deviance code does not go
    through it.

    State (``snake_case`` mirrors of lme4's ``d_*`` members):

    * ``family``: a :class:`hea.family.Family` instance.
    * ``y``, ``weights``, ``offset``: arrays of length ``n``.
    * ``eta``, ``mu``: current linear predictor and response-scale mean.
    * ``sqrt_x_wt``: ``μ_η · sqrt_r_wt`` — X-side √working weights
      (= lme4's ``d_sqrtXwt``).
    * ``sqrt_r_wt``: ``sqrt(weights / V(μ))`` — residual-side √weights
      (= lme4's ``d_sqrtrwt``).
    * ``wt_res``: ``sqrt_r_wt · (y - μ)`` — current weighted residuals.
    * ``wrss``: ``||wt_res||²``.
    * ``log_det_weights``: ``Σ log w[w>0]`` — used by lmer Laplace
      criterion to absorb the prior-weight Jacobian.

    Method shapes follow lme4 but with Pythonic names:

    Mutators (refresh dependent fields in lock-step):

    * :meth:`update_mu`: set ``η = offset + γ``, refresh ``μ`` and ``wrss``.
    * :meth:`update_weights`: refresh ``sqrt_r_wt``, ``sqrt_x_wt``, ``wrss``.
    * :meth:`update_wrss`: refresh ``wt_res`` and ``wrss``.

    Pure-compute (read state, no mutation):

    * :meth:`working_residuals`, :meth:`working_response`,
      :meth:`weighted_working_response` — PIRLS RHS pieces.
    * :meth:`deviance_residuals`, :meth:`deviance`, :meth:`aic` — family-
      driven evaluators.
    * :meth:`laplace` — the Laplace approximation
      ``ldL2 + ||u||² + aic`` (port of ``respModule.cpp:161``).
    """

    __slots__ = (
        "family", "y", "weights", "offset",
        "eta", "mu", "sqrt_x_wt", "sqrt_r_wt", "wt_res", "wrss",
        "log_det_weights",
    )

    def __init__(
        self,
        family: Family,
        y: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        mustart: Optional[np.ndarray] = None,
        etastart: Optional[np.ndarray] = None,
    ):
        y = np.asarray(y, dtype=float)
        n = len(y)

        if weights is None:
            weights = np.ones(n)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (n,):
                raise ValueError(
                    f"weights shape {weights.shape} doesn't match y shape ({n},)"
                )

        if offset is None:
            offset = np.zeros(n)
        else:
            offset = np.asarray(offset, dtype=float)
            if offset.shape != (n,):
                raise ValueError(
                    f"offset shape {offset.shape} doesn't match y shape ({n},)"
                )

        # Initial μ from mustart (user-provided) or family.initialize.
        # Mirror utilities.R:236-258: family$initialize fills mustart, and a
        # user mustart_update overrides afterwards.
        if mustart is None:
            mustart = family.initialize(y, weights)
        else:
            mustart = np.asarray(mustart, dtype=float).copy()

        # Gamma stability fix (utilities.R:250-252): when no etastart is
        # supplied, replace mustart with its mean. Reason: PIRLS on
        # log-link Gamma diverges from a saturated mustart because
        # E[log(y)] ≤ log(E[y]) (Jensen's inequality on the link), so
        # initialising η = log(y) gives an over-shoot on the first step.
        if family.name == "Gamma" and etastart is None:
            mustart = np.full_like(mustart, float(np.mean(mustart)))

        # Initial η. If etastart provided, use it; else linkfun(mustart).
        # NB: lme4 passes this directly to updateMu (utilities.R:257),
        # which adds the offset — so the *initial* η ends up offset-shifted
        # relative to a clean linkfun(mustart). This is the documented lme4
        # behaviour; PIRLS converges from any reasonable starting state, so
        # it doesn't affect the final fit. We replicate it for parity.
        if etastart is not None:
            initial_gamma = np.asarray(etastart, dtype=float).copy()
        else:
            initial_gamma = family.link.link(mustart)

        # log Σ w[w>0] — Jacobian term for the lmer REML/ML criterion.
        good = weights > 0
        log_det_weights = (
            float(np.sum(np.log(weights[good]))) if np.any(good) else 0.0
        )

        # Seed mutable state — start at all-zeros, then call update_mu /
        # update_weights to populate consistently.
        self.family = family
        self.y = y
        self.weights = weights
        self.offset = offset
        self.eta = np.zeros(n)
        self.mu = np.zeros(n)
        self.sqrt_x_wt = np.zeros(n)
        self.sqrt_r_wt = np.zeros(n)
        self.wt_res = np.zeros(n)
        self.wrss = 0.0
        self.log_det_weights = log_det_weights

        # Now drive state to the initial (η, μ, weights, wrss) consistently.
        # Order matters: update_mu sets μ from η, then update_weights uses μ.
        self.update_mu(initial_gamma)
        self.update_weights()

    # ------- mutators -----------------------------------------------------

    def update_mu(self, gamma: np.ndarray) -> float:
        """Set ``η = offset + γ`` and refresh ``μ`` and ``wrss``.

        Port of ``glmResp::updateMu`` (respModule.cpp:169-177). ``γ`` is the
        offset-free linear predictor (typically ``X·β + Z·b`` from
        ``merPredD::linPred``); the offset is added here.
        """
        eta = self.offset + np.asarray(gamma, dtype=float)
        self.eta = eta
        self.mu = self.family.link.linkinv(eta)
        return self.update_wrss()

    def update_wrss(self) -> float:
        """Refresh ``wt_res = sqrt_r_wt · (y - μ)`` and ``wrss``.

        Port of ``lmResp::updateWrss`` (respModule.cpp:56-60). Called by
        both :meth:`update_mu` and :meth:`update_weights`; can also be
        called standalone when only the residual term needs refreshing
        (rare — kept for parity).
        """
        self.wt_res = self.sqrt_r_wt * (self.y - self.mu)
        self.wrss = float(np.dot(self.wt_res, self.wt_res))
        return self.wrss

    def update_weights(self) -> float:
        """Refresh working weights from the current μ and η.

        ``sqrt_r_wt = sqrt(weights / V(μ))``
        ``sqrt_x_wt = μ_η · sqrt_r_wt``

        Port of ``glmResp::updateWts`` (respModule.cpp:179-183). PIRLS
        calls this once per iteration after :meth:`update_mu`. Returns the
        new ``wrss`` (since ``sqrt_r_wt`` changed, the weighted residuals
        change too).
        """
        variance = self.family.variance(self.mu)
        self.sqrt_r_wt = np.sqrt(self.weights / variance)
        self.sqrt_x_wt = self.family.link.mu_eta(self.eta) * self.sqrt_r_wt
        return self.update_wrss()

    # ------- pure-compute (no mutation) -----------------------------------

    def working_residuals(self) -> np.ndarray:
        """``(y - μ) / μ_η`` — port of ``glmResp::wrkResids`` (respModule.cpp:140)."""
        return (self.y - self.mu) / self.family.link.mu_eta(self.eta)

    def working_response(self) -> np.ndarray:
        """``(η - offset) + working_residuals`` — port of ``wrkResp`` (respModule.cpp:144).

        The PIRLS working response ``z`` is what gets regressed against the
        weighted ``X`` in the inner loop.
        """
        return (self.eta - self.offset) + self.working_residuals()

    def weighted_working_response(self) -> np.ndarray:
        """``working_response · sqrt_x_wt`` — port of ``wtWrkResp`` (respModule.cpp:148).

        The right-hand side ``√W · z`` for the PIRLS weighted-LS step.
        """
        return self.working_response() * self.sqrt_x_wt

    def deviance_residuals(self) -> np.ndarray:
        """Family deviance contributions per observation.

        Port of ``glmResp::devResid`` (respModule.cpp:128). Delegates to
        :meth:`hea.family.Family.dev_resids`.
        """
        return self.family.dev_resids(self.y, self.mu, self.weights)

    def deviance(self) -> float:
        """Total deviance ``Σ devResid``.

        Port of ``glmResp::resDev`` (respModule.cpp:165). Matches what
        :func:`stats::deviance.merMod` returns for GLMM.

        Uses ``np.cumsum(...)[-1]`` (sequential left-to-right reduction),
        which empirically lands ~1 ULP from R's ``deviance(m)`` on
        n=1934 binomial GLMM (Contraception), while ``np.sum`` (pairwise)
        lands 3 ULP off in the opposite direction. The 1-ULP cumsum
        agreement is the closest reduction order to R; switching costs
        ~2 ULP per call and propagates to ~1e-6 in converged θ̂.
        """
        return float(np.cumsum(self.deviance_residuals())[-1])

    def aic(self) -> float:
        """Family AIC contribution ``family.aic(y, μ, dev, w, n)``.

        Port of ``glmResp::aic`` (respModule.cpp:124). ``n`` (the binomial
        denominator on R's side) is folded into ``weights`` in hea's
        :class:`Binomial` and ignored by other families, so we pass an
        all-ones array for compatibility with the family signature.
        """
        return float(self.family.aic(
            self.y, self.mu, self.deviance(),
            self.weights, np.ones(len(self.y)),
        ))

    def laplace(self, log_det_l_sq: float, log_det_rx_sq: float,
                sqr_len_u: float) -> float:
        """GLMM Laplace approximation to ``-2 log L_marginal``.

        Port of ``glmResp::Laplace`` (respModule.cpp:161-163):

            laplace = log_det_l_sq + sqr_len_u + aic

        where ``log_det_l_sq = 2 log|L|`` from CHOLMOD's factor of
        ``Λ Z'WZ Λ' + I``, ``sqr_len_u = ||u||²`` is the random-effect
        penalty (from ``merPredD::sqrL(1)``), and ``aic`` carries the
        conditional log-likelihood contribution.

        ``log_det_rx_sq`` (the fixed-effect Cholesky log-det) is accepted
        for signature symmetry with the lmer path; lme4's glmResp Laplace
        does not use it.
        """
        del log_det_rx_sq  # unused in GLMM Laplace; here for signature parity
        return log_det_l_sq + sqr_len_u + self.aic()


class _PredState:
    """PIRLS predictor-side state — port of lme4's ``merPredD``.

    Carries the design pieces (``X``, ``Z``), the parameterised
    Cholesky factor ``Λᵀ(θ)``, the current "base" point ``(β0, u0)``, the
    step ``(δβ, δu)`` away from that base, and the CHOLMOD factor of
    ``M = Λ Z' W Z Λᵀ + I`` for the current working weights ``W``.

    The state is mutable: ``set_theta`` refreshes ``Λᵀ``,
    ``update_xwts_and_decomp`` refreshes the weighted decomposition, and
    ``solve`` writes new ``(δβ, δu)``. This mirrors lme4's C++ class
    (Eigen Maps over R-side memory) — PIRLS reads and rewrites the same
    state object across iterations, with the CHOLMOD symbolic factor
    cached for cheap numeric refactors when only weights change.

    PLS math is done via the Schur complement (single full-system CHOLMOD
    ``M⁻¹b`` solves), matching how the Gaussian path in
    :meth:`gmm._fit_from_components` already operates. Mathematically
    equivalent to lme4's staged ``P/L/Lt/Pt`` solveInPlace sequence in
    ``predModule.cpp:189-214``.
    """

    __slots__ = (
        # Read-only design pieces ----------------------------------------
        "X",                 # (n, p) fixed-effect design
        "Z_sp",              # (n, q) sparse Z (CSC)
        "n", "p", "q",
        # Λᵀ template (built once from ReTerms, structure is fixed) ------
        "_lt_theta_pos", "_lt_indices", "_lt_indptr", "_lt_shape",
        # Persistent state ----------------------------------------------
        "theta", "beta0", "u0", "delb", "delu",
        # Current weighted state (set by update_xwts_and_decomp) --------
        "sqrt_x_wt",
        "lambdat_sp",        # current Λᵀ as CSC (built by set_theta)
        "V",                 # (n, p) = diag(sqrt_x_wt) · X
        "VtV",               # (p, p) = V'·V
        "lamt_ut",           # (q, n) sparse = Λᵀ · √W · Z' (i.e. LamtUt)
        "RZX_unfactored",    # (q, p) = lamt_ut · V
        "M_inv_RZX",         # (q, p) = M⁻¹ · RZX_unfactored
        "RX",                # (p, p) lower Cholesky of (V'V − RZX'·M⁻¹·RZX)
        "log_det_l_sq",      # 2 log|L| from CHOLMOD
        "log_det_rx_sq",     # 2 log|RX|
        "chol_factor",       # CHOLMOD factor of M = LamtUt · LamtUt' + I
        # Cached identity for M assembly (built once) -------------------
        "_eye_q_sp",
    )

    def __init__(self, X: np.ndarray, Z_sp: csc_array, re_terms: ReTerms):
        n, p = X.shape
        q = Z_sp.shape[1]
        if Z_sp.shape[0] != n:
            raise ValueError(
                f"Z rows ({Z_sp.shape[0]}) don't match X rows ({n})"
            )

        self.X = X
        self.Z_sp = Z_sp
        self.n, self.p, self.q = n, p, q

        # Precompute the CSC structure of Λᵀ from the integer template.
        lt_theta_pos, lt_indices, lt_indptr = _sparse_Lt_spec(re_terms.Lambdat)
        self._lt_theta_pos = lt_theta_pos
        self._lt_indices = lt_indices
        self._lt_indptr = lt_indptr
        self._lt_shape = re_terms.Lambdat.shape

        # Initial state: base point at origin, no step.
        self.theta = re_terms.theta.astype(float).copy()
        self.beta0 = np.zeros(p)
        self.u0 = np.zeros(q)
        self.delb = np.zeros(p)
        self.delu = np.zeros(q)

        # Weighted state — populated by update_xwts_and_decomp.
        self.sqrt_x_wt = np.zeros(n)
        self.V = np.zeros((n, p))
        self.VtV = np.zeros((p, p))
        self.lamt_ut = None       # filled by set_theta + xwts
        self.RZX_unfactored = np.zeros((q, p))
        self.M_inv_RZX = np.zeros((q, p))
        self.RX = np.zeros((p, p))
        self.log_det_l_sq = 0.0
        self.log_det_rx_sq = 0.0
        self.chol_factor = None

        # Cache the q×q identity for M = LamtUt · LamtUt' + I.
        self._eye_q_sp = eye_array(q, format="csc")

        # Build initial Λᵀ from θ₀ so callers can call update_xwts_and_decomp
        # immediately.
        self.lambdat_sp = self._build_lambdat(self.theta)

    # ------- internal --------------------------------------------------

    def _build_lambdat(self, theta: np.ndarray) -> csc_array:
        """Rebuild Λᵀ from a new θ vector. Sparse structure stays fixed —
        only the nonzero values change, which lets CHOLMOD reuse its
        symbolic factor across calls.
        """
        data = np.asarray(theta, dtype=float)[self._lt_theta_pos]
        return csc_array(
            (data, self._lt_indices, self._lt_indptr),
            shape=self._lt_shape, copy=False,
        )

    # ------- mutators --------------------------------------------------

    def set_theta(self, theta: np.ndarray) -> None:
        """Set new ``θ`` and refresh ``Λᵀ``. Doesn't touch the weighted
        decomposition — caller must call :meth:`update_xwts_and_decomp`
        next.
        """
        self.theta = np.asarray(theta, dtype=float).copy()
        self.lambdat_sp = self._build_lambdat(self.theta)

    def update_xwts_and_decomp(self, sqrt_x_wt: np.ndarray) -> None:
        """Apply new X-side √working weights and refresh the decomposition.

        Mirrors lme4's ``merPredD::updateXwts`` + ``updateDecomp``
        (predModule.cpp:216-301). Specifically:

        1. ``V = diag(sqrt_x_wt) · X``,
           ``Ut = diag(sqrt_x_wt) · Z'`` (sparse, in-place on Z's pattern),
           ``VtV = V'·V``.
        2. ``lamt_ut = Λᵀ · Ut``.
        3. ``M = lamt_ut · lamt_ut' + I``, factorize via CHOLMOD
           (re-uses symbolic factor when available).
        4. ``ldL2 = 2 log|L|``.
        5. ``RZX_unfactored = lamt_ut · V``;
           ``M_inv_RZX = M⁻¹ · RZX_unfactored``.
        6. ``VtV_schur = VtV − RZX_unfactored' · M_inv_RZX``;
           ``RX = chol(VtV_schur)``; ``ldRX2 = 2 log|RX|``.
        """
        sqrt_x_wt = np.asarray(sqrt_x_wt, dtype=float)
        if sqrt_x_wt.shape != (self.n,):
            raise ValueError(
                f"sqrt_x_wt shape {sqrt_x_wt.shape} doesn't match n={self.n}"
            )
        self.sqrt_x_wt = sqrt_x_wt

        # V = diag(sqrt_x_wt) · X — dense
        self.V = sqrt_x_wt[:, None] * self.X
        self.VtV = self.V.T @ self.V

        # Ut = diag(sqrt_x_wt) · Z' — sparse. Scale each column of Z by
        # sqrt_x_wt[j], then transpose.
        Z_scaled = self.Z_sp.multiply(sqrt_x_wt[:, None]).tocsc()
        Ut = csc_array(Z_scaled.T)

        # lamt_ut = Λᵀ · Ut — sparse @ sparse
        self.lamt_ut = (self.lambdat_sp @ Ut).tocsc()

        # M = lamt_ut · lamt_ut' + I_q. Factorize (re-using symbolic).
        M = (self.lamt_ut @ self.lamt_ut.T + self._eye_q_sp).tocsc()
        if self.chol_factor is None:
            self.chol_factor = cho_factor(M)
        else:
            self.chol_factor.factorize(M)
        self.log_det_l_sq = 2.0 * self.chol_factor.half_log_det()

        # RZX_unfactored = lamt_ut · V (dense, q×p).
        self.RZX_unfactored = np.asarray(self.lamt_ut @ self.V)

        # M⁻¹ · RZX_unfactored (dense, q×p). The Gaussian path uses
        # einsum for this; here we just use the factor's solve.
        if self.p > 0:
            self.M_inv_RZX = self.chol_factor.solve(self.RZX_unfactored)
            VtV_schur = self.VtV - np.einsum(
                "ij,ik->jk", self.RZX_unfactored, self.M_inv_RZX,
            )
            # chol returns lower-triangular L with VtV_schur = L · L'
            try:
                self.RX = np.linalg.cholesky(VtV_schur)
            except np.linalg.LinAlgError as exc:
                raise CholmodError(
                    "Fixed-effect Cholesky failed — Schur complement not "
                    "positive definite. Likely an ill-conditioned design "
                    "matrix or a θ that drove M close to singular."
                ) from exc
            self.log_det_rx_sq = 2.0 * float(np.log(np.diag(self.RX)).sum())
        else:
            self.M_inv_RZX = np.zeros((self.q, 0))
            self.RX = np.zeros((0, 0))
            self.log_det_rx_sq = 0.0

    def solve(self, weighted_response: np.ndarray, *, u_only: bool = False) -> float:
        """Solve the PLS step for ``(δβ, δu)``.

        Given the weighted working response ``z_w`` (length n), compute
        the right-hand side ``Vtr = V'·z_w``, ``Utr = Λᵀ·U·z_w``, then
        solve the block system via the Schur complement:

            δβ = (V'V − RZX'·M⁻¹·RZX)⁻¹ · (Vtr − RZX'·M⁻¹·(Utr − u0))
            δu = M⁻¹ · ((Utr − u0) − RZX·δβ)

        For ``u_only=True`` (used by nAGQ=0 GLMM where β is held fixed),
        skip the δβ step:

            δβ = 0
            δu = M⁻¹ · (Utr − u0)

        Returns ``CcNumer = ||L⁻¹·P·(Utr − u0)||² + ||δβ||²`` —
        lme4's convergence-criterion numerator (predModule.cpp:193, 196).
        We compute its value as ``(Utr − u0)'·M⁻¹·(Utr − u0) + ||δβ||²``
        (same quantity, equivalent under the Cholesky identity).
        """
        z = np.asarray(weighted_response, dtype=float)
        if z.shape != (self.n,):
            raise ValueError(
                f"weighted_response shape {z.shape} doesn't match n={self.n}"
            )
        Vtr = self.V.T @ z
        Utr = np.asarray(self.lamt_ut @ z).ravel()
        offset = Utr - self.u0

        if u_only or self.p == 0:
            # Pure-u path: δβ = 0, δu = M⁻¹ · offset.
            self.delb = np.zeros(self.p)
            self.delu = self.chol_factor.solve(offset)
            cc_numer = float(np.einsum("i,i->", offset, self.delu))
            return cc_numer

        # Joint (δβ, δu) path via Schur complement.
        # cu = M⁻¹ · offset (in factored form: P' L⁻ᵀ L⁻¹ P · offset)
        cu = self.chol_factor.solve(offset)
        # rhs = Vtr − RZX'·cu (where RZX = M⁻¹/²·LamtUt·V ≡ "factored" form;
        # what we have is RZX_unfactored = LamtUt·V, and cu = M⁻¹·offset.
        # The product RZX_unfactored.T @ cu is exactly the rotated quantity.)
        rhs = Vtr - np.einsum("ij,i->j", self.RZX_unfactored, cu)
        # δβ = (VtV_schur)⁻¹ · rhs via two triangular solves on RX.
        cb = solve_triangular(self.RX, rhs, lower=True)
        self.delb = solve_triangular(self.RX.T, cb, lower=False)
        # δu = M⁻¹ · (offset − LamtUt·V · δβ)
        #    = cu − M_inv_RZX · δβ
        self.delu = cu - self.M_inv_RZX @ self.delb
        # CcNumer = (Utr − u0)·M⁻¹·(Utr − u0) + ||δβ||²
        cu_sq = float(np.einsum("i,i->", offset, cu))
        cc_numer = cu_sq + float(np.einsum("i,i->", self.delb, self.delb))
        return cc_numer

    def install_pars(self, f: float = 1.0) -> None:
        """Snapshot the current step: ``u0 ← u0 + f·δu``, ``β0 ← β0 +
        f·δβ``, ``δu = δβ = 0``. Port of ``merPredD::installPars``
        (predModule.cpp:310-315).

        Called by the outer optimizer after PIRLS converges, to lock in
        the new "base" point for downstream solves.
        """
        self.u0 = self.u0 + f * self.delu
        self.beta0 = self.beta0 + f * self.delb
        self.delu = np.zeros(self.q)
        self.delb = np.zeros(self.p)

    # ------- pure-compute (no mutation) --------------------------------

    def beta(self, f: float = 1.0) -> np.ndarray:
        """Fixed-effect coefficients at step factor ``f``: ``β0 + f·δβ``.

        Port of ``merPredD::beta(f)`` (predModule.cpp:92).
        """
        return self.beta0 + f * self.delb

    def u(self, f: float = 1.0) -> np.ndarray:
        """Spherical random effects at step factor ``f``: ``u0 + f·δu``.

        Port of ``merPredD::u(f)`` (predModule.cpp:138).
        """
        return self.u0 + f * self.delu

    def b(self, f: float = 1.0) -> np.ndarray:
        """Non-spherical random effects ``b = Λᵀ · u(f)`` — port of
        ``merPredD::b(f)`` (predModule.cpp:90).
        """
        return np.asarray(self.lambdat_sp.T @ self.u(f)).ravel()

    def lin_pred(self, f: float = 1.0) -> np.ndarray:
        """Offset-free linear predictor ``γ = X·β(f) + Z'·b(f)``.

        Port of ``merPredD::linPred(f)`` (predModule.cpp:94-96). The
        caller (``_GlmResponse.update_mu``) adds the offset to get ``η``.
        """
        return self.X @ self.beta(f) + np.asarray(self.Z_sp @ self.b(f)).ravel()

    def sqr_l_u(self, f: float = 1.0) -> float:
        """``||u(f)||²`` — RE penalty in the Laplace approximation.

        Port of ``merPredD::sqrL(f)`` (predModule.cpp:140). Uses
        sequential cumsum to keep within ~1 ULP of R for small u arrays
        (typical glmer has q = n_groups, often 60-1000); see
        :meth:`_GlmResponse.deviance` for the rationale.
        """
        u_f = self.u(f)
        return float(np.cumsum(u_f * u_f)[-1])


def _internal_glmer_wrk_iter(
    pred: _PredState, resp: _GlmResponse, *, u_only: bool,
) -> float:
    """One PIRLS iteration — port of ``internal_glmerWrkIter`` (external.cpp:268-295).

    Refreshes the working weights from current ``μ``, runs the predictor's
    weighted decomposition, solves the PLS step, then updates the response
    to the new linear predictor. Returns the penalised deviance
    ``Σ deviance_residuals + ||u(1)||²``.

    The leading :meth:`_GlmResponse.update_weights` call matches lme4's
    ``rp->sqrtWrkWt()`` method (respModule.cpp:152-159), which computes
    fresh from current ``μ`` rather than reading a stale stored field.
    Without it, PIRLS would use weights from the previous iteration's
    ``μ`` and could oscillate without converging.

    Caller (``_pwrss_update``) loops this until ``pdev`` converges.
    """
    resp.update_weights()
    pred.update_xwts_and_decomp(resp.sqrt_x_wt)
    pred.solve(resp.weighted_working_response(), u_only=u_only)
    resp.update_mu(pred.lin_pred(1.0))
    return resp.deviance() + pred.sqr_l_u(1.0)


def _pwrss_update(
    pred: _PredState,
    resp: _GlmResponse,
    *,
    u_only: bool,
    tol: float = 1e-7,
    maxit: int = 100,
    verbose: int = 0,
) -> float:
    """Outer PIRLS loop — port of ``pwrssUpdate`` (external.cpp:308-376).

    Iterates :func:`_internal_glmer_wrk_iter` until ``|Δpdev|/|pdev| <
    tol``. On a pdev increase or NaN, step-halves ``(δu, δβ)`` toward the
    previous iteration's values for up to 20 substeps (matching lme4's
    ``maxstephalfit``). Caller (the devfun closure) discards any state
    that should reset between optimizer calls — this function freely
    mutates ``pred`` and ``resp``.

    Returns the converged ``pdev``. Raises ``RuntimeError`` if PIRLS
    fails to converge or step-halving cannot recover from a divergence.
    """
    max_stephalfit = 20
    old_pdev = np.finfo(float).max
    pdev = old_pdev
    converged = False

    for i in range(maxit):
        old_delu = pred.delu.copy()
        old_delb = pred.delb.copy()
        pdev = _internal_glmer_wrk_iter(pred, resp, u_only=u_only)
        if verbose > 2:
            print(f"pwrss iter {i}: pdev={pdev:.10g}")
        if np.abs((old_pdev - pdev) / pdev) < tol:
            converged = True
            break

        # Step-halving on increase or NaN. Mirrors external.cpp:341-369.
        if np.isnan(pdev) or pdev > old_pdev:
            if verbose > 2:
                print("  entering step-halving loop")
            for k in range(max_stephalfit):
                if not (np.isnan(pdev) or pdev > old_pdev):
                    break
                pred.delu = (old_delu + pred.delu) / 2.0
                if not u_only:
                    pred.delb = (old_delb + pred.delb) / 2.0
                resp.update_mu(pred.lin_pred(1.0))
                pdev = resp.deviance() + pred.sqr_l_u(1.0)
            if np.isnan(pdev):
                raise RuntimeError("PIRLS loop produced NaN pdev")
            if (pdev - old_pdev) > tol:
                raise RuntimeError(
                    f"PIRLS step-halving failed to reduce pdev after "
                    f"{max_stephalfit} halvings (pdev={pdev}, "
                    f"old_pdev={old_pdev})"
                )
        old_pdev = pdev

    if not converged:
        raise RuntimeError(
            f"PIRLS did not converge in {maxit} iterations (last pdev={pdev})"
        )
    return pdev


class _DevFunHandle:
    """Callable deviance-function handle returned by ``gmm(..., devFunOnly=True)``
    — lme4's diagnostic entry point (``mkdevfun`` / ``(g)lmer(devFunOnly=TRUE)``).

    Call it with a parameter vector to evaluate ``-2 log L``: the profiled
    REML/ML deviance for LMMs, the Laplace/AGQ deviance for GLMMs. ``par_names``
    /``lower``/``upper`` describe the parameter vector — ``θ`` for the lmer and
    glmer-``nAGQ=0`` closures, ``[θ, β]`` for the glmer ``nAGQ>0`` Stage-1
    closure.
    """

    def __init__(self, fn, par_names, lower, upper):
        self._fn = fn
        self.par_names = list(par_names)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    def __call__(self, par) -> float:
        return float(self._fn(np.asarray(par, dtype=float)))

    def __repr__(self) -> str:
        return f"_DevFunHandle(npar={len(self.par_names)}, par_names={self.par_names})"


def _gh_rule(ord: int) -> np.ndarray:
    """Univariate Gauss-Hermite quadrature rule for adaptive GHQ (``nAGQ>1``).

    Port of lme4's ``GHrule(ord)`` (R/GHrule.R). Returns an ``(ord, 3)`` array
    whose columns are ``(z, w, ldnorm)``: node positions ``z`` and weights ``w``
    for integrating ``f(x)`` against the standard-normal density, plus
    ``ldnorm = log φ(z)``.

    lme4 reads precomputed ``fastGHQuad::gaussHermiteData(ord)`` values from
    ``sysdata.rda`` and rescales them (``w ← w/Σw``, ``x ← x·√2``); we compute
    the same rule at runtime from :func:`scipy.special.roots_hermite` (both use
    the physicists' ``e^{-x²}`` convention, so the nodes are the roots of the
    same Hermite polynomial). The forward/reverse symmetrization
    (``z ← (z−rev z)/2``, ``w ← (w+rev w)/2``; lme4 issue #968) forces exact
    symmetry. Matches ``lme4:::GHrule`` to ≤5e-14 for ``ord ≤ 25``.

    ``ord == 0`` returns an empty ``(0, 3)`` array (mirrors lme4's
    ``asMatrix=TRUE`` zero case). The middle node of an odd-order rule is
    near-zero-but-not-exactly-zero (numerical noise ~1e-16), matching lme4 —
    which matters because :func:`_glmm_agq_deviance` special-cases ``z == 0``.
    """
    ord = int(ord)
    if ord < 0 or ord > 100:
        raise ValueError(f"GH rule order must be in [0, 100]; got {ord}")
    if ord == 0:
        return np.zeros((0, 3))
    x, w = roots_hermite(ord)
    z = x * np.sqrt(2.0)
    w = w / w.sum()
    # Symmetrize forward/reverse — lme4 GHrule.R (issue #968).
    z = (z - z[::-1]) / 2.0
    w = (w + w[::-1]) / 2.0
    ldnorm = -0.5 * np.log(2.0 * np.pi) - 0.5 * z**2
    return np.column_stack([z, w, ldnorm])


def _devc_col(fac: np.ndarray, u: np.ndarray, dev_res: np.ndarray,
              n_levels: int) -> np.ndarray:
    """Per-level ``u² + Σ devResid`` — port of ``devcCol`` (external.cpp:398-406).

    ``fac`` is the 0-based grouping-factor code per observation (length n);
    ``u`` the conditional modes (length ``n_levels``); ``dev_res`` the
    per-observation deviance residuals (length n). Returns a length-``n_levels``
    vector: each level's squared conditional mode plus the sum of the deviance
    residuals of the observations in that level.
    """
    ans = u * u
    np.add.at(ans, fac, dev_res)
    return ans


def _glmm_agq_deviance(pred: "_PredState", resp: "_GlmResponse",
                       gqmat: np.ndarray, fac: np.ndarray,
                       n_levels: int) -> float:
    """Adaptive Gauss-Hermite ``-2 log L`` for a single scalar RE.

    Port of ``glmerAGQ`` (external.cpp:414-460). Assumes ``pred``/``resp`` are
    at the conditional mode for the current ``(θ, β)`` — PIRLS has just
    converged with ``pred.u0 == 0`` so ``pred.u(1) == mode`` and
    ``pred.delu == mode`` (hea never ``install_pars`` during the outer loop;
    see ``_fit_glmm_from_components``). The loop relies on that split:
    ``set u0 = z·sd`` ⇒ ``u(1) = z·sd + mode`` (the adaptive GH shift).

    Each level's posterior integral is approximated by GH nodes centred at the
    mode and scaled by the posterior SD ``sd = 1/diag(L)``. For a single scalar
    RE the system ``M = Λ'U U'Λ + I`` is diagonal, so ``diag(L) = √diag(M)``
    and ``sd = 1/√(rowSums(lamt_ut²) + 1)`` — computed directly in u-indexing
    (robust to CHOLMOD's permutation, and exactly lme4's ``1/L.factor()->x``
    when M is diagonal).
    """
    sqrt2pi = np.sqrt(2.0 * np.pi)
    devc0 = _devc_col(fac, pred.u(1.0), resp.deviance_residuals(), n_levels)
    m_diag = np.asarray(
        pred.lamt_ut.multiply(pred.lamt_ut).sum(axis=1)
    ).ravel() + 1.0
    sd = 1.0 / np.sqrt(m_diag)

    u0_saved = pred.u0.copy()
    mult = np.zeros(n_levels)
    for zknot, w, ldnorm in gqmat:
        if zknot == 0.0:
            # Central node: integrand is exactly ``w`` (devc == devc0 and
            # exp(-ldnorm₀)/√2π == 1). lme4 special-cases this.
            mult += w
        else:
            pred.u0 = zknot * sd
            resp.update_mu(pred.lin_pred(1.0))
            devc = _devc_col(fac, pred.u(1.0), resp.deviance_residuals(),
                             n_levels)
            mult += np.exp(-0.5 * (devc - devc0) - ldnorm) * w / sqrt2pi
    # Restore the conditional-mode state (u0 == 0 ⇒ u(1) == mode).
    pred.u0 = u0_saved
    resp.update_mu(pred.lin_pred(1.0))
    return float(devc0.sum() + pred.log_det_l_sq - 2.0 * np.log(mult).sum())


def _glmm_devfun_factory(
    pred: _PredState,
    resp: _GlmResponse,
    *,
    nagq: int,
    tol_pwrss: float = 1e-7,
    maxit_pwrss: int = 100,
    verbose: int = 0,
    gqmat: Optional[np.ndarray] = None,
    fac: Optional[np.ndarray] = None,
    n_levels: Optional[int] = None,
) -> Callable[[np.ndarray], float]:
    """Build the Laplace deviance evaluator for a given optimization stage.

    Port of ``mkdevfun`` (lmer.R:308-384) — the GLMM branch. Returns a closure
    that takes a parameter vector and returns the Laplace approximation to
    ``-2 log L_marginal``. The closure resets PIRLS state to the snapshotted
    ``lp0`` (offset-free linear predictor) before each evaluation, so the
    optimizer sees ``devfun`` as a pure function of its argument.

    Parameters
    ----------
    pred, resp
        Live :class:`_PredState` / :class:`_GlmResponse` objects. The factory
        snapshots their state — ``lp0`` from ``pred.lin_pred(1)`` and (for
        ``nagq>0``) the base offset from ``resp.offset`` — at call time, so
        the caller must arrange these to the desired Stage-{0,1} starting
        point before calling the factory. For lme4-matching numerics, the
        caller should warm-start ``(β, u)`` to the conditional mode at
        ``θ₀`` via a one-off :func:`_pwrss_update` first — mirroring
        ``mkGlmerDevfun``'s ``.Call(glmerLaplace, ...)`` at modular.R:888.
        Without this warm-up, ``lp0`` snapshots the constructor's zero state
        and PIRLS inside each devfun call takes more iterations; the final
        Laplace converges to the same value but at a different "staleness
        offset" in ``ldL2``, producing ~1e-4 mismatches against lme4.
    nagq : int
        ``0`` for the Stage 0 (θ-only) closure, ``1`` for the Stage 1
        (θ, β) Laplace closure, ``>1`` for the Stage 1 adaptive Gauss-Hermite
        closure — which then requires ``gqmat`` (``_gh_rule(nagq)``), ``fac``
        (per-obs 0-based grouping codes), and ``n_levels`` (a single scalar
        RE; see :func:`_glmm_agq_deviance`).
    tol_pwrss, maxit_pwrss, verbose
        Passed to :func:`_pwrss_update`. Match lme4's
        ``glmerControl(tolPwrss=1e-7, ...)`` defaults.

    Returns
    -------
    callable
        For ``nagq=0``: ``devfun(theta)``. For ``nagq>0``: ``devfun(par)``
        where ``par = concatenate([theta, beta])``.

    Notes
    -----
    The ``u_only`` direction is inverted relative to ``nagq``:

    - ``nagq=0`` → ``u_only=False``. Stage 0 outer optimizer searches over θ,
      so PIRLS must produce a joint (β, u) solve for each candidate θ.
    - ``nagq>0`` → ``u_only=True``. Stage 1 outer optimizer searches over
      (θ, β); β is folded into the offset (lmer.R:347 trick), so PIRLS only
      needs to update u.

    This mirrors lme4's C++ ``pwrssUpdate(rp, pp, ::Rf_asInteger(nAGQ_), ...)``
    (external.cpp:386) which casts the integer ``nAGQ`` to the bool ``uOnly``.
    The R fallback ``glmerPwrssUpdate`` has the opposite convention
    (``uOnly <- nAGQ == 0L``, lmer.R:447) — that's a latent bug in the
    seldom-exercised ``compDev=FALSE`` branch; the C++ behaviour is canonical.
    """
    if nagq < 0:
        raise ValueError(f"nagq must be >= 0, got {nagq}")

    # lp0 — the offset-free linear predictor at the current pred state.
    # Each devfun call resets resp.update_mu(lp0) so PIRLS sees a fixed
    # starting η across optimizer calls (lmer.R:333, 344). Snapshot via
    # .copy() since lin_pred returns a fresh array, but be explicit.
    lp0 = pred.lin_pred(1.0).copy()
    u_only = nagq > 0

    if nagq == 0:
        def devfun_theta(theta: np.ndarray) -> float:
            resp.update_mu(lp0)
            pred.set_theta(np.asarray(theta, dtype=float))
            _pwrss_update(
                pred, resp,
                u_only=u_only, tol=tol_pwrss,
                maxit=maxit_pwrss, verbose=verbose,
            )
            # Refresh weights once more so post-fit reads see a state
            # consistent with the final μ (mkdevfun lmer.R:337).
            resp.update_weights()
            return resp.laplace(
                pred.log_det_l_sq, pred.log_det_rx_sq, pred.sqr_l_u(1.0),
            )
        return devfun_theta

    # nagq > 0 — Stage 1 closure. Take the current resp.offset as
    # base_offset; the outer optimizer's β slice is added to it via X·β
    # before each PIRLS run (lmer.R:347-348, modular.R:996).
    base_offset = resp.offset.copy()
    n_theta = len(pred.theta)
    # nagq > 1 → the Stage-1 closure returns the adaptive Gauss-Hermite
    # deviance (port of updateGlmerDevfun + glmerAGQ) instead of the Laplace
    # value. Requires the precomputed GH rule, the per-obs grouping codes, and
    # the level count (a single scalar RE — enforced by the caller).
    is_agq = nagq > 1
    if is_agq and (gqmat is None or fac is None or n_levels is None):
        raise ValueError(
            "nagq > 1 requires gqmat, fac, and n_levels (the AGQ rule and "
            "single-scalar-RE grouping)"
        )

    def devfun_theta_beta(par: np.ndarray) -> float:
        par = np.asarray(par, dtype=float)
        theta = par[:n_theta]
        spars = par[n_theta:]
        # Order matters: reset offset → reset μ → THEN install the new
        # X·β offset. lme4 (lmer.R:343-348) leaves μ at
        # ``linkinv(baseOffset + lp0)`` deliberately, even though the new
        # offset is ``baseOffset + X·β`` — so the first PIRLS iteration's
        # working weights come from a μ that excludes ``X·β``, while the
        # in-loop ``update_mu`` then computes the next μ from
        # ``linkinv(new_offset + linPred)``. Swapping the order changes the
        # iteration trajectory and produces a ~1e-4 mismatch in ``ldL2``.
        resp.offset = base_offset.copy()
        resp.update_mu(lp0)
        if len(spars) > 0:
            resp.offset = base_offset + pred.X @ spars
        pred.set_theta(theta)
        _pwrss_update(
            pred, resp,
            u_only=u_only, tol=tol_pwrss,
            maxit=maxit_pwrss, verbose=verbose,
        )
        resp.update_weights()
        if is_agq:
            return _glmm_agq_deviance(pred, resp, gqmat, fac, n_levels)
        return resp.laplace(
            pred.log_det_l_sq, pred.log_det_rx_sq, pred.sqr_l_u(1.0),
        )
    return devfun_theta_beta


# ----------------------------------------------------------------------
# Bound-constrained BOBYQA — port of ``minqa``'s Fortran (M. J. D. Powell).
#
# lme4's default GLMM optimizer chain is ``c("bobyqa", "Nelder_Mead")``
# (glmerControl). Stage 0 (θ-only) runs BOBYQA; Stage 1 (θ+β) runs
# Nelder-Mead. To match R's converged ``(θ̂, β̂)`` byte-for-byte, hea Stage 0
# must therefore run the same BOBYQA — not a different derivative-free
# method (e.g. Py-BOBYQA, which is a fresh Cartis/Roberts implementation,
# not a Fortran port).
#
# The port is line-by-line from ``/tmp/minqa_src/minqa/src/*.f``:
#
#   bobyqa.f        →  _bobyqa_driver           (workspace + initial X clamp)
#   bobyqb.f        →  _bobyqa_bobyqb           (main iteration loop)
#   prelim.f        →  _bobyqa_prelim           (initial interpolation set)
#   trsbox.f        →  _bobyqa_trsbox           (trust-region step in box)
#   altmov.f        →  _bobyqa_altmov           (alternative geometry step)
#   rescue.f        →  _bobyqa_rescue           (interpolation-set rescue)
#   updatebobyqa.f  →  _bobyqa_update           (BMAT/ZMAT update)
#
# Fortran conventions kept verbatim for fidelity:
#
# - **1-indexed arrays**. All internal arrays are allocated with shape
#   ``(n+1,)`` or ``(npt+1, n+1)`` etc. Element ``[0]`` is unused. Loops
#   use ``range(1, n+1)``. This lets every Python statement mirror its
#   Fortran origin literally (``XPT(K,J)`` ⇔ ``xpt[K, J]``).
# - **Operation order is preserved**. The Fortran source ordering of
#   additions/subtractions is the only meaningful "implementation"
#   (BOBYQA is sensitive to rounding-error accumulation around small
#   denominators in the Lagrange-function denominator updates), so each
#   reduction, dot product, and sum is unrolled rather than vectorized.
# - **GOTOs become a state-string dispatch**. ``bobyqb.f`` has a tangled
#   GOTO graph (labels 20/60/90/190/210/230/350/360/650/680/720); the
#   port uses ``state = "L60"`` etc. in a ``while True`` loop. This is
#   uglier than structured control flow but maps directly to the
#   Fortran labels — diffing this code against the .f source is the
#   verification strategy.
# - **Known minqa quirks reproduced verbatim**: ``altmov.f`` resets
#   ``IBDSAV=0`` between the K-loop and the XNEW construction
#   (line 178), making the subsequent ``IBDSAV<0`` / ``>0`` boundary
#   handling unreachable. ``rescue.f`` calls ``CALFUN(N,X,IPRINT)``
#   inside its second loop (line 372) although ``X`` is not in the
#   RESCUE argument list — in F77 this picks up whatever the caller's
#   ``X`` slot happens to hold; the apparent intent is the freshly
#   populated ``W(1..N)``, which is what we pass. Both quirks affect
#   only seldom-triggered branches, but for bit-by-bit match against
#   minqa we follow the Fortran exactly.
#
# References:
# - ``/tmp/minqa_src/minqa/src/*.f`` — F77 source (minqa 1.2.8).
# - Powell, M. J. D. (2009), "The BOBYQA algorithm for bound constrained
#   optimization without derivatives", DAMTP technical report 2009/NA06.

def _bobyqa_update(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew, w):
    """Update BMAT and ZMAT for the new KNEW-th interpolation point.

    In-place port of ``updatebobyqa.f``. The vector ``vlag`` has length
    ``n + npt``, BETA is the parameter from the Powell 2006 NEWUOA paper
    eq. (4.11), DENOM is the denominator of the updating formula, and
    ``w`` is working space (first NDIM elements used).
    """
    ONE = 1.0
    ZERO = 0.0
    nptm = npt - n - 1
    # ZTEST = 1e-20 * max(|ZMAT|) — threshold for treating ZMAT entries as 0.
    ztest = ZERO
    for k in range(1, npt + 1):
        for j in range(1, nptm + 1):
            ztest = max(ztest, abs(zmat[k, j]))
    ztest = 1.0e-20 * ztest
    #
    # Apply the rotations that put zeros in the KNEW-th row of ZMAT.
    #
    jl = 1  # noqa: F841 — kept for fidelity with the F77 source
    for j in range(2, nptm + 1):
        if abs(zmat[knew, j]) > ztest:
            temp = np.sqrt(zmat[knew, 1] ** 2 + zmat[knew, j] ** 2)
            tempa = zmat[knew, 1] / temp
            tempb = zmat[knew, j] / temp
            for i in range(1, npt + 1):
                temp = tempa * zmat[i, 1] + tempb * zmat[i, j]
                zmat[i, j] = tempa * zmat[i, j] - tempb * zmat[i, 1]
                zmat[i, 1] = temp
        zmat[knew, j] = ZERO
    #
    # Put the first NPT components of the KNEW-th column of HLAG into W,
    # and calculate the parameters of the updating formula.
    #
    for i in range(1, npt + 1):
        w[i] = zmat[knew, 1] * zmat[i, 1]
    alpha = w[knew]
    tau = vlag[knew]
    vlag[knew] = vlag[knew] - ONE
    #
    # Complete the updating of ZMAT.
    #
    temp = np.sqrt(denom)
    tempb = zmat[knew, 1] / temp
    tempa = tau / temp
    for i in range(1, npt + 1):
        zmat[i, 1] = tempa * zmat[i, 1] - tempb * vlag[i]
    #
    # Finally, update the matrix BMAT.
    #
    for j in range(1, n + 1):
        jp = npt + j
        w[jp] = bmat[knew, j]
        tempa = (alpha * vlag[jp] - tau * w[jp]) / denom
        tempb = (-beta * w[jp] - tau * vlag[jp]) / denom
        for i in range(1, jp + 1):
            bmat[i, j] = bmat[i, j] + tempa * vlag[i] + tempb * w[i]
            if i > npt:
                bmat[jp, i - npt] = bmat[i, j]


def _bobyqa_prelim(calfun, n, npt, x, xl, xu, rhobeg, maxfun,
                   xbase, xpt, fval, gopt, hq, pq, bmat, zmat, ndim, sl, su):
    """Initialize XBASE, XPT, FVAL, GOPT, HQ, PQ, BMAT, ZMAT.

    Line-by-line port of ``prelim.f``. Maintains NF (number of CALFUN
    evaluations) and KOPT (index of best-so-far interpolation point).
    Returns ``(nf, kopt)``.
    """
    HALF = 0.5
    ONE = 1.0
    TWO = 2.0
    ZERO = 0.0
    rhosq = rhobeg * rhobeg
    recip = ONE / rhosq
    np_ = n + 1
    #
    # Set XBASE = X, zero XPT, BMAT, HQ, PQ, ZMAT.
    #
    for j in range(1, n + 1):
        xbase[j] = x[j]
        for k in range(1, npt + 1):
            xpt[k, j] = ZERO
        for i in range(1, ndim + 1):
            bmat[i, j] = ZERO
    for ih in range(1, (n * np_) // 2 + 1):
        hq[ih] = ZERO
    for k in range(1, npt + 1):
        pq[k] = ZERO
        for j in range(1, npt - np_ + 1):
            zmat[k, j] = ZERO
    #
    # Build initial interpolation set point by point.
    #
    nf = 0
    fbeg = 0.0
    stepa = 0.0
    stepb = 0.0
    ipt = 0
    jpt = 0
    kopt = 1
    while True:  # outer loop: GOTO 50
        nfm = nf
        nfx = nf - n
        nf = nf + 1
        if nfm <= 2 * n:
            if 1 <= nfm <= n:
                stepa = rhobeg
                if su[nfm] == ZERO:
                    stepa = -stepa
                xpt[nf, nfm] = stepa
            elif nfm > n:
                stepa = xpt[nf - n, nfx]
                stepb = -rhobeg
                if sl[nfx] == ZERO:
                    stepb = min(TWO * rhobeg, su[nfx])
                if su[nfx] == ZERO:
                    stepb = max(-TWO * rhobeg, sl[nfx])
                xpt[nf, nfx] = stepb
        else:
            itemp = (nfm - np_) // n
            jpt = nfm - itemp * n - n
            ipt = jpt + itemp
            if ipt > n:
                itemp = jpt
                jpt = ipt - n
                ipt = itemp
            xpt[nf, ipt] = xpt[ipt + 1, ipt]
            xpt[nf, jpt] = xpt[jpt + 1, jpt]
        #
        # Calculate the next value of F.
        #
        for j in range(1, n + 1):
            x[j] = min(max(xl[j], xbase[j] + xpt[nf, j]), xu[j])
            if xpt[nf, j] == sl[j]:
                x[j] = xl[j]
            if xpt[nf, j] == su[j]:
                x[j] = xu[j]
        f = calfun(x[1:n + 1])
        fval[nf] = f
        if nf == 1:
            fbeg = f
            kopt = 1
        elif f < fval[kopt]:
            kopt = nf
        #
        # Set the nonzero initial elements of BMAT and the quadratic model.
        #
        if nf <= 2 * n + 1:
            if 2 <= nf <= n + 1:
                gopt[nfm] = (f - fbeg) / stepa
                if npt < nf + n:
                    bmat[1, nfm] = -ONE / stepa
                    bmat[nf, nfm] = ONE / stepa
                    bmat[npt + nfm, nfm] = -HALF * rhosq
            elif nf >= n + 2:
                ih = (nfx * (nfx + 1)) // 2
                temp = (f - fbeg) / stepb
                diff = stepb - stepa
                hq[ih] = TWO * (temp - gopt[nfx]) / diff
                gopt[nfx] = (gopt[nfx] * stepb - temp * stepa) / diff
                if stepa * stepb < ZERO:
                    if f < fval[nf - n]:
                        fval[nf] = fval[nf - n]
                        fval[nf - n] = f
                        if kopt == nf:
                            kopt = nf - n
                        xpt[nf - n, nfx] = stepb
                        xpt[nf, nfx] = stepa
                bmat[1, nfx] = -(stepa + stepb) / (stepa * stepb)
                bmat[nf, nfx] = -HALF / xpt[nf - n, nfx]
                bmat[nf - n, nfx] = -bmat[1, nfx] - bmat[nf, nfx]
                zmat[1, nfx] = np.sqrt(TWO) / (stepa * stepb)
                zmat[nf, nfx] = np.sqrt(HALF) / rhosq
                zmat[nf - n, nfx] = -zmat[1, nfx] - zmat[nf, nfx]
        #
        # Set the off-diagonal second derivatives.
        #
        else:
            ih = (ipt * (ipt - 1)) // 2 + jpt
            zmat[1, nfx] = recip
            zmat[nf, nfx] = recip
            zmat[ipt + 1, nfx] = -recip
            zmat[jpt + 1, nfx] = -recip
            temp = xpt[nf, ipt] * xpt[nf, jpt]
            hq[ih] = (fbeg - fval[ipt + 1] - fval[jpt + 1] + f) / temp
        if not (nf < npt and nf < maxfun):
            break
    return nf, kopt


def _bobyqa_altmov(n, npt, xpt, xopt, bmat, zmat, ndim, sl, su, kopt,
                   knew, adelt, xnew, xalt, glag, hcol, w):
    """Compute alternative geometry step. Port of ``altmov.f``.

    Returns ``(alpha, cauchy)``. Mutates ``xnew``, ``xalt``, ``glag``,
    ``hcol``, ``w``.
    """
    HALF = 0.5
    ONE = 1.0
    ZERO = 0.0
    CONST = ONE + np.sqrt(2.0)
    #
    # Set HCOL to the leading elements of the KNEW-th column of H.
    #
    for k in range(1, npt + 1):
        hcol[k] = ZERO
    for j in range(1, npt - n - 1 + 1):
        temp = zmat[knew, j]
        for k in range(1, npt + 1):
            hcol[k] = hcol[k] + temp * zmat[k, j]
    alpha = hcol[knew]
    ha = HALF * alpha
    #
    # Calculate the gradient of the KNEW-th Lagrange function at XOPT.
    #
    for i in range(1, n + 1):
        glag[i] = bmat[knew, i]
    for k in range(1, npt + 1):
        temp = ZERO
        for j in range(1, n + 1):
            temp = temp + xpt[k, j] * xopt[j]
        temp = hcol[k] * temp
        for i in range(1, n + 1):
            glag[i] = glag[i] + temp * xpt[k, i]
    #
    # Search for a large denominator along lines through XOPT.
    #
    presav = ZERO
    ksav = kopt  # init for safety; overwritten on first PREDSQ > PRESAV
    stpsav = ZERO
    ibdsav = 0
    for k in range(1, npt + 1):
        if k == kopt:
            continue  # GOTO 80 — skip body
        dderiv = ZERO
        distsq = ZERO
        for i in range(1, n + 1):
            temp = xpt[k, i] - xopt[i]
            dderiv = dderiv + glag[i] * temp
            distsq = distsq + temp * temp
        subd = adelt / np.sqrt(distsq)
        slbd = -subd
        ilbd = 0
        iubd = 0
        sumin = min(ONE, subd)
        #
        # Revise SLBD and SUBD because of SL/SU.
        #
        for i in range(1, n + 1):
            temp = xpt[k, i] - xopt[i]
            if temp > ZERO:
                if slbd * temp < sl[i] - xopt[i]:
                    slbd = (sl[i] - xopt[i]) / temp
                    ilbd = -i
                if subd * temp > su[i] - xopt[i]:
                    subd = max(sumin, (su[i] - xopt[i]) / temp)
                    iubd = i
            elif temp < ZERO:
                if slbd * temp > su[i] - xopt[i]:
                    slbd = (su[i] - xopt[i]) / temp
                    ilbd = i
                if subd * temp < sl[i] - xopt[i]:
                    subd = max(sumin, (sl[i] - xopt[i]) / temp)
                    iubd = -i
        #
        # K == KNEW path.
        #
        if k == knew:
            diff = dderiv - ONE
            step = slbd
            vlag_local = slbd * (dderiv - slbd * diff)
            isbd = ilbd
            temp = subd * (dderiv - subd * diff)
            if abs(temp) > abs(vlag_local):
                step = subd
                vlag_local = temp
                isbd = iubd
            tempd = HALF * dderiv
            tempa = tempd - diff * slbd
            tempb = tempd - diff * subd
            if tempa * tempb < ZERO:
                temp = tempd * tempd / diff
                if abs(temp) > abs(vlag_local):
                    step = tempd / diff
                    vlag_local = temp
                    isbd = 0
        else:
            #
            # Other lines through XOPT.
            #
            step = slbd
            vlag_local = slbd * (ONE - slbd)
            isbd = ilbd
            temp = subd * (ONE - subd)
            if abs(temp) > abs(vlag_local):
                step = subd
                vlag_local = temp
                isbd = iubd
            if subd > HALF:
                if abs(vlag_local) < 0.25:
                    step = HALF
                    vlag_local = 0.25
                    isbd = 0
            vlag_local = vlag_local * dderiv
        #
        # PREDSQ for this line, maintain PRESAV.
        #
        temp = step * (ONE - step) * distsq
        predsq = vlag_local * vlag_local * (vlag_local * vlag_local + ha * temp * temp)
        if predsq > presav:
            presav = predsq
            ksav = k
            stpsav = step
            ibdsav = isbd
    #
    # Construct XNEW. The IBDSAV=0 here is verbatim from altmov.f:178 — a
    # known quirk that makes the next two bound-snapping branches dead code.
    # Preserved for bit-by-bit match with minqa.
    #
    ibdsav = 0
    for i in range(1, n + 1):
        temp = xopt[i] + stpsav * (xpt[ksav, i] - xopt[i])
        xnew[i] = max(sl[i], min(su[i], temp))
    if ibdsav < 0:
        xnew[-ibdsav] = sl[-ibdsav]
    if ibdsav > 0:
        xnew[ibdsav] = su[ibdsav]
    #
    # Constrained Cauchy step iteration (labels 100 / 120). IFLAG=0 runs
    # with +GLAG, IFLAG=1 with -GLAG; XALT is the better of the two.
    #
    bigstp = adelt + adelt
    iflag = 0
    csave = 0.0
    while True:  # label 100
        wfixsq = ZERO
        ggfree = ZERO
        for i in range(1, n + 1):
            w[i] = ZERO
            tempa = min(xopt[i] - sl[i], glag[i])
            tempb = max(xopt[i] - su[i], glag[i])
            if tempa > ZERO or tempb < ZERO:
                w[i] = bigstp
                ggfree = ggfree + glag[i] ** 2
        cauchy = 0.0
        step = ZERO
        if ggfree == ZERO:
            cauchy = ZERO
            return alpha, cauchy
        #
        # Try to fix more components of W (label 120 loop).
        #
        while True:  # label 120
            temp = adelt * adelt - wfixsq
            if temp > ZERO:
                wsqsav = wfixsq
                step = np.sqrt(temp / ggfree)
                ggfree = ZERO
                for i in range(1, n + 1):
                    if w[i] == bigstp:
                        temp = xopt[i] - step * glag[i]
                        if temp <= sl[i]:
                            w[i] = sl[i] - xopt[i]
                            wfixsq = wfixsq + w[i] ** 2
                        elif temp >= su[i]:
                            w[i] = su[i] - xopt[i]
                            wfixsq = wfixsq + w[i] ** 2
                        else:
                            ggfree = ggfree + glag[i] ** 2
                if wfixsq > wsqsav and ggfree > ZERO:
                    continue
            break
        #
        # Set remaining components of W and XALT.
        #
        gw = ZERO
        for i in range(1, n + 1):
            if w[i] == bigstp:
                w[i] = -step * glag[i]
                xalt[i] = max(sl[i], min(su[i], xopt[i] + w[i]))
            elif w[i] == ZERO:
                xalt[i] = xopt[i]
            elif glag[i] > ZERO:
                xalt[i] = sl[i]
            else:
                xalt[i] = su[i]
            gw = gw + glag[i] * w[i]
        #
        # Curvature of KNEW-th Lagrange function along W.
        #
        curv = ZERO
        for k in range(1, npt + 1):
            temp = ZERO
            for j in range(1, n + 1):
                temp = temp + xpt[k, j] * w[j]
            curv = curv + hcol[k] * temp * temp
        if iflag == 1:
            curv = -curv
        if curv > -gw and curv < -CONST * gw:
            scale = -gw / curv
            for i in range(1, n + 1):
                temp = xopt[i] + scale * w[i]
                xalt[i] = max(sl[i], min(su[i], temp))
            cauchy = (HALF * gw * scale) ** 2
        else:
            cauchy = (gw + HALF * curv) ** 2
        #
        # IFLAG=0 → flip GLAG and repeat with -GLAG; pick the larger CAUCHY.
        #
        if iflag == 0:
            for i in range(1, n + 1):
                glag[i] = -glag[i]
                w[n + i] = xalt[i]
            csave = cauchy
            iflag = 1
            continue
        if csave > cauchy:
            for i in range(1, n + 1):
                xalt[i] = w[n + i]
            cauchy = csave
        return alpha, cauchy


def _bobyqa_trsbox(n, npt, xpt, xopt, gopt, hq, pq, sl, su, delta,
                   xnew, d, gnew, xbdi, s, hs, hred):
    """Trust-region step within bounds. Port of ``trsbox.f``.

    Returns ``(dsq, crvmin)``. Mutates ``xnew, d, gnew, xbdi, s, hs, hred``.
    """
    HALF = 0.5
    ONE = 1.0
    ONEMIN = -1.0
    ZERO = 0.0
    #
    # Initial: set XBDI from active bounds and GOPT signs; D=0, GNEW=GOPT.
    #
    iterc = 0
    nact = 0
    sqstp = ZERO  # noqa: F841
    for i in range(1, n + 1):
        xbdi[i] = ZERO
        if xopt[i] <= sl[i]:
            if gopt[i] >= ZERO:
                xbdi[i] = ONEMIN
        elif xopt[i] >= su[i]:
            if gopt[i] <= ZERO:
                xbdi[i] = ONE
        if xbdi[i] != ZERO:
            nact = nact + 1
        d[i] = ZERO
        gnew[i] = gopt[i]
    delsq = delta * delta
    qred = ZERO
    crvmin = ONEMIN
    #
    # State-machine dispatch for the GOTO graph (labels 20/30/50/90/100/120/150/190/210).
    # 210 is the "multiply S by HQ, store in HS" routine that's reached from
    # 50/120/150; it dispatches back depending on CRVMIN / ITERC.
    #
    iact = 0
    itcsav = -1
    stepsq = 0.0
    gredsq = 0.0
    ggsav = 0.0
    blen = 0.0
    stplen = 0.0
    sredg = 0.0
    dredg = 0.0
    dredsq = 0.0
    angbd = 0.0
    sdec = 0.0
    angt = 0.0
    cth = 0.0
    sth = 0.0
    xsav = 0.0
    itermax = 0

    state = 'L20'
    while True:
        if state == 'L20':
            beta = ZERO
            state = 'L30'
        elif state == 'L30':
            stepsq = ZERO
            for i in range(1, n + 1):
                if xbdi[i] != ZERO:
                    s[i] = ZERO
                elif beta == ZERO:
                    s[i] = -gnew[i]
                else:
                    s[i] = beta * s[i] - gnew[i]
                stepsq = stepsq + s[i] ** 2
            if stepsq == ZERO:
                state = 'L190'
                continue
            if beta == ZERO:
                gredsq = stepsq
                itermax = iterc + n - nact
            if gredsq * delsq <= 1.0e-4 * qred * qred:
                state = 'L190'
                continue
            state = 'L210'
        elif state == 'L50':
            resid = delsq
            ds = ZERO
            shs = ZERO
            for i in range(1, n + 1):
                if xbdi[i] == ZERO:
                    resid = resid - d[i] ** 2
                    ds = ds + s[i] * d[i]
                    shs = shs + s[i] * hs[i]
            if resid <= ZERO:
                state = 'L90'
                continue
            temp = np.sqrt(stepsq * resid + ds * ds)
            if ds < ZERO:
                blen = (temp - ds) / stepsq
            else:
                blen = resid / (temp + ds)
            stplen = blen
            if shs > ZERO:
                stplen = min(blen, gredsq / shs)
            #
            # Reduce STPLEN to preserve simple bounds; IACT is the new fixed var.
            #
            iact = 0
            for i in range(1, n + 1):
                if s[i] != ZERO:
                    xsum = xopt[i] + d[i]
                    if s[i] > ZERO:
                        temp = (su[i] - xsum) / s[i]
                    else:
                        temp = (sl[i] - xsum) / s[i]
                    if temp < stplen:
                        stplen = temp
                        iact = i
            #
            # Update CRVMIN, GNEW, D. SDEC is the decrease in Q.
            #
            sdec = ZERO
            if stplen > ZERO:
                iterc = iterc + 1
                temp = shs / stepsq
                if iact == 0 and temp > ZERO:
                    crvmin = min(crvmin, temp)
                    if crvmin == ONEMIN:
                        crvmin = temp
                ggsav = gredsq
                gredsq = ZERO
                for i in range(1, n + 1):
                    gnew[i] = gnew[i] + stplen * hs[i]
                    if xbdi[i] == ZERO:
                        gredsq = gredsq + gnew[i] ** 2
                    d[i] = d[i] + stplen * s[i]
                sdec = max(stplen * (ggsav - HALF * stplen * shs), ZERO)
                qred = qred + sdec
            #
            # Restart CG if hit a new bound.
            #
            if iact > 0:
                nact = nact + 1
                xbdi[iact] = ONE
                if s[iact] < ZERO:
                    xbdi[iact] = ONEMIN
                delsq = delsq - d[iact] ** 2
                if delsq <= ZERO:
                    state = 'L90'
                    continue
                state = 'L20'
                continue
            #
            # STPLEN < BLEN: more CG steps or return.
            #
            if stplen < blen:
                if iterc == itermax:
                    state = 'L190'
                    continue
                if sdec <= 0.01 * qred:
                    state = 'L190'
                    continue
                beta = gredsq / ggsav
                state = 'L30'
                continue
            state = 'L90'
        elif state == 'L90':
            crvmin = ZERO
            state = 'L100'
        elif state == 'L100':
            if nact >= n - 1:
                state = 'L190'
                continue
            dredsq = ZERO
            dredg = ZERO
            gredsq = ZERO
            for i in range(1, n + 1):
                if xbdi[i] == ZERO:
                    dredsq = dredsq + d[i] ** 2
                    dredg = dredg + d[i] * gnew[i]
                    gredsq = gredsq + gnew[i] ** 2
                    s[i] = d[i]
                else:
                    s[i] = ZERO
            itcsav = iterc
            state = 'L210'
        elif state == 'L120':
            iterc = iterc + 1
            temp = gredsq * dredsq - dredg * dredg
            if temp <= 1.0e-4 * qred * qred:
                state = 'L190'
                continue
            temp = np.sqrt(temp)
            for i in range(1, n + 1):
                if xbdi[i] == ZERO:
                    s[i] = (dredg * d[i] - dredsq * gnew[i]) / temp
                else:
                    s[i] = ZERO
            sredg = -temp
            #
            # ANGBD from bounds.
            #
            angbd = ONE
            iact = 0
            jumped_to_100 = False
            for i in range(1, n + 1):
                if xbdi[i] == ZERO:
                    tempa = xopt[i] + d[i] - sl[i]
                    tempb = su[i] - xopt[i] - d[i]
                    if tempa <= ZERO:
                        nact = nact + 1
                        xbdi[i] = ONEMIN
                        jumped_to_100 = True
                        break
                    elif tempb <= ZERO:
                        nact = nact + 1
                        xbdi[i] = ONE
                        jumped_to_100 = True
                        break
                    ratio = ONE  # noqa: F841
                    ssq = d[i] ** 2 + s[i] ** 2
                    temp = ssq - (xopt[i] - sl[i]) ** 2
                    if temp > ZERO:
                        temp = np.sqrt(temp) - s[i]
                        if angbd * temp > tempa:
                            angbd = tempa / temp
                            iact = i
                            xsav = ONEMIN
                    temp = ssq - (su[i] - xopt[i]) ** 2
                    if temp > ZERO:
                        temp = np.sqrt(temp) + s[i]
                        if angbd * temp > tempb:
                            angbd = tempb / temp
                            iact = i
                            xsav = ONE
            if jumped_to_100:
                state = 'L100'
                continue
            state = 'L210'
        elif state == 'L150':
            shs = ZERO
            dhs = ZERO
            dhd = ZERO
            for i in range(1, n + 1):
                if xbdi[i] == ZERO:
                    shs = shs + s[i] * hs[i]
                    dhs = dhs + d[i] * hs[i]
                    dhd = dhd + d[i] * hred[i]
            #
            # Search for greatest reduction over equally-spaced ANGT.
            #
            redmax = ZERO
            isav = 0
            redsav = ZERO
            rdprev = 0.0
            rdnext = 0.0
            iu = int(17.0 * angbd + 3.1)
            for i in range(1, iu + 1):
                angt = angbd * float(i) / float(iu)
                sth = (angt + angt) / (ONE + angt * angt)
                temp = shs + angt * (angt * dhd - dhs - dhs)
                rednew = sth * (angt * dredg - sredg - HALF * sth * temp)
                if rednew > redmax:
                    redmax = rednew
                    isav = i
                    rdprev = redsav
                elif i == isav + 1:
                    rdnext = rednew
                redsav = rednew
            if isav == 0:
                state = 'L190'
                continue
            if isav < iu:
                temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext)
                angt = angbd * (float(isav) + HALF * temp) / float(iu)
            cth = (ONE - angt * angt) / (ONE + angt * angt)
            sth = (angt + angt) / (ONE + angt * angt)
            temp = shs + angt * (angt * dhd - dhs - dhs)
            sdec = sth * (angt * dredg - sredg - HALF * sth * temp)
            if sdec <= ZERO:
                state = 'L190'
                continue
            #
            # Update GNEW, D, HRED. Fix variable if angle was bound-restricted.
            #
            dredg = ZERO
            gredsq = ZERO
            for i in range(1, n + 1):
                gnew[i] = gnew[i] + (cth - ONE) * hred[i] + sth * hs[i]
                if xbdi[i] == ZERO:
                    d[i] = cth * d[i] + sth * s[i]
                    dredg = dredg + d[i] * gnew[i]
                    gredsq = gredsq + gnew[i] ** 2
                hred[i] = cth * hred[i] + sth * hs[i]
            qred = qred + sdec
            if iact > 0 and isav == iu:
                nact = nact + 1
                xbdi[iact] = xsav
                state = 'L100'
                continue
            if sdec > 0.01 * qred:
                state = 'L120'
                continue
            state = 'L190'
        elif state == 'L190':
            dsq = ZERO
            for i in range(1, n + 1):
                xnew[i] = max(min(xopt[i] + d[i], su[i]), sl[i])
                if xbdi[i] == ONEMIN:
                    xnew[i] = sl[i]
                if xbdi[i] == ONE:
                    xnew[i] = su[i]
                d[i] = xnew[i] - xopt[i]
                dsq = dsq + d[i] ** 2
            return dsq, crvmin
        elif state == 'L210':
            # Multiply S by HQ, store in HS.
            ih = 0
            for j in range(1, n + 1):
                hs[j] = ZERO
                for i in range(1, j + 1):
                    ih = ih + 1
                    if i < j:
                        hs[j] = hs[j] + hq[ih] * s[i]
                    hs[i] = hs[i] + hq[ih] * s[j]
            for k in range(1, npt + 1):
                if pq[k] != ZERO:
                    temp = ZERO
                    for j in range(1, n + 1):
                        temp = temp + xpt[k, j] * s[j]
                    temp = temp * pq[k]
                    for i in range(1, n + 1):
                        hs[i] = hs[i] + temp * xpt[k, i]
            if crvmin != ZERO:
                state = 'L50'
                continue
            if iterc > itcsav:
                state = 'L150'
                continue
            for i in range(1, n + 1):
                hred[i] = hs[i]
            state = 'L120'
        else:
            raise RuntimeError(f"trsbox: unknown state {state!r}")


def _bobyqa_rescue(calfun, n, npt, xl, xu, maxfun, xbase, xpt, fval,
                   xopt, gopt, hq, pq, bmat, zmat, ndim, sl, su, nf, delta,
                   kopt, vlag, ptsaux, ptsid, w):
    """Rescue ill-conditioned interpolation set. Port of ``rescue.f``.

    Returns ``(nf, kopt)`` (nf set to -1 if MAXFUN exhausted).
    """
    HALF = 0.5
    ONE = 1.0
    ZERO = 0.0
    np_ = n + 1
    sfrac = HALF / float(np_)
    nptm = npt - np_
    #
    # Shift XPT so XOPT becomes origin; clear ZMAT; compute W(NDIM+K) distances.
    #
    sumpq = ZERO
    winc = ZERO
    for k in range(1, npt + 1):
        distsq = ZERO
        for j in range(1, n + 1):
            xpt[k, j] = xpt[k, j] - xopt[j]
            distsq = distsq + xpt[k, j] ** 2
        sumpq = sumpq + pq[k]
        w[ndim + k] = distsq
        winc = max(winc, distsq)
        for j in range(1, nptm + 1):
            zmat[k, j] = ZERO
    #
    # Update HQ for XBASE shift.
    #
    ih = 0
    for j in range(1, n + 1):
        w[j] = HALF * sumpq * xopt[j]
        for k in range(1, npt + 1):
            w[j] = w[j] + pq[k] * xpt[k, j]
        for i in range(1, j + 1):
            ih = ih + 1
            hq[ih] = hq[ih] + w[i] * xopt[j] + w[j] * xopt[i]
    #
    # Shift XBASE, SL, SU, XOPT; clear BMAT; set PTSAUX.
    #
    for j in range(1, n + 1):
        xbase[j] = xbase[j] + xopt[j]
        sl[j] = sl[j] - xopt[j]
        su[j] = su[j] - xopt[j]
        xopt[j] = ZERO
        ptsaux[1, j] = min(delta, su[j])
        ptsaux[2, j] = max(-delta, sl[j])
        if ptsaux[1, j] + ptsaux[2, j] < ZERO:
            temp = ptsaux[1, j]
            ptsaux[1, j] = ptsaux[2, j]
            ptsaux[2, j] = temp
        if abs(ptsaux[2, j]) < HALF * abs(ptsaux[1, j]):
            ptsaux[2, j] = HALF * ptsaux[1, j]
        for i in range(1, ndim + 1):
            bmat[i, j] = ZERO
    #
    # Set provisional interpolation point identifiers PTSID, and nonzero
    # elements of BMAT and ZMAT.
    #
    ptsid[1] = sfrac
    for j in range(1, n + 1):
        jp = j + 1
        jpn = jp + n
        ptsid[jp] = float(j) + sfrac
        if jpn <= npt:
            ptsid[jpn] = float(j) / float(np_) + sfrac
            temp = ONE / (ptsaux[1, j] - ptsaux[2, j])
            bmat[jp, j] = -temp + ONE / ptsaux[1, j]
            bmat[jpn, j] = temp + ONE / ptsaux[2, j]
            bmat[1, j] = -bmat[jp, j] - bmat[jpn, j]
            zmat[1, j] = np.sqrt(2.0) / abs(ptsaux[1, j] * ptsaux[2, j])
            zmat[jp, j] = zmat[1, j] * ptsaux[2, j] * temp
            zmat[jpn, j] = -zmat[1, j] * ptsaux[1, j] * temp
        else:
            bmat[1, j] = -ONE / ptsaux[1, j]
            bmat[jp, j] = ONE / ptsaux[1, j]
            bmat[j + npt, j] = -HALF * ptsaux[1, j] ** 2
    #
    # Remaining identifiers — off-diagonal ZMAT.
    #
    if npt >= n + np_:
        for k in range(2 * np_, npt + 1):
            iw = int((float(k - np_) - HALF) / float(n))
            ip = k - np_ - iw * n
            iq = ip + iw
            if iq > n:
                iq = iq - n
            ptsid[k] = float(ip) + float(iq) / float(np_) + sfrac
            temp = ONE / (ptsaux[1, ip] * ptsaux[1, iq])
            zmat[1, k - np_] = temp
            zmat[ip + 1, k - np_] = -temp
            zmat[iq + 1, k - np_] = -temp
            zmat[k, k - np_] = temp
    nrem = npt
    kold = 1
    knew = kopt
    beta = 0.0
    denom = 0.0
    #
    # Reorder provisional points. Labels 80 / 120 / 260 / 350.
    #
    while True:  # label 80 — outer loop
        for j in range(1, n + 1):
            temp = bmat[kold, j]
            bmat[kold, j] = bmat[knew, j]
            bmat[knew, j] = temp
        for j in range(1, nptm + 1):
            temp = zmat[kold, j]
            zmat[kold, j] = zmat[knew, j]
            zmat[knew, j] = temp
        ptsid[kold] = ptsid[knew]
        ptsid[knew] = ZERO
        w[ndim + knew] = ZERO
        nrem = nrem - 1
        if knew != kopt:
            temp = vlag[kold]
            vlag[kold] = vlag[knew]
            vlag[knew] = temp
            _bobyqa_update(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew, w)
            if nrem == 0:
                return nf, kopt  # GOTO 350
            for k in range(1, npt + 1):
                w[ndim + k] = abs(w[ndim + k])
        #
        # Pick next KNEW (label 120).
        #
        retry_120 = True
        while retry_120:
            dsqmin = ZERO
            for k in range(1, npt + 1):
                if w[ndim + k] > ZERO:
                    if dsqmin == ZERO or w[ndim + k] < dsqmin:
                        knew = k
                        dsqmin = w[ndim + k]
            if dsqmin == ZERO:
                # GOTO 260: finalize new interpolation points.
                return _bobyqa_rescue_finalize(
                    calfun, n, npt, xl, xu, maxfun, xbase, xpt, fval,
                    gopt, hq, pq, bmat, zmat, ndim, sl, su, nf, kopt,
                    ptsaux, ptsid, w,
                )
            #
            # Form W-vector of the chosen original point.
            #
            for j in range(1, n + 1):
                w[npt + j] = xpt[knew, j]
            for k in range(1, npt + 1):
                sum_ = ZERO
                if k == kopt:
                    pass  # F77 CONTINUE
                elif ptsid[k] == ZERO:
                    for j in range(1, n + 1):
                        sum_ = sum_ + w[npt + j] * xpt[k, j]
                else:
                    ip = int(ptsid[k])
                    if ip > 0:
                        sum_ = w[npt + ip] * ptsaux[1, ip]
                    iq = int(float(np_) * ptsid[k] - float(ip * np_))
                    if iq > 0:
                        iw = 1
                        if ip == 0:
                            iw = 2
                        sum_ = sum_ + w[npt + iq] * ptsaux[iw, iq]
                w[k] = HALF * sum_ * sum_
            #
            # VLAG and BETA for the proposed reinstatement.
            #
            for k in range(1, npt + 1):
                sum_ = ZERO
                for j in range(1, n + 1):
                    sum_ = sum_ + bmat[k, j] * w[npt + j]
                vlag[k] = sum_
            beta = ZERO
            for j in range(1, nptm + 1):
                sum_ = ZERO
                for k in range(1, npt + 1):
                    sum_ = sum_ + zmat[k, j] * w[k]
                beta = beta - sum_ * sum_
                for k in range(1, npt + 1):
                    vlag[k] = vlag[k] + sum_ * zmat[k, j]
            bsum = ZERO
            distsq = ZERO
            for j in range(1, n + 1):
                sum_ = ZERO
                for k in range(1, npt + 1):
                    sum_ = sum_ + bmat[k, j] * w[k]
                jp = j + npt
                bsum = bsum + sum_ * w[jp]
                for ip in range(npt + 1, ndim + 1):
                    sum_ = sum_ + bmat[ip, j] * w[ip]
                bsum = bsum + sum_ * w[jp]
                vlag[jp] = sum_
                distsq = distsq + xpt[knew, j] ** 2
            beta = HALF * distsq * distsq + beta - bsum
            vlag[kopt] = vlag[kopt] + ONE
            #
            # KOLD by max DEN with rejection of small denominators.
            #
            denom = ZERO
            vlmxsq = ZERO
            for k in range(1, npt + 1):
                if ptsid[k] != ZERO:
                    hdiag = ZERO
                    for j in range(1, nptm + 1):
                        hdiag = hdiag + zmat[k, j] ** 2
                    den = beta * hdiag + vlag[k] ** 2
                    if den > denom:
                        kold = k
                        denom = den
                vlmxsq = max(vlmxsq, vlag[k] ** 2)
            if denom <= 1.0e-2 * vlmxsq:
                w[ndim + knew] = -w[ndim + knew] - winc
                continue  # GOTO 120
            retry_120 = False
        # falls back to GOTO 80


def _bobyqa_rescue_finalize(calfun, n, npt, xl, xu, maxfun, xbase, xpt, fval,
                            gopt, hq, pq, bmat, zmat, ndim, sl, su, nf, kopt,
                            ptsaux, ptsid, w):
    """Label-260 block of ``rescue.f``: evaluate F at remaining provisional
    points and absorb them into the quadratic model. Returns ``(nf, kopt)``.
    """
    HALF = 0.5
    ZERO = 0.0
    np_ = n + 1
    nptm = npt - np_
    fbase = fval[kopt]
    for kpt in range(1, npt + 1):
        if ptsid[kpt] == ZERO:
            continue
        if nf >= maxfun:
            nf = -1
            return nf, kopt
        ih = 0
        for j in range(1, n + 1):
            w[j] = xpt[kpt, j]
            xpt[kpt, j] = ZERO
            temp = pq[kpt] * w[j]
            for i in range(1, j + 1):
                ih = ih + 1
                hq[ih] = hq[ih] + temp * w[i]
        pq[kpt] = ZERO
        ip = int(ptsid[kpt])
        iq = int(float(np_) * ptsid[kpt] - float(ip * np_))
        xp = 0.0
        xq = 0.0
        if ip > 0:
            xp = ptsaux[1, ip]
            xpt[kpt, ip] = xp
        if iq > 0:
            xq = ptsaux[1, iq]
            if ip == 0:
                xq = ptsaux[2, iq]
            xpt[kpt, iq] = xq
        #
        # VQUAD = current model at the new point.
        #
        vquad = fbase
        ihp = 0
        ihq = 0
        if ip > 0:
            ihp = (ip + ip * ip) // 2
            vquad = vquad + xp * (gopt[ip] + HALF * xp * hq[ihp])
        if iq > 0:
            ihq = (iq + iq * iq) // 2
            vquad = vquad + xq * (gopt[iq] + HALF * xq * hq[ihq])
            if ip > 0:
                iw = max(ihp, ihq) - abs(ip - iq)
                vquad = vquad + xp * xq * hq[iw]
        for k in range(1, npt + 1):
            temp = ZERO
            if ip > 0:
                temp = temp + xp * xpt[k, ip]
            if iq > 0:
                temp = temp + xq * xpt[k, iq]
            vquad = vquad + HALF * pq[k] * temp * temp
        #
        # Evaluate F at the new interpolation point. The Fortran call is
        # ``F = CALFUN(N, X, IPRINT)`` but X is not in RESCUE's arg list —
        # we use W (which contains the clipped new point) as that's the
        # apparent intent.
        #
        for i in range(1, n + 1):
            w[i] = min(max(xl[i], xbase[i] + xpt[kpt, i]), xu[i])
            if xpt[kpt, i] == sl[i]:
                w[i] = xl[i]
            if xpt[kpt, i] == su[i]:
                w[i] = xu[i]
        nf = nf + 1
        f = calfun(w[1:n + 1])
        fval[kpt] = f
        if f < fval[kopt]:
            kopt = kpt
        diff = f - vquad
        #
        # Update quadratic model.
        #
        for i in range(1, n + 1):
            gopt[i] = gopt[i] + diff * bmat[kpt, i]
        for k in range(1, npt + 1):
            sum_ = ZERO
            for j in range(1, nptm + 1):
                sum_ = sum_ + zmat[k, j] * zmat[kpt, j]
            temp = diff * sum_
            if ptsid[k] == ZERO:
                pq[k] = pq[k] + temp
            else:
                ip_k = int(ptsid[k])
                iq_k = int(float(np_) * ptsid[k] - float(ip_k * np_))
                ihq_k = (iq_k * iq_k + iq_k) // 2
                if ip_k == 0:
                    hq[ihq_k] = hq[ihq_k] + temp * ptsaux[2, iq_k] ** 2
                else:
                    ihp_k = (ip_k * ip_k + ip_k) // 2
                    hq[ihp_k] = hq[ihp_k] + temp * ptsaux[1, ip_k] ** 2
                    if iq_k > 0:
                        hq[ihq_k] = hq[ihq_k] + temp * ptsaux[1, iq_k] ** 2
                        iw = max(ihp_k, ihq_k) - abs(iq_k - ip_k)
                        hq[iw] = hq[iw] + temp * ptsaux[1, ip_k] * ptsaux[1, iq_k]
        ptsid[kpt] = ZERO
    return nf, kopt


def _bobyqa_bobyqb(calfun, n, npt, x, xl, xu, rhobeg, rhoend, maxfun, sl, su,
                   stop=None):
    """Main BOBYQA iteration loop. Port of ``bobyqb.f``.

    Returns ``(x, f, nf, ierr)`` where IERR is 0 on normal exit,
    20/320/390/430 for the various Powell error codes.

    ``stop`` (default ``None``) optionally carries NLopt's stopping criteria
    (an object with ``.ftol(f, oldf)``). When supplied — the ``lmer`` path via
    :func:`_nlopt_ln_bobyqa` — the loop also terminates at a trust-region
    improvement whose ``F``-reduction is below ``ftol`` (NLopt ``bobyqa.c:2818``),
    reproducing lme4's default ``nloptwrap`` optimizer. ``None`` (glmer's minqa
    path) runs Powell's native rho→rhoend schedule unchanged.
    """
    HALF = 0.5
    ONE = 1.0
    TEN = 10.0
    TENTH = 0.1
    TWO = 2.0
    ZERO = 0.0
    np_ = n + 1
    nptm = npt - np_
    nh = (n * np_) // 2
    ndim = npt + n
    #
    # Allocate work arrays (1-indexed; element [0] unused).
    #
    xbase = np.zeros(n + 1)
    xpt = np.zeros((npt + 1, n + 1))
    fval = np.zeros(npt + 1)
    xopt = np.zeros(n + 1)
    gopt = np.zeros(n + 1)
    hq = np.zeros(nh + 1)
    pq = np.zeros(npt + 1)
    bmat = np.zeros((ndim + 1, n + 1))
    zmat = np.zeros((npt + 1, nptm + 1))
    xnew = np.zeros(n + 1)
    xalt = np.zeros(n + 1)
    d = np.zeros(n + 1)
    vlag = np.zeros(ndim + 1)
    w = np.zeros(3 * ndim + np_ + n + 1)  # generous; F77 W() length ≥ 3·NDIM
    ierr = 0
    #
    # PRELIM: initial XBASE, XPT, FVAL, GOPT, HQ, PQ, BMAT, ZMAT.
    #
    nf, kopt = _bobyqa_prelim(calfun, n, npt, x, xl, xu, rhobeg, maxfun,
                              xbase, xpt, fval, gopt, hq, pq,
                              bmat, zmat, ndim, sl, su)
    xoptsq = ZERO
    for i in range(1, n + 1):
        xopt[i] = xpt[kopt, i]
        xoptsq = xoptsq + xopt[i] ** 2
    fsave = fval[1]
    if nf < npt:
        ierr = 390
        # GOTO 720
        return _bobyqa_finalize(x, xl, xu, sl, su, xbase, xopt, fval, kopt,
                                fsave, n, nf, ierr)
    kbase = 1
    #
    # Settings for the iterative procedure.
    #
    rho = rhobeg
    delta = rho
    nresc = nf
    ntrits = 0
    diffa = ZERO
    diffb = ZERO
    diffc = 0.0
    itest = 0
    nfsav = nf
    knew = 0
    dnorm = 0.0
    dsq = 0.0
    crvmin = 0.0
    adelt = 0.0
    alpha = 0.0
    cauchy = 0.0
    beta = 0.0
    vquad = 0.0
    ratio = 0.0
    denom = 0.0
    f = fval[kopt]

    state = 'L20'
    while True:
        if state == 'L20':
            # Update GOPT if KOPT changed.
            if kopt != kbase:
                ih = 0
                for j in range(1, n + 1):
                    for i in range(1, j + 1):
                        ih = ih + 1
                        if i < j:
                            gopt[j] = gopt[j] + hq[ih] * xopt[i]
                        gopt[i] = gopt[i] + hq[ih] * xopt[j]
                if nf > npt:
                    for k in range(1, npt + 1):
                        temp = ZERO
                        for j in range(1, n + 1):
                            temp = temp + xpt[k, j] * xopt[j]
                        temp = pq[k] * temp
                        for i in range(1, n + 1):
                            gopt[i] = gopt[i] + temp * xpt[k, i]
            state = 'L60'
        elif state == 'L60':
            # Trust-region step.
            #
            # F77: TRSBOX(N,NPT,XPT,XOPT,GOPT,HQ,PQ,SL,SU,DELTA,XNEW,D,
            #             W,W(NP),W(NP+N),W(NP+2*N),W(NP+3*N),DSQ,CRVMIN)
            # The W partition gives GNEW, XBDI, S, HS, HRED.
            # Use private scratch arrays — F77 reuse of W() is just buffer
            # economy; results don't depend on overlap.
            gnew = np.zeros(n + 1)
            xbdi = np.zeros(n + 1)
            s_arr = np.zeros(n + 1)
            hs_arr = np.zeros(n + 1)
            hred = np.zeros(n + 1)
            # Cache W(1..N) for use in BDTEST below (XNEW snapshot vs SL/SU).
            for jj in range(1, n + 1):
                w[jj] = gnew[jj]
            dsq, crvmin = _bobyqa_trsbox(n, npt, xpt, xopt, gopt, hq, pq, sl, su,
                                         delta, xnew, d, gnew, xbdi,
                                         s_arr, hs_arr, hred)
            # Snapshot GNEW into W(1..N) for the BDTEST branch below.
            for jj in range(1, n + 1):
                w[jj] = gnew[jj]
            dnorm = min(delta, np.sqrt(dsq))
            if dnorm < HALF * rho:
                ntrits = -1
                distsq = (TEN * rho) ** 2
                if nf <= nfsav + 2:
                    state = 'L650'
                    continue
                errbig = max(diffa, diffb, diffc)
                frhosq = 0.125 * rho * rho
                if crvmin > ZERO and errbig > frhosq * crvmin:
                    state = 'L650'
                    continue
                bdtol = errbig / rho
                hit_650 = False
                for j in range(1, n + 1):
                    bdtest = bdtol
                    if xnew[j] == sl[j]:
                        bdtest = w[j]
                    if xnew[j] == su[j]:
                        bdtest = -w[j]
                    if bdtest < bdtol:
                        curv = hq[(j + j * j) // 2]
                        for k in range(1, npt + 1):
                            curv = curv + pq[k] * xpt[k, j] ** 2
                        bdtest = bdtest + HALF * curv * rho
                        if bdtest < bdtol:
                            hit_650 = True
                            break
                if hit_650:
                    state = 'L650'
                    continue
                state = 'L680'
                continue
            ntrits = ntrits + 1
            state = 'L90'
        elif state == 'L90':
            # Severe cancellation guard: shift XBASE if XOPT far from XBASE.
            if dsq <= 1.0e-3 * xoptsq:
                fracsq = 0.25 * xoptsq
                sumpq = ZERO
                for k in range(1, npt + 1):
                    sumpq = sumpq + pq[k]
                    sum_ = -HALF * xoptsq
                    for i in range(1, n + 1):
                        sum_ = sum_ + xpt[k, i] * xopt[i]
                    w[npt + k] = sum_
                    temp = fracsq - HALF * sum_
                    for i in range(1, n + 1):
                        w[i] = bmat[k, i]
                        vlag[i] = sum_ * xpt[k, i] + temp * xopt[i]
                        ip = npt + i
                        for j in range(1, i + 1):
                            bmat[ip, j] = bmat[ip, j] + w[i] * vlag[j] + vlag[i] * w[j]
                # BMAT depending on ZMAT.
                for jj in range(1, nptm + 1):
                    sumz = ZERO
                    sumw = ZERO
                    for k in range(1, npt + 1):
                        sumz = sumz + zmat[k, jj]
                        vlag[k] = w[npt + k] * zmat[k, jj]
                        sumw = sumw + vlag[k]
                    for j in range(1, n + 1):
                        sum_ = (fracsq * sumz - HALF * sumw) * xopt[j]
                        for k in range(1, npt + 1):
                            sum_ = sum_ + vlag[k] * xpt[k, j]
                        w[j] = sum_
                        for k in range(1, npt + 1):
                            bmat[k, j] = bmat[k, j] + sum_ * zmat[k, jj]
                    for i in range(1, n + 1):
                        ip = i + npt
                        temp = w[i]
                        for j in range(1, i + 1):
                            bmat[ip, j] = bmat[ip, j] + temp * w[j]
                # Finalize shift; update HQ, XPT, XBASE, XNEW, SL, SU, XOPT.
                ih = 0
                for j in range(1, n + 1):
                    w[j] = -HALF * sumpq * xopt[j]
                    for k in range(1, npt + 1):
                        w[j] = w[j] + pq[k] * xpt[k, j]
                        xpt[k, j] = xpt[k, j] - xopt[j]
                    for i in range(1, j + 1):
                        ih = ih + 1
                        hq[ih] = hq[ih] + w[i] * xopt[j] + xopt[i] * w[j]
                        bmat[npt + i, j] = bmat[npt + j, i]
                for i in range(1, n + 1):
                    xbase[i] = xbase[i] + xopt[i]
                    xnew[i] = xnew[i] - xopt[i]
                    sl[i] = sl[i] - xopt[i]
                    su[i] = su[i] - xopt[i]
                    xopt[i] = ZERO
                xoptsq = ZERO
            if ntrits == 0:
                state = 'L210'
                continue
            state = 'L230'
        elif state == 'L190':
            # RESCUE.
            nfsav = nf
            kbase = kopt
            ptsaux = np.zeros((3, n + 1))  # 1..2, 1..n
            ptsid = np.zeros(npt + 1)
            nf, kopt = _bobyqa_rescue(
                calfun, n, npt, xl, xu, maxfun, xbase, xpt, fval,
                xopt, gopt, hq, pq, bmat, zmat, ndim, sl, su, nf, delta,
                kopt, vlag, ptsaux, ptsid, w,
            )
            xoptsq = ZERO
            if kopt != kbase:
                for i in range(1, n + 1):
                    xopt[i] = xpt[kopt, i]
                    xoptsq = xoptsq + xopt[i] ** 2
            if nf < 0:
                nf = maxfun
                ierr = 390
                state = 'L720'
                continue
            nresc = nf
            if nfsav < nf:
                nfsav = nf
                state = 'L20'
                continue
            if ntrits > 0:
                state = 'L60'
                continue
            state = 'L210'
        elif state == 'L210':
            # ALTMOV.
            glag = np.zeros(n + 1)
            hcol = np.zeros(npt + 1)
            # W(1..2*N) is the altmov workspace.
            adelt_local = adelt
            alpha, cauchy = _bobyqa_altmov(
                n, npt, xpt, xopt, bmat, zmat, ndim, sl, su, kopt,
                knew, adelt_local, xnew, xalt, glag, hcol, w,
            )
            for i in range(1, n + 1):
                d[i] = xnew[i] - xopt[i]
            state = 'L230'
        elif state == 'L230':
            # Compute VLAG and BETA for current D.
            for k in range(1, npt + 1):
                suma = ZERO
                sumb = ZERO
                sum_ = ZERO
                for j in range(1, n + 1):
                    suma = suma + xpt[k, j] * d[j]
                    sumb = sumb + xpt[k, j] * xopt[j]
                    sum_ = sum_ + bmat[k, j] * d[j]
                w[k] = suma * (HALF * suma + sumb)
                vlag[k] = sum_
                w[npt + k] = suma
            beta = ZERO
            for jj in range(1, nptm + 1):
                sum_ = ZERO
                for k in range(1, npt + 1):
                    sum_ = sum_ + zmat[k, jj] * w[k]
                beta = beta - sum_ * sum_
                for k in range(1, npt + 1):
                    vlag[k] = vlag[k] + sum_ * zmat[k, jj]
            dsq = ZERO
            bsum = ZERO
            dx = ZERO
            for j in range(1, n + 1):
                dsq = dsq + d[j] ** 2
                sum_ = ZERO
                for k in range(1, npt + 1):
                    sum_ = sum_ + w[k] * bmat[k, j]
                bsum = bsum + sum_ * d[j]
                jp = npt + j
                for i in range(1, n + 1):
                    sum_ = sum_ + bmat[jp, i] * d[i]
                vlag[jp] = sum_
                bsum = bsum + sum_ * d[j]
                dx = dx + d[j] * xopt[j]
            beta = dx * dx + dsq * (xoptsq + dx + dx + HALF * dsq) + beta - bsum
            vlag[kopt] = vlag[kopt] + ONE
            #
            # NTRITS=0: Cauchy-step alternative; possibly RESCUE.
            #
            if ntrits == 0:
                denom = vlag[knew] ** 2 + alpha * beta
                if denom < cauchy and cauchy > ZERO:
                    for i in range(1, n + 1):
                        xnew[i] = xalt[i]
                        d[i] = xnew[i] - xopt[i]
                    cauchy = ZERO
                    state = 'L230'
                    continue
                if denom <= HALF * vlag[knew] ** 2:
                    if nf > nresc:
                        state = 'L190'
                        continue
                    ierr = 320
                    state = 'L720'
                    continue
            else:
                # Pick KNEW for trust-region replacement.
                delsq = delta * delta
                scaden = ZERO
                biglsq = ZERO
                knew = 0
                for k in range(1, npt + 1):
                    if k == kopt:
                        continue
                    hdiag = ZERO
                    for jj in range(1, nptm + 1):
                        hdiag = hdiag + zmat[k, jj] ** 2
                    den = beta * hdiag + vlag[k] ** 2
                    distsq = ZERO
                    for j in range(1, n + 1):
                        distsq = distsq + (xpt[k, j] - xopt[j]) ** 2
                    temp = max(ONE, (distsq / delsq) ** 2)
                    if temp * den > scaden:
                        scaden = temp * den
                        knew = k
                        denom = den
                    biglsq = max(biglsq, temp * vlag[k] ** 2)
                if scaden <= HALF * biglsq:
                    if nf > nresc:
                        state = 'L190'
                        continue
                    ierr = 320
                    state = 'L720'
                    continue
            state = 'L360'
        elif state == 'L360':
            # Evaluate CALFUN at XBASE+XNEW.
            for i in range(1, n + 1):
                x[i] = min(max(xl[i], xbase[i] + xnew[i]), xu[i])
                if xnew[i] == sl[i]:
                    x[i] = xl[i]
                if xnew[i] == su[i]:
                    x[i] = xu[i]
            if nf >= maxfun:
                ierr = 390
                state = 'L720'
                continue
            nf = nf + 1
            f = calfun(x[1:n + 1])
            if ntrits == -1:
                fsave = f
                state = 'L720'
                continue
            #
            # VQUAD = quadratic-model prediction of F at XOPT+D.
            #
            fopt = fval[kopt]
            vquad = ZERO
            ih = 0
            for j in range(1, n + 1):
                vquad = vquad + d[j] * gopt[j]
                for i in range(1, j + 1):
                    ih = ih + 1
                    temp = d[i] * d[j]
                    if i == j:
                        temp = HALF * temp
                    vquad = vquad + hq[ih] * temp
            for k in range(1, npt + 1):
                vquad = vquad + HALF * pq[k] * w[npt + k] ** 2
            diff = f - fopt - vquad
            diffc = diffb
            diffb = diffa
            diffa = abs(diff)
            if dnorm > rho:
                nfsav = nf
            #
            # Adjust DELTA after a trust-region step.
            #
            if ntrits > 0:
                if vquad >= ZERO:
                    ierr = 430
                    state = 'L720'
                    continue
                ratio = (f - fopt) / vquad
                if ratio <= TENTH:
                    delta = min(HALF * delta, dnorm)
                elif ratio <= 0.7:
                    delta = max(HALF * delta, dnorm)
                else:
                    delta = max(HALF * delta, dnorm + dnorm)
                if delta <= 1.5 * rho:
                    delta = rho
                #
                # Recompute KNEW, DENOM if new F < FOPT.
                #
                if f < fopt:
                    ksav = knew
                    densav = denom
                    delsq = delta * delta
                    scaden = ZERO
                    biglsq = ZERO
                    knew = 0
                    for k in range(1, npt + 1):
                        hdiag = ZERO
                        for jj in range(1, nptm + 1):
                            hdiag = hdiag + zmat[k, jj] ** 2
                        den = beta * hdiag + vlag[k] ** 2
                        distsq = ZERO
                        for j in range(1, n + 1):
                            distsq = distsq + (xpt[k, j] - xnew[j]) ** 2
                        temp = max(ONE, (distsq / delsq) ** 2)
                        if temp * den > scaden:
                            scaden = temp * den
                            knew = k
                            denom = den
                        biglsq = max(biglsq, temp * vlag[k] ** 2)
                    if scaden <= HALF * biglsq:
                        knew = ksav
                        denom = densav
            #
            # Update BMAT, ZMAT, HQ, PQ for the new KNEW.
            #
            _bobyqa_update(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew, w)
            ih = 0
            pqold = pq[knew]
            pq[knew] = ZERO
            for i in range(1, n + 1):
                temp = pqold * xpt[knew, i]
                for j in range(1, i + 1):
                    ih = ih + 1
                    hq[ih] = hq[ih] + temp * xpt[knew, j]
            for jj in range(1, nptm + 1):
                temp = diff * zmat[knew, jj]
                for k in range(1, npt + 1):
                    pq[k] = pq[k] + temp * zmat[k, jj]
            #
            # Absorb the new interpolation point; update GOPT.
            #
            fval[knew] = f
            for i in range(1, n + 1):
                xpt[knew, i] = xnew[i]
                w[i] = bmat[knew, i]
            for k in range(1, npt + 1):
                suma = ZERO
                for jj in range(1, nptm + 1):
                    suma = suma + zmat[knew, jj] * zmat[k, jj]
                sumb = ZERO
                for j in range(1, n + 1):
                    sumb = sumb + xpt[k, j] * xopt[j]
                temp = suma * sumb
                for i in range(1, n + 1):
                    w[i] = w[i] + temp * xpt[k, i]
            for i in range(1, n + 1):
                gopt[i] = gopt[i] + diff * w[i]
            #
            # Update XOPT, GOPT, KOPT if new F < FOPT.
            #
            if f < fopt:
                kopt = knew
                xoptsq = ZERO
                ih = 0
                for j in range(1, n + 1):
                    xopt[j] = xnew[j]
                    xoptsq = xoptsq + xopt[j] ** 2
                    for i in range(1, j + 1):
                        ih = ih + 1
                        if i < j:
                            gopt[j] = gopt[j] + hq[ih] * d[i]
                        gopt[i] = gopt[i] + hq[ih] * d[j]
                for k in range(1, npt + 1):
                    temp = ZERO
                    for j in range(1, n + 1):
                        temp = temp + xpt[k, j] * d[j]
                    temp = pq[k] * temp
                    for i in range(1, n + 1):
                        gopt[i] = gopt[i] + temp * xpt[k, i]
                # NLopt LN_BOBYQA terminates here when a trust-region step
                # improves F by less than ftol (bobyqa.c:2818 — checked only
                # inside ``f < fopt``, after KOPT/XOPT/GOPT are updated, so the
                # returned point is the improved one). ``stop=None`` (glmer's
                # minqa path) skips this and runs to rhoend, unchanged.
                if stop is not None and stop.ftol(f, fopt):
                    fsave = f
                    ierr = 0
                    state = 'L720'
                    continue
            #
            # Frobenius-norm interpolant gradient check (NTRITS>0 only).
            #
            if ntrits > 0:
                for k in range(1, npt + 1):
                    vlag[k] = fval[k] - fval[kopt]
                    w[k] = ZERO
                for j in range(1, nptm + 1):
                    sum_ = ZERO
                    for k in range(1, npt + 1):
                        sum_ = sum_ + zmat[k, j] * vlag[k]
                    for k in range(1, npt + 1):
                        w[k] = w[k] + sum_ * zmat[k, j]
                for k in range(1, npt + 1):
                    sum_ = ZERO
                    for j in range(1, n + 1):
                        sum_ = sum_ + xpt[k, j] * xopt[j]
                    w[k + npt] = w[k]
                    w[k] = sum_ * w[k]
                gqsq = ZERO
                gisq = ZERO
                for i in range(1, n + 1):
                    sum_ = ZERO
                    for k in range(1, npt + 1):
                        sum_ = sum_ + bmat[k, i] * vlag[k] + xpt[k, i] * w[k]
                    if xopt[i] == sl[i]:
                        gqsq = gqsq + min(ZERO, gopt[i]) ** 2
                        gisq = gisq + min(ZERO, sum_) ** 2
                    elif xopt[i] == su[i]:
                        gqsq = gqsq + max(ZERO, gopt[i]) ** 2
                        gisq = gisq + max(ZERO, sum_) ** 2
                    else:
                        gqsq = gqsq + gopt[i] ** 2
                        gisq = gisq + sum_ * sum_
                    vlag[npt + i] = sum_
                itest = itest + 1
                if gqsq < TEN * gisq:
                    itest = 0
                if itest >= 3:
                    for i in range(1, max(npt, nh) + 1):
                        if i <= n:
                            gopt[i] = vlag[npt + i]
                        if i <= npt:
                            pq[i] = w[npt + i]
                        if i <= nh:
                            hq[i] = ZERO
                        itest = 0
            if ntrits == 0:
                state = 'L60'
                continue
            if f <= fopt + TENTH * vquad:
                state = 'L60'
                continue
            distsq = max((TWO * delta) ** 2, (TEN * rho) ** 2)
            state = 'L650'
        elif state == 'L650':
            knew = 0
            # distsq comes from L60 (NTRITS=-1 branch) or L360.
            if 'distsq' not in dir() or state == 'L650':  # ensure defined
                pass
            for k in range(1, npt + 1):
                sum_ = ZERO
                for j in range(1, n + 1):
                    sum_ = sum_ + (xpt[k, j] - xopt[j]) ** 2
                if sum_ > distsq:
                    knew = k
                    distsq = sum_
            if knew > 0:
                dist = np.sqrt(distsq)
                if ntrits == -1:
                    delta = min(TENTH * delta, HALF * dist)
                    if delta <= 1.5 * rho:
                        delta = rho
                ntrits = 0
                adelt = max(min(TENTH * dist, delta), rho)
                dsq = adelt * adelt
                state = 'L90'
                continue
            if ntrits == -1:
                state = 'L680'
                continue
            if ratio > ZERO:
                state = 'L60'
                continue
            if max(delta, dnorm) > rho:
                state = 'L60'
                continue
            state = 'L680'
        elif state == 'L680':
            # Pick the next RHO.
            if rho > rhoend:
                delta = HALF * rho
                ratio = rho / rhoend
                if ratio <= 16.0:
                    rho = rhoend
                elif ratio <= 250.0:
                    rho = np.sqrt(ratio) * rhoend
                else:
                    rho = TENTH * rho
                delta = max(delta, rho)
                ntrits = 0
                nfsav = nf
                state = 'L60'
                continue
            if ntrits == -1:
                state = 'L360'
                continue
            state = 'L720'
        elif state == 'L720':
            return _bobyqa_finalize(x, xl, xu, sl, su, xbase, xopt, fval, kopt,
                                    fsave, n, nf, ierr, f)
        else:
            raise RuntimeError(f"bobyqb: unknown state {state!r}")


def _bobyqa_finalize(x, xl, xu, sl, su, xbase, xopt, fval, kopt, fsave, n, nf, ierr, f=None):
    """Label 720 of ``bobyqb.f``: write final X and return ``(x, f, nf, ierr)``."""
    if fval[kopt] <= fsave:
        for i in range(1, n + 1):
            x[i] = min(max(xl[i], xbase[i] + xopt[i]), xu[i])
            if xopt[i] == sl[i]:
                x[i] = xl[i]
            if xopt[i] == su[i]:
                x[i] = xu[i]
        f_out = fval[kopt]
    else:
        f_out = f if f is not None else fval[kopt]
    return x, f_out, nf, ierr


def _bobyqa_driver(calfun, x0, lower, upper, *,
                   npt=None, rhobeg=None, rhoend=None, maxfun=10000, stop=None):
    """Public BOBYQA entry. Port of ``bobyqa.f`` (workspace partition,
    bound-aware initial X) plus the ``minqa`` R-wrapper defaults
    (``rhobeg``, ``rhoend``, ``npt`` when ``None``).

    Parameters mirror ``minqa::bobyqa``:
        npt    : interpolation points (default ``min(n+2, 2n)``, floored to ``n+2``)
        rhobeg : initial trust-radius (default ``min(0.95, 0.2*max|par|)``)
        rhoend : final trust-radius   (default ``1e-6 * rhobeg``)
        maxfun : max function evals   (default 10000)

    Returns ``(par, fval, nf, ierr, msg)``.
    """
    x0 = np.asarray(x0, dtype=float).copy()
    n = x0.size
    lower = np.asarray(lower, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    # Apply minqa defaults.
    if rhobeg is None:
        rhobeg = min(0.95, 0.2 * max(np.max(np.abs(x0)), 1e-300))
    if rhoend is None:
        rhoend = 1.0e-6 * rhobeg
    if npt is None:
        npt = max(n + 2, min(n + 2, 2 * n))
    npt = int(max(n + 2, min(int(npt), ((n + 1) * (n + 2)) // 2)))
    #
    # Range-shrink rhobeg if any bound interval is < 2*rhobeg (matches minqa.R).
    #
    rng = upper - lower
    if np.any(rng < 2 * rhobeg):
        rhobeg = 0.2 * np.min(rng)
        rhoend = min(rhoend, rhobeg)
    #
    # NPT validity (matches bobyqa.f IERR=10).
    #
    if not (n + 2 <= npt <= ((n + 2) * (n + 1)) // 2):
        return (x0, float('inf'), 0, 10,
                "bobyqa -- NPT is not in the required interval")
    #
    # 1-indexed work arrays.
    #
    x = np.zeros(n + 1)
    x[1:n + 1] = x0
    xl = np.zeros(n + 1)
    xl[1:n + 1] = lower
    xu = np.zeros(n + 1)
    xu[1:n + 1] = upper
    sl = np.zeros(n + 1)
    su = np.zeros(n + 1)
    #
    # Bound check + initial X clamp (bobyqa.f:99-136).
    #
    for j in range(1, n + 1):
        temp = xu[j] - xl[j]
        if temp < rhobeg + rhobeg:
            return (x[1:n + 1], float('inf'), 0, 20,
                    "bobyqa -- one of the box constraint ranges is too small (< 2*RHOBEG)")
        sl[j] = xl[j] - x[j]
        su[j] = xu[j] - x[j]
        if sl[j] >= -rhobeg:
            if sl[j] >= 0.0:
                x[j] = xl[j]
                sl[j] = 0.0
                su[j] = temp
            else:
                x[j] = xl[j] + rhobeg
                sl[j] = -rhobeg
                su[j] = max(xu[j] - x[j], rhobeg)
        elif su[j] <= rhobeg:
            if su[j] <= 0.0:
                x[j] = xu[j]
                sl[j] = -temp
                su[j] = 0.0
            else:
                x[j] = xu[j] - rhobeg
                sl[j] = min(xl[j] - x[j], -rhobeg)
                su[j] = rhobeg
    #
    # Run the main loop.
    #
    x_out, f_out, nf, ierr = _bobyqa_bobyqb(
        calfun, n, npt, x, xl, xu, rhobeg, rhoend, maxfun, sl, su, stop=stop,
    )
    msgmap = {
        0:   "Normal exit from bobyqa",
        10:  "bobyqa -- NPT is not in the required interval",
        20:  "bobyqa -- one of the box constraint ranges is too small (< 2*RHOBEG)",
        320: "bobyqa detected too much cancellation in denominator",
        390: "bobyqa -- maximum number of function evaluations exceeded",
        430: "bobyqa -- a trust region step failed to reduce q",
    }
    return x_out[1:n + 1].copy(), f_out, nf, ierr, msgmap.get(ierr, "")


# ----------------------------------------------------------------------
# NLopt LN_BOBYQA — lme4's DEFAULT lmer optimizer (``nloptwrap``).
#
# lme4's ``lmer`` optimizes the profiled (RE)ML deviance over θ with
# ``nloptwrap`` = NLopt's ``NLOPT_LN_BOBYQA`` (utilities.R:836-839,
# ``xtol_abs=ftol_abs=1e-8, maxeval=1e5``). That is Powell's BOBYQA — the SAME
# core as the minqa port above — wrapped with three NLopt-specific pieces
# (ported verbatim from ``ref/nlopt/``): (1) a per-axis variable rescaling so
# the initial steps are equal (rescale.c), (2) a default initial-step heuristic
# from the bounds (options.c ``nlopt_set_default_initial_step``), and (3) an
# ``ftol_abs`` stopping test woven into the trust-region loop (stop.c
# ``relstop`` → injected at ``_bobyqa_bobyqb``'s ``f < fopt`` branch). Reusing
# the minqa core + these three wrappers reproduces ``nloptwrap`` to the CHOLMOD
# floor (~1e-9 on θ̂; the residual is scikit-sparse-vs-lme4 arithmetic, the same
# gap that makes lme4's own optimizers disagree by ~1e-5 on flat surfaces).
# ----------------------------------------------------------------------

_DBL_MIN = 2.2250738585072014e-308  # for nlopt_istiny


def _nlopt_default_step(x, lb, ub):
    """Port of NLopt's ``nlopt_set_default_initial_step`` (options.c) — the
    crude per-axis initial-step heuristic BOBYQA uses when no explicit step is
    given (lme4 never sets one). Returns ``dx`` of length ``n``."""
    x = np.asarray(x, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    n = x.size
    dx = np.empty(n)
    for i in range(n):
        step = np.inf
        if (np.isfinite(ub[i]) and np.isfinite(lb[i])
                and (ub[i] - lb[i]) * 0.25 < step and ub[i] > lb[i]):
            step = (ub[i] - lb[i]) * 0.25
        if np.isfinite(ub[i]) and ub[i] - x[i] < step and ub[i] > x[i]:
            step = (ub[i] - x[i]) * 0.75
        if np.isfinite(lb[i]) and x[i] - lb[i] < step and x[i] > lb[i]:
            step = (x[i] - lb[i]) * 0.75
        if np.isinf(step):
            if np.isfinite(ub[i]) and abs(ub[i] - x[i]) < abs(step):
                step = (ub[i] - x[i]) * 1.1
            if np.isfinite(lb[i]) and abs(x[i] - lb[i]) < abs(step):
                step = (x[i] - lb[i]) * 1.1
        if np.isinf(step) or (step != 0.0 and abs(step) < _DBL_MIN):  # istiny
            step = x[i]
        if np.isinf(step) or step == 0.0:
            step = 1.0
        dx[i] = step
    return dx


def _nlopt_compute_rescaling(dx):
    """Port of ``nlopt_compute_rescaling`` (rescale.c): ``s[i]=dx[i]/dx[0]``
    when the initial steps differ (so ``dx[i]/s[i]`` is equal in all
    directions), else all-ones. ``nlopt_rescale`` divides x by s; ``unscale``
    multiplies."""
    dx = np.asarray(dx, float)
    n = dx.size
    s = np.ones(n)
    if n == 1:
        return s
    i = 1
    while i < n and dx[i] == dx[i - 1]:
        i += 1
    if i < n:  # unequal steps → rescale to dx[0]
        for i in range(1, n):
            s[i] = dx[i] / dx[0]
    return s


class _NloptStopInfo:
    """NLopt's ``ftol`` stopping predicate (stop.c ``relstop``/``nlopt_stop_ftol``):
    ``|f-oldf| < ftol_abs`` or ``< ftol_rel·(|f|+|oldf|)/2`` (and never when
    ``oldf`` is non-finite). Used by :func:`_nlopt_ln_bobyqa`."""

    def __init__(self, ftol_abs=1e-8, ftol_rel=0.0):
        self.ftol_abs = ftol_abs
        self.ftol_rel = ftol_rel

    def ftol(self, f, oldf):
        if not np.isfinite(oldf):
            return False
        d = abs(f - oldf)
        return (d < self.ftol_abs
                or d < self.ftol_rel * (abs(f) + abs(oldf)) * 0.5
                or (self.ftol_rel > 0 and f == oldf))


class _NloptResult:
    """scipy-``OptimizeResult``-shaped return from :func:`_nlopt_ln_bobyqa`
    (``.x``/``.fun``/``.nfev``/``.success``) so it drops into the LMM fit path
    where the old ``scipy.optimize.minimize`` result was used."""

    def __init__(self, x, fun, nfev, success, message):
        self.x = x
        self.fun = fun
        self.nfev = nfev
        self.success = success
        self.message = message


def _nlopt_ln_bobyqa(fn, x0, lb, ub, *, ftol_abs=1e-8, ftol_rel=0.0,
                     xtol_abs=1e-8, xtol_rel=0.0, maxeval=100000):
    """NLopt ``NLOPT_LN_BOBYQA`` over box bounds — lme4's default ``lmer``
    optimizer (``nloptwrap``). Wraps the minqa BOBYQA core (:func:`_bobyqa_driver`)
    with NLopt's variable rescaling, default initial step, and ``ftol`` stop.

    ``fn(x) -> float`` is the objective; ``lb``/``ub`` are the (possibly
    ``±inf``) bounds. Returns an :class:`_NloptResult`. Defaults are lme4's
    (``ftol_abs=xtol_abs=1e-8, maxeval=1e5``); ``ftol_rel``/``xtol_rel`` default
    to 0 (NLopt's defaults — lme4 leaves them unset)."""
    x0 = np.asarray(x0, float)
    n = x0.size
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    # (1) default initial step + (2) per-axis rescaling so all steps are equal.
    dx = _nlopt_default_step(x0, lb, ub)
    s = _nlopt_compute_rescaling(dx)
    if np.any(s == 0) or not np.all(np.isfinite(s)):
        raise ValueError("nlopt_bobyqa: invalid rescaling (over/underflow?)")
    x0s = x0 / s
    lbs = lb / s
    ubs = ub / s
    for j in range(n):  # nlopt_reorder_bounds (s could flip sign; here s>0)
        if lbs[j] > ubs[j]:
            lbs[j], ubs[j] = ubs[j], lbs[j]
    rhobeg = abs(dx[0] / s[0])
    rhoend = xtol_rel * rhobeg
    for j in range(n):
        rhoend = max(rhoend, xtol_abs / abs(s[j]))
    # (3) ftol stop woven into the trust-region loop via the ``stop`` hook.
    stop = _NloptStopInfo(ftol_abs, ftol_rel)
    npt = 2 * n + 1

    def calfun(xs):  # objective in scaled space (unscale before calling fn)
        return fn(xs * s)

    par_s, fval, nf, ierr, msg = _bobyqa_driver(
        calfun, x0s, lbs, ubs, npt=npt, rhobeg=rhobeg, rhoend=rhoend,
        maxfun=int(maxeval), stop=stop)
    return _NloptResult(par_s * s, fval, nf, ierr == 0, msg)


# ----------------------------------------------------------------------
# Bounded-simplex Nelder-Mead — port of lme4's ``src/optimizer.cpp``.
#
# lme4 ships its own Nelder-Mead implementation (derived from NLopt 2.2.4's
# ``nldrmd``) for the GLMM outer optimizer. Porting it directly — rather
# than wrapping ``scipy.optimize.minimize(method="Nelder-Mead")`` — lets
# this module match lme4's iteration trajectory byte-for-byte when both
# are run with ``optimizer="Nelder_Mead"``. scipy's Nelder-Mead uses a
# different bounds-handling scheme and different default tolerances; at
# matched ``xtol`` settings the trajectories diverge after a few iterations.
#
# The port preserves lme4's state-machine layout (stages
# ``restart → postreflect → {postexpand | postcontract}``), reflection
# heuristic (``alpha=1, beta=0.5, gamm=2, delta=0.5``), and convergence
# defaults from the R wrapper (``optimizer.R:27-33``: ``maxfun=10000``,
# ``FtolAbs=1e-5``, ``XtolRel=1e-7``, etc.). ``ftol``-style convergence is
# defined in C++ but never invoked by the loop; we keep the parameter for
# parity but it's effectively unused — only ``xtol``, ``maxeval``, and
# ``minf_max`` trigger termination.
#
# References:
# - ``/tmp/lme4/src/optimizer.cpp`` — C++ implementation.
# - ``/tmp/lme4/src/optimizer.h`` — header with ``nm_status``/``nm_stage``
#   enums and the heuristic constants.
# - ``/tmp/lme4/R/optimizer.R`` — R wrapper exposing ``Nelder_Mead()``.
# - NLopt's ``nldrmd.c`` — original algorithm by S. G. Johnson.

_NM_ALPHA = 1.0    # reflection      — optimizer.h:95
_NM_BETA  = 0.5    # contraction
_NM_GAMM  = 2.0    # expansion
_NM_DELTA = 0.5    # shrink


class NMStatus(IntEnum):
    """Return code from :meth:`NelderMead.newf`.

    Mirrors lme4's ``nm_status`` enum (optimizer.h:89). ``active`` means
    "continue iterating"; any other value means the optimizer has stopped.
    """
    active = 0
    x0_not_feasible = 1   # nm_x0notfeasible — raised by ctor, never returned.
    no_feasible = 2       # nm_nofeasible    — raised by ctor, never returned.
    forced = 3            # nm_forced (set_force_stop=True)
    minf_max = 4          # objective dipped below ``minf_max``
    evals = 5             # hit ``maxeval``
    fcvg = 6              # ftol convergence (unused; preserved for parity)
    xcvg = 7              # xtol convergence


class _NMStage(IntEnum):
    """Internal stage of the state machine — optimizer.h:92."""
    restart = 0
    postreflect = 1
    postexpand = 2
    postcontract = 3


def _nm_close(a: float, b: float) -> bool:
    """Two values are within floating-point tolerance — optimizer.cpp:30."""
    return abs(a - b) <= 1e-13 * (abs(a) + abs(b))


def _nm_relstop(vold: float, vnew: float, reltol: float, abstol: float) -> bool:
    """nl_stop's relative-stop predicate — optimizer.h:64-87."""
    if np.isinf(abs(vold)):
        return False
    return (
        abs(vnew - vold) < abstol
        or abs(vnew - vold) < reltol * (abs(vnew) + abs(vold)) * 0.5
        or (reltol > 0 and vnew == vold)
    )


class NelderMead:
    """Bounded-simplex Nelder-Mead — port of ``Nelder_Mead`` in optimizer.cpp.

    The caller drives the iteration via :meth:`xeval` (where to evaluate
    next) and :meth:`newf` (feed the function value back). After
    :meth:`newf` returns a status other than :attr:`NMStatus.active`, the
    best point is at :meth:`xpos` with value :meth:`value`. Or use the
    convenience :meth:`minimize` for the common pattern.

    Parameters
    ----------
    lb, ub
        Element-wise lower/upper bounds; ``-np.inf``/``np.inf`` for
        unbounded coordinates. ``x0`` must be feasible.
    xstep
        Initial step sizes along each coordinate. The R wrapper at
        optimizer.R:5 defaults to ``rep(0.02, n)``; lme4's Stage 1 setup
        at lmer.R:2534-2540 uses ``0.2 * [0.1; min(βSD, 10)]``.
    x0
        Initial point; must lie in ``[lb, ub]``.
    xtol_abs
        Per-coordinate absolute xtol. Defaults to ``|xstep| * 5e-4``
        matching the R wrapper at optimizer.R:6.
    xtol_rel, ftol_abs, ftol_rel
        Relative/absolute tolerances. ``ftol_*`` are stored but never
        consulted by the C++ implementation — included for API parity.
    maxeval
        Maximum function evaluations. Default 10000 (R wrapper default).
    minf_max
        Optimizer terminates when the function dips below this.
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        xstep: np.ndarray,
        x0: np.ndarray,
        *,
        xtol_abs: Optional[np.ndarray] = None,
        ftol_abs: float = 1e-5,
        ftol_rel: float = 1e-15,
        xtol_rel: float = 1e-7,
        maxeval: int = 10000,
        minf_max: float = -np.finfo(float).max,
    ):
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        xstep = np.asarray(xstep, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        n = x0.size
        if lb.size != n or ub.size != n or xstep.size != n:
            raise ValueError(
                f"lb/ub/xstep/x0 size mismatch: {lb.size}/{ub.size}/"
                f"{xstep.size}/{n}"
            )
        if np.any(x0 - lb < 0) or np.any(ub - x0 < 0):
            raise ValueError("initial x0 is not a feasible point")
        if np.any(xstep == 0):
            raise ValueError("xstep must be nonzero for every coordinate")
        if xtol_abs is None:
            xtol_abs = np.abs(xstep) * 5e-4
        xtol_abs = np.asarray(xtol_abs, dtype=float)
        if xtol_abs.size != n:
            raise ValueError(f"xtol_abs size {xtol_abs.size} != n={n}")

        # Build the initial simplex. Vertex 0 = x0; vertex j+1 = x0 + xstep[j]·e_j,
        # pinned into [lb, ub] via the constructor heuristics
        # (optimizer.cpp:71-91): if outside ub, clip to ub when there's
        # room, else flip direction; symmetric for lb. Degenerate ⇒ raise.
        pts = np.tile(x0[:, None], (1, n + 1))
        for i in range(n):
            j = i + 1
            pts[i, j] += xstep[i]
            if pts[i, j] > ub[i]:
                if ub[i] - x0[i] > abs(xstep[i]) * 0.1:
                    pts[i, j] = ub[i]
                else:
                    pts[i, j] = x0[i] - abs(xstep[i])
            if pts[i, j] < lb[i]:
                if x0[i] - lb[i] > abs(xstep[i]) * 0.1:
                    pts[i, j] = lb[i]
                else:
                    pts[i, j] = x0[i] + abs(xstep[i])
                    if pts[i, j] > ub[i]:
                        target = ub[i] if (ub[i] - x0[i] > x0[i] - lb[i]) else lb[i]
                        pts[i, j] = 0.5 * (target + x0[i])
            if _nm_close(pts[i, j], x0[i]):
                raise ValueError("cannot generate feasible simplex")

        self.lb = lb
        self.ub = ub
        self.xstep = xstep
        self.n = n
        self.pts = pts
        self.vals = np.full(n + 1, np.finfo(float).min, dtype=float)
        self.c = np.zeros(n)
        self.xcur = np.zeros(n)
        self.xeval_ = x0.copy()
        self.x = x0.copy()
        self.minf = np.inf
        self.stage = _NMStage.restart
        self.init_pos = 0
        self.xtol_abs = xtol_abs
        self.ftol_abs = ftol_abs
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.maxeval = maxeval
        self.minf_max = minf_max
        self.nevals = 0
        self.force_stop = False
        self._f_old = 0.0
        self._fh = 0.0
        self._fl = 0.0
        self._ih = 0
        self._il = 0

    # ---- public interface -----------------------------------------------

    def xeval(self) -> np.ndarray:
        """Where to evaluate the objective next."""
        return self.xeval_

    def xpos(self) -> np.ndarray:
        """Best parameter vector found so far."""
        return self.x

    def value(self) -> float:
        """Best function value found so far."""
        return self.minf

    def set_force_stop(self, stop: bool) -> None:
        """Request early termination on next :meth:`newf`."""
        self.force_stop = stop

    def newf(self, f: float) -> NMStatus:
        """Install ``f = objective(xeval())`` and step the state machine.

        Port of ``Nelder_Mead::newf`` (optimizer.cpp:101-141).
        """
        self.nevals += 1
        if self.force_stop:
            return NMStatus.forced
        if f < self.minf:
            self.minf = f
            self.x = self.xeval_.copy()
            if self.minf < self.minf_max:
                return NMStatus.minf_max
        if self.maxeval > 0 and self.nevals > self.maxeval:
            return NMStatus.evals
        if self.init_pos <= self.n:
            return self._init(f)
        if self.stage == _NMStage.restart:
            return self._restart(f)
        elif self.stage == _NMStage.postreflect:
            return self._postreflect(f)
        elif self.stage == _NMStage.postexpand:
            return self._postexpand(f)
        elif self.stage == _NMStage.postcontract:
            return self._postcontract(f)
        return NMStatus.active

    def minimize(self, fn: Callable[[np.ndarray], float]) -> NMStatus:
        """Run the optimizer to a stopping condition, calling ``fn`` each step."""
        while True:
            f = fn(self.xeval_)
            status = self.newf(f)
            if status != NMStatus.active:
                return status

    # ---- state-machine stages -------------------------------------------

    def _init(self, f: float) -> NMStatus:
        """Fill ``vals[init_pos]`` and queue the next simplex vertex
        (optimizer.cpp:150-156)."""
        if self.init_pos > self.n:
            raise RuntimeError("init called after n+1 evaluations")
        self.vals[self.init_pos] = f
        self.init_pos += 1
        if self.init_pos > self.n:
            return self._restart(f)
        self.xeval_ = self.pts[:, self.init_pos].copy()
        return NMStatus.active

    def _restart(self, f: float) -> NMStatus:
        """Recompute high/low/centroid, check x-convergence, reflect
        (optimizer.cpp:167-192)."""
        self._il = int(np.argmin(self.vals))
        self._fl = float(self.vals[self._il])
        self._ih = int(np.argmax(self.vals))
        self._fh = float(self.vals[self._ih])
        self.c = (self.pts.sum(axis=1) - self.pts[:, self._ih]) / self.n
        deviations = np.abs(self.pts - self.c[:, None]).max(axis=1)
        if self._x_conv(np.zeros(self.n), deviations):
            return NMStatus.xcvg
        if not self._reflectpt(self.xcur, self.c, _NM_ALPHA, self.pts[:, self._ih]):
            return NMStatus.xcvg
        self.xeval_ = self.xcur.copy()
        self.stage = _NMStage.postreflect
        return NMStatus.active

    def _postreflect(self, f: float) -> NMStatus:
        """Decide what to do with the reflected point — port of
        ``Nelder_Mead::postreflect`` (optimizer.cpp:194-219)."""
        if f < self._fl:
            if not self._reflectpt(self.xeval_, self.c, _NM_GAMM, self.pts[:, self._ih]):
                return NMStatus.xcvg
            self.stage = _NMStage.postexpand
            self._f_old = f
            return NMStatus.active
        if f < self._fh:
            self.vals[self._ih] = f
            self.pts[:, self._ih] = self.xeval_
            return self._restart(f)
        scale = -_NM_BETA if self._fh <= f else _NM_BETA
        if not self._reflectpt(self.xcur, self.c, scale, self.pts[:, self._ih]):
            return NMStatus.xcvg
        self._f_old = f
        self.xeval_ = self.xcur.copy()
        self.stage = _NMStage.postcontract
        return NMStatus.active

    def _postexpand(self, f: float) -> NMStatus:
        """Did expansion improve? Port of ``postexpand`` (optimizer.cpp:221-235)."""
        if f < self.vals[self._ih]:
            self.pts[:, self._ih] = self.xeval_
            self.vals[self._ih] = f
        else:
            self.pts[:, self._ih] = self.xcur
            self.vals[self._ih] = self._f_old
        return self._restart(f)

    def _postcontract(self, f: float) -> NMStatus:
        """Did contraction improve? Port of ``postcontract`` (optimizer.cpp:237-256).

        If yes, accept and restart. Otherwise SHRINK the entire simplex
        toward the best vertex (``il``) and re-evaluate every shrunk vertex.
        """
        if f < self._f_old and f < self._fh:
            self.pts[:, self._ih] = self.xeval_
            self.vals[self._ih] = f
            return self._restart(f)
        best = self.pts[:, self._il].copy()
        for i in range(self.n + 1):
            if i != self._il:
                target = np.empty(self.n)
                if not self._reflectpt(target, best, -_NM_DELTA, self.pts[:, i]):
                    return NMStatus.xcvg
                self.pts[:, i] = target
        self.init_pos = 0
        self.xeval_ = self.pts[:, 0].copy()
        return NMStatus.active

    # ---- helpers --------------------------------------------------------

    def _reflectpt(self, xnew: np.ndarray, c: np.ndarray, scale: float,
                   xold: np.ndarray) -> bool:
        """``xnew = clip(c + scale·(c − xold), lb, ub)`` (optimizer.cpp:269-289).

        Returns ``False`` if ``xnew`` coincides with ``c`` *or* ``xold``
        in every coordinate — signal of a collapsed simplex.
        """
        np.copyto(xnew, c + scale * (c - xold))
        equalc = True
        equalold = True
        for i in range(self.n):
            newx = min(max(xnew[i], self.lb[i]), self.ub[i])
            equalc = equalc and _nm_close(newx, c[i])
            equalold = equalold and _nm_close(newx, xold[i])
            xnew[i] = newx
        return not (equalc or equalold)

    def _x_conv(self, x: np.ndarray, oldx: np.ndarray) -> bool:
        """All coordinates pass relstop — port of ``nl_stop::x`` (optimizer.cpp:299)."""
        for i in range(x.size):
            if not _nm_relstop(oldx[i], x[i], self.xtol_rel, self.xtol_abs[i]):
                return False
        return True


# ---------------------------------------------------------------------------
# Argument plumbing & validation helpers.
# ---------------------------------------------------------------------------


def _resolve_lme_family(family) -> Family:
    """Resolve ``family=`` to a :class:`Family` instance — port of
    modular.R:733-735.

    Accepts ``None`` (→ Gaussian), a :class:`Family` instance, a class /
    callable that returns one (``Poisson`` → ``Poisson()``), or a name
    string (``"poisson"``). Rejects ``quasi*`` families with lme4's exact
    error (modular.R:734).
    """
    if family is None:
        return Gaussian()
    # Reject quasi by string first so the error mentions the input.
    if isinstance(family, str):
        if family in ("quasi", "quasibinomial", "quasipoisson"):
            raise ValueError('"quasi" families cannot be used in glmer')
        cls = getattr(_family_mod, family, None)
        if cls is None:
            raise ValueError(
                f"unknown family {family!r}; expected one of gaussian, "
                "poisson, binomial, Gamma, inverse_gaussian"
            )
        family = cls
    if isinstance(family, Family):
        if isinstance(family, _family_mod.Quasi):
            raise ValueError('"quasi" families cannot be used in glmer')
        return family
    if callable(family):
        result = family()
        if isinstance(result, Family):
            if isinstance(result, _family_mod.Quasi):
                raise ValueError('"quasi" families cannot be used in glmer')
            return result
        raise TypeError(
            f"family must resolve to a Family instance; calling {family!r} "
            f"returned {type(result).__name__}"
        )
    raise TypeError(
        f"family must be None, a Family instance, a Family class, or a "
        f"name string; got {type(family).__name__}"
    )


def _validate_nagq(nAGQ: int) -> int:
    """Validate ``nAGQ=`` per lme4 (modular.R:980-987).

    Integer in [0, 100]. ``nAGQ > 1`` (adaptive Gauss-Hermite) is supported;
    its single-scalar-RE constraint is checked at fit time, not here.
    """
    try:
        n = int(nAGQ)
    except (TypeError, ValueError):
        raise ValueError(f"nAGQ must be an integer; got {nAGQ!r}")
    if n != nAGQ:
        # Reject floats that aren't whole numbers (1.5 etc.). int(1.5) == 1
        # would silently round; force a clean error instead.
        raise ValueError(f"nAGQ must be an integer; got {nAGQ!r}")
    if n < 0 or n > 100:
        raise ValueError(f"nAGQ must be in [0, 100]; got {n}")
    # nAGQ > 1 (adaptive Gauss-Hermite) is supported. The
    # single-scalar-RE constraint (modular.R:918-920) is enforced at fit time
    # in _fit_glmm_from_components, where the RE structure is available.
    return n


_GLMER_CONTROL_DEFAULTS = {
    # glmer default chain c("bobyqa","Nelder_Mead") — both ported. Stage 0
    # runs optimizer[0], Stage 1 runs optimizer[1] (see _normalize_optimizer
    # _chain and _run_outer_stage).
    "optimizer": ["bobyqa", "Nelder_Mead"],
    "restart_edge": False,          # lmer-only, deferred
    "boundary.tol": 1e-5,           # deferred
    "calc.derivs": None,            # lme4 NULL → smart rule (resolved in __init__)
    "use.last.params": False,
    "sparseX": False,               # lme4 no-op (warns); accepted for parity
    "standardize.X": False,         # autoscale sibling, deferred
    "autoscale": None,              # deferred
    "tolPwrss": 1e-7,
    "compDev": True,
    "nAGQ0initStep": True,
    "optCtrl": {},                  # Nelder_Mead kwargs (maxfun, XtolRel, etc.)
    # check.* keys — pre-fit and post-fit validation. Accepted now;
    # enforcement lands incrementally.
    "check.nobs.vs.rankZ": "ignore",
    "check.nobs.vs.nlev": "stop",
    "check.nlev.gtreq.5": "ignore",
    "check.nlev.gtr.1": "stop",
    "check.nobs.vs.nRE": "stop",
    "check.rankX": "message+drop.cols",
    "check.scaleX": "warning",
    "check.formula.LHS": "stop",
    "check.response.not.const": "stop",
    "check.conv.nobsmax": 1e4,
    "check.conv.nparmax": 20,       # glmer default (lmer is 10)
    "check.conv.grad": {"action": "warning", "tol": 2e-3, "relTol": None},
    "check.conv.singular": {"action": "message", "tol": 1e-4},
    "check.conv.hess": {"action": "warning", "tol": 1e-6},
}


_PORTED_OPTIMIZERS = ("bobyqa", "Nelder_Mead")


def _normalize_optimizer_chain(optimizer) -> list:
    """Normalize ``glmerControl(optimizer=)`` to a ``[stage0, stage1]`` chain.

    lme4 replicates a single optimizer to both stages for glmer
    (lmerControl.R:109-112). Each stage must name a ported optimizer
    (``bobyqa`` or ``Nelder_Mead``); anything else (``nloptwrap``,
    ``optimx``, ``L-BFGS-B``, …) raises with a clear message.
    """
    if isinstance(optimizer, str):
        chain = [optimizer, optimizer]
    else:
        chain = list(optimizer)
        if len(chain) == 1:
            chain = [chain[0], chain[0]]
    if len(chain) != 2:
        raise ValueError(
            f"optimizer= must be a string or a length-1/2 sequence; "
            f"got {optimizer!r}"
        )
    bad = [o for o in chain if o not in _PORTED_OPTIMIZERS]
    if bad:
        raise NotImplementedError(
            f"optimizer={optimizer!r}: only {list(_PORTED_OPTIMIZERS)} are "
            f"ported (unsupported: {bad}). nloptwrap / optimx / L-BFGS-B "
            f"require a separate optimizer port."
        )
    return chain


_LMER_PORTED_OPTIMIZERS = ("nloptwrap", "bobyqa", "Nelder_Mead")

# lmer (Gaussian-identity LMM) control defaults. lme4 splits lmerControl
# (lmerControl.R:65-185) from glmerControl: the default optimizer is the
# single ``nloptwrap`` (NLopt LN_BOBYQA), ``restart_edge`` defaults TRUE, and
# ``check.conv.nparmax`` is 10 (glmer's is 20). The glmer-only inner-loop keys
# (tolPwrss / compDev / nAGQ0initStep / check.response.not.const) are carried
# here inertly so the shared ``__init__`` plumbing still finds them; they have
# no effect on the LMM fit path.
_LMER_CONTROL_DEFAULTS = {
    **_GLMER_CONTROL_DEFAULTS,
    "optimizer": "nloptwrap",
    "restart_edge": True,
    "check.conv.nparmax": 10,
}


def _normalize_lmer_optimizer(optimizer) -> str:
    """Normalize ``lmerControl(optimizer=)`` to a single ported optimizer name.

    lme4's lmer runs ONE optimizer over the profiled deviance
    (modular.R ``optimizeLmer``), default ``nloptwrap``. hea ports
    ``nloptwrap`` (NLopt LN_BOBYQA — the bit-exact default), ``bobyqa``
    (minqa), and ``Nelder_Mead`` (lme4's bounded NM). ``nlminbwrap`` /
    ``optimx`` / custom function optimizers raise :class:`NotImplementedError`.
    """
    if isinstance(optimizer, str):
        name = optimizer
    else:
        chain = list(optimizer)
        if len(chain) != 1:
            raise ValueError(
                f"lmer uses a single optimizer; got {optimizer!r}"
            )
        name = chain[0]
    if name not in _LMER_PORTED_OPTIMIZERS:
        raise NotImplementedError(
            f"optimizer={optimizer!r}: lmer supports "
            f"{list(_LMER_PORTED_OPTIMIZERS)} (nloptwrap is lme4's default). "
            f"nlminbwrap / optimx / custom functions need a separate port."
        )
    return name


def _run_outer_stage(optimizer_name, devfun, x0, lb, ub, *,
                     xst, xtol_abs, nm_kwargs, bobyqa_kwargs=None):
    """Run one outer-optimizer stage, dispatching on ``optimizer_name``.

    ``"bobyqa"`` → the ported minqa BOBYQA (finite bounds; one-sided ±inf
    is clamped to ±1e20 exactly as minqa's R wrapper does; ``bobyqa_kwargs``
    carries any ``rhobeg``/``rhoend``/``npt``/``maxfun`` from ``optCtrl``).
    ``"Nelder_Mead"`` → the ported lme4 bounded Nelder-Mead (consumes one-
    sided ±inf bounds directly, with the ``xst`` simplex step, ``xtol_abs``
    tolerance, and ``nm_kwargs`` from ``optCtrl``). Each optimizer ignores
    the other's ``optCtrl`` keys (lme4's per-stage behaviour). Returns
    ``(par, fval, nfeval, status)`` for both branches.
    """
    if optimizer_name == "bobyqa":
        lb_b = np.where(np.isfinite(lb), lb, -1.0e20)
        ub_b = np.where(np.isfinite(ub), ub, 1.0e20)
        par, fval, nf, ierr, _msg = _bobyqa_driver(
            devfun, x0, lb_b, ub_b, **(bobyqa_kwargs or {}))
        return np.asarray(par, dtype=float), float(fval), int(nf), int(ierr)
    nm = NelderMead(lb, ub, xst, x0, xtol_abs=xtol_abs, **nm_kwargs)
    status = nm.minimize(devfun)
    return nm.xpos().copy(), float(nm.value()), int(nm.nevals), int(status)


_NM_OPT_CTRL_KEYS = {"maxfun", "FtolAbs", "FtolRel", "XtolRel",
                     "MinfMax", "verbose"}

# BOBYQA-stage optCtrl keys (minqa::bobyqa args). ``maxfun`` is shared with
# Nelder_Mead; the rest are bobyqa-only. lme4 passes a single optCtrl list to
# whichever optimizer runs each stage, and each ignores the keys it doesn't
# understand — so a mixed chain may legitimately carry both families' keys.
_BOBYQA_OPT_CTRL_KEYS = {"maxfun", "rhobeg", "rhoend", "npt"}


def _bobyqa_kwargs_from_opt_ctrl(opt_ctrl) -> dict:
    """Translate lme4's ``optCtrl`` dict to :func:`_bobyqa_driver` kwargs.

    Picks up the minqa::bobyqa knobs (``rhobeg``, ``rhoend``, ``npt``,
    ``maxfun``) and skips Nelder_Mead-only keys (which the NM stage consumes).
    Genuinely unknown keys — in neither optimizer's vocabulary — raise.
    """
    if opt_ctrl is None or len(opt_ctrl) == 0:
        return {}
    out: dict = {}
    for key, val in opt_ctrl.items():
        if key == "maxfun":
            out["maxfun"] = int(val)
        elif key == "npt":
            out["npt"] = int(val)
        elif key in ("rhobeg", "rhoend"):
            out[key] = float(val)
        elif key in _NM_OPT_CTRL_KEYS:
            pass  # consumed by the Nelder_Mead stage, not bobyqa
        else:
            raise ValueError(
                f"unknown optCtrl key {key!r}; expected one of "
                f"{sorted(_NM_OPT_CTRL_KEYS | _BOBYQA_OPT_CTRL_KEYS)}"
            )
    return out


def _check_hess(H_sub, tol, hesstype=""):
    """Port of lme4's ``checkHess`` (checkConv.R:151-202). Eigenvalue-based
    Hessian diagnostics. Returns ``(messages, code)``."""
    try:
        evd = np.linalg.eigvalsh(H_sub)
    except np.linalg.LinAlgError:
        return (["Problem with Hessian check (infinite or missing values?)"], -6)
    neg = int(np.sum(evd < -tol))
    if neg:
        t = f" {hesstype}" if hesstype else ""
        return ([f"Model failed to converge: degenerate{t} Hessian with "
                 f"{neg} negative eigenvalues"], -3)
    chol_fail = False
    try:
        np.linalg.cholesky(H_sub)
    except np.linalg.LinAlgError:
        chol_fail = True
    if int(np.sum(np.abs(evd) < tol)) or chol_fail:
        t = f"{hesstype} " if hesstype else ""
        return ([f"{t}Hessian is numerically singular: parameters are not "
                 f"uniquely determined"], -4)
    msgs: list[str] = []
    code = 0
    if evd.max() * tol > 1:
        code = 2
        msgs.append("Model is nearly unidentifiable: very large eigenvalue\n"
                    " - Rescale variables?")
    if (evd.min() / evd.max()) < tol:
        code = 3
        msgs.append("Model is nearly unidentifiable: large eigenvalue ratio\n"
                    " - Rescale variables?")
    return (msgs, code)


def _checkconv_grad_hess(grad, hess, *, n_theta, grad_cfg, hess_cfg):
    """Port of lme4's ``checkConv`` gradient + Hessian blocks (checkConv.R:
    60-144). Run only for a non-singular fit with numerical derivatives
    available. Returns ``(messages, code)``.

    Gradient: ``scgrad = chol(H)⁻¹·g`` (R's ``solve(chol(H), g)``); flags when
    ``max(pmin(|scgrad|, |g|)) > tol`` — the parallel-minimum rule means a
    component is bad only if BOTH its scaled and raw gradient are large (flat
    curvature can blow up the scaled gradient alone). Hessian: per
    :func:`_check_hess`, β-block first (GLMM) then the full matrix.
    """
    messages: list[str] = []
    code = 0
    grad = np.asarray(grad, dtype=float).ravel()
    H = np.asarray(hess, dtype=float)
    if np.isnan(grad).any():
        return (["Gradient contains NAs"], -5)

    grad_action = (grad_cfg or {}).get("action", "warning")
    if grad_action != "ignore":
        grad_tol = float((grad_cfg or {}).get("tol", 2e-3))
        try:
            # R's chol(H) is the upper U with H = UᵀU; numpy's is the lower L
            # with H = LLᵀ, so U = Lᵀ and solve(U, g) = solve(Lᵀ, g).
            L = np.linalg.cholesky(0.5 * (H + H.T))
            scgrad = np.linalg.solve(L.T, grad)
            ok = not np.isnan(scgrad).any()
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            messages.append("unable to evaluate scaled gradient")
            code = -1
        else:
            mingrad = np.minimum(np.abs(scgrad), np.abs(grad))
            maxmingrad = float(mingrad.max())
            if maxmingrad > grad_tol:
                comp = int(np.argmax(mingrad)) + 1     # 1-based component
                code = -1
                messages.append(
                    f"Model failed to converge with max|grad| = "
                    f"{maxmingrad:g} (tol = {grad_tol:g}, component {comp})"
                )

    hess_action = (hess_cfg or {}).get("action", "warning")
    if hess_action != "ignore":
        hess_tol = float((hess_cfg or {}).get("tol", 1e-6))
        if H.shape[0] > n_theta:                       # GLMM: β-block first
            mb, cb = _check_hess(H[n_theta:, n_theta:], hess_tol,
                                 "fixed-effect")
            if cb != 0:
                messages.extend(mb)
                code = cb
        mh, ch = _check_hess(0.5 * (H + H.T), hess_tol)
        if ch != 0:
            messages.extend(mh)
            code = ch
    return (messages, code)


def _build_optinfo(
    *,
    theta: np.ndarray,
    theta_bounds: list,
    optim: dict,
    optim_stage0: Optional[dict],
    ctrl: Optional[dict],
    optimizer: tuple = ("bobyqa", "Nelder_Mead"),
    grad=None,
    hess=None,
    n_theta: Optional[int] = None,
    grad_cfg: Optional[dict] = None,
    hess_cfg: Optional[dict] = None,
) -> dict:
    """Port of lme4's ``m@optinfo`` (utilities.R:448) + ``checkConv`` runs.

    Fires the singular-fit check unconditionally (``check.conv.singular``,
    checkConv.R:32-48) and — when not singular and numerical ``grad``/``hess``
    of the Stage-1 deviance are supplied — the gradient and Hessian
    convergence diagnostics (8.14, :func:`_checkconv_grad_hess`). lme4 skips
    the gradient/Hessian checks for a singular fit, and bails entirely when
    no derivatives are available (``calc.derivs`` off). All messages surface
    in :meth:`gmm.summary`'s convergence block.
    """
    SINGULAR_TOL = 1e-4
    messages: list[str] = []

    theta_arr = np.asarray(theta, dtype=float).ravel()
    is_singular = False
    for th, (lo, hi) in zip(theta_arr, theta_bounds):
        lo_gap = th - lo if lo is not None else np.inf
        hi_gap = hi - th if hi is not None else np.inf
        if min(lo_gap, hi_gap) < SINGULAR_TOL:
            is_singular = True
            break
    if is_singular:
        messages.append("boundary (singular) fit: see help('isSingular')")

    derivs = None
    code = 0
    if grad is not None and hess is not None:
        derivs = {"gradient": np.asarray(grad, dtype=float),
                  "Hessian": np.asarray(hess, dtype=float)}
        # lme4 runs the gradient/Hessian checks only for a NON-singular fit.
        if not is_singular:
            conv_msgs, code = _checkconv_grad_hess(
                grad, hess,
                n_theta=n_theta if n_theta is not None else len(theta_arr),
                grad_cfg=grad_cfg, hess_cfg=hess_cfg,
            )
            messages.extend(conv_msgs)

    return {
        "optimizer": "+".join(optimizer),
        "control": dict(ctrl) if ctrl else {},
        "val": theta_arr.copy(),
        "feval": int(optim.get("feval", 0))
                 + (int(optim_stage0.get("feval", 0)) if optim_stage0 else 0),
        "is_singular": is_singular,
        "conv": {
            "opt": int(optim.get("status", 0)),
            "lme4": {"code": code, "messages": messages},
        },
        "derivs": derivs,
        "warnings": list(messages),
    }


def _nm_kwargs_from_opt_ctrl(opt_ctrl) -> dict:
    """Translate lme4's ``optCtrl`` dict to :class:`NelderMead` kwargs.

    lme4 uses R-flavoured names (``maxfun``, ``FtolAbs``, ``XtolRel``, …);
    the Python class uses snake_case (``maxeval``, ``ftol_abs``,
    ``xtol_rel``, …). Map both directions so a user's ``glmerControl(
    optCtrl=list(maxfun=2000))`` does what they expect.
    """
    if opt_ctrl is None or len(opt_ctrl) == 0:
        return {}
    out: dict = {}
    for key, val in opt_ctrl.items():
        if key == "maxfun":
            out["maxeval"] = int(val)
        elif key == "FtolAbs":
            out["ftol_abs"] = float(val)
        elif key == "FtolRel":
            out["ftol_rel"] = float(val)
        elif key == "XtolRel":
            out["xtol_rel"] = float(val)
        elif key == "MinfMax":
            out["minf_max"] = float(val)
        elif key == "verbose":
            # NelderMead doesn't print its own progress (lme4's wrapper does
            # at the R level). Accept but ignore so user code doesn't break.
            pass
        elif key in _BOBYQA_OPT_CTRL_KEYS:
            pass  # consumed by the bobyqa stage, not Nelder_Mead
        else:
            raise ValueError(
                f"unknown optCtrl key {key!r}; expected one of "
                f"{sorted(_NM_OPT_CTRL_KEYS | _BOBYQA_OPT_CTRL_KEYS)}"
            )
    return out


_NLOPT_OPT_CTRL_KEYS = {"maxeval", "ftol_abs", "ftol_rel", "xtol_abs",
                        "xtol_rel", "algorithm"}


def _nlopt_kwargs_from_opt_ctrl(opt_ctrl) -> dict:
    """Translate lme4's ``nloptwrap`` ``optCtrl`` to :func:`_nlopt_ln_bobyqa`
    kwargs. nloptwrap's control vocabulary (lme4 utilities.R:836-839) is
    NLopt-flavoured — ``maxeval``, ``ftol_abs`` / ``ftol_rel``, ``xtol_abs`` /
    ``xtol_rel`` — plus ``algorithm`` (only ``NLOPT_LN_BOBYQA``, lme4's lmer
    default, is ported). Empty ``optCtrl`` → ``{}``, i.e. the lme4 defaults,
    so the default fit stays byte-identical. Unknown keys raise.
    """
    if opt_ctrl is None or len(opt_ctrl) == 0:
        return {}
    out: dict = {}
    for key, val in opt_ctrl.items():
        if key == "maxeval":
            out["maxeval"] = int(val)
        elif key in ("ftol_abs", "ftol_rel", "xtol_abs", "xtol_rel"):
            out[key] = float(val)
        elif key == "algorithm":
            if val != "NLOPT_LN_BOBYQA":
                raise NotImplementedError(
                    f"nloptwrap algorithm={val!r}: only 'NLOPT_LN_BOBYQA' "
                    f"(lme4's lmer default) is ported"
                )
        else:
            raise ValueError(
                f"unknown optCtrl key {key!r} for nloptwrap; expected one of "
                f"{sorted(_NLOPT_OPT_CTRL_KEYS)}"
            )
    return out


def _do_check(action: object) -> bool:
    """lme4's ``doCheck`` (modular.R:6-8): a check runs unless its configured
    level is ``"ignore"`` (or not a string)."""
    return isinstance(action, str) and action != "ignore"


def _emit_check(action: str, msg: str) -> None:
    """Dispatch a pre-fit check message at its action level — ``stop`` raises,
    ``warning`` warns, ``message`` prints. ``ignore`` is filtered upstream by
    :func:`_do_check`."""
    if action == "stop":
        raise ValueError(msg)
    if action == "warning":
        warnings.warn(msg, stacklevel=3)
    elif action == "message":
        print(msg)


def _run_prefit_glmm_checks(re: ReTerms, y: np.ndarray, n: int,
                            ctrl: dict) -> None:
    """Port of lme4's pre-fit identifiability / response validation, run from
    ``lFormula``/``glFormula`` (modular.R): ``checkNlevels`` (167-212),
    ``checkZdims`` (63-85), ``checkZrank`` (88-117), ``checkResponse``
    (296-307). Each fires at its ``glmerControl(check.*=)`` action level
    (ignore/message/warning/stop). The X-side checks live elsewhere —
    ``chkRank.drop.cols`` in :func:`_check_rank_drop_cols`, and ``checkScaleX``
    with autoscale (8.16).
    """
    nlev = {g: len(levs) for g, levs in re.flist_levels.items()}

    # checkNlevels — modular.R:167-212.
    a = ctrl["check.nlev.gtr.1"]
    if _do_check(a) and any(v < 2 for v in nlev.values()):
        _emit_check(a, "grouping factors must have > 1 sampled level")
    a = ctrl["check.nobs.vs.nlev"]
    if _do_check(a):
        bad = [g for g, v in nlev.items() if v >= n]
        if bad:
            _emit_check(
                a, "number of levels of each grouping factor must be < number "
                f"of observations (problems: {', '.join(bad)})")
    a = ctrl["check.nlev.gtreq.5"]
    if _do_check(a) and any(v < 5 for v in nlev.values()):
        _emit_check(a, "grouping factors with < 5 sampled levels may give "
                    "unreliable estimates")

    # checkZdims — per-term #random-effects vs #obs (modular.R:63-85).
    a = ctrl["check.nobs.vs.nRE"]
    if _do_check(a):
        Gp = re.Gp
        for k, key in enumerate(re.cnms):
            n_re = Gp[k + 1] - Gp[k]
            if n_re >= n:
                _emit_check(
                    a, f"number of observations (={n}) <= number of random "
                    f"effects (={n_re}) for term ({key}); the random-effects "
                    "parameters and the residual variance (or scale parameter) "
                    "are probably unidentifiable")

    # checkZrank — rank(Z) vs #obs; opt-in (default "ignore") (modular.R:88-117).
    a = ctrl["check.nobs.vs.rankZ"]
    if _do_check(a):
        small = a.endswith("Small")
        if not (small and re.Z.size > 1e6):
            rank_z = int(np.linalg.matrix_rank(re.Z))
            if n <= rank_z:
                act = "warning" if a.startswith("warning") else "stop"
                _emit_check(
                    act, f"number of observations (={n}) <= rank(Z) (={rank_z});"
                    " the random-effects parameters are probably unidentifiable")

    # checkResponse — constant response (modular.R:296-307).
    a = ctrl["check.response.not.const"]
    if _do_check(a) and len(np.unique(np.asarray(y))) < 2:
        _emit_check(a, "Response is constant")


def _check_scale_x(X: np.ndarray, col_names: list, kind: str) -> None:
    """Port of lme4's ``checkScaleX`` (modular.R:128-158): warn/stop when
    continuous predictor columns are on very different scales — their SDs, or
    pairwise SD ratios, differ by more than ``tol = 1e3``. Fires at the
    ``check.scaleX`` level (ignore/warning/stop); the ``*+rescale`` variants
    map to their warn/message component — ``autoscale=True`` is the supported
    rescaling path (8.16)."""
    if not _do_check(kind) or X.shape[1] == 0:
        return
    tol = 1e3
    cont = np.array([not np.all(np.isin(X[:, j], (0.0, 1.0)))
                     for j in range(X.shape[1])])
    if not cont.any():
        return
    sd = X[:, cont].std(axis=0, ddof=1)
    sd = sd[sd > 0]
    if sd.size == 0:
        return
    logsd = np.abs(np.log(sd))
    iu = np.triu_indices(sd.size, k=1)
    logratio = (np.abs(np.log(np.divide.outer(sd, sd)))[iu]
                if iu[0].size else np.array([0.0]))
    if max(logsd.max(), logratio.max()) > np.log(tol):
        act = {"warn+rescale": "warning",
               "message+rescale": "message"}.get(kind, kind)
        _emit_check(act, "Some predictor variables are on very different "
                    "scales: consider rescaling (or use autoscale=True).")


def _restart_edge(devfun, par, lower, upper, refit, *, btol=1e-5, verbose=0):
    """Port of ``optimizeLmer``'s ``restart_edge`` (modular.R:690-738): if any
    parameter sits exactly on a bound and the inward finite-difference gradient
    is negative, restart the outer optimizer from ``par``. ``refit(par0)``
    re-runs the optimizer and returns the new vector. Returns ``par`` unchanged
    when no boundary improvement is found. (Largely a no-op for hea's
    gradient-based L-BFGS-B LMM optimizer, which won't halt at a false edge.)"""
    par0 = np.asarray(par, dtype=float).copy()
    wl = [i for i in range(len(par0)) if par0[i] == lower[i]]
    wu = [i for i in range(len(par0)) if par0[i] == upper[i]]
    if not wl and not wu:
        return par0
    d0 = devfun(par0)
    grads = []
    for i in wl:
        p = par0.copy()
        p[i] = lower[i] + btol
        grads.append((devfun(p) - d0) / btol)
    for i in wu:
        p = par0.copy()
        p[i] = upper[i] - btol
        grads.append((devfun(p) - d0) / (-btol))
    devfun(par0)  # reset internal state after probing
    grads = np.asarray(grads)
    if np.any(np.isnan(grads)):
        warnings.warn("some gradient components are NA near boundaries, "
                      "skipping boundary check", stacklevel=2)
        return par0
    if np.any(grads < 0):
        if verbose:
            print("some theta parameters on the boundary, restarting")
        return np.asarray(refit(par0), dtype=float)
    return par0


def _check_boundary(devfun, par, fval, lower, upper, boundary_tol, dpars=None):
    """Port of ``check.boundary`` (modular.R:879-907): for each covariance
    parameter within ``boundary_tol`` of a bound, pin it to the bound if that
    lowers the deviance. ``dpars`` restricts the check to the θ indices; ``None``
    checks all parameters."""
    par0 = np.asarray(par, dtype=float).copy()
    idx = range(len(par0)) if dpars is None else dpars
    for i in idx:
        dl, du = par0[i] - lower[i], upper[i] - par0[i]
        if 0 < dl < boundary_tol:
            test = par0.copy()
            test[i] = lower[i]
            if devfun(test) < fval:
                par0[i] = lower[i]
        elif 0 < du < boundary_tol:
            test = par0.copy()
            test[i] = upper[i]
            if devfun(test) < fval:
                par0[i] = upper[i]
    return par0


def _theta_ml(y, mu, weights=None, limit=20, eps=1e-8):
    """ML estimate of the negative-binomial dispersion θ given the response
    ``y`` and fitted means ``μ`` — port of ``MASS::theta.ml``. Newton iteration
    on the θ-score from a method-of-moments start ``n / Σ w(y/μ−1)²``.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    def score(th):
        return float(np.sum(w * (digamma(th + y) - digamma(th) + np.log(th)
                                 + 1 - np.log(th + mu) - (y + th) / (mu + th))))

    def info(th):
        return float(np.sum(w * (-polygamma(1, th + y) + polygamma(1, th)
                                 - 1 / th + 2 / (mu + th)
                                 - (y + th) / (mu + th) ** 2)))

    t0 = n / np.sum(w * (y / mu - 1) ** 2)
    it, delta = 0, 1.0
    while it < limit and abs(delta) > eps:
        t0 = abs(t0)
        delta = score(t0) / info(t0)
        t0 = t0 + delta
        it += 1
    return max(float(t0), 0.0)


def glmer_nb(formula, data, *, interval_width=3.0, tol=5e-5,
             _gmm_kwargs=None, **kwargs):
    """Negative-binomial GLMM with θ estimation — port of ``lme4::glmer.nb``
    (nbinom.R:96-159).

    Fits a Poisson GLMM, estimates the NB dispersion θ from its residuals
    (``MASS::theta.ml`` ⇒ :func:`_theta_ml`), then 1D-optimises the NB profile
    ``-2 log L`` over ``log θ`` (lme4's ``optTheta``: ``optimize`` over
    ``log(th)+c(-3,3)``), refitting the GLMM at each candidate θ. Returns the
    fit at θ̂ with ``m._nb_theta`` set and ``npar``/``AIC``/``BIC`` accounting
    for the estimated dispersion (lmer.R:1053 adds 1 to ``npar`` for nbinom).
    """
    kw = dict(_gmm_kwargs) if _gmm_kwargs is not None else dict(kwargs)
    kw.pop("family", None)

    # 1. Poisson initialization → θ start from the residuals.
    m0 = gmm(formula, data, family=_family_mod.Poisson(), **kw)
    th0 = _theta_ml(np.asarray(m0._resp.y, dtype=float),
                    np.asarray(m0.fitted_values, dtype=float),
                    m0.prior_weights)
    lo, hi = np.log(th0) - interval_width, np.log(th0) + interval_width

    # 2. profile -2logL over log θ (lme4 optTheta).
    def nbdev(t):
        return gmm(formula, data,
                   family=_family_mod.nb(theta=float(np.exp(t))),
                   **kw).deviance_laplace

    res = minimize_scalar(nbdev, bounds=(lo, hi), method="bounded",
                          options={"xatol": tol})
    th_hat = float(np.exp(res.x))

    # 3. final fit at θ̂; account for the estimated dispersion parameter.
    m = gmm(formula, data, family=_family_mod.nb(theta=th_hat), **kw)
    m._nb_theta = th_hat
    m.npar = m.npar + 1
    m.df_resid = m.n - m.npar
    m.AIC = m.deviance_laplace + 2.0 * m.npar
    m.BIC = m.deviance_laplace + np.log(m.n) * m.npar
    return m


def _simulate_family_draw(rng, family, mu, weights, sigma):
    """Draw one response per fitted mean in ``mu`` from ``family`` using R's
    nmath samplers on the bit-exact :class:`RMersenneTwister` stream — byte-
    matching lme4's per-family ``*_simfun`` (predict.R:1012-1090) draw-for-draw.
    """
    name = family.name
    m = len(mu)
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
    out = np.empty(m)
    if name == "poisson":
        for i in range(m):
            out[i] = rng.rpois(float(mu[i]))
    elif name == "binomial":
        for i in range(m):
            out[i] = rng.rbinom(int(round(w[i])), float(mu[i])) / w[i]
    elif name == "Gamma":
        disp = sigma ** 2                     # Gamma_simfun: shape=1/disp
        for i in range(m):
            out[i] = rng.rgamma(1.0 / disp, scale=float(mu[i]) * disp)
    elif name == "negative binomial":
        th = float(family.get_theta(trans=True)[0])
        for i in range(m):
            out[i] = rng.rnbinom(th, float(mu[i]))
    else:
        raise NotImplementedError(
            f"simulate is not implemented for family {name!r}")
    return out


def _simulate_rng(seed):
    """RNG for :meth:`gmm.simulate` / :meth:`gmm.bootMer`, mirroring R's
    ``simulate.merMod`` seed semantics on hea's bit-exact
    :class:`~hea.R.rng.RMersenneTwister` (never numpy).

    R keeps a **single** global ``.Random.seed``, so this shares the one
    process-global R stream that :func:`hea.R.set_seed` controls (it lives in
    :mod:`hea.R.distributions`): an explicit ``seed`` does ``set.seed(seed)`` —
    reseeding that shared stream — while ``seed=None`` continues it (lazily
    time-initialised the way R's ``Randomize()`` seeds ``.Random.seed`` when none
    exists yet). Thus ``hea.R.set_seed(k); model.simulate()`` draws the same
    stream as R's ``set.seed(k); simulate(model)``, and ``model.simulate(seed=k)``
    consumes the identical stream as seeding the public R surface with ``k`` —
    one ``set.seed`` controls one stream, exactly as in R.
    """
    from ..R import distributions as _dist
    if seed is not None:
        _dist.set_seed(int(seed))
    return _dist._r_rng()


def _check_rank_drop_cols(
    X: np.ndarray, col_names: list[str], *,
    tol: float = 1e-7, action: str = "message+drop.cols",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Detect rank-deficient design matrix and drop redundant columns.

    Port of lme4's ``chkRank.drop.cols`` (modular.R:235-293). Uses a
    pivoted QR to find a column subset of full rank, then drops the
    remaining columns. Mirrors the action levels of
    ``glmerControl(check.rankX=)``:

    * ``"ignore"`` → never drop, return as-is.
    * ``"silent.drop.cols"`` → drop, no message.
    * ``"message+drop.cols"`` (default) → drop and print a message.
    * ``"warn+drop.cols"`` → drop and emit ``UserWarning``.
    * ``"stop.deficient"`` → raise on rank deficiency.

    R's ``chkRank.drop.cols`` uses LINPACK QR pivoting (the same as
    ``lm()``); we use SciPy's LAPACK pivoted QR. For "obvious"
    deficiencies (one column is a linear combination of others) both
    methods identify the same dropped column, but pathological cases may
    differ; that's a known follow-up.

    Returns ``(X_kept, kept_names, dropped_names)``.
    """
    if action == "ignore":
        return X, list(col_names), []
    n, p = X.shape
    if p == 0:
        return X, list(col_names), []
    # Order-preserving Householder QR (no pivoting) — keep R[j,j] in the
    # original column ordering. A column is rank-deficient iff its
    # orthogonal component (R diagonal at that row) is small relative to
    # ``|R[0,0]|``. This matches R's ``lm()``/``glmer`` behaviour of
    # dropping the *later* of any pair of collinear columns — and matches
    # LINPACK QR's pivoting heuristic for typical designs (main effects
    # first, interactions later) while avoiding LAPACK-vs-LINPACK pivot
    # divergences. Mirrors R's modular.R:259 (``qr(X, LAPACK=FALSE)``)
    # via the simpler "first p_rank columns that add rank" semantics.
    _, R = _scipy_qr(X, mode="economic")
    diag = np.abs(np.diag(R))
    thresh = tol * (diag[0] if diag.size else 1.0)
    keep_mask = diag > thresh
    rank = int(keep_mask.sum())
    if rank == p:
        return X, list(col_names), []
    if action == "stop.deficient":
        raise ValueError(
            f"the fixed-effects model matrix is column rank deficient "
            f"(rank(X) = {rank} < {p} = p); the fixed effects will be "
            "jointly unidentifiable"
        )
    keep = np.flatnonzero(keep_mask).tolist()
    dropped = [col_names[j] for j in range(p) if not keep_mask[j]]
    msg = (
        f"fixed-effect model matrix is rank deficient so dropping "
        f"{p - rank} column / coefficient"
        if p - rank == 1
        else f"fixed-effect model matrix is rank deficient so dropping "
             f"{p - rank} columns / coefficients"
    )
    if action == "warn+drop.cols":
        warnings.warn(msg, UserWarning, stacklevel=3)
    elif action == "message+drop.cols":
        # lme4's ``message()`` writes to stderr; mirror.
        import sys
        print(msg, file=sys.stderr)
    # ``silent.drop.cols`` falls through without printing.
    X_kept = X[:, keep]
    kept_names = [col_names[j] for j in keep]
    return X_kept, kept_names, dropped


def _normalize_glmer_control(control) -> dict:
    """Merge ``control=`` with lme4's ``glmerControl()`` defaults.

    Unknown keys raise :class:`ValueError`. The ``optimizer`` entry is
    normalized to a two-stage ``[stage0, stage1]`` chain; optimizers other
    than the ported ``bobyqa`` / ``Nelder_Mead`` raise
    :class:`NotImplementedError`.
    """
    if control is None:
        merged = dict(_GLMER_CONTROL_DEFAULTS)
        merged["optCtrl"] = dict(merged["optCtrl"])  # don't share the default mapping
        return merged
    if not isinstance(control, dict):
        raise TypeError(
            f"control= must be a dict; got {type(control).__name__}"
        )
    bad = set(control) - set(_GLMER_CONTROL_DEFAULTS)
    if bad:
        raise ValueError(
            f"unknown control keys: {sorted(bad)}; expected one of "
            f"{sorted(_GLMER_CONTROL_DEFAULTS)}"
        )
    merged = dict(_GLMER_CONTROL_DEFAULTS)
    merged["optCtrl"] = dict(merged["optCtrl"])
    merged.update(control)
    merged["optimizer"] = _normalize_optimizer_chain(merged["optimizer"])
    return merged


def _normalize_lmer_control(control) -> dict:
    """Merge ``control=`` with lme4's ``lmerControl()`` defaults (the LMM /
    Gaussian-identity path). Unknown keys raise :class:`ValueError`. The
    ``optimizer`` entry is normalized to a single ported optimizer name
    (``nloptwrap`` / ``bobyqa`` / ``Nelder_Mead``); others raise
    :class:`NotImplementedError`.
    """
    if control is None:
        merged = dict(_LMER_CONTROL_DEFAULTS)
        merged["optCtrl"] = dict(merged["optCtrl"])  # don't share the default
        return merged
    if not isinstance(control, dict):
        raise TypeError(
            f"control= must be a dict; got {type(control).__name__}"
        )
    bad = set(control) - set(_LMER_CONTROL_DEFAULTS)
    if bad:
        raise ValueError(
            f"unknown control keys: {sorted(bad)}; expected one of "
            f"{sorted(_LMER_CONTROL_DEFAULTS)}"
        )
    merged = dict(_LMER_CONTROL_DEFAULTS)
    merged["optCtrl"] = dict(merged["optCtrl"])
    merged.update(control)
    merged["optimizer"] = _normalize_lmer_optimizer(merged["optimizer"])
    return merged


def _lmer_start_theta(start, theta_shape) -> np.ndarray:
    """Parse lmer ``start=`` to a θ₀ warm-start vector.

    lme4's profiled LMM deviance is over θ only (β is profiled out), so
    ``start`` is a θ vector (ndarray / sequence) or a dict carrying ``theta``
    (alias ``par``). A ``beta`` / ``fixef`` component is glmer-only and raises
    here — there is no β to warm-start on the LMM path. Mirrors lme4's
    ``getStart`` (modular.R:472-533, lmer branch).
    """
    if isinstance(start, dict):
        if "theta" in start and "par" in start:
            raise ValueError("start= must not have both 'theta' and 'par' keys")
        if "theta" in start or "par" in start:
            theta0 = np.asarray(
                start.get("theta", start.get("par")), dtype=float
            ).copy()
        else:
            raise ValueError(
                "lmer start= dict must supply 'theta' (or 'par'); the profiled "
                "LMM deviance has no β to warm-start"
            )
        if "beta" in start or "fixef" in start:
            raise ValueError(
                "lmer profiles β out of the deviance; start= takes only "
                "'theta'/'par', not 'beta'/'fixef'"
            )
        bad = set(start) - {"theta", "par"}
        if bad:
            raise ValueError(f"unrecognised start keys: {sorted(bad)}")
    else:
        theta0 = np.asarray(start, dtype=float).copy()
    if theta0.shape != theta_shape:
        raise ValueError(
            f"start theta has shape {theta0.shape}; expected {theta_shape}"
        )
    return theta0


class gmm:
    """Generalized mixed model — lme4's ``lmer`` + ``glmer`` under one class.

    Gaussian-identity → LMM (``lmer``: ML/REML profiled deviance); any other
    family → GLMM (``glmer``: Laplace). Dispatch on ``family`` is internal.

    Parameters
    ----------
    formula : str
        lme4-style mixed model formula, e.g.
        ``"Reaction ~ 1 + Days + (1+Days|Subject)"``.
    data : polars.DataFrame
        Data table; rows with NA in any referenced column are dropped
        before fitting.
    REML : bool, default True
        Fit by REML (matches lme4's default) or ML.

    Attributes (always set)
    -----------------------
    n, p, q : int
        Sample size, # of fixed-effect coefficients, # of random-effect
        coefficients (= total number of Z columns).
    n_groups : dict[str, int]
        Number of unique levels per (raw) grouping factor.
    sigma : float
        Residual SD (σ̂).
    bhat, se_bhat, t_values : polars.DataFrame
        Fixed-effect estimates / SEs / t-values, one row each, columns
        keyed by R-canonical fixed-effect names (``(Intercept)``,
        ``MachineB``, …).
    sd_re : dict[str, np.ndarray]
        Per-bar component SDs. Keyed by the disambiguated bar key from
        ``ReTerms.cnms`` (e.g. ``"Subject"``, ``"Subject.1"``). Length
        equals the bar's component count (1 for scalar bars).
    corr_re : dict[str, np.ndarray | None]
        Per-bar correlation matrix. ``None`` for scalar bars; a c×c
        matrix for vector bars.
    npar : int
        Total parameter count (fixed effects + θ + 1 residual variance);
        used for likelihood ratio tests.

    Attributes (REML=True only)
    ---------------------------
    REML_criterion : float
        Optimized REML criterion, ``-2 log L_REML``.

    Attributes (REML=False only)
    ----------------------------
    deviance : float
        Optimized ML deviance, ``-2 log L``.
    loglike : float
    df_resid : int
        ``n - npar`` (matches lme4's printed ``df.resid``).

    Attributes (both REML and ML)
    -----------------------------
    AIC, BIC : float
        Information criteria. For ML fits, computed from the ML deviance;
        for REML fits, from the REML criterion (matches lme4's ``AIC()``
        and ``BIC()``). REML AIC/BIC are only comparable across models
        with the same fixed-effects structure.
    """

    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        *,
        family: object = None,
        REML: bool = True,
        weights: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        mustart: Optional[np.ndarray] = None,
        etastart: Optional[np.ndarray] = None,
        nAGQ: int = 1,
        start=None,
        subset=None,
        na_action: str = "na.omit",
        contrasts: Optional[dict] = None,
        verbose: int = 0,
        devFunOnly: bool = False,
        control: Optional[dict] = None,
        nAGQ0initStep: bool = True,
    ):
        self.formula = formula

        # family= validation. Accept None/Family/callable/str; reject
        # quasi* with lme4's exact error message (modular.R:733-735).
        family = _resolve_lme_family(family)

        # glmer.nb: a free-θ negative-binomial family triggers the θ-
        # estimation outer loop (lme4::glmer.nb). Fixed-θ nb(theta=Θ) flows
        # through the standard Laplace path below. Delegate + adopt the fit.
        if isinstance(family, _family_mod.nb) and family.n_theta > 0:
            _m = glmer_nb(formula, data, _gmm_kwargs=dict(
                weights=weights, offset=offset, mustart=mustart,
                etastart=etastart, nAGQ=nAGQ, start=start, subset=subset,
                na_action=na_action, contrasts=contrasts, verbose=verbose,
                control=control, nAGQ0initStep=nAGQ0initStep))
            self.__dict__.update(_m.__dict__)
            self.formula = formula
            return

        # lmer (Gaussian-identity LMM) vs glmer (any other family) split the
        # control defaults, the optimizer surface, and the REML toggle below.
        _is_gaussian_id = (family.name == "gaussian"
                           and family.link.name == "identity")

        # REML is only meaningful for the Gaussian-identity LMM; glmer is
        # ML by construction (the Laplace approximation evaluates the
        # marginal log-likelihood directly). Silently override the default
        # ``REML=True`` for non-Gaussian-identity families so summary /
        # ``__repr__`` print AIC/BIC/logLik rather than reaching for a
        # non-existent ``REML_criterion``.
        if not _is_gaussian_id:
            REML = False
        self.REML = REML

        # nAGQ validation. Integer in [0, 100]; >1 awaits AGQ.
        nAGQ = _validate_nagq(nAGQ)
        # Snapshot the fit knobs that ``_refit_response`` (the bootMer /
        # simulate building block) must preserve when re-fitting on
        # a fresh response: nAGQ, the user-supplied numeric ``offset=`` arg
        # (the formula's ``offset(...)`` terms re-evaluate from data on their
        # own — passing them again would double-count), and the control dict.
        self._nAGQ = int(nAGQ)
        self._offset_arg = None if offset is None else np.asarray(offset, dtype=float)
        self._control_arg = control

        # control= normalization. lmer (Gaussian-identity) and glmer use
        # different lme4 control defaults — lmerControl vs glmerControl: the
        # default optimizer ("nloptwrap" vs c("bobyqa","Nelder_Mead")) and
        # check.conv.nparmax (10 vs 20). Unknown keys raise either way.
        ctrl = (_normalize_lmer_control(control) if _is_gaussian_id
                else _normalize_glmer_control(control))

        # subset= (R's row-filter) and na_action= (R's na.action).
        # subset accepts: bool mask, positive 1-based ints (keep), negative
        # 1-based ints (drop). Filtered before prepare_design so the NA-omit
        # policy that runs inside prepare_design sees the same row set R does.
        if subset is not None:
            data = _apply_subset(data, subset)
        # na_action: prepare_design always uses na.omit (R's default); explicit
        # "na.fail" mode checks for NAs in referenced columns and raises before
        # they're dropped. "na.pass" / "na.exclude" would require carrying NA
        # rows through PIRLS — defer.
        if na_action not in ("na.omit", "na.fail"):
            raise NotImplementedError(
                f"na_action={na_action!r}: only 'na.omit' (default) and "
                f"'na.fail' are supported. R's 'na.pass' / 'na.exclude' "
                f"require carrying NA rows through PIRLS and are deferred."
            )

        # contrasts= dict mapping factor-column name → R contrast name.
        # In-formula ``C(...)`` wraps win over this argument (R semantics).
        if contrasts is not None:
            valid_names = set(CONTRAST_FN_NAMES)
            for col, ctr in contrasts.items():
                if not isinstance(ctr, str) or ctr not in valid_names:
                    raise ValueError(
                        f"contrasts[{col!r}]={ctr!r}: only string names of "
                        f"R contrast functions are supported "
                        f"({sorted(valid_names)}). Numeric matrices and "
                        f"function references are deferred."
                    )
        _n_rows_before_na = data.height
        d = prepare_design(formula, data, contrasts=contrasts)
        if na_action == "na.fail" and d.data.height < _n_rows_before_na:
            raise ValueError(
                f"missing values in object ({_n_rows_before_na - d.data.height} "
                f"row(s) dropped due to NA); pass na_action='na.omit' to drop "
                f"them silently."
            )
        if not d.expanded.bars:
            raise ValueError(
                f"gmm requires at least one random-effect bar; got formula={formula!r}"
            )
        # materialize_bars is called on d.data (response-NA-cleaned) so it
        # applies the same NA-omit policy as materialize() did for X — the
        # resulting Z stays row-aligned with X.
        re = materialize_bars(d.expanded, d.data)
        # R's factor-response convention for Binomial (Y/N → 0/1 with the
        # second declared level as success); for other families this is a
        # plain float-cast. See :func:`hea.family._coerce_response`.
        y = _coerce_response(d.y, family)

        # Sum any ``offset(...)`` atoms from the formula. β̂, û and the
        # variance components are all unchanged by the offset; only the
        # fitted/residual scale shifts. ``y`` here is the *original* response
        # (response scale); ``_fit_from_components`` builds ``y_solve = y -
        # offset`` internally for the Gaussian fit.
        n = len(y)
        off = np.zeros(n)
        for off_node in d.expanded.offsets:
            off = off + _eval_atom(off_node, d.data).values.flatten().astype(float)
        # direct numeric ``offset=`` arg adds to the formula offset.
        if offset is not None:
            offset_arr = np.asarray(offset, dtype=float)
            if offset_arr.shape != (n,):
                raise ValueError(
                    f"offset= must have length {n}; got {offset_arr.shape}"
                )
            off = off + offset_arr

        # pre-fit identifiability / response validation (lme4's
        # checkNlevels / checkZdims / checkZrank / checkResponse). Runs before
        # the X-rank drop, mirroring lFormula/glFormula's ordering; each check
        # fires at its glmerControl(check.*=) action level.
        _run_prefit_glmm_checks(re, y, n, ctrl)

        # rank-deficient column drop (lme4's ``chkRank.drop.cols``).
        # Detect and drop columns at __init__ time so the fit only sees a
        # full-rank X. Without this, the inner-Cholesky in PIRLS dies on
        # rank-deficient designs (e.g. ``poly(age,2) + age:ch`` where
        # ``age:ch`` rebuilds a column that ``poly(age,2)`` already spans).
        X_arr = d.X.to_numpy().astype(float)
        if X_arr.shape == (0, 0):
            X_arr = np.zeros((n, 0))
        X_arr_kept, kept_names, dropped_names = _check_rank_drop_cols(
            X_arr, list(d.X.columns),
            tol=1e-7, action=ctrl["check.rankX"],
        )
        if dropped_names:
            X_for_fit = pl.DataFrame({c: X_arr_kept[:, i]
                                       for i, c in enumerate(kept_names)})
        else:
            X_for_fit = d.X
        self._dropped_cols = dropped_names

        # autoscale (lme4 modular.R:442-453): centre/scale the non-
        # (Intercept) X columns to unit SD before fitting, for conditioning.
        # Reversed post-fit (after _fit_from_components) so the reported β̂/vcov
        # stay in the user's units — the fit is invariant to the reparam.
        self._autoscale_center = None
        self._autoscale_scale = None
        if ctrl["autoscale"]:
            _cols = list(X_for_fit.columns)
            _Xa = X_for_fit.to_numpy().astype(float)
            _center = np.zeros(_Xa.shape[1])
            _scale = np.ones(_Xa.shape[1])
            for _j, _c in enumerate(_cols):
                if _c == "(Intercept)":
                    continue
                _m, _s = float(_Xa[:, _j].mean()), float(_Xa[:, _j].std(ddof=1))
                if _s > 0:
                    _Xa[:, _j] = (_Xa[:, _j] - _m) / _s
                    _center[_j], _scale[_j] = _m, _s
            X_for_fit = pl.DataFrame({_c: _Xa[:, _j]
                                      for _j, _c in enumerate(_cols)})
            self._autoscale_center = _center
            self._autoscale_scale = _scale

        # checkScaleX (modular.R:128-158, run at 461): flag predictors
        # on very different scales (post-autoscale, so quiet once scaled).
        _check_scale_x(X_for_fit.to_numpy().astype(float),
                       list(X_for_fit.columns), ctrl["check.scaleX"])

        # calc.derivs=None is lme4's actual default (NULL); resolve it to the
        # "smart" rule (lmer.R:51-53): compute the post-fit numerical Hessian
        # only when the data AND the (θ, β) optimisation vector are both small
        # enough that it's cheap. check.conv.nobsmax / nparmax are the gating
        # knobs. An explicit True/False from the user passes through unchanged.
        calc_derivs_resolved = ctrl["calc.derivs"]
        if calc_derivs_resolved is None:
            npar_opt = len(re.theta) + X_for_fit.width
            calc_derivs_resolved = bool(
                n < ctrl["check.conv.nobsmax"]
                and npar_opt < ctrl["check.conv.nparmax"]
            )

        # restart_edge defaults TRUE for lmer (Gaussian-identity),
        # FALSE/unsupported for glmer, unless the user set it explicitly.
        restart_edge_resolved = (
            ctrl["restart_edge"] if "restart_edge" in (control or {})
            else _is_gaussian_id
        )
        if restart_edge_resolved and not _is_gaussian_id:
            # lme4's optimizeGlmer raises for restart_edge=TRUE (modular.R:869).
            raise NotImplementedError(
                "restart_edge is not implemented for glmer (matches lme4's "
                "optimizeGlmer); use restart_edge=False."
            )

        fit_inputs = _FitInputs(
            X_df=X_for_fit,
            y=y,
            re_terms=re,
            offset=off,
            family=family,
            reml=REML,
            weights=weights,
            mustart=mustart,
            etastart=etastart,
            start=start,
            nagq0_init_step=ctrl["nAGQ0initStep"]
                if "nAGQ0initStep" in (control or {}) else nAGQ0initStep,
            nAGQ=nAGQ,
            tol_pwrss=ctrl["tolPwrss"],
            calc_derivs=calc_derivs_resolved,
            use_last_params=ctrl["use.last.params"],
            verbose=verbose,
            opt_ctrl=ctrl["optCtrl"],
            optimizer=(ctrl["optimizer"] if _is_gaussian_id
                       else tuple(ctrl["optimizer"])),
            check_conv_grad=ctrl["check.conv.grad"],
            check_conv_hess=ctrl["check.conv.hess"],
            expanded=d.expanded,
            data=d.data,
            dev_fun_only=devFunOnly,
            restart_edge=restart_edge_resolved,
            boundary_tol=ctrl["boundary.tol"],
        )

        # ``devFunOnly=True``: _fit_from_components builds the stage
        # deviance closure, stores it on ``self.devfun`` (a _DevFunHandle), and
        # returns before optimizing — so gmm() hands back the unfitted instance
        # carrying that callable handle (lme4's diagnostic entry point).
        self._fit_from_components(fit_inputs)
        if devFunOnly:
            return

        # undo autoscale on the reported fixed effects: map β̂ and its
        # vcov from the scaled parameterisation back to the original predictor
        # units (lme4 fixef.merMod:973-982). β_orig[j]=β_s[j]/s[j];
        # β_orig[icpt] -= Σ_j β_s[j]·c[j]/s[j]. summary() reads _beta/_se_beta
        # directly (covers GLMM); the bhat/se DataFrames cover the Gaussian API.
        if self._autoscale_scale is not None:
            cols = list(self.column_names)
            s, c = self._autoscale_scale, self._autoscale_center
            icpt = cols.index("(Intercept)") if "(Intercept)" in cols else None
            T = np.eye(len(cols))
            for j in range(len(cols)):
                if j == icpt:
                    continue
                T[j, j] = 1.0 / s[j]
                if icpt is not None:
                    T[icpt, j] = -c[j] / s[j]
            self._beta = T @ self._beta
            self._vcov_beta_arr = T @ self._vcov_beta_arr @ T.T
            self._se_beta = np.sqrt(np.diag(self._vcov_beta_arr))
            _b, _se = self._beta, self._se_beta

            def _df1(arr):
                return pl.DataFrame({cc: [float(arr[i])]
                                     for i, cc in enumerate(cols)})
            self.bhat = _df1(_b)
            self.fixef = self.bhat
            if hasattr(self, "se_bhat"):
                self.se_bhat = _df1(_se)
            if hasattr(self, "t_values"):
                self.t_values = _df1(_b / _se)
            if hasattr(self, "z_values"):
                self.z_values = _df1(_b / _se)
            if hasattr(self, "vcov_beta"):
                self.vcov_beta = pl.DataFrame(
                    {cc: self._vcov_beta_arr[:, i] for i, cc in enumerate(cols)})

    def _fit_from_components(self, inputs: _FitInputs) -> None:
        """Fit the model given pre-assembled design pieces.

        Public ``gmm()`` calls this after running ``prepare_design`` and
        ``materialize_bars``. External callers (``hea.gamm``) call it
        directly after composing smooth random-effect blocks via
        ``smooth2random`` — bypassing the formula parser entirely.

        Dispatches on ``inputs.family``: Gaussian-identity uses the
        profiled-deviance + CHOLMOD path implemented here. Other families
        take the GLMM Laplace path.
        """
        is_gaussian_identity = (
            inputs.family.name == "gaussian"
            and inputs.family.link.name == "identity"
        )
        if not is_gaussian_identity:
            self._fit_glmm_from_components(inputs)
            return
        if inputs.weights is not None:
            raise NotImplementedError(
                "weights= is plumbed through _FitInputs but the Gaussian fit "
                "path does not yet honour non-unit weights."
            )

        # Unpack inputs onto self — same attributes the original __init__ set.
        re = inputs.re_terms
        X_df = inputs.X_df
        y = inputs.y
        X = X_df.to_numpy().astype(float)
        Z = re.Z
        n, p = X.shape
        q = Z.shape[1]
        off = inputs.offset
        y_solve = y - off
        REML = inputs.reml

        self.family = inputs.family
        self._offset = off
        self.data = inputs.data
        self._expanded = inputs.expanded
        self.X = X_df
        self.y = y
        self._y_solve = y_solve
        self.Z = Z
        self.column_names = list(X_df.columns)
        self.n = n
        self.p = p
        self.q = q
        self._re = re

        bar_sizes = _bar_sizes(re.cnms)
        self._bar_sizes = bar_sizes
        self.n_groups = {g: len(levs) for g, levs in re.flist_levels.items()}

        # ------------- profiled-deviance optimization ----------------------
        #
        # Z and Λᵀ are stored sparse (CSC). The hot step — the Cholesky of
        # ``M = Λ Zᵀ Z Λᵀ + I`` — goes through ``sksparse.cholmod`` (CHOLMOD
        # with AMD reordering). The symbolic factor is computed once on the
        # first factorization and reused by ``factor.factorize(M_new)`` every
        # subsequent call; only the numeric re-factor runs inside the
        # optimizer loop. Without this, InstEval-class fits (q ≈ 4k) sit in
        # dense Cholesky for O(q³) flops per deviance eval.
        template = re.Lambdat
        lt_theta_pos, lt_indices, lt_indptr = _sparse_Lt_spec(template)
        Z_sp = csc_array(Z)
        eye_q_sp = eye_array(q, format="csc")
        XtX = X.T @ X
        Xty = X.T @ y_solve
        yty = float(y_solve @ y_solve)
        log2pi = float(np.log(2.0 * np.pi))

        # Cache pieces profile() and other post-fit methods reuse.
        self._template = template
        self._lt_theta_pos = lt_theta_pos
        self._lt_indices = lt_indices
        self._lt_indptr = lt_indptr
        self._lt_shape = template.shape
        self._Z_sp = Z_sp
        self._eye_q_sp = eye_q_sp
        self._chol_factor = None
        self._XtX = XtX
        self._Xty = Xty
        self._yty = yty
        self._log2pi = log2pi

        diag_set = set(_theta_diag_idx(bar_sizes))
        self._diag_set = diag_set
        bounds = [
            (0.0, None) if i in diag_set else (None, None)
            for i in range(len(re.theta))
        ]
        self._theta_bounds = bounds

        # devFunOnly: hand back the lmer profiled-deviance closure over θ.
        if inputs.dev_fun_only:
            tnames = [f"theta{i + 1}" for i in range(len(re.theta))]
            lb = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
            ub = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
            fn = self._reml_deviance if inputs.reml else self._ml_deviance
            self.devfun = _DevFunHandle(fn, tnames, lb, ub)
            self._dev_fun_only = True
            return

        theta0 = re.theta.astype(float).copy()
        if inputs.start is not None:
            # lmer start= warm-start (lme4 getStart, modular.R:472-533): θ only.
            theta0 = _lmer_start_theta(inputs.start, re.theta.shape)
        _devfun_g = self._reml_deviance if REML else self._ml_deviance
        _lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        _hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
        # Outer optimizer over the profiled deviance — honor
        # ``lmerControl(optimizer=)`` / ``optCtrl``. lme4's DEFAULT is
        # ``nloptwrap`` = NLopt LN_BOBYQA (utilities.R:836-839,
        # xtol_abs=ftol_abs=1e-8, maxeval=1e5): the default (empty-optCtrl)
        # path below is byte-identical to before, landing θ̂ on lme4's fit to
        # the CHOLMOD floor (~1e-9). ``bobyqa`` (minqa) and ``Nelder_Mead``
        # (lme4 bounded NM) are the other ported lmer optimizers; they run the
        # SAME profiled deviance single-stage via _run_outer_stage (lme4's
        # default θ step xst=0.02, xt=xst·5e-4 — optimizer.R:5).
        optimizer = inputs.optimizer
        if not isinstance(optimizer, str):
            # the _FitInputs tuple default (e.g. a direct construction that
            # bypasses lmerControl) means "no explicit optimizer" → lme4 default.
            optimizer = "nloptwrap"
        if optimizer == "nloptwrap":
            _nlopt_kw = _nlopt_kwargs_from_opt_ctrl(inputs.opt_ctrl)

            def _run_opt(p0):
                return _nlopt_ln_bobyqa(_devfun_g, p0, _lo, _hi, **_nlopt_kw)
        else:
            _nm_kw = _nm_kwargs_from_opt_ctrl(inputs.opt_ctrl)
            _bob_kw = _bobyqa_kwargs_from_opt_ctrl(inputs.opt_ctrl)

            def _run_opt(p0):
                xst = np.full(p0.size, 0.02)
                par, fval, nf, status = _run_outer_stage(
                    optimizer, _devfun_g, p0, _lo, _hi,
                    xst=xst, xtol_abs=xst * 5e-4,
                    nm_kwargs=_nm_kw, bobyqa_kwargs=_bob_kw)
                return _NloptResult(np.asarray(par, float), float(fval),
                                    int(nf), status == 0, "")

        res = _run_opt(theta0)
        theta_hat = res.x

        # boundary handling on θ (lme4 optimizeLmer:688-740).
        # restart_edge is a near-no-op for BOBYQA (derivative-free, won't halt
        # at a false edge) but ported for parity; check.boundary pins near-zero
        # variance params to 0. The restart refit reuses the chosen optimizer.
        if inputs.restart_edge:
            theta_hat = _restart_edge(_devfun_g, theta_hat, _lo, _hi,
                                      lambda p0: _run_opt(p0).x,
                                      verbose=inputs.verbose)
        if inputs.boundary_tol > 0:
            theta_hat = _check_boundary(_devfun_g, theta_hat,
                                        _devfun_g(theta_hat), _lo, _hi,
                                        inputs.boundary_tol)
        if not np.array_equal(theta_hat, res.x):
            res.x = theta_hat
            res.fun = _devfun_g(theta_hat)
        self.theta = theta_hat
        self._optim = res

        # ------------- recover β̂, σ̂, SE(β̂) at the optimum ------------------
        #
        # Same Cholesky-based profile-deviance math as ``_chol_block``, but
        # we also keep β̂ and û here (the deviance loop discards them).
        # ``F⁻¹ = M⁻¹`` lets us evaluate ``cu' cu`` and ``RZX' RZX`` as inner
        # products against ``M⁻¹(ZLᵀy)`` and ``M⁻¹(ZLᵀX)`` without ever
        # materializing ``cu`` or ``RZX``.
        Lt = self._build_Lt_sparse(theta_hat)
        ZL = Z_sp @ Lt.T
        M = (ZL.T @ ZL + eye_q_sp).tocsc()
        if self._chol_factor is None:
            self._chol_factor = cho_factor(M)
        else:
            self._chol_factor.factorize(M)
        F = self._chol_factor

        # Snapshot Λ and L at the MLE as dense ndarrays (matches m.Z's
        # convention). profile()/_ranef() re-factorize _chol_factor at
        # non-MLE θ, so freezing copies here detaches us from those.
        # L is in CHOLMOD's permuted ordering — lower triangular by
        # construction; that's also the ordering Bates' Fig 2.4 shows.
        self.Lambda = Lt.T.toarray()
        self.L = F.L.toarray()

        # Use the offset-stripped response so this final β̂/û recompute is
        # consistent with the cached Xty/yty the optimizer ran on.
        ZLty = np.asarray(ZL.T @ y_solve).ravel()
        ZLtX = np.asarray(ZL.T @ X)
        M_inv_ZLty = F.solve(ZLty)
        M_inv_ZLtX = F.solve(ZLtX)
        # See _chol_block for why this reach uses einsum instead of @.
        cu_sq = float(np.einsum("i,i->", ZLty, M_inv_ZLty))
        XtX_eff = XtX - np.einsum("ij,ik->jk", ZLtX, M_inv_ZLtX)
        Rx = np.linalg.cholesky(XtX_eff)
        rhs = Xty - np.einsum("ij,i->j", ZLtX, M_inv_ZLty)
        cb = solve_triangular(Rx, rhs, lower=True)
        beta = solve_triangular(Rx.T, cb, lower=False)
        rss = yty - cu_sq - float(np.einsum("i,i->", cb, cb))
        # spherical random-effect coefficients u = M⁻¹(ZLᵀy − ZLᵀX β)
        self._u = F.solve(ZLty - np.einsum("ij,j->i", ZLtX, beta))

        sigma2 = rss / (n - p) if REML else rss / n
        sigma = float(np.sqrt(sigma2))
        self.sigma = sigma
        self.sigma_squared = sigma2

        # Fitted values ŷ = Xβ̂ + Z Λ û + offset (response scale).
        # Residuals = y − ŷ = y_solve − Xβ̂ − Z Λ û (offset cancels).
        self.fitted = np.einsum("ij,j->i", X, beta) + ZL @ self._u + off
        self.residuals = y - self.fitted
        # ε̂ / σ̂ — what lme4 calls Pearson / "Scaled residuals"
        self.scaled_residuals = self.residuals / sigma

        # Var(β̂) = σ̂² (XᵀX_eff)⁻¹ = σ̂² R_x⁻ᵀ R_x⁻¹
        Rx_inv = solve_triangular(Rx, np.eye(p), lower=True)
        vcov_beta = sigma2 * np.einsum("ij,ik->jk", Rx_inv, Rx_inv)
        se_beta = np.sqrt(np.diag(vcov_beta))
        self._vcov_beta_arr = vcov_beta
        self.vcov_beta = pl.DataFrame(
            {c: vcov_beta[:, i] for i, c in enumerate(self.column_names)}
        )

        self._beta = beta
        self._se_beta = se_beta
        self.bhat = pl.DataFrame(
            {c: [float(beta[i])] for i, c in enumerate(self.column_names)}
        )
        self.fixef = self.bhat                            # R-canonical alias
        self.se_bhat = pl.DataFrame(
            {c: [float(se_beta[i])] for i, c in enumerate(self.column_names)}
        )
        t_vals = beta / se_beta
        self.t_values = pl.DataFrame(
            {c: [float(t_vals[i])] for i, c in enumerate(self.column_names)}
        )

        # ------------- per-bar variance components -------------------------
        Sigma_blocks = _per_bar_relative_cov(theta_hat, bar_sizes)
        self.sd_re: dict[str, np.ndarray] = {}
        self.corr_re: dict[str, np.ndarray | None] = {}
        for key, Sigma in zip(re.cnms.keys(), Sigma_blocks):
            d = np.sqrt(np.diag(Sigma))
            self.sd_re[key] = sigma * d
            if Sigma.shape[0] > 1:
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = Sigma / np.outer(d, d)
                corr = np.where(np.isfinite(corr), corr, 0.0)
                np.fill_diagonal(corr, 1.0)
                self.corr_re[key] = corr
            else:
                self.corr_re[key] = None

        # ------------- summary statistics ----------------------------------
        # npar = fixed-effect coefficients + θ entries + 1 residual variance
        self.npar = p + len(theta_hat) + 1
        opt = float(res.fun)
        if REML:
            self.REML_criterion = opt
        else:
            self.deviance = opt
            self.loglike = -0.5 * opt
            self.df_resid = n - self.npar
        # AIC/BIC use the ML deviance for ML fits and the REML criterion
        # for REML fits, matching lme4's ``AIC.merMod`` / ``BIC.merMod``.
        self.AIC = opt + 2.0 * self.npar
        self.BIC = opt + np.log(n) * self.npar

    # ---- GLMM fit -------------------------------------------------------

    def _fit_glmm_from_components(self, inputs: _FitInputs) -> None:
        """Fit a GLMM by Laplace approximation. Mirrors ``glmer`` (lmer.R:148-198).

        Two-stage outer optimization:

        1. **Stage 0** (``nAGQ0initStep=True`` default): optimize the Laplace
           deviance over θ only, with PIRLS doing a joint (β, u) solve each
           call. Provides a warm start for Stage 1.
        2. **Stage 1**: optimize over (θ, β) jointly. β is folded into the
           offset and PIRLS does a u-only solve. Returns the final estimates.

        Both stages use scipy's L-BFGS-B with finite-difference gradients —
        derivative-free in spirit, matching lme4's bobyqa/Nelder_Mead choices.

        The instance gets just the bare minimum after this method: ``theta``,
        ``_beta``, ``bhat``/``fixef``, ``deviance``, plus the live
        ``_pred``/``_resp`` for downstream consumers. Full post-fit attributes
        (``fitted``, ``residuals``, ``AIC``, ``logLik``, ``sigma`` for
        unknown-scale families, σ-component summary tables, plotting hooks)
        are filled in by the post-fit attribute pass.
        """
        re = inputs.re_terms
        X_df = inputs.X_df
        y = inputs.y
        family = inputs.family
        X = X_df.to_numpy().astype(float)
        # polars to_numpy on a 0-column DataFrame returns shape (0, 0); fix
        # to (n, 0) so _PredState's row-count check passes.
        if X.shape == (0, 0):
            X = np.zeros((len(y), 0), dtype=float)
        Z = re.Z
        Z_sp = csc_array(Z)
        n, p = X.shape
        q = Z.shape[1]
        off = inputs.offset

        self.family = family
        self._offset = off
        self.data = inputs.data
        self._expanded = inputs.expanded
        self.X = X_df
        self.y = y
        self.Z = Z
        self.column_names = list(X_df.columns)
        self.n = n
        self.p = p
        self.q = q
        self._re = re

        bar_sizes = _bar_sizes(re.cnms)
        self._bar_sizes = bar_sizes
        self.n_groups = {g: len(levs) for g, levs in re.flist_levels.items()}

        diag_set = set(_theta_diag_idx(bar_sizes))
        self._diag_set = diag_set
        bounds_theta = [
            (0.0, None) if i in diag_set else (None, None)
            for i in range(len(re.theta))
        ]
        self._theta_bounds = bounds_theta
        n_theta = len(re.theta)

        # Build the live PIRLS state. _PredState holds X, Z, Λᵀ(θ);
        # _GlmResponse holds y, weights, offset, μ, and the working-weight
        # state PIRLS mutates each iteration.
        pred = _PredState(X, Z_sp, re)
        resp = _GlmResponse(
            family, y,
            weights=inputs.weights, offset=off,
            mustart=inputs.mustart, etastart=inputs.etastart,
        )

        theta0 = re.theta.astype(float).copy()
        # User-supplied starting values override the defaults. Mirror lme4's
        # ``getStart`` (modular.R:472-533): None → no override; ndarray →
        # θ-only; dict → keys ``theta``/``par`` and ``beta``/``fixef``.
        beta_user_start: Optional[np.ndarray] = None
        if inputs.start is not None:
            if isinstance(inputs.start, dict):
                if "theta" in inputs.start and "par" in inputs.start:
                    raise ValueError(
                        "start= must not have both 'theta' and 'par' keys"
                    )
                if "beta" in inputs.start and "fixef" in inputs.start:
                    raise ValueError(
                        "start= must not have both 'beta' and 'fixef' keys"
                    )
                if "theta" in inputs.start or "par" in inputs.start:
                    theta0 = np.asarray(
                        inputs.start.get("theta", inputs.start.get("par")),
                        dtype=float,
                    ).copy()
                    if theta0.shape != re.theta.shape:
                        raise ValueError(
                            f"start theta has shape {theta0.shape}; expected "
                            f"{re.theta.shape}"
                        )
                if "beta" in inputs.start or "fixef" in inputs.start:
                    beta_user_start = np.asarray(
                        inputs.start.get("beta", inputs.start.get("fixef")),
                        dtype=float,
                    ).copy()
                    if beta_user_start.shape != (p,):
                        raise ValueError(
                            f"start beta has shape {beta_user_start.shape}; "
                            f"expected ({p},)"
                        )
                bad = set(inputs.start) - {"theta", "par", "beta", "fixef"}
                if bad:
                    raise ValueError(f"unrecognised start keys: {sorted(bad)}")
            else:
                theta0 = np.asarray(inputs.start, dtype=float).copy()
                if theta0.shape != re.theta.shape:
                    raise ValueError(
                        f"start has shape {theta0.shape}; expected "
                        f"{re.theta.shape}"
                    )

        nagq0_init_step = inputs.nagq0_init_step
        # nAGQ=0 → skip Stage 1 entirely (LMM-style θ-only outer loop).
        # We still warm-start with a joint PIRLS so the Stage 0 closure
        # sees ``lp0`` at the conditional mode of (β, u) at θ₀.
        do_stage1 = inputs.nAGQ != 0
        # PIRLS inner-loop control, sourced from ``glmerControl(...)``.
        tol_pwrss = inputs.tol_pwrss
        maxit_pwrss = inputs.maxit_pwrss
        # Two-stage outer optimizer chain (glmerControl(optimizer=)). Stage 0
        # runs optimizer[0], Stage 1 runs optimizer[1]; each ∈ {bobyqa,
        # Nelder_Mead}, both ported. Default c("bobyqa","Nelder_Mead").
        optimizer = inputs.optimizer
        verbose_pirls = max(0, inputs.verbose - 2)  # PIRLS prints at v > 2

        # Translate the (lower, upper) tuple bounds into the arrays
        # :class:`NelderMead` expects, with ±inf for one-sided bounds.
        lb_theta = np.array(
            [-np.inf if lo is None else float(lo) for (lo, _) in bounds_theta]
        )
        ub_theta = np.array(
            [np.inf if hi is None else float(hi) for (_, hi) in bounds_theta]
        )
        lb_beta = np.full(p, -np.inf)
        ub_beta = np.full(p, np.inf)

        # Optional Nelder-Mead overrides from ``glmerControl(optCtrl=...)``.
        nm_kwargs = _nm_kwargs_from_opt_ctrl(inputs.opt_ctrl)
        bobyqa_kwargs = _bobyqa_kwargs_from_opt_ctrl(inputs.opt_ctrl)

        if nagq0_init_step or not do_stage1:
            # Stage 0 — joint PIRLS at θ₀, then optimize devfun over θ only.
            # When nAGQ=0 this IS the final fit (skip Stage 1 below).
            #
            # lme4 default optimizer chain is ``c("bobyqa", "Nelder_Mead")``
            # (glmerControl): BOBYQA for Stage 0, Nelder-Mead for Stage 1.
            # Our ported BOBYQA matches R's at the algorithm level; converged
            # θ̂ lands within rhoend (~2e-7) of R's, with β̂ following at the
            # FD-Hessian-amplified scale (1e-7 to 1e-3 depending on contrast).
            _pwrss_update(pred, resp, u_only=False, tol=tol_pwrss,
                          maxit=maxit_pwrss, verbose=verbose_pirls)
            devfun_stage0 = _glmm_devfun_factory(
                pred, resp, nagq=0, tol_pwrss=tol_pwrss,
                maxit_pwrss=maxit_pwrss, verbose=verbose_pirls,
            )
            # devFunOnly with nAGQ=0: return the θ-only Stage-0 closure
            # before optimizing (lme4 lmer.R:151).
            if inputs.dev_fun_only and not do_stage1:
                tnames = [f"theta{i + 1}" for i in range(n_theta)]
                self.devfun = _DevFunHandle(
                    devfun_stage0, tnames, lb_theta, ub_theta)
                self._dev_fun_only = True
                return
            # Stage 0 outer optimizer = optimizer[0]. BOBYQA self-scales via
            # minqa's defaults (rhobeg=min(0.95,0.2·max|par|), rhoend=1e-6·
            # rhobeg, npt=min(n+2,2n), maxfun=10000); Nelder-Mead uses lme4's
            # default θ step xst=0.02 (optimizer.R:5). _run_outer_stage clamps
            # ±inf bounds for BOBYQA and passes them through for Nelder-Mead.
            xst0 = np.full(n_theta, 0.02)
            theta_stage0_x, fval0, feval0, ierr0 = _run_outer_stage(
                optimizer[0], devfun_stage0, theta0, lb_theta, ub_theta,
                xst=xst0, xtol_abs=xst0 * 5e-4, nm_kwargs=nm_kwargs,
                bobyqa_kwargs=bobyqa_kwargs,
            )
            theta_stage0 = np.asarray(theta_stage0_x, dtype=float)
            # Re-anchor pred/resp at the Stage 0 optimum.
            devfun_stage0(theta_stage0)
            # β at Stage 0 optimum (= pp.delb after the joint PIRLS).
            # modular.R:475: ``fixef0 <- rho$pp$delb``. A user-supplied
            # ``start["beta"]`` overrides for the Stage 1 starting β.
            beta_start = beta_user_start if beta_user_start is not None else pred.beta(1.0).copy()
            self._optim_stage0 = {
                "par": theta_stage0, "fval": float(fval0),
                "feval": int(feval0), "status": int(ierr0),
            }
        else:
            # No Stage 0 — go straight to Stage 1 with θ₀ and β=0 (or
            # user-supplied β).
            _pwrss_update(pred, resp, u_only=True, tol=tol_pwrss,
                          maxit=maxit_pwrss, verbose=verbose_pirls)
            theta_stage0 = theta0
            beta_start = beta_user_start if beta_user_start is not None else np.zeros(p)
            self._optim_stage0 = None

        if do_stage1:
            # Stage 1 — optimize over (θ, β) jointly. β is folded into the
            # offset; PIRLS uses u_only=True. The factory snapshots lp0 at
            # the current (post-Stage-0) state and base_offset = resp.offset.
            #
            # nAGQ=1 → Laplace; nAGQ>1 → adaptive Gauss-Hermite.
            # AGQ requires a single scalar RE (modular.R:918-920) and feeds the
            # factory the GH rule + per-obs grouping codes.
            nagq_stage1 = inputs.nAGQ
            agq_kwargs: dict = {}
            if nagq_stage1 > 1:
                # Single scalar RE term: one grouping factor, one bar, one
                # component (modular.R:918-920). ``cnms`` values are a string
                # for scalar bars / a list for vector bars, so count via
                # ``_bar_sizes`` rather than ``len`` on the value.
                if (len(re.flist_levels) != 1 or len(re.cnms) != 1
                        or _bar_sizes(re.cnms)[0] != 1):
                    raise ValueError(
                        "nAGQ > 1 is only available for models with a single, "
                        "scalar random-effects term"
                    )
                # fac = per-obs 0-based level code. Under the single-scalar-RE
                # constraint Z is the n×q indicator (one nonzero per row), so
                # the CSR column indices give each obs's level — in the same
                # u-ordering used by sd / devc0.
                z_csr = Z_sp.tocsr()
                if z_csr.nnz != n:
                    raise ValueError(
                        "AGQ: random-effects design is not a clean indicator "
                        "(expected one nonzero per row)"
                    )
                agq_kwargs = {
                    "gqmat": _gh_rule(nagq_stage1),
                    "fac": z_csr.indices.astype(np.intp),
                    "n_levels": q,
                }
            devfun_stage1 = _glmm_devfun_factory(
                pred, resp, nagq=nagq_stage1, tol_pwrss=tol_pwrss,
                maxit_pwrss=maxit_pwrss, verbose=verbose_pirls, **agq_kwargs,
            )
            # devFunOnly with nAGQ>0: return the [θ, β] Stage-1 closure
            # (built at the Stage-0 optimum) before optimizing (lme4 lmer.R:175).
            if inputs.dev_fun_only:
                tnames = [f"theta{i + 1}" for i in range(n_theta)]
                bnames = list(inputs.X_df.columns)
                self.devfun = _DevFunHandle(
                    devfun_stage1, tnames + bnames,
                    np.concatenate([lb_theta, lb_beta]),
                    np.concatenate([ub_theta, ub_beta]))
                self._dev_fun_only = True
                return
            start_par = np.concatenate([theta_stage0, beta_start])
            lb_par = np.concatenate([lb_theta, lb_beta])
            ub_par = np.concatenate([ub_theta, ub_beta])
            # Stage 1 step sizes — lme4's ``adj=TRUE`` tweak at
            # lmer.R:2533-2540: θ block uses 0.1, β block uses
            # ``min(βSD, 10)``, all scaled by 0.2.
            beta_sd = _beta_sd_from_RX(pred.RX) if p > 0 else np.zeros(0)
            xst1 = 0.2 * np.concatenate([
                np.full(n_theta, 0.1),
                np.minimum(beta_sd, 10.0),
            ])
            xt1 = xst1 * 5e-4
            # Stage 1 outer optimizer = optimizer[1] (lme4 default Nelder_Mead;
            # the βSD-scaled xst1 / xt1 step+tol apply to the NM branch, and
            # BOBYQA self-scales over the joint (θ, β) vector).
            theta_beta_x, fval1, feval1, status1 = _run_outer_stage(
                optimizer[1], devfun_stage1, start_par, lb_par, ub_par,
                xst=xst1, xtol_abs=xt1, nm_kwargs=nm_kwargs,
                bobyqa_kwargs=bobyqa_kwargs,
            )
            # check.boundary on the θ block (modular.R:871-872): pin a
            # near-zero variance param to 0 if it lowers the deviance.
            if inputs.boundary_tol > 0:
                theta_beta_x = _check_boundary(
                    devfun_stage1, theta_beta_x, fval1, lb_par, ub_par,
                    inputs.boundary_tol, dpars=range(n_theta))
                fval1 = devfun_stage1(theta_beta_x)
            theta_hat = theta_beta_x[:n_theta].copy()
            beta_hat = theta_beta_x[n_theta:].copy()
            devfun_stage1(theta_beta_x)
            self._optim = {
                "par": theta_beta_x.copy(), "fval": fval1,
                "feval": feval1, "status": status1,
            }
            self._devfun_stage1 = devfun_stage1  # convergence diagnostics reuse this
        else:
            # nAGQ=0 — Stage 0 IS the final fit. θ̂ from NM, β̂ from the
            # converged PIRLS at θ̂.
            theta_hat = theta_stage0
            beta_hat = pred.beta(1.0).copy() if p > 0 else np.zeros(0)
            self._optim = dict(self._optim_stage0)
            self._optim_stage0 = None
            self._devfun_stage1 = None

        self.theta = theta_hat
        self._beta = beta_hat
        self._pred = pred
        self._resp = resp
        self.method = "glmer.ML"   # lme4's @resp$family != gaussian path

        # ----- post-fit attributes ----------------------------------------

        # Caches that ``_ranef`` / ``predict`` need. Mirror what the
        # Gaussian path stashes in ``_fit_from_components``.
        template = re.Lambdat
        lt_theta_pos, lt_indices, lt_indptr = _sparse_Lt_spec(template)
        self._template = template
        self._lt_theta_pos = lt_theta_pos
        self._lt_indices = lt_indices
        self._lt_indptr = lt_indptr
        self._lt_shape = template.shape
        self._Z_sp = Z_sp
        self._eye_q_sp = eye_array(q, format="csc")
        self._chol_factor = pred.chol_factor

        # Snapshot Λ and L at θ̂ — same shapes as the Gaussian path so
        # downstream code (profile/ranef/plot_design) works unchanged.
        # ``Lambda`` is dense q×q; ``L`` is the lower CHOLMOD factor.
        self.Lambda = pred.lambdat_sp.T.toarray()
        self.L = pred.chol_factor.L.toarray()
        # ``_u`` = spherical RE at the converged state. ``pred.beta0`` and
        # ``pred.u0`` are still zero (lme4 never installPars during the
        # outer loop), so u(1) = delu.
        self._u = pred.u(1.0).copy()

        # Linear predictor / fitted values. ``eta`` includes the offset
        # (resp.eta is computed from offset + γ in update_mu). ``mu``
        # is on the response scale. lme4 names ``fitted_values`` for the
        # response-scale fit; ``linear_predictors`` for ``eta``.
        self.mu = resp.mu.copy()
        self.eta = resp.eta.copy()
        self.linear_predictors = self.eta
        self.fitted = self.mu
        self.fitted_values = self.mu
        # Raw response-scale residuals = y − μ̂. (Type-specific residuals
        # live on ``residuals_of``; ``residuals`` itself follows lme4's
        # default of *deviance* residuals — that's what ``deviance(m)``
        # decomposes.)
        self.residuals = self._deviance_residuals_signed()
        # ``working_weights`` = lme4's ``glmResp$weights`` = (μ_η²·w)/V(μ)
        # = sqrt_x_wt² (respModule.cpp:179-183).
        self.working_weights = resp.sqrt_x_wt ** 2
        self.prior_weights = resp.weights.copy()

        # ----- scale (σ) / dispersion -------------------------------------
        # For canonical-link scale-known families (Poisson, Binomial),
        # lme4 reports σ = 1 (methods.R:236, sigma.merMod). For
        # scale-unknown (Gamma, Inverse-Gaussian, Gaussian-noncanon),
        # σ is the Pearson estimate: √[Σ w·(y−μ)²/V(μ) / df_resid].
        if getattr(family, "scale_known", False):
            self.sigma = 1.0
            self.sigma_squared = 1.0
            use_sc = 0
        else:
            df_resid = max(n - p, 1)
            pearson = resp.weights * (y - resp.mu) ** 2 / family.variance(resp.mu)
            phi = float(np.sum(pearson) / df_resid)
            self.sigma = float(np.sqrt(phi))
            self.sigma_squared = phi
            use_sc = 1

        # ----- SE(β̂) / vcov_beta ------------------------------------------
        # ``calc_derivs=True`` (lme4 default): use the numerical Hessian of
        # the Stage 1 deviance at the optimum. lme4's vcov.merMod
        # (lmer.R:2211-2219): ``V = solve(H)[β,β] + t(solve(H)[β,β])``
        # which is ``2·solve(H)[β,β]`` when symmetric — the H is over
        # (θ, β) and H is the Hessian of the deviance ``-2·logL`` so
        # ``Var(β) = inv(H/2) = 2·inv(H)``.
        # ``calc_derivs=False`` (or Stage 1 unavailable, e.g. nAGQ=0):
        # fall back to the Schur-complement RX, ``Var(β̂) = σ²·RX⁻ᵀ·RX⁻¹``.
        vcov_beta_hess = None
        # the (θ, β) gradient + Hessian feed both vcov AND the
        # convergence diagnostics (checkConv); capture both. They stay None
        # when calc_derivs is off (lme4's checkConv then bails — no checks).
        deriv_grad = None
        deriv_hess = None
        if (inputs.calc_derivs and p > 0 and do_stage1
                and self._devfun_stage1 is not None):
            try:
                opt_par = np.concatenate([theta_hat, beta_hat])
                deriv_grad, H = _deriv12(self._devfun_stage1, opt_par,
                                         fx=self._optim["fval"])
                deriv_hess = H
                H_inv = np.linalg.solve(H, np.eye(H.shape[0]))
                # β-block of inv(H); lme4 then adds its transpose for
                # symmetry (= 2·H_inv[β,β] when H_inv is symmetric).
                bb = H_inv[n_theta:, n_theta:]
                V = bb + bb.T
                # Sanity check: positive definite.
                eig = np.linalg.eigvalsh(0.5 * (V + V.T))
                if eig.min() > 0:
                    vcov_beta_hess = V
                # After the deriv12 sweep, re-anchor pred/resp at the
                # optimum so downstream code (predict, ranef, …) sees the
                # converged state. Mirrors lme4 lmer.R:2614-2617.
                if not inputs.use_last_params:
                    self._devfun_stage1(opt_par)
            except (np.linalg.LinAlgError, RuntimeError):
                vcov_beta_hess = None
        if p == 0:
            vcov_beta = np.zeros((0, 0))
            se_beta = np.zeros(0)
        elif vcov_beta_hess is not None:
            vcov_beta = vcov_beta_hess
            se_beta = np.sqrt(np.diag(vcov_beta))
        else:
            # RX-based fallback.
            Rx_inv = solve_triangular(pred.RX, np.eye(p), lower=True)
            vcov_beta = self.sigma_squared * np.einsum("ij,ik->jk", Rx_inv, Rx_inv)
            se_beta = np.sqrt(np.diag(vcov_beta))
        self._vcov_beta_arr = vcov_beta
        self.vcov_beta = pl.DataFrame(
            {c: vcov_beta[:, i] for i, c in enumerate(self.column_names)}
        )
        self._se_beta = se_beta
        self.bhat = pl.DataFrame(
            {c: [float(beta_hat[i])] for i, c in enumerate(self.column_names)}
        )
        self.fixef = self.bhat
        self.se_bhat = pl.DataFrame(
            {c: [float(se_beta[i])] for i, c in enumerate(self.column_names)}
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.where(se_beta > 0, beta_hat / np.where(se_beta > 0, se_beta, 1.0), 0.0)
        self.t_values = pl.DataFrame(
            {c: [float(t_vals[i])] for i, c in enumerate(self.column_names)}
        )

        # ----- per-bar variance components --------------------------------
        # Same shape as the Gaussian path; the σ factor here is 1 for
        # scale-known families (so Σ_block ≡ relative covariance), else
        # multiplied by σ for unknown-scale parity with lme4's VarCorr.
        Sigma_blocks = _per_bar_relative_cov(theta_hat, bar_sizes)
        self.sd_re: dict[str, np.ndarray] = {}
        self.corr_re: dict[str, np.ndarray | None] = {}
        for key, Sigma in zip(re.cnms.keys(), Sigma_blocks):
            d = np.sqrt(np.diag(Sigma))
            self.sd_re[key] = self.sigma * d
            if Sigma.shape[0] > 1:
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = Sigma / np.outer(d, d)
                corr = np.where(np.isfinite(corr), corr, 0.0)
                np.fill_diagonal(corr, 1.0)
                self.corr_re[key] = corr
            else:
                self.corr_re[key] = None

        # ----- summary statistics -----------------------------------------
        # npar follows ``npar.merMod`` (lmer.R:1049): length(beta) +
        # length(theta) + useSc. useSc = 0 for scale-known.
        self.npar = p + len(theta_hat) + use_sc
        laplace = float(self._optim["fval"])
        # ``m.deviance`` for GLMM == residual deviance (sum of dev_resids),
        # NOT the Laplace value. lme4's ``deviance(m)`` returns the same
        # for glmer fits (methods.R's deviance.merMod). The Laplace
        # criterion is on ``m.deviance_laplace`` for downstream callers.
        self.deviance_laplace = laplace
        self.deviance = float(resp.deviance())   # = Σ dev_resids
        self.loglike = -0.5 * laplace
        self.df_resid = n - self.npar
        # AIC/BIC use the Laplace deviance (lme4's logLik-based formula).
        self.AIC = laplace + 2.0 * self.npar
        self.BIC = laplace + np.log(n) * self.npar

        # convergence diagnostics. Port of ``checkConv`` (checkConv.R)
        # plus ``m@optinfo`` (utilities.R:448): the singular check
        # (``check.conv.singular``) plus, when the Stage-1 (θ, β) gradient /
        # Hessian are available (calc_derivs on), the gradient and Hessian
        # convergence diagnostics (``check.conv.grad`` /
        # ``check.conv.hess``). lme4 skips the latter for a singular fit.
        self.optinfo = _build_optinfo(
            theta=self.theta,
            theta_bounds=self._theta_bounds,
            optim=self._optim,
            optim_stage0=self._optim_stage0,
            ctrl=inputs.opt_ctrl,
            optimizer=inputs.optimizer,
            grad=deriv_grad,
            hess=deriv_hess,
            n_theta=n_theta,
            grad_cfg=inputs.check_conv_grad,
            hess_cfg=inputs.check_conv_hess,
        )

    def _deviance_residuals_signed(self) -> np.ndarray:
        """Signed √dev_resid_i — what ``residuals(m, type="deviance")`` returns.

        ``glmResp::devResid`` (respModule.cpp:128) returns
        ``family$dev.resids(y, μ, w)``, which for most families is the
        per-observation **squared** deviance contribution. R's
        ``residuals.merMod(type="deviance")`` then takes the signed
        square-root — that's what we report by default. For Gaussian
        LMM (no ``_resp``), the deviance residual collapses to ``y − μ``.
        """
        rp = getattr(self, "_resp", None)
        if rp is None:
            # Gaussian-identity LMM path — devresids are just raw residuals.
            return np.asarray(self.y, dtype=float) - np.asarray(self.fitted, dtype=float)
        return np.sign(rp.y - rp.mu) * np.sqrt(rp.deviance_residuals())

    def residuals_of(self, type: str = "deviance") -> np.ndarray:
        """Residuals on the chosen scale — mirrors ``residuals.merMod``.

        Types:

        - ``"deviance"`` (default): signed √dev_resid_i.
        - ``"pearson"``: ``(y − μ) · √w / √V(μ)``.
        - ``"working"``: ``(y − μ) / μ_η`` (PIRLS working residual).
        - ``"response"``: ``y − μ`` on the response scale.

        Port of ``residuals.glmResp`` (respModule.cpp / methods.R:1310-1349).
        For Gaussian-identity (LMM, or GLMM with the trivial family), all
        four collapse to ``y − μ``.
        """
        rp = getattr(self, "_resp", None)
        if rp is None:
            # Gaussian-identity LMM path: every type collapses to y − μ,
            # except "pearson" which scales by √w when prior weights ≠ 1.
            y = np.asarray(self.y, dtype=float)
            mu = np.asarray(self.fitted, dtype=float)
            if type == "response" or type == "deviance" or type == "working":
                return y - mu
            if type == "pearson":
                w = getattr(self, "prior_weights", None)
                if w is None:
                    return y - mu
                return (y - mu) * np.sqrt(np.asarray(w, dtype=float))
            raise ValueError(
                f"unknown residual type {type!r}; expected one of "
                "'deviance', 'pearson', 'working', 'response'"
            )
        if type == "deviance":
            return self._deviance_residuals_signed()
        if type == "pearson":
            return (rp.y - rp.mu) * np.sqrt(rp.weights / self.family.variance(rp.mu))
        if type == "working":
            return rp.working_residuals()
        if type == "response":
            return rp.y - rp.mu
        raise ValueError(
            f"unknown residual type {type!r}; expected one of "
            "'deviance', 'pearson', 'working', 'response'"
        )

    # ---- deviance building blocks --------------------------------------
    #
    # These are used both by _fit_from_components (for the initial ML/REML
    # fit) and by profile() (for the per-grid-point re-optimization).

    def _build_Lt_sparse(self, theta: np.ndarray) -> csc_array:
        """Build Λᵀ as a CSC sparse matrix from the precomputed structure.

        The sparsity pattern is fixed by the integer template, so we just
        swap the numeric entries on each call. Same pattern every call is
        what lets CHOLMOD reuse the symbolic analysis."""
        data = np.asarray(theta, dtype=float)[self._lt_theta_pos]
        return csc_array(
            (data, self._lt_indices, self._lt_indptr),
            shape=self._lt_shape, copy=False,
        )

    def _chol_block(
        self, theta: np.ndarray, *,
        y: np.ndarray | None = None, X: np.ndarray | None = None,
        XtX: np.ndarray | None = None, Xty: np.ndarray | None = None,
        yty: float | None = None,
    ) -> tuple[float, float, float] | None:
        """Core Cholesky step. Returns ``(rss, log|Lz|, log|Rx|)`` at β̂_θ,
        or ``None`` if the factorization fails.

        With defaults this uses the original ``X``/``y`` cached on the fit.
        Overrides let ``profile()`` plug in modified designs (e.g. ``y``
        adjusted by a fixed β_j, or ``X`` with a column removed).

        ``log|Lz|`` is computed as ½·``factor.logdet()`` since
        ``Lz Lzᵀ = M`` means ``|M| = |Lz|²``. ``y`` here is offset-stripped
        (``y_solve``); cached ``Xty/yty`` are built from ``y_solve`` to match."""
        y = self._y_solve if y is None else y
        X = self.X.to_numpy().astype(float) if X is None else X
        XtX = self._XtX if XtX is None else XtX
        Xty = self._Xty if Xty is None else Xty
        yty = self._yty if yty is None else yty
        Lt = self._build_Lt_sparse(theta)
        ZL = self._Z_sp @ Lt.T
        M = (ZL.T @ ZL + self._eye_q_sp).tocsc()
        try:
            if self._chol_factor is None:
                self._chol_factor = cho_factor(M)
            else:
                self._chol_factor.factorize(M)
        except CholmodError:
            return None
        F = self._chol_factor
        ZLty = np.asarray(ZL.T @ y).ravel()
        M_inv_ZLty = F.solve(ZLty)
        # Apple Accelerate's small-size GEMV/GEMM dispatch is non-deterministic
        # across fresh buffers (~1e-12 noise), which L-BFGS-B's finite-diff
        # gradient amplifies into visibly different θ. einsum sidesteps that
        # BLAS path and stays bit-identical.
        cu_sq = float(np.einsum("i,i->", ZLty, M_inv_ZLty))
        # ½·log|M|: CHOLMOD's LLᵀ ⇒ Σ log diag(L); splu fallback ⇒ ½·Σ log|U.diag|.
        # Sidesteps sksparse's slow F.logdet() Python wrapper (~210 µs, 20× this).
        log_det_Lz = F.half_log_det()
        if X.shape[1] > 0:
            ZLtX = np.asarray(ZL.T @ X)
            M_inv_ZLtX = F.solve(ZLtX)
            XtX_eff = XtX - np.einsum("ij,ik->jk", ZLtX, M_inv_ZLtX)
            try:
                Rx = np.linalg.cholesky(XtX_eff)
            except np.linalg.LinAlgError:
                return None
            rhs = Xty - np.einsum("ij,i->j", ZLtX, M_inv_ZLty)
            cb = solve_triangular(Rx, rhs, lower=True)
            rss = yty - cu_sq - float(np.einsum("i,i->", cb, cb))
            log_det_Rx = float(np.log(np.diag(Rx)).sum())
        else:
            rss = yty - cu_sq
            log_det_Rx = 0.0
        if rss <= 0:
            return None
        return rss, log_det_Lz, log_det_Rx

    def _ml_deviance(
        self, theta: np.ndarray, *,
        sigma_fix: float | None = None,
        y: np.ndarray | None = None, X: np.ndarray | None = None,
        XtX: np.ndarray | None = None, Xty: np.ndarray | None = None,
        yty: float | None = None,
    ) -> float:
        """ML deviance at this θ. Defaults to σ profiled out (σ̂² = rss/n);
        pass ``sigma_fix`` to hold σ at a specific value instead."""
        n = len(self.y) if y is None else len(y)
        r = self._chol_block(
            theta, y=y, X=X, XtX=XtX, Xty=Xty, yty=yty,
        )
        if r is None:
            return 1e15
        rss, log_det_Lz, _ = r
        if sigma_fix is None:
            return 2.0 * log_det_Lz + n * (1.0 + self._log2pi + np.log(rss / n))
        s2 = sigma_fix ** 2
        return 2.0 * log_det_Lz + n * (self._log2pi + np.log(s2)) + rss / s2

    def _reml_deviance(self, theta: np.ndarray) -> float:
        """REML ``-2 log L_REML`` at this θ. β profiles out, then σ."""
        n, p = self.n, self.p
        r = self._chol_block(theta)
        if r is None:
            return 1e15
        rss, log_det_Lz, log_det_Rx = r
        df = n - p
        return (
            2.0 * log_det_Lz + 2.0 * log_det_Rx
            + df * (1.0 + self._log2pi + np.log(rss / df))
        )

    # ---- profile likelihood --------------------------------------------

    def _refit_theta(self, obj_fn, theta_start: np.ndarray) -> tuple[float, np.ndarray]:
        """Re-optimize θ against ``obj_fn(theta) → deviance``."""
        res = minimize(
            obj_fn, theta_start, method="L-BFGS-B", bounds=self._theta_bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        return float(res.fun), res.x

    def _post_refit_state(
        self, theta: np.ndarray, *,
        sigma_fix: float | None = None,
        y: np.ndarray | None = None, X: np.ndarray | None = None,
        XtX: np.ndarray | None = None, Xty: np.ndarray | None = None,
        yty: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """At a fixed θ, recover (σ̂, β̂) at the just-found optimum.

        ``profile()`` calls this after each inner θ-refit so each grid
        point carries the full optimized state — needed for ``plot_pairs``
        traces. Cost is one sparse Cholesky + one tri-solve per call.
        ``sigma_fix=None`` profiles σ out (σ̂² = rss/n); pass it explicitly
        when σ was either pinned or optimized as a free variable upstream.
        """
        y_ = self._y_solve if y is None else y
        X_ = self.X.to_numpy().astype(float) if X is None else X
        XtX_ = self._XtX if XtX is None else XtX
        Xty_ = self._Xty if Xty is None else Xty
        yty_ = self._yty if yty is None else yty
        n = len(y_)
        Lt = self._build_Lt_sparse(theta)
        ZL = self._Z_sp @ Lt.T
        M = (ZL.T @ ZL + self._eye_q_sp).tocsc()
        self._chol_factor.factorize(M)
        F = self._chol_factor
        ZLty = np.asarray(ZL.T @ y_).ravel()
        M_inv_ZLty = F.solve(ZLty)
        cu_sq = float(np.einsum("i,i->", ZLty, M_inv_ZLty))
        if X_.shape[1] == 0:
            rss = yty_ - cu_sq
            beta = np.zeros(0)
        else:
            ZLtX = np.asarray(ZL.T @ X_)
            M_inv_ZLtX = F.solve(ZLtX)
            XtX_eff = XtX_ - np.einsum("ij,ik->jk", ZLtX, M_inv_ZLtX)
            Rx = np.linalg.cholesky(XtX_eff)
            rhs = Xty_ - np.einsum("ij,i->j", ZLtX, M_inv_ZLty)
            cb = solve_triangular(Rx, rhs, lower=True)
            beta = solve_triangular(Rx.T, cb, lower=False)
            rss = yty_ - cu_sq - float(np.einsum("i,i->", cb, cb))
        sigma = sigma_fix if sigma_fix is not None else float(np.sqrt(max(rss, 0.0) / n))
        return sigma, beta

    def _dev_with_beta_fixed(
        self, j: int, beta_j_tgt: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Min ML deviance with β_j = ``beta_j_tgt``. Trick: subtract
        ``x_j · β_j_tgt`` from y and drop column j — the remaining fit has
        the same functional form. Returns ``(dev, θ̂, σ̂, β̂)`` where β̂ is
        in the full original column order with ``β_j = beta_j_tgt``."""
        X_full = self.X.to_numpy().astype(float)
        x_j = X_full[:, j]
        X_rest = np.delete(X_full, j, axis=1)
        # ``self._y_solve`` already has the offset removed; subtracting
        # x_j·β_j_tgt on top of that gives the correct adjusted response
        # for the offset-stripped sub-fit.
        y_adj = self._y_solve - x_j * beta_j_tgt
        XtX_rest = X_rest.T @ X_rest
        Xty_rest = X_rest.T @ y_adj
        yty_adj = float(y_adj @ y_adj)
        dev, theta_opt = self._refit_theta(
            lambda th: self._ml_deviance(
                th, y=y_adj, X=X_rest,
                XtX=XtX_rest, Xty=Xty_rest, yty=yty_adj,
            ),
            theta_start,
        )
        sigma_opt, beta_rest = self._post_refit_state(
            theta_opt, y=y_adj, X=X_rest,
            XtX=XtX_rest, Xty=Xty_rest, yty=yty_adj,
        )
        beta_opt = np.empty(self.p)
        beta_opt[j] = beta_j_tgt
        rest_idx = [k for k in range(self.p) if k != j]
        beta_opt[rest_idx] = beta_rest
        return dev, theta_opt, sigma_opt, beta_opt

    def _dev_with_sigma_fixed(
        self, sigma_tgt: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Min ML deviance with σ = ``sigma_tgt`` (β profiles out).
        Returns ``(dev, θ̂, σ_tgt, β̂)``."""
        dev, theta_opt = self._refit_theta(
            lambda th: self._ml_deviance(th, sigma_fix=sigma_tgt),
            theta_start,
        )
        _, beta_opt = self._post_refit_state(theta_opt, sigma_fix=sigma_tgt)
        return dev, theta_opt, float(sigma_tgt), beta_opt

    def _dev_with_sd_fixed(
        self, slot_i: int, sd_tgt: float,
        sigma_start: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Min ML deviance with σ_i = σ · θ[slot_i] pinned at ``sd_tgt``.

        Scalar-bar case: the bar has one θ entry, so pinning ``σ · θ[slot_i]
        = sd_tgt`` is a single nonlinear constraint. We re-parameterize as
        ``(σ, θ_rest)`` with ``θ[slot_i] = sd_tgt / σ`` and minimize jointly.
        Returns ``(dev, θ̂, σ̂, β̂)``."""
        other = np.delete(np.arange(len(self._theta_bounds)), slot_i)
        theta_rest0 = np.asarray(theta_start, dtype=float)[other]

        # Guard θ[slot_i] = sd_tgt/σ from blowing up when L-BFGS-B probes
        # very small σ — without this the implied θ becomes O(1e7) and
        # ``M = ΛᵀZᵀZΛ + I`` factorizes with rcond ≈ 1e-15. Cholmod warns
        # and the gradient gets noisy. Cap θ at 1e4 → cond(M) ≲ 1e8, well
        # away from Cholmod's near-singular threshold; the optimum lives
        # at θ_slot ≈ θ_hat ≪ 1e4 anyway, so the cap never binds.
        sigma_lb = max(1e-8, sd_tgt / 1e4)

        def obj(x):
            sigma = x[0]
            if sigma <= 0:
                return 1e15
            theta = np.zeros(len(self._theta_bounds))
            theta[slot_i] = sd_tgt / sigma
            for k, slot in enumerate(other):
                theta[slot] = x[1 + k]
            return self._ml_deviance(theta, sigma_fix=sigma)

        x0 = np.concatenate([[max(sigma_start, sigma_lb)], theta_rest0])
        bounds = [(sigma_lb, None)] + [self._theta_bounds[k] for k in other]
        res = minimize(
            obj, x0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        # Reconstruct θ at the optimum for warm-start of neighboring points.
        theta_opt = np.zeros(len(self._theta_bounds))
        sigma_opt = float(res.x[0])
        theta_opt[slot_i] = sd_tgt / sigma_opt
        for k, slot in enumerate(other):
            theta_opt[slot] = res.x[1 + k]
        _, beta_opt = self._post_refit_state(theta_opt, sigma_fix=sigma_opt)
        return float(res.fun), theta_opt, sigma_opt, beta_opt

    def _fillmat_walk(
        self, *, direction: int,
        prev_v: float, prev_zeta: float,
        curr_v: float, curr_zeta: float,
        fit_at_v, theta_warm: np.ndarray, sigma_warm: float,
        d_hat: float, delta: float, cutoff: float,
        v_min: float = -np.inf, v_max: float = np.inf,
        max_steps: int = 100, maxmult: float = 10.0, minstep: float = 1e-6,
    ) -> list[tuple]:
        """One-direction profile walk — port of lme4's ``fillmat`` inner
        loop in ``profile.merMod``.

        Extrapolates the next v from the local slope ``Δv/Δζ`` between
        (prev_v, prev_zeta) and (curr_v, curr_zeta) to target |Δζ| ≈
        ``delta`` per step; caps the step at ``maxmult × |Δv|`` to bound
        runaway when ζ flattens. ``direction`` only sets ζ's sign (matching
        R's ``sign(xx - pw)``). Stops when |ζ| ≥ ``cutoff``, v hits a
        bound, or ``max_steps`` is exhausted. Does NOT include
        (curr_v, curr_zeta) in the returned list.
        """
        out: list[tuple] = []
        for _ in range(max_steps):
            if abs(curr_zeta) >= cutoff:
                break
            if curr_v <= v_min or curr_v >= v_max:
                break

            num = curr_v - prev_v
            denom = curr_zeta - prev_zeta
            if denom == 0.0 or not np.isfinite(denom):
                step = minstep
            else:
                step = delta * num / denom
                if step < 0:
                    # Non-monotonic profile — fall back to a tiny step
                    # rather than walking backwards (matches R's
                    # ``warning("unexpected decrease in profile")`` path).
                    step = minstep
                else:
                    maxstep = maxmult * abs(num)
                    if abs(step) > maxstep:
                        step = float(np.sign(step) * maxstep)

            v_new = curr_v + float(np.sign(num)) * step
            boundary_hit = False
            if v_new <= v_min:
                v_new = v_min + 1e-6 * max(abs(step), 1e-12)
                boundary_hit = True
            elif v_new >= v_max:
                v_new = v_max - 1e-6 * max(abs(step), 1e-12)
                boundary_hit = True

            d_new, theta_new, sigma_new, beta_new = fit_at_v(
                v_new, theta_warm, sigma_warm,
            )
            if not np.isfinite(d_new):
                break
            zeta_new = direction * float(np.sqrt(max(0.0, d_new - d_hat)))
            out.append(
                (float(v_new), float(zeta_new), theta_new, sigma_new, beta_new)
            )

            if boundary_hit:
                break

            prev_v, prev_zeta = curr_v, curr_zeta
            curr_v, curr_zeta = v_new, zeta_new
            theta_warm, sigma_warm = theta_new, sigma_new
        return out

    def _profile_param_adaptive(
        self, *, fit_at_v, v_start: float,
        theta_start: np.ndarray, sigma_start: float, beta_start: np.ndarray,
        d_hat: float, is_var_component: bool,
        cutoff: float, delta: float,
        se_for_init: float = 0.0,
        v_min: float = -np.inf, v_max: float = np.inf,
        max_steps_per_dir: int = 100,
    ) -> list[tuple]:
        """Profile one parameter — port of lme4's per-parameter loop in
        ``profile.merMod``.

        Computes one initial "shift" sample (``MLE × 1.01`` for variance
        components, ``MLE + delta·SE`` for fixed effects — matching R's
        ``shiftpar = pw * 1.01`` and ``fe.zeta(est + delta * std)``), then
        walks adaptively in both ζ-directions using :meth:`_fillmat_walk`.
        Output: deepest-negative-ζ first → MLE → shift → deepest-positive-ζ.
        """
        if is_var_component:
            shift_v = 0.001 if v_start == 0.0 else v_start * 1.01
        else:
            shift_v = v_start + delta * se_for_init

        d_shift, theta_shift, sigma_shift, beta_shift = fit_at_v(
            shift_v, theta_start.copy(), sigma_start,
        )
        # shift_v > v_start by construction (multiplicative bump for
        # variance components, additive positive bump for fixed effects),
        # so the shift point lives in the +ζ half — matching R's
        # ``sign(xx - pw)``.
        zeta_shift = float(np.sqrt(max(0.0, d_shift - d_hat)))

        mle_tup = (
            float(v_start), 0.0, theta_start.copy(),
            float(sigma_start), beta_start.copy(),
        )
        shift_tup = (
            float(shift_v), zeta_shift, theta_shift, sigma_shift, beta_shift,
        )

        pos = self._fillmat_walk(
            direction=+1,
            prev_v=float(v_start), prev_zeta=0.0,
            curr_v=float(shift_v), curr_zeta=zeta_shift,
            fit_at_v=fit_at_v,
            theta_warm=theta_shift, sigma_warm=sigma_shift,
            d_hat=d_hat, delta=delta, cutoff=cutoff,
            v_min=v_min, v_max=v_max, max_steps=max_steps_per_dir,
        )
        neg = self._fillmat_walk(
            direction=-1,
            prev_v=float(shift_v), prev_zeta=zeta_shift,
            curr_v=float(v_start), curr_zeta=0.0,
            fit_at_v=fit_at_v,
            theta_warm=theta_start.copy(), sigma_warm=sigma_start,
            d_hat=d_hat, delta=delta, cutoff=cutoff,
            v_min=v_min, v_max=v_max, max_steps=max_steps_per_dir,
        )
        return list(reversed(neg)) + [mle_tup, shift_tup] + pos

    def _glmm_dev_with_theta_fixed(self, slot_i, v_tgt, theta_warm, beta_warm):
        """Profile a GLMM variance component: pin ``θ[slot_i] = v_tgt`` and
        re-optimise the remaining ``(θ_{-i}, β)`` over the Stage-1 Laplace
        devfun, returning ``(dev, θ̂, σ=1, β̂)``. For a scale-known GLMM σ≡1, so
        ``θ_i`` *is* the random-effect SD (the ``.sig0i`` profile axis)."""
        devfun = self._devfun_stage1
        n_theta = len(self.theta)
        free_t = [k for k in range(n_theta) if k != slot_i]
        nf = len(free_t)

        def obj(x):
            theta = np.empty(n_theta)
            theta[slot_i] = v_tgt
            for a, k in enumerate(free_t):
                theta[k] = x[a]
            return devfun(np.concatenate([theta, x[nf:]]))

        x0 = np.concatenate([np.asarray(theta_warm, float)[free_t],
                             np.asarray(beta_warm, float)])
        bounds = [self._theta_bounds[k] for k in free_t] + [(None, None)] * self.p
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 500})
        theta_opt = np.empty(n_theta)
        theta_opt[slot_i] = v_tgt
        for a, k in enumerate(free_t):
            theta_opt[k] = res.x[a]
        return float(res.fun), theta_opt, 1.0, res.x[nf:].copy()

    def _glmm_dev_with_beta_fixed(self, j, v_tgt, theta_warm, beta_warm):
        """Profile a GLMM fixed effect: pin ``β[j] = v_tgt`` and re-optimise
        ``(θ, β_{-j})`` over the Stage-1 Laplace devfun. Returns ``(dev, θ̂, 1, β̂)``."""
        devfun = self._devfun_stage1
        n_theta = len(self.theta)
        p = self.p
        free_b = [k for k in range(p) if k != j]

        def obj(x):
            beta = np.empty(p)
            beta[j] = v_tgt
            for a, k in enumerate(free_b):
                beta[k] = x[n_theta + a]
            return devfun(np.concatenate([x[:n_theta], beta]))

        x0 = np.concatenate([np.asarray(theta_warm, float),
                             np.asarray(beta_warm, float)[free_b]])
        bounds = list(self._theta_bounds) + [(None, None)] * len(free_b)
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 500})
        theta_opt = res.x[:n_theta].copy()
        beta_opt = np.empty(p)
        beta_opt[j] = v_tgt
        for a, k in enumerate(free_b):
            beta_opt[k] = res.x[n_theta + a]
        return float(res.fun), theta_opt, 1.0, beta_opt

    def _profile_glmm(self, n_grid: int, alphamax: float) -> "Profile":
        """Profile a **scale-known** GLMM (Poisson/Binomial) — the
        constrained-Laplace path. Mirrors the Gaussian :meth:`profile` but over
        the Stage-1 ``[θ, β]`` Laplace devfun (no residual σ axis, ``useSc=0``),
        re-optimising the free coordinates at each grid point with one pinned."""
        from scipy.stats import chi2

        if self._devfun_stage1 is None:
            raise NotImplementedError(
                "profile() needs the Stage-1 [θ,β] Laplace devfun, which "
                "nAGQ=0 fits don't build; refit with nAGQ>=1")
        d_hat = float(self._optim["fval"])
        theta_hat = self.theta.copy()
        n_theta = len(theta_hat)
        # useSc=0 for scale-known GLMM ⇒ nptot = #θ + p (no residual σ).
        nptot = n_theta + self.p
        cutoff = float(np.sqrt(chi2.ppf(1.0 - alphamax, nptot)))
        delta = cutoff / 8.0

        bar_keys = list(self.sd_re.keys())
        bar_labels = [f".sig{i + 1:02d}" for i in range(len(bar_keys))]
        slot_offsets = list(np.cumsum([0] + self._bar_sizes[:-1]))
        bar_slots = [int(s) for s in slot_offsets]
        param_names: list[str] = bar_labels + list(self.column_names)

        estimate: dict[str, float] = {}
        for lbl, key in zip(bar_labels, bar_keys):
            estimate[lbl] = float(self.sd_re[key][0])
        for j, name in enumerate(self.column_names):
            estimate[name] = float(self._beta[j])

        def _state_to_row(theta_opt, sigma_opt, beta_opt) -> dict[str, float]:
            row: dict[str, float] = {}
            for lbl, slot in zip(bar_labels, bar_slots):
                row[lbl] = float(theta_opt[slot])  # σ≡1 ⇒ SD = θ
            for j, name in enumerate(self.column_names):
                row[name] = float(beta_opt[j])
            return row

        rows_by_param: dict[str, list[dict[str, float]]] = {p: [] for p in param_names}
        zetas_by_param: dict[str, np.ndarray] = {}

        def _store(samples, lbl):
            zetas_by_param[lbl] = np.array([s[1] for s in samples])
            for s in samples:
                rows_by_param[lbl].append(_state_to_row(s[2], s[3], s[4]))

        for lbl, slot_i in zip(bar_labels, bar_slots):
            samples = self._profile_param_adaptive(
                fit_at_v=lambda v, th_w, sg_w, _slot=slot_i:
                    self._glmm_dev_with_theta_fixed(_slot, v, th_w, self._beta),
                v_start=estimate[lbl], theta_start=theta_hat,
                sigma_start=1.0, beta_start=self._beta,
                d_hat=d_hat, is_var_component=True,
                cutoff=cutoff, delta=delta, v_min=0.0, max_steps_per_dir=n_grid,
            )
            _store(samples, lbl)

        for j, name in enumerate(self.column_names):
            se_j = float(self._se_beta[j])
            samples = self._profile_param_adaptive(
                fit_at_v=lambda v, th_w, sg_w, _j=j:
                    self._glmm_dev_with_beta_fixed(_j, v, th_w, self._beta),
                v_start=estimate[name], theta_start=theta_hat,
                sigma_start=1.0, beta_start=self._beta,
                d_hat=d_hat, is_var_component=False,
                se_for_init=max(se_j, 1e-3),
                cutoff=cutoff, delta=delta, max_steps_per_dir=n_grid,
            )
            _store(samples, name)

        # Profiling mutated pred/resp (each devfun call re-runs PIRLS); restore
        # them to the MLE state so predict()/residuals after profile() are right.
        self._devfun_stage1(self._optim["par"])

        data: dict[str, pl.DataFrame] = {}
        for p in param_names:
            cols = {q: [r[q] for r in rows_by_param[p]] for q in param_names}
            cols["zeta"] = list(zetas_by_param[p])
            data[p] = pl.DataFrame(cols)
        return Profile(data, estimate)

    def profile(self, n_grid: int = 100, alphamax: float = 0.01) -> "Profile":
        """Compute profile-likelihood curves for σ_i, σ, and each β_j.

        Port of lme4's ``profile.merMod``: walks ζ adaptively from the
        MLE using a linear ``Δv/Δζ`` slope estimate from the last two
        points, targeting |Δζ| ≈ ``cutoff/8`` per step. The cutoff is
        ``sqrt(qchisq(1 - alphamax, nptot))`` where ``nptot`` is the
        total number of profiled parameters (variance components + σ +
        fixed effects). Walking stops when |ζ| ≥ cutoff or v hits a
        bound. ``n_grid`` is the maximum steps per direction (R's
        ``maxpts``); in practice most parameters terminate after 8–16
        steps.

        For REML fits we first re-fit by ML, per lme4's convention (the LRT
        statistic requires ML). Only scalar bars ``(1|g)`` are supported in
        this first port.
        """
        from scipy.stats import chi2

        if any(c > 1 for c in self._bar_sizes):
            raise NotImplementedError(
                "profile() currently requires scalar bars (1|g); "
                "vector bars like (1+x|g) need a different parameterization."
            )
        if self._is_glmm():
            # unknown-scale GLMM (Gamma/IG): lme4 itself refuses to
            # profile these (profile.R:74-75); match the message verbatim so
            # confint(method="profile") raises identically.
            if not bool(getattr(self.family, "scale_known", False)):
                raise NotImplementedError(
                    "can't (yet) profile GLMMs with non-fixed scale parameters")
            # scale-known GLMM (Poisson/Binomial): profile the [θ, β]
            # Laplace devfun, re-optimising the free coords with one pinned.
            return self._profile_glmm(n_grid, alphamax)
        if self.REML:
            return gmm(self.formula, self.data, REML=False).profile(
                n_grid=n_grid, alphamax=alphamax,
            )

        d_hat = self.deviance
        theta_hat = self.theta.copy()
        sigma_hat = self.sigma

        # R's lme4: ``cutoff = sqrt(qchisq(1 - alphamax, nptot))`` and
        # ``delta = cutoff * delta.cutoff`` (default ``delta.cutoff = 1/8``).
        # ``nptot`` = #θ + 1 (residual σ, since useSc=True for LMM) + p betas.
        nptot = len(theta_hat) + 1 + self.p
        cutoff = float(np.sqrt(chi2.ppf(1.0 - alphamax, nptot)))
        delta = cutoff / 8.0

        bar_keys = list(self.sd_re.keys())
        bar_labels = [f".sig{i + 1:02d}" for i in range(len(bar_keys))]
        slot_offsets = list(np.cumsum([0] + self._bar_sizes[:-1]))
        bar_slots = [int(s) for s in slot_offsets]
        # Column order, also used as the iteration order for profiled params.
        param_names: list[str] = bar_labels + [".sigma"] + list(self.column_names)

        estimate: dict[str, float] = {}
        for lbl, key in zip(bar_labels, bar_keys):
            estimate[lbl] = float(self.sd_re[key][0])
        estimate[".sigma"] = sigma_hat
        for j, name in enumerate(self.column_names):
            estimate[name] = float(self._beta[j])

        def _state_to_row(theta_opt, sigma_opt, beta_opt) -> dict[str, float]:
            """Map (θ̂, σ̂, β̂) at a grid point into the per-parameter row."""
            row: dict[str, float] = {}
            for lbl, slot in zip(bar_labels, bar_slots):
                row[lbl] = float(sigma_opt * theta_opt[slot])
            row[".sigma"] = float(sigma_opt)
            for j, name in enumerate(self.column_names):
                row[name] = float(beta_opt[j])
            return row

        # Adaptive ζ-stepping per parameter — see _step_adaptive. Each
        # call returns rows ordered most-negative-ζ → MLE → most-positive-ζ.
        rows_by_param: dict[str, list[dict[str, float]]] = {p: [] for p in param_names}
        zetas_by_param: dict[str, np.ndarray] = {}

        def _samples_to_storage(samples: list[tuple], lbl: str):
            zetas_by_param[lbl] = np.array([s[1] for s in samples])
            for s in samples:
                rows_by_param[lbl].append(_state_to_row(s[2], s[3], s[4]))

        # -- σ_i (one per scalar bar) ---------------------------------------
        for lbl, slot_i in zip(bar_labels, bar_slots):
            sd_i = estimate[lbl]
            samples = self._profile_param_adaptive(
                fit_at_v=lambda v, th_w, sg_w, _slot=slot_i:
                    self._dev_with_sd_fixed(_slot, v, sg_w, th_w),
                v_start=sd_i, theta_start=theta_hat,
                sigma_start=sigma_hat, beta_start=self._beta,
                d_hat=d_hat, is_var_component=True,
                cutoff=cutoff, delta=delta,
                v_min=0.0, max_steps_per_dir=n_grid,
            )
            _samples_to_storage(samples, lbl)

        # -- σ ----------------------------------------------------------------
        samples = self._profile_param_adaptive(
            fit_at_v=lambda v, th_w, sg_w:
                self._dev_with_sigma_fixed(v, th_w),
            v_start=sigma_hat, theta_start=theta_hat,
            sigma_start=sigma_hat, beta_start=self._beta,
            d_hat=d_hat, is_var_component=True,
            cutoff=cutoff, delta=delta,
            v_min=0.0, max_steps_per_dir=n_grid,
        )
        _samples_to_storage(samples, ".sigma")

        # -- β_j --------------------------------------------------------------
        for j, name in enumerate(self.column_names):
            beta_j = estimate[name]
            se_j = float(self._se_beta[j])
            samples = self._profile_param_adaptive(
                fit_at_v=lambda v, th_w, sg_w, _j=j:
                    self._dev_with_beta_fixed(_j, v, th_w),
                v_start=beta_j, theta_start=theta_hat,
                sigma_start=sigma_hat, beta_start=self._beta,
                d_hat=d_hat, is_var_component=False,
                se_for_init=max(se_j, 1e-3),
                cutoff=cutoff, delta=delta,
                max_steps_per_dir=n_grid,
            )
            _samples_to_storage(samples, name)

        data: dict[str, pl.DataFrame] = {}
        for p in param_names:
            cols: dict[str, list[float]] = {q: [r[q] for r in rows_by_param[p]] for q in param_names}
            cols["zeta"] = list(zetas_by_param[p])
            data[p] = pl.DataFrame(cols)

        return Profile(data, estimate)

    def _ci_param_layout(self):
        """Shared parameter-row layout for ``confint`` — the variance-component
        SD names (``.sig01``, …), then ``.sigma`` if the family carries a scale
        (``useSc``: LMM, or scale-unknown GLMM), then the fixed-effect names.
        Mirrors lme4's ``profnames(object) ++ names(fixef)``. Returns
        ``(bar_keys, vc_names, use_sc, fixef_names, all_names)``."""
        bar_keys = list(self.sd_re.keys())
        vc_names = [f".sig{i + 1:02d}" for i in range(len(bar_keys))]
        use_sc = not bool(getattr(self.family, "scale_known", False))
        fixef_names = list(self.column_names)
        all_names = vc_names + ([".sigma"] if use_sc else []) + fixef_names
        return bar_keys, vc_names, use_sc, fixef_names, all_names

    @staticmethod
    def _filter_parm(df: pl.DataFrame, parm, all_names) -> pl.DataFrame:
        """Subset a CI table by ``parm`` (names or 0-based indices into
        ``all_names``); ``None`` keeps every row, preserving order."""
        if parm is None:
            return df
        if isinstance(parm, (str, int)):
            parm = [parm]
        keep = [all_names[p] if isinstance(p, (int, np.integer)) else p
                for p in parm]
        # Preserve the requested order rather than the table's.
        order = {name: i for i, name in enumerate(keep)}
        return (df.filter(pl.col("parameter").is_in(keep))
                  .sort(pl.col("parameter").replace_strict(order, default=10**9)))

    def confint(self, parm=None, level: float = 0.95, method: str = "profile",
                *, nsim: int = 500, boot_type: str = "perc", FUN=None,
                seed=None, boot_scale: str = "sdcor",
                use_u: bool = False) -> pl.DataFrame:
        """R: ``confint.merMod`` — fixed-effect & variance-component CIs.

        ``method`` (lme4's three, profile.R:807):

        * ``"profile"`` (default) — invert the profile-ζ curve at
          ``±Φ⁻¹((1+level)/2)`` (:meth:`profile` → :meth:`Profile.confint`).
          Raises lme4's exact message for unknown-scale GLMMs (Gamma/IG).
        * ``"Wald"`` — ``β̂ ± z·SE(β̂)`` from ``vcov``; variance components
          are ``NaN`` (lme4 doesn't Wald-CI those).
        * ``"boot"`` — parametric bootstrap (:meth:`bootMer`) summarised by
          ``boot_type`` ∈ ``{"perc","basic","norm"}`` (``nsim`` reps, ``seed``).
          The default statistic is the variance-component SDs + fixed effects
          on the ``sdcor`` scale (lme4's ``confint`` bootFun); pass ``FUN`` to
          override.

        ``parm`` restricts to a subset (names or 0-based indices). Returns a
        polars frame: a ``parameter`` column + two ``%``-labelled bound columns.
        """
        method = str(method).lower()
        if method == "wald":
            return self._confint_wald(parm, level)
        if method == "profile":
            df = self.profile().confint(level=level)
            _, _, _, _, all_names = self._ci_param_layout()
            return self._filter_parm(df, parm, all_names)
        if method == "boot":
            return self._confint_boot(parm, level, nsim, boot_type, FUN,
                                      seed, boot_scale, use_u)
        raise ValueError(
            f"confint: method must be 'profile'/'Wald'/'boot'; got {method!r}")

    def _confint_wald(self, parm, level: float) -> pl.DataFrame:
        """``method="Wald"`` — ``β̂ ± z·SE`` for fixed effects; ``NaN`` rows for
        the variance components / σ (confint.merMod:843-857)."""
        from scipy.stats import norm

        _, vc_names, use_sc, fixef_names, all_names = self._ci_param_layout()
        z = float(norm.ppf((1 + level) / 2))
        a = (1 - level) / 2
        lo_lbl, hi_lbl = f"{100 * a:.1f}%", f"{100 * (1 - a):.1f}%"
        names, los, his = [], [], []
        for nm in vc_names + ([".sigma"] if use_sc else []):
            names.append(nm)
            los.append(float("nan"))
            his.append(float("nan"))
        for j, nm in enumerate(fixef_names):
            b, se = float(self._beta[j]), float(self._se_beta[j])
            names.append(nm)
            los.append(b - z * se)
            his.append(b + z * se)
        df = pl.DataFrame({"parameter": names, lo_lbl: los, hi_lbl: his})
        return self._filter_parm(df, parm, all_names)

    def _boot_profile_stat(self, boot_scale: str = "sdcor") -> dict:
        """Default ``confint(method="boot")`` statistic — the variance-component
        SDs (``.sig0i`` [+ ``.sigma``]) and fixed effects on the ``sdcor`` scale
        (confint.merMod:860-866's bootFun). Scalar bars only (like profile)."""
        if boot_scale != "sdcor":
            raise NotImplementedError(
                "confint(boot_scale='vcov') not implemented; use 'sdcor'")
        if any(c > 1 for c in self._bar_sizes):
            raise NotImplementedError(
                "bootstrap confint requires scalar bars (1|g); pass a custom "
                "FUN for vector bars")
        bar_keys, vc_names, use_sc, fixef_names, _ = self._ci_param_layout()
        out: dict[str, float] = {}
        for nm, key in zip(vc_names, bar_keys):
            out[nm] = float(self.sd_re[key][0])
        if use_sc:
            out[".sigma"] = float(self.sigma)
        for j, nm in enumerate(fixef_names):
            out[nm] = float(self._beta[j])
        return out

    def _confint_boot(self, parm, level, nsim, boot_type, FUN, seed,
                      boot_scale, use_u) -> pl.DataFrame:
        """``method="boot"`` — bootstrap CIs via :meth:`bootMer` (confint.merMod:859)."""
        if FUN is None:
            def FUN(x):
                return x._boot_profile_stat(boot_scale)
        bb = self.bootMer(FUN, nsim=nsim, seed=seed, use_u=use_u)
        if np.all(np.isnan(bb.t)):
            raise RuntimeError("*all* bootstrap runs failed!")
        df = bb.confint(level=level, type=boot_type)
        _, _, _, _, all_names = self._ci_param_layout()
        return self._filter_parm(df, parm, all_names)

    # ---- predict --------------------------------------------------------

    def _build_X_for_newdata(self, newdata: pl.DataFrame) -> np.ndarray:
        """Materialize the fixed-effect design matrix on ``newdata`` using
        the cached expanded formula. Errors if the resulting column names
        don't match the fit's — that catches the common pitfall of a
        factor column with new or missing levels in ``newdata``."""
        X_new_df = materialize(self._expanded, newdata)
        if list(X_new_df.columns) != self.column_names:
            raise ValueError(
                f"predict: newdata's design matrix columns "
                f"{list(X_new_df.columns)!r} don't match the fit's "
                f"{self.column_names!r}. This usually means a factor column "
                f"in newdata has different levels than the fit's data."
            )
        return X_new_df.to_numpy().astype(float)

    def _build_offset_for_newdata(self, newdata: pl.DataFrame) -> np.ndarray:
        """Evaluate any ``offset(...)`` terms on newdata."""
        off = np.zeros(newdata.height)
        for off_node in self._expanded.offsets:
            off = off + _eval_atom(off_node, newdata).values.flatten().astype(float)
        return off

    def _build_Z_for_newdata(
        self, newdata: pl.DataFrame, *, allow_new_levels: bool = False,
    ) -> np.ndarray:
        """Build a dense Z matrix on ``newdata`` aligned to the fit's RE
        column layout. Group values in newdata are mapped to the fit's
        level indices; unseen levels either zero that row's Z entries
        (``allow_new_levels=True``) or raise (``False``, default — matches
        ``lme4::predict.merMod``).
        """
        n = newdata.height
        q = self.q
        Z_new = np.zeros((n, q))
        fit_levels_by_label: dict = self._re.flist_levels

        # Walk fit's bars on newdata using the same simple-bar generation
        # as materialize_bars, but mapping group codes through fit's
        # level lists so Z_new's columns line up with the fit's Z. Sort
        # by fit-#levels descending (stable) to match materialize_bars.
        simple: list[tuple] = []
        for bar in self._expanded.bars:
            if not (isinstance(bar, BinOp) and bar.op in ("|", "||")):
                continue
            lhs_node = bar.left
            group_nodes = _flatten_nested_group(bar.right)
            is_double = bar.op == "||"
            lhs_ef = _bar_lhs_to_ef(lhs_node)
            if is_double:
                lhs_parts: list[ExpandedFormula] = []
                if lhs_ef.intercept:
                    lhs_parts.append(ExpandedFormula(
                        intercept=True, terms=[], bars=[], offsets=[],
                    ))
                for t in lhs_ef.terms:
                    lhs_parts.append(ExpandedFormula(
                        intercept=False, terms=[t], bars=[], offsets=[],
                    ))
            else:
                lhs_parts = [lhs_ef]
            for g_node in group_nodes:
                new_codes, new_levels, g_label = _eval_group(g_node, newdata)
                fit_levels = fit_levels_by_label.get(g_label)
                if fit_levels is None:
                    raise ValueError(
                        f"predict: grouping factor {g_label!r} from newdata "
                        f"is not in the fit (fit groups: "
                        f"{list(fit_levels_by_label)!r})"
                    )
                lvl_to_fit_idx = {lvl: i for i, lvl in enumerate(fit_levels)}
                mapped = np.full(len(new_codes), -1, dtype=int)
                for i, c in enumerate(new_codes):
                    if c < 0:
                        continue
                    fit_idx = lvl_to_fit_idx.get(new_levels[c], -1)
                    if fit_idx < 0 and not allow_new_levels:
                        raise ValueError(
                            f"predict: new level {new_levels[c]!r} in "
                            f"grouping factor {g_label!r}; pass "
                            f"allow_new_levels=True to treat as population mean."
                        )
                    mapped[i] = fit_idx
                for lef in lhs_parts:
                    Z_lhs, cnames = _materialize_re_lhs(lef, newdata)
                    if Z_lhs.shape[1] == 0:
                        continue
                    simple.append((g_label, fit_levels, mapped, Z_lhs, cnames))

        # Stable sort by fit-#levels descending (matches materialize_bars).
        simple.sort(key=lambda b: -len(b[1]))

        col_offset = 0
        for g_label, fit_levels, mapped, Z_lhs, cnames in simple:
            k = len(fit_levels)
            c = Z_lhs.shape[1]
            valid = mapped >= 0
            lvl = mapped[valid]
            rows = np.where(valid)[0]
            for comp in range(c):
                Z_new[rows, col_offset + lvl * c + comp] = Z_lhs[rows, comp]
            col_offset += k * c

        if col_offset != q:
            raise ValueError(
                f"predict: rebuilt Z has {col_offset} columns, expected "
                f"{q}. Bar structure on newdata doesn't match the fit."
            )
        return Z_new

    def predict(
        self,
        newdata: pl.DataFrame | None = None,
        *,
        re_form=None,
        random_only: bool = False,
        type: str = "response",
        allow_new_levels: bool = False,
        na_action: str = "na.pass",
        se_fit: bool = False,
        terms=None,
    ):
        """R: ``predict.merMod`` — predict at the original or new data.

        Parameters
        ----------
        newdata
            New data frame to predict at. If ``None``, returns predictions
            at the original fit data (i.e. fitted values).
        re_form
            ``None`` (default) — include all random effects (``Xβ + Zb``).
            ``False`` — population-level only (``Xβ``). A formula restricting
            to a subset of bars is not yet implemented.
        random_only
            If ``True``, return only the random-effect contribution (``Zb``).
        type
            ``"response"`` or ``"link"``. Identical for LMMs (identity link);
            kept for R-API compatibility.
        allow_new_levels
            If ``True``, group levels in ``newdata`` that weren't in the
            fit contribute 0 to ``Zb`` (population mean). If ``False``
            (R's default), unseen levels raise.
        na_action
            Only ``"na.pass"`` is supported in this port.
        se_fit
            If ``True``, the returned frame gains an ``se.fit`` column.
            SE uses the joint posterior covariance of ``(û, β̂)``, which
            for LMMs is ``σ̂² · M⁻¹`` where ``M`` is the Henderson MME in
            spherical coordinates.
        terms
            Not implemented (R also marks this as unimplemented).

        Returns
        -------
        pl.DataFrame
            ``{fit}``, plus ``se.fit`` when ``se_fit=True``.
        """
        if terms is not None:
            raise NotImplementedError("predict: terms= is not implemented")
        if type not in ("response", "link"):
            raise ValueError(f"predict: type must be 'response' or 'link', got {type!r}")
        if na_action != "na.pass":
            raise NotImplementedError(
                f"predict: only na.action='na.pass' is supported, got {na_action!r}"
            )
        # R's ``isRE``: re.form=None (include all) and re.form=NA (exclude
        # all) are the two we support; a partial-bars formula needs a
        # separate code path that we haven't ported yet.
        if re_form is None:
            include_re = True
        elif re_form is False:
            include_re = False
        else:
            raise NotImplementedError(
                "predict: re_form= only accepts None (include all RE) or "
                "False (population-level / no RE) in this port"
            )

        is_glmm = self.family.name != "gaussian" or self.family.link.name != "identity"

        # No-arg fast path — matches R's ``na.omit(fitted(object))``.
        # For GLMM, ``self.fitted`` is on the response scale (= μ̂); for LMM
        # μ ≡ η so both ``type`` values are the same value.
        if newdata is None and include_re and not random_only and not se_fit:
            if is_glmm and type == "link":
                return pl.DataFrame({"fit": self.eta.copy()})
            return pl.DataFrame({"fit": self.fitted.copy()})

        # Build X, Z, offset on the appropriate frame.
        if newdata is None:
            X_pred = self.X.to_numpy().astype(float)
            # Same workaround as _fit_glmm_from_components — polars-empty
            # design materialises as (0, 0) instead of (n, 0).
            if X_pred.shape == (0, 0):
                X_pred = np.zeros((self.n, 0), dtype=float)
            offset_pred = self._offset
            n_pred = self.n
        else:
            n_pred = newdata.height
            offset_pred = self._build_offset_for_newdata(newdata)
            X_pred = self._build_X_for_newdata(newdata)
            if X_pred.shape == (0, 0):
                X_pred = np.zeros((n_pred, 0), dtype=float)

        # Linear-predictor on the link scale: η = X·β + Z·b + offset.
        # ``random_only`` drops X·β AND offset (lme4 does the same — see
        # predict.R:464 ``pred <- rep(0, nobs)`` then conditional adds).
        eta_pred = np.zeros(n_pred)
        if not random_only:
            eta_pred = X_pred @ self._beta + offset_pred

        if include_re:
            if newdata is None:
                Z_pred = self.Z
            else:
                Z_pred = self._build_Z_for_newdata(
                    newdata, allow_new_levels=allow_new_levels,
                )
            ZL_pred = Z_pred @ self.Lambda
            eta_pred = eta_pred + ZL_pred @ self._u
        else:
            Z_pred = np.zeros((n_pred, self.q))
            ZL_pred = Z_pred

        # Response-scale conversion. For Gaussian-identity ``link(eta)=eta``
        # so this is a no-op and ``pred == eta_pred``.
        if type == "response" and is_glmm:
            pred = self.family.link.linkinv(eta_pred)
        else:
            pred = eta_pred

        if not se_fit:
            return pl.DataFrame({"fit": pred})

        # se.fit — joint (û, β̂) posterior covariance is ``σ² · M⁻¹``
        # where M is the Henderson MME in spherical (u, β) coordinates:
        #
        #   M = [Λᵀ Z'WZ Λ + I,  Λᵀ Z'WX]
        #       [    X'WZ Λ,        X'WX ]
        #
        # For GLMM, ``W`` is the working-weight diagonal lme4's
        # ``vcov_full`` (lmer.R:2281) reads off the **cached** ``pp$L`` /
        # ``pp$RZX`` factors — i.e. from the *last PIRLS iteration's
        # start*, one ``updateXwts`` behind the converged μ. Building M
        # from ``self.working_weights`` (fresh) instead would be slightly
        # more accurate but differs from lme4 at ~1e-5. We match lme4 by
        # reading the cached weighted matrices off ``self._pred`` so SEs
        # agree byte-for-byte.
        #
        # For Gaussian-identity, working weights ≡ 1, so the stale-vs-fresh
        # distinction collapses; we just rebuild M densely with W=I, which
        # is also what the cached ``_pred`` would give since for LMMs we
        # only need the unweighted version.
        if is_glmm:
            # Use cached weighted blocks from the last PIRLS iter.
            pp = self._pred
            lamt_ut_dense = np.asarray(pp.lamt_ut.todense())
            M_top = lamt_ut_dense @ lamt_ut_dense.T + np.eye(self.q)
            M_brc = pp.RZX_unfactored  # = lamt_ut · V = Λᵀ Z' W X
            M_bot = pp.VtV             # = V' V = X' W X
        else:
            X_fit = self.X.to_numpy().astype(float)
            if X_fit.shape == (0, 0):
                X_fit = np.zeros((self.n, 0), dtype=float)
            ZL_fit = self.Z @ self.Lambda
            M_top = ZL_fit.T @ ZL_fit + np.eye(self.q)
            M_brc = ZL_fit.T @ X_fit
            M_bot = X_fit.T @ X_fit
        M_full = np.block([[M_top, M_brc], [M_brc.T, M_bot]])

        if random_only:
            X_for_se = np.zeros((n_pred, self.p))
        else:
            X_for_se = X_pred
        if include_re:
            ZL_for_se = ZL_pred
        else:
            ZL_for_se = np.zeros((n_pred, self.q))
        ZLX_new = np.hstack([ZL_for_se, X_for_se])
        Minv_ZLX = np.linalg.solve(M_full, ZLX_new.T)
        var_pred = self.sigma_squared * np.einsum("ij,ji->i", ZLX_new, Minv_ZLX)
        # Floor tiny negatives from numerical cancellation.
        var_pred = np.maximum(var_pred, 0.0)
        se = np.sqrt(var_pred)
        # Delta method: SE on response scale = SE on link scale · |dμ/dη|.
        # lme4 does this at predict.R:654 for isGLMM + type=="response".
        if type == "response" and is_glmm:
            se = se * np.abs(self.family.link.mu_eta(eta_pred))
        return pl.DataFrame({"fit": pred, "se.fit": se})

    # ---- lmer-style printing --------------------------------------------

    def _is_glmm(self) -> bool:
        """``True`` for non-Gaussian-identity fits (Laplace path)."""
        fam = getattr(self, "family", None)
        if fam is None:
            return False
        return not (fam.name == "gaussian" and fam.link.name == "identity")

    def _header(self) -> str:
        # lme4's print.merMod tags the header with the S4 class name in
        # brackets (methods.R: ``['glmerMod']`` / ``['lmerMod']``).
        if self._is_glmm():
            return (
                "Generalized linear mixed model fit by maximum likelihood "
                "(Laplace Approximation) ['glmerMod']"
            )
        if self.REML:
            return "Linear mixed model fit by REML ['lmerMod']"
        return "Linear mixed model fit by maximum likelihood ['lmerMod']"

    def _fit_criterion_lines(self) -> list[str]:
        if self.REML:
            return [f"REML criterion at convergence: {self.REML_criterion:.4f}"]
        # GLMM's printed "-2*log(L)" column is the Laplace value (= AIC −
        # 2·npar), not the residual deviance. Mirror lme4's print.merMod
        # (methods.R: print.merMod for glmerMod).
        if self._is_glmm():
            dev_val = self.deviance_laplace
        else:
            dev_val = self.deviance
        # lme4's .prt.aictab rounds the criteria to 1 decimal (digits=1) and
        # prints df.resid as an integer (methods.R / lme4:::print.merMod).
        labels = ["AIC", "BIC", "logLik", "-2*log(L)", "df.resid"]
        vals = [
            f"{self.AIC:.1f}",
            f"{self.BIC:.1f}",
            f"{self.loglike:.1f}",
            f"{dev_val:.1f}",
            f"{self.df_resid}",
        ]
        widths = [max(len(lab), len(v)) for lab, v in zip(labels, vals)]
        hdr = " ".join(lab.rjust(w) for lab, w in zip(labels, widths))
        row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return [hdr, row]

    def _n_obs_line(self) -> str:
        groups = "; ".join(f"{g}, {n}" for g, n in self.n_groups.items())
        return f"Number of obs: {self.n}, groups:  {groups}"

    @staticmethod
    def _format_col(values: list[float]) -> list[str]:
        """Format a numeric column with shared decimal places (R's format())."""
        strs = [f"{v:.4g}" for v in values]
        if any("e" in s or "E" in s for s in strs):
            return strs
        max_dp = max((len(s.split(".")[1]) for s in strs if "." in s), default=0)
        if max_dp == 0:
            return strs
        return [f"{v:.{max_dp}f}" for v in values]

    def _re_table_lines(self, include_variance: bool) -> list[str]:
        max_corr_cols = 0
        for c in self.corr_re.values():
            if c is not None:
                max_corr_cols = max(max_corr_cols, c.shape[0] - 1)

        # Collect per-bar entries: (group_label, name, sd, variance, corrs)
        entries: list[tuple[str, str, float, float, list[float]]] = []
        for key in self.sd_re:
            names = self._re.cnms[key]
            if not isinstance(names, list):
                names = [names]
            sds = self.sd_re[key]
            corr = self.corr_re.get(key)
            for i, (name, s) in enumerate(zip(names, sds)):
                corrs = [corr[i, j] for j in range(i)] if (corr is not None and i > 0) else []
                entries.append((key if i == 0 else "", name, float(s), float(s) ** 2, corrs))
        # Residual SD: lme4 omits this row for scale-known GLMM families
        # (Poisson, Binomial — methods.R: print.merMod for glmerMod) since
        # σ ≡ 1 conveys no information; show it only when the scale is
        # estimated (LMM Gaussian, GLMM with Gamma / Inverse-Gaussian).
        if not (self._is_glmm() and getattr(self.family, "scale_known", False)):
            entries.append(("Residual", "", float(self.sigma), float(self.sigma_squared), []))

        sd_col = self._format_col([e[2] for e in entries])
        var_col = self._format_col([e[3] for e in entries]) if include_variance else None

        rows: list[list[str]] = []
        for idx, (group, name, _s, _v, corrs) in enumerate(entries):
            # Residual row: blank out the Name cell
            if group == "Residual" and idx == len(entries) - 1:
                name_cell = ""
            else:
                name_cell = name
            row = [group, name_cell]
            if var_col is not None:
                row.append(var_col[idx])
            row.append(sd_col[idx])
            row.extend(f"{c:.2f}" for c in corrs)
            rows.append(row)

        header = ["Groups", "Name"]
        if include_variance:
            header.append("Variance")
        header.append("Std.Dev.")
        if max_corr_cols > 0:
            header.append("Corr")
            header.extend([""] * (max_corr_cols - 1))

        ncols = len(header)
        for r in rows:
            r.extend([""] * (ncols - len(r)))
        widths = [len(h) for h in header]
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))

        def fmt(cells: list[str]) -> str:
            return (" " + " ".join(c.ljust(w) for c, w in zip(cells, widths))).rstrip()

        return [fmt(header)] + [fmt(r) for r in rows]

    def _fixef_table(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "coef": self.column_names,
                "Estimate": self._beta.astype(float),
                "Std. Error": self._se_beta.astype(float),
                "t value": (self._beta / self._se_beta).astype(float),
            }
        )

    def _fixef_corr_lines(self) -> list[str]:
        """Correlation-of-fixed-effects block, lme4-style (lower-triangular)."""
        p = self._vcov_beta_arr.shape[0]
        if p <= 1:
            return []
        vcov = self._vcov_beta_arr
        d = np.sqrt(np.diag(vcov))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = vcov / np.outer(d, d)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        names = ["(Intr)" if n == "(Intercept)" else n for n in self.column_names]
        row_w = max(len(n) for n in names[1:])
        cell_w = max(6, max(len(n) for n in names[: p - 1]))
        header = " " * row_w + " " + " ".join(
            names[j].rjust(cell_w) for j in range(p - 1)
        )
        rows = []
        for i in range(1, p):
            cells = " ".join(f"{corr[i, j]:.3f}".rjust(cell_w) for j in range(i))
            rows.append(names[i].ljust(row_w) + " " + cells)
        return ["Correlation of Fixed Effects:", header] + rows

    def __repr__(self) -> str:
        out = [self._header(), f"Formula: {self.formula}"]
        out.extend(self._fit_criterion_lines())
        out.append("Random effects:")
        out.extend(self._re_table_lines(include_variance=False))
        out.append(self._n_obs_line())
        out.append("Fixed Effects:")
        out.append(format_df(self.bhat))
        return "\n".join(out)

    def __str__(self) -> str:
        return self.__repr__()

    def _scaled_residuals_lines(self) -> list[str]:
        # lme4's print.summary.merMod uses ``residuals(., "pearson",
        # scaled=TRUE)`` — i.e. (y − μ)·√w / √V(μ) / σ. For LMM Gaussian
        # this reduces to (y − μ) / σ (matching the raw "scaled" residuals
        # users expect); for GLMM Binomial / Poisson it diverges from the
        # signed deviance residuals stored on ``self.residuals``.
        scaled = self.residuals_of("pearson") / self.sigma
        qs = np.quantile(scaled, [0.0, 0.25, 0.5, 0.75, 1.0])
        labels = ["Min", "1Q", "Median", "3Q", "Max"]
        vals = [f"{v:.4f}" for v in qs]
        widths = [max(len(lab), len(v)) for lab, v in zip(labels, vals)]
        hdr = " ".join(lab.rjust(w) for lab, w in zip(labels, widths))
        row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return ["Scaled residuals:", hdr, row]

    def simulate(self, nsim: int = 1, seed=None, use_u: bool = False):
        """Simulate ``nsim`` response vectors from the fitted model — port of
        ``simulate.merMod`` (predict.R:673-938).

        ``use_u=False`` (default; the parametric-bootstrap path) draws fresh
        random effects ``b ~ N(0, Σ_θ)`` per simulation via the fitted ``Λ``;
        ``use_u=True`` conditions on the fitted ``b̂``. Each draw then samples
        from the response family at the simulated mean. Returns a polars
        DataFrame with columns ``sim_1 … sim_nsim``.

        ``seed`` seeds the **bit-exact** :class:`RMersenneTwister`, and both the
        draw order (all ``q·nsim`` random-effect normals first, column-major,
        then the family draws over the column-major-flattened ``(n, nsim)`` mean
        matrix) and the per-family samplers (``rpois``/``rbinom``/``rgamma``/…)
        mirror ``simulate.merMod`` draw-for-draw. Given **identical** fitted
        parameters it reproduces R's ``simulate(model, seed=s)`` exactly; in
        practice hea's fit differs from lme4's at the ~1e-9 optimiser floor, so
        continuous draws agree to ~1e-9 and discrete draws can occasionally flip
        (most visibly when an obs's μ straddles the rpois inversion/PD boundary
        at μ=10, which switches RNG-consumption and desyncs the stream after it).
        """
        rng = _simulate_rng(seed)
        n, q, nsim = self.n, self.q, int(nsim)
        fam = self.family
        is_gaussian = (fam.name == "gaussian")
        eta_cond = self.fitted if is_gaussian else self.eta
        # Z·b̂ = Z·Λ·û — the fitted random-effects contribution (link scale).
        zb_hat = np.asarray(self._Z_sp @ (self.Lambda @ self._u)).ravel()
        eta_pop = eta_cond - zb_hat

        # (1) Random-effects normals — q·nsim drawn first (lme4 cluster.rand),
        # packed column-major into (q, nsim); skipped when conditioning on û.
        if use_u:
            eta_mat = np.repeat(eta_cond[:, None], nsim, axis=1)
        else:
            u_all = np.asarray(rng.rnorm(q * nsim)).reshape((nsim, q)).T
            reff = np.asarray(self._Z_sp @ (self.Lambda @ u_all))
            # LMM scales the relative RE draw ``ZΛu`` by σ — lme4 writes the
            # whole thing as ``etapred + σ·(sim.reff + ε)`` (predict.R:882),
            # since ``u ~ N(0, σ²I)`` in lme4's parameterization. GLMM leaves
            # the RE contribution **unscaled** (predict.R:890, σ≡1 on the link
            # scale). ``use_u=True`` redraws nothing — it conditions on b̂.
            if is_gaussian:
                reff = self.sigma * reff
            eta_mat = eta_pop[:, None] + reff

        # (2) Response draws over the column-major-flattened (n, nsim) means.
        if is_gaussian:
            resid = np.asarray(rng.rnorm(n * nsim)).reshape((nsim, n)).T
            y_mat = eta_mat + self.sigma * resid
        else:
            mu_flat = fam.link.linkinv(eta_mat).ravel(order="F")
            w = getattr(self, "prior_weights", None)
            w_flat = None if w is None else np.tile(np.asarray(w, float), nsim)
            y_flat = _simulate_family_draw(rng, fam, mu_flat, w_flat,
                                           self.sigma)
            y_mat = y_flat.reshape((nsim, n)).T
        return pl.DataFrame({f"sim_{k + 1}": y_mat[:, k] for k in range(nsim)})

    # ---- refit / parametric bootstrap ----------------------------------

    def _refit_response(self, newresp) -> "gmm":
        """Refit this model to a new response vector, preserving family / REML
        / weights / offset / nAGQ / control — the engine behind :meth:`bootMer`
        and the ``refit(model, newresp=)`` generic (lme4's ``refit.merMod``).

        The response must be a bare data column (a ``cbind(...)`` / transformed
        LHS isn't supported); hea expresses binomial models in proportion form
        (``y_prop`` + ``weights=size``), so this still covers the canonical
        cbpp / Beetle dose-response cases. ``weights`` (the binomial trial
        totals) and the user ``offset=`` arg ride along so the refit is the
        same model with only the response swapped.
        """
        resp = np.asarray(newresp, dtype=float).ravel()
        if resp.shape != (self.n,):
            raise ValueError(
                f"_refit_response: newresp must have length {self.n}; "
                f"got {resp.shape}")
        lhs = self.formula.split("~", 1)[0].strip()
        if lhs not in self.data.columns:
            raise NotImplementedError(
                f"refit(newresp=): response {lhs!r} is not a bare data column "
                f"(cbind() / transformed LHS not supported yet)")
        data = self.data.with_columns(pl.Series(lhs, resp))
        w = getattr(self, "prior_weights", None)
        if w is not None and np.allclose(np.asarray(w, float), 1.0):
            w = None  # unit weights ⇒ pass None (avoids a spurious weighted fit)
        return gmm(
            self.formula, data, family=self.family, REML=self.REML,
            weights=w, offset=self._offset_arg, nAGQ=self._nAGQ,
            control=self._control_arg,
        )

    @staticmethod
    def _boot_apply_fun(FUN, model):
        """Apply a bootstrap statistic ``FUN`` to a fit, returning
        ``(values, names)``. A ``dict`` return carries its own names; anything
        else (ndarray / list / scalar) gets positional ``t1, t2, …`` labels —
        matching how R names an unnamed ``FUN`` result."""
        raw = FUN(model)
        if isinstance(raw, dict):
            return (np.asarray(list(raw.values()), dtype=float).ravel(),
                    list(raw.keys()))
        if isinstance(raw, pl.Series):
            return raw.to_numpy().astype(float).ravel(), [
                f"t{i + 1}" for i in range(raw.len())]
        arr = np.asarray(raw, dtype=float).ravel()
        return arr, [f"t{i + 1}" for i in range(arr.size)]

    def bootMer(self, FUN, nsim: int = 1, seed=None, use_u: bool = False,
                type: str = "parametric", re_form=None, verbose: bool = False,
                parallel: str = "no", ncpus: int = 1) -> "BootMer":
        """Model-based parametric bootstrap — port of ``bootMer`` (bootMer.R).

        Simulates ``nsim`` responses from the fitted model (via
        :meth:`simulate`), refits the model to each (:meth:`_refit_response`),
        and applies ``FUN`` to every refit. Returns a :class:`BootMer` holding
        ``t0 = FUN(self)`` and the ``nsim × len(t0)`` replicate matrix, ready
        for :meth:`BootMer.confint`.

        Parameters mirror lme4: ``use_u`` conditions the RE draws on ``b̂``;
        ``type ∈ {"parametric","semiparametric"}`` (semiparametric needs
        ``use_u=True`` — resampling response residuals — and warns for GLMMs,
        exactly as lme4 does); ``seed`` drives the **bit-exact** RNG so a
        matched seed reproduces R's stream. ``FUN`` must return a numeric
        vector (or a ``{name: value}`` dict to label the statistics).

        All ``nsim`` draws are generated up front in one sequential RNG pass,
        so the refits are pure functions of the simulated data and ``parallel``
        (``"multicore"`` / ``"snow"`` / ``"future"`` → a process pool) never
        perturbs reproducibility. Failed refits become an all-``NaN`` row
        (lme4's ``factory(errval=NA)``).
        """
        import warnings

        nsim = int(nsim)
        if nsim <= 0:
            raise ValueError("bootMer: nsim must be a positive integer")
        if type not in ("parametric", "semiparametric"):
            raise ValueError(
                f"bootMer: type must be 'parametric'/'semiparametric'; got {type!r}")

        t0, t0_names = self._boot_apply_fun(FUN, self)
        if not np.issubdtype(t0.dtype, np.number):
            raise TypeError(
                "bootMer currently only handles functions that return "
                "numeric vectors")

        # (1) Generate all nsim responses up front (one sequential RNG pass).
        if type == "parametric":
            ss = self.simulate(nsim=nsim, seed=seed, use_u=use_u)
            sims = [ss[c].to_numpy() for c in ss.columns]
        else:  # semiparametric — resample response residuals on top of fitted
            if not use_u:
                raise NotImplementedError(
                    "semiparametric bootstrapping with use_u=False is not "
                    "implemented (matches lme4)")
            if self._is_glmm():
                warnings.warn(
                    "semiparametric bootstrapping is questionable for GLMMs")
            rng = _simulate_rng(seed)
            ftd = np.asarray(self.fitted, dtype=float).ravel()
            res_resp = np.asarray(self.residuals_of("response")
                                  if hasattr(self, "residuals_of")
                                  else (self.y - self.fitted), dtype=float).ravel()
            sims = [ftd + res_resp[rng.sample_int(self.n, self.n, replace=True)]
                    for _ in range(nsim)]

        # (2) Refit each simulated response and apply FUN; failures → NaN row.
        def _one(y_k):
            try:
                return self._boot_apply_fun(FUN, self._refit_response(y_k))[0]
            except Exception:  # noqa: BLE001 — lme4's factory swallows + NA-fills
                return np.full(t0.shape, np.nan)

        if parallel == "no" or ncpus <= 1:
            results = []
            for i, y_k in enumerate(sims):
                results.append(_one(y_k))
                if verbose:
                    print(f"{i + 1:5d} : {results[-1]}")
        else:
            results = self._bootmer_parallel(sims, FUN, t0, ncpus)

        t = np.vstack(results)
        nfail = int(np.isnan(t).any(axis=1).sum())
        if nfail > 0:
            warnings.warn(f"some bootstrap runs failed ({nfail}/{nsim})")
        return BootMer(t0, t, t0_names, R=nsim, seed=seed, nfail=nfail)

    def _bootmer_parallel(self, sims, FUN, t0, ncpus):
        """Process-pool refit fan-out for ``parallel != "no"``. Ships a
        picklable fit-spec per simulation to top-level :func:`_bootmer_worker`.
        Falls back to a sequential run (with a warning) if the model / FUN
        can't be pickled — common when ``FUN`` is a lambda or closure."""
        import warnings
        from concurrent.futures import ProcessPoolExecutor

        lhs = self.formula.split("~", 1)[0].strip()
        w = getattr(self, "prior_weights", None)
        if w is not None and np.allclose(np.asarray(w, float), 1.0):
            w = None
        spec_base = (self.formula, lhs, self.data, self.family, self.REML,
                     w, self._offset_arg, self._nAGQ, self._control_arg,
                     FUN, int(t0.size))
        try:
            with ProcessPoolExecutor(max_workers=int(ncpus)) as ex:
                return list(ex.map(_bootmer_worker,
                                   [(*spec_base, y_k) for y_k in sims]))
        except Exception as exc:  # noqa: BLE001 — pickling / spawn failure
            warnings.warn(
                f"bootMer: parallel run failed ({exc!r}); falling back to "
                f"sequential")
            out = []
            for y_k in sims:
                try:
                    out.append(self._boot_apply_fun(FUN, self._refit_response(y_k))[0])
                except Exception:  # noqa: BLE001
                    out.append(np.full(int(t0.size), np.nan))
            return out

    def summary(self, digits: int = 4) -> None:
        from scipy.stats import norm
        out = [self._header()]
        if self._is_glmm():
            out.append(f" Family: {self.family.name}  ( {self.family.link.name} )")
        out.append(f"Formula: {self.formula}")
        out.append("")
        out.extend(self._fit_criterion_lines())
        out.append("")
        out.extend(self._scaled_residuals_lines())
        out.append("")
        out.append("Random effects:")
        out.extend(self._re_table_lines(include_variance=True))
        out.append(self._n_obs_line())
        out.append("")
        out.append("Fixed effects:")
        raw = self._fixef_table().rename({"coef": ""})
        est_arr = raw["Estimate"].to_numpy()
        se_arr  = raw["Std. Error"].to_numpy()
        tval    = raw["t value"].to_numpy()
        est_s, se_s = format_signif_jointly([est_arr, se_arr], digits=digits)
        # GLMM uses z + Pr(>|z|) (asymptotic normal — lme4's print.coefmat
        # for glmerMod); LMM keeps lme4's t-no-p convention.
        if self._is_glmm():
            p_arr = 2.0 * (1.0 - norm.cdf(np.abs(tval)))
            tbl = pl.DataFrame({
                "":           raw[""].to_list(),
                "Estimate":   est_s,
                "Std. Error": se_s,
                "z value":    format_signif(tval, digits=digits),
                "Pr(>|z|)":   format_pval(p_arr),
                " ":          significance_code(p_arr),
            })
            align_cols = ("Estimate", "Std. Error", "z value", "Pr(>|z|)")
        else:
            tbl = pl.DataFrame({
                "":           raw[""].to_list(),
                "Estimate":   est_s,
                "Std. Error": se_s,
                "t value":    format_signif(tval, digits=digits),
            })
            align_cols = ("Estimate", "Std. Error", "t value")
        out.append(format_df(tbl, align={c: "right" for c in align_cols}))
        # lme4 prints the signif-code legend only when at least one row has
        # a Pr-column (i.e. GLMM). LMM summary intentionally omits both
        # p-values and the legend (the df-controversy footnote in lme4's
        # docs: "Why are p-values not displayed").
        if self._is_glmm():
            out.append("---")
            out.append(
                "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )
        corr_lines = self._fixef_corr_lines()
        if corr_lines:
            out.append("")
            out.extend(corr_lines)
        # convergence diagnostics block, appended verbatim after the
        # correlation matrix when there's anything to report. Mirrors lme4's
        # ``print.summary.merMod`` (methods.R:158-176) which prints the
        # collected ``optinfo$conv$lme4$messages`` at the tail.
        opt_messages = getattr(self, "optinfo", {}).get(
            "conv", {}).get("lme4", {}).get("messages", [])
        if opt_messages:
            out.append("")
            opt_name = getattr(self, "optinfo", {}).get(
                "optimizer", "bobyqa+Nelder_Mead")
            out.append(
                f"optimizer ({opt_name}) convergence code: 0 (OK)")
            for msg in opt_messages:
                out.append(msg)
        print("\n".join(out))

    # ---- diagnostic plots ----------------------------------------------

    def _ranef(self):
        """BLUPs in original units with posterior SEs, sliced per bar.

        Returns a list of ``(bar_key, levels, cnames, b_mat, se_mat)`` —
        ``b_mat`` and ``se_mat`` are ``(n_levels, n_components)`` arrays.

        Posterior covariance: ``Var(b̂ | y) = σ² · Λ M⁻¹ Λᵀ``. We pull the
        diagonal in ``O(q²)`` via one dense ``F.solve(Λᵀ_dense)``; ``q``
        well into the thousands triggers heavy work, so this is lazy and
        cached. Defensively re-factorizes ``M`` at θ̂ since callers like
        ``profile()`` over-write the factor during their own optimization.
        """
        cache = getattr(self, "_ranef_cache", None)
        if cache is not None:
            return cache
        Lt = self._build_Lt_sparse(self.theta)
        ZL = self._Z_sp @ Lt.T
        M = (ZL.T @ ZL + self._eye_q_sp).tocsc()
        self._chol_factor.factorize(M)
        F = self._chol_factor
        Lt_dense = Lt.toarray()
        b_full = (Lt_dense.T @ self._u).ravel()
        M_inv_Lt = F.solve(Lt_dense)
        var_b = self.sigma_squared * (Lt_dense * M_inv_Lt).sum(axis=0)
        se_full = np.sqrt(np.clip(var_b, 0.0, None))

        out = []
        Gp = self._re.Gp
        flist = self._re.flist_levels
        for k, key in enumerate(self._re.cnms):
            start, end = Gp[k], Gp[k + 1]
            cnames = self._re.cnms[key]
            cnames = list(cnames) if isinstance(cnames, list) else [cnames]
            c = len(cnames)
            n_levels = (end - start) // c
            b_mat = b_full[start:end].reshape(n_levels, c)
            se_mat = se_full[start:end].reshape(n_levels, c)
            # Recover original group name (lme4 suffixes ".1", ".2" if reused)
            gname = key
            if gname not in flist:
                base, _, tail = key.rpartition(".")
                if tail.isdigit() and base in flist:
                    gname = base
            levels = list(flist[gname])
            out.append((key, levels, cnames, b_mat, se_mat))
        self._ranef_cache = out
        return out

    @property
    def ranef(self) -> dict[str, pl.DataFrame]:
        """BLUPs per random-effect bar — lme4's ``ranef(m)`` shape.

        Returns one polars DataFrame per bar (keyed by bar name, e.g.
        ``"Subject"``, or ``"Subject.1"`` when the same grouping factor
        appears twice). First column carries the level labels under the
        grouping factor's name; remaining columns are the BLUPs, one per
        random-effect component (``(Intercept)``, slope names, …).
        """
        out: dict[str, pl.DataFrame] = {}
        for key, levels, cnames, b_mat, _se in self._ranef():
            gname = key
            if gname not in self.n_groups:
                base, _, tail = key.rpartition(".")
                if tail.isdigit() and base in self.n_groups:
                    gname = base
            cols: dict[str, list] = {gname: list(levels)}
            for j, cn in enumerate(cnames):
                cols[cn] = b_mat[:, j].tolist()
            out[key] = pl.DataFrame(cols)
        return out

    def _pooled_std_blups(self) -> np.ndarray:
        """All BLUPs concatenated, each component scaled by its model SD.

        Used by the 2×2 ``plot()``'s combined random-effect Q-Q panel.
        """
        out = []
        for key, _levels, _cnames, b_mat, _se in self._ranef():
            sds = self.sd_re[key]
            for j, sd in enumerate(sds):
                if sd > 0:
                    out.append(b_mat[:, j] / float(sd))
        if not out:
            return np.array([])
        return np.concatenate(out)

    def plot_observed_fitted(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black", label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        y = np.asarray(self.y, dtype=float)
        yhat = np.asarray(self.fitted, dtype=float)
        ax.scatter(yhat, y, facecolor=facecolor, edgecolor=edgecolor)
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        _label_top_n(ax, yhat, y, scores=self.residuals, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Observed")
        ax.set_title("Observed vs. Fitted")
        return ax

    def plot_residuals(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        yhat = np.asarray(self.fitted, dtype=float)
        r = np.asarray(self.residuals, dtype=float)
        ax.scatter(yhat, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--")
        if smooth:
            xs, ys = _lowess(yhat, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, r, scores=r, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Fitted Plot")
        return ax

    def plot_qq(self, ax=None, figsize=None, label_n=3):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        _qq_plot(ax, self.scaled_residuals, label_n=label_n)
        return ax

    def plot_scale_location(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        yhat = np.asarray(self.fitted, dtype=float)
        s = np.sqrt(np.abs(self.scaled_residuals))
        ax.scatter(yhat, s, facecolor=facecolor, edgecolor=edgecolor)
        if smooth:
            xs, ys = _lowess(yhat, s)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, s, scores=self.scaled_residuals, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\ Residuals}|}$")
        ax.set_title("Scale-Location")
        return ax

    def plot_design(self, *, figsize=None, cmap: str = "BuPu", gamma: float = 0.5):
        """4-panel design-matrix diagnostic (Bates lme4 book Figs 2.3 + 2.4).

        Layout::

            AAA      A = Z'   — transpose of the random-effects design
            BCD      B = Λ    — relative covariance factor
                     C = Z'Z  — cross-product matrix
                     D = L    — sparse Cholesky factor of Λ′Z′ZΛ + I

        Renders each matrix's magnitudes (not just sparsity) with a
        cyan-purple sequential palette and a γ < 1 power norm — matches
        the lattice ``Matrix::image()`` look used in the lme4 book, where
        off-diagonal small values (e.g. plate-sample crossings in Z'Z)
        stay visible alongside the much larger diagonal counts.

        Parameters
        ----------
        cmap
            Sequential matplotlib colormap name. Default ``"BuPu"`` is
            the Brewer Blue-Purple ramp; ``"PuBu"`` / ``"Blues"`` are
            close alternatives.
        gamma
            Exponent for :class:`matplotlib.colors.PowerNorm` — values
            below 1 compress the high end so low non-zeros remain
            visible against a much larger diagonal. Set ``gamma=1`` for
            a linear scale, or pass a larger ``cmap`` if you want a
            stark binary look.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        Z = self.Z if isinstance(self.Z, np.ndarray) else self.Z.toarray()
        ZtZ = Z.T @ Z

        if figsize is None:
            # Bottom row is q×q each; top row spans three columns wide.
            # 10×7 looks good for q in the 6..50 range Bates uses.
            figsize = (10, 7)

        fig = plt.figure(figsize=figsize)
        axd = fig.subplot_mosaic(
            """
            AAA
            BCD
            """,
            gridspec_kw={"height_ratios": [1, 2]},
        )

        def _show(ax, M, *, aspect=None):
            vmax = float(np.abs(M).max() or 1.0)
            norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
            kwargs = {"cmap": cmap, "interpolation": "nearest", "norm": norm}
            if aspect is not None:
                kwargs["aspect"] = aspect
            ax.imshow(M, **kwargs)

        _show(axd["A"], Z.T, aspect="auto")
        axd["A"].set_ylabel("random-effect")
        axd["A"].set_xlabel("Z'")

        _show(axd["B"], self.Lambda)
        axd["B"].set_xlabel("Λ")

        _show(axd["C"], ZtZ)
        axd["C"].set_xlabel("Z'Z")

        _show(axd["D"], self.L)
        axd["D"].set_xlabel("L")

        fig.tight_layout()
        return fig

    def plot_qq_ranef(
        self, figsize=None,
        *, level: float = 0.95, strip: bool = True,
    ):
        """qqmath of BLUPs with conditional-variance bars (Bates Fig. 1.12).

        Pythonic ``qqmath(ranef(., condVar=TRUE), strip=...)``. BLUPs on the
        x-axis at y = Φ⁻¹((i−0.5)/n) (Hazen plotting position, matches
        lme4); horizontal bars of half-width Φ⁻¹((1+level)/2)·SE (default
        95%); vertical line at x=0. ``strip=False`` suppresses per-panel
        titles.
        """
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2))
        panels = []
        for key, _levels, cnames, b_mat, se_mat in self._ranef():
            for j, cname in enumerate(cnames):
                panels.append((f"{key}: {cname}", b_mat[:, j], se_mat[:, j]))
        n_panels = len(panels)
        if figsize is None:
            figsize = (3.2 * n_panels, 3.0)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for ax, (title, b, se) in zip(axes, panels):
            order = np.argsort(b)
            b_s = b[order]
            se_s = se[order]
            n = len(b_s)
            q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            ax.grid(True, color="lightgray", linewidth=0.4)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.errorbar(
                b_s, q, xerr=z * se_s, fmt="o", color="black",
                ecolor="black", markersize=3, linewidth=0.8, capsize=0,
            )
            ax.set_ylabel("Standard normal quantiles")
            ax.set_title(title if strip else "")
        fig.tight_layout()
        return fig

    def plot_ranef(
        self, figsize=None,
        *, level: float = 0.95, strip: bool = True,
        layout: str | tuple[int, int] = "horizontal",
        aspect: float | None = None,
        which: str | list[str] | None = None,
    ):
        """Caterpillar plot — BLUP ± Φ⁻¹((1+level)/2)·SE per level, sorted.

        Pythonic ``dotplot(ranef(., condVar=TRUE))``: defaults to 95%
        prediction intervals to match lme4. ``strip=False`` suppresses
        per-panel titles (Bates Fig. 1.5 convention).

        Parameters
        ----------
        layout : {"horizontal", "vertical"} or ``(nrow, ncol)``
            Panel arrangement. ``"horizontal"`` (default) lays panels in
            a single row — lme4-book convention. ``"vertical"`` stacks
            them in a single column. Pass an explicit ``(nrow, ncol)``
            tuple for a grid; ``nrow * ncol`` must hold every panel.
        aspect : float, optional
            Width-to-height ratio of each subplot in inches. When set,
            ``figsize`` is derived from it together with ``layout`` and
            the largest panel's level count. Ignored when ``figsize`` is
            passed explicitly.
        which : str or list of str, optional
            Restrict the figure to a subset of ranef panels. Accepts:

            * A term key (e.g. ``"Subject"``) — pulls every panel for
              that grouping factor (both the intercept and any slope
              columns of a vector bar).
            * A full panel title (e.g. ``"Subject: Days"``) — picks
              exactly one panel.
            * A list mixing the two forms.

            ``None`` (default) plots every panel.
        """
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2))
        panels = []
        for key, levels, cnames, b_mat, se_mat in self._ranef():
            for j, cname in enumerate(cnames):
                panels.append(
                    (f"{key}: {cname}", b_mat[:, j], se_mat[:, j], levels, key)
                )

        if which is not None:
            wanted = {which} if isinstance(which, str) else set(which)
            filtered = [p for p in panels if p[0] in wanted or p[4] in wanted]
            if not filtered:
                available = sorted({p[0] for p in panels}) + sorted(
                    {p[4] for p in panels}
                )
                raise KeyError(
                    f"plot_ranef(which={which!r}): no matching panel. "
                    f"Available term keys / panel titles: {available!r}."
                )
            panels = filtered

        n_panels = len(panels)

        # Resolve layout to (nrow, ncol).
        if isinstance(layout, tuple):
            if len(layout) != 2 or not all(isinstance(x, int) and x > 0
                                            for x in layout):
                raise TypeError(
                    f"layout: tuple must be (nrow, ncol) of positive ints; got {layout!r}."
                )
            nrow, ncol = layout
            if nrow * ncol < n_panels:
                raise ValueError(
                    f"layout={layout!r}: holds {nrow * ncol} cells but the "
                    f"model has {n_panels} ranef panels."
                )
        elif layout == "horizontal":
            nrow, ncol = 1, n_panels
        elif layout == "vertical":
            nrow, ncol = n_panels, 1
        else:
            raise ValueError(
                f"layout: expected 'horizontal', 'vertical', or (nrow, ncol); got {layout!r}."
            )

        # Pick a sensible figsize when not given. Subplot height tracks
        # the largest panel's level count; width is derived from
        # ``aspect`` when supplied, else a constant 3.2".
        max_levels = max(len(p[3]) for p in panels)
        if figsize is None:
            subplot_h = max(2.5, min(0.18 * max_levels, 12.0))
            subplot_w = aspect * subplot_h if aspect is not None else 3.2
            figsize = (subplot_w * ncol, subplot_h * nrow)

        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel()
        for ax, (title, b, se, levels, _key) in zip(axes_flat, panels):
            order = np.argsort(b)
            b_sorted = b[order]
            se_sorted = se[order]
            labels_sorted = [str(levels[i]) for i in order]
            n = len(b)
            y_pos = np.arange(n)
            for y in y_pos:
                ax.axhline(y, color="lightgray", linewidth=0.4, zorder=0)
            ax.errorbar(
                b_sorted, y_pos, xerr=z * se_sorted,
                fmt="o", color="black", ecolor="black",
                markersize=3, capsize=0, linewidth=0.8,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_yticks(y_pos)
            if n <= 30:
                ax.set_yticklabels(labels_sorted, fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Random Effect")
            ax.set_title(title if strip else "")
        # Hide unused cells (only possible with an explicit (nrow, ncol)
        # tuple that has more cells than panels).
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig

    def plot(self, figsize=None, smooth=True, label_n=3):
        """4-panel diagnostic: Residuals, Q-Q residuals, Scale-Location, Q-Q BLUPs."""
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_residuals(ax=axes[0, 0], smooth=smooth, label_n=label_n)
        self.plot_qq(ax=axes[0, 1], label_n=label_n)
        self.plot_scale_location(ax=axes[1, 0], smooth=smooth, label_n=label_n)
        pooled = self._pooled_std_blups()
        if len(pooled) >= 4:
            _qq_plot(
                axes[1, 1], pooled, label_n=label_n,
                ylabel="Standardized BLUPs (pooled)",
                title="Random-Effects Q-Q",
            )
        else:
            axes[1, 1].set_title("Random-Effects Q-Q (n too small)")
        fig.tight_layout()
        return fig


def _resolve_transform(t):
    """Map a ``transform=`` argument to a (forward-fn, title-format) pair."""
    if t is None:
        return (lambda x: np.asarray(x)), "{}"
    if callable(t):
        return t, "{}"
    if t == "log":
        return np.log, "log({})"
    if t in ("square", "sq"):
        return np.square, "{}²"
    raise ValueError(f"unknown transform {t!r}; use 'log', 'square', or a callable")


def _invert_zeta(
    vals: np.ndarray, zetas: np.ndarray, target: float,
    *, fallback: float = float("nan"),
) -> float:
    """Cubic-spline-interpolate the ζ-curve to find where ζ(v) = target.

    Matches R's ``confint(profile(...))`` which uses ``splines::interpSpline``
    on the ``ζ → v`` mapping — linear interpolation across two adjacent
    grid points loses noticeable curvature near ±z (visible as ~0.25
    units of error in the Dyestuff (Intercept) 99% bounds). Falls back to
    linear interp when there are too few points for a cubic.

    Returns ``fallback`` if ``target`` falls outside the observed ζ range —
    callers pass 0 for variance-component SDs (natural lower bound; matches
    lme4 when the profile flattens to an asymptote above the threshold) and
    NaN for unbounded parameters. Sorts by ζ first so the interpolation
    works even when the curve isn't evaluated on a monotone-in-v grid.
    """
    if target < np.nanmin(zetas) or target > np.nanmax(zetas):
        return fallback
    if len(vals) < 4:
        order = np.argsort(zetas)
        return float(np.interp(target, zetas[order], vals[order]))
    # Match R: fit a forward natural cubic spline ζ = f(v), then numerically
    # invert. The forward direction is monotonic and smooth even at .sig
    # boundary corners (where ζ at v=0 is a finite asymptote, not ±∞), so
    # the spline isn't pulled into the oscillations that fitting v(ζ) on
    # the same data triggers. R uses splines::interpSpline + backSpline.
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    v_order = np.argsort(vals)
    v_sorted, z_sorted = vals[v_order], zetas[v_order]
    fwd = CubicSpline(v_sorted, z_sorted, bc_type="natural", extrapolate=False)
    # Find the bracket: target lies between two consecutive ζ-knots.
    diffs = z_sorted - target
    sign_change = np.where(diffs[:-1] * diffs[1:] <= 0)[0]
    if len(sign_change) == 0:
        return float(np.interp(target, np.sort(zetas), vals[np.argsort(zetas)]))
    i = int(sign_change[0])
    return float(brentq(lambda v: float(fwd(v)) - target, v_sorted[i], v_sorted[i + 1]))


def _norm_inter(t: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Port of ``boot:::norm.inter`` — interpolated order statistics.

    For each probability ``α`` the ``boot`` percentile method does NOT take a
    plain ``quantile(t, α)``; it interpolates between the two adjacent order
    statistics ``t_(k)``/``t_(k+1)`` (``k = ⌊(R+1)α⌋``) on the **normal-quantile**
    scale: ``t_(k) + (Φ⁻¹(α) − Φ⁻¹(k/(R+1))) / (Φ⁻¹((k+1)/(R+1)) − Φ⁻¹(k/(R+1)))
    · (t_(k+1) − t_(k))``. Endpoints (``k = 0`` / ``k ≥ R``) clamp to the min/max;
    an exact integer ``rk`` returns ``t_(k)`` directly. Byte-matches R (verified
    against ``boot.ci`` percentile/basic output).
    """
    from scipy.stats import norm

    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    R = t.size
    alpha = np.atleast_1d(np.asarray(alpha, float))
    tstar = np.sort(t)  # ascending; R's tstar[j] is tstar[j-1] here (0-indexed)
    rk = (R + 1) * alpha
    k = np.trunc(rk).astype(int)
    out = np.empty_like(alpha)
    for i in range(alpha.size):
        ki = int(k[i])
        if ki == rk[i]:            # exact order statistic
            out[i] = tstar[ki - 1]
        elif ki == 0:              # below the first order statistic
            out[i] = tstar[0]
        elif ki >= R:              # at/above the last
            out[i] = tstar[R - 1]
        else:
            t1 = norm.ppf(alpha[i])
            t2 = norm.ppf(ki / (R + 1))
            t3 = norm.ppf((ki + 1) / (R + 1))
            tk, tk1 = tstar[ki - 1], tstar[ki]
            out[i] = tk + (t1 - t2) / (t3 - t2) * (tk1 - tk)
    return out


def _boot_ci_one(t0: float, t_col: np.ndarray, conf: float, kind: str) -> tuple[float, float]:
    """One parameter's bootstrap CI — port of ``boot::boot.ci`` for the three
    types ``confint.bootMer`` exposes (the last two columns of each method):

    * ``perc``  — ``boot:::perc.ci``:  interpolated percentiles at ``α/1−α``.
    * ``basic`` — ``boot:::basic.ci``: ``2·t0 − perc`` (pivot reflection).
    * ``norm``  — ``boot:::norm.ci``:  ``(t0 − bias) ± Φ⁻¹((1+conf)/2)·sd(t)``
      with ``bias = mean(t) − t0`` and ``sd`` the bootstrap-replicate SD
      (divisor ``R−1``).
    """
    from scipy.stats import norm

    t_col = np.asarray(t_col, float)
    finite = t_col[np.isfinite(t_col)]
    if kind == "norm":
        bias = float(finite.mean()) - t0
        merr = float(finite.std(ddof=1)) * float(norm.ppf((1 + conf) / 2))
        return (t0 - bias - merr, t0 - bias + merr)
    if kind == "perc":
        lo, hi = _norm_inter(t_col, np.array([(1 - conf) / 2, (1 + conf) / 2]))
        return (float(lo), float(hi))
    if kind == "basic":
        # basic.ci uses the ((1+conf)/2, (1-conf)/2) order, then 2·t0 − qq.
        qq = _norm_inter(t_col, np.array([(1 + conf) / 2, (1 - conf) / 2]))
        return (float(2 * t0 - qq[0]), float(2 * t0 - qq[1]))
    raise ValueError(f"unknown boot CI type {kind!r}; use perc/basic/norm")


def _bootmer_worker(spec):
    """Top-level (picklable) bootstrap worker for ``ProcessPoolExecutor`` —
    refit on one simulated response and apply ``FUN``. Returns an all-``NaN``
    row on failure (mirrors lme4's ``factory(errval=NA)``)."""
    (formula, lhs, data, family, reml, weights, offset, nagq, control,
     FUN, t0_len, y_k) = spec
    try:
        data_k = data.with_columns(
            pl.Series(lhs, np.asarray(y_k, dtype=float).ravel()))
        m = gmm(formula, data_k, family=family, REML=reml, weights=weights,
                offset=offset, nAGQ=nagq, control=control)
        return gmm._boot_apply_fun(FUN, m)[0]
    except Exception:  # noqa: BLE001
        return np.full(t0_len, np.nan)


class Profile:
    """Profile-likelihood output from :meth:`gmm.profile`.

    Attributes
    ----------
    data : dict[str, polars.DataFrame]
        Per-parameter table with columns ``value`` and ``zeta``. Keys are
        ``.sig01``, ``.sig02``, … for variance-component SDs, ``.sigma``
        for the residual SD, and the R-canonical fixed-effect names
        (``(Intercept)``, ``MachineB``, …).
    estimate : dict[str, float]
        MLE for each profiled parameter, keyed the same way.
    """

    def __init__(self, data: dict[str, pl.DataFrame], estimate: dict[str, float]):
        self.data = data
        self.estimate = estimate

    def confint(self, level: float = 0.95) -> pl.DataFrame:
        """Profile-based confidence intervals at ``level`` (default 95%).

        Inverts each ζ-curve at ±Φ⁻¹((1+level)/2). For variance-component
        SDs (``.sig01``, ``.sig02``, …, ``.sigma``) the lower bound clips
        to 0 when the profile flattens to an asymptote above the threshold
        (matches lme4; see book Fig. 1.8). Unbounded parameters return
        ``NaN`` if the curve doesn't cross the threshold within the grid.
        """
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + level / 2))
        lo_lbl = f"{100 * (1 - level) / 2:.1f}%"
        hi_lbl = f"{100 * (0.5 + level / 2):.1f}%"
        names: list[str] = []
        lo: list[float] = []
        hi: list[float] = []
        for name, df in self.data.items():
            v = df[name].to_numpy()
            s = df["zeta"].to_numpy()
            names.append(name)
            lo_fb = 0.0 if name.startswith(".sig") else float("nan")
            lo.append(_invert_zeta(v, s, -z, fallback=lo_fb))
            hi.append(_invert_zeta(v, s, +z))
        return pl.DataFrame({"parameter": names, lo_lbl: lo, hi_lbl: hi})

    def plot(
        self, absolute: bool = False, figsize: tuple[float, float] | None = None,
        levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95, 0.99),
        *,
        which: str | list[str] | None = None,
        transform: str | "Callable[[np.ndarray], np.ndarray]" | None = None,
        ax=None,
    ):
        """Profile zeta plot — the Pythonic replacement for R's
        ``xyplot(profile(...))``. One subplot per parameter; vertical
        gray lines mark the CI cutoffs for each level in ``levels``.

        With ``absolute=True`` plots ``|ζ|`` (matches book Fig. 1.6).

        ``which`` restricts to one parameter (str) or a subset (list).
        ``transform`` re-scales the x-axis: ``"log"`` for log(v),
        ``"square"`` for v², or any callable. CI cutoff verticals are
        forward-transformed too.

        Pass ``ax`` to draw into a pre-existing Axes (requires ``which`` to
        resolve to a single parameter). Useful for Bates Fig. 1.7-style
        layouts::

            fig, axes = plt.subplots(1, 3, sharey=True)
            pr.plot(which=".sigma", transform="log",    ax=axes[0])
            pr.plot(which=".sigma",                     ax=axes[1])
            pr.plot(which=".sigma", transform="square", ax=axes[2])
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        if which is None:
            names = list(self.data.keys())
        elif isinstance(which, str):
            names = [which]
        else:
            names = list(which)
        unknown = [n for n in names if n not in self.data]
        if unknown:
            raise KeyError(
                f"unknown parameter(s) {unknown!r}; available: {list(self.data)!r}"
            )
        if ax is not None and len(names) != 1:
            raise ValueError("ax= requires a single parameter via which='...'")

        fwd, title_fmt = _resolve_transform(transform)

        if ax is not None:
            axes = [ax]
            fig = ax.figure
        elif len(names) == 1:
            # Single-parameter call: route through ``resolve_ax`` so an
            # active :func:`hea.plot.par` context pulls a cell from the
            # grid (R's ``par(mfrow=...)`` ergonomics). Outside ``par``,
            # this still allocates a fresh figure.
            from ..plot._util import resolve_ax
            ax_single = resolve_ax(None, figsize=figsize)
            axes = [ax_single]
            fig = ax_single.figure
        else:
            n = len(names)
            fig, axes_obj = plt.subplots(
                1, n, figsize=figsize or (3.2 * n, 3.0), sharey=False,
            )
            axes = list(axes_obj)

        for ax_i, name in zip(axes, names):
            df = self.data[name]
            v = df[name].to_numpy()
            s = df["zeta"].to_numpy()
            x = fwd(v)
            y = np.abs(s) if absolute else s
            ax_i.plot(x, y, "o-", ms=3, lw=1)
            if not absolute:
                ax_i.axhline(0, color="k", lw=0.4)
            lo_fb = 0.0 if name.startswith(".sig") else float("nan")
            for lvl in levels:
                z = float(norm.ppf(0.5 + lvl / 2))
                for tgt in (-z, z):
                    fb = lo_fb if tgt < 0 else float("nan")
                    v_at = _invert_zeta(v, s, tgt, fallback=fb)
                    if np.isfinite(v_at):
                        x_at = fwd(np.asarray(v_at)).item()
                        if np.isfinite(x_at):
                            ax_i.axvline(x_at, color="gray", alpha=0.4, lw=0.5)
            ax_i.set_title(title_fmt.format(name))
            ax_i.set_xlabel(name)
        if ax is None:
            axes[0].set_ylabel("|ζ|" if absolute else "ζ")
            fig.tight_layout()
        return fig

    def plot_density(
        self, npts: int = 201, upper: float = 0.999,
        figsize: tuple[float, float] | None = None,
    ):
        """Profile-implied density plot — Pythonic ``densityplot(profile(...))``.

        For each parameter, plots φ(ζ(v))·|dζ/dv| against v: the Jacobian
        transform of N(0,1) in ζ to a density on the parameter scale.
        ζ(v) is interpolated with a PCHIP spline (monotone-preserving) and
        differentiated analytically. The x-range is restricted to ζ within
        ±Φ⁻¹(``upper``); for variance-component SDs the lower bound is
        clipped to 0.
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import PchipInterpolator
        from scipy.stats import norm

        names = list(self.data.keys())
        n = len(names)
        fig, axes = plt.subplots(
            1, n, figsize=figsize or (3.2 * n, 3.0), sharey=False,
        )
        if n == 1:
            axes = [axes]

        z_max = float(norm.ppf(upper))
        for ax, name in zip(axes, names):
            df = self.data[name]
            v = df[name].to_numpy()
            s = df["zeta"].to_numpy()
            order = np.argsort(v)
            v_s, s_s = v[order], s[order]
            spl = PchipInterpolator(v_s, s_s, extrapolate=True)
            lo_fb = 0.0 if name.startswith(".sig") else float("nan")
            v_lo = _invert_zeta(v, s, -z_max, fallback=lo_fb)
            v_hi = _invert_zeta(v, s, +z_max)
            if not np.isfinite(v_lo):
                v_lo = float(v_s[0])
            if not np.isfinite(v_hi):
                v_hi = float(v_s[-1])
            grid = np.linspace(v_lo, v_hi, npts)
            zeta_g = spl(grid)
            dz_dv = spl.derivative()(grid)
            density = norm.pdf(zeta_g) * np.abs(dz_dv)
            ax.plot(grid, density, lw=1)
            ax.set_title(name)
            ax.set_xlabel(name)
        axes[0].set_ylabel("density")
        fig.tight_layout()
        return fig

    def plot_pairs(
        self, *,
        which: list[str] | None = None,
        transform: str | None = None,
        levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95, 0.99),
        figsize: tuple[float, float] | None = None,
    ):
        """Profile pairs plot — port of lme4's ``splom(profile(...))`` (Fig 2.6).

        Lower triangle: bivariate ζ-deviance contours and the two profile
        traces in *ζ-coordinates* ``(ζⱼ, ζᵢ)``, axes clamped to ±max(level).
        Upper triangle: same contours/traces mapped through each
        parameter's backward spline ``v(ζ)`` into *original* parameter
        space ``(vⱼ, vᵢ)``. Diagonal: parameter labels.

        Pass ``transform="log"`` to reproduce Bates Fig 2.7 — the
        equivalent of R's ``splom(log(profile(fm)))``. ζ is invariant
        under monotone reparameterization, so only the upper-triangle
        v-space panels and the diagonal/axis labels change; log is
        applied to variance-component SDs (``.sig*``, ``.sigma``) only,
        leaving fixed-effect parameters on their natural scale.

        The contour at confidence level α is built (Bates, lme4 § 1.5)
        from four anchor points where the level-α curve crosses the
        profile traces. A periodic cubic spline through ``(θ_mean,
        θ_diff)`` gives an angular parameterization; the curve closes
        smoothly via ``(ζᵢ, ζⱼ) = lev · (cos(θ_mean − θ_diff/2),
        cos(θ_mean + θ_diff/2))``. Contour levels default to the lme4
        defaults: √χ²₂(α) for α ∈ {0.50, 0.80, 0.90, 0.95, 0.99}.
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import CubicSpline, PchipInterpolator
        from scipy.stats import chi2

        if which is None:
            names = list(self.data.keys())
        else:
            names = list(which)
            unknown = [n for n in names if n not in self.data]
            if unknown:
                raise KeyError(
                    f"unknown parameter(s) {unknown!r}; available: {list(self.data)!r}"
                )
        n = len(names)
        if n < 2:
            raise ValueError("plot_pairs needs at least 2 parameters")

        zeta_levels = np.sqrt(chi2.ppf(np.asarray(levels), 2))
        mlev = float(zeta_levels.max())

        # Per-parameter v-transform. Matches R's log.thpr / logProf:
        # log applies to .sig* and .sigma only; fixed effects keep
        # natural scale.
        if transform is None:
            tx_fn: dict[str, "Callable[[np.ndarray], np.ndarray]"] = {
                name: (lambda x: np.asarray(x)) for name in names
            }
            tx_label = {name: name for name in names}
        elif transform == "log":
            tx_fn = {
                name: (np.log if name.startswith(".sig") else (lambda x: np.asarray(x)))
                for name in names
            }
            tx_label = {
                name: (f"log({name})" if name.startswith(".sig") else name)
                for name in names
            }
        else:
            raise ValueError(
                f"unknown transform {transform!r}; use 'log' or None"
            )

        fwd: dict[str, PchipInterpolator] = {}
        bwd: dict[str, PchipInterpolator] = {}
        v_lim: dict[str, tuple[float, float]] = {}
        for name in names:
            df = self.data[name]
            v = df[name].to_numpy()
            s = df["zeta"].to_numpy()
            order = np.argsort(v)
            v_s, s_s = v[order], s[order]
            fwd[name] = PchipInterpolator(v_s, s_s, extrapolate=False)
            order_z = np.argsort(s_s)
            v_t = tx_fn[name](v_s)
            bwd[name] = PchipInterpolator(s_s[order_z], v_t[order_z], extrapolate=False)
            # v-axis limits — match R splom.thpr: backward-spline at ±mlev,
            # then clip to the profile grid range so we never advertise an
            # axis range we don't actually have data for.
            v_lo = bwd[name](-mlev)
            v_hi = bwd[name](+mlev)
            v_t_min, v_t_max = float(v_t.min()), float(v_t.max())
            v_lo = v_t_min if not np.isfinite(v_lo) else float(max(v_lo, v_t_min))
            v_hi = v_t_max if not np.isfinite(v_hi) else float(min(v_hi, v_t_max))
            v_lim[name] = (v_lo, v_hi)

        def _trace_zeta(prof_name: str, other_name: str) -> tuple[np.ndarray, np.ndarray]:
            """Return (ζ_prof, ζ_other) along the trace of profile(prof_name).

            ζ_prof is read directly from the ``zeta`` column; ζ_other is
            obtained by sending the optimum v_other through the forward
            spline of ``other_name`` and dropping NaNs (off-grid points).
            """
            df = self.data[prof_name]
            zp = df["zeta"].to_numpy()
            zo = fwd[other_name](df[other_name].to_numpy())
            keep = ~np.isnan(zo)
            return zp[keep], zo[keep]

        def _sacos(x):
            return np.arccos(np.clip(x, -0.999, 0.999))

        def _ad(xc, yc):
            a = (xc + yc) / 2.0
            d = xc - yc
            return np.sign(d) * a, np.abs(d)

        def _contour_pts(sij, sji, level: float, nseg: int = 101):
            """Generate one bivariate-ζ contour at radius ``level``.

            Returns (n+1, 2) array of (ζ_i, ζ_j) points on the closed curve;
            ``None`` if any anchor falls outside the trace splines' domain.
            """
            try:
                yc1 = _sacos(float(sij(+level)) / level)
                xc2 = _sacos(float(sji(+level)) / level)
                yc3 = _sacos(float(sij(-level)) / level)
                xc4 = _sacos(float(sji(-level)) / level)
            except Exception:
                return None
            if any(np.isnan(v) for v in (yc1, xc2, yc3, xc4)):
                return None
            xs = np.empty(4)
            ys = np.empty(4)
            xs[0], ys[0] = _ad(0.0, yc1)
            xs[1], ys[1] = _ad(xc2, 0.0)
            xs[2], ys[2] = _ad(np.pi, yc3)
            xs[3], ys[3] = _ad(xc4, np.pi)
            order = np.argsort(xs)
            xs_s = xs[order]
            ys_s = ys[order]
            # Close the ring for ``bc_type='periodic'``: append the first
            # knot shifted by one period, with the same y value, so that
            # ``y[0] == y[-1]`` (CubicSpline's periodic precondition).
            xs_p = np.concatenate([xs_s, [xs_s[0] + 2 * np.pi]])
            ys_p = np.concatenate([ys_s, [ys_s[0]]])
            try:
                spl = CubicSpline(xs_p, ys_p, bc_type="periodic")
            except ValueError:
                return None
            theta = np.linspace(xs_s[0], xs_s[0] + 2 * np.pi, nseg + 1)
            tdiff = spl(theta)
            # tauij in lme4:::cont returns (col1, col2) where col1 = lev *
            # cos(θ_mean - θ_diff/2) = ζ_j and col2 = lev * cos(θ_mean +
            # θ_diff/2) = ζ_i. Verify at anchor 1 (θ_m = -θ/2, θ_d = θ):
            # col1 = lev·cos(-θ) = sij(+lev) (the j-coord), col2 = lev =
            # zeta_i at +lev. Stack as (ζ_i, ζ_j) to match downstream.
            zj = level * np.cos(theta - tdiff / 2.0)
            zi = level * np.cos(theta + tdiff / 2.0)
            return np.column_stack([zi, zj])

        # Pre-compute contour data for each (i, j) pair, i < j.
        contours: dict[tuple[int, int], dict] = {}
        for jj in range(1, n):
            for ii in range(jj):
                ni, nj = names[ii], names[jj]
                zi_i, zj_i = _trace_zeta(ni, nj)   # along trace of i
                zj_j, zi_j = _trace_zeta(nj, ni)   # along trace of j
                if len(zi_i) < 4 or len(zj_j) < 4:
                    contours[(ii, jj)] = {}
                    continue
                o_i = np.argsort(zi_i)
                o_j = np.argsort(zj_j)
                # Trace splines extrapolate, matching R's interpSpline + predy
                # in lme4:::cont — splom always renders all length(levels)
                # contours, even when one parameter's profile range stops
                # short of mlev = √χ²₂(0.99) (e.g. an Intercept that's
                # orthogonal to the variance components).
                sij = PchipInterpolator(zi_i[o_i], zj_i[o_i], extrapolate=True)
                sji = PchipInterpolator(zj_j[o_j], zi_j[o_j], extrapolate=True)
                pts_per_level = []
                for lev in zeta_levels:
                    pts = _contour_pts(sij, sji, float(lev))
                    pts_per_level.append(pts)
                contours[(ii, jj)] = dict(
                    sij=sij, sji=sji,
                    trace_i=(zi_i[o_i], zj_i[o_i]),
                    trace_j=(zi_j[o_j], zj_j[o_j]),
                    pts=pts_per_level,
                )

        fig, axes = plt.subplots(
            n, n, figsize=figsize or (2.4 * n, 2.4 * n), squeeze=False,
        )

        def _draw_zeta_panel(ax, info, x_is_i: bool):
            """ζ-space panel. ``x_is_i`` controls which axis is ζ_i."""
            zi_grid_i, zj_at_i = info["trace_i"]
            zi_at_j, zj_grid_j = info["trace_j"]
            if x_is_i:
                ax.plot(zi_grid_i, zj_at_i, "-", lw=0.5, color="black")
                ax.plot(zi_at_j, zj_grid_j, "-", lw=0.5, color="black")
            else:
                ax.plot(zj_at_i, zi_grid_i, "-", lw=0.5, color="black")
                ax.plot(zj_grid_j, zi_at_j, "-", lw=0.5, color="black")
            for pts in info["pts"]:
                if pts is None:
                    continue
                if x_is_i:
                    ax.plot(pts[:, 0], pts[:, 1], "-", lw=0.5, color="black")
                else:
                    ax.plot(pts[:, 1], pts[:, 0], "-", lw=0.5, color="black")
            ax.set_xlim(-1.05 * mlev, 1.05 * mlev)
            ax.set_ylim(-1.05 * mlev, 1.05 * mlev)

        def _draw_v_panel(ax, info, ni, nj, x_is_i: bool):
            """v-space panel. Maps each ζ-coordinate through its backward
            spline to recover v.  ``x_is_i`` controls which axis is v_i."""
            zi_grid_i, zj_at_i = info["trace_i"]
            zi_at_j, zj_grid_j = info["trace_j"]
            vi_i = bwd[ni](zi_grid_i)
            vj_i = bwd[nj](zj_at_i)
            vi_j = bwd[ni](zi_at_j)
            vj_j = bwd[nj](zj_grid_j)
            if x_is_i:
                ax.plot(vi_i, vj_i, "-", lw=0.5, color="black")
                ax.plot(vi_j, vj_j, "-", lw=0.5, color="black")
            else:
                ax.plot(vj_i, vi_i, "-", lw=0.5, color="black")
                ax.plot(vj_j, vi_j, "-", lw=0.5, color="black")
            for pts in info["pts"]:
                if pts is None:
                    continue
                vc_i = bwd[ni](pts[:, 0])
                vc_j = bwd[nj](pts[:, 1])
                ok = ~(np.isnan(vc_i) | np.isnan(vc_j))
                if not ok.any():
                    continue
                if x_is_i:
                    ax.plot(vc_i[ok], vc_j[ok], "-", lw=0.5, color="black")
                else:
                    ax.plot(vc_j[ok], vc_i[ok], "-", lw=0.5, color="black")
            ax.set_xlim(*(v_lim[ni] if x_is_i else v_lim[nj]))
            ax.set_ylim(*(v_lim[nj] if x_is_i else v_lim[ni]))

        # Lattice-splom layout: origin at lower-left, so the parameter
        # at display row ``r`` (matplotlib top-down) is ``names[n-1-r]``
        # and at display column ``c`` is ``names[c]``. The diagonal runs
        # from bottom-left (.sig01) to top-right ((Intercept)).
        for r in range(n):
            for c in range(n):
                ax = axes[r, c]
                ax.tick_params(labelsize=8)
                ax.grid(True, color="lightgray", lw=0.3)
                vid_row = n - 1 - r
                vid_col = c
                if vid_row == vid_col:
                    ax.text(
                        0.5, 0.5, tx_label[names[vid_row]], ha="center", va="center",
                        transform=ax.transAxes, fontsize=12,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    for s in ("top", "right", "bottom", "left"):
                        ax.spines[s].set_visible(True)
                    continue
                ii = min(vid_row, vid_col)
                jj = max(vid_row, vid_col)
                info = contours.get((ii, jj), {})
                if not info:
                    continue
                ni, nj = names[ii], names[jj]
                x_is_i = (vid_col == ii)
                # Lower triangle in display (closer to bottom-left,
                # vid_row < vid_col): ζ-space, per lme4 splom.
                # Upper triangle in display: v-space.
                if vid_row < vid_col:
                    _draw_zeta_panel(ax, info, x_is_i=x_is_i)
                else:
                    _draw_v_panel(ax, info, ni, nj, x_is_i=x_is_i)
                if c == 0:
                    ax.set_ylabel(tx_label[names[vid_row]])
                if r == n - 1:
                    ax.set_xlabel(tx_label[names[vid_col]])

        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"Profile({list(self.data)})"


class BootMer:
    """Parametric-bootstrap output from :meth:`gmm.bootMer` — the analogue of
    lme4's ``"bootMer"`` / ``boot`` object.

    Attributes
    ----------
    t0 : numpy.ndarray
        ``FUN`` applied to the original fit (the point estimate).
    t : numpy.ndarray
        The ``R × len(t0)`` matrix of bootstrap replicates — ``FUN`` applied to
        each refit. Rows for failed refits are all-``NaN`` (lme4's
        ``factory(..., errval=NA)``).
    t0_names : list[str]
        Names of the ``t0`` entries (the row labels of a ``confint`` table).
    R : int
        Number of bootstrap samples (``nsim``).
    seed : int
        The seed that drove the simulation (echoes R's ``$seed``).
    nfail : int
        How many of the ``R`` refits failed.
    """

    def __init__(self, t0, t, t0_names, R, seed, nfail=0):
        self.t0 = np.asarray(t0, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.t0_names = list(t0_names)
        self.R = int(R)
        self.seed = seed
        self.nfail = int(nfail)

    def __repr__(self) -> str:
        return (f"BootMer(R={self.R}, statistics={self.t0_names!r}, "
                f"nfail={self.nfail})")

    def confint(self, parm=None, level: float = 0.95,
                type: str = "perc") -> pl.DataFrame:
        """Bootstrap CIs — port of ``confint.bootMer`` (bootMer.R:207-229).

        ``type`` ∈ ``{"perc","basic","norm"}`` selects the ``boot::boot.ci``
        method. ``parm`` restricts to a subset of statistics (names or 0-based
        indices); default is all. Returns a polars frame with a ``parameter``
        column and the two percentage-labelled bound columns, exactly like
        R's ``confint(bootMer(...))``.
        """
        names = self.t0_names
        if parm is None:
            idx = list(range(len(names)))
        elif isinstance(parm, (str, int)):
            parm = [parm]
            idx = None
        else:
            idx = None
        if idx is None:
            idx = []
            for p in parm:
                if isinstance(p, str):
                    idx.append(names.index(p))
                else:
                    idx.append(int(p))
        a = (1 - level) / 2
        lo_lbl = f"{100 * a:.1f}%"
        hi_lbl = f"{100 * (1 - a):.1f}%"
        rows, los, his = [], [], []
        for i in idx:
            col = self.t[:, i]
            lo, hi = _boot_ci_one(float(self.t0[i]), col, level, type)
            rows.append(names[i])
            los.append(lo)
            his.append(hi)
        return pl.DataFrame({"parameter": rows, lo_lbl: los, hi_lbl: his})
